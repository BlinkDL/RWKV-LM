########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys, math, gc, importlib
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

try:
    print('RWKV_MY_TESTING', os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ''

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE"])

if 'x070' in os.environ["RWKV_MY_TESTING"]:
    CHUNK_LEN = 16
    assert HEAD_SIZE == 64 # can change 64 to your HEAD_SIZE

    # check https://github.com/BlinkDL/RWKV-CUDA/blob/main/rwkv7_fast_fused/rwkv7_cuda_benchmark.py
    #
    # use rwkv7_clampw_v3.cpp and rwkv7_clampw_v3_for_h100.cu for 20% faster fwd & bwd kernel on H100s

    flags = ['-res-usage', f'-D_N_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    load(name="rwkv7_clampw", sources=[f'cuda/rwkv7_clampw.cu', 'cuda/rwkv7_clampw.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
    class RWKV7_CLAMPW_CUDA_OP(torch.autograd.Function):
        @staticmethod
        def forward(ctx,r,w,k,v,a,b):
            B,T,H,N = r.shape 
            assert T%CHUNK_LEN == 0 # if T%CHUNK_LEN != 0: pad your input to T%CHUNK_LEN == 0, or change CHUNK_LEN (will be slower)
            assert all(i.dtype==torch.bfloat16 for i in [r,w,k,v,a,b])
            assert all(i.is_contiguous() for i in [r,w,k,v,a,b])
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,N,N, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,N, dtype=torch.float32,device=w.device)
            torch.ops.rwkv7_clampw.forward(r,w,k,v,a,b,y,s,sa)
            ctx.save_for_backward(r,w,k,v,a,b,s,sa)
            return y
        @staticmethod
        def backward(ctx,dy):
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            r,w,k,v,a,b,s,sa = ctx.saved_tensors
            dr,dw,dk,dv,da,db = [torch.empty_like(x) for x in [r,w,k,v,a,b]]
            torch.ops.rwkv7_clampw.backward(r,w,k,v,a,b,dy,s,sa,dr,dw,dk,dv,da,db)
            return dr,dw,dk,dv,da,db
    def RWKV7_CLAMPW_CUDA(r,w,k,v,a,b):
        B,T,HN = r.shape
        r,w,k,v,a,b = [i.view(B,T,HN//64,64) for i in [r,w,k,v,a,b]] # can change 64 to your HEAD_SIZE. have to hard-code the number here, or pytorch will complain
        return RWKV7_CLAMPW_CUDA_OP.apply(r,w,k,v,a,b).view(B,T,HN)

########################################################################################################

load(name="rwkv7_cmix_bf16_v5", sources=["cuda/rwkv7_cmix_bf16_v5.cpp","cuda/rwkv7_cmix_bf16_v5.cu"], extra_cflags=["-O3"],
     extra_cuda_cflags=['-res-usage', "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"],
     is_python_module=False, verbose=True)

class _CmixLayerV2Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_k, key_weight, value_weight):
        out, mixed, act = torch.ops.rwkv7_cmix_bf16_v5.forward(
            x.contiguous(),
            x_k.contiguous(),
            key_weight.contiguous(),
            value_weight.contiguous(),
        )
        ctx.save_for_backward(x, x_k, key_weight, value_weight, mixed, act)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, x_k, key_weight, value_weight, mixed, act = ctx.saved_tensors
        grad_x, grad_x_k, grad_key_weight, grad_value_weight = torch.ops.rwkv7_cmix_bf16_v5.backward(
            grad_out.contiguous(),
            x,
            x_k,
            key_weight,
            value_weight,
            mixed,
            act,
        )
        return grad_x, grad_x_k, grad_key_weight, grad_value_weight

########################################################################################################

load(name="rwkv7_tmix_mix6_bf16_v5", sources=["cuda/rwkv7_tmix_mix6_bf16_v5.cpp","cuda/rwkv7_tmix_mix6_bf16_v5.cu"], extra_cflags=["-O3"],
     extra_cuda_cflags=['-res-usage', "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"],
     is_python_module=False, verbose=True)

from typing import Tuple

def _setup_context(ctx, inputs, output):
    del output
    ctx.save_for_backward(*inputs)

def _backward(ctx, grads):
    return tuple(torch.ops.rwkv7_tmix_mix6_bf16_v5.backward(
        grads[0].contiguous(),
        grads[1].contiguous(),
        grads[2].contiguous(),
        grads[3].contiguous(),
        grads[4].contiguous(),
        grads[5].contiguous(),
        *ctx.saved_tensors,
    ))

torch.library.register_autograd(
    "rwkv7_tmix_mix6_bf16_v5::forward",
    _backward,
    setup_context=_setup_context,
)

def _forward_op(x, x_r, x_w, x_k, x_v, x_a, x_g):
    return torch.ops.rwkv7_tmix_mix6_bf16_v5.forward(
        x.contiguous(),
        x_r.contiguous(),
        x_w.contiguous(),
        x_k.contiguous(),
        x_v.contiguous(),
        x_a.contiguous(),
        x_g.contiguous(),
    )

@torch.jit.script
def _tmix_mix6_bf16_v5_jit(
    x: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outs = torch.ops.rwkv7_tmix_mix6_bf16_v5.forward(x.contiguous(), x_r.contiguous(), x_w.contiguous(), x_k.contiguous(), x_v.contiguous(), x_a.contiguous(), x_g.contiguous())
    return outs[0], outs[1], outs[2], outs[3], outs[4], outs[5]

if os.environ.get("RWKV_JIT_ON") == "1":
    def tmix_mix6_bf16_v5(x, x_r, x_w, x_k, x_v, x_a, x_g):
        return _tmix_mix6_bf16_v5_jit(x, x_r, x_w, x_k, x_v, x_a, x_g)
else:
    def tmix_mix6_bf16_v5(x, x_r, x_w, x_k, x_v, x_a, x_g):
        return tuple(_forward_op(x, x_r, x_w, x_k, x_v, x_a, x_g))

########################################################################################################

load(name="rwkv7_tmix_kk_pre_bf16_v5", sources=["cuda/rwkv7_tmix_kk_pre_bf16_v5.cpp","cuda/rwkv7_tmix_kk_pre_bf16_v5.cu"], extra_cflags=["-O3"],
     extra_cuda_cflags=['-res-usage', "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"],
     is_python_module=False, verbose=True)

assert HEAD_SIZE == 64

def _setup_context(ctx, inputs, output):
     k, k_k, a, k_a, _head_size = inputs
     inv_d = output[3]
     ctx.save_for_backward(k, k_k, a, k_a, inv_d)

def _backward(ctx, grads):
    k, k_k, a, k_a, inv_d = ctx.saved_tensors
    grad_new_k = grads[0].contiguous()
    grad_neg_kk = grads[1].contiguous()
    grad_kka = grads[2].contiguous()

    return tuple(torch.ops.rwkv7_tmix_kk_pre_bf16_v5.backward(
        grad_new_k,
        grad_neg_kk,
        grad_kka,
        k,
        k_k,
        a,
        k_a,
        inv_d,
        64,
    )) + (None,)

torch.library.register_autograd(
    "rwkv7_tmix_kk_pre_bf16_v5::forward",
    _backward,
    setup_context=_setup_context,
)

def _forward_op(k, k_k, a, k_a):
    outs = torch.ops.rwkv7_tmix_kk_pre_bf16_v5.forward(
        k.contiguous(),
        k_k.contiguous(),
        a.contiguous(),
        k_a.contiguous(),
        64,
    )
    return outs[0], outs[1], outs[2]

@torch.jit.script
def _tmix_kk_pre_bf16_v5_jit(
    k: torch.Tensor,
    k_k: torch.Tensor,
    a: torch.Tensor,
    k_a: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outs = torch.ops.rwkv7_tmix_kk_pre_bf16_v5.forward(
        k.contiguous(),
        k_k.contiguous(),
        a.contiguous(),
        k_a.contiguous(),
        64,
    )
    return outs[0], outs[1], outs[2]

if os.environ.get("RWKV_JIT_ON") == "1":
    def tmix_kk_pre_bf16_v5(k, k_k, a, k_a):
        return _tmix_kk_pre_bf16_v5_jit(k, k_k, a, k_a)
else:
    def tmix_kk_pre_bf16_v5(k, k_k, a, k_a):
        return tuple(_forward_op(k, k_k, a, k_a))

########################################################################################################

load(name="rwkv7_tmix_lnx_rkvres_xg_bf16_v1", sources=["cuda/rwkv7_tmix_lnx_rkvres_xg_bf16_v1.cpp","cuda/rwkv7_tmix_lnx_rkvres_xg_bf16_v1.cu"], extra_cflags=["-O3"],
     extra_cuda_cflags=['-res-usage', "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"],
     is_python_module=False, verbose=True)

def _setup_context(ctx, inputs, output):
    x, r, k, v, r_k, weight, bias, g = inputs
    mean = output[1]
    rstd = output[2]
    ctx.save_for_backward(x, r, k, v, r_k, weight, bias, g, mean, rstd)

def _backward(ctx, grads):
    x, r, k, v, r_k, weight, bias, g, mean, rstd = ctx.saved_tensors
    grad_xg = grads[0].contiguous()
    return tuple(torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.backward(
        grad_xg,
        x,
        r,
        k,
        v,
        r_k,
        weight,
        bias,
        g,
        mean,
        rstd,
    ))

torch.library.register_autograd(
    "rwkv7_tmix_lnx_rkvres_xg_bf16_v1::forward",
    _backward,
    setup_context=_setup_context,
)

def _forward_op(x, r, k, v, r_k, weight, bias, g):
    outs = torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.forward(
        x.contiguous(),
        r.contiguous(),
        k.contiguous(),
        v.contiguous(),
        r_k.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        g.contiguous(),
    )
    return outs[0]

@torch.jit.script
def _tmix_lnx_rkvres_xg_bf16_v1_jit(
    x: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r_k: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    outs = torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.forward(
        x.contiguous(),
        r.contiguous(),
        k.contiguous(),
        v.contiguous(),
        r_k.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        g.contiguous(),
    )
    return outs[0]

if os.environ.get("RWKV_JIT_ON") == "1":
    def tmix_lnx_rkvres_xg_bf16_v1(x, r, k, v, r_k, weight, bias, g):
        return _tmix_lnx_rkvres_xg_bf16_v1_jit(x, r, k, v, r_k, weight, bias, g)
else:
    def tmix_lnx_rkvres_xg_bf16_v1(x, r, k, v, r_k, weight, bias, g):
        return _forward_op(x, r, k, v, r_k, weight, bias, g)

########################################################################################################

load(name="rwkv7_tmix_a_gate_bf16", sources=["cuda/rwkv7_tmix_a_gate_bf16.cpp","cuda/rwkv7_tmix_a_gate_bf16.cu"], extra_cflags=["-O3"],
     extra_cuda_cflags=['-res-usage', "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"],
     is_python_module=False, verbose=True)

def _setup_context(ctx, inputs, output):
    del output
    a0, a12 = inputs
    ctx.save_for_backward(a0, a12)

def _backward(ctx, grad_out):
    a0, a12 = ctx.saved_tensors
    return tuple(torch.ops.rwkv7_tmix_a_gate_bf16.backward(
        grad_out.contiguous(),
        a0,
        a12,
    ))

torch.library.register_autograd(
    "rwkv7_tmix_a_gate_bf16::forward",
    _backward,
    setup_context=_setup_context,
)

def _forward_op(a0, a12):
    return torch.ops.rwkv7_tmix_a_gate_bf16.forward(
        a0.contiguous(),
        a12.contiguous(),
    )

@torch.jit.script
def _tmix_a_gate_bf16_jit(
    a0: torch.Tensor,
    a12: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.rwkv7_tmix_a_gate_bf16.forward(
        a0.contiguous(),
        a12.contiguous(),
    )

if os.environ.get("RWKV_JIT_ON") == "1":
    def tmix_a_gate_bf16(a0, a12):
        return _tmix_a_gate_bf16_jit(a0, a12)
else:
    def tmix_a_gate_bf16(a0, a12):
        return _forward_op(a0, a12)

########################################################################################################

load(name="rwkv7_tmix_vres_gate_bf16_v1", sources=["cuda/rwkv7_tmix_vres_gate_bf16_v1.cpp","cuda/rwkv7_tmix_vres_gate_bf16_v1.cu"], extra_cflags=["-O3"],
     extra_cuda_cflags=['-res-usage', "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"],
     is_python_module=False, verbose=True)

def _setup_context(ctx, inputs, output):
    del output
    v, v_first, v0, v12 = inputs
    ctx.save_for_backward(v, v_first, v0, v12)

def _backward(ctx, grad_out):
    v, v_first, v0, v12 = ctx.saved_tensors
    grad_v, grad_v_first, grad_pre = torch.ops.rwkv7_tmix_vres_gate_bf16_v1.backward(
        grad_out.contiguous(),
        v,
        v_first,
        v0,
        v12,
    )
    grad_v0 = grad_pre.sum(dim=(0, 1), keepdim=True)
    return grad_v, grad_v_first, grad_v0.to(v0.dtype), grad_pre.to(v12.dtype)

torch.library.register_autograd(
    "rwkv7_tmix_vres_gate_bf16_v1::forward",
    _backward,
    setup_context=_setup_context,
)

def _forward_op(v, v_first, v0, v12):
    return torch.ops.rwkv7_tmix_vres_gate_bf16_v1.forward(
        v.contiguous(),
        v_first.contiguous(),
        v0.contiguous(),
        v12.contiguous(),
    )

@torch.jit.script
def _tmix_vres_gate_bf16_v1_jit(
    v: torch.Tensor,
    v_first: torch.Tensor,
    v0: torch.Tensor,
    v12: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.rwkv7_tmix_vres_gate_bf16_v1.forward(
        v.contiguous(),
        v_first.contiguous(),
        v0.contiguous(),
        v12.contiguous(),
    )

if os.environ.get("RWKV_JIT_ON") == "1":
    def tmix_vres_gate_bf16_v1(v, v_first, v0, v12):
        return _tmix_vres_gate_bf16_v1_jit(v, v_first, v0, v12)
else:
    def tmix_vres_gate_bf16_v1(v, v_first, v0, v12):
        return _forward_op(v, v_first, v0, v12)

########################################################################################################

L2WRAP_CE_CUDA_V1 = load(name="rwkv7_l2wrap_ce_bf16_v1", sources=["cuda/rwkv7_l2wrap_ce_bf16_v1.cpp","cuda/rwkv7_l2wrap_ce_bf16_v1.cu"], extra_cflags=["-O3"],
     extra_cuda_cflags=['-res-usage', "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"],
     verbose=True)

class L2WrapCrossEntropyCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        logits = logits.contiguous()
        targets = targets.contiguous()
        loss, lse, max_vals, argmax = L2WRAP_CE_CUDA_V1.forward(logits, targets)
        ctx.save_for_backward(logits, targets.view(-1), lse, max_vals, argmax)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, lse, max_vals, argmax = ctx.saved_tensors
        grad_logits = L2WRAP_CE_CUDA_V1.backward(
            grad_output.contiguous().float(),
            logits,
            targets,
            lse,
            max_vals,
            argmax,
        )
        return grad_logits, None

def l2wrap_cross_entropy(logits, targets):
    return L2WrapCrossEntropyCUDA.apply(logits, targets)

########################################################################################################

class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.my_testing = args.my_testing

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C-1) - 0.5
                zigzag[n] = ((n % N) - ((N-1) / 2)) / ((N-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            D_DECAY_LORA = max(32, int(round(  (2.5*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1,1,C) + 0.5 + zigzag*2.5)

            D_AAA_LORA = max(32, int(round(  (2.5*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C)-0.19 + zigzag*0.3 + linear*0.4)

            D_MV_LORA = max(32, int(round(  (1.7*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_GATE_LORA = max(32, int(round(  (5*(C**0.5))  /32)*32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1,1,C)+0.71 - linear*0.1)
            self.k_a = nn.Parameter(torch.zeros(1,1,C)+1.02)
            self.r_k = nn.Parameter(torch.zeros(H,N)-0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!

            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head

        ############################################################
        # slow pytorch version
        # xx = self.time_shift(x) - x
        # xr = x + xx * self.x_r
        # xw = x + xx * self.x_w
        # xk = x + xx * self.x_k
        # xv = x + xx * self.x_v
        # xa = x + xx * self.x_a
        # xg = x + xx * self.x_g
        ############################################################
        # much faster CUDA version
        xr, xw, xk, xv, xa, xg = tmix_mix6_bf16_v5(
            x,
            self.x_r.view(-1),
            self.x_w.view(-1),
            self.x_k.view(-1),
            self.x_v.view(-1),
            self.x_a.view(-1),
            self.x_g.view(-1),
        )
        ############################################################

        r = self.receptance(xr)
        w = self.w0 + torch.tanh(xw @ self.w1) @ self.w2 # will be soft-clamped to (-inf, -0.5) and exp(-exp(w)) in RWKV7_CLAMPW_CUDA kernel
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            ############################################################
            # slow pytorch version
            # v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
            ############################################################
            # much faster CUDA version
            v12 = (xv @ self.v1) @ self.v2
            v = tmix_vres_gate_bf16_v1(v, v_first, self.v0, v12) # add value residual
            ############################################################

        ############################################################
        # slow pytorch version
        # a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        ############################################################
        # much faster CUDA version
        a = tmix_a_gate_bf16(self.a0, (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        ############################################################

        g = torch.sigmoid(xg @ self.g1) @ self.g2

        ############################################################
        # slow pytorch version
        # kk = k * self.k_k
        # kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        # k = k * (1 + (a-1) * self.k_a)
        # x = RWKV7_CLAMPW_CUDA(r, w, k, v, -kk, kk*a)
        ############################################################
        # much faster CUDA version (!!! fixed eps=1e-12 same as pytorch !!!)
        k, neg_kk, kka = tmix_kk_pre_bf16_v5(
            k,
            self.k_k.view(-1),
            a,
            self.k_a.view(-1),
        )
        x = RWKV7_CLAMPW_CUDA(r, w, k, v, neg_kk, kka)
        ############################################################

        ############################################################
        # slow pytorch version
        # x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        # x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        # x = self.output(x * g)
        ############################################################
        # much faster CUDA version (!!! fixed eps=64e-5 and H=64, also fused x*g !!!)
        x = tmix_lnx_rkvres_xg_bf16_v1(
                x,
                r,
                k,
                v,
                self.r_k,
                self.ln_x.weight,
                self.ln_x.bias,
                g,
        )
        x = self.output(x)
        ############################################################

        return x, v_first
    
########################################################################################################

# class RWKV_CMix_x070(MyModule): # slow pytorch version
#     def __init__(self, args, layer_id):
#         super().__init__()
#         self.args = args
#         self.layer_id = layer_id
#         self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

#         with torch.no_grad():
#             ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
#             ddd = torch.ones(1, 1, args.n_embd)
#             for i in range(args.n_embd):
#                 ddd[0, 0, i] = i / args.n_embd
#             self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

#         self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
#         self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

#         self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
#         self.value.weight.data.zero_()

#     @MyFunction
#     def forward(self, x):
#         xx = self.time_shift(x) - x
        
#         k = x + xx * self.x_k
#         k = torch.relu(self.key(k)) ** 2

#         return self.value(k)

class RWKV_CMix_x070(nn.Module): # fast CUDA version
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        self.value.weight.data.zero_()

    def forward(self, x):
        return _CmixLayerV2Fn.apply(x, self.x_k.view(-1), self.key.weight, self.value.weight)

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x))
        return x, v_first


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, grad_output * gy) # original (grad_output, gy) is buggy when grad_output != 1 !!!


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'):
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size            
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def configure_optimizers(self):
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        for n, p in self.named_parameters():
            if ("att.w0" in n):
                lr_2x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0) and (".weight" in n):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))

        if self.trainer.is_global_zero:
            print('decay', lr_decay, '\n')
            print('1x', lr_1x, '\n')
            print('2x', lr_2x, '\n')

        param_dict = {n: p for n, p in self.named_parameters()}
        
        optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
            {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
        ]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, idx):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            if args.grad_cp == 1:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
            else:
                x, v_first = block(x, v_first)

        x = self.ln_out(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits = self(idx)
      
        ############################################################
        # slow pytorch version (!!! SLOW AND TAKES 40% MORE VRAM !!!)
        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        # return L2Wrap.apply(loss, logits)
        ############################################################
        # much faster CUDA version (!!! fixed 1e-4 factor !!!)
        return l2wrap_cross_entropy(logits, targets)

    def training_step_end(self, batch_parts):
        all = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all

    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        n_params = 0
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}", end="")

            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n or n.endswith('_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias') or (".weight" not in n):
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
                print()
            elif n == "emb.weight":
                m[n] = p
                scale = -1e-4
                nn.init.uniform_(m[n], a=scale, b=-scale)
                print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p
                if self.args.vocab_size > self.args.n_embd:
                    scale = 0.5 * math.sqrt(self.args.vocab_size / self.args.n_embd)
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                print(f" [scale {scale}]")
            else:
                assert n.endswith('.weight') # should always be true

                zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']

                for kk in zero:
                    if kk in n:
                        scale = 0

                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                print(f" [scale {scale}]")

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()
            n_params += m[n].numel()

        print('model params', n_params)
        gc.collect()
        torch.cuda.empty_cache()
        return m
