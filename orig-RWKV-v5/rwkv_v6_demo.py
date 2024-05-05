########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import torch, types, os, gc, math
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

'''
This will load RWKV-6 1.6B (L24-D2048) and inference in GPT-mode (slower than RNN-mode for autoregressive generation)

Code output:

Input:
[6699, 304, 25740, 109, 39990, 4600, 4596, 22590, 30449, 4706]

Output:
tensor([[[ -6.8125, -12.8750, -10.7500,  ..., -14.1250, -14.1250, -14.1250],
         [ -4.0625, -11.0625,  -8.3750,  ..., -16.5000, -16.5000, -16.5000],
         [-15.9375, -22.2500, -20.8750,  ..., -31.7500, -31.7500, -31.7500],
         ...,
         [ -6.5000, -16.8750, -14.8125,  ..., -20.7500, -20.7500, -20.7500],
         [ -6.1562, -15.3125, -10.6875,  ..., -29.2500, -29.2500, -29.2500],
         [-11.1250, -21.5000, -19.0000,  ..., -26.2500, -26.2500, -26.2500]]],
       device='cuda:0', dtype=torch.bfloat16)

The Eiffel tower is in the city of
 Paris [probability 94.13%]
 France [probability 0.63%]
 the [probability 0.61%]
 pari [probability 0.46%]
 Se [probability 0.15%]

 [probability 0.14%]
 Par [probability 0.13%]
 Tro [probability 0.13%]
 Tours [probability 0.12%]
 Mont [probability 0.11%]

########################################################################################################

How RWKV-6 works (paper: https://arxiv.org/abs/2404.05892)

RWKV-6 GPT mode (good for training & prefilling): https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v5/rwkv_v6_demo.py

RWKV-6 RNN mode (good for autoregressive generation): https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_v6_demo.py

###############################################################################

The RWKV model:

def forward(self, idx):
    x = self.emb(idx) ######## embedding

    for block in self.blocks:
        x = block(x)

    x = self.ln_out(x) ######## layernorm for output
    x = self.head(x) ######## output projection
    return x

The RWKV block:

def forward(self, x):

    if self.layer_id == 0:
        x = self.ln0(x) ######## extra layernorm after embedding

    x = x + self.att(self.ln1(x)) ######## "att" = RWKV_Tmix_x060
    x = x + self.ffn(self.ln2(x)) ######## "ffn" = RWKV_CMix_x060

    return x

So it's like:

x => emb => block.0.ln0 => +att(block.0.ln1(x)) => +ffn(block.0.ln2(x)) => ... => ln_out => head => logits

###############################################################################

THE RWKV_CMix_x060 BLOCK (replace transformer FFN)

def forward(self, x):
    xx = self.time_shift(x) - x ######## self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
    xk = x + xx * self.time_maa_k
    xr = x + xx * self.time_maa_r

    k = self.key(xk)
    k = torch.relu(k) ** 2
    kv = self.value(k)
    return torch.sigmoid(self.receptance(xr)) * kv

#### Here xx is like "previous token" (timeshift(x)) minus "this token" (x)

#### We mix x with xx using coefficients time_maa_k & time_maa_r to get xk & xr

so xk & xr are like x, but with "some information of previous token" mixed in them

#### We use reluSq and an extra sigmoid(r) gate

###############################################################################

THE RWKV_TMix_x060 BLOCK (replace transformer MHA)

def jit_func(self, x):
    B, T, C = x.size()

    xx = self.time_shift(x) - x
    xxx = x + xx * self.time_maa_x ######## xxx = mix of x & xx

    xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
    xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)

    mw, mk, mv, mr, mg = xxx.unbind(dim=0) ######## xxx => LoRA => mw, mk, mv, mr, mg

    ######## time_maa_* are static mixing coefficients, and m* are dynamic mixing coefficients
    xw = x + xx * (self.time_maa_w + mw)
    xk = x + xx * (self.time_maa_k + mk)
    xv = x + xx * (self.time_maa_v + mv)
    xr = x + xx * (self.time_maa_r + mr)
    xg = x + xx * (self.time_maa_g + mg)

    r = self.receptance(xr) ######## r of RWKV5/6 is similar to transformer q
    k = self.key(xk) ######## k is similar to transformer k
    v = self.value(xv) ######## v is similar to transformer v
    g = F.silu(self.gate(xg)) ######## g is an extra gate

    ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2 ######### xw => LoRA => ww, which is the dynamic part of w
    w = self.time_decay + ww ######### w is the "decay coefficient" for each channel. time_decay is the static part of w

    return r, k, v, g, w

def jit_func_2(self, x, g):
    B, T, C = x.size()
    x = x.view(B * T, C)
    
    x = self.ln_x(x).view(B, T, C) ######### ln_x is GroupNorm = individual LayerNorm for each head
    x = self.output(x * g)
    return x

def forward(self, x):
    B, T, C = x.size()
    H = self.n_head

    r, k, v, g, w = self.jit_func(x)
    x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa) # The RWKV operator

    return self.jit_func_2(x, g)

Explaining the RWKV operator:

#### C is splitted into multiple heads, with head_sz = 64

#### For each head, compute the outer product of k & v, which will be a 64x64 matrix. Let's call it A

#### A will accumulate to build the state S. And S will decay over time (decay speed controlled by w).

S_t = u A_t + A_{t-1} + w_{t-1} A_{t-2} + w_{t-1} w_{t-2} A_{t-3} + ...

#### Multiply r (vector) with S (matrix) to get output

###############################################################################

RWKV can be rewritten as an RNN. Check the code in https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_v6_demo.py

def time_mixing(self, x, state, i:int, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, time_first, time_decay, kw, vw, rw, gw, ow, ln_w, ln_b):
    H = self.n_head
    S = self.head_size

    i1 = (2+S)*i+1
    sx = state[i1] - x
    state[i1] = x
    xxx = x + sx * x_maa
    xxx = torch.tanh(xxx @ tm_w1).view(5, 1, -1)
    xxx = torch.bmm(xxx, tm_w2).view(5, -1)
    mw, mk, mv, mr, mg = xxx.unbind(dim=0)

    xw = x + sx * (w_maa + mw)
    xk = x + sx * (k_maa + mk)
    xv = x + sx * (v_maa + mv)
    xr = x + sx * (r_maa + mr)
    xg = x + sx * (g_maa + mg)

    w = (time_decay + (torch.tanh(xw @ td_w1) @ td_w2).float()).view(H, S, 1)
    w = torch.exp(-torch.exp(w.float())) ######### we are actually using exp(-exo(w)) as decay coefficient, which is always within (0,1)

    r = (rw @ xr).view(H, 1, S)
    k = (kw @ xk).view(H, S, 1)
    v = (vw @ xv).view(H, 1, S)
    g = F.silu(gw @ xg)

    s = state[(2+S)*i+2:(2+S)*(i+1), :].reshape(H, S, S) ######### Because state[] contains states of all blocks, this is fetching the correct state for this block. Note S=64 is head_size

    x = torch.zeros(H, S)
    a = k @ v ######### outer product of k and v (check the shape of k and v)
    x = r @ (time_first * a + s) ######### "time_first" = u
    s = a + w * s

    state[(2+S)*i+2:(2+S)*(i+1), :] = s.reshape(S, -1) ######### Update state
    x = x.flatten()

    x = F.group_norm(x.unsqueeze(0), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).squeeze(0) * g ######### note we are using eps=64e-5 for GroupNorm
    return ow @ x

Let's verify:

s = 0
a = k0@v0
x0 = r0 @ (u a + s) = r0 @ (u k0@v0 + 0)
s = k0@v0
a = k1@v1
x1 = r1 @ (u a + s) = r1 @ (u k1@v1 + k0@v0)
s = k1@v1 + w1 k0@v0
a = k2@v2
x2 = r2 @ (u a + s) = r2 @ (u k2@v2 + k1@v1 + w1 k0@v0)
...

and this agrees with our previous formula:

x_t = r_t @ S_t = r_t @ (u A_t + A_{t-1} + w_{t-1} A_{t-2} + w_{t-1} w_{t-2} A_{t-3} + ...)

###############################################################################
#
# In RWKV v6.0b, we find it's possible to replace GroupNorm by LayerNorm, and remove gate, to save some params and make it faster.
#
# Check https://github.com/BlinkDL/LinearAttentionArena
#
# Finally, if you are training RWKV from scratch, it's VERY IMPORTANT to try my initialization for all parameters.
#
# The self.time_xxx initializations can be seen here.
#
# And we have more initializations in init_params() here, which is actually:
#
# emb.weight => nn.init.uniform_(a=-1e-4, b=1e-4)
# head.weight => nn.init.orthogonal_(gain=0.5*sqrt(n_vocab / n_embd))
#
# att.receptance.weight => nn.init.orthogonal_(gain=1)
# att.key.weight => nn.init.orthogonal_(gain=0.1)
# att.value.weight => nn.init.orthogonal_(gain=1)
# att.gate.weight => nn.init.orthogonal_(gain=0.1)
# att.output.weight => zero
#
# att.ln_x.weight (groupnorm) => ((1 + layer_id) / total_layers) ** 0.7
#
# ffn.key.weight => nn.init.orthogonal_(gain=1)
# ffn.value.weight => zero
# ffn.receptance.weight => zero
#
# !!! If you are using positional embedding, maybe it's better to remove block.0.ln0, and use default initialization for emb.weight instead of my uniform_(a=-1e-4, b=1e-4) !!!
#
########################################################################################################
'''

args = types.SimpleNamespace()

args.n_layer = 24
args.n_embd = 2048

args.vocab_size = 65536
args.ctx_len = 4096

########################################################################################################
# CUDA Kernel
########################################################################################################

args.head_size_a = 64 # don't change
args.head_size_divisor = 8 # don't change

from torch.utils.cpp_extension import load

wkv6_cuda = load(name="wkv6", sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"],
                verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={args.head_size_a}", f"-D_T_={args.ctx_len}"])

class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u): # forward: r, k, v, w, u => y
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
            assert args.head_size_a == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ctx.save_for_backward(r, k, v, w, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.forward(B, T, C, H, r, k, v, w, u, y)
            return y

    @staticmethod
    def backward(ctx, gy): # backward: gy => gr, gk, gv, gw, gu
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, w, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu) # return gradients for r,k,v,w,u

def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)

########################################################################################################
# RWKV TimeMix
########################################################################################################

class RWKV_Tmix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32 # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w

    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, w = self.jit_func(x)
        x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)

        return self.jit_func_2(x, g)
    
########################################################################################################
# RWKV ChannelMix
########################################################################################################

class RWKV_CMix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv

########################################################################################################
# RWKV Block
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

        self.att = RWKV_Tmix_x060(args, layer_id)
        self.ffn = RWKV_CMix_x060(args, layer_id)
        
    def forward(self, x):

        if self.layer_id == 0:
            x = self.ln0(x)

        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x

########################################################################################################
# RWKV Model
########################################################################################################

class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.dim_att = args.n_embd
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)

        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # self.init_params() # !!! When you train RWKV from scratch, try my initialization for best performance !!!

    def forward(self, idx):

        x = self.emb(idx)

        for block in self.blocks:
            x = block(x)

        x = self.ln_out(x)
        x = self.head(x)

        return x
    
    def init_params(self):
        m = self.state_dict()
        n_params = 0

        for n in self.state_dict():
            p = m[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {n}", end="")

            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or n.endswith('_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias'):
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
                print()
            elif n == "emb.weight":
                m[n] = p
                scale = -1e-4
                nn.init.uniform_(m[n], a=scale, b=-scale) # !!! If you are using positional embedding, maybe it's better to remove block.0.ln0, and use default initialization for emb.weight instead of my uniform_(a=-1e-4, b=1e-4) !!!
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

                for kk in [".att.output.", ".ffn.value.", ".ffn.receptance."]:
                    if kk in n:
                        scale = 0
                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                print(f" [scale {scale}]")

                m[n] = torch.empty((shape[0], shape[1]), device=p.device)
                if scale == 0:
                    nn.init.zeros_(m[n])
                else:
                    nn.init.orthogonal_(m[n], gain=scale)

            n_params += m[n].numel()
        
        print('model params', n_params)
        gc.collect()
        torch.cuda.empty_cache()

########################################################################################################
# RWKV Tokenizer (slow version)
########################################################################################################

class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()

tokenizer = RWKV_TOKENIZER("rwkv_vocab_v20230424.txt")

########################################################################################################
# RWKV Inference
########################################################################################################

# use https://huggingface.co/BlinkDL/rwkv-6-world/blob/main/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth

model_params = torch.load("/mnt/e/RWKV-Runner/models/rwkv-final-v6-2.1-1b6.pth", map_location="cpu")

with torch.no_grad():

    model = RWKV(args).bfloat16().cuda()
    model.load_state_dict(model_params)

    prompt = "The Eiffel tower is in the city of"
    input = tokenizer.encode(prompt)
    print(f'\nInput:\n{input}')

    out = model.forward(torch.tensor(input).reshape(1,-1).cuda())
    print(f'\nOutput:\n{out}')

    # let's check the logits for the last token => prediction for the next token
    
    out = out[0,-1] # out.shape = [batch_size(B), seq_len(T), n_emb(C)], so out[0,-1] is the logits for the last token
    
    probs = F.softmax(out.float(), dim=-1) # compute softmax in float (more accurate)

    print(f'\n{prompt}')

    _, indices = torch.topk(probs, 10) # print top-10 possibilities
    for i in range(len(indices)):
        token_id = indices[i].item()
        token = tokenizer.decode([token_id])
        token_prob = probs[token_id].item()
        print(token, f'[probability {token_prob:.2%}]')
