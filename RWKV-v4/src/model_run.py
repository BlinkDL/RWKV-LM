########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types
import copy
import torch
import math, os
from torch.nn import functional as F
import torch.nn as nn

RWKV_HEAD_QK_DIM = 0
print(f'\nRWKV_HEAD_QK_DIM {RWKV_HEAD_QK_DIM}\n')

DEBUG_TIME = False   # True False - show trained time-coeffs

########################################################################################################
# CUDA Kernel
########################################################################################################

if os.environ['RWKV_RUN_DEVICE'] == 'cuda':
    T_MAX = 4096 # increase this if your ctx_len is long
    # it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

    from torch.utils.cpp_extension import load
    wkv_cuda = load(name="wkv", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"],
                    verbose=True, extra_cuda_cflags=['--use_fast_math', '--extra-device-vectorization', f'-DTmax={T_MAX}'])

    class WKV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v):
            ctx.B = B
            ctx.T = T
            ctx.C = C
            assert T <= T_MAX
            assert B * C % min(C, 1024) == 0
            if os.environ['RWKV_FLOAT_MODE'] != 'fp32':
                w = -torch.exp(w.float().contiguous())
                u = u.float().contiguous()
                k = k.float().contiguous()
                v = v.float().contiguous()
            else:
                w = -torch.exp(w.contiguous())
                u = u.contiguous()
                k = k.contiguous()
                v = v.contiguous()
            ctx.save_for_backward(w, u, k, v)
            y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
            wkv_cuda.forward(B, T, C, w, u, k, v, y)
            if os.environ['RWKV_FLOAT_MODE'] == 'fp16':
                return y.half()
            elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
                return y.bfloat16()
            elif os.environ['RWKV_FLOAT_MODE'] == 'fp32':
                return y

        @staticmethod
        def backward(ctx, gy):
            B = ctx.B
            T = ctx.T
            C = ctx.C
            assert T <= T_MAX
            assert B * C % min(C, 1024) == 0
            w, u, k, v = ctx.saved_tensors
            gw = torch.zeros((B, C), device='cuda')
            gu = torch.zeros((B, C), device='cuda')
            gk = torch.zeros((B, T, C), device='cuda')
            gv = torch.zeros((B, T, C), device='cuda')
            if os.environ['RWKV_FLOAT_MODE'] != 'fp32':
                wkv_cuda.backward(B, T, C, w, u, k, v, gy.float().contiguous(), gw, gu, gk, gv)
            else:
                wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            if os.environ['RWKV_FLOAT_MODE'] == 'fp16':
                return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
            elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
                return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
            elif os.environ['RWKV_FLOAT_MODE'] == 'fp32':
                return (None, None, None, gw, gu, gk, gv)

    def RUN_CUDA(B, T, C, w, u, k, v):
        return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())

############################################################################################################

RWKV_CFG = types.SimpleNamespace()

class RWKV_ChannelMix(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, RWKV_CFG.n_embd))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, RWKV_CFG.n_embd))

        hidden_sz = 4 * RWKV_CFG.n_embd
        self.key = nn.Linear(RWKV_CFG.n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(RWKV_CFG.n_embd, RWKV_CFG.n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, RWKV_CFG.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        
        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv

class RWKV_TimeMix(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_decay = nn.Parameter(torch.ones(RWKV_CFG.n_embd))
        self.time_first = nn.Parameter(torch.ones(RWKV_CFG.n_embd) * math.log(0.3))
        
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        self.time_mix_k = nn.Parameter(torch.ones(1,1,RWKV_CFG.n_embd))
        self.time_mix_v = nn.Parameter(torch.ones(1,1,RWKV_CFG.n_embd))
        self.time_mix_r = nn.Parameter(torch.ones(1,1,RWKV_CFG.n_embd))

        self.key = nn.Linear(RWKV_CFG.n_embd, RWKV_CFG.n_embd, bias=False)
        self.value = nn.Linear(RWKV_CFG.n_embd, RWKV_CFG.n_embd, bias=False)
        self.receptance = nn.Linear(RWKV_CFG.n_embd, RWKV_CFG.n_embd, bias=False)

        self.output = nn.Linear(RWKV_CFG.n_embd, RWKV_CFG.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)

        rwkv = torch.sigmoid(r) * RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v)
        
        rwkv = self.output(rwkv)
        return rwkv

class Block(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(RWKV_CFG.n_embd)
        self.ln2 = nn.LayerNorm(RWKV_CFG.n_embd)
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(RWKV_CFG.n_embd)

        if self.layer_id == 0 and RWKV_CFG.model_type == 'RWKV-ffnPre':
            self.ffnPre = RWKV_ChannelMix(layer_id+1000)
        else:
            self.att = RWKV_TimeMix(layer_id)

        self.ffn = RWKV_ChannelMix(layer_id)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)
        if self.layer_id == 0 and RWKV_CFG.model_type == 'RWKV-ffnPre':
            x = x + self.ffnPre(self.ln1(x))
        else:
            x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class RWKV_GPT(nn.Module):
    def __init__(self, MODEL_NAME, RUN_DEVICE, model_type, vocab_size, n_layer, n_embd, ctx_len):
        global RWKV_CFG
        super().__init__()

        RWKV_CFG.RUN_DEVICE = RUN_DEVICE
        RWKV_CFG.model_type = model_type
        RWKV_CFG.vocab_size = vocab_size
        RWKV_CFG.n_layer = n_layer
        RWKV_CFG.n_embd = n_embd
        RWKV_CFG.ctx_len = ctx_len

        print('\nloading RWKV-GPT', MODEL_NAME)

        self.emb = nn.Embedding(vocab_size, n_embd)

        self.blocks = nn.Sequential(*[Block(i) for i in range(n_layer)])

        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        if RWKV_HEAD_QK_DIM > 0:
            self.head_q = nn.Linear(n_embd, RWKV_HEAD_QK_DIM, bias=False)
            self.head_q.scale_init = 0
            self.head_k = nn.Linear(n_embd, RWKV_HEAD_QK_DIM, bias=False)
            self.head_k.scale_init = 0.1
            self.register_buffer("copy_mask", torch.tril(
                torch.ones(ctx_len, ctx_len)))

        self.ctx_len = ctx_len
        self.eval()
        self.load_state_dict(torch.load(MODEL_NAME + '.pth'))
        self.eval()

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."
        
        x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)

        if RWKV_HEAD_QK_DIM > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (1.0 / RWKV_HEAD_QK_DIM)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)

            c = c @ F.one_hot(idx, num_classes=RWKV_CFG.vocab_size).float()
            x = self.head(x) + c
        else:
            x = self.head(x)        

        return x

############################################################################################################

class RWKV_RNN(): # this is running in FP32 at this moment
    def __init__(self, MODEL_NAME, RUN_DEVICE, model_type, n_layer, n_embd, ctx_len):
        self.RUN_DEVICE = RUN_DEVICE
        self.model_type = model_type
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.ctx_len = ctx_len

        self.w = types.SimpleNamespace()

        w = torch.load(MODEL_NAME + '.pth',
                       map_location=torch.device(RUN_DEVICE))
        for x in w.keys():
            w[x] = w[x].float()
            if '.time_' in x:
                w[x] = w[x].squeeze()
            if '.time_decay' in x:
                w[x] = -torch.exp(w[x])
            if DEBUG_TIME and '.time_' in x:
                print(x, w[x].squeeze().cpu().numpy())

            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        self.clear()

    def clear(self):
        self.xx = {}
        self.aa = {}
        self.bb = {}
        self.pp = {}
        self.hk = None

    def save(self, target):
        target.xx = copy.deepcopy(self.xx)
        target.aa = copy.deepcopy(self.aa)
        target.bb = copy.deepcopy(self.bb)
        target.pp = copy.deepcopy(self.pp)
        target.hk = copy.deepcopy(self.hk)

    def load(self, target):
        self.xx = copy.deepcopy(target.xx)
        self.aa = copy.deepcopy(target.aa)
        self.bb = copy.deepcopy(target.bb)
        self.pp = copy.deepcopy(target.pp)
        self.hk = copy.deepcopy(target.hk)

    def LN(self, xx, w):
        return F.layer_norm(xx, (self.n_embd,), weight=w.weight, bias=w.bias)

    def FF(self, xx, w, name):
        if name not in self.xx:
            self.xx[name] = torch.zeros(self.n_embd, device=self.RUN_DEVICE)
        xk = xx * w.time_mix_k + self.xx[name] * (1 - w.time_mix_k)
        xr = xx * w.time_mix_r + self.xx[name] * (1 - w.time_mix_r)
        self.xx[name] = xx

        r = torch.sigmoid(w.receptance.weight @ xr)
        k = torch.square(torch.relu(w.key.weight @ xk))
        kv = w.value.weight @ k

        return r * kv

    def SA(self, xx, w, name):
        if name not in self.xx:
            self.xx[name] = torch.zeros(self.n_embd, device=self.RUN_DEVICE)
            self.aa[name] = torch.zeros(self.n_embd, device=self.RUN_DEVICE)
            self.bb[name] = torch.zeros(self.n_embd, device=self.RUN_DEVICE)
            self.pp[name] = torch.zeros(self.n_embd, device=self.RUN_DEVICE) - 1e30

        xk = xx * w.time_mix_k + self.xx[name] * (1 - w.time_mix_k)
        xv = xx * w.time_mix_v + self.xx[name] * (1 - w.time_mix_v)
        xr = xx * w.time_mix_r + self.xx[name] * (1 - w.time_mix_r)
        self.xx[name] = xx

        r = torch.sigmoid(w.receptance.weight @ xr)

        k = w.key.weight @ xk
        v = w.value.weight @ xv

        pp = self.pp[name]
        aa = self.aa[name]
        bb = self.bb[name]
        ww = w.time_first + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        ww = pp + w.time_decay
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)
        self.aa[name] = e1 * aa + e2 * v
        self.bb[name] = e1 * bb + e2
        self.pp[name] = p

        rwkv = r * a / b

        return w.output.weight @ rwkv

    def run(self, ctx):
        w = self.w
        x = w.emb.weight[ctx[-1]]

        for i in range(self.n_layer):
            if i == 0:
                x = self.LN(x, w.blocks[i].ln0)
            if i == 0 and self.model_type == 'RWKV-ffnPre':
                x = x + self.FF(self.LN(x, w.blocks[i].ln1), w.blocks[i].ffnPre, f'ffnPre.{i}')
            else:
                x = x + self.SA(self.LN(x, w.blocks[i].ln1), w.blocks[i].att, f'att.{i}')
            x = x + self.FF(self.LN(x, w.blocks[i].ln2), w.blocks[i].ffn, f'ffn.{i}')

        x = self.LN(x, w.ln_out)

        if RWKV_HEAD_QK_DIM > 0:
            if self.hk == None:
                self.hk = (w.head_k.weight @ x).unsqueeze(0)
            else:
                self.hk = torch.cat(
                    [self.hk, (w.head_k.weight @ x).unsqueeze(0)], dim=0)
            if self.hk.shape[0] > self.ctx_len:
                self.hk = self.hk[-self.ctx_len:, :]

            q = w.head_q.weight @ x

            x = w.head.weight @ x
            x = x.cpu().numpy().tolist()

            c = (self.hk @ q) / RWKV_HEAD_QK_DIM
            for i in range(len(c)):
                x[ctx[i]] += c[i]
        else:
            x = w.head.weight @ x
            x = x.cpu().numpy().tolist()

        return x
