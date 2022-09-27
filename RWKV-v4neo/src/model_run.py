########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types
import copy
import torch
import math, os
from torch.nn import functional as F
import torch.nn as nn

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
# MyModule = torch.jit.ScriptModule
# MyFunction = torch.jit.script_method

RWKV_HEAD_QK_DIM = 0
print(f'\nRWKV_HEAD_QK_DIM {RWKV_HEAD_QK_DIM}\n')

DEBUG_TIME = False   # True False - show trained time-coeffs

############################################################################################################

class RWKV_RNN(MyModule): # this is running in FP32 at this moment
    def __init__(self, MODEL_NAME, RUN_DEVICE, model_type, n_layer, n_embd, ctx_len):
        super().__init__()

        self.RUN_DEVICE = RUN_DEVICE
        self.model_type = model_type
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.ctx_len = ctx_len

        self.w = types.SimpleNamespace()

        w = torch.load(MODEL_NAME + '.pth', map_location=torch.device(RUN_DEVICE))
        for x in w.keys():
            w[x] = w[x].float()
            if '.time_' in x:
                w[x] = w[x].squeeze()
            if '.time_decay' in x:
                w[x] = -torch.exp(w[x])
            if 'pos_emb_x' in x:
                self.w.pos_emb = (w['pos_emb_x'] + w['pos_emb_y']).reshape(ctx_len+1, -1)[:-1,:]
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
        self.eval()

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

    @MyFunction
    def LN(self, xx, w):
        return F.layer_norm(xx, (self.n_embd,), weight=w.weight, bias=w.bias)

    @MyFunction
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

    @MyFunction
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

    def forward(self, ctx, preprocess_only = False):
        with torch.no_grad():
            w = self.w
            x = w.emb.weight[ctx[-1]]
            try:
                pos_emb = w.pos_emb[len(ctx)-1]
                x = x + pos_emb
            except:
                pass

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

                if preprocess_only:
                    return None

                q = w.head_q.weight @ x

                x = w.head.weight @ x
                x = x

                c = (self.hk @ q) / RWKV_HEAD_QK_DIM
                for i in range(len(c)):
                    x[ctx[i]] += c[i]
            else:
                if preprocess_only:
                    return None

                x = w.head.weight @ x
                x = x

            return x
