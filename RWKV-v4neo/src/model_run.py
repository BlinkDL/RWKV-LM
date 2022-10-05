########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types
import torch
import math, os, gc
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

class RWKV_RNN(MyModule):
    def __init__(self, MODEL_NAME, RUN_DEVICE, model_type, n_layer, n_embd, ctx_len):
        super().__init__()

        self.RUN_DEVICE = RUN_DEVICE
        self.model_type = model_type
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.ctx_len = ctx_len

        w = torch.load(MODEL_NAME + '.pth', map_location='cpu')

        # refine weights and send to correct device

        keys = list(w.keys())
        if 'pos_emb_x' in keys:
            w['pos_emb'] = (w['pos_emb_x'] + w['pos_emb_y']).reshape(ctx_len+1, -1)[:-1,:]

        keys = list(w.keys())
        print_need_newline = False
        for x in keys:
            if '.time_' in x:
                w[x] = w[x].squeeze()
                if DEBUG_TIME:
                    print(x, w[x].numpy())
            if '.time_decay' in x:
                w[x] = w[x].float()
                w[x] = -torch.exp(w[x])
            elif '.time_first' in x:
                w[x] = w[x].float()
            else:
                if os.environ["RWKV_FLOAT_MODE"] == "fp32":
                    w[x] = w[x].float()
                elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                    w[x] = w[x].bfloat16()

            w[x].requires_grad = False
            if RUN_DEVICE == 'cuda' and x != 'emb.weight':
                w[x] = w[x].cuda()

            if ('blocks.' not in x) or ('blocks.0.' in x):
                if print_need_newline:
                    print('\n', end = '')
                    print_need_newline = False
                print(x.ljust(40), str(w[x].dtype).replace('torch.', '').ljust(10), w[x].device)
            else:
                print_need_newline = True
                print('.', end = '', flush = True)

        # store weights in self.w

        keys = list(w.keys())
        self.w = types.SimpleNamespace()
        for x in keys:
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

        self.eval()
        gc.collect()
        torch.cuda.empty_cache()

    @MyFunction
    def LN(self, x, w):
        return F.layer_norm(x, (self.n_embd,), weight=w.weight, bias=w.bias)

    # state: ffn_xx att_xx att_aa att_bb att_pp

    @MyFunction
    def FF(self, x, w, state, i):
        if os.environ["RWKV_FLOAT_MODE"] == "bf16":
            xk = x * w.time_mix_k + state[5*i+0].bfloat16() * (1 - w.time_mix_k)
            xr = x * w.time_mix_r + state[5*i+0].bfloat16() * (1 - w.time_mix_r)
            state[5*i+0] = x.float()
        else:
            xk = x * w.time_mix_k + state[5*i+0] * (1 - w.time_mix_k)
            xr = x * w.time_mix_r + state[5*i+0] * (1 - w.time_mix_r)
            state[5*i+0] = x

        r = torch.sigmoid(w.receptance.weight @ xr)
        k = torch.square(torch.relu(w.key.weight @ xk))
        kv = w.value.weight @ k

        return r * kv

    @MyFunction
    def SA(self, x, w, state, i):
        if os.environ["RWKV_FLOAT_MODE"] == "bf16":
            xk = x * w.time_mix_k + state[5*i+1].bfloat16() * (1 - w.time_mix_k)
            xv = x * w.time_mix_v + state[5*i+1].bfloat16() * (1 - w.time_mix_v)
            xr = x * w.time_mix_r + state[5*i+1].bfloat16() * (1 - w.time_mix_r)
            state[5*i+1] = x.float()
        else:
            xk = x * w.time_mix_k + state[5*i+1] * (1 - w.time_mix_k)
            xv = x * w.time_mix_v + state[5*i+1] * (1 - w.time_mix_v)
            xr = x * w.time_mix_r + state[5*i+1] * (1 - w.time_mix_r)
            state[5*i+1] = x

        r = torch.sigmoid(w.receptance.weight @ xr)

        k = w.key.weight @ xk
        v = w.value.weight @ xv

        if os.environ["RWKV_FLOAT_MODE"] == "bf16":
            kk = k.float()
            vv = v.float()
            aa = state[5*i+2]
            bb = state[5*i+3]
            pp = state[5*i+4]
            ww = w.time_first + kk
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            a = e1 * aa + e2 * vv
            b = e1 * bb + e2
            ww = pp + w.time_decay
            p = torch.maximum(ww, kk)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(kk - p)
            state[5*i+2] = e1 * aa + e2 * vv
            state[5*i+3] = e1 * bb + e2
            state[5*i+4] = p
            rwkv = r * (a / b).bfloat16()
        else:
            aa = state[5*i+2]
            bb = state[5*i+3]
            pp = state[5*i+4]
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
            state[5*i+2] = e1 * aa + e2 * v
            state[5*i+3] = e1 * bb + e2
            state[5*i+4] = p
            rwkv = r * a / b

        return w.output.weight @ rwkv

    def forward(self, ctx, state, preprocess_only = False):
        with torch.no_grad():
            w = self.w

            x = w.emb.weight[ctx[-1]]
            if self.RUN_DEVICE == 'cuda':
                x = x.cuda()
            try:
                pos_emb = w.pos_emb[len(ctx)-1]
                x = x + pos_emb
            except:
                pass             

            if state == None:
                state = torch.zeros(self.n_layer * 5, self.n_embd, device=self.RUN_DEVICE)
                for i in range(self.n_layer):
                    state[5*i+4] -= 1e30

            for i in range(self.n_layer):
                if i == 0:
                    x = self.LN(x, w.blocks[i].ln0)
                x = x + self.SA(self.LN(x, w.blocks[i].ln1), w.blocks[i].att, state, i)
                x = x + self.FF(self.LN(x, w.blocks[i].ln2), w.blocks[i].ffn, state, i)

            if preprocess_only:
                return state

            x = self.LN(x, w.ln_out)
            x = w.head.weight @ x

            return x.float(), state
