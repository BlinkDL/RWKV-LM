########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types
import torch
import math
import os
import gc
from torch.nn import functional as F
import torch.nn as nn
from typing import List, Dict

# try:
#     import torchdynamo
#     MyFunction = torchdynamo.optimize(os.environ["RWKV_RUN_BACKEND"]) # !!!BUGGY!!! wrong output
# except:


def __nop(ob):
    return ob


MyFunction = __nop

RWKV_HEAD_QK_DIM = 0
print(f'\nRWKV_HEAD_QK_DIM {RWKV_HEAD_QK_DIM}\n')

DEBUG_TIME = False   # True False - show trained time-coeffs

############################################################################################################


class RWKV_RNN(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.FLOAT_MODE = args.FLOAT_MODE
        self.RUN_DEVICE = args.RUN_DEVICE

        with torch.no_grad():
            w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
            # refine weights and send to correct device
            keys = list(w.keys())
            if 'pos_emb_x' in keys:
                w['pos_emb'] = (w['pos_emb_x'] + w['pos_emb_y']
                                ).reshape(args.ctx_len+1, -1)[:-1, :]
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
                    if self.FLOAT_MODE == "fp32":
                        w[x] = w[x].float()

                    elif self.FLOAT_MODE == "bf16":
                        w[x] = w[x].bfloat16()

                    elif self.FLOAT_MODE == "fp16":
                        if ('weight' in x or 'bias' in x) and 'ln' in x:
                            w[x] = w[x].half()
                        else:
                            w[x] = w[x].half()

                w[x].requires_grad = False
                if args.RUN_DEVICE == 'cuda' and x != 'emb.weight':

                    if ((x.split('.')[1] == "weight" or x.split('.')[1] == "bias") or int(x.split('.')[1]) < args.cudalayers):
                        w[x] = w[x].cuda()

                if ('blocks.' not in x) or ('blocks.0.' in x):
                    if print_need_newline:
                        print('\n', end='')
                        print_need_newline = False
                    print(x.ljust(40), str(w[x].dtype).replace(
                        'torch.', '').ljust(10), w[x].device)
                else:
                    print_need_newline = True
                    print(
                        '.' if "cpu" in f'{w[x].device}' else "x", end='', flush=True)

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

    def LN(self, x, w):
        if (self.FLOAT_MODE == "fp32"):
            return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)
        elif (self.FLOAT_MODE == "bf16"):
            return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)
        elif (self.FLOAT_MODE == "fp16"):
            # layer_norm is not supported in fp16
            return torch.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)
            # return F.layer_norm(x.float(), (self.args.n_embd,), weight=w.weight.float(), bias=w.bias.float()).half()

    # state[] 0=ffn_xx 1=att_xx 2=att_aa 3=att_bb 4=att_pp

    @MyFunction
    def FF(self, x, state, i, time_mix_k, time_mix_r, kw, vw, rw):
        if self.FLOAT_MODE == "bf16":
            xk = x * time_mix_k + \
                state[5*i+0].type(torch.bfloat16) * (1 - time_mix_k)
            xr = x * time_mix_r + \
                state[5*i+0].type(torch.bfloat16) * (1 - time_mix_r)
            state[5*i+0] = x.float()
        elif self.FLOAT_MODE == "fp16":

            xk = x * time_mix_k + state[5*i+0].half() * (1 - time_mix_k)
            xr = x * time_mix_r + state[5*i+0].half() * (1 - time_mix_r)
            state[5*i+0] = x
        else:
            xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
            xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
            state[5*i+0] = x

        if (self.FLOAT_MODE == "fp16"):
            r = torch.sigmoid(torch.matmul(rw, xr))
            k = torch.square(torch.relu(torch.matmul(kw, xk)))
            kv = torch.matmul(vw, k)
            return (r * kv)
        r = torch.sigmoid(torch.matmul(rw, xr))
        k = torch.square(torch.relu(torch.matmul(kw, xk)))
        kv = torch.matmul(vw, k)
        return (r * kv)

    @MyFunction
    def SA(self, x, state, i, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        if self.FLOAT_MODE == "bf16":
            xk = x * time_mix_k + \
                state[5*i+1].type(torch.bfloat16) * (1 - time_mix_k)
            xv = x * time_mix_v + \
                state[5*i+1].type(torch.bfloat16) * (1 - time_mix_v)
            xr = x * time_mix_r + \
                state[5*i+1].type(torch.bfloat16) * (1 - time_mix_r)
            state[5*i+1] = x.float()
        elif self.FLOAT_MODE == "fp16":
            xk = x * time_mix_k + \
                state[5*i+1].type(torch.half) * (1 - time_mix_k)
            xv = x * time_mix_v + \
                state[5*i+1].type(torch.half) * (1 - time_mix_v)
            xr = x * time_mix_r + \
                state[5*i+1].type(torch.half) * (1 - time_mix_r)
            state[5*i+1] = x
        else:
            xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
            xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
            xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
            state[5*i+1] = x

        if self.FLOAT_MODE == "bf16":
            r = torch.sigmoid(rw @ xr)
            k = kw @ xk
            v = vw @ xv
            kk = k.float()
            vv = v.float()
        elif self.FLOAT_MODE == "fp16":
            # @ is not supported in fp16
            # r = torch.sigmoid(rw @ xr)
            # k = kw @ xk
            # v = vw @ xv
            # kk = k.float()
            # vv = v.float()
            #r = torch.sigmoid(rw.float() @ xr.float())
            r = torch.sigmoid(torch.matmul(rw, xr))
            #k = kw.float() @ xk.float()
            k = torch.matmul(kw, xk)
            #v = vw.float() @ xv.float()
            v = torch.matmul(vw, xv)
            kk = k
            vv = v
        else:
            r = torch.sigmoid(rw @ xr)
            k = kw @ xk
            v = vw @ xv
            kk = k
            vv = v
        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + kk
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        a = e1 * aa + e2 * vv
        b = e1 * bb + e2
        ww = pp + time_decay
        p = torch.maximum(ww, kk)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kk - p)
        state[5*i+2] = e1 * aa + e2 * vv
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = p
        if self.FLOAT_MODE == "bf16":
            wkv = (a / b).type(torch.bfloat16)
        elif self.FLOAT_MODE == "fp16":
            wkv = (a / b).half()
        else:
            wkv = a / b

        if (self.FLOAT_MODE == "fp16"):

            # print types
            return torch.matmul(ow, r * wkv)
            # return (ow.float() @ (r * wkv)).half()
        elif (self.FLOAT_MODE == "bf16"):
            return (ow @ (r * wkv))
        elif (self.FLOAT_MODE == "fp32"):
            return ow @ (r * wkv)

    def forward(self, ctx, state, preprocess_only=False):
        with torch.no_grad():
            w = self.w
            args = self.args

            x = w.emb.weight[ctx[-1]]
            if self.RUN_DEVICE == 'cuda':
                x = x.cuda()
            try:
                pos_emb = w.pos_emb[len(ctx)-1]
                x = x + pos_emb
            except:
                pass

            if state == None:
                state = torch.zeros(
                    args.n_layer * 5, args.n_embd, device=self.RUN_DEVICE)
                for i in range(args.n_layer):
                    state[5*i+4] -= 1e30

            for i in range(args.n_layer):
                if (x.device != w.blocks[i].ln1.weight.device):

                    x = x.to("cpu")
                    state = state.to("cpu")
                if i == 0:
                    x = self.LN(x, w.blocks[i].ln0)

                ww = w.blocks[i].att
                x = x + self.SA(self.LN(x, w.blocks[i].ln1), state, i,
                                ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay,
                                ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)

                ww = w.blocks[i].ffn
                x = x + self.FF(self.LN(x, w.blocks[i].ln2), state, i,
                                ww.time_mix_k, ww.time_mix_r,
                                ww.key.weight, ww.value.weight, ww.receptance.weight)
            if args.RUN_DEVICE == 'cuda':
                state = state.to("cuda")
            if preprocess_only:
                return state
            if args.RUN_DEVICE == 'cuda':
                x = x.to("cuda")
            x = self.LN(x, w.ln_out)
            x = torch.matmul(w.head.weight, x)

            return x, state
