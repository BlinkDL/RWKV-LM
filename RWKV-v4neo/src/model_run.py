########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from ast import Delete
import types
import torch
import math
import os
import gc
from torch.nn import functional as F
import torch.nn as nn
from typing import List, Dict

# Make sure to use nightly build of torchdynamo
# import torchdynamo
# MyFunction = torchdynamo.optimize(
#     "nvfuser")  # !!!BUGGY!!! wrong output

RWKV_HEAD_QK_DIM = 0
print(f'\nRWKV_HEAD_QK_DIM {RWKV_HEAD_QK_DIM}\n')

DEBUG_TIME = False   # True False - show trained time-coeffs

############################################################################################################


class RWKV_RNN(nn.Module):
    def __init__(self, args, argsnumns):
        super().__init__()

        self.args = args
        self.argsnumns = argsnumns
        self.FLOAT_MODE = args["FLOAT_MODE"]
        self.RUN_DEVICE = args["RUN_DEVICE"]
        with torch.no_grad():
            w: Dict[str, torch.Tensor] = torch.load(
                args["MODEL_NAME"] + '.pth', map_location='cpu')

            # refine weights and send to correct device
            keys = list(w.keys())
            if 'pos_emb_x' in keys:
                w['pos_emb'] = (w['pos_emb_x'] + w['pos_emb_y']
                                ).reshape(argsnumns["ctx_len"]+1, -1)[:-1, :]
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
                if args["RUN_DEVICE"] in ["cuda", "proc"] and x != 'emb.weight':

                    if ((x.split('.')[1] == "weight" or x.split('.')[1] == "bias") or int(x.split('.')[1]) < argsnumns["cudalayers"]):

                        w[x] = w[x].cuda(non_blocking=True)
                if (args["RUN_DEVICE"] == 'proc'):
                    if (w[x].device.type == "cpu"):
                        w[x] = w[x].pin_memory()
                    if (('blocks.' not in x)) and x != 'emb.weight':
                        w[x] = w[x].cuda(non_blocking=True)
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
        self.w = w

        self.eval()
        gc.collect()
        torch.cuda.empty_cache()

    # @MyFunction
    def LN(self, x: torch.Tensor, w, b):
        if (self.FLOAT_MODE == "fp32"):
            return torch.layer_norm(x, (self.argsnumns["n_embd"],), weight=w, bias=b)
        elif (self.FLOAT_MODE == "bf16"):
            return torch.layer_norm(x, (self.argsnumns["n_embd"],), weight=w, bias=b)
        else:
            # layer_norm is not supported in fp16
            # layer norm on half-in
            if (x.device.type is "cpu"):
                return torch.layer_norm(x, (self.argsnumns["n_embd"],), weight=w, bias=b)
            else:
                # abandon all hope, ye who enter here
                return torch.layer_norm(
                    x.float(), (self.argsnumns["n_embd"],), weight=w.float(), bias=b.float()).half()

            # return F.layer_norm(x.float(), (self.args["n_embd"],), weight=w.weight.float(), bias=w.bias.float()).half()

    # state[] 0=ffn_xx 1=att_xx 2=att_aa 3=att_bb 4=att_pp
    # @MyFunction
    def FF(self, x, state, i: int, time_mix_k, time_mix_r, kw, vw, rw):
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
            if (x.device.type is "cuda"):
                r = torch.sigmoid(torch.matmul(rw, xr))
                k = torch.square(torch.relu(torch.matmul(kw, xk)))
                kv = torch.matmul(vw, k)
                return (r * kv)
            else:
                r = torch.sigmoid(torch.matmul(rw.float(), xr.float()))
                k = torch.square(torch.relu(
                    torch.matmul(kw.float(), xk.float())))
                kv = torch.matmul(vw.float(), k.float())
                return (r * kv).half()
        r = torch.sigmoid(torch.matmul(rw, xr))
        k = torch.square(torch.relu(torch.matmul(kw, xk)))
        kv = torch.matmul(vw, k)
        return (r * kv)

    # @MyFunction
    def SA(self, x, state, i: int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
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
            if (rw.device.type is "cuda"):
                r = torch.sigmoid(torch.matmul(rw, xr))
                k = torch.matmul(kw, xk)
                v = torch.matmul(vw, xv)
                kk = k
                vv = v
            else:
                r = torch.sigmoid(torch.matmul(rw.float(), xr.float()))
                k = torch.matmul(kw.float(), xk.float())
                v = torch.matmul(vw.float(), xv.float())
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
            wkv = (a / b)
            if (rw.device.type is "cuda"):
                wkv = wkv.half()
        else:
            wkv = a / b

        if (self.FLOAT_MODE == "fp16"):

            # print types
            if (rw.device.type is "cuda"):
                return torch.matmul(ow, r * wkv)
            else:
                return torch.matmul(ow.float(), r * wkv).half()
            # return (ow.float() @ (r * wkv)).half()
        elif (self.FLOAT_MODE == "bf16"):
            return (ow @ (r * wkv))
        else:
            return ow @ (r * wkv)

    def forward(self, ctx: List[int], state: torch.Tensor, preprocess_only: bool = False):
        with torch.no_grad():
            w = self.w
            args = self.args

            x: torch.Tensor = w["emb.weight"][ctx[-1]]

            if self.RUN_DEVICE == 'cuda' or self.RUN_DEVICE == "proc":
                x = x.to(device="cuda", non_blocking=True)

            if ("pos_emb" in w.keys()):
                pos_emb = w["pos_emb"][len(ctx)-1]
                x = x + pos_emb

            for o in range(self.argsnumns["n_layer"]):
                i = o

                if (i >= self.argsnumns["cudalayers"] and self.RUN_DEVICE == "cuda"):

                    x = x.to("cpu")
                    state = state.to("cpu")
                d: dict[str, torch.Tensor] = w
                if (self.RUN_DEVICE == "proc" and i >= self.argsnumns["cudalayers"]):
                    d = {}
                    for rr in w.keys():
                        if ("blocks."+str(i)+"." in rr):

                            d[rr] = w[rr].to("cuda", non_blocking=True)

                if o == 0:
                    x = self.LN(
                        x, d["blocks.0.ln0.weight"], d["blocks.0.ln0.bias"])

                ln1w = d["blocks."+str(i)+".ln1.weight"]
                ln1b = d["blocks."+str(i)+".ln1.bias"]

                tmk = d["blocks."+str(i)+".ffn.time_mix_k"]
                tmr = d["blocks."+str(i)+".ffn.time_mix_r"]
                tmkw = d["blocks."+str(i)+".ffn.key.weight"]
                tmvw = d["blocks."+str(i)+".ffn.value.weight"]
                tmrw = d["blocks."+str(i)+".ffn.receptance.weight"]
                ln2w = d["blocks."+str(i)+".ln2.weight"]
                ln2b = d["blocks."+str(i)+".ln2.bias"]
                atmk = d["blocks."+str(i)+".att.time_mix_k"]
                atmv = d["blocks."+str(i)+".att.time_mix_v"]
                atmr = d["blocks."+str(i)+".att.time_mix_r"]
                atf = d["blocks."+str(i)+".att.time_first"]
                atc = d["blocks."+str(i)+".att.time_decay"]
                atd = d["blocks."+str(i)+".att.key.weight"]
                avw = d["blocks."+str(i)+".att.value.weight"]
                arw = d["blocks."+str(i)+".att.receptance.weight"]
                aow = d["blocks."+str(i)+".att.output.weight"]

                ln = self.LN(x, ln1w, ln1b)
                x = x + self.SA(ln, state, i,
                                atmk, atmv, atmr, atf, atc, atd, avw, arw, aow
                                )

                x = x + self.FF(self.LN(x, ln2w, ln2b), state, i,
                                tmk, tmr, tmkw, tmvw, tmrw)
                if (self.RUN_DEVICE == "proc" and i >= self.argsnumns["cudalayers"]):

                    for rr in w.keys():
                        if ("blocks."+str(i)+"." in rr):

                            del d[rr]

            if args["RUN_DEVICE"] == 'cuda':
                state = state.to("cuda")
            if preprocess_only:
                return x, state
            if args["RUN_DEVICE"] == 'cuda':
                x = x.to("cuda")

            x = self.LN(x, w["ln_out.weight"], w["ln_out.bias"])
            if (args["RUN_DEVICE"] == "cpu" and self.FLOAT_MODE == "fp16"):
                x = torch.matmul(w["head.weight"].float(), x.float()).half()
            else:
                x = torch.matmul(w["head.weight"], x)

            return x, state
