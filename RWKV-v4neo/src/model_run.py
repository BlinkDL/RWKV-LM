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
        if self.RUN_DEVICE == "cpu" and self.FLOAT_MODE == "fp16":
            return torch.layer_norm(x.float(), (self.argsnumns["n_embd"],), weight=w.float(), bias=b.float()).half()
        return torch.layer_norm(x, (self.argsnumns["n_embd"],), weight=w, bias=b)

    def MM(self, x: torch.Tensor, y: torch.Tensor):
        if self.RUN_DEVICE == "cpu" and self.FLOAT_MODE == "fp16":
            return torch.matmul(x.float(), y.float()).half()
        return torch.matmul(x, y)

    def SM(self, x: torch.Tensor):
        if self.RUN_DEVICE == "cpu" and self.FLOAT_MODE == "fp16":
            return torch.softmax(x.float(), dim=-1).half()
        return torch.softmax(x, dim=-1)

    def SG(self, x: torch.Tensor):
        if self.RUN_DEVICE == "cpu" and self.FLOAT_MODE == "fp16":
            return torch.sigmoid(x.float()).half()
        return torch.sigmoid(x)

    def EX(self, x: torch.Tensor):
        if self.RUN_DEVICE == "cpu" and self.FLOAT_MODE == "fp16":
            return torch.exp(x.float()).half()
        return torch.exp(x)

    def RL(self, x: torch.Tensor):
        if self.RUN_DEVICE == "cpu" and self.FLOAT_MODE == "fp16":
            return torch.relu(x.float()).half()
        return torch.relu(x)
    # @MyFunction

    def FF(self, x, state, i: int, time_mix_k, time_mix_r, kw, vw, rw):

        xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
        state[5*i+0] = x

        r = self.SG(self.MM(rw, xr))
        dx = self.MM(kw, xk)
        clamped = self.RL(dx)
        k = torch.square(clamped)
        kv = self.MM(vw, k)
        return (r * kv)

    # @MyFunction

    def SA(self, x, state, i: int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):

        xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
        state[5*i+1] = x

        r = self.SG(self.MM(rw, xr))
        k = self.MM(kw, xk)
        v = self.MM(vw, xv)

        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + k
        p = torch.maximum(pp, ww)
        e1 = self.EX(pp - p)
        e2 = self.EX(ww - p)

        a = e1 * aa + e2 * v
        b = e1 * bb + e2

        ww = pp + time_decay
        p = torch.maximum(ww, k)
        e1 = self.EX(ww - p)
        e2 = self.EX(k - p)
        state[5*i+2] = e1 * aa + e2 * v
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = p

        rwkv = (r * a) / b
        return self.MM(ow, rwkv)

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

            x = self.MM(w["head.weight"], x)

            return x, state
