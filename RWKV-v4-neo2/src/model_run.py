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
from torch import autocast
import numpy as np
from tqdm import tqdm
# Make sure to use nightly build of torchdynamo
# import torchdynamo
# MyFunction = torchdynamo.optimize(
#     "nvfuser")  # !!!BUGGY!!! wrong output

RWKV_HEAD_QK_DIM = 0
print(f'\nRWKV_HEAD_QK_DIM {RWKV_HEAD_QK_DIM}\n')

DEBUG_TIME = False   # True False - show trained time-coeffs


@torch.jit.ignore
def sample(probs, temperature: float = 1.0, top_p_usual: float = 0.8) -> int:

    if probs.device.type == "cpu":
        probs = probs.numpy()
        sorted_probs = np.sort(probs)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(
            cumulative_probs > top_p_usual)])
        probs[probs < cutoff] = 0
        if temperature != 1.0:
            probs = np.power(probs, 1.0 / temperature)
        probs = probs / np.sum(probs)
        out = np.random.choice(a=len(probs), p=probs)
        return out
    else:
        sorted_probs = torch.sort(probs, descending=True)[0]
        cumulative_probs = torch.cumsum(
            sorted_probs.float(), dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[np.argmax(
            cumulative_probs > top_p_usual)])
        probs[probs < cutoff] = 0
        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)

        out: int = torch.multinomial(probs.float(), 1, True)[0]
        return out


class RWKV_RNN(nn.Module):
    def __init__(self, args, argsnumns):
        super().__init__()

        self.args = args
        self.argsnumns = argsnumns
        self.FLOAT_MODE = args["FLOAT_MODE"]
        self.RUN_DEVICE = args["RUN_DEVICE"]
        self.n_layer = 0
        with torch.no_grad():
            w: Dict[str, torch.Tensor] = torch.load(
                args["MODEL_NAME"], map_location='cpu')
            self.n_emb = len(w['blocks.0.ln1.weight'])
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
                if args["RUN_DEVICE"] == "cuda" and x != 'emb.weight':
                    try:
                        if (int(x.split('.')[1]) > self.n_layer):
                            self.n_layer = int(x.split('.')[1])+1
                    except:
                        pass
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

    def FF(self, x, state, i: int, time_mix_k, time_mix_r, kw, vw, rw):

        xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
        state[5*i+0] = x

        r = torch.sigmoid((rw @ xr))
        dx = (kw @ xk)
        clamped = torch.relu(dx)
        k = torch.square(clamped)
        kv = (vw @ k)
        return (r * kv)

    # @MyFunction

    def SA(self, x, state, i: int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):

        xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)

        state[5*i+1] = x

        r = torch.sigmoid((rw @ xr))
        k = (kw @ xk)
        v = (vw @ xv)

        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)

        a = e1 * aa + e2 * v
        b = e1 * bb + e2

        ww = pp + time_decay
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)
        state[5*i+2] = e1 * aa + e2 * v
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = p

        rwkv = (r * a) / b
        return (ow @ rwkv)

    def forward(self, ctx: List[int], state: torch.Tensor, preprocess_only: bool = False):
        with torch.no_grad():
            w = self.w
            args = self.args

            x: torch.Tensor = w["emb.weight"][ctx[-1]]

            if self.RUN_DEVICE == 'cuda':
                x = x.to(device="cuda", non_blocking=True)

            if ("pos_emb" in w.keys()):
                pos_emb = w["pos_emb"][len(ctx)-1]
                x = x + pos_emb

            for o in range(self.n_layer):
                i = o

                d: dict[str, torch.Tensor] = w
                if (i >= self.argsnumns["cudalayers"]):
                    d = {}
                    for rr in w.keys():
                        if ("blocks."+str(i)+"." in rr):

                            d[rr] = w[rr].to("cuda", non_blocking=True)

                if o == 0:
                    x = torch.layer_norm(
                        x, (self.n_emb,), weight=d["blocks.0.ln0.weight"], bias=d["blocks.0.ln0.bias"])

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

                ln = torch.layer_norm(
                    x, (self.n_emb,), weight=ln1w, bias=ln1b)
                sx = self.SA(ln, state, i,
                             atmk, atmv, atmr, atf, atc, atd, avw, arw, aow
                             )
                x = x + sx
                rx = self.FF(torch.layer_norm(x, (self.n_emb,), weight=ln2w, bias=ln2b), state, i,
                             tmk, tmr, tmkw, tmvw, tmrw)
                x += rx
                if (i >= self.argsnumns["cudalayers"]):

                    for rr in w.keys():
                        if ("blocks."+str(i)+"." in rr):

                            del d[rr]
            if preprocess_only:
                return x
            if args["RUN_DEVICE"] == 'cuda':
                x = x.to("cuda")

            x = torch.layer_norm(
                x, (self.n_emb,), weight=w["ln_out.weight"], bias=w["ln_out.bias"])

            x = (w["head.weight"] @ x)

            return x

    @torch.jit.export
    def empty_state(self):
        state = torch.zeros(
            self.n_layer * 5, self.n_emb, device=self.RUN_DEVICE, dtype=torch.float32 if self.FLOAT_MODE == "fp32" else torch.bfloat16 if self.FLOAT_MODE == "bf16" else torch.float16)
        for i in range(self.n_layer):
            state[5*i+4] -= 1e30
        return state

    @torch.jit.ignore
    def loadContext(self, ctx: List[int], state: torch.Tensor, start: int = 0):
        for i in tqdm(range(len(ctx))[start:]):
            x = ctx[: i + 1]
            if i == len(ctx) - 1:
                init_out = self.forward(x, state, preprocess_only=False)
            else:
                o = self.forward(
                    x, state, preprocess_only=True)
        return

    @torch.jit.export
    def sample_logits(self, ozut: torch.Tensor, x: List[int], ctx_len: int, temperature: float = 1.0, top_p_usual: float = 0.8):
        # out[self.UNKNOWN_CHAR] = -float('Inf')
       # out[self.UNKNOWN_CHAR] = -float('Inf')
        lastChar = int(x[-1])

        # turn to float if is half and cpu
        out = ozut
        if out.dtype == torch.half and out.device == torch.device('cpu'):
            out = out.float()
        probs = F.softmax(out, dim=-1)

        return sample(probs, temperature, top_p_usual)

    @torch.jit.export
    def run(self, ctxx: List[int], state1: torch.Tensor, ctxlen: int = 1024, temp: float = 1.2, top_p: float = 0.8, nla: float = 0):

        out1 = self.forward(ctxx, state1, preprocess_only=False)

        out1[0] = -99  # disable <|endoftext|>

        out1[187] += nla

        ttt = self.sample_logits(
            out1,
            ctxx,
            ctxlen,
            temperature=temp,
            top_p_usual=top_p,
        )
        ctxx += [ttt]

        return ctxx
