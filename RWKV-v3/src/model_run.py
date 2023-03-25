########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types
import copy
import torch
import math
from torch.nn import functional as F
import torch.nn as nn

RWKV_K_CLAMP = 60
RWKV_K_EPS = 1e-8
RWKV_HEAD_QK_DIM = 256
print(
    f"\nRWKV_K_CLAMP {RWKV_K_CLAMP} RWKV_K_EPS {RWKV_K_EPS} RWKV_HEAD_QK_DIM {RWKV_HEAD_QK_DIM}\n"
)

DEBUG_TIME = False  # True False - show trained time-coeffs

############################################################################################################

RWKV_CFG = types.SimpleNamespace()


class RWKV_ChannelMix(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
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
        self.time_decay = nn.Parameter(torch.ones(RWKV_CFG.n_embd, 1))
        self.time_curve = torch.tensor(
            [-(RWKV_CFG.ctx_len - 2 - i) for i in range(RWKV_CFG.ctx_len - 1)]
        ).unsqueeze(0)
        self.time_first = nn.Parameter(torch.ones(RWKV_CFG.n_embd, 1) * math.log(0.3))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, RWKV_CFG.n_embd))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, RWKV_CFG.n_embd))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, RWKV_CFG.n_embd))

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

        k = self.key(xk).transpose(-1, -2)
        v = self.value(xv).transpose(-1, -2)
        r = self.receptance(xr)

        k = torch.clamp(k, max=RWKV_K_CLAMP)
        k = torch.exp(k)

        kv = k * v

        self.time_w = torch.cat(
            [
                torch.exp(self.time_decay) * self.time_curve.to(self.time_decay.device),
                self.time_first,
            ],
            dim=-1,
        )
        w = torch.exp(self.time_w)

        w = w[:, -T:].unsqueeze(1)
        wkv = F.conv1d(nn.ZeroPad2d((T - 1, 0, 0, 0))(kv), w, groups=C)
        wk = F.conv1d(nn.ZeroPad2d((T - 1, 0, 0, 0))(k), w, groups=C) + RWKV_K_EPS

        rwkv = torch.sigmoid(r) * (wkv / wk).transpose(-1, -2)

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

        if self.layer_id == 0 and RWKV_CFG.model_type == "RWKV-ffnPre":
            self.ffnPre = RWKV_ChannelMix(layer_id + 1000)
        else:
            self.att = RWKV_TimeMix(layer_id)

        self.ffn = RWKV_ChannelMix(layer_id)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)
        if self.layer_id == 0 and RWKV_CFG.model_type == "RWKV-ffnPre":
            x = x + self.ffnPre(self.ln1(x))
        else:
            x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class RWKV_GPT(nn.Module):
    def __init__(
        self, MODEL_NAME, RUN_DEVICE, model_type, vocab_size, n_layer, n_embd, ctx_len
    ):
        global RWKV_CFG
        super().__init__()

        RWKV_CFG.RUN_DEVICE = RUN_DEVICE
        RWKV_CFG.model_type = model_type
        RWKV_CFG.vocab_size = vocab_size
        RWKV_CFG.n_layer = n_layer
        RWKV_CFG.n_embd = n_embd
        RWKV_CFG.ctx_len = ctx_len

        print("\nloading RWKV-GPT", MODEL_NAME)

        self.emb = nn.Embedding(vocab_size, n_embd)

        self.blocks = nn.Sequential(*[Block(i) for i in range(n_layer)])

        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        if RWKV_HEAD_QK_DIM > 0:
            self.head_q = nn.Linear(n_embd, RWKV_HEAD_QK_DIM, bias=False)
            self.head_q.scale_init = 0
            self.head_k = nn.Linear(n_embd, RWKV_HEAD_QK_DIM, bias=False)
            self.head_k.scale_init = 0.1
            self.register_buffer("copy_mask", torch.tril(torch.ones(ctx_len, ctx_len)))

        self.ctx_len = ctx_len
        self.eval()
        self.load_state_dict(torch.load(MODEL_NAME + ".pth"))
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


class RWKV_RNN:
    def __init__(self, MODEL_NAME, RUN_DEVICE, model_type, n_layer, n_embd, ctx_len):
        self.RUN_DEVICE = RUN_DEVICE
        self.model_type = model_type
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.ctx_len = ctx_len

        self.w = types.SimpleNamespace()

        w = torch.load(MODEL_NAME + ".pth", map_location=torch.device(RUN_DEVICE))
        for x in w.keys():
            if ".time_" in x:
                w[x] = w[x].squeeze()
            if ".time_decay" in x:
                w[x] = torch.exp(-torch.exp(w[x]))
            if ".time_first" in x:
                w[x] = torch.exp(w[x])
            if DEBUG_TIME and ".time_" in x:
                print(x, w[x].squeeze().cpu().numpy())

            xx = x.split(".")
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
                        if xx[i + 1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        self.clear()

    def clear(self):
        self.xx = {}
        self.aa = {}
        self.bb = {}
        self.hk = None

    def save(self, target):
        target.xx = copy.deepcopy(self.xx)
        target.aa = copy.deepcopy(self.aa)
        target.bb = copy.deepcopy(self.bb)
        target.hk = copy.deepcopy(self.hk)

    def load(self, target):
        self.xx = copy.deepcopy(target.xx)
        self.aa = copy.deepcopy(target.aa)
        self.bb = copy.deepcopy(target.bb)
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

        xk = xx * w.time_mix_k + self.xx[name] * (1 - w.time_mix_k)
        xv = xx * w.time_mix_v + self.xx[name] * (1 - w.time_mix_v)
        xr = xx * w.time_mix_r + self.xx[name] * (1 - w.time_mix_r)
        self.xx[name] = xx

        r = torch.sigmoid(w.receptance.weight @ xr)

        k = torch.exp(torch.clamp(w.key.weight @ xk, max=RWKV_K_CLAMP))
        v = w.value.weight @ xv
        kv = k * v

        a = self.aa[name] + w.time_first * kv
        b = self.bb[name] + w.time_first * k
        self.aa[name] = w.time_decay * self.aa[name] + kv
        self.bb[name] = w.time_decay * self.bb[name] + k

        rwkv = r * a / (b + RWKV_K_EPS)

        return w.output.weight @ rwkv

    def run(self, ctx):
        w = self.w
        x = w.emb.weight[ctx[-1]]

        for i in range(self.n_layer):
            if i == 0:
                x = self.LN(x, w.blocks[i].ln0)
            if i == 0 and self.model_type == "RWKV-ffnPre":
                x = x + self.FF(
                    self.LN(x, w.blocks[i].ln1), w.blocks[i].ffnPre, f"ffnPre.{i}"
                )
            else:
                x = x + self.SA(
                    self.LN(x, w.blocks[i].ln1), w.blocks[i].att, f"att.{i}"
                )
            x = x + self.FF(self.LN(x, w.blocks[i].ln2), w.blocks[i].ffn, f"ffn.{i}")

        x = self.LN(x, w.ln_out)

        if RWKV_HEAD_QK_DIM > 0:
            if self.hk == None:
                self.hk = (w.head_k.weight @ x).unsqueeze(0)
            else:
                self.hk = torch.cat(
                    [self.hk, (w.head_k.weight @ x).unsqueeze(0)], dim=0
                )
            if self.hk.shape[0] > self.ctx_len:
                self.hk = self.hk[-self.ctx_len :, :]

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
