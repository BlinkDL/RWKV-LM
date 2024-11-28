########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import torch, types, os, gc, math, json
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
np.set_printoptions(precision=4, suppress=True, linewidth=200)

'''
This will load RWKV-7 "Goose" x070 and inference in GPT-mode (slower than RNN-mode for autoregressive generation)
'''

args = types.SimpleNamespace()

# model download: https://huggingface.co/BlinkDL/rwkv-7-pile

MODEL_PATH = "/mnt/e/RWKV-x070-Pile-168M-20241120-ctx4096.pth"
# MODEL_PATH = "/mnt/program/RWKV-x070-Pile-421M-20241127-ctx4096.pth"

if '168M' in MODEL_PATH:
    args.n_layer = 12
    args.n_embd = 768
    D_DECAY_LORA = 64
    D_AAA_LORA = 64
    D_MV_LORA = 32
    D_GATE_LORA = 128
elif '421M' in MODEL_PATH:
    args.n_layer = 24
    args.n_embd = 1024
    D_DECAY_LORA = 64
    D_AAA_LORA = 64
    D_MV_LORA = 64
    D_GATE_LORA = 128

args.vocab_size = 50304 # "pile" model: 50277 padded to 50304   
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("../RWKV-v4neo/20B_tokenizer.json")

# DTYPE = torch.bfloat16
DTYPE = torch.half # better

args.head_size_a = 64 # don't change
HEAD_SIZE = args.head_size_a

USE_CUDA_KERNEL = True # False => UNOPTIMIZED, VERY SLOW

########################################################################################################
# CUDA Kernel
########################################################################################################

if USE_CUDA_KERNEL:

    from torch.utils.cpp_extension import load

    load(name="wkv7", sources=["cuda/wkv7_op.cpp", f"cuda/wkv7.cu"], is_python_module=False,
                        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
    class WKV_7(torch.autograd.Function):
        @staticmethod
        def forward(ctx, r, w, k, v, a, b):
            with torch.no_grad():
                B, T, C = r.size()
                H = C // HEAD_SIZE
                N = HEAD_SIZE
                assert HEAD_SIZE == C // H
                assert r.dtype == DTYPE
                assert w.dtype == DTYPE
                assert k.dtype == DTYPE
                assert v.dtype == DTYPE
                assert a.dtype == DTYPE
                assert b.dtype == DTYPE
                assert r.is_contiguous()
                assert w.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert a.is_contiguous()
                assert b.is_contiguous()
                y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, memory_format=torch.contiguous_format)
                torch.ops.wkv7.forward(B, T, C, H, r, w, k, v, a, b, y)
                return y

    def RWKV7_OP(r, w, k, v, a, b):
        return WKV_7.apply(r, w, k, v, a, b)

else:

    def RWKV7_OP(r, w, k, v, a, b):
        B, T, C = r.size()
        H = C // HEAD_SIZE
        N = HEAD_SIZE
        r = r.view(B, T, H, N).float()
        k = k.view(B, T, H, N).float()
        v = v.view(B, T, H, N).float()
        a = a.view(B, T, H, N).float()
        b = b.view(B, T, H, N).float()
        w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
        out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
        state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)

        for t in range(T):
            kk = k[:, t, :]
            rr = r[:, t, :]
            vv = v[:, t, :]
            aa = a[:, t, :]
            bb = b[:, t, :]
            sab = torch.einsum('bhik,bhk,bhj->bhij', state, aa, bb)
            state = state * w[: , t, :, None, :] + sab + torch.einsum('bhj,bhi->bhij', kk, vv)
            out[:, t, :] = torch.einsum('bhj,bhij->bhi', rr, state)

        return out.view(B, T, C).to(dtype=DTYPE)

########################################################################################################
# RWKV TimeMix
########################################################################################################

class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        H = self.n_head
        N = self.head_size
        C = args.n_embd

        self.xx_r = nn.Parameter(torch.empty(1,1,C))
        self.xx_w = nn.Parameter(torch.empty(1,1,C))
        self.xx_k = nn.Parameter(torch.empty(1,1,C))
        self.xx_v = nn.Parameter(torch.empty(1,1,C))
        self.xx_a = nn.Parameter(torch.empty(1,1,C))
        self.xx_g = nn.Parameter(torch.empty(1,1,C))

        self.ww_b = nn.Parameter(torch.empty(1,1,C))
        self.ww_w1 = nn.Parameter(torch.empty(C, D_DECAY_LORA))
        self.ww_w2 = nn.Parameter(torch.empty(D_DECAY_LORA, C))

        self.aa_b = nn.Parameter(torch.empty(1,1,C))
        self.aa_w1 = nn.Parameter(torch.empty(C, D_AAA_LORA))
        self.aa_w2 = nn.Parameter(torch.empty(D_AAA_LORA, C))

        if layer_id > 0:
            self.vv_b = nn.Parameter(torch.empty(1,1,C))
            self.vv_w1 = nn.Parameter(torch.empty(C, D_MV_LORA))
            self.vv_w2 = nn.Parameter(torch.empty(D_MV_LORA, C))

        self.gg_w1 = nn.Parameter(torch.empty(C, D_GATE_LORA))
        self.gg_w2 = nn.Parameter(torch.empty(D_GATE_LORA, C))

        self.kk_s = nn.Parameter(torch.empty(1,1,C))
        self.ka_s = nn.Parameter(torch.empty(1,1,C))
        self.rk_s = nn.Parameter(torch.empty(H,N))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!

    def forward(self, x, v0):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x
        xr = x + xx * self.xx_r
        xw = x + xx * self.xx_w
        xk = x + xx * self.xx_k
        xv = x + xx * self.xx_v
        xa = x + xx * self.xx_a
        xg = x + xx * self.xx_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.ww_b + torch.tanh(xw @ self.ww_w1) @ self.ww_w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v0 = v
        else:
            v = v + (v0 - v) * torch.sigmoid(self.vv_b + (xv @ self.vv_w1) @ self.vv_w2)
        a = torch.sigmoid(self.aa_b + (xa @ self.aa_w1) @ self.aa_w2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.gg_w1) @ self.gg_w2

        kk = k * self.kk_s
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.ka_s)

        x = RWKV7_OP(r, w, k, v, -kk, kk*a)

        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.rk_s).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)

        x = self.output(x * g)
        return x, v0
    
########################################################################################################
# RWKV ChannelMix
########################################################################################################

class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            self.xx_k = nn.Parameter(torch.empty(1, 1, args.n_embd))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.xx_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

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

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
    def forward(self, x, v0):

        if self.layer_id == 0:
            x = self.ln0(x)

        xx, v0 = self.att(self.ln1(x), v0)
        x = x + xx
        x = x + self.ffn(self.ln2(x))

        return x, v0

########################################################################################################
# RWKV Model
########################################################################################################

class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.dim_att = args.n_embd
        args.dim_ffn = args.n_embd * 4
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx):

        x = self.emb(idx)

        v0 = torch.empty_like(x)
        for block in self.blocks:
            x, v0 = block(x, v0)

        x = self.ln_out(x)
        x = self.head(x)

        return x

########################################################################################################
# RWKV Inference
########################################################################################################

model_params = torch.load(MODEL_PATH, map_location="cpu")

with torch.no_grad():

    model = RWKV(args).to(dtype=DTYPE).cuda()
    model.load_state_dict(model_params)

    ########################################################################################################

    prompt = "The Eiffel tower is in the city of"
    input = tokenizer.encode(prompt).ids
    print(f'\nInput:\n{input}')

    out = model.forward(torch.tensor(input).reshape(1,-1).cuda())
    print(f'\nOutput:\n{out}')

    # logits of the last token => prediction for the next token    
    out = out[0, -1]
    
    probs = F.softmax(out.float(), dim=-1) # compute softmax in float (more accurate)

    print(f'\n{prompt}')

    _, indices = torch.topk(probs, 10) # print top-10 possibilities
    for i in range(len(indices)):
        token_id = indices[i].item()
        token = tokenizer.decode([token_id])
        token_prob = probs[token_id].item()
        print(token, f'[probability {token_prob:.2%}]')

    ########################################################################################################

    with open(f"misc/lambada_test.jsonl", "r", encoding="utf-8") as f:
        todo = [json.loads(line) for line in f]
        todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

    print('\nCheck LAMBADA...')
    xsum = 0
    xcnt = 0
    xacc = 0
    for d in todo:
        src = [0] + tokenizer.encode(d[0]).ids
        dst = tokenizer.encode(d[1]).ids

        logits = 0
        correct = True
        out = model.forward(torch.tensor(src+dst).reshape(1,-1).cuda())
        for i in range(len(dst)):
            ooo = out[0,len(src)-1+i].float()
            probs = F.softmax(ooo, dim=-1)
            logits += math.log(probs[dst[i]])
            if torch.argmax(probs).item() != dst[i]:
                correct = False

        xcnt += 1
        xsum += logits
        xacc += 1 if correct else 0
        if xcnt % 100 == 0 or xcnt == len(todo):
            print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 2))
