########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import torch, types, os, gc, math, json
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
np.set_printoptions(precision=4, suppress=True, linewidth=200)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)

'''
This will load RWKV-8 "Heron" ROSA-4bit demo and inference in GPT-mode
'''

args = types.SimpleNamespace()

# model download: https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main

args.MODEL_NAME = "/mnt/e/temp/rwkv-rosa4bit-minipile-loss3dot44-20260221-ctx512.pth"

# for 0.1B
args.n_layer = 12
args.n_embd = 768

args.vocab_size = 65536

DTYPE = torch.half

# MyModule = torch.jit.ScriptModule
# MyFunction = torch.jit.script_method
# MyStatic = torch.jit.script

def __nop(ob):
    return ob
MyModule = nn.Module
MyFunction = __nop
MyStatic = __nop

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

tokenizer = RWKV_TOKENIZER("../rwkv_vocab_v20230424.txt")

@MyStatic
def sample_logits(logits, temperature:float=1.0, top_p:float=1.0, top_k:int=0):
    probs = F.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    
    if top_k > 0:
        probs[sorted_ids[top_k:]] = 0

    if top_p < 1:
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.searchsorted(cumulative_probs, top_p)
        cutoff = sorted_probs[cutoff_index]
        probs[probs < cutoff] = 0

        if top_p > 0:
            idx = torch.where(probs == cutoff)[0]
            if len(idx) > 0:
                probs[idx] = cutoff + (top_p - torch.sum(probs).item()) / len(idx)
                # assert abs(torch.sum(probs).item() - top_p) < 1e-6
    
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)

    return torch.multinomial(probs, num_samples=1).item()
        
########################################################################################################
# RWKV-8 ROSA-4bit
########################################################################################################

def rosa_slow_ref(q, k, v):
    n = len(q)
    idx = [0] * n
    ln = [0] * n
    for i in range(n):
        found = False
        for w in range(i+1, 0, -1):
            t = q[i+1-w : i+1]
            for j in range(i-w, -1, -1):
                if k[j : j+w] == t:
                    s = j + w
                    idx[i] = v[s]
                    ln[i] = w
                    found = True
                    break
            if found:
                break
    return idx, ln

class rosa_slow_4bit_layer(nn.Module): # !!! matched 1 => e, matched 0 => -e, unmatched => 0 !!!
    def __init__(self, C):
        super().__init__()
        self.emb = nn.Parameter(torch.full((1,1,C), 1.0))
    def forward(self, q, k, v):
        B, T, C = q.shape
        bits = 4
        e = self.emb
        assert C % bits == 0
        G = C // bits
        qb = (q > 0).to(torch.uint8).cpu()
        kb = (k > 0).to(torch.uint8).cpu()
        vb = (v > 0).to(torch.uint8).cpu()
        ee = e.detach().cpu()
        out = torch.zeros((B, T, C), dtype=q.dtype)
        for b in range(B):
            for g in range(G):
                qsym = [0] * T
                ksym = [0] * T
                vsym = [0] * T
                for bb in range(bits):
                    ch = g * bits + bb
                    qsym = [qsym[t] | (int(qb[b, t, ch]) << bb) for t in range(T)]
                    ksym = [ksym[t] | (int(kb[b, t, ch]) << bb) for t in range(T)]
                    vsym = [vsym[t] | (int(vb[b, t, ch]) << bb) for t in range(T)]
                idx, ln = rosa_slow_ref(qsym, ksym, vsym)
                for t in range(T):
                    if ln[t] > 0:
                        sym = idx[t]
                        for bb in range(bits):
                            ch = g * bits + bb
                            bit = (sym >> bb) & 1
                            sign = 1.0 if bit == 1 else -1.0
                            out[b, t, ch] = sign * ee[0, 0, ch].item()
                    else:
                        for bb in range(bits):
                            ch = g * bits + bb
                            out[b, t, ch] = 0.0
        return out.to(q.device)

class RWKV_ROSA_4bit(nn.Module):
    def __init__(s,C):
        super().__init__()
        s.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        s.x_q = nn.Parameter(torch.zeros(1, 1, C))
        s.x_k = nn.Parameter(torch.zeros(1, 1, C))
        s.x_v = nn.Parameter(torch.zeros(1, 1, C))
        s.q=nn.Linear(C,C)
        s.k=nn.Linear(C,C)
        s.v=nn.Linear(C,C)
        s.rosa_qkv=rosa_slow_4bit_layer(C) # !!! matched 1 => e, matched 0 => -e, unmatched => 0 !!!
        s.o=nn.Linear(C,C)
    def forward(s,x):
        xx = s.time_shift(x) - x
        q = x + xx * s.x_q
        k = x + xx * s.x_k
        v = x + xx * s.x_v
        y = s.rosa_qkv(s.q(q), s.k(k), s.v(v))
        return s.o(y)

########################################################################################################
# RWKV ChannelMix
########################################################################################################

class RWKV_CMix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            self.x_k = nn.Parameter(torch.empty(1, 1, args.n_embd))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

########################################################################################################
# RWKV Block
########################################################################################################

class Block(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln0 = nn.LayerNorm(args.n_embd) # only used in block 0, should be fused with emb
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.ln3 = nn.LayerNorm(args.n_embd)

        self.rosa = RWKV_ROSA_4bit(args.n_embd)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
    @MyFunction
    def forward(self, x, v_first):

        if self.layer_id == 0:
            x = self.ln0(x)

        x = x + self.rosa(self.ln3(x))
        x = x + self.ffn(self.ln2(x))

        return x, v_first

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

        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)

        x = self.ln_out(x)
        x = self.head(x)

        return x

########################################################################################################
# RWKV Inference
########################################################################################################

model_params = torch.load(args.MODEL_NAME, map_location="cpu")

with torch.no_grad():

    model = RWKV(args).to(dtype=DTYPE).cuda()
    model.load_state_dict(model_params, strict=False)

    ########################################################################################################

    prompt = "The apple can be"
    input = tokenizer.encode(prompt)
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

    print('\n\nNow testing STUPIDLY SLOW (recompute everything including FFN of full context for every step) decoding, as I am too busy to write correct code...\n\n---\n')

    prompt = "When"
    # prompt = "The"
    # prompt = "I"
    print(prompt, end='', flush=True)
    input = tokenizer.encode(prompt)

    for i in range(100):
        out = model.forward(torch.tensor(input).reshape(1,-1).cuda())
        out = out[0, -1]
        token_id = sample_logits(out, temperature=1.0, top_p=0.5, top_k=0)
        if token_id == 0:
            break
        try:
            print(tokenizer.decode([token_id]), end='', flush=True)
        except:
            print(repr(tokenizer.decode([token_id])), end='', flush=True)
        input += [token_id]
