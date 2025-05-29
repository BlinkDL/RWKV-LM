########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
#
# This version is GPT-mode + RNN-mode, and a bit more difficult to understand
#
########################################################################################################

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch, copy, time
from typing import List
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)

import torch.nn as nn
from torch.nn import functional as F

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script
# MyModule = nn.Module
# def __nop(ob): return ob
# MyFunction = __nop
# MyStatic = __nop

########################################################################################################
#
print('\nThe RWKV-8 "Heron" Language Model - https://github.com/BlinkDL/RWKV-LM')
print('\nNOTE: this is very inefficient (loads all weights to VRAM), you can actually prefetch .enn from RAM/SSD\n')
#
########################################################################################################

args = types.SimpleNamespace()

# model download: https://huggingface.co/BlinkDL/rwkv-8-pile

args.MODEL_NAME = "/mnt/g/_rwkv_checkpt/rwkv-8-pile-rc00-20250508"

# WORLD_MODE = True
WORLD_MODE = False # pile models

args.n_layer = 12
args.n_embd = 768
args.vocab_size = 65536 if WORLD_MODE else 50304
args.head_size = 64

prompt = "The Eiffel tower is in the city of"
NUM_TRIALS = 3
LENGTH_PER_TRIAL = 100
TEMPERATURE = 1.0
TOP_P = 0.0

########################################################################################################

DTYPE = torch.half

from torch.utils.cpp_extension import load
HEAD_SIZE = args.head_size

load(name="wkv7s", sources=["cuda/wkv7s_op.cpp", f"cuda/wkv7s.cu"], is_python_module=False,
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
class WKV_7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b):
        with torch.no_grad():
            T, C = r.size()
            H = C // HEAD_SIZE
            N = HEAD_SIZE
            assert HEAD_SIZE == C // H
            assert all(x.dtype == DTYPE for x in [r,w,k,v,a,b])
            assert all(x.is_contiguous() for x in [r,w,k,v,a,b])
            y = torch.empty((T, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.wkv7s.forward(1, T, C, H, state, r, w, k, v, a, b, y)
            return y
def RWKV7_OP(state, r, w, k, v, a, b):
    return WKV_7.apply(state, r, w, k, v, a, b)

########################################################################################################

class RWKV_x070(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer
        self.eval()
        
        self.z = torch.load(args.MODEL_NAME + '.pth', map_location='cuda')
        z = self.z
        self.n_head, self.head_size = z['blocks.0.att.r_k'].shape

        keys = list(z.keys())
        for k in keys:
            if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                z[k] = z[k].t()
            z[k] = z[k].squeeze().to(dtype=DTYPE)
            if k.endswith('att.r_k'): z[k] = z[k].flatten()
        assert self.head_size == args.head_size

        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
        z['blocks.0.att.v0'] = z['blocks.0.att.a0'] # actually ignored
        z['blocks.0.att.v1'] = z['blocks.0.att.a1'] # actually ignored
        z['blocks.0.att.v2'] = z['blocks.0.att.a2'] # actually ignored
        self.deepembs = {}

    def forward(self, idx, state, full_output=False):
        if state == None:
            state = [None for _ in range(args.n_layer * 3)]
            for i in range(args.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
                state[i*3+0] = torch.zeros(args.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")
                state[i*3+1] = torch.zeros((args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.float, requires_grad=False, device="cuda")
                state[i*3+2] = torch.zeros(args.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")

        if type(idx) is list:
            if len(idx) > 1:
                return self.forward_seq(idx, state, full_output)
            else:
                return self.forward_one(idx[0], state)
        else:
            return self.forward_one(idx, state)

    @MyFunction
    def forward_one(self, idx:int, state:List[torch.Tensor]):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_one(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, state[i*3+2] = RWKV_x080_CMix_one(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'], z[ffn+'enn.weight'][idx])
                x = x + xx
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x, state
        
    @MyFunction
    def forward_seq(self, idx:List[int], state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_seq(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                # Disk space offloading
                if not (ffn+'enn.weight' in self.deepembs):
                    z[ffn+'enn.weight'].numpy().tofile(ffn+'enn.weight_storage'+'.pt')
                    emb_size = z[ffn+'enn.weight'].numel()
                    emb_dtype = z[ffn+'enn.weight'].dtype
                    z[ffn+'enn.weight'] = torch.from_file(ffn+'enn.weight_storage'+'.pt', size=emb_size, dtype=emb_dtype, device='cpu')
                    self.deepembs[ffn+'enn.weight'] = True
                
                xx, state[i*3+2] = RWKV_x080_CMix_seq(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'], z[ffn+'enn.weight'][idx].cuda())
                x = x + xx
            
            if not full_output: x = x[-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x, state

########################################################################################################

@MyStatic
def RWKV_x070_TMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)

    vk = v.view(H,N,1) @ k.view(H,1,N)
    ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N)
    state = state * w.view(H,1,N) + state @ ab.float() + vk.float()
    xx = (state.to(dtype=x.dtype) @ r.view(H,N,1))

    xx = torch.nn.functional.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)    
    xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    return (xx * g) @ O_, x, state, v_first

@MyStatic
def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    T = x.shape[0]
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    ######## cuda-free method 
    # w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)
    # for t in range(T):
    #     r_, w_, k_, v_, kk_, a_ = r[t], w[t], k[t], v[t], kk[t], a[t]
    #     vk = v_.view(H,N,1) @ k_.view(H,1,N)
    #     ab = (-kk_).view(H,N,1) @ (kk_*a_).view(H,1,N)
    #     state = state * w_.view(H,1,N) + state @ ab.float() + vk.float()
    #     xx[t] = (state.to(dtype=x.dtype) @ r_.view(H,N,1)).view(H*N)

    w = -torch.nn.functional.softplus(-(w0 + w)) - 0.5
    xx = RWKV7_OP(state, r, w, k, v, -kk, kk*a)

    xx = torch.nn.functional.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
    xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
    return (xx * g) @ O_, x[-1,:], state, v_first

########################################################################################################

@MyStatic
def RWKV_x080_CMix_one(x, x_prev, x_k, K_, V_, E_):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return (k @ V_) * E_, x

@MyStatic
def RWKV_x080_CMix_seq(x, x_prev, x_k, K_, V_, E_):
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return (k @ V_) * E_, x[-1,:]

########################################################################################################
#
# The testing code
#
########################################################################################################

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
# RWKV Tokenizer (slow version)
########################################################################################################

if WORLD_MODE:
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
    tokenizer = RWKV_TOKENIZER("rwkv_vocab_v20230424.txt")
else:
    from tokenizers import Tokenizer
    class RWKV_TOKENIZER():
        def __init__(self):
            self.tokenizer = Tokenizer.from_file("../RWKV-v4neo/20B_tokenizer.json")
        def encode(self, x):
            return self.tokenizer.encode(x).ids
        def decode(self, x):
            return self.tokenizer.decode(x)
    tokenizer = RWKV_TOKENIZER()    

########################################################################################################

print(f'\nUsing CUDA {str(DTYPE).replace("torch.","")}. Loading {args.MODEL_NAME} ...')
model = RWKV_x070(args)

init_out, init_state = model.forward(tokenizer.encode(prompt), None)
    
probs = F.softmax(init_out.float(), dim=-1) # compute softmax in float (more accurate)

print(f'\n{prompt}')

_, indices = torch.topk(probs, 10) # print top-10 possibilities
for i in range(len(indices)):
    token_id = indices[i].item()
    token = tokenizer.decode([token_id])
    token_prob = probs[token_id].item()
    print(token, f'[probability {token_prob:.2%}]')

########################################################################################################

for TRIAL in range(NUM_TRIALS):
    print(f'\n\n--[ Trial {TRIAL} ]-----------------', prompt, end="")
    all_tokens = []
    out_last = 0
    out, state = init_out.clone(), copy.deepcopy(init_state)

    min_time = 1e10
    min_time_all = 1e10

    t000 = time.perf_counter()

    for i in range(LENGTH_PER_TRIAL):
        t00 = time.perf_counter()
        token = sample_logits(out, TEMPERATURE, TOP_P)
        all_tokens += [token]
        try:
            tmp = tokenizer.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
                print(tmp, end="", flush=True)
                out_last = i + 1
        except:
            pass
        t0 = time.perf_counter()

        out, state = model.forward(token, state)
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        min_time = min(min_time, t1 - t0)
        min_time_all = min(min_time_all, t1 - t00)
    
    print(f'\n[ {round(1/min_time_all,2)} (real) / {round(1/min_time,2)} (ignore sampling & tokenizer) token/s = {round(time.perf_counter()-t000,3)}s ]', end='')

print('\n')

########################################################################################################

import json, math
with open(f"misc/lambada_test.jsonl", "r", encoding="utf-8") as f:
    todo = [json.loads(line) for line in f]
    todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

print('\nCheck LAMBADA...')
xsum = 0
xcnt = 0
xacc = 0
for d in todo:
    src = [0] + tokenizer.encode(d[0])
    dst = tokenizer.encode(d[1])

    logits = 0
    correct = True
    
    out, _ = model.forward(src+dst, None, full_output=True)

    for i in range(len(dst)):
        ooo = out[len(src)-1+i].float()
        probs = F.softmax(ooo, dim=-1)
        logits += math.log(probs[dst[i]])
        if torch.argmax(probs).item() != dst[i]:
            correct = False

    xcnt += 1
    xsum += logits
    xacc += 1 if correct else 0
    if xcnt % 100 == 0 or xcnt == len(todo):
        print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 2))
