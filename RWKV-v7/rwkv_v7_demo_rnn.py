########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
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

########################################################################################################

args = types.SimpleNamespace()

args.MODEL_NAME = '/mnt/e/RWKV-x070-Pile-168M-20241120-ctx4096'
# args.MODEL_NAME = "/mnt/program/RWKV-x070-Pile-421M-20241127-ctx4096"

if '168M' in args.MODEL_NAME:
    args.n_layer = 12
    args.n_embd = 768
elif '421M' in args.MODEL_NAME:
    args.n_layer = 24
    args.n_embd = 1024

args.vocab_size = 50304 # "pile" model: 50277 padded to 50304
args.head_size = 64

prompt = "The Eiffel tower is in the city of"
NUM_TRIALS = 3
LENGTH_PER_TRIAL = 100
TEMPERATURE = 1.0
TOP_P = 0.0

# DTYPE = torch.bfloat16
DTYPE = torch.half

########################################################################################################

class RWKV_RNN(MyModule):
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
            if k.endswith('att.w0'):
                z[k] = z[k].float()
            else:
                z[k] = z[k].to(dtype=DTYPE)
            z[k] = z[k].squeeze()
            if k.endswith('att.r_k'): z[k] = z[k].flatten()
        assert self.head_size == args.head_size

        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
        z['blocks.0.att.v0'] = z['blocks.0.att.a0'] # actually ignored
        z['blocks.0.att.v1'] = z['blocks.0.att.a1'] # actually ignored
        z['blocks.0.att.v2'] = z['blocks.0.att.a2'] # actually ignored

    @MyFunction
    def forward(self, token:int, state:List[torch.Tensor]):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][token]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                # here i am using a hack to determine "first layer"
                xx, state[i*3+0], state[i*3+1], v_first = time_mixing(self.n_head if i > 0 else -self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'key.weight'], z[att+'value.weight'], z[att+'receptance.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, state[i*3+2] = channel_mixing(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = z['head.weight'] @ x

            return x, state

########################################################################################################

def time_mixing__(H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, kw, vw, rw, ow, ln_w, ln_b):
    xx = x_prev - x
    xr = x + xx * x_r
    xw = x + xx * x_w
    xk = x + xx * x_k
    xv = x + xx * x_v
    xa = x + xx * x_a
    xg = x + xx * x_g

    r = rw @ xr
    w = torch.tanh(xw @ w1) @ w2
    k = kw @ xk
    v = vw @ xv

    if H < 0: # hack to determine "first layer"
        v_first = v
        H = -H
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = k * k_k
    kk = torch.nn.functional.normalize(kk.view(H,N), dim=-1, p=2.0).view(-1)
    k = k * (1 + (a-1) * k_a)

    # naive version
    # w = -torch.nn.functional.softplus(-(w0 + w.float())) - 0.5
    # assert w.dtype == torch.float
    # w = torch.exp(-torch.exp(w))

    # fused version
    w = w0 + w.float()
    assert w.dtype == torch.float
    w = torch.exp(-0.606531*torch.sigmoid(w)) # 0.606531 = exp(-0.5)
    
    vk = v.view(H,N,1) @ k.view(H,1,N)

    ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N)
    
    state = state * w.view(H,1,N) + state @ ab.float() + vk.float()
    
    out = (state.to(dtype=x.dtype) @ r.view(H,N,1)).view(H,N)

    out = torch.nn.functional.group_norm(out.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)    
    out = out + ((r*k*r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)

    return ow @ (out * g), x, state, v_first
try:
    time_mixing = torch.compile(time_mixing__, mode="max-autotune", fullgraph=True, dynamic=False)
except:
    time_mixing = torch.jit.script(time_mixing__)

########################################################################################################

def channel_mixing__(x, x_prev, x_k, kw, vw):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(kw @ k) ** 2
    return vw @ k, x
try:
    channel_mixing = torch.compile(channel_mixing__, mode="max-autotune", fullgraph=True, dynamic=False)
except:
    channel_mixing = torch.jit.script(channel_mixing__)

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

from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("../RWKV-v4neo/20B_tokenizer.json")

print(f'\nUsing CUDA bf16. Loading {args.MODEL_NAME} ...')
model = RWKV_RNN(args)

print(f'\nPreprocessing prompt (note: using RNN mode to pre-fill is very inefficient)')

init_state = [None for _ in range(args.n_layer * 3)]
for i in range(args.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
    init_state[i*3+0] = torch.zeros(args.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")
    init_state[i*3+1] = torch.zeros((args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.float, requires_grad=False, device="cuda")
    init_state[i*3+2] = torch.zeros(args.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")

for token in tokenizer.encode(prompt).ids:
    init_out, init_state = model.forward(token, init_state)
    
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

zero_state = [None for _ in range(args.n_layer * 3)]
for i in range(args.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
    zero_state[i*3+0] = torch.zeros(args.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")
    zero_state[i*3+1] = torch.zeros((args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.float, requires_grad=False, device="cuda")
    zero_state[i*3+2] = torch.zeros(args.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")

import json, math
with open(f"misc/lambada_test.jsonl", "r", encoding="utf-8") as f:
    todo = [json.loads(line) for line in f]
    todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

print('\nCheck LAMBADA... (RNN mode is very slow for this)')
xsum = 0
xcnt = 0
xacc = 0
for d in todo:
    src = [0] + tokenizer.encode(d[0]).ids
    dst = tokenizer.encode(d[1]).ids

    logits = 0
    correct = True
    
    state = copy.deepcopy(zero_state)
    for token in src:
        out, state = model.forward(token, state)

    for i in range(len(dst)):
        probs = F.softmax(out.float(), dim=-1)
        logits += math.log(probs[dst[i]])
        if torch.argmax(probs).item() != dst[i]:
            correct = False
        out, state = model.forward(dst[i], state)

    xcnt += 1
    xsum += logits
    xacc += 1 if correct else 0
    if xcnt % 10 == 0 or xcnt == len(todo):
        print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 2))
