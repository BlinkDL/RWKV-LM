########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import torch, types, os, gc, math, json
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
np.set_printoptions(precision=4, suppress=True, linewidth=200)

'''
This will load RWKV-7 "Goose" x070.rc4a-2411 and inference in GPT-mode (slower than RNN-mode for autoregressive generation)
'''

args = types.SimpleNamespace()

# model download: https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main
MODEL_PATH = "/mnt/e/rwkv-x070-rc4a-172m-pile-20241120-ctx4k.pth"
args.n_layer = 12
args.ctx_len = 4096
args.n_embd = 768

args.vocab_size = 50304 # "pile" model: 50277 padded to 50304   
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("../RWKV-v4neo/20B_tokenizer.json")

# DTYPE = torch.bfloat16
DTYPE = torch.half
RESCALE_LAYER = -1

########################################################################################################
# CUDA Kernel
########################################################################################################

args.head_size_a = 64 # don't change
args.head_size_divisor = 8 # don't change

from torch.utils.cpp_extension import load
T = args.ctx_len
HEAD_SIZE = args.head_size_a

load(name="wkv7", sources=["cuda/wkv7_op.cpp", f"cuda/wkv7.cu"], is_python_module=False,
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={T}"])
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

def RUN_CUDA_RWKV7(r, w, k, v, a, b):
    return WKV_7.apply(r, w, k, v, a, b)

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

        with torch.no_grad():
            ddd = torch.empty(1, 1, args.n_embd)
            self.time_maa_r = nn.Parameter(ddd)
            self.time_maa_w = nn.Parameter(ddd)
            self.time_maa_k = nn.Parameter(ddd)
            self.time_maa_v = nn.Parameter(ddd)
            self.time_maa_a = nn.Parameter(ddd)
            self.time_maa_g = nn.Parameter(ddd)

            self.time_decay = nn.Parameter(torch.empty(1,1,args.dim_att))
            self.time_faaaa = nn.Parameter(torch.empty(self.n_head,self.head_size))
            self.time_aaaaa = nn.Parameter(torch.empty(1,1,args.dim_att))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.empty(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.empty(D_DECAY_LORA, args.dim_att))

            D_AAA_LORA = 64
            self.time_aaa_w1 = nn.Parameter(torch.empty(args.n_embd, D_AAA_LORA))
            self.time_aaa_w2 = nn.Parameter(torch.empty(D_AAA_LORA, args.dim_att))

            D_GATE_LORA = 128
            self.gate_w1 = nn.Parameter(torch.empty(args.n_embd, D_GATE_LORA))
            self.gate_w2 = nn.Parameter(torch.empty(D_GATE_LORA, args.dim_att))

            if layer_id > 0:
                D_MV_LORA = 32
                self.mv_w1 = nn.Parameter(torch.empty(args.n_embd, D_MV_LORA))
                self.mv_w2 = nn.Parameter(torch.empty(D_MV_LORA, args.dim_att))
                self.time_misc_v = nn.Parameter(torch.empty(1,1,args.n_embd))

            self.time_misc_kkk = nn.Parameter(torch.empty(1,1,args.n_embd))
            self.time_misc_a = nn.Parameter(torch.empty(1,1,args.n_embd))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
            self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

    def forward(self, x, v0):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x
        xr = x + xx * self.time_maa_r
        xw = x + xx * self.time_maa_w
        xk = x + xx * self.time_maa_k
        xv = x + xx * self.time_maa_v
        xa = x + xx * self.time_maa_a
        xg = x + xx * self.time_maa_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v0 = v
        else:
            v = v + (v0 - v) * torch.sigmoid(self.time_misc_v + (xv @ self.mv_w1) @ self.mv_w2)
        a = torch.sigmoid(self.time_aaaaa + (xa @ self.time_aaa_w1) @ self.time_aaa_w2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.gate_w1) @ self.gate_w2

        kk = k * self.time_misc_kkk
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.time_misc_a)

        x = RUN_CUDA_RWKV7(r, w, k, v, -kk, kk*a)

        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.time_faaaa).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)

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
            self.time_maa_k = nn.Parameter(torch.empty(1, 1, args.n_embd))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.time_maa_k
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

        # if RESCALE_LAYER > 0:
        #     if (self.layer_id+1) % RESCALE_LAYER == 0:
        #         x = x / 2
        # if self.layer_id == args.n_layer-1:
        #     print(torch.min(x).item(), torch.max(x).item())

        return x, v0

########################################################################################################
# RWKV Model
########################################################################################################

class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.dim_att = args.n_embd
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)

        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

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
keys = list(model_params.keys())
for k in keys:
    layer_id = int(k.split('.')[1]) if ('blocks.' in k) else 0
    
    if '.time_faaaa' in k: model_params[k] = model_params[k].reshape(-1, args.head_size_a)
    
    # if RESCALE_LAYER > 0:
    #     if 'att.output.weight' in k:
    #         model_params[k] = model_params[k] / (2 ** int(layer_id // RESCALE_LAYER))
    #     if 'ffn.value.weight' in k:
    #         model_params[k] = model_params[k] / (2 ** int(layer_id // RESCALE_LAYER))

with torch.no_grad():

    model = RWKV(args).to(dtype=DTYPE).cuda()
    model.load_state_dict(model_params)

    ########################################################################################################

    prompt = "The Eiffel tower is in the city of"
    input = tokenizer.encode(prompt).ids
    print(f'\nInput:\n{input}')

    out = model.forward(torch.tensor(input).reshape(1,-1).cuda())
    print(f'\nOutput:\n{out}')

    # let's check the logits for the last token => prediction for the next token    
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
