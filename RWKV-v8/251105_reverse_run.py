import random, torch, math
from types import SimpleNamespace
import torch, random
from torch import nn
import torch.nn.functional as F
device='cuda'
import random
import numpy as np
def set_seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
set_seed_all(42)

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method

LOAD_NAME = '251105_reverse_L2' # acc = 395782/396500=99.8% for 1-60 digits, only 39.6K params

weights=torch.load(f'{LOAD_NAME}.pth', map_location=device, mmap=True, weights_only=True)
nparams = 0
for k in weights:
    nparams += weights[k].numel()
print('parameter count', nparams)

V,C,T=12,32,129
DIGIT_MAX=60

############################################################################################################################################

def samx_qkv_slow(qqq, kkk, vvv): # slow, only for reference
    n=len(qqq); y=[-1]*n; s=2*n+1; t=[None]*s; f=[-1]*s; m=[0]*s; r=[-1]*s; t[0]={}; g=0; u=1; w=h=0; assert n==len(kkk)==len(vvv)
    for i,(q,k) in enumerate(zip(qqq,kkk)):
        p,x=w,h
        while p!=-1 and q not in t[p]: x=m[p] if x>m[p] else x; p=f[p]
        p,x=(t[p][q],x+1) if p!=-1 else (0,0); v=p
        while f[v]!=-1 and m[f[v]]>=x: v=f[v]
        while v!=-1 and (m[v]<=0 or r[v]<0): v=f[v]
        y[i]=vvv[r[v]+1] if v!=-1 else -1; w,h=p,x; j=u; u+=1; t[j]={}; m[j]=m[g]+1; p=g
        while p!=-1 and k not in t[p]: t[p][k]=j; p=f[p]
        if p==-1: f[j]=0
        else:
            d=t[p][k]
            if m[p]+1==m[d]: f[j]=d
            else:
                b=u; u+=1; t[b]=t[d].copy(); m[b]=m[p]+1; f[b]=f[d]; r[b]=r[d]; f[d]=f[j]=b
                while p!=-1 and t[p][k]==d: t[p][k]=b; p=f[p]
        v=g=j
        while v!=-1 and r[v]<i: r[v]=i; v=f[v]
    return [max(0,y) for y in y] # use "0" for both "no-match" and matched "0"

def samx_qkv_batch_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    assert q.dtype == k.dtype == v.dtype == torch.uint8
    assert q.ndim == k.ndim == v.ndim == 2
    assert q.shape == k.shape == v.shape
    qc = q.detach().contiguous().cpu()
    kc = k.detach().contiguous().cpu()
    vc = v.detach().contiguous().cpu()
    return torch.stack([
        torch.as_tensor(samx_qkv_slow(qq.tolist(), kk.tolist(), vv.tolist()), dtype=q.dtype)
        for qq, kk, vv in zip(qc, kc, vc)
    ]).to(q.device)

class samx_qkv_1bit_layer_op(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e):
        b,t,c = q.shape
        qb = (q>0).to(torch.uint8).transpose(1,2).reshape(-1,t).contiguous()
        kb = (k>0).to(torch.uint8).transpose(1,2).reshape(-1,t).contiguous()
        vb = (v>0).to(torch.uint8).transpose(1,2).reshape(-1,t).contiguous()
        idx = samx_qkv_batch_ref(qb, kb, vb).view(b,c,t).transpose(1,2).contiguous()
        out = (2.0 * idx.to(q.dtype) - 1.0) * e
        return out
class samx_qkv_1bit_layer(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.emb = nn.Parameter(torch.full((1,1,C), 1.0))
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return samx_qkv_1bit_layer_op.apply(q, k, v, self.emb)

class ROSA_QKV_B_1bit(nn.Module):
    def __init__(s,C):
        super().__init__()
        s.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        s.x_q = nn.Parameter(torch.zeros(1, 1, C))
        s.x_k = nn.Parameter(torch.zeros(1, 1, C))
        s.x_v = nn.Parameter(torch.zeros(1, 1, C))
        s.q=nn.Linear(C,C)
        s.k=nn.Linear(C,C)
        s.v=nn.Linear(C,C)
        s.rosa_qkv=samx_qkv_1bit_layer(C)
        s.o=nn.Linear(C,C)
    def forward(s,x):
        xx = s.time_shift(x) - x
        q = x + xx * s.x_q
        k = x + xx * s.x_k
        v = x + xx * s.x_v
        y = s.rosa_qkv(s.q(q), s.k(k), s.v(v))
        return s.o(y)

############################################################################################################################################

from torch.utils.cpp_extension import load
HEAD_SIZE = 16
CHUNK_LEN = 16
flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=False, extra_cuda_cflags=flags)
class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b):
        B,T,H,C = w.shape
        assert T%CHUNK_LEN == 0 # if T%CHUNK_LEN != 0: pad your input to T%CHUNK_LEN == 0, or change CHUNK_LEN (will be slower)
        assert all(i.dtype==torch.float32 for i in [w,q,k,v,z,b])
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.float32 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
        return dw,dq,dk,dv,dz,db
def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
    B,T,HC = q.shape
    q,w,k,v,a,b = [i.view(B,T,HC//16,16) for i in [q,w,k,v,a,b]]
    return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)

class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C
            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x
            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C-1) - 0.5
                zigzag[n] = ((n % N) - ((N-1) / 2)) / ((N-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            D_DECAY_LORA = 8
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1,1,C) + 0.5 + zigzag*2.5)
            D_AAA_LORA = 8
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C)-0.19 + zigzag*0.3 + linear*0.4)
            D_MV_LORA = 8
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4)
            D_GATE_LORA = 8
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1,1,C)+0.71 - linear*0.1)
            self.k_a = nn.Parameter(torch.zeros(1,1,C)+1.02)
            self.r_k = nn.Parameter(torch.zeros(H,N)-0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first

############################################################################################################################################

class FFN(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.zeros(1, 1, C))
        self.key = nn.Linear(C, C * 4, bias=False)
        self.value = nn.Linear(C * 4, C, bias=False)
        with torch.no_grad():
            self.value.weight.data.zero_()
            nn.init.orthogonal_(self.key.weight.data, gain=(4**0.5))
    def forward(self, x):
        xx = self.time_shift(x) - x
        x = x + xx * self.x_k
        x = torch.relu(self.key(x)) ** 2
        return self.value(x)

if '_L4' in LOAD_NAME:
    class MODEL(nn.Module):
        def __init__(s):
            super().__init__()
            args = SimpleNamespace()
            args.n_head = C//HEAD_SIZE
            args.head_size = HEAD_SIZE
            args.n_embd = C
            args.dim_att = C
            args.n_layer = 2

            s.e=nn.Embedding(V,C)

            s.ln1a=nn.LayerNorm(C)
            s.ln1b=nn.LayerNorm(C)
            s.ln1c=nn.LayerNorm(C)
            s.rwkv1=RWKV_Tmix_x070(args,0)
            s.rosa1=ROSA_QKV_B_1bit(C)
            s.ffn1=FFN(C)

            s.ln2a=nn.LayerNorm(C)
            s.ln2b=nn.LayerNorm(C)
            s.ln2c=nn.LayerNorm(C)
            s.rwkv2=RWKV_Tmix_x070(args,1)
            s.rosa2=ROSA_QKV_B_1bit(C)
            s.ffn2=FFN(C)

            s.ln3a=nn.LayerNorm(C)
            s.ln3b=nn.LayerNorm(C)
            s.ln3c=nn.LayerNorm(C)
            s.rwkv3=RWKV_Tmix_x070(args,1)
            s.rosa3=ROSA_QKV_B_1bit(C)
            s.ffn3=FFN(C)

            s.ln4a=nn.LayerNorm(C)
            s.ln4b=nn.LayerNorm(C)
            s.ln4c=nn.LayerNorm(C)
            s.rwkv4=RWKV_Tmix_x070(args,1)
            s.rosa4=ROSA_QKV_B_1bit(C)
            s.ffn4=FFN(C)

            s.lno=nn.LayerNorm(C)
            s.o=nn.Linear(C,V)

        def forward(s,x):
            x = s.e(x)

            xr = s.rosa1(s.ln1c(x))
            xx, v_first = s.rwkv1(s.ln1a(x), torch.empty_like(x))
            x = x + xx + xr
            x = x + s.ffn1(s.ln1b(x))
            xr = s.rosa2(s.ln2c(x))
            xx, v_first = s.rwkv2(s.ln2a(x), v_first)
            x = x + xx + xr
            x = x + s.ffn2(s.ln2b(x))

            xr = s.rosa3(s.ln3c(x))
            xx, v_first = s.rwkv3(s.ln3a(x), v_first)
            x = x + xx + xr
            x = x + s.ffn3(s.ln3b(x))
            xr = s.rosa4(s.ln4c(x))
            xx, v_first = s.rwkv4(s.ln4a(x), v_first)
            x = x + xx + xr
            x = x + s.ffn4(s.ln4b(x))

            x = s.o(s.lno(x))
            return x
elif '_L2' in LOAD_NAME:
    class MODEL(nn.Module):
        def __init__(s):
            super().__init__()
            args = SimpleNamespace()
            args.n_head = C//HEAD_SIZE
            args.head_size = HEAD_SIZE
            args.n_embd = C
            args.dim_att = C
            args.n_layer = 2

            s.e=nn.Embedding(V,C)

            s.ln1a=nn.LayerNorm(C)
            s.ln1b=nn.LayerNorm(C)
            s.ln1c=nn.LayerNorm(C)
            s.rwkv1=RWKV_Tmix_x070(args,0)
            s.rosa1=ROSA_QKV_B_1bit(C)
            s.ffn1=FFN(C)

            s.ln2a=nn.LayerNorm(C)
            s.ln2b=nn.LayerNorm(C)
            s.ln2c=nn.LayerNorm(C)
            s.rwkv2=RWKV_Tmix_x070(args,1)
            s.rosa2=ROSA_QKV_B_1bit(C)
            s.ffn2=FFN(C)

            s.lno=nn.LayerNorm(C)
            s.o=nn.Linear(C,V)

        def forward(s,x):
            x = s.e(x)

            xr = s.rosa1(s.ln1c(x))
            xx, v_first = s.rwkv1(s.ln1a(x), torch.empty_like(x))
            x = x + xx + xr
            x = x + s.ffn1(s.ln1b(x))
            xr = s.rosa2(s.ln2c(x))
            xx, v_first = s.rwkv2(s.ln2a(x), v_first)
            x = x + xx + xr
            x = x + s.ffn2(s.ln2b(x))

            x = s.o(s.lno(x))
            return x

model=MODEL().to(device)
model.load_state_dict(weights)
print('#'*100)

def get_randint(digits):
    lo = 0 if digits==1 else 10**(digits-1); x = random.randint(lo, 10**digits-1)
    return x

TOK = {**{str(i):i for i in range(10)}, ',':10, '#':11}
S='0123456789,#'
src = []
ngood=0; nall=0
agood=0; aall=0
for DIGIT in range(1, DIGIT_MAX+1):
    print(f'testing {DIGIT} digits')
    for ii in range(10):
        raw = str(get_randint(DIGIT))
        raw = raw + ',' + raw[::-1] 
        raw = raw + '#'*(129-len(raw))
        raw = [TOK[c] for c in list(raw)]
        src.append(raw)
    src=torch.tensor(src, device=device, dtype=torch.long)
    dst=src[:,1:]; out=model(src[:,:-1]).argmax(-1)
    for n in range(src.shape[0]):
        xx=''.join(S[t] for t in src[n,:-1].tolist())
        p1=xx.find(',')
        p2=xx.find('#')
        expect=xx[p1+1:p2+1]
        yy=''.join(S[t] for t in out[n,p1:p2].tolist())
        ngood+=(src[n,p1+1:p2+1]==out[n,p1:p2]).sum().item()
        nall+=(p2-p1)
        print('in',xx[:p2+1],'model',yy,'check',''.join(['.' if str(expect)[i]==str(yy)[i] else 'X' for i in range(len(expect))]))
    print('correct digits', ngood, 'all digits', nall)
    agood+=ngood; aall+=nall
    src = []
    if DIGIT%20 == 0:
        print('SUM: correct digits', agood, 'all digits', aall)
