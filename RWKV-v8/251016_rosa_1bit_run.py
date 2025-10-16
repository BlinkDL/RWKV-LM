import torch, random
from torch import nn
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True; torch.set_float32_matmul_precision('high')
device='cuda'

############################################################################################################################################

def rosa(x):
	n=len(x); y=[-1]*n; s=2*n+1; b=[None]*s; c=[-1]*s; d=[0]*s; e=[-1]*s; b[0]={}; g=0; z=1
	for i,t in enumerate(x):
		r=z; z+=1; b[r]={}; d[r]=d[g]+1; p=g
		while p!=-1 and t not in b[p]: b[p][t]=r; p=c[p]
		if p==-1: c[r]=0
		else:
			q=b[p][t]
			if d[p]+1==d[q]: c[r]=q
			else:
				u=z; z+=1; b[u]=b[q].copy(); d[u]=d[p]+1; c[u]=c[q]; e[u]=e[q]
				while p!=-1 and b[p][t]==q: b[p][t]=u; p=c[p]
				c[q]=c[r]=u
		v=g=r; a=-1
		while v!=-1:
			if d[v]>0 and e[v]>=0: a=x[e[v]+1]; break
			v=c[v]
		y[i]=a; v=g
		while v!=-1 and e[v]<i: e[v]=i; v=c[v]
	return y

def rosa_torch(z: torch.Tensor) -> torch.Tensor:
    assert z.dtype==torch.long and z.ndim==2
    zc = z.detach().contiguous().cpu()
    return torch.stack([torch.as_tensor(rosa(r.tolist()), dtype=torch.long) for r in zc]).to(z.device)

class Emb_ROSA(nn.Module):
    def __init__(s,V,C):
        super().__init__()
        s.emb = nn.Embedding(V,C)
    def forward(s,idx):
        idx = rosa_torch(idx)
        out = s.emb(idx.clamp_min(0))
        return out.masked_fill(idx.eq(-1).unsqueeze(-1), 0.0)

############################################################################################################################################

class ROSA_1bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, emb0, emb1, tau: float):
        B,T,C = x.shape
        bits = (x>0).to(torch.long)

        bbb = bits.squeeze().transpose(0,1)
        for c in range(C):
            print('lang ' + ''.join([str(x) for x in bbb[c].tolist()]))
        print()

        idx = torch.empty_like(bits)
        for b in range(B):
            for c in range(C):
                idx[b,:,c] = torch.tensor(rosa(bits[b,:,c].tolist()), dtype=torch.long, device=x.device)
        e0 = emb0.expand_as(x); e1 = emb1.expand_as(x)
        out = torch.where(idx.eq(-1), torch.zeros_like(x), torch.where(idx.eq(1), e1, e0))
        ctx.save_for_backward(bits, idx, x, emb0, emb1)
        ctx.tau = float(tau)
        return out

class ROSA_1bit_LAYER(nn.Module):
    def __init__(self, C: int, tau: float = 1e-3):
        super().__init__()
        self.emb0 = nn.Parameter(torch.full((1,1,C), -1e-5)) # init
        self.emb1 = nn.Parameter(torch.full((1,1,C),  1e-5)) # init
        self.tau = tau
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ROSA_1bit.apply(x, self.emb0, self.emb1, self.tau)

############################################################################################################################################

V,C,B,T,steps=11,32,128,128,1000
lr0,lr1=1e-3,1e-6

print('Run EmbROSA + ROSA 1bit x 2layer')

def batch(B,T,nn=None):
    s=[]
    for _ in range(B):
        if nn == None:
            k=random.randint(1,3); lo=0 if k==1 else 10**(k-1); n=random.randint(lo,10**k-1)
        else:
            assert B == 1
            n = nn
        a=[10]
        while len(a)<T:
            a+=[ord(c)-48 for c in str(n)]+[10]; n+=1
        s.append(a[:T])
    return torch.tensor(s,device=device,dtype=torch.long)

class MODEL(nn.Module):
    def __init__(s):
        super().__init__()
        s.e=nn.Embedding(V,C)
        s.emb_rosa=Emb_ROSA(V,C)
        s.rosa1=ROSA_1bit_LAYER(C)
        s.lin=nn.Linear(C,C)
        s.rosa2=ROSA_1bit_LAYER(C)
        s.o=nn.Linear(C,V)
    def forward(s,x):
        x = s.e(x) + s.emb_rosa(x)
        x = x + s.rosa1(x)
        x = x + s.lin(x)
        x = x + s.rosa2(x)
        x = s.o(x)
        return x

model=MODEL().to(device)
model.load_state_dict(torch.load('251016_rosa_1bit_run.pth', map_location=device, mmap=True, weights_only=True))
model=torch.compile(model)
print('#'*100)

with torch.no_grad():
    S='0123456789A'

    for SAMPLE in range(5):
        x=batch(1,128,int(3.5**(SAMPLE+1))); y=x[:,1:]; z=model(x[:,:-1]).argmax(-1); n=y.numel()
        r=rosa_torch(x)[:,:-1]; rr=''.join([S[x] if x >= 0 else 'X' for x in r[0].tolist()])

        xx=''.join(S[t] for t in x[0,:-1].tolist())
        yy=''.join(S[t] for t in y[0].tolist())
        zz=''.join(S[t] for t in z[0].tolist())
        ry=''.join('.' if r[0,i].item()==y[0,i].item() else '^' for i in range(y.size(1)))
        zy=''.join('.' if z[0,i].item()==y[0,i].item() else '^' for i in range(y.size(1)))
        nry=(r==y).sum().item()
        nzy=(z==y).sum().item()
        print('in  ',xx)
        print('gold',yy)
        print('rosa',rr)
        print('diff',ry)
        print(f'correct {nry}/{n}  acc {nry/n:.3f}')
        print('gold',yy)
        print('pred',zz)
        print('diff',zy)
        print(f'correct {nzy}/{n}  acc {nzy/n:.3f}')
        print('#'*100)
