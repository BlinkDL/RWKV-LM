import torch
from torch import nn

###################################################################################################

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

class ROSA_1bit(torch.autograd.Function): # !!! extremely slow !!!
    @staticmethod
    def forward(ctx, x, emb0, emb1, tau: float):
        B,T,C = x.shape
        bits = (x>0).to(torch.long)
        idx = torch.empty_like(bits)
        for b in range(B):
            for c in range(C):
                idx[b,:,c] = torch.tensor(rosa(bits[b,:,c].tolist()), dtype=torch.long, device=x.device)
        e0 = emb0.expand_as(x); e1 = emb1.expand_as(x)
        out = torch.where(idx.eq(-1), torch.zeros_like(x), torch.where(idx.eq(1), e1, e0))
        ctx.save_for_backward(bits, idx, x, emb0, emb1)
        ctx.tau = float(tau)
        return out
    @staticmethod
    def backward(ctx, gy):
        bits, idx, x, emb0, emb1 = ctx.saved_tensors
        tau = ctx.tau
        B,T,C = x.shape
        mask0 = idx.eq(0).to(gy.dtype)
        mask1 = idx.eq(1).to(gy.dtype)
        g_emb0 = (gy * mask0).sum(dim=(0,1), keepdim=True)
        g_emb1 = (gy * mask1).sum(dim=(0,1), keepdim=True)
        gx = torch.zeros_like(x)
        e0v = emb0.view(-1)
        e1v = emb1.view(-1)
        for b in range(B):
            for c in range(C):
                row_bits = bits[b,:,c].tolist()
                base_idx = idx[b,:,c].tolist()
                vrow = gy[b,:,c].detach().cpu().tolist()
                e0c = float(e0v[c]); e1c = float(e1v[c])                
                base_phi = 0.0 # base phi for reuse
                for t in range(T):
                    it = base_idx[t]
                    if it==1: base_phi += vrow[t]*e1c
                    elif it==0: base_phi += vrow[t]*e0c                
                def phi_from_idx(idx_list): # helper to score an idx
                    s = 0.0
                    for t in range(T):
                        it = idx_list[t]
                        if it==1: s += vrow[t]*e1c
                        elif it==0: s += vrow[t]*e0c
                    return s                
                for t in range(T): # get gradient by flipping this bit
                    mag = max(abs(float(x[b,t,c].item())), tau)
                    if row_bits[t]==1:
                        phi_pos = base_phi
                    else:
                        seq = list(row_bits); seq[t]=1
                        phi_pos = phi_from_idx(rosa(seq))
                    if row_bits[t]==0:
                        phi_neg = base_phi
                    else:
                        seq = list(row_bits); seq[t]=0
                        phi_neg = phi_from_idx(rosa(seq))
                    gx[b,t,c] = (phi_pos - phi_neg) / (2.0*mag)
        return gx, g_emb0, g_emb1, None

class ROSA_1bit_LAYER(nn.Module): # !!! extremely slow !!!
    def __init__(self, C: int, tau: float = 1e-3):
        super().__init__()
        self.emb0 = nn.Parameter(torch.full((1,1,C), -1e-5)) # init
        self.emb1 = nn.Parameter(torch.full((1,1,C),  1e-5)) # init
        self.tau = tau
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ROSA_1bit.apply(x, self.emb0, self.emb1, self.tau)

###################################################################################################

B,T,C = 1,5,3
rosa_1bit = ROSA_1bit_LAYER(C, 1e-3).cuda()
x = torch.tensor(
    [[[-0.1, 2.0, 0.0],
        [ 0.4,-4.2,-1.5],
        [ 1.1, 1.2, 2.5],
        [-3.1,-2.2, 1.5],
        [ 2.1,-3.2,-2.5]]],
    dtype=torch.float32, requires_grad=True, device="cuda"
)
out  = rosa_1bit(x)
print("out:\n", out)
loss = out.mean(); loss.backward()
print("x.grad:\n", x.grad)
print("emb0.grad:\n", rosa_1bit.emb0.grad)
print("emb1.grad:\n", rosa_1bit.emb1.grad)
