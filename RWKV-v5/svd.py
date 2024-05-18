#!/usr/bin/env python3

########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

# orig name: RWKV_v5_demo.py, inference code
# xzl: take a pretrained model, svd decompose, and save to *.pth

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
# import torch.nn as nn
from torch.nn import functional as F
# import matplotlib.pyplot as plt
# import seaborn as sns

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method

########################################################################################################
# tokenizer = RWKV_TOKENIZER("/home/wonkyoc/git/ChatRWKV/tokenizer/rwkv_vocab_v20230424.txt")

# THIS IS NOW UPDATED TO SUPPORT LATEST RWKV-5 WORLD v2 MODELS

args = types.SimpleNamespace()
args.MODEL_NAME = '/bigtemp/xl6yq/RWKV-5-World-0.4B-v2-20231113-ctx4096'
args.n_layer = 24   # xzl: so we cannot figure out automatically???
args.n_embd = 1024

args.vocab_size = 65536

# control the rank of decomposed matrices
# args.svdfac = 8
args.svdfac = 4 
# args.svdfac = 1

args.convert = 1

def plot_weight(worigin, wapprox, k, r):
    """
    worigin: the original weight
    wapprox: the approximated weight
    k: the name of weight
    r: rank
    """
    fig, axes = plt.subplots(1,2)
    fig.suptitle(f"{k}")

    sns.heatmap(worigin, ax=axes[0], vmax=1, vmin=-1)
    axes[0].set_title(f"original weight")
    sns.heatmap(wapprox, ax=axes[1], vmax=1, vmin=-1)
    axes[1].set_title(f"rank={r} weight")

    plt.savefig(f"{LOG_PATH}/{k}-{r}.png")
    plt.close()

def plot_sigma(sigma, k):
    """
    sigma: an importance matrix or a sigma matrix from USV^T
    k: the name of weight
    """


    fig, axes = plt.subplots(1,2)
    fig.suptitle(f"{k}")
    axes[0].semilogy(sigma)
    axes[0].set_title(f"Singular values")
    axes[1].plot(np.cumsum(sigma)/np.sum(sigma))
    axes[1].set_title(f"Singular values: cumulative values")
    plt.savefig(f"{LOG_PATH}/{k}-sigma.png")
    plt.close()

# xzl: this returns a rank 
def build_ranks(sigma):
    """
    sigma: the eigenvalues
    """
    threshold = 0.99
    cum_sigma = np.cumsum(sigma)/np.sum(sigma)
    for i, s in enumerate(cum_sigma):
        if s >= threshold:
            return i

class RWKV_RNN_svd(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.keys = [".att.receptance.", ".att.key.", ".att.value.", ".att.gate."]

    def full_to_svd(self, w):
        # xzl: below just for inference???
        for k in w.keys():
            print(k)  #  also print para names on the way...
            w[k] = w[k].float() # convert to f32 type       # xzl: convert all params ?
        '''
            if      '.time_' in k: w[k] = w[k].squeeze()    
            if '.time_decay' in k: w[k] = torch.exp(-torch.exp(w[k])).unsqueeze(-1)
            if '.time_faaaa' in k: w[k] = w[k].unsqueeze(-1)
        '''

        self.n_head = w['blocks.0.att.time_decay'].shape[0]
        self.head_size = w['blocks.0.ln1.weight'].shape[0] // self.n_head
        self.rank = args.n_embd // args.svdfac 

        neweight = w.copy()

        total = []
        for k, v in w.items():
            for kk in self.keys: 
                if kk in k:                     
                    # SVD https://youtu.be/H7qMMudo3e8?si=-LfBKhF0SXZdALrv
                    U, S, VT = np.linalg.svd(w[k], full_matrices=False)
                    print(f"U {U.shape}")
                    print(f"S {S.shape}")
                    print(f"VT {VT.shape}")
                    # r = build_ranks(S)      # dynamic rank 
                    r = self.rank   # fixed rank 
                    S = np.diag(S)
                    #plot_sigma(S, k)
                    print(f"{k} rank={r}")
                    total.append(r)
                    w_approx = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
                    print(f"U {U[:,:r].shape}")
                    print(f"S {S[0:r, :r].shape}")
                    print(f"VT {VT[:r, :].shape}")

                    # decompose to u1, u2 (factorize s) 
                    U = torch.tensor(U)
                    S = torch.tensor(S)
                    VT = torch.tensor(VT)
                    U1 = torch.sqrt(S[0:r, :r]) @ VT[:r, :]
                    U2 = U[:, :r] @ torch.sqrt(S[0:r, :r])
                    w_app2 = U2 @ U1

                    #plot_weight(v.numpy(), w_approx, k, r)
                    nn = torch.linalg.matrix_norm(torch.tensor(w_approx)-w[k])
                    nn2 = torch.linalg.matrix_norm(w_app2-w[k])
                    print(f"norm(diff) is {nn.item()} {nn2.item()}")

                    # overwrite the orig weights.... 
                    # w[k] = torch.from_numpy(w_approx).to(torch.float32)

                    # save decomposed .... 
                    del neweight[k]
                    k1 = k.replace(".weight", "1.weight")
                    k2 = k.replace(".weight", "2.weight")
                    neweight[k1] = U1.to(torch.float32)
                    neweight[k2] = U2.to(torch.float32)
                    # breakpoint()                    
                    continue
        print(f"avg rank = {sum(total)/len(total)}")
        return neweight

        '''
        self.w = types.SimpleNamespace() # set self.w from w
        self.w.blocks = {}
        for k in w.keys(): # example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
            parts = k.split('.')
            last = parts.pop()
            here = self.w
            for p in parts:
                if p.isdigit():
                    p = int(p)
                    if p not in here: here[p] = types.SimpleNamespace()
                    here = here[p]
                else:
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k])
        '''

    # load svd and recover full matrices
    # return the model as dict
    def svd_to_full(self, w):
        for k in w.keys():
            print(k)  #  also print para names on the way...
            w[k] = w[k].float() # convert to f32 type       

        self.n_head = w['blocks.0.att.time_decay'].shape[0]
        self.head_size = w['blocks.0.ln1.weight'].shape[0] // self.n_head
        self.rank = args.n_embd // args.svdfac 

        shortkeys = ["receptance", "key", "value", "gate"]
        for i in range(args.n_layer):
            for kk in shortkeys:                
                n0 = f"blocks.{i}.att.{kk}.weight"
                n1 = f"blocks.{i}.att.{kk}1.weight"
                n2 = f"blocks.{i}.att.{kk}2.weight"
                w1 = w[n1]
                w2 = w[n2]
                w0 = w2 @ w1
                del w[n1]
                del w[n2]
                w[n0] = w0
        
        return w

    def compare(self, w0, w1): 
        for k0, v0 in w0.items():
            v1 = w1[k0]
            assert(v0.shape == v1.shape)
            #plot_weight(v.numpy(), w_approx, k, r)
            nn = torch.norm(v0-v1)
            print(f"{k0} norm(diff) {nn.item()}")

print(f'\nUsing CPU. Loading {args.MODEL_NAME} ...')
m = RWKV_RNN_svd(args)

print("to load orig model...")
w0 = torch.load(args.MODEL_NAME + '.pth', map_location='cpu') # xzl: load model...
print("model loaded")

if args.convert: 
    print("conver to svd model...")
    w1 = m.full_to_svd(w0)
    torch.save(w1,
                f"{args.MODEL_NAME}-svd-F{args.svdfac}.pth",
                )
    print("saved")

# load back 
print("to load svd model...")
w2 = torch.load(
    f"{args.MODEL_NAME}-svd-F{args.svdfac}.pth",
    map_location='cpu'
    ) # xzl: load model...
print("model loaded")

print("recover full model & cmop...")
w00 = m.svd_to_full(w2) 
m.compare(w0, w00)