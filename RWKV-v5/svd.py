#!/usr/bin/env python3

# xzl: can do the following things: 
#       a pretrained model -->  svd decompose, and save to *.pth
#       our decmoposed model (finetuned) ---> a model in the original format, save to *pth 
#       diff the recovered model vs. the original model 

# Ex

# python3 svd.py --svdfac 8 --decompose 1
# python3 svd.py --decompose 0 


# orig name: RWKV_v5_demo.py, inference code
########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

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

# THIS IS NOW UPDATED TO SUPPORT LATEST RWKV-5 WORLD v2 MODELS

# args = types.SimpleNamespace()

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

def full_to_svd(w):
    selfkeys = [".att.receptance.", ".att.key.", ".att.value.", ".att.gate."]
    # xzl: all params saved in bfloat16 in model file
    for k in w.keys():
        # print(k)  #  also print para names on the way...
        w[k] = w[k].float() # convert to f32 type      
    '''
        if      '.time_' in k: w[k] = w[k].squeeze()    
        if '.time_decay' in k: w[k] = torch.exp(-torch.exp(w[k])).unsqueeze(-1)
        if '.time_faaaa' in k: w[k] = w[k].unsqueeze(-1)
    '''

    self_n_head = w['blocks.0.att.time_decay'].shape[0]
    self_head_size = w['blocks.0.ln1.weight'].shape[0] // self_n_head
    self_rank = args.n_embd // args.svdfac 

    neweight = w.copy()

    total = []
    for k, v in w.items():
        for kk in selfkeys: 
            if kk in k:                     
                # SVD https://youtu.be/H7qMMudo3e8?si=-LfBKhF0SXZdALrv
                U, S, VT = np.linalg.svd(w[k], full_matrices=False)
                # print(f"U {U.shape}")
                # print(f"S {S.shape}")
                # print(f"VT {VT.shape}")
                # r = build_ranks(S)      # dynamic rank 
                r = self_rank   # fixed rank 
                S = np.diag(S)
                # print(f"{k} rank={r}")
                total.append(r)
                w_approx = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
                # print(f"U {U[:,:r].shape}")
                # print(f"S {S[0:r, :r].shape}")
                # print(f"VT {VT[:r, :].shape}")

                # decompose to u1, u2 (factorize s) 
                U = torch.tensor(U)
                S = torch.tensor(S)
                VT = torch.tensor(VT)
                U1 = torch.sqrt(S[0:r, :r]) @ VT[:r, :]
                U2 = U[:, :r] @ torch.sqrt(S[0:r, :r])
                w_app2 = U2 @ U1

                nn = torch.linalg.matrix_norm(torch.tensor(w_approx)-w[k])
                nn2 = torch.linalg.matrix_norm(w_app2-w[k])
                print(f"norm(diff) is {nn.item()} {nn2.item()}")

                # save decomposed .... 
                del neweight[k]
                k1 = k.replace(".weight", "1.weight")
                k2 = k.replace(".weight", "2.weight")
                neweight[k1] = U1.to(torch.float32)
                neweight[k2] = U2.to(torch.float32)
                print(f"{k}->{k1},{k2}")
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
def svd_recover_to_full(w):
    for k in w.keys():
        # print(k)  #  also print para names on the way...
        w[k] = w[k].float() # convert to f32 type for compute

    self_n_head = w['blocks.0.att.time_decay'].shape[0]
    self_head_size = w['blocks.0.ln1.weight'].shape[0] // self_n_head
    self_rank = args.n_embd // args.svdfac 

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
            print(f"{n1},{n2} -> {n0}")

    #  if ffn decomposed, recover 
    if 'blocks.0.ffn.key1.weight' in w:
        shortkeys = ["key"]
        for i in range(args.n_layer):
            for kk in shortkeys:                
                n0 = f"blocks.{i}.ffn.{kk}.weight"
                n1 = f"blocks.{i}.ffn.{kk}1.weight"
                n2 = f"blocks.{i}.ffn.{kk}2.weight"
                w1 = w[n1]
                w2 = w[n2]
                w0 = w2 @ w1
                del w[n1]
                del w[n2]
                w[n0] = w0 
                print(f"{n1},{n2} -> {n0}")
    return w

def compare(w0, w1): 
    for k0, v0 in w0.items():
        v1 = w1[k0]
        assert(v0.shape == v1.shape)
        #plot_weight(v.numpy(), w_approx, k, r)
        nn = torch.norm(v0-v1)
        print(f"{k0} norm(diff) {nn.item()}")

def decompose_orig():
    print(f"to load orig model {args.MODEL_NAME}.pth...")
    w0 = torch.load(args.MODEL_NAME + '.pth', map_location='cpu') # xzl: load model...    
    print("model loaded")

    print("decompose to svd model...")
    w1 = full_to_svd(w0)
    for k in w1.keys():
        w1[k]=w1[k].to(dtype=torch.bfloat16) # still save as bfloat
    torch.save(w1,f"{args.MODEL_NAME}-svd-F{args.svdfac}.pth")
    print("saved")

    # load back 
    print("VALIDATE: to load svd model...")
    w2 = torch.load(
        f"{args.MODEL_NAME}-svd-F{args.svdfac}.pth",
        map_location='cpu'
        ) # xzl: load model...
    print("model loaded")

    print("recover full model & cmp...")
    w00 = svd_recover_to_full(w2) 
    compare(w0, w00)

    print(f"saved to {args.MODEL_NAME}-svd-F{args.svdfac}.pth")

# load a trained (or finetuned) custom model, recover to the orig format, 
#       so it can be exec with unmodified infer engine
def recover():
    print(f"to load custom model {args.MODEL_NAME}.pth...")
    w0 = torch.load(args.MODEL_NAME + '.pth', map_location='cpu') # xzl: load model...
    print("model loaded")

    w00 = svd_recover_to_full(w0) 
    for k in w00.keys():
        w00[k]=w00[k].to(dtype=torch.bfloat16) # still save as bfloat

    torch.save(w00,f"{args.MODEL_NAME}-recover.pth")
    print(f">>> saved to {args.MODEL_NAME}-recover.pth")


DEFAULT_ORIG = '/bigtemp/xl6yq/RWKV-5-World-0.4B-v2-20231113-ctx4096'
DEFAULT_MY = '/u/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L24-D1024-F4-x052attTune/rwkv-0'

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--orig_model", default=DEFAULT_ORIG, type=str)
    parser.add_argument("--my_model", default=DEFAULT_MY, type=str)
    parser.add_argument("-s", "--svdfac", default=4, type=int)
    parser.add_argument("-d", "--decompose", default=1, type=int)
    args = parser.parse_args()

    # defaults ... 
    args.n_layer = 24   # xzl: so we cannot figure out automatically???
    args.n_embd = 1024

    # args.n_layer = 12   
    # args.n_embd = 768
    args.vocab_size = 65536

    args.convert = 1

    if args.decompose: 
        args.MODEL_NAME = args.orig_model
        decompose_orig()
    else: 
        args.MODEL_NAME = args.my_model
        recover()