########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
# RWKV-7 in numpy, by https://github.com/johanwind

import numpy as np
from torch import load as torch_load

layer_norm = lambda x, w, b : (x - x.mean()) / (x.var() + 1e-5)**0.5 * w + b
group_norm = lambda x, w, b : ((x - x.mean(axis=1,keepdims=1)) / (x.var(axis=1,keepdims=1) + 64e-5)**0.5).flatten() * w + b
sigmoid = lambda x : 1/(1+np.exp(-x))

def time_mixing(x, v0, last_x, S, params):
    # use this and remove other param[] if you are testing models trained by RWKV-LM
    # mr,mw,mk,mv,ma,mg, Ww1,Ww2,w_bias, Wa1,Wa2,a_bias, Wv1,Wv2,v_bias, Wg1,Wg2, k_k,k_a,r_k, Wr,Wk,Wv,Wo, ln_w,ln_b = params
    
    mr,mw,mk,mv,ma,mg, w_bias, r_k, Ww1,Ww2, Wa1,Wa2,a_bias, Wg1,Wg2 = params[:15]
    k_k,k_a, Wr,Wk,Wv,Wo, ln_w,ln_b = params[-8:]

    xr,xw,xk,xv,xa,xg = [x + m * (last_x - x) for m in [mr,mw,mk,mv,ma,mg]]

    r = Wr @ xr
    w = np.exp(-sigmoid(np.tanh(xw @ Ww1) @ Ww2 + w_bias)/np.e**0.5)
    k = Wk @ xk
    v = Wv @ xv
    if v0 is None:
        v0 = v
    else:
        Wv2,Wv1,v_bias = params[15:18]
        v += (v0 - v) * sigmoid(xv @ Wv1 @ Wv2 + v_bias)
    a = sigmoid(xa @ Wa1 @ Wa2 + a_bias)
    g = sigmoid(xg @ Wg1) @ Wg2
    kk = k * k_k
    k += k * (a-1) * k_a

    r,w,k,v,kk,a,r_k = [i.reshape(N_HEAD, HEAD_SIZE, 1) for i in [r,w,k,v,kk,a,r_k]]
    kk /= np.maximum(np.linalg.norm(kk, axis=1,keepdims=1), 1e-12)

    S = S * w.mT - S @ kk * (kk*a).mT + v * k.mT
    y = S @ r

    y = group_norm(y, ln_w, ln_b)
    y += ((r * k * r_k).sum(axis=1,keepdims=1) * v).flatten()
    return Wo @ (y * g), v0, x, S

def channel_mixing(x, last_x, mix, Wk, Wv):
    k = Wk @ ( x + mix * (last_x - x) )
    v = Wv @ np.maximum(k, 0)**2
    return v, x


def RWKV7(params, token, state):
    x = params('emb')[0][token]
    x = layer_norm(x, *params('blocks.0.ln0'))

    v0 = None
    for i in range(N_LAYER):
        x_ = layer_norm(x, *params(f'blocks.{i}.ln1'))
        dx, v0, state[0][i,0], state[1][i] = time_mixing(x_, v0, state[0][i,0], state[1][i], params(f'blocks.{i}.att'))
        x = x + dx

        x_ = layer_norm(x, *params(f'blocks.{i}.ln2'))
        dx, state[0][i,1] = channel_mixing(x_, state[0][i,1], *params(f'blocks.{i}.ffn'))
        x = x + dx

    x = layer_norm(x, *params('ln_out'))
    logits = params('head')[0] @ x

    return logits, state


# Verification

# Available at https://huggingface.co/BlinkDL/rwkv7-g1/blob/main/rwkv7-g1-0.1b-20250307-ctx4096.pth
MODEL_FILE = '/mnt/e/RWKV-Runner/models/rwkv7-g1-0.1b-20250307-ctx4096.pth'
N_LAYER = 12
N_EMBD = 768

HEAD_SIZE = 64
N_HEAD = N_EMBD//HEAD_SIZE

if 1: # Reference implementation
    context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."

    # pip install rwkv
    import os
    os.environ["RWKV_V7_ON"] = "1"
    from rwkv.utils import PIPELINE
    from rwkv.model import RWKV as referenceRWKV

    model = referenceRWKV(model=MODEL_FILE[:-4], strategy='cpu fp32')
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
    tokens = pipeline.encode(context)

    reference_logits, state = model.forward(tokens, None)
    reference_logits = reference_logits.numpy()

weights = torch_load(MODEL_FILE, map_location='cpu', weights_only=True)
weights = {k : v.squeeze().float().numpy() for k,v in weights.items()}
params = lambda prefix : [weights[key] for key in weights.keys() if key.startswith(prefix)]

state = (np.zeros((N_LAYER, 2, N_EMBD), dtype=np.float32),
         np.zeros((N_LAYER, N_HEAD, HEAD_SIZE, HEAD_SIZE), dtype=np.float32))
for token in tokens:
    minimal_logits, state = RWKV7(params, token, state)

print('Deviation from official rwkv:', max(abs(minimal_logits-reference_logits))/reference_logits.std())
