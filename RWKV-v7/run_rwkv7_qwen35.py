import numpy as np
import torch
from rwkv.utils import PIPELINE
from transformers import AutoTokenizer

PROBE_TEXT = " Eiffel"
Qwen35_PTH = "/home/qwen35_0_8b_text.pth" # use run_qwen35_make_pth.py to extract it from HF
RWKV7_PTH = "/home/rwkv7-g1d-0.4b-20260210-ctx8192.pth" # from https://huggingface.co/BlinkDL/rwkv7-g1/tree/main

class LLM:
    def __init__(self, name, checkpoint):
        self.name, self.checkpoint = name, checkpoint
        self.TM, self.CM, self.n_layer = (), (), 0
        print(f"\n== {self.name} ==")
        print(f"loading checkpoint: {self.checkpoint}")
        pth = torch.load(self.checkpoint, map_location="cpu", mmap=True)
        self.W = {k: v.detach().cpu().float().numpy().astype(np.float32, copy=False).squeeze() for k, v in pth.items()}
        print(f"loaded: keys={len(self.W):,} params={sum(v.size for v in self.W.values()):,}")

    def encode(self, text): raise NotImplementedError
    def decode(self, tokens): raise NotImplementedError
    def S0(self): raise NotImplementedError
    def EMB(self, token): raise NotImplementedError
    def NORM(self, X): raise NotImplementedError
    def HEAD(self, X): return X @ self.head

    def run_one(self, token, S):
        X = self.EMB(int(token))
        for TM, CM, s in zip(self.TM, self.CM, S):
            X, s[0] = TM(X, s[0])
            X, s[1] = CM(X, s[1])
        return self.HEAD(self.NORM(X)), S

    def forward(self, tokens, S=None):
        S = self.S0() if S is None else S
        logits = None
        for token in tokens:
            logits, S = self.run_one(token, S)
        return logits, S

    def report(self):
        probe_tokens = self.encode(PROBE_TEXT)
        logits, _ = self.forward(probe_tokens)
        print(f"\n== {self.name} top-10 logits ==")
        print(f"text: {PROBE_TEXT!r}")
        print(f"tokens: {probe_tokens}")
        for rank, (token, logit, prob) in enumerate(top_logits(logits), 1):
            print(f"{rank}: token={token} logit={logit:.6f} prob={prob:.8f} text={self.decode([token])!r}")

def SIGMOID(x): return 1.0 / (1.0 + np.exp(-x))
def SILU(x): return x * SIGMOID(x)
def RELUSQ(x): return np.maximum(x, 0.0) ** 2
def LERP(x, y, w): return x + w * (y - x)
def L2_QWEN(x): return x * (np.sum(x * x, axis=-1, keepdims=True) + 1e-6) ** -0.5
def L2_RWKV(x): return x * np.maximum(np.sum(x * x, axis=1, keepdims=True) ** 0.5, 1e-12) ** -1.0
def LAYER_NORM(x, w, b): return (x - x.mean(axis=-1, keepdims=True)) / (x.var(axis=-1, keepdims=True) + 1e-5) ** 0.5 * w + b
def RMS_NORM(x, w): return w * x * (np.mean(x * x, axis=-1, keepdims=True) + 1e-6) ** -0.5

def GROUP_NORM(x, w, b, eps):
    y = (x - x.mean(axis=1, keepdims=True)) / (x.var(axis=1, keepdims=True) + eps) ** 0.5
    return y.reshape(-1) * w + b

def DPLR_RWKV(S, R, W, K, V, A, B):
    S = np.einsum("hvk,hk->hvk", S, W) + np.einsum("hva,ha,hb->hvb", S, A, B) + np.einsum("hv,hk->hvk", V, K)
    return np.einsum("hvk,hk->hv", S, R), S

def DPLR(S, R, W, K, V, A, B):
    S = np.einsum("hk,hkv->hkv", W, S) + np.einsum("hb,ha,hav->hbv", B, A, S) + np.einsum("hk,hv->hkv", K, V)
    return np.einsum("hk,hkv->hv", R, S), S

def GQA(S, q, k, v, H, KV, N):
    S = np.concatenate((S, np.stack((k, v), axis=0).reshape(2, KV, 1, N)), axis=2)
    k, v = S
    k, v = np.repeat(k, H // KV, axis=0), np.repeat(v, H // KV, axis=0)
    a = SOFTMAX(np.einsum("hd,htd->ht", q, k) * (N**-0.5), axis=-1)
    return np.einsum("ht,htd->hd", a, v).reshape(H * N), S

def SOFTMAX(x, axis=-1):
    y = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return y / np.sum(y, axis=axis, keepdims=True)

def top_logits(logits, k=10):
    probs = SOFTMAX(logits)
    ids = np.argpartition(logits, -k)[-k:]
    ids = ids[np.argsort(logits[ids])[::-1]]
    return [(int(i), float(logits[i]), float(probs[i])) for i in ids]

class Qwen35(LLM):
    def __init__(self, checkpoint):
        super().__init__("Qwen3.5", checkpoint)
        W = self.W
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
        self.n_layer, self.C = 24, 1024
        self.H, self.N, self.conv_len = 16, 128, 4
        self.aH, self.aKV, self.aN = 8, 2, 256
        self.gdn_layers = tuple(i % 4 != 3 for i in range(self.n_layer))
        self.emb, self.ln_outW = W["embed_tokens.weight"], W["norm.weight"]+1
        self.head = W["lm_head.weight"].T if "lm_head.weight" in W else self.emb.T
        self.TM = tuple(self.make_GDN(i) if gdn else self.make_GQA(i) for i, gdn in enumerate(self.gdn_layers))
        self.CM = tuple(self.make_FFN(i) for i in range(self.n_layer))

    def encode(self, text): return self.tokenizer.encode(text, add_special_tokens=False)
    def decode(self, tokens): return self.tokenizer.decode(tokens)

    def S0(self):
        S = []
        for gdn in self.gdn_layers:
            if gdn: S.append([{"conv": np.zeros((3*self.H*self.N, self.conv_len - 1), np.float32), "rnn": np.zeros((self.H, self.N, self.N), np.float32)}, None])
            else: S.append([{"kv": np.zeros((2, self.aKV, 0, self.aN), np.float32)}, None])
        return S

    def EMB(self, token): return self.emb[token]
    def NORM(self, X): return RMS_NORM(X, self.ln_outW)

    def make_GDN(self, i):
        p, W, H, N = f"layers.{i}.linear_attn.", self.W, self.H, self.N
        lnW, qkvW, convW = W[f"layers.{i}.input_layernorm.weight"]+1, W[p+"in_proj_qkv.weight"].T, W[p+"conv1d.weight"]
        gW, aW, wW, wB = W[p+"in_proj_z.weight"].T, W[p+"in_proj_b.weight"].T, W[p+"in_proj_a.weight"].T, W[p+"dt_bias"]
        wP, oNorm, oW = np.exp(W[p+"A_log"]), W[p+"norm.weight"], W[p+"out_proj.weight"].T
        def layer(X, state):
            x = RMS_NORM(X, lnW)
            conv = np.concatenate((state["conv"], (x @ qkvW).reshape(3*H*N, 1)), axis=-1)
            state["conv"] = conv[:, 1:].copy()
            qkv = SILU(np.sum(conv * convW, axis=-1))
            q, k, v = np.split(qkv, 3)

            q = L2_QWEN(q.reshape(H, N)) * (N**-0.5)
            k = L2_QWEN(k.reshape(H, N))
            v = v.reshape(H, N)
            w = np.pow(1.0 + np.exp(wB + x @ wW), -wP).reshape(H, 1)
            a = SIGMOID(x @ aW).reshape(H, 1)

            y, state["rnn"] = DPLR(state["rnn"], q, w, k, a*v, -a*w*k, k)
            y = RMS_NORM(y, oNorm).reshape(H*N)
            g = SILU(x @ gW)
            return X + (y * g) @ oW, state
        return layer

    def make_GQA(self, i):
        p, W, C, H, KV, N = f"layers.{i}.self_attn.", self.W, self.C, self.aH, self.aKV, self.aN
        lnW, qgW, q_norm = W[f"layers.{i}.input_layernorm.weight"]+1, W[p+"q_proj.weight"].T, W[p+"q_norm.weight"]+1
        kW, vW, oW, k_norm = W[p+"k_proj.weight"].T, W[p+"v_proj.weight"].T, W[p+"o_proj.weight"].T, W[p+"k_norm.weight"]+1
        qgW = qgW.reshape(C, H, N*2)
        qW, gW = qgW[:, :, :N].reshape(C, H*N), qgW[:, :, N:].reshape(C, H*N)
        rope_dim = N // 4
        inv_freq = 1.0 / (10000000.0 ** (np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim))
        def rotate_half(x):
            h = x.shape[-1] // 2
            return np.concatenate((-x[..., h:], x[..., :h]), axis=-1)
        def layer(X, state):
            x = RMS_NORM(X, lnW)
            q = RMS_NORM((x @ qW).reshape(H, N), q_norm)
            k = RMS_NORM((x @ kW).reshape(KV, N), k_norm)
            v = (x @ vW).reshape(KV, N)

            freq = state["kv"].shape[2] * inv_freq
            cos, sin = np.concatenate((np.cos(freq), np.cos(freq))), np.concatenate((np.sin(freq), np.sin(freq)))
            q0, k0 = q[..., :rope_dim], k[..., :rope_dim]
            q = np.concatenate((q0 * cos + rotate_half(q0) * sin, q[..., rope_dim:]), axis=-1)
            k = np.concatenate((k0 * cos + rotate_half(k0) * sin, k[..., rope_dim:]), axis=-1)

            y, state["kv"] = GQA(state["kv"], q, k, v, H, KV, N)
            g = SIGMOID(x @ gW)
            return X + (y * g) @ oW, state
        return layer

    def make_FFN(self, i):
        W, p = self.W, f"layers.{i}.mlp."
        lnW = W[f"layers.{i}.post_attention_layernorm.weight"]+1
        gW, kW, vW = W[p+"gate_proj.weight"].T, W[p+"up_proj.weight"].T, W[p+"down_proj.weight"].T
        def layer(X, state):
            x = RMS_NORM(X, lnW)
            return X + ((SILU(x @ gW) * (x @ kW)) @ vW), state
        return layer

class RWKV7(LLM):
    def __init__(self, checkpoint):
        super().__init__("RWKV-7", checkpoint)
        W = self.W
        self.tokenizer = PIPELINE(None, "rwkv_vocab_v20230424")
        self.n_layer = 1 + max(int(k.split(".")[1]) for k in W if k.startswith("blocks."))
        self.C, self.N = W["emb.weight"].shape[1], 64
        self.H = self.C // self.N
        self.emb = LAYER_NORM(W["emb.weight"], W["blocks.0.ln0.weight"], W["blocks.0.ln0.bias"])
        self.ln_outW, self.ln_outB, self.head = W["ln_out.weight"], W["ln_out.bias"], W["head.weight"].T
        self.TM = tuple(self.make_TMIX(i) for i in range(self.n_layer))
        self.CM = tuple(self.make_CMIX(i) for i in range(self.n_layer))

    def encode(self, text): return self.tokenizer.encode(text)
    def decode(self, tokens): return self.tokenizer.decode(tokens)

    def S0(self):
        S = []
        for _ in range(self.n_layer):
            S.append([
                {"x": np.zeros(self.C, np.float32), "rnn": np.zeros((self.H, self.N, self.N), np.float32)},
                {"x": np.zeros(self.C, np.float32)},
            ])
        return S

    def EMB(self, token): return (self.emb[token], None)
    def NORM(self, X): return LAYER_NORM(X[0], self.ln_outW, self.ln_outB)

    def make_TMIX(self, i):
        p, W, H, N = f"blocks.{i}.att.", self.W, self.H, self.N
        lnW, lnB = W[f"blocks.{i}.ln1.weight"], W[f"blocks.{i}.ln1.bias"]
        x_r, x_w, x_k, x_v, x_a, x_g = W[p+"x_r"], W[p+"x_w"], W[p+"x_k"], W[p+"x_v"], W[p+"x_a"], W[p+"x_g"]
        rW, kW, vW, oW = W[p+"receptance.weight"].T, W[p+"key.weight"].T, W[p+"value.weight"].T, W[p+"output.weight"].T
        w0, w1, w2, a0, a1, a2, v0, v1, v2 = W[p+"w0"], W[p+"w1"], W[p+"w2"], W[p+"a0"], W[p+"a1"], W[p+"a2"], W[p+"v0"], W[p+"v1"], W[p+"v2"]
        g1, g2, k_k, k_a, r_k, ln_xW, ln_xB = W[p+"g1"], W[p+"g2"], W[p+"k_k"], W[p+"k_a"], W[p+"r_k"], W[p+"ln_x.weight"], W[p+"ln_x.bias"]
        def layer(X, state):
            x0, v_first = X
            x = LAYER_NORM(x0, lnW, lnB)
            prev, state["x"] = state["x"], x
            xr, xw, xk = LERP(x, prev, x_r), LERP(x, prev, x_w), LERP(x, prev, x_k)
            xv, xa, xg = LERP(x, prev, x_v), LERP(x, prev, x_a), LERP(x, prev, x_g)

            r, k, v = xr @ rW, xk @ kW, xv @ vW
            if v_first is None: v_first = v
            else: v = LERP(v, v_first, SIGMOID(v0 + xv @ v1 @ v2))
            w = np.exp(-SIGMOID(w0 + np.tanh(xw @ w1) @ w2) / np.e**0.5)
            a = SIGMOID(a0 + xa @ a1 @ a2)
            kk = k * k_k
            k = LERP(k, k*a, k_a)
            r, w, k, v, kk, a = [z.reshape(H, N) for z in (r, w, k, v, kk, a)]
            kk = L2_RWKV(kk)

            y, state["rnn"] = DPLR_RWKV(state["rnn"], r, w, k, v, kk, -kk*a)
            y = GROUP_NORM(y, ln_xW, ln_xB, 64e-5)
            y += (np.sum(r * k * r_k, axis=1, keepdims=True) * v).reshape(-1)
            g = SIGMOID(xg @ g1) @ g2
            return (x0 + (y * g) @ oW, v_first), state
        return layer

    def make_CMIX(self, i):
        p, W = f"blocks.{i}.ffn.", self.W
        lnW, lnB, x_k, kW, vW = W[f"blocks.{i}.ln2.weight"], W[f"blocks.{i}.ln2.bias"], W[p+"x_k"], W[p+"key.weight"].T, W[p+"value.weight"].T
        def layer(X, state):
            x0, v_first = X
            x = LAYER_NORM(x0, lnW, lnB)
            prev, state["x"] = state["x"], x
            x = LERP(x, prev, x_k)
            return (x0 + RELUSQ(x @ kW) @ vW, v_first), state
        return layer

if __name__ == "__main__":
    for model, checkpoint in ((Qwen35, Qwen35_PTH), (RWKV7, RWKV7_PTH)):
        llm = model(checkpoint)
        llm.report()
