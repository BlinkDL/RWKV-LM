import numpy as np
import torch
from rwkv.utils import PIPELINE

PROBE_TEXT = "One thing that should be learned from the bitter lesson is the great power of general purpose methods"

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

def SIGMOID(x): return 1.0 / (1.0 + np.exp(-x))
def RELUSQ(x): return np.maximum(x, 0.0) ** 2
def LERP(x, y, w): return x + w * (y - x)
def L2_RWKV(x): return x * np.maximum(np.sum(x * x, axis=1, keepdims=True) ** 0.5, 1e-12) ** -1.0
def LAYER_NORM(x, w, b): return (x - x.mean(axis=-1, keepdims=True)) / (x.var(axis=-1, keepdims=True) + 1e-5) ** 0.5 * w + b

def GROUP_NORM(x, w, b, eps):
    y = (x - x.mean(axis=1, keepdims=True)) / (x.var(axis=1, keepdims=True) + eps) ** 0.5
    return y.reshape(-1) * w + b

def DPLR_RWKV(S, R, W, K, V, A, B):
    S = np.einsum("hvk,hk->hvk", S, W) + np.einsum("hva,ha,hb->hvb", S, A, B) + np.einsum("hv,hk->hvk", V, K)
    return np.einsum("hvk,hk->hv", S, R), S

def DPLR_RWKV_CHUNK(P, C, W, K, V, A, B):
    P = np.einsum("hvb,hb->hvb", P, W) + np.einsum("hva,ha,hb->hvb", P, A, B)
    C = np.einsum("hvb,hb->hvb", C, W) + np.einsum("hva,ha,hb->hvb", C, A, B) + np.einsum("hv,hk->hvk", V, K)
    return P, C

class RWKV7(LLM):
    def __init__(self, checkpoint):
        super().__init__("RWKV-7", checkpoint)
        W = self.W
        self.tokenizer = PIPELINE(None, "rwkv_vocab_v20230424")
        self.n_layer = 1 + max(int(k.split(".")[1]) for k in W if k.startswith("blocks."))
        self.C, self.N = W["emb.weight"].shape[1], 64
        self.H = self.C // self.N
        self.rnnP0 = np.broadcast_to(np.eye(self.N, dtype=np.float32), (self.H, self.N, self.N)).copy()
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
                {
                    "x": np.zeros(self.C, np.float32),
                    "rnn": np.zeros((self.H, self.N, self.N), np.float32),
                    "rnnP": self.rnnP0.copy(),
                    "rnnC": np.zeros((self.H, self.N, self.N), np.float32),
                },
                {"x": np.zeros(self.C, np.float32)},
            ])
        return S

    def EMB(self, token): return (self.emb[token], None)
    def NORM(self, X): return LAYER_NORM(X[0], self.ln_outW, self.ln_outB)

    def run_chunks(self, tokens, chunk_len):
        S, chunks = self.S0(), [[] for _ in range(self.n_layer)]
        for i, token in enumerate(tokens, 1):
            _, S = self.run_one(token, S)
            if i % chunk_len == 0:
                for j, s in enumerate(S):
                    tmix = s[0]
                    chunks[j].append((tmix["rnnP"].copy(), tmix["rnnC"].copy()))
                    tmix["rnnP"], tmix["rnnC"] = self.rnnP0.copy(), np.zeros_like(tmix["rnnC"])
        return S, chunks

    def report(self):
        probe_tokens = self.encode(PROBE_TEXT)
        _, direct_S = self.forward(probe_tokens)
        n = len(probe_tokens)
        assert n % 3 == 0
        chunk_len = n // 3
        chunk_S, chunk_pc = self.run_chunks(probe_tokens, chunk_len)
        diffs = []
        chunk_run_ok = []
        chunk_summary_ok = []
        direct_c_ok = []
        for i in range(self.n_layer):
            direct = direct_S[i][0]["rnn"]
            merged = np.zeros_like(direct)
            for P, C in chunk_pc[i]:
                merged = np.einsum("hva,hab->hvb", merged, P) + C
            diff = np.abs(direct - merged)
            diffs.append((diff.max(), diff.mean()))
            direct_c_ok.append(np.allclose(direct, direct_S[i][0]["rnnC"], rtol=1e-5, atol=1e-5))
            chunk_run_ok.append(np.allclose(direct, chunk_S[i][0]["rnn"], rtol=1e-5, atol=1e-5))
            chunk_summary_ok.append(np.allclose(direct, merged, rtol=1e-5, atol=1e-5))
        print(f"\n== {self.name} context parallelism prefill ==")
        print(f"text: {PROBE_TEXT!r}")
        print(f"tokens ({len(probe_tokens)}): {probe_tokens}")
        print(f"chunks: {[chunk_len] * 3}")
        print(f"layers: {self.n_layer}")
        print(f"TMIX rnn state shape per layer: {list(direct_S[0][0]['rnn'].shape)}")
        print(f"chunk 0 P shape={list(chunk_pc[0][0][0].shape)} C shape={list(chunk_pc[0][0][1].shape)}")
        print(f"all layers direct state == direct one-pass C: {all(direct_c_ok)}")
        print(f"all layers direct state == chunk run state: {all(chunk_run_ok)}")
        print(f"all layers direct state == chunk-summary state: {all(chunk_summary_ok)}")
        for i, (max_diff, mean_diff) in enumerate(diffs):
            print(f"layer {i:02d}: ok={chunk_summary_ok[i]} max_abs_diff={max_diff:.8g} mean_abs_diff={mean_diff:.8g}")

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
            b = -kk * a

            y, state["rnn"] = DPLR_RWKV(state["rnn"], r, w, k, v, kk, b)
            state["rnnP"], state["rnnC"] = DPLR_RWKV_CHUNK(state["rnnP"], state["rnnC"], w, k, v, kk, b)
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
    for model, checkpoint in ((RWKV7, RWKV7_PTH),):
        llm = model(checkpoint)
        llm.report()
