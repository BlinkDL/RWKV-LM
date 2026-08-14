import torch
import torch_npu
import sys
sys.path.append("./build")
import wkv7s

HEAD_SIZE = 64
DTYPE = torch.float16

# load(name="wkv7s", sources=["wkv7s_op.cpp", f"wkv7s.cu"], is_python_module=False,
#                     verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
class WKV_7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b):
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
            wkv7s.forward(B, T, C, H, state, r, w, k, v, a, b, y)
            return y

def RWKV7_OP_KERNEL(state, r, w, k, v, a, b):
    return WKV_7.apply(state, r, w, k, v, a, b)


def RWKV7_OP_TORCH(state, r, w, k, v, a, b):
        B, T, C = r.size()
        H = C // HEAD_SIZE
        N = HEAD_SIZE
        r = r.view(B, T, H, N).float()
        k = k.view(B, T, H, N).float()
        v = v.view(B, T, H, N).float()
        a = a.view(B, T, H, N).float()
        b = b.view(B, T, H, N).float()
        w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
        out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)

        for t in range(T):
            kk = k[:, t, :].view(B, H, 1, N)
            rr = r[:, t, :].view(B, H, N, 1)
            vv = v[:, t, :].view(B, H, N, 1)
            aa = a[:, t, :].view(B, H, N, 1)
            bb = b[:, t, :].view(B, H, 1, N)
            state = state * w[: , t, :, None, :] + state @ aa @ bb + vv @ kk
            out[:, t, :] = (state @ rr).view(B, H, N)

        return out.view(B, T, C).to(dtype=DTYPE), state
    
    
if __name__ == "__main__":
    device = "npu"
    B = 1
    T = 1
    C = 1024
    
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)
 
    r = torch.randn(B, T, C, dtype=DTYPE, device=device).contiguous()
    w = torch.randn(B, T, C, dtype=DTYPE, device=device).contiguous()
    k = torch.randn(B, T, C, dtype=DTYPE, device=device).contiguous()
    v = torch.randn(B, T, C, dtype=DTYPE, device=device).contiguous()
    a = torch.randn(B, T, C, dtype=DTYPE, device=device).contiguous()
    b = torch.randn(B, T, C, dtype=DTYPE, device=device).contiguous()
    state = torch.randn(B, C // HEAD_SIZE, HEAD_SIZE, HEAD_SIZE, dtype=torch.float, device=device)
    
    with torch.no_grad():
        y_torch, state_torch = RWKV7_OP_TORCH(state.clone(), r, w, k, v, a, b)
        torch.npu.synchronize()
        state_kernel = state.clone()
        y_kernel = RWKV7_OP_KERNEL(state_kernel, r, w, k, v, a, b)
       
    print(r[0][0][:64])
    print(state[0][0][0][:64])
    
    print(state_torch[0][0][0][:64])
    print(state_kernel[0][0][0][:64])
    
    print(y_torch[0][0][:64])
    print(y_kernel[0][0][:64])
    
    
    # === 比较结果 ===
    abs_diff = (y_kernel - y_torch).abs().float()
    max_diff = abs_diff.max().item()
    print("Max absolute difference:", max_diff)
    print("All close (atol=1e-3):", torch.allclose(y_kernel, y_torch, atol=1e-3, rtol=1e-3))