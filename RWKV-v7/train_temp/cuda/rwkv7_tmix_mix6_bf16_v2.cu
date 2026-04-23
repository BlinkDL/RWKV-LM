#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

__device__ inline float load_bf16(const at::BFloat16* ptr) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(ptr));
}

__device__ inline void store_bf16(at::BFloat16* ptr, float value) {
    *reinterpret_cast<__nv_bfloat16*>(ptr) = __float2bfloat16(value);
}

__device__ inline __nv_bfloat162 load_bf16x2(const at::BFloat16* ptr) {
    return *reinterpret_cast<const __nv_bfloat162*>(ptr);
}

__device__ inline void store_bf16x2(at::BFloat16* ptr, __nv_bfloat162 value) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) = value;
}

inline int64_t ceil_div(int64_t n, int64_t d) {
    return (n + d - 1) / d;
}

__global__ void tmix_mix6_forward_kernel(
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ x_r,
    const at::BFloat16* __restrict__ x_w,
    const at::BFloat16* __restrict__ x_k,
    const at::BFloat16* __restrict__ x_v,
    const at::BFloat16* __restrict__ x_a,
    const at::BFloat16* __restrict__ x_g,
    at::BFloat16* __restrict__ out_r,
    at::BFloat16* __restrict__ out_w,
    at::BFloat16* __restrict__ out_k,
    at::BFloat16* __restrict__ out_v,
    at::BFloat16* __restrict__ out_a,
    at::BFloat16* __restrict__ out_g,
    int64_t bt_size,
    int64_t t_size,
    int64_t c_size) {
    int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_pairs = bt_size * (c_size / 2);
    if (pair_idx >= total_pairs) {
        return;
    }

    int64_t bt = pair_idx / (c_size / 2);
    int64_t c_pair = pair_idx % (c_size / 2);
    int64_t c = c_pair * 2;
    int64_t idx = bt * c_size + c;
    int64_t t = bt % t_size;

    __nv_bfloat162 x_now = load_bf16x2(x + idx);
    __nv_bfloat162 x_prev = __floats2bfloat162_rn(0.0f, 0.0f);
    if (t > 0) {
        x_prev = load_bf16x2(x + idx - c_size);
    }
    __nv_bfloat162 xx = __hsub2(x_prev, x_now);

    store_bf16x2(out_r + idx, __hadd2(x_now, __hmul2(xx, load_bf16x2(x_r + c))));
    store_bf16x2(out_w + idx, __hadd2(x_now, __hmul2(xx, load_bf16x2(x_w + c))));
    store_bf16x2(out_k + idx, __hadd2(x_now, __hmul2(xx, load_bf16x2(x_k + c))));
    store_bf16x2(out_v + idx, __hadd2(x_now, __hmul2(xx, load_bf16x2(x_v + c))));
    store_bf16x2(out_a + idx, __hadd2(x_now, __hmul2(xx, load_bf16x2(x_a + c))));
    store_bf16x2(out_g + idx, __hadd2(x_now, __hmul2(xx, load_bf16x2(x_g + c))));
}

__global__ void tmix_mix6_backward_dx_kernel(
    const at::BFloat16* __restrict__ grad_r,
    const at::BFloat16* __restrict__ grad_w,
    const at::BFloat16* __restrict__ grad_k,
    const at::BFloat16* __restrict__ grad_v,
    const at::BFloat16* __restrict__ grad_a,
    const at::BFloat16* __restrict__ grad_g,
    const at::BFloat16* __restrict__ x_r,
    const at::BFloat16* __restrict__ x_w,
    const at::BFloat16* __restrict__ x_k,
    const at::BFloat16* __restrict__ x_v,
    const at::BFloat16* __restrict__ x_a,
    const at::BFloat16* __restrict__ x_g,
    at::BFloat16* __restrict__ grad_x,
    int64_t bt_size,
    int64_t t_size,
    int64_t c_size) {
    int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_pairs = bt_size * (c_size / 2);
    if (pair_idx >= total_pairs) {
        return;
    }

    int64_t bt = pair_idx / (c_size / 2);
    int64_t c_pair = pair_idx % (c_size / 2);
    int64_t c = c_pair * 2;
    int64_t idx = bt * c_size + c;
    int64_t t = bt % t_size;

    __nv_bfloat162 one = __floats2bfloat162_rn(1.0f, 1.0f);
    __nv_bfloat162 grad = __floats2bfloat162_rn(0.0f, 0.0f);

    __nv_bfloat162 pr = load_bf16x2(x_r + c);
    __nv_bfloat162 pw = load_bf16x2(x_w + c);
    __nv_bfloat162 pk = load_bf16x2(x_k + c);
    __nv_bfloat162 pv = load_bf16x2(x_v + c);
    __nv_bfloat162 pa = load_bf16x2(x_a + c);
    __nv_bfloat162 pg = load_bf16x2(x_g + c);

    grad = __hadd2(grad, __hmul2(load_bf16x2(grad_r + idx), __hsub2(one, pr)));
    grad = __hadd2(grad, __hmul2(load_bf16x2(grad_w + idx), __hsub2(one, pw)));
    grad = __hadd2(grad, __hmul2(load_bf16x2(grad_k + idx), __hsub2(one, pk)));
    grad = __hadd2(grad, __hmul2(load_bf16x2(grad_v + idx), __hsub2(one, pv)));
    grad = __hadd2(grad, __hmul2(load_bf16x2(grad_a + idx), __hsub2(one, pa)));
    grad = __hadd2(grad, __hmul2(load_bf16x2(grad_g + idx), __hsub2(one, pg)));

    if (t + 1 < t_size) {
        grad = __hadd2(grad, __hmul2(load_bf16x2(grad_r + idx + c_size), pr));
        grad = __hadd2(grad, __hmul2(load_bf16x2(grad_w + idx + c_size), pw));
        grad = __hadd2(grad, __hmul2(load_bf16x2(grad_k + idx + c_size), pk));
        grad = __hadd2(grad, __hmul2(load_bf16x2(grad_v + idx + c_size), pv));
        grad = __hadd2(grad, __hmul2(load_bf16x2(grad_a + idx + c_size), pa));
        grad = __hadd2(grad, __hmul2(load_bf16x2(grad_g + idx + c_size), pg));
    }

    store_bf16x2(grad_x + idx, grad);
}

__global__ void tmix_mix6_backward_param_kernel_v2(
    const at::BFloat16* __restrict__ grad_r,
    const at::BFloat16* __restrict__ grad_w,
    const at::BFloat16* __restrict__ grad_k,
    const at::BFloat16* __restrict__ grad_v,
    const at::BFloat16* __restrict__ grad_a,
    const at::BFloat16* __restrict__ grad_g,
    const at::BFloat16* __restrict__ x,
    at::BFloat16* __restrict__ grad_x_r,
    at::BFloat16* __restrict__ grad_x_w,
    at::BFloat16* __restrict__ grad_x_k,
    at::BFloat16* __restrict__ grad_x_v,
    at::BFloat16* __restrict__ grad_x_a,
    at::BFloat16* __restrict__ grad_x_g,
    int64_t bt_size,
    int64_t t_size,
    int64_t c_size) {
    int64_t c_pair = blockIdx.x;
    int64_t c = c_pair * 2;
    if (c + 1 >= c_size) {
        return;
    }

    __shared__ float sr0[256], sr1[256], sw0[256], sw1[256], sk0[256], sk1[256];
    __shared__ float sv0[256], sv1[256], sa0[256], sa1[256], sg0[256], sg1[256];

    float ar0 = 0.0f, ar1 = 0.0f, aw0 = 0.0f, aw1 = 0.0f, ak0 = 0.0f, ak1 = 0.0f;
    float av0 = 0.0f, av1 = 0.0f, aa0 = 0.0f, aa1 = 0.0f, ag0 = 0.0f, ag1 = 0.0f;

    for (int64_t bt = threadIdx.x; bt < bt_size; bt += blockDim.x) {
        int64_t idx = bt * c_size + c;
        int64_t t = bt % t_size;
        float2 x_now = __bfloat1622float2(load_bf16x2(x + idx));
        float2 x_prev = make_float2(0.0f, 0.0f);
        if (t > 0) {
            x_prev = __bfloat1622float2(load_bf16x2(x + idx - c_size));
        }
        float dx0 = x_prev.x - x_now.x;
        float dx1 = x_prev.y - x_now.y;
        float2 gr = __bfloat1622float2(load_bf16x2(grad_r + idx));
        float2 gw = __bfloat1622float2(load_bf16x2(grad_w + idx));
        float2 gk = __bfloat1622float2(load_bf16x2(grad_k + idx));
        float2 gv = __bfloat1622float2(load_bf16x2(grad_v + idx));
        float2 ga = __bfloat1622float2(load_bf16x2(grad_a + idx));
        float2 gg = __bfloat1622float2(load_bf16x2(grad_g + idx));
        ar0 += gr.x * dx0; ar1 += gr.y * dx1;
        aw0 += gw.x * dx0; aw1 += gw.y * dx1;
        ak0 += gk.x * dx0; ak1 += gk.y * dx1;
        av0 += gv.x * dx0; av1 += gv.y * dx1;
        aa0 += ga.x * dx0; aa1 += ga.y * dx1;
        ag0 += gg.x * dx0; ag1 += gg.y * dx1;
    }

    sr0[threadIdx.x] = ar0; sr1[threadIdx.x] = ar1;
    sw0[threadIdx.x] = aw0; sw1[threadIdx.x] = aw1;
    sk0[threadIdx.x] = ak0; sk1[threadIdx.x] = ak1;
    sv0[threadIdx.x] = av0; sv1[threadIdx.x] = av1;
    sa0[threadIdx.x] = aa0; sa1[threadIdx.x] = aa1;
    sg0[threadIdx.x] = ag0; sg1[threadIdx.x] = ag1;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sr0[threadIdx.x] += sr0[threadIdx.x + stride]; sr1[threadIdx.x] += sr1[threadIdx.x + stride];
            sw0[threadIdx.x] += sw0[threadIdx.x + stride]; sw1[threadIdx.x] += sw1[threadIdx.x + stride];
            sk0[threadIdx.x] += sk0[threadIdx.x + stride]; sk1[threadIdx.x] += sk1[threadIdx.x + stride];
            sv0[threadIdx.x] += sv0[threadIdx.x + stride]; sv1[threadIdx.x] += sv1[threadIdx.x + stride];
            sa0[threadIdx.x] += sa0[threadIdx.x + stride]; sa1[threadIdx.x] += sa1[threadIdx.x + stride];
            sg0[threadIdx.x] += sg0[threadIdx.x + stride]; sg1[threadIdx.x] += sg1[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        store_bf16x2(grad_x_r + c, __floats2bfloat162_rn(sr0[0], sr1[0]));
        store_bf16x2(grad_x_w + c, __floats2bfloat162_rn(sw0[0], sw1[0]));
        store_bf16x2(grad_x_k + c, __floats2bfloat162_rn(sk0[0], sk1[0]));
        store_bf16x2(grad_x_v + c, __floats2bfloat162_rn(sv0[0], sv1[0]));
        store_bf16x2(grad_x_a + c, __floats2bfloat162_rn(sa0[0], sa1[0]));
        store_bf16x2(grad_x_g + c, __floats2bfloat162_rn(sg0[0], sg1[0]));
    }
}

} // namespace

std::vector<torch::Tensor> tmix_mix6_forward_v2_cuda(
    torch::Tensor x,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
    auto out_r = torch::empty_like(x);
    auto out_w = torch::empty_like(x);
    auto out_k = torch::empty_like(x);
    auto out_v = torch::empty_like(x);
    auto out_a = torch::empty_like(x);
    auto out_g = torch::empty_like(x);
    const int threads = 256;
    const int64_t bt_size = x.size(0) * x.size(1);
    const int64_t c_size = x.size(2);
    auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t total_pairs = bt_size * (c_size / 2);
    const int blocks = static_cast<int>(ceil_div(total_pairs, static_cast<int64_t>(threads)));
    tmix_mix6_forward_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<at::BFloat16>(),
        x_r.data_ptr<at::BFloat16>(),
        x_w.data_ptr<at::BFloat16>(),
        x_k.data_ptr<at::BFloat16>(),
        x_v.data_ptr<at::BFloat16>(),
        x_a.data_ptr<at::BFloat16>(),
        x_g.data_ptr<at::BFloat16>(),
        out_r.data_ptr<at::BFloat16>(),
        out_w.data_ptr<at::BFloat16>(),
        out_k.data_ptr<at::BFloat16>(),
        out_v.data_ptr<at::BFloat16>(),
        out_a.data_ptr<at::BFloat16>(),
        out_g.data_ptr<at::BFloat16>(),
        bt_size,
        x.size(1),
        c_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_r, out_w, out_k, out_v, out_a, out_g};
}

std::vector<torch::Tensor> tmix_mix6_backward_v2_cuda(
    torch::Tensor grad_r,
    torch::Tensor grad_w,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_a,
    torch::Tensor grad_g,
    torch::Tensor x,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
    auto grad_x = torch::empty_like(x);
    auto grad_x_r = torch::empty({x.size(2)}, x.options());
    auto grad_x_w = torch::empty({x.size(2)}, x.options());
    auto grad_x_k = torch::empty({x.size(2)}, x.options());
    auto grad_x_v = torch::empty({x.size(2)}, x.options());
    auto grad_x_a = torch::empty({x.size(2)}, x.options());
    auto grad_x_g = torch::empty({x.size(2)}, x.options());

    const int threads = 256;
    const int64_t bt_size = x.size(0) * x.size(1);
    const int64_t c_size = x.size(2);
    auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t total_pairs = bt_size * (c_size / 2);
    const int blocks_dx = static_cast<int>(ceil_div(total_pairs, static_cast<int64_t>(threads)));
    tmix_mix6_backward_dx_kernel<<<blocks_dx, threads, 0, stream>>>(
        grad_r.data_ptr<at::BFloat16>(),
        grad_w.data_ptr<at::BFloat16>(),
        grad_k.data_ptr<at::BFloat16>(),
        grad_v.data_ptr<at::BFloat16>(),
        grad_a.data_ptr<at::BFloat16>(),
        grad_g.data_ptr<at::BFloat16>(),
        x_r.data_ptr<at::BFloat16>(),
        x_w.data_ptr<at::BFloat16>(),
        x_k.data_ptr<at::BFloat16>(),
        x_v.data_ptr<at::BFloat16>(),
        x_a.data_ptr<at::BFloat16>(),
        x_g.data_ptr<at::BFloat16>(),
        grad_x.data_ptr<at::BFloat16>(),
        bt_size,
        x.size(1),
        c_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    tmix_mix6_backward_param_kernel_v2<<<static_cast<int>(c_size / 2), threads, 0, stream>>>(
        grad_r.data_ptr<at::BFloat16>(),
        grad_w.data_ptr<at::BFloat16>(),
        grad_k.data_ptr<at::BFloat16>(),
        grad_v.data_ptr<at::BFloat16>(),
        grad_a.data_ptr<at::BFloat16>(),
        grad_g.data_ptr<at::BFloat16>(),
        x.data_ptr<at::BFloat16>(),
        grad_x_r.data_ptr<at::BFloat16>(),
        grad_x_w.data_ptr<at::BFloat16>(),
        grad_x_k.data_ptr<at::BFloat16>(),
        grad_x_v.data_ptr<at::BFloat16>(),
        grad_x_a.data_ptr<at::BFloat16>(),
        grad_x_g.data_ptr<at::BFloat16>(),
        bt_size,
        x.size(1),
        c_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_x, grad_x_r, grad_x_w, grad_x_k, grad_x_v, grad_x_a, grad_x_g};
}
