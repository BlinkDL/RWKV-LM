#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kHeadSize = 64;
constexpr float kLnXEps = 64e-5f;

__device__ inline __nv_bfloat162 load_bf16x2(const at::BFloat16* ptr) {
    return *reinterpret_cast<const __nv_bfloat162*>(ptr);
}

__device__ inline void store_bf16(at::BFloat16* ptr, float value) {
    *reinterpret_cast<__nv_bfloat16*>(ptr) = __float2bfloat16(value);
}

__device__ inline void store_bf16x2(at::BFloat16* ptr, float v0, float v1) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) = __floats2bfloat162_rn(v0, v1);
}

__device__ inline float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

inline int64_t ceil_div(int64_t n, int64_t d) {
    return (n + d - 1) / d;
}

__global__ void tmix_lnx_rkvres_forward_kernel(
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ r,
    const at::BFloat16* __restrict__ k,
    const at::BFloat16* __restrict__ v,
    const at::BFloat16* __restrict__ r_k,
    const at::BFloat16* __restrict__ weight,
    const at::BFloat16* __restrict__ bias,
    at::BFloat16* __restrict__ y,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    float* __restrict__ scale,
    int64_t ngroups,
    int64_t nrows) {
    const int64_t ng = static_cast<int64_t>(blockIdx.x);
    if (ng >= nrows * ngroups) {
        return;
    }

    const int lane = threadIdx.x;
    const int64_t row = ng / ngroups;
    const int64_t group = ng % ngroups;
    const int64_t c0 = group * kHeadSize;
    const int64_t base = row * (ngroups * kHeadSize) + c0;
    const int64_t idx = base + static_cast<int64_t>(lane) * 2;
    const int64_t c = c0 + static_cast<int64_t>(lane) * 2;

    const __nv_bfloat162 xv2 = load_bf16x2(x + idx);
    const float x0 = __low2float(xv2);
    const float x1 = __high2float(xv2);

    float sum = x0 + x1;
    sum = warp_sum(sum);
    const float mean_val = __shfl_sync(0xffffffffu, sum, 0) * (1.0f / kHeadSize);

    const float d0 = x0 - mean_val;
    const float d1 = x1 - mean_val;
    float sq = d0 * d0 + d1 * d1;
    sq = warp_sum(sq);
    const float var_val = __shfl_sync(0xffffffffu, sq, 0) * (1.0f / kHeadSize);
    const float rstd_val = rsqrtf(var_val + kLnXEps);

    const __nv_bfloat162 rv2 = load_bf16x2(r + idx);
    const __nv_bfloat162 kv2 = load_bf16x2(k + idx);
    const __nv_bfloat162 rk2 = load_bf16x2(r_k + c);
    const float scale_local =
        __low2float(rv2) * __low2float(kv2) * __low2float(rk2) +
        __high2float(rv2) * __high2float(kv2) * __high2float(rk2);
    const float scale_val = __shfl_sync(0xffffffffu, warp_sum(scale_local), 0);

    if (lane == 0) {
        mean[ng] = mean_val;
        rstd[ng] = rstd_val;
        scale[ng] = scale_val;
    }

    const __nv_bfloat162 vv2 = load_bf16x2(v + idx);
    const __nv_bfloat162 w2 = load_bf16x2(weight + c);
    const __nv_bfloat162 b2 = load_bf16x2(bias + c);
    const float y0 = d0 * rstd_val * __low2float(w2) + __low2float(b2) + scale_val * __low2float(vv2);
    const float y1 = d1 * rstd_val * __high2float(w2) + __high2float(b2) + scale_val * __high2float(vv2);
    store_bf16x2(y + idx, y0, y1);
}

__global__ void tmix_lnx_rkvres_backward_kernel(
    const at::BFloat16* __restrict__ grad_y,
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ r,
    const at::BFloat16* __restrict__ k,
    const at::BFloat16* __restrict__ v,
    const at::BFloat16* __restrict__ r_k,
    const at::BFloat16* __restrict__ weight,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ scale,
    at::BFloat16* __restrict__ grad_x,
    at::BFloat16* __restrict__ grad_r,
    at::BFloat16* __restrict__ grad_k,
    at::BFloat16* __restrict__ grad_v,
    float* __restrict__ grad_r_k,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    int64_t ngroups,
    int64_t nrows) {
    const int64_t ng = static_cast<int64_t>(blockIdx.x);
    if (ng >= nrows * ngroups) {
        return;
    }

    const int lane = threadIdx.x;
    const int64_t row = ng / ngroups;
    const int64_t group = ng % ngroups;
    const int64_t c0 = group * kHeadSize;
    const int64_t base = row * (ngroups * kHeadSize) + c0;
    const int64_t idx = base + static_cast<int64_t>(lane) * 2;
    const int64_t c = c0 + static_cast<int64_t>(lane) * 2;

    const float mean_val = mean[ng];
    const float rstd_val = rstd[ng];
    const float scale_val = scale[ng];

    const __nv_bfloat162 xv2 = load_bf16x2(x + idx);
    const __nv_bfloat162 gy2 = load_bf16x2(grad_y + idx);
    const __nv_bfloat162 w2 = load_bf16x2(weight + c);
    const __nv_bfloat162 vv2 = load_bf16x2(v + idx);

    const float x0 = __low2float(xv2);
    const float x1 = __high2float(xv2);
    const float gy0 = __low2float(gy2);
    const float gy1 = __high2float(gy2);
    const float w0 = __low2float(w2);
    const float w1 = __high2float(w2);
    const float xhat0 = (x0 - mean_val) * rstd_val;
    const float xhat1 = (x1 - mean_val) * rstd_val;
    const float dxhat0 = gy0 * w0;
    const float dxhat1 = gy1 * w1;

    float sum_dxhat = dxhat0 + dxhat1;
    float sum_dxhat_xhat = dxhat0 * xhat0 + dxhat1 * xhat1;
    sum_dxhat = warp_sum(sum_dxhat);
    sum_dxhat_xhat = warp_sum(sum_dxhat_xhat);
    const float total_dxhat = __shfl_sync(0xffffffffu, sum_dxhat, 0);
    const float total_dxhat_xhat = __shfl_sync(0xffffffffu, sum_dxhat_xhat, 0);
    const float inv_m = 1.0f / kHeadSize;

    const float gx0 = (dxhat0 - total_dxhat * inv_m - xhat0 * total_dxhat_xhat * inv_m) * rstd_val;
    const float gx1 = (dxhat1 - total_dxhat * inv_m - xhat1 * total_dxhat_xhat * inv_m) * rstd_val;
    store_bf16x2(grad_x + idx, gx0, gx1);

    atomicAdd(grad_weight + c + 0, gy0 * xhat0);
    atomicAdd(grad_weight + c + 1, gy1 * xhat1);
    atomicAdd(grad_bias + c + 0, gy0);
    atomicAdd(grad_bias + c + 1, gy1);

    const float v0 = __low2float(vv2);
    const float v1 = __high2float(vv2);
    const float q = __shfl_sync(0xffffffffu, warp_sum(gy0 * v0 + gy1 * v1), 0);
    store_bf16x2(grad_v + idx, gy0 * scale_val, gy1 * scale_val);

    const __nv_bfloat162 rv2 = load_bf16x2(r + idx);
    const __nv_bfloat162 kv2 = load_bf16x2(k + idx);
    const __nv_bfloat162 rk2 = load_bf16x2(r_k + c);
    const float r0 = __low2float(rv2);
    const float r1 = __high2float(rv2);
    const float k0 = __low2float(kv2);
    const float k1 = __high2float(kv2);
    const float rk0 = __low2float(rk2);
    const float rk1 = __high2float(rk2);

    store_bf16x2(grad_r + idx, q * k0 * rk0, q * k1 * rk1);
    store_bf16x2(grad_k + idx, q * r0 * rk0, q * r1 * rk1);
    atomicAdd(grad_r_k + c + 0, q * r0 * k0);
    atomicAdd(grad_r_k + c + 1, q * r1 * k1);
}

__global__ void cast_float_to_bf16_kernel(
    const float* __restrict__ src,
    at::BFloat16* __restrict__ dst,
    int64_t size) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    store_bf16(dst + idx, src[idx]);
}

} // namespace

std::vector<torch::Tensor> tmix_lnx_rkvres_forward_cuda(
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias) {
    auto y = torch::empty_like(x);
    const int64_t b = x.size(0);
    const int64_t t = x.size(1);
    const int64_t c = x.size(2);
    const int64_t ngroups = c / kHeadSize;
    auto mean = torch::empty({b, t, ngroups}, x.options().dtype(torch::kFloat32));
    auto rstd = torch::empty({b, t, ngroups}, x.options().dtype(torch::kFloat32));
    auto scale = torch::empty({b, t, ngroups}, x.options().dtype(torch::kFloat32));

    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t total_groups = b * t * ngroups;
    tmix_lnx_rkvres_forward_kernel<<<static_cast<int>(total_groups), 32, 0, stream>>>(
        x.data_ptr<at::BFloat16>(),
        r.data_ptr<at::BFloat16>(),
        k.data_ptr<at::BFloat16>(),
        v.data_ptr<at::BFloat16>(),
        r_k.data_ptr<at::BFloat16>(),
        weight.data_ptr<at::BFloat16>(),
        bias.data_ptr<at::BFloat16>(),
        y.data_ptr<at::BFloat16>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        scale.data_ptr<float>(),
        ngroups,
        b * t);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, mean, rstd, scale};
}

std::vector<torch::Tensor> tmix_lnx_rkvres_backward_cuda(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor mean,
    torch::Tensor rstd,
    torch::Tensor scale) {
    auto grad_x = torch::empty_like(x);
    auto grad_r = torch::empty_like(r);
    auto grad_k = torch::empty_like(k);
    auto grad_v = torch::empty_like(v);
    auto grad_r_k_fp32 = torch::zeros({r_k.size(0), r_k.size(1)}, r_k.options().dtype(torch::kFloat32));
    auto grad_weight_fp32 = torch::zeros({x.size(2)}, x.options().dtype(torch::kFloat32));
    auto grad_bias_fp32 = torch::zeros({x.size(2)}, x.options().dtype(torch::kFloat32));
    auto grad_r_k = torch::empty_like(r_k);
    auto grad_weight = torch::empty_like(weight);
    auto grad_bias = torch::empty_like(weight);

    const int64_t b = x.size(0);
    const int64_t t = x.size(1);
    const int64_t c = x.size(2);
    const int64_t ngroups = c / kHeadSize;
    const int64_t total_groups = b * t * ngroups;
    auto stream = at::cuda::getCurrentCUDAStream();
    tmix_lnx_rkvres_backward_kernel<<<static_cast<int>(total_groups), 32, 0, stream>>>(
        grad_y.data_ptr<at::BFloat16>(),
        x.data_ptr<at::BFloat16>(),
        r.data_ptr<at::BFloat16>(),
        k.data_ptr<at::BFloat16>(),
        v.data_ptr<at::BFloat16>(),
        r_k.data_ptr<at::BFloat16>(),
        weight.data_ptr<at::BFloat16>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        scale.data_ptr<float>(),
        grad_x.data_ptr<at::BFloat16>(),
        grad_r.data_ptr<at::BFloat16>(),
        grad_k.data_ptr<at::BFloat16>(),
        grad_v.data_ptr<at::BFloat16>(),
        grad_r_k_fp32.data_ptr<float>(),
        grad_weight_fp32.data_ptr<float>(),
        grad_bias_fp32.data_ptr<float>(),
        ngroups,
        b * t);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int threads = 256;
    const int blocks_c = static_cast<int>(ceil_div(c, static_cast<int64_t>(threads)));
    const int blocks_rk = static_cast<int>(ceil_div(r_k.numel(), static_cast<int64_t>(threads)));
    cast_float_to_bf16_kernel<<<blocks_rk, threads, 0, stream>>>(
        grad_r_k_fp32.data_ptr<float>(),
        grad_r_k.data_ptr<at::BFloat16>(),
        r_k.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cast_float_to_bf16_kernel<<<blocks_c, threads, 0, stream>>>(
        grad_weight_fp32.data_ptr<float>(),
        grad_weight.data_ptr<at::BFloat16>(),
        c);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cast_float_to_bf16_kernel<<<blocks_c, threads, 0, stream>>>(
        grad_bias_fp32.data_ptr<float>(),
        grad_bias.data_ptr<at::BFloat16>(),
        c);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_x, grad_r, grad_k, grad_v, grad_r_k, grad_weight, grad_bias};
}
