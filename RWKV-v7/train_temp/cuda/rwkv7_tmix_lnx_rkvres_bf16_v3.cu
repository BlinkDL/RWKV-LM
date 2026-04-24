#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kHeadSize = 64;
constexpr int kWarpSize = 32;
constexpr int kRowsPerBlock = 16;
constexpr int kThreads = kWarpSize * kRowsPerBlock;
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

__global__ void tmix_lnx_rkvres_v3_forward_kernel(
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
    int64_t ngroups,
    int64_t nrows) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t group = static_cast<int64_t>(blockIdx.x);
    const int64_t row = static_cast<int64_t>(blockIdx.y) * kRowsPerBlock + warp;
    if (row >= nrows) {
        return;
    }

    const int64_t c0 = group * kHeadSize;
    const int64_t base = row * (ngroups * kHeadSize) + c0;
    const int64_t idx = base + static_cast<int64_t>(lane) * 2;
    const int64_t c = c0 + static_cast<int64_t>(lane) * 2;
    const int64_t ng = row * ngroups + group;

    const __nv_bfloat162 xv2 = load_bf16x2(x + idx);
    const float x0 = __low2float(xv2);
    const float x1 = __high2float(xv2);

    float sum = warp_sum(x0 + x1);
    const float mean_val = __shfl_sync(0xffffffffu, sum, 0) * (1.0f / kHeadSize);

    const float d0 = x0 - mean_val;
    const float d1 = x1 - mean_val;
    float sq = warp_sum(d0 * d0 + d1 * d1);
    const float var_val = __shfl_sync(0xffffffffu, sq, 0) * (1.0f / kHeadSize);
    const float rstd_val = rsqrtf(var_val + kLnXEps);

    const __nv_bfloat162 rv2 = load_bf16x2(r + idx);
    const __nv_bfloat162 kv2 = load_bf16x2(k + idx);
    const __nv_bfloat162 rk2 = load_bf16x2(r_k + c);
    float scale_sum =
        __low2float(rv2) * __low2float(kv2) * __low2float(rk2) +
        __high2float(rv2) * __high2float(kv2) * __high2float(rk2);
    scale_sum = warp_sum(scale_sum);
    const float scale_val = __shfl_sync(0xffffffffu, scale_sum, 0);

    if (lane == 0) {
        mean[ng] = mean_val;
        rstd[ng] = rstd_val;
    }

    const __nv_bfloat162 vv2 = load_bf16x2(v + idx);
    const __nv_bfloat162 w2 = load_bf16x2(weight + c);
    const __nv_bfloat162 b2 = load_bf16x2(bias + c);
    const float y0 = d0 * rstd_val * __low2float(w2) + __low2float(b2) + scale_val * __low2float(vv2);
    const float y1 = d1 * rstd_val * __high2float(w2) + __high2float(b2) + scale_val * __high2float(vv2);
    store_bf16x2(y + idx, y0, y1);
}

__global__ void tmix_lnx_rkvres_v3_backward_kernel(
    const at::BFloat16* __restrict__ grad_y,
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ r,
    const at::BFloat16* __restrict__ k,
    const at::BFloat16* __restrict__ v,
    const at::BFloat16* __restrict__ r_k,
    const at::BFloat16* __restrict__ weight,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    at::BFloat16* __restrict__ grad_x,
    at::BFloat16* __restrict__ grad_r,
    at::BFloat16* __restrict__ grad_k,
    at::BFloat16* __restrict__ grad_v,
    float* __restrict__ grad_r_k,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    int64_t ngroups,
    int64_t nrows) {
    __shared__ float s_gw0[kRowsPerBlock][kWarpSize];
    __shared__ float s_gw1[kRowsPerBlock][kWarpSize];
    __shared__ float s_gb0[kRowsPerBlock][kWarpSize];
    __shared__ float s_gb1[kRowsPerBlock][kWarpSize];
    __shared__ float s_grk0[kRowsPerBlock][kWarpSize];
    __shared__ float s_grk1[kRowsPerBlock][kWarpSize];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t group = static_cast<int64_t>(blockIdx.x);
    const int64_t row = static_cast<int64_t>(blockIdx.y) * kRowsPerBlock + warp;
    const bool valid = row < nrows;

    const int64_t c0 = group * kHeadSize;
    const int64_t c = c0 + static_cast<int64_t>(lane) * 2;
    const int64_t ng = row * ngroups + group;
    const int64_t base = row * (ngroups * kHeadSize) + c0;
    const int64_t idx = base + static_cast<int64_t>(lane) * 2;

    float gw0 = 0.0f;
    float gw1 = 0.0f;
    float gb0 = 0.0f;
    float gb1 = 0.0f;
    float grk0 = 0.0f;
    float grk1 = 0.0f;

    if (valid) {
        const float mean_val = mean[ng];
        const float rstd_val = rstd[ng];

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

        float total_dxhat = warp_sum(dxhat0 + dxhat1);
        float total_dxhat_xhat = warp_sum(dxhat0 * xhat0 + dxhat1 * xhat1);
        total_dxhat = __shfl_sync(0xffffffffu, total_dxhat, 0);
        total_dxhat_xhat = __shfl_sync(0xffffffffu, total_dxhat_xhat, 0);
        const float inv_m = 1.0f / kHeadSize;

        const float gx0 = (dxhat0 - total_dxhat * inv_m - xhat0 * total_dxhat_xhat * inv_m) * rstd_val;
        const float gx1 = (dxhat1 - total_dxhat * inv_m - xhat1 * total_dxhat_xhat * inv_m) * rstd_val;
        store_bf16x2(grad_x + idx, gx0, gx1);

        gw0 = gy0 * xhat0;
        gw1 = gy1 * xhat1;
        gb0 = gy0;
        gb1 = gy1;

        const __nv_bfloat162 rv2 = load_bf16x2(r + idx);
        const __nv_bfloat162 kv2 = load_bf16x2(k + idx);
        const __nv_bfloat162 rk2 = load_bf16x2(r_k + c);
        const float r0 = __low2float(rv2);
        const float r1 = __high2float(rv2);
        const float k0 = __low2float(kv2);
        const float k1 = __high2float(kv2);
        const float rk0 = __low2float(rk2);
        const float rk1 = __high2float(rk2);

        float scale_sum = r0 * k0 * rk0 + r1 * k1 * rk1;
        scale_sum = warp_sum(scale_sum);
        const float scale_val = __shfl_sync(0xffffffffu, scale_sum, 0);

        const float v0 = __low2float(vv2);
        const float v1 = __high2float(vv2);
        const float q = __shfl_sync(0xffffffffu, warp_sum(gy0 * v0 + gy1 * v1), 0);
        store_bf16x2(grad_v + idx, gy0 * scale_val, gy1 * scale_val);
        store_bf16x2(grad_r + idx, q * k0 * rk0, q * k1 * rk1);
        store_bf16x2(grad_k + idx, q * r0 * rk0, q * r1 * rk1);

        grk0 = q * r0 * k0;
        grk1 = q * r1 * k1;
    }

    s_gw0[warp][lane] = gw0;
    s_gw1[warp][lane] = gw1;
    s_gb0[warp][lane] = gb0;
    s_gb1[warp][lane] = gb1;
    s_grk0[warp][lane] = grk0;
    s_grk1[warp][lane] = grk1;
    __syncthreads();

    if (warp == 0) {
        float sum_gw0 = 0.0f;
        float sum_gw1 = 0.0f;
        float sum_gb0 = 0.0f;
        float sum_gb1 = 0.0f;
        float sum_grk0 = 0.0f;
        float sum_grk1 = 0.0f;
#pragma unroll
        for (int w = 0; w < kRowsPerBlock; ++w) {
            sum_gw0 += s_gw0[w][lane];
            sum_gw1 += s_gw1[w][lane];
            sum_gb0 += s_gb0[w][lane];
            sum_gb1 += s_gb1[w][lane];
            sum_grk0 += s_grk0[w][lane];
            sum_grk1 += s_grk1[w][lane];
        }
        atomicAdd(grad_weight + c + 0, sum_gw0);
        atomicAdd(grad_weight + c + 1, sum_gw1);
        atomicAdd(grad_bias + c + 0, sum_gb0);
        atomicAdd(grad_bias + c + 1, sum_gb1);
        atomicAdd(grad_r_k + c + 0, sum_grk0);
        atomicAdd(grad_r_k + c + 1, sum_grk1);
    }
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

std::vector<torch::Tensor> tmix_lnx_rkvres_v3_forward_cuda(
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

    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 blocks(static_cast<unsigned int>(ngroups), static_cast<unsigned int>(ceil_div(b * t, static_cast<int64_t>(kRowsPerBlock))));
    tmix_lnx_rkvres_v3_forward_kernel<<<blocks, kThreads, 0, stream>>>(
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
        ngroups,
        b * t);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, mean, rstd};
}

std::vector<torch::Tensor> tmix_lnx_rkvres_v3_backward_cuda(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor mean,
    torch::Tensor rstd) {
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
    const int64_t nrows = b * t;
    const int64_t ngroups = c / kHeadSize;
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 blocks(static_cast<unsigned int>(ngroups), static_cast<unsigned int>(ceil_div(nrows, static_cast<int64_t>(kRowsPerBlock))));
    tmix_lnx_rkvres_v3_backward_kernel<<<blocks, kThreads, 0, stream>>>(
        grad_y.data_ptr<at::BFloat16>(),
        x.data_ptr<at::BFloat16>(),
        r.data_ptr<at::BFloat16>(),
        k.data_ptr<at::BFloat16>(),
        v.data_ptr<at::BFloat16>(),
        r_k.data_ptr<at::BFloat16>(),
        weight.data_ptr<at::BFloat16>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        grad_x.data_ptr<at::BFloat16>(),
        grad_r.data_ptr<at::BFloat16>(),
        grad_k.data_ptr<at::BFloat16>(),
        grad_v.data_ptr<at::BFloat16>(),
        grad_r_k_fp32.data_ptr<float>(),
        grad_weight_fp32.data_ptr<float>(),
        grad_bias_fp32.data_ptr<float>(),
        ngroups,
        nrows);
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
