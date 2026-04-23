#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace {

constexpr float kTorchNormalizeDefaultEps = 1.0e-12f;

__device__ inline float load_bf16(const at::BFloat16* ptr) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(ptr));
}

__device__ inline void store_bf16(at::BFloat16* ptr, float value) {
    *reinterpret_cast<__nv_bfloat16*>(ptr) = __float2bfloat16(value);
}

__device__ inline __nv_bfloat162 load_bf16x2(const at::BFloat16* ptr) {
    return *reinterpret_cast<const __nv_bfloat162*>(ptr);
}

__device__ inline void store_bf16x2(at::BFloat16* ptr, float x0, float x1) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) = __floats2bfloat162_rn(x0, x1);
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

inline int choose_threads(int64_t head_size) {
    int threads = 32;
    while (threads < head_size && threads < 256) {
        threads <<= 1;
    }
    return threads;
}

__global__ void tmix_kk_pre_forward64_kernel(
    const at::BFloat16* __restrict__ k,
    const at::BFloat16* __restrict__ k_k,
    const at::BFloat16* __restrict__ a,
    const at::BFloat16* __restrict__ k_a,
    at::BFloat16* __restrict__ new_k,
    at::BFloat16* __restrict__ neg_kk,
    at::BFloat16* __restrict__ kka,
    float* __restrict__ denom,
    int64_t bth_size,
    int64_t h_size) {
    const int64_t bth = static_cast<int64_t>(blockIdx.x);
    if (bth >= bth_size) {
        return;
    }

    const int lane = threadIdx.x;
    const int64_t h = bth % h_size;
    const int64_t base = bth * 64;
    const int64_t c0 = h * 64;
    const int64_t idx = base + static_cast<int64_t>(lane) * 2;
    const int64_t c = c0 + static_cast<int64_t>(lane) * 2;

    const __nv_bfloat162 kv2 = load_bf16x2(k + idx);
    const __nv_bfloat162 kk_scale2 = load_bf16x2(k_k + c);
    const float kv0 = __low2float(kv2);
    const float kv1 = __high2float(kv2);
    const float ks0 = __low2float(kk_scale2);
    const float ks1 = __high2float(kk_scale2);
    const float u0 = kv0 * ks0;
    const float u1 = kv1 * ks1;

    float sum_sq = u0 * u0 + u1 * u1;
    sum_sq = warp_sum(sum_sq);
    const float total_sum_sq = __shfl_sync(0xffffffffu, sum_sq, 0);
    const float norm = sqrtf(total_sum_sq);
    const float d = fmaxf(norm, kTorchNormalizeDefaultEps);
    if (lane == 0) {
        denom[bth] = d;
    }

    const __nv_bfloat162 av2 = load_bf16x2(a + idx);
    const __nv_bfloat162 ka_scale2 = load_bf16x2(k_a + c);
    const float av0 = __low2float(av2);
    const float av1 = __high2float(av2);
    const float ka0 = __low2float(ka_scale2);
    const float ka1 = __high2float(ka_scale2);

    const float new_k0 = kv0 * (1.0f + (av0 - 1.0f) * ka0);
    const float new_k1 = kv1 * (1.0f + (av1 - 1.0f) * ka1);
    const float kk0 = u0 / d;
    const float kk1 = u1 / d;
    const float kka0 = kk0 * av0;
    const float kka1 = kk1 * av1;

    store_bf16x2(new_k + idx, new_k0, new_k1);
    store_bf16x2(neg_kk + idx, -kk0, -kk1);
    store_bf16x2(kka + idx, kka0, kka1);
}

__global__ void tmix_kk_pre_backward64_kernel(
    const at::BFloat16* __restrict__ grad_new_k,
    const at::BFloat16* __restrict__ grad_neg_kk,
    const at::BFloat16* __restrict__ grad_kka,
    const at::BFloat16* __restrict__ k,
    const at::BFloat16* __restrict__ k_k,
    const at::BFloat16* __restrict__ a,
    const at::BFloat16* __restrict__ k_a,
    const at::BFloat16* __restrict__ neg_kk,
    const float* __restrict__ denom,
    at::BFloat16* __restrict__ grad_k,
    float* __restrict__ grad_k_k,
    at::BFloat16* __restrict__ grad_a,
    float* __restrict__ grad_k_a,
    int64_t bth_size,
    int64_t h_size) {
    const int64_t bth = static_cast<int64_t>(blockIdx.x);
    if (bth >= bth_size) {
        return;
    }

    const int lane = threadIdx.x;
    const int64_t h = bth % h_size;
    const int64_t base = bth * 64;
    const int64_t c0 = h * 64;
    const int64_t idx = base + static_cast<int64_t>(lane) * 2;
    const int64_t c = c0 + static_cast<int64_t>(lane) * 2;
    const float d = denom[bth];
    const bool use_norm_branch = d > kTorchNormalizeDefaultEps;

    const __nv_bfloat162 av2 = load_bf16x2(a + idx);
    const __nv_bfloat162 neg_kk2 = load_bf16x2(neg_kk + idx);
    const __nv_bfloat162 gneg_kk2 = load_bf16x2(grad_neg_kk + idx);
    const __nv_bfloat162 gkka2 = load_bf16x2(grad_kka + idx);

    const float av0 = __low2float(av2);
    const float av1 = __high2float(av2);
    const float kk0 = -__low2float(neg_kk2);
    const float kk1 = -__high2float(neg_kk2);
    const float gkk_total0 = -__low2float(gneg_kk2) + __low2float(gkka2) * av0;
    const float gkk_total1 = -__high2float(gneg_kk2) + __high2float(gkka2) * av1;

    float dot = gkk_total0 * kk0 + gkk_total1 * kk1;
    dot = warp_sum(dot);
    const float dot_total = __shfl_sync(0xffffffffu, dot, 0);

    const __nv_bfloat162 kv2 = load_bf16x2(k + idx);
    const __nv_bfloat162 kk_scale2 = load_bf16x2(k_k + c);
    const __nv_bfloat162 ka_scale2 = load_bf16x2(k_a + c);
    const __nv_bfloat162 gnew2 = load_bf16x2(grad_new_k + idx);

    const float kv0 = __low2float(kv2);
    const float kv1 = __high2float(kv2);
    const float ks0 = __low2float(kk_scale2);
    const float ks1 = __high2float(kk_scale2);
    const float ka0 = __low2float(ka_scale2);
    const float ka1 = __high2float(ka_scale2);
    const float gnew0 = __low2float(gnew2);
    const float gnew1 = __high2float(gnew2);
    const float gkka0 = __low2float(gkka2);
    const float gkka1 = __high2float(gkka2);

    float gu0 = gkk_total0 / d;
    float gu1 = gkk_total1 / d;
    if (use_norm_branch) {
        gu0 = (gkk_total0 - kk0 * dot_total) / d;
        gu1 = (gkk_total1 - kk1 * dot_total) / d;
    }

    const float scale0 = 1.0f + (av0 - 1.0f) * ka0;
    const float scale1 = 1.0f + (av1 - 1.0f) * ka1;
    const float grad_k0 = gnew0 * scale0 + gu0 * ks0;
    const float grad_k1 = gnew1 * scale1 + gu1 * ks1;
    const float grad_a0 = gnew0 * kv0 * ka0 + gkka0 * kk0;
    const float grad_a1 = gnew1 * kv1 * ka1 + gkka1 * kk1;
    const float grad_k_k0 = gu0 * kv0;
    const float grad_k_k1 = gu1 * kv1;
    const float grad_k_a0 = gnew0 * kv0 * (av0 - 1.0f);
    const float grad_k_a1 = gnew1 * kv1 * (av1 - 1.0f);

    store_bf16x2(grad_k + idx, grad_k0, grad_k1);
    store_bf16x2(grad_a + idx, grad_a0, grad_a1);
    atomicAdd(grad_k_k + c + 0, grad_k_k0);
    atomicAdd(grad_k_k + c + 1, grad_k_k1);
    atomicAdd(grad_k_a + c + 0, grad_k_a0);
    atomicAdd(grad_k_a + c + 1, grad_k_a1);
}

__global__ void tmix_kk_pre_forward_generic_kernel(
    const at::BFloat16* __restrict__ k,
    const at::BFloat16* __restrict__ k_k,
    const at::BFloat16* __restrict__ a,
    const at::BFloat16* __restrict__ k_a,
    at::BFloat16* __restrict__ new_k,
    at::BFloat16* __restrict__ neg_kk,
    at::BFloat16* __restrict__ kka,
    float* __restrict__ denom,
    int64_t bth_size,
    int64_t h_size,
    int64_t head_size) {
    const int64_t bth = static_cast<int64_t>(blockIdx.x);
    if (bth >= bth_size) {
        return;
    }

    const int64_t h = bth % h_size;
    const int64_t c0 = h * head_size;
    const int64_t base = bth * head_size;

    __shared__ float shared_sum[256];
    float sum_sq = 0.0f;
    for (int64_t n = threadIdx.x; n < head_size; n += blockDim.x) {
        const int64_t idx = base + n;
        const int64_t c = c0 + n;
        const float u = load_bf16(k + idx) * load_bf16(k_k + c);
        sum_sq += u * u;
    }
    shared_sum[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float norm = sqrtf(shared_sum[0]);
    const float d = fmaxf(norm, kTorchNormalizeDefaultEps);
    if (threadIdx.x == 0) {
        denom[bth] = d;
    }

    for (int64_t n = threadIdx.x; n < head_size; n += blockDim.x) {
        const int64_t idx = base + n;
        const int64_t c = c0 + n;
        const float kv = load_bf16(k + idx);
        const float av = load_bf16(a + idx);
        const float kk_scale = load_bf16(k_k + c);
        const float ka_scale = load_bf16(k_a + c);
        const float u = kv * kk_scale;
        const float kkv = u / d;
        const float new_kv = kv * (1.0f + (av - 1.0f) * ka_scale);
        store_bf16(new_k + idx, new_kv);
        store_bf16(neg_kk + idx, -kkv);
        store_bf16(kka + idx, kkv * av);
    }
}

__global__ void tmix_kk_pre_backward_generic_kernel(
    const at::BFloat16* __restrict__ grad_new_k,
    const at::BFloat16* __restrict__ grad_neg_kk,
    const at::BFloat16* __restrict__ grad_kka,
    const at::BFloat16* __restrict__ k,
    const at::BFloat16* __restrict__ k_k,
    const at::BFloat16* __restrict__ a,
    const at::BFloat16* __restrict__ k_a,
    const at::BFloat16* __restrict__ neg_kk,
    const float* __restrict__ denom,
    at::BFloat16* __restrict__ grad_k,
    float* __restrict__ grad_k_k,
    at::BFloat16* __restrict__ grad_a,
    float* __restrict__ grad_k_a,
    int64_t bth_size,
    int64_t h_size,
    int64_t head_size) {
    const int64_t bth = static_cast<int64_t>(blockIdx.x);
    if (bth >= bth_size) {
        return;
    }

    const int64_t h = bth % h_size;
    const int64_t c0 = h * head_size;
    const int64_t base = bth * head_size;
    const float d = denom[bth];
    const bool use_norm_branch = d > kTorchNormalizeDefaultEps;

    __shared__ float shared_dot[256];
    float dot = 0.0f;
    for (int64_t n = threadIdx.x; n < head_size; n += blockDim.x) {
        const int64_t idx = base + n;
        const float av = load_bf16(a + idx);
        const float kk_val = -load_bf16(neg_kk + idx);
        const float gkk = -load_bf16(grad_neg_kk + idx) + load_bf16(grad_kka + idx) * av;
        dot += gkk * kk_val;
    }
    shared_dot[threadIdx.x] = dot;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_dot[threadIdx.x] += shared_dot[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float dot_total = shared_dot[0];
    for (int64_t n = threadIdx.x; n < head_size; n += blockDim.x) {
        const int64_t idx = base + n;
        const int64_t c = c0 + n;
        const float kv = load_bf16(k + idx);
        const float av = load_bf16(a + idx);
        const float kk_scale = load_bf16(k_k + c);
        const float ka_scale = load_bf16(k_a + c);
        const float kkv = -load_bf16(neg_kk + idx);
        const float g_new_k = load_bf16(grad_new_k + idx);
        const float g_kka = load_bf16(grad_kka + idx);
        const float gkk = -load_bf16(grad_neg_kk + idx) + g_kka * av;

        float g_u = gkk / d;
        if (use_norm_branch) {
            g_u = (gkk - kkv * dot_total) / d;
        }

        const float scale = 1.0f + (av - 1.0f) * ka_scale;
        const float g_k = g_new_k * scale + g_u * kk_scale;
        const float g_a = g_new_k * kv * ka_scale + g_kka * kkv;
        const float g_k_k = g_u * kv;
        const float g_k_a = g_new_k * kv * (av - 1.0f);

        store_bf16(grad_k + idx, g_k);
        store_bf16(grad_a + idx, g_a);
        atomicAdd(grad_k_k + c, g_k_k);
        atomicAdd(grad_k_a + c, g_k_a);
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

std::vector<torch::Tensor> tmix_kk_pre_v2_forward_cuda(
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a,
    torch::Tensor k_a,
    int64_t head_size) {
    auto new_k = torch::empty_like(k);
    auto neg_kk = torch::empty_like(k);
    auto kka = torch::empty_like(k);
    auto denom = torch::empty({k.size(0), k.size(1), k.size(2) / head_size}, k.options().dtype(torch::kFloat32));

    const int64_t bth_size = k.size(0) * k.size(1) * (k.size(2) / head_size);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (head_size == 64) {
        tmix_kk_pre_forward64_kernel<<<static_cast<int>(bth_size), 32, 0, stream>>>(
            k.data_ptr<at::BFloat16>(),
            k_k.data_ptr<at::BFloat16>(),
            a.data_ptr<at::BFloat16>(),
            k_a.data_ptr<at::BFloat16>(),
            new_k.data_ptr<at::BFloat16>(),
            neg_kk.data_ptr<at::BFloat16>(),
            kka.data_ptr<at::BFloat16>(),
            denom.data_ptr<float>(),
            bth_size,
            k.size(2) / head_size);
    } else {
        const int threads = choose_threads(head_size);
        tmix_kk_pre_forward_generic_kernel<<<static_cast<int>(bth_size), threads, 0, stream>>>(
            k.data_ptr<at::BFloat16>(),
            k_k.data_ptr<at::BFloat16>(),
            a.data_ptr<at::BFloat16>(),
            k_a.data_ptr<at::BFloat16>(),
            new_k.data_ptr<at::BFloat16>(),
            neg_kk.data_ptr<at::BFloat16>(),
            kka.data_ptr<at::BFloat16>(),
            denom.data_ptr<float>(),
            bth_size,
            k.size(2) / head_size,
            head_size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {new_k, neg_kk, kka, denom};
}

std::vector<torch::Tensor> tmix_kk_pre_v2_backward_cuda(
    torch::Tensor grad_new_k,
    torch::Tensor grad_neg_kk,
    torch::Tensor grad_kka,
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a,
    torch::Tensor k_a,
    torch::Tensor neg_kk,
    torch::Tensor denom,
    int64_t head_size) {
    auto grad_k = torch::empty_like(k);
    auto grad_a = torch::empty_like(a);
    auto grad_k_k_fp32 = torch::zeros({k.size(2)}, k.options().dtype(torch::kFloat32));
    auto grad_k_a_fp32 = torch::zeros({k.size(2)}, k.options().dtype(torch::kFloat32));
    auto grad_k_k = torch::empty_like(k_k);
    auto grad_k_a = torch::empty_like(k_a);

    const int64_t bth_size = k.size(0) * k.size(1) * (k.size(2) / head_size);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (head_size == 64) {
        tmix_kk_pre_backward64_kernel<<<static_cast<int>(bth_size), 32, 0, stream>>>(
            grad_new_k.data_ptr<at::BFloat16>(),
            grad_neg_kk.data_ptr<at::BFloat16>(),
            grad_kka.data_ptr<at::BFloat16>(),
            k.data_ptr<at::BFloat16>(),
            k_k.data_ptr<at::BFloat16>(),
            a.data_ptr<at::BFloat16>(),
            k_a.data_ptr<at::BFloat16>(),
            neg_kk.data_ptr<at::BFloat16>(),
            denom.data_ptr<float>(),
            grad_k.data_ptr<at::BFloat16>(),
            grad_k_k_fp32.data_ptr<float>(),
            grad_a.data_ptr<at::BFloat16>(),
            grad_k_a_fp32.data_ptr<float>(),
            bth_size,
            k.size(2) / head_size);
    } else {
        const int threads = choose_threads(head_size);
        tmix_kk_pre_backward_generic_kernel<<<static_cast<int>(bth_size), threads, 0, stream>>>(
            grad_new_k.data_ptr<at::BFloat16>(),
            grad_neg_kk.data_ptr<at::BFloat16>(),
            grad_kka.data_ptr<at::BFloat16>(),
            k.data_ptr<at::BFloat16>(),
            k_k.data_ptr<at::BFloat16>(),
            a.data_ptr<at::BFloat16>(),
            k_a.data_ptr<at::BFloat16>(),
            neg_kk.data_ptr<at::BFloat16>(),
            denom.data_ptr<float>(),
            grad_k.data_ptr<at::BFloat16>(),
            grad_k_k_fp32.data_ptr<float>(),
            grad_a.data_ptr<at::BFloat16>(),
            grad_k_a_fp32.data_ptr<float>(),
            bth_size,
            k.size(2) / head_size,
            head_size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int64_t total = k.size(2);
    const int threads = 256;
    const int blocks = static_cast<int>(ceil_div(total, static_cast<int64_t>(threads)));
    cast_float_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
        grad_k_k_fp32.data_ptr<float>(),
        grad_k_k.data_ptr<at::BFloat16>(),
        total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cast_float_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
        grad_k_a_fp32.data_ptr<float>(),
        grad_k_a.data_ptr<at::BFloat16>(),
        total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_k, grad_k_k, grad_a, grad_k_a};
}
