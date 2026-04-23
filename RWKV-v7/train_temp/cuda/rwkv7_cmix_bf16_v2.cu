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

__global__ void cmix_mix_forward_kernel(
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ x_k,
    at::BFloat16* __restrict__ out,
    int64_t bt_size,
    int64_t t_size,
    int64_t c_size) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = bt_size * c_size;
    if (idx >= total) {
        return;
    }

    int64_t c = idx % c_size;
    int64_t bt = idx / c_size;
    int64_t t = bt % t_size;

    float x_now = load_bf16(x + idx);
    float x_prev = 0.0f;
    if (t > 0) {
        x_prev = load_bf16(x + idx - c_size);
    }
    float mix = load_bf16(x_k + c);
    store_bf16(out + idx, x_now + (x_prev - x_now) * mix);
}

__global__ void cmix_mix_backward_dx_kernel(
    const at::BFloat16* __restrict__ grad_out,
    const at::BFloat16* __restrict__ x_k,
    at::BFloat16* __restrict__ grad_x,
    int64_t bt_size,
    int64_t t_size,
    int64_t c_size) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = bt_size * c_size;
    if (idx >= total) {
        return;
    }

    int64_t c = idx % c_size;
    int64_t bt = idx / c_size;
    int64_t t = bt % t_size;

    float mix = load_bf16(x_k + c);
    float grad = load_bf16(grad_out + idx) * (1.0f - mix);
    if (t + 1 < t_size) {
        grad += load_bf16(grad_out + idx + c_size) * mix;
    }
    store_bf16(grad_x + idx, grad);
}

__global__ void cmix_mix_backward_dxk_kernel(
    const at::BFloat16* __restrict__ grad_out,
    const at::BFloat16* __restrict__ x,
    float* __restrict__ grad_x_k,
    int64_t bt_size,
    int64_t t_size,
    int64_t c_size) {
    int64_t c = blockIdx.x;
    if (c >= c_size) {
        return;
    }

    __shared__ float shared[256];
    float sum = 0.0f;
    for (int64_t bt = threadIdx.x; bt < bt_size; bt += blockDim.x) {
        int64_t idx = bt * c_size + c;
        int64_t t = bt % t_size;
        float x_now = load_bf16(x + idx);
        float x_prev = 0.0f;
        if (t > 0) {
            x_prev = load_bf16(x + idx - c_size);
        }
        float grad = load_bf16(grad_out + idx);
        sum += grad * (x_prev - x_now);
    }

    shared[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        grad_x_k[c] = shared[0];
    }
}

__global__ void cmix_mix_backward_dxk_vec2_kernel(
    const at::BFloat16* __restrict__ grad_out,
    const at::BFloat16* __restrict__ x,
    float* __restrict__ grad_x_k,
    int64_t bt_size,
    int64_t t_size,
    int64_t c_size) {
    int64_t c_pair = blockIdx.x;
    int64_t c = c_pair * 2;
    if (c + 1 >= c_size) {
        return;
    }

    __shared__ float shared0[256];
    __shared__ float shared1[256];

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int64_t bt = threadIdx.x; bt < bt_size; bt += blockDim.x) {
        int64_t idx = bt * c_size + c;
        int64_t t = bt % t_size;
        float2 x_now = __bfloat1622float2(load_bf16x2(x + idx));
        float2 x_prev = make_float2(0.0f, 0.0f);
        if (t > 0) {
            x_prev = __bfloat1622float2(load_bf16x2(x + idx - c_size));
        }
        float2 grad = __bfloat1622float2(load_bf16x2(grad_out + idx));
        sum0 += grad.x * (x_prev.x - x_now.x);
        sum1 += grad.y * (x_prev.y - x_now.y);
    }

    shared0[threadIdx.x] = sum0;
    shared1[threadIdx.x] = sum1;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared0[threadIdx.x] += shared0[threadIdx.x + stride];
            shared1[threadIdx.x] += shared1[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        grad_x_k[c] = shared0[0];
        grad_x_k[c + 1] = shared1[0];
    }
}

__global__ void cast_float_to_bf16_kernel(
    const float* __restrict__ src,
    at::BFloat16* __restrict__ dst,
    int64_t size) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    store_bf16(dst + idx, src[idx]);
}

__global__ void cast_float_to_bf16_vec2_kernel(
    const float* __restrict__ src,
    at::BFloat16* __restrict__ dst,
    int64_t total_pairs) {
    int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) {
        return;
    }
    int64_t idx = pair_idx * 2;
    store_bf16x2(dst + idx, __floats2bfloat162_rn(src[idx], src[idx + 1]));
}

__global__ void relu_sq_inplace_kernel(at::BFloat16* __restrict__ x, int64_t total) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    float value = load_bf16(x + idx);
    float relu = value > 0.0f ? value : 0.0f;
    store_bf16(x + idx, relu * relu);
}

__global__ void relu_sq_backward_from_output_inplace_kernel(
    at::BFloat16* __restrict__ grad_out,
    const at::BFloat16* __restrict__ out,
    int64_t total) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    float act = load_bf16(out + idx);
    float grad = load_bf16(grad_out + idx) * (2.0f * sqrtf(fmaxf(act, 0.0f)));
    store_bf16(grad_out + idx, grad);
}

__global__ void cmix_mix_forward_vec2_kernel(
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ x_k,
    at::BFloat16* __restrict__ out,
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
    __nv_bfloat162 mix = load_bf16x2(x_k + c);
    store_bf16x2(out + idx, __hadd2(x_now, __hmul2(__hsub2(x_prev, x_now), mix)));
}

__global__ void cmix_mix_backward_dx_vec2_kernel(
    const at::BFloat16* __restrict__ grad_out,
    const at::BFloat16* __restrict__ x_k,
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

    __nv_bfloat162 mix = load_bf16x2(x_k + c);
    __nv_bfloat162 grad = __hmul2(load_bf16x2(grad_out + idx), __hsub2(__floats2bfloat162_rn(1.0f, 1.0f), mix));
    if (t + 1 < t_size) {
        grad = __hadd2(grad, __hmul2(load_bf16x2(grad_out + idx + c_size), mix));
    }
    store_bf16x2(grad_x + idx, grad);
}

__global__ void relu_sq_inplace_vec2_kernel(at::BFloat16* __restrict__ x, int64_t total_pairs) {
    int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) {
        return;
    }

    int64_t idx = pair_idx * 2;
    float2 value = __bfloat1622float2(load_bf16x2(x + idx));
    float x0 = value.x > 0.0f ? value.x : 0.0f;
    float x1 = value.y > 0.0f ? value.y : 0.0f;
    store_bf16x2(x + idx, __floats2bfloat162_rn(x0 * x0, x1 * x1));
}

__global__ void relu_sq_backward_from_output_inplace_vec2_kernel(
    at::BFloat16* __restrict__ grad_out,
    const at::BFloat16* __restrict__ out,
    int64_t total_pairs) {
    int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) {
        return;
    }

    int64_t idx = pair_idx * 2;
    float2 grad_v = __bfloat1622float2(load_bf16x2(grad_out + idx));
    float2 act_v = __bfloat1622float2(load_bf16x2(out + idx));
    float g0 = grad_v.x * (2.0f * sqrtf(fmaxf(act_v.x, 0.0f)));
    float g1 = grad_v.y * (2.0f * sqrtf(fmaxf(act_v.y, 0.0f)));
    store_bf16x2(grad_out + idx, __floats2bfloat162_rn(g0, g1));
}

torch::Tensor cmix_mix_forward_cuda(torch::Tensor x, torch::Tensor x_k) {
    auto out = torch::empty_like(x);
    const int threads = 256;
    const int64_t bt_size = x.size(0) * x.size(1);
    const int64_t c_size = x.size(2);
    const int64_t total = bt_size * c_size;
    auto stream = at::cuda::getCurrentCUDAStream();
    if ((c_size % 2) == 0) {
        const int64_t total_pairs = bt_size * (c_size / 2);
        const int blocks = static_cast<int>(ceil_div(total_pairs, static_cast<int64_t>(threads)));
        cmix_mix_forward_vec2_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<at::BFloat16>(),
            x_k.data_ptr<at::BFloat16>(),
            out.data_ptr<at::BFloat16>(),
            bt_size,
            x.size(1),
            c_size);
    } else {
        const int blocks = static_cast<int>(ceil_div(total, static_cast<int64_t>(threads)));
        cmix_mix_forward_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<at::BFloat16>(),
            x_k.data_ptr<at::BFloat16>(),
            out.data_ptr<at::BFloat16>(),
            bt_size,
            x.size(1),
            c_size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> cmix_mix_backward_cuda(torch::Tensor grad_out, torch::Tensor x, torch::Tensor x_k) {
    auto grad_x = torch::empty_like(x);
    auto grad_x_k_fp32 = torch::empty({x.size(2)}, x.options().dtype(torch::kFloat32));
    auto grad_x_k = torch::empty({x.size(2)}, x.options());

    const int threads = 256;
    const int64_t bt_size = x.size(0) * x.size(1);
    const int64_t c_size = x.size(2);
    const int64_t total = bt_size * c_size;
    auto stream = at::cuda::getCurrentCUDAStream();

    if ((c_size % 2) == 0) {
        const int64_t total_pairs = bt_size * (c_size / 2);
        const int blocks = static_cast<int>(ceil_div(total_pairs, static_cast<int64_t>(threads)));
        cmix_mix_backward_dx_vec2_kernel<<<blocks, threads, 0, stream>>>(
            grad_out.data_ptr<at::BFloat16>(),
            x_k.data_ptr<at::BFloat16>(),
            grad_x.data_ptr<at::BFloat16>(),
            bt_size,
            x.size(1),
            c_size);
    } else {
        const int blocks = static_cast<int>(ceil_div(total, static_cast<int64_t>(threads)));
        cmix_mix_backward_dx_kernel<<<blocks, threads, 0, stream>>>(
            grad_out.data_ptr<at::BFloat16>(),
            x_k.data_ptr<at::BFloat16>(),
            grad_x.data_ptr<at::BFloat16>(),
            bt_size,
            x.size(1),
            c_size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if ((c_size % 2) == 0) {
        cmix_mix_backward_dxk_vec2_kernel<<<static_cast<int>(c_size / 2), threads, 0, stream>>>(
            grad_out.data_ptr<at::BFloat16>(),
            x.data_ptr<at::BFloat16>(),
            grad_x_k_fp32.data_ptr<float>(),
            bt_size,
            x.size(1),
            c_size);
    } else {
        cmix_mix_backward_dxk_kernel<<<static_cast<int>(c_size), threads, 0, stream>>>(
            grad_out.data_ptr<at::BFloat16>(),
            x.data_ptr<at::BFloat16>(),
            grad_x_k_fp32.data_ptr<float>(),
            bt_size,
            x.size(1),
            c_size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if ((c_size % 2) == 0) {
        const int64_t total_pairs = c_size / 2;
        const int dxk_blocks = static_cast<int>(ceil_div(total_pairs, static_cast<int64_t>(threads)));
        cast_float_to_bf16_vec2_kernel<<<dxk_blocks, threads, 0, stream>>>(
            grad_x_k_fp32.data_ptr<float>(),
            grad_x_k.data_ptr<at::BFloat16>(),
            total_pairs);
    } else {
        const int dxk_blocks = static_cast<int>(ceil_div(c_size, static_cast<int64_t>(threads)));
        cast_float_to_bf16_kernel<<<dxk_blocks, threads, 0, stream>>>(
            grad_x_k_fp32.data_ptr<float>(),
            grad_x_k.data_ptr<at::BFloat16>(),
            c_size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_x, grad_x_k};
}

void relu_sq_inplace_cuda(torch::Tensor x) {
    const int threads = 256;
    const int64_t total = x.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    if ((total % 2) == 0) {
        const int64_t total_pairs = total / 2;
        const int blocks = static_cast<int>(ceil_div(total_pairs, static_cast<int64_t>(threads)));
        relu_sq_inplace_vec2_kernel<<<blocks, threads, 0, stream>>>(x.data_ptr<at::BFloat16>(), total_pairs);
    } else {
        const int blocks = static_cast<int>(ceil_div(total, static_cast<int64_t>(threads)));
        relu_sq_inplace_kernel<<<blocks, threads, 0, stream>>>(x.data_ptr<at::BFloat16>(), total);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void relu_sq_backward_from_output_inplace_cuda(torch::Tensor grad_out, torch::Tensor out) {
    const int threads = 256;
    const int64_t total = out.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    if ((total % 2) == 0) {
        const int64_t total_pairs = total / 2;
        const int blocks = static_cast<int>(ceil_div(total_pairs, static_cast<int64_t>(threads)));
        relu_sq_backward_from_output_inplace_vec2_kernel<<<blocks, threads, 0, stream>>>(
            grad_out.data_ptr<at::BFloat16>(),
            out.data_ptr<at::BFloat16>(),
            total_pairs);
    } else {
        const int blocks = static_cast<int>(ceil_div(total, static_cast<int64_t>(threads)));
        relu_sq_backward_from_output_inplace_kernel<<<blocks, threads, 0, stream>>>(
            grad_out.data_ptr<at::BFloat16>(),
            out.data_ptr<at::BFloat16>(),
            total);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

std::vector<torch::Tensor> cmix_layer_forward_v2_cuda(
    torch::Tensor x,
    torch::Tensor x_k,
    torch::Tensor key_weight,
    torch::Tensor value_weight) {
    auto mixed = cmix_mix_forward_cuda(x, x_k);
    auto mixed_2d = mixed.view({-1, mixed.size(2)});
    auto act = at::matmul(mixed_2d, key_weight.transpose(0, 1)).contiguous();
    relu_sq_inplace_cuda(act);
    auto out_2d = at::matmul(act, value_weight.transpose(0, 1));
    auto out = out_2d.view({x.size(0), x.size(1), value_weight.size(0)}).contiguous();
    return {out, mixed, act};
}

std::vector<torch::Tensor> cmix_layer_backward_v2_cuda(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor x_k,
    torch::Tensor key_weight,
    torch::Tensor value_weight,
    torch::Tensor mixed,
    torch::Tensor act) {
    auto grad_out_2d = grad_out.view({-1, grad_out.size(2)}).contiguous();
    auto mixed_2d = mixed.view({-1, mixed.size(2)}).contiguous();

    auto grad_value_weight = at::matmul(grad_out_2d.transpose(0, 1), act);
    auto grad_hidden = at::matmul(grad_out_2d, value_weight).contiguous();
    relu_sq_backward_from_output_inplace_cuda(grad_hidden, act);
    auto grad_key_weight = at::matmul(grad_hidden.transpose(0, 1), mixed_2d);
    auto grad_mixed_2d = at::matmul(grad_hidden, key_weight).contiguous();
    auto grad_mixed = grad_mixed_2d.view_as(x);
    auto mix_grads = cmix_mix_backward_cuda(grad_mixed, x, x_k);

    return {
        mix_grads[0],
        mix_grads[1],
        grad_key_weight.contiguous(),
        grad_value_weight.contiguous(),
    };
}
