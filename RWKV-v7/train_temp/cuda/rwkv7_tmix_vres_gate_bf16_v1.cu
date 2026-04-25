#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <vector>

namespace {

inline int64_t ceil_div(int64_t n, int64_t d) {
    return (n + d - 1) / d;
}

inline bool is_power_of_two(int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

__device__ inline float bf16_to_float(const at::BFloat16* ptr) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(ptr));
}

__device__ inline void store_bf16(at::BFloat16* ptr, float value) {
    __nv_bfloat16 out = __float2bfloat16_rn(value);
    *reinterpret_cast<__nv_bfloat16*>(ptr) = out;
}

__device__ inline float sigmoidf_stable(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void vres_gate_forward_kernel(
    const at::BFloat16* __restrict__ v,
    const at::BFloat16* __restrict__ v_first,
    const at::BFloat16* __restrict__ v0,
    const at::BFloat16* __restrict__ v12,
    at::BFloat16* __restrict__ out,
    int64_t total,
    int64_t c_size) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int64_t c = idx % c_size;
    float vv = bf16_to_float(v + idx);
    float vf = bf16_to_float(v_first + idx);
    float pre = bf16_to_float(v0 + c) + bf16_to_float(v12 + idx);
    float gate = sigmoidf_stable(pre);
    store_bf16(out + idx, vv + (vf - vv) * gate);
}

__global__ void vres_gate_forward_pow2c_kernel(
    const at::BFloat16* __restrict__ v,
    const at::BFloat16* __restrict__ v_first,
    const at::BFloat16* __restrict__ v0,
    const at::BFloat16* __restrict__ v12,
    at::BFloat16* __restrict__ out,
    int64_t total,
    int64_t c_mask) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int64_t c = idx & c_mask;
    float vv = bf16_to_float(v + idx);
    float vf = bf16_to_float(v_first + idx);
    float pre = bf16_to_float(v0 + c) + bf16_to_float(v12 + idx);
    float gate = sigmoidf_stable(pre);
    store_bf16(out + idx, vv + (vf - vv) * gate);
}

__global__ void vres_gate_backward_kernel(
    const at::BFloat16* __restrict__ grad_out,
    const at::BFloat16* __restrict__ v,
    const at::BFloat16* __restrict__ v_first,
    const at::BFloat16* __restrict__ v0,
    const at::BFloat16* __restrict__ v12,
    at::BFloat16* __restrict__ grad_v,
    at::BFloat16* __restrict__ grad_v_first,
    float* __restrict__ grad_pre,
    int64_t total,
    int64_t c_size) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int64_t c = idx % c_size;
    float go = bf16_to_float(grad_out + idx);
    float vv = bf16_to_float(v + idx);
    float vf = bf16_to_float(v_first + idx);
    float pre = bf16_to_float(v0 + c) + bf16_to_float(v12 + idx);
    float gate = sigmoidf_stable(pre);
    float grad_vf = go * gate;
    float grad_vv = go - grad_vf;
    float gp = go * (vf - vv) * gate * (1.0f - gate);

    store_bf16(grad_v + idx, grad_vv);
    store_bf16(grad_v_first + idx, grad_vf);
    grad_pre[idx] = gp;
}

__global__ void vres_gate_backward_pow2c_kernel(
    const at::BFloat16* __restrict__ grad_out,
    const at::BFloat16* __restrict__ v,
    const at::BFloat16* __restrict__ v_first,
    const at::BFloat16* __restrict__ v0,
    const at::BFloat16* __restrict__ v12,
    at::BFloat16* __restrict__ grad_v,
    at::BFloat16* __restrict__ grad_v_first,
    float* __restrict__ grad_pre,
    int64_t total,
    int64_t c_mask) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int64_t c = idx & c_mask;
    float go = bf16_to_float(grad_out + idx);
    float vv = bf16_to_float(v + idx);
    float vf = bf16_to_float(v_first + idx);
    float pre = bf16_to_float(v0 + c) + bf16_to_float(v12 + idx);
    float gate = sigmoidf_stable(pre);
    float grad_vf = go * gate;
    float grad_vv = go - grad_vf;
    float gp = go * (vf - vv) * gate * (1.0f - gate);

    store_bf16(grad_v + idx, grad_vv);
    store_bf16(grad_v_first + idx, grad_vf);
    grad_pre[idx] = gp;
}

} // namespace

torch::Tensor tmix_vres_gate_forward_cuda(
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12) {
    auto out = torch::empty_like(v);
    int64_t total = v.numel();
    int64_t c_size = v.size(2);
    int threads = 128;
    if (const char* env = std::getenv("VRES_GATE_THREADS")) {
        threads = std::atoi(env);
    }
    int blocks = static_cast<int>(ceil_div(total, static_cast<int64_t>(threads)));
    auto stream = at::cuda::getCurrentCUDAStream();
    if (is_power_of_two(c_size)) {
        vres_gate_forward_pow2c_kernel<<<blocks, threads, 0, stream>>>(
            v.data_ptr<at::BFloat16>(),
            v_first.data_ptr<at::BFloat16>(),
            v0.data_ptr<at::BFloat16>(),
            v12.data_ptr<at::BFloat16>(),
            out.data_ptr<at::BFloat16>(),
            total,
            c_size - 1);
    } else {
        vres_gate_forward_kernel<<<blocks, threads, 0, stream>>>(
            v.data_ptr<at::BFloat16>(),
            v_first.data_ptr<at::BFloat16>(),
            v0.data_ptr<at::BFloat16>(),
            v12.data_ptr<at::BFloat16>(),
            out.data_ptr<at::BFloat16>(),
            total,
            c_size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> tmix_vres_gate_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12) {
    auto grad_v = torch::empty_like(v);
    auto grad_v_first = torch::empty_like(v_first);
    auto grad_pre = torch::empty(v.sizes(), v.options().dtype(torch::kFloat32));
    int64_t total = v.numel();
    int64_t c_size = v.size(2);
    int threads = 128;
    if (const char* env = std::getenv("VRES_GATE_THREADS")) {
        threads = std::atoi(env);
    }
    int blocks = static_cast<int>(ceil_div(total, static_cast<int64_t>(threads)));
    auto stream = at::cuda::getCurrentCUDAStream();
    if (is_power_of_two(c_size)) {
        vres_gate_backward_pow2c_kernel<<<blocks, threads, 0, stream>>>(
            grad_out.data_ptr<at::BFloat16>(),
            v.data_ptr<at::BFloat16>(),
            v_first.data_ptr<at::BFloat16>(),
            v0.data_ptr<at::BFloat16>(),
            v12.data_ptr<at::BFloat16>(),
            grad_v.data_ptr<at::BFloat16>(),
            grad_v_first.data_ptr<at::BFloat16>(),
            grad_pre.data_ptr<float>(),
            total,
            c_size - 1);
    } else {
        vres_gate_backward_kernel<<<blocks, threads, 0, stream>>>(
            grad_out.data_ptr<at::BFloat16>(),
            v.data_ptr<at::BFloat16>(),
            v_first.data_ptr<at::BFloat16>(),
            v0.data_ptr<at::BFloat16>(),
            v12.data_ptr<at::BFloat16>(),
            grad_v.data_ptr<at::BFloat16>(),
            grad_v_first.data_ptr<at::BFloat16>(),
            grad_pre.data_ptr<float>(),
            total,
            c_size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_v, grad_v_first, grad_pre};
}
