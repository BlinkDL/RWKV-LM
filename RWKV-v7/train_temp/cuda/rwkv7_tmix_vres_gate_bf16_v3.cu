#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <vector>

namespace {

constexpr int kForwardThreads = 128;
constexpr int kDefaultBackwardThreads = 128;
constexpr int kBackwardTileRows = 16;

inline int64_t ceil_div(int64_t n, int64_t d) {
    return (n + d - 1) / d;
}

inline bool is_power_of_two(int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

inline int read_backward_threads() {
    int threads = kDefaultBackwardThreads;
    if (const char* env = std::getenv("VRES_GATE_BWD_THREADS")) {
        threads = std::atoi(env);
    }
    TORCH_CHECK(
        threads == 64 || threads == 128 || threads == 256 || threads == 512,
        "VRES_GATE_BWD_THREADS must be one of 64, 128, 256, 512");
    return threads;
}

__device__ inline float load_bf16(const at::BFloat16* ptr) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(ptr));
}

__device__ inline void store_bf16(at::BFloat16* ptr, float value) {
    *reinterpret_cast<__nv_bfloat16*>(ptr) = __float2bfloat16_rn(value);
}

__device__ inline float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

template <bool kPow2C>
__global__ void vres_gate_v3_forward_kernel(
    const at::BFloat16* __restrict__ v,
    const at::BFloat16* __restrict__ v_first,
    const at::BFloat16* __restrict__ v0,
    const at::BFloat16* __restrict__ v12,
    at::BFloat16* __restrict__ out,
    int64_t total,
    int64_t c_size_or_mask) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int64_t c = kPow2C ? (idx & c_size_or_mask) : (idx % c_size_or_mask);
    const float vv = load_bf16(v + idx);
    const float vf = load_bf16(v_first + idx);
    const float gate = sigmoidf_fast(load_bf16(v0 + c) + load_bf16(v12 + idx));
    store_bf16(out + idx, vv + (vf - vv) * gate);
}

// Each thread owns one channel and a row tile. This keeps v0 hot in a register,
// writes the final bf16 grad_v12 directly, and reduces only grad_v0 through FP32
// partials. Do not replace this with a full FP32 grad_pre tensor: at the main
// B4T8192C4096 shape that intermediate alone is 512 MiB.
__global__ void vres_gate_v3_backward_tiled_kernel(
    const at::BFloat16* __restrict__ grad_out,
    const at::BFloat16* __restrict__ v,
    const at::BFloat16* __restrict__ v_first,
    const at::BFloat16* __restrict__ v0,
    const at::BFloat16* __restrict__ v12,
    at::BFloat16* __restrict__ grad_v,
    at::BFloat16* __restrict__ grad_v_first,
    at::BFloat16* __restrict__ grad_v12,
    float* __restrict__ partial_v0,
    int64_t rows,
    int64_t c_size) {
    const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= c_size) {
        return;
    }

    const int64_t row_base = static_cast<int64_t>(blockIdx.y) * kBackwardTileRows;
    const float v0_value = load_bf16(v0 + c);
    float sum_v0 = 0.0f;
#pragma unroll
    for (int i = 0; i < kBackwardTileRows; ++i) {
        const int64_t row = row_base + i;
        if (row >= rows) {
            break;
        }
        const int64_t idx = row * c_size + c;
        const float go = load_bf16(grad_out + idx);
        const float vv = load_bf16(v + idx);
        const float vf = load_bf16(v_first + idx);
        const float gate = sigmoidf_fast(v0_value + load_bf16(v12 + idx));
        const float grad_vf = go * gate;
        const float grad_pre = go * (vf - vv) * gate * (1.0f - gate);

        store_bf16(grad_v + idx, go - grad_vf);
        store_bf16(grad_v_first + idx, grad_vf);
        store_bf16(grad_v12 + idx, grad_pre);
        sum_v0 += grad_pre;
    }

    partial_v0[static_cast<int64_t>(blockIdx.y) * c_size + c] = sum_v0;
}

__global__ void vres_gate_v3_reduce_v0_kernel(
    const float* __restrict__ partial_v0,
    at::BFloat16* __restrict__ grad_v0,
    int64_t num_tiles,
    int64_t c_size) {
    const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= c_size) {
        return;
    }

    float sum = 0.0f;
    for (int64_t tile = 0; tile < num_tiles; ++tile) {
        sum += partial_v0[tile * c_size + c];
    }
    store_bf16(grad_v0 + c, sum);
}

} // namespace

torch::Tensor tmix_vres_gate_v3_forward_cuda(
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12) {
    auto out = torch::empty_like(v);
    const int64_t total = v.numel();
    const int64_t c_size = v.size(2);
    const int blocks = static_cast<int>(ceil_div(total, static_cast<int64_t>(kForwardThreads)));
    auto stream = at::cuda::getCurrentCUDAStream();
    if (is_power_of_two(c_size)) {
        vres_gate_v3_forward_kernel<true><<<blocks, kForwardThreads, 0, stream>>>(
            v.data_ptr<at::BFloat16>(),
            v_first.data_ptr<at::BFloat16>(),
            v0.data_ptr<at::BFloat16>(),
            v12.data_ptr<at::BFloat16>(),
            out.data_ptr<at::BFloat16>(),
            total,
            c_size - 1);
    } else {
        vres_gate_v3_forward_kernel<false><<<blocks, kForwardThreads, 0, stream>>>(
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

std::vector<torch::Tensor> tmix_vres_gate_v3_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12) {
    auto grad_v = torch::empty_like(v);
    auto grad_v_first = torch::empty_like(v_first);
    auto grad_v0 = torch::empty_like(v0);
    auto grad_v12 = torch::empty_like(v12);

    const int64_t rows = v.numel() / v.size(2);
    const int64_t c_size = v.size(2);
    const int64_t num_tiles = ceil_div(rows, static_cast<int64_t>(kBackwardTileRows));
    auto partial_v0 = torch::empty(
        {num_tiles, c_size},
        v.options().dtype(torch::kFloat32));
    const int threads = read_backward_threads();
    const dim3 blocks(
        static_cast<unsigned int>(ceil_div(c_size, static_cast<int64_t>(threads))),
        static_cast<unsigned int>(num_tiles));
    auto stream = at::cuda::getCurrentCUDAStream();

    vres_gate_v3_backward_tiled_kernel<<<blocks, threads, 0, stream>>>(
        grad_out.data_ptr<at::BFloat16>(),
        v.data_ptr<at::BFloat16>(),
        v_first.data_ptr<at::BFloat16>(),
        v0.data_ptr<at::BFloat16>(),
        v12.data_ptr<at::BFloat16>(),
        grad_v.data_ptr<at::BFloat16>(),
        grad_v_first.data_ptr<at::BFloat16>(),
        grad_v12.data_ptr<at::BFloat16>(),
        partial_v0.data_ptr<float>(),
        rows,
        c_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int reduce_blocks = static_cast<int>(ceil_div(c_size, static_cast<int64_t>(threads)));
    vres_gate_v3_reduce_v0_kernel<<<reduce_blocks, threads, 0, stream>>>(
        partial_v0.data_ptr<float>(),
        grad_v0.data_ptr<at::BFloat16>(),
        num_tiles,
        c_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_v, grad_v_first, grad_v0, grad_v12};
}
