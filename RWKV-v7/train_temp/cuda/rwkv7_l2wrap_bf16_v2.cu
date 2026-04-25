#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdint>

namespace {

constexpr int64_t L2WRAP_VOCAB = 65536;
constexpr float NEG_INF_F = -3.4028234663852886e+38F;

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t x) {
    return static_cast<float>(x);
}

template <>
__device__ inline float scalar_to_float<at::BFloat16>(at::BFloat16 x) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x));
}

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float x) {
    return static_cast<scalar_t>(x);
}

template <>
__device__ inline at::BFloat16 float_to_scalar<at::BFloat16>(float x) {
    __nv_bfloat16 v = __float2bfloat16_rn(x);
    return *reinterpret_cast<at::BFloat16*>(&v);
}

__device__ inline void reduce_max_first(float& value, int& index, float other_value, int other_index) {
    if ((other_value > value) || (other_value == value && other_index < index)) {
        value = other_value;
        index = other_index;
    }
}

__device__ inline void warp_reduce_max_first(float& value, int& index) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_value = __shfl_down_sync(mask, value, offset);
        int other_index = __shfl_down_sync(mask, index, offset);
        reduce_max_first(value, index, other_value, other_index);
    }
}

template <typename scalar_t, int BLOCK_SIZE>
__global__ void l2wrap_backward_v2_kernel(
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ gy,
    int64_t rows,
    float factor) {
    int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int kWarps = BLOCK_SIZE / 32;
    const int64_t base = row * L2WRAP_VOCAB;

    float local_max = NEG_INF_F;
    int local_idx = 0;

    for (int64_t col = tid; col < L2WRAP_VOCAB; col += BLOCK_SIZE) {
        float v = scalar_to_float(y[base + col]);
        reduce_max_first(local_max, local_idx, v, static_cast<int>(col));
    }

    warp_reduce_max_first(local_max, local_idx);

    __shared__ float warp_val[16];
    __shared__ int warp_idx[16];
    if (lane == 0) {
        warp_val[warp] = local_max;
        warp_idx[warp] = local_idx;
    }
    __syncthreads();

    if (warp == 0) {
        float block_max = lane < kWarps ? warp_val[lane] : NEG_INF_F;
        int block_idx = lane < kWarps ? warp_idx[lane] : 0;
        warp_reduce_max_first(block_max, block_idx);
        if (lane == 0) {
            gy[base + block_idx] = float_to_scalar<scalar_t>(block_max * factor);
        }
    }
}

template <typename scalar_t, int BLOCK_SIZE>
void launch_l2wrap_backward_v2_kernel(
    const scalar_t* y,
    scalar_t* gy,
    int64_t rows,
    float factor,
    cudaStream_t stream) {
    dim3 blocks(static_cast<unsigned int>(rows));
    l2wrap_backward_v2_kernel<scalar_t, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
        y,
        gy,
        rows,
        factor);
}

} // namespace

torch::Tensor l2wrap_backward_v2_cuda(torch::Tensor y) {
    auto gy = torch::empty_like(y);
    const int64_t rows = y.numel() / L2WRAP_VOCAB;
    const float factor = static_cast<float>(1.0e-4 / static_cast<double>(rows));
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(gy.data_ptr(), 0, gy.nbytes(), stream));

    int threads = 512;
    if (const char* env = std::getenv("L2WRAP_V2_THREADS")) {
        threads = std::atoi(env);
    }

    if (y.scalar_type() == torch::kBFloat16) {
        if (threads == 128) {
            launch_l2wrap_backward_v2_kernel<at::BFloat16, 128>(
                y.data_ptr<at::BFloat16>(), gy.data_ptr<at::BFloat16>(), rows, factor, stream);
        } else if (threads == 512) {
            launch_l2wrap_backward_v2_kernel<at::BFloat16, 512>(
                y.data_ptr<at::BFloat16>(), gy.data_ptr<at::BFloat16>(), rows, factor, stream);
        } else {
            launch_l2wrap_backward_v2_kernel<at::BFloat16, 256>(
                y.data_ptr<at::BFloat16>(), gy.data_ptr<at::BFloat16>(), rows, factor, stream);
        }
    } else {
        if (threads == 128) {
            launch_l2wrap_backward_v2_kernel<float, 128>(
                y.data_ptr<float>(), gy.data_ptr<float>(), rows, factor, stream);
        } else if (threads == 512) {
            launch_l2wrap_backward_v2_kernel<float, 512>(
                y.data_ptr<float>(), gy.data_ptr<float>(), rows, factor, stream);
        } else {
            launch_l2wrap_backward_v2_kernel<float, 256>(
                y.data_ptr<float>(), gy.data_ptr<float>(), rows, factor, stream);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return gy;
}
