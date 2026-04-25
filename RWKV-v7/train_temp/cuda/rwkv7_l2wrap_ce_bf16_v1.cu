#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

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

__device__ inline float warp_reduce_sum(float value) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

template <typename scalar_t, int BLOCK_SIZE>
__global__ void l2wrap_ce_forward_v2_kernel(
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ lse,
    float* __restrict__ max_vals,
    int* __restrict__ argmax,
    float* __restrict__ loss_rows,
    int64_t rows,
    int64_t vocab) {
    int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int kWarps = BLOCK_SIZE / 32;
    const int64_t base = row * vocab;
    const int target = static_cast<int>(targets[row]);

    float local_max = NEG_INF_F;
    int local_idx = 0;
    float local_target = 0.0f;

    for (int64_t col = tid; col < vocab; col += BLOCK_SIZE) {
        const float v = scalar_to_float(logits[base + col]);
        reduce_max_first(local_max, local_idx, v, static_cast<int>(col));
        if (static_cast<int>(col) == target) {
            local_target = v;
        }
    }

    warp_reduce_max_first(local_max, local_idx);
    local_target = warp_reduce_sum(local_target);

    __shared__ float warp_val[16];
    __shared__ int warp_idx[16];
    __shared__ float warp_target[16];
    __shared__ float shared_max;
    if (lane == 0) {
        warp_val[warp] = local_max;
        warp_idx[warp] = local_idx;
        warp_target[warp] = local_target;
    }
    __syncthreads();

    if (warp == 0) {
        float block_max = lane < kWarps ? warp_val[lane] : NEG_INF_F;
        int block_idx = lane < kWarps ? warp_idx[lane] : 0;
        float target_val = lane < kWarps ? warp_target[lane] : 0.0f;
        warp_reduce_max_first(block_max, block_idx);
        target_val = warp_reduce_sum(target_val);
        if (lane == 0) {
            shared_max = block_max;
            max_vals[row] = block_max;
            argmax[row] = block_idx;
            warp_target[0] = target_val;
        }
    }
    __syncthreads();

    const float block_max = shared_max;
    float local_sum = 0.0f;
    for (int64_t col = tid; col < vocab; col += BLOCK_SIZE) {
        local_sum += __expf(scalar_to_float(logits[base + col]) - block_max);
    }
    local_sum = warp_reduce_sum(local_sum);

    if (lane == 0) {
        warp_val[warp] = local_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum = lane < kWarps ? warp_val[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            const float row_lse = logf(block_sum) + block_max;
            lse[row] = row_lse;
            loss_rows[row] = row_lse - warp_target[0];
        }
    }
}

__global__ void reduce_loss_kernel(
    const float* __restrict__ loss_rows,
    float* __restrict__ loss,
    int64_t rows) {
    constexpr int BLOCK_SIZE = 256;
    const int tid = threadIdx.x;
    float sum = 0.0f;
    for (int64_t idx = tid; idx < rows; idx += BLOCK_SIZE) {
        sum += loss_rows[idx];
    }
    sum = warp_reduce_sum(sum);

    __shared__ float warp_sum[8];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        warp_sum[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        float block_sum = lane < 8 ? warp_sum[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            loss[0] = block_sum / static_cast<float>(rows);
        }
    }
}

template <typename scalar_t, int BLOCK_SIZE>
__global__ void l2wrap_ce_backward_v2_kernel(
    const float* __restrict__ grad_loss,
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float* __restrict__ lse,
    const float* __restrict__ max_vals,
    const int* __restrict__ argmax,
    scalar_t* __restrict__ grad_logits,
    int64_t rows,
    int64_t vocab) {
    int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const int64_t base = row * vocab;
    const int target = static_cast<int>(targets[row]);
    const int max_idx = argmax[row];
    const float inv_rows = 1.0f / static_cast<float>(rows);
    const float ce_scale = grad_loss[0] * inv_rows;
    const float l2_val = max_vals[row] * (1.0e-4f * inv_rows);
    const float row_lse = lse[row];

    for (int64_t col = tid; col < vocab; col += BLOCK_SIZE) {
        float g = __expf(scalar_to_float(logits[base + col]) - row_lse) * ce_scale;
        if (static_cast<int>(col) == target) {
            g -= ce_scale;
        }
        // Match original L2Wrap: sparse grad is not multiplied by grad_loss.
        if (static_cast<int>(col) == max_idx) {
            g += l2_val;
        }
        grad_logits[base + col] = float_to_scalar<scalar_t>(g);
    }
}

template <typename scalar_t, int BLOCK_SIZE>
void launch_l2wrap_ce_forward_v2_kernel(
    const scalar_t* logits,
    const int64_t* targets,
    float* lse,
    float* max_vals,
    int* argmax,
    float* loss_rows,
    int64_t rows,
    int64_t vocab,
    cudaStream_t stream) {
    dim3 blocks(static_cast<unsigned int>(rows));
    l2wrap_ce_forward_v2_kernel<scalar_t, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
        logits,
        targets,
        lse,
        max_vals,
        argmax,
        loss_rows,
        rows,
        vocab);
}

template <typename scalar_t, int BLOCK_SIZE>
void launch_l2wrap_ce_backward_v2_kernel(
    const float* grad_loss,
    const scalar_t* logits,
    const int64_t* targets,
    const float* lse,
    const float* max_vals,
    const int* argmax,
    scalar_t* grad_logits,
    int64_t rows,
    int64_t vocab,
    cudaStream_t stream) {
    dim3 blocks(static_cast<unsigned int>(rows));
    l2wrap_ce_backward_v2_kernel<scalar_t, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
        grad_loss,
        logits,
        targets,
        lse,
        max_vals,
        argmax,
        grad_logits,
        rows,
        vocab);
}

} // namespace

std::vector<torch::Tensor> l2wrap_ce_forward_v2_cuda(
    torch::Tensor logits,
    torch::Tensor targets,
    int64_t vocab) {
    const int64_t rows = logits.numel() / vocab;
    auto meta_opts = torch::TensorOptions().device(logits.device()).dtype(torch::kFloat32);
    auto int_opts = torch::TensorOptions().device(logits.device()).dtype(torch::kInt32);
    auto lse = torch::empty({rows}, meta_opts);
    auto max_vals = torch::empty({rows}, meta_opts);
    auto argmax = torch::empty({rows}, int_opts);
    auto loss_rows = torch::empty({rows}, meta_opts);
    auto loss = torch::empty({}, meta_opts);

    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int threads = 512;
    if (logits.scalar_type() == torch::kBFloat16) {
        launch_l2wrap_ce_forward_v2_kernel<at::BFloat16, threads>(
            logits.data_ptr<at::BFloat16>(),
            targets.data_ptr<int64_t>(),
            lse.data_ptr<float>(),
            max_vals.data_ptr<float>(),
            argmax.data_ptr<int>(),
            loss_rows.data_ptr<float>(),
            rows,
            vocab,
            stream);
    } else {
        launch_l2wrap_ce_forward_v2_kernel<float, threads>(
            logits.data_ptr<float>(),
            targets.data_ptr<int64_t>(),
            lse.data_ptr<float>(),
            max_vals.data_ptr<float>(),
            argmax.data_ptr<int>(),
            loss_rows.data_ptr<float>(),
            rows,
            vocab,
            stream);
    }
    reduce_loss_kernel<<<1, 256, 0, stream>>>(loss_rows.data_ptr<float>(), loss.data_ptr<float>(), rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {loss, lse, max_vals, argmax};
}

torch::Tensor l2wrap_ce_backward_v2_cuda(
    torch::Tensor grad_loss,
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor lse,
    torch::Tensor max_vals,
    torch::Tensor argmax,
    int64_t vocab) {
    const int64_t rows = logits.numel() / vocab;
    auto grad_logits = torch::empty_like(logits);
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int threads = 512;
    if (logits.scalar_type() == torch::kBFloat16) {
        launch_l2wrap_ce_backward_v2_kernel<at::BFloat16, threads>(
            grad_loss.data_ptr<float>(),
            logits.data_ptr<at::BFloat16>(),
            targets.data_ptr<int64_t>(),
            lse.data_ptr<float>(),
            max_vals.data_ptr<float>(),
            argmax.data_ptr<int>(),
            grad_logits.data_ptr<at::BFloat16>(),
            rows,
            vocab,
            stream);
    } else {
        launch_l2wrap_ce_backward_v2_kernel<float, threads>(
            grad_loss.data_ptr<float>(),
            logits.data_ptr<float>(),
            targets.data_ptr<int64_t>(),
            lse.data_ptr<float>(),
            max_vals.data_ptr<float>(),
            argmax.data_ptr<int>(),
            grad_logits.data_ptr<float>(),
            rows,
            vocab,
            stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_logits;
}
