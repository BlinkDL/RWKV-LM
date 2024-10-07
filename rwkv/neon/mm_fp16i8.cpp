#include <cstdint>
#include <vector>
#include <arm_neon.h> // For NEON intrinsics
#include <omp.h>     // For OpenMP
#include <cstring>    // For memset
#include <torch/extension.h>


#if 1
// cf: rwkv/cuda/operators.cu  kernel_mm_seq_fp16i8()
void kernel_mm_seq_fp16i8(
    const int B, const int N, const int M,
    const at::Half* x, const int x_stride,
    const uint8_t* w, const int w_stride,
    const at::Half* mx,
    const at::Half* rx,
    const at::Half* my,
    const at::Half* ry,
    float* y, const int y_stride) {

    // Parallelize over the batch dimension B and the output feature dimension M using OpenMP
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < B; ++i) {
        for (int k = 0; k < M; ++k) {
            // Initialize local accumulation for y
            float32x4_t y_local_vec = vdupq_n_f32(0.0f);

            // Perform matrix multiplication with on-the-fly dequantization using NEON intrinsics
            int j = 0;
            for (; j <= N - 4; j += 4) {
                // Load x[i * x_stride + j:j+3]
                float16x4_t x_vec_fp16 = vld1_f16(reinterpret_cast<const __fp16*>(&x[i * x_stride + j]));
                float32x4_t x_vec = vcvt_f32_f16(x_vec_fp16);  // Convert to float32

                // Load w[j * w_stride + k:k+3] and dequantize
                uint8x8_t w_u8 = vld1_u8(&w[j * w_stride + k]);
                uint16x8_t w_u16 = vmovl_u8(w_u8);
                float16x4_t w_fp16 = vcvt_f16_u16(vget_low_u16(w_u16));  // Convert to FP16
                float32x4_t w_vec = vaddq_f32(vcvt_f32_f16(w_fp16), vdupq_n_f32(0.5f));

                // Load rx[k], ry[j:j+3], mx[k], and my[j:j+3]
                float16x4_t ry_vec_fp16 = vld1_f16(reinterpret_cast<const __fp16*>(&ry[j]));
                float16x4_t my_vec_fp16 = vld1_f16(reinterpret_cast<const __fp16*>(&my[j]));
                float32x4_t ry_vec = vcvt_f32_f16(ry_vec_fp16);
                float32x4_t my_vec = vcvt_f32_f16(my_vec_fp16);

                // float32x4_t rx_k = vdupq_n_f32(vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(&rx[k]))));
                float32x4_t rx_k = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(&rx[k]))); // xzl
                // float32x4_t mx_k = vdupq_n_f32(vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(&mx[k]))));
                float32x4_t mx_k = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(&mx[k]))); // xzl

                // Compute the accumulation for the current block
                float32x4_t temp = vmlaq_f32(
                    vaddq_f32(mx_k, my_vec),       // mx_k + my[j:j+3]
                    vmulq_f32(w_vec, rx_k),        // w_vec * rx_k
                    ry_vec                         // * ry[j:j+3]
                );

                // Accumulate x * temp
                y_local_vec = vmlaq_f32(y_local_vec, x_vec, temp);
            }

            // Handle remaining elements (not processed by the vectorized loop)
            float y_local = vaddvq_f32(y_local_vec);
            for (; j < N; ++j) {
                float x_val = static_cast<float>(x[i * x_stride + j]);
                float w_val = static_cast<float>(w[j * w_stride + k]) + 0.5f;
                float rx_val = static_cast<float>(rx[k]);
                float ry_val = static_cast<float>(ry[j]);
                float mx_val = static_cast<float>(mx[k]);
                float my_val = static_cast<float>(my[j]);

                y_local += x_val * (w_val * rx_val * ry_val + mx_val + my_val);
            }

            // Store the result in y
            y[i * y_stride + k] = y_local;
        }
    }
}
#endif


#include <arm_neon.h>
#include <torch/extension.h>
#include <omp.h>
#include <cstring>

// cf: rwkv/cuda/operators.cu kernel_mm_one_fp16i8
#if 0 // still bad
void kernel_mm_one_fp16i8(
    int N, int M,
    const at::Half* x_fp16,
    const uint8_t* w, int w_stride,
    const at::Half* mx_fp16,
    const at::Half* rx_fp16,
    const at::Half* my_fp16,
    const at::Half* ry_fp16,
    float* y_fp)
{
    // Ensure that at::Half and __fp16 have the same size
    static_assert(sizeof(at::Half) == sizeof(__fp16), "at::Half and __fp16 must be the same size");

    // Initialize y to zero
    std::fill(y_fp, y_fp + M, float(0.0));

    // Parallelize over the M dimension using OpenMP
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < M; ++k) {
        // Initialize local accumulator in FP32
        float32x4_t y_local_vec = vdupq_n_f32(0.0f);

        int j = 0;
        const int unroll_size = 4; // NEON processes 4 FP32 elements at a time

        // Broadcast rx[k] and mx[k] to all elements
        float rx_k = static_cast<float>(rx_fp16[k]);
        float mx_k = static_cast<float>(mx_fp16[k]);
        float32x4_t rx_k_vec = vdupq_n_f32(rx_k);
        float32x4_t mx_k_vec = vdupq_n_f32(mx_k);

        for (; j <= N - unroll_size; j += unroll_size) {
            // Load x[j:j+3] into a NEON register
            const __fp16* x_ptr = reinterpret_cast<const __fp16*>(&x_fp16[j]);
            float16x4_t x_vec_fp16 = vld1_f16(x_ptr);
            float32x4_t x_vec = vcvt_f32_f16(x_vec_fp16);

            // Load w[j:j+3, k], expand to uint16_t
            const uint8_t* w_ptr = &w[j * w_stride + k];
            uint8x8_t w_u8 = vld1_u8(w_ptr);
            uint16x8_t w_u16 = vmovl_u8(w_u8);

            // Convert w_u16 to float32 and add 0.5f
            float32x4_t w_vec = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w_u16)));
            w_vec = vaddq_f32(w_vec, vdupq_n_f32(0.5f));

            // Load ry[j:j+3] and my[j:j+3]
            const __fp16* ry_ptr = reinterpret_cast<const __fp16*>(&ry_fp16[j]);
            float16x4_t ry_vec_fp16 = vld1_f16(ry_ptr);
            float32x4_t ry_vec = vcvt_f32_f16(ry_vec_fp16);

            const __fp16* my_ptr = reinterpret_cast<const __fp16*>(&my_fp16[j]);
            float16x4_t my_vec_fp16 = vld1_f16(my_ptr);
            float32x4_t my_vec = vcvt_f32_f16(my_vec_fp16);

            // Compute temp = (w_val * rx_k * ry_j) + mx_k + my_j
            float32x4_t temp = vaddq_f32(
                vmulq_f32(w_vec, vmulq_f32(rx_k_vec, ry_vec)),
                vaddq_f32(mx_k_vec, my_vec)
            );

            // Multiply x_vec with temp and accumulate
            y_local_vec = vaddq_f32(y_local_vec, vmulq_f32(x_vec, temp));
        }

        // Sum the elements of y_local_vec manually
        float y_local = vaddvq_f32(y_local_vec);

        // Handle remaining elements
        for (; j < N; ++j) {
            float x_j = static_cast<float>(x_fp16[j]);
            float w_val = static_cast<float>(w[j * w_stride + k]) + 0.5f;
            float ry_j = static_cast<float>(ry_fp16[j]);
            float my_j = static_cast<float>(my_fp16[j]);

            float temp = (w_val * rx_k * ry_j) + mx_k + my_j;
            y_local += x_j * temp;
        }

        // Store the result
        y_fp[k] = y_local;
    }
}
#endif

// cf: rwkv/cuda/operators.cu kernel_mm_one_fp16i8
// works
void kernel_mm_one_fp16i8(
    int N, int M,
    const at::Half* x_fp16,
    const uint8_t* w, int w_stride,
    const at::Half* mx_fp16,
    const at::Half* rx_fp16,
    const at::Half* my_fp16,
    const at::Half* ry_fp16,
    float* y_fp)
{
    // Parallelize over the M dimension using OpenMP
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < M; ++k) {
        // Load rx[k] and mx[k]
        float16_t rx_k = static_cast<float16_t>(rx_fp16[k]);
        float16_t mx_k = static_cast<float16_t>(mx_fp16[k]);

        // Broadcast rx_k and mx_k into NEON vectors
        float16x8_t rx_k_vec = vdupq_n_f16(rx_k);
        float16x8_t mx_k_vec = vdupq_n_f16(mx_k);

        // Initialize NEON accumulators
        float32x4_t y_acc_low = vdupq_n_f32(0.0f);
        float32x4_t y_acc_high = vdupq_n_f32(0.0f);

        int j = 0;
        for (; j <= N - 8; j += 8) {
            // Load x_fp16[j:j+7], ry_fp16[j:j+7], my_fp16[j:j+7]
            float16x8_t x_vec_fp16 = vld1q_f16(reinterpret_cast<const float16_t*>(&x_fp16[j]));
            float16x8_t ry_vec_fp16 = vld1q_f16(reinterpret_cast<const float16_t*>(&ry_fp16[j]));
            float16x8_t my_vec_fp16 = vld1q_f16(reinterpret_cast<const float16_t*>(&my_fp16[j]));

            // Load w[j:j+7, k]
            uint8_t w_vals_u8[8];
            for (int l = 0; l < 8; ++l) {
                w_vals_u8[l] = w[(j + l) * w_stride + k];
            }
            uint8x8_t w_u8 = vld1_u8(w_vals_u8);
            uint16x8_t w_u16 = vmovl_u8(w_u8);

            // Convert w_u16 to float16x8_t and add 0.5
            float16x8_t w_vec_fp16 = vcvtq_f16_u16(w_u16);
            float16x8_t half_vec_fp16 = vdupq_n_f16(0.5f);
            w_vec_fp16 = vaddq_f16(w_vec_fp16, half_vec_fp16);

            // Compute temp_fp16 = (w_vec_fp16 * ry_vec_fp16 * rx_k_vec) + my_vec_fp16 + mx_k_vec
            float16x8_t temp_fp16 = vmulq_f16(w_vec_fp16, ry_vec_fp16);
            temp_fp16 = vmulq_f16(temp_fp16, rx_k_vec);
            temp_fp16 = vaddq_f16(temp_fp16, my_vec_fp16);
            temp_fp16 = vaddq_f16(temp_fp16, mx_k_vec);

            // Multiply x_vec_fp16 * temp_fp16
            float16x8_t prod_fp16 = vmulq_f16(x_vec_fp16, temp_fp16);

            // Convert prod_fp16 to float32 for accumulation
            float32x4_t prod_low = vcvt_f32_f16(vget_low_f16(prod_fp16));
            float32x4_t prod_high = vcvt_f32_f16(vget_high_f16(prod_fp16));

            // Accumulate
            y_acc_low = vaddq_f32(y_acc_low, prod_low);
            y_acc_high = vaddq_f32(y_acc_high, prod_high);
        }

        // Sum accumulators into y_local
        float y_local = vaddvq_f32(y_acc_low) + vaddvq_f32(y_acc_high);

        // Handle remaining elements
        for (; j < N; ++j) {
            float16_t x_j = static_cast<float16_t>(x_fp16[j]);
            float16_t ry_j = static_cast<float16_t>(ry_fp16[j]);
            float16_t my_j = static_cast<float16_t>(my_fp16[j]);

            float16_t w_val = static_cast<float16_t>(w[j * w_stride + k]) + 0.5f;

            float16_t temp_fp16 = (w_val * ry_j * rx_k) + my_j + mx_k;

            // Convert to float32 for accumulation
            float x_j_f32 = static_cast<float>(x_j);
            float temp_f32 = static_cast<float>(temp_fp16);

            y_local += x_j_f32 * temp_f32;
        }

        // Store the result
        y_fp[k] = y_local;
    }
}


// same as kernel_mm_one_fp16i8, but all intermediate results are in FP32
void kernel_mm_one_fp16i8_fp32(
    int N, int M,
    const at::Half* x_fp16,
    const uint8_t* w, int w_stride,
    const at::Half* mx_fp16,
    const at::Half* rx_fp16,
    const at::Half* my_fp16,
    const at::Half* ry_fp16,
    float* y_fp)
{
    // Initialize y to zero
    std::fill(y_fp, y_fp + M, float(0.0));

    // Parallelize over the M dimension using OpenMP
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < M; ++k) {
        // Initialize local accumulator in FP32
        float32x4_t y_local_vec = vdupq_n_f32(0.0f);

        int j = 0;
        const int unroll_size = 4; // NEON processes 4 FP32 elements at a time

        // Broadcast rx[k] and mx[k] to all elements
        float rx_k = static_cast<float>(rx_fp16[k]);
        float mx_k = static_cast<float>(mx_fp16[k]);
        float32x4_t rx_k_vec = vdupq_n_f32(rx_k);
        float32x4_t mx_k_vec = vdupq_n_f32(mx_k);

        for (; j <= N - unroll_size; j += unroll_size) {
            // Load x[j:j+3] into a NEON register
            // xzl: BUG -- cannot cast like this 
            const float* x_ptr = reinterpret_cast<const float*>(&x_fp16[j]);
            float32x4_t x_vec = vld1q_f32(x_ptr);

            // Load w[j:j+3, k], expand to uint16_t
            const uint8_t* w_ptr = &w[j * w_stride + k];
            uint8x8_t w_u8 = vld1_u8(w_ptr);
            uint16x8_t w_u16 = vmovl_u8(w_u8);

            // Convert w_u16 to float32 and add 0.5f
            float32x4_t w_vec = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w_u16)));
            w_vec = vaddq_f32(w_vec, vdupq_n_f32(0.5f));

            // Load ry[j:j+3] and my[j:j+3]
            const float* ry_ptr = reinterpret_cast<const float*>(&ry_fp16[j]);
            float32x4_t ry_vec = vld1q_f32(ry_ptr);

            const float* my_ptr = reinterpret_cast<const float*>(&my_fp16[j]);
            float32x4_t my_vec = vld1q_f32(my_ptr);

            // Compute temp = (w_val * rx_k * ry_j) + mx_k + my_j
            float32x4_t temp = vaddq_f32(
                vmulq_f32(w_vec, vmulq_f32(rx_k_vec, ry_vec)),
                vaddq_f32(mx_k_vec, my_vec)
            );

            // Multiply x_vec with temp and accumulate
            y_local_vec = vaddq_f32(y_local_vec, vmulq_f32(x_vec, temp));
        }

        // Sum the elements of y_local_vec manually
        float y_local = vaddvq_f32(y_local_vec);

        // Handle remaining elements
        for (; j < N; ++j) {
            float x_j = static_cast<float>(x_fp16[j]);
            float w_val = static_cast<float>(w[j * w_stride + k]) + 0.5f;
            float ry_j = static_cast<float>(ry_fp16[j]);
            float my_j = static_cast<float>(my_fp16[j]);

            float temp = (w_val * rx_k * ry_j) + mx_k + my_j;
            y_local += x_j * temp;
        }

        // Store the result
        y_fp[k] = y_local;
    }
}

#if 0
// bad. not broadcasting 
void kernel_mm_one_fp32i8(
    int N, int M,
    const float* x_fp32,
    const uint8_t* w, int w_stride,
    const float* mx_fp32,
    const float* rx_fp32,
    const float* my_fp32,
    const float* ry_fp32,
    float* y_fp)
{
    // Initialize y to zero
    std::fill(y_fp, y_fp + M, float(0.0));

    // Parallelize over the M dimension using OpenMP
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < M; ++k) {
        // Initialize local accumulator in FP32
        float32x4_t y_local_vec = vdupq_n_f32(0.0f);

        int j = 0;
        const int unroll_size = 4; // NEON processes 4 FP32 elements at a time

        // Broadcast rx[k] and mx[k] to all elements
        float rx_k = rx_fp32[k];
        float mx_k = mx_fp32[k];
        float32x4_t rx_k_vec = vdupq_n_f32(rx_k);
        float32x4_t mx_k_vec = vdupq_n_f32(mx_k);

        for (; j <= N - unroll_size; j += unroll_size) {
            // Load x[j:j+3] into a NEON register
            const float* x_ptr = &x_fp32[j];
            float32x4_t x_vec = vld1q_f32(x_ptr);

            // Load w[j:j+3, k], expand to uint16_t
            const uint8_t* w_ptr = &w[j * w_stride + k];
            uint8x8_t w_u8 = vld1_u8(w_ptr);
            uint16x8_t w_u16 = vmovl_u8(w_u8);

            // Convert w_u16 to float32 and add 0.5f
            float32x4_t w_vec = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w_u16)));
            w_vec = vaddq_f32(w_vec, vdupq_n_f32(0.5f));

            // Load ry[j:j+3] and my[j:j+3]
            const float* ry_ptr = &ry_fp32[j];
            float32x4_t ry_vec = vld1q_f32(ry_ptr);

            const float* my_ptr = &my_fp32[j];
            float32x4_t my_vec = vld1q_f32(my_ptr);

            // Compute temp = (w_val * rx_k * ry_j) + mx_k + my_j
            float32x4_t temp = vaddq_f32(
                vmulq_f32(w_vec, vmulq_f32(rx_k_vec, ry_vec)),
                vaddq_f32(mx_k_vec, my_vec)
            );

            // Multiply x_vec with temp and accumulate
            y_local_vec = vaddq_f32(y_local_vec, vmulq_f32(x_vec, temp));
        }

        // Sum the elements of y_local_vec manually
        float y_local = vaddvq_f32(y_local_vec);

        // Handle remaining elements
        for (; j < N; ++j) {
            float x_j = x_fp32[j];
            float w_val = static_cast<float>(w[j * w_stride + k]) + 0.5f;
            float ry_j = ry_fp32[j];
            float my_j = my_fp32[j];

            float temp = (w_val * rx_k * ry_j) + mx_k + my_j;
            y_local += x_j * temp;
        }

        // Store the result
        y_fp[k] = y_local;
    }
}
#endif


// non simd version. just to validate
// verified --- correct 
void kernel_mm_one_fp32i8_nosimd(
    int N, int M,
    const float* x_fp32,
    const uint8_t* w, int w_stride,
    const float* mx_fp32,
    const float* rx_fp32,
    const float* my_fp32,
    const float* ry_fp32,
    float* y_fp)
{
    // Initialize y to zero
    std::fill(y_fp, y_fp + M, float(0.0));

    // Loop over N dimension
    for (int j = 0; j < N; ++j) {
        float x_j = x_fp32[j];
        float ry_j = ry_fp32[j];
        float my_j = my_fp32[j];

        // Loop over M dimension
        for (int k = 0; k < M; ++k) {
            float w_val = static_cast<float>(w[j * w_stride + k]) + 0.5f;
            float rx_k = rx_fp32[k];
            float mx_k = mx_fp32[k];

            float temp = (w_val * ry_j * rx_k) + my_j + mx_k;
            y_fp[k] += x_j * temp;
        }
    }
}

// simd version. checked, good 
void kernel_mm_one_fp32i8(
    int N, int M,
    const float* x_fp32,
    const uint8_t* w, int w_stride,
    const float* mx_fp32,
    const float* rx_fp32,
    const float* my_fp32,
    const float* ry_fp32,
    float* y_fp)
{
    // Parallelize over the M dimension using OpenMP
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < M; ++k) {
        float rx_k = rx_fp32[k];
        float mx_k = mx_fp32[k];
        float y_local = 0.0f;

        // NEON vector constants
        float32x4_t rx_k_vec = vdupq_n_f32(rx_k);
        float32x4_t mx_k_vec = vdupq_n_f32(mx_k);

        // Initialize NEON accumulator
        float32x4_t y_acc_vec = vdupq_n_f32(0.0f);

        int j = 0;
        for (; j <= N - 4; j += 4) {
            // Load x[j:j+3], ry[j:j+3], my[j:j+3]
            float32x4_t x_vec = vld1q_f32(&x_fp32[j]);
            float32x4_t ry_vec = vld1q_f32(&ry_fp32[j]);
            float32x4_t my_vec = vld1q_f32(&my_fp32[j]);

            // Load w[j:j+3, k] individually and pack into a vector
            float w_vals[4] = {
                static_cast<float>(w[(j + 0) * w_stride + k]) + 0.5f,
                static_cast<float>(w[(j + 1) * w_stride + k]) + 0.5f,
                static_cast<float>(w[(j + 2) * w_stride + k]) + 0.5f,
                static_cast<float>(w[(j + 3) * w_stride + k]) + 0.5f
            };
            float32x4_t w_vec = vld1q_f32(w_vals);

            // Compute temp = (w_vec * ry_vec * rx_k_vec) + my_vec + mx_k_vec
            float32x4_t temp = vmulq_f32(w_vec, ry_vec);          // temp = w_vec * ry_vec
            temp = vmulq_f32(temp, rx_k_vec);                     // temp = temp * rx_k_vec
            temp = vaddq_f32(temp, my_vec);                       // temp = temp + my_vec
            temp = vaddq_f32(temp, mx_k_vec);                     // temp = temp + mx_k_vec

            // Accumulate y_acc_vec += x_vec * temp
            y_acc_vec = vmlaq_f32(y_acc_vec, x_vec, temp);
        }

        // Horizontal addition to sum y_acc_vec
        y_local += vaddvq_f32(y_acc_vec);

        // Handle remaining elements
        for (; j < N; ++j) {
            float x_j = x_fp32[j];
            float ry_j = ry_fp32[j];
            float my_j = my_fp32[j];

            float w_val = static_cast<float>(w[j * w_stride + k]) + 0.5f;

            float temp = (w_val * ry_j * rx_k) + my_j + mx_k;

            y_local += x_j * temp;
        }

        // Store the result
        y_fp[k] = y_local;
    }
}

torch::Tensor mm_one_fp32i8(
    torch::Tensor x_fp32,
    torch::Tensor w_uint8,
    torch::Tensor mx_fp32,
    torch::Tensor rx_fp32,
    torch::Tensor my_fp32,
    torch::Tensor ry_fp32)
{
    // Ensure tensors are contiguous and on CPU
    x_fp32 = x_fp32.contiguous();
    w_uint8 = w_uint8.contiguous();
    mx_fp32 = mx_fp32.contiguous();
    rx_fp32 = rx_fp32.contiguous();
    my_fp32 = my_fp32.contiguous();
    ry_fp32 = ry_fp32.contiguous();

    // Validate tensor data types
    TORCH_CHECK(x_fp32.dtype() == torch::kFloat32, "x_fp32 must be Float32");
    TORCH_CHECK(w_uint8.dtype() == torch::kUInt8, "w_uint8 must be UInt8");
    TORCH_CHECK(mx_fp32.dtype() == torch::kFloat32, "mx_fp32 must be Float32");
    TORCH_CHECK(rx_fp32.dtype() == torch::kFloat32, "rx_fp32 must be Float32");
    TORCH_CHECK(my_fp32.dtype() == torch::kFloat32, "my_fp32 must be Float32");
    TORCH_CHECK(ry_fp32.dtype() == torch::kFloat32, "ry_fp32 must be Float32");

    int N = x_fp32.size(0);
    int M = w_uint8.size(1);
    int w_stride = w_uint8.stride(0);

    torch::Tensor y = torch::empty({M}, torch::dtype(torch::kFloat));

    kernel_mm_one_fp32i8(
        N, M,
        x_fp32.data_ptr<float>(),
        w_uint8.data_ptr<uint8_t>(), w_stride,
        mx_fp32.data_ptr<float>(),
        rx_fp32.data_ptr<float>(),
        my_fp32.data_ptr<float>(),
        ry_fp32.data_ptr<float>(),
        y.data_ptr<float>()
    );

    return y;
}

torch::Tensor mm_one_fp16i8(
    torch::Tensor x_fp16,
    torch::Tensor w_uint8,
    torch::Tensor mx_fp16,
    torch::Tensor rx_fp16,
    torch::Tensor my_fp16,
    torch::Tensor ry_fp16)
{
    // Ensure tensors are contiguous and on CPU
    x_fp16 = x_fp16.contiguous();
    w_uint8 = w_uint8.contiguous();
    mx_fp16 = mx_fp16.contiguous();
    rx_fp16 = rx_fp16.contiguous();
    my_fp16 = my_fp16.contiguous();
    ry_fp16 = ry_fp16.contiguous();

    // Validate tensor data types
    TORCH_CHECK(x_fp16.dtype() == torch::kFloat16, "x_fp16 must be Float16");
    TORCH_CHECK(w_uint8.dtype() == torch::kUInt8, "w_uint8 must be UInt8");
    TORCH_CHECK(mx_fp16.dtype() == torch::kFloat16, "mx_fp16 must be Float16");
    TORCH_CHECK(rx_fp16.dtype() == torch::kFloat16, "rx_fp16 must be Float16");
    TORCH_CHECK(my_fp16.dtype() == torch::kFloat16, "my_fp16 must be Float16");
    TORCH_CHECK(ry_fp16.dtype() == torch::kFloat16, "ry_fp16 must be Float16");

    int N = x_fp16.size(0);
    int M = w_uint8.size(1);
    int w_stride = w_uint8.stride(0);

    torch::Tensor y = torch::empty({M}, torch::dtype(torch::kFloat));

    kernel_mm_one_fp16i8(
    // kernel_mm_one_fp16i8_fp32(
        N, M,
        x_fp16.data_ptr<at::Half>(),
        w_uint8.data_ptr<uint8_t>(), w_stride,
        mx_fp16.data_ptr<at::Half>(),
        rx_fp16.data_ptr<at::Half>(),
        my_fp16.data_ptr<at::Half>(),
        ry_fp16.data_ptr<at::Half>(),
        y.data_ptr<float>()
    );

    return y;
}

#if 1
// xzl: to inspect... XXX
torch::Tensor mm_seq_fp16i8(
    torch::Tensor x_fp16,
    torch::Tensor w_uint8,
    torch::Tensor mx_fp16,
    torch::Tensor rx_fp16,
    torch::Tensor my_fp16,
    torch::Tensor ry_fp16)
{
    // Ensure tensors are contiguous and on CPU
    x_fp16 = x_fp16.contiguous();
    w_uint8 = w_uint8.contiguous();
    mx_fp16 = mx_fp16.contiguous();
    rx_fp16 = rx_fp16.contiguous();
    my_fp16 = my_fp16.contiguous();
    ry_fp16 = ry_fp16.contiguous();

    int B = x_fp16.size(0);
    int N = x_fp16.size(1);
    int M = w_uint8.size(1);
    int x_stride = x_fp16.stride(0);
    int w_stride = w_uint8.stride(0);
    int y_stride = M;

    torch::Tensor y = torch::empty({B, M}, torch::kFloat);

    kernel_mm_seq_fp16i8(
        B, N, M,
        x_fp16.data_ptr<at::Half>(), x_stride,
        w_uint8.data_ptr<uint8_t>(), w_stride,
        mx_fp16.data_ptr<at::Half>(),
        rx_fp16.data_ptr<at::Half>(),
        my_fp16.data_ptr<at::Half>(),
        ry_fp16.data_ptr<at::Half>(),
        y.data_ptr<float>(), y_stride
    );

    return y;
}
#endif


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm_one_fp16i8", &mm_one_fp16i8, "Matrix multiplication with int8 weights and float16 inputs (ARM Cortex-A76 optimized)");
    m.def("mm_one_fp32i8", &mm_one_fp32i8, "Matrix multiplication with int8 weights and float32 inputs (ARM Cortex-A76 optimized)");
}
