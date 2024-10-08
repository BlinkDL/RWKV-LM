// by FL Oct 7 2024

#include <cstdint>
#include <vector>
#include <arm_neon.h> // For NEON intrinsics
#include <omp.h>     // For OpenMP
#include <cstring>    // For memset
#include <torch/extension.h>
#include <ATen/ATen.h>


#define DEBUG_KERNEL 1 //uncomment to enable debug prints

// cf: rwkv/cuda/operators.cu kernel_mm_one_fp16i8
// omp, not caring much about memory locality
// works
void kernel_mm_one_fp16i8_v1(
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

// v2. better memory locality, assuming w is contiguous
//  but NOT using openmp; marginally faster than v1
void kernel_mm_one_fp16i8_v2(
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

    // Convert mx_fp16 and rx_fp16 to float16 arrays for vectorization
    const float16_t* mx_fp16_data = reinterpret_cast<const float16_t*>(mx_fp16);
    const float16_t* rx_fp16_data = reinterpret_cast<const float16_t*>(rx_fp16);

    // Process in chunks of 8 (assuming M is a multiple of 8)
    int k_unroll = 8;

    // Loop over N dimension
    for (int j = 0; j < N; ++j) {
        float16_t x_j = static_cast<float16_t>(x_fp16[j]);
        float16_t ry_j = static_cast<float16_t>(ry_fp16[j]);
        float16_t my_j = static_cast<float16_t>(my_fp16[j]);

        // Broadcast x_j, ry_j, my_j into NEON vectors
        float16x8_t x_j_vec = vdupq_n_f16(x_j);
        float16x8_t ry_j_vec = vdupq_n_f16(ry_j);
        float16x8_t my_j_vec = vdupq_n_f16(my_j);

        // Pointer to the start of the current row in w
        const uint8_t* w_row_ptr = &w[j * w_stride];

        int k = 0;
        for (; k <= M - k_unroll; k += k_unroll) {
            // Load w[j, k:k+7]
            uint8x8_t w_u8 = vld1_u8(&w_row_ptr[k]);

            // Convert w_u8 to float16_t and add 0.5
            uint16x8_t w_u16 = vmovl_u8(w_u8);
            float16x8_t w_vec_fp16 = vcvtq_f16_u16(w_u16);
            float16x8_t half_vec_fp16 = vdupq_n_f16(0.5f);
            w_vec_fp16 = vaddq_f16(w_vec_fp16, half_vec_fp16);

            // Load rx_fp16[k:k+7] and mx_fp16[k:k+7]
            float16x8_t rx_k_vec = vld1q_f16(&rx_fp16_data[k]);
            float16x8_t mx_k_vec = vld1q_f16(&mx_fp16_data[k]);

            // Compute temp_fp16 = (w_vec_fp16 * ry_j_vec * rx_k_vec) + my_j_vec + mx_k_vec
            float16x8_t temp_fp16 = vmulq_f16(w_vec_fp16, ry_j_vec); // (w + 0.5) * ry_j
            temp_fp16 = vmulq_f16(temp_fp16, rx_k_vec);              // * rx_k
            temp_fp16 = vaddq_f16(temp_fp16, my_j_vec);              // + my_j
            temp_fp16 = vaddq_f16(temp_fp16, mx_k_vec);              // + mx_k

            // Multiply x_j_vec * temp_fp16
            float16x8_t prod_fp16 = vmulq_f16(x_j_vec, temp_fp16);

            // Convert to float32 for accumulation
            float32x4_t prod_low = vcvt_f32_f16(vget_low_f16(prod_fp16));
            float32x4_t prod_high = vcvt_f32_f16(vget_high_f16(prod_fp16));

            // Accumulate into y_fp[k:k+7]
            float* y_ptr = &y_fp[k];
            float32x4_t y_vec_low = vld1q_f32(y_ptr);
            float32x4_t y_vec_high = vld1q_f32(y_ptr + 4);

            y_vec_low = vaddq_f32(y_vec_low, prod_low);
            y_vec_high = vaddq_f32(y_vec_high, prod_high);

            // Store the results back to y_fp
            vst1q_f32(y_ptr, y_vec_low);
            vst1q_f32(y_ptr + 4, y_vec_high);
        }

        // Handle remaining elements
        for (; k < M; ++k) {
            float16_t w_val = static_cast<float16_t>(w_row_ptr[k]) + 0.5f;
            float16_t rx_k = static_cast<float16_t>(rx_fp16[k]);
            float16_t mx_k = static_cast<float16_t>(mx_fp16[k]);

            float16_t temp_fp16 = (w_val * ry_j * rx_k) + my_j + mx_k;

            // Convert to float32 for accumulation
            float x_j_f32 = static_cast<float>(x_j);
            float temp_f32 = static_cast<float>(temp_fp16);

            y_fp[k] += x_j_f32 * temp_f32;
        }
    }
}

// v3, using openmp, private accumulators to avoid race 
void kernel_mm_one_fp16i8_v3(
    int N, int M,
    const at::Half* x_fp16,
    const uint8_t* w, int w_stride,
    const at::Half* mx_fp16,
    const at::Half* rx_fp16,
    const at::Half* my_fp16,
    const at::Half* ry_fp16,
    float* y_fp)
{
    // Initialize y_fp to zero
    std::fill(y_fp, y_fp + M, float(0.0));

    // Convert mx_fp16 and rx_fp16 to float16 arrays for vectorization
    const float16_t* mx_fp16_data = reinterpret_cast<const float16_t*>(mx_fp16);
    const float16_t* rx_fp16_data = reinterpret_cast<const float16_t*>(rx_fp16);

    int num_threads = omp_get_max_threads();
    if (N<1000 and M<1000) 
        num_threads = 1;

    // Allocate thread-local accumulators
    std::vector<std::vector<float>> y_private(num_threads, std::vector<float>(M, 0.0f));

    // Parallelize over N dimension
    // #pragma omp parallel
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        float* y_thread = y_private[thread_id].data();

        // Loop over N dimension
        #pragma omp for schedule(static)
        for (int j = 0; j < N; ++j) {
            float16_t x_j = static_cast<float16_t>(x_fp16[j]);
            float16_t ry_j = static_cast<float16_t>(ry_fp16[j]);
            float16_t my_j = static_cast<float16_t>(my_fp16[j]);

            // Broadcast x_j, ry_j, my_j into NEON vectors
            float16x8_t x_j_vec = vdupq_n_f16(x_j);
            float16x8_t ry_j_vec = vdupq_n_f16(ry_j);
            float16x8_t my_j_vec = vdupq_n_f16(my_j);

            // Pointer to the start of the current row in w
            const uint8_t* w_row_ptr = &w[j * w_stride];

            int k = 0;
            int k_unroll = 8; // Adjust based on SIMD width and M

            for (; k <= M - k_unroll; k += k_unroll) {
                // Load w[j, k:k+7]
                uint8x8_t w_u8 = vld1_u8(&w_row_ptr[k]);

                // Convert w_u8 to float16_t and add 0.5
                uint16x8_t w_u16 = vmovl_u8(w_u8);
                float16x8_t w_vec_fp16 = vcvtq_f16_u16(w_u16);
                float16x8_t half_vec_fp16 = vdupq_n_f16(0.5f);
                w_vec_fp16 = vaddq_f16(w_vec_fp16, half_vec_fp16);

                // Load rx_fp16[k:k+7] and mx_fp16[k:k+7]
                float16x8_t rx_k_vec = vld1q_f16(&rx_fp16_data[k]);
                float16x8_t mx_k_vec = vld1q_f16(&mx_fp16_data[k]);

                // Compute temp_fp16 = (w_vec_fp16 * ry_j_vec * rx_k_vec) + my_j_vec + mx_k_vec
                float16x8_t temp_fp16 = vmulq_f16(w_vec_fp16, ry_j_vec); // (w + 0.5) * ry_j
                temp_fp16 = vmulq_f16(temp_fp16, rx_k_vec);              // * rx_k
                temp_fp16 = vaddq_f16(temp_fp16, my_j_vec);              // + my_j
                temp_fp16 = vaddq_f16(temp_fp16, mx_k_vec);              // + mx_k

                // Multiply x_j_vec * temp_fp16
                float16x8_t prod_fp16 = vmulq_f16(x_j_vec, temp_fp16);

                // Convert to float32 for accumulation
                float32x4_t prod_low = vcvt_f32_f16(vget_low_f16(prod_fp16));
                float32x4_t prod_high = vcvt_f32_f16(vget_high_f16(prod_fp16));

                // Accumulate into y_thread[k:k+7]
                float* y_ptr = &y_thread[k];
                float32x4_t y_vec_low = vld1q_f32(y_ptr);
                float32x4_t y_vec_high = vld1q_f32(y_ptr + 4);

                y_vec_low = vaddq_f32(y_vec_low, prod_low);
                y_vec_high = vaddq_f32(y_vec_high, prod_high);

                // Store the results back to y_thread
                vst1q_f32(y_ptr, y_vec_low);
                vst1q_f32(y_ptr + 4, y_vec_high);
            }

            // Handle remaining elements
            for (; k < M; ++k) {
                float16_t w_val = static_cast<float16_t>(w_row_ptr[k]) + 0.5f;
                float16_t rx_k = static_cast<float16_t>(rx_fp16[k]);
                float16_t mx_k = static_cast<float16_t>(mx_fp16[k]);

                float16_t temp_fp16 = (w_val * ry_j * rx_k) + my_j + mx_k;

                // Convert to float32 for accumulation
                float x_j_f32 = static_cast<float>(x_j);
                float temp_f32 = static_cast<float>(temp_fp16);

                y_thread[k] += x_j_f32 * temp_f32;
            }
        }
    }

    // Reduce y_private into y_fp
    for (int t = 0; t < num_threads; ++t) {
        for (int k = 0; k < M; ++k) {
            y_fp[k] += y_private[t][k];
        }
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
// opt level similar to kernel_mm_one_fp16i8_v1
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

// cf: rwkv/cuda/operators.cu  kernel_mm_seq_fp16i8()
void kernel_mm_seq_fp16i8(
    const int B, const int N, const int M,
    const at::Half* x_fp16, const int x_stride,
    const uint8_t* w, const int w_stride,
    const at::Half* mx_fp16,
    const at::Half* rx_fp16,
    const at::Half* my_fp16,
    const at::Half* ry_fp16,
    float* y_fp, const int y_stride)
{
    // Convert pointers to float16_t for NEON operations
    const float16_t* mx_fp16_data = reinterpret_cast<const float16_t*>(mx_fp16);
    const float16_t* rx_fp16_data = reinterpret_cast<const float16_t*>(rx_fp16);
    const float16_t* my_fp16_data = reinterpret_cast<const float16_t*>(my_fp16);
    const float16_t* ry_fp16_data = reinterpret_cast<const float16_t*>(ry_fp16);

    // Assume w is contiguous along the columns (inner dimension)
    // Verify this outside the function if necessary

    // Process in chunks of 8 (assuming M is a multiple of 8)
    int k_unroll = 8;

    // Parallelize over batch dimension B
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; ++b) {
        // Pointers to the current batch in x and y
        const at::Half* x_batch = x_fp16 + b * x_stride;
        float* y_batch = y_fp + b * y_stride;

        // Initialize y_batch to zero
        std::fill(y_batch, y_batch + M, float(0.0));

        // Loop over N dimension
        for (int j = 0; j < N; ++j) {
            float16_t x_j = static_cast<float16_t>(x_batch[j]);
            float16_t ry_j = ry_fp16_data[j];
            float16_t my_j = my_fp16_data[j];

            // Broadcast x_j, ry_j, my_j into NEON vectors
            float16x8_t x_j_vec = vdupq_n_f16(x_j);
            float16x8_t ry_j_vec = vdupq_n_f16(ry_j);
            float16x8_t my_j_vec = vdupq_n_f16(my_j);

            // Pointer to the start of the current row in w
            const uint8_t* w_row_ptr = &w[j * w_stride];

            int k = 0;
            for (; k <= M - k_unroll; k += k_unroll) {
                // Load w[j, k:k+7]
                uint8x8_t w_u8 = vld1_u8(&w_row_ptr[k]);

                // Convert w_u8 to float16_t and add 0.5
                uint16x8_t w_u16 = vmovl_u8(w_u8);
                float16x8_t w_vec_fp16 = vcvtq_f16_u16(w_u16);
                float16x8_t half_vec_fp16 = vdupq_n_f16(0.5f);
                w_vec_fp16 = vaddq_f16(w_vec_fp16, half_vec_fp16);

                // Load rx_fp16[k:k+7] and mx_fp16[k:k+7]
                float16x8_t rx_k_vec = vld1q_f16(&rx_fp16_data[k]);
                float16x8_t mx_k_vec = vld1q_f16(&mx_fp16_data[k]);

                // Compute temp_fp16 = (w_vec_fp16 * ry_j_vec * rx_k_vec) + my_j_vec + mx_k_vec
                float16x8_t temp_fp16 = vmulq_f16(w_vec_fp16, ry_j_vec); // (w + 0.5) * ry_j
                temp_fp16 = vmulq_f16(temp_fp16, rx_k_vec);              // * rx_k
                temp_fp16 = vaddq_f16(temp_fp16, my_j_vec);              // + my_j
                temp_fp16 = vaddq_f16(temp_fp16, mx_k_vec);              // + mx_k

                // Multiply x_j_vec * temp_fp16
                float16x8_t prod_fp16 = vmulq_f16(x_j_vec, temp_fp16);

                // Convert to float32 for accumulation
                float32x4_t prod_low = vcvt_f32_f16(vget_low_f16(prod_fp16));
                float32x4_t prod_high = vcvt_f32_f16(vget_high_f16(prod_fp16));

                // Accumulate into y_batch[k:k+7]
                float* y_ptr = &y_batch[k];
                float32x4_t y_vec_low = vld1q_f32(y_ptr);
                float32x4_t y_vec_high = vld1q_f32(y_ptr + 4);

                y_vec_low = vaddq_f32(y_vec_low, prod_low);
                y_vec_high = vaddq_f32(y_vec_high, prod_high);

                // Store the results back to y_batch
                vst1q_f32(y_ptr, y_vec_low);
                vst1q_f32(y_ptr + 4, y_vec_high);
            }

            // Handle remaining elements
            for (; k < M; ++k) {
                float16_t w_val = static_cast<float16_t>(w_row_ptr[k]) + 0.5f;
                float16_t rx_k = rx_fp16_data[k];
                float16_t mx_k = mx_fp16_data[k];

                float16_t temp_fp16 = (w_val * ry_j * rx_k) + my_j + mx_k;

                // Convert to float32 for accumulation
                float x_j_f32 = static_cast<float>(x_j);
                float temp_f32 = static_cast<float>(temp_fp16);

                y_batch[k] += x_j_f32 * temp_f32;
            }
        }
    }
}

void kernel_mm_seq_fp32i8(
    const int B, const int N, const int M,
    const float* x_fp32, const int x_stride,
    const uint8_t* w, const int w_stride,
    const float* mx_fp32,
    const float* rx_fp32,
    const float* my_fp32,
    const float* ry_fp32,
    float* y_fp, const int y_stride)
{
    // Parallelize over the batch dimension B
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; ++b) {
        // Pointers to the current batch in x and y
        const float* x_batch = x_fp32 + b * x_stride;
        float* y_batch = y_fp + b * y_stride;

        // Initialize y_batch to zero
        std::fill(y_batch, y_batch + M, float(0.0f));

        // Loop over N dimension
        for (int j = 0; j < N; ++j) {
            float x_j = x_batch[j];
            float ry_j = ry_fp32[j];
            float my_j = my_fp32[j];

            // Broadcast x_j, ry_j, my_j into NEON vectors
            float32x4_t x_j_vec = vdupq_n_f32(x_j);
            float32x4_t ry_j_vec = vdupq_n_f32(ry_j);
            float32x4_t my_j_vec = vdupq_n_f32(my_j);

            // Pointer to the start of the current row in w
            const uint8_t* w_row_ptr = &w[j * w_stride];

            int k = 0;
            int k_unroll = 4; // Process 4 elements at a time
            for (; k <= M - k_unroll; k += k_unroll) {
                // Load w[j, k:k+3]
                uint8x8_t w_u8 = vld1_u8(&w_row_ptr[k]); // Load 8 bytes
                uint16x8_t w_u16 = vmovl_u8(w_u8);       // Expand to uint16x8_t
                uint16x4_t w_u16_low = vget_low_u16(w_u16); // Lower 4 elements
                uint32x4_t w_u32 = vmovl_u16(w_u16_low); // Expand to uint32x4_t
                float32x4_t w_vec = vcvtq_f32_u32(w_u32); // Convert to float32
                float32x4_t half_vec = vdupq_n_f32(0.5f);
                w_vec = vaddq_f32(w_vec, half_vec); // w_vec = w_vec + 0.5

                // Load rx_fp32[k:k+3] and mx_fp32[k:k+3]
                float32x4_t rx_k_vec = vld1q_f32(&rx_fp32[k]);
                float32x4_t mx_k_vec = vld1q_f32(&mx_fp32[k]);

                // Compute temp = (w_vec * ry_j_vec * rx_k_vec) + my_j_vec + mx_k_vec
                float32x4_t temp = vmulq_f32(w_vec, ry_j_vec); // temp = (w + 0.5) * ry_j
                temp = vmulq_f32(temp, rx_k_vec);              // temp = temp * rx_k
                temp = vaddq_f32(temp, my_j_vec);              // temp = temp + my_j
                temp = vaddq_f32(temp, mx_k_vec);              // temp = temp + mx_k

                // Multiply x_j_vec * temp
                float32x4_t prod = vmulq_f32(x_j_vec, temp);

                // Accumulate into y_batch[k:k+3]
                float32x4_t y_vec = vld1q_f32(&y_batch[k]);
                y_vec = vaddq_f32(y_vec, prod);
                vst1q_f32(&y_batch[k], y_vec);
            }

            // Handle remaining elements
            for (; k < M; ++k) {
                float w_val = static_cast<float>(w_row_ptr[k]) + 0.5f;
                float rx_k = rx_fp32[k];
                float mx_k = mx_fp32[k];

                float temp = (w_val * ry_j * rx_k) + my_j + mx_k;
                y_batch[k] += x_j * temp;
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////
// Python binding
// Prepares the data and calls the kernel, ensuring proper tensor shapes and data types
///////////////////////////////////////////////////////////////////////////////////////////

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
    torch::Tensor ry_fp16,
    int version /*0,1,2,3..*/)
{
#if DEBUG_KERNEL
    auto start = std::chrono::high_resolution_clock::now();
#endif

    // Ensure tensors are contiguous
    x_fp16 = x_fp16.contiguous();
    w_uint8 = w_uint8.contiguous();
    mx_fp16 = mx_fp16.contiguous();
    rx_fp16 = rx_fp16.contiguous();
    my_fp16 = my_fp16.contiguous().view(-1);
    ry_fp16 = ry_fp16.contiguous().view(-1);

    // Validate that w is contiguous in the inner dimension
    TORCH_CHECK(w_uint8.stride(1) == 1, "w_uint8 must be contiguous in the inner dimension");

    int N = x_fp16.size(0);
    int M = w_uint8.size(1);
    int w_stride = M; // Since w is row-major and contiguous

    torch::Tensor y = torch::zeros({M}, torch::dtype(torch::kFloat32));

#if DEBUG_KERNEL
    auto start1 = std::chrono::high_resolution_clock::now();
#endif

    switch (version)
    {
    case 1:
        kernel_mm_one_fp16i8_v1(
            N, M,
            reinterpret_cast<const at::Half*>(x_fp16.data_ptr<at::Half>()),
            w_uint8.data_ptr<uint8_t>(), w_stride,
            reinterpret_cast<const at::Half*>(mx_fp16.data_ptr<at::Half>()),
            reinterpret_cast<const at::Half*>(rx_fp16.data_ptr<at::Half>()),
            reinterpret_cast<const at::Half*>(my_fp16.data_ptr<at::Half>()),
            reinterpret_cast<const at::Half*>(ry_fp16.data_ptr<at::Half>()),
            y.data_ptr<float>());
        break;
    case 2:
        kernel_mm_one_fp16i8_v2(
            N, M,
            reinterpret_cast<const at::Half*>(x_fp16.data_ptr<at::Half>()),
            w_uint8.data_ptr<uint8_t>(), w_stride,
            reinterpret_cast<const at::Half*>(mx_fp16.data_ptr<at::Half>()),
            reinterpret_cast<const at::Half*>(rx_fp16.data_ptr<at::Half>()),
            reinterpret_cast<const at::Half*>(my_fp16.data_ptr<at::Half>()),
            reinterpret_cast<const at::Half*>(ry_fp16.data_ptr<at::Half>()),
            y.data_ptr<float>());
        break;
    case 3:
        kernel_mm_one_fp16i8_v3(
            N, M,
            reinterpret_cast<const at::Half*>(x_fp16.data_ptr<at::Half>()),
            w_uint8.data_ptr<uint8_t>(), w_stride,
            reinterpret_cast<const at::Half*>(mx_fp16.data_ptr<at::Half>()),
            reinterpret_cast<const at::Half*>(rx_fp16.data_ptr<at::Half>()),
            reinterpret_cast<const at::Half*>(my_fp16.data_ptr<at::Half>()),
            reinterpret_cast<const at::Half*>(ry_fp16.data_ptr<at::Half>()),
            y.data_ptr<float>());
        break;
    case 4:     // dynamic, based on N,M
        if (N < 1000 && M < 1000) { // no openmp
            kernel_mm_one_fp16i8_v2(
                N, M,
                reinterpret_cast<const at::Half*>(x_fp16.data_ptr<at::Half>()),
                w_uint8.data_ptr<uint8_t>(), w_stride,
                reinterpret_cast<const at::Half*>(mx_fp16.data_ptr<at::Half>()),
                reinterpret_cast<const at::Half*>(rx_fp16.data_ptr<at::Half>()),
                reinterpret_cast<const at::Half*>(my_fp16.data_ptr<at::Half>()),
                reinterpret_cast<const at::Half*>(ry_fp16.data_ptr<at::Half>()),
                y.data_ptr<float>());
        }
        else {  // openmp
            kernel_mm_one_fp16i8_v3(
                N, M,
                reinterpret_cast<const at::Half*>(x_fp16.data_ptr<at::Half>()),
                w_uint8.data_ptr<uint8_t>(), w_stride,
                reinterpret_cast<const at::Half*>(mx_fp16.data_ptr<at::Half>()),
                reinterpret_cast<const at::Half*>(rx_fp16.data_ptr<at::Half>()),
                reinterpret_cast<const at::Half*>(my_fp16.data_ptr<at::Half>()),
                reinterpret_cast<const at::Half*>(ry_fp16.data_ptr<at::Half>()),
                y.data_ptr<float>());
        }
        break;
    default:
        TORCH_CHECK(false, "Invalid version number");
        break;
    }

#if DEBUG_KERNEL
    auto end  = std::chrono::high_resolution_clock::now();
    printf(">>>>>> %s total: %.2f ms, convert: %.2f ms, kernel %.2f \n", __func__, 
        std::chrono::duration<float, std::milli>(end - start).count(),
        std::chrono::duration<float, std::milli>(start1 - start).count(),
        std::chrono::duration<float, std::milli>(end - start1).count()
        );
#endif

    return y;
}

/*
    x: Shape (B, N)
    w: Shape (N, M)
    mx: Shape (M,)
    rx: Shape (M,)
    my: Shape (N, 1)
    ry: Shape (N, 1)
*/
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
    my_fp16 = my_fp16.contiguous().view(-1); // Ensure shape is (N,)
    ry_fp16 = ry_fp16.contiguous().view(-1); // Ensure shape is (N,)

    // Validate tensor data types
    TORCH_CHECK(x_fp16.dtype() == torch::kFloat16, "x_fp16 must be Float16");
    TORCH_CHECK(w_uint8.dtype() == torch::kUInt8, "w_uint8 must be UInt8");
    TORCH_CHECK(mx_fp16.dtype() == torch::kFloat16, "mx_fp16 must be Float16");
    TORCH_CHECK(rx_fp16.dtype() == torch::kFloat16, "rx_fp16 must be Float16");
    TORCH_CHECK(my_fp16.dtype() == torch::kFloat16, "my_fp16 must be Float16");
    TORCH_CHECK(ry_fp16.dtype() == torch::kFloat16, "ry_fp16 must be Float16");

    // Get dimensions
    int B = x_fp16.size(0);
    int N = x_fp16.size(1);
    int M = w_uint8.size(1);

    // Strides
    int x_stride = x_fp16.stride(0);
    int y_stride = M; // Output y will have shape (B, M)
    int w_stride = M; // Assuming w is row-major and contiguous

    // Ensure that w is contiguous along the columns
    TORCH_CHECK(w_uint8.stride(1) == 1, "w_uint8 must be contiguous along the inner dimension");

    // Allocate output tensor y
    torch::Tensor y = torch::empty({B, M}, torch::dtype(torch::kFloat32));

    // Call the kernel function
    kernel_mm_seq_fp16i8(
        B, N, M,
        reinterpret_cast<const at::Half*>(x_fp16.data_ptr<at::Half>()), x_stride,
        w_uint8.data_ptr<uint8_t>(), w_stride,
        reinterpret_cast<const at::Half*>(mx_fp16.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(rx_fp16.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(my_fp16.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(ry_fp16.data_ptr<at::Half>()),
        y.data_ptr<float>(), y_stride
    );

    return y;
}

torch::Tensor mm_seq_fp32i8(
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
    my_fp32 = my_fp32.contiguous().view(-1); // Ensure shape is (N,)
    ry_fp32 = ry_fp32.contiguous().view(-1); // Ensure shape is (N,)

    // Validate tensor data types
    TORCH_CHECK(x_fp32.dtype() == torch::kFloat32, "x_fp32 must be Float32");
    TORCH_CHECK(w_uint8.dtype() == torch::kUInt8, "w_uint8 must be UInt8");
    TORCH_CHECK(mx_fp32.dtype() == torch::kFloat32, "mx_fp32 must be Float32");
    TORCH_CHECK(rx_fp32.dtype() == torch::kFloat32, "rx_fp32 must be Float32");
    TORCH_CHECK(my_fp32.dtype() == torch::kFloat32, "my_fp32 must be Float32");
    TORCH_CHECK(ry_fp32.dtype() == torch::kFloat32, "ry_fp32 must be Float32");

    // Get dimensions
    int B = x_fp32.size(0);
    int N = x_fp32.size(1);
    int M = w_uint8.size(1);

    // Strides
    int x_stride = x_fp32.stride(0);
    int y_stride = M; // Output y will have shape (B, M)
    int w_stride = M; // Assuming w is row-major and contiguous

    // Ensure that w is contiguous along the columns
    TORCH_CHECK(w_uint8.stride(1) == 1, "w_uint8 must be contiguous along the inner dimension");

    // Allocate output tensor y
    torch::Tensor y = torch::empty({B, M}, torch::dtype(torch::kFloat32));

    // Call the kernel function
    kernel_mm_seq_fp32i8(
        B, N, M,
        x_fp32.data_ptr<float>(), x_stride,
        w_uint8.data_ptr<uint8_t>(), w_stride,
        mx_fp32.data_ptr<float>(),
        rx_fp32.data_ptr<float>(),
        my_fp32.data_ptr<float>(),
        ry_fp32.data_ptr<float>(),
        y.data_ptr<float>(), y_stride
    );

    return y;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm_one_fp16i8", &mm_one_fp16i8, "Matrix multiplication with int8 weights and float16 inputs (ARM Cortex-A76 optimized)");
    m.def("mm_one_fp32i8", &mm_one_fp32i8, "Matrix multiplication with int8 weights and float32 inputs (ARM Cortex-A76 optimized)");
    m.def("mm_seq_fp16i8", &mm_seq_fp16i8, "Sequential matrix multiplication with int8 weights and float16 inputs (ARM Cortex-A76 optimized)");
    m.def("mm_seq_fp32i8", &mm_seq_fp32i8, "Sequential matrix multiplication with int8 weights and float32 inputs (ARM Cortex-A76 optimized)");
}
