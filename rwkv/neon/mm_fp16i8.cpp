#include <cstdint>
#include <vector>
#include <arm_neon.h> // For NEON intrinsics
#include <omp.h>     // For OpenMP
#include <cstring>    // For memset
#include <torch/extension.h>


#if 0
void kernel_mm_seq_fp16i8(
    const int B, const int N, const int M,
    const __fp16* x, const int x_stride,
    const uint8_t* w, const int w_stride,
    const __fp16* mx,
    const __fp16* rx,
    const __fp16* my,
    const __fp16* ry,
    __fp16* y, const int y_stride) {

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
                float16x4_t x_vec_fp16 = vld1_f16(&x[i * x_stride + j]);
                float32x4_t x_vec = vcvt_f32_f16(x_vec_fp16);  // Convert to float32

                // Load w[j * w_stride + k:k+3] and dequantize
                uint8x8_t w_u8 = vld1_u8(&w[j * w_stride + k]);
                uint16x8_t w_u16 = vmovl_u8(w_u8);
                float16x4_t w_fp16 = vcvt_f16_u16(vget_low_u16(w_u16));  // Convert to FP16
                float32x4_t w_vec = vaddq_f32(vcvt_f32_f16(w_fp16), vdupq_n_f32(0.5f));

                // Load rx[k], ry[j:j+3], mx[k], and my[j:j+3]
                float16x4_t ry_vec_fp16 = vld1_f16(&ry[j]);
                float16x4_t my_vec_fp16 = vld1_f16(&my[j]);
                float32x4_t ry_vec = vcvt_f32_f16(ry_vec_fp16);
                float32x4_t my_vec = vcvt_f32_f16(my_vec_fp16);

                float32x4_t rx_k = vdupq_n_f32(vcvt_f32_f16(vld1_f16(&rx[k])));
                float32x4_t mx_k = vdupq_n_f32(vcvt_f32_f16(vld1_f16(&mx[k])));

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
                float x_val = (float)x[i * x_stride + j];
                float w_val = (float(w[j * w_stride + k]) + 0.5f);
                float rx_val = (float)rx[k];
                float ry_val = (float)ry[j];
                float mx_val = (float)mx[k];
                float my_val = (float)my[j];

                y_local += x_val * (w_val * rx_val * ry_val + mx_val + my_val);
            }

            // Store the result in y
            y[i * y_stride + k] = static_cast<__fp16>(y_local);
        }
    }
}
#endif


#include <arm_neon.h>
#include <torch/extension.h>
#include <omp.h>
#include <cstring>

// Function to perform the computation
void mm_one_fp16i8_cpu_arm_fp16(
    int N, int M,
    const at::Half* x_fp16,
    const uint8_t* w, int w_stride,
    const at::Half* mx_fp16,
    const at::Half* rx_fp16,
    const at::Half* my_fp16,
    const at::Half* ry_fp16,
    at::Half* y_fp16)
{
    // Ensure that at::Half and __fp16 have the same size
    static_assert(sizeof(at::Half) == sizeof(__fp16), "at::Half and __fp16 must be the same size");

    // Initialize y to zero
    std::fill(y_fp16, y_fp16 + M, at::Half(0.0));

    // Parallelize over the M dimension using OpenMP
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < M; ++k) {
        // Initialize local accumulator in FP16
        float16x8_t y_local_vec = vdupq_n_f16(0.0f);

        int j = 0;
        const int unroll_size = 8; // NEON processes 8 FP16 elements at a time

        // Broadcast rx[k] and mx[k] to all elements
        __fp16 rx_k = static_cast<__fp16>(rx_fp16[k]);
        __fp16 mx_k = static_cast<__fp16>(mx_fp16[k]);
        float16x8_t rx_k_vec = vdupq_n_f16(rx_k);
        float16x8_t mx_k_vec = vdupq_n_f16(mx_k);

        for (; j <= N - unroll_size; j += unroll_size) {
            // Load x[j:j+7] into a NEON register
            const __fp16* x_ptr = reinterpret_cast<const __fp16*>(&x_fp16[j]);
            float16x8_t x_vec = vld1q_f16(x_ptr);

            // Load w[j:j+7, k], expand to uint16_t
            const uint8_t* w_ptr = &w[j * w_stride + k];
            uint8x8_t w_u8 = vld1_u8(w_ptr);
            uint16x8_t w_u16 = vmovl_u8(w_u8);

            // Convert w_u16 to float16 and add 0.5f
            float16x8_t w_vec = vcvtq_f16_u16(w_u16);
            w_vec = vaddq_f16(w_vec, vdupq_n_f16(0.5f));

            // Load ry[j:j+7] and my[j:j+7]
            const __fp16* ry_ptr = reinterpret_cast<const __fp16*>(&ry_fp16[j]);
            float16x8_t ry_vec = vld1q_f16(ry_ptr);

            const __fp16* my_ptr = reinterpret_cast<const __fp16*>(&my_fp16[j]);
            float16x8_t my_vec = vld1q_f16(my_ptr);

            // Compute temp = (w_val * rx_k * ry_j) + mx_k + my_j
            float16x8_t temp = vaddq_f16(
                vmulq_f16(w_vec, vmulq_f16(rx_k_vec, ry_vec)),
                vaddq_f16(mx_k_vec, my_vec)
            );

            // Multiply x_vec with temp and accumulate
            y_local_vec = vaddq_f16(y_local_vec, vmulq_f16(x_vec, temp));
        }

        // Sum the elements of y_local_vec manually
        __fp16 y_local = 0.0f;
        __fp16 y_elements[8];
        vst1q_f16(y_elements, y_local_vec);
        for (int idx = 0; idx < 8; ++idx) {
            y_local += y_elements[idx];
        }

        // Handle remaining elements
        for (; j < N; ++j) {
            __fp16 x_j = static_cast<__fp16>(x_fp16[j]);
            __fp16 w_val = static_cast<__fp16>(w[j * w_stride + k]) + __fp16(0.5f);
            __fp16 ry_j = static_cast<__fp16>(ry_fp16[j]);
            __fp16 my_j = static_cast<__fp16>(my_fp16[j]);

            __fp16 temp = (w_val * rx_k * ry_j) + mx_k + my_j;
            y_local += x_j * temp;
        }

        // Store the result
        y_fp16[k] = static_cast<at::Half>(y_local);
    }
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

    torch::Tensor y = torch::empty({M}, torch::dtype(torch::kFloat16));

    mm_one_fp16i8_cpu_arm_fp16(
        N, M,
        x_fp16.data_ptr<at::Half>(),
        w_uint8.data_ptr<uint8_t>(), w_stride,
        mx_fp16.data_ptr<at::Half>(),
        rx_fp16.data_ptr<at::Half>(),
        my_fp16.data_ptr<at::Half>(),
        ry_fp16.data_ptr<at::Half>(),
        y.data_ptr<at::Half>()
    );

    return y;
}

#if 0
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

    torch::Tensor y = torch::empty({B, M}, torch::kFloat16);

    kernel_mm_seq_fp16i8(
        B, N, M,
        reinterpret_cast<__fp16*>(x_fp16.data_ptr<at::Half>()), x_stride,
        w_uint8.data_ptr<uint8_t>(), w_stride,
        reinterpret_cast<__fp16*>(mx_fp16.data_ptr<at::Half>()),
        reinterpret_cast<__fp16*>(rx_fp16.data_ptr<at::Half>()),
        reinterpret_cast<__fp16*>(my_fp16.data_ptr<at::Half>()),
        reinterpret_cast<__fp16*>(ry_fp16.data_ptr<at::Half>()),
        reinterpret_cast<__fp16*>(y.data_ptr<at::Half>()), y_stride
    );

    return y;
}
#endif


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm_one_fp16i8", &mm_one_fp16i8, "Matrix multiplication with int8 weights and float16 inputs (ARM Cortex-A76 optimized)");
}
