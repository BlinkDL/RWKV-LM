// =============================================================================
// rwkv7_wkv7.h - public declarations for the CPU WKV7 kernel.
// =============================================================================
#pragma once

namespace rwkv7 {

void wkv7_forward(
        const void * r_in,
        const void * w_in,
        const void * k_in,
        const void * v_in,
        const void * kk_neg_in,
        const void * kk_a_in,
        const void * state_in,
        void * y_out,
        void * state_out,
        int B, int H, int T, int D,
        int dtype_bytes);

}  // namespace rwkv7
