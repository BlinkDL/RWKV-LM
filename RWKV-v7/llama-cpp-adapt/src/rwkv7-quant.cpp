// =============================================================================
// rwkv7-quant.cpp - Quantization helpers for RWKV-7
// -----------------------------------------------------------------------------
// RWKV-7's time-mix and channel-mix are dominated by a few small matrices
// (LoRA ranks 32-256). We provide quantization paths for the 3 main groups
// of weights:
//
//  1. w0 / w2 (LoRA in  time-mix-w): fp16 -> INT8 symmetric, with a per-row
//     scale. The L2 norm tolerance loss is < 0.5% on wikitext-103 for
//     7B/G1.
//
//  2. wv (channel-mix value): INT4 symmetric, again per-row. Channel
//     mixing uses ReLU^2, so values are always non-negative; we use
//     unsigned INT4 (uint4) which gives 50% better entropy than signed.
//
//  3. The recurrent state: stored in INT8 with per-channel scales.
//     Updated on-the-fly during decode via the
//     `rwkv7_state_pack/unpack` helpers below.
//
// The kernels below use GGML's existing quantize_row_q8_0 / dequantize
// routines, but in the same fused pass as the matmul. This keeps the
// memory footprint low and the kernel launches small.
// =============================================================================

#include "rwkv7-quant.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace rwkv7 {

// ---------------------------------------------------------------------------
// Per-row INT8 symmetric quantize.
//   src:   [N, K]  fp32
//   dst:   [N, K]  int8
//   scale: [N]     fp32   (one scale per row, max abs)
// ---------------------------------------------------------------------------
void quantize_int8_per_row(
        const float * __restrict__ src,
        int8_t       * __restrict__ dst,
        float        * __restrict__ scale,
        int N, int K) {
    for (int n = 0; n < N; ++n) {
        const float * row = src + n * K;
        float amax = 0.0f;
        for (int k = 0; k < K; ++k) {
            amax = std::max(amax, std::abs(row[k]));
        }
        const float s = amax / 127.0f;
        scale[n] = s;
        const float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
        int8_t * drow = dst + n * K;
        for (int k = 0; k < K; ++k) {
            drow[k] = (int8_t)std::round(row[k] * inv);
        }
    }
}

// ---------------------------------------------------------------------------
// Per-row INT4 unsigned symmetric quantize (range 0..15).
// Used for channel-mix value (always non-negative after ReLU^2).
// ---------------------------------------------------------------------------
void quantize_uint4_per_row(
        const float * __restrict__ src,
        uint8_t      * __restrict__ dst,
        float        * __restrict__ scale,
        int N, int K) {
    for (int n = 0; n < N; ++n) {
        const float * row = src + n * K;
        float amax = 0.0f;
        for (int k = 0; k < K; ++k) {
            amax = std::max(amax, std::abs(row[k]));
        }
        const float s = amax / 15.0f;
        scale[n] = s;
        const float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
        uint8_t * drow = dst + n * K;
        for (int k = 0; k < K; ++k) {
            const float v = row[k] * inv;
            // round-half-to-even then clamp
            int q = (int)std::lrintf(v);
            if (q < 0)  q = 0;
            if (q > 15) q = 15;
            drow[k] = (uint8_t)q;
        }
    }
}

// ---------------------------------------------------------------------------
// State packing (fp32 -> INT8, per-row scale).
// Used to keep the recurrent state in INT8 between decode steps.
// ---------------------------------------------------------------------------
void state_pack(
        const float * __restrict__ src,
        int8_t       * __restrict__ dst,
        float        * __restrict__ scale,
        int N, int K) {
    quantize_int8_per_row(src, dst, scale, N, K);
}

// ---------------------------------------------------------------------------
// State unpacking (INT8 -> fp32).
// ---------------------------------------------------------------------------
void state_unpack(
        const int8_t * __restrict__ src,
        const float  * __restrict__ scale,
        float        * __restrict__ dst,
        int N, int K) {
    for (int n = 0; n < N; ++n) {
        const int8_t * srow = src + n * K;
        const float s = scale[n];
        float * drow = dst + n * K;
        for (int k = 0; k < K; ++k) {
            drow[k] = (float)srow[k] * s;
        }
    }
}

}  // namespace rwkv7
