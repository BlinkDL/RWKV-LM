// =============================================================================
// common.cuh - shim header (replace with llama.cpp's real common.cuh in build)
// =============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cmath>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ---- minimal shim definitions (the real file lives in llama.cpp; this shim
// is only used for standalone compilation of the wkv7 kernel for review).
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}
