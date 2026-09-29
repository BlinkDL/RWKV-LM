// =============================================================================
// rwkv7-quant.h
// =============================================================================
#pragma once

#include <cstdint>

namespace rwkv7 {

void quantize_int8_per_row(
        const float * src, int8_t * dst, float * scale,
        int N, int K);

void quantize_uint4_per_row(
        const float * src, uint8_t * dst, float * scale,
        int N, int K);

void state_pack(
        const float * src, int8_t * dst, float * scale,
        int N, int K);

void state_unpack(
        const int8_t * src, const float * scale, float * dst,
        int N, int K);

}  // namespace rwkv7
