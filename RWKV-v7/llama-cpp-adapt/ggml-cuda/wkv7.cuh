// =============================================================================
// wkv7.cuh - public WKV7 CUDA kernel declarations
// =============================================================================
#pragma once
#include <cstdint>

struct ggml_backend_cuda_context;
struct ggml_tensor;

void ggml_cuda_op_rwkv7_wkv7(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst);
