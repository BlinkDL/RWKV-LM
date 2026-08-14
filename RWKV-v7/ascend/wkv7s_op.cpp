#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "ATen/ATen.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

// typedef at::BFloat16 bf16;

void ascend_forward(int B, int T, int C, int H, void *state, void *r, void *w, void *k, void *v, void *a, void *b, void *y, void* stream);

void forward(int64_t B, int64_t T, int64_t C, int64_t H,
             torch::Tensor &state, torch::Tensor &r, torch::Tensor &w, torch::Tensor &k,
             torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &y)
{
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    ascend_forward(B, T, C, H,
                   state.data_ptr(), r.data_ptr(), w.data_ptr(), k.data_ptr(),
                   v.data_ptr(), a.data_ptr(), b.data_ptr(), y.data_ptr(), acl_stream);
}

PYBIND11_MODULE(wkv7s, m)
{
    m.def("forward", &forward, "Forward with original order");
}