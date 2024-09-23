#include <torch/extension.h>
#include "ATen/ATen.h"

typedef at::Half bf16;
// typedef at::BFloat16 bf16;

void cuda_forward(int B, int T, int C, int H, bf16 *r, bf16 *w, bf16 *k, bf16 *v, bf16 *a, bf16 *b, bf16 *y);

void forward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &y) {
    cuda_forward(B, T, C, H, r.data_ptr<bf16>(), w.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), a.data_ptr<bf16>(), b.data_ptr<bf16>(), y.data_ptr<bf16>());
}

TORCH_LIBRARY(wkv7, m) {
    m.def("forward", forward);
}
