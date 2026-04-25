#include <torch/extension.h>

torch::Tensor l2wrap_backward_v2_cuda(torch::Tensor y);

namespace {

void check_cuda_contiguous(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

} // namespace

torch::Tensor backward(torch::Tensor y) {
    check_cuda_contiguous(y, "y");
    TORCH_CHECK(y.dim() >= 2, "y must have at least 2 dims");
    TORCH_CHECK(y.size(-1) == 65536, "l2wrap v2 currently expects vocab size 65536");
    TORCH_CHECK(
        y.scalar_type() == torch::kBFloat16 || y.scalar_type() == torch::kFloat32,
        "y must be bf16 or fp32");
    return l2wrap_backward_v2_cuda(y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward", &backward, "RWKV L2Wrap backward v2 (bf16/fp32 CUDA)");
}
