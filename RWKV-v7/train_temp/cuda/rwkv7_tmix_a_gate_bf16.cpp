#include <torch/extension.h>

#include <vector>

torch::Tensor tmix_a_gate_forward_cuda(
    torch::Tensor a0,
    torch::Tensor a12);

std::vector<torch::Tensor> tmix_a_gate_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor a0,
    torch::Tensor a12);

namespace {

void check_bf16_cuda(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, name, " must be bf16");
}

} // namespace

torch::Tensor forward(torch::Tensor a0, torch::Tensor a12) {
    check_bf16_cuda(a0, "a0");
    check_bf16_cuda(a12, "a12");
    TORCH_CHECK(a12.dim() == 3, "a12 must have shape [B, T, C]");
    TORCH_CHECK(a0.numel() == a12.size(2), "a0 must have C elements");
    return tmix_a_gate_forward_cuda(a0, a12);
}

std::vector<torch::Tensor> backward(torch::Tensor grad_out, torch::Tensor a0, torch::Tensor a12) {
    check_bf16_cuda(grad_out, "grad_out");
    check_bf16_cuda(a0, "a0");
    check_bf16_cuda(a12, "a12");
    TORCH_CHECK(a12.dim() == 3, "a12 must have shape [B, T, C]");
    TORCH_CHECK(grad_out.sizes() == a12.sizes(), "grad_out shape mismatch");
    TORCH_CHECK(a0.numel() == a12.size(2), "a0 must have C elements");
    return tmix_a_gate_backward_cuda(grad_out, a0, a12);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "RWKV tmix a gate forward (bf16)");
    m.def("backward", &backward, "RWKV tmix a gate backward (bf16)");
}

TORCH_LIBRARY(rwkv7_tmix_a_gate_bf16, m) {
    m.def("forward(Tensor a0, Tensor a12) -> Tensor");
    m.def("backward(Tensor grad_out, Tensor a0, Tensor a12) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(rwkv7_tmix_a_gate_bf16, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}
