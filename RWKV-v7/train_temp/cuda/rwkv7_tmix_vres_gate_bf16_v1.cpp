#include <torch/extension.h>

#include <vector>

torch::Tensor tmix_vres_gate_forward_cuda(
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12);

std::vector<torch::Tensor> tmix_vres_gate_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12);

namespace {

void check_bf16_cuda(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, name, " must be bf16");
}

} // namespace

torch::Tensor forward(
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12) {
    check_bf16_cuda(v, "v");
    check_bf16_cuda(v_first, "v_first");
    check_bf16_cuda(v0, "v0");
    check_bf16_cuda(v12, "v12");
    TORCH_CHECK(v.dim() == 3, "v must have shape [B, T, C]");
    TORCH_CHECK(v_first.sizes() == v.sizes(), "v_first shape mismatch");
    TORCH_CHECK(v12.sizes() == v.sizes(), "v12 shape mismatch");
    TORCH_CHECK(v0.numel() == v.size(2), "v0 must have C elements");
    return tmix_vres_gate_forward_cuda(v, v_first, v0, v12);
}

std::vector<torch::Tensor> backward(
    torch::Tensor grad_out,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12) {
    check_bf16_cuda(grad_out, "grad_out");
    check_bf16_cuda(v, "v");
    check_bf16_cuda(v_first, "v_first");
    check_bf16_cuda(v0, "v0");
    check_bf16_cuda(v12, "v12");
    TORCH_CHECK(v.dim() == 3, "v must have shape [B, T, C]");
    TORCH_CHECK(grad_out.sizes() == v.sizes(), "grad_out shape mismatch");
    TORCH_CHECK(v_first.sizes() == v.sizes(), "v_first shape mismatch");
    TORCH_CHECK(v12.sizes() == v.sizes(), "v12 shape mismatch");
    TORCH_CHECK(v0.numel() == v.size(2), "v0 must have C elements");
    return tmix_vres_gate_backward_cuda(grad_out, v, v_first, v0, v12);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "RWKV tmix value residual gate forward v1 (bf16)");
    m.def("backward", &backward, "RWKV tmix value residual gate backward v1 (bf16)");
}

TORCH_LIBRARY(rwkv7_tmix_vres_gate_bf16_v1, m) {
    m.def("forward(Tensor v, Tensor v_first, Tensor v0, Tensor v12) -> Tensor");
    m.def("backward(Tensor grad_out, Tensor v, Tensor v_first, Tensor v0, Tensor v12) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(rwkv7_tmix_vres_gate_bf16_v1, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}
