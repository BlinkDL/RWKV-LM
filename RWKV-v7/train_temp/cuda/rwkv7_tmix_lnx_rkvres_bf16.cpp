#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> tmix_lnx_rkvres_forward_cuda(
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias);

std::vector<torch::Tensor> tmix_lnx_rkvres_backward_cuda(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor mean,
    torch::Tensor rstd,
    torch::Tensor scale);

namespace {

constexpr int64_t kHeadSize = 64;

void check_bf16_cuda(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, name, " must be bf16");
}

void check_f32_cuda(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, name, " must be fp32");
}

void check_vec(const torch::Tensor& x, int64_t c, const char* name) {
    TORCH_CHECK(x.dim() == 1, name, " must have shape [C]");
    TORCH_CHECK(x.size(0) == c, name, " shape mismatch");
}

} // namespace

std::vector<torch::Tensor> tmix_lnx_rkvres_forward(
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias) {
    check_bf16_cuda(x, "x");
    check_bf16_cuda(r, "r");
    check_bf16_cuda(k, "k");
    check_bf16_cuda(v, "v");
    check_bf16_cuda(r_k, "r_k");
    check_bf16_cuda(weight, "weight");
    check_bf16_cuda(bias, "bias");
    TORCH_CHECK(x.dim() == 3, "x must have shape [B, T, C]");
    TORCH_CHECK(r.sizes() == x.sizes(), "r shape mismatch");
    TORCH_CHECK(k.sizes() == x.sizes(), "k shape mismatch");
    TORCH_CHECK(v.sizes() == x.sizes(), "v shape mismatch");
    const int64_t c = x.size(2);
    TORCH_CHECK(c % kHeadSize == 0, "C must be divisible by 64");
    TORCH_CHECK(r_k.dim() == 2, "r_k must have shape [H, 64]");
    TORCH_CHECK(r_k.size(0) == c / kHeadSize && r_k.size(1) == kHeadSize, "r_k shape mismatch");
    check_vec(weight, c, "weight");
    check_vec(bias, c, "bias");
    return tmix_lnx_rkvres_forward_cuda(x, r, k, v, r_k, weight, bias);
}

std::vector<torch::Tensor> tmix_lnx_rkvres_backward(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor mean,
    torch::Tensor rstd,
    torch::Tensor scale) {
    check_bf16_cuda(grad_y, "grad_y");
    check_bf16_cuda(x, "x");
    check_bf16_cuda(r, "r");
    check_bf16_cuda(k, "k");
    check_bf16_cuda(v, "v");
    check_bf16_cuda(r_k, "r_k");
    check_bf16_cuda(weight, "weight");
    check_f32_cuda(mean, "mean");
    check_f32_cuda(rstd, "rstd");
    check_f32_cuda(scale, "scale");
    TORCH_CHECK(grad_y.sizes() == x.sizes(), "grad_y shape mismatch");
    TORCH_CHECK(r.sizes() == x.sizes(), "r shape mismatch");
    TORCH_CHECK(k.sizes() == x.sizes(), "k shape mismatch");
    TORCH_CHECK(v.sizes() == x.sizes(), "v shape mismatch");
    const int64_t c = x.size(2);
    TORCH_CHECK(c % kHeadSize == 0, "C must be divisible by 64");
    TORCH_CHECK(r_k.dim() == 2, "r_k must have shape [H, 64]");
    TORCH_CHECK(r_k.size(0) == c / kHeadSize && r_k.size(1) == kHeadSize, "r_k shape mismatch");
    check_vec(weight, c, "weight");
    TORCH_CHECK(mean.dim() == 3 && mean.size(0) == x.size(0) && mean.size(1) == x.size(1) && mean.size(2) == c / kHeadSize, "mean shape mismatch");
    TORCH_CHECK(rstd.dim() == 3 && rstd.size(0) == x.size(0) && rstd.size(1) == x.size(1) && rstd.size(2) == c / kHeadSize, "rstd shape mismatch");
    TORCH_CHECK(scale.dim() == 3 && scale.size(0) == x.size(0) && scale.size(1) == x.size(1) && scale.size(2) == c / kHeadSize, "scale shape mismatch");
    return tmix_lnx_rkvres_backward_cuda(grad_y, x, r, k, v, r_k, weight, mean, rstd, scale);
}

TORCH_LIBRARY(rwkv7_tmix_lnx_rkvres_bf16, m) {
    m.def("forward(Tensor x, Tensor r, Tensor k, Tensor v, Tensor r_k, Tensor weight, Tensor bias) -> Tensor[]");
    m.def("backward(Tensor grad_y, Tensor x, Tensor r, Tensor k, Tensor v, Tensor r_k, Tensor weight, Tensor mean, Tensor rstd, Tensor scale) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(rwkv7_tmix_lnx_rkvres_bf16, CUDA, m) {
    m.impl("forward", &tmix_lnx_rkvres_forward);
    m.impl("backward", &tmix_lnx_rkvres_backward);
}
