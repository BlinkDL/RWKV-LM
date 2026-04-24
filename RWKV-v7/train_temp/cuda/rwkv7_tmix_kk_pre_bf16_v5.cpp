#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> tmix_kk_pre_v5_forward_cuda(
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a,
    torch::Tensor k_a,
    int64_t head_size);

std::vector<torch::Tensor> tmix_kk_pre_v5_backward_cuda(
    torch::Tensor grad_new_k,
    torch::Tensor grad_neg_kk,
    torch::Tensor grad_kka,
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a,
    torch::Tensor k_a,
    torch::Tensor inv_d,
    int64_t head_size);

namespace {

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

void check_common(
    const torch::Tensor& k,
    const torch::Tensor& k_k,
    const torch::Tensor& a,
    const torch::Tensor& k_a,
    int64_t head_size) {
    check_bf16_cuda(k, "k");
    check_bf16_cuda(k_k, "k_k");
    check_bf16_cuda(a, "a");
    check_bf16_cuda(k_a, "k_a");
    TORCH_CHECK(k.dim() == 3, "k must have shape [B, T, C]");
    TORCH_CHECK(a.sizes() == k.sizes(), "a shape mismatch");
    const int64_t c = k.size(2);
    check_vec(k_k, c, "k_k");
    check_vec(k_a, c, "k_a");
    TORCH_CHECK(head_size == 64, "tmix_kk_pre_bf16_v5 currently requires head_size=64");
    TORCH_CHECK(c % head_size == 0, "C must be divisible by head_size");
}

} // namespace

std::vector<torch::Tensor> tmix_kk_pre_v5_forward(
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a,
    torch::Tensor k_a,
    int64_t head_size) {
    check_common(k, k_k, a, k_a, head_size);
    return tmix_kk_pre_v5_forward_cuda(k, k_k, a, k_a, head_size);
}

std::vector<torch::Tensor> tmix_kk_pre_v5_backward(
    torch::Tensor grad_new_k,
    torch::Tensor grad_neg_kk,
    torch::Tensor grad_kka,
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a,
    torch::Tensor k_a,
    torch::Tensor inv_d,
    int64_t head_size) {
    check_bf16_cuda(grad_new_k, "grad_new_k");
    check_bf16_cuda(grad_neg_kk, "grad_neg_kk");
    check_bf16_cuda(grad_kka, "grad_kka");
    check_common(k, k_k, a, k_a, head_size);
    check_f32_cuda(inv_d, "inv_d");
    TORCH_CHECK(grad_new_k.sizes() == k.sizes(), "grad_new_k shape mismatch");
    TORCH_CHECK(grad_neg_kk.sizes() == k.sizes(), "grad_neg_kk shape mismatch");
    TORCH_CHECK(grad_kka.sizes() == k.sizes(), "grad_kka shape mismatch");
    TORCH_CHECK(inv_d.dim() == 3, "inv_d must have shape [B, T, H]");
    TORCH_CHECK(
        inv_d.size(0) == k.size(0) && inv_d.size(1) == k.size(1) && inv_d.size(2) == k.size(2) / head_size,
        "inv_d shape mismatch");
    return tmix_kk_pre_v5_backward_cuda(
        grad_new_k, grad_neg_kk, grad_kka, k, k_k, a, k_a, inv_d, head_size);
}

TORCH_LIBRARY(rwkv7_tmix_kk_pre_bf16_v5, m) {
    m.def("forward(Tensor k, Tensor k_k, Tensor a, Tensor k_a, int head_size) -> Tensor[]");
    m.def("backward(Tensor grad_new_k, Tensor grad_neg_kk, Tensor grad_kka, Tensor k, Tensor k_k, Tensor a, Tensor k_a, Tensor inv_d, int head_size) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(rwkv7_tmix_kk_pre_bf16_v5, CUDA, m) {
    m.impl("forward", &tmix_kk_pre_v5_forward);
    m.impl("backward", &tmix_kk_pre_v5_backward);
}
