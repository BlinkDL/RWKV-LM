#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> l2wrap_ce_forward_v1_cuda(torch::Tensor logits, torch::Tensor targets);
torch::Tensor l2wrap_ce_backward_v1_cuda(
    torch::Tensor grad_loss,
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor lse,
    torch::Tensor max_vals,
    torch::Tensor argmax);

namespace {

void check_cuda_contiguous(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

} // namespace

std::vector<torch::Tensor> forward(torch::Tensor logits, torch::Tensor targets) {
    check_cuda_contiguous(logits, "logits");
    check_cuda_contiguous(targets, "targets");
    TORCH_CHECK(logits.dim() >= 2, "logits must have at least 2 dims");
    TORCH_CHECK(logits.size(-1) == 65536, "l2wrap CE currently expects vocab size 65536");
    TORCH_CHECK(
        logits.scalar_type() == torch::kBFloat16 || logits.scalar_type() == torch::kFloat32,
        "logits must be bf16 or fp32");
    TORCH_CHECK(targets.scalar_type() == torch::kLong, "targets must be int64");
    TORCH_CHECK(targets.numel() == logits.numel() / logits.size(-1), "targets shape mismatch");
    return l2wrap_ce_forward_v1_cuda(logits, targets);
}

torch::Tensor backward(
    torch::Tensor grad_loss,
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor lse,
    torch::Tensor max_vals,
    torch::Tensor argmax) {
    check_cuda_contiguous(grad_loss, "grad_loss");
    check_cuda_contiguous(logits, "logits");
    check_cuda_contiguous(targets, "targets");
    check_cuda_contiguous(lse, "lse");
    check_cuda_contiguous(max_vals, "max_vals");
    check_cuda_contiguous(argmax, "argmax");
    TORCH_CHECK(logits.dim() >= 2, "logits must have at least 2 dims");
    TORCH_CHECK(logits.size(-1) == 65536, "l2wrap CE currently expects vocab size 65536");
    TORCH_CHECK(
        logits.scalar_type() == torch::kBFloat16 || logits.scalar_type() == torch::kFloat32,
        "logits must be bf16 or fp32");
    TORCH_CHECK(grad_loss.scalar_type() == torch::kFloat32, "grad_loss must be fp32");
    TORCH_CHECK(targets.scalar_type() == torch::kLong, "targets must be int64");
    TORCH_CHECK(lse.scalar_type() == torch::kFloat32, "lse must be fp32");
    TORCH_CHECK(max_vals.scalar_type() == torch::kFloat32, "max_vals must be fp32");
    TORCH_CHECK(argmax.scalar_type() == torch::kInt, "argmax must be int32");
    const int64_t rows = logits.numel() / logits.size(-1);
    TORCH_CHECK(grad_loss.numel() == 1, "grad_loss must be scalar");
    TORCH_CHECK(targets.numel() == rows, "targets shape mismatch");
    TORCH_CHECK(lse.numel() == rows, "lse shape mismatch");
    TORCH_CHECK(max_vals.numel() == rows, "max_vals shape mismatch");
    TORCH_CHECK(argmax.numel() == rows, "argmax shape mismatch");
    return l2wrap_ce_backward_v1_cuda(grad_loss, logits, targets, lse, max_vals, argmax);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "RWKV cross entropy + L2Wrap forward metadata (bf16/fp32 CUDA)");
    m.def("backward", &backward, "RWKV cross entropy + L2Wrap fused backward (bf16/fp32 CUDA)");
}
