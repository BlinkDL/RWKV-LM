#include <torch/extension.h>

void cuda_forward(const float *w, const float *k, float *x, float eps, int B, int C, int T);
void cuda_backward(const float *w, const float *k, const float *gwk, float *gw, float *gk, int B, int C, int T);

void forward(torch::Tensor &w, const torch::Tensor &k, torch::Tensor &x, double eps, int64_t B, int64_t C, int64_t T) {
    cuda_forward((const float *)w.data_ptr(), (const float *)k.data_ptr(), (float *)x.data_ptr(), eps, B, C, T);
}
void backward(torch::Tensor &w, const torch::Tensor &k, const torch::Tensor &gwk, torch::Tensor &gw, torch::Tensor &gk, int64_t B, int64_t C, int64_t T) {
    cuda_backward((const float *)w.data_ptr(), (const float *)k.data_ptr(), (const float *)gwk.data_ptr(), (float *)gw.data_ptr(), (float *)gk.data_ptr(), B, C, T);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "timex forward");
    m.def("backward", &backward, "timex backward");
}

TORCH_LIBRARY(timex, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}
