#include <torch/extension.h>

// A simple function that adds two tensors
torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

// Exposing the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensors", &add_tensors, "A function that adds two tensors");
}
