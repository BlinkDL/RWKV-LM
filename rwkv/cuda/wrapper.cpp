#include <torch/extension.h>
#include "ATen/ATen.h"
#include <iostream>
#include <c10/cuda/CUDAGuard.h>

typedef at::Half fp16;

template <typename F>
void cuda_wkv_forward(int B, int T, int C,
                      float *w, float *u, F *k, F *v, F *y,
                      float *aa, float *bb, float *pp);
template <typename F>
void cuda_mm8_seq(int B, int N, int M,
                  F *x, int x_stride,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  F *y, int y_stride);
template <typename F>
void cuda_mm8_one(int N, int M,
                  F *x,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  float *y);

void wkv_forward(int64_t B, int64_t T, int64_t C,
                 torch::Tensor &w, torch::Tensor &u,
                 torch::Tensor &k, torch::Tensor &v, torch::Tensor &y,
                 torch::Tensor &aa, torch::Tensor &bb, torch::Tensor &pp) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
    switch (k.scalar_type()) {
    case c10::ScalarType::Half:
        cuda_wkv_forward(B, T, C,
                         w.data_ptr<float>(), u.data_ptr<float>(),
                         k.data_ptr<fp16>(), v.data_ptr<fp16>(), y.data_ptr<fp16>(),
                         aa.data_ptr<float>(), bb.data_ptr<float>(), pp.data_ptr<float>());
        break;
    case c10::ScalarType::Float:
        cuda_wkv_forward(B, T, C,
                         w.data_ptr<float>(), u.data_ptr<float>(),
                         k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(),
                         aa.data_ptr<float>(), bb.data_ptr<float>(), pp.data_ptr<float>());
        break;
    default:
        assert(false && "Only FP16 and FP32 are currently supported");
    }
}

void mm8_seq(int64_t B, int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &mx, torch::Tensor &rx,
             torch::Tensor &my, torch::Tensor &ry,
             torch::Tensor &y) {
    assert(x.stride(1) == 1);
    assert(w.stride(1) == 1);
    assert(mx.stride(0) == 1 && rx.stride(0) == 1);
    assert(my.stride(0) == 1 && ry.stride(0) == 1);
    assert(y.stride(1) == 1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
    switch (x.scalar_type()) {
    case c10::ScalarType::Half:
        cuda_mm8_seq(
            B, N, M,
            x.data_ptr<fp16>(), x.stride(0),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<fp16>(), rx.data_ptr<fp16>(),
            my.data_ptr<fp16>(), ry.data_ptr<fp16>(),
            y.data_ptr<fp16>(), y.stride(0));
        break;
    case c10::ScalarType::Float:
        cuda_mm8_seq(
            B, N, M,
            x.data_ptr<float>(), x.stride(0),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<float>(), rx.data_ptr<float>(),
            my.data_ptr<float>(), ry.data_ptr<float>(),
            y.data_ptr<float>(), y.stride(0));
        break;
    default:
        assert(false && "Only FP16 and FP32 are currently supported");
    }
}
void mm8_one(int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &mx, torch::Tensor &rx,
             torch::Tensor &my, torch::Tensor &ry,
             torch::Tensor &y) {
    assert(x.stride(0) == 1);
    assert(w.stride(1) == 1);
    assert(mx.stride(0) == 1 && rx.stride(0) == 1);
    assert(my.stride(0) == 1 && ry.stride(0) == 1);
    assert(y.stride(0) == 1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
    switch (x.scalar_type()) {
    case c10::ScalarType::Half:
        cuda_mm8_one(
            N, M,
            x.data_ptr<fp16>(),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<fp16>(), rx.data_ptr<fp16>(),
            my.data_ptr<fp16>(), ry.data_ptr<fp16>(),
            y.data_ptr<float>());
        break;
    case c10::ScalarType::Float:
        cuda_mm8_one(
            N, M,
            x.data_ptr<float>(),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<float>(), rx.data_ptr<float>(),
            my.data_ptr<float>(), ry.data_ptr<float>(),
            y.data_ptr<float>());
        break;
    default:
        assert(false && "Only FP16 and FP32 are currently supported");
    }
}

using torch::Tensor;

#ifndef DISABLE_CUBLAS_GEMM
void gemm_fp16_cublas(Tensor a, Tensor b, Tensor c);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wkv_forward", &wkv_forward, "wkv forward");
    m.def("mm8_seq", &mm8_seq, "mm8 seq");
    m.def("mm8_one", &mm8_one, "mm8 one");
#ifndef DISABLE_CUBLAS_GEMM
    m.def("gemm_fp16_cublas", &gemm_fp16_cublas, "gemv fp16 cublas");
#endif
}

TORCH_LIBRARY(rwkv, m) {
    m.def("wkv_forward", wkv_forward);
    m.def("mm8_seq", mm8_seq);
    m.def("mm8_one", mm8_one);
#ifndef DISABLE_CUBLAS_GEMM
    m.def("gemm_fp16_cublas", gemm_fp16_cublas);
#endif
}
