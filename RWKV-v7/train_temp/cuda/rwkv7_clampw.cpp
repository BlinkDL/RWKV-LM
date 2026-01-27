#include <torch/extension.h>

#ifdef _FP32_
    using bf = float;
#else
    #include <cuda_bf16.h>
    using bf = __nv_bfloat16;
#endif

void cuda_forward(int B, int T, int H, bf*r, bf*w, bf*k, bf*v, bf*a, bf*b, bf*y, float*s, float*sa);

void forward(torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa) {
    int B = r.sizes()[0], T = r.sizes()[1], H = r.sizes()[2];
    cuda_forward(B, T, H, (bf*)r.data_ptr(), (bf*)w.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)y.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr());
}

void cuda_backward(int B, int T, int H, bf*r, bf*w, bf*k, bf*v, bf*a, bf*b, bf*dy, float*s, float*sa, bf*dr, bf*dw, bf*dk, bf*dv, bf*da, bf*db);

void backward(torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &dy,
        torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dr, torch::Tensor &dw, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &da, torch::Tensor &db) {
    int B = r.sizes()[0], T = r.sizes()[1], H = r.sizes()[2];
    cuda_backward(B, T, H, (bf*)r.data_ptr(), (bf*)w.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)dy.data_ptr(), 
            (float*)s.data_ptr(), (float*)sa.data_ptr(), (bf*)dr.data_ptr(), (bf*)dw.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)da.data_ptr(), (bf*)db.data_ptr());
}

TORCH_LIBRARY(rwkv7_clampw, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}
