#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"

typedef at::Half bf16;
// typedef at::BFloat16 bf16;

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
                               F *__restrict__ const _y)
{
    const int e = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    float state[_N_] = {0};
    __shared__ float r[_N_], k[_N_], w[_N_], a[_N_], b[_N_];

    float v[_T_];
    for (int _t = 0; _t < T; _t++)
    {
        const int t = e*T*C + h*_N_ + i + _t * C;
        v[_t] = float(_v[t]);
    }

    for (int _t = 0; _t < T; _t++)
    {
        const int t = e*T*C + h*_N_ + i + _t * C;
        __syncthreads();
        r[i] = float(_r[t]);
        w[i] = __expf(-__expf(float(_w[t])));
        k[i] = float(_k[t]);
        a[i] = float(_a[t]);
        b[i] = float(_b[t]);
        __syncthreads();

        float sa = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            sa += a[j] * state[j];
        }

        float vv = v[_t];
        float y = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = state[j];
            s = s * w[j] + k[j] * vv + sa * b[j];
            y += s * r[j];
        }
        _y[t] = F(y);
    }
}

void cuda_forward(int B, int T, int C, int H, bf16 *r, bf16* w, bf16 *k, bf16 *v, bf16 *a, bf16 *b, bf16 *y)
{
    assert(H*_N_ == C);
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, w, k, v, a, b, y);
}
