#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w, const F *__restrict__ _u,
                               F *__restrict__ const _y)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _w += h*_N_;
    _u += h*_N_;

    __shared__ float r[_N_], k[_N_];

    float state[_N_] = {0};

    for (int _t = b*T*C + h*_N_ + i; _t < (b+1)*T*C + h*_N_ + i; _t += C)
    {
        __syncthreads();
        r[i] = float(_r[_t]);
        k[i] = float(_k[_t]);
        __syncthreads();

        const float v = float(_v[_t]);
        float y = 0;

        for (int j = 0; j < _N_; j++)
        {
            float x = k[j] * v;

            float s = state[j];
            state[j] = s * _w[j] + x;

            y += r[j] * (float(_u[j]) * x + s);
        }
        _y[_t] = F(y);
    }
}

template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C, const int H,
    const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w, const float *__restrict__ __w, const F *__restrict__ _u, const F *__restrict__ const _gy,
    F *__restrict__ const _gr, F *__restrict__ const _gk, F *__restrict__ const _gv, float *__restrict__ _gw, float *__restrict__ _gu)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _w += h*_N_;
    _u += h*_N_;
    __w += h*_N_;

    __shared__ float v[_N_], r[_N_], k[_N_], gy[_N_];
    
    const float w = _w[i];
    const float u = float(_u[i]);
    const float ww = __w[i];
    float state[_N_] = {0}, saaaa[_N_] = {0}, sbbbb[_N_] = {0};

    for (int _t = b*T*C + h*_N_ + i, _tend = (b+1)*T*C + h*_N_ + i; _t < _tend; _t += C)
    {
        __syncthreads();
        v[i] = float(_v[_t]);
        gy[i] = float(_gy[_t]);
        __syncthreads();

        const float k = float(_k[_t]);
        const float r = float(_r[_t]);
        float gr = 0;
        float gu = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float x = v[j] * k;
            float s = state[j];
            state[j] = s * w + x;

            gr += gy[j] * (u * x + s);
            gu += r * x * gy[j];
        }

        _gr[_t] = F(gr);
        _gu[_t] = F(gu);
        
        float gw = 0;
        if (_t < _tend - 2*C)
        {
            __syncthreads();
            gy[i] = float(_gy[_t + 2*C]);
            __syncthreads();

            const float r = float(_r[_t + 2*C]);

            #pragma unroll
            for (int j = 0; j < _N_; j++)
            {
                float x = v[j] * k;
                saaaa[j] = w * (saaaa[j] + sbbbb[j] + x);
                sbbbb[j] = w * (sbbbb[j] + x);
                
                gw += r * ww * saaaa[j] * gy[j];
            }
        }
        _gw[_t] = gw;
    }

    #pragma unroll
    for (int j = 0; j < _N_; ++j)
        state[j] = 0;
    
    for (int _t = (b+1)*T*C + h*_N_ + i - C, _tend = b*T*C + h*_N_ + i; _t >= _tend; _t -= C)
    {
        __syncthreads();
        v[i] = float(_v[_t]);
        gy[i] = float(_gy[_t]);
        __syncthreads();

        const float r = float(_r[_t]);
        float gk = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float x = gy[j] * r;
            float s = state[j];
            state[j] = s * w + x;

            gk += v[j] * (u * x + s);
        }
        _gk[_t] = F(gk);
    }

    #pragma unroll
    for (int j = 0; j < _N_; ++j)
        state[j] = 0;

    for (int _t = (b+1)*T*C + h*_N_ + i - C, _tend = b*T*C + h*_N_ + i; _t >= _tend; _t -= C)
    {
        __syncthreads();
        k[i] = float(_k[_t]);
        r[i] = float(_r[_t]);
        __syncthreads();

        const float gy = float(_gy[_t]);
        float gv = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float x = gy * r[j];
            float s = state[j];
            state[j] = s * float(_w[j]) + x;

            gv += k[j] * (float(_u[j]) * x + s);
        }
        _gv[_t] = F(gv);
    }
}

void cuda_forward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, bf16 *y)
{
    assert(H*_N_ == C);
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, y);
}

void cuda_backward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, float *ww, bf16 *u, bf16 *gy, bf16 *gr, bf16 *gk, bf16 *gv, float *gw, float *gu)
{
    assert(H*_N_ == C);
    kernel_backward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, ww, u, gy, gr, gk, gv, gw, gu);
}
