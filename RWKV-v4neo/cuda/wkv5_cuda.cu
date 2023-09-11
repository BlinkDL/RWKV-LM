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

    __shared__ float state[_N_ * _N_], rr[_N_], kk[_N_];

    for (int j = 0; j < _N_; ++j)
        state[j * _N_ + i] = 0;

    for (int _t = b*T*C + h*_N_ + i; _t < (b+1)*T*C + h*_N_ + i; _t += C)
    {
        __syncthreads();
        rr[i] = float(_r[_t]);
        kk[i] = float(_k[_t]);
        __syncthreads();

        const float vv = float(_v[_t]);
        float yy = 0;

        for (int j = 0; j < _N_; j++)
        {
            float x = kk[j] * vv;

            float s = state[j * _N_ + i];
            state[j * _N_ + i] = s * _w[j] + x;

            yy += rr[j] * (float(_u[j]) * x + s);
        }
        _y[_t] = F(yy);
    }
}

template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C, const int H,
    const F *__restrict__ const r, const F *__restrict__ const k, const F *__restrict__ const v, const float *__restrict__ w, const float *__restrict__ wwww, const F *__restrict__ u, const F *__restrict__ const gy,
    F *__restrict__ const gr, F *__restrict__ const gk, F *__restrict__ const gv, float *__restrict__ gw, float *__restrict__ gu)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    w += h*_N_;
    u += h*_N_;
    wwww += h*_N_;

    __shared__ float state[_N_ * _N_], vv[_N_], rr[_N_], kk[_N_], gyy[_N_];

    #pragma unroll
    for (int j = 0; j < _N_; ++j)
        state[j * _N_ + i] = 0;
    
    const float ww = w[i];
    const float uu = float(u[i]);
    const float wwwww = wwww[i];
    float saaaa[_N_] = {0.0f}, sbbbb[_N_] = {0.0f};

    for (int _t = b*T*C + h*_N_ + i, _tend = (b+1)*T*C + h*_N_ + i; _t < _tend; _t += C)
    {
        __syncthreads();
        vv[i] = float(v[_t]);
        gyy[i] = float(gy[_t]);
        __syncthreads();

        const float kk = float(k[_t]);
        const float rr = float(r[_t]);
        float grr = 0;
        float guu = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float x = vv[j] * kk;
            float s = state[j * _N_ + i];
            state[j * _N_ + i] = s * ww + x;

            grr += gyy[j] * (uu * x + s);
            guu += rr * x * gyy[j];
        }

        gr[_t] = F(grr);
        gu[_t] = F(guu);

        float gww = 0;
        if (_t < _tend - 2*C)
        {
            __syncthreads();
            gyy[i] = float(gy[_t + 2*C]);
            __syncthreads();

            const float rr = float(r[_t + 2*C]);

            #pragma unroll
            for (int j = 0; j < _N_; j++)
            {
                float x = vv[j] * kk;
                saaaa[j] = ww * (saaaa[j] + sbbbb[j] + x);
                sbbbb[j] = ww * (sbbbb[j] + x);
                
                gww += rr * wwwww * saaaa[j] * gyy[j];
            }
        }
        gw[_t] = gww;
    }

    #pragma unroll
    for (int j = 0; j < _N_; ++j)
        state[j * _N_ + i] = 0;
    
    for (int _t = (b+1)*T*C + h*_N_ + i - C, _tend = b*T*C + h*_N_ + i; _t >= _tend; _t -= C)
    {
        __syncthreads();
        vv[i] = float(v[_t]);
        gyy[i] = float(gy[_t]);
        __syncthreads();

        const float rr = float(r[_t]);
        float gkk = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float x = gyy[j] * rr;
            float s = state[j * _N_ + i];
            state[j * _N_ + i] = s * ww + x;

            gkk += vv[j] * (uu * x + s);
        }
        gk[_t] = F(gkk);
    }

    #pragma unroll
    for (int j = 0; j < _N_; ++j)
        state[j * _N_ + i] = 0;

    for (int _t = (b+1)*T*C + h*_N_ + i - C, _tend = b*T*C + h*_N_ + i; _t >= _tend; _t -= C)
    {
        __syncthreads();
        kk[i] = float(k[_t]);
        rr[i] = float(r[_t]);
        __syncthreads();

        const float gyy = float(gy[_t]);
        float gvv = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float x = gyy * rr[j];
            float s = state[j * _N_ + i];
            state[j * _N_ + i] = s * w[j] + x;

            gvv += kk[j] * (u[j] * x + s);
        }
        gv[_t] = F(gvv);
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
