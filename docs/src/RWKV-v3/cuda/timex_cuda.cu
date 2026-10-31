#include <stdio.h>

// require T <= Tmax, T % 4 == 0, B % BF == 0, B % BB === 0 (Tmax and BF and BB are passed by compiler)

#define F4(A, B) ((float4 *)(A))[(B) >> 2]

template <typename F>
__global__ void kernel_forward(const F *__restrict__ const __w, const F *__restrict__ const __k, F *__restrict__ const x,
                               const F eps, const int B, const int C, const int T) {
    const int i = blockIdx.y;
    const int ij = (B * C) / BF;
    const int t = threadIdx.x << 2;

    __shared__ F ww[Tmax];
    __shared__ F kk[Tmax * BF];
    F4(ww, t) = F4(__w, t + T * (i % C));
    
    #pragma unroll
    for (int j = 0; j < BF; j++) {
        F4(kk, t + Tmax * j) = F4(__k, t + T * (i + ij * j));
    }
    __syncthreads();

    float4 s[BF];
    #pragma unroll
    for (int j = 0; j < BF; j++) {
        s[j] = {eps, eps, eps, eps};
    }
    const F *__restrict__ const w = ww + T - t - 4;
    for (int u = 0; u <= t; u++) {
        #pragma unroll
        for (int j = 0; j < BF; j++) {
            const F x = kk[u + Tmax * j];
            s[j].x += w[u + 3] * x;
            s[j].y += w[u + 2] * x;
            s[j].z += w[u + 1] * x;
            s[j].w += w[u + 0] * x;
        }
    }
    #pragma unroll
    for (int j = 0; j < BF; j++) {
        const F *__restrict__ const k = kk + Tmax * j;
        s[j].y += w[t + 3] * k[t + 1];
        s[j].z += w[t + 2] * k[t + 1];
        s[j].z += w[t + 3] * k[t + 2];
        s[j].w += w[t + 1] * k[t + 1];
        s[j].w += w[t + 2] * k[t + 2];
        s[j].w += w[t + 3] * k[t + 3];
        F4(x, t + T * (i + ij * j)) = s[j];
    }
}

template <typename F>
__global__ void kernel_backward_W(const F *__restrict__ const __w, const F *__restrict__ const __k, const F *__restrict__ const __gwk,
                                F *__restrict__ const gw, F *__restrict__ const gk,
                                const int B, const int C, const int T) {
    const int i = blockIdx.y;
    const int t = threadIdx.x << 2;

    __shared__ F k[Tmax];
    __shared__ F gg[Tmax];
    F4(k, t) = F4(__k, t + T * i);
    F4(gg, t) = F4(__gwk, t + T * i);
    __syncthreads();

    float4 s = {0, 0, 0, 0};

    const F *__restrict__ const g = gg + T - t - 4;
    for (int u = 0; u <= t; u++) {
        F x = k[u];
        s.x += g[u + 3] * x;
        s.y += g[u + 2] * x;
        s.z += g[u + 1] * x;
        s.w += g[u + 0] * x;
    }
    s.y += g[t + 3] * k[t + 1];
    s.z += g[t + 2] * k[t + 1];
    s.z += g[t + 3] * k[t + 2];
    s.w += g[t + 1] * k[t + 1];
    s.w += g[t + 2] * k[t + 2];
    s.w += g[t + 3] * k[t + 3];
    F4(gw, t + T * i) = s;
}
void cuda_forward(const float *w, const float *k, float *x, float eps, int B, int C, int T) {
    dim3 gridDim(1, B * C / BF);
    dim3 blockDim(T >> 2);
    kernel_forward<<<gridDim, blockDim>>>(w, k, x, eps, B, C, T);
}

template <typename F>
__global__ void kernel_backward(const F *__restrict__ const __w, const F *__restrict__ const __k, const F *__restrict__ const __gwk,
                                F *__restrict__ const gw, F *__restrict__ const gk,
                                const int B, const int C, const int T) {
    const int i = blockIdx.y;
    const int ij = (B * C) / BB;
    const int t = threadIdx.x << 2;

    __shared__ F w[Tmax];
    __shared__ F kk[Tmax * BB];
    __shared__ F gg[Tmax * BB];
    F4(w, t) = F4(__w, t + T * (i % C));

    #pragma unroll
    for (int j = 0; j < BB; j++) {
        F4(kk, t + Tmax * j) = F4(__k, t + T * (i + ij * j));
        F4(gg, t + Tmax * j) = F4(__gwk, t + T * (i + ij * j));
    }
    __syncthreads();

    float4 s[BB];
    #pragma unroll
    for (int j = 0; j < BB; j++) {
        s[j] = {0, 0, 0, 0};
    }

    for (int u = 0; u <= t; u++) {
        #pragma unroll
        for (int j = 0; j < BB; j++) {
            const F *__restrict__ const g = gg + Tmax * j + T - t - 4;
            F x = kk[u + Tmax * j];
            s[j].x += g[u + 3] * x;
            s[j].y += g[u + 2] * x;
            s[j].z += g[u + 1] * x;
            s[j].w += g[u + 0] * x;
        }
    }
    #pragma unroll
    for (int j = 0; j < BB; j++) {
        const F *__restrict__ const k = kk + Tmax * j;
        const F *__restrict__ const g = gg + Tmax * j + T - t - 4;
        s[j].y += g[t + 3] * k[t + 1];
        s[j].z += g[t + 2] * k[t + 1];
        s[j].z += g[t + 3] * k[t + 2];
        s[j].w += g[t + 1] * k[t + 1];
        s[j].w += g[t + 2] * k[t + 2];
        s[j].w += g[t + 3] * k[t + 3];
        F4(gw, t + T * (i + ij * j)) = s[j];
    }

    #pragma unroll
    for (int j = 0; j < BB; j++) {
        s[j] = {0, 0, 0, 0};
    }

    for (int u = t + 3; u < T; u++) {
        F x = w[u];
        #pragma unroll
        for (int j = 0; j < BB; j++) {
            const F *__restrict__ const g = gg + Tmax * j + T + t - 3;
            s[j].x += g[2 - u] * x;
            s[j].y += g[3 - u] * x;
            s[j].z += g[4 - u] * x;
            s[j].w += g[5 - u] * x;
        }        
    }
    #pragma unroll
    for (int j = 0; j < BB; j++) {
        const F *__restrict__ const g = gg + Tmax * j + T + t - 3;
        s[j].x += g[2 - t] * w[t + 0];
        s[j].x += g[1 - t] * w[t + 1];
        s[j].x += g[0 - t] * w[t + 2];
        s[j].y += g[2 - t] * w[t + 1];
        s[j].y += g[1 - t] * w[t + 2];
        s[j].z += g[2 - t] * w[t + 2];
        F4(gk, t + T * (i + ij * j)) = s[j];
    }
}
void cuda_backward(const float *w, const float *k, const float *gwk, float *gw, float *gk, int B, int C, int T) {
    dim3 gridDim(1, B * C / BB);
    dim3 blockDim(T >> 2);
    kernel_backward<<<gridDim, blockDim>>>(w, k, gwk, gw, gk, B, C, T);
}
