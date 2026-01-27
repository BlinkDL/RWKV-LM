#include <assert.h>

#ifdef _FP32_
    using bf = float;
    #define to_float(u) (u)
    #define to_bf(u) (u)
#else
    #include <cuda_bf16.h>
    using bf = __nv_bfloat16;
    #define to_float(u) (__bfloat162float(u))
    #define to_bf(u) (__float2bfloat16_rn(u))
#endif

using i64 = long long int;
typedef bf * __restrict__ F_;
constexpr float W_SCALE = -0.6065306597f; // -exp(-0.5)

//###################################################################################################### 

template<int N> __launch_bounds__(N,2)
__global__ void forward_kernel(int T,int H,F_ r_,F_ w_,F_ k_,F_ v_,F_ a_,F_ b_,bf* __restrict__ y_,float* s__,float* __restrict__ sa_)
{
    const int bb=blockIdx.y, hh=blockIdx.x, i=threadIdx.x;
    float* __restrict__ s_ = s__ + i64(bb*H+hh) * i64((T/_CHUNK_LEN_)*N*N);
    float state[N];
#pragma unroll
    for (int j=0; j<N; ++j) {
        state[j] = 0.0f;
    }
    __shared__ float r[N];
    __shared__ float w[N];
    __shared__ float k[N];
    __shared__ float a[N];
    __shared__ float b[N];

    for (int t = 0; t < T; ++t)
    {
        const int idx = ((bb*T+t)*H+hh)*N+i;

        __syncthreads();
        r[i] = to_float(r_[idx]);
        w[i] = __expf(W_SCALE / (1.0f + __expf(-to_float(w_[idx]))));
        k[i] = to_float(k_[idx]);
        a[i] = to_float(a_[idx]);
        b[i] = to_float(b_[idx]);
        __syncthreads();

        float sa = 0.0f;
#pragma unroll
        for (int j=0; j<N; ++j) {
            sa += state[j] * a[j];
        }
        sa_[idx] = sa;

        float vi = to_float(v_[idx]);
        float y=0.0f;
#pragma unroll
        for (int j=0; j<N; ++j) {
            float s = state[j];
            s = s * w[j] + (sa * b[j] + k[j] * vi);
            y += s * r[j];
            state[j] = s;
        }

        y_[idx] = to_bf(y);

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (t/_CHUNK_LEN_)*N*N + i;
#pragma unroll
            for (int j=0; j<N; ++j) {
                s_[base+j*N] = state[j];
            }
        }
    }
}
void cuda_forward(int B,int T,int H,bf*r,bf*w,bf*k,bf*v,bf*a,bf*b,bf*y,float*s,float*sa)
{
    forward_kernel<_N_><<<dim3(H,B),dim3(_N_)>>>(T,H,r,w,k,v,a,b,y,s,sa);
}

//###################################################################################################### 

template<int N>
__global__ void backward_kernel(int T, int H, F_ r_, F_ w_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_, float * __restrict__ s__, float * __restrict__ sa_, bf* dr_, bf* dw_, bf* dk_, bf* dv_, bf* da_, bf* db_)
{
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float* __restrict__ s_ = s__ + i64(bb*H+hh) * i64((T/_CHUNK_LEN_)*N*N);

    float stateT[N] = {0}, dstate[N] = {0}, dstateT[N] = {0};
    __shared__ float r[N], w[N], k[N], v[N], a[N], b[N], dy[N], sa[N], dSb_shared[N];
    float ri, wi, ki, ai, bi, dyi;

    for (int t = T-1; t >= 0; t--)
    {
        int idx = bb*T*H*N + t*H*N + hh * N + i;

        __syncthreads();
        r[i] = ri = to_float(r_[idx]);
        float w_sig = 1.0f / (1.0f + __expf(-to_float(w_[idx])));
        w[i] = wi = __expf(W_SCALE * w_sig);
        k[i] = ki = to_float(k_[idx]);
        v[i] = to_float(v_[idx]);
        a[i] = ai = to_float(a_[idx]);
        b[i] = bi = to_float(b_[idx]);
        dy[i] = dyi = to_float(dy_[idx]);
        sa[i] = sa_[idx];
        __syncthreads();

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (t/_CHUNK_LEN_)*N*N + i*N;
            const float4* s4 = (const float4*)(s_ + base);
#pragma unroll
            for (int j4 = 0; j4 < N/4; j4++) {
                float4 q = s4[j4];
                const int j = j4<<2;
                stateT[j+0] = q.x;
                stateT[j+1] = q.y;
                stateT[j+2] = q.z;
                stateT[j+3] = q.w;
            }
        }

        float dr = 0;
#pragma unroll
        for (int j = 0; j < N; j++) {
            dr += stateT[j] * dy[j];
        }
        dr_[idx] = to_bf(dr);

        float iwi = 1.0f / wi;
#pragma unroll
        for (int j = 0; j < N; j++) {
            stateT[j] = (stateT[j] - ki * v[j] - bi * sa[j]) * iwi;
            dstate[j] += dyi * r[j];
            dstateT[j] += ri * dy[j];
        }

        float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
#pragma unroll
        for (int j = 0; j < N; j++) {
            dw += dstateT[j] * stateT[j];
            dk += dstateT[j] * v[j];
            dv += dstate[j] * k[j];
            dSb += dstate[j] * b[j];
            db += dstateT[j] * sa[j];
        }
        dw_[idx] = to_bf(W_SCALE * dw * wi * w_sig * (1.0f - w_sig));

        dk_[idx] = to_bf(dk);
        dv_[idx] = to_bf(dv);
        db_[idx] = to_bf(db);

        __syncthreads();
        dSb_shared[i] = dSb;
        __syncthreads();

        float da = 0;
#pragma unroll
        for (int j = 0; j < N; j++) {
            da += stateT[j]*dSb_shared[j];
        }
        da_[idx] = to_bf(da);

#pragma unroll
        for (int j = 0; j < N; j++) {
            dstate[j] = dstate[j] * w[j] + dSb * a[j];
            dstateT[j] = dstateT[j] * wi + ai * dSb_shared[j];
        }
    }
}

void cuda_backward(int B, int T, int H, bf*r, bf*w, bf*k, bf*v, bf*a, bf*b, bf*dy, float*s, float*sa, bf*dr, bf*dw, bf*dk, bf*dv, bf*da, bf*db)
{
    assert(T%_CHUNK_LEN_ == 0);
    backward_kernel<_N_><<<dim3(H,B), dim3(_N_)>>>(T,H,r,w,k,v,a,b,dy,s,sa,dr,dw,dk,dv,da,db);
}
