#include <assert.h>
#include <cuda_runtime.h>

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

// Fast kernel for H100 etc. (benchmark fwd & bwd speed if you are using consumer GPUs)

//######################################################################################################

template<int N> __launch_bounds__(N,2)
__global__ void forward_kernel_preload(int T,int H,F_ r_,F_ w_,F_ k_,F_ v_,F_ a_,F_ b_,bf* __restrict__ y_,float* s__,float* __restrict__ sa_)
{
    const int bb=blockIdx.y, hh=blockIdx.x, i=threadIdx.x;
    float* __restrict__ s_ = s__ + i64(bb*H+hh) * i64((T/_CHUNK_LEN_)*N*N);
    float state[N];
#pragma unroll
    for (int j=0; j<N; ++j) {
        state[j] = 0.0f;
    }
    __shared__ float r[_CHUNK_LEN_][N];
    __shared__ float w[_CHUNK_LEN_][N];
    __shared__ float k[_CHUNK_LEN_][N];
    __shared__ float a[_CHUNK_LEN_][N];
    __shared__ float b[_CHUNK_LEN_][N];

    for (int t0 = 0; t0 < T; t0 += _CHUNK_LEN_)
    {
        __syncthreads();
#pragma unroll
        for (int tt=0; tt<_CHUNK_LEN_; ++tt) {
            const int idx = ((bb*T+t0+tt)*H+hh)*N+i;
            r[tt][i] = to_float(r_[idx]);
            w[tt][i] = __expf(W_SCALE / (1.0f + __expf(-to_float(w_[idx]))));
            k[tt][i] = to_float(k_[idx]);
            a[tt][i] = to_float(a_[idx]);
            b[tt][i] = to_float(b_[idx]);
        }
        __syncthreads();

        for (int tt=0; tt<_CHUNK_LEN_; ++tt) {
            const int idx = ((bb*T+t0+tt)*H+hh)*N+i;

            float sa = 0.0f;
#pragma unroll
            for (int j=0; j<N; ++j) {
                sa += state[j] * a[tt][j];
            }
            sa_[idx] = sa;

            float vi = to_float(v_[idx]);
            float y=0.0f;
#pragma unroll
            for (int j=0; j<N; ++j) {
                float s = state[j];
                s = s * w[tt][j] + (sa * b[tt][j] + k[tt][j] * vi);
                y += s * r[tt][j];
                state[j] = s;
            }

            y_[idx] = to_bf(y);
        }

        {
            int base = (t0/_CHUNK_LEN_)*N*N + i;
#pragma unroll
            for (int j=0; j<N; ++j) {
                s_[base+j*N] = state[j];
            }
        }
        // Safe without a tail barrier: the next chunk starts with __syncthreads()
        // before any thread overwrites the shared preload buffers.
    }
}
void cuda_forward_v3(int B,int T,int H,bf*r,bf*w,bf*k,bf*v,bf*a,bf*b,bf*y,float*s,float*sa)
{
    forward_kernel_preload<_N_><<<dim3(H,B),dim3(_N_)>>>(T,H,r,w,k,v,a,b,y,s,sa);
}

//######################################################################################################

template<int N, int TILE>
__global__ void backward_kernel_preload(int T, int H, F_ r_, F_ w_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_, float * __restrict__ s__, float * __restrict__ sa_, bf* dr_, bf* dw_, bf* dk_, bf* dv_, bf* da_, bf* db_)
{
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float* __restrict__ s_ = s__ + i64(bb*H+hh) * i64((T/_CHUNK_LEN_)*N*N);

    float stateT[N] = {0}, dstate[N] = {0}, dstateT[N] = {0};
    static_assert(_CHUNK_LEN_%TILE == 0, "TILE must divide _CHUNK_LEN_");
    __shared__ float r[TILE][N];
    __shared__ float w[TILE][N];
    __shared__ float ws[TILE][N];
    __shared__ float k[TILE][N];
    __shared__ float v[TILE][N];
    __shared__ float a[TILE][N];
    __shared__ float b[TILE][N];
    __shared__ float dy[TILE][N];
    __shared__ float sa[TILE][N];
    __shared__ float dSb_shared[N];
    float ri, wi, ki, ai, bi, dyi;

    for (int t0 = T-_CHUNK_LEN_; t0 >= 0; t0 -= _CHUNK_LEN_)
    {
        {
            int base = (t0/_CHUNK_LEN_)*N*N + i*N;
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

        for (int subt=_CHUNK_LEN_-TILE; subt>=0; subt-=TILE) {
            __syncthreads();
#pragma unroll
            for (int tt=0; tt<TILE; ++tt) {
                int idx = bb*T*H*N + (t0+subt+tt)*H*N + hh * N + i;
                r[tt][i] = to_float(r_[idx]);
                float w_sig = 1.0f / (1.0f + __expf(-to_float(w_[idx])));
                float wi = __expf(W_SCALE * w_sig);
                if constexpr (TILE == 8) {
                    ws[tt][i] = W_SCALE * wi * w_sig * (1.0f - w_sig);
                } else {
                    ws[tt][i] = w_sig;
                }
                w[tt][i] = wi;
                k[tt][i] = to_float(k_[idx]);
                v[tt][i] = to_float(v_[idx]);
                a[tt][i] = to_float(a_[idx]);
                b[tt][i] = to_float(b_[idx]);
                dy[tt][i] = to_float(dy_[idx]);
                sa[tt][i] = sa_[idx];
            }
            __syncthreads();

            for (int tt=TILE-1; tt>=0; --tt) {
                int idx = bb*T*H*N + (t0+subt+tt)*H*N + hh * N + i;
                ri = r[tt][i];
                wi = w[tt][i];
                ki = k[tt][i];
                ai = a[tt][i];
                bi = b[tt][i];
                dyi = dy[tt][i];

                float dr = 0;
#pragma unroll
                for (int j = 0; j < N; j++) {
                    dr += stateT[j] * dy[tt][j];
                }
                dr_[idx] = to_bf(dr);

                float iwi = 1.0f / wi;
#pragma unroll
                for (int j = 0; j < N; j++) {
                    stateT[j] = (stateT[j] - ki * v[tt][j] - bi * sa[tt][j]) * iwi;
                    dstate[j] += dyi * r[tt][j];
                    dstateT[j] += ri * dy[tt][j];
                }

                float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
#pragma unroll
                for (int j = 0; j < N; j++) {
                    dw += dstateT[j] * stateT[j];
                    dk += dstateT[j] * v[tt][j];
                    dv += dstate[j] * k[tt][j];
                    dSb += dstate[j] * b[tt][j];
                    db += dstateT[j] * sa[tt][j];
                }
                if constexpr (TILE == 8) {
                    dw_[idx] = to_bf(dw * ws[tt][i]);
                } else {
                    float w_sig = ws[tt][i];
                    dw_[idx] = to_bf(W_SCALE * dw * wi * w_sig * (1.0f - w_sig));
                }

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
                    dstate[j] = dstate[j] * w[tt][j] + dSb * a[tt][j];
                    dstateT[j] = dstateT[j] * wi + ai * dSb_shared[j];
                }
            }
        }
    }
}

void cuda_backward_v3(int B, int T, int H, bf*r, bf*w, bf*k, bf*v, bf*a, bf*b, bf*dy, float*s, float*sa, bf*dr, bf*dw, bf*dk, bf*dv, bf*da, bf*db)
{
    assert(T%_CHUNK_LEN_ == 0);
    backward_kernel_preload<_N_,16><<<dim3(H,B), dim3(_N_)>>>(T,H,r,w,k,v,a,b,dy,s,sa,dr,dw,dk,dv,da,db);
}
