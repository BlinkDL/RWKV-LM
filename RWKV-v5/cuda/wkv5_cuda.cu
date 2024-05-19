#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;


// xzl var naming convention
// _k global ptr, shared by all blocks 
// k_ (or __shared__) shared within a cudablk (a head)
// k (thr private, on registers

// xzl: called with grid(B*H,1,1) threadgroup(_N_,1,1)
template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w, const F *__restrict__ _u,
                               F *__restrict__ const _y) 
{
    // xzl: w is actually eew, y is a blank tnesor
    // each cudablock: a head

    const int b = blockIdx.x / H;       // xzl: batch id for this cudablk
    const int h = blockIdx.x % H;       // xzl: head id for this cudablk
    const int i = threadIdx.x;
    _w += h*_N_;        // xzl: points to the head for this cudablock
    _u += h*_N_;        
    // xzl: _N_ headsize (64), passed as compiler flag

    __shared__ float r[_N_], k[_N_], u[_N_], w[_N_];    // a head, block-local var 
    float state[_N_] = {0};   // xzl: state pased across timesteps (thread private)

    // xzl: each cuda thr handles a scalar in a head (64 thr)
    //  copy u,w from global tensor to this head (NB _w _u already shifted above
    __syncthreads();    
    w[i] = _w[i];
    u[i] = float(_u[i]);
    __syncthreads();

    // xzl: the whole loop belo: serial scan across timesteps in one sequence
    //  t: element index in a (B,T,C) tensor
    //      t+=C .. each iteration processes 1 tiemstemp? (eg t+=1024)
    //      this thr goes to the next token in the seq??
    for (int t = b*T*C + h*_N_ + i; t < (b+1)*T*C + h*_N_ + i; t += C)
    {
        __syncthreads();        // xzl: wait for state from previous timestep
        r[i] = float(_r[t]);    // xzl r is threadblock, i  thr id
        k[i] = float(_k[t]);
        __syncthreads();        // xzl: why syncthreads here???

        const float v = float(_v[t]);
        float y = 0;        // xzl: thr local scalar, accumulate (1 out of _N_

        // xzl: exec in seq by each thr
        // _N_: go through each element in *this* head (in r,k,w,u...)
        //   b/c this thr maintains state 1x_N_ (this head's state _N_x_N_
        // (+=4 b/c of float4)
        // essetinaly, decompose s@w, k@v ... into scalar mul...

        #pragma unroll
        for (int j = 0; j < _N_; j+=4)
        {
            const float4& r_ = (float4&)(r[j]);
            const float4& k_ = (float4&)(k[j]);
            const float4& w_ = (float4&)(w[j]);
            const float4& u_ = (float4&)(u[j]);
            float4& s = (float4&)(state[j]);
            float4 x;
            
            // xzl: scale k (vec) with v[t]...
            x.x = k_.x * v;
            x.y = k_.y * v;
            x.z = k_.z * v;
            x.w = k_.w * v;
            
            // xzl: einsum('i,ij->ij')
            //      elementwise mac.. then accu along row ('
            y += r_.x * (u_.x * x.x + s.x);
            y += r_.y * (u_.y * x.y + s.y);
            y += r_.z * (u_.z * x.z + s.z);
            y += r_.w * (u_.w * x.w + s.w);

            // xzl: state update (elementwise ... carry over to next iteration (timestep
            s.x = s.x * w_.x + x.x;
            s.y = s.y * w_.y + x.y;
            s.z = s.z * w_.z + x.z;
            s.w = s.w * w_.w + x.w;
        }
        _y[t] = F(y);  // xzl: output the scalar (current timestep) to global var. F: typename
    }
}

// xzl: gr/gk/gv grad on activations size (B,T,C)       ALL BLANK
// gw,gu grad on weights size (B,C), will be aggregated in python   ALL BLANK
// gy size (B,T,C)
template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C, const int H,
    const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w, const float *__restrict__ __w, const F *__restrict__ _u, const F *__restrict__ const _gy,
    F *__restrict__ const _gr, F *__restrict__ const _gk, F *__restrict__ const _gv, F *__restrict__ const _gw, F *__restrict__ const _gu)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    // xzl: offset the global ptrs
    _w += h*_N_;        // xzl: ew
    _u += h*_N_;
    __w += h*_N_;       // xzl: eew, bad naming

    // xzl: cudablock var. for the whole head 
    __shared__ float w_[_N_], u_[_N_];
    __shared__ float r[_N_], k[_N_], v[_N_], gy[_N_];
    __syncthreads();    // xzl: parallel load blockmemory, but why 1st barrier needed?
    w_[i] = _w[i];
    u_[i] = float(_u[i]);
    __syncthreads();

    const float w = w_[i];
    const float ww = __w[i];
    const float u = u_[i];

    // xzl: multi versions of the staet?? for multiple passes?                                          
    float state[_N_] = {0}, saaaa[_N_] = {0}, sbbbb[_N_] = {0}, scccc[_N_] = {0}, sdddd[_N_] = {0};

    float gw = 0, gu = 0;
    const int t000 = b*T*C + h*_N_ + i;
    const int t111 = (b+1)*T*C + h*_N_ + i;     // xzl: why not (b+1)*T*C???  easy to compute t222?
    const int t222 = t111 - 2*C;        // xzl: ??? two tokens less

    // xzl: pass 1.... compute gu   (over a sequence...
    for (int t = t000; t < t111; t += C)
    {
        __syncthreads();    // xzl: parallel load to blockemory... will be consuenmd by each thr
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t]); // xzl: _gy grad from y (downstream
        __syncthreads();

        const float k = float(_k[t]);
        float gr = 0, gu_ = 0;

        // xzl: cf forward on loop _N_
        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = state[j];
            float x = k * v[j];         // xzl: kv

            gr += (u * x + s) * gy[j];  //  xzl accum gr..
            gu_ += x * gy[j];
            s = s * w + x;
        }
        _gr[t] = F(gr);                 // output gr
        gu += float(_r[t]) * gu_;
    }
    _gu[b*C + h*_N_ + i] = F(gu);       // output gu....(shape B,C) grad on weights
    
    for (int t = t000; t < t222; t += C) // xzl: pass 2 ...gw (but 2 tokens less
    {
        __syncthreads();
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t + 2*C]);    // xzl: shift y                
        __syncthreads();

        // xzl: gw grad flowed from gy (2 timesteps later, why 2??
        //      gues it depends on saaaa and sbbbb... the state contribs??
        const float k = float(_k[t]);
        float gw_ = 0;
        
        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = saaaa[j];
            float& s2 = sbbbb[j];
            float x = k * v[j];
            
            // xzl: to udnerstand btr... how graident flows wrt states (s,s2)
            float tmp = w * (x + s);
            s = tmp;
            s2 = tmp + w * s2;
            gw_ += s2 * gy[j];      // xzl: graident from y 
        }
        gw += float(_r[t + 2*C]) * gw_;
    }    
    _gw[b*C + h*_N_ + i] = F(ww * gw);  // output gw, .. shape B,C weights

    // gk... 1 token less, grad flow from gy, back in time (from seq end
    //  state: scccc
    for (int t = t111 - C; t >= t000; t -= C)   
    {
        __syncthreads();
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t]);
        __syncthreads();

        const float rr = float(_r[t]);
        float gk = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = scccc[j];
            float x = rr * gy[j];
            
            gk += (u * x + s) * v[j];
            s = x + s * w;
        }
        _gk[t] = F(gk);
    }

    // gv... 1 token less, back in time...
    //      state: sdddd
    for (int t = t111 - C; t >= t000; t -= C)   
    {
        __syncthreads();
        r[i] = float(_r[t]);
        k[i] = float(_k[t]);
        __syncthreads();

        const float gyy = float(_gy[t]);
        float gv = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = sdddd[j];
            float x = gyy * r[j];
            
            gv += (u_[j] * x + s) * k[j];
            s = x + s * w_[j];
        }
        _gv[t] = F(gv);
    }
}

void cuda_forward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, bf16 *y)
{
    assert(H*_N_ == C);
    assert(_N_%4 == 0);
    // xzl: #of cudablocks = batch*#heads   each block: # of thr = headdim (64)
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, y);
}

void cuda_backward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, float *ww, bf16 *u, bf16 *gy, bf16 *gr, bf16 *gk, bf16 *gv, bf16 *gw, bf16 *gu)
{
    assert(H*_N_ == C);
    assert(_N_%4 == 0);
    // xzl: same as forward. each cudablock == 1 head for a sequence
    //      each thr --> one scalar element in a head
    kernel_backward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, ww, u, gy, gr, gk, gv, gw, gu);
}
