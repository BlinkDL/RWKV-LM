#include <metal_stdlib>

using namespace metal;

// xzl: each arg must be either "device" or "constant"

#define _N_ 64          // headsize

// template <typename F>        // TBD type F ... bf16 (only??
typedef bfloat F; 

// note the type of _w (float not bfloat), b/c it's actually eew passed in 
[[host_name("metal_forward")]]
kernel void kernel_forward(
    constant int &B, 
    constant int &T,
    constant int &C, 
    constant int &H, 
    device const F* const _r, 
    device const F* const _k, 
    device const F* const _v, 
    device const float* _w,
    device const F* _u, 
    device F* const _y,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint  tiitg[[thread_index_in_threadgroup]]
) {
    // thread local
    const int b = tgpig.x / H; 
    const int h = tgpig.x % H;
    const int i = tiitg; 
    _w += h*_N_;
    _u += h*_N_;   

    threadgroup float r[_N_], k[_N_], u[_N_], w[_N_];
    float state[_N_] = {0};   // xzl: state pased across timesteps (thread private)

    threadgroup_barrier(mem_flags::mem_threadgroup);
    w[i] = _w[i];
    u[i] = float(_u[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);    // or mem_none?

    for (int t = b*T*C + h*_N_ + i; t < (b+1)*T*C + h*_N_ + i; t += C) {
        // parallel load: from device mem to threadgrounp
        threadgroup_barrier(mem_flags::mem_threadgroup);
        r[i] = float(_r[t]);    // xzl r is threadgrounp, i  thr id
        k[i] = float(_k[t]);
        threadgroup_barrier(mem_flags::mem_threadgroup); // or mem_none?

        // thr private (register
        const float v = float(_v[t]);
        float y = 0;

        #pragma unroll(_N_)
        for (int j = 0; j < _N_; j+=4)
        {
            threadgroup const float4& r_ = (threadgroup float4&)(r[j]);
            threadgroup const float4& k_ = (threadgroup float4&)(k[j]);
            threadgroup const float4& w_ = (threadgroup float4&)(w[j]);
            threadgroup const float4& u_ = (threadgroup float4&)(u[j]);
            thread float4& s =        (thread float4&)(state[j]);
            float4 x;

            x.x = k_.x * v;
            x.y = k_.y * v;
            x.z = k_.z * v;
            x.w = k_.w * v;

            y += r_.x * (u_.x * x.x + s.x);
            y += r_.y * (u_.y * x.y + s.y);
            y += r_.z * (u_.z * x.z + s.z);
            y += r_.w * (u_.w * x.w + s.w);

            // xzl: state update ... carry over to next iteration (timestep
            s.x = s.x * w_.x + x.x;
            s.y = s.y * w_.y + x.y;
            s.z = s.z * w_.z + x.z;
            s.w = s.w * w_.w + x.w;
        }
        _y[t] = F(y);  // xzl: output the scalar (current timestep) to global var. F: typename        
        // _y[t] = static_cast<F>(y);  // seems ok
    }
}

// template 
// [[host_name("metal_forward")]]
// kernel void kernel_forward(
//     constant int &B, 
//     constant int &T,
//     constant int &C, 
//     constant int &H, 
//     const device F* const _r, 
//     const device F* const _k, 
//     const device F* const _v, 
//     const device F* _w, 
//     const device F* _u, 
//     device F* const _y,
//     uint3 tgpig[[threadgroup_position_in_grid]],
//     uint  tiitg[[thread_index_in_threadgroup]]
//     )



[[host_name("metal_backward")]]
kernel void kernel_backward(
    constant int &B, 
    constant int &T,
    constant int &C, 
    constant int &H,
    device const F* const _r, 
    device const F* const _k, 
    device const F* const _v,
    device const float* _w,
    device const float* __w,
    device const F* _u,
    device const F* _gy,
    device F* const _gr,
    device F* const _gk,
    device F* const _gv,
    device F* const _gw,
    device F* const _gu,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint  tiitg[[thread_index_in_threadgroup]]
) {
    const int b = tgpig.x / H;
    const int h = tgpig.x % H;
    const int i = tiitg;
    _w += h*_N_;        // xzl: ew
    _u += h*_N_;
    __w += h*_N_;       // xzl: eew, bad naming

    // xzl: note the type ... all float
    threadgroup float w_[_N_], u_[_N_];
    threadgroup float r[_N_], k[_N_], v[_N_], gy[_N_];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    w_[i] = _w[i];
    u_[i] = float(_u[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);    // or mem_none?

    // thread private.. .loaded from threadgroup
    const float w = w_[i];
    const float ww = __w[i];
    const float u = u_[i];

    // thread private (multi versions of the staet, for different passes computing gardients...
    float state[_N_] = {0}, saaaa[_N_] = {0}, sbbbb[_N_] = {0}, scccc[_N_] = {0}, sdddd[_N_] = {0};

    float gw = 0, gu = 0;
    const int t000 = b*T*C + h*_N_ + i;
    const int t111 = (b+1)*T*C + h*_N_ + i;     // xzl: why not (b+1)*T*C???  easy to compute t222?
    const int t222 = t111 - 2*C;        // xzl: ??? two tokens less

    // xzl: pass 1.... compute gu   (over a sequence...
    for (int t = t000; t < t111; t += C) {
        // parallel load to blockemory... will be consuenmd by each thr
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t]); // xzl: _gy grad from y (downstream
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float k = float(_k[t]);
        float gr = 0, gu_ = 0;

        // xzl: cf forward on loop _N_
        #pragma unroll(_N_)
        for (int j = 0; j < _N_; j++) {
            thread float& s = state[j];
            float x = k * v[j];         // xzl: kv

            gr += (u * x + s) * gy[j];  //  xzl accum gr..
            gu_ += x * gy[j];
            s = s * w + x;
        }
        _gr[t] = F(gr);                 // output gr
        gu += float(_r[t]) * gu_;
    }
    _gu[b*C + h*_N_ + i] = F(gu);       // output gu....(shape B,C) grad on weights
    
    // xzl: pass 2 ...compute gw (but 2 tokens less
    // state: saaaa sbbbb
    for (int t = t000; t < t222; t += C) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t + 2*C]);    // xzl: shift y                
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // xzl: gw grad flowed from gy (2 timesteps later, why 2??
        const float k = float(_k[t]);
        float gw_ = 0;
        
        #pragma unroll(_N_)
        for (int j = 0; j < _N_; j++)
        {
            thread float& s = saaaa[j];
            thread float& s2 = sbbbb[j];
            float x = k * v[j];
            
            // xzl: to udnerstand btr... how graident flows wrt states (s,s2)
            float tmp = w * (x + s);
            s = tmp;
            s2 = tmp + w * s2;
            gw_ += s2 * gy[j];      // xzl: graident from y 
        }
        gw += float(_r[t + 2*C]) * gw_;
    }    
    _gw[b*C + h*_N_ + i] = F(ww * gw);  // output gw (grad on weights) shape: (B,C)

    // gk... 1 token less, grad flow from gy, scan back in time
    //  state: scccc
    for (int t = t111 - C; t >= t000; t -= C) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float rr = float(_r[t]);
        float gk = 0;

        #pragma unroll(_N_)
        for (int j = 0; j < _N_; j++)
        {
            thread float& s = scccc[j];
            float x = rr * gy[j];
            
            gk += (u * x + s) * v[j];
            s = x + s * w;
        }
        _gk[t] = F(gk);
    }

    // gv... 1 token less, back in time...
    //      state: sdddd
    for (int t = t111 - C; t >= t000; t -= C) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        r[i] = float(_r[t]);
        k[i] = float(_k[t]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float gyy = float(_gy[t]);
        float gv = 0;

        #pragma unroll(_N_)
        for (int j = 0; j < _N_; j++)
        {
            thread float& s = sdddd[j];
            float x = gyy * r[j];
            
            gv += (u_[j] * x + s) * k[j];
            s = x + s * w_[j];
        }
        _gv[t] = F(gv);
    }
}
