#include <metal_stdlib>

using namespace metal;

// xzl: each arg must be either "device" or "constant"

#define _N_ 64          // headsize

// type F ... bf16 (only??

// template <typename F>
typedef bfloat F; 

[[host_name("metal_forward")]]
kernel void kernel_forward(
    constant int &B, 
    constant int &T,
    constant int &C, 
    constant int &H, 
    const device F* const _r, 
    const device F* const _k, 
    const device F* const _v, 
    const device F* _w, 
    const device F* _u, 
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
        // parallel load: from device mem to threadblock
        threadgroup_barrier(mem_flags::mem_threadgroup);
        r[i] = float(_r[t]);    // xzl r is threadblock, i  thr id
        k[i] = float(_k[t]);
        threadgroup_barrier(mem_flags::mem_threadgroup); // or mem_none?

        // thr private (register
        const float v = float(_v[t]);
        float y = 0;

        #pragma unroll(_N_)
        for (int j = 0; j < _N_; j+=4)
        {
            const float4& r_ = (float4&)(r[j]);
            const float4& k_ = (float4&)(k[j]);
            const float4& w_ = (float4&)(w[j]);
            const float4& u_ = (float4&)(u[j]);
            float4& s = (float4&)(state[j]);
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
