#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

__device__ inline __nv_bfloat162 load_bf16x2(const at::BFloat16* ptr) {
    return *reinterpret_cast<const __nv_bfloat162*>(ptr);
}

__device__ inline void store_bf16x2(at::BFloat16* ptr, __nv_bfloat162 value) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) = value;
}

__device__ inline void atomic_add_float2(float* ptr, float x0, float x1) {
    atomicAdd(reinterpret_cast<float2*>(ptr), make_float2(x0, x1));
}

inline int64_t ceil_div(int64_t n, int64_t d) {
    return (n + d - 1) / d;
}

__global__ void tmix_mix6_forward_kernel(
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ x_r,
    const at::BFloat16* __restrict__ x_w,
    const at::BFloat16* __restrict__ x_k,
    const at::BFloat16* __restrict__ x_v,
    const at::BFloat16* __restrict__ x_a,
    const at::BFloat16* __restrict__ x_g,
    at::BFloat16* __restrict__ out_r,
    at::BFloat16* __restrict__ out_w,
    at::BFloat16* __restrict__ out_k,
    at::BFloat16* __restrict__ out_v,
    at::BFloat16* __restrict__ out_a,
    at::BFloat16* __restrict__ out_g,
    int64_t bt_size,
    int64_t t_size,
    int64_t c_size) {
    int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_pairs = bt_size * (c_size / 2);
    if (pair_idx >= total_pairs) {
        return;
    }

    int64_t bt = pair_idx / (c_size / 2);
    int64_t c_pair = pair_idx % (c_size / 2);
    int64_t c = c_pair * 2;
    int64_t idx = bt * c_size + c;
    int64_t t = bt % t_size;

    __nv_bfloat162 x_now = load_bf16x2(x + idx);
    __nv_bfloat162 x_prev = __floats2bfloat162_rn(0.0f, 0.0f);
    if (t > 0) {
        x_prev = load_bf16x2(x + idx - c_size);
    }
    __nv_bfloat162 xx = __hsub2(x_prev, x_now);

    store_bf16x2(out_r + idx, __hadd2(x_now, __hmul2(xx, load_bf16x2(x_r + c))));
    store_bf16x2(out_w + idx, __hadd2(x_now, __hmul2(xx, load_bf16x2(x_w + c))));
    store_bf16x2(out_k + idx, __hadd2(x_now, __hmul2(xx, load_bf16x2(x_k + c))));
    store_bf16x2(out_v + idx, __hadd2(x_now, __hmul2(xx, load_bf16x2(x_v + c))));
    store_bf16x2(out_a + idx, __hadd2(x_now, __hmul2(xx, load_bf16x2(x_a + c))));
    store_bf16x2(out_g + idx, __hadd2(x_now, __hmul2(xx, load_bf16x2(x_g + c))));
}

constexpr int TMIX_PARAM_THREADS = 256;
constexpr int TMIX_PARAM_BT_TILE = 8;

__global__ void tmix_mix6_backward_fused_kernel_v5(
    const at::BFloat16* __restrict__ grad_r,
    const at::BFloat16* __restrict__ grad_w,
    const at::BFloat16* __restrict__ grad_k,
    const at::BFloat16* __restrict__ grad_v,
    const at::BFloat16* __restrict__ grad_a,
    const at::BFloat16* __restrict__ grad_g,
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ x_r,
    const at::BFloat16* __restrict__ x_w,
    const at::BFloat16* __restrict__ x_k,
    const at::BFloat16* __restrict__ x_v,
    const at::BFloat16* __restrict__ x_a,
    const at::BFloat16* __restrict__ x_g,
    at::BFloat16* __restrict__ grad_x,
    float* __restrict__ grad_x_r,
    float* __restrict__ grad_x_w,
    float* __restrict__ grad_x_k,
    float* __restrict__ grad_x_v,
    float* __restrict__ grad_x_a,
    float* __restrict__ grad_x_g,
    int64_t bt_size,
    int64_t t_size,
    int64_t c_size) {
    int64_t c_pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_pairs = c_size / 2;
    if (c_pair >= total_pairs) {
        return;
    }

    int64_t c = c_pair * 2;
    int64_t bt_start = static_cast<int64_t>(blockIdx.y) * TMIX_PARAM_BT_TILE;
    int64_t bt_end = min(bt_start + static_cast<int64_t>(TMIX_PARAM_BT_TILE), bt_size);

    __nv_bfloat162 one = __floats2bfloat162_rn(1.0f, 1.0f);
    __nv_bfloat162 pr = load_bf16x2(x_r + c);
    __nv_bfloat162 pw = load_bf16x2(x_w + c);
    __nv_bfloat162 pk = load_bf16x2(x_k + c);
    __nv_bfloat162 pv = load_bf16x2(x_v + c);
    __nv_bfloat162 pa = load_bf16x2(x_a + c);
    __nv_bfloat162 pg = load_bf16x2(x_g + c);
    __nv_bfloat162 one_minus_pr = __hsub2(one, pr);
    __nv_bfloat162 one_minus_pw = __hsub2(one, pw);
    __nv_bfloat162 one_minus_pk = __hsub2(one, pk);
    __nv_bfloat162 one_minus_pv = __hsub2(one, pv);
    __nv_bfloat162 one_minus_pa = __hsub2(one, pa);
    __nv_bfloat162 one_minus_pg = __hsub2(one, pg);

    float ar0 = 0.0f, ar1 = 0.0f, aw0 = 0.0f, aw1 = 0.0f, ak0 = 0.0f, ak1 = 0.0f;
    float av0 = 0.0f, av1 = 0.0f, aa0 = 0.0f, aa1 = 0.0f, ag0 = 0.0f, ag1 = 0.0f;

    __nv_bfloat162 gr = __floats2bfloat162_rn(0.0f, 0.0f);
    __nv_bfloat162 gw = __floats2bfloat162_rn(0.0f, 0.0f);
    __nv_bfloat162 gk = __floats2bfloat162_rn(0.0f, 0.0f);
    __nv_bfloat162 gv = __floats2bfloat162_rn(0.0f, 0.0f);
    __nv_bfloat162 ga = __floats2bfloat162_rn(0.0f, 0.0f);
    __nv_bfloat162 gg = __floats2bfloat162_rn(0.0f, 0.0f);
    bool have_current_grad = false;
    float2 prev_x = make_float2(0.0f, 0.0f);
    bool have_prev_x = false;

    for (int64_t bt = bt_start; bt < bt_end; ++bt) {
        int64_t idx = bt * c_size + c;
        int64_t t = bt % t_size;
        if (!have_current_grad) {
            gr = load_bf16x2(grad_r + idx);
            gw = load_bf16x2(grad_w + idx);
            gk = load_bf16x2(grad_k + idx);
            gv = load_bf16x2(grad_v + idx);
            ga = load_bf16x2(grad_a + idx);
            gg = load_bf16x2(grad_g + idx);
        }

        __nv_bfloat162 next_gr = __floats2bfloat162_rn(0.0f, 0.0f);
        __nv_bfloat162 next_gw = __floats2bfloat162_rn(0.0f, 0.0f);
        __nv_bfloat162 next_gk = __floats2bfloat162_rn(0.0f, 0.0f);
        __nv_bfloat162 next_gv = __floats2bfloat162_rn(0.0f, 0.0f);
        __nv_bfloat162 next_ga = __floats2bfloat162_rn(0.0f, 0.0f);
        __nv_bfloat162 next_gg = __floats2bfloat162_rn(0.0f, 0.0f);
        bool have_next_grad = false;
        if (t + 1 < t_size) {
            int64_t next_idx = idx + c_size;
            next_gr = load_bf16x2(grad_r + next_idx);
            next_gw = load_bf16x2(grad_w + next_idx);
            next_gk = load_bf16x2(grad_k + next_idx);
            next_gv = load_bf16x2(grad_v + next_idx);
            next_ga = load_bf16x2(grad_a + next_idx);
            next_gg = load_bf16x2(grad_g + next_idx);
            have_next_grad = true;
        }

        __nv_bfloat162 dx_grad = __floats2bfloat162_rn(0.0f, 0.0f);
        dx_grad = __hadd2(dx_grad, __hmul2(gr, one_minus_pr));
        dx_grad = __hadd2(dx_grad, __hmul2(gw, one_minus_pw));
        dx_grad = __hadd2(dx_grad, __hmul2(gk, one_minus_pk));
        dx_grad = __hadd2(dx_grad, __hmul2(gv, one_minus_pv));
        dx_grad = __hadd2(dx_grad, __hmul2(ga, one_minus_pa));
        dx_grad = __hadd2(dx_grad, __hmul2(gg, one_minus_pg));
        if (have_next_grad) {
            dx_grad = __hadd2(dx_grad, __hmul2(next_gr, pr));
            dx_grad = __hadd2(dx_grad, __hmul2(next_gw, pw));
            dx_grad = __hadd2(dx_grad, __hmul2(next_gk, pk));
            dx_grad = __hadd2(dx_grad, __hmul2(next_gv, pv));
            dx_grad = __hadd2(dx_grad, __hmul2(next_ga, pa));
            dx_grad = __hadd2(dx_grad, __hmul2(next_gg, pg));
        }
        store_bf16x2(grad_x + idx, dx_grad);

        float2 x_now = __bfloat1622float2(load_bf16x2(x + idx));
        float2 x_prev = make_float2(0.0f, 0.0f);
        if (t > 0) {
            x_prev = (have_prev_x ? prev_x : __bfloat1622float2(load_bf16x2(x + idx - c_size)));
        }
        float dx0 = x_prev.x - x_now.x;
        float dx1 = x_prev.y - x_now.y;
        float2 fr = __bfloat1622float2(gr);
        float2 fw = __bfloat1622float2(gw);
        float2 fk = __bfloat1622float2(gk);
        float2 fv = __bfloat1622float2(gv);
        float2 fa = __bfloat1622float2(ga);
        float2 fg = __bfloat1622float2(gg);

        ar0 += fr.x * dx0; ar1 += fr.y * dx1;
        aw0 += fw.x * dx0; aw1 += fw.y * dx1;
        ak0 += fk.x * dx0; ak1 += fk.y * dx1;
        av0 += fv.x * dx0; av1 += fv.y * dx1;
        aa0 += fa.x * dx0; aa1 += fa.y * dx1;
        ag0 += fg.x * dx0; ag1 += fg.y * dx1;

        prev_x = x_now;
        have_prev_x = (t + 1 < t_size);
        gr = next_gr; gw = next_gw; gk = next_gk; gv = next_gv; ga = next_ga; gg = next_gg;
        have_current_grad = have_next_grad;
    }

    atomic_add_float2(grad_x_r + c, ar0, ar1);
    atomic_add_float2(grad_x_w + c, aw0, aw1);
    atomic_add_float2(grad_x_k + c, ak0, ak1);
    atomic_add_float2(grad_x_v + c, av0, av1);
    atomic_add_float2(grad_x_a + c, aa0, aa1);
    atomic_add_float2(grad_x_g + c, ag0, ag1);
}

__global__ void cast_float_to_bf16_vec2_kernel(
    const float* __restrict__ src,
    at::BFloat16* __restrict__ dst,
    int64_t c_size) {
    int64_t c_pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_pairs = c_size / 2;
    if (c_pair >= total_pairs) {
        return;
    }
    int64_t c = c_pair * 2;
    store_bf16x2(dst + c, __floats2bfloat162_rn(src[c], src[c + 1]));
}

} // namespace

std::vector<torch::Tensor> tmix_mix6_forward_v5_cuda(
    torch::Tensor x,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
    auto out_r = torch::empty_like(x);
    auto out_w = torch::empty_like(x);
    auto out_k = torch::empty_like(x);
    auto out_v = torch::empty_like(x);
    auto out_a = torch::empty_like(x);
    auto out_g = torch::empty_like(x);
    const int threads = 256;
    const int64_t bt_size = x.size(0) * x.size(1);
    const int64_t c_size = x.size(2);
    auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t total_pairs = bt_size * (c_size / 2);
    const int blocks = static_cast<int>(ceil_div(total_pairs, static_cast<int64_t>(threads)));
    tmix_mix6_forward_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<at::BFloat16>(),
        x_r.data_ptr<at::BFloat16>(),
        x_w.data_ptr<at::BFloat16>(),
        x_k.data_ptr<at::BFloat16>(),
        x_v.data_ptr<at::BFloat16>(),
        x_a.data_ptr<at::BFloat16>(),
        x_g.data_ptr<at::BFloat16>(),
        out_r.data_ptr<at::BFloat16>(),
        out_w.data_ptr<at::BFloat16>(),
        out_k.data_ptr<at::BFloat16>(),
        out_v.data_ptr<at::BFloat16>(),
        out_a.data_ptr<at::BFloat16>(),
        out_g.data_ptr<at::BFloat16>(),
        bt_size,
        x.size(1),
        c_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_r, out_w, out_k, out_v, out_a, out_g};
}

std::vector<torch::Tensor> tmix_mix6_backward_v5_cuda(
    torch::Tensor grad_r,
    torch::Tensor grad_w,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_a,
    torch::Tensor grad_g,
    torch::Tensor x,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
    auto grad_x = torch::empty_like(x);
    auto grad_x_r = torch::empty({x.size(2)}, x.options());
    auto grad_x_w = torch::empty({x.size(2)}, x.options());
    auto grad_x_k = torch::empty({x.size(2)}, x.options());
    auto grad_x_v = torch::empty({x.size(2)}, x.options());
    auto grad_x_a = torch::empty({x.size(2)}, x.options());
    auto grad_x_g = torch::empty({x.size(2)}, x.options());

    auto fp32_opts = x.options().dtype(torch::kFloat);
    auto grad_x_r_fp32 = torch::zeros({x.size(2)}, fp32_opts);
    auto grad_x_w_fp32 = torch::zeros({x.size(2)}, fp32_opts);
    auto grad_x_k_fp32 = torch::zeros({x.size(2)}, fp32_opts);
    auto grad_x_v_fp32 = torch::zeros({x.size(2)}, fp32_opts);
    auto grad_x_a_fp32 = torch::zeros({x.size(2)}, fp32_opts);
    auto grad_x_g_fp32 = torch::zeros({x.size(2)}, fp32_opts);

    const int threads = 256;
    const int64_t bt_size = x.size(0) * x.size(1);
    const int64_t c_size = x.size(2);
    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 fused_blocks(
        static_cast<unsigned int>(ceil_div(c_size / 2, static_cast<int64_t>(TMIX_PARAM_THREADS))),
        static_cast<unsigned int>(ceil_div(bt_size, static_cast<int64_t>(TMIX_PARAM_BT_TILE))),
        1);
    tmix_mix6_backward_fused_kernel_v5<<<fused_blocks, TMIX_PARAM_THREADS, 0, stream>>>(
        grad_r.data_ptr<at::BFloat16>(),
        grad_w.data_ptr<at::BFloat16>(),
        grad_k.data_ptr<at::BFloat16>(),
        grad_v.data_ptr<at::BFloat16>(),
        grad_a.data_ptr<at::BFloat16>(),
        grad_g.data_ptr<at::BFloat16>(),
        x.data_ptr<at::BFloat16>(),
        x_r.data_ptr<at::BFloat16>(),
        x_w.data_ptr<at::BFloat16>(),
        x_k.data_ptr<at::BFloat16>(),
        x_v.data_ptr<at::BFloat16>(),
        x_a.data_ptr<at::BFloat16>(),
        x_g.data_ptr<at::BFloat16>(),
        grad_x.data_ptr<at::BFloat16>(),
        grad_x_r_fp32.data_ptr<float>(),
        grad_x_w_fp32.data_ptr<float>(),
        grad_x_k_fp32.data_ptr<float>(),
        grad_x_v_fp32.data_ptr<float>(),
        grad_x_a_fp32.data_ptr<float>(),
        grad_x_g_fp32.data_ptr<float>(),
        bt_size,
        x.size(1),
        c_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int cast_blocks = static_cast<int>(ceil_div(c_size / 2, static_cast<int64_t>(threads)));
    cast_float_to_bf16_vec2_kernel<<<cast_blocks, threads, 0, stream>>>(grad_x_r_fp32.data_ptr<float>(), grad_x_r.data_ptr<at::BFloat16>(), c_size);
    cast_float_to_bf16_vec2_kernel<<<cast_blocks, threads, 0, stream>>>(grad_x_w_fp32.data_ptr<float>(), grad_x_w.data_ptr<at::BFloat16>(), c_size);
    cast_float_to_bf16_vec2_kernel<<<cast_blocks, threads, 0, stream>>>(grad_x_k_fp32.data_ptr<float>(), grad_x_k.data_ptr<at::BFloat16>(), c_size);
    cast_float_to_bf16_vec2_kernel<<<cast_blocks, threads, 0, stream>>>(grad_x_v_fp32.data_ptr<float>(), grad_x_v.data_ptr<at::BFloat16>(), c_size);
    cast_float_to_bf16_vec2_kernel<<<cast_blocks, threads, 0, stream>>>(grad_x_a_fp32.data_ptr<float>(), grad_x_a.data_ptr<at::BFloat16>(), c_size);
    cast_float_to_bf16_vec2_kernel<<<cast_blocks, threads, 0, stream>>>(grad_x_g_fp32.data_ptr<float>(), grad_x_g.data_ptr<at::BFloat16>(), c_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_x, grad_x_r, grad_x_w, grad_x_k, grad_x_v, grad_x_a, grad_x_g};
}
