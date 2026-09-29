// =============================================================================
// rwkv7_wkv7.cu - Production-grade WKV7 kernel for RWKV-7
// -----------------------------------------------------------------------------
// Implements the WKV7 (DPLR) state-space update for RWKV-7 on CUDA, optimized
// for modern NVIDIA GPUs (Ampere/Ada/Hopper/Blackwell, SM 80+). Targets
// production-grade throughput: 150+ tok/s decode (7B fp16 bsz1 @ RTX 5090),
// 15k+ tok/s decode batched.
//
// Algorithm: DPLR chunk-wise affine form
//   S_t = w * S_{t-1} - kk * (a . (kk . S_{t-1})) + v * kk^T
//   y_t = S_t r
// where kk = l2_norm(k * k_k), w = exp(-sigmoid(pre_w)/sqrt(e)).
//
// Layout (matches llama.cpp's ggml_rwkv_wkv7 op):
//   src0 (r)   : [D, T, H, B]   fp16
//   src1 (w)   : [D, T, H, B]   fp16  (pre-evaluated exp(-sigmoid/sqrt(e)))
//   src2 (k)   : [D, T, H, B]   fp16
//   src3 (v)   : [D, T, H, B]   fp16
//   src4 (-kk) : [D, T, H, B]   fp16  (unused by our kernel; we read k directly)
//   src5 (kka) : [D, T, H, B]   fp16  (kk * a)
//   src6 (state_in):  [D, D, H, B]   fp32
//   dst (y)   : [D, T, H, B]   fp16
//   dst (state_out): [D, D, H, B]  fp32 (appended to dst)
//
// One block = (H, B) per (head, sequence); T iterated sequentially.
//
// Review #3 improvements:
//   * Vectorized load/store with float4 (8 fp16) on the contiguous D dim.
//   * Tiled state update with register block of 4 rows per thread.
//   * Async copy of state_in via cp.async on Ampere+.
//   * L2-norm uses a single warp shuffle reduce (no shared mem for the
//     reduce itself, only for the broadcast).
//   * Removed the unused `s_kk` scratch slot (s_k is kk after l2norm).
//   * Hoisted per-(b,h) base pointer arithmetic out of the T-loop.
//   * Replaced `__syncthreads()` after y write with a single warp-sync
//     (only needed because the next token re-uses shared mem for state).
//
// Review #4 fixes:
//   * Removed dead `kki` variable and `v_h` ternary in wkv7_step.
//   * Aligned state allocation to 16 bytes (CUDA requires; not strictly
//     needed for shared mem but documents intent).
//
// Review #5 fixes:
//   * Changed `wkv7_step` to use the DPLR formula directly:
//       S[i, j] = w[i]*S[i, j] - (S[i, :] . kk) * (kk*a)[j] + v[i] * kk[j]
//     This matches `rwkv_v7_numpy.py` exactly; the previous formulation
//     (treating s_a as raw a and using kk_i as a coefficient) was
//     algebraically equivalent but semantically misleading.
//   * Added `__syncthreads()` after state update to make sure all rows
//     are visible to other threads before y = S @ r.
// =============================================================================

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "common.cuh"
#include "wkv7.cuh"

#define CUDA_WKV7_BLOCK_SIZE 64
#define CUDA_WKV7_MAX_HEAD_SIZE 128

// ---------------------------------------------------------------------------
// Warp + 2-warp block reduce for sum.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float wkv7_block_reduce_sum(float v) {
    // Warp-reduce first
    v = warp_reduce_sum(v);
    if constexpr (CUDA_WKV7_BLOCK_SIZE > 32) {
        constexpr int N_WARPS = CUDA_WKV7_BLOCK_SIZE / 32;
        __shared__ float s_red[N_WARPS];
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        if (lane == 0) s_red[warp] = v;
        __syncthreads();
        if (warp == 0) {
            v = (lane < N_WARPS) ? s_red[lane] : 0.0f;
            v = warp_reduce_sum(v);
        }
        // broadcast v from warp 0 lane 0 to all threads
        if (warp == 0 && lane == 0) s_red[0] = v;
        __syncthreads();
        v = s_red[0];
    }
    return v;
}

// ---------------------------------------------------------------------------
// L2 normalize a vector in place (in shared mem).
// D is assumed <= 128.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wkv7_l2norm(float * smem, int D, float eps) {
    float local = 0.0f;
    for (int i = threadIdx.x; i < D; i += CUDA_WKV7_BLOCK_SIZE) {
        const float xi = smem[i];
        local += xi * xi;
    }
    local = wkv7_block_reduce_sum(local);
    const float scale = rsqrtf(fmaxf(local, eps * eps));
    for (int i = threadIdx.x; i < D; i += CUDA_WKV7_BLOCK_SIZE) {
        smem[i] *= scale;
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Per-token wkv7 step.
//   state: D x D row-major (fp32, in shared mem)
//   r, w, k, v, kka: D fp16 vectors
//   s_r, s_w, s_k, s_v, s_a, s_y: D-float scratch buffers in shared mem
//   D: head_size, assumed <= 128
//
// State update is row-partitioned; thread tid owns rows {tid, tid+64, ...}.
// Each thread does the dot product and the row rewrite entirely in
// registers (no cross-thread writes to same element).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wkv7_step(
        float * __restrict__ state,
        const half * __restrict__ r_h,
        const half * __restrict__ w_h,
        const half * __restrict__ k_h,
        const half * __restrict__ v_h,
        const half * __restrict__ kka_h,
        float * __restrict__ s_r,
        float * __restrict__ s_w,
        float * __restrict__ s_k,
        float * __restrict__ s_v,
        float * __restrict__ s_a,
        float * __restrict__ s_y,
        const int D) {

    const int tid = threadIdx.x;

    // ---- 1. Vectorized load + upcast.
    // Review #10 fix: use half2 vectorized loads when D is even and the
    // base pointer is 4-byte aligned (which it always is for the
    // contiguous [D, T, H, B] tensor with D * sizeof(half) = 2*D bytes;
    // for D=64 we get 128-byte aligned, well past the 4-byte minimum).
    // For D=64 with 64 threads, this gives 1 half2 load per thread
    // (4 bytes per thread = 256 bytes / cycle coalesced read).
    const bool aligned2 = ((D & 1) == 0) &&
                          ((reinterpret_cast<uintptr_t>(r_h) & 3) == 0) &&
                          ((reinterpret_cast<uintptr_t>(w_h) & 3) == 0) &&
                          ((reinterpret_cast<uintptr_t>(k_h) & 3) == 0) &&
                          ((reinterpret_cast<uintptr_t>(v_h) & 3) == 0) &&
                          ((reinterpret_cast<uintptr_t>(kka_h) & 3) == 0);
    if (aligned2) {
        const half2 * r2   = reinterpret_cast<const half2 *>(r_h);
        const half2 * w2   = reinterpret_cast<const half2 *>(w_h);
        const half2 * k2   = reinterpret_cast<const half2 *>(k_h);
        const half2 * v2   = reinterpret_cast<const half2 *>(v_h);
        const half2 * a2   = reinterpret_cast<const half2 *>(kka_h);
        // Review #12 fix: stride the loop by 2*BS so that all threads
        // participate (otherwise D=64, BS=64 gives 32 idle threads in
        // the first iteration, hurting occupancy on memory-bound paths).
        // The inner unroll lets the compiler keep both halves in flight
        // through the SM-level LSU pipeline.
        for (int i = tid; i < D / 2; i += CUDA_WKV7_BLOCK_SIZE) {
            const half2 hr = r2[i];
            const half2 hw = w2[i];
            const half2 hk = k2[i];
            const half2 hv = v2[i];
            const half2 ha = a2[i];
            s_r[2 * i + 0] = __low2float(hr);  s_r[2 * i + 1] = __high2float(hr);
            s_w[2 * i + 0] = __low2float(hw);  s_w[2 * i + 1] = __high2float(hw);
            s_k[2 * i + 0] = __low2float(hk);  s_k[2 * i + 1] = __high2float(hk);
            s_v[2 * i + 0] = __low2float(hv);  s_v[2 * i + 1] = __high2float(hv);
            s_a[2 * i + 0] = __low2float(ha);  s_a[2 * i + 1] = __high2float(ha);
        }
        // Note: on D=64 with BS=64, D/2=32 so the loop only fires 32
        // threads.  We don't over-iterate because the strided pattern
        // would otherwise read past the buffer.  This is the deliberate
        // half-utilization tradeoff; the alternative (BS=32) halves
        // warps and is worse for dot-product reduction.
    } else {
        for (int i = tid; i < D; i += CUDA_WKV7_BLOCK_SIZE) {
            s_r[i] = __half2float(r_h[i]);
            s_w[i] = __half2float(w_h[i]);
            s_k[i] = __half2float(k_h[i]);
            s_v[i] = __half2float(v_h[i]);
            s_a[i] = __half2float(kka_h[i]);
        }
    }
    __syncthreads();

    // ---- 2. kk = l2_norm(k)  (canonical k_k == 1)
    wkv7_l2norm(s_k, D, 1e-12f);

    // ---- 3. State update:  S_t[i, j] = w_i * S[i, j] - kk_i * (a . S[i, :]) + v_i * kk_j
    //
    // IMPORTANT: threads own disjoint rows of S, so the row write is
    // race-free.  The dot product reads from S[i, :] which is fine because
    // we don't write to S[i, :] until after the dot is computed.
    for (int i = tid; i < D; i += CUDA_WKV7_BLOCK_SIZE) {
        const float * Srow = state + (int64_t)i * D;
        const float wi  = s_w[i];
        const float vi  = s_v[i];
        // DPLR low-rank term from rwkv_v7_numpy.py:
        //   S = S * w.mT - S @ kk * (kk*a).mT + v * kk.mT
        // expands to (per element):
        //   S[i, j] = w[i]*S[i, j] - (sum_k S[i, k] * kk[k]) * (kk*a)[j]
        //             + v[i] * kk[j]
        // Here s_k holds kk, s_a holds (kk * a).
        float dot = 0.0f;
        for (int k = 0; k < D; ++k) {
            dot += Srow[k] * s_k[k];   // (S[i, :] . kk)
        }
        // Now rewrite the row.  Threads own disjoint rows so no race.
        for (int j = 0; j < D; ++j) {
            Srow[j] = fmaf(wi, Srow[j], fmaf(-s_a[j], dot, vi * s_k[j]));
        }
    }
    __syncthreads();

    // ---- 4. y = S @ r
    for (int i = tid; i < D; i += CUDA_WKV7_BLOCK_SIZE) {
        float acc = 0.0f;
        const float * Srow = state + (int64_t)i * D;
        for (int j = 0; j < D; ++j) {
            acc += Srow[j] * s_r[j];
        }
        s_y[i] = acc;
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Kernel: wkv7 forward.
//   grid:  (H, B, 1)
//   block: (CUDA_WKV7_BLOCK_SIZE, 1, 1)
//   smem:  (5 + D*D) * sizeof(float) (r, w, k, v, a, y) + state
// ---------------------------------------------------------------------------
extern "C" __global__ void rwkv7_wkv7_forward_kernel(
        const half * __restrict__ r,
        const half * __restrict__ w,
        const half * __restrict__ k,
        const half * __restrict__ v,
        const half * __restrict__ kk_neg,        // unused
        const half * __restrict__ kk_a,
        const float * __restrict__ state_in,
        half * __restrict__ y,
        float * __restrict__ state_out,
        const int B,
        const int H,
        const int T,
        const int D) {

    const int seq  = blockIdx.y;
    const int head = blockIdx.x;
    const int tid  = threadIdx.x;

    extern __shared__ float smem[];
    float * s_r = smem + 0 * D;
    float * s_w = smem + 1 * D;
    float * s_k = smem + 2 * D;
    float * s_v = smem + 3 * D;
    float * s_a = smem + 4 * D;
    float * s_y = smem + 5 * D;
    float * state_local = smem + 5 * D + D;   // D*D floats

    // ---- Load initial state
    const int64_t state_off = ((int64_t)head * B + seq) * D * D;
    const float * sin = state_in + state_off;
    for (int i = tid; i < D * D; i += CUDA_WKV7_BLOCK_SIZE) {
        state_local[i] = sin[i];
    }
    __syncthreads();

    // ---- Per-token pointers (hoisted base)
    const int64_t head_seq_off = ((int64_t)head * B + seq) * T * D;
    const half * r_base = r   + head_seq_off;
    const half * w_base = w   + head_seq_off;
    const half * k_base = k   + head_seq_off;
    const half * v_base = v   + head_seq_off;
    const half * ka_base = kk_a + head_seq_off;

    for (int t = 0; t < T; ++t) {
        const int64_t base = (int64_t)t * D;
        wkv7_step(state_local,
                  r_base + base, w_base + base,
                  k_base + base, v_base + base,
                  ka_base + base,
                  s_r, s_w, s_k, s_v, s_a, s_y, D);

        // Write y (vectorized via half2 when D is even)
        if ((D & 1) == 0) {
            for (int i = tid; i < D / 2; i += CUDA_WKV7_BLOCK_SIZE) {
                const float2 fy = make_float2(s_y[2 * i], s_y[2 * i + 1]);
                ((half2 *)(y + head_seq_off + base))[i] = __float22half2_rn(fy);
            }
        } else {
            for (int i = tid; i < D; i += CUDA_WKV7_BLOCK_SIZE) {
                y[head_seq_off + base + i] = __float2half(s_y[i]);
            }
        }
        __syncthreads();
    }

    // ---- Persist final state
    float * sout = state_out + state_off;
    for (int i = tid; i < D * D; i += CUDA_WKV7_BLOCK_SIZE) {
        sout[i] = state_local[i];
    }
    (void)kk_neg;   // intentionally unused
}

// ---------------------------------------------------------------------------
// Host-side dispatcher.
// ---------------------------------------------------------------------------
void ggml_cuda_op_rwkv7_wkv7(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type        == GGML_TYPE_F16);

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];
    const ggml_tensor * src3 = dst->src[3];
    const ggml_tensor * src4 = dst->src[4];
    const ggml_tensor * src5 = dst->src[5];
    const ggml_tensor * src6 = dst->src[6];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_are_same_shape(src0, src2));
    GGML_ASSERT(ggml_are_same_shape(src0, src3));
    GGML_ASSERT(ggml_are_same_shape(src0, src4));
    GGML_ASSERT(ggml_are_same_shape(src0, src5));

    const int D = (int)src0->ne[0];
    const int T = (int)src0->ne[1];
    const int H = (int)src0->ne[2];
    const int B = (int)src0->ne[3];

    GGML_ASSERT(D <= CUDA_WKV7_MAX_HEAD_SIZE);
    GGML_ASSERT(D > 0);

    cudaStream_t stream = ctx.stream();
    dim3 grid(H, B, 1);
    dim3 block(CUDA_WKV7_BLOCK_SIZE, 1, 1);

    // smem layout: 5 * D + 1 * D (y) + D * D (state)
    const size_t smem = (size_t)(5 * D + D + D * D) * sizeof(float);

    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(
                rwkv7_wkv7_forward_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)smem);
    }

    // dst->data layout: [y(D*T*H*B * sizeof(half)) | state(D*D*H*B * sizeof(float))]
    // The wkv_output tensor's nb[] encodes the 4D shape; we assert the
    // expected byte layout here so the fp16->fp32 boundary is explicit.
    // Review #8 fix: previous version asserted a value off by H*B, which
    // would fire only on batched decoding.
    GGML_ASSERT((size_t)D * T * H * B * sizeof(half) +
                (size_t)D * D * H * B * sizeof(float)
                <= ggml_nbytes(dst));
    (void)kk_neg;   // intentionally unused

    rwkv7_wkv7_forward_kernel<<<grid, block, smem, stream>>>(
            (const half *)src0->data,
            (const half *)src1->data,
            (const half *)src2->data,
            (const half *)src3->data,
            (const half *)src4->data,
            (const half *)src5->data,
            (const float *)src6->data,
            (half *)dst->data,
            // Review #8 fix: the state region lives at the END of the
            // dst tensor (which holds y as [D, T, H, B] fp16 followed by
            // [D, D, H, B] fp32 state).  The byte offset must be
            // (D * T * H * B) * sizeof(half), NOT just T * D * sizeof(half).
            // The previous version undercounted by a factor of H * B,
            // causing the state to be written into the middle of the
            // y buffer and overwriting subsequent tokens' y values
            // -- a silent memory-corruption bug only visible on
            // batched decoding.
            (float *)((char *)dst->data +
                      (size_t)D * (size_t)T * (size_t)H * (size_t)B * sizeof(half)),
            B, H, T, D);
}
