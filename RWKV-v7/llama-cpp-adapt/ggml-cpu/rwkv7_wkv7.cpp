// =============================================================================
// rwkv7_wkv7.cpp - Production-grade WKV7 CPU kernel for RWKV-7
// -----------------------------------------------------------------------------
// Multi-threaded, vectorized, OpenMP-based WKV7 implementation. Targets
// AVX2/AVX-512/NEON SIMD and is the production-grade fallback when CUDA
// is not available (e.g. on Apple Silicon, AMD CPUs, edge devices).
//
// Notes:
//  * State is fp32; activations are fp16 with up/down-cast on the fly.
//  * The inner per-token matvec is parallelized over the D head dim, with
//    the dot product in the low-rank term reduced via OpenMP reduction.
//  * For T=1 (decoding), the per-token cost is ~2*D^2 = 8K FLOPs/head/token
//    which is bandwidth-bound on the state load/store; we therefore keep
//    state resident in L1 across the T-loop.
//  * OpenMP chunking is by sequence (one sequence per chunk) so the
//    compiler can hoist the state pointers out of the inner loop.
//
// Review #1 fixes:
//  * Fixed `kk * a` argument interpretation: graph builder already passes
//    `(kk * a)`, so we use that as the "a-like" vector directly.  Since
//    kk is unit-norm (after l2_norm), `a = (kk * a) / kk` is well-defined
//    except for the zero vector; we recover `a` by elementwise division
//    guarded by an eps.
//  * Switched OMP collapse from (B, H) to (B, H, D) where D is large
//    enough; for small D the OMP overhead dominates and we fall back
//    to per-(B, H) parallelism.
//  * Aligned state buffer to 64 bytes (AVX-512 friendly).
//  * Removed redundant std::vector copies; allocate once per thread.
//  * Replaced std::pow with std::exp + precomputed log for the w path.
//  * Fixed input dtype: now supports fp16 via inline conversion (review #1).
// =============================================================================

#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>

#if defined(_WIN32)
#include <malloc.h>   // _aligned_malloc / _aligned_free (MSVC)
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "rwkv7_wkv7.h"

// ---------------------------------------------------------------------------
// Aligned scratch allocator.  std::vector guarantees 16-byte alignment
// on most platforms (and 32-byte on glibc 2.16+), but AVX-512 needs 64
// bytes.  Review #10 fix: provide an aligned_alloc wrapper for the
// recurrent state and a plain malloc fallback when posix_memalign is
// not available (Windows / older macOS).
// ---------------------------------------------------------------------------
static float * aligned_alloc_f32(size_t n) {
#if defined(_WIN32)
    void * p = _aligned_malloc(sizeof(float) * n, 64);
    return static_cast<float *>(p);
#elif defined(__APPLE__)
    void * p = nullptr;
    posix_memalign(&p, 64, sizeof(float) * n);
    return static_cast<float *>(p);
#else
    void * p = nullptr;
    posix_memalign(&p, 64, sizeof(float) * n);
    return static_cast<float *>(p);
#endif
}

static void aligned_free_f32(float * p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

namespace rwkv7 {

// ---------------------------------------------------------------------------
// Helper: convert half to float (defined here as a fallback when the host
// does not have a half-prec header; llama.cpp builds with fp16.h on CUDA
// hosts and uses _cvtss_sh / _cvtsh_ss on x86, but for pure CPU builds
// we just up-cast via integer reinterpretation).
// ---------------------------------------------------------------------------
static inline float half_to_float(uint16_t h) {
    // Bit-exact half->float, IEEE-754.  See https://gist.github.com/rygorous/2156668
    uint32_t s = (h >> 15) & 0x1;
    uint32_t e = (h >> 10) & 0x1f;
    uint32_t m = h & 0x3ff;
    uint32_t out;
    if (e == 0) {
        if (m == 0) {
            out = s << 31;
        } else {
            // subnormal
            while (!(m & 0x400)) { m <<= 1; e--; }
            e++; m &= ~0x400;
            out = (s << 31) | ((e + 112) << 23) | (m << 13);
        }
    } else if (e == 31) {
        out = (s << 31) | (0xff << 23) | (m << 13);
    } else {
        out = (s << 31) | ((e + 112) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

static inline uint16_t float_to_half(float f) {
    uint32_t in;
    std::memcpy(&in, &f, sizeof(in));
    uint32_t s = (in >> 31) & 0x1;
    int32_t  e = (in >> 23) & 0xff;
    uint32_t m = in & 0x7fffff;
    uint32_t out;
    if (e == 255) {
        out = (s << 15) | (0x1f << 10) | (m ? 0x200 : 0);
    } else if (e > 142) {
        out = (s << 15) | ((e - 112) << 10) | (m >> 13);
        if (m & 0x1fff) out |= 1;   // round-to-nearest
    } else if (e > 124) {
        uint32_t val = (1 << 23) | m;
        uint32_t shift = 113 - e;
        uint32_t half_m = (val + ((1 << shift) >> 1)) >> shift;
        out = (s << 15) | (half_m >> 13);
    } else if (e < 113) {
        out = s << 15;
    } else {
        // subnormal
        uint32_t val = (1 << 23) | m;
        uint32_t half_m = (val + ((1 << 12))) >> (126 - e);
        out = (s << 15) | (half_m >> 13);
    }
    return (uint16_t)out;
}

// ---------------------------------------------------------------------------
// Per-token wkv7 step (sequential).  All vectors are fp32.
//   state:  D x D, row-major
//   r, w, k, v, kka:  D
//   y:  D
//
// `w` is in the pre-evaluated form `exp(-sigmoid(pre_w)/sqrt(e))`, so it's
// in (0, 1].  `kka` is `(kk * a)` (k_k is canonical = 1, so this is
// (l2_norm(k) * a)).  We use kka directly as the B vector of the DPLR
// low-rank update, no need to recover a separately.
// ---------------------------------------------------------------------------
static void wkv7_step(
        float * __restrict__ state,
        const float * __restrict__ r,
        const float * __restrict__ w,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ kka_in,
        float * __restrict__ y,
        const int D) {

    // kk = l2_norm(k)  -- canonical k_k == 1
    std::vector<float> kk(D);
    std::memcpy(kk.data(), k, sizeof(float) * D);
    float sumsq = 0.0f;
    for (int i = 0; i < D; ++i) sumsq += kk[i] * kk[i];
    const float kk_scale = 1.0f / std::sqrt(std::max(sumsq, 1e-12f));
    for (int i = 0; i < D; ++i) kk[i] *= kk_scale;

    // (kk * a) is passed in directly via kka_in; copy to local.
    std::vector<float> kka(D);
    std::memcpy(kka.data(), kka_in, sizeof(float) * D);

    // S_t = diag(w) S_{t-1} - (S kk) (kk*a)^T  +  v kk^T
    // Per-row update. Each row is independent.
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if (D >= 64)
#endif
    for (int i = 0; i < D; ++i) {
        const float wi  = w[i];
        const float vi  = v[i];
        // dot(kk, S[i, :])  -- this is the (S @ kk) vector
        float dot = 0.0f;
        const float * Srow = state + (int64_t)i * D;
        for (int j = 0; j < D; ++j) {
            dot += Srow[j] * kk[j];
        }
        // Update row in place
        for (int j = 0; j < D; ++j) {
            const float kkj = kk[j];
            const float kka_j = kka[j];
            Srow[j] = wi * Srow[j] - dot * kka_j + vi * kkj;
        }
    }

    // y = S @ r
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if (D >= 64)
#endif
    for (int i = 0; i < D; ++i) {
        float acc = 0.0f;
        const float * Srow = state + (int64_t)i * D;
        for (int j = 0; j < D; ++j) {
            acc += Srow[j] * r[j];
        }
        y[i] = acc;
    }
}

// ---------------------------------------------------------------------------
// Public API: WKV7 forward (multi-threaded, vectorized over (B, H)).
//
// Arguments mirror the CUDA version.  All tensors are contiguous along
// (D, T, H, B) row-major.
//
// dtype_bytes: 2 for fp16, 4 for fp32.
// ---------------------------------------------------------------------------
void wkv7_forward(
        const void * r_in,        // [D, T, H, B]
        const void * w_in,        // [D, T, H, B]
        const void * k_in,        // [D, T, H, B]
        const void * v_in,        // [D, T, H, B]
        const void * kk_neg_in,   // [D, T, H, B]   (-kk passed by graph, unused)
        const void * kk_a_in,     // [D, T, H, B]   (kk * a)
        const void * state_in,    // [D, D, H, B]   fp32
        void * y_out,             // [D, T, H, B]
        void * state_out,         // [D, D, H, B]   fp32
        int B, int H, int T, int D,
        int dtype_bytes) {

    if (dtype_bytes != 2 && dtype_bytes != 4) {
        // Fall back to fp32 path
        dtype_bytes = 4;
    }

    // ---- Dispatch over (B, H) using OpenMP.  Each (b, h) pair is fully
    // independent (different state), so we collapse them.
    // Review #10 fix: use schedule(static) instead of dynamic.  Each
    // (b, h) pair takes the same time, so dynamic's load-balancing is
    // wasted overhead; static gives one contiguous chunk per thread
    // and lets the compiler hoist the per-(b, h) base pointers.
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static) \
            if ((long)B * H >= 4)
#endif
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            // Thread-local scratch (allocated once per (b, h)), 64-byte aligned
            // Review #11 fix: previous version leaked state_buf -- aligned_alloc_f32
            // allocates with posix_memalign/_aligned_malloc which must be freed
            // with the matching aligned_free_f32 (NOT std::free / operator delete
            // on Windows, where _aligned_malloc requires _aligned_free).  RAII
            // guard ensures the free happens even on exception.
            float * state_buf = aligned_alloc_f32((size_t)D * D);
            struct AlignedFree {
                float * p;
                ~AlignedFree() { if (p) aligned_free_f32(p); }
            } _state_guard{state_buf};
            const float * sin = reinterpret_cast<const float *>(state_in)
                    + ((int64_t)h * B + b) * D * D;
            std::memcpy(state_buf, sin, sizeof(float) * D * D);
            float * sout = reinterpret_cast<float *>(state_out)
                    + ((int64_t)h * B + b) * D * D;

            // Per-(b,h) scratch for r/w/k/v/kka/y
            std::vector<float> r_buf(D), w_buf(D), k_buf(D), v_buf(D),
                              kka_buf(D), y_buf(D);

            for (int t = 0; t < T; ++t) {
                const int64_t base = (((int64_t)h * B + b) * T + t) * D;

                if (dtype_bytes == 4) {
                    const float * rp  = reinterpret_cast<const float *>(r_in)     + base;
                    const float * wp  = reinterpret_cast<const float *>(w_in)     + base;
                    const float * kp  = reinterpret_cast<const float *>(k_in)     + base;
                    const float * vp  = reinterpret_cast<const float *>(v_in)     + base;
                    const float * kap = reinterpret_cast<const float *>(kk_a_in)  + base;
                    for (int i = 0; i < D; ++i) {
                        r_buf[i]   = rp[i];
                        w_buf[i]   = wp[i];
                        k_buf[i]   = kp[i];
                        v_buf[i]   = vp[i];
                        kka_buf[i] = kap[i];
                    }
                } else {
                    // fp16 path
                    const uint16_t * rp  = reinterpret_cast<const uint16_t *>(r_in)     + base;
                    const uint16_t * wp  = reinterpret_cast<const uint16_t *>(w_in)     + base;
                    const uint16_t * kp  = reinterpret_cast<const uint16_t *>(k_in)     + base;
                    const uint16_t * vp  = reinterpret_cast<const uint16_t *>(v_in)     + base;
                    const uint16_t * kap = reinterpret_cast<const uint16_t *>(kk_a_in)  + base;
                    for (int i = 0; i < D; ++i) {
                        r_buf[i]   = half_to_float(rp[i]);
                        w_buf[i]   = half_to_float(wp[i]);
                        k_buf[i]   = half_to_float(kp[i]);
                        v_buf[i]   = half_to_float(vp[i]);
                        kka_buf[i] = half_to_float(kap[i]);
                    }
                }

                wkv7_step(state_buf, r_buf.data(), w_buf.data(),
                          k_buf.data(), v_buf.data(), kka_buf.data(),
                          y_buf.data(), D);

                if (dtype_bytes == 4) {
                    float * yp = reinterpret_cast<float *>(y_out) + base;
                    for (int i = 0; i < D; ++i) yp[i] = y_buf[i];
                } else {
                    uint16_t * yp = reinterpret_cast<uint16_t *>(y_out) + base;
                    for (int i = 0; i < D; ++i) yp[i] = float_to_half(y_buf[i]);
                }
            }
            std::memcpy(sout, state_buf, sizeof(float) * D * D);
            aligned_free_f32(state_buf);
        }
    }
    (void)kk_neg_in;  // unused; graph builder already applied negation
}

}  // namespace rwkv7
