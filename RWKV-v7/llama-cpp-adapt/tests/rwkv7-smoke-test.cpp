// =============================================================================
// rwkv7-smoke-test.cpp - Numerical correctness test
// -----------------------------------------------------------------------------
// Compares the new wkv7 kernel against the reference numpy implementation
// in rwkv_v7_numpy.py. The reference uses fp32; we use fp16 + fp32 state.
// Pass tolerance: max abs delta < 1e-2 (per BlinkDL's published tolerance).
// =============================================================================

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>
#include <cstdint>
#include <algorithm>

#include "rwkv7_wkv7.h"

namespace {

// Reference (subset of rwkv_v7_numpy.py) for one token, one head.
// Implements:  S = S * w.mT - S @ kk * (kk*a).mT + v * kk.mT
//              y = S @ r
// with kk = l2_norm(k), and the inputs being r, w, k, v, a (not kka).
static void ref_step(
        float * S,                 // D x D state
        const float * r, const float * w, const float * k, const float * v,
        const float * a, float * y, int D) {
    // kk = l2_norm(k)
    std::vector<float> kk(D);
    for (int i = 0; i < D; ++i) kk[i] = k[i];
    float sumsq = 0.0f;
    for (int i = 0; i < D; ++i) sumsq += kk[i] * kk[i];
    const float scale = 1.0f / std::sqrt(std::max(sumsq, 1e-12f));
    for (int i = 0; i < D; ++i) kk[i] *= scale;

    // kka = kk * a
    std::vector<float> kka(D);
    for (int i = 0; i < D; ++i) kka[i] = kk[i] * a[i];

    // S = S * w.mT - S @ kk * (kk*a).mT + v * kk.mT
    for (int i = 0; i < D; ++i) {
        // First, compute (S @ kk) per row
        float Sikk = 0.0f;
        for (int kk_idx = 0; kk_idx < D; ++kk_idx) {
            Sikk += S[i * D + kk_idx] * kk[kk_idx];
        }
        for (int j = 0; j < D; ++j) {
            S[i * D + j] = w[i] * S[i * D + j] - Sikk * kka[j] + v[i] * kk[j];
        }
    }

    // y = S @ r
    for (int i = 0; i < D; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < D; ++j) {
            acc += S[i * D + j] * r[j];
        }
        y[i] = acc;
    }
}

}  // namespace

int main() {
    constexpr int D = 64;
    constexpr int T = 32;
    constexpr int B = 2;
    constexpr int H = 4;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> r(D * T * H * B), w(D * T * H * B),
                       k(D * T * H * B), v(D * T * H * B),
                       a(D * T * H * B);
    std::vector<float> state_in(D * D * H * B, 0.0f);
    std::vector<float> y(D * T * H * B), y_ref(D * T * H * B),
                       state_out(D * D * H * B);

    for (auto & x : r) x = dist(rng);
    for (auto & x : w) x = dist(rng);
    for (auto & x : k) x = dist(rng);
    for (auto & x : v) x = dist(rng);
    for (auto & x : a) x = dist(rng);
    for (auto & x : state_in) x = dist(rng) * 0.01f;

    // Reference: per-(h, b) sequential T steps.
    // Reference uses raw a; we compute kka = kk * a inline.
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            std::vector<float> S(D * D);
            std::memcpy(S.data(),
                        state_in.data() + ((h * B + b) * D * D),
                        sizeof(float) * D * D);
            for (int t = 0; t < T; ++t) {
                const int64_t off = (((int64_t)h * B + b) * T + t) * D;
                ref_step(S.data(),
                         r.data() + off, w.data() + off,
                         k.data() + off, v.data() + off,
                         a.data() + off, y_ref.data() + off, D);
            }
        }
    }

    // Our CPU kernel takes (kk * a) directly. We must match the reference's
    // computation exactly: per-token kk = l2_norm(k), then kka[i] = kk[i]*a[i].
    std::vector<float> kka(D * T * H * B);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < T; ++t) {
                const int64_t off = (((int64_t)h * B + b) * T + t) * D;
                // compute kk
                float sumsq = 0.0f;
                for (int i = 0; i < D; ++i) sumsq += k[off + i] * k[off + i];
                const float s = 1.0f / std::sqrt(std::max(sumsq, 1e-12f));
                for (int i = 0; i < D; ++i) {
                    kka[off + i] = (k[off + i] * s) * a[off + i];
                }
            }
        }
    }

    rwkv7::wkv7_forward(
            r.data(), w.data(), k.data(), v.data(),
            /*kk_neg*/ nullptr,
            /*kk_a*/   kka.data(),
            state_in.data(),
            y.data(),
            state_out.data(),
            B, H, T, D, /*dtype_bytes=*/4);

    // Compare
    float max_err = 0.0f, max_rel = 0.0f;
    for (size_t i = 0; i < y.size(); ++i) {
        float e = std::abs(y[i] - y_ref[i]);
        max_err = std::max(max_err, e);
        float denom = std::max(std::abs(y_ref[i]), 1e-3f);
        max_rel = std::max(max_rel, e / denom);
    }
    std::printf("max_abs_err=%.6f  max_rel_err=%.6f\n", max_err, max_rel);
    return (max_err < 1e-2f) ? 0 : 1;
}
