// =============================================================================
// rwkv7-bench.cpp - Micro-benchmark for wkv7 forward
// =============================================================================

#include <cstdio>
#include <chrono>
#include <vector>
#include <random>
#include <cstring>

#include "rwkv7_wkv7.h"

int main(int argc, char ** argv) {
    int D = 64, T = 1, B = 1, H = 32;
    if (argc > 1) D = std::atoi(argv[1]);
    if (argc > 2) T = std::atoi(argv[2]);
    if (argc > 3) B = std::atoi(argv[3]);
    if (argc > 4) H = std::atoi(argv[4]);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> r(D * T * H * B), w(D * T * H * B),
                       k(D * T * H * B), v(D * T * H * B),
                       a(D * T * H * B);
    std::vector<float> state_in(D * D * H * B);
    std::vector<float> y(D * T * H * B), state_out(D * D * H * B);

    for (auto & x : r) x = dist(rng);
    for (auto & x : w) x = dist(rng);
    for (auto & x : k) x = dist(rng);
    for (auto & x : v) x = dist(rng);
    for (auto & x : a) x = dist(rng);
    for (auto & x : state_in) x = dist(rng) * 0.01f;

    // Pre-compute (kk * a) for the kernel, matching how the graph builder
    // passes it.  Review #6 fix: previous version passed raw `a`, which
    // is not what the kernel expects (it takes kk*a).
    std::vector<float> kka(D * T * H * B);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < T; ++t) {
                const int64_t off = (((int64_t)h * B + b) * T + t) * D;
                float sumsq = 0.0f;
                for (int i = 0; i < D; ++i) sumsq += k[off + i] * k[off + i];
                const float s = 1.0f / std::sqrt(std::max(sumsq, 1e-12f));
                for (int i = 0; i < D; ++i) {
                    kka[off + i] = (k[off + i] * s) * a[off + i];
                }
            }
        }
    }

    constexpr int kIters = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < kIters; ++it) {
        rwkv7::wkv7_forward(
                r.data(), w.data(), k.data(), v.data(),
                nullptr, kka.data(), state_in.data(),
                y.data(), state_out.data(),
                B, H, T, D, 4);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;
    std::printf("D=%d T=%d B=%d H=%d  avg=%.2f us/call  tokens/s=%.1f\n",
                D, T, B, H, us, T * B / us * 1e6);
    return 0;
}
