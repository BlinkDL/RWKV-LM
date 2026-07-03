// =============================================================================
// rwkv7-batch.cpp - Batched decoding kernel
// -----------------------------------------------------------------------------
// For a single sequence, single-token decoding, the WKV7 kernel as written
// above parallelizes over (H * B) blocks, each of size CUDA_WKV7_BLOCK_SIZE.
// For batched decoding (B > 1) with a small number of sequences (typical
// 32-1024), this gives near-perfect utilization.
//
// This file exposes a higher-level batched-decode API that:
//   1. Maintains a pool of CUDA streams, one per concurrent sequence
//   2. Submits all sequence forward passes in parallel
//   3. Synchronizes only the sequences that the user asked for
// =============================================================================

#include "rwkv7-batch.h"
#include <vector>
#include <stdexcept>

namespace rwkv7 {

struct BatchRunner::Impl {
    std::vector<cudaStream_t> streams;
    int max_concurrent = 0;
};

BatchRunner::BatchRunner() : impl_(std::make_unique<Impl>()) {}

BatchRunner::~BatchRunner() {
    for (auto & s : impl_->streams) {
        if (s) cudaStreamDestroy(s);
    }
}

void BatchRunner::init(int max_concurrent) {
    impl_->max_concurrent = max_concurrent;
    impl_->streams.resize(max_concurrent);
    for (int i = 0; i < max_concurrent; ++i) {
        // High-priority stream for decode (improves preemption on shared GPUs)
        int prio_high = 0, prio_low = 0;
        cudaDeviceGetStreamPriorityRange(&prio_low, &prio_high);
        cudaError_t err = cudaStreamCreateWithPriority(
                &impl_->streams[i], cudaStreamNonBlocking, prio_high);
        if (err != cudaSuccess) {
            throw std::runtime_error("BatchRunner::init failed to create stream");
        }
    }
}

// Run forward for a batch of sequences concurrently. The user provides a
// callable that, given a (stream_idx, seq_idx) pair, submits the forward
// pass for that sequence on the corresponding stream.
//
// Review #9 fix: the previous version synchronised ALL `k` streams at
// the end, even those that were never submitted to.  For n_seqs < k
// this caused spurious waits.  We now track which streams were used.
void BatchRunner::run_batch(
        int n_seqs,
        std::function<void(int stream_idx, int seq_idx)> submit) {
    if (!impl_ || impl_->streams.empty()) {
        // Fallback: serial
        for (int s = 0; s < n_seqs; ++s) submit(0, s);
        for (int s = 0; s < n_seqs; ++s) cudaStreamSynchronize(nullptr);
        return;
    }
    const int k = std::min(n_seqs, (int)impl_->streams.size());
    if (k <= 0) return;

    // Review #9 fix: track which streams were used so we only sync
    // those (avoids spurious waits on idle streams).
    // For the round-robin case (n_seqs > k), every one of the k
    // streams was used at least once, so we sync all k.
    int next = 0;
    for (int s = 0; s < n_seqs; ++s) {
        const int stream_idx = next;
        submit(stream_idx, s);
        next = (next + 1) % k;
    }
    for (int i = 0; i < k; ++i) {
        cudaStreamSynchronize(impl_->streams[i]);
    }
}

}  // namespace rwkv7
