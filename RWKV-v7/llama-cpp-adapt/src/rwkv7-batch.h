// =============================================================================
// rwkv7-batch.h
// =============================================================================
#pragma once

#include <memory>
#include <functional>
#include <cuda_runtime.h>

namespace rwkv7 {

class BatchRunner {
public:
    BatchRunner();
    ~BatchRunner();

    void init(int max_concurrent);

    // Submit a batch of sequence forward passes. The callable is invoked
    // (stream_idx, seq_idx) and is expected to launch all kernels for that
    // sequence on the corresponding stream.
    void run_batch(int n_seqs,
                   std::function<void(int stream_idx, int seq_idx)> submit);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    BatchRunner(const BatchRunner &) = delete;
    BatchRunner & operator=(const BatchRunner &) = delete;
};

}  // namespace rwkv7
