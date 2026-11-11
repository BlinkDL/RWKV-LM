// =============================================================================
// rwkv7-cudagraph.h
// =============================================================================
#pragma once

#include <memory>
#include <functional>
#include <cuda_runtime.h>

namespace rwkv7 {

class CudaGraph {
public:
    CudaGraph();
    ~CudaGraph();

    // Capture a forward pass into a CUDA graph. The lambda must contain only
    // CUDA kernel launches and host->device copies (no host sync).
    void capture(cudaStream_t stream,
                 int n_seqs, int n_tokens,
                 std::function<void()> run_forward);

    // Replay the captured graph.
    void replay(cudaStream_t stream);

    // True if a graph has been captured.
    bool is_captured() const { return static_cast<bool>(impl_); }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    CudaGraph(const CudaGraph &) = delete;
    CudaGraph & operator=(const CudaGraph &) = delete;
};

}  // namespace rwkv7
