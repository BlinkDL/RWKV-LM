// =============================================================================
// rwkv7-cudagraph.cpp - CUDA Graph capture for RWKV-7 decoding
// -----------------------------------------------------------------------------
// Captures a single-token forward pass into a CUDA graph, then re-instantiates
// it for every decode step. The capture step itself is amortized after
// `n_warmup` calls (typically 2-3).
//
// Why this matters for RWKV-7:
//   * Single-token decode is bandwidth-bound (one D^2 state load/store per
//     head per layer per step), so kernel launch overhead is a real
//     fraction of total time. CUDA Graph eliminates ~30 us/launch.
//   * The recurrent state buffer is fixed-size; only the input token
//     changes, so the graph is naturally re-entrant.
// =============================================================================

#include "rwkv7-cudagraph.h"

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstdio>

namespace rwkv7 {

// ---------------------------------------------------------------------------
// Per-shape CUDA graph. We keep one graph per (n_seqs, n_tokens) shape.
// ---------------------------------------------------------------------------
struct CudaGraph::Impl {
    cudaGraph_t       graph       = nullptr;
    cudaGraphExec_t   graph_exec  = nullptr;
    cudaStream_t      capture_stream = nullptr;
    std::vector<void *> captured_buffers;
    int               n_seqs      = 0;
    int               n_tokens    = 0;
    int               n_warmup    = 0;
};

// ---------------------------------------------------------------------------
// Capture: begin stream capture, run the user-provided lambda, end capture.
// ---------------------------------------------------------------------------
void CudaGraph::capture(
        cudaStream_t stream,
        int n_seqs, int n_tokens,
        std::function<void()> run_forward) {

    // Review #9 fix: tear down any previous capture first.  The Impl is
    // heap-allocated in this function and unique_ptr takes care of the
    // members, but the previous graph_exec (if any) must be destroyed
    // before we instantiate a new one to avoid a leak on re-capture.
    impl_.reset();

    impl_ = std::make_unique<Impl>();
    impl_->n_seqs   = n_seqs;
    impl_->n_tokens = n_tokens;
    impl_->capture_stream = stream;

    // Begin capture (relaxed mode lets us do memory allocation within)
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        impl_.reset();
        throw std::runtime_error(std::string("cudaStreamBeginCapture failed: ")
                                 + cudaGetErrorString(err));
    }

    // Warmup runs to make sure all lazy initializations are done.
    // Review #9 fix: wrap warmup + capture in a try/catch so a runtime
    // exception from `run_forward` doesn't leave the stream in
    // "capturing" state and leak the partial graph.
    try {
        constexpr int kWarmup = 2;
        for (int i = 0; i < kWarmup; ++i) {
            run_forward();
        }
        impl_->n_warmup = kWarmup;

        // Real capture
        run_forward();

        err = cudaStreamEndCapture(stream, &impl_->graph);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaStreamEndCapture failed: ")
                                     + cudaGetErrorString(err));
        }

        err = cudaGraphInstantiate(&impl_->graph_exec, impl_->graph, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaGraphInstantiate failed: ")
                                     + cudaGetErrorString(err));
        }
    } catch (...) {
        // Best-effort cleanup: end the capture (no-op if it already ended)
        // and destroy whatever was partially built.
        cudaGraph_t tmp_graph = impl_->graph;
        impl_->graph = nullptr;
        if (tmp_graph) {
            cudaGraphDestroy(tmp_graph);
        }
        impl_.reset();
        // Cancel the stream capture if it's still in flight.
        cudaStreamCaptureStatus capture_status;
        if (cudaStreamIsCapturing(stream, &capture_status) == cudaSuccess
                && capture_status == cudaStreamCaptureStatusActive) {
            cudaGraph_t dummy;
            cudaStreamEndCapture(stream, &dummy);
            if (dummy) cudaGraphDestroy(dummy);
        }
        throw;
    }
}

// ---------------------------------------------------------------------------
// Replay the captured graph.
// ---------------------------------------------------------------------------
void CudaGraph::replay(cudaStream_t stream) {
    if (!impl_ || !impl_->graph_exec) return;
    cudaError_t err = cudaGraphLaunch(impl_->graph_exec, stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "rwkv7::CudaGraph::replay failed: %s\n",
                     cudaGetErrorString(err));
    }
}

// ---------------------------------------------------------------------------
// Tear-down: destroy the exec and graph.
// ---------------------------------------------------------------------------
CudaGraph::~CudaGraph() {
    if (!impl_) return;
    if (impl_->graph_exec) cudaGraphExecDestroy(impl_->graph_exec);
    if (impl_->graph)      cudaGraphDestroy(impl_->graph);
}

CudaGraph::CudaGraph() = default;

}  // namespace rwkv7
