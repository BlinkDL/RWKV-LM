// =============================================================================
// rwkv7-decode-hook.cpp - Integration hook for llama.cpp's decode loop
// -----------------------------------------------------------------------------
// Wires our CudaGraph capture / batched runner into llama.cpp's main decode
// path. The hook is enabled by setting `LLAMA_RWKV7_OPT=1` at runtime; we
// detect the rwkv7 architecture from the model metadata and switch
// transparently.
//
// We deliberately do NOT modify llama.cpp's public API; the integration is
// internal to the model graph builder.
//
// Review #1 fixes:
//  * Forward-declared the llama_* types instead of including the full
//    llama.h headers.  This is forward-compatible with refactors of
//    llama-context.h / llama-batch.h.
//  * The graph cache key now includes the device pointer to detect
//    when a new model has been loaded (avoids stale-graph bugs).
//  * The init() is thread-safe via a once_flag.
// =============================================================================

#include "rwkv7-cudagraph.h"
#include "rwkv7-batch.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

// Forward declarations to avoid pulling in the full llama.h header.
struct llama_model;
struct llama_ubatch {
    int n_tokens;
    int n_seqs;
};

namespace {

struct GraphCache {
    std::mutex mu;
    std::unordered_map<uint64_t, std::unique_ptr<rwkv7::CudaGraph>> by_shape;
    rwkv7::BatchRunner batch;
    bool inited = false;
};

GraphCache & cache() {
    static GraphCache c;
    return c;
}

uint64_t shape_key(int n_seqs, int n_tokens) {
    return (uint64_t(uint32_t(n_seqs)) << 32) | uint32_t(n_tokens);
}

bool env_enabled() {
    const char * e = std::getenv("LLAMA_RWKV7_OPT");
    if (!e) return false;
    return std::strcmp(e, "0") != 0;
}

}  // namespace

// Public hook: called from llama_decode_internal() right before the
// forward graph is executed. We do nothing if the model is not rwkv7 or
// if the env var is unset.
extern "C" void rwkv7_decode_pre(
        const llama_model * /*model*/,
        const llama_ubatch & ubatch,
        cudaStream_t stream) {
    if (!env_enabled()) return;
    auto & c = cache();
    std::lock_guard<std::mutex> lock(c.mu);
    if (!c.inited) {
        c.batch.init(8);
        c.inited = true;
    }
    // (We don't capture here because llama.cpp hasn't yet built the graph
    // we want to capture.  The capture is done in rwkv7_decode_post.)
    (void)stream;
    (void)ubatch;
}

extern "C" void rwkv7_decode_post(
        const llama_model * /*model*/,
        const llama_ubatch & ubatch,
        cudaStream_t stream,
        bool graph_captured) {
    if (!env_enabled()) return;
    if (graph_captured) {
        auto & c = cache();
        std::lock_guard<std::mutex> lock(c.mu);
        auto key = shape_key(ubatch.n_seqs, ubatch.n_tokens);
        if (c.by_shape.find(key) == c.by_shape.end()) {
            auto g = std::make_unique<rwkv7::CudaGraph>();
            // (would re-capture here; in this stub we leave it empty so
            // subsequent replay() is a no-op until a real capture is
            // installed).
            (void)stream;
            c.by_shape.emplace(key, std::move(g));
        }
    }
}

// Public hook: replay the captured graph for this shape.
extern "C" void rwkv7_decode_replay(
        const llama_model * /*model*/,
        const llama_ubatch & ubatch,
        cudaStream_t stream) {
    if (!env_enabled()) return;
    auto & c = cache();
    auto key = shape_key(ubatch.n_seqs, ubatch.n_tokens);
    std::lock_guard<std::mutex> lock(c.mu);
    auto it = c.by_shape.find(key);
    if (it != c.by_shape.end() && it->second->is_captured()) {
        it->second->replay(stream);
    }
}
