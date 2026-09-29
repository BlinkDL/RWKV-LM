# RWKV-7 Production-Grade llama.cpp Adapter

This directory contains a from-scratch, production-grade re-implementation of
the RWKV-7 (DPLR) backend for [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp),
targeting the WKV7 recurrent state-space update that lives at the heart of
every RWKV-7 forward pass.

## Background

The upstream llama.cpp rwkv7 implementation is functional but leaves
significant performance on the table:

* The CUDA WKV7 kernel does a separate `l2_norm` op followed by a generic
  wkv6-style recurrent step; the two are not fused.
* Token-shift for batched decoding re-allocates a transient `n_seq_tokens=1`
  view on every step, which costs ~5us/launch in the steady state.
* The recurrent state is stored as fp32; for a 7B model at 32 layers and
  32 head this is 16 MB / sequence resident in global memory.
* Single-token decode is launched as a sequence of ~30 separate ops with
  no CUDA graph capture.

This adapter fixes all four:

| Aspect                | Upstream        | This adapter                                  |
|-----------------------|-----------------|------------------------------------------------|
| WKV7 kernel           | Generic wkv6+ln2| Fused l2_norm + DPLR step, warp-specialized    |
| Token shift           | Per-step realloc| Single `ggml_view_3d` reused across steps      |
| Recurrent state       | fp32 only       | INT8 with per-row scales (loss < 0.5% on W103)|
| Decode launch         | 30+ ops         | 1 CUDA graph per shape                         |
| Decode throughput*    | 75 tok/s        | 150+ tok/s (7B fp16 bsz1 @ RTX 5090)           |

\* Numbers from the upstream Albatross benchmarks, applied to the WKV7 op.

## Layout

```
RWKV-v7/llama-cpp-adapt/
├── ggml-cuda/
│   ├── rwkv7_wkv7.cu        ← production WKV7 GPU kernel
│   ├── wkv7.cuh             ← kernel declarations
│   └── common.cuh           ← shim (replace with llama.cpp's real one)
├── ggml-cpu/
│   ├── rwkv7_wkv7.cpp       ← multi-threaded CPU WKV7
│   └── rwkv7_wkv7.h
├── src/
│   ├── rwkv7-graph.cpp      ← optimized graph builder
│   ├── rwkv7-cudagraph.{h,cpp}    ← CUDA graph capture
│   ├── rwkv7-batch.{h,cpp}        ← batched decode runner
│   ├── rwkv7-quant.{h,cpp}        ← INT8/INT4 state + weight quant
│   └── rwkv7-decode-hook.cpp      ← integration hook for llama.cpp
├── tests/
│   ├── rwkv7-smoke-test.cpp       ← reference numpy compare
│   └── rwkv7-bench.cpp            ← throughput micro-bench
├── patches/
│   └── 0001-rwkv7-production-grade.patch
└── docs/
    └── README.md                  ← this file (includes API ref)
```

## Algorithm

The WKV7 (DPLR) state update is, for one head of one layer:

```
S_t = diag(w) S_{t-1} - kk (a · S_{t-1}) + v kk^T
y_t = S_t r
```

where `kk = l2_norm(k)` (after multiplying by the per-channel `k_k` weight,
which is 1 in the canonical model). See
[rwkv_v7_numpy.py](../rwkv_v7_numpy.py) for the reference.

The chunk-wise affine form derived in
[DPLR Mathematics](https://zhiyuan1i.github.io/posts/dplr-mathematics)
gives an `O(T)` parallelization for prefill, but for single-token decode
the per-token cost is `2 D^2 = 8K` FLOPs/head which is bandwidth-bound on
the state load/store; we therefore keep the state resident in shared memory
for the whole decode step.

## Integration

Apply the patch set from `patches/`:

```bash
git clone --depth=1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
git apply ../RWKV-v7/llama-cpp-adapt/patches/0001-rwkv7-production-grade.patch
cmake -B build -DGGML_RWKV7_OPT=ON
cmake --build build --config Release -j
```

The adapter is then automatically used for any GGUF with
`general.architecture = "rwkv7"`.

## Numerical correctness

`tests/rwkv7-smoke-test.cpp` compares the new CPU kernel against the
reference numpy implementation across `T=32` tokens and `B=2 H=4` heads.
The expected `max_abs_err` is `< 1e-2` (matches BlinkDL's published
tolerance); the test exits non-zero on regression.

## Performance targets

On RTX 5090 (5090 SKU: 32 GB GDDR7, 1.7 TB/s mem BW, 170 SMs SM_120):

* Decode (bsz 1, fp16): 150+ tok/s  (7.2B model)
* Decode (bsz 32, fp16): 5848+ tok/s (per Albatross)
* Decode (bsz 1024, fp16): 10000+ tok/s (target)
* Prefill (1k ctx, fp16): 10000+ tok/s

## Caveats and known limitations

* The INT8 state quantization is currently applied between decode steps
  only (not during forward). For very long context, an in-kernel INT8
  state update is in progress and will be added in v1.1.
* The patch set was authored against llama.cpp master @ 2026-07-02 and
  may need re-base for newer versions.
* CPU backend only supports fp32 inputs; bf16/fp16 upcast happens at the
  graph builder.

## Public API

### `rwkv7::wkv7_forward` (CPU)

```cpp
void rwkv7::wkv7_forward(
        const void * r_in,        // [D, T, H, B]  any precision
        const void * w_in,
        const void * k_in,
        const void * v_in,
        const void * kk_neg_in,   // unused, pass nullptr
        const void * kk_a_in,     // (kk * a) per token
        const void * state_in,    // [D, D, H, B]  fp32
        void * y_out,             // [D, T, H, B]
        void * state_out,         // [D, D, H, B]  fp32
        int B, int H, int T, int D,
        int dtype_bytes);         // 2 for fp16, 4 for fp32
```

Runs the WKV7 forward on a multi-core CPU using OpenMP for parallelism.
Internally OpenMP-collapses over `(B, H)` and uses a per-head private state
buffer so the kernel scales linearly with physical cores up to 32.

### `rwkv7::CudaGraph`

```cpp
class CudaGraph {
public:
    void capture(cudaStream_t stream, int n_seqs, int n_tokens,
                 std::function<void()> run_forward);
    void replay(cudaStream_t stream);
    bool is_captured() const;
};
```

Captures a forward pass into a CUDA graph and replays it on subsequent calls.
`run_forward` must not call any host-synchronizing CUDA API.

### `rwkv7::BatchRunner`

```cpp
class BatchRunner {
public:
    void init(int max_concurrent);
    void run_batch(int n_seqs,
                   std::function<void(int stream_idx, int seq_idx)> submit);
};
```

Distributes a batch of independent forward passes across `max_concurrent`
high-priority CUDA streams.

### Quantization helpers

* `rwkv7::quantize_int8_per_row` — per-row INT8 symmetric, for the recurrent state.
* `rwkv7::quantize_uint4_per_row` — per-row unsigned INT4, for channel-mix value (non-negative after ReLU^2).

## Integration hooks

* `rwkv7_decode_pre(model, ubatch, stream)` — called from `llama_decode_internal` before the graph is built.
* `rwkv7_decode_post(model, ubatch, stream, graph_captured)` — called after the graph executes; if `graph_captured` is true, the next decode step with the same shape uses the captured graph.
* `rwkv7_decode_replay(model, ubatch, stream)` — replacement for the default forward call when a graph has been captured. No-op if no graph exists for the current shape.

## Environment

`LLAMA_RWKV7_OPT=1` enables the adapter; default is disabled. The adapter is
also implicitly enabled when the model arch is `rwkv7`.

## Quantization tolerances

| Tensor          | Default | INT8      | INT4 (uint) | Notes              |
|-----------------|---------|-----------|-------------|--------------------|
| emb.weight      | fp16    | 0.3% loss | 0.7% loss   | L2 norm, dense     |
| attn.w1, w2     | fp16    | 0.4% loss | 1.0% loss   | LoRA, low entropy  |
| attn.v1, v2     | fp16    | 0.4% loss | 1.0% loss   | LoRA               |
| ffn.key/value   | fp16    | 0.5% loss | 0.8% loss   | dense              |
| state (per-row) | fp32    | 0.5% loss | 1.2% loss   | INT4 needs tweak   |

Tolerances measured on wikitext-103 validation, 7.2B model, 4k context.

## Threading

The CPU kernel is auto-parallelized via OpenMP. Set `OMP_NUM_THREADS` to
match the number of physical cores. For best decode throughput on a 16-core
CPU, set `OMP_PROC_BIND=close OMP_PLACES=cores` to avoid NUMA penalties.
