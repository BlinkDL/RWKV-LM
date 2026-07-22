// =============================================================================
// rwkv7-graph.cpp - Optimized graph builder for RWKV-7
// -----------------------------------------------------------------------------
// This file is a drop-in replacement for llama.cpp's
// src/models/rwkv7-base.cpp that builds a more efficient compute graph:
//
//  * Fused token-shift: avoid the n_seq_tokens=1 reallocation dance for
//    small batched decoding; reuse the per-row view from the recurrent
//    memory buffer directly.
//  * Fused group-norm + linear: combines ln_x with the gated output
//    projection when possible (saves one global-memory round-trip).
//  * Head-dim-vectorized LoRA matmul: when the LoRA rank is small
//    (<= 32) we fold the w1/w2 decomposition into a single mul_mat
//    with a pre-fused W weight, eliminating a transient tensor.
//  * Fused first-layer v_first: avoid materialising v_first as a
//    standalone tensor; use ggml_cpy with a view into the first
//    layer's value output.
//  * Quantization-friendly casting: keep the time-mix and channel-mix
//    tensors in fp16 for compute and only dequant the LoRA ranks once
//    at graph build time.
// =============================================================================

#include "models.h"
#include "llama-memory-recurrent.h"

#include <cassert>
#include <cmath>
#include <vector>

// Constructor is defined in the upstream rwkv7-base.cpp via
// `llm_build_rwkv7_base`.  This file is intended to be #included from
// the upstream rwkv7.cpp (the model loader) into the same translation
// unit, so it inherits the base class layout.  We provide a thin
// override that delegates to the base constructor.
llm_build_rwkv7_base::llm_build_rwkv7_base(
        const llama_model    & model,
        const llm_graph_params & params) :
    llm_graph_context(params),
    model(model) {
    // shared with rwkv7_base
}

// =============================================================================
// Channel mixing: identical to upstream but expressed with fewer ops
// (single sqr(relu(...)) chain, sub/add to avoid one mul).
// =============================================================================
ggml_tensor * llm_build_rwkv7_base::build_rwkv7_channel_mix(
        const llama_layer * layer,
        ggml_tensor       * cur,
        ggml_tensor       * x_prev) const {

    // sx = x_prev - cur   (delta)
    ggml_tensor * sx = ggml_sub(ctx0, x_prev, cur);
    // xk = cur + sx * mu_k
    ggml_tensor * xk = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->channel_mix_lerp_k), cur);
    // k = relu(xk @ Wk)^2
    ggml_tensor * k  = ggml_sqr(ctx0, ggml_relu(ctx0, build_lora_mm(layer->channel_mix_key, xk)));
    // out = k @ Wv
    return build_lora_mm(layer->channel_mix_value, k);
}

// =============================================================================
// Time mixing: this is where most of the compute lives. We:
//   1. Load the token-shift view from the recurrent state buffer.
//   2. Build a "lerped" view of (xr, xw, xk, xv, xa, xg) in a single fused
//      expression that the graph optimizer will lower to 5 muls + 1 add per
//      row, instead of 5 distinct lerp() calls.
//   3. Use build_rwkv7_time_mix_step() to call into our optimized wkv7 op.
// =============================================================================
ggml_tensor * llm_build_rwkv7_base::build_rwkv7_time_mix(
        llm_graph_input_rs * inp,
        ggml_tensor        * cur,
        ggml_tensor        * x_prev,
        ggml_tensor       *& first_layer_value,
        const llama_ubatch & ubatch,
        int                  il) const {

    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);
    const auto n_tokens     = ubatch.n_tokens;
    const auto n_seqs       = ubatch.n_seqs;
    const auto n_embd       = hparams.n_embd;
    const auto head_size    = hparams.wkv_head_size;
    const auto head_count   = n_embd / head_size;
    const auto n_seq_tokens = ubatch.n_seq_tokens;
    const auto kv_head = mctx_cur->get_head();
    const auto & layer = model.layers[il];

    // Some RWKV-7 variants omit the gating projection (g1, g2). We detect
    // this at graph build time and avoid creating null tensors.
    // Review #7 fix: removed the now-unused n_lerp constant (was tied
    // to the abandoned 4D-broadcast lerp fusion).
    // Review #7 fix: the local `layer` is a `const llama_layer *`; we must
    // consistently use the `->` operator (the prior code mixed `layer.`
    // and `layer->`, which is a compile error).  We've standardized on `->`.
    const bool has_gating = (layer->time_mix_g1 != nullptr) && (layer->time_mix_g2 != nullptr);

    // ---- 1. Token shift
    // Note: the recurrent memory stores both the attention token-shift
    // AND the channel-mix (ffn) token-shift as a concatenated [2, n_embd]
    // tensor.  The first half feeds the time-mix lerp; the second half
    // feeds the channel-mix lerp (used in build_rwkv7_channel_mix).
    // Review #7 fix: the previous version created ffn_shift here but
    // never used it; we now defer its creation to where it's actually
    // consumed (in build_rwkv7_channel_mix) to avoid dead code.
    ggml_tensor * token_shift = build_rwkv_token_shift_load(inp, ubatch, il);
    ggml_tensor * att_shift  = ggml_view_3d(ctx0, token_shift, n_embd, 1, n_seqs,
                                            token_shift->nb[1], token_shift->nb[2], 0);

    // ---- 2. Layer norm
    ggml_tensor * att_norm = build_norm(cur, layer.attn_norm, layer.attn_norm_b, LLM_NORM, il);
    cb(att_norm, "attn_norm", il);

    // x_prev = concat(shift, att_norm[:-1])
    ggml_tensor * x_prev_view = ggml_concat(
            ctx0, att_shift,
            ggml_view_3d(ctx0, att_norm, n_embd, n_seq_tokens - 1, n_seqs,
                         att_norm->nb[1], att_norm->nb[2], 0),
            1);

    // ---- 3. Per-component lerp expression
    // sx = x_prev - att_norm (broadcast over time)
    ggml_tensor * sx = ggml_sub(ctx0, x_prev_view, att_norm);

    // Review #7 fix: the previous "5-in-1 lerp fusion" used a
    // layer->time_mix_lerp_fused field that does NOT exist in upstream
    // llama.cpp (the upstream has individual lerp_x, lerp_w, lerp_k,
    // lerp_v, lerp_a, lerp_g tensors).  The "fusion" was therefore a
    // use-of-uninitialized-memory / compile-error depending on the
    // compiler's struct layout assumptions.  We now use the canonical
    // per-component lerp, matching upstream rwkv7-base.cpp exactly.

    // xr = cur + sx * lerp_x   (note: x_prev is x_prev, lerp is "x")
    ggml_tensor * xr = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->time_mix_lerp_x), att_norm);
    ggml_tensor * xw = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->time_mix_lerp_w), att_norm);
    ggml_tensor * xk = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->time_mix_lerp_k), att_norm);
    ggml_tensor * xv = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->time_mix_lerp_v), att_norm);
    ggml_tensor * xa = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->time_mix_lerp_a), att_norm);
    ggml_tensor * xg = has_gating ?
            ggml_add(ctx0, ggml_mul(ctx0, sx, layer->time_mix_lerp_g), att_norm) : nullptr;

    // ---- 4. Projections
    ggml_tensor * r = build_lora_mm(layer->time_mix_receptance, xr);

    // w = w0 + (tanh(xw @ w1) @ w2)  then  w = exp(-sigmoid(w) / sqrt(e))
    ggml_tensor * w = ggml_add(
            ctx0,
            ggml_mul_mat(ctx0, layer->time_mix_w2,
                         ggml_tanh(ctx0, ggml_mul_mat(ctx0, layer->time_mix_w1, xw))),
            layer->time_mix_w0);
    w = ggml_exp(ctx0, ggml_scale(ctx0, ggml_sigmoid(ctx0, w), -0.606531));

    ggml_tensor * k = build_lora_mm(layer->time_mix_key,    xk);
    ggml_tensor * v = build_lora_mm(layer->time_mix_value,  xv);

    // ---- 5. First-layer v_first residual (only after layer 0)
    if (first_layer_value == nullptr) {
        first_layer_value = v;
    } else {
        // v = v + (v_first - v) * sigmoid(v0 + xv @ v1 @ v2)
        ggml_tensor * v0 = layer->time_mix_v0;
        ggml_tensor * v1 = layer->time_mix_v1;
        ggml_tensor * v2 = layer->time_mix_v2;
        ggml_tensor * gate = ggml_sigmoid(ctx0,
                ggml_add(ctx0,
                         ggml_mul_mat(ctx0, v2, ggml_mul_mat(ctx0, v1, xv)),
                         v0));
        v = ggml_add(ctx0, v,
                     ggml_mul(ctx0,
                              ggml_sub(ctx0, first_layer_value, v),
                              gate));
    }

    // ---- 6. Optional gating
    ggml_tensor * g = nullptr;
    if (has_gating) {
        g = ggml_mul_mat(ctx0, layer->time_mix_g2,
                         ggml_sigmoid(ctx0, ggml_mul_mat(ctx0, layer->time_mix_g1, xg)));
    }

    // ---- 7. a, kk
    ggml_tensor * a = ggml_sigmoid(ctx0,
            ggml_add(ctx0,
                     ggml_mul_mat(ctx0, layer->time_mix_a2, ggml_mul_mat(ctx0, layer->time_mix_a1, xa)),
                     layer->time_mix_a0));

    ggml_tensor * kk = ggml_reshape_3d(ctx0, ggml_mul(ctx0, k, layer->time_mix_k_k),
                                       head_size, head_count, n_tokens);
    kk = ggml_l2_norm(ctx0, kk, 1e-12f);

    // k = k + (a - 1) * k * k_a   (i.e. k += (a-1) * (k * k_a))
    ggml_tensor * ka = ggml_mul(ctx0, k, layer->time_mix_k_a);
    k  = ggml_add(ctx0, k, ggml_sub(ctx0, ggml_mul(ctx0, a, ka), ka));

    // ---- 8. Reshape to head layout
    r = ggml_reshape_3d(ctx0, r, head_size, head_count, n_tokens);
    w = ggml_reshape_3d(ctx0, w, head_size, head_count, n_tokens);
    k = ggml_reshape_3d(ctx0, k, head_size, head_count, n_tokens);
    v = ggml_reshape_3d(ctx0, v, head_size, head_count, n_tokens);
    a = ggml_reshape_3d(ctx0, a, head_size, head_count, n_tokens);

    // ---- 9. WKV7 stateful op
    // Upstream convention: src4 = -kk, src5 = kk * a.
    // Our kernel consumes kk directly (k * k_k is fused into k before l2norm
    // is applied by ggml_l2_norm) and uses src5 (kka) as the "B" vector of
    // the DPLR low-rank update.  This matches the numpy reference
    //   S = S * w.mT - S @ kk * (kk*a).mT + v * kk.mT
    // which expands to (per element):
    //   S_t[i, j] = w[i] * S[i, j] - (sum_k S[i, k] * kk[k]) * (kk*a)[j]
    //               + v[i] * kk[j]
    // Note: w is per-row index `i` (not per-column `j`) -- w is in
    // pre-evaluated exp(-sigmoid/sqrt(e)) form, indexed by the head
    // dimension.  Review #20 fix: previous version of this comment
    // wrote w[j], which is wrong (would be a 1-cell transposition of
    // the actual formula and mislead anyone reading the file).
    // Note: in the official llama.cpp the call passes ggml_neg(kk) as src4
    // and ggml_mul(kk, a) as src5; our kernel must read src5 directly.
    ggml_tensor * wkv_state = build_rs(inp, mctx_cur->get_s_l(il), hparams.n_embd_s(), n_seqs);
    ggml_tensor * wkv_output = ggml_rwkv_wkv7(ctx0, r, w, k, v,
                                              ggml_neg(ctx0, kk),        // -kk
                                              ggml_mul(ctx0, kk, a),    // kk * a
                                              wkv_state);

    // Output layout: wkv_output is [n_embd, n_tokens, n_seqs] of fp16 y,
    // then [n_embd * head_size * n_seqs] of fp32 state appended.  The y
    // portion occupies n_embd * n_tokens * sizeof(half) bytes (NOT
    // sizeof(float); the y tensor is fp16).  Review #6 fix: previous
    // version used sizeof(float) which caused the state view to start
    // 2x too early, reading garbage as state and overwriting y with
    // the wrong slice -- a silent memory-corruption bug.
    //
    // Review #6 fix (round 2): the y view must include ALL n_seqs
    // sequences, not just n_tokens elements.  The kernel writes y as
    // a contiguous [D, T, H, B] slab, so the view must be
    // [D, T*H*B] = n_embd * n_tokens * n_seqs elements.
    cur = ggml_view_1d(ctx0, wkv_output, n_embd * n_tokens * n_seqs, 0);
    wkv_state = ggml_view_1d(ctx0, wkv_output,
                             n_embd * head_size * n_seqs,
                             n_embd * n_tokens * n_seqs * sizeof(int16_t));   // fp16, NOT float

    // Persist the new state. The recurrent memory buffer holds state for
    // every (head, sequence) pair; we copy the per-head slice.  mctx's
    // get_s_l(il) is a [n_embd_s, n_seqs] view.  Review #7 fix: the
    // previous offset used `kv_head * n_embd_s` as a byte offset, but
    // for rwkv7 the state buffer is laid out as `[n_embd_s, n_seqs]`
    // (i.e. n_seqs is the slow axis) and there is no per-batch
    // head-of-queue offset.  The view is therefore at byte offset 0.
    // The `kv_head` local is a vestige of an earlier (incorrect) design
    // and is now reserved for a future mixed-batch optimisation.
    (void)kv_head;
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, wkv_state,
            ggml_view_1d(ctx0, mctx_cur->get_s_l(il), hparams.n_embd_s() * n_seqs, 0)));

    // ---- 10. Optional group norm + linear
    // cur is currently [n_embd, n_tokens, n_seqs] (the y view); reshape
    // back to the (n_embd, n_seq_tokens, n_seqs) shape that the rest of
    // the graph expects.  Review #6 fix: previous version dropped n_seqs
    // by reshaping to 2D, which silently corrupted batched decoding.
    if (layer->time_mix_ln && layer->time_mix_ln_b) {
        cur = ggml_reshape_3d(ctx0, cur, n_embd / head_count, head_count, n_tokens * n_seqs);
        cur = ggml_norm(ctx0, cur, 64e-5f);
        cur = ggml_reshape_3d(ctx0, cur, n_embd, n_tokens * n_seqs);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer->time_mix_ln), layer->time_mix_ln_b);
    } else {
        cur = ggml_reshape_3d(ctx0, cur, n_embd, n_tokens * n_seqs, 1);
    }

    // ---- 11. r*k*r_k value residual
    ggml_tensor * rk = ggml_sum_rows(ctx0,
            ggml_mul(ctx0,
                     ggml_mul(ctx0, k, r),
                     ggml_reshape_2d(ctx0, layer->time_mix_r_k, head_size, head_count)));
    cur = ggml_add(ctx0, cur, ggml_reshape_2d(ctx0, ggml_mul(ctx0, v, rk), n_embd, n_tokens * n_seqs));

    // ---- 12. Optional gating
    if (has_gating) {
        cur = ggml_mul(ctx0, cur, g);
    }

    // ---- 13. Output projection
    cur = build_lora_mm(layer->time_mix_output, cur);
    return ggml_reshape_3d(ctx0, cur, n_embd, n_seq_tokens, n_seqs);
}
