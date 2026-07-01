Here we compare RWKV-7 and Qwen3.5 tensors. Code: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py

Note: these are tensors with similar (but usually not exactly the same) functions.

```
RWKV-7 V65536-L24-D1024 #params = 2*V*D + 4*D + L*(19D+12D*D+2*D*(64+32+64+128)) = 450.834432 M (note blocks.0.v0/v1/v2 is unused, so actually 450.767872 M)

Qwen3.5 V248320-L24-D1024 #params = V*D + D + L*2*D + L*3/4*(12*2*D+16*(2+2*D)+128+10*D*D) + L/4*(256*2+7*D*D) + L*(3*3.5*D*D) = 752.393024 M

RWKV-7 L24-D1024 #state_params = L*(2*D+64*D) = 1.622016 M

Qwen3.5 L24-D1024 #state_params = L*3/4*(3*6*D+2*128*D) + L/4*(2*2*256*T) = 5.050368 + 6.144*(T/1000) M
```

| RWKV-7 | shape | | Qwen3.5 GDN | shape | | Qwen3.5 GQA | shape |
|---|---:|---|---|---:|---|---|---:|
| emb.weight | [65536,1024] |  | embed_tokens.weight | [248320,1024] |  |  |  |
| blocks.0.ln0.weight | [1024] |  |  |  |  |  |  |
| blocks.0.ln0.bias | [1024] |  |  |  |  |  |  |
| blocks.*.ln1.weight | [1024] |  | layers.0.input_layernorm.weight | [1024] |  |  |  |
| blocks.*.ln1.bias | [1024] |  |  |  |  |  |  |
| blocks.*.ln2.weight | [1024] |  | layers.0.post_attention_layernorm.weight | [1024] |  |  |  |
| blocks.*.ln2.bias | [1024] |  |  |  |  |  |  |
| ln_out.weight | [1024] |  | norm.weight | [1024] |  |  |  |
| ln_out.bias | [1024] |  |  |  |  |  |  |
| head.weight | [65536,1024] |  | embed_tokens.weight | [248320,1024] |  |  |  |
|  |  |  |  |  |  |  |  |
| blocks.*.att.x_r | [1,1,1024] |  | layers.0.linear_attn.conv1d.weight[q] | [2048,1,4] |  | layers.3.self_attn.q_norm.weight | [256] |
| blocks.*.att.receptance.weight | [1024,1024] |  | layers.0.linear_attn.in_proj_qkv.weight[q] | [2048,1024] |  | layers.3.self_attn.q_proj.weight[q] | [2048,1024] |
|  |  |  |  |  |  |  |  |
| blocks.*.att.x_w | [1,1,1024] |  |  |  |  |  |  |
| blocks.*.att.w0 | [1,1,1024] |  | layers.0.linear_attn.dt_bias | [16] |  |  |  |
| blocks.*.att.w1 | [1024,64] |  | layers.0.linear_attn.in_proj_a.weight | [16,1024] |  |  |  |
| blocks.*.att.w2 | [64,1024] |  | layers.0.linear_attn.A_log | [16] |  |  |  |
|  |  |  |  |  |  |  |  |
| blocks.*.att.x_k | [1,1,1024] |  | layers.0.linear_attn.conv1d.weight[k] | [2048,1,4] |  | layers.3.self_attn.k_norm.weight | [256] |
| blocks.*.att.key.weight | [1024,1024] |  | layers.0.linear_attn.in_proj_qkv.weight[k] | [2048,1024] |  | layers.3.self_attn.k_proj.weight | [512,1024] |
| blocks.*.att.k_k | [1,1,1024] |  |  |  |  |  |  |
| blocks.*.att.k_a | [1,1,1024] |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| blocks.*.att.x_v | [1,1,1024] |  | layers.0.linear_attn.conv1d.weight[v] | [2048,1,4] |  |  |  |
| blocks.*.att.value.weight | [1024,1024] |  | layers.0.linear_attn.in_proj_qkv.weight[v] | [2048,1024] |  | layers.3.self_attn.v_proj.weight | [512,1024] |
| blocks.*.att.v0 | [1,1,1024] |  |  |  |  |  |  |
| blocks.*.att.v1 | [1024,32] |  |  |  |  |  |  |
| blocks.*.att.v2 | [32,1024] |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| blocks.*.att.x_a | [1,1,1024] |  |  |  |  |  |  |
| blocks.*.att.a0 | [1,1,1024] |  |  |  |  |  |  |
| blocks.*.att.a1 | [1024,64] |  | layers.0.linear_attn.in_proj_b.weight | [16,1024] |  |  |  |
| blocks.*.att.a2 | [64,1024] |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| blocks.*.att.x_g | [1,1,1024] |  |  |  |  |  |  |
| blocks.*.att.g1 | [1024,128] |  | layers.0.linear_attn.in_proj_z.weight | [2048,1024] |  | layers.3.self_attn.q_proj.weight[g] | [2048,1024] |
| blocks.*.att.g2 | [128,1024] |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| blocks.*.att.ln_x.weight | [1024] |  | layers.0.linear_attn.norm.weight | [128] |  |  |  |
| blocks.*.att.ln_x.bias | [1024] |  |  |  |  |  |  |
| blocks.*.att.r_k | [16,64] |  |  |  |  |  |  |
| blocks.*.att.output.weight | [1024,1024] |  | layers.0.linear_attn.out_proj.weight | [1024,2048] |  | layers.3.self_attn.o_proj.weight | [1024,2048] |
|  |  |  |  |  |  |  |  |
| blocks.*.ffn.x_k | [1,1,1024] |  | layers.0.mlp.gate_proj.weight | [3584,1024] |  |  |  |
| blocks.*.ffn.key.weight | [4096,1024] |  | layers.0.mlp.up_proj.weight | [3584,1024] |  |  |  |
| blocks.*.ffn.value.weight | [1024,4096] |  | layers.0.mlp.down_proj.weight | [1024,3584] |  |  |  |
