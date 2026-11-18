# Qwen3.8-27B (dense) - family `qwen35`

HF `Qwen3_5ForConditionalGeneration`, text model `qwen3_5_text`: 64 layers, hidden 5120, FFN 17408, vocab 248320 (untied head),
48 gated-delta-net (GDN) layers + 16 gated full-attention layers (layer l is attention iff `(l+1) % 4 == 0`), 1 MTP layer.
Vision tower is not converted. Host contract: `include/strix/qwen35/config.hpp`, `src/qwen35/`, registry row `qwen3_8_27b`.

## Semantics (reference: `tools/qwen35/ref_forward.py`)
- Layer: `x += Mixer(RMS(x)); x += MLP(RMS(x))`. RMSNorm `y = x*rsqrt(mean(x^2)+1e-6)*(1+w)`; the GGUF stores `1+w` (F32).
  The GDN gated norm is `x*rsqrt(..)*w*silu(z)` with plain `w`.
- Attention: `q_proj` rows are per head `[q 256 | gate 256]` (24 heads), K/V 4 heads x 256, per-head q/k RMSNorm, NeoX rotate-half
  RoPE on the first 64 dims (theta 1e7; text mrope == plain RoPE), GQA group 6, scale 1/16, `out = attn*sigmoid(gate)`.
- GDN: `[q 2048|k 2048|v 6144] = in_proj_qkv(x)`; causal depthwise conv k=4 (state = last 3 inputs, no bias) + SiLU;
  `beta = sigmoid(b)`, `g = ssm_a*softplus(a + dt_bias)` with `ssm_a = -exp(A_log)`; q,k l2-normalised, q scaled 1/sqrt(128);
  `S = S*exp(g); d = (v - S^T k)*beta; S += k (x) d; o = S^T q`; gated norm with `z`; `out_proj`. State f32 [48][128][128].
- MTP: `h' = eh_proj([RMS_e(emb(t_{i+1})) | RMS_h(h_i)])`, one attention layer + MLP with its own KV (RoPE position i),
  `shared_head_norm`, shared `output`. The reference takes `h_i` after the final norm (`--mtp-hidden pre` = residual before it).

## GGUF contract
`general.architecture = qwen35`; `qwen35.{block_count 64, context_length, embedding_length, feed_forward_length,
attention.{head_count, head_count_kv, key_length, value_length, layer_norm_rms_epsilon}, rope.{freq_base, dimension_count},
ssm.{conv_kernel, state_size, group_count, time_step_rank, inner_size}, full_attention_interval, nextn_predict_layers}`;
`tokenizer.ggml.*` (gpt2 BPE, pre `qwen35`). Tensors (full list: `expected_tensors()`): `token_embd, output_norm, output`;
`blk.N.{attn_norm, post_attention_norm, ffn_gate, ffn_up, ffn_down}`; GDN `{attn_qkv, attn_gate, ssm_conv1d, ssm_beta, ssm_alpha,
ssm_dt.bias, ssm_a, ssm_norm, ssm_out}`; attention `{attn_q, attn_k, attn_v, attn_output, attn_q_norm, attn_k_norm}`;
MTP `blk.64.{nextn.enorm, nextn.hnorm, nextn.eh_proj, nextn.shared_head_norm}` plus the attention/FFN set.
Matrices are BF16 or Q8_0, everything else F32. **V-head order is TILED** (engine convention): GGUF value head t holds HF head
`(t%16)*3 + t/16`; applied to v rows of `attn_qkv` and its conv channels, `attn_gate` rows, `ssm_beta/alpha` rows, `ssm_dt`,
`ssm_a`, and the `ssm_out` input columns (blocks of 128).

## Tools
```
python tools/qwen35/convert.py HF_DIR OUT.gguf --pack bf16|q8_0        # streams tensor by tensor
python tools/qwen35/convert.py --synthetic-hf DIR                        # tiny random HF model (real per-head dims)
python tools/qwen35/ref_forward.py M.gguf --ids 3,100,400 --n 32 --out DIR   # ref.npz + summary.txt (f64 accumulation)
python tools/qwen35/xcheck_transformers.py HF_DIR DIR/ref.npz            # one-shot Qwen3Next cross-check (torch)
```
Cross-check on the synthetic model (GGUF bf16 -> ref_forward vs transformers Qwen3NextForCausalLM, f32): max |dlogit| 1.1e-5,
top-1 8/8. Q8_0 uses ggml reference rounding (blocks of 32, d = amax/127 as f16, round half away from zero).
Real pack (2026-11-16): CUDA exact q8_0 vs ref_forward f64 on a 15-token chat prompt + 31 greedy tokens: max |dlogit| 0.032
(0.06 % of the logit range), top-1 46/46, greedy 31/31.

Measurement kits (`scripts/bench/qwen35/`, parametrised; hosts/paths are arguments):
`run_classes.sh BIN GGUF OUT [N]` exact drafting per prompt class (code/json/prose, EXACT gate inside);
`rows_table.sh BIN "MODEL_ARGS" "ROWS"` verify-window ms per rows; `nsys_step.sh NSYS BIN OUT "ARGS"` per-kernel step table;
`turbo_quality.py REF.bin TURBO.bin VOCAB [ALIGNED_ROWS]` TURBO gate (KL, top-1, error clustering).

## Sizes (parameters: trunk 24.35 B + head 1.27 B; vision skipped)
BF16 pack ~51.2 GB per decode pass; Q8_0 pack ~27.2 GB (34/32 bytes per weight). GDN state 151 MB, KV 64 KiB/position (bf16).

## CUDA backend notes

## Dense arithmetic identity (lead decision, 2026-11-16)
The qwen35 CUDA exact identity is `qwen35-cuda-exact-mma3-v1`: BF16 tensor-core `mma.sync m16n8k16` with FP32 accumulate,
the FP32 activation split exactly into three bf16 terms (hi + mid + lo, products with bf16 weights exact), accumulated
lo, mid, hi. One fixed K order for every row count 1..32 (rows ride the N dimension, so each row is bitwise identical to
rows = 1): 32-element K blocks, warp w of 8 owns groups of U consecutive blocks (U = 4 bf16, 8 q8_0), the 8 warp partials are
summed in ascending warp order. q8_0: int8 values are exact in bf16; each 32-block partial starts at zero and is folded as
`tot = tot + fl(d * part)`. Source: src/cuda_core/dense/dense_mma.cu.
- `--dense-arith fp32` keeps the CUDA-core FP32 kernels as the reference / off switch (documented order in dense.hpp).
- FAST policies: `mma2` (hi + mid), `mma1` (hi only; max |dlogit| 8x larger, see RESULT). Not the default.
- Shapes not covered (in % 32, out % 16, f32 weights) fall back to fp32 kernels.

## Multi-session batched decode (CUDA, exact class, default path for slots)

`CudaEngine::configure_slots(n)` gives n sessions (own KV, GDN/conv state, parity, length) that share the weights.
`forward_slots(groups, tokens)` runs one dense pass per layer over the rows of all groups (a group = consecutive rows of one
slot: 1 row = a decode lane, more = a verify/prefill window) with per-session attention and GDN: `cuda_core::RowMap`
(per-row position + cache slot for `kv_store`/`rope_apply`/`attention_decode`) and `gdn_window_slots`/`gdn_commit_slots`
(device slot tables, so one CUDA graph per (rows, groups) serves any set of sessions). Every row is bitwise equal to the
session run alone (`qwen35_cuda_engine_test` section 4; `strix-cuda-qwen35-decode --slots N --slots-verify` on real packs).
Windows commit with `commit_slots(accept[])` (partial accept supported). Bench: `--slots N --arms 8,4,1 --ab-pairs 3 --n-gen S
--ctx-fill C` (strict ABAB, host-timed steps incl. token read-back, SM clock). `StepPlanner`/`plan_rows` lane rows map 1:1 to groups.

## Bounded prefill submission (CUDA, scheduling only)
A long prefill is queued chunk by chunk through `cuda_core::SubmitWindow` (`include/strix/cuda_core/submit_window.hpp`):
before the host queues chunk i it waits for chunk i - N to finish, so at most N chunks are queued ahead of the GPU instead
of the whole prompt before one sync. Same kernels, same per-stream order and event dependencies: output is bitwise
identical (logits hashes equal at 8K and 32K, throughput neutral). `strix-cuda-qwen35-decode` flags:
- `--prefill-inflight N` (default 3): chunks queued ahead of the GPU; `0` = unbounded (the previous form, off switch).
- `--prefill-ramp R` (default 0 = off): the first chunks grow R, 2R, 4R, ... rows (multiples of 128, below
  `--prefill-chunk`) so the card steps up from idle. It changes chunk boundaries, so it is exact only where the prefill
  is chunk-size invariant (measured for this model); it stays off by default.
Both values are printed by `--print-effective-config` and in the prefill line (`chunks`, `inflight`).
