# deepseek41 on CUDA: numeric identity of the layer core

Status: layer-core kernels for sm_120 (`STRIX_ENABLE_CUDA_DS41`, off by default), bitwise equal to the host reference
on synthetic inputs and on real tensors of the pinned model. Attention and the routed experts are separate pieces
(their CUDA kernels follow the same rules; see the end of this document).

Files: `include/strix/cuda/ds41_common.hpp` (conventions, buffers, views), `include/strix/cuda/ds41_core.hpp` (sites),
`src/cuda/models/ds41/ds41_core*.cu`, `src/cuda/models/ds41/ds41_cuda_common.{cu,cuh}`, tests and tools in `tests/cuda/`.

## The identity

The CUDA side has no identity of its own: every site computes the declared FP32 arithmetic of `ds41-exact-v1`
(`ds41_layer.hpp`, `ds41_moe.hpp`, `ds41_formats.hpp`), the same that the host reference and the gfx1151 kernels follow.
A CUDA run is therefore bitwise equal to the host forward op by op, and a pipeline whose first layers run on CUDA
produces the tokens of the all-AMD run. No site carries a tolerance.

What makes this hold:

- **Compiler**: every CUDA translation unit is built with `--fmad=false` (no contraction of a product and an add into
  FMA), `-prec-div=true` and `-prec-sqrt=true` (IEEE round-to-nearest division and square root), `-ftz=false`
  (denormals kept), host code with `-ffp-contract=off`. The FP32 expressions are written in the reference's
  operation order, one rounding per operation.
- **Transcendentals**: `declared_exp`, `declared_log`, `declared_log1p`, `router_score`, `declared_sigmoid`,
  `swiglu_clamped`, `engram_gate_of` are the shared host/device definitions of the headers (basic IEEE operations
  only); `declared_sqrt` is `__fsqrt_rn` on NVIDIA devices (IEEE square root, the correctly rounded value the host
  gets from `sqrtf`). `rint` rounds half to even on both sides.
- **Conversions**: F16 → FP32, E4M3 / E8M0 → FP32 are exact; bfloat16 rounding of Engram rows is the shared
  `round_to_bf16`.
- **Integer products**: a Q8_0 × A8 group dot is at most 32 · 127 · 127 = 516,128 in magnitude, so its int32 sum is
  exact in any order and `float(dot)` is exact (below 2^24). The CUDA kernels compute it with `dp4a` (decode form) or
  with one `mma.m16n8k32.s8` per group (tensor-core form, int32 accumulation, exact); the order of the 32 products
  inside a group is irrelevant, so the tensor-core form permutes `k` identically in the weight and activation
  fragments.
- **Reductions in the declared order** (`ds41_layer.hpp`):
  - *lane dot*: lane `l` of 32 takes elements `l, l + 32, …` in ascending order from 0, `acc = acc + a * b`, then the
    butterfly `v[l] = v[l] + v[l ^ d]`, `d = 16 … 1` (`__shfl_xor_sync`). Addition is commutative in IEEE arithmetic, so
    the butterfly can be evaluated as the tree `v[l] = v[l] + v[l + d]`, `l < d`, result `v[0]`.
  - *chunked dot* (`dot_chunk = 640`): a lane dot per 640-value chunk, the chunk sums added in ascending chunk order
    from 0.
  - *small fixed sums* (four hyper-connection copies, Sinkhorn rows and columns, router weights): ascending index.
- **Max / argmax / top-k** are order-free when no NaN is involved; the NaN cases reproduce the sequential reference
  (A8 amax ignores NaN as `std::max(amax, |v|)` does; a NaN router key is taken only when it is the first untaken
  expert; argmax skips non-finite logits and flags them).

## Sites

| Site (plan op) | CUDA form | Declared order reproduced |
| --- | --- | --- |
| A8 activations (`quantize_a8`) | one warp per 32-value group | `delta = amax / 127`, `code = clamp(rint(x / delta))` with the reference's comparison order, tiny/zero groups, code sums |
| Q8_0 × A8 sites: `attn_q_a`, `attn_kv`, `attn_q_b`, grouped `attn_output_a`, K-split `attn_output_b`, shared expert gate/up/down, `output` | decode form: one warp per output, 32-block chunks staged in shared memory, `dp4a`, 1–8 rows per pass. Tensor-core form: tiled weights (per 16 rows × 32 weights, the 512 fragment bytes and 16 scales, repacked once from the GGUF bytes), one `mma` per group for 16 outputs × 8 rows, any row count | per group `float(dot) * (scale * delta)`; lane class `c` (groups `c, c + 32, …`) summed ascending from 0; the 32 class sums combined in the butterfly tree. In the tensor-core form unit `u` (warp `w` of block `s`, `U = 8 · S` units, `S` = 1, 2 or 4) owns classes `u, u + U, …` and takes the tree levels `d ≥ U` itself; levels `d < U` run in shared memory (`S = 1`) or in a second kernel over a scratch buffer (`S > 1`). K-split sites: one call per part in part order, `out = out + part` from the second (`h0 + h1`). Grouped `attn_output_a`: outputs `[g·1024, (g+1)·1024)` read the A8 slice `g` |
| Q8_0 × A8 sites, many-row form (`pk::enqueue_q8_0_a8_rows`, `cuda_ds41_prefill_q8.hpp`; prefill driver option `prefill_q8`, from 64 rows) | the tiled weights of the tensor-core form; a block computes 64 outputs × 64 rows, each warp a 32 × 16 register tile over the full depth, groups staged by `cp.async` | the same terms and class sums; the warp visits the classes in bit-reversed order `c = bitrev5(j)`, `j = 0 … 31` (the butterfly tree's leaves left to right) and folds the tree as a binary counter: when class `j` ends, its sum is the right operand of the saved sums at the trailing one bits `k` of `j` (`cur = stk[k] + cur`), then saved at the first zero bit; after `j = 31`, `cur` is `v[0]`. `float(dot)` as `(1.5·2^23 + dot) − 1.5·2^23` (exact for dot below 2^22 in magnitude); the FP16 scale through an exact integer / multiply-by-2^112 path. Bitwise the tensor-core form for every row count (`strix-cuda-ds41-prefill-q8-tests`) |
| Concatenated sites (tensor-core form) | `attn_q_a` + `attn_kv` (shared input) and the shared expert's gate + up as one tiled tensor each | each output row is still its own `product_q8_0`; concatenation only changes which launch computes it |
| `hc_mix` | chunk partials of the 24 F16 `fn` rows and of the sum of squares (one warp per dot, chunk, row), then one warp per row: ascending chunk sums, `m * rsqrt(sumsq / n + eps)`, the split, softmax rows, 20 Sinkhorn rounds (16 lanes, one per `comb` entry, row and column sums in ascending index by every lane) | `hc_raw_mixes` + `hc_split_sinkhorn` |
| `hc_pre` + `rms_norm` (+ A8) | one launch per site input: each block rebuilds the collapsed row (`y = Σ_c pre[c] · x[c]` ascending from 0, identity `pre` before layer 0) and its chunk sums of squares, and writes one 640-value part (FP32 and/or A8); in place: one block per row | `hc_collapse`, `rms_norm`, `quantize_a8` |
| `hc_post` | elementwise, the four copies of a position in one thread (in place allowed) | `out[k] = post[k] · sub + (Σ_j comb[j][k] · res[j] ascending from 0)` |
| `router` | one warp per expert, the F32 row read once for up to 8 rows; one warp per row selects | lane dot, `router_score`, biased top-k (lowest id on ties), weights `score / (ascending sum + 1e-20) · scale` |
| `shared_expert` | gate, up (or gate+up tiled), SwiGLU + A8 per row, down | `reference_window` with one Q8_0 expert (raw vector, weight 1) |
| `embed` | F16 row to every copy, identity `pre_carry` | `embed_token` |
| `engram` | gather of 264-byte rows (device or mapped host table), decode + bfloat16 rounding, `engram_kv` F16 lane dot (segments of 256 staged in shared memory, lane classes kept), per (row, copy) the three chunked dots and the gate, `h = h + gate · value` | `engram_rows_to_bf16`, `f16_rows`, `engram_gates`, `engram_inject` (`inv_sqrt_hidden` computed in double on the host) |
| `head` + `argmax` | collapse + `output_norm` + A8, the `output` Q8_0 site, one block per row argmax | `argmax_slice` + `combine_argmax` (equal to the single-node selection: largest finite logit, lowest index on ties, non-finite flagged) |

Every form of a site gives a row the bits of the one-row reference whatever the number of rows, the row passes, the
split or the concatenation.

## What is not bitwise

Nothing in the layer core. Two remarks:

- NaN payloads: IEEE addition is commutative except for which NaN payload propagates; a step that produces a NaN is
  rejected by the head's finiteness check on both sides, so payloads never reach a token.
- A future faster form that changes a reduction order (for example a split-K with a different combination, or
  FP16/FP8 tensor-core accumulation) would be a new identity and would need a quality gate against the reference
  (KL / top-1), as any requantization does. None is used.

## Validation

- `strix-cuda-ds41-core-tests` (ctest, seconds): every site bitwise against its host reference on synthetic inputs:
  A8 (zero, tiny, NaN, infinity groups), Q8_0 sites plain / grouped / K-split / concatenated at 1, 3, 8, 11 and 70 rows
  in both forms and splits 1/2/4, norms with and without collapse and in place, hc mixes at 1/4/9 rows and expansion in
  place, router with ties and image rows, shared expert in both forms, embedding, Engram, argmax with cross-half ties
  and NaN.
- `strix-cuda-ds41-core-real-tests` (ctest; needs `STRIX_DS41_FIXTURE_GGUF`, a file with the dense tensors of layers
  0–2, the embedding and the head): the decode data flow of layers 0, 1 and 2 and the head on the real tensors
  (Engram rows synthetic, attention and routed outputs stand-ins), each site against the host reference with the
  reference's inputs, at 1, 3 and 9 rows.
- `strix-ds41-tap-capture` (host only, `STRIX_DS41_TAP_CAPTURE=ON` builds it without CUDA) runs the host forward over
  layers `[0, N)` and records per step the token, the Engram rows read, every op output buffer and the op hashes;
  `strix-cuda-ds41-core-tap-check WEIGHTS CAPTURE [--chain]` replays every op the layer core implements on the device
  with the captured inputs (or chained within the step) and compares buffers bitwise and logits / token / route ids by
  op hash.
- `strix-cuda-ds41-core-bench [rows] [layer] [iterations] [tiled]` times one layer's core sites in sequence and as one
  CUDA graph, and the head, with the L2 rewritten before every iteration (not a test).

## Rules for the other CUDA pieces

Attention (window ring, compressor, indexer, D512 attention with sink) and the routed experts (IQ2_XXS / Q2_K windows,
combine) keep the identities of `attention.md` and `moe.md`: the same compiler flags, the shared
host/device helpers of the headers, integer dots wherever the reference has them, and the reference's FP32 order for
every reduction (lane classes and butterfly for the expert products, the attention's declared chunk orders, the
two-half combine `(shared + low) + high`).
