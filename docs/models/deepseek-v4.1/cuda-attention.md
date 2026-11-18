# deepseek41 CUDA attention (sm_120)

The CSA2 attention of the deepseek41 family on CUDA, for the layer backend behind `STRIX_ENABLE_CUDA_DS41` (off by
default): `include/strix/cuda/ds41_attention.hpp`, `src/cuda/models/ds41/ds41_attention.cu` (kernels),
`src/cuda/models/ds41/ds41_attention_state.cu` (per-layer state and the layer sequence), tests
`tests/cuda/ds41_attention_tests.cu` (`strix-cuda-ds41-attention-tests`). Conventions (namespace, `enqueue_*`,
checks, flags) are those of `cuda_ds41_common.hpp`; the attention lives in `strix::ds41::cuda::attn`.

## Numeric identity

Every kernel computes the declared arithmetic of `docs/models/deepseek-v4.1/attention.md` (host references in
`src/ds41_attention.cpp`) and, for the attention's own F16 projections, the lane-class order of `f16_rows`
(`docs/models/deepseek-v4.1/layer.md`, "lane dot"). Nothing here introduces a new identity: **all outputs are the host references'
bits** (caches byte for byte, scores, selections and the attention output float for float), so the op hashes of the
host forward (`fnv1a` of `o`) are reproduced.

How the order is kept on this device:

- Translation units are compiled with `--fmad=false -prec-div=true -prec-sqrt=true -ftz=false` and the host side with
  `-ffp-contract=off`: no product is contracted into an FMA except the explicit `fmaf` of the declared chains;
  division and square root are IEEE round-to-nearest; denormals are kept. The F16 projections use `__fmul_rn` /
  `__fadd_rn` (the reference's two roundings per term).
- A 512 vector lives in one warp, lane `l` owning dims `16l..16l+15`, exactly as the declared lanes; the butterfly is
  `v = v + __shfl_xor_sync(v, d)` for `d = 16..1` (addition is commutative, so every lane holds the declared sum).
- `exp_f`, the E4M3 / E2M1 / E8M0 codecs, `log2_ceil`, `order_key` and `rope_pair` are the shared host/device helpers
  of `ds41_attention.hpp` / `ds41_scalar_codecs.hpp` (`DS41A_HD` is `__host__ __device__` under `__CUDACC__`).
- RoPE cos/sin come from `rope_row` (double on the host, rounded once), uploaded once as per-position tables of both
  kinds (`RopeTables`); kernels index them with the row's position read on the device.
- F16 to float uses the hardware conversion (exact for every non-NaN half; the weights hold no NaN).
- Integer parts are exact whatever the order: the indexer's E2M1 x E2M1 block dots (`__dp4a` on doubled codes,
  |sum| <= 4608), and the selection (radix histograms and scans of integer counts).

| site | declared order (see docs/models/deepseek-v4.1/attention.md) | CUDA form |
| --- | --- | --- |
| compressor kv / gate, indexer q_b / weights (F16) | lane class `c = i mod 32`: `acc = acc + w * x` ascending, butterfly | a warp per 1 (few rows) or 4 (more rows) consecutive output rows; 256-element segments loaded as 16 B per lane (the next segment in flight) and transposed through the warp's shared buffer so lane `l` takes `l + 32k`; up to 8 rows per pass; all four projections in one launch |
| window write | kv_norm (lane fma chain, butterfly), RoPE, E4M3 / E8M0 per 32 | warp per row |
| compressor | ratio 2 softmax pool, norm, index key = 512-term fmaf chain per output, k_norm, RoPE, FP4; KV entry RoPE + FP4 | block per 8 rows: warps pool and normalise, 256 threads run the 128 x rows index-key chains, each streaming its `wk` row (the next 64 columns in flight), warps finish rows |
| ratio-2 carry | the pending group's first token | in place: the compress kernel reads it, a second kernel writes the new one |
| index query prep | RoPE, E2M1 / E8M0 per 32, `w * 1/64` | warp per (row, head) |
| index scores | exact block dots, `((dot * sq) * sk) * 0.25`, `((t0 + t1) + t2) + t3`, `score = fmaf(relu(d), w, score)` over heads | block of 4 warps per (32 G positions, RT rows), G x RT = 4: the tile's keys decoded once into shared memory; warp (group, row): lane `h` computes `d_h` of the group's 32 positions, then lane `t` runs the head chain of position `t` |
| selection | key desc, position asc; candidate blocks (max, newest pinned) | block per row: 4-pass radix to the k-th key (warp-aggregated histograms), ordered compaction with block scans; keys in shared memory up to 8192 positions; above, two levels: the exact top-k of every 4096-position slice (an element of the row's top-k is in its slice's top-k), then the top-k of the survivors in position order (`SelectView::survivors`, `select_survivor_words`) |
| attention | chunks of 64 keys, tiles of 16, online softmax, chunk combine, sink, inverse RoPE | block = (chunk, 16 heads, row) for many rows, (chunk, 4 heads, row) otherwise; a warp runs 2 heads; raw tile entries double-buffered by asynchronous copies, dequantised once per block into shared memory; the 32 lane partials (2 heads x 16 keys) reduced by a transposed xor butterfly (the additions of the per-key butterfly, item `l` ends in lane `l`); combine kernel per (head, row) |

## Tests (all bitwise)

`strix-cuda-ds41-attention-tests`:
- F16 projections against `f16_rows` (widths 512..4096, 1..13 rows, two jobs per launch);
- window ring and compressor over position sequences in calls of 1..64 rows (ratio 0, 2, 1), with the ratio-2 carry
  checked after every call;
- index scores and selection: plain at 100..32768 positions, candidate source and users at 20008 (2501 blocks > 2048)
  and 131072, small parameters (top-64, 8 candidate blocks), 1..8 rows (many exact-zero scores: ties);
- attention: window only and window + compressed (ratio 2 and 1), 1, 3, 8 and 37 rows;
- the layer sequence of `AttentionState` over layers 0..24 of the released plan (window-only, ratio-2 KV and index
  sources and their consumers, layer 20 as ratio-1 KV / index / candidate source, layer 24 as candidate user), calls of
  1..64 rows (prefill chunks and decode windows), `o`, rings and compressed caches compared after every layer call;
- with `STRIX_DEEPSEEK41_GGUF=<file>`: layers 0, 1, 2 and 20 with the file's weights over real token embeddings (the
  inputs come through the layer's Q8_0 sites; layer 0 is the real data flow, the others an embedding-derived stand-in
  for the residual stream), 1400 positions in prefill chunks of 256 then decode windows, `o` on sampled rows, rings,
  compressed caches and selections after every call.

Timing at prefill shapes: `strix-cuda-ds41-attention-prefill-bench [T] [reps]` (index scores, selection and attention
per layer kind over a T-row prompt in 64-row windows, synthetic caches).
