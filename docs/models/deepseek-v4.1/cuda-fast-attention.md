# deepseek41 CUDA fast prefill attention

A faster, non-bitwise form of the CSA2 attention and indexer for the CUDA prefill, behind `--fast-attention`
(`CudaPrefillOptions::fast_attention`, default off). The exact kernels (`docs/models/deepseek-v4.1/cuda-attention.md`) stay the reference
arm and the fallback. Header `include/strix/cuda/ds41_fast_attention.hpp`; kernels `src/cuda/models/ds41/ds41_fast_attention.cu`
(attention, FP16 entry rows) and `src/cuda/models/ds41/ds41_fast_index.cu` (FP8 index scores); test and timing
`tests/cuda/ds41_fast_attention_tests.cu` (`strix-cuda-ds41-fast-attention-tests [T] [reps] [--device N]`).

## What changes against the exact path

| part | exact | fast |
| --- | --- | --- |
| cache entries | decoded per tile, per block | decoded once per layer into FP16 rows of 512 (`enqueue_fast_kv`; exact values: E4M3 x 2^e and E2M1 x E4M3 fit FP16, window scales outside 2^-14 .. 2^15 rounded once) |
| attention products | FP32 fmaf chains in the declared lane order | `mma.sync` m16n8k16, FP16 inputs (queries rounded to FP16 after RoPE and a power-of-two per head), FP32 accumulation |
| softmax | declared chunks of 64, tiles of 16, `exp_f`, chunk combine | base-2 online softmax over tiles of 32 keys (`ex2.approx`), P rounded to FP16, the sink added at the end |
| output | FP32 o, then `quantize_a8` | inverse RoPE and `quantize_a8` of o in the epilogue (A8 rows of o written directly; FP32 o optional) |
| index scores | integer block dots (`__dp4a`) and the declared float terms | `mma.sync` m16n8k32 E4M3 x E4M3, FP32 accumulation: each vector's E2M1 values as E4M3 relative to its largest block scale (exact unless a block is more than 2^8 below it), the factors folded into the head weight and the position's score |
| calls | index, selection and attention per 64-row window | per layer the FP16 rows and FP8 index keys; per chunk (1024 rows) one index-score launch, one selection, one attention launch |

The selection itself (`enqueue_select`: exact top-k, candidate pool) is unchanged; it runs on the fast scores, so
near-ties may select differently.

## Kernels

- Attention: a block of 16 warps per (row, 32 heads). The query A fragments (16 heads x 128 dims per warp) stay in
  registers; tiles of 32 key rows are copied by `cp.async` into two swizzled shared buffers (the next tile lands during
  the current one); S partials of four 128-dim slices meet in shared memory, the online softmax runs a warp per two
  heads, PV by `ldmatrix.trans`. About 87 KB of shared memory (one block per SM on sm_89 and sm_120).
- Index scores: a block of 8 warps per (16 rows, 1024 positions); each warp holds the query B fragments of two rows,
  key tiles of 64 positions by `cp.async`; relu, head weights and the head sum in registers.
- Only `mma.sync`, `ldmatrix` and `cp.async`: builds for sm_89 and sm_120. FP8 `mma.sync` needs compute capability
  8.9 (`fast_index_supported`); below it the driver keeps the exact indexer with the fast attention.

## Driver

`--fast-attention` adds per layer an op `fast_kv` (FP16 window rows from the first output row's window, the KV
source's FP16 entries and FP8 index keys), and per chunk the ops `index` (scores + selection) and `attention` in
place of `index_attention`; `o_a` reads the A8 rows the attention wrote. `--exact-index` keeps the exact indexer per
window with the fast attention (to separate the two in the quality gate).
