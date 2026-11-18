# GLM native RTX prefill identity (draft)

`glm-pf-rtx-native-v1` is a hardware-native, numerically changing prefill
candidate for sm_120. It is **not qualified or connected to automatic routing**.
The [explicit full-model producer](../../../src/cuda/models/glm/producer/README.md)
adds a persistent CUDA service, validated compressed cache, opportunistic
expert tiers and the node's transactional wire sender. It is capped at 32K
inputs and has no full-model device or composed-route qualification.
The standalone selector also offers draft `glm-pf-rtx-native-v1-c1`, specified
below. The CUDA implementation covers trellis FFNs, KDA, DSA/indexer and
layer-45 MTP composition as standalone candidates. Node
HIP prefill, decode, launch defaults and identities are unchanged. Compilation,
host fixtures and instruction inspection do not establish native execution,
model fidelity, endpoint quality, or complete-workload qualification.

## Precision and transforms

| Site | Candidate contract | Implementation status |
|---|---|---|
| Trellis projections, dense/shared/routed FFNs | Original mul1 decoded FP16 weights and FP16 input operands; native tensor-core products with FP32 accumulation | FFN standalone implementation |
| Input transform | `RN16(H128(RN16(RN32(x * suh))))`; FP32 butterflies and normalized FP32 scale | Implemented |
| Output transform | `RN32(H128(acc) * svh)`; FP32 butterflies, no intermediate FP16 output cast | Implemented |
| FFN nonlinear | FP32 GLM clamped SwiGLU, then down input transform with its FP16 rounding boundaries | Implemented |
| KDA projections | FP16 decoded operands, FP32 accumulation | Standalone CUTLASS candidate; execution pending |
| KDA convolution, recurrence, gates, norms | FP32 sensitive arithmetic/state; 32-token mixed-precision triangular solve and FP32 inter-chunk carry | Standalone `kda32` candidate and sequential FP32 reference; execution pending |
| MLA/DSA and indexer products | FP32 index projection/dot chains; FP16 attention projections and absorption; BF16 sparse score/PV operands; FP32 online softmax and fixed-order split combine | Standalone `dsa64-f32idx` candidate and FP32 reference; execution pending |
| Router | FP32 logits, sigmoid, bias, lower-expert-ID tie policy, top-8 selection and normalization | FP32 router implemented for MTP; external routes accepted by standalone FFNs |
| Residuals and exported state | FP32 residual/KDA state; DSA latent/index key/gate BF16 and pooled keys FP32 | MTP residuals and node-layout DSA export implemented; device gates pending |
| MTP | Same site-specific precision map, including layer 45 and `eh_proj` | Standalone `mtp45` composition and FP32 reference; real-asset/device gates pending |

The transform is the normalized Walsh-Hadamard H128. For each block, butterfly
strides are 1, 2, 4, 8, 16 across lanes, then 32 and 64 across registers,
followed by multiplication by FP32 `0x1.6a09e6p-4`. Each butterfly uses separate
FP32 additions/subtractions; scaling uses separate FP32 multiplies. Retain signed
scales, permutations, and rounding sites. Do not absorb `suh`/`svh` into weights:
that would change the input rounding contract. Output transforms need a complete
H128 block before SwiGLU.

## Decode and native FFN schedule

The implementation lives in [src/cuda/models/glm](../../../src/cuda/models/glm/).
The codebook, packed reader, fragment permutation, transforms and original
provenance derive from
[glm_trellis_device.hpp](../../../src/glm/moe/glm_trellis_device.hpp).
The port retains the mul1 marker `0x83dcd12d`, unsigned 32-bit product, sum of
its four bytes, and the half fused operation with half constants `0x1eee` and
`0xc931`. An equivalent exact dyadic numerator is rounded once to FP16.
No other codebooks, bias conventions or external scale tensors are admitted.

K3/K4/K5/K6 matrices use `[input/16, output/16, 16*bits]` little-endian 16-bit
storage. Each tile is a circular MSB-first 32-bit-word stream. State `p` is the
16 bits ending at `(p+1)*bits`, modulo `256*bits`. For `lane=p/8`, `f=p%8`, the
logical decoded coordinate is
`i=2*(lane%4)+f%2+8*((f/2)%2)`, `o=lane/4+8*(f/4)`.
Decoded weights have row-major `[input, output]` layout.

Expert groups are consecutive IDs `[16*g, min(16*g+16,E))`, processed in
ascending group order. Decode all three matrices **once per group per layer**,
then process all assigned rows before reusing the buffer. No per-row-tile
decoding or fully decoded layer cache is used. One buffer holds 16 complete
gate/up/down experts: 805,306,368 bytes for full 4096/2048 dimensions. The v1
comparison arm allocates one; C1 allocates two within 2 GB. Both decode empty experts within
the group; timings include this cost.

Dispatch is a GPU histogram, ascending-expert exclusive prefix sum, and atomic
scatter. Row placement in the dispatch buffer may vary; each entry carries its
original token/router-slot index. It never controls an arithmetic reduction.
Inputs require eight distinct valid IDs per token and finite route weights;
the standalone loader validates these before upload. Router slot order is
preserved, including for the weighted merge.

Each persistent kernel uses 256 CTAs, 256 threads/CTA, eight warps, and a
16-row by 128-column tile. Each warp owns one 16x16 output fragment. K advances
in increasing 16-element tiles; there is **one split, no split-K**. Native
`wmma::mma_sync` computes FP16 x FP16 -> FP32 chains from zero. Padded rows
are zero; all admitted widths are multiples of H128. Gate/up use separate
transformed operands (their `suh` can differ), paired accumulators, and a fused
H128/scale/SwiGLU epilogue. Down uses the same MMA kernel family and H128/scale
epilogue. Dense/shared FFNs use the same dispatch and kernels with one expert,
one row assignment and unit merge weight.

SwiGLU is `(g / (1 + declared_exp(-g))) * u`, with `g=min(g,10)` and
`u=max(-10,min(u,10))`. `declared_exp` in `trellis.cuh` fixes range reduction,
degree-7 polynomial, operation order and saturation. It is compiled without
contraction or fast math, with subnormals retained and precise division/sqrt.
This preserves the declared polynomial and clamp policy; it makes no claim of
whole-FFN equality with the node, whose accumulation schedule differs.

Contributions are stored by original `[row,router-slot,channel]`. Merge starts
at positive FP32 zero and, for slots 0 through 7, executes a rounded FP32
multiply by route weight followed by rounded FP32 add. No output atomic adds,
cross-expert tree or fused multiply-add is permitted. Dense/shared merge has
one unit-weight term. CTA task-claim order does not affect this order.

Workspace is explicitly bounded by rows <= 32768, experts <= 288, and aligned
H128 widths <= 16384. For `R=8*C`, hidden H and intermediate I, its dominant
bytes are `16*H*I*6 + R*(8*H+6*I)`, plus dispatch/alignment metadata. At full
GLM expert dimensions this is about 1.17 / 3.76 / 12.62 GB at C=1K/8K/32K.
Compressed matrices, input/output rows, CUDA context and events are additional;
this is not an allocation receipt for an entire producer. No allocations or
host synchronizations occur inside the enqueue API.

Changing operand types, rounding sites, transform ordering, polynomial,
clamps, MMA fragment/tile shape, K tile traversal, split count, accumulator
topology or weighted-merge order requires a **new identity ID** and fresh
qualification. So does replacing the planned chronological FP32 KDA recurrence
with blocked FP16/BF16 products. A different executable/compiler is always a
new hashed bundle requiring validation; an arithmetic-changing compiler output
requires a new identity. Scheduling-only graph capture, dispatch row permutation
and buffer overlap may retain the ID only if they preserve this contract.
Unimplemented attention/state/import schedules must be specified and frozen
before this draft can be admitted as a complete producer identity.

## C1 grouped GEMM candidate (draft)

Select `--path cutlass` in the standalone benchmark for
`glm-pf-rtx-native-v1-c1`; `--path legacy` retains v1 (the executable default).
This changes the accumulator schedule and is a separate numerical identity.
It has compiled for sm_120 and passed host fixtures and static instruction
checks. Native execution, real-weight accuracy, model fidelity and complete
workload qualification remain pending. No throughput is established.

Gate and up are independent grouped FP16 x FP16 -> FP32 GEMMs with distinct
transformed input operands. Down uses the same GEMM family. The adopted
[CUTLASS 4.2.1](https://github.com/NVIDIA/cutlass/tree/v4.2.1) headers (BSD-3-Clause;
fetched at configure time by the producer build, not vendored)
instantiate Sm80 TensorOp templates compiled for sm_120: 128x128x32 with four
stages, or 32x128x32 with three stages for experts with at most 32 assigned rows.
Both use 16x8x16 tensor instructions, eight-element vector copies, multistage
`cp.async`, and device-only persistent grouped scheduling without split-K.
CTA counts come from the actual kernel residency and device SM count during
setup, rather than a fixed grid. Source and SASS checks require HMMA and
LDGSTS inside each variant's mainloop back edge.

The histogram/prefix/scatter remain on the GPU. A GPU kernel builds two fixed
descriptor lists from the histogram offsets, with zero-M descriptors for empty
experts and the other tile class. Each gate/up list has two problems per expert;
down has one. Rows keep their exact M; CUTLASS predicates tails. A is gathered
row-major `[assigned_rows,K]`; B is decoded directly as row-major `[K,N]`.
Widths are multiples of 128, all arena allocations are aligned to 256 bytes,
and every expert offset and row stride is divisible by the 16-byte vector
alignment. No global M padding or padded activation reads are necessary.

GEMMs write FP32 row-major scratch. Separate warp/register H128 kernels apply
the output scales and SwiGLU. The gate/up epilogue also applies down `suh` and
the first RN16 boundary. A separately timed down-input H128 kernel completes
the final RN16 boundary. These are the same transform/scaling/nonlinear
operations as v1; FP32 scratch adds no extra rounding. Down's output H128 and
scales precede the unchanged deterministic router-slot merge. C1 accumulation
order is defined by the pinned CUTLASS instantiations and hashed executable,
and is not claimed bit-equal to v1 or chronological scalar FP32 FMA.

Each decoder CTA processes eight packed tiles and writes directly into B.
Two 16-expert buffers alternate. A caller-owned nonblocking stream decodes
window g+1 while the compute stream consumes window g; ready events protect
consumers, and retired events after down GEMM protect parity-buffer reuse at
g+2. All decode work is joined into the compute stream before enqueue
completion. The enqueue API allocates nothing and never synchronizes the host;
the caller owns streams, events, buffers and their lifetime. Reusing a context
or workspace concurrently from independent compute streams is unsupported.

C1's workspace adds a second decoded window and two FP32 scratch arenas to
the v1 allocation: `16*H*I*6 + R*(4*max(H,I)+4*I)` additional dominant bytes.
It retains the reference-compatible activation arena. Full 288-expert,
4096/2048, 32K top-8 workspace is approximately 19.86 GB, excluding compressed
assets, inputs, outputs, events and CUDA resources. This is a bound on this
standalone workspace, not complete producer capacity qualification.

CUDA events separate input transforms, GPU problem-list construction, gate/up
GEMMs, H128/scales/SwiGLU/down rounding, down transform, down GEMM, output
epilogue and merge. Decode duration is measured on its own stream. The compute
wait interval is reported separately; exposed decode is estimated by clipping
decode completion after compute-window entry to the decode duration, with the
remainder labeled hidden. Event scheduling overhead and GPU contention remain
included; these estimates do not measure uncontended decoder throughput.
Grouped useful TMAC/s includes transforms and epilogues; GEMM-only useful
TMAC/s excludes them; full useful TMAC/s uses the full layer median. Work is
always `C*topk*3*H*I`, excluding padded arithmetic. Phase means are not additive
with the full median or with overlapping decode durations.

Accuracy diagnostics retain every output against the chronological scalar
FP32 FMA implementation and an independent CPU complete-first-row reference.
Errors and ULP histograms are diagnostic evidence, without an invented passing
tolerance. Repeated-output byte equality is a separate deterministic-merge
check, not equivalence to the FP32 reference. No GPU accuracy gate has passed.

## KDA32 attention sublayer candidate (draft)

`glm-pf-rtx-native-v1-kda32` freezes a standalone full-head KDA sublayer.
It consumes input-normalized FP32 hidden rows and produces the attention output
projection, FP32 S `[heads,128,128]` and chronological pre-convolution carry
`[3*heads*128,3]`. Hyperconnections, residuals, input normalization, MoE and
node import are outside this API. It does not select a production route.
Native execution, real-weight error measurements, endpoint statistics and
complete-workload qualification remain pending.

The qkv and output trellis matrices are decoded once each per layer invocation
into one reused FP16 arena. Their transforms preserve the rounding boundaries
above. Ordinary F16 f/g low-rank and beta projections round inputs to FP16;
there is no activation between f_a/f_b or g_a/g_b. Both projection layouts
reuse the pinned CUTLASS hot pipeline: 128x128x32, four stages, 16x8x16
TensorOp, FP32 accumulation, no split-K, one GPU problem descriptor and a
fixed 256-CTA device scheduler. `prepare_kda()` configures the shared-memory
attributes before timing. Enqueue does no allocation or device discovery.

Convolution has four chronological taps, SiLU, per-head q/k L2 normalization
and q scaling by 1/sqrt(128). Vector log-decay is
`g[t,i] = -5 * sigmoid(exp(A_log[h]) * (f[t,i] + dt_bias[i]))`;
beta is sigmoid of the beta projection. Output uses per-head RMS normalization,
BF16 o_norm and sigmoid of the output gate. Sensitive operations and the
128-lane norm reduction tree use FP32; nonlinear exp uses the declared
polynomial, with precise division/sqrt and no contraction/fast math.

For a chunk, define `D(t,s)` as the diagonal product of decays after s through t
and `D(t,-1)` from chunk entry through t. The exact real-arithmetic equations
implemented by the blocked schedule are:

```
K[t,s] = k_t^T D(t,s) k_s                         (s < t)
Q[t,s] = q_t^T D(t,s) k_s                         (s <= t)
delta_t = beta_t * (v_t - k_t^T D(t,-1) S_in
                       - sum(s<t, K[t,s] * delta_s))
o_t = q_t^T D(t,-1) S_in + sum(s<=t, Q[t,s] * delta_s)
S_out = D(end,-1) S_in + sum(s<=end, D(end,s) k_s delta_s^T)
```

This retains vector decay and the state-dependent delta update. Each chunk has
32 tokens, with zero-padded 16-token MMA tails. Local FP32 log-decay prefixes
are subtracted only in causal order; exp sees nonpositive differences.
There are no inverse prefixes, division by tiny cumulative decays, or scalar
decay approximations. The exp-of-prefix formulation is an arithmetic change
from the sequential reference's individual decay products.

One-warp 16x16x16 WMMA products compute triangular K/Q coefficients, incoming
state products, output products and final state increments. Weighted keys,
queries, incoming state, Q coefficients and solved deltas are rounded to FP16
only at these block operand boundaries; MMA accumulates FP32. The strictly
lower triangular delta solve, decay prefixes, output addition and final
persistent-state update use FP32. Native call boundaries define the chunk
partition; partitioning the same prefix into separate invocations changes
rounding and requires its own qualification. No chronological bit-equality
claim is made.

The independent SIMT arm uses ascending-key FP32 FMA projections and a
chronological FP32 recurrence, without MMA or state operand rounding.
Ordinary reference projections consume the original FP32 inputs. Both arms
retain trellis transform rounding, the row-parallel preparation/output code
and the exact same raw weight tensors. Reference recurrence launches contain
at most 32 chronological rows and persist FP32 state between launches.
Double-precision host fixtures independently verify the algebra, vector gates,
nonzero state, causality, zero beta and partial convolution carry; they cannot
validate native MMA numerical behavior.

The benchmark reports full-layer device/wall time including decoding, useful
projection MAC/s and useful layer MAC/s (projections plus three 128x128
recurrence products per head/token). It compares every output and final S/conv
element against the complete FP32 layer reference, and separately checks
native repeatability. Inputs are explicitly labeled synthetic normalized rows;
weights are exported byte-for-byte from a supplied model. No performance or
quality threshold is inferred from compiler or host gates. The DSA/indexer and layer-45 MTP composition candidates are described below.
Graphs and endpoint statistical qualification remain unimplemented.

## DSA64 and MTP45 candidates (draft)

`dsa.cuh` exposes a full 64-head, single-sequence DSA sublayer, with normalized
FP32 hidden input and FP32 projected output. It shares the KDA CUTLASS trellis
projection implementation. Each compressed matrix is decoded once per call;
64-row windows bound query, index and reference attention workspace. Native
attention projection inputs and head-wise absorption operands round to FP16,
with FP32 accumulation. The default identity is
`glm-pf-rtx-native-v1-dsa64-f32idx`. Every index dependency uses the scalar
FP32 reference order: key/gate and head-weight projections widen the stored
F16 weights and multiply FP32 activations without input rounding; query-A
and index-query trellis projections retain their declared transform/RN16
boundaries but accumulate decoded products with ascending-k FP32 FMA.
Key normalization, BF16 key/gate storage and four-token FP32 pooling retain
the reference contract. Each index dot is an ascending-k FP32 FMA chain from
positive zero, followed by RN32 scaling by `128^-1/2`, ReLU, and ascending-head
FP32 FMA with the head weight scaled by `32^-1/2`. Changing only the final
dot kernel would leave rounded upstream inputs and pool-selection drift.

`DsaCall::mma_index=true` (benchmark `--mma-index`, launcher `-MmaIndex`)
explicitly selects `glm-pf-rtx-native-v1-dsa64-mmaidx`: FP16 ordinary/index
projection inputs, TensorOp projection accumulation and BF16 index dots.
This variant has its own identity and selection diagnostics. It cannot inherit
the FP32 index contract. The radix selector preserves lower-pool-ID ties,
ascending token order, 512 complete four-token pools and at most three trailing
tokens from the unfinished pool. It masks pools independently at every query.

Sparse attention splits the selected list into fixed 256-key chunks. One CTA
per query/chunk gathers 32 latent rows once for all 64 heads with 16-byte
`cp.async` copies, zero-filling masked keys. Shared latent/probability rows
are padded to 520/40 elements to spread MMA operand accesses over banks;
scores are stored key-major with a 68-element stride for bank-spread
softmax reads. Eight warps
compute all heads'
QK products using BF16 HMMA and direct column-major latent views, avoiding
per-K-step shared operand repacking. Sixteen warps compute PV using BF16
HMMA and retain FP32 output fragments in registers across the chunk's online
softmax loop. Maxima, denominators, probability evaluation and output rescaling
are FP32; each head uses eight lanes, four keys per lane, and fixed XOR
max/sum trees with offsets 4, 2, 1. Unnormalized PV probabilities round to BF16. A separate combine
kernel merges partial maxima, denominators and outputs in ascending chunk
order using FP32 arithmetic. No floating-point atomics participate. Partial
attention workspace is bounded by 64 queries times nine partial chunks, independent of prefix
length; no full-prefix attention score matrix or expanded K/V is created.

The previous organization serialized QK on one warp, repacked every 16-wide
K slice through shared memory, gathered each latent tile for four independent
head groups, and repeatedly stored/reloaded the output accumulator through
shared memory. Merely emitting HMMA did not establish efficient execution.
The replacement removes those serialization and traffic patterns; its
performance and model fidelity require device measurements. The attention
throughput objective is at least 40 useful TMAC/s at 8192 rows, not a measured
or qualified result.

Head-wise value absorption uses FP16 MMA, followed by the trellis output
projection. The independent scalar FP32 reference preserves trellis transform
rounding and BF16 state storage, uses scalar FMA dot chains and chronological
FP32 online softmax, and does not round attention/absorption/PV operands to
native precision. Output errors remain diagnostics, even when selections match.

Export buffers have the node layout: BF16 latent `[capacity,512]`, index key
and gate `[capacity,128]`, and FP32 complete pooled keys `[capacity/4,128]`.
`enqueue_dsa_export` concatenates latent, index key, index gate and complete
pools, matching the node snapshot order. Every key/gate row survives, including
the unfinished pool. A nonzero `first`
requires a valid preceding history. Only complete pools are scored; no future
pool or future token can be selected. Export the used prefix extents rather
than capacity padding. These are layout contracts, not claims of bit equality
between the new numerical identity and node arithmetic.

`mtp.cuh` declares `glm-pf-rtx-native-v1-mtp45-f32idx` and composes layer 45:
normalized next-token embedding and trunk hidden rows, concatenation in that order, native `eh_proj`, input norm, DSA, residual,
post norm, FP32 router with lower-ID ties, grouped native routed/shared MoE,
residual and shared-head norm. The caller supplies embedding rows for the
matching next-token IDs. Full expert and shared matrices are required; no
vocabulary head or trunk hyperconnection is part of this interface. The
reference uses scalar FP32 attention, router and expert paths.

The standalone DSA benchmark exports complete output/state errors and compares
selected pool membership against the full reference. A differing membership is
near a tie only when its reference score is within
`0.001 + 0.001 * max(abs(score), abs(reference_cutoff))` of the reference's
512th selected score. It reports differing rows, memberships within tolerance,
and memberships outside tolerance. Any mismatch outside tolerance, wrong
cardinality or trailing-token mismatch returns selection status 3 after the
full diagnostics. The bounded launcher records this status and continues to
the next length, then returns 3 after all lengths. Other errors, including
nonfinite results, fail immediately. Output/state errors remain diagnostics pending endpoint qualification. Compiler, host and
instruction gates cannot establish device accuracy, latency or MTP fidelity.

## DSA index arithmetic ablation (v2 drafts)

The standalone `--index-variant` selector offers the following frozen candidate
identities. All use the same raw weights, synthetic input seed, causal pool
rules, 512-pool selector and near-tie diagnostics described above. They are
benchmark candidates, with no change to production resolution.

| Selector | Identity suffix after `glm-pf-rtx-native-` | Index projections | Index scores |
|---|---|---|---|
| `fp32` | `v1-dsa64-f32idx` | Ascending-k scalar FP32 | Ascending-k scalar FP32 |
| `mma` | `v1-dsa64-mmaidx` | FP16 operands, CUTLASS FP32 accumulation | BF16 MMA |
| `mma-proj-fp32-score` | `v2-dsa64-mmap-f32s` | As `mma` | Scalar FP32 |
| `fp32-proj-mma-score` | `v2-dsa64-f32p-mmas` | As `fp32` | BF16 MMA |
| `tf32x3` | `v2-dsa64-tf32x3p-bf16x3s` | CUTLASS FastF32 (three TF32 products) | Three BF16 split products, FP32 accumulation |

The v1 FP32 candidate shares its scalar FP32 query-A with attention. This is
an intentional historical contract but also a projection fallback: key, gate,
query-A, index-query and head-weight execute scalar projection kernels. KV-A,
query-B, value/output projections and absorption retain their native paths.
The v2 candidates separate the index query-A branch when it requires different
arithmetic. Attention query-A, query-B, KV-A and output projections always use
the pinned CUTLASS FP16 path in native v2 execution. The independent full-layer
reference still uses scalar FP32 for every projection. Historical v1 and MTP
identities retain their previous arithmetic.

For `tf32x3`, widen the decoded query-A/index-query trellis operands and
weights exactly from FP16 to FP32; preserve input H128/RN16 and output H128
boundaries. Ordinary key/gate/head-weight projections consume original FP32
input and exactly widened stored FP16 weights. All five index projections use
CUTLASS 4.2.1 `OpMultiplyAddFastF32`: row-major A, row-major trellis B or
column-major ordinary B, 64x64x16 CTA, 32x32x16 warp, 16x8x8 instructions,
three pipeline stages, no split-K. Its pinned high/residual conversions use
round-toward-zero and half-ULP truncation, respectively. FastF32 accumulates
three TF32 cross products and omits residual times residual. Widened weight
preparation occurs once per invocation and is included in decode timing.

The score kernel splits each FP32 operand `x` into
`hi=RN_BF16(x)` and `lo=RN_BF16(RN32(x-hi))`. For every ascending K16 tile,
it accumulates `loA*hiB`, `hiA*loB`, then `hiA*hiB` with BF16 WMMA and FP32
accumulation. Residual times residual is omitted. Two ascending 16-head blocks
cover all 32 heads; scale, ReLU and ascending-head FP32 FMA folding retain the
reference formula. This is near-FP32 arithmetic, with its own reduction and
rounding contract; it does not promise scalar-FP32 equality.

One 256-thread CTA handles a query, eight warps score 16-pool tiles, and future
or incomplete pools never enter scoring. Fully masked tiles are skipped.
Scale/ReLU/head folding and streaming top-512 selection are fused: shared
storage retains at most 512 survivors and 512 incoming pool keys/IDs. Four
radix passes find the threshold. Stable ballot/prefix compaction preserves
ascending pool IDs, lower-ID cutoff ties and chronological output tokens.
Scores are never written to global memory on this path. Other variants retain
the bounded 64-query score window and the existing radix selector. For the
fused candidate, `index_scores` timing includes selection; `select` timing
contains diagnostic export only. Phase comparisons must respect this scope.

`tf32x3` is the provisional candidate to measure, rather than an accuracy- or
latency-selected winner. The objectives are full DSA layer <=60 ms at 8192
rows and <=300 ms at 32768 rows, with no selection mismatches outside the
stated near-tie policy. Native latency, 8K/32K mismatch counts, real model
fidelity and complete-workload qualification remain **NOT RUN** for these v2
identities. Host algebra/selection fixtures and tensor instruction inspection
cannot establish these gates.

The bounded `-VariantSweep` launcher runs all five variants at both 8192 and
32768 rows, plus the provisional candidate at 1024 rows, with one aggregate
570-second deadline and at most five seconds to kill a timed-out child. Logs
and result records carry the variant and identity, and differing selection
counts are emitted by every benchmark. A status-3 mismatch does not stop later
variants; other errors stop the window. No GPU execution occurs in PlanOnly,
build, disassembly, host tests or launcher fixture tests.

## Route and receipt contract (not yet wired)

Proposed requested routes are `auto`, `pair`, and `glm-pf-rtx-native-v1`.
The documented future OFF switch is **`STRIX_GLM_RTX_PREFILL=0`**, equivalent
to pair-only exact prefill. This milestone does not read this variable or alter
the existing resolver. Automatic admission eventually requires an idle producer,
a capacity-qualified profile, a passing statistical receipt for the prompt
length and a predicted endpoint advantage including hand-off. Initially cover
8K and 32K independently; 1K remains pair-local absent endpoint evidence.

Receipts/responses must record:

- Requested/effective route, producer identity, node prefill identity, node
  decode identity, and composed identity; preserve truthful fallback reasons.
- Model asset hashes, tokenizer/template/stop contract, executable bundle hashes
  for producer/orchestrator/both workers, compiler flags and exact revisions.
- Qualification receipt hash/scope, qualification plan and manifest
  hashes, control result, endpoint confidence bounds and per-length verdicts.
- Capacity/cache profile, all allocations including MTP, raw/wire byte totals,
  cache hit/miss and startup/transfer/decode/compute/import/suffix phase timings.
- Frontier `P=N-1`, prefix `[0,P)`, matching MTP coverage, state format/hash,
  begin/frame/commit/abort status, both-rank commit and final suffix status.

Only the composed node-head endpoint can qualify the route statistically.
Native FFN error statistics alone cannot qualify it. Separate exact transport
and same-state plain/speculative gates remain required. Nodes execute the
final prompt row, vocabulary head and MTP seed under their existing identity.
Failure before two-rank commit must discard partial state before pair replay.
Missing data is `NO-REFERENCE`, never an exactness or quality pass.

## Standalone verification

See the [benchmark instructions](../../../src/cuda/models/glm/bench/README.md) for
CPU fixtures, offline compilation/disassembly and a bounded GPU window.
Host tests exhaust all 65,536 mul1 states independently for every admitted rate,
compare packed decoding against node instruction emulation and a scalar bit
reader, verify the fragment bijection, and check transform/scaling boundaries.
They do not execute a GPU or validate tensor-core numerical behavior.


### Paired-half index diagnostic (draft)

The DSA bench additionally evaluates a `node-mimic` index-only reference.
It preserves the split four-lane paired-half trellis projection graph for
q-A and index q-B: FP16 transformed operands and decoded weights, FP32
accumulators, stride-2 then stride-1 lane reduction, ascending split sum,
output H128 and FP16 output scales widened to FP32. Ordinary key, gate and
head-weight projections retain original FP32 inputs with ascending FP32
FMA chains. Index scores use FP32 operands and ascending dimension/head
FMA chains; FP16 scoring is not this reference contract.

The paired operation uses exact binary16 products in binary64 with a
compensated sum and an FP32 result. This is an IEEE approximation; it does
not establish hardware DOT2 truncation equivalence. RMSNorm and key
LayerNorm use the declared FP64-sqrt-to-FP32 boundary in this diagnostic.
BF16 key/gate rounding, four-token channelwise softmax pooling, scale,
ReLU, canonical ties and causal selection follow the index contract.
The index-only arm leaves the layer output and latent state untouched.

Each benchmark invocation emits `selection_pair=variant_vs_node_mimic`
and `selection_pair=node_mimic_vs_scalar_fp32`, including exact symmetric
membership differences and the existing separately counted near ties.
Both comparisons use the model's score cutoff. A provisional agreement
gate requires variant disagreement no greater than model-vs-scalar
for both exact and outside-tolerance memberships. It is not a model
qualification gate; route-vs-node complete-workload qualification remains
required. Missing pair diagnostics are `NO-REFERENCE` and fail the launcher.

The TF32x3/BF16x3 score candidate stages a two-query by 32-pool tile,
reuses hi/residual operands, skips wholly future tiles and fuses scale,
ReLU and head folding before exporting bounded folded scores. A separate
streaming top-512 merge retains stable lower-ID ties and chronological
tokens. Per-head dots remain CTA-local; the workspace never stores the
whole sequence score matrix. Selection time is included in the
`index_scores` phase for this candidate. Compilation and host checks do
not establish selection fidelity or the 300 ms full-layer target at 32K.
All five variants' 8K/32K device measurements are pending.
