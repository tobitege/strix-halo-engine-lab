# The CUDA backend

All CUDA code lives in one tree with the same layering as the HIP side: model-agnostic host layers (graph, serve,
session, spec, xport, policies, identities) are shared by both backends; each vendor backend implements the same
core interfaces (dense, grouped MoE, attention, norms, weight decode); architecture specialisations sit inside the
backend and are chosen by capability, not by model.

| Path | Holds |
|---|---|
| `src/cuda/core/` | model-agnostic device helpers and kernels |
| `src/cuda/arch/` | capability queries (`capability.hpp`); `arch/sm120/` Blackwell-only paths (none on main yet) |
| `src/cuda/models/ds41`, `qwen`, `glm`, `minimax` | what is still model-specific |
| `include/strix/cuda/` | public headers (`strix/cuda/<name>.hpp`, formerly `strix/cuda_<name>.hpp`) |
| `cmake/StrixCudaBackend.cmake` | `strix_cuda_core` (interface target linked by the CUDA model libraries), core source lists |

Entry points stay in `apps/cuda_*` and tests in `tests/cuda/`; the GLM producer keeps its own CMake project
(`src/cuda/models/glm/producer`) and the MiniMax prefill its build scripts (`src/cuda/models/minimax/*.bat`).
`src/glm/rtxpf/import` (the receiving side of the GLM state import) is host code and stays with GLM.

Session memory is planned by the shared host policy `include/strix/serve/session_reservation.hpp`: each request
reserves prompt + max_tokens + the engine's slack (rounded to a granule, capped at the context), and admission
reuses, builds, re-sizes or releases sessions from the free memory. The CUDA Qwen server binds it through
`apps/cuda_qwen_serve_policy.hpp` (`--session-reserve request|max`, `--session-granule N`). Its serve bench kit is
`scripts/bench/cuda/qwen_serve_arms.sh` (strict arm order, one card) with `qwen_serve_bench.py` (cold and warm
prefill, C1 references, concurrency arms with `same_as_C1`).

## Phase 1 (layout, bitwise)

Files were moved, not rewritten. The SASS gate is `scripts/bench/cuda/sass_compare.py
--base <old build> --new <new build>`. It compares architecture/function pairs, requires nonzero CUDA coverage,
and rejects missing artifacts, disassembly failures and symbol collisions. Artifact moves require explicit
relative-path `--rename OLD=NEW` entries; helper or anonymous-namespace symbol changes require exact
`--symbol-rename OLD=NEW` entries, also applied to call targets. Earlier totals from the lossy symbol gate do
not establish this gate; native builds must be checked again. Identical helpers that several files carried
are defined once:

| Helper | Was in | Now |
|---|---|---|
| `cp_async16`, `cp_async4`, `cp_async_commit`, `cp_async_wait<N>` (no clobber) | ds41 `dense_fast`, `prefill_q8`, `prefill_kernels` | `core/ptx.cuh` `core::ptx` |
| `mma_s8` (m16n8k32 s8, `uint4` A) | ds41 `core_mma`, `dense_fast`, `prefill_q8` | `core::ptx` |
| `smem_addr`, `ldsm_x4` | ds41 `fast_attention`, `fast_index` | `core::ptx` |
| `cp_async16` ("memory" clobber) | ds41 `moe_v2`, `router_v2` | `core::ptx_mem` |
| `cp_async16` zero-fill, `cp_async_commit`, `cp_async_wait_all` ("memory") | ds41 `fast_attention`, `fast_index` | `core::ptx_mem` |
| `butterfly` (warp sum, fixed order) | `ds41_cuda_common.cuh`, `ds41_moe_device.cuh` | `core/reduce.cuh` |
| `better` (top-k order) | ds41 `core_moe`, `prefill_kernels`, `router_v2` | `core/reduce.cuh` |
| `block_exclusive_scan` | ds41 `moe`, `parts_host` | `core/reduce.cuh` |
| `dot_float` (exact int to float) | ds41 `dense_fast`, `prefill_q8` | `core/reduce.cuh` |
| compute capability >= 9 cache (PDL, clusters) | ds41 `core_mma`, `core_norm` | `arch/capability.hpp` |

Same job, not the same text (left in place, to be settled with a SASS check when their core is built): the warp
sums `warp_sum` of ds41 `attention` / `fast_attention` and Qwen `moe`, `butterfly` of ds41 `dspark` (no unroll
pragma, explicit width), the BF16 conversions of Qwen `gdn` / `hc` / `qsa` / `plemtp` / `mtp_draft` and MiniMax
(some with a NaN case, some without), the per-file `check()` of five Qwen files (different message prefixes),
`cp_async_wait_all` of ds41 `moe_v2` (`cp.async.wait_all`, not `wait_group 0`), GLM `h128` (`native.cu` vs
`projection.inl`).

## Phase 2 plan: shared CUDA cores behind the HIP interfaces

Each step extracts one family into `src/cuda/core/<family>/` behind the interface the HIP core already exposes,
keeps the current entry points as thin wrappers (old signatures), and lands only with a byte-for-byte gate on the
operator, its partials and the logits / KV state (1 and 8 rows, 8K and 32K, repeated and disjoint routes, graph
capture and replay). Numerical identity is part of the plan (FMA or not, FTZ, div/sqrt, AMD tables, accumulator
type, top-k ties, split-K, rounding): a core selects only between variants of the same identity, by capability and
shape. Order:

1. **Dense IMMA tile (extracted; native qualification pending)** (`core/dense/q8_a8_imma.cu`): tiled Q8_0 x A8 kernels, repack and tuning dispatch
   share `include/strix/cuda/core/{types,plan,dense}.hpp`; old DeepSeek signatures remain field-by-field adapters.
   `include/strix/device/runtime.hpp` provides a capability-query binding, but the execution path does not consume it;
   plan SM fields are metadata checks, not device admission. Workspace and stream stay caller-owned.
   This intermediate extraction still uses model helpers, a fixed HC epilogue and process-global tuning.
   Shared helpers, an explicit epilogue policy and immutable owner-level tuning remain follow-up work before concurrent multi-engine use.
   The extraction gate compares every built CUDA object's and executable's SASS against the base build, moved
   symbols declared one by one with `--symbol-rename OLD=NEW` (an undeclared namespace move is a difference);
   this is an instruction-preserving move, with no new numerical or workload qualification claim.
2. **Common numerics and norms** (`core/norms/`): `ds41_cuda_common.{cuh,cu}` (device, workspace, stream, event
   helpers become `core/runtime` pieces), `ds41_core_norm.cu`, Qwen `hc` with the AMD-identical math
   (`include/strix/cuda/amd_math.cuh` is already in core), MiniMax Gemma norms; one interface, separate arithmetic
   families.
3. **Remaining dense** (`core/dense/`, mirrors `src/core/f16_dense_q8a8.hpp`, `gemv_rows.hpp`,
   `activation_quant.hpp`): ds41 `dense_fast`, `prefill_q8`; Qwen `dense` + `transcode`; GLM `projection.inl` +
   `row_mma.cuh`; MiniMax `fp32_identity.cu` (keeps the `.inc` it shares with HIP).
4. **Grouped MoE** (`core/grouped_moe/`, mirrors `src/moe_core/grouped_core.hpp`, `grouped_policy.hpp`,
   `route.hpp`, `decode.hpp`, `device_plan.hpp`): ds41 `moe`, `moe_device.cuh`, `moe_grouped`, `moe_fast`, `moe_v2`,
   `router_v2`, `core_moe`; Qwen `moe` (already includes the HIP route storage `.inc`); GLM `native.cu` +
   `cutlass_path.inl` + `c1_layout.cuh` (expert order and epilogue kept); MiniMax calls its FP32 variant.
5. **Trellis decode** (`core/trellis/`, mirrors `src/core/trellis_codec.hpp`, `trellis_format.hpp`,
   `src/moe_core/exl3_trellis.hpp`): `core/trellis.cu` split into decode / linear; the GLM decoder and staging
   (`trellis.cuh`, `native.cu`) join with their own descriptor once bit equality is shown.
6. **Attention pieces** (`core/attention/`, mirrors `src/attn_core/flash_core.hpp`, `online_softmax.hpp`,
   `masks.hpp`, `select.hpp`, `index_score_*.hpp`, `sparse_flash_*.hpp`): ds41 `fast_attention` (sparse flash),
   `fast_index` (index score), the selection of `attention`; GLM `dsa`, `kda`; Qwen `qsa`, `gdn`; MiniMax MSA
   primitives. KDA, GDN, MLA and MSA stay distinct kernels under one tile / softmax / mask vocabulary; attention state
   export and replay policies stay model adapters.
7. **Runtime** (`core/runtime/`): buffer, graph and workspace lifetimes out of the ds41 / Qwen prefill drivers and
   the GLM producer runtime; the ds41 expert streamer, hot site and tier site behind one residency contract
   (tierstream kept). Device abstraction through `include/strix/device` (one runtime header, CUDA and HIP
   implementations).
8. **Build**: the GLM producer project and the MiniMax scripts become targets of the common CMake; duplicates are
   removed only when no adapter uses them.

The first Blackwell-only code (`arch/sm120/`, e.g. block-scaled FP4 for routed experts) enters as a policy of the
grouped MoE core of step 4, selected through `arch/capability.hpp`.

## Decode graphs: QSA inside the step graph, draft chain as one replay (Qwen, bitwise)

The 12 QSA attention cores of a decode window used to run as eager launches between 13 graph segments, because their
grids and arguments depend on the positions and the host pooled-key counts. `qwen_qsa::prepare_multi` now builds the
window's `SegmentMap` on the host (validation, pooled ranges, score waves), and `layer_core_device` launches the same
multi-segment kernels reading that map from device memory with a geometry fixed per segment-rows layout (pool and score
kernels walk their blocks and waves grid-stride; score runs past a row's selected count return at once). A window is one
captured graph; each replay first copies the 12 maps from pinned memory. The MTP draft chain of one session is captured
the same way (positions and the pending token from pinned memory, maps 12..18). Kernel bodies and their operation order
are unchanged, so the logits are bit-identical. Switches: `STRIX_QRTX_QSA_GRAPH=0` (cut graphs, eager cores) and
`STRIX_QRTX_DRAFT_GRAPH=0` (eager draft chain), both default on.
The C1 draft chain runs as two graph segments [0, a) and [a, depth): the second only when no draft of the first fell
below min_p (the verify would discard its steps). Kept drafts and verified tokens are unchanged; `a` comes from a
histogram of kept drafts (shared chooser `include/strix/mtp_draft_split.hpp`, expected steps a + P(kept > a) (depth - a
+ 0.25)). `STRIX_QRTX_DRAFT_SPLIT=0` = one segment, `N` = fixed a. Measured (#680, 3 ABAB pairs): prose C1 +7.3 %, json
+1.9 %, edit +-0.2 %, window 16.1 -> 14.5 ms at 1.86 tokens/window.
Context drafter (`STRIX_QRTX_CONTEXT_DRAFT=first|off|ctx|copy`, default `first`, printed in the server's start line): the shared host drafter
`include/strix/mtp_context_draft.hpp` (the HIP engine's `--mtp-ngram-draft=ctx|copy`) bound into `qwen_cohort::step`.
`copy`: a gated context span (long unanimous anchor, learned acceptance above the MTP cycle's) supplies the window's rows
with no predictor forward; otherwise context rows extend or replace the confident MTP prefix. The commit writes the MTP
teacher rows of every committed position, so skipping the predictor leaves its cache as after a drafted window; every
row passes the greedy verify (ids unchanged). `first` (default) uses only the copy-first windows; extension proposals
are scored for learning but never verified, because a rejected extension row costs ~1.2 ms of routed-expert bytes.
Measured (3 ABAB pairs, ids equal): edit C1 +13.6 %, C4 +3.5 %, C8 +2.1 %, json +-0.3 %, prose -0.5..-2.8 % (drafter
host time ~50 us per stream per cycle on requests that never copy). The server's `rtx` stats carry `copy_first_steps`,
`context_rows` and `context_host_us`.

Wide copy windows (`STRIX_QRTX_COPY_ROWS`, request field `copy_rows`, 1..16, <= 8 = off): a session alone in its step
whose context holds a long unanimous anchor (the shared `MtpContextDraft::propose_wide`: anchor >= `copy_anchor`, no known
edit in the span, learned copy acceptance >= `STRIX_QRTX_COPY_WIDE_ACCEPT`) verifies `copy_rows - 1` context rows in one
9..16-row window (the prefill-shaped families) and commits at the first rejected row (exact rollback). Wide serial bit equality is
gated by `qwen_decode_e2e` with `STRIX_QWEN_TMAX=16`: it requires G >= 137 and checks
actual widths 2..16, every full-width rollback keep, and a continuation after each keep against serial token/logit
digests. It does not compare MTP cache payloads or consume the wide commit's cache with a subsequent draft before
reset; those remain separate native qualification requirements. Stats `copy_wide_steps` and `copy_wide_rows`
count executed windows above eight rows and their actual proposal rows; `copy_wide_accepted` counts accepted rows.
The request override applies only to the CUDA Qwen cohort route with a context drafter in `first` or `copy` mode;
the CUDA single-slot loop and other engines do not bind it. Use this route for measurement arms. Shared API
applicability reporting and rejection of unsupported overrides remain pending.
Default `STRIX_QRTX_COPY_WIDE=auto`: on only when a fully accepted wide window beats a fully accepted 8-row window
under the measured window costs (`CopyWideModel`, qwen_cohort_plan.hpp). On sm_120 one session costs 9.65 ms at 1 row,
+1.1 ms per row to 17.2 ms at 8. Windows of 9..16 rows used to run the tiled prefill forms (flat 62.5 ms; forced on,
edit C1 -29 %). The backend's decode row cap (`strix/cuda/core/decode_rows.hpp`, `STRIX_CUDA_DECODE_ROWS_MAX`, default
16, `=8` = old switch point) now runs them on the decode forms: dense projections and router logits as <= 8-row
pieces, MoE row kernels (8-assignment chunks), QSA head kernels and the hc split forms at up to 16 rows, all with the
tiled forms' per-row chains (bits unchanged: qwen_decode_e2e STRIX_QWEN_TMAX=16 T=2..16 + rollback bitwise). A
16-row window costs 29.2 ms (27.6 at 9), so the auto rule (`copy_wide_measured`, by cap) turns wide copy on at 16
rows: edit C1 420.1 vs 372.7 tok/s (+12.7 %, 3 in-process ABAB pairs, ids equal; json / prose flat).
The cost table is a manual sm_120 calibration: recalibrate after kernel changes, qualify other devices separately
(`STRIX_QRTX_COPY_WIDE_MS` overrides the model; `STRIX_QRTX_COPY_WIDE=1` / `=0` force it).

Served decode by prompt class: `scripts/bench/cuda/qwen_serve_classes.py` (prose / edit / json, C1 / C4 / C8, cold and
warm passes, ids of every request) with `scripts/bench/cuda/qwen_classes_prompts.json`, and strict ABAB server arms with
`scripts/bench/cuda/qwen_serve_abab.sh` (server, model flags, arms and client are parameters; an arm's extra server flags
follow `##`, e.g. `F|ENV=v ## --profile fast`; GPU clock samples per arm; bench lines carry `tpc=` tokens per cycle).
Summarise arms with `scripts/bench/cuda/qwen_serve_summary.py OUT`: C1 means only over prompts run in both passes, C1
requests shorter than 16 tokens (the model ends early: prose slice 0 stops after 1 token) are listed and skipped, cold and
warm side by side per arm. Comparing a 2-prompt warm mean that contains a 1-token request with an 8-prompt cold mean is
what made warm look slower than cold in #680.
Arms that differ only by request fields run inside one server load with `scripts/bench/cuda/qwen_serve_inproc_abab.sh`
(`ARMS='A|{"copy_rows":8} ;; B|{"copy_rows":16}'`, the client's `--extra-json`): 3 pairs fit a 10-minute GPU turn.
