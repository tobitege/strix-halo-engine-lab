# Toggles

[Español](toggles.es.md)

This page lists the user-visible switches: CLI flags, `STRIX_*` environment variables and CMake options.
It says what each one does, what the default is, and when other hardware would want a different value.
It does not repeat measured gains. Gains, preconditions and receipts live in
[optimizations.md](optimizations.md) and the machine-readable [optimizations.json](optimizations.json).
Per-model numbers live in [performance.md](performance.md) and [measured-gains.md](measured-gains.md).

The defaults are tuned for one target: gfx1151 (Ryzen AI Max+ 395) with 128 GB of unified memory,
and a pair of these over USB4. On that target the plain command already runs the measured production path.
You only need this page to inspect a path, to debug, or to move the engine to different hardware.

## Rules that apply to every toggle

- An argument on the command line wins over a default. A variable already set in the environment is never changed.
- `--print-effective-config` prints the resolved environment and arguments before any device work.
- `=0` (or the listed `--no-*` flag) turns a default-on item off.
- A missing precondition (pack, library, memory) drops its item and prints one line on stderr.
- Pair apps need the same value on both ranks.

## Profiles: exact vs FAST

| Profile | What it does |
|---|---|
| `--profile default` | Current production path; exports no profile overrides. This is the default. |
| `--profile fast` | Adds only qualified bit-changing options. Rows that are not qualified yet stay off (see `--allow-pending`). |
| `--profile quality` | A higher-precision pack where one exists (Nemotron EXL3). Otherwise equal to `default`. |

Named profiles exist for Qwen, the DeepSeek pair, Gemma4/Nemotron and `strix-glm5-rank`.
MiniMax has no named product profile: `--profile NAME:PROMPT.u32:STEPS` runs a timed session.
GLM rank accepts `--profile default|fast|quality` (also `--profile=NAME`); its bare `--profile` still enables kernel timing
and can be combined with a named profile. GLM `fast` sets only `STRIX_GLM_PF_WMMA=1`; an explicit environment
value wins, and `--baseline` / `STRIX_GLM_BASELINE=1` vetoes the profile addition. `quality` uses default settings
with a notice. Both ranks must agree on the effective WMMA setting; existing startup guards still apply.

See the [Qwen profiles](optimizations.md#qwen-product-profiles),
[DeepSeek pair profiles](optimizations.md#product-profiles-pair) and [GLM FAST row](optimizations.md#glm-pf-wmma-x).
FAST changes the numerics. Use it when throughput matters more than bit-identical output.

| Switch | Engine | Default | What it does |
|---|---|---|---|
| `STRIX_QPF2K_MOE_ROUTE_BF16=1`, `--prefill-moe-route-bf16` | qwen | off (on in `fast`) | BF16 MoE routing in prefill. |
| `STRIX_QPF2K_GDN_FP16_CHUNK=1`, `--gdn-fp16-chunk` | qwen | off (on in `fast`) | FP16 GDN chunk in prefill. |
| `--prefill-exact-fp16` | qwen | off | Forces the exact FP16 prefill path. |
| `STRIX_DS41_FAST_MOE=1` | deepseek | off (`fast`) | Relaxed MoE epilogue. |
| `STRIX_GLM_PF_WMMA=1` | glm | off (on in rank `--profile fast`) | WMMA prefill. Opt-in FAST identity; `=0` disables even in `fast`. |

## The regime rule

The engine picks a kernel family by the number of rows in flight, not by model.

- **Decode at one row** is bound by memory bandwidth. Each weight is read once per token.
  The engine uses the smallest weights it can read: trellis/EXL3 packs (`STRIX_DS41_TRELLIS`, the EXL3 packs of MiniMax, GLM and Nemotron `quality`).
- **Wide verify and batched decode** (MTP, DSpark, several slots) multiply rows per weight read.
  Past a row threshold the matrix-core and integer paths win. The thresholds are toggles:
  `STRIX_INTPREFILL_DENSE_ROWS` (qwen, 512), `STRIX_QPF2K_DENSE_MIN_ROWS` (qwen, 1024 or 512 by site),
  `STRIX_GLM_PF_WMMA_ROWS` (glm, 64), `STRIX_MM_PF_ASYNC_MIN_ROWS` (minimax, 2048),
  `STRIX_GLM_PF_EXACT_X_MIN_ROWS` (glm, 2048).
- **Prefill** is bound by compute. It uses the integer and matrix-core paths, and the FAST profile adds the bit-changing ones.

The crossover depends on the ratio of compute to bandwidth. On gfx1151 that ratio is fixed, so the thresholds are fixed.
On other hardware, move them:

| Hardware differs by | Effect | What to try |
|---|---|---|
| More memory bandwidth (discrete GPU, HBM) | One-row decode is cheaper; small packs matter less | Higher-precision packs; disable trellis with `=0` and compare |
| More WMMA / INT4 throughput | Integer paths win at fewer rows | Lower the `*_MIN_ROWS` / `*_DENSE_ROWS` thresholds |
| Less WMMA / INT4 throughput | Integer paths win only at many rows | Raise the thresholds, or `=0` to keep the exact path |
| Less memory | KV, prefix cache and draft model compete with weights | Smaller prefix-cache budgets, `--slots`, `--prefill-rows N` |
| Faster or slower interconnect | Changes when overlap and splitting pay off | The pair transport switches below |

## CLI flags by area

The tables below list the curated flags. The full list of environment variables follows in the
[environment index](#environment-index).

### Memory and KV

| Flag | Engine | Default | What it does | When other hardware wants it |
|---|---|---|---|---|
| `--residency register\|copy` | qwen | `register` | Registers weights in place instead of copying. Needs a high `RLIMIT_MEMLOCK`. | `copy` when pinned memory is limited or on a discrete GPU with its own VRAM. |
| `--serve-memory-budget-bytes` | qwen | auto | Caps serve memory. | Lower on boxes with less than 128 GB. |
| `--known-prefix-*` | qwen | auto | Budgets for the known-prefix cache. | Lower with less memory. |
| `--prefix-cache on\|off` | qwen-rtx | on | Prefix cache. | Off when VRAM is tight. |
| `--no-prefix-cache`, `--no-prefix-disk`, `--prefix-cache-mb`, `--prefix-cache-dir` | deepseek serve | on | Prefix cache in RAM and on disk. | Smaller `--prefix-cache-mb` with less memory; move `--prefix-cache-dir` to fast storage. |
| `--capacity`, `--slots` | minimax / deepseek serve | per model | Context per slot and number of slots. | Fewer slots or less capacity with less memory. |
| `--prefill-rows auto\|N` | deepseek serve | `auto` | Prefill piece size that fits both ranks after weights and drafter. | Set `N` when auto refuses, or to trade memory for prefill speed. |

### Prefill

| Flag | Engine | Default | What it does | When other hardware wants it |
|---|---|---|---|---|
| `--block-schedule default\|tail-large` | qwen-rtx | `tail-large` | Order of blocks in the large kernels. | `default` on GPUs with a different CU count. |
| `--prefill auto` | glm | `auto` | Picks the prefill path. | - |
| `--prefill-chunk N` | glm | 512 | Prefill chunk size in rows. | Larger with more compute; smaller with less memory. |
| `--prefill-path device` | deepseek serve | `device` | Prefill on the GPU. | - |

### Decode, MTP and DSpark

| Flag | Engine | Default | What it does | When other hardware wants it |
|---|---|---|---|---|
| `--mtp-bundle`, `--rope-yarn 2\|4 --context C` | qwen | - | MTP draft bundle; YaRN for long context. | - |
| `--mtp 0\|1` | glm | resolved (see `--print-effective-config`) | Turns the MTP draft off or on. | `0` when memory is short. |
| `--identity r1b\|r1` | glm | `r1b` | Decode identity. | `r1` to compare. |
| `--dspark`, `--no-dspark`, `--dspark-vocab auto\|full` | deepseek | on when the draft is found | DSpark speculative draft. | `--no-dspark` when memory is short. |
| `--dspark-ranks` | deepseek | 2 | Ranks that run the draft. | - |
| `--dspark-max-inflight` | deepseek | 4 | Requests in flight before the draft turns off. `0` = always draft. | Higher with more compute per row. |
| `--dspark-mode off\|auto\|on`, `--dspark-prior` | minimax | `auto` (`STRIX_MM_SERVE_DSPARK`) | DSpark draft. | `off` when the draft does not fit. |

### Pair transport

| Flag | Engine | Default | What it does | When other hardware wants it |
|---|---|---|---|---|
| `--transport usb4auto\|usb4gpu` (prefill: `usb4-auto`\|`usb4-gpu`) | deepseek pair | `usb4auto` | USB4 transport. `usb4gpu` uses GPU-direct streams when the driver offers them, else the host path. | Other interconnects need their own transport. |
| `--peer-timeout` | deepseek serve | 120 | Seconds to wait for the peer. | Higher on slow links. |
| `--tp-attention`, `--no-tp-attention` | deepseek | on | Splits attention across the two ranks. | Off when the link is slow relative to compute. |
| `--engram-direct`, `--engram-mapped` | deepseek | `direct` | Engram table access. | `mapped` when memory is short. |
| `--decode-xch xfer\|comm`, `STRIX_GLM_DECODE_XCH` | glm | `xfer` (`comm` in baseline) | Decode exchange mode. | Compare both on a different interconnect. |
| `--wb-defer`, `--wb-stream`, `--ce-pacing` (`--no-*`) | deepseek-rtx | on | Write-back and copy-engine pacing. Also `STRIX_DS41_WB_STREAM`, `STRIX_DS41_CE_PACING`. | Off on links where the copy engine is not the bottleneck. |
| `STRIX_DS41_OB_SPLIT=K` | deepseek | off | Splits the o_b exchange into K chunks that overlap compute. Exact. Slower on the host path. | Try on a faster interconnect or with GPU-direct. |

Pair setup: [pair-setup.md](pair-setup.md).

### Exact vs FAST

| Flag | Engine | Default | What it does |
|---|---|---|---|
| `--profile default\|fast\|quality` | qwen, deepseek pair, gemma4/nemotron, glm rank | `default` | See [Profiles](#profiles-exact-vs-fast). |
| `--profile` (bare) | glm rank | off | Kernel timing; preserved alongside named profiles. |
| `--profile NAME:PROMPT.u32:STEPS` | minimax rank | off | Timed session, not a product profile. |
| `--allow-pending` | pair apps | off | Lets `fast` apply rows that are not qualified yet. |
| `--print-effective-config` | all | off | Prints the resolved configuration and exits before device work. |
| `STRIX_DS41_BASELINE=1` | deepseek | off | Turns off every production default. Runs the plain command as it ran before. |

### Debug only

These flags do not change the output. They add checks, traces or timing. Do not use them for benchmarks.

| Flag | Engine | What it does |
|---|---|---|
| `--timing` | various | Per-stage timing. |
| `--verify-trace` | decode apps | Traces verify steps. |
| `--xb-check`, `--decode-check` | various | Compares against a reference path. |
| `--resume-check*` | serve | Checks state after resume. |
| `--selftest`, `--idcheck` | various | Self tests and identity checks. |

Environment variables whose names contain `TRACE`, `DUMP`, `DEBUG`, `LOG`, `PROBE`, `TIMING`, `CHECK` or `STATS` are debug only.

## CMake options

| Option | Default | What it does | When other hardware wants it |
|---|---|---|---|
| `STRIX_HIP_ARCHS` | `gfx1151` | GPU targets to build, separated by `;`. | Set your architecture. Kernels are tuned for gfx1151. |
| `STRIX_FAMILY_DEEPSEEK41` | ON | Builds the DeepSeek V4.1 family. | - |
| `STRIX_FAMILY_GLM5` | OFF | Builds the GLM-5.3 Flash engine. | - |
| `STRIX_FAMILY_MINIMAX` | OFF | Builds the MiniMax modules. | - |
| `STRIX_FAMILY_GEMMA4` | OFF | Builds the Nemotron 3.5 Bolt family. | - |
| `STRIX_FAMILY_H3` | OFF | Builds the H3 family. | - |
| `STRIX_DS41_TRELLIS`, `STRIX_DS41_TRELLIS_LIBRARY`, `STRIX_DS41_TRELLIS_INCLUDE_DIR` | OFF | Links the trellis library for DeepSeek dense packs. Without it the packs are skipped. | Most useful when bandwidth limits decode. |
| `STRIX_DS41_PREFILL_FP16_LT` | OFF | Dense FP16 prefill backend (with `STRIX_DS41_PREFILL_FP16_DENSE=1`). | GPUs with fast FP16 matrix cores. |
| `STRIX_DPP_REDUCE` | OFF | Exact DPP exchanges in guarded Qwen reductions. | Try ON on RDNA GPUs with DPP; leave OFF elsewhere. |
| `STRIX_MINIMAX_GU4_SCALAR` | ON | Exact MiniMax gu4 MoE expert addressing. OFF builds the former reader. | - |

## Environment index

Generated from [optimizations.json](optimizations.json). The area is inferred from the name.
"Kernel paths" are per-kernel choices: each one has an exact default and an `=0` off switch.
Follow the link for the gain, preconditions and off switch. Defaults are shortened; the link has the full text.

### Memory and KV (12)

| Toggle | Engine | Default | Details |
|---|---|---|---|
| `STRIX_DS41_ENGRAM_PAYLOAD_CACHE` | deepseek | auto | [ds-prefill](optimizations.md#ds-prefill) |
| `STRIX_DS41_PREFIX_DIAGNOSTICS` | deepseek | ON: 512 MiB/rank, block 256, reserve ... | [deepseek-pair-agent-prefix-cache](engine/serving/prefix-state-cache.md#deepseek-pair-agent-prefix-cache) |
| `STRIX_PREFIX_CACHE` | qwen | ON on the qualified single tiled-MTP ... | [certified-mtp-subset](engine/serving/prefix-state-cache.md#certified-mtp-subset) |
| `STRIX_PREFIX_CACHE_DISK` | qwen | ON: joining wherever the prefix cache... | [certified-restart-and-in-flight-joining](engine/serving/prefix-state-cache.md#certified-restart-and-in-flight-joining) |
| `STRIX_PREFIX_CACHE_DURABLE` | qwen | ON: joining wherever the prefix cache... | [certified-restart-and-in-flight-joining](engine/serving/prefix-state-cache.md#certified-restart-and-in-flight-joining) |
| `STRIX_PREFIX_CACHE_INFLIGHT` | qwen | ON: joining wherever the prefix cache... | [certified-restart-and-in-flight-joining](engine/serving/prefix-state-cache.md#certified-restart-and-in-flight-joining) |
| `STRIX_PREFIX_CACHE_RETENTION` | qwen | ON on the qualified single tiled-MTP ... | [prefix-retention-policy](engine/serving/prefix-state-cache.md#prefix-retention-policy) |
| `STRIX_PREFIX_CACHE_STAGING_BYTES` | qwen | ON on the qualified single tiled-MTP ... | [certified-mtp-subset](engine/serving/prefix-state-cache.md#certified-mtp-subset) |
| `STRIX_PREFIX_FAIR_RECLAIM_IDLE` | qwen | ON on the qualified single tiled-MTP ... | [prefix-retention-policy](engine/serving/prefix-state-cache.md#prefix-retention-policy) |
| `STRIX_QRTX_PREFIX_CACHE` | qwen-rtx | ON | [qwen-rtx-prefix-cache](optimizations.md#qwen-rtx-prefix-cache) |
| `STRIX_QSA_CACHE_ONLY` | qwen | 1 | [qwen-qsa-cache-only](optimizations.md#qwen-qsa-cache-only) |
| `STRIX_SESSION_CACHE` | minimax | on (two-node serve route) | [mm-sessions](optimizations.md#mm-sessions) |

### Prefill (97)

| Toggle | Engine | Default | Details |
|---|---|---|---|
| `STRIX_DS41_ATTN_LIVE_CHUNKS` | deepseek | auto | [ds-attention](optimizations.md#ds-attention) |
| `STRIX_DS41_ENGRAM_F16_PF` | deepseek | on | [ds-engram-f16-pf](optimizations.md#ds-engram-f16-pf) |
| `STRIX_DS41_MOE_PREFILL` | deepseek | ON | [ds-moe7](optimizations.md#ds-moe7) |
| `STRIX_DS41_PREFILL_FINAL_KV` | deepseek | auto | [ds-prefill](optimizations.md#ds-prefill) |
| `STRIX_DS41_PREFILL_HC_COLLAPSE` | deepseek | on (with STRIX_DS41_PREFILL_HC_FUSE) | [ds-prefill-hc-collapse](optimizations.md#ds-prefill-hc-collapse) |
| `STRIX_DS41_PREFILL_HC_FUSE` | deepseek | on | [ds-prefill-hc-fuse](optimizations.md#ds-prefill-hc-fuse) |
| `STRIX_DS41_PREFILL_MOE_BAND` | deepseek | on | [ds-moe-v2](optimizations.md#ds-moe-v2) |
| `STRIX_DS41_PREFILL_MOE_V2` | deepseek | on | [ds-moe-v2](optimizations.md#ds-moe-v2) |
| `STRIX_DS41_PREFILL_MOE_V2_Q8` | deepseek | on | [ds-moe-v2](optimizations.md#ds-moe-v2) |
| `STRIX_DS41_PREFILL_O_A8_FUSED` | deepseek | on | [ds-o-a8-fused](optimizations.md#ds-o-a8-fused) |
| `STRIX_DS41_PREFILL_O_B_ONE` | deepseek | on | [ds-prefill-o-b-one](optimizations.md#ds-prefill-o-b-one) |
| `STRIX_DS41_PREFILL_O_LOW_A8_FUSED` | deepseek | on | [ds-prefill-o-low-a8](optimizations.md#ds-prefill-o-low-a8) |
| `STRIX_DS41_PREFILL_ROUTER_V2` | deepseek | on | [ds-router-v2](optimizations.md#ds-router-v2) |
| `STRIX_GLM_PF_ATTN_QWIN` | glm | auto | [glm-pf-attn-qwin](optimizations.md#glm-pf-attn-qwin) |
| `STRIX_GLM_PF_AUX_COMBINE` | glm | 1 | [glm-pf-aux-combine](optimizations.md#glm-pf-aux-combine) |
| `STRIX_GLM_PF_DENSE_ROWS` | glm | 3 | [glm-pf-dense-rows-v3](optimizations.md#glm-pf-dense-rows-v3) |
| `STRIX_GLM_PF_HC_EXPAND_ROWS` | glm | 2 | [glm-pf-hc-expand-rows](optimizations.md#glm-pf-hc-expand-rows) |
| `STRIX_GLM_PF_HC_FUSE` | glm | 1 | [glm-pf-hc-fuse](optimizations.md#glm-pf-hc-fuse) |
| `STRIX_GLM_PF_HC_HOIST` | glm | 1, 1 | [glm-pf-sumexp](optimizations.md#glm-pf-sumexp) |
| `STRIX_GLM_PF_IDX` | glm | 3 / 2 | [glm-pf-indexer-v2](optimizations.md#glm-pf-indexer-v2) |
| `STRIX_GLM_PF_IN_TOKEN` | glm | 1 | [glm-pf-in-token](optimizations.md#glm-pf-in-token) |
| `STRIX_GLM_PF_KDA_PACK` | glm | 0 | [glm-pf-kda-pack](optimizations.md#glm-pf-kda-pack) |
| `STRIX_GLM_PF_KDA_SCAN` | glm | 5 | [glm-pf-kda-scan-v5](optimizations.md#glm-pf-kda-scan-v5) |
| `STRIX_GLM_PF_MIXNORM` | glm | 1 | [glm-pf-mixnorm](optimizations.md#glm-pf-mixnorm) |
| `STRIX_GLM_PF_MLA_GEMM` | glm | 3 (v3); 2 selects v2 | [glm-pf-mla-gemm-v2](optimizations.md#glm-pf-mla-gemm-v2) |
| `STRIX_GLM_PF_OBAR` | glm | 10 | [glm-pf-obar-v5](optimizations.md#glm-pf-obar-v5) |
| `STRIX_GLM_PF_PIPELINE` | glm | 2 / 1 | [prefill-pipeline-and-async-exchange-start](optimizations.md#prefill-pipeline-and-async-exchange-start) |
| `STRIX_GLM_PF_SCORES` | glm | 7 | [glm-pf-scores-v3](optimizations.md#glm-pf-scores-v3) |
| `STRIX_GLM_PF_SCRATCH` | glm | auto | [glm-pf-scratch](optimizations.md#glm-pf-scratch) |
| `STRIX_GLM_PF_SELECT` | glm | 3 / 2 | [glm-pf-indexer-v2](optimizations.md#glm-pf-indexer-v2) |
| `STRIX_GLM_PF_SEQPAR` | glm | auto | [glm-pf-seqpar](optimizations.md#glm-pf-seqpar) |
| `STRIX_GLM_PF_STAGED` | glm | 1 | [glm-pf-staged](optimizations.md#glm-pf-staged) |
| `STRIX_GLM_PF_SUMEXP` | glm | 1, 1 | [glm-pf-sumexp](optimizations.md#glm-pf-sumexp) |
| `STRIX_GLM_PF_WMMA` | glm | off (opt-in FAST identity) | [glm-pf-wmma-x](optimizations.md#glm-pf-wmma-x) |
| `STRIX_INTPREFILL_DENSE_ROWS` | qwen | 512 | [qwen-intprefill-dense-rows](optimizations.md#qwen-intprefill-dense-rows) |
| `STRIX_MMX_PREFILL_CORE` | minimax | v4 | [mm-pf-core-v4](optimizations.md#mm-pf-core-v4) |
| `STRIX_MMX_PREFILL_GATHER_TOK` | minimax | 1 | [mm-pf-gather-tok](optimizations.md#mm-pf-gather-tok) |
| `STRIX_MMX_PREFILL_MTFIT` | minimax | 1 | [mm-pf-mtfit](optimizations.md#mm-pf-mtfit) |
| `STRIX_MMX_PREFILL_PIPE` | minimax | 1 | [mm-pf-moe-pipe](optimizations.md#mm-pf-moe-pipe) |
| `STRIX_MMX_PREFILL_TAIL_STREAM` | minimax | 1 | [mm-pf-tail-stream](optimizations.md#mm-pf-tail-stream) |
| `STRIX_MM_PFX` | minimax | on | [mm-rank-pfx](optimizations.md#mm-rank-pfx) |
| `STRIX_MM_PFX_CHUNK` | minimax | 4194304 | [mm-pfx-chunk](optimizations.md#mm-pfx-chunk) |
| `STRIX_MM_PF_ASYNC_START` | minimax | ON | [minimax-grouped-prefill.md](minimax-grouped-prefill.md) |
| `STRIX_MM_PF_ATTEND4` | minimax | 1 | [mm-pf-attend4](optimizations.md#mm-pf-attend4) |
| `STRIX_MM_PF_DENSE_O` | minimax | 1 | [mm-pf-dense-o](optimizations.md#mm-pf-dense-o) |
| `STRIX_MM_PF_DENSE_QKV` | minimax | 1 | [mm-pf-dense-qkv](optimizations.md#mm-pf-dense-qkv) |
| `STRIX_MM_PF_IDX_ROWS` | minimax | 64 | [mm-pf-idx-rows](optimizations.md#mm-pf-idx-rows) |
| `STRIX_MM_PF_IDX_TILE` | minimax | 1 | [mm-pf-idx-tile](optimizations.md#mm-pf-idx-tile) |
| `STRIX_MM_PF_INTERLEAVE` | minimax | XFER on, PFX loaded, interleave on | [mm-routes](optimizations.md#mm-routes) |
| `STRIX_MM_PF_KV_TILE` | minimax | auto | [mm-pf-kv-tile](optimizations.md#mm-pf-kv-tile) |
| `STRIX_MM_PF_LATE_SRC_FENCE` | minimax | ON | [minimax-grouped-prefill.md](minimax-grouped-prefill.md) |
| `STRIX_MM_PF_MSA_BATCH` | minimax | 1 | [mm-pf-msa-batch](optimizations.md#mm-pf-msa-batch) |
| `STRIX_MM_PF_MSA_REWORK` | minimax | 5 | [mm-pf-msa-rework](optimizations.md#mm-pf-msa-rework) |
| `STRIX_MM_PF_MSA_SCRATCH_MB` | minimax | 1152 | [mm-pf-msa-scratch](optimizations.md#mm-pf-msa-scratch) |
| `STRIX_MM_PF_ROUTER_BATCH` | minimax | 1 | [mm-pf-router-batch](optimizations.md#mm-pf-router-batch) |
| `STRIX_MM_PF_ROUTER_ROWS` | minimax | 32 | [mm-pf-router-v2](optimizations.md#mm-pf-router-v2) |
| `STRIX_MM_PF_TABLE_RING` | minimax | 1 in strix-minimax-rank; 0 in strix-m... | [mm-pf-table-ring](optimizations.md#mm-pf-table-ring) |
| `STRIX_MM_PF_TX_UNCACHED` | minimax | 1 | [mm-pf-tx-uncached](optimizations.md#mm-pf-tx-uncached) |
| `STRIX_PF_P1` | minimax | ON (1) | [mm-pf-moe-pipe](optimizations.md#mm-pf-moe-pipe) |
| `STRIX_PREFILL_HC_OUTPUT_WMMA` | qwen | ON | [bit-changing-prefill-options](optimizations.md#bit-changing-prefill-options) |
| `STRIX_PREFILL_INDEX_PROJ_WMMA` | qwen | OFF; opt in with STRIX_PREFILL_INDEX_... | [bit-changing-prefill-options](optimizations.md#bit-changing-prefill-options) |
| `STRIX_PREFILL_INDEX_SCORE_WMMA` | qwen | OFF; opt in with STRIX_PREFILL_INDEX_... | [engine/prefill/index-score-wmma.md](engine/prefill/index-score-wmma.md) |
| `STRIX_PREFILL_MOE_MERGE_VEC` | qwen | 1 | [qwen-prefill-moe-merge-vec](optimizations.md#qwen-prefill-moe-merge-vec) |
| `STRIX_PREFILL_PIPE` | qwen | 255 | [qwen-prefill-pipe](optimizations.md#qwen-prefill-pipe) |
| `STRIX_PREFILL_PLE_BATCH` | qwen | 1 | [qwen-ple-batch](optimizations.md#qwen-ple-batch) |
| `STRIX_PREFILL_ROUTER_SELECT` | qwen | ON | [qwen-router](optimizations.md#qwen-router) |
| `STRIX_QPF2K_COMPACT_IMAGES` | qwen | ON for eligible full-T score-ring owners | [exact-compact-qsa-attention-output](optimizations.md#exact-compact-qsa-attention-output) |
| `STRIX_QPF2K_DEFAULTS` | qwen | ON | [qwen-defaults](optimizations.md#qwen-defaults) |
| `STRIX_QPF2K_DENSE` | qwen | 1 | [qwen-dense](optimizations.md#qwen-dense) |
| `STRIX_QPF2K_EARLY_SETUP` | qwen | ON for eligible fresh plain-command f... | [early-tiled-prefill-geometry-setup](optimizations.md#early-tiled-prefill-geometry-setup) |
| `STRIX_QPF2K_GDN` | qwen | 1 | [qwen-gdn](optimizations.md#qwen-gdn) |
| `STRIX_QPF2K_GDN_CHUNK16` | qwen | auto | [qwen-gdn-chunk16](optimizations.md#qwen-gdn-chunk16) |
| `STRIX_QPF2K_GDN_COOP_LAUNCH` | qwen | 0 | [qwen-gdn](optimizations.md#qwen-gdn) |
| `STRIX_QPF2K_GDN_CORE16` | qwen | 1 | [qwen-gdn](optimizations.md#qwen-gdn) |
| `STRIX_QPF2K_GDN_GATES_BATCH` | qwen | 1 | [qwen-gdn](optimizations.md#qwen-gdn) |
| `STRIX_QPF2K_GDN_PARTS` | qwen | opt-in (bit 128 off; normal mask 55) | [qwen-gdn-witness](optimizations.md#qwen-gdn-witness) |
| `STRIX_QPF2K_HC` | qwen | 1 | [qwen-hc](optimizations.md#qwen-hc) |
| `STRIX_QPF2K_HC_LOWBAND` | qwen | ON (3) through decoder qpf2k defaults | [hc-low-band-layout](optimizations.md#hc-low-band-layout) |
| `STRIX_QPF2K_HC_MASK` | qwen | 14 | [qwen-hc](optimizations.md#qwen-hc) |
| `STRIX_QPF2K_HC_OUTPUT3` | qwen | 1 | [qwen-hc](optimizations.md#qwen-hc) |
| `STRIX_QPF2K_HC_OUTPUT3_BAND` | qwen | 16 | [qwen-hc](optimizations.md#qwen-hc) |
| `STRIX_QPF2K_HC_RECOMPUTE_NORM` | qwen | OFF pending native equality and not-s... | [hc-normalized-recompute-candidate](optimizations.md#hc-normalized-recompute-candidate) |
| `STRIX_QPF2K_HC_UPCOPY` | qwen | auto with HC output v3 | [qwen-hc](optimizations.md#qwen-hc) |
| `STRIX_QPF2K_INDEX_BATCH` | qwen | ON | [exact-indexer-projection-batching](optimizations.md#exact-indexer-projection-batching) |
| `STRIX_QPF2K_MOE_DA16` | qwen | 1 | [qwen-moe](optimizations.md#qwen-moe) |
| `STRIX_QPF2K_MOE_DIRECT_B` | qwen | OFF; explicit candidate only | [qwen-moe-compressed-planes](optimizations.md#qwen-moe-compressed-planes) |
| `STRIX_QPF2K_MOE_HDR` | qwen | 1 | [qwen-moe-header-reuse](optimizations.md#qwen-moe-header-reuse) |
| `STRIX_QPF2K_MOE_MC` | qwen | 1 (mode 1; mode 2 needs an explicit v... | [qwen-moe-merge-commit](optimizations.md#qwen-moe-merge-commit) |
| `STRIX_QPF2K_MOE_ROUTE` | qwen | 14 | [qwen-moe](optimizations.md#qwen-moe) |
| `STRIX_QPF2K_MOE_XOR64` | qwen | 1 | [qwen-moe-xor-rows](optimizations.md#qwen-moe-xor-rows) |
| `STRIX_QPF2K_QSA` | qwen | 1 | [qwen-qsa](optimizations.md#qwen-qsa) |
| `STRIX_QPF2K_QSA_MASK` | qwen | 0x2257 | [qwen-qsa](optimizations.md#qwen-qsa) |
| `STRIX_QPF2K_SCHED` | qwen | 14 | [qwen-schedule](optimizations.md#qwen-schedule) |
| `STRIX_QPF2K_SCORE_ROWS` | qwen | auto (32 rows above 4096 blocks, 16 o... | [qwen-qsa-score-rows](optimizations.md#qwen-qsa-score-rows) |
| `STRIX_QPF2K_SHARED_PRE` | qwen | 1 | [qwen-shared-pre](optimizations.md#qwen-shared-pre) |
| `STRIX_QPF2K_T16K` | qwen | ON from 32768 prompt tokens | [automatic-t16384-prefill](optimizations.md#automatic-t16384-prefill) |
| `STRIX_QPF2K_WMMA_PIN` | qwen | 1 | [qwen-dense](optimizations.md#qwen-dense) |
| `STRIX_SERVE_PREFILL` | qwen | ON: admitted T8192; configured prefix... | [default-prompt-prefill](engine/serving/serving.md#default-prompt-prefill) |

### Decode, MTP and DSpark (22)

| Toggle | Engine | Default | Details |
|---|---|---|---|
| `STRIX_BATCH_TOOL_DRAFT` | qwen | ON (batched decoding) | [qwen-tool-draft](optimizations.md#qwen-tool-draft) |
| `STRIX_DS41_DECODE_PREFETCH` | deepseek | auto | [ds-prefetch](optimizations.md#ds-prefetch) |
| `STRIX_DS41_DEFER_DECODER` | deepseek | on | [ds-defer-decoder](optimizations.md#ds-defer-decoder) |
| `STRIX_DS41_DSPARK_CALIB` | deepseek | auto | [ds-warm](optimizations.md#ds-warm) |
| `STRIX_DS41_R1FUSE_ROUTES` | deepseek | on | [ds-exact-decode-launches](optimizations.md#ds-exact-decode-launches) |
| `STRIX_GLM_R1B_DENSE` | glm | 7 | [glm-r1b-dense-auto](optimizations.md#glm-r1b-dense-auto) |
| `STRIX_GLM_R1B_HC` | glm | 4 | [glm-r1b-hc-v4](optimizations.md#glm-r1b-hc-v4) |
| `STRIX_GLM_SPEC_KDA_TREE` | glm | on (Xfer verify exchange, KDA tree 2,... | [glm-mtp-spec](optimizations.md#glm-mtp-spec) |
| `STRIX_MM_ATTEND4_DECODE` | minimax | 1 | [mm-attend4-decode](optimizations.md#mm-attend4-decode) |
| `STRIX_MM_DRAFT_LIN` | minimax | opt | [mm-draft-lin](optimizations.md#mm-draft-lin) |
| `STRIX_MM_DSPARK_ATTN` | minimax | 2 | [mm-dspark-attn](optimizations.md#mm-dspark-attn) |
| `STRIX_MM_SERVE_DSPARK` | minimax | auto | [mm-serve-dspark](optimizations.md#mm-serve-dspark) |
| `STRIX_MM_SERVE_DSPARK_STAGE` | minimax | 1 | [mm-serve-dspark-stage](optimizations.md#mm-serve-dspark-stage) |
| `STRIX_MOE_R1_GLUE_FUSE` | qwen | 1 (eager plain one-row decode) | [qwen-plain-decode-step](optimizations.md#qwen-plain-decode-step) |
| `STRIX_QSA_DECODE_ATTN_V2` | qwen | 1 (eager plain one-row decode) | [qwen-plain-decode-step](optimizations.md#qwen-plain-decode-step) |
| `STRIX_QSA_DECODE_FUSED_MAXIMA` | qwen | 1 (eager plain one-row decode) | [qwen-plain-decode-step](optimizations.md#qwen-plain-decode-step) |
| `STRIX_QSA_DECODE_POOL` | qwen | 1 (plain append-only serial decode) | [qwen-qsa-decode-pool](optimizations.md#qwen-qsa-decode-pool) |
| `STRIX_QSA_DECODE_SELECT` | qwen | 1 | [qwen-qsa-decode-select](optimizations.md#qwen-qsa-decode-select) |
| `STRIX_QSA_DECODE_VALUE_V2` | qwen | 1 (eager plain one-row decode) | [qwen-plain-decode-step](optimizations.md#qwen-plain-decode-step) |
| `STRIX_QWEN_DECODE_FUSE` | qwen | OFF (opt-in: STRIX_QWEN_DECODE_FUSE=1) | [qwen-decode-fuse](optimizations.md#qwen-decode-fuse) |
| `STRIX_QWEN_YARN_MTP` | qwen | table-backed QSA MTP pool and rolling... | [qwen-yarn-mtp](optimizations.md#qwen-yarn-mtp) |
| `STRIX_TOOL_DRAFT_PREFIX_GATE` | qwen | ON (batched decoding) | [qwen-tool-draft](optimizations.md#qwen-tool-draft) |

### Pair transport (25)

| Toggle | Engine | Default | Details |
|---|---|---|---|
| `STRIX_DIST_ASYNC_START` | glm | 2 / 1 | [prefill-pipeline-and-async-exchange-start](optimizations.md#prefill-pipeline-and-async-exchange-start) |
| `STRIX_DIST_CRC_DEFER` | deepseek | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_DIST_CRC_PARALLEL` | deepseek, glm, minimax | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_DIST_HOST_CHUNK_BYTES` | deepseek, glm, minimax | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_DIST_HOST_CRC_OVERLAP` | deepseek, glm, minimax | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_DIST_STAGED_READY_BYTES` | deepseek | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_DIST_STAGING` | deepseek | auto | [ds-transport](optimizations.md#ds-transport) |
| `STRIX_DIST_USB4_EAGER` | deepseek | auto | [ds-transport](optimizations.md#ds-transport) |
| `STRIX_DS41_CE_PACING` | deepseek-rtx | ON | [rtx-ce-pacing](optimizations.md#rtx-ce-pacing) |
| `STRIX_DS41_ENGRAM_TP` | deepseek | on for steps of up to 8 rows | [ds-engram-tp](optimizations.md#ds-engram-tp) |
| `STRIX_DS41_EXPERT_TP` | deepseek | ON | [ds-core](optimizations.md#ds-core) |
| `STRIX_DS41_F16_PAIR` | deepseek | ON | [ds-core](optimizations.md#ds-core) |
| `STRIX_DS41_TP_DOWN` | deepseek | ON | [ds-core](optimizations.md#ds-core) |
| `STRIX_DS41_TRELLIS_QAKV_PAIR` | deepseek | auto | [ds-trellis](optimizations.md#ds-trellis) |
| `STRIX_DS41_WB_STREAM` | deepseek-rtx | ON | [rtx-wb-stream](optimizations.md#rtx-wb-stream) |
| `STRIX_GLM_MTP_SPEC` | glm | on (Xfer verify exchange, KDA tree 2,... | [glm-mtp-spec](optimizations.md#glm-mtp-spec) |
| `STRIX_GLM_SPEC_XCH` | glm | on (Xfer verify exchange, KDA tree 2,... | [glm-mtp-spec](optimizations.md#glm-mtp-spec) |
| `STRIX_HTTP_EAGER_SLOTS` | qwen | AUTO capacity; at most 4 eager slots;... | [batched-decoding](engine/serving/serving.md#batched-decoding) |
| `STRIX_MMX_LINEAR_V11` | minimax | 1 | [mm-linear-v11](optimizations.md#mm-linear-v11) |
| `STRIX_MMX_PREFILL_PIPE_PAIR` | minimax | auto | [mm-pf-pipe-pair](optimizations.md#mm-pf-pipe-pair) |
| `STRIX_MM_XFER` | minimax | XFER on, PFX loaded, interleave on | [mm-routes](optimizations.md#mm-routes) |
| `STRIX_MTP_JOINT_PLAN` | qwen | calibrated-c3 when STRIX_MTP_BATCH_JO... | [joint-plan-rule](engine/scheduling/joint-serving-windows.md#joint-plan-rule) |
| `STRIX_MTP_JOINT_RULE_FILE` | qwen | calibrated-c3 when STRIX_MTP_BATCH_JO... | [joint-plan-rule](engine/scheduling/joint-serving-windows.md#joint-plan-rule) |
| `STRIX_MTP_PRIME_ROWS` | qwen | ON (STRIX_MTP_PRIME_ROWS=0: disable) | [qwen-mtp-prime-rows](optimizations.md#qwen-mtp-prime-rows) |
| `STRIX_QSA_MTP_POOL` | qwen | 1 (eligible native MTP CLI, the singl... | [qwen-qsa-mtp-pool](optimizations.md#qwen-qsa-mtp-pool) |

### Exact vs FAST (9)

| Toggle | Engine | Default | Details |
|---|---|---|---|
| `STRIX_DS41_COMPRESS_FAST` | deepseek | on | [ds-exact-decode-launches](optimizations.md#ds-exact-decode-launches) |
| `STRIX_DS41_FAST_MOE_DOWN_ROWS` | deepseek | 32 (with STRIX_DS41_FAST_MOE=1) | [ds-fast-moe-down-rows](optimizations.md#ds-fast-moe-down-rows) |
| `STRIX_DS41_FAST_MOE_ORDER` | deepseek | on (with STRIX_DS41_FAST_MOE=1) | [ds-fast-moe-order](optimizations.md#ds-fast-moe-order) |
| `STRIX_DS41_ROUTES_FAST` | deepseek | ON | [ds-core](optimizations.md#ds-core) |
| `STRIX_GLM_PF_EXACTMOE_V` | glm | auto | [glm-pf-exact-moe-auto](optimizations.md#glm-pf-exact-moe-auto) |
| `STRIX_GLM_PF_EXACT_X` | glm | auto | [glm-pf-exact-x](optimizations.md#glm-pf-exact-x) |
| `STRIX_QPF2K_MOE_ROUTE_BF16` | qwen | OFF | [qwen-moe-route-bf16](optimizations.md#qwen-moe-route-bf16) |
| `STRIX_QSA_DECODE_POOL_BF16` | qwen | 1 (plain append-only serial decode un... | [qwen-qsa-decode-pool](optimizations.md#qwen-qsa-decode-pool) |
| `STRIX_QWEN_ROUTER_BF16` | qwen | OFF (opt-in: STRIX_QWEN_ROUTER_BF16=1) | [qwen-exact-bf16-router-weights](optimizations.md#qwen-exact-bf16-router-weights) |

### Kernel paths (121)

| Toggle | Engine | Default | Details |
|---|---|---|---|
| `STRIX_BASELINE` | qwen | fast path | [qwen-tiled](optimizations.md#qwen-tiled) |
| `STRIX_BATCH_CANCEL_POINTS` | qwen | 250 ms; 2 layers; 2048 rows; 2048 row... | [batched-decoding](engine/serving/serving.md#batched-decoding) |
| `STRIX_BATCH_NEST_ROWS` | qwen | 250 ms; 2 layers; 2048 rows; 2048 row... | [batched-decoding](engine/serving/serving.md#batched-decoding) |
| `STRIX_BATCH_SERVICE_LOOKAHEAD` | qwen | 250 ms; 2 layers; 2048 rows; 2048 row... | [batched-decoding](engine/serving/serving.md#batched-decoding) |
| `STRIX_BATCH_SERVICE_MS` | qwen | 250 ms; 2 layers; 2048 rows; 2048 row... | [batched-decoding](engine/serving/serving.md#batched-decoding) |
| `STRIX_BATCH_SHORT_FIRST_ROWS` | qwen | 250 ms; 2 layers; 2048 rows; 2048 row... | [batched-decoding](engine/serving/serving.md#batched-decoding) |
| `STRIX_DISPATCH_FUSE` | qwen | 1 | [qwen-dispatch-fuse](optimizations.md#qwen-dispatch-fuse) |
| `STRIX_DS41_ATTN_LDS_SWIZZLE` | deepseek | on | [ds-attn-swizzle](optimizations.md#ds-attn-swizzle) |
| `STRIX_DS41_ATTN_LEAN` | deepseek | 3 | [ds-attn-lean](optimizations.md#ds-attn-lean) |
| `STRIX_DS41_ATTN_SEG_HPW` | deepseek | auto | [ds-index](optimizations.md#ds-index) |
| `STRIX_DS41_ATTN_TILE_F16` | deepseek | auto | [ds-attention](optimizations.md#ds-attention) |
| `STRIX_DS41_ATTN_WAVE_CAP` | deepseek | on | [ds-attn-wave-cap](optimizations.md#ds-attn-wave-cap) |
| `STRIX_DS41_ATTN_WINDOW_FOLD` | deepseek | auto | [ds-attention](optimizations.md#ds-attention) |
| `STRIX_DS41_BASELINE` | deepseek | fast path | [ds-packs](optimizations.md#ds-packs) |
| `STRIX_DS41_BATCH_PREFETCH` | deepseek | auto | [ds-prefetch](optimizations.md#ds-prefetch) |
| `STRIX_DS41_BATCH_SEGMENTS` | deepseek | ON | [ds-core](optimizations.md#ds-core) |
| `STRIX_DS41_BATCH_SHARED` | deepseek | auto | [ds-overlap](optimizations.md#ds-overlap) |
| `STRIX_DS41_CB4` | deepseek | auto | [ds-packs](optimizations.md#ds-packs) |
| `STRIX_DS41_DENSE_SELECT` | deepseek | auto | [ds-dense](optimizations.md#ds-dense) |
| `STRIX_DS41_ENGRAM_LOOKAHEAD` | deepseek | on | [ds-engram-lookahead](optimizations.md#ds-engram-lookahead) |
| `STRIX_DS41_ENGRAM_SPLIT_READ` | deepseek | on | [ds-engram-split-read](optimizations.md#ds-engram-split-read) |
| `STRIX_DS41_EXPERT_SLOTS` | deepseek | ON | [ds-core](optimizations.md#ds-core) |
| `STRIX_DS41_IDX_CAND_FILTER` | deepseek | auto | [ds-index](optimizations.md#ds-index) |
| `STRIX_DS41_IDX_DIRECT` | deepseek | auto | [ds-index](optimizations.md#ds-index) |
| `STRIX_DS41_LATE_HC_LAYERS` | deepseek | auto | [ds-overlap](optimizations.md#ds-overlap) |
| `STRIX_DS41_MOE_READY` | deepseek | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_DS41_NORM_ILP` | deepseek | on | [ds-exact-decode-launches](optimizations.md#ds-exact-decode-launches) |
| `STRIX_DS41_OVERLAY` | deepseek | auto | [ds-packs](optimizations.md#ds-packs) |
| `STRIX_DS41_PREFETCH_ROW_MB` | deepseek | auto | [ds-prefetch](optimizations.md#ds-prefetch) |
| `STRIX_DS41_PREFETCH_SIDE` | deepseek | auto | [ds-prefetch](optimizations.md#ds-prefetch) |
| `STRIX_DS41_PREPARED_LOAD` | deepseek | ON: discover matching local pack | [ds-prepared-load](optimizations.md#ds-prepared-load) |
| `STRIX_DS41_PREPARED_ROOT` | deepseek | ON: discover matching local pack | [ds-prepared-load](optimizations.md#ds-prepared-load) |
| `STRIX_DS41_READY_LEAN` | deepseek | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_DS41_READY_READS_ONLY` | deepseek | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_DS41_RTX_BASELINE` | deepseek-rtx | fast path | [rtx-kernels](optimizations.md#rtx-kernels) |
| `STRIX_DS41_SERVE_SCHED` | deepseek | auto | [ds-scheduling](optimizations.md#ds-scheduling) |
| `STRIX_DS41_SERVE_SERVICE_MS` | deepseek | 250 ms soft target; short-first on | [deepseek-v41-pair-scheduler](engine/serving/serving.md#deepseek-v41-pair-scheduler) |
| `STRIX_DS41_SERVE_SHORT_FIRST` | deepseek | 250 ms soft target; short-first on | [deepseek-v41-pair-scheduler](engine/serving/serving.md#deepseek-v41-pair-scheduler) |
| `STRIX_DS41_SERVE_WARM` | deepseek | auto | [ds-warm](optimizations.md#ds-warm) |
| `STRIX_DS41_SHARED_DOWN_FUSED` | deepseek | auto | [ds-shared-down](optimizations.md#ds-shared-down) |
| `STRIX_DS41_SHARED_GROUPED` | deepseek | ON | [ds-shared-grouped](optimizations.md#ds-shared-grouped) |
| `STRIX_DS41_TOUCH_BATCH` | deepseek | on | [ds-exact-decode-launches](optimizations.md#ds-exact-decode-launches) |
| `STRIX_DS41_TRELLIS` | deepseek | auto | [ds-packs-trellis](optimizations.md#ds-packs-trellis) |
| `STRIX_DS41_TRELLIS_DENSE_FULL` | deepseek | auto | [ds-packs-trellis](optimizations.md#ds-packs-trellis) |
| `STRIX_DS41_TRELLIS_HEADOUT` | deepseek | auto | [ds-packs-trellis](optimizations.md#ds-packs-trellis) |
| `STRIX_DS41_TRELLIS_MIX` | deepseek | auto | [ds-packs-trellis](optimizations.md#ds-packs-trellis) |
| `STRIX_DS41_TRELLIS_OA_GROUP` | deepseek | auto | [ds-trellis](optimizations.md#ds-trellis) |
| `STRIX_DS41_TRELLIS_OB_READY` | deepseek | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_DS41_TRELLIS_ROW_VIEWS` | deepseek | on | [ds-exact-decode-launches](optimizations.md#ds-exact-decode-launches) |
| `STRIX_DS41_TRELLIS_SITES` | deepseek | auto | [ds-packs-trellis](optimizations.md#ds-packs-trellis) |
| `STRIX_EAGER_CODE_LOAD` | qwen | ON | [eager-code-object-loading](optimizations.md#eager-code-object-loading) |
| `STRIX_EMBED_BUILD_INFO` | qwen | ON | [configured-build-evidence](operations/startup-preflight.md#configured-build-evidence) |
| `STRIX_GLM_ATTN` | glm | fast | [glm-attn-fast](optimizations.md#glm-attn-fast) |
| `STRIX_GLM_BASELINE` | glm | fast path | [glm-identity](optimizations.md#glm-identity) |
| `STRIX_GLM_BOUNDARY` | glm | 1 | [glm-xfer-boundary-fused](optimizations.md#glm-xfer-boundary-fused) |
| `STRIX_GLM_HC_NORM` | glm | 1 (set by the rank app) | [glm-hc-norm-fused](optimizations.md#glm-hc-norm-fused) |
| `STRIX_GLM_HC_PRE_V` | glm | 7 | [glm-hc-pre-v7](optimizations.md#glm-hc-pre-v7) |
| `STRIX_GLM_HC_SITE` | glm | 1 (engine default) | [glm-hc-site](optimizations.md#glm-hc-site) |
| `STRIX_GLM_KDA_LANES` | glm | on (2ct at one row, cqst at 2..8 rows) | [glm-kda-tree](optimizations.md#glm-kda-tree) |
| `STRIX_GLM_KDA_TREE` | glm | on (2ct at one row, cqst at 2..8 rows) | [glm-kda-tree](optimizations.md#glm-kda-tree) |
| `STRIX_GLM_MAX_ROWS_AUTO` | glm | 8192 for two-rank decode prompt prefi... | [glm-max-rows-auto](optimizations.md#glm-max-rows-auto) |
| `STRIX_GLM_MOE` | glm | fast | [glm-moe-fast](optimizations.md#glm-moe-fast) |
| `STRIX_GLM_POST` | glm | fast | [glm-post-exchange-fast](optimizations.md#glm-post-exchange-fast) |
| `STRIX_GLM_SERVE_G2` | glm | on (with MTP) | [glm-serve-g2](optimizations.md#glm-serve-g2) |
| `STRIX_GLM_STEP_FLAG` | glm | 1 (engine default) | [glm-step-flag](optimizations.md#glm-step-flag) |
| `STRIX_GLM_TIERB` | glm | fused | [glm-attn-tierb-fused](optimizations.md#glm-attn-tierb-fused) |
| `STRIX_LM_HEAD_COHORT_LEAN` | qwen | 1 | [qwen-lm-head-cohort-lean](optimizations.md#qwen-lm-head-cohort-lean) |
| `STRIX_LM_HEAD_ROWS_LEAN` | qwen | 2 | [qwen-lm-head-rows-lean](optimizations.md#qwen-lm-head-rows-lean) |
| `STRIX_MINIMAX_GU4_SCALAR` | minimax | ON (build time) | [mm-gu4](optimizations.md#mm-gu4) |
| `STRIX_MMX_CRC` | minimax | deferred | [mm-xfer-crc-deferred](optimizations.md#mm-xfer-crc-deferred) |
| `STRIX_MMX_MOE_DOWN_SPLIT` | minimax | 1 | [mm-moe-down-split](optimizations.md#mm-moe-down-split) |
| `STRIX_MMX_MOE_GU_INTERLEAVE` | minimax | 1 | [mm-moe-gu-interleave](optimizations.md#mm-moe-gu-interleave) |
| `STRIX_MMX_MOE_PIPE` | minimax | 2 | [mm-moe-pipe](optimizations.md#mm-moe-pipe) |
| `STRIX_MMX_MOE_SINGLE` | minimax | 1 | [mm-moe-single](optimizations.md#mm-moe-single) |
| `STRIX_MMX_MOE_TIGHT_GRID` | minimax | 1 | [mm-moe-tight-grid](optimizations.md#mm-moe-tight-grid) |
| `STRIX_MMX_MOE_WIDE` | minimax | 2 | [mm-moe-wide](optimizations.md#mm-moe-wide) |
| `STRIX_MM_ATTEND_DIET` | minimax | 1 | [mm-attend-diet](optimizations.md#mm-attend-diet) |
| `STRIX_MM_BASELINE` | minimax | fast path | [mm-moe-wide](optimizations.md#mm-moe-wide) |
| `STRIX_MM_INDEX_CONC` | minimax | 1 | [mm-index-conc](optimizations.md#mm-index-conc) |
| `STRIX_MM_LOOKUP` | minimax | 1 | [mm-serve-lookup](optimizations.md#mm-serve-lookup) |
| `STRIX_MM_MARKOV` | minimax | cand, C 4096, H 8192, top-C v2 | [mm-markov-cand](optimizations.md#mm-markov-cand) |
| `STRIX_MM_MARKOV_C` | minimax | cand, C 4096, H 8192, top-C v2 | [mm-markov-cand](optimizations.md#mm-markov-cand) |
| `STRIX_MM_MARKOV_H` | minimax | cand, C 4096, H 8192, top-C v2 | [mm-markov-cand](optimizations.md#mm-markov-cand) |
| `STRIX_MM_MARKOV_TOPC` | minimax | cand, C 4096, H 8192, top-C v2 | [mm-markov-cand](optimizations.md#mm-markov-cand) |
| `STRIX_MM_SELECT_ONCE` | minimax | 1 | [mm-select-once](optimizations.md#mm-select-once) |
| `STRIX_MM_SERVE_GAP_MS` | minimax | interleave for pre-existing decoders;... | [minimax-serve-strix-minimax-serve](engine/serving/serving.md#minimax-serve-strix-minimax-serve) |
| `STRIX_MM_SERVE_NEST` | minimax | ON; GPU measured | [minimax-serve-strix-minimax-serve](engine/serving/serving.md#minimax-serve-strix-minimax-serve) |
| `STRIX_MM_SERVE_SCHED` | minimax | interleave for pre-existing decoders;... | [minimax-serve-strix-minimax-serve](engine/serving/serving.md#minimax-serve-strix-minimax-serve) |
| `STRIX_MM_SERVE_SERVICE_MS` | minimax | interleave for pre-existing decoders;... | [minimax-serve-strix-minimax-serve](engine/serving/serving.md#minimax-serve-strix-minimax-serve) |
| `STRIX_MM_SERVE_SHORT_FIRST` | minimax | interleave for pre-existing decoders;... | [minimax-serve-strix-minimax-serve](engine/serving/serving.md#minimax-serve-strix-minimax-serve) |
| `STRIX_MM_SERVE_TIMELINE` | minimax | interleave for pre-existing decoders;... | [minimax-serve-strix-minimax-serve](engine/serving/serving.md#minimax-serve-strix-minimax-serve) |
| `STRIX_MM_XB_INPLACE` | minimax | 1 | [mm-xb-inplace](optimizations.md#mm-xb-inplace) |
| `STRIX_MM_XB_POSTILP` | minimax | 2 | [mm-xb-postilp](optimizations.md#mm-xb-postilp) |
| `STRIX_MM_XB_PUBLISH` | minimax | 1 | [mm-xb-publish](optimizations.md#mm-xb-publish) |
| `STRIX_MM_XB_WAIT` | minimax | 1 | [mm-xb-wait](optimizations.md#mm-xb-wait) |
| `STRIX_MM_XB_WARM` | minimax | 8 | [mm-xb-warm](optimizations.md#mm-xb-warm) |
| `STRIX_MM_XB_WARM_O` | minimax | 8 | [mm-xb-warm-o](optimizations.md#mm-xb-warm-o) |
| `STRIX_MODEL_REGISTER_MLOCK` | qwen | ON when RLIMIT_MEMLOCK covers the reg... | [registered-page-lock](optimizations.md#registered-page-lock) |
| `STRIX_PLE_PREISSUE` | qwen | ON | [qwen-ple-preissue](optimizations.md#qwen-ple-preissue) |
| `STRIX_PLE_RECLAIMABLE` | qwen | auto | [qwen-l1](optimizations.md#qwen-l1) |
| `STRIX_PLE_REWARM` | qwen | ON (bounded) | [serving-ple-table-re-warm](optimizations.md#serving-ple-table-re-warm) |
| `STRIX_PLE_ROLLING` | qwen | table-backed QSA MTP pool and rolling... | [qwen-yarn-mtp](optimizations.md#qwen-yarn-mtp) |
| `STRIX_PLE_STAGE_SERVE` | qwen | ON | [prompt-group-ple-gather-staging](optimizations.md#prompt-group-ple-gather-staging) |
| `STRIX_PLE_WARM_PARALLEL` | qwen | ON | [qwen-ple-warm-parallel](optimizations.md#qwen-ple-warm-parallel) |
| `STRIX_PLE_WINDOW_WARM` | qwen | ON | [qwen-ple-warm](optimizations.md#qwen-ple-warm) |
| `STRIX_PREFLIGHT` | qwen | ON | [operations/startup-preflight.md](operations/startup-preflight.md) |
| `STRIX_PROMPT_BATCH_AUTO` | qwen | ON | [prompt-group-size-by-load](optimizations.md#prompt-group-size-by-load) |
| `STRIX_PROMPT_BATCH_AUTO_QUEUE` | qwen | ON | [prompt-group-size-by-load](optimizations.md#prompt-group-size-by-load) |
| `STRIX_QSA_ATTEND_BATCH` | qwen | 1 | [qwen-qsa-attend-batch](optimizations.md#qwen-qsa-attend-batch) |
| `STRIX_QSA_INDEX_ROWS` | qwen | 1 | [qwen-qsa-index-rows](optimizations.md#qwen-qsa-index-rows) |
| `STRIX_QSA_NORM_GROUP` | qwen | 1 | [qwen-qsa-norm-group](optimizations.md#qwen-qsa-norm-group) |
| `STRIX_QSA_SCORE_PARK` | qwen | 1 | [qwen-qsa-score-park](optimizations.md#qwen-qsa-score-park) |
| `STRIX_QSA_SCORE_ROWS64` | qwen | 1 (from 65,536 blocks; per-group 64-r... | [qwen-qsa-score-rows64](optimizations.md#qwen-qsa-score-rows64) |
| `STRIX_QSA_UNION_MERGE` | qwen | 1 (from 49,152 blocks; range form below) | [qwen-qsa-union-merge](optimizations.md#qwen-qsa-union-merge) |
| `STRIX_QSA_WINDOW_ATTN_V2` | qwen | 1 | [qwen-qsa-window-grouped-score](optimizations.md#qwen-qsa-window-grouped-score) |
| `STRIX_QWEN_ALLON` | qwen | OFF (opt-in: STRIX_QWEN_ALLON=1 / exa... | [qwen-all-on-composition-profile](optimizations.md#qwen-all-on-composition-profile) |
| `STRIX_QWEN_TAIL_LARGE` | qwen | tail-large | [qwen-rtx-tail-large](optimizations.md#qwen-rtx-tail-large) |
| `STRIX_RELEASE_EVENTS` | qwen | ON | [operations/startup-preflight.md](operations/startup-preflight.md) |
| `STRIX_ROUTER_STAGE_ACT` | qwen | ON | [qwen-router](optimizations.md#qwen-router) |
| `STRIX_SERVE_HEALTH_RECORD` | qwen | ON | [operations/service-health.md](operations/service-health.md) |
| `STRIX_SERVE_OBSERVE_EXPIRY_S` | deepseek | 60 seconds | [ds-observation](optimizations.md#ds-observation) |

### Debug only (2)

| Toggle | Engine | Default | Details |
|---|---|---|---|
| `STRIX_DS41_OB_READY_EPILOGUE` | deepseek | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_SERVE_BACKEND_PROBE` | qwen | ON | [operations/service-health.md](operations/service-health.md) |
