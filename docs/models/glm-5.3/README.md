# GLM-5.3 Flash: the USB4 pair and one RTX PRO 6000

[English](README.md) · [Español](README.es.md) · [Main README](../../../README.md)

GLM-5.3 Flash, EXL3 4.05 bpw. On two Strix Halo nodes (TP2 over USB4) it is Production: R1b decode identity by
default, exact MTP self-speculation by default, exact R1 chunked prefill. On one RTX PRO 6000 (Windows) it is an opt-in
draft: exact prefill and greedy decode with the routed experts streamed through the tierstream device-gated cache.
This page keeps every published measurement with its build and conditions and the launch commands. How the tables are
read and what "exact" covers: [methodology](../../methodology.md). Build instructions (trellis library included):
[getting started](../../getting-started/README.md); the pair itself: [two nodes over USB4](../../pair.md).

<a id="glm-pair"></a>

## Run: the pair

Optional GLM token sharding is a qualified FAST memory profile. It is off by default; `STRIX_GLM_KV_SHARD=auto`
(with `STRIX_GLM_PF_PIPELINE=1`) selects sharding only after all permitted replicated scratch/window candidates
fail startup admission on either rank;
`=1` forces sharding and `=0` forces replication. It requires one request,
`STRIX_GLM_PF_PIPELINE=1`, chunked prefill and no MTP or speculation. FAST sharding changes
the summation order, so it is not bitwise equal to the default replicated exact profile.
See the [KV storage and token-sharding guide](../../engine/kv-sharding.md) for the layout,
measured context limits, timings, verification steps and trade-offs. The independent transport
experiment `STRIX_XPORT_SMALL_SPIN=1` enables bounded small-message progress; unset or `0`
restores prior progress after restart. See [feature contracts](../../optimizations.md#kv-cache-sharding-and-shared-transport-progress).

Prefill capacity is chosen at startup. `STRIX_GLM_PF_SCRATCH` (`auto` default, `full`, `routed`) and
`STRIX_GLM_PF_ATTN_QWIN` (`auto` default, or 0..65536 query rows; `0` = the whole chunk) pick the prefill scratch
layout and the attention query window. `auto` tries, in order, full scratch with whole-chunk attention, routed
scratch with whole-chunk attention, then routed scratch with a 2048-row query window; a refusal on either rank
rejects a candidate, and admission counts the shared install/staging bound on device and host. Routed scratch
drops only the unused legacy grouped-MoE arena; the query window bounds attention scratch, while chunk size and
per-row arithmetic stay the same. An explicit value pins its axis. The fallback needs the internal startup vote;
with an external exchange the initial layout is kept. Each rank prints one
`glm capacity: rank=... scratch requested=... resolved=...; qwin requested=... resolved=...; kv=... reason=...`
line. Other values are refused at startup. Restart both ranks after a change. Selection is covered by host
admission-matrix tests; native byte, first-token-time and capacity figures are not published for this control.

Prefill sequence parallelism, `STRIX_GLM_PF_SEQPAR` (`auto` default, `1`, `0`): on chunks with an even number of
rows (at least 2), each rank runs residual/mHC/norm on its half of the rows and the halves are joined by an
overlapped all-gather that reuses the exchange CRC/drain lifecycle; with `STRIX_GLM_PF_PIPELINE=2` and rows a
multiple of 4, each half is split into two tiles. Odd and single-row chunks take the legacy path. It needs two
ranks with a TP2 communicator, no external exchange and no KV sharding: `auto` declines with a stderr line naming
the reason, and `1` without those conditions is refused at startup. The ranks compare their policy and refuse a
mismatch. `0` restores the legacy path. Restart both ranks after a change.

`STRIX_GLM_PF_AUX_COMBINE` is on by default (unset or `1`); `=0` restores the old path.
`STRIX_GLM_PF_IN_TOKEN` (token-ordered MoE gate/up input transform, bitwise) is on by default (unset or `1`);
`=0` restores the tile-ordered transform.
`STRIX_GLM_PF_AUX_INPUT`, `STRIX_GLM_PF_AUX_ACT`, `STRIX_GLM_PF_AUX_DENSE`, and
`STRIX_GLM_PF_AUX_NORM` are opt-in (`=1` enables, unset or `=0` disables). These switches accept only
`0` or `1`.

Prefill kernel variant selectors (`STRIX_GLM_PF_HC`, `_MLA`, `_IDX`, `_SELECT`, `_OBAR`, `_MLA_GEMM`, `_SCORES`,
`_KDA_SCAN`, `_DENSE_ROWS`, `_DENSE_FORM`) share one table used by the engine and by `--print-effective-config`. Unset selects
the default; any set value, empty included, must be a listed variant or startup is refused with
`ENV='v' is not a known variant (allowed: ...; unset = D)`. The effective-config line reports the variant that
runs.

| Variable | Allowed values (default first) |
|---|---|
| `STRIX_GLM_PF_HC` | 2, 1 |
| `STRIX_GLM_PF_MLA` | 2, 1 |
| `STRIX_GLM_PF_IDX` | 3, 2, 1 |
| `STRIX_GLM_PF_SELECT` | 2, 1 |
| `STRIX_GLM_PF_OBAR` | 10, 7, 6, 5, 4, 2 |
| `STRIX_GLM_PF_MLA_GEMM` | 3, 2, 1 |
| `STRIX_GLM_PF_SCORES` | 7, 6, 5, 4, 3, 2 |
| `STRIX_GLM_PF_KDA_SCAN` | 5, 3, 1 |
| `STRIX_GLM_PF_DENSE_ROWS` | 4, 3, 2, 1 |
| `STRIX_GLM_PF_DENSE_FORM` | 0, 1, 2, 3, 4, 5, 6, 7 (physical form of version 4; 0 = automatic per shape) |

With `STRIX_XPORT_TRACE=1`, the prefill exchange prints
`xport prefill bytes=... copied=... messages=... reuse_dependencies=...` when it closes. `reuse_dependencies`
(formerly `exposed_wait`) counts buffer-reuse event edges, two per traced reuse; it is not elapsed time.

Run once per node with `R` set to 0 or 1 and the same tokenized prompt on both:

```sh
./build/strix-glm5-rank MODEL_DIR --world 2 --rank R --transport usb4-auto --prompt PROMPT.u32 --decode 64
```

Default R1b decode uses exact MTP self-speculation and R1 prompt prefill; `--mtp 0` runs plain decode
with the same tokens. Both ranks start the prefill timer after a pair barrier, so the two ranks report the same prefill
wall and neither includes the peer's load time (the barrier wait is printed; `STRIX_GLM_PF_TIMER_BARRIER=0` restores the
old start). `strix-tokenize MODEL_DIR encode|plain|decode FILE` runs the engine's tokenizer (the
same `MODEL_DIR/tokenizer.json` the server loads) on the command line. `--identity r1` selects a different reference identity. Prefill automatically uses
8192 rows if both ranks admit them, otherwise 4096. See the [resolver](../../../apps/glm_rank_defaults.hpp),
[model families](../model-families.md) and [qualification/performance](../../performance.md).

Prefill MLA switches: `STRIX_GLM_PF_SCORES=6` and `STRIX_GLM_PF_OBAR=7` (o-bar v7; `=5` v5) restore earlier stages; default o-bar v10 stages 2 k per batch with the same chains (pair logits bitwise equal to v7 at 8K and 32K; o-bar time 7-9 % lower, TTFT 8K -0.7 %, 32K -0.3 %). `STRIX_GLM_PF_MLA_GEMM=2` restores GEMM v2 (`=1` legacy); default v3 keeps the ascending-k FMA chains (pair logits and state bitwise equal to v2 at 8K and 32K; MLA GEMM time 41-43 % lower, TTFT 8K -1.9 %, 32K -2.3 %).

<a id="glm-serve"></a>

## Serve: the pair

`strix-glm5-serve` is the HTTP server (OpenAI `/v1/chat/completions` and `/v1/completions` with SSE streaming,
`/v1/models`, `/health`). Start rank 1, then rank 0; both take the same engine options (the hello refuses a mismatch)
and read the tokenizer from `MODEL_DIR/tokenizer.json`:

```sh
./build/strix-glm5-serve MODEL_DIR --world 2 --rank 1 --transport usb4-auto
./build/strix-glm5-serve MODEL_DIR --world 2 --rank 0 --transport usb4-auto --host 0.0.0.0 --port 8080 --model-id glm-5.3-flash
```

Defaults: 8 request slots (`--slots`, at most 8), 8192 positions per slot (`--capacity`), exact MTP self-speculation
with one request in flight, batched MTP verify with more (`STRIX_GLM_SERVE_G2=0` = batched plain steps), decode-first
chunked prefill of 512 tokens while others decode (`--prefill-chunk 0` = whole prompts). The chunk is resolved by the same route policy as the CLI: both ranks print `effective serve-prefill-chunk=...` with the dense core its calls take, and the reason whenever a chunk declines the large-call core. `--serial` admits one
request at a time; `--api-key-file F` requires a bearer key. `--selftest N` on rank 0 (no HTTP) runs N built-in
prompts together and each alone and reports EXACT/FAIL with tok/s per concurrency level.

`--prefix-cache-gib G` sets the host-RAM prompt snapshot LRU to G GiB per rank (default 4;
`--prefix-cache-gib 0` turns it off; set the same budget on both ranks). Stores require both ranks to snapshot successfully and each rank to have
at least 16 GiB MemAvailable; missing memory readings and oversize snapshots refuse the store. A hit restores
the longest strict prompt-prefix key, then prefills the suffix through the usual chunk scheduler. With MTP,
the key includes the prompt's first greedy output token and restores at the original prompt frontier.
The contract is exact trunk restoration and unchanged emitted tokens; MTP rows retain the existing
chunk scheduling rule. `strix.prefix_cache` reports `hit_tokens`, `prefill_tokens`, and `restore_ms` when
the cache is on. `--selftest N --selftest-prefix` compares two-turn streams against cleared-cache replays
and requires hits; `--selftest-prefix-long L` uses first turns of about L tokens and prints the turn-2
admit-to-ready time from the cache and cold. The other selftests also restore repeated prompts from the
cache; pass `--prefix-cache-gib 0` for cache-off figures.

The memory watchdog's default RSS ceiling is 8 GiB plus the per-rank prefix-cache
budget, so retained snapshots do not look like an engine leak. `STRIX_GLM_WD_RSS_GIB`
overrides that total (0 disables the RSS check); RAM headroom, swap and GTT limits
are unchanged. Disabling the cache restores the 8 GiB default.

## Run: one RTX PRO 6000

Windows only, opt-in: build with `pwsh src/glm/rtxpf/producer/build.ps1` against a tierstream source export, build
the unit file once with `glm_unit_file.exe`, then run `decode_qualification_window.py --full-card --allow-gpu`. The service itself also takes
`--dgpu auto|<uuid>` to pick the RTX PRO before CUDA starts (`off` is refused: no CPU path). Every
step, the pinned tierstream commit and the options (`--profile`, `--print-effective-config`, `--vram-total-gib`) are
in the [GLM RTX guide](../../../src/glm/rtxpf/README.md).

<a id="how-fast"></a>

## Measured results: the pair

Sources: [performance records](../../performance.md) and [current GLM defaults/evidence](../../optimizations.md#glm).

EXL3 4.05 bpw, two Strix Halo nodes TP2 over USB4. Production engine. R1b is the default decode identity; prompts use exact R1 chunked prefill;
decode uses exact MTP self-speculation by default (`--mtp 0` = plain decode).
`--identity r1` restores R1 throughout. The historical R1 measurements below keep their original identity.

Prompt prefill in two-rank decode runs defaults to 8192 rows when both ranks admit the calibrated load
and all prefill owners; otherwise both use 4096. This applies to production decode with MTP
self-speculation and plain chunked prefill inside `--decode` runs. Selection uses the load admission's RAM
and GTT headroom, staging, reserves, whole-tensor reads and bounce buffer, plus prefill and communication
growth and the device-free guard.
The largest of the load margin, prefill margin and 8 GiB is retained. Normal two-rank Comm decode reuses
its existing configuration exchange and mismatch vote for mode agreement and the minimum proposed rows.
Fixed modes preserve their payload bytes and tag order; mixed auto/fixed modes refuse on both ranks.
Auto communication pools start at 4096 and grow only after agreement on 8192. A fresh admission
check after growth uses one auto-only minimum vote and shrinks both pools on a 4096 fallback.
Two-rank Comm runs without that handshake (prefill-only, reference gates and batch-only) use 4096 in auto
mode, with source `auto-4096-no-handshake`, without a control exchange. Explicit row counts must match.
`--pf-ttft` and gate harnesses keep 4096 unless `--max-rows 8192` is given explicitly.
Fixed row mismatches there cannot be checked at startup without
changing the existing protocol. External `--xch xfer` auto uses 4096 without an agreement exchange.
Reference prefill (`--prefill ref`, automatic prefill under `r1b-all` or `--idcheck` with decode) uses
4096 in auto mode (`auto-4096-ref`) without a row-count agreement exchange.
The exact prefill route preserves its arithmetic across chunk sizes; measured TTFT on the chunked-prefill
route at 8192 rows improves 4 % at 8K and 116.3 to 109.9 s (5.5 %) at 32K, using about 7.3 GiB more
GPU memory per node.
`STRIX_GLM_MAX_ROWS_AUTO=0` restores fixed 4096; `--baseline` keeps 1024, and explicit `--max-rows` keeps its value.
`--print-effective-config` reports the host-independent candidate and startup policy; stderr reports
admission headroom and the final value with its source. See the [catalog](../../optimizations.md#glm-max-rows-auto).
Startup admission includes the prefill path and its workspace for the selected rows and tier, so a rank
that cannot fit them refuses before loading weights. Post-load admission uses the same
budget. Indexer score keys are sized per chunk, not per context capacity. Stderr reports the path,
workspace, install/staging, late safety and startup reserve terms in GiB.

| Mode | Prompt / context | Measured | Build / exactness |
|---|---|---:|---|
| Greedy decode, R1 | 31-token chat prompt + 64 generated tokens | **45.29 ms/token = 22.1 tok/s**; **51 %** of calculated 23.2 ms floor | `1836c99c`; tokens equal the reference |
| Batch decode, R1 | Eight independent rows, benchmark context unspecified | **122.0 ms/step = 65.5 tok/s aggregate** | `1836c99c`; tokens equal the earlier real-pair reference |
| Exact chunked prefill | 1K / 8K / 32K benchmark prompts, kind unspecified | TTFT **4.268 / 34.70 / 144.75 s** (240 / 236 / 226 tok/s) | `875ca821`, integrated in `41d8cfdc`; both ranks' logits bitwise equal the previous exact prefill |
| R1b-fast, now the default decode identity | Same chat benchmark, 96-token alternating comparison | **41.93–41.96 ms/token**, vs R1 47.33–47.48 (**−11.5 %**, **55 %** of 23.2 ms floor) | `8877de1d`; stable within each identity, not bitwise to R1 |
| MTP self-speculation (default) | Chat / code / JSON prompts, 64–128 generated tokens, rank 0 | **26.98 / 22.46 / 24.98 ms/token = 37.1 / 44.5 / 40.0 tok/s** (1.50× / 1.79× / 1.62× plain 40.3–40.6 ms/token); TTFT 8K **31.13 s** (+2.9 %), 32K **140.9 s** (+8.7 %: MTP cache fill of the extra layer) | Full token vectors and prompt logits equal `--mtp 0` on 7 prompts, both ranks; plain TTFT unchanged |
| Exact chunked prefill, X2 linear core (default) | 1K / 8K / 32K benchmark prompts | TTFT **3.78 / 27.93 / 120.26 s** (271 / 293 / 272 tok/s); same run with `STRIX_GLM_PF_EXACT_X=0`: 1K 3.84 s, 8K 29.59 s (**−5.6 %** at 8K) | logits bitwise equal the X=0 path on both ranks; decode anchors and speculative oracle unchanged |
| Exact chunked prefill, X2 linear core + MoE auto (default) | 1K / 8K / 32K benchmark prompts | TTFT **3.74 / 26.85 / 113.73 s** (274 / 305 / 288 tok/s); same run with `STRIX_GLM_PF_EXACTMOE_V=3`: 1K 3.76 s, 8K 28.17 s (**−4.7 %** at 8K) | outputs bitwise equal the v3 path on both ranks (internal FP32 NaN payloads may differ only in the FP16 overflow case); decode anchors and speculative oracle unchanged |
| Exact chunked prefill, X3 MoE core and row buckets (PR #183) | 1K / 8K benchmark prompts, 8,192-row chunks | TTFT **3.18 / 20.32 s**; same run with `STRIX_GLM_PF_EXACTMOE_V=7`: 1K 3.75 s, 8K 25.24 s (**−15 % / −19.5 %**) | two pairs per arm; R1 bitwise (same DOT2 chains, reduce4 and split order) |
| Exact chunked prefill, X3 tail buckets + dense X3 + device MoE plan (PR #215) | 8K / 32K benchmark prompts, 8,192-row chunks | TTFT **17.77 / 76.59 s** (461 / 428 tok/s); same run with the previous defaults (PR #183): 20.22 / 86.66 s (**−12.1 % / −11.6 %**) | one pair per arm; per-rank logits equal the public anchors at 8K and 32K in both arms (same DOT2 chains, reduce4 and split order) |
| Exact chunked prefill, attention kernels: MLA scores v5 / o-bar v7 / absorb-out GEMM v2, indexer scores v3 / select v2 (measured on the PR #183 defaults, before PR #215) | 8K / 32K benchmark prompts, 8,192-row chunks | 32K TTFT **−6.8 %** (−7.2 s, mean of both run orders: 105.44 → 103.66 s previous kernels first, 110.31 → 97.66 s new kernels first); 32K attention kernels on rank 0 **18.6 → 10.3 s**; 8K **25.23 → 23.62 s (−6.4 %)**, two pairs, without the GEMM v2 | logits bitwise equal the previous kernels on both ranks; same binary, previous kernels via their switches; measured before the X3 MoE core (PR #183) |
| Exact chunked prefill, KDA scan v3 + MLA scores v6 (measured on f7e063b7e; pair prefill code unchanged up to fb66457e4) | 8K / 32K benchmark prompts, 8,192-row chunks | 32K TTFT **−1.5 %** (−1.04 s, mean of both run orders: 68.81 → 67.26 s new kernels first, 68.71 → 68.17 s previous kernels first); 32K rank-0 kernels: KDA scan **2.18 → 1.43 s**, MLA scores **2.96 → 2.62 s**; 8K KDA scan alone 16.63 → 16.34 s (−1.7 %) | logits bitwise equal the previous kernels on both ranks; same binary, previous kernels via `STRIX_GLM_PF_KDA_SCAN=1` `STRIX_GLM_PF_SCORES=5`; rank 0 ran under the kernel tracer in every arm |
| Exact chunked prefill, KDA scan v5 (on the PR #254 defaults) | 8K benchmark prompt, 8,192-row chunks | TTFT **16.60 vs 16.60 s** (mean of 8 arms, both run orders: neutral); rank-0 scan **0.355 → 0.340 s**; scan per layer 1.23–1.25 → 1.18–1.20 µs/row (−3.7 %) | logits bitwise equal v3 on both ranks; same binary, v3 via `STRIX_GLM_PF_KDA_SCAN=3`; rank 0 under the kernel tracer; the fourth arm of each pair ran ~5 % slower on every kernel in both orders |
| Exact chunked prefill, pair levers: fused mHC expand + norm and staged prefill exchange (current defaults) | 1K / 8K benchmark prompts, 8,192-row chunks, warm | TTFT **2.83 / 15.27 s** with every pair lever on; staged exchange ABBA: 1K 2.856 / 2.860 → 2.832 / 2.838 s, 8K 15.43 / 15.41 → 15.28 / 15.25 s (**−0.8 % / −0.9 %**, every staged arm faster); fused mHC expand + norm ABBA ×2: **−1.1 % / −1.3 %** | prefill anchors exact 16/16; logits bitwise equal each lever off (`STRIX_GLM_PF_STAGED=0`, `STRIX_GLM_PF_HC_FUSE=0`) |
| Decode fused HC norm (default) | C1 MTP decode and R8 batch, ABA ×2 | **−2.3 %** C1 MTP, **−3.1 %** R8 | tokens and R8 equal the reference; `STRIX_GLM_HC_NORM=0` restores the unfused finish |
| Decode one-launch mHC site (default) | C1 decode `--mtp 0` and R8 batch, fresh-process ABBA | **−1.45 %** C1, R8 unchanged | tokens and R8 equal the reference on both ranks; `STRIX_GLM_HC_SITE=0` restores the three launches |
| Decode step flag checked with the head (default) | C1 decode `--mtp 0` and R8 batch, fresh-process ABBA | **−0.2 %** C1, R8 within noise | tokens and R8 equal the reference on both ranks; `STRIX_GLM_STEP_FLAG=0` restores the mid-step host sync |
| Latest decode, all defaults (#332 `ec76191f2`) | C1 decode `--mtp 0` and R8 batch, two runs | **39.00–39.03 ms/token** (25.6 tok/s, 57 % of the 44.7 tok/s ceiling); R8 **112.14–112.31 ms/step** (71.3 tok/s aggregate, calculated) | 24/24 exact on both ranks |
| Latest exact chunked prefill, all defaults (#300 `32ec66bd3`) | 1K / 8K / 32K benchmark prompts, 8,192-row chunks | TTFT **2.83 / 15.15 / 61.96 s** (541 / 529 tok/s at 8K / 32K); MoE remainder tiles 8K 15.35 → 15.15 s (−1.3 %, alternating pairs), 32K −1.2 % over both run orders (62.58 → 61.96 s, −1.0 %, in one; −1.4 % in the other) | logits bitwise on both ranks |
| Served, `strix-glm5-serve --selftest 8` (defaults) | Concurrent streams C1 / C2 / C4 / C6 / C8, steady state (all streams decoding); 8 built-in short chat prompts in the GLM chat template, 22–59 prompt tokens (mean ~33), up to 128 greedy tokens each, barrier start, prefix cache off | **C1 40.0 / C2 53.0 (G2 off 37.8) / C4 59.9 (53.6) / C6 62.5 (62.4) / C8 69.5 (69.4) tok/s aggregate**; per stream 40.1 / 26.5 / 15.0 / 10.4 / 8.7; C1 plain (MTP off) 25.30; engine CLI `--batch` R8 69–71 | RESULT EXACT: every stream's tokens equal the same prompt run alone (C=1 plain reference) at every level, G2 on and off |
| Served, decode-first chunked prefill (`--prefill-chunk 512`, default) | 4 prompts of ~1,830 tokens arriving every 16 decode rounds into 4 running streams, 128 generated tokens each, cache off | Longest decode stall **6.65 → 2.67 s (−60 %)** vs `--prefill-chunk 0`; a lone request is prefilled whole | exact |
| Served prompt-prefix cache (default, 4 GiB per rank) | Two-turn conversations, first turn ~4,100 tokens, turn 2 adds 53 tokens; alone and batched, against the same request after clearing the cache | Turn-2 admit-to-ready **8.57–8.98 s → 0.82–0.89 s (~10x)** for the 4,120-token prompt (459–481 → 4,629–5,024 tok/s); restore 9–11 ms; 4,067 of 4,120 prompt tokens restored | tokens equal the cleared-cache replay (8/8; 16/16 on short prompts) on both ranks |
| Exact chunked prefill, MLA scores v7 (on the PR #279 defaults) | 8K / 32K benchmark prompts, 8,192-row chunks | rank-0 scores kernel **2.60 → 2.27 s** at 32K (−12 %) and **0.533 → 0.485 s** at 8K (−9 %); TTFT not slower: 32K 65.68 → 65.54 s (−0.2 %, mean of both run orders: 65.574 → 65.524 new first, 65.778 → 65.549 previous first), 8K 16.23 → 16.12 s (−0.6 %, 4 arms) | logits bitwise equal v6 on both ranks; same binary, v6 via `STRIX_GLM_PF_SCORES=6`; rank 0 under the kernel tracer; the TTFT change is inside the ~1 s run-to-run spread at 32K |
| Exact chunked prefill, dense rows v2 (default; `STRIX_GLM_PF_DENSE_ROWS=1` restores v1) | 1K / 8K benchmark prompts (1,024 / 8,192 tokens), interleaved v2 / v1 arms | TTFT 8K **15.06 → 14.92 s** (−0.9 %, 544 → 549 tok/s), 1K warm **2.80 → 2.75 s** (−1.9 %, 366 → 372 tok/s); every v2 arm faster than every v1 arm; KDA f_b / g_b kernels (128 → 4096) **1.19–1.21×** at 8,192 rows. Roofline: at 14.92 s for 8,192 tokens (549 tok/s) the exact pair prefill runs at about 16 % of its linears-only compute floor (31.8 GFLOP/token on 2 x 54 TF = 2.4 s); the binding limit is the exact kernel schedule rather than memory bandwidth or the cable: the MoE and dense linears run on FP32 VALU tiles to keep the bitwise identity, and this change only shortens the dense-row part of that schedule. | logits and state bitwise equal v1 on both ranks (same per-output ascending FMA chain); device gate 80/80 bitwise; the 16,384-long chains of the legacy mHC path keep the v1 tile |
| Exact chunked prefill, mHC mix collapse + norm in one kernel (default; `STRIX_GLM_PF_MIXNORM=0` restores the two kernels) | 8K / 32K benchmark prompts (8,192 / 32,768 tokens), alternating on / off arms | TTFT 8K **15.07 → 14.95 s** (−0.8 %, 544 → 548 tok/s), 32K **63.23 → 62.23 s** (−1.6 %, 518 → 527 tok/s, against the mean of the bracketing off arms). Roofline: at 14.95 s for 8,192 tokens (548 tok/s) and 62.23 s for 32,768 tokens (527 tok/s) the exact pair prefill runs at about 16 % / 15 % of its linears-only compute floor (31.8 GFLOP/token on 2 x 54 TF = 2.4 s at 8K, 9.6 s at 32K); the binding limit is the exact kernel schedule rather than memory bandwidth or the cable: the MoE and dense linears run on FP32 VALU tiles to keep the bitwise identity, and this change only removes one memory round trip per hyper-connection site. | logits bitwise equal the anchors on the pair at 1K, 8K and 32K; device gate bitwise over 1 to 4096 rows and special values (only propagated NaN payloads may differ) |
| FAST WMMA prefill linear cores (`glm-pf-wmma-x`, opt-in `STRIX_GLM_PF_WMMA=1`) | 8K / 32K benchmark prompts, 8,192-row chunks, fresh process per arm | TTFT **15.45 → 12.94 s (−16.3 %)** at 8K and **62.3 → 51.2 s (−17.8 %)** at 32K (mean of both run orders); rank-0 linear kernels at 8K: MoE 5.15 → 3.42 s, dense X3 2.61 → 1.71 s | FAST, not bitwise the exact prefill: pre-registered quality-parity check passed at 1K / 8K / 32K; off (`STRIX_GLM_PF_WMMA=0` or unset, the default) equals the exact logits anchors on both ranks |
| FAST KDA plain-weight projections on WMMA (`glm-pf-kda-wmma-v1`, on with `STRIX_GLM_PF_WMMA=1`; `STRIX_GLM_PF_KDA_WMMA=0` turns it off) | 8K / 32K benchmark prompts, fresh process per arm, against FAST without it | Prefill 8K **9.25 → 9.05 s (−2.1 %)** (2 alternating pairs), 32K **37.74 → 37.13 s (−1.6 %)** (2 off / 3 on runs; one disturbed run of each arm discarded) | FAST, not bitwise: teacher-forced 3 × 256 positions KL 0.0038 / 0.0004 / 0.0004 nats, top-1 98.0 / 100 / 100 %, NLL change −0.006 / +0.001 / +0.002; off keeps the previous bytes |
| FAST fused MLA/DSA attention core (`glm-pf-attn-wmma`, on with `STRIX_GLM_PF_WMMA=1`; `STRIX_GLM_PF_ATTN_WMMA=0` turns it off) | 8K / 32K benchmark prompts, 8,192-row chunks, fresh process per arm, against FAST without the core | Prefill 8K **10.600 → 9.695 s (−8.5 %)** (2 alternating pairs), 32K **44.30 → 40.26 s (−9.1 %)** (one off / on / off set); layer bench: scores + softmax + o-bar 2.6× / 3.2× faster at the 8K / 32K positions | FAST, not bitwise: prefill-logits KL 7.8e-4 nats at 8K, top-1 and greedy equal over 8 decode steps; exact mode unchanged (logits equal the anchors on both ranks); teacher-forced check over 3 prompts pending |
| Exact chunked prefill, mHC expand + norm two rows per block (default; `STRIX_GLM_PF_HC_EXPAND_ROWS=1` restores one row per block) | 8K benchmark prompt (8,192 tokens), alternating on / off arms, other exact prefill levers of that build enabled | TTFT 8K **13.671 → 13.611 s** (−0.44 %, above the 0.024 s spread within an arm) | logits bitwise equal the anchors on both ranks at 8K; host and device gates bitwise against the one-row kernel |
| Exact / FAST chunked prefill, token-ordered MoE gate/up input transform (default; `STRIX_GLM_PF_IN_TOKEN=0` restores the tile-ordered transform) | 8K / 32K benchmark prompts, on / off arms in both orders, fresh process per arm | FAST 8K TTFT **10.55 → 10.21 s** (median of 4 pairs); exact 8K 13.195 → 12.868 s; FAST 32K 43.49 → 42.39 s (one pair); exact 32K: 56.32 -> 53.41 s and 57.63 -> 55.52 s in two pairs (off first, on first): -2.9 / -2.1 s, mean -2.5 s (-4.5 %), identical full logits files on both ranks (an earlier single pair showed -5.5 s from a slow off arm; it did not reproduce). | logits and state digests identical in every arm on both ranks; full logits files identical in the exact arms |
| R1b teacher-forced agreement | 32 prompts, 256 positions each; chat, code, JSON, math, reasoning, translation | **8,126/8,192 = 99.19 %** top-1; mean KL **~1.8e-3 nats** | `4dad4c19`; free runs diverge on **28/32** prompts within 19–227 tokens |

The R≥3 buffer fix is verified on the real pair. The latest integration uses direct exchange for plain
forwards and exact MoE v3 for prefill; the speculative oracle also passes with drained exchange switching.
Prefill rows are single runs. They replace the earlier synthetic-only result.

R1b changes FP32 summation order and is not bit-identical to R1; which identity is closer to float64 at full depth is
unresolved. HIP graphs are **not implemented**: the latest profile (`41d8cfdc`) keeps the GPU busy **96.3 %** of each forward (was 80.8 %; 909 launches instead of 1,679) and leaves only **2.10 ms idle per forward**
and a calculated recoverable gap of **≤1.8 ms** (a bound, not a measured speedup). Host progress across
exchange boundaries requires segmented capture; device kernels and rank skew take priority.

<a id="glm-rtx-results"></a>

## Measured results: one RTX PRO 6000

**GLM-5.3 Flash on one RTX PRO 6000 (draft, opt-in, Windows).** The trunk, dense layers and shared experts stay
on the card; the 43 x 288 routed experts are served by the [tierstream](https://github.com/hyukudan/tierstream)
device-gated cache, and prefill reads them through the same cache, so the card holds one copy of every expert.
Prefill and decode are exact (bitwise digests of logits and state; prefill state equal across row tiles 2048 / 4096 /
8192). Full card, tile 8192: TTFT **11.4 s** at 8K and **30.5 s** at 32K (first request, mean of two loads; 30.5 s at 32K vs
43.7 s at tile 2048, -30 %). At 32K the prefill is bound by GPU compute: the expert staging (about 128 GB host-to-device
per prompt, 8-10 s of make-resident, about 4 GB/s on average, far below PCIe 5 x16) runs behind the kernels with
0.0-0.2 s of GPU idle, so the -30 % is the larger tile's fewer expert weight decodes per prompt and longer GEMM rows
per expert, not an I/O gain; at 8K 2-4 s of staging stay exposed. (Exact; the decode-residency restore now runs
after the prefill, outside time-to-first-token, only when decode follows); decode **13.25 / 13.52 tok/s** warm at 1K / 8K (3.63 / 4.09 cold) with the current decode kernels (24-token bench, ABBA
means of one load). When the few-row decode kernels were added, decode measured 6.00 / 6.15 tok/s warm (3.75 / 3.70 cold) with them on by
default, from 2.43 / 2.36 tok/s with them off in the same service load (full card; host RAM put 1,098 expert units on
the unit file, so per-token file misses are included; 24-token bench). Tokens, every logits digest and the final
session state are bitwise equal on and off, and the 1K/8K qualification passes with them on. The kernels decode each
weight tile straight into tensor-core fragments and keep the pinned GEMMs' accumulation order; turn them off with
`--decode-row-kernels 0` or `STRIX_GLM_RTX_ROW_KERNELS=0`.
Prefill overlap is on by default: while one layer's kernels run, the next layer's experts are made resident. Prefill
state, tokens and logits are bitwise equal with it off. In same-window comparisons it cuts TTFT to 0.56× (8K) and 0.66×
(32K) of the overlap-off time on the full card. Turn it off with `--prefill-overlap 0`.
Prefill clean eviction is on by default: a decode resident the prefill displaces while no pinned block is free goes
back to the unit file instead of pushing a pinned unit out. On the full card it cuts 8K TTFT by 15 % / 12 % (first /
second request), ties at 32K and reads 60 % fewer unit-file bytes, bitwise equal. Turn it off with
`--prefill-clean-evict 0`.
The default allocates the largest admitted tile up to 8192, with activation + workspace growth over
2048 <= 5% of the expert arena at 2048. A larger tile may add file homes only when the 2048 plan has
<= 15% of routed units on file and the increase is <= 5% of routed units. This admits 8192 at the
measured full-card range (2048 baseline ~1.5-4.6% of routed units on file, ~2.4% more at 8192),
and in plans within these bounds when the plan and growth gates pass; elsewhere the old veto on
extra file units applies.
Full card, 8192 vs 2048 reduced TTFT by 30% at 32K and 9% / 5% at 8K, bitwise equal. This does not
establish the best tile at 8K: in one load pair 4096 was 1.3 / 2.0 s faster (first / second request),
in the other within 0.3 s.
Explicit `--row-tile` bypasses the automatic growth and file-home rules (the plan must still fit),
and omits `row_tiles_tried`; automatic candidates report admission reasons and `file_units_over_2048`.
File reads in flight default to regime auto: every queue-1 three-tier plan with at least 70 % of routed
expert units file-homed is admitted when staging >= 2 and readers fit. At about 76 % file share
(24 GiB VRAM + 32 GiB RAM), auto was never slower in the latest measured decode cell means; one earlier
cell was -3.4 %, traced to drive read phases. At about 61 % (48 GiB + 32 GiB), one earlier night
observed cold-1K decode -32 % against the queue-1 mean; the other cells were +3 % at warm 1K,
+7 % at cold 8K and +1 % at warm 8K. A later set at the same budget reversed the cold-1K result:
auto +8.4 % pooled, every auto arm faster. The 70 % cut remains a conservative empirical cutoff,
so 61 % stays at 1.
Full card (about 1-4 % at tile 2048) tied in TTFT: 8K +0.6..+2 %, 32K equal; it stays at 1.
Staging < 2 or readers that do not fit also fall back to 1. Explicit `auto`
still applies at any file share; `--tier-disk-queue 1` turns auto off.
With 24 GiB VRAM + 32 GiB host RAM (most experts streamed from the unit file on disk) it also runs exact: TTFT
**60.3 s** at 1K and **62.4 s** at 8K with the prefill overlap (63.4 / 73.7 s with it off in the same window); decode
**0.67 tok/s** warm (0.63 cold) with one file read at a time; with the default file reads in flight at this budget
(auto, 76 % of the experts on the unit file) decode is **0.77 / 0.75 tok/s** warm / cold at 1K (mean of two loads,
24-token bench). With the opt-in two-stage file prefetch (`--tier-landing-blocks 16`, three striped
drives) decode was 20-40 % faster than without it in the same windows, identical output (repeats varied by 15-55 %).
Build, unit file and options: [GLM RTX guide](../../../src/glm/rtxpf/README.md).

**RTX prefill into the pair (FAST, off by default).** `strix-glm5-serve --rtx-prefill HOST:PORT --import-listen IP:PORT
--import-peer IP:PORT --import-key FILE --import-model-sha SHA` (every rank `--import-listen`; engines with their MTP
layer only) lets the RTX producer prefill long prompts and hand each rank its state. Both ranks stage the hand-off,
vote on it (the authenticated binding, then both restored) before the last prompt token runs, and any failure before
that vote falls back to the pair's own prefill on both ranks; a failure inside the last-token collectives stops the
server (fatal). `--import-dsa-once` (with the producer's `dsa_once`) sends the DSA state to rank 0 only and relays it
to rank 1 over the pair link (8K: 480 -> 316 MB on the producer link, tokens equal to the version 1 hand-off). Imported prompts carry the FAST identity
`glm-pf-rtx-native-v1` (not bitwise the pair's own prefill; each response's `prefill_route` names it).
`--import-gate` fails unless every routed arm really imported (and a suite with no prompt fails). `--import-model-sha` is
an operator attestation: the receiver compares it with the producer's hand-off header, it is not computed from the
tensors the pair loaded. The import settings (receiver on, `--import-dsa-once`, wire version, the attested model SHA)
must agree on every rank: the hello refuses a mismatch at startup. The control block is version 6: a rank of this build
refuses a rank of an older build at hello.

Measured on the release build (greedy, 128 tokens, cache off, 1 GbE between the RTX PC and the nodes, producer started
per run): time to first token with the import 39.4 s at 1K and 45.6 s at 8K, against 2.83 s and 15.7 s for the pair's
own exact prefill (with MTP: 38.6 vs 2.91 s and 44.5 vs 16.3 s), so the route is not faster on this setup. Almost all of
the imported time is the RTX prefill (38-44 s): that run started the producer without its tierstream cache, so it staged
expert layers from the model shards on every request (the producer README shows the cached launch); the pair-side
cost of an import is about 0.4 s (transfer tail, restore, last-token prefill, commit). Whether it pays depends on the RTX
prefill time for long prompts. With the producer launched with its cache (first request, one run, same release build, 1 GbE): 8K imported 21.0 s
(RTX prefill 19.7 s) against 15.7 s for the pair's own prefill, still not faster.

## Every shipped GLM default

The changelog of every default-on GLM lever with its measured gain, exactness evidence and off switch is in
[measured gains](../../measured-gains.md); the full catalog is [docs/optimizations.md](../../optimizations.md#glm).
GLM RTX build, unit file and options: [GLM RTX guide](../../../src/glm/rtxpf/README.md); native RTX prefill notes:
[docs/models/glm/rtx-prefill-native.md](../glm/rtx-prefill-native.md).
