# Measurement kits: what each one measures and how to run it

[English](benchmarking.md) · [Español](benchmarking.es.md) · [Main README](../README.md)

The public benchmark (`scripts/bench/bench.sh`, [reproduce.md](reproduce.md)) gives one table per engine. The
kits below are the lab's day-to-day measurements: one arm against another, in strict ABAB order, with every run
checked for validity. They live in the repository so they survive any cleanup of the machines that run them.
Nothing in them names a host, a user or a private path: binaries, models and prompts are arguments, and the
second node of a pair is the ssh alias in `KIT_PEER`.

## Shared runner (`scripts/bench/common/`)

| File | Role |
|---|---|
| `ticket.sh OUT KIT [ARGS]` | The one command a GPU queue ticket runs. Budget = `GPU_TURN_MAX` (default 120 s) minus `KIT_MARGIN` (5 s); arms that no longer fit are skipped. Ends with `OUT/summary.txt` (at most 60 lines) and `OUT/.done`, so the caller waits once and reads one file. |
| `runlib.sh` | Sampler, page-cache warm, binary check across the two nodes, budget, `kit_run`, `kit_pair_arm` (rank 1 on the peer, rank 0 here). |
| `run1.sh DIR CAP -- CMD` | One measured process with its receipt, also run on the peer for rank 1. |
| `receipt.py` | Parses one run directory into `receipt.json` and one summary line (rules below). |
| `abab.py` | `order` prints the strict ABAB order; `summary OUT [--metric M]` gives per-arm medians, per-round deltas and identity checks. |
| `serve-decode.sh` | Starts a server (and its rank 1 on the peer), waits until it is ready, runs `tools/serve_bench.py` per decode class, then stops it. `--client-cmd` runs any other client instead. |
| `serve_receipts.py` | Turns served results into receipts (`agg_C<n>`, `e2e_C<n>`, `req_C<n>`, `ttft_C<n>`). |
| `forwards_ab.py` | Steady decode cycles by (lanes, target forwards) from the cycle receipts: the in-process A/B of `STRIX_SERVE_JOINT_GROUP_LANES=ab`. |
| `prompt_ids.py` | Prompt files for any length (`p<N>.ids`, `.csv` or `.u32`). Each one is the seed text tokenized once, repeated and truncated, with a sha256 per prompt. |

Every run directory holds `cmd.txt`, `stdout`, `stdout.ts`, `stderr`, `stderr.ts` (the arrival time of every line), `samples.tsv`,
`klog.txt` and `meta.json`, plus `receipt.json` once it has been summarised.

**Validity** (a run that fails any check is listed with its reasons and left out of every median):

- the exit code is not 0;
- kfd `evicted_ms` of the run's processes grew inside the timed window. Growth during model load is only a warning; `--evict-whole-run` makes it fatal, as the decode kits do;
- `MemAvailable` fell below `KIT_MIN_AVAIL_GB` (default 2);
- `--require` lines are missing, or `--reject` lines are present (defaults: `what()`, `timed out`, and `declined` outside cache-policy lines).

Kernel-log events (amdgpu, kfd, thunderbolt, oom, reset) are printed as warnings, or count as failures with `--strict-klog`.

**Warm and cold.** When a process runs two or more prefills, the first one is reported as cold and the last as
warm. The two figures are never mixed. **GPU clocks per token** = busy × clock × window / tokens, with power and
clock integrated over the timed window. On an NVIDIA host the sampler reads `nvidia-smi`, and host memory and
kfd checks are reported as not sampled.

**Arms** are written `'NAME|EXTRA CLI ARGS|ENV=V ENV2=V'`. Inherited `STRIX_*` variables are dropped, so an arm only
sees the levers it names. To get several rounds across tickets, give every ticket the same `OUT`: each summary
covers every receipt in `OUT`, and `abab.py` pairs the arms by `--round`.

**Running a kit**, here on a node through its queue (build first in its own command, never in the same shell as the
ticket):

```
<queue>/gpu-turn.sh LABEL scripts/bench/common/ticket.sh OUT <kit> [kit args]
```

## Kits

| Kit | Measures | Command (inside `ticket.sh OUT ...`) |
|---|---|---|
| `qwen/length-matrix` | Qwen prefill, length × mode × profile: `full` (one node), `pair` (two ranks), `full2` (the same arm on both nodes at once). Optional page-cache warm (`--warm none` skips it). Long context (YaRN, 262K to 1M) through an arm with `--context N`. | `qwen/length-matrix --bin strix-hip-decode-token --model Q.gguf --ids DIR --sizes 8K,32K,128K,256K --modes full,pair --rounds 3` (arms default `exact`, `fast`) |
| `qwen/serve-decode` | Qwen served decode, classes prose / edit / json at 1, 4 and 8 chats; one arm per ticket. | `qwen/serve-decode --bin B --model Q.gguf --arm 'wide\|--mtp-ngram-draft=wide\|' --round 0` |
| `ds41/prefill-pair` | DeepSeek V4.1 pair prefill, length × rows × arm, cold and warm in one process (`--ttft-timing --repeat 2`); the logits FNV of both ranks must agree. | `ds41/prefill-pair --bin strix-deepseek41-prefill --model M --ids DIR --sizes 8K,32K --arm 'exact\|\|'` |
| `ds41/serve-decode` | DeepSeek V4.1 pair served decode, classes prose / edit / json at 1, 4 and 8 chats (server cycle receipts, end-to-end, TTFT). | `ds41/serve-decode --bin-dir DIR --model M --arm 'ctl\|\|' --round 0` (then `abab.py summary OUT --metric e2e_C8`) |
| `ds41/dense-forms` | Shared dense policy `dense-q8a8-f16-v1` weight-decode forms (scalar FP32 / packed FP16) on the DS41 rank-0 chunk shapes: exhaustive stager decode gate, bit identity of every form, strict ABAB kernel times and the per-layer-chunk census. | `ds41/dense-forms --bin strix-hip-ds41-fast-dense-tests [--rows 1024,680]` |
| `mmattn` (microbenches, not ticket kits) | MiniMax pair prefill attention and glue kernels on one node: `flash_keysplit_bench.hip` (FAST MSA flash key-split and row-group forms, OPT 2 / 9 / 12 = key halves / 4-row groups / 8-row groups, DRIFT = neighbour-row list overlap, output fnv + max diff), `xslot_bench.hip` (TP2 exchange slot memory type on the post), `gather_bench.hip` (MoE gather store pattern). | `amdclang++ -x hip --offload-arch=gfx1151 -O3 -std=c++20 -Isrc scripts/bench/mmattn/flash_keysplit_bench.hip -o fkb && ./fkb 4096 4096 1 12 0 2` |
| `glm/prefill-pair` | GLM pair prefill, exact vs FAST, several prompts in one load (`--pf-ttft`), first prompt cold and last warm; rank agreement on logits and state FNV. | `glm/prefill-pair --bin strix-glm5-rank --model DIR --prompts p8192.u32,p8192.u32` |
| `minimax/fast-gate` | MiniMax FAST quality gate using the admitted absolute dNLL rule: reference and candidate, both ranks, then `tools/strix-gate/fast_logit_gate.py` seal and compare (`--dnll-rule abs`). `--rows 0` is the timing route (same run, no logit rows captured inside the timed prefill; report prefill figures from it, not from a dump run). Before launch the kit checks that the peer rank can read every input (model, sessions, path fields of the arm args) and stops without GPU time if one is missing. | GPU: `minimax/fast-gate run --bin strix-minimax-rank --model DIR --arm 'cand\|\|STRIX_X=1' --sessions s8k=p.u32`; CPU: `fast-gate.sh seal ...`, `fast-gate.sh compare --ref ref --cand cand --dnll-rule abs` |
| `minimax/serve-decode` | MiniMax pair served decode by class at 1 and 8 requests, plus the server start time: ready is the server's own `READY rank 0` line and the per-rank `startup:` phase lines land in `startup.txt`. `--start-only` stops at ready. | `minimax/serve-decode --bin strix-minimax-serve --model DIR --arm 'ctl\|\|' --round 0` |
| unit: MiniMax index projection | Exact slab vs FAST `minimax-index-proj-wmma-v1` (direct, x16 tiles, diagnostics), strict ABAB 3 pairs at 1,024 / 2,048 / 4,096 / 8,192 rows, fixed first 12 (4,096/8,192 rows) or 30 (1,024/2,048 rows) of 60 weight sets; x16-only timings require complete byte equality with direct WMMA; historical x16-only figures are withdrawn pending recollection, cold first call, kfd `evicted_ms`. | `STRIX_MMATTN_BENCH=wmma STRIX_MMATTN_WMMA_SERIES=x16 index-proj-tests` (build line in `src/minimax/attn/index_proj_tests.hip`) |
| `vision/battery` | Vision content battery: one greedy answer per synthetic image, graded from the pixels (`score.py`): colours present or absent, pattern, orientation, palette rule. | `vision/battery --bin-dir DIR --model M --vision V.gguf --images IMGDIR`, then `python3 scripts/bench/vision/score.py IMGDIR OUT/*/battery.json` |
| `rtx/ds41-rtx-pair-batch` | DeepSeek V4.1 RTX + pair batched decode C2 / C4 / C8: RTX FRONT on `[0, cut)` with `strix-cuda-ds41-multi-front`, pair TAIL on `strix-deepseek41-batch --hetero-listen`, 2 groups in flight; `--mode both` checks every request against its pair-only run. | node ticket: `rtx/ds41-rtx-pair-batch tail --bin strix-deepseek41-batch --model M --prompts a.csv,...,h.csv --token-file TOK --n 2,4,8 --generate 48`; RTX host: `scripts/bench/rtx/ds41-rtx-pair-batch.sh front --bin strix-cuda-ds41-multi-front --model M --stage NODE:7841 --token-file TOK` |
| `rtx/serve-route` | Served RTX routes (Qwen whole model on one RTX, or any OpenAI-compatible RTX server), decode classes at 1, 4 and 8 chats. | `scripts/bench/rtx/serve-route.sh --route qwen --bin strix-cuda-qwen-serve --model Q.gguf -- --tables T --rope-table R --mtp-sidecar S --draft-vocab V` |
| `turbo/ablate` | Leave-one-out ablation of a TURBO bundle: reads `scripts/bench/turbo/<family>.levers` (one row per lever: env, identity tag, state) and packs the arms default, all, no-<lever>, alt-<lever> into queue tickets of another kit (`--kit qwen/length-matrix` ...). | `bash scripts/bench/turbo/ablate.sh run qwen --out DIR --kit qwen/length-matrix -- --bin ... --model ... --ids ...` |
| `rtx/gpu_guard` | Fail-closed 1 s guard for any RTX run (Windows or Linux): resolves the GPU by name to its UUID, logs `nvidia-smi` telemetry (temperature, slowdown margin, power, limit, memory, clocks, PCIe gen/width) and Windows WHEA / nvlddmkm / TDR events, and on the first event, telemetry silence, temperature margin < 5 C, PCIe replay, CUDA loss pattern or wall limit kills the run, writes a STOP latch and exits non-zero; a failed child is a failure. Read-only on the machine. All arms/agents must share the same explicit `--latch` (or `GPU_GUARD_LATCH`); omission disables cross-run STOP. Windows events cover the entire machine, including other devices; Linux has no event-log coverage. Thermal sensors and replay counters are required before launch and throughout monitoring; unavailable measurement-only fields carry sample counts in the receipt. Linux workloads must retain their process group. | `python scripts/bench/rtx/gpu_guard.py --gpu-name "RTX PRO 6000" --out DIR --latch STOP_FILE --timeout-s 300 -- CMD ...` (summary in `DIR/guard.json`) |
| `logprobs/partials` | Logprob partial kernels (HIP or CUDA): device partials against the FP64 reference and the host lane order, then the kernel cost per step at 1 / 8 / 32 / 128 rows of a 64,640-entry slice (the activated cost of scoring). | `bash scripts/bench/logprobs/partials.sh --bin strix-hip-logprob-partials-tests --rows "1 8 32 128"` |
| `pfab` | Pair prefill A/B in one load: `pf_pair_ab.sh --arms B:-/C:K=V+K=V --reps N` runs every arm of a strict ABAB inside one rank process pair (rank app `--pf-ttft-arms` / `--pf-ttft-reps`, shared `include/strix/env_arms.hpp`; only switches read per call), first prefill cold, the rest warm; `pf_clk.py` prints per prefill the arm, seconds, tok/s, busy, sclk, GPU clocks per token, KFD `evicted_ms`, minimum MemAvailable and the logits hash. Use it for 360 s bundle tickets. | `scripts/bench/pfab/pf_pair_ab.sh --bin BIN --model DIR --prompts A.u32 --out DIR --peer HOST --mode fast --arms B:STRIX_GLM_PF_MLA_WMMA=0/C:- --reps 4` |
| `kfdevict` | KFD queue evictions without root: in-process probe of the process's own eviction events (trigger, restore) and userptr allocations, node sampler of `evicted_ms` against reclaim / compaction counters and GPU busy windows, a controlled memory-pressure generator, and a micro A/B of host-memory backings. | `scripts/bench/kfdevict/README.md` |

The RTX DeepSeek prefill service (`strix-cuda-ds41-prefill-service`) is measured with
`tools/ds41_prefill_service_client.py`. The routes themselves are described in [discrete-gpu.md](discrete-gpu.md).

The decode-class prompts are in `scripts/bench/prompts/classes/`. Drafting acceptance differs by an order of
magnitude between prose, a code edit and CSV-to-JSON, so served decode is always reported per class.

TURBO quality bar: `tools/qualification/turbo_bar/` ([README](../tools/qualification/turbo_bar/README.md)) measures
E (exact route) against R (a smaller pack of the same weights on the same kernels) and T (TURBO): KL, top-1 and
dNLL on full logits, scored tasks, retrieval, paired 95 % intervals (INCONCLUSIVE when unresolved) and a
speed/quality frontier. `rpack_kquant.py` produces R; `models.json` declares E/R/T per model.

## Checks

`scripts/bench/tests/run.sh` checks the syntax of every kit, runs shellcheck when it is installed, and runs
`scripts/bench/tests/test_runner.py`, offline tests of the receipts, validity rules, ABAB summary, served receipts
and prompt formats.

Vision scores are synthetic-image word-matching heuristics, not a general image-understanding accuracy measure. ABAB comparisons require an explicit `--base ARM` within a comparable workload group.

Served runs share the ticket deadline and require both ranks, telemetry and completed requests to pass. Custom clients must write `KIT_RUN_DIR/client-result.json` with positive `completed` and zero `failed`, and exit successfully. Intentional TERM shutdown is recorded separately from server failure. Missing prefill timings remain unqualified; request latency is a separate metric.

TURBO ablations are provisional experiments, not qualification evidence. `no-gdn_chunk` removes the dependent GDN gates as well as the chunk kernel. Groups after the first have a default anchor but no full-bundle anchor; do not claim marginal gains under drift without adding an `all` anchor and rotating arm order. Pending manifest rows are inspection-only until their adapters and qualification gates exist.

MiniMax TURBO flash is pending adapter admission/export; its manifest is inspection-only (`bash scripts/bench/turbo/ablate.sh arms minimax --pending`). Qwen single-process power reports lack prefill execution timestamps, so length-matrix receipts remain invalid when no independently timed prefill line is available. Final-report arrival must not anchor prefill telemetry.

For controlled exact/onfly decode diagnosis, see [the TURBO decode measurement kit](../scripts/bench/turbo/README.md) (header layout census, small-row projection benchmark and diagnostic trace adapters).
