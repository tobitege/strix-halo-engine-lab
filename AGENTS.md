# Documentation and research publication policy

Keep comparative evaluations, external benchmark records, review correspondence and
source-identified external inspection artifacts outside Git. Public documentation
describes project-owned contracts, independent mathematical semantics, implementation
status and qualification evidence without naming unrelated runtime projects.

Amendment (user-approved exception): an outside runtime may appear only as a same-machine
reference row that states its version and command (llama.cpp), and an external quantizer or
format may be named only as format/quantizer attribution (exllamav3). No other outside-runtime
reference is permitted.

Preserve accurate provenance, licenses and notices for actual adopted code, dependencies,
model assets and official mathematical/ISA references. Neutralization must not erase
required attribution or manufacture independent authorship.

Archive original comparative inputs privately before editorial cleanup. Do not rewrite
repository history as part of ordinary documentation editing. Numerical proposals remain
drafts until their compiler, native execution, model fidelity and complete-workload gates
pass; theoretical throughput and external results do not establish local qualification.

# Shared code first (all engines)

Every improvement that can apply to more than one engine is implemented once, in shared code, and
each engine only adapts it. This is the default for new work, not a later refactor.

- Shared homes: `src/core` (linear/integer/IU4 kernels and plans), `src/moe_core` (MoE routing,
  grouped GEMM, activation quantisation policies), `src/attn_core` (attention kernels),
  `include/strix/session` and `include/strix/serve` (engine contract, KV/prefix/conversation
  caches, scheduling, batching, fanout, receipts), `include/strix/pair` and `include/strix/dist`
  (two-node pipelines, transports, exchange overlap).
- Engine code holds only what is truly model-specific: tensor layouts and names, numerics the model
  requires, its family adapter (hooks the shared layer calls) and `models/<model>.toml` settings.
- A variant is a policy or trait on the existing shared kernel, never a parallel copy. If a shared
  path is slower for one engine, fix it inside the shared code until it matches, then make it that
  engine's default and delete the engine-local copy, its switches and dead tests.
- Prototyping inside one engine is allowed only as a step: the same change set (or its PR description)
  states what moves to shared code and which engines adopt it next.
- Reviews reject new engine-local copies of logic that already exists, or obviously belongs, in a
  shared home.

# Production defaults for humans and coding agents

The fast path is the default for all five public engines. Start with the plain command; let the
resolver select qualified production settings. Do not pass legacy opt-in organization flags that
are already defaults: they can force an explicit route and decline the automatic one.

| Engine | Recommended command / contract | Inspect | Baseline |
|---|---|---|---|
| Qwen | `strix-hip-decode-token MODEL.gguf $(cat prompt.ids) --generate 256`; [auto route](docs/getting-started/engine-reference.md#command-line); exact MTP decode is on when the [drafter bundle](docs/engine/speculative/mtp-native-cli.md#default-mtp-drafter-bundle) is beside the model (`--no-mtp` / `STRIX_MTP=0` = plain decode, same tokens) | `--print-effective-config` | `--baseline` / `STRIX_BASELINE=1` |
| DeepSeek pair | `strix-deepseek41-serve MODEL.gguf --rank R --transport usb4auto`, one rank per node; [production defaults](docs/models/deepseek-v4.1/production-defaults.md) | `--print-effective-config` | `STRIX_DS41_BASELINE=1` |
| DeepSeek RTX | Use the [standalone command and required receipt/state/reference assets](docs/models/deepseek-v4.1/rtx-standalone.md#run-and-persistent-worker) | `--print-config` | `--baseline` / `STRIX_DS41_RTX_BASELINE=1` |
| MiniMax | `strix-minimax-rank MODEL_DIR R usb4-auto OUTPUT_DIR`, one rank per node; [resolver](src/minimax/engine/mm_defaults.cpp) | `--print-effective-config` | `STRIX_MM_BASELINE=1` |
| GLM | `strix-glm5-rank MODEL_DIR --world 2 --rank R --transport usb4-auto --prompt PROMPT.u32 --decode 64`, one rank per node; [resolver](apps/glm_rank_defaults.hpp) | `--print-effective-config` | `--baseline` / `STRIX_GLM_BASELINE=1` |

Qwen startup prints one `strix-build-info` and one `strix-startup-event` JSON line on stderr (read-only
[preflight and build identity](docs/operations/startup-preflight.md); not in `--print-effective-config`
runs; `STRIX_RELEASE_EVENTS=0` switches both off). Advisories in that record are not failures.
Inspection exits before device work; preserve the engine's required assets and two-rank launch contract
for execution. Check [all item guards, fallback notices and off switches](docs/optimizations.md) or
the [JSON catalog](docs/optimizations.json). Qwen automatic FP16 prefill excludes explicit `--mtp` single-request commands (the default drafter bundle keeps it);
HTTP serving resolves its tile and prefill arithmetic through the same rule as the single-request route
(`resolve_prompt_route`; [defaults](docs/engine/serving/serving.md#default-prompt-prefill)).
HTTP servers (Qwen, DeepSeek) describe what they execute at `GET /v1/capabilities`; the `strix` object of each response
states the applied policy (requested vs effective sampling). Generation fields the server does not execute are refused
with a 400 naming the field ([list](docs/engine/serving/serving.md#chat-requests)). Qwen executes JSON-constrained
answers (`response_format` / `text.format`, [constrained output](docs/engine/serving/serving.md#constrained-json-output)):
exact to the constrained target, not to unconstrained greedy. Policy host selection uses admitted serial-equivalent MTP windows (`mtp-host`) or `serial-host` fallback;
`--http-selection-mtp off` forces serial for JSON, budgets, scores and transforms. Receipts state route/reason.
Thinking budgets and logit transforms are declared
token-changing request policies; logprobs and SSE progress never change tokens.
Fast FP16 Qwen prefill is its own identity, not dense-A8 serial bit equality. GLM defaults to R1b decode
with R1 prompt prefill and exact MTP self-speculation (`--mtp 0` = plain decode, same tokens); `--identity r1`
restores R1. Exactness references must match that identity.

Qwen HTTP serving: omitted `--http-batch` = as many slots as memory admits (startup line `http slots N (...)`, built on demand), `N` caps, `0` = one request at a time. In-block decode service, short-first, nested admission and boundary cancellation are on (`STRIX_BATCH_SERVICE_MS`, `STRIX_BATCH_SHORT_FIRST_ROWS`, `STRIX_BATCH_NEST_ROWS`, `STRIX_BATCH_CANCEL_POINTS`; `0` = off). Gates: `tests/batch_service_gpu_gate.py`, `tests/http_slots_gpu_gate.py` ([serving](docs/engine/serving/serving.md#scheduling-qualification)).

Qwen certified prefix caching is ON by default only for single tiled-MTP HTTP (`--http-batch 0`
or auto capacity resolving to the single runner), T=2048, default numerical profile/schedule/tails
and model assets, context up to 16384. Gated outputs, raw logits and canonical state digests are
bitwise identical to cache-off in the same binary/configuration. After native residency,
`B = min(4 GiB, A/16, A - reserve)`, aligned down to 4 KiB; A is conservative available memory
and reserve protects headroom, remaining OFF commitments, metadata and staging. Never reduce
context, slots or lazy growth for caching. Startup `qwen cache policy`, `--print-effective-config`,
`/v1/capabilities` report requested/effective/source/reason/phase, including declines/OFF;
`/v1/cache/stats` includes policy only while storage is active. Inspection
is pending without device work. Other automatic configurations decline by name and preserve
cache-off execution. Batch routes, sessions, serial CLI, YaRN and other profiles remain
unqualified for automatic reuse; preserve their explicit-only contracts and scheduler defaults.
OFF controls: `--prefix-cache off`, `--prefix-batch-cert off`, `--session-cache off`.
Explicit `--prefix-cache-host-bytes` keeps its existing behavior. Widening qualification requires
complete token/logit/state equality, unchanged capacity, interleaved cold-cost/noise checks and
useful warm reuse; a startup smoke does not establish these gates.

Use each engine's built-in exactness gates and compare complete token IDs and logits/state digests.
Qwen: profile-pair qualification and operator byte checks; DeepSeek pair: both-rank prefill/greedy/verify/batch
and transport gates; RTX: replica canonical/pressure/plain-speculative chat gates; MiniMax: golden logits,
decode/verify/exchange checks; GLM: G3/G4, identity, prefill and speculative-oracle gates.
Once the bench kit lands, inspect availability and run:

```sh
bash scripts/bench/bench.sh list
bash scripts/bench/bench.sh check --engine E --only exactness --suite quick --json
bash scripts/bench/bench.sh run --engine E --only exactness --suite quick --json
```

Use the engine key reported by `list`. Verdicts are `EXACT`, `MISMATCH` and `NO-REFERENCE`;
missing reference data is not an exactness pass. Do not assume the planned kit is installed today.

For A/B, change one off switch at a time, preserve model/profile/state/capacity, restart startup-cached
readers and both ranks as needed, verify tokens/digests first, then compare medians from interleaved runs.
Do not add individual gains from different workloads or call a changed numerical identity bit-identical.
Qwen long-context answers are checked with `tools/qualification/longctx_fidelity/` (generator ground truth,
CPU tests first; its README states what a pilot does and does not establish). Keep comparative-runtime
reference results out of that public kit. Qwen YaRN (`--rope-yarn`) is an explicit experimental mode: never a
default, never combined with exactness claims against native; new resolver defaults must be classified for rotary
work in `apps/qwen_yarn_admission.inc` before YaRN accepts them.
Kernel/IOMMU settings (including the GPU PerfOpt, `amdgpu.iommu_perfopt`, the lab nodes' default boot) are host decisions made
by the machine owner as root; record the running kernel and that parameter with any timing receipt, never
disable the IOMMU to gain speed (USB4 controllers must stay translated), and compare PerfOpt only on vs off
on the same kernel.
Keep deployment endpoints, machine/account identifiers, private paths, prompts, receipts and unpublished
benchmark/review material out of public changes. Use relative asset placeholders in commands and public
implementation or qualification contracts for links.

## Bit-changing Qwen prefill options

The single registry and active-set identity live in
[include/strix/prefill_identity.hpp](include/strix/prefill_identity.hpp).
Every governed prefill option is on by default once READY, must have an individual
off switch (`--prefill-disable NAME`, config `prefill_disable`, `STRIX_PREFILL_<NAME>=0`),
and must honor the master (`--prefill-exact-fp16`, config `prefill_exact_fp16 = true`,
`STRIX_PREFILL_EXACT_FP16=1`). Keep unfinished options out of `kPrefillOptionsReady`.
Receipts, effective config and HTTP capabilities must report the active name, 16-hex
identity hash and option list. Re-qualify every changed active set: compiler, native
execution, model fidelity and complete-workload gates must pass before READY.
These controls are server-level only; changing a shared prefill plan or prefix-state
cache identity requires a server restart. See the
[option catalog](docs/optimizations.md#bit-changing-prefill-options).

Qwen `hc_output_wmma` is default ON. `hc_output_wmma` passed a pre-declared confirmation held-out set; its NLL endpoint is measured against a rounding-equivalent control whose per-case movement is capped at +-0.02 nats/token (a bounded control allowance, not an unbiased estimate of the uncapped difference); the absolute NLL change vs the exact reference also passed (-0.37 % NLL/token, upper bound +0.38 %). The off switch restores the exact output path.
Controls are server-level only: `STRIX_PREFILL_HC_OUTPUT_WMMA=0`,
`--prefill-disable hc_output_wmma`, or config `prefill_disable = ["hc_output_wmma"]`;
master `STRIX_PREFILL_EXACT_FP16=1`, `--prefill-exact-fp16`, or config
`prefill_exact_fp16 = true`. Restart the server; request-level controls return HTTP 400.
Fast HC with this option is excluded from MTP and nested admission pending qualification;
ordinary MTP/HTTP profiles retain their existing arithmetic. OFF restoration compares logits
and seven state payloads to public main; compiled, lineage and cache identity bytes may differ.

## Exact Qwen bundle controls and candidate indexer

Exact defaults are ON: early static admission (`STRIX_QPF2K_EARLY_SETUP=0`,
`--no-early-setup`, config `qpf2k_early_setup = false`); eager HIP code loading
(`STRIX_EAGER_CODE_LOAD=0`, `--no-eager-code-load`, config `eager_code_load = false`);
automatic T16384 from 32,768 prompt tokens (`STRIX_QPF2K_T16K=0`); exact indexer
batching (`STRIX_QPF2K_INDEX_BATCH=0`). These are outside the bit-changing registry.
Eager loading never overrides a user-set `HIP_ENABLE_DEFERRED_LOADING`. Setup and
code loading move work into startup, without establishing reduced process wall time.
T16384 must fall back to T8192 when its full UMA/GTT reservation does not fit.

`index_proj_wmma` is implemented but outside `kPrefillOptionsReady`: default OFF,
opt in with `STRIX_PREFILL_INDEX_PROJ_WMMA=1`. It uses tag `idxproj16` and the standard
individual/master OFF controls. This explicit draft exception stays OFF until compiler,
native execution, model fidelity and complete-workload gates pass. Existing
`hc_output_wmma` behavior and the default/exact prefill hashes remain unchanged.
See the [native bundle gates](docs/engine/prefill/hc-output-wmma-qualification.md#bundle-native-gates).

HC low-band layout defaults to mode 3 through decoder qpf2k defaults;
`STRIX_QPF2K_HC_LOWBAND=0` restores the old layout. It is live only with `hc_output_wmma` OFF (the exact-FP16 output route),
not with the default output WMMA identity. Only the qpf2k projection
form 4 to FP32 output v3 band 16 uses it. HC output WMMA, check mode, normalized16,
inverse and other routes retain their old layout. Arithmetic and numerical identities
are unchanged; restart for this cached control. Native local CI passed 132/132 cases
without stray stores, including eight T8224 cases with 512-row pool rounding, and
32K last-row logits matched mode 0 bitwise with `hc_output_wmma` OFF. Exact,
bitwise-checked; no timing claim because low-band is inactive on the default route.
See [local CI](docs/engine/prefill/hc-output-wmma-qualification.md#bundle-local-ci-evidence).

Exact [compact QSA output images](docs/engine/prefill/compact-stage-images.md)
default ON for eligible full-T score-ring owners;
`STRIX_QPF2K_COMPACT_IMAGES=0` restores FP32 containers after restart.
Calculated memory saving is 96 MiB per T8192 owner. Native local CI passed the
host/HIP fixtures and 32K last-row bitwise logit comparison. Interleaved 32K A/B
was neutral within noise (OFF 19.95 s, n=2, versus default 20.09 s); no speed
gain claimed. Already rounded BF16 bits widen identically. Numerical,
rotary, kernel-plan and saved-state/cache numerical identities are unchanged.
The physical execution identity must change because it hashes the smaller arena
sizes, offsets and strides; it is not a numerical identity. Effective config
reports the requested policy; compiled organization reports admitted savings.

PLE staged-gather acceptance now tracks successful independent row-0 consumption
and accepts the complete ordered deferred tail only after comparing every staged
row ID, including row 0. This bug fix is unconditional, with no switch; copy
semantics, staging guards, worker-error fallback, slot events and identities stay
unchanged. Native local CI passed all 11 device/host fixtures, including the PLE
launcher and staged-gather tests. Baseline and `STRIX_QPF2K_DEFAULTS=0` each turn
every controlled exact organization item off. At 32K, last-row logits match bitwise between
default and each OFF arm (early setup, eager loading, T16384, index batching, compact
images). Three-round interleaved medians were 20.09 s default, 20.48 s early setup
OFF, 20.34 s T16384 OFF and 20.07 s eager loading OFF. Early setup and T16384 show
measured gains; eager loading is neutral within noise. These gains are not additive.
At 131,072 and 262,143 real-text tokens the default admits automatic T16384, and with
`STRIX_QPF2K_T16K=0` it admits T8192 as before, with no fallback. Open, nonblocking gates
are the seven state payloads, T16384 at 262K and pressure fallback, and staged versus
unstaged PLE gathered bytes; see [local CI](docs/engine/prefill/hc-output-wmma-qualification.md#bundle-local-ci-evidence).
