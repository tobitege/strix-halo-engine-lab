# Decode anomaly measurement kit

Prepare a new pack's content identity before a bounded device run with the CPU-only `strix-model-manifest --warm-content-hash MODEL` command. It uses the engine's stable full-file hash cache and reports the file digest, content identity, byte count and elapsed milliseconds. Run it in the CPU lane and compare its digest with the sealed pack receipt; subsequent model opens reuse the same file-identity-bound cache. This separates first-time hashing from device startup measurements.

The decode layout census reads only a GGUF header and reports eligible projection bytes plus active MoE union bounds:

```
python3 scripts/bench/turbo/decode-layout.py MODEL --verify-rows 8 --emitted-per-pass E
```

Its byte bounds are logical weight visits, not a DRAM counter. The census checks conversion dimensions and the current launch widths for every projection group, and credits savings only for eligible shapes at rows 1–8. Allocated but unused copies (including K=640 shared-down) are reported separately. Embedding lookup, draft, KV, activation traffic, cache residency and repeated loads are excluded. The loader skips unsupported INT4 copies; their allocation is historical.

Use the census shapes for full-model traffic accounting. The projection fixture uses representative shapes; its QSA input has 13,952 outputs, while the measured pack has 13,312 (12,288 + 512 + 512). Its microsecond figures are not whole-model pass times.

For small-row projection timing, build the existing `strix-hip-q8-rows-projection-tests` target. Run `STRIX_Q8_ROWS_ONFLY_BENCH=1 strix-hip-q8-rows-projection-tests` under the deployment's quiet GPU ticket wrapper. It converts/registers each fixture once and compares native with onfly at rows 1, 2, 4 and 8, in both exact and dense-A8 modes, with cold/warm GPU-event medians and route launch counts. A zero launch count is a fallback. Report unique-weight-byte GB/s as a streaming proxy, not measured DRAM bandwidth.

For draft/verification diagnostics, `mtp-study-ticket.py PLAN.json` runs declared arms inside one existing quiet ticket (112 seconds internally). It uses the same deployment variables as the decode study. Each arm declares `tag`, `args` (simple option/value words), optional `env` assignments, `adapter_env` (recipe parameters), `http_batch` (default 16), `requests` (default `C1,C1`), `tokens` (default 192), and `prompts` (a module beside the runner). A recipe parameter replaces its fixed value; do not append duplicate MTP options, which the parser rejects. `mtp-study-prompts.py` supplies code, prose, JSON and editing requests; use `S4,S4` for separate cold/warm observations. The runner rejects incomplete requests and invalid cycle accounting.

`STRIX_MTP_WEIGHT_LOG=1` prints actual draft source/transform groups and target verify/commit groups. `STRIX_MTP_VERIFY_LOG=1` records every evaluated proposal row, including rows after a rejected prefix; `reached`, `eligible`, `accepted`, and `greedy_accept` distinguish prefix acceptance from teacher-forced equality. Logs carry monotonic timestamps for request attribution. These diagnostics change no arithmetic when disabled.

An arm with `logits: true` writes a full target-row dump using `STRIX_MTP_LOGITS_DUMP`. Enable `TURBODEC_KEEP_TOKEN_IDS=1` to retain the completion ledger. `mtp-study-analyze.py RUN_E RUN_T --logits E.logits T.logits --out result.json` computes per-depth acceptance, joint proposal/target agreement and KL only for identical committed and speculative token contexts. Serial prompt-tail dumps are excluded using the recorded prompt length. The dump's window-input hash alone does not establish a full-context match. `--context-only` permits different physical row shapes while still requiring the complete token history, and reports each matched shape pair. Forced full-chain/one-commit diagnostics are excluded from ordinary accounting gaps and must never supply qualification timings. `mtp-study-controller.py RUN... --out costs.json` reports observed warm cost and emissions by row bucket; differing contexts make these descriptive costs, not a causal fit.

`STRIX_IU4_R8_SINGLE_DECODE=auto` is the default policy. The shared resolver admits the measured single-decode path only on gfx1151 wave32 for R7/R8, K6144, N2560; other device/shape combinations retain the split-row path. `0` restores split rows. Explicit forced execution remains diagnostic. Projection byte/resource checks do not establish complete-model quality or whole-run speed.

`STRIX_Q8_ROWS_ONFLY_ROW_INVARIANCE=1` runs a separate same-input/weight row-bucket fixture at K=2048/2560/4096/6144. `mtp-study-row-summary.py LOG --output JSON` requires all 624 observations and its completion marker. Projection-vector argmax is a diagnostic, not a token-level quality gate. `mtp-study-resources.py --binary BIN --output DIR` captures compiled HIP resources and a matched old/new R8 ISA pair. Protocol tests run with `python -m unittest discover -s scripts/bench/turbo -p test_mtp_study.py`.

The shared row-decode resolver and runtime use the same `STRIX_IU4_R8_SINGLE_DECODE` parser:

|Value|Effective row-decode policy|
|---|---|
|unset, `auto`, `1`|Automatic: single weight decode for gfx1151 wave32, 7/8 rows, K=6144 and N=2560; other shapes retain split decode.|
|`0`, `off`|Split decode.|
|`force`|Diagnostic single decode for every legal group count (32/40/64/96) at 7/8 rows.|

Effective configuration prints `turbo_weights.row_decode` requested/effective/enabled/reason/scope/device_requirement fields. Inspection that has not selected a device and projection shape reports `phase=pending-device-and-shape`; it does not claim that every projection uses single decode. Here `enabled` reports whether the automatic policy is enabled; cached device capability and projection shape determine actual kernel dispatch. Full-model before/after qualification remains separate from the projection fixture.

`python3 scripts/bench/turbo/mtp-study-pairs.py OUTPUT_ROOT --before-binary BEFORE --after-binary AFTER --prefix mtp --pairs 3` pools the declared interleaved exact/before/after cold and warm receipts and checks before/after token IDs.

`STRIX_IU4_ROW_REDUCTION=legacy` is the default numerical contract. The explicit value `canonical` (a separate numerical contract whose quality gate is pending) uses S=8 and GC=32 for every onfly G64/G96 row bucket, retains G32/G40 behavior, and appends `iu4-g64g96-gc32-v1` to the active onfly numerical identity. It preserves the separate single-decode `auto`/`0`/`force` selection. The canonical 624-observation projection fixture requires byte equality across row buckets; target-payload and quality qualification remain separate. Cross-row logit comparisons report distinct matched source/reference rows, committed-frontier coverage and completion-ledger agreement so that a partial context match cannot establish a full serial/window equivalence gate.

The source-controlled ticket runner and HTTP client are documented in [the decode-study workflow](decode-study.md). A deployment supplies only its model-specific server command adapter and assets; binary, model and prompt paths are parameters. Use the same binary, model, MTP bundle, prompts, slot budget and explicit numerical options in both arms. Count emitted tokens, retain invalid startup attempts, record both first and last prefill, use strict alternating pairs, and include the GPU busy fraction in clocks per token. Pre-cache files consistently before bounded tickets when startup requires it.

`STRIX_CYCLE_TIMING=1` reports completed host phase intervals; `STRIX_SERVE_RECEIPT=1` reports cycle draft/verification/host totals, rows and emissions. Verification includes commit and reconciliation; it is not pure target-kernel time. Kernel traces supply actual launch names and durations. Greedy draft acceptance cannot be inferred from target-to-target KL alone.

E/T study arms compare bundles: T also selects the turbo prefill policy, so their difference cannot isolate onfly weights. Use the projection fixture for projection diagnostics; bundle qualification requires separate native model-fidelity and complete-workload gates.

The `.levers` manifests are independent study selections, not generated profile-table contracts. For example, MiniMax's main study uses layers 40–59 and `fast_i4`; its profile table uses layers 0–59 and does not select that format. Record the emitted arms and inspect each engine's effective configuration before measuring. The optional final manifest column supplies explicit OFF assignments for omission arms (required when another lever enables the omitted member).

Study tags are single-use, including failed attempts: keep their outputs and choose a fresh tag. Ticket wrappers retain an atomic claim and reject existing runs or completion markers. Standalone C8 uses the inventory prompts and completes only after successful requests and analyses.
