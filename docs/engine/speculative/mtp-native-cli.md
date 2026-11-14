# Explicit native MTP CLI

`--mtp-metadata-only 1` is an explicit diagnostic. It retains the same actual
asset, provenance, rounding and budget admission, but does not require golden
file/path pins. It loads the real owners and reports their plan, semantic and
native/head profile identities, content identities and full file SHA values.
It returns before any target or predictor forward and never activates or
qualifies MTP. The ordinary path still requires independent expected outputs;
the diagnostic cannot substitute for a model fixture invocation.

`strix-hip-decode-token MODEL.gguf TOKEN... --generate N --mtp` selects a separate eager target request. It requires at least three prompt tokens and externally supplied independent full-forward reference rows for every adjacent prompt pair. The default decoder route and optimization profile remain unchanged. Graph, prepared dispatch, overlap, islands, other request controllers and recording/export routes are rejected before native setup rather than silently disabled.

The request requires all of these values (each accepts `--name VALUE` or `--name=VALUE`):

| Options | Meaning |
| --- | --- |
| `--mtp-sidecar`, `--mtp-sidecar-logical-name`, `--mtp-sidecar-sha256`, `--mtp-sidecar-bytes` | Actual local file, its framed content namespace and expected complete file digest/size |
| `--mtp-target-sha256`, `--mtp-target-content` | Expected complete target file digest and existing framed target content identity; these are different hashes |
| `--mtp-source-repository`, `--mtp-source-revision`, `--mtp-source-path` | Declared source provenance, retained with the verified asset |
| `--mtp-converter-revision`, `--mtp-converter-sha256`, `--mtp-norm-converter-sha256` | Pinned conversion reference provenance |
| `--mtp-conversion effective-norm-embedding-first-v1` | Explicit admitted checkpoint interpretation |
| `--mtp-normalized-rounding fp32\|bf16-rne`, `--mtp-projection-rounding fp32\|bf16-rne` | Both native cast boundaries; they must match the independent golden profile |
| `--mtp-sidecar-residency register\|copy` | Explicit sidecar storage choice; the target's existing mapping/residency is retained |
| `--mtp-header-bytes`, `--mtp-sidecar-retained-bytes`, `--mtp-sidecar-setup-bytes` | Positive bounded header and sidecar backing reservations |
| `--mtp-golden`, `--mtp-golden-sha256`, `--mtp-golden-max-bytes` | Independent versioned full-value file, expected complete SHA256 and positive file budget; see [format](mtp-forward-golden-file.md) |
| `--mtp-replay-host-bytes`, `--mtp-runtime-bytes`, `--mtp-target-capture-bytes` | Positive complete replay payload, independent predictor pool and target capture reservations |
| `--mtp-policy-id`, `--mtp-candidate-depth` | Nonzero declared replay policy identity and maximum depth 1..8 |
| `--mtp-max-hidden-error`, `--mtp-max-logit-error` | Optional finite nonnegative declared tolerances, each defaulting to exact value agreement (zero); selected tokens must agree exactly |

The golden rows must start at source position zero, match all prompt source/next tokens, include every BF16 target input and predictor output wide value, and contain every finite FP32 logit. Full-file pinning binds supplied bytes; it does not establish that a reference was independently produced. The decoder never obtains expected values from its own outputs. Mismatching source/model/semantic/cast identities or replay policy prevent generation. Actual replay observations feed the existing qualification and activation functions; no readiness flag is forced.

### Self-generated golden (developer benchmarking)

`--mtp-golden-dump FILE` replaces `--mtp-golden`/`--mtp-golden-sha256` (it rejects them and `--mtp-metadata-only`). With every other MTP option unchanged, it consumes the prompt with the real target, runs the same native MTP forward that replay runs over each prompt pair, writes the complete version-2 golden (it records the target prefill/decode numerical profile pair) for exactly those rows (refusing to overwrite an existing file), prints a one-line JSON receipt with `mtp_golden_sha256`, and exits without replay, qualification or generation. Window options are accepted but no window is prepared. The file binds the actual target/sidecar content, structural plan, semantic contract and native numerical profile; its independent-source digest is SHA256 of `strix-mtp-golden:self-generated-native-forward:not-independent:v1` and its reference ID is `0x314e4547464c4553` (`SELFGEN1`). A run consuming such a golden reports `mtp_reference_independence: self-generated-native-forward-not-independent`.

Two-step use: (1) run with `--mtp-golden-dump prompt.golden` and the target prompt; (2) run the same command line with `--mtp-golden prompt.golden --mtp-golden-sha256 <printed sha256>` instead. Replay then matches exactly (zero tolerances) under the same profile and casts. Drafter values only change acceptance; every emitted token is still target verified, so the generated output is unaffected. This is not independent evidence and does not qualify the drafter. Size `--mtp-golden-max-bytes` for `184 + rows*(24 + 4*wide + 4*vocab)` (about 1.03 MB per row for the 2560x4-wide, 248320-vocab model; 64 MiB covers a 43-token prompt) and `--mtp-replay-host-bytes` slightly above the file size (64 MiB also covers the 43-token replay).

The prompt is consumed once by the real target hidden producer. Its snapshots drive independent replay, leaving that same predictor primed immediately before the final prompt row. The final prompt target selection is emitted once as an unconsumed pending token. Subsequent cycles consume pending inputs, return only new target-verified outputs, and respect EOS, context and remaining output budget. Serial target verification is the default for this explicit route. Target D1 uses its own real record/shadow on its owned stream when the resolved target catalog requests D1.

Optional closed R2/R4/R8 target verification requires `--mtp-target-window-rows 2|4|8` and all positive values `--mtp-window-arena-bytes`, `--mtp-window-capture-bytes`, `--mtp-window-checkpoint-bytes`, `--mtp-window-total-bytes`, `--mtp-window-history-intervals`. It also requires explicit paired wave8 GDN input/output with their existing virtual/prefetch prerequisites and resolved D1. The declared candidate depth must be at least `rows - 1`; for example, eight-row verification requires depth seven or eight. Remaining per-cycle budget, context or history can still choose serial verification. This prepares the existing actual all-head known-window owner, two capture pools and typed occupied-KV checkpoint. A nonaccepted whole window is restored before serial target verification. Whole-chunk tickets retain the actual final session, without invented intermediate sessions.

Reservations cover declared requested payloads, including target D1 capture/shadow requirements. Window total includes the window arena/completion record plus checkpoint/capture payloads. They exclude opaque driver bookkeeping and container capacity overhead, and do not assert physical RAM fit. Snapshot tails follow their declared logical policy. Receipt fields describe actual operations from a completed invocation, separate declared-policy replay activation from hardware/performance qualification, and label cycle-only host wall timing. Source implementation alone establishes no native or model validation result.

`--mtp-spec-min-p 1.0` does not force serial target execution. The cost
controller may probe an unmeasured configured owner and select it when its
expected emitted tokens per unit cost beats serial verification, including
padded prefix windows. Use actual attempted-window counts and row histograms
to identify execution. For an independent serial decode reference, use ordinary
eager generation without MTP, retaining the same actual prefill, numerical
profile and assets; do not infer seriality from a probability threshold.

The verification receipt follows actual window attempts. A configured owner with zero attempts reports `serial-target-fallback` and separately records its requested rows/configuration. Pair-entry dispatch counts explicitly exclude wide R4/R8 dispatches; the wide count is `null` because the current verifier result does not expose it. No total is inferred from requested geometry. The base backing inventory keeps its original pre-optional-request boundary. A distinct `mtp_target_d1_inventory` describes the later actual target D1 presence as unaccounted storage, without inventing backing keys or adding it to the base allocation census.


Optional `--mtp-window-expert-union --mtp-window-expert-union-bytes N` connects
closed target verification to the existing routed-expert union. It requires a
configured target window, resolved A8 gate/up, grouped Q5_1 down, slot-parallel
down and deferred status. It does not select these prerequisites implicitly.
The requested union must contain admitted routed steps; otherwise setup rejects
it. Ordinary generation and serial MTP retain their existing behavior.

The native union workspace is allocated once with its own lease and retained
owner. Its actual size must fit the separate explicit union ceiling and the
existing window arena ceiling, together with the arena and eight-byte completion
record. The same charge is subtracted from the existing total window ceiling
before admitting both capture pools and the rollback checkpoint. Declared
ceilings are never enlarged. No union allocation occurs per cycle.

Receipts report the compiled eligible step count, actual workspace size and
generation, and the GPU-produced CSR summary from the last eligible layer of the
last completed target window. This includes provisionally completed windows that
were subsequently restored. That summary is not an aggregate reuse, traffic or
performance measurement. Checkpoint fixtures additionally expose actual union
and variant dispatch counts and require a nonzero union witness when requested.
Independent replay, complete state restoration and full-cycle cache comparisons
remain mandatory; this integration does not qualify speed or arbitrary models.

### Sampled decoding

`--mtp-sample-temperature`, `--mtp-sample-top-k`, `--mtp-sample-top-p`, `--mtp-sample-min-p`, `--mtp-sample-seed`,
`--mtp-sample-speculation`, `--mtp-sample-schedule` and `--mtp-sample-repeats` select seeded sampling instead of
greedy selection. They are described in [sampled-decoding.md](../decode/sampled-decoding.md).

### Context n-gram extension of the draft chain

`--mtp-ngram-draft[=on|off|auto]` lets the context continue a draft chain that stopped early. The default is
`auto` (D-055): it runs wherever it is admissible (`--mtp-draft-chain`, prefix target windows, greedy
selection, one request, the D-022 controller; in the HTTP server per request) and stays off elsewhere (sampling,
cohort requests, golden dumps, the survival controller). `on` (or the bare flag) requires it and rejects the
command where it cannot run; `off` disables it. A server request can set `"ngram_draft": true|false`. The chain keeps its drafts up to the first one
below the minimum probability; when the longest suffix (2 to 4 tokens) of prompt + output + kept drafts
occurred earlier in the context, the tokens that followed it then are appended, up to the draft depth, in
the same verify window. The target verifies every row, so the output is the greedy output with or without
the switch; the predictor's cache is rebuilt from the committed target rows as for any rejected draft.

Whether to propose is learned per request: every cycle's would-be extension is scored against the tokens the
verifier emitted (host work only), from a prior of 0.2, and an extension is proposed only while its learned
first-row acceptance is at least 0.5. Extension rows size the window with their learned acceptance and are
kept out of the draft calibration. `STRIX_MTP_NGRAM_FLOOR` and `STRIX_MTP_NGRAM_PRIOR` change the gate.
The receipt's `mtp_ngram_draft` counts proposal cycles, extended cycles, extension rows and accepted rows.
`proposal_cycles` are the shadow proposals (host only): rows reach the verifier only in `extended_cycles`
(`extension_rows`). `shadow_scored_cycles` counts the shadow proposals whose confident drafts were all emitted,
and `shadow_right_rows` their leading rows equal to the emitted tokens, i.e. the rows an always-open
extension would have had accepted.

### Context copy/n-gram drafter (`ctx`, `copy`)

`--mtp-ngram-draft=ctx` or `=copy` (experimental, never a default) replaces the extension above with the context
drafter of D-065; both are required like `on` (same admission) and the target still verifies every row. `ctx`:
the longest exact suffix of 2 to 16 tokens of prompt + output + kept drafts, majority continuation, a
per-request learned gate from a 0.75 prior that each row must pass together with the window's min-p threshold,
proposals also after an empty confident prefix. `copy` adds: a context span anchored on four tokens or more
replaces the confident MTP chain when its expected accepted rows exceed the chain's, and a span that already
passes the gate before drafting, and is expected to deliver more accepted rows than the request's MTP cycles
(moving average), supplies the whole chain without predictor forwards (the predictor cache is
rebuilt from the committed target rows, as for any rejected draft). The receipt's `mtp_ngram_draft` then has
`mode`, `selected_cycles`/`selected_rows` (passed the gate), `copy_first_cycles`, `copy_first_declined`, `replaced_cycles`, and
`extended_cycles`/`extension_rows`/`accepted_rows` counted from completed verification. The model qualification
binary checks real cycles against the serial oracle with `STRIX_MTP_COPY_GATE=ctx|copy` together with
`--qualification-prefix-gate` (tokens, full-vocabulary logits and committed state at every emitted row).

It helps requests that copy their context (correcting or editing a given document, code edits); on free
prose, summaries and answers the gate stays closed and nothing changes. Cohort cycles do not use it.

Wide copy verify (opt-in, `--mtp-copy-wide-rows 32`, `16` or `32,16`; absent or 0 = off): with `copy` on the tiled
MTP prompt and one numerical profile (no split `--prefill-numerical-profile`), single requests only. A cycle whose
anchor is unanimous, at least eight tokens long, continues for N-1 context tokens without an edited context token and
whose learned 8+-token unanimous acceptance cells (rows 0, 1-2, 3+) are all at least 0.9 verifies those tokens in one
N-row forward of the N-row tiled tail owner instead of the 8-row window; listed sizes are tried largest first (a
span too short or cut by an edit for 32 may still fill 16). Every listed size needs a tail owner (all-rows heads and
the multi-row head; `--known-prefix-tail-rows` must include each, the default tail set adds them); the sizes share
one state snapshot. Rows are selected by the target's finite first-max rule; a full acceptance keeps the N committed
rows and adopts the last as the pending row. A rejection (or an EOS before the last row) abandons the wide cycle: the
snapshotted recurrent/convolution state and the session are restored, nothing is committed, the predictor is
untouched, the drafter scores the proposal as a shadow (its rejected row lowers the long-unanimous cells), and the
ordinary cycle runs instead (copy-first windows re-cover the accepted prefix); no wide cycle of any size is proposed
again until the committed frontier passes the rejected row. The receipt adds `wide_rows` (the largest size),
`wide_cycles`, `wide_rows_accepted` (committed wide cycles), `wide_abandoned`, `wide_abandoned_rows_wasted`,
`wide_ms_committed`, `wide_ms_abandoned` (totals) and `wide_sizes` (the same per size). `STRIX_MTP_WIDE_GATE=1` with
`--qualification-prefix-gate` (without `STRIX_MTP_COPY_GATE`) checks full and abandoned wide cycles of the largest
size and, with a second size, one full and one abandoned cycle of it (the pre-cycle state exactly, the cooldown of
every size) and the ordinary cycles after them against a 176-row serial oracle (tokens, full logits of every wide
row, committed state).

### Explicit YaRN MTP CLI

An explicit `--rope-yarn 2 --context C --mtp` admits the table-backed CLI
qualification route at positive C <= 524288. Above that band, use factor 4
with C <= 1048576. Supply the existing pinned sidecar, golden and budgets;
`--mtp-golden-dump` captures a fresh version-3 file with the same rotary table,
actual C and profile pair. A bare YaRN command discovers
`<model>.mtp-bundle.json` beside the GGUF and selects MTP after verifying its
sidecar through the persistent content-hash cache. Without a bundle it uses serial
decode with one notice. `STRIX_QWEN_YARN_MTP=0` selects serial; `=1` requires a
bundle (missing bundles refuse). Explicit `--mtp` remains available. HTTP, session/cache restore, cohorts, wide copy and sampling remain
closed. Structurally non-serial verifier arithmetic refuses before mutation.
Host admission and compilation do not establish GPU exactness or model fidelity.

Both `STRIX_QSA_MTP_POOL` and `STRIX_PLE_ROLLING` default on for admitted MTP
CLI at factor 2 through 524288 and factor 4 through 1048576. Either `=0` still
turns that feature off; the selected MTP route remains active. The 1M factor-4 prose
qualification matched all 256 generated tokens, every committed full-vocabulary
logit digest, prompt/final target state and predictor state against serial YaRN
and pool-off MTP. This is equality within the same YaRN identity.

Memory admission tries these storage sets in order at T8192 before reducing
the tile to T4096 and then T2048:

| `storage_ladder_step` | Rolling positions | PLE cap bytes | Score ring rows | PIPE |
|---|---:|---:|---:|---|
| 0 | Existing defaults | Existing default | Existing default | Existing default |
| 1 | 1024 | 268435456 | 256 | Existing default |
| 2 | 256 | 33554432 | 64 | 0 |

Only a memory refusal advances the ladder. Each candidate must fit setup,
calibration, prompt, decode and drain, including the 4 GiB physical margin and
independent UMA/GTT/workspace/metadata limits. An explicit environment value
pins its knob in every step, including PIPE; the ladder never enables an off
switch or overrides a user size. Inspection and run receipts report
`storage_ladder_step` and `yarn_mtp_storage` (chosen positions, cap, ring and
PIPE); admission also lists those fields and the phase peaks. File credit is
verified again when trying another storage candidate; predicted credit is
never admission credit. Serial YaRN and native plans do not use the ladder.

`STRIX_QWEN_YARN_MTP=0` with a plain YaRN command selects the serial comparator;
an explicit `--mtp` conflicts with that control and refuses. Keep the same prompt,
C, descriptor/table, prefill schedule and target profiles. Independently,
`STRIX_QSA_MTP_POOL=0` retains genuine MTP with historical key consumers and its
separately admitted memory envelope. Inspection reports the requested/effective
route, source, identities and named owners. M1 keys are charged once, alongside
predictor KV/raw storage (2304*C), sidecar/Q4 copies, captures, windows and taps.

`STRIX_MTP_STEP_LOGITS_DIGEST=1` emits the same ordered `mtp-step-logits`
generation ledger on plain serial and MTP CLI routes. It starts with the actual
terminal prompt logits, covers only committed consumed positions, and retains
full-vocabulary byte SHA256 values. `STRIX_PROMPT_STATE_DIGEST=1` builds the
read-only registry and emits `qwen-target-state phase=prompt|final` digests over
the same ordered target components and occupied ranges, consumed token history
and pending selection. MTP predictor extensions and opaque lineage remain in
the separate complete-session diagnostics. Missing or incomplete evidence is
NO-REFERENCE; compare complete tokens, ledgers and target state before timing.

### Default MTP (drafter bundle)

The plain production command uses exact MTP speculative decoding by default:

```bash
strix-hip-decode-token MODEL.gguf $(cat prompt.ids) --generate 256
```

The drafter is a versioned model asset, resolved beside the model:

| File | Role |
| --- | --- |
| `MODEL.gguf.mtp-bundle.json` | the bundle: sidecar name, size and SHA-256, target pins, provenance, policy and budgets |
| the sidecar named by the bundle | drafter weights; size and SHA-256 are verified (persistent content-hash cache) |
| `draft-vocab/` | draft vocabulary packs named by the bundle |

Copy [the distill-g recipe](../../../recipes/qwen-distill-g-yarn.mtp-bundle.json) to
`MODEL.gguf.mtp-bundle.json`; the same file serves the native and the YaRN route. `--mtp-bundle FILE` names
another bundle. The distill-g sidecar is not downloaded by the fetcher: it is the distilled draft registered in
`include/strix/mtp_draft_profiles.hpp` (round 2 of the draft distillation, see
[draft profiles](decode-defaults.md#draft-profiles-includestrixmtp_draft_profileshpp)); the base draft published with
the model is a different file and must not be renamed.

Selection is the shared default resolution in `include/strix/session/speculation_default.hpp`; the engine only
declares that its path is exact, faster and needs assets:

- The command must be the plain one: model, at least 17 prompt tokens, `--generate N` with N >= 2, and
  optionally `--context`, `--unqualified-context`, `--print-effective-config`. Any other option owns its route
  and keeps the previous behaviour.
- The automatic prompt route is unchanged (same tiling, same automatic prefill profile, same context). MTP
  replaces only the serial decode rows, and its windows are verified against the target, so the tokens equal the
  plain command's.
- `--no-mtp` or `STRIX_MTP=0` selects plain decode. `--baseline` / `STRIX_BASELINE=1` also does.
- No bundle beside the model: plain decode and one line,
  `MTP default (qwen) unavailable; using plain decode: no drafter bundle at ...`.
- A bundle whose sidecar is missing or does not match, or a route that cannot be admitted: plain decode and one
  line, `MTP bundle setup unavailable; using serial decode: ...`. Neither case is a failure.
- The first run of a prompt family generates the 16-row calibration golden once and caches it under
  `$XDG_CACHE_HOME/strix-engine/goldens` (or `~/.cache`); a `goldens/` directory beside the bundle is read first.

Other engines use the same opt-out for their exact drafters (GLM MTP self-speculation, DeepSeek and MiniMax
DSpark, Gemma4/Nemotron MTP drafter), which were already on by default.

### YaRN MTP bundles

Use `--mtp-bundle FILE` or place the JSON at `<model>.mtp-bundle.json`. Bundle
and explicit MTP flags are mutually exclusive. The version-1 object contains
`version: 1` and `options`, a mapping from existing `--mtp-*` flag names to
strings/numbers (or `true` for switches). All entries in the shipped recipe
are required; unknown entries refuse. The same CLI parser validates all pins,
provenance, policy thresholds, budgets and window settings. Sidecar and draft
vocabulary paths are relative to the JSON. Golden paths/pins and draft profiles
are excluded: the bundle carries the expanded pins itself.

Copy [the distill-g recipe](../../../recipes/qwen-distill-g-yarn.mtp-bundle.json)
beside the model, with its matching sidecar and `draft-vocab/` directory. It
preserves the explicit 1M factor-4 MTP asset, policy and budget recipe; the
requested context and YaRN factor still come from the command. This is a
reproduction recipe, not new compiler, GPU, fidelity or workload qualification.

The golden identity hashes the complete bundle, resolved rotary identity,
capacity, numerical profile pair and prompt token chain. If
`<bundle-dir>/goldens/<identity>.golden` exists it is read. Otherwise the engine
uses `$XDG_CACHE_HOME/strix-engine/goldens` (or
`~/.cache/strix-engine/goldens`). A cache miss runs the existing self-generated
v3 calibration dump in process, publishes via a temporary file and atomic rename,
and logs the path once. A new C/factor or calibration identity pays that one-time
cost; it reuses the same prompt pass, so at 131,072 prompt positions the first run
and a cache-hit repeat differed only by run-to-run noise. The bundle/model directory
is never written. Dump or cache write failure
restarts a fresh serial YaRN session with a notice. Calibration is draft-side
self-generated evidence, not an independent target reference; it changes no target
arithmetic or output tokens. Native commands retain their existing routes.
