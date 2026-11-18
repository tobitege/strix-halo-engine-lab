# Scoped A/B gate runner

`gate.py` is a Python 3 standard-library command launcher and evidence comparator.
It does not load models, resolve execution plans, generate identity text, or select
arithmetic or schedules. Model semantics and tensor/state mappings belong in explicit
engine adapters. Description files must not supply printed identities. Existing
engine gates and independent reference establishment remain necessary.

From the repository root, with Python 3.9 or newer:

```sh
python tools/strix-gate/gate.py tools/strix-gate/example-job.json --output gate-evidence
python -m unittest discover -s tools/strix-gate -p 'test_*.py' -v
```

The example executes synthetic CPU commands for two ranks in each arm. Its emitted
identity, startup, receipt and digest strings are test literals, not engine results.
Keep generated evidence and reports untracked and private: receipts retain the full
inherited environment, commands, working directories and unmodified output bytes.
The output directory must not already exist. Relative paths resolve against the
launcher's current directory, except a command's argv paths use its optional `cwd`.

## Job schema: `strix-gate/v1`

The complete executable example is [example-job.json](example-job.json).
Unknown fields, duplicate JSON keys, nonfinite JSON numbers, unsupported schemas,
invalid types, duplicate selectors, and missing or duplicate rank commands are errors.
The schema has these required fields:

| Field | Contract |
|---|---|
| `schema` | Exactly `strix-gate/v1` |
| `engine`, `ticket` | Nonempty strings, retained as supplied |
| `sources` | Object with nonempty string identifiers `A` and `B`; caller provenance labels, not verified source/build attestations |
| `ranks` | Sorted unique nonnegative integer array; `[0]` for a single rank; **`[0,1]` for a pair** |
| `commands` | Objects `A` and `B`, each an array with exactly one command for every required rank |
| `compare` | Object with arrays `stdout_keys`, `stderr_witnesses`, `digest_fields`; at least one selector overall |

Optional `env` is a string-to-string environment overlay. Optional `timeout_s` is a
finite positive number of seconds (default 60) for the whole arm, including launch.
Every command contains `rank` and a nonempty `argv` string array. Optional `cwd` is a
nonempty path; optional command `env` overrides job `env`, which overrides the
inherited environment. Commands run without shell expansion, with stdin closed.
The caller must declare all required ranks; the engine label does not infer topology.

Every arm launches all its ranks **before waiting for any rank**. Arms run in A/B
order. Output goes directly to separate binary files, avoiding pipe backpressure
between peers. Exits, launch failures and timeouts are recorded per arm/rank.
Timeouts terminate the process group on POSIX and use the system `taskkill /T /F`
with direct-process fallback on Windows. Wrappers must stay in the foreground and
wait for their children; detached processes and remote execution lifecycle management
are outside this runner's contract. A job may use foreground remote launchers, but
rank cleanup on the remote host remains their responsibility.

## Evidence selectors

All selectors apply independently to **every required rank in both arms**.
Comparisons are A-rank-N versus B-rank-N; ranks are not presumed equal to each other.

- `stdout_keys`: strings selecting fields from a whole stdout JSON object,
  JSON object lines, or anchored `key=value` / `key: value` lines. Dots traverse
  nested JSON objects (literal dotted keys in assignment lines). JSON values keep
  their types, list order and repeated record order; object member order is ignored.
  Assignment values keep text after delimiter/outer whitespace trimming. A present
  JSON `null`, empty list or empty string is a value; a missing key is not evidence.
- `stderr_witnesses`: UTF-8 regular expression strings, applied to **raw bytes**
  with multiline mode. Compare ordered full-match bytes, represented as hex in the
  canonical extract. Capturing groups do not remove other matched bytes. No match
  or an empty match fails. Use anchored whole-line patterns such as
  `^auto-prefill identity [^\n]+$` and
  `^rank [01] prefill \[mla\] identity [^\n]+$` to include complete witness text.
  CR bytes are preserved and compared. Selectors deliberately omitting text cannot
  establish equality of that omitted text.
- `digest_fields`: objects `{ "stream": "stdout", "key": "state_fnv", "format": "fnv64" }`.
  Stream is `stdout` or `stderr`, extracted using the same key rules. Formats are
  `fnv64` (exactly 16 hex characters) and `sha256` (64). All occurrences must be
  nonempty valid strings and compare literally, including hex letter case.

Duplicate-key or nonfinite JSON evidence is refused. Other non-JSON log lines are
ignored unless an assignment selector matches them. A malformed JSON line cannot
supply a requested field. Repeated records remain ordered and must have the same
count and values. stdout/stderr capture itself never decodes or normalizes bytes;
key extraction uses UTF-8 with lossless surrogate escaping.

## Receipt and verdicts

`result.json` has `engine`, `ticket`, `sources`, `commands`, `runs`, `verdict`,
`elapsed_s`, `files`, and `differing_fields`, plus a scope statement. Commands retain
resolved `cwd`, argv, arm, rank, label and effective environment. Runs retain arm,
rank, stdout/stderr artifact names, exit code, timeout flag and launch error.

`files` maps relative artifact paths to SHA-256: `job.json`, each raw stdout/stderr
file, and each arm's canonical extract. `result.json` excludes itself from this map.
Canonical JSON uses sorted object keys, ASCII escapes and LF endings. Every mismatch
names the field, rank, arm, reference arm and expected/actual values; evidence and
execution failures also name arm/rank and their expected/actual condition.

Verdict priority is `ERROR` (execution failure), `MISSING-EVIDENCE` (absent/invalid
requested evidence), `MISMATCH` (different present values), then `PASS`. Failure
information is retained even when a higher priority verdict applies. CLI exit status
is 0 for `PASS`, 1 for other completed verdicts and 2 for invalid input or I/O errors.
Invalid jobs are refused before execution and do not emit a completed receipt.

`PASS` means equality **only for the requested evidence**. It does not establish
complete model/token/logit/state fidelity, a reference's correctness, byte equality
of unselected logs, numerical qualification, or a native GPU execution. Request
complete token IDs, all required state/logit digests, and complete identity witnesses
for the engine/configuration being qualified. Do not replace engine-specific gates
or report `EXACT` solely because this runner returned `PASS`.

## CPU tests

The command above runs **25 tests**: equal captures and artifact hashes, value,
witness and digest mismatch, missing key/witness/digest, malformed digest, evidence
missing in both arms, nonzero exit, timeout with partial capture, launch failure,
both ranks, rank-1-only mismatch/missing witness, concurrent pair rendezvous,
ordered tokens, JSON type distinctions, environment precedence, output preservation,
CLI, extraction, stderr digests and invalid input. Synthetic stderr includes the
`decode_identity`, `auto-prefill identity ...`, `decode attention identity ...`,
`rank R prefill [mla] identity ...`, build and startup witness forms. Synthetic
stdout includes Qwen, DeepSeek and MiniMax receipt keys. These checks execute host
fixtures; they do not execute any engine binary.

## FAST logit gate: `fast_logit_gate.py`

A FAST identity (one that changes result bits) is qualified against the exact identity on
teacher-forced logits. `fast_logit_gate.py` (Python 3.9+, numpy) is the shared reducer; the
engine-specific part is only the launcher that runs one arm and the dump hooks of that engine.

```sh
# on each node, right after the arm's engine process exits (one fresh directory per run)
python tools/strix-gate/fast_logit_gate.py seal --dir RUN --rank 0 --arm X --run-id ID \
    --binary BIN --rows 256 --prompt chat=chat.u32 --env MODE=fast --select --attest sel_mode=3 \
    --not-before START_UNIX
# anywhere, with every rank's dumps and manifests in one directory per arm
python tools/strix-gate/fast_logit_gate.py compare REF CAND --ranks 0 1 --tokens chat=chat.u32
python tools/strix-gate/fast_logit_gate.py compare REF REF2 --ranks 0 1 --tokens chat=chat.u32 --expect-identical
```

`seal` refuses a missing, short, stale (older than `--not-before`), nonfinite or already sealed
dump and otherwise writes `manifest-r<rank>.json`: run id, arm, rank, binary sha256, the
launch environment, each prompt's token-file sha256 and each dump's size, sha256 and width.
`--attest sel_mode=N` must equal the mode the engine itself wrote into its selection dump, so an
arm whose switch was not applied cannot be sealed under the wrong name.

`compare` re-hashes every dump against its manifest, requires different run ids for the two
arms and the same prompts, rows and vocabulary slices, then prints per prompt and pooled
KL(ref || cand), top-1 agreement and the NLL change on the real next tokens. The default rule
is pooled KL <= 0.02, top-1 >= 0.93, dNLL <= 0.01 (`--max-kl`, `--min-top1`, `--max-dnll`).
`--expect-identical` is the control for two runs of one identity: any logit or selection
difference fails. Exit 0 = pass, 1 = fail, 2 = identity or input defect (no verdict).

### Floor-relative dNLL (DeepSeek V4.1 FAST prefill only)

```sh
python tools/strix-gate/fast_logit_gate.py compare EXACT CAND --ranks 0 1 --tokens ... \
    --dnll-rule floor --engine ds41 --profile fast-prefill --null-arm NULL_S1 --null-arm NULL_S2 --null-arm NULL_S3
```

Labels are the true next tokens, so a change that carries no information about the text costs
E[dNLL] = KL in expectation: on DS41 every layer-0 change (a one-ulp null included) re-routes MoE
rows and lands at KL ~0.01, dNLL +0.011..+0.014, above the absolute 0.01. For that profile the dNLL
limit is relative to this floor: `--dnll-rule floor` makes it mean + 2 sd (n - 1) of the pooled dNLL
of at least three layer-0 null-control arms (`--null-arm DIR`, sealed runs) measured against the
same reference, on the same prompts and the same binary as the candidate. KL <= 0.02 and
top-1 >= 0.93 are unchanged. Example (DS41 20 prompts x 64 rows): nulls +0.01408 / +0.01140 /
+0.01111 give mean +0.01220, sd 0.00164, limit +0.01547; arms +0.01203 and +0.01458 pass, +0.01600 fails.

Refused with exit 2 (no verdict): fewer than three nulls, a null whose binary differs from the
candidate's, whose prompts or token files differ, or whose run id repeats the reference, the
candidate or another null; `--null-arm` without `--dnll-rule floor`; floor with
`--expect-identical`; and any `--engine` / `--profile` pair that `DNLL_POLICY` in the script does not
grant the floor rule. Today the table holds two entries, `ds41` / `fast-prefill` and `minimax` /
`fast-prefill` (next section); every other engine and profile keeps the absolute dNLL <= 0.01 (the
default, unchanged). Widening the table is a policy
decision. Under the floor rule the report prints and records (`pooled.dnll_rule` in `--json`) the
rule, engine, profile, each null (run id, arm, env, rows, KL, top-1, dNLL), mean, sd, the limit and
the absolute 0.01 verdict as a reference column; with the default rule `pooled.dnll_rule` records
`abs` and the limit.

Floor comparisons also require `seal --execution-witness execution-rR.json` on
the reference, candidate and every null, on every rank. Missing or incompatible
evidence is exit 2, with no verdict. Arm names, `--env` and `--attest sel_mode`
do not establish control eligibility. The launcher/engine must emit this JSON
from the executed configuration and the completed dumps; sealing preserves it
as `execution` and checks its binding to the run:

```json
{
  "schema": "strix-fast-execution/v1",
  "run_id": "unique-run-id", "rank": 0,
  "binary_sha256": "<64 lowercase hex digits>",
  "engine": "ds41", "profile": "exact",
  "model_sha256": "<64 lowercase hex digits>",
  "numerical_config_sha256": "<64 lowercase hex digits>",
  "control": {"kind": "layer0-null", "layer": 0, "magnitude": 1, "unit": "ulp"},
  "prompts": {"<prompt-name>": "<same structured entry as the sealed prompts map>"}
}
```

`prompts` must contain the full token counts/hashes, row counts and dump
names/sizes/hashes/widths (plus selection records/mode when present) described
above, computed by the producer. The placeholders are not usable evidence.
`model_sha256` identifies the complete ordered model asset bundle;
`numerical_config_sha256` hashes the canonical resolved numerical configuration,
including effective switches, arithmetic, tensor layouts and selection policy,
excluding only the separately recorded control perturbation and its random seed.
Producers must use the same canonical encoding across arms and ranks.
The exact reference and nulls must match both identities and use profile `exact`;
the candidate must match the model and witness `ds41` / `fast-prefill`.
Reference and candidate use `"control": null`. Only the one-ULP layer-0 identity
above is admitted as a control. Rank identities must agree; each rank binds its
own binary and dumps. Reports include the first rank's execution evidence.

Trust boundary: these are trusted launcher/engine statements, not signatures or
proof derived from logits. The gate cannot detect a producer that misreports
execution or a deliberately rewritten manifest. CLI engine/profile labels are
checked against the sealed witness. This change defines the consumer contract;
it does not add a native witness producer. Runs without producer evidence cannot
use the floor rule; do not create evidence by relabelling existing runs.

Producer: `strix-deepseek41-prefill-quality MODEL manifest.tsv OUT RANK BUDGET GATE_DIR` (gate mode). Per prompt
of n tokens each arm prefills P = (n - 64) / 128 * 128 rows from position 0 with the arm's policy (the
exact suffix replay, or `fast3b<R>`'s bounded decoder replay), then teacher-forces the 64 rows [P, P + 64)
from that state. Rank 0 writes 65 rows per prompt (the prefill's last row, then the 64) as
`GATE_DIR/<arm>/q-<prompt>-tf-r0.f32`, the prompt's first P + 64 tokens as `GATE_DIR/tok/<prompt>.u32` and
`GATE_DIR/<arm>/execution-r0.json` from the executed configuration (profile `exact` for `exact` and the
`n_l0_s<seed>` layer-0 nulls, `fast-prefill` for `fast3` / `fast3b<R>`). When the prompts run in several
processes (one per GPU ticket), `merge_parts.py RUNDIR RUN ARM...` joins the per-process witnesses of an arm
(refusing any disagreement on binary, model, numerical configuration, profile or control, a prompt in two
parts, or a dump that no longer matches its entry) into one with `run_id` `RUN-ARM`; then `seal --rows 65
--execution-witness` and `compare --ranks 0`.
Mean plus two sample standard deviations from three controls is an empirical
policy threshold, not a confidence bound or model-fidelity result. Synthetic
host tests establish validation and arithmetic only; compiler, native execution,
model fidelity and complete-workload qualification remain separate gates.

DS41 gate arms explicitly disable inherited decoder replay unless their label sets a bound.
Witnesses record the resolved replay mode and hash the actual target and effective overlay
contents; both ranks must agree on that content. The producer uses `row_split=false` and
128-aligned prefill boundaries. Its evidence does not qualify row-split serving or arbitrary
tails. Whole-run replay rejects snapshots, group passes and decoder deferral; serving needs
an explicit admission contract for scheduler/cache operations before qualification.
### Onset-matched floor (MiniMax FAST prefill)

```sh
python tools/strix-gate/fast_logit_gate.py compare EXACT CAND --ranks 0 1 --tokens ... \
    --dnll-rule floor --engine minimax --profile fast-prefill --null-arm N1 --null-arm N2 --null-arm N3
```

On MiniMax one fp16-surviving ulp anywhere flips near-tie block selections (top-16) and expert
routes (top-4 of 128), so a layer-0 null already lands at KL ~0.5, top-1 ~88 % and a dNLL that
swings by +-0.1 between seeds: a floor built on it would pass anything. The rule therefore matches
the null to the candidate: each null is a one-ulp change of the attention output (the tensor MSA
flash replaces) at the candidate's FIRST FAST layer, and the gate checks the null's control layer
against the candidate's witnessed layer set. With >= 3 such nulls (distinct seeds, same binary,
prompts and reference) the candidate must pass all of (`MM_FLOOR` in the script):

| check | rule |
|---|---|
| `kl`, `top1` | pooled KL <= max(0.02, mean + 2 sd of the nulls); top-1 >= min(0.93, mean - 2 sd): the floor only relaxes the absolute bar |
| `dnll` | pooled dNLL <= min(max(0.01, mean + 2 sd), 0.03): never tighter than the absolute 0.01, never looser than 3 x 0.01 |
| `kl_q50`, `kl_q90` | each per-row KL quantile <= max(mean + 2 sd of the nulls' same quantile, tolerance 1e-3 / 0.02): a diffuse shift cannot hide under heavy null tails, and a one-ulp null's median row (rounding only, KL ~1e-5) does not demand bitwise arithmetic; `kl_q99` is reported only (the tail is chaos, and one null site flips fewer near-ties than a FAST layer set) |
| `sign` | rows with abs(dNLL) >= 1e-3: one-sided binomial P(X >= worse) >= 0.01 against p0 = max(0.5, pooled null worse share); more chaos adds rows, not slack |
| `copy` | copy rows (the last 8 prompt tokens occurred earlier and were followed by the true token; token-only, the same rows for every arm): next-token misses beyond the reference's <= worst null + max(1, ceil(1 %)) |

The absolute KL / top-1 / dNLL verdict is printed and stored (`pooled.abs_pass`) as a reference.
The candidate must differ from the reference only in its FAST layer set: every receipt lever other
than `minimax.msa_flash` / `minimax.linear_order` must be equal, and so must every recorded switch outside
the explicit FAST allowlist (`STRIX_MM_PF_MSA_FLASH`, `STRIX_MM_PF_MSA_FLASH_LAYERS`, `STRIX_MM_FAST_PREFILL`,
`STRIX_MMX_PF_ORDER`). The chunk-invariant linear order (`minimax.linear_order` =
`invariant:mmx-pf-linear-s1-v1`) changes MoE / dense rounding in every layer, so its FAST layer set is all 60 layers:
onset 0, layer-0 one-ulp nulls, and an exact reference that differs only by `STRIX_MMX_PF_ORDER`.
Receipts omit defaulted switches and cannot distinguish them from unconsulted switches; a
one-sided entry outside that allowlist is therefore refused, even if it explicitly selects a default.
Mean + 2 sd of three controls is a policy threshold, not a bound: a candidate drawn exactly like
the nulls fails one such check about 11 % of the time (t with 2 degrees of freedom).

### Witness producer: `witness`

```sh
python3 fast_logit_gate.py witness --dir RUN --rank R --run-id ID --binary BIN --rows 256 --select \
    --attest sel_mode=4 --prompt NAME=TOKENS.u32 --engine minimax --engine-log RUN/rR.log --model MODEL_DIR \
    --out RUN/execution-rR.json
python3 fast_logit_gate.py seal ... --execution-witness RUN/execution-rR.json
```

Run by the launcher on the rank's node after the engine exits. The MiniMax adapter reads the
rank's process route receipt (`strix-route-receipt {...}`): engine; profile `exact` (MSA flash
off, EXL3, default prefill identity), `fast-prefill` (MSA flash on, otherwise the same) or
`other`; `fast` = MSA flash mode (10 + `STRIX_MM_PF_MSA_FLASH`: 11 one-term, 12 two-term) and
layers; `numerical_config_sha256` over the effective levers and switch set without output-only
switches, and `config` = those levers and switches (for the comparison above).
The index-projection WMMA member additionally requires `minimax-index-proj-executed` records,
emitted after the prefill completion fence: matching rank, executed layer mask and each sealed
prompt's token count and little-endian token FNV-1a64. A selected startup lever alone is insufficient.
The control
comes only from the engine's own null-control lines in a lab diagnostic build:
`DIAG attn ulp: layer L seed S rows N pos0 P values V prompt "NAME"`.
The prompt name is a JSON string matching a sealed prompt. All events must use one layer and seed,
have positive row/value counts and a row span inside that prompt; every sealed prompt must have
an event. Unlabelled, malformed and incomplete control logs are refused. The lab emitter must
supply these prompt identities; this gate does not add native diagnostic instrumentation.
`model_sha256` binds every file's relative path, size and full-content SHA-256, including all tensor
payloads and large-file middles. Producing a witness reads the complete model bundle; keep assets
immutable from execution through witness generation. Regenerate older header/sample-based witnesses.
The prompts map is the one `seal` computes. Same trust boundary as above: launcher statements,
not proofs of which assets, binary or log belong to a completed run.

When both arms carry selection dumps (`PREFIX-<prompt>-sel-r<rank>.u32`; MiniMax writes them
with `STRIX_MM_QUALITY_SELECT=1` next to `STRIX_MM_QUALITY_OUT`), `compare` also reports, per
layer, the (row, head) slots whose selected block set differs and the first layer and position
where the arms part. This is a cross-run comparison: after the first difference the arms see
different activations, so later differences include propagation, not only local rounding.

`python -m unittest discover -s tools/strix-gate -p 'test_*.py'` covers it with synthetic dumps.

### TURBO relative bar: `compare --mode turbo`

`--mode fast` (default) is the absolute FAST bar above, unchanged. `--mode turbo` judges T (the candidate) against a
reference pack R measured on the same E (the exact arm): `compare E T --mode turbo --reference-pack-arm R_DIR ...`.
Per row, U95(KL_T-KL_R), U95(disagreement_T-disagreement_R) and U95(dNLL_T-dNLL_R) must each be <= 0 (one-sided 95 %,
paired rows). Optional absolute safety limits: `--turbo-max-kl`, `--turbo-min-top1`, `--turbo-max-dnll`. `--null-arm`
controls are measured against E and reported (not a verdict input).

The normal-approximation intervals treat sampled rows (and paired T-minus-R differences) as independent
observations with finite variance and enough samples for the approximation. Tokens within a prompt may be
correlated, making these intervals overconfident; they do not establish prompt-level uncertainty or model
qualification. Such claims require independently sampled prompts and prompt-level resampling or another
validated treatment of clustering. Task Wilson intervals likewise assume independent Bernoulli trials.

Pack identity: `seal --model-sha SHA` records the weights/pack sha (else the execution witness `model_sha256` is used).
E and T must share it unless `--candidate-sha` declares T's; R may differ from E only with `--reference-sha` equal to R's
sealed value. Every participating rank, including null controls, needs a valid SHA-256 digest. A declared digest
must agree with an available execution witness. Missing, invalid, undeclared or mismatching identities are errors (exit 2).

`--task-results JSON`: `{"tasks":[{"name":"t1","category":"code|math|retrieval|tools|json|prose","E":{"passed":8,"total":10},"T":{...},"R":{...}}]}`.
Per category and overall the receipt carries passed/total/rate and a Wilson 95 % interval per arm. Checks per category:
T's drop vs E must not exceed R's drop unless the T and R Wilson intervals overlap; absolute drop of T vs E <= 5 points
code/math, 2 retrieval, 1 tools (override `--task-drop-limit CAT=POINTS`); json must pass 100 % for T.

Verdicts and exit codes: PASS 0, FAIL 1 (every failing check is named), error 2, NO-REFERENCE 3 (no `--reference-pack-arm`
or a task without `R`; never PASS, but a failing check still reports FAIL).

`--receipt OUT.json` writes `strix-turbo-receipt/v1`: `verdict`, `failing`, `commit` (`--commit`), `binary_sha256` {E,T,R},
`pack` {calibration (`--calibration`), E_sha, T_sha, R_sha}, `run`, `hardware` (`--run-manifest JSON`, free keys such as
hardware, gpu_clock_mhz), `routes` {effective_config: lines of `--effective-config FILE`, execution_witness from the seals},
`metrics` (mean + 95 % interval for T and R), `checks`, `tasks`, `controls`, `performance` (`--perf JSON`: tok_s,
bytes_per_token, ttft_ms, p95_ms; absent keys are null and listed in `missing`).

## GLM scored prefill compare: `glm_score_nll.py`

The GLM rank captures the last N prompt rows' logits with `STRIX_GLM_PF_SCORE_LAST=N` and
`STRIX_GLM_PF_SCORE_OUT=DIR/NAME-tf` (one vocabulary slice per rank). `glm_score_nll.py --dir DIR --ref X1 --test I1
--prompt P.u32 --vocab-size V [--rows 256] [--json OUT]` (V is the model's full vocabulary size) prints NLL ref/test, dNLL (offset-1 target = the scored rule), mean KL and
top-1 agreement. The prompt must be the length being gated: a `.u32` file holds 4 bytes per token (a 32,768-byte file
is an 8K prompt). Tests: `python -m unittest test_glm_score_nll` in this directory.
