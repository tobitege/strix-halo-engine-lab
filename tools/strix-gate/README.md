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

When both arms carry selection dumps (`PREFIX-<prompt>-sel-r<rank>.u32`; MiniMax writes them
with `STRIX_MM_QUALITY_SELECT=1` next to `STRIX_MM_QUALITY_OUT`), `compare` also reports, per
layer, the (row, head) slots whose selected block set differs and the first layer and position
where the arms part. This is a cross-run comparison: after the first difference the arms see
different activations, so later differences include propagation, not only local rounding.

`python -m unittest discover -s tools/strix-gate -p 'test_*.py'` covers it with synthetic dumps.
