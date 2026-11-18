# TURBO quality bar

TURBO is a set of speed levers that change the bits of the result. This kit decides, per model, whether the
quality cost of those levers stays inside a limit fixed before the decisive data exists.

Python 3.10+ and numpy only. CPU parts run anywhere; `run_logits.py` and `client.py` talk to an engine/server.
Tests: `python -m unittest discover -s tools/qualification/turbo_bar/tests -t .` (from the repository root).

## Three arms

| arm | meaning |
|---|---|
| **E** | exact route on our weights; the reference of every comparison |
| **R** | a reduced-precision reference pack of the same model, run with controlled/exact arithmetic. Its measured loss vs E is the **limit** TURBO may lose |
| **T** | TURBO: speed levers on our weights; several named variants `T:<name>` |

TURBO passes only when its loss vs E is bounded **below** R's loss vs E with paired 95 % confidence intervals.
Not significant is not a pass: the verdict is INCONCLUSIVE and the sample must be enlarged.

## Frozen protocol (`manifest.py`)

One manifest per model: arms (identity, weights sha256/bytes, binary sha256, args, env, arithmetic
exact|controlled|turbo, `produced_from` for an R built from our weights) and the protocol (checkpoint,
tokenizer and chat-template shas, contexts 8192/32768 plus a 131072 retrieval probe, seeds, sampling
temperature 0 / top_k 1 / top_p 1, corpus sha, tasks sha, margins delta per endpoint, alpha 0.05, bootstrap B and seed).

* `init`, `add-arm`, `freeze`, `check RUNDIR`, `show` (`python -m tools.qualification.turbo_bar.manifest ...`).
* `freeze` stamps a canonical-JSON sha256. Afterwards the protocol, E and R cannot change (a hand edit is
  detected by the core sha); only new T variants are added, each appended to `turbo-ledger.jsonl`.
* Exactly one R per model. Replacing it needs `freeze --new-version --reason TEXT`: the version increases,
  every earlier T is SUPERSEDED, and the analysis refuses to mix versions. R cannot be chosen afterwards.
* Hashing 100 GB is optional: `--sha-from FILE` takes precomputed `sha256sum` output.
* `check RUNDIR` compares the identities, binary and weights shas in `RUNDIR/run.json` with the frozen manifest.
* Freeze task/retrieval inventories with `freeze --tasks-items tasks.jsonl --retrieval-items retrieval.jsonl`.
  These bind every item, hidden check, split and matrix cell into the core. Old unbound scored files are refused.
  HTTP arms also declare `add-arm --effective-route JSON`; this is part of the immutable arm specification.

## Pilot design

* **Corpus** (`corpus.py`): 48 parents, 12 each of code / math / prose / tools. Code and prose come from local
  files (a source file feeds one parent only, order seeded by `--salt`); math and tools text is generated.
  Tokenised by `--tokenize-cmd` (the text file path is appended as the last argument, ids on stdout).
  Per length L (8192, 32768): prompt = ids[:L], continuation = 256 true tokens, and **128 spread positions**
  (one per equal bin of [64, L-1), offset from sha256(salt|parent|L|i)). SEARCH/CONFIRM split per parent by
  sha256 order inside the category. The corpus sha goes in the manifest.
* **Logits**: spread rows come from the engine's teacher-logits spool (`spool.py`, TMLLOG01/02, strict
  validation); the 256 forced continuation steps come from the option-qual runner (`--keep-logits`).
* **Tasks** (`tasks.py`): code (13 templates; execution currently unavailable), math (`ANSWER: n`),
  needle (key-to-value records, answer only by association), json (tool call, exact types). **32 EXPLORE and
  128 CONFIRM** items per kind, all distinct; the verdict uses CONFIRM only.
* **Retrieval** (`retrieval.py`): needle matrix at 8192, 32768 (16 cases each) and a 131072 probe (4): prefix
  new vs restored, concurrency C1 vs C8 (8 keys on one haystack), needle straddling a prefill chunk
  boundary vs mid-chunk. Scored by the generated answer.

## Metrics and statistics

Per position, float64 over the full vocabulary, for an ordered pair a->b: KL(a||b), top-1 disagreement
(argmax, lowest ID on ties), dNLL = NLL_b(label) - NLL_a(label). Averages per parent; the parent is the unit.

Bar per metric: `d = m(E->T) - m(E->R) - delta`, verdict against limit 0 (delta default 0.0). Tasks: for each
item `d = s_R - s_T - delta` (success, paired) and `s_E - s_T` reported next to it with discordant counts.

Continuous CI = the wider envelope of the Student t interval and a seeded percentile bootstrap over units (`stats.py`).
Paired binary tasks instead bound the two discordant probabilities with simultaneous exact binomial
(Clopper-Pearson) bounds, allocating `(1-level)/4` to each of four tails. Their difference is a conservative
interval even with no observed discordances; 128 identical successes cannot clear a 0.01 loss margin.
PASS iff CI_hi < limit; FAIL iff CI_lo > limit; otherwise INCONCLUSIVE with
`n_needed = ceil(n * (halfwidth / |limit - mean|)^2)` and "enlarge sample" (a heuristic, not a power guarantee).
Fewer than 3 units is INCONCLUSIVE.
Where a task or retrieval cell is at a ceiling for all arms, declare a positive margin before freezing.
Retrieval first averages keys within each haystack, then gives each declared matrix cell equal weight.
Its conservative interval averages simultaneous cell bounds (Bonferroni levels); bootstrap units within
each cell are independent haystacks. Fewer than three haystacks in any cell remains INCONCLUSIVE.
SEARCH parents are reported separately; only CONFIRM parents contribute to logits qualification.
Intervals are per comparison, not a correction for repeated variant selection. Frontier selection remains
exploratory; optional sample enlargement or confirmatory selection requires a separately predeclared protocol.

## Controls and reports

* E vs E-repeat must be bitwise identical (KL 0, top-1 100 %, dNLL 0) or the analysis is INVALID
  ("E not reproducible"); the placebo CI of E->E-repeat dNLL must contain 0; an optional rounding-only control
  arm is reported as the noise floor.
* The spread pass runs on the tiled known-prefix route. If an arm's spread logits equal E bit for bit while its
  continuation differs, the spread result is reported NOT-MEASURED (lever inactive) and never counts toward PASS.
  Arms with different prompt ids or positions are refused. Engine receipt route fields
  (`teacher_forced_tiled_lm_head`, `teacher_logits_scope`, `teacher_logits_rows`) are kept per output.
* Results are reported by source (spread / continuation), length and category. Overall logits verdict =
  all three metrics over "all" and every length.
* `frontier.py`: Pareto set (speed vs KL then dNLL), 95 % t interval of the timing pairs; among PASS variants
  the fastest by median, every PASS variant whose speed CI overlaps it is within timing noise, and the least
  damaging of those (lowest KL, tiebreak dNLL) is selected.
* `report.py` writes `receipt.json` and `REPORT.md` with PASS / FAIL / INCONCLUSIVE / INVALID per variant.

## Command sequence

```
python -m tools.qualification.turbo_bar.tasks gen --out tasks.jsonl            # prints the tasks sha
python -m tools.qualification.turbo_bar.corpus --out corpus --salt SALT --tokenize-cmd "TOK MODEL encode" \
    --code-files 'src/**/*.py' --prose-files 'docs/**/*.txt'                   # corpus/corpus.sha256
python -m tools.qualification.turbo_bar.retrieval --out retrieval.jsonl --chunk 4096 --tokenize-cmd "TOK MODEL encode"
python -m tools.qualification.turbo_bar.manifest init --manifest M.json --model NAME --set 'corpus_sha256="..."' ...
python -m tools.qualification.turbo_bar.manifest add-arm --manifest M.json --role E --identity ... --weight W --binary BIN
python -m tools.qualification.turbo_bar.manifest freeze --manifest M.json --tasks-items tasks.jsonl --retrieval-items retrieval.jsonl
python3 -m tools.qualification.turbo_bar.run_logits spread --corpus corpus --manifest M.json --bin BIN --out OUT
python3 -m tools.qualification.turbo_bar.run_logits continuation --manifest M.json --out ARMS.json   # then option_qual.run
python -m tools.qualification.turbo_bar.manifest check --manifest M.json OUT
python -m tools.qualification.turbo_bar.client run --manifest M.json --url URL --items tasks.jsonl --arm E --out res [--concurrency 8]
python -m tools.qualification.turbo_bar.client score --manifest M.json --items tasks.jsonl --results res
python -m tools.qualification.turbo_bar.losses layout --corpus corpus --spread OUT/spread --continuation CONT \
    --arms E,R,T:x,Erep --manifest M.json --out layout.json
python -m tools.qualification.turbo_bar.losses analyze --layout layout.json --out losses.json
python -m tools.qualification.turbo_bar.report --manifest M.json --losses losses.json --tasks res \
    --retrieval resr --frontier F.json --out report
```

`run_logits spread` currently supports Qwen only; other engines need an adapter. It runs one process
per (case, arm) (`Erep` is a second E run), verifies weights and complete output receipts before reuse,
stops before `--soft-seconds` and exits 2 when it must be run again. Incompatible outputs preserve the
old run record and are refused. Existing outputs without bound receipts must be collected afresh.

Qualification requires both prescribed spread positions and all 256 continuation steps. Continuation
receipts must carry the same `output_contract` binding, payload hash and effective-route hash as spread
receipts; the existing option-qual runner does not yet emit this adapter contract. Unbound continuation
files are refused. Spread-only diagnostic layouts remain INCONCLUSIVE.

HTTP responses must supply `strix.qualification_identity` with `identity`, `binary_sha256`, the ordered
`weights_sha256` list and the declared `effective_route` object. Missing or mismatched identities cannot
qualify. Retrieval additionally needs an authoritative `strix.qualification_route`: applied `mode`,
actual `concurrency`, `cut`, `chunk`, `needle_start`, `needle_end`, and boolean `prefix_restored`.
These are required adapter contracts, not a claim that current servers emit them. Restored cells require
successful warm-up as well. Missing evidence is NOT-MEASURED and prevents PASS. Use `--suite retrieval`
for both retrieval client commands. Failed requests cannot disappear from the frozen item inventory.

Code scoring refuses all execution until a qualified OS sandbox backend and trusted external test
controller are available. Isolation must exclude network access, host secrets and writable host mounts,
enforce resource limits, and tear down the whole process tree. The controller must validate completion
and outcomes of every case; interpreter exit status alone is insufficient. The full task suite therefore
cannot currently qualify. Non-code diagnostics do not substitute for its missing code measurements.

The K-quant reference pack retains the original block geometry and byte size: reduced precision does
not imply smaller disk or resident memory. Synthetic round trips are not independent known-byte or
native loader/kernel fidelity qualification; those gates remain open.
