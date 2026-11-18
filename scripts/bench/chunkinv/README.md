# chunkinv bench kit (MiniMax M3.1 prefill numeric order)

Measures the opt-in invariant prefill order (`STRIX_MMX_PF_ORDER=invariant`, identity `mmx-pf-linear-s1-v1`, MoE +
dense) against today's adaptive default on a two-rank pair. No engine change lives here; the only source patch is the
diagnostic null control, applied to a scratch copy for the gate binary and never merged.

Configuration: `kit.env.sh` (`KIT`, `BIN`, `MODEL`, `PROMPTS`, `PEER`, `GATE`). Every GPU step runs inside one pair GPU
turn of the caller's scheduler (120 s or less); sealing and comparison are CPU only and run outside it.

| file | role |
|---|---|
| `diag_attn_ulp_patch.py` | adds `STRIX_MM_DIAG_ATTN_ULP=L:seed` (one ulp on layer L's attention output) to a scratch engine.hip |
| `gate_run.sh` | one arm: two ranks, logits + selection dumps, MemAvailable, kfd evicted_ms |
| `gate_annot.sh` | binds each null-control event to its prompt (run before sealing; `gate_seal.sh` calls it) |
| `gate_seal.sh` | engine rc, execution witness and seal on both ranks (`tools/strix-gate/fast_logit_gate.py`) |
| `gate_cmp.sh` | absolute + floor v1.2 comparison (onset 0) with every sealed `UNIT-n0s1..5` null |
| `gate_timing.py` | TTFT per session, busy, sclk, GPU clocks per token, evicted_ms growth |
| `pair_run.sh` | generic pair run (TTFT, state / tokens FNV, MemAvailable, evicted_ms) |
| `layercut_ticket.sh` | rank side of a layer-cut run at cut block C (PC producer started separately) |

Gate recipe (one unit, e.g. 32K `b32k`):

```
gate_run.sh b-e-a 115 32832 b32k STRIX_MMX_PF_ORDER=adaptive                          # reference
gate_run.sh b-i-a 115 32832 b32k STRIX_MMX_PF_ORDER=invariant                         # candidate
gate_run.sh b-n0sS 115 32832 b32k STRIX_MMX_PF_ORDER=adaptive STRIX_MM_DIAG_ATTN_ULP=0:S   # S = 1..5
gate_seal.sh RID b32k                                                                  # after each turn
gate_cmp.sh b b-e-a b-i-a b32k
```

Use five nulls: with three, a candidate that matches the nulls fails a mean + 2 sd check about 11 % of the time.
Run directories are never reused. A rerun after an engine failure gets a new RID.

## Multi-unit long-context gate (gate32b)

`mkunits.py` builds six independent 32K units (prose, docs, code, json, chat, mixed) from this repository with seven
checkable answers each (five needles at 8-92 % depth, one two-hop pair, one question over the needles).
`gate32b_chain.sh` runs per unit the reference, the candidate and NULLS one-ulp L0 nulls, one pair turn per arm, plus
optional warm arms (a short warm-up session first; only the measured session is sealed), then `gate32b_stats.py`:
per-unit and pooled signed dNLL with a t(5) 95 % CI and a hierarchical bootstrap, the null spread, KL q50/q90 against
the nulls, paired task accuracy (cluster bootstrap + McNemar) and warm == cold bit checks. The decision rule is printed
with the numbers. Hosts, model and binary come from the environment (KIT MODEL PROMPTS PEER BIN GATE TURN PY).
