# MoE route pruning: `moe_route_prune` (optional FAST toggle, default OFF)

Status: optional, bit-changing prefill toggle. It is in no profile: `--profile fast` does not apply it and
`STRIX_QWEN_ALLON` does not switch it on. Unset, `off` or `0` keeps the route plan and the identity bytes unchanged.

```
STRIX_QPF2K_MOE_ROUTE_PRUNE=cap9        # or: --prefill-moe-route-prune cap9
```

Rollback: unset the variable (or pass `off`) and restart the process. The request is read once per process.
The master `--prefill-exact-fp16`, config `prefill_exact_fp16 = true`, or
`STRIX_PREFILL_EXACT_FP16=1` disables pruning in both execution and identity.
Values use decimal notation without whitespace or control characters; identity tags
use the canonical spelling of the parsed policy (for example, `cap09` becomes `cap9`).

`STRIX_MOE_ROUTE_CENSUS=<file>` is a diagnostic capture: synchronization, host copies,
allocation and flushed I/O perturb every captured call. Disable it for performance
qualification. Records from one process are serialized; use a separate file per process.
Open, short-write and flush failures raise a path-specific error and permanently stop
further appends to that writer; discard a failed capture before restarting.

## What it does

After the router has picked the top-k experts of every prefill row, the policy drops the lowest-weight assignments of
each row before the shared CSR route plan is built. Every gate/up/down kernel walks the plan, so a dropped assignment
costs nothing; the ordered merge treats a dropped slot as a zero contribution. Decode rows and window rows never see it.

| spec | keeps |
|---|---|
| `cap<k>` | the k best assignments of each row (1 <= k <= 31) |
| `rel<f>` | assignments whose weight is at least f times the row's weight sum (0 < f < 1) |
| `topp<f>` | the best assignments until their mass reaches f times the row's weight sum (0 < f < 1) |
| trailing `n` | the kept weights are rescaled to the row's original sum (`cap9n`) |

The best assignment of a row is always kept; ties go to the lower slot. The policy is one shared device function
(`src/moe_core/route_prune.hpp`) and one `Prune` instantiation of the shared plan kernels
(`src/moe_core/device_plan.hpp`); the default instantiations are unchanged. It is wired for the Qwen FAST prefill tail
(DA16 merge and commit); any other tail refuses the request instead of merging a pruned plan with an unpruned merge.
When active, stderr carries `qpf2k-moe route prune active` and the prefill identity gains `+rp<spec>`, so prefix caches
and receipts of a pruned prefill never mix with unpruned ones.

## Why there is no free setting

On Qwen3.8-Flash-Next (top-10 of 512 experts, 48 layers) the route weights are flat: the tenth expert still carries
6.3-6.5 % of a row's mass, 0.03 % of the assignments are below 2 % of their row and none below 1 %. Every setting that
saves time removes real expert mass, so every setting changes logits. `cap9` removes 10 % of the assignments,
`cap8` 20 %.

## Measured cost and gain

One node, FAST profile as the base (`--profile fast`), real-text prompts. Quality is the composite FAST gate measured
against the exact route (pooled KL <= 0.02, top-1 >= 93 %, dNLL <= 0.01) over 256 forced continuation rows per case:
eight 8,192-token cases (chat, code, English, Spanish) and two 32,768-token chat cases. "w/o 10" drops the ten
highest-KL rows of the pool.

| arm | 8K pooled KL / top-1 / dNLL | 8K w/o 10 | 32K pooled KL / top-1 / dNLL | 32K w/o 10 |
|---|---|---|---|---|
| FAST (base) | 0.0042 / 97.9 % / -0.037 | 0.0036 / 98.2 % / -0.036 | 0.0154 / 94.1 % / -0.075 | 0.0113 / 95.0 % / -0.073 |
| `cap9` | 0.0053 / 97.1 % / +0.005 | 0.0044 / 97.3 % / +0.005 | 0.0175 / 94.1 % / -0.054 | 0.0125 / 94.6 % / -0.054 |
| `cap8` | 0.0109 / 96.7 % / -0.022 | 0.0096 / 96.8 % / -0.021 | 0.0300 / 92.8 % / +0.030 | 0.0230 / 93.2 % / +0.012 |
| `cap9n` | 0.0077 / 96.7 % / -0.030 | 0.0065 / 96.9 % / -0.028 | 0.0251 / 94.3 % / +0.161 | 0.0195 / 94.8 % / +0.147 |
| `cap8n` | 0.0090 / 96.8 % / -0.024 | 0.0075 / 97.0 % / -0.022 | 0.0749 / 90.0 % / +0.335 | 0.0613 / 91.4 % / +0.311 |

The deviation is not uniform. One chat case is sensitive at both lengths (the FAST base alone gives KL 0.030 and
top-1 89.1 % on it at 32,768 tokens); on that case `cap9` stays at the base's level (0.033), `cap8` reaches 0.057 and
the renormalised variants are worse (`cap9n` 0.048, `cap8n` 0.146). On the other seven 8,192-token cases every arm is
within KL 0.008 pooled. Renormalising removes the small positive dNLL drift on ordinary cases and amplifies the error on
the sensitive one, so the plain caps are the recommended settings.

| arm | 8K ms | 8K tok/s | 8K GPU clocks per token | 32K ms | 32K tok/s | 32K GPU clocks per token |
|---|---|---|---|---|---|---|
| FAST (base) | 4,341.7 | 1,887 | 1.214 M | 17,472.8 | 1,875 | 1.360 M |
| `cap9` | 4,250.1 (-2.1 %) | 1,927 | 1.188 M | 17,122.7 (-2.0 %) | 1,914 | 1.333 M |
| `cap8` | 4,127.9 (-4.9 %) | 1,985 | 1.154 M | 16,715.4 (-4.3 %) | 1,960 | 1.302 M |

Prefill request time, interleaved runs in one session per length (2-3 runs per arm, spread at most 0.6 %); clocks per
token are the request time per token times the mean sampled GPU clock of the session (2,290 MHz at 8K, 2,551 MHz at
32K). The renormalised variants cost the same time as their plain cap within noise.

## Reading

- `cap9` is inside the composite gate at both lengths, with a thin margin at 32,768 tokens that is the FAST base's own.
- `cap8` is inside the gate at 8,192 tokens and outside it at 32,768 tokens: an explicit trade of quality for 4-5 %.
- Both stay opt-in. A profile that wants `cap9` needs its own qualification run with the rest of its set.

## Test

`strix-hip-route-prune-tests` (no model): the prune kernel against an independent host reference for every mode, the
pruned CSR plan against a host CSR of the kept assignments, and the unpruned plan unchanged.
