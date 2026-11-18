# Tiered expert residency (cards smaller than the model)

An option for discrete cards that cannot hold every routed expert: the card keeps the dense core and the hot
experts, the CPU computes the cold experts from host memory with the card's own arithmetic, and a planner moves
experts between the two as the conversation goes. It is not a new default: `auto` keeps today's fully resident
route whenever the model fits, and that route is unchanged (no new allocation, launch or sync).

Bound today: Qwen3.8-Flash-Next on the CUDA route (`strix-cuda-qwen-serve`, `qwen_decode_bench`).

## Flags (one resolver for the server and the CLI)

| Flag / environment | Values | Meaning |
|---|---|---|
| `--expert-residency` / `STRIX_EXPERT_RESIDENCY` | `auto` (default), `full`, `tiered` | `auto` = `full` when everything fits the budget, `tiered` otherwise |
| `--vram-budget-gib` / `STRIX_VRAM_BUDGET_GIB` | number, e.g. `8` or `7.5` | card bytes the engine may use (default: the bound device profile's memory budget when one is declared, else the whole card; existing physical usage is subtracted from that device cap first; an explicit residency budget limits the engine allowance further); emulates a smaller card on a bigger one |
| `STRIX_TIER_CPU_THREADS` | integer | CPU expert threads (default min(16, hardware threads)) |

`strix-cuda-qwen-serve --print-effective-config` resolves without loading the model and prints the choice and its
reason (`expert_residency`, `reason`, `vram_budget_bytes`, `vram_cap_bytes`, `vram_cap_source` = `card` | `residency` |
`device_profile`, `expert_pool_bytes`, `slots_per_layer`, `positions`,
`host_expert_bytes`). The same parser and resolver run inside `qwen_decode_bench`
(`include/strix/residency/residency_cli.hpp`, `include/strix/residency/expert_residency.hpp`); a host test
(`tests/expert_residency_parity_tests.cpp`) checks that both entry points resolve the same request identically.

Card budget (memory owner): the budget pays, in order, the dense core (attention, GDN, router, shared expert, LM
head, MTP head), scratch, the session bytes of the requested context, one zero expert per layer and the margin; the
rest is the expert pool (the same number of slots in every layer). If the requested context does not fit next to
the core, it is reduced in steps of 1,024 positions and the reason says so. The host must hold the experts that
are not on the card plus the host tables (PLE 28.8 GB) plus 4 GiB: the resolver refuses tiered mode, with the
numbers, when the available host memory is smaller (`STRIX_TIER_HOST_CHECK=0` turns the refusal into a warning; the
experts are then read from the page cache or the disk, which is the "cold" figure below).

## How it works

Per MoE layer of a step, on one stream (capturable in the step's CUDA graphs):

1. The card's prologue quantizes the rows (A8) as today.
2. `tier_publish` maps each routed expert id to its slot through the layer's id -> slot table; a miss goes to the
   layer's zero slot (zero-filled weights, finite zero output) and is written, with the row's A8 codes, to an
   exchange in mapped host memory; one sequence number is published last.
3. The card runs the unchanged routed kernels on the slot ids (hits and zero-slot misses), while a host worker
   computes the misses with `strix::cpu::expert::Executor` straight from the memory-mapped GGUF.
4. `tier_gather` waits on the host's sequence number (watchdog: fail closed, never hang), copies the CPU raw dots
   into the card's per-(row, slot) buffer and ORs the CPU status; the unchanged merge and commit follow.

Exactness: every (row, slot, output) value of the routed MoE depends only on that (row, slot) and the expert's
bytes (`include/strix/cuda/qwen_moe.hpp`), the merge reads per-(row, slot) values in slot order and never the
expert ids, and the CPU executor implements the decode chains operation for operation (integer group dots, then
only single-rounding FP32 multiplies/adds in the card's order, the gfx1151 expf twin, the same h quantizer):
`src/cpu/core/routed_expert.hpp`. Tiered output is therefore bitwise the fully resident output; MTP verification
and rollback are unchanged and only emitted tokens are counted.

Between steps the planner (`strix::residency::Planner`) keeps a decayed reuse score per (layer, expert) and
promotes a candidate over the coldest resident expert only when its score beats the victim's by a margin of reuses.
Because the CPU's miss time is proportional to the expert's bytes, saved time per resident byte is the reuse rate,
so scores compare directly; the margin pays the upload and stops thrashing. Promotions are bounded per step; the
first prefill's routes seed the scores.

Why the CPU, not uploads, executes misses (measured on the lab PC, i7-13700K, dual-channel DDR5-5200, RTX PRO
6000 on PCIe 5.0 x16, `scripts/bench/residency/`): an exact CPU expert (3.07 MB) costs 52-57 us at 16 threads
(54-59 GB/s, the DDR wall is 60 GB/s), an upload of the same bytes 84 us (36-40 GB/s pinned) and it reads the
same DDR, so splitting misses between both cannot pass the DDR wall. Rows that share an expert are nearly free on
the CPU (memory-bound), so verification rows only cost through the union of their experts. The per-layer exchange
through mapped memory costs 1.15 us round trip (a stream sync + copies costs 72.8 us).

## Strix Halo

No VRAM/RAM tier: the GPU and the CPU read the same LPDDR5X, so there is nothing to promote and a CPU overflow
arm would compete with the GPU for the same bandwidth. The resolver is never asked there; the HIP routes are
unchanged.

## Other families

The planner and resolver are host-shared. DeepSeek V4.1 on CUDA already keeps a per-layer id -> slot table with
LRU dynamic slots and remote homes (`include/strix/cuda/ds41_hot_residency.hpp`); its LRU refill is the binding
target of this planner (follow-up in a DS41 window). The CPU executor does not bind DS41 (IQ2_XXS / Q2_K experts and
the split combine of `include/strix/ds41_hot_placement.hpp`). GLM and MiniMax have no CUDA route (`src/cuda`).

## Limits of this first version

Declined in tiered mode, each with a reason at start: BF16 route storage, prefill overlap, expert types other than
Q4_K/Q5_K gate-up and Q5_1/Q8_0 down. Prefill runs the misses on the CPU in blocks of 256 rows (a streaming
prefill of whole layers is the follow-up for long prompts). Promotions run on the engine thread between steps.

## Measured

Lab PC: RTX PRO 6000 (96 GB) with `--vram-budget-gib` emulating the smaller card (the card never held more than
the budget: peak 8,217 / 10,261 / 12,257 / 16,128 / 24,346 MiB), i7-13700K, 128 GB DDR5-5200, Qwen3.8-Flash-Next
UD-Q4_K_XL (111.3 GB file, 77.0 GB routed experts), `qwen_decode_bench` greedy 64 tokens per prompt, prompts
prose (33 tokens) then edit (497) then prose then edit in one process: the first prose is **cold** (page cache
empty after a reboot: the experts come from the disk), the last two are **warm**. Context filled: at most 561
positions. MTP on at 24 and 16 GiB (the 2.8 GB MTP head leaves no useful pool below that).

| budget | slots / layer | resident experts | plain tok/s cold | plain tok/s warm (prose / edit) | MTP tok/s cold / warm (prose / edit) | byte hit rate (first prompt / whole run) |
|---|---|---|---|---|---|---|
| full (96 GB card) | 512 | 77.0 GB | - | 86.0 / 84.0 | 167.6 / 203.3 | 1 |
| 24 GiB | 100 | 15.0 GB | 3.4 | 23.6 / 22.2 | 15.9 / 64.6 / 73.4 | 0.77 / 0.62 |
| 16 GiB | 43 | 6.5 GB | 3.5 | 21.4 / 21.7 | 16.0 / 44.8 / 49.0 | 0.48 / 0.38 |
| 12 GiB | 33 | 5.0 GB | 3.4 | 18.2 / 16.4 | not run | 0.40 / 0.32 |
| 10 GiB | 18 | 2.7 GB | 3.5 | 20.9 / 21.2 | not run | 0.27 / 0.21 |
| 8 GiB | 4 | 0.6 GB | 4.9 | 21.4 / 21.3 | not run | 0.08 / 0.06 |

Identity: the greedy tokens of every tiered run hash to the fully resident ones (FNV-1a prose `8bbc29d6b1a5ef4f`,
edit `f8877b10952fc9e2`), and the fully resident route measured with this code is time-identical to the route
before it within noise (ABAB, serial prose 11.56-11.63 vs 11.55-11.60 ms; MTP prose 167.6-169.0 vs
165.9-166.5 tok/s, edit 203.3 vs 203.2-203.9 tok/s).

Reading the table: warm plain decode is about 21 tok/s from 8 to 24 GiB, so in this first version the hit rate
does not limit it. Two host costs do: promotions run on the engine thread between steps (8-9 ms per step at 12-24
GiB, 2.6 ms at 8 GiB; the effective copy rate from the mapped file is ~5 GB/s, not the 36 GB/s of a pinned upload),
which is why 12 GiB is slower than 8 GiB; and a fixed per-layer cost of the exchange plus the CPU job start
(~0.2-0.3 ms x 48 layers, to be confirmed with per-layer timestamps). At 8 GiB the Phase 1 formula (core 11.6 ms +
misses x 55 us) predicts 27.5 tok/s against 21.3 measured. Cold runs are bound by the disk (first touch of the
expert pages). MTP stays the larger lever in tiered mode: verification rows share experts, so 16 GiB with MTP
decodes 2.2x faster than plain.

Next levers (predicted, not yet measured): promotions asynchronous on a copy stream from a registered staging
ring, with the planner margin priced in measured ms per byte (24 GiB warm plain 23 -> ~28 tok/s, 12 GiB at least
the 8 GiB figure); MTP at 12 / 10 GiB with a smaller pool (expected ~40 tok/s warm from the 16 GiB ratio); a
persistent spinning CPU pool during steps (per-layer fixed cost).

The device profile caps total card usage, so physical bytes already in use are subtracted before sizing this engine. An explicit residency budget limits the engine allowance further.
