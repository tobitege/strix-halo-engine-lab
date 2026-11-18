# RTX prefill host placement

The prefill service accepts `--host-gib G --host-pin-order layer|deadline|early|interleave [--file-ring-mib N]`.
`auto` (the default) picks interleave or layer once at load from a measured table (`resolve_host_pin_order` in
`include/strix/ds41_host_pin_plan.hpp`; the choice and its reason are in the service's `describe`): interleave when
the layer order would stream >= 10 GB of experts from the file (streamed bytes minus the admitted host budget),
else layer. Admission uses the lesser of free physical memory and free commit, retaining the configured
floor; unknown headroom declines automatic growth. The effective host budget is capped to that room after staging.
`interleave` places FILE layers among the HOST ones so the disk read of each hides behind earlier host
copies (the copy stream and the disk work in parallel instead of one after the other), and pins the preload slots
first (back-to-back requests leave no idle time for a disk preload). Its file-read ring (`--file-ring-mib 0` = auto)
is 4 GiB taken out of the admitted host budget (at least 8 GiB before the ring). Each automatic page is guarded;
allocation failure releases the partial ring and restores 256 MiB staging with a recomputed layer placement and budget.
An explicit `--file-ring-mib N` is preserved and retains allocation-failure semantics. Whole-layer streaming resolves
automatic order to layer before placement. These decisions do not change inference arithmetic or resident counts.
Measured on one RTX at equal pinned memory (68 GiB layer vs 64 + 4 GiB interleave, ABAB, 4 pairs, 3 reps):
2K warm -13 %, cold -16 %; 8K warm -6 % (3 of 4 pairs faster); 32K -1 % (2 pairs). The one losing case sets the
crossover: layer-order file tier 8.4 GB, 8K +3 % warm / +6 % cold (the 4 GiB ring leaves less pinned, so ~6 GB more
comes from the disk and the 8K compute already hid the reads); at 12-24 GB interleave wins at every length.
Overrides: `--host-pin-order interleave` or `layer` (old ascending fill, 256 MiB staging). Kit:
`scripts/bench/cuda/ds41_prefill_matrix.sh` (service matrix, cold/warm, GPU clock) + `ds41_prefill_summary.py`.
The matrix requires READY and a complete successful repetition set with matching state SHA-256, logits hash and next
token. Missing or differing results fail the command; exit cleanup stops its service and sampler. For custom lengths,
standalone summary calls must supply matching `--lengths N ... --reps R` (defaults: 2048 8192 32768, three reps).
The crossover remains calibrated to the platform and workload above; these host checks do not broaden qualification
or independently validate the timing claims. Host order is outside the service profile levers; use
`--host-pin-order layer --file-ring-mib 256` for legacy placement/staging, including when comparing `--baseline`.
`early` is the legacy ascending fill without the layers the idle preload already holds: the pinned
layers lead the schedule and later FILE reads overlap their compute.
`layer` preserves the ascending host-budget allocation (the default before interleave).
`deadline` selects the longest affordable suffix of the streamed layer schedule,
starting at the last layer and stopping when the next streamed part will not fit.
It excludes the first device-slot-count streamed layers, which the idle preload
already holds on the device. Fully resident layers consume neither a host budget
nor a streaming slot. A partially resident layer costs only its streamed bytes.
Unused budget is intentional: an earlier host burst would precede a later file
deadline. Explicit whole-layer tiers retain their supplied placement.

The policy targets an ascending single-pass request after idle preload. It keeps
file reads together before the host suffix; the loader has four bounded staging
buffers, rather than an independent queue of future file layers. It does not
reschedule requests, increase read-ahead, or change slot counts. Bounded passes
and repeated layer schedules reuse this placement but have different deadlines;
the policy is not an optimizer for those schedules. Cold or back-to-back requests
can consume an unfinished preload.

The per-expert planner first constructs the same resident expert IDs and virtual
order as `layer` with the same budgets and ranking, then changes only the streamed
source. Tensor bytes, group boundaries, device residency, kernels and arithmetic
remain unchanged. Pinned-allocation failures retain the existing file fallback.
The selected host layers are filled in ascending order at startup; under memory
pressure, inspect the effective placement rather than assuming all planned pins
succeeded.

`GET /v1/describe` includes `describe.host_pin_order` and
`describe.host_layers` (absolute layer IDs after pin-allocation fallback).
Each prefill reply includes the same fields under `prefill`. The existing `tiers`,
`resident_experts`, pinned-memory and pin-failure fields provide the remaining
placement and allocation details.

The CPU-only `strix-ds41-host-pin-plan-tests` target checks selection, preload
exclusion, partial-layer costs, slot clamping and budget boundaries. CPU checks
and compilation do not establish device exactness or workload performance.
Qualification requires matching complete token IDs and logits/state digests
between `layer` and `deadline` with the same model, capacity, residency budget,
ranking and numerical identity, followed by interleaved workload measurements.

## Service defaults and memory admission

For service loads with `--pass-rows 0` (the normal mode) and no explicit
`--tiers`, `--host-gib-auto 32` (opt-in) resolves to a 32 GiB cap,
`--host-commit-floor-gib 20` and `--host-pin-wait-s 90`. Explicit values win.
`--host-gib 0` disables optional expert pins and streams those bytes from FILE.
Bounded loads and explicit tiers do not inherit these service defaults; other
binaries and decode-only library objects do not inherit them either.

`--host-gib-auto G` selects the **implicit, opportunistic** policy (omitting both host flags keeps the file-only tier). The
32 GiB budget is a cap, attempted in whole streamed-layer parts, rather than
a guaranteed allocation. A partially resident layer costs only its remaining
experts; a whole-layer provider allocates complete layers. There are no partial
pins within a selected layer part. Unused budget is possible.

Before optional pins, the service allocates mandatory device workspaces,
attention/batch contexts, expert residents and device streaming slots, pinned
FILE staging, every configured context's maximum-capacity Engram rows, and the
maximum-capacity capture buffer when `--fast-export` is enabled. This also
moves normal-mode Engram allocation from the first request to load time.
Request token uploads, pageable hashes, export/serialization copies, queued
pipeline states and runtime/driver allocations still consume additional room.

For each optional allocation of `n` bytes, admission requires both:

- free physical memory >= `n + floor + 4 GiB` (24 GiB reserve at the default);
- free commit >= `2n + floor` (20 GiB floor at the default).

The implicit policy uses current telemetry and one nonthrowing pin attempt.
Physical shortage, commit shortage, allocation failure or unavailable telemetry
immediately demotes that layer to FILE, reading the same tensor bytes. It never
issues a commit-growth probe, retries or sleeps, even if an explicit floor of
zero or an explicit wait was supplied. Its effective wait is always zero; the
resolved wait option remains 90 seconds unless overridden.

An explicit `--host-gib` retains the explicit policy: with a positive floor it
may issue a Windows commit-growth probe and retry against one shared deadline,
including charge-settlement waits. Once that deadline expires, later layers
are demoted without starting another allocation/probe. Explicit wait zero
permits one attempt per layer. Explicit floor zero keeps the legacy allocation
path, including its provider allocation exception behavior. Unknown telemetry
declines floor-governed optional allocations. Floor and wait must be finite
and nonnegative. The wait bounds retries, not total startup time: model reads
and OS/CUDA calls can take additional time.

Successful pins remain held until service destruction; startup telemetry is
not a reservation for future services and pins are not reclaimed dynamically.
A near-full 32 GiB expert tier plus staging holds roughly **35 GB pinned for
the service lifetime**. Mandatory Engram and fast-export buffers can increase
that total depending on capacity and batch contexts. Set `--host-gib` lower,
or use `--host-gib 0`, when Qwen/MiniMax PRO services share the PC. Qualify
shared-service launch orders and peak concurrent workloads before deployment.
Loading/filling and releasing pins can add startup/shutdown latency.

`GET /health` and `GET /v1/describe` expose a `host_tier` snapshot: policy,
budget, requested/effective expert bytes, effective absolute host-layer IDs,
demotions with `physical`, `commit`, `alloc` or `telemetry` reasons, floor,
resolved/effective wait, current free physical/commit bytes (`"unknown"` when
unavailable), and pressure (`ok`, `low`, `unknown`). Pressure reports whether
the configured reserves remain available; it does not reserve future memory.
`total_service_pinned_bytes` includes expert staging, Engram and the
service-owned fast-export capture buffer. Mutable pinned accounting is
published atomically at completed GPU stages (also on failure), so health can
read the last synchronized snapshot while a request runs. Placement is
immutable after load. Driver `describe` also exposes byte accounting and
demotion reasons; the service's `service.pinned_gb` includes capture memory.

`strix-ds41-host-pin-admission-tests` is a CPU-only decision and fault-injection
gate for admission boundaries, unavailable telemetry, allocation failure,
implicit versus explicit retries, zero-floor fallback and shared deadlines.
It does not qualify native execution, memory coexistence or numerical fidelity.

Status (2026-10-27): opt-in. Measured exact, 1K -3.8..-3.9 s (-24..-30 %), 8K -28 %, 30K -12 %; the service
default stays file-only until batch formation, slots/subsets and coexistence with the Qwen/MiniMax PRO services are
qualified. Deployments set the budget (e.g. the hyperloom preset) with `--host-gib-auto 32`.

The FILE tier can be served by two drives at once: [rtx-two-drives.md](rtx-two-drives.md).
