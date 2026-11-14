# Aux lane: where side models run

A serving process runs small side models next to the language model: the prompt scorer of lossy prefill compression,
an image encoder, an embedding model. They are compute-bound and small. The aux lane
(`include/strix/serve/aux_lane.hpp`, `src/serve/aux_lane.cpp`) is the one place that decides which unit runs each
such job, bounds the waiting work, falls back when a unit fails, and counts what happened.

Status: the lane, a GPU executor and a CPU reference executor exist, with the prompt scorer and an image encoder as
clients. There is no NPU executor. With only a GPU executor registered the lane call is the direct call it replaces
(same thread, same arguments, same result), so adopting it changes no output.

## Units

A job may run only on a unit the device declaration grants the `aux` role (see
[Other units: the NPU and the CPU](../../discrete-gpu.md#other-units-the-npu-and-the-cpu)):

- the GPU the process is bound to (always; side models already run there);
- a declared CPU unit with `aux` (the reference executor: exact arithmetic of the host reference, not a fast path);
- a declared NPU with `aux` and its three qualification keys.

`aux::units_from(bound)` (`include/strix/serve/aux_units.hpp`) builds that list from the start gate's result.
Registering an executor for any other unit is an error.

## Jobs, results, executors

| Type | Fields |
|---|---|
| `AuxJob` | `kind` (`score`, `vision`, `embed`), borrowed input spans, `shape`, `session`, `identity`, `deadline` |
| `AuxResult` | `tensor`, `shape`, `integers`, and from the lane: `identity`, `unit`, `unit_id`, `fallbacks`, `queued_ms`, `run_ms` |
| `Executor` | `unit`, `unit_id`, `kind`, `identity`, the function, `threaded` |

`identity` names the arithmetic of a result. Units do not produce the same numbers, so each executor states its own
identity and every result carries the identity of the executor that produced it. A job with an empty identity accepts
any executor of its kind; a job that names one is never answered by another (it fails instead).

An executor is `threaded` (the lane owns one worker thread per unit and runs the job there) or not (the unit belongs to
another thread, for example a GPU context: its jobs run inside `drain(unit_id)` or `run_now`, on the calling thread).

## Choosing the unit

For each job the lane builds an order of executors and tries them in turn:

1. Start from the configured order of the kind (default GPU, NPU, CPU).
2. When both the GPU and the NPU duration of that kind are measured (`Config::rates`), the NPU goes before the GPU
   only if the GPU is busy with main work (`Config::gpu_busy`) and
   `npu_seconds / gpu_seconds < gpu_side_cost / npu_side_cost` (defaults 0.85 / 0.10 = 8.5). On an idle GPU, or
   without measured durations, the GPU goes first.
3. `STRIX_AUX_UNIT=gpu|npu|cpu` (or `Config::force`) restricts the job to one kind of unit, with no fallback to
   another; `auto` or unset keeps the rule above. This is the switch for A/B runs.

An executor that throws makes the lane try the next one; the result reports how many failed first.

## Queue and calls

- `submit(job)` queues and returns a future. Beyond `Config::capacity` waiting jobs (default 64) it throws
  `AuxError` with `queue_full`; the caller keeps its own direct path for that case.
- `drain(unit_id, max_jobs)` runs waiting jobs of a non-threaded unit on the calling thread.
- `run_now(job)` runs one job to completion from the calling thread.
- A job whose `deadline` passed before it started fails with `deadline` and does not run.
- Destroying the lane fails the waiting jobs with `stopped`.

`receipt_json()` returns the counters: submitted, completed, queue_full, no_executor, deadline, failed, fallbacks,
waiting_peak, and per executor the jobs, failures and busy time.

## Clients

- Prompt scorer: `score_executor(unit, id, identity, scorer, threaded)` wraps any `pflash::PromptScorer`;
  `LaneScorer` is a `PromptScorer` whose calls go through the lane. The four score vectors round-trip exactly.
- Vision: `vision_executor(unit, id, identity, encoder, threaded)` wraps a function from an image to `[count][hidden]`
  rows; `encode_image(lane, image, identity)` returns the rows with the identity and unit that produced them.

## Tests

`strix-aux-lane-tests` (host only): unit choice for every branch of the rule, the declared-unit gate, the scorer
through the lane against the direct call bit for bit, fallback, the bounded queue, deadlines, stop, and the vision
client.
