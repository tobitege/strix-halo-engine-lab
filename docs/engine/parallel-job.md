# Parallel job contract (PR1)

PR1 supplies host infrastructure only. `strix/session/parallel_job.hpp` is a
C++20 value API linked through `strix_session`; it needs no model family, HTTP
server or accelerator. It adds no execution mode, queue, sampler or admission
policy. Native fork/COW, external append and full fanout remain unsupported in
both migrated families. Host fixtures establish contract behavior, not model
portability, GPU exactness or performance.

## Frozen value API

All names below are in `strix::session`. Strings and vectors own their values.

| Type | Exact fields / API |
|---|---|
| `TokenBuffer` | Default empty; explicit constructor from `std::span<const std::uint32_t>`; `tokens() const noexcept` returns that immutable span. Private shared immutable storage. |
| `TokenReference` | `TokenBuffer buffer`; `std::uint64_t offset = 0, length = 0`; `tokens() const` returns the checked interval. |
| `Contract` | `std::string task`; `std::uint32_t min_tokens = 0, max_tokens = 1`; `std::vector<std::string> evidence_ids`. |
| `PointSpec` | `std::string id`; `Contract contract`; `TokenReference suffix`. |
| `DependencyEdge` | `std::string before, after`. |
| `Digest` | `std::string id, producer`; `TokenReference evidence`. |
| `MergeSpec` | `std::vector<std::string> point_order`; `std::string instruction`. |
| `JobPolicy` | `std::uint64_t token_budget = 0`; `std::uint32_t max_lanes = 1`. |
| `ParallelJob` | `std::string request_id`; `TokenReference prefix`; `std::vector<PointSpec> points`; `std::vector<DependencyEdge> dependencies`; `std::vector<Digest> digests`; `MergeSpec merge`; `JobPolicy policy`. |
| `JobReceipt` | `std::string request_id`; `std::uint64_t budget_tokens = 0, output_tokens = 0`; `std::vector<std::string> completed_points`. |

`void validate_job(const ParallelJob&)` performs pure structural validation
before state mutation. It rejects empty request/point/digest IDs, duplicate
point/digest IDs, missing endpoints, self/duplicate edges, cycles, missing or
duplicate merge IDs, merge order contrary to dependencies, unknown/duplicate
evidence IDs and evidence whose producer is not a strict DAG predecessor of its
consumer. Point and digest IDs have separate namespaces. Point vector storage
order has no causal meaning; merge order is a topological permutation of all
points. Evidence may travel through transitive dependencies.

The one request budget counts all generated IDs across points and merge,
including stops. Per-point lengths are bounds within that budget: maximum is
positive and at most the job budget, minimum is at most maximum, and the sum of
minima must fit. Maxima need not sum to the budget; the future runner must enforce
the shared remaining allowance. `max_lanes` is positive requested concurrency,
not qualified physical capacity. PR1 does not reserve merge tokens or interpret
task/instruction text. Protocol grammar, digest selection, rendering and audit
belong to PR3; cost/configuration and admission belong to PR2.

References use half-open token intervals, checked by subtraction without
overflow. Empty references are valid. `TokenBuffer` copies construction input
even when the caller retains a mutable vector, pointer or moved-input alias;
copies of buffers share only the new immutable allocation. A reference keeps
that allocation alive after construction storage or the original job dies. A
returned span borrows from its buffer and must not outlive it. Validation cannot
freeze public fields: consumers retain an owning job value, validate that exact
value before admission, and revalidate after edits. No mutable construction
alias can change the tokens in a validated value.

Structural errors and reference bounds throw `std::invalid_argument` without
changing the job or engine. Allocation failures propagate. There is no HTTP
error mapping or protocol implementation in this API. Receipt counters are
owning request-wide reporting values, not a source of capacity or capability.

```cpp
#include "strix/session/parallel_job.hpp"
using namespace strix::session;

int main() {
    ParallelJob job;
    job.request_id = "request";
    job.points = {{"point", {"answer", 0, 16, {}}, {}}};
    job.merge.point_order = {"point"};
    job.policy = {32, 1};
    validate_job(job);
}
```

## Row hints and effective work

`engine.hpp` owns the only `RowHints` definition. Future cost code includes it;
it must not redefine it. `DecodeItem` and `AdvanceItem` end with
`std::optional<RowHints> row_hints = std::nullopt`, preserving old aggregate
callers. An absent optional or an empty `RowHints` preserves the old planner.

| Field | Units and validity |
|---|---|
| `RowHints::physical_row_cap` | `std::optional<std::uint32_t>`: per-lane evaluated forward-row cap for this call. Unset uses native planning; zero is invalid. |
| `RowHints::expected_useful_rows` | `std::optional<double>`: predicted useful forward work for this call, finite in `[0, UINT32_MAX]`, possibly fractional. Zero is an explicit prediction; unset is unknown. |
| `EffectiveRows::physical_rows` | `std::optional<std::uint64_t>`: actual lane-attributed forward rows, including padding. |
| `EffectiveRows::useful_rows` | Same type: evaluated prompt, target and draft rows, including subsequently rejected work. |
| `EffectiveRows::padding_rows` | Same type: evaluated rows carrying no useful lane work. |

Hints are neither a sampler nor an output-token allowance. Expected useful work
does not report accepted outputs. Useful measured rows also differ from accepted
outputs: an output can use retained logits without a new forward row, while
rejected draft/target work can consume rows without publishing IDs.

`Capabilities::row_hints = false` is the default. `validate_row_hints(hints,
supported)` checks numbers and refuses any nondefault unsupported hint. Every
implementing adapter must prevalidate **all** items before mutation.
`resolve_row_hints(hints, supported, qualified_shapes)` also validates strictly
increasing positive native shapes and clamps a physical cap down to the largest
shape at most that cap. No such shape (including an unavailable one-row path)
is an error; never exceed a requested cap. An expected-work-only hint still
requires qualified shapes but does not choose a physical cap. The helper does
not execute a planner. Both current native adapters remain unsupported; Gemma4
refuses nondefault hints at entry to `advance`, before lane or planner work.
MiniMax retains its existing refusal of scalar/batch owner-control APIs.

`StepResult::rows` reports one decode call. `AdvanceResult::rows` is authoritative
for the entire advance, covering input and output; its nested `output.rows`
stays unset. Never add both. Unavailable fields remain unset, including on old
adapters; unknown is never measured zero. Known fields obey
`physical_rows = useful_rows + padding_rows`. For grouped calls, lane work is
attributed to each lane and group-only padding is charged once to the first
result in input order. Sum a field only when every result knows it. A lane cap
applies before group-only padding attribution; the native group may require an
additional tail. No group padding is multiplied by the lane count.

## State and runtime boundary

PR1 reuses `Frontier`, sealed `StateSnapshot`, `KvStore`, the six
`RecurrentStateOps` operations and the `LaneState` base/log/dense contract.
Resume-forks preserve consumed/pending history and sampler/selection cursors.
Prompt-only reuse has a distinct sealed purpose and explicitly initializes a
new sampler/drafter policy. A fully consumed PLAN boundary evaluates the last
input without selecting a new token. An already published pending output must
remain in a resume checkpoint; truncation is not a way to manufacture that PLAN
boundary. Model/numerical domain, schema, rank order/topology, provenance,
generation and capacity checks precede allocation and erased-payload hooks.
Failed rank restoration includes the throwing rank in rollback; incomplete
restoration poisons the destination, and failed release quarantines capacity.

The queue's additive `take(..., bool* blocked)` output observes a failed
selection under the same lock as selection. False means empty/exhausted, true
means an observed inadmissible candidate. The runtime passes `!blocked` to the
existing idle wait. It no longer reads queue occupancy again after `take`:
a submission in that interval is new work, not a previously blocked FIFO head.
This keeps long idle deadlines, control callbacks outside the lock, shutdown
wakes, work counts and scheduling intact. No polling timer is introduced.

## Build and extension points

`cmake/StrixSession.cmake` registers `parallel_job.cpp` in `strix_session` and,
under `BUILD_TESTING`, `session_parallel_job_tests` / CTest
`session_parallel_job`, labeled `cpu;session`, timeout 30 seconds. Root builds,
minimal `tests/session_contract`, storage and `tests/session_runtime` builds
use this same registration. Standalone storage keeps its historical archive
target and CTest aliases. No accelerator language is needed:

```sh
cmake -S tests/session_contract -B build/parallel-contract-p7/minimal -DBUILD_TESTING=ON
cmake --build build/parallel-contract-p7/minimal --config Debug
ctest --test-dir build/parallel-contract-p7/minimal -C Debug --output-on-failure
```

Future PR2 cost/config, PR3 protocol and PR5 runner implementations add actual
sources to `strix_session` and real tests to the central contract list. Absent
future sources are pending, with no placeholder passing tests. The runner must
use existing batched/mixed Engine operations and capability gates; no implicit
scalar fallback or automatic fanout is supplied here. Native qualification
still requires complete physical fork/COW state isolation, all-rank rollback,
grouped suffix/decode and external append, full token/logit/state/cursor and
receipt checks, unchanged capacity/defaults, and complete-workload measurements.
