# Qwen native-control owner host replay

The target links the actual `qwen::HttpSessionOwner` and shared
`SessionRuntime`/`GenerationQueue`/`KvStore`/`ApiService`. CMake extracts the
current production native binding between unique markers; the test compiles
that binding against a bounded native request/callback ABI double. It does not
copy the adopted planner algorithm. Real host sampling code validates policy,
seed and request identity. Neither HIP nor CUDA is built or executed.

Configure this directory for standalone tests, or the repository root with
HIP/CUDA disabled for the centrally registered target. For example:

```sh
cmake -S tests/qwen_session_owner -B build/qwen-owner-r13/standalone
cmake --build build/qwen-owner-r13/standalone --config Debug --target qwen_session_owner_tests session_runtime_tests session_http_replay_tests
ctest --test-dir build/qwen-owner-r13/standalone -C Debug -R '^(qwen-session-owner.*|session_runtime|session_http_replay)$' --no-tests=error --output-on-failure
```

Repeat with Release. Tests use always-active `CHECK` and an independently
checked failing sentinel. The counter selection sets request-identity policy
only in its own process. Ordinary tests require no external reference files.

Coverage includes top-level and nested arrivals, growth/refusal calls, named
FIFO barriers across ordinary/nested work, mutex release/reacquisition, slot
reuse and stale generations, divergent logical/native slots, reverse callback
completion, EOS/limits/cancellation/disconnect, throwing publishers, isolated
sampling refusal, seeds and counter identities, probe proof completion/refusal,
admit/step/finish failures, every queue/reservation/live cleanup and an actual
idle condition-variable control/request wake. Fixed output streams, receipt
values and native traces are independently prescribed in the fixture.

## Independent entry comparison

Before production editing, exact task-entry sources and hashes were archived
privately under the generated evidence tree. `extract_reference.cmake` accepts
only the pinned complete entry file hash and extracts the original owner block
without changing its control-flow bytes. Missing/duplicate/changed anchors
fail closed. The archived harness compiles this block with explicit native/type
seams and the real queue and sampling code. Its source, binary and 17-case
capture remain immutable. A separate extension compiles the same frozen owner
block and frozen pre-change protocol libraries to capture actual HTTP responses;
it never runs the new owner to generate expected data.

Optional CMake inputs `STRIX_QWEN_OWNER_REFERENCE` and
`STRIX_QWEN_HTTP_REFERENCE` bind the exact immutable capture hashes. They add
separate `qwen-session-owner-reference` and
`qwen-session-owner-http-reference` tests. Evidence runs must select these
explicitly along with ordinary tests. CMake never refreshes references or
expectations. The generated report records their provenance and commands.

Request replay compares complete output IDs, terminal receipts and native event
traces. Queue wall time is a nondeterministic observation and is excluded from
the request replay serialization; its nonnegative value is checked. HTTP bytes
retain all fields and framing after replacement of exactly three volatile
envelope values: completion ID, creation time and `strix.queue_ms`. Native
prefill/decode timings and every other receipt value remain in comparisons.

The host ABI double is not the native scheduler. This evidence establishes no
device arithmetic, physical snapshot/COW, configured-drafter, capacity or speed
qualification. See [the adopter boundary](../../docs/engine/session-qwen-owner-adoption.md)
for unrun native tickets and follow-on slices.

The HTTP witness uses explicit idle and teardown control phases. The disconnect
case holds the owner at idle until the initial SSE receipt has disconnected and
the real queue reports the submitted request cancelled before admission. The
queue then retires that request without a native step. After ApiService returns,
the harness explicitly wakes the idle owner to shut down. This avoids depending
on the host double's step-only auto-stop. The three-second readiness assertion,
full response assertions, reference bytes and normalization remain unchanged.
`qwen-session-owner-readiness` selects these phases directly. Existing failure,
active cancellation, owner wake and shutdown checks remain in the main target.
