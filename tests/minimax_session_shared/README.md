# MiniMax shared owner replay

Root, nested and independent frontend projects use the single test-only queue
transform in `cmake/StrixReplayObservation.cmake`. It checks the submission and
publication counts before generating a copy and fails configuration on drift.
Both archived publication lambdas call one observer helper; the current parser
still runs before observation and publication and returns `publish && keep`.
`minimax-replay-observation` checks the actual current/archive source bytes,
one submission/token observer hook each and ten rejected mutations. The current
replay also checks both real queue entry paths, cancellation and parser stop,
with always-active assertions. Frozen responses/trace and the independently
recorded original Hello-padding failures remain separate evidence.

`Lead::runtime()` returns the production `SessionRuntime` owned by Lead.
`Lead::loop()` is a compatibility call to `runtime().run()` with native abort
translation. There is one request owner. The constructor and public
Shared/Policy/Counters/Extra/Drafts contracts are retained.

```text
native service -> Lead::runtime().run()
  SessionRuntime: reservation / TakenJob / ticket / stats / publication / finish
    -> KvStore: physical slot lease / generation / canonical history / release
    -> bounded Lead planner: deadlines / nesting / layer policy / pair control
       -> SessionEngine -> Backend::execute(SessionPlan) -> native Runner
          whole prefill / layer unit / grouped step / draft
  named request -> existing paired session operation -> runtime publication
    -> authoritative local frontier observation -> runtime completion
    -> paired park before any subsequent admission
  named RAM checkpoints -> shared NamedStateCache retention and LRU
    -> family codec / rank agreement / durable files
```

Planner state borrows runtime-owned jobs and stats; it owns neither tickets nor
leases. The runtime retains a selected candidate before deadline deferral,
including its submitted/started/reordered metadata. KvStore issues the physical
slot and generation. Layer work advances a checked, finite native checkpoint;
canonical input remains unchanged until the complete prompt has been consumed.
The native first selected token is published without another forward.
The adapter's logical context bound counts the pending ID after the final
physical row; native public admission and physical row/slot capacity stay
unchanged. A separately executing archived Lead confirmed the near-context
lookup boundary (65 selected IDs from a 64-token prompt in a 128-row arena),
and the current owner preserves that pending selection and context finish. Publication
accounts each committed ID before the callback, preserving native timing and
stop/allowance precedence. Callback errors remain lane-local. Escaping native
errors retain partial output and fail outstanding tickets before cleanup.

`Capabilities::owner_control` requires a bounded control planner; scalar
prefill/decode cannot silently replace this native grouped route. Backend plans
contain no scheduler policy and make no kernel changes. The general engine's
existing admission/idle/session hooks remain available. Ordinary construction keeps generic snapshots OFF. The explicit bounded
[paired-state profile](../../docs/engine/minimax-paired-state.md) supplies eager
capture/restore and actual per-rank views through production family transactions.
Fork/COW/resume/external-append capabilities remain unavailable.
Named sessions retain their existing physical codec, paired compatibility,
RAM/disk budgets and admission multiplier. No automatic prompt cache is enabled.
The shared owner does not implement native copy-on-write, fork, resume or
external append, and its ordinary service does not activate generic paired state. Configured
drafters, partial/parked prefills, named activity and unsafe truncate are excluded
from the explicit state profile.

External stop is observed at native/control boundaries and during timed idle,
including zero-request heartbeat/quiesce/shutdown. The native protocol closes
before the shell shuts down unreserved HTTP queue work, preserving the original
queue accounting boundary. Explicit runtime shutdown still shuts down the queue.
FakeClock supplies owner observation times and waits.

## Reference and wire boundary

The independent reference is revision
`52157f16bdbe600fdb7534cb9c4b071e3330f098`, plus the raw uncommitted sources
recorded by [baseline.json](fixtures/baseline.json). The archive is read only.
Both configurations validate all 89 source hashes and three frozen fixture
hashes. The retained initial capture executable is checked separately.
`minimax_original_lead_tests` compiles the archived original headers, family
sources and queue in its own library, even in a current-source configuration.
It is not a second current-source translation unit.

Current replay executes the migrated production owner in all 55 original
control scenarios and five actual HTTP/SSE replies. Assertions, selected IDs,
receipts, queue work counts and normalizations are preserved. The sole HTTP
normalization remains integer `created`. Mismatch diagnostics go only to the
configured build directory; no fixture is regenerated.

The original source still fails both OFF hello memcmp assertions on the tested
host compiler. Observed differing byte offsets were 12ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ15, 105ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ111, 177ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ183,
204ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ207 and 428ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ431. These are Control/PfProgress padding, not defined fields.
The current serializer writes defined little-endian scalars at the existing
version-3 offsets and zeros padding and the 512-byte packet's reserved tail.
Tests poison every observed padding range and populate active/parked tasks,
ordered rows and draft fields. The original assertions are retained verbatim.
Canonical serialization is not a proven bitwise comparison with historical
nondeterministic native padding. Original failures remain CTest failures;
none is skipped, masked, rebaselined or marked WILL_FAIL.

## Host coverage

The MiniMax source census also includes the shared session contracts, queue,
storage/backend/prefix owners, `SessionRuntime` implementation and the session
CMake module because they directly own the migrated request lifecycle. HTTP,
tokenization, model registry and transport remain under their separate gates;
Gemma4 implementation stays under its family gate. The census is host audit
coverage only. Stale fingerprints remain unresolved, so the source gate fails
closed until those rows receive reviewable provenance.

The observer tooling compiles and executes both the archived and extracted
transport wrapper/control path with a fake transport. It proves an additional
non-Control startup vote changes startup bytes while ordinary controls exclude
it, checks row logits/state sizes and emitted IDs, and verifies required-session
refusal before follower work plus one-request completion. Missing historical
objects are reported unavailable rather than treated as equivalence. This is
instrumentation coverage, not native or numerical qualification.

The final source census includes MiniMax family and linked core/transport
inputs, shared session contracts, `GenerationQueue` API/implementation,
storage/backend/prefix owners, `SessionRuntime`, root CMake registration and
`cmake/StrixSession.cmake`.
Recursive MiniMax/shared-owner additions fail closed. Shared HTTP, tokenizer/
registry and transport implementations remain under separate gates, as do
Gemma4 family files and nested tests. The combined review refreshed ten
attributable migration/integration receipt rows. The original content for seven
device/default/shared-core fingerprints was found in tracked Git history, but
those files changed afterward. Their current content has not been reconciled
against the associated selector gates, so those rows remain unresolved and the
source audit intentionally fails closed. The inventory and provenance detail
stay in the generated audit directory, outside this public replay guide.

| Area | Evidence |
|---|---|
| Whole/FIFO, short-first, deadline deferral, nested park/resume | All 55 frozen production scenarios; exact queue work counters |
| DSpark automatic/legacy, lookup joint rows, taps/retire order | Existing production policy/receipt assertions and frozen row traces |
| Stop at final allowance/inside speculative window; final-row context | Additional actual native-planner tests with an independent deterministic selector |
| Cancel before/open/parked; reuse | Existing scenarios plus a cancellation signalled while the parent is parked, applied at the next native boundary |
| Adapter/store ownership, exhaustion, stale generation, failure | `minimax-session-owner`, including poisoned progress and partial-publication failures |
| Two-rank state hooks | Independently executing fake prefill/step ranks and per-rank release; no echo-only claim |
| Named request completion-before-park, refusal | Existing scenarios plus failed restore followed by ordinary admission |
| Named cache, paired restore/codec failure, admission and LRU | Unchanged MiniMax CPU cache suite using the shared retention owner |
| Idle admin, external work, drain, heartbeat, quiesce and stop | Additional zero-generation owner tests and unchanged shared runtime tests |
| Generic shared owner/storage and existing family regression | Shared session tests and unchanged Gemma4 replay/integration suite |

All test checks are active in Release. This project's targets also undefine
NDEBUG. The frozen replay peer still echoes checkpoints; it does not prove the
complete follower protocol. Independent fake rank execution covers the adapter
operations, not physical attention/draft/selection state. Complete native
both-rank numerical/state comparisons, device builds, named-session operation
failures on real devices and throughput/capacity qualification remain unverified.
Native fanout/COW is a separate remaining completion blocker. The root cache
failure was host path length: in the root multi-config test directory the
durable temporary filename reached 262 characters, beyond the 259-character
Win32 path limit. The atomic write failed, and the test's fixed fixture then
remained because cleanup only ran on success; later invocations refused that
stale path. The fixture now uses an atomically created unique directory under
the system temp root, bounds the actual durable temporary filename to 240
characters, and cleans up through RAII on failure. It never removes a
pre-existing path. Concurrent Debug and Release cache binaries pass from the
same working directory. Root service/build integration and source-receipt
audit debt need separate review; no selector fingerprints are rehashed here.

## Commands and results

From the repository root, with the verified original archive available:

```sh
cmake -S tests/minimax_session_shared -B build/minimax-owner-p6/current
cmake -S tests/minimax_session_shared -B build/minimax-owner-p6/original -DMINIMAX_SESSION_BASELINE=ON
cmake -S tests/gemma4_session -B build/session-owner-p6/gemma4
```

For each of Debug and Release:

```sh
cmake --build build/minimax-owner-p6/current --config CONFIG --parallel 4
ctest --test-dir build/minimax-owner-p6/current -C CONFIG --output-on-failure
cmake --build build/minimax-owner-p6/original --config CONFIG --parallel 4
ctest --test-dir build/minimax-owner-p6/original -C CONFIG --output-on-failure
cmake --build build/session-owner-p6/gemma4 --config CONFIG --parallel 4
ctest --test-dir build/session-owner-p6/gemma4 -C CONFIG --output-on-failure
```

Final CTest counts (pass / fail / skip):

| Configuration | Current project | Archived reference project | Shared session + unchanged Gemma4 |
|---|---|---|---|
| Debug | 4 / 1 / 0 | 0 / 2 / 0 | 15 / 0 / 0 |
| Release | 4 / 1 / 0 | 0 / 2 / 0 | 15 / 0 / 0 |

The current project's sole failure is its archived-original policy target.
Both failures in the reference project are the archived policy assertions,
including the replay executable's policy run. Each reports the same two
historical hello memcmp failures; its frozen 55/5 replay still matches.
All four current-source tests pass in both configurations. An earlier combined
Debug shared run reported one session_runtime failure (14/15); the isolated
rerun passed 1/1 and the final complete rerun passed 15/15. No assertion was
changed to obtain those results. The earlier log is retained.

Final verification checks all 89 archived raw source hashes, all three MiniMax
and three Gemma4 frozen fixture hashes, and the retained initial capture binary
hash. The source checker passes, as does git diff --check. Every one of the
18 edited/new files is also read explicitly as bytes and checked for valid
UTF-8 and absence of raw NUL, including untracked additions. Review covers the
actual tracked diff and untracked source additions. No GPU execution or
accelerator build is part of these host checks.

## Change inventory

Production files changed or added in this extraction:

- `include/strix/session/engine.hpp`
- `include/strix/session/prefix_cache.hpp`
- `include/strix/serve/session_runtime.hpp`
- `include/strix/minimax/session_cache.hpp`
- `src/serve/session_runtime.cpp`
- `src/minimax/serve/minimax_lead.hpp`
- `src/minimax/serve/minimax_control.hpp`
- `src/minimax/serve/minimax_session.hpp`
- `src/minimax/serve/session_engine.hpp` (new)
- `src/minimax/serve/session_engine.cpp` (new)
- `src/minimax/session/runtime.hpp`
- `src/minimax/session/session_cache.cpp`

Tests and the replay guide changed or added:

- `tests/minimax_lead_tests.cpp`
- `tests/session_prefix_cache_tests.cpp`
- `tests/minimax_session_shared/CMakeLists.txt`
- `tests/minimax_session_shared/replay_tests.cpp`
- `tests/minimax_session_shared/session_tests.cpp` (new)
- `tests/minimax_session_shared/README.md`

## Paired state host transaction

`minimax-paired-state` is registered here for standalone and central root family
builds, without accelerator languages. Always-active checks run independently
executed rank memories through production Control framing/replay, state
serialization/transaction, SessionBackend, SessionEngine and KvStore. Fault
adapters cover retained/resource budgets, identity and provenance, partial
prefill refusal, pending/canonical/planner restore, release/reuse, post-write
imports on each rank, rollback failure/quarantine, corrupt frames and lost
install acknowledgements. This is host protocol/storage evidence only.

`native::paired_state_probe` is the bounded both-rank native call site for the
same hooks and actual Runner/follower. It is implemented, not compiled or run
here; it grants no GPU fidelity, page COW, generic continuation or performance
qualification. Frozen expectations and archived original Hello assertions are
unchanged.

## Opt-in consumed-input continuation

`minimax-continuation` is registered centrally and in this standalone project.
It reuses the production paired-state/control harness with independent mutable
rank memories. Always-active Debug/Release checks use an independently computed
causal-row recurrence; token metadata alone cannot satisfy input tests. Cases
include complete and unequal input groups, retained-only zero-row first decode,
pending plus suffix, one grouped mixed forward, stops/budgets/context, invalid
policies/hints/leases, boundary cancellation/resume, terminal sealing, release/
reuse, full snapshots, native writes failing on either rank, malformed results,
install rollback and lost acknowledgements. Frozen replay and archived Hello
assertions are unchanged.

The native `minimax-native-paired-state` executable accepts an optional final
`state|input|fault0|fault1` ticket. The new tickets require <=8 fixture IDs,
MAX_ROWS>=8 and context headroom; each has a 120-second hard process watchdog.
They run the real Runner/follower/KvStore with a cold independent same-shape
native teacher-forced/greedy reference and compare complete raw logits/row bytes
on each rank, native candidates, frontiers, retained selection, receipts and
actual grouped calls/rows. The fault tickets throw after the selected rank's
real native row write and verify complete paired rollback before retry. Native
compiler/link/device checks are NOT RUN; the host cannot find real HIP headers.
The broader legacy-prefill identity, native COW, configured-drafter completeness,
paired completed-turn caching, capacity and performance gates remain open.


## Resident prefix owner test

`minimax-prefix-reuse` / `minimax_prefix_reuse_tests` is registered in the
standalone project and the central session module. Its always-active checks run
production SessionRuntime, Lead, SessionEngine/SessionBackend and control replay
against independent mutable rank images; write counters measure suffix rows.
It covers voting/alignment, LCP and leave-one-token, free-slot selection, lease
invalidation, drafter refusal, no-store and failed/poisoned lanes. The same cache
owns resident descriptors and immutable checkpoints with one eviction policy.
Native generic state/input profiles remain default OFF. Native compilation and
full IDs/raw logits/state/capacity/performance qualification remain pending.
The original source fixtures and expectations are unchanged.
