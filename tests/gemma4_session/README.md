# Gemma4 serving adoption host replay

This project uses the actual Gemma4 Api, ChatJob, Coordinator and fake Backend,
not the generic scripted HTTP oracle. It loads the real tokenizer implementation
with a generated synthetic asset. That implementation requires exactly 262,144
vocabulary entries; the fixture uses placeholders, byte tokens and a few merges,
not downloaded model/tokenizer assets. No fidelity claim follows from it.

## Independent original capture

`fixtures/baseline.json` records revision
`52157f16bdbe600fdb7534cb9c4b071e3330f098`, raw source SHA-256 values,
fixture hashes and the SHA-256 of the original capture executable. The original
Api/Coordinator executable was built and captured **before production edits**.
Release replay passed repeatedly; Debug matched those same frozen expectations.
The binary and objects were retained in the baseline build directory. Original
sources are recoverable by the recorded Git revision; no production copy of the
old owner or kernels is added to this tree. Both current mains use migrated code
and are not an independent native old/new reference.

`legacy_expected.json` freezes normalized HTTP/SSE byte strings and full
receipts. `legacy_trace.json` freezes prompt chunks, grouped forward row slots,
generations, positions and IDs, direct Coordinator receipts, frontiers, complete
output IDs, RNG cursors and state digests. Only created timestamps, generated
request IDs and timing-valued fields/arrays are masked. Slot decisions, cache
counts/bytes, identities, draft counts, windows and scheduler counters are kept.
Measured time fields are not used as deterministic expectations.

The HTTP group uses a condition-variable gate in its first Backend call. The
remaining handlers are enqueued in a checked order before that call is released.
No sleep-based ordering selects which jobs join the next tick. Draft speculation
in the frozen mixed Coordinator case is explicit, so host timing noise cannot
select a different draft window. Existing integration tests separately exercise
measured auto policy.

The capture command originally used was:

```sh
cmake -S tests/gemma4_session -B build/gemma4-session-baseline-p4 -DGEMMA4_SESSION_BASELINE=ON
cmake --build build/gemma4-session-baseline-p4 --config Release --target gemma4_session_tests
python tests/gemma4_serve/make_deployment_fixture.py build/gemma4-session-baseline-p4/tokenizer.json
build/gemma4-session-baseline-p4/Release/gemma4_session_tests build/gemma4-session-baseline-p4/tokenizer.json --capture
```

For reproduction after migration, recover the eight recorded original sources
under `build/gemma4-session-baseline-p4/original/`, preserving relative paths and
recorded bytes. `git show REVISION:PATH` retrieves them; if the checkout used
CRLF, select the CRLF form only when its hash matches the recorded raw hash.
Then add this configure argument (absolute source-root accepted by CMake):

```sh
-DGEMMA4_BASELINE_SOURCE_ROOT=build/gemma4-session-baseline-p4/original
```

Pass an absolute source-root or resolve the relative path from the repository
root before configuring. The configure guard refuses any changed original
source. The baseline target compiles the original Api/Coordinator/ChatJob and
uses their original local headers first. Shared HTTP/queue and unchanged family
support sources are linked as ordinary host dependencies. No migrated runtime
is called by the original Api. Reproduction normally runs without `--capture`
and compares against the checked-in expectations. Capture is compiled out of
the migrated executable; it cannot regenerate expectations to accept a change.

## Migrated execution and regression commands

```sh
cmake -S tests/gemma4_session -B build/gemma4-session-p4
cmake --build build/gemma4-session-p4 --config Debug
ctest --test-dir build/gemma4-session-p4 -C Debug --output-on-failure
cmake --build build/gemma4-session-p4 --config Release
ctest --test-dir build/gemma4-session-p4 -C Release --output-on-failure
```

All checks throw on failure under NDEBUG. Single-config generators can use
Debug/Release subdirectories beneath the authorized build directory.

| Evidence | Path executed | Checks |
|---|---|---|
| Frozen sequential and controlled concurrent HTTP | Actual Api -> shared runtime -> adapter -> grouped Backend | HTTP/SSE bytes, full deterministic receipts, seeded sampling, stop/length, thinking restart, failure recovery, prompt chunks, slot reuse |
| Frozen mixed-phase comparison | Shared runtime -> SessionEngine -> Coordinator::advance -> neutral Backend plans | Complete original grouped row trace/receipts, explicit speculative acceptance and bonus/length boundaries, prefill limit/budget, cancellation and RNG cursor |
| Cache and lane state | Actual adapter/store/shared cache/runtime | Warm physical reuse, fresh sampler cursor, no-store hit/miss without retention, pending full checkpoints, restore equality, generation rejection, no false COW |
| Long-call cancellation | Actual adapter/native planner over fake collectives | New cancellation observed between prefills inside one synchronous advance, no mid-collective interruption |
| Existing Gemma4 batching regression | Actual Api/shared runtime, original assertions | Concurrent mixed lengths, seeded/greedy, thinking, SSE, disconnect, 4/64/whole prefill budget, mismatched prefill arithmetic, cold/warm conversations, failure/reuse |
| Existing Gemma4 integration | Original host/off/chat assertions | Target/speculation, generation lifetimes, tools, reasoning, deployment/drain routes and stream handling |
| Actual factory lifecycle | Concrete Gemma4 factory with owned fake assets | Registry selection, metadata/help no-load boundary, deferred family inspection, external owner execution, listener config, asset release |
| Shared contracts | Existing plus focused new host tests | Unsupported default mixed API, sealed prompt purpose, malformed/foreign payload refusal, byte policies, partial lane failure publication, live cancellation, generic HTTP replay and registry |

The frozen API set includes a backend arithmetic-mode transition to catch policy
lifetime changes. Production cache arithmetic admission is fixed at startup, as
in the original owner; actual backends must retain that identity throughout a
service lifetime. Separate existing batch tests start with unequal arithmetic
and verify chunk-boundary-only reuse. No-store and safe-boundary cancellation
are additional adapter/runtime assertions, not newly invented frozen old results.

## Integration and qualification limits

Root CMake builds `strix-gemma4-session-tests` with the actual family libraries.
`STRIX_LEGACY_MAINS=ON` retains the legacy executable; OFF retains the family,
new server and tests. A family-unbuilt standalone shell is available through
`tests/session_runtime`. The independent recurrent target is integrated when
its source is present and is not a Gemma4 source/API dependency.

Official-tokenizer tests still skip without official assets; synthetic replay
does not replace them. Host compilation/linking is evidence for the ownership
and grouped-plan adoption, not for accelerator ABI or performance.

Native gates still required: compile/link host and device family variants with
legacy ON/OFF; compare the original revision and migrated service at identical
model/profile/draft/context/capacity; compare every token, raw logits, physical
state and phase receipt; exercise real drafter initialization/seeding, cache
failure/decline and unequal-prefill checkpoints; stress cancellation, feature
export/deployment drain and slot reuse; measure interleaved cold/warm complete
workloads. Full-lane snapshots require physical forks and complete drafter
clones, otherwise capture is refused. Native resource adoption and COW remain
unsupported/unqualified. No device execution or speed/no-slowdown claim is made.

## Executed host verification

The standalone Debug and Release suites each passed 15/15. The root OFF
Debug and Release selection each passed 17 tests, with two official-tokenizer
tests reported as skips. Root legacy ON built and its help/inspection executed;
OFF retained the shared server, family and replay targets. The family-unbuilt
shell passed registry tests, inspected metadata, and refused execution with
exit 3. The independently rebuilt original-source reference also matched the
frozen oracle without capture. No accelerator build or execution was performed.
Root host configuration used STRIX_WARNINGS_AS_ERRORS=OFF for existing host
compiler warnings; assertions were not weakened.

The root target set was built in both configurations using:

```sh
cmake -S . -B build/session-integrated-p4 -DSTRIX_FAMILY_GEMMA4=ON -DSTRIX_ENABLE_HIP_RUNTIME=OFF -DSTRIX_LEGACY_MAINS=ON -DSTRIX_WARNINGS_AS_ERRORS=OFF
cmake -S . -B build/session-integrated-p4 -DSTRIX_LEGACY_MAINS=OFF
cmake --build build/session-integrated-p4 --config Debug --target strix-serve strix-gemma4-session-tests strix-gemma4-integration-tests strix-gemma4-features-tests strix-gemma4-serve-tests session_engine_contract_tests session_backend_tests session_kv_tests session_prefix_cache_tests session_runtime_tests session_http_replay_tests session_registry_tests session_fanout_tests session_recurrent_tests -j 4
ctest --test-dir build/session-integrated-p4 -C Debug -R '^(session_|gemma4-session|gemma4-integration-(host|off)|gemma4-deployment|gemma4-features-host|gemma4-serve-(host|tokenizer))' --output-on-failure
```

The same build/test commands were executed with Release. The final incremental
family rebuilds included strix-serve, strix-gemma4-session-tests,
strix-gemma4-integration-tests and strix-gemma4-features-tests before repeating
the root selection. All final build logs were checked for compiler/link errors.
Unbuilt verification used:

```sh
cmake -S tests/session_runtime -B build/session-integrated-p4/unbuilt
cmake --build build/session-integrated-p4/unbuilt --config Release --target strix-serve session_registry_tests -j 4
ctest --test-dir build/session-integrated-p4/unbuilt -C Release -R '^session_registry$' --output-on-failure
```

The source-byte checker was invoked with every path below. Since it scans only
tracked files, every listed file was also explicitly read as bytes and checked
for raw NUL, including untracked sources/fixtures. Frozen fixture SHA-256 values
were verified unchanged and git diff --check passed.

## Exact edited and new files

This list excludes pre-existing unrelated workspace changes.

- `CMakeLists.txt`
- `apps/gemma4_serve_main.cpp`
- `cmake/StrixSession.cmake`
- `cmake/gemma4_d2.cmake`
- `cmake/gemma4_g.cmake`
- `include/strix/serve/engine.hpp`
- `include/strix/serve/session_runtime.hpp`
- `include/strix/serve/run.hpp`
- `include/strix/session/engine.hpp`
- `include/strix/session/kv.hpp`
- `include/strix/session/prefix_cache.hpp`
- `src/session/kv.cpp`
- `src/session/prefix_cache.cpp`
- `src/serve/session_runtime.cpp`
- `src/serve/run.cpp`
- `src/gemma4/engine/api.hpp`
- `src/gemma4/engine/api.cpp`
- `src/gemma4/engine/coordinator.hpp`
- `src/gemma4/engine/coordinator.cpp`
- `src/gemma4/engine/runtime.hpp`
- `src/gemma4/engine/runtime.cpp`
- `src/gemma4/engine/session_engine.hpp`
- `src/gemma4/engine/session_engine.cpp`
- `src/gemma4/engine/session_service.hpp`
- `src/gemma4/engine/session_service.cpp`
- `tests/session_engine_contract_tests.cpp`
- `tests/session_kv_tests.cpp`
- `tests/session_prefix_cache_tests.cpp`
- `tests/session_runtime_tests.cpp`
- `tests/session_registry_tests.cpp`
- `tests/session_runtime/CMakeLists.txt`
- `tests/gemma4_session/CMakeLists.txt`
- `tests/gemma4_session/session_tests.cpp`
- `tests/gemma4_session/replay_tests.cpp`
- `tests/gemma4_session/fixtures/requests.json`
- `tests/gemma4_session/fixtures/legacy_expected.json`
- `tests/gemma4_session/fixtures/legacy_trace.json`
- `tests/gemma4_session/fixtures/baseline.json`
- `tests/gemma4_session/README.md`
- `docs/engine/session-layer.md`
- `docs/engine/backends.md`

## Exact input continuation prerequisite

[Gemma4 input continuation](../../docs/engine/gemma4-continuation.md) describes
admission, state ownership, policy initialization, bounded mixed input, terminal
resume and complete rollback. `gemma4-session-continuation` is registered only
in this nested project and runs real tiny-model HostBackend/HostTarget work.
Its direct serial causal reference checks complete IDs, retained final logits,
canonical KV digests and RNG cursors. Always-active checks cover final-row
consumption at 1/7/8/9/15/16/17/25 tokens, ring wrap, pending anchors, alternating
input/output, zero allowance, stop/context/cancellation/resume, unsupported
admission, failure after evaluation or commit, failed restoration quarantine,
and one grouped forward shared with unrelated legacy work, including long
prompts that would otherwise take a separate prefill call. No fixture,
original capture, replay environment or existing integration assertion changes.

The supported mode is host `g4-v1` with no configured drafter factory. Initial
nonempty shared `prefill` admits input mode; legacy `prepare/start/advance`
requests keep their original semantics. Optional prefill/decode batch APIs,
row hints, native input continuation, fork/COW and request-lane batch flags
remain false. Full host snapshots are independent rollback copies. They do not
share prefix pages or establish a complete fanout request/native qualification.

Run the four host configurations entirely beneath the dedicated build tree:

```sh
cmake -S tests/gemma4_session -B build/gemma4-input-p8/gemma-on -DSTRIX_LEGACY_MAINS=ON -DSTRIX_ENABLE_HIP_RUNTIME=OFF -DSTRIX_ENABLE_CUDA_TRELLIS=OFF -DSTRIX_ENABLE_CUDA_DS41=OFF
cmake --build build/gemma4-input-p8/gemma-on --config Debug
ctest --test-dir build/gemma4-input-p8/gemma-on -C Debug --output-on-failure
cmake --build build/gemma4-input-p8/gemma-on --config Release
ctest --test-dir build/gemma4-input-p8/gemma-on -C Release --output-on-failure
cmake -S tests/gemma4_session -B build/gemma4-input-p8/gemma-off -DSTRIX_LEGACY_MAINS=OFF -DSTRIX_ENABLE_HIP_RUNTIME=OFF -DSTRIX_ENABLE_CUDA_TRELLIS=OFF -DSTRIX_ENABLE_CUDA_DS41=OFF
cmake --build build/gemma4-input-p8/gemma-off --config Debug
ctest --test-dir build/gemma4-input-p8/gemma-off -C Debug --output-on-failure
cmake --build build/gemma4-input-p8/gemma-off --config Release
ctest --test-dir build/gemma4-input-p8/gemma-off -C Release --output-on-failure
```

The nested project declares CXX only; accelerator options cannot enable device
languages here. The suite includes all ten shared contract/storage/recurrent/
fanout/runtime/HTTP/registry cases and all original family replay/batch/off/chat
integrations. Compiler/native execution, complete model/state fidelity, physical
COW isolation, unchanged-capacity and complete-workload gates remain outstanding.

## Final-turn host tests

The explicit `commit_pending` transition and `turn_cache_candidate` producer
are documented in the [family continuation guide](../../docs/engine/gemma4-continuation.md#explicit-final-turn-commit-and-reuse-producer).
Only real HostBackend `g4-v1` without any configured drafter supports them, on
both legacy and input paths. Sealing consumes an existing pending token without
selection/RNG or terminal/accounting reset; it grants no resume permission.
A complete-turn prompt checkpoint reuses all causal physical/auxiliary state
while a new request initializes its own sampling and output budget. Ordinary
Host prefix-cache flags/default admission are unchanged. Completed-turn HTTP
wiring is explicit opt-in; see the integration below.

The `gemma4-session-turn-commit` executable
reuses continuation_tests.cpp's loaded model, observation/fault helpers and
direct serial HostTarget reference. Its assertions remain active in Release;
the original continuation executable and all replay/integration assertions
remain active and unchanged. The test compares consumed/generated IDs, raw
retained logits, physical digests, frontier/pending and RNG/status across stop,
limit, final-context, zero allowance, repeated sealing, thinking transcripts,
row boundaries/ring wrap, and fresh-request restoration. Faults after evaluation
and publication restore the old terminal boundary; failed rollback quarantines
only the affected lane. Captures survive source release and new slot writes.

Full-host retained-size accounting uses captured vector capacities and fixed
object/string metadata, including stream and Coordinator state. Allocation
bookkeeping/RSS and shared cache envelopes are outside the family payload
measure. Unknown stream/drafter sizes are unsupported; historical legacy-only
estimates cannot qualify complete-turn retention. These are full host copies,
not COW. Default native continuation, native completed-turn caching, physical
COW/fork, request-lane batch and all configured drafters remain unsupported. There is no portable backend claim.

```sh
cmake -S tests/gemma4_session -B build/gemma4-turn-r2/gemma-on -DSTRIX_LEGACY_MAINS=ON -DSTRIX_ENABLE_HIP_RUNTIME=OFF -DSTRIX_ENABLE_CUDA_TRELLIS=OFF -DSTRIX_ENABLE_CUDA_DS41=OFF
cmake --build build/gemma4-turn-r2/gemma-on --config Debug -j 4
ctest --test-dir build/gemma4-turn-r2/gemma-on -C Debug --output-on-failure
cmake --build build/gemma4-turn-r2/gemma-on --config Release -j 4
ctest --test-dir build/gemma4-turn-r2/gemma-on -C Release --output-on-failure
cmake -S tests/gemma4_session -B build/gemma4-turn-r2/gemma-off -DSTRIX_LEGACY_MAINS=OFF -DSTRIX_ENABLE_HIP_RUNTIME=OFF -DSTRIX_ENABLE_CUDA_TRELLIS=OFF -DSTRIX_ENABLE_CUDA_DS41=OFF
cmake --build build/gemma4-turn-r2/gemma-off --config Debug -j 4
ctest --test-dir build/gemma4-turn-r2/gemma-off -C Debug --output-on-failure
cmake --build build/gemma4-turn-r2/gemma-off --config Release -j 4
ctest --test-dir build/gemma4-turn-r2/gemma-off -C Release --output-on-failure
```

The nested project is CXX-only; unused accelerator-option warnings do not enable
HIP/CUDA. The existing seven unresolved MiniMax native/default source-audit rows
and archived Hello failures remain open external validation gates. Gate-helper
platform skips and MSA/POSIX Not Run outcomes are not passes. No native compiler,
device/model fidelity, physical capacity or complete-workload timing claim is
made by this family prerequisite.


## Api completed-turn integration

`gemma4_turn_cache_integration_tests` / `gemma4-session-turn-cache` runs the
production `Api::handle`, GenerationQueue and shared SessionRuntime with real
loaded host weights and HostTarget. The ordinary `main` takes exactly one
writable generated-fixture directory. All generated tokenizer assets/scripts
stay there. It reuses the existing tiny-model coefficients, extending vocabulary
rows to represent the production adapter's fixed boundaries, the existing
fixture generator, and the independent serial causal reference. Conditions use
always-active `CHECK`, including Release. Baseline capture does not compile it.
No frozen fixture or expected output is refreshed.

The test resubmits actual rendered/tokenized HTTP history. No cache entry is
fabricated and no test calls commit/candidate between requests. Cache-enabled
cold and warm replies are compared with independent cold execution, including
JSON and complete SSE framing in continuous single-slot and FIFO modes,
generated IDs, raw logits for every physical row,
completion digests, sampling cursors, suffix-only rows, one final-anchor row and
fresh new-job seed/output allowance. Existing IDs/timestamps/timings and the
physical cache-work statistics documented in the [Api contract](../../docs/engine/gemma4-continuation.md#opt-in-api-completed-turn-retention)
are normalized for wire comparison; other fields remain exact.

Additional cases exercise edited/template/topology/conversation misses, missing
metadata, no-store, bounded RAM/payload/transient refusal, LRU/protected keeps,
real final-target/candidate/capture/restore faults, safe slot reuse, stop/EOS,
context capacity, disconnects, and thinking continuation versus incomplete
thinking refusal. Model replacement and renderer-normalized thinking histories
safely miss; the latter do not claim a thinking-history cache hit. The original final-commit and
continuation tests retain stop/context/zero-output/rollback/quarantine boundary
coverage; the integration does not widen unsupported native/drafter/batch/COW
capabilities. Host evidence does not qualify native fidelity or performance.

```sh
cmake -S tests/gemma4_session -B build/gemma4-cache-r3/host-on -DSTRIX_LEGACY_MAINS=ON -DSTRIX_ENABLE_HIP_RUNTIME=OFF -DSTRIX_ENABLE_CUDA_TRELLIS=OFF -DSTRIX_ENABLE_CUDA_DS41=OFF
cmake --build build/gemma4-cache-r3/host-on --config Debug -j 1
ctest --test-dir build/gemma4-cache-r3/host-on -C Debug --output-on-failure
cmake --build build/gemma4-cache-r3/host-on --config Release -j 1
ctest --test-dir build/gemma4-cache-r3/host-on -C Release --output-on-failure
cmake -S tests/gemma4_session -B build/gemma4-cache-r3/host-off -DSTRIX_LEGACY_MAINS=OFF -DSTRIX_ENABLE_HIP_RUNTIME=OFF -DSTRIX_ENABLE_CUDA_TRELLIS=OFF -DSTRIX_ENABLE_CUDA_DS41=OFF
cmake --build build/gemma4-cache-r3/host-off --config Debug -j 1
ctest --test-dir build/gemma4-cache-r3/host-off -C Debug --output-on-failure
cmake --build build/gemma4-cache-r3/host-off --config Release -j 1
ctest --test-dir build/gemma4-cache-r3/host-off -C Release --output-on-failure
```

The CXX-only project cannot compile accelerator sources. Root registration
uses the same source with its own generated-fixture directory. Frozen default
family/shared replays remain independent tests. Known unrelated native/default
source-audit failures, archived Hello failures and platform helper skips remain
open; this nested host matrix does not rebaseline or qualify them.

## Native continuation operator probe

`native_continuation_tests.cpp` is a loaded-device operator probe, with no fake
HIP headers, fallback execution or fixture-generation success path. The nested
CXX-only project compiles it and the actual `device_backend.cpp` into object
libraries. Those translation checks do not link device symbols or register GPU
tests. Root's existing `strix_gemma4_engine_device_host_compile` and new
`strix_gemma4_native_continuation_host_compile` provide the same distinction.
Always-active host continuation/commit tests retain their validation, transaction,
rollback and quarantine coverage. The host device-checked-result regression
exercises retained greedy selection beside another lane's winner array, without
claiming device execution. Real two-turn Api and independent frozen HTTP/SSE/
receipt oracles remain active, including Release.

A Linux/HIP operator builds `strix-gemma4-native-continuation-tests` under the
existing Gemma family, device-backend and BUILD_TESTING guards. Registration is
explicit: `STRIX_GEMMA4_NATIVE_CONTINUATION_TESTS=ON` requires all three paths
`STRIX_GEMMA4_NATIVE_MODEL`, `STRIX_GEMMA4_NATIVE_ROPE` and
`STRIX_GEMMA4_NATIVE_IDENTITY`; a missing path is a configure error. Executing
the probe also validates every asset and fails on missing/mismatched inputs.
No missing reference is a pass. The executable takes exactly these arguments:

```sh
strix-gemma4-native-continuation-tests --model assets/model.gguf --rope assets/rotary --identity assets/continuation-identity.json --case prefix
```

The operator identity JSON must contain actual loader-verified asset hashes,
an explicit admitted context and original input token IDs, for example:

```json
{"identity":"g4-v1","model_hash":"<actual model hash>","rope_hash":"<actual rotary hash>","context":4096,"prompt_ids":[2,17,29,8]}
```

Hashes must match the loaded assets; placeholders cannot pass. `context` must
fit the model and rotary tables and be at least 48. Input IDs must be in the
loaded vocabulary. The probe cycles the supplied original IDs to construct
bounded causal inputs. Wrap requires `context > window + 16`. No new sampler
or model math is provided: each device row is compared bitwise with existing
HostTarget serial passes over independent HostStream transactions. Every copied
or inspected native state digest/frontier must equal that reference. Exposed
raw logits, hidden/taps and device winners are checked per row. Session checks
compare full consumed/generated IDs, pending anchors, RNG cursors and receipt
frontier/digest/counters; they do not compare volatile timings to reference time.

Each case is individually selectable and centrally registered as
`gemma4-native-continuation-CASE`:

| Case | Required evidence |
|---|---|
| `refuse-route` | Inspection-only resolved-route refusal. Central `refuse-a8`, `refuse-wmma`, `refuse-attention` tests run this case in separate processes with the corresponding numerical route environment. |
| `activation` | Actual native factory inspection stays default OFF; explicit inspect remains nonexecuting; unsupported model/cache/typed route refuses. This case is inspection only, without the gpu label. |
| `queue-lifetime` | Three native rows, complete fork, source release and producer target/queue teardown before snapshot digest and another fork. Central `queue-lifetime-no-queue` repeats with `STRIX_G4_KV_QUEUE=0`. |
| `legacy` | Original default-OFF native Coordinator request, including 25-token many-row prompt boundaries, versus opt-in continuation: full IDs, frontier, RNG, finish and state digest. |
| `prefix` | Exact arbitrary boundary 9, all input consumed, zero output/RNG, retained selection with zero new forwards, pending plus suffix, empty no-op, snapshot restore and source release/reuse isolation. |
| `wrap` | Same at `window + 3`, copying complete wrapped rings/tags. |
| `terminal` | Greedy/sampled length and stop, final anchor exactly once, zero-output seal and repeated seal. |
| `context-pending-greedy`, `context-pending-sampled`, `context-full-greedy`, `context-full-sampled` | Individually selected last-context pending seal or fully consumed context; one sampling arm per ticket. |
| `cancel` | Cancellation before pending/suffix work, terminal refusal, explicit resume and fresh source, unchanged output/RNG. |
| `rollback` | Real native evaluation and post-publication injected host failures after earlier chunk commits, restoration, failed-copy quarantine and peer/source isolation. Driver fault injection remains a separate hardware gate. |
| `rollback-wrap` | Same injected failure/peer checks after sliding wrap; requires `context > window + 48`. |
| `mixed` | Actual call/row widths across input/decode/legacy lanes, bounded suffix progress and retained greedy selection beside a native row. |

Every execution case prints actual native forward calls, rows and grouping plus
the resolved identity. Input cases assert zero selected output and target RNG
draws before decode. Configured factories refuse before forwards. Host compilation,
actual HIP compiler/link, device execution, capacity and complete-workload timing
are separate outcomes. These probes do not establish native qualification,
performance, page COW/fork or configured-drafter support. Preserve receipts
privately and keep private deployment identifiers out of public changes.

## Native accounting and Api operator tickets

`gemma4-session-native-memory` is always active in nested and root HOST builds.
It independently enumerates admitted global arrays, complete sliding rings/tags,
aliased producers, vector capacities, publication/pinned descriptors and spare
overlays against the same checked accounting helpers used by DeviceStream.
It checks overflow, zero/unknown limits, old/new/rollback owners and repeated
capture/release lifetimes. It is host algebra evidence, not measured native RAM.
`gemma4-session-turn-cache` additionally exercises real HostTarget HTTP/SSE and
a labeled host memory-reporting wrapper through the native-bound admission
seam. Unknown/zero/overflow and insufficient budgets refuse before forks/seal.
All frozen replay/receipt and existing continuation/commit assertions stay active.

Every registered native ticket now has a **120-second hard timeout**. `wrap`,
`rollback-wrap` and the four context arms remain separate selectable tickets; use a short
loaded exact fixture and explicit small context rather than a long sweep.
A timeout is a failure, never qualification. Larger contexts need separate
bounded assets/tickets; do not raise this timeout to accommodate a sweep.

The same native executable also links the actual Api integration helpers with
`STRIX_G4_NATIVE_TURN_PROBE`. It creates no synthetic model/tokenizer fixture.
`gemma4-native-api-CASE` selects `--case api-CASE`, with CASE:

| Case | Required checks |
|---|---|
| `turn`, `sse` | Two actual Api/SessionRuntime turns; warm suffix versus independent cold same-profile native target; complete IDs, exposed per-row logits, canonical KV digests/frontiers, RNG and stable receipt/wire fields; final anchor once; first source release/slot reuse. |
| `no-store` | First turn retained; warm lookup permitted, no second store/seal, retained charge unchanged. |
| `mismatch` | Caller template identity changes; safe miss and independent cold equality. |
| `pressure`, `retained-pressure` | One-byte transient or retained budget; both turns decline before copies/final target work. This is admission pressure, not physical hardware pressure. |
| `no-retention` | Native continuation activated but TurnCacheConfig disabled; no retention/copies. |
| `fault` | Real native warm restore then an injected rollback-copy failure at optional finalization; completed response/receipt survives and old retained charge remains. |
| `cancel` | Disconnect during real native suffix work; no new retained state, then fresh request cancellation/output policy and safe reuse. |

Registration explicitly supplies `STRIX_G4_GRAPHS=0`, `STRIX_G4_CHUNK_DECODE=0`,
`STRIX_G4_PREFILL_EXACT_MANYROW=0` and exact numerical settings. The resolver
is never silently modified by admission. Graphs/FAST/A8/chunk-private streaming,
active DevicePrefill and configured drafter cache profiles remain unsupported.
Each identity JSON additionally supplies the actual `tokenizer` path (relative
to that JSON or absolute), `api_request` as a complete ordinary non-thinking
chat request with 1..4 output tokens, positive `api_row_budget` <=128,
`max_state_bytes`, `max_transient_bytes` and `retained_budget`. Both rendered
prompts plus outputs must fit the row budget and admitted context. Select an
ordinary request/seed whose output round-trips through the real renderer;
the probe asserts actual token-prefix equality rather than inventing a hit.
Thinking-history normalization remains a safe miss in host integration.
Missing tokenizer, hashes, bounded requests/budgets or ordinary cacheable output
fails the ticket. No reference/fixture bytes are rewritten to pass.

The probe prints actual native retained host/device/pinned payload queries,
cache retained charge, existing device/pinned service-buffer payload, the
additional-work bound and both finite budgets. These are backing allocation
bytes/bounds, not allocator RSS, driver/OS reservations or measured performance.
The observer's own trace storage is separate test overhead. Host CXX object
targets compile both native probe sources, but never link or run them as a
native executable. HIP compile/link and every device ticket remain NOT RUN in
the host worker environment. Physical COW/fork and native capacity/performance
remain later gates.

`gemma4-native-accounting` (`--case accounting`) uses three actual native rows,
independently enumerates device global/ring/tag/publication/overlay bytes and
pinned descriptors, refuses busy and published-but-still-owned transactions,
then checks repeated full copies after source and producer target destruction.
It requires positive `max_transient_bytes` in the identity JSON before stream
allocation. `gemma4-native-cache-refuse-{graph,prefill,chunk}` loads the actual
selected executor and checks that private/unadmitted profiles produce no work
bound before any stream/copy/seal, without changing their resolved options.
These four additional tickets also have a 120-second hard timeout and are NOT
RUN in host-only validation. The default native legacy ticket additionally
checks that ordinary OFF streams keep their previous unknown byte declaration.

## Shared API adopter tests

`gemma4-session-shared-api` links the production `strix_serve` library and the
real Gemma4 family library. It reuses tiny loaded weights and the independent
serial oracle, with always-active Debug/Release assertions. It checks identical
normalized prompt IDs and generated IDs across chat, Messages and Responses;
text completions; SSE/error schemas; live content delivery while the external
owner is paused after its first selected token; tokenizer delegation; policy refusal with
zero target calls; two-slot grouped work and reuse; ordinary stop strings;
feature exports serialized with live generation; trusted and wire no-store;
reasoning-budget continuation; and the completed-turn failure/pressure/cursor
proofs. `gemma4-session-features-api` retains the existing real feature suite.
A separate scripted presentation producer checks tool-event order, spontaneous
reasoning and error framing; it is a protocol test, not model fidelity evidence.
Frozen `gemma4-session-replay` remains an independent byte/receipt/row check;
its expected fixtures are never regenerated by this project.

```sh
cmake -S tests/gemma4_session -B build/gemma4-api-r6/host-on -DSTRIX_LEGACY_MAINS=ON -DSTRIX_ENABLE_HIP_RUNTIME=OFF
cmake --build build/gemma4-api-r6/host-on --config Debug
ctest --test-dir build/gemma4-api-r6/host-on -C Debug --output-on-failure
```

Repeat with Release and a separately configured legacy-OFF tree. The family
module also registers `strix-gemma4-shared-api-tests` against the root
production libraries. CXX-only native object checks are separate from native
compilation, linking, execution, fidelity and performance; all device gates
remain open. The shared POSIX socket API suite retains its platform dependency.
