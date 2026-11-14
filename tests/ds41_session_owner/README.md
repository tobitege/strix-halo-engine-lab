# DS41 actual lead host witness

This executable compiles exact, unique-anchor source islands from
`apps/deepseek41_serve_main.cpp`: request records, the actual owner/control loop,
control encoding/exchange, transfer helpers and session/import handling. The
fixture supplies deterministic native ABI doubles and a small startup shell.
It does not recreate the lead scheduler. Production and fixture both link
`strix_session_owner`; root CMake also links that owner into the real DS41 serve
executable. No fake native headers are installed or passed to device targets.

The root fixture reuses the existing DS41 session/runtime/codec libraries. The
standalone project compiles their actual host sources with the common queue,
ApiService, Runtime, KvStore, cache and IO closure. There is no runtime/queue/cache
copy or reverse storage-to-HTTP dependency.

```sh
cmake -S tests/ds41_session_owner -B build/ds41-owner-r16/host
cmake --build build/ds41-owner-r16/host --config Release --target ds41_session_owner_tests session_runtime_tests
ctest --test-dir build/ds41-owner-r16/host -C Release -R '^(ds41_session_owner(_extended|_checks)?|session_runtime)$' --no-tests=error --timeout 30 --output-on-failure --output-junit host.xml
```

Repeat build and selection in Debug. Each command, including teardown, has an
inclusive 120-second maximum. Configure/build failure forbids stale-binary CTest.
Checks throw under NDEBUG. The sentinel test requires exactly exit 1, empty stdout
and the deliberate failure line on stderr.

Optional `DS41_OWNER_REFERENCE` and `DS41_OWNER_EXTENDED_REFERENCE` select pinned
independent captures. They are distinct records: the first actual original was
compiled/captured before production edits; the supplemental harness was compiled
later against immutable task-entry sources. Never generate expected output with
the adopted owner or relabel supplemental results as the pre-edit original.
An independent archived-source build uses `DS41_OWNER_ENTRY=ON` and
`DS41_OWNER_ROOT` pointing to its complete archived source closure.

The supplemental selection exercises 17 cases: ordinary/mixed requests, invalid
context, stop/length, real ApiService HTTP completion, pool waiting/retry and
permanent refusal, defer/short-first/rescan/cancel, interleave/cancel, imported
first-token reuse, named active conflict, hetero grouping, speculative-double
accept/reject/commit, hetero speculative flushing and drain. Independent complete
token checks supplement exact capture comparison. Native physical identifiers in
the double are 101 + 17 * native-slot and are unrelated to logical handle values.
Runtime tests cover more pending requests than slots, stable borrowing across
promotion, failed validation, exhausted logical capacity, stale/foreign/reused
handles before callbacks, admission/progress, replacement accounting, native
failure cleanup and destruction with active/pending/reserved work.
The adopted-only failure selection executes standard and nonstandard failures
through the actual lead, with one active request and six deferred tickets. It
checks terminal accounting and exactly one HTTP stop; its trace is emitted for
capture. This safe cleanup is not compared with legacy abandoned tickets.

Comparisons preserve complete token IDs, nonvolatile native receipts, queue
counters, outgoing control bytes and native-call order. Declared volatile fields
are steady-clock durations (queue/prefill/decode/first-token/elapsed milliseconds
and their derived decode rate), HTTP completion ID/creation time and the
clock-derived session generation envelope. The latter uses a fixed clock ABI
double independently on both paths; other control bytes are compared exactly.
Raw captures and native logs remain separate. The peer is an explicit ABI double,
not an execution of the real follower, and the startup shell does not exercise
CLI/model inspection, real listeners, tokenizer loading or device resources.

The full requested host matrix is incomplete: all idle/busy control/import
combinations, changed-history/session continuation, heartbeat/quiesce, physical
detach/rollback and socket/fatal lifecycle need additional witnesses. Strict
whole-root warnings and platform portability remain independent gates. No native
fidelity, capacity, speed, no-slowdown, generic fanout or full-suite pass is claimed.
See the [adoption contract](../../docs/engine/session-ds41-owner-adoption.md) for the
separate NOT RUN both-rank and complete-workload operator tickets.
