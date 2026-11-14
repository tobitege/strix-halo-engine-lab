# DS41 serving request ownership

The DS41 lead enters the existing `SessionRuntime::run()`. Runtime owns queue
reservations, taken pending requests, active logical leases, cancellation
inspection, token delivery and terminal ticket accounting. `Running`, deferred,
admission-ready and pool-waiting records retain opaque handles and native
execution metadata. The replaced outer request loop and direct queue take/finish
calls are removed from the lead. `ApiService` remains the frontend.

Runtime pending handles bind an owner and a never-reused generation. Holding a
reservation acquires no logical lease and consumes no native slot. Other pending
requests do not invalidate borrowed inspection. Promotion preserves that storage,
submission/take/observation metadata and accounting, and occurs only after the
original parked/defer/pool admission decisions. Refusal retires the handle before
completion. Stale or foreign handles cannot invoke callbacks. Shutdown, destruction
and escaping owner failures settle outstanding pending work. Fatal pending cleanup
is a safety improvement over abandoned legacy tickets, not a legacy fatal receipt
equivalence claim. Native terminal statistics replace logical publication totals.
Standard and nonstandard native failures settle active native receipts and
pending tickets, stop HTTP, then close queued work before the shell joins HTTP.
Nonstandard failure cleanup is intentional; the legacy thread termination path
is not an independent terminal-receipt oracle.

The bounded control turn retains the original scheduler order. It processes the
existing control work and deferred rescan, then considers one admission candidate
per invocation. Its cursor resumes the same admission sweep before further admin,
prefill or decode work. Yielding performs no native call. The native grouped
decode, prefill pieces/tasks, speculative plans, hetero groups and follower remain
family operations. This adds no default scheduling optimization or fanout protocol.

`BatchSequence::slot`, free and parked slots, pool reservations, pending reset and
snapshot masks, commits and in-flight plans describe physical execution. They stay
in the family because imported state, grouped work and both-rank control refer to
those physical resources. A logical `KvHandle::slot` never selects or resets them.
The request-only adapter uses two logical rank views without adopting device
resources. Generic execution, physical state access, restore, fork and append are
unsupported. Generic batch/cache capabilities are not enabled. Retaining-cache
construction is disabled; existing DS41 RAM and durable cache owners are retained.

The [host fixture](../../tests/ds41_session_owner/README.md) extracts exact source
islands of the actual lead owner and links the real queue, Runtime, logical store,
ApiService and family state/session codecs. It includes a separately captured
pre-edit original and a distinct later supplemental archived-source replay.
Native ABI doubles establish control evidence only. Host coverage is incomplete:
the actual follower/socket lifecycle, all idle/busy admin/import combinations,
changed-history named continuation, heartbeat/quiesce and native failure isolation
still require qualification. This is a real owner adopter with remaining gates,
not completion of the full serve/session consolidation.

One ordinary paired ticket was run before and after the adoption: six greedy
requests (a single prompt, two prompts sharing a 1,280-token cached prefix, an
exact repeat and a two-turn conversation) returned byte-identical responses,
usage counts and prefill logit digests. The default paired route keeps the named
session cache off, so that ticket does not cover parked sessions.

All other native tickets are **NOT RUN**. Each configure, compile, link, per-arm paired
capture, exact comparison and interleaved complete-workload A/B is a separate
inclusive <=120-second ticket, including teardown. Missing toolchains, assets,
hardware or timeout means NOT RUN/incomplete. Required arms are ordinary,
interleave, defer, hetero, pool pressure/refusal, imports/parked state, named
sessions, speculative acceptance/rejection, failures with a live peer and drain.
Capture full IDs, logits, physical state, frontiers, RNG and receipts on both ranks;
verify detach/rollback isolation, unchanged capacity/defaults/kernel objects,
then complete-workload TTFT/throughput with no slowdown. Build HIP and shared and
legacy ON/OFF organizations. Host replay does not qualify these gates.

DS41 durable-index consolidation, remaining Qwen factory/cache policy, GLM and
complete-turn/COW/fork/append remain unfinished. Existing source-admission and
Runtime import-order provenance debt is unchanged. Frozen fanout contracts and
numerical qualification rows are untouched.
