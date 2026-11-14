# Qwen batched HTTP request ownership

The HIP batched HTTP entry constructs `qwen::HttpSessionOwner`, which enters
the existing `serve::SessionRuntime::run()`. SessionRuntime owns every taken
request, FIFO reservation, logical lease generation, publication, terminal
receipt and completion. The previous batched entry's live ticket vector,
deferred ticket, queue take/finish calls and infinite request loop are removed.

`MtpBatchRunner` still owns physical slots, lazy growth, memory admission,
prefill handoffs, nested service boundaries and native scheduling. Its numerical
code is unchanged. A logical lease number never selects or resets a native
slot. Native callbacks carry the complete logical handle, including store and
generation; publication/completion reject stale handles, and cancellation of a
retired handle returns true. The native runner independently chooses its free
physical slot. The logical KvStore has no physical initialization/reset hooks.
Generic prefill/decode, snapshot, fork and append execution are unsupported;
the facade advertises only native owner control and logical request limits.
HTTP keeps its existing executing capabilities and defaults.

Top-level planning advances at most one candidate per common-owner turn. A
three-phase cursor preserves both native `add_slot` calls around the complete
admission sweep, followed by the existing native step. Yielding adds no native
call or service-boundary wait. The synchronous nested callback reenters the
same owner, retains `can_prepare` checks and never grows slots. Its finite
candidate allowance covers the queued candidates plus the configured slot
capacity, preventing an endless stream of request-local refusals from holding
a parked service boundary. Successfully prepared requests consume physical
slots without executing their prompts in this callback.

A named FIFO reservation blocks following requests until all ordinary and
nested native work is idle. The session mutex is released before the existing
fixed named-session protocol and reacquired only before an ordinary native
submission. Constraint refusal and already-formed named receipts pass through
the common completion owner. Ordinary request translation preserves drafting,
sampling validation, seed/request identity, progress and cancellation callbacks.
There is no per-token receipt construction. Terminal native stats replace the
logical publication counter, avoiding double output accounting.

Idle admission uses the queue's timed condition-variable wait, including request
and control wakeups. Queue shutdown drains already accepted/native and reserved
work through the native boundaries, including the final growth pair. Failure
marks engine/probe failure, obtains native terminal receipts, finishes remaining
logical requests and reservations, abandons native state, releases the session
mutex, then closes queued work. The existing HTTP shell retains server/thread
and signal cleanup. Other SessionRuntime adopters retain immediate-close and
generic admission behavior unless they explicitly select these native controls.

Generic SessionRuntime retention is disabled, including construction of its
retaining-cache facade. Existing pooled prefix and named-session cache owners,
their identity guards and policies stay in the native runner/protocol. No
additional retaining cache is introduced.

## Host evidence and limits

[The host replay](../../tests/qwen_session_owner/README.md) compiles the actual
production binding against a bounded native ABI double and links the real
SessionRuntime, GenerationQueue, logical KvStore and ApiService. Independent
output/receipt/trace checks run without any external reference directory. The
separate reference selections compare a frozen task-entry owner extraction and
old-path HTTP response capture. Only explicit volatile HTTP envelope fields
are normalized: completion ID, creation time and queue wall time. Native timing,
all other receipts, output IDs and response framing are retained.

Host equality does not establish native scheduler/device equivalence, numerical
fidelity, capacity equality or performance. Source admission remains UNRESOLVED.
The owner, the host replays and the full host tree build with GCC and warnings
as errors; the HIP serving binary builds and links with the device backend on.
CUDA compilation is NOT RUN here.

A single-node device comparison of the batched HTTP entry before and after this
adoption (same assets and flags, prefix cache on, greedy, 48 new tokens) gave
identical output IDs, text, usage, cached-token counts, finish reasons,
prompt-state digests and schedule-invariant target-state digests for: three
sequential requests sharing a system prefix (one cold, two restoring the prefix
checkpoint), a second conversation turn built on the first answer, and the same
three requests submitted concurrently (at once and 0.25 s apart), in three runs
per arm. The diagnostic final-state digest also covers slot generations and
arithmetic lineage; it follows arrival timing and varied between runs of the
same binary in both arms, so it is not part of this comparison. This is one
workload; the ticket rows below are not replaced by it.
Physical COW/fork/append and a native configured drafter remain code/gate work.

This slice does not migrate Qwen serial HTTP/CLI, its factory, CUDA ownership or
remaining prefix/session cache policy. Next are Qwen serial/factory ownership
and cache policy, followed by DS41 and GLM request/cache/envelope slices. Each
slice must delete replaced ownership and preserve the existing pooled owner.

## Required operator tickets

Every row below is **NOT RUN**. Each build, per-arm capture, comparison and
interleaved A/B ticket is separate, with an inclusive wall-time limit of
120 seconds per arm. Missing assets/toolchains, unsupported hardware and timeout
mean NOT RUN/incomplete. Do not infer a pass from host replay or source hashes.

| Ticket | Required evidence |
|---|---|
| Native build arms | Actual strict HIP/CUDA compilation and linkage; shared and legacy mains ON/OFF configurations; unchanged kernel objects. No fabricated headers/toolchains. |
| Capture arms | Complete cold/warm/OFF/mismatch/thinking/stop/limit/cancel/failure/mixed/speculation/RAM/disk output IDs, logits, physical state, frontiers, RNG and receipts; every applicable rank and the same numerical identity. |
| Comparison arms | Exact complete payload comparisons, original capacity/lazy growth/defaults, session/cache guards, shutdown/failure order and callback boundaries. |
| Interleaved A/B arms | Whole-workload TTFT and throughput, unchanged workload/assets/state/capacity; independent medians and no slowdown. Record kernel and host IOMMU policy. |

Historical source-admission gaps and prior failed/incomplete qualification remain
open; this owner adoption does not refresh their receipts or protected rows.
