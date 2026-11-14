# Bounded MiniMax paired state

MiniMax has an explicit, default-OFF library path from `SessionEngine` storage
hooks through neutral `SessionPlan` dispatch to the native Runner and follower.
It captures eager immutable host images of both ranks and restores through the
existing sequential-rank `KvStore` transaction. Ordinary Lead/service/legacy
construction remains OFF. No environment selector, public HTTP option, startup
vote, second queue or named cache is added.

## Opt-in consumed input and continuation

The separate typed `paired::InputProfile::plain_greedy_r1` activation admits
`SessionEngine::prefill`, `prefill_batch`, `append_external`, `advance`, greedy
`decode`/`decode_batch`, cancellation/resume and explicit terminal-anchor sealing.
Both SessionEngine and its actual native Endpoint must opt in; an identity string
or schema integer alone cannot activate it. Declared slot/context limits must
match the live loaded paired endpoint before admission. The previous boolean state-only
constructor remains supported. Ordinary service/Lead construction stays OFF.
This is a family/library path, with no new request scheduler, HTTP selector,
transport, sampler, model or recurrence hook.

This profile evaluates supplied IDs through native `Engine::step`, with one
row per ready decode lane, one input row (two when consuming a pending anchor
before its suffix) and at most eight physical rows in ONE grouped execution per
advance cycle on both ranks. Full prefill runs bounded grouped cycles, preserving
consecutive-row causal arithmetic without calling the position-zero many-row
prefill API. It consumes every supplied ID; a previous pending anchor is consumed
once before a nonempty suffix. Empty input changes nothing. Input candidates are
never published as generated tokens, and there are zero RNG draws.

At a fully consumed boundary the native returned greedy ID is retained privately
per lane. First decode delivers that ID with zero physical rows; it then becomes
pending. Later decode consumes its pending anchor in the next grouped forward.
This route needs neither old device logits nor hidden/tap import. Arbitrary
sampling/seed fields, constraints, configured drafters, lookup, named activity
and open/parked prefill refuse. The receipt identifies native plain greedy R1
teacher forcing; equivalence to legacy many-row prefill arithmetic is a separate
gate, not claimed here.

`max_tokens` is an output allowance, never a physical-row budget. Zero returns
length without input or output work. Each advance commits at most one supplied
input ID or one output ID per lane. When pending precedes a supplied suffix, that anchor and the first supplied ID
are consecutive causal rows in the same grouped execution. The caller supplies
only the still-unconsumed suffix in the next cycle. Results stay in
input order. `AdvanceResult.rows` alone accounts the whole cycle; decode results
use `StepResult.rows`. Retained-only selection reports zero, and there is no
padding. RowHints admit per-lane shapes 1 and 2, clamping larger caps down to 2. A cap
below the required shape, more than eight total physical rows, zero/invalid hints
and all invalid group items refuse before control/row work.

Successful stop/length/context and cancellation are lane-local terminal states;
explicit resume preserves history and selection state and requires the caller to
supply a fresh cancellation source. Boundary cancellation excludes only that
lane's row; a completed grouped commit wins a later race. `commit_pending` seals
a successful terminal anchor once, selecting/publishing nothing. Repeated seals
are no-ops. It refuses cancelled/active lanes and an anchor beyond physical
capacity. This helper does not enable `turn_commit` or completed-turn caching.
Fork/COW, prefix caching and completed-turn policy integration remain unsupported.

Input snapshots extend schema 2 with the retained ID and terminal lifecycle in
the previously reserved final eight header bytes. Schema 1 remains unchanged.
The fixed 144-byte header and ordinary version-3 512-byte framing stay intact;
additive command 16 binds each teacher-forced row to store/generation/slot/position.
Both ranks independently consume its token and retain the native greedy result.
Pre/post-execution votes and synchronization precede publication; broken votes,
digests or transport abort the pair. Old commands and default startup exchanges
are unchanged, and the new command is emitted only by explicit input activation.

All lane histories, results and complete old paired checkpoints are reserved
before row writes. Canonical store publication is deferred until the whole call
succeeds, including every cycle of full prefill. On a local/peer row failure or
malformed post-write result, the family imports BOTH ranks of EVERY attempted
lane's exact old snapshot. This direct full-image rollback is not partial-prefix
truncation: `partial_restore_supported` stays false, and shorter shared restores
remain refused. A failed rollback quarantines its lane; transport failure aborts
the pair. Immutable source/sibling owners and untouched bystanders survive.
Schema 2 precharges `32*S*C + 8192` bytes for bounded lane/work metadata; retained
and transient row-image charges below remain additive. These are conservative
payload bounds rather than measured allocator RSS.

The always-active continuation host target exercises the production adapter,
backend, control replay and paired Endpoint/Owner using independent rank memories
and an independent causal-row oracle. It tests retained-output overwrite,
unequal/mixed groups, empty/stop/length/context/zero budgets, strict policy/hint/
lease admission, cancellation/resume, release/reuse, complete restoration and
post-write faults on either rank. The native operator has separate `input`,
`fault0` and `fault1` tickets with an independent cold same-shape teacher-forced/
greedy reference, raw per-rank logits/rows, IDs, frontiers and retained selection.
Each executable invocation has a 120-second hard wall watchdog. These native
tickets are implemented but compiler/link/device validation is NOT RUN here.

## Supported state-only boundary

The supported profile is the plain native row planner with **no configured
drafter**, lookup index, named-session runtime, request sampler or partial/parked
prefill. Calls are serialized by one owner, with no concurrent engine readers.
The typed operator entry `native::paired_state_probe` exercises actual storage
hooks, native row export/import and the production follower instead of starting
Lead/HTTP. It requires both ranks, at least three existing physical slots, a
finite host-state budget, loaded assets and the existing engine options.
Its compatibility digest uses the existing loader/executable/effective-profile
helper; model labels are insufficient. It adds no physical slots or capacity.

| State | Treatment in this profile |
|---|---|
| Attention K/V and index-key rows | Both ranks, actual slot, all 60 layers; existing native row representation, 76,800 bytes per position per rank |
| Absolute/indexer position | Canonical consumed length and explicit native row positions; immutable position/RoPE tables remain engine assets |
| Canonical history and selected pending ID | Store-issued history plus independently tracked native producer frontiers on both ranks |
| Owner planner | Closed-prefill checkpoint/limit retained; no active or parked task is admitted |
| Slot generation/store identity | Checked owner lease; fresh rollback capture binds peer destination generation before preparation |
| Control/task cursor | Remains live and monotonic; snapshots never rewind transport rounds or task IDs |
| Logits, hidden rows, taps, routing/work tables | Next plain native row overwrites these temporaries; this profile never reads old head/hidden/tap outputs after restore |
| Lookup n-gram, draft context/counters, RNG/request policies | Absent; configured instances are outside this path |

The engine exposes the last call's logits and taps, rather than a complete
per-slot output checkpoint/import primitive. The input profile retains only the
already returned native greedy ID. Profiles that need previous logits/hidden/tap
outputs, configured drafters, request-local lookup/planner state or named activity
remain unsupported. This path does not invent missing native primitives or
attach another recurrent-state hook to this attention family.

## Capture, ownership and budgets

Capture admits a checked finite retained/transport/scratch bound before export
and image allocation, stages metadata, synchronizes producers on both ranks and
requires the same generation/history/pending frontier. An empty newly leased
lane has no causal rows; its capture supplies the rollback image for fresh restore.
Per-rank views eagerly freeze a whole pair and identify the actual requested rank
and source slot. They are immutable snapshots, not writable device views.

For `n` consumed positions and `S,C` physical slots/capacity, the conservative
base bound is `8*S*C + 4096`, transient bound is
`base + 4*(76800*n) + 32*n + 262144`, and each retained pair charges
`2*(76800*n) + 8*n + 4096`. Capture admits retained owners plus the new pair and
transient bound; restore admits retained owners plus transient work. A resource
adapter separately admits host headroom. Checked addition/multiplication reject
overflow. These are explicit payload/work bounds, not measured allocator RSS.
Owners survive source release, slot reuse, store teardown and sibling changes.
No cross-store compatibility token is issued.

## Transport and restoration

Existing command numbers, version-3 512-byte ordinary framing and startup
exchanges are unchanged. Only explicit state operations emit additive command
15: capture, prepare and install. They use a 144-byte explicit little-endian
header, bounded token/data chunks of at most 65,536 bytes and the existing SHA-256
provider for header/history/row digests. The header carries schema, ordered TP2
identity, capacity, source/destination lease, frontier and planner checkpoint.
No device pointer is serialized. Ordinary round checks plus a separate strictly
increasing state sequence reject stale/duplicate commands. Malformed identity,
length, digest or acknowledgement aborts the pair.

`KvStore` validates sealed metadata and rank order before family hooks. The family
also checks registered C++ payload ownership, bytes, complete pair sizes,
identity and destination generation. Rank 0 prepares peer rows and **all** peer
metadata before importing locally. Rank 1 installs the prepared peer payload and
requires success acknowledgement before restoration returns. No owner reader
runs between these synchronous rank calls. Native import may throw after writes:
the shared store rolls back every attempted rank, including the throwing rank,
using independently frozen old pair images. A failed rollback poisons the lane;
a broken transport aborts the pair. Source/sibling snapshots have no writable
aliases to imports.

Partial restore/truncate stays unsupported. Shared same-length truncate only
clears pending and needs no physical rewind. The state-only activation does not advertise continuation capabilities. The
separate input profile above admits external append and lane resume only for its
plain greedy route. Neither profile advertises `fork_lanes`, `shared_prefix_cow`,
`prefix_cache` or `turn_commit`. Native physical page COW and complete paired
completed-turn policy integration remain separate work.

## Evidence boundary

The always-active host test uses independently executed mutable rank resources,
production control framing/replay, the family transaction and `SessionEngine` /
`KvStore`. It covers isolation, owner lifetime, pending/canonical/planner restore,
identity/provenance/budget refusals, post-write import failures, rollback failure,
corrupt transport and quarantine. The existing frozen scheduler/HTTP replay is
separate evidence for the default OFF path.

The native probe is implemented but has not been compiled or executed here.
Both-rank compiler, full token/logit/state fidelity, native transport faults,
capacity and complete-workload qualification remain open. Host resources do not
establish native numerical equality, physical COW or performance.
