# Gemma4 exact input continuation

`gemma4::engine::SessionEngine` supports an input-only lane mode on the real
`HostBackend` under its existing `g4-v1` arithmetic, with no configured drafter
factory. A separate default-OFF native library activation is described below.
This is a family prerequisite for shared fanout. It does not
implement a full fanout request, physical prefix COW, a portable model backend,
or native qualification. The [shared contract](parallel-job.md) remains the
owner of frontier, handles, row hints and job values.

## Admission and invocation

Acquire a fresh handle from `SessionEngine::store()`, then call the shared
`session::Engine::prefill(ids, handle)`. A nonempty initial call explicitly
admits input mode. No legacy `Request`, sentinel ID, chat request or sampler is
needed. Use the existing loaded manifest and rotary tables to construct
`HostBackend`; those assets and the backend must outlive the engine. For example,
inside a host application with loaded `manifest` and `tables`:

```cpp
using namespace strix;
gemma4::engine::HostBackend backend(manifest, tables, 1);
std::mutex owner_mutex;
gemma4::engine::SessionEngine engine(backend, {}, {}, owner_mutex, 2, 0, 0);
auto lane = engine.store().acquire();
session::Engine& shared = engine;
shared.prefill(prefix_ids, lane);
session::SamplerState policy;
policy.parameters.seeded = true;
policy.parameters.seed = 17;
policy.rng_draws = 0;
policy.max_tokens = 8;
auto first = shared.decode(lane, policy);
policy.rng_draws = first.rng_draws;
shared.append_external(suffix_ids, lane);
auto next = shared.decode(lane, policy);
engine.store().release(lane);
```

Check terminal results before continuing. A terminal lane requires
`resume_lane` before input or decode. Empty input validates the handle/mode and
terminal state, then does nothing, including preserving pending output. Empty
prefill on a fresh handle leaves it unstarted. `append_external` requires an
already admitted input lane. Existing `prepare/start/advance` requests keep
their legacy final-prompt-anchor behavior; they cannot implicitly convert to
input mode. Their HTTP, SSE and receipt formats are unchanged.

An unsupported backend identity or any configured drafter factory refuses input
mode before stream creation or physical writes. This includes a configured
factory whose current request would otherwise disable drafting. Native callers
must explicitly activate the bounded exact route below. No draft state is discarded to admit input. Invalid IDs,
context overflow, unsupported constraints/min-p, nondefault row hints, stale,
foreign, duplicate, poisoned or in-flight handles are rejected before work.
Calls and store operations remain externally serialized per engine. Reentrant
family mutation is refused; asynchronous cancellation only signals a stop
source and joins the synchronous call before append/resume.
The configured slot count must fit the existing one-to-eight-lane planner.

## State transitions and ownership

```text
KvStore owns generation-checked lease + consumed IDs + optional pending ID
  -> SessionEngine owns admission/policy/cancellation + metadata synchronization
    -> Coordinator owns the SAME slot: stream, history, output IDs, RNG, logits
      -> PlanBackend executes existing ForwardPlan -> loaded HostTarget
        -> HostStream transaction commits contiguous attention/sliding storage

fresh --prefill(P)--> consumed P, pending none, retained final output
  --decode--> consumed P, pending selected output, one generated ID
  --decode--> consume pending once, select next, retain new pending
  --append(S)--> consume pending once + all S, pending none, retained final output
active --stop/allowance/context/cancel--> terminal (history/pending preserved)
terminal --resume--> active (same seed/cursors/history; fresh cancellation source)
```

Prefill evaluates every input ID, including the final one, exactly once on
success. Its frontier is `{input_length, nullopt}` on a fresh lane. It retains
the final capped logits, hidden row and taps, without output selection or target
RNG draws. The first decode initializes the existing family sampler from an
explicit seeded policy and cursor. It selects from retained logits with zero
new forward rows. Subsequent decode evaluates the pending anchor exactly once.
There is no prefix replay, next-tile requirement or duplicate final evaluation.

The canonical consumed ledger follows actual interleaved causal IDs. Generated
output IDs remain a separate ledger: a published pending output stays in it
once when append consumes that ID as input. External suffix IDs never enter
generated accounting or draw RNG. History, full snapshots and live receipt
digests use the canonical physical frontier. Retained logits are available as
a borrowed `SessionEngine::retained_logits(handle)` span until the next family
call; the span is empty after selection.

The caller supplies a cumulative cursor and a per-call output allowance. The
first decode establishes temperature, top-p, top-k and seed; later calls must
match them and the returned cursor. Resume preserves those values. Stop IDs
must match during an active turn and may change after explicit resume. The new-turn
stop policy is established by the first decode after resume, including when
bounded input advances and full snapshot restores precede that decode. A zero
allowance returns length without selecting, evaluating or drawing. A stop
output is included once and remains pending; no post-stop output is selected.
Context counts consumed plus pending plus external suffix, independently of
generated counts. Context exhaustion returns an empty context result. A
terminal result preserves any pending anchor. Resume also requires the next
decode to use a fresh, unstopped cancellation token when a prior cancellable
source existed; an unbound token is allowed when no prior source existed.

`advance` can mix bounded input suffixes on already admitted input lanes,
continuation decode, and unrelated legacy requests. One call uses one grouped
forward over ready rows, at most eight rows total. Retained-only selection needs
no forward. A long legacy prompt in this mixed call also contributes rows to
that single bounded forward; it cannot run a separate many-row prefill first.
Legacy-only calls keep their original prefill scheduling. Each input lane admits
a chunk or selects output, never both in the same advance. Carry forward only
the suffix after `input_tokens`; a still
queued suffix must match exactly. Its pending anchor is consumed first and is
excluded from suffix input counts. Finish that suffix before scalar append or
decode; cancellation preserves it across explicit resume. No scalar-loop
implementation of a batch API is supplied.

## Rollback and effective capabilities

Before input writes, metadata and final-output buffers are reserved and a
complete physical/logical checkpoint is copied. Host snapshots independently
copy global KV, sliding rings and tags, preserving the existing contiguous KV
views and arithmetic/reduction order. Slot snapshots also include hidden/taps,
logits, canonical history, output IDs, pending ID and sampler/policy state.
These are full rollback copies, with no page-sharing claim.

A scalar input call restores its entire original boundary even if an earlier
chunk committed before a later evaluation/publication failed. Grouped work
protects every attempted continuation lane. A failed grouped call returns
lane-local errors after restoring those lanes; any unrelated legacy output
already committed remains in its result and is publishable. A recoverable
failure leaves the restored continuation usable. If full restoration fails,
only affected handles are invalidated and must be released. Reset/release
reclaims their physical slots; stale generations remain rejected. Full
store-issued resume snapshots preserve pending outputs, policy initialization
and cancellation state. Partial physical truncation remains unsupported.

| Capability | Effective support |
|---|---|
| Scalar exact input / retained-output decode | Host `g4-v1`, no configured drafter; input mode only |
| `external_append`, `lane_resume` | Same configured support; active input mode / explicit terminal resume |
| `mixed_advance`, lane cancellation | Existing planner, including supported input-mode lanes |
| Temperature, top-p, top-k, seed | Existing sampler; explicit seeded policy for input mode |
| Speculative capability | Configured drafter factory; legacy path only |
| Prompt prefix cache | Existing reusable-arithmetic/chunk admission only; false on the plain HostBackend; full resume snapshots remain available |
| `batch_prefill`, `batch_decode`, `request_lane_batch`, row hints | False; optional grouped APIs refuse, including empty batches |
| `fork_lanes`, `shared_prefix_cow` | False on both backends |
| Native input continuation / terminal commit | Library-only exact route below; default OFF, device qualification pending |
| Constraints, min-p, new selection policies | Unsupported on input mode |

## Host proofs and remaining gates

The nested `gemma4-session-continuation` test uses actual tiny loaded model
weights with `HostBackend` and direct serial `HostTarget` evaluation as its
causal reference. Always-active assertions compare complete IDs, final logits,
canonical physical digests and RNG cursors for lengths 1, 7/8/9, 15/16/17 and 25,
including sliding wrap, sampled/greedy decode and alternating input/output.
Counting and fault wrappers observe real target calls and inject evaluation,
post-publication and rollback-copy failures. They also check quiescence,
duplicate/stale/foreign/poisoned admission, stop/zero/context boundaries,
cancellation/resume, grouped forward shape and preservation of an unrelated
committed output. Frozen HTTP/receipt fixtures are neither regenerated nor
modified. See the [nested test guide](../../tests/gemma4_session/README.md).

Physical native COW still requires attention owners, page addressing/write
barriers and bindings compatible with contiguous consumers, rather than eager
prefix copies. Widening native support still needs complete target/drafter
snapshot and continuation contracts for each additional identity, capacity admission and
failure cleanup. Required gates remain compiler/link ON/OFF, complete native
tokens/logits/physical-state/cursor/receipt equality against independent
references, mixed/speculative/stop/cancel/failure cases, real COW isolation,
unchanged capacity/default decisions and interleaved complete-workload timing.
Host tests establish none of those native or performance gates.

## Explicit final-turn commit and reuse producer

`SessionEngine::commit_pending(KvHandle&)` implements the shared terminal
operation on HostBackend `g4-v1` with **no configured drafter factory**, for
both legacy `Request` and input lanes. Check `caps().turn_commit`; ordinary
Host prefix-cache admission remains false. Other identities, default native backends
and every configured drafter remain unsupported, even if drafting is disabled
for the request or no pending token exists. The production Api can activate
these hooks through the explicit library configuration below. Legacy/default service callers leave this option OFF.

Call it after a successful, quiescent stop, length or context result, without
`resume_lane`. It evaluates the existing published pending ID once at its
already admitted position, including the last context position. It selects no
output and draws no RNG. Consumed advances by one exactly when pending existed;
pending becomes empty. Generated IDs, terminal status, selection cursors and
already delivered accounting remain intact. A zero-output boundary with no
pending ID seals without target work. A second supported sealed commit does
nothing. Cancelled, failed, stale, foreign, poisoned, in-flight or unfinished
input lanes refuse before work. Sealing grants no input/decode permission;
input lanes still require explicit resume, which preserves their old sampler.

```text
successful terminal --commit_pending--> sealed terminal, pending none
sealed input --resume_lane--> active input, same sampler/history
sealed terminal --turn_cache_candidate + capture_prompt--> immutable turn state
fresh prepared NEW request --restore_prompt--> reused history, NEW sampler
```

For example, after receiving the terminal result on `lane`:

```cpp
shared.commit_pending(lane);
if (engine.turn_cache_candidate(lane)) {
    auto turn = engine.store().capture_prompt(lane);
    engine.store().release(lane);
    auto next = engine.store().acquire();
    // next_request.prompt must extend turn.tokens with a nonempty new suffix.
    // Sampling seed, initial cursor and output allowance belong to next_request.
    engine.prepare(next, next_request);
    if (engine.store().restore_prompt(next, turn)) {
        engine.start(next); // already started by restoration; no additional work
        // Use the existing advance planner with prompt.subspan(turn.consumed).
    }
}
```

The complete-turn candidate has a distinct family purpose from chunk/prefill
candidates and from full resume checkpoints. Its tokens include every actually
evaluated answer/thinking ID and delimiter; an unevaluated stop/limit output is
never called consumed. It preserves the full physical stream, sliding rings
and tags plus final hidden/logits/taps. No hybrid truncation or token-count
based payload conversion occurs. The store's existing sealed provenance still
binds interpretation; cross-store reuse is not newly authorized. Exact-length
requests, mismatching history/identity and unsupported draft configurations
safely decline before stream installation. A hit evaluates only the new suffix
and subsequent generated anchors. It never installs old generated counts,
terminal status, policy or RNG in the new request. Captured owners remain
immutable after source release and later slot reuse.

Commit reserves metadata and final-output buffers before evaluation. The same
complete Coordinator/HostStream snapshots protect physical state, canonical
history, outputs, pending, policy/RNG/status and adapter metadata through logical
publication. Evaluation, post-publication or capture errors cannot lose or
reemit delivered output. Restoration failure quarantines only that handle;
release/reuse and unrelated lanes remain available. Failed optional candidate
creation preserves a previously prepared candidate and the sealed output.

`StreamState::retained_bytes()` is optional and unsupported by default.
HostStream reports its object/implementation, copied model-layer descriptors,
cache descriptors, actual K/V vector capacities and tag capacities. Complete
slot accounting adds its fixed metadata/policy, request/stop/generated/history/
queued-input capacities, hidden/logits/taps, dynamically allocated strings and
adapter checkpoint metadata. Queries measure the **captured copy**, whose
vector copy allocation may differ from the live stream, with checked addition
and multiplication. Turn candidates count their full stream, token and final
output capacities, identity and object. Model assets are borrowed and must
outlive the engine; they are not cloned into snapshots. Shared snapshot/token
index envelopes are charged by their shared owner separately. Allocator and
shared-pointer bookkeeping, fragmentation and process RSS are outside this
object/storage byte measure; it is not a process RAM bound.

Unknown/drafter payloads cannot produce complete-turn candidates. Explicitly activated exact native
streams now declare complete retained payloads at quiescent boundaries; busy,
poisoned and derived FAST owners remain unknown. Native cache work additionally
requires the narrow resolved profile below. Existing legacy cache paths keep their historical
estimate-only accounting when a complete query is unavailable, preserving frozen
replays; that estimate must not be used as a complete-turn RAM bound. Host
snapshots are honest full copies, never physical COW. Native completed-turn retention requires the separate bounded library admission
below. COW/fork, request-lane batch and configured drafters remain unsupported.

`gemma4-session-turn-commit` shares
the tiny loaded-model fixture and independent serial causal reference with the
continuation tests. Always-active checks exercise both paths, sampled/greedy,
stop/limit/context and zero-output boundaries, eight-row edges/sliding wrap,
thinking IDs, full history/logits/digests/cursors, one-time pending evaluation,
new-request suffix-only reuse/fresh sampling, retained allocation growth,
refusals, evaluation/publication/rollback failures and peer isolation. Frozen
HTTP/SSE/receipt/row traces and prior continuation/mixed assertions remain active.
Use the commands in the [nested test guide](../../tests/gemma4_session/README.md#final-turn-host-tests).
No portable model/backend, native fidelity, capacity or performance qualification
follows from these host proofs.


## Opt-in Api completed-turn retention

`Api` has a trailing `TurnCacheConfig` argument. Existing callers, including
service/legacy entry points, retain their defaults. `STRIX_G4_SERVE_PREFIX`
does not activate completed-turn retention. The same `GenerationQueue`,
`SessionRuntime`, `KvStore`, Coordinator slot and loaded HostTarget execute the
request, its thinking-budget phases, final commit and next-turn restoration.
There is no second model, sampler, cache or request loop.

An embedding caller supplies authenticated, **owning** metadata per request:

```cpp
engine::TurnCacheConfig cache;
cache.policy.enabled = true;
cache.policy.completed_turns = true;
cache.policy.max_entries = 4;
cache.policy.max_bytes = 64ULL << 20; // strict retained cache charge
cache.max_state_bytes = 8ULL << 20; // complete per-entry payload ceiling
cache.max_transient_bytes = 128ULL << 20; // separate cache-work copy/work budget
cache.request = [](const serve::HttpRequest& request,
                   const engine::ChatRequest& chat)
    -> std::optional<engine::TurnCacheRequest> {
    // Resolve trusted ownership in the embedding application. Return nullopt
    // when it cannot be established; never infer it from family session text.
    // Bind tokenizer asset/version, template and every rendering option in
    // template_id; topology describes the ordered single host rank.
    return application_turn_policy(request, chat);
};
engine::Api api(backend, tokenizer, {}, engine::DraftMode::off, model_name,
                defaults, deployment, engine::Api::Owner::standalone, cache);
```

`application_turn_policy` is supplied by the embedding application, not an
engine symbol. It returns `TurnCacheRequest{{conversation, template_id,
topology, optional_keep}, retention}` using `session::TurnMetadata` and the
existing `serve::CacheRetention`. Use `no_store` to allow lookup without new
retention. Stable tokenizer asset IDs and rendering/template IDs are **trusted
configuration**; the Api does not automatically hash the tokenizer or renderer.
Changing any relevant asset or template/rendering setting requires a different
ID. The callback receives the typed request so the caller can include thinking,
conversation and tool rendering settings. It runs once before submission;
metadata is copied/frozen after `opened`, including across thinking phases.
Callbacks can run in concurrent HTTP handlers; their application state must
be synchronized.
Unknown metadata or callback exceptions decline caching without losing replies.
No HTTP fields, global conversation keys or session-string inference are added.

Model/arithmetic identity comes from the actual backend identity (model/rotary
assets, arithmetic and effective configuration), plus the complete-turn state
schema. Missing model/rotary/backend identity declines. This mode supports `host-reference` / `g4-v1` and the bounded exact native
profile below, without any configured drafter. Native input activation alone
never enables completed-turn retention. Backend/assets must remain immutable for the Api
lifetime; a changed backend identity is declined before lookup and sealing.
Tokens must extend the complete **actually evaluated** stored prefix. Text
that renders or tokenizes differently, including thinking history removed by
the renderer, safely misses; visible-text similarity never authorizes reuse.
The host integration confirms that both returned reasoning fields and raw
assistant thinking text lose the old thinking block under the current renderer.
Those two-turn histories miss; no renderer change or thinking-hit claim is made.

ChatJob freezes terminal admission facts before `take()` moves the result.
Continuation runs before final admission. The pre-commit hook requires a
successful final stop/length boundary, with cancellation/disconnect, no-store,
incomplete thinking, unfinished tool/control content and configured drafters
all excluded.
The current parser does not expose an empty open reasoning channel. Therefore
control-bearing generated content is conservatively declined, except whole
EOS/turn terminal markers. A successful budget-answer continuation with ordinary
final output can retain its closed, actually evaluated thinking prompt. Complete
tool calls and other generated control frames are also conservatively declined.
This restriction affects retention only, never delivered output or continuation.
The sealed-only `turn_cache_candidate` runs in `turn_snapshot`, **after**
`commit_pending` and frontier validation, then `capture_prompt` supplies the
store-sealed immutable candidate. It is never called by `turn_checkpoint`.

Admission precedes the final target call and any eager rollback/candidate copy.
A complete live `retained_bytes` query includes stream and Slot capacities,
request/history/output/string storage and retained logits/hidden/taps. Checked
geometry adds conservative future history, whole global-vector growth and final
outputs. Unknown complete payloads decline. The bound deliberately overcounts
the smaller prompt-purpose payload. It is checked against `max_state_bytes`
before sealing; shared policy checks this declared ceiling against finite RAM,
entry, per-conversation replacement and named-keep limits before model work.

A separate pre-lookup/pre-seal allocation check requires
`max_transient_bytes >= (slots + 4) * max_state_bytes + work_bound`, with checked
arithmetic. It covers eager candidate/rollback/restore installation and index
staging copies for all configured slots; the work bound covers a one-row
HostTarget call, transaction descriptors, final logits/hidden/taps, norm/QKV,
attention scores, PLE/FFN and linear scratch with a conservative 64-copy envelope
for up to 16 host workers and vector growth. Bounds assume the existing host
MSVC/libstdc++ vector implementations' at-most-twofold growth; other standard
libraries decline this mode pending review. This host matrix validates MSVC,
not an additional compiler/platform. The retained cache policy
budget is separate from this additional work budget and normal live model/slot
storage. These are object/backing-allocation bounds, not allocator fragmentation,
thread stacks, process RSS or an OS reservation. The embedding application must
provision normal service memory plus both finite budgets. Full host snapshots
remain eager copies; physical COW is not claimed.

Reply receipts and stats freeze before optional finalization. Its target row,
copy/capture failure or quarantine never changes generated IDs, RNG, finish or
the delivered receipt. On a hit, existing physical `target_calls`/`target_rows`,
`cached_tokens`, `serve_prefix` counters and batch scheduling step counts reflect
skipped work; ordinary timing fields and request IDs/timestamps remain variable.
These are the explicitly compared cache-work statistics. All other serialized
HTTP/SSE payload/finish/receipt fields match independent cold execution in the
host integration. Default frozen replay fixtures are unchanged.

`gemma4-session-turn-cache` exercises the actual Api with loaded tiny host
weights, the real tokenizer/renderer, queue and runtime. It compares complete
IDs, per-row raw logits, physical digests, RNG cursors, suffix-only rows and
one-time final pending evaluation against the existing independent HostTarget
reference. It generates fixtures only in its supplied binary-tree directory.
See the [test guide](../../tests/gemma4_session/README.md#api-completed-turn-integration)
for the host matrix. Native compiler/fidelity, complete-workload performance,
capacity, physical COW/fork and batch qualification remain open.

## Explicit native library continuation (qualification pending)

The existing native factory has a trailing typed option, defaulting to OFF:

```cpp
auto backend = engine::make_device_backend(
    manifest, tables, false, engine::NativeInputContinuation::exact_g4_v1);
std::mutex owner_mutex;
engine::SessionEngine session(*backend, {}, {}, owner_mutex, 2, 0, 0);
```

No Runtime option, environment activation, HTTP switch, new request loop,
model or sampler is introduced. Existing Runtime/service/legacy calls omit
this option and keep input/terminal-commit support OFF. Ordinary request
admission, many-row boundaries, batch scheduling, resolver/inspection exports
and configured drafters retain their existing routes. Only explicit callers
receive the `native_input_continuation` identity record. Inspection never
creates an executor or advertises effective input support.

Admission requires both model and resolved executor identity `g4-v1`, exact
decode attention, active exact prefill linears, F32 KV and no EXL3 tensors.
A8, WMMA prefill, changed attention/core order, changed model identities and
invalid typed options refuse before device allocation. Actual executor stats
must also confirm exact arithmetic before the capability is exposed. Arbitrary
identity text cannot override this route check. Any configured drafter factory
still disables input continuation and terminal commit at SessionEngine admission,
including when a particular request disables drafting.

Input/append/resume and terminal sealing use the same Coordinator and existing
ForwardPlan/DeviceBackend/DeviceTarget. Teacher forcing consumes every input ID,
including the last, with zero generated outputs and RNG draws. Mixed input and
legacy work uses one existing grouped forward, at most eight rows per cycle;
there is no hidden scalar device loop. Retained-logit selection always reads
that lane's retained logits, including when another lane returns device winners.
The last published terminal anchor is evaluated once with no output selection
or RNG; repeated sealing preserves generated IDs, terminal status and cursors.

Rollback snapshots eagerly allocate the original lane's admitted capacity.
Global committed K/V/tags and entire sliding rings/tags are copied; unused
global rows remain outside the committed state. No page COW or fork capability
is claimed. Busy/poisoned sources refuse copying. Copies use the producer queue
and finish before source release/reuse. A failed enqueue drains earlier copies
before destroying their destination. A completed clone retains no borrowed
producer queue; its next executor supplies a writer queue. Publication failures
drain staging readers and poison the affected stream. SessionEngine restores
complete snapshots or quarantines handles when restoration fails. These are
host lifecycle changes only: no kernel, launch arithmetic or row layout changes.
There is no added prefix copy or device-wide synchronization in ordinary decode.

Native completed-turn caching is implemented through the existing Api ->
SessionRuntime -> SessionEngine -> Coordinator -> DeviceBackend path, with the
same explicit finite `TurnCacheConfig` and caller-owned metadata. This is an
operator/library implementation, not native qualification. Service/legacy
constructors and the native factory's default remain OFF. No HTTP option,
second cache, sampler, request loop or finalization hook was added.

The admitted native cache profile requires resolved `g4-v1`, exact attention,
F32 KV, non-EXL3 weights, no configured drafter, **graphs OFF, chunk decode OFF
and many-row prefill OFF**. Callers explicitly set `STRIX_G4_GRAPHS=0`,
`STRIX_G4_CHUNK_DECODE=0` and `STRIX_G4_PREFILL_EXACT_MANYROW=0` before backend
construction; admission never changes those settings. Actual target options
are queried, including options whose stats have not yet seen a pass. Graphs,
FAST mirrors/flash packs, A8, profiling, weight streaming/chunk private packs
and active many-row prefill have no admitted cache-work bound. They safely
decline retention without disabling their ordinary execution modes. The narrow
route has no DevicePrefill object and uses the existing eight-row target for
prompts and finalization. Its numerical kernels/launch arguments are unchanged.

`DeviceStream::memory_payload()` accounts actual retained host objects/config
and descriptor vector capacities, all admitted global K/V/tags, complete sliding
rings/tags, publication device/pinned staging and spare overlay allocations.
`DeviceState` adds its own envelope under explicit native activation; ordinary
OFF callers keep their historical unknown/estimate receipt declaration. A published transaction still holding the
stream makes the query unknown until destruction; poisoned/busy storage and
derived owners also fail closed. Borrowed model assets/queues are not snapshot
owners. Shared references do not duplicate payload charges. Slot, checkpoint,
PrefixSnapshot, history/output and final logits/hidden/taps retain the existing
complete ownership accounting. Complete captured copies are queried, rather
than reported as cache-sized estimates.

Before lookup/restore and before final sealing, the native bound enumerates
full-context stream installation and failed-install owners, constructor tag
staging, publication descriptors/pinned staging, overlay storage and canonical
digest host staging. The target query uses current owned scores/maxima/partials
capacity and the actual eight-row, 64-position attention allocation geometry.
If growth is needed, it charges the **whole new allocation** while the old
buffer remains live. Returned output vectors, pageable staging, row/transaction
maps and descriptors are separate entries. Fixed device/pinned target buffers
are startup service memory, not new cache work. `native_buffer_payload()` exposes
that existing buffer payload, not complete service host memory or driver usage.
The existing `(slots + 4) * max_state_bytes + work_bound` preflight additionally
covers bounded old candidates, rollback, candidate and index envelopes; retained
cache policy charge is a separate finite budget. Every sum/product is checked.
These bounds cover payload/backing allocations on reviewed MSVC/libstdc++ vector
implementations. They never claim allocator bookkeeping, RSS, driver-private
queue/event storage or OS reservation. Applications provision ordinary model/
service memory plus both cache budgets.

Native input activation alone cannot store turns. Unknown/zero bounds or finite
budget refusal precede eager copies and terminal target work. Optional failure
preserves delivered response/receipt and RNG. Sealing consumes the final published
token exactly once, selects nothing, and captures coherent sealed history. Hits
initialize the new request's sampler/cancellation/output state and evaluate only
its suffix. Renderer-removed thinking histories still miss. Native forks are
eager complete copies; `fork_lanes` and `shared_prefix_cow` stay false.

The operator executable and explicit asset/registration instructions are in
[the native probe guide](../../tests/gemma4_session/README.md#native-continuation-operator-probe).
CXX-only translation checks and host regressions are distinct from HIP compilation,
linking and real device execution. All native compiler/execution/fidelity,
capacity and complete-workload performance gates remain pending. Physical page COW/fork, configured drafters, native compiler/device/fidelity,
capacity and performance remain separate gates. Host accounting and wrapper
proofs do not qualify native execution.

## Shared HTTP frontend

Gemma4 `Api` composes production `serve::ApiService` with its existing
`GenerationQueue`, `SessionRuntime`, `SessionEngine` and Coordinator. The
family no longer parses HTTP envelopes, serializes chat/SSE replies or drains
tickets. Its hooks prepare one owning typed context before queue admission;
the same external job and ticket survive a thinking-budget phase restart.
The family parser publishes device-neutral reasoning/content deltas and a
frozen terminal output/receipt. The queue calls the optional
`GenerationOutput::consume_tokens` on the execution owner before waking the
HTTP drain for each published token piece; parsing therefore stays live within
one phase, rather than waiting for its terminal receipt. ApiService alone owns protocol conversion,
framing, disconnect cancellation and accepted-ticket cleanup.

`ApiTokenizer` delegates encode (including the special-token flag), decode
(including the skip-control flag), token bytes, exact lookup and vocabulary
size to the actual family tokenizer. The original `BpeTokenizer` constructor
remains available through a delegating adapter. A family passes the explicit
`ApiTokenizerBinding` value, keeping existing `{}` tokenizer construction
unambiguous for health/deployment-only callers. Gemma4 delegates to its Unicode
BPE/space/byte-fallback tokenizer and resolves EOS by exact lookup. It exposes
no token-type inventory: that optional operation explicitly refuses, and the
runtime refuses constraints before token-type metadata is needed. There is no
asset conversion or second tokenizer implementation.

The same frontend serves chat completions, text completions, Messages,
Responses, tokenize/detokenize, model/capability/health/status/metrics and
Gemma4 feature exports. Messages and Responses normalize into family chat
input before the one rendering/tokenization operation. Text completions accept
raw text or IDs and do not activate completed-turn retention. Family template
options and `dflash` retain their existing validation. `thinking_budget` is
the family two-phase policy; shared constrained-selection policies,
constraints, logprobs, penalties, min-p, request draft overrides and public
session identifiers refuse before queue/device work. Anthropic
`thinking.budget_tokens` is refused rather than accepted without enforcement.
Stop strings filter visible reasoning/content through the shared drain; token
stops remain family execution stops. Requests with host stop strings never
retain a completed-turn candidate. Feature work retains the existing backend
mutex and shared drain admission.

The opt-in completed-turn guards, finite retained/transient budgets and native
activation/accounting contracts above are unchanged. Trusted metadata is
provided before admission; wire-level no-store also prevents retention.
Native input continuation alone still enables no caching. Output and receipts
freeze before optional final-token consumption/capture, and omitted reasoning
or changed template/history safely misses. The ordinary service/legacy
activation and resolver settings stay unchanged.

The host tests exercise the actual shared frontend with loaded tiny weights,
real tokenizer/renderer/ChatJob, shared owner and independent serial target
oracles. Frozen legacy HTTP/SSE/receipt expectations remain unchanged.
Native compiler/link/device fidelity, physical RAM pressure, COW/fork and
complete-workload performance remain separate open gates. Resource backend,
recurrent/parallel/page components without production consumers are held from
landing; this frontend slice establishes no adoption claim for them or for
other model families.
