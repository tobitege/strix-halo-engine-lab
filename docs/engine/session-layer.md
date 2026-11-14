# Shared serve runtime

The [parallel job contract](parallel-job.md) freezes the PR1 job values,
immutable token references, row hints and effective work accounting. It is host
infrastructure; native fork/COW and full fanout remain unsupported. MiniMax has
a separate explicit plain-greedy input/continuation library profile described below.

Gemma4 is the first serving family using `serve::SessionRuntime`. Its Api no
longer owns a batch queue loop. Both the shared registry shell and the legacy
Gemma4 executable use the shared owner. MiniMax now has an extracted registry
service and native lifecycle. The combined source/default selector audit now
covers MiniMax family sources and the shared session contracts, queue/storage
owners, runtime implementation and session build registration. Stale source
receipt rows remain unresolved and fail closed; this audit does not qualify
numerical behavior or native execution.

The MiniMax source census includes all recursive MiniMax family sources,
MiniMax-linked MoE, drafter and transport libraries, MiniMax public headers,
shared session contracts, shared queue/storage/backend/prefix owners,
the `GenerationQueue` API and implementation, `SessionRuntime` implementation,
root CMake registration and
`cmake/StrixSession.cmake`. A newly added file in these scopes is an uncovered
source until reviewed. Shared HTTP parsing and tokenizer/registry code retain
their independent gates; Gemma4 family code and nested tests retain the
separate family gate. This census is review coverage, not device or exactness
qualification. The prior content for seven device/default/shared-core receipt
rows is identifiable in tracked history, but those files changed afterward.
The current changes have not been reconciled with their selector gates, so the
rows remain unresolved and the source gate continues to fail closed.
Host replay establishes scheduler/protocol equivalence for the synthetic cases
below; it does not establish device exactness, model fidelity or performance.

## Ownership and execution

The production call graph is:

```text
strix-serve -> install_service_factories -> Gemma4 ServiceFactory
  -> owned Runtime assets -> Api(Owner::external)
  -> shell listener + one owner thread -> SessionRuntime::run/step
    -> GenerationQueue::take -> KvStore::acquire -> typed family admission
    -> PrefixCache lookup -> KvStore::restore_prompt -> fresh sampler/drafter
    -> Engine::advance(explicit lanes) -> Coordinator::advance
      -> family chunk/draft/group planner -> session::Backend::execute(plan)
        -> existing Backend::prefill_rows[_batch] / Target::forward_rows
    -> validate/account/publish -> checkpoint -> continuation or release/finish
```

The shell owns listen, owner-thread startup, shutdown and join. Standalone Api
compatibility can own a thread, but that thread only calls the shared runtime.
External construction starts no worker. Assets outlive Api, runtime, store and
physical snapshots. Feature exports and model execution share the existing
backend mutex and external-work drain admission. Shared HTTP, queue and JSON
implementations are reused, without a second queue implementation.

`KvStore` generations bind the actual Gemma4 stream lifecycle. Release destroys
its stream and drafter; reuse receives a new checked handle. Full checkpoints
freeze physical state, request/frontier, hidden rows, sampler cursor and local
policy state. A drafter must implement the explicit complete `clone` contract;
unsupported physical/drafter clones refuse full capture. Existing native drafter
implementations are not qualified for full lane resume in this change. Gemma4
advertises neither lane fork nor shared-prefix COW: `StreamState::fork` may copy.
Truncate/append/resume remain unsupported for this adapter.

## Additive mixed-phase contract

Existing Engine prefill, decode and batch signatures remain unchanged. Optional
`Capabilities::mixed_advance` and `Engine::advance(span<AdvanceItem>)` are
unsupported by default. Gemma4 uses this contract instead of pretending that its
mixed planner is a scalar prefill/decode implementation.

Each input binds a unique generation-checked handle, the unadmitted prompt
suffix, current sampler cursor, remaining output allowance, stops and a read-only
cancellation probe. The adapter validates all lanes before work, including
suffixes, sampling policies and duplicate handles. A result reports committed
input progress, committed output IDs, absolute before/after frontiers, cumulative
RNG draws, draft counts, finish reason and a lane-local execution error.

Gemma4 retains one native grouped plan: many-row chunk boundaries, round-robin
prefill limit, prefill row budget, mixed prompt/decode rows, grouped target passes
and existing measured draft decisions. The coordinator contains compute/planning
steps only; it never takes queue jobs or owns shared cache/completion policy.
There is no eager full-prompt prefill or scalar approximation of grouped decode.
The final prompt ID remains the first decode anchor. Output allowance includes
stop and speculative bonus tokens; the family commits its verified prefix before
shared code publishes anything.

The live cancellation probe sees ticket shutdown/disconnect and typed family
cancellation during a synchronous call. The planner checks at entry and between
prefill collectives/before the grouped forward. It never interrupts a kernel or
collective. A completed commit can win a cancellation race. Shared accounting
precedes publication callbacks. Per-lane error results keep committed peer IDs
publishable; tentative/uncertain state is invalidated and released. Malformed
results fail their lane; an untrustworthy result count fails the group.

Basic host scheduling and separate-phase Engine APIs remain available to existing
consumers. Empty/invalid schedules and unsupported policies fail closed. Queue
ordering, eligibility and idle-control hooks retain the existing queue contracts.

## Typed chat phases

`GenerationJob::family_context` carries a validated internal context with a
polymorphic base. It does not enable public named sessions. Nonempty `session`
and `session_id`, unsupported draft overrides, constraints and selection fields
are still rejected before work. Gemma4's public parser retains its field whitelist.
Chat templates, tokenization, tools/reasoning parsing, dflash requests and receipts
remain family-owned.

`SessionContinuation::Mode::append` preserves the previous suffix semantics.
`restart` resets the lease, installs the family's complete newly rendered prompt,
repeats prefix lookup/admission and restores the phase's preserved RNG cursor.
Total output allowance/accounting continues across phases. Gemma4 uses restart
for its checked thinking-answer continuation; it does not append an invented
suffix or evaluate the old output twice. Per-phase prompt counts and receipts
are preserved. Request IDs and clocks are the only wire normalization in replay.

## Prefix cache safety

The actual shared PrefixCache implements Gemma4's LRU, 16 entries,
leave-one-token lookup, duplicate replacement and sole-oversize retention.
`replace_duplicates` and `retain_one_oversize` are implemented policy options.
`charge_token_bytes=false` preserves Gemma4's original physical `state_bytes`
formula: each producer contributes global committed rows or its sliding window,
multiplied by two FP32 KV widths plus a uint64 position tag per row. Public
entries/bytes/hits/misses and effective cached counts retain that contract,
including a sole entry above a zero-byte cap. No-store allows lookup but skips
capture/admission. Disabled caching performs neither.

Prompt reuse has its own sealed `SnapshotPurpose::prompt_reuse`, issued by
`KvStore::capture_prompt`. It freezes only coherent consumed physical rows and
required hidden context, excluding the former pending token and sampler cursor
by an explicit purpose-specific hook. `restore_prompt` requires a fresh lane,
validates seal/domain/ranks before erased payload access, and constructs a new
job sampler/drafter. Full resume snapshots retain pending and all cursor state.
Clearing a full snapshot's pending field or changing its purpose invalidates its
seal; it cannot manufacture a prompt checkpoint. Cross-store payloads remain
refused without matching trusted domain/schema ownership.

The cache's arithmetic policy is fixed at service startup. When prefill differs
from decode, only fully prefill-written, complete chunk checkpoints are reusable.
When they match, finished-request/decode-written consumed rows remain reusable.
Unsupported stream forks or drafter seeding decline reuse and keep cached_tokens
zero. Capture/restore failures fail closed; unproven payloads never reach hooks.
This is prompt reuse, not native COW lane-fork qualification.

## Shell, build and inspection

```sh
strix-serve --model gemma4 --inspect
strix-serve --model nemotron_3_5_bolt_e4b -- MODEL.gguf --rope-tables TABLE_DIR --tokenizer tokenizer.json
strix-serve --model gemma4 -- --help
strix-serve --model gemma4 -- MODEL.gguf --draft off --deploy-check
```

Metadata inspection loads registry/description only. Family help exits before
asset loading. Family effective-config inspection retains the existing deferred
Runtime preparation boundary. Deployment checks use the existing host planner
before Runtime construction and never create a service/owner. Profiles, stream
and draft defaults, deployment callbacks and feature-export lifetimes are shared
with the legacy main through `prepare_serve`; they are not independently parsed
and applied twice. Shell listener options and explicit forwarded family listener
options are both honored.

`strix_session_runtime` contains the owner; `strix_session_shell` contains registry
construction and explicitly references the optional service archive. Gemma4's
engine depends on the runtime, not the shell, avoiding a static-library cycle.
With the family unbuilt, metadata inspection works and execution exits 3.
`STRIX_LEGACY_MAINS` defaults ON. Direct guards in gemma4_d2.cmake gate only the
legacy serve target/probe; root CMake no longer reads/rewrites/evaluates that
module. Family libraries, shared server and tests remain available with it OFF.
`session_recurrent_tests` is independently linked to strix_session and registered
as session_recurrent with cpu;session labels; Gemma4 does not use its API.

## Host evidence and remaining gates

See [the family replay instructions](../../tests/gemma4_session/README.md) for
frozen source hashes, independent old binary capture and the comparison matrix.
The actual old Api/Coordinator was captured before production edits. The adopted
Api/runtime and mixed adapter/runtime compare against frozen expectations;
expectations cannot be generated by the migrated executable. Existing batching
and integration assertions remain intact. The generic scripted HTTP oracle is
additional shared regression evidence, not Gemma4 migration equivalence.

```sh
cmake -S tests/gemma4_session -B build/gemma4-session-p4
cmake --build build/gemma4-session-p4 --config Debug
ctest --test-dir build/gemma4-session-p4 -C Debug --output-on-failure
cmake --build build/gemma4-session-p4 --config Release
ctest --test-dir build/gemma4-session-p4 -C Release --output-on-failure
```

Host checks require no official model/tokenizer downloads. Official tokenizer
qualification remains a separately reported skip without its assets. Native
follow-up must build/link legacy ON/OFF, compare complete IDs/logits/physical
state and per-phase receipts against the original revision under the same
numerical identities, exercise every real drafter/cache failure boundary and
feature/deployment concurrency, then measure interleaved complete-workload
latency/throughput/capacity. No accelerator compilation, device execution,
performance gain or no-slowdown claim is established by these host tests.

## MiniMax extracted service

`apps/minimax_serve_main.cpp` is a compatibility entry into the same registry
factory and shell as `strix-serve`. It inserts family selection internally and
retains the family `--model-id` option. `session_service.*` validates the selected
registry description with `minimax::description`, prepares the existing CLI,
and owns the actual MinimaxApi, Observer and session/admin/metrics closures.
The archive is explicitly installed through `install_session_service` and
`strix_session_link_services`, including host builds. A linked factory reports
native execution unavailable when its native engine is unbuilt; CPU add is not
model execution.

```sh
strix-serve --model minimax --inspect
strix-serve --model minimax -- MODEL_DIR --print-effective-config
strix-serve --model minimax -- MODEL_DIR --rank 0 --transport usb4-auto
strix-serve --model minimax -- MODEL_DIR --rank 1 --transport usb4-auto
strix-minimax-serve MODEL_DIR --rank 0 --transport usb4-auto
```

Each rank retains its launch, assets and handshake requirements. Preparation
retains graph/plan/deployment inspection and refusal before service creation.
No assets are downloaded. `serve_runner.*` contains the extracted native token
exchange, command execution, follower replay and resume qualification checks.
`serve_runtime.*` owns transport, communicator, engine, drafter and named-session
runtime in dependency order. It retains deployment votes, DSpark configuration,
nested capacity, session hello/startup votes and budget/codec admission before
any listener. Session CLI and check-only outcomes use the typed startup exit;
rank 1 uses the explicit follower role and never queries the leader runtime or
creates HTTP/leader ownership.

The leader's resource owner holds `Lead<Engine,Runner,Time>` with its existing
constructor and returns `Lead::runtime()`. The shared shell starts the single
owner thread; it does not call a legacy loop. The common owner implementation
and physical adapter are in `minimax_lead.hpp` and `session_engine.*`, separate
from native resource extraction. The signal flag outlives Lead. Listener/owner
failure closes/wakes tasks and queue, stops HTTP, joins the owner and drains
handlers before service resources retire. READY uses the actual bound port.
Positive temperature retains the API's greedy execution and sampling note.

See [service host tests](../../tests/minimax_service/README.md) and the
[owner/replay tests](../../tests/minimax_session_shared/README.md). Host fakes
exercise the real factory/service/HTTP lifetime, not accelerator execution.
Combined owner replay acceptance, reviewed source receipts and both-rank native
qualification remain required before certifying migration complete. Original
raw-wire hello failures remain historical evidence; frozen fixtures are not
regenerated by the service tests. Pre-existing source-receipt debt is retained
for a separate complete selector review, without regenerating numerical tables.

The executable observer witness compiles the extracted `Borrowed` wrapper and
control exchange helper with a fake transport. It checks startup bytes including
an extra non-Control vote, ordinary-control exclusion, generated IDs, row logits
and state sizes, required-session refusal before follower work, and one-request
completion. The archived 05fedd80 layout is exercised when its object is
available; unavailable history is reported as such, not treated as equivalent.

`STRIX_LEGACY_MAINS=OFF` removes the native compatibility executable while keeping
the service archive, shared shell and host tests. With HIP OFF the native legacy
target is absent even when this option is ON. No native lane fork, shared-prefix
COW, external append, complete recurrent-state portability or performance claim
is added. The device gates above, original-vs-new complete IDs/logits/KV/draft/
RNG/frontiers/receipts, paired failures, sessions and capacity/performance checks
remain mandatory.

## Completed-turn retention

The shared owner and PrefixCache support an explicit default-OFF completed-turn
library policy with separate `turn_commit` admission, sealed prompt reuse,
conversation/template/topology metadata, finite RAM LRU and protected named
keeps. Finalization follows declined continuation and preserves the delivered
receipt on cache failure. See [the contract and host invocation](conversation-cache.md).
`turn_checkpoint` admits before commit. Optional trailing `turn_snapshot` prepares
a candidate after successful commit/frontier validation and before prompt capture;
false refuses caching and throw records a cache failure, preserving the reply.
Cancellation is checked after commit, after preparation and before admission.
Absent hooks preserve existing producers; zero retained-byte bounds refuse before
commit, while nonzero underestimates fail after capture.

The standalone shared host suite passes 11/11 in Debug/Release, legacy ON/OFF.
Central Gemma continuation, turn-commit and turn-cache tests link the production
engine. Gemma4 host completed-turn integration is implemented. Its separate family
validation is not certified by the MiniMax state tests. Host fakes qualify
shared lifecycle only; native, paired-state, fidelity, capacity and performance
gates remain pending.

## Bounded MiniMax generic state

The explicit default-OFF [paired-state library path](minimax-paired-state.md)
connects storage capture/restore and actual rank views through the neutral native
plan boundary and production follower. Its operator profile has no configured
drafter, lookup, named activity or partial/parked prefill. Eager immutable host
owners and shared sequential-rank rollback are implemented; ordinary Lead/service
construction remains OFF. Unsafe partial restore stays refused. A separately
typed default-OFF native plain-greedy R1 profile implements fully consumed input,
retained first selection, external append, mixed/batched grouped rows, cancellation,
resume and terminal-anchor sealing. It uses the same paired complete snapshots
for failure rollback and reports actual physical rows. It adds no scheduler and
keeps arbitrary sampling/drafting unsupported. No page COW, cache/turn capability
or native numerical qualification is claimed. The native operator includes bounded
same-profile teacher-forced references and post-write fault tickets; compiler/link/
device execution remain NOT RUN.

## Consolidation end state and migration order

Completion requires ONE request loop and slot owner, ONE `ApiService` implementing
`/v1/messages`, `/v1/responses` and `/tokenize` for every serving family, and ONE
cache/checkpoint/durable writer backed by pooled payloads. Two migrated launchers
do not complete this work. Extra front ends, planners, cache policies and disk
writers remain explicit migration work. The order is Gemma4, MiniMax, Qwen,
DS41, GLM; each production adopter lands with its shared contract changes.
Generic resource backends, recurrent state, parallel jobs and page implementations
remain held until production adoption. Their frozen fanout contracts are retained.

| Slice | Old implementation to remove after adoption | Compatibility gates |
|---|---|---|
| Gemma4 | Remaining request/slot policy in `src/gemma4/engine/coordinator.*`; the shared ApiService adopter has removed duplicate HTTP routing/drain, and shared RAM retention now replaces the facade's owning index. Keep family rendering/receipt and physical state adapters in `src/gemma4/engine/chat_runner.*` and `src/gemma4/kv/storage.*`; no durable writer is introduced by this RAM slice | Preserve mixed row grouping, pending anchor, sampling/draft state, exact physical snapshot ownership, tokenizer/rendering and HTTP replay; any future durable adopter requires the common envelope/writer and compatible readers |
| MiniMax | Remaining family planning in `src/minimax/serve/minimax_lead.hpp`; HTTP routing/drain already delegates to `ApiService`, and named RAM retention delegates to the same pooled owner. The checkpoint adapter retains family schema and paired manifest policy while common host IO owns lock/read/publication | Both-rank control order, prefill group alignment, decode/prefill arithmetic separation, bounded aborts, session schema/manifest/checksum and old on-disk read/write compatibility |
| Qwen | Slot/request ownership in `apps/hip_mtp_decode_bridge.hpp` and `apps/cuda_qwen_serve_main.cu`; cache policy/IO in `src/prefix_state_cache.cpp`, `apps/hip_prefix_state_cache.hpp`, `src/hip_serving_prefix_cache.hip`, `src/serial_prefix_cache.cpp`, `src/session_state_cache.cpp`, `apps/hip_session_store.hpp`, `apps/cuda_qwen_prefix_cache.hpp`, `include/strix/cuda_qwen_prefix_{store,pool,plan,identity}.hpp` | Adopt the existing pooled payload representation; preserve admitted numerical identities, rotary/profile/cache guards, boundary alignment, canonical state bytes, capacity and durable container compatibility |
| DS41 | Lead/follower request ownership in `apps/deepseek41_serve_main.cpp`; cache policy/IO in `src/deepseek41_prefix_cache.cpp`, `src/deepseek41_session_cache.cpp`, `apps/cuda_ds41_session_cache.hpp` | Both-rank scheduling/transport replay, block alignment, complete logits/state equality, RAM/disk budgets and existing checkpoint/disk-tier compatibility |
| GLM | Request ownership in `src/glm/serve/glm_serve_lead.hpp`; cache policy in `src/glm/serve/glm_serve_prefix_cache.hpp` | Preserve R1/R1b identity, prefill boundaries and speculative-oracle equality; qualify partial-prefix restore before widening whole-snapshot reuse; preserve supported on-disk schemas |

Qwen HIP HTTP request ownership is adopted for both entries: the batched entry
through [`qwen::HttpSessionOwner`](session-qwen-owner-adoption.md) and the serial
entry through [`serve::SerialSessionOwner`](session-qwen-serial-adoption.md). The
CUDA serve entry and cache-policy consolidation remain Qwen migration work.

These deletions concern migrated policy/ownership and IO, preserving adopted
code attribution and family arithmetic. Family compute adapters remain. `PrefixCache` and `NamedStateCache` now delegate RAM retention to the existing
pooled owner. Their separate owning list/map, clocks, byte sums and victim loops
are deleted; checkpoint/durable consolidation remains work.
Remove old mains after the compatibility release and successful production gates.

## MiniMax resident prefill reuse

`STRIX_MM_SERVE_PREFIX=1` requests reuse independently on each rank. Both votes
and nonzero identical `prefill_reuse_rows()` must agree; OFF, a missing vote or
alignment mismatch stays cold. The native session/input library activations
remain separate and default OFF. Normal version-3 commands keep their ordering
and 512-byte frame. Prefix negotiation uses Hello bit 2 and `dpos[0]` alignment;
`pstart` occupies the former tail padding at offset 496 only for suffix prefill.
Cold commands keep zero there. Existing additive state/input commands remain.

The shared `SessionRuntime` owns all leases and its `session::PrefixCache` queries
the ONE retained directory/LRU in `PrefixStateCache`. A resident entry has no host payload
and is not a frozen `StateSnapshot`. The store issues an immutable token certificate for its free slot, generation
and committed prefill frontier, retaining only a weak validation seal itself; loaded assets, execution profile and ordered
rank topology bind its identity. Acquiring a new lease removes the descriptor
before initialization or overwrite. Reset, poison and session/admin execution
invalidate residency before changing state. Native rank-local frontier counters
validate the physical boundary without adding another token index.

Among free slots choose the longest common prefix, cap at `prompt.size()-1`,
and round down to `max_rows * pf_group`. Both ranks really prefill the suffix at
that absolute position. Drafter-context and nested admissions stay cold. Decode
rows never extend the safe prefill frontier. Cancelled partial prefill retains
only previously reused safe rows; a poisoned/failed lane retains none. No-store
requests can use existing residency but cannot retain a new descriptor. Resident
entries share cache limits/eviction with captured checkpoints and cannot evict
protected completed-turn keeps outside the existing policy.

The host prefix target executes the production planner/runtime/adapter/control
seams with independent rank row images and counts suffix writes. Source/control
reference capture is independent of the new path. Native compilation, IDs/raw
logits/state equality per rank, pressure/capacity and interleaved complete-workload
performance remain required; host tests establish none of those native gates.


## Common RAM retention owner

The [conversation cache contract](conversation-cache.md#one-retention-budget-and-lifetime)
describes budget derivation, legacy index/oversize exemptions, strict turn quotas,
typed device snapshots, pooled encoded buffers and outstanding leases. The actual
construction graph is `SessionRuntime -> PrefixCache -> PrefixStateCache` for
Gemma4. MiniMax `ServiceAssets` creates one owner after existing RAM admission;
`make_lead` passes it to resident `SessionRuntime`, and `with_cache_owner` passes
it through the named `Runtime` into `Cache -> NamedStateCache`. There is one
directory, victim authority and ledger per service, with independent lifetimes
for independent CLI/services. Query descriptors are nonowning; retained payloads
live only in the common directory or outstanding immutable read leases.

The host-only source closure is `cmake/StrixStateCache.cmake`; session storage
has no dependency back through `strix_core` or the HTTP frontend. The real
MiniMax replay frontend links shared ApiService and its helpers for current
sources; archived original sources/fixtures keep their original closure.

The common host IO closure now owns `session_state_cache.cpp`, its existing
rotary identity dependency and `session/checkpoint_io.cpp` once, below session,
HTTP and core libraries. MiniMax envelope validation calls the shared session
checkpoint contract; this dependency must not link storage back through HTTP
or core. Qwen named-session and pooled-prefix adapters and MiniMax named-session
adapters use the common lock/read/publication mechanics. Family legacy codecs,
alignment, generations, rank agreement, manifests and retention policies remain
family contracts. Gemma4 retains typed RAM state; no durable device snapshot
writer is implied. See [checkpoint storage](checkpoint-storage.md) for adapter
boundaries and platform limitations. Host test registration is conditional on
the actual adapter test source and libraries; a missing source is pending.

This integration does not complete the common checkpoint/policy migration for
all families. The remaining Qwen, DS41 and GLM request/cache policy paths in the
migration table stay open, in Gemma4 -> MiniMax -> Qwen -> DS41 -> GLM order.
Actual checkpoint splitting/reconciliation is operator work. Resource,
recurrent, parallel and page implementations without production adopters remain
HELD; their contracts stay frozen. Host tests do not close native compilation/
link, model fidelity, unchanged capacity, pressure or complete-workload
no-slowdown gates.
