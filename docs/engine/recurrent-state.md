# Recurrent lane state

[`recurrent_state.hpp`](../../include/strix/session/recurrent_state.hpp) defines
`RecurrentStateOps`, the six synchronous session operations `snapshot`, `fork`,
`append`, `rollback`, `densify` and `serialize`. It also provides the typed
`LaneState<Payload>` ownership utility, identity checks, absolute frontiers and
input/checkpoint validation. This is an adapter contract and an executable host
reference, not a production family implementation. Existing Engine, KvStore,
backend, numerical routes and defaults are unchanged.

## Using the session interface

A family adapter implements `RecurrentStateOps` against its existing KvStore and
backend evaluation path. Calls run serialized at a quiescent synchronization
point, outside kernel loops. The caller retains responsibility for handle
ownership, cancellation/join and the existing explicit resume/new-turn contract.
For example, with an admitted active source and compatible adapter:

```cpp
auto checkpoint = ops.snapshot(source);
auto lane = ops.fork(checkpoint);
LaneFrontier before{checkpoint.consumed, checkpoint.pending};
ops.append(lane, before, supplied_tokens);
ops.densify(lane);
ops.rollback(lane, before, checkpoint);
auto encoded = ops.serialize(checkpoint);
// Release lane through the owning KvStore when finished.
```

`snapshot` returns the existing sealed `StateSnapshot`; `fork` returns a fresh
generation-checked KvHandle. A recurrent checkpoint alone is not a resumable
engine. Every rank's opaque immutable snapshot owner must include:

- recurrent matrices and convolution/history state;
- attention pages, indexes, positions and any rollback data;
- drafter state and proposal RNG, if present;
- target RNG state/resolved seed and draw count;
- selection/constraint/policy cursors, generated/input accounting and turn status.

The snapshot also carries the consumed token history and pending anchor. Its
rank order and topology must match. Adapter hooks may embed `LaneState` inside
their typed complete rank image. Capture must freeze every coupled component;
fork/restore must independently restore all components, not just a length counter.
The host fixture does this through actual KvStore capture/fork/restore hooks.

## Ownership and compatibility

`LaneState<Payload>` has three owners: a refcounted immutable base checkpoint,
an immutable factor log (possibly empty) and an optional uniquely owned private
dense payload. The checkpoint freezes its identity, trust tokens, absolute base
frontier and payload. The lane is move-only. Metadata is copied into immutable
ownership at creation; no caller-held mutable metadata object can replace the
checkpoint payload. The typed utility does not perform erased pointer casts.

`LaneIdentity` requires version 1, schema, ordered topology, rank count/index,
model/assets identity, numerical identity and a nonempty base identifier.
`LaneTrust` additionally checks both domain and schema tokens by address AND
shared ownership, matching KvStore's trusted-token rules. Equal strings, token
histories or rank counts never grant interpretation authority. Adapters share
tokens only for identical C++ payload types and continuation semantics. No global
base or type registry is introduced. `read`, `clone`, `update` and `densify`
validate compatibility before interpreting payloads. Base/log accessors expose
only const owners; trusted adapters must validate identity before using them.

At the erased KvStore boundary, the existing store checks its sealed origin,
complete unchanged metadata, owner identities/order and rank count before calling
typed hooks. Cross-store use requires both trusted tokens. The host fixture casts
its erased Image only inside these hooks after store validation, then validates
the embedded lane identity before reading its payload. Substituting an owner or
changing metadata on a copied StateSnapshot invalidates the seal. Cache entries
freeze metadata independently of caller-held aliases.

The adapter-supplied clone must allocate physically independent payload storage,
including subresources. It must not retain writable aliases or copy descriptors
that still reference mutable shared matrices. `clone` shares base/log ownership
and clones an existing dense continuation into a unique owner. `update` prepares
a private clone, runs adapter work on it and publishes only on success. Work must
not retain a mutable reference. Snapshots/forks survive source release; their
dense continuations cannot be modified by later lane writes. This dense fallback
costs O(dense state) copying and allocation; it is not an O(1) recurrent fork.

Attention-only adapters implement the session operations using genuine KV page
COW and no recurrent matrix. Hybrid dense fallback still shares attention
prefixes using KvPages or the family's equivalent write barrier. The fixture
shares real KvPages owners, clones zero attention prefix pages at fork and
detaches only touched pages on writes. Snapshot transaction capture can introduce
new owners and legitimately cause a later write to detach again.

## Append, rollback and densify

`LaneFrontier` is absolute: `[0, consumed)` is evaluated state, and `pending`,
when present, is the admitted token at `consumed` awaiting evaluation. Append
requires an exact expected frontier, a valid live handle and an active,
quiescent lane on every rank. It validates IDs, pending ID, frontiers and logical
capacity before mutation; reserves history/result storage before backend work;
and evaluates supplied tokens through the adapter's existing arithmetic route.
KvStore's metadata append alone is not evaluation.

`validate_lane_append` supplies overflow-safe admission for the fully consumed
suffix route. Empty input preserves the pending anchor. Nonempty input consumes
the old anchor once, then the entire supplied suffix, leaving no pending token.
An adapter preserving a final pending token instead must explicitly report and
validate that route. External input increments input accounting only; it neither
selects output nor advances target-sampler RNG draws. The fixture uses a typed
integer recurrence and convolution history to prove execution, with independent
seeded output continuation checks. It is not a floating-point model reference.

Rollback specifies both an absolute frontier (including pending) and a matching
sealed complete checkpoint. KvStore then validates provenance before hooks and
restores all rank payloads and metadata. An interior frontier without saved
complete state is unsupported; truncating attention or a recurrence counter is
insufficient. Restoring a checkpoint restores RNG, selection and output cursors
as well as matrices. Normal operation must never discard a committed but
unreported output; append/densify produce no output selection.

For empty logs, densify materializes the current logical recurrent state in a
new private dense owner while preserving base, frontier and all coupled cursors.
The generic utility stages each rank's update; full-lane atomicity belongs to
the adapter transaction. The fixture captures complete state before append,
generation and densify, restoring all ranks on failure. If restoration cannot
be established, it invalidates only the affected lane. KvStore's restore failure
path attempts repair of every attempted rank, including the throwing rank, and
poisons the lane if repair fails. Failed fork releases its unpublished generation;
failed cleanup quarantines capacity for teardown. Unrelated siblings remain
usable, and poisoned published handles can be explicitly released.

## Serialization boundary

`LaneSerialization` returns either supported owning bytes or unsupported with a
reason and no bytes. Adapters without an exportable opaque/device representation
must return unsupported. The shared header supplies no generic memory dump and
no deserializer or production pair wire protocol.

The host fixture encodes a deterministic, bounded little-endian format: a magic
string and version; rank count; consumed/pending/history; each rank's full schema,
topology, model, numerical and base identities; base/current frontiers; base
matrix/conv values; empty-log length; lane-local dense values; drafter, RNG,
selection/accounting/turn cursors; and attention rows. Integers use explicit
eight-byte encoding and strings have explicit bounded lengths. Checked capacity
subtraction prevents length overflow. No pointer, native object representation,
padding or shared ownership token appears in the bytes.

The fixture validates input snapshots by a temporary KvStore fork before any
erased payload interpretation, then releases the temporary generation. This
requires a spare slot, copies dense state and may fail allocation/admission. It
is a test adapter choice, not a required production serialization algorithm.
Production adapters can provide their own trusted typed validation/export path.
Bytes alone do not authorize cross-model, cross-rank or unknown-base restore.
There is deliberately no import API; any future importer must reject malformed
lengths, incompatible identities and unknown bases before payload interpretation.

## Host proof and integration

The standalone project compiles current production `kv.cpp` and `prefix_cache.cpp`
without copied implementations. Checks throw on failure and remain active with
NDEBUG. From the repository root:

```sh
cmake -S tests/session_recurrent -B build/session-recurrent-p4
cmake --build build/session-recurrent-p4 --config Debug
ctest --test-dir build/session-recurrent-p4 -C Debug --output-on-failure
cmake --build build/session-recurrent-p4 --config Release
ctest --test-dir build/session-recurrent-p4 -C Release --output-on-failure
```

For single-configuration generators, configure separate Debug/Release children
of that build directory using `-DCMAKE_BUILD_TYPE`. Root integration can compile
`tests/session_recurrent_tests.cpp`, link `strix_session` and register CTest name
`session_recurrent`; no additional Engine or KvStore API is required.

The proof covers two-rank full-state lifetimes across source/store/cache release,
real attention page sharing/isolation, private dense continuations, teacher-forced
pending handling, seeded output continuation, complete rollback/densify equality,
deterministic serialization, incompatible/mutated metadata, stale/terminal/busy
lanes, limits, unsupported factors, injected capture/clone/allocation/evaluation/
fork failures, second-rank failures after first-rank work, failed repair quarantine
and final resource reclamation.

FactorLog is an immutable adapter-defined byte encoding. Its presence does not
advertise numerical support: the dense reference helpers explicitly throw
UnsupportedOperation on every nonempty log, including clone and densify. A future
factor adapter needs its own schema, evaluation/rollback/densification/export
implementation and admission gates. Factor math may change floating-point
operation order. Compiler, native execution, full coupled-state/token/logit
fidelity and complete-workload performance qualification remain future gates.
This host fixture grants no device exactness, speed claim or automatic numerical
opt-in. A future rebase may publish a new immutable checkpoint only at a validated
sync point; this implementation never mutates or republishes its base.
