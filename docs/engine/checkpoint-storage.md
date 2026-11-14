# Session checkpoint storage

`SessionCompatibility`, `SessionFrontier`, `SessionSection` and
`SessionCheckpoint` in `include/strix/session_state_cache.hpp` describe encoded
session state. Qwen named sessions and MiniMax named sessions use this envelope.
`PrefixCacheCheckpoint` remains the common retention entry. Its immutable encoded
bytes are decoded and validated on restore; retention metadata is not an
independent restore authority. Live `StateSnapshot` ownership and provenance stay
separate. Erased pointers and device owners are never serialized.

The envelope carries the family and format version, full compatibility records,
rank/world topology, complete consumed input history and a separate optional
pending token. Emitted, accepted, predictor, drafter, control, selection and RNG
cursors, lineage, IDs, generations and ordered rank/replicated sections remain
explicit. Unrepresented nondefault fields are refused by a legacy encoder. Adding
fields to the host representation does not qualify a new resume policy.

MiniMax's `Frontier` and `Section` names alias the shared types. Its `Checkpoint`
is a legacy accessor adapter deriving from the shared envelope: `compat` refers
to `compatibility.record`, and `history` refers to `frontier.tokens`. Copy and
move rebind these references to the destination. Runtime history also lives only
in that frontier. The existing administrative integer pending-token accessor
is retained; absence still has optional semantics. MiniMax construction supplies
its historical initial pending token, lineage and world size.

Sections use either the original Qwen buffer or the original MiniMax vector;
validation refuses simultaneous populated storage. An unspecified section rank
inherits the envelope rank. Family codecs retain their stronger completeness,
ordering, vocabulary, topology and replicated-state checks.

## Existing formats

There is no new default codec or automatic file migration. Qwen retains
`strix-session-v1`, its compatibility framing, SHA-256 section/root checks and
`session.bin`. Its historical wire rank/replication fields remain unchanged.
MiniMax retains `minimax-conversation-v1`, complete compatibility text, section
order/offsets/hashes, root checksum, generation filenames and group manifest.
The segmented prefix codec retains its original headers and certificate metadata.
Current callers emit the same file bytes for the same inputs.

Each legacy decoder validates its complete envelope before returning state to a
restore caller. MiniMax retains paired receipt agreement before importing into
the unpublished spare slot. Typed restore allocations and capacity checks precede
destination writes; the active slot changes only after import agreement.
Storage code neither consumes a pending token nor performs inference.

## Common host publication

`session/checkpoint_io.hpp` and `src/session/checkpoint_io.cpp` own:

- Persistent directory locking, with historical blocking Qwen and nonblocking
  MiniMax modes. The `.session.lock` inode is never unlinked or replaced.
- Length admission before allocation, bounded reads and overflow-safe frame
  slicing. Legacy codecs own checksum and schema interpretation.
- Exclusive temporary creation in the destination directory, complete writes,
  file synchronization, atomic replacement and exception-safe temporary cleanup.
- POSIX directory synchronization and durability of newly created ancestor
  entries. Windows uses `FlushFileBuffers` and replacement with
  `MoveFileExW(MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)`.

Windows has no equivalent supported directory `FlushFileBuffers` contract here;
POSIX directory durability is a separate platform gate. A directory-sync failure
after replacement reports uncertain durability; the target still contains a
complete old or new file. Publication never implicitly acquires a directory lock,
so an adopter holding its transaction lock does not lock recursively.

Qwen's `publish_session_checkpoint` and `HttpSessionStore` lock wrapper use this
core. MiniMax's `atomic_write`, `read` and `DirectoryLock` are thin family adapters,
used by the actual Cache checkpoint/resume/erase paths. PrefixStateCache streams
its existing segment and checkpoint codecs through the same publication primitive.
Segment data is synchronized first, then segment names together, then the
checkpoint name. Existing retained segments are synchronized when required.
Direct-IO/read-ahead/chunked-verification choices retain their existing helpers;
`durable=false` and legacy prefix policy still omit durable synchronization.

MiniMax holds its directory lock through quota checks, payload publication,
receipt exchange and manifest publication. Both durable rank receipts must agree
before either rank publishes a group. If manifest agreement fails, each rank
restores its prior small manifest, or removes a newly created manifest. Rollback
IO can itself fail and is reported. Readers continue to reject mismatched groups;
this does not provide a distributed atomic commit across machine crashes.
Immutable orphan payloads remain subject to existing disk accounting/recovery.

A replacement target is protected during quota selection. If preserving the
last good generation plus the new payload cannot fit, publication refuses within
the existing quota. It does not delete that target to finance replacement. No
new durable victim index, cache or eviction policy is introduced.
Admission charges every retained or orphan payload generation, including the
target's old payload, throughout the replacement peak. A successful replacement
requires an explicitly sufficient budget; the new manifest then selects the new
generation. Ordinary LRU eviction of other IDs is unchanged. A peer with less
headroom causes both ranks to refuse before payload publication. Payload or
manifest failures likewise refuse collectively and preserve the last good group
when rollback IO succeeds.

## Qualification and migration

The [independent host project](../../tests/session_checkpoint_io/README.md)
exercises the real adopters, fixed entry-codec hashes, optional private exact-byte
fixtures, injected failures, lock contention and platform replacement. Its
Debug/Release checks stay active. Publication fault checks select the buffered
shared writer on every platform and verify the named write, sync and replacement
failures, old bytes and temporary cleanup. Linux additionally exercises its real
direct/fallback syscall seam and page residency; those cases are unavailable on
other platforms, and filesystem direct-IO availability is reported separately.
The tight-quota replacement test requires safe refusal and old-generation resume;
separate sufficient-peak tests resume the new generation. This durable-admission
contract is a deliberate change from destructive replacement, not numerical or
output equivalence evidence. Host results do not qualify native restore,
complete model state, capacity or performance.

The full migration remains Gemma4 -> MiniMax -> Qwen -> DS41 -> GLM. This slice
adopts the named-session storage paths above and pooled-prefix publication.
Remaining DS41 host/CUDA durable storage, Qwen CUDA storage and GLM policies are
future migrations. Gemma4's typed RAM snapshots do not imply a durable adopter.
Checkpoint splitting, reconciliation and rebase remain integration work.
Generic resource, recurrent, parallel and page components without production
adopters remain HELD. Package I completion, native fidelity and interleaved
complete-workload no-slowdown remain open.
