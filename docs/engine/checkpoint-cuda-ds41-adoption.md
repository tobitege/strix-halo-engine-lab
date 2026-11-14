# CUDA DS41 checkpoint IO adoption

The production CUDA DS41 session store uses `strix::session::checkpoint_io` for
persistent locks, admitted buffered reads, durable file publication and immutable
directory publication. The family retains its codec, names, checksums, admission,
lineage checks, callbacks and lazy resource ownership.

A new generation contains `attention.bin`, `drafter.bin`, `frontier.bin`, then
`manifest.bin`. Its directory becomes visible without overwriting an existing
name. Only the subsequent atomic `ID.current` replacement selects it. The old
generation is removed after that replacement succeeds. The files-flushed and
pointer-flushed callbacks run before their respective visibility commits.
A failure after replacement can leave a complete, visible record with uncertain
durability; it does not promise rollback under persistent media failure.

Both `.owner.lock` (nonblocking lifetime ownership) and `.session.lock` (blocking
transactions) keep persistent lock files and the legacy Windows byte range
`[0,1)`. Only actual owner-lock contention returns `session_tier_busy`.
Shared default locks retain their broad Windows range; named rank locks retain
their existing first-byte range. POSIX uses inode-wide `flock`.

Replacement retains the existing budget and peak reservation. Interrupted
`GENERATION.tmp` directories and legacy `ID.current.tmp` files remain recognized.
Shared private files use `NAME.tmp-SEQUENCE-SEQUENCE`: payload temporaries are
inside the unpublished generation; pointer temporaries are charged alongside the
current pointer. Cleanup removes only recognized unselected generations and
pointer temporaries, preserving unrelated entries, links and lock inodes.

Ordinary unnamed requests that reload an imported launch state do no session IO
or seed copying, including when automatic policy is off. Deferred
`Disk::configure()` performs no filesystem or memory probes and allocates no
storage. Tests observe both family filesystem entry points and shared IO
activity, with a positive shared read-fault witness.

Host checks use the actual family `Disk`, publish and restore paths. An optional
`STRIX_CUDA_DS41_CHECKPOINT_REFERENCE_DIR` CMake argument enables verification of
independent entry records. The test executable also accepts `--capture DIR` for
a fresh reference directory and `--verify DIR` for decoding. References must be
captured before an implementation change and retained with their binary and
hashes; recapturing expectations is not a compatibility test.

This is durable/read/lock adoption only. CUDA arithmetic/backend work, CUDA RAM
ownership, shared checkpoint envelope/request/cache migration, CUDA Qwen prefix
storage and physical copy-on-write remain separate work. Windows host codec and
failure tests do not qualify native model fidelity, POSIX execution, capacity or
performance. No numerical identity is deliberately changed. Fingerprint and allow-table
records are not refreshed; changed source inputs require independent admission
review.
