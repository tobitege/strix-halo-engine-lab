# DS41 named-session checkpoint IO

The production host DS41 named-session path in
[`deepseek41_session_cache.cpp`](../../src/deepseek41_session_cache.cpp) uses
[`checkpoint_io`](../../include/strix/session/checkpoint_io.hpp) for admitted
buffered reads, durable atomic publication and persistent host locks. The shared
core owns temporary creation, full writes, file sync, replacement and parent
sync. DS41 owns codecs, compatibility, RAM retention, disk quotas, rank agreement
and group validation.

Published names and serialized bytes retain the existing formats:

* `ID.GENERATION.rankN.state` is the immutable rank payload.
* `ID.rankN.group` publishes the validated two-rank group.
* `.session-rankN.lock` is a persistent lock file, acquired in rank-0 then rank-1
  order. Each rank needs its own lock even in a shared directory.

The shared default `.session.lock` behavior is unchanged. Explicit private lock
basenames must begin with a dot and contain only ASCII letters, digits, dots,
hyphens or underscores; traversal and trailing dots are refused. Named locks
retain the legacy Windows byte range `[0,1)`. Cleanup never unlinks lock files.

Both historical `.state.tmp.` / `.group.tmp.` names and shared `.tmp-` names
are charged and removed only for the validated owning ID and rank. Symlinks and
malformed lookalikes are excluded from family cleanup. Existing peak-space
admission protects the current ID's committed generation: replacement refuses
when old plus new cannot fit the unchanged quota.

Both payloads must succeed before group publication. A failed manifest vote
attempts to restore each exact previous manifest, or remove a fresh manifest,
under the rank locks using shared IO. Peers also agree on rollback success.
Rollback failures are reported; mixed groups fail restore validation. A parent
sync failure after replacement is an uncertain durable commit. Successful
recovery after clearing a test fault proves bounded retry behavior, not recovery
from persistent media failure or cross-machine power-loss atomicity.

Host tests exercise production Sessions, paired failures, fresh-process restart,
external legacy locks, quotas and exact entry-recorded payload/group bytes:

```sh
cmake --build BUILD --config Debug --target strix-deepseek41-session-cache-tests session_checkpoint_io_tests
ctest --test-dir BUILD -C Debug --no-tests=error --output-on-failure -R '^(strix-deepseek41-session-cache-(tests|restart-tests)|session_checkpoint_io)$'
```

Repeat with Release and active assertions. Optional independently retained entry
files can be supplied with `STRIX_DS41_CHECKPOINT_REFERENCE_DIR` at configure
time. Fixed entry hashes remain checked without that directory. Platform-specific
checks run only where supported; host codec evidence does not qualify native
model execution, both-rank fidelity, capacity or performance.

This writer adoption leaves DS41's RAM owner, checkpoint representation and
request/slot/cache migration pending. CUDA writers, GLM adoption and the remaining
family migrations are separate work. Shared Engine, RowHints, ParallelJob and
RecurrentStateOps contracts are unchanged; components without adopters remain held.
