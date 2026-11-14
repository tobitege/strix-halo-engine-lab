# DS41 prefix checkpoint IO

DS41 `PrefixCache` block and snapshot files use the existing
`session::checkpoint_io` publication and vector reader. The production
`strix_ds41_runtime` target links the existing host-only `strix_state_cache`
library that supplies that IO. The former family platform write, read and
directory-sync implementations are removed.

The 128-byte headers, payloads, checksums, version and final filenames are
unchanged. Chain lookup, snapshot boundaries, unknown-next rules, inventory,
load/use/forget/retain, LRU, rank decisions and RAM/disk accounting remain
family-owned. `disk_bytes` and inventory continue to describe admitted index
entries; they are not durable-commit receipts. `written_files` and
`written_bytes` advance only on successful publication; `write_errors` records
failures. No new retaining cache, index, codec or writer thread is introduced.

The existing Writer still processes its FIFO jobs, including deletion jobs,
and charges the active payload until completion. `flush()` waits for its queue
and active job; it does not become a durability receipt or throw worker errors.
Errors are observable in statistics. A failed read remains a safe miss through
`PrefixCacheReadError`; the existing rank agreement and explicit forget/retry
protocol remain necessary. Allocation and filesystem exceptions in shared IO
are contained by the family byte/status adapters so they cannot terminate its
worker or escape its parallel read body.

Publication uses exclusive same-directory temporaries ending in `.tmp`, keeping
the existing quota counting and restart cleanup. It preserves legacy POSIX 0644
permissions, full writes, sync, checked close, replacement and parent sync.
The owner and quota locks stay in DS41: their nofollow, persistent-inode,
nonblocking and reservation behavior is unchanged. Unsupported quota/owner
platforms continue to refuse. Queue budgets, filesystem free-space floors,
aggregate budgets and both-rank admission are unchanged. Existing LRU eviction
of unrelated snapshots remains; this IO adapter does not delete a committed
target to finance its replacement.

`drop_page_cache=true` keeps buffered writes with post-sync page-drop advice,
and requests automatic direct reads. The vector reader tries direct IO into an
aligned window within its destination vector, safely falls back to buffered IO,
then moves the bytes to the vector front. It has no staging pool or read-ahead
thread. This explicit policy is independent of `STRIX_SESSION_DIRECT_IO`, which
did not govern DS41 prefix reads. Buffered DS41 prefix reads still advise away
their pages, including when `drop_page_cache=false`; that setting disables direct
reads and writer page-drop advice. Platforms without these POSIX facilities use
buffered IO. Existing shared callback-reader defaults remain unchanged.

Failures before replacement preserve the previous complete target and clean the
temporary. A parent-sync failure after replacement leaves the new complete bytes
visible with uncertain durability. It increments `write_errors`, not the success
counters; it cannot roll back visibility or promise power-loss atomicity.
Historically the POSIX prefix writer ignored directory-open/fsync failures and
close errors, and its non-POSIX stream path omitted durable file sync. Shared IO
checks these boundaries. Restart directory synchronization also now reports an
error rather than ignoring it. These are deliberate error-reporting and
durability changes, not equality claims under broken storage.

The [host fixture](../../tests/ds41_prefix_checkpoint_io/README.md) exercises the
actual Writer, codecs and common IO, with an independent entry executable for
file-byte and cross-reader comparisons. Host evidence does not qualify Linux
direct IO, page residency, native models or pair transport on another platform.
Strict native builds, both-rank complete-turn/state/RNG/receipt/capacity checks
and interleaved whole-workload performance remain separate operator gates.

This bounded D4 IO adoption does not unify the DS41 prefix index/LRU with the
common pooled owner. D1 request/slot migration, remaining D4 family/cache and
checkpoint migration, and GLM adoption remain outstanding. DS41 named sessions
and CUDA DS41 already use shared IO and are not migrated again here.
