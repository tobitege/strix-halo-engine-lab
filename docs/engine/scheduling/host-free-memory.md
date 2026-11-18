# Free host memory before a load

On a unified-memory node the device allocations of a model load come out of the same RAM as the page cache.
When `MemFree` does not cover the load (the cache holds files of earlier jobs), the kernel reclaims and
compacts while the load and the first forward run. Compaction migrates pages; when it migrates a page of a
host buffer the GPU runtime has locked, the driver suspends every compute queue of the process until the
mapping is restored. The time is visible in `/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms`.

In a pair this makes the node with the full cache the slow rank: its queues stop for hundreds of
milliseconds at a time during the first seconds after the load, its GPU clock drops because the GPU is
idle, and the other rank waits. It is not a clock or power setting, and it ends once compaction settles.

## Measured (GLM-5.3 pair, 8192-token prompt, FAST profile, fresh process per run)

| Node state at launch | Prefill | Queue suspension inside the prefill |
|---|---|---|
| Both nodes with free memory | 8.92 s | 0 ms on both ranks |
| One node with the page cache full (25 GiB free of 125) | 9.54 to 10.13 s | 583 to 1165 ms on that rank, 0 on the other |

During the load itself the same node accumulated 1.4 to 6.3 s of suspension.

## What the engine does

`core/host_free_memory.hpp` holds one shared step, `prepare_free_memory(need)`, called by a loader before
its first device allocation:

- nothing when `MemFree` already covers `need`;
- otherwise worker processes populate anonymous memory until the page cache has given back the missing
  bytes, and release it together. The load then finds the memory free;
- it never takes more than `MemAvailable` minus a 4 GiB reserve, changes no system setting and needs no
  privilege. The workers are child processes, so the caller's resident size does not grow.
  A worker that exits before reporting causes the pass to fail closed; surviving workers release their
  memory and are reaped. A partial pass is not reported as successful.

The pass costs about one second per 5 GiB it has to take back, so it is **opt-in**:
`STRIX_HOST_FREE_PREPARE=1` (workers: `STRIX_HOST_FREE_PREPARE_THREADS`, default 16). Use it for a cold
timed run on a node that has been reading other files. A server that stays up does not need it: the
suspension only affects the first seconds after its load.

GLM calls the step today, with its physical load plan plus a fifth as `need` (what its load was measured
to take from `MemFree`). Arithmetic is untouched.

Status: with the switch on, the GLM memory watchdog still sees the pass (its pressure-stall guard fired
during the reclaim in the one run made), so the switch stays off until the watchdog is paused across the
pass. With the switch off the load is unchanged.

## Reading a slow rank

Before blaming the clock, compare both nodes: `MemFree` at launch, `evicted_ms` before and after the
prefill, and `pgmigrate_success` / `compact_stall` in `/proc/vmstat`. A rank with free memory shows zero in
all three deltas.
