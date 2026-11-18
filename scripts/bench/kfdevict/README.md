# kfdevict: KFD queue-eviction kit (no root)

Why a GPU process loses its queues mid-run, and whether a mitigation stops it. Every tool reads only what an
unprivileged user can read.

| file | what it does | run |
|---|---|---|
| `kfdevmon.c` | `LD_PRELOAD` probe for the process it is loaded into: its own KFD SMI events (queue eviction with trigger SVM / USERPTR / TTM, restore, unmap) and every KFD allocation (userptr ones with a backtrace and the owning `/proc/self/maps` line). | `gcc -O2 -shared -fPIC -o kfdevmon.so kfdevmon.c -ldl -lpthread`; `LD_PRELOAD=./kfdevmon.so KFDEVMON_OUT=<prefix> <engine ...>` (`KFDEVMON_BT=0`: no backtraces) |
| `evsample.py` | Node sampler: every `evicted_ms` change of every KFD process with the `/proc/vmstat` deltas (reclaim, swap, compaction, migration) of the 2 s before it, `MemAvailable` / page cache / `GPUActive`, and the GPU busy timeline at 4 Hz (busy segments = timed windows). | `python3 evsample.py SECONDS OUT [PERIOD]` |
| `kvstress.py` | Controlled memory pressure from the measuring ticket itself: streams files through the page cache (kswapd + large-folio compaction) and churns `MADV_HUGEPAGE` anonymous memory sized to `MemAvailable - FLOOR`; drops the cache it read at exit. Run it only inside your own GPU ticket. | `python3 kvstress.py SECONDS ANON_GB FLOOR_GB CHUNK_GB FILE...` |
| `kvmicro.hip` | Micro A/B: holds `GTT_GB` of device memory, 850 MB of host buffers in the engine's shape, and times a ~1 ms GPU kernel reading host memory; prints stalls and its own `evicted_ms`. Modes: `none` (hipHostMalloc), `mlock`, `shmpin` (mlock + io_uring pin of ROCr's shmem host maps), `anonpin` (the `STRIX_HOST_ANON_PIN` allocator: private anonymous + MADV_DONTFORK + mlock + io_uring pin + hipHostRegister). `HSA_USERPTR_FOR_PAGED_MEM=0` in the environment tests GTT-backed host memory (O_DIRECT check included). | `hipcc --offload-arch=gfx1151 -O2 -o kvmicro kvmicro.hip -lhsa-runtime64`; `kvmicro SECS GTT_GB HOST_MB MODE [ODIRECT_FILE]` |
| `kvab.py` | One line per node per run: TTFTs, evictions inside GPU busy windows / whole run (count, ms, trigger), userptr MB alive at the first busy window, output FNV lines. | `python3 kvab.py RUNDIR...` |
| `kvwin.py` | The sampler's eviction lines that fall inside GPU busy windows, with their 2 s vmstat deltas (who moved pages). | `python3 kvwin.py EVLOG...` |

Reading: an eviction is harmless outside timed windows; inside them its restore time adds to the measurement. Trigger
USERPTR / SVM with compaction or reclaim in the 2 s before it = page migration of host memory the GPU maps.

| file | what it does | run |
|---|---|---|
| `upsum.py` | kfdevmon userptr allocations grouped by (size, first engine frame) with first/last time; `ALIVE_AT=s` lists the BOs alive at that moment (exposed to page migration during a window). Resolve frames with `addr2line -f -C -e BIN +OFF`. | `python3 upsum.py MONFILE` |
| `postload.py` | Rank-0 userptr allocations at/after an explicit load-end marker; without it the boundary is unknown. This does not identify timed prefill windows. | `python3 postload.py RUNDIR` |
| `evprof.py` | Evictions per bucket, MemAvailable / page cache at the first EV, and per-column maxima of overlapping two-second pressure observations. These are not counter totals. | `python3 evprof.py EVLOG [BUCKET_S]` |
| `cached.py` | Page-cache resident bytes per file under the given directories (mincore, read-only, no root): whose files fill the cache before a load. | `python3 cached.py DIR...` |

`postload.py` accepts `RUNDIR/load-end.boottime`: a single timestamp in seconds captured with
`clock_gettime(CLOCK_BOOTTIME)` at rank 0's load completion, on the same host/boot as its `mon.*` log.
The current engine does not emit this marker; an instrumented collector must record it at that boundary.
Do not substitute log-receipt time, load duration, or first-allocation time. Use a directory containing only
that rank's monitor log. Offsets in the output are relative to the marker, with no grace period.

The legacy sampler repeats rolling vmstat deltas across EV records and PIDs. Its totals cannot be recovered
by summing or deduplicating those observations. Counter totals require a separate timestamped cumulative
vmstat stream sampled once per interval independently of EVs; this report does not claim such totals.
