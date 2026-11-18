#!/usr/bin/env python3
"""kvstress.py SECS CAP_GB FLOOR_GB [READ_FILE READ_GB]: own-process memory pressure for eviction tests (run only inside
my own GPU ticket): loops anon alloc (MADV_HUGEPAGE -> direct compaction) + touch + free, sized min(CAP, MemAvailable-FLOOR);
optionally streams READ_GB of a file through the page cache first (kswapd reclaim), dropped with fadvise DONTNEED at exit."""
import mmap, os, sys, time
secs, cap, floor = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
files = sys.argv[5:] if len(sys.argv) > 5 else []; chunk_gb = float(sys.argv[4]) if len(sys.argv) > 4 else 0
def avail():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable:'): return int(l.split()[1]) / 1048576
t0 = time.time(); n = 0; read = 0; fi = 0; fd = -1; touched = set()
while time.time() - t0 < secs:
    got = 0
    while files and got < chunk_gb * 2**30:  # stream the files through the page cache (kswapd + large-folio compaction)
        if fd < 0: fd = os.open(files[fi % len(files)], os.O_RDONLY); touched.add(files[fi % len(files)])
        b = os.read(fd, 64 << 20); got += len(b)
        if not b: os.close(fd); fd = -1; fi += 1
    read += got
    gb = min(cap, avail() - floor)
    if gb < 1: time.sleep(0.2); continue
    m = mmap.mmap(-1, int(gb * 2**30), flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    try: m.madvise(mmap.MADV_HUGEPAGE)
    except Exception: pass
    for i in range(0, len(m), 4096): m[i] = 1
    m.close(); n += 1
if fd >= 0: os.close(fd)
for f in touched:
    d = os.open(f, os.O_RDONLY); os.posix_fadvise(d, 0, 0, os.POSIX_FADV_DONTNEED); os.close(d)
print('kvstress loops=%d read=%.1fGB files=%d' % (n, read / 2**30, len(touched)))
