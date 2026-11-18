#!/usr/bin/env python3
# cached.py DIR...: page-cache resident GiB per file (mincore via ctypes, read-only, no root); top 8 + total.
import ctypes, mmap, os, sys
libc = ctypes.CDLL(None, use_errno=True); PG = mmap.PAGESIZE; out = []
libc.mmap.restype = ctypes.c_void_p; libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]; libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
for d in sys.argv[1:]:
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            try:
                n = os.path.getsize(p)
                if n < 1 << 28 or os.path.islink(p): continue
                fd = os.open(p, os.O_RDONLY); a = libc.mmap(None, n, 1, 1, fd, 0); os.close(fd)
                if a in (None, ctypes.c_void_p(-1).value): continue
                v = (ctypes.c_ubyte * ((n + PG - 1) // PG))(); libc.mincore(ctypes.c_void_p(a), n, v); libc.munmap(ctypes.c_void_p(a), n)
                out.append((sum(x & 1 for x in v) * PG, p))
            except OSError: pass
out.sort(reverse=True)
for b, p in out[:8]: print('%.1f GiB %s' % (b / 2**30, p[-90:]))
print('total %.1f GiB in %d files' % (sum(b for b, _ in out) / 2**30, len(out)))
