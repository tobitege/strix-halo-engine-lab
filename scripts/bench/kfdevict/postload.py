#!/usr/bin/env python3
# postload.py RUNDIR: rank-0 allocations at/after an explicit load-end CLOCK_BOOTTIME timestamp.
# No load-duration/first-allocation inference, and no inference that these allocations fall in timed prefills.
import glob, math, pathlib, re, sys
d = pathlib.Path(sys.argv[1])
try:
    end = float((d / 'load-end.boottime').read_text())
    if not math.isfinite(end) or end < 0: raise ValueError('invalid timestamp')
except (OSError, ValueError):
    print('%s: load-end boundary unknown (need load-end.boottime on the monitor CLOCK_BOOTTIME clock)' % d.name)
    sys.exit(0)
m = sorted(glob.glob(str(d / 'mon.*')))[0]
n = 0; tot = 0; big = []
for l in open(m, errors='replace'):
    x = re.match(r'([\d.]+) ALLOC USERPTR size=(\d+)', l)
    if not x: continue
    t = float(x.group(1))
    if t < end: continue
    n += 1; s = int(x.group(2)); tot += s
    if s >= 1 << 20: big.append('%.1fMB@+%.3fs' % (s / 1e6, t - end))
print('%s: load-end %.6f boot s; after load %d userptr allocs %.1f MB; >=1MB: %s' % (d.name, end, n, tot / 1e6, ' '.join(big[:12])))
