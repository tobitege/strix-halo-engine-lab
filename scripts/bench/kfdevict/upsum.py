#!/usr/bin/env python3
# upsum.py MONFILE [T0 T1]: userptr allocations grouped by (size, first caller outside hsakmt/hsa-runtime);
# with T0 T1 (sampler clock = mon clock), only allocs made inside [T0, T1] are listed (in-window transient pins).
import re, sys, collections
f = sys.argv[1]; t0 = float(sys.argv[2]) if len(sys.argv) > 3 else None; t1 = float(sys.argv[3]) if t0 is not None else None
g = collections.Counter(); tr = {}; who_of = {}; alive = {}; cur = None
def flush():
    global cur
    if cur: who_of[cur[3]] = cur[2]; g[(cur[1], cur[2])] += 1; a = tr.setdefault((cur[1], cur[2]), [cur[0], cur[0]]); a[1] = cur[0]; alive[cur[3]] = (cur[1], cur[2])
    cur = None
for l in open(f, errors='replace'):
    if l.startswith('    bt '):
        if cur and not re.search(r'libhsakmt|libhsa-runtime|kfdevmon|libamdhip|libc\.so|libstdc', l) and cur[2] == '?':
            m = re.search(r'([^/ ]+)\(([^)]*)\)', l); cur[2] = (m.group(1) + ':' + m.group(2)) if m else l.split()[1][-60:]
        continue
    flush()
    m = re.match(r'([\d.]+) ALLOC USERPTR size=(\d+) .*uaddr=(\w+)', l)
    if m and (t0 is None or t0 <= float(m.group(1)) <= t1): cur = [float(m.group(1)), int(m.group(2)), '?', m.group(3)]
    m = re.match(r'([\d.]+) FREE USERPTR size=(\d+).*uaddr=(\w+)', l)
    if m: alive.pop(m.group(3), None)
flush()
T0 = min(v[0] for v in tr.values()) if tr else 0
for (sz, who), n in sorted(g.items(), key=lambda x: -x[0][0] * x[1])[:14]: print('%5d x %10d B  t %.1f-%.1f  %s' % (n, sz, tr[(sz, who)][0] - T0, tr[(sz, who)][1] - T0, who[:100]))
print('total allocs %d, MB %.1f' % (sum(g.values()), sum(k[0] * v for k, v in g.items()) / 1e6))
# ALIVE_AT=seconds after the first alloc: userptr BOs alive at that moment (exposed to page migration during the window)
import os
if os.environ.get('ALIVE_AT'):
    ta = T0 + float(os.environ['ALIVE_AT']); live = {}
    for l in open(f, errors='replace'):
        m = re.match(r'([\d.]+) (ALLOC|FREE) USERPTR size=(\d+).*uaddr=(\w+)', l)
        if not m or float(m.group(1)) > ta: continue
        if m.group(2) == 'ALLOC': live[m.group(4)] = int(m.group(3))
        else: live.pop(m.group(4), None)
    c = collections.Counter((v, who_of.get(k, '?')[:60]) for k, v in live.items())
    print('alive at +%ss: %d BOs %.1f MB; by size %s' % (os.environ['ALIVE_AT'], len(live), sum(live.values()) / 1e6, sorted(c.items(), key=lambda x: -x[0][0] * x[1])[:10]))
