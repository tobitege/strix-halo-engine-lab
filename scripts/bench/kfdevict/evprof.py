#!/usr/bin/env python3
# evprof.py EVLOG [BUCKET_S]: evictions per time bucket of an evsample log (count of EV lines, evicted_ms added),
# MemAvailable / page cache at the first EV; maxima of overlapping 2s pressure observations (NOT counter totals).
import re, sys, collections
b = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
rows = collections.OrderedDict()
for l in open(sys.argv[1], errors='replace'):
    m = re.match(r'([\d.]+) \S+ EV \d+ \S+ \+(\d+)', l)
    if not m: continue
    k = int(float(m.group(1)) // b); r = rows.setdefault(k, collections.Counter())
    r['n'] += 1; r['ms'] += int(m.group(2))
    for key in ('pgsteal_kswapd', 'pswpout', 'compact_stall', 'pgmigrate_success', 'allocstall_normal'):
        v = re.search(key + r'=(\d+)', l); r[key] = max(r[key], int(v.group(1)) if v else 0)
    if 'MA' not in r:
        ma = re.search(r'MA=(\d+)', l); c = re.search(r' C=(\d+)', l)
        r['MA'] = int(ma.group(1)) if ma else -1; r['C'] = int(c.group(1)) if c else -1
print('# Pressure: overlapping observations; maxima, not counter totals (each column independently).')
print('t_s  n  ev_ms  MA  C  kswapd_steal_2s_max  pswpout_2s_max  compact_stall_2s_max  allocstall_2s_max  migrated_2s_max')
for k, r in rows.items():
    print('%4.0f %3d %5d %6d %6d %9d %7d %5d %5d %8d' % (k * b, r['n'], r['ms'], r['MA'], r['C'], r['pgsteal_kswapd'], r['pswpout'], r['compact_stall'], r['allocstall_normal'], r['pgmigrate_success']))
