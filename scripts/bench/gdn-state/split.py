#!/usr/bin/env python3
# split a rocprofv3 kernel trace into the bench cases (host gaps > 30 ms) and print, per case,
# per-launch median duration of the recurrent-state kernels and of all kernels (fills excluded).
import csv, re, statistics, sys

rows = []
with open(sys.argv[1], newline='') as f:
    for r in csv.DictReader(f):
        name = r.get('Kernel_Name', '')
        if re.search(r'fill|memset|Fill', name):
            continue
        rows.append((int(r['Start_Timestamp']), int(r['End_Timestamp']), name))
rows.sort()
cases = [l.split()[1] for l in open(sys.argv[2]) if l.startswith('case ')]
segs, cur, last = [], [], None
for s, e, n in rows:
    if last is not None and s - last > 30_000_000:
        segs.append(cur); cur = []
    cur.append((s, e, n)); last = e
segs.append(cur)
segs = [g for g in segs if len(g) >= 28]
segs = segs[-len(cases):] if len(segs) >= len(cases) else segs
STATE = re.compile(r'recurren|commit|segments|anyseg|step_kernel')
def short(n):
    m = re.search(r'([a-z_0-9]*(?:recurren|commit|segments|anyseg|step_kernel)[a-z_0-9]*)', n)
    t = re.search(r'I(L[^E]*E)', n)
    return (m.group(1) if m else n[:40]) + ('<' + t.group(1)[:20] + '>' if t else '')
for name, g in zip(cases, segs):
    by = {}
    for s, e, n in g:
        by.setdefault(n, []).append((e - s) / 1000.0)
    total = sum(sum(v) for v in by.values()) / 28.0
    st = [(short(n), statistics.median(v), len(v)) for n, v in by.items() if STATE.search(n)]
    stsum = sum(statistics.median(by[n]) * len(by[n]) / 28.0 for n in by if STATE.search(n))
    print(f"{name}: all-kernels {total:.1f} us/run, state kernels {stsum:.1f} us/run: " +
          "; ".join(f"{a} {b:.1f}us x{c}" for a, b, c in st)[:300])
