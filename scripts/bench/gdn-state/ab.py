#!/usr/bin/env python3
# aggregate out/sp-<arm>-<round>.txt (split.py lines) into one table: per case, state-kernel us per arm
# (median over rounds, with each round's value) and the all-kernel run time median.
import glob, re, statistics, sys
from collections import defaultdict
d = sys.argv[1] if len(sys.argv) > 1 else 'out'
state, total = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
arms, cases = [], []
for path in sorted(glob.glob(f'{d}/sp-*-*.txt'), key=lambda p: (p.split('-')[-1], p)):
    arm = re.search(r'sp-([a-z0-9]+)-(\d+)\.txt', path).group(1)
    if arm not in arms: arms.append(arm)
    for line in open(path):
        m = re.match(r'(\S+): all-kernels ([\d.]+) us/run, state kernels ([\d.]+) us/run', line)
        if not m: continue
        c = m.group(1)
        if c not in cases: cases.append(c)
        total[c][arm].append(float(m.group(2))); state[c][arm].append(float(m.group(3)))
print('case | ' + ' | '.join(f'{a} state us (rounds) / all' for a in arms) + ' | 16-bit vs f32 state')
for c in cases:
    cells, med = [], {}
    for a in arms:
        v = state[c][a]
        if not v: cells.append('-'); continue
        med[a] = statistics.median(v)
        cells.append(f"{med[a]:.1f} ({','.join(f'{x:.1f}' for x in v)}) / {statistics.median(total[c][a]):.0f}")
    rel = ' '.join(f'{a} {100.0 * (med[a] / med["f32"] - 1):+.1f}%' for a in arms if a != 'f32' and a in med and med.get('f32'))
    print(f'{c} | ' + ' | '.join(cells) + f' | {rel}')
