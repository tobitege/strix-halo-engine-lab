#!/usr/bin/env python3
"""kvab.py RUNDIR...: one line per node per run: TTFTs, SMI evictions in busy windows / total (n, ms, triggers), userptr MB alive at first busy window, fnv lines."""
import glob, os, re, sys
def segs(ev):
    boot0, B = None, []
    for l in open(ev):
        if l.startswith('# start'): boot0 = float(re.search(r'boot=([\d.]+)', l).group(1))
        elif l.startswith('B '): p = l.split(); B.append((float(p[1]), int(p[2])))
    out, cur = [], None
    for t, b in B:
        if b >= 80:
            if cur and t - cur[1] <= 1.0: cur[1] = t
            else:
                if cur: out.append(cur)
                cur = [t, t]
    if cur: out.append(cur)
    return boot0, [x for x in out if x[1] - x[0] >= 3.0]
for R in sys.argv[1:]:
    arm = open(os.path.join(R, 'arm-env')).read().split() if os.path.exists(os.path.join(R, 'arm-env')) else []
    for node, sub, rank in (('c1', '', '0'), ('c2', 'c2', '1')):
        d = os.path.join(R, sub); log = os.path.join(d, 'r%s.log' % rank); evf = os.path.join(d, 'ev%s.log' % rank)
        txt = open(log, errors='replace').read() if os.path.exists(log) else ''
        tt = ' '.join(re.findall(r'TTFT ([\d.]+) s', txt)) or 'noTTFT'
        fnv = ' '.join(sorted(set(re.findall(r'fnv[ =:]+([0-9a-f]{8,16})', txt, re.I)))[:3])
        if not os.path.exists(evf): print('%s %s %s TTFT %s (no sampler)' % (os.path.basename(R), node, arm, tt)); continue
        b0, sg = segs(evf); ev = []; alive = None
        for f in glob.glob(os.path.join(d, 'mon.*')):
            pend = None
            for l in open(f, errors='replace'):
                if l.startswith('#') or l.startswith(' '): continue
                p = l.split()
                if len(p) > 6 and p[1] == 'SMI' and p[2] == '9' and pend is None: pend = (int(p[3]) * 1e-9, p[6])
                elif len(p) > 5 and p[1] == 'SMI' and p[2] == 'a' and pend and (len(p) < 7 or p[6] != 'R'):
                    ev.append((pend[0], int(p[3]) * 1e-9 - pend[0], pend[1])); pend = None
                if alive is None and sg and b0 and len(p) > 1 and re.match(r"^[0-9.]+$", p[0]) and float(p[0]) >= b0 + sg[0][0]:
                    m = re.search(r'alive_up=\d+/(\d+)MB', l); alive = m.group(1) if m else None
        inb = [e for e in ev if b0 and any(a - 0.3 <= e[0] - b0 <= b + 0.3 for a, b in sg)]
        trig = {}
        for e in inb: trig[e[2]] = trig.get(e[2], 0) + 1
        st = open(os.path.join(d, 'stress%s.txt' % rank)).read().strip()[-40:] if os.path.exists(os.path.join(d, 'stress%s.txt' % rank)) else ''
        print('%s %s %s TTFT %s | busy-win evict n=%d %.0fms trig=%s | total n=%d %.0fms | up-alive %sMB | fnv %s | %s' % (os.path.basename(R), node,
              ','.join(arm) or 'off', tt, len(inb), sum(e[1] for e in inb) * 1e3, trig, len(ev), sum(e[1] for e in ev) * 1e3, alive, fnv or '-', st))
