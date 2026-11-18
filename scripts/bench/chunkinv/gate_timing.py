#!/usr/bin/env python3
# gate_timing.py RUNDIR : one timing line per gate run: TTFT per session (rank 0, in order: first = cold, last = warm), and over the
# prefill span (end = last mon sample with busy >= 50, start = end - sum(TTFT) - 1 s per session gap - 0.6 s) per rank:
# kfd evicted_ms growth, mean busy %, mean sclk MHz (busy samples), GPU clocks per token = busy x sclk x sum(TTFT) / tokens.
import os
import re
import sys

d = sys.argv[1]
rid = os.path.basename(d.rstrip("/"))
log = open(os.path.join(d, "r0.log"), errors="replace").read()
ses = re.findall(r"^rank 0 ([A-Za-z0-9_.-]+): prompt (\d+) tokens, TTFT ([0-9.]+) s", log, re.M)
if not ses:
    print(f"{rid} no TTFT")
    sys.exit(0)
ttft = sum(float(t) for _, _, t in ses)
tok = sum(int(n) for _, n, _ in ses)
out = [f"{rid} ttft " + " ".join(f"{s}={float(t):.3f}" for s, _, t in ses)]
for r in (0, 1):
    s = []
    try:
        for line in open(os.path.join(d, f"mon{r}.txt"), errors="replace"):
            m = re.match(r"([0-9.]+) busy=(\d+) sclk=.*?(\d+)Mhz.*ev=(\d+)", line)
            if m:
                s.append((float(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))))
    except OSError:
        pass
    busy = [x for x in s if x[1] >= 50]
    if not busy:
        out.append(f"r{r} n/a")
        continue
    end = busy[-1][0]
    start = end - ttft - 1.0 * (len(ses) - 1) - 0.6
    w = [x for x in s if start <= x[0] <= end]
    if not w:
        out.append(f"r{r} n/a")
        continue
    ev = max(x[3] for x in w) - w[0][3]
    b = sum(x[1] for x in w) / len(w)
    wb = [x for x in w if x[1] >= 50] or w
    mhz = sum(x[2] for x in wb) / len(wb)
    gclk = b / 100.0 * mhz * 1e6 * ttft / tok
    out.append(f"r{r} ev {ev} busy {b:.0f}% {mhz:.0f}MHz gclk/tok {gclk / 1e6:.2f}M")
try:
    out.append("mem " + " ".join(open(os.path.join(d, "mem.txt")).read().split()))
except OSError:
    pass
print(" | ".join(out))
