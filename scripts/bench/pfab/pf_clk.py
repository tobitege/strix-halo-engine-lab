#!/usr/bin/env python3
"""pf_clk.py RUNDIR [RUNDIR ...] : one line per prefill of each rank (r0/r1.tlog + samp0/1.txt from pf_pair_ab.sh).

Prints seconds, tok/s, busy-mean sclk, busy %, GPU clocks per token (busy x sclk x seconds / tokens), KFD evicted_ms
during the prefill, minimum MemAvailable, the logits hash, and cold (first prefill of the process) / warm (all later).
"""
import argparse
import re
import sys

LINE = re.compile(r'^(\d+\.\d+) .*pf (\S+)-ttft: (\d+) tokens ([\d.]+) s.*logits fnv (\w+)')


def samples(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) >= 7 and p[0][0].isdigit():
                    rows.append((float(p[0]), float(p[1]), float(p[2]), float(p[4]), float(p[6])))
    except OSError:
        pass
    return rows


def main(run_dirs, expected=None):
    failed = False
    for d in run_dirs:
        counts = []
        for r in (0, 1):
            s = samples(f'{d}/samp{r}.txt')
            try:
                with open(f'{d}/r{r}.tlog', errors='replace') as f:
                    found = [m for m in map(LINE.search, f.read().splitlines()) if m]
            except OSError:
                found = []
            counts.append(len(found))
            if not found or (expected is not None and len(found) != expected):
                print(f'{d} r{r}: expected {expected or "at least one"} measurements, found {len(found)}', file=sys.stderr)
                failed = True
            for i, m in enumerate(found):
                t1, arm, tok, sec, fnv = float(m.group(1)), m.group(2), int(m.group(3)), float(m.group(4)), m.group(5)
                if tok <= 0 or sec <= 0:
                    print(f'{d} r{r} #{i + 1}: invalid timing record', file=sys.stderr)
                    failed = True
                    continue
                w = [x for x in s if t1 - sec <= x[0] <= t1]
                b = [x for x in w if x[2] > 0]
                busy = sum(x[2] for x in w) / max(1, len(w)) / 100
                sclk = sum(x[1] for x in b) / max(1, len(b))
                evict = w[-1][3] - w[0][3] if len(w) > 1 else 0
                avail = min((x[4] for x in w), default=-1)
                tag = 'cold' if i == 0 else 'warm'
                print(f'{d.rstrip("/").split("/")[-1]} r{r} #{i + 1} {tag} {arm} {tok} tok {sec:.3f} s {tok / sec:.1f} tok/s '
                      f'busy {busy * 100:.1f} % sclk {sclk:.0f} MHz clocks/token {busy * sclk * 1e3 * sec / tok:.1f} k '
                      f'evict_ms {evict:.0f} avail_min {avail / 1024:.1f} GiB logits {fnv}')
        if counts[0] != counts[1]:
            print(f'{d}: rank measurement counts differ: {counts}', file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected', type=int, help='required measurement count per rank')
    parser.add_argument('run_dirs', nargs='+')
    args = parser.parse_args()
    if args.expected is not None and args.expected < 1:
        parser.error('--expected must be positive')
    sys.exit(main(args.run_dirs, args.expected))
