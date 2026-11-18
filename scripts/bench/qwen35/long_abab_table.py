#!/usr/bin/env python3
"""Collect long_arm.sh arms (one per GPU window) into a strict ABAB table.

  long_abab_table.py OUT_DIR ARM [ARM ...]

ARM order = run order (A1 B1 A2 B2 ...). Per arm: cold (first prefill) and warm (last prefill) seconds and tok/s from the engine's
"prefill[i/n] ... tokens in S s" lines, busy SM MHz / W / peak MiB from smi.csv, logits sha (identity between scheduling-only arms).
Pairs are formed in order (arm 2k vs arm 2k+1); medians per side are printed for cold and warm separately (never mixed).
"""
import pathlib
import re
import statistics
import sys

PF = re.compile(r'prefill\[(\d+)/(\d+)\] \S+ (\d+) tokens in ([0-9.]+) s')


def arm(out, name):
    d = pathlib.Path(out) / name
    runs = []
    log = d / 'run.log'
    if log.exists():
        for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
            m = PF.search(line)
            if m:
                runs.append((int(m.group(3)), float(m.group(4))))
    clk, pw, peak = [], [], 0.0
    smi = d / 'smi.csv'
    if smi.exists():
        for line in smi.read_text(encoding='utf-8', errors='replace').splitlines():
            f = [x.strip() for x in line.split(',')]
            try:
                c, u, m, p = (float(x) for x in f[:4])
            except ValueError:
                continue
            peak = max(peak, m)
            if u >= 50:
                clk.append(c)
                pw.append(p)
    sha = ''
    summ = d / 'summary.txt'
    if summ.exists():
        for line in summ.read_text(encoding='utf-8', errors='replace').splitlines():
            if line.startswith('logits sha256'):
                sha = line.split()[-1]
    return {
        'name': name, 'n': runs[0][0] if runs else 0,
        'cold': runs[0][1] if runs else float('nan'), 'warm': runs[-1][1] if runs else float('nan'),
        'mhz': statistics.mean(clk) if clk else 0.0, 'w': statistics.mean(pw) if pw else 0.0, 'peak': peak, 'sha': sha,
    }


def main():
    out, names = sys.argv[1], sys.argv[2:]
    arms = [arm(out, n) for n in names]
    print('| arm | tokens | cold s | cold tok/s | warm s | warm tok/s | SM MHz | W | peak MiB | logits sha |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for a in arms:
        n = a['n'] or 1
        print(f"| {a['name']} | {a['n']} | {a['cold']:.2f} | {n / a['cold']:.0f} | {a['warm']:.2f} | {n / a['warm']:.0f} | "
              f"{a['mhz']:.0f} | {a['w']:.0f} | {a['peak']:.0f} | {a['sha']} |")
    a_side, b_side = arms[0::2], arms[1::2]
    for key in ('cold', 'warm'):
        ma = statistics.median(x[key] for x in a_side) if a_side else float('nan')
        mb = statistics.median(x[key] for x in b_side) if b_side else float('nan')
        print(f"{key}: A median {ma:.2f} s, B median {mb:.2f} s, B/A {mb / ma:.4f} ({len(a_side)} A, {len(b_side)} B arms)")


if __name__ == '__main__':
    main()
