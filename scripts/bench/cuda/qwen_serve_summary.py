"""Summary of qwen_serve_abab.sh arms (OUT/<k>-<arm>/bench.txt[.ids]) written by qwen_serve_classes.py.

C1 decode means use only prompts present in BOTH passes of every arm and skip requests flagged short=1 (or with fewer
than --min-decode-tokens tokens when the line predates the flag), so a cold and a warm figure always describe the same
prompts. Prints, per (class, concurrency): cold and warm mean per arm name, warm/cold, and the ratio of every arm to the
first arm; then the ids identity count across arms. Usage: python qwen_serve_summary.py OUT [--min-decode-tokens 16]
"""
import argparse
import glob
import json
import os
import re
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--min-decode-tokens', type=int, default=16)
a = ap.parse_args()
dirs = [d for d in glob.glob(os.path.join(a.out, '*-*')) if os.path.isdir(d)]
dirs.sort(key=lambda d: int(os.path.basename(d).split('-')[0]))
c1 = {}    # (arm, cls, pas, i) -> [tps]
agg = {}   # (arm, cls, pas, C) -> [tps]
short = {}  # (cls, i) -> tokens
ids = {}
id_errors = []
for d in dirs:
    arm = os.path.basename(d).split('-', 1)[1]
    try:
        lines = open(os.path.join(d, 'bench.txt'), encoding='utf-8').read().splitlines()
    except OSError:
        lines = []
    for l in lines:
        m = re.match(r'(cold|warm) (\w+) C1 i=(\d+) .*tokens=(\w+) decode_tps=([\d.]+)', l)
        if m:
            tok = int(m[4]) if m[4].isdigit() else 0
            if tok < a.min_decode_tokens or ' short=1' in l:
                short[(m[2], int(m[3]))] = tok
                continue
            c1.setdefault((arm, m[2], m[1], int(m[3])), []).append(float(m[5]))
            continue
        m = re.match(r'(cold|warm) (\w+) (C[48]) .*agg_decode_tps_after_last_first=([\d.]+)', l)
        if m:
            agg.setdefault((arm, m[2], m[1], m[3]), []).append(float(m[4]))
    try:
        with open(os.path.join(d, 'bench.txt.ids'), encoding='utf-8') as stream:
            run = os.path.basename(d)
            seen = set()
            for line_no, l in enumerate(stream, 1):
                try:
                    r = json.loads(l)
                    key = (r['pas'], r['cls'], r['c'], r['i'])
                    if (not all(isinstance(x, str) and x for x in key[:2])
                            or not all(type(x) is int for x in key[2:])):
                        raise ValueError('invalid request key')
                    if key in seen:
                        raise ValueError('duplicate request record')
                    seen.add(key)
                    seq = r['ids']
                    valid = isinstance(seq, list) and seq and all(type(x) is int and x >= 0 for x in seq)
                    ids.setdefault(key, {})[run] = seq if valid else None
                except (ValueError, KeyError, TypeError) as error:
                    id_errors.append(f'{run}:{line_no}: {error}')
            if not seen:
                id_errors.append(f'{run}: missing nonempty request-key set')
    except (OSError, UnicodeError) as error:
        id_errors.append(f'{os.path.basename(d)}: missing/readable ID evidence: {error}')
arms = list(dict.fromkeys(os.path.basename(d).split('-', 1)[1] for d in dirs))
classes = sorted({k[1] for k in c1} | {k[1] for k in agg})
rows = {}
for cls in classes:
    keys = {(k[2], k[3]) for k in c1 if k[1] == cls}
    common = sorted(i for i in {k[1] for k in keys}
                    if all((arm, cls, p, i) in c1 for arm in arms for p in ('cold', 'warm')))
    for arm in arms:
        for p in ('cold', 'warm'):
            v = [st.mean(c1[(arm, cls, p, i)]) for i in common]
            if v:
                rows.setdefault((cls, 'C1'), {})[(arm, p)] = st.mean(v)
        for C in ('C4', 'C8'):
            for p in ('cold', 'warm'):
                if (arm, cls, p, C) in agg:
                    rows.setdefault((cls, C), {})[(arm, p)] = st.mean(agg[(arm, cls, p, C)])
    if keys:
        rows.setdefault((cls, 'C1'), {})['prompts'] = common
print('arms', ' '.join(arms), '| C1 = mean of per-prompt means over prompts in both passes; short C1 requests skipped:',
      ', '.join(f'{c} i={i} ({t} tok)' for (c, i), t in sorted(short.items())) or 'none')
hdr = ' '.join(f'{x + " " + p:>10s}' for x in arms for p in ('cold', 'warm'))
print(f'{"cls C":10s} {hdr}  ' + ' '.join(f'{x + " w/c":>8s}' for x in arms)
      + ''.join(f' {x + "/" + arms[0] + " c,w":>14s}' for x in arms[1:]))
for (cls, C), v in sorted(rows.items()):
    cells = ' '.join(f'{v[(x, p)]:10.1f}' if (x, p) in v else f'{"-":>10s}' for x in arms for p in ('cold', 'warm'))
    wc = ' '.join(f'{100 * (v[(x, "warm")] / v[(x, "cold")] - 1):+7.1f}%' if (x, 'warm') in v and (x, 'cold') in v
                  else f'{"-":>8s}' for x in arms)
    rel = ''.join(f' {100 * (v.get((x, "cold"), 0) / v[(arms[0], "cold")] - 1):+6.1f},{100 * (v.get((x, "warm"), 0) / v[(arms[0], "warm")] - 1):+6.1f}%'
                  if all((y, p) in v for y in (arms[0], x) for p in ('cold', 'warm')) else f' {"-":>14s}'
                  for x in arms[1:])
    print(f'{cls + " " + C:10s} {cells}  {wc}{rel}')
if not dirs:
    id_errors.append('missing arm runs')
missing = sum(len(dirs) - len(v) for v in ids.values())
diff = [k for k, v in ids.items() if len(v) != len(dirs)
        or len({json.dumps(x) for x in v.values()}) > 1 or any(x is None for x in v.values())]
status = 'FAIL' if diff or id_errors else 'PASS'
print(f'ids: {len(ids)} requests compared across {len(dirs)} arm runs, {len(diff)} differ or missing', diff[:6])
print(f'ids coverage: {status}; {missing} missing run/request records; {len(id_errors)} evidence errors', id_errors[:6])
raise SystemExit(1 if diff or id_errors else 0)
