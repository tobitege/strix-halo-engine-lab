#!/usr/bin/env python3
"""Summarize the projection-row diagnostic without printing its complete log."""
import argparse
import collections
import json
from pathlib import Path
import shlex

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('log', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
records = []
complete = None
for line in a.log.read_text().splitlines():
    if not line.startswith(('onfly-row-invariance ', 'onfly-row-invariance-complete ')):
        continue
    values = dict(item.split('=', 1) for item in shlex.split(line)[1:])
    for key, value in list(values.items()):
        if key not in ['site', 'mode', 'profile']:
            values[key] = float(value) if key == 'max_abs' else int(value)
    if line.startswith('onfly-row-invariance-complete '):
        complete = values
    else:
        records.append(values)
if complete is None or complete['observations'] != len(records) or len(records) != 624:
    raise RuntimeError('Incomplete row-invariance diagnostic')
groups = collections.defaultdict(list)
for r in records:
    if r['rows'] != 1:
        groups[(r['profile'], r['width'], r['mode'])].append(r)
summary = []
for key, values in sorted(groups.items()):
    s = {'profile': key[0], 'width': key[1], 'mode': key[2], 'comparisons': len(values),
         'nonbitwise': sum(not r['bitwise'] for r in values),
         'max_abs': max(r['max_abs'] for r in values),
         'max_ulp32': max(r['max_ulp32'] for r in values),
         'top1_mismatch': sum(not r['top1_agree'] for r in values)}
    summary.append(s)
    print(json.dumps(s, sort_keys=True))
a.output.write_text(json.dumps({'complete': complete, 'summary': summary, 'records': records}, indent=2) + '\n')
print(json.dumps({'complete': complete, 'rows': len(records), 'detail': str(a.output)}))
