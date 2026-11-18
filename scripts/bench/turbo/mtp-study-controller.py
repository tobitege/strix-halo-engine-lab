#!/usr/bin/env python3
"""Summarize observed warm window cost; contexts differ, so this is not a causal fit."""
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import re

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('runs', nargs='+', type=Path)
p.add_argument('--out', required=True, type=Path)
a = p.parse_args()
spec = importlib.util.spec_from_file_location('study', Path(__file__).with_name('mtp-study-analyze.py'))
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
result = {}
for path in a.runs:
    run = study.load_run(path)
    last = max(request['cohort'] for request in run[0])
    groups = collections.defaultdict(collections.Counter)
    for line in (path / 'srv.err').read_text(errors='replace').splitlines():
        if not line.startswith('serve cycle '):
            continue
        fields = dict(re.findall(r'(\w+)=([^ ]+)', line))
        _, request = run[1](int(fields['begin_ns']))
        if request is None or request['cohort'] != last:
            continue
        c = groups[int(fields['rows'])]
        c['passes'] += 1
        for key in ('emitted', 'draft_ms', 'verify_ms', 'host_ms', 'ms'):
            c[key] += float(fields[key])
    pooled = sum(groups.values(), collections.Counter())
    def normalized(c):
        return dict(c, emissions_per_pass=c['emitted'] / c['passes'],
                    ms_per_pass=c['ms'] / c['passes'],
                    emitted_tok_s=1000 * c['emitted'] / c['ms'])
    result[path.name] = dict(rows={str(rows): normalized(c) for rows, c in sorted(groups.items())},
                             pooled=normalized(pooled))
a.out.write_text(json.dumps(result, indent=2) + '\n')
for name, data in result.items():
    c = data['pooled']
    print(name, 'warm', c['passes'], c['emissions_per_pass'], c['ms_per_pass'], c['emitted_tok_s'])
    for rows, c in data['rows'].items():
        print(' rows', rows, 'n', c['passes'], 'E', c['emissions_per_pass'], 'ms', c['ms_per_pass'], 'tok_s', c['emitted_tok_s'])
