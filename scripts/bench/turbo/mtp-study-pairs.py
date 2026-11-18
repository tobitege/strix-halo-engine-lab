"""Pool scoped before/after pairs, preserving cold/warm and token-ID controls."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('root', type=Path)
p.add_argument('--prefix', default='final')
p.add_argument('--pairs', type=int, default=3)
p.add_argument('--before-binary', type=Path, required=True)
p.add_argument('--after-binary', type=Path, required=True)
a = p.parse_args()
if a.pairs < 1:
    p.error('--pairs must be positive')
root = a.root
rows = []
tokens = {}
for pair in range(1, a.pairs + 1):
    for variant in ['before', 'after']:
        for arm in ['E', 'T']:
            tag = f'{a.prefix}-{pair}-{variant}-{arm}'
            run = root / 'runs' / tag
            recs = json.loads((run / 'cycle-receipt-summary.json').read_text())
            assert len(recs) == 2 and all(r['scope'] != 'process aggregate; request attribution unresolved' for r in recs)
            requests = [json.loads(s) for s in (run / 'res.jsonl').read_text().splitlines()]
            assert all(len(r['lanes']) == 1 and len(r['lanes'][0]['token_ids']) == r['lanes'][0]['tokens'] for r in requests), (tag, 'missing token ledger')
            ids = [r['lanes'][0]['token_ids'] for r in requests]
            assert ids[0] == ids[-1], (tag, 'cold/warm ID mismatch')
            tokens[(pair, variant, arm)] = ids[-1]
            for temperature, rec in zip(['cold', 'warm'], recs):
                row = dict(tag=tag, pair=pair, variant=variant, arm=arm, temperature=temperature, **rec)
                rows.append(row)
                print(tag, temperature, 'E', round(rec['E'], 6), 'ms', round(rec['mean_ms']['ms'], 6),
                      'tps', round(rec['receipt_tps'], 4), 'GPU', json.dumps(rec['sampled_gpu'], separators=(',', ':')))
for pair in range(1, a.pairs + 1):
    for arm in ['E', 'T']:
        assert tokens[(pair, 'before', arm)] == tokens[(pair, 'after', arm)], (pair, arm, 'before/after ID mismatch')
        assert tokens[(pair, 'before', arm)] == tokens[(1, 'before', arm)], (pair, arm, 'cross-pair ID mismatch')
summary = {}
for variant in ['before', 'after']:
    for arm in ['E', 'T']:
        selected = [r for r in rows if r['variant'] == variant and r['arm'] == arm and r['temperature'] == 'warm']
        cycles = sum(r['cycles'] for r in selected)
        emitted = sum(r['cycle_emitted'] for r in selected)
        ms = sum(r['sum_ms']['ms'] for r in selected)
        summary[variant + '-' + arm] = dict(cycles=cycles, emitted=emitted, E=emitted/cycles,
                                          ms_per_pass=ms/cycles, tps=1000*emitted/ms)
print('POOLED', json.dumps(summary, separators=(',', ':')))
result = dict(rows=rows, pooled=summary, token_ids_before_after_equal=True,
              binary_sha256={n: hashlib.sha256(path.read_bytes()).hexdigest()
                             for n, path in [('before', a.before_binary), ('after', a.after_binary)]})
(root/(a.prefix+'-timings.json')).write_text(json.dumps(result, indent=2)+'\n')
