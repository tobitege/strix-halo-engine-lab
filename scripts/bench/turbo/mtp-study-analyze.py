#!/usr/bin/env python3
"""Joint proposal/target accounting; only compare logits with identical token contexts."""
import argparse
import collections
import json
from pathlib import Path
import re
import struct
import numpy as np

CLASSES = ['code', 'prose', 'json', 'edit']
MAGIC = 0x3254474C50544D53
HEADER = struct.Struct('<10Q')

def load_run(path):
    anchor = json.loads((path / 'clock-anchor.json').read_text())
    offset = (anchor['epoch_ns'] - anchor['monotonic_ns']) / 1e9
    requests = []
    for n, request in enumerate(map(json.loads, (path / 'res.jsonl').read_text().splitlines())):
        for lane in request['lanes']:
            if 'error' in lane:
                raise ValueError(lane['error'])
            requests.append(dict(lane, cohort=n, begin=request['t0'] + lane['start'], end=request['t0'] + lane['end']))
    def locate(ns):
        epoch = ns / 1e9 + offset
        matches = [(i, r) for i, r in enumerate(requests) if r['begin'] <= epoch <= r['end']]
        return matches[0] if len(matches) == 1 else (None, None)
    comparisons = collections.defaultdict(list)
    totals = collections.defaultdict(lambda: collections.Counter())
    prefixes = {}
    for line in (path / 'srv.err').read_text(errors='replace').splitlines():
        if not line.startswith('mtp-verify '):
            continue
        row = dict(re.findall(r'(\w+)=([^ ]+)', line))
        i, request = locate(int(row['ns']))
        if request is None:
            raise ValueError('unresolved verification timestamp attribution')
        row = {k: float(v) if k == 'p' else int(v) for k, v in row.items()}
        group = (i, row['ns'], row['at'])
        depth, prefix = prefixes.get(group, (1, True))
        if row['d'] != depth:
            raise ValueError('noncontiguous verification comparison depths')
        equal = int(row['proposal'] == row['target'])
        eligible = int(row['d'] < row['budget'])
        reached = int(prefix and eligible)
        expected = int(reached and equal and not row['eos'])
        if any(row[k] != v for k, v in (
                ('equal', equal), ('eligible', eligible), ('reached', reached),
                ('greedy_accept', expected))):
            raise ValueError('verification fields disagree with proposal-ID prefix reconstruction')
        prefixes[group] = (depth + 1, bool(expected))
        comparisons[i].append(row)
        key = (request['cohort'], CLASSES[request['lane']], row['d'])
        count = totals[key]
        count.update(compared=1, equal=row['equal'], reached=row['reached'],
                     accepted=row['accepted'], expected_greedy_accept=row['greedy_accept'],
                     proposal_p_sum=row['p'], proposal_below_07=int(row['p'] < .7),
                     independently_expected_accept=expected,
                     eligible=row.get('eligible', 1), diagnostic=row.get('diagnostic', 0),
                     accounting_gap=(row['accepted'] - expected) if not row.get('diagnostic', 0) else 0)
    for i, request in enumerate(requests):
        if sum(row['accepted'] for row in comparisons[i]) != request['acc']:
            raise ValueError('verification accepted rows disagree with HTTP accepted-token ledger')
    return requests, locate, comparisons, totals

def dump_records(path):
    with path.open('rb') as f:
        while raw := f.read(HEADER.size):
            if len(raw) != HEADER.size:
                raise ValueError('truncated header')
            fields = HEADER.unpack(raw)
            if fields[0] != MAGIC or not 1 <= fields[5] <= 1_000_000:
                raise ValueError('invalid logit record')
            raw = f.read(fields[5] * 4)
            if len(raw) != fields[5] * 4:
                raise ValueError('truncated logits')
            yield fields, np.frombuffer(raw, dtype='<f4').copy()

def indexed_logits(path, run):
    requests, locate, _, _ = run
    out = {}
    counts = collections.Counter()
    windows = {}
    for fields, values in dump_records(path):
        _, position, base, row, rows, vocab, consumed, window_hash, lineage, ns = fields
        i, request = locate(ns)
        if request is None:
            counts['outside_requests'] += 1
            continue
        # Serial prompt-tail execution is also dumped. Generation consumes
        # completion ID zero at the prompt length, not at the first dump.
        # Keep this absolute token-ledger anchor independent of target argmax.
        if base < request['prompt']:
            counts['prompt_rows'] += 1
            continue
        input_index = base - request['prompt']
        if values.size > 1:
            top = np.partition(values, -2)[-2:]
            margin = float(np.max(top) - np.min(top))
            counts['exact_max_ties'] += int(margin == 0)
            counts['min_top1_margin'] = min(counts.get('min_top1_margin', float('inf')), margin)
        ids = request.get('token_ids', [])
        if input_index < 0 or input_index >= len(ids):
            counts['ledger_mismatch'] += 1
            continue
        if position != base + row or not 0 <= row < rows:
            raise ValueError('invalid physical row position')
        inputs = windows.setdefault((i, base, rows), [])
        if row == 0:
            if inputs:
                raise ValueError('duplicate window base')
            if ids[input_index] != consumed:
                counts['ledger_mismatch'] += 1
                continue
        elif not inputs:
            counts['missing_window_base'] += 1
            continue
        if row != len(inputs):
            raise ValueError('noncontiguous window row dump')
        inputs.append(consumed)
        calculated_hash = 14695981039346656037
        for token in inputs:
            for shift in range(0, 32, 8):
                calculated_hash = ((calculated_hash ^ ((token >> shift) & 255)) * 1099511628211) & ((1 << 64) - 1)
        if window_hash != calculated_hash:
            raise ValueError('window input prefix hash mismatch')
        # Only the base input is committed. Later physical rows consume drafts,
        # which may intentionally differ from the eventual completion ledger.
        history = tuple(ids[:input_index]) + tuple(inputs)
        key = (i, position, row, window_hash, history, rows)
        if key in out:
            raise ValueError('duplicate scoped logit row')
        out[key] = (values, request, fields)
        counts['indexed'] += 1
    return out, counts

def compare_logits(left, right, left_run, right_run, context_only=False):
    l, lc = indexed_logits(left, left_run)
    r, rc = indexed_logits(right, right_run)
    stats = collections.defaultdict(list)
    row_pairs = collections.defaultdict(collections.Counter)
    disagreement_examples = []
    joint = collections.defaultdict(lambda: collections.Counter(
        matched_target_rows=0, left_draft_rows=0, right_draft_rows=0,
        left_draft_left_target_equal=0, left_draft_right_target_equal=0,
        right_draft_left_target_equal=0, right_draft_right_target_equal=0,
        left_draft_acceptance_flip=0, right_draft_acceptance_flip=0,
        joint_proposal_rows=0, proposal_equal=0))
    def proposal(run, key, fields, target):
        i, position = key[:2]
        _, _, base, row, physical_rows, *_ = fields
        ns = fields[-1]
        found = [c for c in run[2][i] if c['at'] + c['d'] == position and c['ns'] >= ns
                 and (physical_rows == 1 or c['at'] + 1 == base and c['d'] == row + 1)]
        if not found:
            return None
        selected = min(found, key=lambda c: c['ns'] - ns)
        if selected['target'] != target:
            raise ValueError('logged verification token differs from dumped target first-max')
        return selected
    if context_only:
        # A serial row and a window row can consume exactly the same complete
        # token context. Match that ledger, retaining the row shapes as evidence.
        # Prefer a right-side row zero reference if a context occurs repeatedly.
        references = {}
        for rk in sorted(r, key=lambda k: (k[2], k[3])):
            references.setdefault((rk[0], rk[1], rk[4]), rk)
        pairs = [(key, references[(key[0], key[1], key[4])]) for key in l
                 if (key[0], key[1], key[4]) in references]
    else:
        pairs = [(key, key) for key in l.keys() & r.keys()]
    for key, right_key in pairs:
        x, req, lf = l[key]
        y, other, rf = r[right_key]
        if (req['prompt'] != other['prompt'] or req['lane'] != other['lane'] or
                not req.get('request_sha256') or req['request_sha256'] != other.get('request_sha256')):
            raise ValueError('prompt identity mismatch')
        bitwise = int(x.tobytes() == y.tobytes())
        tx, ty = int(np.argmax(x)), int(np.argmax(y))
        rows = row_pairs[(lf[4], lf[3], rf[4], rf[3])]
        rows.update(rows=1, top1_equal=int(tx == ty), bitwise=bitwise)
        rows['max_abs'] = max(rows['max_abs'], float(np.max(np.abs(x.astype(np.float64) - y))))
        px, py = proposal(left_run, key, lf, tx), proposal(right_run, key, rf, ty)
        if tx != ty and len(disagreement_examples) < 16:
            fields = ('at', 'd', 'budget', 'proposal', 'target', 'reached', 'eligible', 'eos',
                      'greedy_accept', 'diagnostic', 'accepted')
            disagreement_examples.append(dict(
                cls=CLASSES[req['lane']], request=key[0], position=key[1],
                left_rows=lf[4], left_row=lf[3], right_rows=rf[4], right_row=rf[3],
                left_top1=tx, right_top1=ty,
                left_proposal={k: px[k] for k in fields} if px is not None else None,
                right_proposal={k: py[k] for k in fields} if py is not None else None))
        depth = px['d'] if px is not None else py['d'] if py is not None else key[2] + 1
        j = joint[(CLASSES[req['lane']], depth)]
        j['matched_target_rows'] += 1
        for name, p in [('left_draft', px), ('right_draft', py)]:
            if p is not None:
                j[name + '_rows'] += 1
                j[name + '_left_target_equal'] += int(p['proposal'] == tx)
                j[name + '_right_target_equal'] += int(p['proposal'] == ty)
                j[name + '_acceptance_flip'] += int((p['proposal'] == tx) != (p['proposal'] == ty))
        if px is not None and py is not None:
            j['joint_proposal_rows'] += 1
            j['proposal_equal'] += int(px['proposal'] == py['proposal'])
        x, y = x.astype(np.float64), y.astype(np.float64)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError('nonfinite target logits')
        lx, ly = x - np.max(x), y - np.max(y)
        lx -= np.log(np.exp(lx).sum()); ly -= np.log(np.exp(ly).sum())
        stats[CLASSES[req['lane']]].append(dict(
            kl=float(np.sum(np.exp(lx) * (lx - ly))),
            top1_equal=int(tx == ty), bitwise=bitwise))
    return dict(left_records=dict(lc), right_records=dict(rc),
                matched_left_rows=len({key for key, _ in pairs}),
                matched_right_rows=len({key for _, key in pairs}),
                left_frontier_rows=sum(key[2] == 0 for key in l),
                right_frontier_rows=sum(key[2] == 0 for key in r),
                matched_left_frontier_rows=sum(key[2] == 0 for key, _ in pairs),
                matched_right_frontier_rows=len({key for _, key in pairs if key[2] == 0}),
                completion_ledger_pairs=[dict(request=i,
                    equal=bool(left_run[0][i].get('token_ids')) and
                          left_run[0][i].get('token_ids') == right_run[0][i].get('token_ids'))
                    for i in range(min(len(left_run[0]), len(right_run[0])))],
                matching='full-context' if context_only else 'full-context-and-physical-row',
                disagreement_examples=disagreement_examples,
                row_pairs=[dict(left_rows=lr, left_row=lrow, right_rows=rr, right_row=rrow, **c)
                           for (lr, lrow, rr, rrow), c in sorted(row_pairs.items())],
                proposal_vs_targets=[dict(cls=cls, depth=depth, **c,
                    left_draft_vs_left_target=c['left_draft_left_target_equal'] / c['left_draft_rows'] if c['left_draft_rows'] else None,
                    left_draft_vs_right_target=c['left_draft_right_target_equal'] / c['left_draft_rows'] if c['left_draft_rows'] else None,
                    right_draft_vs_left_target=c['right_draft_left_target_equal'] / c['right_draft_rows'] if c['right_draft_rows'] else None,
                    right_draft_vs_right_target=c['right_draft_right_target_equal'] / c['right_draft_rows'] if c['right_draft_rows'] else None)
                    for (cls, depth), c in sorted(joint.items())],
                matched_context_rows=sum(map(len, stats.values())), classes={
                    cls: dict(rows=len(rows), kl=float(np.mean([v['kl'] for v in rows])),
                              top1_agreement=float(np.mean([v['top1_equal'] for v in rows])),
                              bitwise_rows=sum(v['bitwise'] for v in rows))
                    for cls, rows in stats.items()})

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--logits', nargs=2, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--context-only', action='store_true',
                   help='Match identical full token histories across different physical row shapes')
    a = p.parse_args()
    runs = [load_run(path) for path in a.runs]
    result = {'runs': {str(path): [dict(request=n, cls=cls, depth=d, **counts,
                  equality=counts['equal'] / counts['compared'],
                  conditional_acceptance=counts['accepted'] / counts['reached'] if counts['reached'] else None,
                  counterfactual_conditional_acceptance=counts['expected_greedy_accept'] / counts['reached'] if counts['reached'] else None,
                  mean_proposal_p=counts['proposal_p_sum'] / counts['compared'],
                  normal_accounting_gap=counts['accounting_gap'])
                  for (n, cls, d), counts in sorted(run[3].items())]
               for path, run in zip(a.runs, runs)}}
    if a.logits:
        if len(runs) != 2:
            p.error('logit comparison requires exactly two runs')
        result['target_agreement'] = compare_logits(*a.logits, *runs, context_only=a.context_only)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(a.out), compared_rows=[sum(c['compared'] for c in run[3].values()) for run in runs],
                          target_agreement=result.get('target_agreement'))))

if __name__ == '__main__':
    main()
