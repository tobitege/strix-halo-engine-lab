#!/usr/bin/env python3
"""Routes census from GLM decode receipts, in the DeepSeek routes format.

Receipts of a service started with --routes-out carry one "routes" line per
decoded token ("layer:id,id,... layer:id,..."; model layers 3..45). This tool
collects them into the DS routes layout read by count_routes / seed_order
(expert_tier.hpp) and by deepseek41_rank_main --routes:

    # seq NAME rows N
    3:12,40,... 4:7,... ...

NAME is <receipt file>:<path to the step list>; N is the prompt length when the
receipt records it (else 0). The census feeds --expert-routes on the next start.
"""
import argparse
import json
import re
import sys
from pathlib import Path

LINE = re.compile(r'^(\d+:\d+(,\d+)*)( \d+:\d+(,\d+)*)*$')


def step_lists(node, path=''):
    """Yield (path, steps, rows) for every list of step records carrying routes."""
    if isinstance(node, dict):
        steps = node.get('steps')
        if isinstance(steps, list) and any(isinstance(s, dict) and 'routes' in s for s in steps):
            rows = node.get('prompt_tokens', node.get('frontier', 0))
            yield path or '/', steps, int(rows) if isinstance(rows, int) else 0
        for k, v in node.items():
            if k != 'steps':
                yield from step_lists(v, f'{path}/{k}')
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from step_lists(v, f'{path}[{i}]')


def census(paths):
    """Return (text, sequences, tokens) of the DS-format census."""
    out, seqs, tokens = [], 0, 0
    for p in paths:
        rec = json.loads(Path(p).read_text(encoding='utf-8'))
        for where, steps, rows in step_lists(rec):
            lines = [s['routes'] for s in steps if isinstance(s, dict) and s.get('routes')]
            for line in lines:
                if not LINE.match(line):
                    raise ValueError(f'{p}{where}: malformed routes line')
            if not lines:
                continue
            out.append(f'# seq {Path(p).name}:{where} rows {rows}')
            out.extend(lines)
            seqs += 1
            tokens += len(lines)
    return ''.join(x + '\n' for x in out), seqs, tokens


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('receipts', nargs='+', type=Path)
    ap.add_argument('-o', '--out', type=Path, required=True)
    a = ap.parse_args(argv)
    text, seqs, tokens = census(a.receipts)
    if not tokens:
        print('no routes in these receipts (start the service with --routes-out)', file=sys.stderr)
        return 2
    a.out.write_text(text, encoding='utf-8', newline='\n')
    print(json.dumps({'out': str(a.out), 'sequences': seqs, 'tokens': tokens}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
