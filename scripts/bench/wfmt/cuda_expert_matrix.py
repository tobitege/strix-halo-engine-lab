#!/usr/bin/env python3
"""Emit the fixed synthetic CUDA expert capture inventory, never observed sizes.

The independently specified dimensions/strides mirror the capture fixture's public
buffer ABI. Changing that ABI requires reviewing both this matrix and the fixture.
"""
import argparse
import json
from pathlib import Path


def matrix():
    hidden, ffn, routes, experts = 2560, 640, 10, 3
    widths = [hidden * 2, 40, 40, hidden, hidden // 32 * 4, hidden // 32 * 4, 4,
              routes * 2 * ffn * 4, hidden * 4, routes * ffn,
              routes * ffn // 32 * 4, routes * ffn // 32 * 4,
              routes * hidden * 4, hidden * 2]
    strides = [(n + 255) // 256 * 256 + 256 + 16 * v for v, n in enumerate(widths)]
    geometries = {7: (32, 24), 8: (32, 34), 12: (256, 144), 13: (256, 176),
                  18: (256, 98), 20: (32, 18), 21: (256, 110), 22: (256, 82),
                  23: (256, 136), 42: (64, 18)}
    cases = []

    def case(gt, dt, rows, fast, native=False):
        gb, gbytes = geometries[gt]
        db, dbytes = geometries[dt]
        name = f'moe-g{gt}-d{dt}-m{rows}-' + ('fast' if fast else 'exact')
        sections = {
            'source': 2 * experts * ffn * (hidden // gb) * gbytes + experts * hidden * (ffn // db) * dbytes,
            'state': rows * sum(strides[v] for v in [0, 1, 2, 6]) + 128,
            'activation_codes': rows * sum(strides[v] for v in [3, 9]),
            'activation_scales': rows * sum(strides[v] for v in [4, 5, 10, 11]),
            'partials': rows * sum(strides[v] for v in [7, 12]),
            'output': rows * sum(strides[v] for v in [13, 8]),
        }
        arithmetic = ('cuda-native-iq-fp32@1' if native else
                      'cuda-moe-mma@1' if fast else 'cuda-four-chain-imma@1')
        cases.append({'file': name + '.wfmt', 'case': name, 'arithmetic': arithmetic,
                      'shape': [rows, hidden, ffn], 'required_sections': sections})

    for gt in [12, 13]:
        for dt in [7, 8]:
            for rows in [1, 8, 9, 33, 65]:
                for fast in [False, True]:
                    case(gt, dt, rows, fast)
    for gt in [18, 21, 22, 23, 42]:
        for dt in [20, 42]:
            for rows in [1, 9]:
                case(gt, dt, rows, False, True)
    return {'schema': 'wfmt.matrix@1', 'cases': cases}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(matrix(), indent=2) + '\n', encoding='utf-8')
