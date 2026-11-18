#!/usr/bin/env python3
"""FAST gate of a prefill-attention form: compare --dump-logits files of strix-cuda-qwen35-decode runs that prefilled the SAME long real
prompt (big prefill path) and then decoded the SAME teacher-forced tokens (--force-gen-file). The decode rows use the exact decode
attention in every arm, so the difference measures what the prefill left in the KV cache / GDN state (and its last row).

  attn_gate.py REF.bin TEST.bin FORCED.txt [--vocab 248320] [--offset 1]

Row j of a dump predicts forced[j + offset - 1] (offset 1: row 0 = first decode step, fed forced[0], predicts forced[1]).
Prints KL(ref || test) mean / p95 / max, top-1 agreement, mean NLL of the forced tokens under both and dNLL = test - ref, and the
gate verdict at KL <= 0.02, top-1 >= 0.93, dNLL <= 0.01. Errors are listed per row-quarter (clustered vs uniform).
"""
import argparse

import numpy as np


def logsoftmax(x):
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('ref')
    ap.add_argument('test')
    ap.add_argument('forced')
    ap.add_argument('--vocab', type=int, default=248320)
    ap.add_argument('--offset', type=int, default=1)
    a = ap.parse_args()
    ref = np.fromfile(a.ref, dtype=np.float32).reshape(-1, a.vocab).astype(np.float64)
    tst = np.fromfile(a.test, dtype=np.float32).reshape(-1, a.vocab).astype(np.float64)
    forced = [int(t) for t in open(a.forced, encoding='utf-8').read().replace('\n', ',').split(',') if t.strip()]
    n = min(len(ref), len(tst))
    if n == 0 or not np.isfinite(ref[:n]).all() or not np.isfinite(tst[:n]).all():
        print('gate: empty or non-finite dumps')
        return 2
    lr, lt = logsoftmax(ref[:n]), logsoftmax(tst[:n])
    kl = (np.exp(lr) * (lr - lt)).sum(axis=1)
    top1 = (ref[:n].argmax(axis=1) == tst[:n].argmax(axis=1)).mean()
    idx = [j for j in range(n) if 0 <= j + a.offset - 1 < len(forced)]
    tgt = np.array([forced[j + a.offset - 1] for j in idx])
    nll_r = -lr[idx, tgt].mean() if idx else float('nan')
    nll_t = -lt[idx, tgt].mean() if idx else float('nan')
    dnll = nll_t - nll_r
    print(f'rows {n} (ref {len(ref)}, test {len(tst)}), forced targets {len(idx)}')
    print(f'KL mean {kl.mean():.5f} p95 {np.percentile(kl, 95):.5f} max {kl.max():.5f}; top-1 {top1:.4f}; NLL ref {nll_r:.4f} test {nll_t:.4f} dNLL {dnll:+.5f}')
    q = max(1, n // 4)
    print('KL by row quarter: ' + ' '.join(f'{kl[i:i + q].mean():.5f}' for i in range(0, n, q)))
    ok = kl.mean() <= 0.02 and top1 >= 0.93 and dnll <= 0.01
    print('gate: ' + ('PASS' if ok else 'FAIL') + ' (KL <= 0.02, top-1 >= 0.93, dNLL <= 0.01)')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
