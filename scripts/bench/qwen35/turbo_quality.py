#!/usr/bin/env python3
"""Diagnostic comparison of a TURBO run against the BF16 reference: KL(ref || turbo), top-1 agreement, top-1 matching prefix.

Inputs are two --dump-logits files of strix-cuda-qwen35-decode (float32 [positions][vocab], same prompt ids, every prompt row
dumped): the exact BF16 run (reference) and the TURBO run. Errors are printed per position bucket so that clustering (a bug:
one tensor/edge/group) can be told from uniform spread (precision): the KL of the worst 5 % positions vs the median, and the
share of the total KL carried by the worst 5 %. Usage: turbo_quality.py REF.bin TURBO.bin VOCAB ALIGNED_ROWS
"""
import sys

import numpy as np


def main() -> int:
    ref_path, tur_path, vocab = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ref = np.fromfile(ref_path, dtype=np.float32).reshape(-1, vocab).astype(np.float64)
    tur = np.fromfile(tur_path, dtype=np.float32).reshape(-1, vocab).astype(np.float64)
    if ref.shape != tur.shape or not len(ref):
        raise ValueError("logit dumps must have identical nonempty shapes")
    if not np.isfinite(ref).all() or not np.isfinite(tur).all():
        raise ValueError("nonfinite logits")
    n = len(ref)
    # required 4th arg: number of aligned rows (prompt + continuation while both runs fed the same tokens);
    # rows after the two greedy sequences diverge compare different contexts and must not enter the comparison.
    if len(sys.argv) != 5:
        raise ValueError("provide explicitly aligned rows; this diagnostic does not prove conditioning")
    n = int(sys.argv[4])
    if n <= 0 or n > len(ref):
        raise ValueError("aligned row count out of range")
    ref, tur = ref[:n], tur[:n]

    def logsm(x):
        m = x.max(axis=1, keepdims=True)
        return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))

    lr, lt = logsm(ref), logsm(tur)
    pr = np.exp(lr)
    kl = (pr * (lr - lt)).sum(axis=1)
    top_r, top_t = ref.argmax(axis=1), tur.argmax(axis=1)
    agree = float((top_r == top_t).mean())
    first = np.nonzero(top_r != top_t)[0]
    greedy = int(first[0]) if len(first) else n
    srt = np.sort(kl)
    k5 = max(1, n // 20)
    share = float(srt[-k5:].sum() / max(kl.sum(), 1e-30))
    print(f"positions {n}  KL mean {kl.mean():.5f} median {np.median(kl):.5f} p95 {np.percentile(kl, 95):.5f} max {kl.max():.5f}")
    print(f"top-1 agreement {agree * 100:.2f} %  top-1 matching prefix {greedy}/{n}")
    print(f"worst 5 % of positions carry {share * 100:.1f} % of the KL (uniform ~ 5-15 %, a bug clusters far higher)")
    print(f"max |dlogit| {np.abs(ref - tur).max():.4f}  mean |dlogit| {np.abs(ref - tur).mean():.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
