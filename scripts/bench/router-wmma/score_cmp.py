#!/usr/bin/env python3
"""score_cmp.py REF_PREFIX CAND_PREFIX PROMPT.u32 [--rows N] [--ranks K]: FAST quality comparison of two GLM scored
prefills (STRIX_GLM_PF_SCORE_LAST=N STRIX_GLM_PF_SCORE_OUT=PREFIX writes PREFIX-r<k>.f32, the head logits of the last N
prompt rows, vocab split across the TP ranks). Row i sits at position len-N+i and predicts token len-N+i+1.
Prints NLL of both arms, dNLL, mean KL(ref || cand), top-1 agreement and the verdict of the FAST gate rule
(KL <= 0.02, top-1 >= 93 %, dNLL <= +0.01). Exit 0 = PASS, 1 = FAIL, 2 = input error."""
import argparse
import sys

import numpy as np


def logprobs(prefix, rows, ranks):
    parts = []
    for k in range(ranks):
        a = np.fromfile(f"{prefix}-r{k}.f32", dtype="<f4")
        if a.size % rows:
            raise ValueError(f"{prefix}-r{k}.f32: {a.size} floats is not a multiple of {rows} rows")
        parts.append(a.reshape(rows, -1).astype(np.float64))
    x = np.concatenate(parts, axis=1)
    x -= x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("prompt")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--ranks", type=int, default=2)
    ap.add_argument("--max-kl", type=float, default=0.02)
    ap.add_argument("--min-top1", type=float, default=0.93)
    ap.add_argument("--max-dnll", type=float, default=0.01)
    o = ap.parse_args()
    try:
        toks = np.fromfile(o.prompt, dtype="<u4")
        a, b = logprobs(o.ref, o.rows, o.ranks), logprobs(o.cand, o.rows, o.ranks)
    except (OSError, ValueError) as e:
        print(f"score_cmp: {e}", file=sys.stderr)
        return 2
    n = o.rows
    # The last row predicts the token after the prompt (unknown): score rows 0..n-2 against the prompt's next tokens.
    tgt = toks[len(toks) - n + 1:].astype(np.int64)
    r = np.arange(len(tgt))
    na, nb = -a[r, tgt], -b[r, tgt]
    kl = (np.exp(a) * (a - b)).sum(axis=1)
    top1 = float((a.argmax(1) == b.argmax(1)).mean())
    dnll = float(nb.mean() - na.mean())
    ok = kl.mean() <= o.max_kl and top1 >= o.min_top1 and dnll <= o.max_dnll
    print(f"rows {len(tgt)} NLL ref {na.mean():.4f} cand {nb.mean():.4f} dNLL {dnll:+.4f} KL mean {kl.mean():.5f} "
          f"max {kl.max():.4f} top1 {top1 * 100:.1f}% -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
