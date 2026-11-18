#!/usr/bin/env python3
"""Compare the prefill head logits of two arms (STRIX_MM_QUALITY_OUT / STRIX_MM_QUALITY_ROWS dumps of strix-minimax-rank).

usage: logits_compare.py REF_PREFIX TEST_PREFIX ROWS
  REF_PREFIX / TEST_PREFIX: the dump prefix including the arm name, e.g. OUT/q-a (session "a") or OUT/q-cut-cut
  (layer-cut listen "cut"); files PREFIX-tf-r0.f32 and PREFIX-tf-r1.f32 hold [ROWS][vocab slice] per rank and
  PREFIX-logits-r{0,1}.f32 the last row. The two rank slices are concatenated into the full vocabulary.
Prints per-row top-1 agreement, mean / max KL(ref || test) in nats, max |logit difference| and, for each row whose
top-1 differs, the reference's top-1 - top-2 margin (a near tie flips without a defect; a large margin is a defect).
"""
import sys

import numpy as np


def load(prefix, what, rows):
    parts = [np.fromfile(f"{prefix}-{what}-r{r}.f32", dtype=np.float32) for r in (0, 1)]
    if rows:
        for rank, part in enumerate(parts):
            last = f"{prefix}-logits-r{rank}.f32"
            width = np.fromfile(last, dtype=np.float32).size
            if not width or part.size % width:
                raise ValueError(f"{prefix}: rank {rank}: incomplete vocabulary rows (last-row width {width})")
            retained = part.size // width
            if retained != rows:
                raise ValueError(f"{prefix}: rank {rank}: requested {rows} teacher-forced rows, retained {retained}")
            parts[rank] = part.reshape(rows, width)
        return np.concatenate(parts, axis=1).astype(np.float64)
    return np.concatenate(parts)[None, :].astype(np.float64)


def compare(ref, test, label):
    if ref.shape != test.shape:
        raise SystemExit(f"{label}: shape {ref.shape} != {test.shape}")
    ls_r = ref - ref.max(axis=1, keepdims=True)
    ls_t = test - test.max(axis=1, keepdims=True)
    lp_r = ls_r - np.log(np.exp(ls_r).sum(axis=1, keepdims=True))
    lp_t = ls_t - np.log(np.exp(ls_t).sum(axis=1, keepdims=True))
    kl = (np.exp(lp_r) * (lp_r - lp_t)).sum(axis=1)
    top_r, top_t = ref.argmax(axis=1), test.argmax(axis=1)
    agree = top_r == top_t
    srt = np.sort(ref, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    print(f"{label}: rows {ref.shape[0]} top-1 agree {agree.sum()}/{ref.shape[0]} ({100.0 * agree.mean():.2f} %), "
          f"KL mean {kl.mean():.3e} max {kl.max():.3e} nats, max |dlogit| {np.abs(ref - test).max():.4f}, "
          f"ref top1-top2 margin median {np.median(margin):.3f}")
    for i in np.flatnonzero(~agree)[:12]:
        print(f"  row {i}: ref top-1 {top_r[i]} test top-1 {top_t[i]}, ref margin {margin[i]:.4f}, KL {kl[i]:.3e}")


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    ref, test, rows = sys.argv[1], sys.argv[2], int(sys.argv[3])
    compare(load(ref, "logits", 0), load(test, "logits", 0), "last row")
    if rows > 0:
        compare(load(ref, "tf", rows), load(test, "tf", rows), f"last {rows} prompt rows (teacher-forced)")


if __name__ == "__main__":
    main()
