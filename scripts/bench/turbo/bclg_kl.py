#!/usr/bin/env python3
"""KL(ref||cand), top-1 agreement and dNLL between two STRIX_BITCALC_LOGITS_OUT files (records 'BCLG', full vocabulary).

  bclg_kl.py REF CAND [CAND ...]    per cand: records compared (step-aligned), KL mean/max, top-1 %, dNLL on the ref token
Record 0 is the prefill's last row (the prompt-conditioned distribution); later records are decode steps and are
comparable only while both runs emitted the same tokens (stop at the first step whose emitted token differs).
"""
import struct, sys
from itertools import zip_longest
import numpy as np


def records(path):
    with open(path, "rb") as f:
        previous = None
        while True:
            h = f.read(24)
            if not h:
                return
            if len(h) != 24:
                raise ValueError(f"{path}: partial header")
            magic, step, pos, argmax, emitted, n = struct.unpack("<6I", h)
            if magic != 0x42434C47 or n == 0 or max(argmax, emitted) >= n:
                raise ValueError(f"{path}: invalid record header at step {step}")
            if previous is not None and (step, pos, n) != (previous[0] + 1, previous[1] + 1, previous[2]):
                raise ValueError(f"{path}: discontinuous capture at step {step}")
            previous = step, pos, n
            payload = f.read(4 * n)
            if len(payload) != 4 * n:
                raise ValueError(f"{path}: partial payload at step {step}")
            logits = np.frombuffer(payload, dtype="<f4").astype(np.float64)
            if not np.isfinite(logits).all():
                raise ValueError(f"{path}: non-finite logits at step {step}")
            yield step, pos, n, argmax, emitted, logits


def lse(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


def main():
    if len(sys.argv) < 3:
        raise ValueError("usage: bclg_kl.py REF CAND [CAND ...]")
    ref = list(records(sys.argv[1]))
    for cp in sys.argv[2:]:
        cand = list(records(cp))
        kl, t1, dn = [], 0, []
        n = 0
        divergence = "none"
        for rr, cr in zip_longest(ref, cand):
            if rr is None or cr is None:
                raise ValueError(f"{cp}: unequal capture coverage before token divergence")
            s, pos, vocab, a, e, lr = rr
            s2, pos2, vocab2, a2, e2, lc = cr
            if (s, pos, vocab) != (s2, pos2, vocab2):
                raise ValueError(f"{cp}: unaligned step/position/vocabulary: {(s, pos, vocab)} != {(s2, pos2, vocab2)}")
            lpr, lpc = lr - lse(lr), lc - lse(lc)
            kl.append(float((np.exp(lpr) * (lpr - lpc)).sum()))
            t1 += int(a == a2)
            dn.append(float(-lpc[e] + lpr[e]))
            n += 1
            if e != e2:
                divergence = f"step={s},position={pos},ref={e},cand={e2}"
                break
        if not n:
            raise ValueError(f"{cp}: no records")
        print("%s rows=%d KL mean %.5f max %.5f top1 %.1f%% dNLL %+.5f divergence=%s" % (cp, n, np.mean(kl), max(kl), 100 * t1 / n, np.mean(dn), divergence))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        sys.exit(f"bclg_kl: {error}")
