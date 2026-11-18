#!/usr/bin/env python3
"""Hand-off piece schedules for strix-xfer-probe send --schedule (lines "READY_MS BYTES WIDTH").

  schedule.py qwen --context 32768 --prefill-s 7.59 [--block 4096]
      Qwen 3.6 MTP layout (docs/rtx-state-import.md): 48 layers, KV + index rows on every 4th layer (2048 + 256 B/row,
      BF16), MTP rows 2304 B/row once per block, released per layer and block as the CUDA producer does
      (release_by_layer); the fixed blob (GDN recurrent matrix 36 x 3,145,728 B FP32, GDN convolution 36 x 81,920 B,
      PLE convolution 184,320 B, MTP held 20,480 B) per layer of the LAST block, when each layer's state is final.
  schedule.py ds41 --context 32768 --prefill-s 8.2 [--bytes-per-token 983] [--layers 43] [--after]
      DS41 attention state (~0.9 KB/token, RTXNOW-RESULT S2: 32.2 MB at 32K) cut per layer, produced uniformly over
      the prefill; --after = today's v1 path: the whole state once, after the prefill.
  schedule.py uniform --bytes 810e6 --pieces 60 --prefill-s 30.2 [--width 2]
      any other state (GLM v2 0.81 GB / MiniMax 5.03 GB at 32K, wave4 HETPF): BYTES cut in PIECES produced uniformly;
      --prefill-s 0 = everything ready at once (pure transfer seconds).
Prefill time is an input (the RTX receipt), so the probe measures the transport, not a model.
"""
import argparse
import math
import sys


def qwen(a):
    out = []
    n_blocks = math.ceil(a.context / a.block)
    t = 0.0
    for b in range(n_blocks):
        rows = min(a.block, a.context - b * a.block)
        dt = a.prefill_s * rows / a.context
        last = b == n_blocks - 1
        for layer in range(48):
            ready = (t + dt * (layer + 1) / 48) * 1e3
            if layer % 4 == 3:
                out.append((ready, rows * 2048, 2))
                out.append((ready, rows * 256, 2))
            elif last:
                out.append((ready, 3145728, 4))
                out.append((ready, 81920, 2))
        out.append(((t + dt) * 1e3, rows * 2304, 2))
        t += dt
    out.append((t * 1e3, 184320 + 20480, 2))
    return out


def ds41(a):
    total = a.context * a.bytes_per_token
    if a.after:
        return [(a.prefill_s * 1e3, total, 2)]
    per = total // a.layers
    return [(a.prefill_s * 1e3 * (i + 1) / a.layers, per if i < a.layers - 1 else total - per * (a.layers - 1), 2) for i in range(a.layers)]


def uniform(a):
    total, n = int(float(a.bytes)), a.pieces
    per = total // n
    return [(a.prefill_s * 1e3 * (i + 1) / n, per if i < n - 1 else total - per * (n - 1), a.width) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["qwen", "ds41", "uniform"])
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--prefill-s", type=float, required=True)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--bytes-per-token", type=int, default=983)
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--after", action="store_true")
    ap.add_argument("--bytes", default="0")
    ap.add_argument("--pieces", type=int, default=60)
    ap.add_argument("--width", type=int, default=2)
    a = ap.parse_args()
    rows = qwen(a) if a.family == "qwen" else ds41(a) if a.family == "ds41" else uniform(a)
    total = sum(r[1] for r in rows)
    sys.stdout.write("# %s context %d prefill %.3f s: %d pieces, %d bytes\n" % (a.family, a.context, a.prefill_s, len(rows), total))
    for ms, b, w in rows:
        sys.stdout.write("%.3f %d %d\n" % (ms, b, w))


if __name__ == "__main__":
    main()
