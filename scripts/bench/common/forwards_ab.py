#!/usr/bin/env python3
"""Steady decode cycles grouped by (lanes, forwards) from the server's cycle receipts.

    forwards_ab.py STDERR_LOG [--min-lanes 9] [--json OUT]

Reads "serve cycle ... lanes=L ... emitted=E ... ms=M prompt=0 ... forwards=F solo=..." lines
(strix/serve/cycle_receipt.hpp, STRIX_SERVE_RECEIPT=1). Cycles with a prompt unit are skipped. With
STRIX_SERVE_JOINT_GROUP_LANES=ab the two arms alternate by cycle and differ in forwards at the same lanes
(9-16 lanes: 1 vs 2; 17-24: 2 vs 3), so the table is an in-process A/B: per (lanes, forwards) the cycles,
mean and p95 cycle ms, tokens per cycle and tok/s (sum emitted / sum ms), and the rows / padding per cycle.
"""
import argparse
import json
import re
import sys

PAIR = re.compile(r"(\w+)=(\S+)")


def cycles(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            at = line.find("serve cycle ")
            if at < 0:
                continue
            kv = dict(PAIR.findall(line[at:]))
            try:
                # Missing execution evidence is unknown, never a measured zero/steady cycle.
                forwards = int(kv["forwards"])
                if kv["prompt"] not in ("0", "1") or forwards <= 0:
                    continue
                yield dict(lanes=int(kv["lanes"]), forwards=forwards, emitted=int(kv["emitted"]),
                           ms=float(kv["ms"]), prompt=kv["prompt"] == "1", rows=int(kv["rows"]), pad=int(kv["pad"]),
                           solo=kv.get("solo", ""))
            except (KeyError, ValueError):
                continue


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--min-lanes", type=int, default=9)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    groups = {}
    for c in cycles(a.log):
        if c["prompt"] or c["lanes"] < a.min_lanes:
            continue
        groups.setdefault((c["lanes"], c["forwards"]), []).append(c)
    rows = []
    print("lanes forwards cycles mean_ms p95_ms tok/cycle tok/s rows/cycle pad/cycle")
    for (lanes, fwd), cs in sorted(groups.items()):
        ms = sorted(c["ms"] for c in cs)
        total_ms = sum(ms)
        emitted = sum(c["emitted"] for c in cs)
        r = dict(lanes=lanes, forwards=fwd, cycles=len(cs), mean_ms=total_ms / len(cs),
                 p95_ms=ms[min(len(ms) - 1, int(0.95 * len(ms)))], tokens_per_cycle=emitted / len(cs),
                 tok_s=1000.0 * emitted / total_ms if total_ms > 0 else 0.0,
                 rows=sum(c["rows"] for c in cs) / len(cs), pad=sum(c["pad"] for c in cs) / len(cs))
        rows.append(r)
        print(f"{lanes} {fwd} {r['cycles']} {r['mean_ms']:.1f} {r['p95_ms']:.1f} {r['tokens_per_cycle']:.2f} "
              f"{r['tok_s']:.1f} {r['rows']:.1f} {r['pad']:.1f}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
