#!/usr/bin/env python3
"""Strict ABAB ordering and summaries for the scripts/bench kits (standard library only).

  abab.py order --pairs 3 A B [C ...]   print the run order, one arm per line: round r runs the arms rotated by r
                                        (A B / B A for two arms is NOT used: strict A B A B keeps drift one-sided
                                        and the per-round delta pairs each B with the A just before it).
  abab.py summary OUT [--base ARM] [--metric NAME]
                                        read OUT/**/receipt.json; per arm: valid/total, median warm and cold prefill,
                                        tok/s, clock, GPU clocks per token, identities; per-round deltas vs the base
                                        arm (explicit --base within one comparable workload); invalid receipts are listed
                                        with their reasons and excluded.
A comparison with fewer than 3 valid rounds per arm is printed but marked "UNDER 3 PAIRS".
"""
import argparse
import glob
import json
import os
import statistics
import sys


def cmd_order(args):
    arms = args.arms
    for r in range(args.pairs):
        for arm in (arms if args.strict else arms[r % len(arms):] + arms[:r % len(arms)]):
            print("%s %d" % (arm, r))


def load(out):
    recs = []
    for path in sorted(glob.glob(os.path.join(out, "**", "receipt.json"), recursive=True)):
        try:
            with open(path) as stream:
                rec = json.load(stream)
            rec["_dir"] = os.path.relpath(os.path.dirname(path), out)
            recs.append(rec)
        except (OSError, ValueError):
            pass
    return recs


def med(values):
    values = [v for v in values if v]
    return statistics.median(values) if values else 0.0


def value(rec, metric):
    if metric in ("prefill_ms_warm", "prefill_ms_cold", "tok_s_warm", "gpu_clocks_per_token"):
        return rec.get(metric) or 0.0
    return (rec.get("metrics") or {}).get(metric, 0.0)


def logits_verdict(left, right):
    """Compare each round once; no set reduction or discarded missing evidence."""
    if not left or set(left) != set(right):
        return "missing logits evidence/rounds"
    pairs = [(left[r].get("logits_sha"), right[r].get("logits_sha")) for r in sorted(left)]
    if any(a in (None, "", "-") or b in (None, "", "-") for a, b in pairs):
        return "missing logits evidence"
    varied = any(len({p[i] for p in pairs}) > 1 for i in (0, 1))
    verdict = "logits DIFFER" if any(a != b for a, b in pairs) else "bitwise-same logits"
    return verdict + ("; within-arm nondeterminism (or different workloads)" if varied else "")


def cmd_summary(args):
    recs = load(args.out)
    if not recs:
        print("no receipts under %s" % args.out)
        return 1
    order = []
    for rec in sorted(recs, key=lambda r: (r.get("round", 0), r["_dir"])):
        if rec["arm"] not in order:
            order.append(rec["arm"])
    base = args.base
    metric = args.metric
    for rec in recs:
        if not rec.get("valid"):
            print("! INVALID %s: %s" % (rec["_dir"], "; ".join(rec.get("invalid_reasons", []))[:160]))
    print("arm | valid | warm ms | cold ms | warm tok/s | %s | sclk | busy | Gclk/tok | logits | gen" % metric)
    per = {}
    for arm in order:
        ok = [r for r in recs if r["arm"] == arm and r.get("valid")]
        allr = [r for r in recs if r["arm"] == arm]
        per[arm] = {r.get("round", 0): r for r in ok}
        if len(per[arm]) != len(ok):
            print("! duplicate rounds for %s; comparison disabled" % arm)
            per[arm] = {}
        print("%s | %d/%d | %.1f | %.1f | %.0f | %.2f | %.0f | %.0f | %.0f | %s | %s" % (
            arm, len(ok), len(allr), med(r.get("prefill_ms_warm", 0) for r in ok), med(r.get("prefill_ms_cold", 0) for r in ok),
            med(r.get("tok_s_warm", 0) for r in ok), med(value(r, metric) for r in ok), med(r.get("sclk_mhz", 0) for r in ok),
            med(r.get("busy_pct", 0) for r in ok), med(r.get("gpu_clocks_per_token") or 0 for r in ok),
            ",".join(sorted({r.get("logits_sha", "-") for r in ok}))[:40], ",".join(sorted({r.get("gen_sha", "-") for r in ok}))[:28]))
    if base not in per:
        print("select an explicit --base ARM for a comparable workload group")
        return 0
    for arm in order:
        if arm == base:
            continue
        rounds = sorted(set(per[arm]) & set(per[base]))
        if any((per[arm][r].get("tokens"), per[arm][r].get("prefill_tokens"), per[arm][r].get("workload")) !=
               (per[base][r].get("tokens"), per[base][r].get("prefill_tokens"), per[base][r].get("workload")) for r in rounds):
            print("%s vs %s: incompatible workloads; comparison disabled" % (arm, base))
            continue
        deltas = []
        for r in rounds:
            b, c = value(per[base][r], metric), value(per[arm][r], metric)
            if b:
                deltas.append(100.0 * (c - b) / b)
        same = logits_verdict(per[base], per[arm])
        print("%s vs %s: %s per-round %s median %+.2f %% over %d rounds%s; %s" % (
            arm, base, metric, " ".join("%+.2f" % x for x in deltas) or "-", statistics.median(deltas) if deltas else 0.0, len(deltas),
            "" if len(deltas) >= 3 else " UNDER 3 PAIRS", same))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("order")
    o.add_argument("--pairs", type=int, default=3)
    o.add_argument("--rotate", dest="strict", action="store_false", help="rotate the arm order every round")
    o.add_argument("arms", nargs="+")
    s = sub.add_parser("summary")
    s.add_argument("out")
    s.add_argument("--base")
    s.add_argument("--metric", default="prefill_ms_warm")
    args = p.parse_args(argv)
    return {"order": cmd_order, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
