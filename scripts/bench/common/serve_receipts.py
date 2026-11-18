#!/usr/bin/env python3
"""serve_receipts.py DIR --tag TAG --arm ARM --server-up 0|1

Turns the serve_bench.py results of one served run (DIR/<class>-<round>.json, server data in DIR/server/) into
one receipt per (class, round) at DIR/<class>-<round>/receipt.json, in the same schema as receipt.py, so
abab.py compares served decode the same way it compares prefill (e.g. abab.py summary OUT --metric e2e_C8).
Metrics per concurrency n: agg_C<n> (steady aggregate tok/s from server cycle receipts, when present),
e2e_C<n> (client end-to-end tok/s), req_C<n> (per-request p50), ttft_C<n> (s), fail_C<n> (failed requests).
Validity: the server came up, no failed request, and the server process passed the receipt.py checks
(kfd evicted_ms flat, MemAvailable floor, no kernel-log events).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import receipt  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir")
    p.add_argument("--tag", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--server-up", type=int, default=1)
    p.add_argument("--round-base", type=int, default=0, help="ABAB round of this server start (added to the in-run round)")
    p.add_argument("--client", type=int, default=0, help="1: a --client-cmd run; one server-validity receipt")
    p.add_argument("--min-avail-gb", type=float, default=float(os.environ.get("KIT_MIN_AVAIL_GB", "2")))
    p.add_argument("--client-rc", type=int)
    p.add_argument("--expected", default="", help="comma-separated class-round names, including unstarted work")
    p.add_argument("--concurrency", type=lambda x: [int(n) for n in x.split(",")], default=[1, 4, 8])
    p.add_argument("--peer", help="downloaded peer receipt; required when a peer was launched")
    args = p.parse_args(argv)
    sdir = os.path.join(args.dir, "server")
    rows = receipt.samples(os.path.join(sdir, "samples.tsv"))
    ev = [r[8] for r in rows]
    avail = [r[6] for r in rows]
    klog = receipt.read_text(os.path.join(sdir, "klog.txt")).splitlines()
    meta = json.loads(receipt.read_text(os.path.join(sdir, "meta.json")) or "{}")
    base = receipt.process_reasons(meta) + receipt.telemetry_reasons(
        rows, args.min_avail_gb, receipt.growth(ev), klog,
        receipt.read_text(os.path.join(sdir, "sampler.error")))
    if not meta.get("intentional_shutdown"):
        base.append("server exited before intentional shutdown")
    if not args.server_up:
        base.append("server not ready")
    if args.client_rc != 0:
        base.append("client rc=%s" % args.client_rc)
    busy = sum(r[4] for r in rows) / len(rows) if rows else 0.0
    sclk = sum(r[2] for r in rows) / len(rows) / 1e6 if rows else 0.0
    names = ["client"] if args.client else args.expected.split(",") if args.expected else ["missing-0"]
    all_valid = True
    for name in names:
        metrics, reasons = {}, list(base)
        path = os.path.join(args.dir, "client-result.json" if args.client else name + ".json")
        try:
            with open(path) as stream:
                result = json.load(stream)
        except (OSError, ValueError):
            result = {}
        if args.client:
            if result.get("completed", 0) <= 0 or result.get("failed") != 0:
                reasons.append("missing or failed client workload evidence")
            cls, rnd = "client", args.round_base
        else:
            cls, _, round_text = name.rpartition("-")
            rnd = args.round_base + int(round_text)
            bursts = result.get("bursts", [])
            if not bursts or sorted(b.get("concurrency") for b in bursts) != sorted(args.concurrency):
                reasons.append("missing or incomplete bursts")
            for b in bursts:
                n, c, steady = b["concurrency"], b.get("client", {}), b.get("steady", {})
                for key, val in (("agg", steady.get("aggregate_tok_s")), ("e2e", c.get("end_to_end_tok_s")),
                                 ("req", c.get("per_request_tok_s_p50")), ("ttft", c.get("ttft_p50_s")), ("fail", c.get("failed"))):
                    if val is not None:
                        metrics["%s_C%d" % (key, n)] = round(float(val), 3)
                if c.get("failed") != 0 or c.get("requests") != n or c.get("output_tokens", 0) <= 0:
                    reasons.append("C%d incomplete/failed requests" % n)
        rec = dict(tag="%s-%s" % (args.tag, name), arm=args.arm if args.client else args.arm + "-" + cls,
                   round=rnd, workload=cls, host=meta.get("host"), bin_sha=meta.get("bin_sha"),
                   device_bindings=meta.get("device_bindings", []),
                   rc=meta.get("rc"), client_rc=args.client_rc, intentional_shutdown=meta.get("intentional_shutdown", False),
                   wall_ms=meta.get("wall_ms"), mem_start=meta.get("mem"), prefills=0, prefill_ms_cold=0.0,
                   prefill_ms_warm=0.0, tokens=0, tok_s_cold=0.0, tok_s_warm=0.0, sclk_mhz=round(sclk), busy_pct=round(busy, 1),
                   gpu_clocks_per_token=None, logits_sha="-", gen_sha="-", identity=None, metrics=metrics,
                   evicted_ms_run=(receipt.growth(ev) if ev else -1), avail_min_gb=(round(min(avail) / 2 ** 20, 1) if avail else -1),
                   klog_events=len(klog), valid=not reasons, invalid_reasons=reasons)
        dest = os.path.join(args.dir, name)
        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, "receipt.json"), "w") as stream:
            json.dump(rec, stream, indent=1)
        if args.peer:
            receipt.pair_receipts([os.path.join(dest, "receipt.json"), args.peer])
            rec = json.loads(receipt.read_text(os.path.join(dest, "receipt.json")))
            reasons = rec["invalid_reasons"]
        all_valid = all_valid and rec["valid"]
        print("== %s %s %s" % (rec["tag"], "VALID" if rec["valid"] else "INVALID[" + "; ".join(reasons) + "]",
                               " ".join("%s=%g" % kv for kv in sorted(metrics.items()) if not kv[0].startswith("fail"))))

    return int(not all_valid)


if __name__ == "__main__":
    sys.exit(main())
