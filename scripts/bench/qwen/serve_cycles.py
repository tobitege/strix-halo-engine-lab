#!/usr/bin/env python3
"""Per-burst joint MTP cycle table from a served Qwen log (STRIX_MTP_BATCH_LOG=2) and serve_bench results.

Usage: serve_cycles.py --log SRV.ERR --burst NAME:FIRST:LAST:RES.JSON[#K][:EVICTED_MS[:CLOCK_MHZ:BUSY_PCT]] ... [--pairs A,B]

FIRST/LAST are the log line numbers (1-based, inclusive) that bracket the burst. For each burst it prints, over the
joint cycles whose lane count equals the burst's concurrency (the decode window), the cycles, emitted tokens per cycle
and per lane, cycle ms (draft + target), window tok/s (emitted / cycle ms), the host gap between consecutive cycles
(wall-clock step minus cycle ms: admission, streaming and per-lane host work outside the cycle), the owner topologies
(#R2/#R4/#R8 counts, with each one's mean target ms and emissions), and from the result JSON the steady and
end-to-end tok/s and the ordered per-request id hashes. --pairs A,B names two arm prefixes: bursts NAME=A<k> and
B<k> of one concurrency are compared (ids per request, window tok/s, wall tok/s). Only the standard library.

The result burst is the one whose live lanes equal the C of NAME (c<C>-...): use min(concurrency, slots), or
observed steady.active_lanes backed by steady cycles when slots are unknown. Exactly one must match, or
RES.JSON#K names burst K, which must match too; otherwise the run stops (no silent first burst).
The identity verdict is fail-closed: IDENTICAL only with both arms present, every burst successful (0 failed), the
same request count everywhere and a complete non-null hash per request; otherwise UNAVAILABLE (reason). With --pairs
the exit status is 1 unless every compared concurrency is IDENTICAL.
"""
import argparse
import collections
import json
import re
import statistics

PLAN = re.compile(r"joint plan wall_ms=(\d+) sessions=(\d+) probe=(\d) expected=\S+ est_ms=\S+ members=(\S+) "
                  r"draft_ms=([\d.]+) target_ms=([\d.]+)")


def cycles(lines):
    out = []
    for line in lines:
        m = PLAN.search(line)
        if not m:
            continue
        members = []
        for part in m.group(4).split(","):
            fields = part.split(":")
            tail = fields[-1].split("/")
            members.append(dict(rows=int(tail[5]), accepted=int(tail[6]), emitted=int(tail[7])))
        out.append(dict(wall=int(m.group(1)), lanes=int(m.group(2)), probe=int(m.group(3)), members=members,
                        draft=float(m.group(5)), target=float(m.group(6))))
    return out


def topo(members):
    k = [0, 0, 0]
    for m in members:
        k[2 if m["rows"] == 8 else 1 if m["rows"] == 4 else 0] += 1
    return "%d/%d/%d" % tuple(k)


def live_lanes(burst):
    slots = burst.get("slots")
    c = burst.get("concurrency")
    if type(c) is not int or c <= 0:
        return None
    if type(slots) is int and slots > 0:
        return min(c, slots)
    steady = burst.get("steady")
    if isinstance(steady, dict):
        active, count = steady.get("active_lanes"), steady.get("cycles")
        if type(active) is int and 0 < active <= c and type(count) is int and count > 0:
            return active
    return None


def select_burst(res, lanes, index=None):
    """The result burst for a log window of `lanes` live lanes; ValueError when absent, ambiguous or mismatched."""
    bursts = res["bursts"] if isinstance(res, dict) and "bursts" in res else [res]
    if index is not None:
        if not 0 <= index < len(bursts):
            raise ValueError("burst index %d out of range (%d bursts)" % (index, len(bursts)))
        b = bursts[index]
        active = live_lanes(b)
        if active is None:
            raise ValueError("burst %d: unavailable live-lane evidence" % index)
        if active != lanes:
            raise ValueError("burst %d has %s live lanes, the window has %d" % (index, active, lanes))
        return b
    matches = [b for b in bursts if live_lanes(b) == lanes]
    if len(matches) != 1:
        unavailable = sum(live_lanes(b) is None for b in bursts)
        reason = "; %d bursts have unavailable live-lane evidence" % unavailable if unavailable else ""
        raise ValueError("%d bursts with %d live lanes (of %d): name one with RES.JSON#K%s"
                         % (len(matches), lanes, len(bursts), reason))
    return matches[0]


def identity_verdict(arm_a, arm_b):
    """Fail-closed token identity over the bursts (rows) of two arms."""
    if not arm_a or not arm_b:
        return "UNAVAILABLE (arm missing)"
    counts = set()
    for r in arm_a + arm_b:
        ids, requests, failed = r.get("ids"), r.get("requests"), r.get("failed")
        if ids is None or requests is None or failed is None:
            return "UNAVAILABLE (%s: no result evidence)" % r["name"]
        if failed != 0:
            return "UNAVAILABLE (%s: %s failed)" % (r["name"], failed)
        if not requests or len(ids) != requests:
            return "UNAVAILABLE (%s: %d hashes for %s requests)" % (r["name"], len(ids), requests)
        if any(not isinstance(h, str) or not h for h in ids):
            return "UNAVAILABLE (%s: null hash)" % r["name"]
        counts.add(requests)
    if len(counts) != 1:
        return "UNAVAILABLE (request counts differ: %s)" % sorted(counts)
    first = arm_a[0]["ids"]
    return "IDENTICAL" if all(r["ids"] == first for r in arm_a + arm_b) else "DIFFER"


def burst_row(name, cyc, res, lanes, index=None):
    window = [c for c in cyc if c["lanes"] == lanes]
    row = dict(name=name, lanes=lanes, cycles=len(window), all_cycles=len(cyc))
    if window:
        ms = sum(c["draft"] + c["target"] for c in window)
        em = sum(sum(m["emitted"] for m in c["members"]) for c in window)
        row.update(window_tok_s=1000.0 * em / ms, emit_per_cycle=em / len(window), emit_per_lane=em / len(window) / lanes,
                   cycle_ms=ms / len(window), draft_ms=statistics.mean(c["draft"] for c in window),
                   target_ms=statistics.mean(c["target"] for c in window),
                   rows_per_cycle=statistics.mean(sum(max(m["rows"], 2) for m in c["members"]) for c in window),
                   probes=sum(c["probe"] for c in window))
        gaps = [b["wall"] - a["wall"] - (b["draft"] + b["target"]) for a, b in zip(window, window[1:])
                if b["wall"] - a["wall"] < 2000]
        row["gap_ms"] = statistics.mean(gaps) if gaps else None
        by = collections.defaultdict(list)
        for c in window:
            by[topo(c["members"])].append(c)
        row["topologies"] = {k: (len(v), statistics.mean(c["target"] for c in v),
                                 statistics.mean(sum(m["emitted"] for m in c["members"]) for c in v))
                             for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))}
    row["ids"] = row["requests"] = row["failed"] = None
    if res:
        b = select_burst(res, lanes, index)
        row["steady_tok_s"] = b["steady"].get("aggregate_tok_s")
        row["e2e_tok_s"] = b["client"].get("end_to_end_tok_s")
        row["ttft_p50_s"] = b["client"].get("ttft_p50_s")
        row["ids"] = b["client"].get("ids_sha256")
        row["requests"] = b["client"].get("requests")
        row["failed"] = b["client"].get("failed")
    return row


def fmt(v, d=1):
    return "n/a" if v is None else ("%.*f" % (d, v))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--burst", action="append", default=[])
    p.add_argument("--pairs", default="")
    args = p.parse_args()
    with open(args.log, errors="replace") as f:
        lines = f.read().split("\n")
    rows = []
    for spec in args.burst:
        parts = spec.split(":")
        name, first, last, path = parts[0], int(parts[1]), int(parts[2]), parts[3]
        index = None
        if "#" in path:
            path, k = path.rsplit("#", 1)
            index = int(k)
        try:
            with open(path) as f:
                res = json.load(f)
        except (OSError, ValueError):
            res = None
        lanes = int(re.search(r"c(\d+)", name).group(1))
        try:
            row = burst_row(name, cycles(lines[first - 1:last]), res, lanes, index)
        except (ValueError, KeyError, TypeError) as e:
            raise SystemExit("%s: %s: %s" % (name, path, e))
        row["evicted_ms"] = int(parts[4]) if len(parts) > 4 and parts[4] else None
        row["clock_mhz"] = float(parts[5]) if len(parts) > 5 and parts[5] else None
        row["busy_pct"] = float(parts[6]) if len(parts) > 6 and parts[6] else None
        rows.append(row)
    print("| burst | cycles@C | probes | emit/cycle | emit/lane | cycle ms | draft ms | target ms | rows/cycle | gap ms | "
          "window tok/s | steady tok/s | wall tok/s | TTFT p50 s | evicted ms | MHz | busy % | top topologies (n, target ms, emit) |")
    for r in rows:
        tops = "; ".join("%s:%d,%.1f,%.1f" % (k, *v) for k, v in list(r.get("topologies", {}).items())[:3])
        print("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["name"], r["cycles"], r.get("probes", "n/a"), fmt(r.get("emit_per_cycle"), 2), fmt(r.get("emit_per_lane"), 2),
            fmt(r.get("cycle_ms")), fmt(r.get("draft_ms")), fmt(r.get("target_ms")), fmt(r.get("rows_per_cycle")),
            fmt(r.get("gap_ms"), 2), fmt(r.get("window_tok_s")), fmt(r.get("steady_tok_s")), fmt(r.get("e2e_tok_s")),
            fmt(r.get("ttft_p50_s"), 2), r["evicted_ms"], fmt(r["clock_mhz"], 0), fmt(r["busy_pct"], 0), tops))
    ok = True
    if args.pairs:
        a, b = args.pairs.split(",")
        groups = collections.defaultdict(lambda: {a: [], b: []})
        for r in rows:
            m = re.match(r"c(\d+)-(%s|%s)\d+$" % (re.escape(a), re.escape(b)), r["name"])
            if m:
                groups[int(m.group(1))][m.group(2)].append(r)
        for lanes, g in sorted(groups.items()):
            verdict = identity_verdict(g[a], g[b])
            ok = ok and verdict == "IDENTICAL"
            clean = all(r["evicted_ms"] == 0 for r in g[a] + g[b])
            def mean(arm, key):
                vals = [r.get(key) for r in g[arm] if r.get(key) is not None]
                return statistics.mean(vals) if vals else None
            wa, wb, ea, eb = mean(a, "window_tok_s"), mean(b, "window_tok_s"), mean(a, "e2e_tok_s"), mean(b, "e2e_tok_s")
            delta = lambda x, y: 100.0 * (y / x - 1.0) if x and y else None
            print("C%d %s n=%d/%d: window %s -> %s (%s %%), wall %s -> %s (%s %%), ids %s, evictions %s" % (
                lanes, "%s->%s" % (a, b), len(g[a]), len(g[b]), fmt(wa), fmt(wb), fmt(delta(wa, wb), 2), fmt(ea), fmt(eb),
                fmt(delta(ea, eb), 2), verdict, "none" if clean else "PRESENT"))
        if not groups:
            print("no %s/%s bursts: ids UNAVAILABLE (arm missing)" % (a, b))
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
