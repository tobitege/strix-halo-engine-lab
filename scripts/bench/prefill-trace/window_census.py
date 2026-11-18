#!/usr/bin/env python3
"""Window-loop census of one prefill run from a rocprofv3 kernel trace (CSV).

Answers, before any scheduling change to a per-window launch loop (attention windows, index/select/flash/combine
sets): how long the loop spans, how much of it the GPU is idle, which kinds take the time, how the launches spread
over hardware queues, and how many compute units the CU-bound kernel (one CTA per CU) leaves free.

Usage:
  window_census.py TRACE.csv [--run N] [--split KERNEL] [--kinds a,b,...] [--fill KERNEL] [--cus N] [--edge-us U]
Runs are split at launches of --split (default embedding_kernel; --run -1 = the last one, the warm run of a
--repeat process). A cluster is a run of consecutive window-kind launches (by start); any other kernel ends it.
The fill sweep counts the CTAs of running --fill kernels capped at --cus: an over-count of resident CTAs, so the
reported deficit is a LOWER bound of the CU time that kernel leaves free inside the union of cluster spans.
"""
import argparse
import collections
import csv
import sys

DS41_KINDS = ("index_qprep", "index_scores", "select", "flash_rows", "attention_flash_combine", "attention_chunk",
              "attention_combine", "attention_f16wmma", "__amd_rocclr_fillBuffer")


def load(path):
    out = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, strict=True)
        required = {"Kernel_Name", "Start_Timestamp", "End_Timestamp", "Queue_Id", "LDS_Block_Size"}
        required.update(prefix + axis for prefix in ("Grid_Size_", "Workgroup_Size_") for axis in "XYZ")
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("trace is missing required kernel CSV columns")
        for r in reader:
            try:
                n = r["Kernel_Name"]
                dims = [int(r[prefix + axis]) for prefix in ("Grid_Size_", "Workgroup_Size_") for axis in "XYZ"]
                grid, wg = dims[0] * dims[1] * dims[2], dims[3] * dims[4] * dims[5]
                start, end = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
                queue, lds = int(r["Queue_Id"]), int(r["LDS_Block_Size"])
                if not n or min(dims) <= 0 or start < 0 or end < start or lds < 0 or grid % wg:
                    raise ValueError("invalid name, interval, dimensions or LDS size")
                out.append((start, end, n, queue, grid // wg, lds))
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid kernel CSV row %d: %s" % (reader.line_num, exc)) from exc
    if not out:
        raise ValueError("trace contains no kernel launches")
    out.sort()
    return out


def merged(iv):
    spans = []
    for s, en in sorted(iv):
        if en <= s:
            continue
        if not spans or s > spans[-1][1]:
            spans.append((s, en))
        else:
            spans[-1] = (spans[-1][0], max(spans[-1][1], en))
    return spans


def union(iv):
    return sum(e - s for s, e in merged(iv))


def census(k, kinds, fill, cus, edge_ns, cu_lds=65536):
    def win(n):
        return any(n.startswith(w) or n.find("::" + w) >= 0 for w in kinds)

    clusters, cur = [], []
    for x in k:
        if win(x[2]):
            cur.append(x)
        elif cur:
            clusters.append(cur)
            cur = []
    if cur:
        clusters.append(cur)
    spans = [(c[0][0], max(x[1] for x in c)) for c in clusters]
    loops = merged(spans)
    edges = merged([(s, min(e, s + edge_ns)) for s, e in spans] +
                   [(max(s, e - edge_ns), e) for s, e in spans])
    res = {"wall_ms": (max(x[1] for x in k) - k[0][0]) / 1e6, "launches": len(k), "clusters": len(clusters),
           "loop_launches": sum(len(c) for c in clusters)}
    res["span_ms"] = union(loops) / 1e6
    per = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    queues = collections.Counter()
    for c in clusters:
        for x in c:
            p = per[x[2]]
            p[0] += 1
            p[1] += x[1] - x[0]
            p[2][x[4]] += 1
            queues[x[3]] += 1
    res["kinds"] = sorted(((n, p[0], p[1] / 1e6, p[2].most_common(2)) for n, p in per.items()), key=lambda t: -t[2])
    res["queues"] = dict(queues)
    under = edge = inner = fill_busy = shared = busy = 0.0
    # Disjoint loop intervals count wall time once; include kernels from every queue,
    # even those launched outside a cluster or still running from an earlier cluster.
    for s0, e0 in loops:
        active = [(max(s0, x[0]), min(e0, x[1]), *x[2:]) for x in k if x[0] < e0 and x[1] > s0]
        busy += union([(x[0], x[1]) for x in active])
        fk = [x for x in active if fill in x[2]]
        # CTAs that cannot share a CU with a fill CTA (LDS above what the fill CTA leaves) occupy CUs too
        fill_lds = max((x[5] for x in fk), default=0)
        ok = [x for x in active if fill in x[2] or x[5] > cu_lds - fill_lds]
        ev2 = sorted([(x[0], x[4]) for x in ok] + [(x[1], -x[4]) for x in ok])
        level, t = 0, s0
        for ts, dv in ev2 + [(e0, 0)]:
            if ts > t and level < cus:
                shared += (ts - t) * (cus - level) / float(cus)
            t = max(t, ts)
            level += dv
        fill_busy += union([(x[0], x[1]) for x in fk])
        ev = sorted([(x[0], x[4]) for x in fk] + [(x[1], -x[4]) for x in fk])
        level, t = 0, s0
        loop_edges = [(max(s0, s), min(e0, e)) for s, e in edges if s < e0 and e > s0]
        for ts, dv in ev + [(e0, 0)]:
            if ts > t and level < cus:
                w = (cus - level) / float(cus)
                under += ts - t
                edge_time = sum(max(0, min(ts, e) - max(t, s)) for s, e in loop_edges)
                edge += edge_time * w
                inner += (ts - t - edge_time) * w
            t = max(t, ts)
            level += dv
    res.update(busy_ms=busy / 1e6, idle_ms=res["span_ms"] - busy / 1e6,
               fill_busy_ms=fill_busy / 1e6, under_ms=under / 1e6, deficit_edge_ms=edge / 1e6,
               deficit_inner_ms=inner / 1e6, deficit_ms=(edge + inner) / 1e6, free_ms=shared / 1e6)
    return res


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace")
    ap.add_argument("--run", type=int, default=-1)
    ap.add_argument("--split", default="embedding_kernel")
    ap.add_argument("--kinds", default=",".join(DS41_KINDS))
    ap.add_argument("--fill", default="flash_rows")
    ap.add_argument("--cus", type=int, default=40, help="compute units of the device (from its declaration)")
    ap.add_argument("--cu-lds", type=int, default=65536, help="LDS bytes per compute unit (from the device declaration)")
    ap.add_argument("--edge-us", type=int, default=300)
    a = ap.parse_args(argv)
    if a.cus <= 0 or a.cu_lds <= 0 or a.edge_us < 0:
        ap.error("--cus and --cu-lds must be positive; --edge-us must be nonnegative")
    kinds = tuple(w.strip() for w in a.kinds.split(","))
    if not all(kinds) or not a.fill.strip() or not a.split.strip():
        ap.error("--kinds, --fill and --split must contain nonempty kernel names")
    try:
        k = load(a.trace)
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        ap.error(str(exc))
    cut = [i for i, x in enumerate(k) if x[2].find(a.split) >= 0] or [0]
    runs = [(cut[j], cut[j + 1] if j + 1 < len(cut) else len(k)) for j in range(len(cut))]
    if not -len(runs) <= a.run < len(runs):
        ap.error("--run must be between %d and %d" % (-len(runs), len(runs) - 1))
    lo, hi = runs[a.run]
    r = census(k[lo:hi], kinds, a.fill, a.cus, a.edge_us * 1000, a.cu_lds)
    print("assumptions: CUs=%d, LDS=%d B/CU, kinds=%s, fill=%s; one fill CTA per CU"
          % (a.cus, a.cu_lds, ",".join(kinds), a.fill))
    print("run %d of %d: wall %.1f ms, launches %d; window clusters %d, launches %d, span %.1f ms, busy-union %.1f, idle %.1f"
          % (a.run % len(runs), len(runs), r["wall_ms"], r["launches"], r["clusters"], r["loop_launches"], r["span_ms"],
             r["busy_ms"], r["idle_ms"]))
    print("%-58s %7s %9s %8s %s" % ("kind", "n", "incl ms", "us/each", "CTAs (top)"))
    for n, cnt, ms, ctas in r["kinds"][:14]:
        print("%-58s %7d %9.1f %8.1f %s" % (n if len(n) <= 58 else n[:55] + "...", cnt, ms, ms * 1e3 / cnt, ctas))
    print("queues in window clusters:", r["queues"])
    print("%s busy-union %.1f ms; CTAs in flight < %d: %.1f ms; CU-idle equivalent >= %.1f ms (edges %d us: %.1f, interior %.1f)"
          % (a.fill, r["fill_busy_ms"], a.cus, r["under_ms"], r["deficit_ms"], a.edge_us, r["deficit_edge_ms"],
             r["deficit_inner_ms"]))
    print("CU-idle equivalent with the CTAs that cannot share a CU with %s counted (LDS rule, %d B/CU): >= %.1f ms"
          % (a.fill, a.cu_lds, r["free_ms"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
