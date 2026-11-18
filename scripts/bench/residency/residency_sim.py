#!/usr/bin/env python3
"""Byte-weighted expert hit rate vs resident expert bytes, from a router trace (EXTR01).

Trace (written by the CUDA Qwen route's Config::expert_trace; qwen_decode_bench: STRIX_QWEN_EXPERT_TRACE=PATH): magic b"EXTR01\\0\\0", u32 layers, u32 experts, u32 top_k,
u32 reserved, then records: u32 stream, u32 step, u16 rows, u16 reserved, rows * layers * top_k u16 expert ids
(row-major: row, layer, slot). One record = one forward step (a decode row, a verify window or a prefill chunk).
Expert bytes per layer come from gguf_class_bytes.py --json.

Policies (all at the same resident byte budget):
  static_oracle  per-expert frequency over the whole trace (hindsight best static set)
  static_calib   frequency from the --calib traces only (what a shipped profile would give)
  lru            global LRU over (layer, expert), promote on every miss
  planner        decayed reuse score with hysteresis, promote only when the score margin pays the upload
                 (an approximation, not the production planner)
No production parity: global residency differs from uniform per-layer slots; decay, margin,
upload-byte limits, seeds and victim protection differ. Curves are not implementation predictions.
Reported: byte-weighted hit rate (bytes served from the card / active expert bytes), promoted bytes per step.

    python scripts/bench/residency/residency_sim.py --classes qwen-classes.json --trace a.extr [--calib b.extr]
           --budgets-gib 2,4,6,8,12,16,24,32
"""
import argparse
import collections
import json
import struct


def read_trace(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:6] != b"EXTR01":
        raise ValueError(path + ": not an EXTR01 trace")
    layers, experts, top_k, _ = struct.unpack_from("<4I", data, 8)
    pos, steps = 24, []
    while pos < len(data):
        stream, step, rows, _ = struct.unpack_from("<IIHH", data, pos)
        pos += 12
        n = rows * layers * top_k
        ids = struct.unpack_from("<%dH" % n, data, pos)
        pos += 2 * n
        # per layer: union of the rows' experts (an expert is read once per step for all its rows)
        per_layer = []
        for layer in range(layers):
            s = set()
            for r in range(rows):
                base = (r * layers + layer) * top_k
                s.update(ids[base:base + top_k])
            per_layer.append(s)
        steps.append((stream, rows, per_layer))
    return layers, experts, top_k, steps


def segments(steps):
    """Split at prefills (rows > 8): [(prefill_step, [decode steps])]. qwen_decode_bench order: serial T=1,
    windows T=2..8, then MTP verify windows (one segment each)."""
    out = []
    for st in steps:
        if st[1] > 8 or not out:
            out.append((st if st[1] > 8 else None, []))
        if st[1] <= 8:
            out[-1][1].append(st)
    return out


def union_stats(steps):
    """mean distinct experts per layer per step, by rows"""
    by = collections.defaultdict(list)
    for _, rows, per_layer in steps:
        by[rows].append(sum(len(s) for s in per_layer) / len(per_layer))
    return {r: sum(v) / len(v) for r, v in sorted(by.items())}


def expert_bytes(classes, layers, experts):
    per = classes["layer_expert_bytes"]
    return [per[str(layer)] / experts for layer in range(layers)]


def simulate(policy, steps, ebytes, budget, freq=None, decay=0.98, margin=2.0, prior=None):
    resident, used = set(), 0.0
    hit_b = tot_b = promo_b = 0.0
    if policy.startswith("static") or (prior and policy in ("planner", "lru", "prefill_static")):
        if not policy.startswith("static"):
            freq = prior
        order = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        for key, _ in order:
            b = ebytes[key[0]]
            if used + b <= budget:
                resident.add(key)
                used += b
    lru = collections.OrderedDict((k, True) for k in resident)
    score = collections.defaultdict(float)
    for _, _, per_layer in steps:
        for layer, s in enumerate(per_layer):
            for e in s:
                key, b = (layer, e), ebytes[layer]
                tot_b += b
                if key in resident:
                    hit_b += b
                    if policy == "lru":
                        lru.move_to_end(key)
                elif policy == "lru":
                    while used + b > budget and lru:
                        old, _ = lru.popitem(last=False)
                        resident.discard(old)
                        used -= ebytes[old[0]]
                    if used + b <= budget:
                        resident.add(key)
                        lru[key] = True
                        used += b
                        promo_b += b
        if policy == "planner":
            for key in list(score):
                score[key] *= decay
            for layer, s in enumerate(per_layer):
                for e in s:
                    score[(layer, e)] += 1.0
            # candidates: hot non-resident keys; victims: coldest resident keys (score per byte)
            cands = sorted((k for k in score if k not in resident), key=lambda k: (-score[k], k))[:64]
            victims = sorted(resident, key=lambda k: (score[k], k))
            vi = 0
            for c in cands:
                b = ebytes[c[0]]
                if used + b <= budget:
                    resident.add(c)
                    used += b
                    promo_b += b
                    continue
                if vi >= len(victims):
                    break
                v = victims[vi]
                # saved time per resident byte = reuse rate / CPU bandwidth, so scores compare directly; the
                # candidate must beat the victim by `margin` expected reuses over the decay horizon, which pays
                # its upload (CPU and upload bandwidths are about equal here) plus hysteresis against thrashing
                if score[c] - score[v] * ebytes[v[0]] / b > margin:
                    resident.discard(v)
                    used -= ebytes[v[0]]
                    vi += 1
                    if used + b <= budget:
                        resident.add(c)
                        used += b
                        promo_b += b
                else:
                    break
    n = max(1, len(steps))
    return hit_b / max(tot_b, 1.0), promo_b / n, tot_b / n


LABELS = ["serial", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "mtp"]


def main():
    ap = argparse.ArgumentParser(description="Approximate residency policies; curves do not predict the production planner.")
    ap.add_argument("--classes", required=True)
    ap.add_argument("--trace", required=True, action="append")
    ap.add_argument("--budgets-gib", default="2,4,6,8,12,16,24,32")
    ap.add_argument("--policies", default="static_oracle,static_calib,prefill_static,lru,planner")
    ap.add_argument("--segments", default="serial,mtp", help="bench phases to report (" + ",".join(LABELS) + ")")
    a = ap.parse_args()
    with open(a.classes) as f:
        classes = json.load(f)
    traces = []
    for p in a.trace:
        layers, experts, top_k, st = read_trace(p)
        segs = segments(st)
        # qwen_decode_bench runs len(LABELS) phases per prompt; a file may hold several prompts
        for k in range(0, len(segs), len(LABELS)):
            traces.append(("%s#%d" % (p, k // len(LABELS)), segs[k:k + len(LABELS)]))
    ebytes = expert_bytes(classes, layers, experts)

    def freq_of(ss):
        fr = collections.Counter()
        for _, _, per_layer in ss:
            for layer, s in enumerate(per_layer):
                for e in s:
                    fr[(layer, e)] += 1
        return fr
    budgets = [float(b) for b in a.budgets_gib.split(",")]
    print("APPROXIMATION: global slots, decay/margin, upload limits and seed/victim rules differ from production")
    print("experts/layer %d top_k %d; expert MB min %.3f max %.3f; budgets GiB %s" % (
        experts, top_k, min(ebytes) / 1e6, max(ebytes) / 1e6, a.budgets_gib))
    for want in a.segments.split(","):
        idx = LABELS.index(want)
        rows = []
        for ti, (p, segs) in enumerate(traces):
            if idx >= len(segs):
                continue
            pre, dec = segs[idx]
            if not dec:
                continue
            # calibration = the same phase of every OTHER trace (a shipped profile never saw this prompt)
            calib = [st for tj, (_, sj) in enumerate(traces) if tj != ti and idx < len(sj) for st in sj[idx][1]]
            prior = freq_of([pre]) if pre else None
            us = union_stats(dec)
            res = {}
            for pol in a.policies.split(","):
                fr = (freq_of(dec) if pol == "static_oracle" else freq_of(calib) if pol == "static_calib"
                      else prior if pol == "prefill_static" else None)
                if pol == "static_calib" and not calib:
                    continue
                if pol in ("prefill_static", "planner", "lru") and prior is None:
                    continue
                res[pol] = [simulate("static" if pol == "prefill_static" else pol, dec, ebytes, b * 2**30, fr,
                                     prior=prior) for b in budgets]
            name = p.replace("\\", "/").split("/")[-1]
            rows.append((name, len(dec), us, res))
        print("== phase %s" % want)
        for name, n, us, res in rows:
            print("%s: %d steps, distinct experts/layer by rows %s" % (
                name, n, " ".join("%d:%.1f" % (r, v) for r, v in us.items())))
            for pol, cells in res.items():
                print("  %-14s " % pol + " ".join("%5.3f/%-4d" % (h, round(pm / 2**20)) for h, pm, _ in cells))
        # byte-weighted mean over traces per policy
        pols = sorted({pol for _, _, _, r in rows for pol in r})
        for pol in pols:
            cells = [r[pol] for _, _, _, r in rows if pol in r]
            mean = [sum(c[i][0] for c in cells) / len(cells) for i in range(len(budgets))]
            print("  MEAN %-9s " % pol + " ".join("%5.3f     " % m for m in mean))
    print("cell = byte-weighted hit rate / promoted MiB per step; budgets = resident expert bytes")


if __name__ == "__main__":
    main()
