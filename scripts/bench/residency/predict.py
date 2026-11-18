#!/usr/bin/env python3
"""Tiered expert residency: predicted decode tok/s per card budget (the decision table of
docs/engine/memory/expert-residency.md). Inputs are measured, never assumed: a JSON with

  card_overhead_gib   context + driver reserve on the card (measured: card memory used at start minus weights)
  core_gib            always-resident weights in tiered mode (attention, GDN, router, shared, LM head, norms)
  mtp_gib             MTP head (resident when MTP is on)
  scratch_gib         activations, MoE scratch, graphs (measured: peak used - weights - KV at the run's context)
  kv_gib_per_1k       session bytes per 1,024 positions (driver session_bytes)
  layers, top_k       model
  expert_mb           mean routed expert size
  hit_curve           {"serial": [[resident_gib, byte_hit], ...], "mtp": [...]}   (residency_sim.py, planner policy)
  distinct            {"1": d1, "8": d8}   distinct experts per layer per step by rows (residency_sim.py)
  cpu_us_per_expert   {"1": us, "8": us}   one expert for 1 / 8 rows, at the miss counts of a decode layer
  exchange_us         one card -> host -> card exchange per layer (link_bench mapped-flag round trip)
  result_gbps         card zero-copy read of the CPU rawdots
  plain_ms            full-resident plain step (qwen_decode_bench serial T=1, graphs on)
  mtp                 {"window_ms": ..., "tokens_per_window": ..., "rows": 8}  full-resident MTP windows
  budgets_gib         card sizes to predict
  context_k           context to reserve (thousands of positions)

    python scripts/bench/residency/predict.py inputs.json
"""
import json
import sys


def interp(curve, x):
    pts = sorted(curve)
    if x <= pts[0][0]:
        return pts[0][1] * max(x, 0.0) / pts[0][0] if pts[0][0] > 0 else pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def step_ms(base_ms, rows_key, hit, m):
    misses = m["distinct"][rows_key] * (1.0 - hit)                       # per layer
    cpu = misses * m["cpu_us_per_expert"][rows_key]                      # us, CPU on the critical path
    rows = int(rows_key)
    result = misses * rows * 2560 * 4 / (m["result_gbps"] * 1e3) if misses else 0.0  # us (one rawdot per row/slot)
    per_layer = m["exchange_us"] + cpu + result
    return base_ms + m["layers"] * per_layer / 1e3, misses


def main():
    m = json.load(open(sys.argv[1]))
    print("card GiB | expert pool GiB (no MTP / MTP) | plain: hit, misses/layer, tok/s | MTP: hit, misses/layer, tok/s | auto")
    for b in m["budgets_gib"]:
        fixed = m["card_overhead_gib"] + m["core_gib"] + m["scratch_gib"] + m["kv_gib_per_1k"] * m["context_k"]
        pool = b - fixed
        pool_mtp = pool - m["mtp_gib"]
        if pool <= 0:
            print("%5.0f | does not fit (fixed %.2f GiB)" % (b, fixed))
            continue
        h1 = interp(m["hit_curve"]["serial"], pool)
        t1, mi1 = step_ms(m["plain_ms"], "1", h1, m)
        plain = 1000.0 / t1
        mtp_txt, mtp = "n/a", 0.0
        if pool_mtp > 0:
            h8 = interp(m["hit_curve"]["mtp"], pool_mtp)
            t8, mi8 = step_ms(m["mtp"]["window_ms"], "8", h8, m)
            mtp = 1000.0 * m["mtp"]["tokens_per_window"] / t8
            mtp_txt = "%.3f, %.1f, %.1f" % (h8, mi8, mtp)
        pick = "mtp" if mtp > plain else "plain"
        print("%5.0f | %6.2f / %6.2f | %.3f, %.1f, %.1f | %s | %s" % (b, pool, pool_mtp, h1, mi1, plain, mtp_txt, pick))


if __name__ == "__main__":
    main()
