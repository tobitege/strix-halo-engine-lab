#!/usr/bin/env python3
"""gate32b_stats.py: multi-unit paired statistics and task scoring for the invariant-order gate (CPU only).

For every unit U (prompts from mkunits.py) it reads the sealed arms run/U-e (reference, adaptive), run/U-i (candidate,
invariant) and run/U-n0s1..N (adaptive + one-ulp L0 null) through fast_logit_gate.py (manifests and dump hashes are
checked), and reports
  - per unit: mean signed dNLL (candidate - reference, per labelled row) with a 95 % moving-block bootstrap CI
    (16-row blocks, rows share context), KL mean / q50 / q90, the same for every null;
  - pooled over units (equal weight): D = mean of unit means, 95 % CI t(units-1) (primary) and a hierarchical
    bootstrap (units, then row blocks); the pooled null values D_s and the null spread
    N = [min(mean-2sd, min D_s), max(mean+2sd, max D_s)];
  - kl_q50 / q90 of the candidate against each unit's nulls (units above every null; binomial tail at p = 1/(N+1));
  - task accuracy: the greedy answer lines A1..A7 decoded from rank 0's ids, strict per line, reference vs candidate
    paired (cluster bootstrap over units, exact McNemar) and every null arm;
  - warm vs cold (optional --warm U): logits and ids of U-ew / U-iw (warm-up session first) against the cold arms.
Decision (rule fixed before measuring, gate32b): ACCEPT iff (CI95(D) contains 0 or lies inside N) and CI95 of the
paired accuracy difference contains 0.

usage: gate32b_stats.py --kit KIT --gate fast_logit_gate.py --tokenizer tokenizer.json --units a,b,c [--nulls 3]
                        [--warm UNIT] [--out PREFIX]
writes PREFIX.txt (<= 60 lines) and PREFIX.json (per-row vectors kept for audit).
"""
import argparse, importlib.util, json, math, os, re, sys

import numpy as np

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
RNG = np.random.default_rng(20261115)


def load_gate(path):
    spec = importlib.util.spec_from_file_location("fast_logit_gate", path)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g


def block_boot(d, reps=4000, block=16):
    n = d.size
    nb = math.ceil(n / block)
    starts = RNG.integers(0, n - block + 1, size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]
    return d[idx].mean(axis=1)


def ci_pct(x):
    return float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))


def t_ci(v):
    v = np.asarray(v, float)
    h = T975[v.size - 1] * v.std(ddof=1) / math.sqrt(v.size)
    return float(v.mean() - h), float(v.mean() + h)


def binom_tail(k, n, p):
    return float(sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1)))


def gen_ids(log, unit):
    for line in open(log, errors="replace"):
        m = re.match(r"rank 0 (\S+): tokens fnv [0-9a-f]+ ids:(.*)", line)
        if m and m.group(1) == unit:
            return [int(x) for x in m.group(2).split()]
    return None


def score(tk, ids, answers):
    text = tk.decode(ids, skip_special_tokens=True) if ids else ""
    got = {}
    for m in re.finditer(r"A([1-7])\s*[=:]\s*([^\n]*)", text):
        got.setdefault(int(m.group(1)), m.group(2))
    return [int(a.lower() in got.get(k + 1, "").lower()) for k, a in enumerate(answers)], text


def ttft(log):
    m = re.findall(r"TTFT ([0-9.]+) s", open(log, errors="replace").read())
    return [float(x) for x in m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--nulls", type=int, default=3)
    ap.add_argument("--warm", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    g = load_gate(a.gate)
    import tokenizers
    tk = tokenizers.Tokenizer.from_file(a.tokenizer)
    K = a.kit
    run = lambda rid: os.path.join(K, "run", rid)
    units = a.units.split(",")
    L, J = [], {"units": {}}
    um, nm, uq, nq, task_e, task_i, task_n = [], [], [], [], [], [], {}
    L.append(f"gate32b stats: {len(units)} units, {a.nulls} nulls each, ref adaptive (e), cand invariant (i); cold arms")
    for u in units:
        tokens = np.fromfile(os.path.join(K, "prompts", u + ".u32"), "<u4").astype(np.int64)
        ans = json.load(open(os.path.join(K, "prompts", u + ".answers.json")))["answers"]
        ref = g.load_arm(run(f"{u}-e"), [0, 1])
        arms = {"i": g.load_arm(run(f"{u}-i"), [0, 1])}
        for s in range(1, a.nulls + 1):
            arms[f"n{s}"] = g.load_arm(run(f"{u}-n0s{s}"), [0, 1])
        res = {k: g.compare_logits(ref[0], ref[1], v[0], v[1], u, tokens) for k, v in arms.items()}
        d = {k: r["nll_cand"] - r["nll_ref"] for k, r in res.items()}
        st = {k: {"dnll": float(d[k].mean()), "kl": float(r["kl"].mean()), "q50": float(np.quantile(r["kl"], .5)),
                  "q90": float(np.quantile(r["kl"], .9)), "top1": float(r["agree"].mean())} for k, r in res.items()}
        lo, hi = ci_pct(block_boot(d["i"]))
        nulls = [st[f"n{s}"] for s in range(1, a.nulls + 1)]
        um.append(st["i"]["dnll"]); nm.append([x["dnll"] for x in nulls])
        uq.append((st["i"]["q50"], st["i"]["q90"])); nq.append([(x["q50"], x["q90"]) for x in nulls])
        sc = {}
        for k in ["e"] + list(arms):
            rid = f"{u}-{k}" if k in ("e", "i") else f"{u}-n0s{k[1:]}"
            sc[k], txt = score(tk, gen_ids(os.path.join(run(rid), "r0.log"), u), ans)
            sc[k + "_text"] = txt[:400]
        task_e.append(sc["e"]); task_i.append(sc["i"])
        for s in range(1, a.nulls + 1):
            task_n.setdefault(s, []).append(sc[f"n{s}"])
        tt = {k: ttft(os.path.join(run(f"{u}-{k}"), "r0.log")) for k in ("e", "i")}
        L.append(f"{u}: dNLL cand {st['i']['dnll']:+.4f} [{lo:+.4f},{hi:+.4f}] nulls "
                 + ",".join(f"{x['dnll']:+.4f}" for x in nulls)
                 + f" | KL cand {st['i']['kl']:.3f} nulls " + ",".join(f"{x['kl']:.3f}" for x in nulls)
                 + f" | q90 cand {st['i']['q90']:.2f} nulls " + ",".join(f"{x['q90']:.2f}" for x in nulls)
                 + f" | q50 cand {st['i']['q50']:.3f} nulls " + ",".join(f"{x['q50']:.3f}" for x in nulls)
                 + f" | top1 {st['i']['top1']:.3f} | task e {sum(sc['e'])}/7 i {sum(sc['i'])}/7 n "
                 + ",".join(str(sum(sc[f'n{s}'])) for s in range(1, a.nulls + 1))
                 + f" | TTFT cold e {tt['e']} i {tt['i']}")
        J["units"][u] = {"stats": st, "dnll_ci_block": [lo, hi], "rows_d": {k: v.tolist() for k, v in d.items()},
                         "rows_kl": {k: r["kl"].tolist() for k, r in res.items()}, "task": sc, "ttft": tt,
                         "identical_i": bool(res["i"]["identical"])}
    um, nm = np.array(um), np.array(nm)
    D = float(um.mean()); tlo, thi = t_ci(um)
    hb = np.array([np.mean([block_boot(np.array(J["units"][units[j]]["rows_d"]["i"]), reps=1)[0]
                            for j in RNG.integers(0, len(units), len(units))]) for _ in range(2000)])
    hlo, hhi = ci_pct(hb)
    Ds = nm.mean(axis=0)
    nlo = min(Ds.mean() - 2 * Ds.std(ddof=1), Ds.min()); nhi = max(Ds.mean() + 2 * Ds.std(ddof=1), Ds.max())
    diff = um - nm.mean(axis=1)
    dlo, dhi = t_ci(diff)
    p1 = (tlo <= 0 <= thi) or (nlo <= tlo and thi <= nhi)
    L.append(f"POOLED dNLL D {D:+.4f} CI95 t(5) [{tlo:+.4f},{thi:+.4f}] hier-boot [{hlo:+.4f},{hhi:+.4f}]")
    L.append(f"POOLED nulls D_s " + ",".join(f"{x:+.4f}" for x in Ds) + f" spread N [{nlo:+.4f},{nhi:+.4f}]"
             f" | cand - mean(null) per unit: mean {diff.mean():+.4f} CI95 [{dlo:+.4f},{dhi:+.4f}]")
    L.append(f"PART1 dNLL: {'PASS' if p1 else 'FAIL'} (CI contains 0: {tlo <= 0 <= thi}; CI inside N: {nlo <= tlo and thi <= nhi})")
    m = a.nulls
    for qi, qn in ((0, "q50"), (1, "q90")):
        above = sum(uq[j][qi] > max(x[qi] for x in nq[j]) for j in range(len(units)))
        L.append(f"KL {qn}: cand above all {m} nulls in {above}/{len(units)} units (chance {1 / (m + 1):.2f}/unit, "
                 f"P(>= {above}) = {binom_tail(above, len(units), 1 / (m + 1)):.3f})")
    te, ti = np.array(task_e), np.array(task_i)
    dd = (ti - te).mean(axis=1)
    bs = np.array([(ti - te)[RNG.integers(0, len(units), len(units))].mean() for _ in range(10000)])
    blo, bhi = ci_pct(bs)
    b = int(((te == 1) & (ti == 0)).sum()); c = int(((te == 0) & (ti == 1)).sum())
    pm = min(1.0, 2 * sum(math.comb(b + c, j) for j in range(min(b, c) + 1)) / 2 ** (b + c)) if b + c else 1.0
    p2 = blo <= 0 <= bhi
    L.append(f"TASK acc e {te.mean():.3f} ({int(te.sum())}/{te.size}) i {ti.mean():.3f} ({int(ti.sum())}/{ti.size}) "
             f"diff {float((ti - te).mean()):+.3f} CI95 [{blo:+.3f},{bhi:+.3f}] McNemar b={b} c={c} p={pm:.3f}; nulls "
             + ",".join(f"{np.array(v).mean():.3f}" for v in task_n.values()))
    per_q = (te.mean(axis=0), ti.mean(axis=0))
    L.append("TASK per question (Q1..Q5 depth 9/31/53/74/92 %, Q6 two-hop, Q7 max): e "
             + " ".join(f"{x:.2f}" for x in per_q[0]) + " | i " + " ".join(f"{x:.2f}" for x in per_q[1]))
    L.append(f"PART2 task: {'PASS' if p2 else 'FAIL'}")
    if a.warm:
        w = a.warm
        tokens = np.fromfile(os.path.join(K, "prompts", w + ".u32"), "<u4").astype(np.int64)
        for k in ("e", "i"):
            c0, c1 = g.load_arm(run(f"{w}-{k}"), [0, 1]), g.load_arm(run(f"{w}-{k}w"), [0, 1])
            r = g.compare_logits(c0[0], c0[1], c1[0], c1[1], w, tokens)
            ids0, ids1 = gen_ids(os.path.join(run(f"{w}-{k}"), "r0.log"), w), gen_ids(os.path.join(run(f"{w}-{k}w"), "r0.log"), w)
            L.append(f"WARM {w}-{k}w vs cold {w}-{k}: logits identical {r['identical']}, ids identical {ids0 == ids1}, "
                     f"TTFT cold {ttft(os.path.join(run(f'{w}-{k}'), 'r0.log'))} warm {ttft(os.path.join(run(f'{w}-{k}w'), 'r0.log'))}")
    acc = p1 and p2
    L.append(f"DECISION: {'ACCEPT (default ON)' if acc else 'STAYS OPT-IN'}")
    J.update({"pooled": {"D": D, "ci_t": [tlo, thi], "ci_hier": [hlo, hhi], "null_D": Ds.tolist(), "null_spread": [nlo, nhi],
                         "cand_minus_null": [float(diff.mean()), dlo, dhi]},
              "task": {"e": te.tolist(), "i": ti.tolist(), "ci": [blo, bhi], "mcnemar": [b, c, pm]},
              "part1": bool(p1), "part2": bool(p2), "accept": bool(acc)})
    txt = "\n".join(L) + "\n"
    print(txt, end="")
    if a.out:
        open(a.out + ".txt", "w").write(txt)
        json.dump(J, open(a.out + ".json", "w"))


if __name__ == "__main__":
    sys.exit(main())
