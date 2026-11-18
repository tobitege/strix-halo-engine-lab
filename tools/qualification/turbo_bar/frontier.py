"""Speed / quality frontier of T variants and the selection rule.

  python -m tools.qualification.turbo_bar.frontier F.json
F.json: list of {name, speed: [ABAB pair ratios T/E or tok/s samples], quality: {kl_mean, dnll_mean, verdict}}.
Pareto: maximise median speed, minimise loss (KL(E->T) mean, then dNLL; lexicographic). Timing noise is the
95 % t interval over a variant's samples. Selection: among PASS variants take the fastest by median; every PASS
variant whose speed CI overlaps the fastest's CI is "within timing noise"; choose the least damaging of those
(lowest KL, tiebreak dNLL).
"""
import json
import sys

import numpy as np

from tools.qualification.turbo_bar import stats


def _loss(v):
    return (v["quality"]["kl_mean"], v["quality"]["dnll_mean"])


def analyze(variants):
    rows = []
    for v in variants:
        m, lo, hi = stats.t_ci(v["speed"])
        rows.append({"name": v["name"], "median": float(np.median(v["speed"])), "ci": [lo, hi], "n": len(v["speed"]),
                     "kl_mean": v["quality"]["kl_mean"], "dnll_mean": v["quality"]["dnll_mean"],
                     "verdict": v["quality"]["verdict"], "loss": _loss(v)})
    for r in rows:
        r["pareto"] = not any((o["median"] >= r["median"] and o["loss"] <= r["loss"]
                               and (o["median"] > r["median"] or o["loss"] < r["loss"])) for o in rows if o is not r)
    passing = [r for r in rows if r["verdict"] == "PASS"]
    selected, noise = None, []
    if passing:
        fastest = max(passing, key=lambda r: r["median"])
        noise = [r for r in passing if r is fastest or (r["ci"][1] >= fastest["ci"][0] and r["ci"][0] <= fastest["ci"][1])]
        selected = min(noise, key=lambda r: r["loss"])
    for r in rows:
        r["within_noise"] = r in noise
        r.pop("loss")
    return {"variants": rows, "selected": selected["name"] if selected else None,
            "fastest_pass": max(passing, key=lambda r: r["median"])["name"] if passing else None}


def table(res):
    lines = ["| variant | speed median | 95% CI | KL(E->T) | dNLL | verdict | pareto | within noise | selected |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(res["variants"], key=lambda r: -r["median"]):
        lines.append(f"| {r['name']} | {r['median']:.4g} | {r['ci'][0]:.4g}..{r['ci'][1]:.4g} | {r['kl_mean']:.4g} | "
                     f"{r['dnll_mean']:.4g} | {r['verdict']} | {'yes' if r['pareto'] else 'no'} | "
                     f"{'yes' if r['within_noise'] else 'no'} | {'SELECTED' if r['name'] == res['selected'] else ''} |")
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    res = analyze(json.loads(open(argv[0], encoding="utf-8").read()))
    print(table(res))
    print("selected:", res["selected"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
