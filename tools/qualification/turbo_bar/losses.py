"""Three losses (E->R, E->T, R->T) from per-arm logits on identical positions, with null controls.

All arithmetic float64 over the FULL vocabulary. Per position, for an ordered pair a->b:
  KL(a||b); top-1 disagreement (argmax differs, lowest ID on ties); dNLL = NLL_b(label) - NLL_a(label).
Positions are averaged per parent; the parent is the statistical unit. The bar per metric m is
  d_parent = m(E->T) - m(E->R) - delta_m,  verdict against limit 0.

  python -m tools.qualification.turbo_bar.losses layout --corpus DIR --spread DIR [--continuation RUNDIR] \
      --arms E,R,T:x,Erep[,control] --out layout.json
  python -m tools.qualification.turbo_bar.losses analyze --layout layout.json --out losses.json
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np

from tools.qualification.option_qual import metrics, records
from tools.qualification.turbo_bar import spool, stats
from tools.qualification.turbo_bar.common import canon_sha, read_ids, read_json, safe_name, sha256_file, write_json

METRICS = ("kl", "dis", "dnll")
METRIC_LABEL = {"kl": "KL", "dis": "top-1 disagreement", "dnll": "dNLL"}
MARGIN_KEY = {"kl": "kl", "dis": "top1_disagreement", "dnll": "dnll"}


def iter_rows(path, fmt):
    if fmt == "spool":
        for pos, _inp, label, z in spool.read(path):
            yield pos, label, z
    elif fmt == "steps":
        for r in records.read_steps(path):
            if not np.isfinite(r.logits).all():
                raise ValueError("non-finite logits in step record")
            yield r.position, r.label, r.logits
    else:
        raise ValueError(f"unknown format {fmt}")


def pair_name(a, b):
    return f"{a}>{b}"


def bitwise_equal(a, b):
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def case_sums(case, pairs):
    """Stream the arms of one case in lockstep; returns dict(n, sums{pair:{metric}}, bitwise{pair})."""
    arms = sorted({x for p in pairs for x in p})
    if case.get("format", "spool") == "spool":
        ref_inputs = None
        for a in arms:
            inp = spool.header(case["files"][a])["inputs"]
            if ref_inputs is None:
                ref_inputs = inp
            elif not np.array_equal(ref_inputs, inp):
                raise ValueError(f"{case['case']}: arm {a} has different prompt ids; refusing to compare")
    its = {a: iter_rows(case["files"][a], case.get("format", "spool")) for a in arms}
    sums = {pair_name(*p): dict.fromkeys(METRICS, 0.0) for p in pairs}
    bitwise = {pair_name(*p): True for p in pairs}
    n = 0
    positions = []
    sentinel = object()
    while True:
        rows = {a: next(it, sentinel) for a, it in its.items()}
        ended = [a for a, r in rows.items() if r is sentinel]
        if ended:
            if len(ended) != len(arms):
                raise ValueError(f"{case['case']}: arms have different row counts")
            break
        pos = {r[0] for r in rows.values()}
        lab = {r[1] for r in rows.values()}
        if len(pos) != 1 or len(lab) != 1:
            raise ValueError(f"{case['case']}: arms disagree on position or label at row {n}")
        positions.append(pos.pop())
        label = lab.pop()
        lp = {a: metrics.log_softmax(r[2]) for a, r in rows.items()}
        top = {a: int(np.argmax(r[2])) for a, r in rows.items()}
        nll = {a: float(-lp[a][label]) for a in arms}
        for a, b in pairs:
            s = sums[pair_name(a, b)]
            s["kl"] += metrics._kl(lp[a], lp[b], np.exp(lp[a]))
            s["dis"] += int(top[a] != top[b])
            s["dnll"] += nll[b] - nll[a]
            if bitwise[pair_name(a, b)] and not bitwise_equal(rows[a][2], rows[b][2]):
                bitwise[pair_name(a, b)] = False
        n += 1
    return {"n": n, "sums": sums, "bitwise": bitwise, "positions": positions}


ROUTE_FIELDS = ("teacher_forced_tiled_lm_head", "teacher_logits_scope", "teacher_logits_rows")


def route_fields(case, arm):
    """Engine receipt route fields recorded by the runner in the sidecar JSON (PATH.json or PATH minus .bin + .json)."""
    from pathlib import Path
    p = Path(case["files"][arm])
    for cand in (p.with_suffix(".json"), Path(str(p) + ".json")):
        if cand.is_file():
            side = read_json(cand)
            rc = side.get("receipt") or {}
            return {k: rc.get(k, side.get(k)) for k in ROUTE_FIELDS if rc.get(k, side.get(k)) is not None}
    return {}


def parent_means(recs, pair, flt):
    acc = {}
    for r in recs:
        if all(r.get(k) == v for k, v in flt.items()):
            s = acc.setdefault(r["parent"], [dict.fromkeys(METRICS, 0.0), 0])
            for m in METRICS:
                s[0][m] += r["sums"][pair][m]
            s[1] += r["n"]
    return {p: {m: v[0][m] / v[1] for m in METRICS} for p, v in acc.items() if v[1]}


def group_filters(recs):
    lengths = sorted({r["length"] for r in recs})
    sources = sorted({r["source"] for r in recs})
    cats = sorted({r["category"] for r in recs})
    out = {"all": {}}
    out.update({f"length={x}": {"length": x} for x in lengths})
    out.update({f"source={x}": {"source": x} for x in sources})
    out.update({f"category={x}": {"category": x} for x in cats})
    out.update({f"length={x},source={s}": {"length": x, "source": s} for x in lengths for s in sources})
    return out


def combine(verdicts):
    v = list(verdicts)
    if "FAIL" in v:
        return "FAIL"
    return "PASS" if v and all(x == "PASS" for x in v) else "INCONCLUSIVE"


def freeze_layout(layout, manifest, corpus):
    layout["frozen_inputs"] = {"manifest": copy.deepcopy(manifest), "corpus": copy.deepcopy(corpus),
                               "cases_sha256": canon_sha(layout["cases"])}
    return layout


def validate_layout(layout):
    # Unbound synthetic analyses carry no experiment identity. A saved experiment
    # must retain its original inputs; old layouts must be rebuilt, not trusted.
    if not (layout.get("manifest") or {}).get("core_sha256"):
        if layout.get("arm_sha256") or layout.get("frozen_inputs"):
            raise ValueError("missing frozen manifest identity")
        return
    from tools.qualification.turbo_bar import manifest as mf
    from tools.qualification.turbo_bar.run_logits import arm_spec, validate_output
    frozen = layout.get("frozen_inputs")
    if not frozen:
        raise ValueError("layout lacks frozen inputs; rebuild the layout")
    m, corpus = frozen["manifest"], frozen["corpus"]
    identity = {"version": m["version"], "core_sha256": mf.core_sha(m)}
    if (layout["manifest"] != identity or m["frozen"]["core_sha256"] != identity["core_sha256"]
            or (layout.get("E", "E"), layout.get("R", "R"), layout.get("Erep", "Erep")) != ("E", "R", "Erep")
            or canon_sha(corpus) != m["protocol"]["corpus_sha256"]
            or layout["margins"] != m["protocol"]["margins"]
            or layout["bootstrap"] != m["protocol"]["bootstrap"]
            or layout["arm_sha256"] != {a: canon_sha(arm_spec(m, a)) for a in layout["T"]}
            or canon_sha(layout["cases"]) != frozen["cases_sha256"]):
        raise ValueError("layout differs from frozen inventory or protocol")
    sources = {c["source"] for c in layout["cases"]}
    inventory = {f"{c['case']}/{source}": c for c in corpus["cases"] for source in sources}
    if (len(layout["cases"]) != len(inventory)
            or {c["case"] for c in layout["cases"]} != inventory.keys()):
        raise ValueError("layout differs from frozen case inventory")
    arms = {layout.get("E", "E"), layout.get("R", "R"), layout.get("Erep", "Erep"), *layout["T"]}
    if layout.get("control"):
        arms.add(layout["control"])
    for c in layout["cases"]:
        original = inventory[c["case"]]
        if (set(c["files"]) != arms or c["source"] not in ("spread", "continuation")
                or any(c[k] != original[k] for k in ("parent", "category", "length", "split"))):
            raise ValueError("layout differs from frozen case inventory")
        for arm, path in c["files"].items():
            saved = c["validated_files"][arm]
            contract = saved["contract"]
            spec = arm_spec(m, arm)
            keys = ("prompt_ids", "positions") if c["source"] == "spread" else ("prompt_ids", "cont_ids")
            expected = {"manifest_core_sha256": identity["core_sha256"], "manifest_version": m["version"],
                        "arm": arm, "arm_sha256": canon_sha(spec), "case": original["case"], "source": c["source"],
                        "binary_sha256": spec["binary"]["sha256"], "weights_sha256": [w["sha256"] for w in spec["weights"]],
                        "identity": spec["identity"], "prefill_rows": contract["prefill_rows"],
                        **{k + "_sha256": original[k + "_sha256"] for k in keys}}
            side = validate_output(path, expected)
            if (contract != expected or side["payload_sha256"] != saved["payload_sha256"]
                    or sha256_file(Path(path).with_suffix(".json")) != saved["sidecar_sha256"]):
                raise ValueError("logit payload or sidecar replaced since layout creation")


def analyze(layout):
    validate_layout(layout)
    E, R, Ts = layout.get("E", "E"), layout.get("R", "R"), layout["T"]
    erep, ctrl = layout.get("Erep", "Erep"), layout.get("control")
    margins = layout.get("margins", {})
    boot = layout.get("bootstrap", {"B": 10000, "seed": 0})
    B, seed = boot["B"], boot["seed"]
    pairs = [(E, R)] + [(E, t) for t in Ts] + [(R, t) for t in Ts] + [(E, erep)]
    if ctrl:
        pairs.append((E, ctrl))
    recs = []
    for c in layout["cases"]:
        r = case_sums(c, pairs)
        r.update({k: c[k] for k in ("case", "parent", "category", "length", "source")})
        r["split"] = c.get("split")
        recs.append(r)
    out = {"manifest": layout.get("manifest"), "status": "VALID", "invalid_reason": None, "groups": {}, "bar": {},
           "overall": {}, "summary": {}, "nulls": {}, "n_cases": len(recs), "margins": margins}
    out["arm_sha256"] = layout.get("arm_sha256", {})
    # null control (a): E vs E-repeat bitwise identical
    npair = pair_name(E, erep)
    exact = all(r["bitwise"][npair] and all(r["sums"][npair][m] == 0 for m in METRICS) for r in recs)
    out["nulls"]["e_repeat_bitwise_identical"] = bool(exact)
    # lever inactive on the tiled spread route: spread bitwise == E while continuation differs
    inactive = {}
    for t in Ts:
        tp = pair_name(E, t)
        sp = [r for r in recs if r["source"] == "spread"]
        co = [r for r in recs if r["source"] == "continuation"]
        inactive[t] = bool(sp and co and all(r["bitwise"][tp] for r in sp) and not all(r["bitwise"][tp] for r in co))
    out["not_measured"] = {t: "lever inactive on the spread route" for t in Ts if inactive[t]}
    routes = {}
    for c in layout["cases"]:
        for arm in c["files"]:
            rf = route_fields(c, arm)
            if rf:
                routes.setdefault(f"{c['source']}/{arm}", rf)
    out["routes"] = routes
    complete = bool(recs)
    units = {}
    for c, r in zip(layout["cases"], recs):
        key = (r["parent"], r["length"])
        sources = units.setdefault(key, set())
        if r["source"] in sources:
            raise ValueError("duplicate parent/length/source")
        sources.add(r["source"])
        expected = c.get("expected_positions")
        complete &= bool(expected) and r["positions"] == expected
        complete &= r["split"] in ("SEARCH", "CONFIRM")
        if r["source"] == "continuation":
            complete &= r["positions"] == list(range(r["length"] - 1, r["length"] + 255))
    complete &= all(s == {"spread", "continuation"} for s in units.values())
    for parent in {r["parent"] for r in recs}:
        if len({r["split"] for r in recs if r["parent"] == parent}) != 1:
            raise ValueError("parent belongs to multiple splits")
    out["complete_sources"] = bool(complete)
    filters = {g: dict(flt, split="CONFIRM") for g, flt in group_filters(recs).items()}
    filters["search"] = {"split": "SEARCH"}
    for g, flt in filters.items():
        entry = {"n_parents": 0, "pairs": {}}
        for p in pairs:
            pm = parent_means(recs, pair_name(*p), flt)
            entry["n_parents"] = max(entry["n_parents"], len(pm))
            entry["pairs"][pair_name(*p)] = {}
            for m in METRICS:
                mean, lo, hi = stats.ci([v[m] for v in pm.values()], B=B, seed=seed)
                entry["pairs"][pair_name(*p)][m] = {"mean": mean, "lo": lo, "hi": hi, "n": len(pm)}
        out["groups"][g] = entry
        for t in Ts:
            if inactive[t] and flt.get("source") == "spread":
                out["bar"].setdefault(t, {})[g] = {m: {"verdict": "NOT-MEASURED", "message":
                                                       "lever inactive on the spread route", "n": 0} for m in METRICS}
                continue
            tflt = dict(flt, source="continuation") if inactive[t] and "source" not in flt else flt
            er, et = parent_means(recs, pair_name(E, R), tflt), parent_means(recs, pair_name(E, t), tflt)
            units = sorted(set(er) & set(et))
            out["bar"].setdefault(t, {})[g] = {
                m: stats.verdict([et[u][m] - er[u][m] - margins.get(MARGIN_KEY[m], 0.0) for u in units],
                                 0.0, B=B, seed=seed) for m in METRICS}
    # placebo (c): CI of E>E-repeat dNLL contains 0
    pm = parent_means(recs, npair, {})
    _mean, lo, hi = stats.ci([v["dnll"] for v in pm.values()], B=B, seed=seed)
    out["nulls"]["placebo_dnll_ci"] = {"lo": lo, "hi": hi, "contains_zero": bool(lo <= 0 <= hi)}
    if ctrl:
        out["nulls"]["control_noise_floor"] = out["groups"]["all"]["pairs"][pair_name(E, ctrl)]
    if not exact:
        out.update(status="INVALID", invalid_reason="E not reproducible (E vs E-repeat not bitwise identical)")
    elif not out["nulls"]["placebo_dnll_ci"]["contains_zero"]:
        out.update(status="INVALID", invalid_reason="placebo failed (E->E-repeat dNLL CI excludes 0)")
    keys = [g for g in out["groups"] if g == "all" or g.startswith("length=") and "," not in g]
    for t in Ts:
        vs = [out["bar"][t][g][m]["verdict"] for g in keys for m in METRICS]
        v = combine(vs)
        if inactive[t] and v == "PASS":
            v = "INCONCLUSIVE"  # spread NOT-MEASURED never counts toward a PASS
        if not complete:
            v = "INCONCLUSIVE"
            out["not_measured"][t] = "incomplete prescribed sources, positions or split metadata"
        out["overall"][t] = "INVALID" if out["status"] == "INVALID" else v
        a = out["groups"]["all"]["pairs"][pair_name(E, t)]
        out["summary"][t] = {"kl_mean": a["kl"]["mean"], "dnll_mean": a["dnll"]["mean"],
                             "dis_mean": a["dis"]["mean"], "verdict": out["overall"][t],
                             "note": out["not_measured"].get(t)}
    return out


def build_layout(corpus_dir, spread_dir, arms, cont_dir=None, margins=None, manifest=None, boot=None, fmt_cont="steps"):
    from pathlib import Path
    from tools.qualification.turbo_bar.run_logits import output_contract, validate_output
    corpus = read_json(Path(corpus_dir) / "corpus.json")
    if not manifest or not manifest["frozen"] or canon_sha(corpus) != manifest["protocol"]["corpus_sha256"]:
        raise ValueError("layout requires the frozen corpus/manifest")
    cases = []
    labels = {a["label"] if isinstance(a, dict) else a: a for a in arms}
    for c in corpus["cases"]:
        for source, base, fmt in (("spread", spread_dir, "spool"), ("continuation", cont_dir, fmt_cont)):
            if not base:
                continue
            files = {}
            for a in arms:
                files[a] = str(Path(base) / (f"{c['case']}.{safe_name(a)}.bin" if source == "spread"
                                             else f"logits/{c['case']}.{safe_name(a)}.bin"))
            missing = [p for p in files.values() if not Path(p).is_file()]
            if missing:
                raise FileNotFoundError(f"{c['case']} {source}: missing {missing[0]} (+{len(missing) - 1})")
            expected_positions = (read_ids(Path(corpus_dir) / c["positions"]) if source == "spread"
                                  else list(range(c["length"] - 1, c["length"] + 255)))
            if source == "spread" and len(expected_positions) != corpus["n_positions"]:
                raise ValueError("spread selection count differs from prescribed corpus")
            if source == "continuation" and len(read_ids(Path(corpus_dir) / c["cont_ids"])) != 256:
                raise ValueError("qualification requires 256 continuation tokens")
            validated_files = {}
            for arm, path in files.items():
                side = read_json(Path(path).with_suffix(".json"))
                expected = output_contract(manifest, arm, c, corpus_dir, source,
                                           side.get("contract", {}).get("prefill_rows") if source == "spread" else None)
                side = validate_output(path, expected)
                validated_files[arm] = {"contract": expected, "payload_sha256": side["payload_sha256"],
                                        "sidecar_sha256": sha256_file(Path(path).with_suffix(".json"))}
            cases.append({"case": f"{c['case']}/{source}", "parent": c["parent"], "category": c["category"],
                          "length": c["length"], "source": source, "format": fmt, "files": files,
                          "split": c["split"], "expected_positions": expected_positions, "validated_files": validated_files})
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("layout")
    p.add_argument("--corpus", required=True)
    p.add_argument("--spread", required=True)
    p.add_argument("--continuation")
    p.add_argument("--arms", required=True, help="E,R,T:a,T:b,Erep[,control] (file label = sanitized arm id)")
    p.add_argument("--control")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--layout", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "layout":
        from tools.qualification.turbo_bar import manifest as mf
        m = mf.load(a.manifest)
        names = a.arms.split(",")
        Ts = [n for n in names if n.startswith("T:")]
        cases = build_layout(a.corpus, a.spread, names, a.continuation, manifest=m)
        proto = m["protocol"]
        layout = {"E": "E", "R": "R", "T": Ts, "Erep": "Erep", "control": a.control, "cases": cases,
                           "arm_sha256": {n: canon_sha(m["arms"][n]) for n in Ts},
                           "margins": proto["margins"], "bootstrap": proto["bootstrap"],
                           "manifest": {"version": m["version"], "core_sha256": (m["frozen"] or {}).get("core_sha256")}}
        write_json(a.out, freeze_layout(layout, m, read_json(Path(a.corpus) / "corpus.json")))
        return 0
    res = analyze(read_json(a.layout))
    write_json(a.out, res)
    print(res["status"], {t: v for t, v in res["overall"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
