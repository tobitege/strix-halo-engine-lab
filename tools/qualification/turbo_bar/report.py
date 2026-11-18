"""Receipt and REPORT.md of the TURBO quality bar.

  python -m tools.qualification.turbo_bar.report --manifest M --losses L.json --tasks DIR --retrieval DIR \
      --frontier F.json --out DIR
--tasks / --retrieval are directories with scored.<arm>.jsonl (client score). Verdicts: PASS / FAIL /
INCONCLUSIVE / INVALID. Never approve by absence of significance.
"""
import argparse
import sys
from pathlib import Path

from tools.qualification.turbo_bar import frontier as frontier_mod
from tools.qualification.turbo_bar import manifest as mf
from tools.qualification.turbo_bar import stats
from tools.qualification.turbo_bar.common import (canon_sha, http_binding, read_json, read_jsonl, safe_name,
                                                  unique_rows, write_json)
from tools.qualification.turbo_bar.losses import METRIC_LABEL, METRICS, combine


def _load_scored(d, arm, manifest, suite):
    p = Path(d) / f"scored.{safe_name(arm)}.jsonl"
    if not p.is_file():
        raise ValueError(f"missing {suite} results for {arm}")
    binding = http_binding(manifest, arm, suite)
    expected = manifest["protocol"]["suites"][suite]["items"]
    rows = unique_rows(read_jsonl(p))
    if set(rows) != set(expected):
        raise ValueError(f"{suite} {arm}: exact frozen suite coverage required")
    if suite == "tasks":
        required = {(k, s) for k in ("code", "math", "needle", "json") for s in ("explore", "confirm")}
        if {(r["kind"], r["split"]) for r in expected.values()} != required:
            raise ValueError("frozen task suite lacks required kinds/splits")
    else:
        required = {(L, mode, c, cut) for L in manifest["protocol"]["contexts"]
                    for mode in ("new", "restored") for c in (1, 8) for cut in ("straddle", "mid")}
        probe = manifest["protocol"].get("retrieval_probe_context")
        if probe:
            required.update((probe, mode, c, cut) for mode, c, cut in
                            (("new", 1, "straddle"), ("new", 1, "mid"), ("restored", 1, "straddle"),
                             ("new", 8, "straddle")))
        cells = {(r["length"], r["mode"], r["concurrency"], r["cut"]) for r in expected.values()}
        if not required <= cells:
            raise ValueError("frozen retrieval suite lacks required matrix cells")
    for key, r in rows.items():
        if r.get("arm") != arm or r.get("binding") != binding:
            raise ValueError(f"{suite} {arm}: stale experiment/arm binding")
        if any(r.get(k) != v for k, v in expected[key].items()):
            raise ValueError(f"{suite} {arm}: item/check or metadata mismatch")
        if r.get("server_identity") != binding["server_identity"]:
            raise ValueError(f"{suite} {arm}: unverified server identity")
        if type(r.get("ok")) is not int or r["ok"] not in (0, 1) or (r.get("error") and r["ok"]):
            raise ValueError("invalid score or successful failed request")
    return rows


def _paired(rows_by_arm, t, margin, boot):
    E, R, T = rows_by_arm.get("E"), rows_by_arm.get("R"), rows_by_arm.get(t)
    if not (E and R and T):
        return None
    if set(E) != set(R) or set(E) != set(T):
        raise ValueError("paired arms have different item coverage")
    ids = sorted(E)
    return ids, E, R, T


def task_results(d, manifest, ts, margin, boot):
    out = {}
    by = {a: _load_scored(d, a, manifest, "tasks") for a in ["E", "R"] + ts}
    for t in ts:
        res = {"per_kind": {}}
        p = _paired(by, t, margin, boot)
        if p is None:
            out[t] = {"per_kind": {}, "verdict": "INCONCLUSIVE", "note": "missing arm results"}
            continue
        ids, E, R, T = p
        kinds = sorted({E[i]["kind"] for i in ids})
        confirm_v = []
        for k in kinds:
            res["per_kind"][k] = {}
            for split in ("explore", "confirm"):
                sel = [i for i in ids if E[i]["kind"] == k and E[i]["split"] == split]
                if not sel:
                    continue
                e, r, tt = ([x[i]["ok"] for i in sel] for x in (E, R, T))
                vs_r = stats.paired_binary(r, tt, limit=margin, **boot)   # d = s_R - s_T (T loss minus R loss)
                vs_e = stats.paired_binary(e, tt, limit=0.0, **boot)      # d = s_E - s_T (informational)
                res["per_kind"][k][split] = {"n": len(sel), "rate": {"E": sum(e) / len(e), "R": sum(r) / len(r),
                                                                      "T": sum(tt) / len(tt)},
                                             "T_vs_R_loss": vs_r, "T_vs_E_loss": vs_e}
                if split == "confirm":
                    confirm_v.append(vs_r["verdict"])
        res["verdict"] = combine(confirm_v) if confirm_v else "INCONCLUSIVE"
        out[t] = res
    return out


def retrieval_results(d, ts, margin, boot, manifest):
    out = {}
    by = {a: _load_scored(d, a, manifest, "retrieval") for a in ["E", "R"] + ts}
    for t in ts:
        p = _paired(by, t, margin, boot)
        if p is None:
            out[t] = {"matrix": [], "per_length": {}, "verdict": "INCONCLUSIVE", "note": "missing arm results"}
            continue
        ids, E, R, T = p
        cells = {}
        for i in ids:
            r = E[i]
            cells.setdefault((r["length"], r["mode"], f"C{r['concurrency']}", r["cut"]), []).append(i)
        matrix = [{"length": k[0], "mode": k[1], "concurrency": k[2], "cut": k[3], "n": len(v),
                   "measurement": "MEASURED" if all(x[i].get("measured") is True for x in (E, R, T) for i in v)
                   else "NOT-MEASURED",
                   "E": sum(E[i]["ok"] for i in v) / len(v), "R": sum(R[i]["ok"] for i in v) / len(v),
                   "T": sum(T[i]["ok"] for i in v) / len(v),
                   "prefix_hit_T": [T[i].get("prefix_hit") for i in v if T[i].get("prefix_hit") is not None][:1]}
                  for k, v in sorted(cells.items())]
        per_len, vs = {}, []
        for L in sorted({E[i]["length"] for i in ids}):
            sel = [i for i in ids if E[i]["length"] == L]
            per_len[L] = {"n": len({E[i]["group"] for i in sel}),
                          "T_vs_R_loss": grouped_retrieval(E, R, T, sel, margin, boot),
                          "T_vs_E_loss": grouped_retrieval(E, E, T, sel, 0.0, boot), "probe": L >= 131072}
            if L < 131072:
                vs.append(per_len[L]["T_vs_R_loss"]["verdict"])
        measured = all(c["measurement"] == "MEASURED" for c in matrix if c["length"] < 131072)
        out[t] = {"matrix": matrix, "per_length": per_len,
                  "verdict": combine(vs) if vs and measured else "INCONCLUSIVE"}
    return out


def grouped_retrieval(E, A, T, ids, margin, boot):
    """Equal weight per frozen matrix cell; independent haystacks within cells."""
    cells = {}
    for i in ids:
        r = E[i]
        key = (r["mode"], r["concurrency"], r["cut"])
        cells.setdefault(key, {}).setdefault(r["group"], []).append(A[i]["ok"] - T[i]["ok"])
    samples = [[sum(v) / len(v) for v in groups.values()] for groups in cells.values()]
    # A weighted envelope of cell intervals is conservative and preserves equal
    # cell weights. Bootstrap units inside each interval are entire haystacks.
    parts = [stats.verdict(s, margin, level=1 - 0.05 / len(samples), **boot) for s in samples]
    out = {"n": sum(len(s) for s in samples), "n_cells": len(samples), "limit": margin,
           "mean": sum(p["mean"] for p in parts) / len(parts), "lo": None, "hi": None,
           "verdict": "INCONCLUSIVE"}
    if all(p["lo"] is not None for p in parts):
        out["lo"] = sum(p["lo"] for p in parts) / len(parts)
        out["hi"] = sum(p["hi"] for p in parts) / len(parts)
        out["verdict"] = "PASS" if out["hi"] < margin else "FAIL" if out["lo"] > margin else "INCONCLUSIVE"
    return out


def _f(x, nd=5):
    return "n/a" if x is None else f"{x:.{nd}g}"


def _ci(d):
    return f"{_f(d['mean'])} [{_f(d['lo'])}, {_f(d['hi'])}]"


def build(manifest_path, losses_path, tasks_dir=None, retrieval_dir=None, frontier_path=None):
    m = mf.load(manifest_path)
    L = read_json(losses_path)
    prob = []
    if not m["frozen"]:
        prob.append("manifest is not frozen")
    lm = L.get("manifest") or {}
    if m["frozen"] and (lm.get("version") != m["version"] or lm.get("core_sha256") != m["frozen"]["core_sha256"]):
        prob.append("losses were produced under another manifest version (mixing versions is refused)")
    boot = {"B": m["protocol"]["bootstrap"]["B"], "seed": m["protocol"]["bootstrap"]["seed"]}
    margins = m["protocol"]["margins"]
    ts = sorted(L["overall"])
    if any(not t.startswith("T:") or t not in m["arms"] for t in ts):
        prob.append("losses contain an unknown or superseded T arm")
    if any(L.get("arm_sha256", {}).get(t) != canon_sha(m["arms"][t]) for t in ts if t in m["arms"]):
        prob.append("losses lack matching immutable T specification hashes")
    tk, rt = {}, {}
    try:
        tk = task_results(tasks_dir, m, ts, margins["task_success"], boot) if tasks_dir else {}
        rt = retrieval_results(retrieval_dir, ts, margins["retrieval_success"], boot, m) if retrieval_dir else {}
    except (ValueError, KeyError) as e:
        prob.append(str(e))
    overall = {}
    for t in ts:
        comp = {"logits": L["overall"][t], "tasks": tk.get(t, {}).get("verdict", "INCONCLUSIVE"),
                "retrieval": rt.get(t, {}).get("verdict", "INCONCLUSIVE")}
        if prob or L["status"] == "INVALID":
            v = "INVALID"
        elif "FAIL" in comp.values():
            v = "FAIL"
        else:
            v = "PASS" if all(x == "PASS" for x in comp.values()) else "INCONCLUSIVE"
        overall[t] = {"verdict": v, "components": comp}
    fr = None
    if frontier_path:
        variants = read_json(frontier_path)
        for v in variants:
            key = v["name"] if v["name"] in overall else "T:" + v["name"]
            q = v.setdefault("quality", {})
            if key in L["summary"]:
                q["kl_mean"] = L["summary"][key]["kl_mean"]
                q["dnll_mean"] = L["summary"][key]["dnll_mean"]
            else:
                raise ValueError("frontier arm lacks current measured losses")
            if key in overall:
                q["verdict"] = overall[key]["verdict"]
            else:
                q["verdict"] = "INCONCLUSIVE"
        fr = frontier_mod.analyze(variants)
    return {"model": m["model"], "manifest_version": m["version"],
            "manifest_core_sha256": (m["frozen"] or {}).get("core_sha256"), "problems": prob,
            "status": "INVALID" if prob or L["status"] == "INVALID" else "VALID",
            "invalid_reason": "; ".join(prob) or L.get("invalid_reason"), "margins": margins, "losses": L,
            "tasks": tk, "retrieval": rt, "frontier": fr, "overall": overall}


def markdown(r):
    o = [f"# TURBO quality bar: {r['model']} (manifest v{r['manifest_version']})", "",
         f"Status: **{r['status']}**" + (f" ({r['invalid_reason']})" if r["invalid_reason"] else ""), "",
         "## Verdicts", "", "| variant | verdict | logits | tasks | retrieval |", "|---|---|---|---|---|"]
    for t, v in r["overall"].items():
        c = v["components"]
        o.append(f"| {t} | **{v['verdict']}** | {c['logits']} | {c['tasks']} | {c['retrieval']} |")
    L = r["losses"]
    o += ["", "## Null controls", "", f"- E vs E-repeat bitwise identical: {L['nulls']['e_repeat_bitwise_identical']}",
          f"- placebo dNLL CI: {L['nulls']['placebo_dnll_ci']}"]
    if "control_noise_floor" in L["nulls"]:
        o.append("- control arm noise floor (E->control): " +
                 ", ".join(f"{METRIC_LABEL[k]} {_ci(L['nulls']['control_noise_floor'][k])}" for k in METRICS))
    for t, why in L.get("not_measured", {}).items():
        o.append(f"- {t}: spread E->T NOT-MEASURED ({why}); spread results never count toward PASS")
    o += ["", "## Losses (mean [95 % CI] over parents; higher = worse)", ""]
    pairs = list(L["groups"]["all"]["pairs"])
    for g, ge in L["groups"].items():
        o += [f"### {g} ({ge['n_parents']} parents)", "", "| pair | " + " | ".join(METRIC_LABEL[k] for k in METRICS) + " |",
              "|---|---|---|---|"]
        for p in pairs:
            nm = [t for t in L["not_measured"] if p == f"E>{t}" and "source=spread" in g]
            o.append(f"| {p}{' (NOT-MEASURED)' if nm else ''} | " + " | ".join(_ci(ge["pairs"][p][k]) for k in METRICS) + " |")
        o.append("")
    o += ["## Bar: d = m(E->T) - m(E->R) - delta, limit 0", ""]
    for t, groups in L["bar"].items():
        o += [f"### {t}", "", "| group | " + " | ".join(METRIC_LABEL[k] for k in METRICS) + " |", "|---|---|---|---|"]
        for g, vm in groups.items():
            o.append(f"| {g} | " + " | ".join(
                vm[k]["verdict"] if vm[k]["verdict"] == "NOT-MEASURED" else
                f"{vm[k]['verdict']} {_ci(vm[k]) if vm[k]['lo'] is not None else ''}"
                + (f" n_needed={vm[k]['n_needed']}" if vm[k]["verdict"] == "INCONCLUSIVE" else "") for k in METRICS) + " |")
        o.append("")
    o += ["## Tasks (success; loss d = s_R - s_T for T vs R, s_E - s_T for T vs E)", ""]
    for t, tr in r["tasks"].items():
        o += [f"### {t}: {tr['verdict']}", "", "| kind | split | n | E | R | T | T vs R loss | T vs E loss | discordant (R-only/T-only) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k, sp in tr["per_kind"].items():
            for s, x in sp.items():
                a, b = x["T_vs_R_loss"], x["T_vs_E_loss"]
                o.append(f"| {k} | {s} | {x['n']} | {x['rate']['E']:.3f} | {x['rate']['R']:.3f} | {x['rate']['T']:.3f} | "
                         f"{a['verdict']} {_ci(a) if a['lo'] is not None else ''} | {b['verdict']} "
                         f"{_ci(b) if b['lo'] is not None else ''} | {a['a_only']}/{a['b_only']} |")
        o.append("")
    o += ["## Retrieval matrix (success rate)", ""]
    for t, rr in r["retrieval"].items():
        o += [f"### {t}: {rr['verdict']}", "", "| length | mode | conc | cut | n | E | R | T |", "|---|---|---|---|---|---|---|---|"]
        for c in rr["matrix"]:
            o.append(f"| {c['length']} | {c['mode']} | {c['concurrency']} | {c['cut']} | {c['n']} | {c['E']:.2f} | "
                     f"{c['R']:.2f} | {c['T']:.2f} |")
        for ln, x in rr["per_length"].items():
            o.append(f"\nLength {ln}{' (probe, not in verdict)' if x['probe'] else ''}: T vs R {x['T_vs_R_loss']['verdict']}, "
                     f"T vs E {x['T_vs_E_loss']['verdict']}")
        o.append("")
    if r["frontier"]:
        o += ["## Frontier", "", frontier_mod.table(r["frontier"]), "", f"Selected variant: {r['frontier']['selected']}"]
    return "\n".join(o) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--losses", required=True)
    ap.add_argument("--tasks")
    ap.add_argument("--retrieval")
    ap.add_argument("--frontier")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = build(a.manifest, a.losses, a.tasks, a.retrieval, a.frontier)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "receipt.json", r)
    (out / "REPORT.md").write_text(markdown(r), encoding="utf-8", newline="\n")
    print(r["status"], {t: v["verdict"] for t, v in r["overall"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
