#!/usr/bin/env python3
"""FAST-identity quality gate over teacher-forced logit dumps, with run identity.

Two subcommands (Python 3.9+, numpy):

  seal     Run on the node that wrote one rank's dumps, right after the engine exits. Checks that every expected
           dump exists, is fresh, has the declared row count and only finite values, then writes
           manifest-r<rank>.json (schema strix-fast-dump/v1): run id, arm, rank, binary sha256, the environment the
           arm was launched with, per prompt the token-file sha256 and per dump its bytes / sha256 / width.
           A missing, short, stale or already sealed dump is an error: nothing is written.

  compare  Compares a reference run directory with a candidate one (each holding one manifest per rank). Every
           dump is re-hashed against its manifest; run ids must differ between arms and agree between the ranks of
           an arm; prompts, rows and vocabulary widths must agree. Reports per prompt and pooled KL(ref || cand),
           top-1 agreement and the NLL change on the real next tokens, and, when both arms carry block-selection
           dumps, the selected ids that differ per layer and position. Exit 0 = gate passed, 1 = gate failed,
           2 = identity or input error (no verdict). The dNLL limit is absolute (--max-dnll) unless
           --dnll-rule floor is given: then it is mean + 2 sd (n - 1) of >= 3 --null-arm runs, allowed only for
           the engine / profile pairs in DNLL_POLICY; the absolute verdict stays a reference. DS41 FAST prefill:
           dNLL only, layer-0 nulls. MiniMax FAST prefill: onset-matched nulls (one ulp at the candidate's first
           FAST layer) and six checks (KL, top-1, capped dNLL, KL quantiles, sign test, copy rows), see MM_FLOOR.

  witness  Launcher-side producer of the execution witness (schema strix-fast-execution/v1) that `seal
           --execution-witness` binds and the floor rule requires. Reads the rank's engine log after the engine
           exits: today the MiniMax adapter (route receipt -> engine, profile, numerical configuration, FAST layer
           set; null-control log lines -> control), plus the binary, the model bundle and the completed dumps.

Dump layout inside a run directory (PREFIX is the --prefix value, default "q"):
  PREFIX-<prompt>-tf-r<rank>.f32   [rows][width] little-endian FP32: the head logits of the last `rows` prompt
                                   positions, this rank's contiguous vocabulary slice (ranks concatenate in order)
  PREFIX-<prompt>-sel-r<rank>.u32  optional selection records: 8 x u32 header {"SELD", 1, mode, position low / high,
                                   rows, heads, top-k}, rows x heads counts, rows x heads x top-k block ids; the
                                   n-th record with the same first position belongs to the rank's n-th layer
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import sys
import time

import numpy as np

SCHEMA = "strix-fast-dump/v1"
SEL_MAGIC = 0x444C4553


class GateError(Exception):
    """Identity or input defect: the comparison has no verdict."""


# dNLL rules per engine / profile. "abs" (every engine, the default): pooled dNLL <= --max-dnll. "floor": pooled dNLL <=
# mean + FLOOR_SIGMAS * sd (n - 1) of at least FLOOR_MIN_NULLS layer-0 null-control arms measured on the same prompts and
# binary as the candidate; KL and top-1 limits are unchanged. Only the DeepSeek V4.1 FAST prefill profile may use "floor":
# its layer-0 chaos floor (any one-ulp change at layer 0 re-routes MoE rows) already costs dNLL ~0.012 > 0.01. Adding an
# entry here is a policy decision, not a tuning knob.
# MiniMax FAST prefill ("floor" = MM_FLOOR below): one fp16-surviving ulp anywhere flips near-tie block selections and
# top-4 expert routes, so a layer-0 null lands at KL ~0.5 / top-1 ~88 % / dNLL +-0.1, far above any FAST layer set that
# starts later. Its nulls therefore start where the candidate starts (one ulp on the attention output of the candidate's
# first FAST layer, the tensor MSA flash replaces), and KL, top-1 and dNLL must stay inside their spread, with the dNLL
# limit capped, plus three checks that a wide floor cannot loosen: per-row KL quantiles, a sign test on the rows the
# candidate changes, and the next-token misses on copy rows.
DNLL_POLICY = {
    ("ds41", "fast-prefill"): ("abs", "floor"),
    ("minimax", "fast-prefill"): ("abs", "floor"),
}
FLOOR_MIN_NULLS = 3
FLOOR_SIGMAS = 2.0
WITNESS_SCHEMA = "strix-fast-execution/v1"
DS41_NULL_CONTROL = {"kind": "layer0-null", "layer": 0, "magnitude": 1, "unit": "ulp"}
MM_NULL_KIND = "attn-ulp-null"  # MiniMax null control: {"kind", "layer", "magnitude": 1, "unit": "ulp", "seed"}
MM_FLOOR = {
    "policy": "minimax-onset-floor-v1.2",
    "dnll_cap": 0.03,                   # the floor-relative dNLL limit never exceeds 3 x the absolute 0.01
    "kl_quantiles": (0.5, 0.9, 0.99),   # per-row KL quantiles: max(mu + 2 sd of the nulls' same quantile, tolerance)
    # v1.1: a quantile limit is never below these (nats). A one-ulp null's median row only carries rounding (KL ~1e-5,
    # sd ~1e-6), so a purely relative q50 demands bitwise arithmetic; 1e-3 = 1/20 of the absolute mean bar 0.02 still
    # catches a diffuse defect, and q90 / q99 are never held stricter than the absolute mean bar.
    "kl_quantile_tol": {"q50": 1e-3, "q90": 0.02, "q99": 0.02},
    # v1.2: q99 is reported, not required: the tail is chaos and one null site flips fewer near-ties than a FAST layer set
    # (measured: the 40-59 control, absolute PASS, sits at q99 0.025-0.029 vs its single-site nulls 0.017-0.022).
    "required_quantiles": ("q50", "q90"),
    "sign_min_abs": 1e-3,               # rows with |dNLL| >= this (nats) enter the sign test
    "sign_alpha": 0.01,                 # one-sided binomial P(X >= worse) vs p0 = max(0.5, pooled null worse share)
    "copy_k": 8,                        # copy row: the last k prompt tokens occurred before, followed by the true token
    "copy_slack_frac": 0.01,            # copy-row extra misses <= worst null + max(1, ceil(1 % of the copy rows))
}


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def check_fast_block(fast):
    """A witnessed FAST layer set: {"policy": str, "msa_flash_mode": int, "layers": sorted unique non-negative ints}."""
    if fast is None:
        return
    layers = fast.get("layers") if isinstance(fast, dict) else None
    if (not isinstance(fast, dict) or not isinstance(fast.get("policy"), str) or not is_int(fast.get("msa_flash_mode"))
            or not isinstance(layers, list) or not layers or not all(is_int(x) and x >= 0 for x in layers)
            or layers != sorted(set(layers))):
        raise GateError("execution witness: invalid FAST layer set")


def execution_witness(doc):
    """Validate producer evidence and its binding to this rank's sealed inputs/outputs."""
    w = doc.get("execution")
    if not isinstance(w, dict) or w.get("schema") != WITNESS_SCHEMA:
        raise GateError("missing or invalid execution witness")
    for key, expected in (("run_id", doc["run_id"]), ("rank", doc["rank"]),
                          ("binary_sha256", doc["binary"]["sha256"]), ("prompts", doc["prompts"])):
        if w.get(key) != expected:
            raise GateError(f"execution witness differs from sealed {key}")
    for key in ("model_sha256", "numerical_config_sha256"):
        value = w.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise GateError(f"execution witness: invalid {key}")
    if any(not isinstance(w.get(k), str) or not w[k] for k in ("engine", "profile")):
        raise GateError("execution witness: missing engine/profile")
    if "control" not in w or (w["control"] is not None and not isinstance(w["control"], dict)):
        raise GateError("execution witness: missing or invalid control identity")
    check_fast_block(w.get("fast"))
    cfg = w.get("config")
    if cfg is not None and (not isinstance(cfg, dict) or sorted(cfg) != ["levers", "switches"] or not all(
            isinstance(cfg[k], dict) and all(isinstance(a, str) and isinstance(b, str) for a, b in cfg[k].items())
            for k in cfg)):
        raise GateError("execution witness: invalid config (levers / switches maps of strings)")
    return w


def config_difference(ref, cand):
    """First difference outside the FAST allowlist. Receipts omit defaulted switches, so a missing entry cannot prove
    equivalence to an explicit value: refuse one-sided entries too. None = the witnessed maps agree."""
    lr, lc = ({k: v for k, v in c["levers"].items() if k not in MM_FAST_LEVERS} for c in (ref, cand))
    if lr != lc:
        return "lever " + next(k for k in sorted(set(lr) | set(lc)) if lr.get(k) != lc.get(k))
    sr, sc = ref["switches"], cand["switches"]
    for k in sorted((set(sr) | set(sc)) - MM_FAST_SWITCHES):
        if sr.get(k) != sc.get(k):
            return f"switch {k}={sr.get(k)!r} vs {sc.get(k)!r}"
    return None


def null_control_ok(engine, control, onset):
    """DS41: exactly the layer-0 one-ulp control. MiniMax: the one-ulp attention-output control at the candidate's onset."""
    if engine != "minimax":
        # JSON comparison also refuses booleans in place of integer layer/magnitude.
        return json.dumps(control, sort_keys=True) == json.dumps(DS41_NULL_CONTROL, sort_keys=True)
    return (isinstance(control, dict) and sorted(control) == ["kind", "layer", "magnitude", "seed", "unit"]
            and control["kind"] == MM_NULL_KIND and control["unit"] == "ulp" and is_int(control["magnitude"])
            and control["magnitude"] == 1 and is_int(control["layer"]) and control["layer"] == onset
            and is_int(control["seed"]) and control["seed"] >= 0)


def check_floor_execution(ref, cand, nulls, engine, profile):
    """Witness rules of the floor; returns the onset layer the nulls must start at (MiniMax) or None (DS41: layer 0)."""
    onset = None
    for i, (mr, mc) in enumerate(zip(ref, cand)):
        wr, wc = execution_witness(mr), execution_witness(mc)
        if (wc["engine"], wc["profile"]) != (engine, profile) or wc["control"] is not None:
            raise GateError("candidate execution witness disagrees with floor engine/profile or is a control")
        if (wr["engine"], wr["profile"], wr["control"]) != (engine, "exact", None):
            raise GateError("floor reference must witness the exact baseline without a control")
        if wr["model_sha256"] != wc["model_sha256"]:
            raise GateError("floor reference/candidate model identity differs")
        if engine == "minimax":
            if wc.get("fast") is None or wr.get("fast") is not None:
                raise GateError("minimax floor: the candidate must witness its FAST layer set and the reference none")
            if wr.get("config") is None or wc.get("config") is None:
                raise GateError("minimax floor: reference and candidate must witness their configuration (config)")
            diff = config_difference(wr["config"], wc["config"])
            if diff is not None:
                raise GateError(f"minimax floor: the candidate differs from the reference beyond its FAST layer set ({diff})")
            onset = wc["fast"]["layers"][0]
        for _, mn in nulls:
            wn = execution_witness(mn[i])
            if any(wn[k] != wr[k] for k in ("engine", "profile", "model_sha256", "numerical_config_sha256")):
                raise GateError("null execution witness differs from the reference model/baseline numerical configuration")
            if wn.get("fast") is not None:
                raise GateError("null execution witness carries a FAST layer set")
            if not null_control_ok(engine, wn["control"], onset):
                if engine == "minimax":
                    raise GateError(f"null execution witness must attest the MiniMax one-ulp attention-output control at "
                                    f"the candidate's first FAST layer {onset}")
                raise GateError("null execution witness must attest the DS41 one-ulp layer-0 control")
    if engine == "minimax":
        seeds = [mn[0]["execution"]["control"]["seed"] for _, mn in nulls]
        if len(set(seeds)) != len(seeds):
            raise GateError(f"minimax floor: null seeds repeat ({seeds}); the spread needs independent controls")
    return onset


def allowed_dnll_rules(engine, profile):
    return DNLL_POLICY.get((engine, profile), ("abs",))


def check_dnll_policy(rule, engine, profile, n_nulls):
    """Refuses a rule the policy table does not grant, or a floor without enough nulls (GateError = no verdict)."""
    if rule == "abs":
        if n_nulls:
            raise GateError("--null-arm is only used by --dnll-rule floor")
        return
    if not engine or not profile:
        raise GateError("--dnll-rule floor needs --engine and --profile (checked against DNLL_POLICY)")
    if rule not in allowed_dnll_rules(engine, profile):
        granted = ", ".join(f"{e}/{p}" for (e, p), rules in sorted(DNLL_POLICY.items()) if rule in rules)
        raise GateError(f"--dnll-rule {rule} is not allowed for engine {engine} profile {profile} (DNLL_POLICY grants it to: "
                        f"{granted}); use the absolute rule")
    if n_nulls < FLOOR_MIN_NULLS:
        raise GateError(f"--dnll-rule floor needs at least {FLOOR_MIN_NULLS} --null-arm null controls, got {n_nulls}")


def floor_limit(null_dnll):
    """(mean, sd with n - 1, limit = mean + FLOOR_SIGMAS * sd) of the null-control pooled dNLL values."""
    v = np.asarray(null_dnll, dtype=np.float64)
    if v.ndim != 1 or v.size < FLOOR_MIN_NULLS:
        raise GateError(f"floor rule needs at least {FLOOR_MIN_NULLS} null dNLL values, got {v.size}")
    if not np.all(np.isfinite(v)):
        raise GateError("floor rule: a null dNLL is not finite")
    mean, sd = float(v.mean()), float(v.std(ddof=1))
    return mean, sd, mean + FLOOR_SIGMAS * sd


def dnll_verdict(dnll, rule, max_dnll, null_dnll=()):
    """The dNLL part of the gate: {rule, limit, pass, abs_limit, abs_pass[, mean, sd, sigmas, n_nulls]}."""
    out = {"rule": rule, "abs_limit": max_dnll, "abs_pass": bool(dnll <= max_dnll)}
    if rule == "floor":
        mean, sd, limit = floor_limit(null_dnll)
        out.update(mean=mean, sd=sd, sigmas=FLOOR_SIGMAS, n_nulls=len(null_dnll), limit=limit)
    else:
        out["limit"] = max_dnll
    out["pass"] = bool(dnll <= out["limit"])
    return out


def copy_rows(tokens, rows, k=MM_FLOOR["copy_k"]):
    """Mask over the labelled rows (row i at prompt position total - rows + i predicts the next token; the last row has no
    label): True where the last k prompt tokens occurred before and the most recent earlier occurrence was followed by the
    true next token. Token-only, so the same rows for every arm; retrieving them is what attention does."""
    t = np.asarray(tokens, dtype=np.int64)
    total = len(t)
    first = total - rows
    mask = np.zeros(max(0, total - 1 - first), dtype=bool)
    # Only scored history keys need tracking; a long prompt must not grow this map beyond the scored tail.
    last = {t[p - k + 1:p + 1].tobytes(): None for p in range(max(k - 1, first), total - 1)}
    for p in range(k - 1, total - 1):
        key = t[p - k + 1:p + 1].tobytes()
        if p >= first:
            q = last.get(key)
            mask[p - first] = q is not None and t[q + 1] == t[p + 1]
        if key in last:
            last[key] = p
    return mask


def binom_sf(w, n, p):
    """P(X >= w) for X ~ Binomial(n, p), summed exactly in log space."""
    if w <= 0:
        return 1.0
    if w > n or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * lp + (n - j) * lq for j in range(w, n + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(x - m) for x in terms))


def arm_stats(results, copy_masks, policy=MM_FLOOR):
    """Pooled statistics of one arm against the reference: KL / top-1 / dNLL, per-row KL quantiles, the rows it makes worse
    or better on the true token, and next-token misses on copy rows (reference and arm)."""
    kl = np.concatenate([r["kl"] for r in results])
    d = np.concatenate([r["nll_cand"] - r["nll_ref"] for r in results])
    hit_ref = np.concatenate([r["hit_ref"] for r in results])
    hit = np.concatenate([r["hit_cand"] for r in results])
    cm = np.concatenate(copy_masks)
    if cm.size != d.size:
        raise GateError("copy-row mask does not match the labelled rows")
    eps = policy["sign_min_abs"]
    return {"rows": int(kl.size), "kl": float(kl.mean()), "top1": float(np.concatenate([r["agree"] for r in results]).mean()),
            "dnll": float(np.concatenate([r["nll_cand"] for r in results]).mean() - np.concatenate([r["nll_ref"] for r in results]).mean()),
            "kl_q": {f"q{round(q * 100)}": float(np.quantile(kl, q)) for q in policy["kl_quantiles"]},
            "worse": int((d >= eps).sum()), "better": int((d <= -eps).sum()), "copy_rows": int(cm.sum()),
            "copy_miss_ref": int((cm & ~hit_ref).sum()), "copy_miss": int((cm & ~hit).sum()),
            "copy_dnll": float(d[cm].mean()) if cm.any() else 0.0}


def mean_sd(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size < FLOOR_MIN_NULLS or not np.all(np.isfinite(v)):
        raise GateError(f"floor rule needs at least {FLOOR_MIN_NULLS} finite null values, got {v.tolist()}")
    return float(v.mean()), float(v.std(ddof=1))


def minimax_floor_checks(c, nulls, policy=MM_FLOOR, max_kl=0.02, min_top1=0.93, max_dnll=0.01):
    """The MiniMax floor checks of candidate stats `c` against the null stats; every check with required=True must pass.
    The floor only relaxes: KL / top-1 / dNLL limits are the looser of the absolute bar and the null spread (mean +- 2 sd),
    the dNLL limit capped at policy dnll_cap, so a tight or negative null floor never makes a bar stricter than absolute."""
    checks, k = [], FLOOR_SIGMAS

    def add(name, value, limit, ok, required=True, **extra):
        checks.append({"name": name, "value": value, "limit": limit, "pass": bool(ok), "required": required, **extra})

    m, s = mean_sd([n["kl"] for n in nulls])
    limit = max(max_kl, m + k * s)
    add("kl", c["kl"], limit, c["kl"] <= limit, mean=m, sd=s, floor=m + k * s, abs=max_kl)
    m, s = mean_sd([n["top1"] for n in nulls])
    limit = min(min_top1, m - k * s)
    add("top1", c["top1"], limit, c["top1"] >= limit, mean=m, sd=s, floor=m - k * s, abs=min_top1)
    m, s = mean_sd([n["dnll"] for n in nulls])
    limit = min(max(max_dnll, m + k * s), policy["dnll_cap"])
    add("dnll", c["dnll"], limit, c["dnll"] <= limit, mean=m, sd=s, floor=m + k * s, abs=max_dnll, cap=policy["dnll_cap"])
    for q in c["kl_q"]:
        m, s = mean_sd([n["kl_q"][q] for n in nulls])
        tol = policy["kl_quantile_tol"].get(q, 0.0)
        limit = max(m + k * s, tol)
        add(f"kl_{q}", c["kl_q"][q], limit, c["kl_q"][q] <= limit, required=q in policy["required_quantiles"],
            mean=m, sd=s, floor=m + k * s, tol=tol)
    nw, nc = sum(n["worse"] for n in nulls), sum(n["worse"] + n["better"] for n in nulls)
    p0 = max(0.5, nw / nc) if nc else 0.5
    changed = c["worse"] + c["better"]
    p = binom_sf(c["worse"], changed, p0)
    add("sign", p, policy["sign_alpha"], p >= policy["sign_alpha"], worse=c["worse"], better=c["better"], p0=p0)
    extra = c["copy_miss"] - c["copy_miss_ref"]
    worst = max(0, max(n["copy_miss"] - n["copy_miss_ref"] for n in nulls))
    slack = max(1, math.ceil(policy["copy_slack_frac"] * c["copy_rows"]))
    # Nonempty coverage is necessary; it does not establish a statistically powered sample size.
    observed = c["copy_rows"] > 0
    add("copy", extra, worst + slack, observed and extra <= worst + slack, rows=c["copy_rows"], miss_ref=c["copy_miss_ref"],
        miss=c["copy_miss"], worst_null=worst, power=observed,
        reason="observed copy rows" if observed else "insufficient copy evidence")
    return checks


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_pairs(items, what):
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key or key in out:
            raise GateError(f"{what}: expected unique NAME=VALUE, got {item!r}")
        out[key] = value
    return out


def read_selection(path):
    """Returns [(position, rows, heads, topk, mode, counts[rows, heads], ids[rows, heads, topk])] in file order."""
    raw = np.fromfile(path, "<u4")
    out, o = [], 0
    while o < raw.size:
        if raw.size - o < 8 or int(raw[o]) != SEL_MAGIC or int(raw[o + 1]) != 1:
            raise GateError(f"{path}: bad selection record header at word {o}")
        mode, pos = int(raw[o + 2]), int(raw[o + 3]) | (int(raw[o + 4]) << 32)
        rows, heads, topk = int(raw[o + 5]), int(raw[o + 6]), int(raw[o + 7])
        n = rows * heads
        if n == 0 or topk == 0 or raw.size - o - 8 < n * (1 + topk):
            raise GateError(f"{path}: short selection record at word {o}")
        cnt = raw[o + 8:o + 8 + n].reshape(rows, heads)
        ids = raw[o + 8 + n:o + 8 + n * (1 + topk)].reshape(rows, heads, topk)
        if int(cnt.max()) > topk:
            raise GateError(f"{path}: selection count above top-k at word {o}")
        out.append((pos, rows, heads, topk, mode, cnt, ids))
        o += 8 + n * (1 + topk)
    if not out:
        raise GateError(f"{path}: no selection records")
    return out


# ---------------------------------------------------------------------------------------------------------- seal
def seal(a):
    d = pathlib.Path(a.dir)
    manifest = d / f"manifest-r{a.rank}.json"
    if manifest.exists():
        raise GateError(f"{manifest}: already sealed (run directories are never reused)")
    if not a.run_id or not a.arm:
        raise GateError("seal: empty --run-id or --arm")
    binary = pathlib.Path(a.binary)
    if not binary.is_file():
        raise GateError(f"seal: no binary {binary}")
    env = parse_pairs(a.env, "--env")
    attest = parse_pairs(a.attest, "--attest")
    prompts = dump_entries(a, attest)
    doc = {"schema": SCHEMA, "run_id": a.run_id, "arm": a.arm, "rank": a.rank, "prefix": a.prefix,
           "binary": {"name": binary.name, "sha256": sha256_file(binary)}, "env": env, "attest": attest,
           "sealed_unix": int(time.time()), "prompts": prompts}
    if a.model_sha:
        doc["model_sha"] = a.model_sha
    if a.execution_witness:
        try:
            doc["execution"] = json.loads(pathlib.Path(a.execution_witness).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise GateError(f"cannot read execution witness: {e}") from e
        execution_witness(doc)
    tmp = manifest.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, manifest)
    print(f"sealed {manifest.name}: run {a.run_id} arm {a.arm} rank {a.rank} binary {doc['binary']['sha256'][:12]} "
          f"prompts {' '.join(prompts)}")
    return 0


def dump_entries(a, attest):
    """The sealed prompts map of one rank (seal and witness): every expected dump present, fresh, of the declared rows and
    finite; per prompt the token count / sha256 and per dump its bytes / sha256 / width (selection: records / mode)."""
    d = pathlib.Path(a.dir)
    prompts = {}
    for name, tok_path in parse_pairs(a.prompt, "--prompt").items():
        tok = pathlib.Path(tok_path)
        if not tok.is_file() or tok.stat().st_size == 0 or tok.stat().st_size % 4:
            raise GateError(f"seal: bad token file {tok}")
        files = {}
        kinds = [("tf", "f32", True)] + ([("sel", "u32", True)] if a.select else [])
        for kind, ext, required in kinds:
            f = d / f"{a.prefix}-{name}-{kind}-r{a.rank}.{ext}"
            if not f.is_file():
                if required:
                    raise GateError(f"seal: missing dump {f}")
                continue
            st = f.stat()
            if a.not_before is not None and st.st_mtime < a.not_before:
                raise GateError(f"seal: stale dump {f} (written before this run started)")
            entry = {"name": f.name, "bytes": st.st_size, "sha256": sha256_file(f)}
            if kind == "tf":
                if st.st_size == 0 or st.st_size % (a.rows * 4):
                    raise GateError(f"seal: {f} has {st.st_size} bytes, not {a.rows} rows of FP32")
                entry["width"] = st.st_size // (a.rows * 4)
                z = np.memmap(f, "<f4", "r")
                if not np.isfinite(z).all():
                    raise GateError(f"seal: nonfinite logits in {f}")
            else:
                recs = read_selection(f)
                modes = sorted({r[4] for r in recs})
                if len(modes) != 1:
                    raise GateError(f"seal: {f} mixes selection modes {modes}")
                entry["records"], entry["mode"] = len(recs), modes[0]
                if "sel_mode" in attest and str(modes[0]) != attest["sel_mode"]:
                    raise GateError(f"seal: {f} was written by selection mode {modes[0]}, arm declares {attest['sel_mode']}")
            files[kind] = entry
        prompts[name] = {"tokens": tok.stat().st_size // 4, "tokens_sha256": sha256_file(tok), "rows": a.rows, "files": files}
    if not prompts:
        raise GateError("seal: no --prompt")
    return prompts


# ------------------------------------------------------------------------------------------------------- witness
RECEIPT_PREFIX = "strix-route-receipt "
# Switches that only name outputs or the control: excluded from the numerical configuration identity.
MM_OUTPUT_SWITCHES = frozenset({"STRIX_MM_QUALITY_OUT", "STRIX_MM_QUALITY_ROWS", "STRIX_MM_QUALITY_SELECT", "STRIX_MM_STATE_FNV",
                                "STRIX_MM_DIAG_ATTN_ULP"})
# The FAST layer set's own lever and switches: ignored by config_difference (candidate == reference + this layer set).
MM_FAST_SWITCHES = frozenset({"STRIX_MM_PF_MSA_FLASH", "STRIX_MM_PF_MSA_FLASH_LAYERS", "STRIX_MM_FAST_PREFILL",
                              "STRIX_MMX_PF_ORDER", "STRIX_MM_PF_IDX_WMMA", "STRIX_MM_PF_IDX_WMMA_LAYERS", "STRIX_TP2_EXCHANGE_FMT"})
MM_FAST_LEVERS = frozenset({"minimax.msa_flash", "minimax.linear_order", "minimax.index_proj_wmma", "minimax.tp2_exchange_fmt"})
# The chunk-invariant linear order (minimax.linear_order invariant:<identity>) changes MoE/dense rounding in every layer:
# its FAST layer set is all MM_LAYERS layers (onset 0, so its nulls are layer-0 one-ulp controls).
MM_LAYERS = 60
MM_DIAG = re.compile(rb'DIAG attn ulp: layer (\d+) seed (\d+) rows (\d+) pos0 (\d+) values (\d+) prompt ("(?:[^"\\]|\\.)*")\s*')
MM_INDEX_EXEC = re.compile(rb'minimax-index-proj-executed rank (\d+) tokens (\d+) token_fnv ([0-9a-f]{16}) layers ([01]{60})\s*')


def token_fnv1a64(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def model_bundle_sha256(root):
    """Identity of every model asset: sorted relative path, size and complete content digest (including tensor payloads)."""
    root = pathlib.Path(root)
    if not root.is_dir():
        raise GateError(f"witness: no model directory {root}")
    entries = []
    for dirpath, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            p = pathlib.Path(dirpath) / f
            if not p.is_file():
                continue
            size = p.stat().st_size
            if f.endswith(".safetensors"):
                with open(p, "rb") as fh:
                    head = fh.read(8)
                    n = int.from_bytes(head, "little") if len(head) == 8 else -1
                    if n <= 0 or n > (256 << 20) or 8 + n > size:
                        raise GateError(f"witness: {p} has no safetensors header")
            entries.append([p.relative_to(root).as_posix(), size, "full", sha256_file(p)])
    if not entries:
        raise GateError(f"witness: empty model directory {root}")
    return canonical_sha({"files": entries})


def minimax_execution(log, rank, prompts):
    """MiniMax adapter: engine / profile / configuration identities from the rank's process route receipt, the control
    from the null-control log lines ("DIAG attn ulp: layer L seed S ..."; none = no control)."""
    receipts = []
    for line in log.decode("utf-8", "replace").splitlines():
        if line.startswith(RECEIPT_PREFIX):
            try:
                doc = json.loads(line[len(RECEIPT_PREFIX):])
            except ValueError as e:
                raise GateError(f"witness: unreadable route receipt: {e}") from e
            if isinstance(doc, dict) and doc.get("engine") == "minimax" and doc.get("rank") == rank and doc.get("scope") == "process":
                receipts.append(doc)
    if len(receipts) != 1:
        raise GateError(f"witness: expected one minimax process route receipt of rank {rank}, found {len(receipts)}")
    rc, levers = receipts[0], {}
    for lv in rc.get("levers") or []:
        name, eff = (lv.get("name"), lv.get("effective")) if isinstance(lv, dict) else (None, None)
        if not isinstance(name, str) or not isinstance(eff, str) or name in levers:
            raise GateError("witness: malformed or repeated lever in the route receipt")
        levers[name] = eff
    for need in ("minimax.msa_flash", "minimax.moe_prefill_format", "minimax.prefill_identity"):
        if need not in levers:
            raise GateError(f"witness: the route receipt lacks lever {need}")
    switches = (rc.get("switches") or {}).get("set") if isinstance(rc.get("switches"), dict) else None
    if not isinstance(switches, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in switches.items()):
        raise GateError("witness: the route receipt lacks its switch set")
    sw = {k: v for k, v in switches.items() if k not in MM_OUTPUT_SWITCHES}
    m = re.fullmatch(r"([0-9]+)(?::layers=([01]+))?", levers["minimax.msa_flash"])
    if not m or (m.group(1) != "0") != (m.group(2) is not None):
        raise GateError(f"witness: unexpected minimax.msa_flash value {levers['minimax.msa_flash']!r}")
    fast = None
    if m.group(1) != "0":
        layers = [i for i, ch in enumerate(m.group(2)) if ch == "1"]
        if not layers:
            raise GateError("witness: MSA flash selected on no layer")
        fast = {"policy": "msa-flash", "msa_flash_mode": 10 + int(m.group(1)), "layers": layers}
    # Index projections on WMMA (receipts before the lever existed: absent = off). The FAST layer set is the union,
    # so the floor's onset is the first layer either member changes.
    ip = re.fullmatch(r"(0|1:layers=([01]+))", levers.get("minimax.index_proj_wmma", "0"))
    if not ip:
        raise GateError(f"witness: unexpected minimax.index_proj_wmma value {levers['minimax.index_proj_wmma']!r}")
    if ip.group(2) is not None:
        idx_layers = [i for i, ch in enumerate(ip.group(2)) if ch == "1"]
        if not idx_layers:
            raise GateError("witness: index projection WMMA selected on no layer")
        executed = set()
        for line in log.splitlines():
            if not line.startswith(b"minimax-index-proj-executed "):
                continue
            event = MM_INDEX_EXEC.fullmatch(line)
            if event is None or int(event[1]) != rank or event[4].decode() != ip.group(2):
                raise GateError("witness: malformed or inconsistent index WMMA execution evidence")
            if int(event[2]) <= 8:
                raise GateError("witness: index WMMA execution has too few rows")
            executed.add((int(event[2]), event[3].decode()))
        for name, prompt in prompts.items():
            if (prompt["tokens"], prompt.get("tokens_fnv1a64")) not in executed:
                raise GateError(f"witness: missing completed index WMMA execution for prompt {name!r}")
        fast = ({"policy": "index-proj", "msa_flash_mode": 10, "layers": idx_layers} if fast is None else
                {"policy": "msa-flash+index-proj", "msa_flash_mode": fast["msa_flash_mode"],
                 "layers": sorted(set(fast["layers"]) | set(idx_layers))})
    order = levers.get("minimax.linear_order", "adaptive")
    if order != "adaptive":
        mo = re.fullmatch(r"invariant:([a-z0-9.-]+)", order)
        if not mo:
            raise GateError(f"witness: unexpected minimax.linear_order value {order!r}")
        fast = {"policy": mo.group(1) + ("+" + fast["policy"] if fast else ""),
                "msa_flash_mode": fast["msa_flash_mode"] if fast else 0, "layers": list(range(MM_LAYERS))}
    xchg = levers.get("minimax.tp2_exchange_fmt", "f32")
    if xchg not in ("f32", "f16"):
        raise GateError(f"witness: unexpected minimax.tp2_exchange_fmt value {xchg!r}")
    if xchg == "f16":  # every layer's two exchanges round the partials: the FAST onset is layer 0
        fast = fast or {"policy": "", "msa_flash_mode": 10, "layers": []}
        fast["policy"] = (fast["policy"] + "+" if fast["policy"] else "") + "tp2-xchg-f16"
        fast["layers"] = sorted(set(fast["layers"]) | set(range(MM_LAYERS)))
    plain = levers["minimax.moe_prefill_format"] == "exl3" and levers["minimax.prefill_identity"] == "default"
    profile = ("fast-prefill" if fast else "exact") if plain else "other"
    diag, covered = [], set()
    for line in log.splitlines():
        if not line.startswith(b"DIAG attn ulp:"):
            continue
        event = MM_DIAG.fullmatch(line)
        if event is None:
            raise GateError("witness: null-control event lacks a prompt identity or is malformed")
        try:
            prompt = json.loads(event[6])
        except (ValueError, UnicodeError) as e:
            raise GateError("witness: invalid null-control prompt identity") from e
        layer, seed, rows, pos0, values = map(int, event.groups()[:5])
        if prompt not in prompts:
            raise GateError(f"witness: null-control event names unsealed prompt {prompt!r}")
        if rows == 0 or values == 0 or pos0 + rows > prompts[prompt]["tokens"]:
            raise GateError(f"witness: empty or out-of-range null-control event for {prompt!r}")
        covered.add(prompt)
        diag.append((layer, seed))
    control = None
    if diag:
        layer_set, seed_set = {int(x[0]) for x in diag}, {int(x[1]) for x in diag}
        if len(layer_set) != 1 or len(seed_set) != 1:
            raise GateError(f"witness: the null control fired at layers {sorted(layer_set)} with seeds {sorted(seed_set)}")
        if covered != set(prompts):
            raise GateError(f"witness: null-control events missing for prompts {sorted(set(prompts) - covered)}")
        control = {"kind": MM_NULL_KIND, "layer": layer_set.pop(), "magnitude": 1, "unit": "ulp", "seed": seed_set.pop()}
    return {"engine": "minimax", "profile": profile, "fast": fast, "control": control,
            "numerical_config_sha256": canonical_sha({"engine": "minimax", "levers": levers, "switches": sw}),
            "config": {"levers": levers, "switches": sw}, "route_fingerprint": rc.get("fingerprint")}


WITNESS_ADAPTERS = {"minimax": minimax_execution}


def witness(a):
    out = pathlib.Path(a.out)
    if out.exists():
        raise GateError(f"{out}: exists (witnesses are never rewritten)")
    binary = pathlib.Path(a.binary)
    if not binary.is_file():
        raise GateError(f"witness: no binary {binary}")
    try:
        log = pathlib.Path(a.engine_log).read_bytes()
    except OSError as e:
        raise GateError(f"witness: cannot read the engine log: {e}") from e
    prompts = dump_entries(a, parse_pairs(a.attest, "--attest"))
    # FNV binds the native completion record to input bytes; keep the sealed SHA-256 manifest contract unchanged.
    execution_prompts = {name: {**prompts[name], "tokens_fnv1a64": token_fnv1a64(pathlib.Path(path).read_bytes())}
                         for name, path in parse_pairs(a.prompt, "--prompt").items()}
    w = {"schema": WITNESS_SCHEMA, "run_id": a.run_id, "rank": a.rank, "binary_sha256": sha256_file(binary),
         "model_sha256": model_bundle_sha256(a.model), "prompts": prompts,
          **WITNESS_ADAPTERS[a.engine](log, a.rank, execution_prompts)}
    execution_witness({"execution": w, "run_id": a.run_id, "rank": a.rank, "binary": {"sha256": w["binary_sha256"]},
                       "prompts": prompts})
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(w, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, out)
    fast = w["fast"]
    print(f"witness {out.name}: run {a.run_id} rank {a.rank} engine {w['engine']} profile {w['profile']} control {w['control']} "
          f"fast {None if fast is None else (fast['msa_flash_mode'], fast['layers'][0], fast['layers'][-1], len(fast['layers']))}")
    return 0


# ------------------------------------------------------------------------------------------------------- compare
def load_arm(d, ranks):
    d = pathlib.Path(d)
    docs = []
    for r in ranks:
        p = d / f"manifest-r{r}.json"
        if not p.is_file():
            raise GateError(f"{p}: missing manifest (the arm was not sealed)")
        doc = json.loads(p.read_text(encoding="utf-8"))
        if doc.get("schema") != SCHEMA or doc.get("rank") != r:
            raise GateError(f"{p}: not a {SCHEMA} manifest of rank {r}")
        docs.append(doc)
    first = docs[0]
    for doc in docs[1:]:
        for key in ("run_id", "arm", "env", "attest", "prefix"):
            if doc[key] != first[key]:
                raise GateError(f"{d}: ranks disagree on {key}")
        if sorted(doc["prompts"]) != sorted(first["prompts"]):
            raise GateError(f"{d}: ranks disagree on the prompt set")
    if any("execution" in doc for doc in docs):
        witnesses = [execution_witness(doc) for doc in docs]
        for w in witnesses[1:]:
            for key in ("engine", "profile", "model_sha256", "numerical_config_sha256", "control", "fast", "config"):
                if w.get(key) != witnesses[0].get(key):
                    raise GateError(f"{d}: ranks disagree on execution {key}")
    return d, docs


def checked_file(d, entry):
    f = d / entry["name"]
    if not f.is_file() or f.stat().st_size != entry["bytes"]:
        raise GateError(f"{f}: missing or not the sealed size")
    if sha256_file(f) != entry["sha256"]:
        raise GateError(f"{f}: content differs from its manifest")
    return f


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def arm_logits(d, docs, name):
    shards, rows = [], docs[0]["prompts"][name]["rows"]
    for doc in docs:
        p = doc["prompts"][name]
        if p["rows"] != rows:
            raise GateError(f"{d}: ranks disagree on the rows of {name}")
        e = p["files"]["tf"]
        shards.append(np.memmap(checked_file(d, e), "<f4", "r", shape=(rows, e["width"])))
    return shards, rows


def compare_logits(da, a, db, b, name, tokens, chunk=16):
    sa, rows = arm_logits(da, a, name)
    sb, rows_b = arm_logits(db, b, name)
    if rows != rows_b or [s.shape for s in sa] != [s.shape for s in sb]:
        raise GateError(f"{name}: arms disagree on rows or vocabulary widths")
    total = len(tokens)
    if rows > total:
        raise GateError(f"{name}: {rows} rows scored but the prompt has {total} tokens")
    kl, agree, nla, nlb, hia, hib = np.empty(rows), np.empty(rows, bool), [], [], [], []
    identical = True
    for i in range(0, rows, chunk):
        za = np.concatenate([s[i:i + chunk] for s in sa], axis=1).astype(np.float64)
        zb = np.concatenate([s[i:i + chunk] for s in sb], axis=1).astype(np.float64)
        if not (np.isfinite(za).all() and np.isfinite(zb).all()):
            raise GateError(f"{name}: nonfinite logits")
        identical = identical and bool((za == zb).all())
        la, lb = log_softmax(za), log_softmax(zb)
        kl[i:i + chunk] = (np.exp(la) * (la - lb)).sum(axis=1)
        ta, tb = za.argmax(axis=1), zb.argmax(axis=1)
        agree[i:i + chunk] = ta == tb
        for j in range(za.shape[0]):
            pos = total - rows + i + j  # the row at prompt position pos predicts token pos + 1
            if pos + 1 < total:
                nla.append(-la[j, tokens[pos + 1]])
                nlb.append(-lb[j, tokens[pos + 1]])
                hia.append(ta[j] == tokens[pos + 1])
                hib.append(tb[j] == tokens[pos + 1])
    return {"rows": rows, "kl": kl, "agree": agree, "nll_ref": np.array(nla), "nll_cand": np.array(nlb),
            "hit_ref": np.array(hia, dtype=bool), "hit_cand": np.array(hib, dtype=bool), "identical": identical}


def compare_selection(da, a, db, b, name):
    """Per global layer (ranks in order, each rank's records in layer order): slots and block ids that differ."""
    layers, base = [], 0
    for doc_a, doc_b in zip(a, b):
        ea, eb = doc_a["prompts"][name]["files"].get("sel"), doc_b["prompts"][name]["files"].get("sel")
        if ea is None or eb is None:
            return None
        ra, rb = read_selection(checked_file(da, ea)), read_selection(checked_file(db, eb))
        if len(ra) != len(rb):
            raise GateError(f"{name}: arms wrote a different number of selection records")
        seen, depth = {}, 0
        for x, y in zip(ra, rb):
            if x[:4] != y[:4]:
                raise GateError(f"{name}: selection records disagree on position / rows / heads / top-k")
            pos, rows, heads, topk = x[:4]
            layer = seen.get(pos, 0)
            seen[pos] = layer + 1
            depth = max(depth, layer + 1)
            while len(layers) <= base + layer:
                layers.append({"slots": 0, "slot_diff": 0, "id_diff": 0, "first_pos": None, "modes": [x[4], y[4]]})
            acc = layers[base + layer]
            ca, cb = x[5].astype(np.int64), y[5].astype(np.int64)
            slot = np.arange(topk)[None, None, :]
            big = np.int64(1) << 40
            ia = np.sort(np.where(slot < ca[:, :, None], x[6].astype(np.int64), big), axis=2)
            ib = np.sort(np.where(slot < cb[:, :, None], y[6].astype(np.int64), big), axis=2)
            neq = ia != ib
            lost = np.zeros((rows, heads), np.int64)  # ids of the reference set missing from the candidate set
            for r, h in zip(*np.nonzero(neq.any(axis=2))):
                lost[r, h] = len(set(ia[r, h][ia[r, h] < big].tolist()) - set(ib[r, h][ib[r, h] < big].tolist()))
            bad = neq.any(axis=2)
            acc["slots"] += rows * heads
            acc["slot_diff"] += int(bad.sum())
            acc["id_diff"] += int(lost.sum())
            if bad.any():
                p = pos + int(np.nonzero(bad.any(axis=1))[0][0])
                acc["first_pos"] = p if acc["first_pos"] is None else min(acc["first_pos"], p)
        base += depth
    return layers


def compare(a):
    if a.mode == "turbo":
        return compare_turbo(a)
    ranks = sorted(set(a.ranks))
    null_dirs = a.null_arm or []
    if a.expect_identical and (a.dnll_rule != "abs" or null_dirs):
        raise GateError("--expect-identical has no dNLL limit: drop --dnll-rule / --null-arm")
    check_dnll_policy(a.dnll_rule, a.engine, a.profile, len(null_dirs))
    da, ma = load_arm(a.ref, ranks)
    db, mb = load_arm(a.cand, ranks)
    if ma[0]["run_id"] == mb[0]["run_id"]:
        raise GateError("both directories hold the same run id: one run compared with itself")
    tokens = parse_pairs(a.tokens, "--tokens")
    names = sorted(ma[0]["prompts"])
    if names != sorted(mb[0]["prompts"]) or names != sorted(tokens):
        raise GateError("arms and --tokens disagree on the prompt set")
    nulls, seen_ids = [], {ma[0]["run_id"], mb[0]["run_id"]}
    for nd in null_dirs:  # floor rule: every null is a sealed run on the candidate's binary and prompts
        dn, mn = load_arm(nd, ranks)
        if mn[0]["run_id"] in seen_ids:
            raise GateError(f"{nd}: null arm run id {mn[0]['run_id']} repeats the reference, the candidate or another null")
        seen_ids.add(mn[0]["run_id"])
        if [m["binary"]["sha256"] for m in mn] != [m["binary"]["sha256"] for m in mb]:
            raise GateError(f"{nd}: null arm ran another binary than the candidate (the floor must be measured on the same binary)")
        if sorted(mn[0]["prompts"]) != names:
            raise GateError(f"{nd}: null arm prompt set differs from the reference / candidate")
        nulls.append((dn, mn))
    onset = check_floor_execution(ma, mb, nulls, a.engine, a.profile) if a.dnll_rule == "floor" else None
    mm_floor = a.dnll_rule == "floor" and a.engine == "minimax"
    report = {"ref": {k: ma[0][k] for k in ("run_id", "arm", "env", "attest")}, "cand": {k: mb[0][k] for k in ("run_id", "arm", "env", "attest")},
              "ref_binary": [m["binary"]["sha256"] for m in ma], "cand_binary": [m["binary"]["sha256"] for m in mb], "prompts": {}}
    if a.dnll_rule == "floor":
        report["ref"]["execution"] = ma[0]["execution"]
        report["cand"]["execution"] = mb[0]["execution"]
    print(f"ref  run {ma[0]['run_id']} arm {ma[0]['arm']} env {ma[0]['env']} binary {report['ref_binary'][0][:12]}")
    print(f"cand run {mb[0]['run_id']} arm {mb[0]['arm']} env {mb[0]['env']} binary {report['cand_binary'][0][:12]}")
    pooled = {"kl": [], "agree": [], "nll_ref": [], "nll_cand": []}
    identical, sel_slots, sel_slot_diff, sel_id_diff, sel_seen = True, 0, 0, 0, False
    results, copy_masks, tok_of = [], [], {}
    for name in names:
        tok_sha = sha256_file(tokens[name])
        for m in ma + mb + [m for _, mn in nulls for m in mn]:
            if m["prompts"][name]["tokens_sha256"] != tok_sha:
                raise GateError(f"{name}: a sealed run used a different token file than --tokens")
        tok = np.fromfile(tokens[name], "<u4").astype(np.int64)
        if nulls:
            tok_of[name] = tok
        r = compare_logits(da, ma, db, mb, name, tok)
        results.append(r)
        copy_masks.append(copy_rows(tok, r["rows"]))
        identical = identical and r["identical"]
        for k in pooled:
            pooled[k].append(r[k])
        row = {"rows": r["rows"], "kl_mean": float(r["kl"].mean()), "kl_max": float(r["kl"].max()),
               "kl_median": float(np.median(r["kl"])), "top1": float(r["agree"].mean()),
               "nll_ref": float(r["nll_ref"].mean()), "dnll": float(r["nll_cand"].mean() - r["nll_ref"].mean()),
               "identical": r["identical"]}
        print(f"{name}: rows {row['rows']} KL mean {row['kl_mean']:.5f} median {row['kl_median']:.5f} max {row['kl_max']:.4f}; "
              f"top-1 {row['top1']:.4f}; NLL ref {row['nll_ref']:.4f} dNLL {row['dnll']:+.5f}; logits identical {row['identical']}")
        layers = compare_selection(da, ma, db, mb, name)
        if layers is not None:
            sel_seen = True
            slots, sd, idd = (sum(x[k] for x in layers) for k in ("slots", "slot_diff", "id_diff"))
            sel_slots, sel_slot_diff, sel_id_diff = sel_slots + slots, sel_slot_diff + sd, sel_id_diff + idd
            first = next((i for i, x in enumerate(layers) if x["slot_diff"]), None)
            row["selection"] = {"layers": len(layers), "slots": slots, "slot_diff": sd, "id_diff": idd, "first_layer": first,
                                "first_pos": None if first is None else layers[first]["first_pos"],
                                "modes": layers[0]["modes"], "per_layer": [[x["slot_diff"], x["id_diff"]] for x in layers]}
            print(f"{name}: selection modes ref/cand {layers[0]['modes']}: {len(layers)} layers, {slots} (row, head) slots, "
                  f"{sd} with a different set ({sd / slots:.3e}), {idd} ids replaced; first at layer {first}"
                  f"{'' if first is None else ' position ' + str(layers[first]['first_pos'])}")
            if first is not None:
                print(f"{name}: differing slots per layer: {' '.join(str(x['slot_diff']) for x in layers)}")
        report["prompts"][name] = row
    kl = float(np.concatenate(pooled["kl"]).mean())
    top1 = float(np.concatenate(pooled["agree"]).mean())
    dnll = float(np.concatenate(pooled["nll_cand"]).mean() - np.concatenate(pooled["nll_ref"]).mean())
    stats = arm_stats(results, copy_masks)
    print(f"rows {stats['rows']}: KL q50 {stats['kl_q']['q50']:.2e} q90 {stats['kl_q']['q90']:.2e} "
          f"q99 {stats['kl_q']['q99']:.2e}; true token worse {stats['worse']} better {stats['better']}; copy rows "
          f"{stats['copy_rows']} misses ref {stats['copy_miss_ref']} cand {stats['copy_miss']} (copy dNLL {stats['copy_dnll']:+.5f})")
    null_rows, null_stats = [], []
    for dn, mn in nulls:  # pooled exactly like the candidate: same reference, same prompts, same labels
        rs = [compare_logits(da, ma, dn, mn, name, tok_of[name]) for name in names]
        st = arm_stats(rs, copy_masks)
        null_stats.append(st)
        null_rows.append({"dir": str(dn), "run_id": mn[0]["run_id"], "arm": mn[0]["arm"], "env": mn[0]["env"],
                          "execution": mn[0]["execution"], "rows": st["rows"], "kl": st["kl"], "top1": st["top1"],
                          "dnll": st["dnll"], "stats": st})
        print(f"null {mn[0]['arm']} run {mn[0]['run_id']} env {mn[0]['env']}: KL {st['kl']:.5f}, top-1 {st['top1']:.4f}, "
              f"dNLL {st['dnll']:+.5f}; KL q50/q90/q99 {st['kl_q']['q50']:.2e}/{st['kl_q']['q90']:.2e}/{st['kl_q']['q99']:.2e}; "
              f"worse {st['worse']} better {st['better']}; copy misses {st['copy_miss']} (ref {st['copy_miss_ref']})")
    dv = None
    if a.expect_identical:
        ok = identical and sel_slot_diff == 0
        rule = "identical logits and selections"
    else:
        dv = dnll_verdict(dnll, a.dnll_rule, a.max_dnll, [x["dnll"] for x in null_rows])
        dv.update(engine=a.engine, profile=a.profile, nulls=null_rows)
        ok = kl <= a.max_kl and top1 >= a.min_top1 and dv["pass"]
        if mm_floor:
            checks = minimax_floor_checks(stats, null_stats, max_kl=a.max_kl, min_top1=a.min_top1, max_dnll=a.max_dnll)
            dnll_check = next(c for c in checks if c["name"] == "dnll")
            dv.update(policy=MM_FLOOR["policy"], onset=onset, checks=checks, floor_limit=dv["limit"],
                      limit=dnll_check["limit"], cap=MM_FLOOR["dnll_cap"], pass_dnll=dnll_check["pass"])
            dv["pass"] = dnll_check["pass"]
            ok = all(c["pass"] for c in checks if c["required"])
            for c in checks:
                extra = {k: v for k, v in c.items() if k not in ("name", "value", "limit", "pass", "required")}
                print(f"check {c['name']}: {c['value']:.6g} vs limit {c['limit']:.6g} -> {'PASS' if c['pass'] else 'FAIL'}"
                      f"{'' if c['required'] else ' (reported, not required)'} "
                      f"{json.dumps(extra, sort_keys=True)}")
            rule = (f"floor {a.engine}/{a.profile} {MM_FLOOR['policy']} onset layer {onset}, {dv['n_nulls']} nulls: all of "
                    f"{', '.join(c['name'] for c in checks if c['required'])} (floor only relaxes the absolute bar)")
        elif a.dnll_rule == "floor":
            rule = (f"KL <= {a.max_kl}, top-1 >= {a.min_top1}, dNLL <= {dv['limit']:+.5f} (floor {a.engine}/{a.profile}: "
                    f"{dv['n_nulls']} nulls mean {dv['mean']:+.5f} + {FLOOR_SIGMAS:g} sd {dv['sd']:.5f})")
        else:
            rule = f"KL <= {a.max_kl}, top-1 >= {a.min_top1}, dNLL <= {a.max_dnll}"
    report["pooled"] = {"rows": int(sum(len(k) for k in pooled["kl"])), "kl": kl, "top1": top1, "dnll": dnll, "identical": identical,
                        "selection": {"slots": sel_slots, "slot_diff": sel_slot_diff, "id_diff": sel_id_diff} if sel_seen else None,
                        "rule": rule, "dnll_rule": dv, "stats": stats, "pass": ok}
    if mm_floor:
        absolute = kl <= a.max_kl and top1 >= a.min_top1 and dnll <= a.max_dnll
        report["pooled"]["abs_pass"] = bool(absolute)
        print(f"REFERENCE absolute KL <= {a.max_kl}, top-1 >= {a.min_top1}, dNLL <= {a.max_dnll}: "
              f"{'PASS' if absolute else 'FAIL'} (not the verdict under the floor rule)")
    elif dv is not None and a.dnll_rule == "floor":
        print(f"REFERENCE absolute dNLL <= {a.max_dnll}: {'PASS' if dv['abs_pass'] else 'FAIL'} (not the verdict under the floor rule)")
    sel = f"; selection {sel_slot_diff}/{sel_slots} slots differ, {sel_id_diff} ids" if sel_seen else ""
    print(f"POOLED rows {report['pooled']['rows']}: KL {kl:.5f}, top-1 {top1:.4f}, dNLL {dnll:+.5f}, logits identical {identical}{sel} "
          f"[{rule}] -> {'PASS' if ok else 'FAIL'}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return 0 if ok else 1

TURBO_RECEIPT = "strix-turbo-receipt/v1"
TASK_CATEGORIES = ("code", "math", "retrieval", "tools", "json", "prose")
TASK_DROP_POINTS = {"code": 5.0, "math": 5.0, "retrieval": 2.0, "tools": 1.0}  # absolute drop vs E, proposed [I]
PERF_KEYS = ("tok_s", "bytes_per_token", "ttft_ms", "p95_ms")
Z95 = 1.959964
Z95_ONE_SIDED = 1.644854


def wilson(passed, total, z=Z95):
    if total <= 0:
        return [None, None]
    p, z2 = passed / total, z * z
    centre = (p + z2 / (2 * total)) / (1 + z2 / total)
    half = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / (1 + z2 / total)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def checked_model_sha(sha):
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in sha):
        raise GateError("TURBO requires a nonempty SHA-256 model digest for every participating rank")
    return sha.lower()


def arm_sha(manifests):
    """Require a consistent, known weights identity for every rank of a TURBO arm."""
    shas = set()
    for m in manifests:
        declared = m.get("model_sha")
        witnessed = (m.get("execution") or {}).get("model_sha256")
        for sha in (declared, witnessed):
            if sha is not None:
                checked_model_sha(sha)
        if declared is not None and witnessed is not None and declared.lower() != witnessed.lower():
            raise GateError("declared model sha disagrees with execution witness model_sha256")
        shas.add(checked_model_sha(declared or witnessed))
    if len(shas) != 1:
        raise GateError("ranks of one arm record different model shas")
    return shas.pop()


def check_pack_shas(sha_e, sha_t, sha_r, a):
    sha_e, sha_t = checked_model_sha(sha_e), checked_model_sha(sha_t)
    if a.reference_pack_arm:
        sha_r = checked_model_sha(sha_r)
    candidate_sha = checked_model_sha(a.candidate_sha) if a.candidate_sha is not None else None
    reference_sha = checked_model_sha(a.reference_sha) if a.reference_sha is not None else None
    if a.reference_sha and not a.reference_pack_arm:
        raise GateError("--reference-sha without --reference-pack-arm")
    if sha_t != sha_e and not a.candidate_sha:
        raise GateError(f"candidate sha {sha_t} differs from the reference E sha {sha_e} and --candidate-sha does not declare it")
    if candidate_sha and candidate_sha != sha_t:
        raise GateError(f"--candidate-sha {a.candidate_sha} does not match the candidate seal {sha_t}")
    if a.reference_pack_arm:
        if sha_r != sha_e and not a.reference_sha:
            raise GateError(f"reference pack sha {sha_r} differs from E sha {sha_e} and --reference-sha does not declare it")
        if reference_sha and reference_sha != sha_r:
            raise GateError(f"--reference-sha {a.reference_sha} does not match the reference pack seal {sha_r}")


def paired_u95(x, y):
    """One-sided upper 95 % bound of mean(x - y) over rows (rows are paired: same E, same tokens)."""
    d = np.asarray(x, np.float64) - np.asarray(y, np.float64)
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0
    return float(d.mean()) + Z95_ONE_SIDED * se


def mean_ci(v):
    v = np.asarray(v, np.float64)
    se = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
    m = float(v.mean())
    return {"mean": m, "ci95": [m - Z95 * se, m + Z95 * se]}


def turbo_metrics(rs):
    kl = np.concatenate([r["kl"] for r in rs])
    dis = np.concatenate([(~r["agree"]).astype(np.float64) for r in rs])
    dn = np.concatenate([r["nll_cand"] - r["nll_ref"] for r in rs])
    return {"kl": kl, "disagree": dis, "dnll": dn}


def load_task_results(path):
    try:
        doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        tasks = doc["tasks"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise GateError(f"cannot read task results: {e}") from e
    if not isinstance(tasks, list) or not tasks:
        raise GateError("task results: empty tasks list")
    for t in tasks:
        if not isinstance(t, dict) or not t.get("name") or t.get("category") not in TASK_CATEGORIES:
            raise GateError(f"task results: bad task entry {t!r} (name + category in {'|'.join(TASK_CATEGORIES)})")
        for arm in ("E", "T", "R"):
            c = t.get(arm)
            if c is None and arm == "R":
                continue
            if not isinstance(c, dict) or not is_int(c.get("passed")) or not is_int(c.get("total")) \
                    or c["total"] <= 0 or not 0 <= c["passed"] <= c["total"]:
                raise GateError(f"task results: {t.get('name')}: invalid {arm} counts")
    return tasks


def task_summary(tasks):
    """Per category and overall: passed/total/rate/Wilson 95 % interval per arm (None when a task lacks that arm)."""
    out = {}
    groups = {"overall": tasks}
    for t in tasks:
        groups.setdefault(t["category"], []).append(t)
    for cat, ts in groups.items():
        row = {}
        for arm in ("E", "T", "R"):
            if any(t.get(arm) is None for t in ts):
                row[arm] = None
                continue
            p, n = sum(t[arm]["passed"] for t in ts), sum(t[arm]["total"] for t in ts)
            row[arm] = {"passed": p, "total": n, "rate": p / n, "wilson95": wilson(p, n)}
        out[cat] = row
    return out


def task_checks(summary, limits):
    checks = []
    for cat, row in summary.items():
        if cat == "overall":
            continue
        e, t, r = row["E"], row["T"], row["R"]
        drop_t = e["rate"] - t["rate"]
        if r is not None:
            drop_r = e["rate"] - r["rate"]
            overlap = t["wilson95"][1] >= r["wilson95"][0] and r["wilson95"][1] >= t["wilson95"][0]
            checks.append({"name": f"task:{cat}:vs-R", "value": drop_t, "limit": drop_r,
                           "pass": drop_t <= drop_r or overlap, "overlap": overlap})
        if cat in limits:
            checks.append({"name": f"task:{cat}:drop-vs-E", "value": drop_t * 100, "limit": limits[cat],
                           "pass": drop_t * 100 <= limits[cat] + 1e-9})
        if cat == "json":
            checks.append({"name": "task:json:100pct", "value": t["rate"], "limit": 1.0, "pass": t["passed"] == t["total"]})
    return checks


def compare_turbo(a):
    ranks = sorted(set(a.ranks))
    if a.expect_identical or a.dnll_rule != "abs":
        raise GateError("--mode turbo has its own relative rule: drop --expect-identical / --dnll-rule")
    de, me = load_arm(a.ref, ranks)
    dt, mt = load_arm(a.cand, ranks)
    dr, mr = load_arm(a.reference_pack_arm, ranks) if a.reference_pack_arm else (None, None)
    sha_e, sha_t, sha_r = arm_sha(me), arm_sha(mt), (arm_sha(mr) if mr else None)
    check_pack_shas(sha_e, sha_t, sha_r, a)
    tokens = parse_pairs(a.tokens, "--tokens")
    names = sorted(me[0]["prompts"])
    seen = {me[0]["run_id"]}
    others = [("candidate", mt)] + ([("reference pack", mr)] if mr else [])
    nulls = []
    for nd in a.null_arm or []:
        dn, mn = load_arm(nd, ranks)
        arm_sha(mn)  # Null controls also need known, consistent pack identities.
        nulls.append((dn, mn))
        others.append((f"null arm {nd}", mn))
    for label, m in others:
        if m[0]["run_id"] in seen:
            raise GateError(f"{label}: run id {m[0]['run_id']} repeats another arm")
        seen.add(m[0]["run_id"])
    if any(sorted(m[0]["prompts"]) != names for _, m in others) or names != sorted(tokens):
        raise GateError("arms and --tokens disagree on the prompt set")
    toks = {}
    for name in names:
        tok_sha = sha256_file(tokens[name])
        for m in me + [x for _, mm in others for x in mm]:
            if m["prompts"][name]["tokens_sha256"] != tok_sha:
                raise GateError(f"{name}: a sealed run used a different token file than --tokens")
        toks[name] = np.fromfile(tokens[name], "<u4").astype(np.int64)

    def measure(dx, mx):
        return turbo_metrics([compare_logits(de, me, dx, mx, n, toks[n]) for n in names])

    mtr = measure(dt, mt)
    mrr = measure(dr, mr) if mr else None
    abs_t = {"kl": float(mtr["kl"].mean()), "top1": 1 - float(mtr["disagree"].mean()), "dnll": float(mtr["dnll"].mean())}

    def block(m):
        return {"kl": mean_ci(m["kl"]), "top1": mean_ci(1 - m["disagree"]), "dnll": mean_ci(m["dnll"]), "rows": int(len(m["kl"]))}

    metrics = {"T": block(mtr)}
    checks = []
    for flag, key, op in (("turbo_max_kl", "kl", "le"), ("turbo_min_top1", "top1", "ge"), ("turbo_max_dnll", "dnll", "le")):
        lim = getattr(a, flag)
        if lim is not None:
            checks.append({"name": f"abs:{key}", "value": abs_t[key], "limit": lim,
                           "pass": abs_t[key] <= lim if op == "le" else abs_t[key] >= lim})
    if mr:
        metrics["R"] = block(mrr)
        for key, label in (("kl", "KL"), ("disagree", "disagreement"), ("dnll", "dNLL")):
            u = paired_u95(mtr[key], mrr[key])
            checks.append({"name": f"rel:{label}", "value": u, "limit": 0.0, "pass": u <= 0.0, "what": "U95(T - R) <= 0"})
    controls = []
    for dn, mn in nulls:
        mm = measure(dn, mn)
        controls.append({"dir": str(dn), "run_id": mn[0]["run_id"], "arm": mn[0]["arm"], "env": mn[0]["env"],
                         "kl": float(mm["kl"].mean()), "top1": 1 - float(mm["disagree"].mean()), "dnll": float(mm["dnll"].mean())})
    tasks = None
    missing_r = mr is None
    if a.task_results:
        limits = dict(TASK_DROP_POINTS)
        for item in a.task_drop_limit or []:
            k, v = item.split("=", 1)
            limits[k] = float(v)
        tasks = task_summary(load_task_results(a.task_results))
        missing_r = missing_r or tasks["overall"]["R"] is None
        checks += task_checks(tasks, limits)
    print(f"E run {me[0]['run_id']} sha {sha_e}; T run {mt[0]['run_id']} sha {sha_t}; "
          f"R {'run ' + mr[0]['run_id'] + ' sha ' + str(sha_r) if mr else 'absent'}")
    print(f"T vs E: KL {abs_t['kl']:.5f}, top-1 {abs_t['top1']:.4f}, dNLL {abs_t['dnll']:+.5f}")
    if mr:
        print(f"R vs E: KL {metrics['R']['kl']['mean']:.5f}, top-1 {metrics['R']['top1']['mean']:.4f}, "
              f"dNLL {metrics['R']['dnll']['mean']:+.5f}")
    for c in controls:
        print(f"null {c['arm']} run {c['run_id']}: KL {c['kl']:.5f}, top-1 {c['top1']:.4f}, dNLL {c['dnll']:+.5f}")
    for c in checks:
        print(f"check {c['name']}: {c['value']:.6g} vs limit {c['limit']:.6g} -> {'PASS' if c['pass'] else 'FAIL'}")
    failing = [c["name"] for c in checks if not c["pass"]]
    verdict = "FAIL" if failing else ("NO-REFERENCE" if missing_r else "PASS")
    print(f"TURBO verdict {verdict}" + (f": failing {', '.join(failing)}" if failing else ""))
    if a.receipt:
        write_turbo_receipt(a, verdict, failing, me, mt, mr, (sha_e, sha_t, sha_r), metrics, checks, controls, tasks)
    return {"PASS": 0, "FAIL": 1, "NO-REFERENCE": 3}[verdict]


def read_json_opt(path, what):
    if not path:
        return None
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise GateError(f"cannot read {what}: {e}") from e


def write_turbo_receipt(a, verdict, failing, me, mt, mr, shas, metrics, checks, controls, tasks):
    perf_in = read_json_opt(a.perf, "--perf") or {}
    perf = {k: perf_in.get(k) for k in PERF_KEYS}
    perf["missing"] = [k for k in PERF_KEYS if perf_in.get(k) is None]
    cfg = None
    if a.effective_config:
        try:
            cfg = pathlib.Path(a.effective_config).read_text(encoding="utf-8").splitlines()
        except OSError as e:
            raise GateError(f"cannot read --effective-config: {e}") from e
    witnesses = {n: [m["execution"] for m in ms] for n, ms in (("E", me), ("T", mt), ("R", mr or [])) if ms and "execution" in ms[0]}
    doc = {"schema": TURBO_RECEIPT, "verdict": verdict, "failing": failing, "commit": a.commit,
           "binary_sha256": {"E": me[0]["binary"]["sha256"], "T": mt[0]["binary"]["sha256"],
                             "R": mr[0]["binary"]["sha256"] if mr else None},
           "pack": {"calibration": a.calibration, "E_sha": shas[0], "T_sha": shas[1], "R_sha": shas[2]},
           "run": {"E": me[0]["run_id"], "T": mt[0]["run_id"], "R": mr[0]["run_id"] if mr else None},
           "hardware": read_json_opt(a.run_manifest, "--run-manifest"),
           "routes": {"effective_config": cfg, "execution_witness": witnesses or None},
           "metrics": metrics, "checks": checks, "tasks": tasks, "controls": controls, "performance": perf}
    pathlib.Path(a.receipt).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seal")
    s.add_argument("--dir", required=True)
    s.add_argument("--rank", type=int, required=True)
    s.add_argument("--arm", required=True)
    s.add_argument("--run-id", required=True)
    s.add_argument("--binary", required=True)
    s.add_argument("--prefix", default="q")
    s.add_argument("--rows", type=int, required=True)
    s.add_argument("--prompt", action="append", metavar="NAME=TOKENS.u32")
    s.add_argument("--env", action="append", metavar="NAME=VALUE", help="environment the arm was launched with")
    s.add_argument("--attest", action="append", metavar="KEY=VALUE", help="sel_mode=N must equal the selection dump's mode")
    s.add_argument("--execution-witness", metavar="JSON", help="launcher/engine execution evidence; required on every floor arm")
    s.add_argument("--select", action="store_true", help="selection dumps are required")
    s.add_argument("--not-before", type=float, help="unix time the run started: older dumps are stale")
    s.add_argument("--model-sha", metavar="SHA", help="weights/pack sha recorded in the seal (checked by compare --mode turbo)")
    w = sub.add_parser("witness")
    w.add_argument("--dir", required=True)
    w.add_argument("--rank", type=int, required=True)
    w.add_argument("--run-id", required=True)
    w.add_argument("--binary", required=True)
    w.add_argument("--prefix", default="q")
    w.add_argument("--rows", type=int, required=True)
    w.add_argument("--prompt", action="append", metavar="NAME=TOKENS.u32")
    w.add_argument("--attest", action="append", metavar="KEY=VALUE", help="sel_mode=N must equal the selection dump's mode")
    w.add_argument("--select", action="store_true", help="selection dumps are required")
    w.add_argument("--not-before", type=float, help="unix time the run started: older dumps are stale")
    w.add_argument("--engine", required=True, choices=sorted(WITNESS_ADAPTERS))
    w.add_argument("--engine-log", required=True, help="the rank's complete engine stdout + stderr")
    w.add_argument("--model", required=True, help="model directory the rank loaded")
    w.add_argument("--out", required=True, help="execution-r<rank>.json to write (then: seal --execution-witness)")
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("cand")
    c.add_argument("--ranks", type=int, nargs="+", default=[0])
    c.add_argument("--tokens", action="append", metavar="NAME=TOKENS.u32")
    c.add_argument("--max-kl", type=float, default=0.02)
    c.add_argument("--min-top1", type=float, default=0.93)
    c.add_argument("--max-dnll", type=float, default=0.01)
    c.add_argument("--dnll-rule", choices=("abs", "floor"), default="abs",
                   help="abs: dNLL <= --max-dnll (default, every engine); floor: dNLL <= mean + 2 sd of the --null-arm runs "
                        "(only engine/profile pairs in DNLL_POLICY: ds41/fast-prefill, minimax/fast-prefill)")
    c.add_argument("--null-arm", action="append", metavar="DIR",
                   help="sealed null-control run on the candidate's binary and prompts (DS41: layer 0; MiniMax: the candidate's first FAST layer; repeat, >= 3 for floor)")
    c.add_argument("--engine", help="engine of the candidate (floor rule policy check), e.g. ds41")
    c.add_argument("--profile", help="profile of the candidate (floor rule policy check), e.g. fast-prefill")
    c.add_argument("--expect-identical", action="store_true", help="repeat of one identity: any difference fails")
    c.add_argument("--json")
    c.add_argument("--mode", choices=("fast", "turbo"), default="fast",
                   help="fast (default): absolute bar; turbo: T's loss vs E must not exceed the reference pack R's")
    c.add_argument("--reference-pack-arm", metavar="DIR", help="turbo: sealed arm of the reference pack R")
    c.add_argument("--reference-sha", help="turbo: weights/pack sha of R (must equal R's seal --model-sha)")
    c.add_argument("--candidate-sha", help="turbo: declared sha of T when its weights differ from E's (TURBO+)")
    c.add_argument("--task-results", metavar="JSON", help="turbo: per-task pass counts of E, T and R")
    c.add_argument("--task-drop-limit", action="append", metavar="CAT=POINTS",
                   help="turbo: max absolute pass-rate drop of T vs E in points (default code=5 math=5 retrieval=2 tools=1)")
    c.add_argument("--turbo-max-kl", type=float, help="turbo: optional absolute safety limit KL(E||T)")
    c.add_argument("--turbo-min-top1", type=float, help="turbo: optional absolute safety limit top-1")
    c.add_argument("--turbo-max-dnll", type=float, help="turbo: optional absolute safety limit dNLL")
    c.add_argument("--receipt", metavar="OUT.json", help="turbo: write a strix-turbo-receipt/v1 receipt")
    c.add_argument("--commit", help="turbo receipt: commit id")
    c.add_argument("--calibration", help="turbo receipt: calibration id")
    c.add_argument("--run-manifest", metavar="JSON", help="turbo receipt: hardware, gpu_clock_mhz, ... (free keys)")
    c.add_argument("--effective-config", metavar="FILE", help="turbo receipt: the engine's --print-effective-config output")
    c.add_argument("--perf", metavar="JSON", help="turbo receipt: tok_s, bytes_per_token, ttft_ms, p95_ms")
    a = ap.parse_args(argv)
    try:
        if a.cmd in ("seal", "witness") and a.rows <= 0:
            raise GateError(f"{a.cmd}: --rows must be positive")
        return {"seal": seal, "witness": witness, "compare": compare}[a.cmd](a)
    except GateError as e:
        print(f"fast_logit_gate: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
