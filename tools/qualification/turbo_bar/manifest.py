"""Per-model E/R/T manifest with freeze.

  python -m tools.qualification.turbo_bar.manifest init --manifest M.json --model NAME ...
  ... add-arm --manifest M.json --role E|R|T [--name VARIANT] --identity TEXT --weight PATH ... --binary PATH
  ... freeze --manifest M.json [--new-version --reason TEXT]
  ... check --manifest M.json RUNDIR
  ... show --manifest M.json

After ``freeze`` the protocol and the E and R arms cannot change; only new T variants may be added, each
recorded in the append-only ledger ``turbo-ledger.jsonl`` next to the manifest. Replacing R (or anything
frozen) requires ``freeze --new-version --reason TEXT``: the version number increases and every earlier
T result is SUPERSEDED; the analysis refuses to mix versions.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from tools.qualification.turbo_bar.common import canon_sha, read_json, read_jsonl, sha256_file, suite_record, write_json

ARITH = ("exact", "controlled", "turbo")
ROLES = ("E", "R", "T")
DEFAULT_PROTOCOL = {
    "checkpoint_sha256": None, "tokenizer_receipt_sha256": None, "chat_template_sha256": None,
    "contexts": [8192, 32768], "retrieval_probe_context": 131072, "seeds": [1234],
    "sampling": {"temperature": 0, "top_k": 1, "top_p": 1, "seed": 1234},
    "corpus_sha256": None, "tasks_sha256": None, "suites": {},
    "margins": {"kl": 0.0, "top1_disagreement": 0.0, "dnll": 0.0, "task_success": 0.0, "retrieval_success": 0.0},
    "alpha": 0.05, "bootstrap": {"B": 10000, "seed": 20240601},
}
REQUIRED_PROTOCOL = ("checkpoint_sha256", "tokenizer_receipt_sha256", "chat_template_sha256",
                     "corpus_sha256", "tasks_sha256")


class ManifestError(Exception):
    pass


def ledger_path(manifest_path):
    return Path(manifest_path).with_name("turbo-ledger.jsonl")


def ledger_append(manifest_path, event):
    event = dict(event, at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with open(ledger_path(manifest_path), "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def core(m):
    """The immutable part after freeze: model, version, protocol, E and R."""
    return {"model": m["model"], "version": m["version"], "protocol": m["protocol"],
            "arms": {k: m["arms"][k] for k in ("E", "R") if k in m["arms"]}}


def core_sha(m):
    return canon_sha(core(m))


def load(path):
    m = read_json(path)
    fr = m.get("frozen")
    if fr and fr.get("core_sha256") != core_sha(m):
        raise ManifestError("frozen manifest was edited (core sha256 differs from the freeze stamp)")
    return m


def save(path, m):
    write_json(path, m)


def parse_sha_from(path):
    table = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64:
            p = parts[1].lstrip("*").strip()
            table[p] = parts[0].lower()
            table[os.path.basename(p)] = parts[0].lower()
    return table


def file_record(path, table=None, hash_now=True):
    p = str(path)
    sha = None
    if table and (p in table or os.path.basename(p) in table):
        sha = table.get(p) or table[os.path.basename(p)]
    elif hash_now:
        sha = sha256_file(p)
    return {"path": p, "sha256": sha, "bytes": os.path.getsize(p) if os.path.exists(p) else None}


def make_arm(role, name, identity, weights, binary, args=(), env=None, arithmetic="exact", produced_from=None):
    if role not in ROLES:
        raise ManifestError(f"role must be one of {ROLES}")
    if arithmetic not in ARITH:
        raise ManifestError(f"arithmetic must be one of {ARITH}")
    return {"role": role, "name": name, "identity": identity, "weights": weights, "binary": binary,
            "args": list(args), "env": dict(env or {}), "arithmetic": arithmetic, "produced_from": produced_from}


def new(model, **protocol):
    p = json.loads(json.dumps(DEFAULT_PROTOCOL))
    for k, v in protocol.items():
        if k not in p:
            raise ManifestError(f"unknown protocol field {k}")
        if isinstance(p[k], dict) and isinstance(v, dict):
            p[k].update(v)
        else:
            p[k] = v
    return {"schema": 1, "model": model, "version": 1, "protocol": p, "arms": {}, "superseded_arms": [],
            "history": [], "frozen": None}


def arm_key(role, name):
    return "T:" + name if role == "T" else role


def add_arm(path, arm, new_variant_ok=True):
    m = load(path)
    key = arm_key(arm["role"], arm["name"])
    if m["frozen"]:
        if arm["role"] != "T":
            raise ManifestError("manifest is frozen: only new T variants may be added "
                                "(replacing E or R needs freeze --new-version --reason TEXT)")
        if key in m["arms"]:
            raise ManifestError(f"T variant {key} already recorded; T variants are append-only")
    m["arms"][key] = arm
    save(path, m)
    ledger_append(path, {"event": "add-arm", "arm": key, "version": m["version"], "frozen": bool(m["frozen"]),
                         "arm_sha256": canon_sha(arm)})
    return m


def validate_complete(m):
    problems = [f"protocol field {k} not set" for k in REQUIRED_PROTOCOL if not m["protocol"].get(k)]
    for k in ("E", "R"):
        if k not in m["arms"]:
            problems.append(f"arm {k} missing")
    for k, a in m["arms"].items():
        if not a["binary"] or not a["binary"].get("sha256"):
            problems.append(f"arm {k}: binary sha256 missing")
        if not a["weights"] or any(not w.get("sha256") for w in a["weights"]):
            problems.append(f"arm {k}: weights sha256 missing")
    return problems


def freeze(path, new_version=False, reason=None, suites=None):
    m = load(path)
    if new_version:
        if not m["frozen"]:
            raise ManifestError("--new-version applies to a frozen manifest")
        if not reason:
            raise ManifestError("--new-version requires --reason TEXT")
        old = m["version"]
        m["history"].append({"version": old, "frozen": m["frozen"], "reason": reason,
                             "R": m["arms"].get("R"), "E": m["arms"].get("E")})
        for k in [k for k in m["arms"] if k.startswith("T:")]:
            m["superseded_arms"].append({"arm": m["arms"].pop(k), "superseded_version": old})
            ledger_append(path, {"event": "superseded", "arm": k, "version": old, "by_version": old + 1,
                                 "reason": reason})
        m["version"] = old + 1
        m["frozen"] = None
        save(path, m)
        ledger_append(path, {"event": "new-version", "version": m["version"], "reason": reason})
        return m
    if m["frozen"]:
        raise ManifestError("already frozen")
    for name, items in (suites or {}).items():
        m["protocol"]["suites"][name] = suite_record(items)
    problems = validate_complete(m)
    if problems:
        raise ManifestError("cannot freeze: " + "; ".join(problems))
    stamp = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "version": m["version"], "core_sha256": core_sha(m)}
    m["frozen"] = None
    stamp["sha256"] = canon_sha(m)  # everything present at freeze time (incl. any T variants)
    m["frozen"] = stamp
    save(path, m)
    ledger_append(path, {"event": "freeze", "version": m["version"], "sha256": stamp["sha256"],
                         "core_sha256": stamp["core_sha256"]})
    return m


def check_run(m, rundir):
    """Compare run.json of RUNDIR (recorded arm identity/binary/weights shas) with the frozen manifest."""
    problems = []
    if not m["frozen"]:
        return ["manifest is not frozen"]
    rj = Path(rundir) / "run.json"
    if not rj.is_file():
        return [f"{rj} missing"]
    run = read_json(rj)
    if run.get("manifest_core_sha256") != m["frozen"]["core_sha256"]:
        problems.append("run was made under a different manifest core sha256")
    if run.get("manifest_version") != m["version"]:
        problems.append(f"run manifest version {run.get('manifest_version')} != current {m['version']}")
    for name, rec in (run.get("arms") or {}).items():
        a = m["arms"].get("E" if name == "Erep" else name)
        if a is None:
            problems.append(f"arm {name} is not in the frozen manifest (unknown or SUPERSEDED)")
            continue
        if rec.get("identity") != a["identity"]:
            problems.append(f"arm {name}: identity differs")
        if (rec.get("binary_sha256") or "") != a["binary"]["sha256"]:
            problems.append(f"arm {name}: binary sha256 differs")
        want = sorted(w["sha256"] for w in a["weights"])
        got = sorted(w.get("sha256") or "" for w in rec.get("weights", []))
        if want != got:
            problems.append(f"arm {name}: weights sha256 differ")
        if list(rec.get("args", a["args"])) != a["args"] or dict(rec.get("env", a["env"])) != a["env"]:
            problems.append(f"arm {name}: args/env differ")
    if not run.get("arms"):
        problems.append("run.json records no arms")
    return problems


def _kv(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k] = v
    return out


def main(argv=None):
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = top.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--manifest", default="turbo-manifest.json")
        return p
    p = common(sub.add_parser("init"))
    p.add_argument("--model", required=True)
    p.add_argument("--set", action="append", default=[], metavar="FIELD=JSON",
                   help="protocol field, e.g. corpus_sha256=\"abc\" or contexts=[8192,32768]")
    p = common(sub.add_parser("add-arm"))
    p.add_argument("--role", required=True, choices=ROLES)
    p.add_argument("--name", help="T variant name (arm id becomes T:NAME)")
    p.add_argument("--identity", required=True)
    p.add_argument("--weight", action="append", required=True, help="weights file (repeatable)")
    p.add_argument("--sha-from", help="precomputed sha256sum output for weights and binary (no hashing)")
    p.add_argument("--no-hash", action="store_true", help="record null sha for files not in --sha-from (freeze refuses)")
    p.add_argument("--binary", required=True)
    p.add_argument("--arg", action="append", default=[])
    p.add_argument("--env", action="append", default=[], metavar="K=V")
    p.add_argument("--arithmetic", choices=ARITH, default=None)
    p.add_argument("--produced-from", help="JSON: tool, args, source sha for an R built from our weights")
    p.add_argument("--effective-route", help="JSON applied HTTP route expected in the server identity receipt")
    p = common(sub.add_parser("freeze"))
    p.add_argument("--new-version", action="store_true")
    p.add_argument("--reason")
    p.add_argument("--tasks-items", help="JSONL suite to bind into the frozen core")
    p.add_argument("--retrieval-items", help="JSONL suite to bind into the frozen core")
    p = common(sub.add_parser("check"))
    p.add_argument("rundir")
    common(sub.add_parser("show"))
    a = top.parse_args(argv)
    try:
        if a.cmd == "init":
            if Path(a.manifest).exists():
                raise ManifestError("manifest exists")
            m = new(a.model, **{k: json.loads(v) for k, _, v in (s.partition("=") for s in a.set)})
            save(a.manifest, m)
            ledger_append(a.manifest, {"event": "init", "model": a.model})
        elif a.cmd == "add-arm":
            if a.role == "T" and not a.name:
                raise ManifestError("--name is required for T variants")
            table = parse_sha_from(a.sha_from) if a.sha_from else None
            hash_now = not a.no_hash
            arm = make_arm(a.role, a.name or a.role, a.identity,
                           [file_record(w, table, hash_now) for w in a.weight],
                           file_record(a.binary, table, hash_now), a.arg, _kv(a.env),
                           a.arithmetic or {"E": "exact", "R": "controlled", "T": "turbo"}[a.role],
                           json.loads(a.produced_from) if a.produced_from else None)
            if a.effective_route:
                arm["effective_route"] = json.loads(a.effective_route)
            add_arm(a.manifest, arm)
        elif a.cmd == "freeze":
            suites = {k: read_jsonl(p) for k, p in (("tasks", a.tasks_items), ("retrieval", a.retrieval_items)) if p}
            m = freeze(a.manifest, a.new_version, a.reason, suites)
            print("frozen" if m["frozen"] else f"version {m['version']} (draft; edit R, then freeze)")
            if m["frozen"]:
                print(m["frozen"]["sha256"])
        elif a.cmd == "check":
            probs = check_run(load(a.manifest), a.rundir)
            for x in probs:
                print("MISMATCH:", x)
            print("OK" if not probs else "FAILED")
            return 1 if probs else 0
        else:
            m = load(a.manifest)
            print(json.dumps({"model": m["model"], "version": m["version"], "frozen": m["frozen"],
                              "arms": {k: {"identity": v["identity"], "arithmetic": v["arithmetic"]}
                                       for k, v in m["arms"].items()},
                              "superseded": [s["arm"]["name"] for s in m["superseded_arms"]],
                              "protocol": m["protocol"]}, indent=1, sort_keys=True))
    except ManifestError as e:
        print("refused:", e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
