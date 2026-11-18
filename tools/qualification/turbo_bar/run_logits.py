"""Node runner for the TURBO logits passes. Everything comes from arguments; no host or user specifics.

Spread pass (one engine process per (case, arm); outputs already complete are skipped; exit code 2 = run again):
  python3 -m tools.qualification.turbo_bar.run_logits spread --corpus DIR --manifest M.json --bin BIN --out OUT \
      [--arms E,Erep,R,T:x] [--cases ID,..] [--prefill-rows 4096] [--soft-seconds 430] [--reserve-seconds 20] \
      [--stack-kib 65536] [--bin-for ARM=PATH]
  engine call: BIN MODEL <prompt ids...> --teacher-forced-score --generate 1 --known-prefix-window \
      --known-prefix-full-rows --known-prefix-prefill-rows T --known-prefix-output-policy all-rows \
      --teacher-logits-file OUT/spread/CASE.ARM.bin --teacher-logits-rows POSITIONS  [+ the arm's args, env]
  "Erep" is a second run of the E arm (null control). Per output a CASE.ARM.json sidecar records binary and
  weights shas, manifest version, args, env, rc and the engine receipt route fields found in its output.
  OUT/run.json records the arm identities for `manifest check`.

Continuation pass (256 forced steps) is delegated to the option-qual runner:
  python3 -m tools.qualification.turbo_bar.run_logits continuation --manifest M.json --out ARMS.json [--arms ...]
  writes the ARMS.json that runner expects and prints the command to run it with --keep-logits. Build its
  manifest from the same parents (tools.qualification.option_qual.manifest build) and its plan with the arms
  named as ours; the kept logits OUT/logits/CASE.ARM.bin are read by losses.py as format "steps".
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from tools.qualification.turbo_bar import manifest as mf
from tools.qualification.turbo_bar import spool
from tools.qualification.turbo_bar.common import canon_sha, read_ids, read_json, safe_name, sha256_file, write_json

KEEP_ENV = ("PATH", "HOME", "LD_LIBRARY_PATH", "SYSTEMROOT")
ROUTE_RE = re.compile(r'"?(teacher_forced_tiled_lm_head|teacher_logits_scope|teacher_logits_rows)"?\s*[:=]\s*("[^"]*"|[^\s,}]+)')
FIXED = ["--teacher-forced-score", "--generate", "1", "--known-prefix-window", "--known-prefix-full-rows",
         # The tiled known-prefix route requires every budget explicitly (same values as scripts/bench/engines/qwen.sh).
         "--known-prefix-workspace-budget-bytes", "8589934592", "--known-prefix-host-metadata-budget-bytes", "1073741824",
         "--known-prefix-uma-budget-bytes", "137438953472", "--known-prefix-host-headroom-bytes", "4294967296",
         # The known-prefix window states the resolved v5 organization (D1 + paired wave8 GDN input/output)
         # explicitly (apps/hip_decode_options.hpp validate_known_prefill_options); the serial route resolves it.
         "--deferred-moe-status", "--virtual-q8-gdn-input", "--prefetch-q8-gdn-input", "--virtual-q8-gdn-output",
         "--wave8-q8-gdn-input", "--wave8-q8-gdn-output"]


def clean_env(arm_env):
    env = {k: os.environ[k] for k in KEEP_ENV if k in os.environ}
    for k, v in arm_env.items():
        if not k or "=" in k or not isinstance(v, str):
            raise ValueError(f"bad arm env entry {k!r}")
        env[k] = v
    return env


def engine_argv(binary, model, ids, prefill_rows, out_bin, positions, extra):
    # The context must hold the whole teacher sequence; 8192 is the qualified context, any other length is
    # stated as unqualified (as scripts/bench/engines/qwen.sh does).
    ctx = ["--context", str(len(ids))] + ([] if len(ids) == 8192 else ["--unqualified-context"])
    return ([str(binary), str(model)] + [str(i) for i in ids] + FIXED + ctx +
            ["--known-prefix-prefill-rows", str(prefill_rows), "--known-prefix-output-policy", "all-rows",
             "--teacher-logits-file", str(out_bin), "--teacher-logits-rows", str(positions)] + list(extra))


def route_fields(text):
    out = {}
    for k, v in ROUTE_RE.findall(text):
        out.setdefault(k, v.strip('"'))
    return out


def _limits(stack_kib):
    def fn():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_STACK, (stack_kib * 1024, stack_kib * 1024))
        except Exception:
            pass
    return fn if os.name == "posix" else None


def arm_spec(m, name):
    base = "E" if name == "Erep" else name
    if base not in m["arms"]:
        raise SystemExit(f"arm {name} is not in the manifest")
    return m["arms"][base]


def run_one(argv, env, timeout, stack_kib):
    kw = {"start_new_session": True} if os.name == "posix" else {}
    p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=_limits(stack_kib), **kw)
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out.decode("utf-8", "replace"), False
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                import signal
                os.killpg(p.pid, signal.SIGKILL)
            else:
                p.kill()
        finally:
            out, _ = p.communicate()
        return None, out.decode("utf-8", "replace"), True


def output_contract(m, name, case, corpus, source="spread", prefill_rows=None):
    spec = arm_spec(m, name)
    keys = ("prompt_ids", "positions") if source == "spread" else ("prompt_ids", "cont_ids")
    hashes = {k + "_sha256": sha256_file(Path(corpus) / case[k]) for k in keys}
    if any(hashes[k + "_sha256"] != case[k + "_sha256"] for k in keys):
        raise ValueError("corpus prompt/selection/continuation hash mismatch")
    return {"manifest_core_sha256": m["frozen"]["core_sha256"], "manifest_version": m["version"],
            "arm": name, "arm_sha256": canon_sha(spec), "case": case["case"], "source": source,
            "binary_sha256": spec["binary"]["sha256"], "weights_sha256": [w["sha256"] for w in spec["weights"]],
            "identity": spec["identity"], "prefill_rows": prefill_rows, **hashes}


def validate_output(path, expected):
    path = Path(path)
    side = read_json(path.with_suffix(".json"))
    if side.get("contract") != expected or not side.get("complete") or side.get("rc") != 0:
        raise ValueError(f"{path.name}: output receipt does not match frozen experiment")
    receipt = side.get("receipt")
    if (not receipt or side.get("receipt_sha256") != canon_sha(receipt)
            or side.get("payload_sha256") != sha256_file(path)):
        raise ValueError(f"{path.name}: missing or changed effective route/payload receipt")
    if expected["source"] == "spread" and not all(k in receipt for k in
            ("teacher_forced_tiled_lm_head", "teacher_logits_scope", "teacher_logits_rows")):
        raise ValueError(f"{path.name}: incomplete effective route receipt")
    return side


def preserve_run(out, record):
    path = Path(out) / "run.json"
    if path.exists():
        old = read_json(path)
        if any(old.get(k) != v for k, v in record.items() if k != "arms"):
            raise ValueError("output directory belongs to another experiment; old run.json preserved")
        for name, arm in old.get("arms", {}).items():
            if name in record["arms"] and arm != record["arms"][name]:
                raise ValueError("run arm identity changed; old run.json preserved")
        record = dict(record, arms=dict(old["arms"], **record["arms"]))
    elif any((Path(out) / "spread").glob("*.bin")):
        raise ValueError("outputs without run.json cannot be relabelled")
    return record


def spread(a):
    t0 = time.monotonic()
    m = mf.load(a.manifest)
    if not m["frozen"]:
        raise SystemExit("manifest is not frozen")
    corpus = Path(a.corpus)
    doc = read_json(corpus / "corpus.json")
    if canon_sha(doc) != m["protocol"]["corpus_sha256"]:
        raise ValueError("corpus differs from frozen manifest")
    if not m["model"].lower().startswith("qwen"):
        raise ValueError("spread runner currently supports only the Qwen adapter")
    arms = a.arms.split(",") if a.arms else ["E", "Erep", "R"] + [k for k in m["arms"] if k.startswith("T:")]
    bins = dict(x.split("=", 1) for x in a.bin_for)
    out = Path(a.out)
    (out / "spread").mkdir(parents=True, exist_ok=True)
    bin_sha = {}
    run_arms = {}
    for name in arms:
        spec = arm_spec(m, name)
        b = bins.get(name, a.bin)
        if b not in bin_sha:
            bin_sha[b] = sha256_file(b)
        if bin_sha[b] != spec["binary"]["sha256"]:
            raise SystemExit(f"arm {name}: binary sha256 differs from the frozen manifest")
        models = []
        for w in spec["weights"]:
            if not os.path.exists(w["path"]) or os.path.getsize(w["path"]) != w["bytes"]:
                raise SystemExit(f"arm {name}: weights {w['path']} missing or size differs from the manifest")
            models.append({"sha256": sha256_file(w["path"]), "bytes": w["bytes"], "sha_verified_now": True})
            if models[-1]["sha256"] != w["sha256"]:
                raise SystemExit(f"arm {name}: weights sha256 differs from the frozen manifest")
        run_arms[name] = {"identity": spec["identity"], "binary_sha256": bin_sha[b], "weights": models,
                          "args": spec["args"], "env": spec["env"]}
    record = preserve_run(out, {"manifest_core_sha256": m["frozen"]["core_sha256"], "manifest_version": m["version"],
                                "corpus_sha256": canon_sha(doc), "arms": run_arms})
    reusable = set()
    for case in doc["cases"]:
        for name in arms:
            path = out / "spread" / f"{case['case']}.{safe_name(name)}.bin"
            if spool.complete(path):
                validate_output(path, output_contract(m, name, case, corpus, prefill_rows=a.prefill_rows))
                reusable.add(path)
    write_json(out / "run.json", record)
    want = set(a.cases.split(",")) if a.cases else None
    summary = {"complete": 0, "skipped": 0, "ran": 0, "failed": 0, "deferred": 0}
    deferred = False
    for case in doc["cases"]:
        if want and case["case"] not in want:
            continue
        ids = read_ids(corpus / case["prompt_ids"])
        for name in arms:
            spec = arm_spec(m, name)
            path = out / "spread" / f"{case['case']}.{safe_name(name)}.bin"
            if path in reusable:
                summary["skipped"] += 1
                continue
            remaining = a.soft_seconds - (time.monotonic() - t0) - a.reserve_seconds
            if deferred or remaining < 30:
                deferred = True
                summary["deferred"] += 1
                continue
            for stale in (path, path.with_name(path.name + ".partial")):
                if stale.exists():
                    stale.unlink()
            b = bins.get(name, a.bin)
            argv = engine_argv(b, spec["weights"][0]["path"], ids, a.prefill_rows, path, corpus / case["positions"],
                               spec["args"])
            s = time.monotonic()
            rc, text, timed_out = run_one(argv, clean_env(spec["env"]), remaining, a.stack_kib)
            ok = rc == 0 and spool.complete(path)
            side = {"case": case["case"], "arm": name, "identity": spec["identity"], "rc": rc, "timed_out": timed_out,
                    "complete": ok, "seconds": round(time.monotonic() - s, 2), "binary_sha256": run_arms[name]["binary_sha256"],
                    "weights": run_arms[name]["weights"], "manifest_version": m["version"],
                    "manifest_core_sha256": m["frozen"]["core_sha256"], "prefill_rows": a.prefill_rows,
                    "args": spec["args"], "env": {k: v for k, v in spec["env"].items()}, "receipt": route_fields(text),
                    "log_tail": text[-4000:]}
            side.update(contract=output_contract(m, name, case, corpus, prefill_rows=a.prefill_rows),
                        receipt_sha256=canon_sha(side["receipt"]), payload_sha256=sha256_file(path) if ok else None)
            write_json(path.with_suffix(".json"), side)
            if timed_out:
                deferred = True
                summary["deferred"] += 1
            elif ok:
                summary["ran"] += 1
            else:
                summary["failed"] += 1
    summary["seconds"] = round(time.monotonic() - t0, 1)
    with open(out / "spread-summary.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    if summary["deferred"]:
        return 2
    return 1 if summary["failed"] else 0


def continuation(a):
    m = mf.load(a.manifest)
    names = a.arms.split(",") if a.arms else ["E", "R"] + [k for k in m["arms"] if k.startswith("T:")]
    arms = {n: {"env": arm_spec(m, n)["env"], "argv": arm_spec(m, n)["args"]} for n in names}
    write_json(a.out, {"arms": arms})
    models = sorted({arm_spec(m, n)["weights"][0]["path"] for n in names})
    print("wrote", a.out, "- this runner takes one --model per invocation; models:", len(models))
    print("python3 -m tools.qualification.option_qual.run --manifest OQ_MANIFEST --plan OQ_PLAN --arms", a.out,
          "--binary BIN --model MODEL --model-receipt RECEIPT --out OUT --soft-seconds 430 --keep-logits")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spread")
    s.add_argument("--corpus", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--bin", required=True)
    s.add_argument("--bin-for", action="append", default=[], metavar="ARM=PATH")
    s.add_argument("--out", required=True)
    s.add_argument("--arms", default="")
    s.add_argument("--cases", default="")
    s.add_argument("--prefill-rows", type=int, default=8192)
    s.add_argument("--soft-seconds", type=float, default=430.0)
    s.add_argument("--reserve-seconds", type=float, default=20.0)
    s.add_argument("--stack-kib", type=int, default=65536)
    s.add_argument("--verify-weights-sha", action="store_true", help="compatibility flag; weights are always verified")
    c = sub.add_parser("continuation")
    c.add_argument("--manifest", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--arms", default="")
    a = ap.parse_args(argv)
    return spread(a) if a.cmd == "spread" else continuation(a)


if __name__ == "__main__":
    sys.exit(main())
