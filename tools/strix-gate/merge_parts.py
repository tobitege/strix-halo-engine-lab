"""merge_parts.py RUNDIR RUN ARM...: merge the per-part gate outputs of the DS41 quality runner (RUNDIR/p*/ARM) into RUNDIR/ARM.

Each part is one runner process; its execution witness (strix-fast-execution/v1) covers that part's prompts. The merge is a
launcher statement over producer statements: it refuses parts that disagree on schema, rank, binary, engine, profile, model,
numerical configuration or control, a prompt present in two parts, or a dump whose bytes no longer match its witness entry;
then it links the dumps into RUNDIR/ARM, the token files into RUNDIR/tok, and writes one witness with run_id RUN-ARM (the
seal's --run-id) and the union of the prompts. Prints the seal arguments (--prompt list, not-before) per arm.
"""
import hashlib
import json
import os
import pathlib
import sys

KEYS = ("schema", "rank", "binary_sha256", "engine", "profile", "model_sha256", "numerical_config_sha256", "control")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    root, run, arms = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    parts = sorted(p for p in root.glob("p*") if p.is_dir())
    tok = root / "tok"
    tok.mkdir(exist_ok=True)
    for arm in arms:
        merged, prompts, not_before = None, {}, None
        out = root / arm
        if (out / "execution-r0.json").exists():
            print(f"ARM {arm} already merged")
            continue
        if out.exists():
            raise SystemExit(f"{out} exists without a witness")
        out.mkdir()
        for part in parts:
            wf = part / arm / "execution-r0.json"
            if not wf.is_file():
                continue
            w = json.loads(wf.read_text(encoding="utf-8"))
            base = {k: w[k] for k in KEYS}
            if merged is None:
                merged = base
            elif json.dumps(base, sort_keys=True) != json.dumps(merged, sort_keys=True):
                raise SystemExit(f"{arm}: {part.name} disagrees on the execution identity")
            for name, e in w["prompts"].items():
                if name in prompts:
                    raise SystemExit(f"{arm}: prompt {name} in two parts")
                d = part / arm / e["files"]["tf"]["name"]
                t = part / "tok" / f"{name}.u32"
                if sha(d) != e["files"]["tf"]["sha256"] or d.stat().st_size != e["files"]["tf"]["bytes"]:
                    raise SystemExit(f"{arm}: {d} differs from its witness entry")
                if sha(t) != e["tokens_sha256"]:
                    raise SystemExit(f"{arm}: {t} differs from its witness entry")
                os.link(d, out / d.name)
                if not (tok / t.name).exists():
                    os.link(t, tok / t.name)
                elif sha(tok / t.name) != e["tokens_sha256"]:
                    raise SystemExit(f"{arm}: token file {name} differs between parts")
                prompts[name] = e
                st = int((part / "start.unix").read_text().split()[0])  # written by the part launcher before the runner starts (unix seconds)
                not_before = st if not_before is None else min(not_before, st)
        if merged is None:
            raise SystemExit(f"{arm}: no part has a witness")
        doc = dict(merged)
        doc["run_id"] = f"{run}-{arm}"
        doc["prompts"] = prompts
        (out / "execution-r0.json").write_text(json.dumps(doc, sort_keys=True) + "\n", encoding="utf-8")
        args = " ".join(f"--prompt {n}={tok}/{n}.u32" for n in sorted(prompts))
        print(f"ARM {arm} run_id {run}-{arm} prompts {len(prompts)} not_before {not_before} profile {merged['profile']} "
              f"control {json.dumps(merged['control'])}")
        (out / "seal.args").write_text(args + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
