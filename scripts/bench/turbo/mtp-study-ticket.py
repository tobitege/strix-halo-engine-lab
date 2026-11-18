#!/usr/bin/env python3
"""Run a declared multi-arm plan inside one bounded quiet GPU ticket."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("plan", type=Path)
p.add_argument("--budget", type=int, default=112)
a = p.parse_args()
kit = Path(__file__).resolve().parent
root = Path(os.environ["TURBODEC_ROOT"])
plan = json.loads(a.plan.read_text())
begin = time.monotonic()
for arm in plan:
    left = a.budget - int(time.monotonic() - begin) - 5
    if left < arm.get("min_seconds", 35):
        raise SystemExit(f"insufficient startup budget before {arm['tag']}: {left}s")
    env = dict(os.environ, LEFT=str(left), SLOTS="2", MAXTOK=str(arm.get("tokens", 192)))
    env.update({str(k): str(v) for k, v in arm.get("adapter_env", {}).items()})
    env["TURBODEC_HTTP_BATCH"] = str(arm.get("http_batch", 16))
    env["TURBODEC_PROMPT_MODULE"] = str(kit / arm.get("prompts", "decode-study-inventory.py"))
    env["XARGS"] = shlex.join(arm.get("args", []))
    # XARGS is consumed as shell words by the existing adapter. Only simple
    # option/value arguments are accepted; asset paths belong in the adapter.
    if any(shlex.quote(x) != x for x in arm.get("args", [])):
        raise SystemExit("arm arguments must be simple option/value words")
    settings = arm.get("env", [])
    if arm.get("logits", False):
        settings = settings + [f"STRIX_MTP_LOGITS_DUMP={root / (arm['tag'] + '.logits')}"]
    subprocess.run(["bash", str(kit / "decode-study-ticket.sh"), arm["tag"], "source",
                    arm.get("requests", "C1,C1"), "STRIX_PLE_REWARM=0",
                    "STRIX_HTTP_SLOT_TIMING=1", *settings], env=env, check=True)
    result = root / "runs" / arm["tag"] / "res.jsonl"
    rows = [json.loads(x) for x in result.read_text().splitlines()]
    expected_requests = len(arm.get("requests", "C1,C1").split(","))
    if len(rows) != expected_requests:
        raise SystemExit(f"incomplete requests for {arm['tag']}: {len(rows)}/{expected_requests}")
    if any(
        r["errors"] or not r["tokens"] or not all(l["cycle_accounting_valid"] for l in r["lanes"])
        for r in rows
    ):
        raise SystemExit(f"invalid request accounting: {arm['tag']}")
    print(arm["tag"], "valid", "elapsed_s", round(time.monotonic() - begin, 2), flush=True)
    subprocess.run(["python3", str(kit / "decode-study-phase-summary.py"), str(result.parent)],
                   stdout=(result.parent / "phase-summary.txt").open("w"), check=True)
(root / (a.plan.stem + ".done")).write_text("all declared arms valid\n")
