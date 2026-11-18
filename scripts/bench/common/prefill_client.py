#!/usr/bin/env python3
"""Served prefill by prompt length through any OpenAI-compatible server (a serve-decode.sh --client-cmd client).

    prefill_client.py --url URL --out RUN_DIR [--chat] [--body-extra JSON] PROMPT_FILE...

Each prompt file is sent once, in order, as its own C1 request with max_tokens 1 (tools/serve_bench.py does the
request). Rows preserve file order; this client neither groups by length nor infers cold/warm cache state.
Use distinct prefixes when measuring uncached prefill. The context actually filled is the
server's usage.prompt_tokens. Prefill tok/s = prompt_tokens / client TTFT (TTFT includes queueing, tokenization and
the first sampled token, the same for every engine). Writes RUN_DIR/client-result.json (completed, failed, rows).
"""
import argparse
import json
import math
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
SERVE_BENCH = HERE.parents[2] / "tools" / "serve_bench.py"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chat", action="store_true")
    p.add_argument("--body-extra", default="{}")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("files", nargs="+")
    a = p.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, failed = [], 0
    for i, f in enumerate(a.files):
        res = out / f"prefill-{i}.json"
        cmd = [sys.executable, "-I", str(SERVE_BENCH), "--url", a.url, "--prompt-file", f, "--concurrency", "1",
               "--max-tokens", "1", "--body-extra", a.body_extra, "--timeout", str(a.timeout), "--out", str(res)]
        if a.chat:
            cmd.append("--chat")
        rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True).returncode
        row = dict(file=pathlib.Path(f).name, rc=rc)
        try:
            client = json.loads(res.read_text(encoding="utf-8"))["bursts"][0]["client"]
            row.update(prompt_tokens=client["prompt_tokens"], ttft_s=client["ttft_p50_s"], failed=client["failed"])
            if type(row["prompt_tokens"]) is not int or row["prompt_tokens"] <= 0:
                raise ValueError("positive measured usage.prompt_tokens required")
            if (type(row["ttft_s"]) not in (int, float) or not math.isfinite(row["ttft_s"])
                    or row["ttft_s"] <= 0):
                raise ValueError("finite positive TTFT required")
            rate = row["prompt_tokens"] / row["ttft_s"]
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("finite positive prefill throughput required")
            row["prefill_tok_s"] = rate
        except (OSError, KeyError, IndexError, ValueError, TypeError) as e:
            row.update(error=f"{type(e).__name__}: {e}", failed=1, prefill_tok_s=None)
        failed += 1 if rc or row.get("failed") else 0
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out / "client-result.json").write_text(
        json.dumps(dict(completed=len(rows) - failed, failed=failed, rows=rows), indent=1), encoding="utf-8")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
