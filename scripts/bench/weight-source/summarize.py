#!/usr/bin/env python3
"""Summarize run-fair receipts; rates count emitted tokens, prefills count actual prompt tokens.

Writes complete rows/aggregate JSON, prints a bounded table (one row per cell).
GPU clocks/token = integral(busy_fraction * clock_hz * dt) / tokens over the
client's arrival-to-last-token interval, including TTFT. Samples are held until
the next sample. Missing, sparse, or failed telemetry is reported as unavailable.
"""
import argparse
import collections
import json
import pathlib
import statistics


def clocks(samples, start, end, tokens):
    if not samples or not tokens or end <= start:
        return {}
    samples = sorted(samples, key=lambda s: s["time"])
    relevant = [s for s in samples if start - 1 <= s["time"] <= end + 1]
    if not relevant or relevant[0]["time"] > start + 0.5 or relevant[-1]["time"] < end - 0.5:
        return {}
    if any(b["time"] - a["time"] > 1 for a, b in zip(relevant, relevant[1:])):
        return {}
    boundaries = [start] + [s["time"] for s in relevant if start < s["time"] < end] + [end]
    busy_cycles = busy_time = clock_time = 0.0
    for a, b in zip(boundaries, boundaries[1:]):
        prior = [s for s in samples if s["time"] <= a]
        sample = prior[-1] if prior else samples[0]
        duration = b - a
        clock_time += sample["clock_hz"] * duration
        busy_time += sample["busy"] * duration
        busy_cycles += sample["busy"] * sample["clock_hz"] * duration
    return dict(gpu_clocks_per_token=busy_cycles / tokens, clock_mhz=clock_time / (end - start) / 1e6,
                busy_fraction=busy_time / (end - start), telemetry_interval_s=end - start)


def collect(root):
    rows, errors = [], []
    for path in sorted(root.glob("*/run.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["status"] != "complete":
            errors.append(f"{path.parent.name}: incomplete")
        for rec in manifest["records"]:
            data = json.loads((path.parent / rec["file"]).read_text(encoding="utf-8"))
            c = data["bursts"][0]["client"]
            row = dict(arm=manifest["arm"], pair=manifest["pair"], kind=rec["kind"], state=rec["state"],
                       concurrency=rec.get("concurrency", 1), target_tokens=rec.get("target_tokens"),
                       actual_prompt_tokens=c["prompt_tokens"], emitted_tokens=c["output_tokens"],
                       e2e_tok_s=c["end_to_end_tok_s"], request_tok_s=c["per_request_tok_s_p50"],
                       peak_1s_tok_s=c.get("peak_1s_tok_s"), ttft_s=c["ttft_p50_s"],
                       receipt=str(path.parent / rec["file"]), restored_tokens=c.get("prompt_tokens_restored"))
            if rec["kind"] == "prefill":
                row["prefill_tok_s"] = c["prompt_tokens"] / c["ttft_p50_s"] if c["ttft_p50_s"] else None
                denominator = c["prompt_tokens"]
            else:
                denominator = c["output_tokens"]
            if not manifest.get("sampler_errors"):
                row.update(clocks(manifest["samples"], rec["start"], rec["end"], denominator))
            if "gpu_clocks_per_token" not in row:
                errors.append(f"{path.parent.name}/{rec['file']}: missing valid telemetry")
            rows.append(row)
    return rows, errors


def aggregate(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("arm", "kind", "state", "concurrency", "target_tokens"))
        groups[key].append(row)
    result = []
    metrics = ("e2e_tok_s", "request_tok_s", "peak_1s_tok_s", "ttft_s", "prefill_tok_s",
               "gpu_clocks_per_token", "clock_mhz", "busy_fraction", "actual_prompt_tokens")
    for key, values in groups.items():
        row = dict(zip(("arm", "kind", "state", "concurrency", "target_tokens"), key))
        row["pairs"] = sorted({v["pair"] for v in values})
        row["n"] = len(values)
        for metric in metrics:
            nums = [v[metric] for v in values if v.get(metric) is not None]
            if nums:
                row[metric] = dict(mean=statistics.mean(nums), min=min(nums), max=max(nums), n=len(nums))
        result.append(row)
    return result


def fmt(row, key):
    value = row.get(key, {}).get("mean")
    return "-" if value is None else f"{value:.2f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=pathlib.Path)
    p.add_argument("--out", type=pathlib.Path)
    args = p.parse_args()
    rows, errors = collect(args.root)
    groups = aggregate(rows)
    if not (args.root / ".done").exists():
        errors.append("matrix incomplete: no .done marker")
    for row in groups:
        if len(row["pairs"]) < 3:
            errors.append(f"insufficient pairs: {row['arm']} {row['kind']} {row['state']}")
    output = args.out or args.root / "summary.json"
    output.write_text(json.dumps(dict(rows=rows, aggregate=groups, errors=errors), indent=2), encoding="utf-8")
    print("arm kind state C target actual e2e request peak1s prefill clocks/token MHz busy")
    for row in groups[:55]:
        print(f"{row['arm']} {row['kind']} {row['state']} {row['concurrency']} {row['target_tokens'] or '-'} "
              + " ".join(fmt(row, key) for key in ("actual_prompt_tokens", "e2e_tok_s", "request_tok_s", "peak_1s_tok_s",
                                                    "prefill_tok_s", "gpu_clocks_per_token", "clock_mhz", "busy_fraction")))
    print(f"{len(rows)} receipts; {len(errors)} errors; complete JSON: {output}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
