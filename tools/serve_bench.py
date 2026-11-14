#!/usr/bin/env python3
"""Closed-burst load harness for the serving engines (CPU only, standard library only).

For every (concurrency C, max_tokens) pair it sends C streamed requests at once to an OpenAI-style completion
route and reports the metrics every serving change is judged on:

  steady-state aggregate tok/s  tokens emitted by all lanes in the decode cycles in which exactly
                                min(C, slots) lanes decoded and no prompt unit ran, divided by the time of those
                                cycles. Comes from the server's cycle receipts, never from client timestamps.
  per-request tok/s             per request (generated tokens - 1) / (last token time - first token time);
                                p50 and minimum over the burst.
  end-to-end tok/s              sum of output tokens / (last token of the burst - first arrival); TTFT p50/p95.
  context                       active lanes, physical / useful / padding rows per cycle, accepted tokens per
                                physical row (A/R), prompt tokens restored, slots, and the cost model
                                cycle_ms = fixed + row_ms * rows + lane_ms * lanes fitted on the burst's cycles.

Cycle receipts are read from the server log (--log, the bytes the burst appended) and/or from the status route
(--status-path, two snapshots subtracted). Two line forms are understood:

  serve cycle seq=.. lanes=.. rows=.. useful=.. pad=.. emitted=.. draft_rows=a/b/.. draft_ms=.. verify_ms=..
      host_ms=.. ms=.. prompt=0|1 begin_ns=.. end_ns=..        (strix/serve/cycle_receipt.hpp, STRIX_SERVE_RECEIPT=1)
  batch cycle sessions=.. groups=.. rows=.. solo=.. emitted=.. ms=.. begin_ns=.. end_ns=..   (older cycle log)

The older form has no prompt flag and no useful/padding split; the report says so instead of guessing.

A lever is accepted on the aggregate and the end-to-end numbers, never on the sum of per-request rates.
With C above the slot count the burst is min(C, slots) active requests plus queued ones; the report states it.

Examples:
  serve_bench.py --url http://127.0.0.1:8080 --prompt-file prompt.txt --concurrency 1,4,8,16 \\
      --max-tokens 128,512 --receipt --log server.log --out bench.json
  serve_bench.py --replay-events events.json --log server.log --concurrency 4 --max-tokens 128
  serve_bench.py --compare base.json --out candidate.json ...      (token ids identical? rates within noise?)
"""
import argparse
import hashlib
import json
import math
import re
import sys
import threading
import time
import urllib.error
import urllib.request

SCHEMA = "strix.serve_bench.v1"
LEDGER_SCHEMA = "strix.serve.cycle_ledger.v1"
SERVE_CYCLE = re.compile(r"^.*?serve cycle (seq=.*)$", re.M)
BATCH_CYCLE = re.compile(r"^.*?batch cycle (sessions=.*)$", re.M)
PAIR = re.compile(r"(\w+)=(\S+)")
RESTORED_KEYS = ("restored_tokens", "prefix_restored_tokens", "inflight_restored_tokens", "prompt_tokens_restored")
ID_KEYS = ("token_ids", "generated_ids")


class BenchError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------------------
# Cycle receipts


def _number(text):
    try:
        return int(text)
    except ValueError:
        try:
            value = float(text)
        except ValueError:
            return None
        return value if math.isfinite(value) else None


def parse_cycles(log):
    """Server log text -> list of cycles, in log order.

    Each cycle: lanes, rows, emitted, ms (numbers), useful / pad / prompt / draft_ms / verify_ms / host_ms /
    draft_rows (None when the line form does not carry them), begin_ns, end_ns, form ('serve' or 'batch').
    When the log holds 'serve cycle' lines they are used alone (an engine may print both forms for one cycle)."""
    cycles = []
    for line in SERVE_CYCLE.findall(log):
        fields = dict(PAIR.findall(line))
        values = {k: _number(v) for k, v in fields.items() if k != "draft_rows"}
        if any(values.get(k) is None for k in ("lanes", "rows", "useful", "pad", "emitted", "ms")):
            continue
        draft_rows = None
        if re.fullmatch(r"\d+(/\d+)*", fields.get("draft_rows", "")):
            draft_rows = [int(v) for v in fields["draft_rows"].split("/")]
        cycles.append(dict(form="serve", lanes=values["lanes"], rows=values["rows"], useful=values["useful"],
                           pad=values["pad"], emitted=values["emitted"], ms=float(values["ms"]),
                           prompt=bool(values["prompt"]) if values.get("prompt") is not None else None,
                           draft_ms=values.get("draft_ms"), verify_ms=values.get("verify_ms"),
                           host_ms=values.get("host_ms"), draft_rows=draft_rows,
                           begin_ns=values.get("begin_ns"), end_ns=values.get("end_ns")))
    if cycles:
        return cycles
    for line in BATCH_CYCLE.findall(log):
        values = {k: _number(v) for k, v in PAIR.findall(line)}
        if any(values.get(k) is None for k in ("sessions", "rows", "emitted", "ms")):
            continue
        cycles.append(dict(form="batch", lanes=values["sessions"], rows=values["rows"], useful=None, pad=None,
                           emitted=values["emitted"], ms=float(values["ms"]), prompt=None, draft_ms=None,
                           verify_ms=None, host_ms=None, draft_rows=None,
                           begin_ns=values.get("begin_ns"), end_ns=values.get("end_ns")))
    return cycles


def fit_from_sums(s):
    """Least squares of ms = fixed + row_ms * rows + lane_ms * lanes from raw sums (same arithmetic as
    strix::serve::CycleLedger::fit). terms: 3 all identified; 2 rows and lanes moved together or one never moved
    (slope on the one that moved, rows first); 1 mean only; 0 no cycle."""
    out = dict(fixed_ms=0.0, row_ms=0.0, lane_ms=0.0, residual_sd=0.0, cycles=0, terms=0)
    n = s["n"]
    if not n >= 1:
        return out
    out["cycles"] = int(n)
    mr, ml, my = s["r"] / n, s["l"] / n, s["y"] / n
    srr, sll = max(0.0, s["rr"] - n * mr * mr), max(0.0, s["ll"] - n * ml * ml)
    srl, sry, sly = s["rl"] - n * mr * ml, s["ry"] - n * mr * my, s["ly"] - n * ml * my
    syy = max(0.0, s["yy"] - n * my * my)
    rows_moved, lanes_moved = srr > 1e-9 * max(1.0, s["rr"]), sll > 1e-9 * max(1.0, s["ll"])
    det = srr * sll - srl * srl
    if rows_moved and lanes_moved and det > 1e-9 * srr * sll:
        out.update(row_ms=(sry * sll - sly * srl) / det, lane_ms=(sly * srr - sry * srl) / det, terms=3)
    elif rows_moved:
        out.update(row_ms=sry / srr, terms=2)
    elif lanes_moved:
        out.update(lane_ms=sly / sll, terms=2)
    else:
        out["terms"] = 1
    out["fixed_ms"] = my - out["row_ms"] * mr - out["lane_ms"] * ml
    sse = max(0.0, syy - out["row_ms"] * sry - out["lane_ms"] * sly)
    out["residual_sd"] = math.sqrt(sse / (n - out["terms"])) if n > out["terms"] else 0.0
    return out


def cycle_sums(cycles):
    s = dict(n=0.0, r=0.0, l=0.0, rr=0.0, rl=0.0, ll=0.0, y=0.0, ry=0.0, ly=0.0, yy=0.0)
    for c in cycles:
        if not c["lanes"]:
            continue
        r, l, y = float(c["rows"]), float(c["lanes"]), float(c["ms"])
        s["n"] += 1
        s["r"] += r
        s["l"] += l
        s["rr"] += r * r
        s["rl"] += r * l
        s["ll"] += l * l
        s["y"] += y
        s["ry"] += r * y
        s["ly"] += l * y
        s["yy"] += y * y
    return s


def fit_cycles(cycles):
    return fit_from_sums(cycle_sums(cycles))


def _rate(tokens, ms):
    return tokens * 1000.0 / ms if ms > 0 else None


def _ratio(a, b):
    return a / b if b else None


def steady_state(cycles, active):
    """The steady-state window of one burst: cycles with exactly `active` lanes and no prompt unit.

    `active` None (slot count unknown) takes the largest lane count seen. Cycles whose line form has no prompt
    flag cannot be excluded for a prompt unit; prompt_flag reports 'receipt', 'unavailable' or 'none'."""
    if active is None:
        active = max((c["lanes"] for c in cycles), default=0)
    chosen = [c for c in cycles if c["lanes"] == active and c["prompt"] is not True]
    flags = {c["prompt"] is not None for c in chosen}
    ms = sum(c["ms"] for c in chosen)
    emitted = sum(c["emitted"] for c in chosen)
    rows = sum(c["rows"] for c in chosen)
    split = [c for c in chosen if c["useful"] is not None]
    count = len(chosen)
    return dict(
        active_lanes=active, cycles=count, emitted=emitted, window_ms=ms, aggregate_tok_s=_rate(emitted, ms),
        physical_rows_per_cycle=_ratio(rows, count),
        useful_rows_per_cycle=_ratio(sum(c["useful"] for c in split), len(split)) if split else None,
        padding_rows_per_cycle=_ratio(sum(c["pad"] for c in split), len(split)) if split else None,
        accepted_per_row=_ratio(emitted, rows), cycle_ms=_ratio(ms, count),
        prompt_flag="none" if not chosen else "receipt" if flags == {True} else "unavailable",
        source="log")


def find_key(obj, key):
    """First value stored under `key` anywhere in a JSON value (depth first), else None."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_key(value, key)
            if found is not None:
                return found
    return None


def find_ledger(status):
    """The cycle ledger object inside a status document (or the document itself), else None."""
    if isinstance(status, dict) and status.get("schema") == LEDGER_SCHEMA:
        return status
    if isinstance(status, dict):
        for value in status.values():
            found = find_ledger(value)
            if found is not None:
                return found
    elif isinstance(status, list):
        for value in status:
            found = find_ledger(value)
            if found is not None:
                return found
    return None


def ledger_window(before, after, active):
    """Two cumulative ledger snapshots -> the burst's own steady-state row and fit (after - before)."""
    def buckets(ledger):
        return {b["lanes"]: b for b in (ledger or {}).get("by_lanes", [])}
    old, new = buckets(before), buckets(after)
    delta = {}
    for lanes, b in new.items():
        o = old.get(lanes, {})
        row = {k: b[k] - o.get(k, 0) for k in ("cycles", "physical_rows", "useful_rows", "emitted", "total_ms")}
        if row["cycles"] > 0:
            delta[lanes] = row
    if active is None:
        active = max(delta, default=0)
    b = delta.get(active, dict(cycles=0, physical_rows=0, useful_rows=0, emitted=0, total_ms=0.0))
    sums = {k: after["fit_sums"][k] - (before or {}).get("fit_sums", {}).get(k, 0.0) for k in after["fit_sums"]}
    steady = dict(
        active_lanes=active, cycles=b["cycles"], emitted=b["emitted"], window_ms=b["total_ms"],
        aggregate_tok_s=_rate(b["emitted"], b["total_ms"]),
        physical_rows_per_cycle=_ratio(b["physical_rows"], b["cycles"]),
        useful_rows_per_cycle=_ratio(b["useful_rows"], b["cycles"]),
        padding_rows_per_cycle=_ratio(b["physical_rows"] - b["useful_rows"], b["cycles"]),
        accepted_per_row=_ratio(b["emitted"], b["physical_rows"]), cycle_ms=_ratio(b["total_ms"], b["cycles"]),
        prompt_flag="receipt" if b["cycles"] else "none", source="status")
    depth = [a - o for a, o in zip(after["totals"]["draft_rows_by_depth"],
                                   (before or {}).get("totals", {}).get("draft_rows_by_depth", [0] * 64))]
    return steady, fit_from_sums(sums), depth


# ---------------------------------------------------------------------------------------------------------------
# Requests


def percentile(values, q):
    """Linear-interpolated percentile of a non-empty list (q in 0..100); None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def request_metrics(event):
    """One request's event record -> its numbers.

    event: arrival (s), token_times (s, one per content chunk), tokens (usage completion tokens, or None: the
    chunk count is used), optional error. A request with fewer than two tokens has no per-request rate."""
    times = event.get("token_times") or []
    tokens = event.get("tokens")
    if tokens is None:
        tokens = len(times)
    first, last = (times[0], times[-1]) if times else (None, None)
    rate = (tokens - 1) / (last - first) if tokens >= 2 and times and last > first else None
    return dict(tokens=tokens, arrival=event["arrival"], first_token=first, last_token=last,
                ttft_s=first - event["arrival"] if first is not None else None, tok_s=rate,
                prompt_tokens=event.get("prompt_tokens"), restored_tokens=event.get("restored_tokens"),
                ids_sha256=event.get("ids_sha256"), error=event.get("error"))


def burst_metrics(events):
    """The client side of one burst: end-to-end tok/s, TTFT and per-request rates."""
    requests = [request_metrics(e) for e in events]
    done = [r for r in requests if r["error"] is None and r["last_token"] is not None]
    tokens = sum(r["tokens"] for r in done)
    wall = max(r["last_token"] for r in done) - min(r["arrival"] for r in requests) if done else None
    rates = [r["tok_s"] for r in done if r["tok_s"] is not None]
    ttft = [r["ttft_s"] for r in done]
    restored = [r["restored_tokens"] for r in done if r["restored_tokens"] is not None]
    return dict(
        requests=len(requests), failed=len(requests) - len(done), output_tokens=tokens, wall_s=wall,
        end_to_end_tok_s=tokens / wall if wall and wall > 0 else None,
        ttft_p50_s=percentile(ttft, 50), ttft_p95_s=percentile(ttft, 95),
        per_request_tok_s_p50=percentile(rates, 50), per_request_tok_s_min=min(rates) if rates else None,
        sum_per_request_tok_s=sum(rates) if rates else None,
        prompt_tokens=sum(r["prompt_tokens"] or 0 for r in done),
        prompt_tokens_restored=sum(restored) if restored else None,
        ids_sha256=[r["ids_sha256"] for r in requests],
        errors=[r["error"] for r in requests if r["error"] is not None])


def burst_report(concurrency, max_tokens, events, cycles, slots, ledger_before=None, ledger_after=None,
                 limiting=None):
    """Everything the report holds for one (C, max_tokens) burst."""
    active = min(concurrency, slots) if slots else None
    report = dict(concurrency=concurrency, max_tokens=max_tokens, slots=slots, slot_limit=limiting,
                  queued=concurrency - active if active is not None else None, client=burst_metrics(events))
    depth = None
    if ledger_after is not None:
        report["steady"], report["fit"], depth = ledger_window(ledger_before, ledger_after, active)
    if cycles and (ledger_after is None or not report["steady"]["cycles"]):
        report["steady"], report["fit"] = steady_state(cycles, active), fit_cycles(cycles)
        with_depth = [c["draft_rows"] for c in cycles if c["draft_rows"]]
        depth = [sum(column) for column in zip(*with_depth)] if with_depth else None
    if "steady" not in report:
        report["steady"], report["fit"] = steady_state([], active), fit_from_sums(cycle_sums([]))
    report["draft_rows_by_depth"] = depth
    report["log_cycles"] = len(cycles)
    report["prompt_cycles"] = sum(1 for c in cycles if c["prompt"]) if cycles else None
    return report


# ---------------------------------------------------------------------------------------------------------------
# Live client


def request_body(prompt, max_tokens, chat, extra):
    body = dict(max_tokens=max_tokens, temperature=0, stream=True, stream_options=dict(include_usage=True))
    if chat:
        body["messages"] = [dict(role="user", content=prompt)]
    else:
        body["prompt"] = prompt
    body.update(extra)
    return body


def _chunk_has_text(chunk):
    for choice in chunk.get("choices") or []:
        if choice.get("text"):
            return True
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content"):
            return True
    return False


def stream_request(url, body, timeout, clock=time.perf_counter):
    """Sends one streamed request and returns its event record (see request_metrics)."""
    event = dict(arrival=clock(), token_times=[], tokens=None, error=None)
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                now = clock()
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                chunk = json.loads(payload)
                if "error" in chunk:
                    event["error"] = "server error: " + json.dumps(chunk["error"])[:200]
                    break
                if _chunk_has_text(chunk):
                    event["token_times"].append(now)
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    if isinstance(usage.get("completion_tokens"), int):
                        event["tokens"] = usage["completion_tokens"]
                    if isinstance(usage.get("prompt_tokens"), int):
                        event["prompt_tokens"] = usage["prompt_tokens"]
                for key in RESTORED_KEYS:
                    value = find_key(chunk, key)
                    if isinstance(value, int) and not isinstance(value, bool):
                        event["restored_tokens"] = value
                        break
                for key in ID_KEYS:
                    ids = find_key(chunk.get("strix"), key)
                    if isinstance(ids, list):
                        event["ids_sha256"] = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
                        break
    except (urllib.error.URLError, OSError, ValueError) as error:
        event["error"] = f"{type(error).__name__}: {error}"[:200]
    return event


def run_burst(url, prompts, concurrency, max_tokens, chat, extra, timeout):
    """C requests released together; returns their event records in request order."""
    events = [None] * concurrency
    barrier = threading.Barrier(concurrency)

    def worker(index):
        body = request_body(prompts[index % len(prompts)], max_tokens, chat, extra)
        try:
            barrier.wait(timeout=60)
        except threading.BrokenBarrierError:
            pass
        events[index] = stream_request(url, body, timeout)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return events


def get_status(base, path, timeout=30):
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def read_from(path, offset):
    with open(path, "rb") as stream:
        stream.seek(offset)
        return stream.read().decode("utf-8", "replace")


def file_size(path):
    with open(path, "rb") as stream:
        stream.seek(0, 2)
        return stream.tell()


# ---------------------------------------------------------------------------------------------------------------
# Report


def _fmt(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_table(bursts):
    """The metrics table, one row per burst (markdown)."""
    lines = ["| C | max_tokens | active+queued | steady agg tok/s | req tok/s p50 | req tok/s min | e2e tok/s "
             "| TTFT p50 s | TTFT p95 s | rows phys/useful/pad | A/R | cycle ms | restored | fit fixed+row*R+lane*L ms |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in bursts:
        s, c, f = b["steady"], b["client"], b["fit"]
        lanes = (f"{s['active_lanes']}+{b['queued']}" if b["queued"] is not None else f"{s['active_lanes']}+?")
        rows = "/".join(_fmt(s[k]) for k in ("physical_rows_per_cycle", "useful_rows_per_cycle", "padding_rows_per_cycle"))
        fit = (f"{f['fixed_ms']:.1f}+{f['row_ms']:.2f}R+{f['lane_ms']:.2f}L (sd {f['residual_sd']:.1f}, n={f['cycles']}, "
               f"terms={f['terms']})" if f["cycles"] else "n/a")
        lines.append(
            f"| {b['concurrency']} | {b['max_tokens']} | {lanes} | {_fmt(s['aggregate_tok_s'])} | "
            f"{_fmt(c['per_request_tok_s_p50'])} | {_fmt(c['per_request_tok_s_min'])} | {_fmt(c['end_to_end_tok_s'])} | "
            f"{_fmt(c['ttft_p50_s'], 3)} | {_fmt(c['ttft_p95_s'], 3)} | {rows} | {_fmt(s['accepted_per_row'], 2)} | "
            f"{_fmt(s['cycle_ms'])} | {c['prompt_tokens_restored'] if c['prompt_tokens_restored'] is not None else 'n/a'} | "
            f"{fit} |")
    return "\n".join(lines)


def format_notes(bursts):
    notes = []
    for b in bursts:
        tag = f"C={b['concurrency']} max_tokens={b['max_tokens']}"
        s = b["steady"]
        if b["queued"]:
            notes.append(f"{tag}: {s['active_lanes']} active + {b['queued']} queued ({b['slots']} slots"
                         + (f", limited by {b['slot_limit']}" if b["slot_limit"] else "") + ").")
        if b["slots"] is None:
            notes.append(f"{tag}: slot count unknown; the steady window is the largest lane count seen.")
        if not s["cycles"]:
            notes.append(f"{tag}: no cycle receipt in the steady window; the aggregate is not reported "
                         "(client timestamps are not a substitute).")
        elif s["prompt_flag"] == "unavailable":
            notes.append(f"{tag}: the cycle lines carry no prompt flag; cycles next to a prompt unit are included "
                         "and useful/padding rows are unknown.")
        if b["fit"]["cycles"] and b["fit"]["terms"] < 3:
            notes.append(f"{tag}: rows and lanes did not move independently (terms={b['fit']['terms']}); "
                         "the row and lane slopes are not separated in this burst.")
        if b["client"]["failed"]:
            notes.append(f"{tag}: {b['client']['failed']} request(s) failed: {b['client']['errors'][0]}")
    return notes


def compare(base, candidate, noise_pct):
    """Receipt on/off (or any A/B) check: identical token ids per request and rates within noise_pct."""
    rows, ok = [], True
    by_key = {(b["concurrency"], b["max_tokens"]): b for b in base["bursts"]}
    for b in candidate["bursts"]:
        a = by_key.get((b["concurrency"], b["max_tokens"]))
        if a is None:
            continue
        ids_a, ids_b = a["client"]["ids_sha256"], b["client"]["ids_sha256"]
        known = all(i is not None for i in ids_a + ids_b) and bool(ids_a)
        # A burst's requests are interchangeable when they share a prompt: compare as sorted lists.
        same = known and sorted(ids_a) == sorted(ids_b)
        row = dict(concurrency=b["concurrency"], max_tokens=b["max_tokens"],
                   ids="identical" if same else "different" if known else "unavailable")
        for name, pick in (("end_to_end", lambda x: x["client"]["end_to_end_tok_s"]),
                           ("aggregate", lambda x: x["steady"]["aggregate_tok_s"])):
            old, new = pick(a), pick(b)
            row[name + "_delta_pct"] = (new - old) * 100.0 / old if old and new is not None else None
        deltas = [row[k] for k in ("end_to_end_delta_pct", "aggregate_delta_pct") if row[k] is not None]
        row["within_noise"] = bool(deltas) and all(abs(d) <= noise_pct for d in deltas)
        ok = ok and same and row["within_noise"]
        rows.append(row)
    return dict(noise_pct=noise_pct, rows=rows, verdict="PASS" if ok and rows else "FAIL")


# ---------------------------------------------------------------------------------------------------------------
# Entry point


def int_list(text):
    values = [int(v) for v in text.split(",") if v.strip()]
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("expected positive integers separated by commas")
    return values


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", help="server base URL, e.g. http://127.0.0.1:8080")
    p.add_argument("--path", default=None, help="completion route (default /v1/completions, or /v1/chat/completions with --chat)")
    p.add_argument("--chat", action="store_true", help="send chat messages instead of a prompt string")
    p.add_argument("--prompt-file", action="append", default=[], help="UTF-8 prompt file; repeat to cycle prompts over the burst")
    p.add_argument("--concurrency", type=int_list, default=[1, 4, 8, 16])
    p.add_argument("--max-tokens", type=int_list, default=[128])
    p.add_argument("--receipt", action="store_true", help="require cycle receipts (fail when the steady window is empty)")
    p.add_argument("--log", help="server log file (local); the bytes each burst appends are parsed for cycle lines")
    p.add_argument("--status-path", help="status route that carries the cycle ledger and the slot count")
    p.add_argument("--slots", type=int, help="server slot count (default: slots_total of the status route)")
    p.add_argument("--body-extra", default="{}", help="JSON object merged into every request body")
    p.add_argument("--timeout", type=float, default=1800.0, help="per-request timeout in seconds")
    p.add_argument("--settle", type=float, default=0.5, help="seconds to wait after a burst before reading the log")
    p.add_argument("--out", help="write the full result as JSON")
    p.add_argument("--replay-events", help="offline: JSON list of request event records for one burst (no server)")
    p.add_argument("--compare", help="baseline result JSON: report identical token ids and rate deltas")
    p.add_argument("--noise-pct", type=float, default=1.0, help="allowed rate change for --compare (default 1 %%)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        extra = json.loads(args.body_extra)
        if not isinstance(extra, dict):
            raise ValueError("not an object")
    except ValueError as error:
        raise SystemExit(f"--body-extra: {error}")
    bursts = []
    if args.replay_events:
        with open(args.replay_events, encoding="utf-8") as stream:
            events = json.load(stream)
        cycles = parse_cycles(read_from(args.log, 0)) if args.log else []
        bursts.append(burst_report(args.concurrency[0] if len(args.concurrency) == 1 else len(events),
                                   args.max_tokens[0], events, cycles, args.slots))
    else:
        if not args.url or not args.prompt_file:
            raise SystemExit("--url and --prompt-file are required (or --replay-events)")
        prompts = []
        for path in args.prompt_file:
            with open(path, encoding="utf-8") as stream:
                prompts.append(stream.read())
        base = args.url.rstrip("/")
        url = base + (args.path or ("/v1/chat/completions" if args.chat else "/v1/completions"))
        for max_tokens in args.max_tokens:
            for concurrency in args.concurrency:
                status = get_status(base, args.status_path) if args.status_path else None
                slots = args.slots or find_key(status, "slots_total")
                limiting = find_key(status, "slot_limit") or find_key(status, "limiting_term")
                offset = file_size(args.log) if args.log else 0
                events = run_burst(url, prompts, concurrency, max_tokens, args.chat, extra, args.timeout)
                time.sleep(max(args.settle, 0.0))
                cycles = parse_cycles(read_from(args.log, offset)) if args.log else []
                after = get_status(base, args.status_path) if args.status_path else None
                bursts.append(burst_report(concurrency, max_tokens, events, cycles, slots if isinstance(slots, int) else None,
                                           find_ledger(status), find_ledger(after),
                                           limiting if isinstance(limiting, str) else None))
    result = dict(schema=SCHEMA, bursts=bursts, notes=format_notes(bursts))
    print(format_table(bursts))
    for note in result["notes"]:
        print("- " + note)
    failed = any(b["client"]["failed"] for b in bursts)
    if args.receipt and any(not b["steady"]["cycles"] for b in bursts):
        print("FAIL: --receipt was given and a burst has no cycle receipt in its steady window", file=sys.stderr)
        failed = True
    if args.compare:
        with open(args.compare, encoding="utf-8") as stream:
            result["compare"] = compare(json.load(stream), result, args.noise_pct)
        for row in result["compare"]["rows"]:
            print(f"compare C={row['concurrency']} max_tokens={row['max_tokens']}: ids {row['ids']}, "
                  f"e2e {_fmt(row['end_to_end_delta_pct'], 2)} %, aggregate {_fmt(row['aggregate_delta_pct'], 2)} %")
        print("compare verdict: " + result["compare"]["verdict"])
        failed = failed or result["compare"]["verdict"] != "PASS"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=1)
            stream.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
