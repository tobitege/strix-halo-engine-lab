#!/usr/bin/env python3
"""Run receipts shared by every kit under scripts/bench/ (standard library only).

  receipt.py meta DIR --rc N --wall-ms N --mem TEXT [--host-tag T] [--bin-sha S]
      written by runlib.sh kit_run after the process exits.
  receipt.py summarize DIR --tag TAG [--arm ARM] [--round N] [--tokens N] [--metric NAME=REGEX] ...
      parses one run directory into DIR/receipt.json and prints one line.

Timing. Prefill wall is the LAST match of --prefill-regex (default: the engines' `prefill-wall-ms=` and the
pair `prefill-handoff-seconds=` lines). With two or more prefills in one process the FIRST is reported as cold
and the LAST as warm (methodology: never compare a warm figure with a cold one). --metric adds any other number
(last match of a regex with one group), e.g. decode tok/s.

Power, clock and GPU busy are integrated over the warm prefill window [anchor - prefill, anchor], anchor = the
time stamp of the line that carried the figure (stdout.ts or stderr.ts). GPU clocks per token = busy x sclk x window / tokens.

Request figure (--request-json KEY, for producers that print only the result JSON): the figure is the whole request,
so it is published as a prefill UPPER BOUND (prefill_ms_bounds, prefill_source). Arrival stamps only bound ends:
request and decode include all samples through result arrival; prefill includes all samples through witness arrival.
These conservative sampled envelopes include load and reader delay, and cannot isolate decode from prefill.
Each has evicted_ms and valid_<window>; --scope (default request) picks `valid`. Metrics derived from request
duration must use request scope. Power/clock telemetry is labelled as an estimated prefill envelope.

Validity (a receipt with valid=false is excluded by abab.py and its reasons are printed):
  rc != 0; kfd evicted_ms grew inside the timed window (memory pressure moved our buffers while timing; growth
  during model load is a warning, --evict-whole-run makes it fatal); MemAvailable fell below --min-avail-gb;
  kernel log events (amdgpu / kfd / thunderbolt / oom / reset) in the run are a warning (--strict-klog: fatal);
  --require REGEX lines missing from both streams; --reject REGEX lines present.
"""
import argparse
import hashlib
import json
import os
import re
import sys

KIT_EVICT_PAD_S = 0.25  # retain the sample preceding the window, including boundary uncertainty
DEFAULT_PREFILL = [r"prefill-handoff-seconds=([\d.]+)@s", r"prefill-wall-ms=([\d.]+)"]
DEFAULT_REJECT = [r"what\(\)", r"\btimed out\b", r"^(?!.*cache policy).*\bdeclined\b"]
# Producer formats; timing, row counts and agreement evidence share the same matched rows.
FAMILY_PREFILL = {
    "ds41": r"^ttft run (?P<key>\d+): (?P<tokens>\d+) rows, wall (?P<ms>[\d.]+) ms .*logits_fnv (?P<logits>[0-9a-f]{16})$",
    "glm": r"^rank (?P<rank>[01]) pf (?P<key>\S+-ttft): (?P<tokens>\d+) tokens (?P<seconds>[\d.]+) s .*logits fnv (?P<logits>[0-9a-f]{16}), state fnv (?P<state>[0-9a-f]{16})$",
    "minimax": r"^rank (?P<rank>[01]) (?P<key>\S+): prompt (?P<tokens>\d+) tokens, TTFT (?P<seconds>[\d.]+) s, prefill .*$",
}


def family_rows(text, family):
    return [m.groupdict() for m in re.finditer(FAMILY_PREFILL[family], text, re.M)]


def process_reasons(meta):
    rc = meta.get("rc")
    if rc == 0 or (meta.get("intentional_shutdown") and rc == 143):
        return []
    return ["rc=%s" % rc]


def telemetry_reasons(rows, min_avail_gb, evictions, klog=(), sampler_error=""):
    reasons = []
    if len(rows) < 2:
        reasons.append("insufficient samples")
    if sampler_error:
        reasons.append("sampler failed: " + sampler_error.strip()[:100])
    if evictions > 0:
        reasons.append("kfd evicted_ms +%d in the timed window" % evictions)
    avail = [r[6] for r in rows]
    if avail and max(avail) > 0 and min(avail) / 2 ** 20 < min_avail_gb:
        reasons.append("MemAvailable %.1f GiB < %.1f" % (min(avail) / 2 ** 20, min_avail_gb))
    if klog:
        reasons.append("klog %d events" % len(klog))
    return reasons


def pair_receipts(paths, expected=0):
    """Fail both ranks closed, preserving ordered, full-width evidence and peer validity."""
    recs = [json.loads(read_text(p) or "{}") for p in paths]
    reasons = []
    if not all(r.get("valid") for r in recs):
        reasons.append("pair rank invalid or missing receipt")
    if expected:
        evidence = [r.get("rank_evidence", []) for r in recs]
        if any(len(e) != expected for e in evidence):
            reasons.append("rank evidence missing or unexpected count")
        elif evidence[0] != evidence[1]:
            reasons.append("rank evidence mismatch")
    for p, rec in zip(paths, recs):
        if not rec:
            rec = dict(tag="missing-rank", arm="missing-rank", round=0)
        rec["invalid_reasons"] = list(dict.fromkeys(rec.get("invalid_reasons", []) + reasons))
        rec["valid"] = not rec["invalid_reasons"]
        with open(p, "w") as stream:
            json.dump(rec, stream, indent=1)
    return reasons


def pair_verdict_line(paths):
    """One line for the folded pair arm: VALID only if every rank is; each rank's own reasons named."""
    recs = [json.loads(read_text(p) or "{}") for p in paths]
    tag = recs[0].get("tag") or "missing-rank"
    if all(r.get("valid") for r in recs):
        return "== %s PAIR VALID" % tag
    own = []
    for i, r in enumerate(recs):
        why = [x for x in r.get("invalid_reasons", ["missing receipt"]) if x != "pair rank invalid or missing receipt"]
        own.append("r%d: %s" % (i, "; ".join(why) or "-"))
    return "== %s PAIR INVALID[%s]" % (tag, " | ".join(own))


def read_text(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as stream:
            return stream.read()
    except OSError:
        return ""


def sha_file(path, n=16):
    try:
        with open(path, "rb") as stream:
            return hashlib.sha256(stream.read()).hexdigest()[:n]
    except OSError:
        return "-"


def stdout_json(text):
    """Last JSON object printed on stdout (the engines print one result object)."""
    start = text.rfind("\n{")
    for begin in ([start + 1] if start >= 0 else []) + [text.find("{")]:
        if begin < 0:
            continue
        try:
            return json.loads(text[begin:])
        except ValueError:
            continue
    return {}


def samples(path):
    rows = []
    for line in read_text(path).splitlines():
        part = line.split()
        if len(part) >= 9:
            try:
                rows.append([float(x) for x in part[:9]])
            except ValueError:
                pass
    return rows


def stamps(path):
    out = []
    for line in read_text(path).splitlines():
        part = line.split(" ", 1)
        if len(part) == 2:
            try:
                out.append((float(part[0]), part[1]))
            except ValueError:
                pass
    return out


def prefill_values(err, patterns):
    """All prefill figures in ms, in stream order, from the first pattern that matches at all."""
    for pat in patterns:
        scale = 1.0
        if pat.endswith("@s"):
            pat, scale = pat[:-2], 1e3
        found = [float(v) * scale for v in re.findall(pat, err, re.M)]
        if found:
            return pat, found
    return None, []


def growth(values):
    """Sum of increments: evicted_ms is a per-process counter, it reads 0 once the process has exited."""
    return sum(max(0.0, b - a) for a, b in zip(values, values[1:]))


def window_rows(rows, t0, t1):
    """Samples covering [t0, t1]: the last one at or before t0, those inside, the first at or after t1."""
    before = [r for r in rows if r[0] <= t0]
    after = [r for r in rows if r[0] >= t1]
    return before[-1:] + [r for r in rows if t0 < r[0] < t1] + after[:1], bool(before and after)


def request_windows(rows, anchor, request_ms, out, out_tl):
    """Conservative sampled envelopes for a whole-request figure (--request-json).

    Readers may consume both witnesses after the request stops. Neither arrival minus R nor witness arrival
    bounds the request/decode start from below. Include every earlier sample for each scope; decode cannot be
    separated from prefill. Arrival bounds only the corresponding end. Duration bounds are just [0, R].
    A sample interval straddling an end is charged conservatively; coverage still needs at least R before J.
    Telemetry cannot establish coverage before sampling began."""
    reasons = []
    lines = out.splitlines()
    idx = [i for i, line in enumerate(lines) if line.startswith("{")]
    stop_hi = out_tl[idx[-1]][0] if idx and idx[-1] < len(out_tl) else 0.0
    if not anchor:
        reasons.append("no prefill witness (--anchor-regex) to bound the prefill end")
    if not stop_hi:
        reasons.append("result object on stdout not stamped (stdout.ts)")
    elif anchor and stop_hi < anchor:
        reasons.append("result object stamped before the prefill witness")
    if reasons:
        anchor = stop_hi = anchor or stop_hi
    t0 = min(rows[0][0] if rows else 0.0, stop_hi - request_ms / 1e3)
    windows = {}
    for name, (a, b) in (("prefill", (t0, anchor)), ("decode", (t0, stop_hi)), ("request", (t0, stop_hi))):
        w, covered = window_rows(rows, a, b)
        windows[name] = dict(t0=round(a, 3), t1=round(b, 3), evicted_ms=(growth([r[8] for r in w]) if len(w) > 1 else -1),
                             covered=covered and not reasons)
    return dict(windows=windows, reasons=reasons, decode_ms_bound=round(request_ms, 1),
                prefill_ms_bounds=[0.0, round(request_ms, 1)])


def integrate(rows, t0, t1):
    joules = cycles = busy = span = temp = 0.0
    for a, b in zip(rows, rows[1:]):
        lo, hi = max(a[0], t0), min(b[0], t1)
        if hi > lo:
            dt = hi - lo
            joules += a[1] / 1e6 * dt
            cycles += a[2] * dt
            busy += a[4] * dt
            span += dt
            temp = max(temp, a[3] / 1e3)
    return joules, cycles, busy, span, temp


def cmd_meta(args):
    # The gate reports the effective device after environment/CLI selection and discovery.
    # Keep its complete evidence (including declaration source); absence means unknown.
    logs = read_text(os.path.join(args.dir, "stderr")) + "\n" + read_text(os.path.join(args.dir, "stdout"))
    bindings = list(dict.fromkeys(line for line in logs.splitlines() if line.startswith("device: id=")))
    meta = dict(rc=args.rc, wall_ms=args.wall_ms, mem=args.mem.strip(), host=args.host_tag, bin_sha=args.bin_sha,
                intentional_shutdown=args.intentional_shutdown, device_bindings=bindings)
    with open(os.path.join(args.dir, "meta.json"), "w") as stream:
        json.dump(meta, stream)


def cmd_summarize(args):
    d = args.dir
    meta = json.loads(read_text(os.path.join(d, "meta.json")) or "{}")
    err = read_text(os.path.join(d, "stderr"))
    out = read_text(os.path.join(d, "stdout"))
    out_json = stdout_json(out)
    text = err + "\n" + out
    timing, stream_name = (out, "stdout") if args.family else (err, "stderr")
    fr = family_rows(out, args.family) if args.family else []
    pat, pf = prefill_values(timing, args.prefill_regex or DEFAULT_PREFILL) if not args.family else (
        FAMILY_PREFILL[args.family], [float(v["ms"]) if "ms" in v else float(v["seconds"]) * 1000 for v in fr])
    if not pf and not args.family:
        pat, pf = prefill_values(out, args.prefill_regex or DEFAULT_PREFILL)
        timing, stream_name = out, "stdout"
    req_key, request_ms = None, 0.0
    if not pf and not args.family:
        # Producers that print no prefill line (Qwen single-node: only the result JSON) give the whole request in ms
        # as a JSON field (--request-json). The prefill figure is then an UPPER BOUND (it includes the --generate
        # tokens and the readback) and the windows are placed by request_windows() below.
        for key in args.request_json or []:
            if isinstance(out_json.get(key), (int, float)) and out_json[key] > 0:
                req_key, request_ms = key, float(out_json[key])
                pat, pf, timing, stream_name = None, [request_ms], err, "stderr"
                break
    warm = pf[-1] if pf else 0.0
    cold = pf[0] if pf else 0.0
    tl = stamps(os.path.join(d, stream_name + ".ts"))
    anchor = 0.0
    if pat:
        # stderr.ts holds one stamp per stderr line (same order), so the line index gives the arrival time.
        rx = re.compile(pat)
        idx = [i for i, line in enumerate(timing.splitlines()) if rx.search(line)]
        if idx and idx[-1] < len(tl):
            anchor = tl[idx[-1]][0]
    if args.anchor_regex:
        # The figure came from a JSON field (or a line without a usable stamp): anchor on the named end-of-prefill line.
        rx = re.compile(args.anchor_regex)
        idx = [i for i, line in enumerate(err.splitlines()) if rx.search(line)]
        if idx and idx[-1] < len(tl):
            anchor = tl[idx[-1]][0]
    rows = samples(os.path.join(d, "samples.tsv"))
    req = request_windows(rows, anchor, request_ms, out, stamps(os.path.join(d, "stdout.ts"))) if req_key else None
    telemetry_t0, telemetry_t1 = anchor - warm / 1e3, anchor
    if req is not None:
        telemetry_t0, telemetry_t1 = (req["windows"]["prefill"][key] for key in ("t0", "t1"))
    joules, cyc, busy, span, temp = integrate(rows, telemetry_t0, telemetry_t1)
    sp = span or 1.0
    ev = [r[8] for r in rows]
    t0w = anchor - warm / 1e3
    window, covered = window_rows(rows, t0w, anchor) if pf else (rows, True)
    evw = [r[8] for r in window]
    avail = [r[6] for r in rows]
    klog = [l for l in read_text(os.path.join(d, "klog.txt")).splitlines() if l.strip()]
    metrics = {}
    if "generation_request_elapsed_ms" in out_json:
        metrics["request_latency_ms"] = float(out_json["generation_request_elapsed_ms"])
    for item in args.metric or []:
        name, _, rx = item.partition("=")
        found = re.findall(rx, err) or re.findall(rx, read_text(os.path.join(d, "stdout")))
        if found:
            last = found[-1]
            metrics[name] = float(last[0] if isinstance(last, tuple) else last)
    gen = out_json.get("generated_tokens")
    counts = [int(v["tokens"]) for v in fr] or [args.tokens] * len(pf)
    tokens = counts[-1] if counts else args.tokens
    cold_tokens = counts[0] if counts else args.tokens
    rec = dict(
        tag=args.tag, arm=args.arm or args.tag, round=args.round, host=meta.get("host"), bin_sha=meta.get("bin_sha"),
        rc=meta.get("rc"), wall_ms=meta.get("wall_ms"), mem_start=meta.get("mem"),
        device_bindings=meta.get("device_bindings", []),
        workload=args.workload, prefills=len(pf), prefill_ms_cold=cold, prefill_ms_warm=warm, tokens=tokens,
        prefill_tokens=counts, rank_evidence=[{k: v[k] for k in ("key", "tokens", "logits", "state") if k in v} for v in fr],
        tok_s_cold=(cold_tokens / (cold / 1e3) if cold_tokens and cold else 0.0),
        tok_s_warm=(tokens / (warm / 1e3) if tokens and warm else 0.0),
        window_s=round(span, 3), joules=round(joules, 1), watts=round(joules / sp, 1),
        sclk_mhz=round(cyc / sp / 1e6, 0), busy_pct=round(busy / sp, 1), temp_max_c=round(temp, 0),
        gpu_clocks_per_token=(round((busy / sp / 100.0) * (cyc / sp) * span / tokens, 1) if tokens and span else None),
        logits_sha=sha_file(os.path.join(d, "logits.f32")),
        gen_sha=(hashlib.sha256(json.dumps(gen).encode()).hexdigest()[:12] if gen is not None else "-"),
        identity=out_json.get("prefill_identity"), metrics=metrics,
        evicted_ms_run=(growth(ev) if ev else -1), evicted_ms_window=(growth(evw) if len(evw) > 1 else -1),
        avail_min_gb=(round(min(avail) / 2 ** 20, 1) if avail else -1),
        klog_events=len(klog))
    sampler_error = read_text(os.path.join(d, "sampler.error"))
    warnings = []
    common = process_reasons(meta)
    if klog:
        (common if args.strict_klog else warnings).append("klog %d events: %s" % (len(klog), klog[0][-80:]))
    if rows and not any(r[6] for r in rows):
        warnings.append("host memory and kfd evictions not sampled on this host")
    if args.need_prefill and not pf:
        common.append("no prefill figure")
    for rx in args.require or []:
        if not re.search(rx, text, re.M):
            common.append("missing /%s/" % rx[:40])
    for rx in (args.reject if args.reject is not None else DEFAULT_REJECT):
        if re.search(rx, text, re.M):
            common.append("stderr /%s/" % rx[:30])
    if req is None:
        reasons = common + telemetry_reasons(
            rows, args.min_avail_gb, rec["evicted_ms_run"] if args.evict_whole_run else rec["evicted_ms_window"],
            sampler_error=sampler_error)
        if pf and (not anchor or not covered):
            reasons.append("insufficient timed-window coverage")
        if rec["evicted_ms_run"] > 0 and rec["evicted_ms_window"] <= 0:
            warnings.append("kfd evicted_ms +%d outside the timed window (load)" % rec["evicted_ms_run"])
    else:
        # All scopes share process/log/witness gates; only envelope coverage and pressure differ.
        common += telemetry_reasons(rows, args.min_avail_gb, 0, sampler_error=sampler_error)
        verdicts = {}
        for name, w in req["windows"].items():
            r = list(common)
            evn = rec["evicted_ms_run"] if args.evict_whole_run else w["evicted_ms"]
            if evn > 0:
                r.append("kfd evicted_ms +%d in the %s window" % (evn, name))
            if not w["covered"]:
                r.append("insufficient %s-window coverage" % name)
            r += req["reasons"]
            verdicts[name] = r
            rec["valid_" + name] = not r
        reasons = verdicts[args.scope]
        rec.update(scope=args.scope, request_ms=request_ms, request_key=req_key, windows=req["windows"],
                   prefill_source="request-json %s: upper bound (includes the generated tokens and readback)" % req_key,
                   prefill_ms_bounds=req["prefill_ms_bounds"], decode_ms_bound=req["decode_ms_bound"],
                   telemetry_source="estimated prefill envelope (includes load and reader delay)",
                   telemetry_window=dict(t0=telemetry_t0, t1=telemetry_t1),
                   evicted_ms_window=req["windows"][args.scope]["evicted_ms"])
        for name in ("prefill", "decode", "request"):
            if name != args.scope and verdicts[name]:
                warnings.append("%s receipt INVALID (%s)" % (name, "; ".join(verdicts[name])))
        if rec["evicted_ms_run"] > max(0, req["windows"]["request"]["evicted_ms"]):
            warnings.append("kfd evicted_ms +%d outside the request window (after result arrival)" % (
                rec["evicted_ms_run"] - max(0, req["windows"]["request"]["evicted_ms"])))
    rec["valid"] = not reasons
    rec["invalid_reasons"] = reasons
    rec["warnings"] = warnings
    with open(os.path.join(d, "receipt.json"), "w") as stream:
        json.dump(rec, stream, indent=1)
    extra = " ".join("%s=%g" % kv for kv in metrics.items())
    print("== %s %s rc=%s pf_ms warm=%.1f cold=%.1f (%d) tok/s=%.0f sclk=%.0f busy=%.0f W=%.0f Gclk/tok=%s logits=%s gen=%s %s %s" % (
        rec["tag"], "VALID" if rec["valid"] else "INVALID[" + "; ".join(reasons) + "]", rec["rc"], warm, cold, len(pf),
        rec["tok_s_warm"], rec["sclk_mhz"], rec["busy_pct"], rec["watts"],
        ("%.0f" % rec["gpu_clocks_per_token"]) if rec["gpu_clocks_per_token"] else "-", rec["logits_sha"], rec["gen_sha"],
        extra, rec["mem_start"] or ""))
    if req is not None:
        print("   request=%.1f ms prefill<=%.1f (>=%.1f) decode<=%.1f ms; evicted_ms prefill=%g decode=%g request=%g; "
              "valid prefill=%s decode=%s request=%s" % (
                  request_ms, rec["prefill_ms_bounds"][1], rec["prefill_ms_bounds"][0], rec["decode_ms_bound"],
                  *(req["windows"][n]["evicted_ms"] for n in ("prefill", "decode", "request")),
                  *(rec["valid_" + n] for n in ("prefill", "decode", "request"))))
    for w in warnings:
        print("   warn: " + w[:200])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("meta")
    m.add_argument("dir")
    m.add_argument("--rc", type=int, required=True)
    m.add_argument("--wall-ms", type=int, required=True)
    m.add_argument("--mem", default="")
    m.add_argument("--host-tag", default="local")
    m.add_argument("--bin-sha", default="-")
    m.add_argument("--intentional-shutdown", action="store_true")
    pair = sub.add_parser("pair")
    pair.add_argument("paths", nargs=2)
    pair.add_argument("--expected", type=int, default=0)
    s = sub.add_parser("summarize")
    s.add_argument("dir")
    s.add_argument("--tag", required=True)
    s.add_argument("--arm")
    s.add_argument("--workload", default="")
    s.add_argument("--round", type=int, default=0, help="ABAB round index (abab.py pairs arms by round)")
    s.add_argument("--tokens", type=int, default=0, help="prompt tokens, for prefill tok/s and clocks per token")
    s.add_argument("--family", choices=tuple(FAMILY_PREFILL))
    s.add_argument("--prefill-regex", action="append", help="regex with one float group; suffix @s = seconds")
    s.add_argument("--request-json", "--prefill-json", dest="request_json", action="append",
                   help="result-JSON key with the WHOLE request in ms, used when no --prefill-regex line matched; needs "
                        "--anchor-regex (a prefill line) and stdout.ts; gives request/prefill/decode windows")
    s.add_argument("--scope", choices=("request", "prefill", "decode"), default="request",
                   help="with --request-json: which window decides `valid` (the other verdicts are valid_<scope>)")
    s.add_argument("--anchor-regex", help="stderr line that ends the timed prefill (default: the figure's own line)")
    s.add_argument("--metric", action="append", help="NAME=REGEX (one float group, last match)")
    s.add_argument("--min-avail-gb", type=float, default=float(os.environ.get("KIT_MIN_AVAIL_GB", "2")))
    s.add_argument("--require", action="append")
    s.add_argument("--reject", action="append")
    s.add_argument("--need-prefill", action="store_true")
    s.add_argument("--evict-whole-run", action="store_true", help="any eviction in the run invalidates (decode kits)")
    s.add_argument("--strict-klog", action="store_true", help="kernel-log events invalidate instead of warning")
    args = p.parse_args(argv)
    if args.cmd == "pair":
        reasons = pair_receipts(args.paths, args.expected)
        # The per-rank '== tag VALID' lines are printed before this fold; this line is the pair arm's verdict.
        print(pair_verdict_line(args.paths))
        return bool(reasons)
    {"meta": cmd_meta, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
