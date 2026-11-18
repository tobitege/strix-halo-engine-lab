#!/usr/bin/env python3
"""mmflash summaries.

  summary.py timing RUNDIR        TTFT per session and warm-session sampled GPU clocks/token, per rank.
  summary.py table CMPDIR TIMING RUNDIR REF
                                  Only expected.tsv reports, validated against current sealed manifests and exit codes.
                                  Exit 0 = pass, 1 = completed gate failure, 2 = incomplete/error.
"""
import json
import math
import os
import re
import sys


def mon(path, start, end, tokens):
    """Left-held samples, clipped to the rank-local wall-clock interval; no extrapolation.

    Require bracketing samples and no gaps over 2 s (monitor cadence is 0.5 s).
    Include idle samples; integrate the product, never the product of averages.
    """
    samples = []
    try:
        with open(path, encoding="utf-8", errors="replace") as stream:
            for line in stream:
                m = re.fullmatch(r"([0-9.]+) busy=(\d+) sclk=.*?(\d+)Mhz \*.*? ev=(\d+)\s*", line, re.I)
                if not m:
                    return None
                t, b, c, e = map(float, m.groups())
                if not all(map(math.isfinite, (t, b, c, e))) or not 0 <= b <= 100:
                    return None
                if samples and t <= samples[-1][0]:
                    return None
                samples.append((t, b, c, e))
    except (OSError, ValueError):
        return None
    if not (math.isfinite(start) and math.isfinite(end) and end > start and tokens > 0):
        return None
    if not samples or samples[0][0] > start or samples[-1][0] < end:
        return None
    busy = clock = clocks = ev = 0.0
    for (t, b, c, e), following in zip(samples, samples[1:]):
        dt = min(end, following[0]) - max(start, t)
        if dt <= 0:
            continue
        if following[0] - t > 2.0:
            return None
        busy += dt * b
        clock += dt * c
        clocks += dt * b / 100.0 * c
        ev = max(ev, e)
    return busy / (end - start), clock / (end - start), int(ev), clocks / tokens


def warm_window(path, sessions):
    with open(path, encoding="utf-8", errors="replace") as stream:
        records = [json.loads(line.removeprefix("MMFLASH_SESSION ")) for line in stream
                   if line.startswith("MMFLASH_SESSION ")]
    if not sessions or [r["prompt"] for r in records] != sessions:
        raise ValueError("session timing identity/coverage mismatch")
    return records[-1]


def timing(run):
    with open(os.path.join(run, "r0.log"), encoding="utf-8", errors="replace") as stream:
        log = stream.read()
    ttft = re.findall(r"TTFT ([0-9.]+) s", log)
    try:
        with open(os.path.join(run, "sessions"), encoding="utf-8") as stream:
            sess = stream.read().strip().split(",")
    except OSError:
        sess = []
    out = [os.path.basename(run), "ttft " + " ".join("%s=%s" % (s, t) for s, t in zip(sess, ttft)) if sess else " ".join(ttft)]
    for r in (0, 1):
        try:
            w = warm_window(os.path.join(run, "r%d.log" % r), sess)
            measured = mon(os.path.join(run, "mon%d.txt" % r), w["start"], w["end"], w["tokens"])
        except (OSError, ValueError, KeyError, TypeError):
            measured = None
        line = ("r%d warm busy %.0f%% %.0fMHz ev %d gclk/tok %.2fM" % (r, *measured)
                if measured is not None else "r%d warm telemetry unavailable" % r)
        out.append(line)
    try:
        with open(os.path.join(run, "mem.txt"), encoding="utf-8") as stream:
            out.append(stream.read().replace("\n", " ").strip())
    except OSError:
        pass
    print(" | ".join(out))


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def checked_report(cmpdir, run, ref, name, arm, expected_rule):
    report = read_json(os.path.join(cmpdir, name + ".json"))
    with open(os.path.join(cmpdir, name + ".rc"), encoding="utf-8") as stream:
        rc = int(stream.read())
    q = report["pooled"]
    if type(q["pass"]) is not bool or rc != (0 if q["pass"] else 1):
        raise ValueError("comparison exit status/report mismatch")
    for side, rid in (("ref", ref), ("cand", arm)):
        manifests = [read_json(os.path.join(run, rid, "manifest-r%d.json" % rank)) for rank in (0, 1)]
        for manifest in manifests:
            if manifest["run_id"] != rid or manifest["arm"] != rid:
                raise ValueError(side + " sealed identity mismatch")
            for key in ("run_id", "arm", "env", "attest"):
                if report[side][key] != manifest[key]:
                    raise ValueError(side + " report identity mismatch: " + key)
            if set(report["prompts"]) != set(manifest["prompts"]):
                raise ValueError(side + " prompt set mismatch")
        if report[side + "_binary"] != [m["binary"]["sha256"] for m in manifests]:
            raise ValueError(side + " binary mismatch")
    rule = q.get("dnll_rule")
    if (isinstance(rule, dict) and bool(rule.get("checks"))) != (expected_rule == "floor"):
        raise ValueError("comparison rule mismatch")
    return q


def table(cmpdir, tim, run, ref):
    status = 0
    try:
        with open(os.path.join(cmpdir, "expected.tsv"), encoding="utf-8") as stream:
            expected = [line.rstrip("\n").split("\t") for line in stream]
        if any(len(row) != 3 for row in expected) or len({row[0] for row in expected}) != len(expected):
            raise ValueError("invalid expected report list")
    except (OSError, ValueError) as e:
        print("INCOMPLETE expected reports:", e)
        return 2
    for name, arm, expected_rule in expected:
        try:
            q = checked_report(cmpdir, run, ref, name, arm, expected_rule)
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(name, "INCOMPLETE", e)
            status = 2
            continue
        status = max(status, 0 if q["pass"] else 1)
        st, sel = q.get("stats") or {}, q.get("selection") or {}
        kq = st.get("kl_q", {})
        rule = q.get("dnll_rule") if isinstance(q.get("dnll_rule"), dict) else {}
        fails = ["%s %.4g>%.4g" % (c["name"], c.get("value", float("nan")), c.get("limit", float("nan")))
                 for c in rule.get("checks", []) if c.get("required") and not c.get("pass")]
        verdict = "%s pass=%s" % (expected_rule, q["pass"])
        print("%s %s kl=%.4f top1=%.4f dnll=%+.4f q50=%.2e q90=%.3f selid=%.3f fails: %s" % (
            name, verdict, q.get("kl", 0), q.get("top1", 0), q.get("dnll", 0), kq.get("q50", 0), kq.get("q90", 0),
            sel.get("id_diff", 0) / max(1, sel.get("slots", 1)), "; ".join(fails) or "-"))
    print("timing:")
    try:
        with open(tim, encoding="utf-8") as stream:
            for line in stream:
                print(" ", line.strip())
    except OSError:
        print("  none")
    return status


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "timing":
        timing(sys.argv[2])
    elif len(sys.argv) == 6 and sys.argv[1] == "table":
        try:
            result = table(*sys.argv[2:])
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            print("INCOMPLETE summary:", error)
            result = 2
        sys.exit(result)
    else:
        sys.exit(__doc__)
