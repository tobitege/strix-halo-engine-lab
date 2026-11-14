#!/usr/bin/env python3
"""Host tests of tools/serve_bench.py: cycle-line parsing, the cost-model fit, burst metrics, ledger snapshots,
and one live run against a loopback stream server (no engine, no GPU)."""
import contextlib
import http.server
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import serve_bench as sb  # noqa: E402

# A recorded-form server log: the older cycle line exactly as the greedy runtime prints it (with the optional
# witness suffix), between unrelated lines. Four lanes at 4 rows each, then the tail of the burst.
BATCH_LOG = """\
[serve] listening on 127.0.0.1:8080 slots=8
prompt unit rows=2048 ms=3110.250000
batch cycle sessions=1 groups=0 rows=0 solo=1 emitted=6 ms=96.500000 begin_ns=1000000000 end_ns=1096500000
batch cycle sessions=4 groups=1 rows=16 solo=0 emitted=14 ms=168.000000 begin_ns=1100000000 end_ns=1268000000 shared_windows=1 draft_declines=none
batch cycle sessions=4 groups=1 rows=16 solo=0 emitted=13 ms=167.000000 begin_ns=1270000000 end_ns=1437000000 shared_windows=1 draft_declines=margin:2,cap:1
[serve] request 3 finished tokens=128
batch cycle sessions=4 groups=1 rows=16 solo=0 emitted=15 ms=169.000000 begin_ns=1440000000 end_ns=1609000000
batch cycle sessions=3 groups=1 rows=12 solo=0 emitted=9 ms=138.000000 begin_ns=1610000000 end_ns=1748000000
batch cycle sessions=2 groups=1 rows=8 solo=0 emitted=7 ms=108.000000 begin_ns=1750000000 end_ns=1858000000
batch cycle sessions=x groups=1 rows=8 solo=0 emitted=7 ms=108.000000
"""

SERVE_LOG = """\
serve cycle seq=0 lanes=4 rows=16 useful=13 pad=3 emitted=12 draft_rows=4/4/3/2/0/0/0/0 draft_ms=20.000000 verify_ms=140.000000 host_ms=1.000000 ms=161.000000 prompt=1 begin_ns=10 end_ns=20
serve cycle seq=1 lanes=4 rows=16 useful=14 pad=2 emitted=14 draft_rows=4/4/4/2/0/0/0/0 draft_ms=20.000000 verify_ms=140.000000 host_ms=1.000000 ms=160.000000 prompt=0 begin_ns=20 end_ns=30
serve cycle seq=2 lanes=4 rows=16 useful=12 pad=4 emitted=13 draft_rows=4/4/3/1/0/0/0/0 draft_ms=20.000000 verify_ms=141.000000 host_ms=1.000000 ms=162.000000 prompt=0 begin_ns=30 end_ns=40
serve cycle seq=3 lanes=2 rows=8 useful=8 pad=0 emitted=7 draft_rows=2/2/2/2/0/0/0/0 draft_ms=10.000000 verify_ms=96.000000 host_ms=1.000000 ms=108.000000 prompt=0 begin_ns=40 end_ns=50
batch cycle sessions=4 groups=1 rows=16 solo=0 emitted=99 ms=1.000000 begin_ns=1 end_ns=2
"""


def cycle(lanes, rows, ms, emitted=None, prompt=False):
    return dict(form="serve", lanes=lanes, rows=rows, useful=rows, pad=0, emitted=rows if emitted is None else emitted,
                ms=ms, prompt=prompt, draft_ms=0.0, verify_ms=ms, host_ms=0.0, draft_rows=None, begin_ns=0, end_ns=0)


class CycleParsing(unittest.TestCase):
    def test_recorded_batch_log(self):
        cycles = sb.parse_cycles(BATCH_LOG)
        self.assertEqual([c["lanes"] for c in cycles], [1, 4, 4, 4, 3, 2])  # the malformed line is dropped
        self.assertTrue(all(c["form"] == "batch" and c["prompt"] is None and c["useful"] is None for c in cycles))
        self.assertEqual(cycles[1]["begin_ns"], 1100000000)
        steady = sb.steady_state(cycles, 4)
        self.assertEqual((steady["cycles"], steady["emitted"]), (3, 42))
        self.assertAlmostEqual(steady["window_ms"], 504.0)
        self.assertAlmostEqual(steady["aggregate_tok_s"], 42 * 1000.0 / 504.0)
        self.assertAlmostEqual(steady["accepted_per_row"], 42 / 48)
        self.assertEqual(steady["prompt_flag"], "unavailable")
        self.assertIsNone(steady["useful_rows_per_cycle"])
        # Slot count unknown: the largest lane count seen is the window.
        self.assertEqual(sb.steady_state(cycles, None)["active_lanes"], 4)
        # 8 slots and C=8 never reached 8 lanes in this log: the window is empty and no rate is invented.
        empty = sb.steady_state(cycles, 8)
        self.assertEqual(empty["cycles"], 0)
        self.assertIsNone(empty["aggregate_tok_s"])

    def test_serve_lines_win_and_prompt_cycles_leave_the_window(self):
        cycles = sb.parse_cycles(SERVE_LOG)
        self.assertEqual(len(cycles), 4)  # the older form is ignored once receipts exist
        self.assertEqual(cycles[0]["draft_rows"], [4, 4, 3, 2, 0, 0, 0, 0])
        self.assertTrue(all(c["useful"] + c["pad"] == c["rows"] for c in cycles))
        steady = sb.steady_state(cycles, 4)
        self.assertEqual((steady["cycles"], steady["emitted"], steady["prompt_flag"]), (2, 27, "receipt"))
        self.assertAlmostEqual(steady["aggregate_tok_s"], 27 * 1000.0 / 322.0)
        self.assertAlmostEqual(steady["useful_rows_per_cycle"], 13.0)
        self.assertAlmostEqual(steady["padding_rows_per_cycle"], 3.0)
        self.assertAlmostEqual(steady["physical_rows_per_cycle"], 16.0)

    def test_format_matches_the_header(self):
        # The line strix/serve/cycle_receipt.hpp prints (its own test pins the same text).
        line = ("serve cycle seq=7 lanes=4 rows=16 useful=13 pad=3 emitted=14 draft_rows=0/4/0/0/0/0/0/0 "
                "draft_ms=20.500000 verify_ms=138.000000 host_ms=2.750000 ms=161.250000 prompt=1 begin_ns=123 end_ns=456")
        (c,) = sb.parse_cycles("[engine] " + line + "\n")
        self.assertEqual((c["lanes"], c["rows"], c["useful"], c["pad"], c["emitted"], c["prompt"]), (4, 16, 13, 3, 14, True))
        self.assertEqual((c["ms"], c["draft_ms"], c["verify_ms"], c["host_ms"]), (161.25, 20.5, 138.0, 2.75))
        self.assertEqual((c["begin_ns"], c["end_ns"], c["draft_rows"][1]), (123, 456, 4))


class Fit(unittest.TestCase):
    def test_recovers_planted_coefficients(self):
        cycles = [cycle(lanes, lanes * per, 48.0 + 2.69 * lanes * per + 9.78 * lanes)
                  for lanes in range(1, 9) for per in (1, 2, 4, 8)]
        fit = sb.fit_cycles(cycles)
        self.assertEqual((fit["terms"], fit["cycles"]), (3, 32))
        self.assertAlmostEqual(fit["fixed_ms"], 48.0, places=6)
        self.assertAlmostEqual(fit["row_ms"], 2.69, places=7)
        self.assertAlmostEqual(fit["lane_ms"], 9.78, places=6)
        self.assertLess(fit["residual_sd"], 1e-5)

    def test_degenerate_inputs(self):
        collinear = sb.fit_cycles([cycle(lanes, lanes * 4, 48.0 + 7.5 * lanes * 4) for lanes in range(1, 9)])
        self.assertEqual(collinear["terms"], 2)
        self.assertAlmostEqual(collinear["row_ms"], 7.5)
        self.assertEqual(collinear["lane_ms"], 0.0)
        self.assertAlmostEqual(collinear["fixed_ms"], 48.0)
        flat = sb.fit_cycles([cycle(4, 16, 160.0)] * 5)
        self.assertEqual((flat["terms"], flat["row_ms"], flat["lane_ms"]), (1, 0.0, 0.0))
        self.assertAlmostEqual(flat["fixed_ms"], 160.0)
        self.assertEqual(sb.fit_cycles([])["terms"], 0)
        self.assertEqual(sb.fit_cycles([cycle(0, 0, 5.0)])["cycles"], 0)


class Requests(unittest.TestCase):
    EVENTS = [
        dict(arrival=10.0, token_times=[12.0 + 0.1 * i for i in range(11)], tokens=11, restored_tokens=2048, ids_sha256="a"),
        dict(arrival=10.5, token_times=[13.0 + 0.2 * i for i in range(11)], tokens=11, restored_tokens=0, ids_sha256="b"),
        dict(arrival=10.0, token_times=[14.0, 14.5, 16.0], tokens=None, ids_sha256="c"),
        dict(arrival=10.2, token_times=[15.5], tokens=1, ids_sha256="d"),
    ]

    def test_request_metrics(self):
        first = sb.request_metrics(self.EVENTS[0])
        self.assertAlmostEqual(first["ttft_s"], 2.0)
        self.assertAlmostEqual(first["tok_s"], 10.0)  # (11 - 1) / 1.0 s
        self.assertEqual(sb.request_metrics(self.EVENTS[2])["tokens"], 3)  # no usage: chunk count
        self.assertIsNone(sb.request_metrics(self.EVENTS[3])["tok_s"])  # one token has no rate

    def test_burst_metrics(self):
        m = sb.burst_metrics(self.EVENTS)
        self.assertEqual((m["requests"], m["failed"], m["output_tokens"]), (4, 0, 26))
        self.assertAlmostEqual(m["wall_s"], 6.0)  # last token 16.0 - first arrival 10.0
        self.assertAlmostEqual(m["end_to_end_tok_s"], 26 / 6.0)
        self.assertAlmostEqual(m["per_request_tok_s_min"], 1.0)  # (3 - 1) / 2.0 s
        self.assertAlmostEqual(m["per_request_tok_s_p50"], 5.0)  # rates 10, 5, 1
        self.assertAlmostEqual(m["ttft_p50_s"], (2.5 + 4.0) / 2)  # 2.0, 2.5, 4.0, 5.3
        self.assertAlmostEqual(m["ttft_p95_s"], 4.0 + 1.3 * 0.85)
        self.assertEqual(m["prompt_tokens_restored"], 2048)
        self.assertEqual(m["ids_sha256"], ["a", "b", "c", "d"])

    def test_failed_request_is_counted_and_excluded(self):
        events = [self.EVENTS[0], dict(arrival=10.0, token_times=[], tokens=None, error="URLError: refused")]
        m = sb.burst_metrics(events)
        self.assertEqual((m["failed"], m["output_tokens"]), (1, 11))
        self.assertEqual(m["errors"], ["URLError: refused"])
        self.assertIsNone(sb.burst_metrics([events[1]])["end_to_end_tok_s"])

    def test_percentile(self):
        self.assertIsNone(sb.percentile([], 50))
        self.assertEqual(sb.percentile([3.0], 95), 3.0)
        self.assertAlmostEqual(sb.percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)

    def test_queue_is_reported(self):
        report = sb.burst_report(16, 128, self.EVENTS, sb.parse_cycles(SERVE_LOG), 8, limiting="memory")
        self.assertEqual((report["steady"]["active_lanes"], report["queued"]), (8, 8))
        notes = sb.format_notes([report])
        self.assertTrue(any("8 active + 8 queued (8 slots, limited by memory)" in n for n in notes))
        self.assertTrue(any("no cycle receipt in the steady window" in n for n in notes))
        self.assertIn("| 16 | 128 | 8+8 | n/a |", sb.format_table([report]))
        self.assertEqual(report["draft_rows_by_depth"], [14, 14, 12, 7, 0, 0, 0, 0])
        self.assertEqual(report["prompt_cycles"], 1)


def ledger(by_lanes, sums, depth):
    return dict(schema=sb.LEDGER_SCHEMA, by_lanes=by_lanes, fit_sums=sums, totals=dict(draft_rows_by_depth=depth))


class Ledger(unittest.TestCase):
    def test_snapshots_subtract(self):
        warm = [cycle(1, 8, 100.0)] * 3
        burst = [cycle(lanes, lanes * per, 40.0 + 5.0 * lanes * per + 2.0 * lanes, emitted=lanes * per - 1)
                 for lanes in (2, 4) for per in (2, 4)] * 2

        def snapshot(cycles):
            rows = {}
            for c in cycles:
                b = rows.setdefault(c["lanes"], dict(lanes=c["lanes"], cycles=0, physical_rows=0, useful_rows=0,
                                                     emitted=0, total_ms=0.0))
                b["cycles"] += 1
                b["physical_rows"] += c["rows"]
                b["useful_rows"] += c["useful"]
                b["emitted"] += c["emitted"]
                b["total_ms"] += c["ms"]
            return ledger(list(rows.values()), sb.cycle_sums(cycles), [len(cycles)] * 8)

        before, after = snapshot(warm), snapshot(warm + burst)
        # The status document may nest the ledger.
        self.assertIs(sb.find_ledger(dict(strix=dict(deployment=[1, dict(cycle_ledger=after)]))), after)
        self.assertIsNone(sb.find_ledger(dict(slots_total=8)))
        steady, fit, depth = sb.ledger_window(before, after, 4)
        self.assertEqual((steady["cycles"], steady["emitted"], steady["source"]), (4, 2 * (7 + 15), "status"))
        self.assertAlmostEqual(steady["window_ms"], 2 * (88.0 + 128.0))
        self.assertAlmostEqual(steady["physical_rows_per_cycle"], 12.0)
        self.assertEqual((fit["cycles"], fit["terms"]), (8, 3))  # the warm-up cycles are subtracted out
        self.assertAlmostEqual(fit["row_ms"], 5.0, places=6)
        self.assertAlmostEqual(fit["lane_ms"], 2.0, places=6)
        self.assertAlmostEqual(fit["fixed_ms"], 40.0, places=5)
        self.assertEqual(depth, [8] * 8)
        # No earlier snapshot: the whole ledger is the window.
        self.assertEqual(sb.ledger_window(None, after, 1)[0]["cycles"], 3)

    def test_compare(self):
        def result(ids, e2e, agg):
            return dict(bursts=[dict(concurrency=4, max_tokens=128, client=dict(ids_sha256=ids, end_to_end_tok_s=e2e),
                                     steady=dict(aggregate_tok_s=agg))])
        base = result(["a", "b"], 30.0, 80.0)
        self.assertEqual(sb.compare(base, result(["b", "a"], 30.2, 79.5), 1.0)["verdict"], "PASS")
        self.assertEqual(sb.compare(base, result(["a", "x"], 30.0, 80.0), 1.0)["verdict"], "FAIL")
        slow = sb.compare(base, result(["a", "b"], 29.0, 80.0), 1.0)
        self.assertEqual((slow["verdict"], slow["rows"][0]["within_noise"]), ("FAIL", False))
        self.assertEqual(sb.compare(base, result([None, None], 30.0, 80.0), 1.0)["rows"][0]["ids"], "unavailable")


class StreamHandler(http.server.BaseHTTPRequestHandler):
    """Loopback stand-in for a serving engine with two slots: streams max_tokens chunks and appends one cycle line
    per token (lanes = requests in flight, at most 2; one token per 4 ms cycle)."""
    protocol_version = "HTTP/1.0"
    log_path = None
    lock = threading.Lock()
    sequence = 0
    active = 0

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps(dict(slots_total=2, slot_limit="memory")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        count = request["max_tokens"]
        with StreamHandler.lock:
            StreamHandler.active += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for index in range(count):
            time.sleep(0.002)
            with StreamHandler.lock:
                with open(StreamHandler.log_path, "a", encoding="utf-8") as log:
                    log.write(f"serve cycle seq={StreamHandler.sequence} lanes={min(StreamHandler.active, 2)} rows=8 "
                              "useful=6 pad=2 emitted=1 draft_rows=2/2/1/1/0/0/0/0 draft_ms=1.000000 "
                              "verify_ms=2.000000 host_ms=0.500000 ms=4.000000 prompt=0 begin_ns=0 end_ns=0\n")
                StreamHandler.sequence += 1
            chunk = dict(choices=[dict(text=f"t{index}", finish_reason=None)])
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
        final = dict(choices=[dict(text="", finish_reason="length")],
                     usage=dict(prompt_tokens=5, completion_tokens=count),
                     strix=dict(token_ids=list(range(count)), prefix=dict(restored_tokens=3)))
        with StreamHandler.lock:
            StreamHandler.active -= 1
        self.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\ndata: [DONE]\n\n")


class LiveLoopback(unittest.TestCase):
    def test_burst_against_a_loopback_server(self):
        with tempfile.TemporaryDirectory() as folder:
            log = os.path.join(folder, "server.log")
            prompt = os.path.join(folder, "prompt.txt")
            out = os.path.join(folder, "bench.json")
            with open(log, "w", encoding="utf-8") as stream:
                stream.write("batch cycle sessions=9 groups=1 rows=99 solo=0 emitted=99 ms=1.000000\n")  # before the burst
            with open(prompt, "w", encoding="utf-8") as stream:
                stream.write("hello")
            StreamHandler.log_path = log
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StreamHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                argv = ["--url", f"http://127.0.0.1:{server.server_address[1]}", "--prompt-file", prompt,
                        "--concurrency", "1,3", "--max-tokens", "6", "--receipt", "--log", log,
                        "--status-path", "/status", "--settle", "0", "--out", out]
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = sb.main(argv)
                self.assertEqual(code, 0, stdout.getvalue())
                with open(out, encoding="utf-8") as stream:
                    result = json.load(stream)
                self.assertEqual(result["schema"], sb.SCHEMA)
                one, three = result["bursts"]
                self.assertEqual((one["client"]["output_tokens"], three["client"]["output_tokens"]), (6, 18))
                self.assertEqual((three["slots"], three["queued"], three["slot_limit"]), (2, 1, "memory"))
                self.assertEqual(three["log_cycles"], 18)  # only the bytes the burst appended
                self.assertEqual(three["steady"]["active_lanes"], 2)
                self.assertEqual((one["steady"]["active_lanes"], one["steady"]["cycles"], one["queued"]), (1, 6, 0))
                self.assertAlmostEqual(three["steady"]["aggregate_tok_s"], 250.0)  # 1 token per 4 ms cycle
                self.assertAlmostEqual(three["steady"]["accepted_per_row"], 0.125)
                self.assertEqual(three["steady"]["prompt_flag"], "receipt")
                self.assertEqual(three["client"]["prompt_tokens_restored"], 9)
                self.assertEqual(len(set(three["client"]["ids_sha256"])), 1)
                self.assertGreater(three["client"]["end_to_end_tok_s"], 0.0)
                self.assertGreaterEqual(three["client"]["ttft_p95_s"], three["client"]["ttft_p50_s"])
                self.assertTrue(any("2 active + 1 queued" in note for note in result["notes"]))
                self.assertIn("| 3 | 6 | 2+1 | 250.0 |", stdout.getvalue())
                # The same load again, compared with the first run: identical ids, same aggregate.
                compared = io.StringIO()
                with contextlib.redirect_stdout(compared):
                    code = sb.main(argv[:-2] + ["--compare", out, "--noise-pct", "1000000"])
                self.assertEqual(code, 0, compared.getvalue())
                self.assertIn("ids identical", compared.getvalue())
                self.assertIn("compare verdict: PASS", compared.getvalue())
            finally:
                server.shutdown()
                server.server_close()

    def test_unreachable_server_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            prompt = os.path.join(folder, "prompt.txt")
            with open(prompt, "w", encoding="utf-8") as stream:
                stream.write("hello")
            with contextlib.redirect_stdout(io.StringIO()):
                code = sb.main(["--url", "http://127.0.0.1:9", "--prompt-file", prompt, "--concurrency", "2",
                                "--max-tokens", "4", "--timeout", "5"])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main(verbosity=1)
