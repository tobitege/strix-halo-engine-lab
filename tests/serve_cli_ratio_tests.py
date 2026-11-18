#!/usr/bin/env python3
"""Host tests of tools/serve_cli_ratio.py: CLI timing-line parsing per family, the server record from stream chunks,
the ratio gate, and one live run against a loopback stream server (no engine, no GPU)."""
import contextlib
import http.server
import io
import json
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import serve_cli_ratio as scr  # noqa: E402

QWEN = ('{"request":{"available":true,"duration_ms":7494.93,"tokens":8255},'
        '"prefill":{"available":true,"duration_ms":4722.93,"tokens":8192,"gpu_clock_mhz_median":2489},'
        '"decode":{"available":true,"duration_ms":2772,"tokens":63}}\n  "generated_tokens": [3145,7025,283],\n')
GLM = "rank 0 decode: prompt 8192 tokens prefill 4.10 s, 64 tokens, 25.00 ms/token (40.00 tok/s), tokens fnv 0123456789abcdef\n"
MINIMAX = ("rank 0 s: prompt 8192 tokens, TTFT 6.400 s, prefill 1280.0 tok/s (exchange 0.900 s, 12 exchanges)\n"
           "rank 0 s: decode 64 steps at ctx 8192: median(steps 8-64) 30.00 ms/token = 33.33 tok/s; mean 30.10 ms; min 1 max 2\n")
DS41 = "decode: 63 steps after the first, 21.50 tok/s (46.5 ms/token); exchanges 126, 1.0 MB sent\n"
GEMMA = '{"timings": {"prefill_ms": 500.0, "decode_ms": 1000.0}, "prompt_tokens": 1000, "completion_tokens": 50}'


def accounting(prompt, cached=0, imported=0):
    return {"prompt_tokens": prompt, "cached_tokens": cached, "imported_tokens": imported,
            "computed_tokens": prompt - cached - imported}


class Parsing(unittest.TestCase):
    def test_cli_lines(self):
        q = scr.cli_record("qwen", QWEN)
        self.assertEqual(q["prompt_tokens"], 8192)
        self.assertAlmostEqual(q["prefill_tps"], 8192 / 4.72293, places=3)
        self.assertAlmostEqual(q["decode_tps"], 63 / 2.772, places=3)
        self.assertEqual(q["ids"], [3145, 7025, 283])
        g = scr.cli_record("glm", GLM)
        self.assertAlmostEqual(g["prefill_tps"], 8192 / 4.10, places=3)
        self.assertEqual(g["decode_tps"], 40.0)
        m = scr.cli_record("minimax", MINIMAX)
        self.assertEqual((m["prefill_tps"], m["decode_tps"], m["prefill_s"]), (1280.0, 33.33, 6.4))
        self.assertEqual(scr.cli_record("ds41", DS41), {"decode_tps": 21.5})
        e = scr.cli_record("gemma", GEMMA)
        self.assertEqual((e["prefill_tps"], e["decode_tps"]), (2000.0, 50.0))
        self.assertEqual(scr.cli_record("glm", "no timing here"), {})

    def test_server_record(self):
        chunks = [{"choices": [{"text": "a", "token_ids": [5]}]}, {"choices": [{"text": "b", "token_ids": [6]}]},
                  {"choices": [], "usage": {"completion_tokens": 2, "timings": {"prefill_ms": 2000.0, "decode_tokens_per_second": 50.0, "queue_ms": 0.1}}}]
        r = scr.server_record(1000, 10.0, 12.003, 12.5, chunks, accounting(1000))
        self.assertEqual((r["prefill_tps"], r["decode_tps"], r["ids"]), (500.0, 50.0, [5, 6]))
        self.assertAlmostEqual(r["ttft_s"], 2.003)
        final = [{"choices": [{"token_ids": [5]}]}, {"strix": {"prefill_ms": 1000.0, "decode_tokens_per_second": 9.0, "token_ids": [5, 6, 7]}}]
        self.assertEqual(scr.server_record(10, 0.0, 1.0, 2.0, final)["ids"], [5, 6, 7])
        self.assertNotIn("prefill_tps", scr.server_record(10, 0.0, None, 2.0, [{"choices": []}]))

    def test_incomplete_stream_ids(self):
        chunks = [{"choices": [{"token_ids": [5]}]}]
        self.assertIsNone(scr.server_record(10, 0, 1, 2, chunks)["ids"])
        chunks.append({"usage": {"completion_tokens": 2}})
        self.assertIsNone(scr.server_record(10, 0, 1, 2, chunks)["ids"])
        chunks.append({"strix": {"token_ids": [5]}})
        self.assertIsNone(scr.server_record(10, 0, 1, 2, chunks)["ids"])

    def test_gemma_contract(self):
        # This exact fixture is also submitted to the real Gemma validator in shared_api_tests.cpp.
        with open(os.path.join(os.path.dirname(__file__), "serve_cli_ratio_gemma_request.json")) as source:
            expected = json.load(source)
        self.assertEqual(scr.server_request("gemma", "model", 3, [2, 12, 13]), expected)
        chunks = [{"strix": {"generated_ids": [5, 6]}, "usage": {"completion_tokens": 2},
                   "timings": {"prefill_ms": 1000, "decode_tokens_per_second": 2}}]
        rec = scr.server_record(3, 0, 1, 2, chunks, accounting(3))
        self.assertEqual(rec["ids"], [5, 6])
        self.assertTrue(scr.report({"runs": [rec]}, {"runs": [rec]})[1])

    def test_reuse_accounting(self):
        chunks = [{"id": "request-a", "strix": {"token_ids": [1], "prefill_ms": 1000}}]
        receipt = dict(accounting(100, 60, 20), response_id="request-a")
        matched = scr.request_accounting(chunks, [dict(receipt, response_id="other"), receipt])
        rec = scr.server_record(100, 0, 1, 2, chunks, matched)
        self.assertEqual((rec["cached_tokens"], rec["imported_tokens"], rec["computed_tokens"], rec["prefill_tps"]),
                         (60, 20, 20, 20))
        self.assertIsNone(scr.request_accounting(chunks, [dict(receipt, response_id="other")]))
        self.assertIsNone(scr.request_accounting(chunks, [receipt, receipt]))

    def test_ds41_parser_to_report(self):
        cli = {"runs": [scr.cli_record("ds41", DS41 + '\n"generated_tokens": [1,2,3]')]}
        server = {"runs": [dict(cli["runs"][0], **accounting(100), prefill_tps=100, prefill_s=1)]}
        self.assertFalse(scr.report(server, cli)[1])
        self.assertFalse(scr.report(server, cli, skip_decode=True)[1])
        lines, ok = scr.report(server, cli, decode_only=True)
        self.assertTrue(ok, lines)
        self.assertIn("no prefill parity verdict", "\n".join(lines))
        self.assertNotIn("prefill tok/s", "\n".join(lines))
        cli["runs"][0].pop("ids")
        self.assertFalse(scr.report(server, cli, decode_only=True)[1])

    def test_raw_ids_cannot_select_chat(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            scr.main(["server", "--url", "http://unused", "--prompt-ids", "unused", "--out", "unused", "--chat"])
        self.assertEqual(error.exception.code, 2)


class Gate(unittest.TestCase):
    def runs(self, prefill, decode, ids=(1, 2, 3)):
        return {"runs": [{**accounting(1000), "prefill_tps": p, "decode_tps": d, "prefill_s": 1.0, "ttft_s": 1.002, "ids": list(ids)}
                         for p, d in zip(prefill, decode)]}

    def test_within_noise_passes(self):
        server = self.runs([10.0, 995.0, 1000.0, 1005.0], [50.0] * 4)
        server["runs"][0]["warmup"] = True
        lines, ok = scr.report(server, self.runs([1000.0, 1001.0, 999.0], [50.0] * 3))
        self.assertTrue(ok, lines)
        self.assertIn("1.000 ok", lines[1])
        self.assertIn("2.0 ms outside the engine", "\n".join(lines))

    def test_slow_server_fails(self):
        lines, ok = scr.report(self.runs([900.0] * 3, [50.0] * 3), self.runs([1000.0] * 3, [50.0] * 3))
        self.assertFalse(ok)
        self.assertIn("BELOW 0.990", lines[1])
        _, ok = scr.report(self.runs([1000.0] * 3, [40.0] * 3), self.runs([1000.0] * 3, [50.0] * 3))
        self.assertFalse(ok)
        _, ok = scr.report(self.runs([1000.0] * 3, [40.0] * 3), self.runs([1000.0] * 3, [50.0] * 3), skip_decode=True)
        self.assertTrue(ok)

    def test_different_tokens_fail(self):
        lines, ok = scr.report(self.runs([1000.0], [50.0], ids=(1, 2, 3)), self.runs([1000.0], [50.0], ids=(1, 2, 4)))
        self.assertFalse(ok)
        self.assertIn("DIFFERENT", "\n".join(lines))
        _, ok = scr.report(self.runs([1000.0], [50.0], ids=(1, 2, 3, 9)), self.runs([1000.0], [50.0], ids=(1, 2, 3)))
        self.assertFalse(ok)

    def test_missing_ids_fail_every_run(self):
        for side in (0, 1):
            for missing in (None, [], [True], [1.0], [-1]):
                reports = [self.runs([1000.0] * 2, [50.0] * 2) for _ in range(2)]
                reports[side]["runs"][1]["ids"] = missing
                self.assertFalse(scr.report(*reports)[1], (side, missing))
        self.assertFalse(scr.report({"runs": []}, {"runs": []})[1])

    def test_cache_and_import_reuse_fail(self):
        for changes in ({"cached_tokens": 1}, {"imported_tokens": 1}, {"computed_tokens": 999},
                        {"cached_tokens": None}, {"imported_tokens": None}, {"computed_tokens": None}):
            server = self.runs([2000.0] * 2, [50.0] * 2)
            server["runs"][1].update(changes)
            lines, ok = scr.report(server, self.runs([1000.0], [50.0]))
            self.assertFalse(ok, changes)
            self.assertIn("NO-REFERENCE", "\n".join(lines))

    def test_partial_timing_evidence_fails(self):
        for bad in (None, 0, -1, float("nan"), float("inf")):
            server = self.runs([1000.0, bad], [50.0] * 2)
            self.assertFalse(scr.report(server, self.runs([1000.0], [50.0]))[1])

    def test_missing_metric_fails(self):
        _, ok = scr.report({"runs": [{"decode_tps": 5.0}]}, self.runs([1000.0], [5.0]))
        self.assertFalse(ok)


class Handler(http.server.BaseHTTPRequestHandler):
    receipt = None

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"requests": [Handler.receipt] if Handler.receipt else []}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = len(body["prompt"])
        Handler.receipt = dict(accounting(n), response_id="request-a")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in ({"choices": [{"text": "x", "token_ids": [7]}]}, {"choices": [{"text": "y", "token_ids": [8]}]},
                      {"id": "request-a", "choices": [], "usage": {"completion_tokens": 2},
                       "strix": {"prefill_ms": n * 1.0, "decode_tokens_per_second": 25.0, "token_ids": [7, 8]}}):
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


class Live(unittest.TestCase):
    def test_cli_output_ids(self):
        with tempfile.TemporaryDirectory() as d:
            log, ids, out = (os.path.join(d, n) for n in ("cli.log", "output.ids", "cli.json"))
            with open(log, "w") as target:
                target.write(DS41)
            with open(ids, "w") as target:
                target.write("1 2 3")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(scr.main(["cli", "--family", "ds41", "--out", out, log, "--output-ids", ids]), 0)
            with open(out) as source:
                cli = json.load(source)
            self.assertTrue(scr.report(cli, cli, decode_only=True)[1])
            with self.assertRaises(ValueError):
                scr.main(["cli", "--family", "ds41", "--out", out, log, log, "--output-ids", ids])

    def test_loopback(self):
        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                ids, out, cli_log, cli = (os.path.join(d, n) for n in ("ids.txt", "s.json", "cli.log", "c.json"))
                open(ids, "w").write(" ".join(str(i) for i in range(100)))
                open(cli_log, "w").write('"prefill":{"duration_ms":100.0,"tokens":100}, "decode":{"duration_ms":2000,"tokens":50}\n'
                                         '"generated_tokens": [7,8]\n')
                with contextlib.redirect_stdout(io.StringIO()) as text:
                    self.assertEqual(scr.main(["server", "--url", "http://127.0.0.1:%d" % server.server_port, "--prompt-ids", ids,
                                               "--runs", "2", "--out", out]), 0)
                    self.assertEqual(scr.main(["cli", "--family", "qwen", "--out", cli, cli_log]), 0)
                    self.assertEqual(scr.main(["report", out, cli]), 0)
                self.assertIn("output tokens: equal", text.getvalue())
                runs = json.load(open(out))["runs"]
                self.assertEqual([r["warmup"] for r in runs], [True, False, False])
                self.assertEqual(runs[1]["prefill_tps"], 1000.0)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
