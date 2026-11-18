#!/usr/bin/env python3
"""Offline tests of the shared measurement-kit runner (scripts/bench/common): receipts, validity, ABAB summary,
served-decode receipts and prompt files. CPU only, standard library only."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "common")
sys.path.insert(0, COMMON)
import abab  # noqa: E402
import receipt  # noqa: E402
import serve_receipts  # noqa: E402


def make_run(d, lines, samples, stdout="", rc=0, klog=""):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "stderr"), "w") as f:
        f.write("".join(l + "\n" for _, l in lines))
    with open(os.path.join(d, "stderr.ts"), "w") as f:
        f.write("".join("%.3f %s\n" % (t, l[:100]) for t, l in lines))
    with open(os.path.join(d, "samples.tsv"), "w") as f:
        f.write("".join(" ".join(str(x) for x in s) + "\n" for s in samples))
    with open(os.path.join(d, "stdout"), "w") as f:
        f.write(stdout)
    with open(os.path.join(d, "klog.txt"), "w") as f:
        f.write(klog)
    receipt.main(["meta", d, "--rc", str(rc), "--wall-ms", "1000", "--mem", "avail=100"])


def sample(t, ev=0, avail_kb=60 * 2 ** 20, busy=90, sclk=2.9e9):
    return [t, 50e6, sclk, 60000, busy, 1000, avail_kb, avail_kb, ev]


def load(d):
    with open(os.path.join(d, "receipt.json")) as f:
        return json.load(f)


class Receipts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def summarize(self, d, *extra):
        with open(os.devnull, "w") as null:
            old, sys.stdout = sys.stdout, null
            try:
                receipt.main(["summarize", d, "--tag", os.path.basename(d), "--tokens", "8192"] + list(extra))
            finally:
                sys.stdout = old
        return load(d)

    def test_cold_warm_and_clocks(self):
        d = os.path.join(self.out, "a")
        lines = [(100.0, "load"), (102.0, "prefill-wall-ms=1000.0"), (103.0, "x"), (104.0, "prefill-wall-ms=800.0")]
        make_run(d, lines, [sample(100 + 0.2 * i) for i in range(30)])
        r = self.summarize(d, "--need-prefill")
        self.assertTrue(r["valid"], r["invalid_reasons"])
        self.assertEqual((r["prefill_ms_cold"], r["prefill_ms_warm"], r["prefills"]), (1000.0, 800.0, 2))
        self.assertAlmostEqual(r["tok_s_warm"], 8192 / 0.8, places=3)
        # busy 0.9 x 2.9 GHz x 0.8 s / 8192 tokens
        self.assertAlmostEqual(r["gpu_clocks_per_token"], 0.9 * 2.9e9 * r["window_s"] / 8192, delta=1.0)

    def test_eviction_inside_window_invalidates_load_eviction_warns(self):
        lines = [(100.0, "load"), (105.0, "prefill-wall-ms=2000.0")]
        d = os.path.join(self.out, "in")
        make_run(d, lines, [sample(100 + 0.2 * i, ev=(0 if 100 + 0.2 * i < 104 else 50)) for i in range(30)])
        r = self.summarize(d)
        self.assertFalse(r["valid"])
        self.assertIn("timed window", r["invalid_reasons"][0])
        d = os.path.join(self.out, "load")
        # evictions during load, then the counter reads 0 after the process exits: only a warning
        make_run(d, lines, [sample(100 + 0.2 * i, ev=(500 if 101 <= 100 + 0.2 * i < 106 else 0)) for i in range(40)])
        r = self.summarize(d)
        self.assertTrue(r["valid"], r["invalid_reasons"])
        self.assertTrue(any("outside the timed window" in w for w in r["warnings"]))

    def test_json_figure_with_anchor_and_rejects(self):
        d = os.path.join(self.out, "j")
        lines = [(100.0, "qwen cache policy: {\"reason\":\"declined\"}"), (104.0, "qpf2k sched block rows 8192"), (109.0, "exit")]
        make_run(d, lines, [sample(100 + 0.2 * i, ev=(0 if 100 + 0.2 * i < 106 else 9)) for i in range(50)],
                 stdout='{"generation_request_elapsed_ms": 3000, "generated_tokens": [1]}')
        r = self.summarize(d, "--need-prefill")
        self.assertFalse(r["valid"])
        self.assertEqual(r["prefill_ms_warm"], 0.0)
        self.assertEqual(r["metrics"]["request_latency_ms"], 3000.0)
        make_run(d, lines + [(110.0, "terminate called: what(): boom")], [sample(100 + 0.2 * i) for i in range(50)], rc=134)
        r = self.summarize(d)
        self.assertFalse(r["valid"])
        self.assertTrue(any(x.startswith("rc=") for x in r["invalid_reasons"]))

    def test_memory_floor_and_unsampled_host(self):
        d = os.path.join(self.out, "m")
        make_run(d, [(100.0, "prefill-wall-ms=10")], [sample(99.8 + 0.2 * i, avail_kb=2 ** 20) for i in range(5)])
        self.assertFalse(self.summarize(d, "--min-avail-gb", "4")["valid"])
        make_run(d, [(100.0, "prefill-wall-ms=10")], [sample(99.8 + 0.2 * i, avail_kb=0) for i in range(5)])
        r = self.summarize(d, "--min-avail-gb", "4")
        self.assertTrue(r["valid"], r["invalid_reasons"])
        self.assertTrue(any("not sampled" in w for w in r["warnings"]))


class Abab(unittest.TestCase):
    def test_order_and_summary(self):
        with tempfile.TemporaryDirectory() as out:
            for rnd in range(3):
                for arm, ms in (("A", 1000.0), ("B", 900.0)):
                    d = os.path.join(out, "%s-%d" % (arm, rnd))
                    os.makedirs(d)
                    with open(os.path.join(d, "receipt.json"), "w") as f:
                        json.dump(dict(arm=arm, round=rnd, valid=True, prefill_ms_warm=ms, prefill_ms_cold=ms + 50,
                                       tok_s_warm=8192 / ms * 1e3, sclk_mhz=2900, busy_pct=90, gpu_clocks_per_token=1,
                                       logits_sha="aa", gen_sha="g", metrics={}), f)
            r = subprocess.run([sys.executable, "-I", os.path.join(COMMON, "abab.py"), "summary", out, "--base", "A"],
                               capture_output=True, text=True, check=True).stdout
            self.assertIn("B vs A: prefill_ms_warm per-round -10.00 -10.00 -10.00 median -10.00 % over 3 rounds;", r)
            self.assertIn("bitwise-same logits", r)
        o = subprocess.run([sys.executable, "-I", os.path.join(COMMON, "abab.py"), "order", "--pairs", "2", "A", "B"],
                           capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(o, ["A", "0", "B", "0", "A", "1", "B", "1"])


class Served(unittest.TestCase):
    def test_serve_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            s = os.path.join(d, "server")
            make_run(s, [(100.0, "READY")], [sample(100 + 0.2 * i) for i in range(5)])
            with open(os.path.join(s, "meta.json")) as f:
                meta = json.load(f)
            meta["intentional_shutdown"] = True
            with open(os.path.join(s, "meta.json"), "w") as f:
                json.dump(meta, f)
            burst = dict(concurrency=8, client=dict(end_to_end_tok_s=120.5, per_request_tok_s_p50=15.0, ttft_p50_s=0.2,
                                                     failed=0, requests=8, output_tokens=16), steady=dict(aggregate_tok_s=130.0))
            with open(os.path.join(d, "json-0.json"), "w") as f:
                json.dump(dict(bursts=[burst]), f)
            with open(os.devnull, "w") as null:
                old, sys.stdout = sys.stdout, null
                try:
                    serve_receipts.main([d, "--tag", "t", "--arm", "ctl", "--round-base", "2", "--client-rc", "0", "--expected", "json-0", "--concurrency", "8"])
                finally:
                    sys.stdout = old
            r = load(os.path.join(d, "json-0"))
            self.assertTrue(r["valid"], r["invalid_reasons"])
            self.assertEqual((r["arm"], r["round"], r["metrics"]["e2e_C8"], r["metrics"]["agg_C8"]), ("ctl-json", 2, 120.5, 130.0))


class Prompts(unittest.TestCase):
    def test_formats(self):
        with tempfile.TemporaryDirectory() as d:
            seed = os.path.join(d, "seed.txt")
            tk = os.path.join(d, "tk.py")
            with open(seed, "w") as f:
                f.write("x\n")
            with open(tk, "w") as f:
                f.write("print('5,6,7')\n")
            ids_sha = set()
            for fmt in ("ids", "csv", "u32"):
                subprocess.run([sys.executable, "-I", os.path.join(COMMON, "prompt_ids.py"), "--out", d, "--sizes", "5",
                                "--seed", seed, "--tokenize-cmd", "%s %s" % (sys.executable.replace("\\", "/"), tk.replace("\\", "/")),
                                "--format", fmt], check=True, capture_output=True)
                with open(os.path.join(d, "prompts.json")) as f:
                    ids_sha.add(json.load(f)["prompts"][0]["ids_sha256"])
            with open(os.path.join(d, "p5.csv")) as f:
                self.assertEqual(f.read(), "5,6,7,5,6\n")
            self.assertEqual(os.path.getsize(os.path.join(d, "p5.u32")), 20)
            self.assertEqual(len(ids_sha), 1)


if __name__ == "__main__":
    unittest.main()
