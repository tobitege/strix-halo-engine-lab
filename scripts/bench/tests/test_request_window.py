#!/usr/bin/env python3
"""Host regressions for conservative request envelopes and shared receipt validity gates."""
import os
import contextlib
import io
import re
import shlex
import sys
import tempfile
import unittest
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_runner import abab, load, make_run, receipt, sample  # noqa: E402

LENGTH_MATRIX = os.path.join(HERE, "..", "qwen", "length-matrix.sh")
R_MS = 4000.0  # request: start 16.0, prefill block line 20.0, result object stamped 20.5
BLOCK_T, RESULT_T = 20.0, 20.5


def full_mode_args():
    """The receipt arguments length-matrix.sh binds for full mode (the real binding, not a copy)."""
    with open(LENGTH_MATRIX) as stream:
        text = stream.read()
    base = re.search(r"KIT_RECEIPT_ARGS='([^']*)'", text).group(1)
    extra = re.search(r"if \[\[ \$m == full \]\]; then\s+KIT_RECEIPT_ARGS\+='([^']*)'", text).group(1)
    return shlex.split(base + extra)


class RequestWindow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=HERE)

    def tearDown(self):
        self.tmp.cleanup()

    def run_arm(self, evict_at=(), extra=None, stamp_result=True, block=True,
                block_t=BLOCK_T, result_t=RESULT_T, klog="", stderr_text="", rc=0):
        d = os.path.join(self.tmp.name, "arm")
        ev, rows = 0, []
        for i in range(0, 101):  # samples every 0.25 s over [0, 25]
            t = i * 0.25
            ev += sum(100 for e in evict_at if t - 0.25 < e <= t)
            rows.append(sample(t, ev=ev))
        lines = [(1.0, "loading weights"), (16.1, "qpf2k HC out16 active (rows 8192)")]
        if block:
            lines.append((block_t, "qpf2k sched block rows 8192 detached_shared 0 late 0"))
        if stderr_text:
            lines.append((22.0, stderr_text))
        lines.append((22.0, "strix power: {}"))
        out = [(0.5, "banner"), (result_t, "{"), (result_t, '  "generation_request_elapsed_ms":%.2f,' % R_MS),
               (result_t, '  "generated_tokens": [279, 280]'), (result_t, "}")]
        make_run(d, lines, rows, stdout="".join(l + "\n" for _, l in out), klog=klog, rc=rc)
        if stamp_result:
            with open(os.path.join(d, "stdout.ts"), "w") as f:
                f.write("".join("%.3f %s\n" % (t, l) for t, l in out))
        with open(os.devnull, "w") as null:
            old, sys.stdout = sys.stdout, null
            try:
                receipt.main(["summarize", d, "--tag", "arm", "--tokens", "8192"] + (full_mode_args() if extra is None else extra))
            finally:
                sys.stdout = old
        return load(d)

    def test_full_mode_binding_uses_request_windows(self):
        args = full_mode_args()
        self.assertIn("--request-json", args)
        self.assertEqual(args[args.index("--scope") + 1], "request")

    def test_windows_are_conservative_sampled_envelopes(self):
        rec = self.run_arm()
        w = rec["windows"]
        self.assertEqual((w["request"]["t0"], w["request"]["t1"]), (0.0, RESULT_T))
        self.assertEqual((w["prefill"]["t0"], w["prefill"]["t1"]), (0.0, BLOCK_T))
        self.assertEqual((w["decode"]["t0"], w["decode"]["t1"]), (0.0, RESULT_T))
        self.assertTrue(rec["valid"] and rec["valid_prefill"] and rec["valid_decode"] and rec["valid_request"])
        self.assertEqual(rec["prefill_ms_bounds"], [0.0, R_MS])
        self.assertEqual(rec["decode_ms_bound"], R_MS)
        self.assertIn("upper bound", rec["prefill_source"])
        self.assertEqual(rec["request_ms"], R_MS)
        self.assertIn("estimated", rec["telemetry_source"])
        self.assertEqual(rec["window_s"], BLOCK_T)
        self.assertEqual(rec["joules"], 50 * BLOCK_T)

    def test_decode_eviction_counts_in_request_not_prefill(self):
        # The audited regression: an eviction after the prefill block line but before the request stop.
        rec = self.run_arm(evict_at=(20.3,))
        self.assertGreater(rec["windows"]["decode"]["evicted_ms"], 0)
        self.assertGreater(rec["windows"]["request"]["evicted_ms"], 0)
        self.assertEqual(rec["windows"]["prefill"]["evicted_ms"], 0)
        self.assertFalse(rec["valid"])
        self.assertTrue(rec["valid_prefill"])
        self.assertFalse(rec["valid_decode"] or rec["valid_request"])
        self.assertIn("kfd evicted_ms +100 in the request window", rec["invalid_reasons"])
        pf = self.run_arm(evict_at=(20.3,), extra=full_mode_args()[:-2] + ["--scope", "prefill"])
        self.assertTrue(pf["valid"] and pf["valid_prefill"])
        self.assertFalse(pf["valid_request"])

    def test_prefill_eviction_also_counts_in_decode_envelope(self):
        rec = self.run_arm(evict_at=(18.0,))
        self.assertFalse(rec["valid"] or rec["valid_prefill"] or rec["valid_request"])
        self.assertFalse(rec["valid_decode"])

    def test_abab_excludes_decode_pressured_request_duration(self):
        self.run_arm(evict_at=(20.3,))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            abab.cmd_summary(SimpleNamespace(out=self.tmp.name, base="missing", metric="request_latency_ms"))
        self.assertIn("arm | 0/1 |", output.getvalue())

    def test_load_is_conservatively_included_but_teardown_is_outside(self):
        rec = self.run_arm(evict_at=(5.0, 23.0))
        self.assertFalse(rec["valid_prefill"] or rec["valid_decode"] or rec["valid_request"])
        self.assertEqual(rec["windows"]["request"]["evicted_ms"], 100)
        self.assertEqual(rec["evicted_ms_run"], 200)
        self.assertTrue(any("outside the request window" in w for w in rec["warnings"]))

    def test_teardown_only_is_valid_unless_whole_run_is_required(self):
        rec = self.run_arm(evict_at=(23.0,))
        self.assertTrue(all(rec[key] for key in ("valid", "valid_prefill", "valid_decode", "valid_request")))
        rec = self.run_arm(evict_at=(23.0,), extra=full_mode_args() + ["--evict-whole-run"])
        self.assertFalse(any(rec[key] for key in ("valid", "valid_prefill", "valid_decode", "valid_request")))

    def test_delayed_reader_cannot_hide_early_request_eviction(self):
        # Actual request [16, 20], witness emitted at 19.5, read at 21; result read at 21.5.
        for eviction in (16.5, 19.75):
            rec = self.run_arm(evict_at=(eviction,), block_t=21.0, result_t=21.5)
            for key in ("valid", "valid_prefill", "valid_decode", "valid_request"):
                self.assertFalse(rec[key], (eviction, key))
            self.assertEqual(rec["windows"]["request"]["evicted_ms"], 100)

    def test_shared_rejections_invalidate_every_scope(self):
        cases = [(["--strict-klog"], dict(klog="amdgpu reset\n"), "klog"),
                 (["--require", "missing witness"], {}, "missing /"),
                 (["--reject", "bad witness"], dict(stderr_text="bad witness"), "stderr /"),
                 ([], dict(stderr_text="operation timed out"), "stderr /"),
                 ([], dict(rc=1), "rc=1")]
        for options, kw, reason in cases:
            for scope in ("request", "prefill", "decode"):
                rec = self.run_arm(extra=full_mode_args()[:-2] + ["--scope", scope] + options, **kw)
                for key in ("valid", "valid_prefill", "valid_decode", "valid_request"):
                    self.assertFalse(rec[key], (options, scope, key))
                self.assertTrue(any(reason in r for r in rec["invalid_reasons"]))

    def test_non_strict_klog_remains_a_warning(self):
        rec = self.run_arm(klog="amdgpu reset\n")
        self.assertTrue(all(rec[key] for key in ("valid", "valid_prefill", "valid_decode", "valid_request")))
        self.assertTrue(any("klog" in w for w in rec["warnings"]))

    def test_unstamped_result_or_missing_witness_fails_closed(self):
        for kw in (dict(stamp_result=False), dict(block=False)):
            rec = self.run_arm(**kw)
            self.assertFalse(rec["valid"] or rec["valid_decode"] or rec["valid_request"], kw)

    def test_prefill_json_alias(self):
        args = ["--need-prefill", "--prefill-json", "generation_request_elapsed_ms", "--anchor-regex",
                "^qpf2k.sched.block.rows"]
        rec = self.run_arm(evict_at=(20.3,), extra=args)
        self.assertEqual(rec["scope"], "request")
        self.assertFalse(rec["valid"])


class PairVerdict(unittest.TestCase):
    """The pair arm verdict line comes after the fold: a clean rank 0 with an evicting peer is INVALID."""

    def fold(self, r0, r1):
        import json
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, rec in enumerate((r0, r1)):
                paths.append(os.path.join(d, "r%d.json" % i))
                if rec is not None:
                    with open(paths[-1], "w") as stream:
                        json.dump(rec, stream)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = receipt.main(["pair"] + paths)
            return rc, out.getvalue().strip()

    def test_peer_eviction_invalidates_the_arm_line(self):
        r0 = dict(tag="s128K-pair-exact-0", valid=True, invalid_reasons=[])
        r1 = dict(tag="s128K-pair-exact-0-r1", valid=False, invalid_reasons=["kfd evicted_ms +1435 in the timed window"])
        rc, line = self.fold(r0, r1)
        self.assertTrue(rc)
        self.assertEqual(line, "== s128K-pair-exact-0 PAIR INVALID[r0: - | r1: kfd evicted_ms +1435 in the timed window]")

    def test_both_clean_is_valid_and_missing_peer_is_not(self):
        ok = dict(tag="s32K-pair-fast-0", valid=True, invalid_reasons=[])
        self.assertEqual(self.fold(ok, dict(ok, tag="s32K-pair-fast-0-r1")), (False, "== s32K-pair-fast-0 PAIR VALID"))
        rc, line = self.fold(dict(ok), None)
        self.assertTrue(rc)
        self.assertTrue(line.startswith("== s32K-pair-fast-0 PAIR INVALID["), line)


if __name__ == "__main__":
    unittest.main()
