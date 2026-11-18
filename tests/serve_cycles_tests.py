#!/usr/bin/env python3
"""Host tests for scripts/bench/qwen/serve_cycles.py: burst selection and the fail-closed identity verdict."""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location(
    "serve_cycles", os.path.join(HERE, "..", "scripts", "bench", "qwen", "serve_cycles.py"))
sc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sc)


def burst(concurrency, ids, failed=0, slots=8, e2e=50.0):
    return dict(concurrency=concurrency, max_tokens=256, slots=slots,
                steady=dict(aggregate_tok_s=e2e + 10.0),
                client=dict(requests=len(ids), failed=failed, end_to_end_tok_s=e2e, ttft_p50_s=1.0, ids_sha256=ids))


def row(name, ids, requests=None, failed=0):
    return dict(name=name, ids=ids, requests=len(ids) if requests is None and ids is not None else requests,
                failed=failed)


def plan_line(lanes, rows, emitted, target):
    members = ",".join("%d:0:8:32:%d/%d/%d/%d/%d/%d/%d/%d" % (k, rows, rows - 1, rows - 1, rows - 1, rows - 1, rows,
                                                             emitted - 1, emitted) for k in range(lanes))
    return ("joint plan wall_ms=%d sessions=%d probe=0 expected=1.0 est_ms=1.0 members=%s draft_ms=5.0 target_ms=%.1f"
            % (100, lanes, members, target))


class SelectBurst(unittest.TestCase):
    def test_matching_concurrency_not_first(self):
        # Audit case: C2 then C7 in one result; a c7 window must take the C7 burst, not burst 0.
        res = dict(bursts=[burst(2, ["a", "b"], e2e=20.0), burst(7, ["h"] * 7, e2e=90.0)])
        self.assertEqual(sc.select_burst(res, 7)["concurrency"], 7)
        self.assertEqual(sc.select_burst(res, 2)["concurrency"], 2)

    def test_explicit_index_validated(self):
        res = dict(bursts=[burst(2, ["a", "b"]), burst(7, ["h"] * 7)])
        self.assertEqual(sc.select_burst(res, 7, 1)["concurrency"], 7)
        with self.assertRaises(ValueError):
            sc.select_burst(res, 7, 0)
        with self.assertRaises(ValueError):
            sc.select_burst(res, 7, 5)

    def test_ambiguous_or_absent_rejected(self):
        res = dict(bursts=[burst(7, ["h"] * 7), burst(7, ["h"] * 7)])
        with self.assertRaises(ValueError):
            sc.select_burst(res, 7)
        self.assertEqual(sc.select_burst(res, 7, 1)["concurrency"], 7)
        with self.assertRaises(ValueError):
            sc.select_burst(dict(bursts=[burst(2, ["a", "b"])]), 7)

    def test_slot_limited_live_lanes(self):
        # Eight requests on seven slots decode seven live lanes.
        self.assertEqual(sc.select_burst(dict(bursts=[burst(8, ["h"] * 8, slots=7)]), 7)["concurrency"], 8)

    def test_unknown_slots_observed_lanes(self):
        b = burst(8, ["h"] * 8, slots=None)
        b["steady"].update(active_lanes=7, cycles=100)
        self.assertIs(sc.select_burst(b, 7), b)
        self.assertIs(sc.select_burst(b, 7, 0), b)
        with self.assertRaises(ValueError):
            sc.select_burst(b, 8, 0)
        with self.assertRaises(ValueError):
            sc.select_burst(dict(bursts=[b, b]), 7)
        self.assertIs(sc.select_burst(dict(bursts=[b, b]), 7, 1), b)

    def test_unknown_slots_require_valid_observation(self):
        invalid = [None, {}, dict(active_lanes=7), dict(cycles=100)]
        invalid += [dict(active_lanes=n, cycles=100) for n in (None, 0, -1, 9, True, 7.0, "7")]
        invalid += [dict(active_lanes=7, cycles=n) for n in (None, 0, -1, True, 100.0, "100")]
        for steady in invalid:
            with self.subTest(steady=steady):
                b = burst(8, ["h"] * 8, slots=None)
                b["steady"] = steady
                self.assertIsNone(sc.live_lanes(b))
                for index in (None, 0):
                    with self.assertRaisesRegex(ValueError, "unavailable.*evidence"):
                        sc.select_burst(b, 8, index)

    def test_known_slots_take_precedence(self):
        b = burst(8, ["h"] * 8, slots=8)
        b["steady"].update(active_lanes=7, cycles=100)
        self.assertEqual(sc.live_lanes(b), 8)


class IdentityVerdict(unittest.TestCase):
    def test_identical_and_differ(self):
        a = [row("c7-A1", ["x"] * 7), row("c7-A2", ["x"] * 7)]
        self.assertEqual(sc.identity_verdict(a, [row("c7-B1", ["x"] * 7)]), "IDENTICAL")
        self.assertEqual(sc.identity_verdict(a, [row("c7-B1", ["x"] * 6 + ["y"])]), "DIFFER")

    def test_null_hashes_are_not_identity(self):
        v = sc.identity_verdict([row("c2-A1", [None, None])], [row("c2-B1", [None, None])])
        self.assertTrue(v.startswith("UNAVAILABLE"), v)

    def test_missing_hashes_or_result(self):
        self.assertTrue(sc.identity_verdict([row("c2-A1", None, requests=2)], [row("c2-B1", ["a", "b"])])
                        .startswith("UNAVAILABLE"))
        self.assertTrue(sc.identity_verdict([row("c2-A1", [], requests=2)], [row("c2-B1", [], requests=2)])
                        .startswith("UNAVAILABLE"))
        self.assertTrue(sc.identity_verdict([row("c2-A1", [], requests=0)], [row("c2-B1", [], requests=0)])
                        .startswith("UNAVAILABLE"))

    def test_failures_invalidate(self):
        v = sc.identity_verdict([row("c2-A1", ["a", "b"], failed=1)], [row("c2-B1", ["a", "b"])])
        self.assertTrue(v.startswith("UNAVAILABLE"), v)

    def test_absent_arm(self):
        self.assertTrue(sc.identity_verdict([row("c2-A1", ["a", "b"])], []).startswith("UNAVAILABLE"))
        self.assertTrue(sc.identity_verdict([], [row("c2-B1", ["a", "b"])]).startswith("UNAVAILABLE"))

    def test_request_counts_differ(self):
        v = sc.identity_verdict([row("c2-A1", ["a", "b"])], [row("c2-B1", ["a", "b", "c"])])
        self.assertTrue(v.startswith("UNAVAILABLE"), v)


class MainEndToEnd(unittest.TestCase):
    def run_main(self, files, bursts, pairs="A,B"):
        # The spec separator is ':', so the files are named relative to the temporary directory (no drive letter).
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                return self.run_in(files, bursts, pairs)
            finally:
                os.chdir(cwd)

    def run_in(self, files, bursts, pairs):
        log = "srv.err"
        with open(log, "w") as f:
            f.write("\n".join([plan_line(7, 4, 3, 180.0)] * 4 + [plan_line(2, 2, 2, 50.0)] * 4) + "\n")
        specs = []
        for name, first, last, key, extra in bursts:
            path = key + ".json"
            if key in files:
                with open(path, "w") as f:
                    json.dump(files[key], f)
            specs += ["--burst", "%s:%d:%d:%s%s" % (name, first, last, path, extra)]
        out = io.StringIO()
        argv = ["serve_cycles.py", "--log", log] + specs + (["--pairs", pairs] if pairs else [])
        old = sys.argv
        sys.argv = argv
        try:
            with contextlib.redirect_stdout(out):
                rc = sc.main()
        finally:
            sys.argv = old
        return rc, out.getvalue()

    def test_pair_identical_takes_matching_burst(self):
        files = dict(a=dict(bursts=[burst(2, ["p", "q"], e2e=20.0), burst(7, ["h"] * 7, e2e=90.0)]),
                     b=dict(bursts=[burst(2, ["p", "q"], e2e=21.0), burst(7, ["h"] * 7, e2e=99.0)]))
        rc, out = self.run_main(files, [("c7-A1", 1, 4, "a", ":0"), ("c7-B1", 1, 4, "b", ":0")])
        self.assertEqual(rc, 0, out)
        self.assertIn("ids IDENTICAL", out)
        self.assertIn("wall 90.0 -> 99.0", out)

    def test_unknown_slots_pair_automatic_and_explicit(self):
        b = burst(8, ["h"] * 8, slots=None)
        b["steady"].update(active_lanes=7, cycles=100)
        for suffix in ("", "#0"):
            with self.subTest(suffix=suffix):
                rc, out = self.run_main(dict(a=b, b=b),
                                        [("c7-A1", 1, 4, "a", suffix), ("c7-B1", 1, 4, "b", suffix)])
                self.assertEqual(rc, 0, out)
                self.assertIn("ids IDENTICAL", out)

    def test_only_one_arm_fails_closed(self):
        files = dict(a=dict(bursts=[burst(7, ["h"] * 7)]))
        rc, out = self.run_main(files, [("c7-A1", 1, 4, "a", ":0")])
        self.assertEqual(rc, 1, out)
        self.assertNotIn("IDENTICAL", out)

    def test_null_hash_pair_fails_closed(self):
        files = dict(a=dict(bursts=[burst(2, [None, None])]), b=dict(bursts=[burst(2, [None, None])]))
        rc, out = self.run_main(files, [("c2-A1", 5, 8, "a", ""), ("c2-B1", 5, 8, "b", "")])
        self.assertEqual(rc, 1, out)
        self.assertIn("UNAVAILABLE", out)

    def test_missing_result_file_fails_closed(self):
        files = dict(a=dict(bursts=[burst(2, ["p", "q"])]))
        rc, out = self.run_main(files, [("c2-A1", 5, 8, "a", ""), ("c2-B1", 5, 8, "missing", "")])
        self.assertEqual(rc, 1, out)
        self.assertIn("UNAVAILABLE", out)

    def test_mismatched_index_stops(self):
        files = dict(a=dict(bursts=[burst(2, ["p", "q"]), burst(7, ["h"] * 7)]))
        with self.assertRaises(SystemExit):
            self.run_main(files, [("c7-A1", 1, 4, "a", "#0")])


if __name__ == "__main__":
    unittest.main()
