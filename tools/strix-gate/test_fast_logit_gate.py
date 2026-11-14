"""Synthetic CPU tests of fast_logit_gate.py (numpy only; no engine)."""
import contextlib
import io
import os
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fast_logit_gate as g  # noqa: E402

ROWS, WIDTHS, TOKENS = 8, (5, 3), 40


def run(argv):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return g.main(argv)


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.rng = np.random.default_rng(7)
        self.tok = self.root / "p.u32"
        self.rng.integers(0, sum(WIDTHS), TOKENS, dtype=np.uint32).astype("<u4").tofile(self.tok)
        self.binary = self.root / "engine"
        self.binary.write_bytes(b"binary")
        self.logits = self.rng.normal(size=(ROWS, sum(WIDTHS))).astype("<f4")

    def tearDown(self):
        self.tmp.cleanup()

    def selection(self, path, mode, swap=False):
        words = []
        for layer in range(2):
            ids = np.tile(np.arange(4, dtype=np.uint32), (ROWS, 2, 1))
            if swap and layer == 1:
                ids[3, 1, 2] = 9
            ids[:, :, 3] = 77 + mode  # beyond the count: ignored
            words += [g.SEL_MAGIC, 1, mode, 16, 0, ROWS, 2, 4] + [3] * (ROWS * 2) + ids.ravel().tolist()
        np.array(words, "<u4").tofile(path)

    def arm(self, name, run_id, logits=None, mode=4, swap=False, seal=True, select=True):
        d = self.root / name
        d.mkdir()
        z = self.logits if logits is None else logits
        rc = []
        for rank, (lo, hi) in enumerate(((0, WIDTHS[0]), (WIDTHS[0], sum(WIDTHS)))):
            z[:, lo:hi].astype("<f4").tofile(d / f"q-p-tf-r{rank}.f32")
            if select:
                self.selection(d / f"q-p-sel-r{rank}.u32", mode, swap and rank == 1)
            if seal:
                rc.append(run(["seal", "--dir", str(d), "--rank", str(rank), "--arm", name, "--run-id", run_id, "--binary",
                               str(self.binary), "--rows", str(ROWS), "--prompt", f"p={self.tok}", "--env", "MODE=x",
                               "--attest", f"sel_mode={mode}"] + (["--select"] if select else [])))
        return d, rc

    def cmp(self, a, b, *extra):
        return run(["compare", str(a), str(b), "--ranks", "0", "1", "--tokens", f"p={self.tok}", *extra])

    def test_identical_repeat_passes(self):
        a, rc = self.arm("A", "r1")
        b, _ = self.arm("A2", "r2")
        self.assertEqual(rc, [0, 0])
        self.assertEqual(self.cmp(a, b, "--expect-identical"), 0)
        self.assertEqual(self.cmp(a, b), 0)

    def test_small_shift_passes_and_large_fails(self):
        a, _ = self.arm("A", "r1")
        near, _ = self.arm("N", "r2", self.logits + self.rng.normal(scale=1e-3, size=self.logits.shape).astype("<f4"), mode=3)
        far, _ = self.arm("F", "r3", -self.logits, mode=3)
        self.assertEqual(self.cmp(a, near), 0)
        self.assertEqual(self.cmp(a, near, "--expect-identical"), 1)
        self.assertEqual(self.cmp(a, far), 1)

    def test_selection_difference_is_counted(self):
        a, _ = self.arm("A", "r1")
        b, _ = self.arm("X", "r2", mode=3, swap=True)
        out = self.root / "r.json"
        self.assertEqual(self.cmp(a, b, "--json", str(out)), 0)
        self.assertEqual(self.cmp(a, b, "--expect-identical"), 1)
        import json
        sel = json.loads(out.read_text())["prompts"]["p"]["selection"]
        self.assertEqual((sel["layers"], sel["slot_diff"], sel["id_diff"], sel["first_layer"], sel["first_pos"]), (4, 1, 1, 3, 19))
        self.assertEqual(sel["modes"], [4, 3])

    def test_identity_defects_have_no_verdict(self):
        a, _ = self.arm("A", "r1")
        same, _ = self.arm("S", "r1")
        self.assertEqual(self.cmp(a, same), 2)  # one run id on both sides
        unsealed, _ = self.arm("U", "r2", seal=False)
        self.assertEqual(self.cmp(a, unsealed), 2)
        b, _ = self.arm("B", "r3")
        with open(b / "q-p-tf-r1.f32", "r+b") as f:  # changed after sealing
            f.write(b"\0\0\0\0")
        self.assertEqual(self.cmp(a, b), 2)

    def test_seal_rejects_missing_short_stale_reused_and_wrong_mode(self):
        d, _ = self.arm("A", "r1", seal=False)
        base = ["seal", "--dir", str(d), "--arm", "A", "--run-id", "r1", "--binary", str(self.binary), "--rows", str(ROWS),
                "--prompt", f"p={self.tok}", "--select"]
        self.assertEqual(run(base + ["--rank", "0", "--attest", "sel_mode=3"]), 2)  # dump says mode 4
        self.assertEqual(run(base + ["--rank", "0", "--not-before", str(os.path.getmtime(d / "q-p-tf-r0.f32") + 60)]), 2)
        self.assertEqual(run(base + ["--rank", "0"]), 0)
        self.assertEqual(run(base + ["--rank", "0"]), 2)  # sealed twice
        os.remove(d / "q-p-sel-r1.u32")
        self.assertEqual(run(base + ["--rank", "1"]), 2)
        self.selection(d / "q-p-sel-r1.u32", 4)
        with open(d / "q-p-tf-r1.f32", "ab") as f:
            f.write(b"\0\0")
        self.assertEqual(run(base + ["--rank", "1"]), 2)
        bad = self.logits.copy()
        bad[0, 0] = np.nan
        e, rc = self.arm("E", "r9", bad)
        self.assertEqual(rc, [2, 0])


if __name__ == "__main__":
    unittest.main()
