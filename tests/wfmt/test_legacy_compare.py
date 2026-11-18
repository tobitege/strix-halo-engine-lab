"""Offline source-provenance regressions; temporary repositories stay in the checkout."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/bench/wfmt"))
import legacy_compare


class LegacySourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".wfmt-source-", dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "source"
        self.source.mkdir()
        self.git("init", "-q")
        self.file = self.source / "weight.cpp"
        self.file.write_text("baseline\n")
        self.git("add", "weight.cpp")
        self.git("commit", "-qm", "baseline")
        self.baseline = self.git("rev-parse", "HEAD")
        self.file.write_text("candidate\n")
        self.git("commit", "-qam", "candidate")
        self.candidate = self.git("rev-parse", "HEAD")

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.source), "-c", "user.name=Fixture",
             "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgSign=false",
             "-c", "core.autocrlf=false", *args],
            check=True, capture_output=True, text=True).stdout.strip()

    def test_verified_commit_and_tree(self):
        identity = legacy_compare.verify_source(self.source, "HEAD")
        self.assertEqual(identity["commit"], self.candidate)
        self.assertEqual(identity["tree"], self.git("rev-parse", "HEAD^{tree}"))
        self.git("checkout", "--detach", self.baseline)
        self.assertEqual(legacy_compare.verify_source(self.source, self.baseline)["commit"], self.baseline)

    def test_dirty_inputs_refused(self):
        for state in ("unstaged", "staged", "untracked"):
            with self.subTest(state=state):
                self.file.write_text("modified\n")
                if state == "staged":
                    self.git("add", "weight.cpp")
                if state == "untracked":
                    self.git("checkout", "HEAD", "--", "weight.cpp")
                    (self.source / "new.hpp").write_text("new input\n")
                with self.assertRaisesRegex(ValueError, "uncommitted"):
                    legacy_compare.verify_source(self.source, self.candidate)
                self.git("checkout", "HEAD", "--", "weight.cpp")

    def refuse_before_configure(self, baseline, candidate):
        output = self.directory / "receipts"
        args = ["legacy_compare", "--baseline-source", str(self.source),
                "--candidate-source", str(self.source), "--baseline-revision", baseline,
                "--candidate-revision", candidate, "--build-root", str(self.directory / "build"),
                "--output", str(output)]
        with mock.patch.object(sys, "argv", args), \
                mock.patch.object(legacy_compare, "logged", side_effect=ValueError("unexpected build")) as build, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(legacy_compare.main(), 1)
        build.assert_not_called()
        receipt = json.loads((output / "summary.json").read_text())
        self.assertFalse(receipt["pass"])
        self.assertIn("does not match", receipt["error"])

    def test_candidate_checkout_cannot_impersonate_baseline(self):
        self.refuse_before_configure(self.baseline, self.candidate)

    def test_both_checkouts_verified_before_first_build(self):
        self.refuse_before_configure(self.candidate, self.baseline)


if __name__ == "__main__":
    unittest.main()
