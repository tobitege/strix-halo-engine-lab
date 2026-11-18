import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/bench/wfmt"))
spec = importlib.util.spec_from_file_location("capture", ROOT / "scripts/bench/wfmt/capture.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class ArchivedSource(unittest.TestCase):
    def test_archive_override(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests/wfmt") as directory:
            archive = pathlib.Path(directory)
            with mock.patch.object(capture.subprocess, "check_output", side_effect=AssertionError("git invoked")):
                for size in (40, 64):
                    self.assertEqual(capture.source_identity(archive, "A" * size), "a" * size)
                for invalid in ("HEAD", "a" * 39, "g" * 40, "a" * 65, ""):
                    with self.assertRaises(ValueError):
                        capture.source_identity(archive, invalid)
                with self.assertRaises(ValueError):
                    capture.source_identity(None, "a" * 40)
                # Exercise argument parsing through main, stopping before adapter execution.
                args = ["capture.py", "--source", str(archive), "--source-ref", "a" * 40,
                        "--exe", str(archive / "missing"), "--matrix", str(archive / "matrix"),
                        "--out", str(archive / "out"), "--compiler", "test", "--architecture", "host"]
                with (mock.patch.object(sys, "argv", args), mock.patch("builtins.print"),
                      mock.patch.object(capture.platform, "platform", return_value="test")):
                    self.assertEqual(capture.main(), 1)

    def test_git_source(self):
        with mock.patch.object(capture.subprocess, "check_output", return_value="b" * 40 + "\n") as git:
            self.assertEqual(capture.source_identity(ROOT, None), "b" * 40)
            git.assert_called_once()


if __name__ == "__main__":
    unittest.main()
