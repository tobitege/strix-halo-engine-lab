"""Host CLI regressions for the environment census (no compiler/device needed)."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "env_switch_census.py"


class CensusCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="census-test-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "src").mkdir()
        self.source = self.root / "src" / "fixture.cpp"
        self.baseline = self.root / "baseline.txt"
        self.baseline.write_bytes(b"# preserved baseline\r\nSTRIX_BASELINE\r\n")

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--root", str(self.root),
             "--baseline", str(self.baseline), *args],
            capture_output=True, text=True, timeout=30, check=False)

    def test_adjacent_raw_literals_fail_check(self):
        for expression in ('"STRIX_BASELINE" "_NEW"',
                           '"STRIX_BASELINE" /* suffix */\n "_NEW"',
                           '"STRIX_" "BASELINE" "" "_NEW"'):
            with self.subTest(expression=expression):
                self.source.write_text(f'std::getenv({expression});\n', encoding="utf-8")
                result = self.run_cli("--check")
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("STRIX_BASELINE_NEW is read with a raw getenv", result.stderr)

    def test_escaped_raw_literal_fails_closed(self):
        self.source.write_text(r'getenv("STRIX_BASELINE" "\x5fNEW");', encoding="utf-8")
        result = self.run_cli("--check")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("unsupported escape", result.stderr)

    def test_seed_refuses_existing_baseline(self):
        self.source.write_text('getenv("STRIX_NEW_RAW");\n', encoding="utf-8")
        before = self.baseline.read_bytes()
        result = self.run_cli("--seed")
        self.assertEqual(self.baseline.read_bytes(), before)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("already exists", result.stderr)
        self.assertEqual(self.run_cli("--check").returncode, 1)

    def test_first_seed_and_intersection_update(self):
        self.source.write_text('getenv("STRIX_BASELINE");\n', encoding="utf-8")
        self.baseline.unlink()
        self.assertEqual(self.run_cli("--seed").returncode, 0)
        self.source.write_text('getenv("STRIX_NEW_RAW");\n', encoding="utf-8")
        self.assertEqual(self.run_cli("--update").returncode, 0)
        self.assertNotIn("STRIX_BASELINE", self.baseline.read_text(encoding="utf-8"))
        self.assertNotIn("STRIX_NEW_RAW", self.baseline.read_text(encoding="utf-8"))
        self.assertEqual(self.run_cli("--check").returncode, 1)

    def test_grandfather_only_explicit_raw_names(self):
        self.source.write_text('getenv("STRIX_BASELINE"); getenv("STRIX_NEW_RAW"); '
                               'getenv("STRIX_OTHER_RAW");\n', encoding="utf-8")
        result = self.run_cli("--grandfather", "STRIX_NEW_RAW")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        names = self.baseline.read_text(encoding="utf-8").splitlines()
        self.assertIn("STRIX_BASELINE", names)
        self.assertIn("STRIX_NEW_RAW", names)
        self.assertNotIn("STRIX_OTHER_RAW", names)
        self.assertEqual(self.run_cli("--check").returncode, 1)
        self.assertEqual(self.run_cli("--grandfather", "STRIX_OTHER_RAW").returncode, 0)
        self.assertEqual(self.run_cli("--check").returncode, 0)

    def test_grandfather_rejects_non_raw_names_without_writing(self):
        self.source.write_text('getenv("STRIX_NEW_RAW"); '
                               'envcfg::flag("STRIX_GOVERNED", false);\n', encoding="utf-8")
        before = self.baseline.read_bytes()
        for name in ("STRIX_MISSING", "STRIX_GOVERNED"):
            with self.subTest(name=name):
                result = self.run_cli("--grandfather", "STRIX_NEW_RAW", name)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("not currently read raw", result.stderr)
                self.assertEqual(self.baseline.read_bytes(), before)
        self.assertEqual(self.run_cli("--check", "--grandfather", "STRIX_NEW_RAW").returncode, 2)
        self.assertEqual(self.baseline.read_bytes(), before)

    def test_typed_governed_reads_in_report_and_tsv(self):
        cases = [("std::uint32_t", "STRIX_PROBE_DEPTH", "4U"),
                 ("std::uint32_t", "STRIX_PROBE_GPU_LINKS", "8U"),
                 ("unsigned", "STRIX_PROBE_PAUSE_US", "0U"),
                 ("double", "STRIX_PROBE_OUTLIER_US", "10000.0")]
        self.source.write_text("\n".join(
            f'strix::env_number<{kind}>("{name}", {default});'
            for kind, name, default in cases), encoding="utf-8")
        tsv = self.root / "inventory.tsv"
        result = self.run_cli("--check", "--report", "--tsv", str(tsv))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("switches read: 4\n", result.stdout)
        self.assertIn("governed only: 4\n", result.stdout)
        rows = tsv.read_text(encoding="utf-8").splitlines()
        self.assertEqual(set(rows[1:]), {f"{name}\tgoverned\t1\tno" for _, name, _ in cases})


if __name__ == "__main__":
    unittest.main()
