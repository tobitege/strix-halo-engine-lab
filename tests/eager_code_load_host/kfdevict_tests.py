import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "bench" / "kfdevict"


class KfdReports(unittest.TestCase):
    def run_script(self, script, *args):
        return subprocess.check_output(
            [sys.executable, str(SCRIPTS / script), *map(str, args)], text=True
        )

    def test_evprof_overlapping_samples_and_pids(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            log = pathlib.Path(tmp) / "ev.log"
            log.write_text(
                "1.00 00:00:01 EV 1 engine +3 (=3) | 2s: pgsteal_kswapd=100 pswpout=7 | MA=20 C=30\n"
                "1.25 00:00:01 EV 1 engine +4 (=7) | 2s: pgsteal_kswapd=100 pswpout=7 | MA=19 C=30\n"
                "1.25 00:00:01 EV 2 engine +5 (=5) | 2s: pgsteal_kswapd=100 pswpout=7 | MA=19 C=30\n"
                "6.00 00:00:06 EV 2 engine +2 (=7) | 2s: pgsteal_kswapd=40 | MA=18 C=29\n"
            )
            out = self.run_script("evprof.py", log)
            self.assertIn("overlapping observations; maxima, not counter totals", out)
            rows = [line.split() for line in out.splitlines() if line.strip() and line.split()[0].isdigit()]
            self.assertEqual(rows[0][1:5], ["3", "12", "20", "30"])
            self.assertEqual(rows[0][5:7], ["100", "7"])
            self.assertEqual(rows[1][5:7], ["40", "0"])

    def test_postload_unknown_without_marker(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = pathlib.Path(tmp)
            (root / "r0.log").write_text("rank 0: load 40.0 s\n")
            (root / "mon.1").write_text("100.0 ALLOC USERPTR size=100\n145.0 ALLOC USERPTR size=200\n")
            out = self.run_script("postload.py", root)
            self.assertIn("load-end boundary unknown", out)
            self.assertNotIn("after load", out)

    def test_postload_explicit_boottime_no_grace_period(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = pathlib.Path(tmp)
            (root / "r0.log").write_text("rank 0: load 40.0 s\n")
            (root / "mon.1").write_text(
                "100.0 ALLOC USERPTR size=100\n145.0 ALLOC USERPTR size=200\n"
                "150.0 ALLOC USERPTR size=300\n150.1 ALLOC USERPTR size=1000000\n"
                "150.2 ALLOC USERPTR size=2000000\n"
            )
            (root / "load-end.boottime").write_text("150.0\n")
            out = self.run_script("postload.py", root)
            self.assertIn("after load 3 userptr allocs 3.0 MB", out)
            self.assertIn("2.0MB@+0.200s", out)
            self.assertNotIn("prefill", out)

    def test_postload_invalid_marker_is_unknown(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = pathlib.Path(tmp)
            (root / "r0.log").write_text("rank 0: load 40.0 s\n")
            (root / "mon.1").write_text("100.0 ALLOC USERPTR size=100\n")
            for marker in ("nan", "inf", "-1", "bad", ""):
                (root / "load-end.boottime").write_text(marker)
                self.assertIn("load-end boundary unknown", self.run_script("postload.py", root))


if __name__ == "__main__":
    unittest.main()
