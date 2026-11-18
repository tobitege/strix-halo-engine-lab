"""Host regressions for ID coverage, device selection and binding receipts."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/bench/common"))
import receipt
import serve_receipts


class SummaryCoverage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.records = [dict(pas=p, cls="edit", c=1, i=0, ids=[1, 2]) for p in ("cold", "warm")]

    def arm(self, name, records, bench=True):
        d = self.root / name
        d.mkdir()
        if bench:
            (d / "bench.txt").write_text("", encoding="utf-8")
        if records is not None:
            (d / "bench.txt.ids").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    def summary(self):
        return subprocess.run([sys.executable, str(ROOT / "scripts/bench/cuda/qwen_serve_summary.py"), str(self.root)],
                              capture_output=True, text=True, check=False)

    def test_complete_matching_repeated_arms(self):
        for name in ("0-A", "1-B", "2-A", "3-B"):
            self.arm(name, self.records)
        result = self.summary()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("2 requests compared across 4 arm runs, 0 differ or missing", result.stdout)

    def test_missing_file_request_and_empty_evidence(self):
        for records in (None, [], self.records[:1]):
            with self.subTest(records=records):
                self.arm("0-A", self.records)
                self.arm("1-B", records)
                result = self.summary()
                shutil.rmtree(self.root / "0-A")
                shutil.rmtree(self.root / "1-B")
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("missing", result.stdout)

    def test_no_evidence_in_any_run(self):
        for name in ("0-A", "1-B"):
            self.arm(name, None)
        self.assertNotEqual(self.summary().returncode, 0)

    def test_no_runs(self):
        self.assertNotEqual(self.summary().returncode, 0)

    def test_bad_sequences_and_differing_ids(self):
        self.arm("0-A", self.records)
        for ids in (None, [], "1,2", [True, 2], [1, 3]):
            with self.subTest(ids=ids):
                self.arm("1-B", [dict(r, ids=ids) for r in self.records])
                result = self.summary()
                shutil.rmtree(self.root / "1-B")
                self.assertNotEqual(result.returncode, 0)

    def test_duplicate_records(self):
        self.arm("0-A", self.records)
        self.arm("1-B", self.records + self.records[:1])
        result = self.summary()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate", result.stdout)

    def test_missing_bench_does_not_hide_id_file(self):
        self.arm("0-A", self.records)
        self.arm("1-B", [dict(r, ids=[3]) for r in self.records], bench=False)
        self.assertNotEqual(self.summary().returncode, 0)


class DeviceBinding(unittest.TestCase):
    def test_cleanup_preserves_binding_and_cleans_tuning(self):
        bash = shutil.which("bash")
        if not bash:
            self.skipTest("bash unavailable")
        script = '''
source scripts/bench/common/runlib.sh
export STRIX_DEVICES='assets/device declaration.toml' STRIX_DEVICE='test-gpu'
export STRIX_TEST_TUNING=1 STRIX_TEST_KEEP=2 KIT_KEEP_ENV=STRIX_TEST_KEEP
kit_clean_env
[[ ${STRIX_DEVICES-} == 'assets/device declaration.toml' ]] || exit 10
[[ ${STRIX_DEVICE-} == 'test-gpu' ]] || exit 11
[[ ! -v STRIX_TEST_TUNING && ${STRIX_TEST_KEEP-} == 2 ]] || exit 12
'''
        result = subprocess.run([bash, "--noprofile", "--norc", "-c", script], cwd=ROOT,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_effective_binding_reaches_both_receipts(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            server = root / "server"
            server.mkdir()
            binding = 'device: id=test-gpu vendor=nvidia uuid=GPU-test pci=0000:01:00.0 name="Test GPU" arch=sm_89 ordinal=0 attach=pcie roles=decode declared=assets/devices.toml'
            (server / "stderr").write_text("startup\n" + binding + "\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                receipt.main(["meta", str(server), "--rc", "0", "--wall-ms", "1", "--intentional-shutdown"])
                receipt.main(["summarize", str(server), "--tag", "test"])
                serve_receipts.main([str(root), "--tag", "test", "--arm", "A", "--client", "1", "--client-rc", "0"])
            for path in (server / "meta.json", server / "receipt.json", root / "client/receipt.json"):
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")).get("device_bindings"), [binding])

    def test_missing_binding_stays_unknown(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            receipt.main(["meta", tmp, "--rc", "0", "--wall-ms", "1"])
            self.assertEqual(json.loads((Path(tmp) / "meta.json").read_text()).get("device_bindings"), [])


if __name__ == "__main__":
    unittest.main()
