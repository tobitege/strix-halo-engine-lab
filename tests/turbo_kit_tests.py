"""Host regressions; fixtures and subprocess output stay in the checkout."""
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = Path(sys.argv.pop(1)).resolve()
BASH = shutil.which("bash")
ABLATE = ROOT / "scripts/bench/turbo/ablate.sh"


def run(*args, **kwargs):
    return subprocess.run([str(a) for a in args], cwd=ROOT, capture_output=True,
                          text=True, timeout=10, **kwargs)


def record(step=0, pos=10, emitted=0, vocab=2):
    return struct.pack("<6I", 0x42434C47, step, pos, 0, emitted, vocab) + struct.pack("<" + "f" * vocab, *range(vocab))


class BenchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def arms(self, family, *args):
        result = run(BASH, ABLATE.as_posix(), "arms", family, *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return {name: (cli, env) for name, cli, env in
                (line.split("|") for line in result.stdout.splitlines())}

    def test_minimax_manifest_admission(self):
        self.assertEqual(self.arms("minimax")["all"], ("", ""))
        pending = self.arms("minimax", "--pending")["all"][1]
        self.assertIn("STRIX_MM_PF_MSA_FLASH_LAYERS=0-59", pending)
        self.assertNotIn("fast_i4", pending)
        result = run(BASH, ABLATE.as_posix(), "run", "minimax", "--out", "unused", "--kit", "unused", "--queue", "true")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no runnable", result.stderr)

    def test_glm_resolved_execution_switches(self):
        for name, (_, env) in self.arms("glm").items():
            with self.subTest(arm=name):
                result = run(HELPER, "--glm-arm", name, *shlex.split(env))
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_minimax_cli_receipt_matches_policy(self):
        env = {key: value for key, value in os.environ.items() if not key.startswith("STRIX_")}
        result = run(HELPER, "--minimax-cli", "--profile", "turbo", "--print-effective-config", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        self.assertEqual(values["STRIX_MM_PF_MSA_FLASH"], values["prefill.msa_flash_requested"])
        self.assertEqual(values["STRIX_MM_PF_MSA_FLASH_LAYERS"], values["prefill.msa_flash_layers_requested"])
        self.assertEqual(values["STRIX_MM_PF_MSA_FLASH"], "0")

    def test_ticket_size_validation(self):
        for value in ("1", "0", "-1", "abc", "2.5", "08", "999999999999999999999"):
            with self.subTest(value=value):
                result = run(BASH, ABLATE.as_posix(), "run", "qwen", "--per-ticket", value,
                             "--out", "unused", "--kit", "unused", "--queue", "true")
                self.assertEqual(result.returncode, 64, result.stderr)
                self.assertIn("--per-ticket", result.stderr)

    def test_queue_failure_and_valid_two_arm_ticket(self):
        for queue in ("false", "strix-missing-queue-command"):
            result = run(BASH, ABLATE.as_posix(), "run", "qwen", "--per-ticket", "2",
                         "--out", "unused", "--kit", "unused", "--queue", queue)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("failed group 0 (turbo3-bundle-qwen-g0)", result.stderr)
            self.assertNotIn("failed group 1", result.stderr)
        result = run(BASH, ABLATE.as_posix(), "run", "qwen", "--per-ticket", "2",
                     "--out", "unused", "--kit", "unused", "--queue", "true")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pending_is_inspection_only(self):
        result = run(BASH, ABLATE.as_posix(), "run", "glm", "--pending",
                     "--out", "unused", "--kit", "unused", "--queue", "true")
        self.assertEqual(result.returncode, 64)

    def compare(self, ref, cand):
        (self.dir / "ref").write_bytes(ref)
        (self.dir / "cand").write_bytes(cand)
        return run(sys.executable, ROOT / "scripts/bench/turbo/bclg_kl.py", self.dir / "ref", self.dir / "cand")

    def test_logit_alignment_and_coverage(self):
        ref = record() + record(1, 11)
        for cand in (record(1), record(pos=11), record(vocab=3), record(), ref + record(2, 12),
                     ref + b"BCLG", record()[:-1], b"", record() + record(2, 12)):
            with self.subTest(capture=cand[:24]):
                self.assertNotEqual(self.compare(ref, cand).returncode, 0)
        self.assertEqual(self.compare(ref, ref).returncode, 0)

    def test_logit_divergence_is_explicit(self):
        result = self.compare(record() + record(1, 11), record(emitted=1))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rows=1", result.stdout)
        self.assertIn("divergence=step=0,position=10,ref=0,cand=1", result.stdout)

    def test_documented_entry_uses_bash(self):
        text = (ROOT / "docs/benchmarking.md").read_text(encoding="utf-8")
        self.assertIn("`bash scripts/bench/turbo/ablate.sh run qwen", text)
        self.assertNotIn("`scripts/bench/turbo/ablate.sh run qwen", text)

    def test_final_power_report_cannot_anchor_prefill(self):
        # Execute the real matrix with host stubs to capture its receipt arguments.
        qwen = self.dir / "qwen"
        common = self.dir / "common"
        qwen.mkdir(); common.mkdir()
        shutil.copyfile(ROOT / "scripts/bench/qwen/length-matrix.sh", qwen / "length-matrix.sh")
        (common / "runlib.sh").write_text("KIT_T0=$(date +%s)\nkit_die() { exit 64; }\nkit_sha() { echo test; }\nkit_left() { echo 1000; }\nkit_fits() { return 0; }\n", newline="\n")
        (common / "run1.sh").write_text('printf "%s" "$KIT_RECEIPT_ARGS" > "$KIT_OUT/receipt-args"\n', newline="\n")
        (self.dir / "p8.ids").write_text("1 2 3 4 5 6 7 8\n")
        (self.dir / "model").touch()
        env = dict(os.environ, KIT_OUT=self.dir.as_posix(), KIT_SETTLE_S="0")
        result = run(BASH, (qwen / "length-matrix.sh").as_posix(), "--bin", "/usr/bin/true", "--model",
                     (self.dir / "model").as_posix(), "--ids", self.dir.as_posix(), "--sizes", "8", "--modes", "full",
                     "--rounds", "1", "--gen", "9", "--warm", "none", "--arm", "test||", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = shlex.split((self.dir / "receipt-args").read_text())
        power = {"available": True, "duration_ms": 1000, "tokens": 8, "avg_w": 20}
        (self.dir / "stdout").write_text(json.dumps({"power": {"prefill": power, "decode": dict(power, duration_ms=9999)}}, separators=(",", ":")))
        (self.dir / "stdout.ts").write_text("100\n")
        result = run(sys.executable, ROOT / "scripts/bench/common/receipt.py", "summarize", self.dir, "--tag", "synthetic", *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads((self.dir / "receipt.json").read_text())
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["prefills"], 0)
        self.assertIn("no prefill figure", receipt["invalid_reasons"])


if __name__ == "__main__":
    unittest.main()
