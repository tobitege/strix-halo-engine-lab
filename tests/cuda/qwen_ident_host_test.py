"""Exercise ident.sh with a fake inference process; no device/model is needed."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(os.environ.get("STRIX_IDENT_SCRIPT", ROOT / "scripts/bench/cuda/qwen_fast/ident.sh"))
BASH = shutil.which("bash")
MOCK = r'''
import json, os, pathlib, sys
args = sys.argv[1:]
def option(name):
    return args[len(args) - 1 - args[::-1].index(name) + 1]
mode = os.environ.get("MODE", "equal")
if "--print-effective-config" in args or mode == "noop": sys.exit(0)
if mode == "failure": sys.exit(3)
out = pathlib.Path(option("--out"))
rows = int(option("--rows"))
names = [args[i+1].split("=", 1)[0] for i, a in enumerate(args) if a == "--prompt"]
forms = dict.fromkeys(["hc_mma", "mtp_mma", "qsa_mma", "moe_mma", "moe_route_bf16"], False)
executions = [{"prompt": n, "run": 0, "rows": 2, "forms": forms} for n in names]
for n in names:
    if mode == "missing" and n == names[-1]: continue
    data = bytes([1 if mode == "different" else 0]) * (rows * 248320 * 4)
    if mode == "truncated": data = data[:-4]
    if mode == "empty": data = b""
    (out / f"q-{n}-tf-r0.f32").write_bytes(data)
if mode == "extra": (out / "extra.f32").write_bytes(b"extra")
if mode == "no_receipt": sys.exit(0)
if mode == "bad_run": executions[0]["run"] = 1
if mode == "bad_forms": executions[0]["forms"] = {}
if mode == "bad_prompt": executions[0]["prompt"] = "unexpected"
(out / "effective-config.json").write_text("{}" if mode == "bad_receipt" else json.dumps({"executions": executions}))
'''


class IdentityGate(unittest.TestCase):
    def setUp(self):
        # All generated files remain within the checkout, including on Windows.
        scratch = ROOT / ".ident-host-scratch"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="ident-host-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.out = self.work / "out"
        self.mock = self.work / "tf.sh"
        body = self.work / "tf.py"
        body.write_text(MOCK)
        self.mock.write_text(f"#!/usr/bin/env bash\nexec {shlex.quote(Path(sys.executable).as_posix())} {shlex.quote(body.as_posix())} \"$@\"\n", newline="\n")
        self.mock.chmod(0o755)
        # Git Bash may find the Windows Store python3 alias first.
        shim = self.work / "python3"
        shim.write_text(f"#!/usr/bin/env bash\nexec {shlex.quote(Path(sys.executable).as_posix())} \"$@\"\n", newline="\n")
        shim.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.work) + os.pathsep + os.environ["PATH"])
        self.ids = self.work / "prompt.u32"
        self.ids.write_bytes(bytes(12))

    def run_gate(self, arms=("exact||", "fast||")):
        command = [BASH, SCRIPT.as_posix(), "--tf", self.mock.as_posix(), "--model", "model.gguf",
                   "--out", self.out.as_posix(), "--rows", "1",
                   "--prompt", f"one={self.ids.as_posix()}", "--prompt", f"two={self.ids.as_posix()}"]
        for arm in arms:
            command += ["--arm", arm]
        return subprocess.run(command, env=self.env, capture_output=True, text=True, timeout=30)

    def expect(self, code, arms=("exact||", "fast||")):
        result = self.run_gate(arms)
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        return result

    def test_equal_and_fresh_reuse(self):
        self.expect(0)
        self.expect(0)
        self.assertEqual(len(list(self.out.glob("ident.*"))), 2)

    def test_difference(self):
        self.expect(1, ("exact||", "fast|MODE=different|"))

    def test_duplicate_and_alias_names(self):
        for name in ("same", "SAME", "x/../same", "./same", "..", "same\\child"):
            with self.subTest(name=name):
                self.expect(2, ("same||", f"{name}|MODE=different|"))

    def test_stale_dumps_and_inspection_exit(self):
        self.expect(0)
        # Reproduce old-layout stale dumps too, so this test fails on the old script.
        for arm in ("exact", "fast"):
            directory = self.out / arm
            directory.mkdir(exist_ok=True)
            for name in ("one", "two"):
                (directory / f"q-{name}-tf-r0.f32").write_bytes(bytes(248320 * 4))
        self.expect(2, ("exact||--print-effective-config", "fast||--print-effective-config"))

    def test_incomplete_or_invalid_evidence(self):
        for mode in ("noop", "missing", "truncated", "empty", "extra", "no_receipt",
                     "bad_receipt", "bad_run", "bad_forms", "bad_prompt", "failure"):
            with self.subTest(mode=mode):
                self.expect(2, (f"exact|MODE={mode}|", f"fast|MODE={mode}|"))


if __name__ == "__main__":
    if not BASH:
        sys.exit("bash is required")
    unittest.main()
