"""Host regressions for PR 669. Run: python -m unittest discover -s tests -p test_mmflash.py -v

All fixtures/build products stay under the checkout; the C++ fixture uses Release and CTest.
"""
import ast
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
KIT = ROOT / "scripts/bench/mmflash"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


summary = module("mmflash_summary", KIT / "summary.py")
gate = module("mmflash_gate", ROOT / "tools/strix-gate/fast_logit_gate.py")


def run_checked(args, **kwargs):
    p = subprocess.run(args, capture_output=True, text=True, **kwargs)
    if p.returncode:
        raise AssertionError("command failed: %s\n%s" % (args[0], "\n".join((p.stdout + p.stderr).splitlines()[-20:])))
    return p


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mmflash-test-", dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)


class WitnessTests(Fixture):
    def test_generated_hook_prompt_coverage_and_session_timestamps(self):
        engine = self.root / "src/minimax/engine/engine.hip"
        launcher = self.root / "apps/minimax_rank_main.cpp"
        engine.parent.mkdir(parents=True)
        launcher.parent.mkdir()
        shutil.copyfile(ROOT / "src/minimax/engine/engine.hip", engine)
        shutil.copyfile(ROOT / "apps/minimax_rank_main.cpp", launcher)
        run_checked([sys.executable, str(KIT / "null_patch.py"), str(engine)])
        patched = engine.read_text(encoding="utf-8")
        app = launcher.read_text(encoding="utf-8")
        self.assertIn("values %llu prompt %s", patched)
        self.assertIn('strix::envcfg::text("STRIX_MM_DIAG_ATTN_ULP")', patched)
        self.assertIn("mm::mmflash_prompt_json = strix::serve::Json(name).dump();", app)
        # Extract only host statements emitted by the patch, never compile/alter a HIP kernel.
        tree = ast.parse((KIT / "null_patch.py").read_text(encoding="utf-8"))
        hook = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "H" for t in n.targets))
        begin = app[app.index("    mm::mmflash_prompt_json ="):]
        begin = begin[:begin.index("    const auto t0 = Clock::now();")]
        end = app[app.index("    const double mmflash_end ="):]
        end = end[:end.index("    mm::mmflash_prompt_json.clear();") + len("    mm::mmflash_prompt_json.clear();")]
        if not shutil.which("cmake") or not shutil.which("ctest"):
            self.skipTest("CMake/CTest unavailable for emitted C++ host fixture")
        cpp = r'''
#include "strix/env_flags.hpp"
#include "strix/serve/json.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
namespace mm { std::string mmflash_prompt_json; }
using mm::mmflash_prompt_json;
int launches = 0;
#define hipLaunchKernelGGL(...) (++launches)
void session(const std::string& name) {
    const std::vector<unsigned> prompt(9);
    const unsigned L = 0, kQLocal = 2;
    const std::size_t n = prompt.size();
    const unsigned long long pos0 = 0;
''' + begin + hook + end + r'''
}
int main() {
    session("cold");
    session("warm\"\\\n\t");
    return launches == 2 ? 0 : 1;
}
'''
        (self.root / "host.cpp").write_text(cpp, encoding="utf-8")
        (self.root / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.20)
project(mmflash_host LANGUAGES CXX)
enable_testing()
add_executable(mmflash_host host.cpp "''' + ROOT.as_posix() + '''/src/serve/json.cpp")
target_include_directories(mmflash_host PRIVATE "''' + ROOT.as_posix() + '''/include")
target_compile_features(mmflash_host PRIVATE cxx_std_20)
if(MSVC)
  target_compile_options(mmflash_host PRIVATE /W4 /WX)
  target_compile_definitions(mmflash_host PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(mmflash_host PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME mmflash_witness COMMAND mmflash_host)
set_tests_properties(mmflash_witness PROPERTIES ENVIRONMENT "STRIX_MM_DIAG_ATTN_ULP=0:1")
''', encoding="utf-8")
        build = self.root / "build"
        run_checked(["cmake", "-S", str(self.root), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"])
        run_checked(["cmake", "--build", str(build), "--config", "Release"])
        run_checked(["ctest", "--test-dir", str(build), "-C", "Release", "--output-on-failure"])
        output = (build / "Testing/Temporary/LastTest.log").read_bytes()
        events = [line for line in output.splitlines() if line.startswith(b"DIAG attn ulp:")]
        names = ["cold", 'warm"\\\n\t']
        self.assertEqual([json.loads(gate.MM_DIAG.fullmatch(e)[6]) for e in events], names)
        receipt = {"scope": "process", "engine": "minimax", "rank": 0,
                   "levers": [{"name": n, "effective": v} for n, v in
                              (("minimax.msa_flash", "0"), ("minimax.moe_prefill_format", "exl3"),
                               ("minimax.prefill_identity", "default"))], "switches": {"set": {}}}
        prefix = (gate.RECEIPT_PREFIX + json.dumps(receipt)).encode() + b"\n"
        prompts = {name: {"tokens": 9} for name in names}
        witness = gate.minimax_execution(prefix + b"\n".join(events), 0, prompts)
        self.assertEqual(witness["control"]["seed"], 1)
        with self.assertRaisesRegex(gate.GateError, "missing for prompts"):
            gate.minimax_execution(prefix + events[0], 0, prompts)
        with self.assertRaisesRegex(gate.GateError, "unsealed prompt"):
            gate.minimax_execution(prefix + b"\n".join(events), 0, {"cold": {"tokens": 9}})
        windows = [json.loads(line.removeprefix(b"MMFLASH_SESSION ")) for line in output.splitlines()
                   if line.startswith(b"MMFLASH_SESSION ")]
        self.assertEqual([w["prompt"] for w in windows], names)
        self.assertTrue(all(w["tokens"] == 9 and w["end"] > w["start"] for w in windows))


class ReportTests(Fixture):
    def setUp(self):
        super().setUp()
        self.cmp = self.root / "cmp"
        self.cmp.mkdir()
        self.run = self.root / "run"
        self.report = {"prompts": {"warm": {}}, "pooled": {"pass": True, "selection": None,
                       "dnll_rule": {"checks": [{"name": "floor", "required": True, "pass": True}]}}}
        for side, rid in (("ref", "exact"), ("cand", "f0-59m11")):
            arm = self.run / rid
            arm.mkdir(parents=True)
            identity = {"run_id": rid, "arm": rid, "env": {}, "attest": {}}
            self.report[side] = identity
            self.report[side + "_binary"] = ["hash0", "hash1"]
            for rank in (0, 1):
                manifest = dict(identity, prompts={"warm": {}}, binary={"sha256": "hash%d" % rank})
                (arm / ("manifest-r%d.json" % rank)).write_text(json.dumps(manifest), encoding="utf-8")
        (self.cmp / "expected.tsv").write_text("f0-59m11\tf0-59m11\tfloor\n", encoding="utf-8")

    def write_report(self, report=None, rc=0):
        (self.cmp / "f0-59m11.json").write_text(json.dumps(report or self.report), encoding="utf-8")
        (self.cmp / "f0-59m11.rc").write_text(str(rc), encoding="utf-8")

    def table(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = summary.table(str(self.cmp), str(self.root / "timing.txt"), str(self.run), "exact")
        return status, output.getvalue()

    def test_only_expected_current_reports_and_missing_evidence(self):
        (self.cmp / "unrelated.json").write_text(json.dumps(self.report), encoding="utf-8")
        status, output = self.table()
        self.assertEqual(status, 2)
        self.assertNotIn("floor pass=True", output)
        self.assertNotIn("unrelated", output)
        self.write_report()
        self.assertEqual(self.table()[0], 0)

    def test_completed_gate_failure_distinct_from_error(self):
        self.report["pooled"]["pass"] = False
        self.write_report(rc=1)
        status, output = self.table()
        self.assertEqual(status, 1)
        self.assertIn("floor pass=False", output)
        self.write_report(rc=2)
        self.assertEqual(self.table()[0], 2)
        self.write_report(rc=0)
        self.assertEqual(self.table()[0], 2)

    def test_report_identity_and_rule_must_match_current_manifests(self):
        for side in ("ref", "cand"):
            for key in ("run_id", "arm", "env", "attest"):
                with self.subTest(side=side, key=key):
                    report = copy.deepcopy(self.report)
                    report[side][key] = "old"
                    self.write_report(report)
                    self.assertEqual(self.table()[0], 2)
        for key, value in (("ref_binary", ["old", "hash1"]), ("cand_binary", []), ("prompts", {"other": {}})):
            report = copy.deepcopy(self.report)
            report[key] = value
            self.write_report(report)
            self.assertEqual(self.table()[0], 2)
        self.report["pooled"]["dnll_rule"] = None
        self.write_report()
        self.assertEqual(self.table()[0], 2)


class WrapperTests(Fixture):
    def test_fresh_invocation_missing_report_and_completed_failure(self):
        bash = shutil.which("bash")
        if not bash:
            self.skipTest("Bash unavailable")
        kit = self.root / "kit"
        kit.mkdir()
        for name in ("ranges.sh", "summary.py"):
            shutil.copyfile(KIT / name, kit / name)
        shim = self.root / "bin"
        shim.mkdir()
        python = sys.executable.replace("\\", "/")
        (shim / "python3").write_text('#!/bin/bash\nexec "' + python + '" "$@"\n', encoding="utf-8")
        (shim / "sleep").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        turn = self.root / "turn"
        turn.write_text('#!/bin/bash\nshift\nexec "$@"\n', encoding="utf-8")
        fake = self.root / "fake.py"
        fake.write_text('''import json, os, pathlib, sys
run = pathlib.Path(os.environ["MMF_RUN"])
def identity(rid):
    return {"run_id": rid, "arm": rid, "env": {}, "attest": {}}
if sys.argv[1] == "seal":
    arm = run / sys.argv[2]
    arm.mkdir(parents=True, exist_ok=True)
    for rank in (0, 1):
        m = dict(identity(arm.name), prompts={"warm": {}}, binary={"sha256": "hash"})
        (arm / ("manifest-r%d.json" % rank)).write_text(json.dumps(m))
        (arm / ("r%d.log" % rank)).write_text("TTFT 1.0 s\\n")
    (arm / "sessions").write_text("warm")
    (arm / "seal.verdict").write_text("OK\\n")
else:
    mode = os.environ["MMF_FAKE_FLOOR"] if "--dnll-rule" in sys.argv else "pass"
    if mode == "missing": sys.exit(2)
    passed = mode == "pass"
    report = {"ref": identity(pathlib.Path(sys.argv[2]).name),
              "cand": identity(pathlib.Path(sys.argv[3]).name),
              "ref_binary": ["hash", "hash"], "cand_binary": ["hash", "hash"],
              "prompts": {"warm": {}}, "pooled": {"pass": passed, "selection": None,
              "dnll_rule": {"checks": [{"name": "floor", "required": True, "pass": passed}]}
                  if "--dnll-rule" in sys.argv else None}}
    pathlib.Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(report))
    sys.exit(0 if passed else 1)
''', encoding="utf-8")
        (kit / "arm.sh").write_text('#!/bin/bash\nexit "${MMF_ARM_RC:-0}"\n', encoding="utf-8")
        (kit / "seal.sh").write_text('#!/bin/bash\npython3 "' + fake.as_posix() + '" seal "$1"\n', encoding="utf-8")
        for path in (shim / "python3", shim / "sleep", turn):
            path.chmod(0o755)
        for index, (mode, arm_rc, expected) in enumerate((("pass", "0", 0), ("missing", "0", 2),
                                                        ("fail", "0", 1), ("pass", "3", 2))):
            with self.subTest(mode=mode, arm_rc=arm_rc):
                run = self.root / ("run%d" % index)
                env = dict(os.environ, MMF_RUN=run.as_posix(), MMF_TURN=turn.as_posix(),
                           MMF_GATE=fake.as_posix(), MMF_PROMPTS=self.root.as_posix(), MMF_SESSIONS="warm",
                           MMF_FAKE_FLOOR=mode, MMF_ARM_RC=arm_rc)
                run_checked([sys.executable, str(fake), "seal", "exact"], env=env)
                # An earlier parent-level report must never enter the new summary.
                stale = self.root / "cmp"
                stale.mkdir(exist_ok=True)
                (stale / "old.json").write_text('{"pooled":{"pass":true}}', encoding="utf-8")
                shim_path = ('/' + shim.drive[0].lower() + shim.as_posix()[2:]) if os.name == "nt" else shim.as_posix()
                command = 'export PATH="' + shim_path + ':$PATH"; exec bash "$1" exact n0s1 f0-59m11'
                proc = subprocess.run([bash, "-c", command, "test", (kit / "ranges.sh").as_posix()],
                                      env=env, capture_output=True, text=True)
                dirs = sorted(self.root.glob("ranges.*"), key=lambda p: p.stat().st_mtime_ns)
                self.assertEqual(len(dirs), index + 1)
                current = dirs[-1]
                output = (current / "ranges.summary").read_text()
                self.assertEqual(proc.returncode, expected, (proc.stdout + proc.stderr + output)[-2000:])
                self.assertEqual((current / "ranges.status").read_text().strip(), str(expected))
                self.assertEqual((current / "ranges.done").exists(), expected != 2)
                self.assertNotIn("old", output)
                if expected == 2:
                    self.assertNotIn("floor pass=True", output)
                if expected == 1:
                    self.assertIn("floor pass=False", output)


class TimingTests(Fixture):
    def monitor(self, rows):
        path = self.root / "mon.txt"
        path.write_text("".join("%s busy=%s sclk=0: %sMhz * ev=0\n" % row for row in rows), encoding="utf-8")
        return str(path)

    def test_warm_only_integrates_product_including_low_busy(self):
        path = self.monitor([(0, 100, 2000), (1, 100, 2000), (2, 100, 1000),
                             (3, 20, 2000), (4, 0, 0)])
        b, c, _, clocks = summary.mon(path, 2, 4, 10)
        self.assertEqual((b, c), (60, 1500))
        self.assertAlmostEqual(clocks, 140)  # (1*1000 + .2*2000) / 10; not 180.
        self.assertAlmostEqual(summary.mon(path, 2.5, 3.5, 10)[3], 70)

    def test_unavailable_instead_of_zero_or_extrapolation(self):
        path = self.monitor([(0, 0, 1000), (1, 0, 1000)])
        self.assertEqual(summary.mon(path, 0, 1, 10)[3], 0)
        for start, end, tokens in ((-1, 1, 10), (0, 2, 10), (0, 1, 0), (1, 1, 10)):
            self.assertIsNone(summary.mon(path, start, end, tokens))
        path = self.monitor([(0, 100, 1000), (3, 100, 1000)])
        self.assertIsNone(summary.mon(path, 0, 3, 10))
        path = self.monitor([(1, 100, 1000), (0, 100, 1000)])
        self.assertIsNone(summary.mon(path, 0, 1, 10))

    def test_per_rank_session_identity_and_missing_timestamps(self):
        (self.root / "sessions").write_text("cold,warm\n", encoding="utf-8")
        for rank, interval, clock in ((0, (2, 3), 1000), (1, (20, 21), 2000)):
            rows = [{"prompt": "cold", "start": 0, "end": 1, "tokens": 10},
                    {"prompt": "warm", "start": interval[0], "end": interval[1], "tokens": 10}]
            (self.root / ("r%d.log" % rank)).write_text(
                "TTFT 1.0 s\n" * 2 + "".join("MMFLASH_SESSION " + json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            shutil.copyfile(self.monitor([(t, 100, clock) for t in interval]), self.root / ("mon%d.txt" % rank))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            summary.timing(str(self.root))
        self.assertIn("gclk/tok 100.00M", output.getvalue())
        self.assertIn("gclk/tok 200.00M", output.getvalue())
        (self.root / "r1.log").write_text("TTFT 1.0 s\n", encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            summary.timing(str(self.root))
        self.assertIn("r1 warm telemetry unavailable", output.getvalue())


if __name__ == "__main__":
    unittest.main()
