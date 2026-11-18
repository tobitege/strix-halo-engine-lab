"""Standalone host regressions; no engine build or GPU required.

Run: python tests/qsa_score_split_loads_tests.py
Builds one extracted production accessor in Release, then runs CTest.
All temporary files stay under the checkout and are removed on exit.
"""
import csv
import io
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/bench/prefill-trace/kind_excess.py"
HEADER = "Kernel_Name,Start_Timestamp,End_Timestamp\n"


class TraceTests(unittest.TestCase):
    def run_trace(self, short, long, *args):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder = pathlib.Path(directory)
            (folder / "short.csv").write_text(short, encoding="utf-8")
            (folder / "long.csv").write_text(long, encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(SCRIPT), str(folder / "short.csv"),
                 str(folder / "long.csv"), *args], capture_output=True, text=True)

    def test_invalid_traces_reject_without_partial_output(self):
        invalid = [
            ("", 1), (HEADER, 1),
            (HEADER.replace("Start_Timestamp", "Start_Timestmp") + "k,0,100\n", 1),
            (HEADER + "good,0,100\nbad,broken,200\n", 3),
            (HEADER + "bad,0,broken\n", 2),
            (HEADER + "bad,200,100\n", 2),
            (HEADER + "bad,-1,100\n", 2),
            (HEADER + "   ,0,100\n", 2),
            (HEADER + "bad,0\n", 2),
            (HEADER + "bad,0,100,extra\n", 2),
            (HEADER + '"unterminated,0,100\n', 2),
            ('"unterminated header\n', 1),
            (HEADER.rstrip() + ",Kernel_Name\nk,0,100,k\n", 1),
        ]
        for content, line in invalid:
            for bad_side in ("short", "long"):
                with self.subTest(content=content, bad_side=bad_side):
                    valid = HEADER + "good,0,1000000\n"
                    result = self.run_trace(content if bad_side == "short" else valid,
                                            content if bad_side == "long" else valid)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, "")
                    self.assertIn(f"{bad_side}.csv:{line}:", result.stderr)

    def test_full_keys_and_phase_regex_survive_label_collision(self):
        prefix = "strix::qsa_flash::score_kernel<" + "long_template_argument_" * 6
        first, second = prefix + "false>", prefix + "true>"
        def trace(name, duration):
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Kernel_Name", "Start_Timestamp", "End_Timestamp"])
            writer.writerow([f"void {name}(float*)", 0, duration])
            return output.getvalue()
        result = self.run_trace(trace(first, 1000000), trace(second, 4000000),
                                "--phase", "split=true>", "--phase", "merged=false>")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("split | 0.0 | 4.0 | +4.0", result.stdout)
        self.assertIn("merged | 1.0 | 0.0 | -4.0", result.stdout)
        self.assertIn(first.removeprefix("strix::"), result.stdout)
        self.assertIn(second.removeprefix("strix::"), result.stdout)
        self.assertEqual(len([line for line in result.stdout.splitlines() if line.startswith("  ")]), 2)

    def test_valid_totals_and_normalization(self):
        result = self.run_trace(HEADER + "void strix::k(float*),0,1000000\n",
                                HEADER + "void (anonymous namespace)::k(float*),0,5000000\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("total | 1.0 | 5.0 | +1.0", result.stdout)
        self.assertEqual(len([line for line in result.stdout.splitlines() if line.startswith("  ")]), 1)


def checked(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("\n".join((result.stdout + result.stderr).splitlines()[-20:]))
    return result


def score_policy_tests(source=None, before=False):
    if not shutil.which("cmake") or not shutil.which("ctest"):
        raise unittest.SkipTest("CMake/CTest unavailable for extracted C++ host checks")
    if source is None:
        source = (ROOT / "src/hip_qsa_prefill_flash.inc").read_text(encoding="utf-8")
    accessor = re.search(r"inline bool score_split_loads\(\) \{.*?^\}", source,
                         re.MULTILINE | re.DOTALL).group()
    # Compile the actual accessor, using the production env registry and shared
    # suppression policy. No HIP declarations or mock policy implementation.
    driver = r'''
#include "strix/env_flags.hpp"
#include "strix/qpf2k_defaults.hpp"
#include <cstdio>
#include <string_view>
namespace strix::qsa_flash {
ACCESSOR
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    using namespace strix;
    qpf2k::baseline = envcfg::flag("STRIX_BASELINE", false, EnvFlagMatch::exact);
    const bool expected = std::string_view(argv[1]) == "on";
    if (qsa_flash::score_split_loads() != expected) {
        std::fputs("wrong effective split-load policy\n", stderr); return 1;
    }
    bool recorded = false;
    for (const auto& entry : envcfg::registered())
        recorded |= entry.name == "STRIX_QSA_SCORE_SPLIT_LOADS";
    if (!recorded) return 3;
    if (expected) {
        qpf2k::baseline = true;
        if (qsa_flash::score_split_loads()) return 4;
        qpf2k::baseline = false;
        qpf2k::gsq_model = true;
        if (qsa_flash::score_split_loads()) return 5;
        qpf2k::gsq_model = false;
        if (!qsa_flash::score_split_loads()) return 6;
    }
    return 0;
}
'''.replace("ACCESSOR", accessor)
    cmake = f'''
cmake_minimum_required(VERSION 3.20)
project(qsa_score_split_loads_host LANGUAGES NONE)
include(CheckLanguage)
check_language(CXX)
if(NOT CMAKE_CXX_COMPILER)
  file(WRITE "${{CMAKE_BINARY_DIR}}/no-compiler" "unavailable")
  return()
endif()
enable_language(CXX)
enable_testing()
add_executable(score_policy score_policy.cpp)
target_compile_features(score_policy PRIVATE cxx_std_20)
target_include_directories(score_policy PRIVATE "{ROOT.as_posix()}/include")
if(MSVC)
  target_compile_options(score_policy PRIVATE /W4 /WX)
  target_compile_definitions(score_policy PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(score_policy PRIVATE -Wall -Wextra -Werror)
endif()
'''
    cases = [("default", "on", "", "", ""),
             ("individual_off", "off", "0", "", ""),
             ("master_off", "off", "", "0", ""),
             ("baseline", "off", "", "", "1"),
             ("master_overrides_on", "off", "1", "0", ""),
             ("baseline_overrides_on", "off", "1", "", "1")]
    for name, expected, split, defaults, baseline in cases:
        cmake += f'''
add_test(NAME {name} COMMAND score_policy {expected})
set_tests_properties({name} PROPERTIES ENVIRONMENT
  "STRIX_QSA_SCORE_SPLIT_LOADS={split};STRIX_QPF2K_DEFAULTS={defaults};STRIX_BASELINE={baseline}")
'''
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        folder = pathlib.Path(directory)
        (folder / "score_policy.cpp").write_text(driver, encoding="utf-8")
        (folder / "CMakeLists.txt").write_text(cmake, encoding="utf-8")
        build = folder / "build"
        command = ["cmake", "-S", str(folder), "-B", str(build)]
        if sys.platform == "win32":
            command += ["-G", "Visual Studio 17 2022", "-A", "x64"]
        else:
            command += ["-DCMAKE_BUILD_TYPE=Release"]
        configured = subprocess.run(command, capture_output=True, text=True)
        if configured.returncode:
            if "could not find any instance of Visual Studio" in configured.stderr:
                raise unittest.SkipTest("MSVC unavailable for extracted C++ host checks")
            raise RuntimeError("\n".join((configured.stdout + configured.stderr).splitlines()[-20:]))
        if (build / "no-compiler").exists():
            raise unittest.SkipTest("No C++ compiler available for extracted host checks")
        checked(["cmake", "--build", str(build), "--config", "Release"])
        command = ["ctest", "--test-dir", str(build), "-C", "Release", "--output-on-failure"]
        if before:
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0 or "17% tests passed, 5 tests failed out of 6" not in result.stdout:
                raise RuntimeError("Original accessor did not fail the five expected policy cases")
            print("Before fix: five policy regression cases fail, as expected.")
            return
        result = checked(command)
        print("\n".join(result.stdout.splitlines()[-20:]))


class ScorePolicyTests(unittest.TestCase):
    def test_extracted_production_policy(self):
        score_policy_tests()


if __name__ == "__main__":
    if "--verify-before" in sys.argv:
        original = checked(["git", "show", "HEAD:scripts/bench/prefill-trace/kind_excess.py"]).stdout
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            original_path = pathlib.Path(directory) / "kind_excess.py"
            original_path.write_text(original, encoding="utf-8")
            current_script, SCRIPT = SCRIPT, original_path
            output = io.StringIO()
            prior = unittest.TextTestRunner(stream=output).run(
                unittest.defaultTestLoader.loadTestsFromTestCase(TraceTests))
            SCRIPT = current_script
            failed = {getattr(test, "test_case", test)._testMethodName for test, _ in prior.failures}
            if prior.errors or failed != {"test_invalid_traces_reject_without_partial_output",
                                          "test_full_keys_and_phase_regex_survive_label_collision"}:
                raise RuntimeError("Original trace script did not fail the expected regressions")
            print("Before fix: input-validation and full-kind-key regressions fail, as expected.")
        score_policy_tests(checked(["git", "show", "HEAD:src/hip_qsa_prefill_flash.inc"]).stdout, before=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if not result.wasSuccessful():
        sys.exit(1)
