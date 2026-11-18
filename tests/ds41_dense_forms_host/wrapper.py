"""Exercise the real kit wrapper with a host-only runlib/executable fixture."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(os.environ.get("DENSE_FORMS_SCRIPT", ROOT / "scripts/bench/ds41/dense-forms.sh"))
BASH = os.environ.get("DENSE_FORMS_BASH") or shutil.which("bash")
RUNLIB = r'''
KIT_PY=$TEST_PYTHON
kit_die() { echo "$*" >&2; exit 1; }
kit_clean_env() { :; }
kit_sha() { echo fixture; }
kit_fits() { return 0; }
kit_run() {
    local dir=$1 label=${1##*/} rc=0
    shift 2
    mkdir -p "$dir"
    echo "$label" >> "$CALL_LOG"
    "$@" > "$dir/stdout" 2> "$dir/stderr" || rc=$?
    if [[ $label == "$FAIL_AT" ]]; then
        case $FAIL_MODE in
            stderr) : > "$dir/stdout"; echo 'HIP init failed' > "$dir/stderr"; rc=2 ;;
            timeout) rc=124 ;;
            mismatch) printf 'BITS FAIL\nPASS\n' > "$dir/stdout" ;;
            incomplete) printf 'wpack ok\n' > "$dir/stdout" ;;
            dispatch) printf 'dispatch form 9\nPASS\n' > "$dir/stdout" ;;
        esac
    fi
    printf '{"rc":%s}\n' "$rc" > "$dir/meta.json"
    if [[ $label == "$FAIL_AT" ]]; then
        case $FAIL_MODE in
            missing_meta) rm "$dir/meta.json" ;;
            bad_meta) echo broken > "$dir/meta.json" ;;
        esac
    fi
    return 0
}
'''
BINARY = '#!/usr/bin/env bash\nprintf "dispatch form %s\\nPASS\\n" "${STRIX_F16_Q8A8_WPACK:-1}"\n'


class WrapperTests(unittest.TestCase):
    def run_kit(self, fail_at="", mode=""):
        # All fixture files remain within this checkout, including on Windows.
        with tempfile.TemporaryDirectory(prefix="fixture-", dir=Path(__file__).parent) as tmp:
            path = Path(tmp)
            (path / "ds41").mkdir()
            (path / "common").mkdir()
            shutil.copyfile(SCRIPT, path / "ds41/dense-forms.sh")
            (path / "common/runlib.sh").write_text(RUNLIB, encoding="utf-8", newline="\n")
            binary = path / "binary"
            binary.write_text(BINARY, encoding="utf-8", newline="\n")
            binary.chmod(0o755)
            env = dict(os.environ, TEST_PYTHON=Path(sys.executable).as_posix(), KIT_OUT="out",
                       CALL_LOG="calls", FAIL_AT=fail_at, FAIL_MODE=mode,
                       STRIX_F16_Q8A8_WPACK="0")  # default gate must explicitly unset this
            result = subprocess.run([BASH, "ds41/dense-forms.sh", "--bin", "./binary"],
                                    cwd=path, env=env, text=True, capture_output=True, timeout=30)
            calls = (path / "calls").read_text().splitlines()
            return result, calls

    def test_success_and_separate_dispatch_processes(self):
        result, calls = self.run_kit()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, ["bits", "bits-off", "f1024", "f680"])
        self.assertIn("dispatch default=1 OFF=0 PASS", result.stdout)

    def test_failures_abort_before_next_run(self):
        stages = ["bits", "bits-off", "f1024", "f680"]
        for index, stage in enumerate(stages):
            for mode in ["stderr", "timeout", "mismatch", "incomplete", "missing_meta", "bad_meta"]:
                with self.subTest(stage=stage, mode=mode):
                    result, calls = self.run_kit(stage, mode)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("INVALID/ABORT", result.stdout)
                    self.assertEqual(calls, stages[:index + 1])

    def test_dispatch_must_match(self):
        for stage in ["bits", "bits-off"]:
            with self.subTest(stage=stage):
                result, calls = self.run_kit(stage, "dispatch")
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("expected library dispatch form", result.stdout)
                self.assertNotIn("f1024", calls)


if __name__ == "__main__":
    unittest.main()
