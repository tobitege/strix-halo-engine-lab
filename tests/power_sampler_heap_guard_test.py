"""Check the actual heap-test preprocessor branches without a glibc dependency."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class HeapGuardTests(unittest.TestCase):
    def test_allocator_guard(self):
        source = (ROOT / "tests/power_sampler_tests.cpp").read_text()
        start = source.index("void ring_memory_fixed_at_construction() {")
        end = source.index("\n}\n", start) + 3
        body = source[start:end]
        cases = [
            ("glibc_plain", True, None, None, True),
            ("gcc_asan", True, "__SANITIZE_ADDRESS__", None, False),
            ("gcc_tsan", True, "__SANITIZE_THREAD__", None, False),
            ("clang_plain", True, None, "none", True),
            ("clang_asan", True, None, "address_sanitizer", False),
            ("clang_tsan", True, None, "thread_sanitizer", False),
            ("non_glibc", False, None, None, False),
        ]
        with tempfile.TemporaryDirectory(prefix="heap-guard-", dir=ROOT) as tmp:
            path = Path(tmp) / "guard.cpp"
            for name, glibc, gcc_macro, feature, measure in cases:
                with self.subTest(name=name):
                    # Isolate the simulated compiler/allocator from host builtins.
                    prefix = "#undef __GLIBC__\n#undef __SANITIZE_ADDRESS__\n#undef __SANITIZE_THREAD__\n"
                    prefix += "#ifdef __has_feature\n#undef __has_feature\n#endif\n"
                    if glibc:
                        prefix += "#define __GLIBC__ 2\n"
                    if gcc_macro:
                        prefix += f"#define {gcc_macro} 1\n"
                    if feature is not None:
                        prefix += "#define __has_feature(x) TEST_FEATURE_##x\n"
                        for candidate in ("address_sanitizer", "thread_sanitizer"):
                            prefix += f"#define TEST_FEATURE_{candidate} {int(candidate == feature)}\n"
                    path.write_text(prefix + body)
                    msvc = Path(COMPILER).name.lower() in ("cl", "cl.exe")
                    flags = (["/nologo", "/EP", "/TP", "/Zc:preprocessor", "/W4", "/WX"]
                             if msvc else ["-E", "-P", "-x", "c++", "-Wall", "-Wextra", "-Werror",
                                           "-Wno-builtin-macro-redefined"])
                    result = subprocess.run([COMPILER, *flags, str(path)],
                                            capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                    self.assertEqual("mallinfo2()" in result.stdout, measure, name)
                    self.assertEqual("SKIP ring_memory_fixed_at_construction" in result.stdout,
                                     not measure, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True, help="C++ compiler executable")
    args = parser.parse_args()
    COMPILER = args.compiler
    unittest.main(argv=[__file__])
