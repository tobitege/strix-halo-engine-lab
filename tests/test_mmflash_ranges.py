"""Host checks of the launcher's actual parsing, environment and null-selection functions."""
import pathlib
import shutil
import subprocess
import unittest


class RangesTests(unittest.TestCase):
    def test_combined_onsets(self):
        source = (pathlib.Path(__file__).resolve().parents[1] / "scripts/bench/mmflash/ranges.sh").read_text()
        # Load only the shell functions: no GPU tickets, detached jobs or filesystem setup.
        start = source.index("declare -A flash_ranges=")
        stop = source.index("\n[[ $REF", start)
        functions = source[start:stop]
        bash = shutil.which("bash")
        self.assertIsNotNone(bash)
        for suffix, onset in (("i30-59", 30), ("i40-59", 40), ("i50-59", 40), ("", 40)):
            with self.subTest(suffix=suffix):
                arm = "f40-59m11" + suffix
                command = functions + f"\nparse_candidate {arm}\nenv_of {arm}\nnulls_for {arm} n30s1 n40s1 n50s1 n{onset}s2\n"
                result = subprocess.run([bash, "-uc", command], capture_output=True, text=True, check=True)
                lines = result.stdout.splitlines()
                expected = "STRIX_MM_PF_MSA_FLASH=1 STRIX_MM_PF_MSA_FLASH_LAYERS=40-59"
                if suffix:
                    expected += " STRIX_MM_PF_IDX_WMMA=1 STRIX_MM_PF_IDX_WMMA_LAYERS=" + suffix[1:]
                self.assertEqual(lines, [expected, f"n{onset}s1", f"n{onset}s2"])


if __name__ == "__main__":
    unittest.main()
