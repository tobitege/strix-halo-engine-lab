import tempfile
import unittest
from pathlib import Path

import numpy as np

import quality


class QualityTests(unittest.TestCase):
    def test_spread_has_true_next_token(self):
        self.assertEqual(quality.selections(9, 3), [3, 6])
        self.assertEqual(quality.selections(10, 3), [3, 6])
        with self.assertRaises(ValueError):
            quality.selections(9, 0)

    def test_comparison_and_different_input_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            a = np.array([[1., 2., 3.]], dtype="<f4")
            quality.spool.write(d / "E", [0, 1], [0], a)
            quality.spool.write(d / "T", [0, 1], [0], a + 3)
            result = quality.compare(d / "E", ["T=" + str(d / "T")])["arms"]["T"]
            self.assertAlmostEqual(result["kl_ref_run"], 0)
            self.assertAlmostEqual(result["dnll_run_minus_ref"], 0)
            self.assertEqual(result["top1_agreement"], 1)
            self.assertFalse(result["bitwise_equal"])
            quality.spool.write(d / "bad", [0, 2], [0], a)
            with self.assertRaises(ValueError):
                quality.compare(d / "E", ["T=" + str(d / "bad")])


if __name__ == "__main__":
    unittest.main()
