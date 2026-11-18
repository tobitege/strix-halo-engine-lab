import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.qualification.turbo_bar import losses, spool
from tools.qualification.turbo_bar.tests.helpers import INPUTS, make_layout


class LossTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_known_kl(self):
        pa, pb = np.array([.5, .3, .2]), np.array([.4, .4, .2])
        inputs = [0, 1, 2, 0]
        spool.write(self.tmp / "a.bin", inputs, [1], [np.log(pa)])
        spool.write(self.tmp / "b.bin", inputs, [1], [np.log(pb)])
        case = {"case": "c", "format": "spool", "files": {"a": str(self.tmp / "a.bin"), "b": str(self.tmp / "b.bin")}}
        r = losses.case_sums(case, [("a", "b")])
        s = r["sums"]["a>b"]
        self.assertAlmostEqual(s["kl"], .5 * math.log(.5 / .4) + .3 * math.log(.3 / .4), places=6)
        self.assertEqual(s["dis"], 0)      # argmax 0 for both (tie in b resolves to the lowest id)
        # label is inputs[2] = 2 with p=.2 in both -> dNLL = 0
        self.assertAlmostEqual(s["dnll"], 0.0, places=6)

    def test_e_repeat_null_and_analysis(self):
        res = losses.analyze(make_layout(self.tmp))
        self.assertEqual(res["status"], "VALID")
        self.assertTrue(res["nulls"]["e_repeat_bitwise_identical"])
        e = res["groups"]["all"]["pairs"]["E>Erep"]
        for m in losses.METRICS:
            self.assertEqual((e[m]["mean"], e[m]["lo"], e[m]["hi"]), (0.0, 0.0, 0.0))
        self.assertIn(res["overall"]["T:a"], ("PASS", "FAIL", "INCONCLUSIVE"))
        self.assertIn("source=spread", res["groups"])
        self.assertIn("category=code", res["groups"])
        self.assertIn("length=8192", res["groups"])
        # E->T is smaller than E->R by construction
        g = res["groups"]["all"]["pairs"]
        self.assertLess(g["E>T:a"]["kl"]["mean"], g["E>R"]["kl"]["mean"])

    def test_not_reproducible_is_invalid(self):
        res = losses.analyze(make_layout(self.tmp, erep="perturbed"))
        self.assertEqual((res["status"], res["overall"]["T:a"]), ("INVALID", "INVALID"))
        self.assertIn("E not reproducible", res["invalid_reason"])

    def test_control_noise_floor(self):
        res = losses.analyze(make_layout(self.tmp, control=True))
        self.assertIn("control_noise_floor", res["nulls"])

    def test_lever_inactive_on_spread_route_is_not_measured(self):
        res = losses.analyze(make_layout(self.tmp, t_spread_equals_e=True, noise_t=0.01, noise_r=1.0, parents=8))
        self.assertIn("T:a", res["not_measured"])
        self.assertEqual(res["bar"]["T:a"]["source=spread"]["kl"]["verdict"], "NOT-MEASURED")
        self.assertNotEqual(res["overall"]["T:a"], "PASS")

    def test_prompt_ids_must_match(self):
        z = np.zeros((1, 8), dtype=np.float32)
        spool.write(self.tmp / "a.bin", INPUTS, [2], z)
        spool.write(self.tmp / "b.bin", [7] + INPUTS[1:], [2], z)
        case = {"case": "c", "format": "spool", "files": {"a": str(self.tmp / "a.bin"), "b": str(self.tmp / "b.bin")}}
        with self.assertRaisesRegex(ValueError, "different prompt ids"):
            losses.case_sums(case, [("a", "b")])


if __name__ == "__main__":
    unittest.main()
