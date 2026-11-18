import unittest

import numpy as np

from tools.qualification.turbo_bar import stats


class StatsTests(unittest.TestCase):
    def test_pass_fail_inconclusive(self):
        rng = np.random.default_rng(0)
        self.assertEqual(stats.verdict(-1 + 0.1 * rng.normal(size=30), 0.0, B=500)["verdict"], "PASS")
        self.assertEqual(stats.verdict(1 + 0.1 * rng.normal(size=30), 0.0, B=500)["verdict"], "FAIL")
        v = stats.verdict(0.05 + rng.normal(size=12), 0.0, B=500)
        self.assertEqual(v["verdict"], "INCONCLUSIVE")
        self.assertEqual(v["message"], "enlarge sample")
        self.assertGreater(v["n_needed"], 12)

    def test_never_pass_when_ci_crosses(self):
        rng = np.random.default_rng(5)
        for _ in range(40):
            d = rng.normal(loc=rng.uniform(-1, 1), size=int(rng.integers(3, 20)))
            v = stats.verdict(d, 0.0, B=200)
            if v["lo"] <= 0 <= v["hi"]:
                self.assertEqual(v["verdict"], "INCONCLUSIVE")
            if v["verdict"] == "PASS":
                self.assertLess(v["hi"], 0)

    def test_small_n_and_unbounded(self):
        self.assertEqual(stats.verdict([-5.0, -5.0], 0.0)["verdict"], "INCONCLUSIVE")
        v = stats.verdict([-1, 1] * 6, 0.0, B=200)
        self.assertEqual((v["verdict"], v["n_needed"]), ("INCONCLUSIVE", "unbounded"))

    def test_ci_is_union_of_t_and_bootstrap(self):
        d = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
        _m, lo, hi = stats.ci(d, B=2000, seed=1)
        _mt, lt, ht = stats.t_ci(d)
        self.assertLessEqual(lo, lt + 1e-12)
        self.assertGreaterEqual(hi, ht - 1e-12)

    def test_paired_binary_counts(self):
        a = [1] * 20 + [0] * 5 + [1] * 5
        b = [1] * 20 + [1] * 5 + [0] * 5
        r = stats.paired_binary(a, b, B=200)
        self.assertEqual((r["a_only"], r["b_only"], r["both"], r["neither"]), (5, 5, 20, 0))

    def test_binary_zero_discordance(self):
        for outcomes in ([1] * 128, [0] * 128, [0, 1] * 64):
            r = stats.paired_binary(outcomes, outcomes, limit=0.01)
            self.assertEqual(r["verdict"], "INCONCLUSIVE")
            self.assertGreater(r["hi"], 0.02)
            self.assertLess(r["lo"], -0.02)
        self.assertEqual(stats.paired_binary([1] * 1024, [1] * 1024, limit=0.01)["verdict"], "PASS")

    def test_binary_bounds_and_direction(self):
        r = stats.paired_binary([1] * 128, [0] * 128, limit=0.01)
        self.assertEqual(r["verdict"], "FAIL")
        reverse = stats.paired_binary([0] * 128, [1] * 128, limit=0.01)
        self.assertEqual(reverse["verdict"], "PASS")
        self.assertAlmostEqual(r["hi"], -reverse["lo"])
        self.assertAlmostEqual(r["lo"], -reverse["hi"])
        with self.assertRaises(ValueError):
            stats.paired_binary([1, 0, 0.5], [1, 1, 1])

    def test_binary_ceiling_false_pass_probability(self):
        # Exact enumeration under a 2% loss: the chance of falsely establishing
        # the 1% margin must not exceed the declared 5% error budget.
        import math
        n, p = 128, 0.02
        false_pass = 0.0
        for failures in range(n + 1):
            result = stats.paired_binary([1] * n, [0] * failures + [1] * (n - failures), limit=0.01)
            if result["verdict"] == "PASS":
                false_pass += math.comb(n, failures) * p ** failures * (1 - p) ** (n - failures)
        self.assertLessEqual(false_pass, 0.05)


if __name__ == "__main__":
    unittest.main()
