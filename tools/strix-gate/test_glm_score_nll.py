import os
import tempfile
import unittest

import numpy as np

import glm_score_nll as g


class GlmScoreNll(unittest.TestCase):
    def write(self, d, name, logits):
        half = logits.shape[1] // 2
        logits[:, :half].astype("<f4").tofile(os.path.join(d, f"{name}-tf-r0.f32"))
        logits[:, half:].astype("<f4").tofile(os.path.join(d, f"{name}-tf-r1.f32"))

    def test_identical_and_perturbed(self):
        rng = np.random.default_rng(1)
        rows, vocab, n = 8, 16, 40
        toks = rng.integers(0, vocab, n).astype("<u4")
        ref = rng.normal(size=(rows, vocab))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "p.u32")
            toks.tofile(p)
            self.write(d, "X1", ref)
            self.write(d, "Y1", ref.copy())
            bumped = ref.copy()
            tgt = toks[n - rows + 1:].astype(int)
            bumped[np.arange(len(tgt)), tgt] += 1.0  # raises the target logit: test NLL must drop
            self.write(d, "Z1", bumped)
            same = g.compare(g.log_softmax("X1", d, rows, vocab), g.log_softmax("Y1", d, rows, vocab), toks, rows)
            self.assertEqual(same["offset1"]["dnll"], 0.0)
            self.assertAlmostEqual(same["kl_mean"], 0.0, places=12)
            self.assertEqual(same["top1"], 1.0)
            moved = g.compare(g.log_softmax("X1", d, rows, vocab), g.log_softmax("Z1", d, rows, vocab), toks, rows)
            self.assertLess(moved["offset1"]["dnll"], 0.0)
            self.assertGreater(moved["kl_mean"], 0.0)
            self.assertEqual(moved["offset1"]["rows"], rows - 1)

    def test_empty_truncated_unequal_and_trailing_shards(self):
        with tempfile.TemporaryDirectory() as d:
            for sizes in ((128, 0), (0, 0), (128, 64), (64, 64), (128, 124), (128, 129), (128, 132)):
                with self.subTest(sizes=sizes):
                    for rank, size in enumerate(sizes):
                        with open(os.path.join(d, f"X-tf-r{rank}.f32"), "wb") as f:
                            f.write(bytes(size))
                    with self.assertRaises(SystemExit):
                        g.log_softmax("X", d, 4, 16)

    def test_nonfinite_and_invalid_dimensions(self):
        with tempfile.TemporaryDirectory() as d:
            for rank in (0, 1):
                for value in (np.nan, np.inf, -np.inf):
                    logits = np.zeros((4, 16))
                    logits[0, rank * 8] = value
                    self.write(d, "X", logits)
                    with self.assertRaises(SystemExit):
                        g.log_softmax("X", d, 4, 16)
            for rows, vocab in ((0, 16), (-1, 16), (4, 0), (4, -2), (4, 15)):
                with self.assertRaises(SystemExit):
                    g.log_softmax("X", d, rows, vocab)

    def test_shape_and_target_validation(self):
        ref = np.zeros((4, 16))
        for candidate in (np.zeros((4, 8)), np.zeros((1, 16)), np.full((4, 16), np.nan)):
            with self.assertRaises(SystemExit):
                g.compare(ref, candidate, np.zeros(5, dtype="<u4"), 4)
        with self.assertRaises(SystemExit):
            g.compare(ref, ref, np.full(5, 16, dtype="<u4"), 4)

    def test_cli_requires_model_vocabulary(self):
        with tempfile.TemporaryDirectory() as d:
            prompt = os.path.join(d, "p.u32")
            np.zeros(5, dtype="<u4").tofile(prompt)
            self.write(d, "X", np.zeros((4, 16)))
            self.write(d, "Y", np.zeros((4, 16)))
            args = ["--dir", d, "--ref", "X", "--test", "Y", "--prompt", prompt, "--rows", "4", "--vocab-size", "16"]
            self.assertEqual(g.main(args), 0)
            # Both runs lose the same rank; identical surviving data must never pass.
            for name in ("X", "Y"):
                with open(os.path.join(d, f"{name}-tf-r1.f32"), "wb"):
                    pass
            with self.assertRaises(SystemExit):
                g.main(args)

    def test_short_prompt_refused(self):
        with self.assertRaises(SystemExit):
            g.compare(np.zeros((4, 2)), np.zeros((4, 2)), np.zeros(4, dtype="<u4"), 4)


if __name__ == "__main__":
    unittest.main()
