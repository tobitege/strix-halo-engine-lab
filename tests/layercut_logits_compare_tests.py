"""Host fixtures for complete, rank-local teacher-forced vocabulary rows."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "logits_compare", Path(__file__).resolve().parents[1] / "scripts/bench/layercut/logits_compare.py")
comparator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparator)


class LogitsRows(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.directory.cleanup)
        self.prefix = str(Path(self.directory.name) / "arm")

    def dumps(self, widths, retained):
        for rank, (width, rows) in enumerate(zip(widths, retained)):
            np.arange(width, dtype=np.float32).tofile(f"{self.prefix}-logits-r{rank}.f32")
            np.arange(width * rows, dtype=np.float32).tofile(f"{self.prefix}-tf-r{rank}.f32")

    def test_short_tail_is_not_two_half_vocabularies(self):
        # 2049 prompt tokens / block 2048 retains only one row on each arm.
        self.dumps((8, 6), (1, 1))
        with self.assertRaisesRegex(ValueError, "requested 2.*retained 1"):
            comparator.load(self.prefix, "tf", 2)

    def test_complete_rows_preserve_rank_vocabularies(self):
        self.dumps((8, 6), (2, 2))
        actual = comparator.load(self.prefix, "tf", 2)
        expected = np.concatenate((np.arange(16).reshape(2, 8), np.arange(12).reshape(2, 6)), axis=1)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(comparator.load(self.prefix, "logits", 0).shape, (1, 14))

    def test_incomplete_vocabulary_row(self):
        self.dumps((8, 6), (2, 2))
        np.arange(10, dtype=np.float32).tofile(f"{self.prefix}-tf-r1.f32")
        with self.assertRaisesRegex(ValueError, "rank 1: incomplete vocabulary rows"):
            comparator.load(self.prefix, "tf", 2)

    def test_each_rank_must_retain_requested_rows(self):
        self.dumps((8, 6), (2, 1))
        with self.assertRaisesRegex(ValueError, "rank 1: requested 2.*retained 1"):
            comparator.load(self.prefix, "tf", 2)

    def test_missing_vocabulary_width(self):
        self.dumps((0, 6), (2, 2))
        with self.assertRaisesRegex(ValueError, "rank 0: incomplete vocabulary rows"):
            comparator.load(self.prefix, "tf", 2)


if __name__ == "__main__":
    unittest.main()
