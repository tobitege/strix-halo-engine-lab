import re
import tempfile
import unittest
from pathlib import Path

from tools.qualification.turbo_bar import retrieval, tasks
from tools.qualification.turbo_bar.tests.helpers import fake_tokenizer


class RetrievalTests(unittest.TestCase):
    def test_c8_restored_estimated(self):
        items = retrieval.make_case(0, 2048, "restored", 8, "straddle", 512)
        self.assertEqual(len(items), 8)
        self.assertEqual(len({x["id"] for x in items}), 8)
        self.assertIsNotNone(items[0]["warmup"])
        hay = {x["messages"][0]["content"].split("\n\nQuestion:")[0] for x in items}
        self.assertEqual(len(hay), 1)               # same haystack, different keys
        keys = set()
        for x in items:
            text = x["messages"][0]["content"]
            key = re.search(r"archive code of (.+?)\? Answer", text).group(1)
            keys.add(key)
            self.assertIn(f"The archive code of {key} is {x['check']['answer']}.", text)
            self.assertTrue(tasks.score_item(x, x["check"]["answer"]))
        self.assertEqual(len(keys), 8)

    def test_straddle_and_mid_measured(self):
        cmd = fake_tokenizer(tempfile.mkdtemp())
        for cut in ("straddle", "mid"):
            it = retrieval.make_case(1, 2048, "new", 1, cut, 512, cmd=cmd)[0]
            m = it["meta"]
            self.assertTrue(m["measured"])
            self.assertTrue(m["cut_ok"], (cut, m))
            self.assertEqual(m["crosses_boundary"], cut == "straddle")
            self.assertIsNone(it["warmup"])

    def test_build_matrix(self):
        items = retrieval.build([(2048, 8), (4096, 4)], 512)
        self.assertEqual(len({(x["mode"], x["concurrency"], x["cut"]) for x in items if x["length"] == 2048}), 8)
        self.assertEqual(sum(1 for x in items if x["primary"]), 12)

    def test_single_interior_boundary(self):
        # length = 2 chunks has exactly one interior boundary (the 8K / 4096 pilot case).
        items = retrieval.build([(1024, 2)], 512)
        self.assertTrue(items)
        self.assertTrue(all(x["length"] == 1024 for x in items))


if __name__ == "__main__":
    unittest.main()
