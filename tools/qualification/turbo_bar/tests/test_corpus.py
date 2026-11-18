import tempfile
import unittest
from pathlib import Path

from tools.qualification.turbo_bar import corpus
from tools.qualification.turbo_bar.common import read_ids, read_json
from tools.qualification.turbo_bar.tests.helpers import fake_tokenizer


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cmd = fake_tokenizer(self.tmp)
        src = self.tmp / "src"
        src.mkdir()
        for i in range(60):
            (src / f"f{i:02d}.txt").write_text(" ".join(f"w{i}_{j}" for j in range(500)), encoding="utf-8")
        self.code = [str(src / "f0*.txt"), str(src / "f1*.txt")]
        self.prose = [str(src / "f[2-5]*.txt")]

    def build(self, name, salt="s1"):
        return corpus.build(self.tmp / name, salt, self.cmd, self.code, self.prose, lengths=(600, 1200), per_category=4,
                            search_per_category=1)

    def test_positions_and_cases(self):
        doc, sha = self.build("a")
        self.assertEqual(len(doc["parents"]), 16)
        self.assertEqual(len(doc["cases"]), 32)
        for c in doc["cases"]:
            pos = [int(x) for x in (self.tmp / "a" / c["positions"]).read_text().split()]
            self.assertEqual(len(pos), 128)
            self.assertEqual(pos, sorted(set(pos)))
            self.assertTrue(64 <= pos[0] and pos[-1] < c["length"] - 1)
            self.assertEqual(len(read_ids(self.tmp / "a" / c["prompt_ids"])), c["length"])
            self.assertEqual(len(read_ids(self.tmp / "a" / c["cont_ids"])), 256)
        self.assertEqual(read_json(self.tmp / "a" / "corpus.json")["salt"], "s1")

    def test_deterministic_and_salt_dependent(self):
        _, s1 = self.build("a")
        _, s2 = self.build("b")
        self.assertEqual(s1, s2)
        self.assertEqual(corpus.spread_positions("s", "p", 8192), corpus.spread_positions("s", "p", 8192))
        self.assertNotEqual(corpus.spread_positions("s", "p", 8192), corpus.spread_positions("t", "p", 8192))

    def test_splits_and_distinct_sources(self):
        doc, _ = self.build("a")
        for cat in corpus.CATEGORIES:
            ps = [p for p in doc["parents"] if p["category"] == cat]
            self.assertEqual(sorted(p["split"] for p in ps), ["CONFIRM"] * 3 + ["SEARCH"])
        used = [s["sha256"] for p in doc["parents"] for s in p["sources"]]
        self.assertEqual(len(used), len(set(used)))
        self.assertTrue(all(p["sources"] for p in doc["parents"] if p["category"] in ("code", "prose")))


if __name__ == "__main__":
    unittest.main()
