import json
import tempfile
import unittest
from pathlib import Path

from tools.qualification.turbo_bar import manifest as mf
from tools.qualification.turbo_bar.common import read_jsonl, sha256_file, write_json
from tools.qualification.turbo_bar.tests.helpers import frozen_manifest


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path = frozen_manifest(self.tmp)

    def t_arm(self, name="a"):
        return mf.make_arm("T", name, f"id-{name}", [mf.file_record(self.tmp / "w_e.gguf")], mf.file_record(self.tmp / "bin"),
                           arithmetic="turbo")

    def test_freeze_stamps_and_refuses_edits(self):
        m = mf.load(self.path)
        self.assertEqual(len(m["frozen"]["sha256"]), 64)
        with self.assertRaises(mf.ManifestError):
            mf.add_arm(self.path, mf.make_arm("R", "R", "other", m["arms"]["R"]["weights"], m["arms"]["R"]["binary"]))
        with self.assertRaises(mf.ManifestError):
            mf.freeze(self.path)
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        raw["arms"]["R"]["identity"] = "forgiving"
        write_json(self.path, raw)
        with self.assertRaises(mf.ManifestError):
            mf.load(self.path)

    def test_freeze_needs_protocol_and_shas(self):
        p = self.tmp / "x.json"
        mf.save(p, mf.new("toy"))
        with self.assertRaises(mf.ManifestError):
            mf.freeze(p)

    def test_t_variants_append_only_with_ledger(self):
        mf.add_arm(self.path, self.t_arm("a"))
        mf.add_arm(self.path, self.t_arm("b"))
        with self.assertRaises(mf.ManifestError):
            mf.add_arm(self.path, self.t_arm("a"))
        events = [(e["event"], e.get("arm")) for e in read_jsonl(mf.ledger_path(self.path))]
        self.assertIn(("add-arm", "T:a"), events)
        self.assertIn(("add-arm", "T:b"), events)
        self.assertIn(("freeze", None), events)
        self.assertEqual(mf.core_sha(mf.load(self.path)), mf.load(self.path)["frozen"]["core_sha256"])

    def test_new_version_supersedes(self):
        mf.add_arm(self.path, self.t_arm("a"))
        with self.assertRaises(mf.ManifestError):
            mf.freeze(self.path, new_version=True)
        m = mf.freeze(self.path, new_version=True, reason="R was too weak")
        self.assertEqual((m["version"], m["frozen"]), (2, None))
        self.assertEqual([s["arm"]["name"] for s in m["superseded_arms"]], ["a"])
        self.assertNotIn("T:a", m["arms"])
        self.assertTrue(any(e["event"] == "superseded" for e in read_jsonl(mf.ledger_path(self.path))))
        mf.add_arm(self.path, mf.make_arm("R", "R", "new R", m["arms"]["R"]["weights"], m["arms"]["R"]["binary"]))
        self.assertEqual(mf.freeze(self.path)["version"], 2)

    def test_check_run(self):
        m = mf.load(self.path)
        rd = self.tmp / "run"
        rd.mkdir()
        good = {"manifest_core_sha256": m["frozen"]["core_sha256"], "manifest_version": 1,
                "arms": {n: {"identity": m["arms"][n]["identity"], "binary_sha256": m["arms"][n]["binary"]["sha256"],
                             "weights": [{"sha256": m["arms"][n]["weights"][0]["sha256"]}], "args": [], "env": {}}
                         for n in ("E", "R")}}
        write_json(rd / "run.json", good)
        self.assertEqual(mf.check_run(m, rd), [])
        good["arms"]["R"]["weights"][0]["sha256"] = "0" * 64
        write_json(rd / "run.json", good)
        self.assertTrue(any("weights" in p for p in mf.check_run(m, rd)))

    def test_cli_sha_from(self):
        p = self.tmp / "cli.json"
        w = self.tmp / "w_e.gguf"
        sums = self.tmp / "sums.txt"
        sums.write_text(f"{'9' * 64}  {w}\n{'8' * 64} *bin\n", encoding="utf-8")
        self.assertEqual(mf.main(["init", "--manifest", str(p), "--model", "toy"]), 0)
        self.assertEqual(mf.main(["add-arm", "--manifest", str(p), "--role", "E", "--identity", "x", "--weight", str(w),
                                  "--binary", str(self.tmp / "bin"), "--sha-from", str(sums)]), 0)
        arm = mf.load(p)["arms"]["E"]
        self.assertEqual((arm["weights"][0]["sha256"], arm["binary"]["sha256"]), ("9" * 64, "8" * 64))
        self.assertNotEqual(arm["weights"][0]["sha256"], sha256_file(w))


if __name__ == "__main__":
    unittest.main()
