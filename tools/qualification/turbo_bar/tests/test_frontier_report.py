import tempfile
import unittest
from pathlib import Path

from tools.qualification.turbo_bar import frontier, losses, report, run_logits
from tools.qualification.turbo_bar import manifest as mf
from tools.qualification.turbo_bar.common import canon_sha, http_binding, read_json, sha256_file, suite_record, write_json, write_jsonl
from tools.qualification.turbo_bar.tests.helpers import INPUTS, POSITIONS, frozen_manifest, make_layout


def V(name, speeds, kl, dnll, verdict="PASS"):
    return {"name": name, "speed": speeds, "quality": {"kl_mean": kl, "dnll_mean": dnll, "verdict": verdict}}


class FrontierTests(unittest.TestCase):
    def test_selection_within_timing_noise(self):
        res = frontier.analyze([
            V("fast", [1.30, 1.32, 1.28, 1.30], 0.010, 0.002),
            V("near", [1.29, 1.31, 1.30, 1.28], 0.002, 0.001),
            V("slow", [1.10, 1.11, 1.09, 1.10], 0.0005, 0.0),
            V("bad", [1.60, 1.61, 1.59, 1.60], 0.5, 0.2, "FAIL")])
        by = {r["name"]: r for r in res["variants"]}
        self.assertEqual(res["fastest_pass"], "fast")
        self.assertTrue(by["near"]["within_noise"] and by["fast"]["within_noise"])
        self.assertFalse(by["slow"]["within_noise"])
        self.assertEqual(res["selected"], "near")
        self.assertTrue(by["bad"]["pareto"] and by["fast"]["pareto"] and by["slow"]["pareto"])
        self.assertTrue(by["near"]["pareto"])
        self.assertIn("SELECTED", frontier.table(res))

    def test_no_pass_selects_nothing(self):
        self.assertIsNone(frontier.analyze([V("a", [1.0, 1.1, 1.2], 0.1, 0.1, "INCONCLUSIVE")])["selected"])


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mpath = frozen_manifest(self.tmp)
        self.m = mf.load(self.mpath)
        items = [{"id": f"{kind}-{split}-{i:03d}", "kind": kind, "split": split}
                 for kind in ("code", "math", "needle", "json") for split in ("explore", "confirm") for i in range(60)]
        self.m["frozen"] = None
        self.m["protocol"]["suites"]["tasks"] = suite_record(items)
        for a in self.m["arms"].values():
            a["effective_route"] = {"profile": "test"}
        self.m["arms"]["T:a"] = dict(self.m["arms"]["E"], role="T", name="a")
        layout = make_layout(self.tmp, noise_r=1.0, noise_t=0.01, parents=6)
        corpus = {"cases": [], "n_positions": len(POSITIONS)}
        for c in layout["cases"]:
            if c["source"] != "spread":
                continue
            original = {k: c[k] for k in ("parent", "category", "length", "split")}
            original["case"] = c["parent"]
            for key, ids in (("prompt_ids", INPUTS), ("positions", POSITIONS),
                             ("cont_ids", [INPUTS[i % len(INPUTS)] for i in range(256)])):
                path = self.tmp / f"{c['parent']}.{key}.ids"
                path.write_text(" ".join(map(str, ids)), encoding="utf-8")
                original[key] = path.name
                original[key + "_sha256"] = sha256_file(path)
            corpus["cases"].append(original)
        self.m["protocol"]["corpus_sha256"] = canon_sha(corpus)
        self.m["protocol"]["bootstrap"] = layout["bootstrap"]
        layout["margins"] = self.m["protocol"]["margins"]
        mf.save(self.mpath, self.m)
        self.m = mf.freeze(self.mpath)
        layout["manifest"] = {"version": 1, "core_sha256": self.m["frozen"]["core_sha256"]}
        layout["arm_sha256"] = {"T:a": canon_sha(self.m["arms"]["T:a"])}
        originals = {c["case"]: c for c in corpus["cases"]}
        for c in layout["cases"]:
            c["validated_files"] = {}
            receipt = dict(teacher_forced_tiled_lm_head=True, teacher_logits_scope="selected",
                           teacher_logits_rows=len(c["expected_positions"]))
            for arm, path in c["files"].items():
                contract = run_logits.output_contract(self.m, arm, originals[c["parent"]], self.tmp,
                                                      c["source"], 2048 if c["source"] == "spread" else None)
                sidecar = Path(path).with_suffix(".json")
                payload_sha = sha256_file(path)
                write_json(sidecar, dict(contract=contract, complete=True, rc=0, receipt=receipt,
                                        receipt_sha256=canon_sha(receipt), payload_sha256=payload_sha))
                c["validated_files"][arm] = dict(contract=contract, payload_sha256=payload_sha,
                                                 sidecar_sha256=sha256_file(sidecar))
        losses.freeze_layout(layout, self.m, corpus)
        self.losses = losses.analyze(layout)
        write_json(self.tmp / "losses.json", self.losses)
        td = self.tmp / "tasks"
        td.mkdir()
        for arm, fails in (("E", 0), ("R", 20), ("T_a", 2)):
            name = "T:a" if arm == "T_a" else arm
            binding = http_binding(self.m, name, "tasks")
            rows = [dict(rec, id=key, arm=name, ok=int(int(key[-3:]) >= fails), binding=binding,
                         server_identity=binding["server_identity"], measured=True)
                    for key, rec in self.m["protocol"]["suites"]["tasks"]["items"].items()]
            write_jsonl(td / f"scored.{arm}.jsonl", rows)
        write_json(self.tmp / "front.json", [{"name": "a", "speed": [1.2, 1.25, 1.22]}])

    def test_report_writes_receipt_and_markdown(self):
        out = self.tmp / "out"
        rc = report.main(["--manifest", str(self.mpath), "--losses", str(self.tmp / "losses.json"),
                          "--tasks", str(self.tmp / "tasks"), "--frontier", str(self.tmp / "front.json"), "--out", str(out)])
        self.assertEqual(rc, 0)
        r = read_json(out / "receipt.json")
        md = (out / "REPORT.md").read_text(encoding="utf-8")
        self.assertEqual(r["status"], "VALID")
        self.assertIn("E>R", md)
        self.assertIn("Null controls", md)
        tk = r["tasks"]["T:a"]["per_kind"]["math"]["confirm"]
        self.assertEqual(tk["n"], 60)
        self.assertEqual(tk["T_vs_R_loss"]["b_only"], 18)    # T solved where R failed
        self.assertIn(r["overall"]["T:a"]["verdict"], ("PASS", "FAIL", "INCONCLUSIVE"))
        self.assertEqual(r["overall"]["T:a"]["components"]["retrieval"], "INCONCLUSIVE")  # missing -> never PASS
        self.assertNotEqual(r["overall"]["T:a"]["verdict"], "PASS")
        self.assertIsNotNone(r["frontier"])

    def test_mixed_manifest_version_is_invalid(self):
        bad = dict(self.losses, manifest={"version": 2, "core_sha256": "x"})
        write_json(self.tmp / "bad.json", bad)
        r = report.build(self.mpath, self.tmp / "bad.json")
        self.assertEqual(r["status"], "INVALID")
        self.assertTrue(all(v["verdict"] == "INVALID" for v in r["overall"].values()))

    def test_frontier_overwrites_stale_quality(self):
        write_json(self.tmp / "front.json", [{"name": "a", "speed": [1.2, 1.25, 1.22],
                                               "quality": {"kl_mean": -999, "dnll_mean": -999}}])
        r = report.build(self.mpath, self.tmp / "losses.json", frontier_path=self.tmp / "front.json")
        v = r["frontier"]["variants"][0]
        self.assertEqual(v["kl_mean"], self.losses["summary"]["T:a"]["kl_mean"])
        self.assertEqual(v["dnll_mean"], self.losses["summary"]["T:a"]["dnll_mean"])

    def test_unknown_losses_arm_is_invalid(self):
        changed = dict(self.losses, overall={"T:old": "PASS"})
        write_json(self.tmp / "old.json", changed)
        r = report.build(self.mpath, self.tmp / "old.json")
        self.assertEqual(r["status"], "INVALID")
        self.assertIn("unknown or superseded", r["invalid_reason"])
        self.assertTrue(all(v["verdict"] == "INVALID" for v in r["overall"].values()))


if __name__ == "__main__":
    unittest.main()
