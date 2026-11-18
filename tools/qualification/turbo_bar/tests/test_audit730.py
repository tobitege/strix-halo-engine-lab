import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.qualification.turbo_bar import losses, manifest as mf, run_logits, spool
from tools.qualification.turbo_bar.common import canon_sha, read_json, safe_name, sha256_file, write_json
from tools.qualification.turbo_bar.tests.helpers import INPUTS, POSITIONS, V


class FrozenLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        case = dict(case="p0", parent="p0", category="math", length=len(INPUTS), split="CONFIRM",
                    prompt_ids="prompt.ids", positions="positions.ids")
        for key, ids in (("prompt_ids", INPUTS), ("positions", POSITIONS)):
            path = self.root / case[key]
            path.write_text(" ".join(map(str, ids)))
            case[key + "_sha256"] = sha256_file(path)
        corpus = dict(cases=[case], n_positions=len(POSITIONS))
        write_json(self.root / "corpus.json", corpus)
        arm = dict(identity="test", binary=dict(sha256="binary"), weights=[dict(sha256="weights")])
        m = dict(model="toy", version=1, arms={a: copy.deepcopy(arm) for a in ("E", "R", "T:a")},
                 protocol=dict(corpus_sha256=canon_sha(corpus), margins={}, bootstrap=dict(B=20, seed=0)))
        m["frozen"] = dict(core_sha256=mf.core_sha(m))
        receipt = dict(teacher_forced_tiled_lm_head=True, teacher_logits_scope="selected", teacher_logits_rows=4)
        for a in ("E", "R", "T:a", "Erep"):
            path = self.root / f"p0.{safe_name(a)}.bin"
            spool.write(path, INPUTS, POSITIONS, np.ones((len(POSITIONS), V), dtype=np.float32))
            write_json(path.with_suffix(".json"), dict(
                contract=run_logits.output_contract(m, a, case, self.root, "spread", 2048),
                complete=True, rc=0, receipt=receipt, receipt_sha256=canon_sha(receipt), payload_sha256=sha256_file(path)))
        cases = losses.build_layout(self.root, self.root, ["E", "R", "T:a", "Erep"], manifest=m)
        self.layout = losses.freeze_layout(dict(E="E", R="R", T=["T:a"], Erep="Erep", control=None,
            cases=cases, margins=m["protocol"]["margins"], bootstrap=m["protocol"]["bootstrap"],
            arm_sha256={"T:a": canon_sha(m["arms"]["T:a"])},
            manifest=dict(version=1, core_sha256=m["frozen"]["core_sha256"])), m, corpus)
        # Exercise the saved JSON path, not just an in-memory object.
        write_json(self.root / "layout.json", self.layout)
        self.layout = read_json(self.root / "layout.json")

    def test_unchanged_saved_layout_is_valid(self):
        self.assertEqual(losses.analyze(self.layout)["status"], "VALID")

    def test_replacement_payload_even_with_matching_new_sidecar_is_rejected(self):
        path = self.root / "p0.T_a.bin"
        spool.write(path, INPUTS, POSITIONS, np.zeros((len(POSITIONS), V), dtype=np.float32))
        side = read_json(path.with_suffix(".json"))
        side["payload_sha256"] = sha256_file(path)
        write_json(path.with_suffix(".json"), side)
        with self.assertRaisesRegex(ValueError, "replaced"):
            losses.analyze(self.layout)

    def test_changed_contract_or_receipt_is_rejected(self):
        path = self.root / "p0.T_a.json"
        original = read_json(path)
        for field in ("contract", "receipt"):
            side = copy.deepcopy(original)
            side[field]["identity"] = "replacement"
            side["receipt_sha256"] = canon_sha(side["receipt"])
            write_json(path, side)
            with self.assertRaises(ValueError):
                losses.analyze(self.layout)

    def test_frozen_inventory_protocol_and_legacy_layout_are_rejected(self):
        for change in ("drop_case", "positions", "bootstrap", "margins", "binding", "arms"):
            layout = copy.deepcopy(self.layout)
            if change == "drop_case":
                layout["cases"].clear()
            elif change == "positions":
                layout["cases"][0]["expected_positions"] = [1]
            elif change == "bootstrap":
                layout["bootstrap"]["seed"] += 1
            elif change == "margins":
                layout["margins"] = {"kl": 100}
            elif change == "arms":
                layout["arm_sha256"]["T:a"] = "stale"
            else:
                del layout["frozen_inputs"]
            with self.subTest(change=change), self.assertRaises(ValueError):
                losses.analyze(layout)


if __name__ == "__main__":
    unittest.main()
