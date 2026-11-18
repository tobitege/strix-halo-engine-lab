import copy
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.qualification.turbo_bar import client, losses, report, run_logits, spool, tasks
from tools.qualification.turbo_bar.common import (canon_sha, http_binding, read_json, read_jsonl, safe_name,
                                                  sha256_file, suite_record, write_json, write_jsonl)
from tools.qualification.turbo_bar.tests.helpers import INPUTS, POSITIONS, V, make_layout


def http_fixture(items, suite="tasks"):
    spec = {"identity": "test", "binary": {"sha256": "binary"}, "weights": [{"sha256": "weights"}],
            "effective_route": {"profile": "test"}}
    return {"version": 1, "frozen": {"core_sha256": "core"},
            "arms": {a: copy.deepcopy(spec) for a in ("E", "R", "T:a")},
            "protocol": {"contexts": [8192], "seeds": [1234], "suites": {suite: suite_record(items)}}}


def task_items(per_split=3):
    return [{"id": f"{k}-{s}-{i}", "kind": k, "split": s, "messages": [], "check": {}, "max_tokens": 1}
            for k in ("math", "code", "needle", "json") for s in ("explore", "confirm") for i in range(per_split)]


def scored(m, arm, suite="tasks"):
    binding = http_binding(m, arm, suite)
    return [dict(v, id=k, arm=arm, ok=int(arm != "R"), binding=copy.deepcopy(binding),
                 server_identity=copy.deepcopy(binding["server_identity"]), measured=True)
            for k, v in m["protocol"]["suites"][suite]["items"].items()]


class AuditRegressions(unittest.TestCase):
    def setUp(self):
        self.tmpctx = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpctx.cleanup)
        self.tmp = Path(self.tmpctx.name)

    def test_code_reply_never_runs_without_sandbox(self):
        marker = self.tmp / "outside-sandbox"
        reply = f"open({str(marker)!r}, 'w').write('unsafe')"
        with patch.object(subprocess, "run") as spawn:
            with self.assertRaisesRegex(tasks.ScoringUnavailable, "sandbox"):
                tasks.score_code(reply, {"tests": "assert True"})
            spawn.assert_not_called()
        self.assertFalse(marker.exists())

    def test_early_zero_exit_is_not_a_test_pass(self):
        with self.assertRaises(tasks.ScoringUnavailable):
            tasks.score_code("import sys; sys.exit(0)", {"entry": "f", "tests": "assert f() == 1"})

    def test_resume_preserves_old_run_record(self):
        old = {"manifest_core_sha256": "v1", "manifest_version": 1, "arms": {}}
        write_json(self.tmp / "run.json", old)
        before = (self.tmp / "run.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "old run.json preserved"):
            run_logits.preserve_run(self.tmp, dict(old, manifest_core_sha256="v2", manifest_version=2))
        self.assertEqual((self.tmp / "run.json").read_bytes(), before)

    def test_reuse_checks_every_receipt_binding_and_payload(self):
        path = self.tmp / "case.E.bin"
        spool.write(path, INPUTS, POSITIONS, np.zeros((len(POSITIONS), V), np.float32))
        contract = {"source": "spread", "manifest_core_sha256": "core", "manifest_version": 1,
                    "arm": "E", "arm_sha256": "spec", "identity": "id", "binary_sha256": "bin",
                    "weights_sha256": ["w"], "prompt_ids_sha256": "prompt", "positions_sha256": "selection"}
        route = {"teacher_forced_tiled_lm_head": "rows8-r1-reduction-tree",
                 "teacher_logits_scope": "selected", "teacher_logits_rows": "4"}
        side = {"contract": contract, "complete": True, "rc": 0, "receipt": route,
                "receipt_sha256": canon_sha(route), "payload_sha256": sha256_file(path)}
        write_json(path.with_suffix(".json"), side)
        self.assertEqual(run_logits.validate_output(path, contract)["receipt"], route)
        for key in contract:
            with self.subTest(key=key):
                expected = dict(contract, **{key: "changed"})
                with self.assertRaisesRegex(ValueError, "does not match"):
                    run_logits.validate_output(path, expected)
        changed = copy.deepcopy(side)
        changed["receipt"]["teacher_logits_rows"] = "3"
        write_json(path.with_suffix(".json"), changed)
        with self.assertRaisesRegex(ValueError, "route/payload"):
            run_logits.validate_output(path, contract)
        write_json(path.with_suffix(".json"), side)
        path.write_bytes(path.read_bytes() + b"extra")
        with self.assertRaisesRegex(ValueError, "route/payload"):
            run_logits.validate_output(path, contract)

    def test_scored_binding_and_exact_coverage(self):
        m = http_fixture(task_items())
        path = self.tmp / "scored.T_a.jsonl"
        rows = scored(m, "T:a")
        write_jsonl(path, rows)
        self.assertEqual(len(report._load_scored(self.tmp, "T:a", m, "tasks")), len(rows))
        variants = [rows[:-1], rows + [rows[0]]]
        for field, value in (("binding", {}), ("item_sha256", "stale-check"), ("server_identity", {}),
                             ("split", "explore"), ("arm", "R")):
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            variants.append(changed)
        for changed in variants:
            with self.subTest(change=changed[0]):
                write_jsonl(path, changed)
                with self.assertRaises(ValueError):
                    report._load_scored(self.tmp, "T:a", m, "tasks")
        write_jsonl(path, rows)
        m["arms"]["T:a"]["effective_route"] = {"profile": "changed"}
        with self.assertRaisesRegex(ValueError, "binding"):
            report._load_scored(self.tmp, "T:a", m, "tasks")

    def test_task_pairing_cannot_intersect_away_failures(self):
        # 8 paired items per split: the exact conservative binary bound (stats.binary_ci) needs n >= 7 to show a
        # uniform T win at margin 0 (1 - (0.05/4)**(1/n) <= 0.5); 3 items stay INCONCLUSIVE by design.
        m = http_fixture(task_items(8))
        for arm in ("E", "R", "T:a"):
            write_jsonl(self.tmp / f"scored.{safe_name(arm)}.jsonl", scored(m, arm))
        res = report.task_results(self.tmp, m, ["T:a"], 0, {"B": 100, "seed": 1})
        self.assertEqual(res["T:a"]["verdict"], "PASS")
        path = self.tmp / "scored.T_a.jsonl"
        write_jsonl(path, [r for r in read_jsonl(path) if r["kind"] == "math" and r["split"] == "confirm"])
        with self.assertRaisesRegex(ValueError, "coverage"):
            report.task_results(self.tmp, m, ["T:a"], 0, {"B": 100, "seed": 1})

    def test_client_checks_items_and_actual_server_identity(self):
        items = [{"id": "x", "kind": "math", "split": "confirm", "check": {"answer": 1},
                  "messages": [], "max_tokens": 1}]
        m = http_fixture(items)
        binding = http_binding(m, "E", "tasks")
        raw = {"id": "x", "arm": "E", "binding": binding, "item_sha256": canon_sha(items[0]),
               "warmup": False, "reply": "ANSWER: 1", "strix": {"qualification_identity": binding["server_identity"]}}
        path = self.tmp / "results.E.jsonl"
        write_jsonl(path, [raw])
        client.score(items, self.tmp, manifest=m)
        self.assertEqual(read_jsonl(self.tmp / "scored.E.jsonl")[0]["ok"], 1)
        for stale in (dict(raw, binding={}), dict(raw, strix={}), dict(raw, item_sha256="old")):
            write_jsonl(path, [stale])
            with self.assertRaises(ValueError):
                client.score(items, self.tmp, manifest=m)
        with self.assertRaisesRegex(ValueError, "items/checks"):
            client.score([dict(items[0], check={"answer": 2})], self.tmp, manifest=m)

    def test_missing_or_short_continuation_cannot_pass(self):
        layout = make_layout(self.tmp, t_spread_equals_e=True)
        layout["margins"] = {"kl": 10, "top1_disagreement": 10, "dnll": 10}
        layout["cases"] = [c for c in layout["cases"] if c["source"] == "spread"]
        self.assertEqual(losses.analyze(layout)["overall"]["T:a"], "INCONCLUSIVE")
        full = make_layout(self.tmp)
        full["margins"] = layout["margins"]
        for c in full["cases"]:
            if c["source"] == "continuation":
                for f in c["files"].values():
                    p = Path(f)
                    data = p.read_bytes()
                    p.write_bytes(data[:-(len(data) // 256)])
        result = losses.analyze(full)
        self.assertFalse(result["complete_sources"])
        self.assertEqual(result["overall"]["T:a"], "INCONCLUSIVE")

    def test_wrong_spread_positions_cannot_pass(self):
        layout = make_layout(self.tmp)
        layout["margins"] = {"kl": 10, "top1_disagreement": 10, "dnll": 10}
        layout["cases"][0]["expected_positions"] = [1, 2, 3, 4]
        self.assertEqual(losses.analyze(layout)["overall"]["T:a"], "INCONCLUSIVE")

    def test_search_parents_do_not_change_confirm_verdict(self):
        layout = make_layout(self.tmp, parents=6)
        for c in layout["cases"]:
            if c["parent"] in ("p0", "p1", "p2"):
                c["split"] = "SEARCH"
        heldout = copy.deepcopy(layout)
        heldout["cases"] = [c for c in heldout["cases"] if c["split"] == "CONFIRM"]
        both, confirm = losses.analyze(layout), losses.analyze(heldout)
        self.assertEqual(both["groups"]["all"], confirm["groups"]["all"])
        self.assertEqual(both["bar"]["T:a"]["all"], confirm["bar"]["T:a"]["all"])
        self.assertEqual(both["groups"]["search"]["n_parents"], 3)

    def test_signed_zero_is_not_bitwise_repeat_or_inactive(self):
        layout = make_layout(self.tmp)
        case = layout["cases"][0]
        for arm, sign in (("E", 0.0), ("Erep", -0.0)):
            z = np.stack([r[3] for r in spool.read(case["files"][arm])])
            z[0, 0] = sign
            spool.write(case["files"][arm], INPUTS, POSITIONS, z)
        res = losses.analyze(layout)
        self.assertFalse(res["nulls"]["e_repeat_bitwise_identical"])
        self.assertEqual(res["status"], "INVALID")
        self.assertFalse(losses.bitwise_equal(np.array([0.0], np.float32), np.array([-0.0], np.float32)))

    def test_restore_requires_warmup_and_applied_route(self):
        item = {"kind": "retrieval", "group": "g", "mode": "restored", "concurrency": 8, "cut": "straddle",
                "meta": {"measured": True, "cut_ok": True, "chunk": 4096, "needle_start": 4090, "needle_end": 4100}}
        route = {"mode": "restored", "concurrency": 8, "cut": "straddle", "chunk": 4096,
                 "needle_start": 4090, "needle_end": 4100, "prefix_restored": True}
        result = {"strix": {"qualification_route": route}}
        warm = {"g": [{"reply": "ok", "error": None}]}
        self.assertTrue(client.route_measured(item, result, warm))
        self.assertFalse(client.route_measured(item, result, {}))
        self.assertFalse(client.route_measured(item, result, {"g": [{"reply": None, "error": "timeout"}]}))
        self.assertFalse(client.route_measured(item, {"strix": {"prefix_hit": True}}, warm))
        for field, value in (("prefix_restored", False), ("concurrency", 1), ("cut", "mid"), ("chunk", 2048)):
            bad = {"strix": {"qualification_route": dict(route, **{field: value})}}
            self.assertFalse(client.route_measured(item, bad, warm))

    def test_retrieval_resamples_haystacks_with_equal_cell_weights(self):
        e, r, t = {}, {}, {}
        for conc, value in ((1, 1), (8, 0)):
            for g in range(3):
                for k in range(conc):
                    key = f"{conc}-{g}-{k}"
                    e[key] = {"mode": "new", "concurrency": conc, "cut": "mid", "group": f"{conc}-{g}", "ok": 1}
                    r[key], t[key] = {"ok": value}, {"ok": 0}
        result = report.grouped_retrieval(e, r, t, list(e), 0, {"B": 100, "seed": 1})
        self.assertEqual(result["n"], 6)
        self.assertEqual(result["mean"], 0.5)  # C8 keys do not outweigh C1 haystacks
        one_each = [i for i in e if e[i]["group"].endswith("-0")]
        self.assertEqual(report.grouped_retrieval(e, r, t, one_each, 0, {"B": 100, "seed": 1})["verdict"],
                         "INCONCLUSIVE")

    def test_unmeasured_or_partial_retrieval_cannot_pass(self):
        items = []
        for mode in ("new", "restored"):
            for conc in (1, 8):
                for cut in ("straddle", "mid"):
                    for g in range(3):
                        group = f"{mode}-{conc}-{cut}-{g}"
                        for k in range(conc):
                            items.append({"id": f"{group}-{k}", "kind": "retrieval", "split": "retrieval",
                                          "length": 8192, "mode": mode, "concurrency": conc, "cut": cut, "group": group})
        m = http_fixture(items, "retrieval")
        for arm in ("E", "R", "T:a"):
            write_jsonl(self.tmp / f"scored.{safe_name(arm)}.jsonl", scored(m, arm, "retrieval"))
        boot = {"B": 100, "seed": 1}
        self.assertEqual(report.retrieval_results(self.tmp, ["T:a"], 0, boot, m)["T:a"]["verdict"], "PASS")
        path = self.tmp / "scored.T_a.jsonl"
        rows = read_jsonl(path)
        rows[0]["measured"] = False
        write_jsonl(path, rows)
        r = report.retrieval_results(self.tmp, ["T:a"], 0, boot, m)["T:a"]
        self.assertEqual(r["verdict"], "INCONCLUSIVE")
        self.assertTrue(any(c["measurement"] == "NOT-MEASURED" for c in r["matrix"]))
        write_jsonl(path, rows[:-1])
        with self.assertRaisesRegex(ValueError, "coverage"):
            report.retrieval_results(self.tmp, ["T:a"], 0, boot, m)

    def test_incomplete_gapped_v1_and_empty_v2_rejected(self):
        p = self.tmp / "bad.bin"
        spool.write(p, INPUTS, [0, 1, 2, 3], np.zeros((4, V), np.float32), version=1)
        self.assertFalse(spool.complete(p))
        with self.assertRaises(spool.SpoolError):
            list(spool.read(p))
        spool.write(p, INPUTS, list(range(len(INPUTS) - 1)), np.zeros((len(INPUTS) - 1, V), np.float32), version=1)
        data = bytearray(p.read_bytes())
        off = spool.header(p)["data_offset"]
        data[off:off + spool.ROW_HEAD.size] = spool.ROW_HEAD.pack(INPUTS[1], 1)
        p.write_bytes(data)
        self.assertFalse(spool.complete(p))
        with self.assertRaisesRegex(spool.SpoolError, "row index"):
            list(spool.read(p))
        p.write_bytes(spool.MAGIC2 + struct.pack("<QQQ", V, 0, len(INPUTS))
                      + np.asarray(INPUTS, dtype="<u4").tobytes() + spool.FOOTER + struct.pack("<Q", 0))
        self.assertFalse(spool.complete(p))


if __name__ == "__main__":
    unittest.main()
