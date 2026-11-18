"""Synthetic CPU tests of fast_logit_gate.py (numpy only; no engine)."""
import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fast_logit_gate as g  # noqa: E402

ROWS, WIDTHS, TOKENS = 8, (5, 3), 40


def run(argv):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return g.main(argv)


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).resolve().parents[2])
        self.root = pathlib.Path(self.tmp.name)
        self.rng = np.random.default_rng(7)
        self.tok = self.root / "p.u32"
        self.rng.integers(0, sum(WIDTHS), TOKENS, dtype=np.uint32).astype("<u4").tofile(self.tok)
        self.binary = self.root / "engine"
        self.binary.write_bytes(b"binary")
        self.logits = self.rng.normal(size=(ROWS, sum(WIDTHS))).astype("<f4")

    def tearDown(self):
        self.tmp.cleanup()

    def selection(self, path, mode, swap=False):
        words = []
        for layer in range(2):
            ids = np.tile(np.arange(4, dtype=np.uint32), (ROWS, 2, 1))
            if swap and layer == 1:
                ids[3, 1, 2] = 9
            ids[:, :, 3] = 77 + mode  # beyond the count: ignored
            words += [g.SEL_MAGIC, 1, mode, 16, 0, ROWS, 2, 4] + [3] * (ROWS * 2) + ids.ravel().tolist()
        np.array(words, "<u4").tofile(path)

    def arm(self, name, run_id, logits=None, mode=4, swap=False, seal=True, select=True, witness=None, model_sha=None):
        d = self.root / name
        d.mkdir()
        z = self.logits if logits is None else logits
        rc = []
        for rank, (lo, hi) in enumerate(((0, WIDTHS[0]), (WIDTHS[0], sum(WIDTHS)))):
            z[:, lo:hi].astype("<f4").tofile(d / f"q-p-tf-r{rank}.f32")
            if select:
                self.selection(d / f"q-p-sel-r{rank}.u32", mode, swap and rank == 1)
            if seal:
                evidence = []
                if witness:
                    files = {}
                    for kind, ext in [("tf", "f32")] + ([("sel", "u32")] if select else []):
                        f = d / f"q-p-{kind}-r{rank}.{ext}"
                        files[kind] = {"name": f.name, "bytes": f.stat().st_size, "sha256": g.sha256_file(f)}
                        files[kind].update({"width": hi - lo} if kind == "tf" else {"records": 2, "mode": mode})
                    w = {"schema": "strix-fast-execution/v1", "run_id": run_id, "rank": rank,
                         "binary_sha256": g.sha256_file(self.binary), "engine": "ds41",
                         "profile": "fast-prefill" if witness == "candidate" else "exact",
                         "model_sha256": "a" * 64, "numerical_config_sha256": ("c" if witness == "candidate" else "b") * 64,
                         "control": {"kind": "layer0-null", "layer": 0, "magnitude": 1, "unit": "ulp"} if witness == "null" else None,
                         "prompts": {"p": {"tokens": TOKENS, "tokens_sha256": g.sha256_file(self.tok), "rows": ROWS, "files": files}}}
                    wp = d / f"execution-r{rank}.json"
                    wp.write_text(json.dumps(w), encoding="utf-8")
                    evidence = ["--execution-witness", str(wp)]
                rc.append(run(["seal", "--dir", str(d), "--rank", str(rank), "--arm", name, "--run-id", run_id, "--binary",
                               str(self.binary), "--rows", str(ROWS), "--prompt", f"p={self.tok}", "--env", "MODE=x",
                               "--attest", f"sel_mode={mode}"] + (["--select"] if select else []) + evidence
                               + (["--model-sha", model_sha] if model_sha else [])))
        return d, rc

    def cmp(self, a, b, *extra):
        return run(["compare", str(a), str(b), "--ranks", "0", "1", "--tokens", f"p={self.tok}", *extra])

    def test_identical_repeat_passes(self):
        a, rc = self.arm("A", "r1")
        b, _ = self.arm("A2", "r2")
        self.assertEqual(rc, [0, 0])
        self.assertEqual(self.cmp(a, b, "--expect-identical"), 0)
        self.assertEqual(self.cmp(a, b), 0)

    def test_small_shift_passes_and_large_fails(self):
        a, _ = self.arm("A", "r1")
        near, _ = self.arm("N", "r2", self.logits + self.rng.normal(scale=1e-3, size=self.logits.shape).astype("<f4"), mode=3)
        far, _ = self.arm("F", "r3", self.noisy(2.0, 5), mode=3)
        self.assertEqual(self.cmp(a, near), 0)
        self.assertEqual(self.cmp(a, near, "--expect-identical"), 1)
        self.assertEqual(self.cmp(a, far), 1)

    def test_selection_difference_is_counted(self):
        a, _ = self.arm("A", "r1")
        b, _ = self.arm("X", "r2", mode=3, swap=True)
        out = self.root / "r.json"
        self.assertEqual(self.cmp(a, b, "--json", str(out)), 0)
        self.assertEqual(self.cmp(a, b, "--expect-identical"), 1)
        import json
        sel = json.loads(out.read_text())["prompts"]["p"]["selection"]
        self.assertEqual((sel["layers"], sel["slot_diff"], sel["id_diff"], sel["first_layer"], sel["first_pos"]), (4, 1, 1, 3, 19))
        self.assertEqual(sel["modes"], [4, 3])

    def test_identity_defects_have_no_verdict(self):
        a, _ = self.arm("A", "r1")
        same, _ = self.arm("S", "r1")
        self.assertEqual(self.cmp(a, same), 2)  # one run id on both sides
        unsealed, _ = self.arm("U", "r2", seal=False)
        self.assertEqual(self.cmp(a, unsealed), 2)
        b, _ = self.arm("B", "r3")
        with open(b / "q-p-tf-r1.f32", "r+b") as f:  # changed after sealing
            f.write(b"\0\0\0\0")
        self.assertEqual(self.cmp(a, b), 2)

    def test_seal_rejects_missing_short_stale_reused_and_wrong_mode(self):
        d, _ = self.arm("A", "r1", seal=False)
        base = ["seal", "--dir", str(d), "--arm", "A", "--run-id", "r1", "--binary", str(self.binary), "--rows", str(ROWS),
                "--prompt", f"p={self.tok}", "--select"]
        self.assertEqual(run(base + ["--rank", "0", "--attest", "sel_mode=3"]), 2)  # dump says mode 4
        self.assertEqual(run(base + ["--rank", "0", "--not-before", str(os.path.getmtime(d / "q-p-tf-r0.f32") + 60)]), 2)
        self.assertEqual(run(base + ["--rank", "0"]), 0)
        self.assertEqual(run(base + ["--rank", "0"]), 2)  # sealed twice
        os.remove(d / "q-p-sel-r1.u32")
        self.assertEqual(run(base + ["--rank", "1"]), 2)
        self.selection(d / "q-p-sel-r1.u32", 4)
        with open(d / "q-p-tf-r1.f32", "ab") as f:
            f.write(b"\0\0")
        self.assertEqual(run(base + ["--rank", "1"]), 2)
        bad = self.logits.copy()
        bad[0, 0] = np.nan
        e, rc = self.arm("E", "r9", bad)
        self.assertEqual(rc, [2, 0])

    # ---- floor-relative dNLL rule (DS41 FAST prefill only)
    def noisy(self, scale, seed):
        rng = np.random.default_rng(seed)
        return self.logits + rng.normal(scale=scale, size=self.logits.shape).astype("<f4")

    def floor_args(self, nulls, engine="ds41", profile="fast-prefill"):
        out = ["--dnll-rule", "floor", "--engine", engine, "--profile", profile]
        for n in nulls:
            out += ["--null-arm", str(n)]
        return out

    def test_dsq3_numbers(self):
        nulls = [0.01408, 0.01140, 0.01111]
        mean, sd, limit = g.floor_limit(nulls)
        self.assertAlmostEqual(mean, 0.01220, places=5)
        self.assertAlmostEqual(sd, 0.00164, places=5)
        self.assertAlmostEqual(limit, 0.01547, places=5)
        for dnll, ok, abs_ok in ((0.01203, True, False), (0.01458, True, False), (0.01600, False, False), (0.00485, True, True)):
            v = g.dnll_verdict(dnll, "floor", 0.01, nulls)
            self.assertEqual((v["pass"], v["abs_pass"], v["n_nulls"], v["sigmas"]), (ok, abs_ok, 3, 2.0), dnll)
        v = g.dnll_verdict(0.01203, "abs", 0.01)
        self.assertEqual((v["pass"], v["limit"]), (False, 0.01))
        with self.assertRaises(g.GateError):
            g.floor_limit([0.014, 0.011])
        with self.assertRaises(g.GateError):
            g.floor_limit([0.014, 0.011, float("nan")])

    def test_policy_table(self):
        self.assertEqual(g.allowed_dnll_rules("ds41", "fast-prefill"), ("abs", "floor"))
        for e, p in (("ds41", "exact"), ("qwen", "fast-prefill"), ("minimax", "fast"), ("glm", "fast-prefill")):
            self.assertEqual(g.allowed_dnll_rules(e, p), ("abs",))
            with self.assertRaises(g.GateError):
                g.check_dnll_policy("floor", e, p, 3)
        with self.assertRaises(g.GateError):
            g.check_dnll_policy("floor", "ds41", "fast-prefill", 2)
        with self.assertRaises(g.GateError):
            g.check_dnll_policy("floor", None, None, 3)
        with self.assertRaises(g.GateError):
            g.check_dnll_policy("abs", None, None, 1)
        g.check_dnll_policy("floor", "ds41", "fast-prefill", 3)

    def test_floor_end_to_end(self):
        import json
        a, _ = self.arm("A", "r1", witness="baseline")
        nulls = [self.arm(f"N{i}", f"n{i}", self.noisy(0.3, 10 + i), mode=3, witness="null")[0] for i in range(3)]
        cand, _ = self.arm("C", "c1", self.noisy(0.3, 20), mode=3, witness="candidate")
        out = self.root / "f.json"
        rc = self.cmp(a, cand, *self.floor_args(nulls), "--json", str(out))
        p = json.loads(out.read_text())["pooled"]
        d = p["dnll_rule"]
        vals = [n["dnll"] for n in d["nulls"]]
        self.assertEqual([n["run_id"] for n in d["nulls"]], ["n0", "n1", "n2"])
        self.assertEqual(d["nulls"][0]["execution"]["control"], g.DS41_NULL_CONTROL)
        self.assertAlmostEqual(d["mean"], float(np.mean(vals)), places=12)
        self.assertAlmostEqual(d["sd"], float(np.std(vals, ddof=1)), places=12)
        self.assertAlmostEqual(d["limit"], d["mean"] + 2 * d["sd"], places=12)
        self.assertEqual((d["rule"], d["engine"], d["profile"], d["abs_limit"]), ("floor", "ds41", "fast-prefill", 0.01))
        self.assertEqual(d["abs_pass"], p["dnll"] <= 0.01)
        want = p["kl"] <= 0.02 and p["top1"] >= 0.93 and p["dnll"] <= d["limit"]
        self.assertEqual((rc, p["pass"]), (0 if want else 1, want))
        self.assertIn("floor ds41/fast-prefill", p["rule"])

    def test_floor_refusals_have_no_verdict(self):
        a, _ = self.arm("A", "r1", witness="baseline")
        nulls = [self.arm(f"N{i}", f"n{i}", self.noisy(1e-3, 10 + i), mode=3, witness="null")[0] for i in range(3)]
        cand, _ = self.arm("C", "c1", self.noisy(1e-3, 20), mode=3, witness="candidate")
        self.assertEqual(self.cmp(a, cand, *self.floor_args(nulls[:2])), 2)  # two nulls
        self.assertEqual(self.cmp(a, cand, *self.floor_args([])), 2)  # no nulls
        self.assertEqual(self.cmp(a, cand, *self.floor_args(nulls, "qwen")), 2)  # engine not in the policy table
        self.assertEqual(self.cmp(a, cand, *self.floor_args(nulls, "ds41", "exact")), 2)  # profile not in the table
        self.assertEqual(self.cmp(a, cand, "--dnll-rule", "floor", *sum((["--null-arm", str(n)] for n in nulls), [])), 2)
        self.assertEqual(self.cmp(a, cand, "--null-arm", str(nulls[0])), 2)  # nulls without the floor rule
        self.assertEqual(self.cmp(a, cand, *self.floor_args(nulls), "--expect-identical"), 2)
        self.assertEqual(self.cmp(a, cand, *self.floor_args([nulls[0], nulls[1], nulls[0]])), 2)  # one null twice
        self.assertEqual(self.cmp(a, cand, *self.floor_args([nulls[0], nulls[1], cand])), 2)  # candidate as a null
        other = self.root / "engine2"
        other.write_bytes(b"other binary")
        self.binary, keep = other, self.binary
        foreign, _ = self.arm("NB", "nb", self.noisy(1e-3, 30), mode=3, witness="null")
        self.binary = keep
        self.assertEqual(self.cmp(a, cand, *self.floor_args([nulls[0], nulls[1], foreign])), 2)  # null on another binary
        self.assertIn(self.cmp(a, cand, *self.floor_args(nulls)), (0, 1))  # the valid call reaches a verdict

    def floor_fixture(self, witness="null"):
        # Depress only the next-token logit; the winning alternative stays unchanged.
        ref = np.zeros_like(self.logits)
        labels = np.concatenate((np.fromfile(self.tok, "<u4")[-ROWS + 1:], [0]))
        ref[np.arange(ROWS), (labels + 1) % ref.shape[1]] = 1

        def shifted(dnll):
            z = ref.copy()
            mass = np.exp(1) + z.shape[1] - 1
            z[np.arange(ROWS), labels] = -np.log((mass * np.exp(dnll) - 1) / (mass - 1))
            return z

        a, rc = self.arm("A", "ref", ref, witness="baseline", select=False)
        self.assertEqual(rc, [0, 0])
        cand, rc = self.arm("C", "cand", shifted(0.016), witness="candidate", select=False)
        self.assertEqual(rc, [0, 0])
        nulls = []
        for i, dnll in enumerate((0.04, 0.05, 0.06)):
            d, rc = self.arm(f"layer0-null-{i}", f"null-{i}", shifted(dnll), witness=witness, select=False)
            self.assertEqual(rc, [0, 0])
            nulls.append(d)
        return a, cand, nulls

    def test_floor_rejects_inflated_non_controls_despite_names(self):
        a, cand, nulls = self.floor_fixture(witness="baseline")
        out = self.root / "refused.json"
        self.assertEqual(self.cmp(a, cand, *self.floor_args(nulls), "--json", str(out)), 2)
        self.assertFalse(out.exists())

    def test_floor_rejects_missing_control_witness(self):
        a, cand, nulls = self.floor_fixture(witness=None)
        self.assertEqual(self.cmp(a, cand, *self.floor_args(nulls)), 2)

    def test_floor_witness_incompatibilities(self):
        a, cand, nulls = self.floor_fixture()
        for target, key, value in ((nulls[0], "control", None),
                                   (nulls[0], "control", {**g.DS41_NULL_CONTROL, "layer": 1}),
                                   (nulls[0], "control", {**g.DS41_NULL_CONTROL, "magnitude": 2}),
                                   (nulls[0], "model_sha256", "d" * 64),
                                   (nulls[0], "numerical_config_sha256", "d" * 64),
                                   (cand, "engine", "qwen"), (cand, "profile", "exact"),
                                   (a, "control", g.DS41_NULL_CONTROL)):
            with self.subTest(key=key, value=value):
                paths = [target / f"manifest-r{r}.json" for r in range(2)]
                originals = [p.read_text(encoding="utf-8") for p in paths]
                for p, original in zip(paths, originals):
                    doc = json.loads(original)
                    doc["execution"][key] = value
                    p.write_text(json.dumps(doc), encoding="utf-8")
                self.assertEqual(self.cmp(a, cand, *self.floor_args(nulls)), 2)
                for p, original in zip(paths, originals):
                    p.write_text(original, encoding="utf-8")

    def test_floor_rejects_rank_witness_disagreement(self):
        a, cand, nulls = self.floor_fixture()
        p = nulls[0] / "manifest-r1.json"
        original = json.loads(p.read_text(encoding="utf-8"))
        for value in (None, {**original["execution"], "numerical_config_sha256": "d" * 64}):
            doc = dict(original)
            if value is None:
                del doc["execution"]
            else:
                doc["execution"] = value
            p.write_text(json.dumps(doc), encoding="utf-8")
            self.assertEqual(self.cmp(a, cand, *self.floor_args(nulls)), 2)

    def test_seal_binds_execution_witness(self):
        d, rc = self.arm("A", "r1", witness="null", select=False)
        self.assertEqual(rc, [0, 0])
        p = d / "execution-r0.json"
        original = json.loads(p.read_text(encoding="utf-8"))
        (d / "manifest-r0.json").unlink()
        base = ["seal", "--dir", str(d), "--rank", "0", "--arm", "A", "--run-id", "r1", "--binary", str(self.binary),
                "--rows", str(ROWS), "--prompt", f"p={self.tok}", "--execution-witness", str(p)]
        for key, value in (("run_id", "other"), ("rank", 1), ("binary_sha256", "d" * 64),
                           ("prompts", {}), ("model_sha256", ""), ("control", "layer0-null")):
            with self.subTest(key=key):
                p.write_text(json.dumps({**original, key: value}), encoding="utf-8")
                self.assertEqual(run(base), 2)
                self.assertFalse((d / "manifest-r0.json").exists())
        p.write_text("{", encoding="utf-8")
        self.assertEqual(run(base), 2)
        p.write_text(json.dumps(original), encoding="utf-8")
        self.assertEqual(run(base), 0)

    def test_default_rule_unchanged(self):
        import json
        a, _ = self.arm("A", "r1")
        for i, scale in enumerate((1e-3, 0.3, 1.0)):
            cand, _ = self.arm(f"C{i}", f"c{i}", self.noisy(scale, 40 + i), mode=3)
            out = self.root / f"d{i}.json"
            rc = self.cmp(a, cand, "--json", str(out))
            p = json.loads(out.read_text())["pooled"]
            want = p["kl"] <= 0.02 and p["top1"] >= 0.93 and p["dnll"] <= 0.01
            self.assertEqual((rc, p["pass"]), (0 if want else 1, want))
            self.assertEqual(p["rule"], "KL <= 0.02, top-1 >= 0.93, dNLL <= 0.01")
            self.assertEqual((p["dnll_rule"]["rule"], p["dnll_rule"]["limit"], p["dnll_rule"]["nulls"]), ("abs", 0.01, []))

    def turbo(self, t_logits, r_logits, *extra, r_sha="b" * 64, t_sha="a" * 64, declare=True, with_r=True):
        e, _ = self.arm("E", "e1", select=False, model_sha="a" * 64)
        t, _ = self.arm("T", "t1", t_logits, select=False, model_sha=t_sha)
        args = ["--mode", "turbo"]
        if with_r:
            r, _ = self.arm("R", "r1", r_logits, select=False, model_sha=r_sha)
            args += ["--reference-pack-arm", str(r)]
            if declare:
                args += ["--reference-sha", r_sha]
        return self.cmp(e, t, *args, *extra)

    def test_turbo_requires_known_pack_identities(self):
        e, erc = self.arm("E", "e1", select=False)
        t, trc = self.arm("T", "t1", select=False)
        r, rrc = self.arm("R", "r1", select=False)
        self.assertEqual(erc + trc + rrc, [0] * 6)
        self.assertEqual(self.cmp(e, t, "--mode", "turbo", "--reference-pack-arm", str(r)), 2)
        for value in (None, "", "A", "g" * 64, 42):
            with self.subTest(sha=value), self.assertRaises(g.GateError):
                g.arm_sha([{"model_sha": value}])
        with self.assertRaises(g.GateError):
            g.arm_sha([{"model_sha": "a" * 64}, {}])
        self.assertEqual(g.arm_sha([{"execution": {"model_sha256": "A" * 64}}]), "a" * 64)
        self.assertEqual(g.arm_sha([{"model_sha": "a" * 64, "execution": {"model_sha256": "A" * 64}}]), "a" * 64)

    def test_turbo_rejects_declared_witness_sha_conflict(self):
        e, _ = self.arm("E", "e1", select=False, witness="exact", model_sha="b" * 64)
        t, _ = self.arm("T", "t1", select=False, witness="exact", model_sha="b" * 64)
        r, _ = self.arm("R", "r1", select=False, witness="exact", model_sha="b" * 64)
        self.assertEqual(self.cmp(e, t, "--mode", "turbo", "--reference-pack-arm", str(r)), 2)
        with self.assertRaises(g.GateError):
            g.arm_sha([{"model_sha": "b" * 64, "execution": {"model_sha256": "a" * 64}}])

    def test_turbo_absolute_task_failures_without_task_reference(self):
        e, _ = self.arm("E", "e1", select=False, model_sha="a" * 64)
        t, _ = self.arm("T", "t1", select=False, model_sha="a" * 64)
        r, _ = self.arm("R", "r1", select=False, model_sha="a" * 64)
        path, receipt = self.root / "tasks.json", self.root / "receipt.json"
        args = ["--mode", "turbo", "--reference-pack-arm", str(r), "--task-results", str(path), "--receipt", str(receipt)]
        for category in ("json", "code", "math", "retrieval", "tools"):
            task = {"name": "case", "category": category, "E": {"passed": 10, "total": 10},
                    "T": {"passed": 0, "total": 10}}
            with self.subTest(category=category):
                path.write_text(json.dumps({"tasks": [task]}), encoding="utf-8")
                self.assertEqual(self.cmp(e, t, *args), 1)
                doc = json.loads(receipt.read_text(encoding="utf-8"))
                self.assertEqual(doc["verdict"], "FAIL")
                self.assertTrue(any(category in name for name in doc["failing"]))
                self.assertFalse(any(c["name"].endswith(":vs-R") for c in doc["checks"]))
                task["T"]["passed"] = 10
                path.write_text(json.dumps({"tasks": [task]}), encoding="utf-8")
                self.assertEqual(self.cmp(e, t, *args), 3)

    def test_turbo_pass_fail_and_no_reference(self):
        near = self.logits + self.rng.normal(scale=1e-2, size=self.logits.shape).astype("<f4")
        self.assertEqual(self.turbo(near, self.noisy(2.0, 5)), 0)
        self.tearDown(); self.setUp()
        self.assertEqual(self.turbo(self.noisy(2.0, 5), near), 1)
        self.tearDown(); self.setUp()
        self.assertEqual(self.turbo(near, None, with_r=False), 3)

    def test_turbo_reference_sha_must_be_declared_and_match(self):
        near = self.logits + self.rng.normal(scale=1e-2, size=self.logits.shape).astype("<f4")
        self.assertEqual(self.turbo(near, self.noisy(2.0, 5), declare=False), 2)
        self.tearDown(); self.setUp()
        e, _ = self.arm("E", "e1", select=False, model_sha="a" * 64)
        t, _ = self.arm("T", "t1", near, select=False, model_sha="a" * 64)
        r, _ = self.arm("R", "r1", self.noisy(2.0, 5), select=False, model_sha="b" * 64)
        base = ["--mode", "turbo", "--reference-pack-arm", str(r)]
        self.assertEqual(self.cmp(e, t, *base, "--reference-sha", "c" * 64), 2)
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(io.StringIO()):
            g.main(["compare", str(e), str(t), "--ranks", "0", "1", "--tokens", f"p={self.tok}", *base, "--reference-sha", "c" * 64])
        self.assertIn("c" * 64, buf.getvalue())
        self.assertIn("b" * 64, buf.getvalue())
        self.assertEqual(self.cmp(e, t, *base, "--reference-sha", "b" * 64), 0)

    def test_turbo_candidate_sha_must_be_declared(self):
        near = self.logits + self.rng.normal(scale=1e-2, size=self.logits.shape).astype("<f4")
        self.assertEqual(self.turbo(near, self.noisy(2.0, 5), t_sha="d" * 64), 2)
        self.tearDown(); self.setUp()
        self.assertEqual(self.turbo(near, self.noisy(2.0, 5), "--candidate-sha", "d" * 64, t_sha="d" * 64), 0)

    def test_task_results_wilson_and_verdict(self):
        lo, hi = g.wilson(8, 10)
        self.assertAlmostEqual(lo, 0.4902, places=3)
        self.assertAlmostEqual(hi, 0.9433, places=3)
        tasks = {"tasks": [{"name": "a", "category": "code", "E": {"passed": 8, "total": 10},
                            "T": {"passed": 8, "total": 10}, "R": {"passed": 8, "total": 10}},
                           {"name": "b", "category": "tools", "E": {"passed": 10, "total": 10},
                            "T": {"passed": 10, "total": 10}, "R": {"passed": 9, "total": 10}}]}
        f = self.root / "tasks.json"
        f.write_text(json.dumps(tasks), encoding="utf-8")
        near = self.logits + self.rng.normal(scale=1e-2, size=self.logits.shape).astype("<f4")
        out = self.root / "rc.json"
        self.assertEqual(self.turbo(near, self.noisy(2.0, 5), "--task-results", str(f), "--receipt", str(out)), 0)
        rc = json.loads(out.read_text())
        self.assertEqual(rc["tasks"]["code"]["T"]["wilson95"], [lo, hi])
        self.assertEqual(rc["tasks"]["overall"]["T"]["total"], 20)
        self.tearDown(); self.setUp()
        tasks["tasks"][1]["T"]["passed"] = 8  # tools drop 20 points: over the 1-point cap and worse than R
        f = self.root / "tasks.json"; f.write_text(json.dumps(tasks), encoding="utf-8")
        self.assertEqual(self.turbo(near, self.noisy(2.0, 5), "--task-results", str(f)), 1)
        self.tearDown(); self.setUp()
        tasks["tasks"][1]["T"]["passed"] = 10
        del tasks["tasks"][0]["R"]
        f = self.root / "tasks.json"; f.write_text(json.dumps(tasks), encoding="utf-8")
        self.assertEqual(self.turbo(near, self.noisy(2.0, 5), "--task-results", str(f)), 3)

    def test_turbo_receipt_keys(self):
        near = self.logits + self.rng.normal(scale=1e-2, size=self.logits.shape).astype("<f4")
        for name, body in (("man.json", {"hardware": "strix", "gpu_clock_mhz": 2900}), ("perf.json", {"tok_s": 12.5})):
            (self.root / name).write_text(json.dumps(body), encoding="utf-8")
        cfg = self.root / "cfg.txt"
        cfg.write_text("route=a\nroute=b\n", encoding="utf-8")
        out = self.root / "rc.json"
        self.assertEqual(self.turbo(near, self.noisy(2.0, 5), "--receipt", str(out), "--commit", "abc", "--calibration", "cal1",
                                    "--run-manifest", str(self.root / "man.json"), "--effective-config", str(cfg),
                                    "--perf", str(self.root / "perf.json")), 0)
        rc = json.loads(out.read_text())
        for k in ("schema", "verdict", "commit", "binary_sha256", "pack", "hardware", "routes", "metrics", "controls", "performance"):
            self.assertIn(k, rc)
        self.assertEqual(rc["schema"], "strix-turbo-receipt/v1")
        self.assertEqual(rc["routes"]["effective_config"], ["route=a", "route=b"])
        self.assertEqual(rc["hardware"]["gpu_clock_mhz"], 2900)
        self.assertEqual((rc["pack"]["R_sha"], rc["pack"]["calibration"]), ("b" * 64, "cal1"))
        self.assertEqual(rc["performance"]["tok_s"], 12.5)
        self.assertEqual(rc["performance"]["missing"], ["bytes_per_token", "ttft_ms", "p95_ms"])

    def test_fast_mode_flag_is_a_no_op(self):
        a, _ = self.arm("A", "r1")
        cand, _ = self.arm("C", "c1", self.noisy(0.3, 41), mode=3)
        outs = []
        for extra in ((), ("--mode", "fast")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = self.cmp(a, cand, "--json", str(self.root / "j.json"), *extra)
            outs.append((rc, buf.getvalue(), (self.root / "j.json").read_text()))
        self.assertEqual(outs[0], outs[1])
        self.assertNotIn("TURBO", outs[0][1])


MR, MW, MT = 64, (6, 5), 120  # MiniMax fixture: rows, rank vocabulary widths, prompt tokens


class MiniMaxFloor(unittest.TestCase):
    """MiniMax onset-matched floor and the launcher-side witness producer (`witness`)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).resolve().parents[2])
        self.root = pathlib.Path(self.tmp.name)
        rng = np.random.default_rng(11)
        head = rng.integers(0, sum(MW), MT - MR, dtype=np.int64)
        self.tokens = np.resize(head, MT)  # period MT - MR: the scored tail repeats the start (copy rows)
        self.tok = self.root / "p.u32"
        self.tokens.astype("<u4").tofile(self.tok)
        self.binary = self.root / "strix-minimax-rank"
        self.binary.write_bytes(b"binary")
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}", encoding="utf-8")
        header = json.dumps({"w": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}}).encode()
        (self.model / "model-00001-of-00001.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 4)
        self.labels = np.concatenate((self.tokens[MT - MR + 1:], [0]))
        self.ref = rng.normal(size=(MR, sum(MW)))
        self.ref[np.arange(MR), self.labels] += 4.0  # the reference predicts the true token well
        self.rng = rng

    def tearDown(self):
        self.tmp.cleanup()

    def receipt(self, rank, fast=None, extra="b", order=None):
        flash = "0" if fast is None else f"{fast[0] - 10}:layers=" + "".join("1" if fast[1] <= L else "0" for L in range(60))
        doc = {"schema": "strix-route-receipt-v1", "scope": "process", "engine": "minimax", "rank": rank, "fingerprint": "f",
               "levers": [{"name": "minimax.moe_prefill_format", "requested": "default", "effective": "exl3", "reason": "disabled"},
                          {"name": "minimax.msa_flash", "requested": "x", "effective": flash, "reason": "selected"},
                          {"name": "minimax.prefill_identity", "requested": "default", "effective": "default", "reason": "selected"}],
               "switches": {"set": {"STRIX_MM_PF_MSA_FLASH": "0" if fast is None else str(fast[0] - 10), "STRIX_X": extra,
                                    "STRIX_MM_QUALITY_OUT": f"/runs/{rank}-{fast}-{extra}-{self.rng.integers(1 << 30)}/q"}}}
        if fast is not None:
            doc["switches"]["set"]["STRIX_MM_PF_MSA_FLASH_LAYERS"] = f"{fast[1]}-59"
        if order is not None:
            doc["levers"].append({"name": "minimax.linear_order", "requested": "invariant", "effective": order, "reason": "selected"})
            doc["switches"]["set"]["STRIX_MMX_PF_ORDER"] = "invariant"
        return "strix-route-receipt " + json.dumps(doc)

    def mm_arm(self, name, logits, fast=None, control=None, extra="b", log=None, prompts=("p",), order=None):
        """Dumps for two ranks, an engine log per rank, then `witness` and `seal` exactly as a launcher would."""
        d = self.root / name
        d.mkdir()
        rcs = []
        for rank, (lo, hi) in enumerate(((0, MW[0]), (MW[0], sum(MW)))):
            for prompt in prompts:
                logits[:, lo:hi].astype("<f4").tofile(d / f"q-{prompt}-tf-r{rank}.f32")
            lines = ["mm rank: load", self.receipt(rank, fast, extra, order)]
            if control is not None:
                lines += [f"DIAG attn ulp: layer {control[0]} seed {control[1]} rows {min(64, MT - p)} pos0 {p} values 99"
                          f" prompt {json.dumps(prompt)}" for prompt in prompts for p in (0, 64)]
            (d / f"r{rank}.log").write_text("\n".join(log(rank) if log else lines) + "\n", encoding="utf-8")
            common = ["--dir", str(d), "--rank", str(rank), "--run-id", name, "--binary", str(self.binary), "--rows", str(MR)]
            for prompt in prompts:
                common += ["--prompt", f"{prompt}={self.tok}"]
            rc = run(["witness", *common, "--engine", "minimax", "--engine-log", str(d / f"r{rank}.log"), "--model",
                      str(self.model), "--out", str(d / f"execution-r{rank}.json")])
            if rc == 0:
                rc = run(["seal", *common, "--arm", name, "--execution-witness", str(d / f"execution-r{rank}.json")])
            rcs.append(rc)
        return d, rcs

    def noisy(self, scale, seed, rows=None):
        z = self.ref.copy()
        r = np.random.default_rng(seed)
        idx = np.arange(MR) if rows is None else rows
        z[idx] += r.normal(scale=scale, size=(len(idx), z.shape[1]))
        return z

    def cmp(self, ref, cand, nulls, *extra, out=None):
        args = ["compare", str(ref), str(cand), "--ranks", "0", "1", "--tokens", f"p={self.tok}", "--dnll-rule", "floor",
                "--engine", "minimax", "--profile", "fast-prefill", *sum((["--null-arm", str(n)] for n in nulls), []), *extra]
        return run(args + (["--json", str(out)] if out else []))

    def fixture(self, onset=30, null_scale=0.3, cand=None, null_layer=None, seeds=(1, 2, 3)):
        ref, rc = self.mm_arm("ref", self.ref)
        self.assertEqual(rc, [0, 0])
        nulls = []
        for i, s in enumerate(seeds):
            d, rc = self.mm_arm(f"null{i}", self.noisy(null_scale, 100 + i), control=(onset if null_layer is None else null_layer, s))
            self.assertEqual(rc, [0, 0])
            nulls.append(d)
        c, rc = self.mm_arm("cand", self.noisy(null_scale * 0.7, 200) if cand is None else cand, fast=(12, onset))
        self.assertEqual(rc, [0, 0])
        return ref, c, nulls

    def test_copy_rows_and_binomial_tail(self):
        t = np.array([1, 2, 3, 4, 9, 1, 2, 3, 4, 9, 1, 2, 3, 5])
        # rows = last 6 positions 8..13; labelled 8..12; k = 3: (2,3,4)->9 at 8 repeats 3 (9), (3,4,9)->1, (4,9,1)->2,
        # (9,1,2)->3, (1,2,3)->4 but the true next token is 5
        self.assertEqual(g.copy_rows(t, 6, k=3).tolist(), [True, True, True, True, False])
        self.assertEqual(g.copy_rows(t, 6, k=6).tolist(), [False, False, True, True, False])
        self.assertEqual(g.copy_rows(self.tokens, MR).sum(), MR - 1 - 7)
        self.assertEqual(g.binom_sf(0, 5, 0.5), 1.0)
        self.assertAlmostEqual(g.binom_sf(3, 3, 0.5), 0.125)
        self.assertAlmostEqual(g.binom_sf(2, 3, 0.5), 0.5)
        from math import comb
        for w in (0, 7, 15, 20):
            want = sum(comb(20, j) * 0.6 ** j * 0.4 ** (20 - j) for j in range(w, 21))
            self.assertAlmostEqual(g.binom_sf(w, 20, 0.6), want, places=12)

    def test_witness_linear_order(self):
        """The invariant linear order is a FAST layer set over every layer: onset 0, config equal to the exact base."""
        e, _ = self.mm_arm("e", self.ref)
        c, _ = self.mm_arm("c", self.noisy(0.1, 2), order="invariant:mmx-pf-linear-s1-v1")
        we, wc = (json.loads((d / "execution-r0.json").read_text(encoding="utf-8")) for d in (e, c))
        self.assertEqual(wc["profile"], "fast-prefill")
        self.assertEqual(wc["fast"], {"policy": "mmx-pf-linear-s1-v1", "msa_flash_mode": 0, "layers": list(range(60))})
        self.assertIsNone(g.config_difference(we["config"], wc["config"]))

    def test_witness_index_proj_member(self):
        """minimax.index_proj_wmma joins the FAST layer set (union); absent lever = off; its switches are FAST-only."""
        def run(fast, idx, evidence=True, event_rank=0, event_tokens=16, event_layers=None, order=None):
            doc = json.loads(self.receipt(0, fast, order=order)[len("strix-route-receipt "):])
            if idx is not None:
                doc["levers"].append({"name": "minimax.index_proj_wmma", "requested": "1", "effective": idx, "reason": "selected"})
                doc["switches"]["set"]["STRIX_MM_PF_IDX_WMMA"] = "1"
            log = "strix-route-receipt " + json.dumps(doc)
            digest = g.token_fnv1a64(bytes(64))
            if evidence and idx and idx.startswith("1:layers="):
                log += (f"\nminimax-index-proj-executed rank {event_rank} tokens {event_tokens} token_fnv {digest} layers " +
                        (event_layers if event_layers is not None else idx[len("1:layers="):]))
            return g.minimax_execution(log.encode(), 0, {"p": {"tokens": 16, "tokens_fnv1a64": digest}})
        on = "1:layers=" + "".join("1" if 40 <= L else "0" for L in range(60))
        both = run((11, 40), on)
        self.assertEqual(both["fast"], {"policy": "msa-flash+index-proj", "msa_flash_mode": 11, "layers": list(range(40, 60))})
        self.assertEqual(both["profile"], "fast-prefill")
        combined = run((11, 40), on, order="invariant:mmx-pf-linear-s1-v1")
        self.assertEqual(combined["fast"], {"policy": "mmx-pf-linear-s1-v1+msa-flash+index-proj",
                                            "msa_flash_mode": 11, "layers": list(range(60))})
        self.assertEqual(combined["profile"], "fast-prefill")
        self.assertIsNone(g.config_difference(run(None, None)["config"], combined["config"]))
        with self.assertRaises(g.GateError):
            run((11, 40), on, evidence=False, order="invariant:mmx-pf-linear-s1-v1")
        alone = run(None, "1:layers=" + "".join("1" if 30 <= L else "0" for L in range(60)))
        self.assertEqual(alone["fast"], {"policy": "index-proj", "msa_flash_mode": 10, "layers": list(range(30, 60))})
        self.assertEqual(run((11, 40), "0")["fast"]["policy"], "msa-flash")
        self.assertIsNone(run(None, None)["fast"])
        for bad in ("1", "1:layers=" + "0" * 60, "2:layers=1"):
            with self.assertRaises(g.GateError):
                run(None, bad)
        self.assertIsNone(g.config_difference(run((11, 40), None)["config"], both["config"]))
        for kwargs in ({"evidence": False}, {"event_rank": 1}, {"event_tokens": 8}, {"event_tokens": 17},
                       {"event_layers": "0" * 60}):
            with self.subTest(kwargs=kwargs), self.assertRaises(g.GateError):
                run(None, on, **kwargs)

    def test_witness_producer(self):
        e, _ = self.mm_arm("e", self.ref)
        n, _ = self.mm_arm("n", self.noisy(0.1, 1), control=(30, 2))
        c, _ = self.mm_arm("c", self.noisy(0.1, 2), fast=(12, 30))
        o, _ = self.mm_arm("o", self.noisy(0.1, 3), fast=(11, 30), extra="other")
        w = {k: json.loads((d / "execution-r0.json").read_text(encoding="utf-8")) for k, d in (("e", e), ("n", n), ("c", c), ("o", o))}
        self.assertEqual((w["e"]["profile"], w["e"]["fast"], w["e"]["control"]), ("exact", None, None))
        self.assertEqual(w["n"]["control"], {"kind": "attn-ulp-null", "layer": 30, "magnitude": 1, "unit": "ulp", "seed": 2})
        self.assertEqual(w["c"]["profile"], "fast-prefill")
        self.assertEqual(w["c"]["fast"], {"policy": "msa-flash", "msa_flash_mode": 12, "layers": list(range(30, 60))})
        self.assertEqual(w["o"]["fast"]["msa_flash_mode"], 11)
        # output paths and the control are not numerical configuration; the FAST layer set is only outside the base
        self.assertEqual(w["e"]["numerical_config_sha256"], w["n"]["numerical_config_sha256"])
        self.assertNotEqual(w["e"]["numerical_config_sha256"], w["c"]["numerical_config_sha256"])
        self.assertIsNone(g.config_difference(w["e"]["config"], w["c"]["config"]))
        self.assertEqual(g.config_difference(w["e"]["config"], w["o"]["config"]), "switch STRIX_X='b' vs 'other'")
        lev = json.loads(json.dumps(w["c"]["config"]))
        lev["levers"]["minimax.prefill_identity"] = "fp32"
        self.assertEqual(g.config_difference(w["e"]["config"], lev), "lever minimax.prefill_identity")
        self.assertNotIn("STRIX_MM_QUALITY_OUT", w["e"]["config"]["switches"])
        self.assertEqual(len({x["model_sha256"] for x in w.values()}), 1)
        r0 = json.loads((c / "manifest-r0.json").read_text(encoding="utf-8"))
        self.assertEqual(r0["execution"]["prompts"], r0["prompts"])
        bad = {"no receipt": lambda r: ["mm rank: load"],
               "two receipts": lambda r: [self.receipt(r), self.receipt(r)],
               "other rank only": lambda r: [self.receipt(1 - r)],
               "control at two layers": lambda r: [self.receipt(r), 'DIAG attn ulp: layer 3 seed 1 rows 9 pos0 0 values 1 prompt "p"',
                                                   'DIAG attn ulp: layer 4 seed 1 rows 9 pos0 0 values 1 prompt "p"'],
               "bad flash value": lambda r: [self.receipt(r).replace('"effective": "0"', '"effective": "1"')]}
        for i, (why, log) in enumerate(bad.items()):
            with self.subTest(why=why):
                d, rc = self.mm_arm(f"bad{i}", self.ref, log=log)
                self.assertEqual(rc, [2, 2])
                self.assertFalse((d / "execution-r0.json").exists() or (d / "manifest-r0.json").exists())
        model = self.model
        self.model = self.root / "no-such-model"
        self.assertEqual(self.mm_arm("nomodel", self.ref)[1], [2, 2])
        self.model = model
        header = (self.model / "model-00001-of-00001.safetensors").read_bytes()
        (self.model / "model-00001-of-00001.safetensors").write_bytes(header.replace(b"F16", b"F32"))
        changed, _ = self.mm_arm("changed", self.ref)
        self.assertNotEqual(json.loads((changed / "execution-r0.json").read_text(encoding="utf-8"))["model_sha256"],
                            w["e"]["model_sha256"])

    def test_model_identity_binds_payload(self):
        weights = self.model / "model-00001-of-00001.safetensors"
        before = g.model_bundle_sha256(self.model)
        payload = weights.read_bytes()
        weights.write_bytes(payload[:-1] + b"\x01")
        self.assertNotEqual(before, g.model_bundle_sha256(self.model))

    def test_copy_rows_matches_full_history(self):
        rng = np.random.default_rng(625)
        for total in (4, 19, 128):
            tokens = rng.integers(0, 3, size=total)
            for rows in (1, min(8, total), total):
                for k in (1, 3, 8):
                    expected = []
                    for p in range(total - rows, total - 1):
                        earlier = [q for q in range(k - 1, p) if p >= k - 1 and
                                   np.array_equal(tokens[q - k + 1:q + 1], tokens[p - k + 1:p + 1])]
                        expected.append(bool(earlier) and tokens[earlier[-1] + 1] == tokens[p + 1])
                    self.assertEqual(g.copy_rows(tokens, rows, k).tolist(), expected)

    def test_model_identity_binds_large_file_middle(self):
        large = self.model / "weights.bin"
        with large.open("wb") as f:
            f.truncate((64 << 20) + 1)
        before = g.model_bundle_sha256(self.model)
        with large.open("r+b") as f:
            f.seek(32 << 20)
            f.write(b"\x01")
        self.assertNotEqual(before, g.model_bundle_sha256(self.model))

    def test_floor_refuses_payload_replacement(self):
        ref, cand, nulls = self.fixture()
        weights = self.model / "model-00001-of-00001.safetensors"
        payload = weights.read_bytes()
        weights.write_bytes(payload[:-1] + b"\x01")
        other, rc = self.mm_arm("new-weights", self.ref, fast=(12, 30))
        self.assertEqual(rc, [0, 0])
        self.assertEqual(self.cmp(ref, other, nulls), 2)

    def test_floor_refuses_one_sided_switches(self):
        base = {"levers": {}, "switches": {}}
        override = {"levers": {}, "switches": {"STRIX_MMX_PREFILL_PIPE_TAIL": "0"}}
        self.assertIsNotNone(g.config_difference(base, override))
        self.assertIsNotNone(g.config_difference(override, base))
        self.assertIsNone(g.config_difference(override, override))
        ref, cand, nulls = self.fixture()
        for i, value in enumerate(("0", "1")):
            def log(rank):
                line = self.receipt(rank, (12, 30))
                doc = json.loads(line[len(g.RECEIPT_PREFIX):])
                doc["switches"]["set"]["STRIX_MMX_PREFILL_PIPE_TAIL"] = value
                return [g.RECEIPT_PREFIX + json.dumps(doc)]
            other, rc = self.mm_arm(f"override{i}", self.ref, log=log)
            self.assertEqual(rc, [0, 0])
            self.assertEqual(self.cmp(ref, other, nulls), 2)

    def test_control_requires_per_prompt_nonzero_events(self):
        event = "DIAG attn ulp: layer 30 seed 7 rows 64 pos0 0 values 99"
        bad = {
            "unbound": [event, event.replace("pos0 0", "pos0 64")],
            "only-p": [event + ' prompt "p"', event.replace("pos0 0", "pos0 64") + ' prompt "p"'],
            "zero-values": [event + ' prompt "p"', event.replace("values 99", "values 0") + ' prompt "q"'],
            "zero-rows": [event + ' prompt "p"', event.replace("rows 64", "rows 0") + ' prompt "q"'],
            "out-of-range": [event + ' prompt "p"', event.replace("pos0 0", f"pos0 {MT}") + ' prompt "q"'],
            "unknown": [event + ' prompt "p"', event + ' prompt "unknown"'],
        }
        for name, events in bad.items():
            with self.subTest(name=name):
                d, rc = self.mm_arm(name, self.ref, prompts=("p", "q"), log=lambda r: [self.receipt(r), *events])
                self.assertEqual(rc, [2, 2])
                self.assertFalse((d / "execution-r0.json").exists())
        self.assertEqual(self.mm_arm("covered", self.ref, prompts=("p", "q"), control=(30, 7))[1], [0, 0])

    def test_floor_end_to_end(self):
        ref, cand, nulls = self.fixture()
        out = self.root / "f.json"
        rc = self.cmp(ref, cand, nulls, out=out)
        p = json.loads(out.read_text(encoding="utf-8"))["pooled"]
        d = p["dnll_rule"]
        names = [c["name"] for c in d["checks"]]
        self.assertEqual(names, ["kl", "top1", "dnll", "kl_q50", "kl_q90", "kl_q99", "sign", "copy"])
        self.assertEqual((d["policy"], d["onset"], d["n_nulls"]), ("minimax-onset-floor-v1.2", 30, 3))
        for q in d["checks"][3:6]:
            self.assertEqual(q["limit"], max(q["floor"], q["tol"]))
        kl = [n["stats"]["kl"] for n in d["nulls"]]
        c = d["checks"][0]
        self.assertAlmostEqual(c["floor"], float(np.mean(kl) + 2 * np.std(kl, ddof=1)), places=12)
        self.assertEqual(c["limit"], max(0.02, c["floor"]))  # the floor only relaxes the absolute bar
        self.assertEqual(d["limit"], min(max(0.01, d["floor_limit"]), 0.03))
        self.assertEqual([x["name"] for x in d["checks"] if not x["required"]], ["kl_q99"])
        self.assertEqual(p["pass"], all(x["pass"] for x in d["checks"] if x["required"]))
        self.assertEqual(rc, 0 if p["pass"] else 1)
        self.assertTrue(p["pass"], [x for x in d["checks"] if not x["pass"]])  # a candidate drawn like the nulls passes
        self.assertEqual(p["abs_pass"], p["kl"] <= 0.02 and p["top1"] >= 0.93 and p["dnll"] <= 0.01)

    def test_floor_without_copy_observations_cannot_pass(self):
        self.tokens = np.random.default_rng(680).integers(0, sum(MW), MT, dtype=np.int64)
        self.assertFalse(g.copy_rows(self.tokens, MR).any())
        self.tokens.astype("<u4").tofile(self.tok)
        ref, cand, nulls = self.fixture(null_scale=0, cand=self.ref)
        out = self.root / "no-copy.json"
        self.assertEqual(self.cmp(ref, cand, nulls, out=out), 1)
        pooled = json.loads(out.read_text(encoding="utf-8"))["pooled"]
        self.assertFalse(pooled["pass"])
        checks = {c["name"]: c for c in pooled["dnll_rule"]["checks"]}
        self.assertTrue(all(c["pass"] for name, c in checks.items() if name != "copy"))
        self.assertFalse(checks["copy"]["pass"])
        self.assertEqual(checks["copy"]["rows"], 0)
        self.assertEqual(checks["copy"]["reason"], "insufficient copy evidence")

    def test_floor_never_tighter_than_absolute(self):
        ref, _ = self.mm_arm("ref", self.ref)
        nulls = []
        for i, up in enumerate((0.30, 0.32, 0.34)):  # nulls that all improve the true token: floor limit < 0
            z = self.ref.copy()
            z[np.arange(MR - 1), self.labels[:-1]] += up
            nulls.append(self.mm_arm(f"null{i}", z, control=(40, i + 1))[0])
        z = self.ref.copy()
        z[np.arange(MR - 1), self.labels[:-1]] -= 0.02  # slightly worse than exact, far inside the absolute 0.01
        cand, _ = self.mm_arm("cand", z, fast=(11, 40))
        out = self.root / "tight.json"
        self.cmp(ref, cand, nulls, out=out)
        dn = next(c for c in json.loads(out.read_text(encoding="utf-8"))["pooled"]["dnll_rule"]["checks"] if c["name"] == "dnll")
        self.assertLess(dn["floor"], 0.0)
        self.assertEqual(dn["limit"], 0.01)
        self.assertTrue(dn["pass"], dn)

    def test_onset_seed_and_base_refusals(self):
        ref, cand, nulls = self.fixture()
        self.assertIn(self.cmp(ref, cand, nulls), (0, 1))
        for i in range(3):  # layer-0 nulls cannot judge a 30-59 candidate
            self.mm_arm(f"l0-{i}", self.noisy(0.3, 300 + i), control=(0, i + 1))
        self.assertEqual(self.cmp(ref, cand, [self.root / f"l0-{i}" for i in range(3)]), 2)
        for i, s in enumerate((5, 5, 6)):  # repeated seed
            self.mm_arm(f"rep-{i}", self.noisy(0.3, 400 + i), control=(30, s))
        self.assertEqual(self.cmp(ref, cand, [self.root / f"rep-{i}" for i in range(3)]), 2)
        other, _ = self.mm_arm("other", self.noisy(0.3, 500), fast=(12, 30), extra="changed")  # more than the layer set differs
        self.assertEqual(self.cmp(ref, other, nulls), 2)
        self.assertEqual(self.cmp(ref, cand, nulls[:2]), 2)
        self.assertEqual(self.cmp(cand, cand, nulls), 2)
        self.assertEqual(run(["compare", str(ref), str(cand), "--ranks", "0", "1", "--tokens", f"p={self.tok}", "--dnll-rule",
                              "floor", "--engine", "minimax", "--profile", "fast", *sum((["--null-arm", str(n)] for n in nulls), [])]), 2)

    def test_dnll_cap_keeps_a_wide_floor_a_bar(self):
        ref, rc = self.mm_arm("ref", self.ref)
        nulls = []
        for i, dn in enumerate((0.0, 0.08, 0.16)):  # nulls lowering the true token by very different amounts: floor ~ +0.24
            z = self.ref.copy()
            z[np.arange(MR - 1), self.labels[:-1]] -= dn * 3
            nulls.append(self.mm_arm(f"null{i}", z, control=(30, i + 1))[0])
        z = self.ref.copy()
        z[np.arange(MR - 1), self.labels[:-1]] -= 0.25
        cand, _ = self.mm_arm("cand", z, fast=(12, 30))
        out = self.root / "cap.json"
        self.assertEqual(self.cmp(ref, cand, nulls, out=out), 1)
        d = json.loads(out.read_text(encoding="utf-8"))["pooled"]["dnll_rule"]
        dn = next(c for c in d["checks"] if c["name"] == "dnll")
        self.assertGreater(dn["floor"], 0.03)
        self.assertEqual(dn["limit"], 0.03)
        self.assertLess(dn["value"], dn["floor"])
        self.assertFalse(dn["pass"])

    def test_sign_and_copy_catch_a_defect_inside_a_wide_floor(self):
        rows = np.arange(MR - 1)
        copy = np.nonzero(g.copy_rows(self.tokens, MR))[0]
        base = self.ref.copy()
        base[np.arange(MR), self.labels] += 4.0  # confident reference: every labelled row hits the true token
        nulls_z = []
        for i in range(3):  # chaotic nulls: large symmetric changes on non-copy rows (3 up, 3 down), copy rows untouched
            z = base.copy()
            pick = np.random.default_rng(600 + i).choice(np.setdiff1d(rows, copy), 6, replace=False)
            z[pick, self.labels[pick]] += np.array([-3.0, 3.0] * 3)
            nulls_z.append(z)
        cand = base.copy()
        cand[copy[:12], self.labels[copy[:12]]] -= 12.0  # a retrieval defect: 12 copy rows lose the true token
        ref, _ = self.mm_arm("ref", base)
        nulls = [self.mm_arm(f"null{i}", z, control=(30, i + 1))[0] for i, z in enumerate(nulls_z)]
        c, _ = self.mm_arm("cand", cand, fast=(11, 30))
        out = self.root / "defect.json"
        self.assertEqual(self.cmp(ref, c, nulls, out=out), 1)
        checks = {x["name"]: x for x in json.loads(out.read_text(encoding="utf-8"))["pooled"]["dnll_rule"]["checks"]}
        self.assertFalse(checks["sign"]["pass"])
        self.assertFalse(checks["copy"]["pass"])
        self.assertEqual((checks["copy"]["value"], checks["copy"]["worst_null"]), (12, 0))


if __name__ == "__main__":
    unittest.main()
