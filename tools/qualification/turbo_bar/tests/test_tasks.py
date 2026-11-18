import ast
import json
import re
import unittest

from tools.qualification.turbo_bar import tasks


class TaskTests(unittest.TestCase):
    def test_math_requires_complete_integer_field(self):
        for suffix in ('.5', '/2', 'e2', 'E+2', ' dollars', ',5', '**.5', '\nextra'):
            with self.subTest(suffix=suffix):
                self.assertFalse(tasks.score_math('ANSWER: 42' + suffix, {'answer': 42}))
        for reply in ('ANSWER: 42', 'Reasoning\nANSWER: **42**', 'ANSWER: +42\n'):
            self.assertTrue(tasks.score_math(reply, {'answer': 42}))
        self.assertTrue(tasks.score_math('ANSWER: -1,234', {'answer': -1234}))
        self.assertFalse(tasks.score_math('ANSWER: 42\nANSWER: 42/2', {'answer': 42}))

    @classmethod
    def setUpClass(cls):
        cls.items = tasks.generate()

    def of(self, kind, split=None):
        return [x for x in self.items if x["kind"] == kind and (split is None or x["split"] == split)]

    def test_counts_and_distinct(self):
        for kind in tasks.KINDS:
            e, c = self.of(kind, "explore"), self.of(kind, "confirm")
            self.assertEqual((len(e), len(c)), (32, 128))
            shas = [x["sha"] for x in e + c]
            self.assertEqual(len(shas), len(set(shas)), kind)
        self.assertEqual(len({x["id"] for x in self.items}), len(self.items))
        self.assertEqual(tasks.summary_sha(self.items), tasks.summary_sha(tasks.generate()))

    def test_math(self):
        it = self.of("math")[0]
        a = it["check"]["answer"]
        self.assertTrue(tasks.score_item(it, f"steps...\nANSWER: {a}"))
        self.assertTrue(tasks.score_item(it, f"ANSWER: 1\nmore\nANSWER: {a}"))
        self.assertFalse(tasks.score_item(it, f"ANSWER: {a + 1}"))
        self.assertFalse(tasks.score_item(it, f"the answer is {a}"))

    def test_needle(self):
        for it in self.of("needle")[:5]:
            text = it["messages"][0]["content"]
            v = it["check"]["answer"]
            key = re.search(r"archive code of (.+?)\? Answer", text).group(1)
            self.assertIn(f"The archive code of {key} is {v}.", text)
            self.assertNotIn(v, text.split("Question:")[1])
            self.assertTrue(tasks.score_item(it, v))
            self.assertFalse(tasks.score_item(it, "100000" if v != "100000" else "100001"))
        sents = re.split(r"(?<=\.)\s",self.of("needle")[0]["messages"][0]["content"])
        self.assertGreater(len(set(sents)), 0.9 * len(sents))   # varied filler, not one repeated sentence

    def test_json(self):
        it = next(x for x in self.of("json")
                  if any(isinstance(v, int) and not isinstance(v, bool) for v in x["check"]["arguments"].values()))
        c = it["check"]
        good = json.dumps({"name": c["name"], "arguments": c["arguments"]})
        self.assertTrue(tasks.score_item(it, " \n" + good + "\t"))
        for reply in ("Sure: " + good, good + " trailing", good + good, "[" + good + "]",
                      "```json\n" + good + "\n```", "Sure: " + good + " trailing"):
            self.assertFalse(tasks.score_item(it, reply), reply)
        self.assertFalse(tasks.score_item(it, json.dumps({"name": c["name"], "arguments": dict(c["arguments"], extra=1)})))
        self.assertFalse(tasks.score_item(it, json.dumps({"name": "other", "arguments": c["arguments"]})))
        k = next(k for k, v in c["arguments"].items() if isinstance(v, int) and not isinstance(v, bool))
        self.assertFalse(tasks.score_item(it, json.dumps({"name": c["name"], "arguments": dict(c["arguments"], **{k: str(c["arguments"][k])})})))

    def test_code_scorer(self):
        self.assertGreaterEqual(len(tasks.CODE_TEMPLATES), 10)
        it = next(x for x in self.of("code") if x["check"]["entry"] == "sum_multiples")
        a, b = map(int, re.search(r"divisible by (\d+) or by (\d+)", it["messages"][0]["content"]).groups())
        good = (f"Here:\n```python\ndef sum_multiples(n):\n    return sum(i for i in range(1, n) if i % {a} == 0 or i % {b} == 0)\n```\n")
        bad = good.replace("range(1, n)", "range(1, n + 1)")
        raw = tasks.extract_code(good)
        self.assertTrue(raw.startswith("def sum_multiples(n):"))
        self.assertEqual(tasks.extract_code(raw), raw)
        for reply in (good, bad, "no code at all", raw, "import sys; sys.exit(0)"):
            with self.assertRaisesRegex(tasks.ScoringUnavailable, "sandbox"):
                tasks.score_item(it, reply)

    def test_every_code_template_has_working_tests(self):
        # a lookup table built from the hidden asserts must pass, a constant must not
        for it in self.of("code", "explore"):
            tree = ast.parse(it["check"]["tests"])
            table = {}
            for node in tree.body:
                call, exp = node.test.left, node.test.comparators[0]
                table[ast.unparse(call)] = ast.unparse(exp)
            entry = it["check"]["entry"]
            src = f"def {entry}(*a):\n    t = {{{', '.join(f'{k!r}: {v}' for k, v in table.items())}}}\n" \
                  f"    return t[{entry!r} + '(' + ', '.join(map(repr, a)) + ')']\n"
            # These are trusted test-generated functions, never model replies.
            ns = {}
            exec(src, ns)
            exec(it["check"]["tests"], ns)
            exec(f"def {entry}(*a):\n    return None\n", ns)
            with self.assertRaises(AssertionError):
                exec(it["check"]["tests"], ns)


if __name__ == "__main__":
    unittest.main()
