"""Protocol tests: a shared window is insufficient without a shared committed prefix."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('mtp_study', Path(__file__).with_name('mtp-study-analyze.py'))
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)

def input_hash(inputs):
    h = 14695981039346656037
    for token in inputs:
        for shift in range(0, 32, 8):
            h = ((h ^ ((token >> shift) & 255)) * 1099511628211) & ((1 << 64) - 1)
    return h

def write_dump(path, windows):
    with path.open('wb') as f:
        for base, inputs in windows:
            for row, consumed in enumerate(inputs):
                f.write(study.HEADER.pack(study.MAGIC, base + row, base, row, len(inputs), 3,
                                          consumed, input_hash(inputs[:row + 1]), 0, 1))
                f.write(np.array([0., 1., 2.], dtype='<f4').tobytes())

def run(ids, request_sha='same'):
    request = dict(token_ids=ids, prompt=100, lane=0, request_sha256=request_sha)
    return [request], lambda ns: (0, request), {0: []}, {}

class ContextMatching(unittest.TestCase):
    def test_window_size_is_part_of_strict_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp) / 'r8.bin', Path(tmp) / 'r1.bin'
            ids = list(range(5, 13))
            write_dump(left, [(100, ids)])
            write_dump(right, [(100, ids[:1])])
            strict = study.compare_logits(left, right, run(ids), run(ids))
            self.assertEqual(strict['matched_context_rows'], 0)
            relaxed = study.compare_logits(left, right, run(ids), run(ids), context_only=True)
            self.assertEqual(relaxed['matched_context_rows'], 1)

    def test_missing_and_unattributed_verification_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'clock-anchor.json').write_text(json.dumps(dict(epoch_ns=0, monotonic_ns=0)))
            lane = dict(start=0, end=2, lane=0, acc=1)
            (path / 'res.jsonl').write_text(json.dumps(dict(t0=0, lanes=[lane])) + '\n')
            (path / 'srv.err').write_text('')
            with self.assertRaisesRegex(ValueError, 'HTTP accepted-token ledger'):
                study.load_run(path)
            lane['acc'] = 0
            (path / 'res.jsonl').write_text(json.dumps(dict(t0=0, lanes=[lane])) + '\n')
            self.assertEqual(study.load_run(path)[2][0], [])
            (path / 'srv.err').write_text('mtp-verify ns=3000000000\n')
            with self.assertRaisesRegex(ValueError, 'unresolved verification'):
                study.load_run(path)
            (path / 'res.jsonl').write_text(json.dumps(dict(t0=0, lanes=[lane, lane])) + '\n')
            (path / 'srv.err').write_text('mtp-verify ns=1000000000\n')
            with self.assertRaisesRegex(ValueError, 'unresolved verification'):
                study.load_run(path)

    def test_verification_prefix_and_http_ledger_are_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'clock-anchor.json').write_text(json.dumps(dict(epoch_ns=0, monotonic_ns=0)))
            lane = dict(start=0, end=2, lane=0, acc=1)
            (path / 'res.jsonl').write_text(json.dumps(dict(t0=0, lanes=[lane])) + '\n')
            line = ('mtp-verify ns=1000000000 at=100 d=1 rows=2 active=2 budget=2 diagnostic=0 '
                    'proposal=6 target=6 equal=1 eligible=1 reached=1 accepted=1 greedy_accept=1 eos=0 p=0.9\n')
            (path / 'srv.err').write_text(line)
            loaded = study.load_run(path)
            self.assertEqual(loaded[3][(0, 'code', 1)]['independently_expected_accept'], 1)
            (path / 'srv.err').write_text(line.replace('greedy_accept=1', 'greedy_accept=0'))
            with self.assertRaisesRegex(ValueError, 'prefix reconstruction'):
                study.load_run(path)
            (path / 'srv.err').write_text(line)
            lane['acc'] = 0
            (path / 'res.jsonl').write_text(json.dumps(dict(t0=0, lanes=[lane])) + '\n')
            with self.assertRaisesRegex(ValueError, 'HTTP accepted-token ledger'):
                study.load_run(path)

    def test_prompt_tail_does_not_rebase_completion_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.bin'
            write_dump(path, [(94, [42]), (99, [9]), (100, [5, 42]), (101, [8])])
            indexed, counts = study.indexed_logits(path, run([5, 8]))
            self.assertEqual(counts['prompt_rows'], 2)
            self.assertEqual(counts['indexed'], 3)
            self.assertEqual(counts['ledger_mismatch'], 0)
            self.assertEqual(counts['missing_window_base'], 0)
            self.assertEqual(len(indexed), 3)

    def test_cross_row_matching_requires_complete_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp) / 'l.bin', Path(tmp) / 'r.bin'
            write_dump(left, [(100, [5, 42])])
            write_dump(right, [(100, [5]), (101, [42])])
            matched = study.compare_logits(left, right, run([5, 42]), run([5, 42]), context_only=True)
            self.assertEqual(matched['matched_context_rows'], 2)
            self.assertEqual(matched['matched_right_rows'], 2)
            self.assertEqual(matched['matched_right_frontier_rows'], matched['right_frontier_rows'])
            self.assertTrue(matched['completion_ledger_pairs'][0]['equal'])
            self.assertTrue(any(r['left_row'] == 1 and r['right_row'] == 0 for r in matched['row_pairs']))
            changed = study.compare_logits(left, right, run([5, 8]), run([5, 8]), context_only=True)
            self.assertEqual(changed['matched_context_rows'], 1)
            self.assertEqual(changed['matched_right_rows'], 1)

    def test_speculative_input_need_not_match_eventual_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.bin'
            write_dump(path, [(100, [5, 42])])
            indexed, counts = study.indexed_logits(path, run([5, 8]))
            self.assertEqual(counts['indexed'], 2)
            self.assertEqual(len(indexed), 2)

    def test_identical_current_input_does_not_hide_changed_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp) / 'l.bin', Path(tmp) / 'r.bin'
            write_dump(left, [(100, [5]), (102, [9])])
            write_dump(right, [(100, [5]), (102, [9])])
            result = study.compare_logits(left, right, run([5, 6, 9]), run([5, 7, 9]))
            self.assertEqual(result['matched_context_rows'], 1)
            self.assertEqual(result['classes']['code']['bitwise_rows'], 1)

    def test_prompt_identity_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.bin'
            write_dump(path, [(100, [5])])
            with self.assertRaisesRegex(ValueError, 'prompt identity'):
                study.compare_logits(path, path, run([5], 'left'), run([5], 'right'))

    def test_corrupt_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.bin'
            path.write_bytes(b'partial')
            with self.assertRaisesRegex(ValueError, 'truncated'):
                list(study.dump_records(path))

if __name__ == '__main__':
    unittest.main()
