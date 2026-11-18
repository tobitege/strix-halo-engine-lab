"""Offline SSE and benchmark exit-status regressions; no server or GPU required."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('qwen_serve_bench', ROOT / 'scripts/bench/cuda/qwen_serve_bench.py')
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def event(obj):
    return b'data: ' + json.dumps(obj).encode() + b'\n\n'


def completion(ids, *, count=None, finish='length', done=True, error=None, full_ids=True, rtx=True):
    metadata = {'token_ids': ids} if full_ids else {}
    if rtx:
        metadata['rtx'] = {'ids': ids[:4096], 'live_slots': 1, 'session_positions': 8192}
    obj = {'usage': {'completion_tokens': len(ids) if count is None else count},
           'strix': metadata, 'choices': [{'text': '', 'finish_reason': finish}]}
    data = event({'choices': [{'text': 'text', 'finish_reason': None}]})
    if error is not None:
        data += event({'error': error})
    return data + event(obj) + (b'data: [DONE]\n\n' if done else b'')


class StreamTests(unittest.TestCase):
    def read(self, data):
        with mock.patch.object(BENCH.urllib.request, 'urlopen', return_value=io.BytesIO(data)):
            return BENCH.stream('http://unused', [1, 2], 5000, 1)

    def test_complete_ids_beyond_diagnostic_limit(self):
        ids = list(range(5000))
        result = self.read(completion(ids))
        self.assertIsNone(result['error'])
        self.assertEqual(result['ids'], ids)
        self.assertEqual(result['tokens'], len(ids))

    def test_shared_ids_without_rtx_metadata(self):
        self.assertEqual(self.read(completion([7, 8], rtx=False))['ids'], [7, 8])

    def test_invalid_or_incomplete_streams(self):
        good = completion([7, 8])
        cases = {
            'missing_ids': completion([7, 8], full_ids=False),
            'wrong_count': completion([7, 8], count=3),
            'negative_count': completion([], count=-1),
            'non_integer_count': completion([7, 8], count=2.0),
            'invalid_ids': completion([7, '8']),
            'missing_done': completion([7, 8], done=False),
            'missing_finish': completion([7, 8], finish=None),
            'error_finish': completion([7, 8], finish='error'),
            'cancelled_finish': completion([7, 8], finish='cancelled'),
            'error_event': completion([7, 8], error={'message': 'failed'}),
            'malformed_json': b'data: broken\n\n' + good,
            'missing_usage': event({'strix': {'token_ids': [7, 8]},
                                    'choices': [{'finish_reason': 'stop'}]}) + b'data: [DONE]\n\n',
        }
        for name, data in cases.items():
            with self.subTest(name=name):
                self.assertTrue(self.read(data)['error'])


class BenchmarkTests(unittest.TestCase):
    def run_bench(self, reference, arm, prefill=None):
        with tempfile.TemporaryDirectory(prefix='.pr649-test-', dir=ROOT) as folder:
            prompt = Path(folder) / 'prompt.ids'
            prompt.write_text(' '.join(['1'] * 64), encoding='utf-8')
            output = Path(folder) / 'output.txt'
            argv = ['bench', 'http://unused', str(output), '--prompt-ids', str(prompt),
                    '--lens', '2' if prefill is not None else '', '--arms', '1',
                    '--dec-prompt', '2', '--max-tokens', '5000']
            data = ([prefill, prefill] if prefill is not None else []) + [reference, arm]
            # ExitStack owns every fake response even if a failed C1 stops the benchmark early.
            with contextlib.ExitStack() as stack:
                responses = [stack.enter_context(io.BytesIO(item)) for item in data]
                stack.enter_context(mock.patch('sys.argv', argv))
                stack.enter_context(mock.patch.object(BENCH, 'wait_health', return_value=0))
                stack.enter_context(mock.patch.object(BENCH.urllib.request, 'urlopen', side_effect=responses))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                status = BENCH.main()
            return status or 0, output.read_text(encoding='utf-8')

    def test_equal_complete_streams_succeed(self):
        data = completion(list(range(5000)))
        status, log = self.run_bench(data, data)
        self.assertEqual(status, 0)
        self.assertIn('same_as_C1=1/1', log)

    def test_difference_after_4096_fails(self):
        ids = list(range(5000))
        changed = ids.copy()
        changed[4096] += 1
        status, log = self.run_bench(completion(ids), completion(changed))
        self.assertNotEqual(status, 0)
        self.assertIn('same_as_C1=0/1', log)

    def test_different_lengths_after_4096_fail(self):
        ids = list(range(5000))
        status, log = self.run_bench(completion(ids), completion(ids[:-1]))
        self.assertNotEqual(status, 0)
        self.assertIn('same_as_C1=0/1', log)

    def test_invalid_reference_or_arm_never_counts_as_equal(self):
        good = completion([7, 8])
        for invalid in (completion([7, 8], full_ids=False), completion([7, 8], done=False),
                        completion([7, 8], count=3), completion([7, 8], error={'message': 'failed'})):
            for reference, arm in ((invalid, good), (good, invalid), (invalid, invalid)):
                with self.subTest(reference=reference, arm=arm):
                    status, log = self.run_bench(reference, arm)
                    self.assertNotEqual(status, 0)
                    self.assertNotIn('same_as_C1=1/1', log)

    def test_invalid_prefill_fails_run(self):
        good = completion([7, 8])
        status, _ = self.run_bench(good, good, prefill=completion([7, 8], done=False))
        self.assertNotEqual(status, 0)


if __name__ == '__main__':
    unittest.main()
