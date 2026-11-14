#!/usr/bin/env python3
"""CPU tests for tools/pflash_bench.py: synthetic fixtures and a local stub server, no dataset, no GPU."""
import importlib.util
import io
import json
import re
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('pflash_bench', ROOT / 'tools/pflash_bench.py')
B = importlib.util.module_from_spec(SPEC)
sys.modules['pflash_bench'] = B
SPEC.loader.exec_module(B)

SQUAD = {'data': [{'title': 'Rivers', 'paragraphs': [
    {'context': 'The Arlen river rises in the Morrow hills and reaches the sea at Port Hale.',
     'qas': [{'id': 'q1', 'question': 'Where does the Arlen river reach the sea?',
              'answers': [{'text': 'Port Hale'}, {'text': 'at Port Hale'}]},
             {'id': 'q2', 'question': 'Who named the river?', 'answers': [], 'is_impossible': True}]},
    {'context': 'The Tessin bridge was finished in 1871 by the engineer Odile Varn.',
     'qas': [{'id': 'q3', 'question': 'Who built the Tessin bridge?', 'answers': [{'text': 'Odile Varn'}]}]}]},
    {'title': 'Filler', 'paragraphs': [{'context': f'Filler paragraph number {n} about nothing in particular. ' * 6,
                                        'qas': []} for n in range(40)]}]}
HOTPOT = [{'_id': 'h1', 'question': 'Which town hosts the museum founded by the author of Grey Tides?',
           'answer': 'Lindholm',
           'context': [['Grey Tides', ['Grey Tides is a novel by Mara Quist.']],
                       ['Mara Quist', ['Mara Quist founded a museum ', 'in Lindholm.']],
                       ['Other', ['Unrelated text. ' * 20]]],
           'supporting_facts': [['Grey Tides', 0], ['Mara Quist', 1]]}]


def longbench_record(rid, answer, size):
    return {'_id': rid, 'domain': 'Code', 'difficulty': 'easy', 'length': 'short',
            'question': f'Which option is right for {rid}?', 'choice_A': 'alpha', 'choice_B': 'beta',
            'choice_C': 'gamma', 'choice_D': 'delta', 'answer': answer, 'context': 'x ' * (size // 2)}


def row(item_id, arm, correct, set_name='niah', ctx=16384, text=None, ttft=1.0, cached=0, receipt=None, error=''):
    return {'set': set_name, 'id': item_id, 'ctx': ctx, 'arm': arm, 'error': error, 'correct': correct,
            'text': text if text is not None else ('right' if correct else 'wrong'), 'ttft_s': ttft, 'total_s': ttft + 1,
            'server_ttft_ms': None, 'prompt_tokens': 16000 if arm == 'full' else 4800, 'cached_tokens': cached,
            'est_tokens': 16000, 'receipt': receipt}


COMPRESSED = {'raw': 16000, 'kept': 4800, 'reason': 'compressed', 'ms': 900.0}
BELOW = {'raw': 900, 'kept': 900, 'reason': 'below_threshold', 'ms': 0.0}


class StubServer:
    """Chat endpoint that answers the needle question; the compressed arm follows `mode`."""

    def __init__(self, mode='good', status=200):
        self.mode, self.status, self.bodies = mode, status, []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'

            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])).decode('utf-8'))
                stub.bodies.append(body)
                if stub.status != 200:
                    self.send_response(stub.status)
                    self.end_headers()
                    self.wfile.write(b'{"error":{"message":"busy"}}')
                    return
                user = body['messages'][-1]['content']
                key = re.search(r'magic number for (\S+) mentioned', user).group(1)
                number = re.search(rf'numbers for {re.escape(key)} is: (\d+)', user).group(1)
                compressed = body.get('prefill_compression') == 'auto'
                answer = number
                usage = {'prompt_tokens': len(user) // 4, 'prompt_tokens_details': {'cached_tokens': 0}}
                if compressed and stub.mode != 'ignore':
                    raw = len(user) // 4
                    if stub.mode == 'fallback':
                        receipt = {'raw': raw, 'kept': raw, 'reason': 'flat_scores', 'ms': 12.0}
                    else:
                        receipt = {'raw': raw, 'kept': raw * 3 // 10, 'reason': 'compressed', 'ms': 250.0}
                        usage['prompt_tokens'] = receipt['kept']
                        if stub.mode == 'lossy':
                            answer = 'I could not find it.'
                    usage['timings'] = {'pflash': receipt}
                self.send_response(200)
                if body.get('stream'):
                    self.send_header('Content-Type', 'text/event-stream')
                    self.end_headers()
                    chunks = [': progress {"phase":"prefill"}', '',
                              'data: ' + json.dumps({'choices': [{'delta': {'role': 'assistant'}}]}),
                              'data: ' + json.dumps({'choices': [{'delta': {'content': answer[:3]}}]}),
                              'data: ' + json.dumps({'choices': [{'delta': {'content': answer[3:]}}]}),
                              'data: ' + json.dumps({'choices': [], 'usage': usage}), 'data: [DONE]', '']
                    self.wfile.write('\n'.join(chunks).encode('utf-8'))
                else:
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    out = {'choices': [{'message': {'role': 'assistant', 'content': answer}}], 'usage': usage,
                           'timings': {'ttft_ms': 432.0}}
                    self.wfile.write(json.dumps(out).encode('utf-8'))

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.httpd.server_address[1]}'
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(5)


def run_main(argv):
    out, err = io.StringIO(), io.StringIO()
    code = B.main(argv, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


class SizeTests(unittest.TestCase):
    def test_context_sizes_parse_sorted_and_unique(self):
        self.assertEqual(B.parse_ctx('32k, 16k,120000,16K'), [16384, 32768, 120000])
        for bad in ('', '0k', '16m', 'x'):
            with self.assertRaises(B.BenchError):
                B.parse_ctx(bad)

    def test_bucket_is_the_smallest_that_holds_the_prompt(self):
        ctxs = [16384, 32768]
        self.assertEqual(B.bucket_for(16384, ctxs), (16384, ''))
        self.assertEqual(B.bucket_for(16385, ctxs), (32768, ''))
        self.assertEqual(B.bucket_for(8193, ctxs), (16384, ''))
        self.assertEqual(B.bucket_for(8192, ctxs), (None, 'too_short'))
        self.assertEqual(B.bucket_for(32769, ctxs), (None, 'too_long'))


class ScoringTests(unittest.TestCase):
    def test_sentence_choice_is_strict_unless_loose(self):
        self.assertEqual(B.extract_choice('**The correct answer is (C)**.'), 'C')
        self.assertEqual(B.extract_choice('The correct answer is B'), 'B')
        self.assertEqual(B.extract_choice('I think (D) fits best.'), '')
        self.assertEqual(B.extract_choice('A or maybe (D) fits best.', loose=True), 'D')
        self.assertEqual(B.extract_choice('The correct answer is (E)'), '')

    def test_final_choice_takes_the_stated_or_trailing_letter(self):
        self.assertEqual(B.extract_choice('Reasoning...\nfinal answer: b', style='final'), 'B')
        self.assertEqual(B.extract_choice('C\n', style='final'), 'C')
        self.assertEqual(B.extract_choice('The answer is D', style='final'), 'D')
        self.assertEqual(B.extract_choice('It is a panda', style='final'), '')
        self.assertEqual(B.extract_choice('No letter here.', style='final'), '')

    def test_contains_accepts_any_reference_and_ignores_case_and_spacing(self):
        item = B.Item('ruler', 'x', 16384, 'p', ['Port  Hale', 'the harbour'], 'contains')
        self.assertTrue(B.score(item, 'It reaches the sea at port hale.'))
        self.assertFalse(B.score(item, 'Port Vale'))
        self.assertFalse(B.score(B.Item('ruler', 'x', 16384, 'p', [''], 'contains'), 'anything'))
        choice = B.Item('longbench2', 'y', 16384, 'p', ['b'], 'choice')
        self.assertTrue(B.score(choice, 'The correct answer is (B)'))
        self.assertFalse(B.score(choice, 'The correct answer is (A)'))


class BuilderTests(unittest.TestCase):
    def test_niah_is_deterministic_and_places_one_needle_at_the_depth(self):
        a = B.build_niah([2048], [0, 50, 100], 2, 7, 4.0)
        b = B.build_niah([2048], [0, 50, 100], 2, 7, 4.0)
        self.assertEqual([i.prompt for i in a], [i.prompt for i in b])
        self.assertEqual(len(a), 6)
        self.assertNotEqual(a[0].prompt, B.build_niah([2048], [0], 1, 8, 4.0)[0].prompt)
        positions = {}
        for item in a:
            self.assertGreaterEqual(item.est_tokens, 2048)          # sits above a threshold of the bucket size
            self.assertLess(item.est_tokens, 2048 * 1.25)
            self.assertEqual(item.prompt.count('special magic numbers for'), 1)
            self.assertEqual(item.prompt.count(item.answers[0]), 1)
            positions.setdefault(item.meta['depth'], []).append(item.prompt.index(item.answers[0]) / len(item.prompt))
        self.assertLess(max(positions[0]), 0.1)
        self.assertTrue(all(0.4 < p < 0.6 for p in positions[50]))
        self.assertGreater(min(positions[100]), 0.9)

    def test_niah_can_use_a_haystack_file_text(self):
        hay = '\n\n'.join(f'Paragraph {n} of the haystack text.' for n in range(5))
        item = B.build_niah([512], [50], 1, 0, 4.0, hay)[0]
        self.assertIn('Paragraph 3 of the haystack text.', item.prompt)
        self.assertEqual(item.prompt.count(item.answers[0]), 1)

    def test_ruler_from_squad_keeps_the_gold_document_and_skips_unanswerable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'squad.json'
            path.write_text(json.dumps(SQUAD), encoding='utf-8')
            items = B.build_ruler(path, [1024], 10, 3, 4.0)
            self.assertEqual(sorted(i.id for i in items), ['ruler-1024-q1', 'ruler-1024-q3'])
            q1 = next(i for i in items if i.id.endswith('q1'))
            self.assertEqual(q1.answers, ['Port Hale', 'at Port Hale'])
            self.assertEqual(q1.prompt.count('The Arlen river rises'), 1)
            self.assertIn('Document 1:\n', q1.prompt)
            self.assertTrue(q1.prompt.endswith('Question: Where does the Arlen river reach the sea?'))
            self.assertGreater(q1.meta['documents'], 5)
            self.assertGreaterEqual(q1.est_tokens, 1024)
            self.assertEqual([i.prompt for i in items], [i.prompt for i in B.build_ruler(path, [1024], 10, 3, 4.0)])

    def test_ruler_from_hotpot_keeps_every_supporting_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'hotpot.json'
            path.write_text(json.dumps(HOTPOT), encoding='utf-8')
            item = B.build_ruler(path, [64], 5, 0, 4.0)[0]
            self.assertEqual(item.answers, ['Lindholm'])
            self.assertIn('Grey Tides is a novel by Mara Quist.', item.prompt)
            self.assertIn('Mara Quist founded a museum in Lindholm.', item.prompt)
            empty = Path(tmp) / 'empty.json'
            empty.write_text('[]', encoding='utf-8')
            with self.assertRaises(B.BenchError):
                B.build_ruler(empty, [64], 5, 0, 4.0)

    def test_choice_sets_bucket_by_size_and_count_what_they_skip(self):
        records = [longbench_record('a', 'B', 3000), longbench_record('b', 'C', 6000),
                   longbench_record('c', 'A', 100), longbench_record('d', 'D', 40000),
                   longbench_record('e', 'Z', 3000)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.json'
            path.write_text(json.dumps(records), encoding='utf-8')
            items, skipped = B.build_choice('longbench2', path, [1024, 2048], 0, 0, 4.0)
            self.assertEqual({i.id: i.ctx for i in items}, {'longbench2-a': 1024, 'longbench2-b': 2048})
            self.assertEqual(skipped, {'too_short': 1, 'too_long': 1, 'bad_record': 1})
            a = next(i for i in items if i.id == 'longbench2-a')
            self.assertEqual((a.answers, a.metric, a.meta['choice_style']), (['B'], 'choice', 'sentence'))
            self.assertIn('(B) beta\n', a.prompt)
            self.assertIn('"The correct answer is (insert answer here)"', a.prompt)
            lines = Path(tmp) / 'lcq.jsonl'
            lines.write_text('\n'.join(json.dumps({'prompt': 'repo text ' * 400 + f'Question {n}?', 'correct_letter': 'c'})
                                       for n in range(3)), encoding='utf-8')
            items, skipped = B.build_choice('longcodeqa', lines, [1024], 2, 0, 4.0)
            self.assertEqual((len(items), skipped), (2, {}))
            self.assertEqual((items[0].answers, items[0].meta['choice_style']), (['C'], 'final'))
            self.assertTrue(B.score(items[0], 'C'))


class StatisticsTests(unittest.TestCase):
    def test_receipt_is_found_in_every_accepted_place(self):
        want = {'raw': 100, 'kept': 30, 'reason': 'compressed', 'ms': 250.0}
        self.assertEqual(B.find_receipt({'timings': {'pflash': want}}, {}), want)
        self.assertEqual(B.find_receipt({'pflash': {'raw_tokens': 100, 'kept_tokens': 30, 'reason': 'compressed',
                                                    'seconds': 0.25}}, {}), want)
        self.assertEqual(B.find_receipt({}, {'pflash': want}), want)
        self.assertIsNone(B.find_receipt({'prompt_tokens': 5}, {}))
        self.assertIsNone(B.find_receipt(None, None))

    def test_exact_test_on_flipped_answers(self):
        self.assertEqual(B.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(B.mcnemar_exact(1, 1), 1.0)
        self.assertAlmostEqual(B.mcnemar_exact(5, 0), 2 / 32)
        self.assertAlmostEqual(B.mcnemar_exact(0, 5), 2 / 32)
        self.assertAlmostEqual(B.mcnemar_exact(3, 1), 2 * 5 / 16)

    def test_interval_is_seeded_and_brackets_the_change(self):
        diffs = [-1] * 6 + [1] * 2 + [0] * 92
        low, high = B.paired_interval(diffs, 2000, 1)
        self.assertEqual((low, high), B.paired_interval(diffs, 2000, 1))
        self.assertLessEqual(low, -4)
        self.assertGreaterEqual(high, -4)
        self.assertLess(low, high)
        self.assertGreaterEqual(low, -100)
        self.assertEqual(B.paired_interval([0] * 50, 2000, 1), (0, 0))
        self.assertEqual(B.paired_interval([], 2000, 1), (0, 0))
        self.assertEqual(B.paired_interval([-1] * 10, 500, 1), (-10, -10))

    def test_summary_counts_pairs_flips_fallbacks_and_cache_hits(self):
        rows = []
        for n in range(6):                                           # 6 pairs: 5 full correct, 4 compressed correct
            rows.append(row(f'i{n}', 'full', n != 5, ttft=10.0))
            rows.append(row(f'i{n}', 'pflash', n not in (0, 1), ttft=4.0, receipt=COMPRESSED))
        rows.append(row('f0', 'full', True, text='same', ttft=2.0))  # fallback that answers like the full arm
        rows.append(row('f0', 'pflash', True, text='same', ttft=2.0, receipt=BELOW))
        rows.append(row('f1', 'full', True, text='one', ttft=2.0))   # fallback that does not
        rows.append(row('f1', 'pflash', True, text='two', ttft=2.0, receipt=BELOW))
        rows.append(row('c0', 'full', True, ttft=10.0))              # reused prefix: out of TTFT
        rows.append(row('c0', 'pflash', True, ttft=0.1, cached=9000, receipt=COMPRESSED))
        rows.append(row('e0', 'full', True))                         # failed request: pair excluded
        rows.append(row('e0', 'pflash', False, error='http 500'))
        rows.append(row('n0', 'full', True))                         # server sent no receipt
        rows.append(row('n0', 'pflash', True))
        rows.append(row('o0', 'full', True))                         # one arm only
        cells = B.summarize(rows, 500, 0)
        self.assertEqual([(c['set'], c['ctx']) for c in cells], [('niah', 16384), ('niah', 'all'), ('all', 'all')])
        t = cells[-1]
        self.assertEqual((t['items'], t['pairs'], t['errors']), (12, 10, 1))
        self.assertEqual((t['full_correct'], t['pflash_correct'], t['delta']), (9, 8, -1))
        self.assertEqual((t['lost'], t['gained']), (2, 1))
        self.assertEqual((t['compressed'], t['fallback'], t['no_receipt'], t['fallback_mismatch']), (7, 2, 1, 1))
        self.assertEqual(t['reasons'], {'compressed': 7, 'below_threshold': 2})
        self.assertEqual((t['cache_hits'], t['ttft_pairs']), (1, 9))
        self.assertEqual((t['ttft_full_s'], t['ttft_pflash_s'], t['speedup']), (10.0, 4.0, 2.5))
        self.assertAlmostEqual(t['kept_fraction'], 0.3)
        self.assertEqual((t['drafter_ms'], t['prompt_tokens']), (900.0, 16000))
        self.assertLessEqual(t['ci_low'], t['delta'])
        self.assertGreaterEqual(t['ci_high'], t['delta'])
        report = B.render_report(cells, {'engine': 'stub', 'arms': ['full', 'pflash'], 'seed': 0})
        self.assertIn('| all | all | 16000 | 10 | 9 | 8 | -1 |', report)
        self.assertIn('1 pair(s) reused a cached prefix', report)
        self.assertIn('1 fallback answer(s) differ', report)
        self.assertEqual(len({line.count('|') for line in report.splitlines() if line.startswith('|')}), 1)

    def test_a_later_row_replaces_an_earlier_one(self):
        rows = [row('i0', 'full', True), row('i0', 'pflash', False, receipt=COMPRESSED),
                row('i0', 'pflash', True, receipt=COMPRESSED)]
        self.assertEqual(B.summarize(rows, 100, 0)[-1]['delta'], 0)

    def test_gate_verdicts(self):
        def total(n, lost, receipt=COMPRESSED, text=None):
            rows = []
            for i in range(n):
                rows.append(row(f'i{i}', 'full', True, text='same'))
                rows.append(row(f'i{i}', 'pflash', i >= lost, text=text or ('same' if i >= lost else 'x'),
                                receipt=receipt))
            return B.summarize(rows, 1000, 0)[-1]
        self.assertEqual(B.gate(total(100, 1), 0.02, 0.06), ('PASS', []))
        self.assertEqual(B.gate(total(100, 0), 0.02, 0.06)[0], 'PASS')
        verdict, why = B.gate(total(100, 5), 0.02, 0.06)
        self.assertEqual(verdict, 'FAIL')
        self.assertIn('accuracy change -0.050', why[0])
        verdict, why = B.gate(total(20, 1), 0.06, 0.06)              # mean passes, the small sample does not
        self.assertEqual((verdict, len(why)), ('FAIL', 1))
        self.assertIn('interval low end', why[0])
        self.assertEqual(B.gate(total(10, 0, receipt=None), 0.02, 0.06)[0], 'NOT_APPLIED')
        self.assertEqual(B.gate(total(10, 0, receipt=BELOW), 0.02, 0.06)[0], 'NOT_APPLIED')
        self.assertEqual(B.gate(B.summarize([row('i0', 'full', True)], 10, 0)[-1], 0.02, 0.06)[0], 'NO_DATA')
        mixed = [row('a', 'full', True, text='one'), row('a', 'pflash', True, text='two', receipt=BELOW),
                 row('b', 'full', True), row('b', 'pflash', True, receipt=COMPRESSED)]
        verdict, why = B.gate(B.summarize(mixed, 100, 0)[-1], 0.02, 0.06)
        self.assertEqual(verdict, 'FAIL')
        self.assertIn('fallback', why[0])


class EndToEndTests(unittest.TestCase):
    ARGS = ['--sets', 'niah', '--ctx', '1k', '--niah-depths', '10,90', '--niah-repeats', '2', '--resamples', '300']

    def test_stream_run_writes_report_receipt_and_rows_then_resumes(self):
        with StubServer('good') as stub, tempfile.TemporaryDirectory() as tmp:
            rows_path, receipt_path = Path(tmp) / 'rows.jsonl', Path(tmp) / 'receipt.json'
            argv = self.ARGS + ['--base-url', stub.url, '--engine', 'stub', '--rows', str(rows_path),
                                '--receipt', str(receipt_path), '--gate', '--extra-json', '{"top_k": 1}']
            code, out, err = run_main(argv)
            self.assertEqual(code, B.EXIT_OK, err)
            self.assertEqual(len(stub.bodies), 8)
            flagged = [b for b in stub.bodies if 'prefill_compression' in b]
            self.assertEqual(len(flagged), 4)
            self.assertTrue(all(b['prefill_compression'] == 'auto' for b in flagged))
            self.assertTrue(all(b['temperature'] == 0 and b['stream'] and b['top_k'] == 1 for b in stub.bodies))
            self.assertEqual(['prefill_compression' in b for b in stub.bodies[:4]], [False, True, True, False])
            by_user = {}
            for body in stub.bodies:                                 # both arms send the same user message
                by_user.setdefault(body['messages'][1]['content'], set()).add(body['messages'][0]['content'])
            self.assertEqual(len(by_user), 4)
            self.assertTrue(all(len(tags) == 2 for tags in by_user.values()))
            self.assertIn('| all | all |', out)
            self.assertIn('**Gate: PASS**', out)
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            self.assertEqual(receipt['config']['schema'], B.SCHEMA)
            total = receipt['cells'][-1]
            self.assertEqual((total['pairs'], total['full_correct'], total['pflash_correct']), (4, 4, 4))
            self.assertEqual((total['compressed'], total['no_receipt']), (4, 0))
            self.assertAlmostEqual(total['kept_fraction'], 0.3, places=2)
            self.assertIsNotNone(total['speedup'])
            self.assertEqual(receipt['gate']['verdict'], 'PASS')
            self.assertTrue(all('text' not in r for r in receipt['rows']))
            self.assertEqual(len(rows_path.read_text(encoding='utf-8').splitlines()), 8)
            code, out2, err = run_main(argv)                         # everything is already in the rows file
            self.assertEqual((code, len(stub.bodies)), (B.EXIT_OK, 8))
            self.assertIn('resuming: 8 request(s)', err)
            code, out3, _ = run_main(['--from-rows', str(rows_path), '--resamples', '300', '--gate'])
            self.assertEqual(code, B.EXIT_OK)
            self.assertEqual(out3.splitlines()[4:], out.splitlines()[4:])

    def test_lossy_answers_fail_the_gate(self):
        with StubServer('lossy') as stub:
            code, out, _ = run_main(self.ARGS + ['--base-url', stub.url, '--gate'])
            self.assertEqual(code, B.EXIT_GATE_FAIL)
            self.assertIn('| all | all | ', out)
            self.assertIn('| 4 | 4 | 0 | -4 | -4 .. -4 | 4 / 0 |', out)
            self.assertIn('**Gate: FAIL**', out)

    def test_a_server_that_ignores_the_field_is_reported_as_not_applied(self):
        with StubServer('ignore') as stub:
            code, out, _ = run_main(self.ARGS + ['--base-url', stub.url, '--gate', '--cache-bust', 'item'])
            self.assertEqual(code, B.EXIT_NOT_APPLIED)
            self.assertIn('**Gate: NOT_APPLIED**', out)
            tags = {}
            for body in stub.bodies:
                tags.setdefault(body['messages'][1]['content'], set()).add(body['messages'][0]['content'])
            self.assertTrue(all(len(t) == 1 for t in tags.values()))

    def test_fallback_replies_match_the_full_arm_and_do_not_count_as_applied(self):
        with StubServer('fallback') as stub:
            code, out, _ = run_main(self.ARGS + ['--base-url', stub.url, '--gate', '--no-stream'])
            self.assertEqual(code, B.EXIT_NOT_APPLIED)
            self.assertIn('receipt reasons: flat_scores=4', out)
            self.assertNotIn('fallback answer(s) differ', out)
            self.assertIn('| 0.43 | 0.43 | 1.00x |', out)            # server timings stand in for the client clock
            self.assertTrue(all(b['stream'] is False for b in stub.bodies))

    def test_request_errors_and_bad_usage_exit_2(self):
        with StubServer('good', status=503) as stub:
            code, out, err = run_main(self.ARGS + ['--base-url', stub.url])
            self.assertEqual(code, B.EXIT_ERRORS)
            self.assertIn('http 503', err)
            self.assertIn('8 request(s) failed', out)
        for argv in (['--sets', 'nope'], ['--sets', 'ruler'], ['--arms', 'full,fast'], ['--ctx', 'big'],
                     ['--extra-json', '{"prefill_compression": "auto"}'], ['--from-rows', 'no-such-rows.jsonl'],
                     ['--sets', 'longbench2', '--longbench2-file', 'no-such-file.json']):
            code, out, err = run_main(argv + ['--dry-run'])
            self.assertEqual(code, B.EXIT_ERRORS, argv)
            self.assertIn('pflash_bench:', err)

    def test_dry_run_calls_nothing(self):
        code, out, _ = run_main(['--sets', 'niah', '--ctx', '1k,2k', '--dry-run', '--base-url', 'http://127.0.0.1:9'])
        self.assertEqual(code, B.EXIT_OK)
        self.assertEqual(len(out.splitlines()), 2)
        self.assertIn('niah ctx=1024 items=10', out)


if __name__ == '__main__':
    unittest.main()
