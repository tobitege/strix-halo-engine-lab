#!/usr/bin/env python3
"""Quality and TTFT harness for prefill compression (LOSSY class, per-request opt-in).

Runs every benchmark item twice against a running server, once per arm:
  full    today's request, the whole prompt is prefilled
  pflash  the same request plus "prefill_compression": "auto"
and reports, per set and context bucket, the paired accuracy difference with a 95 % interval, the answers that
flipped, the median time to first token of both arms and the share of the prompt that reached the target.

Sets: niah (generated here, no download), ruler (QA built from a SQuAD- or HotpotQA-format file), longbench2 and
longcodeqa (multiple choice, read from local files).  Nothing is downloaded by this tool.

  tools/pflash_bench.py --base-url http://127.0.0.1:8080 --engine glm --sets niah,ruler --ctx 16k,32k \
      --arms full,pflash --ruler-file squad.json --receipt receipt.json
  tools/pflash_bench.py --from-rows rows.jsonl --gate          # report again from saved rows
  tools/pflash_bench.py --sets niah --ctx 16k --dry-run        # build the prompts, call nothing

Exit codes: 0 ok / gate PASS, 2 usage or request errors, 3 gate FAIL, 4 compression was never applied.
Standard library only.
"""
import argparse
import hashlib
import http.client
import json
import math
import random
import re
import statistics
import sys
import time
import urllib.parse
from dataclasses import dataclass, field, asdict
from pathlib import Path

SCHEMA = 'strix-pflash-bench-v1'
SETS = ('niah', 'ruler', 'longbench2', 'longcodeqa')
ARMS = ('full', 'pflash')
REQUEST_FIELD = 'prefill_compression'
DEFAULT_MAX_TOKENS = {'niah': 48, 'ruler': 48, 'longbench2': 128, 'longcodeqa': 128}
EXIT_OK, EXIT_ERRORS, EXIT_GATE_FAIL, EXIT_NOT_APPLIED = 0, 2, 3, 4


class BenchError(Exception):
    pass


@dataclass
class Item:
    set: str
    id: str
    ctx: int                 # context bucket in tokens
    prompt: str              # user message
    answers: list            # accepted strings ('contains') or one letter ('choice')
    metric: str              # 'contains' | 'choice'
    est_tokens: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class Reply:
    text: str = ''
    ttft_s: float = None     # client clock: request sent -> first content
    total_s: float = None
    usage: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    error: str = ''


# ---------------------------------------------------------------- sizes and buckets

def parse_ctx(text):
    """'16k,32k,120000' -> [16384, 32768, 120000] (k = 1024), ascending, unique."""
    out = []
    for part in str(text).split(','):
        part = part.strip().lower()
        if not part:
            continue
        m = re.fullmatch(r'(\d+)(k?)', part)
        if not m or int(m.group(1)) == 0:
            raise BenchError(f'bad context size: {part!r}')
        out.append(int(m.group(1)) * (1024 if m.group(2) else 1))
    if not out:
        raise BenchError('no context size given')
    return sorted(set(out))


def est_tokens(text, chars_per_token):
    return int(math.ceil(len(text) / chars_per_token))


def bucket_for(tokens, ctxs):
    """Smallest bucket that holds the prompt.  Bucket c covers (previous, c]; the first one covers (c/2, c].
    Returns (ctx, '') or (None, 'too_short' | 'too_long')."""
    if tokens > ctxs[-1]:
        return None, 'too_long'
    if tokens <= ctxs[0] // 2:
        return None, 'too_short'
    for c in ctxs:
        if tokens <= c:
            return c, ''
    return None, 'too_long'


# ---------------------------------------------------------------- scoring

def normalize(text):
    return ' '.join(str(text).lower().split())


CHOICE_PATTERNS = (
    re.compile(r'The correct answer is \(([A-D])\)'),
    re.compile(r'The correct answer is ([A-D])'),
)
CHOICE_LOOSE = re.compile(r'(?<![A-Za-z])\(?([A-D])\)?(?![A-Za-z])')
FINAL_PATTERNS = (                                   # replies that state only a letter (longcodeqa)
    re.compile(r'Final Answer:\s*([ABCD])', re.IGNORECASE),
    re.compile(r'(?<![A-Za-z])([ABCD])\s*$', re.IGNORECASE),
)


def extract_choice(text, loose=False, style='sentence'):
    """Letter A-D, or '' when the reply does not state one.  style 'sentence' wants "The correct answer is (X)",
    style 'final' wants "Final Answer: X" or a trailing letter; loose also takes the last bare letter."""
    cleaned = str(text).replace('*', '')
    if style == 'final':
        cleaned = cleaned.strip()
    for pattern in (FINAL_PATTERNS if style == 'final' else CHOICE_PATTERNS):
        m = pattern.search(cleaned)
        if m:
            return m.group(1).upper()
    if loose:
        found = CHOICE_LOOSE.findall(cleaned)
        if found:
            return found[-1]
    return ''


def score(item, text, loose_choice=False):
    if item.metric == 'choice':
        style = item.meta.get('choice_style', 'sentence')
        return extract_choice(text, loose_choice, style) == str(item.answers[0]).strip().upper()
    if item.metric == 'contains':
        got = normalize(text)
        return any(normalize(a) and normalize(a) in got for a in item.answers)
    raise BenchError(f'unknown metric {item.metric!r}')


# ---------------------------------------------------------------- set builders

_WORDS = ('amber anchor basin birch canal cedar delta ember fennel garnet harbor island jasper kettle lantern meadow '
          'nectar orchard pebble quarry ribbon saddle timber umbra valley willow yarrow zephyr copper marble thistle '
          'bramble cobalt drift ferry glacier hollow ivory juniper kiln ledger mason north oxbow pier quill reef '
          'slate tundra upland vessel wharf yonder').split()
_VERBS = 'crosses follows guards holds joins keeps leaves marks meets passes reaches shelters turns watches'.split()
_NAMES = ('Avelin Brisco Calder Dunmore Estrel Farrow Galven Hadrow Iskar Jorvan Kestrel Lorne Marwick Norrin Ostrel '
          'Penhal Quenby Rastin Selwyn Tavish Ulmer Varden Wexley Yarmon Zennor').split()

NIAH_TEMPLATE = ('Some special magic numbers are hidden within the following text. Make sure to memorize them. '
                 'I will quiz you about a number afterwards.\n\n{context}\n\n'
                 'What is the special magic number for {key} mentioned in the provided text? '
                 'Answer with the number only.')
RULER_TEMPLATE = ('Answer the question based on the given documents. Only give me the answer and do not output any '
                  'other words.\n\nThe following are given documents.\n\n{context}\n\n'
                  'Answer the question based on the given documents. Only give me the answer and do not output any '
                  'other words.\n\nQuestion: {question}')
CHOICE_TEMPLATE = ('Please read the following text and answer the question below.\n\n<text>\n{context}\n</text>\n\n'
                   'What is the correct answer to this question: {question}\nChoices:\n(A) {A}\n(B) {B}\n(C) {C}\n'
                   '(D) {D}\n\nFormat your response as follows: "The correct answer is (insert answer here)".')


def _filler_paragraph(rng):
    sentences = []
    for _ in range(rng.randint(3, 6)):
        words = [rng.choice(_WORDS) for _ in range(rng.randint(7, 14))]
        words.insert(rng.randint(1, 3), rng.choice(_VERBS))
        sentences.append(' '.join(words).capitalize() + '.')
    return ' '.join(sentences)


def build_niah(ctxs, depths, repeats, seed, chars_per_token, haystack=None, fill=1.1):
    """One needle per item at a relative depth; filler is generated (seeded) or cut from `haystack` paragraphs."""
    items = []
    paragraphs = [p.strip() for p in haystack.split('\n\n') if p.strip()] if haystack else None
    for ctx in ctxs:
        budget = int(ctx * fill * chars_per_token) - len(NIAH_TEMPLATE)
        for depth in depths:
            for rep in range(repeats):
                rng = random.Random(f'niah|{seed}|{ctx}|{depth}|{rep}')
                key = f'{rng.choice(_NAMES)}-{rng.choice(_WORDS)}'
                number = str(rng.randint(1000000, 9999999))
                needle = f'One of the special magic numbers for {key} is: {number}.'
                body, size, start = [], 0, rng.randrange(len(paragraphs)) if paragraphs else 0
                while size < budget:
                    para = paragraphs[(start + len(body)) % len(paragraphs)] if paragraphs else _filler_paragraph(rng)
                    body.append(para)
                    size += len(para) + 2
                body.insert(min(len(body), round(len(body) * depth / 100)), needle)
                prompt = NIAH_TEMPLATE.format(context='\n\n'.join(body), key=key)
                items.append(Item('niah', f'niah-{ctx}-d{depth}-r{rep}', ctx, prompt, [number], 'contains',
                                  est_tokens(prompt, chars_per_token), {'depth': depth}))
    return items


def _read_json_records(path):
    """A JSON array, an object holding one array, or JSON lines."""
    text = Path(path).read_text(encoding='utf-8')
    stripped = text.lstrip()
    if stripped.startswith('['):
        return json.loads(text)
    if stripped.startswith('{'):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            for key in ('data', 'rows', 'examples'):
                if isinstance(data.get(key), list):
                    return data[key]
            return [data]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _qa_pool(records):
    """(questions, documents) from SQuAD format (articles with paragraphs) or HotpotQA format (context pairs)."""
    questions, docs, seen = [], [], {}

    def doc_id(text):
        if text not in seen:
            seen[text] = len(docs)
            docs.append(text)
        return seen[text]

    for rec in records:
        if 'paragraphs' in rec:                                   # SQuAD article
            for para in rec['paragraphs']:
                gold = doc_id(para['context'])
                for qa in para.get('qas', []):
                    answers = sorted({a['text'] for a in qa.get('answers', []) if a.get('text')})
                    if answers and not qa.get('is_impossible', False):
                        questions.append((str(qa.get('id', len(questions))), qa['question'], answers, [gold]))
        elif 'context' in rec and 'question' in rec:              # HotpotQA record
            by_title = {title: doc_id(f'{title}\n' + ''.join(sents)) for title, sents in rec['context']}
            support = sorted({by_title[t] for t, _ in rec.get('supporting_facts', []) if t in by_title})
            if rec.get('answer') and support:
                questions.append((str(rec.get('_id', len(questions))), rec['question'], [rec['answer']], support))
    return questions, docs


def build_ruler(path, ctxs, count, seed, chars_per_token, fill=1.1):
    """Gold document(s) plus distractor documents up to the bucket size, shuffled, numbered."""
    questions, docs = _qa_pool(_read_json_records(path))
    if not questions:
        raise BenchError(f'{path}: no answerable question found (SQuAD or HotpotQA format expected)')
    items = []
    for ctx in ctxs:
        rng = random.Random(f'ruler|{seed}|{ctx}')
        picked = rng.sample(range(len(questions)), min(count, len(questions)))
        budget = int(ctx * fill * chars_per_token) - len(RULER_TEMPLATE)
        for qi in picked:
            qid, question, answers, gold = questions[qi]
            chosen, size = list(gold), sum(len(docs[g]) + 16 for g in gold)
            order = [d for d in range(len(docs)) if d not in gold]
            rng.shuffle(order)
            for d in order:
                if size >= budget:
                    break
                chosen.append(d)
                size += len(docs[d]) + 16
            rng.shuffle(chosen)
            context = '\n\n'.join(f'Document {n + 1}:\n{docs[d]}' for n, d in enumerate(chosen))
            prompt = RULER_TEMPLATE.format(context=context, question=question)
            items.append(Item('ruler', f'ruler-{ctx}-{qid}', ctx, prompt, answers, 'contains',
                              est_tokens(prompt, chars_per_token), {'documents': len(chosen)}))
    return items


def _first(rec, names, default=None):
    for name in names:
        if name in rec and rec[name] not in (None, ''):
            return rec[name]
    return default


def _choice_item(set_name, rec, index, ctxs, chars_per_token, skipped):
    letter = str(_first(rec, ('answer', 'correct_letter', 'correct_answer', 'label'), '')).strip().upper().strip('()')
    question = _first(rec, ('question', 'query'))
    rid = str(_first(rec, ('_id', 'id', 'question_id'), index))
    prebuilt = _first(rec, ('prompt',)) if question is None else None
    if letter not in ('A', 'B', 'C', 'D') or (question is None and prebuilt is None):
        skipped['bad_record'] = skipped.get('bad_record', 0) + 1
        return None
    if prebuilt is not None:                                      # the file already holds the full prompt
        prompt = prebuilt
    else:
        choices = rec.get('choices') if isinstance(rec.get('choices'), (list, tuple)) else None
        opts = {k: (choices[i] if choices and i < len(choices) else _first(rec, (f'choice_{k}', k, k.lower()), ''))
                for i, k in enumerate('ABCD')}
        context = _first(rec, ('context', 'repo_text', 'document', 'text'), '')
        prompt = CHOICE_TEMPLATE.format(context=context, question=question, **opts)
    tokens = est_tokens(prompt, chars_per_token)
    ctx, why = bucket_for(tokens, ctxs)
    if ctx is None:
        skipped[why] = skipped.get(why, 0) + 1
        return None
    meta = {k: rec[k] for k in ('domain', 'sub_domain', 'difficulty', 'length', 'repo') if k in rec}
    meta['choice_style'] = 'final' if prebuilt is not None else 'sentence'
    return Item(set_name, f'{set_name}-{rid}', ctx, prompt, [letter], 'choice', tokens, meta)


def build_choice(set_name, path, ctxs, count, seed, chars_per_token):
    """Four-choice questions; each lands in the smallest bucket that holds it.  Returns (items, skipped counts)."""
    skipped, items = {}, []
    for index, rec in enumerate(_read_json_records(path)):
        item = _choice_item(set_name, rec, index, ctxs, chars_per_token, skipped)
        if item:
            items.append(item)
    out = []
    for ctx in ctxs:
        group = [i for i in items if i.ctx == ctx]
        random.Random(f'{set_name}|{seed}|{ctx}').shuffle(group)
        out.extend(group[:count] if count else group)
    return out, skipped


# ---------------------------------------------------------------- transport

def find_receipt(usage, timings):
    """The compression receipt of a reply, normalised to {raw, kept, reason, ms}; None when the server sent none."""
    for holder in ((usage or {}).get('timings'), usage, timings):
        if isinstance(holder, dict) and isinstance(holder.get('pflash'), dict):
            r = holder['pflash']
            raw, kept = _first(r, ('raw', 'raw_tokens')), _first(r, ('kept', 'kept_tokens'))
            ms = _first(r, ('ms', 'total_ms'))
            if ms is None and _first(r, ('seconds', 's')) is not None:
                ms = float(_first(r, ('seconds', 's'))) * 1000.0
            return {'raw': int(raw) if raw is not None else None, 'kept': int(kept) if kept is not None else None,
                    'reason': str(r.get('reason', '')), 'ms': float(ms) if ms is not None else None}
    return None


class HttpClient:
    """POST /v1/chat/completions; with stream=True the first content delta stops the TTFT clock."""

    def __init__(self, base_url, model='', timeout=3600.0, api_key='', stream=True):
        url = urllib.parse.urlsplit(base_url if '://' in base_url else 'http://' + base_url)
        if url.scheme not in ('http', 'https') or not url.hostname:
            raise BenchError(f'bad base url: {base_url!r}')
        self.scheme, self.host, self.port = url.scheme, url.hostname, url.port
        self.prefix = url.path.rstrip('/')
        self.model, self.timeout, self.api_key, self.stream = model, timeout, api_key, stream

    def complete(self, messages, max_tokens, extra=None):
        body = {'messages': messages, 'max_tokens': max_tokens, 'temperature': 0, 'stream': self.stream}
        if self.model:
            body['model'] = self.model
        body.update(extra or {})
        headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream' if self.stream else 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        reply = Reply()
        cls = http.client.HTTPSConnection if self.scheme == 'https' else http.client.HTTPConnection
        conn = cls(self.host, self.port, timeout=self.timeout)
        start = time.perf_counter()
        try:
            conn.request('POST', self.prefix + '/v1/chat/completions', json.dumps(body).encode('utf-8'), headers)
            resp = conn.getresponse()
            if resp.status != 200:
                reply.error = f'http {resp.status}: {resp.read(300).decode("utf-8", "replace")}'
                return reply
            if self.stream:
                self._read_stream(resp, reply, start)
            else:
                self._absorb(json.loads(resp.read().decode('utf-8')), reply, start, final=True)
        except (OSError, http.client.HTTPException, ValueError) as exc:
            reply.error = f'{type(exc).__name__}: {exc}'
        finally:
            reply.total_s = time.perf_counter() - start
            conn.close()
        return reply

    def _read_stream(self, resp, reply, start):
        for raw in resp:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:'):
                continue                                            # comments (progress) and blank lines
            data = line[5:].strip()
            if data == '[DONE]':
                break
            self._absorb(json.loads(data), reply, start, final=False)

    @staticmethod
    def _absorb(obj, reply, start, final):
        if isinstance(obj.get('error'), (dict, str)):
            reply.error = json.dumps(obj['error'])[:300]
            return
        for choice in obj.get('choices') or []:
            part = choice.get('message' if final else 'delta') or {}
            piece = part.get('content') or ''
            if (piece or part.get('reasoning_content')) and reply.ttft_s is None and not final:
                reply.ttft_s = time.perf_counter() - start
            reply.text += piece
        if isinstance(obj.get('usage'), dict):
            reply.usage = obj['usage']
        if isinstance(obj.get('timings'), dict):
            reply.timings = obj['timings']


# ---------------------------------------------------------------- run

def nonce(seed, item_id, arm, mode):
    if mode == 'off':
        return ''
    key = f'{seed}|{item_id}|{arm if mode == "arm" else ""}'
    return hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]


def build_messages(item, arm, seed, cache_bust, system):
    tag = nonce(seed, item.id, arm, cache_bust)
    head = system + (f' Session tag: {tag}.' if tag else '')
    return [{'role': 'system', 'content': head}, {'role': 'user', 'content': item.prompt}]


def run_one(client, item, arm, opts):
    extra = dict(opts.get('extra') or {})
    if arm == 'pflash':
        extra[REQUEST_FIELD] = 'auto'
    max_tokens = opts.get('max_tokens') or DEFAULT_MAX_TOKENS.get(item.set, 64)
    messages = build_messages(item, arm, opts.get('seed', 0), opts.get('cache_bust', 'arm'),
                              opts.get('system', 'You are a helpful assistant.'))
    reply = client.complete(messages, max_tokens, extra)
    usage = reply.usage or {}
    server_ttft = _first(reply.timings or {}, ('ttft_ms',))
    if server_ttft is None:
        server_ttft = _first(usage.get('timings') or {}, ('ttft_ms',))
    ttft = reply.ttft_s if reply.ttft_s is not None else (float(server_ttft) / 1000.0 if server_ttft is not None else None)
    row = {'set': item.set, 'id': item.id, 'ctx': item.ctx, 'arm': arm, 'error': reply.error,
           'correct': bool(not reply.error and score(item, reply.text, opts.get('loose_choice', False))),
           'text': reply.text, 'ttft_s': ttft, 'total_s': reply.total_s,
           'server_ttft_ms': float(server_ttft) if server_ttft is not None else None,
           'prompt_tokens': usage.get('prompt_tokens'),
           'cached_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0,
           'est_tokens': item.est_tokens, 'receipt': find_receipt(usage, reply.timings)}
    return row


def run(items, client, arms, opts, done=None, sink=None, log=None):
    """Both arms per item; the arm that goes first alternates so that neither is always the warm one."""
    rows = []
    done = done or set()
    for index, item in enumerate(items):
        order = list(arms) if index % 2 == 0 else list(reversed(arms))
        for arm in order:
            if (item.id, arm) in done:
                continue
            row = run_one(client, item, arm, opts)
            rows.append(row)
            if sink:
                sink(row)
            if log:
                ttft = f'{row["ttft_s"]:.2f}s' if row['ttft_s'] is not None else '-'
                log(f'[{index + 1}/{len(items)}] {item.id} {arm}: '
                    f'{"error " + row["error"][:80] if row["error"] else ("ok" if row["correct"] else "wrong")} ttft={ttft}')
    return rows


# ---------------------------------------------------------------- statistics

def mcnemar_exact(lost, gained):
    """Two-sided exact p for the discordant pairs (binomial, p = 1/2)."""
    n = lost + gained
    if n == 0:
        return 1.0
    k = min(lost, gained)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def paired_interval(diffs, resamples, seed):
    """95 % percentile interval of the summed paired difference (answers gained minus lost), seeded bootstrap."""
    n = len(diffs)
    if n == 0:
        return 0, 0
    if not any(diffs):
        return 0, 0
    rng = random.Random(f'boot|{seed}|{n}')
    sums = sorted(sum(rng.choices(diffs, k=n)) for _ in range(resamples))
    return sums[int(0.025 * (resamples - 1))], sums[int(math.ceil(0.975 * (resamples - 1)))]


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def pair_rows(rows):
    """{(set, ctx, id): {arm: row}}; a later row for the same key replaces an earlier one (resume)."""
    pairs = {}
    for row in rows:
        pairs.setdefault((row['set'], row['ctx'], row['id']), {})[row['arm']] = row
    return pairs


def summarize_group(pairs, resamples=10000, seed=0, cache_tolerance=64):
    cell = {'items': len(pairs), 'pairs': 0, 'errors': 0, 'full_correct': 0, 'pflash_correct': 0, 'lost': 0, 'gained': 0,
            'compressed': 0, 'fallback': 0, 'no_receipt': 0, 'fallback_mismatch': 0, 'cache_hits': 0, 'reasons': {}}
    diffs, ttft_full, ttft_pflash, ratios, kept, drafter_ms, sizes = [], [], [], [], [], [], []
    for arms in pairs.values():
        full, pf = arms.get('full'), arms.get('pflash')
        cell['errors'] += sum(1 for r in (full, pf) if r and r['error'])
        if not full or not pf or full['error'] or pf['error']:
            continue
        cell['pairs'] += 1
        cell['full_correct'] += full['correct']
        cell['pflash_correct'] += pf['correct']
        diffs.append(int(pf['correct']) - int(full['correct']))
        sizes.append(full.get('prompt_tokens'))
        cell['lost'] += full['correct'] and not pf['correct']
        cell['gained'] += pf['correct'] and not full['correct']
        receipt = pf.get('receipt')
        if not receipt:
            cell['no_receipt'] += 1
        else:
            cell['reasons'][receipt['reason']] = cell['reasons'].get(receipt['reason'], 0) + 1
            if receipt['reason'] == 'compressed':
                cell['compressed'] += 1
                if receipt['raw'] and receipt['kept'] is not None:
                    kept.append(receipt['kept'] / receipt['raw'])
                drafter_ms.append(receipt['ms'])
            else:
                cell['fallback'] += 1
                cell['fallback_mismatch'] += pf['text'] != full['text']   # a fallback must answer like the full arm
        if max(full['cached_tokens'], pf['cached_tokens']) > cache_tolerance:
            cell['cache_hits'] += 1                                    # a reused prefix makes TTFT incomparable
        elif full['ttft_s'] is not None and pf['ttft_s'] is not None:
            ttft_full.append(full['ttft_s'])
            ttft_pflash.append(pf['ttft_s'])
            if pf['ttft_s'] > 0:
                ratios.append(full['ttft_s'] / pf['ttft_s'])
    low, high = paired_interval(diffs, resamples, seed)
    cell.update(delta=cell['pflash_correct'] - cell['full_correct'], ci_low=low, ci_high=high,
                p_value=mcnemar_exact(cell['lost'], cell['gained']), ttft_pairs=len(ttft_full),
                ttft_full_s=_median(ttft_full), ttft_pflash_s=_median(ttft_pflash), speedup=_median(ratios),
                kept_fraction=_median(kept), drafter_ms=_median(drafter_ms), prompt_tokens=_median(sizes))
    for key in ('lost', 'gained', 'full_correct', 'pflash_correct', 'fallback_mismatch'):
        cell[key] = int(cell[key])
    return cell


def summarize(rows, resamples=10000, seed=0, cache_tolerance=64):
    pairs = pair_rows(rows)
    cells = []

    def add(label_set, label_ctx, keep):
        group = {k: v for k, v in pairs.items() if keep(k)}
        if group:
            cell = summarize_group(group, resamples, seed, cache_tolerance)
            cell.update(set=label_set, ctx=label_ctx)
            cells.append(cell)

    for s in sorted({k[0] for k in pairs}):
        for c in sorted({k[1] for k in pairs if k[0] == s}):
            add(s, c, lambda k, s=s, c=c: k[0] == s and k[1] == c)
        add(s, 'all', lambda k, s=s: k[0] == s)
    add('all', 'all', lambda k: True)
    return cells


def gate(total, max_mean_loss, max_low_loss):
    """Verdict on the 'all' cell.  Rates are fractions of the paired answers."""
    n = total['pairs']
    if n == 0:
        return 'NO_DATA', ['no completed pair']
    if total['compressed'] == 0:
        return 'NOT_APPLIED', ['no reply carried a receipt with reason=compressed '
                               '(server flag off, prompts below the threshold, or a server without the feature)']
    why = []
    if total['fallback_mismatch']:
        why.append(f'{total["fallback_mismatch"]} fallback answer(s) differ from the full arm')
    if total['delta'] / n < -max_mean_loss:
        why.append(f'accuracy change {total["delta"] / n:+.3f} is below -{max_mean_loss}')
    if total['ci_low'] / n < -max_low_loss:
        why.append(f'interval low end {total["ci_low"] / n:+.3f} is below -{max_low_loss}')
    return ('FAIL', why) if why else ('PASS', [])


# ---------------------------------------------------------------- report

def _num(value, fmt):
    return '-' if value is None else format(value, fmt)


def render_report(cells, config, verdict=None):
    lines = ['# Prefill compression: full vs compressed', '',
             f'engine `{config.get("engine", "-")}`, arms `{",".join(config.get("arms", ARMS))}`, '
             f'seed {config.get("seed", 0)}, greedy. Class: LOSSY, per-request opt-in.', '',
             '| set | ctx | prompt tok | pairs | full | compressed | change | 95 % interval | lost / gained | p | TTFT full s | '
             'TTFT compressed s | speed-up | kept | scorer ms | applied / fallback / none |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for c in cells:
        lines.append(
            f'| {c["set"]} | {c["ctx"]} | {_num(c["prompt_tokens"], ".0f")} | {c["pairs"]} | {c["full_correct"]} | {c["pflash_correct"]} | '
            f'{c["delta"]:+d} | {c["ci_low"]:+d} .. {c["ci_high"]:+d} | {c["lost"]} / {c["gained"]} | '
            f'{c["p_value"]:.3f} | {_num(c["ttft_full_s"], ".2f")} | {_num(c["ttft_pflash_s"], ".2f")} | '
            f'{_num(c["speedup"], ".2f")}x | {_num(c["kept_fraction"], ".3f")} | {_num(c["drafter_ms"], ".0f")} | '
            f'{c["compressed"]} / {c["fallback"]} / {c["no_receipt"]} |')
    total = cells[-1] if cells else None
    lines.append('')
    if total:
        notes = []
        if total['errors']:
            notes.append(f'{total["errors"]} request(s) failed and their pairs are excluded')
        if total['cache_hits']:
            notes.append(f'{total["cache_hits"]} pair(s) reused a cached prefix and are excluded from TTFT')
        if total['fallback_mismatch']:
            notes.append(f'{total["fallback_mismatch"]} fallback answer(s) differ from the full arm')
        if total['reasons']:
            notes.append('receipt reasons: ' + ', '.join(f'{k or "?"}={v}' for k, v in sorted(total['reasons'].items())))
        lines.extend(f'- {n}' for n in notes)
    lines.append('- change and interval are counts of answers (compressed minus full) over the same questions; '
                 'the interval is a seeded paired bootstrap; p is the exact test on the answers that flipped')
    lines.append('- speed-up is the median of the per-question TTFT ratios, measured on the client clock')
    if verdict:
        lines.append('')
        lines.append(f'**Gate: {verdict[0]}**' + (' (' + '; '.join(verdict[1]) + ')' if verdict[1] else ''))
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------- command line

def build_items(args, ctxs, log):
    items, skipped = [], {}
    sets = [s.strip() for s in args.sets.split(',') if s.strip()]
    for name in sets:
        if name not in SETS:
            raise BenchError(f'unknown set {name!r}; known: {", ".join(SETS)}')
        if name == 'niah':
            haystack = Path(args.niah_haystack).read_text(encoding='utf-8') if args.niah_haystack else None
            depths = [int(d) for d in args.niah_depths.split(',')]
            if any(d < 0 or d > 100 for d in depths):
                raise BenchError('--niah-depths are percentages 0..100')
            items += build_niah(ctxs, depths, args.niah_repeats, args.seed, args.chars_per_token, haystack, args.fill)
            continue
        path = {'ruler': args.ruler_file, 'longbench2': args.longbench2_file, 'longcodeqa': args.longcodeqa_file}[name]
        if not path:
            raise BenchError(f'set {name} needs --{name}-file (this tool downloads nothing)')
        if not Path(path).is_file():
            raise BenchError(f'{path}: no such file')
        if name == 'ruler':
            items += build_ruler(path, ctxs, args.count or 20, args.seed, args.chars_per_token, args.fill)
        else:
            got, skip = build_choice(name, path, ctxs, args.count, args.seed, args.chars_per_token)
            items += got
            skipped[name] = skip
            if skip:
                log(f'{name}: skipped ' + ', '.join(f'{k}={v}' for k, v in sorted(skip.items())))
    return items, skipped


def load_rows(path):
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def make_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--base-url', default='http://127.0.0.1:8080')
    p.add_argument('--model', default='')
    p.add_argument('--api-key', default='')
    p.add_argument('--engine', default='', help='label written to the receipt')
    p.add_argument('--sets', default='niah')
    p.add_argument('--ctx', default='16k,32k')
    p.add_argument('--arms', default='full,pflash')
    p.add_argument('--count', type=int, default=0, help='questions per set and bucket (0: all; ruler: 20)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--chars-per-token', type=float, default=4.0, help='prompt size estimate used for buckets')
    p.add_argument('--fill', type=float, default=1.1,
                   help='generated prompts (niah, ruler) aim at ctx x fill estimated tokens')
    p.add_argument('--niah-depths', default='0,25,50,75,100')
    p.add_argument('--niah-repeats', type=int, default=2)
    p.add_argument('--niah-haystack', default='', help='text file used as filler instead of generated text')
    p.add_argument('--ruler-file', default='', help='SQuAD- or HotpotQA-format JSON')
    p.add_argument('--longbench2-file', default='')
    p.add_argument('--longcodeqa-file', default='')
    p.add_argument('--max-tokens', type=int, default=0)
    p.add_argument('--system', default='You are a helpful assistant.')
    p.add_argument('--extra-json', default='', help='JSON object merged into every request body')
    p.add_argument('--cache-bust', choices=('arm', 'item', 'off'), default='arm',
                   help='tag in the system message so that one arm cannot reuse the prefix of the other')
    p.add_argument('--cache-tolerance', type=int, default=64, help='cached tokens above this drop the pair from TTFT')
    p.add_argument('--loose-choice', action='store_true', help='accept a bare last letter as the choice')
    p.add_argument('--no-stream', action='store_true', help='one JSON reply; TTFT then comes from server timings')
    p.add_argument('--timeout', type=float, default=3600.0)
    p.add_argument('--rows', default='', help='JSON lines of every request; an existing file is resumed')
    p.add_argument('--from-rows', default='', help='report from saved rows, call nothing')
    p.add_argument('--receipt', default='', help='write the JSON receipt here')
    p.add_argument('--report', default='', help='write the markdown report here (default: stdout)')
    p.add_argument('--resamples', type=int, default=10000)
    p.add_argument('--gate', action='store_true', help='turn the result into an exit code')
    p.add_argument('--gate-max-mean-loss', type=float, default=0.02, help='largest accepted accuracy loss (fraction)')
    p.add_argument('--gate-max-low-loss', type=float, default=0.06, help='largest accepted loss at the interval low end')
    p.add_argument('--dry-run', action='store_true', help='build the prompts and print their sizes')
    return p


def main(argv=None, client=None, out=None, err=None):
    out, err = out or sys.stdout, err or sys.stderr
    args = make_parser().parse_args(argv)

    def log(message):
        print(message, file=err, flush=True)

    try:
        arms = [a.strip() for a in args.arms.split(',') if a.strip()]
        if not arms or any(a not in ARMS for a in arms) or len(set(arms)) != len(arms):
            raise BenchError(f'--arms takes {",".join(ARMS)}')
        extra = json.loads(args.extra_json) if args.extra_json else {}
        if not isinstance(extra, dict) or REQUEST_FIELD in extra:
            raise BenchError(f'--extra-json must be an object without "{REQUEST_FIELD}" (the arm sets it)')
        config = {'schema': SCHEMA, 'engine': args.engine, 'model': args.model, 'arms': arms, 'seed': args.seed,
                  'sets': args.sets, 'ctx': args.ctx, 'chars_per_token': args.chars_per_token,
                  'cache_bust': args.cache_bust, 'request_field': REQUEST_FIELD, 'extra': extra}
        skipped = {}
        if args.from_rows:
            rows = load_rows(args.from_rows)
        else:
            ctxs = parse_ctx(args.ctx)
            items, skipped = build_items(args, ctxs, log)
            if not items:
                raise BenchError('no item to run')
            if args.dry_run:
                for s in sorted({i.set for i in items}):
                    for c in ctxs:
                        sizes = [i.est_tokens for i in items if i.set == s and i.ctx == c]
                        if sizes:
                            print(f'{s} ctx={c} items={len(sizes)} est_tokens min={min(sizes)} '
                                  f'median={int(statistics.median(sizes))} max={max(sizes)}', file=out)
                return EXIT_OK
            rows, done, sink, handle = [], set(), None, None
            if args.rows:
                if Path(args.rows).is_file():
                    rows = [r for r in load_rows(args.rows) if not r['error']]
                    done = {(r['id'], r['arm']) for r in rows}
                    log(f'resuming: {len(done)} request(s) already in {args.rows}')
                handle = open(args.rows, 'a', encoding='utf-8')

                def sink(row):
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                    handle.flush()
            client = client or HttpClient(args.base_url, args.model, args.timeout, args.api_key, not args.no_stream)
            opts = {'extra': extra, 'max_tokens': args.max_tokens, 'seed': args.seed, 'cache_bust': args.cache_bust,
                    'system': args.system, 'loose_choice': args.loose_choice}
            try:
                rows += run(items, client, arms, opts, done, sink, log)
            finally:
                if handle:
                    handle.close()
        cells = summarize(rows, args.resamples, args.seed, args.cache_tolerance)
        if not cells:
            raise BenchError('no row to report')
        total = cells[-1]
        verdict = gate(total, args.gate_max_mean_loss, args.gate_max_low_loss) if args.gate else None
        report = render_report(cells, config, verdict)
        if args.report:
            Path(args.report).write_text(report, encoding='utf-8')
        else:
            out.write(report)
        if args.receipt:
            receipt = {'config': config, 'skipped': skipped, 'cells': cells,
                       'gate': {'verdict': verdict[0], 'reasons': verdict[1]} if verdict else None,
                       'rows': [{k: v for k, v in r.items() if k != 'text'} for r in rows]}
            Path(args.receipt).write_text(json.dumps(receipt, indent=1) + '\n', encoding='utf-8')
        if verdict:
            if verdict[0] == 'NOT_APPLIED':
                return EXIT_NOT_APPLIED
            if verdict[0] != 'PASS':
                return EXIT_GATE_FAIL
        return EXIT_ERRORS if total['errors'] else EXIT_OK
    except (BenchError, OSError, json.JSONDecodeError, KeyError) as exc:
        log(f'pflash_bench: {exc}')
        return EXIT_ERRORS


if __name__ == '__main__':
    sys.exit(main())
