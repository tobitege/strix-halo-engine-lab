"""Qwen served decode by prompt class (prose / edit / json) at C1 / C4 / C8 against one running server.

Each class has 8 distinct prompts. Pass 1 ("cold": first requests of the process) runs the C1 reference of every
prompt alone, then C4 (prompts 0-3) and C8 (all 8) concurrently; pass 2 ("warm") repeats C1 for the first --warm-c1
prompts (default all 8, so cold and warm C1 means cover the SAME prompts), then C4 and C8.
A C1 request that ends before --min-decode-tokens (the model stops early, e.g. prose slice 0 ends with end-of-text
after 1 token) has no meaningful decode rate: its line carries short=1 and qwen_serve_summary.py leaves it out of the
C1 decode means (rtxmax3: averaging a 1-token request into a 2-prompt warm mean made warm look 22 % slower than cold).
Every concurrent stream must equal its C1 ids (same_as_C1). The ids of every request are written to OUT.ids (one JSON
line per request) so two server arms can be compared for identity.

prose = raw continuation of a token-id file (--prose-ids, comma/space separated ids); edit / json = chat turns
(Qwen template, thinking disabled) built from --prompts with the tokenizer given by --tokenizer (tokenizers package).
Hosts, model and file paths are arguments; nothing here is machine specific.
"""
import argparse
import json
import socket
import threading
import time
import urllib.request
from urllib.parse import urlsplit
from pathlib import Path


EXTRA = {}  # --extra-json: request fields merged into every body (measurement switches, e.g. {"copy_rows": 8})


def stream(base, prompt, max_tokens, timeout):
    body = dict(prompt=prompt, max_tokens=max_tokens, temperature=0, stream=True, return_token_ids=True)
    body.update(EXTRA)
    req = urllib.request.Request(base + '/v1/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    first, usage, strix, error = None, None, None, None
    done = False
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.startswith(b'data:'):
                continue
            if line[5:].strip() == b'[DONE]':
                done = True
                break
            try:
                obj = json.loads(line[5:])
            except ValueError:
                error = 'malformed SSE data'
                continue
            usage = obj.get('usage') or usage
            if 'error' in obj:
                error = obj['error'] or 'SSE error'
            if isinstance(obj.get('strix'), dict) and 'rtx' in obj['strix']:
                strix = obj['strix']
            if first is None and any((c.get('text') or '') for c in obj.get('choices') or []):
                first = time.perf_counter()
    t1 = time.perf_counter()
    rtx = (strix or {}).get('rtx', {})
    return dict(t_first=first, t_end=t1, ttft=(first - t0) if first else None, wall=t1 - t0,
                tokens=(usage or {}).get('completion_tokens'), ids=rtx.get('ids'), error=error, done=done,
                decode_tps=(strix or {}).get('decode_tokens_per_second'), tpc=rtx.get('tokens_per_cycle'), cf=rtx.get('copy_first_steps'),
                cx=rtx.get('context_extended_steps'), cr=rtx.get('context_rows'), hu=rtx.get('context_host_us'),
                rps=rtx.get('rows_per_step'), cw=rtx.get('copy_wide_steps'))


def response_error(result, max_tokens, reference=None):
    if result.get('error') is not None:
        return f'request error: {result["error"]}'
    if not result.get('done'):
        return 'missing SSE terminator'
    ids, tokens = result.get('ids'), result.get('tokens')
    if (not isinstance(ids, list) or not ids or
            any(type(token) is not int or not 0 <= token <= 0xffffffff for token in ids)):
        return 'missing or invalid token IDs'
    if type(tokens) is not int or tokens != len(ids) or not 0 < tokens <= max_tokens:
        return 'invalid completion token count'
    if reference is not None and ids != reference:
        return 'IDs differ from cold C1'
    return None


def probe_endpoint(base):
    """Refuse any existing listener, including a non-HTTP service. Arms are local."""
    url = urlsplit(base)
    if url.scheme != 'http' or url.hostname not in ('localhost', '127.0.0.1', '::1') or url.path not in ('', '/'):
        raise ValueError('ABAB requires a local HTTP base URL')
    port = url.port or 80
    for family, kind, proto, _, address in socket.getaddrinfo(url.hostname, port, type=socket.SOCK_STREAM):
        with socket.socket(family, kind, proto) as sock:
            sock.settimeout(2)
            if sock.connect_ex(address) == 0:
                raise RuntimeError(f'endpoint already occupied: {base}')
    return port


def barrier(n, fn):
    bar = threading.Barrier(n)
    out = [None] * n

    def one(i):
        bar.wait()
        try:
            out[i] = fn(i)
        except Exception as ex:  # noqa: BLE001
            out[i] = dict(error=str(ex))
    th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    t0 = time.perf_counter()
    for t in th:
        t.start()
    for t in th:
        t.join()
    return out, time.perf_counter() - t0


def wait_health(base, limit):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < limit:
        try:
            with urllib.request.urlopen(base + '/health', timeout=5) as r:
                if r.status == 200:
                    return time.perf_counter() - t0
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    raise SystemExit('server not healthy')


def chat_ids(tok, text):
    s = '<|im_start|>user\n' + text + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    return tok.encode(s, add_special_tokens=False).ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('out', nargs='?')
    ap.add_argument('--probe-endpoint', action='store_true')
    ap.add_argument('--prompts')
    ap.add_argument('--tokenizer')
    ap.add_argument('--prose-ids')
    ap.add_argument('--prose-len', type=int, default=1024)
    ap.add_argument('--classes', default='prose,edit,json')
    ap.add_argument('--max-tokens', type=int, default=128)
    ap.add_argument('--health-s', type=int, default=600)
    ap.add_argument('--warm-c1', type=int, default=8, help='C1 prompts repeated in the warm pass (same prompts as cold)')
    ap.add_argument('--min-decode-tokens', type=int, default=16, help='C1 requests with fewer tokens are flagged short=1')
    ap.add_argument('--conc', default='4,8', help='concurrent levels after C1 (comma list; empty = C1 only)')
    ap.add_argument('--extra-json', default='', help='JSON object merged into every request body (per-request switches)')
    a = ap.parse_args()
    if a.probe_endpoint:
        print(probe_endpoint(a.base))
        return
    if not all((a.out, a.prompts, a.tokenizer, a.prose_ids)) or a.max_tokens <= 0:
        ap.error('out, --prompts, --tokenizer, --prose-ids and positive --max-tokens are required')
    if a.extra_json:
        EXTRA.update(json.loads(a.extra_json))
    from tokenizers import Tokenizer  # local tokenizer.json, no download
    tok = Tokenizer.from_file(a.tokenizer)
    kit = json.loads(Path(a.prompts).read_text(encoding='utf-8'))
    prose = [int(x) for x in Path(a.prose_ids).read_text().replace(',', ' ').split()]
    sets = {
        'prose': [prose[i * (a.prose_len + 7): i * (a.prose_len + 7) + a.prose_len] for i in range(8)],
        'edit': [chat_ids(tok, 'Here is a Python module:\n\n```python\n' + kit['module'] + '```\n\n' + e + kit['edit_suffix']) for e in kit['edit']],
        'json': [chat_ids(tok, j) for j in kit['json']],
    }
    out = open(a.out, 'a')
    ids_out = open(a.out + '.ids', 'a')

    def say(*x):
        s = ' '.join(str(v) for v in x)
        print(s, flush=True)
        out.write(s + '\n')
        out.flush()
    say('health_s', round(wait_health(a.base, a.health_s), 1))
    with urllib.request.urlopen(a.base + '/v1/capabilities', timeout=5) as response:
        capabilities = json.load(response)
    Path(a.out + '.server.json').write_text(json.dumps(capabilities, indent=2), encoding='utf-8')
    failures = 0

    def check(result, reference=None):
        nonlocal failures
        error = response_error(result, a.max_tokens, reference)
        if error:
            failures += 1
            say('FAIL', error)
        return error is None

    for pas in ('cold', 'warm'):
        for cls in a.classes.split(','):
            P = sets[cls]
            ref = []
            for i, p in enumerate(P if pas == 'cold' else P[:a.warm_c1]):
                r = stream(a.base, p, a.max_tokens, 900)
                check(r, sets[cls + '_ref'][i] if pas == 'warm' else None)
                ref.append(r['ids'])
                ids_out.write(json.dumps(dict(pas=pas, cls=cls, c=1, i=i, ids=r['ids'])) + '\n')
                say(f'{pas} {cls} C1 i={i} prompt={len(p)} tokens={r["tokens"]} decode_tps={r["decode_tps"]} '
                    f'short={int((r["tokens"] or 0) < a.min_decode_tokens)} tpc={r["tpc"]} cf={r["cf"]} cx={r["cx"]} cr={r["cr"]} hu={r["hu"]} rps={r["rps"]} cw={r.get("cw")} err={r["error"]}')
            if pas == 'cold':
                sets[cls + '_ref'] = ref
            for c in [int(x) for x in a.conc.split(',') if x.strip()]:
                res, wall = barrier(c, lambda i: stream(a.base, P[i], a.max_tokens, 900))
                toks = sum((r.get('tokens') or 0) for r in res)
                same = sum(check(r, sets[cls + '_ref'][i]) for i, r in enumerate(res))
                first_end = max(r.get('t_first') or 0 for r in res)
                last = max(r.get('t_end') or 0 for r in res)
                dec = toks / (last - first_end) if last > first_end else None
                for i, r in enumerate(res):
                    ids_out.write(json.dumps(dict(pas=pas, cls=cls, c=c, i=i, ids=r.get('ids'))) + '\n')
                say(f'{pas} {cls} C{c} wall={wall:.2f} tokens={toks} agg_tps={toks / wall:.1f} '
                    f'agg_decode_tps_after_last_first={dec and round(dec, 1)} same_as_C1={same}/{c} '
                    f'errs={[r.get("error") for r in res if r.get("error")]}')
    ids_out.close()
    say('FAILED' if failures else 'done', 'failures', failures)
    out.close()
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
