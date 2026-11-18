"""CUDA Qwen serve bench (strix-cuda-qwen-serve over /v1/completions, greedy, token ids).

Prefill TTFT per length (first request = cold, second = warm; distinct prompt slices so a prefix cache cannot hit), then
C1 references for N prompts and the concurrency arms; every stream must equal its C1 ids (same_as_C1). Each line also
carries the server's live sessions and per-request session capacity (strix.rtx stats) so slot admission is visible.

  python -I qwen_serve_bench.py BASE_URL OUT.txt --prompt-ids IDS.txt [--lens 8192] [--arms 1,4,8,16] [--max-tokens 128]
IDS.txt: whitespace/comma separated token ids (a long prose stream; slices of it are the prompts).
"""
import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path


def wait_health(base, limit):
    t0 = time.time()
    while time.time() - t0 < limit:
        try:
            with urllib.request.urlopen(base + '/health', timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    raise SystemExit('server never became healthy')


def stream(base, prompt, max_tokens, timeout):
    body = dict(prompt=prompt, max_tokens=max_tokens, temperature=0, stream=True, return_token_ids=True)
    req = urllib.request.Request(base + '/v1/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    first, usage, strix, error = None, None, {}, None
    done, finish = False, None
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
                error = error or 'invalid JSON in stream'
                continue
            if not isinstance(obj, dict):
                error = error or 'invalid stream event'
                continue
            usage = obj.get('usage') or usage
            if 'error' in obj:
                error = obj['error'] or 'stream error'
            if isinstance(obj.get('strix'), dict):
                strix.update(obj['strix'])
            for choice in obj.get('choices') or []:
                if choice.get('finish_reason') is not None:
                    finish = choice['finish_reason']
            if first is None and any((c.get('text') or '') for c in obj.get('choices') or []):
                first = time.perf_counter()
    t1 = time.perf_counter()
    ids = strix.get('token_ids')
    tokens = usage.get('completion_tokens') if isinstance(usage, dict) else None
    if not done or finish not in ('stop', 'length'):
        error = error or 'incomplete or unsuccessful stream'
    if not isinstance(ids, list) or any(type(token) is not int for token in ids):
        error = error or 'missing or invalid strix.token_ids'
    if type(tokens) is not int or tokens < 0:
        error = error or 'missing or invalid completion_tokens'
        tokens = None
    elif isinstance(ids, list) and len(ids) != tokens:
        error = error or 'token ID count differs from completion_tokens'
    rtx = strix.get('rtx', {})
    return dict(t0=t0, t_first=first, t_end=t1, ttft=(first - t0) if first else None, wall=t1 - t0,
                tokens=tokens, ids=ids, error=error,
                live=rtx.get('live_slots'), cap=rtx.get('session_positions'),
                prefill_ms=(strix or {}).get('prefill_ms'), decode_tps=(strix or {}).get('decode_tokens_per_second'))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('out')
    ap.add_argument('--prompt-ids', required=True)
    ap.add_argument('--lens', default='8192')
    ap.add_argument('--arms', default='1,4,8,16')
    ap.add_argument('--dec-prompt', type=int, default=1024)
    ap.add_argument('--max-tokens', type=int, default=128)
    ap.add_argument('--timeout', type=int, default=900)
    a = ap.parse_args()
    failed = False
    ids = [int(x) for x in Path(a.prompt_ids).read_text().replace(',', ' ').split()]
    out = open(a.out, 'a', encoding='utf-8')

    def say(*x):
        s = ' '.join(str(v) for v in x)
        print(s, flush=True)
        out.write(s + '\n')
        out.flush()
    say('health_s', round(wait_health(a.base, 600), 1))
    off = [0]

    def take(n):
        p = ids[off[0]:off[0] + n]
        off[0] += n + 7
        if len(p) < n:
            raise SystemExit('prompt id file too short')
        return p
    say('# prefill TTFT (s): client ttft, server prefill_ms; max_tokens 16; cold = first request of the length, warm = second')
    for L in [int(x) for x in a.lens.split(',') if x]:
        for run in ('cold', 'warm'):
            r = stream(a.base, take(L), 16, a.timeout)
            failed = failed or r['error'] is not None
            tps = round(L / (r['prefill_ms'] / 1e3)) if r['prefill_ms'] else None
            say(f'pf L={L} {run} ttft={r["ttft"] and round(r["ttft"], 3)} prefill_ms={r["prefill_ms"]} tok/s={tps} err={r["error"]}')
    arms = [int(x) for x in a.arms.split(',') if x]
    prompts = [take(a.dec_prompt) for _ in range(max(arms))]
    ref = []
    say(f'# decode: C1 refs, prompt {a.dec_prompt}, {a.max_tokens} tokens')
    with open(a.out + '.ids', 'a', encoding='utf-8') as f:
        for i, p in enumerate(prompts):
            r = stream(a.base, p, a.max_tokens, a.timeout)
            ref.append(r)
            failed = failed or r['error'] is not None
            f.write(json.dumps(r['ids']) + '\n')
            say(f'C1 i={i} tokens={r["tokens"]} decode_tps={r["decode_tps"]} ttft={r["ttft"] and round(r["ttft"], 3)} cap={r["cap"]} err={r["error"]}')
    for c in arms:
        res, wall = barrier(c, lambda i: stream(a.base, prompts[i], a.max_tokens, a.timeout))
        toks = sum((r.get('tokens') or 0) for r in res)
        same = sum(1 for i, r in enumerate(res)
                   if ref[i]['error'] is None and r.get('error') is None and r.get('ids') == ref[i]['ids'])
        failed = failed or same != c
        first_end = max(r.get('t_first') or 0 for r in res)
        last = max(r.get('t_end') or 0 for r in res)
        dec_agg = toks / (last - first_end) if last > first_end else None
        live = max((r.get('live') or 0) for r in res)
        say(f'C{c} wall={round(wall, 2)} tokens={toks} agg_tps={round(toks / wall, 1)} '
            f'agg_decode_tps_after_last_first={dec_agg and round(dec_agg, 1)} ttft_max={round(max(r.get("ttft") or 0 for r in res), 3)} '
            f'live_sessions_max={live} same_as_C1={same}/{c} errs={[r.get("error") for r in res if r.get("error")]}')
    say('FAILED' if failed else 'done')
    out.close()
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
