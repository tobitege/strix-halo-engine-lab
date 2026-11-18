"""Prefill TTFT client for one running Qwen server (OpenAI /v1/completions with token-id prompts).

For every length: a cold request then a warm request, each on a distinct slice of the token file so a prefix
cache cannot hit; prints client TTFT and the server's own prefill_ms (strix.rtx / strix fields when present).
Usage: prefill_client.py BASE_URL IDS_FILE OUT_TXT [--lens 8192,32768] [--runs 2] [--max-tokens 4]
IDS_FILE: whitespace- or comma-separated token ids of one long text (tokenised for this model).
"""
import argparse
import json
import socket
import subprocess
import threading
import time
import urllib.request
import urllib.parse
from contextlib import contextmanager
from pathlib import Path


def require_alive(child):
    if child is not None and child.poll() is not None:
        raise RuntimeError('launched server exited')


@contextmanager
def server(base, command, log, limit):
    if not command:
        yield None
        return
    address = urllib.parse.urlsplit(base)
    if address.hostname != '127.0.0.1' or not address.port or limit <= 0:
        raise ValueError('launch requires a loopback port and a positive time limit')
    # Refuse an existing listener before launching anything. Never look up a port owner PID.
    with socket.socket() as probe:
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(('127.0.0.1', address.port))
    with open(log, 'wb') as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        # Popen owns the native Windows process handle, including across an early bind failure.
        def stop():
            if child.poll() is None:
                child.kill()
        timer = threading.Timer(limit, stop)
        timer.daemon = True
        timer.start()
        try:
            yield child
        finally:
            timer.cancel()
            timer.join()
            stop()
            child.wait(timeout=10)


def wait_health(base, limit, child=None):
    t0 = time.time()
    while time.time() - t0 < limit:
        require_alive(child)
        try:
            with urllib.request.urlopen(base + '/health', timeout=5) as r:
                if r.status == 200:
                    require_alive(child)
                    return time.time() - t0
        except OSError:
            pass
        time.sleep(2)
    raise SystemExit(f'server not healthy after {limit} s')


def stream(base, prompt, max_tokens, timeout):
    body = dict(prompt=prompt, max_tokens=max_tokens, temperature=0, stream=True, return_token_ids=True)
    req = urllib.request.Request(base + '/v1/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    first, strix, error = None, None, None
    cached = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.startswith(b'data:') or b'[DONE]' in line:
                continue
            try:
                obj = json.loads(line[5:])
            except ValueError:
                continue
            if obj.get('error'):
                error = obj['error']
            if isinstance(obj.get('strix'), dict):
                strix = obj['strix']
                for receipt in (strix, strix.get('rtx') or {}):
                    if 'prefix_cached' in receipt:
                        cached = max(cached or 0, int(receipt['prefix_cached']))
            details = (obj.get('usage') or {}).get('prompt_tokens_details') or {}
            if 'cached_tokens' in details:
                cached = max(cached or 0, int(details['cached_tokens']))
            if first is None and any((c.get('text') or '') for c in obj.get('choices') or []):
                first = time.perf_counter()
    ttft = (first - t0) if first else None
    if cached is None or cached != 0:
        raise ValueError(f'cache-free timing requires reported zero cached tokens (got {cached})')
    if error:
        raise RuntimeError(f'server error: {error}')
    receipt = strix or {}
    return dict(ttft=ttft, prefill_ms=receipt.get('prefill_ms', (receipt.get('rtx') or {}).get('prefill_ms')),
                error=error, cached_tokens=cached)


def prompt_slices(ids, lengths, runs):
    if runs < 1 or not lengths or any(n <= 0 for n in lengths):
        raise ValueError('lengths and runs must be positive')
    required = sum(lengths) * runs
    if len(ids) < required:
        raise ValueError(f'need {required} disjoint token IDs, got {len(ids)}')
    off = 0
    for length in lengths:
        for run in range(runs):
            prompt = ids[off:off + length]
            if len(prompt) != length:
                raise ValueError('truncated prompt slice')
            off += length
            yield run, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('ids')
    ap.add_argument('out')
    ap.add_argument('--lens', default='8192,32768')
    ap.add_argument('--runs', type=int, default=2, help='requests per length; the first is cold, the last warm')
    ap.add_argument('--max-tokens', type=int, default=4)
    ap.add_argument('--health-s', type=int, default=900)
    ap.add_argument('--limit-s', type=int, default=900)
    ap.add_argument('--server-log')
    ap.add_argument('--launch', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    ids = [int(x) for x in Path(a.ids).read_text().replace(',', ' ').split()]
    prompts = list(prompt_slices(ids, [int(x) for x in a.lens.split(',')], a.runs))
    with server(a.base, a.launch, a.server_log, a.limit_s) as child, open(a.out, 'a') as out:
        def say(s):
            print(s, flush=True)
            out.write(s + '\n')
            out.flush()
        say(f'health_s {wait_health(a.base, a.health_s, child):.1f}')
        for run, prompt in prompts:
            require_alive(child)
            r = stream(a.base, prompt, a.max_tokens, a.limit_s)
            require_alive(child)
            length = len(prompt)
            tag = 'cold' if run == 0 else ('warm' if run == a.runs - 1 else f'run{run}')
            pf = r['prefill_ms']
            say(f'pf L={length} {tag} ttft={r["ttft"] and round(r["ttft"], 3)} prefill_ms={pf} '
                f'tok/s={pf and round(length / (pf / 1e3))} cached_tokens={r["cached_tokens"]} err={r["error"]}')


if __name__ == '__main__':
    main()
