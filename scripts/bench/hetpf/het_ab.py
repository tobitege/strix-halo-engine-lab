#!/usr/bin/env python3
"""hetpf client: the same prompt through three routes, one JSON line per request (appended to --out).

  python het_ab.py --arms A,B,A,B --ids FILE --tokens N --key KEYFILE --label T1 --mode fast --out OUT.jsonl
                   [--service http://127.0.0.1:PORT] [--pair http://NODE:PORT] [--max-tokens 65]

  A  RTX -> pair: POST /v1/prefill on the CUDA prefill service with "handoff": PAIR/admin/import-state (the service runs
     the prefill, exports the state, hashes it and POSTs it; rank 0 relays it to rank 1, both import), then a streamed
     /v1/completions on the pair whose prompt is the same ids (it takes the parked state: first token = the state's next
     token, the rest are pair decode steps).
  B  pair alone: the streamed /v1/completions with the same ids (the pair prefills; --mode names its policy).
  R  RTX alone: the service prefill without a hand-off (its next_token is the first token).
Per record: epoch start/end (for the GPU clock monitors), wall and per-phase seconds, ttft_s, the generated ids, first-token
text latency and text chunks/s, the service and import replies (sha256 / digest / position) and the
server's accounting. Failed terminal or hand-off gates invalidate the sample and campaign.
Transport and ID loading reuse tools/ds41_hetero_import_driver.py.
"""
import argparse
import http.client
import json
import os
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'tools'))
from ds41_hetero_import_driver import Server, read_ids  # noqa: E402


def service_prefill(url, key, ids, session, handoff=None, reserve=0):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=1800)
    body = {'tokens': ids, 'session': session}
    if handoff:
        body.update({'handoff': handoff, 'api_key': key, 'reserve_tokens': reserve})
    h = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key, 'x-api-key': key}
    t0 = time.perf_counter()
    c.request('POST', '/v1/prefill', json.dumps(body), h)
    r = c.getresponse()
    reply = json.loads(r.read() or b'{}')
    c.close()
    return r.status, reply, time.perf_counter() - t0


def stream(srv, ids, max_tokens, rec):
    # The older driver's text helper discards DONE/error events. Use its transport
    # with a strict SSE reader so a truncated HTTP-200 cannot pass qualification.
    status, _ = srv.get('/metrics')  # arm existing request accounting
    if status != 200:
        raise RuntimeError('cannot arm request accounting')
    c = srv.conn()
    t0 = time.perf_counter()
    rec['stream_epoch_start'] = time.time()
    arrivals, final, finished, done = [], None, False, False
    try:
        c.request('POST', '/v1/completions', json.dumps({
            'prompt': ids, 'max_tokens': max_tokens, 'stream': True, 'temperature': 0,
            'return_token_ids': True, 'stream_options': {'include_usage': True}}),
            dict(srv.headers, **{'Content-Type': 'application/json'}))
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f'completions HTTP {r.status}')
        while True:
            line = r.readline()
            if not line:
                break
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                done = True
                break
            event = json.loads(data)
            if event.get('error'):
                raise RuntimeError(f'stream error: {event["error"]}')
            for choice in event.get('choices', []):
                if choice.get('text'):
                    arrivals.append(time.perf_counter() - t0)
                reason = choice.get('finish_reason')
                if reason is not None:
                    if reason not in ('stop', 'length'):
                        raise RuntimeError(f'stream finish: {reason}')
                    finished = True
            if 'token_ids' in (event.get('strix') or {}):
                final = event
        out_ids = (final or {}).get('strix', {}).get('token_ids')
        usage = (final or {}).get('usage') or {}
        if (not done or not finished or not out_ids or
                any(type(x) is not int for x in out_ids) or len(out_ids) > max_tokens or
                usage.get('completion_tokens') != len(out_ids) or usage.get('prompt_tokens') != len(ids)):
            raise RuntimeError('incomplete terminal IDs/usage/receipt')
    finally:
        c.close()
    rec.update(gen_ids=out_ids, n_generated=len(out_ids), n_chunks=len(arrivals),
               server_stats={k: v for k, v in final['strix'].items() if k != 'token_ids'})
    if arrivals:
        rec['first_text_s'] = arrivals[0]
        if len(arrivals) > 1 and arrivals[-1] > arrivals[0]:
            rec['text_span_s'] = arrivals[-1] - arrivals[0]
            rec['text_chunks_s'] = (len(arrivals) - 1) / rec['text_span_s']
    # finish_entry can be published just after DONE: bounded polling by response ID.
    for _ in range(20):
        status, body = srv.get('/api/status?detail=1')
        rows = json.loads(body).get('requests', []) if status == 200 else []
        receipt = next((v for v in rows if final.get('id') and final['id'] in (v.get('id'), v.get('response_id'))), None)
        if receipt and receipt.get('status') == 'ok' and receipt.get('stop_reason') in ('stop', 'length'):
            rec['request_receipt'] = receipt
            return
        time.sleep(0.1)
    raise RuntimeError('missing successful request accounting receipt')


def validate_handoff(reply, rec, count):
    imp = reply.get('import') or {}
    if (imp.get('imported') is not True or imp.get('position') != count or
            not reply.get('sha256') or imp.get('sha256') != reply['sha256'] or
            not reply.get('digest') or imp.get('digest') != reply['digest'] or
            imp.get('next_token') != reply.get('next_token')):
        raise RuntimeError('invalid import receipt')
    receipt = rec.get('request_receipt') or {}
    if (not rec.get('gen_ids') or rec['gen_ids'][0] != reply.get('next_token') or
            receipt.get('prompt_tokens') != count or receipt.get('imported_tokens') != count or
            receipt.get('computed_tokens') != 0 or receipt.get('cached_tokens') != 0):
        raise RuntimeError('request did not consume the complete imported prompt/next token')
    rec['continued'] = True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arms', required=True)
    p.add_argument('--ids', required=True)
    p.add_argument('--tokens', type=int, required=True)
    p.add_argument('--key', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--attempt-offset', type=int, default=0)
    p.add_argument('--mode', default='-', help='pair prefill policy of this process (exact|fast), recorded')
    p.add_argument('--out', required=True)
    p.add_argument('--service')
    p.add_argument('--pair')
    p.add_argument('--max-tokens', type=int, default=65)
    a = p.parse_args()
    key = open(a.key).read().strip()
    ids = read_ids(a.ids, a.tokens)
    srv = Server(a.pair, key) if a.pair else None
    seen = {}
    failed = False
    for i, arm in enumerate(a.arms.split(',')):
        seen[arm] = seen.get(arm, a.attempt_offset) + 1
        rec = {'run_id': a.run_id, 'valid': False, 'label': a.label, 'mode': a.mode, 'arm': arm, 'index': i, 'nth_in_process': seen[arm], 'tokens': len(ids),
               'epoch_start': time.time()}
        try:
            if arm in ('A', 'R'):
                handoff = a.pair.rstrip('/') + '/admin/import-state' if arm == 'A' else None
                st, reply, wall = service_prefill(a.service, key, ids, f'{a.run_id}-{a.label}-{i}-{arm}', handoff, a.max_tokens + 16)
                rec.update({'status': st, 'service_wall_s': wall, 'service': {k: reply.get(k) for k in (
                    'next_token', 'logits_fnv', 'state_bytes', 'sha256', 'digest', 'seconds', 'fast', 'import', 'error')}})
                if st != 200:
                    raise RuntimeError(f'prefill HTTP {st}: {str(reply.get("error"))[:300]}')
                if type(reply.get('next_token')) is not int:
                    raise RuntimeError('missing service next_token')
                if arm == 'R':
                    rec['ttft_s'] = wall
                else:
                    stream(srv, ids, a.max_tokens, rec)
                    rec['first_text_latency_s'] = wall + rec['first_text_s'] if 'first_text_s' in rec else None
                    validate_handoff(reply, rec, len(ids))
            elif arm == 'B':
                stream(srv, ids, a.max_tokens, rec)
                rec['first_text_latency_s'] = rec.get('first_text_s')
            else:
                raise ValueError('arm must be A, B or R')
            rec['valid'] = True
        except Exception as e:  # recorded, the next arm still runs
            failed = True
            rec['error'] = f'{type(e).__name__}: {e}'[:400]
        rec['epoch_end'] = time.time()
        with open(a.out, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(f"{a.label} {arm}#{seen[arm]} valid={rec['valid']} {rec.get('error', '')}", flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
