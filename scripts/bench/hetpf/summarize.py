#!/usr/bin/env python3
"""hetpf summary (<= 60 lines): python summarize.py RUN.jsonl [--mon LABEL=DIR ...] [--smi PC.samples] [--link LINK.json]

Groups by campaign, mode and arm, preserving attempt ordinals after failures.
A/B latency and throughput measure text chunks; R latency is response availability.
GPU activity integrates sampled busy*MHz with clipped intervals and full coverage.
All results remain provisional pending numerical fidelity and native qualification.
"""
import argparse
import json
import re
import statistics as st
import sys
from collections import defaultdict


def load_mon(path):
    rows = []
    try:
        for line in open(path, errors='replace'):
            m = re.match(r'(\d+) busy=(\d+) sclk=.*?(\d+)Mhz', line)
            if m:
                rows.append((int(m.group(1)) / 1e9, float(m.group(2)) / 100.0, float(m.group(3))))
    except OSError:
        pass
    return rows


def load_smi(path):
    rows = []
    try:
        for line in open(path, errors='replace'):
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 3 and parts[0].isdigit():
                rows.append((int(parts[0]) / 1e9, float(parts[1]) / 100.0, float(parts[2])))
    except (OSError, ValueError):
        pass
    return rows


def cycles(rows, t0, t1):
    """Left-held sample product integral; gaps > 1.5 s or missing edges are unavailable."""
    if t1 <= t0:
        return None, None
    total = clock = covered = 0.0
    for (start, busy, mhz), (end, _, _) in zip(rows, rows[1:]):
        dt = min(end, t1) - max(start, t0)
        if dt <= 0:
            continue
        if end <= start or end - start > 1.5:
            return None, None
        total += busy * mhz * dt / 1000
        clock += mhz * dt
        covered += dt
    if abs(covered - (t1 - t0)) > 1e-6:
        return None, None
    return total, clock / covered


def pair_cycles(mon, start, end):
    cs = [cycles(rows, start, end)[0] for rows in mon]
    return sum(cs) if len(cs) == 2 and all(c is not None for c in cs) else None


def valid(r):
    return (r.get('valid') is True and not r.get('error') and
            (r['arm'] != 'A' or r.get('continued') is True))


def cold_warm(rs):
    per = defaultdict(list)
    for r in rs:
        per[(r['run_id'], r['label'])].append(r)
    cold = [r for r in rs if r['nth_in_process'] == 1]
    warm = [max(v, key=lambda r: r['nth_in_process']) for v in per.values()
            if any(r['nth_in_process'] > 1 for r in v)]
    return cold, warm


def prefix(a, b):
    n = 0
    for x, y in zip(a or [], b or []):
        if x != y:
            break
        n += 1
    return n


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def f(x, d=3):
    return '-' if x is None else f'{x:.{d}f}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('jsonl')
    p.add_argument('--mon', action='append', default=[])
    p.add_argument('--smi')
    p.add_argument('--link')
    a = p.parse_args()
    recs = [json.loads(line) for line in open(a.jsonl) if line.strip()]
    if len({r.get('run_id') for r in recs}) != 1 or not all(r.get('run_id') for r in recs):
        raise ValueError('summary requires exactly one identified campaign')
    mons = {}
    for m in a.mon:
        lab, d = m.split('=', 1)
        mons[lab] = (load_mon(d + '/mon0.txt'), load_mon(d + '/mon1.txt'))
    smi = load_smi(a.smi) if a.smi else []
    if a.link:
        L = json.load(open(a.link))
        print(f"link: rtt median {L['rtt_us']['median']:.0f} us (p90 {L['rtt_us']['p90']:.0f}); upload {L['upload']['median_MB_s']:.1f} MB/s, "
              f"download {L['download']['median_MB_s']:.1f} MB/s ({L['upload']['median_Gbit_s']:.3f} / {L['download']['median_Gbit_s']:.3f} Gbit/s)")
    errs = [r for r in recs if not valid(r)]
    print('PROVISIONAL: KL/dNLL fidelity gate not implemented; clocks require synchronized hosts.')
    for r in errs[:4]:
        print(f"ERROR {r['label']} {r['arm']}#{r['nth_in_process']}: {r.get('error', 'invalid gate')[:160]}")
    for r in recs:  # derived per record
        s = (r.get('service') or {}).get('seconds') or {}
        if r['arm'] == 'R' and s:
            r['internal_first_estimate_s'] = (s.get('queue') or 0) + (s.get('prefill') or 0)
        r['latency_s'] = r.get('ttft_s') if r['arm'] == 'R' else r.get('first_text_latency_s')
        t0 = r['epoch_start']
        n = r['tokens']
        if r['arm'] in ('A', 'R') and smi and s:
            c, mhz = cycles(smi, t0, t0 + (s.get('queue') or 0) + (s.get('prefill') or 0))
            r['rtx_mclk_tok'] = c * 1e3 / n if c is not None else None
            r['rtx_mhz'] = mhz
        mon = mons.get(r['label'])
        if r['arm'] in ('A', 'B'):
            r['pair_clock_coverage'] = 'incomplete'
        if mon and r['arm'] in ('A', 'B') and r.get('first_text_s') is not None:
            start = r['stream_epoch_start']
            first = start + r['first_text_s']
            if r['arm'] == 'B':
                c = pair_cycles(mon, start, first)
                r['pair_pf_mclk_tok'] = c * 1e3 / n if c is not None else None
                r['pair_clock_coverage'] = 'complete' if c is not None else 'incomplete'
            if r.get('text_span_s') and r.get('n_chunks', 0) > 1:
                c = pair_cycles(mon, first, first + r['text_span_s'])
                r['pair_dec_mclk_chunk'] = c * 1e3 / (r['n_chunks'] - 1) if c is not None else None
                if c is None:
                    r['pair_clock_coverage'] = 'incomplete'
                elif r['arm'] == 'A':
                    r['pair_clock_coverage'] = 'complete'
    groups = defaultdict(list)
    for r in recs:
        if valid(r):
            groups[(r['run_id'], r['mode'] if r['arm'] != 'R' else 'rtx', r['arm'])].append(r)
    print('group | n | latency cold | latency warm | all latency | text chunks/s warm | Mclk/prompt-token RTX | pair to first text | Mclk/text-chunk')
    for (run_id, mode, arm), rs in sorted(groups.items()):
        cold, warm = cold_warm(rs)
        print(f"{run_id}/{mode}/{arm} | {len(rs)} | {f(med(r.get('latency_s') for r in cold))} | {f(med(r.get('latency_s') for r in warm))} | "
              f"{' '.join(f(r.get('latency_s'), 2) for r in rs)} | {f(med(r.get('text_chunks_s') for r in warm), 2)} | "
              f"{f(med(r.get('rtx_mclk_tok') for r in warm), 2)} | {f(med(r.get('pair_pf_mclk_tok') for r in warm), 2)} | "
              f"{f(med(r.get('pair_dec_mclk_chunk') for r in warm), 1)}")
    A = [r for r in recs if r['arm'] == 'A' and valid(r)]
    if A:
        print('A breakdown (s): label#n prefill export sha256 transfer import[sha parse queue relay device] first-text latency | bytes | continued')
        for r in A:
            s = r['service']['seconds']
            t = ((r['service'].get('import') or {}).get('timing')) or {}
            print(f"  {r['label']}#{r['nth_in_process']} {f(s.get('prefill'), 2)} {f(s.get('export'))} {f(s.get('sha256'))} "
                  f"{f(s.get('transfer'))} {f(s.get('import'))}[{f(t.get('sha256_s'))} {f(t.get('parse_s'))} {f(t.get('queue_s'))} "
                  f"{f(t.get('replicate_s'))} {f(t.get('device_s'))}] {f(r.get('first_text_s'))} {f(r.get('latency_s'), 2)} | "
                  f"{r['service'].get('state_bytes')} | {r.get('continued')}")
    for run_id in sorted({r['run_id'] for r in recs}):
        candidates = [r for r in recs if r['run_id'] == run_id and r['arm'] == 'B' and valid(r) and r.get('gen_ids')]
        ref = next((r for r in candidates if r['mode'] == 'exact'), None) or next(iter(candidates), None)
        if ref:
            print(f"{run_id} reference {ref['mode']}/B {ref['label']}")
            for (rid, mode, arm), rs in sorted(groups.items()):
                ids = [r['gen_ids'] for r in rs if r.get('gen_ids')]
                if rid == run_id and ids:
                    print(f"  {mode}/{arm}: prefix vs ref {[prefix(v, ref['gen_ids']) for v in ids]}")
    incomplete = sum(r.get('pair_clock_coverage') == 'incomplete' for r in recs)
    print(f'Clock coverage: {incomplete} incomplete pair windows; unavailable clocks shown as - (never zero).')
    estimates = [r.get('internal_first_estimate_s') for r in recs if r['arm'] == 'R' and valid(r)]
    print(f'R internal queue+prefill estimate (s): {f(med(estimates))}; table uses measured response latency.')
    nxt = {r['service'].get('next_token') for r in recs if r.get('service') and r['service'].get('next_token') is not None}
    shas = {str(r['service'].get('sha256'))[:16] for r in recs if r['arm'] in ('A', 'R') and r.get('service')}
    print(f"RTX next_token {sorted(nxt)} | state sha256 {sorted(shas)}")

    return 1 if errs else 0


if __name__ == '__main__':
    sys.exit(main())
