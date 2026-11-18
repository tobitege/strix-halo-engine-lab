#!/usr/bin/env python3
"""Hand-off owners probe: the prefill host's link with one copy per owner vs a copy per rank (ABAB).

Sends the per-rank byte streams of a family's prompt-state hand-off (strix/pair/state_owners.hpp; the byte counts are
the ones tests/pair_state_owners_tests.cpp pins) from this host to two sinks, one per rank, both at once over the same
NIC, in layer frames, and times first byte -> both sinks acknowledged the END frame. No GPU, no model: it measures
the link the planner's host_bytes term models. The relay leg (pair interconnect) is not emulated here: it is timed in
the ranks themselves (strix-minimax-rank prints "relay %.3f s").

  sink (on each rank host):  owners_probe.py sink --port P [--count K]
  send (on the prefill host): owners_probe.py send --to HOST0:P0,HOST1:P1 --family minimax|glm --tokens N [--pairs 3]

Arms: A = a copy per rank (MiniMax K/V/IK to both; GLM v1), B = one copy per owner (MiniMax IK once; GLM v2, DSA once).
Strict ABAB; the first arm of the run is reported as cold, the rest as warm.
"""
import argparse
import socket
import statistics
import struct
import sys
import threading
import time

FRAME = struct.Struct('<IIQQ')  # magic, layer, rows, bytes (the MiniMax hand-off frame)
END = 0xFFFFFFFF


def shapes(family, tokens):
    """Per rank: list of (layer frames) byte sizes for arm A and arm B."""
    if family == 'minimax':
        p = tokens - 1
        a = [[1280 * p] * 60, [1280 * p] * 60]
        b = [[1280 * p] * 60, [1024 * p] * 60]
        return a, b
    if family == 'glm':
        p = tokens - 1
        dsa = 1536 * p + 512 * (p // 4)          # latent + ikey + igate rows, FP32 pool per closed 4-row block
        kda = (4194304 + 294912) // 2            # one rank's KDA heads (S + conv)
        a = [[kda] * 34 + [dsa] * 12, [kda] * 34 + [dsa] * 12]
        b = [[kda] * 34 + [dsa] * 12, [kda] * 34]
        return a, b
    raise SystemExit('unknown family ' + family)


def recv_exact(s, n):
    out = bytearray()
    while len(out) < n:
        got = s.recv(n - len(out))
        if not got:
            raise SystemExit('peer closed')
        out += got
    return bytes(out)


def sink(args):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 << 20)
    srv.bind(('', args.port))
    srv.listen(4)
    buf = bytearray(64 << 20)
    view = memoryview(buf)
    for _ in range(args.count):
        c, _ = srv.accept()
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        total = 0
        while True:
            hdr = recv_exact(c, FRAME.size)
            if len(hdr) != FRAME.size:
                raise SystemExit('sink: short header')
            _, layer, _, n = FRAME.unpack(hdr)
            if layer == END:
                break
            left = n
            while left:
                got = c.recv_into(view, min(left, len(buf)))
                if got == 0:
                    raise SystemExit('sink: peer closed')
                left -= got
            total += n
        c.sendall(struct.pack('<Q', total))
        c.close()
    srv.close()


def send_one(host, port, frames, out, idx, t0):
    s = socket.create_connection((host, port), timeout=600)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32 << 20)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    payload = memoryview(bytearray(max(frames) if frames else 1))
    t0.wait()
    for layer, n in enumerate(frames):
        s.sendall(FRAME.pack(0x58524D4D, layer, 0, n))
        s.sendall(payload[:n])
    s.sendall(FRAME.pack(0x58524D4D, END, 0, 0))
    got = struct.unpack('<Q', recv_exact(s, 8))[0]
    out[idx] = (time.perf_counter(), got)
    s.close()


def send(args):
    to = [x.rsplit(':', 1) for x in args.to.split(',')]
    a, b = shapes(args.family, args.tokens)
    rows = []
    for k in range(2 * args.pairs):
        arm, frames = ('A', a) if k % 2 == 0 else ('B', b)
        out = [None, None]
        gate = threading.Event()
        th = [threading.Thread(target=send_one, args=(to[r][0], int(to[r][1]), frames[r], out, r, gate)) for r in range(2)]
        for t in th:
            t.start()
        time.sleep(0.3)  # both connections up
        t0 = time.perf_counter()
        gate.set()
        for t in th:
            t.join()
        sent = sum(sum(f) for f in frames)
        if any(o is None or o[1] != sum(frames[r]) for r, o in enumerate(out)):
            raise SystemExit('send: a sink acknowledged a different byte count')
        sec = max(o[0] for o in out) - t0
        rows.append((k, arm, sent, sec))
        print('%s %s tokens=%d arm=%s %s: %.1f MB in %.3f s = %.1f MB/s (rank0 %.1f MB, rank1 %.1f MB)' % (
            args.family, 'cold' if k == 0 else 'warm', args.tokens, arm, 'per-rank' if arm == 'A' else 'one-copy',
            sent / 1e6, sec, sent / 1e6 / sec, sum(frames[0]) / 1e6, sum(frames[1]) / 1e6), flush=True)
        time.sleep(args.gap)
    warm = [r for r in rows if r[0] > 0]
    for arm in 'AB':
        s = sorted(r[3] for r in warm if r[1] == arm)
        if s:
            print('%s warm arm %s: median %.3f s over %d (min %.3f max %.3f)' % (
                args.family, arm, statistics.median(s), len(s), s[0], s[-1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sk = sub.add_parser('sink')
    sk.add_argument('--port', type=int, required=True)
    sk.add_argument('--count', type=int, default=6)
    sd = sub.add_parser('send')
    sd.add_argument('--to', required=True, help='HOST0:PORT0,HOST1:PORT1 (rank order)')
    sd.add_argument('--family', choices=['minimax', 'glm'], required=True)
    sd.add_argument('--tokens', type=int, required=True)
    sd.add_argument('--pairs', type=int, default=3)
    sd.add_argument('--gap', type=float, default=1.0)
    args = ap.parse_args()
    sink(args) if args.cmd == 'sink' else send(args)


if __name__ == '__main__':
    sys.exit(main())
