#!/usr/bin/env python3
"""Useful TCP throughput and round-trip time of one host<->node link, both directions, standard library only.

  node:  python3 link_probe.py serve --bind ADDR --port P [--idle-exit S]
  host:  python3 link_probe.py probe --host ADDR --port P [--mib 64] [--reps 3] [--pings 200] [--state-bytes B] [--json OUT]

One connection per measurement; first byte = op, then an 8-byte little-endian count:
  'p' COUNT  ping-pong of COUNT 1-byte messages (TCP_NODELAY): the application-level RTT
  'u' BYTES  the host sends BYTES, the node answers 8 bytes when it has them all (upload, host -> node)
  'd' BYTES  the node sends BYTES (download, node -> host)
Throughput is bytes / (first byte sent .. last byte acknowledged or received), in MB/s (1e6 B/s) and Gbit/s.
--state-bytes prints the predicted upload time of a state of that size at the measured median rate.
The server has idle and absolute lifetime limits, bounded workers, counts and connection deadlines.
"""
import argparse
import json
import socket
import socketserver
import statistics
import struct
import sys
import threading
import time

CHUNK = 1 << 20


def recv_exact(s, n):
    buf = bytearray()
    while len(buf) < n:
        b = s.recv(min(CHUNK, n - len(buf)))
        if not b:
            raise ConnectionError('peer closed')
        buf += b
    return bytes(buf)


def drain(s, n):
    got = 0
    while got < n:
        b = s.recv(min(CHUNK, n - got))
        if not b:
            raise ConnectionError('peer closed')
        got += len(b)


class DeadlineSocket:
    """One absolute deadline, including slow clients that keep making progress."""
    def __init__(self, sock, deadline):
        self.sock, self.deadline = sock, deadline

    def call(self, method, arg):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError('probe deadline')
        self.sock.settimeout(left)
        return getattr(self.sock, method)(arg)

    def recv(self, n):
        return self.call('recv', n)

    def sendall(self, data):
        return self.call('sendall', data)


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s = DeadlineSocket(self.request, min(self.server.end, time.monotonic() + self.server.operation_timeout))
        try:
            op = recv_exact(s, 1)
            (n,) = struct.unpack('<Q', recv_exact(s, 8))
            if op not in (b'p', b'u', b'd') or n > (10000 if op == b'p' else 256 * CHUNK):
                return
            if op == b'p':
                for _ in range(n):
                    s.sendall(recv_exact(s, 1))
            elif op == b'u':
                drain(s, n)
                s.sendall(struct.pack('<Q', n))
            else:
                block = b'\0' * CHUNK
                while n:
                    k = min(CHUNK, n)
                    s.sendall(block[:k])
                    n -= k
        except (OSError, ConnectionError):
            pass


class ProbeServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, address, lifetime=180, operation_timeout=30, workers=4):
        self.slots = threading.BoundedSemaphore(workers)
        self.last = time.monotonic()
        self.end = self.last + lifetime
        self.operation_timeout = operation_timeout
        super().__init__(address, Handler)
        self.timeout = 0.2

    def process_request(self, request, address):
        if time.monotonic() >= self.end or not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        self.last = time.monotonic()
        try:
            super().process_request(request, address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()


def serve(a):
    if a.bind in ('0.0.0.0', '::', ''):
        raise ValueError('bind a specific interface address')
    with ProbeServer((a.bind, a.port), a.lifetime, a.operation_timeout) as srv:
        print(f'link_probe serving {a.bind}:{a.port}', flush=True)
        while time.monotonic() < min(srv.end, srv.last + a.idle_exit):
            srv.handle_request()


def connect(a):
    s = socket.create_connection((a.host, a.port), timeout=60)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s


def probe(a):
    out = {'run_id': a.run_id, 'host': a.host, 'port': a.port, 'mib': a.mib, 'reps': a.reps}
    s = connect(a)
    s.sendall(b'p' + struct.pack('<Q', a.pings))
    rtt = []
    for _ in range(a.pings):
        t0 = time.perf_counter()
        s.sendall(b'x')
        recv_exact(s, 1)
        rtt.append((time.perf_counter() - t0) * 1e6)
    s.close()
    rtt.sort()
    out['rtt_us'] = {'min': rtt[0], 'p10': rtt[len(rtt) // 10], 'median': statistics.median(rtt),
                     'p90': rtt[len(rtt) * 9 // 10], 'max': rtt[-1]}
    n = a.mib << 20
    block = b'\0' * CHUNK
    for op in ('u', 'd'):
        rates = []
        for _ in range(a.reps):
            s = connect(a)
            t0 = time.perf_counter()
            s.sendall(op.encode() + struct.pack('<Q', n))
            if op == 'u':
                left = n
                while left:
                    k = min(CHUNK, left)
                    s.sendall(block[:k])
                    left -= k
                recv_exact(s, 8)
            else:
                drain(s, n)
            dt = time.perf_counter() - t0
            s.close()
            rates.append(n / dt / 1e6)
        out['upload' if op == 'u' else 'download'] = {'MB_s': rates, 'median_MB_s': statistics.median(rates),
                                                      'median_Gbit_s': statistics.median(rates) * 8e-3}
    if a.state_bytes:
        out['state_upload_s'] = a.state_bytes / (out['upload']['median_MB_s'] * 1e6)
    r = out['rtt_us']
    print(f"rtt us min {r['min']:.0f} median {r['median']:.0f} p90 {r['p90']:.0f} | upload {out['upload']['median_MB_s']:.1f} MB/s "
          f"({out['upload']['median_Gbit_s']:.3f} Gbit/s) | download {out['download']['median_MB_s']:.1f} MB/s "
          f"({out['download']['median_Gbit_s']:.3f} Gbit/s)"
          + (f" | state {a.state_bytes / 1e6:.1f} MB -> {out['state_upload_s']:.3f} s" if a.state_bytes else ''), flush=True)
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('serve')
    s.add_argument('--bind', required=True)
    s.add_argument('--port', type=int, required=True)
    s.add_argument('--idle-exit', type=float, default=120)
    s.add_argument('--lifetime', type=float, default=180)
    s.add_argument('--operation-timeout', type=float, default=30)
    c = sub.add_parser('probe')
    c.add_argument('--host', required=True)
    c.add_argument('--run-id', required=True)
    c.add_argument('--port', type=int, required=True)
    c.add_argument('--mib', type=int, default=64)
    c.add_argument('--reps', type=int, default=3)
    c.add_argument('--pings', type=int, default=200)
    c.add_argument('--state-bytes', type=float, default=0)
    c.add_argument('--json')
    a = p.parse_args()
    serve(a) if a.cmd == 'serve' else probe(a)


if __name__ == '__main__':
    sys.exit(main())
