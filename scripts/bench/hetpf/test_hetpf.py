"""Offline regressions for the measurement kit; no sockets, GPU or remote hosts."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
KIT = ROOT / 'scripts/bench/hetpf'
BASH = shutil.which('bash')  # Absolute path avoids Windows selecting the WSL launcher.


def load(name):
    spec = importlib.util.spec_from_file_location(name, KIT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


client, summary, probe = (load(n) for n in ('het_ab', 'summarize', 'link_probe'))


class FakeSocket:
    def __init__(self, data):
        self.data, self.sent, self.timeouts = data, [], []

    def setsockopt(self, *args):
        pass

    def settimeout(self, value):
        self.timeouts.append(value)

    def recv(self, n):
        data, self.data = self.data[:n], self.data[n:]
        return data

    def sendall(self, data):
        self.sent.append(data)


class ProbeTests(unittest.TestCase):
    def test_operation_caps(self):
        for op, count in ((b'd', 2**64 - 1), (b'u', 256 * probe.CHUNK + 1), (b'p', 10001)):
            sock = FakeSocket(op + probe.struct.pack('<Q', count))
            srv = types.SimpleNamespace(end=probe.time.monotonic() + 10, operation_timeout=2)
            probe.Handler(sock, None, srv)
            self.assertEqual(sock.sent, [])
        sock = FakeSocket(b'd' + probe.struct.pack('<Q', 3))
        probe.Handler(sock, None, srv)
        self.assertEqual(sock.sent, [b'\0' * 3])

    def test_absolute_connection_deadline(self):
        sock = FakeSocket(b'abc')
        with mock.patch.object(probe.time, 'monotonic', side_effect=[1, 4, 6]):
            wrapped = probe.DeadlineSocket(sock, 5)
            self.assertEqual(wrapped.recv(1), b'a')
            self.assertEqual(wrapped.recv(1), b'b')
            with self.assertRaises(TimeoutError):
                wrapped.recv(1)
        self.assertEqual(sock.timeouts, [4, 1])

    def test_concurrent_cap(self):
        srv = object.__new__(probe.ProbeServer)
        srv.slots = threading.BoundedSemaphore(1)
        srv.end = probe.time.monotonic() + 10
        srv.shutdown_request = mock.Mock()
        with mock.patch.object(probe.socketserver.ThreadingTCPServer, 'process_request') as launch:
            srv.process_request('first', None)
            srv.process_request('second', None)
        self.assertEqual(launch.call_count, 1)
        srv.shutdown_request.assert_called_once_with('second')

    def test_absolute_server_lifetime_and_bind(self):
        args = types.SimpleNamespace(bind='127.0.0.1', port=0, lifetime=2, operation_timeout=1, idle_exit=20)
        srv = mock.MagicMock(end=2, last=0)
        srv.handle_request.side_effect = lambda: setattr(srv, 'last', 100)
        with mock.patch.object(probe, 'ProbeServer') as factory, mock.patch.object(probe.time, 'monotonic', side_effect=[0, 1, 2]), contextlib.redirect_stdout(io.StringIO()):
            factory.return_value.__enter__.return_value = srv
            probe.serve(args)
        self.assertEqual(srv.handle_request.call_count, 2)
        args.bind = '0.0.0.0'
        with self.assertRaises(ValueError):
            probe.serve(args)


def terminal(ids=None):
    ids = [7, 8, 9] if ids is None else ids
    return {'id': 'response', 'choices': [{'finish_reason': 'length'}],
            'usage': {'prompt_tokens': 2, 'completion_tokens': len(ids)}, 'strix': {'token_ids': ids}}


def receipt():
    return {'id': 'request', 'response_id': 'response', 'status': 'ok', 'stop_reason': 'length',
            'prompt_tokens': 2, 'imported_tokens': 2, 'computed_tokens': 0, 'cached_tokens': 0}


class FakeServer:
    headers = {}

    def __init__(self, events, done=True):
        wire = b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in events)
        if done:
            wire += b'data: [DONE]\n\n'
        self.response = io.BytesIO(wire)
        self.response.status = 200
        self.connection = mock.Mock()
        self.connection.getresponse.return_value = self.response
        self.account = receipt()

    def conn(self):
        return self.connection

    def get(self, path):
        return 200, json.dumps({'requests': [self.account]}).encode()


class ClientTests(unittest.TestCase):
    def test_terminal_stream_gate(self):
        cases = [FakeServer([terminal()], done=False), FakeServer([]),
                 FakeServer([{'error': 'failed'}, terminal()]),
                 FakeServer([dict(terminal(), usage={})]),
                 FakeServer([dict(terminal(), choices=[])]),
                 FakeServer([terminal([])])]
        for srv in cases:
            with self.subTest(srv=srv), self.assertRaises(RuntimeError):
                client.stream(srv, [1, 2], 3, {})
            srv.connection.close.assert_called_once()

    def test_text_chunks_are_not_tokens(self):
        srv = FakeServer([{'choices': [{'text': 'combined UTF-8'}]}, {'choices': [{'text': 'tail'}]}, terminal()])
        rec = {}
        with mock.patch.object(client.time, 'perf_counter', side_effect=[0, 1, 3]):
            client.stream(srv, [1, 2], 3, rec)
        self.assertEqual(rec['n_generated'], 3)
        self.assertEqual(rec['n_chunks'], 2)
        self.assertEqual(rec['text_chunks_s'], .5)
        self.assertNotIn('decode_tok_s', rec)
        self.assertNotIn('ttft_s', rec)

    def test_handoff_gate(self):
        reply = {'next_token': 7, 'sha256': 'sha', 'digest': 'digest', 'import': {
            'imported': True, 'position': 2, 'next_token': 7, 'sha256': 'sha', 'digest': 'digest'}}
        rec = {'gen_ids': [7, 8], 'request_receipt': receipt()}
        client.validate_handoff(reply, rec, 2)
        self.assertTrue(rec['continued'])
        for field, value in [('computed_tokens', 2), ('imported_tokens', 0), ('cached_tokens', 2)]:
            bad = copy.deepcopy(rec)
            bad['request_receipt'][field] = value
            with self.assertRaises(RuntimeError):
                client.validate_handoff(reply, bad, 2)
        bad = dict(rec, gen_ids=[100])
        with self.assertRaises(RuntimeError):
            client.validate_handoff(reply, bad, 2)
        for field in reply['import']:
            bad_reply = copy.deepcopy(reply)
            del bad_reply['import'][field]
            with self.assertRaises(RuntimeError):
                client.validate_handoff(bad_reply, rec, 2)

    def test_missing_or_cancelled_accounting(self):
        for account in ({}, dict(receipt(), stop_reason='cancelled')):
            srv = FakeServer([terminal()])
            srv.account = account
            with mock.patch.object(client.time, 'sleep'), self.assertRaises(RuntimeError):
                client.stream(srv, [1, 2], 3, {})

    def test_client_campaign_failure_status(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            d = Path(directory)
            (d / 'key').write_text('test')
            (d / 'ids').write_text('1,2')
            args = ['het_ab', '--arms', 'R', '--run-id', 'run', '--label', 'R0', '--ids', str(d / 'ids'),
                    '--key', str(d / 'key'), '--tokens', '2', '--out', str(d / 'out')]
            with mock.patch.object(sys, 'argv', args), mock.patch.object(client, 'service_prefill', return_value=(200, {}, 1)), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(client.main(), 1)
            self.assertFalse(json.loads((d / 'out').read_text())['valid'])


class SummaryTests(unittest.TestCase):
    def run_summary(self, records):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / 'runs.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in records))
            out = io.StringIO()
            with mock.patch.object(sys, 'argv', ['summary', str(path)]), contextlib.redirect_stdout(out):
                status = summary.main()
            return status, out.getvalue()

    def row(self, **changes):
        r = dict(run_id='run', label='R0', arm='R', mode='-', nth_in_process=1, valid=True,
                 epoch_start=0, epoch_end=10, tokens=2, ttft_s=10,
                 service={'seconds': {'queue': .5, 'prefill': .5}})
        r.update(changes)
        return r

    def test_measured_r_latency(self):
        status, out = self.run_summary([self.row()])
        self.assertEqual(status, 0)
        self.assertIn('run/rtx/R | 1 | 10.000 | -', out)
        self.assertIn('estimate (s): 1.000', out)
        self.assertIn('PROVISIONAL', out)

    def test_failed_cold_stays_unavailable(self):
        status, out = self.run_summary([self.row(valid=False, error='stream failed'), self.row(nth_in_process=2, ttft_s=2)])
        self.assertEqual(status, 1)
        self.assertIn('run/rtx/R | 1 | - | 2.000', out)

    def test_invalid_handoff_excluded(self):
        status, out = self.run_summary([self.row(arm='A', continued=False)])
        self.assertEqual(status, 1)
        self.assertNotIn('run/-/A |', out)

    def test_campaigns_do_not_mix(self):
        with self.assertRaises(ValueError):
            self.run_summary([self.row(), self.row(run_id='other')])

    def test_correlated_clocks_and_coverage(self):
        rows = [(0, 0, 500), (.5, 1, 2500), (1, 1, 2500)]
        self.assertEqual(summary.cycles(rows, 0, 1), (1.25, 1500))
        self.assertEqual(summary.cycles(rows, .25, .75), (.625, 1500))
        self.assertEqual(summary.cycles(rows, -.1, 1), (None, None))
        self.assertEqual(summary.cycles([(0, 1, 500), (3, 1, 500)], 0, 3), (None, None))
        self.assertIsNone(summary.pair_cycles([rows, []], 0, 1))
        self.assertEqual(summary.pair_cycles([[(0, 0, 500), (1, 0, 500)]] * 2, 0, 1), 0)


def function(source, name):
    start = source.index(name + '() {')
    end = source.index('\n}', start) + 2
    return source[start:end]


@unittest.skipUnless(BASH, 'Bash unavailable')
class ShellTests(unittest.TestCase):
    def run_bash(self, script, *args):
        # Keep generated scripts and outputs inside the checkout, including on Windows.
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', dir=ROOT, delete=False, newline='\n') as f:
            f.write(script)
            path = Path(f.name)
        try:
            return subprocess.run([BASH, str(path), *map(str, args)], cwd=ROOT, capture_output=True, text=True, timeout=20)
        finally:
            path.unlink()

    def test_shell_syntax(self):
        for name in ('run-pc.sh', 'pair_serve.sh'):
            result = subprocess.run([BASH, '-n', str(KIT / name)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr[-1000:])

    def test_campaign_directory_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            marker = Path(directory) / 'ready'
            marker.write_text('old')
            # Preflight executes before environment checks or any network operation.
            r = subprocess.run([BASH, str(KIT / 'run-pc.sh'), directory], capture_output=True, text=True, timeout=10)
            self.assertEqual(r.returncode, 64, r.stderr[-1000:])
            self.assertEqual(marker.read_text(), 'old')

    def test_driver_cleanup_exit_and_signals(self):
        source = (KIT / 'run-pc.sh').read_text()
        for ending, code in [('exit 0', 0), ('kill -TERM $$', 143), ('kill -INT $$', 130)]:
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                script = 'OUT=$(cd "$1" && pwd); SP=; MP=; CP=; WP=\n'
                script += function(source, 'stop_svc') + '\n'
                script += 'stop_ticket() { echo stopped > "$OUT/ticket-stopped"; }; coord() { :; }\n'
                start = source.index('cleanup() {')
                script += source[start:source.index('# Background', start)]
                script += '''
sleep 60 & SP=$!
sleep 60 & MP=$!
sleep 60 & CP=$!
# EXIT cleanup must reap all three, then a second EXIT trap verifies no live child.
owned="$SP $MP $CP"
verify() { for p in $owned; do kill -0 "$p" 2>/dev/null && exit 99; done; }
'''
                # Use the production cleanup verbatim; check children after stop_svc via wrapper.
                script += '''
original_stop=$(declare -f stop_svc)
eval "${original_stop/stop_svc ()/original_stop_svc ()}"
stop_svc() { original_stop_svc; verify; }
''' + ending + '\n'
                r = self.run_bash(script, directory)
                self.assertEqual(r.returncode, code, r.stderr[-1000:])
                self.assertEqual((Path(directory) / 'status').read_text().strip(), str(code))
                self.assertTrue((Path(directory) / ('.done' if code == 0 else '.failed')).exists())
                self.assertTrue((Path(directory) / 'ticket-stopped').exists())

    def test_ticket_cleanup_on_signals(self):
        source = (KIT / 'pair_serve.sh').read_text()
        for signal, code in [('TERM', 143), ('INT', 130)]:
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                script = 'OUT=$(cd "$1" && pwd); M0=101; P0=102; P1=103; REMOTE_OWNED=1; PEER=peer; SSH=remote\n'
                script += function(source, 'cleanup') + '\n'
                # Record ownership signals without touching any real process or host.
                script += '''remote() { printf '%s\\n' "$2" >> "$OUT/remote"; }
kill() { printf '%s\\n' "$*" >> "$OUT/kills"; }
wait() { printf '%s\\n' "$*" >> "$OUT/waits"; }
scp() { :; }; sleep() { :; }; timeout() { shift 3; "$@"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
'''
                script += f'command kill -{signal} $$\n'
                r = self.run_bash(script, directory)
                self.assertEqual(r.returncode, code, r.stderr[-1000:])
                d = Path(directory)
                self.assertEqual((d / 'status').read_text().strip(), str(code))
                self.assertTrue((d / 'done').exists())
                self.assertIn('-TERM -- -102', (d / 'kills').read_text())
                self.assertIn('-KILL -- -101', (d / 'kills').read_text())
                self.assertIn('103', (d / 'waits').read_text())
                self.assertIn('m1 r1', (d / 'remote').read_text())

    def test_remote_monitor_pid_location(self):
        source = (KIT / 'pair_serve.sh').read_text()
        launch = next(line for line in source.splitlines() if '$SSH' in line and 'mon1.txt' in line and 'echo' in line)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            script = '''
BASE=$(cd "$1" && pwd); OUT=$BASE/local; REMOTE=$BASE/remote; mkdir "$OUT" "$REMOTE"
SSH=remote; PEER=peer; MON=:; TO=1
# Simulate separate filesystems without SSH/network access.
remote() { command=${2//"$OUT"/"$REMOTE"}; bash -c "$command"; }
'''+launch+'''
test -s "$REMOTE/m1.pid" && test ! -e "$OUT/m1.pid"
'''
            r = self.run_bash(script, directory)
            self.assertEqual(r.returncode, 0, r.stderr[-1000:])
        self.assertIn('timeout -k 2 $TO', launch)


if __name__ == '__main__':
    unittest.main()
