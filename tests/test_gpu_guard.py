"""Host-only regression fixtures: python tests/test_gpu_guard.py (no GPU or compiler needed)."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/bench/rtx/gpu_guard.py'
spec = importlib.util.spec_from_file_location('gpu_guard', SOURCE)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='guard-test-', dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.a = SimpleNamespace(out=str(self.path), latch=str(self.path / 'STOP'),
                                 stale_s=0.3, timeout_s=3, tlimit_min_c=5, temp_max_c=88,
                                 replay_every_s=0.1, quiet_before_s=60, post_s=0.1,
                                 cmd=[sys.executable, '-c', 'pass'])
        self.g = guard.Guard(self.a)
        self.g.uuid = 'GPU-test'
        self.g.log = Mock()
        if not hasattr(guard.signal, 'SIGKILL'):
            self.enterContext(patch.object(guard.signal, 'SIGKILL', 9, create=True))

    def row(self, temp='40', margin='50', watts='100', sm='1500'):
        return f'now,GPU-test,P0,{temp},{margin},{watts},300,95,0,1000,{sm},1000,0,4,16\n'

    def receipt(self, g=None, code=0):
        with contextlib.redirect_stdout(io.StringIO()):
            (g or self.g).summary(0, code, 'start', 'end')
        return json.loads((self.path / 'guard.json').read_text())

    def run_main(self, runner=None, extra=(), memory='0'):
        with patch.object(guard, 'resolve', return_value=('GPU-test', 'test GPU')), \
             patch.object(guard, 'smi', return_value=(0, memory)), \
             patch.object(guard, 'supervise', side_effect=runner) as supervised, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = guard.main(['--gpu-name', 'test', '--out', str(self.path), '--latch', self.a.latch,
                             *extra, '--', sys.executable, '-c', 'pass'])
        return rc, supervised

    def test_first_trip_still_kills_late_published_child(self):
        with patch.object(guard, 'WIN', False), patch.object(guard.os, 'killpg', create=True) as kill:
            self.g.fire(90, 'first')
            self.g.child = SimpleNamespace(pid=123, poll=lambda: None)
            self.g.fire(94, 'later')
        kill.assert_called_once_with(123, guard.signal.SIGKILL)
        self.assertEqual(self.g.trip, (90, 'first'))
        self.assertEqual((self.path / 'STOP').read_text().count('STOP'), 1)

    def test_trip_during_publication_kills_new_child(self):
        child = SimpleNamespace(pid=123)
        def spawn(*args, **kwargs):
            self.g.fire(90, 'during launch')
            return child
        with patch.object(guard, 'WIN', False), patch.object(guard.subprocess, 'Popen', side_effect=spawn), \
             patch.object(guard.os, 'killpg', create=True) as kill, patch.object(self.g, 'watch'):
            self.g.launch({}, re.compile('FAIL'))
        kill.assert_called_once_with(123, guard.signal.SIGKILL)

    def test_event_error_is_fatal(self):
        self.g.event_reader(SimpleNamespace(stdout=io.StringIO('ERR access denied\n')))
        self.assertEqual(self.g.trip[0], 91)
        self.assertIsNone(self.g.last_hb)

    @unittest.skipUnless(shutil.which('powershell') or shutil.which('pwsh'), 'PowerShell unavailable')
    def test_event_query_errors_and_empty_set(self):
        # Execute the actual watcher against a local Get-WinEvent fixture, without querying the host log.
        shell = shutil.which('powershell') or shutil.which('pwsh')
        watcher = str(SOURCE.with_name('winevents.ps1')).replace("'", "''")
        for error_id, healthy in [('NoMatchingEventsFound', True), ('AccessDenied', False), ('ServiceFailure', False)]:
            with self.subTest(error_id=error_id):
                harness = self.path / 'events.ps1'
                harness.write_text("""
function Get-WinEvent {
    [CmdletBinding()] param($FilterHashtable)
    $e = [System.Management.Automation.ErrorRecord]::new(
        [System.InvalidOperationException]::new('fixture query error'),
        'ERROR_ID', [System.Management.Automation.ErrorCategory]::InvalidOperation, $null)
    $PSCmdlet.ThrowTerminatingError($e)
}
function Start-Sleep { param($Milliseconds) exit 0 }
& 'WATCHER'
exit $LASTEXITCODE
""".replace('ERROR_ID', error_id).replace('WATCHER', watcher), encoding='utf-8')
                p = subprocess.run([shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(harness)],
                                   capture_output=True, text=True, timeout=15)
                self.assertEqual(p.returncode, 0 if healthy else 1, p.stderr)
                self.assertEqual('HB ' in p.stdout, healthy, p.stdout)
                self.assertEqual('ERR ' in p.stdout, not healthy, p.stdout)

    def test_invalid_regex_starts_no_process(self):
        with patch.object(guard.subprocess, 'Popen') as spawn:
            rc, supervised = self.run_main(extra=('--fail-pattern', '['))
        self.assertEqual(rc, 96)
        spawn.assert_not_called()
        supervised.assert_not_called()
        self.assertIn('guard failed', json.loads((self.path / 'guard.json').read_text())['reason'])

    def test_exception_and_interrupt_reap_owned_processes(self):
        for error, expected in [(RuntimeError('fixture'), 97), (KeyboardInterrupt(), 130)]:
            with self.subTest(error=type(error).__name__):
                (self.path / 'STOP').unlink(missing_ok=True)
                owned = []
                def runner(g, uuid, pat):
                    g.a.cmd = [sys.executable, '-c', 'import time; time.sleep(60)']
                    monitor = subprocess.Popen(g.a.cmd, stdout=subprocess.PIPE)
                    g.monitors.append((monitor, 'fixture'))
                    g.launch(dict(os.environ), pat)
                    owned.extend([g, monitor, g.child])
                    raise error
                rc, _ = self.run_main(runner)
                self.assertEqual(rc, expected)
                self.assertTrue(all(p.poll() is not None for p in owned[1:]))
                self.assertTrue(all(not t.is_alive() for t in owned[0].threads))
                self.assertEqual(json.loads((self.path / 'guard.json').read_text())['exit'], expected)

    def test_descendant_owned_after_root_exit(self):
        # Real OS processes, no GPU: the root exits while a silent worker remains alive.
        self.g.a.cmd = [sys.executable, '-c',
                       'import subprocess,sys; subprocess.Popen([sys.executable,"-c",'
                       '"import time; time.sleep(60)"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)']
        try:
            self.g.launch(dict(os.environ), re.compile('FAIL'))
            self.g.child.wait(timeout=10)
            self.assertTrue(self.g.workload_active())
            self.g.fire(92, 'post-root trip')
            if guard.WIN:
                end = time.monotonic() + 5
                while self.g.workload_active() and time.monotonic() < end:
                    time.sleep(0.02)
                self.assertFalse(self.g.workload_active())
        finally:
            self.g.close()

    def test_exited_root_does_not_skip_group_kill(self):
        self.g.child = SimpleNamespace(pid=123, poll=lambda: 0)
        with patch.object(guard, 'WIN', False), patch.object(guard.os, 'killpg', create=True) as kill:
            self.g.kill_child()
        kill.assert_called_once_with(123, guard.signal.SIGKILL)

    def test_shutdown_awaits_latch_persistence(self):
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        real_fsync = os.fsync
        def delayed_fsync(fd):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('test release timed out')
            real_fsync(fd)
        with patch.object(guard.os, 'fsync', side_effect=delayed_fsync):
            firing = threading.Thread(target=self.g.fire, args=(90, 'race'))
            firing.start()
            self.assertTrue(entered.wait(5))
            closing = threading.Thread(target=lambda: (self.g.close(), closed.set()))
            closing.start()
            try:
                self.assertFalse(closed.wait(0.1))
            finally:
                release.set()
                firing.join(5)
                closing.join(5)
        self.assertTrue(closed.is_set())
        self.assertIn('race', (self.path / 'STOP').read_text())

    def test_latch_error_in_receipt(self):
        self.g.a.latch = str(self.path / 'missing' / 'STOP')
        self.g.fire(90, 'event')
        d = self.receipt(code=90)
        self.assertIsNotNone(d['latch_error'])

    def test_initial_health_requires_samples_and_heartbeat(self):
        with patch.object(guard, 'WIN', True):
            self.assertFalse(self.g.ready())
            self.g.last_smi = self.g.replay_at = time.monotonic()
            self.assertFalse(self.g.ready())
            self.g.event_reader(SimpleNamespace(stdout=io.StringIO('HB now\n')))
            self.assertTrue(self.g.ready())

    def test_monitor_exit_and_postwindow_staleness(self):
        for exited in (True, False):
            with self.subTest(exited=exited), patch.object(guard, 'WIN', True):
                g = guard.Guard(self.a)
                g.log = Mock()
                g.last_smi = g.replay_at = time.monotonic()
                g.last_hb = time.monotonic() - (0 if exited else 10)
                g.monitors = [(SimpleNamespace(poll=lambda: 1 if exited else None), 'event')]
                g.check_health(time.monotonic(), time.monotonic())
                self.assertEqual(g.trip[0], 91)

    def test_stalled_replay_does_not_block_deadlines(self):
        real_spawn = subprocess.Popen
        processes = []
        entered = threading.Event()
        def stall(*args, **kwargs):
            p = real_spawn([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
            processes.append(p)
            entered.set()
            return p
        for wall in (False, True):
            with self.subTest(wall=wall):
                g = guard.Guard(self.a)
                g.log = Mock()
                g.last_smi = g.last_hb = g.replay_at = time.monotonic()
                if not wall:
                    g.last_smi -= 10
                started = time.monotonic()
                entered.clear()
                with patch.object(guard.subprocess, 'Popen', side_effect=stall):
                    g.watch(g.replay_worker, 'GPU-test')
                    self.assertTrue(entered.wait(5))
                    g.check_health(started, started - 10 if wall else started)
                    self.assertEqual(g.trip[0], 94 if wall else 91)
                    g.close()
                self.assertLess(time.monotonic() - started, 2)
        self.assertTrue(all(p.poll() is not None for p in processes))

    def test_required_thermal_fields_are_finite(self):
        for temp, margin in [('[N/A]', '50'), ('40', '[N/A]'), ('nan', '50'), ('40', 'inf')]:
            with self.subTest(temp=temp, margin=margin):
                g = guard.Guard(self.a)
                g.log = Mock()
                g.uuid = 'GPU-test'
                g.smi_reader(SimpleNamespace(stdout=io.StringIO(self.row(temp, margin))))
                self.assertEqual(g.trip[0], 91)
                self.assertIsNone(g.last_smi)

    def test_unavailable_replay_is_fatal(self):
        real_spawn = subprocess.Popen
        def unavailable(*args, **kwargs):
            return real_spawn([sys.executable, '-c', 'print("Replays Since Reset : N/A")'], **kwargs)
        with patch.object(guard.subprocess, 'Popen', side_effect=unavailable):
            self.g.replay_worker('GPU-test')
        self.assertEqual(self.g.trip[0], 93)
        self.assertFalse(self.g.ready())

    def test_idle_zero_admitted_invalid_memory_refused(self):
        for memory, expected in [('0', 0), ('nan', 96), ('inf', 96), ('[N/A]', 96), ('2049', 96)]:
            with self.subTest(memory=memory):
                def accept(g, uuid, pat):
                    g.launched = True
                rc, supervised = self.run_main(accept, memory=memory)
                self.assertEqual(rc, expected)
                self.assertEqual(supervised.call_count, 1 if expected == 0 else 0)

    def test_unavailable_measurements_preserve_receipt(self):
        self.g.smi_reader(SimpleNamespace(stdout=io.StringIO(
            self.row(watts='[N/A]', sm='nan') + self.row(watts='100', sm='1500') + self.row(watts='inf', sm='1200'))))
        d = self.receipt()
        self.assertEqual(d['busy_mean_w'], 100)
        self.assertEqual(d['busy_w_samples'], 1)
        self.assertEqual(d['busy_mean_sm_mhz'], 1350)
        self.assertEqual(d['busy_sm_samples'], 2)
        self.assertEqual(d['busy_s'], 3)


    def integration(self, mode, command=None, extra=()):
        real_spawn = subprocess.Popen
        processes = []
        queries = 0
        def spawn(cmd, **kw):
            nonlocal queries
            if cmd[0] == 'nvidia-smi':
                if '-q' in cmd:
                    queries += 1
                    code = ('import time; time.sleep(60)' if mode == 'stalled_replay' and queries > 1
                            else 'print("Replays Since Reset : 0")')
                else:
                    code = 'import time\nwhile True:\n print(' + repr(self.row().strip()) + ',flush=True); time.sleep(0.05)'
                cmd = [sys.executable, '-c', code]
            elif cmd[0] == 'powershell':
                if mode == 'dead_event':
                    code = 'raise SystemExit(1)'
                elif mode == 'scan_event':
                    code = 'import time; time.sleep(0.4); print("EVT fault",flush=True); print("HB now",flush=True); time.sleep(60)'
                elif mode == 'post_exit':
                    code = 'import time; print("HB now",flush=True); time.sleep(0.7)'
                else:
                    code = 'import time\nwhile True:\n print("HB now",flush=True); time.sleep(0.05)'
                cmd = [sys.executable, '-c', code]
            p = real_spawn(cmd, **kw)
            processes.append(p)
            return p
        with patch.object(guard.subprocess, 'Popen', side_effect=spawn), \
             patch.object(guard, 'resolve', return_value=('GPU-test', 'fixture')), \
             patch.object(guard, 'smi', return_value=(0, '0')), \
             patch.object(guard, 'STARTUP_QUIET_S', 0.05), \
             patch.object(guard, 'MONITOR_STARTUP_S', 2), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = guard.main(['--gpu-name', 'test', '--out', str(self.path), '--latch', self.a.latch,
                             '--stale-s', '0.5', '--timeout-s', '3', '--replay-every-s', '0.1',
                             '--post-s', '0.1', *extra, '--', *(command or self.a.cmd)])
        self.assertTrue(all(p.poll() is not None for p in processes))
        return rc

    @unittest.skipUnless(guard.WIN, 'Windows event supervision')
    def test_dead_event_watcher_refuses_launch(self):
        marker = self.path / 'launched'
        command = [sys.executable, '-c', f'from pathlib import Path; Path({str(marker)!r}).touch()']
        self.assertEqual(self.integration('dead_event', command), 96)
        self.assertFalse(marker.exists())

    @unittest.skipUnless(guard.WIN, 'Windows event supervision')
    def test_initial_scan_must_finish_before_launch(self):
        marker = self.path / 'launched'
        command = [sys.executable, '-c', f'from pathlib import Path; Path({str(marker)!r}).touch()']
        self.assertEqual(self.integration('scan_event', command), 96)
        self.assertFalse(marker.exists())

    @unittest.skipUnless(guard.WIN, 'Windows event supervision')
    def test_postwindow_event_watcher_exit_trips(self):
        self.assertEqual(self.integration('post_exit', extra=('--post-s', '1')), 91)

    def test_supervisor_wall_limit_with_stalled_replay(self):
        started = time.monotonic()
        command = [sys.executable, '-c', 'import time; time.sleep(60)']
        self.assertEqual(self.integration('stalled_replay', command, ('--timeout-s', '1')), 94)
        self.assertLess(time.monotonic() - started, 3)

    def test_clean_completion_and_child_failure(self):
        self.assertEqual(self.integration('healthy'), 0)
        self.assertEqual(self.integration('healthy', [sys.executable, '-c', 'raise SystemExit(37)']), 37)


if __name__ == '__main__':
    unittest.main()
