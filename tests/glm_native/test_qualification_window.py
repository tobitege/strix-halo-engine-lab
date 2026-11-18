"""CPU-only gate tests; real Windows jobs run harmless Python children, no sockets."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import types
import sys
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('window', ROOT/'src/cuda/models/glm/producer/qualification_window.py')
window = importlib.util.module_from_spec(spec)
spec.loader.exec_module(window)


class WindowTests(unittest.TestCase):
    def args(self):
        return types.SimpleNamespace(build=Path('.'), model=Path('MODEL'), prompt_dir=Path('PROMPTS'),
                                     reference_commit='a6d126c4', timeout=10800, allow_gpu=False,
                                     check_inputs=False, diagnostic_self_check=False, service_sha256='e'*64)

    def health(self):
        return dict(model_sha256='a'*64, identity=window.IDENTITY, capacity=32768, row_tile=2048,
                    residency=dict(weight_source='shards', file_layers=2, file_ring_mask=3,
                                   max_device_layers=19, max_pinned_layers=22))

    def receipt(self):
        return dict(status='LOCALLY_SEALED_BOTH', model_sha256='a'*64,
                    producer_identity=window.IDENTITY, weight_source='shards', producer_executable_sha256='e'*64,
                    capacity_cache_profile=self.health(),
                    phase_timings=dict(file_tier_bytes=2, file_ring0_bytes=1, file_ring1_bytes=1))

    def test_plan_never_starts_process_or_socket(self):
        with patch.object(window.subprocess, 'Popen') as start, patch.object(window.socket, 'socket') as sock, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(window.run(self.args()), 0)
            start.assert_not_called()
            sock.assert_not_called()
        plan = window.plan(self.args())
        self.assertEqual(plan['requests_per_phase'], [1024,1024,8192,8192,32768,32768])
        self.assertEqual(len(plan['phases']), 2)
        for phase in plan['phases']:
            cmd = phase['command']
            self.assertIn('--model', cmd)
            self.assertNotIn('--cache', cmd)
            self.assertNotIn('--row-tile', cmd)
        self.assertIn('--require-file-tier', plan['phases'][1]['command'])
        self.assertIn('PENDING', plan['cache_reader_gate'])
        self.assertIn('--cache', plan['cache_reader_gate'])

    def test_uuid_and_deadline_checked_before_process(self):
        args = self.args()
        args.allow_gpu = True
        with patch.dict(window.os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), patch.object(window, 'start') as start:
            with self.assertRaises(ValueError):
                window.run(args)
            start.assert_not_called()
        for timeout in (0,10801):
            args.timeout = timeout
            with self.assertRaises(ValueError):
                window.run(args)

    def test_sealed_different_reference_is_not_a_pass(self):
        outputs = [(0,'SEALED '+'b'*64,''),(0,'SEALED '+'c'*64,'')]
        self.assertEqual(window.require_match(self.receipt(), outputs, ['b'*64,'c'*64], 'a'*64, True, 'e'*64), ['b'*64,'c'*64])
        with self.assertRaisesRegex(ValueError,'MISMATCH'):
            window.require_match(self.receipt(), outputs, ['d'*64,'c'*64], 'a'*64, True, 'e'*64)
        for rc, out in ((1,'SEALED '+'b'*64),(0,'unsealed'),(0,'SEALED short')):
            with self.assertRaises(ValueError):
                window.require_match(self.receipt(), [(rc,out,''),outputs[1]], ['b'*64,'c'*64], 'a'*64, True, 'e'*64)

    def test_file_tier_zero_layers_reads_or_ring_rejected(self):
        for field,value in (('file_layers',0),('file_layers',1),('file_ring_mask',1)):
            health=self.health()
            health['residency'][field]=value
            with self.assertRaises(ValueError):
                window.require_health(health,'a'*64,True)
        outputs=[(0,'SEALED '+'b'*64,''),(0,'SEALED '+'c'*64,'')]
        for field in ('file_tier_bytes','file_ring0_bytes','file_ring1_bytes'):
            receipt=self.receipt()
            receipt['phase_timings'][field]=0
            with self.assertRaises(ValueError):
                window.require_match(receipt,outputs,['b'*64,'c'*64],'a'*64,True, 'e'*64)

    def test_two_consecutive_requests_every_length_both_phases(self):
        args=self.args()
        deadline=Mock()
        deadline.left.return_value=100
        proc=Mock()
        proc.poll.return_value=None
        prompts={n:[] for n in window.LENGTHS}
        with tempfile.TemporaryDirectory(dir=os.environ.get('GLM_NATIVE_TEST_BUILD',ROOT)) as td, \
             patch.object(window,'start',return_value=proc), patch.object(window,'free_port',return_value=1234), \
             patch.object(window,'http',return_value=self.health()), \
             patch.object(window,'request_pair',return_value=(self.receipt(),['b'*64,'c'*64])) as pair:
            results=[]
            for name,flags in window.PHASES:
                window.phase(args,name,flags,Path(td),Path(td)/'key',b'k',dict(model_sha256='a'*64),{},prompts,deadline,results)
            self.assertEqual([r['tokens'] for r in results],list(window.REQUESTS)*2)
            self.assertEqual(pair.call_count,12)
            self.assertEqual(proc.kill.call_count,2)

    def test_timeout_kills_receivers_and_service(self):
        args=self.args()
        deadline=window.Deadline(10)
        receiver=Mock()
        receiver.poll.return_value=None
        with patch.object(window,'free_port',return_value=1234), patch.object(window,'start',return_value=receiver), \
             patch.object(window,'http',side_effect=TimeoutError), patch.object(deadline,'sleep'):
            with self.assertRaises(TimeoutError):
                window.request_pair(args,Path('.'),1234,Path('key'),b'k',1024,[],
                                    dict(prompts={'1024':'d'*64},model_sha256='a'*64),{},deadline,True)
            self.assertEqual(receiver.kill.call_count,2)
            self.assertEqual(receiver.wait.call_args.kwargs,dict(timeout=5))
        service=Mock()
        service.poll.return_value=None
        with tempfile.TemporaryDirectory(dir=os.environ.get('GLM_NATIVE_TEST_BUILD',ROOT)) as td, \
             patch.object(window,'start',return_value=service), patch.object(window,'free_port',return_value=1234), \
             patch.object(deadline,'left',side_effect=[1,TimeoutError()]):
            with self.assertRaises(TimeoutError):
                window.phase(args,'default',[],Path(td),Path('key'),b'k',{}, {}, {},deadline,[])
            service.kill.assert_called_once()

    def test_deadline_is_shared_and_not_reset(self):
        with patch.object(window.time,'monotonic',side_effect=[100,101,111]):
            deadline=window.Deadline(10)
            self.assertEqual(deadline.left(),9)
            with self.assertRaises(TimeoutError):
                deadline.left()

    def test_receipt_requires_pinned_executable(self):
        outputs = [(0,'SEALED '+'b'*64,''),(0,'SEALED '+'c'*64,'')]
        for value in (None, 'f'*64, 'short'):
            receipt = self.receipt()
            receipt['producer_executable_sha256'] = value
            with self.assertRaisesRegex(ValueError, 'pinned service'):
                window.require_match(receipt, outputs, ['b'*64,'c'*64], 'a'*64, False, 'e'*64)

    @unittest.skipUnless(os.name == 'nt', 'Windows file sharing contract')
    def test_pinned_copies_survive_rebuild_and_refuse_mutation(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('GLM_NATIVE_TEST_BUILD',ROOT)) as td:
            root = Path(td)
            build, out = root/'build', root/'run'
            build.mkdir()
            out.mkdir()
            for kind in ('service', 'receiver'):
                (build/f'glm_native_{kind}.exe').write_bytes(kind.encode())
            with window.PinnedExecutables(build, out) as pinned:
                a = self.args()
                a.build, a.pinned_build = build, pinned.root
                for kind in ('service', 'receiver'):
                    source = build/f'glm_native_{kind}.exe'
                    source.write_bytes(b'rebuilt candidate')
                    copy = window.executable(a, kind)
                    self.assertEqual(copy.read_bytes(), kind.encode())
                    self.assertEqual(window.sha(copy), pinned.digests[kind+'_sha256'])
                    with self.assertRaises(OSError):
                        copy.write_bytes(b'changed')
                    with self.assertRaises(OSError):
                        copy.unlink()
                with self.assertRaises(OSError):
                    pinned.root.rename(out/'replacement')
                self.assertEqual(window.service_command(a, [], 'key', 1)[0], str(window.executable(a, 'service')))

    @unittest.skipUnless(os.name == 'nt', 'Windows independent supervisor')
    def test_supervisor_bounds_blocked_validation(self):
        real_job = window.windows_job()
        class BlockedValidation(real_job):
            def start(self, command):
                # The worker is blocked before loading a model or launching any
                # GPU child; only the independent supervisor can interrupt it.
                return super().start([sys.executable, '-B', '-c', 'import time; time.sleep(60)'])
        with tempfile.TemporaryDirectory(dir=os.environ.get('GLM_NATIVE_TEST_BUILD',ROOT)) as td:
            a = self.args()
            a.build, a.timeout, a.check_inputs = Path(td), 1, True
            begin = time.monotonic()
            with patch.object(window, 'windows_job', return_value=BlockedValidation), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(window.run(a), 4)
            self.assertLess(time.monotonic()-begin, 4)
            self.assertIn('INCOMPLETE', output.getvalue())
            self.assertIn(window.CACHE_GATE, output.getvalue())

    @unittest.skipUnless(os.name == 'nt', 'Windows job descendant containment')
    def test_job_close_kills_worker_and_descendant_only(self):
        import ctypes
        from ctypes import wintypes as w
        with tempfile.TemporaryDirectory(dir=os.environ.get('GLM_NATIVE_TEST_BUILD',ROOT)) as td:
            pidfile = Path(td)/'descendant.pid'
            childcode = 'import time; time.sleep(60)'
            code = ('import subprocess,sys,time; from pathlib import Path; '
                    'p=subprocess.Popen([sys.executable,"-B","-c",sys.argv[2]]); '
                    'Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)')
            outsider = subprocess.Popen([sys.executable, '-B', '-c', childcode],
                                        creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
            child = None
            try:
                with window.windows_job()() as job:
                    worker, _ = job.start([sys.executable, '-B', '-c', code, str(pidfile), childcode])
                    end = time.monotonic()+5
                    while not pidfile.exists() and time.monotonic()<end:
                        time.sleep(.01)
                    self.assertTrue(pidfile.exists())
                    job.k.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
                    job.k.OpenProcess.restype = w.HANDLE
                    child = job.k.OpenProcess(0x100000, False, int(pidfile.read_text()))
                    self.assertTrue(child)
                    with self.assertRaises(TimeoutError):
                        job.wait(worker, .1)
                self.assertEqual(job.k.WaitForSingleObject(child, 2000), 0)
                self.assertIsNone(outsider.poll())
            finally:
                if child:
                    job.k.CloseHandle(child)
                outsider.kill()
                outsider.wait(timeout=5)

    def test_private_inputs_deferred_until_contained_worker(self):
        with patch.object(window, 'small_json') as read:
            a = window.arguments(['--build', 'BUILD', '--inputs', 'PRIVATE.json', '--check-inputs'])
            self.assertIsNone(a.model)
            read.assert_not_called()

    def test_private_reference_pins_and_prompt_bytes(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('GLM_NATIVE_TEST_BUILD',ROOT)) as td:
            root=Path(td)
            args=self.args()
            args.prompt_dir=root
            prov=dict(service_sha256='e'*64,receiver_sha256='f'*64,manifest_sha256='d'*64,
                      model_sha256='a'*64,prompts={})
            for n in window.LENGTHS:
                raw=struct.pack(f'<{n}I',*range(n))
                (root/f'p{n}.u32').write_bytes(raw)
                (root/f'p{n}.json').write_text(json.dumps(list(range(n))))
                prov['prompts'][str(n)]=window.sha(root/f'p{n}.u32')
            args.reference_service_sha256=prov['service_sha256']
            args.reference_provenance=root/'provenance.json'
            args.reference_provenance.write_text(json.dumps(prov))
            args.reference_provenance_sha256=window.sha(args.reference_provenance)
            rows=[dict(tokens=n,capacity=32768,status='LOCALLY_SEALED_BOTH',state_sha256=['b'*64,'c'*64])
                  for n in window.REQUESTS]
            args.reference_results=root/'results.json'
            args.reference_results.write_text(json.dumps(rows))
            args.reference_results_sha256=window.sha(args.reference_results)
            _,refs,_=window.load_inputs(args)
            self.assertEqual(refs[32768],['b'*64,'c'*64])
            if os.name == 'nt':
                # End-to-end supervisor -> suspended/assigned worker -> private
                # input validation -> pinned copies, using inert file contents.
                for kind in ('service', 'receiver'):
                    (root/f'glm_native_{kind}.exe').write_bytes(kind.encode())
                inputs = root/'private-inputs.json'
                inputs.write_text(json.dumps({name: str(getattr(args, name)) for name in
                    ('model', 'prompt_dir', 'reference_results', 'reference_provenance',
                     'reference_service_sha256', 'reference_results_sha256', 'reference_provenance_sha256')}))
                completed = subprocess.run([sys.executable, '-B', str(Path(window.__file__)),
                    '--build', str(root), '--inputs', str(inputs), '--check-inputs', '--timeout', '10'],
                    capture_output=True, text=True, timeout=15,
                    creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
                self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
                messages = [json.loads(line) for line in completed.stdout.splitlines()]
                result = next(row for row in messages if row.get('verdict') == 'INPUTS-VALID')
                self.assertEqual(result['candidate']['service_sha256'], window.sha(root/'glm_native_service.exe'))
                self.assertEqual(result['cache_reader_gate'], window.CACHE_GATE)
            args.reference_service_sha256='0'*64
            with self.assertRaisesRegex(ValueError,'executable'):
                window.load_inputs(args)
            args.reference_service_sha256=prov['service_sha256']
            (root/'p1024.u32').write_bytes(b'x'*4096)
            with self.assertRaisesRegex(ValueError,'prompt 1024'):
                window.load_inputs(args)
            args.reference_results.write_text('[]')
            with self.assertRaisesRegex(ValueError,'pin mismatch'):
                window.load_inputs(args)


if __name__ == '__main__':
    unittest.main()
