"""Independent HTTP reference-match gate, plan-only unless --allow-gpu.

Two resident direct-shard services (default, then forced file tier), each with
six requests and two local CPU test receivers per request. Private reference
results/provenance are supplied with pinned file digests; never commit them.
Expected wall budget: 90--150 minutes including two cold source-hash startups;
hard aggregate deadline: at most 180 minutes, plus <=15 seconds owned-child cleanup.
Internal OLD/NEW comparison is an optional extra diagnostic, not the reference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import socket
import struct
import subprocess
import sys
import shutil
import contextlib
import time
import urllib.error
import urllib.request

IDENTITY = 'glm-pf-rtx-native-v1'
LENGTHS = (1024, 8192, 32768)
REQUESTS = tuple(n for n in LENGTHS for _ in range(2))
PHASES = (('default', []), ('file', ['--max-device-layers', '19',
                                   '--max-pinned-layers', '22', '--require-file-tier']))
MAX_SECONDS = 180 * 60
SERVICE_EXIT_SECONDS = 120  # releasing ~90 GB device + pinned host memory after kill
CACHE_GATE = 'PENDING: --cache overlapped-reader/H2D-overlap qualification is a separate gate; no on-disk cache is available'


def windows_job():
    # Also works when CPU tests import this file by absolute path.
    import importlib.util
    spec = importlib.util.spec_from_file_location('glm_windows_job', Path(__file__).with_name('windows_job.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WindowsJob


def locked_file(path, directory=False):
    """Read handle denying writes/deletion for the entire pinned run."""
    import ctypes
    from ctypes import wintypes as w
    import msvcrt
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p, w.DWORD, w.DWORD, w.HANDLE]
    k.CreateFileW.restype = w.HANDLE
    k.CloseHandle.argtypes = [w.HANDLE]
    handle = k.CreateFileW(str(Path(path).resolve()), 0x80000000, 1, None, 3,
                           0x02000000 if directory else 0, None)
    if handle == w.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    if directory:
        @contextlib.contextmanager
        def held_directory():
            try:
                yield
            finally:
                k.CloseHandle(handle)
        return held_directory()
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        k.CloseHandle(handle)
        raise
    return os.fdopen(fd, 'rb')


class PinnedExecutables:
    def __init__(self, build, out):
        self.root = out / 'executables'
        self.root.mkdir()  # unique run, never replace an earlier snapshot
        self.held = contextlib.ExitStack()
        self.digests = {}
        try:
            for kind in ('service', 'receiver'):
                name = f'glm_native_{kind}.exe'
                with locked_file(build / name) as source, (self.root / name).open('xb') as target:
                    if os.fstat(source.fileno()).st_size > 100 * 1024 * 1024:
                        raise ValueError('candidate executable exceeds 100 MiB bound')
                    shutil.copyfileobj(source, target, 1 << 20)
                held = self.held.enter_context(locked_file(self.root / name))
                h = hashlib.sha256()
                for chunk in iter(lambda: held.read(1 << 20), b''):
                    h.update(chunk)
                self.digests[kind + '_sha256'] = h.hexdigest()
            self.held.enter_context(locked_file(self.root, directory=True))
        except BaseException:
            self.held.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.held.close()


def executable(a, kind):
    return getattr(a, 'pinned_build', a.build) / f'glm_native_{kind}.exe'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def small_json(path, expected=None):
    path = Path(path)
    if path.stat().st_size > 8 << 20:
        raise ValueError('JSON input exceeds 8 MiB')
    raw = path.read_bytes()
    if expected is not None and hashlib.sha256(raw).hexdigest() != digest(expected):
        raise ValueError('private reference file pin mismatch')
    return json.loads(raw)


def digest(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('expected full lowercase SHA-256')
    return value


def load_inputs(a):
    # Pin the private reference files AND the executable recorded in provenance.
    prov = small_json(a.reference_provenance, a.reference_provenance_sha256)
    if prov['service_sha256'] != digest(a.reference_service_sha256):
        raise ValueError('reference executable provenance mismatch')
    digest(prov['manifest_sha256'])
    digest(prov['receiver_sha256'])
    digest(prov['model_sha256'])
    if a.reference_commit != 'a6d126c4':
        raise ValueError('this gate requires the independent a6d126c4 reference')
    rows = small_json(a.reference_results, a.reference_results_sha256)
    refs = {}
    if [r['tokens'] for r in rows] != list(REQUESTS):
        raise ValueError('reference requires two consecutive requests per length')
    for r in rows:
        n = r['tokens']
        states = r['state_sha256']
        if (r['capacity'] != 32768 or r['status'] != 'LOCALLY_SEALED_BOTH'
                or len(states) != 2):
            raise ValueError('reference capacity/seal/ranks mismatch')
        for state in states:
            digest(state)
        if refs.setdefault(n, states) != states:
            raise ValueError('reference repeats differ')
    prompts = {}
    for n in LENGTHS:
        path = a.prompt_dir / f'p{n}.u32'
        raw = path.read_bytes()
        toks = small_json(a.prompt_dir / f'p{n}.json')
        if (len(raw) != 4*n or len(toks) != n
                or any(type(t) is not int or not 0 <= t < 154880 for t in toks)
                or raw != struct.pack(f'<{n}I', *toks)
                or hashlib.sha256(raw).hexdigest() != prov['prompts'][str(n)]):
            raise ValueError(f'prompt {n} bytes/IDs/reference pin mismatch')
        prompts[n] = toks
    return prov, refs, prompts


def service_command(a, flags, key, port):
    return [str(executable(a, 'service')), '--model', str(a.model),
            '--capacity', '32768', '--key-file', str(key), '--bind', '127.0.0.1',
            '--port', str(port), *flags, '--allow-gpu']


def plan(a):
    return dict(mode='REFERENCE-MATCH', capacity=32768, row_tile='auto (2048)',
                expected_wall_minutes=[90, 150], aggregate_timeout_seconds=a.timeout,
                cleanup_limit_seconds=15, requests_per_phase=list(REQUESTS),
                phases=[dict(name=name, command=service_command(a, flags, 'PRIVATE_KEY', 'LOCAL_PORT'))
                        for name, flags in PHASES],
                receivers_per_request=2, reference_commit=a.reference_commit,
                diagnostic_self_check=a.diagnostic_self_check,
                cache_reader_gate=CACHE_GATE,
                deadline_enforcement='independent supervisor process; Windows KILL_ON_JOB_CLOSE job for validation and all descendants',
                qualification='DRAFT; no hidden-buffer independent reference or node continuation evidence')


class Deadline:
    def __init__(self, seconds, end=None):
        self.end = time.monotonic() + seconds if end is None else end

    def left(self):
        left = self.end - time.monotonic()
        if left <= 0:
            raise TimeoutError('aggregate deadline; incomplete is never a pass')
        return left

    def sleep(self, seconds):
        time.sleep(min(seconds, self.left()))


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def http(url, deadline, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=min(1200 if body else 3, deadline.left())) as response:
        return json.load(response)


def stop(proc, wait_seconds=5):
    if proc is not None and proc.poll() is None:
        proc.kill()  # only a process created here, never a name/global device kill
        # A service releasing ~90 GB of device memory and tens of GiB of pinned host
        # memory needs well over 5 s to exit; a stop that never completes is an
        # error, not the aggregate deadline.
        try:
            proc.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f'owned process {proc.pid} did not exit {wait_seconds} s after kill') from exc


def stop_all(processes):
    error = None
    for proc in processes:
        try:
            stop(proc)
        except Exception as exc:
            error = exc
    if error is not None:
        raise error


def start(command, **kwargs):
    if os.name == 'nt':
        kwargs['creationflags'] = subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(command, **kwargs)


def require_health(health, model_sha, forced):
    if (health.get('model_sha256') != model_sha or health.get('identity') != IDENTITY
            or health.get('capacity') != 32768 or health.get('row_tile') != 2048
            or health.get('residency', {}).get('weight_source') != 'shards'):
        raise ValueError('candidate model/identity/capacity/default tile/source mismatch')
    r = health['residency']
    if forced and (r.get('file_layers', 0) < 2 or r.get('file_ring_mask') != 3
                   or r.get('max_device_layers') != 19 or r.get('max_pinned_layers') != 22):
        raise ValueError('forced file tier must admit >=2 layers and both ring parities')


def require_match(receipt, outputs, expected, model_sha, forced, producer_sha):
    states = []
    for rc, out, err in outputs:
        words = out.strip().split()
        if rc != 0 or len(words) != 2 or words[0] != 'SEALED':
            raise ValueError(f'receiver failed: rc={rc}: {out} {err}')
        states.append(digest(words[1]))
    if states != expected:
        raise ValueError('MISMATCH: per-rank SEALED state differs from independent reference')
    if (receipt.get('status') != 'LOCALLY_SEALED_BOTH'
            or receipt.get('model_sha256') != model_sha
            or receipt.get('producer_identity') != IDENTITY
            or receipt.get('weight_source') != 'shards'):
        raise ValueError('candidate receipt seal/model/identity/source mismatch')
    if receipt.get('producer_executable_sha256') != digest(producer_sha):
        raise ValueError('candidate receipt executable differs from pinned service SHA-256')
    if forced:
        t = receipt.get('phase_timings', {})
        if any(t.get(k, 0) <= 0 for k in ('file_tier_bytes', 'file_ring0_bytes', 'file_ring1_bytes')):
            raise ValueError('forced-file request must read bytes through both rings')
        require_health(receipt['capacity_cache_profile'], model_sha, True)
    return states


def request_pair(a, out, port, keyfile, key, n, toks, prov, refs, deadline, forced):
    receivers = []
    try:
        ports = [free_port(), free_port()]
        for rank in (0, 1):
            deadline.left()
            command = [str(executable(a, 'receiver')), f'127.0.0.1:{ports[rank]}',
                       str(keyfile), prov['prompts'][str(n)], prov['model_sha256'],
                       str(rank), str(n-1), str(n)]
            receivers.append(start(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        deadline.sleep(.5)
        body = dict(tokens=toks, identity=IDENTITY,
                    handoff=dict(rank0=f'127.0.0.1:{ports[0]}', rank1=f'127.0.0.1:{ports[1]}',
                                 key=key.hex(), node_identity='r1'))
        receipt = http(f'http://127.0.0.1:{port}/v1/prefill', deadline, body)
        outputs = []
        for proc in receivers:
            stdout, stderr = proc.communicate(timeout=deadline.left())
            outputs.append((proc.returncode, stdout, stderr))
        states = require_match(receipt, outputs, refs[n], prov['model_sha256'], forced, a.service_sha256)
        return receipt, states
    finally:
        stop_all(receivers)


def phase(a, name, flags, out, keyfile, key, prov, refs, prompts, deadline, results):
    service = None
    with (out / f'service-{name}.log').open('w', encoding='utf-8') as log:
        try:
            port = free_port()
            deadline.left()
            service = start(service_command(a, flags, keyfile, port), stdout=log, stderr=subprocess.STDOUT)
            while True:
                deadline.left()
                if service.poll() is not None:
                    raise RuntimeError(f'service exited during startup: {service.returncode}')
                try:
                    health = http(f'http://127.0.0.1:{port}/health', deadline)
                    break
                except (OSError, urllib.error.URLError):
                    deadline.sleep(2)
            require_health(health, prov['model_sha256'], name == 'file')
            (out / f'health-{name}.json').write_text(json.dumps(health, indent=2))
            for i, n in enumerate(REQUESTS):
                begin = time.monotonic()
                receipt, states = request_pair(a, out, port, keyfile, key, n, prompts[n], prov, refs,
                                               deadline, name == 'file')
                results.append(dict(phase=name, index=i, tokens=n, state_sha256=states,
                                    reference_equal=True, wall_s=time.monotonic()-begin))
                (out / f'receipt-{name}-{i}.json').write_text(json.dumps(receipt, indent=2))
                (out / 'results.json').write_text(json.dumps(results, indent=2))
        finally:
            stop(service, wait_seconds=SERVICE_EXIT_SECONDS)


def diagnostic(a, out, deadline):
    command = [str(executable(a, 'service')), '--model', str(a.model), '--capacity', '32768',
               '--self-check-schedule', '--max-device-layers', '19', '--max-pinned-layers', '22',
               '--require-file-tier', '--allow-gpu']
    for n in LENGTHS:
        command += ['--prompt', str(a.prompt_dir / f'p{n}.u32')]
    child = None
    with (out / 'internal-diagnostic.jsonl').open('w') as log:
        try:
            deadline.left()
            child = start(command, stdout=log, stderr=subprocess.STDOUT)
            if child.wait(timeout=deadline.left()) != 0:
                raise ValueError('internal diagnostic failed')
        finally:
            stop(child)
    states = [json.loads(line) for line in (out / 'internal-diagnostic.jsonl').read_text().splitlines()
              if line.startswith('{') and '"verdict"' in line]
    if [s['tokens'] for s in states] != list(LENGTHS) or any(
            s['verdict'] != 'EXACT' or s['differing_bytes'] or s['memcmp_failures'] for s in states):
        raise ValueError('internal diagnostic incomplete/mismatched')


def run(a):
    if not 1 <= a.timeout <= MAX_SECONDS:
        raise ValueError('aggregate deadline must be 1..10800 seconds')
    if not a.allow_gpu and not a.check_inputs:
        print(json.dumps(plan(a), indent=2))
        return 0
    if a.allow_gpu and not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',
                                       os.environ.get('CUDA_VISIBLE_DEVICES', '')):
        raise ValueError('set CUDA_VISIBLE_DEVICES to one full authorized PRO UUID')
    # The supervisor does no input IO. The worker, validation, HTTP calls and
    # every descendant are contained before they can execute any user code.
    print(json.dumps(dict(cache_reader_gate=CACHE_GATE)), flush=True)
    return supervised_run(a)


def supervised_run(a):
    if os.name != 'nt':
        raise RuntimeError('hard-deadline qualification requires Windows job objects')
    deadline = Deadline(a.timeout)
    out = a.build / ('qualification-' + time.strftime('%Y%m%d-%H%M%S') + '-' + secrets.token_hex(6))
    out.mkdir()
    argv = []
    for name in ('build', 'model', 'prompt_dir', 'reference_results', 'reference_provenance',
                 'reference_results_sha256', 'reference_provenance_sha256', 'reference_service_sha256',
                 'reference_commit', 'timeout', 'inputs'):
        value = getattr(a, name, None)
        if value is not None:
            argv += ['--' + name.replace('_', '-'), str(value)]
    argv += ['--allow-gpu' if a.allow_gpu else '--check-inputs']
    if a.diagnostic_self_check:
        argv += ['--diagnostic-self-check']
    code = ('import runpy,sys; from pathlib import Path; '
            'm=runpy.run_path(sys.argv[1]); '
            'sys.exit(m["run_worker"](m["arguments"](sys.argv[4:],resolve_inputs=True),Path(sys.argv[2]),float(sys.argv[3])))')
    try:
        with windows_job()() as job:
            process, _ = job.start([sys.executable, '-B', '-c', code, str(Path(__file__).resolve()),
                                    str(out.resolve()), str(deadline.end), *argv])
            result = job.wait(process, deadline.left())
            deadline.left()  # a late completion never becomes a qualification pass
            return result
    except TimeoutError:
        print('INCOMPLETE: absolute deadline; complete owned job terminated; no qualification pass')
        print(CACHE_GATE)
        return 4
    finally:
        # The supervisor owns credential cleanup even if validation/worker stalls.
        (out / 'import.key').unlink(missing_ok=True)


def run_worker(a, out, end):
    deadline = Deadline(a.timeout, end=end)
    prov, refs, prompts = load_inputs(a)
    deadline.left()
    with PinnedExecutables(a.build, out) as pinned:
        a.pinned_build = pinned.root
        a.service_sha256 = pinned.digests['service_sha256']
        return execute_pinned(a, out, prov, refs, prompts, deadline, pinned.digests)


def execute_pinned(a, out, prov, refs, prompts, deadline, candidate):
    deadline.left()
    if not a.allow_gpu:
        print(json.dumps(dict(verdict='INPUTS-VALID', reference=prov, candidate=candidate, cache_reader_gate=CACHE_GATE)))
        return 0
    key = secrets.token_bytes(32)
    keyfile = out / 'import.key'
    keyfile.write_bytes(key)
    evidence = dict(candidate=candidate, reference=prov, reference_commit=a.reference_commit,
                    reference_row_tile=256, reference_capacity=32768, reference_source='cache',
                    reference_results_sha256=a.reference_results_sha256,
                    reference_provenance_sha256=a.reference_provenance_sha256,
                    expected_state_sha256=refs, plan=plan(a))
    (out / 'provenance.json').write_text(json.dumps(evidence, indent=2))
    results = []
    try:
        for name, flags in PHASES:
            phase(a, name, flags, out, keyfile, key, prov, refs, prompts, deadline, results)
        if a.diagnostic_self_check:
            diagnostic(a, out, deadline)
        deadline.left()
        print(json.dumps(dict(verdict='REFERENCE-MATCH', requests=len(results), output=str(out),
                              cache_reader_gate=CACHE_GATE,
                              qualification='DRAFT; independent hidden-buffer and node gates remain')))
        return 0
    except (TimeoutError, subprocess.TimeoutExpired):
        print('INCOMPLETE: aggregate deadline; owned children stopped; no qualification pass')
        return 4
    finally:
        keyfile.unlink()  # do not retain a live handoff credential in evidence


def arguments(argv=None, resolve_inputs=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, help='private JSON argument defaults; keep outside Git')
    p.add_argument('--build', type=Path, required=True)
    for name in ('model', 'prompt-dir', 'reference-results', 'reference-provenance'):
        p.add_argument('--' + name, type=Path)
    for name in ('reference-results-sha256', 'reference-provenance-sha256', 'reference-service-sha256'):
        p.add_argument('--' + name)
    p.add_argument('--reference-commit', default='a6d126c4')
    p.add_argument('--timeout', type=int, default=MAX_SECONDS)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--allow-gpu', action='store_true')
    mode.add_argument('--check-inputs', action='store_true', help='CPU-only: validate private pins/prompts/executables')
    p.add_argument('--diagnostic-self-check', action='store_true')
    pre, _ = p.parse_known_args(argv)
    resolve_inputs = resolve_inputs or not (pre.allow_gpu or pre.check_inputs)
    if pre.inputs and resolve_inputs:
        config = small_json(pre.inputs)
        allowed = {'model', 'prompt_dir', 'reference_results', 'reference_provenance',
                   'reference_results_sha256', 'reference_provenance_sha256', 'reference_service_sha256',
                   'reference_commit'}
        if set(config) - allowed:
            p.error('unsupported private input field')
        p.set_defaults(**config)
    a = p.parse_args(argv)
    for name in ('model', 'prompt_dir', 'reference_results', 'reference_provenance'):
        value = getattr(a, name)
        if value is None:
            if not a.inputs or resolve_inputs:
                p.error('--' + name.replace('_', '-') + ' is required (or supply --inputs)')
        else:
            setattr(a, name, Path(value))
    for name in ('reference_results_sha256', 'reference_provenance_sha256', 'reference_service_sha256'):
        if getattr(a, name) is None and (not a.inputs or resolve_inputs):
            p.error('--' + name.replace('_', '-') + ' is required (or supply --inputs)')
    return a


if __name__ == '__main__':
    raise SystemExit(run(arguments()))
