"""Host-only failure and cleanup tests; mock service/client, no sockets or GPU."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT, BASH = Path(sys.argv[1]), sys.argv[2]
# scratch under the build tree when given (out-of-tree builds have no ROOT/build), else the system temp
WORK = Path(sys.argv[3]) if len(sys.argv) > 3 else None
if WORK is not None:
    WORK.mkdir(parents=True, exist_ok=True)
del sys.argv[1:]
GOOD = dict(tokens=2, http_status=200, sha256='a' * 64, logits_fnv='b' * 16,
            next_token=7, wall_s=1.0, seconds={'prefill': 0.5})


class BenchmarkTests(unittest.TestCase):
    def test_summary_rejects_invalid_and_differing_results(self):
        cases = [([], False), ([GOOD], False), ([GOOD] * 3, False), ([GOOD] * 2, True)]
        for field, value in [('tokens', 3), ('http_status', 500), ('sha256', None),
                             ('logits_fnv', ''), ('next_token', None), ('wall_s', 0)]:
            bad = dict(GOOD, **{field: value})
            cases.append(([bad, bad], False))
        for field, value in [('sha256', 'c' * 64), ('logits_fnv', 'd' * 16), ('next_token', 8)]:
            cases.append(([GOOD, dict(GOOD, **{field: value})], False))
        missing = copy.deepcopy(GOOD)
        del missing['sha256']
        cases.append(([missing, missing], False))
        with tempfile.TemporaryDirectory(dir=WORK) as td:
            out = Path(td)
            for records, success in cases:
                with self.subTest(records=records):
                    (out / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
                    p = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/bench/cuda/ds41_prefill_summary.py'),
                                        str(out), '--lengths', '2', '--reps', '2'], capture_output=True, text=True)
                    self.assertEqual(p.returncode == 0, success, p.stdout + p.stderr)

    def test_matrix_failure_propagation_and_cleanup(self):
        with tempfile.TemporaryDirectory(dir=WORK) as td:
            root = Path(td)
            scripts = root / 'scripts/bench/cuda'
            scripts.mkdir(parents=True)
            (root / 'tools').mkdir()
            bin_dir = root / 'bin'
            bin_dir.mkdir()
            for name in ('ds41_prefill_matrix.sh', 'ds41_prefill_summary.py'):
                shutil.copyfile(ROOT / 'scripts/bench/cuda' / name, scripts / name)
            (root / 'tools/ds41_prefill_service_client.py').write_text('''import json, os, sys
args = sys.argv
mode = os.environ['MOCK_MODE']
if mode in ('startup', 'timeout'): sys.exit(8)
count = 1 if mode in ('client', 'partial') else int(args[args.index('--repeat') + 1])
with open(args[args.index('--json') + 1], 'a') as f:
    for i in range(count):
        r = json.loads(os.environ['MOCK_RECORD'])
        if mode == 'different': r['next_token'] += i
        if mode == 'status': r['http_status'] = 500
        if mode == 'missing': del r['sha256']
        f.write(json.dumps(r) + '\\n')
sys.exit(7 if mode == 'client' else 0)
''', encoding='utf-8')
            def script(path, content):
                path.write_text('#!/bin/bash\n' + content, encoding='utf-8', newline='\n')
                path.chmod(0o755)
            script(root / 'service.sh', '''echo "$BASHPID" > "$MOCK_SERVICE_PID"
[ "$MOCK_MODE" = startup ] && exit 3
if [ "$MOCK_MODE" != timeout ]; then echo '{"event":"READY"}'; fi
trap 'exit 0' TERM
while :; do sleep 0.05; done
''')
            script(bin_dir / 'nvidia-smi', 'echo "100, 50, 1000"\n')
            script(bin_dir / 'python3', f'exec "{Path(sys.executable).as_posix()}" "$@"\n')
            # Speed up polling; record the actual sampler subshell PID for cleanup assertions.
            (root / 'bash_env').write_text('''sleep() {
  if [ "$1" = 1 ]; then echo "$BASHPID" > "$MOCK_SAMPLER_PID"; fi
  command sleep 0.05
}
''', encoding='utf-8', newline='\n')
            (root / 'ids.txt').write_text('1,2', encoding='utf-8')
            for mode in ('startup', 'timeout', 'client', 'partial', 'status', 'missing', 'different', 'good'):
                with self.subTest(mode=mode):
                    print(f'matrix lifecycle case: {mode}', flush=True)
                    out = root / mode
                    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
                               MOCK_MODE=mode, MOCK_RECORD=json.dumps(GOOD),
                               MOCK_SERVICE_PID=(root / 'service.pid').as_posix(),
                               MOCK_SAMPLER_PID=(root / 'sampler.pid').as_posix(), LENGTHS='2', REPS='2', READY_S='20')
                    for name in ('service.pid', 'sampler.pid'):
                        (root / name).unlink(missing_ok=True)
                    try:
                        p = subprocess.run([BASH, '-c', 'source "$1"; shift; source "$0" "$@"',
                                            (scripts / 'ds41_prefill_matrix.sh').as_posix(), (root / 'bash_env').as_posix(), out.as_posix(), '--',
                                            (root / 'service.sh').as_posix(), 'model.gguf', (root / 'ids.txt').as_posix()],
                                           env=env, capture_output=True, text=True, timeout=15)
                        self.assertEqual(p.returncode == 0, mode == 'good', p.stdout + p.stderr)
                        for name in ('service.pid', 'sampler.pid'):
                            pid_file = root / name
                            if pid_file.exists():
                                p = subprocess.run([BASH, '-c', 'kill -0 "$1" 2>/dev/null', 'check', pid_file.read_text().strip()])
                                self.assertNotEqual(p.returncode, 0, f'{name} survived cleanup')
                    finally:
                        for name in ('service.pid', 'sampler.pid'):
                            pid_file = root / name
                            if pid_file.exists():
                                subprocess.run([BASH, '-c', 'kill "$1" 2>/dev/null || true', 'cleanup', pid_file.read_text().strip()])


if __name__ == '__main__':
    unittest.main(verbosity=2, failfast=True)
