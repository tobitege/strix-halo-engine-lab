"""Host regressions: mocked HTTP/socket APIs and fake child processes; no network or GPU."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'scripts/bench/cuda'
spec = importlib.util.spec_from_file_location('classes', SCRIPTS / 'qwen_serve_classes.py')
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def good():
    # Every key stream() returns, including the drafting counters the report prints (None = not reported).
    return dict(ids=[7, 8], tokens=2, error=None, done=True, t_first=1, t_end=2, decode_tps=2,
                tpc=None, cf=None, cx=None, cr=None, hu=None, rps=None)


class Responses(unittest.TestCase):
    def test_response_validation(self):
        self.assertIsNone(client.response_error(good(), 2, [7, 8]))
        for changes in (dict(ids=None), dict(ids=[]), dict(ids=[True, 8]), dict(ids=[-1, 8]),
                        dict(ids=['7', 8]), dict(ids=[2**32, 8]), dict(tokens=1), dict(tokens=True), dict(tokens=3),
                        dict(error='SSE failure'), dict(done=False)):
            with self.subTest(changes=changes):
                self.assertIsNotNone(client.response_error(good() | changes, 2))
        self.assertIsNotNone(client.response_error(good(), 2, [7, 9]))

    def test_sse(self):
        event = json.dumps(dict(usage=dict(completion_tokens=2), strix=dict(rtx=dict(ids=[7, 8])))).encode()
        for tail, prefix, valid in ((b'data: [DONE]\n', b'', True), (b'', b'', False),
                                    (b'data: [DONE]\n', b'data: broken\n', False),
                                    (b'data: [DONE]\n', b'data: {"error":"failed"}\n', False)):
            with self.subTest(tail=tail, prefix=prefix):
                with patch.object(client.urllib.request, 'urlopen', return_value=io.BytesIO(prefix + b'data: ' + event + b'\n' + tail)):
                    result = client.stream('http://127.0.0.1', [1], 2, 1)
                self.assertEqual(client.response_error(result, 2) is None, valid)

    def test_cold_warm_and_concurrent_exit_status(self):
        # Each pass defaults to 8 C1 + 4 C4 + 8 C8 requests.
        # Calls 0..19: cold C1/C4/C8; 20..39: warm C1/C4/C8.
        request_count = 2 * (8 + 4 + 8)
        for bad_call, change in ((None, {}), (0, dict(ids=None)), (12, dict(ids=[7, 9])),
                                 (12, dict(ids=None)), (12, dict(error='SSE error')),
                                 (20, dict(ids=[7, 9])), (32, dict(done=False))):
            with self.subTest(bad_call=bad_call, change=change), tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
                path = Path(tmp)
                (path / 'prompts').write_text(json.dumps(dict(module='', edit=[], json=[])))
                (path / 'prose').write_text(' '.join(str(i) for i in range(100)))
                args = ['classes', 'http://127.0.0.1', str(path / 'out'), '--prompts', str(path / 'prompts'),
                        '--tokenizer', 'unused', '--prose-ids', str(path / 'prose'), '--classes', 'prose',
                        '--prose-len', '1', '--max-tokens', '2']
                counter = iter(range(request_count))

                def response(*_):
                    return good() | (change if next(counter) == bad_call else {})

                tokenizer = types.SimpleNamespace(Tokenizer=types.SimpleNamespace(from_file=lambda _: None))
                with patch.dict(sys.modules, tokenizers=tokenizer), patch.object(sys, 'argv', args), \
                     patch.object(client, 'stream', side_effect=response), patch.object(client, 'wait_health', return_value=0), \
                     patch.object(client.urllib.request, 'urlopen', return_value=io.BytesIO(b'{"engine":"fixture"}')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    if bad_call is None:
                        client.main()
                    else:
                        with self.assertRaises(SystemExit) as ex:
                            client.main()
                        self.assertNotEqual(ex.exception.code, 0)
                report = (path / 'out').read_text()
                self.assertEqual('done failures 0' in report, bad_call is None)
                self.assertEqual(len((path / 'out.ids').read_text().splitlines()), request_count)
                self.assertEqual(json.loads((path / 'out.server.json').read_text())['engine'], 'fixture')

    def test_occupied_endpoint(self):
        with patch.object(client.socket, 'getaddrinfo', return_value=[(2, 1, 6, '', ('127.0.0.1', 8095))]), \
             patch.object(client.socket, 'socket') as factory:
            sock = factory.return_value.__enter__.return_value
            sock.connect_ex.return_value = 0
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                client.probe_endpoint('http://127.0.0.1:8095')
            sock.connect_ex.return_value = 111
            self.assertEqual(client.probe_endpoint('http://127.0.0.1:8095'), 8095)
        with self.assertRaises(ValueError):
            client.probe_endpoint('http://remote.invalid:8095')


class Wrapper(unittest.TestCase):
    def run_arm(self, mode):
        bash = shutil.which('bash')
        self.assertIsNotNone(bash, 'bash is required for wrapper regressions')
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            path = Path(tmp)
            fake = path / 'bin'
            fake.mkdir()
            scripts = {
                'python': '#!/bin/bash\n[ "$MODE" != occupied ] || exit 3\necho 8095\n',
                'nvidia-smi': '''#!/bin/bash
case " $* " in
  *" --id=1 "*) echo '1, 4000 MiB, 80 %, 1500 MHz' ;;
  *) printf '0, 200 MiB, 0 %%, 300 MHz\n1, 4000 MiB, 80 %%, 1500 MHz\n' ;;
esac
''',
                'serve': '''#!/bin/bash
echo launched > "$OUT/launched"
[ "$MODE" != bindfail ] || exit 2
echo 'strix-cuda-qwen-serve: loaded fixture effective configuration'
echo 'strix-cuda-qwen-serve: listening on 127.0.0.1:8095 model fixture'
if [ "$MODE" = death ]; then sleep 0.3; exit 2; fi
exec sleep 60
''',
                'client': '''#!/bin/bash
echo requested > "$OUT/requested"
echo 'cold prose C1 fixture' > "$2"
case "$MODE" in
  failure) exit 7 ;;
  timeout|death) sleep 3 ;;
  *) sleep 0.3 ;;
esac
''',
            }
            for name, source in scripts.items():
                (fake / name).write_text(source, encoding='utf-8', newline='\n')
                (fake / name).chmod(0o755)
            env = os.environ | dict(MODE=mode,
                                    SERVE_EXE='./bin/serve', SERVE_ARGS='fixture', ARMS='A|FIXTURE=1', ORDER='A',
                                    CLIENT='./bin/client http://127.0.0.1:8095', OUT='./out',
                                    GPU_INDEX='1', READY_S='2', LIMIT_S='1' if mode == 'timeout' else '5')
            result = subprocess.run([bash, '-c', 'export PATH="$PWD/bin:$PATH"; exec bash "$1"',
                                     'fixture', (SCRIPTS / 'qwen_serve_abab.sh').as_posix()], cwd=path, env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode == 0, mode == 'success', result.stderr[-1500:])
            if mode in ('occupied', 'bindfail'):
                self.assertFalse((path / 'out/requested').exists())
            if mode == 'occupied':
                self.assertFalse((path / 'out/launched').exists())
            if mode == 'success':
                summary = (path / 'out/summary.txt').read_text()
                self.assertIn('mean SM MHz 1500 (n=', summary)
                self.assertNotIn('(n=0)', summary)
                self.assertIn('mean util 80 %', summary)
                launch = (path / 'out/1-A/launch.txt').read_text()
                self.assertIn('FIXTURE=1', launch)
                self.assertRegex(launch, r'[0-9a-f]{64}')

    def test_arms(self):
        for mode in ('success', 'failure', 'timeout', 'occupied', 'bindfail', 'death'):
            with self.subTest(mode=mode):
                self.run_arm(mode)


if __name__ == '__main__':
    unittest.main()
