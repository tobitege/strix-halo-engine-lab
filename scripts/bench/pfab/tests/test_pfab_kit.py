"""Host-only pair runner fixtures; SSH is a local process stub, never a network call."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
KIT = ROOT / 'scripts/bench/pfab'


def fixture_dir():
    return tempfile.TemporaryDirectory(dir=os.environ.get('PFAB_TEST_TMP'))


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('pf_clk', KIT / 'pf_clk.py')
        cls.clk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.clk)

    def summary(self, path, expected=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = self.clk.main([str(path)], expected) if expected is not None else self.clk.main([str(path)])
        return rc, out.getvalue()

    @staticmethod
    def log(path, rank, count):
        (path / f'r{rank}.tlog').write_text(''.join(
            f'{100 + i}.000 rank {rank} pf gemv.{"B" if i % 2 == 0 else "C"}-ttft: 32 tokens 1.0 s logits fnv abcd\n'
            for i in range(count)))

    def test_missing_and_empty_logs_fail(self):
        with fixture_dir() as tmp:
            path = Path(tmp)
            self.assertNotEqual(self.summary(path)[0], 0)
            self.log(path, 0, 0)
            self.log(path, 1, 0)
            self.assertNotEqual(self.summary(path)[0], 0)

    def test_expected_counts_on_both_ranks(self):
        with fixture_dir() as tmp:
            path = Path(tmp)
            for counts in ((4, 3), (3, 3), (5, 4), (4, 4)):
                for rank, count in enumerate(counts):
                    self.log(path, rank, count)
                self.assertEqual(self.summary(path, 4)[0] == 0, counts == (4, 4))

    def test_all_later_prefills_are_warm(self):
        with fixture_dir() as tmp:
            path = Path(tmp)
            for rank in (0, 1):
                self.log(path, rank, 4)
            _, output = self.summary(path)
            for rank in (0, 1):
                rows = [line.split() for line in output.splitlines() if f' r{rank} ' in line]
                self.assertEqual([row[3] for row in rows], ['cold', 'warm', 'warm', 'warm'])
                self.assertEqual([row[4] for row in rows], ['gemv.B', 'gemv.C', 'gemv.B', 'gemv.C'])


def mock_ssh(args):
    while args and args[0].startswith('-'):
        args = args[2:] if args[0] == '-o' else args[1:]
    if args.pop(0) != 'fixture-peer':
        return 90
    try:
        command = shlex.split(args[0])
    except ValueError:
        return 91
    local, remote = os.environ['PFAB_OUT'], os.environ['PFAB_REMOTE_OUT']
    command = [remote + arg[len(local):] if arg == local or arg.startswith(local + '/') else arg for arg in command]
    if command[0] == 'sha256sum':
        print(hashlib.sha256(Path(command[-1]).read_bytes()).hexdigest(), command[-1])
        return 0
    if command[0] == 'cat':
        if os.environ['PFAB_SCENARIO'] == 'transfer' and command[-1].endswith('samp1.txt'):
            return 23
        try:
            sys.stdout.buffer.write(Path(command[-1]).read_bytes())
        except OSError:
            return 24
        return 0
    if command[0] != 'bash':
        # Also support the pre-fix script for regression checks, entirely locally.
        return subprocess.call([os.environ['PFAB_BASH'], '-c', args[0].replace(local, remote)])
    return subprocess.call([os.environ['PFAB_BASH'], *command[1:]])


def mock_scp(args):
    while args and args[0].startswith('-'):
        args = args[2:] if args[0] == '-o' else args[1:]
    local, remote = os.environ['PFAB_OUT'], os.environ['PFAB_REMOTE_OUT']
    paths = [arg.removeprefix('fixture-peer:').replace(local, remote) if arg.startswith('fixture-peer:') else arg
             for arg in args]
    for source in paths[:-1]:
        if os.environ['PFAB_SCENARIO'] == 'transfer' and source.endswith('samp1.txt'):
            return 23
        try:
            shutil.copy2(source, paths[-1])
        except OSError:
            return 24
    return 0


def mock_rank(args):
    rank = int(args[args.index('--rank') + 1])
    Path(os.environ['PFAB_ARGV'] + str(rank)).write_text(json.dumps(args))
    prompts = args[args.index('--pf-ttft') + 1].split(',')
    arms = ['gemv']
    reps = 1
    if '--pf-ttft-arms' in args:
        arms = ['gemv.' + arm.split(':')[0] for arm in re.split('[;/]', args[args.index('--pf-ttft-arms') + 1])]
        reps = int(args[args.index('--pf-ttft-reps') + 1])
    rows = [f'rank {rank} pf {arm}-ttft: 32 tokens 1.0 s logits fnv abcd'
            for _ in range(reps) for arm in arms for _ in prompts]
    scenario = os.environ['PFAB_SCENARIO']
    if scenario == 'incomplete':
        rows.pop()
    print('\n'.join(rows), flush=True)
    # Emit complete valid logs even on failure: this isolates exit propagation.
    return 17 if scenario == f'rank{rank}' else 124 if scenario == 'timeout' and rank == 0 else 0


def mock_python(args):
    if 'sample' in args and any(arg.endswith('pf_samp.py') for arg in args):
        Path(args[args.index('sample') + 1]).write_text('100.0 1000 50 0 0 1024 2048\n')
        return 0
    return subprocess.call([sys.executable, *args])


class PairTests(unittest.TestCase):
    def run_pair(self, scenario='ok', arms='B:-;C:STRIX_GLM_PF_MLA_WMMA=0'):
        with fixture_dir() as tmp:
            path = Path(tmp)
            shims = path / 'shims'
            shims.mkdir()
            # Exercise spaces, apostrophes, semicolons, glob and substitution syntax as data.
            special = " space's;$(false)[x]" if scenario == 'ok' else ''
            if scenario != 'ok':
                arms = arms.replace(';', '/')
            binary = path / ('rank' + special)
            model = (path / ('model' + special)).as_posix()
            prompts = (path / ('prompt' + special)).as_posix() + ',' + (path / ('second' + special)).as_posix()
            out = (path / ('output' + special)).as_posix()
            remote = (path / ('peer-output' + special)).as_posix()
            env = os.environ.copy()
            env.update(PFAB_PYTHON=sys.executable.replace('\\', '/'), PFAB_TEST_DRIVER=Path(__file__).as_posix(),
                       PFAB_OUT=out, PFAB_REMOTE_OUT=remote, PFAB_ARGV=str(path / 'argv'),
                       PFAB_SCENARIO=scenario, PFAB_BASH=BASH)
            env['PATH'] = str(shims) + os.pathsep + str(Path(BASH).parent) + os.pathsep + env['PATH']
            for name, operation in (('ssh', 'ssh'), ('scp', 'scp'), ('python3', 'python')):
                shim = shims / name
                shim.write_text('#!/usr/bin/env bash\nexec "$PFAB_PYTHON" "$PFAB_TEST_DRIVER" ' + operation + ' "$@"\n')
                shim.chmod(0o755)
            binary.write_text('#!/usr/bin/env bash\nexec "$PFAB_PYTHON" "$PFAB_TEST_DRIVER" rank "$@"\n')
            binary.chmod(0o755)
            args = [BASH, str(KIT / 'pf_pair_ab.sh'), '--bin', binary.as_posix(), '--model', model,
                    '--prompts', prompts, '--out', out, '--peer', 'fixture-peer', '--budget', '60']
            if arms:
                args += ['--arms', arms, '--reps', '2']
            run = subprocess.run(args, env=env, cwd=path, capture_output=True, text=True, timeout=35)
            detail = (run.stdout + run.stderr)[-2000:]
            if scenario == 'ok':
                self.assertEqual(run.returncode, 0, detail)
                for rank in (0, 1):
                    actual = json.loads(Path(env['PFAB_ARGV'] + str(rank)).read_text())
                    self.assertEqual(actual[0], model)
                    self.assertEqual(actual[actual.index('--pf-ttft') + 1], prompts)
                    if arms:
                        self.assertEqual(actual[actual.index('--pf-ttft-arms') + 1], arms)
                        self.assertEqual(actual[actual.index('--pf-ttft-reps') + 1], '2')
                self.assertEqual(run.stdout.count(' tok/s '), 16 if arms else 4, detail)
            else:
                self.assertNotEqual(run.returncode, 0, detail)
                self.assertIn('== bin ', run.stdout, detail)  # Reached cleanup and summary.

    def test_semicolon_arms_and_quoted_paths(self):
        self.run_pair()

    def test_slash_arms(self):
        self.run_pair(arms='B:-/C:STRIX_GLM_PF_MLA_WMMA=0')

    def test_no_arms(self):
        self.run_pair(arms='')

    def test_failed_runs_and_transfer(self):
        for scenario in ('rank0', 'rank1', 'timeout', 'transfer', 'incomplete'):
            with self.subTest(scenario=scenario):
                self.run_pair(scenario)


if __name__ == '__main__':
    mode, *arguments = sys.argv[1:]
    if mode in ('ssh', 'scp', 'rank', 'python'):
        sys.exit({'ssh': mock_ssh, 'scp': mock_scp, 'rank': mock_rank, 'python': mock_python}[mode](arguments))
    BASH = arguments[0] if arguments else None
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SummaryTests if mode == 'summary' else PairTests)
    sys.exit(not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful())
