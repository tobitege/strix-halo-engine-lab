#!/usr/bin/env python3
"""Phase admission checks execute without SSH, models or GPUs."""
import re
import os
import pathlib
import shutil
import subprocess
import tempfile
from collections import Counter

script = pathlib.Path(__file__).resolve().parents[1] / 'tools/minimax/session_cache_gate.sh'
scratch = pathlib.Path(__file__).resolve().parents[1] / 'build/minimax-audit-p8/phasing'
scratch.mkdir(parents=True, exist_ok=True)
bash = shutil.which('bash')
assert bash, 'bash required'
subprocess.run([bash, '-n', str(script)], check=True)
with tempfile.TemporaryDirectory(prefix='mm-phases-', dir=scratch) as directory:
    root = pathlib.Path(directory)
    posix = subprocess.check_output([bash, '-c', 'pwd'], cwd=root, text=True).strip()
    env = dict(os.environ, OUT=posix, BIN='/absent', MODEL='/absent', DSPARK='/absent', CPU_TEST='/absent', NODE1='peer', STRIX_HASH_CACHE='/hash-cache')
    def refuses(phase, message):
        result = subprocess.run([bash, str(script), phase], env=env, capture_output=True, text=True, timeout=10)
        assert result.returncode == 2, result
        assert message in result.stdout, result
        assert not list(root.glob('turn-*')), 'phase started before admission checks'
    refuses('restore-on', 'missing prepare receipt')
    (root / 'prepare.done').write_text('prepare\n')
    refuses('restore-on', 'missing prewarm receipt')
    (root / 'prewarm.done').write_text('prewarm\n')
    refuses('restore-on', 'prewarm cache directory changed')
    (root / 'prewarm.cache-dir').write_text('/hash-cache\n')
    refuses('restore-on', 'save phase prerequisites missing')
    (root / 'save-on.done').write_text('save-on\n')
    refuses('restore-on', 'save phase prerequisites missing')
    (root / 'restore-on.done').write_text('restore-on\n')
    refuses('restore-on', 'phase already attempted')
# Execute the actual prewarm launch wrapper against a real staged binary and
# compare every hash-only path with the host's resolved dependency inventory.
source = script.read_text()
runner = re.search(r"    hash_runner='(.*?)'\n    localcmd=", source, re.S).group(1)
staged = pathlib.Path(shutil.which('echo')).resolve()
ldd = shutil.which('ldd')
assert ldd, 'ldd-compatible dependency resolver required'
deps = subprocess.run([ldd, str(staged)], capture_output=True, text=True, check=True).stdout
resolved = re.findall(r"/[^\s()]+", deps)
assert resolved and 'not found' not in deps, deps
def bash_path(value):
    cygpath = shutil.which('cygpath')
    if not cygpath:
        return str(value)
    return subprocess.check_output([cygpath, '-u', str(value)], text=True).strip()
with tempfile.TemporaryDirectory(prefix='mm-hash-runner-', dir=scratch) as directory:
    root = pathlib.Path(directory)
    fake = root / 'hash-engine'
    shutil.copyfile(staged, fake)
    result = subprocess.run([bash, '-c', runner, 'gate', bash_path(fake), '/model', '/draft', '0', bash_path(staged)],
                            capture_output=True, text=True, check=True)
    args = re.findall(r'--session-hash-file ([^\s]+)', result.stdout)
    expected = [bash_path(fake), *resolved, bash_path(staged), *resolved]
    assert Counter(args) == Counter(expected), (args, expected, deps)
    assert '--session-hash-only' in result.stdout and '--session-dspark' in result.stdout
    # A loader diagnostic with an unresolved dependency must refuse before the
    # staged binary is executed.
    ldd_stub = root / 'ldd'
    ldd_log = root / 'ldd-refusal.log'
    ldd_stub.write_text('#!/bin/sh\nprintf "%s\\n" "libmissing.so => not found" > "$LDD_LOG"\nprintf "%s\\n" "libmissing.so => not found"\n')
    ldd_stub.chmod(0o755)
    marker = root / 'executed'
    fake.write_text('#!/bin/sh\nprintf executed > "$HASH_MARKER"\n')
    fake.chmod(0o755)
    env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ.get('PATH', ''), HASH_MARKER=str(marker), LDD_LOG=str(ldd_log))
    refused = subprocess.run([bash, '-c', runner, 'gate', bash_path(fake), '/model', '/draft', '0'],
                             env=env, capture_output=True, text=True)
    assert refused.returncode == 2 and not marker.exists(), refused
    assert ldd_log.read_text().strip() == 'libmissing.so => not found'
    assert not marker.exists(), 'binary ran after dependency prerequisite refusal'
assert source.index('launch_pair prewarm') < source.index('prewarm.receipt')
assert 'SESSION_GATE_END - 30' in source and '+ 690' in source
print('PASS phase syntax, prerequisite refusal before setup, no restore without save evidence, immutable phase receipts')

# Both scheduler arms live inside each existing serve phase, with distinct witnesses.
assert 'default_pair_arm "$@" off' in source
assert 'default_pair_arm "$1-adaptive" "$2" "$3" "$4" default' in source
assert 'unset_env+=(-u STRIX_MM_SERVE_SCHED' in source
assert 'for suffix in "" "-adaptive"' in source
assert 'tag.removesuffix("-adaptive")' in source

# Execute both launch arrays with inherited and forwarded scheduler overrides.
functions = source[source.index('default_pair() {'):source.index('compare_default() {')]
with tempfile.TemporaryDirectory(prefix='mm-scheduler-', dir=scratch) as directory:
    root = pathlib.Path(directory)
    binary = root / 'engine'
    binary.write_text('#!/bin/bash\nprintf "%s %s %s\\n" "${STRIX_MM_SERVE_SCHED-unset}" "${STRIX_MM_SERVE_NEST-unset}" "${MM_DEFAULT_PROBE}"\n')
    binary.chmod(0o755)
    harness = functions + r'''
set -eu
OUT=$1; MODEL=model; DSPARK=draft; NODE1=peer
ENV_ARGS=(STRIX_MM_SERVE_SCHED=off STRIX_MM_SERVE_NEST=0)
export STRIX_MM_SERVE_SCHED=off STRIX_MM_SERVE_NEST=0
check_session_receipts() { :; }
copy() { :; }
launch_pair() { "${localcmd[@]}"; "${remote[@]}"; }
default_pair off-serve-off serve "$2" off
'''
    result = subprocess.run([bash, '-c', harness, 'gate', str(root), str(binary)],
                            capture_output=True, text=True, check=True, timeout=10)
    records = result.stdout.splitlines()
    assert len(records) == 4, records
    assert all(line.startswith('off 0 ') and line.endswith('/off-serve-off') for line in records[:2]), records
    assert all(line.startswith('unset unset ') and line.endswith('/off-serve-off-adaptive') for line in records[2:]), records
print('PASS serve phases execute scheduler-off and unmodified default scheduler arms')
