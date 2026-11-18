"""Script-only host regressions; original mode must expose the audited PR-head defect.

Configure this directory directly and run ctest --output-on-failure. No engine build,
GPU, model, network, registry change, scheduled task or native executable is required.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--work', type=Path, required=True)
parser.add_argument('--bash', required=True)
parser.add_argument('--powershell', default='')
parser.add_argument('--cmake', required=True)
parser.add_argument('--finding', type=int, required=True)
parser.add_argument('--version', choices=['fixed', 'original'], required=True)
parser.add_argument('--original-ref', default='')
args = parser.parse_args()
root = args.root.resolve()
work = args.work.resolve()
assert work.is_relative_to(root), 'all test artifacts must stay in the checkout'


def source(name):
    if args.version == 'original':
        return subprocess.check_output(['git', 'show', f'{args.original_ref}:{name}'], cwd=root).decode('utf-8')
    return (root / name).read_text(encoding='utf-8')


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')
    return path


def run(command, **kwargs):
    env = os.environ.copy()
    # Windows PowerShell must build its own module path when launched from pwsh.
    env.pop('PSModulePath', None)
    env.pop('PSMODULEPATH', None)
    env.update(TEMP=str(tmp), TMP=str(tmp), TMPDIR=str(tmp))
    env.update(kwargs.pop('env', {}))
    return subprocess.run(command, cwd=tmp, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=40, **kwargs)


def ok(result):
    assert result.returncode == 0, result.stdout[-2500:]


def ps_quote(path):
    return "'" + str(path).replace("'", "''") + "'"


def ps(code):
    script = write(tmp / 'driver.ps1', "$ErrorActionPreference = 'Stop'\n" + code +
                   '\nif ($null -ne $LASTEXITCODE) { exit $LASTEXITCODE }\n')
    return run([args.powershell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script)])


def shell_path(path):
    value = str(path.resolve()).replace('\\', '/')
    if os.name == 'nt':
        value = '/' + value[0].lower() + value[2:]
    return value


def bash(script, *argv, env=None):
    return run([args.bash, shell_path(script), *argv], env=env or {})


def cmake(script):
    return run([args.cmake, '-P', str(write(tmp / 'check.cmake', script))])


def test_1():
    package = tmp / 'package'
    prefix = tmp / 'installed'
    config = prefix / 'config'
    # The original uninstaller recursively removes this prefix. Its resolved path
    # is checked here before invocation, and every artifact is inside the checkout.
    assert prefix.resolve().is_relative_to(work)
    for name in ('install.ps1', 'uninstall.ps1', 'strix-start.ps1', 'smoke.ps1'):
        write(package / name, source('packaging/windows/' + name))
    write(package / 'bin/strix-test.exe', 'installed executable fixture')
    write(package / 'bin/strix-modified.exe', 'original bytes')
    write(package / 'share/strix/support/build-test.txt', 'host fixture')
    for name in ('serve.args', 'devices.example.toml'):
        write(package / 'etc/strix' / name, source('packaging/windows/' + name))
    write(package / 'etc/strix/serve.toml', source('packaging/serve.toml'))
    write(prefix / 'existing.txt', 'pre-existing data')
    write(config / 'serve.args', 'user settings')
    command = f'& {ps_quote(package / "install.ps1")} -Prefix {ps_quote(prefix)} -ConfigDir {ps_quote(config)}\n'
    ok(ps(command))
    # Reinstall exercises ownership retention without adopting added content.
    ok(ps(command))
    write(prefix / 'model.gguf', 'user model')
    write(prefix / 'bin/strix-modified.exe', 'user modification')
    result = ps('function Get-ScheduledTask {}\n' +
                f'& {ps_quote(package / "uninstall.ps1")} -Prefix {ps_quote(prefix)}\n')
    ok(result)
    for name, content in [('existing.txt', 'pre-existing data'), ('model.gguf', 'user model'),
                          ('config/serve.args', 'user settings'), ('bin/strix-modified.exe', 'user modification')]:
        path = prefix / name
        assert path.is_file() and path.read_text() == content, f'uninstall lost {name}'
    assert not (prefix / 'bin/strix-test.exe').exists(), 'owned executable not removed'
    assert not (prefix / 'share').exists(), 'empty owned directories not removed'
    # A colliding pre-existing executable must survive installation too.
    write(prefix / 'bin/strix-test.exe', 'pre-existing executable')
    assert ps(command).returncode != 0
    assert (prefix / 'bin/strix-test.exe').read_text() == 'pre-existing executable'


def test_2():
    unit = source('packaging/linux/strix.service.in')
    command = next(line.split('=', 1)[1] for line in unit.splitlines() if line.startswith('ExecStart='))
    argv = shlex.split(command.replace('${STRIX_MODEL}', 'model.gguf').replace('$STRIX_SERVE_ARGS', ''))
    # Contract check of the actual installed command, without loading a model.
    assert '--serve-config' not in argv, 'service uses an unimplemented option'
    assert '--http-serve' in argv, 'service does not enable HTTP'
    assert any(token.isdecimal() for token in argv[2:]), 'missing bootstrap token'
    assert argv[1] == 'model.gguf'


def test_3():
    packaging = source('packaging/StrixPackaging.cmake')
    block = packaging.split('    # DEB\n', 1)[1].split('    # RPM:', 1)[0]
    # Simulate the Debian database on this host, executing the actual dependency rules.
    block = block.replace('EXISTS "/etc/debian_version"', 'TRUE')
    header = '''cmake_minimum_required(VERSION 3.24)
macro(find_program)
endmacro()
set(STRIX_PACKAGE_BACKEND hip)
set(STRIX_DPKG_DEB present)
'''
    result = cmake(header + 'set(STRIX_DPKG_SHLIBDEPS present)\n' + block + '''
if(NOT CPACK_DEBIAN_HIP_PACKAGE_SHLIBDEPS)
    message(FATAL_ERROR "HIP ABI scan disabled")
endif()
if(CPACK_DEBIAN_HIP_PACKAGE_DEPENDS MATCHES "libc6|libstdc|libgcc-s1")
    message(FATAL_ERROR "unversioned runtimes replace the ABI scan")
endif()
''')
    ok(result)
    missing = cmake(header + 'set(STRIX_DPKG_SHLIBDEPS FALSE)\n' + block)
    assert missing.returncode != 0 and 'HIP DEBs require' in missing.stdout


def metadata(ds=True):
    # Execute the production CMake metadata generator with installation stubbed out.
    packaging = source('packaging/StrixPackaging.cmake')
    begin = packaging.find('    # Launcher and smoke share')
    if begin < 0:
        # Original artifacts had no metadata. Do not invent it in the fixture.
        return
    block = packaging[begin:packaging.index('    install(FILES "${_strix_pkg_dir}/windows/smoke.ps1"', begin)]
    dest = tmp / 'package/share/strix/support'
    dest.mkdir(parents=True, exist_ok=True)
    ok(cmake('cmake_minimum_required(VERSION 3.24)\nmacro(install)\nendmacro()\n' +
             f'set(_strix_pkg_out "{dest.as_posix()}")\nset(STRIX_CUDA_DS41_ONLY {"ON" if ds else "OFF"})\n' + block))


def test_4():
    package = tmp / 'package'
    config = package / 'config'
    write(config / 'serve.args', source('packaging/windows/serve.args'))
    write(package / 'bin/strix-cuda-ds41-replica.exe', 'not executed')
    metadata()
    # Run the launcher's real argument parser and selection, stopping before device or process work.
    launcher = source('packaging/windows/strix-start.ps1').split('$devices =', 1)[0]
    launcher += '\n@{binary=$Binary; argv=@($argv)} | ConvertTo-Json -Compress\n'
    path = write(package / 'select.ps1', launcher)
    result = ps(f'& {ps_quote(path)} -ConfigDir {ps_quote(config)}\n')
    ok(result)
    selection = json.loads(result.stdout)
    assert selection['binary'] == 'strix-cuda-ds41-replica'
    assert '--bind' not in selection['argv'] and '--port' not in selection['argv']


def package_checker():
    workflow = source('.github/workflows/packages.yml')
    start = workflow.index('      - name: Inspect package contents')
    end = workflow.index('\n      - name:', start + 1)
    lines = workflow[start:end].split('        run: |\n', 1)[1].splitlines()
    return write(tmp / 'check-packages.sh', '\n'.join(line[10:] for line in lines) + '\n')


def test_5():
    # Run the actual CI inspection step with deterministic archive-listing tools.
    checker = package_checker()
    write(tmp / 'packaging/smoke.sh', source('packaging/smoke.sh'))
    directory = tmp / 'build/pkg-host'
    names = ['strix-core.deb', 'strix-tools.deb', 'strix-core.rpm', 'strix-core.tar.gz']
    for name in names:
        write(directory / name, 'usr/bin/strix-tool\n')
    mock = tmp / 'mock-bin'
    for tool in ('dpkg-deb', 'rpm', 'tar'):
        path = write(mock / tool, '''#!/usr/bin/env bash
if [ "$#" -ne 2 ]; then exit 2; fi
case "$1" in -I|-qip) echo metadata; exit 0;; esac
if grep -q BROKEN "$2"; then exit 3; fi
cat "$2"
''')
        path.chmod(0o755)
    env = {'PATH': str(mock) + os.pathsep + os.environ['PATH']}
    ok(bash(checker, env=env))
    for name in names:
        archive = directory / name
        write(archive, 'usr/share/strix/lab/private.txt\n')
        assert bash(checker, env=env).returncode != 0, f'forbidden content accepted in {name}'
        write(archive, 'BROKEN\n')
        assert bash(checker, env=env).returncode != 0, f'listing error accepted in {name}'
        write(archive, 'usr/bin/strix-tool\n')


def test_6():
    script = write(tmp / 'smoke.sh', source('packaging/smoke.sh'))
    prefix = tmp / 'image/usr'
    (prefix / 'bin').mkdir(parents=True)
    manifest = prefix / 'share/strix/support/executables-tools.txt'
    write(manifest, 'strix-bitcalc-cmp\n')
    tool = write(prefix / 'bin/strix-bitcalc-cmp', '#!/usr/bin/env bash\necho "ModuleNotFoundError: No module named numpy" >&2\nexit 1\n')
    tool.chmod(0o755)
    command = ['--root', shell_path(tmp / 'image')]
    assert bash(script, *command).returncode != 0, 'missing numpy accepted as a successful load'
    write(tool, '#!/usr/bin/env bash\necho "usage: strix-bitcalc-cmp"\nexit 0\n')
    ok(bash(script, *command))
    write(tool, '#!/usr/bin/env bash\necho "unexpected failure" >&2\nexit 1\n')
    assert bash(script, *command).returncode != 0, 'arbitrary nonempty error accepted'
    write(tool, '#!/usr/bin/env bash\necho "usage: strix-bitcalc-cmp FILE" >&2\nexit 2\n')
    ok(bash(script, *command))
    for code in (1, 3, 64):
        write(tool, f'#!/usr/bin/env bash\necho "usage: strix-bitcalc-cmp FILE" >&2\nexit {code}\n')
        ok(bash(script, *command))
    for message, code in (('usage: another-tool FILE', 1),
                          ('failure\\nusage: strix-bitcalc-cmp FILE', 1),
                          ('usage: strix-bitcalc-cmp FILE', 124),
                          ('usage: strix-bitcalc-cmp FILE', 139),
                          ('usage: strix-bitcalc-cmp FILE\\nImportError: broken', 1)):
        write(tool, f'#!/usr/bin/env bash\nprintf "{message}\\n" >&2\nexit {code}\n')
        assert bash(script, *command).returncode != 0, (message, code)
    nohelp_manifest = write(manifest.with_name('without-help-tools.txt'),
                            'strix-inspect-gguf\nstrix-model-describe\nstrix-model-manifest\n')
    for name in nohelp_manifest.read_text().splitlines():
        nohelp_tool = prefix / 'bin' / name
        write(manifest, f'strix-bitcalc-cmp\n{name}\n')
        write(tool, '#!/usr/bin/env bash\necho "usage: strix-bitcalc-cmp"\n')
        for code in (1, 2):
            write(nohelp_tool, f'#!/usr/bin/env bash\n[ "$#" -eq 0 ] || exit 99\necho "model required" >&2\nexit {code}\n')
            nohelp_tool.chmod(0o755)
            ok(bash(script, *command))
        for message, code in (('', 1), ('model required', 0), ('failure', 3),
                              ('failure', 124), ('failure', 139),
                              ('error while loading shared libraries', 1)):
            write(nohelp_tool, f'#!/usr/bin/env bash\nprintf "{message}" >&2\nexit {code}\n')
            assert bash(script, *command).returncode != 0, (name, message, code)
        nohelp_tool.unlink()
    nohelp_manifest.unlink()
    write(manifest, 'strix-bitcalc-cmp\n')
    tool.unlink()
    assert bash(script, *command).returncode != 0, 'missing expected executable accepted'
    manifest.unlink()
    assert bash(script, *command).returncode != 0, 'empty bin accepted'


def test_7():
    package = tmp / 'package'
    write(package / 'bin/strix-cuda-ds41-replica.exe', 'not executed')
    manifest = package / 'share/strix/support/executables-cuda.txt'
    write(manifest, 'strix-cuda-ds41-replica.exe\n')
    path = write(package / 'smoke.ps1', source('packaging/windows/smoke.ps1'))
    metadata()
    log = tmp / 'processes.txt'
    # No executable is launched: emulate the process API and record its actual argv.
    mocks = '''function Start-Process {
param($FilePath, $ArgumentList, $RedirectStandardOutput, $RedirectStandardError,
      [switch]$NoNewWindow, [switch]$PassThru, $WindowStyle)
Add-Content -LiteralPath LOG_PATH -Value "$FilePath $ArgumentList"
if ($RedirectStandardOutput) { Set-Content -LiteralPath $RedirectStandardOutput -Value 'usage: fixture' }
if ($RedirectStandardError) { Set-Content -LiteralPath $RedirectStandardError -Value '' }
$p = [pscustomobject]@{ ExitCode = 0; HasExited = $true }
$p | Add-Member ScriptMethod WaitForExit { param($ms) return $true }
$p | Add-Member ScriptMethod Kill {}
return $p
}
'''.replace('LOG_PATH', ps_quote(log))
    result = ps(mocks + f'& {ps_quote(path)} -Prefix {ps_quote(package)} -EffectiveConfigArgs fixture\n')
    ok(result)
    calls = log.read_text(encoding='utf-8-sig')
    assert 'strix-cuda-ds41-replica.exe fixture --print-config' in calls, 'DeepSeek inspection skipped'
    health = ps(mocks + f'& {ps_quote(path)} -Prefix {ps_quote(package)} -ServeArgs fixture\n')
    assert health.returncode != 0 and 'no installed HTTP server' in health.stdout, health.stdout
    (package / 'bin/strix-cuda-ds41-replica.exe').unlink()
    missing = ps(mocks + f'& {ps_quote(path)} -Prefix {ps_quote(package)} -EffectiveConfigArgs fixture\n')
    assert missing.returncode != 0, 'explicit inspection without a server skipped'
    # Qwen inspection must reach the executable with its supported CLI option.
    metadata(ds=False)
    write(package / 'bin/strix-cuda-qwen-serve.exe', 'not executed')
    write(manifest, 'strix-cuda-qwen-serve.exe\n')
    inspected = ps(mocks + f'& {ps_quote(path)} -Prefix {ps_quote(package)} -EffectiveConfigArgs fixture\n')
    ok(inspected)
    assert 'strix-cuda-qwen-serve.exe fixture --print-effective-config' in log.read_text(encoding='utf-8-sig')


def test_9():
    package, prefix, config = tmp / 'package', tmp / 'installed', tmp / 'config'
    installer = write(package / 'install.ps1', source('packaging/windows/install.ps1'))
    for name in ('a', 'b'):
        write(package / f'bin/strix-{name}.exe', f'payload {name}')
    for name in ('serve.toml', 'serve.args', 'devices.example.toml'):
        write(package / 'etc/strix' / name, 'configuration fixture')
    command = f'& {ps_quote(installer)} -Prefix {ps_quote(prefix)} -ConfigDir {ps_quote(config)}\n'
    copy_failure = '''$script:copies = 0
function Copy-Item {
param($LiteralPath, $Path, $Destination, [switch]$Force)
$script:copies++
if ($script:copies -eq 2) { throw 'injected second-copy failure' }
if ($LiteralPath) { Microsoft.PowerShell.Management\\Copy-Item -LiteralPath $LiteralPath -Destination $Destination -Force:$Force }
else { Microsoft.PowerShell.Management\\Copy-Item -Path $Path -Destination $Destination -Force:$Force }
}
'''
    failed = ps(copy_failure + command)
    assert failed.returncode != 0 and 'injected second-copy failure' in failed.stdout
    assert not list((prefix / 'bin').glob('*.exe')), 'failed copy left unowned payload'
    ok(ps(command))
    manifest = prefix / 'install-manifest.json'
    old_manifest = manifest.read_bytes()
    old_payload = {p.name: p.read_bytes() for p in (prefix / 'bin').glob('*.exe')}
    write(package / 'bin/strix-a.exe', 'upgraded a')
    assert ps(copy_failure + command).returncode != 0
    assert manifest.read_bytes() == old_manifest, 'failed upgrade changed ownership'
    assert {p.name: p.read_bytes() for p in (prefix / 'bin').glob('*.exe')} == old_payload
    # Failure while preparing the new manifest must restore the whole old payload.
    manifest_failure = '''function Set-Content {
param($LiteralPath, $Path, $Value, [Parameter(ValueFromPipeline=$true)]$InputObject)
process {
if ($LiteralPath -like '*manifest.json') { throw 'injected manifest failure' }
Microsoft.PowerShell.Management\\Set-Content -Path $Path -Value $Value
}
}
'''
    failed = ps(manifest_failure + command)
    assert failed.returncode != 0 and 'injected manifest failure' in failed.stdout
    assert manifest.read_bytes() == old_manifest
    assert {p.name: p.read_bytes() for p in (prefix / 'bin').glob('*.exe')} == old_payload
    ok(ps(command))
    # OS integration fails after commit; a normal reinstall must still succeed.
    failed = ps("function New-ScheduledTaskAction { throw 'injected task failure' }\n" + command.rstrip() + ' -RegisterTask\n')
    assert failed.returncode != 0 and 'injected task failure' in failed.stdout
    ok(ps(command))
    invalid_config = write(tmp / 'config-file', 'existing file')
    fresh = tmp / 'fresh'
    failed = ps(f'& {ps_quote(installer)} -Prefix {ps_quote(fresh)} -ConfigDir {ps_quote(invalid_config)}\n')
    assert failed.returncode != 0
    assert not fresh.exists(), 'invalid ConfigDir mutated the prefix'


def test_10():
    package = tmp / 'package'
    script = write(package / 'smoke.ps1', source('packaging/windows/smoke.ps1'))
    manifest = write(package / 'share/strix/support/executables-cuda.txt', 'strix-a.exe\nstrix-b.exe\n')
    a = write(package / 'bin/strix-a.exe', 'never executed')
    b = write(package / 'bin/strix-b.exe', 'never executed')
    mocks = '''function Start-Process {
param($FilePath, $ArgumentList, $RedirectStandardOutput, $RedirectStandardError,
      [switch]$NoNewWindow, [switch]$PassThru)
Set-Content -LiteralPath $RedirectStandardOutput -Value 'usage: fixture'
Set-Content -LiteralPath $RedirectStandardError -Value ''
$p = [pscustomobject]@{ ExitCode = 0 }
$p | Add-Member ScriptMethod WaitForExit { param($ms) return $true }
return $p
}
'''
    command = mocks + f'& {ps_quote(script)} -Prefix {ps_quote(package)}\n'
    ok(ps(command))
    b.unlink()
    assert ps(command).returncode != 0, 'missing one expected executable accepted'
    a.unlink()
    assert ps(command).returncode != 0, 'missing all expected executables accepted'
    write(manifest, '')
    assert ps(command).returncode != 0, 'empty expected-executable manifest accepted'
    manifest.unlink()
    assert ps(command).returncode != 0, 'missing expected-executable manifest accepted'


def test_8():
    matrix = write(tmp / 'support.cmake', source('packaging/StrixSupportMatrix.cmake'))
    script = f'''cmake_minimum_required(VERSION 3.24)
include("{matrix.as_posix()}")
function(check system processor backend arch expected)
    set(CMAKE_SYSTEM_NAME "${{system}}")
    set(CMAKE_SYSTEM_PROCESSOR "${{processor}}")
    strix_support_status("${{backend}}" "${{arch}}" actual)
    if(NOT actual STREQUAL expected)
        message(FATAL_ERROR "${{system}}/${{processor}}/${{backend}}/${{arch}}: ${{actual}}, wanted ${{expected}}")
    endif()
endfunction()
check(Linux x86_64 cuda 120 unqualified)
check(Windows AMD64 cuda 120 qualified)
check(Windows AMD64 cuda sm_120-real qualified)
check(Windows ARM64 cuda 120 unqualified)
check(Windows AMD64 cuda 89 unqualified)
check(Linux x86_64 hip gfx1151 qualified)
check(Windows AMD64 hip gfx1151 unqualified)
check(Linux x86_64 hip gfx950 unqualified)
'''
    ok(cmake(script))


with tempfile.TemporaryDirectory(prefix=f'package-685-{args.finding}-', dir=work) as directory:
    tmp = Path(directory).resolve()
    assert tmp.is_relative_to(work)
    try:
        globals()[f'test_{args.finding}']()
    except AssertionError:
        if args.version != 'original':
            raise
        print(f'finding {args.finding}: original defect reproduced')
    else:
        assert args.version == 'fixed', f'finding {args.finding}: regression did not fail on original source'
        print(f'finding {args.finding}: fixed regression passed')
