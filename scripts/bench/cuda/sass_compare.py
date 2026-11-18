#!/usr/bin/env python3
"""Compare CUDA build trees by relative artifact path, architecture and function.

All instructions and symbol identities are preserved. Explicit one-to-one mappings
allow reviewed artifact moves (--rename OLD=NEW, relative paths) and full symbol
moves (--symbol-rename OLD=NEW, including path-derived anonymous namespaces).
Symbol mappings apply to baseline definitions and call targets, never substrings.

  sass_compare.py --base BUILD_A --new BUILD_B [--base ... --new ...]
                  [--rename old/path.cu.obj=new/path.cu.obj]
                  [--symbol-rename OLD=NEW] [--cuobjdump PATH] [--out report.txt] [--no-exe]

Exit 0 requires nonzero CUDA coverage in every pair and no differences or errors.
Paired executables and explicitly declared added CUDA objects (--new-host-only PATH)
are treated as host-only only after successful, empty disassembly.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

OBJ = ('.cu.obj', '.cu.o')
SYMBOL = re.compile(r'[A-Za-z_$][0-9A-Za-z_$]*')
ARCH = re.compile(r'(?:arch\s*=|code for)\s*(sm_[0-9a-z]+)\b')
INSTRUCTION = re.compile(r'/\*[0-9a-fA-F]+\*/\s+(?:@!?\w+\s+)?[A-Z][A-Z0-9.]*\b')


def collect(root, with_exe):
    if not os.path.isdir(root):
        raise ValueError('not a build directory: ' + root)
    found = {}

    def walk_error(error):
        raise error

    for d, _, names in os.walk(root, onerror=walk_error):
        for n in names:
            path = os.path.join(d, n)
            binary = n.endswith(('.exe', '.dll', '.so')) or ('.' not in n and os.access(path, os.X_OK))
            if n.endswith(OBJ) or (with_exe and binary):
                found[os.path.relpath(path, root).replace(os.sep, '/')] = path
    return found


def parse_sass(output, symbols):
    funcs, body, name, arch = {}, [], None, None

    def normalize(text):
        return SYMBOL.sub(lambda m: symbols.get(m.group(), m.group()), text)

    def finish():
        if name is None:
            return
        if arch is None or not any(INSTRUCTION.search(line) for line in body):
            raise ValueError('missing architecture or body for ' + name)
        key = (arch, normalize(name))
        if key in funcs:
            raise ValueError('duplicate/normalization collision: ' + repr(key))
        funcs[key] = '\n'.join(body)

    for line in output.splitlines():
        s = line.strip()
        match = ARCH.fullmatch(s)
        if s.startswith('Fatbin') or match:
            finish()
            name, body = None, []
            arch = match.group(1) if match else None
        elif s.startswith('Function :'):
            finish()
            name, body = s.split('Function :', 1)[1].strip(), []
            if not name:
                raise ValueError('empty function name')
        elif name is not None and s and not s.startswith(('---', '=====')):
            body.append(normalize(s))
    finish()
    if not funcs and output.strip():
        raise ValueError('unrecognized or incomplete disassembly')
    return funcs


def sass(cuobjdump, path, symbols=None):
    r = subprocess.run([cuobjdump, '-sass', path], capture_output=True, text=True, errors='strict')
    if r.returncode != 0:
        raise ValueError('cuobjdump failed (%d): %s' % (r.returncode, path))
    # A successful empty dump is allowed only for paired non-object binaries.
    return parse_sass(r.stdout, symbols or {})


def mappings(values, paths=False):
    result = {}
    for value in values:
        old, sep, new = value.partition('=')
        if not sep or not old or not new:
            raise ValueError('expected OLD=NEW: ' + value)
        if paths:
            old, new = old.replace('\\', '/'), new.replace('\\', '/')
            if any(p.startswith('/') or ':' in p or any(x in ('', '.', '..') for x in p.split('/'))
                   for p in (old, new)):
                raise ValueError('mapping must use relative artifact paths: ' + value)
        elif not SYMBOL.fullmatch(old) or not SYMBOL.fullmatch(new):
            raise ValueError('mapping must use complete symbols: ' + value)
        if old in result or new in result.values():
            raise ValueError('ambiguous mapping: ' + value)
        result[old] = new
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', action='append', required=True)
    ap.add_argument('--new', action='append', required=True)
    ap.add_argument('--rename', action='append', default=[], help='old/path=new/path for moved artifacts')
    ap.add_argument('--symbol-rename', action='append', default=[], help='exact old_symbol=new_symbol for reviewed moves')
    ap.add_argument('--new-host-only', action='append', default=[], metavar='PATH',
                    help='added relative CUDA object path requiring successful empty disassembly')
    ap.add_argument('--cuobjdump', default=shutil.which('cuobjdump') or 'cuobjdump')
    ap.add_argument('--out')
    ap.add_argument('--no-exe', action='store_true', help='objects only (faster)')
    a = ap.parse_args()
    if len(a.base) != len(a.new):
        ap.error('--base and --new come in pairs')
    try:
        rename = mappings(a.rename, paths=True)
        symbols = mappings(a.symbol_rename)
        host_only = mappings([path + '=' + path for path in a.new_host_only], paths=True)
    except ValueError as error:
        ap.error(str(error))
    lines, same, diff = [], 0, 0
    for b, n in zip(a.base, a.new):
        coverage = 0
        try:
            fb, fn = collect(b, not a.no_exe), collect(n, not a.no_exe)
            targets = [rename.get(x, x) for x in fb]
            if len(set(targets)) != len(targets):
                raise ValueError('artifact mapping collision')
            added = set(fn) - set(targets)
            for name in host_only:
                if name not in added or not name.endswith(OBJ):
                    raise ValueError('host-only declaration is not an added object: ' + name)
        except (OSError, ValueError) as error:
            diff += 1
            lines.append('ERROR ' + str(error))
            continue
        for name, path in sorted(fb.items()):
            tgt = rename.get(name, name)
            if tgt not in fn:
                diff += 1
                lines.append('BASE-ONLY ' + name)
                continue
            try:
                sb, sn = sass(a.cuobjdump, path, symbols), sass(a.cuobjdump, fn[tgt])
                if (not sb and name.endswith(OBJ)) or (not sn and tgt.endswith(OBJ)):
                    raise ValueError('CUDA object has no SASS: ' + name)
            except (OSError, ValueError) as error:
                diff += 1
                lines.append('ERROR ' + str(error))
                continue
            if not sb and not sn:
                lines.append('HOST-ONLY ' + name)
                continue
            coverage += 1
            if sb == sn:
                same += 1
                continue
            diff += 1
            kb, kn = set(sb), set(sn)
            bad = sorted(k for k in kb & kn if sb[k] != sn[k])
            lines.append('DIFF %s: functions %d/%d, only-base %d, only-new %d, body differs %d %s'
                         % (name, len(kb), len(kn), len(kb - kn), len(kn - kb), len(bad), bad[:2]))
        for name in sorted(set(fn) - set(targets)):
            if name in host_only:
                try:
                    if sass(a.cuobjdump, fn[name]):
                        raise ValueError('declared host-only object has SASS: ' + name)
                except (OSError, ValueError) as error:
                    diff += 1
                    lines.append('ERROR ' + str(error))
                else:
                    lines.append('NEW-HOST-ONLY ' + name)
                continue
            diff += 1
            lines.append('NEW-ONLY ' + name)
        if not coverage:
            diff += 1
            lines.append('ERROR no CUDA coverage: %s -> %s' % (b, n))
    summary = 'SASS compare: %d equal, %d different/errors' % (same, diff)
    report = summary + '\n' + '\n'.join(lines) + '\n'
    if a.out:
        with open(a.out, 'w', newline='\n') as f:
            f.write(report)
    print(report if len(report) < 4000 else summary)
    return 1 if diff else 0


if __name__ == '__main__':
    sys.exit(main())
