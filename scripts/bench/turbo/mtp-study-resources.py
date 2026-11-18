#!/usr/bin/env python3
"""CPU-only IU4 R7/R8 resource audit. Extracts HIP code objects via llvm-objdump."""
import argparse
import collections
import json
from pathlib import Path
import re
import subprocess


def run(argv):
    return subprocess.check_output([str(x) for x in argv], text=True, stderr=subprocess.STDOUT)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--llvm', type=Path, default=Path('/opt/rocm/llvm/bin'))
    p.add_argument('--arch', default='gfx1151')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    extracted = run([a.llvm / 'llvm-objdump', '--offloading', a.binary.resolve()])
    (a.output / 'offloading.txt').write_text(extracted)
    objects = [Path(x.strip()) for x in re.findall(r'Extracting offload bundle:\s*(.+)', extracted)
               if 'hip-amdgcn-amd-amdhsa--' + a.arch in x]
    if not objects:
        raise RuntimeError('No matching extracted HIP code object; inspect offloading.txt')
    fields = ['private_segment_fixed_size', 'sgpr_spill_count', 'vgpr_spill_count',
              'vgpr_count', 'sgpr_count', 'group_segment_fixed_size', 'wavefront_size']
    records = []
    for obj in objects:
        notes = run([a.llvm / 'llvm-readelf', '--notes', obj])
        blocks = []
        # LDS fields precede .name in AMDGPU YAML, while private/VGPR fields
        # follow it. Keep the complete kernel record, not just the name's tail.
        for m in re.finditer(r'^  - \.args:.*?(?=^  - \.args:|\Z)', notes, re.M | re.S):
            block = m[0]
            name = re.search(r'\.name:\s+(\S*project_rows_iu4_kernel\S+)', block)
            if name:
                blocks.append((name[1], block))
        if not blocks:
            continue
        (a.output / (obj.name + '.notes')).write_text(notes)
        # One demangler process per object, avoiding hundreds of fork/execs.
        demangled = subprocess.check_output([str(a.llvm / 'llvm-cxxfilt')],
            input='\n'.join(x[0] for x in blocks) + '\n', text=True).splitlines()
        for (sym, block), dem in zip(blocks, demangled):
            template = re.search(r', (\d+)u, (\d+)u, (\d+)u, (true|false)(?:, (true|false))?>', dem)
            row = re.search(r'RowsSite<(\d+)u', dem)
            bools = re.findall(r'Lb([01])E', sym)
            rec = {'object': str(obj), 'symbol': sym, 'name': dem,
                   'rows': int(row[1]) if row else None, 'mangled_bool_arguments': bools}
            if not template:
                raise RuntimeError('Cannot parse IU4 template: ' + dem)
            rec.update(S=int(template[1]), G=int(template[2]), GC=int(template[3]),
                       from_slot=template[4] == 'true', single_decode=template[5] == 'true')
            rec['site'] = dem.split('project_rows_iu4_kernel<', 1)[1].split('>, ', 1)[0] + '>'
            for f in fields:
                m = re.search(r'\.' + f + r':\s+(\d+)', block)
                rec[f] = int(m[1]) if m else None
            records.append(rec)
    if not records:
        raise RuntimeError('No IU4 metadata found')
    # Read ISA for one matched R8 GDN pair, retaining static instruction evidence.
    chosen = [r for r in records if r['rows'] == 8 and r['G'] == 40 and r['from_slot']
              and 'GdnInputRowsSite' in r['name']]
    for index, rec in enumerate(chosen[:2]):
        isa = run([a.llvm / 'llvm-objdump', '-d', '--disassemble-symbols=' + rec['symbol'], rec['object']])
        target = a.output / ('r8-' + ('single' if rec['single_decode'] else 'halves') + '-' + str(index) + '.isa')
        target.write_text(isa)
        count = collections.Counter()
        for line in isa.splitlines():
            m = re.match(r'^\s+([sv][a-z0-9_.]+|ds_[a-z0-9_]+|global_[a-z0-9_]+|flat_[a-z0-9_]+|buffer_[a-z0-9_]+)\s', line)
            if m:
                count[m[1]] += 1
        rec['isa_file'] = str(target)
        rec['static_mnemonics'] = dict(count)
        rec['static_instructions'] = sum(count.values())
    summaries = []
    for rows in [7, 8]:
        for single in [False, True]:
            arm = [r for r in records if r['rows'] == rows and r['single_decode'] == single]
            if not arm:
                continue
            summary = {'rows': rows, 'single_decode': single, 'variants': len(arm)}
            for f in fields:
                vals = [r[f] for r in arm if r[f] is not None]
                summary['max_' + f] = max(vals) if vals else None
            summary['spilling_variants'] = sum(bool(r['private_segment_fixed_size']) or
                bool(r['vgpr_spill_count']) or bool(r['sgpr_spill_count']) for r in arm)
            summaries.append(summary)
    (a.output / 'resources.json').write_text(json.dumps(records, indent=2) + '\n')
    (a.output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    for s in summaries:
        print(json.dumps(s, sort_keys=True))
    for r in sorted((r for r in records if r['single_decode'] and r['rows'] in (7, 8)),
                    key=lambda r: (r['private_segment_fixed_size'] or 0, r['vgpr_count'] or 0), reverse=True)[:3]:
        print(json.dumps({k: r.get(k) for k in ['site', 'rows', 'S', 'G', 'GC', 'from_slot',
              'single_decode', 'private_segment_fixed_size', 'vgpr_count', 'vgpr_spill_count']}))
    print('ISA read:', len(chosen[:2]), 'matched R8 GDN variants; full evidence:', a.output)


if __name__ == '__main__':
    main()
