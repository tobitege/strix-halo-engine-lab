"""Offline sm_120 KDA instruction evidence; no CUDA runtime calls."""
import argparse
import sys
import json
from pathlib import Path
import re

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('sass',type=Path);ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args()
records=[]
for section in re.split(r'Function\s*:\s*',(sys.stdin.read() if str(a.sass) == '-' else a.sass.read_text()))[1:]:
    name=section.splitlines()[0]
    ops={op:[int(x,16) for x in re.findall(r'/\*([0-9a-f]+)\*/[^\n]*\b'+op+r'\b',section)] for op in ('HMMA','LDGSTS','FFMA','LDL','STL')}
    loops=[]
    for pc,target in re.findall(r'/\*([0-9a-f]+)\*/[^\n]*\bBRA(?:\.U)?(?:\s+!?UP\d+,)?\s+0x([0-9a-f]+)',section):
        pc,target=int(pc,16),int(target,16)
        if target<pc and any(target<=x<=pc for x in ops['HMMA']):
            loops.append(dict(begin=hex(target),back_edge=hex(pc),hmma=sum(target<=x<=pc for x in ops['HMMA']),
                              ldgsts=sum(target<=x<=pc for x in ops['LDGSTS'])))
    if 'GemmGrouped' in name:
        label='projection_column' if 'ColumnMajorTensorOpMultiplicandCrosswise' in name else 'projection_row'
        if not any(l['ldgsts'] for l in loops):raise SystemExit('projection lacks MMA/cp.async K loop')
    elif 'triangular' in name:label='triangular'
    elif 'base_products' in name:label='incoming_state_products'
    elif 'block_productsILb1' in name:label='state_carry_products'
    elif 'block_productsILb0' in name:label='output_products'
    elif 'sequential' in name:label='sequential_reference'
    elif 'projection_reference' in name:label='projection_reference'
    else:continue
    if label.endswith('reference'):
        if ops['HMMA'] or not ops['FFMA']:raise SystemExit('reference must have scalar FFMA, no HMMA')
    elif not ops['HMMA']:raise SystemExit('missing native MMA: '+label)
    if label in ('triangular','incoming_state_products','output_products') and not loops:
        raise SystemExit('missing confirmed chunk MMA K backedge: '+label)
    # The fixed two-tile carry loop may be fully unrolled; verify both tiles.
    if label=='state_carry_products' and not loops and len(ops['HMMA'])!=4:
        raise SystemExit('carry needs two 16-token MMA tiles')
    records.append(dict(site=label,symbol=name,counts={op:len(v) for op,v in ops.items()},
                        hmma_addresses=[hex(v) for v in ops['HMMA']],loops=loops))
expected={'projection_row','projection_column','triangular','incoming_state_products','state_carry_products','output_products','sequential_reference','projection_reference'}
if len(records)!=len(expected) or {r['site'] for r in records}!=expected:raise SystemExit('missing/duplicate KDA kernels')
a.out.write_text(json.dumps(dict(architecture='sm_120',sites=records),indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ('site','counts','loops')} for r in records]))
