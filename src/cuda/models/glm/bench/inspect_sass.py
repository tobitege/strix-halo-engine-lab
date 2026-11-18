"""Offline check: HMMA inside native K back edges, absent from SIMT reference."""
import argparse
import sys
import json
from pathlib import Path
import re

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('sass',type=Path)
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args()
sections=re.split(r'Function\s*:\s*',(sys.stdin.read() if str(a.sass) == '-' else a.sass.read_text()))
records=[]
cutlass_records=[]
decoder_records=[]
for section in sections[1:]:
    name=section.splitlines()[0]
    if 'cutlass' in name and 'GemmGrouped' in name:
        tile=re.search(r'GemmShapeILi(32|128)ELi128ELi32',name)
        if not tile:
            raise SystemExit('Unexpected CUTLASS tile')
        ops={op:[int(x,16) for x in re.findall(
            r'/\*([0-9a-f]+)\*/[^\n]*\b'+op,section)] for op in ('HMMA','LDGSTS')}
        loops=[]
        for pc,target in re.findall(r'/\*([0-9a-f]+)\*/[^\n]*\bBRA\s+0x([0-9a-f]+)',section):
            pc,target=int(pc,16),int(target,16)
            counts={op:sum(target<=x<=pc for x in addresses) for op,addresses in ops.items()}
            if target<pc and all(counts.values()):
                loops.append(dict(begin=target,back_edge=pc,counts=counts))
        if not loops:
            raise SystemExit('CUTLASS needs HMMA and LDGSTS in the same K back edge')
        # The smallest qualifying back edge is the K pipeline, excluding the
        # surrounding persistent grouped-problem loop.
        loop=min(loops,key=lambda v:v['back_edge']-v['begin'])
        cutlass_records.append(dict(tile=[int(tile[1]),128,32],stages=4 if tile[1]=='128' else 3,
            mainloop=dict(begin=hex(loop['begin']),back_edge=hex(loop['back_edge']),**loop['counts']),
            static_instruction_counts={op:len(v) for op,v in ops.items()}))
        continue
    if 'decode_window' in name:
        vector_loads=len(re.findall(r'\bLDG?\.E\.128\b',section))
        half_stores=len(re.findall(r'\bSTG\.E\.U16\b',section))
        warp_sync=len(re.findall(r'\bWARPSYNC\b',section))
        if vector_loads<4 or half_stores<32 or not warp_sync:
            raise SystemExit('Decoder needs K3..K6 vector loads, FP16 stores and warp synchronization')
        if re.search(r'\b(?:LDL|STL)\b',section):
            raise SystemExit('Decoder spilled to local memory')
        decoder_records.append(dict(vector_load_16B_count=vector_loads,
            direct_FP16_store_count=half_stores,warp_sync_count=warp_sync,local_memory=False))
    m=re.search(r'grouped_mmaILb([01])ELb([01])E',name)
    if not m:
        continue
    gu,ref=map(int,m.groups())
    hmma=[int(x,16) for x in re.findall(r'/\*([0-9a-f]+)\*/\s+HMMA\.16816\.F32',section)]
    ffma=len(re.findall(r'\bFFMA\b',section))
    loops=[]
    for pc,target in re.findall(r'/\*([0-9a-f]+)\*/[^\n]*\bBRA\s+0x([0-9a-f]+)',section):
        pc,target=int(pc,16),int(target,16)
        if target<pc and any(target<=h<=pc for h in hmma):
            loops.append(dict(begin=hex(target),back_edge=hex(pc)))
    if ref:
        if hmma or not ffma:
            raise SystemExit('SIMT reference needs scalar FFMA and no HMMA')
    elif len(hmma)<(4 if gu else 2) or not loops:
        raise SystemExit('Native grouped MMA has no confirmed HMMA K loop')
    records.append(dict(gate_up=bool(gu),reference=bool(ref),instruction='HMMA.16816.F32',
                        addresses=[hex(h) for h in hmma],loops=loops,scalar_ffma_count=ffma))
if len(records)!=4:
    raise SystemExit('Expected four native/reference gate-up/down specializations')
if len(cutlass_records)!=2 or {r['tile'][0] for r in cutlass_records}!={32,128}:
    raise SystemExit('Expected hot and cold CUTLASS kernels')
if len(decoder_records)!=1:
    raise SystemExit('Expected one runtime-dispatched decode window kernel')
result=dict(legacy_and_reference=records,cutlass=cutlass_records,decoder=decoder_records)
a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
