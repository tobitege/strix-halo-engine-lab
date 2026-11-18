"""Offline DSA/MTP sm_120 evidence; never loads CUDA or runs a device."""
import argparse
import sys
import json
from pathlib import Path
import re
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('sass',type=Path);ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args();records=[]
for section in re.split(r'Function\s*:\s*',(sys.stdin.read() if str(a.sass) == '-' else a.sass.read_text()))[1:]:
    name=section.splitlines()[0];label=None
    if 'GemmGrouped' in name:label='cutlass_projection_or_moe'
    elif 'sparse_mma' in name:label='sparse_scores_and_PV'
    elif 'index_scoresILi0' in name:label='index_MMA'
    elif 'index_fp32_tiled' in name:label='index_tiled_reference'
    elif 'projection_fp32_tiled' in name:label='projection_tiled_reference'
    elif 'tiled_index' in name:label='index_BF16x3_tiled'
    elif 'streaming_index' in name:label='index_streaming_select'
    elif 'paired_projection' in name:label='paired_half_model'
    elif 'index_scoresILi1' in name:label='index_reference'
    elif 'absorbILb0ELb0' in name:label='q_absorb_MMA'
    elif 'absorbILb1ELb0' in name:label='v_absorb_MMA'
    elif 'absorbILb0ELb1' in name:label='q_absorb_reference'
    elif 'absorbILb1ELb1' in name:label='v_absorb_reference'
    elif 'projection_reference' in name:label='projection_reference'
    elif 'sparse_reference_scores' in name:label='sparse_scores_reference'
    elif 'sparse_reference_pv' in name:label='sparse_PV_reference'
    elif 'router_logits' in name:label='MTP_router_FP32'
    else:continue
    ops={op:[int(x,16) for x in re.findall(r'/\*([0-9a-f]+)\*/[^\n]*\b'+op+r'\b',section)] for op in ('HMMA','LDGSTS','FFMA','DADD','DMUL','LDL','STL')}
    loops=[]
    for pc,target in re.findall(r'/\*([0-9a-f]+)\*/[^\n]*\bBRA(?:\.U)?(?:\s+!?UP\d+,)?\s+0x([0-9a-f]+)',section):
        pc,target=int(pc,16),int(target,16)
        if target<pc and any(target<=x<=pc for x in ops['HMMA']):loops.append(dict(begin=hex(target),back_edge=hex(pc),hmma=sum(target<=x<=pc for x in ops['HMMA']),ldgsts=sum(target<=x<=pc for x in ops['LDGSTS'])))
    scalar=label.endswith('reference') or label=='MTP_router_FP32'
    if label=='paired_half_model':
        if ops['HMMA'] or not ops['DADD'] or not ops['DMUL']:raise SystemExit('missing IEEE double paired model')
    elif label=='index_streaming_select':
        if ops['HMMA']:raise SystemExit('staged top-k must not rescore')
    elif scalar:
        if ops['HMMA'] or not ops['FFMA']:raise SystemExit('scalar reference/router instruction failure: '+label)
    else:
        if not ops['HMMA']:raise SystemExit('missing MMA: '+label)
        if not loops and label!='index_BF16x3_tiled':raise SystemExit('missing MMA backedge: '+label)
        if ops['LDL'] or ops['STL']:raise SystemExit('MMA kernel local spill: '+label)
    if label in ('cutlass_projection_or_moe','sparse_scores_and_PV') and not any(loop['ldgsts'] for loop in loops):raise SystemExit('missing MMA/cp.async loop: '+label)
    if label=='cutlass_projection_or_moe' and 'OpMultiplyAddFastF32' in name and not re.search(r'HMMA[^;]*TF32',section):raise SystemExit('missing TF32 FastF32 projection')
    bf16_count=len(re.findall(r'HMMA[^;]*BF16',section))
    if label in ('sparse_scores_and_PV','index_MMA','index_BF16x3_tiled') and bf16_count<4:raise SystemExit('missing BF16 score/PV or index products')
    if label=='index_BF16x3_tiled' and bf16_count!=96:raise SystemExit('expected fully unrolled 2 pool blocks x8 K16 x3 BF16 products x2 HMMA sites')
    if label=='sparse_scores_and_PV':
        # QK can be fully unrolled. A CTA barrier separates its HMMA sites
        # from PV; both regions must be inside the gathered-key backedge.
        bars=[int(x,16) for x in re.findall(r'/\*([0-9a-f]+)\*/[^\n]*\bBAR\.',section)]
        if not any(min(ops['HMMA'])<pc<max(ops['HMMA']) for pc in bars):raise SystemExit('missing distinct QK and PV HMMA regions')
    if label in ('index_tiled_reference','projection_tiled_reference'):
        if ops['LDL'] or ops['STL'] or not all(re.search(r'\b'+op+r'\b',section) for op in ('LDS','STS','BAR')):raise SystemExit('missing shared-memory reuse or local spill: '+label)
    records.append(dict(tf32_hmma=len(re.findall(r'HMMA[^;]*TF32',section)),site=label,symbol=name,counts={k:len(v) for k,v in ops.items()},bf16_hmma=len(re.findall(r'HMMA[^;]*BF16',section)),loops=loops))
required={'index_tiled_reference','projection_tiled_reference','cutlass_projection_or_moe','sparse_scores_and_PV','index_MMA','index_BF16x3_tiled','index_streaming_select','paired_half_model','index_reference','q_absorb_MMA','v_absorb_MMA','q_absorb_reference','v_absorb_reference','projection_reference','sparse_scores_reference','sparse_PV_reference','MTP_router_FP32'}
if {r['site'] for r in records}!=required:raise SystemExit('missing DSA/MTP instruction sites')
if sum(r['tf32_hmma']>0 for r in records if r['site']=='cutlass_projection_or_moe')<2:raise SystemExit('missing row/column FastF32 kernels')
a.out.write_text(json.dumps(dict(architecture='sm_120',sites=records),indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ('site','counts','bf16_hmma','loops')} for r in records]))
