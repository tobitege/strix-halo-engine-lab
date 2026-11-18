"""Static instruction sanity for the full service; never loads CUDA."""
import argparse
import sys
import json
from pathlib import Path
import re

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('sass', type=Path)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
sass = (sys.stdin.read() if str(a.sass) == '-' else a.sass.read_text())
if 'sm_120' not in sass:
    raise SystemExit('missing sm_120 code')
records = {}
for section in re.split(r'Function\s*:\s*', sass)[1:]:
    name = section.splitlines()[0]
    if 'producer' not in name:
        continue
    for site in ('hc_pre_kernel', 'hc_expand_kernel', 'hc_final_kernel', 'router', 'embed', 'norm'):
        if site not in name:
            continue
        counts = {op: len(re.findall(r'\b'+op+r'\b', section)) for op in ('HMMA', 'FFMA', 'FADD', 'FMUL', 'LDL', 'STL')}
        if counts['HMMA']:
            raise SystemExit('HC/nonlinear/router site unexpectedly uses MMA: '+site)
        if site in ('router', 'hc_pre_kernel', 'norm') and not counts['FFMA']:
            raise SystemExit('missing FP32 chain/tree: '+site)
        if site in ('hc_expand_kernel', 'hc_final_kernel') and counts['FFMA']:
            raise SystemExit('HC residual order contracted: '+site)
        records[site] = dict(symbol=name, counts=counts)
if set(records) != {'hc_pre_kernel', 'hc_expand_kernel', 'hc_final_kernel', 'router', 'embed', 'norm'}:
    raise SystemExit('missing full-model runtime kernels')
a.out.write_text(json.dumps(dict(architecture='sm_120', sites=records), indent=2)+'\n')
print('PASS full-model HC/norm/embedding/router SASS')
