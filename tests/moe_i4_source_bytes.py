"""Check S12 implementation bytes without requiring HIP or model assets."""
from pathlib import Path
import re
files = [
    'src/core/activation_quant.hpp','src/core/tiled_core.hpp','src/core/load_cohort.hpp',
    'src/core/moe_i4_arithmetic.hpp','src/core/moe_i4_chain.hpp',
    'src/core/moe_i4_chain_plan.hpp',
    'src/moe_core/device_plan.hpp','src/moe_core/device_plan_index.hpp',
    'src/minimax/engine/engine.hip','src/minimax/moe/mmx_prefill.hip',
    'src/minimax/moe/mmx_prefill.hpp','src/minimax/moe/mmx_prefill_iu4.inc',
    'src/glm/engine/engine.hip','src/glm/prefill/glm_pf_exact_moe.hip',
    'src/glm/prefill/glm_pf_exact_moe.hpp','src/glm/prefill/glm_pf_iu4.inc',
    'tests/moe_i4_chain_host_tests.cpp',
]
root = Path(__file__).resolve().parents[1]
for name in files:
    data = (root / name).read_bytes()
    assert b'\0' not in data, f'{name}: raw NUL'
    s = data.decode('utf-8')
    assert not re.search(r'^(<<<<<<<|=======|>>>>>>>)', s, re.M), f'{name}: conflict marker'
    i = 0
    while i < len(s):
        if s.startswith('//', i):
            end = s.find('\n', i); i = len(s) if end < 0 else end + 1
        elif s.startswith('/*', i):
            end = s.find('*/', i+2); assert end >= 0, f'{name}: unterminated comment'; i = end+2
        elif s.startswith('R"', i):
            end = s.index('(', i+2); delimiter=s[i+2:end]
            i=s.index(')'+delimiter+'"',end)+len(delimiter)+2
        elif s[i] in '\"\'':
            quote=s[i]
            if quote=="'" and i and s[i-1].isalnum() and i+1<len(s) and s[i+1].isdigit():
                i+=1;continue
            start=i;i+=1
            while i<len(s) and s[i]!=quote:
                assert s[i] not in '\r\n', f'{name}:{s.count(chr(10),0,start)+1}: raw newline in literal'
                if s[i]=='\\': i+=2
                else: i+=1
            assert i<len(s), f'{name}: unterminated literal'
            i+=1
        else: i+=1
print(f'PASS: {len(files)} C++/HIP files; no raw NUL, literal newline or conflict marker')
