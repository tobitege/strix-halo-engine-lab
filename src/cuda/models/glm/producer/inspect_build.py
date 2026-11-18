"""Disassemble compiler artifacts in RAM; write only small SASS evidence JSON.

No CUDA runtime or service executable is loaded. Avoids large .sass files.
"""
import argparse
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build', type=Path, required=True)
p.add_argument('--cuobjdump', type=Path, required=True)
a = p.parse_args()
here = Path(__file__).resolve().parent
for artifact, inspector, output in [
    ('producer_native.obj', here/'../bench/inspect_sass.py', 'producer-moe-sass.json'),
    ('producer_kda.obj', here/'../bench/inspect_kda_sass.py', 'producer-kda-sass.json'),
    ('producer_dsa.obj', here/'../bench/inspect_dsa_sass.py', 'producer-dsa-sass.json'),
    ('producer_runtime.obj', here/'inspect_runtime_sass.py', 'producer-runtime-sass.json'),
    ('glm_native_bench.exe', here/'../bench/inspect_sass.py', 'bench-moe-sass.json'),
    ('glm_kda_bench.exe', here/'../bench/inspect_kda_sass.py', 'bench-kda-sass.json'),
    ('glm_dsa_bench.exe', here/'../bench/inspect_dsa_sass.py', 'bench-dsa-sass.json'),
]:
    binary=a.build/artifact
    if not binary.exists():binary=binary.with_suffix('.lib')
    sass = subprocess.run([str(a.cuobjdump), '--dump-sass', str(binary)],
                          check=True, capture_output=True, text=True).stdout
    subprocess.run([sys.executable, '-B', str(inspector), '-', '--out', str(a.build/output)],
                   input=sass, text=True, check=True)
