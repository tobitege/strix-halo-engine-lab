#!/usr/bin/env bash
# Recurrent-state storage ABAB at kernel level (GDN deferred-window fixture), single GPU, inside one GPU ticket.
#   run.sh BIN OUT [arms...]   BIN = strix-hip-gdn-deferred-window-tests, OUT = new directory, arms default f32 f16 bf16
# Rounds A B C x3, cold (MALL flushed), each arm under a rocprofv3 kernel trace; split.py splits the trace per case,
# ab.py prints one table of state-kernel us per arm (median of rounds, each round listed). ROCPROF overrides the tracer.
set -eu
bin=$1 out=$2; shift 2
if [ $# -eq 0 ]; then arms=(f32 f16 bf16); else arms=("$@"); fi
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
prof=${ROCPROF:-rocprofv3}
mkdir "$out"
for round in 1 2 3; do for arm in "${arms[@]}"; do
    "$prof" --kernel-trace --output-format csv -d "$out/tr-$arm-$round" -o run -- "$bin" bench-state16 "$arm" cold > "$out/ab-$arm-$round.log" 2>&1
    python3 "$here/split.py" "$(find "$out/tr-$arm-$round" -name '*kernel_trace.csv' | head -1)" "$out/ab-$arm-$round.log" > "$out/sp-$arm-$round.txt"
done; done
python3 "$here/ab.py" "$out"
