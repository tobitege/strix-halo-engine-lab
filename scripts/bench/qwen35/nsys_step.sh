#!/usr/bin/env bash
# Per-kernel step table of one decode configuration under Nsight Systems. Usage:
#   nsys_step.sh NSYS BIN OUTPREFIX "APP_ARGS" [TOP=9]
set -u
NS=$1; BIN=$2; O=$3; ARGS=$4; TOP=${5:-9}
D=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC2086
"$NS" profile -f true --trace=cuda --cuda-graph-trace=node -o "$O" "$BIN" $ARGS > /dev/null 2>&1
"$NS" stats -r cuda_gpu_trace -f csv -o "$O" "$O.nsys-rep" > /dev/null 2>&1
python "$D/nsys_step_table.py" "${O}_cuda_gpu_trace.csv" | head -"$TOP" | cut -c1-170
