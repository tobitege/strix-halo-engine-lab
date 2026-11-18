#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
set -u
K=${TURBODEC_ROOT:?Set TURBODEC_ROOT to the study runner and output directory}; L=${TURBODEC_LAB:?Set TURBODEC_LAB to the deployment script directory}
MODEL=${TURBODEC_MODEL:?Set TURBODEC_MODEL to the target model file}
cd "$K" || exit 1
source "$SCRIPT_DIR/decode-study-ticket.sh"
study_claim tdc8 || exit $?
while [ ! -f tdprobe.done ]; do sleep 15; done
printf 'start %s\n' "$(date -Is)" > tdc8.status
for arm in E T; do
 args=''; wrap=''; [ "$arm" = T ] && { args='--profile turbo --turbo-weights onfly'; wrap="/opt/rocm/bin/rocprofv3 --kernel-trace --hip-trace --output-format csv -d $K/trace-tdc8-T -o k --"; }
 tag=tdc8-$arm
 cat "$MODEL" >/dev/null
 printf '%s precached %s\n' "$tag" "$(grep MemAvailable /proc/meminfo)" >> tdc8.status
 SLOTS=8 MAXTOK=64 LEFT=110 GPU_TURN_MAX=120 "$L/gpu-turn.sh" "$tag-time" env TURBODEC_LIST_COUNTERS="$arm" XARGS="$args" WRAP="$wrap" bash "$SCRIPT_DIR/decode-study-c8-ticket.sh" "$tag" > "$tag.ticket.log" 2>&1
 rc=$?; printf '%s rc=%s\n' "$tag" "$rc" >> tdc8.status
 [ "$rc" = 0 ] || exit "$rc"
 done
python3 -I "$SCRIPT_DIR/decode-study-class-summary.py" "$K/runs/tdc8-E" "$K/runs/tdc8-T" > tdc8.sum 2>&1 || exit $?
python3 -I "$SCRIPT_DIR/decode-study-trace.py" "$K/runs/tdc8-T" "$K/trace-tdc8-T" > tdc8-trace.sum 2>&1 || exit $?
touch tdc8.done
