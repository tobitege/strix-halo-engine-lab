#!/bin/bash
# Paired kernel/API traces are diagnostics, not throughput-comparison receipts.
set -u
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
K=${TURBODEC_ROOT:?study runner directory required};TAG=${1:-tdtrace};V=${2:-v2};BEGIN=$(date +%s)
source "$SCRIPT_DIR/decode-study-ticket.sh"
study_claim "$TAG" || exit $?
timeout 5 /opt/rocm/bin/rocprofv3 --list-avail > "$K/counters-avail.txt" 2>&1 || true
for arm in E T;do
 remain=$((115-($(date +%s)-BEGIN)));[ "$remain" -lt 30 ] && { printf 'insufficient time for %s\n' "$arm" > "$K/$TAG.partial";exit 1; }
 left=$((remain-10));if [ "$arm" = E ] && [ "$left" -gt 45 ];then left=45;fi
 c8=0;[ "$arm" = T ] && c8=1
 args='';[ "$arm" = T ] && args='--profile turbo --turbo-weights onfly'
 out=$K/trace-$TAG-$arm
  TURBODEC_C8_IF_BUDGET=$c8 TURBODEC_TICKET_BEGIN=$BEGIN TURBODEC_PROMPT_MODULE=${TURBODEC_CODE_PROMPT_MODULE:-$SCRIPT_DIR/decode-study-inventory.py} LEFT=$left SLOTS=2 MAXTOK=64 XARGS="$args" WRAP="/opt/rocm/bin/rocprofv3 --kernel-trace --hip-trace --output-format csv -d $out -o k --" bash "${TURBODEC_TICKET_RUNNER:-$SCRIPT_DIR/decode-study-ticket.sh}" "$TAG-$arm" "$V" C1,C1 STRIX_PLE_REWARM=0 STRIX_HTTP_SLOT_TIMING=1 STRIX_CYCLE_TIMING_SLOW=0 || exit $?
 python3 -I "$SCRIPT_DIR/decode-study-trace.py" "$K/runs/$TAG-$arm" "$out" > "$K/$TAG-$arm.sum" 2>&1
 if [ "$?" != 0 ];then printf '%s invalid/unscoped trace\n' "$arm" > "$K/$TAG.invalid";exit 1;fi
 if [ -s "$K/runs/$TAG-$arm/c8-res.jsonl" ];then python3 -I "$SCRIPT_DIR/decode-study-trace.py" "$K/runs/$TAG-$arm" "$out" --scope c8 > "$K/$TAG-$arm-c8.sum" 2>&1 || true;fi
 done
printf 'traces E/T valid elapsed %ss\n' "$(( $(date +%s)-BEGIN ))" > "$K/$TAG.done"
