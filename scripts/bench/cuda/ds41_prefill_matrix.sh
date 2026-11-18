#!/bin/bash
# DS41 CUDA prefill matrix: starts strix-cuda-ds41-prefill-service once, POSTs each LENGTH REPS times (first = cold,
# last = warm, never mixed), samples the GPU (memory, utilisation, SM clock) every second, and writes summary.txt.
#
#   ds41_prefill_matrix.sh OUT_DIR -- SERVICE_EXE MODEL.gguf IDS_FILE [service flags...]
# Environment: LENGTHS ("2048 8192 32768"), REPS (3), PORT (8097), API_KEY_FILE (optional), GPU_INDEX (0: the
# nvidia-smi index to sample; set CUDA_VISIBLE_DEVICES/CUDA_DEVICE_ORDER yourself), READY_S (400).
# IDS_FILE holds token ids (comma/newline separated); each length uses its first N ids (tiled when shorter). A
# directory of ids-M.txt files picks, per length, the smallest M >= N.
# With --timing among the service flags the summary adds the per-request accounting (run / dma / host parts).
set -euo pipefail
OUT=$1; shift; [ "${1:-}" = "--" ] && shift
SVC=$1 MODEL=$2 IDS=$3; shift 3
LENGTHS=${LENGTHS:-"2048 8192 32768"} REPS=${REPS:-3} PORT=${PORT:-8097} GPU_INDEX=${GPU_INDEX:-0} READY_S=${READY_S:-400}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
mkdir -p "$OUT"
for f in requests.jsonl marks.txt smi.samples client.err; do : > "$OUT/$f"; done
SP= MP=
cleanup() {
  if [ -n "$MP" ]; then kill "$MP" 2>/dev/null || true; wait "$MP" 2>/dev/null || true; fi
  if [ -n "$SP" ]; then
    if command -v taskkill > /dev/null 2>&1 && [ -r "/proc/$SP/winpid" ]; then
      taskkill //F //T //PID "$(cat "/proc/$SP/winpid")" > /dev/null 2>&1 || kill "$SP" 2>/dev/null || true
    else kill "$SP" 2>/dev/null || true; fi
    wait "$SP" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
KEY=(); [ -n "${API_KEY_FILE:-}" ] && KEY=(--api-key "$(cat "$API_KEY_FILE")")
smi() { nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,utilization.gpu,clocks.sm --format=csv,noheader,nounits | tr -d ' ' || true; }
echo "start $(date +%T) gpu$GPU_INDEX $(smi) uname $(uname -sr)" > "$OUT/summary.txt"
"$SVC" "$MODEL" --host 127.0.0.1 --port "$PORT" "${KEY[@]}" "$@" > "$OUT/svc.out" 2> "$OUT/svc.err" &
SP=$!
( while kill -0 $SP 2>/dev/null; do echo "$(date +%s.%N | cut -c1-14) $(smi)" >> "$OUT/smi.samples"; sleep 1; done ) & MP=$!
ready=0
for ((attempt=0; attempt<READY_S; attempt+=2)); do
  kill -0 "$SP" 2>/dev/null || break
  if grep -q '"event":"READY"' "$OUT/svc.out"; then ready=1; break; fi
  sleep 2
done
if [ "$ready" != 1 ]; then echo "service did not become READY" >&2; exit 1; fi
echo "ready $(date +%T): $(grep -c '"event":"READY"' "$OUT/svc.out")" >> "$OUT/summary.txt"
PY=$(command -v python3 || command -v python)
for n in $LENGTHS; do
  f=$IDS
  if [ -d "$IDS" ]; then  # a directory of ids-M.txt files: the smallest M >= n (else the largest)
    f=$(ls "$IDS"/ids-*.txt | sed 's/.*ids-\([0-9]*\)\.txt/\1 &/' | sort -n | awk -v n="$n" '!found && $1>=n{f=$2; found=1} {l=$2} END{print found ? f : l}')
  fi
  echo "len $n begin $(date +%s.%N | cut -c1-14) $(basename "$f")" >> "$OUT/marks.txt"
  "$PY" -I "$ROOT/tools/ds41_prefill_service_client.py" --service "http://127.0.0.1:$PORT" --prompt-ids "$f" --tokens "$n" \
    --repeat "$REPS" "${KEY[@]}" --json "$OUT/requests.jsonl" > /dev/null 2>> "$OUT/client.err"
  echo "len $n end $(date +%s.%N | cut -c1-14)" >> "$OUT/marks.txt"
done
grep -E 'rror|refus' "$OUT/svc.err" | head -4 | cut -c1-300 >> "$OUT/summary.txt" || true
"$PY" -I "$HERE/ds41_prefill_summary.py" "$OUT" --lengths $LENGTHS --reps "$REPS" >> "$OUT/summary.txt" 2>&1
