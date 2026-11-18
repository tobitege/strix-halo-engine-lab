#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Run after an earlier study completes; one quiet bounded ticket per diagnostic arm.
set -u
K=${TURBODEC_ROOT:?Set TURBODEC_ROOT to the study runner and output directory}; L=${TURBODEC_LAB:?Set TURBODEC_LAB to the deployment script directory}
WAIT=${1:-tdv3}; STAMP=${2:-tdprobe}
MODEL=${TURBODEC_MODEL:?Set TURBODEC_MODEL to the target model file}
cd "$K" || exit 1
source "$SCRIPT_DIR/decode-study-ticket.sh"
study_claim "$STAMP" || exit $?
while [ ! -f "$WAIT.done" ]; do sleep 15; done
if ! python3 - "$K" "$WAIT" <<'PY'
import json,sys
from pathlib import Path
k=Path(sys.argv[1]);stamp=sys.argv[2]
try:
 for arm in (k/(stamp+".runs")).read_text().splitlines():
  rows=[json.loads(x) for x in (k/"runs"/arm/"res.jsonl").read_text().splitlines()]
  assert len(rows)>=2 and all(r["tokens"]>0 and not r["errors"] for r in rows)
except Exception:sys.exit(1)
PY
then printf 'invalid prior ABAB\n' > "$STAMP.invalid"; exit 1; fi
printf 'start %s\n' "$(date -Is)" > "$STAMP.status"
for arm in E T; do
  args=''; [ "$arm" = T ] && args='--profile turbo --turbo-weights onfly'
  tag=$STAMP-prof-$arm; out=$K/trace-$tag
  cat "$MODEL" >/dev/null
  printf '%s precached %s\n' "$tag" "$(grep MemAvailable /proc/meminfo)" >> "$STAMP.status"
  SLOTS=2 MAXTOK=64 LEFT=110 GPU_TURN_MAX=120 "$L/gpu-turn.sh" "$tag-time" env XARGS="$args" WRAP="/opt/rocm/bin/rocprofv3 --kernel-trace --hip-trace --output-format csv -d $out -o k --" bash "${TURBODEC_TICKET_RUNNER:-$SCRIPT_DIR/decode-study-ticket.sh}" "$tag" v2 C1,C1 > "$tag.ticket.log" 2>&1 || exit $?
  printf '%s rc=%s\n' "$tag" "$?" >> "$STAMP.status"
  python3 -I "$SCRIPT_DIR/decode-study-trace.py" "$K/runs/$tag" "$out" > "$tag.sum" 2>&1
 done
for arm in E T; do
  args=''; [ "$arm" = T ] && args='--profile turbo --turbo-weights onfly'
  tag=$STAMP-class-$arm
  cat "$MODEL" >/dev/null
  printf '%s precached %s\n' "$tag" "$(grep MemAvailable /proc/meminfo)" >> "$STAMP.status"
  SLOTS=2 MAXTOK=128 LEFT=115 TURBODEC_PROMPT_MODULE=${TURBODEC_CLASS_PROMPT_MODULE:-$SCRIPT_DIR/decode-study-classes.py} GPU_TURN_MAX=120 "$L/gpu-turn.sh" "$tag-time" env XARGS="$args" TURBODEC_PROMPT_MODULE=${TURBODEC_CLASS_PROMPT_MODULE:-$SCRIPT_DIR/decode-study-classes.py} bash "${TURBODEC_TICKET_RUNNER:-$SCRIPT_DIR/decode-study-ticket.sh}" "$tag" v2 S4,S4 > "$tag.ticket.log" 2>&1 || exit $?
  printf '%s rc=%s\n' "$tag" "$?" >> "$STAMP.status"
 done
tag=$STAMP-native-N
cat "$MODEL" >/dev/null
printf '%s precached %s\n' "$tag" "$(grep MemAvailable /proc/meminfo)" >> "$STAMP.status"
SLOTS=2 MAXTOK=192 LEFT=115 GPU_TURN_MAX=120 "$L/gpu-turn.sh" "$tag-time" env XARGS='--profile turbo' bash "${TURBODEC_TICKET_RUNNER:-$SCRIPT_DIR/decode-study-ticket.sh}" "$tag" v2 C1,C1 > "$tag.ticket.log" 2>&1 || exit $?
printf '%s rc=%s\n' "$tag" "$?" >> "$STAMP.status"
python3 -I "$SCRIPT_DIR/decode-study-class-summary.py" "$K/runs/$STAMP-class-E" "$K/runs/$STAMP-class-T" > "$STAMP-class.sum" 2>&1 || exit $?
printf 'end %s\n' "$(date -Is)" >> "$STAMP.status"
touch "$STAMP.done"
