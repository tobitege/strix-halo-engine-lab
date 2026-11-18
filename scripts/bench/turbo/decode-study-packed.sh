#!/bin/bash
# CPU pre-cache occurs before each ticket; each ticket contains one E/T pair.
set -u
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
K=${TURBODEC_ROOT:?Set TURBODEC_ROOT to the study runner and output directory}
L=${TURBODEC_LAB:?Set TURBODEC_LAB to the deployment script directory}
MODEL=${TURBODEC_MODEL:?Set TURBODEC_MODEL to the target model file}
STAMP=${1:-tdpack}; COUNT=${2:-3}; V=${3:-v2}; START=${4:-1}
cd "$K" || exit 1
[ ! -e "$STAMP.done" ] && [ ! -e "$STAMP.complete" ] || { echo 'existing stamp; choose a fresh one' >&2; exit 2; }
[ "$START" = 1 ] && : > "$STAMP.status"
for p in $(seq "$START" "$COUNT"); do
  tag=$STAMP-p$p
  cat "$MODEL" >/dev/null
  printf '%s precached %s\n' "$tag" "$(grep MemAvailable /proc/meminfo)" >> "$STAMP.status"
  SLOTS=2 MAXTOK=192 GPU_TURN_MAX=120 "$L/gpu-turn.sh" "$tag-time" env TURBODEC_ROOT="$K" TURBODEC_PAIR_BUDGET=115 bash "$SCRIPT_DIR/decode-study-pair-ticket.sh" "$tag" "$V" C1,C1 > "$tag.ticket.log" 2>&1
  rc=$?; printf '%s ticket_rc=%s\n' "$tag" "$rc" >> "$STAMP.status"
  if [ "$rc" != 0 ] || [ ! -f "$tag.done" ]; then printf '%s invalid; stopped\n' "$tag" > "$STAMP.failed"; touch "$STAMP.complete"; exit 1; fi
 done
runs=(); for p in $(seq 1 "$COUNT"); do for arm in E T; do runs+=("$K/runs/$STAMP-p$p-$arm"); done; done
python3 -I "$SCRIPT_DIR/decode-study-class-summary.py" "${runs[@]}" > "$STAMP.clock.sum" || exit $?
python3 -I "$SCRIPT_DIR/decode-study-phase-summary.py" "${runs[@]}" > "$STAMP.phase.sum" || exit $?
if ! python3 - "${runs[@]}" <<'PY'
import json, sys
from pathlib import Path
for run in sys.argv[1:]:
    requests = json.loads((Path(run) / 'cycle-receipt-summary.json').read_text())
    if len(requests) != 2:
        raise SystemExit(f'{run}: two scoped requests required')
    for request in requests:
        if ('aggregate' in request['scope'] or request['cycles'] <= 0
                or request['cycle_emitted'] <= 0
                or request['prefill_or_noncycle_emitted'] != 1
                or request['json_accepted'] + request['cycles'] != request['cycle_emitted']):
            raise SystemExit(f'{run}: invalid emitted-token/request accounting')
PY
then
  printf 'invalid scoped receipts; see phase summary\n' > "$STAMP.failed"
  touch "$STAMP.complete"
  exit 1
fi
touch "$STAMP.done" "$STAMP.complete"
