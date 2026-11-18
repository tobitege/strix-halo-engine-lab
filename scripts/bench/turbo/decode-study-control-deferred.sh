#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
set -u
K=${TURBODEC_ROOT:?study directory required};L=${TURBODEC_LAB:?deployment scripts directory required};MODEL=${TURBODEC_MODEL:?target model path required}
cd "$K" || exit 1
while [ ! -f tdpack.complete ];do sleep 15;done
[ -f tdpack.done ] || { printf 'qualification invalid; no control ticket\n' > tdctrl.failed;touch tdctrl.complete;exit 1; }
cat "$MODEL" >/dev/null
GPU_TURN_MAX=120 "$L/gpu-turn.sh" tdctrl-time env TURBODEC_ROOT="$K" bash "$SCRIPT_DIR/decode-study-control-ticket.sh" tdctrl v2 > tdctrl.ticket.log 2>&1 || exit $?
python3 -I "$SCRIPT_DIR/decode-study-class-summary.py" "$K/runs/tdctrl-"{N,E,T} > tdctrl.clock.sum || exit $?
python3 -I "$SCRIPT_DIR/decode-study-phase-summary.py" "$K/runs/tdctrl-"{N,E,T} > tdctrl.phase.sum || exit $?
[ -f tdctrl.done ] || printf 'native/classes diagnostic partial or failed\n' > tdctrl.failed
touch tdctrl.complete
