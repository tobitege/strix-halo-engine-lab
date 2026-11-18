#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Deferred diagnostics run only after all qualification pairs succeeded.
set -u
K=${TURBODEC_ROOT:?study directory required};L=${TURBODEC_LAB:?deployment scripts directory required};MODEL=${TURBODEC_MODEL:?target model path required}
cd "$K" || exit 1
while [ ! -f tdpack.complete ]; do sleep 15; done
[ -f tdpack.done ] || { printf 'qualification invalid; no diagnostics tickets issued\n' > tdextra.failed;touch tdextra.complete;exit 1; }
cat "$MODEL" >/dev/null
GPU_TURN_MAX=120 "$L/gpu-turn.sh" tdctrl-time env TURBODEC_ROOT="$K" bash "$SCRIPT_DIR/decode-study-control-ticket.sh" tdctrl v2 > tdctrl.ticket.log 2>&1 || exit $?
python3 -I "$SCRIPT_DIR/decode-study-class-summary.py" "$K/runs/tdctrl-"{N,E,T} > tdctrl.clock.sum || exit $?
python3 -I "$SCRIPT_DIR/decode-study-phase-summary.py" "$K/runs/tdctrl-"{N,E,T} > tdctrl.phase.sum || exit $?
if [ ! -f tdctrl.done ]; then printf 'native/classes diagnostic partial or failed; stopped\n' > tdextra.failed;touch tdextra.complete;exit 1;fi
cat "$MODEL" >/dev/null
GPU_TURN_MAX=120 "$L/gpu-turn.sh" tdtrace-time env TURBODEC_ROOT="$K" bash "$SCRIPT_DIR/decode-study-trace-ticket.sh" tdtrace v2 > tdtrace.ticket.log 2>&1 || exit $?
[ -f tdtrace.done ] || printf 'paired trace failed/unscoped\n' > tdextra.failed
touch tdextra.complete
