#!/usr/bin/env bash
# glm_score.sh PEER BIN MODEL PROMPT.u32 OUT_PREFIX [ENV=VAL ...]: one GLM TP2 scored prefill for the FAST router gate.
# Rank 0 runs here, rank 1 on PEER (ssh host; BIN, MODEL, PROMPT and the OUT_PREFIX directory must exist at the same
# paths there). Every STRIX_* variable of the caller is dropped; the arm is exactly the ENV list. The last ROWS (256)
# prompt rows are scored (STRIX_GLM_PF_SCORE_LAST), the logits land in OUT_PREFIX-r<k>.f32 (rank 1's copied back), the
# logs in OUT_PREFIX-r<k>.log, MemAvailable and the GPU clock around the run in OUT_PREFIX.meta. Run inside a pair GPU
# turn. Compare two arms with score_cmp.py REF_PREFIX CAND_PREFIX PROMPT.u32.
# Arms of the gate: E (exact) -, F STRIX_GLM_PF_WMMA=1, W STRIX_GLM_PF_WMMA=1 STRIX_GLM_PF_ROUTER_WMMA=1.
set -u
PEER=${1:?PEER}; BIN=${2:?BIN}; MODEL=${3:?MODEL}; PROMPT=${4:?PROMPT}; OUT=${5:?OUT_PREFIX}; shift 5
ROWS=${ROWS:-256}; CAPA=${CAPA:-8448}; ARMT=${ARMT:-110}
SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=5)
for v in $(env | grep -oE "^STRIX_[A-Z0-9_]*"); do unset "$v"; done
BASE=(STRIX_DIST_USB4_TIMEOUT_MS=120000 STRIX_DIST_WAIT_TIMEOUT_MS=120000 HSA_DISABLE_FRAGMENT_ALLOCATOR=1
      STRIX_GLM_PF_SCORE_LAST=$ROWS STRIX_GLM_PF_SCORE_OUT=$OUT)
ARGS=(--world 2 --transport usb4-auto --mtp 0 --slots 1 --capacity "$CAPA" --prompt "$PROMPT" --decode 1)
meta() { echo "$1 $(date +%T) MemAvailable $(awk '/MemAvailable/ {printf "%.1f GiB", $2 / 1048576}' /proc/meminfo) sclk $(grep -h -m1 '\*' /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | awk '{print $2}' | head -1) evicted $(cat /sys/class/kfd/kfd/proc/*/stats_*/evicted_ms 2>/dev/null | awk '{s += $1} END {print s + 0}') ms" >> "$OUT.meta"; }
: > "$OUT.meta"; meta start
H0=$(sha256sum "$BIN" | cut -c1-16); H1=$("${SSH[@]}" "$PEER" "sha256sum $BIN" | cut -c1-16)
[ "$H0" = "$H1" ] || { echo "glm_score: binaries differ ($H0 / $H1)" >&2; exit 3; }
"${SSH[@]}" "$PEER" "env ${BASE[*]} $* timeout -k 5 $ARMT $BIN $MODEL --rank 1 ${ARGS[*]}" > "$OUT-r1.log" 2>&1 & P1=$!
env "${BASE[@]}" "$@" timeout -k 5 "$ARMT" "$BIN" "$MODEL" --rank 0 "${ARGS[@]}" > "$OUT-r0.log" 2>&1; rc0=$?
wait $P1; rc1=$?
meta end
"${SSH[@]}" "$PEER" "cat $OUT-r1.f32" > "$OUT-r1.f32" 2>/dev/null
echo "glm_score $OUT rc $rc0/$rc1 $(grep -ah -m1 'prefill gemv:' "$OUT-r0.log" | cut -c1-120)"
grep -ah "prefill identity" "$OUT-r0.log" | cut -c1-160 | head -4
[ $rc0 = 0 ] && [ $rc1 = 0 ]
