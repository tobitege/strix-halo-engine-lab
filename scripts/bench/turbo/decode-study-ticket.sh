#!/bin/bash
# decode-study-ticket.sh TAG BUILD_LABEL ARMS [ENV...]: serve ticket (through gpu-turn.sh): batched HTTP server of build V, cohort arms, clock samples.
# Shared by the study wrappers; a failed invocation keeps its claim for diagnosis.
study_claim() {
  local tag=$1 path
  for path in "$K/$tag.done" "$K/$tag.complete" "$K/runs/$tag" "$K/runs/$tag-"* "$K/trace-$tag" "$K/trace-$tag-"*; do
    if [ -e "$path" ]; then echo "existing study output: $path" >&2; return 2; fi
  done
  mkdir "$K/$tag.claim" 2>/dev/null || { echo "study already claimed: $tag" >&2; return 2; }
}
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then return 0; fi
K=${TURBODEC_ROOT:?Set TURBODEC_ROOT to the output directory}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ADAPTER=${TURBODEC_SERVER_ADAPTER:?Set TURBODEC_SERVER_ADAPTER to the deployment recipe}
BIN=${TURBODEC_BIN:?Set TURBODEC_BIN to the decoder executable}
export TURBODEC_MODEL=${TURBODEC_MODEL:?Set TURBODEC_MODEL to the target model file}
export TURBODEC_PROMPT_MODULE=${TURBODEC_PROMPT_MODULE:-$SCRIPT_DIR/decode-study-classes.py}
cd "$K"
TAG=$1; V=$2; ARMS=$3; shift 3; E="$*"; T0=$(date +%s); LEFT=${LEFT:-115}
OUT=$K/runs/$TAG; mkdir -p "$K/runs"; mkdir "$OUT" 2>/dev/null || { echo duplicate; exit 2; }
python3 -c 'import json,time; print(json.dumps({"epoch_ns":time.time_ns(),"monotonic_ns":time.monotonic_ns(),"boottime_ns":time.clock_gettime_ns(time.CLOCK_BOOTTIME),"monotonic_raw_ns":time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)}))' > "$OUT/clock-anchor.json"
PORT=$((18950 + RANDOM % 40))
echo "== $TAG $(date -Is) $(hostname) bin $(sha256sum "$BIN" | cut -c1-16) env [$E] arms $ARMS" > "$OUT/log.txt"
( while :; do echo "$(date +%s.%N) $(grep -h "\*" /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | head -1) busy=$(head -n1 /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | head -1) mem_kb=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)"; sleep 0.25; done ) > $OUT/clk.txt 2>/dev/null &
CP=$!
SRVENV="STRIX_MTP_DRAFT_LOG=1 STRIX_CYCLE_TIMING=1 STRIX_SERVE_RECEIPT=1 STRIX_MTP_BATCH_LOG=2 STRIX_HTTP_EAGER_SLOTS=${SLOTS:-8} $E" SRVTIME=$((LEFT-($(date +%s)-T0))) bash "$ADAPTER" "$BIN" "$PORT" "$OUT/srv.err" --http-batch "${TURBODEC_HTTP_BATCH:-16}" ${XARGS:-} &
SP=$!
python3 -I "$SCRIPT_DIR/decode-study-startup-sampler.py" "$SP" "$BIN" "$OUT/startup-proc.jsonl" &
PP=$!
READY=0
while [ "$(( $(date +%s)-T0 ))" -lt "$((LEFT-8))" ]; do if grep -q "listening on" "$OUT/srv.err" 2>/dev/null || curl -fsS --max-time .25 "http://127.0.0.1:$PORT/v1/models" > "$OUT/readiness-models.json" 2>/dev/null; then READY=1; break; fi; kill -0 "$SP" 2>/dev/null || break; sleep 1; done
echo "listening after $(( $(date +%s)-T0 )) s: $(grep -c "listening on" $OUT/srv.err)" >> $OUT/log.txt
grep -m2 "http slots\|exceeds explicit budget" $OUT/srv.err | cut -c1-200 >> $OUT/log.txt
CLIENT_BUDGET=$((LEFT - 8 - ($(date +%s)-T0)))
CLIENT_RC=1
if [ "$READY" = 1 ] && [ "$CLIENT_BUDGET" -gt 0 ]; then
  timeout "$CLIENT_BUDGET" python3 "$SCRIPT_DIR/decode-study-client.py" "$PORT" "$OUT/res.jsonl" "$ARMS" "${MAXTOK:-256}" >> "$OUT/log.txt" 2>&1
  CLIENT_RC=$?
  echo "client rc $CLIENT_RC" >> "$OUT/log.txt"
fi
if [ "${TURBODEC_C8_IF_BUDGET:-0}" = 1 ] && [ "$READY" = 1 ];then bash "$SCRIPT_DIR/decode-study-c8-hook.sh" "$PORT" "$OUT" "$((LEFT-($(date +%s)-T0)))";fi
kill -INT $SP 2>/dev/null; for i in $(seq 1 15); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -TERM $SP 2>/dev/null; wait $SP
kill $CP $PP 2>/dev/null
grep -m3 "lever active" $OUT/srv.err | cut -c1-200 >> $OUT/log.txt
echo "$TAG done $(( $(date +%s)-T0 )) s" >> $OUT/log.txt
exit "$CLIENT_RC"
