#!/bin/bash
# Strict ABAB server arms for the Qwen CUDA server: one server process per arm (load, client, kill this arm's server only),
# GPU samples every second (index, MiB, util, SM MHz) beside every arm. Ends with OUT/summary.txt.
#   SERVE_EXE   server executable (strix-cuda-qwen-serve)
#   SERVE_ARGS  model / tables / sidecar / port / slot flags (the deploy declaration's values)
#   ARMS        "A|ENV=v ENV2=v ;; B|ENV=w ## --profile fast"   (name|environment of the arm [## extra server flags of the arm])
#   ORDER       arm names in run order, e.g. "A B A B A B"
#   CLIENT      classes client command with exactly one local HTTP URL; the arm's output path is appended
#   OUT         output directory;  GPU_INDEX (default 0): the nvidia-smi index the server is pinned to
set -u
: "${SERVE_EXE:?}" "${SERVE_ARGS:?}" "${ARMS:?}" "${ORDER:?}" "${CLIENT:?}" "${OUT:?}"
GPU_INDEX=${GPU_INDEX:-0}
mkdir -p "$OUT"
smi() { nvidia-smi --id="$GPU_INDEX" --query-gpu=index,memory.used,utilization.gpu,clocks.sm --format=csv,noheader; }
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE=
for arg in $CLIENT; do
  case "$arg" in http://*) [ -z "$BASE" ] || exit 1; BASE=$arg ;; esac
done
[ -n "$BASE" ] || { echo 'CLIENT must name one local HTTP endpoint' >&2; exit 1; }
SPID= MPID= CPID=
stop_server() {
  [ -n "$SPID" ] || return 0
  local wpid
  wpid=$(cat /proc/"$SPID"/winpid 2>/dev/null || true)
  if [ -z "$wpid" ] || ! taskkill //F //PID "$wpid" >/dev/null 2>&1; then kill "$SPID" 2>/dev/null; fi
  wait "$SPID" 2>/dev/null || true
  SPID=
}
cleanup() {
  if [ -n "$CPID" ]; then kill "$CPID" 2>/dev/null; wait "$CPID" 2>/dev/null; fi
  if [ -n "$MPID" ]; then kill "$MPID" 2>/dev/null; wait "$MPID" 2>/dev/null; fi
  stop_server
}
trap cleanup EXIT
trap 'exit 130' INT TERM
declare -A ENVS FLAGS
IFS=';' read -ra LIST <<< "${ARMS//;;/;}"
for spec in "${LIST[@]}"; do
  [ -n "${spec// /}" ] || continue
  n=$(echo "$spec" | cut -d'|' -f1 | tr -d ' '); rest=$(echo "$spec" | cut -d'|' -f2-)
  ENVS[$n]=${rest%%##*}
  FLAGS[$n]=""; [[ "$rest" == *"##"* ]] && FLAGS[$n]=${rest#*##}
done
echo "start $(date +%T) smi $(smi)" > "$OUT/summary.txt"
k=0; failed=0
for name in $ORDER; do
  k=$((k + 1)); A="$OUT/$k-$name"; mkdir -p "$A"
  PORT=$(python "$HERE/qwen_serve_classes.py" "$BASE" --probe-endpoint) || exit 1
  { printf 'executable %s\nargs %s\nenv %s\nGPU_INDEX %s\n' "$SERVE_EXE" "$SERVE_ARGS" "${ENVS[$name]}" "$GPU_INDEX"
    sha256sum "$SERVE_EXE" || exit 1
  } > "$A/launch.txt" || exit 1
  # shellcheck disable=SC2086
  env CUDA_VISIBLE_DEVICES=$GPU_INDEX ${ENVS[$name]} "$SERVE_EXE" $SERVE_ARGS ${FLAGS[$name]} > "$A/serve.log" 2>&1 &
  SPID=$!
  printf 'pid %s\n' "$SPID" >> "$A/launch.txt"
  deadline=$((SECONDS + ${READY_S:-600}))
  until grep -qE "^strix-cuda-qwen-serve: listening on .*:${PORT} model " "$A/serve.log"; do
    if ! kill -0 "$SPID" 2>/dev/null || [ "$SECONDS" -ge "$deadline" ]; then
      echo "$name failed before readiness; see $A/serve.log" >&2; exit 1
    fi
    sleep 0.1
  done
  kill -0 "$SPID" 2>/dev/null || exit 1
  : > "$A/smi.samples"
  ( while kill -0 $SPID 2>/dev/null; do echo "$(date +%T) $(smi)" >> "$A/smi.samples"; sleep 1; done ) & MPID=$!
  timeout "${LIMIT_S:-900}" $CLIENT "$A/bench.txt" > "$A/client.log" 2>&1 & CPID=$!
  dead=0
  while kill -0 "$CPID" 2>/dev/null; do
    if ! kill -0 "$SPID" 2>/dev/null; then dead=1; kill "$CPID" 2>/dev/null; break; fi
    sleep 0.1
  done
  wait "$CPID"; rc=$?; CPID=
  kill -0 "$SPID" 2>/dev/null || dead=1
  [ "$dead" -eq 0 ] || rc=1
  [ "$rc" -eq 0 ] || failed=1
  kill "$MPID" 2>/dev/null; wait "$MPID" 2>/dev/null; MPID=
  stop_server
  { echo "== $k $name rc $rc env [${ENVS[$name]}]"
    grep -cE ' C[148] ' "$A/bench.txt" | sed 's/^/result lines: /'
    # sample fields: time, index, MiB, MiB, util, %, SM MHz, MHz (per GPU)
    awk -v g="$GPU_INDEX," '$2==g {u+=$5; m++} $2==g && $5+0>=50 {s+=$7; n++}
      END{printf "busy-sample mean SM MHz %.0f (n=%d), mean util %.0f %%\n", n?s/n:0, n, m?u/m:0}' "$A/smi.samples"
  } >> "$OUT/summary.txt"
done
echo "end $(date +%T) smi $(smi)" >> "$OUT/summary.txt"
exit "$failed"
