#!/usr/bin/env bash
# Strict ABAB prefill arms of the CUDA Qwen server on one GPU: each arm = a fresh server process, the prefill client
# (cold + warm per length), GPU clock / utilisation samples, then the server is stopped. Outputs only under $OUT.
# Run it inside the GPU-ownership wrapper of your lab (this script takes no lock and posts nothing).
#
# Required env: SERVE (strix-cuda-qwen-serve binary), MODEL (merged GGUF), TABLES, ROPE, IDS (token ids of a long text),
#               OUT (output dir). Optional: MTP, VOCAB, PORT (8095), GPU_INDEX (0), LIMIT_S (900 per arm),
#               CLIENT_ARGS ("--lens 8192,32768 --runs 2"), COMMON ("--max-positions 34816 --http-slots 1").
# ARMS="name|ENV=v ENV2=v|--flags ;; name2||--profile fast ;; ..."  (order = run order, e.g. exactA fastA exactB fastB ...)
# Summary: $OUT/summary.txt (per arm: pf lines, busy-sample mean SM MHz, GPU busy fraction, peak MiB); $OUT/.done at exit.
set -u
: "${SERVE:?}" "${MODEL:?}" "${TABLES:?}" "${ROPE:?}" "${IDS:?}" "${OUT:?}" "${ARMS:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-8095}; GPU=${GPU_INDEX:-0}; PY=${PYTHON:-python}
mkdir -p "$OUT"; trap 'touch "$OUT/.done"' EXIT
smi() { nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu,clocks.sm --format=csv,noheader,nounits | tr -d ' \r'; }
# The client owns the server's native process handle, preflights the port, checks liveness
# during readiness, and kills only that child on success, failure or the arm deadline.
EXTRA=()
[ -n "${MTP:-}" ] && EXTRA+=(--mtp-sidecar "$MTP")
[ -n "${VOCAB:-}" ] && EXTRA+=(--draft-vocab "$VOCAB")
echo "start $(date +%T) gpu $GPU $(smi)" > "$OUT/summary.txt"
IFS=';;' read -ra LIST <<< "${ARMS// ;; /;;}"
for spec in "${LIST[@]}"; do
  [ -n "$spec" ] || continue
  name=$(echo "$spec" | cut -d'|' -f1 | tr -d ' '); envs=$(echo "$spec" | cut -d'|' -f2); flags=$(echo "$spec" | cut -d'|' -f3)
  A="$OUT/$name"; mkdir -p "$A"
  # shellcheck disable=SC2086
  env $envs "$PY" -I "$HERE/prefill_client.py" "http://127.0.0.1:$PORT" "$IDS" "$A/client.txt" \
    ${CLIENT_ARGS:---lens 8192,32768 --runs 2} --limit-s "${LIMIT_S:-900}" --server-log "$A/serve.log" \
    --launch "$SERVE" --model "$MODEL" --tables "$TABLES" --rope-table "$ROPE" "${EXTRA[@]}" --port "$PORT" \
    ${COMMON:---max-positions 34816 --http-slots 1} $flags --prefix-cache off > "$A/client.log" 2>&1 &
  SPID=$!
  ( while kill -0 $SPID 2>/dev/null; do echo "$(date +%s.%N | cut -c1-14) $(smi)" >> "$A/smi.samples"; sleep 0.5; done ) & MPID=$!
  wait "$SPID"; rc=$?
  kill "$MPID" 2>/dev/null
  { echo "== arm $name rc $rc env [$envs] flags [$flags]"
    grep -E 'profile|identity|VRAM after load|rror' "$A/serve.log" | head -4 | cut -c1-240
    grep -E '^pf ' "$A/client.txt" | cut -c1-200
    awk -F, '$2+0>=50{s+=$3; n++} {t++} $1+0>m{m=$1+0} END{printf "busy-sample mean SM MHz %.0f, busy fraction %.2f (n=%d), peak MiB %d\n", n?s/n:0, t?n/t:0, t, m}' \
      <(cut -d' ' -f2 "$A/smi.samples")
  } >> "$OUT/summary.txt"
  [ "$rc" -eq 0 ] || exit "$rc"
done
echo "end $(date +%T) $(smi)" >> "$OUT/summary.txt"
