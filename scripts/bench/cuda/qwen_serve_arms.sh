#!/bin/bash
# Sequential strix-cuda-qwen-serve arms on one CUDA card, strict order as given (e.g. A B A B), one bench per arm.
#   ARMS="name|ENV=v ENV2=v|--flags|BUILD_DIR ;; name2|...|...|..."   (BUILD_DIR: directory holding strix-cuda-qwen-serve(.exe))
#   SERVE_ARGS="--model M.gguf --tables DIR --rope-table R.bin [--mtp-sidecar S.gguf --draft-vocab V.bin]"
#   BENCH_ARGS="--prompt-ids IDS.txt --lens 8192 --arms 1,4,8,16 --max-tokens 128"
#   OUT=dir [PORT=8095] [LIMIT_S=900] [GPU=0] bash qwen_serve_arms.sh
# Per arm: OUT/<name>/{serve.log,bench.txt,bench.txt.ids,smi.samples}; OUT/summary.txt (bench lines, mean busy SM clock,
# peak card memory, memory seen on the other cards) and OUT/.done at the end. Only the card GPU is used
# (CUDA_VISIBLE_DEVICES); the script stops only the server it started.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
: "${ARMS:?}" "${SERVE_ARGS:?}" "${BENCH_ARGS:?}" "${OUT:?}"
PORT=${PORT:-8095}; GPU=${GPU:-0}; mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPU
trap 'touch "$OUT/.done"' EXIT
smi() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu,clocks.sm --format=csv,noheader,nounits | tr '\n' ' '; }
echo "start $(date +%T) smi $(smi)" > "$OUT/summary.txt"
IFS=';;' read -ra LIST <<< "${ARMS// ;; /;;}"
for spec in "${LIST[@]}"; do
  [ -n "$spec" ] || continue
  name=$(echo "$spec" | cut -d'|' -f1 | tr -d ' '); envs=$(echo "$spec" | cut -d'|' -f2); flags=$(echo "$spec" | cut -d'|' -f3)
  bd=$(echo "$spec" | cut -d'|' -f4 | tr -d ' '); A="$OUT/$name"; mkdir -p "$A"
  exe="$bd/strix-cuda-qwen-serve"; [ -x "$exe.exe" ] && exe="$exe.exe"
  # shellcheck disable=SC2086
  env $envs "$exe" $SERVE_ARGS --port "$PORT" $flags > "$A/serve.log" 2>&1 &
  SPID=$!
  ( while kill -0 $SPID 2>/dev/null; do echo "$(date +%T) $(smi)" >> "$A/smi.samples"; sleep 1; done ) & MPID=$!
  # shellcheck disable=SC2086
  timeout "${LIMIT_S:-900}" python -I "$HERE/qwen_serve_bench.py" "http://127.0.0.1:$PORT" "$A/bench.txt" $BENCH_ARGS > "$A/client.log" 2>&1; rc=$?
  kill $SPID 2>/dev/null; sleep 1
  if kill -0 $SPID 2>/dev/null && command -v taskkill > /dev/null; then
    wpid=$(cat /proc/$SPID/winpid 2>/dev/null); [ -n "$wpid" ] && taskkill //F //PID "$wpid" > /dev/null 2>&1
  fi
  wait $SPID 2>/dev/null; sleep 3; kill $MPID 2>/dev/null
  { echo "== arm $name rc $rc env [$envs] flags [$flags]"
    grep -E 'VRAM after load|cohort decode|rror' "$A/serve.log" | head -3 | cut -c1-240
    echo "admission: $(grep -c ' build: ' "$A/serve.log") builds, $(grep -c ' resize: ' "$A/serve.log") resizes, $(grep -c 'request waits' "$A/serve.log") waits"
    grep -E '^pf |^C[0-9]+ wall' "$A/bench.txt" | cut -c1-240
    awk -F'[ ,]+' -v g="$GPU" '{for(i=2;i+3<=NF;i+=4){ if($i==g){ if($(i+2)+0>=50){s+=$(i+3);n++} if($(i+1)+0>m)m=$(i+1)+0 } else if($(i+1)+0>o)o=$(i+1)+0 }}
      END{printf "busy-sample mean SM MHz %.0f (n=%d), card peak MiB %d, other cards max MiB %d\n", n?s/n:0, n, m, o}' "$A/smi.samples"
  } >> "$OUT/summary.txt"
done
