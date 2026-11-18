#!/usr/bin/env bash
# Real-model class runs of strix-cuda-qwen35-spec (PRO idx 0 only). Usage:
#   run_classes.sh BIN GGUF OUTDIR [N_GEN=512] [SOURCES=copy,chain,hybrid]
# Needs STRIX_DEVICES (strix-devices-v1 declaring the PRO) in the environment; prompts: prompts/*.ids (tokenize_prompts.py).
# Per prompt: serial vs spec (exactness gate inside: EXACT: PASS/FAIL), E mean/p50/p90, avg + best-1s tok/s, SM clock avg.
set -u
status=0
cp=
cleanup() {
  if [[ -n "$cp" ]]; then kill "$cp" 2>/dev/null || :; wait "$cp" 2>/dev/null || :; cp=; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
BIN=$1; GGUF=$2; OUT=$3; N=${4:-512}; SRC=${5:-copy,chain,hybrid}
D=$(cd "$(dirname "$0")" && pwd)/prompts
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$OUT" || exit 1
for f in "$D"/*.ids; do
  n=$(basename "$f" .ids)
  nvidia-smi --id=0 --query-gpu=clocks.sm --format=csv,noheader,nounits -lms 200 > "$OUT/$n.clk" 2>/dev/null &
  cp=$!
  "$BIN" --model "$GGUF" --prompt-file "$f" --n-gen "$N" --sources "$SRC" --label "$n" > "$OUT/$n.txt" 2>&1
  rc=$?
  cleanup
  if (( rc != 0 )) || ! grep -aq '^EXACT: PASS' "$OUT/$n.txt" || grep -aq '^EXACT: FAIL' "$OUT/$n.txt"; then
    status=1
  fi
  clk=$(awk '{s+=$1;n++} END{if(n)printf "%.0f",s/n}' "$OUT/$n.clk")
  echo "$n clk_avg_MHz=$clk $(grep -a '^serial:\|^spec\|^EXACT' "$OUT/$n.txt" | cut -c1-260 | tr '\n' '|')"
done

exit "$status"
