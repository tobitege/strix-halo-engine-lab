#!/usr/bin/env bash
# One long-context prefill ARM of the CUDA Qwen3.8-27B engine (strix-cuda-qwen35-decode) on one GPU: cold + warm prefills in one
# process (--prefill-repeat 2: first = cold, last = warm), GPU clock / utilisation / power / memory samples, the last-token logits
# dump of the last prefill (identity checks between arms: cmp of the .f32 files). Long tests (128K+) run ONE arm per GPU window:
# call this once per window and collect the arms with long_abab_table.py.
#
#   long_arm.sh OUT_DIR ARM_NAME LEN [engine flags ...]
# Required env: EXE (strix-cuda-qwen35-decode binary). Optional: GPU_INDEX (0), WEIGHTS ("--synthetic-full bf16 --turbo-weights onfly"),
#               CHUNK (4096), POLICY (fast8), REPEAT (2), LIMIT_S (900). Outputs only under OUT_DIR/ARM_NAME.
set -u
OUT=$1; ARM=$2; LEN=$3; shift 3
: "${EXE:?}"
GPU=${GPU_INDEX:-0}
A="$OUT/$ARM"; mkdir -p "$A"
WEIGHTS=${WEIGHTS:---synthetic-full bf16 --turbo-weights onfly}
( while true; do nvidia-smi -i "$GPU" --query-gpu=clocks.sm,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits >> "$A/smi.csv"; sleep 1; done ) &
SP=$!
t0=$(date +%s)
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=$GPU timeout "${LIMIT_S:-900}" "$EXE" $WEIGHTS --prompt-len "$LEN" --n-gen 1 --prefill-arith "${POLICY:-fast8}" \
  --prefill-chunk "${CHUNK:-4096}" --prefill-repeat "${REPEAT:-2}" --max-pos "$LEN" --prefill-dump "$A/last.f32" "$@" > "$A/run.log" 2>&1
rc=$?
kill $SP 2>/dev/null
{
  echo "arm $ARM len $LEN rc $rc wall_s $(( $(date +%s) - t0 )) flags [$*]"
  grep -a "prefill\|identity" "$A/run.log" | grep -v "^\s*$" | head -6 | cut -c1-200
  awk -F', *' '$2 >= 50 {n++; c+=$1; p+=$4} {if ($3 > m) m = $3} END {if (n) printf "busy samples %d: SM MHz %.0f, W %.0f, peak MiB %.0f\n", n, c/n, p/n, m; else printf "no busy samples, peak MiB %.0f\n", m}' "$A/smi.csv"
  [ -f "$A/last.f32" ] && echo "logits sha256 $(sha256sum "$A/last.f32" | cut -c1-16)"
} > "$A/summary.txt"
cat "$A/summary.txt"
exit $rc
