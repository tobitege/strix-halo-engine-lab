#!/usr/bin/env bash
# Verify-window rows table of strix-cuda-qwen35-decode (ms per step for R rows). Usage:
#   rows_table.sh BIN "MODEL_ARGS" "ROWS" [EXTRA_ARGS]
#   MODEL_ARGS: "--model PACK.gguf [--turbo-weights onfly]" or "--synthetic-full bf16|q8_0"
#   ROWS: e.g. "1 2 4 8 16 32". Needs STRIX_DEVICES (strix-devices-v1 declaring the GPU) and CUDA_VISIBLE_DEVICES set by the caller.
set -u
BIN=$1; MODEL=$2; ROWS=$3; EXTRA=${4:-}
for R in $ROWS; do
  printf 'R=%s ' "$R"
  # shellcheck disable=SC2086
  "$BIN" $MODEL --decode-rows "$R" --ctx-fill 16 $EXTRA 2>&1 | grep '^decode' | sed 's/.*steps, //' | cut -c1-80
done
