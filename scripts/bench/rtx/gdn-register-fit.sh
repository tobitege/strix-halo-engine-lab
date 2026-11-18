#!/usr/bin/env bash
# Shared CUDA GDN register-fit A/B: ONE bounded workload, to be run by the PRO operator INSIDE the shared guard (scripts/bench/rtx/gpu_guard.py, <=210 seconds for this complete workload).
# Fail-closed: any rc != 0, missing digest/prefill line/logits dump, or any logits sha != first arm stops before the next arm.
#   gdn-register-fit.sh BASE_BIN_DIR CANDIDATE_BIN_DIR OUTPUT_DIR        (8K only; Qwen3.8-27B synthetic FAST8 prefill, same arm shape as the RTXLONG3 8K rung arm B)
set -uo pipefail
BASE_DIR=$1; NEW_DIR=$2; OUT=$3; LEN=8192
: "${STRIX_DEVICES:?Pass the device declaration through STRIX_DEVICES}"
bin_dir() { if [[ $1 == base ]]; then printf '%s' "$BASE_DIR"; else printf '%s' "$NEW_DIR"; fi; }
S=$OUT/summary.txt; mkdir -p "$OUT"
echo "gq start $(date +%T) len $LEN base $(sha256sum "$BASE_DIR/strix-cuda-qwen35-decode.exe" | cut -c1-16) new $(sha256sum "$NEW_DIR/strix-cuda-qwen35-decode.exe" | cut -c1-16)" > "$S"
# 1) kernel fixture: both builds pass, and the window-path digest is identical build-to-build
for v in base new; do
  "$(bin_dir "$v")/cuda_core_test.exe" > "$OUT/test-$v.log" 2>&1; rc=$?
  dg=$(grep -a '^gdn digest' "$OUT/test-$v.log" || echo none)
  echo "test $v rc $rc $dg" >> "$S"
  [ $rc = 0 ] && [ "$dg" != none ] || { echo "STOP: test $v failed"; tail -3 "$OUT/test-$v.log" | cut -c1-160; } >> "$S"
  [ $rc = 0 ] && [ "$dg" != none ] || exit 1
done
[ "$(grep -a '^test base' "$S" | cut -d' ' -f5-)" = "$(grep -a '^test new' "$S" | cut -d' ' -f5-)" ] || { echo "STOP: digest differs" >> "$S"; exit 1; }
echo "digest equal" >> "$S"
# 2) end-to-end ABAB x3, chunk 4096, 2 slots, inflight 3, 2 prefills per process (first = cold, last = warm); 10 s idle between arms
ref=
for i in 1 2 3; do for v in base new; do
  n=$v$i; d=$OUT/$n; mkdir -p "$d"
  "$(bin_dir "$v")/strix-cuda-qwen35-decode.exe" --synthetic-full bf16 --turbo-weights onfly --prompt-len "$LEN" --n-gen 1 --prefill-arith fast8 \
    --max-pos "$LEN" --prefill-chunk 4096 --prefill-slots 2 --prefill-inflight 3 --prefill-ramp 0 --prefill-repeat 2 \
    --prefill-dump "$d/last.f32" > "$d/run.log" 2>&1; rc=$?
  sha=$( [ -f "$d/last.f32" ] && sha256sum "$d/last.f32" | cut -c1-16 || echo none)
  echo "arm $n rc $rc sha $sha $(grep -a '^prefill\[' "$d/run.log" | tr '\n' ' ' | cut -c1-240)" >> "$S"
  if [ $rc != 0 ] || [ "$sha" = none ] || ! grep -aq '^prefill\[' "$d/run.log"; then echo "STOP: arm $n failed" >> "$S"; exit 1; fi
  ref=${ref:-$sha}
  [ "$sha" = "$ref" ] || { echo "STOP: arm $n logits sha $sha != $ref" >> "$S"; exit 1; }
  sleep 10
done; done
echo "gq end $(date +%T) all logits sha $ref" >> "$S"
