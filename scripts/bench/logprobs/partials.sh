#!/usr/bin/env bash
# Logprob partial kernels (strix/logprob_partials.hpp): correctness against the FP64 reference and the kernel cost per
# step at several row counts (DS41 rank slice: 64,640 entries per row). One GPU, one process per arm.
# Usage: bash partials.sh --bin PATH [--rows "1 8 32 128"] [--top 20]
#   PATH: strix-hip-logprob-partials-tests or strix-cuda-logprob-partials-tests (select the device with the
#   platform's visibility variable before calling; the script never chooses one).
set -euo pipefail
bin="" rows="1 8 32 128" top=20
while [ $# -gt 0 ]; do
    case "$1" in
        --bin) bin=$2; shift 2 ;;
        --rows) rows=$2; shift 2 ;;
        --top) top=$2; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done
[ -x "$bin" ] || { echo "--bin must name the test binary" >&2; exit 2; }
"$bin"
for r in $rows; do "$bin" --bench "$r" "$top"; done
