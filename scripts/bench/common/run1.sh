#!/usr/bin/env bash
# run1.sh DIR CAP_S -- CMD...  [env KIT_TAG KIT_ARM KIT_ROUND KIT_TOKENS KIT_RECEIPT_ARGS]
# One measured process (sampler, stamped stderr, klog, meta) followed by its receipt line. Used locally by the
# kits and remotely on the pair peer (ssh PEER bash <repo>/scripts/bench/common/run1.sh ...), so both ranks of a
# pair arm carry the same validity data.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/runlib.sh"
dir=$1 cap=$2; shift 2
[[ ${1:-} == -- ]] && shift
# The arm's own STRIX_* assignments travel as `env A=B` words inside CMD, so a clean slate is safe here.
kit_clean_env
ulimit -s 65536 2>/dev/null || true
kit_run "$dir" "$cap" "$@"
read -ra extra <<<"${KIT_RECEIPT_ARGS:-}"
kit_receipt "$dir" "${KIT_TAG:-$(basename "$dir")}" --arm "${KIT_ARM:-${KIT_TAG:-$(basename "$dir")}}" \
    --workload "${KIT_WORKLOAD:-}" --round "${KIT_ROUND:-0}" --tokens "${KIT_TOKENS:-0}" "${extra[@]}"
