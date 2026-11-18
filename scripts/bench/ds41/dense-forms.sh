#!/usr/bin/env bash
# Shared dense policy dense-q8a8-f16-v1 (DeepSeek V4.1 binds it): its weight-decode forms on the DS41 rank-0 chunk shapes,
# through the kernel test binary. Form 0 = one FP32 product per weight, form 1 = two weights per packed FP16 operation.
#
#   scripts/bench/common/ticket.sh OUT ds41/dense-forms --bin BIN [--rows 1024,680] [--reps 20] [--rounds 11]
#
#   BIN  strix-hip-ds41-fast-dense-tests. First --no-timing: the stager decode gate (all 65,536 FP16 scales x 256 codes,
#        packed against scalar, bit for bit), every form against form 0 bit for bit, the library's instantiation against
#        form 0, in separate default/OFF processes. Then per rows: --forms REPS ROUNDS ROWS (reps 1..1000,
#        rounds 2..101, rows 1..16384), strict ABAB rounds (every round runs every form once; cold =
#        round 0, warm = median of the others, d = per-round difference to form 0), "census" = launches per layer-chunk
#        x warm ms and x113 layer-chunks (rank 0 at 11.6K in 1,024-row chunks; the last chunk of a rank is ~680 rows).
# No prefill receipt (receipt.py wants a prefill anchor); each run's dir keeps the sampler (clock, kfd evictions) and meta.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/../common/runlib.sh"
bin='' rows=1024,680 reps=20 rounds=11
while (($#)); do
    case $1 in
        --bin) bin=$2 ;; --rows) rows=$2 ;; --reps) reps=$2 ;; --rounds) rounds=$2 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) kit_die "unknown argument $1" ;;
    esac
    shift 2
done
[[ -x $bin ]] || kit_die "--bin: no executable $bin"
out=${KIT_OUT:?run through common/ticket.sh}
kit_clean_env
# This kit has no receipt: promote both process status and test completion to its verdict.
check_run() {
    "$KIT_PY" - "$1" "$2" "${3:-}" <<'PY'
import json
import pathlib
import sys

directory, label, expected_form = sys.argv[1:]
try:
    path = pathlib.Path(directory)
    rc = json.loads((path / "meta.json").read_text())["rc"]
    lines = (path / "stdout").read_text().splitlines()
    if type(rc) is not int or rc != 0:
        raise ValueError(f"rc={rc}")
    if not lines or lines[-1] != "PASS" or any("FAIL" in line for line in lines):
        raise ValueError("missing successful completion or failed check")
    if expected_form and f"dispatch form {expected_form}" not in lines:
        raise ValueError(f"expected library dispatch form {expected_form}")
except (OSError, ValueError, KeyError, TypeError) as error:
    print(f"== {label} INVALID/ABORT: {error}")
    sys.exit(1)
PY
}
echo "== bin $(kit_sha "$bin") rows $rows reps $reps rounds $rounds"
(unset STRIX_F16_Q8A8_WPACK; kit_run "$out/bits" 60 "$bin" --no-timing)
check_run "$out/bits" bits 1 || exit 1
grep -aE '^(wpack|fm .*bit-equal)|FAIL|^PASS' "$out/bits/stdout" | head -12 | sed 's/^/== bits /'
(export STRIX_F16_Q8A8_WPACK=0; kit_run "$out/bits-off" 60 "$bin" --no-timing)
check_run "$out/bits-off" bits-off 0 || exit 1
echo '== bits dispatch default=1 OFF=0 PASS'
for r in ${rows//,/ }; do
    kit_fits 20 || { echo "== rows $r SKIP (left $(kit_left) s)"; continue; }
    kit_run "$out/f$r" 60 "$bin" --forms "$reps" "$rounds" "$r"
    check_run "$out/f$r" "rows $r" || exit 1
    grep -aE '^(fm|census)|FAIL' "$out/f$r/stdout" | grep -v 'form 0 |' | cut -c1-200 | sed 's/^/== /'
done
