#!/usr/bin/env bash
# unit.sh BUILD_DIR OUT_DIR [--trace]: device unit gate + timing of the FAST router product on the shared FP16 WMMA core
# (glm-pf-router-wmma-v1, mm-pf-router-wmma-v1). Runs strix-hip-glm5-pf-router-wmma-tests and
# strix-hip-mm-router-wmma-tests twice each (gate, then --timing), records the GPU clock and MemAvailable around the
# timing runs, and with --trace one rocprofv3 kernel trace per test (per-kernel durations: exact, WMMA forms, fallback,
# selection). Writes OUT_DIR/summary.txt (<= 60 lines) and OUT_DIR/unit.done. Run inside the node's GPU turn.
set -u
B=${1:?BUILD_DIR}; OUT=${2:?OUT_DIR}; TRACE=${3:-}
mkdir -p "$OUT" || exit 1
rm -f "$OUT/unit.done" || exit 1
S=$OUT/summary.txt; : > "$S" || exit 1
failed=0
find_bin() { find "$B" -name "$1" -type f -perm -u+x | head -1; }
clock() { for f in /sys/class/drm/card*/device/pp_dpm_sclk; do [ -r "$f" ] && grep -m1 '\*' "$f" | awk '{print $2}' && return; done; echo "?"; }
mem() { awk '/MemAvailable/ {printf "%.1f GiB", $2 / 1048576}' /proc/meminfo; }
for t in strix-hip-glm5-pf-router-wmma-tests strix-hip-mm-router-wmma-tests; do
  bin=$(find_bin "$t"); [ -n "$bin" ] || { echo "MISSING $t" >> "$S"; failed=1; continue; }
  "$bin" > "$OUT/$t.gate.log" 2>&1; rc=$?
  [ "$rc" -eq 0 ] || failed=1
  { echo "== $t gate rc $rc"; grep -E "^(FAIL|DRIFT|PASS)" "$OUT/$t.gate.log" | head -14; } >> "$S"
  c0=$(clock); m0=$(mem)
  "$bin" --timing > "$OUT/$t.time.log" 2>&1; rc=$?
  [ "$rc" -eq 0 ] || failed=1
  { echo "== $t timing rc $rc sclk $c0 -> $(clock) MemAvailable $m0 -> $(mem)"; grep -E "^TIME" "$OUT/$t.time.log" | head -8; } >> "$S"
  if [ "$TRACE" = --trace ]; then
    if ! command -v rocprofv3 > /dev/null; then
      echo "MISSING rocprofv3" >> "$S"; failed=1; continue
    fi
    rocprofv3 --kernel-trace --stats -d "$OUT/trace-$t" -o run -- "$bin" --timing > "$OUT/$t.trace.log" 2>&1 || failed=1
    st=$(find "$OUT/trace-$t" -name '*kernel_stats.csv' | head -1)
    if [ -z "$st" ]; then
      echo "MISSING $t kernel stats" >> "$S"; failed=1; continue
    fi
    echo "== $t kernel stats (name, calls, total ms, mean us)" >> "$S"
    python3 - "$st" <<'PY' >> "$S" || failed=1
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
rows.sort(key=lambda r: -float(r.get("TotalDurationNs", 0)))
for r in rows[:8]:
    n = r["Name"].replace("void ", "")[:70]
    print("  %-70s %6s %9.2f %9.1f" % (n, r["Calls"], float(r["TotalDurationNs"]) / 1e6, float(r["AverageNs"]) / 1e3))
PY
  fi
done
[ "$failed" -eq 0 ] || exit 1
touch "$OUT/unit.done"
