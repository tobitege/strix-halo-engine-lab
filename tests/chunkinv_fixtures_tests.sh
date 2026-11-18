#!/bin/bash
# Host-only exit-status regression: isolated PATH, fake fixtures/compiler, no GPU or real timeout.
set -euo pipefail
runner=${1:?fixture runner}; root=${2:?scratch directory}
mkdir -p "$root"
root=$(cd "$root" && pwd)
bin=$root/bin; build=$root/build; compiler=$root/compiler
mkdir -p "$bin" "$build" "$compiler"
# Keep optional hipcc discovery independent of the host's installed tools.
for tool in mkdir grep dirname chmod; do
  printf '#!/bin/bash\nexec %q "$@"\n' "$(command -v "$tool")" > "$bin/$tool"
done
cat > "$bin/timeout" <<'EOF'
#!/bin/bash
if [[ $FAIL_KIND == timeout && ${STRIX_G4_PREFILL_FLASH_ABSSEG:-0} == 1 && ${STRIX_CHUNKINV_NO_CARRY:-0} == 0 ]]; then exit 124; fi
shift
exec "$@"
EOF
cat > "$build/strix-hip-gdn-fp16-chunk-tests" <<'EOF'
#!/bin/bash
echo 'open tail: fake host fixture'
if [[ $FAIL_KIND == gdn$STRIX_QPF2K_GDN_FP16_CHUNK ]]; then exit 7; fi
exit 0
EOF
cat > "$build/strix-gemma4-prefill-attention-device-tests" <<'EOF'
#!/bin/bash
echo 'aligned plans: fake host fixture'
if [[ $FAIL_KIND == g4$STRIX_G4_PREFILL_FLASH_ABSSEG || ( $FAIL_KIND == nocarry && ${STRIX_CHUNKINV_NO_CARRY:-0} == 1 ) ]]; then exit 9; fi
exit 0
EOF
cat > "$compiler/hipcc" <<'EOF'
#!/bin/bash
if [[ $FAIL_KIND == build ]]; then exit 11; fi
out=${!#}
printf '%s\n' '#!/bin/bash' 'echo "fake probe $KPAIR_ZERO"' 'if [[ $FAIL_KIND == probe-$KPAIR_ZERO ]]; then exit 13; fi' 'exit 0' > "$out"
chmod +x "$out"
EOF
chmod +x "$bin/"* "$build/"* "$compiler/hipcc"
failures=0
check() {
  local name=$1 kind=$2 with_compiler=$3 expected=$4 marker=$5
  local path=$bin source=$build output rc
  if [[ $with_compiler == yes ]]; then path=$compiler:$path; fi
  if [[ $kind == missing ]]; then source=$root/missing; fi
  if output=$(PATH="$path" FAIL_KIND="$kind" CHUNKINV_KEYS=32 STRIX_CHUNKINV_NO_CARRY=0 \
      "$BASH" "$runner" "$source" "$root/$name" 2>&1); then rc=0; else rc=$?; fi
  if [[ $rc != "$expected" || $output != *"$marker"* ]]; then
    printf 'FAIL %s: expected rc %s and marker %s, got rc %s\n' "$name" "$expected" "$marker" "$rc"
    failures=$((failures + 1))
  fi
}
check optional-skip none no 0 'SKIP wmma probe: optional hipcc unavailable'
check all-pass none yes 0 'wmma probe pair rc 0'
check first-gdn gdn1 no 1 'gdn rev 3 rc 0'
check second-gdn gdn3 yes 1 'wmma probe pair rc 0'
check block-share g40 no 1 'g4 absseg=0 n 32 rc 9'
check absolute-segments g41 no 1 'g4 absseg=1 n 32 rc 9'
check timed-out timeout no 1 'g4 absseg=1 n 32 rc 124'
check negative-control nocarry no 1 'g4 absseg no-carry n 32 rc 9'
check missing-fixtures missing no 1 'gdn rev 1 rc 127'
check compiler-error build yes 1 'wmma probe build rc 11'
check first-probe probe-one yes 1 'wmma probe pair rc 0'
check second-probe probe-pair yes 1 'wmma probe pair rc 13'
printf 'chunkinv_fixtures_tests: %s failures (12 cases)\n' "$failures"
[[ $failures == 0 ]]
