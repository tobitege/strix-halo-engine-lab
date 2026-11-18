#!/bin/bash
# chunkinv-attn-state-v1 GPU fixtures: fixtures.sh BUILD_DIR OUT_DIR
# GDN FAST open-tail contract (rev 2/3), the dense flash split identities (block share vs dense-flash-absseg-v1) with
# the WMMA K-pair carry (chunk_contract.hpp section 4; NO_CARRY=1 is the negative control), and the K-pair probe.
# Run inside the node's GPU turn wrapper; prints one line per fixture. CHUNKINV_KEYS overrides the prompt lengths.
set -u
B=${1:?build dir}; O=${2:?out dir}; mkdir -p "$O" || exit 1
failed=0
record_status() { if [ "$1" -ne 0 ]; then failed=1; fi; }
for rev in 1 3; do STRIX_QPF2K_GDN_FP16_CHUNK=$rev timeout 60 "$B/strix-hip-gdn-fp16-chunk-tests" > "$O/gdn-rev$rev.txt" 2>&1
  rc=$?; record_status "$rc"
  echo "gdn rev $rev rc $rc | $(grep 'open tail:' "$O/gdn-rev$rev.txt")"; done
G=$B/strix-gemma4-prefill-attention-device-tests
for n in ${CHUNKINV_KEYS:-8192 9000}; do
  for mode in 0 1; do STRIX_G4_PREFILL_FLASH_ABSSEG=$mode timeout 100 "$G" --chunkinv "$n" > "$O/g4-$mode-$n.txt" 2>&1
    rc=$?; record_status "$rc"
    echo "g4 absseg=$mode n $n rc $rc: $(grep 'aligned plans' "$O/g4-$mode-$n.txt")"; done
  STRIX_CHUNKINV_NO_CARRY=1 STRIX_G4_PREFILL_FLASH_ABSSEG=1 timeout 100 "$G" --chunkinv "$n" > "$O/g4-nocarry-$n.txt" 2>&1
  rc=$?; record_status "$rc"
  echo "g4 absseg no-carry n $n rc $rc: $(grep 'aligned plans' "$O/g4-nocarry-$n.txt")"
done
P=$(dirname "$0")/wmma_kpair_probe.hip
if command -v hipcc > /dev/null; then
  hipcc --offload-arch="${CHUNKINV_ARCH:-gfx1151}" -O2 -w "$P" -o "$O/wmma_kpair_probe" > "$O/wmma-build.txt" 2>&1
  rc=$?; record_status "$rc"; echo "wmma probe build rc $rc"
  if [ "$rc" -eq 0 ]; then
    for m in one pair; do
      KPAIR_ZERO=$m timeout 60 "$O/wmma_kpair_probe" > "$O/wmma-$m.txt" 2>&1
      rc=$?; record_status "$rc"; echo "wmma probe $m rc $rc (output: $O/wmma-$m.txt)"
    done
  fi
else
  echo "SKIP wmma probe: optional hipcc unavailable"
fi
exit "$failed"
