#!/bin/bash
# seal.sh RID PROMPTS [--rank1-only] : after the ticket (CPU): engine rc / errors, execution witness + seal on each rank
# (rank 1 on $MMF_PEER from its own log and dumps), copy rank 1's sealed dumps here, drop the peer's copies. Exit 0 = sealed.
# --rank1-only resumes a seal whose rank-0 half already ran (witnesses are never rewritten).
# Environment: as arm.sh, plus MMF_GATE (tools/strix-gate/fast_logit_gate.py, same path on both nodes).
set -u
RID=$1 PR=$2 ONLY1=${3:-}
: "${MMF_PEER:?}" "${MMF_BIN:?}" "${MMF_MODEL:?}" "${MMF_PROMPTS:?}" "${MMF_RUN:?}" "${MMF_GATE:?}"
R=$MMF_RUN/$RID BIN=$MMF_BIN Z=$MMF_PROMPTS
S2="ssh -n -o BatchMode=yes -o ConnectTimeout=5 $MMF_PEER"
fail() { echo "SEAL-FAIL $RID: $*" | tee "$R/seal.verdict"; exit 3; }
[ "$(cat "$R/rc" 2>/dev/null)" = "0 0" ] || fail "engine rc $(cat "$R/rc" 2>/dev/null)"
scp -q "$MMF_PEER:$R/r1.log" "$MMF_PEER:$R/mon1.txt" "$R/" || fail "copy r1.log / mon1.txt"
grep -a -E "rror|what\(\)|timed out|ranks DIFFER" "$R/r0.log" "$R/r1.log" | head -3 | grep -q . && fail "engine log reports an error"
grep -a -q "ranks AGREE" "$R/r0.log" || fail "no rank agreement line"
[ "$(sha256sum < "$BIN")" = "$($S2 "sha256sum < $BIN")" ] || fail "binary differs between the ranks"
$S2 "test -r $MMF_GATE" || fail "gate tool missing on the peer"
PA=""; for p in ${PR//,/ }; do PA="$PA --prompt $p=$Z/$p.u32"; done
C="--dir $R --run-id $RID --binary $BIN --rows 256 --select --attest sel_mode=4 --not-before $(cut -d. -f1 "$R/t0") $PA"
EA=""; for e in $(cat "$R/arm-env"); do EA="$EA --env $e"; done
G="nice python3 $MMF_GATE"
if [ "$ONLY1" != --rank1-only ]; then
  $G witness $C --rank 0 --engine minimax --engine-log "$R/r0.log" --model "$MMF_MODEL" --out "$R/execution-r0.json" > "$R/seal.out" 2>&1 || fail "witness r0: $(tail -1 "$R/seal.out")"
  $G seal $C --rank 0 --arm "$RID" --execution-witness "$R/execution-r0.json" $EA >> "$R/seal.out" 2>&1 || fail "seal r0: $(tail -1 "$R/seal.out")"
fi
$S2 "$G witness $C --rank 1 --engine minimax --engine-log $R/r1.log --model $MMF_MODEL --out $R/execution-r1.json && $G seal $C --rank 1 --arm $RID --execution-witness $R/execution-r1.json $EA" >> "$R/seal.out" 2>&1 || fail "rank 1: $(tail -1 "$R/seal.out")"
F=""; for p in ${PR//,/ }; do F="$F $MMF_PEER:$R/q-$p-tf-r1.f32 $MMF_PEER:$R/q-$p-sel-r1.u32"; done
scp -q $F "$MMF_PEER:$R/manifest-r1.json" "$MMF_PEER:$R/execution-r1.json" "$R/" || fail "copy of the rank 1 dumps"
for p in ${PR//,/ }; do [ -s "$R/q-$p-tf-r1.f32" ] || fail "rank 1 dump of $p missing after copy"; done
$S2 "rm -f $R/q-*"
rm -f "$R"/q-*-logits-r0.f32 "$R"/q-*-row-r0.f32
echo "OK" > "$R/seal.verdict"; echo "sealed $RID"
