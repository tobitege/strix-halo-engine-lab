#!/bin/bash
# gate_seal.sh RID PROMPTS : after the GPU turn (CPU only): engine rc / errors, annotation, execution witness + seal on each
# rank (rank 1 on the peer, from its own log and dumps), copy rank 1's sealed dumps here, drop the peer's copies. 0 = sealed.
set -u
. "$(dirname "$0")/kit.env.sh"; D=$(cd "$(dirname "$0")" && pwd)
RID=$1 PR=$2; R=$KIT/run/$RID
fail() { echo "SEAL-FAIL $RID: $*" | tee $R/seal.verdict; exit 3; }
[ "$(cat $R/rc 2>/dev/null)" = "0 0" ] || fail "engine rc $(cat $R/rc 2>/dev/null)"
scp -q $PEER:$R/r1.log $PEER:$R/mon1.txt $R/ || fail "copy r1.log / mon1.txt"
grep -a -E "rror|what\(\)|timed out|ranks DIFFER" $R/r0.log $R/r1.log | head -3 | grep -q . && fail "engine log reports an error"
bash $D/gate_annot.sh $R/r0.log; bash $D/gate_annot.sh $R/r1.log; scp -q $R/r1.log $PEER:$R/r1.log
grep -a -q "ranks AGREE" $R/r0.log || fail "no rank agreement line"
[ "$(sha256sum < $BIN)" = "$($S2 "sha256sum < $BIN")" ] || fail "binary differs between the ranks"
PA=""; for p in ${PR//,/ }; do PA="$PA --prompt $p=$PROMPTS/$p.u32"; done
C="--dir $R --run-id $RID --binary $BIN --rows 256 --select --attest sel_mode=4 --not-before $(cut -d. -f1 $R/t0) $PA"
EA=""; for e in $(cat $R/arm-env); do EA="$EA --env $e"; done
G="nice python3 $GATE"
$G witness $C --rank 0 --engine minimax --engine-log $R/r0.log --model $MODEL --out $R/execution-r0.json > $R/seal.out 2>&1 || fail "witness r0: $(tail -1 $R/seal.out)"
$G seal $C --rank 0 --arm $RID --execution-witness $R/execution-r0.json $EA >> $R/seal.out 2>&1 || fail "seal r0: $(tail -1 $R/seal.out)"
$S2 "$G witness $C --rank 1 --engine minimax --engine-log $R/r1.log --model $MODEL --out $R/execution-r1.json && $G seal $C --rank 1 --arm $RID --execution-witness $R/execution-r1.json $EA" >> $R/seal.out 2>&1 || fail "rank 1: $(tail -1 $R/seal.out)"
F=""; for p in ${PR//,/ }; do F="$F $PEER:$R/q-$p-tf-r1.f32 $PEER:$R/q-$p-sel-r1.u32"; done
scp -q $F $PEER:$R/manifest-r1.json $PEER:$R/execution-r1.json $R/ || fail "copy of the rank 1 dumps"
for p in ${PR//,/ }; do [ -s $R/q-$p-tf-r1.f32 ] || fail "rank 1 dump of $p missing after copy"; done
$S2 "rm -f $R/q-*"
rm -f $R/q-*-logits-r0.f32 $R/q-*-row-r0.f32
echo "OK" > $R/seal.verdict; echo "sealed $RID"
