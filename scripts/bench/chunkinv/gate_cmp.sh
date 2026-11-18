#!/bin/bash
# gate_cmp.sh UNIT REF CAND PROMPTS : absolute + floor (v1.2, onset 0) comparison of CAND against REF with every sealed
# null arm UNIT-n0s1..5 -> $KIT/cmp/CAND-vs-REF{,-abs}.{txt,json}. Use 5 nulls: with 3, a candidate equal to the nulls
# fails a mu+2sd check about 11 % of the time.
. "$(dirname "$0")/kit.env.sh"
cd $KIT; mkdir -p cmp; u=$1 ref=$2 c=$3 pr=$4
T=""; for p in ${pr//,/ }; do T="$T --tokens $p=$PROMPTS/$p.u32"; done
nl=""; for s in 1 2 3 4 5; do [ "$(cat run/$u-n0s$s/seal.verdict 2>/dev/null)" = OK ] && nl="$nl --null-arm run/$u-n0s$s"; done
nice python3 $GATE compare run/$ref run/$c --ranks 0 1 $T --json cmp/$c-vs-$ref-abs.json > cmp/$c-vs-$ref-abs.txt 2>&1
nice python3 $GATE compare run/$ref run/$c --ranks 0 1 $T --dnll-rule floor --engine minimax --profile fast-prefill $nl --json cmp/$c-vs-$ref.json > cmp/$c-vs-$ref.txt 2>&1
echo "rc $? nulls:$nl" >> cmp/$c-vs-$ref.txt; grep -a -E '^check|POOLED' cmp/$c-vs-$ref.txt
