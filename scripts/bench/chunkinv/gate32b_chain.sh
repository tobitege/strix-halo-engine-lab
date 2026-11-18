#!/bin/bash
# gate32b_chain.sh : the multi-unit invariant-order gate in one detached chain on rank 0's node (setsid nohup).
# Env: KIT (run root, same path on the peer), MODEL, PROMPTS (mkunits.py output, also on the peer), PEER, BIN, GATE,
# TURN (pair GPU-turn wrapper: TURN LABEL CMD...), PY (python with numpy + tokenizers for the statistics),
# UNITS (default prose,docs,code,json,chat,mixed), NULLS (default 3), WARM (unit for the warm arms, empty = none),
# STEPS (greedy steps, default 64), CAP (capacity, default 32832), SECS (engine timeout, default 115).
# Per unit: U-e (adaptive), U-i (invariant), U-n0s1..NULLS (adaptive + one-ulp L0 null); then WARM-ew / WARM-iw with
# a 256-token warm-up session first. Each arm = one GPU turn (label gate32b-RID; the turn wrapper drops the environment,
# so the kit values are passed explicitly), sealed outside the turn; one retry under a new RID. Writes $KIT/gate32b.summary (arm lines + statistics) and touches $KIT/gate32b.done.
set -u
D=$(cd "$(dirname "$0")" && pwd)
: "${KIT:?}" "${MODEL:?}" "${PROMPTS:?}" "${PEER:?}" "${BIN:?}" "${GATE:?}" "${TURN:?}" "${PY:?}"
export KIT MODEL PROMPTS PEER BIN GATE
UNITS=${UNITS:-prose,docs,code,json,chat,mixed} NULLS=${NULLS:-3} WARM=${WARM-prose}
export STEPS=${STEPS:-64}; CAP=${CAP:-32832} SECS=${SECS:-115}
S=$KIT/gate32b.summary; mkdir -p $KIT/tk; : > $S; rm -f $KIT/gate32b.done
echo "start $(date +%F' '%T) bin $(sha256sum < $BIN | cut -c1-12) units $UNITS nulls $NULLS warm '$WARM' steps $STEPS" >> $S
envof() { case $1 in e|ew) echo STRIX_MMX_PF_ORDER=adaptive;; i|iw) echo STRIX_MMX_PF_ORDER=invariant;;
  n0s*) echo "STRIX_MMX_PF_ORDER=adaptive STRIX_MM_DIAG_ATTN_ULP=0:${1#n0s}";; esac; }
fails=0
arm() { local u=$1 a=$2 sess=$3 try rid t0
  for try in "" r; do rid=$u-$a$try; t0=$(date +%s)
    $TURN gate32b-$rid env KIT=$KIT MODEL=$MODEL PROMPTS=$PROMPTS PEER=$PEER BIN=$BIN GATE=$GATE STEPS=$STEPS \
      bash $D/gate_run.sh $rid $SECS $CAP $sess $(envof $a) > $KIT/tk/$rid.out 2>&1
    # seal the measured (last) session only: a short warm-up session writes no selection records
    if bash $D/gate_seal.sh $rid ${sess##*,} >> $KIT/tk/$rid.out 2>&1; then
      if [ -n "$try" ]; then rm -rf $KIT/run/$u-$a; mv $KIT/run/$rid $KIT/run/$u-$a
        ssh -n -o BatchMode=yes $PEER "rm -rf $KIT/run/$u-$a; mv $KIT/run/$rid $KIT/run/$u-$a"; fi
      local R=$KIT/run/$u-$a
      echo "$u-$a sealed (try '$try') wall $(( $(date +%s) - t0 )) s $(grep -a -o 'TTFT [0-9.]* s' $R/r0.log | tr '\n' ' ')$(tr '\n' ' ' < $R/mem.txt | cut -c1-120) evmax r0 $(grep -o 'ev=[0-9]*\|evicted_ms=[0-9]*' $R/mon0.txt | cut -d= -f2 | sort -n | tail -1) r1 $(grep -o 'ev=[0-9]*\|evicted_ms=[0-9]*' $R/mon1.txt | cut -d= -f2 | sort -n | tail -1)" >> $S
      fails=0; return 0; fi
    echo "  $rid: $(grep -a 'gate_run\|SEAL-FAIL' $KIT/tk/$rid.out | tail -2 | tr '\n' ' ' | cut -c1-220)" >> $S
  done
  fails=$((fails + 1)); [ $fails -ge 2 ] && { echo "ABORT: two arms in a row failed twice" >> $S; touch $KIT/gate32b.done; exit 1; }
  return 1; }
for u in ${UNITS//,/ }; do
  arm $u e $u; arm $u i $u
  for s in $(seq 1 $NULLS); do arm $u n0s$s $u; done
done
[ -n "$WARM" ] && { arm $WARM ew w256,$WARM; arm $WARM iw w256,$WARM; }
WA=""; [ -n "$WARM" ] && [ -f $KIT/run/$WARM-ew/seal.verdict ] && [ -f $KIT/run/$WARM-iw/seal.verdict ] && WA="--warm $WARM"
nice $PY $D/gate32b_stats.py --kit $KIT --gate $GATE --tokenizer $MODEL/tokenizer.json --units $UNITS --nulls $NULLS $WA \
  --out $KIT/gate32b-stats > $KIT/tk/stats.out 2>&1 || echo "stats failed: $(tail -3 $KIT/tk/stats.out | tr '\n' ' ' | cut -c1-300)" >> $S
[ -s $KIT/gate32b-stats.txt ] && cat $KIT/gate32b-stats.txt >> $S
echo "done $(date +%F' '%T)" >> $S; touch $KIT/gate32b.done
