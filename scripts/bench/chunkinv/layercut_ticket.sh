#!/bin/bash
# layercut_ticket.sh MODE TAG : one layer-cut pair run inside the caller's pair GPU turn (<= 112 s); the PC producer
# (RTX, layers [0,K)) is started separately and sends to the listening ranks. MODE l8 (p8k, K=28) | l32 (p32k, K=30).
# Env: LCBIN (rank binary with layer-cut), LCPROMPTS (p8k.u32, p32k.u32, code1k.u32), ORD=adaptive|invariant
# (STRIX_MMX_PF_ORDER), C = cut block, N = layer-cut sessions; pair unit = C/2 rows, group 2. Uses pair_run.sh.
# Writes $KIT/lc/TAG.sum (TTFT, state / tokens fnv, MemAvailable, kfd evicted_ms) and $KIT/lc/TAG.done.
. "$(dirname "$0")/kit.env.sh"; D=$(cd "$(dirname "$0")" && pwd)
MODE=$1 TAG=$2; O=$KIT/lc/$TAG; mkdir -p $KIT/lc
ORD=${ORD:-invariant}; C=${C:-2048}; N=${N:-2}; LP=${LP:-47410}; MR=$((C / 2)); P=${LCPROMPTS:?}
E="STRIX_SESSION_CACHE=0 STRIX_MM_STATE_FNV=1 STRIX_MM_PF_GROUP=2 STRIX_MMX_PF_ORDER=$ORD"
case $MODE in
  l8)  K=28; ARGS="--slots 1 --capacity 8256 --max-rows $MR --session w:$P/code1k.u32:1 --layercut-listen cut:$P/p8k.u32:$LP:16:$N" ;;
  l32) K=30; ARGS="--slots 1 --capacity 32832 --max-rows $MR --layercut-listen cut:$P/p32k.u32:$LP:16:$N" ;;
  *) echo "bad mode" > $KIT/lc/$TAG.done; exit 2 ;;
esac
bash $D/pair_run.sh --bin ${LCBIN:?} --model $MODEL --peer $PEER --out $O --secs 112 $E -- $ARGS --cut $K --cut-block $C --cut-act link
{
  echo "$(date +%T) $MODE $TAG ORD=$ORD N=$N K=$K C=$C MR=$MR"
  cat $O/summary.txt
  grep -a -h -E "rank [01] layercut: prefill|rror|what\(\)|terminate" $O/r0.log $O/r1.log | cut -c1-230 | head -12
} > $KIT/lc/$TAG.sum 2>&1
touch $KIT/lc/$TAG.done
