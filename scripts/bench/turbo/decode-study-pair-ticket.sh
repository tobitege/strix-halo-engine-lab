#!/bin/bash
# Invoked inside one bounded GPU ticket. Deployment supplies the declared server adapter and sealed assets.
set -u
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
K=${TURBODEC_ROOT:?Set TURBODEC_ROOT to the study runner and output directory}
TAG=${1:?pair tag required}; V=${2:-v2}; ARMS=${3:-C1,C1}
source "$SCRIPT_DIR/decode-study-ticket.sh"
study_claim "$TAG" || exit $?
BEGIN=$(date +%s); DEADLINE=${TURBODEC_PAIR_BUDGET:-110}
for arm in E T; do
  LEFT=$((DEADLINE - ($(date +%s) - BEGIN) - 8))
  [ "$arm" = E ] && [ "$LEFT" -gt 75 ] && LEFT=75
  [ "$LEFT" -lt 30 ] && { printf 'insufficient paired startup budget\n' > "$K/$TAG.invalid"; exit 1; }
  args=''; [ "$arm" = T ] && args='--profile turbo --turbo-weights onfly'
  TURBODEC_PROMPT_MODULE=${TURBODEC_CODE_PROMPT_MODULE:-$SCRIPT_DIR/decode-study-inventory.py} LEFT=$LEFT SLOTS=${SLOTS:-2} MAXTOK=${MAXTOK:-128} XARGS="$args" bash "${TURBODEC_TICKET_RUNNER:-$SCRIPT_DIR/decode-study-ticket.sh}" "$TAG-$arm" "$V" "$ARMS" STRIX_PLE_REWARM=0 STRIX_HTTP_SLOT_TIMING=1 || exit $?
  if ! python3 - "$K/runs/$TAG-$arm/res.jsonl" <<'PY'
import json,sys
try:
 rows=[json.loads(x) for x in open(sys.argv[1])]
 assert len(rows)>=2 and all(r['tokens']>0 and not r['errors'] for r in rows)
except Exception:sys.exit(1)
PY
  then printf '%s invalid startup or requests\n' "$arm" > "$K/$TAG.invalid"; exit 1; fi
 done
printf 'E T valid, elapsed %ss\n' "$(( $(date +%s) - BEGIN ))" > "$K/$TAG.done"
