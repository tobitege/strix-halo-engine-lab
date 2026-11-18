#!/bin/bash
# Invoked inside one GPU ticket: native TURBO control, then fixed-class E/T probes.
set -u
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
K=${TURBODEC_ROOT:?study runner directory required}; TAG=${1:-tdctrl};V=${2:-v2};BEGIN=$(date +%s)
source "$SCRIPT_DIR/decode-study-ticket.sh"
study_claim "$TAG" || exit $?
for arm in N E T; do
 remain=$((115-($(date +%s)-BEGIN)))
 if [ "$remain" -lt 30 ]; then printf '%s skipped: remaining %ss <30s\n' "$arm" "$remain" > "$K/$TAG.partial"; exit 1; fi
 left=$((remain-8));[ "$left" -gt 38 ] && left=38
 args='';prompt=${TURBODEC_CLASS_PROMPT_MODULE:-$SCRIPT_DIR/decode-study-classes.py};arms=S4,S4;tokens=64
 [ "$arm" = N ] && { args='--profile turbo';prompt=${TURBODEC_CODE_PROMPT_MODULE:-$SCRIPT_DIR/decode-study-inventory.py};arms=C1,C1;tokens=192; }
 [ "$arm" = T ] && args='--profile turbo --turbo-weights onfly'
  LEFT=$left SLOTS=2 MAXTOK=$tokens TURBODEC_PROMPT_MODULE=$prompt XARGS="$args" bash "${TURBODEC_TICKET_RUNNER:-$SCRIPT_DIR/decode-study-ticket.sh}" "$TAG-$arm" "$V" "$arms" STRIX_PLE_REWARM=0 STRIX_HTTP_SLOT_TIMING=1 || exit $?
 if ! python3 - "$K/runs/$TAG-$arm/res.jsonl" <<'PY'
import json,sys
try:
 rows=[json.loads(x) for x in open(sys.argv[1])]
 assert len(rows)>=2 and all(r['tokens']>0 and not r['errors'] for r in rows)
except Exception:sys.exit(1)
PY
 then printf '%s invalid startup/requests\n' "$arm" > "$K/$TAG.invalid"; exit 1; fi
 done
printf 'N/Eclasses/Tclasses valid elapsed %ss\n' "$(( $(date +%s)-BEGIN ))" > "$K/$TAG.done"
