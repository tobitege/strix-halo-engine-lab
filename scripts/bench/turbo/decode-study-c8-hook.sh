#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Optional same-server diagnostic; failure never invalidates completed C1 requests.
K=${TURBODEC_ROOT:?study directory required};PORT=$1;OUT=$2;LOCAL_LEFT=$3
NOW=$(date +%s);REMAIN=$((115-(NOW-${TURBODEC_TICKET_BEGIN:-NOW})))
if [ "$REMAIN" -lt 40 ] || [ "$LOCAL_LEFT" -lt 22 ];then printf 'C8 skipped global=%ss local=%ss\n' "$REMAIN" "$LOCAL_LEFT" > "$OUT/c8.skipped";exit 0;fi
SEC=$((REMAIN-15));[ "$SEC" -gt 25 ] && SEC=25
LIM=$((LOCAL_LEFT-12));[ "$SEC" -gt "$LIM" ] && SEC=$LIM
TURBODEC_PROMPT_MODULE=${TURBODEC_CODE_PROMPT_MODULE:-$SCRIPT_DIR/decode-study-inventory.py} timeout "$SEC" python3 "$SCRIPT_DIR/decode-study-client.py" "$PORT" "$OUT/c8-res.jsonl" C8 16 > "$OUT/c8.log" 2>&1
printf 'C8 diagnostic rc=%s budget=%ss\n' "$?" "$SEC" >> "$OUT/c8.log"
exit 0
