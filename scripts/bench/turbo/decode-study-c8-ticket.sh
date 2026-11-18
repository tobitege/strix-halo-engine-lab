#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
K=${TURBODEC_ROOT:?Set TURBODEC_ROOT to the study runner and output directory}
export TURBODEC_PROMPT_MODULE=${TURBODEC_PROMPT_MODULE:-${TURBODEC_CODE_PROMPT_MODULE:-$SCRIPT_DIR/decode-study-inventory.py}}
if [ "${TURBODEC_LIST_COUNTERS:-}" = E ]; then /opt/rocm/bin/rocprofv3 --list-avail > "$K/counters-avail.txt" 2>&1; fi
exec bash "${TURBODEC_TICKET_RUNNER:-$SCRIPT_DIR/decode-study-ticket.sh}" "$1" v2 C8,C8
