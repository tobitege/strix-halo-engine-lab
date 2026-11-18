#!/usr/bin/env bash
# ticket.sh OUT KIT [KIT ARGS...]: the one command a GPU queue ticket runs.
#   <queue>/gpu-turn.sh LABEL  <repo>/scripts/bench/common/ticket.sh OUT qwen/length-matrix --bin B --model M ...
# KIT is a path under scripts/bench/ without ".sh". The ticket budget is GPU_TURN_MAX (the queue's cap, default
# 120 s) minus KIT_MARGIN (default 5 s); kits skip arms that no longer fit instead of overrunning the ticket.
# Ends by writing OUT/summary.txt (<= 60 lines, from the receipts) and OUT/.done, so a caller waits once on the
# marker and reads only the summary. Never builds: build first (in its own command), then queue the ticket.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ $# -ge 2 ]] || { echo 'usage: ticket.sh OUT KIT [ARGS...]' >&2; exit 64; }
out=$1 kit=$2; shift 2
script=$here/../$kit.sh
[[ -f $script ]] || { echo "ticket: no kit scripts/bench/$kit.sh" >&2; exit 64; }
mkdir -p -- "$out"; rm -f -- "$out/.done"
export KIT_OUT=$out KIT_T0=$(date +%s)
export KIT_BUDGET=${KIT_BUDGET:-$(( ${GPU_TURN_MAX:-120} - ${KIT_MARGIN:-5} ))}
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/runlib.sh"
{
    echo "ticket $(date '+%F %T') kit=$kit budget=${KIT_BUDGET}s mem[$(kit_mem)]"
    bash "$script" "$@"
    echo "kit rc=$? after $(( $(date +%s) - KIT_T0 )) s"
} >"$out/ticket.log" 2>&1
kit_summary_done "$out"
