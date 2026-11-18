#!/usr/bin/env bash
# MiniMax M3.x pair: served decode by class (prose / edit / json) at 1/8 concurrent requests, one arm per call,
# and the server start time (the per-rank "startup:" phase lines are copied to OUT/<arm>-<round>/startup.txt).
#
#   scripts/bench/common/ticket.sh OUT minimax/serve-decode --bin BIN --model DIR [--arm 'NAME|ARGS|ENV']
#       [--round N] [--port 18781] [--slots 8] [--capacity 2048] [--concurrency 1,8] [--max-tokens 128]
#       [--classes prose] [--start-only] [--chat]
#
# BIN is strix-minimax-serve. Rank 0 serves HTTP here, rank 1 runs on KIT_PEER (same --bin/--model paths unless
# --peer-bin/--peer-model). Ready = the server's own "READY rank 0" line, so the measured start needs no HTTP client.
# --start-only stops at READY (start-time diagnosis; fits a short ticket). One server start per call fits one GPU
# ticket; strict ABAB = tickets A0 B0 A1 B1 with the SAME OUT (abab.py pairs arms by --round).
# Requests ask for the generated token ids (return_token_ids), so two arms compare with serve_bench.py --compare.
# Example arms: 'ctl||'  'serial||STRIX_SERVE_STARTUP_OVERLAP=0'.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/../common/runlib.sh"
bin='' model='' arm='ctl||' round=0 port=18781 slots=8 capacity=2048 conc=1,8 maxt=128 classes=prose pbin='' pmodel=''
start_only=0 chat=()
base_env=${MINIMAX_SERVE_ENV:-STRIX_SERVE_RECEIPT=1 HSA_DISABLE_FRAGMENT_ALLOCATOR=1 STRIX_DIST_USB4_TIMEOUT_MS=120000 STRIX_DIST_WAIT_TIMEOUT_MS=120000}
while (($#)); do
    case $1 in
        --bin) bin=$2 ;; --model) model=$2 ;; --arm) arm=$2 ;; --round) round=$2 ;; --port) port=$2 ;;
        --slots) slots=$2 ;; --capacity) capacity=$2 ;; --concurrency) conc=$2 ;; --max-tokens) maxt=$2 ;;
        --classes) classes=$2 ;; --peer-bin) pbin=$2 ;; --peer-model) pmodel=$2 ;;
        --start-only) start_only=1; shift; continue ;;
        --chat) chat=(--chat); shift; continue ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) kit_die "unknown argument $1" ;;
    esac
    shift 2
done
[[ -x $bin && -e $model ]] || kit_die 'need --bin (strix-minimax-serve) and --model'
pbin=${pbin:-$bin} pmodel=${pmodel:-$model}
kit_same_binary "$bin" "$pbin"
IFS='|' read -r name args envs <<<"$arm"
read -ra A <<<"$args"; read -ra E <<<"$envs"; read -ra B <<<"$base_env"
common=(--transport usb4-auto --slots "$slots" --capacity "$capacity")
peer=$(printf '%q ' env "${B[@]}" "${E[@]}" "$pbin" "$pmodel" "${common[@]}" "${A[@]}" --rank 1)
client=()
((start_only)) && client=(--client-cmd 'printf "{\"completed\":1,\"failed\":0}\n" >"$KIT_RUN_DIR/client-result.json"')
tag=$name-$round
bash "$here/../common/serve-decode.sh" --tag "$tag" --arm "$name" --round "$round" --port "$port" "${chat[@]}" \
    --ready-log '^READY rank 0' --body-extra '{"return_token_ids":true}' --classes "$classes" --concurrency "$conc" --max-tokens "$maxt" "${client[@]}" \
    --peer-cmd "$peer" -- env "${B[@]}" "${E[@]}" "$bin" "$model" "${common[@]}" "${A[@]}" --rank 0 --bind 127.0.0.1 --port "$port"
rc=$?
d=${KIT_OUT:-bench-out}/$tag
{
    grep -ahE '^rank [01] startup:|^READY rank 0' "$d/server/stdout" 2>/dev/null
    if [[ -n ${KIT_PEER:-} ]]; then
        timeout -k 1 5 ssh -n -o BatchMode=yes -o ConnectTimeout=2 "$KIT_PEER" \
            "grep -ahE '^rank [01] startup:' $(printf %q "${KIT_PEER_OUT:-${KIT_OUT:-bench-out}}/$tag-r1/stdout")" 2>/dev/null
    fi
} >"$d/startup.txt" 2>/dev/null
sed 's/^/  /' "$d/startup.txt" 2>/dev/null
exit "$rc"
