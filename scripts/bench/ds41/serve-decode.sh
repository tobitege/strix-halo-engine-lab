#!/usr/bin/env bash
# DeepSeek V4.1 pair: served decode by class (prose / edit / json) at 1/4/8 concurrent chats, one arm per call.
#
#   scripts/bench/common/ticket.sh OUT ds41/serve-decode --bin-dir DIR --model MODEL [--arm 'NAME|ARGS|ENV']
#       [--round N] [--port 18741] [--slots 8] [--concurrency 1,4,8] [--max-tokens 128] [--classes prose,edit,json]
#
# Rank 0 serves HTTP here, rank 1 runs on KIT_PEER (same --bin-dir/--model paths unless --peer-bin-dir/--peer-model).
# One server start per call fits one GPU ticket; strict ABAB = queue tickets A0 B0 A1 B1 A2 B2 with the SAME OUT:
# every ticket's summary covers all receipts in OUT (abab.py pairs arms by --round).
# Example arms: 'ctl||'  'plain|--no-dspark|'  'rows8||STRIX_DS41_COPY_ROWS=8'.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/../common/runlib.sh"
bdir='' model='' arm='ctl||' round=0 port=18741 slots=8 conc=1,4,8 maxt=128 classes=prose,edit,json pbdir='' pmodel=''
base_env=${DS41_SERVE_ENV:-STRIX_SERVE_RECEIPT=1 STRIX_DIST_USB4_TIMEOUT_MS=120000 STRIX_DIST_WAIT_TIMEOUT_MS=120000}
while (($#)); do
    case $1 in
        --bin-dir) bdir=$2 ;; --model) model=$2 ;; --arm) arm=$2 ;; --round) round=$2 ;; --port) port=$2 ;;
        --slots) slots=$2 ;; --concurrency) conc=$2 ;; --max-tokens) maxt=$2 ;; --classes) classes=$2 ;;
        --peer-bin-dir) pbdir=$2 ;; --peer-model) pmodel=$2 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) kit_die "unknown argument $1" ;;
    esac
    shift 2
done
bin=$bdir/strix-deepseek41-serve
[[ -x $bin && -e $model ]] || kit_die 'need --bin-dir (with strix-deepseek41-serve) and --model'
pbin=${pbdir:-$bdir}/strix-deepseek41-serve pmodel=${pmodel:-$model}
kit_same_binary "$bin" "$pbin"
IFS='|' read -r name args envs <<<"$arm"
read -ra A <<<"$args"; read -ra E <<<"$envs"; read -ra B <<<"$base_env"
common=(--transport usb4auto --peer-timeout 120 --prefill-path device --slots "$slots" --prefill-rows auto
    --http-port "$port" --http-bind 127.0.0.1)
peer=$(printf '%q ' env "${B[@]}" "${E[@]}" "$pbin" "$pmodel" --rank 1 "${common[@]}" "${A[@]}")
bash "$here/../common/serve-decode.sh" --tag "$name-$round" --arm "$name" --round "$round" --port "$port" --chat \
    --body-extra '{"enable_thinking":false}' --ready-log '"READY"' --classes "$classes" --concurrency "$conc" \
    --max-tokens "$maxt" --peer-cmd "$peer" -- env "${B[@]}" "${E[@]}" "$bin" "$model" --rank 0 "${common[@]}" "${A[@]}"
