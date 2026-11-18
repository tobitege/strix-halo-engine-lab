#!/usr/bin/env bash
# Qwen served decode by class (prose / edit / json) at 1/4/8 concurrent chats on one node, one arm per call.
#
#   scripts/bench/common/ticket.sh OUT qwen/serve-decode --bin BIN --model MODEL [--arm 'NAME|ARGS|ENV'] [--round N]
#       [--port 18671] [--context 33280] [--batch 4] [--concurrency 1,4,8] [--max-tokens 128] [--ids 'STARTUP TOKEN IDS']
#
#   BIN  strix-hip-decode-token (served with --http-serve; MTP sidecar / draft vocab are found as for the CLI or
#        given in the arm ARGS). Example arms: 'def||'  'wide|--mtp-ngram-draft=wide --mtp-copy-wide-rows 32,16|'.
# Strict ABAB across tickets: same OUT, --round 0 A, 0 B, 1 A, 1 B, ...; abab.py summary OUT --metric e2e_C8.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/../common/runlib.sh"
bin='' model='' arm='def||' round=0 port=18671 ctx=33280 batch=4 conc=1,4,8 maxt=128 classes=prose,edit,json
# The server takes a positional startup prompt (predictor qualification); default: the chat header tokens.
ids='248045 846 198'
while (($#)); do
    case $1 in
        --bin) bin=$2 ;; --model) model=$2 ;; --arm) arm=$2 ;; --round) round=$2 ;; --port) port=$2 ;; --context) ctx=$2 ;;
        --batch) batch=$2 ;; --concurrency) conc=$2 ;; --max-tokens) maxt=$2 ;; --classes) classes=$2 ;; --ids) ids=$2 ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) kit_die "unknown argument $1" ;;
    esac
    shift 2
done
[[ -x $bin && -e $model ]] || kit_die 'need --bin and --model'
IFS='|' read -r name args envs <<<"$arm"; read -ra A <<<"$args"; read -ra E <<<"$envs"; read -ra I <<<"$ids"
bash "$here/../common/serve-decode.sh" --tag "$name-$round" --arm "$name" --round "$round" --port "$port" --chat \
    --body-extra '{"enable_thinking":false}' --ready-log 'listening on' --classes "$classes" --concurrency "$conc" \
    --max-tokens "$maxt" -- env STRIX_SERVE_RECEIPT=1 "${E[@]}" "$bin" "$model" "${I[@]}" --context "$ctx" --generate 1 \
    --http-default-max-tokens "$maxt" --http-serve --http-port "$port" --http-bind 127.0.0.1 --http-batch "$batch" "${A[@]}"
