#!/usr/bin/env bash
# Qwen prefill length x mode x profile matrix (warm page cache, strict ABAB rounds, validity per run).
#
#   scripts/bench/common/ticket.sh OUT qwen/length-matrix --bin BIN --model MODEL --ids DIR \
#       [--sizes 8K,32K,128K,256K] [--modes full,pair,full2] [--arm 'NAME|ARGS|ENV' ...] [--rounds 3] [--gen 1]
#
#   BIN    strix-hip-decode-token          MODEL  the Qwen GGUF          DIR  p<N>.ids from common/prompt_ids.py
#   modes  full  = one node; pair = rank 0 here, rank 1 on KIT_PEER (same --bin/--model/--ids paths there unless
#                  --peer-bin/--peer-model/--peer-ids); full2 = the same single-node arm on both nodes at once
#                  (shared-host contention check)
#   arms   default 'exact||' and 'fast|--profile fast --allow-pending|'; ARGS are extra CLI words, ENV are
#          space-separated NAME=VALUE engine levers for that arm only (inherited STRIX_* are dropped).
#   --warm pagecache (default: read the model into the page cache first, cold-disk time is not prefill time) | none
#   Long context (YaRN, 262K-1M): add --context N to the arm ARGS and sizes up to 1024K, e.g.
#          --sizes 256K,512K,1024K --arm 'yarn|--context 1048576|'
# Each run: prefill wall ms (cold = first, warm = last prefill in the process), tok/s, GPU clock, busy, watts,
# GPU clocks per token, logits sha (STRIX_BITCALC_LOGITS_OUT), kfd evicted_ms, MemAvailable minimum, klog events.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/../common/runlib.sh"
bin='' model='' ids='' sizes=8K modes=full rounds=3 gen=1 arms=() pbin='' pmodel='' pids='' need_override='' warm=pagecache
pair_env=${QWEN_PAIR_ENV:-STRIX_QWEN_PF_PIPE=1 STRIX_QWEN_PF_PIPE_DECODE=1 STRIX_DIST_USB4_TIMEOUT_MS=120000 STRIX_DIST_WAIT_TIMEOUT_MS=120000}
transport=${QWEN_PAIR_TRANSPORT:-usb4auto}
while (($#)); do
    case $1 in
        --bin) bin=$2 ;; --model) model=$2 ;; --ids) ids=$2 ;; --sizes) sizes=$2 ;; --modes) modes=$2 ;;
        --rounds) rounds=$2 ;; --gen) gen=$2 ;; --arm) arms+=("$2") ;; --need) need_override=$2 ;;
        --peer-bin) pbin=$2 ;; --peer-model) pmodel=$2 ;; --peer-ids) pids=$2 ;; --transport) transport=$2 ;; --warm) warm=$2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) kit_die "unknown argument $1" ;;
    esac
    shift 2
done
[[ -x $bin && -f $model && -d $ids ]] || kit_die 'need --bin (executable), --model (file) and --ids (dir)'
((${#arms[@]})) || arms=('exact||' 'fast|--profile fast --allow-pending|')
pbin=${pbin:-$bin} pmodel=${pmodel:-$model} pids=${pids:-$ids}
out=${KIT_OUT:-bench-out}; mkdir -p -- "$out"

tokens_of() { local s=${1^^}; [[ $s == *K ]] && echo $((${s%K} * 1024)) || echo "$s"; }
# Seconds an arm needs (8K..256K, measured on the reference pair; --need overrides for every arm).
need_of() {
    [[ -n $need_override ]] && { echo "$need_override"; return; }
    local n=$1 m=$2
    if ((n <= 8192)); then [[ $m == pair ]] && echo 28 || echo 18
    elif ((n <= 32768)); then [[ $m == pair ]] && echo 42 || echo 34
    elif ((n <= 131072)); then [[ $m == pair ]] && echo 72 || echo 84
    elif ((n <= 262144)); then [[ $m == pair ]] && echo 150 || echo 200
    else echo 400; fi
}

if [[ $modes == *pair* || $modes == *full2* ]]; then
    echo "peer mem [$(kit_peer "python3 -I $(printf %q "$(kit_peer_root)")/scripts/bench/common/meminfo.py")]"
    kit_same_binary "$bin" "$pbin"
    [[ $warm == pagecache ]] && kit_peer "bash -c $(printf %q "$(declare -f kit_warm_pagecache kit_log); kit_warm_pagecache $(printf %q "$pmodel")")" &
fi
[[ $warm == pagecache ]] && kit_warm_pagecache "$model"; wait
echo "bin $(kit_sha "$bin") model $(basename "$model") rounds $rounds"

IFS=, read -ra S <<<"$sizes"; IFS=, read -ra M <<<"$modes"
for ((r = 0; r < rounds; r++)); do
    for s in "${S[@]}"; do
        n=$(tokens_of "$s"); f=$ids/p$n.ids; pf=$pids/p$n.ids
        [[ -s $f ]] || { echo "== s$s SKIP: no $f"; continue; }
        for m in "${M[@]}"; do
            for a in "${arms[@]}"; do
                IFS='|' read -r name args envs <<<"$a"
                read -ra A <<<"$args"; read -ra E <<<"$envs"
                tag=s$s-$m-$name-$r need=$(need_of "$n" "$m")
                kit_fits "$need" || { echo "== $tag SKIP (left $(kit_left) s < $need)"; continue; }
                cap=$(($(kit_left) - 3))
                export KIT_TAG=$tag KIT_ARM=s$s-$m-$name KIT_ROUND=$r KIT_TOKENS=$n KIT_WORKLOAD=s$s-$m KIT_RECEIPT_ARGS='--need-prefill'
                # Single-node runs print no prefill line: the figure is the whole request (prefill + --gen tokens +
                # readback) from the result JSON, published as a prefill upper bound. Arrival stamps give only
                # conservative envelopes including earlier samples. Require request validity for all metrics
                # derived from its duration, including prefill bounds and prompt tokens per request second.
                if [[ $m == full ]]; then
                    KIT_RECEIPT_ARGS+=' --request-json generation_request_elapsed_ms --anchor-regex ^qpf2k.sched.block.rows --scope request'
                fi
                echo "#### $tag args[${args}] env[${envs}] t+$(($(date +%s) - KIT_T0))s"
                case $m in
                    full)
                        bash "$here/../common/run1.sh" "$out/$tag" "$cap" -- env "${E[@]}" STRIX_BITCALC_LOGITS_OUT="$out/$tag/logits.f32" \
                            "$bin" "$model" --prompt-ids "$f" --generate "$gen" "${A[@]}" ;;
                    full2)
                        pcmd=(env "${E[@]}" STRIX_BITCALC_LOGITS_OUT="${KIT_PEER_OUT:-$out}/$tag-peer/logits.f32" "$pbin" "$pmodel" --prompt-ids "$pf" --generate "$gen" "${A[@]}")
                        kit_peer "cd $(printf %q "$(kit_peer_root)") && KIT_TAG=$tag-peer KIT_ARM=$KIT_ARM@peer KIT_ROUND=$r KIT_TOKENS=$n KIT_HOST_TAG=peer bash scripts/bench/common/run1.sh $(printf %q "${KIT_PEER_OUT:-$out}/$tag-peer") $cap -- $(printf '%q ' "${pcmd[@]}")" >"$out/$tag-peer.log" 2>&1 &
                        bash "$here/../common/run1.sh" "$out/$tag" "$cap" -- env "${E[@]}" STRIX_BITCALC_LOGITS_OUT="$out/$tag/logits.f32" \
                            "$bin" "$model" --prompt-ids "$f" --generate "$gen" "${A[@]}"
                        wait; mkdir -p -- "$out/$tag-peer"
                        kit_peer "cat $(printf %q "${KIT_PEER_OUT:-$out}/$tag-peer/receipt.json")" >"$out/$tag-peer/receipt.json" 2>/dev/null
                        sed 's/^/  peer| /' "$out/$tag-peer.log" | grep -a '^  peer| ==' ;;
                    pair)
                        read -ra PE <<<"$pair_env"
                        TR=(); [[ $transport != def ]] && TR=(--transport "$transport")
                        p1=(env "${PE[@]}" "${E[@]}" "$pbin" "$pmodel" --prompt-ids "$pf" --generate "$gen" "${A[@]}" --rank 1 "${TR[@]}")
                        p0=(env "${PE[@]}" "${E[@]}" STRIX_BITCALC_LOGITS_OUT="$out/$tag-r0/logits.f32" "$bin" "$model" --prompt-ids "$f" --generate "$gen" "${A[@]}" --rank 0 "${TR[@]}")
                        kit_pair_arm "$out/$tag" "$cap" 'qwen-pair rank=1|^strix-hip-decode-token: |what\(\)' p1 p0 ;;
                    *) echo "== $tag SKIP: unknown mode $m" ;;
                esac
                sleep "${KIT_SETTLE_S:-2}"
            done
        done
    done
done
echo "length-matrix done $(($(date +%s) - KIT_T0)) s"
