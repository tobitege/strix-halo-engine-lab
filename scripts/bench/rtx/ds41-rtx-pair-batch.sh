#!/usr/bin/env bash
# DeepSeek V4.1 RTX + pair batched decode (C2 / C4 / C8): the RTX runs layers [0, cut) of every request on
# strix-cuda-ds41-multi-front, the two Strix Halo nodes run [cut, 40) and the head on strix-deepseek41-batch,
# requests split into groups so the RTX computes one group while the pair steps the other
# (docs/discrete-gpu.md "DeepSeek V4.1 Flash, RTX + pair"; docs/models/deepseek-v4.1/pipeline-stages.md).
#
# 1. TAIL, on the rank-0 node inside a GPU ticket (rank 1 on KIT_PEER):
#      ticket.sh OUT rtx/ds41-rtx-pair-batch tail --bin BIN --model MODEL --prompts a.csv,...,h.csv --token-file TOK
#          [--listen 0.0.0.0:7841] [--cut 22] [--groups 2] [--n 2,4,8] [--generate 48] [--mode both|batch] [--arm ...]
# 2. FRONT, on the RTX host, started right after the ticket starts (it waits for the TAIL with --connect-wait):
#      ds41-rtx-pair-batch.sh front --bin strix-cuda-ds41-multi-front --model MODEL --stage NODE:7841 --token-file TOK
#          [--cut 22] [--slots 8] [--connect-wait 120] [--json OUT.json] [-- extra FRONT args, e.g. --expert-tp --overlay OV]
#
#   prompts: comma-separated token ids per request (common/prompt_ids.py --format csv, or strix-tokenize ... encode of
#            the 8 kit prompts in scripts/bench/prompts/kit.json); TOK: 64 hex characters (the session secret), the
#            same file on both sides; keep the stage port on a private link (it is unauthenticated beyond TOK).
#   --mode both: every request alone on the pair first (reference), then batched through the split; "GATE ... PASS"
#            means every split, batched request equals its pair-only run. --mode batch: timing only.
# Receipt metrics: C<n> = "decode window" aggregate tok/s of N = n (the published figure), plus the GATE lines.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/../common/runlib.sh"
role=${1:-help}; shift || true
bin='' model='' prompts='' tok='' listen=0.0.0.0:7841 cut=22 groups=2 n=2,4,8 gen=48 mode=both arm='split||' stage=''
slots=8 wait_s=120 json='' extra=() pbin='' pmodel='' pprompts=''
while (($#)); do
    case $1 in
        --bin) bin=$2 ;; --model) model=$2 ;; --prompts) prompts=$2 ;; --token-file) tok=$2 ;; --listen) listen=$2 ;;
        --cut) cut=$2 ;; --groups) groups=$2 ;; --n) n=$2 ;; --generate) gen=$2 ;; --mode) mode=$2 ;; --arm) arm=$2 ;;
        --stage) stage=$2 ;; --slots) slots=$2 ;; --connect-wait) wait_s=$2 ;; --json) json=$2 ;;
        --peer-bin) pbin=$2 ;; --peer-model) pmodel=$2 ;; --peer-prompts) pprompts=$2 ;;
        --) shift; extra=("$@"); break ;;
        *) kit_die "unknown argument $1" ;;
    esac
    shift 2
done
case $role in
tail)
    [[ -x $bin && -e $model && -n $prompts && -f $tok ]] || kit_die 'tail needs --bin, --model, --prompts and --token-file'
    pbin=${pbin:-$bin} pmodel=${pmodel:-$model} pprompts=${pprompts:-$prompts}
    kit_same_binary "$bin" "$pbin"
    IFS='|' read -r name args envs <<<"$arm"; read -ra A <<<"$args"; read -ra E <<<"$envs"
    B=(STRIX_DIST_USB4_TIMEOUT_MS=120000 STRIX_DIST_WAIT_TIMEOUT_MS=120000)
    run=(--transport usb4auto --n "$n" --generate "$gen" --mode "$mode")
    p1=(env "${B[@]}" "${E[@]}" "$pbin" "$pmodel" --rank 1 --prompts "$pprompts" "${run[@]}" "${A[@]}")
    p0=(env "${B[@]}" "${E[@]}" "$bin" "$model" --rank 0 --prompts "$prompts" "${run[@]}" --hetero-listen "$listen"
        --hetero-token-file "$tok" --hetero-cut "$cut" --hetero-groups "$groups" "${A[@]}")
    M=(); IFS=, read -ra NS <<<"$n"
    for k in "${NS[@]}"; do M+=(--metric "C$k=^N.$k.hetero.cut.*decode.window.([0-9.]+).tok/s"); done
    export KIT_TAG=$name KIT_ARM=$name KIT_ROUND=${KIT_ROUND:-0} KIT_TOKENS=0 KIT_RECEIPT_ARGS="--evict-whole-run ${M[*]}"
    kit_pair_arm "${KIT_OUT:-bench-out}/$name" "$(($(kit_left) - 3))" 'READY|rank 1|what\(\)' p1 p0
    grep -aE '^GATE|^N [0-9]+ hetero' "${KIT_OUT:-bench-out}/$name-r0/stderr" "${KIT_OUT:-bench-out}/$name-r0/stdout" 2>/dev/null |
        cut -d: -f2- | cut -c1-200 | head -12 ;;
front)
    [[ -n $bin && -e $model && -n $stage && -f $tok ]] || kit_die 'front needs --bin, --model, --stage and --token-file'
    J=(); [[ -n $json ]] && J=(--json "$json")
    # CUDA_VISIBLE_DEVICES / STRIX_DEVICES select the RTX (set them in the environment of this call).
    exec "$bin" "$model" --stage "$stage" --cut "$cut" --slots "$slots" --token-file "$tok" --connect-wait "$wait_s" \
        --sessions 1 "${J[@]}" "${extra[@]}" ;;
*) sed -n '2,22p' "$0"; exit 64 ;;
esac
