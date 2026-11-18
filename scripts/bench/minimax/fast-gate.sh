#!/usr/bin/env bash
# MiniMax FAST quality gate: teacher-forced logit dumps of a reference arm and candidate,
# on the same binary and prompts; the admitted rule is absolute dNLL.
#
#   GPU (one arm per ticket):
#     ticket.sh OUT minimax/fast-gate run --bin BIN --model MODEL_DIR --arm 'NAME|ARGS|ENV' --sessions a=A.u32,b=B.u32
#         [--capacity 34816] [--rows 256]
#     --rows 0 = timing route: same run without the quality dump env (no logit rows are captured inside the
#     timed prefill); seal/compare need a run with rows > 0. Both ranks' inputs are checked before launch.
#   CPU (after the tickets, no GPU):
#     fast-gate.sh seal    --out OUT --arm NAME --bin BIN --model MODEL_DIR --sessions ...   (both ranks)
#     fast-gate.sh compare --out OUT --ref NAME --cand NAME [--dnll-rule abs] [gate args...]
#
# run: strix-minimax-rank on both ranks (rank 1 on KIT_PEER) with STRIX_SESSION_CACHE=0 STRIX_MM_STATE_FNV=1 and
# the quality dump env (STRIX_MM_QUALITY_OUT/ROWS/SELECT); sessions run in order in one process (first = cold,
# last = warm, "rank 0 <name>: prompt N tokens, TTFT X s"). A run directory is never reused.
# seal/compare uses tools/strix-gate/fast_logit_gate.py absolute KL, top-1 and dNLL limits.
# MiniMax has no admitted floor policy or null-control witnesses.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/../common/runlib.sh"
gate=$KIT_ROOT/tools/strix-gate/fast_logit_gate.py
mode=${1:-help}; shift || true
bin='' model='' arm='' sessions='' cap=34816 rows=256 out=${KIT_OUT:-bench-out} ref='' cand='' nulls=() rest=() rule=abs
while (($#)); do
    case $1 in
        --bin) bin=$2 ;; --model) model=$2 ;; --arm) arm=$2 ;; --sessions) sessions=$2 ;; --capacity) cap=$2 ;;
        --rows) rows=$2 ;; --out) out=$2 ;; --ref) ref=$2 ;; --cand) cand=$2 ;; --null) nulls+=("$2") ;;
        --dnll-rule) rule=$2 ;;
        --engine|--profile|--ranks|--null-arm|--prefix) kit_die "$1 is owned by the MiniMax wrapper" ;;
        *) rest+=("$1"); shift; continue ;;
    esac
    shift 2
done
IFS='|' read -r name args envs <<<"${arm:-x||}"
read -ra A <<<"$args"; read -ra E <<<"$envs"
SA=() PR=()
IFS=, read -ra SS <<<"$sessions"
for s in "${SS[@]}"; do SA+=(--session "${s%%=*}:${s#*=}:1"); PR+=(--prompt "${s%%=*}=${s#*=}"); done
case $mode in
run)
    [[ -x $bin && -e $model && ${#SS[@]} -gt 0 && -n $name ]] || kit_die 'run needs --bin, --model, --sessions and --arm'
    [[ $rows =~ ^[0-9]+$ ]] || kit_die "--rows must be a count (0 = timing route without quality dumps)"
    kit_same_binary "$bin"
    kit_peer_inputs "$model" "${SS[@]}" "${A[@]}"
    d=$out/$name
    [[ -e $d-r0 ]] && kit_die "run dir $d-r0 exists (never reused)"
    peer_out=${KIT_PEER_OUT:-$out}
    q=(STRIX_SESSION_CACHE=0 STRIX_MM_STATE_FNV=1) q1=() q0=()
    if ((rows > 0)); then
        q+=(STRIX_MM_QUALITY_ROWS=$rows STRIX_MM_QUALITY_SELECT=1)
        q1=(STRIX_MM_QUALITY_OUT="$peer_out/$name-r1/q") q0=(STRIX_MM_QUALITY_OUT="$d-r0/q")
    fi
    p1=(env "${q[@]}" "${q1[@]}" "${E[@]}" "$bin" "$model" 1 usb4-auto "$peer_out/$name-r1/w"
        --slots 1 --capacity "$cap" --max-rows 4096 "${SA[@]}" "${A[@]}")
    p0=(env "${q[@]}" "${q0[@]}" "${E[@]}" "$bin" "$model" 0 usb4-auto "$d-r0/w"
        --slots 1 --capacity "$cap" --max-rows 4096 "${SA[@]}" "${A[@]}")
    export KIT_TAG=$name KIT_ARM=$name KIT_ROUND=0 KIT_TOKENS=0 KIT_EXPECTED=0
    export KIT_RECEIPT_ARGS='--need-prefill --family minimax --require ranks.AGREE'
    printf '%s\n' "${E[@]}" >"$out/$name.env"
    kit_pair_arm "$d" "$(($(kit_left) - 3))" 'rank 1|what\(\)' p1 p0 ;;
seal)
    [[ -n $name && -n $bin ]] || kit_die 'seal needs --arm and --bin'
    EA=(); while read -r e; do [[ -n $e ]] && EA+=(--env "$e"); done <"$out/$name.env"
    for rk in 0 1; do
        dir=$out/$name-r$rk; ((rk == 1)) && dir=${KIT_PEER_OUT:-$out}/$name-r1
        cmd=("$KIT_PY" -I "$gate" seal --dir "$dir" --rank "$rk" --arm "$name" --run-id "$name" --binary "$bin" --rows "$rows" "${PR[@]}" "${EA[@]}" "${rest[@]}")
        if ((rk == 0)); then "${cmd[@]}"; else kit_peer "cd $(printf %q "$(kit_peer_root)") && $(printf '%q ' "${cmd[@]}")"; fi || kit_die "seal rank $rk failed"
    done
    # Assemble both ranks under one directory; manifests bind each copied dump by hash.
    mkdir -p -- "$out/$name-sealed"
    cp -- "$out/$name-r0/manifest-r0.json" "$out/$name-r0/"q-*-r0.* "$out/$name-sealed/" || kit_die 'copy rank 0 failed'
    (set -o pipefail; kit_peer "cd $(printf %q "${KIT_PEER_OUT:-$out}/$name-r1") && tar -cf - -- manifest-r1.json q-*-r1.*" |
        tar -C "$out/$name-sealed" -xf -) || kit_die 'copy rank 1 failed'
    [[ -s $out/$name-sealed/manifest-r1.json ]] || kit_die 'missing rank 1 manifest'
    echo "sealed $name" ;;
compare)
    [[ -n $ref && -n $cand ]] || kit_die 'compare needs --ref and --cand'
    [[ $rule == abs && ${#nulls[@]} == 0 ]] || kit_die 'MiniMax supports --dnll-rule abs only; null controls are not admitted'
    "$KIT_PY" -I "$gate" compare "$out/$ref-sealed" "$out/$cand-sealed" \
        "${PR[@]/#--prompt/--tokens}" "${rest[@]}" --ranks 0 1 --dnll-rule abs ;;
*) sed -n '2,16p' "$0"; exit 64 ;;
esac
