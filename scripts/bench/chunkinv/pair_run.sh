#!/usr/bin/env bash
# Chunk-invariance pair run: one two-rank strix-minimax-rank process pair, run from rank 0's node inside the caller's
# GPU turn. Records per rank: TTFT lines, state/tokens FNV, MemAvailable at launch, kfd evicted_ms (last sample of the
# rank process), busy / sclk / temperature samples. Hosts, binaries and model come from arguments only.
# usage: pair_run.sh --bin RANK_BIN --model DIR --peer SSH_HOST --out DIR [--secs N] [ENV=VAL ...] -- [rank args]
set -u -o pipefail
bin='' model='' peer='' out='' secs=600
envs=()
while [[ $# -gt 0 && $1 != -- ]]; do
    if [[ $1 == --* && $# -lt 2 ]]; then echo "missing value for $1" >&2; exit 2; fi
    case $1 in
        --bin) bin=$2; shift 2 ;; --model) model=$2; shift 2 ;; --peer) peer=$2; shift 2 ;;
        --out) out=$2; shift 2 ;; --secs) secs=$2; shift 2 ;;
        *=*) envs+=("$1"); shift ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done
[[ $# -gt 0 ]] || { echo 'missing -- before rank arguments' >&2; exit 2; }
shift
[[ -n $bin && -n $model && -n $peer && -n $out ]] || { sed -n 5p "$0" >&2; exit 2; }
[[ $peer != -* ]] || { echo 'invalid peer' >&2; exit 2; }
# SSH passes a command through the peer's login shell. POSIX single quotes preserve
# every byte (including newlines); only the explicitly selected bash parses scripts.
ssh_peer() {
    local command='' arg
    for arg in bash -c "$@"; do
        command+="'${arg//\'/\'\\\'\'}' "
    done
    ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$peer" "$command"
}
root=$out
mkdir -p -- "$root" || exit 2
root_abs=$(cd -- "$root" && pwd -P) || exit 2
out=$(mktemp -d "$root_abs/run.XXXXXXXX") || exit 2
run=${out##*/}
remote_out=$(ssh_peer 'mkdir -p -- "$1" && cd -- "$1" && mkdir -- "$2" && cd -- "$2" && pwd -P' -- "$root" "$run") || exit 2
echo "run directory: $out"
memavail='awk "/MemAvailable/{printf \"%.1f\", \$2/1048576}" /proc/meminfo'
mem0=$(bash -c "$memavail") || exit 1
mem1=$(ssh_peer "$memavail") || exit 1
[[ -n $mem0 && -n $mem1 ]] || { echo 'missing MemAvailable' >&2; exit 1; }
echo "memavail_gib r0 $mem0 r1 $mem1" > "$out/mem.txt" || exit 1
# Monitor: GPU busy/sclk/temp and the rank's kfd evicted_ms (summed over its GPUs), once a second.
mon='d=/sys/class/drm/card1/device; [ -d $d ] || d=/sys/class/drm/card0/device
while :; do ev=0; for p in $(pgrep -f "[s]trix-minimax-rank"); do for f in /sys/class/kfd/kfd/proc/$p/stats_*/evicted_ms; do
[ -r "$f" ] && ev=$((ev + $(cat "$f"))); done; done
echo "$(date +%s.%N) busy=$(cat $d/gpu_busy_percent) sclk=$(grep "\*" $d/pp_dpm_sclk | tr -d "\n") temp=$(cat $d/hwmon/hwmon*/temp1_input) evicted_ms=$ev"
sleep 1; done'
bash -c "$mon" > "$out/mon0.txt" 2>/dev/null & m0=$!
m1=''
cleanup() {
    kill "$m0" 2>/dev/null || :
    wait "$m0" 2>/dev/null || :
    if [[ $m1 =~ ^[0-9]+$ ]]; then
        ssh_peer 'kill -- "$1" 2>/dev/null || :' -- "$m1"
    fi
}
trap cleanup EXIT
m1=$(ssh_peer 'nohup bash -c "$2" > "$1/mon1.txt" 2>/dev/null < /dev/null & echo $!' -- "$remote_out" "$mon") || exit 1
[[ $m1 =~ ^[0-9]+$ ]] || { echo 'invalid monitor PID' >&2; exit 1; }
# Keep each host's launch directory: relative binary/model/prompt paths retain
# their original meanings. Only output paths are absolute.
ssh_peer 'log=$1; shift; exec "$@" > "$log" 2>&1' -- "$remote_out/r1.log" env "${envs[@]}" setsid timeout -k 2 "$secs" "$bin" "$model" 1 usb4-auto "$remote_out/r1" "$@" & peer_pid=$!
env "${envs[@]}" setsid timeout -k 2 "$secs" "$bin" "$model" 0 usb4-auto "$out/r0" "$@" > "$out/r0.log" 2>&1; rc0=$?
wait "$peer_pid"; rc1=$?
failed=0
cleanup || failed=1
trap - EXIT
for artifact in mon1.txt r1.log; do
    ssh_peer 'cat -- "$1"' -- "$remote_out/$artifact" > "$out/$artifact" || failed=1
done
[[ $rc0 == 0 && $rc1 == 0 ]] || failed=1
for r in 0 1; do
    [[ -s $out/mon$r.txt ]] || failed=1
    for kind in state tokens; do
        grep -a -q -E "$kind fnv [0-9a-f]+" "$out/r$r.log" || failed=1
    done
done
grep -a -q -E 'prompt [0-9]+ tokens, TTFT [0-9.]+ s, prefill [0-9.]+ tok/s' "$out/r0.log" || failed=1
ev() { grep -o 'evicted_ms=[0-9]*' "$1" 2>/dev/null | awk -F= '{ if ($2 + 0 > m) m = $2 + 0 } END { print m + 0 }'; }
{
    echo "rc $rc0/$rc1 $(cat "$out/mem.txt") evicted_ms r0 $(ev "$out/mon0.txt") r1 $(ev "$out/mon1.txt")"
    grep -a -h -o -E 'prompt [0-9]+ tokens, TTFT [0-9.]+ s, prefill [0-9.]+ tok/s' "$out/r0.log"
    grep -a -h -o -E '(state|tokens) fnv [0-9a-f]+' "$out/r0.log" "$out/r1.log" | sort | uniq -c
    for r in 0 1; do
        awk -v r=$r '{for(i=2;i<=NF;i++){split($i,a,"=");if(a[1]=="busy"){b+=a[2];n++} } for(i=2;i<=NF;i++) if($i ~ /^[0-9]+Mhz$/){s+=$i+0; break}}
            END{if(n)printf "r%d mean busy %.0f%% sclk %.0f MHz over %d s\n", r, b/n, s/n, n}' "$out/mon$r.txt"
    done
    echo "complete=$((1 - failed))"
} > "$out/summary.txt" || failed=1
cat "$out/summary.txt" || failed=1
[[ $failed == 0 ]] || echo 'pair run failed or required measurements are missing' >&2
exit "$failed"
