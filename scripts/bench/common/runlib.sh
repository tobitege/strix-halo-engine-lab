#!/usr/bin/env bash
# Shared runner library for the measurement kits under scripts/bench/<area>/ (sourced, not executed).
#
# Every kit run is one directory holding: cmd.txt, stdout, stderr (+ stderr.ts: arrival time of every line),
# samples.tsv (GPU power/clock/temp/busy, GTT, MemAvailable, MemFree, kfd evicted_ms of the run's process tree),
# klog.txt (kernel events inside the run window), meta.json (rc, wall, binary sha, host memory at start) and,
# after kit_receipt, receipt.json + one summary line. Validity, warm/cold and ABAB rules live in receipt.py /
# abab.py so every family reports them the same way.
#
# No host, user or model path is built in: peers come from KIT_PEER (ssh alias), binaries and models from the
# kit's arguments. Environment knobs (all optional):
#   KIT_OUT          output root (default ./bench-out)            KIT_BUDGET   seconds for the whole ticket
#   KIT_PEER         ssh alias of the second node for pair arms   KIT_PEER_ROOT repo checkout on the peer (default: same path)
#   KIT_SAMPLE_S     sampler period (default 0.2)                 KIT_KEEP_ENV  space-separated STRIX_* names kept by kit_clean_env
#   KIT_PYTHON       python interpreter (default python3)
# shellcheck disable=SC2034
KIT_LIB=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
KIT_ROOT=$(cd -- "$KIT_LIB/../../.." && pwd)
KIT_PY=${KIT_PYTHON:-python3}
KIT_T0=${KIT_T0:-$(date +%s)}
export KIT_T0

kit_log() { printf '%s %s\n' "$(date +%T)" "$*" >&2; }
kit_die() { printf 'kit: %s\n' "$*" >&2; exit 2; }
kit_py() { "$KIT_PY" -I "$KIT_LIB/$1" "${@:2}"; }

# Seconds left in the ticket budget (KIT_BUDGET unset = unlimited, prints 999999).
kit_left() {
    if [[ -z ${KIT_BUDGET:-} ]]; then echo 999999; return; fi
    echo $((KIT_BUDGET - ($(date +%s) - KIT_T0)))
}
# kit_fits NEED: true when at least NEED seconds (+3 s margin) remain.
kit_fits() { (($(kit_left) > $1 + 3)); }

# Drop inherited engine levers; device binding is infrastructure, not an arm's tuning.
# KIT_KEEP_ENV lists additional exceptions.
kit_clean_env() {
    local v keep=" STRIX_DEVICES STRIX_DEVICE ${KIT_KEEP_ENV:-} "
    for v in $(env | sed -n 's/^\(STRIX_[A-Z0-9_]*\)=.*/\1/p'); do
        [[ $keep == *" $v "* ]] || unset "$v"
    done
}

kit_mem() { kit_py meminfo.py; }
kit_sha() { sha256sum "$1" | cut -c1-16; }

# kit_peer CMD...: run a command on KIT_PEER (non-interactive, never reads stdin).
kit_peer() {
    [[ -n ${KIT_PEER:-} ]] || kit_die 'KIT_PEER is not set (ssh alias of the second node)'
    ssh -n -o BatchMode=yes -o ServerAliveInterval=15 "$KIT_PEER" "$@"
}
kit_peer_root() { echo "${KIT_PEER_ROOT:-$KIT_ROOT}"; }

# kit_peer_inputs ARG...: fail fast, before any GPU time, when the peer rank cannot read an input the local rank
# reads. Each ARG is a path, NAME=PATH, or a ':'-separated spec (e.g. d:/p/x.u32:128); every absolute field that
# exists locally must be readable on KIT_PEER under the same path. A missing input names every missing path.
kit_peer_inputs() {
    local a f miss need=()
    for a in "$@"; do
        IFS=: read -ra F <<<"${a#*=}"
        for f in "${F[@]}"; do [[ $f == /* && -e $f ]] && need+=("$f"); done
    done
    ((${#need[@]})) || return 0
    miss=$(kit_peer "for f in $(printf '%q ' "${need[@]}"); do [ -r \"\$f\" ] || echo \"\$f\"; done") ||
        kit_die 'peer input check failed (peer unreachable)'
    [[ -z $miss ]] || kit_die "peer $KIT_PEER is missing inputs: $(echo $miss)"
    kit_log "peer inputs ok (${#need[@]} paths)"
}

# kit_same_binary LOCAL_BIN [PEER_BIN]: abort a pair ticket when the two nodes run different binaries.
kit_same_binary() {
    local a b
    a=$(kit_sha "$1")
    b=$(kit_peer sha256sum "${2:-$1}" | cut -c1-16)
    kit_log "binary local $a peer $b"
    [[ $a == "$b" ]] || kit_die "binaries differ (local $a, peer $b)"
}

# kit_warm_pagecache FILE...: read the model into the page cache with 8 parallel readers (cold-disk effects are
# not part of any prefill figure; the ticket header records how long it took).
kit_warm_pagecache() {
    local f sz blk=$((16 * 1024 * 1024)) n=8 per i t0
    t0=$(date +%s)
    for f in "$@"; do
        [[ -f $f ]] || continue
        sz=$(stat -c %s "$f"); per=$(((sz / blk + n) / n))
        for ((i = 0; i < n; i++)); do
            timeout 120 dd if="$f" of=/dev/null bs=$blk skip=$((i * per)) count=$per status=none &
        done
    done
    wait
    kit_log "page-cache warm $(($(date +%s) - t0)) s"
}

# Sampler: one row per KIT_SAMPLE_S seconds:
#   time power_uW sclk_Hz temp_mC busy_pct gtt_used_B MemAvailable_kB MemFree_kB evicted_ms
# evicted_ms sums /sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms over the run's process and its children.
# timeout owns a process group. Stop only that group, with a bounded grace period.
kit_stop_group() {
    local pid=${1:-} rc=0
    KIT_STOP_RC=0
    [[ -n $pid ]] || return 0
    kill -TERM "$pid" 2>/dev/null || true
    sleep 1
    pkill -KILL -P "$pid" 2>/dev/null || true
    kill -KILL -- "-$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || rc=$?
    KIT_STOP_RC=$rc
}

kit_sample_loop() {
    local dir=$1 kind=$2 hw=$3 dev=$4 ev p f
    set -o pipefail
    while :; do
        if [[ $kind == nvidia ]]; then
            nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu,utilization.gpu,memory.used \
                --format=csv,noheader,nounits -i "${KIT_NVIDIA_GPU:-0}" |
                awk -F', *' -v t="$(date +%s.%N)" \
                'NF != 5 {exit 1} {printf "%s %d %d %d %d %d 0 0 0\n", t, $1*1e6, $2*1e6, $3*1000, $4, $5*1048576; n++} END {if (!n) exit 1}' || {
                    echo 'nvidia sampler failed' >>"$dir/sampler.error"; return 1;
                }
        else
            ev=0
            if [[ -s $dir/pid ]]; then
                for p in $(cat "$dir/pid") $(pgrep -P "$(cat "$dir/pid")" 2>/dev/null); do
                    for f in /sys/class/kfd/kfd/proc/"$p"/stats_*/evicted_ms; do
                        [[ -r $f ]] && ev=$((ev + $(cat "$f")))
                    done
                done
            fi
            printf '%s %s %s %s %s %s %s %s\n' "$(date +%s.%N)" \
                "$(cat "$hw/power1_average")" "$(cat "$hw/freq1_input")" \
                "$(cat "$hw/temp1_input")" "$(cat "$dev/gpu_busy_percent")" \
                "$(cat "$dev/mem_info_gtt_used")" \
                "$(awk '/^MemAvailable:/{a=$2} /^MemFree:/{f=$2} END{printf "%s %s", a, f}' /proc/meminfo)" "$ev" || return 1
        fi
        sleep "${KIT_SAMPLE_S:-0.2}"
    done
}
kit_sampler_start() {
    local dir=$1 kind=amd hw='' dev=''
    KIT_SAMPLER_DIR=$dir
    KIT_SAMPLER=''
    if [[ ${KIT_SAMPLER_KIND:-auto} == nvidia || ( ${KIT_SAMPLER_KIND:-auto} == auto && ! -d /sys/class/kfd ) ]] &&
        command -v nvidia-smi >/dev/null 2>&1; then
        kind=nvidia
    else
        hw=$(ls -d /sys/class/drm/card*/device/hwmon/hwmon* 2>/dev/null | head -1)
        dev=${hw%/hwmon/*}
        [[ -r $hw/power1_average && -r $hw/freq1_input && -r $dev/mem_info_gtt_used ]] || {
            echo 'no GPU telemetry source' >"$dir/sampler.error"; return 1;
        }
    fi
    timeout -k 1 999999 bash -c "$(declare -f kit_sample_loop); kit_sample_loop \"\$@\"" bench-sampler \
        "$dir" "$kind" "$hw" "$dev" >>"$dir/samples.tsv" 2>>"$dir/sampler.error" &
    KIT_SAMPLER=$!
}
kit_sampler_stop() {
    if [[ -n ${KIT_SAMPLER:-} ]]; then
        kill -0 "$KIT_SAMPLER" 2>/dev/null || echo 'sampler exited early' >>"$KIT_SAMPLER_DIR/sampler.error"
        kit_stop_group "$KIT_SAMPLER"
    fi
    KIT_SAMPLER=''
}

# Timestamp both producer streams and wait for both stampers before reading receipts.
kit_start_run() {
    local dir=$1 cap=$2 outfd errfd
    shift 2
    local buffering=()
    command -v stdbuf >/dev/null && buffering=(stdbuf -oL -eL)
    exec {outfd}> >(kit_py stamp.py "$dir/stdout"); KIT_OUT_LOG=$!
    exec {errfd}> >(kit_py stamp.py "$dir/stderr"); KIT_ERR_LOG=$!
    ( echo "$BASHPID" >"$dir/pid"; exec timeout -k 1 "$cap" "${buffering[@]}" "$@" ) >&$outfd 2>&$errfd &
    KIT_RUN_PID=$!
    exec {outfd}>&- {errfd}>&-
}
kit_logs_done() { wait "$KIT_OUT_LOG" "$KIT_ERR_LOG"; }

# kit_run DIR CAP_S CMD...: run one process under the sampler with time-stamped stderr. Never returns non-zero
# for a failing command; the receipt records rc and marks the run invalid instead.
kit_run() {
    local dir=$1 cap=$2 k0 s e rc
    shift 2
    rm -rf -- "$dir"; mkdir -p -- "$dir"
    printf '%q ' "$@" >"$dir/cmd.txt"; echo >>"$dir/cmd.txt"
    local mem; mem=$(kit_mem)
    kit_sampler_start "$dir"
    k0=$(date '+%Y-%m-%d %H:%M:%S'); s=$(date +%s%N)
    kit_start_run "$dir" "$cap" "$@"
    rc=0; wait "$KIT_RUN_PID" || rc=$?
    kit_logs_done
    e=$(date +%s%N); sleep 0.3; kit_sampler_stop
    journalctl -k --since "$k0" --no-pager 2>/dev/null | grep -aiE 'evict|amdgpu|kfd|thunderbolt|nhi|oom|reset' |
        grep -v '^-- ' | head -20 >"$dir/klog.txt" || true
    local bin=$1; [[ $bin == env ]] && bin=$(awk '{for(i=2;i<=NF;i++) if ($i !~ /=/) {print $i; exit}}' "$dir/cmd.txt")
    local shutdown=(); [[ -f $dir/intentional-shutdown ]] && shutdown=(--intentional-shutdown)
    kit_py receipt.py meta "$dir" "${shutdown[@]}" --rc "$rc" --wall-ms $(((e - s) / 1000000)) --mem "$mem" \
        --host-tag "${KIT_HOST_TAG:-local}" --bin-sha "$( [[ -f $bin ]] && kit_sha "$bin" || echo -)"
    return 0
}

# kit_receipt DIR TAG [receipt.py options]: write receipt.json and print one summary line.
kit_receipt() { kit_py receipt.py summarize "$1" --tag "$2" "${@:3}"; }

# kit_summary_done OUT: final step of every ticket: summary.txt (<= 60 lines) and a .done marker.
kit_summary_done() {
    local out=$1
    {
        if [[ -f $out/ticket.log ]]; then
            grep -a '^ticket \|^kit rc' "$out/ticket.log"
            grep -a '^== \|SKIP\|ABORT' "$out/ticket.log" | head -38
        fi
        kit_py abab.py summary "$out" || true
    } >"$out/summary.full" 2>&1
    head -60 "$out/summary.full" >"$out/summary.tmp"
    mv -f "$out/summary.tmp" "$out/summary.txt"
    rm -f "$out/summary.full"
    touch "$out/.done"
}

# kit_pair_arm BASE CAP READY_RX PEER_CMD_ARRAY LOCAL_CMD_ARRAY
# Two-rank arm: rank 1 starts on KIT_PEER (run1.sh there, same receipt rules), the kit waits until the peer's
# stderr matches READY_RX (or the peer exits, max KIT_PEER_WAIT s, default 120), then runs rank 0 locally.
# Receipts: BASE-r0/receipt.json (arm KIT_ARM) and BASE-r1/receipt.json copied back (arm KIT_ARM@r1).
# The peer writes under KIT_PEER_OUT (default: the same path as KIT_OUT).
kit_pair_arm() {
    local base=$1 cap=$2 ready=$3 l0 i pdir cap0 root
    local -n peer_cmd=$4 local_cmd=$5
    root=$(kit_peer_root)
    pdir=${KIT_PEER_OUT:-$KIT_OUT}/$(basename "$base")-r1
    kit_peer "cd $(printf %q "$root") && KIT_TAG=$(printf %q "${KIT_TAG:-x}-r1") KIT_ARM=$(printf %q "${KIT_ARM:-x}@r1") \
        KIT_WORKLOAD=$(printf %q "${KIT_WORKLOAD:-}") KIT_ROUND=${KIT_ROUND:-0} KIT_TOKENS=${KIT_TOKENS:-0} KIT_HOST_TAG=peer KIT_RECEIPT_ARGS=$(printf %q "${KIT_RECEIPT_ARGS:-}") \
        bash scripts/bench/common/run1.sh $(printf %q "$pdir") $cap -- $(printf '%q ' "${peer_cmd[@]}")" >"$base-r1.log" 2>&1 &
    local pid=$!
    l0=$(date +%s)
    for ((i = 0; i < ${KIT_PEER_WAIT:-120} * 2; i++)); do
        kit_peer "grep -aqE $(printf %q "$ready") $(printf %q "$pdir/stderr") $(printf %q "$pdir/stdout") 2>/dev/null" && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    echo "  peer ready after $(($(date +%s) - l0)) s"
    cap0=$((cap - ($(date +%s) - l0) - 2)); ((cap0 < 20)) && cap0=20
    KIT_HOST_TAG=local bash "$KIT_LIB/run1.sh" "$base-r0" "$cap0" -- "${local_cmd[@]}"
    wait "$pid"
    mkdir -p -- "$base-r1"
    kit_peer "cat $(printf %q "$pdir/receipt.json")" >"$base-r1/receipt.json" 2>/dev/null || rm -f -- "$base-r1/receipt.json"
    kit_py receipt.py pair "$base-r0/receipt.json" "$base-r1/receipt.json" --expected "${KIT_EXPECTED:-0}" || true
    sed 's/^/  r1| /' "$base-r1.log" | head -8
}
