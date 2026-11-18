#!/bin/bash
# arm.sh RID SECS CAP SESSIONS [ENV=VAL ...] : one MiniMax pair gate arm (rank 0 here, rank 1 on $MMF_PEER), run INSIDE one
# pair GPU ticket. SESSIONS = comma list of prompt names ($MMF_PROMPTS/<name>.u32), run in order in one process (first =
# cold, last = warm). Writes $MMF_RUN/RID on both nodes: q-* dumps, r0.log / r1.log, mon0/mon1 (busy, sclk, kfd
# evicted_ms of this user's rank processes), mem.txt (MemFree / MemAvailable), rc. A run directory is never reused.
# Environment: MMF_PEER (ssh host of rank 1), MMF_BIN, MMF_MODEL, MMF_PROMPTS, MMF_RUN (same paths on both nodes).
set -u
RID=$1 SECS=$2 CAP=$3 SESS=$4; shift 4
: "${MMF_PEER:?}" "${MMF_BIN:?}" "${MMF_MODEL:?}" "${MMF_PROMPTS:?}" "${MMF_RUN:?}"
R=$MMF_RUN/$RID BIN=$MMF_BIN
S2="ssh -n -o BatchMode=yes -o ConnectTimeout=5 $MMF_PEER"
case $RID in *[!A-Za-z0-9._-]*) echo "arm: bad RID"; exit 64;; esac
[ -x "$BIN" ] || { echo "arm: no binary"; exit 3; }
[ -e "$R" ] && { echo "arm: run dir exists (never reused)"; exit 3; }
$S2 "[ ! -e $R ] && mkdir -p $R" || { echo "arm: peer run dir exists or cannot be made"; exit 3; }
mkdir -p "$R" || exit 3
date +%s.%N > "$R/t0"
MI="grep -E '^(MemFree|MemAvailable):' /proc/meminfo | awk '{printf \"%s %.1f GiB \", \$1, \$2/1048576}'"
{ echo "r0 $(bash -c "$MI")"; echo "r1 $($S2 "$MI")"; } > "$R/mem.txt"
for v in $(env | grep -oE "^STRIX_[A-Z0-9_]*" | grep -v EXPECT_TBS); do unset "$v"; done
ENVS=(STRIX_SESSION_CACHE=0 STRIX_MM_STATE_FNV=1 STRIX_MM_QUALITY_OUT=$R/q STRIX_MM_QUALITY_ROWS=256 STRIX_MM_QUALITY_SELECT=1 "$@")
printf '%s\n' "$@" > "$R/arm-env"
echo "$SESS" > "$R/sessions"
A="--slots 1 --capacity $CAP --max-rows 4096"
for s in ${SESS//,/ }; do [ -s "$MMF_PROMPTS/$s.u32" ] || { echo "arm: no prompt $s"; exit 3; }; A="$A --session $s:$MMF_PROMPTS/$s.u32:1"; done
MON='d=/sys/class/drm/card1/device; [ -d $d ] || d=/sys/class/drm/card0/device; while :; do ev=0; for p in $(pgrep -u $(id -u) -f "[s]trix-minimax-rank"); do for f in /sys/class/kfd/kfd/proc/$p/stats_*/evicted_ms; do [ -r $f ] && ev=$((ev+$(cat $f))); done; done; echo "$(date +%s.%N) busy=$(cat $d/gpu_busy_percent) sclk=$(grep "\*" $d/pp_dpm_sclk | tr -d "\n") ev=$ev"; sleep 0.5; done'
bash -c "$MON" > "$R/mon0.txt" 2>/dev/null & M0=$!
$S2 "nohup bash -c '$MON' > $R/mon1.txt 2>/dev/null & echo \$!" > "$R/m1.pid"
LOCAL=
cleanup() { kill $M0 2>/dev/null; $S2 "kill $(cat "$R/m1.pid" 2>/dev/null) 2>/dev/null; pgrep -f '[s]trix-minimax-rank .* $R/r1' | xargs -r kill -KILL" 2>/dev/null; [ -n "$LOCAL" ] && kill -- "-$LOCAL" 2>/dev/null; }
trap cleanup EXIT
$S2 "cd $R && exec env ${ENVS[*]} setsid timeout -k 2 $SECS $BIN $MMF_MODEL 1 usb4-auto $R/r1 $A > $R/r1.log 2>&1" & PEER=$!
(cd "$R" && exec env "${ENVS[@]}" setsid timeout -k 2 "$SECS" "$BIN" "$MMF_MODEL" 0 usb4-auto "$R/r0" $A > "$R/r0.log" 2>&1) & LOCAL=$!
wait "$LOCAL"; rc0=$?; wait "$PEER"; rc1=$?; LOCAL=
echo "$rc0 $rc1" > "$R/rc"
date +%s.%N > "$R/t1"
echo "arm $RID rc $rc0/$rc1 $(grep -a -h -o -E 'TTFT [0-9.]+ s' "$R/r0.log" | tr '\n' ' ')"
