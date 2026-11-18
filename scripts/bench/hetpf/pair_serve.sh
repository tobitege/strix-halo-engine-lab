#!/bin/bash
# pair_serve.sh OUT MODE HOLD: one DS41 pair serve process (rank 0 here, rank 1 on $PEER) for a remote client of the hetpf kit,
# meant to run INSIDE one pair GPU ticket on the rank-0 node. MODE exact | fast (fast = STRIX_DS41_FAST_PREFILL=1 on both
# ranks, the shared pair-defaults resolver). Writes OUT/ready once the server is READY, stops when OUT/stop exists or HOLD
# seconds after start, then OUT/done. GPU busy and sclk of both nodes are sampled every 0.25 s (OUT/mon0.txt, OUT/mon1.txt:
# "epoch_ns busy=% sclk=<dpm line>").
# Environment (same paths on both nodes): PEER (rank-1 ssh host), BIN (strix-deepseek41-serve), MODEL (Q2 gguf), TRELLIS (dir
# with ours-K4 and ours-K5), BIND (address the client reaches), PORT, KEYFILE (bearer key file), CTX (default 33280), SLOTS
# (default 2), ROCM (default /opt/rocm; its bin/ and lib/ are used), CACHE (--cache-profile, default off: a repeated prompt
# must be prefilled, not served from the prefix cache), EXTRA (more serve arguments).
set -u
OUT=$1; MODE=$2; HOLD=$3
: "${PEER:?}" "${BIN:?}" "${MODEL:?}" "${TRELLIS:?}" "${BIND:?}" "${PORT:?}" "${KEYFILE:?}"
CTX=${CTX:-33280}; SLOTS=${SLOTS:-2}; ROCM=${ROCM:-/opt/rocm}; EXTRA=${EXTRA:-}; CACHE=${CACHE:-off}
export PATH=$ROCM/bin:/usr/bin:/bin LD_LIBRARY_PATH=$ROCM/lib
SSH="timeout -k 2 20 ssh -n -o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=15 -o ServerAliveCountMax=8"
T0=$(date +%s)
for v in $(env | grep -oE "^STRIX_[A-Z0-9_]*"); do unset "$v"; done
ENVS="STRIX_DIST_USB4_TIMEOUT_MS=120000 STRIX_DIST_WAIT_TIMEOUT_MS=120000 HSA_DISABLE_FRAGMENT_ALLOCATOR=1"
ENVS="$ENVS STRIX_DS41_TRELLIS=$TRELLIS/ours-K4 STRIX_DS41_TRELLIS_MIX=head=$TRELLIS/ours-K5 STRIX_DS41_TRELLIS_SITES=all"
ENVS="$ENVS STRIX_DS41_TRELLIS_DENSE_FULL=1 STRIX_DS41_TRELLIS_HEADOUT=1"
case $MODE in fast) ENVS="$ENVS STRIX_DS41_FAST_PREFILL=1";; exact) ;; *) echo "bad MODE $MODE"; exit 64;; esac
A="--transport usb4auto --peer-timeout 120 --prefill-path device --slots $SLOTS --context $CTX --prefill-rows auto --no-dspark"
A="$A --http-port $PORT --http-bind $BIND --http-api-key-file $KEYFILE --cache-profile $CACHE $EXTRA"
# This is the DS41 experiment adapter. Reuse lifecycle helpers before adding engines.
TO=$((HOLD + 15))
MON='d=/sys/class/drm/card1/device; [ -d $d ] || d=/sys/class/drm/card0/device; while :; do echo "$(date +%s%N) busy=$(cat $d/gpu_busy_percent) sclk=$(grep "\*" $d/pp_dpm_sclk | tr -d "\n")"; sleep 0.25; done'
HOST_INFO='uname -r; printf "amdgpu.iommu_perfopt="; cat /sys/module/amdgpu/parameters/iommu_perfopt 2>/dev/null || echo unavailable'
M0=; P0=; P1=; REMOTE_OWNED=0
cleanup() {
  rc=$?; trap - EXIT INT TERM
  # The remote PID files are created AND read on rank 1. Each owns a setsid group.
  [ "$REMOTE_OWNED" = 0 ] || $SSH "$PEER" "for f in m1 r1; do p=\$(cat $OUT/\$f.pid 2>/dev/null); case \$p in ''|*[!0-9]*) continue;; esac; [ \$p -gt 1 ] && kill -TERM -- -\$p 2>/dev/null; done" || :
  for p in "$P0" "$M0"; do [ -n "$p" ] && kill -TERM -- -"$p" 2>/dev/null; done
  sleep 1
  [ "$REMOTE_OWNED" = 0 ] || $SSH "$PEER" "for f in m1 r1; do p=\$(cat $OUT/\$f.pid 2>/dev/null); case \$p in ''|*[!0-9]*) continue;; esac; [ \$p -gt 1 ] && kill -KILL -- -\$p 2>/dev/null; done" || :
  for p in "$P0" "$M0"; do [ -n "$p" ] && kill -KILL -- -"$p" 2>/dev/null; done
  [ -n "$P1" ] && kill "$P1" 2>/dev/null
  for p in "$P0" "$P1" "$M0"; do [ -n "$p" ] && wait "$p" 2>/dev/null; done
  timeout -k 2 20 scp -q "$PEER:$OUT/mon1.txt" "$OUT/" 2>/dev/null || :
  printf '%s\n' "$rc" > "$OUT/status"
  touch "$OUT/done"
  exit "$rc"
}
mkdir -p "$OUT"
# Refuse stale coordination even when this adapter is invoked directly.
for f in ready stop done wrapper.pid; do [ ! -e "$OUT/$f" ] || exit 64; done
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo $$ > "$OUT/wrapper.pid"
$SSH "$PEER" "mkdir -p $(dirname "$OUT") && mkdir $OUT" || exit 3
REMOTE_OWNED=1
H0=$(sha256sum "$BIN" | cut -c1-16); H1=$($SSH "$PEER" "sha256sum $BIN" | cut -c1-16)
echo "start $(date -Is) mode $MODE bin $H0/$H1" > "$OUT/info.txt"
echo "envs $ENVS" >> "$OUT/info.txt"; echo "args $A" >> "$OUT/info.txt"
bash -c "$HOST_INFO" > "$OUT/host0.txt"
$SSH "$PEER" "$HOST_INFO" > "$OUT/host1.txt"
[ -n "$H0" ] && [ "$H0" = "$H1" ] || exit 4
# Inspection exits before device work and records the resolved settings for each rank.
timeout -k 5 "$TO" env $ENVS "$BIN" "$MODEL" --rank 0 $A --print-effective-config > "$OUT/config0.txt" || exit 5
$SSH "$PEER" "timeout -k 5 $TO env $ENVS $BIN $MODEL --rank 1 $A --print-effective-config" > "$OUT/config1.txt" || exit 5
setsid timeout -k 2 "$TO" bash -c "$MON" > "$OUT/mon0.txt" 2>/dev/null & M0=$!
$SSH "$PEER" "setsid timeout -k 2 $TO bash -c '$MON' > $OUT/mon1.txt 2>/dev/null < /dev/null & echo \$! > $OUT/m1.pid" || exit 6
$SSH "$PEER" "cd $OUT && export PATH=$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH; setsid env $ENVS timeout -k 5 $TO $BIN $MODEL --rank 1 $A & p=\$!; echo \$p > r1.pid; wait \$p" > "$OUT/r1.log" 2>&1 & P1=$!
(cd "$OUT" && exec setsid env $ENVS timeout -k 5 "$TO" "$BIN" "$MODEL" --rank 0 $A < /dev/null) > "$OUT/r0.log" 2>&1 & P0=$!
for _ in $(seq 1 240); do grep -q '"READY"' "$OUT/r0.log" && break; kill -0 "$P0" 2>/dev/null || break; [ ! -e "$OUT/stop" ] || exit 143; sleep 0.5; done
grep -q '"READY"' "$OUT/r0.log" || exit 7
echo "ready t+$(( $(date +%s) - T0 )) s" | tee -a "$OUT/info.txt" > "$OUT/ready"
while [ ! -e "$OUT/stop" ] && kill -0 "$P0" 2>/dev/null && [ $(( $(date +%s) - T0 )) -lt "$HOLD" ]; do sleep 0.5; done
[ -e "$OUT/stop" ] && kill -0 "$P0" 2>/dev/null || exit 8
exit 0
