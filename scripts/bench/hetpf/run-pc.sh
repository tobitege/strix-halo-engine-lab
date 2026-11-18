#!/bin/bash
# run-pc.sh OUT: the hetpf DS41 campaign from the CUDA host (Git Bash or Linux), one detached run that ends with OUT/summary.txt
# and OUT/.done. Steps: (1) link probe host<->rank-0 node (node side via ssh, no GPU); (2) the CUDA prefill service on the
# selected device, R (RTX alone) x2 while it is the only GPU user; (3) for each TICKETS entry "MODE:ARMS" one pair serve
# process inside a pair GPU ticket (pair_serve.sh), the arms run from here as soon as the pair is READY, then OUT/stop;
# (4) service stopped, node monitors fetched, summarize.py. Never chains a build: binaries must exist.
# Environment (no defaults for hosts or paths):
#   NODE0 / NODE0_ADDR  ssh host of rank 0 / the address this host reaches its HTTP port on;  NODE1  ssh host of rank 1
#   NODE_KIT            dir on both nodes holding pair_serve.sh, link_probe.py and the key file (KEYNAME, default key.txt)
#   NODE_ENV            "BIN=... MODEL=... TRELLIS=... ROCM=..." for pair_serve.sh (paths valid on both nodes)
#   TURN                the node's pair-ticket command prefix, e.g. "GPU_TURN_MAX=120 <lab>/gpu-turn-pair.sh"
#   SVC MODEL_RTX KEY IDS TOKENS   service exe, its gguf (native path), bearer key file, prompt ids file, prompt length
#   SVC_ARGS            extra service arguments (default: --exact --host-gib 60 --host-commit-floor-gib 20 --host-pin-wait-s 90)
#   TICKETS             default "fast:A,B,A,B fast:B,A,B,A exact:B,A,B";  HOLD per ticket (s, default 115); PORT (18790)
#   COORD               optional coordination file: "PC-BIG start/end" lines are appended to it;  WAIT_READY (s, 1800)
set -u
set -o pipefail
OUT=$1
# Atomic ownership: existing directories (even empty ones) cannot be reused.
mkdir "$OUT" || { echo "OUT must be a new campaign directory" >&2; exit 64; }
K=$(cd "$(dirname "$0")" && pwd)
: "${NODE0:?}" "${NODE0_ADDR:?}" "${NODE1:?}" "${NODE_KIT:?}" "${NODE_ENV:?}" "${TURN:?}" "${SVC:?}" "${MODEL_RTX:?}" "${KEY:?}" "${IDS:?}" "${TOKENS:?}"
PORT=${PORT:-18790}; SPORT=${SPORT:-8097}; HOLD=${HOLD:-115}; WAIT_READY=${WAIT_READY:-1800}; KEYNAME=${KEYNAME:-key.txt}
TICKETS=${TICKETS:-"fast:A,B,A,B fast:B,A,B,A exact:B,A,B"}
SVC_ARGS=${SVC_ARGS:-"--exact --host-gib 60 --host-commit-floor-gib 20 --host-pin-wait-s 90"}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=${PY:-python}; SSH="timeout -k 2 20 ssh -n -o BatchMode=yes -o ConnectTimeout=8"; S=$OUT/summary.txt; J=$OUT/runs.jsonl
P() { case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) cygpath -m "$1";; *) echo "$1";; esac; }
log() { echo "$(date +%T) $*" | tee -a "$OUT/driver.log"; }
coord() { [ -n "${COORD:-}" ] && echo "- $(date +%H:%M) ${COORD_NAME:-hetpf}: $*" >> "$COORD"; return 0; }
RUN_ID=$($PY -c 'import uuid; print(uuid.uuid4().hex)') || exit 2
printf '%s\n' "$RUN_ID" > "$OUT/run-id"
SP=; WP=; MP=; CP=; D=; FAILED=0
OWN=setsid
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) OWN=;; esac
stop_svc() {
  if [ -n "$WP" ]; then taskkill //F //T //PID "$WP" > /dev/null 2>&1 || :; fi
  for p in "$SP" "$MP" "$CP"; do
    [ -n "$p" ] || continue
    wp=$(cat /proc/"$p"/winpid 2>/dev/null || :)
    [ -z "$wp" ] || taskkill //F //T //PID "$wp" > /dev/null 2>&1 || :
    kill -TERM -- -"$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null || :
  done
  sleep 1
  for p in "$SP" "$MP" "$CP"; do
    [ -n "$p" ] || continue
    kill -KILL -- -"$p" 2>/dev/null || kill -KILL "$p" 2>/dev/null || :
    wait "$p" 2>/dev/null || :
  done
  SP=; WP=; MP=; CP=
}
stop_ticket() {
  [ -n "$D" ] || return 0
  $SSH "$NODE0" "touch $D/stop; p=\$(cat $D/wrapper.pid 2>/dev/null); case \$p in ''|*[!0-9]*) ;; *) [ \$p -gt 1 ] && kill -TERM \$p 2>/dev/null;; esac" || :
  # Also cancel a queued ticket. The wrapper has its own bounded resource cleanup.
  $SSH "$NODE0" "sleep 3; p=\$(cat $D/ticket.pid 2>/dev/null); case \$p in ''|*[!0-9]*) ;; *) [ \$p -gt 1 ] && kill -TERM -- -\$p 2>/dev/null;; esac" || :
}
cleanup() {
  rc=$?; trap - EXIT INT TERM
  stop_svc
  stop_ticket
  coord "PC-BIG end status=$rc"
  printf '%s\n' "$rc" > "$OUT/status"
  if [ "$rc" = 0 ]; then touch "$OUT/.done"; else touch "$OUT/.failed"; fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Background wait lets Bash service cancellation promptly, including a blocked client.
client() { $OWN "$@" & CP=$!; wait "$CP"; r=$?; CP=; return "$r"; }
{ uname -r; printf 'amdgpu.iommu_perfopt='; cat /sys/module/amdgpu/parameters/iommu_perfopt 2>/dev/null || echo unavailable; } > "$OUT/host.txt"
printf '%s\n' "$SVC_ARGS" > "$OUT/service-args.txt"
# (1) link
$SSH "$NODE0" "mkdir -p $NODE_KIT/runs && cd $NODE_KIT && (setsid python3 link_probe.py serve --bind $NODE0_ADDR --lifetime 180 --port $((PORT + 1)) --idle-exit 60 > runs/$RUN_ID-link.out 2>&1 < /dev/null &)"; sleep 2
$PY -I "$(P "$K/link_probe.py")" probe --host "$NODE0_ADDR" --port $((PORT + 1)) --mib 64 --reps 3 --state-bytes 32.2e6 \
  --run-id "$RUN_ID" --json "$(P "$OUT/link.json")" > "$OUT/link.txt" 2>&1 || FAILED=1; log "link $(tail -n 1 "$OUT/link.txt")"
# (2) service
u0=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "$u0" -lt 2048 ] || { log "ABORT device busy ($u0 MiB)"; exit 4; }
coord "PC-BIG start (DS41 prefill service + pair hand-off tickets, device index $CUDA_VISIBLE_DEVICES only, <= 30 min)"
$OWN timeout -k 5 1800 "$SVC" "$MODEL_RTX" --host 127.0.0.1 --port $SPORT --positions $((TOKENS + 512)) --api-key "$(cat "$KEY")" $SVC_ARGS > "$OUT/svc.out" 2> "$OUT/svc.err" &
SP=$!; WP=$(cat /proc/$SP/winpid 2>/dev/null || echo "")
$OWN timeout -k 2 1800 bash -c 'while kill -0 "$1" 2>/dev/null; do echo "$(date +%s%N), $(nvidia-smi -i "$2" --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader,nounits)"; sleep 0.5; done' _ "$SP" "$CUDA_VISIBLE_DEVICES" >> "$OUT/smi.samples" & MP=$!
for i in $(seq 1 150); do grep -q '"event":"READY"' "$OUT/svc.out" 2>/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 2; done
grep -q '"event":"READY"' "$OUT/svc.out" || { log "ABORT service not ready"; tail -n 5 "$OUT/svc.err" >> "$S"; stop_svc; exit 5; }
log "service ready after $((i * 2)) s"; sleep 10
client $PY -I "$(P "$K/het_ab.py")" --run-id "$RUN_ID" --arms R,R --ids "$(P "$IDS")" --tokens "$TOKENS" --key "$(P "$KEY")" --label R0 --out "$(P "$J")" \
  --service http://127.0.0.1:$SPORT >> "$OUT/client.log" 2>&1 || FAILED=1
# (3) pair tickets
n=0; MONS=()
for t in $TICKETS; do
  n=$((n + 1)); mode=${t%%:*}; arms=${t#*:}; lab=T$n-$mode; D=$NODE_KIT/runs/$RUN_ID-$lab
  $SSH "$NODE0" "mkdir -p $NODE_KIT/runs && mkdir $D && cd $NODE_KIT && (setsid bash -c 'echo \$\$ > $D/ticket.pid; $TURN hetpf-$lab env PEER=$NODE1 BIND=$NODE0_ADDR PORT=$PORT KEYFILE=$NODE_KIT/$KEYNAME $NODE_ENV bash $NODE_KIT/pair_serve.sh $D $mode $HOLD > $D/ticket.out 2>&1; echo \$? > $D/ticket.status; touch $D/ticket.done' < /dev/null > /dev/null 2>&1 &)" || { FAILED=1; break; }
  log "$lab submitted ($arms)"; t0=$(date +%s); ok=0
  while [ $(( $(date +%s) - t0 )) -lt "$WAIT_READY" ]; do
    st=$($SSH "$NODE0" "[ -e $D/ready ] && echo R; [ -e $D/ticket.done ] && echo D" | tr -d '\n')
    case $st in *D*) break;; *R*) ok=1; break;; esac; sleep 3
  done
  [ $ok = 1 ] || { FAILED=1; log "$lab never ready (${st:-timeout})"; stop_ticket; D=; continue; }
  log "$lab ready after $(( $(date +%s) - t0 )) s"
  client $PY -I "$(P "$K/het_ab.py")" --run-id "$RUN_ID" --arms "$arms" --ids "$(P "$IDS")" --tokens "$TOKENS" --key "$(P "$KEY")" --label "$lab" --mode "$mode" \
    --out "$(P "$J")" --service http://127.0.0.1:$SPORT --pair "http://$NODE0_ADDR:$PORT" >> "$OUT/client.log" 2>&1 || FAILED=1
  $SSH "$NODE0" "touch $D/stop"
  for _ in $(seq 1 60); do $SSH "$NODE0" "[ -e $D/done ]" && break; sleep 2; done
  mkdir -p "$OUT/$lab"; for fl in info.txt mon0.txt mon1.txt ticket.out host0.txt host1.txt config0.txt config1.txt status; do timeout -k 2 20 scp -q "$NODE0:$D/$fl" "$OUT/$lab/" 2>/dev/null; done; MONS+=(--mon "$lab=$(P "$OUT/$lab")")
  $SSH "$NODE0" "[ \"\$(cat $D/status 2>/dev/null)\" = 0 ]" || { FAILED=1; stop_ticket; }
  D=
  log "$lab done: $(grep -E '^(ready|stop|rc)' "$OUT/$lab/info.txt" 2>/dev/null | tr '\n' ' ')"
done
client $PY -I "$(P "$K/het_ab.py")" --run-id "$RUN_ID" --arms R --attempt-offset 2 --ids "$(P "$IDS")" --tokens "$TOKENS" --key "$(P "$KEY")" --label R0 --out "$(P "$J")" \
  --service http://127.0.0.1:$SPORT >> "$OUT/client.log" 2>&1 || FAILED=1
stop_svc; sleep 3
{ grep -E 'rror|refus' "$OUT/svc.err" | head -n 3 | cut -c1-200
  $PY -I "$(P "$K/summarize.py")" "$(P "$J")" "${MONS[@]}" --smi "$(P "$OUT/smi.samples")" --link "$(P "$OUT/link.json")" 2>&1 || FAILED=1
  awk -F', ' '$3+0>c{c=$3+0} END{print "RTX max SM MHz " c}' "$OUT/smi.samples"; } >> "$S"
log "summary written"

exit "$FAILED"
