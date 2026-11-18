#!/usr/bin/env bash
# Served decode by prompt class (prose / edit / json) at 1/4/8 concurrent chats, through tools/serve_bench.py.
#
#   serve-decode.sh --tag TAG --port P [--classes prose,edit,json] [--concurrency 1,4,8] [--max-tokens 128]
#                   [--rounds 1] [--ready-path /v1/models | --ready-log REGEX] [--ready-s 240] [--chat]
#                   [--body-extra JSON] [--arm NAME] [--round N (ABAB index of this server start)] [--peer-cmd 'RANK1 SERVER CMD'] [--client-cmd 'CMD']
#                   -- SERVER_CMD...
# --client-cmd replaces the class bursts by any client (run with KIT_PORT, KIT_URL and KIT_RUN_DIR exported), so
# Clients must write KIT_RUN_DIR/client-result.json with completed>0 and failed=0.
# Other served kits reuse the same server start / validity / stop.
# --peer-cmd starts rank 1 of a pair server on KIT_PEER first (common/run1.sh there: same sampler and receipt,
# capped by the ticket budget, so it exits with rank 0); --ready-log polls the server stderr instead of HTTP
# (DS41 serve prints "READY", the Qwen server "listening on").
# Starts SERVER_CMD under the shared sampler (OUT/TAG/server: stamped stderr, samples, klog, meta), waits for
# GET http://127.0.0.1:P<ready-path> to answer, runs one warm-up request, then for each round and class one
# serve_bench.py burst set (aggregate tok/s from the server's cycle receipts when the log carries them, per-request
# and end-to-end tok/s, TTFT, token ids), stops the server and writes one receipt per (class, round) with
# metrics agg_C<n> / e2e_C<n>. Class prompts: scripts/bench/prompts/classes/<class>.txt (or --prompt-dir).
# The decode class matters: drafting (MTP / n-gram) acceptance differs by an order of magnitude between them.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/runlib.sh"
tag='' port='' classes=prose,edit,json conc=1,4,8 maxt=128 rounds=1 ready=/v1/models ready_log='' peer_cmd='' client_cmd='' round_base=0 ready_s=240 chat=() extra='{}' arm=''
prompt_dir=$here/../prompts/classes
while (($#)); do
    case $1 in
        --tag) tag=$2 ;; --port) port=$2 ;; --classes) classes=$2 ;; --concurrency) conc=$2 ;; --max-tokens) maxt=$2 ;;
        --rounds) rounds=$2 ;; --ready-path) ready=$2 ;; --ready-log) ready_log=$2 ;; --peer-cmd) peer_cmd=$2 ;; --client-cmd) client_cmd=$2 ;; --round) round_base=$2 ;; --ready-s) ready_s=$2 ;; --body-extra) extra=$2 ;;
        --prompt-dir) prompt_dir=$2 ;; --arm) arm=$2 ;;
        --chat) chat=(--chat); shift; continue ;;
        --) shift; break ;;
        *) kit_die "unknown argument $1" ;;
    esac
    shift 2
done
[[ -n $tag && -n $port && $# -gt 0 ]] || kit_die 'need --tag, --port and -- SERVER_CMD'
out=${KIT_OUT:-bench-out}; d=$out/$tag; sdir=$d/server
rm -rf -- "$d"; mkdir -p -- "$sdir"
kit_clean_env
printf '%q ' "$@" >"$sdir/cmd.txt"; echo >>"$sdir/cmd.txt"
# One absolute deadline, including startup, all clients and bounded teardown.
export KIT_BUDGET=${KIT_BUDGET:-115}
ppid_='' spid='' client_pid='' peer_run_dir='' server_rc=125 intentional=0 cleaned=0
cleanup() {
    ((cleaned)) && return
    cleaned=1
    [[ -z $client_pid ]] || kit_stop_group "$client_pid"
    if [[ -n $spid ]]; then
        kill -0 "$spid" 2>/dev/null && intentional=1
        kit_stop_group "$spid"; server_rc=$KIT_STOP_RC
    fi
    if [[ -n $ppid_ ]]; then
        # Only the timeout group recorded by this run1 belongs to us. The remote
        # wrapper remains alive to record the actual status and sampler failures.
        timeout -k 1 3 ssh -n -o BatchMode=yes -o ConnectTimeout=2 "$KIT_PEER" \
            "p=\$(cat $(printf %q "$peer_run_dir/pid") 2>/dev/null); if test -n \"\$p\" && kill -0 \"\$p\" 2>/dev/null; then touch $(printf %q "$peer_run_dir/intentional-shutdown"); kill -TERM \"\$p\"; sleep 1; kill -KILL -- -\"\$p\" 2>/dev/null || true; fi" >/dev/null 2>&1 || true
        for ((j=0; j<20; j++)); do kill -0 "$ppid_" 2>/dev/null || break; sleep 0.1; done
        kit_stop_group "$ppid_"
    fi
    kit_sampler_stop
    [[ -z $spid ]] || kit_logs_done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
phase() {
    local left=$(($(kit_left) - 16)) rc=0
    ((left > 0)) || return 124
    timeout -k 1 "$left" "$@" &
    client_pid=$!
    wait "$client_pid" || rc=$?
    client_pid=''
    return "$rc"
}
# Readiness without an external client (a node without curl used to wait out the whole ticket with the server
# long READY): one HTTP/1.0 GET over bash /dev/tcp, ready on a 2xx status line.
http_ready() (
    exec 9<>"/dev/tcp/127.0.0.1/$port" || exit 1
    printf 'GET %s HTTP/1.0\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n' "$ready" >&9
    IFS= read -r -t 2 status <&9 || exit 1
    [[ $status =~ ^HTTP/[0-9.]+[[:space:]]2[0-9][0-9] ]]
) 2>/dev/null
if [[ -n $peer_cmd ]]; then
    peer_run_dir=${KIT_PEER_OUT:-$out}/$tag-r1
    timeout -k 1 "$(kit_left)" ssh -n -o BatchMode=yes -o ConnectTimeout=3 "$KIT_PEER" \
        "cd $(printf %q "$(kit_peer_root)") && KIT_TAG=$(printf %q "$tag-r1") KIT_HOST_TAG=peer KIT_RECEIPT_ARGS='--evict-whole-run --strict-klog' bash scripts/bench/common/run1.sh $(printf %q "$peer_run_dir") $(($(kit_left) - 4)) -- $peer_cmd" >"$d/peer.log" 2>&1 &
    ppid_=$!
fi
mem=$(kit_mem); kit_sampler_start "$sdir"
k0=$(date '+%Y-%m-%d %H:%M:%S'); s0=$(date +%s%N)
left=$(($(kit_left) - 16))
((left > 0)) || exit 124
kit_start_run "$sdir" "$left" "$@"
spid=$KIT_RUN_PID
url=http://127.0.0.1:$port
up=0; ready_end=$(($(date +%s) + ready_s))
while (($(kit_left) > 16 && $(date +%s) < ready_end)); do
    if [[ -n $ready_log ]]; then grep -aqE "$ready_log" "$sdir/stderr" "$sdir/stdout" 2>/dev/null && { up=1; break; }
    else http_ready && { up=1; break; }; fi
    kill -0 "$spid" 2>/dev/null || break
    sleep 0.2
 done
echo "  server ready=$up after $((($(date +%s%N) - s0) / 1000000)) ms"
client_rc=125
IFS=, read -ra C <<<"$classes"
expected=()
for ((r=0; r<rounds; r++)); do for c in "${C[@]}"; do expected+=("$c-$r"); done; done
if ((up)) && [[ -n $client_cmd ]]; then
    phase env KIT_PORT="$port" KIT_URL="$url" KIT_RUN_DIR="$d" bash -c "$client_cmd" >"$d/client.txt" 2>&1
    client_rc=$?
elif ((up)); then
    client_rc=0
    for ((r=0; r<rounds; r++)); do
        for c in "${C[@]}"; do
            [[ -s $prompt_dir/$c.txt ]] || { echo "== $tag-$c SKIP: missing prompt"; client_rc=2; continue; }
            if ((r == 0)) && [[ $c == "${C[0]}" ]]; then
                phase "$KIT_PY" -I "$KIT_ROOT/tools/serve_bench.py" --url "$url" "${chat[@]}" \
                    --prompt-file "$prompt_dir/$c.txt" --concurrency 1 --max-tokens 16 --body-extra "$extra" \
                    --timeout "$(kit_left)" >"$d/warmup.txt" 2>&1 || { client_rc=$?; break 2; }
            fi
            phase "$KIT_PY" -I "$KIT_ROOT/tools/serve_bench.py" --url "$url" "${chat[@]}" --prompt-file "$prompt_dir/$c.txt" \
                --concurrency "$conc" --max-tokens "$maxt" --log "$sdir/stderr" --body-extra "$extra" \
                --timeout "$(kit_left)" --out "$d/$c-$r.json" >"$d/$c-$r.txt" 2>&1 || { client_rc=$?; break 2; }
        done
    done
fi
# A server that already died is never counted as an intentional shutdown.
cleanup
trap - EXIT INT TERM
e0=$(date +%s%N)
timeout -k 1 1 journalctl -k --since "$k0" --no-pager 2>/dev/null | grep -aiE 'evict|amdgpu|kfd|thunderbolt|nhi|oom|reset' | grep -v '^-- ' |
    head -20 >"$sdir/klog.txt" || true
shutdown=(); ((intentional)) && shutdown=(--intentional-shutdown)
kit_py receipt.py meta "$sdir" --rc "$server_rc" "${shutdown[@]}" --wall-ms $(((e0 - s0) / 1000000)) --mem "$mem" --host-tag "${KIT_HOST_TAG:-local}" \
    --bin-sha "$( [[ -f $1 ]] && kit_sha "$1" || echo -)"
peer_args=()
if [[ -n $ppid_ ]]; then
    timeout -k 1 2 ssh -n -o BatchMode=yes -o ConnectTimeout=1 "$KIT_PEER" "cat $(printf %q "$peer_run_dir/receipt.json")" >"$d/peer-receipt.json" 2>/dev/null || true
    peer_args=(--peer "$d/peer-receipt.json")
fi
kit_py serve_receipts.py "$d" --tag "$tag" --arm "${arm:-$tag}" --server-up "$up" --round-base "$round_base" \
    --client "$([[ -n $client_cmd ]] && echo 1 || echo 0)" --client-rc "$client_rc" \
    --expected "$(IFS=,; echo "${expected[*]}")" --concurrency "$conc" "${peer_args[@]}"
receipt_rc=$?
((client_rc)) || client_rc=$receipt_rc
exit "$client_rc"
