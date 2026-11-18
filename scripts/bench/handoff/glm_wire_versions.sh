#!/bin/bash
# GLM hand-off wire versions over a real link: the production sender (tests/glm_native/sender.cpp) against the real
# receiver + staging (tests/glm_native/receiver.cpp), synthetic state, one rank per arm. Compares wire v1/v2 (full index
# rows) with v3/v4 (index tail, strix/block_summary_state.hpp) at the same frontier. No GPU; the receiver side normally
# runs inside the node's GPU turn so no timing run shares the link or the disk.
#
# Usage (start the send side FIRST: it retries until each receiver listens):
#   glm_wire_versions.sh send SENDER_EXE CONNECT_IP:PORT KEYFILE ARMS OUT.tsv
#   glm_wire_versions.sh recv RECEIVER_EXE BIND_IP:PORT KEYFILE ARMS OUT.tsv
#   glm_wire_versions.sh key KEYFILE            # 32 random raw bytes (copy the same file to both sides)
# ARMS: one arm per line, "VERSION RANK FRONTIER" (e.g. "2 0 32767" then "4 0 32767", repeated for ABAB). Both sides
# read the same ARMS file in the same order. Output TSV: side version rank frontier rc seconds detail.
# The first arm of a run is the cold one (TCP / page cache); report it apart from the later (warm) arms.
set -u
side=${1:?side}
if [ "$side" = key ]; then head -c 32 /dev/urandom > "${2:?keyfile}"; exit 0; fi
exe=${2:?exe} ep=${3:?endpoint} key=${4:?keyfile} arms=${5:?arms} out=${6:?out}
prompt=$(printf '1%.0s' $(seq 64))
model=$(printf '2%.0s' $(seq 64))
printf 'side\tversion\trank\tfrontier\trc\tseconds\tdetail\n' > "$out"
while read -r version rank frontier; do
    [ -z "${version:-}" ] && continue
    case "$version" in \#*) continue ;; esac
    if [ "$side" = recv ]; then
        t0=$(date +%s.%N)
        res=$(GLM_TEST_WIRE_VERSION=$version "$exe" "$ep" "$key" "$prompt" "$model" "$rank" "$frontier" "$((frontier + 1))" 2>&1 < /dev/null)
        rc=$?
        t1=$(date +%s.%N)
        secs=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b - a }')
        printf 'recv\t%s\t%s\t%s\t%s\t%s\t%s\n' "$version" "$rank" "$frontier" "$rc" "$secs" "$(echo "$res" | tail -1)" >> "$out"
    else
        rc=1
        res=""
        tries=0
        while [ $rc -ne 0 ] && [ $tries -lt 3000 ]; do  # up to ~10 min waiting for this arm's receiver
            res=$(GLM_TEST_WIRE_VERSION=$version GLM_TEST_FRONTIER=$frontier "$exe" "$ep" "$key" "$prompt" "$model" ok "$rank" 2>&1 < /dev/null)
            rc=$?
            res=$(printf '%s\n' "$res" | tail -1)
            case "$res" in
                LOCALLY_SEALED*) ;;  # success requires both the marker and the executable's zero status
                *) if [ $rc -eq 0 ]; then rc=1; fi ;;
            esac
            if [ $rc -ne 0 ]; then tries=$((tries + 1)); sleep 0.2; fi
        done
        secs=$(echo "$res" | sed -n 's/.*seconds=\([0-9.e+-]*\).*/\1/p')
        printf 'send\t%s\t%s\t%s\t%s\t%s\t%s\n' "$version" "$rank" "$frontier" "$rc" "${secs:-NA}" "$res" >> "$out"
    fi
    # Keep the failure receipt, but never advance to another arm after a failed transfer.
    if [ $rc -ne 0 ]; then exit "$rc"; fi
done < "$arms"
