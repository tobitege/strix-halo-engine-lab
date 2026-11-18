#!/usr/bin/env bash
# Emulate smaller known devices on one large NVIDIA card: run a route's own benchmark command once per device
# profile (strix-device-profiles-v1, docs/devices.md) with --device-profile, and keep each run's memory budget,
# memory plan and receipt lines. The card is pinned by UUID; nothing here names a host, a user or a path.
#
#   scripts/bench/rtx/devprof-emulate.sh --gpu-uuid GPU-xxxxxxxx-... --out DIR \
#       [--profiles "rtx-5090-32 rtx-5080-16"] [--repeat 2] -- <route command ...>
#
# The route command is the route's usual bench / serve-smoke invocation (DS41 replica phases, Qwen serve smoke, GLM
# producer); this script only appends `--device-profile NAME`. Every arm runs the same command, so the
# arms differ only in the declared device. Outputs: DIR/<profile>-<n>.log, DIR/summary.txt (<= 60 lines).
# Run it inside your GPU turn (the turn script owns the card); it never selects a card by index.
set -euo pipefail

gpu_uuid="" out="" profiles="rtx-5090-32 rtx-5080-16" repeat=2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-uuid) gpu_uuid="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        --profiles) profiles="$2"; shift 2 ;;
        --repeat) repeat="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done
[[ -n "$gpu_uuid" && -n "$out" && $# -gt 0 ]] || { sed -n '2,13p' "$0" >&2; exit 2; }
mkdir -p "$out"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu_uuid"

summary="$out/summary.txt"
{
    echo "devprof-emulate $(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=$gpu_uuid repeat=$repeat"
    echo "command: $*"
} > "$summary"
for profile in $profiles; do
    for ((n = 1; n <= repeat; ++n)); do
        log="$out/$profile-$n.log"
        before=$(nvidia-smi --id="$gpu_uuid" --query-gpu=memory.used,clocks.sm --format=csv,noheader,nounits 2>/dev/null || echo "?")
        set +e
        "$@" --device-profile "$profile" > "$log" 2>&1
        rc=$?
        set -e
        after=$(nvidia-smi --id="$gpu_uuid" --query-gpu=memory.used,clocks.sm --format=csv,noheader,nounits 2>/dev/null || echo "?")
        echo "== $profile run $n rc=$rc smi_before=[$before] smi_after=[$after]" >> "$summary"
        # The budget / plan / refusal lines every bound route prints, and the route's own result lines.
        grep -aE 'memory budget|memory plan|refusing|minimum|tok/s|TTFT|prefill_ms|decode_ms' "$log" | cut -c1-300 | head -8 >> "$summary" || true
    done
done
echo "summary: $summary"
