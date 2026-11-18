#!/usr/bin/env bash
# Served RTX routes (one GPU workstation, CUDA): decode classes at 1/4/8 chats through the shared serve runner.
#
#   rtx/serve-route.sh --route qwen --bin strix-cuda-qwen-serve --model GGUF --port 8093 [--slots 8]
#       [--classes prose,edit,json] [--concurrency 1,4,8] [--arm 'NAME|ARGS|ENV'] [-- extra server args]
#   rtx/serve-route.sh --route custom --port P --tag T -- SERVER_CMD...       (any OpenAI-compatible RTX server)
#
# Qwen: whole model on one RTX (exact CUDA prefill, exact greedy MTP decode); give the table / rope / MTP sidecar /
# draft-vocab arguments the server's --help lists after "--", e.g. -- --tables T --rope-table R --mtp-sidecar S
# --draft-vocab V --max-positions 34816. Select the GPU with CUDA_VISIBLE_DEVICES (by UUID when several are
# present) and STRIX_DEVICES (the device declaration) in the caller's environment.
# Sampling uses nvidia-smi (power, SM clock, temperature, utilisation, memory used); host memory and kfd evictions
# are AMD-node checks and are reported as not sampled. Output: OUT/<tag>/... receipts, compared with abab.py.
# The RTX + pair DeepSeek split is rtx/ds41-rtx-pair-batch.sh; the RTX DeepSeek prefill service is measured by
# tools/ds41_prefill_service_client.py against strix-cuda-ds41-prefill-service.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/../common/runlib.sh"
route='' bin='' model='' port=8093 slots=8 classes=prose,edit,json conc=1,4,8 arm='rtx||' tag='' extra=()
while (($#)); do
    case $1 in
        --route) route=$2 ;; --bin) bin=$2 ;; --model) model=$2 ;; --port) port=$2 ;; --slots) slots=$2 ;;
        --classes) classes=$2 ;; --concurrency) conc=$2 ;; --arm) arm=$2 ;; --tag) tag=$2 ;;
        --) shift; extra=("$@"); break ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) kit_die "unknown argument $1" ;;
    esac
    shift 2
done
export KIT_SAMPLER_KIND=${KIT_SAMPLER_KIND:-nvidia}
IFS='|' read -r name args envs <<<"$arm"; read -ra A <<<"$args"; read -ra E <<<"$envs"
case $route in
qwen)
    [[ -n $bin && -e $model ]] || kit_die 'qwen route needs --bin and --model'
    bash "$here/../common/serve-decode.sh" --tag "${tag:-qwen-$name}" --arm "qwen-$name" --port "$port" --chat \
        --body-extra '{"enable_thinking":false}' --classes "$classes" --concurrency "$conc" -- \
        env "${E[@]}" "$bin" --model "$model" --port "$port" --http-slots "$slots" "${A[@]}" "${extra[@]}" ;;
custom)
    [[ -n $tag && ${#extra[@]} -gt 0 ]] || kit_die 'custom route needs --tag and -- SERVER_CMD'
    bash "$here/../common/serve-decode.sh" --tag "$tag" --arm "$tag" --port "$port" --chat --classes "$classes" \
        --concurrency "$conc" -- "${extra[@]}" ;;
*) sed -n '2,16p' "$0"; exit 64 ;;
esac
