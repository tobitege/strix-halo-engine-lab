#!/usr/bin/env bash
# FAST-vs-exact quality gate of the CUDA Qwen engine (KL / top-1 / dNLL), over teacher-forced logit dumps.
#
#   gate.sh --tf PATH/qwen_tf_logits --model GGUF --tables DIR --rope-table FILE --out DIR \
#           --prompt NAME=tokens.u32 [--prompt ...] [--mtp-sidecar GGUF] [--rows 64] [--fast-args "--hc-mma off"] \
#           [--wrap "GPU_WRAPPER ..."] [--max-kl 0.02] [--min-top1 0.93] [--max-dnll 0.01] [-- extra qwen_tf_logits args]
#
# Arm "exact" runs --profile exact, arm "fast" runs --profile fast (+ --fast-args); both through the same shared option
# parser as the server. --wrap is a command prefix for a GPU-serialising wrapper (also: GATE_WRAP). Each arm is sealed with
# tools/strix-gate/fast_logit_gate.py and the arms are compared; exit 0 = gate passed, 1 = failed, 2 = identity/input error.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
gate_py="${GATE_PY:-$here/../../../../tools/strix-gate/fast_logit_gate.py}"
python_bin="${PYTHON:-python3}"
tf="" out="" rows=64 wrap="${GATE_WRAP:-}" fast_args="" max_kl=0.02 min_top1=0.93 max_dnll=0.01
engine=() prompts=()
while [ $# -gt 0 ]; do
    case "$1" in
        --tf) tf=$2; shift 2;;
        --out) out=$2; shift 2;;
        --rows) rows=$2; shift 2;;
        --wrap) wrap=$2; shift 2;;
        --fast-args) fast_args=$2; shift 2;;
        --max-kl) max_kl=$2; shift 2;;
        --min-top1) min_top1=$2; shift 2;;
        --max-dnll) max_dnll=$2; shift 2;;
        --prompt) prompts+=("$2"); engine+=(--prompt "$2"); shift 2;;
        --model|--tables|--rope-table|--mtp-sidecar|--draft-vocab|--max-positions) engine+=("$1" "$2"); shift 2;;
        --) shift; engine+=("$@"); break;;
        *) echo "unknown option $1" >&2; exit 2;;
    esac
done
if [ -z "$tf" ] || [ -z "$out" ] || [ ${#prompts[@]} -eq 0 ]; then sed -n 2,10p "$0" >&2; exit 2; fi
[ -x "$tf" ] || { echo "not executable: $tf" >&2; exit 2; }
mkdir -p "$out"
# Profile arms must not inherit numerical overrides or profile-export provenance.
for key in STRIX_QRTX_HC_MMA STRIX_QRTX_MTP_MMA STRIX_QRTX_QSA_MMA STRIX_QRTX_MOE_MMA \
           STRIX_QPF2K_MOE_ROUTE_BF16 STRIX_QPF2K_QSA_BUDGET STRIX_QPF2K_GDN_FP16_CHUNK \
           STRIX_QWEN_PROFILE_ALLOW_PENDING ${!STRIX_QWEN_PROFILE_OWNED_@}; do
    unset "$key"
done
stamp=$(date +%Y%m%d%H%M%S)
seal_prompts=() cmp_tokens=()
for p in "${prompts[@]}"; do seal_prompts+=(--prompt "$p"); cmp_tokens+=(--tokens "$p"); done
run_arm() {  # arm profile extra...
    local arm=$1 profile=$2; shift 2
    local dir="$out/$arm"
    rm -rf "$dir"; mkdir -p "$dir"
    echo "== arm $arm (profile $profile $*)"
    # $wrap is a word-split command prefix on purpose.
    # shellcheck disable=SC2086
    $wrap "$tf" "${engine[@]}" --profile "$profile" "$@" --rows "$rows" --out "$dir" --prefix q > "$out/$arm.log" 2>&1 \
        || { tail -n 20 "$out/$arm.log" >&2; echo "arm $arm failed (see $out/$arm.log)" >&2; exit 2; }
    tail -n 12 "$out/$arm.log"
    # This file comes from the executing process, not a separate inspection launch.
    local effective
    effective=$("$python_bin" - "$dir/effective-config.json" "$arm" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f:
    c = json.load(f)
forms = c['forms']
names = ['hc_mma', 'mtp_mma', 'qsa_mma', 'moe_mma']
if set(forms) != set(names + ['moe_route_bf16']) or any(type(v) is not bool for v in forms.values()):
    raise SystemExit('invalid resolved forms')
fast = any(forms.values())
contract = 'qwen-cuda;prefill=exact'
if fast:
    contract = ('qwen-cuda;prefill=fast;forms=' + (','.join(n for n in names if forms[n]) or 'none')
                + ';moe_route_bf16=' + str(int(forms['moe_route_bf16'])))
ids = ['qwen-cuda-hc-f16mma-v1', 'qwen-cuda-mtp-f16mma-v1',
       'qwen-cuda-qsa-f16mma-v1', 'qwen-cuda-moe-i8fma-v1']
if (c['entry'] != 'serve' or c['arithmetic_contract'] != contract
        or c['form_identities'] != ','.join(i for n, i in zip(names, ids) if forms[n])
        or fast != (sys.argv[2] == 'fast')):
    raise SystemExit('arm does not match resolved arithmetic contract')
executions = c.get('executions')
if not isinstance(executions, list) or not executions:
    raise SystemExit('missing completed execution receipts')
targets = ['hc_mma', 'qsa_mma', 'moe_mma', 'moe_route_bf16']
for execution in executions:
    active = execution['forms']
    if (set(active) != set(forms) or any(type(v) is not bool for v in active.values())
            or type(execution['rows']) is not int or execution['rows'] <= 0
            or any(active[n] and not forms[n] for n in forms)
            or any(active[n] != forms[n] for n in targets)):
        raise SystemExit('invalid executed forms')
    if any(active[n] for n in targets) != (sys.argv[2] == 'fast'):
        raise SystemExit('arm has no executed target-changing form (MTP-only is not a target-logit gate)')
print(json.dumps(c, sort_keys=True, separators=(',', ':')))
PY
    ) || { echo "invalid effective configuration for $arm" >&2; exit 2; }
    "$python_bin" "$gate_py" seal --dir "$dir" --rank 0 --arm "$arm" --run-id "$arm-$stamp" --binary "$tf" --rows "$rows" \
        --prefix q "${seal_prompts[@]}" --env "profile=$profile" --env "extra=$*" --env "effective_config=$effective"
}
# Last CLI values win, including over common arguments after --.
run_arm exact exact --hc-mma off --mtp-mma off --qsa-mma off --moe-mma off \
    --prefill-moe-route-bf16 off --qsa-budget off --gdn-fp16-chunk off --output-head-profile off
# shellcheck disable=SC2086
run_arm fast fast $fast_args
echo "== compare exact (reference) vs fast"
set +e
"$python_bin" "$gate_py" compare "$out/exact" "$out/fast" --ranks 0 "${cmp_tokens[@]}" \
    --max-kl "$max_kl" --min-top1 "$min_top1" --max-dnll "$max_dnll" --json "$out/verdict.json"
rc=$?
case $rc in
    0) echo "VERDICT: PASS (KL <= $max_kl, top-1 >= $min_top1, dNLL <= $max_dnll)";;
    1) echo "VERDICT: FAIL";;
    *) echo "VERDICT: no verdict (identity or input error)";;
esac
exit $rc
