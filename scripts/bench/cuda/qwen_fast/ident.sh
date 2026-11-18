#!/usr/bin/env bash
# ident.sh: byte identity of teacher-forced logit dumps between arms (execution evidence; kernel admission must be qualified separately).
# Each arm runs qwen_tf_logits once (one process, all prompts) and every dump file is compared with the first arm's.
#
#   ident.sh --tf TF --model M --tables T --rope-table R [--mtp-sidecar G] --out DIR [--rows 64] [--max-positions N] \
#            --prompt NAME=ids.u32 [--prompt ...] --arm 'NAME|ENV=v ENV2=w|--flags' [--arm ...]
#
# Arm: name, space-separated environment assignments (may be empty), qwen_tf_logits flags (e.g. --profile fast).
# Prints one line per (arm, file): IDENTICAL <bytes> <sha256 prefix> or DIFFERENT; exit 0 = all identical, 1 = a
# difference, 2 = an arm failed. No host names or paths are built in.
set -eu
set -o pipefail
tf= model= tables= rope= mtp= out= rows=64 pos=
prompts=() arms=()
while [ $# -gt 0 ]; do
    [ $# -ge 2 ] || { echo "ident.sh: missing argument for $1" >&2; exit 2; }
    case $1 in
        --tf) tf=$2; shift 2;;
        --model) model=$2; shift 2;;
        --tables) tables=$2; shift 2;;
        --rope-table) rope=$2; shift 2;;
        --mtp-sidecar) mtp=$2; shift 2;;
        --out) out=$2; shift 2;;
        --rows) rows=$2; shift 2;;
        --max-positions) pos=$2; shift 2;;
        --prompt) prompts+=(--prompt "$2"); shift 2;;
        --arm) arms+=("$2"); shift 2;;
        *) echo "ident.sh: unknown argument $1" >&2; exit 2;;
    esac
done
[ -n "$tf" ] && [ -n "$model" ] && [ -n "$out" ] && [ ${#prompts[@]} -gt 0 ] && [ ${#arms[@]} -ge 2 ] ||
    { echo "ident.sh: need --tf --model --out, a --prompt and two --arm" >&2; exit 2; }
extra=()
[ -n "$tables" ] && extra+=(--tables "$tables")
[ -n "$rope" ] && extra+=(--rope-table "$rope")
[ -n "$mtp" ] && extra+=(--mtp-sidecar "$mtp")
[ -n "$pos" ] && extra+=(--max-positions "$pos")
# Restrict labels to portable single path components, including on case-insensitive hosts.
# Validate before launching anything; prompt labels also determine dump filenames.
valid_name() { [[ $1 =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; }
names=() prompt_names=()
for a in "${arms[@]}"; do
    [[ $a == *'|'*'|'* ]] || { echo "ident.sh: malformed arm" >&2; exit 2; }
    n=${a%%|*}; rest=${a#*|}; flags=${rest#*|}
    valid_name "$n" && [[ $flags != *'|'* ]] || { echo "ident.sh: unsafe arm name/flags" >&2; exit 2; }
    for old in "${names[@]}"; do
        [[ ${old,,} != "${n,,}" ]] || { echo "ident.sh: duplicate arm $n" >&2; exit 2; }
    done
    names+=("$n")
done
for ((i=1; i<${#prompts[@]}; i+=2)); do
    spec=${prompts[i]}; n=${spec%%=*}
    valid_name "$n" && [[ $spec == *=?* ]] || { echo "ident.sh: unsafe prompt" >&2; exit 2; }
    for old in "${prompt_names[@]}"; do
        [[ ${old,,} != "${n,,}" ]] || { echo "ident.sh: duplicate prompt $n" >&2; exit 2; }
    done
    prompt_names+=("$n")
done
[[ $rows =~ ^[1-9][0-9]*$ ]] || { echo "ident.sh: invalid rows" >&2; exit 2; }
mkdir -p "$out"
run=$(mktemp -d "$out/ident.XXXXXXXX")
echo "identity run: $run"
for i in "${!arms[@]}"; do
    a=${arms[i]}; n=${names[i]}; rest=${a#*|}; envs=${rest%%|*}; flags=${rest#*|}
    dir=$run/arm-$i
    mkdir "$dir"
    # Preserve word splitting for the documented arm syntax, but never expand globs.
    read -r -a env_args <<< "$envs"
    read -r -a flag_args <<< "$flags"
    if env "${env_args[@]}" "$tf" --model "$model" "${extra[@]}" "${flag_args[@]}" "${prompts[@]}" --rows "$rows" --runs 1 --out "$dir" --prefix q \
        > "$dir/run.log" 2>&1; then
        :
    else
        rc=$?; echo "arm $n rc $rc (see $dir/run.log)"; exit 2
    fi
    # A new directory proves freshness. Require the complete execution receipt and
    # exact dump inventory/size (Qwen vocabulary is 248320 FP32 values per row).
    if ! python3 - "$dir" "$rows" "${prompt_names[@]}" <<'PY'
import json
import pathlib
import sys
try:
    directory, rows, *names = sys.argv[1:]
    directory = pathlib.Path(directory)
    receipt = directory / "effective-config.json"
    if receipt.is_symlink():
        raise ValueError("symlink receipt")
    executions = json.loads(receipt.read_text())["executions"]
    if not isinstance(executions, list) or len(executions) != len(names):
        raise ValueError("incomplete executions")
    if sorted(e["prompt"] for e in executions) != sorted(names):
        raise ValueError("execution prompt mismatch")
    for e in executions:
        if e["run"] != 0 or type(e["rows"]) is not int or e["rows"] <= 0:
            raise ValueError("invalid execution")
        if not isinstance(e["forms"], dict) or not e["forms"] or any(type(v) is not bool for v in e["forms"].values()):
            raise ValueError("missing execution forms")
    expected = {f"q-{name}-tf-r0.f32" for name in names}
    if {p.name for p in directory.glob("*.f32")} != expected:
        raise ValueError("incomplete or unexpected dumps")
    for name in expected:
        p = directory / name
        if p.is_symlink() or not p.is_file() or p.stat().st_size != int(rows) * 248320 * 4:
            raise ValueError(f"invalid dump size/type: {name}")
except (OSError, ValueError, KeyError, TypeError) as error:
    print(f"ident.sh: invalid execution evidence: {error}", file=sys.stderr)
    sys.exit(2)
PY
    then
        exit 2
    fi
done
status=0
ref=${names[0]}
for ((i=1; i<${#names[@]}; ++i)); do
    n=${names[i]}
    for p in "${prompt_names[@]}"; do
        b=q-$p-tf-r0.f32; f=$run/arm-0/$b
        if cmp -s "$f" "$run/arm-$i/$b"; then
            echo "$ref == $n $b IDENTICAL $(stat -c %s "$f") B $(sha256sum "$f" | cut -c1-16)"
        else
            echo "$ref vs $n $b DIFFERENT"; status=1
        fi
    done
done
exit $status
