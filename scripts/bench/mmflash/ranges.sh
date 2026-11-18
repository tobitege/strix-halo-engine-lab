#!/bin/bash
# ranges.sh REF ARM... : MiniMax MSA flash layer-range qualification in one detached run. Prints its fresh
# ranges.XXXXXX directory (summary, timing, status; done marker only for complete evidence). ARM syntax:
#   e<tag>            exact arm (STRIX_MM_PF_MSA_FLASH=0)
#   n<L>s<seed>       one-ulp null on the attention output of layer L (the gate's onset control; diagnostic hook binary)
#   f<A>-<B>m<11|12>  candidate: MSA flash on layers A..B, mode 11 = STRIX_MM_PF_MSA_FLASH=1 (one-term), 12 = 2 (two-term)
#   d<A>-<B>s<seed>g<mag>  diagnostic multi-site null: +-mag ulps on every layer A..B (absolute comparison only, never a floor)
#   h<arm>            the e / f arm plus the binary16 TP2 exchange (STRIX_TP2_EXCHANGE_FMT=f16, FAST onset layer 0): floor
#                     rule against the n0s* nulls, e.g. he (exchange only) or hf0-59m11 (MSA flash 0-59 + exchange)
# Every arm is one pair ticket ($MMF_TURN LABEL CMD...), two lanes. Candidates get the floor rule (onset nulls n<A>s*,
# --dnll-rule floor --engine minimax --profile fast-prefill, needs the MiniMax onset rule in the gate tool) and the
# absolute comparison; summary.py writes the table. Environment: as seal.sh, plus MMF_TURN, MMF_SESSIONS (default
# cpros8k,ccode8k,cdata8k), MMF_CAP (8256), MMF_SECS (105), MMF_LABEL (ticket label prefix).
set -uo pipefail
[ "$#" -ge 2 ] || exit 64
REF=$1; shift
: "${MMF_RUN:?}" "${MMF_TURN:?}" "${MMF_GATE:?}" "${MMF_PROMPTS:?}"
H=$(dirname "$0")
mkdir -p "$MMF_RUN" || exit 2
K=$(mktemp -d "$(dirname "$MMF_RUN")/ranges.XXXXXX") || exit 2
S=$K/ranges.summary
echo "qualification $K"
SESS=${MMF_SESSIONS:-cpros8k,ccode8k,cdata8k} CAP=${MMF_CAP:-8256} SECS=${MMF_SECS:-105} LBL=${MMF_LABEL:-mmflash}
mkdir -p "$K/tk" "$K/cmp" || exit 2
: > "$S"; : > "$K/cmp/expected.tsv"
say() { echo "$*" >> "$S"; }
declare -A flash_ranges=() index_ranges=() flash_modes=() onsets=()
parse_candidate() {
  local a=$1
  [[ $a =~ ^f([0-9]+)-([0-9]+)m(11|12)(i([0-9]+)-([0-9]+))?$ ]] || return 1
  flash_ranges[$a]=${BASH_REMATCH[1]}-${BASH_REMATCH[2]}
  flash_modes[$a]=$(( ${BASH_REMATCH[3]} - 10 ))
  index_ranges[$a]=${BASH_REMATCH[4]#i}
  local onset=$((10#${BASH_REMATCH[1]}))
  if [[ -n ${BASH_REMATCH[4]} ]] && (( 10#${BASH_REMATCH[5]} < onset )); then onset=$((10#${BASH_REMATCH[5]})); fi
  onsets[$a]=$onset
}
nulls_for() {
  local a=$1 n; shift
  for n in "$@"; do case $n in n${onsets[$a]}s*) printf '%s\n' "$n";; esac; done
}
env_of() {
  case $1 in
    e*) echo "STRIX_MM_PF_MSA_FLASH=0";;
    n*) local x=${1#n}; echo "STRIX_MM_PF_MSA_FLASH=0 STRIX_MM_DIAG_ATTN_ULP=${x%%s*}:${x##*s}";;
    f*) local ix=${index_ranges[$1]}
        echo "STRIX_MM_PF_MSA_FLASH=${flash_modes[$1]} STRIX_MM_PF_MSA_FLASH_LAYERS=${flash_ranges[$1]}${ix:+ STRIX_MM_PF_IDX_WMMA=1 STRIX_MM_PF_IDX_WMMA_LAYERS=$ix}";;
    h*) local b; b=$(env_of "${1#h}") || return 1; echo "$b STRIX_TP2_EXCHANGE_FMT=f16";;
    d*) local x=${1#d} r s g; r=${x%%s*}; s=${x#*s}; g=${s##*g}; s=${s%%g*}; echo "STRIX_MM_PF_MSA_FLASH=0 STRIX_MM_DIAG_ATTN_ULP=$r:$s:$g";;
    *) return 1;;
  esac
}
[[ $REF =~ ^[A-Za-z0-9._-]+$ ]] && [ "$REF" != . ] && [ "$REF" != .. ] || exit 64
declare -A seen=()
for a in "$@"; do
  [[ $a =~ ^(h?e[A-Za-z0-9._-]*|n[0-9]+s[0-9]+|h?f[0-9]+-[0-9]+m(11|12)(i[0-9]+-[0-9]+)?|d[0-9]+-[0-9]+s[0-9]+g[0-9]+)$ ]] || exit 64
  [ -z "${seen[$a]:-}" ] || exit 64
  seen[$a]=1
  case $a in f*) parse_candidate "$a" || exit 64;;
    hf*) parse_candidate "${a#h}" || exit 64; onsets[$a]=0;;
    he*) onsets[$a]=0;; esac
  case $a in f*|d*|h*) printf '%s\t%s\tabsolute\n' "$a-abs" "$a" >> "$K/cmp/expected.tsv";; esac
  case $a in f*|h*) printf '%s\t%s\tfloor\n' "$a" "$a" >> "$K/cmp/expected.tsv";; esac
done
read -r -a TURN <<< "$MMF_TURN"
arm() {
  local a=$1 envs
  read -r -a envs <<< "$(env_of "$a")"
  if "${TURN[@]}" "$LBL-$a" bash "$H/arm.sh" "$a" "$SECS" "$CAP" "$SESS" "${envs[@]}" > "$K/tk/$a.out" 2>&1 &&
      bash "$H/seal.sh" "$a" "$SESS" >> "$K/tk/$a.out" 2>&1 &&
      python3 "$H/summary.py" timing "$MMF_RUN/$a" > "$K/tk/$a.timing" 2>> "$K/tk/$a.out"; then
    echo 0 > "$K/tk/$a.rc"
  else echo 2 > "$K/tk/$a.rc"; fi
}
say "start $(date +%T) ref $REF arms $*"
L0=(); L1=(); i=0; for a in "$@"; do if [ $((i % 2)) = 0 ]; then L0+=("$a"); else L1+=("$a"); fi; i=$((i + 1)); done
( for a in "${L0[@]}"; do arm "$a"; done ) &
( sleep 20; for a in "${L1[@]}"; do arm "$a"; done ) &
wait
say "tickets done $(date +%T)"
T=(); IFS=, read -r -a prompts <<< "$SESS"
for p in "${prompts[@]}"; do T+=(--tokens "$p=$MMF_PROMPTS/$p.u32"); done
ok() { [ "$(cat "$K/tk/$1.rc" 2>/dev/null)" = 0 ] && [ "$(cat "$MMF_RUN/$1/seal.verdict" 2>/dev/null)" = OK ]; }
status=0; : > "$K/timing.txt"
for a in "$@"; do
  if ok "$a"; then cat "$K/tk/$a.timing" >> "$K/timing.txt"
  else say "  $a: INCOMPLETE (see tk/$a.out)"; status=2; fi
done
compare() {
  local name=$1; shift
  nice python3 "$MMF_GATE" compare "$@" --json "$K/cmp/$name.json" > "$K/cmp/$name.txt" 2>&1
  echo "$?" > "$K/cmp/$name.rc"
}
for a in "$@"; do
  case $a in f*|d*|h*) ;; *) continue;; esac
  ok "$a" || { say "  $a: not sealed"; continue; }
  compare "$a-abs" "$MMF_RUN/$REF" "$MMF_RUN/$a" --ranks 0 1 "${T[@]}"
  [ "${a#f}" = "$a" ] && [ "${a#h}" = "$a" ] && continue
  nl=(); missing_null=0
  while IFS= read -r n; do
    nl+=(--null-arm "$MMF_RUN/$n"); ok "$n" || missing_null=1
  done < <(nulls_for "$a" "$@")
  [ "$missing_null" = 0 ] || { say "  $a: INCOMPLETE null population"; continue; }
  compare "$a" "$MMF_RUN/$REF" "$MMF_RUN/$a" --ranks 0 1 "${T[@]}" --dnll-rule floor --engine minimax \
      --profile fast-prefill "${nl[@]}"
done
python3 "$H/summary.py" table "$K/cmp" "$K/timing.txt" "$MMF_RUN" "$REF" >> "$S" 2>&1
rc=$?; [ "$rc" -le 2 ] || rc=2
[ "$rc" -le "$status" ] || status=$rc
say "status $status $(date +%T) (0 pass, 1 completed gate failure, 2 incomplete)"
echo "$status" > "$K/ranges.status"
[ "$status" -eq 2 ] || touch "$K/ranges.done"
exit "$status"
