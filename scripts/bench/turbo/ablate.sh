#!/usr/bin/env bash
# Leave-one-out ablation of a TURBO lever manifest, packed into GPU tickets.
#
#   ablate.sh arms FAMILY [--pending] [--only id,id]       print the arms, one 'NAME|ARGS|ENV' per line
#   ablate.sh run  FAMILY --out DIR --kit KIT [--queue CMD] [--per-ticket N] [--pending] [--only ids]
#                  [--rounds R] -- KITARGS...               one ticket per group of arms (default arm always first)
#
# FAMILY = a scripts/bench/turbo/<family>.levers manifest (id|group|env|args|tag|state|note|off-env). ALL takes the first lever
# of every group whose state is "main" (plus "pending" with --pending); an alternative of a group is its own arm
# 'alt-<id>' (ALL with that lever replacing the group's first). Arms: default, all, no-<id> for each lever of ALL, alt-*.
# KIT is a path under scripts/bench/ (e.g. qwen/length-matrix); KITARGS go to it after the generated --arm words.
# QUEUE is the ticket wrapper (default: gpu-turn.sh LABEL); a ticket runs common/ticket.sh DIR/gN KIT ... and the label
# is turbo3-bundle-<family>-gN. Hosts, models and binaries come only from KITARGS and the environment of the kit.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mode=${1:-help}; fam=${2:-}; shift 2 2>/dev/null || true
pending=0 only='' out='' kit='' queue='' per=5 rounds=3
while (($#)); do
    case $1 in
        --pending) pending=1; shift ;; --only) only=$2; shift 2 ;; --out) out=$2; shift 2 ;; --kit) kit=$2; shift 2 ;;
        --queue) queue=$2; shift 2 ;; --per-ticket) per=$2; shift 2 ;; --rounds) rounds=$2; shift 2 ;;
        --) shift; break ;; *) echo "ablate: unknown $1" >&2; exit 64 ;;
    esac
done
man=$here/$fam.levers
[[ -f $man ]] || { echo "ablate: no manifest $man" >&2; exit 64; }
ids=() grp=() env=() arg=() state=() off=()
while IFS='|' read -r i g e a t s n o; do
    [[ -z $i || $i == \#* ]] && continue
    [[ $s == main || $pending == 1 ]] || continue
    [[ -n $only && ",$only," != *",$i,"* ]] && continue
    [[ -n $o ]] || { echo "ablate: missing OFF assignments for $i" >&2; exit 64; }
    ids+=("$i") grp+=("$g") env+=("$e") arg+=("$a") state+=("$s") off+=("$o")
done <"$man"
seen=' '; first=()
for k in "${!ids[@]}"; do [[ $seen == *" ${grp[k]} "* ]] || { seen+="${grp[k]} "; first+=("$k"); }; done
armof() { # armof NAME skip_index [replace_index]
    local e='' a='' k
    for k in "${first[@]}"; do
        if [[ $k == "$2" ]]; then e+="${off[k]} "; continue; fi
        [[ -n ${3:-} && ${grp[k]} == "${grp[$3]}" ]] && continue
        e+="${env[k]} " a+="${arg[k]} "
    done
    [[ -n ${3:-} ]] && { e+="${env[$3]} " a+="${arg[$3]} "; }
    a=$(echo $a); e=$(echo $e); echo "$1|$a|$e"
}
emit() {
    echo 'default||'
    armof all -1
    for k in "${first[@]}"; do armof "no-${ids[k]}" "$k"; done
    for k in "${!ids[@]}"; do [[ " ${first[*]} " == *" $k "* ]] || armof "alt-${ids[k]}" -1 "$k"; done
}
case $mode in
    arms) emit ;;
    run)
        # Bound the decimal input before shell arithmetic (including overflow / octal traps).
        [[ $per =~ ^[1-9][0-9]{0,8}$ ]] && ((per >= 2)) || {
            echo 'ablate: --per-ticket must be an integer >=2 (at most 9 digits)' >&2; exit 64;
        }
        ((pending == 0)) || { echo 'ablate: pending controls are inspection-only; qualify their adapters before running' >&2; exit 64; }
        ((${#first[@]})) || { echo "ablate: $fam has no runnable bundle levers" >&2; exit 64; }
        [[ -n $out && -n $kit ]] || { echo 'ablate run needs --out and --kit' >&2; exit 64; }
        mapfile -t A < <(emit); d=${A[0]}; rest=("${A[@]:1}"); g=0
        for ((i = 0; i < ${#rest[@]}; i += per - 1)); do
            words=(--arm "$d"); for a in "${rest[@]:i:per-1}"; do words+=(--arm "$a"); done
            label=turbo3-bundle-$fam-g$g; dir=$out/g$g
            wrapper=${queue:-gpu-turn.sh}
            $wrapper "$label" bash "$here/../common/ticket.sh" "$dir" "$kit" "${words[@]}" --rounds "$rounds" "$@"
            rc=$?
            if ((rc != 0)); then
                echo "ablate: failed group $g ($label), output $dir, queue status $rc; retry this group before continuing" >&2
                exit "$rc"
            fi
            g=$((g + 1))
        done ;;
    *) sed -n '2,14p' "$0" ;;
esac
