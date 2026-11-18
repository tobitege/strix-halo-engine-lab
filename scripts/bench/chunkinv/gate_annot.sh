#!/bin/bash
# gate_annot.sh LOG : the diagnostic hook prints no prompt; append prompt "NAME" from the session line that follows
# (in place, idempotent). Run before sealing so the witness can bind each null-control event to its prompt.
f=$1; grep -a -q 'values [0-9]* prompt "' $f && exit 0
awk '/^DIAG attn ulp:/ {b[++n]=$0; next} { if (match($0, /^rank [01] [A-Za-z0-9_.-]+: prompt [0-9]+ tokens/)) { split($0, a, " "); nm=a[3]; sub(/:$/, "", nm); for (i=1;i<=n;i++) print b[i] " prompt \"" nm "\""; n=0 } print } END {for (i=1;i<=n;i++) print b[i]}' $f > $f.tmp && mv $f.tmp $f
