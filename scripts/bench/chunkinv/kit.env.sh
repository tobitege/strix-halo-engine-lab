# Sourced by the chunkinv gate kit. Every value can be overridden from the environment; nothing here is node-specific
# beyond the defaults. KIT = run root on rank 0's node (the same path must exist on the peer).
KIT=${KIT:-$HOME/ai-cluster-lab/chunkinv}
BIN=${BIN:-$KIT/bin/gate/strix-minimax-rank}      # gate binary = clean build + diag_attn_ulp_patch.py
MODEL=${MODEL:?set MODEL to the MiniMax M3.1 model directory}
PROMPTS=${PROMPTS:?set PROMPTS to the directory holding <name>.u32 prompt files}
PEER=${PEER:?set PEER to the rank-1 ssh host}
GATE=${GATE:-$KIT/fast_logit_gate.py}             # copy of tools/strix-gate/fast_logit_gate.py (same path on the peer)
S2="ssh -n -o BatchMode=yes -o ConnectTimeout=5 $PEER"
