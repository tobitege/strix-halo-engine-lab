CUDA weight-format migration groundwork

P10 is in progress. Dense execution already calls shared CUDA core code through
the source protocols and typed launch records. This change also introduces
numerical forms and scheduling defaults; it is not solely pending extraction.
Freezing the preceding executable and ISA, comparing them with the changed paths,
and collecting complete-workload evidence remain qualification gates. Synthetic
fixture inventories and host checks do not qualify these execution changes.

The host descriptor/reference, admission, resolver and image-lifecycle contracts
remain in the shared host layer. CUDA headers, resident views, instruction traits
and launch dispatch belong to the CUDA backend. Qwen owns tensor roles and graph
epilogues. Different IMMA, native floating-point and native FP4 arithmetic retain
distinct contracts.

Synthetic dense captures cover boundary rows, output tails, every output store,
aligned and unaligned segment boundaries, activation operands, register partials
and complete production outputs. Separate processes are required for cached
launch-policy alternatives. The legacy expert capture matrix covers the old
format combinations and native IQ execution. No capture fixture qualifies a new
FAST arithmetic contract by itself.

Run the capture/compare and ISA tools in scripts/bench/wfmt before promoting a
source binding. Existing payload bytes and arithmetic are mandatory baselines.
