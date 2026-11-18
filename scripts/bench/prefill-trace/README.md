# prefill-trace: window-loop census from a kernel trace

`window_census.py` reads one rocprofv3 kernel trace (`rocprofv3 --kernel-trace --output-format csv`) of a prefill
process and reports, for one run of it (default: the last, i.e. the warm run of a `--repeat 2` process):

- the per-window launch loop: number of clusters, launches, span, GPU busy-union and idle inside it;
- inclusive time, count and CTAs per kernel kind, and the launches per hardware queue;
- how many compute units the CU-bound kernel (`--fill`, default the shared flash core `flash_rows`) leaves free:
  a sweep of the CTAs in flight capped at `--cus`, split into the loop edges and its interior, then again with
  every kernel whose LDS cannot share a CU with a fill CTA (`--cu-lds`) counted as occupying one.

Both CU figures are lower bounds of free CU time under the one-fill-CTA-per-CU assumption
(running kernels' CTAs are counted as resident). They are not speedup ceilings: CTA repacking
can recover wave-tail capacity hidden by these counts, even when the reported free time is zero.
A speedup ceiling needs an independently justified occupancy/work model. Do not use these
figures alone to reject a scheduling proposal.

A cluster is every nonempty consecutive sequence of matching launches in start-time order;
a nonmatching launch ends it. Timing uses the union of cluster spans. GPU busy/idle and the
LDS sweep include all run kernels clipped to that union, including other queues and kernels
started before a cluster. The fill sweep likewise includes all overlapping fill launches.
Cluster counts and per-kind inclusive times remain separate launch metrics. Edges are the union
of each cluster's clipped first/last `--edge-us` intervals; the interior is the complement
inside the loop union. Overlaps and touching edges count once. Full symbols are used for
matching, grouping and run splitting; only the displayed kind column is shortened.

```
python3 window_census.py r0_kernel_trace.csv [--run N] [--split embedding_kernel] [--kinds a,b,...]
                         [--fill flash_rows] [--cus 40] [--cu-lds 65536] [--edge-us 300]
python3 -m unittest test_window_census.py
```

`--kinds` defaults to the DS41 window set (index, select, flash or exact attention, combine, empty top-k memset); GLM
and MiniMax window loops pass their own kernel names. `--cus` and `--cu-lds` come from the device declaration
(gfx1151: 40 CUs, 64 KiB LDS per CU). The CLI echoes the selected kernel and capacity assumptions;
these defaults do not detect the traced family or device. Set the overrides explicitly for
other families/devices. Empty or malformed traces, invalid run indices and invalid capacity/edge
arguments produce CLI errors. No host names or paths: the trace path is the only input.

## kind_excess.py: long-context excess per phase and kernel kind

`python3 kind_excess.py SHORT_kernel_trace.csv LONG_kernel_trace.csv [--mult 4] [--top 12] [--phase NAME=REGEX ...]`

Both traces must contain kernel rows with `Kernel_Name`, `Start_Timestamp` and
`End_Timestamp` headers, nonempty normalized names and integer timestamps satisfying
`0 <= start <= end`. Invalid input fails with a file/line diagnostic before any
comparison is printed. Aggregation and phase regexes use full normalized names;
only display labels are shortened, and colliding labels are shown in full.
compares two single-prefill traces whose prompts differ by `--mult` in length (e.g. 32K and 128K) and prints, per
phase and per kernel kind, the time in each and `LONG - mult x SHORT`: the work that grows faster than the prompt
(attention, indexer scoring and selection). Default phases are the Qwen prefill kernels; pass `--phase` for other
families. CPU only; the trace paths are the only inputs.
