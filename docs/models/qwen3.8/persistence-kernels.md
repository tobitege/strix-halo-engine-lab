# Qwen attention, draft and recurrent dispatch experiments

These independent runtime switches default to OFF. They do not inherit
`STRIX_DISPATCH_FUSE` and do not change a numerical profile, model asset,
selection budget, acceptance rule or persistent state layout.

Exception by window size: the two verify-window tails (`STRIX_QPERSIST_QSA_TAIL`,
`STRIX_QPERSIST_GDN_TAIL`) are bit-identical to the separate launches but not
faster at every size. With the variable unset they follow the decode cycle's
dispatch-fuse resolution (`STRIX_DISPATCH_FUSE`) at the sizes where the fused
kernel costs less than the launches it replaces: the GDN tail in 2-row
windows, the QSA tail in 2- and 4-row windows. `=1` takes every admitted
size, any other value none. Kernel trace, fused versus separate launches
(us per layer): GDN tail 36.4 vs 37.5 (2 rows), 47.6 vs 43.3 (4), 84.0 vs
60.1 (8); QSA tail 64.1 vs 66.8 (2), 64.2 vs 73.0 (4), 117.6 vs 86.9 (8).
The draft-side uses of `STRIX_QPERSIST_QSA_TAIL` stay opt-in.

| Switch | Candidate behavior |
| --- | --- |
| `STRIX_QPERSIST_DRAFT_SCORE=1` | Runs the MTP single-row QK score, including draft and teacher forward, with the existing window wave kernel: four selected slots per wave, eight waves per workgroup. |
| `STRIX_QPERSIST_QSA_TAIL=1` | Runs softmax, chronological value accumulation and sigmoid gating in one workgroup per head, including the existing A8 epilogue when used by a window. |
| `STRIX_QPERSIST_MTP_CACHE=1` | Validates a complete draft index/KV row and publishes its BF16 cache bytes in one workgroup. |
| `STRIX_QPERSIST_GDN_TAIL=1` | Owns a complete 128-column GDN value head with sixteen recurrence waves, then runs gated norm and its A8 epilogue after a local barrier. |

The score candidate retains the 256-leaf ordered dot-product tree. For lane
`l`, the first local expression is
`((p[l]+p[l+128])+(p[l+64]+p[l+192])) + ((p[l+32]+p[l+160])+(p[l+96]+p[l+224]))`;
shuffle distances 16, 8, 4, 2, 1 then reproduce the remaining tree. Missing
features remain positive-zero leaves. Each selected slot is independent.

The tail invokes the incumbent softmax and value bodies with a workgroup
barrier between them. It keeps the BF16-rounded probability values, softmax leaves and
reduction pairs, ascending selected-slot accumulation and BF16/gate epilogue.
It retains the window's A8 code layout and pitch-64 delta layout. The window
candidate requires a complete BF16 tail, at most eight rows, head dimension at
most 256, and no cohort slot. Grouped/fast prefill and int8 windows use their
existing paths. The draft score and tail fall back above head dimension 256.

Cache publication checks FP32 finiteness and BF16 cast overflow before any
state store. Sticky errors suppress publication. It retains the original
index and interleaved K/V row addresses and the same BF16 conversion. There
are no cross-workgroup spin barriers or new retained cache allocations.

The GDN tail candidate admits complete R2/R4/R8 calls with 128-feature heads
and eight columns per recurrence wave. It retains both recurrent storage
layouts. Tiled or scatter recurrence options and split cohort calls keep their
existing kernels. The tile address is explicitly uniform within its wave, so
the compiler can retain scalar placement for the shared row inputs. Each of
the sixteen waves applies the original tile body;
all 512 threads then join the existing 128-leaf norm tree with unused LDS
entries for the surplus lanes. Only the first 128 threads publish the norm and
A8 results. The deferred matrix writes, convolution history and retained rows
keep their original ownership. Resource use and occupancy require native
inspection; a removed launch does not establish a faster kernel.

## Qualification status

Single-user, one-device native identity validation passed on `gfx1151` against
the frozen project baseline `100845d`: a Release build with warnings treated as errors and all
103 planned qualification jobs passed. These comprise seven component jobs,
one warmup, 23 prefix-gate processes and 72 decode processes with 32 generated
tokens each. Code and prose at 1K, 8K and 32K context cover candidate OFF,
each switch individually and the joint candidate. A short prompt additionally
covers candidate OFF and each individual switch. The six contexts also pass joint composition with the
existing dispatch and QSA-pool options, including production min-p 0.7.
Six further decode processes validate the three attention/draft switches
together with `STRIX_QPERSIST_GDN_TAIL=0` at production min-p 0.7 against the
same frozen baseline, with the same full observation scope.

The forced prefix cases cover every rejection position and complete acceptance
in R2/R4/R8, plus owner changes and serial fallback. They compare every physical
logit row, all seven state components including discarded cache tails, and
physical write generations. Baseline/OFF/joint runs cover the six contexts and
the short prompt; an additional pair covers composition on the short prompt.
Decode observations additionally compare the initial full vocabulary row,
initialized draft buffers including discarded tails, and active draft logits.
All compared records match exactly and replay errors are zero. Both recurrent
storage layouts pass the GDN component fixture.

These results qualify the declared identity cases. Throughput and production
enablement require a separate complete-workload timing campaign. Diagnostic
observers must be absent from timing binaries. Keep all switches OFF until
that performance decision; no throughput improvement is asserted here.

The native component target is `strix-hip-qpersist-attention-tests` with
`BUILD_TESTING=ON`. It compares raw scores before softmax, probabilities,
value/gate outputs, dense A8 codes/deltas, full cache buffers, guards and
sticky status. It exercises 112 attention cases (R1/R2/R4/R8, dimensions
128/256, uneven selected counts through 2051, dense and ordinary outputs) and
63 cache cases, including nonfinite inputs, cast overflow and prior errors.
The GDN candidate additionally uses `strix-hip-gdn-deferred-window-tests`,
including pending-prefix commits, both layouts, owner transitions and comparison
against serial outputs, workspace bytes, matrix and convolution history.
These fixtures do not establish model fidelity by themselves.

Admission requires a clean HIP build, those component gates, candidate OFF
and each switch ON against the frozen baseline, every rejection depth in
R2/R4/R8, six complete prompts through 32K, all physical logits and state
components including discarded cache tails, and composition with the existing
production options. Timing follows identity validation in a separate campaign.
