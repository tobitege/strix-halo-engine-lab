# GDN FP16 chunked prefill: `qwen-gdn-fp16-chunk`, revisions 2 and 3

Status: FAST (bit-changing) engine item, default OFF; `--profile fast` applies it as a provisional row. `STRIX_QPF2K_GDN_FP16_CHUNK=1` replaces the conv, the recurrence
and the gated norm of the GDN prefill row route with a chunked route on FP16 WMMA operands with FP32 accumulation.
Unset or `0` keeps the exact production route. The item is ignored under the baseline and for GSQ checkpoints, and
shapes it cannot take stay exact. When it is on, stderr carries one line
`GDN FAST lever qwen-gdn-fp16-chunk rev 2: ON (...)` and the QPF2K witness names the route.

It is the prefill option `gdn_fp16_chunk` (identity tag `gdnchunk16`): `STRIX_PREFILL_EXACT_FP16=1` or
`--prefill-disable gdn_fp16_chunk` turn it off, and the GDN check modes run the exact route.

The mathematics is the public chunked (WY) form of the gated delta rule. The rounding points below define the identity.

Source: `src/hip_gdn_fp16_chunk.inc` (engine hook and norm epilogue in `src/hip_gdn_launcher.hip`).
Test: `tests/hip_gdn_fp16_chunk_tests.hip` (target `strix-hip-gdn-fp16-chunk-tests`).

## Numerical contract (revision 2)

Per value head `h`, key head `h % key_heads`, chunks of 32 rows from the call start, `m` valid rows. Padded rows have
q = k = v = 0, beta = 0 and carry the prefix of row `m - 1`.

- **Conv.** q, k and v are the production conv values: BF16 inputs (history below row 0), `sum = 0; sum += x_j * w_j`
  for taps 0..3, SiLU `sum / (1 + expf(-sum))`, `round_bf16`. q and k are L2-normed with the production tree, and q is
  scaled by `rsqrtf(128)`. `Q16 = fp16(q)`, `K16 = fp16(k)`. v stays exact (BF16).
- **Decay.** `g = gate`, with values below -110 raised to -110 (NaN kept). `G` is the inclusive prefix over the chunk
  (shuffle ladder 1, 2, 4, 8, 16). `e_r = expf(G_r)`, `f_r = expf(G_{m-1} - G_r)`, `eL = e_{m-1}`.
- **Prep.** `KK = W(K16, K16^T)`, `QK = W(Q16, K16^T)`. `W` is a native `v_wmma_f32_16x16x16_f16` chain with K16
  panels in ascending order. `D_rs = expf(G_r - G_s)`.
  - `A_rs = (beta_r * KK_rs) * D_rs` for `s < r`.
  - `P16_rs = fp16(QK_rs * D_rs)` for `s <= r`.
  - `T = (I + A)^-1` by FP32 forward substitution, pivots ascending, `x_i = fmaf(-A_ip, x_p, x_i)`.
  - `T'16_rs = fp16(T_rs * beta_s)`.
- **Scan.** The FP32 state `S[v][k]` is kept as `S^T` accumulator tiles.
  - `X16 = fp16(S)`; `KS = W(K16, X16^T)`; `QS = W(Q16, X16^T)`.
  - `Y16 = fp16(fmaf(-e_r, KS, v))`; `U = W(T'16, Y16)`; `U16 = fp16(U)`; `U2_16 = fp16(U * f_r)`.
  - `S^T <- W(K16^T, U2_16, seed eL * S^T)`.
  - `O = W(P16, U16, seed QS * e_r)`.
- **Output.** O (FP32) goes through the production gated-norm expression: `gdn_window_key_sum` of squares / 128 + eps,
  `rsqrtf`, times the norm weight and `sigmoid(z)`. The result then goes to CORE16, dense A8 or the FP32 core, as on
  the production route.
- **Bits.** Results are deterministic run to run. Splits at multiples of 32 are bitwise invariant.

## Kernels

| Kernel | Grid | Threads | Work |
|---|---|---:|---|
| `prep_kernel` | (key heads, chunks) | 256 | Conv of q, k and the group's v (sliding windows); Q16/K16/v to scratch; KK/QK; prefix; A, P16, (e, f); T'16 per value head |
| `scan_kernel` | value heads | 256 | 8 waves × 16 value columns. `S^T` held in registers. Each chunk is staged once in LDS (Q16, K16, K16^T, T'16, P16, v, e/f) from registers loaded during the previous chunk. z is loaded during the chunk. O is staged in the dead Q/K buffer for the norm epilogue. 41.7 KB LDS, ≤ 240 VGPR (6 waves/SIMD: all 48 heads resident) |
| `conv_history_kernel` | conv channel blocks | 128 | New BF16 conv history, after both readers |

**Persistent scratch** grows per (device, stream) and is never shrunk. It holds Q16/K16 and v rows, T'16/P16 and (e, f).

## Test

`strix-hip-gdn-fp16-chunk-tests` exits with 77 when no GPU is present.

**Checks against an FP64 token recurrence** (production conv in FP32, FP64 L2, recurrence and gated norm), for raw
core, normed output and final state:
- relL2 <= 3e-3 and max error <= 2e-2 relative to max(1, |ref|);
- exact conv history;
- untouched inputs.

**Cases:**
- lengths 1-1000 and every tail class;
- resets and tiny decays;
- both state layouts and several head groupings;
- bitwise run-to-run equality and aligned split invariance.

**`--time`** reports the median prep, scan + norm, raw scan and total at T = 8192 with 48/16 heads.

## Revision 3 draft

`STRIX_QPF2K_GDN_FP16_CHUNK=3` selects **qwen-gdn-fp16-chunk rev 3**. It is default OFF and remains a FAST draft:
compiler, native operator, fidelity and complete-workload gates are pending. `=1` retains revision 2; unset/`=0`
retains the exact route. The existing master/individual OFF, baseline, GSQ, shape and check-mode guards apply.
Restart after changing the switch. The stderr identity and recurrence witness name revision 3; arithmetic identity
uses `gdnchunk16r3` instead of `gdnchunk16`, including the compiled/state/cache hash. The same GDN option owns its
unchanged persistent allocation; the extra mask bit is a revision discriminator, not a separately enabled option.

Only preparation changes. `prep_kernel<B16,ConvRows,FactorDecay>` uses 16-row convolution segments instead of 8,
reusing weights and the three-row halo for longer. Every row retains its four-tap expression, BF16 boundaries,
L2 tree and Q16/K16/v16 values. It still fuses all q/k/v convolution with WY preparation. Gate projections and
transforms remain on their existing route; they are not recomputed in prep. Scan geometry stays C32 with eight
waves, FP32 state/accumulators and the same epilogue. No larger chunk or compensated scan is introduced.

For finite prefix endpoints and final prefix in `[-60,0]`, off-diagonal decay becomes
`D_rs = expf(G_r) * expf(-G_s)`; the diagonal stays exactly 1. Other intervals retain `expf(G_r-G_s)`.
Two row factors replace a triangular matrix of exponentials; factors and products remain normal and finite in
this bounded range. This is algebraically the same decay, with different FP32 rounding and subsequent FP16 T/P
boundaries. If each exponential has relative error at most E, the factor product's relative error against the
real exponential of the stored prefixes is at most `(1+E)^2*(1+2^-24)-1`; native E must be established on the
qualification toolchain. This local bound does not prove the recurrence/output screen or model fidelity.

Conv history remains exact. The fixed operation order and unchanged C32 call-relative boundaries preserve
run-to-run determinism and aligned-split invariance within this revision. Arbitrary partial-chunk splits still
change the numerical partition. Neither revision is bit-identical to the production token recurrence.

The existing fixture additionally checks both layouts, all small-tail classes, states scaled by `2^-12` and
`2^12`, near-unit decay, reset/tiny-decay boundaries and splits at 32/64/96/128. Preparation checks cover both
input representations, head groups 1..4, lengths 1/15/16/17/31/32/33/65, exact Q/K/v and decay metadata, and exact
T/P restoration for fallback chunks. It prints pooled and worst-per-head state errors and a final maxima line;
the existing relL2 `3e-3` and normalized-max `2e-2` bands are unchanged. No rev-3 measured figures are available.

On a qualified HIP node, build the existing target and run separate processes (startup-cached selector):

```sh
cmake --build build --target strix-hip-gdn-fp16-chunk-tests strix-qwen-startup-policy-tests -j8
./build/strix-qwen-startup-policy-tests
for revision in 0 1 3; do
  STRIX_QPF2K_GDN_FP16_CHUNK=$revision ./build/strix-hip-gdn-fp16-chunk-tests || exit $?
done
STRIX_QPF2K_GDN_FP16_CHUNK=1 ./build/strix-hip-gdn-fp16-chunk-tests --time
STRIX_QPF2K_GDN_FP16_CHUNK=3 ./build/strix-hip-gdn-fp16-chunk-tests --time
```

The operator fixture calls the chunk route directly, so `=0` exercises its retained rev-2 implementation; engine
OFF restoration requires complete output/state comparison on the production entry point. Inspect prep VGPR/LDS
and spill counts before timing: longer conv segments may reduce parallelism, and extra row-factor LDS/branches
may outweigh reduced SFU work. There is no speed claim until native interleaved complete-workload timing passes.
