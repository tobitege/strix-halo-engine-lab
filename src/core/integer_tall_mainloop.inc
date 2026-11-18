// Shared tall IU8 execution, recovered from the project-owned dense kernel.
// Adapter supplies tensor addresses and exact half conversion; Plan supplies
// tile order, raw-buffer load level and the number of K parts per launch.
using Plan = typename Operand::Plan;
using Ops = typename Operand::Ops;
using v4i = typename Operand::Int4;
using v8i = typename Operand::Int8;
constexpr unsigned RT = MT, OT = NT, Lean = Plan::lean, Parts = Plan::parts;
constexpr bool ColFirst = Plan::column_first;
constexpr unsigned tall_waves = 8;
const auto& v0 = ad.tensor();
    __shared__ float part[tall_waves][RT * OT][8][32];
    __shared__ float4 dsh[RT][32][2][2];
    [[maybe_unused]] float first[RT * OT];
#pragma unroll 1
    for (unsigned kp = 0; kp < Parts; ++kp) {
    auto v = v0;
    if constexpr (Parts > 1) {
        v.k_offset = v0.k_offset + kp * v0.width;
        v.codes = v0.codes + kp * v0.width;
        v.deltas = v0.deltas + kp * v0.width / 32U;
        if (kp) __syncthreads();  // the first part's reads of part are done
    }
    const unsigned w = threadIdx.x / 32U, l = threadIdx.x % 32U, h = l >> 4U, lo = l & 15U;
    const unsigned r0 = (ColFirst ? blockIdx.y : blockIdx.x) * 16U * RT, o0 = (ColFirst ? blockIdx.x : blockIdx.y) * 16U * OT;
    const unsigned groups = v.width / 32U, row_bytes = (v.row_width ? v.row_width : v.width) / 32U * 34U;
    const signed char* arow[RT];
    bool live[RT];
#pragma unroll
    for (unsigned rt = 0; rt < RT; ++rt) {
        const unsigned row = r0 + rt * 16U + lo;
        live[rt] = row < v.rows;
        arow[rt] = reinterpret_cast<const signed char*>(v.codes) + static_cast<std::size_t>(live[rt] ? row : 0U) * v.x_stride;
    }
    const std::uint8_t* wrow[OT];
#pragma unroll
    for (unsigned t = 0; t < OT; ++t) {
        const unsigned o = o0 + t * 16U + lo;
        wrow[t] = v.weight + v.k_offset / 32U * 34U + static_cast<std::size_t>(o < v.count ? o : v.count - 1U) * row_bytes;
    }
    float acc[4][RT][OT][8];
#pragma unroll
    for (unsigned j = 0; j < 4; ++j)
#pragma unroll
        for (unsigned rt = 0; rt < RT; ++rt)
#pragma unroll
            for (unsigned t = 0; t < OT; ++t)
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) acc[j][rt][t][e] = 0.0F;
    v8i magic{Ops::dot_magic, Ops::dot_magic, Ops::dot_magic, Ops::dot_magic, Ops::dot_magic, Ops::dot_magic, Ops::dot_magic, Ops::dot_magic};
    if constexpr (Lean >= 1) asm volatile("" : "+v"(magic));
    // Lean >= 2: the activation codes as a buffer (raw, stride 0, offsets in bytes) and each row's byte offset
    const std::size_t abytes = static_cast<std::size_t>(v.rows) * v.x_stride;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"  // num_records is int in some ROCm builtins, unsigned in others (clamped value)
    [[maybe_unused]] const __amdgpu_buffer_rsrc_t abuf = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<std::int8_t*>(v.codes), static_cast<short>(0), static_cast<int>(abytes < 0x7fffffffU ? abytes : std::size_t{0x7fffffffU}), 0x31004000);
#pragma clang diagnostic pop
    [[maybe_unused]] unsigned aoff[RT];
#pragma unroll
    for (unsigned rt = 0; rt < RT; ++rt) {
        const unsigned row = r0 + rt * 16U + lo;
        aoff[rt] = (row < v.rows ? row : 0U) * v.x_stride;
    }
    // Lean >= 3: the weight tensor as a buffer and each output row's byte offset (K-offset included)
    const std::size_t wbytes = static_cast<std::size_t>(v.count) * row_bytes;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"  // num_records is int in some ROCm builtins, unsigned in others (clamped value)
    [[maybe_unused]] const __amdgpu_buffer_rsrc_t wbuf = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<std::uint8_t*>(v.weight), static_cast<short>(0), static_cast<int>(wbytes < 0x7fffffffU ? wbytes : std::size_t{0x7fffffffU}), 0x31004000);
#pragma clang diagnostic pop
    [[maybe_unused]] unsigned woff[OT];
#pragma unroll
    for (unsigned t = 0; t < OT; ++t) {
        const unsigned o = o0 + t * 16U + lo;
        woff[t] = (o < v.count ? o : v.count - 1U) * row_bytes + v.k_offset / 32U * 34U;
    }
    // Lean >= 4: one group's weight records (OT tiles) as WMMA operands and f32 scales; Lean >= 5 keeps each scale an
    // opaque f32 VGPR, so the scale products are plain v_mul_f32 (dual-issuable) instead of v_fma_mix with the f16 scale.
    [[maybe_unused]] const auto load_w = [&](unsigned gg, v4i (&b0)[OT], v4i (&b1)[OT], float (&sc)[OT]) __attribute__((always_inline)) {
#pragma unroll
        for (unsigned t = 0; t < OT; ++t) {
            const int off = static_cast<int>(woff[t] + 34U * gg);
            const auto dw16 = __builtin_amdgcn_raw_buffer_load_b16(wbuf, off, 0, 0);
            const auto q0 = __builtin_amdgcn_raw_buffer_load_b128(wbuf, off + 2, 0, 0);
            const auto q1 = __builtin_amdgcn_raw_buffer_load_b128(wbuf, off + 18, 0, 0);
            b0[t] = v4i{static_cast<int>(q0[0]), static_cast<int>(q0[1]), static_cast<int>(q0[2]), static_cast<int>(q0[3])};
            b1[t] = v4i{static_cast<int>(q1[0]), static_cast<int>(q1[1]), static_cast<int>(q1[2]), static_cast<int>(q1[3])};
            sc[t] = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<unsigned short>(dw16)));
            if constexpr (Lean == 5 || Lean == 6) asm volatile("" : "+v"(sc[t]));
        }
    };
    [[maybe_unused]] v4i pb0[OT], pb1[OT];
    [[maybe_unused]] float pscale[OT];
    if constexpr (Lean == 6 || Lean == 8)
        if (w < groups) load_w(w, pb0, pb1, pscale);
    for (unsigned s0 = 0; s0 < groups; s0 += 32U) {
        __syncthreads();  // the previous chunk's deltas are read
        auto* d = reinterpret_cast<float*>(dsh);
        for (unsigned i = threadIdx.x; i < RT * 32U * 16U; i += 32U * tall_waves) {
            const unsigned rt = i / 512U, gl = (i / 16U) % 32U, rr = i % 16U, r = r0 + rt * 16U + rr, g = s0 + gl;
            if constexpr (Lean >= 2)
                d[((rt * 32U + gl) * 2U + (rr & 1U)) * 8U + (rr >> 1U)] =
                    g < groups ? v.deltas[static_cast<std::size_t>(r < v.rows ? r : 0U) * (v.x_stride / 32U) + g] : 0.0F;
            else
                d[((rt * 32U + gl) * 2U + (rr & 1U)) * 8U + (rr >> 1U)] =
                    r < v.rows && g < groups ? v.deltas[static_cast<std::size_t>(r) * (v.x_stride / 32U) + g] : 0.0F;
        }
        __syncthreads();
        // Plain unrolled loop with continue, as the split-v2 tall kernel: the lambda form keeps different live ranges.
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const unsigned g = s0 + w + 8U * j;
            if (g >= groups) continue;
            v4i a0[RT], a1[RT];
#pragma unroll
            for (unsigned rt = 0; rt < RT; ++rt) {
                if constexpr (Lean >= 2) {
                    const auto x0 = __builtin_amdgcn_raw_buffer_load_b128(abuf, static_cast<int>(aoff[rt] + 32U * g), 0, 0);
                    const auto x1 = __builtin_amdgcn_raw_buffer_load_b128(abuf, static_cast<int>(aoff[rt] + 32U * g + 16U), 0, 0);
                    a0[rt] = v4i{static_cast<int>(x0[0]), static_cast<int>(x0[1]), static_cast<int>(x0[2]), static_cast<int>(x0[3])};
                    a1[rt] = v4i{static_cast<int>(x1[0]), static_cast<int>(x1[1]), static_cast<int>(x1[2]), static_cast<int>(x1[3])};
                    continue;
                }
                a0[rt] = v4i{0, 0, 0, 0};
                a1[rt] = v4i{0, 0, 0, 0};
                if (live[rt]) {
                    const auto* x = reinterpret_cast<const int4*>(arow[rt] + 32U * g);
                    const int4 x0 = x[0], x1 = x[1];
                    a0[rt] = v4i{x0.x, x0.y, x0.z, x0.w};
                    a1[rt] = v4i{x1.x, x1.y, x1.z, x1.w};
                }
            }
            // Lean >= 4: every weight record of the chunk issued before the first WMMA (Lean 3 left each tile's buffer loads
            // next to its WMMA pair, waiting vmcnt(0) per tile).
            [[maybe_unused]] v4i wb0[OT], wb1[OT];
            [[maybe_unused]] float wscale[OT];
            if constexpr (Lean == 6 || Lean == 8) {
                // Lean 6 / 8: this group's records were loaded one group ahead; issue the next group's now
#pragma unroll
                for (unsigned t = 0; t < OT; ++t) {
                    wb0[t] = pb0[t];
                    wb1[t] = pb1[t];
                    wscale[t] = pscale[t];
                }
                const unsigned gn = j < 3U ? g + 8U : s0 + 32U + w;
                if (gn < groups) load_w(gn, pb0, pb1, pscale);
            } else if constexpr (Lean >= 4) {
                load_w(g, wb0, wb1, wscale);
            }
#pragma unroll
            for (unsigned t = 0; t < OT; ++t) {
                v4i b0, b1;
                float scale;
                if constexpr (Lean >= 4) {
                    b0 = wb0[t];
                    b1 = wb1[t];
                    scale = wscale[t];
                } else if constexpr (Lean >= 3) {
                    const int off = static_cast<int>(woff[t] + 34U * g);
                    const auto dw16 = __builtin_amdgcn_raw_buffer_load_b16(wbuf, off, 0, 0);
                    const auto q0 = __builtin_amdgcn_raw_buffer_load_b128(wbuf, off + 2, 0, 0);
                    const auto q1 = __builtin_amdgcn_raw_buffer_load_b128(wbuf, off + 18, 0, 0);
                    b0 = v4i{static_cast<int>(q0[0]), static_cast<int>(q0[1]), static_cast<int>(q0[2]), static_cast<int>(q0[3])};
                    b1 = v4i{static_cast<int>(q1[0]), static_cast<int>(q1[1]), static_cast<int>(q1[2]), static_cast<int>(q1[3])};
                    scale = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<unsigned short>(dw16)));
                } else {
                    const auto raw = Operand::load_raw(wrow[t], g);
                    b0 = v4i{static_cast<int>(raw.q[0]), static_cast<int>(raw.q[1]), static_cast<int>(raw.q[2]), static_cast<int>(raw.q[3])};
                    b1 = v4i{static_cast<int>(raw.q[4]), static_cast<int>(raw.q[5]), static_cast<int>(raw.q[6]), static_cast<int>(raw.q[7])};
                    scale = Lean >= 1 ? ad.half_select(raw.d) : ad.half_exact(raw.d);
                }
#pragma unroll
                for (unsigned rt = 0; rt < RT; ++rt) {
                    const float4 d0 = dsh[rt][g - s0][h][0], d1 = dsh[rt][g - s0][h][1];
                    const float rd[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
                    v8i c = integer_wmma<8>(a0[rt], b0, magic);
                    c = integer_wmma<8>(a1[rt], b1, c);
                    float sc = scale;
                    if constexpr (Lean >= 7) asm("" : "+v"(sc) : "v"(c[0]));  // not volatile: no barrier for the LDS reads
#pragma unroll
                    for (unsigned e = 0; e < 8; ++e) {
                        const float dot = __fsub_rn(__int_as_float(c[e]), 0x1.8p23F);
                        acc[j][rt][t][e] = __fadd_rn(acc[j][rt][t][e], __fmul_rn(dot, __fmul_rn(sc, rd[e])));
                    }
                }
            }
        }
    }
#pragma unroll
    for (unsigned rt = 0; rt < RT; ++rt)
#pragma unroll
        for (unsigned t = 0; t < OT; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e)
                part[w][rt * OT + t][e][l] = (acc[0][rt][t][e] + acc[2][rt][t][e]) + (acc[1][rt][t][e] + acc[3][rt][t][e]);
    __syncthreads();
    constexpr unsigned per_thread = RT * OT * 8U * 32U / (32U * tall_waves);
    float ys[per_thread];
#pragma unroll
    for (unsigned q = 0; q < per_thread; ++q) {
        const unsigned i = threadIdx.x + q * 32U * tall_waves;
        const unsigned ln = i % 32U, e = (i / 32U) % 8U, tile = i / 256U, rt = tile / OT, t = tile % OT;
        const unsigned o = o0 + t * 16U + (ln & 15U), r = r0 + rt * 16U + 2U * e + (ln >> 4U);
        const float s0 = part[0][tile][e][ln] + part[4][tile][e][ln], s1 = part[1][tile][e][ln] + part[5][tile][e][ln];
        const float s2 = part[2][tile][e][ln] + part[6][tile][e][ln], s3 = part[3][tile][e][ln] + part[7][tile][e][ln];
        float y = (s0 + s2) + (s1 + s3);
        if constexpr (Parts > 1) {
            if (kp == 0U) {
                first[q] = y;
                continue;
            }
            y = first[q] + y;
        }
        ys[q] = y;
        if (o < v.count && r < v.rows) {
            float* dst = v.out + static_cast<std::size_t>(r) * v.out_stride + o;
            *dst = v.accumulate ? *dst + y : y;
        }
    }
    // Optional A8 of the outputs (OT even: the block owns whole 32-output groups). ys[q] is tile q (rt = q / OT, t = q % OT)
    // at row 2w + l / 16 and column l % 16 of the tile, so tiles t = 2g and 2g + 1 hold columns lo and lo + 16 of the same
    // row and group in one lane: each half wave quantizes its row's group with quantize_a8_kernel's operations, its first
    // maximum step (lanes L and L ^ 16) being the lane's own fmaxf(|x0|, |x1|), then the 8 / 4 / 2 / 1 steps in the half.
    if constexpr (OT % 2U == 0U) {
        if (v.a8_codes && kp + 1U == Parts) {
#pragma unroll
            for (unsigned rt = 0; rt < RT; ++rt)
#pragma unroll
                for (unsigned g = 0; g < OT / 2U; ++g) {
                    const float x0 = ys[rt * OT + 2U * g], x1 = ys[rt * OT + 2U * g + 1U];
                    float amax = fmaxf(fabsf(x0), fabsf(x1));
                    for (int stride = 8; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 16));
                    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
                    const float dl = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
                    const float rounded0 = tiny ? 0.0F : rintf(x0 / dl), rounded1 = tiny ? 0.0F : rintf(x1 / dl);
                    const int c0 = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded0)));
                    const int c1 = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded1)));
                    int sum = c0 + c1;
                    for (int stride = 8; stride; stride /= 2) sum += __shfl_xor(sum, stride, 16);
                    const unsigned r = r0 + rt * 16U + 2U * w + (l >> 4U), o = o0 + g * 32U;
                    if (r < v.rows && o < v.count) {
                        std::int8_t* code = v.a8_codes + static_cast<std::size_t>(r) * v.a8_stride + o + lo;
                        code[0] = static_cast<std::int8_t>(c0);
                        code[16] = static_cast<std::int8_t>(c1);
                        if (lo == 0U) {
                            const std::size_t gi = static_cast<std::size_t>(r) * (v.a8_stride / 32U) + o / 32U;
                            v.a8_deltas[gi] = dl;
                            v.a8_sums[gi] = sum;
                        }
                    }
                }
        }
    }
    }
