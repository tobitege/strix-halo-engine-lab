// qpf2k/attn: attend over an FP16 shadow of the BF16 KV (bench candidate F). Included after hip_qsa_fast_prefill.inc.
//
// attend_body's loaders convert every K and V word of every chunk of every tile from BF16 to FP16 (bf16x2_to_f16x2:
// two shifts / masks and one v_cvt_pkrtz per word) and zero the slots past the tile's visible prefix (one v_cndmask per
// word). Neighbouring tiles select mostly the same blocks, so each KV word is converted ~u x (rows / TR) times. Here
// kv_to_f16_kernel converts the layer's KV once per call with the same bf16x2_to_f16x2 (identical FP16 bits), and the
// loaders copy FP16 words: no conversion, and no zeroing when every slot of the chunk is visible (wave-uniform test;
// the zeroing path is attend_body's). The compute waves are attend_body's statement for statement, so every WMMA
// operand, the softmax and the epilogue are the same: outputs bitwise equal. BF16 KV only, one slice.
namespace qsa_fast {
// out[i] = bf16x2_to_f16x2(in[i]) for i < words (the KV rows [K heads | V heads] of tokens [first, first + count)).
__global__ __launch_bounds__(256) void kv_to_f16_kernel(const unsigned* in, unsigned* out, std::uint64_t first_word, std::uint64_t words) {
    const std::uint64_t i0 = (static_cast<std::uint64_t>(blockIdx.x) * 256U + threadIdx.x) * 4U;
    if (i0 + 4U <= words) {
        const auto x = load16(in + first_word + i0);
        *reinterpret_cast<u32x4*>(out + first_word + i0) = u32x4{bf16x2_to_f16x2(x.x), bf16x2_to_f16x2(x.y), bf16x2_to_f16x2(x.z), bf16x2_to_f16x2(x.w)};
    } else {
        for (std::uint64_t i = i0; i < words; ++i) out[first_word + i] = bf16x2_to_f16x2(in[first_word + i]);
    }
}

template <unsigned Rows, unsigned TR, unsigned G, unsigned D>
__device__ __forceinline__ void attend_shadow_body(const Views<Rows>& v, const std::uint16_t* kv, std::uint32_t kv_heads,
    std::uint32_t heads, std::uint32_t union_at, unsigned tile, unsigned yb) {
    constexpr unsigned HPW = 16U / TR, HG = G * HPW, Loaders = 128U, Compute = 2U * G;
    constexpr unsigned KT = kChunk * 32U / Loaders;
    __shared__ __align__(16) _Float16 k_lds[2][kChunk][kPitch];
    __shared__ __align__(16) _Float16 v_lds[2][kDim][kChunk];
    __shared__ __align__(16) float xch[G][2][32][8];
    __shared__ uint2 meta_lds[2][4];
    const unsigned t = threadIdx.x, lane = t % 32U, w = t / 32U, col = lane % 16U, half = lane / 16U;
    const bool loader = w >= Compute;
    const unsigned r0 = tile * TR;
    const unsigned per_kv = heads / kv_heads, groups = per_kv / HG;
    const unsigned kvh = yb / groups, hg = yb % groups;
    unsigned tile_visible = 0U;
    for (unsigned i = 0; i < TR; ++i) tile_visible = max(tile_visible, v.visible[r0 + i]);
    const auto* header = reinterpret_cast<const std::uint32_t*>(v.scratch[r0] + union_at);
    const auto* pairs = reinterpret_cast<const uint2*>(header + 4U + TR);
    const unsigned chunks = header[0] / 4U;
    const unsigned c0 = 0U, c1 = chunks;

    if (loader) {
        struct R { u32x4 k[KT]; u32x4 v[4]; unsigned valid; };
        const auto meta_at = [&](unsigned c) { return lane < 4U ? pairs[4U * c + lane] : make_uint2(0U, 0U); };
        const auto token_of = [](uint2 m, unsigned slot) {
            return static_cast<unsigned>(__shfl(static_cast<int>(m.x), static_cast<int>(slot / 4U), 32)) * 4U + slot % 4U;
        };
        const auto row_elements = static_cast<std::uint64_t>(kv_heads) * kDim * 2U;
        const auto* k_base = kv + static_cast<std::uint64_t>(kvh) * kDim;
        const auto* v_base = kv + static_cast<std::uint64_t>(kv_heads + kvh) * kDim;
        const unsigned lt = t - Compute * 32U;
        constexpr unsigned all_valid = (1U << (KT + 4U)) - 1U;
        const auto fetch = [&](uint2 m, R& g) {
            g.valid = 0U;
#pragma unroll
            for (unsigned i = 0; i < KT; ++i) {
                const unsigned task = lt + i * Loaders;
                const unsigned token = token_of(m, task / 32U);
                g.valid |= (token < tile_visible ? 1U : 0U) << i;
                const unsigned at = token < tile_visible ? token : tile_visible - 1U;
                g.k[i] = load16(k_base + at * row_elements + 8U * (task % 32U));
            }
#pragma unroll
            for (unsigned h = 0; h < 4U; ++h) {
                const unsigned token = token_of(m, 4U * (lt % 4U) + h);
                g.valid |= (token < tile_visible ? 1U : 0U) << (KT + h);
                const unsigned at = token < tile_visible ? token : tile_visible - 1U;
                g.v[h] = load16(v_base + at * row_elements + 8U * (lt / 4U));
            }
        };
        const auto stage_with = [&]<bool Mask>(const R& g, uint2 m, unsigned buf) {
            const auto keep = [&](u32x4 x, unsigned b) {
                if constexpr (Mask) return ((g.valid >> b) & 1U) ? x : u32x4{0U, 0U, 0U, 0U};
                else return x;
            };
#pragma unroll
            for (unsigned i = 0; i < KT; ++i) {
                const unsigned task = lt + i * Loaders;
                const u32x4 x = keep(g.k[i], i);
                *reinterpret_cast<uint4*>(&k_lds[buf][task / 32U][8U * (task % 32U)]) = uint4{x.x, x.y, x.z, x.w};
            }
            unsigned f[4][4];
#pragma unroll
            for (unsigned h = 0; h < 4U; ++h) {
                const u32x4 x = keep(g.v[h], KT + h);
                f[h][0] = x.x; f[h][1] = x.y; f[h][2] = x.z; f[h][3] = x.w;
            }
            const unsigned feature = 8U * (lt / 4U), slot = 4U * (lt % 4U);
#pragma unroll
            for (unsigned k = 0; k < 4U; ++k) {
                *reinterpret_cast<uint2*>(&v_lds[buf][v_row(feature + 2U * k)][slot]) =
                    make_uint2((f[0][k] & 0xffffU) | (f[1][k] << 16U), (f[2][k] & 0xffffU) | (f[3][k] << 16U));
                *reinterpret_cast<uint2*>(&v_lds[buf][v_row(feature + 2U * k + 1U)][slot]) =
                    make_uint2((f[0][k] >> 16U) | (f[1][k] & 0xffff0000U), (f[2][k] >> 16U) | (f[3][k] & 0xffff0000U));
            }
            if (lt < 4U) meta_lds[buf][lt] = m;
        };
        const auto stage = [&](const R& g, uint2 m, unsigned buf) {
            // wave-uniform: zeroing only when some slot of this wave's tasks is past the visible prefix
            if (__builtin_amdgcn_ballot_w32(g.valid != all_valid) == 0U) stage_with.template operator()<false>(g, m, buf);
            else stage_with.template operator()<true>(g, m, buf);
        };
        if (c0 < c1) {
            const auto chunk = [&](unsigned c) { return c < c1 ? c : c1 - 1U; };
            R r[D];
            uint2 held[D], raw[D];
            {
                R first;
                const uint2 m = meta_at(c0);
                fetch(m, first);
                stage(first, m, 0U);
            }
#pragma unroll
            for (unsigned d = 0; d < D; ++d) { held[d] = meta_at(chunk(c0 + 1U + d)); fetch(held[d], r[d]); }
#pragma unroll
            for (unsigned d = 0; d < D; ++d) raw[d] = meta_at(chunk(c0 + 1U + D + d));
            lds_barrier();
            for (unsigned c = c0; c < c1; c += D) {
#pragma unroll
                for (unsigned k = 0; k < D; ++k) {
                    const unsigned step = c + k;
                    if (step >= c1) break;
                    stage(r[k], held[k], (step - c0 + 1U) & 1U);
                    held[k] = raw[k];
                    fetch(held[k], r[k]);
                    raw[k] = meta_at(chunk(step + 1U + 2U * D));
                    lds_barrier();
                    lds_barrier();
                }
            }
        }
        return;
    }

    const unsigned grp = w / 2U, fh = w % 2U;
    const unsigned row = col % TR, head = kvh * per_kv + hg * HG + grp * HPW + col / TR, r = r0 + row;
    float score_scale;
    f16x16 qf[8];
    {
        const float* q = v.q_gate[r] + static_cast<std::uint64_t>(head) * 2U * kDim;
        float peak = 0.0F;
        for (unsigned i = half * (kDim / 2U); i < (half + 1U) * (kDim / 2U); i += 4U) {
            const float4 x = *reinterpret_cast<const float4*>(q + i);
            peak = fmaxf(peak, fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w))));
        }
        peak = fmaxf(peak, swap16(peak));
        const int e = q_exponent(peak);
        const float scale = ldexpf(1.0F, e);
        score_scale = ldexpf(1.0F, -e) / sqrtf(static_cast<float>(kDim)) * 1.4426950408889634F;
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k)
#pragma unroll
            for (unsigned i = 0; i < 16U; i += 4U) {
                const float4 x = *reinterpret_cast<const float4*>(q + fh * (kDim / 2U) + 16U * k + i);
                qf[k][i] = static_cast<_Float16>(x.x * scale); qf[k][i + 1U] = static_cast<_Float16>(x.y * scale);
                qf[k][i + 2U] = static_cast<_Float16>(x.z * scale); qf[k][i + 3U] = static_cast<_Float16>(x.w * scale);
            }
    }
    const unsigned visible = v.visible[r], bit = 1U << row, weights = header[4U + row];
    f32x8 acc[8];
#pragma unroll
    for (unsigned n = 0; n < 8U; ++n) acc[n] = f32x8{};
    float m_run = -INFINITY, l_run = 0.0F;
    if (c0 < c1) lds_barrier();
    for (unsigned c = c0; c < c1; ++c) {
        const unsigned buf = (c - c0) & 1U;
        uint2 meta[4];
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) meta[i] = meta_lds[buf][i];
        strix::attn_core::selected_flash_step(qf, acc, k_lds[buf], v_lds[buf], xch[grp], col, half, fh, lane,
            m_run, l_run, SelectedFlashPolicy<false>{meta, bit, visible, weights, half, score_scale, nullptr});
    }
    const float* gate_in = v.q_gate[r] + static_cast<std::uint64_t>(head) * 2U * kDim + kDim;
    auto out = v.image(r) + static_cast<std::uint64_t>(head) * kDim;
#pragma unroll
    for (unsigned n = 0; n < 8U; ++n)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const unsigned d = fh * (kDim / 2U) + 16U * n + 2U * e + half;
            const float attn = round_bf16(l_run > 0.0F ? acc[n][e] / l_run : 0.0F);
            const float gate = round_bf16(1.0F / (1.0F + expf(-gate_in[d])));
            out[d] = round_bf16(attn * gate);
        }
}
// One launch over every group of a call (as attend_multi_kernel): blockIdx.x = group * (Rows / TR) + tile. kv is the
// FP16 shadow (kv_to_f16_kernel of the BF16 KV, same layout).
template <unsigned Rows, unsigned TR, unsigned G, unsigned D = 3U>
__global__ __launch_bounds__((2U * G + 4U) * 32U) void attend_shadow_multi_kernel(const Views<Rows>* groups, const std::uint16_t* kv,
    std::uint32_t kv_heads, std::uint32_t heads, std::uint32_t union_at) {
    attend_shadow_body<Rows, TR, G, D>(groups[blockIdx.x / (Rows / TR)], kv, kv_heads, heads, union_at, blockIdx.x % (Rows / TR), blockIdx.y);
}
// Occupancy probe: the same body with a 3-workgroups-per-WGP register budget (bench only).
template <unsigned Rows, unsigned TR, unsigned G, unsigned D = 2U>
__global__ __attribute__((amdgpu_flat_work_group_size(1, (2U * G + 4U) * 32U), amdgpu_waves_per_eu(8, 8)))
void attend_shadow_occ_multi_kernel(const Views<Rows>* groups, const std::uint16_t* kv, std::uint32_t kv_heads, std::uint32_t heads,
    std::uint32_t union_at) {
    attend_shadow_body<Rows, TR, G, D>(groups[blockIdx.x / (Rows / TR)], kv, kv_heads, heads, union_at, blockIdx.x % (Rows / TR), blockIdx.y);
}
// Candidate I (scheduling only, same body): one-dimensional grid ordered group-major with the (kv head, head group)
// index between group and tile, x = (group * ys + y) * (Rows / TR) + tile, so the workgroups resident at a time
// cover as few groups (row windows, whose unions overlap) as possible: the KV blocks they gather stay in the
// last-level cache. Launched with optional dynamic LDS to cap workgroups per WGP (fewer groups in flight).
template <unsigned Rows, unsigned TR, unsigned G, unsigned D = 3U>
__global__ __launch_bounds__((2U * G + 4U) * 32U) void attend_shadow_order_kernel(const Views<Rows>* groups, const std::uint16_t* kv,
    std::uint32_t kv_heads, std::uint32_t heads, std::uint32_t union_at, std::uint32_t ys) {
    constexpr unsigned tiles = Rows / TR;
    const unsigned bx = blockIdx.x, tile = bx % tiles, y = (bx / tiles) % ys, g = bx / (tiles * ys);
    attend_shadow_body<Rows, TR, G, D>(groups[g], kv, kv_heads, heads, union_at, tile, y);
}
} // namespace qsa_fast
