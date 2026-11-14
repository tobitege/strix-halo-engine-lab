// Adjacent TR=4 tiles, the original twelve-head group per tile. Included after hip_qsa_fast_prefill.inc.
// Twelve compute + four loader waves (512 threads); one image per tile, 45 KiB LDS. Each compute wave
// owns only one query/feature half: no doubled per-thread Q/acc. One image per tile and a one-step
// register queue bound the footprint; the launcher rejects residency loss or spills. No global scratch is added.
// Only coincident chunks share loads: union order/chunk boundaries never change to chase reuse.
namespace qsa_fast {
template <unsigned Rows, bool Capture = false>
__device__ __forceinline__ void attend_kv_pair_body(const Views<Rows>& v, const std::uint16_t* kv,
    std::uint32_t kv_heads, std::uint32_t heads, std::uint32_t union_at, unsigned tile, unsigned yb) {
    constexpr unsigned TR = 4U, HPW = 4U, G = 3U, HG = G * HPW, Compute = 4U * G, Loaders = 128U, Tasks = 2U * kChunk * 32U / Loaders;
    static_assert(Rows % (2U * TR) == 0U);
    static_assert(2U * kChunk * kPitch * sizeof(_Float16) + 2U * kDim * kChunk * sizeof(_Float16) +
        2U * G * 2U * 32U * 8U * sizeof(float) + 8U * sizeof(uint2) == 45632U);
    __shared__ __align__(16) _Float16 k_lds[2][kChunk][kPitch];
    __shared__ __align__(16) _Float16 v_lds[2][kDim][kChunk];
    __shared__ __align__(16) float xch[2][G][2][32][8];
    __shared__ uint2 meta_lds[2][4];
    const unsigned t = threadIdx.x, lane = t % 32U, w = t / 32U, col = lane % 16U, half = lane / 16U;
    const unsigned per_kv = heads / kv_heads, groups = per_kv / HG;
    const unsigned kvh = yb / groups, hg = yb % groups;
    const std::uint32_t* headers[2];
    const uint2* pairs[2];
    unsigned chunks[2], tile_visible[2] = {};
#pragma unroll
    for (unsigned side = 0U; side < 2U; ++side) {
        const unsigned first = (tile + side) * TR;
        headers[side] = reinterpret_cast<const std::uint32_t*>(v.scratch[first] + union_at);
        pairs[side] = reinterpret_cast<const uint2*>(headers[side] + 4U + TR);
        chunks[side] = headers[side][0] / 4U;
#pragma unroll
        for (unsigned r = 0U; r < TR; ++r) tile_visible[side] = max(tile_visible[side], v.visible[first + r]);
    }
    const unsigned steps = max(chunks[0], chunks[1]);
    if (w >= Compute) {
        const unsigned lt = t - Compute * 32U;
        const auto row_elements = static_cast<std::uint64_t>(kv_heads) * kDim * 2U;
        const auto* k_base = kv + static_cast<std::uint64_t>(kvh) * kDim;
        const auto* v_base = kv + static_cast<std::uint64_t>(kv_heads + kvh) * kDim;
        struct Regs { u32x4 k[Tasks], value[Tasks]; unsigned dest[Tasks]; };
        // An owner is the first occurrence among BOTH chunks, including repeated padding block zero.
        // A mask-zero block is still fetched: NaN * zero in value WMMA is observable.
        const auto meta_at = [&](unsigned c, unsigned b) {
            return c < chunks[b / 4U] ? pairs[b / 4U][4U * c + b % 4U] : make_uint2(0U, 0U);
        };
        const auto fetch = [&](unsigned c, Regs& reg) {
#pragma unroll
            for (unsigned i = 0U; i < Tasks; ++i) {
                const uint2 m = meta_at(c, i);
                const unsigned token = m.x * 4U + lt / 32U;
                bool owner = c < chunks[i / 4U];
                unsigned dest = 0U, visible = 0U;
#pragma unroll
                for (unsigned b = 0U; b < 8U; ++b) {
                    const bool same = c < chunks[b / 4U] && meta_at(c, b).x == m.x;
                    if (b < i && same) owner = false;
                    if (same) { dest |= 1U << b; visible = visible < tile_visible[b / 4U] ? tile_visible[b / 4U] : visible; }
                }
                reg.dest[i] = owner ? dest : 0U;
                // Each task is one token's eight features, with no cross-token/vector aliasing.
                // Zero before conversion exactly as attend_body; no clamped nonfinite escapes.
                reg.k[i] = u32x4{}; reg.value[i] = u32x4{};
                if (owner && token < visible) {
                    const auto at = static_cast<std::uint64_t>(token) * row_elements + 8U * (lt % 32U);
                    reg.k[i] = load16(k_base + at); reg.value[i] = load16(v_base + at);
                }
            }
        };
        const auto stage = [&](unsigned c, const Regs& reg) {
#pragma unroll
            for (unsigned i = 0U; i < Tasks; ++i) {
                const unsigned token = meta_at(c, i).x * 4U + lt / 32U;
#pragma unroll
                for (unsigned b = 0U; b < 8U; ++b) if (reg.dest[i] & (1U << b)) {
                    const unsigned side = b / 4U, slot = 4U * (b % 4U) + lt / 32U;
                    const u32x4 k = token < tile_visible[side] ? reg.k[i] : u32x4{};
                    const u32x4 value = token < tile_visible[side] ? reg.value[i] : u32x4{};
                    *reinterpret_cast<uint4*>(&k_lds[side][slot][8U * (lt % 32U)]) =
                        uint4{bf16x2_to_f16x2(k.x), bf16x2_to_f16x2(k.y), bf16x2_to_f16x2(k.z), bf16x2_to_f16x2(k.w)};
                    const unsigned f[4] = {bf16x2_to_f16x2(value.x), bf16x2_to_f16x2(value.y),
                        bf16x2_to_f16x2(value.z), bf16x2_to_f16x2(value.w)};
#pragma unroll
                    for (unsigned j = 0U; j < 4U; ++j) {
                        v_lds[side][v_row(8U * (lt % 32U) + 2U * j)][slot] =
                            __builtin_bit_cast(_Float16, static_cast<unsigned short>(f[j] & 0xffffU));
                        v_lds[side][v_row(8U * (lt % 32U) + 2U * j + 1U)][slot] =
                            __builtin_bit_cast(_Float16, static_cast<unsigned short>(f[j] >> 16U));
                    }
                }
            }
            if (lt < 8U) meta_lds[lt / 4U][lt % 4U] = meta_at(c, lt);
        };
        if (steps) {
            Regs reg;
            fetch(0U, reg);
            for (unsigned c = 0U; c < steps; ++c) {
                stage(c, reg);
                lds_barrier();
                if (c + 1U < steps) fetch(c + 1U, reg);
                lds_barrier();
                lds_barrier(); // no image may be overwritten before both tiles finish value WMMA
            }
        }
        return;
    }
    const unsigned side = w / (2U * G), grp = w / 2U % G, fh = w % 2U;
    const unsigned r0 = (tile + side) * TR;
    const auto* header = headers[side];
    const unsigned row = col % TR, head = kvh * per_kv + hg * HG + grp * HPW + col / TR, r = r0 + row;
    // Q of this lane's column: a power-of-two scale from the largest |q| of the 256 features (the lane pair
    // (col, col + 16) reads half each), then this wave's 128 features as FP16 B fragments.
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
        // Scores in base 2: the softmax below uses exp2 (v_exp_f32).
        score_scale = ldexpf(1.0F, -e) / sqrtf(static_cast<float>(kDim)) * 1.4426950408889634F;
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k) {
#pragma unroll
            for (unsigned i = 0; i < 16U; i += 4U) {
                const float4 x = *reinterpret_cast<const float4*>(q + fh * (kDim / 2U) + 16U * k + i);
                qf[k][i] = static_cast<_Float16>(x.x * scale); qf[k][i + 1U] = static_cast<_Float16>(x.y * scale);
                qf[k][i + 2U] = static_cast<_Float16>(x.z * scale); qf[k][i + 3U] = static_cast<_Float16>(x.w * scale);
            }
            asm volatile("" : : "v"(qf[k]) : "memory"); // bound FP32 query-load lifetimes, as in attend_occ_body
        }
    }
    const unsigned visible = v.visible[r], bit = 1U << row, weights = header[4U + row];
    f32x8 acc[8];
#pragma unroll
    for (unsigned n = 0; n < 8U; ++n) acc[n] = f32x8{};
    float m_run = -INFINITY, l_run = 0.0F;
    for (unsigned c = 0U; c < steps; ++c) {
        lds_barrier(); // both tile images have been staged; loaders fetch the next step now
        if (c >= chunks[side]) { lds_barrier(); lds_barrier(); continue; }
        uint2 meta[4];
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) meta[i] = meta_lds[side][i];
        strix::attn_core::selected_flash_step(qf, acc, k_lds[side], v_lds[side], xch[side][grp], col, half, fh, lane,
            m_run, l_run, SelectedFlashPolicy<false>{meta, bit, visible, weights, half, score_scale, nullptr});
    }
    if constexpr (Capture) { // fixture-only final online state, before the lossy output rounding
        float* state = v.scratch[r] + static_cast<std::uint64_t>(head) * kPartial;
        if (fh == 0U && half == 0U) { state[0] = m_run; state[1] = l_run; }
#pragma unroll
        for (unsigned n = 0; n < 8U; ++n)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) state[2U + fh * (kDim / 2U) + 16U * n + 2U * e + half] = acc[n][e];
    }
    // One slice: combine_kernel's arithmetic for one split, in place.
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
template <unsigned Rows, bool Capture = false>
__global__ __launch_bounds__(512) __attribute__((amdgpu_waves_per_eu(8))) void attend_order_kv_pair_kernel(
    const Views<Rows>* groups, const std::uint16_t* kv, std::uint32_t kv_heads, std::uint32_t heads,
    std::uint32_t union_at, std::uint32_t ys) {
    constexpr unsigned pairs = Rows / 8U;
    const unsigned bx = blockIdx.x, tile = 2U * (bx % pairs), y = (bx / pairs) % ys, g = bx / (pairs * ys);
    attend_kv_pair_body<Rows, Capture>(groups[g], kv, kv_heads, heads, union_at, tile, y);
}
} // namespace qsa_fast
