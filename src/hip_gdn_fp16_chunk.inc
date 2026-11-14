// qwen-gdn-fp16-chunk, numerical revisions 2/3: chunked (C = 32) gated delta rule in WY form on FP16 WMMA operands with
// FP32 accumulation, for the GDN prefill on gfx1151. Bit-changing (FAST) identity; the production token recurrence is
// untouched. The route starts from the raw input-projection rows (conv fused) and ends at the gated-norm output.
//
// Interface: rows.qkv holds per row the raw projection values of q (key_heads x 128) | k (key_heads x 128) |
// v (value_heads x 128) in FP32; conv_state the BF16 conv history (4 per channel, entries 1..3 = rows -3..-1);
// rows.beta / rows.gate one value per (row, value head) after the production gates kernel (beta = sigmoid, gate =
// log-decay); value head h reads key head h % key_heads; the FP32 state stays in its GdnRecurrentStorage layout.
// The includer provides GdnRecurrentStorage, GdnPrefillRows, gdn_window_state_index, SmallWeightKind, load_small,
// float_to_bf16, bf16_to_float, round_bf16, gdn_window_key_sum, kGdnWindowKernel and <hip/hip_runtime.h>,
// <cstdint>, <mutex>, <stdexcept>, <vector>.
//
// Contract (per value head h, key head kh = h % key_heads, chunks of 32 rows from the call start, m valid rows;
// padded rows have q = k = v = 0, beta = 0, g = 0):
//   q, k, v  production conv values: sum = 0, sum += x_j * w_j (j = 0..3, x = BF16 of the input or the history),
//            SiLU sum / (1 + expf(-sum)), round_bf16. q, k L2-normed with the production tree (gdn_window_key_sum,
//            rsqrtf(sum + eps)), q *= rsqrtf(128). Q16 = fp16(q), K16 = fp16(k); v stays exact (BF16 value).
//   g_r      gate_r, values below -110 raised to -110 (NaN kept). G = inclusive prefix over the chunk (shuffle-up
//            ladder 1, 2, 4, 8, 16). e_r = expf(G_r), f_r = expf(G_{m-1} - G_r), eL = e_{m-1}; padded rows
//            carry G_{m-1} (so eL = e_31).
//   KK = W(K16, K16^T), QK = W(Q16, K16^T): native v_wmma_f32_16x16x16_f16 chains, K16 panels ascending.
//   D_rs = expf(G_r - G_s) (s <= r).  A_rs = (beta_r * KK_rs) * D_rs (s < r, else 0, FP32).
//   P16_rs = fp16(QK_rs * D_rs) (s <= r, else 0).
//   T = (I + A)^-1, FP32 forward substitution by pivots p = 0..30 ascending: x_i = fmaf(-A_ip, x_p, x_i).
//   T'16_rs = fp16(T_rs * beta_s).
//   Per chunk, FP32 state S[v][k] (kept as S^T tiles in accumulators):
//     X16 = fp16(S); KS = W(K16, X16^T), QS = W(Q16, X16^T) (8 panels)
//     Y16 = fp16(fmaf(-e_r, KS_rv, v_rv)); U = W(T'16, Y16) (lower panels); U16 = fp16(U); U2_16 = fp16(U * f_r)
//     S^T <- W(K16^T, U2_16, seed eL * S^T)                                (2 panels, state stays FP32)
//     O = W(P16, U16, seed QS * e_r)                                       (lower panels; rows < m are output)
//   Output: the epilogue receives O FP32 per (row, head) in the gated-norm lane layout (channel p * 32 + lane).
// Every call finishes its partial chunk; aligned (multiple of 32) call splits are bitwise invariant; results are
// run-to-run deterministic (fixed reduction orders, no atomics on values).
// Rev 3 keeps C32 and scan unchanged: prep walks 16-row conv segments and uses exp(G_r)*exp(-G_s) for bounded
// prefixes [-60,0] (diagonal 1); other intervals keep exp(G_r-G_s). This changes T/P rounding; separately qualify.
namespace gdn_fp16_chunk {
constexpr unsigned kC = 32U, kD = 128U, kSQ = 136U, kST = 40U, kMaxGroup = 4U;
constexpr float kGateFloor = -110.0F;
// Startup-cached; the fixture and engine use the same specialization selector. Rev 2 stays unchanged.
inline unsigned revision() {
    static const unsigned value = [] { const char* v = std::getenv("STRIX_QPF2K_GDN_FP16_CHUNK");
        return v && v[0] == '3' && v[1] == '\0' ? 3U : 2U; }();
    return value;
}
inline const char* identity() { return revision() == 3U ? "qwen-gdn-fp16-chunk rev 3" : "qwen-gdn-fp16-chunk rev 2"; }
typedef _Float16 h16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned u32x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ f32x8 wmma(h16x16 a, h16x16 b, f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
// 16 consecutive FP16 values (A row / B column fragment of the native layout).
__device__ __forceinline__ h16x16 frag(const _Float16* p) {
    const auto* v = reinterpret_cast<const u32x4*>(p);
    struct Raw { u32x4 a, b; } raw{v[0], v[1]};
    h16x16 f;
    __builtin_memcpy(&f, &raw, sizeof(f));
    return f;
}
// Accumulator tile (lane: column lane % 16, rows 2e + lane / 16) -> RNE FP16 B fragment (lane: column lane % 16,
// rows 0..15): lanes l and l ^ 16 exchange their packed pairs with one permlanex16 per pair.
__device__ __forceinline__ h16x16 acc_frag(const f32x8& acc) {
    const bool hi = (threadIdx.x & 16U) != 0U;
    const unsigned s0 = hi ? 0x01000504U : 0x05040100U, s1 = hi ? 0x03020706U : 0x07060302U;
    unsigned w[8];
#pragma unroll
    for (unsigned e = 0; e < 8U; e += 2U) {
        const _Float16 pair[2] = {static_cast<_Float16>(acc[e]), static_cast<_Float16>(acc[e + 1U])};
        unsigned a;
        __builtin_memcpy(&a, pair, 4);
        const unsigned b = __builtin_amdgcn_permlanex16(a, a, 0x76543210U, 0xfedcba98U, true, false);
        w[e] = __builtin_amdgcn_perm(b, a, s0);
        w[e + 1U] = __builtin_amdgcn_perm(b, a, s1);
    }
    h16x16 f;
    __builtin_memcpy(&f, w, sizeof(f));
    return f;
}
__device__ __forceinline__ float floor_gate(float g) { return g < kGateFloor ? kGateFloor : g; }
__device__ __forceinline__ float silu_bf16(float sum) { return round_bf16(sum / (1.0F + expf(-sum))); }

// Scratch: Q16 / K16 rows [key_heads][rows_p][128]; T'16 and P16 [value_heads][chunks][2][32][32];
// (e, f) [value_heads][chunks][32]; v (BF16 bits, exact) [value_heads][rows_p][128].
struct Scratch {
    _Float16* q16 = nullptr;
    _Float16* k16 = nullptr;
    _Float16* tp16 = nullptr;
    float2* ef = nullptr;
    std::uint16_t* v16 = nullptr;
};
inline std::uint64_t scratch_bytes(std::uint32_t count, std::uint32_t key_heads, std::uint32_t value_heads) {
    const std::uint64_t n = (count + kC - 1U) / kC, rows_p = n * kC;
    return 2U * (2U * key_heads * rows_p * kD + value_heads * n * 2U * kC * kC) + 8U * value_heads * n * kC +
        2U * value_heads * rows_p * kD;
}
inline Scratch scratch_view(void* base, std::uint32_t count, std::uint32_t key_heads, std::uint32_t value_heads) {
    const std::uint64_t n = (count + kC - 1U) / kC, rows_p = n * kC;
    Scratch s;
    s.q16 = static_cast<_Float16*>(base);
    s.k16 = s.q16 + key_heads * rows_p * kD;
    s.tp16 = s.k16 + key_heads * rows_p * kD;
    s.ef = reinterpret_cast<float2*>(s.tp16 + value_heads * n * 2U * kC * kC);
    s.v16 = reinterpret_cast<std::uint16_t*>(s.ef + value_heads * n * kC);
    return s;
}
// Grow-only per (device, stream, slot). Growth drains the stream before releasing the smaller block (its earlier
// launches are its only users), so steady-state calls never allocate.
inline void* persistent_scratch(hipStream_t stream, std::uint64_t bytes, unsigned slot = 0U) {
    struct Entry { int device; hipStream_t stream; unsigned slot; void* block; std::uint64_t bytes; };
    static std::mutex mutex;
    static std::vector<Entry> entries;
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) throw std::runtime_error("GDN fp16 chunk scratch: no device");
    const std::lock_guard<std::mutex> lock(mutex);
    Entry* entry = nullptr;
    for (auto& candidate : entries)
        if (candidate.device == device && candidate.stream == stream && candidate.slot == slot) entry = &candidate;
    if (entry && entry->bytes >= bytes) return entry->block;
    if (entry && entry->block) {
        if (hipStreamSynchronize(stream) != hipSuccess || hipFree(entry->block) != hipSuccess)
            throw std::runtime_error("GDN fp16 chunk scratch: release on growth failed");
        entry->block = nullptr;
        entry->bytes = 0;
    }
    void* block = nullptr;
    if (hipMalloc(&block, bytes) != hipSuccess) {
        (void)hipGetLastError();
        throw std::runtime_error("GDN fp16 chunk scratch: allocation failed");
    }
    if (!entry) {
        entries.push_back(Entry{device, stream, slot, nullptr, 0});
        entry = &entries.back();
    }
    entry->block = block;
    entry->bytes = bytes;
    return block;
}

// The conv inputs: BF16 history, depthwise weights (4 taps per channel) and, when the projection wrote them, the raw
// projection rows as BF16 (x16: per row q | k | v, then z at z_offset). The projection rounds every output to BF16, so
// x16 holds the same values as the FP32 rows (rows.qkv / rows.z); B16 kernels read it instead.
struct Conv {
    const std::uint16_t* history;
    const void* weights;
    SmallWeightKind kind;
    const std::uint16_t* x16 = nullptr;
    std::uint64_t x16_stride = 0, z_offset = 0;
};
template <bool B16>
__device__ __forceinline__ float in_qkv(const GdnPrefillRows& rows, const Conv& conv, std::uint64_t row, std::uint64_t ch) {
    if constexpr (B16) return bf16_to_float(conv.x16[row * conv.x16_stride + ch]);
    else return bf16_to_float(float_to_bf16(rows.qkv[row * rows.stride + ch]));
}
template <bool B16>
__device__ __forceinline__ float in_z(const GdnPrefillRows& rows, const Conv& conv, std::uint64_t row, std::uint64_t ch) {
    if constexpr (B16) return bf16_to_float(conv.x16[row * conv.x16_stride + conv.z_offset + ch]);
    else return rows.z[row * rows.stride + ch];
}
// prep: one CTA per (key head, chunk) for its value_heads / key_heads value heads.
template <bool B16, unsigned ConvRows = 8U, bool FactorDecay = false>
__global__ __launch_bounds__(256) void prep_kernel(GdnPrefillRows rows, std::uint32_t count, Conv conv,
    std::uint32_t key_heads, std::uint32_t value_heads, float epsilon, Scratch out) {
    __shared__ __attribute__((aligned(16))) unsigned char qk_raw[2U * kC * kSQ * 2U];  // Ks | Qs, then A
    __shared__ float KK[kC][kC + 1U], QK[kC][kC + 1U];
    static_assert(ConvRows == 8U || ConvRows == 16U);
    constexpr unsigned ConvJobs = kC / ConvRows;
    __shared__ float Gs[kMaxGroup][kC], Bs[kMaxGroup][kC];
    // Empty in rev 2 after specialization; no scratch/scan layout change.
    __shared__ float decay[FactorDecay ? kMaxGroup : 1U][FactorDecay ? kC : 1U];
    __shared__ float inverse[FactorDecay ? kMaxGroup : 1U][FactorDecay ? kC : 1U];
    static_assert(kMaxGroup * kC * (kC + 1U) * 4U <= sizeof(qk_raw), "A alias");
    auto* Ks = reinterpret_cast<_Float16*>(qk_raw);
    auto* Qs = Ks + kC * kSQ;
    auto* A = reinterpret_cast<float(*)[kC][kC + 1U]>(qk_raw);
    const unsigned kh = blockIdx.x, chunk = blockIdx.y, n = gridDim.y, tid = threadIdx.x, lane = tid % 32U, wave = tid / 32U;
    const unsigned l16 = lane % 16U, rh = lane / 16U;
    const std::uint32_t first = chunk * kC, m = min(kC, count - first), group = value_heads / key_heads;
    const std::uint64_t rows_p = static_cast<std::uint64_t>(n) * kC;
    const float q_scale = rsqrtf(static_cast<float>(kD));
    // Conv jobs (block, ConvRows-row segment): block 0 = q, 1 = k, 2 + j = v of value head kh + j * key_heads. A wave walks
    // its rows with a sliding window (each input row loaded once), lane channels p * 32 + lane.
    for (unsigned job = wave; job < (2U + group) * ConvJobs; job += 8U) {
        const unsigned block = job / ConvJobs, r0 = ConvRows * (job % ConvJobs);
        const bool is_v = block >= 2U;
        const unsigned h = kh + (block - 2U) * key_heads;
        const std::uint64_t base = (block == 0U ? static_cast<std::uint64_t>(kh)
            : block == 1U ? static_cast<std::uint64_t>(key_heads) + kh : 2U * static_cast<std::uint64_t>(key_heads) + h) * kD;
        float w4[4][kGdnWindowKernel], x[3][4];
#pragma unroll
        for (unsigned p = 0; p < 4U; ++p) {
#pragma unroll
            for (unsigned j = 0; j < kGdnWindowKernel; ++j)
                w4[p][j] = load_small(conv.weights, conv.kind, (base + p * 32U + lane) * kGdnWindowKernel + j);
#pragma unroll
            for (unsigned i = 0; i < 3U; ++i) {
                const auto u = static_cast<std::int64_t>(first + r0) - 3 + i;
                const std::uint64_t ch = base + p * 32U + lane;
                x[i][p] = r0 >= m ? 0.0F : u < 0 ? bf16_to_float(conv.history[ch * kGdnWindowKernel + 1U + static_cast<unsigned>(u + 3)])
                    : in_qkv<B16>(rows, conv, static_cast<std::uint64_t>(u), ch);
            }
        }
        for (unsigned rr = 0; rr < ConvRows; ++rr) {
            const unsigned r = r0 + rr;
            float v[4] = {};
            if (r < m) {
#pragma unroll
                for (unsigned p = 0; p < 4U; ++p) {
                    const float now = in_qkv<B16>(rows, conv, first + r, base + p * 32U + lane);
                    float sum = 0.0F;
                    sum += x[0][p] * w4[p][0];
                    sum += x[1][p] * w4[p][1];
                    sum += x[2][p] * w4[p][2];
                    sum += now * w4[p][3];
                    v[p] = silu_bf16(sum);
                    x[0][p] = x[1][p]; x[1][p] = x[2][p]; x[2][p] = now;
                }
                if (!is_v) {
                    float sq[4];
#pragma unroll
                    for (unsigned p = 0; p < 4U; ++p) sq[p] = v[p] * v[p];
                    const float inv = rsqrtf(gdn_window_key_sum(sq) + epsilon);
#pragma unroll
                    for (unsigned p = 0; p < 4U; ++p) v[p] = block == 0U ? v[p] * inv * q_scale : v[p] * inv;
                }
            }
            if (is_v) {
                std::uint16_t* v16 = out.v16 + (h * rows_p + first + r) * kD;
#pragma unroll
                for (unsigned p = 0; p < 4U; ++p) v16[p * 32U + lane] = r < m ? float_to_bf16(v[p]) : std::uint16_t{0};
            } else {
                _Float16* lds = (block == 0U ? Qs : Ks) + r * kSQ;
                _Float16* glob = (block == 0U ? out.q16 : out.k16) + (kh * rows_p + first + r) * kD;
#pragma unroll
                for (unsigned p = 0; p < 4U; ++p) {
                    const auto hv = static_cast<_Float16>(v[p]);
                    lds[p * 32U + lane] = hv;
                    glob[p * 32U + lane] = hv;
                }
            }
        }
    }
    if (wave < group) {
        const unsigned h = kh + wave * key_heads;
        float g = 0.0F, b = 0.0F;
        if (lane < m) {
            const std::uint64_t o = static_cast<std::uint64_t>(first + lane) * rows.stride + h;
            g = floor_gate(rows.gate[o]);
            b = rows.beta[o];
        }
#pragma unroll
        for (unsigned d = 1U; d < 32U; d *= 2U) {
            const float u = __shfl_up(g, d, 32);
            if (lane >= d) g += u;
        }
        // Padded rows carry G_{m-1} exactly (the ladder may associate row 31 differently).
        const float last = __shfl(g, static_cast<int>(m - 1U), 32);
        if (lane >= m) g = last;
        Gs[wave][lane] = g;
        Bs[wave][lane] = b;
        const float eg = expf(g);
        out.ef[(static_cast<std::uint64_t>(h) * n + chunk) * kC + lane] = make_float2(eg, expf(last - g));
        if constexpr (FactorDecay) {
            decay[wave][lane] = eg;
            // A bounded prefix keeps both factors normal. Reset/tiny/nonfinite chunks use the rev-2 expression.
            inverse[wave][lane] = last >= -60.0F && last <= 0.0F ? expf(-g) : 0.0F;
        }
    }
    __syncthreads();
    // KK / QK lower tiles (0,0) (1,0) (1,1): waves 0-2 KK, 3-5 QK.
    if (wave < 6U) {
        const bool isq = wave >= 3U;
        const unsigned job = wave % 3U, rt = job == 0U ? 0U : 1U, st = job == 2U ? 1U : 0U;
        const _Float16* lhs = isq ? Qs : Ks;
        f32x8 acc = {};
#pragma unroll
        for (unsigned p = 0; p < kD / 16U; ++p)
            acc = wmma(frag(lhs + (16U * rt + l16) * kSQ + 16U * p), frag(Ks + (16U * st + l16) * kSQ + 16U * p), acc);
        float (*dst)[kC + 1U] = isq ? QK : KK;
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) dst[16U * rt + 2U * e + rh][16U * st + l16] = acc[e];
    }
    __syncthreads();
    for (unsigned idx = tid; idx < group * kC * kC; idx += 256U) {
        const unsigned j = idx / (kC * kC), rs = idx % (kC * kC), r = rs / kC, s = rs % kC, h = kh + j * key_heads;
        float d = 0.0F;
        if (s <= r) {
            if constexpr (FactorDecay) {
                // exp(a-b) -> exp(a)*exp(-b): O(C) SFUs instead of O(C^2), with FP32 product rounding.
                // Check both endpoints too: arbitrary fixture/API gates need not be nonpositive.
                const bool bounded = Gs[j][kC - 1U] >= -60.0F && Gs[j][kC - 1U] <= 0.0F &&
                    Gs[j][r] >= -60.0F && Gs[j][r] <= 0.0F && Gs[j][s] >= -60.0F && Gs[j][s] <= 0.0F;
                d = bounded ? (s == r ? 1.0F : decay[j][r] * inverse[j][s]) : expf(Gs[j][r] - Gs[j][s]);
            } else d = expf(Gs[j][r] - Gs[j][s]);
        }
        A[j][r][s] = s < r ? Bs[j][r] * KK[r][s] * d : 0.0F;
        out.tp16[((static_cast<std::uint64_t>(h) * n + chunk) * 2U + 1U) * kC * kC + rs] =
            static_cast<_Float16>(s <= r ? QK[r][s] * d : 0.0F);
    }
    __syncthreads();
    // T = (I + A)^-1: lane = row, wave = 4 columns; T' = T diag(beta).
    for (unsigned j = 0; j < group; ++j) {
        const unsigned h = kh + j * key_heads, j0 = 4U * wave;
        float x[4];
#pragma unroll
        for (unsigned c = 0; c < 4U; ++c) x[c] = lane == j0 + c ? 1.0F : 0.0F;
#pragma unroll
        for (unsigned p = 0; p < kC - 1U; ++p) {
            if (p < j0) continue;
            const float a = A[j][lane][p];
#pragma unroll
            for (unsigned c = 0; c < 4U; ++c) {
                const float xp = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x[c]), static_cast<int>(p)));
                x[c] = fmaf(-a, xp, x[c]);
            }
        }
        _Float16 t[4];
#pragma unroll
        for (unsigned c = 0; c < 4U; ++c) t[c] = static_cast<_Float16>(x[c] * Bs[j][j0 + c]);
        u32x2 packed;
        __builtin_memcpy(&packed, t, sizeof(packed));
        *reinterpret_cast<u32x2*>(out.tp16 + (static_cast<std::uint64_t>(h) * n + chunk) * 2U * kC * kC + lane * kC + j0) = packed;
    }
}

// Core epilogues. RawCore (kRows = false) stores the raw FP32 core rows. A row epilogue (kRows = true) gets each valid
// row's 128 values of the head in the gated-norm lane layout (channel p * 32 + lane) with that row's z and its own
// per-lane constants (weights(w) once per CTA): epilogue(row, value_head, value, z, w).
struct RawCore {
    static constexpr bool kRows = false;
    __device__ void weights(unsigned, float (&)[4]) const {}
    __device__ void operator()(std::uint64_t, unsigned, const float (&)[4], const float (&)[4], const float (&)[4]) const {}
};

// Row epilogues opt into scalar wave rows with `static constexpr bool kWaveRows = true` (they then form their output
// pointers with strix::core::wave_row); without it rows stay per-lane.
template <class E> inline constexpr bool kEpilogueWaveRows = false;
template <class E> requires requires { E::kWaveRows; } inline constexpr bool kEpilogueWaveRows<E> = E::kWaveRows;

// scan: one CTA (8 waves) per value head; wave w owns value columns 16 w .. 16 w + 15 and holds S^T for them
// (8 tiles, 64 FP32 per lane). Each chunk's operands are staged once in LDS from registers loaded during the previous
// chunk; v is already conv'd in prep; z (row epilogues) is loaded late in the chunk. A wave is 32 consecutive threads.
template <GdnRecurrentStorage Storage, bool B16, class Epilogue = RawCore>
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(6))) void scan_kernel(GdnPrefillRows rows, std::uint32_t count, float* state,
    std::uint32_t key_heads, Conv conv, Scratch in, Epilogue epilogue = {}) {
    __shared__ __attribute__((aligned(16))) unsigned char qk_raw[2U * kC * kSQ * 2U];  // Qs | Ks, then O (FP32)
    __shared__ __attribute__((aligned(16))) _Float16 KTs[kD * kST];
    __shared__ __attribute__((aligned(16))) _Float16 Ts[kC * kST];
    __shared__ __attribute__((aligned(16))) _Float16 Ps[kC * kST];
    __shared__ __attribute__((aligned(16))) std::uint16_t Vs[kC * kSQ];
    __shared__ float2 Gs[kC];
    static_assert(kC * kSQ * 4U <= sizeof(qk_raw), "O alias");
    auto* Qs = reinterpret_cast<_Float16*>(qk_raw);
    auto* Ks = Qs + kC * kSQ;
    auto* Os = reinterpret_cast<float*>(qk_raw);
    const unsigned head = blockIdx.x, tid = threadIdx.x, lane = tid % 32U, w = tid / 32U, l16 = lane % 16U, rh = lane / 16U;
    const unsigned col = 16U * w + l16;
    // Row policy of the epilogue: with kWaveRows the wave's row index is a scalar, so z and epilogue rows w + 8 i get
    // scalar address bases (core/wave_uniform.hpp) and no per-lane row pointers stay live across the chunk loop beside
    // the 64-register state.
    const unsigned wrow = kEpilogueWaveRows<Epilogue> ? strix::core::wave_index(tid) : w;
    const std::uint32_t n = (count + kC - 1U) / kC;
    const std::uint64_t rows_p = static_cast<std::uint64_t>(n) * kC, kh = head % key_heads;
    const _Float16* q16 = in.q16 + kh * rows_p * kD;
    const _Float16* k16 = in.k16 + kh * rows_p * kD;
    const std::uint16_t* v16 = in.v16 + static_cast<std::uint64_t>(head) * rows_p * kD;
    float wn[4];
    epilogue.weights(lane, wn);

    u32x2 pq[4], pk[4];
    u32x4 ptp;
    float2 pg;
    u32x4 pv[2];      // v rows (BF16 bits)
    float zr[4][4];   // z of rows w + 8 i, channels p * 32 + lane (loaded late in the chunk)
    const auto prefetch = [&](std::uint32_t c) {
        const std::uint64_t r0 = static_cast<std::uint64_t>(c) * kC;
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            const unsigned idx = tid + 256U * j, r = idx / 32U, i = (idx % 32U) * 4U;
            pq[j] = *reinterpret_cast<const u32x2*>(q16 + (r0 + r) * kD + i);
            pk[j] = *reinterpret_cast<const u32x2*>(k16 + (r0 + r) * kD + i);
        }
        ptp = *reinterpret_cast<const u32x4*>(in.tp16 + ((static_cast<std::uint64_t>(head) * n + c) * 2U + tid / 128U) * kC * kC +
            (tid % 128U) * 8U);
        pg = in.ef[(static_cast<std::uint64_t>(head) * n + c) * kC + tid % kC];
#pragma unroll
        for (unsigned j = 0; j < 2U; ++j) {
            const unsigned idx = tid + 256U * j, r = idx / 16U, i = (idx % 16U) * 8U;
            pv[j] = *reinterpret_cast<const u32x4*>(v16 + (r0 + r) * kD + i);
        }
    };
    const auto prefetch_z = [&](std::uint32_t c) {
        if constexpr (Epilogue::kRows) {
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) {
                const std::uint32_t row = c * kC + wrow + 8U * i, t = row < count ? row : count - 1U;
#pragma unroll
                for (unsigned p = 0; p < 4U; ++p) zr[i][p] = in_z<B16>(rows, conv, t, static_cast<std::uint64_t>(head) * kD + 32U * p + lane);
            }
        }
    };
    // Incoming state: S^T tile kt, element e = S[col][16 kt + 2 e + rh].
    f32x8 st[8];
#pragma unroll
    for (unsigned kt = 0; kt < 8U; ++kt)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e)
            st[kt][e] = state[gdn_window_state_index<Storage>(head, 16U * kt + 2U * e + rh, col, kD)];
    prefetch(0U);
    for (std::uint32_t c = 0; c < n; ++c) {
        const std::uint32_t first = c * kC, m = min(kC, count - first);
        // Stage the chunk.
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            const unsigned idx = tid + 256U * j, r = idx / 32U, i = (idx % 32U) * 4U;
            *reinterpret_cast<u32x2*>(Qs + r * kSQ + i) = pq[j];
            *reinterpret_cast<u32x2*>(Ks + r * kSQ + i) = pk[j];
            _Float16 k4[4];
            __builtin_memcpy(k4, &pk[j], sizeof(k4));
#pragma unroll
            for (unsigned q = 0; q < 4U; ++q) KTs[(i + q) * kST + r] = k4[q];
        }
        {
            const unsigned x = tid % 128U, r = x / 4U, c8 = (x % 4U) * 8U;
            *reinterpret_cast<u32x4*>((tid < 128U ? Ts : Ps) + r * kST + c8) = ptp;
        }
        if (tid < kC) Gs[tid] = pg;
#pragma unroll
        for (unsigned j = 0; j < 2U; ++j) {
            const unsigned idx = tid + 256U * j, r = idx / 16U, i = (idx % 16U) * 8U;
            *reinterpret_cast<u32x4*>(Vs + r * kSQ + i) = pv[j];
        }
        __syncthreads();
        // Next chunk's operands (clamped: an unconditional issue keeps the loads in flight across the compute).
        prefetch(min(c + 1U, n - 1U));

        // KS = W(K16, X16^T), QS = W(Q16, X16^T).
        f32x8 ks[2] = {}, qs[2] = {};
#pragma unroll
        for (unsigned kt = 0; kt < 8U; ++kt) {
            const h16x16 xb = acc_frag(st[kt]);
#pragma unroll
            for (unsigned rt = 0; rt < 2U; ++rt) {
                ks[rt] = wmma(frag(Ks + (16U * rt + l16) * kSQ + 16U * kt), xb, ks[rt]);
                qs[rt] = wmma(frag(Qs + (16U * rt + l16) * kSQ + 16U * kt), xb, qs[rt]);
            }
            __builtin_amdgcn_sched_barrier(0);   // keeps each panel's fragment loads next to their use
        }
        // Y = v - e KS; U = W(T', Y); U16, U2_16 = fp16(U f).
        h16x16 yb[2], ub[2], ub2[2];
#pragma unroll
        for (unsigned rt = 0; rt < 2U; ++rt) {
            f32x8 y;
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned r = 16U * rt + 2U * e + rh;
                y[e] = fmaf(-Gs[r].x, ks[rt][e], bf16_to_float(Vs[r * kSQ + col]));
            }
            yb[rt] = acc_frag(y);
        }
#pragma unroll
        for (unsigned rt = 0; rt < 2U; ++rt) {
            f32x8 u = {};
#pragma unroll
            for (unsigned s = 0; s <= rt; ++s) u = wmma(frag(Ts + (16U * rt + l16) * kST + 16U * s), yb[s], u);
            ub[rt] = acc_frag(u);
            f32x8 u2;
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) u2[e] = u[e] * Gs[16U * rt + 2U * e + rh].y;
            ub2[rt] = acc_frag(u2);
        }
        __builtin_amdgcn_sched_barrier(0);
        prefetch_z(c);   // the epilogue's z: in flight across the state update and O
        __builtin_amdgcn_sched_barrier(0);
        // S^T <- W(K16^T, U2_16, seed eL S^T).
        const float el = Gs[kC - 1U].x;
#pragma unroll
        for (unsigned kt = 0; kt < 8U; ++kt) {
            f32x8 acc;
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) acc[e] = st[kt][e] * el;
#pragma unroll
            for (unsigned s = 0; s < 2U; ++s) acc = wmma(frag(KTs + (16U * kt + l16) * kST + 16U * s), ub2[s], acc);
            st[kt] = acc;
        }
        // O = W(P16, U16, seed e QS).
        if constexpr (Epilogue::kRows) __syncthreads();   // every wave is done with Qs / Ks (O aliases them)
#pragma unroll
        for (unsigned rt = 0; rt < 2U; ++rt) {
            f32x8 o;
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) o[e] = qs[rt][e] * Gs[16U * rt + 2U * e + rh].x;
#pragma unroll
            for (unsigned s = 0; s <= rt; ++s) o = wmma(frag(Ps + (16U * rt + l16) * kST + 16U * s), ub[s], o);
            if constexpr (Epilogue::kRows) {
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) Os[(16U * rt + 2U * e + rh) * kSQ + col] = o[e];
            } else {
                float* core = rows.core + static_cast<std::uint64_t>(first) * rows.stride + head * kD + col;
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) {
                    const unsigned r = 16U * rt + 2U * e + rh;
                    if (r < m) core[static_cast<std::uint64_t>(r) * rows.stride] = o[e];
                }
            }
        }
        if constexpr (Epilogue::kRows) {
            __syncthreads();
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) {
                const unsigned r = wrow + 8U * i;
                if (r < m) {
                    const float value[4] = {Os[r * kSQ + lane], Os[r * kSQ + 32U + lane], Os[r * kSQ + 64U + lane], Os[r * kSQ + 96U + lane]};
                    epilogue(first + r, head, value, zr[i], wn);
                }
            }
        }
        __syncthreads();   // the next stage overwrites every LDS buffer
    }
#pragma unroll
    for (unsigned kt = 0; kt < 8U; ++kt)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e)
            state[gdn_window_state_index<Storage>(head, 16U * kt + 2U * e + rh, col, kD)] = st[kt][e];
}

// The new conv history: the BF16 inputs of rows count - 4 .. count - 1 (the old history below row 0); launched after
// both readers of the old history.
template <bool B16>
__global__ __launch_bounds__(128) void conv_history_kernel(GdnPrefillRows rows, std::uint32_t count, Conv conv, std::uint16_t* conv_state) {
    const unsigned ch = threadIdx.x, base = blockIdx.x * kD;
    std::uint16_t* history = conv_state + static_cast<std::uint64_t>(base + ch) * kGdnWindowKernel;
    const std::uint16_t old[3] = {history[1], history[2], history[3]};
    std::uint16_t next[kGdnWindowKernel];
#pragma unroll
    for (unsigned i = 0; i < kGdnWindowKernel; ++i) {
        const auto row = static_cast<std::int64_t>(count) - static_cast<std::int64_t>(kGdnWindowKernel) + i;
        next[i] = row >= 0 ? float_to_bf16(in_qkv<B16>(rows, conv, static_cast<std::uint64_t>(row), base + ch))
                           : old[static_cast<unsigned>(row + 3)];
    }
#pragma unroll
    for (unsigned i = 0; i < kGdnWindowKernel; ++i) history[i] = next[i];
}
} // namespace gdn_fp16_chunk

// Stages (the fixture times them separately) and the full route. Requirements: head_dim 128, value_heads a multiple
// of key_heads with value_heads / key_heads <= 4, rows.stride % 4 == 0, rows.qkv 16-byte aligned. conv.x16 (optional)
// selects the BF16 projection rows.
inline void enqueue_gdn_fp16_chunk_prep(const gdn_fp16_chunk::Scratch& scratch, GdnPrefillRows rows, std::uint32_t count,
    gdn_fp16_chunk::Conv conv, std::uint32_t key_heads, std::uint32_t value_heads, float epsilon, hipStream_t stream) {
    const std::uint32_t n = (count + gdn_fp16_chunk::kC - 1U) / gdn_fp16_chunk::kC;
    const auto launch = [&]<bool B16>() {
        if (gdn_fp16_chunk::revision() == 3U)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_fp16_chunk::prep_kernel<B16, 16U, true>), dim3(key_heads, n), dim3(256), 0,
                stream, rows, count, conv, key_heads, value_heads, epsilon, scratch);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_fp16_chunk::prep_kernel<B16>), dim3(key_heads, n), dim3(256), 0,
            stream, rows, count, conv, key_heads, value_heads, epsilon, scratch);
    };
    if (conv.x16) launch.template operator()<true>(); else launch.template operator()<false>();
}
template <GdnRecurrentStorage Storage, class Epilogue = gdn_fp16_chunk::RawCore>
void enqueue_gdn_fp16_chunk_scan(const gdn_fp16_chunk::Scratch& scratch, GdnPrefillRows rows, std::uint32_t count, float* state,
    gdn_fp16_chunk::Conv conv, std::uint32_t key_heads, std::uint32_t value_heads, hipStream_t stream, Epilogue epilogue = {}) {
    if (conv.x16) hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_fp16_chunk::scan_kernel<Storage, true, Epilogue>), dim3(value_heads), dim3(256),
        0, stream, rows, count, state, key_heads, conv, scratch, epilogue);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_fp16_chunk::scan_kernel<Storage, false, Epilogue>), dim3(value_heads), dim3(256), 0,
        stream, rows, count, state, key_heads, conv, scratch, epilogue);
}
inline bool gdn_fp16_chunk_supported(std::uint32_t head_dim, std::uint32_t key_heads, std::uint32_t value_heads,
    const GdnPrefillRows& rows) {
    return head_dim == gdn_fp16_chunk::kD && key_heads && value_heads % key_heads == 0U &&
        value_heads / key_heads <= gdn_fp16_chunk::kMaxGroup && rows.stride % 4U == 0U &&
        reinterpret_cast<std::uintptr_t>(rows.qkv) % 16U == 0U;
}
// BF16 projection rows for the route (grow-only per (device, stream)): `rows` x `width` elements.
inline std::uint16_t* gdn_fp16_chunk_input16(hipStream_t stream, std::uint32_t rows, std::uint64_t width) {
    return static_cast<std::uint16_t*>(gdn_fp16_chunk::persistent_scratch(stream, 2U * rows * width, 1U));
}
template <GdnRecurrentStorage Storage, class Epilogue = gdn_fp16_chunk::RawCore>
void launch_gdn_fp16_chunk_conv_fast(GdnPrefillRows rows, std::uint32_t count, float* state, std::uint16_t* conv_state,
    const void* conv_weights, SmallWeightKind conv_kind, std::uint32_t key_heads, std::uint32_t value_heads, float epsilon,
    hipStream_t stream, Epilogue epilogue = {}, const std::uint16_t* x16 = nullptr, std::uint64_t x16_stride = 0,
    std::uint64_t z_offset = 0) {
    if (!count) return;
    if (!gdn_fp16_chunk_supported(gdn_fp16_chunk::kD, key_heads, value_heads, rows))
        throw std::runtime_error("GDN fp16 chunk: unsupported geometry or alignment");
    using namespace gdn_fp16_chunk;
    void* base = persistent_scratch(stream, scratch_bytes(count, key_heads, value_heads));
    const auto scratch = scratch_view(base, count, key_heads, value_heads);
    const Conv conv{conv_state, conv_weights, conv_kind, x16, x16_stride, z_offset};
    enqueue_gdn_fp16_chunk_prep(scratch, rows, count, conv, key_heads, value_heads, epsilon, stream);
    enqueue_gdn_fp16_chunk_scan<Storage, Epilogue>(scratch, rows, count, state, conv, key_heads, value_heads, stream, epilogue);
    if (x16) hipLaunchKernelGGL(HIP_KERNEL_NAME(conv_history_kernel<true>), dim3(2U * key_heads + value_heads), dim3(128), 0, stream,
        rows, count, conv, conv_state);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(conv_history_kernel<false>), dim3(2U * key_heads + value_heads), dim3(128), 0, stream,
        rows, count, conv, conv_state);
    if (hipGetLastError() != hipSuccess) throw std::runtime_error("GDN fp16 chunk: launch failed");
}
