// Opt-in cohort part 2. Every grid retains the session-local x coordinate;
// y selects the session (scorers) or a real, unpadded row (row-local stages).
// The legacy kernel entry points and their template instantiations stay intact.
constexpr unsigned kQsa2Rows = 32U, kQsa2Sessions = kQsa2Rows / 2U;
struct Qsa2NormArgs {
    float* values[kQsa2Rows];
    std::uint64_t position[kQsa2Rows];
    bool skip[kQsa2Rows];
};
struct Qsa2StoreRow {
    const float *index_key, *k, *v;
    std::uint16_t *raw, *kv, *journal;
    std::uint64_t position;
};
struct Qsa2StoreArgs { Qsa2StoreRow row[kQsa2Rows]; };
struct Qsa2IndexSession {
    QsaRowsScoreViews<8U> score;
    const std::uint16_t* raw;
    float* keys;
    std::uint16_t* packed;
    unsigned first, blocks, rows, ratio, grid;
    bool pooled;
};
struct Qsa2IndexArgs { Qsa2IndexSession session[kQsa2Sessions]; };
struct Qsa2SelectRow {
    float* scores;
    std::uint32_t* selected;
    unsigned visible, ratio, topk;
};
struct Qsa2SelectArgs { Qsa2SelectRow row[kQsa2Rows]; };
struct Qsa2AttentionSession { const std::uint16_t* kv; unsigned first, rows, maximum, slots, grid; bool compact; };
struct Qsa2AttentionArgs {
    QsaRowsAttentionViews<kQsa2Rows> views;
    Qsa2AttentionSession session[kQsa2Sessions];
    unsigned owner[kQsa2Rows];
};
// Leave room for the scalar arguments and rotary-source descriptors in the
// 4 KiB kernarg budget. No device allocation, upload, or lifetime fence is needed.
static_assert(sizeof(Qsa2NormArgs) < 3584U && sizeof(Qsa2StoreArgs) < 3584U);
static_assert(sizeof(Qsa2IndexArgs) < 3584U && sizeof(Qsa2SelectArgs) < 3584U);
static_assert(sizeof(Qsa2AttentionArgs) < 3584U);

template <unsigned Rows> __device__ __forceinline__ QsaRowsScoreViews<Rows> qsa2_score_views(const Qsa2IndexSession& s) {
    QsaRowsScoreViews<Rows> v{};
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r) {
        v.index_query[r] = s.score.index_query[r]; v.block_scores[r] = s.score.block_scores[r]; v.blocks[r] = s.score.blocks[r];
    }
    return v;
}
// The rows of one session, read in place from the kernel arguments. A local copy of the views indexed by the
// row of the wave lands in scratch memory (304 bytes per thread in the score kernels); these pointers do not.
template <unsigned Rows> struct Qsa2AttentionRows {
    const float* const* q_gate;
    const std::uint32_t* const* selected;
    float* const* scores;
    float* const* output;
    const std::uint32_t* count;
    bool compact_output;
};
template <unsigned Rows> __device__ __forceinline__ Qsa2AttentionRows<Rows> qsa2_attention_views(
    const Qsa2AttentionArgs& args, const Qsa2AttentionSession& s) {
    return {args.views.q_gate + s.first, args.views.selected + s.first, args.views.scores + s.first,
        args.views.output + s.first, args.views.count + s.first, s.compact};
}
template <class Body> __device__ __forceinline__ void qsa2_dispatch(unsigned rows, Body body) {
    if (rows == 2U) body(std::integral_constant<unsigned, 2U>{});
    else if (rows == 4U) body(std::integral_constant<unsigned, 4U>{});
    else body(std::integral_constant<unsigned, 8U>{});
}

// Exact body of block_score_rows_kernel; only blockIdx is virtualized.
template <SmallWeightKind Kind, unsigned Rows, class... Source>
__device__ __forceinline__ void qsa2_block_score_body(QsaRowsScoreViews<Rows> views, const std::uint16_t* raw_keys,
    std::uint32_t index_heads, std::uint32_t head_dim, std::uint32_t compress_ratio,
    const void* key_norm, float epsilon, std::uint32_t rotary_dim, float theta, uint3 cohort_block, Source... rotary_source) {
    const auto block = static_cast<std::uint32_t>(cohort_block.x);
    const auto feature = static_cast<std::uint32_t>(threadIdx.x);
    if (feature >= head_dim) return;
    extern __shared__ float shared[];
    auto* pooled = shared;
    auto* scratch = shared + head_dim;
    float mean = 0.0F;
    const auto token_start = static_cast<std::uint64_t>(block) * compress_ratio;
    if (!(rotary_source.check(token_start) && ...)) return;
    for (std::uint32_t offset = 0; offset < compress_ratio; ++offset)
        mean += bf16_to_float(raw_keys[(token_start + offset) * head_dim + feature]);
    mean /= static_cast<float>(compress_ratio);
    mean = round_bf16(mean);
    pooled[feature] = mean;
    scratch[feature] = mean * mean;
    __syncthreads();
    for (std::uint32_t stride = head_dim / 2U; stride > 0; stride >>= 1U) {
        if (feature < stride) scratch[feature] += scratch[feature + stride];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(scratch[0] / static_cast<float>(head_dim) + epsilon);
    pooled[feature] = round_bf16(mean * inv_rms * load_small_weight<Kind>(key_norm, feature));
    __syncthreads();
    float rotated = pooled[feature];
    if constexpr (sizeof...(Source) == 0) {
        if (feature < rotary_dim) {
            const auto half = rotary_dim / 2U;
            const auto pair = feature % half;
            const float exponent = 2.0F * static_cast<float>(pair) / static_cast<float>(rotary_dim);
            const float angle = static_cast<float>(token_start) * powf(theta, -exponent);
            const float cosine = cosf(angle);
            const float sine = sinf(angle);
            if (feature < half) rotated = pooled[feature] * cosine - pooled[feature + half] * sine;
            else rotated = pooled[feature] * cosine + pooled[feature - half] * sine;
        }
    } else {
        if (feature < rotary_dim) {
            const auto half = rotary_dim / 2U;
            const auto pair = feature % half;
            const float2 cs = (rotary_source.cs[token_start * 32ULL + pair], ...);
            const float cosine = cs.x;
            const float sine = cs.y;
            if (feature < half) rotated = pooled[feature] * cosine - pooled[feature + half] * sine;
            else rotated = pooled[feature] * cosine + pooled[feature - half] * sine;
        }
    }
    __syncthreads();
    pooled[feature] = round_bf16(rotated);
    __syncthreads();
    __shared__ float total;
    constexpr unsigned Span = Rows <= 8U ? Rows : 8U;
    const unsigned base = Rows <= 8U ? 0U : static_cast<unsigned>(cohort_block.y) * 8U;
    for (unsigned rr = 0; rr < Span; ++rr) {
        const unsigned r = base + rr;
        if (block >= views.blocks[r]) continue; // uniform per block
        if (feature == 0) total = 0.0F;
        __syncthreads();
        for (std::uint32_t head = 0; head < index_heads; ++head) {
            scratch[feature] = views.index_query[r][static_cast<std::uint64_t>(head) * head_dim + feature] * pooled[feature];
            __syncthreads();
            for (std::uint32_t stride = head_dim / 2U; stride > 0; stride >>= 1U) {
                if (feature < stride) scratch[feature] += scratch[feature + stride];
                __syncthreads();
            }
            if (feature == 0) total += fmaxf(scratch[0], 0.0F);
            __syncthreads();
        }
        if (feature == 0) views.block_scores[r][block] = total / sqrtf(static_cast<float>(head_dim));
    }
}

// Exact body of block_pool_kernel; only blockIdx is virtualized.
template <SmallWeightKind Kind, bool Batch = false, bool PackedOnly = false, class... Source>
__device__ __forceinline__ void qsa2_block_pool_body(const std::uint16_t* raw_keys, std::uint32_t first_block, std::uint32_t head_dim,
    std::uint32_t compress_ratio, const void* key_norm, float epsilon, std::uint32_t rotary_dim, float theta, float* pooled_keys,
    const uint2* bounds, std::uint32_t groups, std::uint16_t* packed_keys,
    uint3 cohort_block, Source... rotary_source) {
    auto local = static_cast<std::uint32_t>(cohort_block.x);
    if constexpr (Batch) {
        const auto absolute = first_block + local;
        const auto group = qsa_prefill_block_group(bounds, groups, absolute);
        if (group == groups) return;
        const auto range = bounds[group];
        if (absolute < range.x) return;
        local = absolute - range.x;
        first_block = range.x;
    }
    const auto block = first_block + local;
    const auto feature = static_cast<std::uint32_t>(threadIdx.x);
    if (feature >= head_dim) return;
    extern __shared__ float shared[];
    auto* pooled = shared;
    auto* scratch = shared + head_dim;
    float mean = 0.0F;
    const auto token_start = static_cast<std::uint64_t>(block) * compress_ratio;
    if (!(rotary_source.check(token_start) && ...)) return;
    for (std::uint32_t offset = 0; offset < compress_ratio; ++offset)
        mean += bf16_to_float(raw_keys[(token_start + offset) * head_dim + feature]);
    mean /= static_cast<float>(compress_ratio);
    mean = round_bf16(mean);
    pooled[feature] = mean;
    scratch[feature] = mean * mean;
    __syncthreads();
    for (std::uint32_t stride = head_dim / 2U; stride > 0; stride >>= 1U) {
        if (feature < stride) scratch[feature] += scratch[feature + stride];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(scratch[0] / static_cast<float>(head_dim) + epsilon);
    pooled[feature] = round_bf16(mean * inv_rms * load_small_weight<Kind>(key_norm, feature));
    __syncthreads();
    float rotated = pooled[feature];
    if constexpr (sizeof...(Source) == 0) {
        if (feature < rotary_dim) {
            const auto half = rotary_dim / 2U;
            const auto pair = feature % half;
            const float exponent = 2.0F * static_cast<float>(pair) / static_cast<float>(rotary_dim);
            const float angle = static_cast<float>(token_start) * powf(theta, -exponent);
            const float cosine = cosf(angle);
            const float sine = sinf(angle);
            if (feature < half) rotated = pooled[feature] * cosine - pooled[feature + half] * sine;
            else rotated = pooled[feature] * cosine + pooled[feature - half] * sine;
        }
    } else {
        if (feature < rotary_dim) {
            const auto half = rotary_dim / 2U;
            const auto pair = feature % half;
            const float2 cs = (rotary_source.cs[token_start * 32ULL + pair], ...);
            const float cosine = cs.x;
            const float sine = cs.y;
            if (feature < half) rotated = pooled[feature] * cosine - pooled[feature + half] * sine;
            else rotated = pooled[feature] * cosine + pooled[feature - half] * sine;
        }
    }
    const float stored = round_bf16(rotated);
    // PackedOnly permits a null pooled_keys pointer and never stores an FP32 copy.
    if constexpr (!PackedOnly) pooled_keys[static_cast<std::uint64_t>(block) * head_dim + feature] = stored;
    // packed_keys (STRIX_QSA_POOL_PACKED): the same key as its upper 16 bits. round_bf16 leaves the lower 16 bits zero,
    // so widening (bits << 16) gives back the stored float bit for bit, NaN payloads and signed zeros included.
    if (packed_keys) packed_keys[static_cast<std::uint64_t>(block) * head_dim + feature] = static_cast<std::uint16_t>(__float_as_uint(stored) >> 16U);
}

// Exact body of attention_score_rows_wave_kernel; only blockIdx is virtualized.
template <unsigned Rows, unsigned PerWave = 1U, bool Int8 = false>
__device__ __forceinline__ void qsa2_attention_wave_body(Qsa2AttentionRows<Rows> views,
    const std::uint16_t* kv_state, std::uint32_t maximum_selected, std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t slots, uint3 cohort_block) {
#pragma clang fp contract(off)
    static_assert(kThreads == 256U);
    const auto lane = threadIdx.x % 32U;
    const auto wave = cohort_block.x * 8U + threadIdx.x / 32U;
    const auto groups = (slots + PerWave - 1U) / PerWave;
    const auto r = wave / (attention_heads * groups);
    if (r >= Rows) return;
    const auto head = wave / groups % attention_heads, first = wave % groups * PerWave;
    if (first >= views.count[r]) return;
    const auto kv_head = head / (attention_heads / kv_heads);
    const auto* q = views.q_gate[r] + static_cast<std::uint64_t>(head) * head_dim * 2ULL;
    float qv[8];
#pragma unroll
    for (unsigned j = 0; j < 8U; ++j) { const auto feature = lane + 32U * j; qv[j] = feature < head_dim ? q[feature] : 0.0F; }
    float p[PerWave][8];
#pragma unroll
    for (unsigned i = 0; i < PerWave; ++i) {
        const auto slot = first + i < views.count[r] ? first + i : first;
        const auto token = views.selected[r][slot];
        const auto* k = kv_state + static_cast<std::uint64_t>(token) * kv_heads * head_dim * 2ULL +
            static_cast<std::uint64_t>(kv_head) * head_dim;
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            const auto feature = lane + 32U * j;
            float product = 0.0F;
            if constexpr (Int8) {
                if (feature < head_dim)
                    product = qv[j] * kv_int8_load(kv_state, token, kv_heads * head_dim, kv_heads, 0U, kv_head, head_dim, feature);
            } else if (feature < head_dim) product = qv[j] * bf16_to_float(k[feature]);
            p[i][j] = product;
        }
    }
#pragma unroll
    for (unsigned i = 0; i < PerWave; ++i) {
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) p[i][j] += p[i][j + 4U];
        p[i][0] += p[i][2]; p[i][1] += p[i][3];
        float sum = p[i][0] + p[i][1];
#pragma unroll
        for (unsigned stride = 16U; stride > 0U; stride >>= 1U) sum += __shfl_down(sum, stride, 32);
        if (lane == 0U && first + i < views.count[r])
            views.scores[r][static_cast<std::uint64_t>(head) * maximum_selected + first + i] = sum / sqrtf(static_cast<float>(head_dim));
    }
}

// Exact body of attention_score_rows_group_kernel; only blockIdx is virtualized.
template <unsigned Rows, unsigned G, unsigned Slots, bool Scatter, bool Int8 = false>
__device__ __forceinline__ void qsa2_attention_group_body(Qsa2AttentionRows<Rows> views,
    const std::uint16_t* kv_state, std::uint32_t maximum_selected, std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t slots, uint3 cohort_block) {
#pragma clang fp contract(off)
    const auto lane = threadIdx.x % 32U;
    const auto wave = cohort_block.x * 8U + threadIdx.x / 32U;
    const auto groups = (slots + Slots - 1U) / Slots, head_groups = attention_heads / G;
    const auto r = wave / (head_groups * groups);
    if (r >= Rows) return;
    const auto head0 = wave / groups % head_groups * G, first = wave % groups * Slots;
    if (first >= views.count[r]) return;
    const auto kv_head = head0 / (attention_heads / kv_heads);
    float qv[G][8];
#pragma unroll
    for (unsigned h = 0; h < G; ++h) {
        const auto* q = views.q_gate[r] + static_cast<std::uint64_t>(head0 + h) * head_dim * 2ULL;
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) { const auto feature = lane + 32U * j; qv[h][j] = feature < head_dim ? q[feature] : 0.0F; }
    }
    const float scale = sqrtf(static_cast<float>(head_dim));
    const auto last = min(first + Slots, views.count[r]);
    for (auto slot = first; slot < last; ++slot) {
        const auto token = views.selected[r][slot];
        const auto* k = kv_state + static_cast<std::uint64_t>(token) * kv_heads * head_dim * 2ULL + static_cast<std::uint64_t>(kv_head) * head_dim;
        float kf[8];
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            const auto feature = lane + 32U * j;
            if constexpr (Int8)
                kf[j] = feature < head_dim ? kv_int8_load(kv_state, token, kv_heads * head_dim, kv_heads, 0U, kv_head, head_dim, feature) : 0.0F;
            else kf[j] = feature < head_dim ? bf16_to_float(k[feature]) : 0.0F;
        }
        constexpr unsigned P = G <= 2U ? 2U : G <= 4U ? 4U : G <= 8U ? 8U : 16U;
        float partial[P];
#pragma unroll
        for (unsigned h = 0; h < P; ++h) {
            if (h >= G) { partial[h] = 0.0F; continue; }
            float p[8];
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) p[j] = lane + 32U * j < head_dim ? qv[h][j] * kf[j] : 0.0F;
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) p[j] += p[j + 4U];
            p[0] += p[2]; p[1] += p[3];
            partial[h] = p[0] + p[1];
        }
        if constexpr (Scatter) {
            qsa_scatter<P>(partial, lane);
            const auto h = qsa_scatter_head<P>(lane);
            if (h < G && (lane & ((32U / P) - 1U)) == 0U)
                views.scores[r][static_cast<std::uint64_t>(head0 + h) * maximum_selected + slot] = partial[0] / scale;
        } else {
#pragma unroll
            for (unsigned h = 0; h < G; ++h) {
                float sum = partial[h];
                sum += qsa_lane_xor<16U>(sum); sum += qsa_lane_xor<8U>(sum); sum += qsa_lane_xor<4U>(sum);
                sum += qsa_lane_xor<2U>(sum); sum += qsa_lane_xor<1U>(sum);
                if (lane == 0U) views.scores[r][static_cast<std::uint64_t>(head0 + h) * maximum_selected + slot] = sum / scale;
            }
        }
    }
}

// Exact body of attention_score_rows_group256_kernel; only blockIdx is virtualized.
template <unsigned Rows, unsigned G, unsigned Slots, bool Int8 = false>
__device__ __forceinline__ void qsa2_attention_group256_body(Qsa2AttentionRows<Rows> views,
    const std::uint16_t* kv_state, std::uint32_t maximum_selected, std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t slots, uint3 cohort_block) {
#pragma clang fp contract(off)
    static_assert(Slots <= 32U);
    constexpr unsigned Dim = 256U; // head_dim == Dim (checked by the caller)
    const auto lane = threadIdx.x % 32U;
    const auto wave = cohort_block.x * 8U + threadIdx.x / 32U;
    const auto groups = (slots + Slots - 1U) / Slots, head_groups = attention_heads / G;
    const auto r = wave / (head_groups * groups);
    if (r >= Rows) return;
    const auto head0 = wave / groups % head_groups * G, first = wave % groups * Slots;
    if (first >= views.count[r]) return;
    const auto kv_head = head0 / (attention_heads / kv_heads);
    float qv[G][8];
#pragma unroll
    for (unsigned h = 0; h < G; ++h) {
        const auto* q = views.q_gate[r] + static_cast<std::uint64_t>(head0 + h) * Dim * 2ULL;
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) qv[h][j] = q[lane + 32U * j];
    }
    const float scale = sqrtf(static_cast<float>(head_dim));
    const auto last = min(first + Slots, views.count[r]);
    const auto tokens = lane < last - first ? views.selected[r][first + lane] : 0U;
    float* const scores = views.scores[r]; // loaded once: a store could alias the argument block
    const auto* kbase = kv_state + static_cast<std::uint64_t>(kv_head) * Dim + lane;
    const auto row_elements = static_cast<std::uint64_t>(kv_heads) * Dim * 2ULL;
    // Raw BF16 bits travel to the next slot; they become floats where the products use them
    // (int8 KV: the dequantized values travel instead).
    using Raw = std::conditional_t<Int8, float, std::uint16_t>;
    Raw raw[8];
    const auto load = [&](std::uint32_t index, Raw (&k)[8]) {
        const auto token = static_cast<std::uint32_t>(__builtin_amdgcn_readlane(static_cast<int>(tokens), static_cast<int>(index)));
        if constexpr (Int8) {
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) k[j] = kv_int8_load(kv_state, token, kv_heads * Dim, kv_heads, 0U, kv_head, Dim, lane + 32U * j);
        } else {
            const auto* row = kbase + static_cast<std::uint64_t>(token) * row_elements;
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) k[j] = row[32U * j];
        }
    };
    load(0U, raw);
    constexpr unsigned P = G <= 2U ? 2U : G <= 4U ? 4U : G <= 8U ? 8U : 16U;
    for (auto slot = first; slot < last; ++slot) {
        float kf[8];
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            if constexpr (Int8) kf[j] = raw[j];
            else kf[j] = bf16_to_float(raw[j]);
        }
        if (slot + 1U < last) load(slot + 1U - first, raw);
        float partial[P];
#pragma unroll
        for (unsigned h = 0; h < P; ++h) {
            if (h >= G) { partial[h] = 0.0F; continue; }
            float p[8];
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) p[j] = qv[h][j] * kf[j];
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) p[j] += p[j + 4U];
            p[0] += p[2]; p[1] += p[3];
            partial[h] = p[0] + p[1];
        }
        qsa_scatter<P>(partial, lane);
        const auto h = qsa_scatter_head<P>(lane);
        if (h < G && (lane & ((32U / P) - 1U)) == 0U)
            scores[static_cast<std::uint64_t>(head0 + h) * maximum_selected + slot] = partial[0] / scale;
    }
}

template <SmallWeightKind Kind, class... Source>
__global__ void qsa2_norm_kernel(Qsa2NormArgs args, unsigned heads, unsigned dim, unsigned stride,
    const void* weights, float epsilon, unsigned rotary, float theta, Source... source) {
    const auto r = blockIdx.y;
    if (!args.skip[r]) norm_rope_body<Kind, Source...>(args.values[r], heads, dim, stride, weights,
        epsilon, rotary, theta, args.position[r], GraphPositionKernelArguments{}, source...);
}
__global__ void qsa2_store_kernel(Qsa2StoreArgs args, unsigned index_dim, unsigned kv_width) {
    const auto& r = args.row[blockIdx.y];
    const auto feature = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (feature < index_dim) {
        const auto bits = float_to_bf16(r.index_key[feature]);
        r.raw[r.position * static_cast<std::uint64_t>(index_dim) + feature] = bits;
        if (r.journal) r.journal[feature] = bits;
    }
    if (feature < kv_width) {
        const auto row = r.position * static_cast<std::uint64_t>(kv_width) * 2ULL;
        r.kv[row + feature] = float_to_bf16(r.k[feature]);
        r.kv[row + kv_width + feature] = float_to_bf16(r.v[feature]);
    }
}
__global__ void qsa2_store_int8_kernel(Qsa2StoreArgs args, unsigned width, unsigned heads, unsigned dim) {
    const auto& r = args.row[blockIdx.z];
    const auto half = static_cast<std::uint32_t>(blockIdx.y);
    kv_int8_store_head(half ? r.v : r.k, r.kv, r.position, width, heads, half,
        static_cast<std::uint32_t>(blockIdx.x), dim);
}
template <SmallWeightKind Kind, class... Source>
__global__ void qsa2_pool_kernel(Qsa2IndexArgs args, unsigned dim, const void* norm, float epsilon,
    unsigned rotary, float theta, Source... source) {
    const auto& s = args.session[blockIdx.y];
    if (!s.pooled || blockIdx.x >= s.blocks - s.first) return;
    if (!s.keys && s.packed)
        qsa2_block_pool_body<Kind, false, true, Source...>(s.raw, s.first, dim, s.ratio, norm, epsilon, rotary, theta,
            s.keys, nullptr, 0U, s.packed, uint3{blockIdx.x, 0U, 0U}, source...);
    else qsa2_block_pool_body<Kind, false, false, Source...>(s.raw, s.first, dim, s.ratio, norm, epsilon, rotary, theta,
        s.keys, nullptr, 0U, s.packed, uint3{blockIdx.x, 0U, 0U}, source...);
}
template <SmallWeightKind Kind, class... Source>
__global__ void qsa2_index_raw_kernel(Qsa2IndexArgs args, unsigned heads, unsigned dim, const void* norm,
    float epsilon, unsigned rotary, float theta, Source... source) {
    const auto& s = args.session[blockIdx.y];
    if (s.pooled || blockIdx.x >= s.blocks) return;
    qsa2_dispatch(s.rows, [&](auto width) {
        constexpr auto Rows = decltype(width)::value;
        qsa2_block_score_body<Kind, Rows, Source...>(qsa2_score_views<Rows>(s), s.raw, heads, dim, s.ratio,
            norm, epsilon, rotary, theta, uint3{blockIdx.x, 0U, 0U}, source...);
    });
}
template <unsigned Wanted>
__global__ __launch_bounds__(256) void qsa2_index_pooled_kernel(Qsa2IndexArgs args, unsigned dim) {
    const auto& s = args.session[blockIdx.y];
    if (!s.pooled || blockIdx.x >= s.grid) return;
    qsa2_dispatch(s.rows, [&](auto width) {
        constexpr auto Rows = decltype(width)::value, R = Wanted < Rows ? Wanted : Rows;
        // The session's views are read in place (entries 0..Rows-1 of its 8-entry table).
        if (!s.keys && s.packed) block_score_pooled_rows_body<Rows, R, std::uint16_t, false, QsaRowsScoreViews<8U>>(s.score, s.packed, s.blocks, dim, nullptr);
        else block_score_pooled_rows_body<Rows, R, float, false, QsaRowsScoreViews<8U>>(s.score, s.keys, s.blocks, dim, nullptr);
    });
}
__global__ __launch_bounds__(kSelectThreads) void qsa2_select_kernel(Qsa2SelectArgs args) {
    const auto& r = args.row[blockIdx.x];
    const auto complete = r.visible / r.ratio;
    if (complete <= r.topk) {
        for (std::uint32_t i = threadIdx.x; i < r.visible; i += blockDim.x) r.selected[i] = i;
    } else select_blocks_parallel_body(r.scores, complete, r.topk, r.ratio, r.visible,
        r.selected, GraphSelectionArguments{}, false);
}
template <unsigned PerWave, bool Int8>
__global__ __launch_bounds__(256) void qsa2_score_kernel(Qsa2AttentionArgs args, unsigned heads, unsigned kv_heads, unsigned dim) {
    const auto& s = args.session[blockIdx.y];
    if (blockIdx.x >= s.grid) return;
    qsa2_dispatch(s.rows, [&](auto width) {
        constexpr auto Rows = decltype(width)::value;
        qsa2_attention_wave_body<Rows, PerWave, Int8>(qsa2_attention_views<Rows>(args, s), s.kv,
            s.maximum, heads, kv_heads, dim, s.slots, uint3{blockIdx.x, 0U, 0U});
    });
}
template <unsigned G, unsigned Slots, bool Scatter, bool Prefetch, bool Int8>
__global__ __launch_bounds__(256) void qsa2_score_group_kernel(Qsa2AttentionArgs args, unsigned heads, unsigned kv_heads, unsigned dim) {
    const auto& s = args.session[blockIdx.y];
    if (blockIdx.x >= s.grid) return;
    qsa2_dispatch(s.rows, [&](auto width) {
        constexpr auto Rows = decltype(width)::value;
        if constexpr (Prefetch) qsa2_attention_group256_body<Rows, G, Slots, Int8>(qsa2_attention_views<Rows>(args, s), s.kv,
            s.maximum, heads, kv_heads, dim, s.slots, uint3{blockIdx.x, 0U, 0U});
        else qsa2_attention_group_body<Rows, G, Slots, Scatter, Int8>(qsa2_attention_views<Rows>(args, s), s.kv,
            s.maximum, heads, kv_heads, dim, s.slots, uint3{blockIdx.x, 0U, 0U});
    });
}
template <bool Int8>
__global__ void qsa2_score_block_kernel(Qsa2AttentionArgs args, unsigned heads, unsigned kv_heads, unsigned dim) {
    const auto r = blockIdx.y;
    const auto& s = args.session[args.owner[r]];
    if (blockIdx.x >= s.grid) return;
    attention_score_body<Int8>(args.views.q_gate[r], s.kv, args.views.selected[r], args.views.count[r],
        s.maximum, heads, kv_heads, dim, args.views.scores[r], GraphSelectionArguments{});
}
__global__ void qsa2_softmax_kernel(Qsa2AttentionArgs args) {
    const auto r = blockIdx.y;
    attention_softmax_body(args.views.scores[r], args.views.count[r], args.session[args.owner[r]].maximum, GraphSelectionArguments{});
}
template <bool Int8, bool Dense, bool Tail>
__global__ void qsa2_output_kernel(Qsa2AttentionArgs args, unsigned heads, unsigned kv_heads,
    unsigned dim, std::int8_t* codes, float* deltas) {
    const auto r = static_cast<unsigned>(blockIdx.y);
    const auto& s = args.session[args.owner[r]];
    const RoundedStageImage image{args.views.output[r], s.compact};
    if constexpr (Tail) attention_tail_body<Int8, Dense>(args.views.q_gate[r], s.kv, args.views.selected[r], args.views.scores[r],
        args.views.count[r], s.maximum, heads, kv_heads, dim, image, GraphSelectionArguments{}, codes, deltas, r);
    else {
        attention_output_body<Int8>(args.views.q_gate[r], s.kv, args.views.selected[r], args.views.scores[r], args.views.count[r],
            s.maximum, heads, kv_heads, dim, image, GraphSelectionArguments{});
        if constexpr (Dense) {
            const auto i = static_cast<std::uint64_t>(blockIdx.x) * dim + threadIdx.x;
            q8_rows::dense_a8_quantize_lane(image[i], codes + static_cast<std::uint64_t>(r) * heads * dim + i,
                deltas + (i / 32U) * q8_rows::kDenseA8Rows + r);
        }
    }
}

void enqueue_qsa_cohort_part2(const QsaCohortPrepared& prepared, std::span<const HipQsaCohortSegment> segments,
    const std::vector<QsaPrefillPool*>& pools, const q8_rows::DenseA8Slot& slot) {
    const auto& config = *segments[0].config;
    const auto& model = config.model;
    const auto& a = std::visit([](const auto& p) -> const PreparedIndexedLaunch& { return p.row[0]; }, prepared.segments[0]);
    const auto& tensors = a.tensors;
    const auto stream = hip_execution_stream();
    const auto count = static_cast<unsigned>(prepared.count), sessions = static_cast<unsigned>(segments.size());
    const auto rotary = checked_u32(model.rotary_dim(), "rotary_dim");
    const auto index_heads = checked_u32(model.indexer_n_heads, "indexer_n_heads");
    const bool int8 = model.attention_kv_storage == AttentionKvStorage::int8_token;
    Qsa2StoreArgs store{}; Qsa2IndexArgs index{}; Qsa2SelectArgs select{}; Qsa2AttentionArgs attention{};
    unsigned build_blocks = 0U, raw_blocks = 0U, pool_blocks = 0U;
    static const unsigned pooled_rows = [] {
        const char* v = std::getenv("STRIX_QSA_POOLED_ROWS"); const int n = v ? std::atoi(v) : 4;
        return n == 1 || n == 2 || n == 8 ? static_cast<unsigned>(n) : 4U;
    }();
    for (unsigned s = 0; s < sessions; ++s) std::visit([&](const auto& p) {
        constexpr auto Rows = static_cast<unsigned>(std::tuple_size_v<decltype(p.row)>);
        const auto first = prepared.first[s];
        auto& idx = index.session[s]; idx.rows = Rows; idx.ratio = p.selection[0].ratio;
        idx.raw = reinterpret_cast<const std::uint16_t*>(p.row[0].raw_state.address);
        auto& att = attention.session[s]; att.first = first; att.rows = Rows;
        att.kv = reinterpret_cast<const std::uint16_t*>(p.row[0].kv_state.address);
        att.maximum = checked_u32(p.row[0].workspace_layout.maximum_selected_tokens, "maximum_selected_tokens");
        for (unsigned r = 0; r < Rows; ++r) {
            const auto i = first + r; const auto& w = p.row[r].workspace; const auto& sel = p.selection[r];
            require_qsa_score_source(!p.row[r].workspace_layout.block_scores_bytes, w.block_scores != nullptr);
            store.row[i] = {w.current_index_key, w.current_k, w.current_v,
                reinterpret_cast<std::uint16_t*>(p.row[r].raw_state.address), reinterpret_cast<std::uint16_t*>(p.row[r].kv_state.address),
                p.journal ? p.journal + static_cast<std::uint64_t>(r) * a.index_dim : nullptr, p.position[r]};
            idx.score.index_query[r] = w.index_query; idx.score.block_scores[r] = w.block_scores;
            idx.score.blocks[r] = sel.complete_blocks > sel.block_topk ? sel.complete_blocks : 0U;
            idx.blocks = std::max(idx.blocks, idx.score.blocks[r]);
            select.row[i] = {w.block_scores, w.selected_indices, sel.visible, sel.ratio, sel.block_topk};
            attention.owner[i] = s; attention.views.q_gate[i] = w.q_gate; attention.views.selected[i] = w.selected_indices;
            attention.views.scores[i] = w.attention_scores; attention.views.output[i] = w.attention_output; attention.views.count[i] = sel.selected_count;
            att.slots = std::max(att.slots, sel.selected_count); att.compact = w.compact_attention_output;
        }
        auto* pool = pools[s]; bind_qwen_pool(pool, model);
        idx.pooled = pool && (pool->keys || pool->packed) && idx.blocks <= pool->capacity && a.index_dim == 128U && index_heads == 4U;
        if (idx.pooled) {
            idx.keys = pool->keys; idx.packed = pool->packed;
            idx.first = std::min(pool->valid, idx.blocks);
            build_blocks = std::max(build_blocks, idx.blocks - idx.first);
            const auto reuse = std::min(pooled_rows, Rows);
            idx.grid = ((Rows / reuse) * ((idx.blocks + 31U) / 32U) + 7U) / 8U;
            pool_blocks = std::max(pool_blocks, idx.grid);
        } else raw_blocks = std::max(raw_blocks, idx.blocks);
    }, prepared.segments[s]);
    // Norms preserve the per-session skip-index decision, including unused workspace bytes.
    const auto norm = [&](const TensorPair& weights, float* WorkspacePointers::* target, unsigned heads, unsigned dim, unsigned stride, bool is_index) {
        Qsa2NormArgs args{};
        for (unsigned s = 0; s < sessions; ++s) std::visit([&](const auto& p) {
            bool skip = is_index && qsa_skip_index_q_enabled();
            for (unsigned r = 0; skip && r < p.row.size(); ++r) skip = p.selection[r].complete_blocks <= p.selection[r].block_topk;
            for (unsigned r = 0; r < p.row.size(); ++r) {
                const auto i = prepared.first[s] + r;
                args.values[i] = p.row[r].workspace.*target; args.position[i] = p.position[r]; args.skip[i] = skip;
            }
        }, prepared.segments[s]);
        dispatch_small_kind(weights.manifest->ggml_type, [&](auto kind) {
            with_qwen_rotary(model, stream, [&](auto... source) {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa2_norm_kernel<decltype(kind)::value, decltype(source)...>), dim3(heads, count), dim3(dim),
                    2ULL * dim * sizeof(float), stream, args, heads, dim, stride, reinterpret_cast<const void*>(weights.runtime->address),
                    model.rms_norm_epsilon, rotary, model.rope_theta, source...);
            });
        });
        check_hip(hipGetLastError(), "QSA2 cohort norm/rope launch");
    };
    norm(tensors.index_q_norm, &WorkspacePointers::index_query, index_heads, a.index_dim, a.index_dim, true);
    norm(tensors.q_norm, &WorkspacePointers::q_gate, a.attention_heads, a.head_dim, a.head_dim * 2U, false);
    norm(tensors.k_norm, &WorkspacePointers::current_k, a.kv_heads, a.head_dim, a.head_dim, false);
    const auto bf16_width = int8 ? 0U : a.kv_width;
    hipLaunchKernelGGL(qsa2_store_kernel, dim3((std::max(bf16_width, a.index_dim) + kThreads - 1U) / kThreads, count),
        dim3(kThreads), 0, stream, store, a.index_dim, bf16_width);
    check_hip(hipGetLastError(), "QSA2 cohort raw-index/KV append");
    if (int8) {
        hipLaunchKernelGGL(qsa2_store_int8_kernel, dim3(a.kv_heads, 2U, count), dim3(kThreads), 0, stream, store, a.kv_width, a.kv_heads, a.head_dim);
        check_hip(hipGetLastError(), "QSA2 cohort int8 KV append");
    }
    dispatch_small_kind(tensors.index_k_norm.manifest->ggml_type, [&](auto kind) {
        with_qwen_rotary(model, stream, [&](auto... source) {
            if (build_blocks) hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa2_pool_kernel<decltype(kind)::value, decltype(source)...>),
                dim3(build_blocks, sessions), dim3(a.index_dim), 2ULL * a.index_dim * sizeof(float), stream,
                index, a.index_dim, reinterpret_cast<const void*>(tensors.index_k_norm.runtime->address), model.rms_norm_epsilon,
                rotary, model.rope_theta, source...);
            if (raw_blocks) hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa2_index_raw_kernel<decltype(kind)::value, decltype(source)...>),
                dim3(raw_blocks, sessions), dim3(a.index_dim), 2ULL * a.index_dim * sizeof(float), stream,
                index, index_heads, a.index_dim, reinterpret_cast<const void*>(tensors.index_k_norm.runtime->address),
                model.rms_norm_epsilon, rotary, model.rope_theta, source...);
        });
    });
    check_hip(hipGetLastError(), "QSA2 cohort index build/score launch");
    if (pool_blocks) {
        const auto launch = [&]<unsigned R>() {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa2_index_pooled_kernel<R>), dim3(pool_blocks, sessions), dim3(kThreads), 0, stream, index, a.index_dim);
        };
        if (pooled_rows == 1U) launch.template operator()<1U>(); else if (pooled_rows == 2U) launch.template operator()<2U>();
        else if (pooled_rows == 8U) launch.template operator()<8U>(); else launch.template operator()<4U>();
        check_hip(hipGetLastError(), "QSA2 cohort pooled score launch");
    }
    for (unsigned s = 0; s < sessions; ++s) if (index.session[s].pooled) pools[s]->valid = std::max(pools[s]->valid, index.session[s].blocks);
    hipLaunchKernelGGL(qsa2_select_kernel, dim3(count), dim3(kSelectThreads), 0, stream, select);
    check_hip(hipGetLastError(), "QSA2 cohort selection launch");
    const auto env_choice = [](const char* name, int fallback) {
        const char* v = std::getenv(name); const int n = v ? std::atoi(v) : fallback;
        return n == 0 || n == 2 || n == 3 || n == 4 || n == 6 || n == 12 ? static_cast<unsigned>(n) : static_cast<unsigned>(fallback);
    };
    static const unsigned head_group = env_choice("STRIX_QSA_PREFILL_GROUP", 12);
    static const unsigned window_slots = [] { const char* v = std::getenv("STRIX_QSA_WINDOW_SLOTS"); const int n = v ? std::atoi(v) : 8;
        return n == 4 || n == 16 ? static_cast<unsigned>(n) : 8U; }();
    const auto per_kv = a.kv_heads && a.attention_heads % a.kv_heads == 0U ? a.attention_heads / a.kv_heads : 0U;
    const bool grouped = qsa_window_grouped_requested() && head_group && a.head_dim <= kThreads && per_kv && per_kv % head_group == 0U;
    const auto per_wave = qsa_score_slots_per_wave();
    unsigned score_blocks = 0U;
    for (unsigned s = 0; s < sessions; ++s) {
        auto& att = attention.session[s];
        static_cast<void>(prepare_indexed_attention_score_grid(a.attention_heads, att.slots));
        if (a.head_dim > kThreads) att.grid = static_cast<unsigned>(prepare_indexed_attention_score_grid(a.attention_heads, att.slots));
        else {
            const auto slots = grouped ? window_slots : per_wave, heads = grouped ? a.attention_heads / head_group : a.attention_heads;
            att.grid = (att.rows * heads * ((att.slots + slots - 1U) / slots) + 7U) / 8U;
        }
        score_blocks = std::max(score_blocks, att.grid);
    }
    const bool dense = config.dense_a8_projections && tensors.out.manifest->ggml_type == 8U;
    if (dense && (a.head_dim % 32U || a.head_dim > 1024U || a.attention_width != a.attention_heads * a.head_dim))
        throw std::invalid_argument("QSA rows dense A8 value/gate quantizer requires whole 32-feature head blocks");
    const auto attend = [&]<bool Int8>() {
        if (a.head_dim > kThreads) hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa2_score_block_kernel<Int8>), dim3(score_blocks, count),
            dim3(kThreads), 0, stream, attention, a.attention_heads, a.kv_heads, a.head_dim);
        else if (grouped) {
            static const bool scatter = [] { const char* v = std::getenv("STRIX_QSA_PREFILL_SCATTER"); return !v || v[0] != '0'; }();
            static const bool prefetch = [] { const char* v = std::getenv("STRIX_QSA_SCORE_PREFETCH"); return !v || v[0] != '0'; }();
            const auto launch = [&]<unsigned G, unsigned S>() {
                const auto run = [&]<bool Scatter, bool Prefetch>() {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa2_score_group_kernel<G, S, Scatter, Prefetch, Int8>), dim3(score_blocks, sessions),
                        dim3(kThreads), 0, stream, attention, a.attention_heads, a.kv_heads, a.head_dim);
                };
                if (scatter && prefetch && a.head_dim == 256U) run.template operator()<true, true>();
                else if (scatter) run.template operator()<true, false>(); else run.template operator()<false, false>();
            };
            const auto by_group = [&]<unsigned G>() {
                if (window_slots == 4U) launch.template operator()<G, 4U>(); else if (window_slots == 16U) launch.template operator()<G, 16U>();
                else launch.template operator()<G, 8U>();
            };
            switch (head_group) {
            case 2U: by_group.template operator()<2U>(); break; case 3U: by_group.template operator()<3U>(); break;
            case 4U: by_group.template operator()<4U>(); break; case 12U: by_group.template operator()<12U>(); break;
            default: by_group.template operator()<6U>(); break;
            }
        } else {
            const auto launch = [&]<unsigned P>() {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa2_score_kernel<P, Int8>), dim3(score_blocks, sessions), dim3(kThreads), 0, stream,
                    attention, a.attention_heads, a.kv_heads, a.head_dim);
            };
            if (per_wave == 2U) launch.template operator()<2U>(); else if (per_wave == 4U) launch.template operator()<4U>();
            else if (per_wave == 8U) launch.template operator()<8U>(); else launch.template operator()<1U>();
        }
        check_hip(hipGetLastError(), "QSA2 cohort attention score launch");
        const bool tail = qpersist_qsa_tail_enabled() && (!Int8 || qsa_tail_int8_enabled()) && qsa_cohort_tail_enabled() && a.head_dim <= kThreads;
        if (tail) count_qsa_launch(Int8 ? &QsaLaunchStats::tail_int8 : &QsaLaunchStats::tail_bf16);
        else {
            hipLaunchKernelGGL(qsa2_softmax_kernel, dim3(a.attention_heads, count), dim3(kThreads), 0, stream, attention);
            check_hip(hipGetLastError(), "QSA2 cohort softmax launch");
        }
        const auto output = [&]<bool Dense, bool Tail>() {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa2_output_kernel<Int8, Dense, Tail>), dim3(a.attention_heads, count),
                dim3(Tail ? kThreads : a.head_dim), 0, stream, attention, a.attention_heads, a.kv_heads, a.head_dim,
                Dense ? slot.codes : nullptr, Dense ? slot.deltas : nullptr);
        };
        if (tail && dense) output.template operator()<true, true>(); else if (tail) output.template operator()<false, true>();
        else if (dense) output.template operator()<true, false>(); else output.template operator()<false, false>();
        check_hip(hipGetLastError(), "QSA2 cohort value/gate launch");
    };
    if (int8) attend.template operator()<true>(); else attend.template operator()<false>();
}
