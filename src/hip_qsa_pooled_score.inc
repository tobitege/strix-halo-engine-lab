// Prefill groups (Rows > 8): the pooled/normalised/rotated key of a complete block depends only
// on its raw keys, so it is formed once per prefill call (block_pool_kernel: the statements of
// block_score_rows_kernel's pooled part) and every row of every group scores it
// (block_score_pooled_rows_kernel). Per (row, block, head) the score is the 128-feature LDS tree
// of block_score_rows_kernel done in registers: lane l holds features l + 32j, so strides 64
// and 32 are the in-lane pairs (j, j + 2) and then (0, 1), and 16..1 are the xor reduce-scatter
// of attention_score_rows_group_kernel (each step adds the partials of lanes L and L ^ D). The
// relu sum over the heads, in head order from 0, and the division are unchanged.
// One wave scores RowsPerWave consecutive rows against 32 consecutive blocks (lane i keeps
// block base + i); index head_dim 128, 4 index heads.
// Key changes only the load: BF16 upper halves widen exactly before the shared FP32 arithmetic.
// Maxima (one serial decode row, STRIX_QSA_DECODE_FUSED_MAXIMA): the wave also writes the 4-block group maxima of its 32
// blocks with row_maxima_kernel's expression fmaxf(fmaxf(s0, s1), fmaxf(s2, s3)) over the scores it stores (-inf past
// the row's blocks), so the selection reads them without the separate row_maxima_kernel launch.
// Views: any score views with at least Rows entries (the merged cohort kernels pass a session's 8-entry views in
// place for 2 or 4 rows; a narrowed local copy indexed by the wave's row lands in scratch memory).
template <unsigned Rows, unsigned RowsPerWave, class Key, bool Maxima, class Views = QsaRowsScoreViews<Rows>>
__device__ __forceinline__ void block_score_pooled_rows_body(const Views& views, const Key* pooled_keys,
    std::uint32_t scored_blocks, std::uint32_t head_dim, float* maxima) {
#pragma clang fp contract(off)
    constexpr unsigned Heads = 4U, Dim = 128U; // head_dim == Dim (checked by the caller)
    const auto lane = threadIdx.x % 32U;
    const auto wave = blockIdx.x * 8U + threadIdx.x / 32U;
    const auto chunks = (scored_blocks + 31U) / 32U;
    const auto r0 = wave / chunks * RowsPerWave, base = wave % chunks * 32U;
    if (r0 >= Rows) return;
    float q[RowsPerWave][Heads][4];
    std::uint32_t blocks[RowsPerWave];
#pragma unroll
    for (unsigned rr = 0; rr < RowsPerWave; ++rr) {
        blocks[rr] = views.blocks[r0 + rr];
#pragma unroll
        for (unsigned h = 0; h < Heads; ++h)
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) q[rr][h][j] = views.index_query[r0 + rr][h * Dim + lane + 32U * j];
    }
    float result[RowsPerWave];
#pragma unroll
    for (unsigned rr = 0; rr < RowsPerWave; ++rr) result[rr] = 0.0F;
    const auto last = min(32U, scored_blocks - base);
    for (unsigned i = 0; i < last; ++i) {
        const auto* key = pooled_keys + static_cast<std::uint64_t>(base + i) * Dim + lane;
        float k0, k1, k2, k3;
        if constexpr (std::is_same_v<Key, std::uint16_t>) {
            k0 = bf16_to_float(key[0]); k1 = bf16_to_float(key[32]);
            k2 = bf16_to_float(key[64]); k3 = bf16_to_float(key[96]);
        } else {
            k0 = key[0]; k1 = key[32]; k2 = key[64]; k3 = key[96];
        }
#pragma unroll
        for (unsigned rr = 0; rr < RowsPerWave; ++rr) {
            float partial[Heads];
#pragma unroll
            for (unsigned h = 0; h < Heads; ++h) {
                const float p0 = q[rr][h][0] * k0, p1 = q[rr][h][1] * k1, p2 = q[rr][h][2] * k2, p3 = q[rr][h][3] * k3;
                partial[h] = (p0 + p2) + (p1 + p3);
            }
            qsa_scatter<Heads, 0U>(partial, lane);
            // Lane 8h holds head h's sum.
            float total = 0.0F;
#pragma unroll
            for (unsigned h = 0; h < Heads; ++h) total += fmaxf(__shfl(partial[0], static_cast<int>(8U * h), 32), 0.0F);
            if (lane == i) result[rr] = total / sqrtf(static_cast<float>(head_dim));
        }
    }
#pragma unroll
    for (unsigned rr = 0; rr < RowsPerWave; ++rr)
        if (lane < last && base + lane < blocks[rr]) views.block_scores[r0 + rr][base + lane] = result[rr];
    if constexpr (Maxima) {
        static_assert(Rows == 1U && RowsPerWave == 1U);
        const float stored = lane < last && base + lane < blocks[0] ? result[0] : -INFINITY;
        const auto first = static_cast<int>(lane & ~3U);
        const float s0 = __shfl(stored, first, 32), s1 = __shfl(stored, first + 1, 32), s2 = __shfl(stored, first + 2, 32),
            s3 = __shfl(stored, first + 3, 32);
        if ((lane & 3U) == 0U && base + lane < blocks[0]) maxima[(base + lane) / 4U] = fmaxf(fmaxf(s0, s1), fmaxf(s2, s3));
    }
}
template <unsigned Rows, unsigned RowsPerWave, class Key = float>
__global__ __launch_bounds__(256) void block_score_pooled_rows_kernel(QsaRowsScoreViews<Rows> views, const Key* pooled_keys,
    std::uint32_t scored_blocks, std::uint32_t head_dim) {
    block_score_pooled_rows_body<Rows, RowsPerWave, Key, false>(views, pooled_keys, scored_blocks, head_dim, nullptr);
}
template <class Key = float>
__global__ __launch_bounds__(256) void block_score_pooled_decode_maxima_kernel(QsaRowsScoreViews<1U> views, const Key* pooled_keys,
    std::uint32_t scored_blocks, std::uint32_t head_dim, float* maxima) {
    block_score_pooled_rows_body<1U, 1U, Key, true>(views, pooled_keys, scored_blocks, head_dim, maxima);
}
