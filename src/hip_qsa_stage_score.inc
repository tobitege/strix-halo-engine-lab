// Shared by the real stage batch/split and the standalone device byte fixture.
// All grid geometry and template choices come from the original call maximum.
void qpf2k_stage_score_launch(const qsa_flash::ScoreGroup<64U>* descriptors,
    const void* keys, bool packed, const uint2* ranges, std::uint32_t index_dim,
    std::uint32_t max_blocks, std::uint32_t first, std::uint32_t count, hipStream_t stream, bool allow_band = true) {
    if (allow_band && qsa_flash::index_band()) {
        qsa_flash::launch_index_band<64U, true>(QsaRowsScoreViews<64U>{}, keys, packed, max_blocks,
            index_dim, nullptr, 0U, descriptors + first, count, stream);
        return;
    }
    constexpr unsigned W = qsa_flash::kScoreWaves;
    const auto score = [&]<unsigned R, bool S>() {
        if (packed)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::score_kernel<64U, R, W, false, false, true, 1U, true, true, S>),
                dim3(((max_blocks + 31U) / 32U + W - 1U) / W, 64U / R, count), dim3(W * 32U), 0, stream,
                QsaRowsScoreViews<64U>{}, keys, 0U, index_dim, nullptr, 0U,
                ranges, nullptr, descriptors + first);
        else
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::score_kernel<64U, R, W, false, false, true, 1U, true, false, S>),
                dim3(((max_blocks + 31U) / 32U + W - 1U) / W, 64U / R, count), dim3(W * 32U), 0, stream,
                QsaRowsScoreViews<64U>{}, keys, 0U, index_dim, nullptr, 0U,
                ranges, nullptr, descriptors + first);
    };
    const auto rows = [&]<bool S>() {
        switch (qsa_flash::score_rows(max_blocks)) {
        case 32U: score.template operator()<32U, S>(); break;
        case 64U: score.template operator()<64U, S>(); break;
        default: score.template operator()<qsa_flash::kScoreRows, S>(); break;
        }
    };
    if (qsa_flash::score_split_loads()) rows.template operator()<true>();
    else rows.template operator()<false>();
}
