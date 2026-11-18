#include "hip_qsa_stage_score.inc"
// STRIX_QPF2K_QSA_MASK bit 9 (0x200): stage-major submission of one layer-chunk.
// Draft, default OFF. Bits 0/1 are required; bit 2 chooses the same wave/LDS norm
// as the per-group path. Admission is completed before any workspace is changed.
struct Qpf2kStageSlot {
    qsa_index::NormGroupViews<64U>* norms = nullptr;
    QsaRowsStoreViews<64U>* stores = nullptr;
    qsa_flash::ScoreGroup<64U>* scores = nullptr;
    uint2* bounds = nullptr;
    float2* tables = nullptr;
    float* maxima = nullptr;
    std::uint64_t maxima_capacity = 0;
    std::uint32_t group_capacity = 0;
    hipEvent_t done = nullptr;
    bool pending = false;
};
static_assert(sizeof(qsa_index::NormGroupViews<64U>) + sizeof(QsaRowsStoreViews<64U>) +
    sizeof(qsa_flash::ScoreGroup<64U>) + sizeof(uint2) <= 20480U,
    "YaRN admission descriptor bound must track stage slot payloads");
struct Qpf2kStagePool {
    std::mutex mutex;
    std::array<Qpf2kStageSlot, 2> ring;
    std::size_t next = 0;
    std::uint32_t rows = 0;
    std::uint64_t context_tokens = 0;
    std::uint32_t ratio = 0;
    int device = -1;
    bool compact = false;
};
Qpf2kStagePool& qpf2k_stage_pool() { static Qpf2kStagePool pool; return pool; }
void qpf2k_stage_setup(std::uint32_t rows, std::uint64_t context_tokens, std::uint32_t ratio, bool compact) {
    auto& pool = qpf2k_stage_pool();
    const std::lock_guard lock(pool.mutex);
    if (!rows || rows > known_prefill_max_rows || !context_tokens || !ratio)
        throw std::invalid_argument("QSA stage setup: invalid capacity");
    const auto capacity = known_prefill_pool_rows(rows);
    const auto device = hip_prefill_device();
    if (pool.rows && !PrefillStageReservation{pool.rows, pool.context_tokens, pool.ratio, pool.device, pool.compact}.fits(
        {capacity, context_tokens, ratio, device, compact}))
        return; // Optional ring mismatch: the per-group path remains available.
    if (pool.rows) return;
    const auto commit = [&] {
        pool.rows = capacity; pool.context_tokens = context_tokens; pool.ratio = ratio; pool.device = device; pool.compact = compact;
    };
    if ((qpf2k_qsa_mask() & 0x203U) != 0x203U || !qsa_norm_group_enabled()) { commit(); return; }
    const auto groups = known_prefill_stage_groups(capacity);
    const auto maxima = static_cast<std::uint64_t>(capacity) * ((context_tokens / ratio + qsa_flash::kGroup - 1U) / qsa_flash::kGroup);
    std::array<Qpf2kStageSlot, 2> reserved{};
    try {
        for (auto& slot : reserved) {
            const auto host_alloc = [&]<class T>(T*& p) {
                check_hip(hipHostMalloc(reinterpret_cast<void**>(&p), groups * sizeof(T), hipHostMallocDefault),
                    "QSA stage setup descriptors");
            };
            host_alloc(slot.norms); host_alloc(slot.stores); host_alloc(slot.scores); host_alloc(slot.bounds);
            check_hip(hipMalloc(reinterpret_cast<void**>(&slot.tables),
                static_cast<std::uint64_t>(groups) * 64U * 64U * sizeof(float2)), "QSA stage setup tables");
            // Even an empty scored prefix gets a valid private allocation. No execution-time replacement.
            if (!compact) check_hip(hipMalloc(reinterpret_cast<void**>(&slot.maxima), std::max(UINT64_C(1), maxima) * sizeof(float)),
                "QSA stage setup maxima");
            check_hip(hipEventCreateWithFlags(&slot.done, hipEventDisableTiming), "QSA stage setup event");
            slot.group_capacity = std::max(slot.group_capacity, groups);
            slot.maxima_capacity = compact ? 0U : maxima;
        }
    } catch (...) {
        // Setup has submitted no readers. Release only these unexposed partial reservations.
        for (auto& slot : reserved) {
            if (slot.norms) (void)hipHostFree(slot.norms);
            if (slot.stores) (void)hipHostFree(slot.stores);
            if (slot.scores) (void)hipHostFree(slot.scores);
            if (slot.bounds) (void)hipHostFree(slot.bounds);
            if (slot.tables) (void)hipFree(slot.tables);
            if (slot.maxima) (void)hipFree(slot.maxima);
            if (slot.done) (void)hipEventDestroy(slot.done);
        }
        return; // Allocation pressure declines batching before any reader is submitted.
    }
    pool.ring = reserved;
    commit();
}
void release_qpf2k_stage_slots() {
    auto& pool = qpf2k_stage_pool();
    const std::lock_guard lock(pool.mutex);
    for(auto& slot:pool.ring) {
        if(slot.pending) check_hip(hipEventSynchronize(slot.done),"QSA stage retirement consumer wait");
        for(auto* p:{static_cast<void*>(slot.norms),static_cast<void*>(slot.stores),
                    static_cast<void*>(slot.scores),static_cast<void*>(slot.bounds)})
            if(p) check_hip(hipHostFree(p),"QSA stage retirement descriptor free");
        if(slot.tables) check_hip(hipFree(slot.tables),"QSA stage retirement table free");
        if(slot.maxima) check_hip(hipFree(slot.maxima),"QSA stage retirement maxima free");
        if(slot.done) check_hip(hipEventDestroy(slot.done),"QSA stage retirement event free");
        slot={};
    }
    pool.next = 0; pool.rows = 0; pool.context_tokens = 0; pool.ratio = 0; pool.device = -1; pool.compact = false;
}
struct Qpf2kStageBatch {
    std::unique_lock<std::mutex> submission_lock;
    Qpf2kStageSlot* slot = nullptr;
    std::uint32_t max_blocks = 0, groups = 0;
    Qpf2kStageBatch() = default;
    Qpf2kStageBatch(const Qpf2kStageBatch&) = delete;
    Qpf2kStageBatch& operator=(const Qpf2kStageBatch&) = delete;
    ~Qpf2kStageBatch() {
        // Includes deferred selection's final read of private maxima. Also protect
        // queued work on an exceptional exit before the next use of this ring slot.
        if (slot) {
            if (hipEventRecord(slot->done, hip_execution_stream()) != hipSuccess) std::terminate();
            slot->pending = true;
        }
    }
};
qsa_index::NormGroupViews<64U> prepare_qpf2k_norm_rows(
    const std::vector<QsaKnownRows<8U>>& tiles, std::uint32_t group) {
    qsa_index::NormGroupViews<64U> n{};
    for (unsigned r=0;r<64U;++r) {
        const auto& tile=tiles.at(static_cast<std::size_t>(group)*8U+r/8U);
        const auto& w=tile.row[r%8U].workspace;
        n.position[r]=tile.position[r%8U];
        n.values[0][r]=w.index_query; n.values[1][r]=w.q_gate; n.values[2][r]=w.current_k;
    }
    return n;
}
// Only descriptor addresses and grid.z differ between embedded and compact calls.
// Dispatch and grid.x always use the original call's maximum, never a slice maximum.
void qpf2k_stage_score_slice(const Qpf2kStageBatch& batch, QsaPrefillPool* pool,
    std::uint32_t index_dim, std::uint32_t first, std::uint32_t count) {
    if (!batch.slot || !count || first >= batch.groups || count > batch.groups - first)
        throw std::invalid_argument("QSA stage score slice outside prepared descriptors");
    auto& slot = *batch.slot;
    const auto max_blocks = batch.max_blocks;
    const auto stream = hip_execution_stream();
    if (max_blocks) {
        if (qsa_flash::score_rows(max_blocks) != 16U) {
            static const bool witness = [] { std::fprintf(stderr, "qpf2k qsa score rows > 16 active\n"); return true; }();
            (void)witness;
        }
        if (qsa_flash::index_band())
            qpf2k_band_reduce(pool->capacity, pool->packed ? static_cast<const void*>(pool->packed) : pool->keys,
                pool->packed != nullptr, max_blocks, stream);
        if (!qsa_flash::index_band() && qpf2k_index_score_wmma_active) {
            qsa_flash::launch_index_score_wmma<64U, true>(QsaRowsScoreViews<64U>{},
                pool->packed ? static_cast<const void*>(pool->packed) : pool->keys, pool->packed != nullptr,
                max_blocks, nullptr, 0U, slot.scores + first, count, stream);
            qpf2k_index_score_wmma_witness();
        } else qpf2k_stage_score_launch(slot.scores, pool->packed ? static_cast<const void*>(pool->packed) : pool->keys,
            pool->packed != nullptr, static_cast<const uint2*>(qpf2k_key_ranges(pool->capacity)), index_dim,
            max_blocks, first, count, stream);
        check_hip(hipGetLastError(), "QSA stage batch score launch");
        qpf2k_launch_score.fetch_add(1U, std::memory_order_relaxed);
    }
}
bool qpf2k_stage_batch_launch(const std::vector<QsaKnownRows<8U>>& tiles,
    const HipIndexedAttentionLauncherConfig& config, QsaPrefillPool* pool, Qpf2kStageBatch& batch, const QsaScoreRingGeometry* compact = nullptr) {
    constexpr std::size_t MaxGroups = known_prefill_max_rows / 64U;
    auto& state = qpf2k_stage_pool();
    batch.submission_lock = std::unique_lock(state.mutex);
    auto& ring = state.ring;
    auto& next = state.next;
    const auto decline = [] { return false; };
    if ((qpf2k_qsa_mask() & 0x203U) != 0x203U || !qsa_norm_group_enabled() ||
        tiles.empty() || tiles.size() % 8U || tiles.size() / 8U > MaxGroups || tiles.size() / 8U > UINT32_MAX ||
        config.model.attention_kv_storage != AttentionKvStorage::bf16 || !pool || !pool->keys)
        return decline();
    if (!PrefillStageReservation{state.rows, state.context_tokens, state.ratio, state.device, state.compact}.fits(
        {static_cast<std::uint32_t>(tiles.size() * 8U), state.context_tokens,
            static_cast<std::uint32_t>(config.model.indexer_compress_ratio), hip_prefill_device(), compact != nullptr}))
        return decline();
    const auto& model = config.model;
    bind_qwen_pool(pool, model);
    const auto& a = tiles[0].row[0];
    const auto& tensors = a.tensors;
    const auto rotary = checked_u32(model.rotary_dim(), "rotary_dim");
    if (a.index_dim != 128U || model.indexer_n_heads != 4U || rotary / 2U > 64U ||
        rotary > a.index_dim || rotary > a.head_dim)
        return decline();
    const std::array<const TensorPair*, 3> norm_tensors{&tensors.index_q_norm, &tensors.q_norm, &tensors.k_norm};
    const std::array<std::uint32_t, 3> dims{a.index_dim, a.head_dim, a.head_dim},
        strides{a.index_dim, 2U * a.head_dim, a.head_dim};
    const auto type = norm_tensors[0]->manifest->ggml_type;
    bool wave_norm = (qpf2k_qsa_mask() & 4U) && (rotary / 2U) % 32U == 0U;
    for (unsigned s = 0; s < 3U; ++s) {
        if (norm_tensors[s]->manifest->ggml_type != type || dims[s] < 2U || dims[s] > 256U ||
            (dims[s] & (dims[s] - 1U))) return decline();
        wave_norm = wave_norm && dims[s] >= 32U;
    }
    const auto groups = static_cast<std::uint32_t>(tiles.size() / 8U);
    std::vector<qsa_index::NormGroupViews<64U>> norms;
    std::vector<QsaRowsStoreViews<64U>> stores;
    std::vector<qsa_flash::ScoreGroup<64U>> scores;
    std::vector<uint2> bounds;
    try {
        norms.resize(groups); stores.resize(groups); scores.resize(groups); bounds.resize(groups);
    } catch (const std::bad_alloc&) {
        return decline();
    }
    std::uint32_t valid = pool->valid, max_new = 0, max_blocks = 0, max_heads = 0;
    std::uint64_t maxima_count = 0;
    const auto ratio = tiles[0].selection[0].ratio, topk = tiles[0].selection[0].block_topk;
    if (!ratio) return decline();
    if (ratio != state.ratio)
        throw std::runtime_error("QSA stage batch differs from the configured compression ratio");
    for (std::uint32_t g = 0; g < groups; ++g) {
        auto& n = norms[g]; auto& st = stores[g]; auto& sc = scores[g];
        n = prepare_qpf2k_norm_rows(tiles,g);
        bool skip_query = qsa_skip_index_q_enabled();
        for (unsigned r = 0; r < 64U; ++r) {
            const auto& tile = tiles[static_cast<std::size_t>(g) * 8U + r / 8U];
            const auto& row = tile.row[r % 8U];
            const auto& sel = tile.selection[r % 8U];
            const auto position = tile.position[r % 8U];
            // The ordinary admission checks one tile at a time. Moving all stores
            // ahead of all pools additionally requires one contiguous state owner.
            if (position != tiles[0].position[0] + static_cast<std::uint64_t>(g) * 64U + r ||
                row.raw_state.address != a.raw_state.address || row.kv_state.address != a.kv_state.address ||
                row.raw_state.bytes != a.raw_state.bytes || row.kv_state.bytes != a.kv_state.bytes ||
                row.index_dim != a.index_dim || row.head_dim != a.head_dim || row.kv_width != a.kv_width ||
                row.attention_heads != a.attention_heads || row.kv_heads != a.kv_heads ||
                tile.journal != tiles[0].journal || sel.ratio != ratio || sel.block_topk != topk ||
                reinterpret_cast<std::uintptr_t>(row.workspace.index_query) % 16U)
                return decline();
            for (const auto member : {&IndexedTensors::index_q_norm, &IndexedTensors::q_norm,
                &IndexedTensors::k_norm, &IndexedTensors::index_k_norm})
                if ((row.tensors.*member).runtime != (tensors.*member).runtime ||
                    (row.tensors.*member).manifest != (tensors.*member).manifest) return decline();
            skip_query = skip_query && sel.complete_blocks <= sel.block_topk;
            st.position[r] = position;
            st.index_key[r] = row.workspace.current_index_key;
            st.k[r] = row.workspace.current_k; st.v[r] = row.workspace.current_v;
            sc.views.index_query[r] = row.workspace.index_query;
            sc.views.block_scores[r] = row.workspace.block_scores;
            sc.views.blocks[r] = sel.complete_blocks > sel.block_topk ? sel.complete_blocks : 0U;
            sc.blocks = std::max(sc.blocks, sc.views.blocks[r]);
        }
        for (unsigned s = 0; s < 3U; ++s) {
            n.weights[s] = reinterpret_cast<const void*>(norm_tensors[s]->runtime->address);
            n.dim[s] = dims[s]; n.stride[s] = strides[s];
        }
        n.heads[0] = skip_query ? 0U : 4U; n.heads[1] = a.attention_heads; n.heads[2] = a.kv_heads;
        max_heads = std::max(max_heads, n.heads[0] + n.heads[1] + n.heads[2]);
        // Earlier serial journal writes are overwritten before any consumer. Never
        // let parallel groups race to publish those same eight rows.
        st.journal = g + 1U == groups ? tiles.back().journal : nullptr;
        if (sc.blocks > pool->capacity ||
            (!compact && sc.blocks > qpf2k_maxima_stride * qsa_flash::kGroup) ||
            (compact && sc.blocks > compact->blocks)) return decline();
        const auto count = sc.blocks > valid ? sc.blocks - valid : 0U;
        bounds[g] = make_uint2(valid, count);
        valid = std::max(valid, sc.blocks);
        max_new = std::max(max_new, count); max_blocks = std::max(max_blocks, sc.blocks);
        sc.maxima_stride = (sc.blocks + qsa_flash::kGroup - 1U) / qsa_flash::kGroup;
        maxima_count += static_cast<std::uint64_t>(64U) * sc.maxima_stride;
    }
    // Two pinned descriptor slots, with private tables and embedded-only maxima.
    // Protect descriptors through the last score/selection consumer. Compact
    // score/maxima storage has its own per-slice owner lease below the caller.
    auto& slot = ring[next];
    const auto stream = hip_execution_stream();
    if (slot.pending) check_hip(hipEventSynchronize(slot.done), "QSA stage batch slot wait");
    slot.pending = false;
    if (!state.rows || hip_prefill_device() != state.device || groups * 64U > state.rows ||
        !slot.done || groups > slot.group_capacity || (!compact && maxima_count > slot.maxima_capacity))
        return decline();
    // The range and score launch arguments below must only reuse this reservation.
    if ((max_new || max_blocks) && !qpf2k_key_ranges(pool->capacity, true)) return decline();
    if (qsa_flash::index_band() && (max_new || max_blocks) && !qpf2k_key_norms(pool->capacity, true)) return decline();
    std::uint64_t offset = 0;
    for (std::uint32_t g = 0; g < groups; ++g) {
        norms[g].table = slot.tables + static_cast<std::uint64_t>(g) * 64U * 64U;
        scores[g].maxima = scores[g].blocks && !compact ? slot.maxima + offset : nullptr;
        offset += static_cast<std::uint64_t>(64U) * scores[g].maxima_stride;
        slot.norms[g] = norms[g]; slot.stores[g] = stores[g]; slot.scores[g] = scores[g]; slot.bounds[g] = bounds[g];
    }
    batch.max_blocks = max_blocks; batch.groups = groups;
    batch.slot = &slot; // arm lifetime protection before submitting any descriptor reader
    next = (next + 1U) % ring.size();
    with_qwen_rotary(model, stream, [&](auto... rotary_source) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::rope_table_kernel<64U, true, decltype(rotary_source)...>), dim3(64U, groups), dim3(64), 0, stream,
            qsa_index::NormGroupViews<64U>{}, rotary, model.rope_theta, slot.norms, rotary_source...);
    });
    dispatch_small_kind(type, [&](auto kind) {
        if (wave_norm)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::norm_rope_wave_kernel<decltype(kind)::value, 64U, true>),
                dim3((64U * max_heads + 7U) / 8U, groups), dim3(256), 0, stream,
                qsa_index::NormGroupViews<64U>{}, model.rms_norm_epsilon, rotary, slot.norms);
        else
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::norm_rope_group_kernel<decltype(kind)::value, 64U, true>),
                dim3(max_heads, 64U, groups), dim3(256), 0, stream,
                qsa_index::NormGroupViews<64U>{}, model.rms_norm_epsilon, rotary, slot.norms);
    });
    check_hip(hipGetLastError(), "QSA stage batch norm/RoPE launch");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(store_rows_kernel<64U, true>),
        dim3((std::max(a.kv_width, a.index_dim) + kThreads - 1U) / kThreads, 64U, groups), dim3(kThreads), 0, stream,
        QsaRowsStoreViews<64U>{}, reinterpret_cast<std::uint16_t*>(a.raw_state.address),
        reinterpret_cast<std::uint16_t*>(a.kv_state.address), a.index_dim, a.kv_width, slot.stores);
    check_hip(hipGetLastError(), "QSA stage batch store launch");
    if (max_new) {
        const auto total_new = valid - pool->valid;
        dispatch_small_kind(tensors.index_k_norm.manifest->ggml_type, [&](auto kind) {
            with_qwen_rotary(model, stream, [&](auto... rotary_source) {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(block_pool_kernel<decltype(kind)::value, true, false, decltype(rotary_source)...>), dim3(total_new),
                    dim3(a.index_dim), 2ULL * a.index_dim * sizeof(float), stream,
                    reinterpret_cast<const std::uint16_t*>(a.raw_state.address), pool->valid, a.index_dim, ratio,
                    reinterpret_cast<const void*>(tensors.index_k_norm.runtime->address), model.rms_norm_epsilon, rotary,
                    model.rope_theta, pool->keys, slot.bounds, groups, pool->packed, rotary_source...);
            });
        });
        check_hip(hipGetLastError(), "QSA stage batch pool launch");
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::key_range_kernel<true>), dim3((total_new + 255U) / 256U),
            dim3(256), 0, stream, static_cast<const float*>(pool->keys), pool->valid, total_new,
            qpf2k_key_ranges(pool->capacity), slot.bounds, groups,
            qsa_flash::index_band() ? qpf2k_key_norms(pool->capacity) : nullptr);
        check_hip(hipGetLastError(), "QSA stage batch range launch");
    }
    if (!compact) qpf2k_stage_score_slice(batch, pool, a.index_dim, 0U, groups);
    if (wave_norm) qpf2k_launch_norm_wave.fetch_add(1U, std::memory_order_relaxed);
    pool->valid = valid;
    if (!compact && qpf2k_trace_enabled())
        std::fprintf(stderr, "qpf2k qsa stage_batch groups %u pool %u range %u score %u rope 1 norm 1 store 1 fallback 0%s\n",
            groups, max_new ? 1U : 0U, max_new ? 1U : 0U, max_blocks ? 1U : 0U,
            !max_blocks || qsa_flash::score_rows(max_blocks) == 16U ? "" : qsa_flash::score_rows(max_blocks) == 32U ? " score_rows 32" : " score_rows 64");
    return true;
}
