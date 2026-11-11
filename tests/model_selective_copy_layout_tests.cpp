#include "strix/hip_model_residency.hpp"
#include "strix/host_memory.hpp"
#include "strix/process_environment.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void expect(bool value, std::source_location at = std::source_location::current()) {
    if (!value) throw std::runtime_error("selective copy layout failed at line " + std::to_string(at.line()));
}
template<class Action> void rejects(Action action) {
    bool failed = false;
    try { action(); } catch (const std::exception&) { failed = true; }
    expect(failed);
}
strix::ManifestTensor tensor(strix::TensorRole role, std::uint64_t offset, std::uint64_t bytes) {
    strix::ManifestTensor result;
    result.name = "fixture." + std::to_string(offset);
    result.role = role;
    result.ggml_type = 24; // One byte per element; synthetic byte-layout fixture.
    result.absolute_offset = offset;
    result.byte_size = bytes;
    result.dimensions = {bytes};
    return result;
}
void seal(strix::ModelManifest& manifest) {
    manifest.layout_identity = strix::model_manifest_layout_identity(manifest);
}
strix::HipModelResidencyPolicy policy(strix::HipModelCopyTier tier, std::uint64_t payload) {
    strix::HipModelResidencyPolicy result;
    result.copy_tier = tier;
    result.copy_payload_bytes = payload;
    result.reservation_limits.retained_bytes = std::numeric_limits<std::uint64_t>::max();
    result.reservation_limits.setup_peak_bytes = std::numeric_limits<std::uint64_t>::max();
    return result;
}
void expert_selection_contract() {
    using namespace strix;
    constexpr std::uint64_t page = 4096, file_bytes = 98304;
    ModelManifest manifest;
    manifest.tensors = {tensor(TensorRole::normalization, 4103, 17),
        tensor(TensorRole::moe_expert, 8201, 12320), tensor(TensorRole::dense_projection, 28675, 31),
        tensor(TensorRole::moe_expert, 40971, 12336), tensor(TensorRole::moe_expert, 65549, 64),
        tensor(TensorRole::moe_expert, 73731, 128), tensor(TensorRole::moe_expert, 81925, 128),
        tensor(TensorRole::embedding, 90119, 128)};
    // Selection follows typed placement/residency/role, not a suggestive name.
    manifest.tensors[0].name = "blk.0.ffn_gate_exps.weight";
    manifest.tensors[1].name = "typed.expert.without.filename.convention";
    manifest.tensors[3].name = "blk.1.ffn_down_exps.weight";
    manifest.tensors[4].placement.kind = PlacementKind::remote;
    manifest.tensors[5].residency = ResidencyPolicy::file_backed_candidate;
    manifest.tensors[6].residency = ResidencyPolicy::bounded_cache_candidate;
    seal(manifest);
    const auto identity = manifest.layout_identity;
    auto experts = policy(HipModelCopyTier::experts, 24656);
    expect(experts.coalesce_gap_bytes == 0);
    const auto selected = build_selective_copy_plan(manifest, experts, file_bytes, page);
    expect(selected.tensors == std::vector<std::uint32_t>({1, 3}));
    expect(selected.payload_bytes == 24656 && selected.arena_bytes == 32768 && selected.ranges.size() == 2);
    expect(selected.ranges[0].file_begin == 8192 && selected.ranges[0].file_end == 24576 &&
        selected.ranges[0].arena_offset == 0);
    expect(selected.ranges[1].file_begin == 40960 && selected.ranges[1].file_end == 57344 &&
        selected.ranges[1].arena_offset == 16384);
    for (const auto index : selected.tensors) {
        const auto& t = manifest.tensors[index];
        const auto range = std::find_if(selected.ranges.begin(), selected.ranges.end(), [&](const auto& r) {
            return r.file_begin <= t.absolute_offset && t.absolute_offset < r.file_end &&
                t.byte_size <= r.file_end - t.absolute_offset;
        });
        expect(range != selected.ranges.end());
        const auto relative = range->arena_offset + t.absolute_offset - range->file_begin;
        expect(relative % page == t.absolute_offset % page && relative <= selected.arena_bytes &&
            t.byte_size <= selected.arena_bytes - relative);
    }
    // all (STRIX_MODEL_COPY_ALL): every local tensor except unpromoted file-backed candidates, page congruent.
    const auto all_payload = selective_copy_payload_bytes(manifest, HipModelCopyTier::all);
    expect(all_payload == 17 + 12320 + 31 + 12336 + 128 + 128);
    const auto every = build_selective_copy_plan(manifest, policy(HipModelCopyTier::all, all_payload), file_bytes, page);
    expect(every.tensors == std::vector<std::uint32_t>({0, 1, 2, 3, 6, 7}) && every.payload_bytes == all_payload);
    for (const auto index : every.tensors) {
        const auto& t = manifest.tensors[index];
        const auto range = std::find_if(every.ranges.begin(), every.ranges.end(), [&](const auto& r) {
            return r.file_begin <= t.absolute_offset && t.byte_size <= r.file_end - t.absolute_offset;
        });
        expect(range != every.ranges.end() &&
            (range->arena_offset + t.absolute_offset - range->file_begin) % page == t.absolute_offset % page);
    }
    auto gapped = policy(HipModelCopyTier::all, all_payload); gapped.coalesce_gap_bytes = 4096;
    rejects([&] { (void)build_selective_copy_plan(manifest, gapped, file_bytes, page); });
    for (const auto role : {TensorRole::unknown, TensorRole::embedding, TensorRole::output_head,
        TensorRole::normalization, TensorRole::attention_projection, TensorRole::dense_projection,
        TensorRole::moe_router, TensorRole::recurrent_state, TensorRole::sparse_index,
        TensorRole::ngram_table, TensorRole::ple_parameter, TensorRole::mtp})
        expect(!selective_copy_eligible(tensor(role, 0, 4), HipModelCopyTier::experts));
    for (const auto index : {1U, 3U}) {
        expect(selective_copy_eligible(manifest.tensors[index], HipModelCopyTier::experts));
        for (const auto tier : {HipModelCopyTier::none, HipModelCopyTier::normalization, HipModelCopyTier::hot_nonexpert})
            expect(!selective_copy_eligible(manifest.tensors[index], tier));
    }
    for (const auto index : {4U, 5U, 6U})
        expect(!selective_copy_eligible(manifest.tensors[index], HipModelCopyTier::experts));
    const auto norms = build_selective_copy_plan(manifest, policy(HipModelCopyTier::normalization, 17), file_bytes, page);
    const auto hot = build_selective_copy_plan(manifest, policy(HipModelCopyTier::hot_nonexpert, 48), file_bytes, page);
    expect(norms.tensors == std::vector<std::uint32_t>({0}) &&
        hot.tensors == std::vector<std::uint32_t>({0, 2}));
    for (const auto gap : {std::uint64_t{0}, std::uint64_t{4096}}) {
        HipModelResidencyPolicy none; none.coalesce_gap_bytes = gap;
        const auto empty = build_selective_copy_plan(manifest, none, file_bytes, page);
        expect(empty.tensors.empty() && empty.ranges.empty() && empty.payload_bytes == 0 && empty.arena_bytes == 0);
    }
    for (const auto pin : {std::uint64_t{0}, std::uint64_t{24655}, std::uint64_t{24657}}) {
        auto bad = experts; bad.copy_payload_bytes = pin;
        rejects([&] { (void)build_selective_copy_plan(manifest, bad, file_bytes, page); });
    }
    for (const auto gap : {std::uint64_t{1}, std::uint64_t{4096}, UINT64_MAX}) {
        auto bad = experts; bad.coalesce_gap_bytes = gap;
        rejects([&] { (void)build_selective_copy_plan(manifest, bad, file_bytes, page); });
    }
    for (unsigned missing = 0; missing < 3; ++missing) {
        auto bad = experts;
        if (missing == 0) bad.copy_payload_bytes.reset();
        if (missing == 1) bad.reservation_limits.retained_bytes.reset();
        if (missing == 2) bad.reservation_limits.setup_peak_bytes.reset();
        rejects([&] { (void)build_selective_copy_plan(manifest, bad, file_bytes, page); });
    }
    auto bad = experts; bad.mode = HipModelResidencyMode::resident_copy;
    rejects([&] { (void)build_selective_copy_plan(manifest, bad, file_bytes, page); });
    bad = experts; bad.promote_file_backed_candidates = true;
    rejects([&] { (void)build_selective_copy_plan(manifest, bad, file_bytes, page); });
    auto malformed = manifest; malformed.tensors[1].dimensions = {12319}; seal(malformed);
    rejects([&] { (void)build_selective_copy_plan(malformed, experts, file_bytes, page); });
    expect(hip_model_copy_tier_name(HipModelCopyTier::none) == "none" &&
        hip_model_copy_tier_name(HipModelCopyTier::normalization) == "normalization" &&
        hip_model_copy_tier_name(HipModelCopyTier::hot_nonexpert) == "hot_nonexpert" &&
        hip_model_copy_tier_name(HipModelCopyTier::experts) == "experts");
    rejects([&] { (void)hip_model_copy_tier_name(static_cast<HipModelCopyTier>(99)); });
    expect(manifest.layout_identity == identity && model_manifest_layout_identity(manifest) == identity);
    // Pure selection preserves the existing bounded EOF rule, including a partial final page.
    ModelManifest eof; eof.tensors = {tensor(TensorRole::moe_expert, 53255, 12280)}; seal(eof);
    const auto tail = build_selective_copy_plan(eof, policy(HipModelCopyTier::experts, 12280), 65535, page);
    expect(tail.ranges.size() == 1 && tail.ranges[0].file_begin == 53248 &&
        tail.ranges[0].file_end == 65535 && tail.ranges[0].arena_offset == 0 && tail.arena_bytes == 12287);
}
}

int main() try {
    // STRIX_MODEL_COPY_ALL=0 keeps the registered backing where host and GPU share memory. Where host memory is
    // a separate pool the copied arena is the only form, so 0 selects it too. (First call: the mode is read once.)
    expect(strix::process_environment::set("STRIX_MODEL_COPY_ALL", "0", true));
    expect(strix::hip_model_copy_all_mode() == (strix::host_memory_is_separate_pool() ? 1U : 0U));
    {
        using namespace strix;
        ModelManifest manifest;
        manifest.tensors={tensor(TensorRole::embedding,4103,17),
            tensor(TensorRole::ngram_table,8192,16384),
            tensor(TensorRole::normalization,24579,31),
            tensor(TensorRole::dense_projection,32769,31)};
        manifest.tensors[1].residency=ResidencyPolicy::file_backed_candidate;
        manifest.tensors[3].placement.kind=PlacementKind::remote;
        HipModelResidencyPolicy p;
        const auto ranges=build_registered_residency_ranges(manifest,p,65536,4096);
        expect(ranges.size()==2 && ranges[0].begin==4096 && ranges[0].end==8192 &&
            ranges[1].begin==24576 && ranges[1].end==28672);
        // Coalescing can commit otherwise reclaimable pages; charge the gap.
        p.coalesce_gap_bytes=16384;
        const auto coalesced=build_registered_residency_ranges(manifest,p,65536,4096);
        expect(coalesced.size()==1 && coalesced[0].begin==4096 && coalesced[0].end==28672);
        p.coalesce_gap_bytes=0; p.promote_file_backed_candidates=true;
        const auto promoted=build_registered_residency_ranges(manifest,p,65536,4096);
        expect(promoted.size()==1 && promoted[0].begin==4096 && promoted[0].end==28672);
        rejects([&] { (void)build_registered_residency_ranges(manifest,p,4096,4096); });
        rejects([&] { (void)build_registered_residency_ranges(manifest,p,65536,3); });
    }
    using namespace strix;
    ModelManifest manifest;
    manifest.tensors = {tensor(TensorRole::normalization, 4103, 33),
        tensor(TensorRole::output_head, 8190, 10), tensor(TensorRole::dense_projection, 20003, 7),
        tensor(TensorRole::moe_expert, 32768, 4096), tensor(TensorRole::embedding, 49152, 1024)};
    seal(manifest);
    const auto identity = manifest.layout_identity;
    for (const auto mode : {HipModelResidencyMode::registered_file_mapping, HipModelResidencyMode::resident_copy}) {
        HipModelResidencyPolicy defaults; defaults.mode = mode;
        const auto empty = build_selective_copy_plan(manifest, defaults, 65536, 4096);
        expect(empty.tensors.empty() && empty.ranges.empty() && empty.payload_bytes == 0 && empty.arena_bytes == 0);
    }
    auto norms = policy(HipModelCopyTier::normalization, 33);
    const auto a = build_selective_copy_plan(manifest, norms, 65536, 4096);
    expect(a.payload_bytes == 33 && a.arena_bytes == 4096);
    expect(a.tensors.size() == 1 && a.tensors.front() == 0 && a.ranges.size() == 1);
    expect(a.ranges[0].file_begin == 4096 && a.ranges[0].file_end == 8192 && a.ranges[0].arena_offset == 0);
    auto hot = policy(HipModelCopyTier::hot_nonexpert, 50);
    const auto b = build_selective_copy_plan(manifest, hot, 65536, 4096);
    expect(b.payload_bytes == 50 && b.arena_bytes == 12288 && b.tensors.size() == 3 && b.ranges.size() == 2);
    expect(b.ranges[0].file_begin == 4096 && b.ranges[0].file_end == 12288 && b.ranges[0].arena_offset == 0);
    expect(b.ranges[1].file_begin == 16384 && b.ranges[1].file_end == 20480 && b.ranges[1].arena_offset == 8192);
    for (const auto index : b.tensors) {
        const auto& t = manifest.tensors.at(index);
        const auto range = std::find_if(b.ranges.begin(), b.ranges.end(), [&](const auto& r) {
            return t.absolute_offset >= r.file_begin && t.absolute_offset < r.file_end &&
                t.byte_size <= r.file_end - t.absolute_offset;
        });
        expect(range != b.ranges.end());
        const auto relative = range->arena_offset + t.absolute_offset - range->file_begin;
        expect(relative % 4096 == t.absolute_offset % 4096);
        expect(relative <= b.arena_bytes && t.byte_size <= b.arena_bytes - relative);
    }
    expect(manifest.layout_identity == identity && model_manifest_layout_identity(manifest) == identity);
    // Registration coalescing must not pull the unselected gap into the arena.
    hot.coalesce_gap_bytes = 65536;
    const auto coalesced = build_selective_copy_plan(manifest, hot, 65536, 4096);
    expect(coalesced.ranges.size() == b.ranges.size() && coalesced.arena_bytes == b.arena_bytes);
    for (const auto role : {TensorRole::embedding, TensorRole::moe_expert, TensorRole::ngram_table,
                           TensorRole::unknown, TensorRole::mtp})
        expect(!selective_copy_eligible(tensor(role, 0, 4), HipModelCopyTier::hot_nonexpert));
    for (const auto role : {TensorRole::output_head, TensorRole::normalization, TensorRole::attention_projection,
                           TensorRole::dense_projection, TensorRole::moe_router, TensorRole::recurrent_state,
                           TensorRole::sparse_index, TensorRole::ple_parameter})
        expect(selective_copy_eligible(tensor(role, 0, 4), HipModelCopyTier::hot_nonexpert));
    auto excluded = tensor(TensorRole::normalization, 0, 4);
    excluded.placement.kind = PlacementKind::remote;
    expect(!selective_copy_eligible(excluded, HipModelCopyTier::normalization));
    excluded.placement.kind = PlacementKind::local;
    for (const auto residency : {ResidencyPolicy::file_backed_candidate, ResidencyPolicy::bounded_cache_candidate}) {
        excluded.residency = residency;
        expect(!selective_copy_eligible(excluded, HipModelCopyTier::hot_nonexpert));
    }
    auto bad = norms; bad.copy_tier = static_cast<HipModelCopyTier>(99);
    rejects([&] { (void)build_selective_copy_plan(manifest, bad, 65536, 4096); });
    bad = norms; bad.mode = HipModelResidencyMode::resident_copy;
    rejects([&] { (void)build_selective_copy_plan(manifest, bad, 65536, 4096); });
    bad = norms; bad.promote_file_backed_candidates = true;
    rejects([&] { (void)build_selective_copy_plan(manifest, bad, 65536, 4096); });
    auto invalid = manifest; invalid.unknown_tensor_count = 1; seal(invalid);
    rejects([&] { (void)build_selective_copy_plan(invalid, norms, 65536, 4096); });
    invalid = manifest; invalid.tensors.clear(); seal(invalid);
    rejects([&] { (void)build_selective_copy_plan(invalid, norms, 65536, 4096); });
    for (unsigned fault = 0; fault < 3; ++fault) {
        invalid = manifest;
        if (fault == 0) invalid.tensors[0].byte_size = 0;
        if (fault == 1) invalid.tensors[0].dimensions.clear();
        if (fault == 2) invalid.tensors[0].absolute_offset = 65530;
        seal(invalid);
        rejects([&] { (void)build_selective_copy_plan(invalid, norms, 65536, 4096); });
    }
    rejects([&] { (void)build_selective_copy_plan(manifest, norms, 65536, 0); });
    rejects([&] { (void)build_selective_copy_plan(manifest, norms, 65536, 3000); });
    bad = norms; bad.copy_payload_bytes = 32;
    rejects([&] { (void)build_selective_copy_plan(manifest, bad, 65536, 4096); });
    for (unsigned missing = 0; missing < 3; ++missing) {
        bad = norms;
        if (missing == 0) bad.copy_payload_bytes.reset();
        if (missing == 1) bad.reservation_limits.retained_bytes.reset();
        if (missing == 2) bad.reservation_limits.setup_peak_bytes.reset();
        rejects([&] { (void)build_selective_copy_plan(manifest, bad, 65536, 4096); });
    }
    invalid = manifest; invalid.tensors[0].dimensions = {32}; seal(invalid);
    rejects([&] { (void)build_selective_copy_plan(invalid, norms, 65536, 4096); });
    // The final range is clipped to real file bytes: no read past mmap EOF.
    invalid = {}; invalid.tensors = {tensor(TensorRole::normalization, 65530, 5)}; seal(invalid);
    const auto tail = build_selective_copy_plan(invalid, policy(HipModelCopyTier::normalization, 5), 65535, 4096);
    expect(tail.ranges.size() == 1 && tail.ranges[0].file_end == 65535 && tail.arena_bytes == 4095);
    // Overflow in tensor extent and in page padding are distinct errors.
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    invalid = {}; invalid.tensors = {tensor(TensorRole::normalization, maximum - 2, 4)}; seal(invalid);
    rejects([&] { (void)build_selective_copy_plan(invalid, policy(HipModelCopyTier::normalization, 4), maximum, 4096); });
    invalid.tensors[0].byte_size = 1; invalid.tensors[0].dimensions = {1}; seal(invalid);
    rejects([&] { (void)build_selective_copy_plan(invalid, policy(HipModelCopyTier::normalization, 1), maximum, 4096); });
    expert_selection_contract();
    std::cout << "selective model copy selection, page congruence and bounded layout PASS\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
