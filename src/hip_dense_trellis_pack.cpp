#include "strix/hip_check.hpp"
#include "strix/hip_dense_trellis_pack.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/model_content_hash.hpp"
#include <hip/hip_runtime.h>
#include <array>
#include <atomic>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace strix {
namespace {
constexpr std::uint64_t transform_bytes = 8ULL * 2560U * 4U;
constexpr std::uint64_t partial_bytes = 4ULL * 8U * 12288U * 4U;
constexpr unsigned stream_slots = 32U;
using strix::hip_check;
void require(bool value, const char* message) { if (!value) throw std::invalid_argument(message); }
std::string text(const gguf::FileInfo& info, const std::string& key) {
    const auto* entry = info.find_metadata(key);
    return entry && entry->string_value ? *entry->string_value : std::string{};
}
struct Slot {
    float* transformed = nullptr;
    float* partials = nullptr;
    ~Slot() {
        if (partials) static_cast<void>(hipFree(partials));
        if (transformed) static_cast<void>(hipFree(transformed));
    }
};
std::atomic<HipDenseTrellisPack::Impl*> active{nullptr};
thread_local bool permitted = false;
} // namespace
struct HipDenseTrellisPack::Impl {
    struct Tensor { HipDenseTrellisWeights weights; unsigned layer; };
    std::shared_ptr<const MappedModelFile> mapped;
    std::shared_ptr<const HipModelResidency> residency;
    std::unordered_map<const void*, Tensor> tensors;
    std::atomic<std::uint64_t> used_layers{0};
    std::array<Slot, stream_slots> slots;
    std::unordered_map<void*, unsigned> assigned;
    std::mutex mutex;
    HipDenseTrellisWorkspace workspace(void* stream) {
        const std::lock_guard lock(mutex);
        auto found = assigned.find(stream);
        if (found == assigned.end()) {
            require(assigned.size() < slots.size(), "dense trellis reserved stream slots exhausted");
            const auto index = static_cast<unsigned>(assigned.size());
            found = assigned.emplace(stream, index).first;
        }
        auto& slot = slots[found->second];
        return {slot.transformed, slot.partials, transform_bytes, partial_bytes};
    }
};
HipDenseTrellisPack::~HipDenseTrellisPack() {
    auto* expected = impl.get();
    static_cast<void>(active.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel));
}
bool hip_dense_trellis_active() noexcept { return active.load(std::memory_order_acquire) != nullptr; }
bool hip_dense_trellis_permitted() noexcept { return permitted; }
std::uint64_t hip_dense_trellis_used_layers() noexcept {
    const auto* state = active.load(std::memory_order_acquire);
    return state ? state->used_layers.load(std::memory_order_relaxed) : 0U;
}
HipDenseTrellisScope::HipDenseTrellisScope(bool enabled) : previous_(permitted) {
    require(!enabled || hip_dense_trellis_active(), "dense trellis arithmetic has no active pinned pack");
    permitted = enabled;
}
HipDenseTrellisScope::~HipDenseTrellisScope() { permitted = previous_; }
bool project_hip_dense_trellis_registered(const void* original, const HipDenseTrellisRows& views,
    unsigned rows, unsigned inputs, unsigned outputs, void* stream) {
    auto* registry = active.load(std::memory_order_acquire);
    if (!registry || !permitted) return false;
    const auto it = registry->tensors.find(original);
    if (it == registry->tensors.end()) return false;
    require(it->second.weights.inputs == inputs && it->second.weights.outputs == outputs, "dense trellis site geometry differs from selected tensor");
    launch_hip_dense_trellis_rows(it->second.weights, views, rows, registry->workspace(stream), stream);
    registry->used_layers.fetch_or(1ULL << it->second.layer, std::memory_order_relaxed);
    return true;
}
std::shared_ptr<const HipDenseTrellisPack> load_hip_dense_trellis_pack(const std::filesystem::path& path,
    const MappedModelFile& target, const ModelManifest& target_manifest, const ModelRuntimeBinding& target_binding,
    AllocationGenerationSource& generations) {
    require(!hip_dense_trellis_active(), "only one dense trellis pack may be active");
    const auto info = gguf::read_file(path);
    require(text(info, "strix.dense_trellis.format") == "TRELLIS128-mul1-K5-v1" && text(info, "strix.dense_trellis.mix") == "G1" &&
        text(info, "strix.dense_trellis.recipe") == "fp32-transform-fma-split4-v1" &&
        text(info, "strix.dense_trellis.scope") == "decode-r1-r2-r4-r8" && text(info, "strix.dense_trellis.marker") == "83dcd12d",
        "dense trellis pack format, mix or recipe differs");
    require(text(info, "strix.dense_trellis.target_content") == model_content_identity_hex(target.content_identity()) &&
        text(info, "strix.dense_trellis.target_file_sha256") == model_content_identity_hex(target.file_sha256()),
        "dense trellis pack source checkpoint differs");
    require(target_binding.tensors.size() == target_manifest.tensors.size(), "dense trellis source binding differs");
    auto pack = std::make_shared<HipDenseTrellisPack>();
    pack->impl = std::make_shared<HipDenseTrellisPack::Impl>();
    auto& state = *pack->impl;
    state.mapped = std::make_shared<const MappedModelFile>(MappedModelFile::open_read_only(path, "dense-trellis-pack", generations));
    pack->sha256 = model_content_identity_hex(state.mapped->file_sha256());
    require(pack->sha256 == dense_trellis_g1_pack_sha256, "dense trellis pack bytes differ from the selected G1 artifact");
    const auto manifest = build_model_manifest(info);
    require(manifest.tensors.size() == 144U, "dense trellis G1 requires exactly forty-eight tensor triples");
    std::map<std::string, std::size_t> source, components;
    for (std::size_t i = 0; i < target_manifest.tensors.size(); ++i) source.emplace(target_manifest.tensors[i].name, i);
    for (std::size_t i = 0; i < manifest.tensors.size(); ++i)
        require(components.emplace(manifest.tensors[i].name, i).second, "dense trellis duplicate component");
    HipModelResidencyPolicy policy;
    policy.reservation_limits.retained_bytes = state.mapped->bytes() + (64ULL << 20U);
    policy.reservation_limits.setup_peak_bytes = policy.reservation_limits.retained_bytes;
    state.residency = std::make_shared<const HipModelResidency>(HipModelResidency::create(state.mapped, manifest, generations, policy));
    const auto& binding = state.residency->binding();
    require(binding.tensors.size() == manifest.tensors.size(), "dense trellis pack binding differs");
    for (unsigned layer = 0; layer < 48U; ++layer) {
        const auto base = "blk." + std::to_string(layer) + (layer % 4U == 3U ? ".attn_q.weight" : ".attn_gate.weight");
        const auto found = source.find(base);
        require(found != source.end(), "dense trellis selected tensor is absent from target");
        const auto& original = target_manifest.tensors[found->second];
        const auto output = layer % 4U == 3U ? 12288U : 6144U;
        require(original.ggml_type == 8U && original.dimensions == std::vector<std::uint64_t>{2560U, output}, "dense trellis target shape or type differs");
        const auto component = [&](const char* suffix, unsigned kind, std::uint64_t bytes) {
            const auto part = components.find(base + suffix);
            require(part != components.end(), "dense trellis component is absent");
            const auto& tensor = manifest.tensors[part->second];
            require(tensor.ggml_type == kind && tensor.byte_size == bytes, "dense trellis component type or size differs");
            return binding.tensors[part->second].address;
        };
        const auto packed_bytes = static_cast<std::uint64_t>(2560U) * output * 5U / 8U;
        HipDenseTrellisWeights weights{reinterpret_cast<const std::uint32_t*>(component(".trellis", 24U, packed_bytes)),
            reinterpret_cast<const std::uint16_t*>(component(".suh", 1U, 2560U * 2U)),
            reinterpret_cast<const std::uint16_t*>(component(".svh", 1U, output * 2U)), 2560U, output, 5U};
        require(state.tensors.emplace(reinterpret_cast<const void*>(target_binding.tensors[found->second].address), HipDenseTrellisPack::Impl::Tensor{weights, layer}).second,
            "dense trellis duplicate target address");
        pack->packed_bytes += packed_bytes + (2560ULL + output) * 2U;
        pack->replaced_bytes += original.byte_size;
    }
    // No device allocation is permitted in a captured projection. All streams
    // borrow distinct preallocated slots; first binding only mutates host maps.
    state.assigned.reserve(stream_slots);
    for (auto& slot : state.slots) {
        hip_check(hipMalloc(reinterpret_cast<void**>(&slot.transformed), transform_bytes));
        hip_check(hipMalloc(reinterpret_cast<void**>(&slot.partials), partial_bytes));
    }
    pack->workspace_bytes = stream_slots * (transform_bytes + partial_bytes);
    pack->tensors = state.tensors.size(); pack->mix = "G1";
    HipDenseTrellisPack::Impl* expected = nullptr;
    require(active.compare_exchange_strong(expected, &state, std::memory_order_release), "dense trellis concurrent pack activation");
    return pack;
}
} // namespace strix
