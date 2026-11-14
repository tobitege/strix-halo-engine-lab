#include "strix/hip_model_residency.hpp"
#include "strix/host_memory.hpp"
#include <algorithm>
#include <cstdlib>
#include <string_view>
#include <limits>
#include <stdexcept>
namespace strix {
std::vector<CopyResidencyRange> build_registered_residency_ranges(const ModelManifest& manifest,
    const HipModelResidencyPolicy& policy, std::uint64_t file_bytes, std::uint64_t page) {
    if (!page || (page & (page-1U)) || !file_bytes)
        throw std::invalid_argument("registration requires file bytes and power-of-two page");
    const auto add=[](std::uint64_t a,std::uint64_t b) {
        if (b>UINT64_MAX-a) throw std::overflow_error("registration range overflow");
        return a+b;
    };
    const auto up=[&](std::uint64_t n) { return add(n,(page-n%page)%page); };
    std::vector<CopyResidencyRange> ranges;
    for (const auto& tensor:manifest.tensors) {
        if (tensor.placement.kind==PlacementKind::remote ||
            (tensor.residency==ResidencyPolicy::file_backed_candidate && !policy.promote_file_backed_candidates) ||
            ((policy.copy_tier==HipModelCopyTier::experts || policy.copy_tier==HipModelCopyTier::all) &&
                selective_copy_eligible(tensor,policy.copy_tier))) continue;
        if (tensor.absolute_offset>file_bytes || tensor.byte_size>file_bytes-tensor.absolute_offset)
            throw std::invalid_argument("manifest tensor exceeds mapped model file");
        const auto begin=tensor.absolute_offset-tensor.absolute_offset%page;
        const auto end=std::min(up(add(tensor.absolute_offset,tensor.byte_size)),up(file_bytes));
        if (end<=begin) throw std::logic_error("registered tensor range is empty");
        ranges.push_back({begin,end});
    }
    std::sort(ranges.begin(),ranges.end(),[](const auto& a,const auto& b) {
        return a.begin!=b.begin ? a.begin<b.begin : a.end<b.end;
    });
    std::vector<CopyResidencyRange> merged;
    for (const auto range:ranges) {
        if (!merged.empty() && range.begin<=add(merged.back().end,policy.coalesce_gap_bytes))
            merged.back().end=std::max(merged.back().end,range.end);
        else merged.push_back(range);
    }
    return merged;
}
std::vector<CopyResidencyRange> build_copy_residency_ranges(const ModelManifest& manifest,const HipModelResidencyPolicy& policy) {
    const auto add=[](std::uint64_t a,std::uint64_t b){if(b>std::numeric_limits<std::uint64_t>::max()-a)throw std::overflow_error("HIP model residency range overflow");return a+b;};
    std::vector<CopyResidencyRange> ranges;
    for(const auto& tensor:manifest.tensors){
        if(tensor.placement.kind==PlacementKind::remote || (tensor.residency==ResidencyPolicy::file_backed_candidate && !policy.promote_file_backed_candidates))continue;
        ranges.push_back({tensor.absolute_offset,add(tensor.absolute_offset,tensor.byte_size)});
    }
    std::sort(ranges.begin(),ranges.end(),[](const auto& a,const auto& b){return a.begin!=b.begin?a.begin<b.begin:a.end<b.end;});
    std::vector<CopyResidencyRange> merged;
    for(const auto range:ranges){if(!merged.empty() && range.begin<=add(merged.back().end,policy.coalesce_gap_bytes))merged.back().end=std::max(merged.back().end,range.end);else merged.push_back(range);}
    return merged;
}
std::uint64_t copy_residency_tensor_offset(const ManifestTensor& tensor,const std::vector<CopyResidencyRange>& ranges){
    for(const auto range:ranges)if(tensor.absolute_offset>=range.begin && tensor.absolute_offset<=range.end && tensor.byte_size<=range.end-tensor.absolute_offset)return tensor.absolute_offset-range.begin;
    throw std::invalid_argument("tensor has no copied residency range");
}

std::string_view hip_model_copy_tier_name(HipModelCopyTier tier) {
    switch (tier) {
    case HipModelCopyTier::none: return "none";
    case HipModelCopyTier::normalization: return "normalization";
    case HipModelCopyTier::hot_nonexpert: return "hot_nonexpert";
    case HipModelCopyTier::experts: return "experts";
    case HipModelCopyTier::all: return "all";
    }
    throw std::invalid_argument("invalid model copy tier");
}
bool selective_copy_eligible(const ManifestTensor& tensor, HipModelCopyTier tier) noexcept {
    if (tier == HipModelCopyTier::all)
        return tensor.placement.kind == PlacementKind::local && tensor.residency != ResidencyPolicy::file_backed_candidate;
    if (tensor.placement.kind != PlacementKind::local || tensor.residency != ResidencyPolicy::resident)
        return false;
    if (tier == HipModelCopyTier::normalization) return tensor.role == TensorRole::normalization;
    if (tier == HipModelCopyTier::experts) return tensor.role == TensorRole::moe_expert;
    if (tier != HipModelCopyTier::hot_nonexpert) return false;
    switch (tensor.role) {
    case TensorRole::output_head: case TensorRole::normalization: case TensorRole::attention_projection:
    case TensorRole::dense_projection: case TensorRole::moe_router: case TensorRole::recurrent_state:
    case TensorRole::sparse_index: case TensorRole::ple_parameter: return true;
    default: return false;
    }
}
std::uint64_t selective_copy_payload_bytes(const ModelManifest& manifest, HipModelCopyTier tier) noexcept {
    std::uint64_t total = 0;
    for (const auto& tensor : manifest.tensors)
        if (selective_copy_eligible(tensor, tier)) total += tensor.byte_size;
    return total;
}
unsigned hip_model_copy_all_mode() noexcept {
    static const unsigned mode = [] {
        const char* v = std::getenv("STRIX_MODEL_COPY_ALL");
        // A registered mapping stays host memory. Where host memory is a separate pool, the model belongs in the
        // GPU pool, so the copied arena is the only form there and unset or 0 select the hipMalloc arena.
        if (!v || !v[0] || v[0] == '0') return host_memory_is_separate_pool() ? 1U : 0U;
        if (std::string_view(v) == "fine") return 2U;
        if (std::string_view(v) == "uc") return 3U;
        return 1U;
    }();
    return mode;
}
SelectiveCopyPlan build_selective_copy_plan(const ModelManifest& manifest,
    const HipModelResidencyPolicy& policy, std::uint64_t file_bytes, std::uint64_t page) {
    (void)hip_model_copy_tier_name(policy.copy_tier);
    SelectiveCopyPlan result;
    if (policy.copy_tier == HipModelCopyTier::none) {
        if (policy.copy_payload_bytes) throw std::invalid_argument("model copy payload requires a selective tier");
        return result;
    }
    if (policy.mode != HipModelResidencyMode::registered_file_mapping || policy.promote_file_backed_candidates)
        throw std::invalid_argument("selective model copy requires registered mode without file candidate promotion");
    if ((policy.copy_tier == HipModelCopyTier::experts || policy.copy_tier == HipModelCopyTier::all) && policy.coalesce_gap_bytes)
        throw std::invalid_argument("expert/all model copy requires zero registration coalescing gap");
    if (!policy.copy_payload_bytes || !*policy.copy_payload_bytes || !policy.reservation_limits.retained_bytes ||
        !policy.reservation_limits.setup_peak_bytes)
        throw std::invalid_argument("selective model copy requires payload and retained/setup budgets");
    if (!page || (page & (page - 1U)) || !file_bytes || file_bytes > SIZE_MAX ||
        manifest.tensors.size() > UINT32_MAX || manifest.unknown_tensor_count || !manifest.layout_identity ||
        manifest.layout_identity != model_manifest_layout_identity(manifest))
        throw std::invalid_argument("selective model copy requires bounded known manifest and power-of-two page");
    const auto add = [](std::uint64_t a, std::uint64_t b) {
        if (b > UINT64_MAX - a) throw std::overflow_error("selective model copy range overflow");
        return a + b;
    };
    const auto up = [&](std::uint64_t value) { return add(value, (page - value % page) % page); };
    std::vector<SelectiveCopyRange> ranges;
    for (std::size_t i = 0; i < manifest.tensors.size(); ++i) {
        const auto& tensor = manifest.tensors[i];
        if (!selective_copy_eligible(tensor, policy.copy_tier)) continue;
        const auto geometry = gguf::tensor_type_geometry(tensor.ggml_type);
        if (!geometry || !geometry->block_elements || !geometry->block_bytes || tensor.dimensions.empty() ||
            !tensor.dimensions[0] || tensor.dimensions[0] % geometry->block_elements)
            throw std::invalid_argument("selective model copy tensor type or row geometry is invalid");
        std::uint64_t elements = 1;
        for (const auto dim : tensor.dimensions) {
            if (!dim || elements > UINT64_MAX / dim) throw std::overflow_error("selective tensor dimensions overflow");
            elements *= dim;
        }
        const auto blocks = elements / geometry->block_elements;
        if (blocks > UINT64_MAX / geometry->block_bytes) throw std::overflow_error("selective tensor bytes overflow");
        if (!tensor.byte_size || blocks * geometry->block_bytes != tensor.byte_size ||
            tensor.absolute_offset > file_bytes || tensor.byte_size > file_bytes - tensor.absolute_offset)
            throw std::invalid_argument("selective model copy tensor extent differs from source geometry");
        result.payload_bytes = add(result.payload_bytes, tensor.byte_size);
        result.tensors.push_back(static_cast<std::uint32_t>(i));
        ranges.push_back({tensor.absolute_offset - tensor.absolute_offset % page,
            std::min(up(add(tensor.absolute_offset, tensor.byte_size)), file_bytes), 0});
    }
    if (result.tensors.empty() || result.payload_bytes != *policy.copy_payload_bytes)
        throw std::invalid_argument("selective model copy payload pin or selection differs");
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {return a.file_begin < b.file_begin;});
    for (const auto& range : ranges) {
        if (!result.ranges.empty() && range.file_begin <= result.ranges.back().file_end)
            result.ranges.back().file_end = std::max(result.ranges.back().file_end, range.file_end);
        else result.ranges.push_back(range);
    }
    for (auto& range : result.ranges) {
        range.arena_offset = up(result.arena_bytes);
        result.arena_bytes = add(range.arena_offset, range.file_end - range.file_begin);
    }
    if (result.arena_bytes > SIZE_MAX) throw std::overflow_error("selective copy arena exceeds size_t");
    return result;
}
}
