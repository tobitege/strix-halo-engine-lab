#include "strix/hip_check.hpp"
#include "strix/hip_qwen_router_bf16.hpp"
#include "strix/qwen_router_bf16.hpp"
#include "strix/q6p_registry.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace strix {
namespace {
using strix::hip_check;
}
HipQwenRouterBf16::~HipQwenRouterBf16() {
    for (const auto& c : copies) {
        q6p::unregister_tensor(c.original, static_cast<const unsigned char*>(c.device));
        (void)hipFree(c.device);
    }
}
UmaOwnerInventory HipQwenRouterBf16::owner_inventory(std::shared_ptr<const HipQwenRouterBf16> owner) const {
    if (owner.get() != this) throw std::invalid_argument("router BF16 inventory requires its actual owner");
    std::vector<UmaOwnerView> views;
    for (const auto& c : copies) {
        UmaOwnerView view;
        view.view_identity = next_uma_owner_view_identity();
        view.backing = {0x5157524246313631ULL, c.lease, 1U};
        view.category = "qwen_router_bf16";
        view.allocation_bytes = view.view_bytes = c.bytes;
        view.allocation_pin = owner;
        view.address_space = UmaAddressSpace::device;
        view.address = reinterpret_cast<std::uintptr_t>(c.device);
        views.push_back(std::move(view));
    }
    return build_uma_owner_inventory(views);
}
std::shared_ptr<const HipQwenRouterBf16> load_hip_qwen_router_bf16(const MappedModelFile& mapped,
    const ModelManifest& manifest, const ModelRuntimeBinding& binding, bool enabled) {
    auto owner = std::make_shared<HipQwenRouterBf16>();
    if (!enabled || !q6p::kernels_enabled()) return owner;
    if (binding.tensors.size() != manifest.tensors.size() || binding.manifest_layout_identity != manifest.layout_identity)
        throw std::invalid_argument("router BF16 binding differs from manifest");
    mapped.validate_unchanged();
    owner->copies.reserve(manifest.tensors.size());
    std::array<std::uint16_t, qwen_router_bf16_staging_bytes / 2U> staging;
    for (std::size_t i = 0; i < manifest.tensors.size(); ++i) {
        const auto& t = manifest.tensors[i];
        if (!qwen_router_bf16_candidate(t)) continue;
        if (t.absolute_offset > mapped.bytes() || t.byte_size > mapped.bytes() - t.absolute_offset)
            throw std::invalid_argument("router BF16 source exceeds mapping");
        const auto* runtime = find_runtime_tensor_binding(binding, static_cast<std::uint32_t>(i));
        if (!runtime || !runtime->address || runtime->bytes < t.byte_size)
            throw std::invalid_argument("router BF16 target binding is incomplete");
        const auto* original = reinterpret_cast<const void*>(runtime->address);
        const auto* source = reinterpret_cast<const std::byte*>(mapped.base_address() + t.absolute_offset);
        // Check the WHOLE immutable F32 tensor before allocation/registration.
        if (!router_bf16_exact({source, static_cast<std::size_t>(t.byte_size)})) { ++owner->skipped; continue; }
        if (const auto* existing = q6p::find(original)) {
            if (existing->kind != q6p::Kind::bf16 || existing->rows != t.dimensions[1] || existing->cols != t.dimensions[0])
                throw std::invalid_argument("router BF16 registry shape differs from target");
            ++owner->reused;
            continue;
        }
        void* device = nullptr;
        const auto bytes = t.byte_size / 2U;
        hip_check(hipMalloc(&device, bytes), "router BF16 allocation");
        owner->copies.push_back({original, device, bytes, next_uma_owner_view_identity()});
        for (std::uint64_t offset = 0; offset < bytes; ) {
            const auto count = std::min<std::uint64_t>(staging.size(), (bytes - offset) / 2U);
            for (std::uint64_t j = 0; j < count; ++j) {
                std::uint32_t bits;
                std::memcpy(&bits, source + offset * 2U + j * 4U, 4U);
                staging[j] = static_cast<std::uint16_t>(bits >> 16U);
            }
            hip_check(hipMemcpy(static_cast<unsigned char*>(device) + offset, staging.data(), count * 2U,
                hipMemcpyHostToDevice), "router BF16 upload");
            offset += count * 2U;
        }
        q6p::register_tensor(original, {static_cast<const unsigned char*>(device),
            static_cast<std::uint32_t>(t.dimensions[1]), static_cast<std::uint32_t>(t.dimensions[0]), q6p::Kind::bf16});
        owner->bytes += bytes;
    }
    mapped.validate_unchanged();
    return owner;
}
} // namespace strix
