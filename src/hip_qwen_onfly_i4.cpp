#include "strix/hip_check.hpp"
#include "strix/hip_qwen_onfly_i4.hpp"
#include "strix/qwen_onfly_i4.hpp"
#include "strix/q6p_registry.hpp"
#include "core/iu4_onfly.hpp"
#include "core/wmma_tiled.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace strix {
namespace {
using strix::hip_check;
unsigned group_index(unsigned group) { return group == 1U ? 0U : group == 2U ? 1U : group == 4U ? 2U : 3U; }
void retire_device_copy(void* device, int allocation_device) noexcept {
    // Snapshots retain the backing through submission. Final retirement fences all
    // streams before release, including launches racing owner destruction/replacement.
    int previous_device = 0;
    if (hipGetDevice(&previous_device) != hipSuccess || hipSetDevice(allocation_device) != hipSuccess) {
        std::fprintf(stderr, "onfly INT4: device selection failed; retaining device allocation\n");
        return;
    }
    if (hipDeviceSynchronize() == hipSuccess) (void)hipFree(device);
    else std::fprintf(stderr, "onfly INT4: completion fence failed; retaining device allocation\n");
    (void)hipSetDevice(previous_device);
}
}  // namespace
HipQwenOnflyI4::~HipQwenOnflyI4() {
    if (!copies.empty())
        std::fprintf(stderr, "turbo-weights onfly: INT4 kernel launches %llu, registry hits %llu\n",
                     static_cast<unsigned long long>(q6p::iu4::launches.load()), static_cast<unsigned long long>(q6p::iu4::hits()));
    for (const auto& c : copies) {
        q6p::iu4::unregister_tensor(c.original, static_cast<const unsigned char*>(c.device));
    }
}
UmaOwnerInventory HipQwenOnflyI4::owner_inventory(std::shared_ptr<const HipQwenOnflyI4> owner) const {
    if (owner.get() != this) throw std::invalid_argument("onfly INT4 inventory requires its actual owner");
    std::vector<UmaOwnerView> views;
    for (const auto& c : copies) {
        UmaOwnerView view;
        view.view_identity = next_uma_owner_view_identity();
        view.backing = {0x5157524f4e464c59ULL, c.lease, 1U};
        view.category = "qwen_onfly_i4";
        view.allocation_bytes = view.view_bytes = c.bytes;
        view.allocation_pin = owner;
        view.address_space = UmaAddressSpace::device;
        view.address = reinterpret_cast<std::uintptr_t>(c.device);
        views.push_back(std::move(view));
    }
    return build_uma_owner_inventory(views);
}
std::string HipQwenOnflyI4::receipt_json() const {
    auto identity = resolved.json();
    identity.pop_back();
    std::string out = "  \"qwen_onfly_i4\":" + identity + ",\"tensors\":" +
        std::to_string(copies.size()) + ",\"source_q8_bytes\":" + std::to_string(source_bytes) + ",\"copy_bytes\":" +
        std::to_string(copy_bytes) + ",\"skipped\":" + std::to_string(skipped) + ",\"convert_seconds\":" +
        std::to_string(convert_seconds) + ",\"source_resident\":true,\"groups\":{";
    for (unsigned i = 0; i < 4U; ++i)
        out += std::string(i ? "," : "") + "\"" + std::string(onfly_i4::group_name(1U << i)) + "\":{\"tensors\":" +
            std::to_string(group_tensors[i]) + ",\"bytes\":" + std::to_string(group_copy_bytes[i]) + "}";
    return out + "}},\n";
}
std::shared_ptr<const HipQwenOnflyI4> load_hip_qwen_onfly_i4(const ModelManifest& manifest, const ModelRuntimeBinding& binding,
                                                             const onfly_i4::Resolved& resolved) {
    auto owner = std::make_shared<HipQwenOnflyI4>(resolved);
    if (!resolved.active()) return owner;
    if (binding.tensors.size() != manifest.tensors.size() || binding.manifest_layout_identity != manifest.layout_identity)
        throw std::invalid_argument("onfly INT4 binding differs from manifest");
    const auto begin = std::chrono::steady_clock::now();
    int allocation_device = 0;
    hip_check(hipGetDevice(&allocation_device), "onfly INT4 allocation device");
    for (std::size_t i = 0; i < manifest.tensors.size(); ++i) {
        const auto& t = manifest.tensors[i];
        if (!qwen_onfly_i4_candidate(t, resolved)) continue;
        const auto* runtime = find_runtime_tensor_binding(binding, static_cast<std::uint32_t>(i));
        if (!runtime || !runtime->address || runtime->bytes < t.byte_size) throw std::invalid_argument("onfly INT4 source binding is incomplete");
        const auto* original = reinterpret_cast<const void*>(runtime->address);
        if (q6p::iu4::find(original)) { ++owner->skipped; continue; }
        const auto k = static_cast<std::uint32_t>(t.dimensions[0]), n = static_cast<std::uint32_t>(t.dimensions[1]);
        const auto padded = core::onfly::padded_outputs(n), bytes = core::onfly::copy_bytes(n, k);
        void* device = nullptr;
        hip_check(hipMalloc(&device, bytes), "onfly INT4 allocation");
        const std::shared_ptr<void> backing(device, [allocation_device](void* p) { retire_device_copy(p, allocation_device); });
        auto* base = static_cast<unsigned char*>(device);
        owner->copies.push_back({original, device, bytes, next_uma_owner_view_identity(), backing});
        hip_check(strix::core::convert_q8_0_iu4(static_cast<const std::uint8_t*>(original), static_cast<int>(k), static_cast<int>(n),
                                                base, reinterpret_cast<std::uint16_t*>(base + core::onfly::packed_bytes(padded, k)), nullptr,
                                                resolved.scale_rule), "onfly INT4 convert");
        hip_check(hipStreamSynchronize(nullptr), "onfly INT4 convert sync");
        q6p::iu4::register_tensor(original, {base, n, k, backing});
        owner->copy_bytes += bytes;
        owner->source_bytes += t.byte_size;
        const auto g = group_index(onfly_i4::classify(t.name));
        ++owner->group_tensors[g];
        owner->group_copy_bytes[g] += bytes;
    }
    owner->convert_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    return owner;
}
} // namespace strix
