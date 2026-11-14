#include "strix/hip_check.hpp"
#include "strix/hip_direct_gpu_pool_allocator.hpp"
#include "strix/hip_runtime_allocator.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix::transport {
namespace {
using strix::hip_check;
bool same_view(const TransportBufferView& a, const TransportBufferView& b) {
    return a.address == b.address && a.bytes == b.bytes && a.allocation_generation == b.allocation_generation &&
        a.domain == b.domain && a.access == b.access;
}
} // namespace
HipDirectGpuPoolCalls HipDirectGpuPoolCalls::native() {
    HipDirectGpuPoolCalls calls; calls.observations = HipDirectGpuExportCalls::native();
    calls.allocate_with_flags = hipExtMallocWithFlags; calls.free_allocation = hipFree; return calls;
}
struct HipDirectGpuPoolAllocator::Impl {
    struct Slot {
        void* raw = nullptr;
        std::uintptr_t aligned = 0;
        std::uint64_t requested = 0, charged = 0;
        DirectGpuExportAllocation allocation;
        TransportBufferView exported;
        bool identified = false, accounted = false, bound = false, free_complete = false;
    };
    DirectGpuExportDevice device;
    HipDirectGpuPoolPolicy policy;
    HipDirectGpuPoolCalls calls;
    HipDirectGpuExportIo probe;
    std::array<Slot, 2> slots;
    std::weak_ptr<void> active_use;
    bool failed = false;
    Impl(DirectGpuExportDevice d, HipDirectGpuPoolPolicy p, HipDirectGpuPoolCalls c)
        : device(d), policy(p), calls(std::move(c)), probe(calls.observations) {}
    void current() {
        try {
            if (probe.inspect_device() != device) throw std::runtime_error("uncached pool device/UUID/HIP versions changed");
        } catch (...) { failed = true; throw; }
    }
    std::uint64_t bytes() const noexcept {
        std::uint64_t total = 0;
        for (const auto& slot : slots) if (slot.raw || (slot.free_complete && slot.charged)) {
            if (slot.charged > std::numeric_limits<std::uint64_t>::max() - total) return std::numeric_limits<std::uint64_t>::max();
            total += slot.charged;
        }
        return total;
    }
    DirectGpuExportAllocation inspect(const Slot& slot) {
        TransportBufferView view{reinterpret_cast<std::uintptr_t>(slot.raw), 1, 1,
            TransportMemoryDomain::gpu_transport_pool, TransportBufferAccess::read_write};
        return probe.inspect_allocation(view);
    }
    bool release(Slot& slot) noexcept {
        if (!slot.raw && !slot.free_complete) return true;
        try {
            current();
            if (slot.free_complete) { slot.charged = 0; return true; }
            if (slot.identified && inspect(slot) != slot.allocation) throw std::runtime_error("uncached pool native allocation changed before free");
            current();
            const auto result = calls.free_allocation(slot.raw);
            if (result != hipSuccess) throw std::runtime_error("uncached pool native free unproved");
            // Never retry a successful native free against a potentially reused
            // address if the post-callback device check rejects publication.
            slot.raw = nullptr; slot.free_complete = true;
            current(); slot.charged = 0; return true;
        } catch (...) { failed = true; return false; }
    }
};
HipDirectGpuPoolAllocator::HipDirectGpuPoolAllocator(DirectGpuExportDevice device, HipDirectGpuPoolPolicy policy, HipDirectGpuPoolCalls calls)
    : impl_(std::make_unique<Impl>(device, policy, std::move(calls))) {
    if (!policy.experimental_opt_in || policy.maximum_gpu_bytes == 0 || policy.maximum_gpu_bytes > 128ULL * 1024ULL * 1024ULL ||
        policy.maximum_alignment == 0 || policy.maximum_alignment > 4096 ||
        policy.allocation_quantum == 0 || policy.allocation_quantum > policy.maximum_gpu_bytes ||
        (policy.allocation_quantum & (policy.allocation_quantum - 1)) != 0 || device.index < 0 ||
        device.runtime_version <= 0 || device.driver_version <= 0 ||
        !std::any_of(device.uuid.begin(), device.uuid.end(), [](auto byte) { return byte != 0; }) ||
        !impl_->calls.allocate_with_flags || !impl_->calls.free_allocation) {
        throw std::invalid_argument("uncached transport allocation requires explicit opt-in, local device and bounded native calls");
    }
    impl_->current();
}
HipDirectGpuPoolAllocator::~HipDirectGpuPoolAllocator() {
    if (drain() != DirectPoolReleaseStatus::released) static_cast<void>(impl_.release());
}
std::uintptr_t HipDirectGpuPoolAllocator::allocate_device(std::uint64_t bytes, std::uint64_t alignment) {
    if (impl_->failed) throw std::logic_error("uncached transport allocator is failed");
    // Request the operator-selected backing quantum without treating it as
    // proof of the native extent. Inspect and charge the actual extent below.
    const auto allocation_quantum = impl_->policy.allocation_quantum;
    const auto padded = HipRuntimeAllocator::checked_total(bytes, alignment);
    if (padded > std::numeric_limits<std::uint64_t>::max() - (allocation_quantum - 1))
        throw std::length_error("uncached transport allocation rounding overflows");
    const auto total = (padded + allocation_quantum - 1) & ~(allocation_quantum - 1);
    const auto retained = impl_->bytes();
    if (alignment > impl_->policy.maximum_alignment || total > std::numeric_limits<std::size_t>::max() ||
        retained > impl_->policy.maximum_gpu_bytes || total > impl_->policy.maximum_gpu_bytes - retained) throw std::length_error("uncached transport allocation exceeds admitted bytes/alignment");
    auto found = std::find_if(impl_->slots.begin(), impl_->slots.end(), [](const auto& slot) { return !slot.raw && !slot.charged; });
    if (found == impl_->slots.end()) throw std::length_error("uncached transport allocator owns at most two pool allocations");
    impl_->current(); *found = {}; found->requested = bytes; found->charged = total;
    try {
        // Slot adopts output before an error or exception can obscure partial
        // allocation success. This is policy requested, not cache qualification.
        hip_check(impl_->calls.allocate_with_flags(&found->raw, static_cast<std::size_t>(total), hipDeviceMallocUncached), "HIP uncached transport allocation");
        if (!found->raw) throw std::runtime_error("HIP uncached allocation returned no owning pointer");
        impl_->current(); found->allocation = impl_->inspect(*found); found->identified = true;
        const auto& native = found->allocation; const auto raw = reinterpret_cast<std::uintptr_t>(found->raw);
        if (native.base == raw && native.bytes != 0 && native.bytes <= std::numeric_limits<std::uintptr_t>::max() - native.base) {
            found->charged = native.bytes; found->accounted = true;
        }
        if (!native.device_memory || native.managed || native.device != impl_->device.index || native.base != raw || native.bytes < total ||
            native.bytes > std::numeric_limits<std::uintptr_t>::max() - native.base) throw std::runtime_error("uncached pool native extent/type/device is incompatible");
        if (native.bytes > impl_->policy.maximum_gpu_bytes - retained) throw std::length_error("uncached pool full native allocation exceeds admission");
        found->aligned = HipRuntimeAllocator::align_address(raw, alignment);
        if (found->aligned - raw > native.bytes || bytes > native.bytes - (found->aligned - raw)) throw std::runtime_error("uncached pool aligned view exceeds native allocation");
        for (const auto& other : impl_->slots) if (&other != &*found && other.raw && other.identified &&
            native.base < other.allocation.base + other.allocation.bytes && other.allocation.base < native.base + native.bytes) {
            throw std::runtime_error("uncached pool native allocations overlap");
        }
        impl_->current(); if (impl_->inspect(*found) != native) throw std::runtime_error("uncached pool native extent changed during allocation");
        impl_->current();
        return found->aligned;
    } catch (...) {
        impl_->failed = true;
        if (found->raw) { static_cast<void>(impl_->release(*found)); }
        else { found->charged = 0; }
        throw;
    }
}
std::uintptr_t HipDirectGpuPoolAllocator::allocate_host(std::uint64_t, std::uint64_t) { throw std::logic_error("uncached transport allocator is device-only"); }
void HipDirectGpuPoolAllocator::free_host(std::uintptr_t, std::uint64_t) noexcept { impl_->failed = true; }
void HipDirectGpuPoolAllocator::free_device(std::uintptr_t address, std::uint64_t bytes) noexcept {
    if (!impl_->active_use.expired()) { impl_->failed = true; return; }
    auto found = std::find_if(impl_->slots.begin(), impl_->slots.end(), [&](const auto& slot) { return slot.aligned == address && slot.requested == bytes; });
    if (found == impl_->slots.end() || !address) { impl_->failed = true; return; }
    static_cast<void>(impl_->release(*found));
}
void HipDirectGpuPoolAllocator::bind_owned_pool(const DirectTransportLinkPoolBinding& pool) {
    if (pool.tx.address == pool.rx.address) throw std::invalid_argument("uncached pool requires independent TX/RX bindings");
    std::array<Impl::Slot*, 2> matched{};
    std::size_t index = 0;
    for (const auto* view : {&pool.tx, &pool.rx}) {
        auto found = std::find_if(impl_->slots.begin(), impl_->slots.end(), [&](const auto& slot) { return slot.raw && slot.aligned == view->address && slot.requested == view->bytes; });
        if (found == impl_->slots.end() || !view->allocation_generation || view->domain != TransportMemoryDomain::gpu_transport_pool ||
            view->access != TransportBufferAccess::read_write || (found->bound && !same_view(found->exported, *view))) throw std::invalid_argument("uncached pool export binding is foreign");
        matched[index++] = &*found;
    }
    impl_->current();
    matched[0]->exported = pool.tx; matched[1]->exported = pool.rx;
    for (auto* slot : matched) slot->bound = true;
}
DirectPoolReleaseStatus HipDirectGpuPoolAllocator::release_owned_pool(const DirectTransportLinkPoolBinding& pool) const noexcept {
    if (!impl_->active_use.expired()) return DirectPoolReleaseStatus::pending;
    if (!pool.tx.address && !pool.rx.address) return impl_->bytes() == 0 ? DirectPoolReleaseStatus::released : DirectPoolReleaseStatus::failed_retained;
    for (const auto* view : {&pool.tx, &pool.rx}) {
        const auto found = std::find_if(impl_->slots.begin(), impl_->slots.end(), [&](const auto& slot) { return slot.bound && same_view(slot.exported, *view); });
        if (found == impl_->slots.end() || found->raw || !found->free_complete || found->charged) return DirectPoolReleaseStatus::failed_retained;
    }
    return DirectPoolReleaseStatus::released;
}
std::shared_ptr<DirectGpuPoolUseLease> HipDirectGpuPoolAllocator::retain_owned_pool(const DirectTransportLinkPoolBinding& pool) {
    if (impl_->failed || !impl_->active_use.expired()) throw std::logic_error("uncached pool is failed or already retained by an importer");
    for (const auto* view : {&pool.tx, &pool.rx}) {
        const auto found = std::find_if(impl_->slots.begin(), impl_->slots.end(), [&](const auto& slot) { return slot.bound && same_view(slot.exported, *view); });
        if (found == impl_->slots.end() || !found->raw || found->free_complete) throw std::invalid_argument("uncached import lease is foreign or retired");
    }
    impl_->current();
    class Guard final : public DirectGpuPoolUseLease {
    public:
        Guard(const Impl* implementation, DirectTransportLinkPoolBinding binding) : implementation_(implementation), binding_(binding) {}
        bool current() const noexcept override {
            if (implementation_->failed) return false;
            for (const auto* view : {&binding_.tx, &binding_.rx}) {
                const auto found = std::find_if(implementation_->slots.begin(), implementation_->slots.end(), [&](const auto& slot) { return slot.bound && same_view(slot.exported, *view); });
                if (found == implementation_->slots.end() || !found->raw || found->free_complete) return false;
            }
            return true;
        }
    private:
        const Impl* implementation_;
        DirectTransportLinkPoolBinding binding_;
    };
    auto guard = std::make_shared<Guard>(impl_.get(), pool); impl_->active_use = guard;
    return guard;
}
DirectPoolReleaseStatus HipDirectGpuPoolAllocator::drain() noexcept {
    if (!impl_->active_use.expired()) return DirectPoolReleaseStatus::pending;
    bool complete = true; for (auto& slot : impl_->slots) if (!impl_->release(slot)) complete = false;
    return complete ? DirectPoolReleaseStatus::released : DirectPoolReleaseStatus::failed_retained;
}
std::uint64_t HipDirectGpuPoolAllocator::retained_bytes() const noexcept { return impl_->bytes(); }
bool HipDirectGpuPoolAllocator::failed() const noexcept { return impl_->failed; }
bool HipDirectGpuPoolAllocator::accounting_complete() const noexcept {
    return std::none_of(impl_->slots.begin(), impl_->slots.end(), [](const auto& slot) { return slot.raw && !slot.accounted; });
}
} // namespace strix::transport
