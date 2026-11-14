#include "strix/hip_runtime_allocator.hpp"
#include "strix/hip_setup_quiescence.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace strix {
namespace {

// Pageable aligned host memory. Windows has no posix_memalign; its aligned pair must be used
// together (memory from _aligned_malloc is not valid for free). Callers pass checked_total's
// non-zero size and power-of-two alignment.
void* allocate_pageable(std::size_t alignment, std::size_t bytes) noexcept {
#ifdef _WIN32
    return _aligned_malloc(bytes, alignment);
#else
    void* pointer = nullptr;
    return posix_memalign(&pointer, alignment, bytes) == 0 ? pointer : nullptr;
#endif
}
void release_pageable(void* pointer) noexcept {
#ifdef _WIN32
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
}

std::atomic<std::uint64_t> next_allocation_lease{1};
std::uint64_t allocation_lease() {
    auto value=next_allocation_lease.load(std::memory_order_relaxed);
    for(;;) {
        if(!value || value==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("HIP allocation lease identity exhausted");
        if(next_allocation_lease.compare_exchange_weak(value,value+1U,std::memory_order_relaxed)) return value;
    }
}

void check_hip(
    hipError_t error,
    const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            ": " +
            hipGetErrorString(error));
    }
}

bool valid_alignment(
    std::uint64_t alignment) noexcept {
    return alignment != 0 &&
           (alignment &
            (alignment - 1ULL)) == 0;
}

}  // namespace


void initialize_hip_fresh_runtime_memory(
    const ExecutionMemoryBinding& scratch,
    const SessionStateMemoryBinding& state) {
    if (scratch.arena_base == 0 ||
        scratch.arena_bytes == 0 ||
        scratch.allocation_generation == 0) {
        throw std::invalid_argument(
            "fresh HIP runtime initialization requires bound scratch memory");
    }

    // Validate every host/device binding before the first potentially queued
    // operation, so a later schema error cannot unwind pending initialization.
    if ((state.device_bytes != 0 &&
         (state.device_base == 0 || state.device_allocation_generation == 0)) ||
        (state.host_bytes != 0 &&
         (state.host_base == 0 || state.host_allocation_generation == 0))) {
        throw std::invalid_argument("fresh HIP runtime initialization requires bound state");
    }
    if (scratch.arena_bytes > std::numeric_limits<std::size_t>::max() ||
        state.device_bytes > std::numeric_limits<std::size_t>::max() ||
        state.host_bytes > std::numeric_limits<std::size_t>::max() ||
        scratch.arena_bytes > std::numeric_limits<std::uintptr_t>::max() - scratch.arena_base ||
        state.device_bytes > std::numeric_limits<std::uintptr_t>::max() - state.device_base ||
        state.host_bytes > std::numeric_limits<std::uintptr_t>::max() - state.host_base) {
        throw std::overflow_error("fresh HIP runtime initialization size does not fit size_t");
    }
    const HipSetupQuiescenceGuard setup_guard;
    setup_guard.run([&] {
        check_hip(
            hipMemset(
                reinterpret_cast<void*>(scratch.arena_base),
                0,
                static_cast<std::size_t>(scratch.arena_bytes)),
            "hipMemset(fresh scratch)");

        if (state.device_bytes != 0) {
            check_hip(
                hipMemset(
                    reinterpret_cast<void*>(state.device_base),
                    0,
                    static_cast<std::size_t>(state.device_bytes)),
                "hipMemset(fresh device state)");
        }
        if (state.host_bytes != 0) {
            std::memset(
                reinterpret_cast<void*>(state.host_base),
                0,
                static_cast<std::size_t>(state.host_bytes));
        }
    });
}

HipRuntimeAllocator::HipRuntimeAllocator(
    HipHostAllocationMode host_mode)
    : host_mode_(host_mode) {}

HipRuntimeAllocator::~HipRuntimeAllocator() {
    release_all();
}

std::uint64_t HipRuntimeAllocator::checked_total(
    std::uint64_t bytes,
    std::uint64_t alignment) {
    if (bytes == 0 ||
        !valid_alignment(alignment)) {
        throw std::invalid_argument(
            "HIP runtime allocation requires non-zero bytes and power-of-two alignment");
    }
    const auto padding =
        alignment - 1ULL;
    if (padding >
        std::numeric_limits<std::uint64_t>::max() -
            bytes) {
        throw std::overflow_error(
            "HIP runtime allocation size overflow");
    }
    return bytes + padding;
}

std::uintptr_t HipRuntimeAllocator::align_address(
    std::uintptr_t address,
    std::uint64_t alignment) {
    const auto mask =
        static_cast<std::uintptr_t>(
            alignment - 1ULL);
    if (address >
        std::numeric_limits<std::uintptr_t>::max() -
            mask) {
        throw std::overflow_error(
            "HIP runtime aligned address overflow");
    }
    return (address + mask) & ~mask;
}

std::uintptr_t HipRuntimeAllocator::allocate_device(
    std::uint64_t bytes,
    std::uint64_t alignment) {
    const auto total =
        checked_total(bytes, alignment);
    const auto lease=allocation_lease();

    void* raw = nullptr;
    check_hip(
        hipMalloc(
            &raw,
            static_cast<std::size_t>(total)),
        "hipMalloc");

    try {
        const auto raw_address =
            reinterpret_cast<std::uintptr_t>(raw);
        const auto aligned =
            align_address(
                raw_address,
                alignment);

        std::lock_guard lock(mutex_);
        const auto [iterator, inserted] =
            device_backing_.emplace(
                aligned,
                BackingAllocation{
                    raw_address,
                    total, bytes, lease,
                });
        static_cast<void>(iterator);
        if (!inserted) {
            throw std::runtime_error(
                "duplicate aligned HIP device allocation address");
        }
        return aligned;
    } catch (...) {
        static_cast<void>(
            hipFree(raw));
        throw;
    }
}

std::uintptr_t HipRuntimeAllocator::allocate_host(
    std::uint64_t bytes,
    std::uint64_t alignment) {
    const auto total =
        checked_total(bytes, alignment);
    const auto lease=allocation_lease();

    if (host_mode_ ==
        HipHostAllocationMode::pageable_aligned) {
        const auto host_alignment =
            std::max<std::uint64_t>(
                alignment,
                sizeof(void*));
        void* pointer =
            allocate_pageable(
                static_cast<std::size_t>(
                    host_alignment),
                static_cast<std::size_t>(
                    bytes));
        if (pointer == nullptr) {
            throw std::bad_alloc();
        }

        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        try {
            std::lock_guard lock(mutex_);
            const auto [iterator, inserted] = pageable_host_.emplace(
                address,BackingAllocation{address,bytes,bytes,lease});
            static_cast<void>(iterator);
            if(!inserted) throw std::runtime_error("duplicate pageable host allocation address");
            return address;
        } catch(...) {release_pageable(pointer);throw;}
    }

    void* raw = nullptr;
    check_hip(
        hipHostMalloc(
            &raw,
            static_cast<std::size_t>(total),
            hipHostMallocMapped),
        "hipHostMalloc");

    try {
        const auto raw_address =
            reinterpret_cast<std::uintptr_t>(
                raw);
        const auto aligned =
            align_address(
                raw_address,
                alignment);

        std::lock_guard lock(mutex_);
        const auto [iterator, inserted] =
            pinned_host_backing_.emplace(
                aligned,
                BackingAllocation{
                    raw_address,
                    total, bytes, lease,
                });
        static_cast<void>(iterator);
        if (!inserted) {
            throw std::runtime_error(
                "duplicate aligned HIP host allocation address");
        }
        return aligned;
    } catch (...) {
        static_cast<void>(
            hipHostFree(raw));
        throw;
    }
}

HipRuntimeAllocationDescription HipRuntimeAllocator::describe_allocation(
    std::uintptr_t address,std::uint64_t bytes,HipRuntimeAllocationSpace space) const {
    if(!address || !bytes || (space!=HipRuntimeAllocationSpace::device && space!=HipRuntimeAllocationSpace::host))
        throw std::invalid_argument("HIP allocation inventory needs a live exact allocation and closed space");
    std::lock_guard lock(mutex_);
    const auto& records=space==HipRuntimeAllocationSpace::device?device_backing_:
        host_mode_==HipHostAllocationMode::pageable_aligned?pageable_host_:pinned_host_backing_;
    const auto found=records.find(address);
    if(found==records.end() || found->second.requested_bytes!=bytes)
        throw std::invalid_argument("HIP allocation inventory address/extent is not owned by this allocator");
    const auto& actual=found->second;
    if(!actual.lease_identity || !actual.raw_address || address<actual.raw_address ||
        actual.raw_bytes>std::numeric_limits<std::uintptr_t>::max()-actual.raw_address ||
        address-actual.raw_address>actual.raw_bytes || bytes>actual.raw_bytes-(address-actual.raw_address))
        throw std::invalid_argument("HIP allocation inventory backing is inconsistent or overflowing");
    return {actual.lease_identity,actual.raw_address,actual.raw_bytes,
        static_cast<std::uint64_t>(address-actual.raw_address),bytes};
}

std::uint64_t HipRuntimeAllocator::allocation_lease_identity(
    std::uintptr_t address,std::uint64_t bytes,bool host) const {
    return describe_allocation(address,bytes,host?HipRuntimeAllocationSpace::host:HipRuntimeAllocationSpace::device).lease_identity;
}

UmaOwnerInventory hip_runtime_memory_owner_inventory(
    std::shared_ptr<const RuntimeExecutionMemoryOwner> owner,UmaMemoryObservation observation) {
    if(!owner) throw std::invalid_argument("runtime inventory requires an actual shared memory owner");
    const auto self=owner->weak_from_this().lock();
    if(!self || self.get()!=owner.get() || owner.owner_before(self) || self.owner_before(owner))
        throw std::invalid_argument("runtime inventory requires its exact memory owner control block");
    const auto& allocator_pin=owner->retained_allocator();
    const auto* allocator=dynamic_cast<const HipRuntimeAllocator*>(allocator_pin.get());
    if(!allocator) throw std::invalid_argument("runtime inventory requires a retained actual HIP allocator");
    const auto allocator_self=allocator->weak_from_this().lock();
    if(!allocator_self || allocator_self.get()!=allocator ||
        allocator_pin.owner_before(allocator_self) || allocator_self.owner_before(allocator_pin))
        throw std::invalid_argument("runtime inventory requires its exact allocator control block");
    owner->validate_inventory_ownership();
    const auto& scratch=owner->scratch();const auto& state=owner->state();
    std::vector<UmaOwnerView> views;views.reserve(3U);
    const auto append=[&](std::uintptr_t address,std::uint64_t bytes,std::uint64_t generation,bool host,const char* category) {
        if(!bytes) return;
        const auto actual=allocator->describe_allocation(address,bytes,
            host?HipRuntimeAllocationSpace::host:HipRuntimeAllocationSpace::device);
        views.push_back({next_uma_owner_view_identity(),{HipRuntimeAllocator::inventory_issuer,actual.lease_identity,generation},
            UmaBackingKind::allocated_storage,category,actual.raw_bytes,actual.view_offset,actual.view_bytes,owner,
            host?UmaAddressSpace::host:UmaAddressSpace::device,address});
    };
    append(scratch.arena_base,scratch.arena_bytes,scratch.allocation_generation,false,"runtime.scratch");
    append(state.device_base,state.device_bytes,state.device_allocation_generation,false,"runtime.state.device");
    append(state.host_base,state.host_bytes,state.host_allocation_generation,true,"runtime.state.host");
    return build_uma_owner_inventory(views,std::move(observation));
}

void HipRuntimeAllocator::free_device(
    std::uintptr_t address,
    std::uint64_t) noexcept {
    std::uintptr_t raw = 0;
    {
        std::lock_guard lock(mutex_);
        const auto found =
            device_backing_.find(
                address);
        if (found == device_backing_.end()) {
            return;
        }
        raw = found->second.raw_address;
        device_backing_.erase(found);
    }
    static_cast<void>(
        hipFree(
            reinterpret_cast<void*>(raw)));
}

void HipRuntimeAllocator::free_host(
    std::uintptr_t address,
    std::uint64_t) noexcept {
    if (host_mode_ ==
        HipHostAllocationMode::pageable_aligned) {
        {
            std::lock_guard lock(mutex_);
            const auto found =
                pageable_host_.find(
                    address);
            if (found == pageable_host_.end()) {
                return;
            }
            pageable_host_.erase(found);
        }
        release_pageable(
            reinterpret_cast<void*>(
                address));
        return;
    }

    std::uintptr_t raw = 0;
    {
        std::lock_guard lock(mutex_);
        const auto found =
            pinned_host_backing_.find(
                address);
        if (found ==
            pinned_host_backing_.end()) {
            return;
        }
        raw = found->second.raw_address;
        pinned_host_backing_.erase(found);
    }
    static_cast<void>(
        hipHostFree(
            reinterpret_cast<void*>(raw)));
}

void HipRuntimeAllocator::release_all() noexcept {
    std::unordered_map<
        std::uintptr_t,
        BackingAllocation> devices;
    std::unordered_map<
        std::uintptr_t,
        BackingAllocation> pinned;
    std::unordered_map<
        std::uintptr_t,
        BackingAllocation> pageable;

    {
        std::lock_guard lock(mutex_);
        devices.swap(device_backing_);
        pinned.swap(pinned_host_backing_);
        pageable.swap(pageable_host_);
    }

    for (const auto& [address, allocation] :
         devices) {
        static_cast<void>(address);
        static_cast<void>(
            hipFree(
                reinterpret_cast<void*>(
                    allocation.raw_address)));
    }
    for (const auto& [address, allocation] :
         pinned) {
        static_cast<void>(address);
        static_cast<void>(
            hipHostFree(
                reinterpret_cast<void*>(
                    allocation.raw_address)));
    }
    for (const auto& [address, bytes] :
         pageable) {
        static_cast<void>(bytes);
        release_pageable(
            reinterpret_cast<void*>(
                address));
    }
}

}  // namespace strix
