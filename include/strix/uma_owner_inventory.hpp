#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace strix {
// Process-local unique view token across all inventory producers. This token
// is not an allocation lease, address, generation or physical-page identity.
[[nodiscard]] std::uint64_t next_uma_owner_view_identity();
// Issuer-scoped allocation/backing lease, NOT a virtual address or file-content
// hash. All zero means unknown. A known key requires all three fields nonzero.
// A copy has a distinct key even if every byte equals the source. CPU mapping,
// registration and GPU views of one backing reuse its SAME key/generation.
struct UmaBackingIdentity {
    std::uint64_t issuer = 0, allocation = 0, generation = 0;
    bool operator==(const UmaBackingIdentity&) const = default;
};
enum class UmaBackingKind { allocated_storage, file_mapping };
enum class UmaAddressSpace { unspecified, host, device };
struct UmaOwnerView {
    std::uint64_t view_identity = 0;
    UmaBackingIdentity backing;
    UmaBackingKind kind = UmaBackingKind::allocated_storage;
    std::string category; // weights/state/scratch/repack/graph/PLE/staging, etc.
    std::uint64_t allocation_bytes = 0, view_offset = 0, view_bytes = 0;
    // Caller supplies the ACTUAL allocator/mapping lease, kept by the snapshot.
    // An opaque pin cannot independently prove physical page residency.
    std::shared_ptr<const void> allocation_pin;
    // Actual API-returned view address, when supplied. Distinct CPU/GPU
    // addresses may name one backing; addresses never establish backing keys.
    UmaAddressSpace address_space = UmaAddressSpace::unspecified;
    std::uintptr_t address = 0;
};
struct UmaMemoryObservation {
    // One coherent observation supplied by caller, never discovered or guessed
    // from a device name. Zero observation id means every field below is unknown.
    std::uint64_t observation_identity = 0;
    std::optional<std::uint64_t> mem_total_bytes, mem_available_bytes;
    std::optional<std::uint64_t> process_rss_bytes, gtt_reported_bytes, process_peak_bytes;
};
struct UmaCategoryViews { std::string category; std::uint64_t logical_view_bytes = 0; };
struct UmaBackingSummary {
    UmaBackingIdentity identity;
    UmaBackingKind kind = UmaBackingKind::allocated_storage;
    std::uint64_t allocation_bytes = 0, covered_view_bytes = 0, alias_view_bytes = 0;
    std::uint64_t views = 0;
};
struct UmaOwnerInventory {
    std::vector<UmaOwnerView> pinned_views;
    std::vector<UmaBackingSummary> known_backings;
    std::vector<UmaCategoryViews> logical_categories;
    UmaMemoryObservation observation;
    std::uint64_t logical_view_bytes = 0, unknown_backing_views = 0;
    std::uint64_t known_allocated_storage_bytes = 0, known_file_mapping_bytes = 0;
    std::uint64_t known_covered_view_bytes = 0, known_alias_view_bytes = 0;
    // Null if ANY backing is unknown. These are declared retained virtual/storage
    // extents, NOT measured physical/resident bytes or free UMA capacity.
    std::optional<std::uint64_t> unique_backing_reserved_bytes;
    // Known allocations counted once, each unknown view's allocation counted
    // separately. Conservative upper bound on declared reservations only.
    std::uint64_t reservation_upper_bound_bytes = 0;
};
// Read-only host accounting. Deduplicates only explicitly equal backing leases;
// rejects partial keys, duplicate view IDs, inconsistent same-key extents,
// out-of-range slices and integer overflow. Does not allocate GPU memory, admit
// an execution or add RSS+GTT. Caller must attest that each generation/pin is
// actually live; an opaque shared_ptr is retention, not a hardware lease oracle.
// Existing MemoryPlan/decide_admission remain separate.
[[nodiscard]] UmaOwnerInventory build_uma_owner_inventory(
    std::span<const UmaOwnerView>, UmaMemoryObservation observation = {});

struct UmaReservationLimits {
    std::optional<std::uint64_t> retained_bytes, setup_peak_bytes;
};
struct UmaReservationPlan {
    UmaOwnerInventory retained;
    UmaOwnerInventory setup_peak;
};
struct UmaReservationEstimate {
    std::uint64_t retained_reserved_bytes = 0, setup_peak_reserved_bytes = 0;
    // File-mapping bytes of the live inventory admitted as reclaimable (already subtracted above); 0 by default.
    std::uint64_t reclaimable_file_mapping_credit_bytes = 0;
};
// Pre-allocation growth estimate: new copies/repack storage are additional
// backings; aliases contribute zero here. setup_temporary_bytes coexist with
// both source and new retained copies. No fake pins for future allocations.
[[nodiscard]] UmaReservationEstimate plan_uma_reservation_growth(
    const UmaOwnerInventory& live, std::uint64_t additional_new_retained_bytes,
    std::uint64_t setup_temporary_bytes, UmaReservationLimits limits = {});
// Same, with an explicit credit for reclaimable file-mapping pages of the live inventory:
// clean read-only unregistered cache, or pages the caller drops and re-reads
// (STRIX_PLE_RECLAIMABLE). retained = live + growth - credit. The credit cannot exceed the live inventory's known
// file-mapping bytes; the caller owns the physical bound it encodes.
[[nodiscard]] UmaReservationEstimate plan_uma_reservation_growth(
    const UmaOwnerInventory& live, std::uint64_t additional_new_retained_bytes,
    std::uint64_t setup_temporary_bytes, UmaReservationLimits limits, std::uint64_t reclaimable_file_mapping_credit_bytes);
// Setup-only views are live concurrently with retained views, e.g. repack
// staging. A retained repack copy has its own backing, while registered aliases
// keep the source key. Limits use the conservative declared reservation bound,
// never RSS/GTT or an asserted physical-page count. Both snapshots retain pins.
[[nodiscard]] UmaReservationPlan build_uma_reservation_plan(
    std::span<const UmaOwnerView> retained, std::span<const UmaOwnerView> setup_only,
    UmaReservationLimits limits = {}, UmaMemoryObservation observation = {});
} // namespace strix
