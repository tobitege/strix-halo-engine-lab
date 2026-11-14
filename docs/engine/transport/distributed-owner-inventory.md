# Stage-owned storage inventory

The first-generation `HipDistributedStageResources` / `HipDistributedTokenParticipant`
owner inventory and its callable model fixtures were test-only and were removed
in R3-B1 (no production main reached them). The retained runtime seam is below.

Ordinary runtime memory has a separate retained-owner seam:
`RuntimeExecutionMemoryOwner::allocate(..., shared_allocator, generations)`
retains the actual allocator through pool release. The original overload with
an allocator reference keeps its borrowed lifetime contract. Moving an owner
transfers its allocator pin and original binding/lease snapshots. A live
inventory requires the owner to stay in place while its views are used.

`hip_runtime_memory_owner_inventory(actual_shared_memory_owner)` requires the
actual memory and HIP allocator control blocks. It compares all original
binding fields, generations and allocation leases before exposing the three
scratch/device-state/host-state pools. The allocator's optional lease hook
defaults to opaque; that allows retained lifetime without claiming inventory
evidence for custom allocators. Known HIP allocations report real reservation
padding. The returned views retain the memory owner and its allocator without
storing that result in either owner. The CLI can join these views with model
views through the same backing-key accounting; this is independent of D1 and
PLE owners.

Neither scope aggregates serving caches, separate ordinary D1 allocations,
all PLE/custom backends or external transport into a global inventory. Those
consumers need their own actual allocator/owner seams. Declared storage must
not be presented as complete process memory, RSS, PFNs or coherent DMA
capacity. Optional supplied memory observations stay separate from storage
reservations. All new fixtures remain source-only pending final validation.
