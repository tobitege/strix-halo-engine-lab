#include "strix/hip_check.hpp"
#include "strix/hip_direct_gpu_stream_gpu.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace strix::transport {
namespace {
static_assert(std::endian::native == std::endian::little, "native terminal record requires little-endian host/GPU ABI");
static_assert(offsetof(GpuCommitRecord, sequence) == 16 && offsetof(GpuCommitRecord, payload_bytes) == 24);
using strix::hip_check;
bool same_view(const TransportBufferView& a, const TransportBufferView& b) {
    return a.address == b.address && a.bytes == b.bytes && a.allocation_generation == b.allocation_generation && a.domain == b.domain && a.access == b.access;
}
bool same_descriptor(const TransferDescriptor& a, const TransferDescriptor& b) {
    return a.protocol_version == b.protocol_version && a.connection_epoch == b.connection_epoch && a.model_plan_id == b.model_plan_id &&
        a.request_id == b.request_id && a.generation == b.generation && a.boundary_id == b.boundary_id && a.sequence == b.sequence &&
        a.shape_fingerprint == b.shape_fingerprint && a.dtype_id == b.dtype_id && a.byte_length == b.byte_length;
}
bool same_operation(const DirectDriverOperation& a, const DirectDriverOperation& b) {
    return a.imported.device == b.imported.device && a.imported.import_generation == b.imported.import_generation &&
        a.imported.lease.physical_link_id == b.imported.lease.physical_link_id && a.imported.lease.connection_epoch == b.imported.lease.connection_epoch &&
        a.imported.lease.pool_binding_identity == b.imported.lease.pool_binding_identity && a.imported.lease.link_index == b.imported.lease.link_index &&
        a.imported.lease.lease_generation == b.imported.lease.lease_generation && same_view(a.imported.pool.tx, b.imported.pool.tx) &&
        same_view(a.imported.pool.rx, b.imported.pool.rx) && a.handle == b.handle && a.receive_path == b.receive_path &&
        same_descriptor(a.descriptor, b.descriptor) && same_view(a.buffer, b.buffer);
}
} // namespace
HipDirectGpuStreamGpuCalls HipDirectGpuStreamGpuCalls::native() {
    HipDirectGpuStreamGpuCalls calls; calls.observations = HipDirectGpuExportCalls::native();
    calls.create_stream = hipStreamCreateWithFlags; calls.query_stream = hipStreamQuery; calls.destroy_stream = hipStreamDestroy;
    calls.stream_device = hipStreamGetDevice; calls.stream_id = hipStreamGetId; calls.stream_capture = hipStreamIsCapturing;
    calls.allocate_host = [](void** pointer, std::size_t bytes, unsigned flags) { return hipHostMalloc(pointer, bytes, flags); }; calls.free_host = hipHostFree;
    calls.create_event = hipEventCreateWithFlags; calls.record_event = hipEventRecord; calls.query_event = hipEventQuery;
    calls.destroy_event = hipEventDestroy; calls.copy = hipMemcpyAsync; return calls;
}
struct HipDirectGpuStreamGpu::Impl {
    struct Event { hipEvent_t handle = nullptr; bool recorded = false, complete = false, destroy_unknown = false; };
    struct Operation {
        DirectDriverOperation binding; std::array<Event, 3> events; bool released = false, acquired = false, unpacked = false, consumed = false;
        Operation() {} // Explicit construction is also accepted by HIP traits.
    };
    std::shared_ptr<DirectGpuExportImportLease> import_lease;
    DirectGpuExportBinding binding;
    HipDirectGpuStreamExecutionOwner execution;
    HipDirectGpuStreamGpuPolicy policy;
    HipDirectGpuStreamGpuCalls calls;
    HipDirectGpuExportIo probe;
    std::vector<DirectGpuExportAllocation> execution_extents;
    hipStream_t stream = nullptr;
    unsigned long long stream_identity = 0;
    std::array<void*, 2> host{};
    std::array<bool, 2> host_free_unknown{};
    std::array<std::optional<Operation>, 2> operations;
    std::optional<DirectDriverImportBinding> imported;
    std::optional<std::pair<TransferDescriptor, TransportBufferView>> staged;
    std::optional<std::chrono::steady_clock::time_point> last_progress_time;
    bool failed = false, stopping = false, closed = false, stream_destroy_unknown = false;
    Impl(std::shared_ptr<DirectGpuExportImportLease> lease, HipDirectGpuStreamExecutionOwner e, HipDirectGpuStreamGpuPolicy p, HipDirectGpuStreamGpuCalls c)
        : import_lease(std::move(lease)), binding(import_lease ? import_lease->binding() : DirectGpuExportBinding{}), execution(std::move(e)), policy(p), calls(std::move(c)), probe(calls.observations) {}
    void current(bool productive = true) {
        if (productive && (failed || stopping)) throw std::logic_error("GPU transport stream is failed or stopping");
        if (productive) {
            if (!import_lease) throw std::logic_error("GPU transport requires an active guarded import lease");
            import_lease->require_current();
        }
        if (probe.inspect_device() != binding.device) throw std::runtime_error("GPU transport device/UUID/HIP versions changed");
        const auto observed = execution.current_buffers();
        if (observed.size() != execution.buffers.size()) throw std::runtime_error("GPU transport execution owners changed");
        for (std::size_t i = 0; i < observed.size(); ++i) {
            if (!same_view(observed[i], execution.buffers[i]) || (execution_extents.size() == observed.size() && probe.inspect_allocation(observed[i]) != execution_extents[i])) {
                throw std::runtime_error("GPU transport execution allocation or generation changed");
            }
        }
        for (std::size_t i = 0; i < 2; ++i) {
            if (probe.inspect_allocation(i ? binding.pool.rx : binding.pool.tx) != binding.allocations[i] ||
                probe.inspect_descriptor(binding.descriptors[i]) != binding.descriptor_identities[i]) throw std::runtime_error("GPU transport exported pool/FD changed");
        }
        if (stream) {
            hipDevice_t device = -1; unsigned long long identity = 0; hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
            hip_check(calls.stream_device(stream, &device), "HIP transport stream device"); hip_check(calls.stream_id(stream, &identity), "HIP transport stream identity");
            hip_check(calls.stream_capture(stream, &capture), "HIP transport capture state");
            if (device != binding.device.index || !identity || (stream_identity && identity != stream_identity) || capture != hipStreamCaptureStatusNone) throw std::runtime_error("GPU transport stream replaced or captured");
            if (!stream_identity) stream_identity = identity;
        }
        if (probe.inspect_device() != binding.device) throw std::runtime_error("GPU transport device changed during inspection");
        const auto final_buffers = execution.current_buffers();
        if (final_buffers.size() != execution.buffers.size() || !std::equal(final_buffers.begin(), final_buffers.end(), execution.buffers.begin(), same_view)) throw std::runtime_error("GPU transport execution owner changed during inspection");
        if (productive) import_lease->require_current();
    }
    Operation& require(const DirectDriverOperation& operation) {
        current();
        if (!import_lease) throw std::logic_error("GPU transport operation requires an active guarded import");
        import_lease->require_current(); auto& entry = operations[operation.receive_path ? 1U : 0U];
        if (!entry || !same_operation(entry->binding, operation)) throw std::invalid_argument("GPU transport operation is foreign or stale");
        return *entry;
    }
    void segments(std::span<const TransportBufferView> views, std::uint64_t bytes, bool writing) {
        if (views.empty() || views.size() > 64) throw std::invalid_argument("HIP transport requires bounded registered segments");
        std::uint64_t covered = 0;
        for (const auto& view : views) {
            if (!view.bytes || view.bytes > bytes - covered || view.domain != TransportMemoryDomain::gpu_device ||
                (view.access != TransportBufferAccess::read_only && view.access != TransportBufferAccess::write_only && view.access != TransportBufferAccess::read_write) ||
                (writing && view.access == TransportBufferAccess::read_only) || (!writing && view.access == TransportBufferAccess::write_only)) throw std::invalid_argument("HIP transport segment bytes/access is invalid");
            const auto owner_view = std::find_if(execution.buffers.begin(), execution.buffers.end(), [&](const auto& candidate) {
                return view.allocation_generation == candidate.allocation_generation && view.domain == candidate.domain && view.address >= candidate.address &&
                    view.address - candidate.address <= candidate.bytes && view.bytes <= candidate.bytes - (view.address - candidate.address) &&
                    (writing ? candidate.access != TransportBufferAccess::read_only : candidate.access != TransportBufferAccess::write_only);
            });
            if (owner_view == execution.buffers.end()) throw std::invalid_argument("HIP transport segment is outside retained registered owner");
            covered += view.bytes;
        }
        if (covered != bytes) throw std::invalid_argument("HIP transport segments do not exactly cover descriptor");
        if (writing) for (std::size_t a = 0; a < views.size(); ++a) for (std::size_t b = a + 1; b < views.size(); ++b) {
            if (views[a].address < views[b].address + views[b].bytes && views[b].address < views[a].address + views[a].bytes) throw std::invalid_argument("HIP transport destination segments overlap");
        }
    }
    void terminal(const DirectDriverOperation& operation, const TransportBufferView& view) {
        const auto& pool = operation.receive_path ? binding.pool.rx : binding.pool.tx;
        if (pool.bytes < 4096 || view.address != pool.address + pool.bytes - 4096 || view.bytes != sizeof(GpuCommitRecord) ||
            view.allocation_generation != pool.allocation_generation || view.domain != pool.domain || view.access != TransportBufferAccess::read_write) {
            throw std::invalid_argument("GPU terminal region does not match final owned pool frame");
        }
    }
    std::chrono::steady_clock::time_point monotonic_now() {
        const auto value = calls.now();
        if (last_progress_time && value < *last_progress_time) throw std::runtime_error("HIP transport completion clock regressed; ownership retained");
        last_progress_time = value; return value;
    }
    void wait(Event& event, std::chrono::steady_clock::time_point start) {
        if (!event.recorded) throw std::logic_error("unrecorded HIP event cannot prove queued work complete");
        const auto duration = std::chrono::duration_cast<std::chrono::steady_clock::duration>(policy.completion_timeout);
        if (start > std::chrono::steady_clock::time_point::max() - duration) throw std::overflow_error("HIP transport completion deadline overflow; ownership retained");
        const auto deadline = start + duration;
        const auto require_before_deadline = [&] {
            if (monotonic_now() >= deadline) throw std::runtime_error("HIP transport completion deadline expired; ownership retained");
        };
        for (std::uint64_t count = 0; count < policy.maximum_completion_polls; ++count) {
            require_before_deadline(); current(); const auto result = calls.query_event(event.handle); current(); require_before_deadline();
            if (result == hipSuccess) { event.complete = true; return; }
            if (result != hipErrorNotReady) hip_check(result, "HIP transport event completion");
            if (count + 1 < policy.maximum_completion_polls) { calls.pause(policy.completion_pause); require_before_deadline(); }
        }
        throw std::runtime_error("HIP transport completion budget exhausted; ownership retained");
    }
    void record(Event& event) {
        const auto start = monotonic_now();
        hip_check(calls.record_event(event.handle, stream), "HIP transport system-fence event record"); event.recorded = true;
        current(); wait(event, start);
    }
    void destroy(Event& event) {
        if (!event.handle) return;
        if (event.destroy_unknown) throw std::runtime_error("HIP transport event destruction previously unproved");
        event.destroy_unknown = true;
        hip_check(calls.destroy_event(event.handle), "HIP transport event destroy"); event.handle = nullptr; event.destroy_unknown = false;
        current(false);
    }
};
HipDirectGpuStreamGpu::HipDirectGpuStreamGpu(std::shared_ptr<DirectGpuExportImportLease> lease, HipDirectGpuStreamExecutionOwner execution,
    HipDirectGpuStreamGpuPolicy policy, HipDirectGpuStreamGpuCalls calls)
    : impl_(std::make_unique<Impl>(std::move(lease), std::move(execution), policy, std::move(calls))) {
    auto& i = *impl_;
    try {
        if (!i.import_lease || !policy.experimental_opt_in || !policy.maximum_completion_polls || policy.maximum_completion_polls > 1000000 || !policy.maximum_execution_retained_bytes ||
            policy.completion_timeout.count() <= 0 || policy.completion_timeout > std::chrono::minutes{1} || policy.completion_pause.count() <= 0 || policy.completion_pause > std::chrono::seconds{1} || policy.completion_pause > policy.completion_timeout ||
            !i.execution.lifetime || i.execution.buffers.empty() || i.execution.buffers.size() > 64 || !i.execution.current_buffers ||
            !i.calls.create_stream || !i.calls.query_stream || !i.calls.destroy_stream || !i.calls.stream_device || !i.calls.stream_id || !i.calls.stream_capture ||
            !i.calls.allocate_host || !i.calls.free_host || !i.calls.create_event || !i.calls.record_event || !i.calls.query_event || !i.calls.destroy_event || !i.calls.copy || !i.calls.now || !i.calls.pause) {
            throw std::invalid_argument("HIP transport requires explicit opt-in, complete retained execution owner and bounded native calls");
        }
        i.import_lease->require_current(); i.current();
        for (const auto& pool : {i.binding.pool.tx, i.binding.pool.rx}) {
            if (pool.bytes < 8192 || pool.bytes % 4096 || pool.address % 4096) throw std::invalid_argument("HIP transport requires frame-aligned payload and terminal pool capacity");
        }
        std::uint64_t retained = 0;
        for (const auto& view : i.execution.buffers) {
            if (view.domain != TransportMemoryDomain::gpu_device || !view.address || !view.bytes || !view.allocation_generation ||
                (view.access != TransportBufferAccess::read_only && view.access != TransportBufferAccess::write_only && view.access != TransportBufferAccess::read_write) ||
                view.bytes > std::numeric_limits<std::uintptr_t>::max() - view.address) throw std::invalid_argument("HIP transport execution owner must bind resident GPU buffers");
            const auto allocation = i.probe.inspect_allocation(view);
            if (!allocation.device_memory || allocation.managed || allocation.device != i.binding.device.index || !allocation.base || !allocation.bytes ||
                allocation.bytes > std::numeric_limits<std::uintptr_t>::max() - allocation.base || view.address < allocation.base ||
                view.address - allocation.base > allocation.bytes || view.bytes > allocation.bytes - (view.address - allocation.base)) throw std::invalid_argument("HIP transport execution view exceeds native owner");
            for (const auto& pool : i.binding.allocations) if (allocation.base < pool.base + pool.bytes && pool.base < allocation.base + allocation.bytes) throw std::invalid_argument("HIP transport execution owner aliases an imported allocation");
            if (std::find(i.execution_extents.begin(), i.execution_extents.end(), allocation) == i.execution_extents.end()) {
                if (allocation.bytes > policy.maximum_execution_retained_bytes - retained) throw std::length_error("HIP transport retained execution capacity exceeds admission");
                retained += allocation.bytes;
            }
            i.execution_extents.push_back(allocation);
        }
        const auto destination = std::find_if(i.execution.buffers.begin(), i.execution.buffers.end(), [&](const auto& view) { return same_view(view, i.execution.receive_destination); });
        if (destination == i.execution.buffers.end() || destination->access != TransportBufferAccess::read_write) throw std::invalid_argument("HIP transport acquire destination is not a complete readable/writable registered owner");
        const auto destination_extent = i.execution_extents[static_cast<std::size_t>(destination - i.execution.buffers.begin())];
        for (const auto& pool : i.binding.allocations) if (destination_extent.base < pool.base + pool.bytes && pool.base < destination_extent.base + destination_extent.bytes) {
            throw std::invalid_argument("HIP transport acquire destination aliases an imported allocation");
        }
        hip_check(i.calls.create_stream(&i.stream, hipStreamNonBlocking), "HIP transport owned stream create");
        if (!i.stream || i.stream == hipStreamLegacy || i.stream == hipStreamPerThread) throw std::runtime_error("HIP transport requires distinct owned stream");
        hip_check(i.calls.stream_id(i.stream, &i.stream_identity), "HIP transport owned stream identity");
        if (!i.stream_identity) throw std::runtime_error("HIP transport stream has no stable identity");
        i.current();
        for (auto& host : i.host) { hip_check(i.calls.allocate_host(&host, sizeof(GpuCommitRecord), hipHostMallocDefault), "HIP transport pinned metadata allocation"); if (!host) throw std::runtime_error("HIP transport metadata allocation missing"); }
        i.current();
    } catch (...) {
        i.failed = true;
        if (close() != DirectPoolReleaseStatus::released) static_cast<void>(impl_.release());
        throw;
    }
}
HipDirectGpuStreamGpu::~HipDirectGpuStreamGpu() { if (close() != DirectPoolReleaseStatus::released) static_cast<void>(impl_.release()); }
hipStream_t HipDirectGpuStreamGpu::stream() const noexcept { return impl_->import_lease && impl_->imported && !impl_->stopping && !impl_->failed ? impl_->stream : nullptr; }
const DirectGpuExportBinding& HipDirectGpuStreamGpu::exported_binding() const noexcept { return impl_->binding; }
bool HipDirectGpuStreamGpu::failed() const noexcept { return impl_->failed; }
void HipDirectGpuStreamGpu::require_current() { try { impl_->current(); } catch (...) { impl_->failed = true; throw; } }
std::shared_ptr<DirectGpuExportImportLease> HipDirectGpuStreamGpu::import_lease() const noexcept { return impl_->import_lease; }
void HipDirectGpuStreamGpu::bind_import(const DirectDriverImportBinding& imported) {
    require_current(); auto& i = *impl_;
    if (i.imported || !imported.import_generation || imported.lease.pool_binding_identity != i.binding.pool_binding_identity ||
        !imported.lease.connection_epoch || imported.lease.connection_epoch != imported.device.interface.connection_epoch ||
        imported.lease.physical_link_id != imported.device.interface.physical_link_id || !same_view(imported.pool.tx, i.binding.pool.tx) || !same_view(imported.pool.rx, i.binding.pool.rx)) throw std::invalid_argument("HIP transport connection import is foreign");
    i.imported = imported;
}
void HipDirectGpuStreamGpu::stage_outbound_segments(const TransferDescriptor& descriptor, std::span<const TransportBufferView> sources, const TransportBufferView& tx) {
    try {
        auto& i = *impl_; i.current();
        if (!i.imported || i.operations[0] || i.staged || !validate_descriptor(descriptor, {1, i.imported->lease.connection_epoch, std::numeric_limits<std::uint32_t>::max()}) ||
            !descriptor.sequence || !descriptor.generation || tx.address < i.binding.pool.tx.address || (tx.address - i.binding.pool.tx.address) % 4096 || tx.address - i.binding.pool.tx.address > i.binding.pool.tx.bytes - 4096 ||
            descriptor.byte_length > i.binding.pool.tx.bytes - 4096 - (tx.address - i.binding.pool.tx.address) || tx.bytes != descriptor.byte_length ||
            tx.allocation_generation != i.binding.pool.tx.allocation_generation || tx.domain != i.binding.pool.tx.domain || tx.access != TransportBufferAccess::read_write) throw std::invalid_argument("HIP transport TX stage lacks exact exclusive physical binding");
        i.segments(sources, descriptor.byte_length, false); i.staged = std::make_pair(descriptor, tx);
        std::uintptr_t destination = tx.address;
        for (const auto& source : sources) {
            hip_check(i.calls.copy(reinterpret_cast<void*>(destination), reinterpret_cast<const void*>(source.address), static_cast<std::size_t>(source.bytes), hipMemcpyDeviceToDevice, i.stream), "HIP transport owned outbound pack");
            i.current(); destination += source.bytes;
        }
    } catch (...) { impl_->failed = true; throw; }
}
void HipDirectGpuStreamGpu::prepare(const DirectDriverOperation& operation) {
    require_current(); auto& i = *impl_; auto& entry = i.operations[operation.receive_path ? 1U : 0U];
    if (!i.import_lease) throw std::logic_error("HIP transport preparation requires active guarded import");
    i.import_lease->require_current();
    if (!i.imported) throw std::logic_error("HIP transport preparation requires an actual bound connection import");
    DirectDriverOperation expected = operation; expected.imported = *i.imported;
    if (!same_operation(expected, operation) || entry || !operation.handle || !operation.imported.import_generation || operation.imported.lease.pool_binding_identity != i.binding.pool_binding_identity ||
        !same_view(operation.imported.pool.tx, i.binding.pool.tx) || !same_view(operation.imported.pool.rx, i.binding.pool.rx)) throw std::invalid_argument("HIP transport preparation lacks exclusive exact import");
    const auto& pool = operation.receive_path ? i.binding.pool.rx : i.binding.pool.tx;
    if (!validate_descriptor(operation.descriptor, {1, i.imported->lease.connection_epoch, std::numeric_limits<std::uint32_t>::max()}) ||
        !operation.descriptor.sequence || !operation.descriptor.generation || operation.buffer.address < pool.address || (operation.buffer.address - pool.address) % 4096 ||
        operation.buffer.address - pool.address > pool.bytes - 4096 || operation.descriptor.byte_length > pool.bytes - 4096 - (operation.buffer.address - pool.address) ||
        operation.buffer.bytes < operation.descriptor.byte_length || operation.buffer.allocation_generation != pool.allocation_generation ||
        operation.buffer.domain != pool.domain || operation.buffer.access != TransportBufferAccess::read_write) throw std::invalid_argument("HIP transport operation payload exceeds exact owned data-frame window");
    if (!operation.receive_path && (!i.staged || !same_descriptor(i.staged->first, operation.descriptor) || !same_view(i.staged->second, operation.buffer))) throw std::invalid_argument("HIP transport producer has no exact owned staged source");
    entry.emplace(); entry->binding = operation;
    try {
        for (auto& event : entry->events) { hip_check(i.calls.create_event(&event.handle, hipEventDisableTiming), "HIP transport system-fence event create"); if (!event.handle) throw std::runtime_error("HIP transport event missing"); i.current(); }
    } catch (...) { i.failed = true; throw; }
}
DirectDriverVisibilityProof HipDirectGpuStreamGpu::publish_terminal(const DirectDriverOperation& operation, const TransportBufferView& terminal, const GpuCommitRecord& record) {
    try {
        auto& i = *impl_; auto& entry = i.require(operation); i.terminal(operation, terminal);
        if (operation.receive_path || entry.released || !validate_gpu_commit_record(record, operation.descriptor)) throw std::invalid_argument("HIP transport producer commit is foreign or duplicated");
        std::memcpy(i.host[0], &record, sizeof(record));
        hip_check(i.calls.copy(reinterpret_cast<void*>(terminal.address), i.host[0], sizeof(record), hipMemcpyHostToDevice, i.stream), "HIP transport terminal publish copy");
        i.current(); i.record(entry.events[0]); entry.released = true;
        return {operation, DirectDriverVisibility::producer_release_to_external_dma};
    } catch (...) { impl_->failed = true; throw; }
}
DirectDriverCommitRead HipDirectGpuStreamGpu::read_terminal(const DirectDriverOperation& operation, const TransportBufferView& terminal) {
    try {
        auto& i = *impl_; auto& entry = i.require(operation); i.terminal(operation, terminal);
        if (!operation.receive_path || entry.acquired) throw std::invalid_argument("HIP transport acquire is foreign or duplicated");
        if (operation.descriptor.byte_length > i.execution.receive_destination.bytes || operation.buffer.bytes < operation.descriptor.byte_length) throw std::length_error("HIP transport payload exceeds owned acquire destination");
        hip_check(i.calls.copy(reinterpret_cast<void*>(i.execution.receive_destination.address), reinterpret_cast<const void*>(operation.buffer.address),
            static_cast<std::size_t>(operation.descriptor.byte_length), hipMemcpyDeviceToDevice, i.stream), "HIP transport full payload acquire copy");
        i.current();
        hip_check(i.calls.copy(i.host[1], reinterpret_cast<const void*>(terminal.address), sizeof(GpuCommitRecord), hipMemcpyDeviceToHost, i.stream), "HIP transport terminal acquire copy");
        i.current(); i.record(entry.events[1]); GpuCommitRecord record; std::memcpy(&record, i.host[1], sizeof(record));
        if (!validate_gpu_commit_record(record, operation.descriptor)) throw std::runtime_error("HIP transport acquired terminal record is invalid");
        entry.acquired = true; i.current(); return {record, {operation, DirectDriverVisibility::consumer_acquire_from_external_dma}};
    } catch (...) { impl_->failed = true; throw; }
}
TransportBufferView HipDirectGpuStreamGpu::acquired_receive_view(const DirectDriverOperation& operation) {
    auto& entry = impl_->require(operation);
    if (!operation.receive_path || !entry.acquired) throw std::logic_error("HIP transport receive payload has not acquired completion");
    auto view = impl_->execution.receive_destination; view.bytes = operation.descriptor.byte_length; return view;
}
void HipDirectGpuStreamGpu::consume_received_segments(const DirectDriverOperation& operation, std::span<const TransportBufferView> destinations) {
    try {
        auto& i = *impl_; auto& entry = i.require(operation);
        if (!operation.receive_path || !entry.acquired || entry.unpacked || entry.consumed) throw std::logic_error("HIP transport receive unpack requires one acquired payload");
        i.segments(destinations, operation.descriptor.byte_length, true);
        const auto acquired = acquired_receive_view(operation);
        for (const auto& destination : destinations) if (destination.address < acquired.address + acquired.bytes && acquired.address < destination.address + destination.bytes &&
            !(destinations.size() == 1 && same_view(destination, acquired))) throw std::invalid_argument("HIP transport unpack destination partially aliases acquire source");
        entry.unpacked = true; std::uintptr_t source = acquired.address;
        for (const auto& destination : destinations) {
            if (destination.address != source) hip_check(i.calls.copy(reinterpret_cast<void*>(destination.address), reinterpret_cast<const void*>(source), static_cast<std::size_t>(destination.bytes), hipMemcpyDeviceToDevice, i.stream), "HIP transport owned inbound unpack");
            i.current(); source += destination.bytes;
        }
    } catch (...) { impl_->failed = true; throw; }
}
DirectDriverVisibilityProof HipDirectGpuStreamGpu::consumer_completion(const DirectDriverOperation& operation) {
    try {
        auto& i = *impl_; auto& entry = i.require(operation);
        if (!operation.receive_path || !entry.acquired || !entry.unpacked) throw std::invalid_argument("HIP transport consumer release lacks acquired/unpacked payload");
        if (entry.consumed) return {operation, DirectDriverVisibility::consumer_release_complete};
        i.record(entry.events[2]); entry.consumed = true; i.current(); return {operation, DirectDriverVisibility::consumer_release_complete};
    } catch (...) { impl_->failed = true; throw; }
}
void HipDirectGpuStreamGpu::retire(const DirectDriverOperation& operation) {
    try {
        auto& i = *impl_; auto& entry = i.require(operation);
        if (!(operation.receive_path ? entry.consumed : entry.released)) throw std::logic_error("HIP transport operation still owns GPU work");
        for (auto& event : entry.events) i.destroy(event);
        i.operations[operation.receive_path ? 1U : 0U].reset();
        if (!operation.receive_path) i.staged.reset();
    } catch (...) { impl_->failed = true; throw; }
}
void HipDirectGpuStreamGpu::request_stop() noexcept { impl_->stopping = true; }
DirectDriverQuiescenceStatus HipDirectGpuStreamGpu::poll_quiescence() noexcept {
    if (impl_->closed) return DirectDriverQuiescenceStatus::quiescent;
    try {
        impl_->current(false); const auto result = impl_->stream ? impl_->calls.query_stream(impl_->stream) : hipSuccess; impl_->current(false);
        if (result == hipErrorNotReady) return DirectDriverQuiescenceStatus::pending;
        hip_check(result, "HIP transport stream quiescence"); return DirectDriverQuiescenceStatus::quiescent;
    } catch (...) { impl_->failed = true; return DirectDriverQuiescenceStatus::failed_retained; }
}
DirectPoolReleaseStatus HipDirectGpuStreamGpu::close() noexcept {
    auto& i = *impl_; i.stopping = true;
    if (i.closed) return DirectPoolReleaseStatus::released;
    try {
        // No stream/work exists on early argument-validation failure.
        if (i.stream && poll_quiescence() != DirectDriverQuiescenceStatus::quiescent) return DirectPoolReleaseStatus::failed_retained;
        for (auto& entry : i.operations) if (entry) for (auto& event : entry->events) i.destroy(event);
        if (i.stream) {
            if (i.stream_destroy_unknown) throw std::runtime_error("HIP transport stream destruction unproved");
            i.stream_destroy_unknown = true; hip_check(i.calls.destroy_stream(i.stream), "HIP transport stream destroy"); i.stream = nullptr; i.stream_destroy_unknown = false;
            i.current(false);
        }
        for (std::size_t index = 0; index < i.host.size(); ++index) if (i.host[index]) {
            i.current(false);
            if (i.host_free_unknown[index]) throw std::runtime_error("HIP transport pinned metadata free unproved");
            i.host_free_unknown[index] = true; hip_check(i.calls.free_host(i.host[index]), "HIP transport pinned metadata free"); i.host[index] = nullptr; i.host_free_unknown[index] = false;
            i.current(false);
        }
        i.operations = {}; i.execution.lifetime.reset(); i.import_lease.reset(); i.closed = true; return DirectPoolReleaseStatus::released;
    } catch (...) { i.failed = true; return DirectPoolReleaseStatus::failed_retained; }
}
} // namespace strix::transport
