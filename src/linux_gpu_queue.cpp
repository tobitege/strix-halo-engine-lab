#include "strix/linux_gpu_queue.hpp"

#include <chrono>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strix::transport {
namespace {
bool equal_fence(const sgu_fence& a, const sgu_fence& b) noexcept {
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}
bool endpoint_equal(sgu_fence a, sgu_fence b) noexcept {
    a.connection_epoch = b.connection_epoch = 0;
    a.pool_binding_identity = b.pool_binding_identity = 0;
    a.import_generation = b.import_generation = 0;
    return equal_fence(a, b);
}
template<class Range> bool nonzero(const Range& value) {
    return std::any_of(std::begin(value), std::end(value), [](auto byte) { return byte != 0; });
}
[[noreturn]] void malformed(const char* message) { throw std::runtime_error(message); }
#if defined(__linux__)
struct NativeQueueFd {
    int fd = -1;
    struct stat identity{};
    std::function<void()> preflight;
    void current() {
        preflight();
        struct stat now{};
        if (fd < 0 || ::fstat(fd, &now) || !S_ISCHR(now.st_mode) ||
            now.st_dev != identity.st_dev || now.st_ino != identity.st_ino || now.st_rdev != identity.st_rdev)
            malformed("GPU queue owned descriptor identity changed");
        const int flags = ::fcntl(fd, F_GETFL), descriptor_flags = ::fcntl(fd, F_GETFD);
        if (flags < 0 || descriptor_flags < 0 || !(flags & O_NONBLOCK) || !(descriptor_flags & FD_CLOEXEC))
            malformed("GPU queue descriptor flags changed");
    }
    ~NativeQueueFd() {
        if (fd < 0) return;
        // On uncertainty never close a possibly foreign FD, and never retry an
        // ambiguous Linux close. The kernel still fences its own import lifetime.
        try { current(); const int owned = fd; fd = -1; static_cast<void>(::close(owned)); }
        catch (...) { fd = -1; }
    }
};
#endif
}
GpuQueueIo open_linux_gpu_queue(const std::string& path, std::function<void()> preflight) {
#if defined(__linux__)
    if (path.empty() || path.front() != '/' || !preflight || path.find('\0') != std::string::npos)
        throw std::invalid_argument("GPU queue requires an absolute configured character-device route and preflight");
    preflight();
    struct stat before{};
    if (::lstat(path.c_str(), &before) || !S_ISCHR(before.st_mode))
        malformed("GPU queue route must be a real character device, not a symlink");
    auto owner = std::make_shared<NativeQueueFd>(); owner->preflight = std::move(preflight);
    owner->fd = ::open(path.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW);
    if (owner->fd < 0) throw std::system_error(errno, std::generic_category(), "open configured GPU queue");
    if (::fstat(owner->fd, &owner->identity)) malformed("GPU queue fstat failed");
    if (before.st_dev != owner->identity.st_dev || before.st_ino != owner->identity.st_ino ||
        before.st_rdev != owner->identity.st_rdev) malformed("GPU queue route replaced during open");
    owner->current();
    return {owner, [owner](unsigned long command, void* argument) {
        owner->current(); errno = 0; const int result = ::ioctl(owner->fd, command, argument);
        const int error = result < 0 ? errno : 0; owner->current(); return GpuQueueSyscall{result, error};
    }, [owner] { owner->current(); }, [owner] { owner->current(); return owner->fd; },
    [owner](unsigned long command, void* argument) {
        owner->preflight(); errno = 0; const int result = ::ioctl(owner->fd, command, argument);
        return GpuQueueSyscall{result, result < 0 ? errno : 0};
    }};
#else
    static_cast<void>(path); static_cast<void>(preflight);
    throw std::runtime_error("GPU queue requires Linux");
#endif
}
struct LinuxGpuQueue::Impl {
    struct Entry {
        Entry() {} // Explicit construction also satisfies Clang/HIP nested optional traits.
        GpuQueueOperation operation;
        std::uint32_t frames = 0, completed_frames = 0;
        std::uint64_t completion_time = 0;
        bool submitted = false, complete = false, consumed = false, produced = false;
    };
    GpuQueueIo io;
    GpuQueueExpected expected;
    GpuQueuePool pool;
    sgu_query observed{};
    sgu_fence fence{};
    std::array<std::array<std::optional<Entry>, SGU_MAX_QUEUE_DEPTH>, 2> operations{};
    std::array<std::uint64_t, 2> prepared_sequence{}, submitted_sequence{};
    std::uint64_t last_handle = 0, last_epoch = 0, capacity = 0;
    std::uint32_t capabilities = 0;
    bool imported = false, armed = false, stopped = false, failed = false, import_ambiguous = false, cleanup_ambiguous = false;
    std::chrono::steady_clock::time_point verified{}; // lean: the last FD/pool identity check
    Impl(GpuQueueIo calls, GpuQueueExpected e) : io(std::move(calls)), expected(e) {}
    void current(bool productive = true) {
        if (productive && failed) malformed("GPU queue is failed; only bounded cleanup/recovery is allowed");
        verify();
    }
    GpuQueueSyscall raw(unsigned long command, void* argument) {
        current(false); const auto result = io.ioctl_call(command, argument); current(false); return result;
    }
    // Hot operations between session transitions: lean skips the FD/pool metadata syscalls, never the
    // failed state, operation identity or reply checks.
    void hot() {
        if (!expected.lean) { current(); return; }
        if (failed) malformed("GPU queue is failed; only bounded cleanup/recovery is allowed");
        // Bounded staleness: the clock read is not a system call; the identity check (fstat + two fcntl) runs
        // once per interval instead of five times around every operation.
        const auto now = std::chrono::steady_clock::now();
        if (now - verified >= expected.identity_interval) verify();
    }
    void verify() {
        io.require_current();
        if (pool.lifetime) pool.require_current();
        verified = std::chrono::steady_clock::now();
    }
    GpuQueueSyscall hot_raw(unsigned long command, void* argument) {
        if (!expected.lean) return raw(command, argument);
        const auto result = io.ioctl_owned(command, argument);
        // An unexpected failure is attributed only after the descriptor is known to be ours.
        if (result.result < 0 && result.error != EAGAIN && result.error != EINTR) verify();
        return result;
    }
    bool outstanding() const {
        for (const auto& direction : operations)
            for (const auto& candidate : direction) if (candidate && candidate->submitted && !candidate->complete) return true;
        return false;
    }
    void checked(unsigned long command, void* argument, bool hot_path = false) {
        const auto result = hot_path ? hot_raw(command, argument) : raw(command, argument);
        if (result.result != 0 || result.error != 0)
            throw std::system_error(result.error ? result.error : EIO, std::generic_category(), "GPU queue ioctl");
    }
    Entry& entry(const GpuQueueOperation& operation) {
        hot();
        if (!imported || operation.direction > 1 || operation.slot >= expected.depth)
            throw std::invalid_argument("GPU queue operation direction/slot is foreign");
        auto& candidate = operations[operation.direction][operation.slot];
        if (!candidate || candidate->operation != operation || operation.epoch != fence.connection_epoch ||
            operation.import_generation != fence.import_generation)
            throw std::invalid_argument("GPU queue operation identity is stale or foreign");
        return *candidate;
    }
    sgu_action action(unsigned long command, const GpuQueueOperation* operation = nullptr, bool hot_path = false) {
        sgu_action value{}; value.fence = fence;
        if (operation) { value.handle = operation->handle; value.sequence = operation->sequence; value.direction = operation->direction; }
        checked(command, &value, hot_path);
        if (value.reserved || value.handle != (operation ? operation->handle : 0) ||
            value.sequence != (operation ? operation->sequence : 0) || value.direction != (operation ? operation->direction : 0))
            malformed("GPU queue action returned foreign operation identity");
        if (command == SGU_IOC_RECOVER) {
            if (!endpoint_equal(value.fence, fence) || value.fence.connection_epoch || value.fence.pool_binding_identity || value.fence.import_generation)
                malformed("GPU queue recovery returned an imported or foreign identity");
        } else if (!equal_fence(value.fence, fence)) malformed("GPU queue action returned foreign binding");
        return value;
    }
};
LinuxGpuQueue::LinuxGpuQueue(GpuQueueIo io, GpuQueueExpected expected)
    : impl_(std::make_unique<Impl>(std::move(io), expected)) {
    auto& i = *impl_;
    if (!i.io.lifetime || !i.io.ioctl_call || !i.io.require_current || expected.depth < 2 || expected.depth > SGU_MAX_QUEUE_DEPTH ||
        (expected.terminal != GpuQueueTerminal::application && expected.terminal != GpuQueueTerminal::kernel_header) || (expected.lean && !i.io.ioctl_owned) ||
        expected.ring_frames < 32 || expected.ring_frames > SGU_MAX_FRAMES || !nonzero(expected.local) || !nonzero(expected.peer) || expected.local == expected.peer)
        throw std::invalid_argument("GPU queue requires complete identity and bounded multi-slot geometry");
    i.observed.fence.abi_major = SGU_ABI_MAJOR; i.observed.fence.abi_minor = SGU_ABI_MINOR;
    i.checked(SGU_IOC_QUERY, &i.observed);
    const auto& q = i.observed;
    if (q.fence.abi_major != SGU_ABI_MAJOR || q.fence.abi_minor != SGU_ABI_MINOR || q.fence.abi_cookie != SGU_ABI_COOKIE ||
        !nonzero(q.fence.device_instance) || !std::equal(expected.local.begin(), expected.local.end(), q.fence.local_uuid) ||
        !std::equal(expected.peer.begin(), expected.peer.end(), q.fence.peer_uuid) || q.fence.connection_epoch || q.fence.pool_binding_identity ||
        q.fence.import_generation || q.state != SGU_STATE_OPEN || q.frame_bytes != SGU_FRAME_BYTES || q.ring_frames != expected.ring_frames ||
        q.maximum_window_bytes != static_cast<std::uint64_t>(q.ring_frames) * SGU_FRAME_BYTES || q.maximum_imported_bytes != SGU_MAX_IMPORTED_BYTES ||
        q.maximum_sg_entries != SGU_MAX_SG_ENTRIES || q.maximum_event_batch != SGU_EVENT_BATCH || q.reserved)
        malformed("GPU queue QUERY differs from exact expected extension/endpoint/geometry");
    i.fence = q.fence;
    sgu_queue_config config{}; config.fence = i.fence; config.depth = expected.depth;
    i.checked(SGU_IOC_QUEUE, &config);
    const auto required = SGU_CAP_MULTI_OP | SGU_CAP_NONBLOCK_SUBMIT | SGU_CAP_SESSION_RECOVERY;
    if (!equal_fence(config.fence, i.fence) || config.depth != expected.depth || config.maximum_depth != SGU_MAX_QUEUE_DEPTH ||
        (config.features & required) != required || config.reserved)
        malformed("GPU queue optional queue/recovery interface is unavailable");
    i.capabilities = config.features;
    if (expected.terminal == GpuQueueTerminal::kernel_header && !(config.features & SGU_CAP_TERMINAL_HEADER))
        malformed("GPU queue kernel-validated terminal requested but this driver lacks SGU_CAP_TERMINAL_HEADER (patch 0016)");
    i.capacity = static_cast<std::uint64_t>((q.ring_frames - expected.depth) / expected.depth) * SGU_FRAME_BYTES;
    if (!i.capacity) malformed("GPU queue has no disjoint payload capacity");
}
LinuxGpuQueue::~LinuxGpuQueue() {
    if (!impl_) return;
    if (!release()) static_cast<void>(impl_.release()); // Retain exact FD/pool/GPU owners on unknown DMA completion.
}
const sgu_query& LinuxGpuQueue::observation() const noexcept { return impl_->observed; }
const sgu_fence& LinuxGpuQueue::binding() const noexcept { return impl_->fence; }
std::uint32_t LinuxGpuQueue::depth() const noexcept { return impl_->expected.depth; }
std::uint64_t LinuxGpuQueue::slot_capacity() const noexcept { return impl_->capacity; }
GpuQueueTerminal LinuxGpuQueue::terminal() const noexcept { return impl_->expected.terminal; }
std::uint32_t LinuxGpuQueue::required_terminal_bytes() const noexcept {
    return impl_->expected.terminal == GpuQueueTerminal::kernel_header ? SGU_TERMINAL_HEADER_BYTES : 0U;
}
std::uint64_t LinuxGpuQueue::terminal_offset(std::uint32_t slot) const {
    if (slot >= depth()) throw std::invalid_argument("GPU queue terminal slot is out of range");
    return static_cast<std::uint64_t>(impl_->expected.ring_frames - depth() + slot) * SGU_FRAME_BYTES;
}
void LinuxGpuQueue::import_pool(GpuQueuePool pool, std::uint64_t epoch, std::uint64_t pool_id) {
    auto& i = *impl_; i.current();
    if (i.imported || i.import_ambiguous || i.armed || !epoch || epoch <= i.last_epoch || !pool_id || !pool.lifetime || !pool.require_current || !pool.gpu_quiescent)
        throw std::invalid_argument("GPU queue import requires a fresh epoch and complete retained pool");
    const auto validate = [&](const sgu_buffer& b) {
        return b.fd >= 0 && !b.reserved && b.full_bytes && b.full_bytes <= SGU_MAX_IMPORTED_BYTES && b.window_offset <= b.full_bytes &&
            b.window_bytes == i.observed.maximum_window_bytes && b.window_bytes <= b.full_bytes - b.window_offset &&
            b.window_offset % SGU_FRAME_BYTES == 0 && b.allocation_generation && b.fd_inode;
    };
    if (!validate(pool.tx) || !validate(pool.rx) || pool.rx.full_bytes > SGU_MAX_IMPORTED_BYTES - pool.tx.full_bytes ||
        pool.tx.fd == pool.rx.fd || (pool.tx.fd_device == pool.rx.fd_device && pool.tx.fd_inode == pool.rx.fd_inode))
        throw std::invalid_argument("GPU queue pool extents or backing descriptors alias/exceed admission");
    pool.require_current();
    sgu_import value{}; value.fence = i.fence; value.fence.connection_epoch = epoch; value.fence.pool_binding_identity = pool_id;
    value.tx = pool.tx; value.rx = pool.rx;
    i.pool = std::move(pool); i.import_ambiguous = true;
    try {
        i.checked(SGU_IOC_IMPORT, &value);
        if (!endpoint_equal(value.fence, i.fence) || value.fence.connection_epoch != epoch || value.fence.pool_binding_identity != pool_id ||
            !value.fence.import_generation || std::memcmp(&value.tx, &i.pool.tx, sizeof(value.tx)) || std::memcmp(&value.rx, &i.pool.rx, sizeof(value.rx)) || value.reserved[0] || value.reserved[1])
            malformed("GPU queue IMPORT returned changed identity or extents");
        i.fence = value.fence; i.imported = true; i.import_ambiguous = false; i.stopped = false; i.last_epoch = epoch;
        i.prepared_sequence = {}; i.submitted_sequence = {};
    } catch (...) { i.failed = true; throw; }
}
void LinuxGpuQueue::arm() {
    auto& i = *impl_; i.current();
    if (!i.imported || i.armed || i.stopped) throw std::logic_error("GPU queue ARM requires one fresh import");
    try { const auto value = i.action(SGU_IOC_ARM); if (value.result || value.queued_frames) malformed("GPU queue ARM result is invalid"); i.armed = true; }
    catch (...) { i.failed = true; throw; }
}
GpuQueueOperation LinuxGpuQueue::prepare(std::uint32_t direction, std::uint32_t slot, std::uint64_t sequence,
    std::uint64_t bytes, std::uint32_t terminal_bytes) {
    auto& i = *impl_; i.hot();
    if (!i.armed || i.stopped || direction > 1 || slot >= depth() || !bytes || bytes > i.capacity || !terminal_bytes || terminal_bytes > SGU_FRAME_BYTES ||
        (required_terminal_bytes() && terminal_bytes != required_terminal_bytes()) ||
        !sequence || sequence <= i.prepared_sequence[direction] || i.operations[direction][slot] || i.last_handle == UINT64_MAX)
        throw std::invalid_argument("GPU queue PREPARE_SLOT violates geometry, ownership or monotonic identity");
    GpuQueueOperation operation{i.fence.connection_epoch, i.fence.import_generation, ++i.last_handle, sequence,
        bytes, static_cast<std::uint64_t>(slot) * i.capacity, direction, slot, terminal_bytes};
    sgu_prepare_slot value{}; value.slot = slot;
    value.operation.fence = i.fence; value.operation.handle = operation.handle; value.operation.sequence = sequence;
    value.operation.pool_offset = operation.pool_offset; value.operation.payload_bytes = bytes;
    value.operation.terminal_bytes = terminal_bytes; value.operation.direction = direction;
    value.operation.flags = i.expected.terminal == GpuQueueTerminal::kernel_header ? SGU_PREPARE_TERMINAL_HEADER : 0ULL;
    // Reserve userspace state before exposing identity. A failed ioctl is never
    // blindly replayed and no slot becomes reusable on an ambiguous result.
    auto& entry = i.operations[direction][slot]; entry.emplace(); entry->operation = operation;
    entry->frames = static_cast<std::uint32_t>((bytes + SGU_FRAME_BYTES - 1) / SGU_FRAME_BYTES) + 1;
    try { i.checked(SGU_IOC_PREPARE_SLOT, &value, true); i.prepared_sequence[direction] = sequence; return operation; }
    catch (...) { i.failed = true; throw; }
}
void LinuxGpuQueue::gpu_produced(const GpuQueueOperation& operation) {
    auto& entry = impl_->entry(operation);
    if (operation.direction != SGU_DIRECTION_TX || entry.submitted || entry.produced) throw std::logic_error("GPU queue producer release is duplicated or foreign");
    entry.produced = true;
}
bool LinuxGpuQueue::submit(const GpuQueueOperation& operation) {
    auto& i = *impl_; auto& entry = i.entry(operation);
    if (!i.armed || i.stopped || entry.submitted || operation.sequence <= i.submitted_sequence[operation.direction])
        throw std::logic_error("GPU queue SUBMIT is duplicated or out of order");
    if (operation.direction == SGU_DIRECTION_TX && !entry.produced) throw std::logic_error("GPU queue TX lacks producer release");
    // Never let a later prepared operation overtake an EAGAIN predecessor.
    for (const auto& other : i.operations[operation.direction])
        if (other && !other->submitted && other->operation.sequence < operation.sequence)
            throw std::logic_error("GPU queue SUBMIT would strand an earlier prepared sequence");
    sgu_action value{}; value.fence = i.fence; value.handle = operation.handle; value.sequence = operation.sequence; value.direction = operation.direction;
    try {
        const auto result = i.hot_raw(SGU_IOC_SUBMIT, &value);
        if (result.result < 0 && result.error == EAGAIN) {
            if (!equal_fence(value.fence, i.fence) || value.handle != operation.handle || value.sequence != operation.sequence ||
                value.direction != operation.direction || value.result || value.queued_frames || value.reserved)
                malformed("GPU queue EAGAIN changed the operation");
            return false;
        }
        if (result.result != 0 || result.error) throw std::system_error(result.error ? result.error : EIO, std::generic_category(), "GPU queue SUBMIT");
        if (!equal_fence(value.fence, i.fence) || value.handle != operation.handle || value.sequence != operation.sequence ||
            value.direction != operation.direction || value.result || value.reserved || value.queued_frames != entry.frames)
            malformed("GPU queue SUBMIT acknowledgment is foreign or incomplete");
        entry.submitted = true; i.submitted_sequence[operation.direction] = operation.sequence; return true;
    } catch (...) { i.failed = true; throw; }
}
std::optional<GpuQueueCompletion> LinuxGpuQueue::poll() {
    auto& i = *impl_; i.hot();
    if (!i.imported || !i.armed || i.stopped) throw std::logic_error("GPU queue EVENT requires an armed import");
    // Nothing submitted is incomplete: a valid EVENT cannot exist, so lean does not ask the kernel.
    if (i.expected.lean && !i.outstanding()) return std::nullopt;
    sgu_event event{}; event.fence = i.fence;
    try {
        const auto result = i.hot_raw(SGU_IOC_EVENT, &event);
        if (result.result < 0 && (result.error == EAGAIN || result.error == EINTR)) return std::nullopt;
        if (result.result != 0 || result.error) throw std::system_error(result.error ? result.error : EIO, std::generic_category(), "GPU queue EVENT");
        if (equal_fence(event.fence, i.fence) && event.status == SGU_EVENT_FAILED && event.count <= SGU_EVENT_BATCH) {
            std::uint32_t flags = 0;
            for (std::uint32_t n = 0; n < event.count; ++n) flags |= event.frames[n].error_flags;
            if ((flags & SGU_FRAME_ERROR_TERMINAL) && i.expected.terminal == GpuQueueTerminal::kernel_header)
                throw GpuQueueTerminalMismatch("GPU queue peer terminal header rejected by the kernel (SGU_FRAME_ERROR_TERMINAL): "
                    "epoch, sequence, payload size or slot differ between peers; RECOVER with a fresh epoch");
        }
        if (!equal_fence(event.fence, i.fence) || event.direction > 1 || event.reserved || event.status != SGU_EVENT_COMPLETE ||
            !event.count || event.count > SGU_EVENT_BATCH || !event.monotonic_ns)
            malformed("GPU queue EVENT is failed, canceled or foreign");
        Impl::Entry* entry = nullptr;
        for (auto& candidate : i.operations[event.direction]) if (candidate && candidate->operation.handle == event.handle) entry = &*candidate;
        if (!entry || !entry->submitted || entry->complete || event.sequence != entry->operation.sequence || event.payload_bytes != entry->operation.payload_bytes ||
            event.total_frames != entry->frames || event.first_frame != entry->completed_frames || event.count > entry->frames - entry->completed_frames ||
            (entry->completion_time && event.monotonic_ns != entry->completion_time))
            malformed("GPU queue EVENT cursor or operation identity is invalid");
        for (std::uint32_t n = 0; n < SGU_EVENT_BATCH; ++n) {
            const auto& frame = event.frames[n];
            if (n >= event.count) {
                const sgu_frame_result zero{}; if (std::memcmp(&frame, &zero, sizeof(frame))) malformed("GPU queue unused EVENT fields are nonzero");
                continue;
            }
            const auto logical = event.first_frame + n; const bool terminal = logical + 1 == event.total_frames;
            const auto offset = terminal ? terminal_offset(entry->operation.slot) : entry->operation.pool_offset + static_cast<std::uint64_t>(logical) * SGU_FRAME_BYTES;
            const auto bytes = terminal ? entry->operation.terminal_bytes : static_cast<std::uint32_t>(std::min<std::uint64_t>(SGU_FRAME_BYTES, entry->operation.payload_bytes - static_cast<std::uint64_t>(logical) * SGU_FRAME_BYTES));
            if (frame.pool_offset != offset || frame.physical_slot != offset / SGU_FRAME_BYTES || frame.capacity != SGU_FRAME_BYTES ||
                frame.expected_bytes != bytes || frame.actual_bytes != bytes || frame.error_flags || frame.reserved)
                malformed("GPU queue EVENT frame geometry, length or error status differs");
        }
        entry->completed_frames += event.count; entry->completion_time = event.monotonic_ns;
        if (entry->completed_frames != entry->frames) return std::nullopt;
        entry->complete = true; return GpuQueueCompletion{entry->operation, event.monotonic_ns};
    } catch (...) { i.failed = true; throw; }
}
void LinuxGpuQueue::gpu_consumed(const GpuQueueOperation& operation) {
    auto& entry = impl_->entry(operation);
    if (operation.direction != SGU_DIRECTION_RX || !entry.complete || entry.consumed) throw std::logic_error("GPU queue RX credit lacks one completed GPU consumer");
    entry.consumed = true;
}
void LinuxGpuQueue::peer_consumed(const GpuQueueOperation& operation) {
    auto& entry = impl_->entry(operation);
    if (operation.direction != SGU_DIRECTION_TX || !entry.complete || entry.consumed) throw std::logic_error("GPU queue TX credit lacks one authenticated peer consumer");
    entry.consumed = true;
}
void LinuxGpuQueue::retire(const GpuQueueOperation& operation) {
    auto& i = *impl_; const auto& entry = i.entry(operation);
    if (!entry.complete || !entry.consumed) throw std::logic_error("GPU queue RETIRE requires controller and consumer ownership completion");
    try { const auto value = i.action(SGU_IOC_RETIRE, &operation, true); if (value.result || value.queued_frames) malformed("GPU queue RETIRE acknowledgment is invalid"); i.operations[operation.direction][operation.slot].reset(); }
    catch (...) { i.failed = true; throw; }
}
sgu_stats LinuxGpuQueue::stats() {
    auto& i = *impl_; i.current(); sgu_stats value{}; value.fence = i.fence;
    try { i.checked(SGU_IOC_STATS, &value);
        if (!equal_fence(value.fence, i.fence) || value.queue_depth != depth() || value.reserved[0] || value.reserved[1]) malformed("GPU queue STATS binding changed");
        return value;
    } catch (...) { i.failed = true; throw; }
}
void LinuxGpuQueue::invalidate() noexcept { impl_->failed = true; }
bool LinuxGpuQueue::failed() const noexcept { return impl_->failed; }
bool LinuxGpuQueue::stop() noexcept {
    auto& i = *impl_;
    if (i.import_ambiguous) return false;
    if (!i.imported || i.stopped) return true;
    try { const auto value = i.action(SGU_IOC_STOP);
        if (value.result != SGU_RESULT_CONTROLLER_STOPPED || value.queued_frames) malformed("GPU queue STOP lacks controller quiescence");
        i.stopped = true; i.armed = false; return true;
    } catch (...) { i.failed = true; return false; }
}
bool LinuxGpuQueue::release() noexcept {
    auto& i = *impl_;
    if (i.cleanup_ambiguous || !stop()) return false;
    if (!i.imported) return !i.import_ambiguous;
    try {
        i.current(false); if (!i.pool.gpu_quiescent()) return false; i.current(false);
        i.cleanup_ambiguous = true;
        const auto value = i.action(SGU_IOC_RELEASE); if (value.result || value.queued_frames) malformed("GPU queue RELEASE acknowledgment is invalid");
        i.cleanup_ambiguous = false; i.imported = false; i.operations = {}; i.pool = {};
        i.fence.connection_epoch = i.fence.pool_binding_identity = i.fence.import_generation = 0;
        return true;
    } catch (...) { i.failed = true; return false; }
}
bool LinuxGpuQueue::recover() noexcept {
    auto& i = *impl_;
    if (i.cleanup_ambiguous || !stop() || !i.imported) return false;
    try {
        i.current(false); if (!i.pool.gpu_quiescent()) return false; i.current(false);
        i.cleanup_ambiguous = true;
        const auto value = i.action(SGU_IOC_RECOVER);
        if (value.result != SGU_RESULT_SESSION_RECOVERED || value.queued_frames) malformed("GPU queue RECOVER lacks released session identity");
        i.cleanup_ambiguous = false; i.fence = value.fence; i.operations = {}; i.pool = {}; i.imported = false; i.armed = false; i.stopped = false;
        i.failed = false; i.prepared_sequence = {}; i.submitted_sequence = {}; return true;
    } catch (...) { i.failed = true; return false; }
}

GpuQueueResetResult LinuxGpuQueue::reset_controller_group(
    std::span<LinuxGpuQueue* const> members, bool include_network) noexcept {
    GpuQueueResetResult result;
    try {
        if (members.empty() || members.size() > SGU_MAX_RESET_MEMBERS) {
            result.error = EINVAL; return result;
        }
        sgu_reset_group request{};
        request.count = static_cast<__u32>(members.size());
        request.flags = include_network ? SGU_RESET_GROUP_NETWORK : 0U;
        for (std::size_t index = 0; index < members.size(); ++index) {
            if (!members[index]) { result.error = EINVAL; return result; }
            auto& i = *members[index]->impl_;
            if (!(i.capabilities & SGU_CAP_GROUP_RESET) || !i.io.native_descriptor) {
                result.error = EOPNOTSUPP; return result;
            }
            i.current(false);
            if (!i.imported || i.import_ambiguous || i.cleanup_ambiguous || !i.stopped || i.armed) {
                result.error = EBUSY; return result;
            }
            for (const auto& direction : i.operations) for (const auto& entry : direction)
                if (entry) { result.error = EBUSY; return result; }
            if (!i.pool.gpu_quiescent()) { result.error = EBUSY; return result; }
            const int fd = i.io.native_descriptor();
            if (fd < 0) { result.error = EBADF; return result; }
            for (std::size_t prior = 0; prior < index; ++prior) {
                if (members[index] == members[prior] || request.members[prior].fd == fd) {
                    result.error = EINVAL; return result;
                }
            }
            request.members[index].fd = fd;
            request.members[index].fence = i.fence;
            i.current(false);
        }
        const auto submitted = request;
        // A thrown post-ioctl identity check may follow an actual reset. No
        // member may keep using its old state even if no result is delivered.
        result.may_have_reset = true;
        for (auto* member : members) member->impl_->failed = true;
        auto& first = *members.front()->impl_;
        const auto call = first.raw(SGU_IOC_RESET_GROUP, &request);
        for (auto* member : members) member->impl_->current(false);
        if (call.result || call.error) {
            result.error = call.error ? call.error : EIO; return result;
        }
        if (request.result != SGU_RESULT_GROUP_RESET || request.count != submitted.count ||
            request.flags != submitted.flags || request.reserved ||
            std::memcmp(request.members, submitted.members, sizeof(request.members))) {
            result.error = EPROTO; return result;
        }
        result.success = true;
    } catch (...) { result.error = EIO; }
    return result;
}

} // namespace strix::transport
