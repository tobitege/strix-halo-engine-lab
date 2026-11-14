#include "strix/hip_check.hpp"
#include "strix/hip_direct_transport_gpu_export.hpp"

#include <hip/hip_runtime_api.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strix::transport {
namespace {
using strix::hip_check;
[[maybe_unused, noreturn]] void unsupported() { throw std::logic_error("HIP DMA-BUF FD export requires supported Linux HIP/kernel/device"); }
} // namespace
HipDirectGpuExportCalls HipDirectGpuExportCalls::native() {
    HipDirectGpuExportCalls calls;
    calls.get_device = hipGetDevice; calls.runtime_version = hipRuntimeGetVersion; calls.driver_version = hipDriverGetVersion;
    calls.device_uuid = hipDeviceGetUuid; calls.pointer_attributes = hipPointerGetAttributes; calls.address_range = hipMemGetAddressRange;
    calls.dma_buf_range_export = [](int* descriptor, hipDeviceptr_t address, std::size_t bytes) {
#if defined(__linux__) && defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 7
        return hipMemGetHandleForAddressRange(descriptor, address, bytes, hipMemRangeHandleTypeDmaBufFd, 0);
#else
        static_cast<void>(descriptor); static_cast<void>(address); static_cast<void>(bytes); return hipErrorNotSupported;
#endif
    };
#if defined(__linux__)
    calls.get_fd_flags = [](int descriptor) { return ::fcntl(descriptor, F_GETFD); };
    calls.set_fd_flags = [](int descriptor, int flags) { return ::fcntl(descriptor, F_SETFD, flags); };
    calls.close_fd = [](int descriptor) { return ::close(descriptor); };
    calls.fd_identity = [](int descriptor) {
        struct stat identity{};
        if (descriptor < 0 || ::fstat(descriptor, &identity) != 0) throw std::runtime_error("HIP export FD inspection failed");
        return DirectGpuExportFdIdentity{static_cast<std::uint64_t>(identity.st_dev), static_cast<std::uint64_t>(identity.st_ino), false};
    };
#else
    calls.get_fd_flags = [](int) { return -1; }; calls.set_fd_flags = [](int, int) { return -1; }; calls.close_fd = [](int) { return -1; };
    calls.fd_identity = [](int) -> DirectGpuExportFdIdentity { unsupported(); };
#endif
    return calls;
}
HipDirectGpuExportIo::HipDirectGpuExportIo(HipDirectGpuExportCalls calls) : calls_(std::move(calls)) {
    if (!calls_.get_device || !calls_.runtime_version || !calls_.driver_version || !calls_.device_uuid ||
        !calls_.pointer_attributes || !calls_.address_range || !calls_.dma_buf_range_export ||
        !calls_.get_fd_flags || !calls_.set_fd_flags || !calls_.close_fd || !calls_.fd_identity) {
        throw std::invalid_argument("HIP export requires a complete native call boundary");
    }
}
DirectGpuExportDevice HipDirectGpuExportIo::inspect_device() {
    DirectGpuExportDevice result;
    int device = -1, runtime = 0, driver = 0;
    hip_check(calls_.get_device(&device), "HIP export active device");
    hipUUID uuid{}; hip_check(calls_.device_uuid(&uuid, device), "HIP export device UUID");
    hip_check(calls_.runtime_version(&runtime), "HIP export runtime version");
    hip_check(calls_.driver_version(&driver), "HIP export driver version");
    static_assert(sizeof(uuid.bytes) == result.uuid.size());
    std::memcpy(result.uuid.data(), uuid.bytes, result.uuid.size());
    result.index = device; result.runtime_version = runtime; result.driver_version = driver;
    return result;
}
DirectGpuExportAllocation HipDirectGpuExportIo::inspect_allocation(const TransportBufferView& view) {
    hipPointerAttribute_t attributes{};
    hip_check(calls_.pointer_attributes(&attributes, reinterpret_cast<const void*>(view.address)), "HIP export pointer attributes");
    hipDeviceptr_t base = nullptr; std::size_t bytes = 0;
    hip_check(calls_.address_range(&base, &bytes, reinterpret_cast<hipDeviceptr_t>(view.address)), "HIP export native allocation extent");
    return {reinterpret_cast<std::uintptr_t>(base), bytes, attributes.device,
            attributes.type == hipMemoryTypeDevice, attributes.isManaged != 0};
}
void HipDirectGpuExportIo::export_dma_buf(const DirectGpuExportAllocation& allocation, int& descriptor) {
#if defined(__linux__) && defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 7
    if (descriptor != -1 || allocation.base == 0 || allocation.bytes == 0 ||
        allocation.bytes > std::numeric_limits<std::size_t>::max()) throw std::invalid_argument("HIP export requires an empty FD slot and bounded native range");
    // HIP returns an owned DMA-BUF fd, distinct from a generic VMM/opaque fd.
    // The owner has adopted the output slot before the native call can fail.
    hip_check(calls_.dma_buf_range_export(&descriptor, reinterpret_cast<hipDeviceptr_t>(allocation.base),
        static_cast<std::size_t>(allocation.bytes)), "HIP DMA-BUF range export");
    if (descriptor < 0) throw std::runtime_error("HIP export returned no owned descriptor");
    const auto flags = calls_.get_fd_flags(descriptor);
    if (flags < 0 || calls_.set_fd_flags(descriptor, flags | FD_CLOEXEC) != 0) throw std::runtime_error("HIP export FD close-on-exec setup failed");
#else
    static_cast<void>(allocation); static_cast<void>(descriptor); unsupported();
#endif
}
DirectGpuExportFdIdentity HipDirectGpuExportIo::inspect_descriptor(int descriptor) {
#if defined(__linux__)
    const auto flags = calls_.get_fd_flags(descriptor);
    if (descriptor < 0 || flags < 0) throw std::runtime_error("HIP export FD inspection failed");
    auto identity = calls_.fd_identity(descriptor); identity.close_on_exec = (flags & FD_CLOEXEC) != 0;
    return identity;
#else
    static_cast<void>(descriptor); unsupported();
#endif
}
bool HipDirectGpuExportIo::close_descriptor(int descriptor) noexcept {
#if defined(__linux__)
    try { return descriptor >= 0 && calls_.close_fd(descriptor) == 0; } catch (...) { return false; }
#else
    static_cast<void>(descriptor); return false;
#endif
}
} // namespace strix::transport
