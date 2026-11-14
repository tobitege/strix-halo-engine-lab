#include "strix/file_snapshot.hpp"
#include "strix/mapped_model_file.hpp"

#include "strix/model_content_hash.hpp"
#include "strix/ple_direct.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <share.h>
#endif

namespace strix {
namespace {

// The operating-system side of a mapped model file. Each supported platform supplies the same
// primitives; MappedModelFile below is one flow over them.
//   FileState             device, inode, size, mtime (s, ns), ctime (s, ns) of an open file
//   open_descriptor       read-only descriptor, or -1
//   descriptor_state      state of the open file; false when it cannot be read
//   content_path          the path the full-file hash opens for this descriptor
//   path_names_open_file  whether that path names the open file
//   map_read_only         read-only mapping of the whole file, or nullptr
//   read_some             one positioned read; 0 at end of file, throws on error
using FileState = std::array<std::uint64_t,7>;

#if defined(__linux__)

FileState file_state(const struct stat& value) {
    return {static_cast<std::uint64_t>(value.st_dev),static_cast<std::uint64_t>(value.st_ino),
        static_cast<std::uint64_t>(value.st_size),static_cast<std::uint64_t>(value.st_mtim.tv_sec),
        static_cast<std::uint64_t>(value.st_mtim.tv_nsec),static_cast<std::uint64_t>(value.st_ctim.tv_sec),
        static_cast<std::uint64_t>(value.st_ctim.tv_nsec)};
}
int open_descriptor(const std::filesystem::path& path) noexcept {
    return ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
}
bool descriptor_state(int fd, FileState& out) noexcept {
    struct stat value {};
    if (::fstat(fd, &value) != 0 || value.st_size < 0) return false;
    out = file_state(value);
    return true;
}
// The hash reads the descriptor itself, so it can never read another file.
std::filesystem::path content_path(const std::filesystem::path&, int fd) {
    return std::filesystem::path("/proc/self/fd") / std::to_string(fd);
}
bool path_names_open_file(const std::filesystem::path&, const FileState&) noexcept { return true; }
void* map_read_only(int fd, std::uint64_t bytes) noexcept {
    void* address = ::mmap(nullptr, static_cast<std::size_t>(bytes), PROT_READ, MAP_PRIVATE, fd, 0);
    return address == MAP_FAILED ? nullptr : address;
}
void unmap(std::uintptr_t base, std::uint64_t bytes) noexcept {
    static_cast<void>(::munmap(reinterpret_cast<void*>(base), static_cast<std::size_t>(bytes)));
}
void close_descriptor(int fd) noexcept { static_cast<void>(::close(fd)); }
std::size_t read_some(int fd, std::byte* data, std::size_t bytes, std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()))
        throw std::overflow_error("mapped model read range does not fit off_t");
    for (;;) {
        const auto result = ::pread(fd, data, bytes, static_cast<off_t>(offset));
        if (result >= 0) return static_cast<std::size_t>(result);
        if (errno != EINTR) throw std::runtime_error("mapped model pread failed");
    }
}

#elif defined(_WIN32)

HANDLE os_handle(int fd) noexcept {
    return fd < 0 ? INVALID_HANDLE_VALUE : reinterpret_cast<HANDLE>(_get_osfhandle(fd));
}
// In the slots of the Linux state: volume serial and file index for device and inode, then the
// size, the last-write time and the creation time (100 ns ticks; the nanosecond slots stay 0).
// ChangeTime is left out, as in the hash cache of model_content_hash.cpp: metadata-only
// hard-link operations move it. Weaker than Linux in one respect: NTFS completes the last-write
// time only when the writing handle closes, so a same-size write through a handle that is still
// open is not seen.
bool handle_state(HANDLE handle, FileState& out) noexcept {
    BY_HANDLE_FILE_INFORMATION info{};
    if (handle == INVALID_HANDLE_VALUE || !GetFileInformationByHandle(handle, &info) ||
        (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
        return false;
    const auto u64 = [](DWORD high, DWORD low) { return (std::uint64_t{high} << 32U) | low; };
    out = {info.dwVolumeSerialNumber, u64(info.nFileIndexHigh, info.nFileIndexLow),
        u64(info.nFileSizeHigh, info.nFileSizeLow),
        u64(info.ftLastWriteTime.dwHighDateTime, info.ftLastWriteTime.dwLowDateTime), 0,
        u64(info.ftCreationTime.dwHighDateTime, info.ftCreationTime.dwLowDateTime), 0};
    return true;
}
int open_descriptor(const std::filesystem::path& path) noexcept {
    int fd = -1;
    return _wsopen_s(&fd, path.c_str(), _O_RDONLY | _O_BINARY | _O_NOINHERIT, _SH_DENYNO, 0) == 0 ? fd : -1;
}
bool descriptor_state(int fd, FileState& out) noexcept { return handle_state(os_handle(fd), out); }
// Windows has no /proc/self/fd: the hash opens the file by its own path, and that path is
// checked against the open file before and after the pass.
std::filesystem::path content_path(const std::filesystem::path& path, int) { return path; }
bool path_names_open_file(const std::filesystem::path& path, const FileState& state) noexcept {
    const HANDLE handle = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) return false;
    FileState named{};
    const bool read = handle_state(handle, named);
    CloseHandle(handle);
    return read && named == state;
}
void* map_read_only(int fd, std::uint64_t) noexcept {
    const HANDLE section = CreateFileMappingW(os_handle(fd), nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (section == nullptr) return nullptr;
    void* address = MapViewOfFile(section, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(section); // the view keeps the section alive
    return address;
}
void unmap(std::uintptr_t base, std::uint64_t) noexcept {
    static_cast<void>(UnmapViewOfFile(reinterpret_cast<void*>(base)));
}
void close_descriptor(int fd) noexcept { static_cast<void>(_close(fd)); }
// A positioned read on the descriptor's own handle; one call takes at most a DWORD of bytes.
std::size_t read_some(int fd, std::byte* data, std::size_t bytes, std::uint64_t offset) {
    OVERLAPPED at{};
    at.Offset = static_cast<DWORD>(offset & 0xffffffffU);
    at.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    DWORD result = 0;
    if (ReadFile(os_handle(fd), data, static_cast<DWORD>(std::min<std::size_t>(bytes, std::size_t{1} << 30U)), &result, &at))
        return result;
    if (GetLastError() == ERROR_HANDLE_EOF) return 0;
    throw std::runtime_error("mapped model read failed");
}

#endif

}  // namespace

MappedModelFile::MappedModelFile(
    MappedModelFile&& other) noexcept
    : fd_(other.fd_),
      mapping_base_(other.mapping_base_),
      bytes_(other.bytes_),
      allocation_generation_(
          other.allocation_generation_),
      logical_name_(
          std::move(other.logical_name_)),
      content_identity_(
          other.content_identity_), file_sha256_(other.file_sha256_),file_state_(other.file_state_) {
    other.fd_ = -1;
    other.mapping_base_ = 0;
    other.bytes_ = 0;
    other.allocation_generation_ = 0;
    other.content_identity_ = {};
    other.file_sha256_ = {};
    other.file_state_ = {};
}

MappedModelFile&
MappedModelFile::operator=(
    MappedModelFile&& other) noexcept {
    if (this != &other) {
        release();
        fd_ = other.fd_;
        mapping_base_ =
            other.mapping_base_;
        bytes_ = other.bytes_;
        allocation_generation_ =
            other.allocation_generation_;
        logical_name_ =
            std::move(other.logical_name_);
        content_identity_ =
            other.content_identity_;
        file_sha256_=other.file_sha256_;
        file_state_=other.file_state_;

        other.fd_ = -1;
        other.mapping_base_ = 0;
        other.bytes_ = 0;
        other.allocation_generation_ = 0;
        other.content_identity_ = {};
        other.file_sha256_ = {};
        other.file_state_ = {};
    }
    return *this;
}

MappedModelFile::~MappedModelFile() {
    release();
}

MappedModelFile MappedModelFile::open_read_only(
    const std::filesystem::path& path,
    std::string logical_name,
    AllocationGenerationSource& generations) {
#if !defined(__linux__) && !defined(_WIN32)
    static_cast<void>(path);
    static_cast<void>(logical_name);
    static_cast<void>(generations);
    throw std::runtime_error(
        "mapped model files require Linux or Windows");
#else
    if (logical_name.empty()) {
        throw std::invalid_argument(
            "mapped model logical name must be non-empty");
    }

    const int fd =
        open_descriptor(path);
    if (fd < 0) {
        throw std::runtime_error(
            "cannot open model file read-only");
    }

    MappedModelFile mapped;
    mapped.fd_ = fd;
    mapped.logical_name_ =
        std::move(logical_name);

    try {
        strix::LoaderCapture::record(path, fd);
        FileState before {};
        if (!descriptor_state(fd, before) ||
            before[2] == 0) {
            throw std::runtime_error(
                "cannot stat non-empty model file");
        }
        mapped.bytes_ =
            before[2];
        if (mapped.bytes_ >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error(
                "model file size does not fit host size_t");
        }

        const auto hashed_path =
            content_path(
                path,
                fd);
        if (!path_names_open_file(
                hashed_path,
                before)) {
            throw std::runtime_error(
                "model file changed while establishing content identity");
        }
        const ModelContentFile content_file{
            mapped.logical_name_,
            hashed_path,
        };
        const auto identities=hash_single_model_content_file(content_file);
        if(identities.file_bytes!=mapped.bytes_)
            throw std::runtime_error("model file extent changed during its full-file hash");
        mapped.content_identity_=identities.content_identity;
        mapped.file_sha256_=identities.file_sha256;

        FileState after_hash {};
        if (!descriptor_state(fd, after_hash) ||
            after_hash != before ||
            !path_names_open_file(
                hashed_path,
                before)) {
            throw std::runtime_error(
                "model file changed while establishing content identity");
        }

        void* address =
            map_read_only(
                fd,
                mapped.bytes_);
        if (address == nullptr) {
            throw std::runtime_error(
                "cannot mmap model file read-only");
        }
        mapped.mapping_base_ =
            reinterpret_cast<std::uintptr_t>(
                address);
        mapped.allocation_generation_ =
            generations.next();

        FileState after_map {};
        if (!descriptor_state(fd, after_map) ||
            after_map != after_hash) {
            throw std::runtime_error(
                "model file changed while mapping");
        }
        mapped.file_state_=after_map;
    } catch (...) {
        mapped.release();
        throw;
    }

    return mapped;
#endif
}

void MappedModelFile::validate_unchanged() const {
#if defined(__linux__) || defined(_WIN32)
    FileState current {};
    if(!valid() || fd_<0 || !file_sha256_.valid() || !content_identity_.valid() ||
        !descriptor_state(fd_,current) || current!=file_state_)
        throw std::runtime_error("mapped model changed after its original full-file hash");
#else
    throw std::runtime_error("mapped model file validation requires Linux or Windows");
#endif
}
void MappedModelFile::advise_dontneed(std::uint64_t byte_offset, std::uint64_t bytes) const noexcept {
#if defined(__linux__)
    if (fd_ >= 0 && bytes && byte_offset < bytes_)
        (void)::posix_fadvise(fd_, static_cast<off_t>(byte_offset), static_cast<off_t>(std::min(bytes, bytes_ - byte_offset)),
            POSIX_FADV_DONTNEED);
#else
    (void)byte_offset; (void)bytes;
#endif
}

std::size_t MappedModelFile::prefault_parallel(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) const noexcept {
#if defined(__linux__)
    const auto page_size = ::sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return 0;
    const auto page = static_cast<std::uint64_t>(page_size);
    constexpr std::uint64_t chunk = std::uint64_t{64} << 20;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> chunks;
    try {
        for (const auto& [offset, length] : ranges) {
            if (!length || offset >= bytes_) continue;
            const auto begin = offset / page * page;
            const auto end = offset + std::min(length, bytes_ - offset);
            // Normal advice for the pass: an earlier MADV_RANDOM on the range would disable readahead.
            (void)::madvise(reinterpret_cast<void*>(mapping_base_ + begin), static_cast<std::size_t>(end - begin), MADV_NORMAL);
            for (auto at = begin; at < end; at += chunk) chunks.emplace_back(at, std::min(chunk, end - at));
        }
    } catch (...) {
        return 0;
    }
    // Workers only issue madvise calls and touch atomics: nothing below throws once the chunk list exists.
    std::atomic<std::size_t> next{0}, done{0};
    auto worker = [&]() noexcept {
        for (std::size_t c; (c = next.fetch_add(1)) < chunks.size();) {
            auto* address = reinterpret_cast<void*>(mapping_base_ + chunks[c].first);
            const auto length = static_cast<std::size_t>(chunks[c].second);
#if defined(MADV_POPULATE_READ)
            if (::madvise(address, length, MADV_POPULATE_READ) == 0) { done.fetch_add(1); continue; }
#endif
            (void)::madvise(address, length, MADV_WILLNEED);
        }
    };
    const auto workers = std::min<std::size_t>(chunks.size(), std::min(16U, std::max(1U, std::thread::hardware_concurrency())));
    std::vector<std::thread> threads;
    try {
        threads.reserve(workers);
        for (std::size_t w = 1; w < workers; ++w) threads.emplace_back(worker);
    } catch (...) {}  // fewer threads: the chunks left are taken by the threads already running and this one
    if (workers) worker();
    for (auto& t : threads) t.join();
    return done.load();
#else
    (void)ranges;
    return 0;
#endif
}

void MappedModelFile::page_out(std::uint64_t byte_offset, std::uint64_t bytes) const noexcept {
    if (!valid() || !bytes || byte_offset >= bytes_) return;
#if defined(__linux__)
    const auto page_size = ::sysconf(_SC_PAGESIZE);
#elif defined(_WIN32)
    SYSTEM_INFO system{};
    GetSystemInfo(&system);
    const auto page_size = static_cast<long>(system.dwPageSize);
#else
    const long page_size = 0;
#endif
    if (page_size <= 0) return;
    const auto page = static_cast<std::uint64_t>(page_size);
    const auto first = byte_offset / page * page;
    const auto last = std::min((byte_offset + std::min(bytes, bytes_ - byte_offset) + page - 1U) / page * page,
        (bytes_ + page - 1U) / page * page);
    auto* address = reinterpret_cast<void*>(mapping_base_ + first);
    const auto length = static_cast<std::size_t>(last - first);
#if defined(__linux__)
#if defined(MADV_PAGEOUT)
    constexpr int pageout = MADV_PAGEOUT;
#else
    constexpr int pageout = 21;
#endif
    (void)::madvise(address, length, pageout);
#elif defined(_WIN32)
    // Documented for pages that are not locked: VirtualUnlock removes them from the working set (it then reports
    // ERROR_NOT_LOCKED, which is expected here).
    (void)VirtualUnlock(address, length);
#else
    (void)address; (void)length;
#endif
}

int MappedModelFile::drop_page_cache(std::uint64_t byte_offset, std::uint64_t bytes) const noexcept {
#if defined(__linux__)
    if (fd_ < 0) return EBADF;
    if (!bytes || byte_offset >= bytes_ || bytes > bytes_ - byte_offset) return EINVAL;
    return ::posix_fadvise(fd_, static_cast<off_t>(byte_offset), static_cast<off_t>(bytes), POSIX_FADV_DONTNEED);
#else
    (void)byte_offset; (void)bytes;
    return ENOSYS;
#endif
}

int MappedModelFile::open_direct_read_only() const {
#if defined(__linux__)
    try { validate_unchanged(); }
    catch (const std::exception& error) { throw ple_reclaim::DirectError(error.what()); }
    const auto path = "/proc/self/fd/" + std::to_string(fd_);
    const int direct = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (direct < 0) {
        const auto message = "PLE O_DIRECT: reopen unsupported/failed; rolling refused (errno " + std::to_string(errno) + ")";
        if (errno == EINVAL || errno == EOPNOTSUPP) throw ple_reclaim::DirectUnsupported(message);
        throw ple_reclaim::DirectError(message);
    }
    struct stat state{};
    if (::fstat(direct, &state) || file_state(state) != file_state_) {
        ::close(direct);
        throw ple_reclaim::DirectError("PLE O_DIRECT: reopened backing changed; rolling refused");
    }
    return direct;
#else
    throw ple_reclaim::DirectUnsupported("PLE O_DIRECT requires Linux; rolling refused");
#endif
}

void MappedModelFile::read_at(
    std::uint64_t byte_offset,
    std::span<std::byte> destination) const {
#if !defined(__linux__) && !defined(_WIN32)
    static_cast<void>(byte_offset);
    static_cast<void>(destination);
    throw std::runtime_error(
        "mapped model direct reads require Linux or Windows");
#else
    if (!valid() ||
        fd_ < 0) {
        throw std::logic_error(
            "mapped model file is not initialized");
    }
    if (byte_offset > bytes_ ||
        destination.size() >
            bytes_ - byte_offset) {
        throw std::out_of_range(
            "mapped model read exceeds file range");
    }
    std::size_t completed = 0;
    while (completed < destination.size()) {
        const auto result =
            read_some(
                fd_,
                destination.data() + completed,
                destination.size() - completed,
                byte_offset +
                    static_cast<std::uint64_t>(
                        completed));
        if (result == 0) {
            throw std::runtime_error(
                "mapped model pread reached unexpected EOF");
        }
        completed += result;
    }
#endif
}

ModelRuntimeBinding
MappedModelFile::build_runtime_binding(
    const ModelManifest& manifest) const {
    if (!valid() ||
        !content_identity_.valid()) {
        throw std::logic_error(
            "mapped model file is not initialized");
    }
    if (manifest.layout_identity == 0 ||
        manifest.layout_identity !=
            model_manifest_layout_identity(
                manifest)) {
        throw std::invalid_argument(
            "mapped model binding requires valid manifest identity");
    }

    ModelRuntimeBinding binding;
    binding.manifest_layout_identity =
        manifest.layout_identity;
    binding.model_content_identity =
        content_identity_;
    binding.tensors.reserve(
        manifest.tensors.size());

    for (std::size_t index = 0;
         index < manifest.tensors.size();
         ++index) {
        const auto& tensor =
            manifest.tensors[index];

        RuntimeTensorBinding runtime;
        runtime.tensor_index =
            static_cast<std::uint32_t>(
                index);
        runtime.bytes =
            tensor.byte_size;
        runtime.owner_node =
            tensor.placement.node;

        if (tensor.placement.kind ==
            PlacementKind::remote) {
            runtime.storage =
                RuntimeTensorStorage::remote;
            binding.tensors.push_back(
                runtime);
            continue;
        }

        if (tensor.absolute_offset >
                bytes_ ||
            tensor.byte_size >
                bytes_ -
                    tensor.absolute_offset) {
            throw std::invalid_argument(
                "manifest tensor range exceeds mapped model file");
        }

        const auto maximum =
            std::numeric_limits<
                std::uintptr_t>::max();
        if (tensor.absolute_offset >
            static_cast<std::uint64_t>(
                maximum - mapping_base_)) {
            throw std::overflow_error(
                "mapped tensor address overflows uintptr_t");
        }

        runtime.storage =
            RuntimeTensorStorage::file_mapped;
        runtime.address =
            mapping_base_ +
            static_cast<std::uintptr_t>(
                tensor.absolute_offset);
        runtime.allocation_generation =
            allocation_generation_;
        binding.tensors.push_back(
            runtime);
    }

    validate_model_runtime_binding(
        manifest,
        binding);
    return binding;
}

void MappedModelFile::release() noexcept {
#if defined(__linux__) || defined(_WIN32)
    if (mapping_base_ != 0 &&
        bytes_ != 0) {
        unmap(
            mapping_base_,
            bytes_);
    }
    if (fd_ >= 0) {
        close_descriptor(fd_);
    }
#endif
    fd_ = -1;
    mapping_base_ = 0;
    bytes_ = 0;
    allocation_generation_ = 0;
    logical_name_.clear();
    content_identity_ = {};
    file_sha256_ = {};
    file_state_ = {};
}

}  // namespace strix
