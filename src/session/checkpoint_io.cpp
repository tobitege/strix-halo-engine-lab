#include "strix/session/checkpoint_io.hpp"
#include "strix/unbuffered_file.hpp"
#include "../core/host_lane.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <vector>
#include <fstream>
#include <limits>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/syscall.h>
#endif
#endif

namespace strix::session::checkpoint_io {
namespace {
// Checkpoint bytes share the memory bus with the decode weight stream: every chunk moved here is drawn from the
// process-wide byte budget (unlimited unless a session is decoding).
void governed(std::uint64_t bytes) { ::strix::core::shared_byte_budget().acquire(bytes); }
thread_local std::uint64_t observed_activity = 0;
void inject(Operation op) {
    if (faults().fail == op) {
        if (faults().skip) --faults().skip;
        else throw Error(op, "checkpoint IO injected failure");
    }
}
void check(bool ok, Operation op, const char* message) {
    if (!ok) throw Error(op, message);
}
std::filesystem::path parent(const std::filesystem::path& p) {
    return p.parent_path().empty() ? std::filesystem::path(".") : p.parent_path();
}
}
std::uint64_t activity() noexcept { return observed_activity; }
Faults& faults() noexcept { static thread_local Faults value; return value; }
std::span<const std::byte> take(std::span<const std::byte> bytes, std::size_t& at, std::uint64_t n) {
    check(at <= bytes.size() && n <= bytes.size() - at, Operation::read, "checkpoint frame bounds");
    auto out = bytes.subspan(at, static_cast<std::size_t>(n));
    at += static_cast<std::size_t>(n); return out;
}
void sync_directory(const std::filesystem::path& dir, bool durable) {
    ++observed_activity;
    if (!durable) return;
    inject(Operation::directory_sync);
#ifndef _WIN32
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    check(fd >= 0, Operation::directory_sync, "checkpoint directory open failed");
    int result; do { result = ::fsync(fd); } while (result < 0 && errno == EINTR);
    const auto closed = ::close(fd);
    check(result == 0 && closed == 0, Operation::directory_sync, "checkpoint directory fsync failed");
#else
    // Windows has no supported directory FlushFileBuffers contract. Replacement uses
    // MOVEFILE_WRITE_THROUGH; file contents are explicitly flushed before that call.
    (void)dir;
#endif
}
void create_directories(const std::filesystem::path& dir, bool durable) {
    ++observed_activity;
    std::vector<std::filesystem::path> missing;
    for (auto p = std::filesystem::absolute(dir); !std::filesystem::exists(p); p = p.parent_path())
        missing.push_back(p);
    std::filesystem::create_directories(dir);
    for (auto it = missing.rbegin(); it != missing.rend(); ++it) sync_directory(it->parent_path(), durable);
}
void sync_existing(const std::filesystem::path& p) {
    ++observed_activity;
    inject(Operation::sync);
#ifdef _WIN32
    const auto h = CreateFileW(p.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    check(h != INVALID_HANDLE_VALUE, Operation::sync, "checkpoint retained file open failed");
    const auto ok = FlushFileBuffers(h); CloseHandle(h);
    check(ok != 0, Operation::sync, "checkpoint retained file fsync failed");
#else
    const auto fd = ::open(p.c_str(), O_RDWR | O_CLOEXEC);
    check(fd >= 0, Operation::sync, "checkpoint retained file open failed");
    int ok; do { ok = ::fsync(fd); } while (ok < 0 && errno == EINTR);
    const auto closed = ::close(fd);
    check(ok == 0 && closed == 0, Operation::sync, "checkpoint retained file fsync failed");
#endif
}
void publish_directory(const std::filesystem::path& temporary, const std::filesystem::path& target, bool durable) {
    ++observed_activity;
    check(std::filesystem::equivalent(parent(temporary), parent(target)), Operation::replace,
        "checkpoint directory publication requires the same parent");
    check(!std::filesystem::is_symlink(temporary) && std::filesystem::is_directory(temporary),
        Operation::replace, "checkpoint publication source is not a directory");
    check(!std::filesystem::exists(target) && !std::filesystem::is_symlink(target),
        Operation::replace, "checkpoint immutable directory target exists");
    inject(Operation::replace);
#ifdef _WIN32
    check(MoveFileExW(temporary.c_str(), target.c_str(), durable ? MOVEFILE_WRITE_THROUGH : 0) != 0,
        Operation::replace, "checkpoint immutable directory rename failed");
#elif defined(__linux__) && defined(SYS_renameat2)
    // rename() alone can overwrite an empty directory; RENAME_NOREPLACE cannot.
    check(::syscall(SYS_renameat2, AT_FDCWD, temporary.c_str(), AT_FDCWD, target.c_str(), 1U) == 0,
        Operation::replace, "checkpoint immutable directory rename failed");
#elif defined(__APPLE__)
    check(::renamex_np(temporary.c_str(), target.c_str(), RENAME_EXCL) == 0,
        Operation::replace, "checkpoint immutable directory rename failed");
#else
    throw Error(Operation::replace, "checkpoint no-replace directory publication unsupported");
#endif
    sync_directory(parent(target), durable); // A failure here cannot undo visibility.
}
struct DirectoryLock::Impl {
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
    DWORD low = MAXDWORD, high = MAXDWORD;
    ~Impl() { if (handle != INVALID_HANDLE_VALUE) { OVERLAPPED o{}; UnlockFileEx(handle,0,low,high,&o); CloseHandle(handle); } }
#else
    int fd = -1;
    ~Impl() { if (fd >= 0) { (void)::flock(fd, LOCK_UN); ::close(fd); } }
#endif
};
DirectoryLock::DirectoryLock(const std::filesystem::path& dir, LockMode mode, const std::string& basename, LockRange range) : impl_(std::make_unique<Impl>()) {
    ++observed_activity;
#ifndef _WIN32
    (void)range; // flock always locks the inode, including legacy first-byte callers.
#endif
    check(!dir.empty(), Operation::create, "checkpoint empty lock directory");
    // A leading dot also excludes Windows device names (CON, NUL, COM1, ...).
    // Validate before directory creation or opening any host path.
    check(basename.size() > 1 && basename.size() <= 255 && basename.front() == '.' && basename.back() != '.' &&
        basename.find("..") == std::string::npos && std::all_of(basename.begin(), basename.end(), [](unsigned char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                c == '.' || c == '-' || c == '_';
        }), Operation::create, "checkpoint lock basename must be a private filename");
    checkpoint_io::create_directories(dir, true);
    const auto path = dir / basename; // Persistent inode: never remove or replace.
#ifdef _WIN32
    if (range == LockRange::first_byte || basename != ".session.lock") { impl_->low = 1; impl_->high = 0; }
    impl_->handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    check(impl_->handle != INVALID_HANDLE_VALUE, Operation::create, "checkpoint directory lock open failed");
    OVERLAPPED o{};
    if (!LockFileEx(impl_->handle, LOCKFILE_EXCLUSIVE_LOCK | (mode == LockMode::nonblocking ? LOCKFILE_FAIL_IMMEDIATELY : 0),
        0, impl_->low, impl_->high, &o)) {
        const auto error = GetLastError();
        throw Error(Operation::create, "checkpoint directory locked/unavailable",
            mode == LockMode::nonblocking && error == ERROR_LOCK_VIOLATION);
    }
#else
    impl_->fd = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    check(impl_->fd >= 0, Operation::create, "checkpoint directory lock open failed");
    int result; do { result = ::flock(impl_->fd, LOCK_EX | (mode == LockMode::nonblocking ? LOCK_NB : 0)); }
        while (result < 0 && errno == EINTR);
    if (result != 0) throw Error(Operation::create, "checkpoint directory locked/unavailable",
        mode == LockMode::nonblocking && (errno == EWOULDBLOCK || errno == EAGAIN));
#endif
}
DirectoryLock::~DirectoryLock() = default;
struct Publication::Impl {
    std::filesystem::path target, temporary;
    Options options;
    std::uint64_t offset = 0;
    bool committed = false, created = false;
    std::unique_ptr<state_file_io::SequentialWriter> direct;
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
    void close() noexcept {
        direct.reset();
#ifdef _WIN32
        if (handle != INVALID_HANDLE_VALUE) { CloseHandle(handle); handle = INVALID_HANDLE_VALUE; }
#else
        if (fd >= 0) { ::close(fd); fd = -1; }
#endif
    }
    ~Impl() {
        close();
        if (created && !committed && !temporary.empty()) { std::error_code ec; std::filesystem::remove(temporary, ec); }
    }
};
Publication::Publication(const std::filesystem::path& target, Options options) {
    ++observed_activity;
    inject(Operation::allocation);
    impl_ = std::make_unique<Impl>();
    auto& p = *impl_; p.target = target; p.options = options;
    checkpoint_io::create_directories(parent(target), options.durable);
    static std::atomic<std::uint64_t> sequence{0};
    // CREATE_NEW/O_EXCL is authoritative even across processes and crashed leftovers.
    const std::filesystem::path prefix(options.temporary_prefix);
    check(options.temporary_prefix.find_first_of("/\\") == std::string::npos && !prefix.has_root_path() &&
        options.temporary_prefix!="." && options.temporary_prefix!="..", Operation::create, "checkpoint temporary prefix is not a filename");
    check(options.temporary_suffix.empty() || (options.temporary_suffix.front() == '.' &&
        options.temporary_suffix.size() > 1 && std::all_of(options.temporary_suffix.begin()+1,
        options.temporary_suffix.end(), [](unsigned char c) { return (c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'); })),
        Operation::create, "checkpoint temporary suffix is not an extension");
    p.temporary = options.temporary_prefix.empty() ? target : parent(target)/options.temporary_prefix;
    p.temporary += ".tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(++sequence);
    p.temporary += options.temporary_suffix;
    inject(Operation::create);
    if (options.direct == DirectIo::automatic && state_file_io::unbuffered_io_enabled()) {
        p.direct = std::make_unique<state_file_io::SequentialWriter>(p.temporary, true, options.permissions);
        check(p.direct->open(), Operation::create, "checkpoint temporary create failed");
        p.created = true; return;
    }
#ifdef _WIN32
    p.handle = CreateFileW(p.temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    check(p.handle != INVALID_HANDLE_VALUE, Operation::create, "checkpoint temporary create failed");
#else
    p.fd = ::open(p.temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, options.permissions);
    check(p.fd >= 0, Operation::create, "checkpoint temporary create failed");
#endif
    p.created = true;
}
Publication::~Publication() = default;
void Publication::write(std::span<const std::byte> bytes) {
    auto& p = *impl_;
    check(!p.committed, Operation::write, "checkpoint already published");
    while (!bytes.empty()) {
        inject(Operation::write);
        check(p.offset < faults().fail_write_at, Operation::write, "checkpoint write failed");
        auto n = std::min<std::uint64_t>(bytes.size(), 1U << 26);
        if (faults().max_write) n = std::min(n, faults().max_write);
        n = std::min(n, faults().fail_write_at - p.offset);
        check(n <= UINT64_MAX - p.offset, Operation::write, "checkpoint write overflow");
        governed(n);
        std::uint64_t done = 0;
        if (p.direct) { p.direct->write(bytes.data(), n); done = n; }
        else {
#ifdef _WIN32
            DWORD written = 0;
            check(WriteFile(p.handle, bytes.data(), static_cast<DWORD>(n), &written, nullptr) != 0,
                Operation::write, "checkpoint write failed"); done = written;
#else
            ssize_t written; do { written = ::write(p.fd, bytes.data(), static_cast<std::size_t>(n)); } while (written < 0 && errno == EINTR);
            check(written > 0, Operation::write, "checkpoint write failed"); done = static_cast<std::uint64_t>(written);
#endif
        }
        check(done > 0 && done <= n, Operation::write, "checkpoint short write failed");
        p.offset += done; bytes = bytes.subspan(static_cast<std::size_t>(done));
    }
}
void Publication::commit(const std::function<void()>& flushed) {
    auto& p = *impl_; check(!p.committed, Operation::replace, "checkpoint already published");
    if (p.options.durable) inject(Operation::sync);
    if (p.direct) { p.direct->finish(p.options.durable); p.direct.reset(); }
    else {
#ifdef _WIN32
        if (p.options.durable) check(FlushFileBuffers(p.handle) != 0, Operation::sync, "checkpoint fsync failed");
        const auto h = p.handle; p.handle = INVALID_HANDLE_VALUE;
        check(CloseHandle(h) != 0, Operation::sync, "checkpoint close failed");
#else
        if (p.options.durable) {
            int result; do { result = ::fsync(p.fd); } while (result < 0 && errno == EINTR);
            check(result == 0, Operation::sync, "checkpoint fsync failed");
        }
#if defined(POSIX_FADV_DONTNEED)
        if (p.options.drop_page_cache) (void)::posix_fadvise(p.fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
        const auto fd = p.fd; p.fd = -1;
        check(::close(fd) == 0, Operation::sync, "checkpoint close failed");
#endif
    }
    if (flushed) flushed();
    inject(Operation::replace);
#ifdef _WIN32
    check(MoveFileExW(p.temporary.c_str(), p.target.c_str(), MOVEFILE_REPLACE_EXISTING |
        (p.options.durable ? MOVEFILE_WRITE_THROUGH : 0)) != 0, Operation::replace, "checkpoint rename failed");
#else
    check(::rename(p.temporary.c_str(), p.target.c_str()) == 0, Operation::replace, "checkpoint rename failed");
#endif
    p.committed = true; // A following directory-sync error is an uncertain durable commit.
    sync_directory(parent(p.target), p.options.durable && p.options.sync_parent);
}
std::future<void> atomic_write_background(std::filesystem::path path, std::shared_ptr<const std::vector<std::byte>> bytes,
                                          Options options) {
    if (!bytes) throw std::invalid_argument("checkpoint background publication needs its bytes");
    return ::strix::core::shared_host_lane().submit(::strix::core::HostPriority::background,
        [path = std::move(path), bytes = std::move(bytes), options] { atomic_write(path, std::span<const std::byte>(*bytes), options); });
}
void atomic_write(const std::filesystem::path& p, std::span<const std::byte> bytes, Options options) {
    Publication out(p, options); out.write(bytes); out.commit();
}
void read(const std::filesystem::path& p, std::uint64_t minimum, std::uint64_t maximum,
    const std::function<std::span<std::byte>(std::uint64_t)>& allocate, DirectIo direct,
    const std::function<void(std::span<const std::byte>, std::uint64_t)>& observe, std::size_t chunk_bytes) {
    ++observed_activity;
    check(chunk_bytes > 0, Operation::read, "checkpoint zero read chunk");
    auto admitted = [&](std::uint64_t n) {
        check(n >= minimum && n <= maximum && n <= std::numeric_limits<std::size_t>::max(), Operation::read, "checkpoint size outside admission budget");
        inject(Operation::allocation); auto bytes = allocate(n);
        check(bytes.size() == n && (n == 0 || bytes.data()), Operation::allocation, "checkpoint allocation failed"); return bytes;
    };
    if (direct == DirectIo::automatic && state_file_io::unbuffered_io_enabled()) {
        state_file_io::SequentialReader in(p); check(in.open(), Operation::read, "checkpoint open failed");
        auto bytes = admitted(in.size());
        for (std::size_t at = 0; at < bytes.size();) {
            inject(Operation::read); const auto n = std::min({bytes.size() - at, state_file_io::chunk_bytes, chunk_bytes});
            governed(n);
            in.read(bytes.data() + at, n);
            if (observe) observe(bytes.subspan(at,n),at);
            at += n;
        }
        check(!in.grew(), Operation::read, "checkpoint grew during read"); return;
    }
    std::ifstream in(p, std::ios::binary | std::ios::ate);
    check(bool(in), Operation::read, "checkpoint open failed");
    const auto end = in.tellg(); check(end >= 0, Operation::read, "checkpoint size failed");
    auto bytes = admitted(static_cast<std::uint64_t>(end)); in.seekg(0);
    for (std::size_t at = 0; at < bytes.size();) {
        inject(Operation::read); const auto n = std::min({bytes.size() - at, chunk_bytes, std::size_t{1} << 26});
        governed(n);
        in.read(reinterpret_cast<char*>(bytes.data() + at), static_cast<std::streamsize>(n));
        check(bool(in), Operation::read, "checkpoint truncated during read");
        if (observe) observe(bytes.subspan(at,n),at);
        at += n;
    }
    check(in.peek() == std::ifstream::traits_type::eof(), Operation::read, "checkpoint grew during read");
    // Small observer chunks may leave prefetched bytes in the stream buffer.
    // Recheck the opened file itself so truncation cannot hide behind that buffer.
    in.clear(); in.seekg(0, std::ios::end);
    check(in.tellg() == end, Operation::read, "checkpoint length changed during read");
}
void read(const std::filesystem::path& path, std::vector<std::uint8_t>& out,
          DirectIo direct, bool drop_page_cache) {
#ifdef _WIN32
    (void)direct; (void)drop_page_cache;
    checkpoint_io::read(path, 0, std::numeric_limits<std::size_t>::max(), [&](std::uint64_t n) {
        out.resize(static_cast<std::size_t>(n)); return std::as_writable_bytes(std::span(out));
    });
#else
    ++observed_activity;
    struct File {
        int fd = -1;
        ~File() { if (fd >= 0) ::close(fd); }
    } file;
    bool is_direct = false;
#if defined(__linux__)
    if (direct == DirectIo::automatic && !state_file_io::fault_injection().refuse_direct.load()) {
        file.fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        is_direct = file.fd >= 0;
    }
#else
    (void)direct;
#endif
    if (file.fd < 0) file.fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    check(file.fd >= 0, Operation::read, "checkpoint open failed");
    struct stat st{};
    check(::fstat(file.fd, &st) == 0 && st.st_size >= 0, Operation::read, "checkpoint size failed");
    constexpr std::size_t alignment = 4096, chunk = 16U << 20;
    const auto n = static_cast<std::uint64_t>(st.st_size);
    check(n <= std::numeric_limits<std::size_t>::max() - 3 * alignment,
        Operation::allocation, "checkpoint aligned allocation overflow");
    const auto size = static_cast<std::size_t>(n);
    inject(Operation::allocation);
    out.resize(((size + alignment-1) / alignment + 2) * alignment);
    auto* base = out.data();
    const auto offset = (alignment - reinterpret_cast<std::uintptr_t>(base) % alignment) % alignment;
    auto* dst = base + offset;
    std::size_t at = 0;
    while (at < size) {
        inject(Operation::read);
        const auto want = std::min(chunk, (size-at+alignment-1)/alignment*alignment);
        governed(want);
        ssize_t got;
        do { got = ::pread(file.fd, dst+at, want, static_cast<off_t>(at)); } while (got < 0 && errno == EINTR);
        if (got < 0 && is_direct && at == 0) {
            ::close(file.fd);
            file.fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
            check(file.fd >= 0, Operation::read, "checkpoint buffered fallback open failed");
            is_direct = false; continue;
        }
        check(got > 0 && static_cast<std::size_t>(got) <= size-at,
            Operation::read, "checkpoint truncated or grew during read");
        at += static_cast<std::size_t>(got);
    }
    struct stat end{};
    check(::fstat(file.fd, &end) == 0 && end.st_size == st.st_size,
        Operation::read, "checkpoint length changed during read");
#if defined(POSIX_FADV_DONTNEED)
    if (!is_direct && drop_page_cache) (void)::posix_fadvise(file.fd, 0, 0, POSIX_FADV_DONTNEED);
#else
    (void)drop_page_cache;
#endif
    const auto fd = file.fd; file.fd = -1;
    check(::close(fd) == 0, Operation::read, "checkpoint read close failed");
    if (offset) std::memmove(base, dst, size);
    out.resize(size);
#endif
}
} // namespace strix::session::checkpoint_io
