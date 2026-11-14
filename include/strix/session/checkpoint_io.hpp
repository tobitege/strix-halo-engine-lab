#pragma once
// Host-only publication mechanics. Callers own codecs, admission and rank protocols.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::session::checkpoint_io {
enum class LockMode { blocking, nonblocking };
// platform_default preserves the broad default lock and one-byte named locks.
enum class LockRange { platform_default, first_byte };
enum class DirectIo { buffered, automatic };
struct Options {
    bool durable = true;
    DirectIo direct = DirectIo::buffered;
    unsigned permissions = 0600;
    std::string temporary_prefix; // optional legacy accounting name, within target directory
    bool sync_parent = true; // a grouped adopter may sync all payload names before its manifest
    bool drop_page_cache = false; // POSIX buffered publication: advise after sync, before close
    std::string temporary_suffix; // optional accounting extension (for example .tmp)
};
enum class Operation { none, allocation, create, read, write, sync, replace, directory_sync };
class Error : public std::runtime_error {
public:
    Error(Operation op, const char* message, bool busy = false) : std::runtime_error(message), operation(op), lock_busy(busy) {}
    Operation operation;
    bool lock_busy; // Only an actual nonblocking lock conflict, never other IO failures.
};
// Thread-local, deterministic host-test seams. Defaults perform real platform IO.
struct Faults {
    Operation fail = Operation::none;
    std::uint64_t max_write = 0, fail_write_at = UINT64_MAX;
    unsigned skip = 0; // matching operations before the injected failure
};
Faults& faults() noexcept;
// Thread-local observation of entry into shared filesystem mechanics (no reset).
std::uint64_t activity() noexcept;
// Overflow-safe byte framing shared by the legacy readers; no allocation.
std::span<const std::byte> take(std::span<const std::byte>, std::size_t& cursor, std::uint64_t size);
void create_directories(const std::filesystem::path&, bool durable);
void sync_directory(const std::filesystem::path&, bool durable = true);
void sync_existing(const std::filesystem::path&);
// Atomically expose an unpublished directory without replacing any existing name.
// Same physical parent only; sync failure after rename is an uncertain commit.
void publish_directory(const std::filesystem::path& temporary, const std::filesystem::path& target,
                       bool durable = true);
class DirectoryLock {
public:
    // Default callers retain .session.lock and its platform lock range. Named
    // locks use a validated basename and the legacy one-byte Windows range.
    // Every lock inode is persistent: never unlink/recreate it during cleanup.
    DirectoryLock(const std::filesystem::path&, LockMode, const std::string& basename = ".session.lock",
                  LockRange = LockRange::platform_default);
    ~DirectoryLock();
    DirectoryLock(const DirectoryLock&) = delete;
    DirectoryLock& operator=(const DirectoryLock&) = delete;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// No implicit directory lock: adopters can publish while holding their protocol lock.
// Exclusive same-directory temporary, full writes, close, replacement, directory sync.
class Publication {
public:
    explicit Publication(const std::filesystem::path& target, Options = {});
    ~Publication();
    Publication(const Publication&) = delete;
    Publication& operator=(const Publication&) = delete;
    void write(std::span<const std::byte>);
    // Callback runs after contents are flushed/closed, BEFORE visibility changes.
    void commit(const std::function<void()>& flushed = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
void atomic_write(const std::filesystem::path&, std::span<const std::byte>, Options = {});
// The same publication as a background job of the process-wide host lane. Every publication and read in this file
// draws the bytes it moves from the process-wide byte budget, which is capped while a session decodes.
std::future<void> atomic_write_background(std::filesystem::path, std::shared_ptr<const std::vector<std::byte>>, Options = {});
// Allocate only after the open file's length passes admission. The returned span must
// have exactly that length. Read/growth failures never return a partially valid record.
void read(const std::filesystem::path&, std::uint64_t minimum, std::uint64_t maximum,
          const std::function<std::span<std::byte>(std::uint64_t)>& allocate,
          DirectIo = DirectIo::buffered,
          const std::function<void(std::span<const std::byte>, std::uint64_t)>& observe = {},
          std::size_t chunk_bytes = 1U << 26);
// Whole-file vector reader. automatic tries direct IO independently of the session
// environment switch, into an aligned window of the destination vector; no staging
// pool or read-ahead thread. Buffered fallback can advise away its clean pages.
// On platforms without direct IO this remains buffered. Failure throws; no partial
// record is returned. Existing callback-reader policies and defaults are unchanged.
void read(const std::filesystem::path&, std::vector<std::uint8_t>&,
          DirectIo, bool drop_page_cache);
} // namespace strix::session::checkpoint_io
