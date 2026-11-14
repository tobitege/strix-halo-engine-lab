#include "strix/deepseek41_prefix_cache.hpp"

#include "strix/ds41_attention.hpp"
#include "strix/env_flags.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/session/checkpoint_io.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>

#if defined(__unix__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace strix::deepseek41 {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL, kFnvPrime = 1099511628211ULL;
constexpr std::uint64_t kNodeOverhead = 256;  // bookkeeping bytes counted per node and per snapshot

std::uint64_t fnv_bytes(const void* data, std::size_t n, std::uint64_t h) {
    const auto* b = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * kFnvPrime;
    return h;
}

std::uint64_t chain(std::uint64_t parent, std::span<const std::uint32_t> block) {
    std::uint64_t h = fnv_bytes(&parent, sizeof(parent), kFnvOffset);
    h = fnv_bytes(block.data(), block.size() * 4U, h);
    return h == 0U ? 1U : h;  // 0 means "no prefix"
}

double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

// ---- disk format ----
// Checksum of the files: four 64-bit lanes over 8-byte words (a bijective rotate-multiply step per word), the tail
// bytes FNV-folded, lanes and length mixed at the end. Several GB/s, so verifying a read costs little next to it.
std::uint64_t rotl(std::uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
std::uint64_t checksum(const std::uint8_t* p, std::size_t n) {
    constexpr std::uint64_t k = 0x9E3779B97F4A7C15ULL;
    std::uint64_t h[4] = {kFnvOffset, kFnvOffset ^ 1U, kFnvOffset ^ 2U, kFnvOffset ^ 3U};
    std::size_t i = 0;
    for (; i + 32U <= n; i += 32U)
        for (int l = 0; l < 4; ++l) {
            std::uint64_t w;
            std::memcpy(&w, p + i + 8U * static_cast<std::size_t>(l), 8);
            h[l] = rotl(h[l] ^ w, 31) * k;
        }
    std::uint64_t t = fnv_bytes(p + i, n - i, kFnvOffset);
    std::uint64_t out = static_cast<std::uint64_t>(n) * k;
    for (int l = 0; l < 4; ++l) out = rotl(out ^ h[l], 27) * kFnvPrime + k;
    out ^= t;
    out ^= out >> 33U;
    out *= 0xff51afd7ed558ccdULL;
    out ^= out >> 33U;
    return out;
}

constexpr char kMagicBlock[8] = {'D', 'S', '4', '1', 'P', 'C', 'B', '\0'};
constexpr char kMagicSnapshot[8] = {'D', 'S', '4', '1', 'P', 'C', 'S', '\0'};
constexpr std::uint32_t kVersion = 1U, kKindBlock = 1U, kKindSnapshot = 2U;

struct Header {
    char magic[8];
    std::uint32_t version, header_bytes;
    std::uint8_t model_id[32];
    std::uint64_t layout_hash;
    std::uint32_t block, layers;
    std::uint64_t key, parent;
    std::uint32_t depth, kind;
    std::uint64_t seq;
    std::uint64_t payload_bytes, payload_checksum;
    std::uint64_t reserved;
    std::uint64_t header_checksum;  // of the 120 bytes before it
};
static_assert(sizeof(Header) == 128U);
constexpr std::size_t kHeaderBytes = sizeof(Header);

void put_raw(std::vector<std::uint8_t>& out, const void* p, std::size_t n) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    out.insert(out.end(), b, b + n);
}
template <typename T>
void put(std::vector<std::uint8_t>& out, T v) {
    put_raw(out, &v, sizeof(T));
}

// layer layout table: per layer {owns_kv, owns_index, 0, 0, ratio (u32)}
std::vector<std::uint8_t> layout_table(const std::vector<AttentionLayerState>& layout) {
    std::vector<std::uint8_t> t;
    for (const auto& L : layout) {
        t.push_back(L.owns_kv ? 1U : 0U);
        t.push_back(L.owns_index ? 1U : 0U);
        t.push_back(0U);
        t.push_back(0U);
        put<std::uint32_t>(t, L.ratio);
    }
    return t;
}

// Fills the header (checksums included) at the front of `file` (whose first 128 bytes are reserved for it).
void seal(std::vector<std::uint8_t>& file, const char (&magic)[8], const std::array<std::uint8_t, 32>& model, std::uint64_t layout_hash, std::uint32_t block,
          std::uint32_t layers, std::uint64_t key, std::uint64_t parent, std::uint32_t depth, std::uint32_t kind, std::uint64_t seq) {
    Header h{};
    std::memcpy(h.magic, magic, 8);
    h.version = kVersion;
    h.header_bytes = static_cast<std::uint32_t>(kHeaderBytes);
    std::memcpy(h.model_id, model.data(), 32);
    h.layout_hash = layout_hash;
    h.block = block;
    h.layers = layers;
    h.key = key;
    h.parent = parent;
    h.depth = depth;
    h.kind = kind;
    h.seq = seq;
    h.payload_bytes = file.size() - kHeaderBytes;
    h.payload_checksum = checksum(file.data() + kHeaderBytes, file.size() - kHeaderBytes);
    h.header_checksum = checksum(reinterpret_cast<const std::uint8_t*>(&h), offsetof(Header, header_checksum));
    std::memcpy(file.data(), &h, kHeaderBytes);
}

// The header of `file` if it is well formed (magic of `kind`, version, checksums, size); nullopt otherwise.
bool parse_header(const std::vector<std::uint8_t>& file, std::size_t size, std::uint32_t kind, Header& h) {
    if (size < kHeaderBytes) return false;
    std::memcpy(&h, file.data(), kHeaderBytes);
    if (std::memcmp(h.magic, kind == kKindBlock ? kMagicBlock : kMagicSnapshot, 8) != 0 || h.version != kVersion || h.header_bytes != kHeaderBytes ||
        h.kind != kind)
        return false;
    if (h.header_checksum != checksum(reinterpret_cast<const std::uint8_t*>(&h), offsetof(Header, header_checksum))) return false;
    if (h.payload_bytes != size - kHeaderBytes) return false;
    return h.payload_checksum == checksum(file.data() + kHeaderBytes, size - kHeaderBytes);
}

// ---- file I/O: codecs and scheduling stay here; mechanics are shared ----
namespace io = session::checkpoint_io;
void sync_dir(const fs::path& dir) { io::sync_directory(dir); }

bool write_file(const fs::path& path, const std::vector<std::uint8_t>& data, bool drop) noexcept {
    try {
        io::Options options;
        options.permissions = 0644;
        options.drop_page_cache = drop;
        options.temporary_suffix = ".tmp"; // existing quota and restart cleanup count this extension
        io::atomic_write(path, std::as_bytes(std::span(data)), options);
        return true;
    } catch (...) {
        // Also contains allocation/filesystem errors on the existing asynchronous worker.
        // A post-replacement sync failure is visible with uncertain durability, not rollback.
        return false;
    }
}

bool read_file(const fs::path& path, std::vector<std::uint8_t>& out, std::size_t& size, bool direct) noexcept {
    try {
        // Historically buffered reads drop pages even when direct reads are disabled.
        io::read(path, out, direct ? io::DirectIo::automatic : io::DirectIo::buffered, true);
        size = out.size();
        return true;
    } catch (...) { return false; }
}

// Runs fn(i) for i in [0, n) on up to `threads` threads.
template <typename F>
void parallel_for(std::size_t n, std::uint32_t threads, F&& fn) {
    const std::size_t t = std::min<std::size_t>(std::max<std::uint32_t>(threads, 1U), n);
    if (t <= 1U) {
        for (std::size_t i = 0; i < n; ++i) fn(i);
        return;
    }
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> pool;
    for (std::size_t k = 0; k < t; ++k)
        pool.emplace_back([&] {
            for (std::size_t i; (i = next.fetch_add(1)) < n;) fn(i);
        });
    for (auto& th : pool) th.join();
}

std::string hex16(std::uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
    return b;
}

}  // namespace

// A block (and maybe a snapshot) of the disk tier: what lookups need is kept in memory, the data stay in the files.
struct PrefixCache::DiskEntry {
    std::uint64_t key = 0, parent = 0;
    std::uint32_t depth = 0, children = 0;
    std::vector<std::uint32_t> tokens;
    std::uint64_t block_bytes = 0, block_digest = 0;
    bool snapshot = false;
    std::uint64_t snap_bytes = 0, snap_digest = 0, last_use = 0, logits_fnv = 0;
    std::uint32_t next_token = 0;
};

// The explicit service directory owns this quota across all its namespaces/processes.
// Files awaiting deletion still count physically; reservations count payloads not yet written.
class DiskQuota {
public:
    DiskQuota(const fs::path& root, std::uint64_t budget, std::uint64_t queue_budget, std::uint64_t min_free)
        : root_(root), budget_(budget), queue_budget_(queue_budget), min_free_(min_free) {
#if defined(__unix__)
        fs::create_directories(root_);
        lock_ = ::open((root_ / ".strix-prefix-quota").c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
        static std::atomic<unsigned> serial{0};
        file_ = root_ / (".strix-prefix-reservation-" + std::to_string(::getpid()) + "-" +
                        std::to_string(Clock::now().time_since_epoch().count()) + "-" + std::to_string(serial++));
        reservation_ = ::open(file_.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (reservation_ >= 0) {
            if (::flock(reservation_, LOCK_EX | LOCK_NB) != 0 || !publish()) {
                ::close(reservation_); reservation_ = -1;
            }
        }
#endif
    }
    ~DiskQuota() {
#if defined(__unix__)
        if (reservation_ >= 0) { ::close(reservation_); std::error_code ec; fs::remove(file_, ec); }
        if (lock_ >= 0) ::close(lock_);
#endif
    }
    bool reserve(std::uint64_t n) {
#if defined(__unix__)
        if (lock_ < 0 || reservation_ < 0 || ::flock(lock_, LOCK_EX | LOCK_NB) != 0) return false;
        struct Unlock { int fd; ~Unlock() { ::flock(fd, LOCK_UN); } } unlock{lock_};
        if (!publish()) return false;
        std::uint64_t used = 0, queued = 0;
        std::error_code ec;
        fs::recursive_directory_iterator it(root_, ec), end;
        for (; !ec && it != end; it.increment(ec)) {
            const auto& p = it->path();
            if (it->is_symlink(ec)) { it.disable_recursion_pending(); continue; }
            if (!it->is_regular_file(ec)) continue;
            std::uint64_t bytes = 0;
            if (p.filename().string().starts_with(".strix-prefix-reservation-")) {
                const int fd = ::open(p.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
                if (fd < 0) return false;
                // An unlocked reservation belongs to a dead writer, hence has no queued payloads.
                if (::flock(fd, LOCK_EX | LOCK_NB) == 0) ::flock(fd, LOCK_UN);
                else if (::pread(fd, &bytes, sizeof(bytes), 0) != static_cast<ssize_t>(sizeof(bytes))) { ::close(fd); return false; }
                ::close(fd);
                if (bytes > UINT64_MAX - queued) return false;
                queued += bytes;
            } else if (p.extension() == ".blk" || p.extension() == ".snp" || p.extension() == ".tmp") {
                bytes = it->file_size(ec);
            }
            if (ec || bytes > budget_ || used > budget_ - bytes) return false;
            used += bytes;
        }
        if (ec || n > budget_ || used > budget_ - n) return false;
        if (queue_budget_ != 0U && (n > queue_budget_ || queued > queue_budget_ - n)) return false;
        const auto space = fs::space(root_, ec);
        if (min_free_ != 0U && (ec || space.available < min_free_ || queued > space.available - min_free_ ||
            n > space.available - min_free_ - queued)) return false;
        pending_ += n;
        if (publish()) return true;
        pending_ -= n;
#else
        (void)n;
#endif
        return false;
    }
    void release(std::uint64_t n) {
        pending_ -= n;
#if defined(__unix__)
        // A busy quota leaves an overestimate until our next publication, never an underestimate.
        if (lock_ >= 0 && ::flock(lock_, LOCK_EX | LOCK_NB) == 0) { (void)publish(); ::flock(lock_, LOCK_UN); }
#endif
    }

private:
    bool publish() {
#if defined(__unix__)
        return ::pwrite(reservation_, &pending_, sizeof(pending_), 0) == static_cast<ssize_t>(sizeof(pending_));
#else
        return false;
#endif
    }
    fs::path root_, file_;
    std::uint64_t budget_, queue_budget_, min_free_, pending_ = 0;
    int lock_ = -1, reservation_ = -1;
};

class DiskOwner {
public:
    explicit DiskOwner(const fs::path& dir) {
#if defined(__unix__)
        fd_ = ::open((dir / ".strix-prefix-owner").c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd_ >= 0 && ::flock(fd_, LOCK_EX | LOCK_NB) == 0) return;
        if (fd_ >= 0) ::close(fd_);
#else
        (void)dir;
#endif
        throw std::runtime_error("prefix disk namespace already owned or locking unavailable");
    }
    ~DiskOwner() {
#if defined(__unix__)
        ::close(fd_);
#endif
    }
private:
    int fd_ = -1;
};

// The writer thread: file writes and deletions in call order (a deletion after a write of the same file wins).
#ifdef STRIX_DS41_PREFIX_IO_TESTING
namespace prefix_io_test {
namespace {
thread_local WriterHook hook = nullptr;
thread_local void* hook_context = nullptr;
}
void set_writer_hook(WriterHook value, void* context) noexcept { hook = value; hook_context = context; }
}
#endif

struct PrefixCache::Writer {
    struct Job {
        fs::path path;
        std::shared_ptr<const std::vector<std::uint8_t>> data;  // null: delete the file
    };
    explicit Writer(bool drop, std::uint64_t budget, std::unique_ptr<DiskQuota> quota, std::unique_ptr<DiskOwner> owner)
        : drop_(drop), budget_(budget), quota_(std::move(quota)), owner_(std::move(owner)), thread_([this] { run(); }) {}
    ~Writer() {
        {
            const std::lock_guard lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }
    bool reserve(std::uint64_t n) {
        const std::lock_guard lock(mutex_);
        const auto current = queued.load();
        if (n > UINT64_MAX - current || (budget_ != 0U && (n > budget_ || current > budget_ - n))) return false;
        if (quota_ && !quota_->reserve(n)) return false;
        queued.store(current + n);
        peak.store(std::max(peak.load(), current + n));
        return true;
    }
    void release(std::uint64_t n) {
        const std::lock_guard lock(mutex_);
        queued.fetch_sub(n);
        if (quota_) quota_->release(n);
    }
    void push(Job job) {
        {
            const std::lock_guard lock(mutex_);
            queue_.push_back(std::move(job));
        }
        cv_.notify_all();
    }
    void flush() {
        std::unique_lock lock(mutex_);
        idle_.wait(lock, [&] { return queue_.empty() && !busy_; });
    }
    std::atomic<std::uint64_t> files{0}, bytes{0}, errors{0}, nanos{0}, queued{0}, peak{0};

private:
    void run() {
        for (;;) {
            Job job;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) return;
                job = std::move(queue_.front());
                queue_.pop_front();
                busy_ = true;
            }
#ifdef STRIX_DS41_PREFIX_IO_TESTING
            if (hook_) hook_(hook_context_, true, job.path, bool(job.data));
#endif
            if (job.data) {
                const auto t0 = Clock::now();
                if (write_file(job.path, *job.data, drop_)) {
                    files.fetch_add(1);
                    bytes.fetch_add(job.data->size());
                } else {
                    errors.fetch_add(1);
                }
                nanos.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count()));
            } else {
                std::error_code ec;
                fs::remove(job.path, ec);
            }
#ifdef STRIX_DS41_PREFIX_IO_TESTING
            if (hook_) hook_(hook_context_, false, job.path, bool(job.data));
#endif
            {
                const std::lock_guard lock(mutex_);
                if (job.data) {
                    queued.fetch_sub(job.data->size());
                    if (quota_) quota_->release(job.data->size());
                }
                busy_ = false;
                if (queue_.empty()) idle_.notify_all();
            }
        }
    }
    bool drop_;
    std::uint64_t budget_;
    std::unique_ptr<DiskQuota> quota_;
    std::unique_ptr<DiskOwner> owner_;
    std::mutex mutex_;
    std::condition_variable cv_, idle_;
    std::deque<Job> queue_;
    bool stop_ = false, busy_ = false;
#ifdef STRIX_DS41_PREFIX_IO_TESTING
    // Copied at construction; fault setup/observation executes on this SAME worker.
    prefix_io_test::WriterHook hook_ = prefix_io_test::hook;
    void* hook_context_ = prefix_io_test::hook_context;
#endif
    std::thread thread_;
};

std::uint64_t prefix_key(std::span<const std::uint32_t> tokens, std::uint32_t block, std::uint32_t blocks) {
    std::uint64_t key = 0;
    for (std::uint32_t k = 0; k < blocks; ++k) key = chain(key, tokens.subspan(std::size_t{k} * block, block));
    return key;
}

namespace {

std::string hex(const std::array<std::uint8_t, 32>& d) {
    static const char* digits = "0123456789abcdef";
    std::string out;
    for (const auto v : d) {
        out.push_back(digits[v >> 4U]);
        out.push_back(digits[v & 15U]);
    }
    return out;
}

// size + SHA-256 of the first and last `edge` bytes of a file ("-" when it cannot be read)
std::string file_digest(const fs::path& path, std::size_t edge) {
    std::error_code ec;
    const std::uint64_t size = fs::file_size(path, ec);
    if (ec) return "-";
    std::ifstream f(path, std::ios::binary);
    if (!f) return "-";
    std::vector<std::byte> bytes(static_cast<std::size_t>(std::min<std::uint64_t>(size, edge)));
    f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (size > edge) {
        std::vector<std::byte> tail(static_cast<std::size_t>(std::min<std::uint64_t>(size - edge, edge)));
        f.seekg(static_cast<std::streamoff>(size - tail.size()));
        f.read(reinterpret_cast<char*>(tail.data()), static_cast<std::streamsize>(tail.size()));
        bytes.insert(bytes.end(), tail.begin(), tail.end());
    }
    return std::to_string(size) + ":" + hex(sha256_bytes(bytes).bytes);
}

// a file, or every regular file of a directory (sorted by name: name, size, digest of the first 1 MiB)
std::string path_digest(const std::string& spec) {
    std::error_code ec;
    const fs::path p(spec);
    if (fs::is_directory(p, ec)) {
        std::vector<fs::path> files;
        for (const auto& de : fs::directory_iterator(p, ec))
            if (de.is_regular_file()) files.push_back(de.path());
        std::sort(files.begin(), files.end());
        std::string all;
        for (const auto& f : files) all += f.filename().string() + "=" + file_digest(f, std::size_t{1} << 20U) + ",";
        const auto* b = reinterpret_cast<const std::byte*>(all.data());
        return "dir:" + std::to_string(files.size()) + ":" + hex(sha256_bytes(std::span<const std::byte>(b, all.size())).bytes);
    }
    return file_digest(p, std::size_t{16} << 20U);
}

std::string env_or(const char* name, const char* fallback = "") {
    const char* v = std::getenv(name);
    return v != nullptr && *v != 0 ? std::string(v) : std::string(fallback);
}

}  // namespace

PrefixNamespace prefix_namespace_from_env(bool expert_tp_decode) {
    // bump the version when the meaning of a state changes without any of the inputs below changing
    std::string t = "ds41-prefix-ns=1;";
    const auto overlay = env_or("STRIX_DS41_OVERLAY");
    t += "overlay=" + (overlay.empty() ? std::string("-") : path_digest(overlay)) + ";";
    t += "cb4=" + std::string(!overlay.empty() && strix::env_flag_default_off("STRIX_DS41_CB4") ? "1" : "0") + ";";
    const auto trellis = env_or("STRIX_DS41_TRELLIS");
    const auto sites = env_or("STRIX_DS41_TRELLIS_SITES");
    const auto mix = env_or("STRIX_DS41_TRELLIS_MIX");
    const bool trellis_on = !trellis.empty() && (!sites.empty() || !mix.empty());
    t += "trellis=" + (trellis_on ? path_digest(trellis) : std::string("-")) + ";";
    t += "trellis_sites=" + (trellis_on ? sites : std::string("-")) + ";";
    t += "trellis_mix=" + (trellis_on && !mix.empty() ? (fs::exists(mix) ? path_digest(mix) : mix) : std::string("-")) + ";";
    t += "shexp_tp=" + std::string(trellis_on && strix::env_flag_default_off("STRIX_DS41_SHEXP_TP") ? "1" : "0") + ";";
    const auto ep_map = env_or("STRIX_DS41_EP_MAP");
    t += "ep_map=" + (ep_map.empty() ? std::string("-") : path_digest(ep_map)) + ";";
    t += "prefill=ep;";
    // Approximate prefill states must never alias the incumbent namespace.
    if (strix::env_flag_default_off("STRIX_DS41_PREFILL_BF16_ATTN")) t += "prefill_attention=bf16-hilo-v1;";
    else if (strix::env_flag_default_off("STRIX_DS41_FAST_ATTN")) t += "prefill_attention=f16wmma-v1;";
    // FAST MoE: every routed entry takes the relaxed kernels whatever the chunking and route selectors (window / sparse
    // paths are off under the lever), so its arithmetic is one identity
    if (strix::env_flag_default_off("STRIX_DS41_FAST_MOE")) t += "prefill_moe=relaxed-v1;";
    if (strix::env_flag_default_off("STRIX_DS41_FAST_I4")) {
        if (strix::env_flag_default_off("STRIX_DS41_FAST_MOE"))
            throw std::invalid_argument("FAST_I4 and FAST_MOE are distinct policies");
        const auto scope = env_or("STRIX_DS41_FAST_I4_SCOPE");
        if (!scope.empty() && scope != "all" && scope != "gu")
            throw std::invalid_argument("STRIX_DS41_FAST_I4_SCOPE must be all or gu");
        t += "prefill_moe=gguf-iu4-direct-v1:" + (scope.empty() ? std::string("all") : scope) + ";";
    }
    if (strix::env_flag_default_off("STRIX_DS41_PREFILL_FP16_DENSE")) {
        const auto picks = env_or("STRIX_DS41_FP16_CACHE");
        if (picks.empty() || !strix::env_flag_default_off("STRIX_DS41_FP16_CACHE_READONLY"))
            throw std::invalid_argument("FP16 prefill prefix cache requires a frozen solution cache (STRIX_DS41_FP16_CACHE_READONLY=1)");
        t += "prefill_dense=fp16-lt-v1:" + path_digest(picks) + ";";
    }
    // every knob that changes an approximate prefill's arithmetic (unset in the exact default: no change to its namespace)
    if (strix::env_flag_default_off("STRIX_DS41_PREFILL_BF16_ATTN") || strix::env_flag_default_off("STRIX_DS41_PREFILL_FP16_DENSE") || strix::env_flag_default_off("STRIX_DS41_FAST_ATTN"))
        for (const char* k : {"STRIX_DS41_PREFILL_APPROX_LAYERS", "STRIX_DS41_BF16_ATTN_VARIANT", "STRIX_DS41_PREFILL_FP16_PROJ",
                              "STRIX_DS41_PREFILL_FP16_MIN_ROWS", "STRIX_DS41_PREFILL_FP16_A8", "STRIX_DS41_PREFILL_FP16_FAST",
                              "STRIX_DS41_PREFILL_FP16_WSPLIT"})
            t += std::string(k + 11) + "=" + env_or(k) + ";";

    t += std::string("decode=") + (expert_tp_decode ? "ds41-expert-tp2-i1152-v1" : "ep") + ";";
    const auto* b = reinterpret_cast<const std::byte*>(t.data());
    PrefixNamespace ns;
    ns.text = t;
    ns.tag = hex(sha256_bytes(std::span<const std::byte>(b, t.size())).bytes).substr(0, 16);
    return ns;
}

std::array<std::uint8_t, 32> prefix_model_id(const std::filesystem::path& model, std::uint32_t layers, const PrefixNamespace& ns) {
    std::vector<std::byte> bytes(std::size_t{16} << 20U);
    std::ifstream f(model, std::ios::binary);
    if (!f) throw std::invalid_argument("prefix cache: cannot read " + model.string());
    f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    bytes.resize(static_cast<std::size_t>(f.gcount()));
    const std::uint64_t size = fs::file_size(model);
    const auto* s = reinterpret_cast<const std::byte*>(&size);
    bytes.insert(bytes.end(), s, s + 8);
    const auto* l = reinterpret_cast<const std::byte*>(&layers);
    bytes.insert(bytes.end(), l, l + 4);
    const auto* n = reinterpret_cast<const std::byte*>(ns.text.data());
    bytes.insert(bytes.end(), n, n + ns.text.size());
    return sha256_bytes(bytes).bytes;
}

std::vector<std::uint32_t> plan_prefill_pieces(std::uint32_t resume, std::uint32_t shared, std::uint32_t total, std::uint32_t block,
                                               std::uint32_t max_rows, std::uint32_t min_rows, bool before_tail) {
    if (block == 0U || block % 8U != 0U || max_rows < 8U) throw std::invalid_argument("plan_prefill_pieces: block a multiple of 8, max_rows >= 8");
    if (resume > total) throw std::invalid_argument("plan_prefill_pieces: resume beyond the prompt");
    min_rows = std::max<std::uint32_t>(min_rows, 1U);
    if (min_rows > max_rows / 2U) throw std::invalid_argument("plan_prefill_pieces: min_rows > max_rows / 2");
    std::vector<std::uint32_t> ends;
    std::uint32_t last_full = total / block * block;  // the last full block, unless it leaves a too short tail
    if (total - last_full != 0U && total - last_full < min_rows) last_full = last_full >= block ? last_full - block : 0U;
    if (shared > resume && shared < last_full) ends.push_back(shared);
    if (last_full > resume && last_full < total) ends.push_back(last_full);
    if (before_tail && last_full >= block && last_full - block > resume) ends.push_back(last_full - block);
    ends.push_back(total);
    if (resume == total) return {};
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    // split pieces longer than max_rows: at block boundaries when max_rows allows, else at multiples of 8
    std::vector<std::uint32_t> out;
    std::uint32_t at = resume;
    const std::uint32_t step = max_rows >= block ? max_rows / block * block : max_rows / 8U * 8U;
    for (const auto e : ends) {
        if (e != total && (e - at < min_rows || total - e < min_rows)) continue;  // would make a piece too short
        while (e - at > max_rows) {
            // a max_rows step, unless it leaves less than min_rows before e
            const std::uint32_t next = e - (at + step) >= min_rows ? at + step : (e - min_rows) / 8U * 8U;
            at = next;
            out.push_back(at);
        }
        out.push_back(e);
        at = e;
    }
    return out;
}

PrefillRunPlan plan_prefill_runs(std::uint32_t resume, std::uint32_t shared, std::uint32_t total, std::uint32_t block,
                                 std::uint32_t max_rows, std::uint32_t min_rows, bool before_tail, std::uint32_t reach, std::uint32_t extra) {
    if (block == 0U || block % 8U != 0U || max_rows < 8U) throw std::invalid_argument("plan_prefill_runs: block a multiple of 8, max_rows >= 8");
    if (resume > total) throw std::invalid_argument("plan_prefill_runs: resume beyond the prompt");
    min_rows = std::max<std::uint32_t>(min_rows, 1U);
    if (min_rows > max_rows / 2U) throw std::invalid_argument("plan_prefill_runs: min_rows > max_rows / 2");
    PrefillRunPlan out;
    if (resume == total) return out;
    // the runs: max_rows steps from resume (plan_prefill_pieces' split), then total
    const std::uint32_t step = max_rows / 8U * 8U;
    for (std::uint32_t at = resume; total - at > max_rows;) {
        at = total - (at + step) >= min_rows ? at + step : (total - min_rows) / 8U * 8U;
        out.ends.push_back(at);
    }
    out.ends.push_back(total);
    // the cache's boundaries (plan_prefill_pieces): inside a run, a snapshot point
    std::uint32_t last_full = total / block * block;
    if (total - last_full != 0U && total - last_full < min_rows) last_full = last_full >= block ? last_full - block : 0U;
    std::vector<std::uint32_t> cuts;
    if (shared > resume && shared < last_full) cuts.push_back(shared);
    if (last_full > resume && last_full < total) cuts.push_back(last_full);
    if (before_tail && last_full >= block && last_full - block > resume) cuts.push_back(last_full - block);
    if (const auto e = extra / block * block; e > resume && e + min_rows <= total) cuts.push_back(e);
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    for (const auto p : cuts)
        if (!std::binary_search(out.ends.begin(), out.ends.end(), p)) out.points.push_back(p);
    if (reach != 0U) {  // far points split their run (latest first: a split moves the end of the points before it)
        for (auto it = out.points.rbegin(); it != out.points.rend(); ++it) {
            const auto end = *std::upper_bound(out.ends.begin(), out.ends.end(), *it);
            if (end - *it > reach) out.ends.insert(std::upper_bound(out.ends.begin(), out.ends.end(), *it), *it);
        }
        std::erase_if(out.points, [&](std::uint32_t p) { return std::binary_search(out.ends.begin(), out.ends.end(), p); });
    }
    return out;
}

PrefillRunPlan select_prefix_prefill_plan(std::uint32_t resume, std::uint32_t shared, std::uint32_t total, std::uint32_t block,
                                          std::uint32_t max_rows, std::uint32_t min_rows, bool capture, bool inrun,
                                          bool before_tail, std::uint32_t reach, std::uint32_t stable) {
    if (!capture) return {plan_prefill_pieces(resume, 0U, total, max_rows, max_rows, min_rows), {}};
    if (inrun) return plan_prefill_runs(resume, shared, total, block, max_rows, min_rows, before_tail, reach, stable);
    return {plan_prefill_pieces(resume, shared, total, block, max_rows, min_rows, before_tail), {}};
}

PrefixCache::PrefixCache(const PrefixCacheOptions& options, std::shared_ptr<PrefixStateCache> owner) : options_(options), owner_(std::move(owner)) {
    if (options.block == 0U || options.block % 8U != 0U) throw std::invalid_argument("deepseek41 prefix cache: block must be a positive multiple of 8");
    if (!owner_) {
        PrefixStateCachePolicy policy;
        policy.host_byte_budget=options.budget_bytes ? options.budget_bytes : UINT64_MAX;
        policy.retention=PrefixCacheRetention::lru;
        owner_=std::make_shared<PrefixStateCache>(policy,[](std::uint64_t n) {
            return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[static_cast<std::size_t>(n)],std::default_delete<std::byte[]>()),n};
        });
    }
    scope_=owner_->new_scope();
    if (has_disk()) open_disk();
}

PrefixCache::~PrefixCache() {
    if (writer_) writer_->flush();
    clear_ram();
}

fs::path PrefixCache::block_path(std::uint64_t key) const { return options_.dir / ("b-" + hex16(key) + ".blk"); }
fs::path PrefixCache::snapshot_path(std::uint64_t key) const { return options_.dir / ("s-" + hex16(key) + ".snp"); }

void PrefixCache::flush() {
    if (writer_) writer_->flush();
}

void PrefixCache::clear_ram() {
    owner_->remove(scope_,[](const auto&) { return true; });
}

void PrefixCache::clear() {
    clear_ram();
    if (!writer_) return;
    for (const auto& [key, e] : disk_) {
        writer_->push({block_path(key), nullptr});
        if (e->snapshot) writer_->push({snapshot_path(key), nullptr});
    }
    disk_.clear();
    stats_.disk_bytes = stats_.disk_blocks = stats_.disk_snapshots = 0;
}

PrefixCacheStats PrefixCache::stats() const {
    auto s = stats_;
    const auto ram=owner_->chain_stats(scope_);
    s.bytes=ram.bytes; s.blocks=ram.blocks; s.snapshots=ram.snapshots;
    s.evicted_snapshots=ram.evicted_snapshots; s.evicted_blocks=ram.evicted_blocks;
    if (writer_) {
        s.written_files = writer_->files.load();
        s.written_bytes = writer_->bytes.load();
        s.write_errors = writer_->errors.load();
        s.queued_bytes = writer_->queued.load();
        s.queue_peak_bytes = writer_->peak.load();
        s.write_seconds = static_cast<double>(writer_->nanos.load()) * 1e-9;
    }
    return s;
}

void PrefixCache::set_layout(const AttentionState& s) {
    const std::uint32_t B = options_.block;
    if (!layout_.empty()) return;
    for (const auto& L : s.layers) {
        AttentionLayerState r;
        r.owns_kv = L.owns_kv;
        r.owns_index = L.owns_index;
        r.ratio = L.ratio;
        if (L.owns_kv && (L.ratio == 0U || B % L.ratio != 0U)) throw std::invalid_argument("deepseek41 prefix cache: block not a multiple of a compressor ratio");
        layout_.push_back(r);
    }
}

ModelContentIdentity PrefixCache::ram_id(std::uint64_t key) const {
    ModelContentIdentity id;
    std::memcpy(id.bytes.data(),&scope_,8);
    std::memcpy(id.bytes.data()+8,&key,8);
    id.bytes.back()=0xd4;
    return id;
}
std::shared_ptr<const PrefixCacheSegment> PrefixCache::ram_node(std::uint64_t key, std::uint32_t depth, std::span<const std::uint32_t> tokens) const {
    auto n=owner_->chain_segment(scope_,ram_id(key));
    if(!n || n->block_index+1!=depth || !std::equal(tokens.begin(),tokens.end(),n->tokens.begin(),n->tokens.end())) return {};
    if(depth>1 && (!n->ancestor || n->ancestor->block_index+2!=depth)) return {};
    return n;
}
std::shared_ptr<const PrefixCacheCheckpoint> PrefixCache::ram_snapshot(std::uint64_t key, bool touch) const {
    return owner_->chain_checkpoint(scope_,ram_id(key),touch);
}
const std::uint8_t* PrefixCache::ram_layer(const PrefixCacheSegment& n, std::size_t layer, bool index) const {
    std::size_t at=0;
    for(std::size_t l=0;l<layer;++l) if(layout_[l].owns_kv)
        at+=std::size_t{options_.block/layout_[l].ratio}*(ds41::attn::kKvEntryBytes+ds41::attn::kIdxEntryBytes);
    if(index) at+=std::size_t{options_.block/layout_[layer].ratio}*ds41::attn::kKvEntryBytes;
    return reinterpret_cast<const std::uint8_t*>(n.payload.data.get())+at;
}
namespace {
std::uint32_t ram_next(const PrefixCacheCheckpoint& c) { std::uint32_t v; std::memcpy(&v,c.fixed.data.get(),4); return v; }
std::uint64_t ram_logits(const PrefixCacheCheckpoint& c) { std::uint64_t v; std::memcpy(&v,c.fixed.data.get()+8,8); return v; }
}

const PrefixCache::DiskEntry* PrefixCache::disk_entry(std::uint64_t key, std::uint64_t parent, std::uint32_t depth,
                                                      std::span<const std::uint32_t> block_tokens) const {
    const auto it = disk_.find(key);
    if (it == disk_.end()) return nullptr;
    const DiskEntry& e = *it->second;
    if (e.depth != depth || e.parent != parent || !std::equal(block_tokens.begin(), block_tokens.end(), e.tokens.begin(), e.tokens.end())) return nullptr;
    return &e;
}

PrefixLookup PrefixCache::lookup(std::span<const std::uint32_t> tokens, std::uint32_t min_rows) {
    PrefixLookup out;
    ++stats_.lookups;
    const std::uint32_t B = options_.block;
    std::uint64_t key = 0, disk_bytes = 0;
    for (std::uint32_t k = 0; std::size_t{k + 1U} * B <= tokens.size(); ++k) {
        const auto piece = tokens.subspan(std::size_t{k} * B, B);
        const std::uint64_t parent = key;
        key = chain(key, piece);
        const auto n = ram_node(key, k + 1U, piece);
        const auto snap = n ? ram_snapshot(key) : nullptr;
        const DiskEntry* d = disk_.empty() ? nullptr : disk_entry(key, parent, k + 1U, piece);
        if (n == nullptr && d == nullptr) break;
        if (n == nullptr) disk_bytes += d->block_bytes;
        out.shared = (k + 1U) * B;
        const std::size_t left = tokens.size() - out.shared;
        const bool ram_snap = bool(snap);
        const bool disk_snap = d != nullptr && d->snapshot;
        const bool whole = left == 0U && (ram_snap || disk_snap) && (ram_snap ? ram_next(*snap) : d->next_token) != kPrefixUnknownNextToken;
        if ((ram_snap || disk_snap) && (whole || left >= min_rows)) {
            out.resume = out.shared;
            out.key = key;
            out.next_token = ram_snap ? ram_next(*snap) : d->next_token;
            out.logits_fnv = ram_snap ? ram_logits(*snap) : d->logits_fnv;
            out.disk_bytes = disk_bytes + (ram_snap ? 0U : d->snap_bytes);
            out.disk = out.disk_bytes != 0U;
        }
    }
    if (out.resume > 0U) {
        ++stats_.hits;
        stats_.hit_tokens += out.resume;
    }
    return out;
}

std::uint32_t PrefixCache::covered(std::span<const std::uint32_t> tokens, std::uint32_t position) const {
    const std::uint32_t B = options_.block;
    std::uint64_t key = 0;
    std::uint32_t out = 0;
    for (std::uint32_t k = 0; std::size_t{k + 1U} * B <= std::min<std::size_t>(position, tokens.size()); ++k) {
        const auto piece = tokens.subspan(std::size_t{k} * B, B);
        const std::uint64_t parent = key;
        key = chain(key, piece);
        if (ram_node(key, k + 1U, piece) == nullptr) break;
        if (has_disk() && disk_entry(key, parent, k + 1U, piece) == nullptr) break;
        out = (k + 1U) * B;
    }
    return out;
}

AttentionState PrefixCache::load(std::span<const std::uint32_t> tokens, std::uint32_t position, std::uint64_t key) {
    const std::uint32_t B = options_.block;
    const auto fail = [](const std::string& what) { throw std::runtime_error("deepseek41 prefix cache: " + what); };
    last_load_ = {};
    if (position == 0U || position % B != 0U || position > tokens.size()) fail("state: not a block boundary of the prompt");
    const std::uint32_t D = position / B;
    // the path: per depth the RAM node, else the disk entry
    std::vector<std::shared_ptr<const PrefixCacheSegment>> ram(D);
    std::vector<const DiskEntry*> dsk(D, nullptr);
    std::uint64_t k_key = 0;
    for (std::uint32_t k = 0; k < D; ++k) {
        const auto piece = tokens.subspan(std::size_t{k} * B, B);
        const std::uint64_t parent = k_key;
        k_key = chain(k_key, piece);
        ram[k] = ram_node(k_key, k + 1U, piece);
        if (ram[k] == nullptr) dsk[k] = disk_.empty() ? nullptr : disk_entry(k_key, parent, k + 1U, piece);
        if (ram[k] == nullptr && dsk[k] == nullptr) fail("state: missing block " + std::to_string(k));
    }
    if (k_key != key) fail("state: the key does not match the prompt");
    const auto snap = ram[D-1U] ? ram_snapshot(key) : nullptr;
    const bool ram_snap = bool(snap);
    const DiskEntry* dleaf = nullptr;
    if (!ram_snap) {
        const auto it = disk_.find(key);
        if (it == disk_.end() || !it->second->snapshot || it->second->depth != D) fail("state: no snapshot at the boundary");
        dleaf = it->second.get();
    }
    if (layout_.empty()) fail("state: empty cache");
    const std::size_t layers = layout_.size();
    // per KV-source layer: offsets of its KV and index-key entries in a block payload
    const std::size_t table = layers * 8U;
    std::vector<std::size_t> kv_off(layers, 0), idx_off(layers, 0);
    {
        std::size_t at = table + std::size_t{B} * 4U;
        for (std::size_t l = 0; l < layers; ++l) {
            if (!layout_[l].owns_kv) continue;
            const std::size_t per = B / layout_[l].ratio;
            kv_off[l] = at;
            at += per * ds41::attn::kKvEntryBytes;
            idx_off[l] = at;
            at += per * ds41::attn::kIdxEntryBytes;
        }
    }
    // read the disk pieces (in parallel), verify
    struct Read {
        std::uint32_t depth = 0;  // 0: the snapshot
        std::vector<std::uint8_t> data;
        std::size_t size = 0;
        bool ok = false;
    };
    std::vector<Read> reads;
    for (std::uint32_t k = 0; k < D; ++k)
        if (dsk[k] != nullptr) reads.push_back({k + 1U, {}, 0, false});
    if (dleaf != nullptr) reads.push_back({0U, {}, 0, false});
    if (!reads.empty()) {
        flush();
        const auto t1 = Clock::now();
        const std::uint64_t layout_hash = [&] {
            const auto t = layout_table(layout_);
            return checksum(t.data(), t.size());
        }();
        const auto expect_table = layout_table(layout_);
        parallel_for(reads.size(), options_.io_threads, [&](std::size_t i) {
            auto& r = reads[i];
            const bool is_snap = r.depth == 0U;
            const DiskEntry& e = is_snap ? *dleaf : *dsk[r.depth - 1U];
            if (!read_file(is_snap ? snapshot_path(e.key) : block_path(e.key), r.data, r.size, options_.drop_page_cache)) return;
            Header h{};
            if (!parse_header(r.data, r.size, is_snap ? kKindSnapshot : kKindBlock, h)) return;
            if (std::memcmp(h.model_id, options_.model_id.data(), 32) != 0 || h.layout_hash != layout_hash || h.block != B || h.layers != layers ||
                h.key != e.key || h.depth != e.depth || h.parent != e.parent)
                return;
            if (h.payload_bytes < table || std::memcmp(r.data.data() + kHeaderBytes, expect_table.data(), table) != 0) return;
            if (!is_snap) {
                if (r.size != e.block_bytes) return;
                if (std::memcmp(r.data.data() + kHeaderBytes + table, e.tokens.data(), std::size_t{B} * 4U) != 0) return;
                if (h.payload_checksum != e.block_digest) return;
            } else if (h.payload_checksum != e.snap_digest) {
                return;
            }
            r.ok = true;
        });
        last_load_.disk_seconds = since(t1);
        for (const auto& r : reads) {
            last_load_.disk_bytes += r.size;
            ++last_load_.disk_files;
        }
        stats_.read_files += last_load_.disk_files;
        stats_.read_bytes += last_load_.disk_bytes;
        stats_.read_seconds += last_load_.disk_seconds;
        for (const auto& r : reads)
            if (!r.ok) {
                ++stats_.read_errors;
                throw PrefixCacheReadError("deepseek41 prefix cache: disk " + std::string(r.depth == 0U ? "snapshot" : "block") + " " +
                                           (r.depth == 0U ? snapshot_path(dleaf->key) : block_path(dsk[r.depth - 1U]->key)).string() +
                                           " is missing or does not verify");
            }
    }
    const auto t2 = Clock::now();
    std::vector<const std::uint8_t*> disk_payload(D, nullptr);
    const Read* snap_read = nullptr;
    for (const auto& r : reads) {
        if (r.depth == 0U) snap_read = &r;
        else disk_payload[r.depth - 1U] = r.data.data() + kHeaderBytes;
    }
    AttentionState s;
    s.position = position;
    s.tokens.assign(tokens.begin(), tokens.begin() + position);
    s.layers.resize(layers);
    // snapshot pieces
    std::vector<const std::uint8_t*> ring_src(layers, nullptr), carry_src(layers, nullptr);
    std::vector<std::size_t> ring_n(layers, 0), carry_n(layers, 0);
    std::vector<char> carry_valid(layers, 0);
    {
        const std::uint8_t* p = ram_snap ? reinterpret_cast<const std::uint8_t*>(snap->fixed.data.get()) : snap_read->data.data() + kHeaderBytes + table;
        const std::uint8_t* end = ram_snap ? p + snap->fixed.bytes : snap_read->data.data() + snap_read->size;
        const auto get = [&](void* dst, std::size_t n) {
            if (static_cast<std::size_t>(end - p) < n) throw PrefixCacheReadError("deepseek41 prefix cache: snapshot file too short");
            std::memcpy(dst, p, n);
            p += n;
        };
        std::uint32_t next = 0, pos = 0;
        std::uint64_t lf = 0;
        get(&next, 4);
        get(&pos, 4);
        get(&lf, 8);
        if (pos != position) throw PrefixCacheReadError("deepseek41 prefix cache: snapshot position differs");
        s.next_token = next;
        for (std::size_t l = 0; l < layers; ++l) {
            std::uint64_t rn = 0, cn = 0, cv = 0;
            get(&rn, 8);
            get(&cn, 8);
            get(&cv, 8);
            if ((!ram_snap && (rn > (std::uint64_t{1} << 30U) || cn > (std::uint64_t{1} << 20U))) ||
                static_cast<std::uint64_t>(end - p) < rn + cn * 4U)
                throw PrefixCacheReadError("deepseek41 prefix cache: snapshot sizes");
            ring_src[l] = p;
            ring_n[l] = rn;
            p += rn;
            carry_src[l] = p;
            carry_n[l] = cn;
            p += cn * 4U;
            carry_valid[l] = cv != 0U ? 1 : 0;
        }
    }
    for (std::size_t l = 0; l < layers; ++l) {
        auto& L = s.layers[l];
        L.owns_kv = layout_[l].owns_kv;
        L.owns_index = layout_[l].owns_index;
        L.ratio = layout_[l].ratio;
        L.topk_known = false;
        L.ring.assign(ring_src[l], ring_src[l] + ring_n[l]);
        if (!L.owns_kv) continue;
        const std::size_t per = B / L.ratio, kvb = per * ds41::attn::kKvEntryBytes, ixb = per * ds41::attn::kIdxEntryBytes;
        L.kv.resize(kvb * D);
        L.idx_keys.resize(ixb * D);
        for (std::uint32_t k = 0; k < D; ++k) {
            if (ram[k] != nullptr) {
                std::memcpy(L.kv.data() + kvb * k, ram_layer(*ram[k],l,false), kvb);
                std::memcpy(L.idx_keys.data() + ixb * k, ram_layer(*ram[k],l,true), ixb);
            } else {
                std::memcpy(L.kv.data() + kvb * k, disk_payload[k] + kv_off[l], kvb);
                std::memcpy(L.idx_keys.data() + ixb * k, disk_payload[k] + idx_off[l], ixb);
            }
        }
        L.carry.resize(carry_n[l]);
        if (carry_n[l] != 0U) std::memcpy(L.carry.data(), carry_src[l], carry_n[l] * 4U);
        L.carry_valid = carry_valid[l] != 0;
    }
    last_load_.assemble_seconds = since(t2);
    return s;
}

void PrefixCache::use(const AttentionState& loaded, std::uint64_t key) {
    const std::uint32_t B = options_.block;
    const std::uint32_t D = loaded.position / B;
    const std::uint64_t now = ++clock_;
    bool from_disk = false;
    std::uint64_t logits = 0;
    const auto snap=ram_snapshot(key);
    if(snap && snap->position/B==D) {
        (void)ram_snapshot(key,true);
        logits=ram_logits(*snap);
    }
    else from_disk=true;
    if (const auto d = disk_.find(key); d != disk_.end() && d->second->snapshot) {
        d->second->last_use = now;
        if (from_disk) logits = d->second->logits_fnv;
    }
    // a block read from disk?
    if (!from_disk) {
        std::uint64_t k_key = 0;
        for (std::uint32_t k = 0; k < D && !from_disk; ++k) {
            const auto piece = std::span<const std::uint32_t>(loaded.tokens).subspan(std::size_t{k} * B, B);
            k_key = chain(k_key, piece);
            if (ram_node(k_key, k + 1U, piece) == nullptr) from_disk = true;
        }
    }
    if (!from_disk) return;
    ++stats_.disk_hits;
    if (!options_.promote) return;
    std::vector<std::uint64_t> keys(D);
    std::uint64_t k_key = 0;
    for (std::uint32_t k = 0; k < D; ++k) keys[k] = k_key = chain(k_key, std::span<const std::uint32_t>(loaded.tokens).subspan(std::size_t{k} * B, B));
    (void)insert_ram(loaded, logits, 0U, keys);
}

AttentionState PrefixCache::state(std::span<const std::uint32_t> tokens, std::uint32_t position, std::uint64_t key) {
    auto s = load(tokens, position, key);
    use(s, key);
    return s;
}

std::uint64_t PrefixCache::snapshot_bytes(const AttentionState& s, std::uint32_t block) {
    if (block == 0U) throw std::invalid_argument("snapshot_bytes: block must be positive");
    const auto blocks = (s.tokens.size() + block - 1U) / block;
    std::uint64_t n = kNodeOverhead + s.tokens.size() * 4U + blocks * (kNodeOverhead + s.layers.size() * (2U * sizeof(std::vector<std::uint8_t>) + 64U));
    n += s.layers.size() * (sizeof(std::vector<std::uint8_t>) + sizeof(std::vector<float>) + 65U);
    for (const auto& L : s.layers) n += L.ring.size() + L.kv.size() + L.idx_keys.size() + L.carry.size() * 4U;
    return n;
}

bool PrefixCache::insert(const AttentionState& s, std::uint64_t logits_fnv, std::uint32_t kv_from) {
    namespace attn = ds41::attn;
    const std::uint32_t B = options_.block;
    if (s.position == 0U || s.position % B != 0U || s.tokens.size() != s.position || kv_from % B != 0U || kv_from > s.position) return false;
    set_layout(s);
    if (s.layers.size() != layout_.size()) throw std::invalid_argument("deepseek41 prefix cache: layer count differs from the cache's");
    for (std::size_t l = 0; l < layout_.size(); ++l) {
        const auto& L = s.layers[l];
        if (L.owns_kv != layout_[l].owns_kv || L.ratio != layout_[l].ratio) throw std::invalid_argument("deepseek41 prefix cache: layer roles differ");
        const std::size_t entries = L.owns_kv ? std::size_t{(s.position - kv_from) / L.ratio} : 0U;
        if (L.owns_kv && (L.kv.size() != entries * attn::kKvEntryBytes || L.idx_keys.size() != entries * attn::kIdxEntryBytes))
            throw std::invalid_argument("deepseek41 prefix cache: cache sizes of layer " + std::to_string(l));
    }
    const std::uint32_t blocks = s.position / B;
    std::vector<std::uint64_t> keys(blocks);
    {
        std::uint64_t key = 0;
        for (std::uint32_t k = 0; k < blocks; ++k) keys[k] = key = chain(key, std::span<const std::uint32_t>(s.tokens).subspan(std::size_t{k} * B, B));
    }
    const bool ram = insert_ram(s, logits_fnv, kv_from, keys);
    const bool disk = has_disk() && insert_disk(s, logits_fnv, kv_from, keys);
    if (ram || disk) ++stats_.inserts;
    return ram || disk;
}

bool PrefixCache::insert_ram(const AttentionState& s, std::uint64_t logits_fnv, std::uint32_t kv_from, const std::vector<std::uint64_t>& keys) {
    namespace attn = ds41::attn;
    const auto B=options_.block;
    const auto blocks=s.position/B;
    const std::span<const std::uint32_t> tokens(s.tokens);
    std::uint32_t have=0;
    std::shared_ptr<const PrefixCacheSegment> tip;
    for(std::uint32_t k=0;k<blocks;++k) {
        auto n=ram_node(keys[k],k+1,tokens.subspan(std::size_t{k}*B,B));
        if(!n) break;
        tip=std::move(n); have=k+1;
    }
    if(have*B<kv_from) { ++stats_.refused; return false; }
    for(auto k=have;k<blocks;++k) if(owner_->chain_segment(scope_,ram_id(keys[k]))) { ++stats_.refused; return false; }
    if(have==blocks && ram_snapshot(keys.back(),true)) { ++clock_; return true; }
    std::uint64_t block_payload=0;
    const auto block_metadata=kNodeOverhead+std::uint64_t{B}*4+s.layers.size()*(2*sizeof(std::vector<std::uint8_t>)+64);
    for(const auto& l:s.layers) if(l.owns_kv) block_payload+=std::uint64_t{B/l.ratio}*(attn::kKvEntryBytes+attn::kIdxEntryBytes);
    std::uint64_t fixed_payload=16+s.layers.size()*24;
    std::uint64_t snap_charge=kNodeOverhead+s.layers.size()*(sizeof(std::vector<std::uint8_t>)+sizeof(std::vector<float>)+65);
    for(const auto& l:s.layers) { fixed_payload+=l.ring.size()+l.carry.size()*4; snap_charge+=l.ring.size()+l.carry.size()*4; }
    std::vector<std::shared_ptr<PrefixCacheSegment>> fresh;
    for(auto k=have;k<blocks;++k) {
        auto n=std::make_shared<PrefixCacheSegment>();
        n->digest=ram_id(keys[k]); n->parent=k?ram_id(keys[k-1]):ram_id(0);
        n->identity=ram_id(0); n->block_index=k; n->block_rows=B;
        n->tokens.assign(tokens.begin()+std::size_t{k}*B,tokens.begin()+std::size_t{k+1}*B);
        n->ancestor=tip; n->charge=block_metadata+block_payload; n->payload.bytes=block_payload;
        tip=n; fresh.push_back(std::move(n));
    }
    auto cp=std::make_shared<PrefixCacheCheckpoint>();
    cp->identity=ram_id(0); cp->digest=ram_id(keys.back()); cp->position=s.position; cp->block_rows=B;
    cp->segments.push_back(tip); cp->charge=snap_charge; cp->fixed.bytes=fixed_payload;
    cp->retention.scope=scope_; cp->retention.max_bytes=options_.budget_bytes?options_.budget_bytes:UINT64_MAX;
    const bool ok=owner_->retain_chain(cp,[&](PrefixCacheCheckpoint& target,const PrefixCacheAllocator& allocate) {
        for(auto& n:fresh) {
            n->payload=allocate(block_payload);
            auto* out=n->payload.data.get();
            for(const auto& l:s.layers) if(l.owns_kv) {
                const auto first=(n->block_index*B-kv_from)/l.ratio;
                const auto kvb=std::size_t{B/l.ratio}*attn::kKvEntryBytes, ixb=std::size_t{B/l.ratio}*attn::kIdxEntryBytes;
                std::memcpy(out,l.kv.data()+first*attn::kKvEntryBytes,kvb); out+=kvb;
                std::memcpy(out,l.idx_keys.data()+first*attn::kIdxEntryBytes,ixb); out+=ixb;
            }
        }
        target.fixed=allocate(fixed_payload);
        auto* out=target.fixed.data.get();
        const auto put=[&](const void* p,std::size_t n) { if(n) std::memcpy(out,p,n); out+=n; };
        put(&s.next_token,4); put(&s.position,4); put(&logits_fnv,8);
        for(const auto& l:s.layers) {
            const std::uint64_t rn=l.ring.size(),cn=l.carry.size(),cv=l.carry_valid?1:0;
            put(&rn,8); put(&cn,8); put(&cv,8); put(l.ring.data(),l.ring.size()); put(l.carry.data(),l.carry.size()*4);
        }
    });
    if(ok) ++clock_; else ++stats_.refused;
    return ok;
}

bool PrefixCache::insert_disk(const AttentionState& s, std::uint64_t logits_fnv, std::uint32_t kv_from, const std::vector<std::uint64_t>& keys) {
    namespace attn = ds41::attn;
    const std::uint32_t B = options_.block;
    const std::uint32_t blocks = s.position / B;
    const std::span<const std::uint32_t> tokens(s.tokens);
    const std::size_t layers = layout_.size();
    std::uint32_t have = 0;
    for (std::uint32_t k = 0; k < blocks; ++k) {
        if (disk_entry(keys[k], k > 0U ? keys[k - 1U] : 0U, k + 1U, tokens.subspan(std::size_t{k} * B, B)) == nullptr) break;
        have = k + 1U;
    }
    for (std::uint32_t k = have; k < blocks; ++k)
        if (disk_.contains(keys[k])) {  // collision
            ++stats_.disk_refused;
            return false;
        }
    DiskEntry* leaf_existing = have == blocks ? disk_.at(keys[blocks - 1U]).get() : nullptr;
    if (leaf_existing != nullptr && leaf_existing->snapshot) {
        leaf_existing->last_use = ++clock_;
        return true;
    }
    // blocks below kv_from come from the RAM tier
    for (std::uint32_t k = have; k < blocks && k * B < kv_from; ++k)
        if (ram_node(keys[k], k + 1U, tokens.subspan(std::size_t{k} * B, B)) == nullptr) {
            ++stats_.disk_refused;
            return false;
        }
    const auto table = layout_table(layout_);
    const std::uint64_t layout_hash = checksum(table.data(), table.size());
    std::uint64_t block_file = kHeaderBytes + table.size() + std::uint64_t{B} * 4U;
    for (const auto& L : layout_)
        if (L.owns_kv) block_file += std::uint64_t{B / L.ratio} * (attn::kKvEntryBytes + attn::kIdxEntryBytes);
    std::uint64_t snap_file = kHeaderBytes + table.size() + 16U;
    for (const auto& L : s.layers) snap_file += 24U + L.ring.size() + L.carry.size() * 4U;
    const std::uint64_t need = snap_file + std::uint64_t{blocks - have} * block_file;
    // Make room in this tier before the reservation: the service-directory quota counts the files on disk, so a
    // full tier evicted only after a successful reservation was refused by its own quota forever (no write, no
    // eviction).  The decision uses only this tier's accounting and `need`, the same inputs the eviction used when it
    // ran after the admission, so the ranks evict alike whatever the resource admission below decides.  A refused
    // insert may therefore have freed up to `need` of least recently used states, which the next accepted one would
    // have freed anyway.  The deletions are written out first so the quota no longer counts them.
    if (options_.disk_budget_bytes != 0U) {
        if (need > options_.disk_budget_bytes) {
            ++stats_.disk_refused;
            return false;
        }
        const std::uint64_t room = options_.disk_budget_bytes - need;   // need <= budget: no wrap
        DiskEntry* tip = have > 0U ? disk_.at(keys[have - 1U]).get() : nullptr;
        if (tip != nullptr) ++tip->children;
        const auto evicted = stats_.disk_evicted_snapshots;
        while (stats_.disk_bytes > room && stats_.disk_snapshots > 0U) evict_disk_one();
        if (tip != nullptr) --tip->children;
        if (stats_.disk_bytes > room) {
            if (tip != nullptr) prune_disk(tip->key);
            ++stats_.disk_refused;
            return false;
        }
        if (stats_.disk_evicted_snapshots != evicted) writer_->flush();
    }
    // Reserve before allocating/serializing payloads, and keep the active write charged.  A write still in flight
    // counts twice in the quota (its reservation and its partial file) and pending deletions still count, so a refusal
    // is retried once after the writer drains.
    bool reserved = writer_->reserve(need);
    if (!reserved) {
        writer_->flush();
        reserved = writer_->reserve(need);
    }
    std::error_code space_error;
    const auto space = fs::space(options_.dir, space_error);
    bool allowed = reserved && (options_.min_free_bytes == 0U ||
        (!space_error && space.available >= options_.min_free_bytes && space.available - options_.min_free_bytes >= writer_->queued.load()));
    const bool local = allowed;
    if (options_.disk_admission) allowed = options_.disk_admission(local);
    struct Reservation {
        Writer& writer;
        std::uint64_t remaining;
        ~Reservation() { if (remaining != 0U) writer.release(remaining); }
    } reservation{*writer_, reserved ? need : 0U};
    if (!allowed || !reserved) {
        ++stats_.disk_refused;
        ++stats_.disk_resource_refused;
        std::fprintf(stderr, "prefix persistence bypass: %s\n", !reserved ? "queue or service-directory quota unavailable" :
                     !local ? "filesystem free-space floor" : "peer resource refusal");
        return false;
    }
    const std::uint64_t seq = ++clock_;
    for (std::uint32_t k = have; k < blocks; ++k) {
        auto file = std::make_shared<std::vector<std::uint8_t>>(kHeaderBytes, std::uint8_t{0});
        file->reserve(block_file);
        put_raw(*file, table.data(), table.size());
        put_raw(*file, tokens.data() + std::size_t{k} * B, std::size_t{B} * 4U);
        const auto rn = k * B < kv_from ? ram_node(keys[k], k + 1U, tokens.subspan(std::size_t{k} * B, B)) : nullptr;
        for (std::size_t l = 0; l < layers; ++l) {
            const auto& L = s.layers[l];
            if (!L.owns_kv) continue;
            const std::size_t per = B / L.ratio;
            if (rn != nullptr) {
                put_raw(*file, ram_layer(*rn,l,false), per*attn::kKvEntryBytes);
                put_raw(*file, ram_layer(*rn,l,true), per*attn::kIdxEntryBytes);
            } else {
                const std::size_t first = (std::size_t{k} * B - kv_from) / L.ratio;
                put_raw(*file, L.kv.data() + first * attn::kKvEntryBytes, per * attn::kKvEntryBytes);
                put_raw(*file, L.idx_keys.data() + first * attn::kIdxEntryBytes, per * attn::kIdxEntryBytes);
            }
        }
        const std::uint64_t parent = k > 0U ? keys[k - 1U] : 0U;
        seal(*file, kMagicBlock, options_.model_id, layout_hash, B, static_cast<std::uint32_t>(layers), keys[k], parent, k + 1U, kKindBlock, seq);
        auto e = std::make_unique<DiskEntry>();
        e->key = keys[k];
        e->parent = parent;
        e->depth = k + 1U;
        e->tokens.assign(tokens.begin() + std::size_t{k} * B, tokens.begin() + std::size_t{k + 1U} * B);
        e->block_bytes = file->size();
        std::memcpy(&e->block_digest, file->data() + offsetof(Header, payload_checksum), 8);
        stats_.disk_bytes += e->block_bytes;
        ++stats_.disk_blocks;
        if (k > 0U) ++disk_.at(parent)->children;
        disk_.emplace(keys[k], std::move(e));
        const auto queued = file->size();
        writer_->push({block_path(keys[k]), std::move(file)});
        reservation.remaining -= queued;
    }
    DiskEntry& leaf = *disk_.at(keys[blocks - 1U]);
    auto file = std::make_shared<std::vector<std::uint8_t>>(kHeaderBytes, std::uint8_t{0});
    file->reserve(snap_file);
    put_raw(*file, table.data(), table.size());
    put<std::uint32_t>(*file, s.next_token);
    put<std::uint32_t>(*file, s.position);
    put<std::uint64_t>(*file, logits_fnv);
    for (const auto& L : s.layers) {
        put<std::uint64_t>(*file, L.ring.size());
        put<std::uint64_t>(*file, L.carry.size());
        put<std::uint64_t>(*file, L.carry_valid ? 1U : 0U);
        put_raw(*file, L.ring.data(), L.ring.size());
        put_raw(*file, L.carry.data(), L.carry.size() * 4U);
    }
    seal(*file, kMagicSnapshot, options_.model_id, layout_hash, B, static_cast<std::uint32_t>(layers), leaf.key, leaf.parent, leaf.depth, kKindSnapshot, seq);
    leaf.snapshot = true;
    leaf.snap_bytes = file->size();
    std::memcpy(&leaf.snap_digest, file->data() + offsetof(Header, payload_checksum), 8);
    leaf.next_token = s.next_token;
    leaf.logits_fnv = logits_fnv;
    leaf.last_use = seq;
    stats_.disk_bytes += leaf.snap_bytes;
    ++stats_.disk_snapshots;
    const auto queued = file->size();
    writer_->push({snapshot_path(leaf.key), std::move(file)});
    reservation.remaining -= queued;
    return true;
}

void PrefixCache::drop_disk_snapshot(DiskEntry& e) {
    if (!e.snapshot) return;
    e.snapshot = false;
    stats_.disk_bytes -= e.snap_bytes;
    --stats_.disk_snapshots;
    e.snap_bytes = e.snap_digest = 0;
    writer_->push({snapshot_path(e.key), nullptr});
}

void PrefixCache::remove_disk(std::uint64_t key) {
    const auto it = disk_.find(key);
    if (it == disk_.end()) return;
    DiskEntry& e = *it->second;
    drop_disk_snapshot(e);
    if (e.depth > 1U)
        if (const auto p = disk_.find(e.parent); p != disk_.end() && p->second->children > 0U) --p->second->children;
    stats_.disk_bytes -= e.block_bytes;
    --stats_.disk_blocks;
    writer_->push({block_path(key), nullptr});
    disk_.erase(it);
}

void PrefixCache::evict_disk_one() {
    DiskEntry* victim = nullptr;
    for (auto& [key, e] : disk_)
        if (e->snapshot && (victim == nullptr || e->last_use < victim->last_use || (e->last_use == victim->last_use && e->key < victim->key))) victim = e.get();
    if (victim == nullptr) return;
    ++stats_.disk_evicted_snapshots;
    drop_disk_snapshot(*victim);
    prune_disk(victim->key);
}

void PrefixCache::prune_disk(std::uint64_t key) {
    for (;;) {
        const auto it = disk_.find(key);
        if (it == disk_.end() || it->second->snapshot || it->second->children != 0U) return;
        const std::uint64_t parent = it->second->depth > 1U ? it->second->parent : 0U;
        ++stats_.disk_evicted_blocks;
        remove_disk(key);
        if (parent == 0U) return;
        key = parent;
    }
}

// Entries whose chain to the root is broken go (with their files), then blocks no snapshot below needs.
void PrefixCache::sweep_disk() {
    std::vector<DiskEntry*> order;
    for (auto& [key, e] : disk_) order.push_back(e.get());
    std::sort(order.begin(), order.end(), [](const DiskEntry* a, const DiskEntry* b) { return a->depth != b->depth ? a->depth < b->depth : a->key < b->key; });
    std::unordered_set<std::uint64_t> valid;
    std::vector<std::uint64_t> drop;
    for (const auto* e : order) {
        bool ok = e->depth == 1U ? e->parent == 0U : valid.contains(e->parent) && disk_.at(e->parent)->depth + 1U == e->depth;
        if (ok) valid.insert(e->key);
        else drop.push_back(e->key);
    }
    for (const auto k : drop) {
        ++stats_.disk_dropped;
        remove_disk(k);
    }
    // children counts from scratch, then leaves without a snapshot (deepest first)
    for (auto& [key, e] : disk_) e->children = 0;
    for (auto& [key, e] : disk_)
        if (e->depth > 1U) ++disk_.at(e->parent)->children;
    order.clear();
    for (auto& [key, e] : disk_) order.push_back(e.get());
    std::sort(order.begin(), order.end(), [](const DiskEntry* a, const DiskEntry* b) { return a->depth != b->depth ? a->depth > b->depth : a->key < b->key; });
    std::vector<std::uint64_t> keys;
    for (const auto* e : order) keys.push_back(e->key);
    for (const auto k : keys) {
        const auto it = disk_.find(k);
        if (it != disk_.end() && !it->second->snapshot && it->second->children == 0U) {
            ++stats_.disk_dropped;
            remove_disk(k);
        }
    }
}

void PrefixCache::forget(std::span<const std::uint32_t> tokens, std::uint32_t position) {
    const std::uint32_t B = options_.block;
    std::uint64_t key = 0;
    std::vector<std::uint64_t> path;
    for (std::uint32_t k = 0; std::size_t{k + 1U} * B <= std::min<std::size_t>(position, tokens.size()); ++k) {
        key = chain(key, tokens.subspan(std::size_t{k} * B, B));
        if (disk_.contains(key)) path.push_back(key);
    }
    for (const auto k : path) {
        ++stats_.disk_dropped;
        remove_disk(k);
    }
    sweep_disk();
}

std::vector<PrefixDiskItem> PrefixCache::inventory() const {
    std::vector<PrefixDiskItem> out;
    for (const auto& [key, e] : disk_) {
        out.push_back({key, kKindBlock, e->depth, e->block_digest});
        if (e->snapshot) out.push_back({key, kKindSnapshot, e->depth, e->snap_digest});
    }
    std::sort(out.begin(), out.end(), [](const PrefixDiskItem& a, const PrefixDiskItem& b) { return a.key != b.key ? a.key < b.key : a.kind < b.kind; });
    return out;
}

std::uint64_t PrefixCache::retain(std::span<const PrefixDiskItem> peer) {
    const auto before = stats_.disk_dropped;
    const auto has = [&](const PrefixDiskItem& x) {
        const auto it = std::lower_bound(peer.begin(), peer.end(), x, [](const PrefixDiskItem& a, const PrefixDiskItem& b) {
            return a.key != b.key ? a.key < b.key : a.kind < b.kind;
        });
        return it != peer.end() && *it == x;
    };
    std::vector<std::uint64_t> drop;
    for (auto& [key, e] : disk_) {
        if (!has({key, kKindBlock, e->depth, e->block_digest})) {
            drop.push_back(key);
        } else if (e->snapshot && !has({key, kKindSnapshot, e->depth, e->snap_digest})) {
            ++stats_.disk_dropped;
            drop_disk_snapshot(*e);
        }
    }
    std::sort(drop.begin(), drop.end());
    for (const auto k : drop) {
        ++stats_.disk_dropped;
        remove_disk(k);
    }
    sweep_disk();
    return stats_.disk_dropped - before;
}

void PrefixCache::open_disk() {
    const auto t0 = Clock::now();
    std::error_code ec;
    fs::create_directories(options_.dir, ec);
    if (ec) throw std::runtime_error("deepseek41 prefix cache: cannot create " + options_.dir.string() + ": " + ec.message());
    writer_ = std::make_unique<Writer>(options_.drop_page_cache, options_.queue_budget_bytes,
        options_.quota_dir.empty() ? nullptr : std::make_unique<DiskQuota>(options_.quota_dir, options_.aggregate_disk_budget_bytes,
            options_.queue_budget_bytes, options_.min_free_bytes),
        options_.quota_dir.empty() ? nullptr : std::make_unique<DiskOwner>(options_.dir));
    struct File {
        fs::path path;
        std::uint32_t kind = 0;
        Header h{};
        std::vector<std::uint8_t> data;
        std::size_t size = 0;
        bool ok = false;
    };
    std::vector<File> files;
    for (const auto& de : fs::directory_iterator(options_.dir, ec)) {
        if (!de.is_regular_file()) continue;
        const auto name = de.path().filename().string();
        const auto ext = de.path().extension().string();
        if (ext == ".tmp") {  // an interrupted write
            fs::remove(de.path(), ec);
            ++stats_.ignored_files;
            continue;
        }
        if (ext == ".blk" && name.rfind("b-", 0) == 0) files.push_back(File{de.path(), kKindBlock, {}, {}, 0, false});
        else if (ext == ".snp" && name.rfind("s-", 0) == 0) files.push_back(File{de.path(), kKindSnapshot, {}, {}, 0, false});
    }
    std::sort(files.begin(), files.end(), [](const File& a, const File& b) { return a.path < b.path; });
    stats_.scanned_files = files.size();
    parallel_for(files.size(), options_.io_threads, [&](std::size_t i) {
        auto& f = files[i];
        if (!read_file(f.path, f.data, f.size, options_.drop_page_cache)) return;
        if (!parse_header(f.data, f.size, f.kind, f.h)) return;
        if (std::memcmp(f.h.model_id, options_.model_id.data(), 32) != 0 || f.h.block != options_.block || f.h.depth == 0U) return;
        if (f.h.payload_bytes < std::uint64_t{f.h.layers} * 8U) return;
        f.ok = true;
        if (f.kind == kKindBlock) {  // keep the tokens only
            f.data.erase(f.data.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(f.size, kHeaderBytes + f.h.layers * 8U + std::size_t{options_.block} * 4U)),
                         f.data.end());
        } else {
            f.data.resize(std::min<std::size_t>(f.size, kHeaderBytes + f.h.layers * 8U + 16U));
        }
        f.data.shrink_to_fit();
    });
    // the layout: the first valid file's table (every file must carry the same)
    std::vector<std::uint8_t> table;
    std::uint64_t layout_hash = 0;
    for (const auto& f : files)
        if (f.ok) {
            table.assign(f.data.begin() + kHeaderBytes, f.data.begin() + kHeaderBytes + f.h.layers * 8U);
            layout_hash = f.h.layout_hash;
            break;
        }
    if (!table.empty() && layout_.empty()) {
        for (std::size_t l = 0; l < table.size() / 8U; ++l) {
            AttentionLayerState r;
            r.owns_kv = table[l * 8U] != 0U;
            r.owns_index = table[l * 8U + 1U] != 0U;
            std::memcpy(&r.ratio, table.data() + l * 8U + 4U, 4);
            if (r.owns_kv && (r.ratio == 0U || options_.block % r.ratio != 0U)) {
                layout_.clear();
                table.clear();
                break;
            }
            layout_.push_back(r);
        }
    }
    std::uint64_t block_file = kHeaderBytes + table.size() + std::uint64_t{options_.block} * 4U;
    for (const auto& L : layout_)
        if (L.owns_kv) block_file += std::uint64_t{options_.block / L.ratio} * (ds41::attn::kKvEntryBytes + ds41::attn::kIdxEntryBytes);
    std::uint64_t max_seq = 0;
    for (auto& f : files) {
        if (f.ok && (table.empty() || f.h.layout_hash != layout_hash || f.h.layers * 8U != table.size() ||
                     std::memcmp(f.data.data() + kHeaderBytes, table.data(), table.size()) != 0))
            f.ok = false;
        if (f.ok && f.kind == kKindBlock && f.size != block_file) f.ok = false;
        if (!f.ok) continue;
        if (f.kind == kKindBlock) {
            if (disk_.contains(f.h.key)) {
                f.ok = false;
                continue;
            }
            auto e = std::make_unique<DiskEntry>();
            e->key = f.h.key;
            e->parent = f.h.parent;
            e->depth = f.h.depth;
            e->tokens.resize(options_.block);
            std::memcpy(e->tokens.data(), f.data.data() + kHeaderBytes + table.size(), std::size_t{options_.block} * 4U);
            if (chain(e->parent, e->tokens) != e->key) {  // the name chain must match the tokens
                f.ok = false;
                continue;
            }
            e->block_bytes = f.size;
            e->block_digest = f.h.payload_checksum;
            stats_.disk_bytes += e->block_bytes;
            ++stats_.disk_blocks;
            max_seq = std::max(max_seq, f.h.seq);
            disk_.emplace(e->key, std::move(e));
        }
    }
    for (auto& f : files) {
        if (!f.ok || f.kind != kKindSnapshot) continue;
        const auto it = disk_.find(f.h.key);
        if (it == disk_.end() || it->second->depth != f.h.depth || it->second->parent != f.h.parent || it->second->snapshot) {
            f.ok = false;
            continue;
        }
        auto& e = *it->second;
        std::uint32_t pos = 0;
        std::memcpy(&e.next_token, f.data.data() + kHeaderBytes + table.size(), 4);
        std::memcpy(&pos, f.data.data() + kHeaderBytes + table.size() + 4U, 4);
        std::memcpy(&e.logits_fnv, f.data.data() + kHeaderBytes + table.size() + 8U, 8);
        if (pos != e.depth * options_.block) {
            f.ok = false;
            continue;
        }
        e.snapshot = true;
        e.snap_bytes = f.size;
        e.snap_digest = f.h.payload_checksum;
        e.last_use = f.h.seq;
        max_seq = std::max(max_seq, f.h.seq);
        stats_.disk_bytes += e.snap_bytes;
        ++stats_.disk_snapshots;
    }
    for (const auto& f : files)
        if (!f.ok) {  // corrupt, another model or layout, a duplicate, a snapshot without its block
            ++stats_.ignored_files;
            fs::remove(f.path, ec);
        }
    const auto dropped = stats_.disk_dropped;
    sweep_disk();
    stats_.ignored_files += stats_.disk_dropped - dropped;
    stats_.disk_dropped = dropped;
    clock_ = std::max(clock_, max_seq);
    writer_->flush();
    sync_dir(options_.dir);
    stats_.scan_seconds = since(t0);
}

}  // namespace strix::deepseek41
