#include "strix/file_snapshot.hpp"
#include "strix/ds41_engram.hpp"

#include "strix/ds41_formats.hpp"
#include "core/direct_row_fetch.hpp"
#include "strix/env_flags.hpp"

#include <chrono>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <fstream>
#include <unordered_map>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <malloc.h>
#include <sys/stat.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strix::ds41 {
namespace {

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

constexpr std::uint64_t page_size = 4096;

// Positional read (pread; thread-safe on a shared descriptor) and page-aligned buffers.
#if defined(_WIN32)
// Native Windows: the descriptor wraps a file handle opened with CreateFileW; FILE_FLAG_NO_BUFFERING is the
// O_DIRECT analogue (offsets, lengths and buffers page aligned, as the direct path reads whole pages), and
// ReadFile with an OVERLAPPED offset on a synchronous handle is a positional read like pread.
int open_rows_file(const std::filesystem::path& file, bool direct) {
    const DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS | (direct ? FILE_FLAG_NO_BUFFERING : 0U);
    const HANDLE h = ::CreateFileW(file.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, flags, nullptr);
    if (h == INVALID_HANDLE_VALUE) return -1;
    const int fd = ::_open_osfhandle(reinterpret_cast<intptr_t>(h), _O_RDONLY | _O_BINARY);
    if (fd < 0) ::CloseHandle(h);
    return fd;
}
long long read_at(int fd, std::uint8_t* out, std::size_t bytes, std::uint64_t offset) {
    OVERLAPPED at{};
    at.Offset = static_cast<DWORD>(offset & 0xffffffffU);
    at.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    DWORD got = 0;
    const auto handle = reinterpret_cast<HANDLE>(::_get_osfhandle(fd));
    if (!::ReadFile(handle, out, static_cast<DWORD>(bytes), &got, &at)) return -1;
    return static_cast<long long>(got);
}
std::uint8_t* alloc_pages(std::size_t bytes) { return static_cast<std::uint8_t*>(::_aligned_malloc(bytes, page_size)); }
void free_pages(std::uint8_t* p) { ::_aligned_free(p); }
void close_fd(int fd) { ::_close(fd); }
#else
long long read_at(int fd, std::uint8_t* out, std::size_t bytes, std::uint64_t offset) {
    return static_cast<long long>(::pread(fd, out, bytes, static_cast<off_t>(offset)));
}
std::uint8_t* alloc_pages(std::size_t bytes) {
    void* raw = nullptr;
    return ::posix_memalign(&raw, page_size, bytes) == 0 ? static_cast<std::uint8_t*>(raw) : nullptr;
}
void free_pages(std::uint8_t* p) { std::free(p); }
void close_fd(int fd) { ::close(fd); }
#endif

} // namespace

std::uint64_t engram_payload_cache_budget() {
    const char* flag = std::getenv("STRIX_DS41_ENGRAM_PAYLOAD_CACHE");
    if (flag != nullptr && std::strcmp(flag, "0") == 0) return 0;  // default on; =0: off
    std::ifstream info("/proc/meminfo");
    std::string key, unit;
    std::uint64_t kib = 0, available = 0;
    while (info >> key >> kib >> unit) if (key == "MemAvailable:") { available = kib * 1024U; break; }
    std::uint64_t budget = std::min<std::uint64_t>(available / 64U, 256ULL << 20U);
    if (const char* v = std::getenv("STRIX_DS41_ENGRAM_PAYLOAD_CACHE_MIB")) {
        char* end = nullptr;
        errno = 0;
        const auto mib = std::strtoull(v, &end, 10);
        if (errno != 0 || end == v || *end != 0 || *v == '-' || mib > 256U) return 0;
        budget = std::min<std::uint64_t>(budget, mib << 20U);
    }
    return budget;
}

// One open reader owns the file identity. Model files must remain immutable for its lifetime,
// as for resident weights. Absolute byte offsets also deduplicate overlapping table aliases.
// Fixed open-addressed index: no per-row allocations or tombstones, and all storage fits budget.
struct EngramRowReader::PayloadCache {
    struct Entry {
        std::uint64_t offset = 0;
        std::array<std::uint8_t, engram_row_bytes> bytes{};
        bool referenced = false;
    };
    std::vector<Entry> entries;
    std::vector<std::uint32_t> index;  // slot + 1; zero = empty
    std::size_t used = 0, hand = 0;
    std::uint64_t evictions = 0;

    explicit PayloadCache(std::uint64_t budget) {
        // At most four index words per entry after rounding the table up to a power of two.
        const std::size_t capacity = static_cast<std::size_t>(std::min<std::uint64_t>(budget / (sizeof(Entry) + 16U), 1U << 24U));
        if (capacity == 0) return;
        std::size_t size = 1;
        while (size < 2U * capacity) size *= 2U;
        entries.resize(capacity);
        index.resize(size);
    }
    std::size_t bucket(std::uint64_t offset) const {
        // Mix all offset bits; 264-byte strides otherwise leave the low bits unbalanced.
        offset ^= offset >> 30U; offset *= 0xbf58476d1ce4e5b9ULL;
        offset ^= offset >> 27U; offset *= 0x94d049bb133111ebULL;
        offset ^= offset >> 31U;
        return static_cast<std::size_t>(offset) & (index.size() - 1U);
    }
    bool get(std::uint64_t offset, std::uint8_t* out) {
        if (entries.empty()) return false;
        std::size_t p = bucket(offset);
        while (index[p] != 0U) {
            auto& e = entries[index[p] - 1U];
            if (e.offset == offset) {
                e.referenced = true;
                std::memcpy(out, e.bytes.data(), engram_row_bytes);
                return true;
            }
            p = (p + 1U) & (index.size() - 1U);
        }
        return false;
    }
    void erase(std::uint64_t offset) {
        const auto mask = index.size() - 1U;
        std::size_t hole = bucket(offset);
        while (entries[index[hole] - 1U].offset != offset) hole = (hole + 1U) & mask;
        std::size_t p = (hole + 1U) & mask;
        while (index[p] != 0U) {
            const auto home = bucket(entries[index[p] - 1U].offset);
            if (((hole - home) & mask) < ((p - home) & mask)) {
                index[hole] = index[p];
                hole = p;
            }
            p = (p + 1U) & mask;
        }
        index[hole] = 0U;
    }
    void put(std::uint64_t offset, const std::uint8_t* bytes) {
        if (entries.empty()) return;
        std::size_t slot = used;
        if (used < entries.size()) ++used;
        else {
            while (entries[hand].referenced) {
                entries[hand].referenced = false;
                hand = (hand + 1U) % entries.size();
            }
            slot = hand;
            hand = (hand + 1U) % entries.size();
            erase(entries[slot].offset);
            ++evictions;
        }
        auto& e = entries[slot];
        e.offset = offset; e.referenced = true;
        std::memcpy(e.bytes.data(), bytes, engram_row_bytes);
        std::size_t p = bucket(offset);
        while (index[p] != 0U) p = (p + 1U) & (index.size() - 1U);
        index[p] = static_cast<std::uint32_t>(slot + 1U);
    }
};

EngramHasher::EngramHasher(const deepseek41::EngramContract& c) : contract_(c) {
    const std::size_t layers = c.layer_ids.size(), cols = c.rows_per_token();
    require(c.max_ngram_size >= 2 && c.n_heads >= 1 && !c.layer_ids.empty() &&
                c.primes.size() == layers * cols && c.multipliers.size() == layers * c.max_ngram_size &&
                c.pad_id < c.token_map.size(),
            "Engram contract geometry");
    pad_ = c.token_map[c.pad_id];
    offsets_.resize(layers * cols);
    for (std::size_t e = 0; e < layers; ++e) {
        std::uint64_t sum = 0;
        for (std::size_t k = 0; k < cols; ++k) {
            offsets_[e * cols + k] = sum;
            sum += c.primes[e * cols + k];
        }
        require(sum == c.rows[e], "Engram primes do not sum to the table rows");
    }
}

void EngramHasher::push(std::uint32_t token, std::span<std::uint64_t> rows) {
    const auto& c = contract_;
    const std::size_t layers = c.layer_ids.size(), cols = c.rows_per_token(), n = c.max_ngram_size;
    const bool dead = (token & kEngramDeadBit) != 0U;
    token &= ~kEngramDeadBit;
    require(token < c.token_map.size() && rows.size() == layers * cols, "Engram hash request");
    history_.push_back(dead ? kDead : c.token_map[token]);
    const std::size_t t = history_.size() - 1;
    std::vector<std::uint64_t> ids(n);
    bool blocked = false;
    for (std::size_t s = 0; s < n; ++s) {
        blocked = blocked || t < s || history_[t - s] == kDead;  // nothing reads through a dead (image) position
        ids[s] = blocked ? pad_ : history_[t - s];
    }
    for (std::size_t e = 0; e < layers; ++e) {
        std::uint64_t rolling = ids[0] * c.multipliers[e * n];
        for (std::size_t s = 1; s < n; ++s) {
            rolling ^= ids[s] * c.multipliers[e * n + s];
            for (std::uint32_t h = 0; h < c.n_heads; ++h) {
                const std::size_t k = (s - 1) * c.n_heads + h;
                rows[e * cols + k] = rolling % c.primes[e * cols + k] + offsets_[e * cols + k];
            }
        }
    }
}

std::vector<std::uint64_t> EngramHasher::hash_sequence(std::span<const std::uint32_t> tokens) {
    reset();
    const std::size_t per = static_cast<std::size_t>(layers()) * columns();
    std::vector<std::uint64_t> out(tokens.size() * per);
    for (std::size_t i = 0; i < tokens.size(); ++i) push(tokens[i], std::span<std::uint64_t>(out.data() + i * per, per));
    return out;
}

void EngramHasher::truncate(std::uint64_t positions) {
    require(positions <= history_.size(), "Engram history cannot grow by truncation");
    history_.resize(positions);
}

EngramRowReader::EngramRowReader(const std::filesystem::path& file, std::vector<EngramTable> tables, unsigned threads,
                                 bool direct_io, std::uint64_t payload_cache_bytes)
    : tables_(std::move(tables)) {
    require(threads >= 1 && threads <= 256 && !tables_.empty(), "Engram reader configuration");
    // Resolve before opening the descriptor: invalid governed settings must not leak it.
    const bool ring = envcfg::flag("STRIX_DS41_ENGRAM_URING", true, EnvFlagMatch::exact);
    const unsigned depth = envcfg::number<unsigned>("STRIX_DS41_ENGRAM_URING_DEPTH", 256U, 0U, 4096U);
    if (payload_cache_bytes != 0U) {
        try { cache_ = std::make_unique<PayloadCache>(payload_cache_bytes); }
        catch (const std::bad_alloc&) { cache_.reset(); }  // an optional cache never prevents loading
        if (cache_ && cache_->entries.empty()) cache_.reset();
    }
#if defined(_WIN32)
    if (direct_io) {
        fd_ = open_rows_file(file, true);
        direct_ = fd_ >= 0;
    }
    if (fd_ < 0) fd_ = open_rows_file(file, false);
    if (fd_ < 0)
        throw std::runtime_error("Engram reader cannot open the model file: Win32 error " + std::to_string(::GetLastError()));
    struct _stat64 st{};
    if (::_fstat64(fd_, &st) != 0) { close_fd(fd_); throw std::runtime_error("Engram reader cannot stat the model file"); }
#else
    if (direct_io) {
        fd_ = ::open(file.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        direct_ = fd_ >= 0;
    }
    if (fd_ < 0) fd_ = ::open(file.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw std::runtime_error("Engram reader cannot open the model file: " + std::string(std::strerror(errno)));
    strix::LoaderCapture::record(file, fd_);
    struct stat st{};
    if (::fstat(fd_, &st) != 0) { close_fd(fd_); throw std::runtime_error("Engram reader cannot stat the model file"); }
#endif
    file_bytes_ = static_cast<std::uint64_t>(st.st_size);
    for (const auto& t : tables_)
        if (t.rows == 0 || t.file_offset > file_bytes_ || t.rows > (file_bytes_ - t.file_offset) / engram_row_bytes) {
            close_fd(fd_);
            throw std::invalid_argument("Engram table lies outside the model file");
        }
    if (direct_) {
        // reads in flight on the ring (STRIX_DS41_ENGRAM_URING_DEPTH, default 256: the measured best of 64..512 on the
        // NVMe drives of both nodes; one ring against 96 threads: +15-25 % rows per second and one busy core, not 96 threads)
        if (ring && depth != 0U) fetcher_ = core::DirectRowFetcher::make(fd_, depth);
    }
    for (unsigned i = 0; i < threads; ++i) threads_.emplace_back([this] { worker(); });
}

EngramRowReader::~EngramRowReader() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : threads_) t.join();
    if (cache_) {
        const auto s = stats();
        std::fprintf(stderr, "engram_payload_cache requests=%llu hits=%llu misses=%llu duplicates=%llu hit_bytes=%llu duplicate_bytes=%llu read_bytes=%llu output_bytes=%llu capacity=%llu entries=%llu storage_bytes=%llu evictions=%llu\n",
                     static_cast<unsigned long long>(s.requests), static_cast<unsigned long long>(s.payload_hits),
                     static_cast<unsigned long long>(s.payload_misses), static_cast<unsigned long long>(s.batch_duplicates),
                     static_cast<unsigned long long>(s.payload_hit_bytes), static_cast<unsigned long long>(s.duplicate_bytes),
                     static_cast<unsigned long long>(s.page_bytes), static_cast<unsigned long long>(s.output_bytes),
                     static_cast<unsigned long long>(s.cache_capacity), static_cast<unsigned long long>(s.cache_entries),
                     static_cast<unsigned long long>(s.cache_storage_bytes), static_cast<unsigned long long>(s.cache_evictions));
    }
    if (fd_ >= 0) close_fd(fd_);
}

bool EngramRowReader::ring_io() const {
    std::lock_guard<std::mutex> lock(read_mutex_);
    return fetcher_ && fetcher_->usable();
}

void EngramRowReader::read_one(const EngramRowRequest& r, std::uint8_t* out, std::uint8_t* page) {
    if (r.table >= tables_.size() || r.row >= tables_[r.table].rows) throw std::out_of_range("Engram row out of range");
    const std::uint64_t offset = tables_[r.table].file_offset + r.row * engram_row_bytes;
    if (!direct_) {
        std::size_t got = 0;
        while (got < engram_row_bytes) {
            const auto n = read_at(fd_, out + got, engram_row_bytes - got, offset + got);
            if (n <= 0) throw std::runtime_error("Engram row read failed");
            got += static_cast<std::size_t>(n);
        }
        page_reads_.fetch_add(1, std::memory_order_relaxed);
        page_bytes_.fetch_add(engram_row_bytes, std::memory_order_relaxed);
        return;
    }
    const std::uint64_t first = offset & ~(page_size - 1);
    const std::uint64_t last = (offset + engram_row_bytes + page_size - 1) & ~(page_size - 1);
    const std::size_t length = static_cast<std::size_t>(last - first);  // one or two pages
    std::size_t got = 0;
    while (got < length) {
        const auto n = read_at(fd_, page + got, length - got, first + got);
        if (n <= 0) throw std::runtime_error("Engram row direct read failed");
        got += static_cast<std::size_t>(n);
        if (got % page_size) throw std::runtime_error("Engram direct read returned a partial page");
    }
    std::memcpy(out, page + (offset - first), engram_row_bytes);
    page_reads_.fetch_add(1, std::memory_order_relaxed);
    page_bytes_.fetch_add(length, std::memory_order_relaxed);
}

void EngramRowReader::worker() {
    auto* page = alloc_pages(2 * page_size);
    if (page == nullptr) std::abort();
    std::uint64_t seen = 0;
    for (;;) {
        std::span<const EngramRowRequest> batch;
        std::uint8_t* out = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            // A late worker must not capture an already completed, empty batch: its next_
            // fetch could otherwise consume an index after the caller starts a new batch.
            wake_.wait(lock, [&] { return stop_ || (generation_ != seen && !batch_.empty()); });
            if (stop_) break;
            seen = generation_;
            batch = batch_;
            out = out_;
            ++active_;
        }
        std::size_t done = 0;
        std::string error;
        for (;;) {
            const std::size_t i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= batch.size()) break;
            try {
                const auto dest = destinations_.empty() ? i : destinations_[i];
                read_one(batch[i], out + dest * engram_row_bytes, page);
            } catch (const std::exception& e) {
                if (error.empty()) error = e.what();
            }
            ++done;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            finished_ += done;
            --active_;
            if (!error.empty() && !failed_) { failed_ = true; error_ = error; }
        }
        done_.notify_all();
    }
    free_pages(page);
}

void EngramRowReader::read(std::span<const EngramRowRequest> requests, std::span<std::uint8_t> out) {
    std::lock_guard<std::mutex> caller(read_mutex_);
    require(out.size() == requests.size() * engram_row_bytes, "Engram read output size");
    if (requests.empty()) return;
    const auto start = std::chrono::steady_clock::now();
    std::vector<EngramRowRequest> misses;
    std::vector<std::size_t> destinations, canonical;
    std::uint64_t hits = 0, duplicates = 0;
    if (cache_) {
        std::unordered_map<std::uint64_t, std::size_t> seen;
        seen.reserve(requests.size());
        canonical.resize(requests.size());
        // Validate before consulting the cache or waking workers, including cached table aliases.
        for (const auto& r : requests)
            if (r.table >= tables_.size() || r.row >= tables_[r.table].rows) throw std::out_of_range("Engram row out of range");
        for (std::size_t i = 0; i < requests.size(); ++i) {
            const auto& r = requests[i];
            const auto offset = tables_[r.table].file_offset + r.row * engram_row_bytes;
            const auto [it, unique] = seen.emplace(offset, i);
            canonical[i] = it->second;
            if (!unique) { ++duplicates; continue; }
            if (cache_->get(offset, out.data() + i * engram_row_bytes)) ++hits;
            else { misses.push_back(r); destinations.push_back(i); }
        }
    }
    const std::span<const EngramRowRequest> todo = cache_ ? std::span<const EngramRowRequest>(misses) : requests;
    bool fetched = false;
    if (fetcher_ && fetcher_->usable() && !todo.empty()) {
        std::vector<core::RowFetch> rows(todo.size());
        for (std::size_t i = 0; i < todo.size(); ++i) {
            const auto& r = todo[i];
            if (r.table >= tables_.size() || r.row >= tables_[r.table].rows) throw std::out_of_range("Engram row out of range");
            rows[i] = {tables_[r.table].file_offset + r.row * engram_row_bytes, static_cast<std::uint32_t>(engram_row_bytes)};
        }
        core::RowFetchStats fs;
        try {
            fetcher_->fetch(rows, [&](std::uint32_t i, const std::uint8_t* p) {
                std::memcpy(out.data() + (destinations.empty() ? i : destinations[i]) * engram_row_bytes, p, engram_row_bytes);
            }, &fs);
        } catch (const std::runtime_error& e) {
            throw std::runtime_error(std::string("Engram batch read failed: ") + e.what());
        }
        page_reads_.fetch_add(fs.rows, std::memory_order_relaxed);  // rows read from the file (a coalesced read serves several)
        page_bytes_.fetch_add(fs.read_bytes, std::memory_order_relaxed);
        fetched = true;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    batch_ = fetched ? std::span<const EngramRowRequest>() : todo;
    destinations_ = destinations;
    out_ = out.data();
    finished_ = 0;
    failed_ = false;
    error_.clear();
    next_.store(0, std::memory_order_relaxed);
    if (!batch_.empty()) { ++generation_; wake_.notify_all(); }
    // Every request is counted once, and no worker may still hold this batch when read() returns.
    done_.wait(lock, [&] { return finished_ == batch_.size() && active_ == 0; });
    batch_ = {};
    destinations_ = {};
    out_ = nullptr;
    if (failed_) throw std::runtime_error("Engram batch read failed: " + error_);
    if (cache_) {
        for (std::size_t j = 0; j < misses.size(); ++j) {
            const auto& r = misses[j];
            cache_->put(tables_[r.table].file_offset + r.row * engram_row_bytes,
                        out.data() + destinations[j] * engram_row_bytes);
        }
        for (std::size_t i = 0; i < requests.size(); ++i)
            if (canonical[i] != i) std::memcpy(out.data() + i * engram_row_bytes, out.data() + canonical[i] * engram_row_bytes, engram_row_bytes);
    }
    stats_.requests += requests.size();
    stats_.payload_hits += hits;
    stats_.payload_misses += misses.size();
    stats_.batch_duplicates += duplicates;
    ++stats_.batches;
    stats_.output_bytes += out.size();
    stats_.busy_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

bool EngramRowReader::payload_cache() const noexcept { return cache_ && !cache_->entries.empty(); }

EngramReaderStats EngramRowReader::stats() const {
    std::lock_guard<std::mutex> caller(read_mutex_);
    auto s = stats_;
    s.page_reads = page_reads_.load();
    s.page_bytes = page_bytes_.load();
    s.payload_hit_bytes = s.payload_hits * engram_row_bytes;
    s.duplicate_bytes = s.batch_duplicates * engram_row_bytes;
    if (cache_) {
        s.cache_capacity = cache_->entries.size();
        s.cache_entries = cache_->used;
        s.cache_storage_bytes = cache_->entries.size() * sizeof(PayloadCache::Entry) + cache_->index.size() * sizeof(std::uint32_t);
        s.cache_evictions = cache_->evictions;
    }
    return s;
}

std::vector<EngramTable> engram_tables(const deepseek41::TargetModel& model) {
    std::vector<EngramTable> out;
    for (std::size_t e = 0; e < model.engram.layer_ids.size(); ++e) {
        const std::string name = "blk." + std::to_string(model.engram.layer_ids[e]) + ".engram_embd.weight";
        bool found = false;
        for (const auto& t : model.manifest.tensors)
            if (t.name == name) {
                require(t.byte_size == model.engram.rows[e] * engram_row_bytes, "Engram table size differs from its rows");
                out.push_back({t.absolute_offset, model.engram.rows[e]});
                found = true;
            }
        require(found, "Engram table missing from the manifest");
    }
    return out;
}

} // namespace strix::ds41
