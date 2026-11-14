#include <memory>
#include <atomic>
#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <optional>
#include <string>
#include <stdexcept>
#include <system_error>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#if defined(__linux__)
#include <sys/stat.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#define STRIX_SHA_NI 1
#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define STRIX_SHA_NI_TARGET
#else
#include <cpuid.h>
#define STRIX_SHA_NI_TARGET __attribute__((target("sha,sse4.1,ssse3")))
#endif
#endif
#if defined(STRIX_USE_OPENSSL_SHA256)
#include <openssl/evp.h>
#include <memory>
#endif

namespace strix {
namespace {
std::atomic<std::uint64_t> cache_hits{0}, cache_misses{0};


constexpr std::array<std::uint32_t, 64> k{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

constexpr std::uint32_t rotr(
    std::uint32_t value,
    unsigned shift) noexcept {
    return (value >> shift) |
           (value << (32U - shift));
}

class Sha256 {
public:
    void update(
        const std::uint8_t* data,
        std::size_t size) {
        if (size != 0 && data == nullptr) {
            throw std::invalid_argument(
                "SHA-256 update received null data");
        }
        if (size >
            std::numeric_limits<std::uint64_t>::max() -
                total_bytes_) {
            throw std::overflow_error(
                "SHA-256 input length overflow");
        }
        total_bytes_ +=
            static_cast<std::uint64_t>(size);

        for (std::size_t index = 0;
             index < size;
             ++index) {
            buffer_[buffer_size_++] =
                data[index];
            if (buffer_size_ ==
                buffer_.size()) {
                transform(buffer_.data());
                buffer_size_ = 0;
            }
        }
    }

    void update(
        std::string_view value) {
        update(
            reinterpret_cast<
                const std::uint8_t*>(
                    value.data()),
            value.size());
    }

    void update_u64(
        std::uint64_t value) {
        std::array<std::uint8_t, 8> bytes{};
        for (unsigned index = 0;
             index < 8;
             ++index) {
            bytes[index] =
                static_cast<std::uint8_t>(
                    (value >> (index * 8U)) &
                    0xffULL);
        }
        update(
            bytes.data(),
            bytes.size());
    }

    [[nodiscard]] ModelContentIdentity
    finalize() {
        const auto message_bytes =
            total_bytes_;
        if (message_bytes >
            std::numeric_limits<std::uint64_t>::max() /
                8ULL) {
            throw std::overflow_error(
                "message length exceeds SHA-256 64-bit bit counter");
        }
        const auto bit_length =
            message_bytes * 8ULL;

        buffer_[buffer_size_++] =
            0x80U;

        if (buffer_size_ > 56U) {
            while (buffer_size_ <
                   buffer_.size()) {
                buffer_[buffer_size_++] =
                    0;
            }
            transform(buffer_.data());
            buffer_size_ = 0;
        }

        while (buffer_size_ < 56U) {
            buffer_[buffer_size_++] = 0;
        }

        for (unsigned index = 0;
             index < 8;
             ++index) {
            buffer_[63U - index] =
                static_cast<std::uint8_t>(
                    (bit_length >>
                     (index * 8U)) &
                    0xffULL);
        }
        transform(buffer_.data());
        buffer_size_ = 0;

        ModelContentIdentity identity;
        for (std::size_t word = 0;
             word < state_.size();
             ++word) {
            identity.bytes[word * 4U] =
                static_cast<std::uint8_t>(
                    state_[word] >> 24U);
            identity.bytes[word * 4U + 1U] =
                static_cast<std::uint8_t>(
                    state_[word] >> 16U);
            identity.bytes[word * 4U + 2U] =
                static_cast<std::uint8_t>(
                    state_[word] >> 8U);
            identity.bytes[word * 4U + 3U] =
                static_cast<std::uint8_t>(
                    state_[word]);
        }
        return identity;
    }

private:
    void transform(
        const std::uint8_t* block) noexcept {
        std::array<std::uint32_t, 64>
            schedule{};

        for (unsigned index = 0;
             index < 16;
             ++index) {
            const auto offset =
                index * 4U;
            schedule[index] =
                (static_cast<std::uint32_t>(
                     block[offset])
                 << 24U) |
                (static_cast<std::uint32_t>(
                     block[offset + 1U])
                 << 16U) |
                (static_cast<std::uint32_t>(
                     block[offset + 2U])
                 << 8U) |
                static_cast<std::uint32_t>(
                    block[offset + 3U]);
        }

        for (unsigned index = 16;
             index < 64;
             ++index) {
            const auto s0 =
                rotr(
                    schedule[index - 15U],
                    7U) ^
                rotr(
                    schedule[index - 15U],
                    18U) ^
                (schedule[index - 15U] >>
                 3U);
            const auto s1 =
                rotr(
                    schedule[index - 2U],
                    17U) ^
                rotr(
                    schedule[index - 2U],
                    19U) ^
                (schedule[index - 2U] >>
                 10U);
            schedule[index] =
                schedule[index - 16U] +
                s0 +
                schedule[index - 7U] +
                s1;
        }

        auto a = state_[0];
        auto b = state_[1];
        auto c = state_[2];
        auto d = state_[3];
        auto e = state_[4];
        auto f = state_[5];
        auto g = state_[6];
        auto h = state_[7];

        for (unsigned index = 0;
             index < 64;
             ++index) {
            const auto sigma1 =
                rotr(e, 6U) ^
                rotr(e, 11U) ^
                rotr(e, 25U);
            const auto choose =
                (e & f) ^
                ((~e) & g);
            const auto temp1 =
                h +
                sigma1 +
                choose +
                k[index] +
                schedule[index];
            const auto sigma0 =
                rotr(a, 2U) ^
                rotr(a, 13U) ^
                rotr(a, 22U);
            const auto majority =
                (a & b) ^
                (a & c) ^
                (b & c);
            const auto temp2 =
                sigma0 +
                majority;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{
        0x6a09e667U,
        0xbb67ae85U,
        0x3c6ef372U,
        0xa54ff53aU,
        0x510e527fU,
        0x9b05688cU,
        0x1f83d9abU,
        0x5be0cd19U,
    };
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffer_size_ = 0;
    std::uint64_t total_bytes_ = 0;
};

struct FileDigest {
    ModelContentIdentity digest;
    std::uint64_t bytes = 0;
};

#if defined(STRIX_USE_OPENSSL_SHA256)
// Only the bulk file digest uses the optional provider. Canonical aggregate
// framing and the portable byte oracle retain the same implementation.
class FileSha256 {
public:
    FileSha256() : context_(EVP_MD_CTX_new(), &EVP_MD_CTX_free) {
        if (!context_ || EVP_DigestInit_ex(context_.get(), EVP_sha256(), nullptr) != 1)
            throw std::runtime_error("cannot initialize full-file SHA-256 provider");
    }
    void update(const std::uint8_t* data, std::size_t size) {
        if (size != 0 && data == nullptr)
            throw std::invalid_argument("SHA-256 update received null data");
        if (size > std::numeric_limits<std::uint64_t>::max() - total_bytes_)
            throw std::overflow_error("SHA-256 input length overflow");
        if (size != 0 && EVP_DigestUpdate(context_.get(), data, size) != 1)
            throw std::runtime_error("full-file SHA-256 provider update failed");
        total_bytes_ += static_cast<std::uint64_t>(size);
    }
    ModelContentIdentity finalize() {
        ModelContentIdentity result;
        unsigned int bytes = 0;
        if (EVP_DigestFinal_ex(context_.get(), result.bytes.data(), &bytes) != 1 ||
            bytes != result.bytes.size())
            throw std::runtime_error("full-file SHA-256 provider finalization failed");
        return result;
    }
private:
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context_;
    std::uint64_t total_bytes_ = 0;
};
#elif defined(_WIN32)
class FileSha256 {
    Sha256Fast hash_;
public:
    void update(const std::uint8_t* data,std::size_t size) {
        hash_.update(std::as_bytes(std::span(data,size)));
    }
    ModelContentIdentity finalize(){return hash_.finalize();}
};
#else
using FileSha256 = Sha256;
#endif

[[maybe_unused]] FileDigest hash_file_stable_uncached(const std::filesystem::path& path, const FileHashProgress& progress={});

#if defined(__linux__) || defined(_WIN32)
// Verified-identity cache, like git's index: a full-file SHA-256 of a ~60 GB model costs ~48 s of one
// core on every start. The digest is reused only while the file keeps its device, inode, size, mtime
// and ctime on Linux; Windows v3 uses volume/file id, size, write and creation times.
// Windows excludes ChangeTime because metadata-only hard-link operations move it. A changed
// file is always re-hashed. STRIX_HASH_CACHE=0 disables it; STRIX_HASH_CACHE=DIR picks the directory
// (default $XDG_CACHE_HOME/strix-engine/file-sha256, else ~/.cache/strix-engine/file-sha256).
struct HashCacheKey {
#ifdef _WIN32
    std::filesystem::path file;
#else
    std::string file;
#endif
    std::string stamp;
};
std::optional<HashCacheKey> hash_cache_key(const std::filesystem::path& path, [[maybe_unused]] const FileIdentity* expected = nullptr) {
#ifdef _WIN32
    const wchar_t* setting = _wgetenv(L"STRIX_HASH_CACHE");
    if (setting && std::wstring_view(setting) == L"0") return std::nullopt;
    std::filesystem::path dir;
    if (setting && setting[0]) dir = setting;
    else if (const wchar_t* local = _wgetenv(L"LOCALAPPDATA"); local && local[0])
        dir = std::filesystem::path(local) / "strix-engine/file-sha256";
    else return std::nullopt;
    HANDLE handle = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) return std::nullopt;
    BY_HANDLE_FILE_INFORMATION info{};
    const bool ok = GetFileInformationByHandle(handle, &info) != 0;
    CloseHandle(handle);
    if (!ok || (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) return std::nullopt;
    const auto u64 = [](DWORD high, DWORD low) { return (std::uint64_t(high) << 32U) | low; };
    const auto size = u64(info.nFileSizeHigh, info.nFileSizeLow);
    const auto name = std::to_string(info.dwVolumeSerialNumber) + "-" +
        std::to_string(u64(info.nFileIndexHigh, info.nFileIndexLow));
    // Version and availability marker prevent reuse across different stamp schemas.
    const auto stamp = "v3 " + std::to_string(size) + " " +
        std::to_string(u64(info.ftLastWriteTime.dwHighDateTime, info.ftLastWriteTime.dwLowDateTime)) + " " +
        std::to_string(u64(info.ftCreationTime.dwHighDateTime, info.ftCreationTime.dwLowDateTime));
    return HashCacheKey{dir / name, stamp};
#else
    const char* setting = std::getenv("STRIX_HASH_CACHE");
    if (setting && setting[0] == '0' && setting[1] == '\0') return std::nullopt;
    std::filesystem::path dir;
    if (setting && setting[0]) dir = setting;
    else if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && xdg[0]) dir = std::filesystem::path(xdg) / "strix-engine/file-sha256";
    else if (const char* home = std::getenv("HOME"); home && home[0]) dir = std::filesystem::path(home) / ".cache/strix-engine/file-sha256";
    else return std::nullopt;
    struct stat st {};
    if (expected) {
        st.st_dev = expected->dev; st.st_ino = expected->ino; st.st_size = static_cast<off_t>(expected->size);
        st.st_mtim = {expected->mtime_sec, expected->mtime_ns};
        st.st_ctim = {expected->ctime_sec, expected->ctime_ns};
    } else if (::stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return std::nullopt;
    char name[64], stamp[160];
    std::snprintf(name, sizeof(name), "%llx-%llx", static_cast<unsigned long long>(st.st_dev),
        static_cast<unsigned long long>(st.st_ino));
    std::snprintf(stamp, sizeof(stamp), "v1 %lld %lld.%09ld %lld.%09ld", static_cast<long long>(st.st_size),
        static_cast<long long>(st.st_mtim.tv_sec), st.st_mtim.tv_nsec, static_cast<long long>(st.st_ctim.tv_sec),
        st.st_ctim.tv_nsec);
    return HashCacheKey{(dir / name).string(), stamp};
#endif
}
std::optional<FileDigest> hash_cache_read(const HashCacheKey& key) {
    std::ifstream in(key.file);
    std::string stamp, hex;
    if (!std::getline(in, stamp) || stamp != key.stamp || !std::getline(in, hex) || hex.size() != 64U) return std::nullopt;
    FileDigest digest;
    for (std::size_t i = 0; i < 32U; ++i) {
        const auto nibble = [](char c) -> int {
            return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        };
        const int hi = nibble(hex[2U * i]), lo = nibble(hex[2U * i + 1U]);
        if (hi < 0 || lo < 0) return std::nullopt;
        digest.digest.bytes[i] = static_cast<std::uint8_t>(hi * 16 + lo);
    }
    digest.bytes = std::stoull(stamp.substr(3, stamp.find(' ', 3) - 3));
    if (!digest.digest.valid()) return std::nullopt;
    return digest;
}
void hash_cache_write(const HashCacheKey& key, const FileDigest& digest) {
    std::error_code error;
    std::filesystem::create_directories(std::filesystem::path(key.file).parent_path(), error);
#ifdef _WIN32
    // Each writer owns its temporary file; concurrent workers never truncate one another.
    static std::atomic<unsigned long long> sequence{0};
    auto temporary = key.file;
    temporary += L".tmp-" + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(sequence++);
    HANDLE out = CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (out == INVALID_HANDLE_VALUE) return;
    const auto contents = key.stamp + "\n" + model_content_identity_hex(digest.digest) + "\n";
    DWORD written = 0;
    const bool complete = WriteFile(out, contents.data(), static_cast<DWORD>(contents.size()), &written, nullptr) != 0 &&
        written == contents.size() && FlushFileBuffers(out) != 0;
    const bool closed = CloseHandle(out) != 0;
    if (!complete || !closed || !MoveFileExW(temporary.c_str(), key.file.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        DeleteFileW(temporary.c_str());
#else
    const auto temporary = key.file + ".tmp";
    {
        std::ofstream out(temporary, std::ios::trunc);
        if (!out) return;
        constexpr char digits[] = "0123456789abcdef";
        std::string hex(64U, '0');
        for (std::size_t i = 0; i < 32U; ++i) {
            hex[2U * i] = digits[digest.digest.bytes[i] >> 4U];
            hex[2U * i + 1U] = digits[digest.digest.bytes[i] & 0x0fU];
        }
        out << key.stamp << '\n' << hex << '\n';
        if (!out) return;
    }
    std::filesystem::rename(temporary, key.file, error);
#endif
}
#endif

FileDigest hash_file_stable(const std::filesystem::path& path, FileContentHashStats* stats=nullptr, const FileHashProgress& progress={}) {
    if(stats)stats->cache_hit=false;
    const auto start = std::chrono::steady_clock::now();
    const auto report = [&](const char* status) {
        if (const char* log = std::getenv("STRIX_HASH_LOG"); log && std::string_view(log) == "1")
            std::fprintf(stderr, "session content_hash=%s ms=%.3f path=%s\n", status,
                std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(), path.string().c_str());
    };

#if defined(__linux__) || defined(_WIN32)
    const auto key = hash_cache_key(path);
    if (key)
        if (auto cached = hash_cache_read(*key)) {
#ifdef _WIN32
            const auto after = hash_cache_key(path);
            if (after && after->file == key->file && after->stamp == key->stamp) {
#endif
                ++cache_hits; report("hit");
                if (stats) stats->cache_hit = true;
                if (progress) progress(cached->bytes, cached->bytes);
                return *cached;
#ifdef _WIN32
            }
#endif
        }
    ++cache_misses;
    const auto digest = hash_file_stable_uncached(path,progress);
    // Store only if the file still carries the stamp it had before hashing.
    if (key)
#ifdef _WIN32
    {
        const auto after = hash_cache_key(path);
        if (!after || after->file != key->file || after->stamp != key->stamp)
            throw std::runtime_error("model content file changed while hashing");
        hash_cache_write(*key, digest);
    }
#else
        if (const auto after = hash_cache_key(path); after && after->stamp == key->stamp) hash_cache_write(*key, digest);
#endif
    report("miss");
    return digest;
#else
    ++cache_misses;
    auto digest = hash_file_stable_uncached(path,progress);
    report("miss");
    return digest;
#endif
}

[[maybe_unused]] FileDigest hash_file_stable_uncached(
    const std::filesystem::path& path, const FileHashProgress& progress) {
    std::error_code error;
    const auto before =
        std::filesystem::file_size(
            path,
            error);
    if (error) {
        throw std::runtime_error(
            "cannot stat model content file before hashing");
    }

    // Unbuffered 8 MiB reads, each on its own thread while this one hashes the block before: the pass
    // costs the slower of reading and hashing instead of their sum. (With MSVC's library, std::ifstream's
    // 1 MiB reads went through the stream buffer at about 500 MB/s, against about 3 GB/s unbuffered.)
#ifdef _WIN32
    std::unique_ptr<std::FILE, int (*)(std::FILE*)> input(_wfopen(path.c_str(), L"rb"), &std::fclose);
#else
    std::unique_ptr<std::FILE, int (*)(std::FILE*)> input(std::fopen(path.c_str(), "rb"), &std::fclose);
#endif
    if (!input || std::setvbuf(input.get(), nullptr, _IONBF, 0) != 0) {
        throw std::runtime_error(
            "cannot open model content file");
    }

    FileSha256 hash;
    std::array<std::vector<std::uint8_t>, 2> buffers{
        std::vector<std::uint8_t>(8U << 20U), std::vector<std::uint8_t>(8U << 20U)};
    const auto read_block = [file = input.get()](std::vector<std::uint8_t>& buffer) {
        return std::fread(buffer.data(), 1, buffer.size(), file);
    };
    std::uint64_t bytes = 0;

    if(progress)progress(0,before);
    // Declared after the file and the buffers: an exception from the hash waits for the read in flight.
    auto pending = std::async(std::launch::async, read_block, std::ref(buffers[0]));
    for (std::size_t current = 0;; current ^= 1U) {
        const auto count = pending.get();
        if (count == 0) break;
        pending = std::async(std::launch::async, read_block, std::ref(buffers[current ^ 1U]));
        hash.update(buffers[current].data(), count);
        bytes += static_cast<std::uint64_t>(count);
        if(progress)progress(bytes,before);
    }
    if (std::ferror(input.get())) {
        throw std::runtime_error(
            "error while hashing model content file");
    }

    const auto after =
        std::filesystem::file_size(
            path,
            error);
    if (error ||
        before != after ||
        bytes != before) {
        throw ModelContentChanged(
            "model content file changed while hashing");
    }

    return FileDigest{
        hash.finalize(),
        bytes,
    };
}

}  // namespace

FileHashCacheStats file_hash_cache_stats() noexcept { return {cache_hits.load(), cache_misses.load()}; }

ModelContentIdentity sha256_file_content(
    const std::filesystem::path& path) {
    return hash_file_stable(path).digest;
}

ModelContentIdentity sha256_file_content(
    const std::filesystem::path& path, FileContentHashStats* stats) {
    return hash_file_stable(path,stats).digest;
}
ModelContentIdentity sha256_file_content(
    const std::filesystem::path& path, FileContentHashStats* stats,
    const FileHashProgress& progress) {
    return hash_file_stable(path,stats,progress).digest;
}

ModelContentIdentity sha256_file_content(const std::filesystem::path& path, FileHashPolicy policy) {
    return policy == FileHashPolicy::full ? hash_file_stable_uncached(path).digest : sha256_file_content(path);
}

ModelContentIdentity sha256_file_content(const std::filesystem::path& path, const FileIdentity& expected) {
#if defined(__linux__)
    const auto key = hash_cache_key(path, &expected);
    if (key)
        if (auto cached = hash_cache_read(*key)) { ++cache_hits; return cached->digest; }
    ++cache_misses;
    const auto changed = [] { throw std::runtime_error("model files changed during startup"); };
    const auto close_file = [](FILE* file) { std::fclose(file); };
    std::unique_ptr<FILE, decltype(close_file)> input(std::fopen(path.c_str(), "rb"), close_file);
    if (!input || fd_identity(::fileno(input.get())) != expected) changed();
    FileSha256 hash;
    std::vector<std::uint8_t> buffer(1024U * 1024U);
    std::uint64_t bytes = 0;
    for (;;) {
        const auto count = std::fread(buffer.data(), 1, buffer.size(), input.get());
        hash.update(buffer.data(), count);
        bytes += count;
        if (count < buffer.size()) break;
    }
    if (std::ferror(input.get()) || bytes != expected.size ||
        fd_identity(::fileno(input.get())) != expected) changed();
    const FileDigest digest{hash.finalize(), bytes};
    if (key) hash_cache_write(*key, digest);
    return digest.digest;
#else
    (void)path; (void)expected;
    throw std::runtime_error("loader-bound hashing requires Linux");
#endif
}

ModelContentIdentity sha256_bytes(std::span<const std::byte> bytes) {
    Sha256 digest;
    digest.update(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
    return digest.finalize();
}

// ---- SHA-256 of in-memory bytes on the CPU's SHA extensions (same digest as sha256_bytes) ----
namespace {

#if defined(STRIX_SHA_NI)
bool cpu_has_sha_ni() noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    int r[4] = {};
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    const bool sse41 = (r[2] & (1 << 19)) != 0, ssse3 = (r[2] & (1 << 9)) != 0;
    __cpuidex(r, 7, 0);
    return sse41 && ssse3 && (r[1] & (1 << 29)) != 0;
#else
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (__get_cpuid_max(0, nullptr) < 7U) return false;
    __get_cpuid(1, &a, &b, &c, &d);
    const bool sse41 = (c & (1U << 19U)) != 0U, ssse3 = (c & (1U << 9U)) != 0U;
    __get_cpuid_count(7, 0, &a, &b, &c, &d);
    return sse41 && ssse3 && (b & (1U << 29U)) != 0U;
#endif
}

// `blocks` 64-byte blocks into state (the FIPS 180-4 compression, the rounds on SHA256RNDS2 / MSG1 / MSG2).
STRIX_SHA_NI_TARGET void sha_ni_blocks(std::uint32_t state[8], const std::uint8_t* data, std::size_t blocks) noexcept {
    const __m128i bswap = _mm_set_epi64x(0x0c0d0e0f08090a0bLL, 0x0405060700010203LL);
    __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(state));         // DCBA
    __m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(state + 4));      // HGFE
    tmp = _mm_shuffle_epi32(tmp, 0xB1);                                             // CDAB
    s1 = _mm_shuffle_epi32(s1, 0x1B);                                               // EFGH
    __m128i s0 = _mm_alignr_epi8(tmp, s1, 8);                                       // ABEF
    s1 = _mm_blend_epi16(s1, tmp, 0xF0);                                            // CDGH
    for (; blocks != 0U; --blocks, data += 64) {
        const __m128i abef = s0, cdgh = s1;
        __m128i w[4];
        for (int g = 0; g < 16; ++g) {
            __m128i& m = w[g & 3];
            if (g < 4) {
                m = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 16 * g)), bswap);
            } else {
                // W[t] = sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16], four words at a time
                const __m128i x = _mm_add_epi32(_mm_sha256msg1_epu32(w[g & 3], w[(g + 1) & 3]), _mm_alignr_epi8(w[(g + 3) & 3], w[(g + 2) & 3], 4));
                m = _mm_sha256msg2_epu32(x, w[(g + 3) & 3]);
            }
            __m128i msg = _mm_add_epi32(m, _mm_loadu_si128(reinterpret_cast<const __m128i*>(k.data() + 4 * g)));
            s1 = _mm_sha256rnds2_epu32(s1, s0, msg);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            s0 = _mm_sha256rnds2_epu32(s0, s1, msg);
        }
        s0 = _mm_add_epi32(s0, abef);
        s1 = _mm_add_epi32(s1, cdgh);
    }
    tmp = _mm_shuffle_epi32(s0, 0x1B);   // FEBA
    s1 = _mm_shuffle_epi32(s1, 0xB1);    // DCHG
    s0 = _mm_blend_epi16(tmp, s1, 0xF0); // DCBA
    s1 = _mm_alignr_epi8(s1, tmp, 8);    // HGFE
    _mm_storeu_si128(reinterpret_cast<__m128i*>(state), s0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(state + 4), s1);
}

const bool kShaNi = cpu_has_sha_ni();
#endif

}  // namespace

const char* sha256_fast_provider() noexcept {
#if defined(STRIX_SHA_NI)
    if (kShaNi) return "sha-ni";
#endif
    return "portable";
}

struct Sha256Fast::Impl {
    Sha256 portable;  // without the SHA extensions
    std::uint32_t state[8] = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU, 0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    std::uint8_t buf[64] = {};
    std::size_t buf_n = 0;
    std::uint64_t total = 0;
    bool done = false;
};
Sha256Fast::Sha256Fast() : impl_(new Impl) {}
Sha256Fast::~Sha256Fast() { delete impl_; }
void Sha256Fast::update(std::span<const std::byte> bytes) {
    auto& m = *impl_;
    if (m.done) throw std::logic_error("SHA-256 update after finalize");
    const auto* p = reinterpret_cast<const std::uint8_t*>(bytes.data());
    std::size_t n = bytes.size();
#if defined(STRIX_SHA_NI)
    if (kShaNi) {
        m.total += n;
        if (m.buf_n != 0U) {
            const std::size_t take = std::min(n, 64U - m.buf_n);
            std::memcpy(m.buf + m.buf_n, p, take);
            m.buf_n += take;
            p += take;
            n -= take;
            if (m.buf_n < 64U) return;
            sha_ni_blocks(m.state, m.buf, 1);
            m.buf_n = 0;
        }
        if (n >= 64U) {
            sha_ni_blocks(m.state, p, n / 64U);
            p += n / 64U * 64U;
            n %= 64U;
        }
        std::memcpy(m.buf, p, n);
        m.buf_n = n;
        return;
    }
#endif
    m.portable.update(p, n);
}
ModelContentIdentity Sha256Fast::finalize() {
    auto& m = *impl_;
    if (m.done) throw std::logic_error("SHA-256 finalize twice");
    m.done = true;
#if defined(STRIX_SHA_NI)
    if (kShaNi) {
        std::uint8_t tail[128] = {};
        std::memcpy(tail, m.buf, m.buf_n);
        tail[m.buf_n] = 0x80U;
        const std::size_t tail_blocks = m.buf_n + 9U <= 64U ? 1U : 2U;
        const std::uint64_t bits = m.total * 8U;
        for (unsigned i = 0; i < 8U; ++i) tail[tail_blocks * 64U - 1U - i] = static_cast<std::uint8_t>(bits >> (8U * i));
        sha_ni_blocks(m.state, tail, tail_blocks);
        ModelContentIdentity id;
        for (std::size_t w = 0; w < 8U; ++w)
            for (std::size_t b = 0; b < 4U; ++b) id.bytes[w * 4U + b] = static_cast<std::uint8_t>(m.state[w] >> (24U - 8U * b));
        return id;
    }
#endif
    return m.portable.finalize();
}

ModelContentIdentity sha256_bytes_fast(std::span<const std::byte> bytes) {
    Sha256Fast h;
    h.update(bytes);
    return h.finalize();
}

ModelContentIdentity build_model_content_identity(
    std::span<const ModelContentFile> files) {
    if (files.empty()) {
        throw std::invalid_argument(
            "model content identity requires at least one file");
    }

    std::vector<const ModelContentFile*>
        ordered;
    ordered.reserve(files.size());
    for (const auto& file : files) {
        if (file.logical_name.empty()) {
            throw std::invalid_argument(
                "model content logical file name must be non-empty");
        }
        ordered.push_back(&file);
    }

    std::sort(
        ordered.begin(),
        ordered.end(),
        [](const auto* left,
           const auto* right) {
            return left->logical_name <
                   right->logical_name;
        });

    for (std::size_t index = 1;
         index < ordered.size();
         ++index) {
        if (ordered[index - 1]->logical_name ==
            ordered[index]->logical_name) {
            throw std::invalid_argument(
                "model content logical file names must be unique");
        }
    }

    Sha256 aggregate;
    aggregate.update(
        "STRIX-MODEL-CONTENT-V1");

    for (const auto* file : ordered) {
        const auto digest =
            hash_file_stable(file->path);
        aggregate.update_u64(
            static_cast<std::uint64_t>(
                file->logical_name.size()));
        aggregate.update(
            file->logical_name);
        aggregate.update_u64(
            digest.bytes);
        aggregate.update(
            digest.digest.bytes.data(),
            digest.digest.bytes.size());
    }

    const auto identity =
        aggregate.finalize();
    if (!identity.valid()) {
        throw std::logic_error(
            "model content SHA-256 identity unexpectedly all zero");
    }
    return identity;
}

SingleModelContentIdentity hash_single_model_content_file(const ModelContentFile& file) {
    return hash_single_model_content_file(file, FileHashPolicy::cached);
}
SingleModelContentIdentity hash_single_model_content_file(const ModelContentFile& file, FileHashPolicy policy) {
    if(file.logical_name.empty()) throw std::invalid_argument("model content logical file name must be non-empty");
    const auto digest=policy == FileHashPolicy::full ? hash_file_stable_uncached(file.path) : hash_file_stable(file.path);
    Sha256 aggregate;aggregate.update("STRIX-MODEL-CONTENT-V1");
    aggregate.update_u64(static_cast<std::uint64_t>(file.logical_name.size()));
    aggregate.update(file.logical_name);aggregate.update_u64(digest.bytes);
    aggregate.update(digest.digest.bytes.data(),digest.digest.bytes.size());
    const auto identity=aggregate.finalize();
    if(!identity.valid() || !digest.digest.valid()) throw std::logic_error("model content SHA-256 identity unexpectedly all zero");
    return {digest.digest,identity,digest.bytes};
}
std::string model_content_identity_hex(
    const ModelContentIdentity& identity) {
    constexpr char digits[] =
        "0123456789abcdef";
    std::string output;
    output.resize(
        identity.bytes.size() * 2U);

    for (std::size_t index = 0;
         index < identity.bytes.size();
         ++index) {
        const auto byte =
            identity.bytes[index];
        output[index * 2U] =
            digits[byte >> 4U];
        output[index * 2U + 1U] =
            digits[byte & 0x0fU];
    }
    return output;
}

}  // namespace strix
