#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iterator>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void write_file(
    const std::filesystem::path& path,
    std::string_view content) {
    std::ofstream output(
        path,
        std::ios::binary);
    output.write(
        content.data(),
        static_cast<std::streamsize>(
            content.size()));
    if (!output) {
        throw std::runtime_error(
            "failed to write hash fixture");
    }
}

#ifdef _WIN32
class ScopedEnvironment {
public:
    ScopedEnvironment(const wchar_t* name, const std::wstring& value) : name_(name) {
        if (const auto* old = _wgetenv(name)) { existed_ = true; old_ = old; }
        if (_wputenv_s(name, value.c_str()) != 0) throw std::runtime_error("cannot set test environment");
    }
    ~ScopedEnvironment() { _wputenv_s(name_.c_str(), existed_ ? old_.c_str() : L""); }
private:
    std::wstring name_, old_;
    bool existed_ = false;
};

void windows_cache_tests(const std::filesystem::path& root) {
    const auto cache = root / L"cache-\u00e9";
    ScopedEnvironment setting(L"STRIX_HASH_CACHE", cache.wstring());
    const auto file = root / "windows-cache.bin";
    write_file(file, "abc");
    strix::FileContentHashStats stats;
    const auto digest = strix::sha256_file_content(file, &stats);
    expect(!stats.cache_hit, "Windows first hash misses");
    expect(strix::sha256_file_content(file, &stats) == digest && stats.cache_hit,
           "Windows unchanged file hits persistent cache");
    expect(std::distance(std::filesystem::directory_iterator(cache), std::filesystem::directory_iterator{}) == 1,
           "Windows cache publishes one record and leaves no temporary files");

    const auto before_stats = strix::file_hash_cache_stats();
    std::vector<std::uint64_t> progress;
    expect(strix::sha256_file_content(file, &stats, [&](auto done, auto total) {
        expect(done == total && total == 3, "cache hit reports complete file progress");
        progress.push_back(done);
    }) == digest && stats.cache_hit && progress.size() == 1,
           "per-call stats and progress agree on the cached digest");
    const auto after_stats = strix::file_hash_cache_stats();
    expect(after_stats.hits == before_stats.hits + 1 && after_stats.misses == before_stats.misses,
           "process cache counters retain the shared API alongside RTX progress");
    expect(strix::sha256_file_content(file, strix::FileHashPolicy::full) == digest,
           "explicit full hashing preserves the shared policy API");
    expect(strix::hash_single_model_content_file({"model", file}, strix::FileHashPolicy::full).file_sha256 == digest,
           "explicit full single-model identity preserves the shared policy API");

    const auto stamp = std::filesystem::last_write_time(file);
    std::filesystem::last_write_time(file, stamp + std::chrono::seconds(2));
    expect(strix::sha256_file_content(file, &stats) == digest && !stats.cache_hit,
           "Windows touching unchanged content invalidates cache");
    expect(strix::sha256_file_content(file, &stats) == digest && stats.cache_hit,
           "Windows refreshed cache replaces the previous record");
    // A metadata-only NTFS event (adding/removing another hard link moves ChangeTime) keeps the record.
    const auto link = root / "windows-cache-link.bin";
    std::filesystem::create_hard_link(file, link);
    std::filesystem::remove(link);
    expect(strix::sha256_file_content(file, &stats) == digest && stats.cache_hit,
           "Windows hard-link removal (ChangeTime only) keeps the cached digest");
    write_file(file, "abcd");
    const auto changed = strix::sha256_file_content(file, &stats);
    expect(changed != digest && !stats.cache_hit,
           "Windows rewritten content (new size and write time) invalidates cache");
    expect(strix::sha256_file_content(file, &stats) == changed && stats.cache_hit,
           "Windows changed file gets a reusable digest");

    HANDLE handle = CreateFileW(file.c_str(), FILE_READ_ATTRIBUTES | FILE_WRITE_ATTRIBUTES,
        FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open creation-time fixture");
    FILETIME creation{};
    const bool read = GetFileTime(handle, &creation, nullptr, nullptr) != 0;
    creation.dwLowDateTime ^= 0x10000U;
    const bool set = read && SetFileTime(handle, &creation, nullptr, nullptr) != 0;
    CloseHandle(handle);
    expect(set, "Windows creation-time fixture updated");
    expect(strix::sha256_file_content(file, &stats) == changed && !stats.cache_hit,
           "Windows creation-time change invalidates cache");
    {
        ScopedEnvironment disabled(L"STRIX_HASH_CACHE", L"0");
        expect(strix::sha256_file_content(file, &stats) == changed && !stats.cache_hit,
               "Windows disabled cache ignores existing record");
        const auto fresh = root / "disabled.bin";
        write_file(fresh, "abc");
        expect(strix::sha256_file_content(fresh, &stats) == digest && !stats.cache_hit,
               "Windows disabled cache hashes fresh files");
        expect(strix::sha256_file_content(fresh, &stats) == digest && !stats.cache_hit,
               "Windows disabled cache never produces a hit");
    }
    {
        ScopedEnvironment default_setting(L"STRIX_HASH_CACHE", L"");
        ScopedEnvironment local(L"LOCALAPPDATA", (root / "local").wstring());
        ScopedEnvironment home(L"HOME", (root / "wrong-home").wstring());
        ScopedEnvironment xdg(L"XDG_CACHE_HOME", (root / "wrong-xdg").wstring());
        expect(strix::sha256_file_content(file, &stats) == changed && !stats.cache_hit,
               "Windows default cache starts empty");
        expect(strix::sha256_file_content(file, &stats) == changed && stats.cache_hit,
               "Windows LOCALAPPDATA cache hits");
        expect(std::filesystem::is_directory(root / "local/strix-engine/file-sha256") &&
               !std::filesystem::exists(root / "wrong-home") && !std::filesystem::exists(root / "wrong-xdg"),
               "Windows default cache uses LOCALAPPDATA even when HOME and XDG are set");
    }
}
#endif

}  // namespace

int main() {
    expect(strix::model_content_identity_hex(strix::sha256_bytes({})) ==
               "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
           "SHA-256 empty buffer matches known vector");
    constexpr std::string_view memory_vector = "abc";
    expect(strix::model_content_identity_hex(strix::sha256_bytes(
               std::as_bytes(std::span(memory_vector.data(), memory_vector.size())))) ==
               "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
           "SHA-256 memory buffer matches known vector");
    {  // the fast provider (SHA-NI when the CPU has it) gives sha256_bytes' digest at every length around the block edges
        std::vector<std::byte> data(70000);
        std::uint32_t x = 12345U;
        for (auto& b : data) {
            x = x * 1664525U + 1013904223U;
            b = static_cast<std::byte>(x >> 24U);
        }
        bool same = true;
        for (std::size_t n = 0; n < 300U; ++n)
            same = same && strix::sha256_bytes_fast(std::span(data.data(), n)) == strix::sha256_bytes(std::span(data.data(), n));
        for (const std::size_t n : {1000U, 4095U, 4096U, 65536U, 70000U})
            same = same && strix::sha256_bytes_fast(std::span(data.data(), n)) == strix::sha256_bytes(std::span(data.data(), n));
        same = same && strix::sha256_bytes_fast(std::span(data.data() + 1, 777U)) == strix::sha256_bytes(std::span(data.data() + 1, 777U));
        for (const std::size_t cut : {0U, 1U, 63U, 64U, 65U, 1000U, 69997U}) {  // incremental, split anywhere (cut + 3 <= size)
            strix::Sha256Fast h;
            h.update(std::span(data.data(), cut));
            h.update(std::span(data.data() + cut, 3U));
            h.update(std::span(data.data() + cut + 3U, data.size() - cut - 3U));
            same = same && h.finalize() == strix::sha256_bytes(data);
        }
        expect(same, "fast SHA-256 provider matches the portable digest");
        expect(strix::model_content_identity_hex(strix::sha256_bytes_fast(
                   std::as_bytes(std::span(memory_vector.data(), memory_vector.size())))) ==
                   "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
               "fast SHA-256 matches the known vector");
        std::cout << "sha256 fast provider: " << strix::sha256_fast_provider() << '\n';
    }
    const auto root =
        std::filesystem::temp_directory_path() /
        "strix-model-content-hash-test";
    std::error_code error;
    std::filesystem::remove_all(
        root,
        error);
    std::filesystem::create_directories(
        root);

#ifdef _WIN32
    windows_cache_tests(root);
#endif

    const auto empty_path =
        root / "empty.bin";
    const auto abc_path =
        root / "abc.bin";
    write_file(empty_path, "");
    write_file(abc_path, "abc");

    const auto empty =
        strix::sha256_file_content(
            empty_path);
    const auto abc =
        strix::sha256_file_content(
            abc_path);

    expect(
        strix::model_content_identity_hex(
            empty) ==
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855",
        "SHA-256 empty-file vector matches");
    expect(
        strix::model_content_identity_hex(
            abc) ==
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad",
        "SHA-256 abc vector matches");

    const strix::ModelContentFile one_file{"abc.bin",abc_path};
    const auto single=strix::hash_single_model_content_file(one_file);
    const std::vector<strix::ModelContentFile> one_files{one_file};
    expect(single.file_sha256==abc && single.file_bytes==3 &&
        single.content_identity==strix::build_model_content_identity(one_files),
        "one full-file pass preserves raw digest, byte count and historical single-file aggregate framing");
    bool empty_name_rejected=false;
    try{(void)strix::hash_single_model_content_file({"",abc_path});}
    catch(const std::invalid_argument&){empty_name_rejected=true;}
    expect(empty_name_rejected,"single-file digest helper preserves nonempty logical identity contract");
    const std::string million_a(1'000'000, 'a');
    const auto million_path = root / "million-a.bin";
    write_file(million_path, million_a);
    expect(strix::model_content_identity_hex(strix::sha256_file_content(million_path)) ==
               "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
           "SHA-256 million-a known vector matches");
    // The reader alternates two 8 MiB blocks: sizes around one, two and three blocks.
    for (const auto size : {55U, 56U, 63U, 64U, 65U, 1048575U, 1048576U, 1048577U, 2097189U,
             8388607U, 8388608U, 8388609U, 16777216U, 16777217U, 25165825U}) {
        std::string payload(size, '\0');
        for (std::size_t index = 0; index < payload.size(); ++index)
            payload[index] = static_cast<char>((index * 37U + index / 97U) % 256U);
        const auto path = root / "chunk-boundary.bin";
        write_file(path, payload);
        expect(strix::sha256_file_content(path) == strix::sha256_bytes(
                   std::as_bytes(std::span<const char>(payload.data(), payload.size()))),
               "full-file backend matches portable byte oracle across SHA and I/O chunk boundaries");
    }

    const auto shard_a =
        root / "a.gguf";
    const auto shard_b =
        root / "b.gguf";
    write_file(shard_a, "first shard");
    write_file(shard_b, "second shard");

    std::vector<strix::ModelContentFile>
        files{
            {"model-00002.gguf", shard_b},
            {"model-00001.gguf", shard_a},
        };
    const auto identity =
        strix::build_model_content_identity(
            files);

    std::reverse(
        files.begin(),
        files.end());
    const auto reordered =
        strix::build_model_content_identity(
            files);
    expect(identity == reordered,
           "aggregate model identity is stable under input file ordering");

    write_file(
        shard_b,
        "second shard changed");
    const auto changed =
        strix::build_model_content_identity(
            files);
    expect(changed != identity,
           "aggregate model identity changes when shard content changes");

    files[1].logical_name =
        files[0].logical_name;
    bool duplicate_rejected = false;
    try {
        static_cast<void>(
            strix::build_model_content_identity(
                files));
    } catch (const std::invalid_argument&) {
        duplicate_rejected = true;
    }
    expect(duplicate_rejected,
           "aggregate identity rejects duplicate logical shard names");

    std::filesystem::remove_all(
        root,
        error);

    if (failures == 0) {
        std::cout << "model content hash tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
