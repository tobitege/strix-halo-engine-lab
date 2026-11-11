#pragma once

#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/runtime_allocator.hpp"

#include <cstddef>
#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <utility>

namespace strix {

class MappedModelFile {
public:
    MappedModelFile() = default;
    MappedModelFile(
        const MappedModelFile&) = delete;
    MappedModelFile& operator=(
        const MappedModelFile&) = delete;

    MappedModelFile(
        MappedModelFile&& other) noexcept;
    MappedModelFile& operator=(
        MappedModelFile&& other) noexcept;

    ~MappedModelFile();

    [[nodiscard]] static MappedModelFile open_read_only(
        const std::filesystem::path& path,
        std::string logical_name,
        AllocationGenerationSource& generations);

    [[nodiscard]] bool valid() const noexcept {
        return mapping_base_ != 0;
    }

    [[nodiscard]] std::uintptr_t base_address() const noexcept {
        return mapping_base_;
    }

    [[nodiscard]] std::uint64_t bytes() const noexcept {
        return bytes_;
    }

    [[nodiscard]] std::uint64_t allocation_generation() const noexcept {
        return allocation_generation_;
    }

    [[nodiscard]] const ModelContentIdentity&
    content_identity() const noexcept {
        return content_identity_;
    }

    // Actual raw digest from the SAME full-file pass/FD used to establish the
    // framed content identity at open; invalid on empty/moved-from owners.
    [[nodiscard]] const ModelContentIdentity& file_sha256() const noexcept {return file_sha256_;}
    // Stable in-process eviction key for the actual backing, including aliases.
    [[nodiscard]] std::array<std::uint64_t,2> file_backing_identity() const noexcept {
        return {file_state_[0], file_state_[1]};
    }
    // Setup-only FD/stat witness against the original hash+map observation.
    // No rehash and no claim of protection against arbitrary privileged writers.
    // Windows compares volume, file index, size, write and creation time; a same-size write
    // through a handle that is still open is not seen (NTFS completes the write time on close).
    void validate_unchanged() const;
    [[nodiscard]] const std::string&
    logical_name() const noexcept {
        return logical_name_;
    }

    [[nodiscard]] ModelRuntimeBinding
    build_runtime_binding(
        const ModelManifest& manifest) const;

    // Reopen the actual mapped inode as an owned O_DIRECT descriptor. Throws
    // before rolling admission on unsupported filesystems; never buffered.
    [[nodiscard]] int open_direct_read_only() const;

    void read_at(
        std::uint64_t byte_offset,
        std::span<std::byte> destination) const;

    // Drops the unmapped clean page-cache pages of [byte_offset, byte_offset + bytes) (POSIX_FADV_DONTNEED).
    // Advisory, no error: used before a large copy arena is allocated (STRIX_MODEL_COPY_ALL).
    void advise_dontneed(std::uint64_t byte_offset, std::uint64_t bytes) const noexcept;
    // Same drop, but reports failure: 0 on success, else an errno value (EBADF without an open descriptor,
    // EINVAL for a range outside the file, or posix_fadvise's own error). Used by STRIX_PLE_RECLAIMABLE.
    [[nodiscard]] int drop_page_cache(std::uint64_t byte_offset, std::uint64_t bytes) const noexcept;
    // Best-effort read of the given [byte_offset, byte_offset + bytes) ranges of the mapping into the page cache from up
    // to 16 threads, in 64 MiB chunks with normal readahead advice (MADV_POPULATE_READ; MADV_WILLNEED where that
    // fails). A cold page cache otherwise faults the file in from one thread. Resident pages are only mapped. The bytes
    // are unchanged; residency afterwards is not guaranteed. Returns the chunks populated synchronously.
    std::size_t prefault_parallel(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) const noexcept;

private:
    void release() noexcept;

    int fd_ = -1;
    std::uintptr_t mapping_base_ = 0;
    std::uint64_t bytes_ = 0;
    std::uint64_t allocation_generation_ = 0;
    std::string logical_name_;
    ModelContentIdentity content_identity_;
    ModelContentIdentity file_sha256_;
    std::array<std::uint64_t,7> file_state_{};
};

}  // namespace strix
