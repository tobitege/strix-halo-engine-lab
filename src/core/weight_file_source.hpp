#pragma once
// POSIX file source for core::WeightLoadQueue (positional reads, safe from several workers), and the queue's
// environment-resolved defaults.
#include "weight_load_queue.hpp"
#include "strix/env_flags.hpp"

#include <cerrno>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <unistd.h>

namespace strix::core {

class FileWeightSource final : public WeightSource {
public:
    // Unbuffered (O_DIRECT: the weights never enter the page cache) when the file system allows it and `direct`.
    explicit FileWeightSource(const std::string& path, bool direct = true) : owned_(true) {
        if (direct) fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        direct_ = fd_ >= 0;
        if (fd_ < 0) fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("weight load: cannot open " + path + ": " + std::strerror(errno));
    }
    // A descriptor that stays with its owner (a pack reader's files).
    FileWeightSource(int borrowed_fd, bool direct) : fd_(borrowed_fd), direct_(direct), owned_(false) {}
    ~FileWeightSource() override {
        if (owned_ && fd_ >= 0) ::close(fd_);
    }
    FileWeightSource(const FileWeightSource&) = delete;
    FileWeightSource& operator=(const FileWeightSource&) = delete;
    [[nodiscard]] int fd() const { return fd_; }
    [[nodiscard]] bool direct() const { return direct_; }
    [[nodiscard]] std::uint64_t alignment() const override { return direct_ ? WeightLoadManifest::page : 1; }
    std::uint64_t read(std::uint64_t offset, std::uint64_t bytes, std::uint8_t* dst) const override {
        for (;;) {
            const ssize_t n = ::pread(fd_, dst, static_cast<std::size_t>(bytes), static_cast<off_t>(offset));
            if (n >= 0) return static_cast<std::uint64_t>(n);
            if (errno != EINTR) throw std::runtime_error(std::string("weight load: pread: ") + std::strerror(errno));
        }
    }

private:
    int fd_ = -1;
    bool direct_ = false, owned_ = false;
};

// STRIX_LOAD_WORKERS (read workers, default 4) and STRIX_LOAD_STAGING_MIB (resident staging budget; the engine's own
// default when unset). The byte-rate governor is the process one.
inline WeightLoadOptions weight_load_options_from_environment(std::uint64_t default_staging_bytes) {
    WeightLoadOptions o;
    const int workers = envcfg::integer("STRIX_LOAD_WORKERS", 4);
    if (workers < 1 || workers > 64) throw std::invalid_argument("STRIX_LOAD_WORKERS must be 1..64");
    o.workers = static_cast<unsigned>(workers);
    const int mib = envcfg::integer("STRIX_LOAD_STAGING_MIB", 0);
    if (mib < 0) throw std::invalid_argument("STRIX_LOAD_STAGING_MIB must not be negative");
    o.staging_budget_bytes = mib != 0 ? static_cast<std::uint64_t>(mib) << 20 : default_staging_bytes;
    o.rate = &shared_byte_budget();
    o.hash = envcfg::text("STRIX_LOAD_HASH_OUT") != nullptr;
    return o;
}

}  // namespace strix::core
