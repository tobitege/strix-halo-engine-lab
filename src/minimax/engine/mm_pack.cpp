// MiniMax engine: safetensors pack index and O_DIRECT reads (see mm_common.hpp).
#include "mm_common.hpp"
#include "strix/minimax/model_assets.hpp"
#include "strix/minimax/model_description.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <set>

namespace strix::minimax::engine::detail {

namespace {

constexpr std::uint64_t kAlign = 4096;
constexpr std::uint64_t kBounce = std::uint64_t{64} << 20;

// Keep ownership on the constructor stack until PackReader is fully initialized.
struct OwnedFd {
    int value;
    explicit OwnedFd(int fd) noexcept : value(fd) {}
    ~OwnedFd() { reset(); }
    OwnedFd(const OwnedFd&) = delete;
    OwnedFd& operator=(const OwnedFd&) = delete;
    OwnedFd(OwnedFd&& other) noexcept : value(other.release()) {}
    int release() noexcept { const int fd = value; value = -1; return fd; }
    void reset(int fd = -1) noexcept { if (value >= 0) ::close(value); value = fd; }
};

void pread_all(int fd, void* dst, std::uint64_t n, std::uint64_t off, const std::string& what) {
    auto* d = static_cast<std::uint8_t*>(dst);
    while (n > 0) {
        const ssize_t r = ::pread(fd, d, static_cast<std::size_t>(n), static_cast<off_t>(off));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) throw std::runtime_error("pread " + what + ": " + std::strerror(errno));
        d += r;
        off += static_cast<std::uint64_t>(r);
        n -= static_cast<std::uint64_t>(r);
    }
}

float bf16f(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16U;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace

PackReader::PackReader(std::string dir) : dir_(std::move(dir)) {
    const auto files = assets::pack_files(dir_, &snapshot_);
    const bool want_direct = std::getenv("STRIX_MM_LOAD_DIRECT") == nullptr || std::atoi(std::getenv("STRIX_MM_LOAD_DIRECT")) != 0;
    description::PackHeaders headers(description::check_enabled());
    std::set<std::pair<std::uint64_t,std::uint64_t>> indexed_files;
    weight::TensorIndex common; common.container = weight::Container::safetensors; common.schema = "safetensors@1";
    std::vector<OwnedFd> opened;
    opened.reserve(files.size());
    for (std::uint32_t f = 0; f < files.size(); ++f) {
        const std::string path = files[f].string();
        OwnedFd execution(want_direct ? ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT) : -1);
        int fd = execution.value;
        direct_.push_back(fd >= 0 ? 1 : 0);
        if (fd < 0) { execution.reset(::open(path.c_str(), O_RDONLY | O_CLOEXEC)); fd = execution.value; }
        if (fd < 0) throw std::runtime_error("open " + path);
        opened.push_back(std::move(execution));
        fds_.push_back(fd);
        snapshot_.push_back(loaded_file(path, fd, true));
        // Header: 8-byte length + JSON, read through the aligned bounce once it exists; a plain read is simpler here.
        OwnedFd header(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
        const int hfd = header.value;
        if (hfd < 0) throw std::runtime_error("open " + path);
        snapshot_.push_back(loaded_file(path, hfd));
        struct stat st{};
        if (::fstat(hfd, &st) != 0 || st.st_size < 8) {
            throw std::runtime_error("invalid safetensors header length: " + path);
        }
        // The asset receipt retains every spelling, but aliases of one opened
        // physical shard contribute its tensor table exactly once.
        if (!indexed_files.emplace(static_cast<std::uint64_t>(st.st_dev),static_cast<std::uint64_t>(st.st_ino)).second) {
            continue;
        }
        std::uint64_t hb = 0;
        pread_all(hfd, &hb, 8, 0, path);
        const auto payload_bytes = description::header_payload_bytes(hb, static_cast<std::uint64_t>(st.st_size));
        std::string head(hb, '\0');
        pread_all(hfd, head.data(), hb, 8, path);
        header.reset();
        std::shared_ptr<const weight::Source> owner;
        auto status = weight::open_source(path, owner, std::filesystem::absolute(files[f]).lexically_relative(std::filesystem::absolute(dir_)).generic_string());
        weight::TensorIndex shard;
        if (status) status = weight::index_safetensors(owner, shard);
        if (!status) throw std::runtime_error("MiniMax index: " + status.detail);
        common.tensors.insert(common.tensors.end(), shard.tensors.begin(), shard.tensors.end());
        for (const auto& entry : headers.add(head, payload_bytes)) {
            TensorInfo t;
            t.dtype = entry.dtype;
            t.shape = entry.shape;
            t.file = f;
            t.offset = 8 + hb + entry.begin;
            t.bytes = entry.end - entry.begin;
            if (!tensors_.emplace(entry.name, std::move(t)).second)
                throw std::runtime_error("duplicate tensor across safetensors shards: " + entry.name);
        }
    }
    auto admission = weight::bind_minimax(common, weights_);
    if (!admission) throw std::runtime_error("MiniMax admission: " + admission.detail);
    for (std::size_t f = 0; f < fds_.size(); ++f)
        sources_.push_back(std::make_unique<strix::core::FileWeightSource>(fds_[f], direct_[f] != 0));
    void* p = nullptr;
    if (posix_memalign(&p, kAlign, kBounce) != 0) throw std::runtime_error("bounce buffer");
    bounce_ = static_cast<std::uint8_t*>(p);
    for (auto& fd : opened) (void)fd.release();
}

PackReader::~PackReader() {
    for (int fd : fds_) ::close(fd);
    std::free(bounce_);
}

const TensorInfo& PackReader::info(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("tensor not in pack: " + name);
    return it->second;
}

void PackReader::read(const TensorInfo& t, std::uint64_t offset, std::uint64_t n, void* dst) {
    if (offset + n > t.bytes) throw std::runtime_error("read past tensor end");
    const int fd = fds_[t.file];
    std::uint64_t pos = t.offset + offset;
    auto* d = static_cast<std::uint8_t*>(dst);
    bytes_read_ += n;
    if (!direct_[t.file]) {
        pread_all(fd, d, n, pos, dir_);
        return;
    }
    // O_DIRECT: aligned windows through the bounce buffer.
    while (n > 0) {
        const std::uint64_t a0 = pos & ~(kAlign - 1), skip = pos - a0;
        const std::uint64_t want = std::min<std::uint64_t>(n, kBounce - skip);
        const std::uint64_t len = (skip + want + kAlign - 1) & ~(kAlign - 1);
        std::uint64_t got = 0;
        while (got < len) {  // a short read only at the end of the file
            const ssize_t r = ::pread(fd, bounce_ + got, static_cast<std::size_t>(len - got), static_cast<off_t>(a0 + got));
            if (r < 0 && errno == EINTR) continue;
            if (r < 0) throw std::runtime_error(std::string("pread (O_DIRECT): ") + std::strerror(errno));
            if (r == 0) break;
            got += static_cast<std::uint64_t>(r);
        }
        if (got < skip + want) throw std::runtime_error("short O_DIRECT read");
        std::memcpy(d, bounce_ + skip, want);
        d += want;
        pos += want;
        n -= want;
    }
}

std::vector<float> PackReader::floats(const std::string& name) {
    const TensorInfo& t = info(name);
    std::vector<std::uint8_t> raw(t.bytes);
    read(t, 0, t.bytes, raw.data());
    std::vector<float> out;
    if (t.dtype == "BF16") {
        out.resize(t.bytes / 2);
        for (std::size_t i = 0; i < out.size(); ++i) {
            std::uint16_t b;
            std::memcpy(&b, raw.data() + 2 * i, 2);
            out[i] = bf16f(b);
        }
    } else if (t.dtype == "F32") {
        out.resize(t.bytes / 4);
        std::memcpy(out.data(), raw.data(), t.bytes);
    } else if (t.dtype == "F16") {
        out.resize(t.bytes / 2);
        for (std::size_t i = 0; i < out.size(); ++i) {
            _Float16 h;
            std::memcpy(&h, raw.data() + 2 * i, 2);
            out[i] = static_cast<float>(h);
        }
    } else {
        throw std::runtime_error("unsupported dtype " + t.dtype + " for " + name);
    }
    return out;
}

}  // namespace strix::minimax::engine::detail
