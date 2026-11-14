#include "strix/vision/gguf_weights.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <istream>
#include <limits>
#include <streambuf>
#include <string>
#include <stdexcept>
#include <utility>

namespace strix::vision {
namespace {

// Admission before allocation: nothing is allocated from the file's own length until that length is inside the
// budget and the header (bounded by `limits`) describes a well-formed file of exactly that length.
std::vector<std::byte> read_all(const std::filesystem::path& path, const gguf::Limits& limits, const HostBudget& budget) {
    const std::string name = path.filename().string();
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in) throw std::runtime_error("vision weights: cannot open " + name);
    in.seekg(0, std::ios::end);
    const auto end = in.tellg();
    if (end < 0) throw std::runtime_error("vision weights: cannot size " + name);
    const auto size = static_cast<std::uint64_t>(end);
    if (budget.exact_bytes != 0 && size != budget.exact_bytes)
        throw std::invalid_argument("vision weights: " + name + " is " + std::to_string(size) + " bytes, the admitted file is " +
                                    std::to_string(budget.exact_bytes));
    if (size > budget.max_bytes || std::cmp_greater(size, std::numeric_limits<std::size_t>::max()) ||
        std::cmp_greater(size, std::numeric_limits<std::streamsize>::max()))
        throw std::invalid_argument("vision weights: " + name + " is " + std::to_string(size) + " bytes, over the host budget of " +
                                    std::to_string(budget.max_bytes));
    in.seekg(0, std::ios::beg);
    (void)gguf::read(in, size, limits);  // throws on a malformed header; the tensor table is re-read from the bytes
    in.clear();
    in.seekg(0, std::ios::beg);
    if (!in) throw std::runtime_error("vision weights: cannot rewind " + name);
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    std::size_t done = 0;
    while (done < bytes.size()) {  // bounded chunks: one read of several GiB is not portable
        const std::size_t chunk = std::min<std::size_t>(bytes.size() - done, std::size_t{256} << 20U);
        in.read(reinterpret_cast<char*>(bytes.data() + done), static_cast<std::streamsize>(chunk));
        if (static_cast<std::size_t>(in.gcount()) != chunk) throw std::runtime_error("vision weights: short read");
        done += chunk;
    }
    return bytes;
}

struct ViewStream final : std::streambuf {
    explicit ViewStream(const std::vector<std::byte>& b) {
        char* p = const_cast<char*>(reinterpret_cast<const char*>(b.data()));
        setg(p, p, p + b.size());
    }
    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode) override {
        char* base = dir == std::ios_base::beg ? eback() : dir == std::ios_base::cur ? gptr() : egptr();
        if (off < eback() - base || off > egptr() - base) return pos_type(off_type(-1));
        setg(eback(), base + off, egptr());
        return pos_type(gptr() - eback());
    }
    pos_type seekpos(pos_type pos, std::ios_base::openmode mode) override { return seekoff(off_type(pos), std::ios_base::beg, mode); }
};

gguf::FileInfo parse(const std::vector<std::byte>& bytes, const gguf::Limits& limits) {
    ViewStream view(bytes);
    std::istream in(&view);
    return gguf::read(in, bytes.size(), limits);
}

}  // namespace

GgufWeights::GgufWeights(const std::filesystem::path& path, const gguf::Limits& limits, const HostBudget& budget)
    : bytes_(read_all(path, limits, budget)) {
    info_ = parse(bytes_, limits);
    index();
}

GgufWeights::GgufWeights(gguf::FileInfo info, std::vector<std::byte> bytes) : info_(std::move(info)), bytes_(std::move(bytes)) {
    index();
}

void GgufWeights::index() {
    for (std::size_t i = 0; i < info_.tensors.size(); ++i)
        if (!by_name_.emplace(info_.tensors[i].name, i).second)
            throw std::invalid_argument("vision weights: duplicate tensor " + info_.tensors[i].name);
}

bool GgufWeights::has(std::string_view name) const { return by_name_.find(std::string(name)) != by_name_.end(); }

Tensor GgufWeights::tensor(std::string_view name) {
    const auto it = by_name_.find(std::string(name));
    if (it == by_name_.end()) throw std::out_of_range("vision weights: no tensor " + std::string(name));
    const gguf::TensorInfo& t = info_.tensors[it->second];
    Tensor out;
    std::size_t element = 0;
    switch (t.ggml_type) {
        case 0: out.type = Tensor::Type::f32; element = 4; break;
        case 1: out.type = Tensor::Type::f16; element = 2; break;
        case 30: out.type = Tensor::Type::bf16; element = 2; break;
        default: throw std::invalid_argument("vision weights: tensor " + t.name + " has an unsupported storage type");
    }
    if (t.dimensions.empty()) throw std::invalid_argument("vision weights: tensor " + t.name + " has no dimensions");
    std::uint64_t rows = t.dimensions.back(), cols = 1;
    if (t.dimensions.size() == 1) {
        cols = 1;
    } else {
        for (std::size_t d = 0; d + 1 < t.dimensions.size(); ++d) {
            if (t.dimensions[d] != 0 && cols > std::numeric_limits<std::uint64_t>::max() / t.dimensions[d])
                throw std::invalid_argument("vision weights: tensor " + t.name + " is too large");
            cols *= t.dimensions[d];
        }
    }
    if (rows == 0 || cols == 0 || rows > std::numeric_limits<std::uint64_t>::max() / cols / element ||
        rows * cols * element != t.byte_size || t.absolute_offset > bytes_.size() ||
        t.byte_size > bytes_.size() - t.absolute_offset || t.absolute_offset % element != 0)
        throw std::invalid_argument("vision weights: tensor " + t.name + " does not fit the file");
    out.data = bytes_.data() + t.absolute_offset;
    out.rows = static_cast<std::size_t>(rows);
    out.cols = static_cast<std::size_t>(cols);
    return out;
}

}  // namespace strix::vision
