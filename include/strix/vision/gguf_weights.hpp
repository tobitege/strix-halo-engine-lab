#pragma once
// WeightSource over one GGUF file held in host memory: the tensor lookup every tower shipped as GGUF shares.
// A tensor with GGUF dimensions (d0, ..., dn) is the row-major matrix [dn][d0 * ... * d(n-1)], which is the layout
// of the source framework's [out][in...] weight; a 1-D tensor is [d0][1]. Storage types: F32, F16, BF16.
#include "strix/gguf_reader.hpp"
#include "strix/vision/adapter.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace strix::vision {

// Host memory one weight file may take. The file's length is untrusted input: it is compared with this budget before
// a byte is allocated, so a huge (sparse, padded or simply wrong) file is refused unread whatever its header says.
struct HostBudget {
    std::uint64_t max_bytes = std::uint64_t{4} << 30U;  // largest file held in host memory
    std::uint64_t exact_bytes = 0;                      // nonzero: the admitted file's length; any other is refused
};

class GgufWeights final : public WeightSource {
public:
    // Reads the whole file, in this order on one open file: length against the budget, the bounded header through
    // the reader, then the bytes. Throws std::invalid_argument on a file that is over the budget or not of the
    // admitted length, and std::runtime_error on I/O failure or a header the reader refuses.
    explicit GgufWeights(const std::filesystem::path& path, const gguf::Limits& limits = {}, const HostBudget& budget = {});
    // A file already in memory (tests, callers that hashed the bytes first).
    GgufWeights(gguf::FileInfo info, std::vector<std::byte> bytes);

    bool has(std::string_view name) const override;
    Tensor tensor(std::string_view name) override;  // std::out_of_range when absent, std::invalid_argument when unusable

    const gguf::FileInfo& info() const noexcept { return info_; }
    const std::vector<std::byte>& bytes() const noexcept { return bytes_; }

private:
    void index();
    gguf::FileInfo info_;
    std::vector<std::byte> bytes_;
    std::unordered_map<std::string, std::size_t> by_name_;
};

}  // namespace strix::vision
