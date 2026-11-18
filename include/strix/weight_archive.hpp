#pragma once

#include "strix/gguf_reader.hpp"
#include "strix/weight/admission.hpp"
#include "strix/serve/json.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strix::weights {
// Host storage shared by format readers and all device uploaders. Views keep it alive.
class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    [[nodiscard]] const std::uint8_t* data() const { return data_; }
    [[nodiscard]] std::size_t size() const { return size_; }
private:
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
#ifdef _WIN32
    void* file_ = nullptr;
    void* mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};
struct SafeTensor {
    std::string name, dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t offset = 0, bytes = 0;
};
struct SafeFile {
    std::shared_ptr<MappedFile> storage;
    std::vector<SafeTensor> tensors;
    weight::TensorIndex index;
};
[[nodiscard]] serve::Json read_json(const std::filesystem::path& path);
[[nodiscard]] SafeFile read_safetensors(const std::filesystem::path& path);

struct Tensor {
    std::shared_ptr<MappedFile> storage;
    std::vector<std::uint64_t> shape; // row-major; GGUF extents are reversed once on import
    std::string dtype;             // safetensors dtype, or "GGUF"
    std::uint32_t ggml_type = 0;
    std::uint64_t offset = 0, size = 0;
    [[nodiscard]] std::span<const std::byte> bytes() const;
};
using Archive = std::map<std::string, Tensor>;
[[nodiscard]] Archive load_gguf(const std::filesystem::path& path, const gguf::FileInfo& info);
// A file, or directory with model.safetensors[.index.json]. Index shards must be local basenames.
[[nodiscard]] Archive load_safetensors(const std::filesystem::path& path);
[[nodiscard]] std::vector<float> decode_row(const Tensor& tensor, std::uint64_t row);

struct Nvfp4 {
    Tensor packed, block_scales, global_scale;
    std::uint64_t rows = 0, columns = 0;
    static constexpr std::uint64_t block_size = 16;
    // Adjacent columns use low then high nibble. No transpose or backend swizzle.
    [[nodiscard]] std::vector<float> decode_row(std::uint64_t row) const;
};
// Export layout: base.weight U8 [O,I/2], base.weight_scale F8_E4M3 [O,I/16],
// base.weight_scale_2 F32 scalar. Global scale is a dequantization multiplier.
[[nodiscard]] Nvfp4 nvfp4(const Archive& archive, const std::string& base,
                         std::uint64_t rows, std::uint64_t columns);
} // namespace strix::weights
