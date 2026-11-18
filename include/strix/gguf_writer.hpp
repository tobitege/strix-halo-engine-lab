#pragma once
// Minimal streaming GGUF v3 writer for derived packs (converted weights): the key/value section is copied byte for byte from the
// source file (so every config / tokenizer key stays identical), the tensor table is rebuilt, tensor data is appended in table order.
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/gguf_reader.hpp"

namespace strix::gguf {

struct OutTensor {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint32_t ggml_type = 0;
    std::uint64_t bytes = 0;
    std::uint64_t offset = 0;  // relative to the data section; filled by the writer
};

class PackWriter {
public:
    // `source` is the parsed header of `source_path` (its key/value bytes are copied). `tensors` is the output table, in data order.
    PackWriter(const std::string& out_path, const std::string& source_path, const FileInfo& source, std::vector<OutTensor> tensors)
        : out_(out_path, std::ios::binary | std::ios::trunc), tensors_(std::move(tensors)), align_(source.alignment) {
        if (!out_) throw std::runtime_error("cannot create " + out_path);
        std::ifstream src(source_path, std::ios::binary);
        if (!src) throw std::runtime_error("cannot open " + source_path);
        std::vector<char> kv(static_cast<std::size_t>(source.metadata_end - 24));
        src.seekg(24);
        src.read(kv.data(), static_cast<std::streamsize>(kv.size()));
        if (!src) throw std::runtime_error("short read of GGUF metadata");
        std::uint64_t rel = 0;
        for (auto& t : tensors_) {
            t.offset = rel;
            rel += pad(t.bytes);
        }
        data_bytes_ = rel;
        put(&kMagic, 4);
        u32(3);
        u64(tensors_.size());
        u64(source.metadata_count);
        out_.write(kv.data(), static_cast<std::streamsize>(kv.size()));
        for (const auto& t : tensors_) {
            u64(t.name.size());
            out_.write(t.name.data(), static_cast<std::streamsize>(t.name.size()));
            u32(static_cast<std::uint32_t>(t.dimensions.size()));
            for (auto d : t.dimensions) u64(d);
            u32(t.ggml_type);
            u64(t.offset);
        }
        pos_ = 24 + kv.size();
        for (const auto& t : tensors_) pos_ += 8 + t.name.size() + 4 + 8 * t.dimensions.size() + 4 + 8;
        pad_to(align_up(pos_));
        data_start_ = pos_;
    }
    // Append bytes of tensor `i` (call in order; any number of chunks); finish_tensor() pads to the alignment.
    void append(const void* p, std::uint64_t n) { out_.write(static_cast<const char*>(p), static_cast<std::streamsize>(n)); pos_ += n; written_ += n; }
    void finish_tensor() {
        if (written_ != tensors_.at(cur_).bytes) throw std::runtime_error("pack writer: size mismatch for " + tensors_[cur_].name);
        pad_to(align_up(pos_));
        written_ = 0;
        ++cur_;
    }
    std::uint64_t close() {
        if (cur_ != tensors_.size()) throw std::runtime_error("pack writer: tensors missing");
        out_.flush();
        if (!out_) throw std::runtime_error("pack writer: write failed");
        return pos_;
    }
    [[nodiscard]] std::uint64_t data_start() const { return data_start_; }

private:
    static constexpr char kMagic[4] = {'G', 'G', 'U', 'F'};
    [[nodiscard]] std::uint64_t align_up(std::uint64_t v) const { return (v + align_ - 1) / align_ * align_; }
    [[nodiscard]] std::uint64_t pad(std::uint64_t v) const { return align_up(v); }
    void pad_to(std::uint64_t target) {
        static const char zeros[64] = {};
        while (pos_ < target) {
            const auto n = std::min<std::uint64_t>(64, target - pos_);
            out_.write(zeros, static_cast<std::streamsize>(n));
            pos_ += n;
        }
    }
    void put(const void* p, std::size_t n) { out_.write(static_cast<const char*>(p), static_cast<std::streamsize>(n)); }
    void u32(std::uint32_t v) { put(&v, 4); }  // little-endian host (x86)
    void u64(std::uint64_t v) { put(&v, 8); }
    std::ofstream out_;
    std::vector<OutTensor> tensors_;
    std::uint64_t align_ = 32, pos_ = 0, data_start_ = 0, data_bytes_ = 0, written_ = 0;
    std::size_t cur_ = 0;
};

}  // namespace strix::gguf
