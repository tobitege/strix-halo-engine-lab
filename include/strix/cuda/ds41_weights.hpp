#pragma once

// Tensors of a deepseek41 GGUF (the full model or a subset with the same names) by name: host bytes of a tensor or of
// part of it (portable stream reads, no mapping), and a device copy on first use (uploaded in 256 MB pieces).
// Optional requantization overlay (strix/ds41_weight_overlay.hpp semantics): a small GGUF whose tensors replace tensors
// of the model by name, each with the model's shape, storage type and byte size (only the bytes differ), so every
// kernel and declared order is unchanged. Routed-expert tensors may not be overlaid (the expert readers use the model
// file's offsets).

#include "strix/cuda/ds41_common.hpp"

#include "strix/deepseek41_model.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::ds41::cuda {

class TensorFile {
public:
    explicit TensorFile(std::string path, std::string overlay = {}) : path_(std::move(path)), info_(deepseek41::read_file(path_)) {
        for (const auto& t : info_.tensors) index_[t.name] = &t;
        if (overlay.empty()) return;
        overlay_path_ = std::move(overlay);
        overlay_info_ = gguf::read_file(overlay_path_);
        for (const auto& t : overlay_info_.tensors) {
            const auto it = index_.find(t.name);
            if (it == index_.end()) throw std::runtime_error("weights overlay: " + t.name + " is not a model tensor");
            const auto& m = *it->second;
            if (m.dimensions != t.dimensions || m.ggml_type != t.ggml_type || m.byte_size != t.byte_size)
                throw std::runtime_error("weights overlay: " + t.name + " differs from the model in shape, type or size");
            if (t.name.ends_with("_exps.weight")) throw std::runtime_error("weights overlay: routed-expert tensors are not supported");
            overlay_index_[t.name] = &t;
            overlay_bytes_ += t.byte_size;
        }
        if (const auto* id = overlay_info_.find_metadata("ds41.requant.identity"); id != nullptr && id->string_value)
            overlay_identity_ = *id->string_value;
        else
            throw std::runtime_error("weights overlay: missing ds41.requant.identity");
    }
    // "" without an overlay, else: path, identity, tensor count and bytes
    [[nodiscard]] std::string overlay_description() const {
        if (overlay_path_.empty()) return {};
        return overlay_path_ + " identity " + overlay_identity_ + " tensors " + std::to_string(overlay_index_.size()) + " bytes " +
               std::to_string(overlay_bytes_);
    }
    [[nodiscard]] const gguf::FileInfo& info() const { return info_; }
    [[nodiscard]] const gguf::TensorInfo& tensor(const std::string& name) const {
        const auto t = index_.find(name);
        if (t == index_.end()) throw std::runtime_error("weights file has no tensor " + name);
        return *t->second;
    }
    [[nodiscard]] deepseek41::Config config() const {
        const auto* e = info_.find_metadata("deepseek41.config");
        if (!e || !e->string_value) throw std::runtime_error("weights file has no deepseek41.config");
        return deepseek41::parse_config_json(*e->string_value);
    }
    // bytes [offset, offset + size) of a tensor (size 0 = to its end)
    [[nodiscard]] std::vector<std::uint8_t> host(const std::string& name, std::uint64_t offset = 0, std::uint64_t size = 0) const {
        const auto& t = tensor(name);
        if (size == 0) size = t.byte_size - offset;
        if (offset + size > t.byte_size) throw std::runtime_error("read outside tensor " + name);
        std::vector<std::uint8_t> out(size);
        const auto o = overlay_index_.find(name);
        const bool over = o != overlay_index_.end();
        std::ifstream in(over ? overlay_path_ : path_, std::ios::binary);
        in.seekg(static_cast<std::streamoff>((over ? o->second->absolute_offset : t.absolute_offset) + offset));
        in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(size));
        if (!in) throw std::runtime_error("cannot read tensor " + name);
        return out;
    }
    template <class T>
    [[nodiscard]] std::vector<T> host_as(const std::string& name) const {
        const auto b = host(name);
        std::vector<T> out(b.size() / sizeof(T));
        std::copy_n(b.data(), out.size() * sizeof(T), reinterpret_cast<std::uint8_t*>(out.data()));
        return out;
    }
    const std::uint8_t* device(const std::string& name) {
        auto it = device_.find(name);
        if (it != device_.end()) return it->second.get();
        const auto& t = tensor(name);
        DeviceBuffer<std::uint8_t> buf(t.byte_size);
        constexpr std::uint64_t piece = 256ULL << 20;
        for (std::uint64_t done = 0; done < t.byte_size; done += piece) {
            const auto part = host(name, done, std::min(piece, t.byte_size - done));
            buf.upload(part.data(), part.size(), done);
        }
        return device_.emplace(name, std::move(buf)).first->second.get();
    }
    const float* device_f32(const std::string& name) { return reinterpret_cast<const float*>(device(name)); }
    const std::uint16_t* device_f16(const std::string& name) { return reinterpret_cast<const std::uint16_t*>(device(name)); }
    void release(const std::string& name) { device_.erase(name); }

private:
    std::string path_;
    gguf::FileInfo info_;
    std::map<std::string, const gguf::TensorInfo*> index_;
    std::string overlay_path_, overlay_identity_;
    gguf::FileInfo overlay_info_;
    std::map<std::string, const gguf::TensorInfo*> overlay_index_;
    std::uint64_t overlay_bytes_ = 0;
    std::map<std::string, DeviceBuffer<std::uint8_t>> device_;
};

}  // namespace strix::ds41::cuda
