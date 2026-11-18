#pragma once
// MiniMax H3: minimal safetensors reader (host). Header = u64 LE length + JSON {name: {dtype, shape, data_offsets}},
// tensor bytes follow the header; offsets are relative to the end of the header. Reads are positioned (seek + read)
// under a mutex, so one reader can serve several threads.
#include "strix/weight/admission.hpp"
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>
#include <memory>
#if !defined(_WIN32)
#include <sys/types.h>
#endif

#include "h3/common/fixture.hpp"  // JsonParser
#include "h3/block/checkpoint_names.hpp"

namespace strix::h3::block {

enum class StDtype { f32, f16, bf16, i8, u8, i32, i64, f64, other };

inline StDtype parse_st_dtype(const std::string& s) {
    if (s == "F32") return StDtype::f32;
    if (s == "F16") return StDtype::f16;
    if (s == "BF16") return StDtype::bf16;
    if (s == "I8") return StDtype::i8;
    if (s == "U8") return StDtype::u8;
    if (s == "I32") return StDtype::i32;
    if (s == "I64") return StDtype::i64;
    if (s == "F64") return StDtype::f64;
    return StDtype::other;
}
inline std::size_t st_dtype_size(StDtype d) {
    switch (d) {
        case StDtype::f32: case StDtype::i32: return 4;
        case StDtype::f16: case StDtype::bf16: return 2;
        case StDtype::i8: case StDtype::u8: return 1;
        case StDtype::i64: case StDtype::f64: return 8;
        default: return 0;
    }
}

struct StTensor {
    std::string name, dtype_name;
    StDtype dtype = StDtype::other;
    std::vector<std::int64_t> shape;
    std::uint64_t begin = 0, end = 0;  // absolute file offsets
    std::uint64_t numel() const {
        std::uint64_t n = 1;
        for (auto s : shape) n *= static_cast<std::uint64_t>(s);
        return n;
    }
    std::uint64_t nbytes() const { return end - begin; }
};

class SafeTensors {
public:
    explicit SafeTensors(const std::string& path, CheckpointNames names = CheckpointNames::unchanged) : path_(path) {
        if (std::filesystem::is_directory(path)) {
            // Tensor names must be unique across checkpoint shards. No payload is read here.
            for (const auto& e : std::filesystem::directory_iterator(path)) {
                if (e.path().extension() != ".safetensors") continue;
                auto child = std::make_unique<SafeTensors>(e.path().string(), names);
                for (const auto& [name, t] : child->tensors()) {
                    if (!tensors_.emplace(name, t).second) throw std::runtime_error("safetensors: duplicate sharded tensor " + name);
                    owners_[name] = child.get();
                }
                index_.container = weight::Container::safetensors; index_.schema = "safetensors@1";
                index_.tensors.insert(index_.tensors.end(), child->index_.tensors.begin(), child->index_.tensors.end());
                shards_.push_back(std::move(child));
            }
            if (shards_.empty()) throw std::runtime_error("safetensors: no completed checkpoint shards");
            return;
        }
        auto status = weight::open_source(path, source_);
        weight::TensorIndex index;
        if (status) status = weight::index_safetensors(source_, index);
        if (!status) throw std::runtime_error("safetensors: " + status.detail);
        index_ = index;
        for (const auto& [key,value] : index.metadata) metadata_[key] = value;
        for (const auto& raw : index.tensors) {
            StTensor t; t.name = checkpoint_name(raw.name, names); t.dtype_name = raw.wire_dtype;
            t.dtype = parse_st_dtype(t.dtype_name);
            for (auto dim : raw.extents) t.shape.push_back(static_cast<std::int64_t>(dim));
            t.begin = raw.payload.range.offset; t.end = t.begin + raw.payload.range.bytes;
            const auto canonical = t.name;
            if (!tensors_.emplace(canonical,std::move(t)).second)
                throw std::runtime_error("safetensors: duplicate canonical tensor " + canonical);
        }
    }

    ~SafeTensors() {
        if (f_) std::fclose(f_);
    }
    SafeTensors(const SafeTensors&) = delete;
    SafeTensors& operator=(const SafeTensors&) = delete;

    const std::map<std::string, StTensor>& tensors() const { return tensors_; }
    weight::Status admitted_weights(weight::WeightMap& out) const { return weight::bind_h3(index_,out); }
    const std::map<std::string, std::string>& metadata() const { return metadata_; }
    const StTensor* find(const std::string& name) const {
        auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }
    const StTensor& at(const std::string& name) const {
        const StTensor* t = find(name);
        if (!t) throw std::runtime_error("safetensors: missing tensor " + name);
        return *t;
    }
    // Raw bytes of a tensor into dst (nbytes of the tensor).
    void read_raw(const StTensor& t, void* dst) const {
        if (!shards_.empty()) { owners_.at(t.name)->read_raw(t, dst); return; }
        auto status = source_->read({t.begin,t.nbytes()}, {reinterpret_cast<weight::Byte*>(dst),static_cast<std::size_t>(t.nbytes())});
        if (!status) throw std::runtime_error("safetensors: " + status.detail);
    }

    template <class T> std::vector<T> read_vec(const std::string& name, StDtype expect) const {
        const StTensor& t = at(name);
        if (t.dtype != expect) throw std::runtime_error("safetensors: unexpected dtype " + t.dtype_name + " for " + name);
        std::vector<T> v(static_cast<std::size_t>(t.nbytes() / sizeof(T)));
        read_raw(t, v.data());
        return v;
    }
    const std::string& path() const { return path_; }

private:
    std::vector<std::unique_ptr<SafeTensors>> shards_;
    std::map<std::string, SafeTensors*> owners_;
    std::string path_;
    std::shared_ptr<const weight::Source> source_;
    weight::TensorIndex index_;
    std::FILE* f_ = nullptr;
    std::uint64_t data_base_ = 0;
    std::map<std::string, StTensor> tensors_;
    std::map<std::string, std::string> metadata_;
    mutable std::mutex mu_;
};

}  // namespace strix::h3::block
