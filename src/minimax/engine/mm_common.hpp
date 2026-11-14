#pragma once
// MiniMax engine (fork of the DeepSeek V4.1 TP2 engine): shared pieces. The small FP32 kernels are copied verbatim from
// the minimax-m3 layer gates (tests/minimax_gate_common.hpp) so the engine reproduces the G4 harness
// bit for bit; the safetensors reader replaces the gate's mmap Pack with O_DIRECT reads (the DS DirectLoader pattern):
// no file page is ever handed to HIP, and resident weights do not compete with the page cache.
#include <cstdint>
#include <memory>
#include "core/weight_file_source.hpp"
#include "strix/file_snapshot.hpp"
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::minimax::engine::detail {

constexpr unsigned kHidden = 6144, kQ = 64 * 128, kKV = 4 * 128, kIdxQ = 4 * 128, kDense = 12288, kExpertI = 3072;
constexpr unsigned kLayers = 60, kFirstMoe = 3, kExperts = 128, kTopE = 4;
constexpr float kEps = 1e-6F, kAlpha = 1.702F, kLimit = 7.0F;
constexpr std::uint64_t kVocab = 200064, kVocabPad = 782 * 128;  // rank 0's vocabulary share (H128 blocks)
inline const std::string kP = "language_model.model.";

// ---------------------------------------------------------------- safetensors pack, O_DIRECT reads
struct TensorInfo {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint32_t file = 0;
    std::uint64_t offset = 0;  // absolute byte offset in the file
    std::uint64_t bytes = 0;
};
class PackReader {
public:
    explicit PackReader(std::string dir);
    ~PackReader();
    PackReader(const PackReader&) = delete;
    PackReader& operator=(const PackReader&) = delete;
    const TensorInfo& info(const std::string& name) const;
    // Bytes [offset, offset + n) of the tensor into dst (any host memory, typically the pinned staging buffer).
    void read(const TensorInfo& t, std::uint64_t offset, std::uint64_t n, void* dst);
    // Plain tensors (BF16 / F16 / F32) converted to FP32, as the gates' Pack::floats.
    std::vector<float> floats(const std::string& name);
    // The tensor's file for core::WeightLoadQueue: the same descriptor and O_DIRECT mode, read without the bounce
    // buffer from any number of workers. count_read keeps gib_read() covering the bytes the queue reads.
    const strix::core::WeightSource& source(const TensorInfo& t) const { return *sources_.at(t.file); }
    void count_read(std::uint64_t n) { bytes_read_ += n; }
    const LoaderSnapshot& loader_snapshot() const { return snapshot_; }
    double gib_read() const { return static_cast<double>(bytes_read_) / 1073741824.0; }

private:
    LoaderSnapshot snapshot_;
    std::string dir_;
    std::vector<int> fds_;
    std::vector<int> direct_;  // 1 if opened with O_DIRECT
    std::vector<std::unique_ptr<strix::core::FileWeightSource>> sources_;
    std::map<std::string, TensorInfo> tensors_;
    std::uint8_t* bounce_ = nullptr;
    std::uint64_t bytes_read_ = 0;
};

}  // namespace strix::minimax::engine::detail
