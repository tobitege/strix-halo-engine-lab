#pragma once
// MiniMax M3.1 Flash EXL3 pack reader for the RTX PRO prefill: safetensors index + headers,
// plain reads of tensor byte ranges, and the engine's rank slices of an EXL3 linear (src/minimax/engine/engine.hip
// Lib::stage_out / stage_in, G4 layout), so both TP halves can be computed here exactly as the two nodes do.
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace strix::minimax::rtxpf::host {

struct TensorInfo {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint32_t file = 0;
    std::uint64_t offset = 0;  // absolute byte offset in the file
    std::uint64_t bytes = 0;
};

// One EXL3 matrix slice in host memory (engine device layout = file layout): trellis [input/16][output/16][bits*8] u32,
// suh [input] FP16, svh [output] FP16.
struct Exl3Host {
    std::vector<std::uint8_t> trellis;
    std::vector<std::uint16_t> suh, svh;
    std::uint32_t input = 0, output = 0, bits = 0;
};

class Pack {
public:
    explicit Pack(std::string dir, unsigned prefix_layers = 0);  // 0: original full pack
    ~Pack();
    Pack(const Pack&) = delete;
    Pack& operator=(const Pack&) = delete;
    const TensorInfo& info(const std::string& name) const;
    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    void read(const TensorInfo& t, std::uint64_t offset, std::uint64_t n, void* dst);
    std::vector<float> floats(const std::string& name);  // BF16 / F16 / F32 tensor as FP32
    // Output slice [o0, o0 + n): those outputs' tiles of every input row, matching svh (H128-aligned).
    Exl3Host slice_out(const std::string& prefix, std::uint64_t o0, std::uint64_t n);
    // Input slice [i0, i0 + n): the contiguous trellis rows, matching suh (a partial sum, reduced across ranks).
    Exl3Host slice_in(const std::string& prefix, std::uint64_t i0, std::uint64_t n);
    double gib_read() const { return static_cast<double>(bytes_read_) / 1073741824.0; }

private:
    std::string dir_;
    std::vector<std::FILE*> files_;
    std::map<std::string, TensorInfo> tensors_;
    std::uint64_t bytes_read_ = 0;
    std::string whole_name_;  // slice_out: the last whole trellis read (the other rank's slice reuses it)
    std::vector<std::uint8_t> whole_;
};

float half_to_float(std::uint16_t h);

}  // namespace strix::minimax::rtxpf::host
