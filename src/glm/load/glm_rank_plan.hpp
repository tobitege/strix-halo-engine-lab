#pragma once
// GLM-5.3-Flash TP2 rank-slice plan (design notes §5.2): which file bytes
// each rank needs, as byte "pieces", from the split contract (tools/glm/split_plan.py -> design notes)
// and the safetensors headers. A rank's slice of a tensor is the concatenation of its pieces, in this layout:
//   EXL3 group, mode out [o-ranges]: trellis tiles [K/16][n/16][w] of every output range (per input tile row: the
//       range's n/16 tiles, ranges in order), then suh whole, then svh of the ranges, then mul1;
//   EXL3 group, mode in [i-ranges]: trellis rows [i/16 .. ) (contiguous), then suh of the ranges, svh whole, mul1;
//   EXL3 group, replicate: trellis, suh, svh, mul1 whole;
//   plain tensor, rows [ranges]: leading-dim rows of the ranges in order; replicate: whole.
// Ranges with several entries (KDA qkv_proj q|k|v head ranges, conv1d rows) keep the order of the split JSON. This is
// the layout MiniMax's Lib::stage_out / stage_in build (gate D validated it on GLM rank slices) and test_split.py slices.
#include "strix/weight/admission.hpp"
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace strix::glm::load {

struct TensorInfo {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint32_t file = 0;
    std::uint64_t offset = 0;  // absolute byte offset in the file
    std::uint64_t bytes = 0;
};

// The safetensors headers of a model directory (model.safetensors.index.json, or every *.safetensors when absent).
class Index {
public:
    explicit Index(const std::string& dir);
    // A table without files (offsets as given), e.g. from saved headers: plan sizes only.
    explicit Index(std::map<std::string, TensorInfo> tensors) : tensors_(std::move(tensors)) {}
    [[nodiscard]] const std::map<std::string, TensorInfo>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const std::vector<std::string>& files() const noexcept { return files_; }
    [[nodiscard]] const TensorInfo* find(const std::string& name) const;
    const weight::WeightMap& weight_map() const noexcept { return weights_; }

private:
    weight::WeightMap weights_;
    std::vector<std::string> files_;
    std::map<std::string, TensorInfo> tensors_;
};

enum class Mode { replicate, out, in, rows };
struct SplitEntry {
    bool exl3 = false;
    Mode mode = Mode::replicate;
    std::vector<std::pair<std::int64_t, std::int64_t>> ranges[2];
};
// The split contract (JSON written by tools/glm/split_plan.py).
class Split {
public:
    explicit Split(const std::string& json_path);
    [[nodiscard]] const std::map<std::string, SplitEntry>& entries() const noexcept { return entries_; }
    [[nodiscard]] double rank_bytes(unsigned rank) const { return rank_bytes_[rank]; }

private:
    std::map<std::string, SplitEntry> entries_;
    double rank_bytes_[2] = {0, 0};
};

struct Piece {
    std::uint32_t file = 0;
    std::uint64_t src = 0, len = 0;  // file byte range
};
struct Slice {
    std::string name;  // group (EXL3) or tensor (plain) name
    bool exl3 = false;
    std::vector<Piece> pieces;  // concatenated = the rank's slice bytes (layout above)
    std::uint64_t bytes = 0;
    // EXL3: component sizes inside the slice and the slice's logical shape (rows = outputs N, cols = inputs K)
    std::uint64_t tr_bytes = 0, suh_bytes = 0, svh_bytes = 0, mul1_bytes = 0, rows = 0, cols = 0, bits = 0;
};

// The slices of `rank` for every split entry present in the index (entries missing from the index are skipped and
// counted in `missing`), in split-JSON (name) order. Throws on any inconsistency (shape, alignment, size).
std::vector<Slice> rank_plan(const Index& index, const Split& split, unsigned rank, std::size_t* missing = nullptr);

// Pieces merged where contiguous in the same file (fewer, larger reads), order kept.
std::vector<Piece> coalesce(const std::vector<Piece>& pieces);

}  // namespace strix::glm::load
