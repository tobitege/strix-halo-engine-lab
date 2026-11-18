#pragma once
#include "strix/weight/admission.hpp"

#include "strix/ds41_formats.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

// Native FP8 dense weights of the original deepseek41 checkpoint and their exact product with A8 activations.
//
// The checkpoint (safetensors) stores the dense projections (attention wq_a, wq_b, wkv, wo_a, wo_b; shared expert
// w1, w3, w2) as F8_E4M3 codes [rows][width] with one F8_E8M0 exponent per 32 x 32 block, scale tensor
// [rows / 32][width / 32]: weight (r, c) = e4m3(code) * 2^(scale[r / 32][c / 32] - 127). The model file holds Q8_0
// requantizations of these tensors; this form reads the original codes instead (exact weights, 8.0078 bits per
// weight against Q8_0's 8.5). The output head is BF16 in the checkpoint and has no native FP8 form.
//
// Integer form. Every E4M3 value is an integer multiple of 2^-9: N(c) = e4m3(c) * 2^9 is (8 + m) << (e - 1) for
// e >= 1 and m for e = 0 (sign applied), |N| <= 229376 = 448 * 2^9. The dot of 32 codes with 32 A8 codes is then an
// exact int32 (|D| <= 32 * 229376 * 127 < 2^30), so its order is free, as for the sudot4 products of Q8_0.
//
// Declared product (identity fp8_product_identity), one weight row with one A8 activation row over `width` columns:
// group g (columns 32g .. 32g + 31, block column of the row's 32 x 32 block) contributes
//   float(D_g) * (s_g * delta_g),  D_g = sum_j N(code_j) * x_j (exact),  s_g = 2^(e_g - 136) (FP32, exact)
// lane l accumulates groups l, l + 32, ... in ascending order from 0 (acc = acc + term, no fused operations) and the
// lanes are combined by the butterfly v[l] += v[l ^ d] for d = 16, 8, 4, 2, 1; the result is lane 0. This is the
// lane order of the Q8_0 x A8 sites with the Q8_0 block term replaced.
namespace strix::ds41 {

inline constexpr std::size_t fp8_block = 32;
// Exponents with a normal FP32 dot scale 2^(e - 136); the loader rejects tensors outside (the 1017 checkpoint: 114-119).
inline constexpr std::uint8_t fp8_min_exponent = 10, fp8_max_exponent = 254;
// Numerical identities of the native FP8 steps (the Q8_0 forms are q8_0-a8-product-v1 etc. in deepseek41_plan.cpp).
inline constexpr const char* fp8_product_identity = "e4m3-b32x32-a8-int-product-v1";
inline constexpr const char* fp8_grouped_identity = "a8+e4m3-b32x32-a8-int-product-grouped8-v1";
inline constexpr const char* fp8_ksplit_identity = "e4m3-b32x32-a8-int-ksplit2-h0+h1-v1";
// Shared expert: gate and up as fp8 products (FP32 outputs), act = swiglu_clamped(gate, up, 10) (ds41_moe.hpp), A8 of act
// (quantize_a8), down as an fp8 product.
inline constexpr const char* fp8_swiglu_identity = "e4m3-b32x32-a8-int-swiglu-clamp10-v1";

// e4m3(c) * 2^9 as a signed integer. The NaN codes 0x7f / 0xff give 245760 / -245760 (rejected by fp8_tensor_check).
[[nodiscard]] constexpr std::int32_t e4m3_integer(std::uint8_t c) noexcept {
    const std::uint32_t e = (c >> 3U) & 15U, m = c & 7U;
    const auto n = static_cast<std::int32_t>(e != 0U ? (m | 8U) << (e - 1U) : m);
    return (c & 0x80U) != 0U ? -n : n;
}
// 2^(e - 136) = 2^(e - 127) * 2^-9 for e in [fp8_min_exponent, fp8_max_exponent].
[[nodiscard]] float fp8_dot_scale(std::uint8_t e) noexcept;

// Device layout of one tensor: the codes [rows][width] (the checkpoint's bytes), then the exponents [rows / 32][width / 32].
[[nodiscard]] constexpr std::size_t fp8_tensor_bytes(std::size_t rows, std::size_t width) noexcept {
    return rows * width + rows / fp8_block * (width / fp8_block);
}
// Empty if the tensor is usable, else the reason (geometry, a NaN code, an exponent outside the range).
[[nodiscard]] std::string fp8_tensor_check(std::span<const std::uint8_t> codes, std::span<const std::uint8_t> scales, std::size_t rows,
                                           std::size_t width);

// The declared product: `codes` and `scales` point at the first column of the row and at its block exponent (so a
// K-split part passes codes + k_offset and scales + k_offset / 32); x and deltas hold `width` A8 codes and width / 32 deltas.
[[nodiscard]] float product_fp8(const std::uint8_t* codes, const std::uint8_t* scales, const std::int8_t* x, const float* deltas,
                                std::size_t width);
// Weight (r, c) of a tensor in the device layout as FP32 (exact; the reference of tests and tools).
[[nodiscard]] float fp8_weight(const std::uint8_t* tensor, std::size_t rows, std::size_t width, std::size_t r, std::size_t c);

// Site classes of the model's Q8_0 tensors that have a native FP8 original, and the name mapping
// (blk.L.attn_q_a.weight -> layers.L.attn.wq_a, codes "<name>.weight", exponents "<name>.scale").
// Classes: q_a, q_b, kv, o_a, o_b, shexp (gate, up and down of the shared expert together).
[[nodiscard]] std::optional<std::string> fp8_site_class(const std::string& model_name);
[[nodiscard]] std::optional<std::string> fp8_original_name(const std::string& model_name);
// Comma-separated classes ("all" = every class) to the set; throws on an unknown class.
[[nodiscard]] std::vector<std::string> parse_fp8_site_classes(const std::string& list);

// Shared loader resolution. Index filenames may be hidden, absolute, outside dir or symlinks.
struct SafetensorsResolution {
    std::map<std::string, std::string> shard_of;
    std::map<std::string, std::filesystem::path> files;
    std::vector<std::filesystem::path> unindexed_shards;  // preserve loader scan order
    bool indexed = false;
};
[[nodiscard]] SafetensorsResolution resolve_safetensors_checkpoint(const std::filesystem::path& dir);

// Read-only view of a safetensors checkpoint directory: model.safetensors.index.json (or, without it, every
// *.safetensors file) and the shard headers it names. Reads tensor bytes with plain positioned reads and drops them
// from the page cache afterwards (Linux), so loading does not leave a cached copy next to the resident weights.
class SafetensorsCheckpoint {
public:
    explicit SafetensorsCheckpoint(const std::filesystem::path& dir, bool reject_duplicate_names = false);
    struct Tensor {
        std::filesystem::path file;
        std::string dtype;
        std::vector<std::uint64_t> shape;
        std::uint64_t offset = 0, bytes = 0;  // absolute in the file
        weight::Slice source;
    };
    // nullptr if the index has no such tensor or its shard is missing.
    [[nodiscard]] const Tensor* find(const std::string& name);
    void read(const Tensor& t, std::uint8_t* out) const;
    weight::Status admitted_weights(weight::WeightMap& out) const { return weight::bind_ds41(index_,out); }
    const weight::TensorIndex& tensor_index() const noexcept { return index_; }
    [[nodiscard]] const std::filesystem::path& dir() const noexcept { return dir_; }

private:
    void load_header(const std::filesystem::path& file);
    weight::TensorIndex index_;
    bool reject_duplicate_names_ = false;
    std::filesystem::path dir_;
    std::map<std::string, std::string> shard_of_;  // tensor name -> shard file name
    std::map<std::string, bool> loaded_;           // shard file name -> header parsed
    std::map<std::string, Tensor> tensors_;
};

// One native FP8 tensor in the device layout (codes then exponents), read and checked; throws with the reason.
struct Fp8Tensor {
    std::size_t rows = 0, width = 0;
    std::vector<std::uint8_t> bytes;
};
[[nodiscard]] Fp8Tensor read_fp8_tensor(SafetensorsCheckpoint& checkpoint, const std::string& original, std::size_t rows,
                                        std::size_t width);

}  // namespace strix::ds41
