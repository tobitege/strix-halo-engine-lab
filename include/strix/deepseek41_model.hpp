#pragma once

#include "strix/gguf_reader.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

// DeepSeek V4.1 Flash (GGUF architecture "deepseek41") and its speculative draft ("deepseek41-dspark"):
// configuration, Engram hash contract, per-layer attention attributes, the exact tensor inventory derived
// from the configuration, the manifest and file admission by SHA-256. Host only; nothing here executes.
namespace strix::deepseek41 {

// Retain the numeric metadata arrays this model needs (the compressed-token map has one entry per token).
[[nodiscard]] gguf::Limits reader_limits() noexcept;
// The vision file's metadata keys carry its hyphenated architecture name (`deepseek4-vision.*`).
[[nodiscard]] gguf::Limits vision_reader_limits() noexcept;
[[nodiscard]] gguf::FileInfo read_file(const std::filesystem::path& path);

struct RopeScaling {
    std::string type;
    double factor = 0.0, beta_fast = 0.0, beta_slow = 0.0, original_max_position_embeddings = 0.0;
};

// `text_config` of the `deepseek41.config` JSON (the full upstream configuration embedded in the file).
struct Config {
    std::uint32_t vocab_size = 0, hidden_size = 0, moe_intermediate_size = 0, num_hidden_layers = 0;
    std::uint32_t num_attention_heads = 0, num_key_value_heads = 0, head_dim = 0, qk_rope_head_dim = 0;
    std::uint32_t q_lora_rank = 0, o_lora_rank = 0, o_groups = 0;
    std::uint32_t n_routed_experts = 0, n_shared_experts = 0, num_experts_per_tok = 0;
    std::uint64_t max_position_embeddings = 0;
    std::uint32_t sliding_window = 0, index_n_heads = 0, index_head_dim = 0, index_topk = 0;
    std::uint32_t candidate_source_layer_id = 0, candidate_topk_blocks = 0, candidate_block_size = 0;
    std::uint32_t hc_mult = 0, hc_sinkhorn_iters = 0, num_nextn_predict_layers = 0;
    double rope_theta = 0.0, compress_rope_theta = 0.0, rms_norm_eps = 0.0, hc_eps = 0.0;
    double swiglu_limit = 0.0, routed_scaling_factor = 0.0;
    std::string scoring_func, hidden_act, topk_method;
    bool norm_topk_prob = false;
    RopeScaling rope_scaling;
    // One entry per target layer followed by one per draft stage.
    std::vector<std::uint32_t> compress_ratios;
    std::vector<std::uint32_t> kv_source_layer_ids, index_source_layer_ids;
    std::vector<std::uint32_t> engram_layer_ids;
    std::vector<std::uint64_t> engram_num_embeddings;
    std::uint32_t engram_n_heads = 0, engram_head_dim = 0, engram_max_ngram_size = 0;
    std::uint64_t engram_vocab_size = 0;
    std::uint32_t engram_compressed_vocab_size = 0, engram_pad_token_id = 0;
    std::uint32_t dspark_block_size = 0, dspark_markov_rank = 0, dspark_n_routed_experts = 0;
    std::uint32_t dspark_num_experts_per_tok = 0, dspark_noise_token_id = 0;
    std::vector<std::uint32_t> dspark_target_layer_ids;
    std::uint32_t bos_token_id = 0, eos_token_id = 0, pad_token_id = 0, image_token_id = 0;
};

// Strict parse of the embedded JSON; every field above is required and type-checked.
[[nodiscard]] Config parse_config_json(std::string_view json);

// Engram n-gram hash tables (layers 1 and 14 in this model). Each position hashes the
// (max_ngram_size - 1) n-grams ending there, each over n_heads heads, so one token reads
// rows_per_token() rows per Engram layer; every (n-gram, head) owns a prime-sized bucket range.
struct EngramContract {
    std::string encoding;                    // "e4m3_e8m0_32_row264"
    std::uint32_t row_bytes = 0;             // head_dim FP8 E4M3 bytes + head_dim / 32 E8M0 scales
    std::vector<std::uint32_t> layer_ids;
    std::vector<std::uint64_t> rows;         // table rows per Engram layer
    std::vector<std::uint64_t> primes;       // [layer][n-gram size][head], flattened
    std::vector<std::uint64_t> multipliers;  // [layer][lookback], flattened, odd
    std::vector<std::uint32_t> token_map;    // token id -> compressed id
    std::uint32_t compressed_vocab_size = 0, pad_id = 0, max_ngram_size = 0, n_heads = 0;
    [[nodiscard]] std::uint32_t rows_per_token() const noexcept { return (max_ngram_size - 1U) * n_heads; }
};

// The per-layer attention organisation, all derived from the configuration.
struct LayerAttributes {
    std::uint32_t compress_ratio = 0;         // 0: sliding window only
    bool kv_source = false;                   // compresses its own KV (and owns index keys)
    bool index_source = false;                // runs its own indexer (top-k over compressed positions)
    bool compressor_gate = false;             // softmax-gated pooling (ratio > 1)
    bool candidate_source = false;            // level one of the two-level top-k
    bool uses_candidates = false;             // level two: scores only inside the candidate blocks
    bool engram = false;
    std::int32_t kv_source_layer = -1;        // whose compressed KV this layer attends to
    std::int32_t index_source_layer = -1;     // whose top-k selection this layer reuses
};

struct TargetModel {
    Config config;
    EngramContract engram;
    std::vector<LayerAttributes> layers;
    ModelManifest manifest;
};

struct DraftModel {
    std::uint32_t stages = 0, block_size = 0, markov_rank = 0, noise_token_id = 0;
    std::uint32_t n_routed_experts = 0, num_experts_per_tok = 0;
    std::vector<std::uint32_t> target_layer_ids;
    ModelManifest manifest;
};

struct ExpectedTensor {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint32_t ggml_type = 0;
    TensorRole role = TensorRole::unknown;
    ResidencyPolicy residency = ResidencyPolicy::resident;
    std::int32_t layer = -1;
};

[[nodiscard]] std::vector<LayerAttributes> layer_attributes(const Config& config);
// The exact inventory of the admitted files: names, shapes (GGUF order) and storage types.
[[nodiscard]] std::vector<ExpectedTensor> expected_target_tensors(const Config& config);
[[nodiscard]] std::vector<ExpectedTensor> expected_draft_tensors(const Config& config);

// Full structural validation (metadata cross-checks, Engram contract, inventory: no missing, extra or
// mismatched tensor). Throws std::invalid_argument naming the first violation.
[[nodiscard]] TargetModel validate_target(const gguf::FileInfo& info);
[[nodiscard]] DraftModel validate_draft(const gguf::FileInfo& info, const Config& target);

// The image tower file (`general.architecture = deepseek4-vision`): a pre-norm ViT (RMSNorm, fused qkv with bias,
// gated SwiGLU without bias, 2D rotary positions, final norm), a pixel-unshuffle + two-linear aligner into the
// language width and three learned rows (image start, row end, image end). Everything here is read from the file.
struct VisionConfig {
    std::uint32_t num_hidden_layers = 0, hidden_size = 0, num_attention_heads = 0, intermediate_size = 0;
    std::uint32_t patch_size = 0, downsample_ratio = 0, projection_size = 0;
    std::uint32_t max_image_tokens = 0, min_pixels = 0, max_wh_ratio = 0;  // max_wh_ratio 0: unlimited
    std::uint32_t image_token_id = 0;
    std::uint32_t language_layers = 0, language_experts = 0;  // the language model this tower was released with
    double rope_theta = 0.0, rms_norm_eps = 0.0;
    std::string checkpoint_variant, source_revision;
    [[nodiscard]] std::uint32_t head_dim() const noexcept { return num_attention_heads ? hidden_size / num_attention_heads : 0U; }
    [[nodiscard]] std::uint32_t patch_features() const noexcept { return 3U * patch_size * patch_size; }
    [[nodiscard]] std::uint32_t aligner_inputs() const noexcept { return hidden_size * downsample_ratio * downsample_ratio; }
    [[nodiscard]] std::uint32_t cell() const noexcept { return patch_size * downsample_ratio; }  // pixels per image row side
};

struct VisionModel {
    VisionConfig config;
    ModelManifest manifest;
};

[[nodiscard]] std::vector<ExpectedTensor> expected_vision_tensors(const VisionConfig& config);
// Strict: architecture, every metadata key with its type, value ranges, the pinned source revision and the exact
// BF16 tensor inventory. Throws std::invalid_argument naming the first violation.
[[nodiscard]] VisionModel validate_vision(const gguf::FileInfo& info);
// The tower must belong to the admitted language model: image token id, width, layer and expert counts.
void require_vision_matches_target(const VisionConfig& vision, const Config& target);

enum class PinnedFileKind : std::uint8_t { target_q2, draft_mxfp4, vision_bf16 };

struct PinnedFile {
    PinnedFileKind kind;
    std::string_view file_name;
    std::uint64_t bytes;
    std::string_view sha256_hex;
};

// The 1017 release (general.source.revision 1bae0e1088993b9b82de08f59de5be5a34e42170) as converted to GGUF.
inline constexpr std::string_view pinned_source_revision = "1bae0e1088993b9b82de08f59de5be5a34e42170";
inline constexpr std::array<PinnedFile, 3> pinned_files{{
    {PinnedFileKind::target_q2, "DeepSeek-V4.1-Flash-1017-Q2.gguf", 365'713'686'528ULL,
     "5365629079214ff28becbbab7c0900ef9865b79c0f0c64d951924bb362ae5b67"},
    {PinnedFileKind::draft_mxfp4, "DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf", 7'966'294'016ULL,
     "d3efe0b1f6b6fea54f1c58c5cbeb3fded5967b15b64476aed00c95806ad3e3e9"},
    {PinnedFileKind::vision_bf16, "DeepSeek-V4.1-Flash-1017-vision-BF16.gguf", 970'555'552ULL,
     "861f1097e8686111fa23dc4a6bf139fd0c4cc114db2f1b4090a267f7683c0dbf"},
}};

[[nodiscard]] const PinnedFile& pinned_file(PinnedFileKind kind) noexcept;
// Admission is by the raw file's SHA-256 and byte count only; throws std::invalid_argument on mismatch.
void admit_file(PinnedFileKind kind, std::uint64_t file_bytes, const ModelContentIdentity& file_sha256);

}  // namespace strix::deepseek41
