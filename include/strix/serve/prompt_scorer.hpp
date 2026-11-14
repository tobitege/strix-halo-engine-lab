#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// Prompt scorer for LOSSY prefill compression (opt-in per request, default OFF): the math that turns the
// residual stream of a small drafter into per-token relevance, unit-start probabilities and a request mask.
// Pure host, no device types. This is the CPU reference: scalar loops in a fixed order with double
// accumulators, so the same inputs give the same outputs on one build; it is not a fast path.
//
// Layout conventions: every matrix is row-major [out][in]; a residual is [tokens][hidden].
namespace strix::serve::pflash {

// Scorer output, one entry per scorer token; the four vectors have equal length. The selection layer
// consumes them as spans.
struct ScoreBuffers {
    std::vector<float> token_score;               // relevance to the request, >= 0
    std::vector<float> cut_prob;                  // probability that a unit starts at this token
    std::vector<std::uint8_t> request_mask;       // non-zero: the token carries the request
    std::vector<std::uint32_t> token_byte_begin;  // first byte of the token, non-decreasing
};

class PromptScorer {
public:
    virtual ~PromptScorer() = default;
    // `session` names the conversation whose scorer state may be reused; 0 means none.
    virtual ScoreBuffers score(std::string_view rendered, std::uint64_t session) = 0;
};

// ---- QK mass -------------------------------------------------------------------------------------
// One attention block's query/key projections used without a position term. For query rows T and keys j:
//   q_in = rms_norm(x) * attn_norm;  Q_h = rms_norm(Wq_h q_in) * q_norm;  K_g = rms_norm(Wk_g q_in) * k_norm
//   p[h,t,:] = softmax over allowed keys of (Q_h[t] . K_{h / (heads / kv_heads)}[j]) / sqrt(head_dim)
//   mass[j] = sum over h and t of p[h,t,j] / (heads * |T|)
struct QkHead {
    std::uint32_t hidden = 0;
    std::uint32_t heads = 0;
    std::uint32_t kv_heads = 0;           // divides heads
    std::uint32_t head_dim = 0;
    // Rows of wq per head: head_dim, or 2 * head_dim when the projection carries a second (gate) half
    // per head; only the first head_dim rows of each head are used.
    std::uint32_t q_rows_per_head = 0;
    float rms_eps = 1e-6f;
    std::vector<float> attn_norm;         // [hidden]
    std::vector<float> wq;                // [heads * q_rows_per_head][hidden]
    std::vector<float> wk;                // [kv_heads * head_dim][hidden]
    std::vector<float> q_norm;            // [head_dim]
    std::vector<float> k_norm;            // [head_dim]
};

// Throws std::invalid_argument when the shapes are inconsistent.
void validate(const QkHead& head);

// Normalised keys [tokens][kv_heads][head_dim]; cacheable per conversation.
[[nodiscard]] std::vector<float> qk_keys(const QkHead& head, std::span<const float> residual);

// mass[tokens]. `key_allowed` has one entry per token (0: the key is masked for every row). A row with no
// allowed key contributes nothing. An empty `query_rows` gives all zeros.
[[nodiscard]] std::vector<float> qk_mass(const QkHead& head, std::span<const float> residual,
                                         std::span<const float> keys,
                                         std::span<const std::uint32_t> query_rows,
                                         std::span<const std::uint8_t> key_allowed);

// Keys a query window [query_begin, query_end) may score: everything before it, never itself, and the
// tokens after it only when `suffix_candidates` is set.
[[nodiscard]] std::vector<std::uint8_t> window_key_mask(std::uint32_t tokens, std::uint32_t query_begin,
                                                        std::uint32_t query_end, bool suffix_candidates);
// Keys a scattered request set may score: every token that is not itself a query row.
[[nodiscard]] std::vector<std::uint8_t> request_key_mask(std::uint32_t tokens,
                                                         std::span<const std::uint32_t> query_rows);

struct WeightedMass {
    std::span<const float> mass;
    float weight = 1.0f;
};
// sum(weight * mass) / sum(weight); all masses have equal length and the weights are positive.
[[nodiscard]] std::vector<float> mix_masses(std::span<const WeightedMass> parts);
// Weight of the i-th history query window, newest first: 0.5^(i + 1). The current query weighs 1.
[[nodiscard]] float history_weight(std::uint32_t index);

// Score of the unit [begin, end): the sum of its token masses, divided by its length when `density`.
[[nodiscard]] float pool_span(std::span<const float> mass, std::uint32_t begin, std::uint32_t end, bool density);
// Centred box mean of odd width `kernel` (>= 1); the window is clipped at both ends and divided by the
// number of tokens it holds. kernel 1 returns the input.
[[nodiscard]] std::vector<float> box_smooth(std::span<const float> scores, std::uint32_t kernel);
// Flatness measure: exp(entropy of the normalised positive scores) / max(selected, 1). Large values mean
// the scorer found nothing that stands out. Returns 0 when no score is positive.
[[nodiscard]] float score_spread(std::span<const float> scores, std::uint32_t selected);

// ---- Probes --------------------------------------------------------------------------------------
// Both probes read the same residual: layer_norm(x) * norm_weight + norm_bias, fc1, GELU, fc2 -> one logit.
struct ProbeMlp {
    std::uint32_t hidden = 0;
    std::uint32_t width = 0;
    float norm_eps = 1e-5f;
    std::vector<float> norm_weight;       // [hidden]
    std::vector<float> norm_bias;         // [hidden]
    std::vector<float> fc1;               // [width][hidden]
    std::vector<float> fc1_bias;          // [width]
    std::vector<float> fc2;               // [width]
    float fc2_bias = 0.0f;
};
enum class Gelu : std::uint8_t {
    tanh,  // 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
    erf,   // 0.5 x (1 + erf(x / sqrt(2)))
};
void validate(const ProbeMlp& mlp);
// One raw logit per token.
[[nodiscard]] std::vector<float> probe_logits(const ProbeMlp& mlp, std::span<const float> residual, Gelu gelu);

struct SegmentProbe {
    ProbeMlp mlp;
    std::vector<float> conv;              // odd length
    float conv_bias = 0.0f;
    float threshold = 0.5f;               // a unit starts where cut_prob is above this
    std::uint32_t min_segment = 1;
    std::uint32_t max_segment = 1;
    std::string schema;
    std::string oversize_split;           // "", "line_aligned" or "best_score"
};
// cut_prob[t] = sigmoid(raw[t] + conv_bias + sum_k conv[k] * raw[t + k - radius]), zero outside the prompt;
// raw comes from the tanh GELU. Turning the probabilities into units belongs to the selection layer.
[[nodiscard]] std::vector<float> segment_cut_prob(const SegmentProbe& probe, std::span<const float> residual);

struct RequestProbe {
    ProbeMlp mlp;
    float tau = 0.0f;                     // a token is request text when its raw logit is >= tau
    std::string schema;
    std::string candidate_rule;           // optional, "" when absent
};
// Raw logits (erf GELU, no output activation).
[[nodiscard]] std::vector<float> request_logits(const RequestProbe& probe, std::span<const float> residual);
// Mask of the maximal runs of raw >= tau that stay inside one message and are at least `min_run` long.
// `message_of_token` gives the message index of each token; NaN ends a run.
[[nodiscard]] std::vector<std::uint8_t> request_runs(std::span<const float> raw, float tau,
                                                     std::span<const std::uint32_t> message_of_token,
                                                     std::uint32_t min_run = 2);
// Indices of the set tokens of `mask`; when there are more than `cap`, the last `cap` are kept.
[[nodiscard]] std::vector<std::uint32_t> request_rows(std::span<const std::uint8_t> mask, std::uint32_t cap);

// ---- Weight files (GGUF) -------------------------------------------------------------------------
// Loaders fail closed: any missing or mistyped key or tensor, an unknown schema, or a `runtime_sha256`
// that differs from the one recorded in the file throws std::runtime_error. `runtime_sha256` is the hash
// of the drafter weights the file was trained against; the caller computes it from the drafter it loaded.
inline constexpr std::string_view kFeatureTap = "post_block14_residual_before_block15";
inline constexpr std::string_view kScoringHeadSchema = "qwen3_5_0_8b_nope_qk_mass_v1";
inline constexpr std::string_view kSegmentProbeSchemaV1 = "qwen3_5_0_8b_segment_probe_v1";
inline constexpr std::string_view kSegmentProbeSchemaV2 = "qwen3_5_0_8b_segment_probe_v2";
inline constexpr std::string_view kRequestProbeSchema = "qwen3_5_0_8b_request_probe_v1";
inline constexpr std::string_view kRequestExtraction = "request_runs_v2_minrun2_ge_raw_logit";

// Trained replacement for the two projections of a QkHead; the norms come from the drafter.
struct ScoringHeadFile {
    std::string schema;
    std::string base_model;
    std::uint32_t hidden = 0;
    std::uint32_t q_rows = 0;             // heads * head_dim
    std::uint32_t k_rows = 0;             // kv_heads * head_dim
    std::vector<float> wq;                // [q_rows][hidden]
    std::vector<float> wk;                // [k_rows][hidden]
};
[[nodiscard]] ScoringHeadFile load_scoring_head(const std::filesystem::path& path, std::string_view runtime_sha256);
[[nodiscard]] SegmentProbe load_segment_probe(const std::filesystem::path& path, std::string_view runtime_sha256);
[[nodiscard]] RequestProbe load_request_probe(const std::filesystem::path& path, std::string_view runtime_sha256);

}  // namespace strix::serve::pflash
