#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

#include "strix/serve/prompt_scorer.hpp"

// Drafter of the LOSSY prefill compression (opt-in per request, default OFF): a small hybrid language model
// that reads the whole prompt once and never generates. Blocks 0-14 run over the prompt; the query/key
// projections of block 15, used without a position term, turn the block-14 residual into a per-token
// relevance ("QK mass", strix::serve::pflash::qk_mass).
//
// This header is the CPU reference: scalar loops in a fixed order, float storage, double accumulators. The
// same inputs give the same outputs on one build. It is the oracle for the device implementation and is
// far too slow to serve requests.
//
// Scoring needs only the drafter's own weights. Trained add-ons (replacement scoring projections, a unit
// boundary probe, a request probe) are optional inputs and absent by default.
//
// Layout conventions: every matrix is row-major [out][in]; a residual is [tokens][hidden].
namespace strix::pflash {

struct DrafterConfig {
    std::uint32_t hidden = 0;
    std::uint32_t feed_forward = 0;
    std::uint32_t vocabulary = 0;
    std::uint32_t blocks = 15;                   // blocks that run; the scoring block is the next one
    std::uint32_t full_attention_interval = 4;   // block b is full attention when (b + 1) % interval == 0
    // Full attention blocks.
    std::uint32_t heads = 0;
    std::uint32_t kv_heads = 0;                  // divides heads; query head h reads KV head h / (heads / kv_heads)
    std::uint32_t head_dim = 0;
    std::uint32_t rope_dims = 0;                 // even, <= head_dim: leading dims of each head that rotate
    float rope_base = 0.0f;
    // Delta-rule blocks.
    std::uint32_t delta_heads = 0;
    std::uint32_t delta_head_dim = 0;            // key and value width of one head
    std::uint32_t conv_kernel = 0;
    float rms_eps = 1e-6f;
};

// Gated delta rule over a width W = delta_heads * delta_head_dim.
struct DeltaBlock {
    std::vector<float> qkv;                      // [3 W][hidden]: queries, keys, values
    std::vector<float> gate;                     // [W][hidden]: output gate
    std::vector<float> beta;                     // [delta_heads][hidden]: write strength
    std::vector<float> alpha;                    // [delta_heads][hidden]: decay input
    std::vector<float> conv;                     // [3 W][conv_kernel], the last tap is on the current token
    std::vector<float> dt_bias;                  // [delta_heads]
    std::vector<float> a;                        // [delta_heads], negative: decay = exp(a * softplus(alpha + dt_bias))
    std::vector<float> norm;                     // [delta_head_dim]
    std::vector<float> out;                      // [hidden][W]
};

// Causal attention with a rotary position term and a sigmoid output gate.
struct AttentionBlock {
    std::vector<float> wq;                       // [heads * 2 * head_dim][hidden]: per head, query rows then gate rows
    std::vector<float> wk;                       // [kv_heads * head_dim][hidden]
    std::vector<float> wv;                       // [kv_heads * head_dim][hidden]
    std::vector<float> wo;                       // [hidden][heads * head_dim]
    std::vector<float> q_norm;                   // [head_dim]
    std::vector<float> k_norm;                   // [head_dim]
};

struct DrafterBlock {
    bool full_attention = false;
    std::vector<float> attn_norm;                // [hidden]
    DeltaBlock delta;                            // filled when !full_attention
    AttentionBlock attention;                    // filled when full_attention
    std::vector<float> ffn_norm;                 // [hidden]
    std::vector<float> ffn_gate;                 // [feed_forward][hidden]
    std::vector<float> ffn_up;                   // [feed_forward][hidden]
    std::vector<float> ffn_down;                 // [hidden][feed_forward]
};

struct DrafterWeights {
    DrafterConfig config;
    std::vector<DrafterBlock> blocks;            // config.blocks entries
    serve::pflash::QkHead scorer;                // the drafter's own scoring block
};

// Throws std::invalid_argument when a shape disagrees with the config.
void validate(const DrafterWeights& weights);

// Reads the running blocks and the scoring projections from a drafter GGUF (F32, F16 or BF16 tensors). The
// later blocks, the output head and the token table are not read. Fails closed: a missing or mistyped key or
// tensor, or a shape that disagrees with the metadata, throws std::runtime_error.
[[nodiscard]] DrafterWeights load_drafter(const std::filesystem::path& gguf);

// Rows of the token table for `tokens`, [tokens][hidden]; only those rows are read from the file.
[[nodiscard]] std::vector<float> load_embeddings(const std::filesystem::path& gguf,
                                                 std::span<const std::uint32_t> tokens);

// Residual after the last running block, [tokens][hidden], from the embedded prompt [tokens][hidden].
// Positions start at 0.
[[nodiscard]] std::vector<float> forward(const DrafterWeights& weights, std::span<const float> embedded);

// A run of scorer tokens that asks something: the request of the current turn or of an earlier one.
struct QueryWindow {
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
};

struct ScoreOptions {
    std::uint32_t request_cap = 1024;            // at most this many query rows per window, the last ones
    bool suffix_candidates = false;              // tokens after a window may be scored by it
    // Optional trained add-ons; null means absent, which is the default.
    const serve::pflash::ScoringHeadFile* scoring_head = nullptr;   // replaces the two scoring projections
    const serve::pflash::SegmentProbe* segment_probe = nullptr;     // fills cut_prob
    const serve::pflash::RequestProbe* request_probe = nullptr;     // picks the request rows of the current turn
    std::span<const std::uint32_t> message_of_token;                // needed by request_probe
};

// Scores a prompt from its block-14 residual.
//   token_score: QK mass of the current window, mixed with the history windows (newest first) at weights
//     0.5^(i + 1). Without a request probe the query rows are the window's tokens and the keys are the
//     tokens before it; with one, the rows are the probe's request runs inside the window and every other
//     token is a key.
//   request_mask: the query rows of the current window.
//   cut_prob: the segment probe's output, or all zeros without one (the selection layer then cuts units
//     by length and line starts only).
// Throws std::invalid_argument on windows outside the prompt or sizes that disagree.
[[nodiscard]] serve::pflash::ScoreBuffers score_residual(const DrafterWeights& weights,
                                                         std::span<const float> residual,
                                                         QueryWindow current,
                                                         std::span<const QueryWindow> history,
                                                         std::span<const std::uint32_t> token_byte_begin,
                                                         const ScoreOptions& options = {});

// What the scorer needs to know about a rendered prompt; produced by the caller's tokenizer and chat
// adapter.
struct TokenizedPrompt {
    std::vector<std::uint32_t> tokens;
    std::vector<std::uint32_t> token_byte_begin;
    QueryWindow current;
    std::vector<QueryWindow> history;            // newest first
};

// PromptScorer over the CPU reference. `embed` maps tokens to [tokens][hidden] (load_embeddings, or a table
// held by the caller). The session argument is ignored: nothing is cached between calls.
class CpuDrafterScorer final : public serve::pflash::PromptScorer {
public:
    using Tokenize = std::function<TokenizedPrompt(std::string_view)>;
    using Embed = std::function<std::vector<float>(std::span<const std::uint32_t>)>;

    CpuDrafterScorer(DrafterWeights weights, Tokenize tokenize, Embed embed, ScoreOptions options = {});

    serve::pflash::ScoreBuffers score(std::string_view rendered, std::uint64_t session) override;

private:
    DrafterWeights weights_;
    Tokenize tokenize_;
    Embed embed_;
    ScoreOptions options_;
};

}  // namespace strix::pflash
