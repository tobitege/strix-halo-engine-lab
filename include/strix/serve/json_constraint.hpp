#pragma once

#include "strix/serve/json.hpp"
#include "strix/serve/bpe_tokenizer.hpp"
#include <atomic>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::serve {
struct ConstraintLimits {
    std::size_t schema_bytes = 65536;
    std::size_t nodes = 4096;
    std::size_t depth = 64;
    std::size_t properties = 128;
    std::size_t enum_values = 1024;
    std::size_t max_string_length = 1048576;
    std::size_t max_output_bytes = 1048576;
    std::size_t compilation_steps = 65536; // Also bounds each cursor transition.
    // Aggregate limits for one compilation, including all references/alternatives.
    std::size_t literal_bytes = 1048576;
    std::size_t trie_nodes = 65536;
    std::size_t alternative_expansions = 65536;
    std::size_t mask_cache_bytes = 0; // Optional per-program LRU; zero disables it.
};
class SchemaError : public std::runtime_error {
public:
    SchemaError(std::string path, std::string keyword, std::string message);
    std::string path;
    std::string keyword;
};
class ConstraintError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
struct ConstraintData;
struct ConstraintState;
// Optional compilation diagnostics, including failed compilations. No global state.
struct ConstraintCompilationStats {
    std::size_t expansion_recursion_high_water = 0;
    std::size_t expansion_stack_high_water = 0;
    std::size_t compilation_steps = 0;
};
struct ConstraintMemoStats { std::uint64_t hits = 0, misses = 0; };
class ConstraintProgram {
public:
    ConstraintMemoStats memo_stats() const;
private:
    friend ConstraintProgram compile_json_schema(const Json&, const ConstraintLimits&, ConstraintCompilationStats*);
    explicit ConstraintProgram(std::shared_ptr<const ConstraintData> data) : data_(std::move(data)) {}
    std::shared_ptr<const ConstraintData> data_;
    friend class ConstraintCursor;
    friend struct ConstraintAccess;
};
// strix-json-schema-v1: JSON with 0..2 spaces, tabs or newlines per gap
// (including leading/trailing gaps); no CR. Members may occur in any order,
// without duplicates. Strings use strict UTF-8 and paired UTF-16 escapes;
// length counts decoded Unicode scalars. Integers have no fraction/exponent.
// enum/const use only Json::dump(): compact, original member order, UTF-8.
// The cursor begins at the answer boundary: reasoning/framing is a caller concern.
// Copies share immutable pushdown state; admission consumes only the new byte.
// advance(false) leaves the cursor unchanged. complete() marks an accepting
// boundary (a number may still extend); EOS may be selected at that boundary.
// Output bytes, depth and string lengths are bounded. Admission reserves a
// completing suffix within those limits. Byte/shape exhaustion refuses admission;
// evaluation exhaustion throws ConstraintError, never relaxes the language. An optional bounded mask LRU uses exact serialized state keys;
// states with additional names or oversized keys bypass it. It defaults off.
ConstraintProgram compile_json_schema(const Json&, const ConstraintLimits& = {}, ConstraintCompilationStats* = nullptr);
ConstraintProgram compile_json_object();

// Programs shared between requests. A compiled program is immutable and does not depend on the request, so
// requests with the same schema and limits share one: the schema is compiled once, and the masks of the
// program's bounded mask memo (exact serialized cursor state plus the identity of the token trie, see
// walk rules above) are reused by every request that reaches the same state with the same tokenizer.
// Per-request state (cursor, output budget, EOS overlay) stays with the request.
//   key    : semantics_version, every field of the limits, and Json::dump() of the schema, compared in full
//            (a hash only selects the bucket). Two schemas that differ in member order are two entries.
//   bound  : at most max_programs entries and key_bytes of retained schema text, least recently used out
//            first; each cached program has a mask memo of mask_cache_bytes unless the caller's limits
//            name their own. A schema whose text exceeds a quarter of key_bytes is compiled and not kept.
//            Pending entries count toward both bounds; if they fill the cache, new keys are not kept.
//   errors : a failed compilation throws what compile_json_schema() throws and is not kept.
// Concurrent requests for a schema that is being compiled wait for that one compilation.
struct ConstraintProgramCacheConfig {
    std::size_t max_programs = 16;            // 0: no sharing, every call compiles (previous behaviour)
    std::size_t key_bytes = 4U << 20U;
    std::size_t mask_cache_bytes = 8U << 20U; // per cached program; 0 leaves the mask memo off
};
struct ConstraintProgramCacheStats {
    std::uint64_t hits = 0;       // served from a finished entry
    std::uint64_t joined = 0;     // waited for a compilation another request had started
    std::uint64_t misses = 0;     // compiled here
    std::uint64_t failures = 0;   // compilations that threw
    std::uint64_t evictions = 0;
    std::uint64_t uncached = 0;   // compiled without an entry (disabled, oversized, or pending capacity)
    std::uint64_t compile_ns = 0; // time in compile_json_schema() over all misses
    std::size_t programs = 0, key_bytes = 0;
};
class ConstraintProgramCache {
public:
    // Part of every key: raise it with any change to what a schema compiles to.
    static constexpr std::uint32_t semantics_version = 1;
    explicit ConstraintProgramCache(ConstraintProgramCacheConfig = {});
    ~ConstraintProgramCache();
    ConstraintProgramCache(const ConstraintProgramCache&) = delete;
    ConstraintProgramCache& operator=(const ConstraintProgramCache&) = delete;
    std::shared_ptr<const ConstraintProgram> json_schema(const Json& schema, const ConstraintLimits& limits = {});
    std::shared_ptr<const ConstraintProgram> json_object();
    ConstraintProgramCacheStats stats() const;
    const ConstraintProgramCacheConfig& config() const;
    void clear(); // drops finished entries; requests keep the programs they hold
private:
    friend struct ConstraintProgramCacheTestAccess;
    // A per-instance compilation barrier for deterministic host tests; never set by serving.
    ConstraintProgramCache(ConstraintProgramCacheConfig, void (*before_compile)(void*), void* context);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// The process-wide cache used by the serving layer of every engine.
//   STRIX_CONSTRAINT_PROGRAM_CACHE=0        compile per request (previous behaviour)
//   STRIX_CONSTRAINT_MASK_MEMO_BYTES=8M     mask memo of each cached program; 0 turns the memo off
ConstraintProgramCache& shared_constraint_program_cache();

class ConstraintCursor {
public:
    explicit ConstraintCursor(const ConstraintProgram&);
    bool advance(unsigned char byte);
    [[nodiscard]] bool complete() const;
    // A refused byte never poisons the cursor; admitted prefixes remain productive.
    [[nodiscard]] bool dead() const;
private:
    std::shared_ptr<const ConstraintData> data_;
    std::shared_ptr<const ConstraintState> state_;
    friend struct ConstraintAccess;
};
// Atomic helper: throws ConstraintError on refusal; original cursor is unchanged.
ConstraintCursor advance_token(ConstraintCursor cursor, std::string_view bytes);
class TokenByteTrie {
public:
    // Only normal/byte tokens are eligible; excluded IDs further restrict them.
    explicit TokenByteTrie(const BpeTokenizer&, std::span<const std::uint32_t> excluded = {});
    // Also useful for tokenizer-independent host fixtures. IDs are vector positions.
    explicit TokenByteTrie(std::span<const std::string> bytes, std::span<const std::uint32_t> excluded = {});
    [[nodiscard]] std::size_t vocab_size() const { return vocab_size_; }
    // Whether the ID is an eligible (inserted) token: exactly the IDs a mask walk can admit.
    [[nodiscard]] bool contains(std::uint32_t id) const { return id < members_.size() && members_[id]; }
    struct Node { std::vector<std::pair<unsigned char, std::size_t>> edges; std::vector<std::uint32_t> ids; };
    [[nodiscard]] const std::vector<Node>& nodes() const { return nodes_; }
private:
    void insert(std::string_view, std::uint32_t);
    std::size_t vocab_size_ = 0;
    std::vector<Node> nodes_{1};
    std::vector<bool> members_;
    std::shared_ptr<const unsigned char> identity_ = std::make_shared<const unsigned char>(static_cast<unsigned char>(0));
    friend struct ConstraintAccess;
};
struct AllowedTokens {
    std::size_t vocab_size = 0;
    std::vector<std::uint64_t> words;
    [[nodiscard]] bool contains(std::size_t id) const;
    void set(std::size_t id, bool value = true);
};
AllowedTokens allowed(const ConstraintProgram&, const ConstraintCursor&, const TokenByteTrie&,
                      std::span<const std::uint32_t> eos_ids = {});
// The same walk, abandoned (nullopt, nothing cached) once `stop` is set: for a walk whose result may no
// longer be wanted. A completed walk returns exactly allowed().
std::optional<AllowedTokens> allowed_until(const ConstraintProgram&, const ConstraintCursor&, const TokenByteTrie&,
                                           std::span<const std::uint32_t> eos_ids, const std::atomic<bool>& stop);
std::uint32_t constrained_greedy(std::span<const float> logits, const AllowedTokens&);
}  // namespace strix::serve

