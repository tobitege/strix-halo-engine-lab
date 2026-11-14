#pragma once

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace strix::serve {

// Index of prompt prefixes whose lane state exists (sealed) or is being prefilled (in flight).
// It stores token ids and entry numbers only; the lane state stays with its existing owner.
// Not thread-safe: the admission thread owns it.

// Identity of the state a prefix belongs to. Any mismatch means no match.
struct PrefixKey {
    std::uint64_t model_id = 0;
    std::uint64_t template_id = 0;
    std::uint64_t format_id = 0;
    friend bool operator==(const PrefixKey&, const PrefixKey&) = default;
};

// entry == 0 means no match. tokens is the exact frontier the caller may fork at.
struct PrefixMatch {
    std::uint64_t entry = 0;
    std::uint64_t tokens = 0;
    bool inflight = false;
};

enum class PrefixState : std::uint8_t { absent, inflight, sealed, withdrawn };

struct PrefixStatus {
    PrefixState state = PrefixState::absent;
    std::uint64_t tokens = 0; // announced length while in flight, sealed frontier once published
    std::uint32_t lane = 0;
    std::uint32_t leases = 0;
};

// Chained block hash: previous chain value and one full block of token ids.
using PrefixBlockHash = std::uint64_t (*)(std::uint64_t chain, std::span<const std::uint32_t> block);
std::uint64_t prefix_block_hash(std::uint64_t chain, std::span<const std::uint32_t> block) noexcept;

// STRIX_SERVE_PREFIX_FORK: only the exact value "1" enables the fork path; unset keeps today's
// block-aligned join.
bool prefix_fork_enabled_from_env();

class PrefixDirectory {
public:
    // hash == nullptr selects prefix_block_hash; tests pass a weak hash to force collisions.
    explicit PrefixDirectory(std::uint32_t block_tokens = 64, std::uint64_t min_tokens = 256,
                             PrefixBlockHash hash = nullptr);

    // Longest usable prefix of tokens among the entries of the same identity.
    // Chained block hashes find candidates; a token compare decides, so a hash collision cannot match.
    // Sealed entry: usable only at exactly its frontier. In-flight entry: usable at the common length,
    // the producer then has to stop there. One suffix token is always left to the caller, because the
    // suffix prefill produces the lane's first target row. Matches shorter than min_tokens are dropped.
    // Ties prefer a sealed entry, then the older one.
    PrefixMatch lookup(const PrefixKey& key, std::span<const std::uint32_t> tokens) const;

    // Registers a prefill in flight on lane. Returns 0 when the prompt is shorter than min_tokens.
    // Entry numbers are never reused.
    std::uint64_t announce(const PrefixKey& key, std::span<const std::uint32_t> tokens, std::uint32_t lane);

    // The state is sealed at exactly frontier_tokens. A frontier of zero, below min_tokens, beyond the
    // announced tokens, or different from an earlier seal withdraws the entry instead.
    void publish(std::uint64_t entry, std::uint64_t frontier_tokens);

    // Cancel, failure or eviction. The entry stops matching at once. While leases remain it reports
    // PrefixState::withdrawn, so waiters fall back to their own prefill; it is dropped at the last release.
    void withdraw(std::uint64_t entry) noexcept;

    // A waiter or a fork holds a lease while it depends on the entry. acquire fails unless the entry
    // is in flight or sealed.
    bool acquire(std::uint64_t entry) noexcept;
    void release(std::uint64_t entry) noexcept;
    PrefixStatus status(std::uint64_t entry) const noexcept;

    std::size_t size() const noexcept { return entries_.size(); }
    std::uint32_t block_tokens() const noexcept { return block_tokens_; }
    std::uint64_t min_tokens() const noexcept { return min_tokens_; }

    // Admission grouping over waiting and in-flight prompts (list in-flight prompts first: the lowest
    // index of a group is its producer). Each prompt joins at most one group; a group has two or more
    // members and a common prefix of tokens >= min_tokens that leaves every member one suffix token.
    // Groups are chosen by prefill tokens saved, tokens * (members - 1), longest prefix on ties, so a
    // nested prefix yields the deep group first and the remaining prompts group on the shallow one.
    // Result is ordered by producer; members ascend.
    struct Group {
        std::uint64_t tokens = 0;
        std::vector<std::uint32_t> members;
        std::uint32_t producer = 0;
    };
    std::vector<Group> plan_groups(std::span<const std::span<const std::uint32_t>> prompts) const;

private:
    struct Entry {
        PrefixKey key;
        std::vector<std::uint32_t> tokens;
        std::vector<std::uint64_t> chain; // chain[d] covers d full blocks; chain[0] is the identity seed
        std::uint32_t lane = 0;
        std::uint32_t leases = 0;
        PrefixState state = PrefixState::inflight;
    };

    std::uint64_t seed(const PrefixKey& key) const noexcept;
    void chain_of(const PrefixKey& key, std::span<const std::uint32_t> tokens,
                  std::vector<std::uint64_t>& chain) const;
    void unindex(std::uint64_t id, const Entry& entry, std::size_t from_depth) noexcept;

    std::uint32_t block_tokens_;
    std::uint64_t min_tokens_;
    PrefixBlockHash hash_;
    std::uint64_t next_ = 1;
    std::unordered_map<std::uint64_t, Entry> entries_;
    std::unordered_multimap<std::uint64_t, std::uint64_t> index_; // chain value -> entry
};

} // namespace strix::serve
