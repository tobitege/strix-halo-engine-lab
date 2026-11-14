#pragma once

// Embedding rows in the token stream (image input and other non-text rows).
//
// A prompt stays a sequence of 32-bit ids. A row that is not a vocabulary token is a
// virtual id:
//
//     bit 31 = 1 | handle (17 bits) | row (14 bits)
//
// The handle names one interned span of rows (one image after its encoder), the row is
// the index inside that span. Handles are interned by digest per process, so the same
// content gives the same ids: everything that compares or hashes token ids (prefix
// directories, turn caches, family prefix caches) keeps working without change, and a
// match that ends inside a span is valid because the row index is part of the id.
//
// Three pieces:
//   EmbedInterner  process-wide digest -> handle table; hands out leases
//   EmbedTable     immutable per-request view (handle -> rows); owns leases, so keeping
//                  the table alive keeps its handles and payloads alive
//   expand_embed_rows  replaces each placeholder token of a rendered prompt by the ids
//                  of one span
//
// Text-only prompts contain no virtual id and never touch any of this.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace strix::session {

inline constexpr std::uint32_t kEmbedIdBit = 0x8000'0000U;
inline constexpr unsigned kEmbedHandleBits = 17;
inline constexpr unsigned kEmbedRowBits = 14;
inline constexpr std::uint32_t kEmbedMaxHandles = 1U << kEmbedHandleBits;
inline constexpr std::uint32_t kEmbedMaxRows = 1U << kEmbedRowBits; // rows of one span

[[nodiscard]] constexpr bool is_embed_id(std::uint32_t id) noexcept { return (id & kEmbedIdBit) != 0; }
// handle < kEmbedMaxHandles and row < kEmbedMaxRows are the caller's precondition.
[[nodiscard]] constexpr std::uint32_t embed_id(std::uint32_t handle, std::uint32_t row) noexcept {
    return kEmbedIdBit | (handle << kEmbedRowBits) | row;
}
[[nodiscard]] constexpr std::uint32_t embed_handle(std::uint32_t id) noexcept {
    return (id & ~kEmbedIdBit) >> kEmbedRowBits;
}
[[nodiscard]] constexpr std::uint32_t embed_row(std::uint32_t id) noexcept { return id & (kEmbedMaxRows - 1U); }

[[nodiscard]] std::size_t count_embed_ids(std::span<const std::uint32_t> ids) noexcept;
[[nodiscard]] inline bool has_embed_ids(std::span<const std::uint32_t> ids) noexcept {
    return count_embed_ids(ids) != 0;
}
// Prevalidation of an engine without Capabilities::embed_rows: throws
// std::invalid_argument when ids holds a virtual id. Nothing is written before it.
void reject_embed_ids(std::span<const std::uint32_t> ids);

// Folds the identity of the producer of the rows (tower, weights, arithmetic) into a
// prefix format id, so a change of producer invalidates prefixes through the key.
// adapter_identity == 0 (no producer loaded) returns format_id unchanged.
[[nodiscard]] constexpr std::uint64_t embed_format_id(std::uint64_t format_id,
                                                      std::uint64_t adapter_identity) noexcept {
    if (adapter_identity == 0) return format_id;
    std::uint64_t x = format_id ^ (adapter_identity + 0x9E37'79B9'7F4A'7C15ULL + (format_id << 6) + (format_id >> 2));
    x ^= x >> 33;
    x *= 0xFF51'AFD7'ED55'8CCDULL;
    x ^= x >> 33;
    return x;
}

// Digest of the content AND of the producer identity (same pixels through another tower
// are another span). 32 bytes, e.g. SHA-256.
using EmbedDigest = std::array<std::uint8_t, 32>;

enum class EmbedRole : std::uint8_t {
    content = 0, // an encoder output row
    begin = 1,   // learned start-of-span row
    newline = 2, // learned row separator
    end = 3,     // learned end-of-span row
};

// Rows added to the hidden state after one LM layer (same rows x width as the span).
struct EmbedDeepRows {
    std::uint32_t layer = 0;
    std::vector<std::uint16_t> data; // bf16, rows * width
};

// One span of rows. Immutable once interned.
struct EmbedRows {
    EmbedDigest digest{};
    std::uint32_t rows = 0;  // 1 .. kEmbedMaxRows
    std::uint32_t width = 0; // LM hidden width
    std::uint32_t axes = 0;  // position axes per row; 0: the position is the prompt index
    // The vocabulary token a row stands for where the LM needs one (router flags,
    // n-gram tables, usage accounting).
    std::uint32_t real_token_id = 0;
    std::vector<std::uint16_t> data;  // bf16, rows * width, row-major
    std::vector<EmbedRole> role;      // rows entries; empty at intern: every row is content
    std::vector<std::int32_t> lm_pos; // rows * axes, relative to the first row of the span
    std::vector<EmbedDeepRows> deep;

    [[nodiscard]] std::span<const std::uint16_t> row(std::uint32_t index) const noexcept {
        return std::span<const std::uint16_t>(data).subspan(static_cast<std::size_t>(index) * width, width);
    }
    [[nodiscard]] std::size_t bytes() const noexcept;
};

class EmbedInterner;

// Shared ownership of one handle. The handle keeps its payload while a lease exists.
class EmbedLease {
public:
    EmbedLease() noexcept = default;
    EmbedLease(const EmbedLease& other);
    EmbedLease(EmbedLease&& other) noexcept;
    EmbedLease& operator=(EmbedLease other) noexcept;
    ~EmbedLease();

    [[nodiscard]] explicit operator bool() const noexcept { return rows_ != nullptr; }
    [[nodiscard]] std::uint32_t handle() const noexcept { return handle_; }
    [[nodiscard]] const EmbedRows& rows() const noexcept { return *rows_; }
    // The ids of the whole span in row order.
    [[nodiscard]] std::vector<std::uint32_t> ids() const;

private:
    friend class EmbedInterner;
    std::shared_ptr<void> owner_; // interner state; outlives the interner object
    std::shared_ptr<const EmbedRows> rows_;
    std::uint32_t handle_ = 0;
};

struct EmbedInternerStatus {
    std::uint64_t epoch = 0;     // bumped each time the handle space wrapped
    std::uint32_t live = 0;      // handles with a lease
    std::uint32_t retired = 0;   // handles reserved for a digest, payload dropped
    std::uint64_t interned = 0;  // intern calls that created or revived a handle
    std::uint64_t shared = 0;    // intern calls that found a live handle
    std::uint64_t live_bytes = 0;
};

// Thread-safe. One per process (or per serving owner).
class EmbedInterner {
public:
    explicit EmbedInterner(std::uint32_t capacity = kEmbedMaxHandles); // 1 .. kEmbedMaxHandles
    ~EmbedInterner();
    EmbedInterner(const EmbedInterner&) = delete;
    EmbedInterner& operator=(const EmbedInterner&) = delete;

    // Same digest -> same handle for the life of an epoch, also after every lease of it
    // was dropped (the payload is dropped then, the handle stays reserved, and the next
    // intern revives it). A live digest keeps its first payload; the argument is then
    // discarded. Throws std::invalid_argument for a malformed span or for a digest that
    // is live with another shape, std::length_error when every handle is leased.
    //
    // When the handle space is used up, the epoch is bumped: the reserved handles
    // without a lease are freed for reuse. An owner of cached state whose ids hold
    // virtual ids must hold a lease per handle or drop that state when epoch() changed.
    [[nodiscard]] EmbedLease intern(EmbedRows rows);
    // A lease on a live digest; empty when the digest is unknown or its payload dropped.
    [[nodiscard]] EmbedLease find(const EmbedDigest& digest) const;
    [[nodiscard]] std::uint64_t epoch() const noexcept;
    [[nodiscard]] EmbedInternerStatus status() const noexcept;

private:
    struct State;
    std::shared_ptr<State> state_;
    friend class EmbedLease;
};

// What leaves the process instead of a handle (pair frames, proxies).
struct EmbedManifestEntry {
    std::uint32_t handle = 0;
    std::uint32_t rows = 0;
    std::uint32_t width = 0;
    EmbedDigest digest{};
};

// Immutable handle -> rows view of one request. Engines resolve virtual ids here.
class EmbedTable {
public:
    EmbedTable() = default;
    explicit EmbedTable(std::vector<EmbedLease> leases); // empty leases are refused

    [[nodiscard]] bool empty() const noexcept { return leases_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return leases_.size(); }
    [[nodiscard]] std::span<const EmbedLease> leases() const noexcept { return leases_; }
    // nullptr when the handle is not part of this request.
    [[nodiscard]] const EmbedRows* find(std::uint32_t handle) const noexcept;
    // Prevalidation: every virtual id of ids names a handle of this table and a row
    // inside it. Throws std::invalid_argument otherwise.
    void validate(std::span<const std::uint32_t> ids) const;
    [[nodiscard]] std::vector<EmbedManifestEntry> manifest() const;

private:
    std::vector<EmbedLease> leases_; // sorted by handle, unique
};

// Rewrites virtual ids made in another process (described by its manifest) to the
// handles of local, which must hold the same digests with the same shape.
[[nodiscard]] std::vector<std::uint32_t> remap_embed_ids(std::span<const std::uint32_t> ids,
                                                         std::span<const EmbedManifestEntry> remote,
                                                         const EmbedTable& local);

struct EmbedRun {
    std::size_t source = 0; // index of the placeholder in the rendered tokens
    std::size_t offset = 0; // first id of the run in the expanded tokens
    std::size_t size = 0;   // ids of the run (virtual and real)
    std::size_t rows = 0;   // virtual ids of the run
};

struct ExpandedPrompt {
    std::vector<std::uint32_t> tokens;
    std::vector<EmbedRun> runs;
    // Maps an offset of the rendered tokens (e.g. a stable prefix length) to the
    // expanded tokens. An offset behind a placeholder lands behind its whole run.
    [[nodiscard]] std::size_t map_offset(std::size_t source_offset) const noexcept;
};

// The n-th occurrence of placeholder is replaced by layouts[n]: the ids of one span in
// prompt order, virtual ids for rows that come from the table and real ids for special
// tokens that stay vocabulary tokens. Throws std::invalid_argument when the rendered
// tokens already hold a virtual id, when the number of placeholders differs from the
// number of layouts, or when a layout is empty or holds the placeholder.
[[nodiscard]] ExpandedPrompt expand_embed_rows(std::span<const std::uint32_t> tokens, std::uint32_t placeholder,
                                               std::span<const std::vector<std::uint32_t>> layouts);

} // namespace strix::session
