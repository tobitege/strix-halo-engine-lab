#pragma once

#include "strix/serve/tokenizer.hpp"

#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace strix::serve {

// Memo of prompt encodings. A chat resends its whole history every turn; the memo keeps the text and ids of recent
// prompts and, for a new text, reuses the ids up to the last restart point (EncodeMark) that lies inside the bytes
// shared with a kept text, and encodes only the rest. The result is exactly encode() of the whole text: restart
// points come from the tokenizer, and one is used only when the shared bytes cover its lookahead window. A
// tokenizer without restart points gets whole-text hits only.
//
// Entries are keyed by an identity (tokenizer, special-token parsing, caller salt for the template / policy that
// produced the text); texts of different identities never meet. Memory is bounded by bytes and entries with
// least-recently-used eviction. A text that continues a kept one replaces it, so one conversation holds one entry
// while conversations that only share a prefix keep their own. Thread-safe; encoding runs outside the lock.
class TokenMemo {
public:
    struct Limits {
        std::size_t max_bytes = std::size_t{64} << 20;
        std::size_t max_entries = 32;
    };
    struct Stats {
        std::uint64_t calls = 0, whole = 0, partial = 0, missed = 0;
        std::uint64_t reused_bytes = 0, encoded_bytes = 0, evictions = 0;
        std::size_t entries = 0, bytes = 0;
    };
    // Encodes one text and appends its restart points, offsets relative to that text.
    using Encode = std::function<std::vector<std::uint32_t>(std::string_view, std::vector<EncodeMark>&)>;

    TokenMemo() : TokenMemo(default_limits()) {}
    explicit TokenMemo(Limits limits) : limits_(limits) {}

    // STRIX_SERVE_TOKEN_MEMO (default on), STRIX_SERVE_TOKEN_MEMO_MB, STRIX_SERVE_TOKEN_MEMO_ENTRIES.
    [[nodiscard]] static bool enabled_by_default();
    [[nodiscard]] static Limits default_limits();

    // The callback caller supplies a unique identity for immutable vocabulary/configuration.
    [[nodiscard]] std::vector<std::uint32_t> encode(std::uint64_t identity, std::size_t lookahead, std::string_view text,
                                                    const Encode& encode);
    // Keyed by retained tokenizer lifetime, parsing flag and salt, with field-by-field equality.
    [[nodiscard]] std::vector<std::uint32_t> encode(const ApiTokenizer& tokenizer, std::string_view text, bool parse_special,
                                                    std::uint64_t salt = 0);
    [[nodiscard]] Stats stats() const;
    void clear();

private:
    struct Key {
        std::uint64_t identity = 0;
        std::shared_ptr<const void> tokenizer;
        bool parse_special = false;
        bool operator==(const Key&) const = default;
    };
    struct Entry {
        Key key;
        std::string text;
        std::vector<std::uint32_t> ids;
        std::vector<EncodeMark> marks;
        [[nodiscard]] std::size_t bytes() const noexcept {
            // Include spare capacity, the string terminator, entry and list node, and an allowance
            // for shared ownership (entry + lifetime token). Allocator metadata is not portable.
            return sizeof(Entry) + sizeof(std::shared_ptr<const Entry>) + 8U * sizeof(void*) +
                   text.capacity() + 1U + ids.capacity() * sizeof(std::uint32_t) + marks.capacity() * sizeof(EncodeMark);
        }
    };
    using EntryPtr = std::shared_ptr<const Entry>;
    [[nodiscard]] std::vector<std::uint32_t> encode(Key key, std::size_t lookahead, std::string_view text,
                                                    const Encode& encode);

    Limits limits_;
    mutable std::mutex mutex_;
    std::list<EntryPtr> entries_; // most recently used first
    std::size_t bytes_ = 0;
    Stats stats_;
};

} // namespace strix::serve
