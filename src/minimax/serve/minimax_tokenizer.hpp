#pragma once

#include "strix/serve/bpe_tokenizer.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::minimax::serve {

// MiniMax M3.x tokenizer, token-identical with HuggingFace `tokenizers` 0.22 on the model's
// tokenizer.json (BPE, NFC normalizer, o200k-style Split + ByteLevel pre-tokenizer, ByteLevel decoder).
//
// encode(text, parse_special):
//   1. added tokens (tokenizer.json `added_tokens`, all `normalized: false`) are found in the raw text
//      leftmost-longest, like the HF AddedVocabulary aho-corasick; with parse_special == false the
//      matches of *special* added tokens are dropped (kept as text, HF `encode_special_tokens = true`)
//      while non-special added tokens (<think>, <mm:think>, <tool_call>, ...) are still matched;
//   2. every other segment is NFC-normalized (Unicode 9.0 tables, = unicode-normalization-alignments),
//      split with the pre-tokenizer regex (Unicode 16.0 classes, = Oniguruma), byte-level mapped and
//      merged by rank.
// Invalid UTF-8 input bytes (impossible through the HF Python API) are kept as single opaque
// "characters": they are never normalized, fall in the regex's symbol class and decode back to the
// same bytes, so decode(encode(x)) == x for any byte string.
// decode(ids, skip_special): byte-level tokens are joined and converted with Rust's
// String::from_utf8_lossy semantics; added tokens are emitted as their text (HF 0.22 decodes the
// byte-level runs between added tokens separately); special added tokens are dropped when skip_special;
// ids outside the vocabulary are ignored (as in HF).
class MinimaxTokenizer {
public:
    static std::shared_ptr<const MinimaxTokenizer> load(const std::string& tokenizer_json_path);
    ~MinimaxTokenizer();

    [[nodiscard]] std::vector<std::uint32_t> encode(std::string_view text, bool parse_special = true) const;
    // encode() that also appends its restart points (strix::serve::EncodeMark): the end of every added token
    // that became an id. mark_lookahead() is the longest added token, the window a restart point needs.
    [[nodiscard]] std::vector<std::uint32_t> encode_marked(std::string_view text, bool parse_special,
                                                           std::vector<strix::serve::EncodeMark>& marks) const;
    [[nodiscard]] std::size_t mark_lookahead() const noexcept;
    [[nodiscard]] std::string decode(std::span<const std::uint32_t> ids, bool skip_special = false) const;

    // ID of an added token's content or of a vocabulary token (byte-level form, e.g. "Ġthe"), or UINT32_MAX.
    [[nodiscard]] std::uint32_t token_id(std::string_view token_text) const;
    // Vocabulary / added-token text of an id (byte-level form for vocabulary tokens); empty when unknown.
    [[nodiscard]] std::string token_text(std::uint32_t id) const;
    // Raw bytes the id decodes to (added tokens: their content).
    [[nodiscard]] std::string token_bytes(std::uint32_t id) const;
    [[nodiscard]] bool is_added(std::uint32_t id) const noexcept;
    [[nodiscard]] bool is_special(std::uint32_t id) const noexcept;

    [[nodiscard]] std::uint32_t bos_id() const noexcept;  // "]~b]"  (200019)
    [[nodiscard]] std::uint32_t eos_id() const noexcept;  // "[e~["  (200020)
    [[nodiscard]] std::uint32_t pad_id() const noexcept;  // "]!p~[" (200000)
    // Highest id + 1 over the model vocabulary and the added tokens (200061 for M3.1).
    [[nodiscard]] std::size_t vocab_size() const noexcept;

    // Building blocks, exposed for tests.
    [[nodiscard]] static std::string nfc(std::string_view text);
    [[nodiscard]] static std::vector<std::string> pre_tokenize(std::string_view normalized_text);
    [[nodiscard]] static const char* class_unicode_version() noexcept;
    [[nodiscard]] static const char* nfc_unicode_version() noexcept;

private:
    MinimaxTokenizer();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::vector<std::uint32_t> encode_marks(std::string_view text, bool parse_special, std::vector<strix::serve::EncodeMark>* marks) const;
};

}  // namespace strix::minimax::serve
