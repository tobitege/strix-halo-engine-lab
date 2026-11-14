#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::serve {

// A restart point of an encoding of a text T: byte offset `bytes` (the end of a special token) and the number of
// ids emitted before it. For every text U that equals T on the first `bytes + mark_lookahead()` bytes,
// encode(U) == encode(T)[0, ids) ++ encode(U.substr(bytes)): the scan carries no state across a special token,
// and no special token that starts before `bytes` reads past that window.
struct EncodeMark {
    std::size_t bytes = 0;
    std::size_t ids = 0;
};

// Byte-level BPE tokenizer read from a GGUF file's tokenizer metadata (`tokenizer.ggml.tokens`,
// `token_type`, `merges`, `pre`). Encoding splits the text on control and user-defined tokens
// (when special parsing is on), pre-tokenizes every other piece with the model's pattern, maps its
// bytes to the byte-level alphabet and applies the merges by rank. Decoding inverts the byte map.
// The `qwen35`, the DeepSeek V3/V4 (`joyai-llm`, `deepseek-v3`) and the GLM-4/5 (`glm4`) pre-tokenizers are
// implemented; other tokenizers are rejected at load.
class BpeTokenizer {
public:
    enum class TokenType : std::int32_t { normal = 1, unknown = 2, control = 3, user_defined = 4, unused = 5, byte = 6 };

    static std::shared_ptr<const BpeTokenizer> load_gguf(const std::string& path);
    // A Hugging Face tokenizer.json of the GLM-4/5 layout (byte-level BPE, the GLM split pattern, ignore_merges);
    // added tokens become control (special) or user-defined tokens; eos = <|endoftext|>.
    static std::shared_ptr<const BpeTokenizer> load_hf_json(const std::string& path);
    static std::shared_ptr<const BpeTokenizer> from_parts(std::vector<std::string> tokens, std::vector<std::int32_t> types,
        std::span<const std::string> merges, std::string pre);
    ~BpeTokenizer();

    // parse_special: control and user-defined token texts in `text` become their IDs.
    [[nodiscard]] std::vector<std::uint32_t> encode(std::string_view text, bool parse_special = true) const;
    // encode() that also appends its restart points (see EncodeMark), in increasing byte order.
    [[nodiscard]] std::vector<std::uint32_t> encode_marked(std::string_view text, bool parse_special,
                                                           std::vector<EncodeMark>& marks) const;
    // Bytes of the longest control / user-defined token text: the window an EncodeMark needs after its offset.
    [[nodiscard]] std::size_t mark_lookahead() const noexcept;
    // Incremental form of encode(). The caller keeps `text` unchanged and a cursor that starts at zero; each call
    // encodes about `slice_bytes` more text and appends the ids, which are final: the ids appended over all calls are
    // exactly encode(text, parse_special). Returns false (and appends nothing) once the text is fully encoded.
    // A slice ends at a special token or between an ASCII letter and a following space, where every implemented
    // pre-tokenizer starts a new piece whatever follows; a text without such places is encoded in one final call.
    struct EncodeCursor {
        std::size_t encoded = 0;  // bytes [0, encoded) are encoded
        std::size_t scanned = 0;  // bytes [encoded, scanned) hold no special token start
        std::size_t searched = 0; // no slice end exists in (encoded, searched]
    };
    bool encode_some(std::string_view text, bool parse_special, EncodeCursor& cursor, std::size_t slice_bytes,
                     std::vector<std::uint32_t>& out) const;
    // Raw bytes of the tokens; control tokens are rendered as their text unless skip_control.
    [[nodiscard]] std::string decode(std::span<const std::uint32_t> ids, bool skip_control = false) const;
    [[nodiscard]] std::string token_bytes(std::uint32_t id) const;
    [[nodiscard]] std::size_t vocab_size() const noexcept;
    [[nodiscard]] TokenType type(std::uint32_t id) const;
    // ID of an exact token text (special or normal byte-level form), or UINT32_MAX.
    [[nodiscard]] std::uint32_t find(std::string_view token_text) const;
    [[nodiscard]] const std::string& pre() const noexcept;
    [[nodiscard]] std::uint32_t eos_token() const noexcept;

    // Pre-tokenizer pieces of plain text (no special-token splitting), for tests.
    [[nodiscard]] static std::vector<std::string> pre_tokenize(std::string_view text);
    [[nodiscard]] static std::vector<std::string> pre_tokenize(std::string_view text, std::string_view pre);

private:
    BpeTokenizer();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::vector<std::uint32_t> encode_marks(std::string_view text, bool parse_special, std::vector<EncodeMark>* marks) const;
};

// Incremental decoder: feeds token bytes and returns only complete UTF-8 characters; a sequence
// cut between tokens is held until its last byte arrives (invalid bytes become U+FFFD at flush).
class Utf8StreamDecoder {
public:
    std::string push(std::string_view bytes);
    std::string flush();
    [[nodiscard]] std::size_t pending() const noexcept { return held_.size(); }
private:
    std::string held_;
};

}  // namespace strix::serve
