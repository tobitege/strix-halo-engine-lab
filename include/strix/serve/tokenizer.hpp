#pragma once

#include "strix/serve/bpe_tokenizer.hpp"
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::serve {

// One text being encoded in steps. next() appends the next run of ids and returns true, or returns false (appending
// nothing) once the text is complete. Appended ids are final: no later step changes them, and the ids appended over
// all steps are exactly ApiTokenizer::encode() of the same text.
class IncrementalEncoding {
public:
    virtual ~IncrementalEncoding() = default;
    virtual bool next(std::vector<std::uint32_t>& out) = 0;
};

// Device-neutral tokenizer operations used at request and token-piece boundaries.
// Optional metadata must refuse explicitly; absence is never an invented token ID.
class ApiTokenizer {
public:
    using TokenType = BpeTokenizer::TokenType;
    ApiTokenizer() = default;
    ApiTokenizer(const ApiTokenizer&) {} // a copy starts a new memo lifetime
    ApiTokenizer& operator=(const ApiTokenizer&) {
        memo_identity_ = std::make_shared<const char>();
        return *this;
    }
    // Encodings/configuration must remain immutable within a memo lifetime. Entries retain this
    // token, so destroying/replacing the tokenizer cannot recycle its identity while entries live.
    [[nodiscard]] std::shared_ptr<const void> memo_identity() const { return memo_identity_; }
    virtual ~ApiTokenizer() = default;
    virtual std::vector<std::uint32_t> encode(std::string_view, bool parse_special = true) const = 0;
    virtual std::string decode(std::span<const std::uint32_t>, bool skip_control = false) const = 0;
    // encode() with its restart points (EncodeMark). The default reports none, which is always correct: a memo of
    // such a tokenizer reuses only whole texts.
    virtual std::vector<std::uint32_t> encode_marked(std::string_view text, bool parse_special,
                                                     std::vector<EncodeMark>& marks) const {
        (void)marks;
        return encode(text, parse_special);
    }
    virtual std::size_t mark_lookahead() const { return 0; }
    // Incremental form of encode(); the tokenizer must outlive the returned object. The default yields the whole
    // encoding in one step, so every tokenizer supports the interface with identical ids.
    virtual std::unique_ptr<IncrementalEncoding> encode_incremental(std::string text, bool parse_special = true,
                                                                    std::size_t slice_bytes = 16384) const {
        (void)slice_bytes;
        class OneStep final : public IncrementalEncoding {
        public:
            OneStep(const ApiTokenizer& tokenizer, std::string text, bool special)
                : tokenizer_(tokenizer), text_(std::move(text)), special_(special) {}
            bool next(std::vector<std::uint32_t>& out) override {
                if (done_) return false;
                const auto ids = tokenizer_.encode(text_, special_);
                out.insert(out.end(), ids.begin(), ids.end());
                done_ = true;
                return true;
            }
        private:
            const ApiTokenizer& tokenizer_;
            std::string text_;
            bool special_, done_ = false;
        };
        return std::make_unique<OneStep>(*this, std::move(text), parse_special);
    }
    virtual std::string token_bytes(std::uint32_t) const = 0;
    virtual std::size_t vocab_size() const = 0;
    virtual std::uint32_t find(std::string_view) const = 0;
    virtual TokenType type(std::uint32_t) const {
        throw std::invalid_argument("tokenizer does not expose token types");
    }
    virtual std::uint32_t eos_token() const {
        throw std::invalid_argument("tokenizer does not expose an EOS token");
    }
private:
    std::shared_ptr<const void> memo_identity_ = std::make_shared<const char>();
};

// An explicit binding keeps the legacy ApiService(config, {}, ...) constructor
// unambiguous for health/deployment-only services without a tokenizer.
struct ApiTokenizerBinding {
    explicit ApiTokenizerBinding(std::shared_ptr<const ApiTokenizer> tokenizer) : value(std::move(tokenizer)) {}
    std::shared_ptr<const ApiTokenizer> value;
};

// Delegates every operation; asset loading and tokenizer algorithms stay owned
// by BpeTokenizer. The existing ApiService constructor uses this adapter.
class BpeApiTokenizer final : public ApiTokenizer {
public:
    explicit BpeApiTokenizer(std::shared_ptr<const BpeTokenizer> tokenizer) : tokenizer_(std::move(tokenizer)) {
        if (!tokenizer_) throw std::invalid_argument("API tokenizer is required");
    }
    std::vector<std::uint32_t> encode(std::string_view text, bool special = true) const override { return tokenizer_->encode(text, special); }
    std::vector<std::uint32_t> encode_marked(std::string_view text, bool special, std::vector<EncodeMark>& marks) const override {
        return tokenizer_->encode_marked(text, special, marks);
    }
    std::size_t mark_lookahead() const override { return tokenizer_->mark_lookahead(); }
    std::string decode(std::span<const std::uint32_t> ids, bool skip = false) const override { return tokenizer_->decode(ids, skip); }
    std::unique_ptr<IncrementalEncoding> encode_incremental(std::string text, bool special = true,
                                                            std::size_t slice_bytes = 16384) const override {
        class Steps final : public IncrementalEncoding {
        public:
            Steps(std::shared_ptr<const BpeTokenizer> tokenizer, std::string text, bool special, std::size_t slice)
                : tokenizer_(std::move(tokenizer)), text_(std::move(text)), special_(special), slice_(slice) {}
            bool next(std::vector<std::uint32_t>& out) override {
                return tokenizer_->encode_some(text_, special_, cursor_, slice_, out);
            }
        private:
            std::shared_ptr<const BpeTokenizer> tokenizer_;
            std::string text_;
            bool special_;
            std::size_t slice_;
            BpeTokenizer::EncodeCursor cursor_;
        };
        return std::make_unique<Steps>(tokenizer_, std::move(text), special, slice_bytes);
    }
    std::string token_bytes(std::uint32_t id) const override { return tokenizer_->token_bytes(id); }
    std::size_t vocab_size() const override { return tokenizer_->vocab_size(); }
    std::uint32_t find(std::string_view text) const override { return tokenizer_->find(text); }
    TokenType type(std::uint32_t id) const override { return tokenizer_->type(id); }
    std::uint32_t eos_token() const override { return tokenizer_->eos_token(); }
private:
    std::shared_ptr<const BpeTokenizer> tokenizer_;
};

} // namespace strix::serve
