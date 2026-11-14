#pragma once

#include "chat.hpp"
#include "tokenizer.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/tokenizer.hpp"

namespace strix::gemma4::serve {

// Family rendering and tokenization over the shared generation contract.
struct PreparedPrompt {
    std::string rendered;
    std::vector<std::uint32_t> ids, stop_ids;
    bool starts_in_reasoning = false;
    [[nodiscard]] strix::serve::GenerationJob job(std::uint32_t max_tokens) const;
};
class Adapter {
public:
    explicit Adapter(std::shared_ptr<const Tokenizer> tokenizer);
    [[nodiscard]] PreparedPrompt prepare(const ChatInput& input) const;
    [[nodiscard]] const Tokenizer& tokenizer() const noexcept { return *tokenizer_; }
private:
    std::shared_ptr<const Tokenizer> tokenizer_;
};

// The shared API uses {name, arguments} internally and creates wire IDs itself.
[[nodiscard]] strix::serve::AssistantOutput shared_output(const Output& output);
[[nodiscard]] Json shared_calls(const Json& calls);

// Delegates the Unicode/space/byte-fallback tokenizer without reinterpreting assets.
[[nodiscard]] std::shared_ptr<const strix::serve::ApiTokenizer> api_tokenizer(std::shared_ptr<const Tokenizer>);

} // namespace strix::gemma4::serve
