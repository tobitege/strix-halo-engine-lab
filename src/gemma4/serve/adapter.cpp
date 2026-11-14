#include "adapter.hpp"

#include <stdexcept>

namespace strix::gemma4::serve {

namespace {
class ApiTokenizer final : public strix::serve::ApiTokenizer {
public:
    explicit ApiTokenizer(std::shared_ptr<const Tokenizer> t) : tokenizer_(std::move(t)) {
        if (!tokenizer_) throw std::invalid_argument("Gemma4 API tokenizer is required");
    }
    std::vector<std::uint32_t> encode(std::string_view text, bool special) const override { return tokenizer_->encode(text, special); }
    std::string decode(std::span<const std::uint32_t> ids, bool skip) const override { return tokenizer_->decode(ids, skip); }
    std::string token_bytes(std::uint32_t id) const override { return tokenizer_->token_bytes(id); }
    std::size_t vocab_size() const override { return tokenizer_->vocab_size(); }
    std::uint32_t find(std::string_view text) const override { return tokenizer_->find(text); }
    std::uint32_t eos_token() const override {
        const auto id = tokenizer_->find("<eos>");
        if (id >= tokenizer_->vocab_size()) throw std::invalid_argument("Gemma4 tokenizer has no EOS token");
        return id;
    }
    // The family tokenizer exposes no token-type inventory. Constraints are
    // refused by runtime capability before this unsupported operation is needed.
private:
    std::shared_ptr<const Tokenizer> tokenizer_;
};
} // namespace
std::shared_ptr<const strix::serve::ApiTokenizer> api_tokenizer(std::shared_ptr<const Tokenizer> tokenizer) {
    return std::make_shared<ApiTokenizer>(std::move(tokenizer));
}

strix::serve::GenerationJob PreparedPrompt::job(std::uint32_t maximum) const {
    if (!maximum) throw std::invalid_argument("gemma4 adapter: max_tokens must be positive");
    strix::serve::GenerationJob out;
    out.prompt = ids;
    out.stop_tokens = stop_ids;
    out.max_tokens = maximum;
    return out;
}
Adapter::Adapter(std::shared_ptr<const Tokenizer> tokenizer) : tokenizer_(std::move(tokenizer)) {
    if (!tokenizer_ || tokenizer_->find("<bos>") != 2 || tokenizer_->find("<eos>") != 1 || tokenizer_->find("<turn|>") != 106)
        throw std::invalid_argument("gemma4 adapter: incompatible tokenizer boundaries");
}
PreparedPrompt Adapter::prepare(const ChatInput& input) const {
    PreparedPrompt out;
    out.rendered = render_chat(input);
    out.ids = tokenizer_->encode(out.rendered);
    out.stop_ids = {tokenizer_->find("<eos>"), tokenizer_->find("<turn|>")};
    out.starts_in_reasoning = out.rendered.ends_with("<|channel>thought\n");
    return out;
}
strix::serve::AssistantOutput shared_output(const Output& output) {
    strix::serve::AssistantOutput out;
    out.reasoning = output.reasoning; out.content = output.content;
    out.reasoning_complete = output.reasoning_complete;
    out.unfinished_call = output.unfinished_call;
    out.tool_calls=shared_calls(output.tool_calls);
    out.call_complete.assign(out.tool_calls.size(),true);
    return out;
}

Json shared_calls(const Json& calls) {
    auto out=Json::array();
    for (const auto& call : calls.as_array()) {
        const auto* fn = call.find("function");
        if (!fn || !fn->find("name") || !fn->find("arguments")) throw std::invalid_argument("gemma4 adapter: invalid parsed call");
        Json normalized=Json::object();
        if(const auto* id=call.find("id")) normalized.set("id",*id);
        normalized.set("name",*fn->find("name"));normalized.set("arguments",*fn->find("arguments"));
        out.push(std::move(normalized));
    }
    return out;
}

} // namespace strix::gemma4::serve
