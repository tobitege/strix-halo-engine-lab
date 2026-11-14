// Region adapter of the Qwen chat family against the family's own renderer.
#include "pflash_regions_test_support.hpp"

#include "strix/serve/chat_template.hpp"

using namespace pflash_regions_test;

namespace {

std::string render(const Json& messages, const Json* tools) {
    strix::serve::ChatTemplateInput input;
    input.messages = messages;
    if (tools) input.tools = *tools;
    return strix::serve::render_chat_template(input);
}

}  // namespace

int main() {
    quiet_aborts();
    try {
        run_family({"qwen", render, qwen_regions, {"<|im_start|>", "<|im_end|>"}, "<|im_start|>user\n", {}, true});

        // A user turn that is one <tool_response> block is tool traffic, not the query (the template's own rule).
        Json messages = Json::array();
        messages.as_array().push_back(message("user", "What is the weather?"));
        messages.as_array().push_back(message("assistant", "Calling the tool."));
        messages.as_array().push_back(message("user", "<tool_response>\nsunny\n</tool_response>"));
        const std::string rendered = render(messages, nullptr);
        const RegionMap map = qwen_regions(messages, nullptr, rendered);
        check_structure(map, rendered, "qwen tool response turn");
        check_raw(map, "qwen tool response turn");
        check(map.query_message == 0U, "qwen tool response turn: the query is the earlier user message");

        // The template prints texts stripped: the region is the stripped text.
        Json padded = Json::array();
        padded.as_array().push_back(message("system", "  Be brief.\n"));
        padded.as_array().push_back(message("user", "\n\n  padded question \n"));
        const std::string padded_rendered = render(padded, nullptr);
        const RegionMap padded_map = qwen_regions(padded, nullptr, padded_rendered);
        expect_region(padded_map, padded_rendered, 0, RegionKind::system, "Be brief.", false, "qwen stripped system");
        expect_region(padded_map, padded_rendered, 1, RegionKind::query, "padded question", true, "qwen stripped query");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: exception: " << error.what() << '\n';
        return 1;
    }
    return finish("pflash-regions-qwen");
}
