// Region adapter of the DeepSeek chat family against the family's own renderer.
#include "pflash_regions_test_support.hpp"

#include "strix/serve/deepseek_chat.hpp"

using namespace pflash_regions_test;

namespace {

constexpr std::string_view kUser = "<\xEF\xBD\x9CUser\xEF\xBD\x9C>";
constexpr std::string_view kAssistant = "<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>";

std::string render(const Json& messages, const Json* tools) {
    strix::serve::ChatTemplateInput input;
    input.messages = messages;
    if (tools) input.tools = *tools;
    return strix::serve::render_deepseek_v41_chat(input);
}

}  // namespace

int main() {
    quiet_aborts();
    try {
        run_family({"deepseek", render, deepseek_regions, {kUser, kAssistant}, kUser, "\n\n", false});

        // Consecutive user messages share one turn (blocks joined by a blank line): each keeps its own region.
        Json messages = Json::array();
        messages.as_array().push_back(message("user", "first block"));
        messages.as_array().push_back(message("user", "second block"));
        const std::string rendered = render(messages, nullptr);
        const RegionMap map = deepseek_regions(messages, nullptr, rendered);
        check_structure(map, rendered, "deepseek merged turns");
        expect_region(map, rendered, 0, RegionKind::history_user, "first block", true, "deepseek merged first");
        expect_region(map, rendered, 1, RegionKind::query, "second block", true, "deepseek merged second");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: exception: " << error.what() << '\n';
        return 1;
    }
    return finish("pflash-regions-deepseek");
}
