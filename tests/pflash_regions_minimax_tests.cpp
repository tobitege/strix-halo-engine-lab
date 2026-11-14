// Region adapter of the MiniMax chat family against the family's own renderer.
#include "pflash_regions_test_support.hpp"

#include "minimax/serve/minimax_chat.hpp"

using namespace pflash_regions_test;

namespace {

std::string render(const Json& messages, const Json* tools) {
    return strix::minimax::serve::render_minimax_chat(messages, tools);
}

}  // namespace

int main() {
    quiet_aborts();
    try {
        run_family({"minimax", render, minimax_regions, {"]~b]", "[e~[", "]~!b["}, "]~b]user\n", {}, true});

        // A root message is the system turn and the next system message the developer turn: both mandatory.
        Json messages = Json::array();
        messages.as_array().push_back(message("root", "Root policy text."));
        messages.as_array().push_back(message("developer", "Developer instructions."));
        messages.as_array().push_back(message("user", "a question"));
        const std::string rendered = render(messages, nullptr);
        const RegionMap map = minimax_regions(messages, nullptr, rendered);
        check_structure(map, rendered, "minimax root");
        expect_region(map, rendered, 0, RegionKind::system, "Root policy text.", false, "minimax root");
        expect_region(map, rendered, 1, RegionKind::system, "Developer instructions.", false, "minimax developer");
        expect_region(map, rendered, 2, RegionKind::query, "a question", true, "minimax query");

        // Reasoning carried inside the content is rewritten by the template: no region, the miss is counted.
        Json inline_reasoning = Json::array();
        inline_reasoning.as_array().push_back(message("user", "first"));
        inline_reasoning.as_array().push_back(message("assistant", "<mm:think>\nhm\n</mm:think>\nanswer"));
        inline_reasoning.as_array().push_back(message("user", "second"));
        const std::string rewritten = render(inline_reasoning, nullptr);
        const RegionMap partial = minimax_regions(inline_reasoning, nullptr, rewritten);
        check_structure(partial, rewritten, "minimax inline reasoning");
        check(partial.unlocated == 1U && partial.regions.size() == 2U, "minimax inline reasoning: one miss");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: exception: " << error.what() << '\n';
        return 1;
    }
    return finish("pflash-regions-minimax");
}
