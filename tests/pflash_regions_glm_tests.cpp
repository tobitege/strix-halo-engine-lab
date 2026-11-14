// Region adapter of the GLM chat family against the family's own renderer.
#include "pflash_regions_test_support.hpp"

#include "strix/serve/glm_chat.hpp"

using namespace pflash_regions_test;

namespace {

std::string render(const Json& messages, const Json* tools) {
    strix::serve::ChatTemplateInput input;
    input.messages = messages;
    if (tools) input.tools = *tools;
    return strix::serve::render_glm_chat(input);
}

}  // namespace

int main() {
    quiet_aborts();
    try {
        run_family({"glm", render, glm_regions, {"<|user|>", "<|assistant|>", "<|system|>", "<think>"}, "<|user|>", {}, true});

        // A system message in the middle of the conversation is its own turn and stays mandatory.
        Json messages = Json::array();
        messages.as_array().push_back(message("user", "first question"));
        messages.as_array().push_back(message("assistant", "first answer"));
        messages.as_array().push_back(message("system", "From now on answer in French."));
        messages.as_array().push_back(message("user", "second question"));
        const std::string rendered = render(messages, nullptr);
        const RegionMap map = glm_regions(messages, nullptr, rendered);
        check_structure(map, rendered, "glm late system");
        expect_region(map, rendered, 2, RegionKind::system, "From now on answer in French.", false, "glm late system");
        expect_region(map, rendered, 3, RegionKind::query, "second question", true, "glm late system query");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: exception: " << error.what() << '\n';
        return 1;
    }
    return finish("pflash-regions-glm");
}
