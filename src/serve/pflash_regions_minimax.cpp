#include "strix/serve/pflash_regions.hpp"

// MiniMax chat template (minimax_chat.cpp): a system turn (root message or the built-in identity), a developer turn
// (first system / developer message, then the tool definitions), then "]~b]user\n" text "[e~[", assistant turns
// "]~b]ai\n" + reasoning block or "</mm:think>" + answer + tool calls + "[e~[", and tool messages as "<response>"
// entries of one tool turn. Texts are printed as given; an assistant message that carries its reasoning inside the
// content is rewritten by the template and is not located.
namespace strix::serve::pflash {

RegionMap minimax_regions(const Json& messages, const Json* tools, std::string_view rendered, const RegionOptions& options) {
    detail::Family family;
    family.user = {{"]~b]user\n"}, {"[e~["}, false};
    family.assistant = {{"</mm:think>"}, {"[e~[", "]<]minimax[>[<tool_call>"}, false};
    family.tool = {{"\n<response>"}, {"</response>"}, false};
    family.tools_open = "<tools>\n";
    family.tools_close = "</tools>";
    return detail::map_regions(messages, tools, rendered, options, family);
}

}  // namespace strix::serve::pflash
