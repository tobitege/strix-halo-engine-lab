#include "strix/serve/pflash_regions.hpp"

// GLM chat template (glm_chat.cpp): "[gMASK]<sop>", an optional "<|system|>" tools block first, then one turn per
// message: "<|system|>" text, "<|user|>" text (both as given), "<|assistant|><think>...</think>" followed by the
// stripped answer and the tool calls, and tool messages as <tool_response> entries of an <|observation|> block. A
// turn has no end tag: it ends where the next "<|role|>" tag, a tool call or the prompt end begins.
namespace strix::serve::pflash {

RegionMap glm_regions(const Json& messages, const Json* tools, std::string_view rendered, const RegionOptions& options) {
    detail::Family family;
    family.system = {{"<|system|>"}, {"<|"}, false};
    family.user = {{"<|user|>"}, {"<|"}, false};
    family.assistant = {{"</think>"}, {"<|", "<tool_call>"}, true};
    family.tool = {{"<tool_response>"}, {"</tool_response>"}, false};
    family.tools_open = "<tools>\n";
    family.tools_close = "</tools>";
    family.tools_before_system = true;
    return detail::map_regions(messages, tools, rendered, options, family);
}

}  // namespace strix::serve::pflash
