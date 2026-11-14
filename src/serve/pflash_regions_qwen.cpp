#include "strix/serve/pflash_regions.hpp"

// Qwen chat template (chat_template.cpp): the system turn holds the tool definitions block and then the merged leading
// system / developer texts; "<|im_start|>user\n" text "<|im_end|>"; the assistant answer follows the role line or the
// closed reasoning block and ends at "<|im_end|>" or at the first tool call; tool messages are <tool_response> blocks
// inside a user turn. Every text is printed stripped. The query is the last user message that is not itself one
// <tool_response> block, the same rule the template uses to decide which reasoning blocks it keeps.
namespace strix::serve::pflash {

RegionMap qwen_regions(const Json& messages, const Json* tools, std::string_view rendered, const RegionOptions& options) {
    detail::Family family;
    family.system.strip = true;
    family.user = {{"<|im_start|>user\n"}, {"<|im_end|>"}, true};
    family.assistant = {{"<|im_start|>assistant\n", "</think>\n\n"}, {"<|im_end|>", "\n\n<tool_call>"}, true};
    family.tool = {{"\n<tool_response>\n"}, {"\n</tool_response>"}, true};
    family.tools_open = "<tools>";
    family.tools_close = "</tools>";
    family.tools_before_system = true;
    family.query_skips_tool_responses = true;
    return detail::map_regions(messages, tools, rendered, options, family);
}

}  // namespace strix::serve::pflash
