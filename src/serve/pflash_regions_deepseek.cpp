#include "strix/serve/pflash_regions.hpp"

// DeepSeek chat encoding (deepseek_chat.cpp): begin-of-sentence, the first system text (tool definitions are appended
// to it), user turns whose blocks (text parts, user texts and <tool_result> entries of merged messages) are joined by
// a blank line, and assistant turns where the answer follows "</think>" (chat mode or closed reasoning) and ends at the
// end-of-sentence tag or at the tool-call block. Texts are printed as given. The tool definitions are not located as
// a region: they sit outside every region and are kept as rendered. Control tags start with "<" + U+FF5C.
namespace strix::serve::pflash {

RegionMap deepseek_regions(const Json& messages, const Json* tools, std::string_view rendered, const RegionOptions& options) {
    static constexpr std::string_view tag = "<\xEF\xBD\x9C";
    static constexpr std::string_view user = "<\xEF\xBD\x9CUser\xEF\xBD\x9C>";
    detail::Family family;
    family.user = {{user, "\n\n"}, {tag, "\n\n"}, false};
    family.assistant = {{"</think>"}, {tag, "\n\n"}, false};
    family.user_part_separator = "\n\n";
    family.tool = {{"<tool_result>"}, {"</tool_result>"}, false};
    return detail::map_regions(messages, tools, rendered, options, family);
}

}  // namespace strix::serve::pflash
