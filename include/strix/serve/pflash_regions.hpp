#pragma once

#include "strix/serve/json.hpp"
#include "strix/serve/prompt_compress.hpp"

#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

// Region adapters for prompt compression before prefill (LOSSY, opt-in, default off): one function per chat family
// that maps a request's messages onto byte ranges of its rendered prompt and says which ranges the selection layer
// may shorten. The adapters do not render: they take the text the family renderer produced and locate each message's
// text in it, checking the turn markers the family template writes around it. A text that cannot be located (media
// parts, content the template rewrites, markers that do not match) gets no region; bytes outside every region are
// always kept, so a miss can only make the prompt longer, never drop something by mistake.
//
// What is never droppable: system / developer / root messages, tool definitions, tool results, every turn after
// the query turn, and all template bytes (role tags, envelopes, reasoning blocks, tool-call frames, generation
// prompt), which are outside every region. A request with tools or tool traffic has no droppable region at all
// and sets `tools`; the caller answers it with bypass(raw, Reason::tools).
//
// What is droppable: the text of user and assistant turns before the query turn (`compress_history`) and the text
// of the query turn itself (`compress_query_turn`). The query turn stays droppable by default because a one-shot
// long prompt carries document and question in the same message; the selection layer keeps the request tokens the
// scorer marks (Policy::query_pin) and every message of at most Policy::skeleton_tokens.
namespace strix::serve::pflash {

enum class RegionKind : std::uint8_t {
    system,             // system / developer / root text
    tool_schema,        // the tool definitions block of the system turn
    history_user,       // user text before the query turn
    history_assistant,  // assistant answer text before the query turn
    tool_result,        // tool message text
    query,              // text of the query turn (the last user message that is not a tool response)
    tail,               // text of turns after the query turn
};

struct RegionOptions {
    bool compress_history = true;
    bool compress_query_turn = true;
};

inline constexpr std::uint32_t no_message = std::numeric_limits<std::uint32_t>::max();

struct RegionMap {
    std::vector<Region> regions;    // ascending and disjoint; bytes of the rendered prompt; Region::message = index
    std::vector<RegionKind> kinds;  // one per region
    std::uint32_t query_message = no_message;
    std::uint32_t unlocated = 0;    // messages whose text was not found in the rendered prompt (kept as rendered)
    bool tools = false;             // tool definitions, tool calls or tool results present: serve the request raw
};

// `messages` / `tools` are the request values given to the family renderer and `rendered` is its output.
RegionMap qwen_regions(const Json& messages, const Json* tools, std::string_view rendered, const RegionOptions& options = {});
RegionMap glm_regions(const Json& messages, const Json* tools, std::string_view rendered, const RegionOptions& options = {});
RegionMap deepseek_regions(const Json& messages, const Json* tools, std::string_view rendered, const RegionOptions& options = {});
RegionMap minimax_regions(const Json& messages, const Json* tools, std::string_view rendered, const RegionOptions& options = {});

const char* region_kind_name(RegionKind kind);

namespace detail {

// The bytes a template writes right before / right after a message text. An empty list is not checked. In `after`
// the end of the prompt always matches.
struct Marks {
    std::vector<std::string_view> before, after;
    bool strip = false;  // the template prints the text without surrounding whitespace
};

struct Family {
    Marks system, user, assistant, tool;
    std::string_view tools_open, tools_close;  // tool definitions block; empty: not located (kept as rendered)
    std::string_view user_part_separator;      // written between the text parts of a user message (default: none)
    bool tools_before_system = false;          // the block precedes the system text in the rendered prompt
    bool query_skips_tool_responses = false;   // a user turn that is one <tool_response> block is not the query
};

RegionMap map_regions(const Json& messages, const Json* tools, std::string_view rendered, const RegionOptions& options,
    const Family& family);

}  // namespace detail

}  // namespace strix::serve::pflash
