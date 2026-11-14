#pragma once

#include "strix/serve/json.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::serve {

class ChatTemplateError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

// Bytes following the end-thinking control in this template.
inline constexpr const char* thinking_answer_separator = "\n\n";

// Inputs of the model's chat template (the GGUF's Qwen3.8 template, rendered natively).
struct ChatTemplateInput {
    Json messages;                     // [{role, content, reasoning_content?, tool_calls?}, ...]
    Json tools;                        // null or [tool, ...]; each tool is emitted with tojson
    bool add_generation_prompt = true;
    std::optional<bool> enable_thinking;          // unset = thinking on
    std::optional<std::string> reasoning_effort;  // xhigh (default), high (= xhigh), medium, low
    std::optional<bool> preserve_thinking;        // unset = keep earlier reasoning blocks
};

// Renders the prompt text exactly as the model's template does for text content. Image and
// video parts are rejected (the engine has no vision input); a tool call's `arguments` may be an
// object or a JSON string (parsed here, as the template requires an object).
[[nodiscard]] std::string render_chat_template(const ChatTemplateInput& input);

// The generated text of one assistant turn, split as the template writes it back.
struct AssistantOutput {
    std::string reasoning;   // text before </think> when thinking is on
    std::string content;     // text after it, before the first tool call
    Json tool_calls;         // [{name, arguments(object)}]
    // per tool call: its frame was closed (a call cut by the budget, a stop string or EOS is parsed but not closed);
    // empty: every call is closed (parsers that only return closed calls)
    std::vector<bool> call_complete;
    bool unfinished_call = false;  // a tool-call frame did not close (whether or not a call could be parsed from it)
    bool reasoning_complete = false;
    std::string withheld;          // tool_choice "none": the notice appended to content when a call was withheld
};
// thinking: the prompt ended with "<think>\n" (generation starts inside the reasoning block).
// tools: the request's tools, used to type parameter values (string unless the schema says otherwise).
[[nodiscard]] AssistantOutput parse_assistant_output(const std::string& text, bool thinking, const Json& tools);

// ---- tool_choice "none" (shared by every chat API and model family) ----
// Removing the tool definitions is not enough once the history holds earlier calls: the model can imitate them.
// The request keeps its history (calls and results stay renderable), gets a closing instruction, and any call the
// model still emits is withheld from the reply.
inline constexpr const char* no_tools_hint =
    "Tools are disabled for this reply. Do not call a tool; answer in plain text using the conversation so far.";
// True for tool_choice "none" and {"type":"none"}.
[[nodiscard]] bool tool_choice_is_none(const Json& body);
// The messages with `hint` added at the end, when the history holds a tool call or a tool result and ends with a
// user or tool message: appended to a closing user message, or added as a user message after a closing tool
// result. nullopt: nothing to add (the messages render as they are).
[[nodiscard]] std::optional<Json> messages_with_no_tools_hint(const Json& messages, std::string_view hint = no_tools_hint);
// Withholds every tool call of a parsed turn: parsed calls are dropped, an unparsed frame is cut from `marker` on,
// and a one-line notice (also left in `withheld`) is appended to the content so the reply is never empty.
void withhold_tool_calls(AssistantOutput& output, std::string_view marker);

// Python str.strip() whitespace.
[[nodiscard]] std::string strip(const std::string& text);

}  // namespace strix::serve
