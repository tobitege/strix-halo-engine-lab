// tool_choice "none" after a history with tool calls: every family renderer keeps the history renderable without
// the tool definitions, the closing instruction lands after the last tool result, and a call the model still emits
// is withheld from the reply. Built once for the shared renderers (Qwen, GLM, DeepSeek) and once for MiniMax.
#include "strix/serve/chat_template.hpp"
#ifdef STRIX_TOOLNONE_MINIMAX
#include "minimax/serve/minimax_chat.hpp"
#else
#include "strix/serve/deepseek_chat.hpp"
#include "strix/serve/glm_chat.hpp"
#endif

#include <functional>
#include <iostream>
#include <string>

using namespace strix::serve;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::cerr << "FAIL: " << what << "\n"; }
}
bool has(const std::string& text, std::string_view part) { return text.find(part) != std::string::npos; }

const char* kHistory = R"([
  {"role":"system","content":"You are helpful."},
  {"role":"user","content":"Weather in Oslo?"},
  {"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Oslo\"}"}}]},
  {"role":"tool","tool_call_id":"call_1","content":"RESULT-SUNNY-17C"}
])";
const char* kTools = R"([{"type":"function","function":{"name":"get_weather","description":"TOOLDESC-XYZ",
  "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}])";

using Render = std::function<std::string(const Json& messages, const Json* tools)>;

void replay(const std::string& family, const Render& render) {
    const auto history = parse_json(kHistory);
    const auto tools = parse_json(kTools);
    std::string with_tools, base, hinted_text;
    try { with_tools = render(history, &tools); } catch (const std::exception& e) { check(false, family + ": history with tools renders: " + e.what()); }
    try { base = render(history, nullptr); } catch (const std::exception& e) { check(false, family + ": history without tools renders: " + e.what()); }
    check(has(with_tools, "TOOLDESC-XYZ"), family + ": the definitions are in the prompt when tools are offered");
    check(!has(base, "TOOLDESC-XYZ"), family + ": no definitions without tools");
    check(has(base, "RESULT-SUNNY-17C") && has(base, "get_weather") && has(base, "Oslo"), family + ": earlier call and result stay in the prompt");

    // closing tool result: the instruction is a user message after it
    const auto hinted = messages_with_no_tools_hint(history);
    check(hinted.has_value() && hinted->size() == history.size() + 1U, family + ": instruction added after a closing tool result");
    if (hinted) {
        try { hinted_text = render(*hinted, nullptr); } catch (const std::exception& e) { check(false, family + ": hinted history renders: " + e.what()); }
        const auto result = hinted_text.find("RESULT-SUNNY-17C"), hint = hinted_text.find(no_tools_hint);
        check(result != std::string::npos && hint != std::string::npos && hint > result, family + ": instruction follows the tool result");
        check(!has(hinted_text, "TOOLDESC-XYZ") && has(hinted_text, "get_weather"), family + ": hinted prompt keeps the call, not the definitions");
        std::size_t common = 0;
        while (common < base.size() && common < hinted_text.size() && base[common] == hinted_text[common]) ++common;
        check(common > base.find("RESULT-SUNNY-17C"), family + ": the history up to the tool result renders the same bytes");
    }

    // closing user message: the instruction is appended to it
    auto longer = history;
    longer.push(parse_json(R"({"role":"assistant","content":"It is sunny, 17 C."})"));
    longer.push(parse_json(R"({"role":"user","content":"And tomorrow?"})"));
    const auto appended = messages_with_no_tools_hint(longer);
    check(appended.has_value() && appended->size() == longer.size(), family + ": instruction appended to a closing user message");
    if (appended) {
        std::string text;
        try { text = render(*appended, nullptr); } catch (const std::exception& e) { check(false, family + ": appended history renders: " + e.what()); }
        const auto question = text.find("And tomorrow?"), hint = text.find(no_tools_hint);
        check(question != std::string::npos && hint != std::string::npos && hint > question && has(text, "RESULT-SUNNY-17C"),
            family + ": instruction follows the last user text");
    }
}

// A parsed (or unparsed) call is withheld: no call, no frame text, a notice instead.
void withheld(const std::string& family, AssistantOutput out, std::string_view marker, bool prose, bool named) {
    withhold_tool_calls(out, marker);
    check(out.tool_calls.is_array() && out.tool_calls.size() == 0U && !out.unfinished_call && out.call_complete.empty(), family + ": no tool call remains");
    check(!has(out.content, marker), family + ": no call frame in the content [" + out.content + "]");
    check(!out.withheld.empty() && out.content.ends_with(out.withheld) && has(out.withheld, "tool_choice is \"none\""), family + ": notice appended");
    check(prose == has(out.content, "Checking."), family + ": prose before the call is kept [" + out.content + "]");
    if (named) check(has(out.withheld, "get_weather"), family + ": notice names the withheld tool [" + out.withheld + "]");
}
}  // namespace

int main() try {
    // request forms
    check(tool_choice_is_none(parse_json(R"({"tool_choice":"none"})")) && tool_choice_is_none(parse_json(R"({"tool_choice":{"type":"none"}})")),
        "tool_choice none forms");
    check(!tool_choice_is_none(parse_json(R"({"tool_choice":"auto"})")) && !tool_choice_is_none(parse_json(R"({"messages":[]})")) &&
              !tool_choice_is_none(parse_json(R"({"tool_choice":{"type":"function","function":{"name":"none"}}})")), "other tool_choice forms");
    // nothing is added without tool activity, after a closing assistant message, or with an empty instruction
    check(!messages_with_no_tools_hint(parse_json(R"([{"role":"system","content":"s"},{"role":"user","content":"hi"}])")), "no tool history: unchanged");
    {
        auto prefill = parse_json(kHistory);
        prefill.push(parse_json(R"({"role":"assistant","content":"It is"})"));
        check(!messages_with_no_tools_hint(prefill), "closing assistant message: unchanged");
        check(!messages_with_no_tools_hint(parse_json(kHistory), ""), "empty instruction: unchanged");
        auto parts = parse_json(kHistory);
        parts.push(parse_json(R"({"role":"assistant","content":"Sunny."})"));
        parts.push(parse_json(R"({"role":"user","content":[{"type":"text","text":"More?"}]})"));
        const auto out = messages_with_no_tools_hint(parts, "HINT");
        const auto* content = out ? out->as_array().back().find("content") : nullptr;
        check(content && content->is_array() && content->size() == 2U && content->as_array()[1].find("text")->as_string() == "\n\nHINT",
            "closing user message with parts: a text part is added");
    }
    // a turn without a call is untouched
    {
        AssistantOutput plain;
        plain.content = "Sunny, 17 C.";
        plain.tool_calls = Json::array();
        withhold_tool_calls(plain, "<tool_call>");
        check(plain.content == "Sunny, 17 C." && plain.withheld.empty(), "plain answer unchanged");
    }
#ifdef STRIX_TOOLNONE_MINIMAX
    namespace mm = strix::minimax::serve;
    const std::string minimax_tool_marker = "]<]minimax[>[<tool_call>";  // the block opening the template writes
    replay("minimax", [](const Json& messages, const Json* tools) { return mm::render_minimax_chat(messages, tools); });
    // its output parser leaves a call block in the text: the block is cut at the family marker
    AssistantOutput raw;
    raw.tool_calls = Json::array();
    raw.content = std::string("Checking.\n") + minimax_tool_marker + "\n<invoke name=\"get_weather\">\n<parameter name=\"city\">Oslo</parameter>\n</invoke>\n";
    withheld("minimax", raw, minimax_tool_marker, true, false);
    AssistantOutput bare;
    bare.tool_calls = Json::array();
    bare.content = std::string(minimax_tool_marker) + "\n<invoke name=\"get_weather\">";
    withheld("minimax bare", bare, minimax_tool_marker, false, false);
#else
    const auto input_of = [](const Json& messages, const Json* tools) {
        ChatTemplateInput input;
        input.messages = messages;
        if (tools) input.tools = *tools;
        input.enable_thinking = false;
        return input;
    };
    replay("qwen", [&](const Json& m, const Json* t) { return render_chat_template(input_of(m, t)); });
    replay("glm", [&](const Json& m, const Json* t) { return render_glm_chat(input_of(m, t)); });
    replay("deepseek", [&](const Json& m, const Json* t) { return render_deepseek_v41_chat(input_of(m, t)); });

    const std::string qwen_call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>";
    withheld("qwen", parse_assistant_output("Checking.\n\n" + qwen_call, false, Json()), "<tool_call>", true, true);
    withheld("qwen bare", parse_assistant_output(qwen_call, false, Json()), "<tool_call>", false, true);
    withheld("qwen cut", parse_assistant_output("Checking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOs", false, Json()), "<tool_call>", true, false);
    const std::string glm_call = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>";
    withheld("glm", parse_glm_output("ok</think>Checking.\n" + glm_call, false, Json()), glm_tool_marker, true, true);
    withheld("glm bare", parse_glm_output("ok</think>" + glm_call, false, Json()), glm_tool_marker, false, true);
    const std::string d = "\xEF\xBD\x9C" "DSML\xEF\xBD\x9C";
    const std::string ds_call = "<" + d + " calls>\n<" + d + " invoke name=\"get_weather\">\n<" + d + " parameter name=\"city\" string=\"true\">Oslo</" + d +
        " parameter>\n</" + d + " invoke>\n</" + d + " calls>";
    withheld("deepseek", parse_deepseek_v41_output("Checking.\n\n" + ds_call, false, Json()), deepseek_v41_tool_marker, true, false);
    withheld("deepseek bare", parse_deepseek_v41_output(ds_call, false, Json()), deepseek_v41_tool_marker, false, false);
    withheld("deepseek cut", parse_deepseek_v41_output("Checking.\n\n<" + d + " calls>\n<" + d + " invoke name=\"get_w", false, Json()), deepseek_v41_tool_marker, true, false);
#endif
    if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
    std::cout << "tool_choice none: ok\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "exception: " << e.what() << "\n";
    return 1;
}
