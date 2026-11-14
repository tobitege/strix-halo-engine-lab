#include "strix/serve/chat_template.hpp"

#include <cstdint>

namespace strix::serve {
namespace {

// Python str.isspace() for one code point.
bool python_space(std::uint32_t cp) {
    return (cp >= 0x09U && cp <= 0x0DU) || (cp >= 0x1CU && cp <= 0x20U) || cp == 0x85U || cp == 0xA0U || cp == 0x1680U ||
        (cp >= 0x2000U && cp <= 0x200AU) || cp == 0x2028U || cp == 0x2029U || cp == 0x202FU || cp == 0x205FU || cp == 0x3000U;
}
// Code point starting at text[i] and its length (invalid bytes count as one non-space unit).
std::pair<std::uint32_t, std::size_t> code_point(const std::string& text, std::size_t i) {
    const auto b0 = static_cast<unsigned char>(text[i]);
    if (b0 < 0x80U) return {b0, 1U};
    const std::size_t n = b0 >= 0xF0U ? 4U : b0 >= 0xE0U ? 3U : b0 >= 0xC0U ? 2U : 1U;
    if (n == 1U || i + n > text.size()) return {0xFFFDU, 1U};
    std::uint32_t cp = n == 2U ? (b0 & 0x1FU) : n == 3U ? (b0 & 0x0FU) : (b0 & 0x07U);
    for (std::size_t k = 1; k < n; ++k) cp = (cp << 6U) | (static_cast<unsigned char>(text[i + k]) & 0x3FU);
    return {cp, n};
}

[[noreturn]] void raise(const std::string& message) { throw ChatTemplateError(message); }

bool is_string(const Json* value) { return value && value->is_string(); }
bool defined(const Json* value) { return value != nullptr; }

std::string render_content(const Json* content, bool is_system) {
    if (!content || content->is_null()) return {};
    if (content->is_string()) return content->as_string();
    if (content->is_array()) {
        std::string out;
        for (const auto& item : content->as_array()) {
            const auto* type = item.find("type");
            const bool image = item.find("image") || item.find("image_url") || (type && type->is_string() && type->as_string() == "image");
            const bool video = !image && (item.find("video") || (type && type->is_string() && type->as_string() == "video"));
            if (image) raise(is_system ? "System message cannot contain images." : "Image input is not supported by this engine.");
            if (video) raise(is_system ? "System message cannot contain videos." : "Video input is not supported by this engine.");
            const auto* text = item.find("text");
            if (!text) raise("Unexpected item type in content.");
            if (text->is_string()) out += text->as_string();
            else if (!text->is_null()) out += text->dump_spaced();
        }
        return out;
    }
    raise("Unexpected content type.");
}

std::string role_of(const Json& message) {
    const auto* role = message.find("role");
    if (!role || !role->is_string()) raise("Every message needs a string role.");
    return role->as_string();
}

// The arguments object of a tool call: an object as given, or a JSON string parsed into one.
Json call_arguments(const Json& call, const std::string& name) {
    const auto* arguments = call.find("arguments");
    if (!arguments || arguments->is_null()) return Json();
    if (arguments->is_object()) return *arguments;
    if (arguments->is_string()) {
        if (strip(arguments->as_string()).empty()) return Json::object();
        Json parsed;
        try { parsed = parse_request_json(arguments->as_string()); }
        catch (const JsonError&) { raise("Tool call arguments for function \"" + name + "\" are not valid JSON."); }
        if (!parsed.is_object()) raise("Tool call arguments for function \"" + name + "\" must be a JSON object.");
        return parsed;
    }
    raise("Tool call arguments for function \"" + name + "\" must be an object/mapping or a JSON string.");
}

}  // namespace

std::string strip(const std::string& text) {
    std::size_t begin = 0, end = text.size();
    while (begin < end) {
        const auto [cp, n] = code_point(text, begin);
        if (!python_space(cp)) break;
        begin += n;
    }
    while (end > begin) {
        std::size_t start = end - 1U;
        while (start > begin && (static_cast<unsigned char>(text[start]) & 0xC0U) == 0x80U) --start;
        const auto [cp, n] = code_point(text, start);
        if (start + n != end || !python_space(cp)) break;
        end = start;
    }
    return text.substr(begin, end - begin);
}

std::string render_chat_template(const ChatTemplateInput& input) {
    if (!input.messages.is_array() || input.messages.size() == 0U) raise("No messages provided.");
    const auto& messages = input.messages.as_array();
    // Leading system/developer messages are merged into one system block.
    std::size_t num_sys = 0;
    std::string merged_system;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto role = role_of(messages[i]);
        if (num_sys == i && (role == "system" || role == "developer")) {
            const auto text = strip(render_content(messages[i].find("content"), true));
            if (!text.empty()) merged_system += (merged_system.empty() ? "" : "\n") + text;
            ++num_sys;
        }
    }
    std::string reasoning_instructions;
    if (!input.enable_thinking || *input.enable_thinking) {
        auto effort = input.reasoning_effort.value_or("xhigh");
        if (effort == "high") effort = "xhigh";
        if (effort != "xhigh" && effort != "medium" && effort != "low")
            raise("Unexpected reasoning effort " + effort + ". Supported types are xhigh (default), medium, and low.");
        if (effort == "xhigh")
            reasoning_instructions = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
                "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
        else if (effort == "low")
            reasoning_instructions = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
                "without unnecessary elaboration.";
    }
    std::string out;
    if (input.tools.is_array() && input.tools.size() > 0U) {
        out += "<|im_start|>system\n";
        if (!reasoning_instructions.empty()) out += reasoning_instructions + "\n\n";
        out += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const auto& tool : input.tools.as_array()) { out += "\n"; out += tool.dump_spaced(); }
        out += "\n</tools>";
        out += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
               "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
               "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
               "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
               "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
               "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
               "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
               "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
        if (!merged_system.empty()) out += "\n\n" + merged_system;
        out += "<|im_end|>\n";
    } else if (!merged_system.empty()) {
        out += "<|im_start|>system\n" + (reasoning_instructions.empty() ? std::string{} : reasoning_instructions + "\n\n") +
            merged_system + "<|im_end|>\n";
    } else if (!reasoning_instructions.empty()) {
        out += "<|im_start|>system\n" + reasoning_instructions + "<|im_end|>\n";
    }
    // The last user message that is not only a tool response.
    std::size_t last_query = messages.size() - 1U;
    for (std::size_t k = messages.size(); k-- > 0;) {
        if (role_of(messages[k]) != "user") continue;
        const auto content = strip(render_content(messages[k].find("content"), false));
        const bool tool_response = content.starts_with("<tool_response>") && content.ends_with("</tool_response>");
        if (!tool_response) { last_query = k; break; }
    }
    for (std::size_t i = num_sys; i < messages.size(); ++i) {
        const auto& message = messages[i];
        const auto role = role_of(message);
        const auto content = strip(render_content(message.find("content"), false));
        if (role == "system" || role == "developer") raise("System message must be at the beginning.");
        if (role == "user") {
            out += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (role == "assistant") {
            const auto* reasoning_value = message.find("reasoning_content");
            const auto reasoning = strip(is_string(reasoning_value) ? reasoning_value->as_string() : std::string{});
            if (!input.preserve_thinking || *input.preserve_thinking || i > last_query)
                out += "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>" + std::string(thinking_answer_separator) + content;
            else out += "<|im_start|>assistant\n" + content;
            const auto* calls = message.find("tool_calls");
            if (calls && calls->is_array()) {
                bool first = true;
                for (const auto& entry : calls->as_array()) {
                    const auto* call = &entry;
                    if (const auto* function = entry.find("function"); defined(function)) call = function;
                    const auto* name = call->find("name");
                    if (!name || name->is_null()) raise("Tool call is missing a function name.");
                    const auto name_text = name->is_string() ? name->as_string() : name->dump_spaced();
                    if (first) out += strip(content).empty() ? "<tool_call>\n<function=" : "\n\n<tool_call>\n<function=";
                    else out += "\n<tool_call>\n<function=";
                    out += name_text + ">\n";
                    first = false;
                    const auto arguments = call_arguments(*call, name_text);
                    if (arguments.is_object())
                        for (const auto& [key, value] : arguments.as_object())
                            out += "<parameter=" + key + ">\n" + (value.is_string() ? value.as_string() : value.dump_spaced()) + "\n</parameter>\n";
                    out += "</function>\n</tool_call>";
                }
            }
            out += "<|im_end|>\n";
        } else if (role == "tool") {
            if (i > 0 && role_of(messages[i - 1U]) != "tool") out += "<|im_start|>user";
            out += "\n<tool_response>\n" + content + "\n</tool_response>";
            const bool last = i + 1U == messages.size();
            if (last || role_of(messages[i + 1U]) != "tool") out += "<|im_end|>\n";
        } else {
            raise("Unexpected message role.");
        }
    }
    if (input.add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        out += input.enable_thinking && !*input.enable_thinking ? "<think>\n\n</think>" + std::string(thinking_answer_separator) : "<think>\n";
    }
    return out;
}

namespace {
const Json* parameter_schema(const Json& tools, const std::string& function, const std::string& parameter) {
    if (!tools.is_array()) return nullptr;
    for (const auto& tool : tools.as_array()) {
        const auto* spec = tool.find("function");
        if (!spec) spec = &tool;
        const auto* name = spec->find("name");
        if (!name || !name->is_string() || name->as_string() != function) continue;
        const auto* parameters = spec->find("parameters");
        if (!parameters) parameters = spec->find("input_schema");
        const auto* properties = parameters ? parameters->find("properties") : nullptr;
        return properties ? properties->find(parameter) : nullptr;
    }
    return nullptr;
}
// A parameter value as the schema types it: strings stay text; other types are parsed as JSON
// when they parse, else kept as text.
Json typed_value(const std::string& text, const Json* schema) {
    const auto* type = schema ? schema->find("type") : nullptr;
    const bool string_type = !type || (type->is_string() && type->as_string() == "string");
    if (string_type) return Json(text);
    try { return parse_json(text); } catch (const JsonError&) { return Json(text); }
}
}  // namespace

AssistantOutput parse_assistant_output(const std::string& text, bool thinking, const Json& tools) {
    AssistantOutput out;
    out.tool_calls = Json::array();  // also when the reply ends inside the thinking block
    std::string rest = text;
    if (thinking) {
        const auto end = rest.find("</think>");
        if (end == std::string::npos) { out.reasoning = strip(rest); return out; }
        out.reasoning = strip(rest.substr(0, end));
        out.reasoning_complete = true;
        rest = rest.substr(end + 8U);
    }
    const auto first_call = rest.find("<tool_call>");
    out.content = strip(first_call == std::string::npos ? rest : rest.substr(0, first_call));
    for (auto at = first_call; at != std::string::npos; at = rest.find("<tool_call>", at)) {
        const auto close = rest.find("</tool_call>", at);
        const auto block = rest.substr(at + 11U, close == std::string::npos ? std::string::npos : close - at - 11U);
        at = close == std::string::npos ? rest.size() : close + 12U;
        if (close == std::string::npos) out.unfinished_call = true;
        const auto function = block.find("<function=");
        if (function == std::string::npos) continue;
        const auto name_end = block.find('>', function);
        if (name_end == std::string::npos) continue;
        const auto name = strip(block.substr(function + 10U, name_end - function - 10U));
        Json arguments = Json::object();
        for (auto p = block.find("<parameter=", name_end); p != std::string::npos; p = block.find("<parameter=", p)) {
            const auto key_end = block.find('>', p);
            if (key_end == std::string::npos) break;
            const auto key = strip(block.substr(p + 11U, key_end - p - 11U));
            auto value_begin = key_end + 1U;
            if (value_begin < block.size() && block[value_begin] == '\n') ++value_begin;
            auto value_end = block.find("</parameter>", value_begin);
            const auto next = value_end == std::string::npos ? block.size() : value_end;
            auto value = block.substr(value_begin, next - value_begin);
            if (!value.empty() && value.back() == '\n') value.pop_back();
            arguments.set(key, typed_value(value, parameter_schema(tools, name, key)));
            p = value_end == std::string::npos ? block.size() : value_end + 12U;
        }
        Json call = Json::object();
        call.set("name", Json(name));
        call.set("arguments", std::move(arguments));
        out.tool_calls.push(std::move(call));
        out.call_complete.push_back(close != std::string::npos);
    }
    return out;
}

bool tool_choice_is_none(const Json& body) {
    const auto* choice = body.find("tool_choice");
    const auto* type = choice && choice->is_object() ? choice->find("type") : choice;
    return type && type->is_string() && type->as_string() == "none";
}

std::optional<Json> messages_with_no_tools_hint(const Json& messages, std::string_view hint) {
    if (!messages.is_array() || messages.size() == 0U || hint.empty()) return std::nullopt;
    const auto role_of = [](const Json& message) -> std::string {
        const auto* role = message.is_object() ? message.find("role") : nullptr;
        return role && role->is_string() ? role->as_string() : std::string{};
    };
    bool tool_history = false;
    for (const auto& message : messages.as_array()) {
        if (role_of(message) == "tool") tool_history = true;
        if (const auto* calls = message.is_object() ? message.find("tool_calls") : nullptr; calls && calls->is_array() && calls->size() > 0U)
            tool_history = true;
    }
    if (!tool_history) return std::nullopt;
    const auto& last = messages.as_array().back();
    const auto role = role_of(last);
    if (role != "user" && role != "tool") return std::nullopt;
    Json out = Json::array();
    for (std::size_t i = 0; i + 1U < messages.size(); ++i) out.push(messages.as_array()[i]);
    Json closing = last;
    const auto* content = last.find("content");
    if (role == "user" && content && content->is_string()) {
        closing.set("content", Json(content->as_string() + (content->as_string().empty() ? "" : "\n\n") + std::string(hint)));
        out.push(std::move(closing));
    } else if (role == "user" && content && content->is_array()) {
        Json parts = *content;
        Json part = Json::object(); part.set("type", Json("text")); part.set("text", Json("\n\n" + std::string(hint)));
        parts.push(std::move(part));
        closing.set("content", std::move(parts));
        out.push(std::move(closing));
    } else {
        out.push(std::move(closing));
        Json added = Json::object(); added.set("role", Json("user")); added.set("content", Json(std::string(hint)));
        out.push(std::move(added));
    }
    return out;
}

void withhold_tool_calls(AssistantOutput& output, std::string_view marker) {
    bool withheld = output.unfinished_call || (output.tool_calls.is_array() && output.tool_calls.size() > 0U);
    std::string names;
    if (output.tool_calls.is_array()) for (const auto& call : output.tool_calls.as_array())
        if (const auto* name = call.is_object() ? call.find("name") : nullptr; name && name->is_string() && !name->as_string().empty())
            names += (names.empty() ? "" : ", ") + name->as_string();
    if (!marker.empty()) if (const auto at = output.content.find(marker); at != std::string::npos) {
        output.content.erase(at);
        withheld = true;
    }
    output.tool_calls = Json::array();
    output.call_complete.clear();
    output.unfinished_call = false;
    output.withheld.clear();
    if (!withheld) return;
    const auto last = output.content.find_last_not_of(" \t\r\n");
    output.content.erase(last == std::string::npos ? 0U : last + 1U);
    output.withheld = std::string(output.content.empty() ? "" : "\n\n") + "[tool call" + (names.empty() ? "" : " to " + names) +
        " withheld: tool_choice is \"none\"]";
    output.content += output.withheld;
}

}  // namespace strix::serve
