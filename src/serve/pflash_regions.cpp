#include "strix/serve/pflash_regions.hpp"

#include "strix/serve/chat_template.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

namespace strix::serve::pflash {

const char* region_kind_name(RegionKind kind) {
    switch (kind) {
    case RegionKind::system: return "system";
    case RegionKind::tool_schema: return "tool_schema";
    case RegionKind::history_user: return "history_user";
    case RegionKind::history_assistant: return "history_assistant";
    case RegionKind::tool_result: return "tool_result";
    case RegionKind::query: return "query";
    case RegionKind::tail: return "tail";
    }
    return "unknown";
}

namespace detail {
namespace {

std::string role_of(const Json& message) {
    if (!message.is_object()) return {};
    const Json* role = message.find("role");
    return role && role->is_string() ? role->as_string() : std::string{};
}

bool non_empty(const Json* value) {
    if (!value || value->is_null()) return false;
    if (value->is_array()) return !value->as_array().empty();
    if (value->is_string()) return !value->as_string().empty();
    return true;
}

// The text of a message as the templates print it for text content: a string, or the text parts of a list joined
// by `separator`. Anything else (media parts, objects) has no single text to look for.
std::optional<std::string> message_text(const Json& message, std::string_view separator = {}) {
    const Json* content = message.is_object() ? message.find("content") : nullptr;
    if (!content || content->is_null()) return std::string{};
    if (content->is_string()) return content->as_string();
    if (!content->is_array()) return std::nullopt;
    std::string out;
    bool first = true;
    for (const Json& part : content->as_array()) {
        if (!first) out += separator;
        first = false;
        if (part.is_string()) { out += part.as_string(); continue; }
        if (!part.is_object()) return std::nullopt;
        const Json* type = part.find("type");
        const Json* text = part.find("text");
        if (!type || !type->is_string() || type->as_string() != "text" || !text || !text->is_string()) return std::nullopt;
        out += text->as_string();
    }
    return out;
}

bool ends_at(std::string_view rendered, std::size_t position, const std::vector<std::string_view>& marks) {
    if (marks.empty()) return true;
    return std::any_of(marks.begin(), marks.end(), [&](std::string_view mark) {
        return position >= mark.size() && rendered.substr(position - mark.size(), mark.size()) == mark;
    });
}

bool starts_at(std::string_view rendered, std::size_t position, const std::vector<std::string_view>& marks) {
    if (marks.empty() || position == rendered.size()) return true;
    return std::any_of(marks.begin(), marks.end(), [&](std::string_view mark) {
        return rendered.substr(position, mark.size()) == mark;
    });
}

// First occurrence of `text` at or after `from` that sits between the family's markers.
std::optional<std::size_t> locate(std::string_view rendered, std::size_t from, std::string_view text, const Marks& marks) {
    for (std::size_t at = rendered.find(text, from); at != std::string_view::npos; at = rendered.find(text, at + 1U))
        if (ends_at(rendered, at, marks.before) && starts_at(rendered, at + text.size(), marks.after)) return at;
    return std::nullopt;
}

bool is_tool_response_turn(const std::string& text) {
    const std::string body = strip(text);
    return body.starts_with("<tool_response>") && body.ends_with("</tool_response>");
}

}  // namespace

RegionMap map_regions(const Json& messages, const Json* tools, std::string_view rendered, const RegionOptions& options,
    const Family& family) {
    RegionMap map;
    if (!messages.is_array() || rendered.size() >= no_message) return map;
    const auto& list = messages.as_array();

    map.tools = non_empty(tools);
    for (std::size_t i = 0; i < list.size(); ++i) {
        const std::string role = role_of(list[i]);
        if (role == "tool" || (list[i].is_object() && non_empty(list[i].find("tool_calls")))) map.tools = true;
        if (role != "user") continue;
        if (family.query_skips_tool_responses) {
            const auto text = message_text(list[i]);
            if (text && is_tool_response_turn(*text)) { map.tools = true; continue; }
        }
        map.query_message = static_cast<std::uint32_t>(i);
    }

    struct Found { Region region; RegionKind kind; };
    std::vector<Found> found;
    const auto add = [&](std::size_t message, std::size_t begin, std::size_t end, RegionKind kind, bool droppable) {
        found.push_back({Region{static_cast<std::uint32_t>(message), static_cast<std::uint32_t>(begin),
            static_cast<std::uint32_t>(end), droppable && !map.tools}, kind});
    };

    std::size_t cursor = 0;
    if (non_empty(tools) && !family.tools_open.empty()) {
        const std::size_t open = rendered.find(family.tools_open);
        const std::size_t begin = open == std::string_view::npos ? open : open + family.tools_open.size();
        const std::size_t close = open == std::string_view::npos ? open : rendered.find(family.tools_close, begin);
        if (close != std::string_view::npos && close > begin) {
            add(0, begin, close, RegionKind::tool_schema, false);
            if (family.tools_before_system) cursor = close;
        }
    }
    const std::size_t tool_regions = found.size();

    for (std::size_t i = 0; i < list.size(); ++i) {
        const std::string role = role_of(list[i]);
        const bool before_query = map.query_message != no_message && i < map.query_message;
        const Marks* marks = nullptr;
        RegionKind kind = RegionKind::tail;
        bool droppable = false;
        if (role == "system" || role == "developer" || role == "root") {
            marks = &family.system;
            kind = RegionKind::system;
        } else if (role == "user") {
            marks = &family.user;
            if (i == map.query_message) { kind = RegionKind::query; droppable = options.compress_query_turn; }
            else if (before_query) { kind = RegionKind::history_user; droppable = options.compress_history; }
        } else if (role == "assistant") {
            marks = &family.assistant;
            if (before_query) { kind = RegionKind::history_assistant; droppable = options.compress_history; }
        } else if (role == "tool") {
            marks = &family.tool;
            kind = RegionKind::tool_result;
        } else {
            continue;
        }
        auto text = message_text(list[i], role == "user" ? family.user_part_separator : std::string_view{});
        if (!text) { ++map.unlocated; continue; }
        if (marks->strip) *text = strip(*text);
        if (text->empty()) continue;
        const auto at = locate(rendered, cursor, *text, *marks);
        if (!at) { ++map.unlocated; continue; }
        add(i, *at, *at + text->size(), kind, droppable);
        cursor = *at + text->size();
    }

    // Message regions are ascending by construction; the tool definitions block goes where it was rendered. A
    // message region that overlaps it was matched inside the definitions: drop that region (its bytes stay kept).
    if (tool_regions == 1U) {
        const Region block = found.front().region;
        const auto overlaps = [&](const Found& f) { return f.region.begin < block.end && block.begin < f.region.end; };
        const auto first = found.begin() + 1;
        const auto kept = std::remove_if(first, found.end(), overlaps);
        map.unlocated += static_cast<std::uint32_t>(found.end() - kept);
        found.erase(kept, found.end());
        std::stable_sort(found.begin(), found.end(),
            [](const Found& a, const Found& b) { return a.region.begin < b.region.begin; });
    }
    map.regions.reserve(found.size());
    map.kinds.reserve(found.size());
    for (const Found& f : found) {
        map.regions.push_back(f.region);
        map.kinds.push_back(f.kind);
    }
    return map;
}

}  // namespace detail

}  // namespace strix::serve::pflash
