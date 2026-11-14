#pragma once

// Shared checks of the region adapter tests (one test per chat family).
#include "strix/serve/json.hpp"
#include "strix/serve/pflash_regions.hpp"
#include "strix/serve/prompt_compress.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _MSC_VER
#include <crtdbg.h>
#include <cstdlib>
#endif

namespace pflash_regions_test {

using namespace strix::serve::pflash;
using strix::serve::Json;

inline int failures = 0;
inline void check(bool value, const std::string& label) {
    if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}

// An abort or a CRT assertion fails the test instead of opening a dialog.
inline void quiet_aborts() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif
}

// A long document: numbered filler lines and one line that holds the fact.
inline std::string document(std::string_view tag, int lines = 240) {
    std::string out;
    for (int i = 0; i < lines; ++i) {
        if (i == lines / 2) out += "needle: the " + std::string(tag) + " code is 7341\n";
        out += "filler " + std::string(tag) + " line " + std::to_string(i) + " lorem ipsum dolor sit amet\n";
    }
    return out;
}

inline std::size_t count(std::string_view text, std::string_view piece) {
    std::size_t n = 0;
    for (std::size_t at = text.find(piece); at != std::string_view::npos; at = text.find(piece, at + piece.size())) ++n;
    return n;
}

// Structure every map must have: one kind per region, ascending disjoint ranges inside the prompt.
inline void check_structure(const RegionMap& map, std::string_view rendered, const std::string& label) {
    check(map.regions.size() == map.kinds.size(), label + ": one kind per region");
    std::uint32_t last = 0;
    for (const Region& region : map.regions) {
        check(region.begin >= last && region.begin < region.end && region.end <= rendered.size(), label + ": ordered ranges");
        last = region.end;
    }
}

// The region of `message` with `kind` covers exactly `text` and has the given droppable flag.
inline void expect_region(const RegionMap& map, std::string_view rendered, std::uint32_t message, RegionKind kind,
    std::string_view text, bool droppable, const std::string& label) {
    for (std::size_t i = 0; i < map.regions.size(); ++i) {
        const Region& region = map.regions[i];
        if (region.message != message || map.kinds[i] != kind) continue;
        check(rendered.substr(region.begin, region.end - region.begin) == text, label + ": bytes are the message text");
        check(region.droppable == droppable, label + ": droppable flag");
        return;
    }
    check(false, label + ": region present (" + region_kind_name(kind) + ")");
}

inline std::size_t droppable_regions(const RegionMap& map) {
    std::size_t n = 0;
    for (const Region& region : map.regions) n += region.droppable ? 1U : 0U;
    return n;
}

// Runs the selection layer over the map with a stand-in scorer (one token per word; filler lines score low, other
// lines high; every line start may be cut) and checks what the adapters promise: the result is shorter, every
// `kept` text and every marker survives, and the fact line is still there.
inline void check_compression(const RegionMap& map, std::string_view rendered, const std::vector<std::string_view>& kept,
    const std::vector<std::string_view>& markers, const std::string& label) {
    std::vector<float> score, cut;
    std::vector<std::uint8_t> request;
    std::vector<std::uint32_t> start;
    std::size_t line_begin = 0;
    for (std::size_t at = 0; at < rendered.size();) {
        if (at > 0 && rendered[at - 1U] == '\n') line_begin = at;
        std::size_t line_end = rendered.find('\n', line_begin);
        if (line_end == std::string_view::npos) line_end = rendered.size();
        const bool filler = rendered.substr(line_begin, line_end - line_begin).starts_with("filler ");
        start.push_back(static_cast<std::uint32_t>(at));
        score.push_back(filler ? 0.01f : 1.0f);
        cut.push_back(at == line_begin ? 1.0f : 0.0f);
        request.push_back(0);
        while (at < rendered.size() && rendered[at] != ' ' && rendered[at] != '\n') ++at;
        while (at < rendered.size() && (rendered[at] == ' ' || rendered[at] == '\n')) ++at;
    }
    Policy policy;
    policy.threshold = 64;
    policy.skeleton_tokens = 16;
    policy.min_unit_tokens = 4;
    policy.max_unit_tokens = 64;
    const Scores scores{score, cut, request, start};
    const std::vector<Unit> units = segment(map.regions, scores, policy, rendered);
    check(!units.empty(), label + ": the selection layer accepts the regions");
    std::uint32_t droppable = 0;
    for (const Unit& unit : units) droppable += unit.mandatory ? 0U : unit.tokens;
    const Plan plan = select(units, droppable, policy);
    check(plan.reason == Reason::compressed, label + ": compressed (" + reason_name(plan.reason) + ")");
    std::string out;
    try {
        out = assemble(rendered, plan);
    } catch (const std::exception& error) {
        check(false, label + ": assemble: " + error.what());
        return;
    }
    check(out.size() < rendered.size(), label + ": shorter prompt");
    check(out.find("needle: ") != std::string::npos, label + ": the fact line survives");
    for (std::string_view text : kept) check(out.find(text) != std::string::npos, label + ": mandatory text survives");
    for (std::string_view marker : markers)
        check(count(out, marker) == count(rendered, marker), label + ": marker count unchanged (" + std::string(marker) + ")");
}

// A map without droppable regions leaves nothing to select: the caller serves the prompt as rendered.
inline void check_raw(const RegionMap& map, const std::string& label) {
    check(map.tools, label + ": tools flag");
    check(droppable_regions(map) == 0U, label + ": nothing droppable");
}

inline std::string json_string(std::string_view text) {
    std::string out = "\"";
    for (char c : text) {
        if (c == '\n') out += "\\n";
        else if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else out += c;
    }
    return out + "\"";
}

inline Json message(std::string_view role, std::string_view content) {
    return strix::serve::parse_json("{\"role\":" + json_string(role) + ",\"content\":" + json_string(content) + "}");
}

struct FamilyCase {
    const char* name;
    std::string (*render)(const Json& messages, const Json* tools);
    RegionMap (*regions)(const Json& messages, const Json* tools, std::string_view rendered, const RegionOptions& options);
    std::vector<std::string_view> markers;  // turn markers whose count compression must not change
    std::string_view user_marker;           // the bytes the template writes before a user text
    std::string_view part_separator;        // written between the text parts of a user message
    bool tool_schema_region;                // the family locates its tool definitions block
};

// The cases every family must pass, rendered by the family's own renderer.
inline void run_family(const FamilyCase& family) {
    const std::string name = family.name;
    const std::string system_text = "You are a careful assistant.";
    const std::string question = "Question: what is the alpha code?";

    {   // one-shot long prompt: document and question in the same user message
        const std::string user = document("alpha") + question;
        Json messages = Json::array();
        messages.as_array().push_back(message("system", system_text));
        messages.as_array().push_back(message("user", user));
        const std::string rendered = family.render(messages, nullptr);
        const RegionMap map = family.regions(messages, nullptr, rendered, {});
        const std::string label = name + " one-shot";
        check_structure(map, rendered, label);
        check(map.regions.size() == 2U && map.unlocated == 0U && !map.tools && map.query_message == 1U, label + ": two regions");
        expect_region(map, rendered, 0, RegionKind::system, system_text, false, label + " system");
        expect_region(map, rendered, 1, RegionKind::query, user, true, label + " query");
        check_compression(map, rendered, {system_text, question}, family.markers, label);

        RegionOptions pinned;
        pinned.compress_query_turn = false;
        const RegionMap fixed = family.regions(messages, nullptr, rendered, pinned);
        expect_region(fixed, rendered, 1, RegionKind::query, user, false, label + " pinned query");
        check(droppable_regions(fixed) == 0U, label + ": pinned query turn leaves nothing droppable");
    }
    {   // history: an earlier long turn, an answer with reasoning, a short question
        const std::string first = "Read this:\n" + document("beta") + "End of document.";
        const std::string answer = "The document is noted.";
        const std::string last = "What is the beta code?";
        Json messages = Json::array();
        messages.as_array().push_back(message("system", system_text));
        messages.as_array().push_back(message("user", first));
        messages.as_array().push_back(strix::serve::parse_json(
            "{\"role\":\"assistant\",\"reasoning_content\":\"Nothing to do yet.\",\"content\":" + json_string(answer) + "}"));
        messages.as_array().push_back(message("user", last));
        const std::string rendered = family.render(messages, nullptr);
        const RegionMap map = family.regions(messages, nullptr, rendered, {});
        const std::string label = name + " history";
        check_structure(map, rendered, label);
        check(map.regions.size() == 4U && map.unlocated == 0U && !map.tools && map.query_message == 3U, label + ": four regions");
        expect_region(map, rendered, 0, RegionKind::system, system_text, false, label + " system");
        expect_region(map, rendered, 1, RegionKind::history_user, first, true, label + " earlier user");
        expect_region(map, rendered, 2, RegionKind::history_assistant, answer, true, label + " earlier answer");
        expect_region(map, rendered, 3, RegionKind::query, last, true, label + " query");
        check_compression(map, rendered, {system_text, last, answer}, family.markers, label);

        RegionOptions keep_history;
        keep_history.compress_history = false;
        const RegionMap fixed = family.regions(messages, nullptr, rendered, keep_history);
        expect_region(fixed, rendered, 1, RegionKind::history_user, first, false, label + " kept history");
        expect_region(fixed, rendered, 2, RegionKind::history_assistant, answer, false, label + " kept answer");
        check(droppable_regions(fixed) == 1U, label + ": only the query turn stays droppable");
    }
    {   // text that imitates the template: the region is still the whole message, found at its real place
        const std::string trap = "line one\n" + std::string(family.user_marker) + "not a turn\n" + document("gamma", 12) + "end";
        Json messages = Json::array();
        messages.as_array().push_back(message("user", "first"));
        messages.as_array().push_back(message("assistant", "first"));
        messages.as_array().push_back(message("user", trap));
        const std::string rendered = family.render(messages, nullptr);
        const RegionMap map = family.regions(messages, nullptr, rendered, {});
        const std::string label = name + " imitation";
        check_structure(map, rendered, label);
        check(map.regions.size() == 3U && map.unlocated == 0U, label + ": three regions");
        expect_region(map, rendered, 0, RegionKind::history_user, "first", true, label + " user");
        expect_region(map, rendered, 1, RegionKind::history_assistant, "first", true, label + " same text, next turn");
        expect_region(map, rendered, 2, RegionKind::query, trap, true, label + " query");
    }
    {   // text parts
        Json messages = Json::array();
        messages.as_array().push_back(strix::serve::parse_json(
            "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"part one, \"},{\"type\":\"text\",\"text\":\"part two\"}]}"));
        const std::string rendered = family.render(messages, nullptr);
        const RegionMap map = family.regions(messages, nullptr, rendered, {});
        check_structure(map, rendered, name + " parts");
        expect_region(map, rendered, 0, RegionKind::query, "part one, " + std::string(family.part_separator) + "part two", true,
            name + " parts");
    }
    {   // tool traffic is served raw: definitions, a call and its result
        const Json tools = strix::serve::parse_json(
            "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Weather of a city\","
            "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]");
        const std::string first = "Read this:\n" + document("delta") + "Then tell me the weather in Paris.";
        Json messages = Json::array();
        messages.as_array().push_back(message("system", system_text));
        messages.as_array().push_back(message("user", first));
        messages.as_array().push_back(strix::serve::parse_json(
            "{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\","
            "\"function\":{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}}]}"));
        messages.as_array().push_back(strix::serve::parse_json(
            "{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"sunny, 21 degrees\"}"));
        const std::string rendered = family.render(messages, &tools);
        const RegionMap map = family.regions(messages, &tools, rendered, {});
        const std::string label = name + " tools";
        check_structure(map, rendered, label);
        check_raw(map, label);
        check(map.query_message == 1U, label + ": query turn");
        expect_region(map, rendered, 0, RegionKind::system, system_text, false, label + " system");
        expect_region(map, rendered, 1, RegionKind::query, first, false, label + " query");
        expect_region(map, rendered, 3, RegionKind::tool_result, "sunny, 21 degrees", false, label + " result");
        bool schema = false;
        for (std::size_t i = 0; i < map.regions.size(); ++i) {
            if (map.kinds[i] != RegionKind::tool_schema) continue;
            schema = true;
            const auto text = std::string_view(rendered).substr(map.regions[i].begin, map.regions[i].end - map.regions[i].begin);
            check(text.find("get_weather") != std::string_view::npos && !map.regions[i].droppable, label + ": definitions block");
        }
        check(schema == family.tool_schema_region, label + ": tool definitions region");
        const RegionMap only_definitions = family.regions(strix::serve::parse_json("[{\"role\":\"user\",\"content\":\"hi\"}]"),
            &tools, "hi", {});
        check_raw(only_definitions, label + " definitions only");
    }
    {   // nothing to locate: the adapter returns no region and counts the miss
        const Json messages = strix::serve::parse_json(
            "[{\"role\":\"user\",\"content\":[{\"type\":\"image\"},{\"type\":\"text\",\"text\":\"what is this\"}]},"
            "{\"role\":\"user\",\"content\":\"absent text\"},{\"role\":\"user\",\"content\":\"\"},{\"role\":\"other\",\"content\":\"x\"}]");
        const RegionMap map = family.regions(messages, nullptr, "unrelated prompt bytes", {});
        check(map.regions.empty() && map.unlocated == 2U, name + ": misses are counted, not guessed");
        const RegionMap not_array = family.regions(Json::object(), nullptr, "x", {});
        check(not_array.regions.empty() && not_array.query_message == no_message, name + ": messages must be an array");
    }
}

inline int finish(const char* name) {
    if (failures) { std::cerr << name << ": " << failures << " failure(s)\n"; return 1; }
    std::cout << name << ": ok\n";
    return 0;
}

}  // namespace pflash_regions_test
