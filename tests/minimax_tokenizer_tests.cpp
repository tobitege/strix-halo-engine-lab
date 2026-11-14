// MiniMax M3.x tokenizer and chat template: the C++ implementation equals HuggingFace `tokenizers` /
// transformers' apply_chat_template on the fixtures written by tools/minimax/make_tokenizer_fixtures.py.
//   strix-minimax-tokenizer-tests src/minimax/serve/fixtures/minimax_tokenizer_fixtures.json [tokenizer.json]
// Without tokenizer.json, memo checks use a synthetic byte vocabulary; model golden IDs are skipped.
#include "minimax/serve/minimax_chat.hpp"
#include "minimax/serve/minimax_tokenizer.hpp"
#include "strix/serve/token_memo.hpp"
#include "strix/serve/json.hpp"

#include <random>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using strix::serve::Json;
namespace mm = strix::minimax::serve;

struct Tally { std::size_t pass = 0, fail = 0; };
std::map<std::string, Tally> tallies;
int printed = 0;

std::string escape(const std::string& s) {
    std::string out;
    strix::serve::append_json_string(out, s);
    return out.size() > 300U ? out.substr(0, 300U) + "..." : out;
}
std::string ids_text(const std::vector<std::uint32_t>& ids) {
    std::ostringstream os;
    os << '[';
    for (std::size_t i = 0; i < ids.size() && i < 40U; ++i) os << (i ? "," : "") << ids[i];
    if (ids.size() > 40U) os << ",...(" << ids.size() << ")";
    os << ']';
    return os.str();
}
void check(const std::string& group, bool ok, const std::string& detail) {
    auto& t = tallies[group];
    if (ok) { ++t.pass; return; }
    ++t.fail;
    if (printed++ < 40) std::cerr << "FAIL [" << group << "] " << detail << '\n';
}
std::vector<std::uint32_t> ids_of(const Json& j) {
    std::vector<std::uint32_t> out;
    for (const auto& v : j.as_array()) out.push_back(static_cast<std::uint32_t>(v.as_int()));
    return out;
}
std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

mm::MinimaxChatOptions options_of(const Json& o) {
    mm::MinimaxChatOptions out;
    for (const auto& [key, value] : o.as_object()) {
        if (key == "add_generation_prompt") out.add_generation_prompt = value.as_bool();
        else if (key == "thinking_mode") out.thinking_mode = value.as_string();
        else if (key == "template_model_id") out.template_model_id = value.as_string();
        else if (key == "reasoning_effort") out.reasoning_effort = value.as_string();
        else if (key == "effort") out.effort = value.as_string();
        else if (key == "model_identity") out.model_identity = value.as_string();
        else throw std::runtime_error("unknown chat option " + key);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: strix-minimax-tokenizer-tests FIXTURES.json [tokenizer.json]\n";
        return 2;
    }
    const auto fixtures = strix::serve::parse_json(read_file(argv[1]));
    std::shared_ptr<const mm::MinimaxTokenizer> tok;
    if (argc == 3 && std::ifstream(argv[2]).good()) {
        const auto t0 = std::chrono::steady_clock::now();
        tok = mm::MinimaxTokenizer::load(argv[2]);
        std::cout << "loaded " << argv[2] << " in "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s, vocab_size " << tok->vocab_size()
                  << ", bos " << tok->bos_id() << ", eos " << tok->eos_id() << '\n';
        check("meta", tok->vocab_size() == static_cast<std::size_t>(fixtures.find("vocab_size")->as_int()), "vocab_size");
        check("meta", tok->bos_id() == static_cast<std::uint32_t>(fixtures.find("bos")->as_int()), "bos");
        check("meta", tok->eos_id() == static_cast<std::uint32_t>(fixtures.find("eos")->as_int()), "eos");
        check("meta", tok->token_id("<mm:think>") == 200059U && tok->token_id("Ġthe") != UINT32_MAX && tok->token_id("nope nope") == UINT32_MAX,
              "token_id");
        check("meta", tok->is_special(200019U) && !tok->is_special(200059U) && tok->is_added(200059U) && !tok->is_added(100U), "special flags");
    } else {
        std::cout << "SKIP model tokenizer checks (no tokenizer.json given or found); synthetic memo checks run\n";
    }
    std::cout << "unicode: regex classes " << mm::MinimaxTokenizer::class_unicode_version() << ", NFC "
              << mm::MinimaxTokenizer::nfc_unicode_version() << '\n';

    // NFC
    for (const auto& c : fixtures.find("nfc")->as_array()) {
        const auto& text = c.find("text")->as_string();
        const auto got = mm::MinimaxTokenizer::nfc(text);
        check("nfc", got == c.find("nfc")->as_string(), escape(text) + " -> " + escape(got) + " want " + escape(c.find("nfc")->as_string()));
    }
    // Restart points and the prompt memo: random texts with added tokens, pieces of them, combining marks and split
    // UTF-8, cut at any byte and continued differently, always give the ids of a full encode.
    {
        const auto memo_tok = tok ? tok : mm::MinimaxTokenizer::load(STRIX_MINIMAX_MEMO_FIXTURE);
        static const std::vector<std::string> parts = {"<mm:think>", "</mm:think>", "<mm:", "<", ">", "]<]minimax[>[", "]<]",
            "e\xCC\x81", "\xCC\x81", "caf\xC3\xA9", "\xC3", "\xE6\x97\xA5", "\xF0\x9F",
            " the", " and", "\n\n", " ", "12", "{\"a\":1}", "x"};
        std::mt19937_64 rng(20261114ULL);
        strix::serve::TokenMemo memo(strix::serve::TokenMemo::Limits{std::size_t{1} << 20, 6});
        const std::size_t lookahead = memo_tok->mark_lookahead();
        std::vector<std::string> kept;
        bool split = true, same = true;
        for (int round = 0; round < 3000; ++round) {
            std::string text = kept.empty() || rng() % 5U == 0U ? std::string() : kept[rng() % kept.size()];
            text.resize(rng() % (text.size() + 1U));
            for (std::size_t n = rng() % 40U; n > 0U; --n) text += parts[rng() % parts.size()];
            std::vector<strix::serve::EncodeMark> marks;
            const auto ids = memo_tok->encode_marked(text, true, marks);
            same = same && ids == memo_tok->encode(text, true);
            for (const auto& mark : marks) {
                auto joined = memo_tok->encode(std::string_view(text).substr(0, mark.bytes), true);
                const auto tail = memo_tok->encode(std::string_view(text).substr(mark.bytes), true);
                joined.insert(joined.end(), tail.begin(), tail.end());
                split = split && joined == ids;
            }
            same = same && ids == memo.encode(1U, lookahead, text, [&](std::string_view rest, std::vector<strix::serve::EncodeMark>& out) {
                return memo_tok->encode_marked(rest, true, out); });
            if (kept.size() < 16U) kept.push_back(text); else kept[rng() % kept.size()] = text;
        }
        check("memo", split, "a restart point does not split the encoding");
        check("memo", same, "memo ids differ from a full encode");
        check("memo", memo.stats().partial > 300U, "the memo reused too few prefixes");
    }
    // Invalid UTF-8 survives encode/decode unchanged.
    if (tok) {
        static constexpr char kRaw[] = "ok \xff\xfe bad \xe2\x82 tail \xc3";
        const std::string raw(kRaw, sizeof(kRaw) - 1U);
        const auto ids = tok->encode(raw);
        // decode applies from_utf8_lossy, so compare the byte tokens instead
        std::string bytes;
        for (const auto id : ids) bytes += tok->token_bytes(id);
        check("invalid-utf8", bytes == raw, "raw bytes round trip");
    }

    // encode / decode
    if (tok) {
        std::size_t total_tokens = 0;
        double seconds = 0;
        for (const auto& c : fixtures.find("encode")->as_array()) {
            const auto& text = c.find("text")->as_string();
            const auto want = ids_of(*c.find("ids"));
            const auto t0 = std::chrono::steady_clock::now();
            const auto got = tok->encode(text, true);
            seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            total_tokens += got.size();
            check("encode", got == want, escape(text) + " -> " + ids_text(got) + " want " + ids_text(want));
            const auto want_plain = ids_of(*c.find("ids_plain"));
            const auto got_plain = tok->encode(text, false);
            check("encode-plain", got_plain == want_plain, escape(text) + " -> " + ids_text(got_plain) + " want " + ids_text(want_plain));
            const auto dec = tok->decode(want, false);
            check("decode", dec == c.find("decoded")->as_string(), ids_text(want) + " -> " + escape(dec));
            const auto dec_skip = tok->decode(want, true);
            check("decode-skip", dec_skip == c.find("decoded_skip")->as_string(), ids_text(want) + " -> " + escape(dec_skip));
        }
        std::cout << "encode speed: " << total_tokens << " tokens in " << seconds << " s\n";
        for (const auto& c : fixtures.find("decode")->as_array()) {
            const auto ids = ids_of(*c.find("ids"));
            const auto dec = tok->decode(ids, false);
            check("decode-ids", dec == c.find("text")->as_string(), ids_text(ids) + " -> " + escape(dec) + " want " + escape(c.find("text")->as_string()));
            const auto dec_skip = tok->decode(ids, true);
            check("decode-ids-skip", dec_skip == c.find("text_skip")->as_string(), ids_text(ids) + " -> " + escape(dec_skip));
        }
    }

    // chat template
    for (const auto& c : fixtures.find("chat")->as_array()) {
        const auto& name = c.find("name")->as_string();
        const auto messages = strix::serve::parse_json(c.find("messages_json")->as_string());
        Json tools;
        const Json* tools_ptr = nullptr;
        if (const auto* t = c.find("tools_json"); t && t->is_string()) {
            tools = strix::serve::parse_json(t->as_string());
            tools_ptr = &tools;
        }
        std::string text;
        try {
            text = mm::render_minimax_chat(messages, tools_ptr, options_of(*c.find("options")));
        } catch (const std::exception& e) {
            check("chat-text", false, name + ": threw " + e.what());
            continue;
        }
        const auto& want = c.find("text")->as_string();
        if (text != want) {
            std::size_t at = 0;
            while (at < text.size() && at < want.size() && text[at] == want[at]) ++at;
            check("chat-text", false, name + ": differs at byte " + std::to_string(at) + ": got " +
                  escape(text.substr(at > 40U ? at - 40U : 0U, 120U)) + " want " + escape(want.substr(at > 40U ? at - 40U : 0U, 120U)));
        } else {
            check("chat-text", true, name);
        }
        if (tok) {
            const auto ids = tok->encode(text, true);
            check("chat-ids", ids == ids_of(*c.find("ids")), name + ": ids " + ids_text(ids));
        }
    }
    // the template refuses a tool message without a preceding tool call
    try {
        (void)mm::render_minimax_chat(strix::serve::parse_json(R"([{"role":"tool","content":"x"}])"));
        check("chat-errors", false, "tool message without a tool call accepted");
    } catch (const mm::MinimaxChatError&) {
        check("chat-errors", true, "");
    }
    // extension: OpenAI string arguments render like the parsed object
    {
        const auto a = mm::render_minimax_chat(strix::serve::parse_json(
            R"([{"role":"user","content":"x"},{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"f","arguments":"{\"a\": 1.0, \"b\": [true, null]}"}}]}])"));
        const auto b = mm::render_minimax_chat(strix::serve::parse_json(
            R"([{"role":"user","content":"x"},{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"f","arguments":{"a":1.0,"b":[true,null]}}}]}])"));
        check("chat-errors", a == b && a.find("<a>1.0") != std::string::npos, "string arguments");
    }
    check("python-format", mm::python_json_dumps(strix::serve::parse_json(R"({"a":1e5,"b":-0.0,"c":1E-5,"d":123456789012345678,"e":0.0001,"f":1e16,"g":1e15,"h":-0})")) ==
              R"({"a": 100000.0, "b": -0.0, "c": 1e-05, "d": 123456789012345678, "e": 0.0001, "f": 1e+16, "g": 1000000000000000.0, "h": 0})",
          "python_json_dumps numbers");

    // oracle: decoded text of a recorded generation re-encodes to the HF ids
    if (tok) {
        if (const auto* o = fixtures.find("oracle")) {
            const auto& text = o->find("text")->as_string();
            const auto want = ids_of(*o->find("reencoded_ids"));
            const auto source = ids_of(*o->find("ids"));
            const auto got = tok->encode(text, true);
            check("oracle", got == want, "re-encoding differs from HF");
            check("oracle", tok->decode(source, false) == text, "decode of the oracle ids differs from HF");
            std::cout << "oracle " << o->find("source")->as_string() << ": " << source.size() << " ids, HF re-encoding reproduces them: "
                      << (o->find("reproduces")->as_bool() ? "yes" : "no") << ", C++ re-encoding equals HF: " << (got == want ? "yes" : "no")
                      << ", equals the oracle ids: " << (got == source ? "yes" : "no") << '\n';
        }
    }

    std::size_t pass = 0, fail = 0;
    for (const auto& [group, t] : tallies) {
        std::cout << "  " << group << ": " << t.pass << " pass, " << t.fail << " fail\n";
        pass += t.pass;
        fail += t.fail;
    }
    std::cout << (fail ? "FAILED" : "OK") << ": " << pass << " pass, " << fail << " fail\n";
    return fail ? 1 : 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
}
