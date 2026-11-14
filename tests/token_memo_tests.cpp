// TokenMemo: the memoised encoding equals a full encode for every text, and a chat turn re-encodes only its tail.
//   strix-token-memo-tests                                   property tests on synthetic tokenizers
//   strix-token-memo-tests --bench <gguf|tokenizer.json> [qwen|deepseek|glm] [rounds]
//                                                            host ms of render + encode per turn, memo off / on
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/chat_template.hpp"
#include "strix/serve/deepseek_chat.hpp"
#include "strix/serve/glm_chat.hpp"
#include "strix/serve/json.hpp"
#include "strix/serve/token_memo.hpp"
#include "strix/serve/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _MSC_VER
#include <crtdbg.h>
#endif

using namespace strix::serve;

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

std::string utf8(std::uint32_t cp) {
    std::string out;
    if (cp < 0x80U) out += static_cast<char>(cp);
    else if (cp < 0x800U) { out += static_cast<char>(0xC0U | (cp >> 6)); out += static_cast<char>(0x80U | (cp & 0x3FU)); }
    else { out += static_cast<char>(0xE0U | (cp >> 12)); out += static_cast<char>(0x80U | ((cp >> 6) & 0x3FU)); out += static_cast<char>(0x80U | (cp & 0x3FU)); }
    return out;
}

// A small byte-level vocabulary with merges, the chat controls of the three templates and added tokens that nest
// ("<a>" inside "<a>b", "<|im" inside "<|im_start|>"): a restart point is only valid with its lookahead window.
std::shared_ptr<const BpeTokenizer> make_tokenizer(const std::string& pre, bool swap_ids = false) {
    std::array<std::string, 256> alphabet{};
    std::uint32_t extra = 0;
    for (std::uint32_t b = 0; b < 256U; ++b) {
        const bool direct = (b >= 0x21U && b <= 0x7EU) || (b >= 0xA1U && b <= 0xACU) || (b >= 0xAEU);
        alphabet[b] = utf8(direct ? b : 256U + extra++);
    }
    const auto symbol = [&](const std::string& raw) {
        std::string s;
        for (const char c : raw) s += alphabet[static_cast<unsigned char>(c)];
        return s;
    };
    std::vector<std::string> tokens(alphabet.begin(), alphabet.end());
    std::vector<std::int32_t> types(256, 1);
    std::vector<std::string> merges;
    const std::vector<std::pair<std::string, std::string>> pairs = {
        {"t", "h"}, {"th", "e"}, {" ", "t"}, {" t", "h"}, {" th", "e"}, {"i", "n"}, {"a", "n"}, {"an", "d"},
        {" ", "a"}, {" a", "nd"}, {"e", "r"}, {"1", "2"}, {"12", "3"}, {"\n", "\n"}, {" ", " "}, {"'", "s"},
        {"\xC3", "\xA9"}, {".", "\n"}, {"l", "l"}, {"o", "r"}, {" ", "w"}, {" w", "or"}, {"H", "e"}, {"He", "ll"},
        {"\xE6", "\x97"}, {"\xE6\x97", "\xA5"}, {"<", "|"}, {"|", ">"}, {"\"", ":"}, {"{", "\""}};
    for (const auto& [a, b] : pairs) {
        merges.push_back(symbol(a) + " " + symbol(b));
        tokens.push_back(symbol(a + b));
        types.push_back(1);
    }
    for (const char* control : {"<|im_start|>", "<|im_end|>", "<|endoftext|>", "[gMASK]", "<sop>", "<|system|>", "<|user|>",
                                "<|assistant|>", "<|observation|>", "<\xEF\xBD\x9C" "begin\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>",
                                "<\xEF\xBD\x9CUser\xEF\xBD\x9C>", "<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>",
                                "<\xEF\xBD\x9C" "end\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>"}) {
        tokens.emplace_back(control);
        types.push_back(3);
    }
    for (const char* added : {"<think>", "</think>", "<tool_call>", "</tool_call>", "<tools>", "</tools>", "<tool_response>",
                              "</tool_response>", "<a>", "<a>b", "<a>bcd e", "<|im", "[e f]", "<tool call>"}) {
        tokens.emplace_back(added);
        types.push_back(4);
    }
    if (swap_ids) std::swap(tokens[97], tokens[98]);
    return BpeTokenizer::from_parts(std::move(tokens), std::move(types), merges, pre);
}

using Rng = std::mt19937_64;
std::size_t pick(Rng& rng, std::size_t n) { return static_cast<std::size_t>(rng() % n); }

// Text made of words, multi-byte UTF-8, whitespace runs, special-token texts and pieces of them.
std::string random_text(Rng& rng, std::size_t pieces) {
    static const std::vector<std::string> parts = {
        "the", " the", " and", "Hello", " world", " ", "  ", "\n", "\n\n", ".\n", "123", "12", "'s", "in", "er",
        "caf\xC3\xA9", "\xE6\x97\xA5\xE6\x9C\xAC", "\xF0\x9F\x98\x80", "\xC3", "\xA9", "\xE6\x97", "\xF0\x9F",
        "<|im_start|>", "<|im_end|>", "<|im", "<|im_", "<|", "<", ">", "|>", "<think>", "</think>", "<thin", "<a>", "<a>b",
        "<a>bcd e", "<a>bcd", "<a", "b", "cd e", "[e f]", "[e", "[gMASK]", "<sop>", "<|user|>", "<tool_call>", "<tool call>",
        "{\"a\":1}", "\"", "\\n", "\t", "x", "Z", "0"};
    std::string out;
    for (std::size_t i = 0; i < pieces; ++i) out += parts[pick(rng, parts.size())];
    return out;
}

Json tool(Rng& rng, int index) {
    Json props = Json::object();
    const int n = 1 + static_cast<int>(pick(rng, 3));
    for (int i = 0; i < n; ++i) {
        Json p = Json::object();
        p.set("type", i % 2 ? "string" : "number");
        p.set("description", random_text(rng, 1 + pick(rng, 6)));
        props.set("p" + std::to_string(i), std::move(p));
    }
    Json params = Json::object();
    params.set("type", "object");
    params.set("properties", std::move(props));
    Json fn = Json::object();
    fn.set("name", "tool_" + std::to_string(index));
    fn.set("description", random_text(rng, 2 + pick(rng, 10)));
    fn.set("parameters", std::move(params));
    Json out = Json::object();
    out.set("type", "function");
    out.set("function", std::move(fn));
    return out;
}

Json message(const char* role, std::string content) {
    Json m = Json::object();
    m.set("role", role);
    m.set("content", std::move(content));
    return m;
}

// One user turn and its reply; with tools, sometimes a call and its result first.
void add_turn(Rng& rng, std::vector<Json>& messages, bool tools, std::size_t pieces, int& call) {
    messages.push_back(message("user", random_text(rng, 1 + pick(rng, pieces))));
    if (tools && pick(rng, 3) == 0U) {
        Json args = Json::object();
        args.set("p0", random_text(rng, 1 + pick(rng, 4)));
        args.set("p1", static_cast<int>(pick(rng, 100)));
        Json fn = Json::object();
        fn.set("name", "tool_0");
        fn.set("arguments", std::move(args));
        Json one = Json::object();
        const std::string id = "c" + std::to_string(++call);
        one.set("id", id);
        one.set("type", "function");
        one.set("function", std::move(fn));
        Json calls = Json::array();
        calls.push(std::move(one));
        Json m = message("assistant", "");
        m.set("tool_calls", std::move(calls));
        messages.push_back(std::move(m));
        Json result = message("tool", random_text(rng, 1 + pick(rng, pieces)));
        result.set("tool_call_id", id);
        messages.push_back(std::move(result));
    }
    Json reply = message("assistant", random_text(rng, 1 + pick(rng, pieces)));
    if (pick(rng, 3) == 0U) reply.set("reasoning_content", random_text(rng, 1 + pick(rng, pieces)));
    messages.push_back(std::move(reply));
}

struct Chat {
    std::string system;
    std::vector<Json> tools;
    std::vector<Json> messages; // without the system message
    bool thinking = true;
};

using Renderer = std::function<std::string(const ChatTemplateInput&)>;

// The prompt of the next request of a chat: its history plus one new user message.
std::string render(const Renderer& renderer, const Chat& chat, const std::string& ask) {
    ChatTemplateInput input;
    Json messages = Json::array();
    if (!chat.system.empty()) messages.push(message("system", chat.system));
    for (const auto& m : chat.messages) messages.push(m);
    messages.push(message("user", ask));
    input.messages = std::move(messages);
    if (!chat.tools.empty()) {
        Json tools = Json::array();
        for (const auto& t : chat.tools) tools.push(t);
        input.tools = std::move(tools);
    }
    if (!chat.thinking) input.enable_thinking = false;
    return renderer(input);
}

const std::array<std::pair<const char*, Renderer>, 3>& renderers() {
    static const std::array<std::pair<const char*, Renderer>, 3> all = {{
        {"qwen", [](const ChatTemplateInput& in) { return render_chat_template(in); }},
        {"deepseek", [](const ChatTemplateInput& in) { return render_deepseek_v41_chat(in); }},
        {"glm", [](const ChatTemplateInput& in) { return render_glm_chat(in); }},
    }};
    return all;
}

void check(TokenMemo& memo, const ApiTokenizer& tokenizer, const std::string& text, bool special, const char* what) {
    const auto got = memo.encode(tokenizer, text, special);
    const auto want = tokenizer.encode(text, special);
    if (got != want) throw std::runtime_error(std::string("memo ids differ from a full encode: ") + what + " (" +
                                              std::to_string(text.size()) + " bytes)");
}

// Every restart point splits the text into two independently encodable halves.
void marks_split(const BpeTokenizer& bpe, Rng& rng) {
    for (int round = 0; round < 400; ++round) {
        const std::string text = random_text(rng, 1 + pick(rng, 60));
        for (const bool special : {true, false}) {
            std::vector<EncodeMark> marks;
            const auto ids = bpe.encode_marked(text, special, marks);
            require(ids == bpe.encode(text, special), "encode_marked differs from encode");
            std::size_t last = 0;
            for (const auto& mark : marks) {
                require(mark.bytes > last && mark.bytes <= text.size() && mark.ids <= ids.size(), "restart point out of order");
                last = mark.bytes;
                auto joined = bpe.encode(std::string_view(text).substr(0, mark.bytes), special);
                require(joined.size() == mark.ids, "restart point id count");
                const auto tail = bpe.encode(std::string_view(text).substr(mark.bytes), special);
                joined.insert(joined.end(), tail.begin(), tail.end());
                require(joined == ids, "restart point does not split the encoding");
            }
        }
    }
}

// Raw texts that share a prefix cut at any byte: inside a UTF-8 sequence, inside a special token, just after one.
void raw_prefixes(const ApiTokenizer& tokenizer, Rng& rng) {
    TokenMemo memo(TokenMemo::Limits{std::size_t{1} << 20, 6});
    std::vector<std::string> kept;
    for (int round = 0; round < 6000; ++round) {
        std::string text;
        if (!kept.empty() && pick(rng, 5) != 0U) {
            const std::string& from = kept[pick(rng, kept.size())];
            text = from.substr(0, pick(rng, 4) == 0U ? from.size() : pick(rng, from.size() + 1U)) + random_text(rng, pick(rng, 12));
        } else {
            text = random_text(rng, pick(rng, 80));
        }
        const bool special = pick(rng, 8) != 0U;
        check(memo, tokenizer, text, special, "raw prefix");
        if (kept.size() < 24U) kept.push_back(text); else kept[pick(rng, kept.size())] = text;
    }
    const auto stats = memo.stats();
    require(stats.partial > 500U && stats.whole > 0U && stats.evictions > 0U,
            "raw prefixes: the memo paths were not all taken: partial " + std::to_string(stats.partial) + " whole " +
                std::to_string(stats.whole) + " evictions " + std::to_string(stats.evictions));
    require(stats.entries <= 6U && stats.bytes <= (std::size_t{1} << 20), "raw prefixes: memory bound");
}

// Random chats through each template: turns appended, edits in the middle, tools reordered or changed, the system
// prompt changed, thinking toggled, several sessions that share a prefix, a template switch on one memo.
void chats(const ApiTokenizer& tokenizer, Rng& rng, std::size_t renderer_index, TokenMemo::Limits limits) {
    TokenMemo memo(limits);
    const auto& [name, renderer] = renderers()[renderer_index];
    int call = 0, rendered = 0, rejected = 0;
    std::vector<Chat> sessions(1);
    sessions.reserve(5); // `chat` below stays valid when a session is forked
    sessions[0].system = random_text(rng, 20);
    for (int i = 0; i < 3; ++i) sessions[0].tools.push_back(tool(rng, i));
    for (int step = 0; step < 700; ++step) {
        Chat& chat = sessions[pick(rng, sessions.size())];
        const char* what = "turn appended";
        switch (pick(rng, 14)) {
        case 0: if (!chat.messages.empty()) { // edit in the middle
                    Json& m = chat.messages[pick(rng, chat.messages.size())];
                    if (const auto* role = m.find("role"); role && role->as_string() == "user") m.set("content", random_text(rng, 9));
                    what = "edit in the middle";
                } break;
        case 1: if (chat.tools.size() > 1U) { std::shuffle(chat.tools.begin(), chat.tools.end(), rng); what = "tools reordered"; } break;
        case 2: chat.system = random_text(rng, pick(rng, 25)); what = "system prompt changed"; break;
        case 3: if (sessions.size() < 5U) { sessions.push_back(chat); what = "session forked"; } break;
        case 4: chat.thinking = !chat.thinking; what = "thinking toggled"; break;
        case 5: if (pick(rng, 2) == 0U) chat.tools.clear(); else chat.tools.push_back(tool(rng, static_cast<int>(chat.tools.size())));
                what = "tools changed"; break;
        case 6: if (chat.messages.size() > 8U) { chat.messages.resize(chat.messages.size() / 2U); what = "history truncated"; } break;
        default: break;
        }
        const std::string ask = random_text(rng, 1 + pick(rng, 10));
        std::string text;
        try {
            // a second template on the same memo and tokenizer now and then: the text decides, never the template
            text = render(pick(rng, 12) == 0U ? renderers()[(renderer_index + 1U) % 3U].second : renderer, chat, ask);
        } catch (const ChatTemplateError&) {
            ++rejected; // a template refused this history (for example a tool call after truncation): drop it
            chat.messages.clear();
            continue;
        }
        ++rendered;
        check(memo, tokenizer, text, true, what);
        if (pick(rng, 6) == 0U) check(memo, tokenizer, text, true, "same prompt again");
        add_turn(rng, chat.messages, !chat.tools.empty(), 14, call);
    }
    const auto stats = memo.stats();
    require(rendered > 500, std::string(name) + ": too few histories rendered (" + std::to_string(rejected) + " rejected)");
    require(stats.partial > 200U, std::string(name) + ": the memo reused too few prefixes");
    require(stats.entries <= limits.max_entries && stats.bytes <= limits.max_bytes, std::string(name) + ": memory bound");
}

// A tokenizer without restart points: whole-text hits only, still exact.
void unmarked(const std::shared_ptr<const BpeTokenizer>& bpe, Rng& rng) {
    class Plain final : public ApiTokenizer {
    public:
        explicit Plain(std::shared_ptr<const BpeTokenizer> t) : t_(std::move(t)) {}
        std::vector<std::uint32_t> encode(std::string_view text, bool special = true) const override { return t_->encode(text, special); }
        std::string decode(std::span<const std::uint32_t> ids, bool skip = false) const override { return t_->decode(ids, skip); }
        std::string token_bytes(std::uint32_t) const override { return {}; }
        std::size_t vocab_size() const override { return 1; }
        std::uint32_t find(std::string_view) const override { return 0; }
    private:
        std::shared_ptr<const BpeTokenizer> t_;
    } plain(bpe);
    TokenMemo memo(TokenMemo::Limits{std::size_t{1} << 20, 8});
    std::string text;
    for (int i = 0; i < 200; ++i) {
        text += random_text(rng, 5);
        check(memo, plain, text, true, "unmarked tokenizer");
        check(memo, plain, text, true, "unmarked tokenizer, again");
    }
    const auto stats = memo.stats();
    require(stats.partial == 0U && stats.whole == 200U, "unmarked tokenizer: only whole-text hits are allowed");
}

// Identity: the same text under another tokenizer, flag or salt never returns the other's ids; a tiny memo keeps
// nothing it cannot hold; concurrent sessions stay exact.
void identity_bounds_threads(Rng& rng) {
    const BpeApiTokenizer a(make_tokenizer("qwen35")), b(make_tokenizer("glm4"));
    TokenMemo memo(TokenMemo::Limits{std::size_t{1} << 20, 8});
    for (int i = 0; i < 300; ++i) {
        const std::string text = random_text(rng, 40);
        check(memo, a, text, true, "tokenizer a");
        check(memo, b, text, true, "tokenizer b");
        check(memo, a, text, false, "tokenizer a, no special parsing");
        require(memo.encode(a, text, true, 7U) == a.encode(text, true), "salted identity");
    }
    TokenMemo tiny(TokenMemo::Limits{64, 4});
    const std::string big = random_text(rng, 400);
    check(tiny, a, big, true, "text over the byte bound");
    require(tiny.stats().entries == 0U && tiny.stats().bytes == 0U, "an oversize text must not be kept");

    TokenMemo shared(TokenMemo::Limits{std::size_t{4} << 20, 6});
    std::vector<std::string> failures(8);
    std::vector<std::thread> threads;
    const std::string common = random_text(rng, 200);
    for (std::size_t t = 0; t < failures.size(); ++t)
        threads.emplace_back([&, t] {
            try {
                Rng local(1234U + t);
                std::string text = common;
                for (int i = 0; i < 150; ++i) {
                    text += "<|im_end|>\n<|im_start|>" + random_text(local, 12);
                    check(shared, a, text, true, "concurrent sessions");
                }
            } catch (const std::exception& error) { failures[t] = error.what(); }
        });
    for (auto& thread : threads) thread.join();
    for (const auto& failure : failures) require(failure.empty(), failure);
    require(shared.stats().entries <= 6U, "concurrent sessions: entry bound");
}

void tokenizer_lifetime() {
    const auto first = make_tokenizer("qwen35"), second = make_tokenizer("qwen35", true);
    require(first->vocab_size() == second->vocab_size() && first->encode("a") != second->encode("a"), "replacement fixture");
    std::optional<BpeApiTokenizer> tokenizer(std::in_place, first);
    const auto* storage = &*tokenizer;
    TokenMemo whole, partial;
    const std::string prefix = "a<a>" + std::string(128, 'a');
    (void)whole.encode(*tokenizer, "a", true);
    (void)partial.encode(*tokenizer, prefix + "x", true);
    tokenizer.reset();
    tokenizer.emplace(second);
    require(&*tokenizer == storage, "replacement must reuse storage");
    require(whole.encode(*tokenizer, "a", true) == second->encode("a"), "replacement whole IDs");
    require(partial.encode(*tokenizer, prefix + "y", true) == second->encode(prefix + "y"), "replacement partial IDs");
    require(whole.stats().whole == 0U && partial.stats().partial == 0U, "replacement cannot reuse old lifetime");
    *tokenizer = BpeApiTokenizer(first);
    require(whole.encode(*tokenizer, "a", true) == first->encode("a"), "assignment must renew lifetime");
    require(whole.stats().whole == 0U, "assignment cannot reuse old lifetime");
}

void retained_capacity() {
    const std::string text(300000, 'a');
    std::size_t allocated = 0;
    const TokenMemo::Encode heavy = [&](std::string_view, std::vector<EncodeMark>& marks) {
        std::vector<std::uint32_t> ids;
        ids.reserve(100004);
        ids.resize(30000, 7U);
        marks.reserve(30000);
        for (std::size_t i = 1; i <= 30000U; ++i) marks.push_back({i * 10U, i});
        allocated = text.size() + 1U + ids.capacity() * sizeof(std::uint32_t) + marks.capacity() * sizeof(EncodeMark);
        return ids;
    };
    TokenMemo roomy(TokenMemo::Limits{std::size_t{8} << 20, 8});
    require(roomy.encode(1, 0, text, heavy).size() == 30000U, "capacity-heavy result");
    const auto charge = roomy.stats().bytes;
    require(charge > allocated, "resident charge must include retained capacity and entry overhead");
    TokenMemo small(TokenMemo::Limits{std::size_t{1} << 20, 8});
    require(small.encode(1, 0, text, heavy).size() == 30000U, "oversize result remains valid");
    require(small.stats().entries == 0U && small.stats().bytes == 0U, "capacity-heavy entry exceeds budget");
    TokenMemo evict(TokenMemo::Limits{charge * 2U - 1U, 8});
    (void)evict.encode(1, 0, text, heavy);
    (void)evict.encode(2, 0, text, heavy);
    require(evict.stats().entries == 1U && evict.stats().evictions == 1U && evict.stats().bytes == charge,
            "eviction must subtract the retained-capacity charge");
    (void)small.encode(3, 0, "a", [](std::string_view, std::vector<EncodeMark>& marks) {
        marks.reserve(100000);
        marks.push_back({1, 1});
        return std::vector<std::uint32_t>{7U};
    });
    require(small.stats().entries == 0U, "spare mark capacity exceeds budget");
    evict.clear();
    require(evict.stats().entries == 0U && evict.stats().bytes == 0U, "clear releases the resident charge");
}

// ---- benchmark: host time of one chat turn's render + encode ------------------------------------------------------

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::string prose(Rng& rng, std::size_t words) {
    static const std::vector<std::string> vocab = {
        "the", "server", "returns", "a", "list", "of", "files", "when", "request", "is", "valid", "and", "error", "function",
        "value", "for", "each", "line", "in", "buffer", "check", "that", "index", "stays", "below", "limit", "then", "write",
        "result", "to", "disk", "caf\xC3\xA9", "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E", "configuration", "tokenizer", "prefix",
        "return", "if", "(x", "==", "0)", "{", "}", ";", "std::vector<int>", "const", "auto&", "0x1F", "3.14159", "\n", "\n    "};
    std::string out;
    for (std::size_t i = 0; i < words; ++i) { if (i) out += ' '; out += vocab[pick(rng, vocab.size())]; }
    return out;
}

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const auto mid = samples.size() / 2U;
    return samples.size() % 2U ? samples[mid] : (samples[mid - 1U] + samples[mid]) / 2.0;
}

int bench(const std::string& path, const std::string& family, int rounds) {
    const bool gguf = path.size() > 5U && path.substr(path.size() - 5U) == ".gguf";
    const BpeApiTokenizer tokenizer(gguf ? BpeTokenizer::load_gguf(path) : BpeTokenizer::load_hf_json(path));
    const Renderer* renderer = nullptr;
    for (const auto& [name, r] : renderers()) if (family == name) renderer = &r;
    if (!renderer) throw std::invalid_argument("bench: family is qwen, deepseek or glm");
    std::cout << "tokmemo bench family=" << family << " vocab=" << tokenizer.vocab_size() << " lookahead=" << tokenizer.mark_lookahead()
              << " rounds=" << rounds << "\ntokens  bytes  msgs  render_median_ms  encode_full_median_ms  encode_memo_median_ms  memo_first_ms  tail_tokens  exact\n";
    for (const std::size_t target : {std::size_t{8000}, std::size_t{32000}, std::size_t{115000}}) {
        Rng rng(42);
        Chat chat;
        chat.system = "You are a coding assistant. " + prose(rng, 300);
        for (int i = 0; i < 12; ++i) {
            Json t = tool(rng, i);
            chat.tools.push_back(std::move(t));
        }
        int call = 0;
        const auto turn = [&] {
            chat.messages.push_back(message("user", prose(rng, 40 + pick(rng, 80))));
            if (pick(rng, 2) == 0U) {
                Json args = Json::object();
                args.set("p0", prose(rng, 6));
                Json fn = Json::object();
                fn.set("name", "tool_" + std::to_string(pick(rng, 12)));
                fn.set("arguments", std::move(args));
                Json one = Json::object();
                const std::string id = "call_" + std::to_string(++call);
                one.set("id", id);
                one.set("type", "function");
                one.set("function", std::move(fn));
                Json calls = Json::array();
                calls.push(std::move(one));
                Json m = message("assistant", "");
                m.set("tool_calls", std::move(calls));
                chat.messages.push_back(std::move(m));
                Json result = message("tool", prose(rng, 200 + pick(rng, 600)));
                result.set("tool_call_id", id);
                chat.messages.push_back(std::move(result));
            }
            chat.messages.push_back(message("assistant", prose(rng, 100 + pick(rng, 300))));
        };
        // grow the history to the target, estimating tokens from bytes, then settle on the measured count
        std::size_t tokens = 0;
        while (tokens < target) {
            turn();
            if (chat.messages.size() % 8U > 2U && tokens + 4000U < target) { tokens += 400U; continue; }
            tokens = tokenizer.encode(render(*renderer, chat, "next"), true).size();
        }
        TokenMemo memo(TokenMemo::Limits{});
        const std::string before = render(*renderer, chat, prose(rng, 50));
        auto t0 = std::chrono::steady_clock::now();
        (void)memo.encode(tokenizer, before, true);
        const double first_ms = ms_since(t0);
        std::vector<double> render_ms, full_ms, memo_ms;
        turn();
        const std::string ask = prose(rng, 50);
        std::size_t tail_tokens = 0, bytes = 0, total = 0;
        bool exact = true;
        for (int round = 0; round < rounds; ++round) {
            memo.clear();
            (void)memo.encode(tokenizer, before, true);
            t0 = std::chrono::steady_clock::now();
            const std::string text = render(*renderer, chat, ask);
            render_ms.push_back(ms_since(t0));
            const auto encoded_before = memo.stats().encoded_bytes;
            std::vector<std::uint32_t> want, got;
            const auto full_sample = [&] {
                t0 = std::chrono::steady_clock::now();
                want = tokenizer.encode(text, true);
                full_ms.push_back(ms_since(t0));
            };
            const auto memo_sample = [&] {
                t0 = std::chrono::steady_clock::now();
                got = memo.encode(tokenizer, text, true);
                memo_ms.push_back(ms_since(t0));
            };
            if (round % 2 == 0) { full_sample(); memo_sample(); }
            else { memo_sample(); full_sample(); }
            exact = exact && got == want;
            bytes = text.size();
            total = want.size();
            tail_tokens = (memo.stats().encoded_bytes - encoded_before) * want.size() / text.size();
        }
        std::cout << total << "  " << bytes << "  " << chat.messages.size() << "  " << median(render_ms) << "  " << median(full_ms) << "  " << median(memo_ms)
                  << "  " << first_ms << "  ~" << tail_tokens << "  " << (exact ? "yes" : "NO") << '\n';
        if (!exact) return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    try {
        if (argc == 2 && std::string(argv[1]) == "--lifetime-regression") { tokenizer_lifetime(); return 0; }
        if (argc == 2 && std::string(argv[1]) == "--capacity-regression") { retained_capacity(); return 0; }
        if (argc >= 3 && std::string(argv[1]) == "--bench")
            return bench(argv[2], argc >= 4 ? argv[3] : "qwen", argc >= 5 ? std::max(1, std::atoi(argv[4])) : 5);
        const std::uint64_t seed = argc >= 3 && std::string(argv[1]) == "--seed" ? std::stoull(argv[2]) : 20261114ULL;
        Rng rng(seed);
        tokenizer_lifetime();
        retained_capacity();
        for (const char* pre : {"qwen35", "deepseek-v3", "glm4"}) {
            const auto bpe = make_tokenizer(pre);
            const BpeApiTokenizer tokenizer(bpe);
            require(tokenizer.mark_lookahead() == bpe->mark_lookahead() && bpe->mark_lookahead() >= 12U, "lookahead");
            marks_split(*bpe, rng);
            raw_prefixes(tokenizer, rng);
            for (std::size_t r = 0; r < 3U; ++r) {
                chats(tokenizer, rng, r, TokenMemo::Limits{std::size_t{8} << 20, 12});
                chats(tokenizer, rng, r, TokenMemo::Limits{std::size_t{96} << 10, 3}); // eviction under pressure
            }
            unmarked(bpe, rng);
        }
        identity_bounds_threads(rng);
    } catch (const std::exception& error) {
        std::cerr << "token memo tests failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "token memo tests passed\n";
    return 0;
}
