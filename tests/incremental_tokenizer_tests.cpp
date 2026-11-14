// Incremental encoding must yield exactly the one-shot ids, with every yielded id final, for every slice size
// and every pre-tokenizer; the prompt stream runs it on the host lane.
#include "core/host_lane.hpp"
#include "strix/serve/prompt_stream.hpp"
#include "strix/serve/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace {
using strix::serve::BpeTokenizer;
using Ids = std::vector<std::uint32_t>;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

std::string utf8(std::uint32_t cp) {
    std::string s;
    if (cp < 0x80U) s.push_back(static_cast<char>(cp));
    else if (cp < 0x800U) {
        s.push_back(static_cast<char>(0xC0U | (cp >> 6U)));
        s.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    } else if (cp < 0x10000U) {
        s.push_back(static_cast<char>(0xE0U | (cp >> 12U)));
        s.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
        s.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    } else {
        s.push_back(static_cast<char>(0xF0U | (cp >> 18U)));
        s.push_back(static_cast<char>(0x80U | ((cp >> 12U) & 0x3FU)));
        s.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
        s.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    }
    return s;
}

// The byte-level alphabet of the vocabulary: printable bytes stand for themselves, the others for U+0100 onward.
std::array<std::string, 256> byte_alphabet() {
    std::array<std::string, 256> out{};
    std::uint32_t extra = 0;
    for (std::uint32_t b = 0; b < 256U; ++b) {
        const bool direct = (b >= 0x21U && b <= 0x7EU) || (b >= 0xA1U && b <= 0xACU) || (b >= 0xAEU);
        out[b] = utf8(direct ? b : 256U + extra++);
    }
    return out;
}

std::shared_ptr<const BpeTokenizer> make_tokenizer(const std::string& pre) {
    const auto alphabet = byte_alphabet();
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
        {"\xC3", "\xA9"}, {".", "\n"}, {"l", "l"}, {"o", "r"}, {" ", "w"}, {" w", "or"}, {"H", "e"}, {"He", "ll"}};
    for (const auto& [a, b] : pairs) {
        merges.push_back(symbol(a) + " " + symbol(b));
        tokens.push_back(symbol(a + b));
        types.push_back(1);
    }
    for (const char* control : {"<|im_start|>", "<|im_end|>"}) {
        tokens.emplace_back(control);
        types.push_back(3);
    }
    // Added tokens with a letter and a space inside: a slice must never end inside one.
    for (const char* added : {"<think>", "<tool call>", "[e f]", "e f"}) {
        tokens.emplace_back(added);
        types.push_back(4);
    }
    return BpeTokenizer::from_parts(std::move(tokens), std::move(types), merges, pre);
}

struct Random {
    std::uint64_t state;
    std::uint32_t next(std::uint32_t bound) {
        state ^= state << 13U;
        state ^= state >> 7U;
        state ^= state << 17U;
        return static_cast<std::uint32_t>((state >> 11U) % bound);
    }
};

std::vector<std::string> fragments() {
    std::vector<std::string> f = {
        "the", "and", "in", "there", "Hello", "WORLD", "x", "don't", "we're", "it's", "I'll", "they've", "'S", "'",
        " ", " ", " ", "  ", "   ", "\n", "\r\n", "\n\n", " \n ", "\t", " \t ",
        "1", "12", "12345", "3.14", "1 2", "7a", "a7",
        ".", ",", "!?", "...", "(", ")", "-", "_", "$", ".\n", " .", " ,\n\n",
        "<|im_start|>", "<|im_end|>", "<think>", "<tool call>", "[e f]", "e f", "<|im_", "<", "[", "<tool cal",
        " a b c ", "a  b", "a \n b", "z z", "e  f", "user\n", "assistant\n"};
    f.push_back(utf8(0x4E2DU) + utf8(0x6587U));                 // CJK
    f.push_back(utf8(0x65E5U) + utf8(0x672CU) + utf8(0x8A9EU));
    f.push_back(utf8(0x304BU) + utf8(0x306AU));                 // kana
    f.push_back(utf8(0xE9U));                                   // e acute
    f.push_back(utf8(0xFCU) + "ber");
    f.push_back("na" + utf8(0xEFU) + "ve");
    f.push_back("e" + utf8(0x301U));                            // combining mark
    f.push_back(utf8(0x1F600U));                                // emoji
    f.push_back(utf8(0xA0U));                                   // no-break space
    f.push_back(utf8(0x2028U));                                 // line separator
    f.push_back(utf8(0x662U) + utf8(0x663U));                   // non-ASCII digits
    f.push_back(std::string(1, '\xC3'));                        // malformed UTF-8
    f.push_back(std::string("\xE4\xB8"));
    f.push_back(std::string(1, '\xFF'));
    f.push_back(std::string(1, '\x80'));
    return f;
}

std::size_t check_text(const BpeTokenizer& tokenizer, const std::string& text, bool special, const std::string& label) {
    const Ids full = tokenizer.encode(text, special);
    std::size_t slices_seen = 0;
    for (const std::size_t slice : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{7}, std::size_t{16},
                                    std::size_t{64}, std::size_t{100000}}) {
        BpeTokenizer::EncodeCursor cursor;
        Ids out;
        std::size_t last = 0;
        while (tokenizer.encode_some(text, special, cursor, slice, out)) {
            require(cursor.encoded > last && cursor.encoded <= text.size(), label + ": the cursor did not advance");
            last = cursor.encoded;
            // The watermark: everything yielded so far is a prefix of the one-shot ids.
            require(out.size() <= full.size() && std::equal(out.begin(), out.end(), full.begin()),
                    label + ": a yielded id is not final");
            if (slice == 1U) {
                // Every slice end splits the text into two parts that encode independently.
                Ids joined = tokenizer.encode(std::string_view(text).substr(0, cursor.encoded), special);
                const Ids rest = tokenizer.encode(std::string_view(text).substr(cursor.encoded), special);
                joined.insert(joined.end(), rest.begin(), rest.end());
                require(joined == full, label + ": a slice end is not an independent boundary");
                ++slices_seen;
            }
        }
        require(out == full && cursor.encoded == text.size(), label + ": incremental ids differ from the one-shot ids");
        require(!tokenizer.encode_some(text, special, cursor, slice, out) && out == full, label + ": extra step");
    }
    return slices_seen;
}

void equivalence() {
    const auto parts = fragments();
    for (const char* pre : {"qwen35", "deepseek-v3", "glm4"}) {
        const auto tokenizer = make_tokenizer(pre);
        Random random{0x9E3779B97F4A7C15ULL};
        std::size_t slices = 0, specials = 0;
        const auto im_start = tokenizer->find("<|im_start|>");
        for (int n = 0; n < 400; ++n) {
            std::string text;
            const auto count = random.next(40U);
            for (std::uint32_t i = 0; i < count; ++i) text += parts[random.next(static_cast<std::uint32_t>(parts.size()))];
            for (const bool special : {true, false}) {
                slices += check_text(*tokenizer, text, special, std::string(pre) + (special ? " special" : " plain"));
                if (special)
                    for (const auto id : tokenizer->encode(text, true)) specials += id == im_start;
            }
        }
        // Texts with no slice end at all still encode, in one final step.
        for (const std::string& text : {std::string(), std::string(300, 'a'), std::string(300, ' '), std::string(300, '\n'),
                                       std::string("a"), std::string(" "), std::string("a "), std::string(" a")})
            check_text(*tokenizer, text, true, std::string(pre) + " edge");
        require(slices > 2000 && specials > 100, std::string(pre) + ": the corpus did not exercise slices and specials");
        BpeTokenizer::EncodeCursor bad;
        bad.encoded = 5;
        Ids out;
        bool refused = false;
        try {
            (void)tokenizer->encode_some("abc", true, bad, 1, out);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        require(refused, "an invalid cursor was accepted");
    }
}

std::string prose(std::size_t words, std::uint64_t seed) {
    const std::vector<std::string> vocabulary = {"the", "and", "there", "in", "world", "Hello", "token", "stream", "of",
                                                 "numbers", "12", "2026", "it's", "prefill", "chunk", "watermark"};
    Random random{seed};
    std::string text = "<|im_start|>user\n";
    for (std::size_t i = 0; i < words; ++i) {
        text += vocabulary[random.next(static_cast<std::uint32_t>(vocabulary.size()))];
        const auto r = random.next(20U);
        text += r == 0 ? ".\n" : r == 1 ? ", " : " ";
    }
    return text + "<|im_end|>\n<|im_start|>assistant\n";
}

// A tokenizer that only implements the one-shot call: the default incremental form yields it in one step.
struct OneShotOnly final : strix::serve::ApiTokenizer {
    std::shared_ptr<const BpeTokenizer> inner;
    Ids encode(std::string_view text, bool special = true) const override { return inner->encode(text, special); }
    std::string decode(std::span<const std::uint32_t> ids, bool skip = false) const override { return inner->decode(ids, skip); }
    std::string token_bytes(std::uint32_t id) const override { return inner->token_bytes(id); }
    std::size_t vocab_size() const override { return inner->vocab_size(); }
    std::uint32_t find(std::string_view text) const override { return inner->find(text); }
};

void interfaces_and_stream() {
    const auto bpe = make_tokenizer("qwen35");
    const auto api = std::make_shared<strix::serve::BpeApiTokenizer>(bpe);
    const std::string text = prose(3000, 7);
    const Ids full = api->encode(text, true);
    {
        auto steps = api->encode_incremental(text, true, 256);
        Ids out;
        std::size_t count = 0;
        while (steps->next(out)) ++count;
        require(out == full && count > 20, "API incremental encoding differs or did not slice");
        auto fallback = std::make_shared<OneShotOnly>();
        fallback->inner = bpe;
        auto one = fallback->encode_incremental(text, true, 256);
        Ids whole;
        require(one->next(whole) && whole == full && !one->next(whole) && whole == full, "default incremental form");
    }
    for (const bool enabled : {true, false}) {
        strix::core::HostLaneConfig config;
        config.enabled = enabled;
        strix::core::HostLane lane(config);
        auto stream = strix::serve::PromptStream::start(api, text, true, 64, &lane);
        // The first chunk is available before the whole prompt needs to be.
        const std::size_t first = stream->wait_for(128);
        require(first >= 128 && first <= full.size(), "first chunk watermark");
        Ids prompt;
        std::size_t takes = 0;
        for (bool done = false; !done; ++takes) {
            done = stream->take(prompt, true);
            require(prompt.size() <= full.size() && std::equal(prompt.begin(), prompt.end(), full.begin()),
                    "stream yielded an id that is not final");
        }
        require(prompt == full && stream->complete() && stream->final_count() == full.size(), "stream ids differ");
        require(stream->take(prompt, false) && stream->take(prompt, true) && prompt == full, "take after completion");
        const auto stats = lane.stats();
        require(enabled ? stats.latency_jobs > 0 && stats.critical_jobs == 0 && stats.inline_jobs == 0 : stats.inline_jobs > 0,
                "stream did not use the latency class of the lane");
        Ids ahead(full.size() + 1U, 0U);
        bool refused = false;
        try {
            (void)stream->take(ahead, false);
        } catch (const std::logic_error&) {
            refused = true;
        }
        require(refused, "a prompt ahead of its stream was accepted");
    }
}

// --bench TOKENIZER [WORDS]: one-shot time against the time until the first chunks are final (not a test).
int bench(const std::string& path, std::size_t words) {
    using Clock = std::chrono::steady_clock;
    const auto ms = [](Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
    const bool gguf = path.size() > 5U && path.compare(path.size() - 5U, 5U, ".gguf") == 0;
    const auto tokenizer = path == "synthetic" ? make_tokenizer("qwen35")
                           : gguf ? BpeTokenizer::load_gguf(path) : BpeTokenizer::load_hf_json(path);
    for (int round = 0; round < 3; ++round) {
        const std::string text = prose(words, 11U + static_cast<std::uint64_t>(round)); // new text: cold piece cache
        BpeTokenizer::EncodeCursor cursor;
        Ids out;
        const auto begin = Clock::now();
        double first[3] = {-1.0, -1.0, -1.0};
        const std::size_t marks[3] = {2048, 4096, 8192};
        while (tokenizer->encode_some(text, true, cursor, 16384, out))
            for (int m = 0; m < 3; ++m)
                if (first[m] < 0.0 && out.size() > marks[m]) first[m] = ms(Clock::now() - begin);
        const double incremental = ms(Clock::now() - begin);
        const std::string again = prose(words, 101U + static_cast<std::uint64_t>(round));
        const auto begin_one = Clock::now();
        const Ids full = tokenizer->encode(again, true);
        const double one_shot = ms(Clock::now() - begin_one);
        std::cout << "bench round " << round << " bytes " << text.size() << " ids " << out.size() << " one_shot_ms " << one_shot
                  << " incremental_total_ms " << incremental << " first_2048_ms " << first[0] << " first_4096_ms " << first[1]
                  << " first_8192_ms " << first[2] << " (other text ids " << full.size() << ")\n";
    }
    return 0;
}
// Text with a large word space (words built from syllables), so the piece cache misses as it does on real
// prompts, mixed with the common words of prose().
std::string varied_prose(std::size_t words, std::uint64_t seed) {
    static const char* const syllables[] = {"ba", "ne", "to", "ri", "sal", "ven", "kor", "mi", "du", "pha", "ly", "gre",
                                            "on", "tu", "wex", "zo", "cha", "ip", "ser", "qua", "lo", "fen", "ym", "hu"};
    Random random{seed * 2654435761ULL + 17U};
    std::string text = "<|im_start|>user\n";
    const std::string common = prose(words / 2U, seed);
    text.append(common, 17U, common.size() - 17U - 33U);
    for (std::size_t i = 0; i < words / 2U; ++i) {
        const std::uint32_t parts = 2U + random.next(3U);
        for (std::uint32_t k = 0; k < parts; ++k) text += syllables[random.next(24U)];
        text += random.next(16U) == 0 ? ".\n" : " ";
    }
    return text + "<|im_end|>\n<|im_start|>assistant\n";
}

double process_cpu_seconds() {
#if defined(_WIN32)
    FILETIME created, exited, kernel, user;
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) return 0.0;
    const auto ticks = [](const FILETIME& t) { return (static_cast<std::uint64_t>(t.dwHighDateTime) << 32U) | t.dwLowDateTime; };
    return static_cast<double>(ticks(kernel) + ticks(user)) * 1e-7;
#else
    return static_cast<double>(std::clock()) / static_cast<double>(CLOCKS_PER_SEC);
#endif
}

// --bench-concurrent TOKENIZER [WORDS] [ROUNDS]: PROMPTS prompt streams started at once on a lane with W
// latency workers (W 0: the previous arrangement, every prompt on the one critical thread), without a decode
// and next to a decode proxy. The proxy marks the byte budget, reads a buffer larger than the caches and runs
// one critical job per step, as a decode step does; its step time shows what the tokenizers cost a decode.
// Every arm checks the stream ids against the one-shot encoding. Not a test.
int bench_concurrent(const std::string& path, std::size_t words, int rounds) {
    if (rounds <= 0) throw std::invalid_argument("--bench-concurrent: ROUNDS must be positive");
    using Clock = std::chrono::steady_clock;
    const auto ms = [](Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
    const bool gguf = path.size() > 5U && path.compare(path.size() - 5U, 5U, ".gguf") == 0;
    const std::shared_ptr<const BpeTokenizer> bpe = path == "synthetic" ? make_tokenizer("qwen35")
                                                    : gguf ? BpeTokenizer::load_gguf(path) : BpeTokenizer::load_hf_json(path);
    const auto api = std::make_shared<strix::serve::BpeApiTokenizer>(bpe);
    const std::vector<std::uint64_t> buffer(8U * 1024U * 1024U, 3U); // 64 MiB
    struct Arm { int workers; unsigned decode_workers; bool decode; };
    const std::vector<Arm> arms = {{0, 1, false}, {1, 1, false}, {2, 2, false}, {4, 4, false}, {8, 8, false},
                                   {0, 1, true},  {4, 1, true},  {4, 2, true},  {4, 4, true}};
    std::uint64_t seed = 1000;
    bool identical = true;
    std::cout << "hardware_threads " << std::thread::hardware_concurrency() << " words " << words << " rounds " << rounds
              << " (median round by wall time)\n"
              << "workers decode prompts | wall_ms ids_per_s | ready_avg_ms ready_max_ms | wait_avg_us wait_max_ms | "
                 "busy_us_per_slice cpu_ms peak | step_p50_ms step_p99_ms steps\n";
    for (const Arm& arm : arms) {
        for (const std::size_t prompts : {std::size_t{0}, std::size_t{1}, std::size_t{4}, std::size_t{8}, std::size_t{16}}) {
            if (prompts == 0 && !(arm.decode && arm.workers == 0)) continue; // one decode-only reference line
            struct Round { double wall = 0, ids_ps = 0, ready_avg = 0, ready_max = 0, wait_avg = 0, wait_max = 0, slice = 0, cpu = 0, p50 = 0, p99 = 0; std::uint64_t peak = 0; std::size_t steps = 0; };
            std::vector<Round> results;
            for (int round = 0; round < rounds; ++round) {
                std::vector<std::string> texts;
                std::size_t slices = 0;
                for (std::size_t i = 0; i < prompts; ++i) {
                    texts.push_back(varied_prose(words, ++seed));
                    slices += (texts.back().size() + 16383U) / 16384U;
                }
                strix::core::HostLaneConfig config;
                config.latency_threads = arm.workers;
                config.latency_decode_threads = arm.decode_workers;
                strix::core::HostLane lane(config);
                std::atomic<bool> stop{false};
                std::vector<double> steps;
                std::thread decode;
                if (arm.decode) {
                    decode = std::thread([&] {
                        std::uint64_t sink = 0;
                        while (!stop.load(std::memory_order_relaxed)) {
                            const auto begin = Clock::now();
                            {
                                strix::core::ByteBudget::DecodeScope scope(lane.budget());
                                for (std::size_t i = 0; i < buffer.size(); i += 8U) sink += buffer[i];
                                sink += lane.submit(strix::core::HostPriority::critical, [&buffer] {
                                    std::uint64_t sum = 0;
                                    for (std::size_t i = 0; i < 65536U; ++i) sum += buffer[i];
                                    return sum;
                                }).get();
                            }
                            steps.push_back(ms(Clock::now() - begin));
                        }
                        if (sink == 1U) std::cout << ' ';
                    });
                    std::this_thread::sleep_for(std::chrono::milliseconds(30)); // the decode is under way first
                }
                const double cpu_begin = process_cpu_seconds();
                const auto begin = Clock::now();
                std::vector<std::shared_ptr<strix::serve::PromptStream>> streams;
                for (const auto& text : texts) streams.push_back(strix::serve::PromptStream::start(api, text, true, 16384, &lane));
                std::vector<std::thread> waiters;
                std::vector<double> ready(prompts, 0.0);
                for (std::size_t i = 0; i < prompts; ++i)
                    waiters.emplace_back([&, i] {
                        (void)streams[i]->wait_for(static_cast<std::size_t>(-1));
                        ready[i] = ms(Clock::now() - begin);
                    });
                for (auto& waiter : waiters) waiter.join();
                Round r;
                r.wall = ms(Clock::now() - begin);
                r.cpu = (process_cpu_seconds() - cpu_begin) * 1e3;
                if (prompts == 0) std::this_thread::sleep_for(std::chrono::milliseconds(400));
                stop = true;
                if (decode.joinable()) decode.join();
                const auto stats = lane.stats();
                std::size_t ids = 0;
                for (std::size_t i = 0; i < prompts; ++i) {
                    Ids got;
                    (void)streams[i]->take(got, true);
                    ids += got.size();
                    identical = identical && got == api->encode(texts[i], true);
                    r.ready_avg += ready[i] / static_cast<double>(prompts);
                    r.ready_max = std::max(r.ready_max, ready[i]);
                }
                const bool previous = arm.workers == 0;
                const std::uint64_t jobs = (previous ? stats.critical_jobs : stats.latency_jobs) - (previous ? steps.size() : 0U);
                r.ids_ps = r.wall > 0 ? static_cast<double>(ids) / (r.wall / 1e3) : 0.0;
                r.wait_avg = jobs ? static_cast<double>(previous ? stats.critical_wait_ns : stats.latency_wait_ns) / 1e3 / static_cast<double>(jobs) : 0.0;
                r.wait_max = static_cast<double>(previous ? stats.critical_wait_max_ns : stats.latency_wait_max_ns) / 1e6;
                r.slice = slices ? static_cast<double>(previous ? stats.critical_busy_ns : stats.latency_busy_ns) / 1e3 / static_cast<double>(slices) : 0.0;
                r.peak = previous ? 1U : stats.latency_peak_running;
                r.steps = steps.size();
                if (!steps.empty()) {
                    std::sort(steps.begin(), steps.end());
                    r.p50 = steps[steps.size() / 2U];
                    r.p99 = steps[std::min(steps.size() - 1U, steps.size() * 99U / 100U)];
                }
                results.push_back(r);
            }
            std::sort(results.begin(), results.end(), [](const Round& a, const Round& b) { return a.wall < b.wall; });
            const Round& r = results[results.size() / 2U];
            std::cout << arm.workers << (arm.decode ? "/" + std::to_string(arm.decode_workers) : std::string()) << ' '
                      << (arm.decode ? "decode" : "idle") << ' ' << prompts << " | " << r.wall << ' ' << r.ids_ps << " | "
                      << r.ready_avg << ' ' << r.ready_max << " | " << r.wait_avg << ' ' << r.wait_max << " | " << r.slice << ' '
                      << r.cpu << ' ' << r.peak << " | " << r.p50 << ' ' << r.p99 << ' ' << r.steps << std::endl;
        }
    }
    std::cout << "ids " << (identical ? "identical" : "DIFFER") << " to the one-shot encoding in every arm\n";
    return identical ? 0 : 1;
}
void benchmark_rejects_nonpositive_rounds() {
    for (const int rounds : {0, -1}) {
        bool rejected = false;
        try { (void)bench_concurrent("missing-tokenizer", 100, rounds); }
        catch (const std::invalid_argument& error) {
            rejected = std::string(error.what()) == "--bench-concurrent: ROUNDS must be positive";
        }
        require(rejected, "nonpositive rounds must be rejected before loading assets");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 3 && std::string(argv[1]) == "--bench-concurrent")
            return bench_concurrent(argv[2], argc >= 4 ? static_cast<std::size_t>(std::stoull(argv[3])) : 20000U,
                                    argc >= 5 ? std::stoi(argv[4]) : 3);
        if (argc >= 3 && std::string(argv[1]) == "--bench")
            return bench(argv[2], argc >= 4 ? static_cast<std::size_t>(std::stoull(argv[3])) : 100000U);
        benchmark_rejects_nonpositive_rounds();
        equivalence();
        interfaces_and_stream();
    } catch (const std::exception& error) {
        std::cerr << "incremental tokenizer tests failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "incremental tokenizer tests passed\n";
    return 0;
}
