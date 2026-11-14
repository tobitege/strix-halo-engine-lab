#include "qwen_prefill_options.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <thread>

namespace {
using namespace strix::moe_core;
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> void invalid(F fn) {
    try { fn(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("accepted malformed prune spec");
}
void parser() {
    for (const char* text : {"", "off", "0"}) check(!parse_route_prune(text).active(), "off spec");
    for (const char* text : {"cap\t9", "cap\n9", "cap 9", "cap\r9", "cap\v9", "cap\f9",
            "cap9\n", " cap9", "cap9\"", "cap9\\", "cap", "cap0", "cap32", "cap1.5",
            "capnan", "capinf", "rel0", "rel1", "topp-1", "rel0x1p-2", "rel1e-1000"})
        invalid([&] { (void)parse_route_prune(text); });
    const std::string nul("cap\0" "9", 5U);
    invalid([&] { (void)parse_route_prune(nul); });
    check(route_prune_tag("cap09") == "cap9", "canonical cap");
    check(route_prune_tag("rel0.5000n") == "rel0.5n", "canonical relative");
    check(route_prune_tag("topp5e-1") == "topp0.5", "canonical top-p");
    for (const char* text : {"cap1", "cap31n", "rel.123456789", "topp0.9999999n"}) {
        const auto p = parse_route_prune(text);
        const auto q = parse_route_prune(route_prune_tag(p));
        check(p.mode == q.mode && p.value == q.value && p.renorm == q.renorm, "canonical roundtrip");
    }
}

// Deterministic failures at each record boundary, including buffered flush failure.
struct FaultIo {
    static inline int fail_at = 0, calls = 0, opens = 0, closes = 0;
    static inline char handle = 0;
    static inline std::vector<int> sequence;
    static std::FILE* open(const char*) {
        ++opens;
        return fail_at == -1 ? nullptr : reinterpret_cast<std::FILE*>(&handle);
    }
    static std::size_t write(const void*, std::size_t, std::size_t count, std::FILE*) {
        ++calls;
        sequence.push_back(1);
        std::this_thread::yield();
        return calls == fail_at ? count - 1U : count;
    }
    static int flush(std::FILE*) {
        ++calls;
        sequence.push_back(2);
        return calls == fail_at ? EOF : 0;
    }
    static void close(std::FILE*) { ++closes; }
    static void reset(int at) { fail_at = at; calls = opens = closes = 0; sequence.clear(); }
};
void census() {
    const float weights[2]{0.25F, 0.75F};
    const std::uint32_t ids[2]{1, 3};
    for (int at : {-1, 1, 2, 3, 4}) {
        FaultIo::reset(at);
        {
            RouteCensusWriter<FaultIo> writer("census-failure.bin");
            const auto rejected = [&] {
                bool caught = false;
                try { writer.append(1, 2, 4, weights, ids); }
                catch (const std::runtime_error& error) {
                    caught = true;
                    check(std::string(error.what()).find("census-failure.bin") != std::string::npos, "failure path");
                }
                check(caught, "census silently accepted failed IO");
            };
            rejected();
            const int calls = FaultIo::calls;
            rejected();
            check(FaultIo::calls == calls && FaultIo::opens == 1, "IO continued after failure");
            check(calls == std::max(at, 0), "IO continued within partial record");
        }
        check(FaultIo::closes == (at == -1 ? 0 : 1), "census handle lifetime");
    }
    FaultIo::reset(0);
    {
        RouteCensusWriter<FaultIo> writer("concurrent.bin");
        std::array<std::thread, 4> threads;
        for (auto& thread : threads) thread = std::thread([&] {
            for (int i = 0; i < 64; ++i) writer.append(1, 2, 4, weights, ids);
        });
        for (auto& thread : threads) thread.join();
    }
    check(FaultIo::sequence.size() == 1024U, "concurrent record count");
    for (std::size_t i = 0; i < FaultIo::sequence.size(); ++i)
        check(FaultIo::sequence[i] == (i % 4U == 3U ? 2 : 1), "interleaved census records");
    const char* path = "route-prune-census-test.bin";
    (void)std::remove(path);
    {
        RouteCensusWriter<> writer(path);
        writer.append(1, 2, 4, weights, ids);
        writer.append(1, 2, 4, weights, ids);
    }
    std::ifstream input(path, std::ios::binary);
    for (int record = 0; record < 2; ++record) {
        std::array<std::uint32_t, 4> head{};
        std::array<float, 2> actual_weights{};
        std::array<std::uint32_t, 2> actual_ids{};
        input.read(reinterpret_cast<char*>(head.data()), sizeof(head));
        input.read(reinterpret_cast<char*>(actual_weights.data()), sizeof(actual_weights));
        input.read(reinterpret_cast<char*>(actual_ids.data()), sizeof(actual_ids));
        check(input.good() && head == std::array<std::uint32_t, 4>{0x52434e53U, 1, 2, 4} &&
              std::equal(actual_weights.begin(), actual_weights.end(), weights) &&
              std::equal(actual_ids.begin(), actual_ids.end(), ids), "census wire record changed");
    }
    check(input.peek() == std::char_traits<char>::eof(), "unexpected census bytes");
    input.close();
    check(std::remove(path) == 0, "remove census test file");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        check(argc >= 2, "missing test mode");
        const std::string_view mode(argv[1]);
        if (mode == "parser") parser();
        else if (mode == "census") census();
        else if (mode == "identity") {
            check(argc == 4, "missing CLI/config controls");
            // Warm the cached request before startup overrides: the effective policy must still update.
            (void)strix::prefill_route_prune_active();
            bool exact = false;
            unsigned disabled = 0;
            std::istringstream config(std::string("prefill_exact_fp16 = ") + (argv[3][0] == '1' ? "true" : "false"));
            strix::decode_cli::read_prefill_config(config, exact, disabled);
            strix::set_prefill_identity_override(exact, disabled);
            strix::set_prefill_identity_override(argv[2][0] == '1', 0);
            const auto policy = strix::prefill_route_prune_active();
            std::cout << "{\"policy\":" << policy.mode << ','
                      << strix::decode_cli::prefill_identity_fields(0U, 0U) << "}\n";
        } else throw std::runtime_error("unknown test mode");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
