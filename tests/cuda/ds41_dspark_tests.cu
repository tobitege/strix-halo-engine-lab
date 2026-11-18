// CUDA DSpark drafter against the host reference (ds41_dspark.cpp) bitwise on the real weights (STRIX_DS41_Q2_GGUF and
// STRIX_DS41_DSPARK_GGUF): main rows, ring seeding with wrap-around, draft steps at positions 130 / 133 / 7 (tokens,
// confidence, logits, every stage's streams), appends from device memory after a verify, restricted and real draft
// vocabularies (STRIX_DS41_DSPARK_VOCAB_DIR), then a smoke timing. STRIX_DS41_DSPARK_BENCH=1: timing only.
#include "strix/cuda/ds41_dspark.hpp"
#include "strix/ds41_dspark_vocab.hpp"
#include "../ds41_dspark_model.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {
using namespace strix::ds41::dspark;
using namespace ds41_dspark_test;
using strix::ds41::cuda::CudaDrafter;
using strix::ds41::cuda::CudaDraftTimes;

int failures = 0;
// STRIX_DS41_DSPARK_TILE_HEAD=1: the drafter keeps its head only in the tiled layout (the PRO replica reads the target's
// tiled head): every check below then covers the tiled head kernel and the extra-row gather from the tiles
strix::ds41::cuda::CudaDrafterOptions drafter_options() {
    strix::ds41::cuda::CudaDrafterOptions o;
    const char* v = std::getenv("STRIX_DS41_DSPARK_TILE_HEAD");
    o.tile_head = v && v[0] == '1';
    return o;
}
void expect(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}
template <class T>
std::size_t mismatches(const std::vector<T>& a, const std::vector<T>& b) {
    std::size_t bad = a.size() != b.size();
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        if constexpr (std::is_same_v<T, float>) bad += std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]);
        else bad += a[i] != b[i];
    }
    return bad;
}
void report(const std::string& label, std::size_t bad) {
    expect(bad == 0U, label + ": " + std::to_string(bad) + " values differ");
    std::cout << label << ": " << (bad == 0U ? "bit-exact" : "MISMATCH") << std::endl;
}
void compare_drafts(const Draft& dev, const Draft& host, const std::string& label) {
    std::size_t stage_bad = 0;
    for (std::uint32_t s = 0; s < stages; ++s) stage_bad += mismatches(dev.stage_out[s], host.stage_out[s]);
    const std::vector<std::uint32_t> dt(dev.tokens.begin(), dev.tokens.end()), ht(host.tokens.begin(), host.tokens.end());
    const std::vector<float> dc(dev.confidence.begin(), dev.confidence.end()), hc_(host.confidence.begin(), host.confidence.end());
    report(label + ": stage streams", stage_bad);
    report(label + ": head input, Markov embeddings, logits",
           mismatches(dev.head_in, host.head_in) + mismatches(dev.markov, host.markov) + mismatches(dev.logits, host.logits));
    report(label + ": drafted tokens and confidence", mismatches(dt, ht) + mismatches(dc, hc_));
    std::cout << label << ": vocabulary " << dev.candidates[0] << "; tokens";
    for (const auto t : dev.tokens) std::cout << ' ' << t;
    std::cout << ", confidence";
    for (const auto c : dev.confidence) std::cout << ' ' << c;
    std::cout << std::endl;
}
std::vector<std::uint8_t> host_ring(const State& s, std::uint32_t stage) { return s.rings[stage].entries; }
// The graph-launched draft step (no logits kept) against the host draft: tokens and confidence.
void compare_graph(CudaDrafter& device, std::uint32_t token, const Draft& host, const std::string& label) {
    const auto g = device.draft(token, false);
    const std::vector<std::uint32_t> dt(g.tokens.begin(), g.tokens.end()), ht(host.tokens.begin(), host.tokens.end());
    const std::vector<float> dc(g.confidence.begin(), g.confidence.end()), hc_(host.confidence.begin(), host.confidence.end());
    report(label + " (graph): drafted tokens and confidence", mismatches(dt, ht) + mismatches(dc, hc_));
}

void test_real(const Files& f) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto model = load_model(f);
    const auto& w = model->w;
    CudaDrafter device(w, drafter_options());
    std::cout << "head: " << (drafter_options().tile_head ? "tiled" : "rows") << std::endl;
    std::cout << "real weights loaded and uploaded in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
              << " s, drafter device bytes " << device.device_bytes() << std::endl;

    // Main rows: 130 committed rows of synthetic target hidden states (positions 0..129).
    const std::uint32_t seed_rows = 130;
    const auto hidden_rows = normal_values(static_cast<std::size_t>(seed_rows) * main_width, 1.0F, 77);
    const auto host_main = main_rows(w, hidden_rows, seed_rows);
    std::vector<float> dev_main;
    for (std::uint32_t r0 = 0; r0 < seed_rows; r0 += 64U) {
        const auto n = std::min(64U, seed_rows - r0);
        const auto part = device.main_rows(
            std::span<const float>(hidden_rows).subspan(static_cast<std::size_t>(r0) * main_width, static_cast<std::size_t>(n) * main_width), n);
        dev_main.insert(dev_main.end(), part.begin(), part.end());
    }
    report("main_proj + main_norm, 130 rows", mismatches(dev_main, host_main));

    // Seed the rings: 128 rows, then 2 (wrap-around), on both sides.
    State state;
    append(w, state, std::span<const float>(host_main).first(static_cast<std::size_t>(128) * hidden), 128);
    append(w, state, std::span<const float>(host_main).subspan(static_cast<std::size_t>(128) * hidden), 2);
    device.append(std::span<const float>(host_main).first(static_cast<std::size_t>(128) * hidden), 128);
    device.append(std::span<const float>(host_main).subspan(static_cast<std::size_t>(128) * hidden), 2);
    std::size_t ring_bad = 0;
    for (std::uint32_t s = 0; s < stages; ++s) ring_bad += mismatches(device.ring(s), host_ring(state, s));
    report("ring seeding, 130 positions over 128 slots, 3 stages", ring_bad);

    const auto host1 = draft(w, state, 1000U, true);
    CudaDraftTimes times;
    const auto dev1 = device.draft(1000U, true, &times);
    compare_drafts(dev1, host1, "draft at position 130");
    compare_graph(device, 1000U, host1, "draft at position 130");
    // Route statistics (diagnostics): the draft is unchanged with them on (a recaptured graph), every stage counts the
    // block's rows x top_k routes once per draft, and turning them off restores the previous graph's result.
    device.set_route_stats(true);
    compare_graph(device, 1000U, host1, "draft at position 130, route stats on");
    compare_graph(device, 1000U, host1, "draft at position 130, route stats on, graph reused");
    {
        const auto rs = device.route_stats();
        std::size_t bad = rs.size() != std::size_t{stages} * experts ? 1U : 0U;
        for (std::uint32_t s = 0; s < stages && bad == 0U; ++s) {
            std::uint64_t sum = 0;
            for (std::uint32_t e = 0; e < experts; ++e) sum += rs[std::size_t{s} * experts + e];
            bad += sum == 2ULL * block_size * top_k ? 0U : 1U;  // two drafts
        }
        report("route stats: each stage counts 2 drafts x block rows x top_k", bad);
    }
    device.clear_route_stats();
    device.set_route_stats(false);
    compare_graph(device, 1000U, host1, "draft at position 130, route stats off again");

    // Verify accepts two drafts: 3 committed rows appended from hidden states in device memory.
    const auto accepted_hidden = normal_values(static_cast<std::size_t>(3) * main_width, 1.0F, 78);
    append(w, state, main_rows(w, accepted_hidden, 3), 3);
    {
        strix::ds41::cuda::DeviceBuffer<float> rows_on_device(accepted_hidden.size());
        rows_on_device.upload(accepted_hidden.data(), accepted_hidden.size());
        device.append_hidden_device(rows_on_device.get(), 3, nullptr);
        ring_bad = 0;
        for (std::uint32_t s = 0; s < stages; ++s) ring_bad += mismatches(device.ring(s), host_ring(state, s));
    }
    report("append after a verify (3 committed rows from hidden states in device memory)", ring_bad);
    const auto host2 = draft(w, state, host1.tokens[3], true);
    const auto dev2 = device.draft(host1.tokens[3], true);
    compare_drafts(dev2, host2, "draft at position 133");
    compare_graph(device, host1.tokens[3], host2, "draft at position 133");

    // A short context (ring not yet full): reset, 7 rows appended as 3 + 4.
    device.reset();
    State s7;
    append(w, s7, std::span<const float>(host_main).first(static_cast<std::size_t>(7) * hidden), 7);
    device.append(std::span<const float>(host_main).first(static_cast<std::size_t>(3) * hidden), 3);
    device.append(std::span<const float>(host_main).subspan(static_cast<std::size_t>(3) * hidden, static_cast<std::size_t>(4) * hidden), 4);
    std::size_t bad = 0;
    for (std::uint32_t s = 0; s < stages; ++s) bad += mismatches(device.ring(s), host_ring(s7, s));
    report("7 rows appended as 3 + 4 equal 7 at once", bad);
    {
        const auto h7 = draft(w, s7, 42U, true);
        compare_drafts(device.draft(42U, true), h7, "draft at position 7");
        compare_graph(device, 42U, h7, "draft at position 7");
    }
    Vocab v;
    v.core = 32768;
    v.extra = {40000U, 100000U, 128000U};
    device.set_vocabulary(v);
    {
        const auto hv = draft(w, s7, 42U, true, v);
        compare_drafts(device.draft(42U, true), hv, "draft at position 7, vocabulary 32768 + 3");
        compare_graph(device, 42U, hv, "draft at position 7, vocabulary 32768 + 3");
    }
    if (const char* vocab_dir = std::getenv("STRIX_DS41_DSPARK_VOCAB_DIR"); vocab_dir && *vocab_dir) {
        const auto es = read_vocabulary_file(std::string(vocab_dir) + "/pack-es16k.bin");
        const std::vector<std::uint32_t> prompt(es.begin(), es.begin() + 40);
        auto choice = choose_vocabulary(std::string("auto:") + vocab_dir, prompt);
        std::cout << "automatic draft vocabulary: " << choice.label << std::endl;
        device.set_vocabulary(choice.vocabulary);
        const auto hauto = draft(w, s7, 42U, true, choice.vocabulary);
        compare_drafts(device.draft(42U, true), hauto, "draft at position 7, " + choice.label);
        compare_graph(device, 42U, hauto, "draft at position 7, " + choice.label);  // the graph captured here is reused below
        const std::vector<std::uint32_t> context{120000U, 120001U, 5U, 120000U};
        const auto added = device.add_vocabulary(context);
        const auto added_host = add_context(choice.vocabulary, context);
        expect(added == added_host && added == 2U, "context rows: the two new ids are added once");
        const auto hc2 = draft(w, s7, 43U, true, choice.vocabulary);
        compare_drafts(device.draft(43U, true), hc2, "draft at position 7, + 2 context rows");
        compare_graph(device, 43U, hc2, "draft at position 7, + 2 context rows");
    }
    device.set_vocabulary(Vocab{});

    // A ring started at a later position (an imported target state): empty, partial (7 rows) and wrapped (150 rows).
    for (const std::uint32_t k : {0U, 7U, 150U}) {
        const std::uint64_t start = 1000;
        device.reset(start);
        State sp;
        for (auto& r : sp.rings) { r.next = start; r.start = start; }
        for (std::uint32_t r0 = 0; r0 < k;) {
            const std::uint32_t n = std::min(128U, k - r0);
            const auto rows = std::span<const float>(host_main).subspan(static_cast<std::size_t>(r0 % 2U) * hidden, static_cast<std::size_t>(n) * hidden);
            if (n <= seed_rows - (r0 % 2U)) {
                append(w, sp, rows, n);
                device.append(rows, n);
            }
            r0 += n;
        }
        const std::string label = "ring started at 1000, " + std::to_string(k) + " rows";
        const auto hs = draft(w, sp, 7U, true);
        compare_drafts(device.draft(7U, true), hs, label);
        compare_graph(device, 7U, hs, label);
    }
    // Back-to-back appends from one device buffer rewritten in between (the input is consumed when append returns).
    {
        device.reset(0);
        State sb;
        const auto a_rows = normal_values(static_cast<std::size_t>(3) * main_width, 1.0F, 91);
        const auto b_rows = normal_values(static_cast<std::size_t>(3) * main_width, 1.0F, 92);
        append(w, sb, main_rows(w, a_rows, 3), 3);
        append(w, sb, main_rows(w, b_rows, 3), 3);
        strix::ds41::cuda::DeviceBuffer<float> src(a_rows.size());
        src.upload(a_rows.data(), a_rows.size());
        device.append_hidden_device(src.get(), 3, nullptr);
        src.upload(b_rows.data(), b_rows.size());
        device.append_hidden_device(src.get(), 3, nullptr);
        std::size_t rb = 0;
        for (std::uint32_t s = 0; s < stages; ++s) rb += mismatches(device.ring(s), host_ring(sb, s));
        report("two appends from one device buffer rewritten in between", rb);
    }
    for (int i = 0; i < 3; ++i) {
        CudaDraftTimes t;
        (void)device.draft(1000U + static_cast<std::uint32_t>(i), false, &t);
        std::cout << "smoke draft step " << i << ": device " << t.device_ms << " ms, wall " << t.wall_ms << " ms" << std::endl;
    }
}

void bench(const Files& f) {
    const auto model = load_model(f);
    CudaDrafter device(model->w, drafter_options());
    if (const char* v = std::getenv("STRIX_DS41_DSPARK_VOCAB"); v && *v) {
        std::vector<std::uint32_t> prompt{1U, 2U, 3U};
        if (const char* lang = std::getenv("STRIX_DS41_DSPARK_BENCH_LANG"); lang && *lang && std::string(v).rfind("auto:", 0) == 0) {
            const auto pack = read_vocabulary_file(std::string(v).substr(5) + "/pack-" + lang + "16k.bin");
            prompt.assign(pack.begin(), pack.begin() + 64);
        }
        const auto choice = choose_vocabulary(v, prompt);
        device.set_vocabulary(choice.vocabulary);
        std::cout << "draft vocabulary: " << choice.label << std::endl;
    }
    const auto hidden_rows = normal_values(static_cast<std::size_t>(130) * main_width, 1.0F, 77);
    device.append_hidden(std::span<const float>(hidden_rows).first(static_cast<std::size_t>(128) * main_width), 128);
    device.append_hidden(std::span<const float>(hidden_rows).subspan(static_cast<std::size_t>(128) * main_width), 2);
    const auto accepted = normal_values(static_cast<std::size_t>(3) * main_width, 1.0F, 78);
    std::uint32_t token = 1000;
    for (int i = 0; i < 12; ++i) {
        CudaDraftTimes t;
        const auto d = device.draft(token, false, &t);
        const auto a0 = std::chrono::steady_clock::now();
        device.append_hidden(accepted, 3);
        const double append_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a0).count();
        std::cout << "bench step " << i << ": draft device " << t.device_ms << " ms, wall " << t.wall_ms << " ms, append 3 rows " << append_ms
                  << " ms, tokens";
        for (const auto x : d.tokens) std::cout << ' ' << x;
        std::cout << std::endl;
        token = d.tokens[3];
    }
}

}  // namespace

int main() try {
    Files f;
    if (const char* q2 = std::getenv("STRIX_DS41_Q2_GGUF")) f.q2 = q2;
    if (const char* d = std::getenv("STRIX_DS41_DSPARK_GGUF")) f.dspark = d;
    if (f.q2.empty() || f.dspark.empty()) {
        std::cout << "set STRIX_DS41_Q2_GGUF and STRIX_DS41_DSPARK_GGUF" << std::endl;
        return 2;
    }
    if (const char* b = std::getenv("STRIX_DS41_DSPARK_BENCH"); b && *b == '1') {
        bench(f);
        return 0;
    }
    test_real(f);
    if (failures) {
        std::cerr << failures << " failure(s)" << std::endl;
        return 1;
    }
    std::cout << "ds41 cuda dspark tests passed" << std::endl;
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 2;
}
