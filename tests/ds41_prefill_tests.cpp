#include "strix/ds41_prefill.hpp"
#include "strix/cuda/ds41_replay.hpp"
#include "strix/ds41_memory_budget.hpp"
#include "strix/ds41_state_transfer.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;
void expect(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void test_routes() {
    using namespace strix::ds41;
    std::mt19937 random(3);
    const std::uint32_t rows = 300, top_k = 6, experts = 40, split = 20;
    std::vector<std::uint32_t> ids;
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::vector<std::uint32_t> all(experts);
        for (std::uint32_t e = 0; e < experts; ++e) all[e] = e;
        std::shuffle(all.begin(), all.end(), random);
        ids.insert(ids.end(), all.begin(), all.begin() + top_k);
    }
    for (const std::uint32_t batch : {16U, 32U}) {
        const auto r = build_prefill_routes(ids, rows, top_k, experts, split, batch);
        bool ordered = r.begins.front() == 0U && r.begins.back() == rows * top_k;
        for (std::uint32_t e = 0; e < experts; ++e)
            for (auto i = r.begins[e]; i < r.begins[e + 1U]; ++i) {
                ordered = ordered && ids[r.entry_slots[i]] == e && r.entry_rows[i] == r.entry_slots[i] / top_k;
                if (i > r.begins[e]) ordered = ordered && r.entry_slots[i - 1U] < r.entry_slots[i];
            }
        expect(ordered, "prefill routes: entries by expert in ascending slot");
        std::uint32_t covered = 0;
        bool batches = r.batch_expert.size() <= prefill_max_batches(rows, top_k, experts, batch);
        for (std::size_t b = 0; b < r.batch_expert.size(); ++b) {
            batches = batches && r.batch_count[b] >= 1U && r.batch_count[b] <= batch && r.batch_begin[b] == covered &&
                r.batch_begin[b] + r.batch_count[b] <= r.begins[r.batch_expert[b] + 1U] && (b == 0U || r.batch_expert[b - 1U] <= r.batch_expert[b]);
            covered += r.batch_count[b];
        }
        expect(batches && covered == rows * top_k, "prefill routes: batches tile the entries in order");
        bool order = true;
        for (std::uint32_t row = 0; row < rows; ++row) {
            std::uint32_t low = 0;
            for (std::uint32_t k = 0; k < top_k; ++k) {
                const auto s = r.order[row * top_k + k];
                order = order && s / top_k == row && (k == 0U || ids[r.order[row * top_k + k - 1U]] < ids[s]);
                low += ids[s] < split ? 1U : 0U;
            }
            order = order && low == r.low_counts[row];
        }
        expect(order, "prefill routes: per-row combine order and low counts");
    }
    // An expert range keeps exactly the entries of those experts (renumbered from first), the same
    // batches, and the global combine order.
    {
        const auto all = build_prefill_routes(ids, rows, top_k, experts, split, 16U);
        const auto half = build_prefill_routes(ids, rows, top_k, experts, split, 16U, split, experts);
        bool same = half.begins.size() == experts - split + 1U && half.order == all.order && half.low_counts == all.low_counts;
        for (std::uint32_t e = 0; same && e < experts - split; ++e) {
            const auto b0 = all.begins[split + e], n = all.begins[split + e + 1U] - b0;
            same = same && half.begins[e + 1U] - half.begins[e] == n;
            for (std::uint32_t i = 0; same && i < n; ++i)
                same = half.entry_slots[half.begins[e] + i] == all.entry_slots[b0 + i] && half.entry_rows[half.begins[e] + i] == all.entry_rows[b0 + i];
        }
        std::size_t batches = 0;
        for (std::size_t b = 0; b < all.batch_expert.size(); ++b) batches += all.batch_expert[b] >= split;
        same = same && half.batch_expert.size() == batches;
        expect(same, "prefill routes: an expert range is the filtered, renumbered full routes");
    }
    bool rejected = false;
    try { (void)build_prefill_routes(std::vector<std::uint32_t>{1, 1}, 1, 2, 4, 2, 16); } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "prefill routes reject a repeated expert");
}

void test_plan() {
    using namespace strix::ds41;
    const SuffixReplayGeometry g; // 40 layers, decoder from 20, window 128, drafter 128 rows of 3 layers
    const auto fresh = plan_suffix_replay(g, 0, 8192);
    expect(fresh.layers[39].out_begin == 8191U && fresh.layers[38].out_begin == 8192U - 128U &&
           fresh.layers[37].out_begin == 8192U - 255U && fresh.layers[20].out_begin == 8192U - 2414U,
           "fresh 8K plan: decoder layer 38 - m outputs 128 + 127 m rows (2414 at layer 20)");
    expect(fresh.layers[39].kv_begin == 8192U - 128U && fresh.layers[20].kv_begin == 8192U - 2541U &&
           fresh.encoder_output_begin == 8192U - 2541U, "fresh 8K plan: window K/V reach 127 rows below the outputs");
    expect(fresh.layers[0].out_begin == 0U && fresh.layers[19].out_begin == 0U, "fresh plan: the encoder runs every row");
    std::uint64_t decoder = 0;
    for (std::uint32_t l = 20; l < 40; ++l) decoder += 8192U - fresh.layers[l].out_begin;
    expect(decoder == 24'150U, "fresh 8K plan: 24,150 decoder row-layers");
    std::cout << "suffix replay 8K: decoder row-layers " << decoder << " of " << 20U * 8192U << ", total " << fresh.replay_row_layers
              << " of " << fresh.full_row_layers << " (" << 100.0 * static_cast<double>(fresh.replay_row_layers) / static_cast<double>(fresh.full_row_layers)
              << " %)\n";
    std::size_t checked = 0;
    for (const std::uint64_t cached : {0ULL, 1ULL, 100ULL, 127ULL, 128ULL, 129ULL, 1000ULL, 5000ULL})
        for (const std::uint64_t added : {1ULL, 2ULL, 50ULL, 127ULL, 128ULL, 129ULL, 300ULL, 2414ULL, 2600ULL, 4000ULL}) {
            for (const std::uint32_t draft : {0U, 128U}) {
                SuffixReplayGeometry geometry = g;
                geometry.draft_rows = draft;
                const auto plan = plan_suffix_replay(geometry, cached, cached + added);
                const auto problem = verify_suffix_replay(geometry, plan);
                expect(problem.empty(), "suffix replay plan C=" + std::to_string(cached) + " T=" + std::to_string(cached + added) +
                    " draft " + std::to_string(draft) + ": " + problem);
                ++checked;
            }
        }
    // Small geometries exercise every boundary of the cone.
    for (const std::uint32_t layers : {2U, 3U, 6U})
        for (const std::uint32_t window : {1U, 2U, 4U})
            for (std::uint64_t cached = 0; cached < 12U; ++cached)
                for (std::uint64_t added = 1; added < 30U; ++added) {
                    SuffixReplayGeometry geometry{layers, layers / 2U, window, static_cast<std::uint32_t>(added % 5U), 1U};
                    const auto plan = plan_suffix_replay(geometry, cached, cached + added);
                    const auto problem = verify_suffix_replay(geometry, plan);
                    expect(problem.empty(), "small suffix replay plan: " + problem);
                    ++checked;
                }
    // The checker is not vacuous: one row less (or more) at any decoder layer is caught.
    std::size_t caught = 0, mutations = 0;
    for (std::uint32_t l = 20; l < 40; ++l)
        for (const int delta : {-1, 1}) {
            auto plan = plan_suffix_replay(g, 0, 8192);
            plan.layers[l].out_begin = static_cast<std::uint64_t>(static_cast<std::int64_t>(plan.layers[l].out_begin) + delta);
            caught += !verify_suffix_replay(g, plan).empty();
            ++mutations;
            plan = plan_suffix_replay(g, 0, 8192);
            plan.layers[l].kv_begin = static_cast<std::uint64_t>(static_cast<std::int64_t>(plan.layers[l].kv_begin) + delta);
            caught += !verify_suffix_replay(g, plan).empty();
            ++mutations;
        }
    expect(caught == mutations, "every one-row mutation of the 8K plan is rejected");
    std::cout << checked << " suffix replay plans equal their brute-force dependency closure; " << caught << "/" << mutations
              << " one-row mutations rejected\n";
}

} // namespace

void test_passes() {
    using namespace strix::ds41;
    for (const auto total : {1U, 63U, 64U, 127U, 128U, 8191U, 8192U, 32767U, 32768U, 32769U,
                             65535U, 65536U, 65537U, 131071U, 131072U, 1048575U, 1048576U, UINT32_MAX}) {
        for (const auto capacity : {4096U, 32768U, 65536U, 131072U}) {
            const auto tail = std::min(8192U, capacity - 64U);
            const auto passes = plan_prefill_passes(total, capacity, 64, tail);
            std::uint64_t position = 0;
            for (std::size_t i = 0; i < passes.size(); ++i) {
                const auto& p = passes[i];
                expect(p.begin == position && p.rows != 0 && p.rows <= capacity, "passes cover positions once within capacity");
                expect(p.begin % 64U == 0U && (i + 1U == passes.size() || p.rows % 64U == 0U), "pass starts preserve attention window alignment");
                position += p.rows;
            }
            expect(position == total && (passes.size() == 1U || passes.back().rows >= tail), "final pass retains the complete suffix");
        }
    }
    const auto short_tail = plan_prefill_passes(32769, 32768, 64, 8192);
    expect(short_tail.size() == 2U && short_tail[0].rows == 24576U && short_tail[1].rows == 8193U,
           "one-token remainder rebalances to a decoder-sized final pass");
    unsigned caught = 0;
    for (const auto& args : std::vector<std::vector<std::uint32_t>>{{0, 64, 64, 0}, {1, 65, 64, 0},
             {1, 64, 0, 0}, {65, 64, 64, 64}, {1, 64, 64, 65}}) {
        try { (void)plan_prefill_passes(args[0], args[1], args[2], args[3]); }
        catch (const std::invalid_argument&) { ++caught; }
    }
    expect(caught == 5U, "invalid capacities and impossible aligned suffix rejected");
}

void test_memory_admission() {
    using namespace strix::ds41;
    constexpr std::uint64_t gib = 1ULL << 30;
    expect(host_commit_fits({10*gib, 0, true}, 6*gib, 4*gib), "memory admission accepts exact floor boundary");
    expect(!host_commit_fits({10*gib-1, 0, true}, 6*gib, 4*gib), "memory admission rejects a one-byte deficit");
    expect(!host_commit_fits({3*gib, 0, true}, 0, 4*gib), "memory admission preserves floor without allocation");
    expect(!host_commit_fits({UINT64_MAX, 0, true}, UINT64_MAX, 1), "memory admission cannot overflow allocation plus floor");
    expect(host_commit_fits({}, UINT64_MAX, UINT64_MAX), "unavailable platform probe leaves admission to caller");
    expect(long_request_host_reserve(1048576) < 9*gib && long_request_host_reserve(1048576) > 8*gib,
           "1M bounded CPU tail is between 8 and 9 GiB");
    expect(long_request_host_reserve(UINT32_MAX) > long_request_host_reserve(1048576), "extreme context reservation does not wrap");
}

void test_state_transfer_framing() {
    using namespace strix::ds41;
    std::vector<std::uint8_t> bytes{'D','S','4','1','A','T','S','T'};
    const auto put = [&](std::uint64_t v, unsigned width) { for (unsigned i = 0; i < width; ++i) bytes.push_back(static_cast<std::uint8_t>(v >> (i*8U))); };
    put(1,4); put(1,4); put(7,4); put(2,4);
    for (unsigned layer = 0; layer < 2; ++layer) {
        put(3,1); put(1,4);
        for (unsigned field = 0; field < 5; ++field) {
            const unsigned count = layer + field + 1U, width = field >= 3 ? 4 : 1;
            put(count,8);
            for (unsigned i = 0; i < count * width; ++i) bytes.push_back(static_cast<std::uint8_t>(layer*73U + field*11U + i));
        }
    }
    put(1,8); put(42,4); put(123,8); put(456,8);  // structural fixture, opaque digest/checksum
    const auto plan = plan_state_transfer_v1(bytes);
    std::map<std::string, std::vector<std::uint8_t>> sections;
    for (const auto& section : plan.sections) sections[section.name].resize(static_cast<std::size_t>(section.bytes));
    std::uint64_t covered = 0;
    for (const auto& slice : plan.slices) {
        expect(slice.file_offset == covered, "v2 slices cover each serialized byte in order");
        covered += slice.bytes;
        std::copy_n(bytes.begin() + slice.file_offset, static_cast<std::size_t>(slice.bytes), sections[slice.section].begin() + slice.section_offset);
    }
    std::vector<std::uint8_t> rebuilt;
    for (const auto& slice : plan.slices) {
        const auto begin = sections[slice.section].begin() + slice.section_offset;
        rebuilt.insert(rebuilt.end(), begin, begin + slice.bytes);
    }
    expect(rebuilt == bytes && covered == bytes.size(), "v2 framing preserves every v1 byte, including opaque trailer");
    unsigned rejected = 0;
    for (unsigned mutation = 0; mutation < 5U; ++mutation) {
        auto bad = bytes;
        if (mutation == 0U) bad.pop_back();
        if (mutation == 1U) bad.push_back(0);
        if (mutation == 2U) bad[0] ^= 1U;
        if (mutation == 3U) bad[12] = 2U;
        if (mutation == 4U) std::fill_n(bad.begin() + 29, 8, std::uint8_t{255});  // vector length UINT64_MAX
        try { (void)plan_state_transfer_v1(bad); }
        catch (const std::invalid_argument&) { ++rejected; }
    }
    expect(rejected == 5U, "v2 framing rejects truncation, trailing bytes, history mismatch and overflowing lengths");
}

void test_moe_growth() {
    using strix::ds41::try_grow_moe_scratch;
    using strix::ds41::bounded_moe_window_rows;
    expect(bounded_moe_window_rows(4096, 1024, 5800, 8192, true, 0) == 6144,
           "automatic pass growth remains enabled");
    expect(bounded_moe_window_rows(4096, 1024, 5800, 8192, false, 0) == 4096,
           "explicit window (including default value) or S4 off forbids growth");
    expect(bounded_moe_window_rows(4096, 1024, 5800, 8192, true, 2048) == 2048,
           "scheduled slice caps scratch below base window");
    expect(bounded_moe_window_rows(4096, 1024, 5800, 8192, true, 5000) == 5000,
           "scheduled slice caps growth without rounding above executed rows");
    struct Scratch {
        std::uint32_t rows;
        std::uint64_t bytes;
        std::uint64_t& used;
        Scratch(std::uint32_t r, std::uint64_t b, std::uint64_t& u)
            : rows(r), bytes(b), used(u) { used += bytes; }
        ~Scratch() { used -= bytes; }
    };
    const auto size = [](std::uint32_t rows) {
        return strix::ds41::prefill_moe_layout(rows, 8, 128, 7168, 2048, 2048).bytes;
    };
    std::uint64_t used = 0;
    auto scratch = std::make_unique<Scratch>(4096, size(4096), used);
    const auto* original = scratch.get();
    auto budget = used + size(6144) - 1U;
    unsigned attempts = 0;
    const auto allocate = [&] {
        ++attempts;
        if (size(6144) > budget - used) throw std::bad_alloc();
        return std::make_unique<Scratch>(6144, size(6144), used);
    };
    expect(!try_grow_moe_scratch(scratch, size(6144), budget - used, allocate),
           "growth refuses insufficient simultaneous headroom");
    expect(attempts == 0 && scratch.get() == original && used == size(4096),
           "refusal preserves old allocation without calling allocator");
    // Headroom can disappear after the query; the allocator enforces the actual budget.
    expect(!try_grow_moe_scratch(scratch, size(6144), size(6144), allocate),
           "growth catches allocation failure after admission");
    expect(attempts == 1 && scratch.get() == original && scratch->rows == 4096 && used == size(4096),
           "allocation failure retains usable old window and scratch");
    expect((5800U + scratch->rows - 1U) / scratch->rows == 2U,
           "fallback still executes two windows");
    budget = used + size(6144);
    expect(try_grow_moe_scratch(scratch, size(6144), budget - used, allocate),
           "growth succeeds with room for both allocations");
    expect(attempts == 2 && scratch->rows == 6144 && used == size(6144),
           "successful growth releases old allocation only after replacement");
}

void test_moe_schedule() {
    using strix::ds41::moe_joint_layer;
    using strix::ds41::moe_window_rows;
    // 11,594 rows split 5,795 / 5,800: one window of 6,144 rows instead of two of 4,096; never below the configured window
    expect(moe_window_rows(4096, 1024, 5800, 8192) == 6144U, "moe window: the pass rows in whole chunks");
    expect(moe_window_rows(4096, 1024, 1476, 8192) == 4096U, "moe window: at least the configured window");
    expect(moe_window_rows(4096, 1024, 0, 8192) == 4096U, "moe window: no rows keeps the configured window");
    expect(moe_window_rows(4096, 1024, 16384, 8192) == 8192U, "moe window: at most the cap");
    expect(moe_window_rows(4096, 1024, 16384, 1000) == 4096U, "moe window: a cap below the window keeps the window");
    expect(moe_window_rows(4096, 1024, 9000, 9500) == 9216U, "moe window: the cap in whole chunks");
    bool rejected = false;
    try { (void)moe_window_rows(0, 1024, 10, 8192); } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "moe window rejects a zero window");
    // joint by the layer's rows: symmetric in (peer, own), so both ranks take the same form
    expect(moe_joint_layer(512, 512, 1024) && moe_joint_layer(700, 324, 1024) && moe_joint_layer(324, 700, 1024),
           "joint layer: at most max_rows rows");
    expect(!moe_joint_layer(606, 605, 1024) && !moe_joint_layer(0, 0, 0), "joint layer: above max_rows, or max_rows 0, never");
}

void test_bounded_plan() {
    using strix::ds41::plan_bounded_replay;
    const strix::ds41::SuffixReplayGeometry g;  // 40 layers, first decoder 20, window 128, draft 128 rows x 3 layers
    for (const std::uint32_t R : {128U, 256U, 512U, 1024U}) {
        for (const std::uint64_t T : {2952ULL, 11594ULL, 32768ULL}) {
            const auto p = plan_bounded_replay(g, 0, T, R);
            bool ok = p.layers.size() == 40U && p.encoder_output_begin == 0U && p.ring_rows == 0U;
            for (const auto& l : p.layers) {
                const std::uint64_t want = l.layer < 20U ? 0U : (T - R) / 256U * 256U;
                ok = ok && l.out_begin == want && l.kv_begin == want;
            }
            ok = ok && p.replay_row_layers == T * 20U + (T - (T - R) / 256U * 256U) * 20U && p.full_row_layers == T * 40U;
            expect(ok, "bounded plan: encoder every row, every decoder layer its fixed-floor suffix (R " + std::to_string(R) + ", T " + std::to_string(T) + ")");
        }
    }
    // a prompt shorter than the bound: every row everywhere (the full prefill's rows, keys floored at 0)
    const auto shortp = plan_bounded_replay(g, 0, 100, 128);
    expect(shortp.layers[39].out_begin == 0U && shortp.layers[20].kv_begin == 0U, "bounded plan: a prompt within the bound runs every row");
    strix::ds41::SuffixReplayGeometry gd = g;
    gd.draft_rows = 512;
    bool draft_rejected = false;
    try { (void)plan_bounded_replay(gd, 0, 8192, 256); }
    catch (const std::invalid_argument& e) {
        const std::string message = e.what();
        draft_rejected = message.find("draft_rows = 512") != std::string::npos && message.find("R = 256") != std::string::npos;
    }
    expect(draft_rejected, "bounded plan refuses oversized draft with both values");
    for (const auto draft : {0U, 128U, 256U}) {
        gd.draft_rows = draft;
        const auto pd = plan_bounded_replay(gd, 0, 1001, 256);
        for (const auto window_rows : {8U, 16U}) {
            const auto floor = strix::ds41::bounded_replay_key_floor(1001, 256);
            expect(floor == 512U && pd.layers[20].out_begin / window_rows * window_rows == floor &&
                   pd.layers[39].kv_begin == floor && pd.layers[19].out_begin == 0U,
                   "bounded plan accepts draft <= R with identical fixed floor for windows 8 and 16");
        }
    }
    const auto p8 = plan_bounded_replay(g, 0, 8192, 128);
    expect(p8.replay_row_layers - 8192U * 20U == 5'120U, "bounded 8K plan: fixed floor covers 256 rows per decoder");
    std::uint32_t rejected = 0;
    for (const std::uint32_t R : {0U, 100U, 200U}) {
        try { (void)plan_bounded_replay(g, 0, 8192, R); } catch (const std::invalid_argument&) { ++rejected; }
    }
    try { (void)plan_bounded_replay(g, 8192, 8192, 128); } catch (const std::invalid_argument&) { ++rejected; }
    expect(rejected == 4U, "bounded plan rejects a bound that is not a positive multiple of the window, and no new row");
}

int main() {
    test_routes();
    test_plan();
    test_moe_schedule();
    test_moe_growth();
    test_bounded_plan();
    // Bounded replay must leave complete final rings, all requested head rows,
    // and the encoder/source-cache inputs. Exercise unaligned prompt lengths
    // and call sizes, as well as the 8K/32K qualification shapes.
    using namespace strix::ds41;
    using namespace strix::ds41::cuda;
    bool bounded_ok = true;
    for (const auto total : {1U, 63U, 64U, 127U, 128U, 129U, 255U, 256U, 257U, 511U, 8192U, 32768U})
        for (const auto call : {1U, 7U, 64U})
            for (const auto head : {1U, 64U, 256U}) {
                SuffixReplayGeometry geo;
                geo.draft_rows = 0;
                const auto plan = plan_suffix_replay(geo, 0, total);
                const auto back = [&](std::uint64_t v) { return v > head - 1U ? v - (head - 1U) : 0U; };
                for (const auto& layer : plan.layers) {
                    const bool dec = layer.layer >= geo.first_decoder;
                    auto out = static_cast<std::uint32_t>(dec ? back(layer.out_begin) : layer.out_begin) / call * call;
                    auto kv = static_cast<std::uint32_t>(dec ? back(layer.kv_begin) : layer.kv_begin) / call * call;
                    auto off_out = out, off_kv = kv;
                    bound_decoder_replay(total, 0U, call, off_out, off_kv);
                    bounded_ok &= off_out == out && off_kv == kv;
                    if (dec) bound_decoder_replay(total, 256U, call, out, kv);
                    bounded_ok &= kv <= out && out < total;
                    bounded_ok &= total - out >= std::min(total, head);
                    bounded_ok &= total - kv >= std::min(total, 128U);
                    bounded_ok &= !dec ? (out == 0U && kv == 0U) :
                        (total - out <= 256U + call - 1U && total - kv <= 383U + call - 1U);
                }
            }
    expect(bounded_ok, "bounded replay: off identity, head coverage, complete rings and aligned bounds");
    std::size_t rejected = 0;
    const auto invalid = [&](auto&& f) { try { f(); } catch (const std::invalid_argument&) { ++rejected; } };
    validate_bounded_replay(0, false, 0, 3, 1000, 1000, false, false);
    validate_bounded_replay(256, true, 0, 40, 64, 128, true, true);
    invalid([&] { validate_bounded_replay(128, true, 0, 40, 64, 0, true, true); });
    invalid([&] { validate_bounded_replay(256, false, 0, 40, 64, 0, true, true); });
    invalid([&] { validate_bounded_replay(256, true, 0, 3, 64, 0, true, true); });
    invalid([&] { validate_bounded_replay(256, true, 0, 40, 257, 0, true, true); });
    invalid([&] { validate_bounded_replay(256, true, 0, 40, 64, 257, true, true); });
    invalid([&] { validate_bounded_replay(256, true, 0, 40, 64, 0, false, true); });
    invalid([&] { validate_bounded_replay(256, true, 0, 40, 64, 0, true, false); });
    expect(rejected == 7U, "bounded replay rejects unsupported and contradictory configurations");
    test_passes();
    test_memory_admission();
    test_state_transfer_framing();
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "all deepseek41 prefill host checks passed\n";
    return 0;
}
