#include "minimax/serve/minimax_sched.hpp"
#include <iostream>
#include <limits>
#include <sstream>
#include <tuple>
#include "strix/serve/short_first.hpp"
#include "minimax/serve/minimax_control.hpp"
#ifndef _WIN32
#include <future>
#include <thread>
#include <chrono>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include "strix/serve/engine.hpp"

struct Trace {
    void* dd = this;
    unsigned aborts = 0;
    void abort(const char*) { ++aborts; }
    bool open = false;
    std::uint64_t active = 0, parked = 0, parent = 0, generation = 0;
    std::vector<unsigned> calls;
    bool prefill_open() const { return open || parked != 0; }
    void prefill_check(std::uint64_t id, unsigned) const { if (id != active || open) throw std::runtime_error("wrong completion"); }
    void require_active(std::uint64_t id, unsigned bank = 0) const { if (!open || active != id || bank != (parent != 0 ? 1U : 0U)) throw std::runtime_error("wrong task"); }
    void prefill_park(std::uint64_t id) { require_active(id, parent != 0 ? 1U : 0U); parked = id; open = false; calls.push_back(9); }
    void prefill_resume_parked(std::uint64_t id) { if (open || parked == 0 || parked != id) throw std::runtime_error("wrong parent"); active = id; parked = 0; parent = 0; open = true; calls.push_back(11); }
    void cancel_all() { if (open) prefill_cancel(active); if (parked) prefill_cancel(parked); }
    void retire_contexts(unsigned mask) { calls.push_back(1300 + mask); }
    void set_taps(bool on) { calls.push_back(on ? 81 : 80); }
    void prefill_begin(unsigned slot, std::span<const unsigned> prompt, bool context, std::uint64_t id, std::uint64_t p, std::uint64_t g, std::uint64_t resident = 0) {
        if (open || (p != 0 && (parked != p || id == p))) throw std::runtime_error("wrong begin");
        active = id; parent = p; generation = g;
        calls.push_back(200 + slot + static_cast<unsigned>(resident + prompt.size()) + (context ? 10 : 0)); open = true;
    }
    bool prefill_unit(unsigned layers, unsigned*) { calls.push_back(layers); return false; }
    void prefill_cancel(std::uint64_t id = 0) {
        if (open && active == id) open = false;
        else if (parked != 0 && parked == id) parked = 0;
        else throw std::runtime_error("wrong cancel ID");
        calls.push_back(6);
    }
    void draft(unsigned slot, unsigned anchor, std::uint64_t pos) { calls.push_back(slot + anchor + static_cast<unsigned>(pos)); }
    void step(std::span<const unsigned> rows, unsigned append) { calls.push_back(300 + static_cast<unsigned>(rows.size()) + append); }
};
struct Control {
    unsigned command = 0, bank = 0, plen = 0, pslot = 0, pstart = 0, pflags = 0, layers = 0, rows = 0, append = 0, dslots = 0;
    unsigned row[8]{}, dslot[8]{}, danchor[8]{};
    std::uint64_t dpos[8]{}, task_id = 0, parent_task_id = 0, slot_generation = 0;
};
namespace ms = strix::minimax::serve;
#ifndef _WIN32
// The production 512-byte exchange and replay run on both endpoints of a real host socket.
struct WireTrace : Trace {
    int socket = -1;
    unsigned delay_ms = 0;
    unsigned pair_rank = 0;
    unsigned rank() const { return pair_rank; }
    strix::minimax::engine::PfProgress prefill_progress(bool frozen = false) const {
        strix::minimax::engine::PfProgress p;
        p.id = frozen ? parked : active;
        p.open = frozen ? parked != 0 : open;
        p.parent = frozen ? 0 : parent;
        p.bank = frozen || parent == 0 ? 0U : 1U;
        return p;
    }
    void exchange_bytes(const void* out, void* in, std::size_t bytes) {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        if (::send(socket, out, bytes, MSG_NOSIGNAL) != static_cast<ssize_t>(bytes) ||
            ::recv(socket, in, bytes, MSG_WAITALL) != static_cast<ssize_t>(bytes))
            throw std::runtime_error("control socket exchange");
    }
    void abort(const char* reason) { Trace::abort(reason); (void)::shutdown(socket, SHUT_RDWR); }
    void step(std::span<const strix::minimax::engine::Row> rows, unsigned append) {
        calls.push_back(300 + static_cast<unsigned>(rows.size()) + append);
    }
};
#endif

int main() {
    int bad = 0;
    const auto check = [&](bool ok) { if (!ok) { ++bad; std::cerr << "scheduler check failed\n"; } };
    for (bool open : {false, true}) for (unsigned flags : {0U, 1U, 2U}) {
        Trace trace; trace.open = open;
        Control c; c.command = static_cast<unsigned>(ms::Command::taps); c.pflags = flags;
        bool rejected = false;
        try { (void)ms::replay_command(c, trace, trace, [](std::size_t n) { return std::vector<unsigned>(n); }); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected == (open || flags > 1));
        check(rejected ? trace.calls.empty() : trace.calls.size() == 1);
    }
    for (unsigned mask : {1U, 255U, 256U}) {
        Trace trace; trace.open = true;
        Control c; c.command = static_cast<unsigned>(ms::Command::retire); c.append = mask;
        bool rejected = false;
        try { (void)ms::replay_command(c, trace, trace, [](std::size_t n) { return std::vector<unsigned>(n); }); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected == (mask > 255));
        check(rejected ? trace.calls.empty() : trace.calls == std::vector<unsigned>({1300 + mask}));
    }
    check(!ms::protects(10, 10, true));
    check(!ms::protects(10, 9, false));
    check(ms::protects(10, 9, true));
    check(!ms::protects(10, 11, true));
    check(ms::off_admission(ms::Sched::off, true, true));
    check(ms::off_admission(ms::Sched::interleave, false, false));
    check(!ms::off_admission(ms::Sched::interleave, true, false));
    check(!ms::off_admission(ms::Sched::interleave, false, true));
    check(!ms::waiting_short(10, 10, 20));  // a burst-mate is not an older waiting child
    check(!ms::waiting_short(21, 10, 20));
    check(ms::waiting_short(9, 10, 20));
    // A short submitted during the long prefill has a pre-existing decoder and may nest.
    const bool protected_child = ms::protects(30, 5, true);
    check(protected_child && !ms::waiting_short(30, 10, 20));
    check(ms::nest_timing(false, protected_child, ms::waiting_short(30, 10, 20)));
    check(ms::nest_timing(false, false, ms::waiting_short(9, 10, 20)));
    check(!ms::nest_timing(true, true, true));
    check(!ms::nest_timing(false, false, ms::waiting_short(10, 10, 20)));
    check(ms::nest_fits(32, 512, true, true, true, true, false, true, {}, 100, 1500, 20, 50, 0, protected_child));
    check(!ms::reorder_prompts(ms::Sched::interleave, true, false));
    check(ms::reorder_prompts(ms::Sched::interleave, true, true));
    check(!ms::reorder_prompts(ms::Sched::off, true, true));
    check(ms::sched_mode(nullptr) == ms::Sched::interleave);
    check(ms::sched_mode("interleave") == ms::Sched::interleave);
    check(ms::sched_mode("off") == ms::Sched::off);
    check(ms::sched_mode("0") == ms::Sched::off);
    check(ms::decode_action(true, true) == ms::DecodeAction::cancel);
    check(ms::decode_action(true, false) == ms::DecodeAction::cancel);
    check(ms::decode_action(false, true) == ms::DecodeAction::context);
    check(ms::decode_action(false, false) == ms::DecodeAction::step);
    check(ms::service_ms(nullptr) == 250);
    for (const char* s : {"nan", "inf", "-1", "1e300", "abc", "250x"}) {
        bool rejected = false;
        try { (void)ms::service_ms(s); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected);
    }
    // A warm-up sample must not strand the model at two-layer units that never teach it again.
    ms::LayerModel model;
    check(std::abs(model.layer_ms(190) - 29.0) < 1e-9 && model.fixed_ms == 5.0);
    model.observe(190, 8, 800);
    check(std::abs(model.layer_ms(190) - 99.375) < 1e-9);  // first sample is unbounded
    check(ms::unit_layers(250, false, 1500, 0, model.fixed_ms, model.layer_ms(190)) == 2);
    const double warm = model.layer_ms(190);
    model.observe(190, 2, 60);
    check(model.layer_ms(190) == warm);  // wait for four accumulated layers
    model.observe(190, 2, 60);
    check(model.layer_ms(190) < warm);
    for (unsigned i = 2; i < 64; ++i) model.observe(190, 2, 60);
    check(model.layer_ms(190) >= 25 && model.layer_ms(190) <= 30.1);
    check(ms::unit_layers(250, false, 1500, 0, model.fixed_ms, model.layer_ms(190)) >= 8);
    const double steady = model.layer_ms(190);
    model.observe(190, 8, 80000);
    check(model.layer_ms(190) <= 1.3 * steady + 1e-9);  // sample <= 2x, EMA <= 1.3x
    const double high = model.layer_ms(190);
    model.observe(190, 8, 0);
    check(model.layer_ms(190) >= 0.85 * high - 1e-9);    // sample >= 0.5x

    // Isolate the overhead EMA with short prefills at the known prior layer rate. Resetting a
    // prefill discards its partial layer sample, but preserves the learned overhead and layer costs.
    ms::LayerModel overhead;
    for (unsigned i = 0; i < 64; ++i) {
        const unsigned layers = 1 + i % 3;
        overhead.observe(190, layers, layers * 29.0 + 20);
        overhead.reset_pending();
    }
    check(std::abs(overhead.fixed_ms - 20) < 1e-6);
    check(overhead.layer_ms(190) == 29.0);
    overhead.observe(190, 1, 10000);
    check(std::abs(overhead.fixed_ms - 29) < 1e-6);  // residual is capped at 50 ms
    overhead.reset_pending();
    for (unsigned i = 0; i < 64; ++i) {
        overhead.observe(190, 1, 0);
        overhead.reset_pending();
    }
    check(overhead.fixed_ms >= 0 && overhead.fixed_ms < 1e-6);

    // Partial samples cannot cross a bucket change or a prefill boundary.
    ms::LayerModel partial;
    partial.observe(190, 2, 60);
    partial.observe(300, 2, 80);
    partial.observe(190, 2, 60);
    check(partial.layer_ms(190) == 29.0 && partial.layer_ms(300) == 40.0);
    partial.reset_pending();
    partial.observe(190, 2, 60);
    check(partial.layer_ms(190) == 29.0);
    partial.observe(190, 2, 60);
    check(partial.per_layer[ms::LayerModel::bucket(190)] > 0);

    // Keep the within-bucket scaling and the rows_seen EMA: scale up, never down.
    ms::LayerModel scaling;
    scaling.observe(160, 4, 405);
    check(scaling.layer_ms(160) == 100 && scaling.layer_ms(129) == 100);
    check(scaling.layer_ms(240) == 150);
    scaling.observe(240, 4, 400 + scaling.fixed_ms);
    check(std::abs(scaling.layer_ms(240) - 100 * 240.0 / 184) < 1e-9);
    check(scaling.layer_ms(129) == 100);
    check(scaling.layer_ms(300) == 40);  // another bucket still uses its prior
    check(ms::unit_layers(250, false, 1500, 0, 5, 100) == 2);
    check(ms::unit_layers(250, true, 150, 50, 5, 20) == 4);
    check(ms::unit_layers(250, true, 150, 130, 5, 20) == 0);
    check(ms::unit_layers(250, false, 1500, 0, 5, 300) == 0);  // caller explicitly forces idle progress
    check(ms::remaining_ms(1500, 100, 400, 20) == 980);
    check(ms::remaining_ms(1500, 100, 1400, 20) == 0);
    check(!ms::runs_whole(true, 200, 0, true, 1500, 100, 1400));
    check(ms::runs_whole(true, 980, 250, true, 1500, 100, 400, 20));
    check(!ms::runs_whole(true, 981, 250, true, 1500, 100, 400, 20));
    // Consecutive admissions accumulate age; candidate row is included in the full next cycle.
    double age = 0;
    check(ms::runs_whole(true, 1000, 250, true, 1500, 80, age, 20));
    age += 1000;
    check(!ms::runs_whole(true, 1000, 250, true, 1500, 90, age, 20));
    check(!ms::runs_whole(true, 1000, 250, true, std::min(1500.0, 1000.0), 80, 0, 20));
    // A 60-layer long prompt at a recent 207 ms/layer uses 12 units and keeps the modeled gap below 1.3 s.
    unsigned long_left = 60, long_units = 0;
    while (long_left > 0) {
        const unsigned layers = std::min(long_left, ms::unit_layers(250, true, 1300, 100, 5, 207, 10, 20));
        check(layers == std::min(long_left, 5U));
        check(10 + 5 + layers * 207 + 100 + 20 <= 1300);
        long_left -= layers;
        ++long_units;
    }
    check(long_units == 12);
    // Include cold-model learning: initial two-layer units must recover to the same gap-sized service.
    ms::LayerModel cold;
    long_left = 60;
    long_units = 0;
    while (long_left > 0) {
        const unsigned layers = std::min(long_left, ms::unit_layers(250, true, 1300, 100,
            cold.fixed_ms, cold.upper_layer_ms(4539), 10, 20));
        check(layers > 0);
        if (layers == 0) break;
        const double elapsed = 5 + layers * 207.0;
        check(10 + elapsed + 100 + 20 <= 1300);
        cold.observe(4539, layers, elapsed);
        long_left -= layers;
        ++long_units;
    }
    check(long_left == 0 && long_units >= 12 && long_units <= 15);
    ms::LayerModel upper;
    upper.observe(4539, 4, 4 * 207.0 + 5);
    check(upper.upper_layer_ms(4539) >= 207);
    upper.observe(4539, 1, 900);
    check(upper.upper_layer_ms(4539) >= 900);  // immediate, not delayed by EMA aggregation
    for (unsigned i = 0; i < 7; ++i) upper.observe(4539, 1, 200);
    check(upper.upper_layer_ms(4539) >= 900);
    upper.observe(4539, 1, 200);
    check(upper.upper_layer_ms(4539) < 900);  // bounded recent history

    check(ms::unit_layers(0, false, 1500, 0, 5, 20) == ~0U);
    check(ms::unit_layers(0, true, 1500, 100, 5, 20) == 69);
    // Whole prefill: a short burst prompt (1050 ms) fits the gap less the next cycle and the decoders' wait so far; the
    // service target alone would split it.
    check(ms::runs_whole(true, 1050, 250, true, 1500, 100, 50));
    check(!ms::runs_whole(true, 1050, 250, false, 1500, 0, 0));
    check(ms::runs_whole(true, 1350, 250, true, 1500, 100, 50));    // exactly at the bound
    check(!ms::runs_whole(true, 1351, 250, true, 1500, 100, 50));
    check(!ms::runs_whole(true, 1050, 250, true, 1500, 100, 400));  // decoders already waited behind a previous unit
    check(!ms::runs_whole(true, 200, 250, true, 1500, 100, 1400));  // depleted gap beats service
    check(!ms::runs_whole(true, 4000, 250, true, 1500, 100, 0));    // long prompt: bounded units
    check(!ms::runs_whole(true, 4000, 250, false, 1500, 0, 0));     // idle long prompt keeps its service opportunities
    check(!ms::runs_whole(false, 100, 250, true, 1500, 100, 0));    // a started task keeps interleaving
    check(!ms::runs_whole(true, 200, 250, true, 100, 150, 0));      // cycle alone exceeds gap
    check(!ms::runs_whole(true, 300, 250, true, 100, 150, 0));
    const double inf = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN();
    check(!ms::runs_whole(true, 300, 250, true, inf, 100, 0));      // nonfinite gaps fail closed
    check(!ms::runs_whole(true, 300, 250, true, nan, 100, 0));
    check(!ms::runs_whole(true, nan, 250, true, 1500, 100, 0));
    check(ms::runs_whole(false, 4000, 0, false, 1500, 0, 0));       // service 0: an idle server runs whole
    check(ms::runs_whole(true, 1400, 0, true, 1500, 100, 0));       // service 0 with decoders: a fresh prompt within the gap
    check(!ms::runs_whole(true, 1600, 0, true, 1500, 100, 0));
    check(!ms::runs_whole(false, 100, 0, true, 1500, 100, 0));
    check(ms::gap_ms(nullptr) == 1500);
    check(ms::gap_ms("800") == 800);
    for (const char* s : {"nan", "inf", "0", "0.5", "-1", "1e300", "abc"}) {
        bool rejected = false;
        try { (void)ms::gap_ms(s); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected);
    }
    // Fake time: every idle resume admits a service opportunity, even for a long prompt.
    unsigned left = 600, boundaries = 0;
    double clock_ms = 0;
    while (left > 0) {
        const unsigned n = std::min(left, ms::unit_layers(250, false, 1500, 0, 5, 20));
        check(5 + n * 20 <= 250);
        left -= n;
        clock_ms += 5 + n * 20;
        ++boundaries;
    }
    check(boundaries == 50 && clock_ms == 12250);
    // A runtime change between resumes only affects later admissions, including off/coarse modes.
    ms::RequestPolicy runtime;
    const auto admitted = runtime;
    const unsigned first_resume = admitted.unit_layers(false, 0, 5, 20);
    check(first_resume == 12 && !admitted.runs_whole(false, 4000, false, 0, 0));
    runtime.sched = ms::Sched::off;
    runtime.service_ms = 0;
    runtime.gap_ms = 100;
    check(admitted.unit_layers(false, 0, 5, 20) == first_resume);
    check(!admitted.runs_whole(false, 4000, false, 0, 0));
    const auto later = runtime;
    check(later.runs_whole(false, 4000, false, 0, 0) && later.unit_layers(false, 0, 5, 20) == ~0U);
    runtime.sched = ms::Sched::interleave;
    runtime.service_ms = 50;
    check(runtime.unit_layers(false, 0, 5, 20) == 2);
    check(admitted.sched == ms::Sched::interleave && admitted.service_ms == 250 && admitted.gap_ms == 1500);
    ms::NestQuota quota;
    auto fits = [&](unsigned rows = 183, double waited = 100, double predicted = 1050, bool enabled = true,
                    bool parent = true, bool child = true, bool ready = true, bool parked = false, bool free_slot = true) {
        return ms::nest_fits(rows, 512, enabled, parent, child, ready, parked, free_slot, quota,
                             predicted, 1500, waited, 100, 20, true);
    };
    check(fits()); // 100 + 20 + 1050 + 100 + 20 = 1290 <= 1300
    check(!fits(183, 111));
    check(!fits(513) && !fits(0));
    check(!ms::nest_fits(183, 512, true, true, true, true, false, true, quota, 100, nan, 0, 0, 20, false));
    check(!ms::nest_fits(183, 512, true, true, true, true, false, true, quota, 100, inf, 0, 0, 20, false));
    check(!fits(183, 100, inf) && !fits(183, 100, nan));
    check(!fits(183, 100, 1050, false));
    check(!fits(183, 100, 1050, true, false));
    check(!fits(183, 100, 1050, true, true, false));
    check(!fits(183, 100, 1050, true, true, true, false));
    check(!fits(183, 100, 1050, true, true, true, true, true));
    check(!fits(183, 100, 1050, true, true, true, true, false, false));
    quota.admitted();
    check(!fits());
    quota.progressed = true;
    check(fits());
    quota.admitted();
    quota.progressed = true;
    check(!fits()); // third child rejected despite parent progress
    quota = {};
    quota.wall_ms = 2600;
    check(!fits());
    quota = {};
    check(ms::nest_fits(183, 512, true, true, true, true, false, true, quota, 1200, 1300, 0, 0, 20, false));
    check(!ms::nest_fits(183, 512, true, true, true, true, false, true, quota, 1261, 1300, 0, 0, 20, false));
    // A 4608-row idle unit exceeds the 250 ms budget: progress still requires one layer.
    const ms::LayerModel idle_model;
    check(ms::unit_layers(250, false, 1500, 0, idle_model.fixed_ms, idle_model.layer_ms(4608)) == 0);
    check(ms::resume_layers(ms::unit_layers(250, false, 1500, 0, idle_model.fixed_ms,
                                          idle_model.layer_ms(4608))) == 1);
    // Learn a slow CHECK plus setup/publication, excluding child engine service once.
    const double switch_cost = ms::nest_switch_ms(1150 + 20 + 100 + 10, 1150);
    check(switch_cost == 130);
    check(ms::nest_fits(183, 512, true, true, true, true, false, true, quota, 1150, 1300, 0, 100, 20, true));
    check(!ms::nest_fits(183, 512, true, true, true, true, false, true, quota, 1150, 1300, 0, 100, switch_cost, true));
    for (unsigned rows : {511U, 512U}) check(fits(rows));
    for (unsigned rows : {0U, 1U, 511U, 512U, 513U}) {
        std::string bytes(static_cast<std::size_t>(rows) * 4, '\0');
        std::istringstream input(bytes);
        bool rejected = false;
        try { check(ms::nested_fixture(input, 512, 8256, 128).size() == rows); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected == (rows == 0 || rows == 513));
    }
    for (unsigned bytes : {1U, 3U, 5U, 2049U}) {
        std::istringstream input(std::string(bytes, '\0'));
        bool rejected = false;
        try { (void)ms::nested_fixture(input, 512, 8256, 128); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected);
    }
#ifndef _WIN32
    for (unsigned opcode : {2U, 5U, 6U, 9U, 10U, 11U, 12U}) {
        for (std::uint64_t id : {std::uint64_t{0}, std::uint64_t{999}}) {
            int sockets[2]{};
            if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) throw std::runtime_error("socketpair");
            const auto rank = [&](unsigned r) {
                WireTrace engine;
                engine.socket = sockets[r]; engine.pair_rank = r; engine.active = 42; engine.parent = 41;
                engine.open = opcode != 11 && opcode != 12; engine.parked = 41;
                ms::Control out;
                if (r == 0) { out.command = opcode; out.task_id = id; out.layers = 1; out.parent_task_id = opcode == 10 ? 999U : 0U; out.bank = opcode == 10 || opcode == 5 ? 1U : 0U; }
                bool rejected = false;
                try {
                    const auto peer = ms::exchange(engine, out);
                    const auto& command = r == 0 ? out : peer;
                    (void)ms::replay_command(command, engine, engine, [](unsigned n) { return std::vector<unsigned>(n); });
                } catch (const std::runtime_error&) { rejected = true; }
                return rejected && engine.aborts > 0 && engine.calls.empty() && engine.active == 42 && engine.parked == 41;
            };
            auto follower = std::async(std::launch::async, [&] { return rank(1); });
            const bool lead_ok = rank(0);
            check(lead_ok && follower.get());
            (void)::close(sockets[0]); (void)::close(sockets[1]);
        }
    }
    {
        int sockets[2]{};
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) throw std::runtime_error("socketpair");
        WireTrace lead, follower;
        lead.socket = sockets[0]; follower.socket = sockets[1]; follower.pair_rank = 1; follower.delay_ms = 100;
        ms::Control completion;
        completion.command = static_cast<unsigned>(ms::Command::check); completion.task_id = 42;
        auto peer = std::async(std::launch::async, [&] { return ms::exchange(follower, ms::Control{}); });
        const auto start = std::chrono::steady_clock::now();
        (void)ms::exchange(lead, completion);
        check(peer.get().task_id == 42);
        const double check_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const double learned = ms::nest_switch_ms(1150 + 20 + check_ms, 1150);
        check(learned >= 100);
        check(!ms::nest_fits(183, 512, true, true, true, true, false, true, quota,
                             1150, 1300, 0, 100, learned, true));
        (void)::close(sockets[0]); (void)::close(sockets[1]);
    }
#endif
    Trace trace;
    Control c;
    auto replay = [&] { return ms::replay_command(c, trace, trace, [](unsigned n) { return std::vector<unsigned>(n); }); };
    c.command = 2; c.task_id = 1; c.plen = 4; c.pslot = 1; c.pflags = 1; check(replay());
    c.command = 5; c.layers = 2; check(replay());
    c.command = 3; c.rows = 2; c.append = 1; check(replay());
    c.command = 5; c.layers = 1; check(replay());
    c.command = 6; check(replay());
    c.command = 2; check(replay());
    c.command = 4; check(!replay());
    check(trace.calls == std::vector<unsigned>({215, 2, 303, 1, 6, 215, 6}));
    check(!trace.open);
    c.command = 1; check(replay());
    c.command = 8; c.pflags = 0; check(replay());
    c.command = 7; c.dslots = 1; c.dslot[0] = 1; c.danchor[0] = 2; c.dpos[0] = 3; check(replay());
    check(trace.calls[trace.calls.size() - 2] == 80 && trace.calls.back() == 6);
    c.command = 99;
    bool unknown = false;
    try { (void)replay(); } catch (const std::runtime_error&) { unknown = true; }
    check(unknown);
    // Explicit identities and generations survive replay; cancelling the parked parent leaves the child active.
    c = {};
    c.command = 2; c.task_id = 41; c.slot_generation = 9; c.plen = 4608;
    check(replay() && trace.active == 41 && trace.generation == 9);
    c.command = 9; check(replay() && trace.parked == 41 && !trace.open);
    c.command = 10; c.task_id = 42; c.parent_task_id = 41; c.bank = 1; c.pslot = 2; c.plen = 183;
    check(replay() && trace.active == 42 && trace.parent == 41);
    c.command = 5; c.layers = 7; check(replay());
    c.command = 6; c.task_id = 41; check(replay() && trace.open && trace.active == 42 && trace.parked == 0);
    c.task_id = 42; check(replay() && !trace.open);
    c = {}; c.command = 2; c.task_id = 51; check(replay());
    c.command = 9; check(replay());
    c.command = 10; c.task_id = 52; c.parent_task_id = 51; c.bank = 1; check(replay());
    c.command = 6; check(replay() && trace.parked == 51);
    c.command = 11; c.task_id = 51; check(replay() && trace.open && trace.active == 51);
    c.command = 5; c.task_id = 52;
    bool stale = false;
    try { (void)replay(); } catch (const std::runtime_error&) { stale = true; }
    check(stale);
    c.task_id = 51; c.bank = 1;
    const auto aborts_before_bank = trace.aborts;
    bool wrong_bank = false;
    try { (void)replay(); } catch (const std::runtime_error&) { wrong_bank = true; }
    check(wrong_bank && trace.aborts == aborts_before_bank + 1);
    c.command = 6; c.task_id = 51; check(replay());
    // Every task-addressed wire opcode rejects zero before receive or state mutation.
    for (unsigned opcode : {2U, 5U, 6U, 9U, 10U, 11U, 12U}) {
        c = {}; c.command = opcode;
        const auto calls = trace.calls;
        const auto aborts = trace.aborts;
        bool rejected = false;
        try { (void)replay(); } catch (const std::runtime_error&) { rejected = true; }
        check(rejected && trace.aborts == aborts + 1 && trace.calls == calls);
    }
    namespace sv = strix::serve;
    sv::GenerationQueue queue;
    auto submit = [&](unsigned rows) {
        sv::GenerationJob j;
        j.prompt.resize(rows);
        return queue.submit(std::move(j));
    };
    auto long_job = submit(100);
    auto tie1 = submit(2);
    auto tie2 = submit(2);
    auto cancelled = submit(1);
    cancelled->cancel();
    auto take = [&] {
        auto t = queue.take(false, sv::QueueSelection::short_first);
        auto ticket = t->ticket;
        queue.finish(*t, {});
        return ticket;
    };
    check(take() == tie1);
    check(long_job->queue_position() == 0 && tie2->queue_position() == 1);
    check(cancelled->stats().finish == "cancelled");
    check(take() == tie2);
    for (unsigned i = 2; i < sv::kMaxQueueOvertakes; ++i) {
        auto short_job = submit(1);
        check(take() == short_job);
    }
    auto short_job = submit(1);
    check(take() == long_job);
    check(take() == short_job);
    auto first = submit(100);
    auto second = submit(1);
    auto fifo = queue.take(false);
    check(fifo->ticket == first);
    queue.finish(*fifo, {});
    check(take() == second);
    check(queue.snapshot().running == 0 && queue.snapshot().queued == 0);
    // Exercise the production queue's atomic timestamp gate, not just the policy harness.
    auto unprotected_long = submit(1000);
    auto unprotected_short = submit(10);
    auto gated = queue.take(false, sv::QueueSelection::short_first, {},
        [](auto) { return false; });
    check(gated && gated->ticket == unprotected_long && !gated->reordered);
    if (gated) queue.finish(*gated, {});
    check(queue.has_waiting([](const auto& j, auto) { return j.prompt.size() == 10; }));
    check(take() == unprotected_short);
    auto protected_long = submit(1000);
    auto protected_short = submit(10);
    gated = queue.take(false, sv::QueueSelection::short_first, {}, [](auto) { return true; },
        [](const auto& j, auto submitted) { return j.prompt.size() <= 512 && submitted.time_since_epoch().count() > 0; });
    check(gated && gated->ticket == protected_short && gated->reordered);
    if (gated) queue.finish(*gated, {});
    check(take() == protected_long);
    auto aged_long = submit(1000);
    for (unsigned i = 0; i < sv::kMaxQueueOvertakes; ++i) {
        auto small = submit(10);
        auto chosen = queue.take(false, sv::QueueSelection::short_first,
            [](const sv::GenerationJob& j) { return j.prompt.size() <= 512; });
        check(chosen && chosen->ticket == small);
        if (chosen) queue.finish(*chosen, {});
    }
    auto blocked = submit(10);
    check(!queue.take(false, sv::QueueSelection::short_first,
        [](const sv::GenerationJob& j) { return j.prompt.size() <= 512; }));
    check(take() == aged_long);
    check(take() == blocked);
    return bad == 0 ? 0 : 1;
}
