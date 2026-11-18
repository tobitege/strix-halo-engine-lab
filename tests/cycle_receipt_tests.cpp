// Host tests of the serving cycle receipt and ledger (no device).
#include "strix/serve/cycle_receipt.hpp"
#include "strix/serve/json.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#if defined(_MSC_VER)
#include <crtdbg.h>
#endif

using namespace strix::serve;

namespace {
int failures = 0;
void check(bool value, const char* label) {
    if (!value) {
        ++failures;
        std::cerr << "FAIL: " << label << '\n';
    }
}
bool near(double a, double b, double tolerance) { return std::fabs(a - b) <= tolerance; }
const Json& at(const Json& j, const char* key) {
    const auto* value = j.find(key);
    if (!value) throw std::runtime_error(std::string("missing ") + key);
    return *value;
}

// Small deterministic generator: the tests must not depend on the platform's random engine.
struct Lcg {
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    std::uint32_t next(std::uint32_t bound) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>((state >> 33) % bound);
    }
    double noise() { return (static_cast<double>(next(2001)) - 1000.0) / 1000.0; }  // [-1, 1]
};

CycleReceipt receipt(std::uint32_t lanes, std::uint32_t physical, std::uint32_t useful, double ms) {
    CycleReceipt r;
    r.lanes = lanes;
    r.set_rows(physical, useful);
    r.emitted = useful;
    r.total_ms = ms;
    return r;
}

void fit_recovers_planted_coefficients() {
    // Rows per lane vary, so rows and lanes are not collinear and all three terms are identified.
    CycleLedger exact;
    CycleLedger noisy;
    Lcg rng;
    for (int i = 0; i < 4000; ++i) {
        const std::uint32_t lanes = 1 + rng.next(8);
        const std::uint32_t rows = lanes * (1U << rng.next(3));
        const double ms = 48.0 + 2.69 * rows + 9.78 * lanes;
        check(exact.add(receipt(lanes, rows, rows, ms)), "exact add");
        check(noisy.add(receipt(lanes, rows, rows, ms + 0.5 * rng.noise())), "noisy add");
    }
    const auto f = exact.fit();
    check(f.terms == 3 && f.cycles == 4000, "exact fit identifies three terms");
    check(near(f.fixed_ms, 48.0, 1e-6) && near(f.row_ms, 2.69, 1e-7) && near(f.lane_ms, 9.78, 1e-6),
        "exact fit recovers planted coefficients");
    check(f.residual_sd < 1e-4, "exact fit has no residual");
    const auto g = noisy.fit();
    check(near(g.fixed_ms, 48.0, 0.1) && near(g.row_ms, 2.69, 0.01) && near(g.lane_ms, 9.78, 0.05),
        "noisy fit recovers planted coefficients");
    // Uniform noise of half-width 0.5 has sd 0.5 / sqrt(3) = 0.289.
    check(near(g.residual_sd, 0.2887, 0.02), "noisy fit reports the noise sd");
}

void fit_degenerate_inputs() {
    CycleLedger empty;
    check(empty.fit().cycles == 0 && empty.fit().terms == 0, "empty ledger has no fit");

    // Fixed rows per lane: rows and lanes are collinear; the slope is reported per row and the lane term is 0.
    CycleLedger collinear;
    for (std::uint32_t lanes = 1; lanes <= 8; ++lanes)
        for (int repeat = 0; repeat < 3; ++repeat) collinear.add(receipt(lanes, lanes * 4, lanes * 4, 48.0 + 7.5 * lanes * 4));
    const auto c = collinear.fit();
    check(c.terms == 2 && near(c.row_ms, 7.5, 1e-9) && c.lane_ms == 0.0 && near(c.fixed_ms, 48.0, 1e-8),
        "collinear rows and lanes give the per-row slope");

    // Only the lane count moves.
    CycleLedger lanes_only;
    for (std::uint32_t lanes = 1; lanes <= 4; ++lanes) lanes_only.add(receipt(lanes, 16, 16, 100.0 + 3.0 * lanes));
    const auto l = lanes_only.fit();
    check(l.terms == 2 && l.row_ms == 0.0 && near(l.lane_ms, 3.0, 1e-9) && near(l.fixed_ms, 100.0, 1e-8),
        "constant rows give the per-lane slope");

    // Nothing moves: the mean.
    CycleLedger flat;
    for (int i = 0; i < 5; ++i) flat.add(receipt(4, 16, 16, 160.0 + (i % 2 ? 1.0 : -1.0)));
    const auto m = flat.fit();
    check(m.terms == 1 && m.row_ms == 0.0 && m.lane_ms == 0.0 && near(m.fixed_ms, 159.8, 1e-9), "flat input gives the mean");
    check(m.residual_sd > 0.9 && m.residual_sd < 1.2, "flat input reports its spread");

    // Cycles without a lane are counted but do not enter the fit.
    CycleLedger idle;
    idle.add(receipt(0, 0, 0, 5.0));
    check(idle.totals().cycles == 1 && idle.fit().cycles == 0, "idle cycles stay out of the fit");
}

void rows_identity_and_totals() {
    CycleReceipt r;
    r.set_rows(16, 13);
    check(r.physical_rows == 16 && r.useful_rows == 13 && r.padding_rows == 3 && r.consistent(), "set_rows splits padding");
    r.set_rows(4, 9);
    check(r.useful_rows == 4 && r.padding_rows == 0 && r.consistent(), "useful rows are clamped to physical rows");
    r.add_draft_depth(0);
    r.add_draft_depth(1);
    r.add_draft_depth(3, 2);
    r.add_draft_depth(40);
    check(r.draft_rows_by_depth[0] == 1 && r.draft_rows_by_depth[2] == 2 && r.draft_rows_by_depth[7] == 1,
        "draft depths land in their entries and deep drafts share the last one");

    CycleLedger ledger(8);
    CycleReceipt bad = receipt(2, 8, 6, 60.0);
    bad.padding_rows = 5;
    check(!bad.consistent() && !ledger.add(bad) && ledger.rejected() == 1 && ledger.totals().cycles == 0,
        "inconsistent rows are rejected");
    CycleReceipt nan = receipt(2, 8, 6, 60.0);
    nan.verify_ms = std::nan("");
    check(!ledger.add(nan) && ledger.rejected() == 2, "non-finite times are rejected");

    Lcg rng;
    std::uint64_t emitted = 0;
    for (std::uint32_t i = 0; i < 20; ++i) {
        CycleReceipt c = receipt(1 + i % 4, 4 * (1 + i % 4), 1 + rng.next(4 * (1 + i % 4)), 50.0 + i);
        c.prompt_unit_ran = i % 5 == 0;
        c.draft_ms = 1.0;
        c.verify_ms = 2.0;
        c.host_ms = 0.5;
        c.add_draft_depth(1 + i % 3);
        emitted += c.emitted;
        check(ledger.add(c), "consistent receipt accepted");
    }
    const auto t = ledger.totals();
    check(t.cycles == 20 && t.prompt_cycles == 4, "cycle and prompt counts");
    check(t.useful_rows + t.padding_rows == t.physical_rows, "useful + padding = physical in the totals");
    check(t.emitted == emitted && near(t.draft_ms, 20.0, 1e-12) && near(t.verify_ms, 40.0, 1e-12) && near(t.host_ms, 10.0, 1e-12),
        "token and time totals");
    check(t.draft_rows_by_depth[0] + t.draft_rows_by_depth[1] + t.draft_rows_by_depth[2] == 20, "draft rows by depth");
    std::uint64_t steady = 0;
    for (std::size_t lanes = 0; lanes <= kCycleLaneBuckets; ++lanes) steady += ledger.lanes(lanes).cycles;
    check(steady == 16, "lane table holds only cycles without a prompt unit");
    check(ledger.lanes(4).cycles == 4 && ledger.lanes(4).physical_rows == 64, "lane table row for four lanes");

    // Ring: capacity 8, 20 cycles added, oldest first.
    const auto recent = ledger.recent();
    check(recent.size() == 8 && near(recent.front().total_ms, 62.0, 1e-12) && near(recent.back().total_ms, 69.0, 1e-12),
        "ring keeps the most recent receipts in order");
    check(ledger.recent(3).size() == 3 && near(ledger.recent(3).front().total_ms, 67.0, 1e-12), "ring honours a limit");

    CycleLedger wide;
    wide.add(receipt(200, 400, 400, 900.0));
    check(wide.lanes(kCycleLaneBuckets).cycles == 1 && wide.lanes(5000).cycles == 1, "lane counts past the table share its last row");
    wide.clear();
    check(wide.totals().cycles == 0 && wide.fit().cycles == 0 && wide.recent().empty(), "clear empties the ledger");
}

void json_round_trip() {
    CycleLedger ledger(16);
    Lcg rng;
    for (std::uint32_t i = 0; i < 40; ++i) {
        const std::uint32_t lanes = 1 + rng.next(4), rows = lanes * (1U << rng.next(3));
        CycleReceipt c = receipt(lanes, rows, rows - rng.next(lanes), 48.0 + 7.5 * rows + 0.1 * rng.noise());
        c.draft_ms = 3.25;
        c.verify_ms = c.total_ms - 4.0;
        c.host_ms = 0.75;
        c.prompt_unit_ran = i % 7 == 0;
        c.begin_ns = 1000 * static_cast<std::int64_t>(i);
        c.end_ns = c.begin_ns + 900;
        c.add_draft_depth(1 + i % 8);
        ledger.add(c);
    }
    const Json j = parse_json(ledger.json(4));
    const auto t = ledger.totals();
    const auto f = ledger.fit();
    check(at(j, "schema").as_string() == "strix.serve.cycle_ledger.v1", "schema");
    check(at(j, "cycles").as_int() == 40 && at(j, "rejected").as_int() == 0 && at(j, "ring_capacity").as_int() == 16, "header");
    const Json& totals = at(j, "totals");
    check(static_cast<std::uint64_t>(at(totals, "physical_rows").as_int()) == t.physical_rows &&
        static_cast<std::uint64_t>(at(totals, "useful_rows").as_int()) == t.useful_rows &&
        static_cast<std::uint64_t>(at(totals, "padding_rows").as_int()) == t.padding_rows &&
        static_cast<std::uint64_t>(at(totals, "emitted").as_int()) == t.emitted &&
        static_cast<std::uint64_t>(at(totals, "lanes").as_int()) == t.lanes &&
        static_cast<std::uint64_t>(at(totals, "prompt_cycles").as_int()) == t.prompt_cycles, "integer totals round trip");
    check(at(totals, "total_ms").as_double() == t.total_ms && at(totals, "verify_ms").as_double() == t.verify_ms &&
        at(totals, "draft_ms").as_double() == t.draft_ms && at(totals, "host_ms").as_double() == t.host_ms,
        "time totals round trip exactly");
    const auto& depths = at(totals, "draft_rows_by_depth").as_array();
    bool depths_equal = depths.size() == kCycleDraftDepths;
    for (std::size_t i = 0; depths_equal && i < depths.size(); ++i)
        depths_equal = static_cast<std::uint64_t>(depths[i].as_int()) == t.draft_rows_by_depth[i];
    check(depths_equal, "draft rows by depth round trip");
    const Json& fit = at(j, "fit");
    check(at(fit, "fixed_ms").as_double() == f.fixed_ms && at(fit, "row_ms").as_double() == f.row_ms &&
        at(fit, "lane_ms").as_double() == f.lane_ms && at(fit, "residual_sd").as_double() == f.residual_sd &&
        static_cast<std::uint64_t>(at(fit, "cycles").as_int()) == f.cycles && at(fit, "terms").as_int() == f.terms,
        "fit round trip");

    // The raw sums refit to the same coefficients, and two snapshots subtract.
    const Json& sums = at(j, "fit_sums");
    CycleLedger::FitSums s;
    s.n = at(sums, "n").as_double();
    s.r = at(sums, "r").as_double();
    s.l = at(sums, "l").as_double();
    s.rr = at(sums, "rr").as_double();
    s.rl = at(sums, "rl").as_double();
    s.ll = at(sums, "ll").as_double();
    s.y = at(sums, "y").as_double();
    s.ry = at(sums, "ry").as_double();
    s.ly = at(sums, "ly").as_double();
    s.yy = at(sums, "yy").as_double();
    const auto refit = CycleLedger::fit(s);
    check(refit.row_ms == f.row_ms && refit.fixed_ms == f.fixed_ms && refit.lane_ms == f.lane_ms, "fit sums refit");

    std::uint64_t bucket_cycles = 0, bucket_rows = 0;
    for (const Json& b : at(j, "by_lanes").as_array()) {
        const auto lanes = static_cast<std::size_t>(at(b, "lanes").as_int());
        const auto expect = ledger.lanes(lanes);
        check(static_cast<std::uint64_t>(at(b, "cycles").as_int()) == expect.cycles &&
            static_cast<std::uint64_t>(at(b, "emitted").as_int()) == expect.emitted &&
            at(b, "total_ms").as_double() == expect.total_ms, "lane table row round trip");
        bucket_cycles += static_cast<std::uint64_t>(at(b, "cycles").as_int());
        bucket_rows += static_cast<std::uint64_t>(at(b, "physical_rows").as_int());
    }
    check(bucket_cycles == t.cycles - t.prompt_cycles && bucket_rows <= t.physical_rows, "lane table covers the steady cycles");

    const auto& recent = at(j, "recent").as_array();
    const auto expect = ledger.recent(4);
    check(recent.size() == 4 && expect.size() == 4, "recent receipts are limited");
    for (std::size_t i = 0; i < recent.size() && i < expect.size(); ++i) {
        const Json& r = recent[i];
        check(static_cast<std::uint32_t>(at(r, "lanes").as_int()) == expect[i].lanes &&
            static_cast<std::uint32_t>(at(r, "physical_rows").as_int()) == expect[i].physical_rows &&
            at(r, "useful_rows").as_int() + at(r, "padding_rows").as_int() == at(r, "physical_rows").as_int() &&
            at(r, "total_ms").as_double() == expect[i].total_ms && at(r, "prompt_unit_ran").as_bool() == expect[i].prompt_unit_ran &&
            at(r, "begin_ns").as_int() == expect[i].begin_ns && at(r, "end_ns").as_int() == expect[i].end_ns,
            "recent receipt round trip");
    }
    check(at(parse_json(ledger.json()), "recent").as_array().empty(), "default JSON carries no receipts");
    check(at(parse_json(CycleLedger().json()), "cycles").as_int() == 0, "empty ledger serialises");
}

void log_line_format() {
    CycleReceipt r = receipt(4, 16, 13, 161.25);
    r.emitted = 14;
    r.draft_ms = 20.5;
    r.verify_ms = 138.0;
    r.host_ms = 2.75;
    r.prompt_unit_ran = true;
    r.begin_ns = 123;
    r.end_ns = 456;
    r.add_draft_depth(2, 4);
    r.admitted = 6;
    r.forwards = 2;
    r.solo[static_cast<std::size_t>(CycleSoloReason::no_rows)] = 1;
    check(format_cycle_receipt(r, 7) ==
        "serve cycle seq=7 lanes=4 rows=16 useful=13 pad=3 emitted=14 draft_rows=0/4/0/0/0/0/0/0 draft_ms=20.500000 "
        "verify_ms=138.000000 host_ms=2.750000 ms=161.250000 prompt=1 begin_ns=123 end_ns=456 admitted=6 forwards=2 "
        "solo=0/1/0/0/0/0",
        "log line format");
    check(std::string(cycle_solo_reason_name(1)) == "no_rows" && std::string(cycle_solo_reason_name(99)) == "unknown",
        "solo reason names");
    CycleLedger ledger(4);
    check(ledger.add(r), "receipt with forwards accepted");
    const auto j = parse_json(ledger.json());
    const auto& totals = at(j, "totals");
    check(at(totals, "forwards").as_int() == 2 && at(totals, "admitted").as_int() == 6, "ledger totals carry admitted and forwards");
    const auto& solo = at(totals, "solo").as_array();
    check(solo.size() == kCycleSoloReasons && solo[1].as_int() == 1, "ledger totals carry solo reasons");
}
} // namespace


// A window of eight requests like a load test: serial prompt units, a late lane, a full cohort, a tail.
void run_window_partition_and_rates() {
    using strix::serve::RunWindow;
    RunWindow w;
    check(!w.open(), "run closed at start");
    const std::int64_t ms = 1000000;
    std::int64_t t = 5 * ms;
    for (int i = 0; i < 8; ++i) w.request(t);
    check(w.open(), "run opens with the first request");
    for (int i = 0; i < 7; ++i) {  // seven prompts before any decode step: one first token each
        w.admission(t, t + 150 * ms, 20, 1);
        t += 150 * ms;
        if (i) w.hold();
    }
    w.host_control(2 * ms);
    t += 2 * ms;
    w.step(t, t + 54 * ms, 7, 7, 7);  // the eighth lane is late: a seven-lane step is not the cohort
    t += 54 * ms;
    w.admission(t, t + 200 * ms, 46, 1);
    t += 200 * ms;
    for (int i = 0; i < 10; ++i) {
        w.step(t, t + 74 * ms, 8, 8, 8);
        t += 74 * ms;
        w.host_control(ms / 10);
        t += ms / 10;
    }
    const std::int64_t last_full = t - ms / 10;
    w.step(t, t + 29 * ms, 1, 1, 1);  // tail: one lane left
    t += 29 * ms + 3 * ms;            // 3 ms nobody claimed
    const auto r = w.close(t);
    check(!w.open(), "run closes");
    check(r.consistent(), "run consistent");
    check(r.requests == 8 && r.peak_lanes == 8 && r.prompt_units == 8 && r.prompt_tokens == 186, "run counts");
    check(r.holds == 6, "run holds");
    check(r.admission_outputs == 8 && r.ramp.outputs == 7 && r.full.outputs == 80 && r.partial.outputs == 1, "run outputs by class");
    check(r.outputs() == 96, "run outputs");
    check(r.ramp.steps == 1 && r.full.steps == 10 && r.partial.steps == 1, "run steps by class");
    check(r.admission_ns == 1250 * ms && r.ramp.ns == 54 * ms && r.full.ns == 740 * ms && r.partial.ns == 29 * ms, "run interval sums");
    check(r.host_control_ns == 3 * ms && r.other_ns() == 3 * ms, "run host control and unclaimed time");
    check(r.wall_ns() == r.admission_ns + r.host_control_ns + r.ramp.ns + r.full.ns + r.partial.ns + r.other_ns(), "run partition");
    check(r.last_full_ns == last_full && r.tail_ns() == 32 * ms + ms / 10, "run tail");
    check(r.first_output_ns - r.begin_ns == 150 * ms, "run first output");
    // the steady rate comes from the full-cohort steps only; the whole-run rate from the window
    check(near(r.steady_rate(), 80.0 / 0.740, 1e-9), "run steady rate");
    check(near(r.whole_rate(), 96.0 * 1e9 / static_cast<double>(r.wall_ns()), 1e-9), "run whole rate");
    check(r.whole_rate() < r.steady_rate(), "whole-run rate below the steady rate");
    const std::string line = strix::serve::format_run_receipt(r, "pack=p draft=on");
    check(line.rfind("serve run requests=8 cancelled=0 peak_lanes=8 outputs=96 ", 0) == 0, "run line head");
    check(line.find(" full_steps=10 full_outputs=80 full_rows=80 full_ms=740.000") != std::string::npos, "run line full class");
    check(line.find(" admission_ms=1250.000 holds=6") != std::string::npos, "run line admission");
    check(line.size() > 16 && line.compare(line.size() - 15, 15, "pack=p draft=on") == 0, "run line identity");
}

// Cancelled lanes, held proposals (a step that commits nothing) and a first event with several tokens.
void run_window_edges() {
    using strix::serve::RunWindow;
    const std::int64_t ms = 1000000;
    RunWindow w;
    w.step(0, ms, 2, 2, 2);  // nothing is open: ignored
    w.admission(0, ms, 4, 1);
    w.request(10 * ms);
    w.request(10 * ms);
    w.admission(10 * ms, 30 * ms, 100, 3);  // the first event carries three tokens (a prompt-lookup window)
    w.admission(30 * ms, 40 * ms, 50, 0);   // a piece of a resumable prefill: no token yet
    w.step(40 * ms, 60 * ms, 2, 6, 5);      // drafted rows: six rows, five tokens committed
    w.step(60 * ms, 80 * ms, 2, 2, 0);      // every proposal held: time counts, no output
    w.cancel();
    w.step(80 * ms, 95 * ms, 1, 1, 1);      // partial cohort after the cancel
    w.step(95 * ms, 90 * ms, 1, 1, 1);      // a clock that runs backwards is refused
    w.step(95 * ms, 96 * ms, 0, 0, 0);      // no lane
    const auto r = w.close(100 * ms);
    check(r.consistent(), "edges consistent");
    check(r.requests == 2 && r.cancelled == 1 && r.peak_lanes == 2, "edges counts");
    check(r.admission_outputs == 3 && r.prompt_units == 2 && r.prompt_tokens == 150, "edges first event of three tokens");
    check(r.full.steps == 2 && r.full.outputs == 5 && r.full.rows == 8 && r.full.ns == 40 * ms, "edges held step counts its time");
    check(r.partial.steps == 1 && r.partial.outputs == 1 && r.ramp.steps == 0, "edges partial cohort");
    check(near(r.steady_rate(), 5.0 / 0.040, 1e-9), "edges steady rate is not inferred from the partial step");
    check(r.outputs() == 9 && r.other_ns() == 5 * ms, "edges outputs and unclaimed time");
    check(r.last_output_ns == 95 * ms && r.first_output_ns == 30 * ms, "edges output stamps");
    // a second window starts clean; the digest depends on lane and token order
    w.request(200 * ms);
    w.token(0, 7), w.token(1, 9);
    const auto a = w.close(201 * ms);
    w.request(300 * ms);
    w.token(1, 9), w.token(0, 7);
    const auto b = w.close(301 * ms);
    check(a.requests == 1 && a.outputs() == 0 && a.full.steps == 0 && a.tail_ns() == 0, "second window is clean");
    check(a.digest != b.digest && a.digest != strix::serve::RunReceipt{}.digest, "digest orders lanes and tokens");
    check(a.steady_rate() == 0.0 && a.whole_rate() == 0.0, "empty window has no rate");
}
int main() {
#if defined(_MSC_VER)
    // A failed assertion or abort must fail the test, never open a dialog.
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif
    try {
        fit_recovers_planted_coefficients();
        fit_degenerate_inputs();
        rows_identity_and_totals();
        json_round_trip();
        log_line_format();
        run_window_partition_and_rates();
        run_window_edges();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: exception: " << e.what() << '\n';
        return 1;
    }
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "cycle receipt tests passed\n";
    return 0;
}
