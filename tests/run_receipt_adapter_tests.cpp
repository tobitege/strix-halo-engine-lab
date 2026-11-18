// Host regressions for the shared receipt adapter and the production row planner.
#include "strix/serve/cycle_receipt.hpp"
#include "strix/deepseek41_batch_schedule.hpp"

#include <iostream>
#include <string_view>

using namespace strix::serve;
using strix::deepseek41::BatchSequence;
namespace {
int failures = 0;
void check(bool ok, const char* label) {
    if (!ok) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
constexpr std::int64_t ms = 1000000;

struct Lane {
    BatchSequence seq;
    RunTokenCursor cursor;
    std::size_t published = 0, max_tokens = 32;
    std::vector<std::uint32_t> stops;
    std::uint64_t commit(RunWindow& w) {
        return cursor.commit(w, seq.slot, seq.generated, published, max_tokens, stops);
    }
    void publish() {
        seq.generated.resize(retained_output_end(seq.generated, published, max_tokens, stops));
        published = seq.generated.size();
    }
};

void stop_inside_window() {
    for (const bool stop : {false, true}) {
        RunWindow w, expected;
        w.request(ms); expected.request(ms);
        Lane lane;
        lane.seq.generated = {9, 10, 11}; // verified speculative window
        if (stop) lane.stops = {9};
        else lane.max_tokens = 1;
        w.step(ms, 11 * ms, 1, 3, lane.commit(w));
        lane.publish(); // may close the final request immediately
        const auto r = w.close(11 * ms);
        expected.token(0, 9);
        check(lane.seq.generated == std::vector<std::uint32_t>{9}, "publication keeps exactly the first token");
        check(r.outputs() == lane.published && r.full.outputs == 1, "discarded tokens excluded from counts");
        check(r.digest == expected.close(11 * ms).digest, "discarded tokens excluded from digest");
        check(r.full.ns == 10 * ms && r.whole_rate() == 100 && r.steady_rate() == 100, "full forward interval retained");
    }
    Lane lane;
    RunWindow w;
    w.request(ms);
    lane.seq.generated = {7};
    w.restored(ms, lane.commit(w)); lane.publish();
    lane.stops = {9}; lane.seq.generated.insert(lane.seq.generated.end(), {8, 9, 10});
    w.step(ms, 2 * ms, 1, 3, lane.commit(w)); lane.publish();
    check(w.close(2 * ms).outputs() == 3 && lane.published == 3, "stop bounds start after previously published output");
}

void restored_pending() {
    for (const bool waiting : {false, true})
        for (const std::size_t length : {1U, 3U}) {
            Lane lane;
            lane.max_tokens = length;
            RunWindow w, expected;
            w.request(ms); expected.request(ms);
            if (waiting) w.admission(ms, 2 * ms, 0, 0); // waiting prefill replaced by imported state
            lane.seq.generated.assign(1, 41);
            w.restored(2 * ms, lane.commit(w));
            w.restored(2 * ms, lane.commit(w)); // repeated notification must not duplicate a pending token
            expected.token(0, 41);
            lane.publish();
            if (length > 1) {
                lane.seq.generated.insert(lane.seq.generated.end(), {42, 43});
                w.step(2 * ms, 4 * ms, 1, 2, lane.commit(w));
                expected.token(0, 42); expected.token(0, 43);
                lane.publish();
            }
            const auto r = w.close(5 * ms);
            check(r.outputs() == length && r.admission_outputs == 1, "restored first output counted once");
            check(r.first_output_ns == 2 * ms && r.last_output_ns == (length == 1 ? 2 : 4) * ms, "restored timestamps");
            check(r.digest == expected.close(5 * ms).digest, "restored token present in digest");
            check(r.prompt_tokens == 0 && r.prompt_units == (waiting ? 1U : 0U), "restored token invents no local prompt work");
            check(r.consistent() && r.whole_rate() > 0, "restored whole rate");
        }
}

void row_plan() {
    std::array<Lane, 4> lanes;
    std::vector<BatchSequence*> seqs;
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        auto& s = lanes[i].seq;
        s.slot = static_cast<std::uint32_t>(i);
        s.prompt.assign(20, 7); s.position = 1; s.fed = 1;
        seqs.push_back(&s);
    }
    RunWindow w;
    for (const auto& lane : lanes) { (void)lane; w.request(ms); }
    const auto step = [&](std::int64_t begin, std::int64_t end, std::uint32_t max_rows) {
        const auto plan = strix::deepseek41::plan_step(seqs, 8, max_rows);
        const auto work = run_step_work(plan.owner, [&](std::uint32_t i) { return seqs[i]->reading_prompt(); });
        strix::deepseek41::apply_step(seqs, plan, std::vector<std::uint32_t>(plan.rows.size(), 42));
        std::uint64_t emitted = 0;
        for (auto& lane : lanes) emitted += lane.commit(w);
        w.planned_step(begin, end, work, emitted);
        for (auto& lane : lanes) lane.publish();
        return work;
    };
    const auto prompt = step(ms, 3 * ms, 8);
    check(prompt.prompt_rows == 8 && prompt.decode_lanes == 0 && lanes[1].seq.fed == 1, "only planned prompt owner participates");
    lanes[0].seq.fed = 19; // next row finishes prompt, still classified from PRE-step state
    const auto final_prompt = step(3 * ms, 5 * ms, 1);
    check(final_prompt.prompt_rows == 1 && final_prompt.decode_lanes == 0, "last prompt row remains admission");
    const auto mixed = step(5 * ms, 8 * ms, 8);
    check(mixed.prompt_rows == 7 && mixed.decode_lanes == 1, "actual mixed plan counts owners");
    const auto decode = step(8 * ms, 10 * ms, 1); // other three requests still reading prompts
    check(decode.decode_lanes == 1 && decode.prompt_rows == 0, "unplanned requests excluded from decode cohort");
    const auto r = w.close(10 * ms);
    check(r.prompt_tokens == 16 && r.admission_ns == 4 * ms && r.admission_outputs == 1, "row prompt accounting");
    check(r.mixed.steps == 1 && r.mixed.outputs == 1 && r.mixed.ns == 3 * ms, "mixed duration and outputs explicit");
    check(r.peak_lanes == 1 && r.full.steps == 1 && r.full.ns == 2 * ms, "mixed and prompt steps never qualify steady");
    check(r.outputs() == 3 && r.consistent() && r.other_ns() == 0, "row accounting partitions all time and outputs");
    const std::vector<std::uint32_t> owners{0, 0, 0};
    check(run_step_work(owners, [](std::uint32_t) { return false; }).decode_lanes == 1, "speculative rows count one decode lane");
    check(format_run_receipt(r).find("steady_basis=forward_only") != std::string::npos, "rate basis explicit");
}

void prefill_digest() {
    const auto receipt = [](std::uint32_t token, bool resumable) {
        RunWindow w;
        Lane lane;
        lane.max_tokens = 1;
        w.request(ms);
        if (resumable) w.admission(ms, 2 * ms, 0, lane.commit(w));
        lane.seq.generated = {token};
        w.admission(resumable ? 2 * ms : ms, 4 * ms, 20, lane.commit(w));
        lane.publish();
        return w.close(4 * ms);
    };
    for (const bool resumable : {false, true}) {
        const auto a = receipt(17, resumable), b = receipt(18, resumable);
        check(a.outputs() == 1 && a.admission_outputs == 1 && a.prompt_tokens == 20, "prefill-only count");
        check(a.digest != b.digest && a.digest != RunReceipt{}.digest, "first prefill token changes digest");
        check(a.first_output_ns == 4 * ms && a.last_output_ns == 4 * ms, "prefill-only timestamps");
    }
    check(receipt(17, false).digest == receipt(17, true).digest, "resumable completion uses common commit order");
}
} // namespace

int main(int argc, char** argv) {
    const std::string_view which = argc == 2 ? argv[1] : "";
    if (which == "stop") stop_inside_window();
    else if (which == "restored") restored_pending();
    else if (which == "rows") row_plan();
    else if (which == "prefill") prefill_digest();
    else return 2;
    return failures ? 1 : 0;
}
