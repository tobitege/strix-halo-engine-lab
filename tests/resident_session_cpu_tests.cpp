// Host tests for the resident runner and session protocol (include/strix/serve/resident_session.hpp) with a fake
// single-node engine: arm switching through the real process environment, reload-only arms, receipts with logits
// digests, plan budget, and failure isolation between jobs of one session. No GPU, no model.
#include "strix/serve/resident_session.hpp"
#include "strix/env_flags.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

namespace rp = strix::serve::resident;
namespace fs = std::filesystem;
using strix::serve::Json;

void write_file(const fs::path& p, const std::string& s) {
    std::ofstream f(p, std::ios::binary);
    f << s;
}

void write_u32(const fs::path& p, const std::vector<std::uint32_t>& v) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
}

Json read_json(const fs::path& p) { return strix::serve::parse_json(rp::read_text_file(p)); }

struct FakeEngine {
    std::vector<std::string> seen;  // "<FAKE_RT value>" per run
    int resets = 0;
    int levers = 0;
    double sleep_s = 0.0;
    rp::Hooks hooks() {
        rp::Hooks h;
        h.engine_names = {"fake"};
        h.levers = {{"FAKE_RT", rp::LeverClass::runtime}, {"FAKE_RELOAD", rp::LeverClass::reload}};
        h.capacity = 64;
        h.apply_levers = [this] { ++levers; };
        h.reset = [this] { ++resets; };
        h.run = [this](const std::vector<std::uint32_t>& prompt, unsigned steps, const rp::LogitsSink* sink, bool all) {
            const char* v = std::getenv("FAKE_RT");
            seen.emplace_back(v ? v : "-");
            if (sleep_s > 0) std::this_thread::sleep_for(std::chrono::duration<double>(sleep_s));
            rp::RunIo io;
            const float rt = v ? 1.0f : 0.0f;
            std::vector<float> row = {static_cast<float>(prompt.size()), rt, 0.5f};
            if (sink) (*sink)(row.data(), row.size());
            for (unsigned i = 0; i < steps; ++i) {
                io.tokens.push_back(prompt.back() + i);
                if (i > 0 && all && sink) (*sink)(row.data(), row.size());
            }
            io.prefill_s = 0.010;
            io.decode_s = 0.001 * steps;
            return io;
        };
        h.state_digest = [](std::size_t n) { return rp::hex64(n); };
        return h;
    }
};

std::string plan_json(const fs::path& dir, const std::string& engine) {
    return R"({"engine":")" + engine + R"(","order":"abab","repeats":2,"warmup":1,"steps":4,"logits_last_n":2,
      "capture":["timing","tokens","state-digest"],
      "prompts":[{"name":"p1","path":")" + (dir / "p1.u32").generic_string() + R"("},
                 {"name":"big","path":")" + (dir / "big.u32").generic_string() + R"("}],
      "arms":[{"name":"A","env":{},"prompts":["p1"],"capture":["timing","tokens","logits"]},
              {"name":"B","env":{"FAKE_RT":"1"},"prompts":["p1"]},
              {"name":"C","env":{"FAKE_RELOAD":"1"},"prompts":["p1"]}]})";
}

std::string job_json(const std::string& id, const std::string& plan, const fs::path& out, double budget) {
    return R"({"id":")" + id + R"(","plan":)" + plan + R"(,"out":")" + out.generic_string() + R"(","budget_s":)" +
           std::to_string(budget) + "}";
}

void test_sha256() {
    CHECK(rp::sha256_hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(rp::sha256_hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    const std::string m(1000, 'a');  // multi-block
    CHECK(rp::sha256_hex(m.data(), m.size()) == "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
    CHECK(rp::fnv1a64_bytes("a", 1) == rp::fnv1a64("a"));
}

void test_parse_job() {
    const std::string plan = R"({"engine":"fake","steps":1,"prompts":[{"name":"p","path":"/x"}],"arms":[{"name":"a"}]})";
    const rp::Job j = rp::parse_job(R"({"id":"me-1","plan":)" + plan + R"(,"out":"/abs/out","budget_s":30})");
    CHECK(j.id == "me-1");
    CHECK(j.budget_s == 30.0);
    CHECK(rp::parse_job(R"({"id":"a","plan":)" + plan + R"(,"out":"/o"})").budget_s == rp::kDefaultBudgetS);
    CHECK(rp::parse_job(R"({"id":"a","plan":)" + plan + R"(,"out":"/o","budget_s":5000})").budget_s == rp::kMaxBudgetS);
    CHECK(rp::parse_plan(j.plan_text).engine == "fake");
    bool threw = false;
    try {
        (void)rp::parse_job(R"({"id":"x","plan":{},"out":"relative"})");
    } catch (const rp::PlanError&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        (void)rp::parse_job(R"({"id":"x","plan":{},"out":"/a","extra":1})");
    } catch (const rp::PlanError&) {
        threw = true;
    }
    CHECK(threw);
}

void test_session(const fs::path& root) {
    const fs::path dir = root / "session";
    fs::create_directories(dir);
    write_u32(root / "p1.u32", {7, 8, 9});
    write_u32(root / "big.u32", std::vector<std::uint32_t>(100, 1));
    const std::string good = plan_json(root, "fake");
    // 1: good plan; 2: broken JSON; 3: another engine; 4: missing prompt; 5: plan budget; 6: capacity; then stop.
    write_file(dir / "job-1.json", job_json("alice-1", good, root / "out1", 0));
    write_file(dir / "job-2.json", "{not json");
    write_file(dir / "job-3.json", job_json("bob-1", plan_json(root, "other"), root / "out3", 0));
    std::string missing = good;
    missing.replace(missing.find("p1.u32"), 6, "zz.u32");
    write_file(dir / "job-4.json", job_json("bob-2", missing, root / "out4", 0));
    write_file(dir / "job-5.json", job_json("carol-1", good, root / "out5", 0.03));
    std::string big = good;
    const std::string from = R"("prompts":["p1"],"capture")";
    big.replace(big.find(from), from.size(), R"("prompts":["big"],"capture")");
    write_file(dir / "job-6.json", job_json("dave-1", big, root / "out6", 0));
    write_file(dir / "stop-7", "");

    FakeEngine eng;
    rp::Hooks h = eng.hooks();
    rp::SessionOptions so;
    so.poll_s = 0.001;
    so.idle_s = 5.0;
    so.load_s = 12.5;
    // Job 5 sleeps per run so its 30 ms budget ends it after a few slots.
    std::size_t served = 0;
    {
        // Jobs 1-4 and 6 run fast; switch the sleep on only for job 5 by watching the done files is not possible in
        // one thread, so every run sleeps a little: jobs without a budget are unaffected.
        eng.sleep_s = 0.004;
        served = rp::run_session(dir, h, so);
    }
    CHECK(served == 6);
    CHECK(fs::exists(dir / "ready.r0"));
    CHECK(read_json(dir / "ready.r0").find("load_s")->as_double() == 12.5);

    // Job 1: A and B resident, C reported as needs reload; env switched per arm and restored afterwards.
    const Json s1 = read_json(dir / "done-1.r0");
    CHECK(s1.find("state")->as_string() == "ok");
    CHECK(s1.find("id")->as_string() == "alice-1");
    CHECK(s1.find("needs_reload")->size() == 1);
    CHECK(s1.find("needs_reload")->as_array()[0].as_string() == "FAKE_RELOAD=1");
    CHECK(fs::exists(root / "out1" / "summary.json"));
    CHECK(fs::exists(root / "out1" / "status.json"));
    CHECK(std::getenv("FAKE_RT") == nullptr);
    std::size_t a_runs = 0, b_runs = 0, c_err = 0;
    for (const auto& e : fs::directory_iterator(root / "out1")) {
        const std::string name = e.path().filename().string();
        if (name.rfind("run-", 0) != 0) continue;
        const Json r = read_json(e.path());
        const std::string arm = r.find("arm")->as_string();
        if (arm == "C") {
            CHECK(r.find("error") && r.find("error")->as_string() == "needs reload: FAKE_RELOAD=1");
            ++c_err;
            continue;
        }
        CHECK(r.find("logits_fnv") != nullptr);
        CHECK(r.find("logits_sha256") != nullptr && r.find("logits_sha256")->as_string().size() == 64);
        CHECK(r.find("n_tokens")->as_int() == 4);
        if (arm == "A") {
            ++a_runs;
            CHECK(r.find("logits_path") != nullptr);
            if (r.find("logits_path")) CHECK(fs::file_size(root / "out1" / r.find("logits_path")->as_string()) == 2 * 3 * 4);
        } else {
            ++b_runs;
            CHECK(r.find("state_digest") && r.find("state_digest")->as_string() == rp::hex64(3));
        }
    }
    CHECK(a_runs == 4 && b_runs == 4);  // 2 passes x (1 warmup + 1 run) each
    CHECK(c_err == 3);  // its own reload group: 1 warmup + 2 passes
    // The fake engine saw FAKE_RT unset for A, "1" for B; the first 4 runs of the session are A, A, B, B.
    CHECK(eng.seen.size() >= 4);
    if (eng.seen.size() >= 4) CHECK(eng.seen[0] == "-" && eng.seen[1] == "-" && eng.seen[2] == "1" && eng.seen[3] == "1");

    // Job 2: broken JSON -> error, the session went on.
    const Json s2 = read_json(dir / "done-2.r0");
    CHECK(s2.find("state")->as_string() == "error");
    CHECK(s2.find("error")->as_string().find("invalid JSON") != std::string::npos);
    // Job 3: engine not served.
    CHECK(read_json(dir / "done-3.r0").find("state")->as_string() == "error");
    CHECK(fs::exists(root / "out3" / "status.json"));
    // Job 4: the prompt file is missing -> every slot gets an error receipt, the plan itself completes.
    const Json s4 = read_json(dir / "done-4.r0");
    CHECK(s4.find("state")->as_string() == "ok");
    CHECK(s4.find("run_errors")->as_int() == s4.find("runs")->as_int());
    // Job 5: 30 ms budget, ~4 ms per run -> stopped before the end.
    const Json s5 = read_json(dir / "done-5.r0");
    CHECK(s5.find("state")->as_string() == "over-budget");
    CHECK(s5.find("arm_s")->find("A") != nullptr);
    CHECK(s5.find("skipped")->as_int() > 0);
    // Job 6: arm A's prompt (100 tokens + 4 steps) exceeds the capacity 64 -> error receipts for A only.
    const Json s6 = read_json(dir / "done-6.r0");
    CHECK(s6.find("state")->as_string() == "ok");
    CHECK(s6.find("run_errors")->as_int() == 4 + 3);  // A (capacity) + C (reload)
    CHECK(eng.resets > 0 && eng.levers > 0);
}

std::vector<Json> run_receipts(const fs::path& out) {
    std::vector<Json> receipts;
    for (const auto& entry : fs::directory_iterator(out))
        if (entry.path().filename().string().rfind("run-", 0) == 0) receipts.push_back(read_json(entry.path()));
    return receipts;
}

rp::Plan batch_plan(const fs::path& root, std::string_view extra = {}) {
    return rp::parse_plan(R"({"engine":"fake","steps":4,"repeats":2,"warmup":1,"concurrency":2,
        "batch_mode":"continuous","mtp":{"depth":3,"rows":8},
        "requests":[{"prompt":0,"seed":7},{"prompt":1,"max_tokens":2}],
        "prompts":[{"name":"p","path":")" + (root / "batch-p.u32").generic_string() +
        R"("},{"name":"q","path":")" + (root / "batch-q.u32").generic_string() +
        R"("}],"arms":[{"name":"A","prompts":["p"]},{"name":"B","env":{"FAKE_RT":"1"}}])" +
        std::string(extra) + "}");
}

rp::BatchIo fake_batch(const rp::BatchSpec& spec, const std::vector<std::vector<std::uint32_t>>& prompts) {
    rp::BatchIo io;
    for (const auto& r : spec.requests) {
        std::vector<std::uint32_t> tokens;
        for (std::uint32_t i = 0; i < r.max_tokens; ++i) tokens.push_back(prompts[r.prompt].back() + i);
        io.tokens.push_back(std::move(tokens));
        io.ttft_s.push_back(0.01);
    }
    for (std::uint32_t i = 0; i < 65; ++i)
        io.cycles.push_back({2, 1, 2, static_cast<double>(i + 1), i == 0 ? -1 : 4});
    io.prefill_s = 0.01;
    io.decode_s = 0.02;
    io.note = "fake batch";
    return io;
}

void test_batch_match(const fs::path& root, bool perturb) {
    const auto plan = batch_plan(root);
    const auto out = root / (perturb ? "batch-mismatch" : "batch-match");
    FakeEngine eng;
    auto h = eng.hooks();
    std::size_t batches = 0, barriers = 0, agreements = 0;
    const auto solo_run = h.run;
    h.run = [solo_run](const std::vector<std::uint32_t>& prompt, unsigned steps,
                       const rp::LogitsSink* sink, bool all) {
        CHECK(sink == nullptr && !all);
        return solo_run(prompt, steps, sink, all);
    };
    h.world = 2;
    h.agree = [&](const std::array<std::uint32_t, 8>& f) {
        if (f[1] == rp::kTagBatch) {
            ++agreements;
            CHECK(f[2] == 11 && f[3] == 0); // prompt + generation positions, including repeated prompts
            CHECK(f[4] == 2 && f[5] == 1 && f[6] == 3 && f[7] == 8);
        }
        return f[0] != 0;
    };
    h.barrier = [&](std::uint64_t) { ++barriers; };
    h.run_batch = [&](const rp::BatchSpec& spec, const std::vector<std::vector<std::uint32_t>>& prompts) {
        ++batches;
        CHECK(spec.requests.size() == 2 && spec.requests[0].seed == 7 && spec.requests[1].max_tokens == 2);
        CHECK(spec.mode == rp::BatchMode::Continuous && spec.mtp.depth == 3 && spec.mtp.rows == 8);
        auto io = fake_batch(spec, prompts);
        if (perturb) ++io.tokens[1][0];
        return io;
    };
    const auto outcome = rp::run_plan(plan, out, 0, h);
    CHECK(outcome.state == "ok" && outcome.errors == 0 && outcome.runs == 8 && outcome.slots == 8);
    CHECK(batches == 8 && agreements == 8 && barriers == 24);
    CHECK(eng.seen.size() == 16 && eng.resets == 25);
    CHECK(std::getenv("FAKE_RT") == nullptr);
    const auto receipts = run_receipts(out);
    CHECK(receipts.size() == 8);
    std::size_t warmups = 0;
    for (const auto& receipt : receipts) {
        if (receipt.find("warmup")->as_bool()) ++warmups;
        const auto& b = *receipt.find("batch");
        CHECK(b.find("status")->as_string() == "ok" && b.find("mode")->as_string() == "continuous");
        CHECK(b.find("n")->as_int() == 2 && b.find("mtp")->find("depth")->as_int() == 3);
        CHECK(b.find("all_match_solo")->as_bool() == !perturb);
        const auto& requests = b.find("requests")->as_array();
        CHECK(requests[0].find("match_solo")->as_bool());
        CHECK(requests[1].find("match_solo")->as_bool() == !perturb);
        CHECK(requests[0].find("tokens_fnv")->as_string() == rp::batch_tokens_fnv({9, 10, 11, 12}));
        CHECK(requests[0].find("solo_tokens_fnv")->as_string() == requests[0].find("tokens_fnv")->as_string());
        CHECK(requests[1].find("n_tokens")->as_int() == 2 && requests[0].find("seed")->as_int() == 7);
        const auto& c = *b.find("cycles");
        CHECK(c.find("n")->as_int() == 65 && c.find("rows_sum")->as_int() == 130);
        CHECK(c.find("useful_rows_sum")->as_int() == 65 && c.find("lanes_max")->as_int() == 2);
        CHECK(c.find("ms_mean")->as_double() == 33 && c.find("ms_p50")->as_double() == 33);
        CHECK(c.find("expert_union_mean")->as_double() == 4 && c.find("detail")->size() == 64);
        CHECK(c.find("detail")->as_array()[0].find("expert_union")->as_int() == -1);
        CHECK(std::abs(b.find("agg_tok_s")->as_double() - 200) < 1e-8);
        CHECK(std::abs(b.find("decode_tok_s")->as_double() - 200) < 1e-8);
        CHECK(std::abs(b.find("solo")->find("total_s")->as_double() - 0.026) < 1e-8);
        CHECK(b.find("note")->as_string() == "fake batch");
    }
    CHECK(warmups == 4);
}

void test_batch_unsupported(const fs::path& root, bool returned) {
    FakeEngine eng;
    auto h = eng.hooks();
    std::size_t batches = 0;
    if (returned) h.run_batch = [&](const rp::BatchSpec&, const std::vector<std::vector<std::uint32_t>>&) {
        ++batches;
        rp::BatchIo io;
        io.unsupported = "cohort limit";
        return io;
    };
    const auto out = root / (returned ? "batch-limit" : "batch-absent");
    const auto outcome = rp::run_plan(batch_plan(root), out, 0, h);
    CHECK(outcome.state == "ok" && outcome.errors == 0 && outcome.runs == 8);
    CHECK(eng.seen.size() == 16 && batches == (returned ? 8U : 0U));
    for (const auto& receipt : run_receipts(out)) {
        const auto& b = *receipt.find("batch");
        CHECK(b.size() == 2 && b.find("status")->as_string() == "unsupported");
        CHECK(b.find("reason")->as_string() == (returned ? "cohort limit" : "engine has no batched path"));
    }
}

void test_batch_budget_and_no_solo(const fs::path& root) {
    for (const bool budget : {false, true}) {
        FakeEngine eng;
        auto h = eng.hooks();
        std::size_t batches = 0;
        h.run_batch = [&](const rp::BatchSpec& spec, const std::vector<std::vector<std::uint32_t>>& prompts) {
            ++batches;
            if (budget) std::this_thread::sleep_for(std::chrono::milliseconds(120));
            auto io = fake_batch(spec, prompts);
            io.cycles.clear();
            return io;
        };
        const auto out = root / (budget ? "batch-budget" : "batch-no-solo");
        const auto plan = batch_plan(root, budget ? "" : R"(,"solo_control":false)");
        const auto outcome = rp::run_plan(plan, out, budget ? 0.1 : 0, h);
        CHECK(outcome.state == (budget ? "over-budget" : "ok"));
        CHECK(eng.seen.empty());
        CHECK(batches == (budget ? 1U : 8U));
        for (const auto& receipt : run_receipts(out)) {
            const auto& b = *receipt.find("batch");
            CHECK(!b.find("all_match_solo")->as_bool());
            CHECK(b.find("requests")->as_array()[0].find("match_solo")->is_null());
            CHECK(b.find("cycles")->find("expert_union_mean")->is_null());
            CHECK(b.find("cycles")->find("ms_p50")->as_double() == 0);
        }
    }
}

void test_batch_defaults(const fs::path& root) {
    auto plan = batch_plan(root);
    plan.requests.reset();
    plan.batch_mode.reset();
    plan.mtp.reset();
    plan.concurrency = 3;
    const auto spec = rp::batch_spec(plan);
    CHECK(spec.requests.size() == 3 && spec.requests[0].prompt == 0 && spec.requests[1].prompt == 1 &&
          spec.requests[2].prompt == 0);
    CHECK(spec.requests[2].max_tokens == 4 && spec.requests[2].seed == 0);
    CHECK(spec.mode == rp::BatchMode::Lockstep && spec.mtp.depth == 0 && spec.mtp.rows == 0);
    plan.prompts.resize(1);
    CHECK(rp::batch_spec(plan).requests[2].prompt == 0);
    const auto io = fake_batch(spec, {{9}, {20}});
    std::vector<std::optional<rp::RunIo>> solos(3);
    auto even_io = io;
    even_io.cycles = {{1, 1, 1, 2, -1}, {1, 1, 1, 4, -1}};
    const auto receipt = rp::batch_receipt(spec, even_io, solos);
    CHECK(receipt.find("cycles")->find("ms_p50")->as_double() == 3);
    CHECK(receipt.find("cycles")->find("expert_union_mean")->is_null());
}

void test_batch_request_budget(const fs::path& root) {
    FakeEngine eng;
    eng.sleep_s = 0.12;
    auto h = eng.hooks();
    h.run_batch = fake_batch;
    const auto out = root / "batch-request-budget";
    const auto outcome = rp::run_plan(batch_plan(root), out, 0.1, h);
    CHECK(outcome.state == "over-budget" && outcome.runs == 1 && outcome.skipped == 7);
    CHECK(eng.seen.size() == 1);
    const auto receipts = run_receipts(out);
    CHECK(receipts.size() == 1);
    const auto& b = *receipts[0].find("batch");
    CHECK(b.find("requests")->as_array()[0].find("match_solo")->as_bool());
    CHECK(b.find("requests")->as_array()[1].find("match_solo")->is_null());
    CHECK(!b.find("all_match_solo")->as_bool());
}

void test_serial_receipt_unchanged(const fs::path& root) {
    const auto plan = rp::parse_plan(R"({"engine":"fake","steps":4,"repeats":1,"warmup":0,
        "capture":["timing","tokens"],"prompts":[{"name":"p","path":")" +
        (root / "batch-p.u32").generic_string() + R"("}],"arms":[{"name":"A"}]})");
    FakeEngine eng;
    auto h = eng.hooks();
    h.run_batch = [](const rp::BatchSpec&, const std::vector<std::vector<std::uint32_t>>&) -> rp::BatchIo {
        CHECK(false); // The optional hook must never affect a serial plan.
        return {};
    };
    const auto out = root / "serial-unchanged";
    const auto outcome = rp::run_plan(plan, out, 0, h);
    CHECK(outcome.state == "ok" && outcome.runs == 1 && eng.seen.size() == 1);
    // Reconstruct the original serial path's deterministic record, including the logits anchors.
    const auto slots = rp::schedule(plan, rp::group_arms(plan, h.levers));
    rp::RunRecord expected{slots[0], {}};
    auto& r = expected.result;
    r.tokens = {9, 10, 11, 12};
    r.prefill_ms = 1000.0 * 0.010;
    r.ttft_ms = r.prefill_ms;
    r.prefill_tok_s = 3.0 / 0.010;
    r.decode_ms = 1000.0 * 0.004;
    r.decode_tok_s = 3.0 / 0.004;
    r.cold_run = true;  // the first run after the load
    const std::vector<float> row = {3.0f, 0.0f, 0.5f};
    r.logits_fnv = rp::hex64(rp::fnv1a64_bytes(row.data(), row.size() * sizeof(float)));
    r.logits_sha256 = rp::sha256_hex(row.data(), row.size() * sizeof(float));
    const std::string got = rp::read_text_file(out / (rp::receipt_name(plan, slots[0]) + rp::rank_suffix(0)));
    Json input = Json::object();
    input.set("request", Json(std::uint64_t{0}));
    input.set("n_tokens", Json(std::uint64_t{3}));
    input.set("tokens_fnv", Json(rp::hex64(rp::token_digest(std::vector<std::uint32_t>{7, 8, 9}))));
    r.prompt_inputs.push(std::move(input));
    // write_text_atomic terminates every file with '\n'.
    CHECK(got == rp::receipt_json(plan, expected).dump() + "\n");
    CHECK(!run_receipts(out)[0].find("batch"));
}

void test_idle_end(const fs::path& root) {
    const fs::path dir = root / "idle";
    fs::create_directories(dir);
    FakeEngine eng;
    rp::Hooks h = eng.hooks();
    rp::SessionOptions so;
    so.poll_s = 0.001;
    so.idle_s = 0.05;
    CHECK(rp::run_session(dir, h, so) == 0);
}

void test_input_agreement(const fs::path& root) {
    for (const bool batch : {false, true}) {
        auto p = rp::parse_plan(R"({"engine":"fake","steps":1,"warmup":0,"repeats":1,
            "prompts":[{"name":"p","path":"p"},{"name":"q","path":"q"}],"arms":[{"name":"A","prompts":["p"]}]})");
        if (batch) { p.concurrency = 2; p.solo_control = false; }
        FakeEngine eng;
        auto h = eng.hooks();
        h.world = 2;
        h.prompt_max = 2;
        h.read_prompt = [](const std::string&) { return std::vector<std::uint32_t>{1, 2, 3}; };
        h.run_batch = fake_batch;
        std::vector<std::array<std::uint32_t, 8>> transcript;
        h.agree = [&](const auto& fields) { transcript.push_back(fields); return true; };
        const auto out = root / (batch ? "input-batch" : "input-serial");
        CHECK(rp::run_plan(p, out, 0, h).errors == 0);
        const auto receipt = run_receipts(out).front();
        const auto* inputs = receipt.find("prompt_inputs");
        CHECK(inputs && inputs->size() == (batch ? 2U : 1U));
        if (inputs && inputs->size() > 0) {
            CHECK(inputs->as_array()[0].find("n_tokens")->as_int() == 2);
            CHECK(inputs->as_array()[0].find("tokens_fnv")->as_string() ==
                  rp::hex64(rp::token_digest(std::vector<std::uint32_t>{1, 2})));
        }
        for (int variant = 0; variant < 3; ++variant) {
            std::size_t index = 0;
            h.agree = [&](const auto& fields) {
                if (index >= transcript.size() || fields != transcript[index++])
                    throw std::runtime_error("peer inputs differ");
                return true;
            };
            h.read_prompt = [=](const std::string& path) {
                if (batch && path == "p") return std::vector<std::uint32_t>{1, 2, 3};
                if (variant == 0) return std::vector<std::uint32_t>{1, 9, 3}; // same count, stale content
                if (variant == 1) return std::vector<std::uint32_t>{1}; // different count
                return std::vector<std::uint32_t>{1, 2, 9}; // only discarded tail differs
            };
            int device_calls = 0;
            h.apply_levers = [&] { ++device_calls; };
            h.reset = [&] { ++device_calls; };
            h.barrier = [&](std::uint64_t) { ++device_calls; };
            bool failed = false;
            try { (void)rp::run_plan(p, out / std::to_string(variant), 0, h); }
            catch (const std::runtime_error&) { failed = true; }
            CHECK(failed == (variant != 2));
            if (variant != 2) CHECK(device_calls == 0);
        }
    }
}

void test_transaction_and_fatal_errors(const fs::path& root) {
    rp::detail::unset_env("FAKE_RT");
    auto p = rp::parse_plan(R"({"engine":"fake","steps":1,"warmup":0,"repeats":1,
        "prompts":[{"name":"p","path":"unused"}],"arms":[{"name":"A","env":{"FAKE_RT":"bad"}}]})");
    FakeEngine eng;
    auto h = eng.hooks();
    h.read_prompt = [](const std::string&) { return std::vector<std::uint32_t>{1}; };
    std::string object = "baseline";
    h.apply_levers = [&] {
        const char* v = strix::envcfg::text("FAKE_RT");
        object = v ? v : "baseline"; // deliberately partially mutate before failure
        if (object == "bad") throw std::invalid_argument("resolver failed");
    };
    bool fatal = false;
    try { (void)rp::run_plan(p, root / "apply-failed", 0, h); }
    catch (const rp::PlanError&) { CHECK(false); }
    catch (const std::runtime_error&) { fatal = true; }
    CHECK(fatal && object == "baseline" && strix::envcfg::text("FAKE_RT") == nullptr);

    h.validate_arm = [](const rp::Arm& a) {
        if (!a.env.empty()) throw std::invalid_argument("invalid arm before mutation");
    };
    const auto dir = root / "recover-validation";
    fs::create_directories(dir);
    const auto bad = R"({"engine":"fake","steps":1,"warmup":0,"repeats":1,
        "prompts":[{"name":"p","path":"unused"}],"arms":[{"name":"A","env":{"FAKE_RT":"bad"}}]})";
    const auto good = R"({"engine":"fake","steps":1,"warmup":0,"repeats":1,
        "prompts":[{"name":"p","path":"unused"}],"arms":[{"name":"A"}]})";
    write_file(dir / "job-1.json", job_json("bad", bad, dir / "bad", 0));
    write_file(dir / "job-2.json", job_json("good", good, dir / "good", 0));
    write_file(dir / "stop-3", "");
    CHECK(rp::run_session(dir, h, {}) == 2);
    CHECK(read_json(dir / "done-1.r0").find("state")->as_string() == "error");
    CHECK(read_json(dir / "done-2.r0").find("state")->as_string() == "ok");
    CHECK(eng.seen.size() == 1 && eng.seen.front() == "-");

    h.validate_arm = {};
    for (const bool plan_error : {false, true}) {
        const auto fatal_dir = root / (plan_error ? "fatal-plan-error" : "fatal-invalid-argument");
        fs::create_directories(fatal_dir);
        write_file(fatal_dir / "job-1.json", job_json("run", good, fatal_dir / "out", 0));
        write_file(fatal_dir / "job-2.json", job_json("must-not-run", good, fatal_dir / "next", 0));
        write_file(fatal_dir / "stop-3", "");
        int calls = 0;
        h.run = [&](const auto&, unsigned, const rp::LogitsSink*, bool) -> rp::RunIo {
            ++calls;
            if (plan_error) throw rp::PlanError("engine PlanError is also fatal");
            throw std::invalid_argument("engine failure");
        };
        fatal = false;
        try { (void)rp::run_session(fatal_dir, h, {}); }
        catch (const std::runtime_error&) { fatal = true; }
        CHECK(fatal && calls == 1 && !fs::exists(fatal_dir / "done-1.r0"));
        CHECK(object == "baseline" && strix::envcfg::text("FAKE_RT") == nullptr);
    }
}

void test_input_limits(const fs::path& root) {
    const auto path = root / "large-input";
    write_file(path, std::string(rp::kMaxPlanBytes + 1, ' '));
    bool refused = false;
    try { (void)rp::read_text_file(path); } catch (const rp::PlanError&) { refused = true; }
    CHECK(refused);
    // Sparse file: exercise the u32 limit without allocating a large test buffer.
    fs::resize_file(path, rp::kMaxPromptBytes + 4);
    refused = false;
    try { (void)rp::read_u32_file(path.string()); } catch (const std::runtime_error&) { refused = true; }
    CHECK(refused);
}

}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / ("resident-session-test-" + std::to_string(rp::detail::process_id()));
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    test_input_agreement(root);
    test_transaction_and_fatal_errors(root);
    test_input_limits(root);
    test_sha256();
    test_parse_job();
    write_u32(root / "batch-p.u32", {7, 8, 9});
    write_u32(root / "batch-q.u32", {19, 20});
    test_batch_match(root, false);
    test_batch_match(root, true);
    test_batch_unsupported(root, false);
    test_batch_unsupported(root, true);
    test_batch_budget_and_no_solo(root);
    test_batch_defaults(root);
    test_batch_request_budget(root);
    test_serial_receipt_unchanged(root);
    test_session(root);
    test_idle_end(root);
    fs::remove_all(root, ec);
    if (failures == 0) std::printf("resident_session_cpu_tests: ok\n");
    return failures != 0;
}
