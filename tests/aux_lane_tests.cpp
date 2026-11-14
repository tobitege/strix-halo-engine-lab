// Host-only tests for the aux lane: unit choice, the bounded queue, fallback, and the scorer and vision clients.
#include "strix/serve/aux_lane.hpp"
#include "strix/serve/aux_units.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

#ifdef _MSC_VER
#include <crtdbg.h>
#endif

using namespace strix::serve;
using namespace strix::serve::aux;

namespace {
int failures = 0;
void check(bool value, const char* label) {
    if (!value) {
        ++failures;
        std::cerr << "FAIL: " << label << '\n';
    }
}
template <class F> void fails_with(F f, AuxError::Code code, const char* label) {
    try {
        f();
    } catch (const AuxError& e) {
        check(e.code == code, label);
        return;
    } catch (const std::exception&) {
    }
    check(false, label);
}
template <class F> void invalid(F f, const char* label) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return;
    } catch (const std::exception&) {
    }
    check(false, label);
}
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    if (*value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

// A deterministic scorer: every output is a function of the text, the session and a per-instance salt.
struct FakeScorer final : pflash::PromptScorer {
    explicit FakeScorer(float salt_) : salt(salt_) {}
    pflash::ScoreBuffers score(std::string_view rendered, std::uint64_t session) override {
        ++calls;
        thread = std::this_thread::get_id();
        if (fail) throw std::runtime_error("scorer down");
        pflash::ScoreBuffers out;
        for (std::size_t i = 0; i < rendered.size(); ++i) {
            const auto c = static_cast<unsigned char>(rendered[i]);
            out.token_score.push_back(salt + static_cast<float>(c) / 7.0f + static_cast<float>(session));
            out.cut_prob.push_back(1.0f / (1.0f + static_cast<float>(c)));
            out.request_mask.push_back(static_cast<std::uint8_t>(c == ' ' ? 0 : (i % 2 ? 255 : 1)));
            out.token_byte_begin.push_back(static_cast<std::uint32_t>(i) + 0x7000'0000U); // above the float-exact range
        }
        return out;
    }
    float salt;
    bool fail = false;
    std::atomic<int> calls{0};
    std::thread::id thread;
};

bool same(const pflash::ScoreBuffers& a, const pflash::ScoreBuffers& b) {
    const auto bits = [](const std::vector<float>& x, const std::vector<float>& y) {
        return x.size() == y.size() && (x.empty() || std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0);
    };
    return bits(a.token_score, b.token_score) && bits(a.cut_prob, b.cut_prob) && a.request_mask == b.request_mask &&
           a.token_byte_begin == b.token_byte_begin;
}

Executor stub(UnitKind unit, std::string id, JobKind kind = JobKind::score, std::string identity = "id-a") {
    Executor e;
    e.unit = unit;
    e.unit_id = std::move(id);
    e.kind = kind;
    e.identity = std::move(identity);
    e.run = [](const AuxJob&) { return AuxResult{}; };
    return e;
}

Config three_units() {
    Config config;
    config.units = {{UnitKind::gpu, "main"}, {UnitKind::npu, "npu"}, {UnitKind::cpu, "host"}};
    return config;
}

void choice() {
    const std::vector<Executor> executors{stub(UnitKind::cpu, "host"), stub(UnitKind::npu, "npu", JobKind::score, "id-n"),
                                          stub(UnitKind::gpu, "main"), stub(UnitKind::gpu, "main", JobKind::vision)};
    auto config = three_units();
    using Order = std::vector<std::size_t>;
    check(plan(config, executors, JobKind::score, "", false) == Order({2, 1, 0}), "idle: gpu, npu, cpu");
    check(plan(config, executors, JobKind::score, "", true) == Order({2, 1, 0}), "busy without measured rates: the gpu keeps the job");
    auto& rates = config.rates[static_cast<std::size_t>(JobKind::score)];
    rates = {0.10, 0.50}; // the npu is 5x slower: below the 8.5x bound
    check(plan(config, executors, JobKind::score, "", true) == Order({1, 2, 0}), "busy and npu within the bound: npu first");
    check(plan(config, executors, JobKind::score, "", false) == Order({2, 1, 0}), "idle gpu: the npu never goes first");
    rates = {0.10, 1.00}; // 10x slower
    check(plan(config, executors, JobKind::score, "", true) == Order({2, 1, 0}), "busy and npu beyond the bound: gpu first");
    rates = {0.10, 0.50};
    config.npu_side_cost = 0.20; // bound 4.25x
    check(plan(config, executors, JobKind::score, "", true) == Order({2, 1, 0}), "a higher npu side cost lowers the bound");
    config.npu_side_cost = 0.10;
    config.order[0] = {UnitKind::npu, UnitKind::gpu, UnitKind::cpu};
    check(plan(config, executors, JobKind::score, "", false) == Order({2, 1, 0}), "measured rates beat a configured npu-first order when idle");
    rates = {};
    check(plan(config, executors, JobKind::score, "", false) == Order({1, 2, 0}), "without rates the configured order stands");
    config.order[0] = {UnitKind::gpu, UnitKind::npu, UnitKind::cpu};
    check(plan(config, executors, JobKind::score, "id-a", false) == Order({2, 0}), "identity filters the executors");
    check(plan(config, executors, JobKind::score, "id-x", false).empty(), "unknown identity: nothing");
    check(plan(config, executors, JobKind::vision, "", true) == Order({3}), "kinds do not mix");
    check(plan(config, executors, JobKind::embed, "", false).empty(), "no executor of the kind");
    config.force = UnitKind::cpu;
    check(plan(config, executors, JobKind::score, "", false) == Order({0}), "forced unit only");
    config.force = UnitKind::npu;
    check(plan(config, executors, JobKind::vision, "", false).empty(), "forced unit without an executor: nothing");

    set_env("STRIX_AUX_UNIT", "cpu");
    check(with_environment(Config{}).force == UnitKind::cpu, "environment forces a unit");
    set_env("STRIX_AUX_UNIT", "auto");
    check(!with_environment(config).force.has_value(), "auto clears the override");
    set_env("STRIX_AUX_UNIT", "tpu");
    invalid([] { (void)with_environment(Config{}); }, "bad STRIX_AUX_UNIT");
    set_env("STRIX_AUX_UNIT", "");
    check(!with_environment(Config{}).force.has_value(), "unset environment changes nothing");
}

void declared_units() {
    using namespace strix::device;
    const std::string gpu = "schema = \"strix-devices-v1\"\n[devices.main]\nvendor = \"amd\"\npci = \"0000:c5:00.0\"\n"
                            "roles = [\"pair\", \"decode\", \"prefill\"]\nattach = { kind = \"integrated\" }\n";
    const std::string npu = "[devices.npu]\nkind = \"npu\"\nvendor = \"amd\"\npci = \"0000:c5:00.1\"\nroles = [\"aux\"]\n";
    const std::string qualified = "driver = \"amdxdna\"\nfirmware = \"1\"\ncorun_qualification = \"q\"\n";
    Visible igpu;
    igpu.ordinal = 0;
    igpu.pci = "0000:c5:00.0";
    igpu.integrated = true;
    const auto units_of = [&](const std::string& text) {
        const auto declaration = parse_declaration(text);
        return units_from(resolve(&declaration, {igpu}, Request{}));
    };
    const auto bare = units_of(gpu);
    check(bare.size() == 1 && bare[0].kind == UnitKind::gpu && bare[0].id == "main", "the bound gpu is always an aux unit");
    check(units_from(resolve(nullptr, {igpu}, Request{})).size() == 1, "no declaration: the gpu only");
    check(units_of(gpu + npu).size() == 1, "an unqualified npu is not an aux unit");
    const auto with_npu = units_of(gpu + npu + qualified);
    check(with_npu.size() == 2 && with_npu[1].kind == UnitKind::npu && with_npu[1].id == "npu", "a qualified npu is an aux unit");
    check(units_of(gpu + "[devices.host]\nkind = \"cpu\"\nroles = [\"host\"]\n").size() == 1, "a cpu without aux is not an aux unit");
    const auto with_cpu = units_of(gpu + "[devices.host]\nkind = \"cpu\"\nroles = [\"host\", \"aux\"]\n");
    check(with_cpu.size() == 2 && with_cpu[1].kind == UnitKind::cpu && with_cpu[1].id == "host", "a cpu with aux is an aux unit");

    Config config;
    config.units = bare;
    AuxLane lane(config);
    invalid([&] { lane.add(stub(UnitKind::npu, "npu")); }, "an executor on an undeclared npu is refused");
    invalid([&] { lane.add(stub(UnitKind::cpu, "host")); }, "an executor on an undeclared cpu is refused");
    invalid([&] { lane.add(stub(UnitKind::gpu, "other")); }, "an executor on another gpu id is refused");
    invalid([&] { lane.add(Executor{}); }, "an executor without a function is refused");
    invalid([] { Config zero; zero.capacity = 0; AuxLane bad(zero); }, "zero capacity");
    check(!lane.has(JobKind::score), "nothing registered");
    fails_with([&] { (void)lane.run_now(score_job("x", 0)); }, AuxError::Code::no_executor, "no executor");
}

void scoring() {
    const std::string prompt = "the quick brown fox";
    auto gpu = std::make_shared<FakeScorer>(0.25f);
    auto cpu = std::make_shared<FakeScorer>(0.25f);
    FakeScorer direct(0.25f);
    const auto expected = direct.score(prompt, 7);

    // 1. Only the GPU executor (what a server has today): the lane call is the direct call, on the caller's thread.
    {
        Config config;
        config.units = {{UnitKind::gpu, "main"}};
        AuxLane lane(config);
        lane.add(score_executor(UnitKind::gpu, "main", "drafter-f16", gpu, false));
        LaneScorer scorer(lane);
        const auto got = scorer.score(prompt, 7);
        check(same(got, expected), "lane scorer equals the direct scorer bit for bit");
        check(gpu->calls == 1 && gpu->thread == std::this_thread::get_id(), "a non-threaded executor runs on the calling thread");
        check(scorer.last().unit == UnitKind::gpu && scorer.last().unit_id == "main" && scorer.last().identity == "drafter-f16" &&
                  scorer.last().fallbacks == 0,
              "result names its unit and identity");
        const auto c = lane.counters();
        check(c.submitted == 1 && c.completed == 1 && c.units.size() == 1 && c.units[0].jobs == 1 && c.fallbacks == 0, "counters after one job");
        check(lane.receipt_json().find("\"units\":[{\"unit\":\"gpu\",\"id\":\"main\",\"kind\":\"score\",\"identity\":\"drafter-f16\",\"jobs\":1,"
                                       "\"failed\":0,") != std::string::npos,
              "receipt lists the unit");
    }
    // 2. GPU + CPU reference: the GPU answers; when it fails the CPU reference answers on its own thread.
    {
        Config config;
        config.units = {{UnitKind::gpu, "main"}, {UnitKind::cpu, "host"}};
        AuxLane lane(config);
        lane.add(score_executor(UnitKind::cpu, "host", "drafter-ref", cpu, true));
        lane.add(score_executor(UnitKind::gpu, "main", "drafter-f16", gpu, false));
        LaneScorer scorer(lane);
        (void)scorer.score(prompt, 7);
        check(gpu->calls == 2 && cpu->calls == 0 && scorer.last().unit == UnitKind::gpu, "gpu first while it works");
        gpu->fail = true;
        const auto got = scorer.score(prompt, 7);
        check(same(got, expected), "fallback result equals the reference");
        check(cpu->calls == 1 && cpu->thread != std::this_thread::get_id(), "the cpu reference runs on the lane's worker");
        check(scorer.last().unit == UnitKind::cpu && scorer.last().fallbacks == 1 && scorer.last().identity == "drafter-ref",
              "fallback is reported");
        // A caller that needs one identity gets no silent substitute.
        LaneScorer strict(lane, "drafter-f16");
        fails_with([&] { (void)strict.score(prompt, 7); }, AuxError::Code::failed, "identity-bound job fails instead of changing unit");
        cpu->fail = true;
        fails_with([&] { (void)scorer.score(prompt, 7); }, AuxError::Code::failed, "every unit failed");
        const auto c = lane.counters();
        check(c.fallbacks == 2 && c.failed == 2 && c.completed == 2 && c.submitted == 4, "fallback counters");
        gpu->fail = false;
        cpu->fail = false;
        // Forced CPU reference (the A/B arm): the GPU is not touched.
        config.force = UnitKind::cpu;
        AuxLane forced(config);
        forced.add(score_executor(UnitKind::gpu, "main", "drafter-f16", gpu, false));
        forced.add(score_executor(UnitKind::cpu, "host", "drafter-ref", cpu, true));
        const int before = gpu->calls;
        LaneScorer reference(forced);
        check(same(reference.score(prompt, 7), expected) && gpu->calls == before && reference.last().unit == UnitKind::cpu, "forced cpu reference");
    }
    // 3. Packing is exact, and a malformed result is refused.
    {
        check(same(unpack_scores(pack_scores(expected)), expected), "pack and unpack are exact");
        auto packed = pack_scores(expected);
        packed.tensor.back() = 1.5f;
        invalid([&] { (void)unpack_scores(packed); }, "mask that is not a byte");
        packed = pack_scores(expected);
        packed.integers.pop_back();
        invalid([&] { (void)unpack_scores(packed); }, "short integer channel");
        auto uneven = expected;
        uneven.cut_prob.pop_back();
        invalid([&] { (void)pack_scores(uneven); }, "unequal buffers");
        check(unpack_scores(pack_scores(pflash::ScoreBuffers{})).token_score.empty(), "empty prompt");
    }
}

void queue() {
    auto gpu = std::make_shared<FakeScorer>(1.0f);
    Config config;
    config.units = {{UnitKind::gpu, "main"}, {UnitKind::cpu, "host"}};
    config.capacity = 2;
    const std::string a = "first", b = "second", c = "third";
    {
        AuxLane lane(config);
        lane.add(score_executor(UnitKind::gpu, "main", "g", gpu, false));
        auto fa = lane.submit(score_job(a, 1));
        auto fb = lane.submit(score_job(b, 2));
        fails_with([&] { (void)lane.submit(score_job(c, 3)); }, AuxError::Code::queue_full, "bounded queue refuses the third job");
        check(gpu->calls == 0, "nothing runs before the owner drains");
        check(lane.drain("host") == 0, "draining another unit takes nothing");
        check(lane.drain("main", 1) == 1 && gpu->calls == 1, "drain honours its job limit");
        check(lane.drain("main") == 1 && lane.drain("main") == 0, "drain empties the queue");
        FakeScorer direct(1.0f);
        check(same(unpack_scores(fa.get()), direct.score(a, 1)) && same(unpack_scores(fb.get()), direct.score(b, 2)), "queued jobs keep their order");
        auto late = score_job(c, 3);
        late.deadline = Clock::now() - std::chrono::milliseconds(1);
        auto fl = lane.submit(std::move(late));
        check(lane.drain("main") == 1, "an expired job is taken");
        fails_with([&] { (void)fl.get(); }, AuxError::Code::deadline, "an expired job fails instead of running late");
        check(gpu->calls == 2, "an expired job does not reach the executor");
        const auto counts = lane.counters();
        check(counts.queue_full == 1 && counts.deadline == 1 && counts.waiting_peak == 2 && counts.completed == 2 && counts.submitted == 3,
              "queue counters");
    }
    // Threaded executor: jobs complete without a drain, and a queued GPU failure moves to the worker.
    {
        auto cpu = std::make_shared<FakeScorer>(1.0f);
        config.capacity = 64;
        AuxLane lane(config);
        lane.add(score_executor(UnitKind::gpu, "main", "g", gpu, false));
        lane.add(score_executor(UnitKind::cpu, "host", "r", cpu, true));
        gpu->fail = true;
        auto f = lane.submit(score_job(a, 1));
        check(lane.drain("main") == 1, "the gpu owner takes the job");
        const auto result = f.get();
        check(result.unit == UnitKind::cpu && result.fallbacks == 1 && cpu->calls == 1, "a queued job falls back to the cpu worker");
        gpu->fail = false;
    }
    // Stopping the lane fails what is still waiting.
    std::future<AuxResult> orphan;
    {
        AuxLane lane(config);
        lane.add(score_executor(UnitKind::gpu, "main", "g", gpu, false));
        orphan = lane.submit(score_job(a, 1));
    }
    fails_with([&] { (void)orphan.get(); }, AuxError::Code::stopped, "a waiting job fails when the lane stops");
}

void vision() {
    std::vector<std::uint8_t> pixels(4 * 6 * 3);
    for (std::size_t i = 0; i < pixels.size(); ++i) pixels[i] = static_cast<std::uint8_t>(i * 5);
    ImageView image;
    image.height = 4;
    image.width = 6;
    image.channels = 3;
    image.pixels = pixels;
    // Reference encoder: one row per image line, hidden = 2 (sum and first pixel of the line).
    const auto encoder = [](float scale) {
        return ImageEncoder([scale](const ImageView& v, std::uint32_t& count, std::uint32_t& hidden) {
            count = v.height;
            hidden = 2;
            std::vector<float> rows;
            const std::size_t line = static_cast<std::size_t>(v.width) * v.channels;
            for (std::uint32_t y = 0; y < v.height; ++y) {
                float sum = 0.0f;
                for (std::size_t i = 0; i < line; ++i) sum += static_cast<float>(v.pixels[y * line + i]);
                rows.push_back(sum * scale);
                rows.push_back(static_cast<float>(v.pixels[y * line]));
            }
            return rows;
        });
    };
    std::uint32_t count = 0, hidden = 0;
    const auto expected = encoder(1.0f)(image, count, hidden);

    Config config;
    config.units = {{UnitKind::gpu, "main"}, {UnitKind::cpu, "host"}};
    AuxLane lane(config);
    lane.add(vision_executor(UnitKind::gpu, "main", "vision-f16-v1", encoder(1.0f), false));
    lane.add(vision_executor(UnitKind::cpu, "host", "vision-ref-f32", encoder(1.0f), true));
    const auto rows = encode_image(lane, image);
    check(rows.count == 4 && rows.hidden == 2 && rows.rows == expected && rows.unit == UnitKind::gpu && rows.identity == "vision-f16-v1",
          "image rows from the gpu executor");
    const auto reference = encode_image(lane, image, "vision-ref-f32");
    check(reference.rows == expected && reference.unit == UnitKind::cpu && reference.unit_id == "host", "image rows from the cpu reference");
    fails_with([&] { (void)encode_image(lane, image, "vision-int8-v1"); }, AuxError::Code::no_executor, "unknown vision identity");
    check(lane.has(JobKind::vision) && !lane.has(JobKind::score), "kinds registered");
    fails_with([&] { (void)lane.run_now(score_job("x", 0)); }, AuxError::Code::no_executor, "a vision lane does not score");

    ImageView broken = image;
    broken.width = 5;
    invalid([&] { (void)vision_job(broken); }, "pixels that do not match the shape");
    AuxLane bad(config);
    bad.add(vision_executor(UnitKind::gpu, "main", "v", [](const ImageView&, std::uint32_t& n, std::uint32_t& h) {
        n = 3;
        h = 3;
        return std::vector<float>(4);
    }, false));
    fails_with([&] { (void)encode_image(bad, image); }, AuxError::Code::failed, "an encoder with a wrong row count fails the job");
    invalid([] { (void)image_rows(AuxResult{{1.0f}, {2, 2}, {}, {}, UnitKind::gpu, {}, 0, 0.0, 0.0}); }, "malformed vision result");
}
} // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    try {
        choice();
        declared_units();
        scoring();
        queue();
        vision();
    } catch (const std::exception& e) {
        std::cerr << "aux lane tests: unexpected " << e.what() << '\n';
        return 1;
    }
    if (failures) {
        std::cerr << "aux lane tests: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "aux lane tests: ok\n";
    return 0;
}
