// Two-node Gemma 4 route (see pair.hpp). Transport, pipe, staging and CLI are the shared pair modules
// (strix/pair/*.hpp, xport/hip_pipe_staging.hpp); this file only binds them to the coordinator.
#include "pair.hpp"
#include "gemma4/prefill/pair.hpp"
#include "gemma4/prefill/prefill.hpp"
#include "strix/dist/pipeline_timeline.hpp"
#include "strix/env_flags.hpp"
#include "strix/pair/link.hpp"
#include "strix/pair/regime.hpp"
#include "xport/hip_pipe_staging.hpp"
#include "xport/pair_device_link.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace strix::gemma4::engine {
namespace {
using Json = strix::serve::Json;
constexpr std::string_view prefix = "STRIX_G4_PAIR";
constexpr std::uint16_t tcp_port = 29873;
unsigned env_cut(const ModelConfig& config) {
    const char* value = std::getenv("STRIX_G4_PAIR_CUT");
    if (!value || !*value) return pair_default_cut;
    char* end = nullptr;
    const auto cut = std::strtoul(value, &end, 10);
    if (*end || cut == 0 || cut >= config.layers) throw std::invalid_argument("STRIX_G4_PAIR_CUT must be in [1, layers)");
    return unsigned(cut);
}
std::uint64_t fnv(std::uint64_t hash, std::string_view bytes) {
    for (const char b : bytes) { hash ^= static_cast<unsigned char>(b); hash *= 1099511628211ULL; }
    return hash;
}
struct StageReset { ~StageReset() { prefill::pair_stage = {}; } };
void add_breakdown(Json& pair_json) {
    const auto& ps = prefill::pair_stage;
    if (!strix::env_flag_default_off("STRIX_G4_PAIR_BREAKDOWN")) return;
    pair_json.set("linear_ms", ps.linear_ms); pair_json.set("attention_ms", ps.attention_ms); pair_json.set("other_ms", ps.other_ms);
}
} // namespace
bool pair_enabled() { return strix::env_flag_default_off("STRIX_G4_PAIR"); }
std::size_t pair_chunk_count(std::size_t prompt, std::size_t chunk) {
    std::size_t count = 0, processed = 0;
    while (const auto n = prompt_chunk_rows(prompt, processed, processed, chunk)) { processed += n; ++count; }
    return count;
}
int run_pair(const Options& options, const Runtime& runtime, std::span<const std::uint32_t> prompt,
             const std::function<void(std::span<const std::uint32_t>, strix::serve::Json)>& emit) {
    using Clock = std::chrono::steady_clock;
    const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
    auto& backend = *runtime.backend;
    const auto& config = backend.config();
    const auto chunk = backend.prefill_chunk_rows();
    if (!chunk || options.streams != 1) throw std::invalid_argument("pair route needs the device many-row prefill and --streams 1");
    const auto rank = pair::env_rank(prefix);
    const auto cut = env_cut(config);
    const auto count = pair_chunk_count(prompt.size(), chunk);
    // Per-regime selection: a prompt of one chunk has nothing to overlap, so it runs on rank 1 alone.
    const auto mode = pair::choose({prompt.size(), count, 0, 0, true}, {});
    if (mode != pair::Mode::pipeline) {
        auto pair_json = Json::object();
        pair_json.set("rank", std::uint64_t(rank)); pair_json.set("mode", std::string("single")); pair_json.set("chunks", std::uint64_t(count));
        if (rank == 0U) { auto receipt = Json::object(); receipt.set("pair", pair_json); emit({}, receipt); return 0; }
        Coordinator coordinator(backend, runtime.drafts);
        auto request = options.request; request.prompt.assign(prompt.begin(), prompt.end());
        const auto slot = coordinator.submit(std::move(request));
        coordinator.run();
        auto receipt = coordinator.receipt(slot); receipt.set("stream", std::uint64_t(slot)); receipt.set("pair", pair_json);
        emit(coordinator.result(slot).ids, receipt);
        return coordinator.result(slot).status == Status::failed ? 1 : 0;
    }
    // Both ranks must run the same model, tables, profile and cut: their words go in every seam header.
    std::uint64_t identity = fnv(14695981039346656037ULL, backend.identity().dump());
    identity = fnv(identity, std::to_string(cut) + ":" + std::to_string(prompt.size()));
    // Control on the host transport; with --transport usb4gpu seam pieces move device to device, one way, with the
    // seam CRC computed on the device (the shared pair::DeviceLink, as the Qwen pair).
    pair::DeviceLink link(unsigned(rank), 0x50503447ULL /* "G4PP" */, pair::open_device_link(unsigned(rank), prefix, tcp_port), "g4-pair");
    const auto linked = Clock::now();
    const auto capacity = prefill::pair_seam_bytes(config, cut, unsigned(chunk));
    pair::PipeOptions pipe_options;
    pipe_options.profile_label = "g4-pair-profile";
    pipe_options.profile = strix::env_flag_default_off("STRIX_G4_PAIR_PROFILE");
    pipe_options.validate = [&, identity, count, cut](const pair::PipeHeader& h, std::uint64_t sequence) {
        using W = prefill::PairWord;
        if (h[W::magic] != prefill::pair_seam_magic || h[1] != sequence || h[42] != 3U || h[41] != count ||
            h[W::cut] != cut || h[W::identity] != identity || h[W::rows] == 0U || h[W::rows] > chunk ||
            h[W::bytes] != prefill::pair_seam_bytes(config, cut, unsigned(h[W::rows])))
            throw std::runtime_error("gemma4 pair seam header mismatch (model, profile, cut, prompt or chunk)");
        return std::size_t(h[W::bytes]);
    };
    // STRIX_PAIR_TIMING=1: per-chunk compute + seam phases on the shared timeline (same columns as Qwen).
    dist::PipelineTimeline timeline("g4-prefill", unsigned(rank),
        {"chunk", "d2h-wait", "crc", "wire", "ack-wait", "header-wait", "h2d-wait"});
    if (timeline.enabled()) { pipe_options.timeline = &timeline; pipe_options.timing = {1, 2, 3, 4, 5, 6}; }
    pair::Pipe pipe(rank, count,
        [&](std::span<const std::byte> data) { link.send(data); },
        [&](std::span<std::byte> data) { link.receive(data); },
        [&] { link.abort("gemma4 pair prefill stopped"); }, std::move(pipe_options),
        [&] {
            auto staging = pair::make_hip_pipe_staging(capacity, [] { return static_cast<hipStream_t>(prefill::pair_stage.queue); });
            staging->route_device([&link](std::size_t bytes) { return link.device_checked(bytes); });
            return staging; },
        [](std::span<const std::byte> data) { return dist::crc32c(data); },
        [&link](bool sending, pair::PipeStaging& staging, std::size_t offset, std::size_t bytes) {
            return link.piece(sending, staging, offset, bytes); });
    StageReset reset;
    prefill::pair_stage = {}; prefill::pair_stage.pipe = &pipe; prefill::pair_stage.rank = rank;
    prefill::pair_stage.cut = cut; prefill::pair_stage.identity = identity;
    auto pair_json = Json::object();
    pair_json.set("rank", std::uint64_t(rank)); pair_json.set("cut", std::uint64_t(cut));
    pair_json.set("mode", std::string("pipeline"));
    pair_json.set("chunks", std::uint64_t(count)); pair_json.set("seam_bytes_per_chunk", std::uint64_t(capacity));
    if (rank == 0U) {
        // Stage A: the coordinator's chunk schedule for this prompt, layers [0, cut) only. No decode.
        auto state = backend.create_stream(std::uint32_t(prompt.size() + options.request.max_tokens));
        std::size_t processed = 0;
        auto chunks = Json::array();
        const auto start = Clock::now();
        std::uint64_t step = 0;
        while (const auto n = prompt_chunk_rows(prompt.size(), processed, state->frontier(), chunk)) {
            const auto begin = Clock::now();
            auto tx = state->begin();
            (void)backend.prefill_rows(*tx, prompt.subspan(processed, n), false, false);
            tx->commit(n);
            processed += n;
            const auto end = Clock::now();
            chunks.push(ms(end - begin));
            if (timeline.enabled()) { timeline.add(step, 0, begin, end); timeline.mark(step, begin, end); }
            ++step;
        }
        pipe.finish(count);
        link.finish();
        pair_json.set("stage_ms", ms(Clock::now() - start)); pair_json.set("chunk_ms", chunks);
        pair_json.set("seal_wait_ms", prefill::pair_stage.seal_ms);
        add_breakdown(pair_json);
        pair_json.set("link_to_done_ms", ms(Clock::now() - linked));
        timeline.report(ms(Clock::now() - linked));
        auto receipt = Json::object(); receipt.set("pair", pair_json); receipt.set("prompt_tokens", std::uint64_t(prompt.size()));
        emit({}, receipt);
        return 0;
    }
    Coordinator coordinator(backend, runtime.drafts);
    auto request = options.request; request.prompt.assign(prompt.begin(), prompt.end());
    const auto slot = coordinator.submit(std::move(request));
    coordinator.run();
    pipe.finish(count);
    link.finish();
    const auto& result = coordinator.result(slot);
    auto receipt = coordinator.receipt(slot); receipt.set("stream", std::uint64_t(slot));
    pair_json.set("seam_wait_ms", prefill::pair_stage.wait_ms);
    { auto chunks = Json::array(); for (const auto v : prefill::pair_stage.chunk_ms) chunks.push(v); pair_json.set("chunk_ms", chunks); }
    add_breakdown(pair_json);
    pair_json.set("link_to_done_ms", ms(Clock::now() - linked));
    timeline.report(ms(Clock::now() - linked));
    receipt.set("pair", pair_json);
    emit(result.ids, receipt);
    return result.status == Status::failed ? 1 : 0;
}
} // namespace strix::gemma4::engine
