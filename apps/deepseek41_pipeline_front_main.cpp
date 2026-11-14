#include "strix/device/hip_gate.hpp"
#include "strix/deepseek41_attention_host.hpp"
#include "strix/deepseek41_device_forward.hpp"
#include "strix/ds41_pipeline_front.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/serve/api.hpp"
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/deepseek_chat.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/http_server.hpp"
#include "strix/serve/json.hpp"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <mutex>
#include <thread>

#include <chrono>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// The FRONT of a deepseek41 pipeline (strix/ds41-pipeline-v1, docs/models/deepseek-v4.1/pipeline-stages.md): the embedding and target
// layers [0, K) on this node's backend, the rest on a TAIL (strix-deepseek41-rank --role pipeline-stage --first-layer K),
// one boundary per step over TCP; the TAIL answers with the greedy tokens.
//   strix-deepseek41-pipeline-front MODEL.gguf --backend hip --layers K --stage-connect HOST:PORT --stage-token-file PATH
//       --prompt-ids a,b,c [--generate G] [--max-positions N] [--op-hashes] [--stream-experts]
//       [--fill N --fill-ids FILE]  context before the prompt in windows of up to 8 rows (as strix-deepseek41-rank --fill)
//       [--export-at N --state-out FILE]  after N fed tokens: the FRONT's [0, K) and the TAIL's [K, n) states merged into
//                                         FILE (a whole forward continues from it: strix-deepseek41-rank --import-state)
//       [--handoff HOST:PORT]  at --export-at, the TAIL's state moves to a second TAIL (export -> import over TCP) and the
//                              run continues there
//       [--close]  end the TAIL session at the end (CLOSE)
//       [--tail-layers N]  the TAIL's end layer when it runs a trimmed model (default: every layer)
//   HTTP (instead of --prompt-ids): --http HOST:PORT [--context N] [--model-id ID] [--http-api-key-file F]
//       [--default-max-tokens N]: the OpenAI/Anthropic routes of strix_serve (one request at a time, greedy), GET /health
//       ({"status","stage":"connected|disconnected","layers"}), POST /admin/drain, POST /admin/export-state (the merged
//       FRONT [0, K) + TAIL [K, n) state after the last request: application/octet-stream, X-State-Format,
//       X-State-Sha256, X-State-Position, X-State-Next-Token); READY {json} once the listener is up.
// Output lines "pos P fed T argmax N logits_fnv F" diff against the whole run's (strix-deepseek41-rank) and the TAIL's.
// Backends: hip (DeviceForward, layers [0, K), head off), host (HostForward on the CPU, one row per step). The interface (ds41::pipe::FrontBackend) takes a CUDA backend.
namespace {

namespace pipe = strix::ds41::pipe;
namespace ds = strix::deepseek41;

struct Args {
    std::string model, backend = "hip", connect, token_file, prompt, fill_ids, state_out, handoff;
    std::string http, model_id = "deepseek-v4.1-flash", api_key_file;
    std::uint32_t context = 4096, default_max_tokens = 1024;
    std::uint32_t layers = 0, tail_layers = 0, generate = 8, max_positions = 4096, fill = 0, export_at = 0;
    bool op_hashes = false, stream_experts = false, close = false;
};

// The HIP backend: DeviceForward over layers [0, K) without the head.
class HipFront final : public pipe::FrontBackend {
public:
    HipFront(const Args& a) : cut_(a.layers) {
        ds::DeviceForwardOptions o;
        o.layers = a.layers;
        o.max_positions = a.max_positions;
        o.head = false;
        o.op_hashes = a.op_hashes;
        o.stream_experts = a.stream_experts;
        fwd_ = std::make_unique<ds::DeviceForward>(a.model, o);
    }
    pipe::Boundary step(std::span<const std::uint32_t> tokens) override {
        (void)fwd_->step_window(tokens);
        return fwd_->export_boundary();
    }
    void commit(std::uint32_t kept) override { fwd_->commit_window(kept); }
    void reset() override { fwd_->reset(); }
    [[nodiscard]] ds::AttentionState export_state() const override { return fwd_->export_state(); }
    void import_state(const ds::AttentionState& s) override { fwd_->import_state(s); }
    [[nodiscard]] std::uint32_t position() const override { return fwd_->next_position(); }
    [[nodiscard]] std::uint32_t cut() const override { return cut_; }
    [[nodiscard]] std::uint32_t max_rows() const override { return ds::DeviceForward::max_window_rows; }
    [[nodiscard]] std::string backend() const override { return "hip"; }
    [[nodiscard]] std::string numeric_identity() const override { return "ds41-exact-v1"; }
    [[nodiscard]] ds::DeviceForward& forward() { return *fwd_; }

private:
    std::uint32_t cut_;
    std::unique_ptr<ds::DeviceForward> fwd_;
};

// The CPU backend: HostForward over layers [0, K) without the head (the reference the device forward equals op by op).
// One row per step; no state export/import.
class HostFront final : public pipe::FrontBackend {
public:
    HostFront(const Args& a, const ds::TargetModel& model) : cut_(a.layers), plan_(pipe::plan_boundary(model.layers, a.layers)),
                                                            geo_(pipe::boundary_geometry(model.config)) {
        ds::HostForwardOptions o;
        o.layers = a.layers;
        o.max_positions = a.max_positions;
        o.head = false;
        // the requantization overlay of the device forward's STRIX_DS41_OVERLAY (the TAIL's weights must be the same)
        if (const char* e = std::getenv("STRIX_DS41_OVERLAY"); e != nullptr && *e != 0) o.weights_overlay = e;
        fwd_ = std::make_unique<ds::HostForward>(a.model, o);
        if (!fwd_->overlay_description().empty()) std::printf("FRONT host overlay: %s\n", fwd_->overlay_description().c_str());
    }
    pipe::Boundary step(std::span<const std::uint32_t> tokens) override {
        if (tokens.size() != 1U) throw std::invalid_argument("host FRONT: one row per step");
        const std::uint32_t p = fwd_->position();
        (void)fwd_->step(tokens[0]);
        pipe::Boundary b;
        b.position = p;
        b.rows = 1;
        b.tokens = {tokens[0]};
        b.stream = fwd_->buffer(ds::Buffer::stream);
        const auto& mix = fwd_->buffer(ds::Buffer::mix_ffn);
        pipe::Section m{pipe::SectionKind::mix_ffn, cut_ - 1U, 0U, 1U, {}};
        m.bytes.resize(mix.size() * 4U);
        std::memcpy(m.bytes.data(), mix.data(), m.bytes.size());
        b.sections.push_back(std::move(m));
        const auto& at = fwd_->attention();
        for (const auto& s : plan_.kv_sources) {
            const std::uint32_t first = p / s.ratio, n = (p + 1U) / s.ratio - first;
            pipe::Section k{pipe::SectionKind::kv_entries, s.layer, first, n, {}};
            const auto kv = at.kv_cache(s.layer).subspan(std::size_t{first} * geo_.kv_entry_bytes, std::size_t{n} * geo_.kv_entry_bytes);
            const auto ix = at.idx_cache(s.layer).subspan(std::size_t{first} * geo_.idx_entry_bytes, std::size_t{n} * geo_.idx_entry_bytes);
            k.bytes.assign(kv.begin(), kv.end());
            k.bytes.insert(k.bytes.end(), ix.begin(), ix.end());
            b.sections.push_back(std::move(k));
        }
        for (const auto& s : plan_.topk_sources) {
            const auto& sel = at.topk(s.layer);
            pipe::Section k{pipe::SectionKind::topk, s.layer, 0U, 1U, {}};
            const auto put = [&](std::uint32_t v) {
                for (int i = 0; i < 4; ++i) k.bytes.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
            };
            put(static_cast<std::uint32_t>(sel.size()));
            for (const auto v : sel) put(v);
            b.sections.push_back(std::move(k));
        }
        if (plan_.candidate_source >= 0) {
            const std::uint32_t count = (p + 1U) / plan_.candidate_ratio, stride = (count + geo_.candidate_block - 1U) / geo_.candidate_block;
            pipe::Section k{pipe::SectionKind::candidates, static_cast<std::uint32_t>(plan_.candidate_source), stride, 1U, {}};
            const auto c = at.candidates().first(stride);
            k.bytes.assign(c.begin(), c.end());
            b.sections.push_back(std::move(k));
        }
        return b;
    }
    void commit(std::uint32_t kept) override {
        if (kept != 1U) throw std::invalid_argument("host FRONT: one row per step");
    }
    void reset() override { throw std::logic_error("host FRONT: no reset (start a new process)"); }
    [[nodiscard]] ds::AttentionState export_state() const override { throw std::logic_error("host FRONT: no state export"); }
    void import_state(const ds::AttentionState&) override { throw std::logic_error("host FRONT: no state import"); }
    [[nodiscard]] std::uint32_t position() const override { return fwd_->position(); }
    [[nodiscard]] std::uint32_t cut() const override { return cut_; }
    [[nodiscard]] std::uint32_t max_rows() const override { return 1U; }
    [[nodiscard]] std::string backend() const override { return "host"; }
    [[nodiscard]] std::string numeric_identity() const override { return "ds41-exact-v1"; }
    [[nodiscard]] const ds::HostForward& forward() const { return *fwd_; }

private:
    std::uint32_t cut_;
    pipe::BoundaryPlan plan_;
    pipe::BoundaryGeometry geo_;
    std::unique_ptr<ds::HostForward> fwd_;
};

std::vector<std::uint32_t> ids(const std::string& text) {
    std::vector<std::uint32_t> v;
    std::stringstream ss(text);
    for (std::string t; std::getline(ss, t, ',');)
        if (!t.empty()) v.push_back(static_cast<std::uint32_t>(std::stoul(t)));
    return v;
}

std::atomic<bool> g_stop{false};
extern "C" void on_signal(int) { g_stop.store(true); }

// One request at a time through the pipeline: the FRONT's layers here, the TAIL's over the session. Prompt rows go in
// windows (the first row alone, then up to 8 rows); the last row's token is the first generated one; then one row per
// step. Greedy, exact per request (every request starts from a reset).
class PipelineEngine final : public strix::serve::GenerationEngine {
public:
    PipelineEngine(pipe::FrontBackend& front, pipe::StageClient& client, std::uint32_t context) : front_(front), client_(client), context_(context) {}
    strix::serve::GenerationStats generate(strix::serve::GenerationJob& job) override {
        const std::lock_guard lock(mutex_);
        strix::serve::GenerationStats st;
        st.prompt_tokens = job.prompt.size();
        st.sampling = "greedy";
        if (job.prompt.empty() || job.prompt.size() + 1U >= context_) {
            st.finish = "context";
            return st;
        }
        const auto t0 = std::chrono::steady_clock::now();
        try {
            if (front_.position() != 0U) {
                front_.reset();
                client_.reset();
            }
            std::uint32_t next = 0;
            const auto run_rows = [&](std::span<const std::uint32_t> rows) {
                const auto r = client_.step(front_.step(rows));
                next = r.tokens.back();
            };
            run_rows(std::span(job.prompt).first(1));
            for (std::size_t i = 1; i < job.prompt.size();) {
                const std::size_t n = std::min<std::size_t>(front_.max_rows(), job.prompt.size() - i);
                run_rows(std::span(job.prompt).subspan(i, n));
                i += n;
            }
            st.prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            const auto t1 = std::chrono::steady_clock::now();
            for (;;) {
                ++st.generated_tokens;
                ++st.cycles;
                const std::uint32_t one[1] = {next};
                const bool more = job.on_tokens ? job.on_tokens(one) : true;
                if (!more) { st.finish = "cancelled"; break; }
                if (std::find(job.stop_tokens.begin(), job.stop_tokens.end(), next) != job.stop_tokens.end()) { st.finish = "stop"; break; }
                if (st.generated_tokens >= job.max_tokens) { st.finish = "length"; break; }
                if (front_.position() + 1U >= context_) { st.finish = "context"; break; }
                run_rows(one);
            }
            st.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
            connected_.store(true);
        } catch (const pipe::WireError&) {
            connected_.store(false);
            throw;
        }
        return st;
    }
    [[nodiscard]] std::uint64_t context_tokens() const override { return context_; }
    // The merged state of the last request (FRONT [0, K) + TAIL [K, n)), serialized.
    std::vector<std::uint8_t> export_state(ds::AttentionState* out) {
        const std::lock_guard lock(mutex_);
        const auto tail_msg = client_.export_state();
        const auto merged = ds::merge_attention_states(front_.export_state(), ds::deserialize_attention_state(tail_msg.state));
        if (out != nullptr) *out = merged;
        return ds::serialize_attention_state(merged);
    }
    [[nodiscard]] bool connected() const noexcept { return connected_.load(); }

private:
    pipe::FrontBackend& front_;
    pipe::StageClient& client_;
    std::uint32_t context_;
    std::mutex mutex_;
    std::atomic<bool> connected_{true};
};

int serve_http(const Args& a, pipe::FrontBackend& front, pipe::StageClient& client) {
    namespace serve = strix::serve;
    const auto tokenizer = serve::BpeTokenizer::load_gguf(a.model);
    PipelineEngine engine(front, client, a.context);
    serve::ApiConfig config;
    config.model_id = a.model_id;
    config.default_max_tokens = a.default_max_tokens;
    config.default_thinking = false;
    config.sampling_policy = "greedy";
    config.engine_description = "deepseek41 pipeline: FRONT layers [0, " + std::to_string(front.cut()) + ") (" + front.backend() + ") + TAIL " + a.connect +
                                ", greedy, one request at a time";
    config.stop_tokens = {tokenizer->eos_token()};
    config.chat_renderer = [](const serve::ChatTemplateInput& in) { return serve::render_deepseek_v41_chat(in); };
    config.output_parser = [](const std::string& t, bool thinking, const serve::Json& tools) { return serve::parse_deepseek_v41_output(t, thinking, tools); };
    config.tool_call_marker = serve::deepseek_v41_tool_marker;
    std::string key;
    if (!a.api_key_file.empty()) {
        std::ifstream in(a.api_key_file);
        std::getline(in, key);
        while (!key.empty() && (key.back() == '\r' || key.back() == ' ')) key.pop_back();
        if (key.empty()) throw std::invalid_argument("--http-api-key-file is empty");
        config.api_key = key;
    }
    serve::GenerationQueue queue(64);
    std::atomic<bool> draining{false};
    config.on_drain = [&] {
        draining.store(true);
        queue.shutdown();
    };
    config.status_extra = [&](serve::Json& out) {
        serve::Json p = serve::Json::object();
        p.set("stage", serve::Json(a.connect));
        p.set("stage_state", serve::Json(engine.connected() ? "connected" : "disconnected"));
        p.set("cut", serve::Json(static_cast<std::uint64_t>(front.cut())));
        p.set("boundary_bytes_sent", serve::Json(client.bytes_sent()));
        out.set("pipeline", std::move(p));
    };
    serve::ApiService api(config, tokenizer, queue, a.context, false);
    const auto authorized = [&](const serve::HttpRequest& r) {
        if (key.empty()) return true;
        const auto bearer = r.header("authorization");
        return bearer == "Bearer " + key || r.header("x-api-key") == key;
    };
    const auto [host, port] = pipe::split_host_port(a.http);
    serve::HttpServerConfig http;
    http.bind_address = host;
    http.port = port;
    serve::HttpServer server(http, [&](const serve::HttpRequest& request, serve::HttpResponseWriter& response) {
        if (request.method == "GET" && (request.path == "/health" || request.path == "/v1/health")) {
            serve::Json out = serve::Json::object();
            const bool up = engine.connected() && !draining.load();
            out.set("status", serve::Json(draining.load() ? "draining" : engine.connected() ? "ok" : "failed"));
            out.set("stage", serve::Json(engine.connected() ? "connected" : "disconnected"));
            out.set("model", serve::Json(a.model_id));
            serve::Json layers = serve::Json::array();
            layers.push(serve::Json(0));
            layers.push(serve::Json(static_cast<std::uint64_t>(front.cut())));
            out.set("layers", layers);
            response.send(up ? 200 : 503, "application/json", out.dump());
            return;
        }
        if (request.method == "POST" && request.path == "/admin/export-state") {
            if (!authorized(request)) {
                response.send(401, "application/json", R"({"error":{"message":"unauthorized"}})");
                return;
            }
            try {
                ds::AttentionState st;
                const auto bytes = engine.export_state(&st);
                const auto sha = strix::model_content_identity_hex(strix::sha256_bytes(std::as_bytes(std::span(bytes))));
                response.send(200, "application/octet-stream", std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()),
                              {{"X-State-Format", "strix/ds41-attention-state-v1"}, {"X-State-Sha256", sha},
                               {"X-State-Position", std::to_string(st.position)}, {"X-State-Next-Token", std::to_string(st.next_token)}});
            } catch (const std::exception& e) {
                serve::Json err = serve::Json::object();
                err.set("error", serve::Json(std::string(e.what())));
                response.send(500, "application/json", err.dump());
            }
            return;
        }
        api.handle(request, response);
    });
    server.listen();
    api.set_state("ready");
    std::printf("READY {\"role\":\"pipeline-front\",\"layers\":[0,%u],\"stage\":\"%s\",\"http\":\"%s:%u\",\"model\":\"%s\"}\n", front.cut(), a.connect.c_str(),
                host.c_str(), server.port(), a.model_id.c_str());
    std::fflush(stdout);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::thread http_thread([&server] { server.serve(); });
    std::thread watcher([&] {
        while (!g_stop.load() && !queue.stopping()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        queue.shutdown();
    });
    std::string failure;
    try {
        queue.run(engine);
    } catch (const std::exception& e) {
        failure = e.what();
    }
    g_stop.store(true);
    watcher.join();
    server.stop();
    http_thread.join();
    try {
        client.close();
    } catch (const std::exception&) {
    }
    if (!failure.empty()) {
        std::printf("{\"event\":\"POISONED\",\"reason\":\"%s\"}\n", failure.c_str());
        return 3;
    }
    std::printf("{\"event\":\"STOPPED\"}\n");
    return 0;
}

int run(const Args& a) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto model = ds::validate_target(ds::read_file(a.model));
    const auto& c = model.config;
    if (a.layers == 0U || a.layers >= c.num_hidden_layers) throw std::invalid_argument("--layers K must be in [1, " + std::to_string(c.num_hidden_layers) + ")");
    std::unique_ptr<pipe::FrontBackend> owner;
    HipFront* hip = nullptr;
    HostFront* cpu = nullptr;
    if (a.backend == "hip") {
        owner = std::make_unique<HipFront>(a);
        hip = static_cast<HipFront*>(owner.get());
        std::printf("FRONT hip: layers [0, %u), device weights %.2f GiB loaded in %.1f s\n", a.layers,
                    static_cast<double>(hip->forward().weight_bytes()) / 1073741824.0, hip->forward().load_seconds());
    } else if (a.backend == "host") {
        owner = std::make_unique<HostFront>(a, model);
        cpu = static_cast<HostFront*>(owner.get());
        std::printf("FRONT host: layers [0, %u) on the CPU (HostForward)\n", a.layers);
    } else {
        throw std::invalid_argument("--backend is hip or host (a CUDA backend implements ds41::pipe::FrontBackend)");
    }
    auto& front = *owner;
    const auto cut_plan = pipe::plan_boundary(model.layers, a.layers);
    pipe::Hello hello;
    hello.cut = a.layers;
    hello.layers = a.tail_layers != 0U ? a.tail_layers : c.num_hidden_layers;  // the TAIL's end (a trimmed model in tests)
    hello.hidden = c.hidden_size;
    hello.hc = c.hc_mult;
    hello.vocab = c.vocab_size;
    hello.max_positions = a.max_positions;
    hello.model_identity = model.manifest.layout_identity;
    hello.backend = front.backend();
    hello.numeric_identity = front.numeric_identity();
    pipe::ClientOptions co;
    co.connect = a.connect;
    co.token = pipe::read_token_file(a.token_file);
    auto client = std::make_unique<pipe::StageClient>(co, hello);
    std::printf("TAIL %s: layers [%u, %u), world %u, %s\n", a.connect.c_str(), client->ack().first_layer, client->ack().end_layer, client->ack().world,
                client->ack().numeric_identity.c_str());
    if (!a.http.empty()) return serve_http(a, front, *client);
    std::printf("READY {\"role\":\"pipeline-front\",\"layers\":[0,%u],\"stage\":\"%s\",\"model_identity\":\"%016llx\"}\n", a.layers, a.connect.c_str(),
                static_cast<unsigned long long>(hello.model_identity));
    std::fflush(stdout);

    std::uint64_t boundary_bytes = 0, steps = 0;
    double front_seconds = 0.0, decode_seconds = 0.0;
    std::size_t decode_steps = 0;
    const auto one_step = [&](std::span<const std::uint32_t> rows) {
        const auto s0 = std::chrono::steady_clock::now();
        const auto b = front.step(rows);
        front_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
        const auto bytes = pipe::step_bytes(cut_plan, pipe::boundary_geometry(c), b.position, b.rows);
        boundary_bytes += bytes.total();
        ++steps;
        auto r = client->step(b);
        if (a.op_hashes && rows.size() == 1U)
            for (const auto& [name, hash] : hip ? hip->forward().last_step_hashes() : cpu->forward().last_step_hashes())
                std::printf("  %s %016llx\n", name.c_str(), static_cast<unsigned long long>(hash));
        return r;
    };
    if (a.fill != 0U) {  // as strix-deepseek41-rank --fill: the first row alone, then windows of 8, every row kept
        const auto filler = [&] {
            std::ifstream f(a.fill_ids);
            std::stringstream ss;
            ss << f.rdbuf();
            return ids(ss.str());
        }();
        if (filler.empty()) throw std::invalid_argument("--fill needs --fill-ids FILE");
        std::vector<std::uint32_t> window{filler[0]};
        (void)one_step(window);
        for (std::uint32_t i = 1; i < a.fill;) {
            window.clear();
            while (window.size() < front.max_rows() && i < a.fill) window.push_back(filler[i++ % filler.size()]);
            (void)one_step(window);
        }
        std::printf("filled %u context tokens\n", a.fill);
    }
    auto seq = ids(a.prompt);
    const std::size_t total = seq.size() + a.generate;
    const std::size_t base = front.position();
    for (std::size_t pos = 0; pos < seq.size(); ++pos) {
        if (a.export_at != 0U && pos == a.export_at) {
            // the pipeline state after a.export_at fed tokens: FRONT [0, K) + TAIL [K, n)
            const auto tail_msg = client->export_state();
            const auto tail = ds::deserialize_attention_state(tail_msg.state);
            auto mine = front.export_state();
            const auto merged = ds::merge_attention_states(mine, tail);
            std::printf("exported at position %u: FRONT %zu B + TAIL %zu B, merged digest %016llx, next token %u\n", merged.position,
                        ds::serialize_attention_state(mine).size(), tail_msg.state.size(), static_cast<unsigned long long>(merged.digest()),
                        merged.next_token);
            std::printf("  TAIL manifest %s\n", tail_msg.manifest.c_str());
            if (!a.state_out.empty()) ds::save_attention_state(a.state_out, merged);
            if (!a.handoff.empty()) {  // move the TAIL: the second TAIL imports the first's state
                pipe::ClientOptions ho = co;
                ho.connect = a.handoff;
                auto second = std::make_unique<pipe::StageClient>(ho, hello);
                second->import_state(tail_msg);
                client->close();
                client = std::move(second);
                std::printf("TAIL handed off to %s at position %u\n", a.handoff.c_str(), merged.position);
            }
        }
        const std::uint32_t one[1] = {seq[pos]};
        const auto s0 = std::chrono::steady_clock::now();
        const auto r = one_step(one);
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
        if (pos >= 1U) {
            decode_seconds += dt;
            ++decode_steps;
        }
        const auto next = r.tokens[0];
        std::printf("pos %zu fed %u argmax %u logits_fnv %016llx (%.1f s)\n", base + pos, seq[pos], next, static_cast<unsigned long long>(r.logits_fnv[0]),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        std::fflush(stdout);
        if (pos + 1U == seq.size() && seq.size() < total) seq.push_back(next);
    }
    std::printf("pipeline: %llu steps, %.1f MB of boundaries (%.1f KB/step), %.2f s in the FRONT's layers, %.2f s waiting for the TAIL (wire + TAIL); "
                "decode %.1f ms/token over %zu steps\n",
                static_cast<unsigned long long>(steps), static_cast<double>(boundary_bytes) / 1e6,
                steps ? static_cast<double>(boundary_bytes) / 1e3 / static_cast<double>(steps) : 0.0, front_seconds, client->wire_seconds(),
                decode_steps ? decode_seconds * 1e3 / static_cast<double>(decode_steps) : 0.0, decode_steps);
    if (a.close) client->close();
    return 0;
}

}  // namespace

int main(int argc, char** argv) try {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        const auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
            return std::string(argv[++i]);
        };
        const auto u32 = [&] { return static_cast<std::uint32_t>(std::stoul(next())); };
        if (s == "--backend") a.backend = next();
        else if (s == "--layers") a.layers = u32();
        else if (s == "--tail-layers") a.tail_layers = u32();
        else if (s == "--stage-connect") a.connect = next();
        else if (s == "--stage-token-file") a.token_file = next();
        else if (s == "--prompt-ids") a.prompt = next();
        else if (s == "--generate") a.generate = u32();
        else if (s == "--max-positions") a.max_positions = u32();
        else if (s == "--op-hashes") a.op_hashes = true;
        else if (s == "--stream-experts") a.stream_experts = true;
        else if (s == "--fill") a.fill = u32();
        else if (s == "--fill-ids") a.fill_ids = next();
        else if (s == "--export-at") a.export_at = u32();
        else if (s == "--state-out") a.state_out = next();
        else if (s == "--handoff") a.handoff = next();
        else if (s == "--close") a.close = true;
        else if (s == "--http") a.http = next();
        else if (s == "--context") a.context = u32();
        else if (s == "--model-id") a.model_id = next();
        else if (s == "--http-api-key-file") a.api_key_file = next();
        else if (s == "--default-max-tokens") a.default_max_tokens = u32();
        else if (a.model.empty()) a.model = s;
        else throw std::invalid_argument("unknown argument " + s);
    }
    if (!a.http.empty()) a.max_positions = std::max(a.max_positions, a.context);
    if (a.model.empty() || a.connect.empty() || a.token_file.empty() || (a.prompt.empty() && a.http.empty()))
        throw std::invalid_argument("usage: strix-deepseek41-pipeline-front MODEL.gguf --layers K --stage-connect HOST:PORT --stage-token-file PATH --prompt-ids a,b,c");
    return run(a);
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
