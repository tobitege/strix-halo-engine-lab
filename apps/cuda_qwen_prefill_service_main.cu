// strix-cuda-qwen-prefill-service: the bit-exact CUDA Qwen prefill (include/strix/cuda_qwen_prefill.hpp) resident behind a
// small HTTP/1.1 server. One request at a time on the GPU, in arrival order.
//
//   GET  /health      {"status":"ok"|"failed","busy":bool,"queued":n,"model":"qwen3.8-flash-next","max_positions":n,...}
//   POST /v1/prefill  {"tokens":[u32...], "handoff":"http://<node>:<port>/admin/rtx-import", "session"?:any,
//                      "reserve_tokens"?:n, "handoff_codec"?:"bp-v1"|"identity"}
//     Optional block_schedule: "default"|"tail-large", final_block_rows: N, trace: true (trace returned in reply).
//     computes rows [0, P-1) of the P prompt tokens and streams the node state to the node's --rtx-state-import routes
//     while it computes (begin, multi-segment chunks per finished block, the fixed blob, commit). The node then runs the
//     last prompt row itself. Reply: {"session","position","tokens","manifest_sha256","seconds":{queue,prefill,transfer,
//     import,total},"import":<node commit reply>}. 400 bad request, 413 too long, 502 hand-off failure, 503 busy/failed.
//   Cooperative prefill (opt-in): "local_rows":q (0 < q < P-1, a multiple of the node's prefix-cache block, the node
//     started with --prefix-cache-host-bytes) makes the service POST the node's own completion of prompt[0, q+1)
//     (max_tokens 1; "node_completions" URL, default the hand-off origin + /v1/completions) while it computes, and send
//     only the rows >= q plus the fixed state: the node restores its checkpoint of [0, q) and the imported rows [q, P-1).
//     The reply comes after both finished ("seconds":{"node_prefix":...}).
//
// Hand-off flags: --handoff-codec bp-v1|identity (default bp-v1 when the node's layout lists it, else plain chunks;
// docs/rtx-state-import.md "Chunk encoding"), --handoff-encode-threads N (4), --handoff-connections N (default 2: chunk
// POSTs in flight), --handoff-chunk-mib N (32). Release per layer (default; --no-release-by-layer: after each block):
// every block's rows and the last block's GDN recurrent/conv slices are sent per layer while the next layers compute.
#include "strix/device/cuda_enumerator.hpp"
#include "strix/cuda_qwen_prefill.hpp"
#include "strix/serve/json.hpp"

#include "qwen_handoff.hpp"
#include "qwen_http.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <future>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace qp = strix::cuda::qwen_prefill;
namespace qh = strix::qwen_handoff;
namespace http = strix::qwen_http;
using strix::serve::Json;
using Clock = std::chrono::steady_clock;

namespace {

double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// Identities naming this producer's arithmetic in the node's lineage (nonzero, stable).
constexpr std::uint64_t fnv(const char* s) {
    std::uint64_t h = 14695981039346656037ULL;
    while (*s) { h ^= static_cast<unsigned char>(*s++); h *= 1099511628211ULL; }
    return h;
}
constexpr std::uint64_t kNumerical = fnv("strix-cuda-qwen-prefill:a8-single-profile:hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1");
constexpr std::uint64_t kReduction = fnv("strix-cuda-qwen-prefill:hip-gfx1151-order-v1");
constexpr std::uint64_t kStateUpdate = fnv("strix-cuda-qwen-prefill:state-v1");

Json trace_json(const strix::qwen_prefill_trace::Trace& trace) {
    Json events = Json::array();
    std::map<std::uint64_t, double> starts, compute_starts, mtp_starts, compute_ms;
    Json blocks = Json::array();
    auto recorded = trace.snapshot();
    std::stable_sort(recorded.begin(), recorded.end(), [](const auto& a, const auto& b) { return a.ms < b.ms; });
    for (const auto& e : recorded) {
        Json v = Json::object();
        v.set("stage", Json(e.stage)); v.set("id", Json(e.id)); v.set("ms", Json(e.ms));
        if (!e.component.empty() && e.stage != "chunk_piece") v.set("component", Json(e.component));
        v.set("first", Json(e.first)); v.set("rows", Json(e.rows)); v.set("bytes", Json(e.bytes));
        if (e.stage == "chunk_piece") v.set("chunk", Json(static_cast<std::uint64_t>(std::stoull(e.component))));
        if (e.stage == "block_start") starts[e.id] = e.ms;
        if (e.stage == "compute_start") compute_starts[e.id] = e.ms;
        if (e.stage == "compute_end" && compute_starts.count(e.id)) compute_ms[e.id] += e.ms-compute_starts.at(e.id);
        if (e.stage == "mtp_compute_start") mtp_starts[e.id] = e.ms;
        if (e.stage == "mtp_compute_end" && mtp_starts.count(e.id)) compute_ms[e.id] += e.ms-mtp_starts.at(e.id);
        if (e.stage == "block_end") {
            if (starts.count(e.id)) v.set("elapsed_ms", Json(e.ms-starts.at(e.id)));
            v.set("compute_ms", Json(compute_ms[e.id]));
            blocks.as_array().push_back(v);
        }
        events.push(v);
    }
    Json t = Json::object(); t.set("clock", Json("steady_clock"));
    t.set("unit", Json("ms")); t.set("events", events); t.set("blocks", blocks);
    t.set("dropped_events", Json(trace.dropped()));
    return t;
}

struct Job {
    std::vector<std::uint32_t> tokens;
    std::string handoff, api_key, codec;
    bool has_codec = false;
    bool trace = false, tail_large = false, has_schedule = false;
    std::uint32_t final_rows = 0;
    bool has_final_rows = false;
    std::uint64_t local_rows = 0;
    std::string node_completions;
    Json session;
    Clock::time_point queued_at;
    std::promise<std::pair<int, std::string>> reply;
};

class Service {
public:
    Service(qp::Prefill& p, const qp::Config& config, std::string handoff_key, qh::Options handoff)
        : p_(p), config_(config), max_positions_(config.max_positions), handoff_key_(std::move(handoff_key)), handoff_(std::move(handoff)) {
        worker_ = std::thread([this] { loop(); });
    }
    ~Service() {
        { std::lock_guard<std::mutex> l(m_); stop_ = true; }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
    std::string health() {
        std::lock_guard<std::mutex> l(m_);
        Json j = Json::object();
        j.set("status", Json(failed_ ? "failed" : "ok"));
        j.set("busy", Json(busy_));
        j.set("queued", Json(static_cast<std::uint64_t>(q_.size())));
        j.set("model", Json("qwen3.8-flash-next"));
        j.set("max_positions", Json(static_cast<std::uint64_t>(max_positions_)));
        j.set("served", Json(served_));
        j.set("block_schedule", Json(config_.tail_large ? "tail-large" : "default"));
        j.set("final_block_rows", Json(static_cast<std::uint64_t>(config_.final_block_rows)));
        j.set("block_capacity", Json(static_cast<std::uint64_t>(*std::max_element(config_.block_schedule.begin(), config_.block_schedule.end()))));
        j.set("describe", Json(p_.describe()));
        return j.dump();
    }
    std::future<std::pair<int, std::string>> submit(std::unique_ptr<Job> job) {
        auto f = job->reply.get_future();
        std::lock_guard<std::mutex> l(m_);
        if (failed_) { job->reply.set_value({503, R"({"error":"service failed"})"}); return f; }
        if (q_.size() >= 16) { job->reply.set_value({503, R"({"error":"queue full"})"}); return f; }
        q_.push_back(std::move(job));
        cv_.notify_all();
        return f;
    }

private:
    void loop() {
        for (;;) {
            std::unique_ptr<Job> job;
            {
                std::unique_lock<std::mutex> l(m_);
                cv_.wait(l, [&] { return stop_ || !q_.empty(); });
                if (stop_ && q_.empty()) return;
                job = std::move(q_.front());
                q_.pop_front();
                busy_ = true;
            }
            auto r = run(*job);
            {
                std::lock_guard<std::mutex> l(m_);
                busy_ = false;
                ++served_;
            }
            job->reply.set_value(std::move(r));
        }
    }
    std::pair<int, std::string> run(Job& job) {
        const auto t_start = Clock::now();
        const std::uint64_t position = job.tokens.size() - 1;
        std::unique_ptr<strix::qwen_prefill_trace::Trace> trace;
        try {
            if (job.trace) trace = std::make_unique<strix::qwen_prefill_trace::Trace>();
            qp::RunOptions run_options;
            run_options.trace = trace.get();
            if (job.has_schedule) run_options.tail_large = job.tail_large;
            if (job.has_final_rows) run_options.final_block_rows = job.final_rows;
            qh::Options options = handoff_;
            options.trace = trace.get();
            if (job.has_codec) options.codec = job.codec;
            (void)qp::plan_blocks(position, config_.block_schedule,
                job.has_schedule ? job.tail_large : config_.tail_large,
                job.has_final_rows ? job.final_rows : config_.final_block_rows);
            qh::Sender sender(job.handoff, job.api_key.empty() ? handoff_key_ : job.api_key, 600, options);
            const auto& layout = sender.layout();
            if (layout.max_context && job.tokens.size() + 1 > layout.max_context)
                return {413, R"({"error":"prompt does not fit the node context"})"};
            const std::vector<std::uint32_t> prefix(job.tokens.begin(), job.tokens.begin() + static_cast<std::ptrdiff_t>(position));
            if (job.local_rows >= position) return {400, R"({"error":"local_rows must be below the prompt length - 1"})"};
            sender.begin(prefix, position, kNumerical, kReduction, kStateUpdate, job.local_rows);
            // Cooperative: the node prefills prompt[0, q] itself meanwhile (its prefix cache keeps the checkpoint at q).
            std::future<std::pair<int, std::string>> node_prefix;
            double node_prefix_s = 0;
            if (job.local_rows) {
                std::string url = job.node_completions;
                if (url.empty()) {
                    const auto u = http::parse_url(job.handoff);
                    url = "http://" + u.host + ":" + u.port + "/v1/completions";
                }
                std::string body = "{\"prompt\":[";
                for (std::uint64_t i = 0; i <= job.local_rows; ++i) { if (i) body += ','; body += std::to_string(job.tokens[i]); }
                body += "],\"max_tokens\":1,\"temperature\":0}";
                const auto key = job.api_key.empty() ? handoff_key_ : job.api_key;
                node_prefix = std::async(std::launch::async, [url, body = std::move(body), key, &node_prefix_s] {
                    const auto t0 = Clock::now();
                    const auto u = http::parse_url(url);
                    http::Client client("http://" + u.host + ":" + u.port, 600);
                    http::Headers h{{"Content-Type", "application/json"}};
                    if (!key.empty()) h.emplace_back("Authorization", "Bearer " + key);
                    auto r = client.request("POST", u.path, h,
                        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(body.data()), body.size()));
                    node_prefix_s = secs(t0, Clock::now());
                    return r;
                });
            }
            const auto t_prefill = Clock::now();
            const auto timing = p_.run(job.tokens, position, [&](const qp::Piece& piece) {
                qh::Piece out;
                out.fixed = piece.fixed;
                out.component = piece.fixed ? sender.fixed_id(piece.component) : sender.row_id(piece.component);
                out.first = piece.first;
                out.count = piece.count;
                out.data = piece.data;
                out.bytes = piece.bytes;
                out.trace_id = piece.trace_id;
                sender.enqueue(out);
            }, run_options);
            const auto t_computed = Clock::now();
            const std::string node = sender.commit();
            if (node_prefix.valid()) {
                const auto [status, body] = node_prefix.get();
                if (status != 200) throw std::runtime_error("node prefix completion: HTTP " + std::to_string(status) + " " + body.substr(0, 300));
            }
            const auto t_done = Clock::now();
            Json reply = Json::object();
            reply.set("session", job.session);
            reply.set("position", Json(position));
            reply.set("tokens", Json(static_cast<std::uint64_t>(job.tokens.size())));
            Json s = Json::object();
            s.set("queue", Json(secs(job.queued_at, t_start)));
            s.set("setup", Json(secs(t_start, t_prefill)));
            s.set("prefill", Json(timing.total_s));
            s.set("first_block", Json(timing.first_block_s));
            s.set("transfer_tail", Json(secs(t_computed, t_done)));
            s.set("send", Json(sender.timing().send_s));
            s.set("commit", Json(sender.timing().commit_s));
            s.set("encode", Json(sender.timing().encode_s));
            s.set("total", Json(secs(job.queued_at, t_done)));
            if (job.local_rows) s.set("node_prefix", Json(node_prefix_s));
            reply.set("seconds", s);
            if (trace) {
                reply.set("trace", trace_json(*trace));
            }
            reply.set("state_bytes", Json(sender.timing().bytes));
            reply.set("chunks", Json(sender.timing().chunks));
            reply.set("wire_bytes", Json(sender.timing().wire_bytes));
            reply.set("encoding", Json(sender.timing().encoding));
            if (job.local_rows) reply.set("local_rows", Json(job.local_rows));
            try { reply.set("import", strix::serve::parse_json(node)); } catch (...) { reply.set("import", Json(node)); }
            return {200, reply.dump()};
        } catch (const std::exception& e) {
            Json err = Json::object();
            err.set("error", Json(std::string(e.what())));
            if (trace) err.set("trace", trace_json(*trace));
            const std::string what = e.what();
            // a device error leaves the GPU state unknown: refuse further work
            if (what.find("cuda") != std::string::npos || what.find("CUDA") != std::string::npos) {
                std::lock_guard<std::mutex> l(m_);
                failed_ = true;
                return {503, err.dump()};
            }
            if (dynamic_cast<const std::invalid_argument*>(&e)) return {400, err.dump()};
            return {502, err.dump()};
        }
    }

    qp::Prefill& p_;
    qp::Config config_;
    std::uint32_t max_positions_;
    std::string handoff_key_;
    qh::Options handoff_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::unique_ptr<Job>> q_;
    bool busy_ = false, failed_ = false, stop_ = false;
    std::uint64_t served_ = 0;
    std::thread worker_;
};

void handle(Service& svc, http::Sock s, std::uint32_t max_positions, const std::string& api_key) {
    http::set_timeouts(s, 3600);
    http::Message m;
    std::string carry;
    const auto err = http::read_message(s, m, std::size_t{64} << 20, false, carry);
    if (!err.empty()) { http::respond(s, 400, R"({"error":")" + err + "\"}"); http::close_sock(s); return; }
    const auto sp1 = m.start.find(' '), sp2 = m.start.find(' ', sp1 + 1);
    const std::string method = m.start.substr(0, sp1), path = m.start.substr(sp1 + 1, sp2 - sp1 - 1);
    if (path == "/health") { http::respond(s, 200, svc.health()); http::close_sock(s); return; }
    if (!api_key.empty()) {
        const auto auth = m.header("authorization"), key = m.header("x-api-key");
        if (auth != "Bearer " + api_key && key != api_key) { http::respond(s, 401, R"({"error":"unauthorized"})"); http::close_sock(s); return; }
    }
    if (path != "/v1/prefill" || method != "POST") { http::respond(s, 404, R"({"error":"not found"})"); http::close_sock(s); return; }
    auto job = std::make_unique<Job>();
    job->queued_at = Clock::now();
    try {
        const auto j = strix::serve::parse_json(m.body);
        for (const auto& t : j.find("tokens")->as_array()) job->tokens.push_back(static_cast<std::uint32_t>(t.as_int()));
        job->handoff = j.find("handoff")->as_string();
        if (const auto* x = j.find("session")) job->session = *x;
        if (const auto* x = j.find("handoff_api_key")) job->api_key = x->as_string();
        if (const auto* x = j.find("handoff_codec")) { job->codec = x->as_string(); job->has_codec = true; }
        if (const auto* x = j.find("local_rows")) job->local_rows = static_cast<std::uint64_t>(x->as_int());
        if (const auto* x = j.find("trace")) job->trace = x->as_bool();
        if (const auto* x = j.find("block_schedule")) {
            const auto schedule = x->as_string();
            if (schedule != "default" && schedule != "tail-large") throw std::invalid_argument("block_schedule");
            job->has_schedule = true; job->tail_large = schedule == "tail-large";
        }
        if (const auto* x = j.find("final_block_rows")) {
            auto n = x->as_int();
            if (n < 0 || n > qp::kMaxBlockRows) throw std::invalid_argument("final_block_rows");
            job->has_final_rows = true; job->final_rows = static_cast<std::uint32_t>(n);
        }
        if (const auto* x = j.find("node_completions")) job->node_completions = x->as_string();
    } catch (const std::exception& e) {
        http::respond(s, 400, R"({"error":"bad request body"})");
        http::close_sock(s);
        return;
    }
    if (job->tokens.size() < 2) { http::respond(s, 400, R"({"error":"need at least 2 tokens"})"); http::close_sock(s); return; }
    if (job->tokens.size() - 1 > max_positions) { http::respond(s, 413, R"({"error":"prompt too long"})"); http::close_sock(s); return; }
    auto f = svc.submit(std::move(job));
    const auto [status, body] = f.get();
    http::respond(s, status, body);
    http::close_sock(s);
}

}  // namespace

int main(int argc, char** argv) {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared card, restricts the process to the chosen one and prints the "device:" receipt line.
    strix::device::bind_cuda_main(argc, argv);
    qp::Config c;
    c.model_gguf = "Qwen3.8-Flash-Next-UD-Q4_K_XL-merged.gguf";
    std::string bind = "127.0.0.1", port = "8096", api_key, handoff_key;
    qh::Options handoff;
    // Defaults (measured net2 2026-10-24: exact and faster at every length 1K-32K): bp-v1 chunks when the node lists the
    // codec (else plain), 2 connections. --handoff-codec identity --handoff-connections 1 restores the plain stream.
    handoff.codec = "bp-v1";
    handoff.connections = 2;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(1); } return argv[++i]; };
        if (a == "--model") c.model_gguf = next();
        else if (a == "--mtp-sidecar") c.mtp_sidecar_gguf = next();
        else if (a == "--tables") c.tables_dir = next();
        else if (a == "--rope-table") c.rope_table = next();
        else if (a == "--max-positions") c.max_positions = static_cast<std::uint32_t>(std::atoi(next().c_str()));
        else if (a == "--block-schedule") {
            const auto v = next();
            if (v != "default" && v != "tail-large") { std::fprintf(stderr, "invalid block schedule\n"); return 1; }
            c.tail_large = v == "tail-large";
        }
        else if (a == "--final-block-rows") {
            const auto v = next();
            long long n = -1;
            try { std::size_t used = 0; n = std::stoll(v, &used); if (used != v.size()) n = -1; } catch (const std::exception&) { n = -1; }
            if (n < 0 || n > qp::kMaxBlockRows) { std::fprintf(stderr, "invalid final block rows\n"); return 1; }
            c.final_block_rows = static_cast<std::uint32_t>(n);
        }
        else if (a == "--blocks") {
            c.block_schedule.clear();
            std::string v = next();
            for (std::size_t at = 0; at < v.size();) {
                const auto comma = v.find(',', at);
                c.block_schedule.push_back(static_cast<std::uint32_t>(std::atoi(v.substr(at, comma - at).c_str())));
                at = comma == std::string::npos ? v.size() : comma + 1;
            }
        }
        else if (a == "--release-by-layer") c.release_by_layer = true;
        else if (a == "--no-release-by-layer") c.release_by_layer = false;
        else if (a == "--bind") bind = next();
        else if (a == "--port") port = next();
        else if (a == "--api-key-file") { std::FILE* f = std::fopen(next().c_str(), "rb"); char b[256] = {}; if (f) { std::fread(b, 1, 255, f); std::fclose(f); } api_key = b; while (!api_key.empty() && (api_key.back() == '\n' || api_key.back() == '\r')) api_key.pop_back(); }
        else if (a == "--handoff-api-key-file") { std::FILE* f = std::fopen(next().c_str(), "rb"); char b[256] = {}; if (f) { std::fread(b, 1, 255, f); std::fclose(f); } handoff_key = b; while (!handoff_key.empty() && (handoff_key.back() == '\n' || handoff_key.back() == '\r')) handoff_key.pop_back(); }
        else if (a == "--handoff-codec") handoff.codec = next();
        else if (a == "--handoff-encode-threads") handoff.encode_threads = static_cast<unsigned>(std::atoi(next().c_str()));
        else if (a == "--handoff-connections") handoff.connections = static_cast<unsigned>(std::max(1, std::atoi(next().c_str())));
        else if (a == "--handoff-chunk-mib") handoff.chunk_bytes = static_cast<std::uint64_t>(std::max(1, std::atoi(next().c_str()))) << 20;
        else { std::fprintf(stderr, "unknown flag %s\n", a.c_str()); return 1; }
    }
    if (c.tables_dir.empty() || c.rope_table.empty()) { std::fprintf(stderr, "--tables and --rope-table are required\n"); return 1; }
    if (c.block_schedule.empty() ||
        c.final_block_rows > *std::max_element(c.block_schedule.begin(), c.block_schedule.end())) {
        std::fprintf(stderr, "invalid final block rows\n");
        return 1;
    }
    http::startup();
    // Bind before loading so a busy port fails fast.
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    addrinfo* res = nullptr;
    if (getaddrinfo(bind.c_str(), port.c_str(), &hints, &res) != 0 || !res) { std::fprintf(stderr, "cannot resolve %s\n", bind.c_str()); return 2; }
    http::Sock ls = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one), sizeof(one));
    if (ls == http::kNoSock || ::bind(ls, res->ai_addr, static_cast<int>(res->ai_addrlen)) != 0 || ::listen(ls, 16) != 0) {
        std::fprintf(stderr, "cannot listen on %s:%s\n", bind.c_str(), port.c_str());
        return 2;
    }
    freeaddrinfo(res);
    std::printf("{\"event\":\"loading\"}\n");
    std::fflush(stdout);
    std::unique_ptr<qp::Prefill> loaded;
    try { loaded = std::make_unique<qp::Prefill>(c); }
    catch (const std::exception& e) { std::fprintf(stderr, "invalid configuration: %s\n", e.what()); return 1; }
    qp::Prefill& p = *loaded;
    std::printf("{\"event\":\"loaded\",\"describe\":%s}\n", p.describe().c_str());
    Service svc(p, c, handoff_key, handoff);
    std::printf("{\"event\":\"READY\",\"bind\":\"%s\",\"port\":%s}\n", bind.c_str(), port.c_str());
    std::fflush(stdout);
    for (;;) {
        const http::Sock s = ::accept(ls, nullptr, nullptr);
        if (s == http::kNoSock) continue;
        std::thread([&svc, s, &c, api_key] { handle(svc, s, c.max_positions, api_key); }).detach();
    }
}
