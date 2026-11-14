// Armed instrumentation of the serve layer (no GPU): the armed flag (GET /metrics, GET /api/status?detail=1), request
// ids (X-Request-Id accepted, echoed, used as the ring id), per-request prompt accounting (prompt = cached + imported +
// computed), first-token times, the fixed-bucket histograms of the queue (run() and take()/finish() paths) and the
// metrics text they render.
#include "strix/serve/api.hpp"
#include "strix/serve/http_server.hpp"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#define MSG_NOSIGNAL 0
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <atomic>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <thread>

namespace {
#ifdef _WIN32
using TestSocket = SOCKET;
constexpr TestSocket invalid_socket = INVALID_SOCKET;
void close_socket(TestSocket fd) { (void)::closesocket(fd); }
int socket_option(TestSocket fd,int level,int option,const void* value,int size) {
    return ::setsockopt(fd,level,option,static_cast<const char*>(value),size);
}
struct SocketRuntime {
    SocketRuntime() { WSADATA data{}; if (WSAStartup(MAKEWORD(2,2),&data)) throw std::runtime_error("Winsock initialization failed"); }
    ~SocketRuntime() { WSACleanup(); }
};
#else
using TestSocket = int;
constexpr TestSocket invalid_socket = -1;
void close_socket(TestSocket fd) { (void)::close(fd); }
int socket_option(TestSocket fd,int level,int option,const void* value,socklen_t size) {
    return ::setsockopt(fd,level,option,value,size);
}
struct SocketRuntime {};
#endif
[[maybe_unused]] void set_receive_timeout(TestSocket fd,unsigned seconds) {
#ifdef _WIN32
    const DWORD timeout=seconds*1000;
#else
    timeval timeout{};timeout.tv_sec=seconds;
#endif
    (void)socket_option(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
}
using namespace strix::serve;
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::cerr << "FAIL: " << what << '\n'; }
}

// Byte alphabet and the chat specials (types: 1 normal, 3 control, 4 user-defined).
std::shared_ptr<const BpeTokenizer> tiny_tokenizer() {
    std::vector<std::string> tokens;
    std::vector<std::int32_t> types;
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        int cp = b;
        const bool direct = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        if (!direct) cp = 256 + extra++;
        std::string s;
        if (cp < 0x80) s.push_back(static_cast<char>(cp));
        else { s.push_back(static_cast<char>(0xC0 | (cp >> 6))); s.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
        tokens.push_back(s);
        types.push_back(1);
    }
    for (const char* special : {"<|endoftext|>", "<|im_start|>", "<|im_end|>"}) { tokens.push_back(special); types.push_back(3); }
    for (const char* special : {"<tool_call>", "</tool_call>", "<think>", "</think>"}) { tokens.push_back(special); types.push_back(4); }
    return BpeTokenizer::from_parts(std::move(tokens), std::move(types), {}, "qwen35");
}

// run() path: answers "ok" + the stop token; reports 2 cached and 3 imported prompt tokens, a first token 5 ms after
// generate() starts (only while armed, as engines do) and a decode time of 10 ms. "FAIL" in the prompt throws.
class FixedEngine final : public GenerationEngine {
public:
    explicit FixedEngine(std::shared_ptr<const BpeTokenizer> t) : tokenizer_(std::move(t)) {}
    GenerationStats generate(GenerationJob& job) override {
        const auto text = tokenizer_->decode(job.prompt);
        if (text.find("FAIL") != std::string::npos) throw std::runtime_error("scripted failure");
        auto ids = tokenizer_->encode("ok", false);
        ids.push_back(tokenizer_->find("<|im_end|>"));
        GenerationStats stats;
        stats.prompt_tokens = job.prompt.size();
        stats.cached_tokens = 2;
        stats.imported_tokens = 3;
        stats.generated_tokens = ids.size();
        stats.prefill_ms = 4.0;
        stats.decode_ms = 10.0;
        stats.finish = "stop";
        if (observe_armed()) stats.first_token_ms = 5.0;
        if (text.find("IMPORTED") != std::string::npos) {  // a 9,000-token prompt with 8,192 rows imported, 1 s of prefill
            stats.prompt_tokens = 9000;
            stats.cached_tokens = 0;
            stats.imported_tokens = 8192;
            stats.prefill_ms = 1000.0;
        }
        (void)job.on_tokens(ids);
        return stats;
    }
    std::uint64_t context_tokens() const override { return 4096; }
private:
    std::shared_ptr<const BpeTokenizer> tokenizer_;
};

const Json& at(const Json& json, std::string_view key) {
    if (const auto* value = json.find(key)) return *value;
    throw std::runtime_error("missing key " + std::string(key) + " in " + json.dump());
}
struct Reply { int status = 0; std::string headers, body; };
TestSocket connect_to(std::uint16_t port) {
    const TestSocket fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) { close_socket(fd); return invalid_socket; }
    return fd;
}
std::string request_text(const std::string& method, const std::string& path, const std::string& body, const std::string& extra, bool close) {
    return method + " " + path + " HTTP/1.1\r\nHost: x\r\n" + (close ? "Connection: close\r\n" : "") + "Content-Type: application/json\r\n" + extra +
        "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
}
Reply parse_reply(const std::string& raw) {
    Reply out;
    const auto end = raw.find("\r\n\r\n");
    if (end == std::string::npos) return out;
    out.headers = raw.substr(0, end);
    out.status = std::atoi(raw.c_str() + 9);
    auto rest = raw.substr(end + 4U);
    if (out.headers.find("Transfer-Encoding: chunked") == std::string::npos) { out.body = rest; return out; }
    for (std::size_t pos = 0; pos < rest.size();) {
        const auto line = rest.find("\r\n", pos);
        if (line == std::string::npos) break;
        const auto size = std::stoul(rest.substr(pos, line - pos), nullptr, 16);
        if (!size) break;
        out.body += rest.substr(line + 2U, size);
        pos = line + 2U + size + 2U;
    }
    return out;
}
Reply http(std::uint16_t port, const std::string& method, const std::string& path, const std::string& body = {}, const std::string& extra = {}) {
    const TestSocket fd = connect_to(port);
    if (fd == invalid_socket) return {};
    const auto request = request_text(method, path, body, extra, true);
    (void)::send(fd, request.data(), request.size(), MSG_NOSIGNAL);
    std::string raw;
    char buffer[4096];
    for (;;) {
        const auto n = ::recv(fd, buffer, sizeof(buffer), 0);
        if (n <= 0) break;
        raw.append(buffer, static_cast<std::size_t>(n));
    }
    close_socket(fd);
    return parse_reply(raw);
}
// Requests on one keep-alive connection (the server answers them on the same thread), Content-Length replies; the
// last one closes the connection.
std::vector<Reply> http_sequence(std::uint16_t port, const std::string& path, const std::string& body, const std::vector<std::string>& extras) {
    const TestSocket fd = connect_to(port);
    if (fd == invalid_socket) return {};
    std::vector<Reply> out;
    std::string raw;
    char buffer[4096];
    const auto read_one = [&]() {
        for (;;) {
            const auto end = raw.find("\r\n\r\n");
            if (end != std::string::npos) {
                const auto cl = raw.find("Content-Length: ");
                if (cl != std::string::npos && cl < end) {
                    const auto length = std::stoul(raw.substr(cl + 16U));
                    if (raw.size() >= end + 4U + length) {
                        const auto one = raw.substr(0, end + 4U + length);
                        raw.erase(0, one.size());
                        return parse_reply(one);
                    }
                }
            }
            const auto n = ::recv(fd, buffer, sizeof(buffer), 0);
            if (n <= 0) return Reply{};
            raw.append(buffer, static_cast<std::size_t>(n));
        }
    };
    for (std::size_t i = 0; i < extras.size(); ++i) {
        const auto request = request_text("POST", path, body, extras[i], i + 1U == extras.size());
        (void)::send(fd, request.data(), request.size(), MSG_NOSIGNAL);
        out.push_back(read_one());
    }
    close_socket(fd);
    return out;
}
std::string header_value(const Reply& r, const std::string& name) {
    const auto key = "\r\n" + name + ": ";
    const auto at_key = r.headers.find(key);
    if (at_key == std::string::npos) return {};
    const auto from = at_key + key.size();
    return r.headers.substr(from, r.headers.find("\r\n", from) - from);
}
// The newest ring record of the status document.
Json newest(std::uint16_t port) {
    const auto status = parse_json(http(port, "GET", "/api/status").body);
    return at(status, "requests").as_array().at(0);
}

// Metrics text exposition format: every sample belongs to a family declared before it with one HELP and one TYPE; names are
// valid; histogram samples are <family>_bucket / _sum / _count.
bool valid_exposition(const std::string& text, std::string& why) {
    static const std::regex name_re("[a-zA-Z_:][a-zA-Z0-9_:]*");
    std::map<std::string, std::string> types;
    std::set<std::string> helps;
    std::size_t pos = 0;
    while (pos < text.size()) {
        const auto end = text.find('\n', pos);
        if (end == std::string::npos) { why = "no final newline"; return false; }
        const auto line = text.substr(pos, end - pos);
        pos = end + 1U;
        if (line.rfind("# HELP ", 0) == 0) {
            const auto name = line.substr(7, line.find(' ', 7) - 7);
            if (!helps.insert(name).second) { why = "two HELP lines for " + name; return false; }
            continue;
        }
        if (line.rfind("# TYPE ", 0) == 0) {
            const auto name = line.substr(7, line.find(' ', 7) - 7);
            const auto type = line.substr(line.find(' ', 7) + 1U);
            if (!std::regex_match(name, name_re)) { why = "bad family name " + name; return false; }
            if (!types.emplace(name, type).second) { why = "two TYPE lines for " + name; return false; }
            if (!helps.count(name)) { why = "TYPE before HELP for " + name; return false; }
            continue;
        }
        const auto name = line.substr(0, line.find_first_of("{ "));
        if (!std::regex_match(name, name_re)) { why = "bad sample name in: " + line; return false; }
        std::string family = name;
        if (!types.count(family))
            for (const char* suffix : {"_bucket", "_sum", "_count"}) {
                const std::string s(suffix);
                if (name.size() > s.size() && name.compare(name.size() - s.size(), s.size(), s) == 0) {
                    const auto base = name.substr(0, name.size() - s.size());
                    if (types.count(base) && types[base] == "histogram") family = base;
                }
            }
        if (!types.count(family)) { why = "sample without a TYPE: " + line; return false; }
    }
    return true;
}
// The value of one sample line (exact series text before the value), or -1.
double sample(const std::string& text, const std::string& series) {
    const auto key = "\n" + series + " ";
    auto pos = text.find(key);
    if (pos == std::string::npos && text.rfind(series + " ", 0) == 0) pos = 0;
    else if (pos == std::string::npos) return -1.0;
    else ++pos;
    return std::stod(text.substr(pos + series.size() + 1U));
}

// In-process responses (no socket): the body of a non-streamed reply.
struct Sink final : HttpResponseWriter {
    std::string body;
    int status = 0;
    void send(int s, std::string_view, std::string_view b, const std::vector<std::pair<std::string, std::string>>&) override { status = s; body = b; }
    bool begin_stream(int, std::string_view, const std::vector<std::pair<std::string, std::string>>&) override { return false; }
    bool write(std::string_view) override { return false; }
    void end_stream() override {}
    [[nodiscard]] bool client_gone() const override { return false; }
};
std::string get(ApiService& api, const std::string& path, const std::string& query = {}) {
    HttpRequest r;
    r.method = "GET";
    r.path = path;
    r.query = query;
    Sink s;
    api.handle(r, s);
    return s.body;
}
// One taken job finished with a prompt split, a first token and a decode time (as a batching engine reports them).
bool run_one(GenerationQueue& q, std::shared_ptr<GenerationTicket>* ticket_out = nullptr) {
    GenerationJob job;
    job.prompt.assign(10, 65U);
    auto ticket = q.submit(std::move(job));
    if (ticket_out) *ticket_out = ticket;
    auto taken = q.take(false);
    if (!taken) return false;
    GenerationStats st;
    st.prompt_tokens = 10;
    st.cached_tokens = 4;
    st.generated_tokens = 5;
    st.decode_ms = 100.0;
    if (taken->job.observed) st.first_token_ms = 30.0;
    q.finish(*taken, st);
    return true;
}
// Waits up to `limit` for the flag to read `armed`; the seconds it took, or -1.
double wait_armed(bool armed, double limit) {
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (observe_armed() == armed) return s;
        if (s > limit) return -1.0;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}
}  // namespace

int main() try {
    [[maybe_unused]] SocketRuntime sockets;
    const auto tokenizer = tiny_tokenizer();
    FixedEngine engine(tokenizer);
    GenerationQueue queue;
    ApiConfig config;
    config.model_id = "test-model";
    config.stop_tokens = {tokenizer->find("<|im_end|>")};
    ApiService api(config, tokenizer, queue, engine.context_tokens(), false);
    HttpServerConfig http_config;
    http_config.port = 0;
    HttpServer server(http_config, [&](const HttpRequest& r, HttpResponseWriter& w) { api.handle(r, w); });
    server.listen();
    const auto port = server.port();
    std::thread http_thread([&] { server.serve(); });
    std::thread engine_thread([&] { queue.run(engine); });
    const std::string chat = R"({"messages":[{"role":"user","content":"Hi"}]})";

    std::string unarmed_instance;
    try {
    // ---- unarmed: nothing new is reported, and only GET /metrics or ?detail=1 arms
    {
        const auto r = http(port, "POST", "/v1/chat/completions", chat);
        check(r.status == 200, "unarmed chat " + std::to_string(r.status));
        const auto status = parse_json(http(port, "GET", "/api/status").body);
        check(!status.find("observed") && !status.find("armed") && !status.find("armed_since"), "unarmed status has no armed members " + status.dump());
        check(status.find("instance") && at(at(status, "instance"), "id").as_string().size() == 16U &&
                  at(at(status, "instance"), "id").as_string().find_first_not_of("0123456789abcdef") == std::string::npos &&
                  at(at(status, "instance"), "start_time").is_number() && at(at(status, "instance"), "start_time").as_int() > 1000000000,
              "the instance identity is present unarmed " + status.dump());
        unarmed_instance = at(status, "instance").dump();
        const auto& e = at(status, "requests").as_array().at(0);
        check(!e.find("cached_tokens") && !e.find("first_token_ms") && !e.find("computed_tokens"), "unarmed ring record has no armed fields");
        const auto imported = http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"IMPORTED"}]})");
        const auto ie = at(parse_json(http(port, "GET", "/api/status").body), "requests").as_array().at(0);
        check(imported.status == 200 && at(ie, "prefill_tok_s").as_double() == 808.0,
              "prefill rate of the computed rows only: (9000 - 8192) / 1 s = 808 tok/s " + ie.dump());
        for (const char* q : {"/api/status?detail=0", "/api/status?details=1", "/api/status?detail=10", "/api/status?xdetail=1"})
            (void)http(port, "GET", q);
        check(!observe_armed(), "status queries other than detail=1 do not arm");
        check(queue.observed().requests == 0U, "nothing observed unarmed");
    }
    // ---- request ids (not armed-gated)
    {
        const std::string id = "0123456789abcdef.1";
        const auto r = http(port, "POST", "/v1/chat/completions", chat, "X-Request-Id: " + id + "\r\n");
        check(header_value(r, "X-Request-Id") == id, "chat echoes the request id [" + r.headers + "]");
        const auto body_id = at(parse_json(r.body), "id").as_string();
        check(body_id.rfind("chatcmpl-", 0) == 0, "the response body keeps its own id");
        check(at(newest(port), "id").as_string() == id, "the ring record carries the request id");
        for (const std::string& ok : {std::string("abcdefgh"), std::string(64, 'a'), std::string("A-b_c.9Z")}) {
            const auto rr = http(port, "POST", "/v1/completions", R"({"prompt":"Hi","max_tokens":4})", "X-Request-Id: " + ok + "\r\n");
            check(header_value(rr, "X-Request-Id") == ok && at(newest(port), "id").as_string() == ok, "accepted id " + ok);
        }
        for (const std::string& bad : {std::string("abcdefg"), std::string(65, 'a'), std::string("bad id!!"), std::string("abc:defgh"),
                 std::string("abc/defgh"), std::string("\xC3\xA9tiquette1")}) {
            const auto rr = http(port, "POST", "/v1/chat/completions", chat, "X-Request-Id: " + bad + "\r\n");
            check(rr.status == 200 && header_value(rr, "X-Request-Id").empty(), "rejected id is not echoed: " + bad);
            check(at(newest(port), "id").as_string().rfind("chatcmpl-", 0) == 0, "rejected id keeps the generated ring id: " + bad);
        }
        const auto stream = http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"Hi"}],"stream":true})",
            "X-Request-Id: stream-0001\r\n");
        check(stream.status == 200 && header_value(stream, "X-Request-Id") == "stream-0001" && stream.body.find("[DONE]") != std::string::npos,
            "streamed chat echoes the id");
        check(at(newest(port), "id").as_string() == "stream-0001", "streamed request ring id");
        const auto msg = http(port, "POST", "/v1/messages", R"({"max_tokens":8,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"Hi"}]})",
            "X-Request-Id: message-0001\r\n");
        check(msg.status == 200 && header_value(msg, "X-Request-Id") == "message-0001", "messages echoes the id " + std::to_string(msg.status) + " " + msg.body);
        const auto bad_json = http(port, "POST", "/v1/chat/completions", "{nope", "X-Request-Id: badjson-0001\r\n");
        check(bad_json.status == 400 && header_value(bad_json, "X-Request-Id") == "badjson-0001", "an error response echoes the id");
        const auto failed = http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"FAIL"}]})",
            "X-Request-Id: failed-00001\r\n");
        check(failed.status == 500 && header_value(failed, "X-Request-Id") == "failed-00001", "a failed generation echoes the id");
        const auto failed_record = newest(port);
        check(at(failed_record, "id").as_string() == "failed-00001" && at(failed_record, "status").as_string() == "error",
            "the failed request's ring record carries the id " + failed_record.dump());
        const auto seq = http_sequence(port, "/v1/chat/completions", chat,
            {"X-Request-Id: keepalive-01\r\n", "", "X-Request-Id: bad\r\n", "X-Request-Id: keepalive-04\r\n"});
        check(seq.size() == 4U && header_value(seq[0], "X-Request-Id") == "keepalive-01" && header_value(seq[1], "X-Request-Id").empty() &&
            header_value(seq[2], "X-Request-Id").empty() && header_value(seq[3], "X-Request-Id") == "keepalive-04",
            "ids do not leak between the requests of a keep-alive connection");
        {
            const auto status = parse_json(http(port, "GET", "/api/status").body);
            const auto& ring = at(status, "requests").as_array();
            check(at(ring.at(0), "id").as_string() == "keepalive-04" && at(ring.at(1), "id").as_string().rfind("chatcmpl-", 0) == 0 &&
                at(ring.at(2), "id").as_string().rfind("chatcmpl-", 0) == 0 && at(ring.at(3), "id").as_string() == "keepalive-01",
                "keep-alive ring ids " + status.dump());
        }
        {
            std::vector<Reply> replies(6);
            std::vector<std::thread> clients;
            for (std::size_t i = 0; i < replies.size(); ++i)
                clients.emplace_back([&, i] {
                    replies[i] = http(port, "POST", "/v1/chat/completions", chat, "X-Request-Id: parallel-" + std::to_string(1000 + i) + "\r\n");
                });
            for (auto& c : clients) c.join();
            bool own = true;
            for (std::size_t i = 0; i < replies.size(); ++i) own = own && header_value(replies[i], "X-Request-Id") == "parallel-" + std::to_string(1000 + i);
            check(own, "concurrent connections echo their own ids");
            const auto status = parse_json(http(port, "GET", "/api/status").body);
            std::set<std::string> ids;
            for (const auto& e : at(status, "requests").as_array()) ids.insert(at(e, "id").as_string());
            bool all = true;
            for (std::size_t i = 0; i < replies.size(); ++i) all = all && ids.count("parallel-" + std::to_string(1000 + i));
            check(all, "concurrent requests have their own ring records");
        }
        check(!observe_armed(), "request ids do not arm");
    }
    // ---- a job submitted before arming that finishes after it is not observed (its measurements would be partial)
    GenerationQueue pre;
    GenerationJob pre_job;
    pre_job.prompt.assign(6, 65U);
    auto pre_ticket = pre.submit(std::move(pre_job));
    check(pre_ticket && !pre_ticket->observed(), "a job submitted unarmed is not observed");
    // ---- ?detail=1 arms (any position in the query)
    {
        const auto s = http(port, "GET", "/api/status?x=1&detail=1");
        check(s.status == 200 && observe_armed(), "GET /api/status?detail=1 arms");
        const auto status = parse_json(s.body);
        check(status.find("instance") && at(at(status, "instance"), "id").as_string().size() == 16U, "armed status has the instance id");
        check(at(status, "instance").dump() == unarmed_instance, "arming does not change the instance identity");
        check(at(status, "armed").as_bool() && at(status, "armed_since").is_number(), "armed since");
        auto pre_taken = pre.take(false);
        check(pre_taken && !pre_taken->job.observed, "the job keeps its unobserved mark");
        GenerationStats pre_stats;
        pre_stats.prompt_tokens = 6;
        pre_stats.generated_tokens = 3;
        pre_stats.decode_ms = 100.0;
        pre_stats.first_token_ms = 20.0;
        pre.finish(*pre_taken, pre_stats);
        check(pre.observed().requests == 0U && pre.observed().ttft_seconds.count == 0U, "a job that spans the arming is not counted");
        check(at(at(status, "observed"), "requests").as_int() == 0, "nothing observed before arming");
    }
    // ---- armed: C8 split, first token, histograms
    {
        const auto before = queue.observed();
        const auto r = http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"Hello there"}]})",
            "X-Request-Id: armed-000001\r\n");
        check(r.status == 200, "armed chat");
        const auto e = newest(port);
        const auto prompt = static_cast<std::uint64_t>(at(e, "prompt_tokens").as_int());
        check(at(e, "cached_tokens").as_int() == 2 && at(e, "imported_tokens").as_int() == 3 &&
            static_cast<std::uint64_t>(at(e, "computed_tokens").as_int()) == prompt - 5U, "C8 split in the ring record " + e.dump());
        check(at(e, "reused_tokens").as_int() == 2, "the legacy reused_tokens name stays");
        const double ftm = at(e, "first_token_ms").as_double();
        check(ftm >= 5.0 && ftm == 5.0 + at(e, "queue_ms").as_double(), "run() makes first_token_ms submit-relative " + e.dump());
        check(at(e, "response_id").as_string().rfind("chatcmpl-", 0) == 0, "response_id names the body id");
        const auto after = queue.observed();
        check(after.requests == before.requests + 1U && after.prompt_tokens == before.prompt_tokens + prompt &&
            after.cached_tokens == before.cached_tokens + 2U && after.imported_tokens == before.imported_tokens + 3U &&
            after.computed_tokens == before.computed_tokens + prompt - 5U, "observed C8 totals");
        check(after.ttft_seconds.count == before.ttft_seconds.count + 1U && after.ttft_seconds.counts[0] == before.ttft_seconds.counts[0] + 1U,
            "the first token time goes to the 0.05 s bucket");
        // 3 generated tokens in 10 ms = 300 tok/s: above the last bound
        check(after.decode_tokens_per_second.counts.back() == before.decode_tokens_per_second.counts.back() + 1U, "decode rate above the last bound");
        const auto failed_before = queue.observed().requests;
        (void)http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"FAIL"}]})");
        check(queue.observed().requests == failed_before, "failed requests are not observed");
        const auto metrics = http(port, "GET", "/metrics").body;
        std::string why;
        check(valid_exposition(metrics, why), "metrics are valid exposition text: " + why);
        check(sample(metrics, "strix_instrumentation_armed") == 1.0, "armed gauge");
        check(metrics.find("e+") == std::string::npos, "no value in exponent notation (integers print as integers)");
        check(sample(metrics, "strix_ttft_seconds_sum") == queue.observed().ttft_seconds.sum &&
                  sample(metrics, "strix_decode_tokens_per_second_sum") == queue.observed().decode_tokens_per_second.sum,
              "histogram sums round-trip exactly");
        check(metrics.find("strix_ttft_seconds_bucket{le=\"0.05\"}") != std::string::npos &&
                  metrics.find("strix_decode_tokens_per_second_bucket{le=\"2\"}") != std::string::npos, "bucket bounds as written");
        check(metrics.find("strix_engine_info{instance=\"") != std::string::npos && metrics.find("model=\"test-model\"} 1\n") != std::string::npos,
            "engine info");
        check(sample(metrics, "strix_prompt_tokens_imported_total") == static_cast<double>(queue.observed().imported_tokens), "imported counter");
        check(sample(metrics, "strix_prompt_tokens_computed_total") == static_cast<double>(queue.observed().computed_tokens), "computed counter");
        const auto ob = queue.observed();
        check(sample(metrics, "strix_ttft_seconds_count") == static_cast<double>(ob.ttft_seconds.count) &&
            sample(metrics, "strix_ttft_seconds_bucket{le=\"+Inf\"}") == static_cast<double>(ob.ttft_seconds.count), "+Inf bucket equals the count");
        double previous = 0.0;
        bool monotone = true;
        for (const auto bound : ttft_seconds_bounds) {
            std::ostringstream series;
            series << "strix_ttft_seconds_bucket{le=\"" << bound << "\"}";
            const double v = sample(metrics, series.str());
            monotone = monotone && v >= previous;
            previous = v;
        }
        check(monotone && previous <= sample(metrics, "strix_ttft_seconds_count"), "cumulative buckets");
        check(sample(metrics, "strix_decode_tokens_per_second_count") == static_cast<double>(ob.decode_tokens_per_second.count), "decode count");
        const auto status = parse_json(http(port, "GET", "/api/status").body);
        const auto& o = at(status, "observed");
        check(at(o, "since").as_string() == "armed" && static_cast<std::uint64_t>(at(o, "requests").as_int()) == ob.requests, "status observed block");
        const auto& h = at(o, "ttft_seconds");
        const auto& buckets = at(h, "buckets").as_array();
        check(buckets.size() == ttft_seconds_bounds.size() + 1U && at(buckets.back(), "le").as_string() == "+Inf" &&
            static_cast<std::uint64_t>(at(buckets.back(), "count").as_int()) == ob.ttft_seconds.count &&
            static_cast<std::uint64_t>(at(h, "count").as_int()) == ob.ttft_seconds.count &&
            at(buckets.front(), "le").as_double() == 0.05 &&
            static_cast<std::uint64_t>(at(buckets.front(), "count").as_int()) == ob.ttft_seconds.counts[0], "status histogram (cumulative buckets)");
    }
    } catch (const std::exception& error) {
        check(false, std::string("exception: ") + error.what());
    }
    server.stop();
    queue.shutdown();
    http_thread.join();
    engine_thread.join();

    // ---- take()/finish() path (batching engines): first_token_ms reported submit-relative by the engine
    {
        GenerationQueue batch;
        const auto before = batch.observed();
        for (const std::uint32_t n : {7U, 9U, 11U}) {
            GenerationJob job;
            job.prompt.assign(n, 65U);
            (void)batch.submit(std::move(job));
        }
        // queued and cancelled before it ran: counted as a request, no prompt accounting
        GenerationJob cancelled_job;
        cancelled_job.prompt.assign(5, 65U);
        auto cancelled = batch.submit(std::move(cancelled_job));
        cancelled->cancel();
        std::vector<TakenJob> taken;
        while (auto t = batch.take(false)) taken.push_back(std::move(*t));
        check(taken.size() == 3U, "three jobs taken (the cancelled one finished by take)");
        const double firsts[] = {50.0, 120.0, 61000.0};  // ms: 0.05 s (bucket le=0.05), 0.12 s (le=0.25), above 60 s
        for (std::size_t i = 0; i < taken.size(); ++i) {
            GenerationStats st;
            st.prompt_tokens = taken[i].job.prompt.size();
            st.cached_tokens = i;  // 0, 1, 2
            st.generated_tokens = 10;
            st.decode_ms = 1000.0;  // 10 tok/s: bucket le=10
            st.first_token_ms = firsts[i];
            st.finish = "length";
            batch.finish(taken[i], st);
        }
        GenerationJob error_job;
        error_job.prompt.assign(3, 65U);
        (void)batch.submit(std::move(error_job));
        auto failed = batch.take(false);
        GenerationStats st;
        st.generated_tokens = 4;
        batch.finish(*failed, st, "engine failure");
        const auto o = batch.observed();
        check(o.requests == before.requests + 4U, "batched: three finished + one cancelled observed, the failed one not");
        check(o.prompt_tokens == 27U && o.cached_tokens == 3U && o.computed_tokens == 24U && o.imported_tokens == 0U, "batched C8 totals");
        check(o.ttft_seconds.count == 3U && o.ttft_seconds.counts[0] == 1U && o.ttft_seconds.counts[2] == 1U &&
            o.ttft_seconds.counts.back() == 1U, "batched first-token buckets (bounds are inclusive)");
        check(o.decode_tokens_per_second.count == 3U && o.decode_tokens_per_second.counts[3] == 3U, "batched decode buckets");
        check(o.ttft_seconds.sum == (50.0 + 120.0 + 61000.0) / 1000.0, "first-token sum");
    }
    // ---- one-token generations have no decode rate; label values are escaped
    {
        ObserveHistogram<3> h;
        const std::array<double, 3> b{1.0, 2.0, 3.0};
        for (const double v : {0.5, 1.0, 1.5, 3.0, 3.5, -1.0, std::numeric_limits<double>::quiet_NaN()}) h.observe(v, b);
        check(h.counts[0] == 2U && h.counts[1] == 1U && h.counts[2] == 1U && h.counts[3] == 1U && h.count == 5U && h.sum == 9.5,
            "histogram placement (negative and NaN ignored)");
        check(ObserveHistogram<3>::bucket(1.0, b) == 0U && ObserveHistogram<3>::bucket(3.5, b) == 3U &&
                  ObserveHistogram<3>::bucket(-1.0, b) == 4U && ObserveHistogram<3>::bucket(2e12, b) == 4U,
              "bucket of a value (above the last bound: N; not an observation: N + 1)");
        h.add(4U, 7.0);
        check(h.count == 5U && h.sum == 9.5, "adding a non-observation changes nothing");
        GenerationQueue q;
        GenerationJob job;
        job.prompt.assign(4, 65U);
        (void)q.submit(std::move(job));
        auto t = q.take(false);
        GenerationStats one;
        one.prompt_tokens = 4;
        one.generated_tokens = 1;
        one.decode_ms = 0.0;
        q.finish(*t, one);
        check(q.observed().decode_tokens_per_second.count == 0U && q.observed().ttft_seconds.count == 0U, "no rate for one token, no first token reported");
        ApiConfig c2;
        c2.model_id = "m\"x\\y\nz";
        GenerationQueue q2;
        ApiService api2(c2, tokenizer, q2, 4096, false);
        struct Capture final : HttpResponseWriter {
            std::string body, headers;
            void send(int, std::string_view, std::string_view b, const std::vector<std::pair<std::string, std::string>>& h) override {
                body = b;
                for (const auto& [k, v] : h) headers += k + ": " + v + "\n";
            }
            bool begin_stream(int, std::string_view, const std::vector<std::pair<std::string, std::string>>&) override { return false; }
            bool write(std::string_view) override { return false; }
            void end_stream() override {}
            [[nodiscard]] bool client_gone() const override { return false; }
            void add_response_header(std::string k, std::string v) override { headers += k + ": " + v + "\n"; }
        } capture;
        HttpRequest get;
        get.method = "GET";
        get.path = "/metrics";
        get.headers = {{"x-request-id", "metrics-0001"}};
        api2.handle(get, capture);
        check(capture.body.find("model=\"m\\\"x\\\\y\\nz\"} 1\n") != std::string::npos, "label value escaping");
        check(capture.headers == "X-Request-Id: metrics-0001\n", "the id is echoed on GET responses too");
        std::string why;
        check(valid_exposition(capture.body, why), "escaped metrics valid: " + why);
    }
    // ---- the arming expires a fixed time after the last arming request; the next one arms again; observations stay
    {
        observe_disarm();  // the main section's period (60 s) ends here: a fresh start
        GenerationQueue q;
        ApiConfig c;
        c.model_id = "expiry";
        c.observe_expiry_seconds = 0.3;
        ApiService api3(c, tokenizer, q, 4096, false);
        (void)get(api3, "/metrics");
        check(observe_armed(), "a scrape arms (expiry 0.3 s)");
        check(run_one(q) && q.observed().requests == 1U && q.observed().ttft_seconds.count == 1U, "a request of the armed period is observed");
        // a request submitted armed that finishes after the expiry: not observed (its per-step measurements stopped)
        GenerationJob spanning;
        spanning.prompt.assign(8, 65U);
        auto spanning_ticket = q.submit(std::move(spanning));
        check(spanning_ticket->observed(), "submitted armed");
        const double expired_after = wait_armed(false, 5.0);
        check(expired_after >= 0.25 && expired_after < 5.0, "unarmed about 0.3 s after the scrape: " + std::to_string(expired_after));
        auto spanning_taken = q.take(false);
        GenerationStats st;
        st.prompt_tokens = 8;
        st.generated_tokens = 3;
        st.decode_ms = 50.0;
        st.first_token_ms = 10.0;
        q.finish(*spanning_taken, st);
        check(!spanning_ticket->observed() && q.observed().requests == 1U && q.observed().ttft_seconds.count == 1U,
            "a request whose armed period expired before it finished is not observed");
        // expired: exactly the unarmed surface
        const auto status = parse_json(get(api3, "/api/status"));
        check(!status.find("observed") && !status.find("armed") && !status.find("armed_since"), "expired status has no armed members " + status.dump());
        const auto metrics = get(api3, "/metrics");  // this scrape arms again
        check(observe_armed(), "the next scrape arms again");
        check(sample(metrics, "strix_instrumentation_armed") == 1.0 && sample(metrics, "strix_ttft_seconds_count") == 1.0,
            "re-armed: what was observed before the expiry stays");
        std::string why;
        check(valid_exposition(metrics, why), "re-armed metrics valid: " + why);
        check(run_one(q) && q.observed().requests == 2U, "a request of the new period is observed");
        // scraped more often than the expiry: never expires; plain status reads neither arm nor extend
        bool stayed = true;
        for (int i = 0; i < 10; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(60));
            (void)get(api3, "/api/status", i % 2 ? "detail=1" : "");
            if (i % 2 == 0) (void)get(api3, "/metrics");
            stayed = stayed && observe_armed();
        }
        check(stayed, "armed while scraped every 0.12 s or more often");
        const auto t_plain = std::chrono::steady_clock::now();
        bool plain_unarmed = false;
        while (std::chrono::steady_clock::now() - t_plain < std::chrono::seconds(5) && !plain_unarmed) {
            (void)get(api3, "/api/status");
            plain_unarmed = !observe_armed();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        check(plain_unarmed, "plain status polling does not keep the instrumentation armed");
        const auto again = parse_json(get(api3, "/api/status", "detail=1"));
        check(at(again, "armed").as_bool() && at(at(again, "observed"), "requests").as_int() == 2 &&
            at(at(at(again, "observed"), "ttft_seconds"), "count").as_int() == 2, "re-armed by ?detail=1, the observed totals kept " + again.dump());
        observe_disarm();
    }
    // ---- an expiry between a request's submit and its finish, deterministically: the engine's per-step load sees the
    // instrumentation unarmed (it stops measuring), and the finish never counts the request; also across a re-arm
    {
        observe_disarm();
        GenerationQueue q;
        const auto step = [&](TakenJob& t, std::uint64_t& measured) {  // one engine step: measures only while armed
            if (observe_armed() && t.job.observed) ++measured;
        };
        for (const int variant : {0, 1, 2}) {  // 0: armed throughout; 1: expired before the last step; 2: expired and re-armed
            (void)observe_arm(0.0);
            GenerationJob job;
            job.prompt.assign(5, 65U);
            auto ticket = q.submit(std::move(job));
            auto t = q.take(false);
            std::uint64_t measured = 0;
            step(*t, measured);
            if (variant >= 1) observe_disarm();
            if (variant == 2) (void)observe_arm(0.0);
            step(*t, measured);
            const auto before = q.observed().requests;
            check(t->job.observation_complete() == (variant == 0), "observation_complete() at the engine's finish, variant " + std::to_string(variant));
            GenerationStats st;
            st.prompt_tokens = 5;
            st.generated_tokens = 2;
            if (t->job.observation_complete()) st.proposed = measured;  // engines report armed-only measurements of a complete period
            q.finish(*t, st);
            const bool counted = q.observed().requests == before + 1U;
            const std::string v = std::to_string(variant);
            if (variant == 0) check(counted && ticket->observed() && measured == 2U, "armed throughout: counted with every step measured");
            else check(!counted && !ticket->observed(), "expiry before the finish (variant " + v + "): not counted (measured " +
                std::to_string(measured) + " of 2 steps)");
        }
        observe_disarm();
    }
    // ---- several services: an arming request keeps the period at least its service's expiry more (never shortens it);
    // expiry 0 keeps it until the process ends (or a disarm); the expiry does not depend on the service that armed
    {
        GenerationQueue qa, qb, qc;
        ApiConfig ca, cb, cc;
        ca.observe_expiry_seconds = 0.2;
        cb.observe_expiry_seconds = 1.0;
        cc.observe_expiry_seconds = 0.0;
        ApiService shorter(ca, tokenizer, qa, 4096, false), longer(cb, tokenizer, qb, 4096, false), never(cc, tokenizer, qc, 4096, false);
        const auto secs = [](std::chrono::steady_clock::time_point t0) {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        };
        observe_disarm();
        auto t0 = std::chrono::steady_clock::now();
        (void)get(longer, "/metrics");
        (void)get(shorter, "/metrics");  // a shorter expiry does not shorten the period
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        check(observe_armed(), "long then short: still armed at 0.5 s");
        (void)wait_armed(false, 5.0);
        const double t_long = secs(t0);
        check(!observe_armed() && t_long >= 0.95 && t_long < 5.0, "long then short: expires after about 1 s (" + std::to_string(t_long) + ")");
        observe_disarm();
        (void)get(never, "/metrics");
        (void)get(shorter, "/metrics");
        std::this_thread::sleep_for(std::chrono::milliseconds(600));
        check(observe_armed(), "expiry 0 then a short expiry: armed (never expires)");
        observe_disarm();
        check(!observe_armed(), "a disarm ends a period without an expiry");
        {  // the service that armed is destroyed while armed: the period still expires
            GenerationQueue qd;
            ApiConfig cd;
            cd.observe_expiry_seconds = 0.2;
            ApiService gone(cd, tokenizer, qd, 4096, false);
            (void)get(gone, "/metrics");
            check(observe_armed(), "armed by a service about to be destroyed");
        }
        t0 = std::chrono::steady_clock::now();
        check(wait_armed(false, 5.0) >= 0.0 && secs(t0) < 5.0, "the period of a destroyed service still expires");
    }
    // ---- concurrent scrapers, an engine and the expiry (run under ThreadSanitizer too): every request accounted once
    {
        observe_disarm();
        GenerationQueue q;
        ApiConfig c;
        c.model_id = "concurrent";
        c.observe_expiry_seconds = 0.02;
        const auto period_before = observe_period();
        ApiService api4(c, tokenizer, q, 4096, false);
        std::atomic<bool> stop{false};
        std::atomic<std::uint64_t> finished{0}, observed_tickets{0};
        std::thread engine4([&] {
            for (;;) {
                auto t = q.take(true);
                if (!t) {
                    if (q.stopping()) break;
                    continue;  // a cancellation-only batch returns control without shutting down
                }
                GenerationStats st;
                st.prompt_tokens = t->job.prompt.size();
                st.cached_tokens = 2;
                st.imported_tokens = 3;
                st.generated_tokens = 4;
                st.decode_ms = 20.0;
                if (t->job.observed) st.first_token_ms = 1.0;  // engines measure the requests submitted armed
                std::this_thread::sleep_for(std::chrono::microseconds(300));
                auto ticket = t->ticket;
                q.finish(*t, st);
                finished.fetch_add(1U);
                if (ticket->observed()) observed_tickets.fetch_add(1U);
            }
        });
        std::vector<std::thread> scrapers;
        const auto t_start = std::chrono::steady_clock::now();
        for (int k = 0; k < 6; ++k)
            scrapers.emplace_back([&, k] {
                while (!stop.load()) {
                    // arming scrapers pause in every other 60 ms window (longer than the expiry): the period expires and
                    // re-arms throughout the run; plain status readers never pause
                    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_start).count();
                    const bool quiet = (ms / 60) % 2 == 1;
                    if (k % 3 == 0 && !quiet) (void)get(api4, "/metrics");
                    else if (k % 3 == 1 && !quiet) (void)get(api4, "/api/status", "detail=1");
                    else if (k % 3 == 2) (void)get(api4, "/api/status");
                    std::this_thread::sleep_for(std::chrono::milliseconds(k < 3 ? 7 : 1));
                }
            });
        std::thread submitter([&] {
            for (int i = 0; i < 400; ++i) {
                GenerationJob job;
                job.prompt.assign(9, 65U);
                while (!q.submit(job)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            }
        });
        submitter.join();
        while (finished.load() < 400U) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        stop.store(true);
        for (auto& s : scrapers) s.join();
        q.shutdown();
        engine4.join();
        const auto o = q.observed();
        check(o.requests == observed_tickets.load() && o.requests <= 400U, "observed requests = tickets still observed at their end: " +
            std::to_string(o.requests) + " / " + std::to_string(observed_tickets.load()));
        check(o.prompt_tokens == 9U * o.requests && o.cached_tokens == 2U * o.requests && o.imported_tokens == 3U * o.requests &&
            o.computed_tokens == 4U * o.requests && o.ttft_seconds.count == o.requests, "every observed request accounted once, with its first token");
        check(observe_period() >= period_before + 4U, "the arming expired and re-armed under load (period " + std::to_string(period_before) +
            " -> " + std::to_string(observe_period()) + ")");
        observe_disarm();
    }
    std::cout << "serve observe tests: failures=" << failures << '\n';
    return failures ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << "serve observe tests: " << error.what() << '\n';
    return 1;
}
