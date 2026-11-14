// strix-cuda-ds41-multi-front: a strix/ds41-pipeline-multi-v1 FRONT (ds41_pipeline_multi.hpp) on the CUDA decode step for
// several requests at once. One CudaPrefill holds layers [0, K) and N sequence slots (CudaPrefill::set_sequences); a batch
// TAIL (strix-deepseek41-batch / strix-deepseek41-serve with --hetero-listen on the pair) drives it: every SEQ_STEP runs
// one slot's rows here (the slot's own state, the same step as a FRONT of one request, so its bits do not depend on the
// other slots) and its boundary goes back as SEQ_BOUNDARY while the TAIL computes the other requests' layers [K, 40).
//
//   strix-cuda-ds41-multi-front MODEL.gguf --stage HOST:PORT[,HOST:PORT..] --cut K --slots N --token-file FILE [--positions P]
//       [--tiers DD..] [--expert-tp --overlay OV.gguf] [--fuse-hc] [--fuse-shared] [--trellis-shexp PACK]
//       [--connect-wait S] [--sessions M] [--json OUT] [--blocking] [--no-boost] [--eager]
//
// Every boundary is sent straight from the step's mapped output buffer (header, tokens, stream, sections in one
// gathered send; the cut's sections are copied by the step's own graph), as the single-request FRONT does. --sessions M:
// serve M TAIL sessions one after the other (reconnecting; 0 = until killed); session i connects to the i-th --stage
// address and every later session to the last one (e.g. a local smoke TAIL first, then the pair's arms).
// Per session one "multi_summary" JSON line:
// steps, rows, medians of the CUDA step, the device time, the send and the wait for the next request.
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "strix/device/cuda_enumerator.hpp"
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <avrt.h>
#include <mmsystem.h>
#else
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "strix/cuda_ds41_prefill.hpp"
#include "strix/cuda_ds41_weights.hpp"
#include "strix/deepseek41_attention_state.hpp"
#include "strix/ds41_pipeline_multi.hpp"
#include "strix/ds41_pipeline_wire.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace strix::ds41::cuda;
namespace pipe = strix::ds41::pipe;
using Clock = std::chrono::steady_clock;

namespace {

[[noreturn]] void usage(const char* why) {
    std::fprintf(stderr, "strix-cuda-ds41-multi-front: %s\n", why);
    std::exit(2);
}
double ms_between(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v.size() % 2U ? v[v.size() / 2U] : 0.5 * (v[v.size() / 2U - 1U] + v[v.size() / 2U]);
}

// ---- client socket (TCP_NODELAY; spinning non-blocking receive unless --blocking), as strix-cuda-ds41-pipeline-front ----
#ifdef _WIN32
using sock_t = SOCKET;
constexpr sock_t kBad = INVALID_SOCKET;
int last_error() { return WSAGetLastError(); }
bool would_block(int e) { return e == WSAEWOULDBLOCK; }
void close_socket(sock_t s) { closesocket(s); }
#else
using sock_t = int;
constexpr sock_t kBad = -1;
int last_error() { return errno; }
bool would_block(int e) { return e == EAGAIN || e == EWOULDBLOCK || e == EINTR; }
void close_socket(sock_t s) { ::close(s); }
#endif

class Client {
public:
    Client(const std::string& host, std::uint16_t port, double wait_s, bool spin) {
#ifdef _WIN32
        static bool init = [] {
            WSADATA w;
            return WSAStartup(MAKEWORD(2, 2), &w) == 0;
        }();
        if (!init) throw std::runtime_error("WSAStartup failed");
#endif
        const auto deadline = Clock::now() + std::chrono::duration<double>(wait_s);
        for (;;) {
            addrinfo hints{}, *ai = nullptr;
            hints.ai_family = AF_INET;
            hints.ai_socktype = SOCK_STREAM;
            if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &ai) != 0) throw std::runtime_error("cannot resolve " + host);
            s_ = socket(AF_INET, SOCK_STREAM, 0);
            const bool ok = s_ != kBad && connect(s_, ai->ai_addr, static_cast<int>(ai->ai_addrlen)) == 0;
            freeaddrinfo(ai);
            if (ok) break;
            if (s_ != kBad) close_socket(s_);
            s_ = kBad;
            if (Clock::now() > deadline) throw std::runtime_error("cannot connect to " + host + ":" + std::to_string(port));
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        int one = 1;
        setsockopt(s_, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof one);
        int buf = 8 << 20;  // a whole micro-batch of boundaries fits in the send buffer (the step's output is reused after the send)
        setsockopt(s_, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<const char*>(&buf), sizeof buf);
        setsockopt(s_, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&buf), sizeof buf);
#ifdef _WIN32
        u_long nb = spin ? 1 : 0;
        ioctlsocket(s_, FIONBIO, &nb);
#else
        const int f = fcntl(s_, F_GETFL);
        fcntl(s_, F_SETFL, spin ? (f | O_NONBLOCK) : (f & ~O_NONBLOCK));
#endif
    }
    ~Client() {
        if (s_ != kBad) close_socket(s_);
    }
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    void send(pipe::MsgType type, std::uint64_t request, const std::vector<std::uint8_t>& payload, std::uint32_t flags = 0) {
        send_parts(type, request, {{payload.data(), payload.size()}}, flags);
    }
    // header + several payload pieces as one gathered send (no staging copy of the pieces); returns when every byte is
    // in the socket's send buffer
    void send_parts(pipe::MsgType type, std::uint64_t request, const std::vector<std::pair<const std::uint8_t*, std::size_t>>& parts,
                    std::uint32_t flags = 0) {
        std::uint64_t bytes = 0;
        for (const auto& [p, n] : parts) bytes += n;
        pipe::Header h;
        h.type = type;
        h.request = request;
        h.bytes = bytes;
        h.flags = flags;
        const auto head = pipe::encode_header(h);
        std::vector<std::pair<const std::uint8_t*, std::size_t>> all;
        all.emplace_back(head.data(), head.size());
        for (const auto& x : parts)
            if (x.second) all.push_back(x);
        std::size_t done = 0;
#ifdef _WIN32
        std::vector<WSABUF> b;
        for (const auto& [p, n] : all) b.push_back({static_cast<ULONG>(n), reinterpret_cast<CHAR*>(const_cast<std::uint8_t*>(p))});
        DWORD sent = 0;
        if (WSASend(s_, b.data(), static_cast<DWORD>(b.size()), &sent, 0, nullptr, nullptr) != 0) {
            if (!would_block(last_error())) throw std::runtime_error("send failed");
            sent = 0;
        }
        done = sent;
#endif
        std::size_t at = 0, base = 0;
        while (at < all.size()) {
            const auto [p, n] = all[at];
            if (done >= base + n) {
                base += n;
                ++at;
                continue;
            }
            const std::size_t off = done - base;
            const auto k = ::send(s_, reinterpret_cast<const char*>(p + off), static_cast<int>(n - off), 0);
            if (k > 0) {
                done += static_cast<std::size_t>(k);
                continue;
            }
            if (k < 0 && would_block(last_error())) continue;
            throw std::runtime_error("send failed");
        }
        sent_ += done;
    }
    pipe::Message receive() {
        std::array<std::uint8_t, pipe::kHeaderBytes> head{};
        read_all(head.data(), head.size());
        pipe::Message m;
        m.header = pipe::decode_header(head);
        m.payload.resize(static_cast<std::size_t>(m.header.bytes));
        read_all(m.payload.data(), m.payload.size());
        return m;
    }
    pipe::Message expect(pipe::MsgType type) {
        auto m = receive();
        if (m.header.type == pipe::MsgType::error) throw std::runtime_error("TAIL error: " + pipe::decode_text(m.payload));
        if (m.header.type != type) throw std::runtime_error(std::string("expected ") + pipe::to_string(type) + ", got " + pipe::to_string(m.header.type));
        return m;
    }
    [[nodiscard]] std::uint64_t bytes_sent() const noexcept { return sent_; }

private:
    void read_all(std::uint8_t* p, std::size_t n) {
        std::size_t got = 0;
        while (got < n) {
            const auto k = ::recv(s_, reinterpret_cast<char*>(p + got), static_cast<int>(n - got), 0);
            if (k > 0) {
                got += static_cast<std::size_t>(k);
                continue;
            }
            if (k == 0) throw std::runtime_error("TAIL closed the connection");
            if (would_block(last_error())) continue;  // spin
            throw std::runtime_error("receive failed");
        }
    }
    sock_t s_ = kBad;
    std::uint64_t sent_ = 0;
};

// The STEP payload of a cut after layer K - 1 for the step's R rows without its stream (the stream is gathered from the
// step's mapped output): tokens, mix_ffn@K-1, the KV entries the step completed on the KV sources < K, the rows' top-k of
// the index sources < K and their candidate blocks, all from the sections the step's graph copied (as
// strix-cuda-ds41-pipeline-front's build_step).
pipe::StepMsg boundary_without_stream(CudaPrefill& p, const CudaStepResult& s, std::uint32_t pos, const std::vector<std::uint32_t>& fed,
                                      std::uint32_t K, const pipe::BoundaryPlan& plan, const pipe::BoundaryGeometry& geo, const std::vector<float>& mix) {
    const auto R = static_cast<std::uint32_t>(fed.size());
    pipe::StepMsg msg;
    auto& b = msg.boundary;
    b.position = pos;
    b.rows = R;
    b.tokens = fed;
    {
        pipe::Section mx{pipe::SectionKind::mix_ffn, K - 1U, 0U, R, {}};
        mx.bytes.resize(std::size_t{R} * geo.mix_floats() * 4U);
        std::memcpy(mx.bytes.data(), mix.data(), mx.bytes.size());
        b.sections.push_back(std::move(mx));
    }
    for (const auto& src : plan.kv_sources) {
        const CudaStepSections::Kv* e = nullptr;
        for (const auto& k : s.sections.kv)
            if (k.layer == src.layer) e = &k;
        const std::uint32_t first = pos / src.ratio, n = (pos + R) / src.ratio - first;
        pipe::Section ks{pipe::SectionKind::kv_entries, src.layer, first, n, {}};
        if (e != nullptr) {
            if (e->first != first || e->count != n) throw std::logic_error("the step's KV section covers other entries");
            ks.bytes = e->kv;
            ks.bytes.insert(ks.bytes.end(), e->idx_keys.begin(), e->idx_keys.end());
        } else if (n != 0U) {
            std::vector<std::uint8_t> kv(std::size_t{n} * geo.kv_entry_bytes), ix(std::size_t{n} * geo.idx_entry_bytes);
            p.copy_kv_entries(src.layer, first, n, kv.data(), ix.data());
            ks.bytes = kv;
            ks.bytes.insert(ks.bytes.end(), ix.begin(), ix.end());
        }
        b.sections.push_back(std::move(ks));
    }
    for (const auto& src : plan.topk_sources) {
        std::vector<std::uint32_t> counts(R), ids(std::size_t{R} * geo.topk);
        const CudaStepSections::Topk* e = nullptr;
        for (const auto& k : s.sections.topk)
            if (k.layer == src.layer) e = &k;
        if (e != nullptr) {
            std::copy(e->counts.begin(), e->counts.begin() + R, counts.begin());
            std::copy(e->ids.begin(), e->ids.begin() + static_cast<std::ptrdiff_t>(std::size_t{R} * geo.topk), ids.begin());
        } else {
            p.copy_topk(src.layer, R, counts.data(), ids.data());
        }
        pipe::Section ts{pipe::SectionKind::topk, src.layer, 0U, R, {}};
        for (std::uint32_t r = 0; r < R; ++r) {
            const std::size_t at = ts.bytes.size();
            ts.bytes.resize(at + 4U * (1U + counts[r]));
            std::memcpy(ts.bytes.data() + at, &counts[r], 4);
            std::memcpy(ts.bytes.data() + at + 4, ids.data() + std::size_t{r} * geo.topk, 4U * counts[r]);
        }
        b.sections.push_back(std::move(ts));
    }
    if (plan.candidate_source >= 0) {
        const std::uint32_t count = (pos + R) / plan.candidate_ratio;
        const std::uint32_t stride = (count + geo.candidate_block - 1U) / geo.candidate_block;
        pipe::Section cs{pipe::SectionKind::candidates, static_cast<std::uint32_t>(plan.candidate_source), stride, R, {}};
        const std::uint32_t full = p.cand_stride();
        std::vector<std::uint8_t> rows(std::size_t{R} * full);
        if (s.sections.candidates.size() >= rows.size()) std::copy_n(s.sections.candidates.begin(), rows.size(), rows.begin());
        else p.copy_candidates(R, rows.data());
        for (std::uint32_t r = 0; r < R; ++r)
            cs.bytes.insert(cs.bytes.end(), rows.begin() + static_cast<std::ptrdiff_t>(std::size_t{r} * full),
                            rows.begin() + static_cast<std::ptrdiff_t>(std::size_t{r} * full + stride));
        b.sections.push_back(std::move(cs));
    }
    return msg;
}

// encode() of the STEP without its stream, split where the stream goes (payload layout: flags, position, rows, sections,
// u64 stream floats, tokens, stream, sections), so header + head + stream + rest is byte-identical to encode() with it.
struct Gathered {
    std::vector<std::uint8_t> rest;
    std::size_t split = 0;
    const std::uint8_t* stream = nullptr;
    std::size_t stream_bytes = 0;
    [[nodiscard]] std::vector<std::pair<const std::uint8_t*, std::size_t>> parts() const {
        return {{rest.data(), split}, {stream, stream_bytes}, {rest.data() + split, rest.size() - split}};
    }
};
Gathered gather(const pipe::StepMsg& no_stream, const float* stream, std::size_t floats) {
    Gathered g;
    g.rest = pipe::encode(no_stream);
    const std::uint64_t n = floats;
    std::memcpy(g.rest.data() + 16, &n, 8);  // little-endian host (the wire's order)
    g.split = 24U + 4U * no_stream.boundary.rows;
    g.stream = reinterpret_cast<const std::uint8_t*>(stream);
    g.stream_bytes = floats * 4U;
    return g;
}

}  // namespace

int main(int argc, char** argv) {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared card, restricts the process to the chosen one and prints the "device:" receipt line.
    strix::device::bind_cuda_main(argc, argv);
    if (argc < 2) usage("MODEL.gguf [options]");
    const std::string model = argv[1];
    CudaPrefillOptions o;
    o.slots = 0;
    o.moe_chunk_rows = 64;
    o.chunk_rows = 64;
    o.max_positions = 2048;
    o.decode_rows = 8;
    std::string stage, token_file, tiers, json_out;
    std::uint32_t cut = 22, slots = 8, sessions = 1;
    double connect_wait = 600.0;
    bool spin = true, boost = true;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage(("missing value of " + a).c_str());
            return argv[++i];
        };
        const auto u32 = [&] { return static_cast<std::uint32_t>(std::stoul(next())); };
        if (a == "--stage") stage = next();
        else if (a == "--cut") cut = u32();
        else if (a == "--slots") slots = u32();
        else if (a == "--token-file") token_file = next();
        else if (a == "--positions") o.max_positions = u32();
        else if (a == "--tiers") tiers = next();
        else if (a == "--expert-tp") o.expert_tp = true;
        else if (a == "--overlay") o.weights_overlay = next();
        else if (a == "--fuse-hc") o.decode_fuse_hc = true;
        else if (a == "--fuse-shared") o.decode_fuse_shared = true;
        else if (a == "--trellis-shexp") o.trellis_shexp = next();
        else if (a == "--connect-wait") connect_wait = std::stod(next());
        else if (a == "--sessions") sessions = u32();
        else if (a == "--json") json_out = next();
        else if (a == "--blocking") spin = false;
        else if (a == "--no-boost") boost = false;
        else if (a == "--eager") o.decode_graphs = false;
        else usage(("unknown option " + a).c_str());
    }
    if (stage.empty() || token_file.empty()) usage("--stage HOST:PORT and --token-file are required");
    if (slots == 0U || slots > 16U) usage("--slots in 1..16");
    std::vector<std::pair<std::string, std::uint16_t>> stages;
    {
        std::stringstream ss(stage);
        for (std::string item; std::getline(ss, item, ',');) {
            const auto colon = item.rfind(':');
            if (colon == std::string::npos) usage("--stage HOST:PORT[,HOST:PORT..]");
            stages.emplace_back(item.substr(0, colon), static_cast<std::uint16_t>(std::stoul(item.substr(colon + 1U))));
        }
    }
    o.first_layer = 0;
    o.end_layer = cut;
    if (!tiers.empty()) {
        o.tiers.assign(40, ExpertTier::file);
        for (std::size_t l = 0; l < tiers.size() && l < 40; ++l)
            o.tiers[l] = tiers[l] == 'D' ? ExpertTier::device : tiers[l] == 'H' ? ExpertTier::host : ExpertTier::file;
    }
    std::ofstream js;
    if (!json_out.empty()) js.open(json_out, std::ios::app);
    const auto emit = [&](const std::string& line) {
        std::printf("%s\n", line.c_str());
        std::fflush(stdout);
        if (js) {
            js << line << "\n";
            js.flush();
        }
    };
#ifdef _WIN32
    if (boost) {  // the thread that steps, sends and spins on the next request: high class, time critical, 1 ms timer, MMCSS
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
        timeBeginPeriod(1);
        DWORD task = 0;
        (void)AvSetMmThreadCharacteristicsW(L"Pro Audio", &task);
    }
#endif
    int failures = 0;
    try {
        const auto token = pipe::read_token_file(token_file);
        TensorFile header(model);
        const auto target = strix::deepseek41::validate_target(header.info());
        const auto cfg = header.config();
        const auto geo = pipe::boundary_geometry(cfg);
        const auto plan = pipe::plan_boundary(target.layers, cut);
        const auto t_load = Clock::now();
        CudaPrefill p(model, o);
        p.set_sequences(slots);
        emit("{\"event\":\"loaded\",\"describe\":" + p.describe() + ",\"slots\":" + std::to_string(slots) + ",\"cut\":" + std::to_string(cut) +
             ",\"load_s\":" + std::to_string(ms_between(t_load, Clock::now()) / 1e3) + "}");
        std::vector<float> mix(std::size_t{o.decode_rows} * 24U);
        for (std::uint32_t session = 0; sessions == 0U || session < sessions; ++session) {
            try {
                const auto& [host, port] = stages[std::min<std::size_t>(session, stages.size() - 1U)];
                Client c(host, port, connect_wait, spin);
                pipe::MultiHello mh;
                auto& h = mh.hello;
                h.token = token;
                h.cut = cut;
                h.layers = cfg.num_hidden_layers;
                h.hidden = cfg.hidden_size;
                h.hc = cfg.hc_mult;
                h.vocab = cfg.vocab_size;
                h.max_positions = o.max_positions;
                h.model_identity = target.manifest.layout_identity;
                h.backend = "cuda";
                h.numeric_identity = o.expert_tp ? "ds41-expert-tp2-i1152-v1" : "ds41-exact-v1";
                if (!o.trellis_shexp.empty()) h.numeric_identity += "+trellis-shexp";
                mh.slots = slots;
                mh.max_rows = o.decode_rows;
                c.send(pipe::MsgType::multi_hello, 1U, pipe::encode(mh));
                const auto ack = pipe::decode_hello_ack(c.expect(pipe::MsgType::hello_ack).payload);
                if (ack.status != 0U) throw std::runtime_error("TAIL refused the session: " + ack.message);
                emit("{\"event\":\"multi_session\",\"index\":" + std::to_string(session) + ",\"tail\":\"" + ack.engine + "\",\"world\":" +
                     std::to_string(ack.world) + "}");
                std::vector<double> step_ms, device_ms, send_ms, wait_ms;
                std::uint64_t steps = 0, rows = 0, imports = 0, resets = 0;
                auto t_idle = Clock::now();
                const auto t_session = Clock::now();
                for (;;) {
                    auto m = c.receive();
                    const auto t_got = Clock::now();
                    const auto type = m.header.type;
                    if (type == pipe::MsgType::close) {
                        c.send(pipe::MsgType::ack, m.header.request, {});
                        break;
                    }
                    try {
                        if (type == pipe::MsgType::seq_step) {
                            const auto q = pipe::decode_seq_step(m.payload, o.decode_rows);
                            if (q.slot >= slots) throw std::runtime_error("no slot " + std::to_string(q.slot));
                            p.select_sequence(q.slot);
                            if (q.commit != 0U) p.commit_window(q.commit, q.tokens[0]);
                            if (q.position != p.position())
                                throw std::runtime_error("slot " + std::to_string(q.slot) + " is at position " + std::to_string(p.position()) +
                                                         ", the step at " + std::to_string(q.position));
                            const auto t0 = Clock::now();
                            CudaStepIo io;
                            io.stream_view = true;
                            io.mix_out = mix.data();
                            io.stop_layer = cut;
                            io.sections_cut = cut;
                            const auto s = p.step(q.tokens, io);
                            const auto t1 = Clock::now();
                            const auto msg = boundary_without_stream(p, s, q.position, q.tokens, cut, plan, geo, mix);
                            const auto g = gather(msg, s.stream_view, q.tokens.size() * std::size_t{geo.hc} * geo.hidden);
                            c.send_parts(pipe::MsgType::seq_boundary, m.header.request, g.parts(), q.slot);
                            const auto t2 = Clock::now();
                            ++steps;
                            rows += q.tokens.size();
                            if (steps > 2U * slots) {  // past every slot's eager warm-up steps
                                step_ms.push_back(ms_between(t0, t1));
                                device_ms.push_back(s.device_ms);
                                send_ms.push_back(ms_between(t1, t2));
                                wait_ms.push_back(ms_between(t_idle, t_got));
                            }
                            t_idle = Clock::now();
                            continue;
                        }
                        if (type == pipe::MsgType::seq_reset) {
                            const auto slot = pipe::decode_u32(m.payload);
                            if (slot >= slots) throw std::runtime_error("no slot " + std::to_string(slot));
                            p.select_sequence(slot);
                            p.reset_sequence();
                            ++resets;
                            c.send(pipe::MsgType::ack, m.header.request, pipe::encode_u32(slot));
                            t_idle = Clock::now();
                            continue;
                        }
                        if (type == pipe::MsgType::seq_import) {
                            const auto [slot, sm] = pipe::decode_seq_import(m.payload);
                            if (slot >= slots) throw std::runtime_error("no slot " + std::to_string(slot));
                            p.select_sequence(slot);
                            p.import_state(strix::deepseek41::deserialize_attention_state(sm.state));
                            ++imports;
                            c.send(pipe::MsgType::ack, m.header.request, pipe::encode_u32(slot));
                            t_idle = Clock::now();
                            continue;
                        }
                        throw std::runtime_error(std::string("unexpected ") + pipe::to_string(type));
                    } catch (const std::exception& e) {
                        c.send(pipe::MsgType::error, m.header.request, pipe::encode_text(e.what()));
                        throw;
                    }
                }
                std::ostringstream os;
                os << "{\"event\":\"multi_summary\",\"index\":" << session << ",\"steps\":" << steps << ",\"rows\":" << rows << ",\"imports\":" << imports
                   << ",\"resets\":" << resets << ",\"seconds\":" << ms_between(t_session, Clock::now()) / 1e3 << ",\"bytes_sent\":" << c.bytes_sent()
                   << ",\"median_step_ms\":" << median(step_ms) << ",\"median_device_ms\":" << median(device_ms)
                   << ",\"median_send_ms\":" << median(send_ms) << ",\"median_wait_ms\":" << median(wait_ms) << "}";
                emit(os.str());
            } catch (const std::exception& e) {  // this session failed (TAIL error, lost connection): the next one
                ++failures;
                emit(std::string("{\"event\":\"multi_session_failed\",\"index\":") + std::to_string(session) + ",\"error\":\"" + e.what() + "\"}");
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-cuda-ds41-multi-front: %s\n", e.what());
        return 1;
    }
    return failures ? 1 : 0;
}
