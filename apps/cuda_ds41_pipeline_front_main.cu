// strix-cuda-ds41-pipeline-front: a strix/ds41-pipeline-v1 FRONT on the CUDA decode step. Layers [0, K) run here
// (CudaPrefill::step with the stream out), every step's boundary (stream, FFN mix, the cut's KV entries, top-k rows and
// candidate rows) goes to a pipeline-stage TAIL [K, 40) over TCP, and the TAIL's greedy token comes back.
//
//   strix-cuda-ds41-pipeline-front MODEL.gguf --import-state FILE --stage HOST --cuts K1,K2,.. [--port-base 29500]
//       --token-file FILE [--layers END] [--tiers DHF..] [--steps N] [--ref PAIR_LOG] [--blocking] [--no-nodelay]
//       [--connect-wait S] [--timeline-out CSV] [--json OUT]
//
// The object holds layers [0, END) (END >= every cut; experts resident per --tiers); for each cut K it imports the
// state, connects to HOST:(port-base + K), sends HELLO and IMPORT_STATE (the same state: the TAIL keeps its layers and
// the KV sources it reads), then decodes greedily --steps steps (step i feeds the token step i - 1 returned), compares
// with the reference log and closes. Per step: the CUDA step (wall and device), the boundary sections' copies, the
// message encode, the send, and the wait for the RESULT (the TAIL's receive + compute + reply + the wire).
//
// --fuse-hc / --fuse-shared: opt-in exact decode launch fusions (default off).
// The pair's P15 target identity: --expert-tp --overlay c-u6.gguf (ds41-expert-tp2-i1152-v1 on the c-u6 dense weights).
// mix-B (ds41-trellis-mixB-v1 on the pair): add --trellis-shexp PACK_DIR (the K4 pack; the shared expert of decode steps).
// Verify windows (DSpark pilot, no commit protocol yet): --windows FILE (lines "KEEP T0,T1,..,TR-1", R 1..8) after the
// --steps greedy steps; --passes fnv,time,..: each pass imports the state again on both sides, then per window: one
// CUDA step of the R rows, one STEP of R rows (fnv: kStepWantRowFnv, the TAIL returns every row's full-logits FNV),
// the RESULT (R tokens), then COMMIT KEEP to the TAIL and, on this side, a re-import + replay of the committed tokens
// (the CUDA decode has no partial commit yet; outside the window's timing). --window-ref LOG compares every row with the
// pair-only forced replay of the same windows (lines "win W ... row r ... fed T argmax A ... logits_fnv F").
// Pass kinds: time sends each STEP gathered straight from the step's mapped output buffer (the stream is never copied on
// the host: header, head + tokens, stream, sections in one gathered send); time0 copies the stream out of the step and
// encodes the payload before the send (the earlier path); fnv is time0 + per-row FNVs + a byte check of the gathered
// pieces against the encoded payload.
//
// Speculative decoding across the cut (ds41_pipeline_dspark.hpp; the TAIL drafts: strix-deepseek41-rank --stage-dspark):
//   --dspark-prompts FILE [--dspark-max-new N] [--dspark-eos a,b] [--dspark-row-fnv] [--dspark-timeout-ms T]
//   [--dspark-retries N] [--fault-drop-verify N] [--fault-dup-verify N] [--fault-stale-verify N] [--dspark-windows-out F]
// FILE: one prompt per line, "NAME ids,ids,.. [REF ids,..]" (REF: the expected greedy tokens). Per prompt: RESET on both
// sides (no prefill: CudaPrefill::reset_sequence), the prompt as fill STEPs of 8 rows (kStepFill: the TAIL commits every
// row and feeds its hidden to the drafter), SPEC_BEGIN, then per window the CUDA step of the TAIL's plan, a VERIFY STEP
// (request = epoch, window), the SPEC_ACCEPT (kept rows, published tokens, the next plan) and commit_window(kept, next)
// here; SPEC_COMMIT at the end. A decision that does not come within the timeout: the same VERIFY again (the TAIL answers
// it from its decision log). A TAIL without the drafter: one-row STEPs (fallback, same tokens). Faults (tests): every
// N-th VERIFY not sent the first time / sent twice / preceded by a copy from the previous epoch.
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
#include "strix/ds41_pipeline_dspark.hpp"
#include "strix/ds41_pipeline_wire.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
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
    std::fprintf(stderr, "strix-cuda-ds41-pipeline-front: %s\n", why);
    std::exit(2);
}
double ms_between(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
std::string hex(std::uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
    return b;
}
double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v.size() % 2U ? v[v.size() / 2U] : 0.5 * (v[v.size() / 2U - 1U] + v[v.size() / 2U]);
}

// ---- client socket (TCP_NODELAY; spinning non-blocking receive unless --blocking) ----
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
    Client(const std::string& host, std::uint16_t port, double wait_s, bool nodelay, bool spin) : spin_(spin) {
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
        int one = nodelay ? 1 : 0;
        setsockopt(s_, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof one);
        int buf = 4 << 20;
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

    // header + payload as one gathered send
    void send(pipe::MsgType type, std::uint64_t request, const std::vector<std::uint8_t>& payload, std::uint32_t flags = 0) {
        pipe::Header h;
        h.type = type;
        h.request = request;
        h.bytes = payload.size();
        h.flags = flags;
        const auto head = pipe::encode_header(h);
#ifdef _WIN32
        WSABUF b[2] = {{static_cast<ULONG>(head.size()), reinterpret_cast<CHAR*>(const_cast<std::uint8_t*>(head.data()))},
                       {static_cast<ULONG>(payload.size()), reinterpret_cast<CHAR*>(const_cast<std::uint8_t*>(payload.data()))}};
        DWORD sent = 0;
        if (WSASend(s_, b, 2, &sent, 0, nullptr, nullptr) != 0) {
            if (!would_block(last_error())) throw std::runtime_error("send failed");
            sent = 0;
        }
        std::size_t done = sent;
#else
        std::size_t done = 0;
#endif
        const std::size_t total = head.size() + payload.size();
        while (done < total) {
            const std::uint8_t* p = done < head.size() ? head.data() + done : payload.data() + (done - head.size());
            const std::size_t n = done < head.size() ? head.size() - done : total - done;
            const auto k = ::send(s_, reinterpret_cast<const char*>(p), static_cast<int>(n), 0);
            if (k > 0) {
                done += static_cast<std::size_t>(k);
                continue;
            }
            if (k < 0 && would_block(last_error())) continue;
            throw std::runtime_error("send failed");
        }
    }
    // header + several payload pieces as one gathered send (no staging copy of the pieces)
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
        std::size_t done = 0;  // bytes of `all` sent
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
        std::size_t at = 0, base = 0;  // piece index, bytes before it
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
    // true when a message can be read within `ms` milliseconds (its first byte is here)
    bool readable(int ms) {
        fd_set r;
        FD_ZERO(&r);
        FD_SET(s_, &r);
        timeval tv{ms / 1000, (ms % 1000) * 1000};
        const int n = select(static_cast<int>(s_ + 1), &r, nullptr, nullptr, &tv);
        if (n < 0) throw std::runtime_error("select failed");
        return n > 0;
    }
    // expect(), dropping SPEC_ACCEPTs that arrive late (answers to duplicated VERIFYs of a finished window)
    pipe::Message expect_after_spec(pipe::MsgType type) {
        for (;;) {
            auto m = receive();
            if (m.header.type == pipe::MsgType::spec_accept && type != pipe::MsgType::spec_accept) {
                ++late_accepts;
                continue;
            }
            if (m.header.type == pipe::MsgType::error) throw std::runtime_error("TAIL error: " + pipe::decode_text(m.payload));
            if (m.header.type != type) throw std::runtime_error(std::string("expected ") + pipe::to_string(type) + ", got " + pipe::to_string(m.header.type));
            return m;
        }
    }
    std::uint32_t late_accepts = 0;
    pipe::Message expect(pipe::MsgType type) {
        auto m = receive();
        if (m.header.type == pipe::MsgType::error) throw std::runtime_error("TAIL error: " + pipe::decode_text(m.payload));
        if (m.header.type != type) throw std::runtime_error(std::string("expected ") + pipe::to_string(type) + ", got " + pipe::to_string(m.header.type));
        return m;
    }

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
    bool spin_ = true;
};

struct RefStep {
    std::uint32_t fed = 0, argmax = 0;
    std::uint64_t fnv = 0;
};
std::map<std::uint32_t, RefStep> read_ref(const std::string& path) {
    std::ifstream in(path);
    if (!in) usage("cannot read the reference log");
    std::map<std::uint32_t, RefStep> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("pos ", 0) != 0) continue;
        std::istringstream s(line);
        std::string w;
        std::uint32_t pos = 0;
        RefStep r;
        while (s >> w) {
            if (w == "pos") s >> pos;
            else if (w == "fed") s >> r.fed;
            else if (w == "argmax") s >> r.argmax;
            else if (w == "logits_fnv") s >> std::hex >> r.fnv >> std::dec;
        }
        out.emplace(pos, r);
    }
    return out;
}

// One step's STEP message of a cut after layer K - 1 for R = fed.size() rows: tokens, stream, mix_ffn@K-1, the KV entries
// the step completed on the KV sources < K, the rows' top-k of the index sources < K, the rows' candidate blocks.
pipe::StepMsg build_step(CudaPrefill& p, const CudaStepResult& s, std::uint32_t pos, const std::vector<std::uint32_t>& fed, std::uint32_t K,
                         const pipe::BoundaryPlan& plan, const pipe::BoundaryGeometry& geo, const std::vector<float>& stream, const std::vector<float>& mix,
                         bool copy_sections, std::uint32_t flags, bool with_stream = true) {
    const auto R = static_cast<std::uint32_t>(fed.size());
    pipe::StepMsg msg;
    msg.flags = flags;
    auto& b = msg.boundary;
    b.position = pos;
    b.rows = R;
    b.tokens = fed;
    if (with_stream) b.stream.assign(stream.begin(), stream.begin() + static_cast<std::ptrdiff_t>(std::size_t{R} * geo.hc * geo.hidden));
    {
        pipe::Section mx{pipe::SectionKind::mix_ffn, K - 1U, 0U, R, {}};
        mx.bytes.resize(std::size_t{R} * geo.mix_floats() * 4U);
        std::memcpy(mx.bytes.data(), mix.data(), mx.bytes.size());
        b.sections.push_back(std::move(mx));
    }
    const auto find_kv = [&](std::uint32_t layer) -> const CudaStepSections::Kv* {
        for (const auto& e : s.sections.kv)
            if (e.layer == layer) return &e;
        return nullptr;
    };
    const auto find_topk = [&](std::uint32_t layer) -> const CudaStepSections::Topk* {
        for (const auto& e : s.sections.topk)
            if (e.layer == layer) return &e;
        return nullptr;
    };
    for (const auto& src : plan.kv_sources) {
        if (!copy_sections && find_kv(src.layer) != nullptr) {
            const auto* e = find_kv(src.layer);
            pipe::Section ks{pipe::SectionKind::kv_entries, src.layer, e->first, e->count, e->kv};
            ks.bytes.insert(ks.bytes.end(), e->idx_keys.begin(), e->idx_keys.end());
            b.sections.push_back(std::move(ks));
            continue;
        }
        const std::uint32_t first = pos / src.ratio, n = (pos + R) / src.ratio - first;
        pipe::Section ks{pipe::SectionKind::kv_entries, src.layer, first, n, {}};
        if (n != 0U) {
            std::vector<std::uint8_t> kv(std::size_t{n} * geo.kv_entry_bytes), ix(std::size_t{n} * geo.idx_entry_bytes);
            p.copy_kv_entries(src.layer, first, n, kv.data(), ix.data());
            ks.bytes = kv;
            ks.bytes.insert(ks.bytes.end(), ix.begin(), ix.end());
        }
        b.sections.push_back(std::move(ks));
    }
    for (const auto& src : plan.topk_sources) {
        std::vector<std::uint32_t> counts(R), ids(std::size_t{R} * geo.topk);
        if (!copy_sections && find_topk(src.layer) != nullptr) {
            const auto* e = find_topk(src.layer);
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
        if (!copy_sections && s.sections.candidates.size() >= rows.size()) {
            std::copy(s.sections.candidates.begin(), s.sections.candidates.begin() + static_cast<std::ptrdiff_t>(rows.size()), rows.begin());
        } else {
            p.copy_candidates(R, rows.data());
        }
        for (std::uint32_t r = 0; r < R; ++r)
            cs.bytes.insert(cs.bytes.end(), rows.begin() + static_cast<std::ptrdiff_t>(std::size_t{r} * full),
                            rows.begin() + static_cast<std::ptrdiff_t>(std::size_t{r} * full + stride));
        b.sections.push_back(std::move(cs));
    }
    return msg;
}

// A STEP payload without its stream bytes (build_step(..., with_stream = false)) -> the pieces of the same payload with the
// stream taken from `stream` (rows x hc x hidden FP32, e.g. the step's mapped output buffer): byte-identical to
// pipe::encode of the STEP with the stream (layout: flags, position, rows, sections, u64 stream floats, tokens, stream,
// sections).
struct GatherStep {
    std::vector<std::uint8_t> rest;  // encode() of the STEP without the stream; its u64 stream count patched
    std::size_t split = 0;           // the stream goes at this offset of `rest`
    const std::uint8_t* stream = nullptr;
    std::size_t stream_bytes = 0;
    [[nodiscard]] std::vector<std::pair<const std::uint8_t*, std::size_t>> parts() const {
        return {{rest.data(), split}, {stream, stream_bytes}, {rest.data() + split, rest.size() - split}};
    }
};
GatherStep gather_step(const pipe::StepMsg& no_stream, const float* stream, std::size_t floats) {
    GatherStep g;
    g.rest = pipe::encode(no_stream);
    const std::uint64_t n = floats;
    std::memcpy(g.rest.data() + 16, &n, 8);  // little-endian host (the wire's order)
    g.split = 24U + 4U * no_stream.boundary.rows;
    g.stream = reinterpret_cast<const std::uint8_t*>(stream);
    g.stream_bytes = floats * 4U;
    return g;
}

struct Window {
    std::uint32_t keep = 0;
    std::vector<std::uint32_t> tokens;
};
std::vector<Window> read_windows(const std::string& path) {
    std::ifstream in(path);
    if (!in) usage("cannot read the windows file");
    std::vector<Window> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        Window w;
        std::string toks;
        ls >> w.keep >> toks;
        std::stringstream ts(toks);
        for (std::string t; std::getline(ts, t, ',');) w.tokens.push_back(static_cast<std::uint32_t>(std::stoul(t)));
        if (w.tokens.empty() || w.tokens.size() > 8U || w.keep == 0U || w.keep > w.tokens.size()) usage("bad windows line");
        out.push_back(std::move(w));
    }
    return out;
}
struct WinRef {
    std::uint32_t pos = 0, fed = 0, argmax = 0;
    std::uint64_t fnv = 0;
};
// "win W R r keep k row r pos P fed T argmax A top X logits_fnv F" (the rank app's --forced-windows lines)
std::map<std::pair<std::uint32_t, std::uint32_t>, WinRef> read_window_ref(const std::string& path) {
    std::ifstream in(path);
    if (!in) usage("cannot read the window reference");
    std::map<std::pair<std::uint32_t, std::uint32_t>, WinRef> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("win ", 0) != 0) continue;
        std::istringstream s(line);
        std::string w;
        std::uint32_t win = 0, row = 0;
        WinRef r;
        while (s >> w) {
            if (w == "win") s >> win;
            else if (w == "row") s >> row;
            else if (w == "pos") s >> r.pos;
            else if (w == "fed") s >> r.fed;
            else if (w == "argmax") s >> r.argmax;
            else if (w == "logits_fnv") s >> std::hex >> r.fnv >> std::dec;
        }
        out.emplace(std::make_pair(win, row), r);
    }
    return out;
}

// --msg-log FILE: every message this FRONT sends to the TAIL (also in --dry-run), "DS4M" u32 version 1 then per message
// u32 type + u64 length + payload; strix-deepseek41-pipeline-replay --msg-log sends it to a TAIL (recorded boundaries).
std::ofstream g_msg_log;
void log_msg(pipe::MsgType type, const std::vector<std::uint8_t>& payload) {
    if (!g_msg_log.is_open()) return;
    const auto t = static_cast<std::uint32_t>(type);
    const std::uint64_t n = payload.size();
    g_msg_log.write(reinterpret_cast<const char*>(&t), 4);
    g_msg_log.write(reinterpret_cast<const char*>(&n), 8);
    g_msg_log.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(n));
    g_msg_log.flush();
}

template <class Emit>
int run_windows(CudaPrefill& p, Client* c, std::uint64_t& req, const strix::deepseek41::AttentionState& state, const std::vector<std::uint8_t>& state_bytes,
                const std::vector<Window>& windows, const std::map<std::pair<std::uint32_t, std::uint32_t>, WinRef>& wref,
                const std::vector<std::string>& passes, std::uint32_t K, const pipe::BoundaryPlan& plan, const pipe::BoundaryGeometry& geo,
                std::vector<float>& stream, std::vector<float>& mix, bool copy_sections, const Emit& emit) {
    int failures = 0;
    for (std::size_t pi = 0; pi < passes.size(); ++pi) {
        const bool fnv_pass = passes[pi] == "fnv";
        // time: the STEP sent gathered from the step's mapped output (no host copies of the stream); time0: the stream copied
        // out of the step, the payload encoded, then sent (the previous path); fnv: as time0 + a byte check of the gathered
        // pieces against the encoded payload
        const bool gathered = passes[pi] == "time";
        std::uint32_t gather_checked = 0, gather_differs = 0;
        // both sides back to the imported state
        p.import_state(state);
        {
            pipe::StateMsg sm;
            sm.manifest = "{\"layers\":[0,40],\"position\":" + std::to_string(state.position) + "}";
            sm.state = state_bytes;
            const auto sp = pipe::encode(sm);
            log_msg(pipe::MsgType::import_state, sp);
            if (c) {
                c->send(pipe::MsgType::import_state, req++, sp);
                (void)c->expect(pipe::MsgType::ack);
            }
        }
        std::vector<std::uint32_t> committed;  // tokens committed since the imported state
        std::map<std::uint32_t, std::vector<double>> tot, stp, dev, wai;
        std::uint32_t rows_cmp = 0, rows_ok = 0;
        for (std::size_t wi = 0; wi < windows.size(); ++wi) {
            const auto& w = windows[wi];
            const auto R = static_cast<std::uint32_t>(w.tokens.size());
            const std::uint32_t pos = p.position();
            const auto t0 = Clock::now();
            CudaStepIo io;
            io.stream_out = gathered ? nullptr : stream.data();
            io.stream_view = gathered || fnv_pass;
            io.mix_out = mix.data();
            io.stop_layer = K;
            if (!copy_sections) io.sections_cut = K;
            const auto s = p.step(w.tokens, io);
            const auto t1 = Clock::now();
            const auto msg = build_step(p, s, pos, w.tokens, K, plan, geo, stream, mix, copy_sections, fnv_pass ? pipe::kStepWantRowFnv : 0U, !gathered);
            const auto t2 = Clock::now();
            const std::size_t floats = std::size_t{R} * geo.hc * geo.hidden;
            std::vector<std::uint8_t> payload;
            GatherStep gs;
            if (gathered) gs = gather_step(msg, s.stream_view, floats);
            else payload = pipe::encode(msg);
            const auto t3 = Clock::now();
            if ((fnv_pass || !c) && !gathered)  // the STEP as the TAIL reads it (format check; not in the timing passes)
                pipe::validate_boundary(pipe::decode_step(payload, geo, 8, false).boundary, plan, geo);
            if (fnv_pass && s.stream_view) {  // the gathered pieces == the encoded payload, byte for byte
                pipe::StepMsg ns = msg;
                ns.boundary.stream.clear();
                const auto g = gather_step(ns, s.stream_view, floats);
                std::size_t at = 0;
                bool same = true;
                for (const auto& [ptr, n] : g.parts()) {
                    same = same && at + n <= payload.size() && std::memcmp(payload.data() + at, ptr, n) == 0;
                    at += n;
                }
                same = same && at == payload.size();
                ++gather_checked;
                gather_differs += same ? 0U : 1U;
            }
            const auto t3b = Clock::now();
            pipe::Result r;
            if (!gathered) log_msg(pipe::MsgType::step, payload);
            if (c) {
                if (gathered) c->send_parts(pipe::MsgType::step, req++, gs.parts(), 0);
                else c->send(pipe::MsgType::step, req++, payload, 0);
            }
            const auto t4 = Clock::now();
            if (c) r = pipe::decode_result(c->expect(pipe::MsgType::result).payload, geo.vocab, 8);
            else {  // --dry-run: no TAIL (tokens = the fed tokens shifted)
                r.tokens.assign(w.tokens.begin() + 1, w.tokens.end());
                r.tokens.push_back(0U);
                r.logits_fnv.assign(R, 0U);
            }
            const auto t5 = Clock::now();
            if (r.tokens.size() != R) throw std::runtime_error("RESULT rows differ from the window");
            // commit KEEP rows on the TAIL; here: re-import + replay of the committed tokens (no partial commit yet)
            {
                std::vector<std::uint8_t> kp(4);
                std::memcpy(kp.data(), &w.keep, 4);
                log_msg(pipe::MsgType::commit, kp);
                if (c) c->send(pipe::MsgType::commit, req++, kp);
            }
            if (c) {
                (void)c->expect(pipe::MsgType::ack);
            }
            const auto t6 = Clock::now();
            committed.insert(committed.end(), w.tokens.begin(), w.tokens.begin() + w.keep);
            p.import_state(state);
            for (std::size_t at = 0; at < committed.size(); at += 8U) {
                const std::vector<std::uint32_t> part(committed.begin() + static_cast<std::ptrdiff_t>(at),
                                                      committed.begin() + static_cast<std::ptrdiff_t>(std::min(committed.size(), at + 8U)));
                CudaStepIo ri;
                ri.stream_out = stream.data();
                ri.mix_out = mix.data();
                ri.stop_layer = K;
                (void)p.step(part, ri);
            }
            const auto t7 = Clock::now();
            std::ostringstream os;
            os << "{\"event\":\"window\",\"pass\":\"" << passes[pi] << "\",\"cut\":" << K << ",\"win\":" << wi << ",\"R\":" << R << ",\"keep\":" << w.keep
               << ",\"pos\":" << pos << ",\"payload_bytes\":" << (gathered ? gs.rest.size() + gs.stream_bytes : payload.size()) << ",\"gathered\":" << (gathered ? "true" : "false") << ",\"fed\":[";
            for (std::uint32_t i = 0; i < R; ++i) os << (i ? "," : "") << w.tokens[i];
            os << "],\"argmax\":[";
            for (std::uint32_t i = 0; i < R; ++i) os << (i ? "," : "") << r.tokens[i];
            os << "],\"logits_fnv\":[";
            for (std::uint32_t i = 0; i < R; ++i) os << (i ? "," : "") << "\"" << hex(r.logits_fnv[i]) << "\"";
            os << "]";
            std::uint32_t ok_rows = 0, cmp_rows = 0;
            for (std::uint32_t i = 0; i < R; ++i) {
                const auto it = wref.find({static_cast<std::uint32_t>(wi), i});
                if (it == wref.end()) continue;
                ++cmp_rows;
                const bool ok = it->second.pos == pos + i && it->second.fed == w.tokens[i] && it->second.argmax == r.tokens[i] &&
                                (!fnv_pass || it->second.fnv == r.logits_fnv[i]);
                ok_rows += ok ? 1U : 0U;
            }
            if (cmp_rows) {
                os << ",\"rows_compared\":" << cmp_rows << ",\"rows_matched\":" << ok_rows;
                rows_cmp += cmp_rows;
                rows_ok += ok_rows;
                if (ok_rows != cmp_rows) ++failures;
            }
            os << ",\"step_ms\":" << ms_between(t0, t1) << ",\"device_ms\":" << s.device_ms << ",\"sections_ms\":" << ms_between(t1, t2)
               << ",\"encode_ms\":" << ms_between(t2, t3) << ",\"check_ms\":" << ms_between(t3, t3b) << ",\"send_ms\":" << ms_between(t3b, t4)
               << ",\"wait_result_ms\":" << ms_between(t4, t5) << ",\"window_ms\":" << ms_between(t0, t5) - ms_between(t3, t3b) << ",\"commit_ms\":" << ms_between(t5, t6) << ",\"replay_ms\":" << ms_between(t6, t7)
               << ",\"graphs\":" << (s.graphs ? "true" : "false") << "}";
            emit(os.str());
            // timing: the first window of each R in a pass pays a graph capture
            if (s.graphs && s.graph_captures == 0U) {
                tot[R].push_back(ms_between(t0, t5) - ms_between(t3, t3b));
                stp[R].push_back(ms_between(t0, t1));
                dev[R].push_back(s.device_ms);
                wai[R].push_back(ms_between(t4, t5));
            }
        }
        for (const auto& [R, v] : tot) {
            std::ostringstream os;
            os << "{\"event\":\"window_summary\",\"pass\":\"" << passes[pi] << "\",\"cut\":" << K << ",\"R\":" << R << ",\"windows\":" << v.size()
               << ",\"median_window_ms\":" << median(v) << ",\"median_step_ms\":" << median(stp[R]) << ",\"median_device_ms\":" << median(dev[R])
               << ",\"median_wait_result_ms\":" << median(wai[R]) << "}";
            emit(os.str());
        }
        std::ostringstream os;
        os << "{\"event\":\"windows_done\",\"pass\":\"" << passes[pi] << "\",\"cut\":" << K << ",\"windows\":" << windows.size()
           << ",\"rows_compared\":" << rows_cmp << ",\"rows_matched\":" << rows_ok << ",\"gather_checked\":" << gather_checked
           << ",\"gather_differs\":" << gather_differs << "}";
        if (gather_differs) ++failures;
        emit(os.str());
    }
    return failures;
}

struct SpecPrompt {
    std::string name;
    std::vector<std::uint32_t> ids, ref;
};
std::vector<SpecPrompt> read_spec_prompts(const std::string& path) {
    std::ifstream in(path);
    if (!in) usage("cannot read --dspark-prompts");
    std::vector<SpecPrompt> out;
    std::string line;
    const auto list = [](const std::string& t) {
        std::vector<std::uint32_t> v;
        std::stringstream ss(t);
        for (std::string x; std::getline(ss, x, ',');)
            if (!x.empty()) v.push_back(static_cast<std::uint32_t>(std::stoul(x)));
        return v;
    };
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        SpecPrompt sp;
        std::string ids, tag, ref;
        ls >> sp.name >> ids >> tag >> ref;
        sp.ids = list(ids);
        if (tag == "REF") sp.ref = list(ref);
        if (sp.ids.empty()) usage("empty prompt in --dspark-prompts");
        out.push_back(std::move(sp));
    }
    return out;
}

struct SpecOptions {
    std::uint32_t max_new = 128, timeout_ms = 5000, retries = 4;
    std::vector<std::uint32_t> eos;
    bool row_fnv = false;
    std::uint32_t drop_verify = 0, dup_verify = 0, stale_verify = 0;
    std::string windows_out;
};

// One prompt through the split with the TAIL's drafter; returns the published tokens.
template <class Emit>
std::vector<std::uint32_t> run_spec_prompt(CudaPrefill& p, Client& c, std::uint64_t& req, std::uint32_t& epoch_counter, const SpecPrompt& sp,
                                           const SpecOptions& so, std::uint32_t K, const pipe::BoundaryPlan& plan, const pipe::BoundaryGeometry& geo,
                                           std::vector<float>& mix, const Emit& emit, std::FILE* wout, bool prefilled = false) {
    // ---- both sides at position 0 (prefilled: both sides already hold the prompt's state, the FRONT's next token is
    // the first generated one)
    if (!prefilled) {
        c.send(pipe::MsgType::reset, req++, {});
        (void)c.expect_after_spec(pipe::MsgType::ack);
        p.reset_sequence();
    }
    const std::uint32_t fnv_flag = so.row_fnv ? pipe::kStepWantRowFnv : 0U;
    // one CUDA step of `rows` and its STEP pieces (the stream gathered from the step's mapped output)
    struct Sent {
        GatherStep gs;
        std::uint32_t pos = 0;
        double step_ms = 0.0, device_ms = 0.0;
    };
    const auto make_step = [&](const std::vector<std::uint32_t>& rows, std::uint32_t flags) {
        Sent out;
        out.pos = p.position();
        const auto t0 = Clock::now();
        CudaStepIo io;
        io.stream_view = true;
        io.mix_out = mix.data();
        io.stop_layer = K;
        io.sections_cut = K;
        const auto s = p.step(rows, io);
        out.step_ms = ms_between(t0, Clock::now());
        out.device_ms = s.device_ms;
        static const std::vector<float> none;
        const auto msg = build_step(p, s, out.pos, rows, K, plan, geo, none, mix, false, flags, false);
        out.gs = gather_step(msg, s.stream_view, std::size_t{static_cast<std::uint32_t>(rows.size())} * geo.hc * geo.hidden);
        return out;
    };
    // ---- the prompt as fill windows of 8 rows
    const auto t_fill = Clock::now();
    std::uint32_t first = prefilled ? p.next_token() : 0U;
    for (std::size_t at = 0; !prefilled && at < sp.ids.size(); at += 8U) {
        const std::vector<std::uint32_t> rows(sp.ids.begin() + static_cast<std::ptrdiff_t>(at),
                                              sp.ids.begin() + static_cast<std::ptrdiff_t>(std::min(sp.ids.size(), at + 8U)));
        const auto st = make_step(rows, pipe::kStepFill | fnv_flag);
        c.send_parts(pipe::MsgType::step, req++, st.gs.parts(), 0);
        const auto r = pipe::decode_result(c.expect_after_spec(pipe::MsgType::result).payload, geo.vocab, 8);
        if (r.tokens.size() != rows.size()) throw std::runtime_error("fill RESULT rows differ");
        first = r.tokens.back();
        p.commit_window(static_cast<std::uint32_t>(rows.size()), first);
        if (wout) {
            std::fprintf(wout, "%zu ", rows.size());
            for (std::size_t i = 0; i < rows.size(); ++i) std::fprintf(wout, "%s%u", i ? "," : "", rows[i]);
            std::fprintf(wout, "\n");
        }
    }
    const double fill_ms = ms_between(t_fill, Clock::now());
    std::vector<std::uint32_t> out{first};
    const auto is_eos = [&](std::uint32_t t) { return std::find(so.eos.begin(), so.eos.end(), t) != so.eos.end(); };
    if (is_eos(first) || so.max_new <= 1U) return out;
    // ---- SPEC_BEGIN
    const std::uint32_t epoch = ++epoch_counter;
    const auto t_gen = Clock::now();
    double second_ms = -1.0;  // SPEC_BEGIN -> the first token published after `first`
    {
        pipe::SpecBegin b{epoch, p.position(), first, so.max_new, fnv_flag, so.eos};
        c.send(pipe::MsgType::spec_begin, req++, pipe::encode(b));
    }
    auto a0 = pipe::decode_spec_accept(c.expect(pipe::MsgType::spec_accept).payload);
    while (a0.status == pipe::SpecStatus::ok && (a0.epoch != epoch || a0.window != 0U))  // a late answer of the previous sequence
        a0 = pipe::decode_spec_accept(c.expect(pipe::MsgType::spec_accept).payload);
    std::uint32_t windows = 0, resends = 0, ignored = 0, rows_run = 0;
    std::map<std::uint32_t, std::pair<std::uint32_t, double>> by_r;
    std::vector<double> wait_ms;
    if (a0.status != pipe::SpecStatus::ok) {
        // ---- fallback: one-row STEPs of the same identity
        emit("{\"event\":\"dspark_fallback\",\"prompt\":\"" + sp.name + "\",\"status\":" + std::to_string(static_cast<std::uint32_t>(a0.status)) + "}");
        std::uint32_t next = first;
        while (out.size() < so.max_new) {
            const auto st = make_step({next}, fnv_flag);
            c.send_parts(pipe::MsgType::step, req++, st.gs.parts(), 0);
            const auto r = pipe::decode_result(c.expect_after_spec(pipe::MsgType::result).payload, geo.vocab, 8);
            next = r.tokens.at(0);
            p.commit_window(1U, next);
            out.push_back(next);
            if (second_ms < 0.0) second_ms = ms_between(t_gen, Clock::now());
            ++windows;
            if (is_eos(next)) break;
        }
    } else {
        pipe::SpecFrontState fs{epoch, 1U};
        std::vector<std::uint32_t> plan_rows = a0.plan;
        for (;;) {
            // the CUDA step of the plan, then its VERIFY (resent until its decision arrives)
            const auto tw = Clock::now();
            const auto st = make_step(plan_rows, pipe::kStepVerify | fnv_flag);
            const std::uint64_t rq = pipe::spec_request(epoch, fs.window);
            const auto send_verify = [&] { c.send_parts(pipe::MsgType::step, rq, st.gs.parts(), 0); };
            ++windows;
            const auto t_sent = Clock::now();
            if (so.stale_verify && windows % so.stale_verify == 0U) c.send_parts(pipe::MsgType::step, pipe::spec_request(epoch - 1U, fs.window), st.gs.parts(), 0);
            if (!(so.drop_verify && windows % so.drop_verify == 0U)) send_verify();
            if (so.dup_verify && windows % so.dup_verify == 0U) send_verify();
            pipe::SpecAccept a;
            std::uint32_t tries = 0;
            for (;;) {
                if (!c.readable(static_cast<int>(so.timeout_ms))) {
                    if (++tries > so.retries) throw std::runtime_error("no decision for window " + std::to_string(fs.window));
                    ++resends;
                    send_verify();
                    continue;
                }
                auto m = c.receive();
                if (m.header.type == pipe::MsgType::error) throw std::runtime_error("TAIL error: " + pipe::decode_text(m.payload));
                if (m.header.type != pipe::MsgType::spec_accept) throw std::runtime_error(std::string("expected SPEC_ACCEPT, got ") + pipe::to_string(m.header.type));
                a = pipe::decode_spec_accept(m.payload);
                if (a.status == pipe::SpecStatus::ok && fs.expected(a)) break;
                ++ignored;  // a stale answer or a duplicate of an earlier decision
            }
            const auto t_acc = Clock::now();
            if (a.rows != plan_rows.size() || a.position != st.pos || a.kept == 0U) throw std::runtime_error("decision does not match the window");
            p.commit_window(a.kept, a.next_token);
            if (p.position() != a.committed) throw std::runtime_error("committed positions differ");
            out.insert(out.end(), a.published.begin(), a.published.end());
            if (second_ms < 0.0 && !a.published.empty()) second_ms = ms_between(t_gen, t_acc);
            rows_run += a.rows;
            wait_ms.push_back(ms_between(t_sent, t_acc));
            auto& br = by_r[a.rows];
            ++br.first;
            br.second += ms_between(tw, t_acc);
            if (wout) {
                std::fprintf(wout, "%u ", a.kept);
                for (std::size_t i = 0; i < plan_rows.size(); ++i) std::fprintf(wout, "%s%u", i ? "," : "", plan_rows[i]);
                std::fprintf(wout, "\n");
            }
            std::ostringstream os;
            os << "{\"event\":\"dspark_window\",\"prompt\":\"" << sp.name << "\",\"w\":" << a.window << ",\"pos\":" << a.position << ",\"R\":" << a.rows
               << ",\"kept\":" << a.kept << ",\"rows\":[";
            for (std::size_t i = 0; i < plan_rows.size(); ++i) os << (i ? "," : "") << plan_rows[i];
            os << "],\"argmax\":[";
            for (std::size_t i = 0; i < a.argmax.size(); ++i) os << (i ? "," : "") << a.argmax[i];
            os << "],\"row_fnv\":[";
            for (std::size_t i = 0; i < a.row_fnv.size(); ++i) os << (i ? "," : "") << "\"" << hex(a.row_fnv[i]) << "\"";
            os << "],\"published\":" << a.published.size() << ",\"step_ms\":" << st.step_ms << ",\"device_ms\":" << st.device_ms
               << ",\"wait_ms\":" << ms_between(t_sent, t_acc) << ",\"window_ms\":" << ms_between(tw, t_acc) << ",\"tail_draft_ms\":" << a.draft_ms
               << ",\"tail_control_ms\":" << a.control_ms << "}";
            emit(os.str());
            if (a.done != pipe::SpecDone::more) break;
            plan_rows = a.plan;
            fs.window = a.window + 1U;
        }
        c.send(pipe::MsgType::spec_commit, req++, pipe::encode_spec_commit(epoch, fs.window));
        const auto ack = c.expect_after_spec(pipe::MsgType::ack);
        if (pipe::decode_u32(ack.payload) != p.position()) throw std::runtime_error("SPEC_COMMIT: the TAIL is at another position");
    }
    const double gen_s = ms_between(t_gen, Clock::now()) / 1e3;
    std::ostringstream os;
    os << "{\"event\":\"dspark_prompt\",\"prompt\":\"" << sp.name << "\",\"prompt_tokens\":" << sp.ids.size() << ",\"prefilled\":" << (prefilled ? "true" : "false")
       << ",\"second_token_ms\":" << second_ms << ",\"fill_ms\":" << fill_ms
       << ",\"generated\":" << out.size() << ",\"gen_s\":" << gen_s << ",\"tok_s\":" << (gen_s > 0 ? static_cast<double>(out.size() - 1U) / gen_s : 0.0)
       << ",\"windows\":" << windows << ",\"tokens_per_window\":" << (windows ? static_cast<double>(out.size() - 1U) / windows : 0.0)
       << ",\"rows\":" << rows_run << ",\"resends\":" << resends << ",\"ignored\":" << ignored << ",\"late_accepts\":" << c.late_accepts << ",\"median_wait_ms\":" << median(wait_ms) << ",\"by_R\":{";
    bool firstr = true;
    for (const auto& [R, v] : by_r) {
        os << (firstr ? "" : ",") << "\"" << R << "\":[" << v.first << "," << v.second / v.first << "]";
        firstr = false;
    }
    os << "}";
    if (!sp.ref.empty()) {
        const std::size_t n = std::min(out.size(), sp.ref.size());
        std::size_t same = 0;
        while (same < n && out[same] == sp.ref[same]) ++same;
        os << ",\"ref_compared\":" << n << ",\"ref_same_prefix\":" << same << ",\"ref_match\":" << (same == n ? "true" : "false");
    }
    os << ",\"tokens\":[";
    for (std::size_t i = 0; i < out.size(); ++i) os << (i ? "," : "") << out[i];
    os << "]}";
    emit(os.str());
    return out;
}

// ---------------------------------------------------------------------------------------------- end to end (--e2e-*)
// One process: the prefill on this object (layers [0, 40), experts [0, R) resident + the rest from the pinned host tier
// through the slots), the state handoff to the TAIL, then the split decode with the FRONT's layers [0, K) made resident
// (CudaPrefill::decode_residency) -- no second process, no reload.

// The TAIL's part of a state: layers [K, n) plus the caches of the KV sources < K that those layers read (the mirrors
// DeviceForward::import_state takes); every other layer absent. The bytes on the wire are what the TAIL imports.
strix::deepseek41::AttentionState tail_handoff_state(const strix::deepseek41::AttentionState& s, const pipe::BoundaryPlan& plan, std::uint32_t K) {
    auto out = strix::deepseek41::slice_attention_state(s, K, static_cast<std::uint32_t>(s.layers.size()));
    for (const auto& src : plan.kv_sources) {
        const auto& S = s.layers.at(src.layer);
        auto& L = out.layers[src.layer];
        L.owns_kv = S.owns_kv;
        L.ratio = S.ratio;
        L.kv = S.kv;
        L.idx_keys = S.idx_keys;
    }
    return out;
}

struct E2eOptions {
    std::uint32_t warm = 0;        // untimed prefills of each prompt before the timed one
    bool full_state = false;       // send the whole state (else the TAIL's part)
    bool release = true;           // decode_residency frees the run-only buffers
    std::string save_dir;          // the exported states (after the decode; outside the timing)
    int replay_session = -1;       // in that session: after each prompt, the split decode again from the saved state
    std::uint32_t selftest = 0;    // no TAIL: N steps per check
};

std::string hex64(std::uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(v));
    return b;
}

// One TAIL session: per prompt, warm prefills, then the timed prefill + handoff + generation (R1 or DSpark, the
// TAIL decides), then optionally the replay from the saved state through the import path.
template <class Emit>
int run_e2e_session(CudaPrefill& p, Client& c, std::uint64_t& req, std::uint32_t& epochs, const std::vector<SpecPrompt>& prompts,
                    const SpecOptions& so, const E2eOptions& eo, bool replay, std::uint32_t K, const pipe::BoundaryPlan& plan,
                    const pipe::BoundaryGeometry& geo, std::vector<float>& mix, const Emit& emit, std::size_t session) {
    int failures = 0;
    for (const auto& sp : prompts) {
        for (std::uint32_t w = 0; w < eo.warm; ++w) {
            const auto r = p.run(sp.ids);
            emit("{\"event\":\"e2e_warm\",\"session\":" + std::to_string(session) + ",\"prompt\":\"" + sp.name + "\",\"prefill_s\":" +
                 std::to_string(r.seconds) + ",\"next_token\":" + std::to_string(r.next_token) + "}");
        }
        const auto t0 = Clock::now();
        const auto r = p.run(sp.ids);
        const auto t1 = Clock::now();
        const auto st = p.export_state();
        const auto t2 = Clock::now();
        const auto hs = eo.full_state ? st : tail_handoff_state(st, plan, K);
        pipe::StateMsg sm;
        sm.manifest = "{\"layers\":[" + std::to_string(eo.full_state ? 0U : K) + ",40],\"position\":" + std::to_string(st.position) + "}";
        sm.state = strix::deepseek41::serialize_attention_state(hs);
        const auto payload = pipe::encode(sm);
        const auto t3 = Clock::now();
        // the state goes out on a thread while this one makes the FRONT's layers resident for the decode
        const std::uint64_t req_import = req++;
        double send_ms = 0.0;
        std::exception_ptr send_error;
        std::thread sender([&] {
            try {
                const auto a = Clock::now();
                c.send(pipe::MsgType::import_state, req_import, payload);
                send_ms = ms_between(a, Clock::now());
            } catch (...) {
                send_error = std::current_exception();
            }
        });
        double residency_ms = 0.0;
        std::exception_ptr res_error;
        try {
            const auto a = Clock::now();
            (void)p.decode_residency(K, eo.release);
            residency_ms = ms_between(a, Clock::now());
        } catch (...) {
            res_error = std::current_exception();
        }
        sender.join();
        if (send_error) std::rethrow_exception(send_error);
        if (res_error) std::rethrow_exception(res_error);
        (void)c.expect_after_spec(pipe::MsgType::ack);
        const auto t5 = Clock::now();
        std::ostringstream ho;
        ho << "{\"event\":\"e2e_handoff\",\"session\":" << session << ",\"prompt\":\"" << sp.name << "\",\"tokens\":" << sp.ids.size()
           << ",\"prefill_s\":" << ms_between(t0, t1) / 1e3 << ",\"run_s\":" << r.seconds << ",\"next_token\":" << r.next_token
           << ",\"engram_s\":" << r.engram_seconds << ",\"expert_io_s\":" << r.expert_io_seconds << ",\"expert_gb_streamed\":" << double(r.expert_bytes_streamed) / 1e9
           << ",\"export_ms\":" << ms_between(t1, t2) << ",\"encode_ms\":" << ms_between(t2, t3) << ",\"state_mb\":" << double(st.bytes()) / 1e6
           << ",\"sent_mb\":" << double(payload.size()) / 1e6 << ",\"handoff\":\"" << (eo.full_state ? "full" : "tail") << "\",\"send_ms\":" << send_ms
           << ",\"residency_ms\":" << residency_ms << ",\"import_ack_ms\":" << ms_between(t3, t5) << ",\"handoff_s\":" << ms_between(t1, t5) / 1e3
           << ",\"ready_s\":" << ms_between(t0, t5) / 1e3 << ",\"digest\":\"" << hex64(st.digest(true)) << "\",\"residency\":" << p.decode_residency_report() << "}";
        emit(ho.str());
        const auto out = run_spec_prompt(p, c, req, epochs, sp, so, K, plan, geo, mix, emit, nullptr, true);
        if (!eo.save_dir.empty()) {
            const auto path = std::filesystem::path(eo.save_dir) / (sp.name + "-s" + std::to_string(session) + ".state");
            strix::deepseek41::save_attention_state(path, st);
        }
        if (replay) {  // the same split decode from the exported state through the import path (both sides import it)
            p.import_state(st);
            c.send(pipe::MsgType::import_state, req++, payload);
            (void)c.expect_after_spec(pipe::MsgType::ack);
            SpecPrompt rp = sp;
            rp.name = sp.name + "#replay";
            const auto out2 = run_spec_prompt(p, c, req, epochs, rp, so, K, plan, geo, mix, emit, nullptr, true);
            const std::size_t n = std::min(out.size(), out2.size());
            std::size_t same = 0;
            while (same < n && out[same] == out2[same]) ++same;
            const bool ok = same == n && out.size() == out2.size();
            if (!ok) ++failures;
            emit("{\"event\":\"e2e_replay\",\"session\":" + std::to_string(session) + ",\"prompt\":\"" + sp.name + "\",\"tokens\":" + std::to_string(out.size()) +
                 ",\"replay_tokens\":" + std::to_string(out2.size()) + ",\"same_prefix\":" + std::to_string(same) + ",\"match\":" + (ok ? "true" : "false") + "}");
        }
    }
    return failures;
}

// No TAIL (--dry-run --e2e-selftest N): per prompt, run -> export -> the TAIL slice (round trip + the TAIL's import
// needs) -> decode_residency(K), then
//   A: N greedy steps of the whole model (layers >= K read sparsely, eager) after the run (vs --ref when given);
//   B: the same N rows after import_state(exported) (== A, tokens + logits FNV);
//   C / D: the FRONT range [0, K) only (stream + mix out) after import_state, graphs vs eager (== stream FNV).
// A prompt name seen before must give the same state digest and tokens (the run after a decode_residency undoes it).
template <class Emit>
int run_e2e_selftest(CudaPrefill& p, const std::vector<SpecPrompt>& prompts, const E2eOptions& eo, std::uint32_t K, const pipe::BoundaryPlan& plan,
                     const std::map<std::uint32_t, RefStep>& ref, std::uint32_t hidden, const Emit& emit) {
    int failures = 0;
    std::map<std::string, std::pair<std::uint64_t, std::vector<std::uint32_t>>> seen;
    const std::uint32_t N = eo.selftest;
    for (const auto& sp : prompts) {
        const auto t0 = Clock::now();
        const auto r = p.run(sp.ids);
        const auto t1 = Clock::now();
        const auto st = p.export_state();
        const auto hs = tail_handoff_state(st, plan, K);
        const auto bytes = strix::deepseek41::serialize_attention_state(hs);
        const auto back = strix::deepseek41::deserialize_attention_state(bytes);
        bool slice_ok = back.digest() == hs.digest() && back.position == st.position && back.next_token == st.next_token && back.tokens == st.tokens;
        for (std::size_t l = K; l < back.layers.size(); ++l) slice_ok = slice_ok && strix::deepseek41::attention_layer_present(back.layers[l]);
        for (const auto& src : plan.kv_sources) {
            const auto& L = back.layers[src.layer];
            slice_ok = slice_ok && L.owns_kv && L.kv == st.layers[src.layer].kv && L.idx_keys == st.layers[src.layer].idx_keys;
        }
        if (!slice_ok) ++failures;
        const auto full_bytes = strix::deepseek41::serialize_attention_state(st).size();
        const auto t2 = Clock::now();
        (void)p.decode_residency(K, eo.release);
        const auto t3 = Clock::now();
        emit("{\"event\":\"selftest_prefill\",\"prompt\":\"" + sp.name + "\",\"tokens\":" + std::to_string(sp.ids.size()) + ",\"prefill_s\":" +
             std::to_string(ms_between(t0, t1) / 1e3) + ",\"next_token\":" + std::to_string(r.next_token) + ",\"digest\":\"" + hex64(st.digest(true)) +
             "\",\"slice_mb\":" + std::to_string(double(bytes.size()) / 1e6) + ",\"full_mb\":" + std::to_string(double(full_bytes) / 1e6) +
             ",\"slice_ok\":" + (slice_ok ? "true" : "false") + ",\"residency_ms\":" + std::to_string(ms_between(t2, t3)) + ",\"residency\":" +
             p.decode_residency_report() + "}");
        // A: whole-model greedy steps after the run
        std::vector<std::uint32_t> fed, arg;
        std::vector<std::uint64_t> fnv;
        std::uint32_t next = r.next_token, ref_ok = 0, ref_cmp = 0;
        for (std::uint32_t k = 0; k < N; ++k) {
            const std::uint32_t pos = p.position();
            const auto s = p.step(std::vector<std::uint32_t>{next});
            fed.push_back(next);
            arg.push_back(s.tokens.at(0));
            fnv.push_back(s.logits_fnv.at(0));
            const auto it = ref.find(pos);
            if (it != ref.end()) {
                ++ref_cmp;
                if (it->second.fed == next && it->second.argmax == s.tokens[0] && it->second.fnv == s.logits_fnv[0]) ++ref_ok;
            }
            next = s.tokens[0];
        }
        if (ref_cmp && ref_ok != ref_cmp) ++failures;
        // B: the same rows after import_state(exported)
        p.import_state(st);
        std::uint32_t b_ok = 0;
        for (std::uint32_t k = 0; k < N; ++k) {
            const auto s = p.step(std::vector<std::uint32_t>{fed[k]});
            b_ok += (s.tokens.at(0) == arg[k] && s.logits_fnv.at(0) == fnv[k]) ? 1U : 0U;
        }
        if (b_ok != N) ++failures;
        // C / D: the FRONT range only, graphs then eager
        std::vector<float> so_(std::size_t{4} * hidden), mo(24);
        std::vector<std::uint64_t> cg, ce;
        std::uint32_t graph_steps = 0;
        for (int pass = 0; pass < 2; ++pass) {
            p.import_state(st);
            p.set_decode_graphs(pass == 0);
            for (std::uint32_t k = 0; k < N; ++k) {
                CudaStepIo io;
                io.stop_layer = K;
                io.stream_out = so_.data();
                io.mix_out = mo.data();
                const auto s = p.step(std::vector<std::uint32_t>{fed[k]}, io);
                if (pass == 0 && s.graphs) ++graph_steps;
                (pass == 0 ? cg : ce).push_back(fnv1a(so_.data(), so_.size()) ^ (fnv1a(mo.data(), mo.size()) * 1099511628211ULL));
            }
        }
        p.set_decode_graphs(true);
        const bool cd_ok = cg == ce;
        if (!cd_ok) ++failures;
        bool rerun_ok = true;
        const auto it = seen.find(sp.name);
        if (it != seen.end()) rerun_ok = it->second.first == st.digest(true) && it->second.second == arg;
        else seen[sp.name] = {st.digest(true), arg};
        if (!rerun_ok) ++failures;
        std::ostringstream os;
        os << "{\"event\":\"selftest\",\"prompt\":\"" << sp.name << "\",\"steps\":" << N << ",\"ref_compared\":" << ref_cmp << ",\"ref_matched\":" << ref_ok
           << ",\"import_matched\":" << b_ok << ",\"front_graph_steps\":" << graph_steps << ",\"front_graphs_eq_eager\":" << (cd_ok ? "true" : "false")
           << ",\"rerun_same\":" << (rerun_ok ? "true" : "false") << ",\"slice_ok\":" << (slice_ok ? "true" : "false") << ",\"tokens\":[";
        for (std::size_t i = 0; i < arg.size(); ++i) os << (i ? "," : "") << arg[i];
        os << "]}";
        emit(os.str());
    }
    return failures;
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
    o.max_positions = 2600;
    o.decode_rows = 1;
    std::string state_file, host, cuts_arg, token_file, ref_file, tiers, timeline_out, json_out, windows_file, window_ref_file, passes_arg = "fnv,time";
    std::uint32_t steps = 64, port_base = 29500;
    double connect_wait = 600.0;
    bool nodelay = true, spin = true, gather = true, copy_sections = false, boost = true, dry_run = false, ref_feed = false;
    std::string msg_log;
    std::string dspark_prompts, e2e_prompts;
    E2eOptions eo;
    SpecOptions so;
    std::vector<std::uint32_t> stage_ports;  // --stage-ports: the i-th session's port (else port-base + K)
    int fault_session = -1, eos_session = -1;  // the FRONT faults / the EOS list only in that session (-1: all)
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage(("missing value of " + a).c_str());
            return argv[++i];
        };
        if (a == "--import-state") state_file = next();
        else if (a == "--stage") host = next();
        else if (a == "--cuts") cuts_arg = next();
        else if (a == "--port-base") port_base = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--token-file") token_file = next();
        else if (a == "--layers") o.end_layer = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--tiers") tiers = next();
        else if (a == "--steps") steps = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--ref") ref_file = next();
        else if (a == "--blocking") spin = false;
        else if (a == "--no-nodelay") nodelay = false;
        else if (a == "--eager") o.decode_graphs = false;
        else if (a == "--no-gather") gather = false;
        else if (a == "--copy-sections") copy_sections = true;
        else if (a == "--no-boost") boost = false;  // no priority class / time-critical / MMCSS for the stepping thread  // synchronous copies after the step (before the in-graph copies)  // tokens only are compared (a TAIL without gathered logits)
        else if (a == "--connect-wait") connect_wait = std::stod(next());
        else if (a == "--positions") o.max_positions = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--timeline-out") timeline_out = next();
        else if (a == "--json") json_out = next();
        else if (a == "--fuse-hc") o.decode_fuse_hc = true;
        else if (a == "--fuse-shared") o.decode_fuse_shared = true;
        else if (a == "--expert-tp") o.expert_tp = true;
        else if (a == "--overlay") o.weights_overlay = next();
        else if (a == "--trellis-shexp") o.trellis_shexp = next();
        else if (a == "--windows") windows_file = next();
        else if (a == "--window-ref") window_ref_file = next();
        else if (a == "--passes") passes_arg = next();
        else if (a == "--dry-run") dry_run = true;
        else if (a == "--dspark-prompts") dspark_prompts = next();
        else if (a == "--stage-ports") {
            std::stringstream ps(next());
            for (std::string t; std::getline(ps, t, ',');) stage_ports.push_back(static_cast<std::uint32_t>(std::stoul(t)));
        }
        else if (a == "--fault-session") fault_session = std::stoi(next());
        else if (a == "--eos-session") eos_session = std::stoi(next());
        else if (a == "--dspark-max-new") so.max_new = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--dspark-eos") {
            std::stringstream es(next());
            for (std::string t; std::getline(es, t, ',');) so.eos.push_back(static_cast<std::uint32_t>(std::stoul(t)));
        }
        else if (a == "--dspark-row-fnv") so.row_fnv = true;
        else if (a == "--dspark-timeout-ms") so.timeout_ms = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--dspark-retries") so.retries = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--dspark-windows-out") so.windows_out = next();
        else if (a == "--fault-drop-verify") so.drop_verify = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--fault-dup-verify") so.dup_verify = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--fault-stale-verify") so.stale_verify = static_cast<std::uint32_t>(std::stoul(next()));  // no TAIL: the windows' CUDA steps, STEP messages (format-checked) and replays only
        else if (a == "--ref-feed") ref_feed = true;  // R1 steps feed the --ref log's fed tokens (teacher forcing; --dry-run: no TAIL)
        else if (a == "--msg-log") msg_log = next();  // no TAIL: the windows' CUDA steps, STEP messages (format-checked) and replays only
        // end to end (one process: prefill + handoff + split decode)
        else if (a == "--e2e-prompts") e2e_prompts = next();  // "NAME ids,.." per line: prefill here, handoff, SPEC_BEGIN (R1 when the TAIL has no drafter)
        else if (a == "--e2e-warm") eo.warm = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--e2e-full-state") eo.full_state = true;
        else if (a == "--e2e-keep-buffers") eo.release = false;
        else if (a == "--e2e-save-dir") eo.save_dir = next();
        else if (a == "--e2e-replay-session") eo.replay_session = std::stoi(next());
        else if (a == "--e2e-selftest") eo.selftest = static_cast<std::uint32_t>(std::stoul(next()));  // with --dry-run: no TAIL
        else if (a == "--fast-dense") o.fast_dense = true;
        else if (a == "--fast-moe") o.fast_moe = true;
        else if (a == "--fast-attention") o.fast_attention = true;
        else if (a == "--no-fast-index") o.fast_index = false;
        else if (a == "--chunk-rows") o.chunk_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--moe-chunk-rows") o.moe_chunk_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--slots") o.slots = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--host-commit-floor-gib") o.host_commit_floor_gib = std::stod(next());
        else if (a == "--host-pin-wait-s") o.host_pin_wait_s = std::stod(next());
        else usage(("unknown option " + a).c_str());
    }
    const bool e2e_dry = !e2e_prompts.empty() && dry_run;
    if (!e2e_dry && ((state_file.empty() && dspark_prompts.empty() && e2e_prompts.empty()) || host.empty() || cuts_arg.empty() || token_file.empty()))
        usage("--import-state (or --dspark-prompts / --e2e-prompts), --stage, --cuts and --token-file are required");
    if (e2e_dry && cuts_arg.empty()) usage("--e2e-prompts --dry-run needs --cuts K");
    if (!dspark_prompts.empty() || !e2e_prompts.empty()) o.decode_rows = 8;
    std::vector<std::uint32_t> cuts;
    {
        std::stringstream ss(cuts_arg);
        std::string item;
        while (std::getline(ss, item, ',')) cuts.push_back(static_cast<std::uint32_t>(std::stoul(item)));
    }
    std::uint32_t end = 0;
    for (const auto k : cuts) end = std::max(end, k);
    if (!e2e_prompts.empty()) o.end_layer = 40U;  // the prefill runs every layer; the decode steps stop at the cut
    else if (o.end_layer == 40U || o.end_layer < end) o.end_layer = end;
    if (!tiers.empty()) {
        o.tiers.assign(40, ExpertTier::file);
        for (std::size_t l = 0; l < tiers.size() && l < 40; ++l)
            o.tiers[l] = tiers[l] == 'D' ? ExpertTier::device : tiers[l] == 'H' ? ExpertTier::host : ExpertTier::file;
    }
    std::map<std::uint32_t, RefStep> ref;
    if (!ref_file.empty()) ref = read_ref(ref_file);
    std::vector<Window> windows;
    if (!windows_file.empty()) {
        windows = read_windows(windows_file);
        o.decode_rows = 8;
    }
    std::map<std::pair<std::uint32_t, std::uint32_t>, WinRef> wref;
    if (!window_ref_file.empty()) wref = read_window_ref(window_ref_file);
    std::vector<std::string> passes;
    {
        std::stringstream ss(passes_arg);
        for (std::string item; std::getline(ss, item, ',');) passes.push_back(item);
    }
    std::ofstream js, tl;
    if (!json_out.empty()) js.open(json_out, std::ios::app);
    if (!timeline_out.empty()) {
        tl.open(timeline_out, std::ios::app);
        tl << "cut,pos,fed,token,match,step_ms,step_device_ms,sections_ms,encode_ms,send_ms,wait_result_ms,total_ms\n";
    }
    const auto emit = [&](const std::string& line) {
        std::printf("%s\n", line.c_str());
        std::fflush(stdout);
        if (js) js << line << "\n";
    };
    int failures = 0;
    if (ref_feed && ref.empty()) usage("--ref-feed needs --ref");
    if (!msg_log.empty()) {
        g_msg_log.open(msg_log, std::ios::binary | std::ios::trunc);
        if (!g_msg_log) usage("cannot write the message log");
        const std::uint32_t head[2] = {0x4D345344U, 1U};  // "DS4M", version
        g_msg_log.write(reinterpret_cast<const char*>(head), 8);
    }
#ifdef _WIN32
    if (boost) {  // the thread that steps, sends and spins on the RESULT: high class, time critical, 1 ms timer, MMCSS
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
        timeBeginPeriod(1);
        DWORD task = 0;
        (void)AvSetMmThreadCharacteristicsW(L"Pro Audio", &task);
    }
#endif
    try {
        if (!e2e_prompts.empty()) {
            const auto prompts = read_spec_prompts(e2e_prompts);
            TensorFile header(model);
            const auto target = strix::deepseek41::validate_target(header.info());
            const auto cfg = header.config();
            const auto geo = pipe::boundary_geometry(cfg);
            std::size_t longest = 0;
            for (const auto& sp : prompts) longest = std::max(longest, sp.ids.size());
            o.max_positions = std::max<std::uint32_t>(o.max_positions, static_cast<std::uint32_t>(longest + so.max_new + 16U));
            if (!eo.save_dir.empty()) std::filesystem::create_directories(eo.save_dir);
            const auto t_load = Clock::now();
            CudaPrefill p(model, o);
            emit("{\"event\":\"loaded\",\"describe\":" + p.describe() + ",\"load_s\":" + std::to_string(ms_between(t_load, Clock::now()) / 1e3) + "}");
            if (dry_run) {
                const auto K = cuts.at(0);
                const auto plan = pipe::plan_boundary(target.layers, K);
                failures += run_e2e_selftest(p, prompts, eo, K, plan, ref, cfg.hidden_size, emit);
                emit("{\"event\":\"e2e_selftest_done\",\"cut\":" + std::to_string(K) + ",\"failures\":" + std::to_string(failures) + "}");
                return failures ? 1 : 0;
            }
            const auto token = pipe::read_token_file(token_file);
            std::uint32_t epochs = static_cast<std::uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count() % 100000U) * 16U + 1U;
            std::vector<float> mix(std::size_t{8} * 24U);
            for (std::size_t ci = 0; ci < cuts.size(); ++ci) {
                const auto K = cuts[ci];
                const auto plan = pipe::plan_boundary(target.layers, K);
                const std::uint32_t port = ci < stage_ports.size() ? stage_ports[ci] : port_base + K;
                emit("{\"event\":\"e2e_session\",\"index\":" + std::to_string(ci) + ",\"cut\":" + std::to_string(K) + ",\"port\":" + std::to_string(port) + "}");
              try {
                Client c(host, static_cast<std::uint16_t>(port), connect_wait, nodelay, spin);
                std::uint64_t req = 1;
                pipe::Hello h;
                h.token = token;
                h.cut = K;
                h.layers = cfg.num_hidden_layers;
                h.hidden = cfg.hidden_size;
                h.hc = cfg.hc_mult;
                h.vocab = cfg.vocab_size;
                h.max_positions = o.max_positions;
                h.model_identity = target.manifest.layout_identity;
                h.backend = "cuda";
                h.numeric_identity = o.expert_tp ? "ds41-expert-tp2-i1152-v1" : "ds41-exact-v1";
                c.send(pipe::MsgType::hello, req++, pipe::encode(h));
                const auto ack = pipe::decode_hello_ack(c.expect(pipe::MsgType::hello_ack).payload);
                if (ack.status != 0U) throw std::runtime_error("TAIL refused the session: " + ack.message);
                failures += run_e2e_session(p, c, req, epochs, prompts, so, eo, eo.replay_session == static_cast<int>(ci), K, plan, geo, mix, emit, ci);
                c.send(pipe::MsgType::close, req++, {});
                try {
                    (void)c.expect(pipe::MsgType::ack);
                } catch (const std::exception&) {
                }
                emit("{\"event\":\"e2e_session_done\",\"index\":" + std::to_string(ci) + ",\"failures\":" + std::to_string(failures) + "}");
              } catch (const std::exception& e) {
                ++failures;
                emit(std::string("{\"event\":\"e2e_session_failed\",\"index\":") + std::to_string(ci) + ",\"error\":\"" + e.what() + "\"}");
              }
            }
            return failures ? 1 : 0;
        }
        if (!dspark_prompts.empty()) {
            const auto prompts = read_spec_prompts(dspark_prompts);
            const auto token = pipe::read_token_file(token_file);
            TensorFile header(model);
            const auto target = strix::deepseek41::validate_target(header.info());
            const auto cfg = header.config();
            const auto geo = pipe::boundary_geometry(cfg);
            std::size_t longest = 0;
            for (const auto& sp : prompts) longest = std::max(longest, sp.ids.size());
            o.max_positions = std::max<std::uint32_t>(o.max_positions, static_cast<std::uint32_t>(longest + so.max_new + 16U));
            const auto t_load = Clock::now();
            CudaPrefill p(model, o);
            emit("{\"event\":\"loaded\",\"describe\":" + p.describe() + ",\"load_s\":" + std::to_string(ms_between(t_load, Clock::now()) / 1e3) + "}");
            std::FILE* wout = so.windows_out.empty() ? nullptr : std::fopen(so.windows_out.c_str(), "w");
            for (std::size_t ci = 0; ci < cuts.size(); ++ci) {
                const auto K = cuts[ci];
                const auto plan = pipe::plan_boundary(target.layers, K);
                const std::uint32_t port = ci < stage_ports.size() ? stage_ports[ci] : port_base + K;
                SpecOptions sso = so;
                if (fault_session >= 0 && static_cast<std::size_t>(fault_session) != ci) sso.drop_verify = sso.dup_verify = sso.stale_verify = 0;
                if (eos_session >= 0 && static_cast<std::size_t>(eos_session) != ci) sso.eos.clear();
                emit("{\"event\":\"dspark_session\",\"index\":" + std::to_string(ci) + ",\"cut\":" + std::to_string(K) + ",\"port\":" + std::to_string(port) +
                     ",\"faults\":[" + std::to_string(sso.drop_verify) + "," + std::to_string(sso.dup_verify) + "," + std::to_string(sso.stale_verify) +
                     "],\"eos\":" + std::to_string(sso.eos.size()) + "}");
              try {
                Client c(host, static_cast<std::uint16_t>(port), connect_wait, nodelay, spin);
                std::uint64_t req = 1;
                pipe::Hello h;
                h.token = token;
                h.cut = K;
                h.layers = cfg.num_hidden_layers;
                h.hidden = cfg.hidden_size;
                h.hc = cfg.hc_mult;
                h.vocab = cfg.vocab_size;
                h.max_positions = o.max_positions;
                h.model_identity = target.manifest.layout_identity;
                h.backend = "cuda";
                h.numeric_identity = o.expert_tp ? "ds41-expert-tp2-i1152-v1" : "ds41-exact-v1";
                c.send(pipe::MsgType::hello, req++, pipe::encode(h));
                const auto ack = pipe::decode_hello_ack(c.expect(pipe::MsgType::hello_ack).payload);
                if (ack.status != 0U) throw std::runtime_error("TAIL refused the session: " + ack.message);
                std::vector<float> mix(std::size_t{8} * 24U);
                std::uint32_t epochs = static_cast<std::uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count() % 100000U) * 16U + 1U;
                std::uint32_t ref_ok = 0, ref_cmp = 0;
                for (const auto& sp : prompts) {
                    if (wout) std::fprintf(wout, "# %s\n", sp.name.c_str());
                    const auto out = run_spec_prompt(p, c, req, epochs, sp, sso, K, plan, geo, mix, emit, wout);
                    if (!sp.ref.empty()) {
                        const std::size_t n = std::min(out.size(), sp.ref.size());
                        ++ref_cmp;
                        if (std::equal(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(n), sp.ref.begin())) ++ref_ok;
                        else ++failures;
                    }
                }
                c.send(pipe::MsgType::close, req++, {});
                try {
                    (void)c.expect(pipe::MsgType::ack);
                } catch (const std::exception&) {
                }
                emit("{\"event\":\"dspark_done\",\"cut\":" + std::to_string(K) + ",\"prompts\":" + std::to_string(prompts.size()) + ",\"ref_compared\":" +
                     std::to_string(ref_cmp) + ",\"ref_matched\":" + std::to_string(ref_ok) + "}");
              } catch (const std::exception& e) {  // this session failed (TAIL error, lost connection): the next session
                ++failures;
                emit(std::string("{\"event\":\"dspark_session_failed\",\"index\":") + std::to_string(ci) + ",\"error\":\"" + e.what() + "\"}");
              }
            }
            if (wout) std::fclose(wout);
            return failures ? 1 : 0;
        }
        const auto state = strix::deepseek41::load_attention_state(state_file);
        const auto token = pipe::read_token_file(token_file);
        TensorFile header(model);
        const auto target = strix::deepseek41::validate_target(header.info());
        const auto cfg = header.config();
        const auto geo = pipe::boundary_geometry(cfg);
        const auto t_load = Clock::now();
        CudaPrefill p(model, o);
        emit("{\"event\":\"loaded\",\"describe\":" + p.describe() + ",\"load_s\":" + std::to_string(ms_between(t_load, Clock::now()) / 1e3) + "}");
        const auto state_bytes = strix::deepseek41::serialize_attention_state(state);

        for (const auto K : cuts) {
          try {
            const auto plan = pipe::plan_boundary(target.layers, K);
            if (dry_run && !ref_feed) {
                std::uint64_t req = 1;
                std::vector<float> stream(std::size_t{8} * 4U * cfg.hidden_size), mix(std::size_t{8} * 24U);
                failures += run_windows(p, nullptr, req, state, state_bytes, windows, wref, passes, K, plan, geo, stream, mix, copy_sections, emit);
                continue;
            }
            p.import_state(state);
            const auto t_conn = Clock::now();
            std::unique_ptr<Client> cp;
            if (!dry_run) cp = std::make_unique<Client>(host, static_cast<std::uint16_t>(port_base + K), connect_wait, nodelay, spin);
            std::uint64_t req = 1;
            {
                pipe::Hello h;
                h.token = token;
                h.cut = K;
                h.layers = cfg.num_hidden_layers;
                h.hidden = cfg.hidden_size;
                h.hc = cfg.hc_mult;
                h.vocab = cfg.vocab_size;
                h.max_positions = o.max_positions;
                h.model_identity = target.manifest.layout_identity;
                h.backend = "cuda";
                h.numeric_identity = o.expert_tp ? "ds41-expert-tp2-i1152-v1" : "ds41-exact-v1";
                const auto hp = pipe::encode(h);
                log_msg(pipe::MsgType::hello, hp);
                if (cp) {
                    cp->send(pipe::MsgType::hello, req++, hp);
                    const auto ack = pipe::decode_hello_ack(cp->expect(pipe::MsgType::hello_ack).payload);
                    if (ack.status != 0U) throw std::runtime_error("TAIL refused the session: " + ack.message);
                }
                pipe::StateMsg sm;
                sm.manifest = "{\"layers\":[0,40],\"position\":" + std::to_string(state.position) + "}";
                sm.state = state_bytes;
                const auto sp = pipe::encode(sm);
                log_msg(pipe::MsgType::import_state, sp);
                if (cp) {
                    cp->send(pipe::MsgType::import_state, req++, sp);
                    (void)cp->expect(pipe::MsgType::ack);
                }
            }
            emit("{\"event\":\"session\",\"cut\":" + std::to_string(K) + ",\"port\":" + std::to_string(port_base + K) + ",\"connect_import_s\":" +
                 std::to_string(ms_between(t_conn, Clock::now()) / 1e3) + "}");
            std::uint32_t next = state.next_token, matched = 0, compared = 0;
            std::vector<double> tot, stp, dev, sec, enc, snd, wai;
            std::vector<float> stream(std::size_t{8} * 4U * cfg.hidden_size), mix(std::size_t{8} * 24U);
            for (std::uint32_t k = 0; k < steps; ++k) {
                const std::uint32_t pos = p.position();
                if (ref_feed) {  // teacher forcing: the reference's token at this position
                    const auto rf = ref.find(pos);
                    if (rf == ref.end()) break;
                    next = rf->second.fed;
                }
                const std::vector<std::uint32_t> fed{next};
                const auto t0 = Clock::now();
                CudaStepIo io;
                io.stream_out = stream.data();
                io.mix_out = mix.data();
                io.stop_layer = K;
                if (!copy_sections) io.sections_cut = K;  // the step's graph copies the cut's sections into pinned memory
                const auto s = p.step(fed, io);
                const auto t1 = Clock::now();
                const auto msg = build_step(p, s, pos, fed, K, plan, geo, stream, mix, copy_sections, 0U);
                const auto t2 = Clock::now();
                const auto payload = pipe::encode(msg);
                const auto t3 = Clock::now();
                log_msg(pipe::MsgType::step, payload);
                pipe::Result r;
                if (cp) cp->send(pipe::MsgType::step, req, payload, 0);
                const auto t4 = Clock::now();
                if (cp) r = pipe::decode_result(cp->expect(pipe::MsgType::result).payload, cfg.vocab_size, 8);
                else {  // --dry-run with --ref-feed: no TAIL
                    r.tokens = {0U};
                    r.logits_fnv = {0U};
                }
                const auto t5 = Clock::now();
                ++req;
                if (r.tokens.empty()) throw std::runtime_error("empty RESULT");
                next = r.tokens[0];
                std::string cmp;
                bool ok = true;
                const auto it = ref.find(pos);
                if (it != ref.end()) {
                    ok = !cp || (it->second.fed == fed[0] && it->second.argmax == next && (!gather || r.logits_fnv[0] == 0U || it->second.fnv == r.logits_fnv[0]));
                    ++compared;
                    matched += ok ? 1U : 0U;
                    if (!ok) ++failures;
                    cmp = ",\"ref_argmax\":" + std::to_string(it->second.argmax) + ",\"match\":" + (ok ? "true" : "false");
                }
                const double ms_total = ms_between(t0, t5);
                std::ostringstream os;
                os << "{\"event\":\"step\",\"cut\":" << K << ",\"pos\":" << pos << ",\"fed\":" << fed[0] << ",\"token\":" << next << ",\"logits_fnv\":\""
                   << hex(r.logits_fnv[0]) << "\",\"step_ms\":" << ms_between(t0, t1) << ",\"device_ms\":" << s.device_ms << ",\"engram_ms\":"
                   << s.engram_seconds * 1e3 << ",\"sections_ms\":" << ms_between(t1, t2) << ",\"encode_ms\":" << ms_between(t2, t3)
                   << ",\"send_ms\":" << ms_between(t3, t4) << ",\"wait_result_ms\":" << ms_between(t4, t5) << ",\"total_ms\":" << ms_total
                   << ",\"graphs\":" << (s.graphs ? "true" : "false") << cmp << "}";
                emit(os.str());
                if (tl)
                    tl << K << "," << pos << "," << fed[0] << "," << next << "," << (ok ? 1 : 0) << "," << ms_between(t0, t1) << "," << s.device_ms << ","
                       << ms_between(t1, t2) << "," << ms_between(t2, t3) << "," << ms_between(t3, t4) << "," << ms_between(t4, t5) << "," << ms_total << "\n";
                if (k >= 2U) {
                    tot.push_back(ms_total);
                    stp.push_back(ms_between(t0, t1));
                    dev.push_back(s.device_ms);
                    sec.push_back(ms_between(t1, t2));
                    enc.push_back(ms_between(t2, t3));
                    snd.push_back(ms_between(t3, t4));
                    wai.push_back(ms_between(t4, t5));
                }
            }
            if (!windows.empty()) failures += run_windows(p, cp.get(), req, state, state_bytes, windows, wref, passes, K, plan, geo, stream, mix, copy_sections, emit);
            log_msg(pipe::MsgType::close, {});
            if (cp) {
                cp->send(pipe::MsgType::close, req++, {});
                try {
                    (void)cp->expect(pipe::MsgType::ack);
                } catch (const std::exception&) {
                }
            }
            std::ostringstream os;
            os << "{\"event\":\"summary\",\"cut\":" << K << ",\"steps\":" << steps << ",\"median_total_ms\":" << median(tot)
               << ",\"tok_s\":" << (median(tot) > 0 ? 1e3 / median(tot) : 0.0) << ",\"median_step_ms\":" << median(stp)
               << ",\"median_device_ms\":" << median(dev) << ",\"median_sections_ms\":" << median(sec) << ",\"median_encode_ms\":" << median(enc)
               << ",\"median_send_ms\":" << median(snd) << ",\"median_wait_result_ms\":" << median(wai) << ",\"compared\":" << compared
               << ",\"matched\":" << matched << ",\"boost\":" << (boost ? "true" : "false") << ",\"graph_sections\":" << (copy_sections ? "false" : "true") << ",\"spin\":" << (spin ? "true" : "false") << ",\"nodelay\":" << (nodelay ? "true" : "false") << "}";
            emit(os.str());
          } catch (const std::exception& e) {  // this cut failed (TAIL absent or refused): the next cut
            ++failures;
            emit(std::string("{\"event\":\"cut_failed\",\"cut\":") + std::to_string(K) + ",\"error\":\"" + e.what() + "\"}");
          }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-cuda-ds41-pipeline-front: %s\n", e.what());
        return 1;
    }
    return failures ? 1 : 0;
}
