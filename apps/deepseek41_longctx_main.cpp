#include "strix/device/hip_gate.hpp"
#include "strix/hip_check.hpp"
#include "strix/deepseek41_attention_state.hpp"
#include "strix/deepseek41_device_forward.hpp"
#include "strix/deepseek41_prefill.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/dist/open_transport.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

// deepseek41 long context: a prompt filled by the device prefill (or an attention state read from a file), handed to the
// decode forward (DeviceForward::import_state), then greedy decode. One node (world 1, --stream-experts: the one-node
// control) or one rank of two (--rank R --transport usb4-auto), or both ranks in one process (--transport emulate, few
// layers). Prints the memory of the device at every stage, the decode speed and one line per generated token
// (token, FNV of the full logits) so runs and nodes compare with diff.
//   strix-deepseek41-longctx MODEL.gguf --prompt FILE (comma-separated ids) [--rows T]
//       [--world 1 | --rank R --transport usb4-auto | emulate] [--layers L]
//       [--state-in FILE] [--state-out FILE] [--teacher N] [--generate G] [--max-positions P]
//       [--prefill-only] [--stream-experts] [--no-tp] [--row-split] [--no-gather] [--timing]
//       [--chunk C] [--window W] [--lanes N] [--moe-chunk R] [--connect-timeout S] [--tokens-out FILE] [--segment S]
//       [--logits-out FILE]
// --logits-out F: rank 0 writes the full (gathered) logits of every generated step to F, f32 [G][vocab] (quality
// checks of declared mixed identities, e.g. ds41-prefill-ep+decode-tp: the prefill and the decode forward are
// separate objects here, so STRIX_DS41_EXPERT_TP applies to the decode only).
// --segment S: the prefill runs in pieces of S rows (DevicePrefill::extend), its row buffers sized for S rows only.
// Arms:
//   default        prefill the first T - N prompt tokens (N = --teacher, 0), then the decode forward feeds the last N
//                  prompt tokens one by one (teacher forced), then generates G tokens;
//   --state-in F   no prefill: the state of F (its tokens must be a prefix of the prompt), then as above from there;
//   --state-out F  writes the state after the prefill (before any decode step);
//   --state-in F --prefill-rest  the prefill continues from the state of F (DevicePrefill::import_state + extend) up to
//                  T - N: a long prompt filled over several runs (pair turns), one state file per step.
// With N > 0 the state after the N steps is compared with the state after the whole prompt when a prefill of it is known
// (--state-in of the whole prompt): equal hashes = prefill and decode agree at that length.
namespace {

using strix::hip_check;

struct Args {
    std::string model, prompt_path, transport, state_in, state_out, tokens_out, logits_out;
    std::uint32_t rank = 0, world = 1, layers = 40, rows = 0, teacher = 0, generate = 16, max_positions = 0;
    std::uint32_t chunk = 1024, window = 8, lanes = 4, moe_chunk = 4096, segment = 0;
    unsigned connect_timeout = 60;
    bool stream_experts = false, tp = true, row_split = false, gather = true, timing = false, prefill_only = false, prefill_rest = false;
};

std::uint64_t fnv(const std::vector<float>& v) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const float f : v) {
        std::uint32_t b;
        std::memcpy(&b, &f, 4);
        for (int k = 0; k < 4; ++k, b >>= 8U) h = (h ^ (b & 0xffU)) * 1099511628211ULL;
    }
    return h;
}

double since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

// Device memory used / total (GiB) and the host's MemAvailable (GiB): one line per stage.
void memory_line(const char* who, const char* stage) {
    std::size_t free_b = 0, total_b = 0;
    hip_check(hipMemGetInfo(&free_b, &total_b), "hipMemGetInfo");
    double avail = 0.0;
    std::ifstream mi("/proc/meminfo");
    for (std::string k; mi >> k;) {
        std::uint64_t v = 0;
        mi >> v;
        if (k == "MemAvailable:") avail = static_cast<double>(v) / 1048576.0;
        mi.ignore(64, '\n');
    }
    std::printf("%s memory [%s]: device used %.2f GiB of %.2f (free %.2f), host MemAvailable %.2f GiB\n", who, stage,
                static_cast<double>(total_b - free_b) / 1073741824.0, static_cast<double>(total_b) / 1073741824.0,
                static_cast<double>(free_b) / 1073741824.0, avail);
    std::fflush(stdout);
}

std::vector<std::uint32_t> read_prompt(const std::string& path, std::uint32_t rows) {
    std::ifstream in(path);
    if (!in) throw std::invalid_argument("cannot read " + path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<std::uint32_t> ids;
    std::stringstream ss(text);
    for (std::string t; std::getline(ss, t, ',');)
        if (t.find_first_not_of(" \r\n\t") != std::string::npos) ids.push_back(static_cast<std::uint32_t>(std::stoul(t)));
    if (rows != 0U) {
        if (rows > ids.size()) throw std::invalid_argument("prompt shorter than --rows");
        ids.resize(rows);
    }
    return ids;
}

// Both ranks here before either goes on: a 4-byte exchange, waited for.
void barrier(strix::dist::Communicator* comm, std::uint32_t round) {
    if (comm == nullptr) return;
    void* buffers = nullptr;
    hipStream_t stream = nullptr;
    hip_check(hipMalloc(&buffers, 8), "barrier");
    hip_check(hipStreamCreate(&stream), "barrier");
    hip_check(hipMemsetAsync(buffers, 0, 8, stream), "barrier");
    comm->exchange(stream, buffers, static_cast<char*>(buffers) + 4, 4,
                   strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 7U, round});
    hip_check(hipStreamSynchronize(stream), "barrier");
    (void)hipStreamDestroy(stream);
    (void)hipFree(buffers);
}

struct RankResult {
    std::vector<std::uint32_t> generated;
    std::vector<std::uint64_t> logits_fnv;
    std::uint64_t state_digest = 0;
};

RankResult run(const Args& a, const std::vector<std::uint32_t>& prompt, std::uint32_t rank, strix::dist::Communicator* comm, bool print) {
    const bool two = comm != nullptr;
    char who[32];
    std::snprintf(who, sizeof(who), two ? "rank %u" : "node", rank);
    const auto T = static_cast<std::uint32_t>(prompt.size());
    if (a.teacher >= T) throw std::invalid_argument("--teacher must leave at least one prompt token to the prefill or the state");
    if (print) memory_line(who, "start");
    RankResult out;

    // 1. the attention state of the prompt's first S tokens: a file, the device prefill, or a file the prefill extends
    strix::deepseek41::AttentionState state;
    const std::uint32_t S = T - a.teacher;  // the prefill's end (a file alone may end earlier: decode feeds the rest)
    std::uint32_t have = 0;                 // positions of the state so far
    if (!a.state_in.empty()) {
        const auto t0 = std::chrono::steady_clock::now();
        state = strix::deepseek41::load_attention_state(a.state_in);
        if (state.position > T || !std::equal(state.tokens.begin(), state.tokens.end(), prompt.begin()))
            throw std::invalid_argument("--state-in: its tokens are not a prefix of the prompt");
        if (state.layers.size() != a.layers) throw std::invalid_argument("--state-in: layer count differs from --layers");
        have = state.position;
        if (print)
            std::printf("%s state read: %u positions, %.1f MB, digest %016llx, next token %u (%.1f s)\n", who, have,
                        static_cast<double>(state.bytes()) / 1e6, static_cast<unsigned long long>(state.digest(true)), state.next_token, since(t0));
    }
    if (a.state_in.empty() || (a.prefill_rest && have < S)) {
        strix::deepseek41::PrefillOptions o;
        o.layers = a.layers;
        o.max_positions = S;
        o.max_rows = a.segment != 0U && a.segment < S - have ? a.segment : 0U;
        if (have != 0U && o.max_rows == 0U) o.max_rows = S - have;
        o.chunk_rows = a.chunk;
        o.window_rows = a.window;
        o.attention_lanes = a.lanes;
        o.moe_chunk_rows = a.moe_chunk;
        o.tp_attention = a.tp && !a.row_split;
        o.row_split = two && a.row_split;
        o.stream_experts = !two && a.stream_experts;
        o.timing = a.timing;
        o.rank = rank;
        o.world = two ? 2U : 1U;
        o.communicator = comm;
        strix::deepseek41::PrefillResult r;
        {
            strix::deepseek41::DevicePrefill pre(a.model, o);
            if (print) {
                std::printf("%s prefill: layers [0, %u), device weights %.2f GiB loaded in %.1f s (%s), rows per run %u\n", who, a.layers,
                            static_cast<double>(pre.weight_bytes()) / 1073741824.0, pre.load_seconds(),
                            two ? (o.row_split ? "row split" : o.tp_attention ? "EP + TP attention" : "EP") : (o.stream_experts ? "experts streamed" : "all resident"),
                            o.max_rows != 0U ? o.max_rows : S);
                memory_line(who, "prefill loaded");
            }
            if (have != 0U) pre.import_state(state);
            barrier(comm, 1U);
            const std::uint32_t step = o.max_rows != 0U ? o.max_rows : S;
            double total_s = 0.0;
            for (std::uint32_t p0 = have; p0 < S; p0 += step) {
                const std::span<const std::uint32_t> part(prompt.data() + p0, std::min(step, S - p0));
                r = p0 == 0U ? pre.run(part) : pre.extend(part);
                total_s += r.seconds;
                if (print && part.size() < S) {
                    std::printf("%s prefill run [%u, %zu): %.2f s = %.1f tok/s\n", who, p0, p0 + part.size(), r.seconds,
                                static_cast<double>(part.size()) / r.seconds);
                    memory_line(who, "after the run");
                }
            }
            if (print) {
                std::printf("%s prefill: rows [%u, %u) in %.2f s = %.1f tok/s (last run: engram %.2f s, head %.3f s), next token %u, logits_fnv %016llx\n",
                            who, have, S, total_s, static_cast<double>(S - have) / total_s, r.engram_seconds, r.head_seconds, r.next_token,
                            static_cast<unsigned long long>(fnv(r.logits)));
                memory_line(who, "prefill done");
            }
            state = pre.export_state();
        }
        state.tokens.assign(prompt.begin(), prompt.begin() + S);
        state.next_token = r.next_token;
        if (print) {
            std::printf("%s state: %u positions, %.1f MB on the host, digest %016llx\n", who, S, static_cast<double>(state.bytes()) / 1e6,
                        static_cast<unsigned long long>(state.digest(true)));
            memory_line(who, "prefill freed");
        }
        if (!a.state_out.empty()) {
            const auto t0 = std::chrono::steady_clock::now();
            strix::deepseek41::save_attention_state(a.state_out, state);
            if (print) std::printf("%s state written to %s (%.1f s)\n", who, a.state_out.c_str(), since(t0));
        }
    }
    const std::uint32_t S_state = state.position;  // decode feeds prompt[S_state, T) teacher forced
    out.state_digest = state.digest(true);
    if (a.prefill_only) return out;

    // 2. the decode forward from that state
    strix::deepseek41::DeviceForwardOptions o;
    o.layers = a.layers;
    o.max_positions = a.max_positions != 0U ? a.max_positions : T + a.generate + 8U;
    o.rank = rank;
    o.world = two ? 2U : 1U;
    o.communicator = comm;
    o.op_hashes = false;
    o.gather_logits = a.gather;
    o.tp_attention = two && a.tp;
    o.stream_experts = !two && a.stream_experts;
    o.layer_timing = false;
    strix::deepseek41::DeviceForward fwd(a.model, o);
    if (print) {
        std::printf("%s decode: device weights %.2f GiB loaded in %.1f s, max positions %u (%s)\n", who, static_cast<double>(fwd.weight_bytes()) / 1073741824.0,
                    fwd.load_seconds(), o.max_positions, two ? (o.tp_attention ? "EP + TP attention" : "EP") : (o.stream_experts ? "experts streamed" : "all resident"));
        memory_line(who, "decode loaded");
    }
    {
        const auto t0 = std::chrono::steady_clock::now();
        fwd.import_state(state);
        const auto back = fwd.export_state();  // round trip: the forward holds exactly the state
        const auto h0 = state.hashes(), h1 = back.hashes();
        std::size_t bad = h0.size() == h1.size() ? 0U : 1U;
        for (std::size_t i = 0; i < h0.size() && i < h1.size(); ++i)
            if (h0[i] != h1[i] && (h0[i].first.rfind("topk", 0) != 0 || state.layers.empty())) ++bad;
        if (print) {
            std::printf("%s state imported in %.1f s: round trip %s (%zu hashes)\n", who, since(t0), bad == 0U ? "EQUAL" : "DIFFERENT", h0.size());
            memory_line(who, "state imported");
        }
        if (bad != 0U) throw std::runtime_error("import_state round trip differs");
    }
    barrier(comm, 2U);

    // 3. the last N prompt tokens teacher forced, then G greedy tokens
    std::uint32_t token = 0;
    std::vector<float> logits;
    for (std::uint32_t p = S_state; p < T; ++p) {
        logits = fwd.step(prompt[p]);
        if (print && a.gather)
            std::printf("%s teacher pos %u fed %u argmax %u logits_fnv %016llx\n", who, p, prompt[p], fwd.next_token(),
                        static_cast<unsigned long long>(fnv(logits)));
    }
    token = S_state < T ? fwd.next_token() : state.next_token;
    if (S_state < T && print) {
        const auto st = fwd.export_state();
        std::printf("%s state after the teacher-forced tokens: position %u, digest %016llx, next token %u\n", who, st.position,
                    static_cast<unsigned long long>(st.digest(true)), token);
    }
    std::FILE* tf = print && !a.tokens_out.empty() ? std::fopen(a.tokens_out.c_str(), "w") : nullptr;
    std::FILE* lf = print && a.gather && !a.logits_out.empty() ? std::fopen(a.logits_out.c_str(), "wb") : nullptr;
    double decode_s = 0.0;
    std::uint32_t timed = 0;
    for (std::uint32_t g = 0; g < a.generate; ++g) {
        out.generated.push_back(token);
        logits = fwd.step(token);
        const auto next = fwd.next_token();
        if (g >= 2U) {  // the first steps pay one-time costs
            decode_s += fwd.last_step_seconds();
            ++timed;
        }
        const std::uint64_t h = a.gather ? fnv(logits) : 0U;
        if (lf != nullptr) std::fwrite(logits.data(), sizeof(float), logits.size(), lf);
        out.logits_fnv.push_back(h);
        if (print) {
            std::printf("%s gen %u pos %u fed %u argmax %u logits_fnv %016llx step %.1f ms\n", who, g, T + g, token, next,
                        static_cast<unsigned long long>(h), fwd.last_step_seconds() * 1e3);
            if (tf != nullptr) std::fprintf(tf, "%u %u %016llx\n", T + g, token, static_cast<unsigned long long>(h));
        }
        token = next;
    }
    if (tf != nullptr) std::fclose(tf);
    if (lf != nullptr) std::fclose(lf);
    if (print) {
        std::printf("%s decode at context %u: %u steps after the first two, %.2f tok/s (%.1f ms/token)\n", who, T, timed,
                    timed ? timed / decode_s : 0.0, timed ? decode_s * 1e3 / timed : 0.0);
        std::string text;
        for (const auto t : out.generated) text += (text.empty() ? "" : ",") + std::to_string(t);
        std::printf("%s generated ids: %s\n", who, text.c_str());
        memory_line(who, "decode done");
        if (comm != nullptr) {
            const auto& s = comm->stats();
            std::printf("%s exchanges %llu, %.1f MB sent, %.2f s in exchange (wire %.2f)\n", who, static_cast<unsigned long long>(s.exchanges),
                        static_cast<double>(s.bytes_sent) / 1e6, s.seconds, s.wire);
        }
    }
    return out;
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
        const auto num = [&] { return static_cast<std::uint32_t>(std::stoul(next())); };
        if (s == "--prompt") a.prompt_path = next();
        else if (s == "--rows") a.rows = num();
        else if (s == "--layers") a.layers = num();
        else if (s == "--world") a.world = num();
        else if (s == "--rank") { a.rank = num(); a.world = 2; }
        else if (s == "--transport") { a.transport = next(); a.world = 2; }
        else if (s == "--state-in") a.state_in = next();
        else if (s == "--state-out") a.state_out = next();
        else if (s == "--tokens-out") a.tokens_out = next();
        else if (s == "--logits-out") a.logits_out = next();
        else if (s == "--teacher") a.teacher = num();
        else if (s == "--generate") a.generate = num();
        else if (s == "--max-positions") a.max_positions = num();
        else if (s == "--prefill-only") a.prefill_only = true;
        else if (s == "--prefill-rest") a.prefill_rest = true;
        else if (s == "--stream-experts") a.stream_experts = true;
        else if (s == "--no-tp") a.tp = false;
        else if (s == "--row-split") a.row_split = true;
        else if (s == "--no-gather") a.gather = false;
        else if (s == "--timing") a.timing = true;
        else if (s == "--chunk") a.chunk = num();
        else if (s == "--window") a.window = num();
        else if (s == "--lanes") a.lanes = num();
        else if (s == "--moe-chunk") a.moe_chunk = num();
        else if (s == "--segment") a.segment = num();
        else if (s == "--connect-timeout") a.connect_timeout = static_cast<unsigned>(std::stoul(next()));
        else if (a.model.empty()) a.model = s;
        else throw std::invalid_argument("unknown argument " + s);
    }
    if (a.model.empty() || a.prompt_path.empty()) throw std::invalid_argument("usage: strix-deepseek41-longctx MODEL.gguf --prompt FILE ...");
    const auto prompt = read_prompt(a.prompt_path, a.rows);
    if (a.world == 1U) {
        run(a, prompt, 0U, nullptr, true);
        return 0;
    }
    constexpr std::uint64_t kPlan = 0x64734C43ULL;  // the group id of this app's ranks
    if (a.transport == "emulate") {  // both ranks in this process (memory transport): only for a few layers
        auto [t0, t1] = strix::dist::memory_pair();
        std::exception_ptr failure;
        RankResult r1;
        std::thread peer([&, t = std::move(t1)]() mutable {
            try {
                strix::dist::Communicator comm(strix::dist::Group{1U, 2U, 1U, kPlan}, std::move(t), std::size_t{22} << 20U);
                r1 = run(a, prompt, 1U, &comm, false);
            } catch (...) {
                failure = std::current_exception();
            }
        });
        RankResult r0;
        try {
            strix::dist::Communicator comm(strix::dist::Group{0U, 2U, 1U, kPlan}, std::move(t0), std::size_t{22} << 20U);
            r0 = run(a, prompt, 0U, &comm, true);
        } catch (...) {
            peer.join();
            throw;
        }
        peer.join();
        if (failure) std::rethrow_exception(failure);
        const bool same = r0.generated == r1.generated && r0.logits_fnv == r1.logits_fnv && r0.state_digest == r1.state_digest;
        std::printf("ranks: %s (state digest %016llx / %016llx)\n", same ? "identical tokens and logits" : "DIFFERENT",
                    static_cast<unsigned long long>(r0.state_digest), static_cast<unsigned long long>(r1.state_digest));
        return same ? 0 : 1;
    }
    ::alarm(a.connect_timeout + 30U);  // a stream on a dead USB4 link blocks open(): the alarm ends the process
    auto opened = strix::dist::open_pair(a.transport, a.rank, {std::chrono::seconds(a.connect_timeout)});
    ::alarm(0);
    if (!opened.receipt.empty()) std::printf("usb4 topology: %s\n", opened.receipt.c_str());
    strix::dist::Communicator comm(strix::dist::Group{a.rank, 2U, 1U, kPlan}, std::move(opened.host), std::move(opened.direct),
                                   std::size_t{22} << 20U);
    std::printf("connected (%s)\n", comm.transport_name().c_str());
    run(a, prompt, a.rank, &comm, true);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
