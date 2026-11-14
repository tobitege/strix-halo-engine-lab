#include "strix/device/hip_gate.hpp"
#include "strix/deepseek41_device_forward.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/dist/open_transport.hpp"
#include "strix/hip_check.hpp"
#include "strix/ds41_layer.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

// Quality gate runner for requantization candidates (two ranks, expert-parallel, split-vocabulary head).
// Every mode feeds a fixed token sequence teacher-forced in verify windows (rows of a window are bitwise the rows
// of single steps), so one run scores thousands of positions; the context runs on across the sequence's segments.
//   strix-deepseek41-requant-eval MODEL.gguf --rank R --transport usb4auto | emulate [--overlay CAND.gguf]
//       [--cb4] [--native-fp8 DIR [--native-fp8-sites LIST]] --seq SEQ.u32 [--rows 8] [--layers 40] [--max-positions N] [--limit N]
//       --save-ref DIR             write this rank's vocabulary slice of every row's logits (DIR/ref-rR.f32)
//     | --compare-ref DIR --out F  per row, against the saved reference slice: lse of both, sum_i P_i (p_i - q_i) over
//                                  the slice, both argmaxes and the target logits (F.rR, binary, see RowStats)
//     | --greedy PROMPTS --out F   build a sequence: for each line "name gen ids,..." feed the prompt, then generate
//                                  gen tokens greedily (single steps); F.u32 holds the sequence, F.segs the segments
//     | --calib F                  single steps with op outputs downloaded: per-channel sums of squares of the
//                                  inputs of the attention and shared-expert projections (rank 0 writes F)
// Rank 0 and rank 1 must get the same arguments; each writes its own files.
namespace {

struct Args {
    std::string model, transport = "emulate", overlay, seq_path, save_ref, compare_ref, out, greedy, calib;
    std::string native_fp8, native_fp8_sites;  // DeviceForwardOptions::native_fp8 / native_fp8_sites (empty: as the options)
    std::uint32_t rank = 0, layers = 40, rows = 8, max_positions = 0, limit = 0;
    unsigned connect_timeout = 60;
    bool cb4 = false, tp = false, router_bf16 = false, planar = false, f8 = false;
    std::uint32_t bench = 0;
};

// One row of --compare-ref output (little endian, 48 bytes).
struct RowStats {
    double lse_p, lse_q;   // log-sum-exp of the reference / candidate logits over this rank's slice
    double s;              // sum_i softmax_slice(p)_i * (p_i - q_i)
    float max_p, max_q;    // slice maxima
    std::uint32_t arg_p, arg_q;  // global vocabulary ids of the slice argmaxes (first index on ties)
    float tgt_p, tgt_q;    // logits of the next token if it lies in this slice, else NaN
};
static_assert(sizeof(RowStats) == 48);

void check(hipError_t e, const char* what) {
    ::strix::hip_check_prefixed("hip: ", e, what);
}

std::vector<std::uint32_t> read_u32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const auto n = static_cast<std::size_t>(f.tellg());
    if (n % 4U != 0U) throw std::runtime_error(path + ": size is not a multiple of 4");
    std::vector<std::uint32_t> v(n / 4U);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
    return v;
}

// A verify window must not start before the first compressed entry of the attention: positions below this run as
// single steps (the rows are bitwise the same either way).
constexpr std::size_t warm_positions = 128;
std::uint32_t window_rows(std::size_t pos, std::uint32_t rows, std::size_t left) {
    return pos < warm_positions ? 1U : static_cast<std::uint32_t>(std::min<std::size_t>(rows, left));
}

struct Calib {
    std::map<std::string, std::pair<std::vector<double>, std::uint64_t>> sums;
    void add(const std::string& label, const std::vector<float>& v) {
        // inputs of q_a and kv (attn norm), q_b (q norm), wo_a (attention output), wo_b (o_a output), shared expert (ffn norm)
        const bool want = label.starts_with("rms_norm@") || label.starts_with("attention@") || label.starts_with("o_a@");
        if (!want || label.starts_with("hc_")) return;
        auto& [s, n] = sums[label];
        if (s.empty()) s.assign(v.size(), 0.0);
        for (std::size_t i = 0; i < v.size(); ++i) s[i] += static_cast<double>(v[i]) * static_cast<double>(v[i]);
        ++n;
    }
    void write(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        for (const auto& [label, e] : sums) {
            const std::string head = label + " " + std::to_string(e.first.size()) + " " + std::to_string(e.second) + "\n";
            f.write(head.data(), static_cast<std::streamsize>(head.size()));
            f.write(reinterpret_cast<const char*>(e.first.data()), static_cast<std::streamsize>(e.first.size() * 8U));
        }
        if (!f) throw std::runtime_error("cannot write " + path);
    }
};

void run_rank(const Args& a, std::uint32_t rank, std::unique_ptr<strix::dist::Transport> transport, bool print,
              std::unique_ptr<strix::dist::DeviceExchange> direct = nullptr) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    strix::dist::Communicator comm(strix::dist::Group{rank, 2U, 1U, 0x64733431ULL}, std::move(transport), std::move(direct), 512U << 10U);
    std::vector<std::uint32_t> seq;
    struct Prompt {
        std::string name;
        std::uint32_t gen = 0;
        std::vector<std::uint32_t> ids;
    };
    std::vector<Prompt> prompts;
    if (!a.greedy.empty()) {
        std::ifstream f(a.greedy);
        if (!f) throw std::runtime_error("cannot open " + a.greedy);
        for (std::string line; std::getline(f, line);) {
            if (line.empty()) continue;
            std::stringstream ls(line);
            Prompt p;
            std::string ids;
            ls >> p.name >> p.gen >> ids;
            std::stringstream is(ids);
            for (std::string t; std::getline(is, t, ',');) p.ids.push_back(static_cast<std::uint32_t>(std::stoul(t)));
            prompts.push_back(std::move(p));
        }
    } else {
        seq = read_u32(a.seq_path);
        if (a.limit != 0U && seq.size() > a.limit) seq.resize(a.limit);
    }
    std::size_t need = seq.size();
    for (const auto& p : prompts) need += p.ids.size() + p.gen;
    need += a.bench;

    strix::deepseek41::DeviceForwardOptions o;
    o.layers = a.layers;
    o.max_positions = a.max_positions != 0U ? a.max_positions : static_cast<std::uint32_t>(need + 16U);
    o.rank = rank;
    o.world = 2;
    o.communicator = &comm;
    o.op_hashes = !a.calib.empty();
    o.gather_logits = false;  // each rank keeps its slice; the gate combines the slices' statistics offline
    o.weights_overlay = a.overlay;
    o.cb4_overlay = a.cb4;
    o.tp_attention = a.tp;
    o.router_bf16 = a.router_bf16;
    o.f16_as_f8 = a.f8;
    o.packed_planar = a.planar;
    if (!a.native_fp8.empty()) o.native_fp8 = a.native_fp8;
    if (!a.native_fp8_sites.empty()) o.native_fp8_sites = a.native_fp8_sites;
    Calib calib;
    if (!a.calib.empty() && rank == 0U) o.buffer_tap = [&](const std::string& l, const std::vector<float>& v) { calib.add(l, v); };
    strix::deepseek41::DeviceForward fwd(a.model, o);
    const auto V = 129280U;
    const auto slice = strix::ds41::vocab_slice(V, 2U, rank);
    if (print) {
        std::printf("rank %u: layers [0, %u), device weights %.2f GiB loaded in %.1f s\n", rank, a.layers,
                    static_cast<double>(fwd.weight_bytes()) / 1073741824.0, fwd.load_seconds());
        const auto d = fwd.overlay_description();
        std::printf("%s\n", d.empty() ? "overlay none (base model)" : d.c_str());
        if (a.cb4 || a.router_bf16 || a.f8) std::printf("packed: %zu tensors (CB4/U5/U6 and BF16 routers)\n", fwd.cb4_tensors());
        if (!a.native_fp8.empty()) std::printf("storage: %s\n", fwd.storage_description().c_str());
        std::fflush(stdout);
    }
    const auto stream = static_cast<hipStream_t>(fwd.stream());
    std::vector<float> host(std::size_t{strix::deepseek41::DeviceForward::max_window_rows} * slice.count);
    const auto fetch = [&](std::uint32_t R) {  // this rank's slice of the last window's logits
        const auto lr = fwd.device_logits_rows(slice.first);
        check(hipMemcpy2DAsync(host.data(), std::size_t{slice.count} * 4U, lr.p, lr.stride * 4U, std::size_t{slice.count} * 4U, R,
                               hipMemcpyDeviceToHost, stream),
              "logits download");
        check(hipStreamSynchronize(stream), "logits download");
    };

    if (!a.greedy.empty()) {  // build the greedy sequence
        std::vector<std::uint32_t> out;
        std::string segs;
        for (const auto& p : prompts) {
            const std::size_t start = out.size();
            std::size_t i = 0;
            while (i + 1U < p.ids.size()) {  // all prompt tokens but the last in windows
                const auto R = window_rows(out.size() + i, a.rows, p.ids.size() - 1U - i);
                (void)fwd.step_window(std::span<const std::uint32_t>(p.ids.data() + i, R));
                fwd.commit_window(R);
                i += R;
            }
            out.insert(out.end(), p.ids.begin(), p.ids.end());
            std::uint32_t tok = p.ids.back();
            for (std::uint32_t g = 0; g < p.gen; ++g) {
                (void)fwd.step(tok);
                tok = fwd.next_token();
                out.push_back(tok);
            }
            // the last generated token has not been fed: feed it so the next prompt's context holds the whole segment
            (void)fwd.step(out.back());
            segs += p.name + " " + std::to_string(start) + " " + std::to_string(p.ids.size()) + " " + std::to_string(p.gen) + "\n";
            if (print) {
                std::printf("greedy %s: prompt %zu, generated %u (%.1f s)\n", p.name.c_str(), p.ids.size(), p.gen, elapsed());
                std::fflush(stdout);
            }
        }
        if (print) {
            std::ofstream f(a.out + ".u32", std::ios::binary);
            f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size() * 4U));
            std::ofstream s(a.out + ".segs");
            s << segs;
            std::printf("wrote %s.u32 (%zu tokens) and %s.segs\n", a.out.c_str(), out.size(), a.out.c_str());
        }
        return;
    }

    if (a.bench != 0U) {  // greedy decode after the sequence as context: single steps, nothing downloaded
        std::size_t pos = 0;
        while (pos < seq.size()) {
            const auto R = window_rows(pos, a.rows, seq.size() - pos);
            (void)fwd.step_window(std::span<const std::uint32_t>(seq.data() + pos, R));
            fwd.commit_window(R);
            pos += R;
        }
        std::uint32_t tok = fwd.last_tokens().back();
        double total = 0.0;
        std::uint64_t h = 14695981039346656037ULL;
        for (std::uint32_t i = 0; i < a.bench; ++i) {
            (void)fwd.step(tok);
            tok = fwd.next_token();
            h = (h ^ tok) * 1099511628211ULL;
            if (i >= 4U) total += fwd.last_step_seconds();  // the first steps pay one-time costs
        }
        if (print)
            std::printf("bench: context %zu, %u greedy tokens, %.2f tok/s (%.2f ms/token after 4), tokens_fnv %016llx\n", seq.size(), a.bench,
                        static_cast<double>(a.bench - 4U) / total, total * 1e3 / static_cast<double>(a.bench - 4U),
                        static_cast<unsigned long long>(h));
        return;
    }

    if (!a.calib.empty()) {  // single steps with the op outputs downloaded
        for (std::size_t pos = 0; pos < seq.size(); ++pos) {
            (void)fwd.step(seq[pos]);
            if (print && (pos % 64U == 0U || pos + 1U == seq.size())) {
                std::printf("calib pos %zu (%.1f s)\n", pos, elapsed());
                std::fflush(stdout);
            }
        }
        if (rank == 0U) {
            calib.write(a.calib);
            std::printf("wrote %s (%zu inputs, %zu positions)\n", a.calib.c_str(), calib.sums.size(), seq.size());
        }
        return;
    }

    const std::string suffix = ".r" + std::to_string(rank);
    std::ofstream ref_out, stats_out;
    std::ifstream ref_in;
    if (!a.save_ref.empty()) ref_out.open(a.save_ref + "/ref" + suffix + ".f32", std::ios::binary);
    if (!a.compare_ref.empty()) {
        ref_in.open(a.compare_ref + "/ref" + suffix + ".f32", std::ios::binary);
        stats_out.open(a.out + suffix, std::ios::binary);
        if (!ref_in || !stats_out) throw std::runtime_error("cannot open the reference or the output");
    }
    if (ref_out.fail() && !a.save_ref.empty()) throw std::runtime_error("cannot create the reference file");
    std::vector<float> ref(slice.count);
    std::size_t pos = 0, top1 = 0, scored = 0;
    const double t_start = elapsed();
    while (pos < seq.size()) {
        const auto R = window_rows(pos, a.rows, seq.size() - pos);
        (void)fwd.step_window(std::span<const std::uint32_t>(seq.data() + pos, R));
        fwd.commit_window(R);
        fetch(R);
        for (std::uint32_t r = 0; r < R; ++r) {
            const float* q = host.data() + std::size_t{r} * slice.count;
            const std::size_t row = pos + r;
            if (ref_out.is_open()) ref_out.write(reinterpret_cast<const char*>(q), static_cast<std::streamsize>(slice.count * 4U));
            if (!stats_out.is_open()) continue;
            ref_in.read(reinterpret_cast<char*>(ref.data()), static_cast<std::streamsize>(slice.count * 4U));
            if (!ref_in) throw std::runtime_error("the reference is shorter than the sequence");
            RowStats st{};
            std::uint32_t ip = 0, iq = 0;
            for (std::uint32_t i = 1; i < slice.count; ++i) {
                if (ref[i] > ref[ip]) ip = i;
                if (q[i] > q[iq]) iq = i;
            }
            st.max_p = ref[ip];
            st.max_q = q[iq];
            double zp = 0.0, zq = 0.0, s = 0.0;
            for (std::uint32_t i = 0; i < slice.count; ++i) {
                const double ep = std::exp(static_cast<double>(ref[i]) - st.max_p);
                zp += ep;
                zq += std::exp(static_cast<double>(q[i]) - st.max_q);
                s += ep * (static_cast<double>(ref[i]) - static_cast<double>(q[i]));
            }
            st.lse_p = st.max_p + std::log(zp);
            st.lse_q = st.max_q + std::log(zq);
            st.s = s / zp;
            st.arg_p = slice.first + ip;
            st.arg_q = slice.first + iq;
            st.tgt_p = st.tgt_q = std::numeric_limits<float>::quiet_NaN();
            if (row + 1U < seq.size() && seq[row + 1U] >= slice.first && seq[row + 1U] < slice.first + slice.count) {
                st.tgt_p = ref[seq[row + 1U] - slice.first];
                st.tgt_q = q[seq[row + 1U] - slice.first];
            }
            if (ip == iq) ++top1;
            ++scored;
            stats_out.write(reinterpret_cast<const char*>(&st), sizeof st);
        }
        pos += R;
        if (print && (pos % 512U < R || pos == seq.size())) {
            std::printf("pos %zu (%.1f s, %.1f pos/s)%s\n", pos, elapsed(), static_cast<double>(pos) / (elapsed() - t_start),
                        scored ? (" slice top-1 " + std::to_string(top1) + "/" + std::to_string(scored)).c_str() : "");
            std::fflush(stdout);
        }
    }
    if (ref_out.is_open()) ref_out.close();
    if (stats_out.is_open()) stats_out.close();
    if (print) std::printf("done: %zu positions in %.1f s\n", seq.size(), elapsed() - t_start);
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
        if (s == "--rank") a.rank = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--transport") a.transport = next();
        else if (s == "--layers") a.layers = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--overlay") a.overlay = next();
        else if (s == "--cb4") a.cb4 = true;
        else if (s == "--tp-attention") a.tp = true;
        else if (s == "--router-bf16") a.router_bf16 = true;
        else if (s == "--f16-as-f8") a.f8 = true;
        else if (s == "--planar") a.planar = true;
        else if (s == "--native-fp8") a.native_fp8 = next();
        else if (s == "--native-fp8-sites") a.native_fp8_sites = next();
        else if (s == "--bench") a.bench = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--seq") a.seq_path = next();
        else if (s == "--limit") a.limit = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--rows") a.rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--max-positions") a.max_positions = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--save-ref") a.save_ref = next();
        else if (s == "--compare-ref") a.compare_ref = next();
        else if (s == "--out") a.out = next();
        else if (s == "--greedy") a.greedy = next();
        else if (s == "--calib") a.calib = next();
        else if (s == "--connect-timeout") a.connect_timeout = static_cast<unsigned>(std::stoul(next()));
        else if (a.model.empty()) a.model = s;
        else throw std::invalid_argument("unknown argument " + s);
    }
    const int modes = !a.save_ref.empty() + !a.compare_ref.empty() + !a.greedy.empty() + !a.calib.empty() + (a.bench != 0U);
    if (a.model.empty() || modes < 1 || (a.greedy.empty() && a.seq_path.empty()) || a.rows < 1U ||
        a.rows > strix::deepseek41::DeviceForward::max_window_rows || ((!a.compare_ref.empty() || !a.greedy.empty()) && a.out.empty()) ||
        (modes > 1 && !(modes == 2 && !a.save_ref.empty() && !a.compare_ref.empty())))
        throw std::invalid_argument("usage: strix-deepseek41-requant-eval MODEL.gguf --rank R --transport usb4auto|emulate [--overlay O] "
                                    "--seq S.u32 (--save-ref DIR | --compare-ref DIR --out F) | --greedy PROMPTS --out F | --calib F");
    if (a.transport == "emulate") {
        auto [t0, t1] = strix::dist::memory_pair();
        std::exception_ptr failure;
        std::thread peer([&, t = std::move(t1)]() mutable {
            try {
                run_rank(a, 1U, std::move(t), false);
            } catch (...) {
                failure = std::current_exception();
            }
        });
        try {
            run_rank(a, 0U, std::move(t0), true);
        } catch (...) {
            peer.join();
            throw;
        }
        peer.join();
        if (failure) std::rethrow_exception(failure);
        return 0;
    }
    ::alarm(a.connect_timeout);  // a blocked open() of a dead link would hold both nodes
    auto opened = strix::dist::open_pair(a.transport, a.rank, {std::chrono::seconds(a.connect_timeout)});
    ::alarm(0);
    if (!opened.receipt.empty()) std::printf("usb4 topology: %s\n", opened.receipt.c_str());
    auto transport = std::move(opened.host);
    std::printf("connected (%s)\n", transport->name().c_str());
    std::fflush(stdout);
    run_rank(a, a.rank, std::move(transport), true, std::move(opened.direct));
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
