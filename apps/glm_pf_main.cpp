// strix-glm5-pf: gate and timing of the chunked prefill seam (design notes).
//   strix-glm5-pf MODEL_DIR --prompt T.u32 [--layers N] [--capacity C] [--max-rows R] [--split P] [--decode K] [--time 0|1]
// One node (world 1), slot 0. For each identity (STRIX_GLM_PF_TIERB-independent; both run here):
//   ref   : the reference prefill (decode row by row) -> next token, last-row logits, every state byte, K decode tokens;
//   gemv  : chunked prefill, Tier B = strix_gemv R = 1 per row (the non-D10 proof): must equal ref BIT FOR BIT
//           (token, logits, state, decode tokens);
//   wmma  : chunked prefill, Tier B = pinned WMMA (glm-pf-wmma-v1, D10): deviation vs ref recorded; the prompt split at
//           P (two prefill calls, the second at pos0 = P) must equal the one-call wmma run bit for bit (chunk-size and
//           resume invariance).
// --time 1: wall time of each prefill (after a warm-up run of the same identity). Exit 0 only if every gate passes.
// Truncated stacks (--layers < 45) need STRIX_GLM_SMOKE_HEAD=1 and a synthetic model (the engine checks).
#include "strix/device/hip_gate.hpp"
#include <strix/glm/engine.hpp>

#include "glm/common/glm_math.hpp"
#include "glm/prefill/glm_pf_path.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ge = strix::glm::engine;
namespace pf = strix::glm::pf;

namespace {

std::vector<std::uint32_t> read_u32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot read " + path);
    const auto n = static_cast<std::size_t>(f.tellg());
    if (n == 0 || n % 4) throw std::runtime_error("bad token file " + path);
    std::vector<std::uint32_t> v(n / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
    return v;
}

struct Run {
    std::uint32_t token = 0;
    std::vector<float> logits;
    std::vector<std::uint8_t> state;
    std::vector<std::uint32_t> decode;
    double ms = 0;
};

std::vector<float> logits_of(ge::Engine& e) {
    std::vector<float> l(e.vocab_slice());
    if (hipMemcpy(l.data(), e.last_logits(), l.size() * 4, hipMemcpyDeviceToHost) != hipSuccess) throw std::runtime_error("logits d2h");
    return l;
}

Run run(ge::Engine& e, const std::vector<std::uint32_t>& toks, unsigned split, unsigned decode_k) {
    Run r;
    e.reset(0);
    const auto t0 = std::chrono::steady_clock::now();
    if (split > 0 && split < toks.size()) {
        (void)e.prefill(0, std::span<const std::uint32_t>(toks.data(), split), 0);
        r.token = e.prefill(0, std::span<const std::uint32_t>(toks.data() + split, toks.size() - split), split);
    } else {
        r.token = e.prefill(0, toks, 0);
    }
    const auto t1 = std::chrono::steady_clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.logits = logits_of(e);
    r.state = e.state_bytes(0);
    std::uint32_t tok = r.token;
    for (unsigned k = 0; k < decode_k; ++k) {
        const ge::Row row{0, tok, toks.size() + k};
        tok = e.step({&row, 1})[0];
        r.decode.push_back(tok);
    }
    return r;
}

bool same(const Run& a, const Run& b, std::string& why) {
    why.clear();
    if (a.token != b.token) why += " token";
    if (a.logits.size() != b.logits.size() || std::memcmp(a.logits.data(), b.logits.data(), a.logits.size() * 4) != 0) why += " logits";
    if (a.state != b.state) why += " state";
    if (a.decode != b.decode) why += " decode";
    return why.empty();
}

void deviation(const Run& ref, const Run& x, const char* tag) {
    double se = 0, sr = 0, mx = 0;
    for (std::size_t i = 0; i < ref.logits.size(); ++i) {
        const double d = static_cast<double>(x.logits[i]) - ref.logits[i];
        se += d * d;
        sr += static_cast<double>(ref.logits[i]) * ref.logits[i];
        mx = std::fmax(mx, std::fabs(d));
    }
    std::size_t same_bytes = 0;
    for (std::size_t i = 0; i < ref.state.size() && i < x.state.size(); ++i) same_bytes += ref.state[i] == x.state[i];
    unsigned dec_eq = 0;
    while (dec_eq < ref.decode.size() && dec_eq < x.decode.size() && ref.decode[dec_eq] == x.decode[dec_eq]) ++dec_eq;
    std::printf("DEV %s vs ref: next token %s (%u vs %u); last-row logits rel-L2 %.3e, max |d| %.3e; state bytes equal %.4f %%; "
                "first %u/%zu decode tokens equal\n",
                tag, ref.token == x.token ? "same" : "DIFFERENT", x.token, ref.token, sr > 0 ? std::sqrt(se / sr) : 0.0, mx,
                ref.state.empty() ? 0.0 : 100.0 * static_cast<double>(same_bytes) / static_cast<double>(ref.state.size()), dec_eq,
                ref.decode.size());
}

}  // namespace

int main(int argc, char** argv) {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    try {
        if (argc < 2) throw std::invalid_argument("usage: strix-glm5-pf MODEL_DIR --prompt T.u32 [options]");
        ge::Options opt;
        opt.model_dir = argv[1];
        opt.world = 1;
        opt.mtp = false;
        opt.max_rows = 1024;
        std::string prompt;
        unsigned split = 0, decode_k = 8, layers = 45;
        bool timing = false;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
                return argv[++i];
            };
            if (a == "--prompt") prompt = next();
            else if (a == "--layers") layers = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--capacity") opt.capacity = std::stoull(next());
            else if (a == "--max-rows") opt.max_rows = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--split") split = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--decode") decode_k = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--time") timing = next() == "1";
            else throw std::invalid_argument("unknown option " + a);
        }
        if (prompt.empty()) throw std::invalid_argument("--prompt needed");
        opt.layers = layers;
        const auto toks = read_u32(prompt);
        if (toks.size() + decode_k > opt.capacity) throw std::invalid_argument("prompt + decode beyond --capacity");
        ge::Engine e(opt, nullptr);
        const auto lr = e.load();
        std::printf("loaded %.1f s, %.2f GiB device; prompt %zu tokens, layers %u, max_rows %u, split %u, decode %u\n", lr.seconds, lr.device_gib,
                    toks.size(), layers, opt.max_rows, split, decode_k);
        const unsigned kh = strix::glm::kKdaHeads, mh = strix::glm::kMlaHeads;
        int fails = 0;
        // ref
        e.set_prefill(ge::reference_prefill());
        if (timing) (void)run(e, toks, 0, 0);
        const Run ref = run(e, toks, 0, decode_k);
        std::printf("ref : token %u, %.1f ms (%.3f ms/token)\n", ref.token, ref.ms, ref.ms / static_cast<double>(toks.size()));
        // gemv (non-D10 proof)
        e.set_prefill(pf::make_prefill_path(kh, mh, opt.capacity, opt.max_rows, pf::TierB::gemv));
        const Run gv = run(e, toks, 0, decode_k);
        std::printf("gemv: token %u, %.1f ms (%.3f ms/token)\n", gv.token, gv.ms, gv.ms / static_cast<double>(toks.size()));
        std::string why;
        const bool g_ok = same(ref, gv, why);
        std::printf("%s chunked prefill Tier B gemv == reference prefill (token, last-row logits, all state bytes, %u decode tokens)%s\n",
                    g_ok ? "OK  " : "FAIL", decode_k, why.c_str());
        fails += !g_ok;
        if (!g_ok) {
            std::printf("pf gate: FAIL (gemv prerequisite; wmma runs skipped)\n");
            return 1;
        }
        // wmma (D10)
        e.set_prefill(pf::make_prefill_path(kh, mh, opt.capacity, opt.max_rows, pf::TierB::wmma));
        if (timing) (void)run(e, toks, 0, 0);
        const Run wm = run(e, toks, 0, decode_k);
        std::printf("wmma: token %u, %.1f ms (%.3f ms/token)\n", wm.token, wm.ms, wm.ms / static_cast<double>(toks.size()));
        deviation(ref, wm, "wmma");
        const Run w2 = run(e, toks, 0, decode_k);
        const bool det = same(wm, w2, why);
        std::printf("%s wmma two runs bitwise equal%s\n", det ? "OK  " : "FAIL", why.c_str());
        fails += !det;
        if (!det) {
            std::printf("pf gate: FAIL (wmma determinism; split run skipped)\n");
            return 1;
        }
        if (split > 0) {
            const Run ws = run(e, toks, split, decode_k);
            const bool inv = same(wm, ws, why);
            std::printf("%s wmma prompt split at %u (two prefill calls) == one call, bitwise%s\n", inv ? "OK  " : "FAIL", split, why.c_str());
            fails += !inv;
        }
        std::printf("pf gate: %s\n", fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 2;
    }
}
