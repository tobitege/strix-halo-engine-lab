// strix-hip-qwen35-spec: exact block drafting (shared strix::spec BlockDrafter + Arbiter) over the qwen35 HIP window engine.
// Modes: default = serial reference (--check 1) + speculative run, token-for-token gate (EXACT: PASS/FAIL), E mean/p50/p90,
// avg + best-1s tok/s; --cost-table = verify window ms per rows 1..16 (commit full / half); --prefill-bench N [--pf-sample] =
// prefill tok/s over N synthetic ids in 16-row windows (full run, or sampled windows at positions spread over N).
// Sources: copy (n-gram / prompt lookup); with --mtp F.gguf also chain + hybrid (native MTP head from a side GGUF: blk.<layers>.*,
// bf16 through the shared dense dispatch). --chain-hidden post|pre, --chain-k N; --mtp-ref = mtp_forward vs a host reference.
// Home: specific = app glue; drafting logic, VerifyTarget binding and the MTP chain glue are shared (strix/spec/*.hpp), engine is HipEngine.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/qwen35/hip_engine.hpp"
#include "strix/spec/block_drafter.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/spec/engine_target.hpp"
#include "strix/spec/mtp_chain.hpp"
#include "../src/core/dense_linear.hpp"

namespace sp = strix::spec;
namespace q = strix::qwen35;
struct HipChainDev {
    static float* alloc(q::HipEngine& e, std::size_t n) { return e.alloc_f32(n); }
    static void release(q::HipEngine& e, float* p) { e.free_f32(p); }
    static void copy(q::HipEngine& e, float* d, const float* s, std::size_t n) { e.copy_f32_async(d, s, n); }
};
using HipMtpChain = sp::MtpChain<q::HipEngine, HipChainDev>;
using Clock = std::chrono::steady_clock;

namespace {
double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }
std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}
std::vector<std::uint32_t> parse_ids(const std::string& text) {
    std::vector<std::uint32_t> v;
    std::string t;
    for (const char c : text + ",") {
        if (c >= '0' && c <= '9') t += c;
        else if (!t.empty()) { v.push_back(strix::core::parse_u32(t)); t.clear(); }
    }
    return v;
}
double quantile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min<std::size_t>(v.size() - 1, std::size_t(p * double(v.size())))];
}
// ---- host reference of the MTP layer (fp64 accumulation, bf16-rounded KV as the device cache) -------------------------------
struct HostT {
    std::vector<std::uint8_t> raw;
    unsigned in = 0, out = 0, fmt = 0;
};
float h2f(std::uint16_t h) {
    const unsigned e = (h >> 10) & 31U, m = h & 1023U;
    float v = e == 0 ? std::ldexp(float(m), -24) : std::ldexp(float(m | 1024U), int(e) - 25);
    return (h & 0x8000U) ? -v : v;
}
float bf2f(std::uint16_t b) { const std::uint32_t u = std::uint32_t(b) << 16U; float f; std::memcpy(&f, &u, 4); return f; }
std::uint16_t f2bf(float f) {
    std::uint32_t u; std::memcpy(&u, &f, 4);
    return std::uint16_t((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U);
}
HostT host_tensor(const std::string& file, const std::string& name) {
    const auto info = strix::gguf::read_file(file);
    const strix::gguf::TensorInfo* t = nullptr;
    for (const auto& x : info.tensors)
        if (x.name == name + ".weight" || x.name == name) t = &x;
    if (!t) throw std::runtime_error("host ref: missing " + name);
    HostT r;
    r.fmt = t->ggml_type; r.in = unsigned(t->dimensions.at(0)); r.out = t->dimensions.size() > 1 ? unsigned(t->dimensions[1]) : 1U;
    r.raw.resize(t->byte_size);
    std::ifstream f(file, std::ios::binary);
    f.seekg(std::streamoff(t->absolute_offset));
    f.read(reinterpret_cast<char*>(r.raw.data()), std::streamsize(r.raw.size()));
    if (!f) throw std::runtime_error("host ref: short read " + name);
    return r;
}
void host_row(const HostT& t, std::size_t row, std::vector<float>& out) {  // dequantized row
    out.resize(t.in);
    if (t.fmt == 0U) std::memcpy(out.data(), t.raw.data() + row * t.in * 4U, t.in * 4U);
    else if (t.fmt == 30U) for (unsigned i = 0; i < t.in; ++i) { std::uint16_t b; std::memcpy(&b, &t.raw[(row * t.in + i) * 2U], 2); out[i] = bf2f(b); }
    else if (t.fmt == 8U) {
        for (unsigned i = 0; i < t.in; ++i) {
            const std::uint8_t* b = &t.raw[(row * (t.in / 32U) + i / 32U) * 34U];
            std::uint16_t h; std::memcpy(&h, b, 2);
            out[i] = h2f(h) * float(static_cast<signed char>(b[2 + i % 32U]));
        }
    } else throw std::runtime_error("host ref: unsupported fmt");
}
// y[o] = row o of t . x (fp64 accumulation), threads over outputs
void host_matvec(const HostT& t, const float* x, float* y) {
    const unsigned nt = std::max(1U, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (unsigned w = 0; w < nt; ++w)
        th.emplace_back([&, w] {
            std::vector<float> row;
            for (unsigned o = w; o < t.out; o += nt) {
                host_row(t, o, row);
                double s = 0;
                for (unsigned i = 0; i < t.in; ++i) s += double(row[i]) * double(x[i]);
                y[o] = float(s);
            }
        });
    for (auto& x_ : th) x_.join();
}
void host_rms(const float* x, const std::vector<float>& g, unsigned n, float eps, float* y) {
    double ss = 0;
    for (unsigned i = 0; i < n; ++i) ss += double(x[i]) * double(x[i]);
    const float r = 1.0F / std::sqrt(float(ss / n) + eps);
    for (unsigned i = 0; i < n; ++i) y[i] = (x[i] * r) * g[i];
}
std::vector<float> host_f32(const std::string& file, const std::string& name) {
    const HostT t = host_tensor(file, name);
    std::vector<float> v;
    host_row(t, 0, v);
    return v;
}

// mtp_forward on rows [pos0=0 .. R) vs the host reference. Prints max|d| over logits, top-1 agreement.
int run_mtp_ref(q::HipEngine& E, const std::string& model, const std::string& mtp_file, const std::vector<std::uint32_t>& prompt) {
    const auto& c = E.config();
    const unsigned H = c.hidden, D = c.head_dim, NH = c.heads, NK = c.kv_heads, I = c.intermediate, V = c.vocab;
    const unsigned R = std::min<unsigned>(4U, unsigned(prompt.size()) - 1U);
    E.reset();
    const auto o = E.forward_commit(prompt.data(), R, q::LogitRows::last);
    E.sync();
    std::vector<float> hid(std::size_t(R) * H);
    if (hipMemcpy(hid.data(), o.hidden_post, hid.size() * 4U, hipMemcpyDeviceToHost) != hipSuccess) throw std::runtime_error("hidden copy");
    const auto m = E.mtp_forward(prompt.data() + 1, o.hidden_post, R, 0, true);
    q::HipRowsOut ro;
    ro.logits = m.logits;
    ro.top = m.top;
    std::vector<float> dev_logits(std::size_t(R) * V);
    E.read_logits(ro, R, dev_logits.data());
    std::vector<std::uint32_t> dev_top(R);
    E.read_top(ro, R, dev_top.data());

    const std::string b = "blk." + std::to_string(c.layers) + ".";
    const auto enorm = host_f32(mtp_file, b + "nextn.enorm"), hnorm = host_f32(mtp_file, b + "nextn.hnorm"),
               shnorm = host_f32(mtp_file, b + "nextn.shared_head_norm"), an = host_f32(mtp_file, b + "attn_norm"),
               pn = host_f32(mtp_file, b + "post_attention_norm"), qn = host_f32(mtp_file, b + "attn_q_norm"), kn = host_f32(mtp_file, b + "attn_k_norm");
    const HostT eh = host_tensor(mtp_file, b + "nextn.eh_proj"), wq = host_tensor(mtp_file, b + "attn_q"), wk = host_tensor(mtp_file, b + "attn_k"),
                wv = host_tensor(mtp_file, b + "attn_v"), wo = host_tensor(mtp_file, b + "attn_output"), wg = host_tensor(mtp_file, b + "ffn_gate"),
                wu = host_tensor(mtp_file, b + "ffn_up"), wd = host_tensor(mtp_file, b + "ffn_down"), emb = host_tensor(model, "token_embd"),
                head = host_tensor(model, "output");
    std::vector<std::vector<float>> kc(R, std::vector<float>(NK * D)), vc(R, std::vector<float>(NK * D));
    double max_logit = 0, max_ref = 0;
    unsigned top_ok = 0;
    for (unsigned r = 0; r < R; ++r) {
        std::vector<float> e, cat(2U * H), mx(H), xn(H);
        host_row(emb, prompt[1 + r], e);
        host_rms(e.data(), enorm, H, c.rms_eps, cat.data());
        host_rms(&hid[std::size_t(r) * H], hnorm, H, c.rms_eps, cat.data() + H);
        host_matvec(eh, cat.data(), mx.data());
        host_rms(mx.data(), an, H, c.rms_eps, xn.data());
        std::vector<float> q(NH * 2U * D), k(NK * D), v(NK * D);
        host_matvec(wq, xn.data(), q.data());
        host_matvec(wk, xn.data(), k.data());
        host_matvec(wv, xn.data(), v.data());
        const unsigned half = c.rotary_dim / 2U;
        auto rope = [&](std::vector<float>& vec, const float* src, const std::vector<float>& gain) {
            std::vector<float> n(D);
            host_rms(src, gain, D, c.rms_eps, n.data());
            for (unsigned t = 0; t < D; ++t) {
                float y = n[t];
                if (t < c.rotary_dim) {
                    const double ang = double(r) * std::pow(c.rope_theta, -2.0 * double(t % half) / double(c.rotary_dim));
                    const float cs = float(std::cos(ang)), sn = float(std::sin(ang));
                    y = t < half ? n[t] * cs - n[t + half] * sn : n[t] * cs + n[t - half] * sn;
                }
                vec[t] = y;
            }
        };
        std::vector<float> ctx(NH * D);
        for (unsigned kh = 0; kh < NK; ++kh) {
            std::vector<float> kr(D);
            rope(kr, &k[kh * D], kn);
            for (unsigned t = 0; t < D; ++t) { kc[r][kh * D + t] = bf2f(f2bf(kr[t])); vc[r][kh * D + t] = bf2f(f2bf(v[kh * D + t])); }
        }
        for (unsigned h = 0; h < NH; ++h) {
            std::vector<float> qr(D);
            rope(qr, &q[h * 2U * D], qn);
            const unsigned kh = h / (NH / NK);
            std::vector<double> sc(r + 1U);
            double mxs = -1e30;
            for (unsigned p = 0; p <= r; ++p) {
                double s = 0;
                for (unsigned t = 0; t < D; ++t) s += double(qr[t]) * double(kc[p][kh * D + t]);
                sc[p] = s / std::sqrt(double(D));
                mxs = std::max(mxs, sc[p]);
            }
            double den = 0;
            for (auto& s : sc) { s = std::exp(s - mxs); den += s; }
            for (unsigned t = 0; t < D; ++t) {
                double acc = 0;
                for (unsigned p = 0; p <= r; ++p) acc += sc[p] * double(vc[p][kh * D + t]);
                const float g = q[h * 2U * D + D + t];
                ctx[h * D + t] = float(acc / den) * (1.0F / (1.0F + std::exp(-g)));
            }
        }
        std::vector<float> proj(H);
        host_matvec(wo, ctx.data(), proj.data());
        for (unsigned i = 0; i < H; ++i) mx[i] += proj[i];
        host_rms(mx.data(), pn, H, c.rms_eps, xn.data());
        std::vector<float> g(I), u(I), act(I);
        host_matvec(wg, xn.data(), g.data());
        host_matvec(wu, xn.data(), u.data());
        for (unsigned i = 0; i < I; ++i) act[i] = (g[i] / (1.0F + std::exp(-g[i]))) * u[i];
        host_matvec(wd, act.data(), proj.data());
        for (unsigned i = 0; i < H; ++i) mx[i] += proj[i];
        host_rms(mx.data(), shnorm, H, c.rms_eps, xn.data());
        std::vector<float> lg(V);
        host_matvec(head, xn.data(), lg.data());
        double md = 0, mr = 0;
        unsigned top = 0;
        for (unsigned i = 0; i < V; ++i) {
            md = std::max(md, double(std::fabs(lg[i] - dev_logits[std::size_t(r) * V + i])));
            mr = std::max(mr, double(std::fabs(lg[i])));
            if (lg[i] > lg[top]) top = i;
        }
        std::printf("mtp-ref row %u (pos %u, token %u): max|d logits| %.5f (max|ref| %.3f), top-1 host %u device %u %s\n", r, r, prompt[1 + r], md, mr,
                    top, dev_top[r], top == dev_top[r] ? "EQUAL" : "DIFFER");
        max_logit = std::max(max_logit, md);
        max_ref = std::max(max_ref, mr);
        top_ok += top == dev_top[r];
    }
    const bool pass = top_ok == R && max_logit <= 2e-2 * std::max(1.0, max_ref);
    std::printf("MTP-REF: %s (rows %u, max|d logits| %.5f, top-1 equal %u/%u)\n", pass ? "PASS" : "FAIL", R, max_logit, top_ok, R);
    return pass ? 0 : 3;
}
}  // namespace

int main(int argc, char** argv) try {
    std::string model, ids_file, label = "run", sources, mtp_file, chain_hidden = "post", dump_ref, dump_out;
    std::uint32_t n_gen = 256, max_block = 16, max_pos = 0, pf_bench = 0, trace_n = 0, prefill_rows = 16, chain_k = 7, ref_n = 0;
    bool check = true, cost_table = false, pf_sample = false, seed_costs = false, pf_twice = false, mtp_ref = false;
    std::uint32_t copy_arm = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--prompt-file") ids_file = next();
        else if (a == "--n-gen") n_gen = strix::core::parse_u32(next());
        else if (a == "--sources") sources = next();
        else if (a == "--max-block") max_block = strix::core::parse_u32(next());
        else if (a == "--max-pos") max_pos = strix::core::parse_u32(next());
        else if (a == "--check") check = next() != "0";
        else if (a == "--cost-table") cost_table = true;
        else if (a == "--prefill-bench") pf_bench = strix::core::parse_u32(next());
        else if (a == "--pf-sample") pf_sample = true;
        else if (a == "--pf-twice") pf_twice = true;  // two full prefills: cold first, warm last
        else if (a == "--prefill-rows") prefill_rows = strix::core::parse_u32(next());  // >16 = FAST wide windows (dense-q8a8-f16-v1)
        else if (a == "--seed-costs") seed_costs = next() != "0";
        else if (a == "--trace") trace_n = strix::core::parse_u32(next());
        else if (a == "--label") label = next();
        else if (a == "--mtp") mtp_file = next();                  // side GGUF with blk.<layers>.* (MTP head)
        else if (a == "--chain-hidden") chain_hidden = next();      // post | pre (hidden fed to the MTP layer)
        else if (a == "--chain-k") chain_k = strix::core::parse_u32(next());
        else if (a == "--copy-arm") copy_arm = strix::core::parse_u32(next());  // also run the copy-only baseline for N tokens
        else if (a == "--mtp-ref") mtp_ref = true;                  // mtp_forward vs host reference
        else if (a == "--ref-n") ref_n = strix::core::parse_u32(next());  // serial reference length (default n-gen): the first ref-n ids are compared
        else if (a == "--dump-ref") dump_ref = next();              // serial greedy ids -> file (for cmp)
        else if (a == "--dump-out") dump_out = next();              // speculative emitted ids -> file
        else throw std::invalid_argument("unknown argument " + a);
    }
    if (sources.empty()) sources = mtp_file.empty() ? "copy" : "copy,chain,hybrid";
    if (chain_hidden != "post" && chain_hidden != "pre") throw std::invalid_argument("--chain-hidden post|pre");
    if (model.empty() || (ids_file.empty() && !cost_table && !pf_bench))
        throw std::invalid_argument("usage: --model F.gguf (--prompt-file ids | --cost-table | --prefill-bench N [--pf-sample]) [--n-gen N] "
                                    "[--sources copy,chain,hybrid|none] [--mtp F.gguf] [--chain-hidden post|pre] [--chain-k N] [--mtp-ref] [--dump-ref F] [--dump-out F] "
                                    "[--max-block 1..16] [--check 0|1] [--max-pos N] [--label S]");
    std::vector<std::uint32_t> prompt;
    if (!ids_file.empty()) {
        std::ifstream f(ids_file);
        std::stringstream ss;
        ss << f.rdbuf();
        prompt = parse_ids(ss.str());
    }
    q::HipEngineOptions opt;
    opt.max_positions = std::max<std::uint32_t>(max_pos, std::uint32_t(prompt.size()) + n_gen + 128U + (cost_table ? 512U : 0U));
    if (pf_bench) opt.max_positions = std::max(opt.max_positions, pf_bench + 64U);
    opt.quiet = true;
    opt.prefill_rows = prefill_rows;
    auto E = q::HipEngine::from_gguf(model, opt);
    if (!mtp_file.empty()) E->load_mtp(mtp_file);
    const auto vocab = E->config().vocab;
    std::unique_ptr<HipMtpChain> chain_glue;
    if (E->has_mtp()) chain_glue = std::make_unique<HipMtpChain>(*E, chain_hidden == "pre", q::HipEngine::mtp_max_rows());
    std::printf("loaded: weights %.2f GB/pass, state %.0f MB/pass, max_positions %u\n", double(E->weight_bytes()) / 1e9,
                double(E->state_bytes_per_pass()) / 1e6, opt.max_positions);

    auto prefill_ids = [&](const std::vector<std::uint32_t>& ids, std::uint32_t& anchor, bool feed_chain = false) {
        E->reset();
        if (chain_glue) chain_glue->reset();
        const std::size_t W = E->prefill_rows();
        for (std::size_t p = 0; p < ids.size(); p += W) {
            const auto rows = std::uint32_t(std::min<std::size_t>(W, ids.size() - p));
            const bool last = p + rows == ids.size();
            const auto o = E->forward_commit(ids.data() + p, rows, last ? q::LogitRows::last : q::LogitRows::none);
            if (last) E->read_top(o, 1, &anchor);
            if (feed_chain && chain_glue) chain_glue->observe_prefill(o, rows, ids.data() + p, last ? anchor : ids[p + rows]);
        }
        E->sync();
    };
    auto warm = [&]() {  // every window width once (commit 0 discards): kernels / caches hot before any timed wall
        std::vector<std::uint32_t> t(q::HipEngine::max_rows(), 1);
        for (std::uint32_t r = 1; r <= q::HipEngine::max_rows(); ++r) {
            E->forward_rows(t.data(), r, q::LogitRows::all);
            E->commit(0);
        }
        E->sync();
    };

    if (mtp_ref) {
        if (!E->has_mtp() || prompt.size() < 3) throw std::invalid_argument("--mtp-ref needs --mtp F.gguf and --prompt-file with >= 3 ids");
        return run_mtp_ref(*E, model, mtp_file, prompt);
    }

    if (pf_bench) {
        std::vector<std::uint32_t> ids(pf_bench);
        for (std::uint32_t i = 0; i < pf_bench; ++i) ids[i] = std::uint32_t(mix(77 + i) % vocab);
        const std::uint32_t R = E->prefill_rows();
        if (pf_twice) {
            std::uint32_t anchor = 0;
            for (int pass = 0; pass < 2; ++pass) {
                const double t0 = now_ms();
                prefill_ids(ids, anchor);
                const double ms = now_ms() - t0;
                std::printf("prefill[%s] %s: %u tokens in %.1f ms = %.1f tok/s (window %u, anchor %u)\n", label.c_str(), pass ? "WARM" : "COLD",
                            pf_bench, ms, 1000.0 * pf_bench / ms, E->prefill_rows(), anchor);
            }
        } else if (!pf_sample || pf_bench < R) {
            std::uint32_t anchor = 0;
            std::vector<std::uint32_t> warm_ids(ids.begin(), ids.begin() + std::min<std::size_t>(R, ids.size()));
            prefill_ids(warm_ids, anchor);
            const double t0 = now_ms();
            prefill_ids(ids, anchor);
            const double ms = now_ms() - t0;
            std::printf("prefill[%s]: %u tokens in %.1f ms = %.1f tok/s (%.2f ms/window of %u)\n", label.c_str(), pf_bench, ms,
                        1000.0 * pf_bench / ms, ms / (double(pf_bench) / R), R);
        } else {
            // Windows at positions spread over [0, N): window cost is linear in position (attention), so trapezoid over samples.
            std::uint32_t anchor = 0;
            std::vector<std::uint32_t> warm_ids(ids.begin(), ids.begin() + std::min<std::size_t>(R, ids.size()));
            prefill_ids(warm_ids, anchor);
            const unsigned S = 9;
            std::vector<double> pos, cost;
            for (unsigned s = 0; s < S; ++s) {
                const std::uint32_t at = std::uint32_t((std::uint64_t(pf_bench - R) * s / (S - 1)) / R * R);
                E->bench_seek(at);
                double sum = 0;
                for (int rep = 0; rep < 4; ++rep) {
                    const double t0 = now_ms();
                    E->forward_commit(ids.data(), R, q::LogitRows::none);
                    E->sync();
                    sum += now_ms() - t0;
                    E->bench_seek(at);
                }
                pos.push_back(at);
                cost.push_back(sum / 4);
                std::printf("window pos %u: %.2f ms\n", at, sum / 4);
            }
            double total = 0;
            for (unsigned s = 0; s + 1 < S; ++s) total += 0.5 * (cost[s] + cost[s + 1]) * (pos[s + 1] - pos[s]) / R;
            total += cost[S - 1] * (double(pf_bench) - pos[S - 1]) / R;
            std::printf("prefill[%s] SAMPLED: %u tokens ~ %.0f ms = %.1f tok/s (9 positions x 4 windows, trapezoid)\n", label.c_str(), pf_bench,
                        total, 1000.0 * pf_bench / total);
        }
        return 0;
    }

    if (cost_table) {
        std::uint32_t anchor = 0;
        std::vector<std::uint32_t> ids(256);
        for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = std::uint32_t(mix(5 + i) % vocab);
        prefill_ids(ids, anchor);
        warm();
        std::vector<std::uint32_t> t(q::HipEngine::max_rows(), anchor);
        std::printf("cost table (%s, ctx %zu+)\nrows verify_ms(commit full) verify_ms(commit half) tok_per_s_if_all_accepted\n", label.c_str(), ids.size());
        for (const std::uint32_t r : {1U, 2U, 3U, 4U, 6U, 8U, 12U, 16U}) {
            double full = 0, half = 0;
            const int reps = 6;
            for (int rep = 0; rep < 2 * reps; ++rep) {
                const bool h = rep & 1;
                const double t0 = now_ms();
                E->forward_rows(t.data(), r, q::LogitRows::all);
                E->commit(h ? std::max(1U, r / 2) : r);
                E->sync();
                (h ? half : full) += now_ms() - t0;
            }
            full /= reps;
            half /= reps;
            std::printf("%u %.3f %.3f %.1f\n", r, full, half, 1000.0 * r / full);
        }
        double none = 0, lst = 0, ser = 0;
        for (int rep = 0; rep < 12; ++rep) {
            const bool l = rep & 1;
            const double t0 = now_ms();
            E->forward_rows(t.data(), 1, l ? q::LogitRows::last : q::LogitRows::none);
            E->commit(0);
            E->sync();
            (l ? lst : none) += now_ms() - t0;
        }
        for (int rep = 0; rep < 6; ++rep) {
            const double t0 = now_ms();
            E->forward_commit(t.data(), 1, q::LogitRows::last);
            E->sync();
            ser += now_ms() - t0;
        }
        std::printf("R=1 window: no-logits %.3f ms, last %.3f ms -> lm_head %.3f ms; forward_commit(1) (no snapshot) %.3f ms\n", none / 6, lst / 6,
                    (lst - none) / 6, ser / 6);
        return 0;
    }

    // ---- exactness reference: serial greedy
    if (ref_n == 0 || ref_n > n_gen) ref_n = n_gen;
    std::uint32_t anchor = 0;
    std::vector<std::uint32_t> ref;
    double serial_ms = 0;
    if (check) {
        prefill_ids(prompt, anchor);
        warm();
        ref.push_back(anchor);
        std::uint32_t cur = anchor;
        const double t0 = now_ms();
        while (ref.size() < ref_n) {
            const auto o = E->forward_commit(&cur, 1, q::LogitRows::last);
            E->read_top(o, 1, &cur);
            ref.push_back(cur);
        }
        serial_ms = now_ms() - t0;
        std::printf("serial: %u tokens in %.1f ms = %.2f tok/s (%.3f ms/token)\n", ref_n, serial_ms, 1000.0 * (ref_n - 1) / serial_ms,
                    serial_ms / (ref_n - 1));
    }

    auto spec_run = [&](bool use_chain, std::uint32_t n_run, std::size_t& bad_out, std::vector<std::uint32_t>& out, double& wall_out) {
        const double tp0 = now_ms();
        prefill_ids(prompt, anchor, use_chain);
        const double prefill_ms = now_ms() - tp0;
        warm();
        sp::BlockConfig bc;
        bc.max_block = max_block;
        bc.copy = sources.find("copy") != std::string::npos || !use_chain;
        bc.chain = use_chain && chain_glue && sources.find("chain") != std::string::npos;
        bc.hybrid = use_chain && chain_glue && sources.find("hybrid") != std::string::npos;
        bc.chain_max_k = chain_k;
        if (sources == "none" && use_chain) bc.copy = bc.chain = bc.hybrid = false;
        sp::BlockDrafter drafter(bc, use_chain ? chain_glue.get() : nullptr);
        if (seed_costs) {
            std::vector<std::uint32_t> t(q::HipEngine::max_rows(), anchor);
            for (const std::uint32_t r : {1U, 2U, 4U, 8U, 16U}) {
                double sum = 0;
                for (int rep = 0; rep < 3; ++rep) {
                    const double t0 = now_ms();
                    E->forward_rows(t.data(), r, q::LogitRows::all);
                    E->commit(0);
                    E->sync();
                    sum += now_ms() - t0;
                }
                drafter.seed_verify_ms(r, sum / 3);
            }
        }
        drafter.commit(prompt);
        out.assign(1, anchor);
        drafter.commit(std::span<const std::uint32_t>(out.data(), 1));
        sp::EngineVerifyTarget<q::HipEngine> target(*E);
        if (use_chain && chain_glue) target.set_observer([&](const q::HipRowsOut& o, std::uint32_t rows) { chain_glue->observe_verify(o, rows); });
        sp::BlockSpeculator spec(drafter, target);
        spec.start(anchor);
        std::vector<double> tick_ms{0.0}, per_step;
        std::vector<std::uint32_t> cum{1};
        std::uint64_t windows[17] = {};
        std::uint64_t by_source[3] = {0, 0, 0};
        const double t0 = now_ms();
        while (out.size() < n_run) {
            sp::StepParams p;
            p.remaining = n_run - std::uint32_t(out.size());
            p.max_rows = std::min<std::uint32_t>(16U, max_block);
            p.context = E->committed();
            const auto r = spec.step(out, p);
            if (r.emitted == 0) break;
            if (per_step.size() < trace_n)
                std::printf("T%zu src%u rows%u prop%u acc%u em%u ver%.2f dr%.2f\n", per_step.size(), unsigned(r.source), r.rows, r.proposed,
                            r.accepted, r.emitted, r.verify_ms, r.draft_ms);
            ++by_source[std::size_t(r.source)];
            if (r.rows <= 16U) ++windows[r.rows];
            per_step.push_back(double(r.emitted));
            tick_ms.push_back(now_ms() - t0);
            cum.push_back(std::uint32_t(out.size()));
        }
        const double wall = now_ms() - t0;
        std::size_t bad = (out.size() != n_run || (!ref.empty() && n_run == n_gen && ref.size() != n_gen)) ? 1U : 0U;
        for (std::size_t i = 0; i < out.size() && i < ref.size(); ++i)
            if (out[i] != ref[i]) { ++bad; if (bad == 1) std::fprintf(stderr, "MISMATCH at token %zu: spec %u serial %u\n", i, out[i], ref[i]); }
        double best = 0;
        for (std::size_t i = 0; i < tick_ms.size(); ++i) {
            std::size_t j = i;
            while (j + 1 < tick_ms.size() && tick_ms[j] - tick_ms[i] < 1000.0) ++j;
            const double dt = tick_ms[j] - tick_ms[i];
            if (dt >= 900.0) best = std::max(best, 1000.0 * double(cum[j] - cum[i]) / dt);
        }
        const double mean_e = per_step.empty() ? 0.0 : double(out.size() - 1) / double(per_step.size());
        std::printf("spec[%s%s]: %zu tokens, %zu steps (serial %llu copy %llu chain %llu), E mean %.2f p50 %.0f p90 %.0f, avg %.2f tok/s "
                    "(%.3f ms/step), best-1s %.2f tok/s, prompt %zu tok prefill %.0f ms\n",
                    label.c_str(), use_chain ? "" : "/copy-only", out.size(), per_step.size(), (unsigned long long)by_source[0], (unsigned long long)by_source[1],
                    (unsigned long long)by_source[2], mean_e, quantile(per_step, 0.5), quantile(per_step, 0.9),
                    1000.0 * double(out.size() - 1) / wall, wall / double(std::max<std::size_t>(per_step.size(), 1)), best, prompt.size(), prefill_ms);
        for (unsigned r = 1; r <= 16U; ++r)
            if (windows[r]) std::printf("verify-window %u: %llu calls\n", r, (unsigned long long)windows[r]);
        if (use_chain && chain_glue)
            std::printf("mtp[%s]: hidden %s chain_k %u, mtp_forward calls %llu rows %llu, draft() %llu calls %.1f ms total = %.2f ms/draft, %.2f ms/mtp_forward call; "
                        "chain-source passes %llu (%.1f %%), hybrid blocks %llu (ext proposed %llu accepted %llu)\n",
                        label.c_str(), chain_hidden.c_str(), chain_k, (unsigned long long)chain_glue->calls(), (unsigned long long)chain_glue->rows(),
                        (unsigned long long)chain_glue->drafts(), chain_glue->draft_wall_ms(),
                        chain_glue->draft_wall_ms() / double(std::max<std::uint64_t>(chain_glue->drafts(), 1)),
                        chain_glue->draft_wall_ms() / double(std::max<std::uint64_t>(chain_glue->calls(), 1)), (unsigned long long)by_source[2],
                        100.0 * double(by_source[2]) / double(std::max<std::size_t>(per_step.size(), 1)), (unsigned long long)drafter.hybrid_blocks,
                        (unsigned long long)drafter.hybrid_ext_proposed, (unsigned long long)drafter.hybrid_ext_accepted);
        bad_out = bad;
        wall_out = wall;
    };
    auto dump = [](const std::string& f, const std::vector<std::uint32_t>& v, std::size_t n) {
        if (f.empty()) return;
        std::ofstream o(f);
        for (std::size_t i = 0; i < std::min(n, v.size()); ++i) o << v[i] << "\n";
    };
    std::size_t bad = 0;
    double wall = 0;
    std::vector<std::uint32_t> out;
    spec_run(true, n_gen, bad, out, wall);
    dump(dump_ref, ref, ref.size());
    dump(dump_out, out, ref.size());
    if (!ref.empty()) {
        std::printf("EXACT: %s (%zu tokens compared, %zu mismatches)\n", bad == 0 ? "PASS" : "FAIL", std::min(out.size(), ref.size()), bad);
        std::printf("speedup vs serial: %.2fx\n", serial_ms / double(ref_n - 1) / (wall / double(out.size() - 1)));
    }
    if (copy_arm) {  // copy-only baseline on the same prompt (no MTP feed), first copy_arm tokens
        std::size_t bad2 = 0;
        double wall2 = 0;
        std::vector<std::uint32_t> out2;
        spec_run(false, copy_arm, bad2, out2, wall2);
        bad += bad2;
    }
    return bad == 0 ? 0 : 2;
} catch (const std::exception& e) {
    std::fprintf(stderr, "strix-hip-qwen35-spec: %s\n", e.what());
    return 1;
}
