// Gate of the DeepSeek V4.1 image tower on the real vision file (not a CTest test: it needs the admitted file and
// the dumps of tools/vision/ds41_ref.py).
//
//   strix-deepseek41-vision-gate --model FILE.gguf --ref DIR [--backend host|hip] [--only a,b]
//                                [--host-max-patches N] [--split | --no-split] [--unpinned] [--diagnose a,b]
//
// Per case of DIR/cases.txt:
//   preprocess  PNG -> adapter.patches() against the reference patches (max |diff| <= 1e-6, which implies equal
//               resized pixels: one 8-bit step is 7.8e-3 in normalised units);
//   encoder     reference patches in -> tower -> projector, per-row cosine and relative L2 of the image rows against
//               the reference rows; the special rows must be the file's tensors bit for bit.
// Thresholds: host FP32 backend cosine >= 0.99999 (graph); device backend mean >= 0.9995, min >= 0.998, relative
// L2 <= 2 % (arithmetic class FAST). The device backend also prints the per-layer time receipt.
#include "../src/deepseek41/vision_adapter.hpp"

#include "strix/deepseek41_model.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/vision/compare.hpp"
#include "strix/vision/encoder.hpp"
#include "strix/vision/gguf_weights.hpp"
#include "strix/vision/image.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace strix;
namespace ds = strix::deepseek41;
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

template <class T>
std::vector<T> read_raw(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.filename().string());
    in.seekg(0, std::ios::end);
    const auto bytes = static_cast<std::size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::vector<T> out(bytes / sizeof(T));
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(T)));
    if (bytes % sizeof(T) != 0 || static_cast<std::size_t>(in.gcount()) != bytes) throw std::runtime_error("short read of " + path.filename().string());
    return out;
}

struct Case {
    std::string name;
    int width = 0, height = 0, resized_w = 0, resized_h = 0;
    unsigned grid_h = 0, grid_w = 0, span_rows = 0;
    bool graph_only = false;
};

std::vector<Case> read_cases(const std::filesystem::path& dir) {
    std::ifstream in(dir / "cases.txt");
    if (!in) throw std::runtime_error("cannot open cases.txt");
    std::vector<Case> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        Case c;
        int graph = 0;
        row >> c.name >> c.width >> c.height >> c.resized_w >> c.resized_h >> c.grid_h >> c.grid_w >> c.span_rows >> graph;
        if (!row) throw std::runtime_error("bad cases.txt line: " + line);
        c.graph_only = graph != 0;
        out.push_back(c);
    }
    return out;
}

struct Match {
    double cos_mean = 0.0, cos_min = 1.0, rel_l2 = 0.0;
    unsigned worst_row = 0;
};
// Rows `pick` of got and want ([rows][width]).
Match compare(const float* got, const float* want, std::size_t width, const std::vector<unsigned>& pick) {
    Match m;
    double err = 0.0, norm = 0.0;
    for (const unsigned r : pick) {
        double ab = 0.0, aa = 0.0, bb = 0.0;
        for (std::size_t i = 0; i < width; ++i) {
            const double a = static_cast<double>(got[r * width + i]), b = static_cast<double>(want[r * width + i]);
            ab += a * b;
            aa += a * a;
            bb += b * b;
            err += (a - b) * (a - b);
        }
        norm += bb;
        const double c = ab / std::sqrt(aa * bb + 1e-300);
        m.cos_mean += c;
        if (c < m.cos_min) {
            m.cos_min = c;
            m.worst_row = r;
        }
    }
    m.cos_mean /= static_cast<double>(std::max<std::size_t>(pick.size(), 1));
    m.rel_l2 = std::sqrt(err / (norm + 1e-300));
    return m;
}

bool listed(const std::string& only, const std::string& name) {
    if (only.empty()) return true;
    std::istringstream in(only);
    std::string item;
    while (std::getline(in, item, ','))
        if (item == name) return true;
    return false;
}

#ifdef STRIX_DS41_VISION_GATE_HIP
// Where a device error enters: the residual stream after selected blocks, host FP32 against the device with and
// without split activations, and the rows that end worst. A diagnostic, not a gate.
int diagnose(const std::string& names, const std::filesystem::path& dir, const ds::VisionAdapter& adapter,
             vision::EncoderConfig cfg, const vision::EncoderWeights& weights) {
    cfg.deepstack_taps = {0U, 1U, 3U, 5U, 7U, 11U, 15U, 19U, 23U, 27U, 29U, 31U};
    std::erase_if(cfg.deepstack_taps, [&](unsigned t) { return t >= cfg.layers; });
    for (const Case& c : read_cases(dir)) {
        if (!listed(names, c.name)) continue;
        vision::PatchGrid g;
        g.grid_h = c.grid_h;
        g.grid_w = c.grid_w;
        g.rows = c.grid_h * c.grid_w;
        g.patch_dim = cfg.patch_dim();
        g.data = read_raw<float>(dir / (c.name + ".patches.f32"));
        g.y.resize(g.rows);
        g.x.resize(g.rows);
        std::vector<unsigned> all(g.rows);
        for (unsigned r = 0; r < g.rows; ++r) {
            g.y[r] = static_cast<std::uint16_t>(r / c.grid_w);
            g.x[r] = static_cast<std::uint16_t>(r % c.grid_w);
            all[r] = r;
        }
        const vision::PositionInputs pos = adapter.positions(cfg, weights, g);
        auto t0 = Clock::now();
        const vision::EncoderOutput want = vision::ReferenceEncoder(cfg, weights).encode(g, pos);
        std::printf("diagnose %s: %u patches, host reference %.1f s\n", c.name.c_str(), g.rows, ms_since(t0) / 1000.0);
        const std::size_t W = cfg.width;
        const auto norm = [&](const std::vector<float>& h, unsigned r) {
            double s = 0.0;
            for (std::size_t i = 0; i < W; ++i) s += static_cast<double>(h[r * W + i]) * static_cast<double>(h[r * W + i]);
            return std::sqrt(s);
        };
        for (const bool split : {false, true}) {
            vision::EncoderOptions opt;
            opt.split_activations = split;
            vision::Encoder enc(cfg, opt);
            enc.load(weights);
            const vision::EncoderOutput got = enc.encode(g, pos);
            std::printf(" split %d: after block | cos mean | cos min | relL2 | rows < 0.998 | ref row norm median / max\n", split ? 1 : 0);
            for (std::size_t t = 0; t < cfg.deepstack_taps.size(); ++t) {
                const Match m = compare(got.taps[t].data(), want.taps[t].data(), W, all);
                unsigned bad = 0;
                std::vector<double> norms(g.rows);
                for (unsigned r = 0; r < g.rows; ++r) {
                    norms[r] = norm(want.taps[t], r);
                    const unsigned one[1] = {r};
                    bad += compare(got.taps[t].data(), want.taps[t].data(), W, std::vector<unsigned>(one, one + 1)).cos_min < 0.998;
                }
                std::sort(norms.begin(), norms.end());
                std::printf("   %2u | %.6f | %.6f | %.5f | %4u | %.2f / %.2f\n", cfg.deepstack_taps[t], m.cos_mean, m.cos_min, m.rel_l2, bad,
                            norms[norms.size() / 2], norms.back());
            }
            // The rows that end worst (output after the final norm), with their size in the residual stream.
            std::vector<std::pair<double, unsigned>> rank(g.rows);
            for (unsigned r = 0; r < g.rows; ++r) {
                const unsigned one[1] = {r};
                rank[r] = {compare(got.hidden.data(), want.hidden.data(), W, std::vector<unsigned>(one, one + 1)).cos_min, r};
            }
            std::sort(rank.begin(), rank.end());
            const std::vector<float>& last = want.taps.back();
            std::vector<double> ordered(g.rows);
            for (unsigned r = 0; r < g.rows; ++r) ordered[r] = norm(last, r);
            std::vector<double> sorted = ordered;
            std::sort(sorted.begin(), sorted.end());
            for (std::size_t i = 0; i < std::min<std::size_t>(6, rank.size()); ++i) {
                const unsigned r = rank[i].second;
                const auto above = static_cast<std::size_t>(std::lower_bound(sorted.begin(), sorted.end(), ordered[r]) - sorted.begin());
                std::printf("   worst row %4u (y %2u x %2u) cos %.6f | residual norm %.2f = percentile %.1f | device norm %.2f\n", r,
                            static_cast<unsigned>(g.y[r]), static_cast<unsigned>(g.x[r]), rank[i].first, ordered[r],
                            100.0 * static_cast<double>(above) / static_cast<double>(g.rows), norm(got.taps.back(), r));
            }
            std::fflush(stdout);
        }
    }
    return 0;
}
#endif

int run(int argc, char** argv) {
    std::string model, ref, backend_name = "host", only, diagnose_names;
    unsigned host_max_patches = 200;
    bool split = false, no_split = false, unpinned = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value of " + a);
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--ref") ref = next();
        else if (a == "--backend") backend_name = next();
        else if (a == "--only") only = next();
        else if (a == "--host-max-patches") host_max_patches = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--split") split = true;
        else if (a == "--no-split") no_split = true;
        else if (a == "--unpinned") unpinned = true;
        else if (a == "--diagnose") diagnose_names = next();
        else throw std::invalid_argument("unknown option " + a);
    }
    if (model.empty() || ref.empty()) throw std::invalid_argument("usage: --model FILE.gguf --ref DIR [--backend host|hip]");
    const bool device = backend_name == "hip";
    if (!device && backend_name != "host") throw std::invalid_argument("--backend host|hip");

    // Admission: the pinned byte count and SHA-256, then the strict structural validation.
    auto t0 = Clock::now();
    // The pinned length is checked on the open file, before a byte of it is read into memory.
    vision::HostBudget budget;
    if (!unpinned) budget.exact_bytes = ds::pinned_file(ds::PinnedFileKind::vision_bf16).bytes;
    vision::GgufWeights source{std::filesystem::path(model), ds::vision_reader_limits(), budget};
    const double read_ms = ms_since(t0);
    t0 = Clock::now();
    const ModelContentIdentity sha = sha256_bytes_fast(std::span<const std::byte>(source.bytes()));
    // --unpinned: a development tower with the same structure (never a gate result).
    if (!unpinned) ds::admit_file(ds::PinnedFileKind::vision_bf16, source.bytes().size(), sha);
    const ds::VisionModel vm = ds::validate_vision(source.info());
    std::printf("%s %s: %zu bytes, sha256 ok, %zu tensors (read %.0f ms, hash+validate %.0f ms)\n", unpinned ? "UNPINNED" : "admitted",
                unpinned ? "development file" : std::string(ds::pinned_file(ds::PinnedFileKind::vision_bf16).file_name).c_str(),
                source.bytes().size(),
                vm.manifest.tensors.size(), read_ms, ms_since(t0));

    ds::VisionAdapter adapter(vm.config, model_content_identity_hex(sha));
    vision::EncoderWeights weights;
    adapter.bind(source, weights);
    const vision::EncoderConfig cfg = adapter.config();
    vision::validate(cfg, weights);
    std::printf("identity %s\n", adapter.identity().c_str());

#ifdef STRIX_DS41_VISION_GATE_HIP
    if (!diagnose_names.empty()) return diagnose(diagnose_names, std::filesystem::path(ref), adapter, cfg, weights);
#else
    if (!diagnose_names.empty()) throw std::runtime_error("--diagnose needs the device build");
#endif
    std::unique_ptr<vision::EncoderBackend> backend;
#ifdef STRIX_DS41_VISION_GATE_HIP
    vision::Encoder* gpu = nullptr;
    if (device) {
        std::string why;
        if (!vision::hip_supported(cfg, &why)) throw std::runtime_error("tower not supported on the device: " + why);
        vision::EncoderOptions opt = adapter.device_options();  // the family's policy unless an arm is forced
        if (split) opt.split_activations = true;
        if (no_split) opt.split_activations = false;
        std::printf("split activations %d\n", opt.split_activations ? 1 : 0);
        auto enc = std::make_unique<vision::Encoder>(cfg, opt);
        t0 = Clock::now();
        enc->load(weights);
        std::printf("device load %.0f ms, %.1f MiB\n", ms_since(t0), static_cast<double>(enc->device_bytes()) / 1048576.0);
        gpu = enc.get();
        backend = std::move(enc);
    }
#else
    (void)split;
    (void)no_split;
    if (device) throw std::runtime_error("this build has no device backend");
#endif
    if (!backend) backend = std::make_unique<vision::ReferenceEncoder>(cfg, weights);
    std::printf("backend %s\n", backend->arithmetic());

    const double cos_mean_gate = device ? 0.9995 : 0.99999, cos_min_gate = device ? 0.998 : 0.99999;
    const double rel_gate = device ? 0.02 : 0.01;
    bool ok = true;
    unsigned ran = 0;
    const std::filesystem::path dir(ref);
    for (const Case& c : read_cases(dir)) {
        if (!listed(only, c.name)) continue;
        const std::vector<float> ref_patches = read_raw<float>(dir / (c.name + ".patches.f32"));
        const unsigned patches = c.grid_h * c.grid_w;
        if (ref_patches.size() != std::size_t{patches} * cfg.patch_dim()) throw std::runtime_error(c.name + ": reference patch shape");

        // 1. preprocessing
        if (!c.graph_only) {
            const std::vector<std::uint8_t> png = read_raw<std::uint8_t>(dir / (c.name + ".png"));
            t0 = Clock::now();
            const vision::Image image = vision::decode(png);
            const vision::PatchGrid ours = adapter.patches(image);
            const double pre_ms = ms_since(t0);
            const vision::Size target = adapter.target(image.width, image.height);
            const bool shape = ours.grid_h == c.grid_h && ours.grid_w == c.grid_w && target.width == c.resized_w &&
                               target.height == c.resized_h && ours.data.size() == ref_patches.size();
            // A NaN or infinity on either side fails here (it would vanish in a max or a > reduction).
            const vision::AbsDiff diff = shape ? vision::abs_diff(ours.data, ref_patches, 1e-6) : vision::AbsDiff{};
            const double worst = diff.worst;
            const std::size_t over = diff.over;
            const bool pass = shape && diff.pass;
            ok = ok && pass;
            std::printf("preprocess %-14s %4dx%-4d -> %4dx%-4d grid %3ux%-3u  max|diff| %.3g  over %zu  non-finite %zu  %.1f ms  %s\n", c.name.c_str(),
                        image.width, image.height, target.width, target.height, ours.grid_h, ours.grid_w, worst, over, diff.non_finite, pre_ms,
                        pass ? "PASS" : "FAIL");
        }

        // 2. encoder, reference patches in
        if (!device && patches > host_max_patches) {
            std::printf("encoder    %-14s skipped on the host backend (%u patches > %u)\n", c.name.c_str(), patches, host_max_patches);
            continue;
        }
        const std::filesystem::path rows_path = dir / (c.name + ".rows.f32");
        if (!std::filesystem::exists(rows_path)) {
            std::printf("encoder    %-14s no reference rows\n", c.name.c_str());
            ok = false;
            continue;
        }
        vision::PatchGrid g;
        g.grid_h = c.grid_h;
        g.grid_w = c.grid_w;
        g.rows = patches;
        g.patch_dim = cfg.patch_dim();
        g.data = ref_patches;
        g.y.resize(patches);
        g.x.resize(patches);
        for (unsigned r = 0; r < patches; ++r) {
            g.y[r] = static_cast<std::uint16_t>(r / c.grid_w);
            g.x[r] = static_cast<std::uint16_t>(r % c.grid_w);
        }
        const vision::PositionInputs pos = adapter.positions(cfg, weights, g);
        t0 = Clock::now();
        vision::EncoderOutput hidden = backend->encode(g, pos);
        const double encode_ms = ms_since(t0);
        std::vector<float> hidden_copy = hidden.hidden;
        t0 = Clock::now();
        vision::ProjectedRows rows;
        adapter.project(hidden, g, *backend, rows);
        const double project_ms = ms_since(t0);

        const vision::SpanLayout layout = adapter.layout(g);
        const std::vector<float> want = read_raw<float>(rows_path);
        bool pass = rows.rows == c.span_rows && layout.rows() == rows.rows && want.size() == std::size_t{rows.rows} * rows.width;
        Match m, h;
        bool special = true;
        if (pass) {
            std::vector<float> got(rows.bf16.size());
            for (std::size_t i = 0; i < got.size(); ++i) got[i] = vision::from_bf16(rows.bf16[i]);
            std::vector<unsigned> image_rows;
            for (unsigned r = 0; r < rows.rows; ++r) {
                if (layout.role[r] == vision::SpanLayout::Role::image) {
                    image_rows.push_back(r);
                } else {
                    special = special && std::memcmp(&got[std::size_t{r} * rows.width], &want[std::size_t{r} * rows.width], rows.width * sizeof(float)) == 0;
                }
            }
            m = compare(got.data(), want.data(), rows.width, image_rows);
            const std::filesystem::path hidden_path = dir / (c.name + ".hidden.f32");
            if (std::filesystem::exists(hidden_path)) {
                const std::vector<float> want_hidden = read_raw<float>(hidden_path);
                if (want_hidden.size() == hidden_copy.size()) {
                    std::vector<unsigned> all(patches);
                    for (unsigned r = 0; r < patches; ++r) all[r] = r;
                    h = compare(hidden_copy.data(), want_hidden.data(), cfg.width, all);
                }
            }
            pass = special && m.cos_mean >= cos_mean_gate && m.cos_min >= cos_min_gate && m.rel_l2 <= rel_gate;
        }
        ok = ok && pass;
        ++ran;
        std::printf("encoder    %-14s patches %5u rows %5u  rows cos mean %.6f min %.6f (row %u) relL2 %.5f | tower cos mean %.6f min %.6f relL2 %.5f | "
                    "special %s  encode %.1f ms project %.1f ms  %s\n",
                    c.name.c_str(), patches, rows.rows, m.cos_mean, m.cos_min, m.worst_row, m.rel_l2, h.cos_mean, h.cos_min, h.rel_l2,
                    special ? "exact" : "DIFFER", encode_ms, project_ms, pass ? "PASS" : "FAIL");
#ifdef STRIX_DS41_VISION_GATE_HIP
        if (gpu != nullptr) {
            // Receipt: a warm run without per-layer synchronisation, then one with it.
            vision::EncodeStats warm;
            (void)gpu->encode(g, pos, warm);
            vision::EncodeStats layers;
            layers.per_layer = true;
            (void)gpu->encode(g, pos, layers);
            double lo = 1e300, hi = 0.0, sum = 0.0;
            for (const double v : layers.layer_ms) {
                lo = std::min(lo, v);
                hi = std::max(hi, v);
                sum += v;
            }
            const vision::EncoderPlan plan = vision::plan(cfg, patches, 1U);
            std::printf("receipt    %-14s tower %.1f ms warm | per layer mean %.2f min %.2f max %.2f ms (sum %.1f) | device %.1f MiB (plan: panels %.1f, "
                        "workspace %.1f)\n",
                        c.name.c_str(), warm.total_ms, layers.layer_ms.empty() ? 0.0 : sum / static_cast<double>(layers.layer_ms.size()),
                        layers.layer_ms.empty() ? 0.0 : lo, hi, sum, static_cast<double>(gpu->device_bytes()) / 1048576.0,
                        static_cast<double>(plan.weight_bytes) / 1048576.0, static_cast<double>(plan.workspace_bytes()) / 1048576.0);
        }
#endif
        std::fflush(stdout);
    }
    if (ran == 0) ok = false;
    std::printf("deepseek41-vision-gate (%s%s): %s\n", backend->arithmetic(), unpinned ? ", UNPINNED development file" : "",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::printf("deepseek41-vision-gate: ERROR %s\n", e.what());
        return 2;
    }
}
