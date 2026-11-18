// End-to-end gate and timing of the CUDA Qwen prefill (one model load, several jobs).
//   qwen_prefill_e2e <tables dir> <rope table> <max positions> job...
//   job = gate:<tokens file>:<N>:<HIP --validation-final-state dir>   bitwise state comparison of rows [0, N)
//       | time:<tokens file>:<N>                                     prefill of rows [0, N), timing only
//       | rtxs:<node export file>                                    the node's STRIX_RTX_EXPORT_STATE file (any route, incl.
//                                                                    MTP): same prefix, every component compared
// gate compares attention-kv.raw (rows [0, N) of each of the 12 slices; the export holds the whole context per slice),
// sparse-index.raw, recurrent-matrix.raw, gdn-convolution.raw, ple-convolution.raw and prints the first difference
// per component. Block schedule: STRIX_QWEN_BLOCKS (comma list of rows), default the Config default.
#include "strix/cuda/qwen_prefill.hpp"
#include "strix/serve/json.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qp = strix::cuda::qwen_prefill;

static std::vector<std::uint8_t> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "missing %s\n", p.c_str()); std::exit(2); }
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static std::vector<std::uint32_t> tokens_of(const std::string& path) {
    std::ifstream f(path);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (auto& ch : s) if (ch == ',') ch = ' ';
    std::istringstream in(s);
    std::vector<std::uint32_t> t;
    std::uint32_t v;
    while (in >> v) t.push_back(v);
    return t;
}

static std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> out;
    std::string cur;
    for (std::size_t i = 0; i < s.size(); ++i) {
        // keep drive letters ("D:\...") inside one field
        if (s[i] == c && !(i == 1 || (i >= 2 && s[i - 2] == c && i + 1 < s.size() && s[i + 1] == '\\'))) { out.push_back(cur); cur.clear(); }
        else cur.push_back(s[i]);
    }
    out.push_back(cur);
    return out;
}

static int run(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: qwen_prefill_e2e tables rope max_positions job...\n"); return 1; }
    qp::Config c;
    c.model_gguf = std::getenv("STRIX_QWEN_GGUF") ? std::getenv("STRIX_QWEN_GGUF") : "Qwen3.8-Flash-Next-UD-Q4_K_XL-merged.gguf";
    if (const char* s = std::getenv("STRIX_QWEN_MTP")) c.mtp_sidecar_gguf = s;
    c.tables_dir = argv[1];
    c.rope_table = argv[2];
    c.max_positions = static_cast<std::uint32_t>(std::atoi(argv[3]));
    if (const char* b = std::getenv("STRIX_QWEN_BLOCKS")) {
        c.block_schedule.clear();
        for (const auto& x : split(b, ',')) c.block_schedule.push_back(static_cast<std::uint32_t>(std::atoi(x.c_str())));
    }
    if (const char* v = std::getenv("STRIX_QWEN_TAIL_LARGE")) c.tail_large = v[0] == '1';
    if (const char* v = std::getenv("STRIX_QWEN_FINAL_BLOCK_ROWS")) c.final_block_rows = static_cast<std::uint32_t>(std::stoul(v));
    if (const char* v = std::getenv("STRIX_QWEN_BY_LAYER")) c.release_by_layer = v[0] == '1';
    // STRIX_QWEN_NO_EXPORT=1: the state stays on the device (no pinned staging, no pieces): RTX-only timing of long prompts
    if (const char* v = std::getenv("STRIX_QWEN_NO_EXPORT")) c.export_state = v[0] != '1';
    std::printf("loading...\n");
    std::fflush(stdout);
    qp::Prefill p(c);
    std::printf("loaded %s\n", p.describe().c_str());
    std::fflush(stdout);
    int bad = 0;
    for (int a = 4; a < argc; ++a) {
        const std::string job = argv[a];
        const auto colon = job.find(':');
        const std::string kind = job.substr(0, colon);
        if (kind == "rtxs") {
            const std::string path = job.substr(colon + 1);
            const auto raw = slurp(path);
            std::uint64_t hn = 0;
            for (int b = 0; b < 8; ++b) hn |= static_cast<std::uint64_t>(raw[8 + b]) << (8 * b);
            const auto h = strix::serve::parse_json(std::string_view(reinterpret_cast<const char*>(raw.data()) + 16, hn));
            const std::uint8_t* payload = raw.data() + 16 + hn;
            const std::uint64_t position = static_cast<std::uint64_t>(h.find("position")->as_int());
            std::vector<std::uint32_t> tokens;
            for (const auto& t : h.find("tokens")->as_array()) tokens.push_back(static_cast<std::uint32_t>(t.as_int()));
            std::map<std::string, std::vector<std::uint8_t>> got;
            const auto t = p.run(tokens, position, [&](const qp::Piece& piece) {
                auto& v = got[piece.component];
                if (piece.fixed) { if (v.size() < piece.first + piece.bytes) v.resize(piece.first + piece.bytes); std::memcpy(v.data() + piece.first, piece.data, piece.bytes); }
                else v.insert(v.end(), piece.data, piece.data + piece.bytes);
            });
            std::printf("rtxs %s: position %llu, prefill %.3f s\n", path.c_str(), static_cast<unsigned long long>(position), t.total_s);
            std::uint64_t at = 0;
            for (const auto& r : h.find("rows")->as_array()) {
                const std::string name = r.find("id")->as_string();
                const std::uint64_t bpr = static_cast<std::uint64_t>(r.find("bytes_per_row")->as_int());
                const std::uint64_t shift = static_cast<std::uint64_t>(r.find("shift")->as_int());
                const std::uint64_t len = bpr * (position > shift ? position - shift : 0);
                const auto& v = got[name];
                if (v.size() != len) { std::printf("  %s: produced %zu bytes, node %llu\n", name.c_str(), v.size(), static_cast<unsigned long long>(len)); ++bad; }
                else {
                    std::uint64_t i = 0;
                    while (i < len && v[i] == payload[at + i]) ++i;
                    if (i < len) { std::printf("  %s: FIRST DIFF at byte %llu of %llu (row %llu)\n", name.c_str(), static_cast<unsigned long long>(i), static_cast<unsigned long long>(len), static_cast<unsigned long long>(i / bpr)); ++bad; }
                    else std::printf("  %s: %llu bytes identical\n", name.c_str(), static_cast<unsigned long long>(len));
                }
                at += len;
            }
            for (const auto& f : h.find("fixed")->as_array()) {
                const std::string name = f.find("id")->as_string();
                const std::uint64_t len = static_cast<std::uint64_t>(f.find("bytes")->as_int());
                const std::uint64_t off = static_cast<std::uint64_t>(f.find("offset")->as_int());
                const auto& v = got[name];
                if (v.size() != len) { std::printf("  %s: produced %zu bytes, node %llu\n", name.c_str(), v.size(), static_cast<unsigned long long>(len)); ++bad; continue; }
                std::uint64_t i = 0;
                while (i < len && v[i] == payload[at + off + i]) ++i;
                if (i < len) { std::printf("  %s: FIRST DIFF at byte %llu of %llu\n", name.c_str(), static_cast<unsigned long long>(i), static_cast<unsigned long long>(len)); ++bad; }
                else std::printf("  %s: %llu bytes identical\n", name.c_str(), static_cast<unsigned long long>(len));
            }
            std::fflush(stdout);
            continue;
        }
        if (kind == "xsched") {
            // xsched:TOKENS:N[:R] - the same prompt under the original block order, tail-large and
            // (if R > 0) a final-block reservation of R rows; every exported component must be
            // byte-identical across the schedules (no node capture needed).
            std::string rest = job.substr(colon + 1);
            const std::size_t from = (rest.size() > 2 && rest[1] == ':') ? 2 : 0;
            const auto k1 = rest.find(':', from);
            const std::string tfile = rest.substr(0, k1);
            rest = rest.substr(k1 + 1);
            const auto k2 = rest.find(':');
            const std::uint64_t n = std::strtoull(rest.substr(0, k2).c_str(), nullptr, 10);
            const std::uint32_t reserve = k2 == std::string::npos ? 0U : static_cast<std::uint32_t>(std::strtoul(rest.substr(k2 + 1).c_str(), nullptr, 10));
            const auto tokens = tokens_of(tfile);
            if (tokens.size() < n) { std::printf("%s: only %zu tokens\n", job.c_str(), tokens.size()); ++bad; continue; }
            struct Arm { const char* name; qp::RunOptions options; };
            std::vector<Arm> arms;
            arms.push_back({"default", {}});
            arms.back().options.tail_large = false;
            arms.back().options.final_block_rows = 0;
            arms.push_back({"tail-large", {}});
            arms.back().options.tail_large = true;
            arms.back().options.final_block_rows = 0;
            if (reserve) {
                arms.push_back({"final-block-rows", {}});
                arms.back().options.tail_large = false;
                arms.back().options.final_block_rows = reserve;
            }
            std::vector<std::map<std::string, std::vector<std::uint8_t>>> states;
            for (const auto& arm : arms) {
                std::map<std::string, std::vector<std::uint8_t>> got;
                const auto t = p.run(tokens, n, [&](const qp::Piece& piece) {
                    auto& v = got[piece.component];
                    // place rows/bytes at their absolute offset so arrival order cannot matter
                    const std::uint64_t at = piece.fixed ? piece.first : piece.first * (piece.count ? piece.bytes / piece.count : 0);
                    if (v.size() < at + piece.bytes) v.resize(at + piece.bytes);
                    std::memcpy(v.data() + at, piece.data, piece.bytes);
                }, arm.options);
                std::printf("xsched %llu rows %s: %.3f s, blocks", static_cast<unsigned long long>(n), arm.name, t.total_s);
                for (double b : t.block_s) std::printf(" %.3f", b);
                std::printf("\n");
                states.push_back(std::move(got));
            }
            for (std::size_t s = 1; s < states.size(); ++s) {
                if (states[s].size() != states[0].size()) { std::printf("  %s: %zu components vs %zu\n", arms[s].name, states[s].size(), states[0].size()); ++bad; continue; }
                std::uint64_t total = 0;
                for (const auto& [name, ref] : states[0]) {
                    const auto it = states[s].find(name);
                    if (it == states[s].end() || it->second != ref) {
                        std::printf("  %s vs default: %s DIFFERS\n", arms[s].name, name.c_str());
                        ++bad;
                    }
                    total += ref.size();
                }
                std::printf("  %s vs default: %zu components, %llu bytes compared\n", arms[s].name, states[0].size(), static_cast<unsigned long long>(total));
            }
            std::fflush(stdout);
            continue;
        }
        // fields after the kind: path:N[:dir] where paths may contain a drive colon
        std::string rest = job.substr(colon + 1);
        auto take_path = [&]() {
            std::size_t from = (rest.size() > 2 && rest[1] == ':') ? 2 : 0;
            const auto k = rest.find(':', from);
            std::string v = rest.substr(0, k);
            rest = k == std::string::npos ? std::string() : rest.substr(k + 1);
            return v;
        };
        const std::string tfile = take_path();
        const std::uint64_t n = std::strtoull(take_path().c_str(), nullptr, 10);
        const auto tokens = tokens_of(tfile);
        if (tokens.size() < n) { std::printf("%s: only %zu tokens\n", job.c_str(), tokens.size()); ++bad; continue; }
        std::map<std::string, std::vector<std::uint8_t>> got;
        std::uint64_t bytes = 0;
        const auto t = p.run(tokens, n, [&](const qp::Piece& piece) {
            bytes += piece.bytes;
            if (kind == "gate") {
                auto& v = got[piece.component];
                v.insert(v.end(), piece.data, piece.data + piece.bytes);
            }
        });
        std::printf("%s %llu rows: %.3f s, %.0f tok/s, first block %.3f s, state %.1f MB, blocks", kind.c_str(),
                    static_cast<unsigned long long>(n), t.total_s, n / t.total_s, t.first_block_s, bytes / 1e6);
        for (double b : t.block_s) std::printf(" %.3f", b);
        std::printf("\n");
        std::fflush(stdout);
        if (kind != "gate") continue;
        const std::string sdir = take_path();
        const auto compare = [&](const std::string& what, const std::uint8_t* x, const std::uint8_t* y, std::uint64_t len) {
            for (std::uint64_t i = 0; i < len; ++i)
                if (x[i] != y[i]) {
                    std::printf("  %s: FIRST DIFF at byte %llu of %llu (cuda %02x hip %02x)\n", what.c_str(), static_cast<unsigned long long>(i),
                                static_cast<unsigned long long>(len), x[i], y[i]);
                    ++bad;
                    return;
                }
            std::printf("  %s: %llu bytes identical\n", what.c_str(), static_cast<unsigned long long>(len));
        };
        const auto kv = slurp(sdir + "/attention-kv.raw");
        const auto idx = slurp(sdir + "/sparse-index.raw");
        const std::uint64_t cap = kv.size() / 12 / 2048;
        for (std::uint32_t o = 0; o < 12; ++o) {
            const auto name = "kv.L" + std::to_string(4 * o + 3);
            compare(name, got[name].data(), kv.data() + o * cap * 2048, n * 2048);
            const auto iname = "index.L" + std::to_string(4 * o + 3);
            compare(iname, got[iname].data(), idx.data() + o * cap * 256, n * 256);
        }
        for (const auto& [name, file] : std::map<std::string, std::string>{{"recurrent_matrix", "recurrent-matrix.raw"},
                                                                             {"gdn_convolution", "gdn-convolution.raw"},
                                                                             {"ple_convolution", "ple-convolution.raw"},
                                                                             {"ngram_history", "ngram-history.raw"},
                                                                             {"position_history", "position-history.raw"}}) {
            const auto ref = slurp(sdir + "/" + file);
            if (got[name].size() != ref.size()) { std::printf("  %s: size %zu vs %zu\n", name.c_str(), got[name].size(), ref.size()); ++bad; continue; }
            compare(name, got[name].data(), ref.data(), ref.size());
        }
        std::fflush(stdout);
    }
    std::printf("%s\n", bad ? "E2E FAIL" : "E2E ALL BIT-IDENTICAL");
    return bad ? 1 : 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {  // e.g. the VRAM guard's refusal: say why instead of an unexplained abort
        std::fprintf(stderr, "qwen_prefill_e2e: %s\n", e.what());
        std::printf("E2E ERROR: %s\n", e.what());
        return 4;
    }
}
