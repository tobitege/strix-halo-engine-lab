#pragma once
// STRIX_GLM_FAST_I4_PACK=<dir>: optional pre-converted INT4 routed-expert pack for STRIX_GLM_MODEL_FORMAT=fast_i4, so a
// load reads the INT4 layers instead of converting them (and does not read their routed EXL3 trellis at all).
//   <dir>/glm53-fast_i4-r<rank>of<world>/manifest.txt   header (format, model digest, rank slice, group, converter) +
//                                                        one line per stored layer: "layer L bytes B checksum C"
//   <dir>/glm53-fast_i4-r<rank>of<world>/L<layer>.i4    the layer's regions in order wq0 ws0 wq1 ws1 [wq2 ws2] (scope)
// A layer is used only when the header matches exactly and its file has the recorded size; its checksum is verified
// while reading (STRIX_GLM_FAST_I4_PACK_VERIFY=0 skips it) and a mismatch refuses the load. Missing layers are
// converted; STRIX_GLM_FAST_I4_PACK_WRITE=1 writes them (tmp + rename, manifest last), at most
// STRIX_GLM_FAST_I4_PACK_LAYERS=N new layers, and never with less than the layer + 8 GiB free on the file system.
#include <sys/statvfs.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strix::glm::engine::fast_i4 {

inline constexpr const char* kConverter = "glm-fast-i4-g64-v1;iu4_convert_kernel(mul1_pair decode, s=fp16(max|w|/7), q=rne clamp7, trellis K16xN16 tiles)";
inline constexpr std::uint64_t kChunk = std::uint64_t{256} << 20;  // checksum / transfer chunk (both directions)

// 64-bit checksum of one chunk: 8 interleaved FNV-style lanes over 8-byte words (4 threads), folded in order.
inline std::uint64_t chunk_hash(const std::uint8_t* p, std::uint64_t n) {
    constexpr unsigned kParts = 4;
    std::uint64_t part[kParts] = {};
    const std::uint64_t per = (n / kParts) & ~std::uint64_t{7};
    auto run = [&](unsigned t) {
        const std::uint8_t* b = p + per * t;
        const std::uint64_t len = t + 1 == kParts ? n - per * t : per;
        std::uint64_t h = 0xcbf29ce484222325ULL ^ (t + 1);
        std::uint64_t i = 0;
        for (; i + 8 <= len; i += 8) {
            std::uint64_t w;
            std::memcpy(&w, b + i, 8);
            h = (h ^ w) * 0x100000001b3ULL;
            h ^= h >> 29;
        }
        for (; i < len; ++i) h = (h ^ b[i]) * 0x100000001b3ULL;
        part[t] = h;
    };
    std::vector<std::thread> th;
    for (unsigned t = 1; t < kParts; ++t) th.emplace_back(run, t);
    run(0);
    for (auto& t : th) t.join();
    std::uint64_t h = 0x84222325cbf29ce4ULL;
    for (auto v : part) h = (h ^ v) * 0x100000001b3ULL;
    return h;
}
inline std::uint64_t fold(std::uint64_t h, std::uint64_t v) { return (h ^ v) * 0x100000001b3ULL + 0x9e3779b97f4a7c15ULL; }
inline std::string hex(std::uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(v));
    return b;
}


// Model digest: FNV over config.json + model.safetensors.index.json contents and every *.safetensors name and size.
inline std::uint64_t model_digest(const std::string& dir) {
    namespace fs = std::filesystem;
    std::uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&](const std::string& s) {
        for (const char ch : s) h = (h ^ static_cast<unsigned char>(ch)) * 0x100000001b3ULL;
    };
    for (const char* f : {"config.json", "model.safetensors.index.json"}) {
        std::ifstream in(dir + "/" + f, std::ios::binary);
        mix(std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
    }
    std::vector<std::pair<std::string, std::uint64_t>> shards;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.is_regular_file() && e.path().extension() == ".safetensors") shards.emplace_back(e.path().filename().string(), e.file_size());
    std::sort(shards.begin(), shards.end());
    for (const auto& [n, s] : shards) mix(n + ":" + std::to_string(s) + ";");
    return h;
}

struct Pack {
    std::string dir;     // the rank directory (empty: no pack)
    std::string header;  // expected header text
    std::map<unsigned, std::pair<std::uint64_t, std::uint64_t>> layers;  // layer -> (bytes, checksum) usable
    bool write = false, verify = true;
    unsigned write_left = ~0U;
    double read_seconds = 0, write_seconds = 0;
    unsigned read_layers = 0, written_layers = 0;

    static Pack open(const std::string& model_dir, unsigned rank, unsigned world, unsigned slice_i, unsigned matrices, const std::string& smooth,
                     const std::string& scale = std::string()) {  // scale: empty = the legacy rule (headers unchanged)
        Pack p;
        const char* d = std::getenv("STRIX_GLM_FAST_I4_PACK");
        auto flag = [](const char* name, bool dflt) {
            const char* v = std::getenv(name);
            if (!v || !*v) return dflt;
            if (std::strcmp(v, "1") == 0) return true;
            if (std::strcmp(v, "0") == 0) return false;
            throw std::invalid_argument(std::string(name) + " must be 0 or 1");
        };
        p.write = flag("STRIX_GLM_FAST_I4_PACK_WRITE", false);
        p.verify = flag("STRIX_GLM_FAST_I4_PACK_VERIFY", true);
        if (const char* n = std::getenv("STRIX_GLM_FAST_I4_PACK_LAYERS"); n && *n) p.write_left = static_cast<unsigned>(std::stoul(n));
        if (!d || !*d) {
            if (p.write) throw std::invalid_argument("STRIX_GLM_FAST_I4_PACK_WRITE=1 needs STRIX_GLM_FAST_I4_PACK=<dir>");
            return p;
        }
        p.dir = std::string(d) + "/glm53-fast_i4-r" + std::to_string(rank) + "of" + std::to_string(world);
        std::ostringstream h;
        h << "format glm-fast-i4-pack 1\nmodel " << hex(model_digest(model_dir)) << "\nrank " << rank << " world " << world
          << " routed-slice I " << slice_i << " offset " << rank * slice_i << "\ngroup 64\nmatrices "
          << (matrices == 3U ? "gate,up,down" : "gate,up") << "\nsmooth " << smooth << "\nconverter " << kConverter
          << (scale.empty() ? std::string() : ";scale=" + scale) << "\n";
        p.header = h.str();
        std::ifstream in(p.dir + "/manifest.txt");
        if (!in) {
            std::printf("glm fast_i4 pack: %s has no manifest (layers will be converted%s)\n", p.dir.c_str(), p.write ? " and written" : "");
            return p;
        }
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (text.compare(0, p.header.size(), p.header) != 0) {
            std::printf("glm fast_i4 pack: %s manifest does not match this model / rank / converter: ignored%s\n", p.dir.c_str(),
                        p.write ? " (not overwritten: write disabled for a mismatching pack)" : "");
            p.write = false;
            return p;
        }
        std::istringstream body(text.substr(p.header.size()));
        std::string word;
        unsigned l = 0;
        std::uint64_t bytes = 0;
        std::string sum;
        while (body >> word) {
            if (word != "layer" || !(body >> l >> word >> bytes >> word >> sum)) throw std::runtime_error("glm fast_i4 pack: bad manifest line in " + p.dir);
            std::error_code ec;
            const auto size = std::filesystem::file_size(p.dir + "/L" + std::to_string(l) + ".i4", ec);
            if (!ec && size == bytes) p.layers[l] = {bytes, std::stoull(sum, nullptr, 16)};
            else std::printf("glm fast_i4 pack: layer %u file missing or wrong size: converted instead\n", l);
        }
        std::printf("glm fast_i4 pack: %s matches (%zu layers stored)\n", p.dir.c_str(), p.layers.size());
        return p;
    }
    bool has(unsigned layer) const { return layers.count(layer) != 0; }
    bool want_write(std::uint64_t bytes) {
        if (!write || write_left == 0 || dir.empty()) return false;
        std::filesystem::create_directories(dir);
        struct statvfs s {};
        if (statvfs(dir.c_str(), &s) != 0) return false;
        const std::uint64_t free_bytes = static_cast<std::uint64_t>(s.f_bavail) * s.f_frsize;
        if (free_bytes < bytes + (std::uint64_t{8} << 30)) {
            std::printf("glm fast_i4 pack: %.1f GiB free < layer %.2f GiB + 8 GiB reserve: stop writing\n",
                        static_cast<double>(free_bytes) / double(1ULL << 30), static_cast<double>(bytes) / double(1ULL << 30));
            write = false;
            return false;
        }
        return true;
    }
    void save_manifest() const {
        std::ostringstream m;
        m << header;
        for (const auto& [l, v] : layers) m << "layer " << l << " bytes " << v.first << " checksum " << hex(v.second) << "\n";
        const std::string tmp = dir + "/manifest.txt.tmp";
        {
            std::ofstream out(tmp, std::ios::trunc);
            out << m.str();
            if (!out) throw std::runtime_error("glm fast_i4 pack: cannot write " + tmp);
        }
        std::filesystem::rename(tmp, dir + "/manifest.txt");
    }
};

}  // namespace strix::glm::engine::fast_i4
