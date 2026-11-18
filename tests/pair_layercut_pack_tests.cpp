// Host-only regression: a cut pack never opens files that contain only suffix layers.
#include "cuda/models/minimax/mm_rtx_pack.hpp"
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;
void shard(const fs::path& file, const std::string& name, std::uint16_t value) {
    const std::string header = "{\"" + name + "\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
    const std::uint64_t bytes = header.size();
    std::ofstream out(file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&bytes), 8);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(&value), 2);
}
}
int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error("need fixture output directory");
    const fs::path dir = argv[1];
    fs::create_directories(dir);
    shard(dir / "embedding.safetensors", "language_model.model.embed_tokens.weight", 0x3f80);
    shard(dir / "prefix.safetensors", "language_model.model.layers.0.input_layernorm.weight", 0x4000);
    {
        std::ofstream out(dir / "model.safetensors.index.json");
        out << R"({"weight_map":{"language_model.model.embed_tokens.weight":"embedding.safetensors","language_model.model.layers.0.input_layernorm.weight":"prefix.safetensors","language_model.model.layers.1.input_layernorm.weight":"absent-suffix.safetensors","language_model.model.norm.weight":"absent-norm.safetensors"}})";
    }
    {
        strix::minimax::rtxpf::host::Pack pack(dir.string(), 1);
        if (pack.floats("language_model.model.embed_tokens.weight")[0] != 1.0F ||
            pack.floats("language_model.model.layers.0.input_layernorm.weight")[0] != 2.0F ||
            pack.has("language_model.model.layers.1.input_layernorm.weight"))
            throw std::runtime_error("prefix pack shape/content");
    }
    bool rejected = false;
    try { strix::minimax::rtxpf::host::Pack pack(dir.string(), 2); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("requested missing layer was accepted");
    std::puts("PASS: prefix-only pack reads existing prefix and excludes missing suffix");
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
