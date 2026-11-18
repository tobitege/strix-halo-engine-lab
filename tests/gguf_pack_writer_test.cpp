// Host test: PackWriter keeps the source key/value section byte for byte, writes an aligned tensor table with the NVFP4 tile
// pack type, and the result parses with gguf::read_file (geometry of kNvfp4TilePack = nvfp4_bytes).
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "strix/gguf_reader.hpp"
#include "strix/gguf_writer.hpp"

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("FAIL: %s\n", m); ++fails; } } while (0)

static void u32(std::string& s, std::uint32_t v) { s.append(reinterpret_cast<const char*>(&v), 4); }
static void u64(std::string& s, std::uint64_t v) { s.append(reinterpret_cast<const char*>(&v), 8); }
static void str(std::string& s, const std::string& v) { u64(s, v.size()); s += v; }

int main() {
    // source: 2 KV (a string, a uint32), 2 tensors: f32 [4], bf16 [128,16] (4096 bytes)
    std::string src = "GGUF";
    u32(src, 3); u64(src, 2); u64(src, 2);
    str(src, "general.architecture"); u32(src, 8); str(src, "qwen35");
    str(src, "qwen35.block_count"); u32(src, 4); u32(src, 7);
    str(src, "norm.weight"); u32(src, 1); u64(src, 4); u32(src, 0); u64(src, 0);
    str(src, "w.weight"); u32(src, 2); u64(src, 128); u64(src, 16); u32(src, 30); u64(src, 32);
    while (src.size() % 32) src += '\0';
    src.append(16, 'n'); src.append(16, '\0');
    src.append(4096, 'w');
    {
        std::ofstream f("pw_src.gguf", std::ios::binary);
        f.write(src.data(), static_cast<std::streamsize>(src.size()));
    }
    const auto info = strix::gguf::read_file("pw_src.gguf");
    CHECK(info.metadata_count == 2 && info.metadata_end > 24, "metadata span recorded");
    const std::uint64_t pack_bytes = 1 * 1 * 1152;  // out 16, in 128 -> one record
    std::vector<strix::gguf::OutTensor> outs = {
        {"norm.weight", {4}, 0, 16, 0}, {"w.weight", {128, 16}, strix::gguf::kNvfp4TilePack, pack_bytes, 0}, {"strix.nvfp4.gscale", {1}, 0, 4, 0}};
    std::uint64_t total;
    {
        strix::gguf::PackWriter w("pw_out.gguf", "pw_src.gguf", info, outs);
        w.append("nnnnnnnnnnnnnnnn", 16); w.finish_tensor();
        std::vector<unsigned char> rec(pack_bytes, 0xA5);
        w.append(rec.data(), 1000); w.append(rec.data() + 1000, pack_bytes - 1000); w.finish_tensor();
        const float gs = 0.25f;
        w.append(&gs, 4); w.finish_tensor();
        total = w.close();
    }
    const auto out = strix::gguf::read_file("pw_out.gguf");
    CHECK(out.file_size == total, "file size");
    CHECK(out.metadata_count == 2 && out.metadata_end == info.metadata_end, "kv section length kept");
    CHECK(out.tensors.size() == 3, "3 tensors");
    CHECK(out.tensors[1].ggml_type == strix::gguf::kNvfp4TilePack && out.tensors[1].byte_size == pack_bytes, "tile pack geometry");
    CHECK(out.tensors[2].byte_size == 4 && out.tensors[2].absolute_offset % 32 == 0, "gscale table aligned");
    const auto* a = out.find_metadata("qwen35.block_count");
    CHECK(a && a->unsigned_value && *a->unsigned_value == 7, "kv value kept");
    std::ifstream f("pw_out.gguf", std::ios::binary);
    f.seekg(static_cast<std::streamoff>(out.tensors[2].absolute_offset));
    float g = 0; f.read(reinterpret_cast<char*>(&g), 4);
    CHECK(g == 0.25f, "gscale value");
    f.seekg(static_cast<std::streamoff>(out.tensors[1].absolute_offset + 1100));
    unsigned char b = 0; f.read(reinterpret_cast<char*>(&b), 1);
    CHECK(b == 0xA5, "tile bytes");
    std::remove("pw_src.gguf"); std::remove("pw_out.gguf");
    std::printf(fails ? "gguf_pack_writer_test FAILED\n" : "gguf_pack_writer_test PASS\n");
    return fails ? 1 : 0;
}
