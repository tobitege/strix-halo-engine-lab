// Host tests of the DeepSeek V4.1 image tower adapter: strict file validation, the GGUF weight source, the resize
// table, patch order, the 3x3 unshuffle + aligner projector against a direct computation, the span layout and the
// cache key. The tower here is a small synthetic one with the real file's metadata keys and tensor names.
#include "../src/deepseek41/vision_adapter.hpp"

#include "strix/deepseek41_model.hpp"
#include "strix/vision/compare.hpp"
#include "strix/vision/embed_cache.hpp"
#include "strix/vision/encoder.hpp"
#include "strix/vision/gguf_weights.hpp"
#include "strix/vision/image.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace strix;
namespace ds = strix::deepseek41;

int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            ++failures;                                                          \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
        }                                                                        \
    } while (false)

bool throws(const std::function<void()>& fn, const char* needle = "") {
    try {
        fn();
    } catch (const std::exception& e) {
        if (std::strstr(e.what(), needle) != nullptr) return true;
        std::printf("  unexpected message: %s\n", e.what());
        return false;
    }
    return false;
}

// ---- a GGUF v3 file in memory ----------------------------------------------------------------------------------

using Value = std::variant<std::uint32_t, float, std::string>;
struct TensorSpec {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::uint32_t type = 30;
};

template <class T>
void put(std::string& b, T v) {
    b.append(reinterpret_cast<const char*>(&v), sizeof v);
}
void put_text(std::string& b, const std::string& s) {
    put<std::uint64_t>(b, s.size());
    b += s;
}

struct File {
    std::vector<std::pair<std::string, Value>> meta;
    std::vector<TensorSpec> tensors;

    void set(const std::string& key, Value v) {
        for (auto& m : meta)
            if (m.first == key) {
                m.second = std::move(v);
                return;
            }
        meta.emplace_back(key, std::move(v));
    }
    void erase(const std::string& key) {
        std::erase_if(meta, [&](const auto& m) { return m.first == key; });
    }
    TensorSpec& tensor(const std::string& name) {
        for (auto& t : tensors)
            if (t.name == name) return t;
        throw std::logic_error("no tensor " + name);
    }
    // Element e of tensor t holds the BF16 of value(t, e) (types 30) so every weight is deterministic and distinct.
    static float value(std::size_t t, std::size_t e) {
        const std::uint32_t h = static_cast<std::uint32_t>(t * 2654435761U + e * 40503U + 12345U);
        return (static_cast<float>((h >> 8U) % 2001U) - 1000.0F) * (1.0F / 4096.0F);
    }
    std::string bytes() const {
        std::string b = "GGUF";
        put<std::uint32_t>(b, 3);
        put<std::uint64_t>(b, tensors.size());
        put<std::uint64_t>(b, meta.size());
        for (const auto& [key, v] : meta) {
            put_text(b, key);
            if (const auto* u = std::get_if<std::uint32_t>(&v)) {
                put<std::uint32_t>(b, 4);
                put<std::uint32_t>(b, *u);
            } else if (const auto* f = std::get_if<float>(&v)) {
                put<std::uint32_t>(b, 6);
                put<float>(b, *f);
            } else {
                put<std::uint32_t>(b, 8);
                put_text(b, std::get<std::string>(v));
            }
        }
        std::uint64_t offset = 0;
        std::vector<std::uint64_t> counts;
        for (const auto& t : tensors) {
            put_text(b, t.name);
            put<std::uint32_t>(b, static_cast<std::uint32_t>(t.dims.size()));
            std::uint64_t n = 1;
            for (const auto d : t.dims) {
                put<std::uint64_t>(b, d);
                n *= d;
            }
            put<std::uint32_t>(b, t.type);
            put<std::uint64_t>(b, offset);
            counts.push_back(n);
            offset += (n * 2U + 31U) / 32U * 32U;
        }
        while (b.size() % 32 != 0) b.push_back('\0');
        for (std::size_t t = 0; t < tensors.size(); ++t) {
            const std::size_t start = b.size();
            for (std::uint64_t e = 0; e < counts[t]; ++e) put<std::uint16_t>(b, vision::to_bf16(value(t, e)));
            b.append((counts[t] * 2U + 31U) / 32U * 32U - (b.size() - start), '\0');
        }
        return b;
    }
    gguf::FileInfo info() const {
        const std::string b = bytes();
        std::istringstream in(b, std::ios::in | std::ios::binary);
        return gguf::read(in, b.size(), ds::vision_reader_limits());
    }
    vision::GgufWeights weights() const {
        const std::string b = bytes();
        std::vector<std::byte> raw(b.size());
        std::memcpy(raw.data(), b.data(), b.size());
        return vision::GgufWeights(info(), std::move(raw));
    }
};

// layers 2, width 32, 2 heads (dim 16), SwiGLU 24, patch 2, 3x3 merge (cell 6), projection 40, at most 16 image rows.
File small_file() {
    const std::string a = "deepseek4-vision.";
    File f;
    f.set("general.architecture", std::string("deepseek4-vision"));
    f.set("general.source.revision", std::string(ds::pinned_source_revision));
    f.set(a + "checkpoint_variant", std::string("v4.1-flash"));
    f.set(a + "rope.freq_base", 10000.0F);
    f.set(a + "attention.layer_norm_rms_epsilon", 1e-6F);
    f.set(a + "block_count", std::uint32_t{2});
    f.set(a + "embedding_length", std::uint32_t{32});
    f.set(a + "feed_forward_length", std::uint32_t{24});
    f.set(a + "attention.head_count", std::uint32_t{2});
    f.set(a + "projection_length", std::uint32_t{40});
    f.set(a + "patch_size", std::uint32_t{2});
    f.set(a + "downsample_ratio", std::uint32_t{3});
    f.set(a + "image.max_tokens", std::uint32_t{16});
    f.set(a + "image.min_pixels", std::uint32_t{144});
    f.set(a + "image.max_width_height_ratio", std::uint32_t{0});
    f.set(a + "image.token_id", std::uint32_t{5});
    f.set(a + "language.block_count", std::uint32_t{4});
    f.set(a + "language.expert_count", std::uint32_t{8});
    const std::uint64_t W = 32, M = 24, P = 40;
    f.tensors = {{"aligner.w1.bias", {P}}, {"aligner.w1.weight", {W * 9, P}}, {"aligner.w2.bias", {P}},
                 {"aligner.w2.weight", {P, P}}, {"image_end", {P}}, {"image_newline", {P}}, {"image_start", {P}}};
    for (int l = 0; l < 2; ++l) {
        const std::string p = "vision.blocks." + std::to_string(l) + ".";
        f.tensors.push_back({p + "attn.wo.bias", {W}});
        f.tensors.push_back({p + "attn.wo.weight", {W, W}});
        f.tensors.push_back({p + "attn.wqkv.bias", {3 * W}});
        f.tensors.push_back({p + "attn.wqkv.weight", {W, 3 * W}});
        f.tensors.push_back({p + "mlp.w1.weight", {W, 2 * M}});
        f.tensors.push_back({p + "mlp.w2.weight", {M, W}});
        f.tensors.push_back({p + "norm1.weight", {W}});
        f.tensors.push_back({p + "norm2.weight", {W}});
    }
    f.tensors.push_back({"vision.norm.weight", {W}});
    f.tensors.push_back({"vision.patch_embed.proj.bias", {W}});
    f.tensors.push_back({"vision.patch_embed.proj.weight", {12, W}});
    return f;
}

ds::VisionConfig real_config() {
    ds::VisionConfig c;
    c.num_hidden_layers = 32;
    c.hidden_size = 1024;
    c.num_attention_heads = 16;
    c.intermediate_size = 2816;
    c.patch_size = 14;
    c.downsample_ratio = 3;
    c.projection_size = 5120;
    c.max_image_tokens = 1024;
    c.min_pixels = 295936;
    c.image_token_id = 129264;
    c.language_layers = 40;
    c.language_experts = 384;
    c.rope_theta = 10000.0;
    c.rms_norm_eps = 1e-6;
    c.checkpoint_variant = "v4.1-flash";
    c.source_revision = std::string(ds::pinned_source_revision);
    return c;
}
const std::string kSha(64, 'a');

void test_validate() {
    const File good = small_file();
    const ds::VisionModel m = ds::validate_vision(good.info());
    CHECK(m.config.num_hidden_layers == 2 && m.config.hidden_size == 32 && m.config.head_dim() == 16);
    CHECK(m.config.cell() == 6 && m.config.aligner_inputs() == 288 && m.config.patch_features() == 12);
    CHECK(m.config.image_token_id == 5 && m.config.max_wh_ratio == 0 && m.config.min_pixels == 144);
    CHECK(std::fabs(m.config.rms_norm_eps - 1e-6) < 1e-12 && m.config.rope_theta == 10000.0);
    CHECK(m.manifest.tensors.size() == 26 && m.manifest.architecture == "deepseek4-vision");
    // The pinned file: 7 projector/special tensors, 8 per block, 3 tower tensors; 970 MB of BF16.
    const auto expected = ds::expected_vision_tensors(real_config());
    CHECK(expected.size() == 266);
    std::uint64_t elements = 0;
    for (const auto& t : expected) {
        std::uint64_t n = 1;
        for (const auto d : t.dimensions) n *= d;
        elements += n;
        CHECK(t.ggml_type == 30);
    }
    CHECK(elements * 2U < ds::pinned_file(ds::PinnedFileKind::vision_bf16).bytes);
    CHECK(ds::pinned_file(ds::PinnedFileKind::vision_bf16).bytes - elements * 2U < (1U << 16U));  // header + padding

    // The hyphenated key prefix is an opt-in of the vision reader: the strict reader of the other files refuses it.
    CHECK(throws(
        [&] {
            const std::string b = good.bytes();
            std::istringstream in(b, std::ios::in | std::ios::binary);
            (void)gguf::read(in, b.size(), ds::reader_limits());
        },
        "invalid GGUF metadata key"));
    CHECK(ds::vision_reader_limits().allow_hyphen_in_keys && !ds::reader_limits().allow_hyphen_in_keys && !gguf::Limits{}.allow_hyphen_in_keys);

    const auto bad = [&](const std::function<void(File&)>& edit, const char* needle) {
        File f = good;
        edit(f);
        return throws([&] { (void)ds::validate_vision(f.info()); }, needle);
    };
    const std::string a = "deepseek4-vision.";
    CHECK(bad([&](File& f) { f.set("general.architecture", std::string("deepseek41")); }, "general.architecture"));
    CHECK(bad([&](File& f) { f.erase(a + "downsample_ratio"); }, "missing metadata"));
    CHECK(bad([&](File& f) { f.set(a + "patch_size", 2.0F); }, "must be unsigned"));
    CHECK(bad([&](File& f) { f.set(a + "rope.freq_base", std::uint32_t{10000}); }, "must be a float"));
    CHECK(bad([&](File& f) { f.set(a + "checkpoint_variant", std::string("v4.1-pro")); }, "variant"));
    CHECK(bad([&](File& f) { f.set("general.source.revision", std::string("0000")); }, "source revision"));
    CHECK(bad([&](File& f) { f.set(a + "attention.head_count", std::uint32_t{3}); }, "head count"));
    CHECK(bad([&](File& f) { f.set(a + "attention.head_count", std::uint32_t{16}); }, "multiple of 4"));
    CHECK(bad([&](File& f) { f.set(a + "image.min_pixels", std::uint32_t{35}); }, "min_pixels"));
    CHECK(bad([&](File& f) { f.set(a + "image.min_pixels", std::uint32_t{36 * 16 + 1}); }, "min_pixels"));
    CHECK(bad([&](File& f) { f.set(a + "attention.layer_norm_rms_epsilon", 0.0F); }, "epsilon"));
    CHECK(bad([&](File& f) { f.set(a + "downsample_ratio", std::uint32_t{0}); }, "downsample_ratio"));
    CHECK(bad([&](File& f) { f.tensors.push_back({"vision.pos_embed", {32}}); }, "tensor count"));
    CHECK(bad([&](File& f) { f.tensor("image_end").name = "image_stop"; }, "unexpected tensor image_stop"));
    CHECK(bad([&](File& f) { f.tensor("aligner.w1.weight").dims = {40, 288}; }, "unexpected dimensions"));
    CHECK(bad([&](File& f) { f.tensor("vision.norm.weight").type = 1; }, "storage type"));
    CHECK(bad([&](File& f) { f.set(a + "feed_forward_length", std::uint32_t{25}); }, "unexpected dimensions"));

    ds::Config target;
    target.image_token_id = 5;
    target.vocab_size = 64;
    target.hidden_size = 40;
    target.num_hidden_layers = 4;
    target.n_routed_experts = 8;
    ds::require_vision_matches_target(m.config, target);
    const auto mismatch = [&](const std::function<void(ds::Config&)>& edit) {
        ds::Config t = target;
        edit(t);
        return throws([&] { ds::require_vision_matches_target(m.config, t); });
    };
    CHECK(mismatch([](ds::Config& t) { t.image_token_id = 6; }));
    CHECK(mismatch([](ds::Config& t) { t.vocab_size = 5; }));
    CHECK(mismatch([](ds::Config& t) { t.hidden_size = 48; }));
    CHECK(mismatch([](ds::Config& t) { t.num_hidden_layers = 5; }));
    CHECK(mismatch([](ds::Config& t) { t.n_routed_experts = 4; }));
}

void test_weight_source() {
    const File f = small_file();
    vision::GgufWeights w = f.weights();
    CHECK(w.has("image_start") && !w.has("image_stop"));
    const vision::Tensor q = w.tensor("vision.blocks.1.attn.wqkv.weight");
    CHECK(q.type == vision::Tensor::Type::bf16 && q.rows == 96 && q.cols == 32);  // [out][in]
    const vision::Tensor b = w.tensor("aligner.w1.bias");
    CHECK(b.rows == 40 && b.cols == 1);
    std::size_t index = 0;
    for (; index < f.tensors.size(); ++index)
        if (f.tensors[index].name == "vision.blocks.1.attn.wqkv.weight") break;
    bool same = true;
    for (std::size_t e = 0; e < q.size(); e += 97) same = same && q.at(e) == vision::from_bf16(vision::to_bf16(File::value(index, e)));
    CHECK(same);
    CHECK(throws([&] { (void)w.tensor("nope"); }, "no tensor"));
    // A tensor that runs past the end of the file is refused, not read.
    gguf::FileInfo info = f.info();
    std::vector<std::byte> cut(w.bytes().begin(), w.bytes().end() - 64);
    vision::GgufWeights truncated(std::move(info), std::move(cut));
    CHECK(throws([&] { (void)truncated.tensor("vision.patch_embed.proj.weight"); }, "does not fit"));
}

void test_tables() {
    ds::VisionAdapter a(real_config(), kSha);
    const vision::EncoderConfig c = a.config();
    CHECK(c.layers == 32 && c.width == 1024 && c.heads == 16 && c.head_dim() == 64 && c.mlp_width == 2816);
    CHECK(c.norm == vision::EncoderConfig::Norm::rmsnorm && c.mlp == vision::EncoderConfig::Mlp::swiglu && c.mlp_out() == 5632);
    CHECK(c.qkv_bias && c.out_bias && !c.mlp_bias && c.patch_bias && c.final_norm && c.swiglu_limit == 0.0F);
    CHECK(c.pos == vision::EncoderConfig::Pos::rope2d && c.patch == 14 && c.patch_dim() == 588 && c.deepstack_taps.empty());
    vision::validate(c);
    std::string why;
    CHECK(vision::hip_supported(c, &why));
    const vision::ResizeRule r = a.resize();
    CHECK(r.multiple == 14 && r.min_pixels == 295936 && r.max_pixels == 1806336 && r.max_rows == 1024 && r.max_aspect == 0.0F);
    CHECK(a.span_rows(13, 13) == 184 && a.span_rows(31, 31) == 994 && a.span_rows(511, 1) == 1024 && a.max_span_rows() == 1024);
    CHECK(a.projector_weight_elements() == 5120ULL * 9216ULL + 5120ULL * 5120ULL);
    CHECK(a.device_options().split_activations);
    struct Row {
        int w, h, tw, th;
    };
    // Canvas sizes printed by the reference processor of the checkpoint (its resize plan) for these image sizes:
    // sides are multiples of the patch, small images grow to the minimum area, the whole span stays within 1024 rows,
    // and an extreme aspect ratio collapses to one column or one row of 3 x 3 cells.
    for (const Row t : {Row{546, 546, 546, 546}, Row{16, 16, 546, 546}, Row{1, 1, 546, 546}, Row{1600, 200, 1610, 210},
                        Row{200, 1600, 210, 1610}, Row{517, 389, 630, 476}, Row{1008, 756, 1008, 756},
                        Row{1344, 1344, 1302, 1302}, Row{4000, 3000, 1512, 1134}, Row{5000, 60, 5012, 70},
                        Row{10, 100000, 42, 21462}, Row{100000, 10, 42882, 42}, Row{544, 544, 546, 546},
                        Row{1343, 1345, 1288, 1302}, Row{300, 301, 546, 546}, Row{1302, 1302, 1302, 1302},
                        Row{1303, 1303, 1302, 1302}, Row{3, 4000, 14, 19866}, Row{1, 65000, 42, 21462},
                        Row{65000, 1, 42882, 42}}) {
        const vision::Size s = a.target(t.w, t.h);
        CHECK(s.width == t.tw && s.height == t.th);
        const unsigned mh = (static_cast<unsigned>(s.height) / 14U + 2U) / 3U, mw = (static_cast<unsigned>(s.width) / 14U + 2U) / 3U;
        CHECK(s.width % 14 == 0 && s.height % 14 == 0 && a.span_rows(mh, mw) <= a.max_span_rows());
    }
    CHECK(throws([&] { (void)a.target(0, 5); }, "empty image"));

    // Identity: the file digest and every recipe field are in it.
    ds::VisionRecipe other;
    other.swiglu_gate_first = false;
    ds::VisionRecipe mean;
    mean.mean = {0.485F, 0.456F, 0.406F};
    CHECK(a.identity() == ds::VisionAdapter(real_config(), kSha).identity());
    CHECK(a.identity() != ds::VisionAdapter(real_config(), std::string(64, 'b')).identity());
    CHECK(a.identity() != ds::VisionAdapter(real_config(), kSha, other).identity());
    CHECK(a.identity() != ds::VisionAdapter(real_config(), kSha, mean).identity());
    ds::VisionRecipe fill, exact_pixels;
    fill.pad_fill = 128;
    exact_pixels.pixels_bf16 = false;
    CHECK(a.identity() != ds::VisionAdapter(real_config(), kSha, fill).identity());
    CHECK(a.identity() != ds::VisionAdapter(real_config(), kSha, exact_pixels).identity());
    CHECK(a.identity().find("-r2-f127-b1") != std::string::npos);
    CHECK(a.identity().find(kSha) != std::string::npos && a.identity().find("deepseek41-vision/v4.1-flash/") == 0);
    CHECK(throws([] { ds::VisionAdapter bad(real_config(), "abc"); }, "digest"));
    ds::VisionConfig ratio = real_config();
    ratio.max_wh_ratio = 4;
    CHECK(throws([&] { ds::VisionAdapter bad(ratio, kSha); }, "ratio"));

    // Layout of the minimum image: start, 13 x (13 image rows + row end), end; 1D positions; one token id.
    vision::PatchGrid g;
    g.grid_h = g.grid_w = 39;
    g.rows = 39 * 39;
    const vision::SpanLayout s = a.layout(g);
    using Role = vision::SpanLayout::Role;
    CHECK(s.rows() == 184 && s.grid_h == 13 && s.grid_w == 13 && s.real_token_id == 129264);
    CHECK(s.role.front() == Role::start && s.role.back() == Role::end && s.role[1] == Role::image && s.role[13] == Role::image);
    CHECK(s.role[14] == Role::newline && s.role[15] == Role::image && s.role[182] == Role::newline);
    unsigned images = 0, newlines = 0;
    bool sequential = s.lm_pos.size() == s.role.size();
    for (std::uint32_t i = 0; i < s.rows(); ++i) {
        images += s.role[i] == Role::image;
        newlines += s.role[i] == Role::newline;
        sequential = sequential && s.lm_pos[i][0] == i && s.lm_pos[i][1] == 0 && s.lm_pos[i][2] == 0;
    }
    CHECK(images == 169 && newlines == 13 && sequential);
    ds::VisionRecipe open;
    open.newline_after_last_row = false;
    const vision::SpanLayout t = ds::VisionAdapter(real_config(), kSha, open).layout(g);
    CHECK(t.rows() == 183 && t.role[181] == Role::image && t.role[182] == Role::end);
    // A grid that is not a multiple of 3 has partial cells at the right and at the bottom: the merged grid rounds up.
    vision::PatchGrid odd;
    odd.grid_h = 38;
    odd.grid_w = 40;
    odd.rows = 38U * 40U;
    const vision::SpanLayout u = a.layout(odd);
    CHECK(u.grid_h == 13 && u.grid_w == 14 && u.rows() == 13U * 15U + 2U && u.role[14] == Role::image && u.role[15] == Role::newline);
    odd.rows = 39U * 40U;
    CHECK(throws([&] { (void)a.layout(odd); }, "rows differ"));
}

void test_patches_and_projector() {
    const File f = small_file();
    const ds::VisionModel model = ds::validate_vision(f.info());
    vision::GgufWeights source = f.weights();
    ds::VisionAdapter adapter(model.config, kSha);
    vision::EncoderWeights weights;
    adapter.bind(source, weights);
    const vision::EncoderConfig cfg = adapter.config();
    vision::validate(cfg, weights);

    // 18 x 12 pixels is already a legal size (cells of 6, 144 <= 216 pixels, 6 rows): no resample.
    vision::Image im;
    im.width = 18;
    im.height = 12;
    im.rgb.resize(18 * 12 * 3);
    for (std::size_t i = 0; i < im.rgb.size(); ++i) im.rgb[i] = static_cast<std::uint8_t>((i * 37U + 11U) % 256U);
    CHECK(adapter.target(18, 12) == (vision::Size{18, 12}));
    const vision::PatchGrid g = adapter.patches(im);
    CHECK(g.grid_h == 6 && g.grid_w == 9 && g.rows == 54 && g.patch_dim == 12 && g.tile_end.empty());
    bool order = g.data.size() == 54U * 12U && g.y.size() == 54 && g.x.size() == 54;
    for (unsigned r = 0; order && r < g.rows; ++r) {
        order = g.y[r] == r / 9U && g.x[r] == r % 9U;
        for (unsigned c = 0; c < 3; ++c)
            for (unsigned py = 0; py < 2; ++py)
                for (unsigned px = 0; px < 2; ++px) {  // feature (channel, y, x): the layout of the patch weight
                    const std::uint8_t v = im.rgb[((g.y[r] * 2U + py) * 18U + g.x[r] * 2U + px) * 3U + c];
                    // the reference order (a true division by 255), then the nearest BF16 value
                    const float want = vision::from_bf16(vision::to_bf16((static_cast<float>(v) / 255.0F - 0.5F) / 0.5F));
                    order = order && g.data[r * 12U + c * 4U + py * 2U + px] == want;
                }
    }
    CHECK(order);
    // A smaller image is scaled up to the minimum area; the grid always merges 3 x 3.
    vision::Image tiny;
    tiny.width = tiny.height = 4;
    tiny.rgb.assign(48, 200);
    const vision::PatchGrid up = adapter.patches(tiny);
    CHECK(up.grid_h == 6 && up.grid_w == 6 && up.rows == 36);
    // 16 x 9 pixels: the canvas is 16 x 10 (sides rounded up to the patch, not to the 3 x 3 cell); the image keeps
    // its size and sits on a grey canvas, half of the one-pixel margin rounded to even (0.5 -> 0: the top row is
    // image, the bottom row is grey). The 5 x 8 grid has partial cells: 2 x 3 merged rows.
    vision::Image wide;
    wide.width = 16;
    wide.height = 9;
    wide.rgb.resize(16 * 9 * 3);
    for (std::size_t i = 0; i < wide.rgb.size(); ++i) wide.rgb[i] = static_cast<std::uint8_t>((i * 29U + 3U) % 251U);
    CHECK(adapter.target(16, 9) == (vision::Size{16, 10}));
    const vision::PatchGrid pg = adapter.patches(wide);
    CHECK(pg.grid_h == 5 && pg.grid_w == 8 && pg.rows == 40);
    const float grey = vision::from_bf16(vision::to_bf16((127.0F / 255.0F - 0.5F) / 0.5F));
    bool canvas = pg.data.size() == 40U * 12U;
    for (unsigned r = 0; canvas && r < pg.rows; ++r)
        for (unsigned c = 0; c < 3; ++c)
            for (unsigned py = 0; py < 2; ++py)
                for (unsigned px = 0; px < 2; ++px) {
                    const unsigned y = (r / 8U) * 2U + py, x = (r % 8U) * 2U + px;
                    const float want = y < 9U ? vision::from_bf16(vision::to_bf16(
                                                    (static_cast<float>(wide.rgb[(y * 16U + x) * 3U + c]) / 255.0F - 0.5F) / 0.5F))
                                              : grey;
                    canvas = canvas && pg.data[r * 12U + c * 4U + py * 2U + px] == want;
                }
    CHECK(canvas);
    {
        // Partial cells are completed with zero rows before the first aligner linear: direct computation.
        vision::ReferenceEncoder tower(cfg, weights);
        const vision::EncoderOutput hidden = tower.encode(pg, adapter.positions(cfg, weights, pg));
        vision::ProjectedRows rows;
        vision::EncoderOutput copy = hidden;
        adapter.project(copy, pg, tower, rows);
        const vision::SpanLayout layout = adapter.layout(pg);
        CHECK(rows.rows == 10 && layout.rows() == 10 && layout.grid_h == 2 && layout.grid_w == 3);
        const vision::Tensor w1 = source.tensor("aligner.w1.weight"), b1 = source.tensor("aligner.w1.bias");
        const vision::Tensor w2 = source.tensor("aligner.w2.weight"), b2 = source.tensor("aligner.w2.bias");
        double worst = 0.0;
        for (unsigned my = 0; my < 2; ++my)
            for (unsigned mx = 0; mx < 3; ++mx) {
                std::vector<double> in(288, 0.0), mid(40);
                for (unsigned c = 0; c < 32; ++c)
                    for (unsigned dy = 0; dy < 3; ++dy)
                        for (unsigned dx = 0; dx < 3; ++dx)
                            if (my * 3U + dy < 5U && mx * 3U + dx < 8U)
                                in[c * 9U + dy * 3U + dx] =
                                    static_cast<double>(hidden.hidden[((my * 3U + dy) * 8U + mx * 3U + dx) * 32U + c]);
                for (unsigned o = 0; o < 40; ++o) {
                    double acc = static_cast<double>(b1.at(o));
                    for (unsigned k = 0; k < 288; ++k) acc += in[k] * static_cast<double>(w1.at(o * 288U + k));
                    const double x = static_cast<double>(static_cast<float>(acc));
                    mid[o] = static_cast<double>(static_cast<float>(0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440))));
                }
                for (unsigned o = 0; o < 40; ++o) {
                    double acc = static_cast<double>(b2.at(o));
                    for (unsigned k = 0; k < 40; ++k) acc += mid[k] * static_cast<double>(w2.at(o * 40U + k));
                    const double got = static_cast<double>(vision::from_bf16(rows.bf16[(1U + my * 4U + mx) * 40U + o]));
                    worst = std::max(worst, std::fabs(got - acc) / (std::fabs(acc) + 1.0));
                }
            }
        CHECK(worst < 1.0e-2);  // BF16 rows; a missing or misplaced zero row is an error of order one
    }

    vision::ReferenceEncoder backend(cfg, weights);
    const vision::Digest digest{};
    vision::EmbedCache cache;
    const vision::CachedEncode first = vision::encode_cached(adapter, backend, weights, &cache, digest, g);
    const vision::CachedEncode again = vision::encode_cached(adapter, backend, weights, &cache, digest, g);
    CHECK(!first.hit && again.hit && first.rows == again.rows);
    ds::VisionAdapter other(model.config, std::string(64, 'c'));
    other.bind(source, weights);
    CHECK(!vision::encode_cached(other, backend, weights, &cache, digest, g).hit);  // another file digest: another key

    // The projector against a direct computation from the tower output and the raw tensors.
    const vision::ProjectedRows& rows = *first.rows;
    const vision::SpanLayout layout = adapter.layout(g);
    CHECK(rows.rows == 10 && rows.width == 40 && layout.rows() == rows.rows && rows.deep.empty());  // 1 + 2 x (3 + 1) + 1
    const vision::EncoderOutput hidden = backend.encode(g, adapter.positions(cfg, weights, g));
    const vision::Tensor w1 = source.tensor("aligner.w1.weight"), b1 = source.tensor("aligner.w1.bias");
    const vision::Tensor w2 = source.tensor("aligner.w2.weight"), b2 = source.tensor("aligner.w2.bias");
    double worst = 0.0;
    for (unsigned my = 0; my < 2; ++my)
        for (unsigned mx = 0; mx < 3; ++mx) {
            std::vector<double> in(288), mid(40);
            for (unsigned c = 0; c < 32; ++c)
                for (unsigned dy = 0; dy < 3; ++dy)
                    for (unsigned dx = 0; dx < 3; ++dx)
                        in[c * 9U + dy * 3U + dx] = static_cast<double>(hidden.hidden[((my * 3U + dy) * 9U + mx * 3U + dx) * 32U + c]);
            for (unsigned o = 0; o < 40; ++o) {
                double acc = static_cast<double>(b1.at(o));
                for (unsigned k = 0; k < 288; ++k) acc += in[k] * static_cast<double>(w1.at(o * 288U + k));
                const double x = static_cast<double>(static_cast<float>(acc));
                mid[o] = static_cast<double>(static_cast<float>(0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440))));
            }
            const unsigned row = 1U + my * 4U + mx;
            CHECK(layout.role[row] == vision::SpanLayout::Role::image);
            for (unsigned o = 0; o < 40; ++o) {
                double acc = static_cast<double>(b2.at(o));
                for (unsigned k = 0; k < 40; ++k) acc += mid[k] * static_cast<double>(w2.at(o * 40U + k));
                const double got = static_cast<double>(vision::from_bf16(rows.bf16[row * 40U + o]));
                worst = std::max(worst, std::fabs(got - acc) / (std::fabs(acc) + 1.0));
            }
        }
    CHECK(worst < 6e-3);  // BF16 storage of the rows: 8 significant bits
    // Special rows are the file's tensors, bit for bit, at the layout's positions.
    const auto is_row = [&](unsigned row, const char* name) {
        const vision::Tensor t = source.tensor(name);
        for (unsigned o = 0; o < 40; ++o)
            if (rows.bf16[row * 40U + o] != vision::to_bf16(t.at(o))) return false;
        return true;
    };
    CHECK(is_row(0, "image_start") && is_row(4, "image_newline") && is_row(8, "image_newline") && is_row(9, "image_end"));
    CHECK(layout.role[0] == vision::SpanLayout::Role::start && layout.role[4] == vision::SpanLayout::Role::newline &&
          layout.role[9] == vision::SpanLayout::Role::end);

    // The unshuffle option really changes the aligner input; the default is the row-major one checked above.
    ds::VisionRecipe col;
    col.unshuffle_row_major = false;
    ds::VisionAdapter swapped(model.config, kSha, col);
    swapped.bind(source, weights);
    const vision::CachedEncode third = vision::encode_cached(swapped, backend, weights, nullptr, digest, g);
    CHECK(third.rows->bf16 != rows.bf16 && third.rows->rows == rows.rows);

    // Misuse is refused.
    ds::VisionAdapter unbound(model.config, kSha);
    vision::EncoderOutput h2 = hidden;
    vision::ProjectedRows out;
    CHECK(throws([&] { unbound.project(h2, g, backend, out); }, "before bind"));
    vision::PatchGrid wrong = g;
    wrong.grid_w = 8;
    CHECK(throws([&] { adapter.project(h2, wrong, backend, out); }, "patch grid"));
    CHECK(throws([&] { (void)adapter.patches(vision::Image{}); }, "empty image"));
}

// Malformed inputs an external audit found accepted (each of these failed before its fix).
void test_refusals() {
    // Minimum complete span must be rejected before resize floating-point arithmetic.
    for (unsigned budget = 1; budget < 4; ++budget) {
        auto small = real_config();
        small.max_image_tokens = budget;
        small.min_pixels = 1764;
        CHECK(throws([&] { ds::VisionAdapter bad(small, kSha); }, "four span rows"));
    }
    auto minimum = real_config();
    minimum.max_image_tokens = 4;
    minimum.min_pixels = 1764;
    CHECK(ds::VisionAdapter(minimum, kSha).target(42, 42) == (vision::Size{42, 42}));

    // Image rows alone fit 1024; the complete span does not, including partial merge cells.
    const File grid_file = small_file();
    auto grid_config = ds::validate_vision(grid_file.info()).config;
    grid_config.max_image_tokens = 1024;
    ds::VisionAdapter grid_adapter(grid_config, kSha);
    auto grid_weights = grid_file.weights();
    vision::EncoderWeights encoder_weights;
    grid_adapter.bind(grid_weights, encoder_weights);
    vision::ReferenceEncoder grid_dense(grid_adapter.config(), encoder_weights);
    for (unsigned side : {95U, 96U}) {
        vision::PatchGrid grid;
        grid.grid_h = grid.grid_w = side;
        grid.rows = side * side;
        vision::EncoderOutput hidden;
        hidden.width = grid_config.hidden_size;
        hidden.rows = grid.rows;
        hidden.hidden.assign(std::size_t{hidden.rows} * hidden.width, 0.0F);
        vision::ProjectedRows out;
        for (bool coordinates : {false, true}) {
            if (coordinates) {
                grid.y.resize(grid.rows);
                grid.x.resize(grid.rows);
                for (unsigned row = 0; row < grid.rows; ++row) {
                    grid.y[row] = static_cast<std::uint16_t>(row / side);
                    grid.x[row] = static_cast<std::uint16_t>(row % side);
                }
            }
            CHECK(throws([&] { (void)grid_adapter.layout(grid); }, "span exceeds"));
            CHECK(throws([&] { grid_adapter.project(hidden, grid, grid_dense, out); }, "span exceeds"));
            CHECK(out.bf16.empty());
        }
    }
    // Recipe identity: neighbouring representable floats are different recipes, non-finite values are refused.
    ds::VisionRecipe base, near_mean, near_std;
    near_mean.mean[0] = std::nextafter(base.mean[0], 1.0F);
    near_std.stddev[2] = std::nextafter(base.stddev[2], 1.0F);
    CHECK(base.id() != near_mean.id() && base.id() != near_std.id() && near_mean.id() != near_std.id());
    CHECK(base.id() == ds::VisionRecipe{}.id());
    ds::VisionRecipe nan_mean, inf_std;
    nan_mean.mean[1] = std::numeric_limits<float>::quiet_NaN();
    inf_std.stddev[0] = std::numeric_limits<float>::infinity();
    CHECK(throws([&] { ds::VisionAdapter bad(real_config(), kSha, nan_mean); }, "recipe mean"));
    CHECK(throws([&] { ds::VisionAdapter bad(real_config(), kSha, inf_std); }, "recipe std"));

    // A reference with a NaN is a failed comparison, never a silent pass.
    const std::vector<float> got{1.0F, 2.0F, 3.0F};
    std::vector<float> want = got;
    CHECK(vision::abs_diff(got, want, 1e-6).pass);
    want[1] = std::numeric_limits<float>::quiet_NaN();
    const vision::AbsDiff nan_ref = vision::abs_diff(got, want, 1e-6);
    CHECK(!nan_ref.pass && nan_ref.non_finite == 1);
    CHECK(!vision::abs_diff(want, got, 1e-6).pass && !vision::abs_diff(want, want, 1e-6).pass);
    want[1] = std::numeric_limits<float>::infinity();
    CHECK(!vision::abs_diff(got, want, 1e-6).pass);
    want[1] = 2.5F;
    const vision::AbsDiff off = vision::abs_diff(got, want, 1e-6);
    CHECK(!off.pass && off.over == 1 && off.non_finite == 0 && off.worst == 0.5);
    CHECK(!vision::abs_diff(std::span<const float>(got).first(2), want, 1e-6).pass);

    // A legal file geometry (patch 1, ratio 8, 16384 rows) admits an 8 x 65544 image without a resize; its patch
    // row is wider than a patch coordinate can state.
    ds::VisionConfig thin = real_config();
    thin.patch_size = 1;
    thin.downsample_ratio = 8;
    thin.max_image_tokens = 16384;
    thin.min_pixels = 64;
    const ds::VisionAdapter thin_adapter(thin, kSha);
    vision::Image strip;
    strip.width = 65544;
    strip.height = 8;
    strip.rgb.assign(std::size_t{65544} * 8U * 3U, 0);
    CHECK(thin_adapter.target(strip.width, strip.height) == (vision::Size{65544, 8}));
    CHECK(throws([&] { (void)thin_adapter.patches(strip); }, "axis"));

    // Grid products that wrap 32 bits: 196608 = 3 * 65536 per axis, so rows and merged tokens wrap to 0.
    const ds::VisionAdapter a(real_config(), kSha);
    vision::PatchGrid wrap;
    wrap.grid_h = wrap.grid_w = 196608;
    wrap.rows = wrap.grid_h * wrap.grid_w;
    CHECK(wrap.rows == 0);
    CHECK(throws([&] { (void)a.layout(wrap); }, "patch grid"));
    vision::PatchGrid many;  // every product fits, the row count is over the file's limit
    many.grid_h = many.grid_w = 99;
    many.rows = 99 * 99;
    CHECK(throws([&] { (void)a.layout(many); }, "image rows"));
    vision::PatchGrid zero;
    CHECK(throws([&] { (void)a.layout(zero); }, "patch grid"));
    vision::PatchGrid lying;  // rows that are not grid_h * grid_w
    lying.grid_h = lying.grid_w = 39;
    lying.rows = 3;
    CHECK(throws([&] { (void)a.layout(lying); }, "patch grid"));

    const File f = small_file();
    const ds::VisionModel model = ds::validate_vision(f.info());
    vision::GgufWeights source = f.weights();
    ds::VisionAdapter bound(model.config, kSha);
    vision::EncoderWeights weights;
    bound.bind(source, weights);
    vision::ReferenceEncoder backend(bound.config(), weights);
    vision::EncoderOutput empty;
    empty.width = model.config.hidden_size;
    vision::ProjectedRows out;
    CHECK(throws([&] { bound.project(empty, wrap, backend, out); }, "patch grid"));
    CHECK(throws([&] { bound.project(empty, zero, backend, out); }, "patch grid"));

    // A weight file's length is checked against the host budget and the admitted length before anything is read.
    const std::string good = f.bytes();
    const std::filesystem::path dir = std::filesystem::temp_directory_path();
    const std::filesystem::path good_path = dir / "strix-ds41-vision-test.gguf", junk_path = dir / "strix-ds41-vision-junk.gguf";
    const auto put_file = [](const std::filesystem::path& p, const std::string& bytes) {
        std::ofstream o(p, std::ios::out | std::ios::binary | std::ios::trunc);
        o.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    };
    put_file(good_path, good);
    put_file(junk_path, std::string(4096, '\x5a'));
    const gguf::Limits limits = ds::vision_reader_limits();
    vision::HostBudget exact, under, other;
    exact.exact_bytes = good.size();
    under.max_bytes = good.size() - 1U;
    other.exact_bytes = good.size() + 32U;
    {
        vision::GgufWeights from_file(good_path, limits, exact);
        CHECK(from_file.bytes().size() == good.size() && std::memcmp(from_file.bytes().data(), good.data(), good.size()) == 0);
        CHECK(from_file.has("image_start") && from_file.info().tensors.size() == 26);
        vision::HostBudget fits;
        fits.max_bytes = good.size();
        CHECK(vision::GgufWeights(good_path, limits, fits).bytes().size() == good.size());
    }
    CHECK(throws([&] { vision::GgufWeights w(good_path, limits, under); }, "host budget"));
    CHECK(throws([&] { vision::GgufWeights w(good_path, limits, other); }, "admitted file"));
    vision::HostBudget tiny;
    tiny.max_bytes = 1024;
    CHECK(throws([&] { vision::GgufWeights w(junk_path, limits, tiny); }, "host budget"));  // refused unread, not parsed
    bool malformed = false;  // inside the budget: the header is what refuses it
    try {
        vision::GgufWeights w(junk_path, limits);
    } catch (const std::exception& e) {
        malformed = std::strstr(e.what(), "host budget") == nullptr;
    }
    CHECK(malformed);
    CHECK(throws([&] { vision::GgufWeights w(dir / "strix-ds41-vision-absent.gguf", limits); }, "cannot open"));
    std::error_code ignored;
    std::filesystem::remove(good_path, ignored);
    std::filesystem::remove(junk_path, ignored);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // a crash keeps the failures printed before it
    try {
        test_validate();
        test_weight_source();
        test_tables();
        test_patches_and_projector();
        test_refusals();
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf("deepseek41-vision: %s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
