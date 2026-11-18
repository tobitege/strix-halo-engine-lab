// vision::ApiMediaBridge: a real fixture image through decode -> patches -> encode (host reference) -> interned
// span; a repeated image is a hit without decode; WebP and over-budget images are refused with a client error.
#include "strix/vision/api_media.hpp"

#include "vision_encoder_toy.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <new>
#include <string>

namespace {

int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

using namespace strix;
using namespace strix::vision;

// The toy tower behind a real image path: a fixed 2 x 2 grid of the decoded image's size class.
class ImageToy final : public VisionAdapter {
public:
    explicit ImageToy(vision_toy::Tower& t) : tower_(t), inner_(t) {}
    EncoderConfig config() const override { return inner_.config(); }
    std::string identity() const override { return inner_.identity(); }
    ResizeRule resize() const override { return inner_.resize(); }
    PatchGrid patches(const Image& image) const override {
        ++decoded;
        return vision_toy::make_grid(tower_.cfg, 2, 2, static_cast<std::uint64_t>(image.width) * 131U + static_cast<std::uint64_t>(image.height));
    }
    void bind(WeightSource& s, EncoderWeights& w) override { inner_.bind(s, w); }
    void project(EncoderOutput& h, const PatchGrid& g, DenseOps& d, ProjectedRows& out) override { inner_.project(h, g, d, out); }
    SpanLayout layout(const PatchGrid& g) const override { return inner_.layout(g); }
    mutable unsigned decoded = 0;
    unsigned projections() const { return inner_.projections; }

private:
    vision_toy::Tower& tower_;
    vision_toy::ToyAdapter inner_;
};

EncoderConfig toy_config() {
    EncoderConfig c;
    c.layers = 1;
    c.width = 16;
    c.heads = 2;
    c.mlp_width = 24;
    c.patch = 2;
    c.channels = 3;
    c.norm = EncoderConfig::Norm::layernorm;
    c.mlp = EncoderConfig::Mlp::gelu;
    c.qkv_bias = c.out_bias = c.mlp_bias = true;
    c.pos = EncoderConfig::Pos::learned_interp;
    c.pos_grid = 4;
    return c;
}

serve::media::MediaItem item_of(const std::string& bytes, std::uint8_t tag) {
    serve::media::MediaItem m;
    m.bytes = bytes;
    m.digest.fill(tag);
    return m;
}

template <class F>
int status_of(F&& f) {
    try { f(); } catch (const serve::media::MediaError& e) { return e.status; } catch (const std::exception&) { return -1; }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef STRIX_VISION_FIXTURE_DIR
    const std::string dir = argc >= 2 ? argv[1] : STRIX_VISION_FIXTURE_DIR;
#else
    if (argc < 2) { std::fprintf(stderr, "usage: vision-api-media FIXTURE_DIR\n"); return 2; }
    const std::string dir = argv[1];
#endif
    std::ifstream f(dir + "/rgb_noise.png", std::ios::binary);
    const std::string png((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(!png.empty());

    auto tower = vision_toy::make_tower(toy_config(), 5, 8);
    ImageToy adapter(tower);
    ReferenceEncoder backend(tower.cfg, tower.w);
    EmbedCache cache;
    session::EmbedInterner interner;
    unsigned loads = 0;
    ApiMediaOptions o;
    o.placeholder = "<image>";
    o.max_rows = 64;
    o.lm_width = 8;
    o.on_first_encode = [&] { ++loads; };
    ApiMediaBridge bridge(adapter, backend, tower.w, &cache, interner, o);
    CHECK(bridge.identity().find(backend.arithmetic()) != std::string::npos);

    std::vector<std::uint16_t> rows1;
    std::uint32_t handle1 = 0;
    {
        const auto s1 = bridge.encode(item_of(png, 1), 64);
        CHECK(s1.lease && s1.layout == s1.lease.ids() && s1.layout.size() == 4U);
        CHECK(!s1.cache_hit && s1.grid_h == 2U && s1.grid_w == 2U && adapter.decoded == 1U && loads == 1U);
        CHECK(s1.lease.rows().width == 8U && s1.lease.rows().real_token_id == 7U);
        // live lease: no decode, same ids
        const auto s2 = bridge.encode(item_of(png, 1), 64);
        CHECK(s2.cache_hit && s2.layout == s1.layout && adapter.decoded == 1U && adapter.projections() == 1U);
        rows1 = s1.lease.rows().data;
        handle1 = s1.lease.handle();
    }
    // every lease dropped: the EmbedCache rows are re-interned, still without a decode or an encode
    const auto s3 = bridge.encode(item_of(png, 1), 64);
    CHECK(s3.cache_hit && adapter.decoded == 1U && adapter.projections() == 1U && s3.lease.rows().data == rows1);
    CHECK(s3.lease.handle() == handle1);  // same digest -> same handle within an epoch
    // a different digest is a new encode
    const auto s4 = bridge.encode(item_of(png, 2), 64);
    CHECK(!s4.cache_hit && adapter.decoded == 2U && s4.lease.handle() != s3.lease.handle());
    // over the row budget: 400 before the encoder runs
    CHECK(status_of([&] { (void)bridge.encode(item_of(png, 3), 3); }) == 400);
    CHECK(adapter.projections() == 2U);
    // WebP stays refused with a client error naming the formats that work
    const std::string webp = std::string("RIFF\x24\x00\x00\x00WEBPVP8 ", 16) + std::string(24, '\0');
    int webp_status = 0;
    try { (void)bridge.encode(item_of(webp, 4), 64); } catch (const serve::media::MediaError& e) {
        webp_status = e.status;
        CHECK(std::string(e.what()).find("PNG or JPEG") != std::string::npos);
    }
    CHECK(webp_status == 400);
    CHECK(bridge.stats().refused == 2U);

    // A failed lazy upload releases partial panels and retries before any backend encode.
    unsigned attempts = 0, rollbacks = 0, panels = 0;
    o.on_first_encode = [&] {
        ++attempts;
        CHECK(panels == 0U);
        panels = 1U;
        if (attempts == 1U) throw std::bad_alloc();
    };
    o.on_first_encode_failure = [&] { ++rollbacks; panels = 0U; };
    session::EmbedInterner retry_interner;
    ApiMediaBridge retry(adapter, backend, tower.w, nullptr, retry_interner, o);
    const auto projections = adapter.projections();
    bool allocation_failed = false;
    try { (void)retry.encode(item_of(png, 5), 64); }
    catch (const std::bad_alloc&) { allocation_failed = true; }
    CHECK(allocation_failed && attempts == 1U && rollbacks == 1U && panels == 0U);
    CHECK(adapter.projections() == projections && retry.stats().encodes == 0U);
    const auto recovered = retry.encode(item_of(png, 5), 64);
    CHECK(recovered.lease && attempts == 2U && rollbacks == 1U && panels == 1U);
    const auto fresh = retry.encode(item_of(png, 6), 64);
    CHECK(fresh.lease && attempts == 2U && retry.stats().encodes == 2U);

    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::puts("vision-api-media: ok");
    return 0;
}
