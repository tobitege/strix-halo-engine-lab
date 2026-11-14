// DeepSeek V4.1 image rows towards the prefill: layout() -> EmbedRows -> virtual ids -> EmbedSplice, and the memory
// image input adds to a slot capacity report.

#include "../src/deepseek41/vision_adapter.hpp"
#include "../src/deepseek41/vision_splice.hpp"

#include "strix/deepseek41_model.hpp"
#include "strix/serve/slot_capacity.hpp"
#include "strix/session/embed_rows.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strix;
namespace ds = strix::deepseek41;
namespace ss = strix::session;

int failures = 0;
void check(bool ok, const char* what, int line) {
    if (ok) return;
    ++failures;
    std::cerr << "FAIL line " << line << ": " << what << "\n";
}
#define CHECK(x) check((x), #x, __LINE__)

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

void span_to_splice() {
    ds::VisionAdapter adapter(real_config(), std::string(64, 'a'));
    vision::PatchGrid grid;
    grid.grid_h = 6;
    grid.grid_w = 9;
    grid.rows = 54;
    const unsigned n = adapter.span_rows(2, 3); // merged 2 x 3
    vision::ProjectedRows rows;
    rows.rows = n;
    rows.width = 5120;
    rows.bf16.resize(std::size_t{n} * 5120);
    for (std::size_t i = 0; i < rows.bf16.size(); ++i) rows.bf16[i] = static_cast<std::uint16_t>(i * 7U);
    ss::EmbedDigest digest{};
    digest[0] = 9;
    const ss::EmbedRows span = ds::vision_embed_rows(adapter, grid, rows, digest);
    CHECK(span.rows == n && span.width == 5120 && span.axes == 1 && span.real_token_id == 129264);
    CHECK(span.role.front() == ss::EmbedRole::begin && span.role.back() == ss::EmbedRole::end && span.deep.empty());
    CHECK(span.lm_pos.size() == n && span.lm_pos.front() == 0 && span.lm_pos.back() == static_cast<std::int32_t>(n) - 1);
    CHECK(span.data == rows.bf16);

    ss::EmbedInterner interner;
    const ss::EmbedLease lease = interner.intern(span);
    const auto table = std::make_shared<const ss::EmbedTable>(std::vector<ss::EmbedLease>{lease});
    // the renderer's placeholder (the image token) becomes the span's virtual ids
    const std::vector<std::uint32_t> rendered{1, 50, 129264, 51};
    const std::vector<std::vector<std::uint32_t>> layouts{lease.ids()};
    const ss::ExpandedPrompt prompt = ss::expand_embed_rows(rendered, 129264, layouts);
    CHECK(prompt.tokens.size() == 3 + n);
    const ss::ResolvedEmbedRun run = ss::resolve_embed_run(prompt.tokens, table, ds::vision_resolve_rules(adapter));
    CHECK(run.tokens.size() == prompt.tokens.size() && run.tokens[0] == 1 && run.tokens[1] == 50 && run.tokens.back() == 51);
    for (unsigned r = 0; r < n; ++r) CHECK(run.tokens[2 + r] == 129264);
    CHECK(run.splice.segments.size() == 1 && run.splice.segments[0].run_row == 2 && run.splice.segments[0].rows == n);
    CHECK(run.splice.segments[0].extras && run.splice.width == 5120);
    run.splice.validate(run.tokens.size(), 5120);

    vision::ProjectedRows shorter = rows;
    shorter.rows = n - 1;
    bool refused = false;
    try {
        (void)ds::vision_embed_rows(adapter, grid, shorter, digest);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    CHECK(refused);
}

void memory_terms() {
    ds::VisionAdapter adapter(real_config(), std::string(64, 'a'));
    const ds::VisionMemory m = ds::vision_memory(adapter, 4096);
    const vision::EncoderPlan p = vision::plan(adapter.config(), 9216, 1, adapter.device_options());
    CHECK(m.tower_weights == p.weight_bytes && m.tower_workspace == p.workspace_bytes() && m.tower_weights != 0);
    CHECK(m.projector_panels == adapter.projector_weight_elements() * 2U);
    CHECK(m.projector_panels / (1U << 20U) == 140);  // (9216 + 5120) x 5120 FP16
    CHECK(m.splice_staging == 4096ULL * 5120U * 2U + 256U);
    CHECK(m.rows_per_image == std::uint64_t{adapter.max_span_rows()} * 5120U * 2U);
    CHECK(m.splice_growth == m.splice_staging);
    CHECK(m.device_once() == m.tower_weights + m.tower_workspace + m.projector_panels + 2U * m.splice_staging);

    const std::uint64_t GiB = 1ULL << 30U;
    serve::CapacityPool device{"device", 8 * GiB, 1 * GiB, {GiB / 4, 0}};
    serve::CapacityPool host{"host", 64 * GiB, 1 * GiB, {1 * GiB, 0}};
    const auto before = serve::report_slot_capacity({device, host});
    ds::add_vision_capacity(device, host, m, 4);
    const auto after = serve::report_slot_capacity({device, host});
    CHECK(device.headroom_bytes == 1 * GiB + m.device_once());
    CHECK(host.cost.retained_bytes == 1 * GiB + 4 * m.rows_per_image);
    CHECK(before.slots >= 2 && after.slots < before.slots && after.slots >= 1);
    CHECK(after.limiting == before.limiting);
    CHECK(after.terms.size() == 2 && after.terms[0].headroom_bytes == device.headroom_bytes);
}
}  // namespace

int main() {
    span_to_splice();
    memory_terms();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "deepseek41 vision splice tests passed\n";
    return 0;
}
