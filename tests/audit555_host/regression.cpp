// Test doubles execute the real CLI gate and flag replacement; no HIP, model file or full build.
#include "strix/deepseek41_prefill.hpp"
#include "strix/ds41_engram.hpp"
#include "../../src/deepseek41/vision_splice.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> bool rejected(F&& f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}
enum class Fault { none, splice, router, dead, both_extras, arm_hashes, stream_only };
Fault fault = Fault::none;
bool identical = false;
}

namespace strix::ds41 {
#include "engram.inc"
}

namespace strix::deepseek41 {
struct DevicePrefill::Impl {
    PrefillOptions options;
    session::EmbedSplice splice;
    std::vector<std::pair<std::string, std::uint64_t>> state;
};
DevicePrefill::DevicePrefill(const std::filesystem::path&, const PrefillOptions& o) : impl_(std::make_unique<Impl>()) {
    impl_->options = o;
}
DevicePrefill::~DevicePrefill() = default;
void DevicePrefill::set_embed_rows(session::EmbedSplice s) { impl_->splice = std::move(s); }
std::vector<std::uint16_t> DevicePrefill::embedding_rows(std::span<const std::uint32_t> tokens) const {
    std::vector<std::uint16_t> data;
    for (auto token : tokens) data.push_back(static_cast<std::uint16_t>(0x3c00U + (identical ? 0U : token * 8U)));
    return data;
}
std::vector<std::pair<std::string, std::uint64_t>> DevicePrefill::state_hashes() const { return impl_->state; }
PrefillResult DevicePrefill::run(std::span<const std::uint32_t> tokens) {
    auto splice = std::exchange(impl_->splice, {});
    auto data = embedding_rows(tokens);
    bool image = false;
    for (const auto& segment : splice.segments) {
        image = image || segment.extras;
        if (fault == Fault::splice) continue;
        for (std::uint32_t i = 0; i < segment.rows; ++i) {
            auto bits = segment.data[i];
            // Exact normal finite conversion used by this fixture's table rows.
            if (splice.format == session::EmbedRowFormat::bf16)
                bits = static_cast<std::uint16_t>((static_cast<unsigned>(bits) - 0x3800U) << 3U);
            data[segment.run_row + i] = bits;
        }
    }
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < data.size(); ++i) sum += (i + 1U) * data[i];
    PrefillResult r;
    r.embed_image_router_checked = image && fault != Fault::router && fault != Fault::both_extras;
    r.embed_dead_hash_checked = image && fault != Fault::dead && fault != Fault::both_extras;
    if (r.embed_image_router_checked || r.embed_dead_hash_checked) sum += 100;
    r.logits = {static_cast<float>(sum)};
    r.next_token = static_cast<std::uint32_t>(sum % 31U);
    impl_->state = {{"state", sum}};
    if (impl_->options.stream_hashes && !(fault == Fault::arm_hashes && !splice.empty()))
        r.stream_hashes.assign(impl_->options.layers, sum + (fault == Fault::stream_only && !splice.empty() ? 1U : 0U));
    return r;
}
}

#include "gate.inc"

namespace {
void flag_regression() {
    struct Storage {
        std::vector<std::uint8_t> bytes = std::vector<std::uint8_t>(512, 0);
        void zero(int) { std::fill(bytes.begin(), bytes.end(), std::uint8_t{0}); }
        void upload(const std::vector<std::uint8_t>& data, int) { std::copy(data.begin(), data.end(), bytes.begin()); }
    };
    struct Memory { Storage image_rows; bool image_rows_set = false; } m;
    struct Config { std::size_t hidden_size = 1; } c;
    const int st = 0;
    const auto replace = [&](const strix::session::EmbedSplice& embeds, std::uint32_t Tn) {
        const auto T = Tn;
        const auto embed_extras = embeds.extras_rows(T);
        std::vector<std::uint32_t> global_row(T);
        for (std::uint32_t i = 0; i < T; ++i) global_row[i] = i;
#include "flags.inc"
        (void)c;
    };
    const std::uint16_t row = 0;
    strix::session::EmbedSplice image;
    image.width = 1;
    image.segments.push_back({400, 1, &row, true});
    for (bool short_splice : {false, true}) {
        replace(image, 512);
        require(m.image_rows.bytes[400] == 1, "image row not installed");
        strix::session::EmbedSplice short_text;
        short_text.width = 1;
        if (short_splice) short_text.segments.push_back({0, 1, &row, false});
        replace(short_text, 128);
        replace({}, 512);
        require(std::all_of(m.image_rows.bytes.begin(), m.image_rows.bytes.end(), [](auto x) { return x == 0; }),
                "long image / short text or splice / long text left a stale flag");
        require(!m.image_rows_set, "clean flag buffer remained dirty");
    }
}

int run_gate(Args a, std::vector<std::uint32_t> prompt = {1, 2, 3, 4}) {
    strix::deepseek41::DevicePrefill pre({}, options(a, prompt.size()));
    const auto reference = pre.run(prompt);
    return embed_identity_gate(a, pre, prompt, reference);
}
Args gate_args() {
    Args a;
    a.embed_identity = true;
    a.layers = 2;
    a.embed_a = 1;
    a.embed_b = 4;
    return a;
}
void control_regression() {
    auto a = gate_args();
    require(run_gate(a) == 0, "valid distinct control did not pass");
    a.embed_b = 2;
    require(run_gate(a) != 0, "one-row control passed");
    a.embed_b = 4;
    identical = true;
    require(run_gate(a) != 0, "identical-row control passed");
    identical = false;
    fault = Fault::splice;
    require(run_gate(a) != 0, "bypassed substitution passed");
}
void image_regression() {
    for (const auto f : {Fault::router, Fault::dead, Fault::both_extras}) {
        fault = f;
        require(run_gate(gate_args()) != 0, "missing image semantics passed");
    }
}
void hash_regression() {
    auto a = gate_args();
    require(options(a, 4).stream_hashes, "identity did not enable hashes without reference/output file");
    strix::deepseek41::DevicePrefill pre({}, options(a, 4));
    const std::vector<std::uint32_t> prompt{1, 2, 3, 4};
    auto ref = pre.run(prompt);
    ref.stream_hashes.clear();
    require(rejected([&] { (void)embed_identity_gate(a, pre, prompt, ref); }), "empty reference hashes passed");
    ref.stream_hashes = {1};
    require(rejected([&] { (void)embed_identity_gate(a, pre, prompt, ref); }), "partial reference hashes passed");
    fault = Fault::arm_hashes;
    require(rejected([&] { (void)run_gate(a); }), "empty arm hashes passed");
    fault = Fault::stream_only;
    require(run_gate(a) != 0, "stream-only mismatch passed");
}
void dispatch_regression() {
    auto a = gate_args();
    validate_embed_identity_args(a);
    a.ttft = true;
    require(rejected([&] { validate_embed_identity_args(a); }), "timing silently discarded gate");
    a.ttft = false;
    a.world = 2;
    require(rejected([&] { validate_embed_identity_args(a); }), "world 2 silently discarded gate");
    a.embed_identity = false;
    validate_embed_identity_args(a);
}
void engram_regression() {
    strix::deepseek41::EngramContract c;
    c.layer_ids = {0}; c.max_ngram_size = 3; c.n_heads = 1;
    c.primes = {101, 103}; c.rows = {204}; c.multipliers = {3, 5, 7};
    c.token_map = {1, 2, 3, 4}; c.pad_id = 3; c.compressed_vocab_size = 5;
    strix::ds41::EngramHasher h(c);
    const auto dead = h.hash_sequence(std::vector<std::uint32_t>{0, strix::ds41::engram_dead_token(1)});
    const auto pad = h.hash_sequence(std::vector<std::uint32_t>{3, 3, 3});
    const auto plain = h.hash_sequence(std::vector<std::uint32_t>{0, 1});
    require(dead[2] == pad[4] && dead[3] == pad[5], "dead row differs from full-padding oracle");
    require(dead[2] != plain[2] || dead[3] != plain[3], "dead fixture has no observable effect");
}
void capacity_regression() {
    strix::deepseek41::VisionMemory m;
    m.tower_weights = 1; m.tower_workspace = 2; m.projector_panels = 3;
    m.splice_staging = 4; m.splice_growth = 4;
    require(m.device_once() == 14, "admission omitted old allocation during growth");
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "one test case required");
        const std::string name = argv[1];
        if (name == "flags") flag_regression();
        else if (name == "control") control_regression();
        else if (name == "image") image_regression();
        else if (name == "hashes") hash_regression();
        else if (name == "dispatch") dispatch_regression();
        else if (name == "engram") engram_regression();
        else if (name == "capacity") capacity_regression();
        else throw std::runtime_error("unknown case");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
