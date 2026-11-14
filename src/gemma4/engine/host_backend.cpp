#include "host_backend.hpp"
#include <algorithm>
#include <cstdlib>

namespace strix::gemma4::engine {
namespace {
class HostState final : public StreamState {
public:
    explicit HostState(const ModelConfig& config) : state_(config) {}
    explicit HostState(std::unique_ptr<HostStream> state) : state_(std::move(*state)) {}
    std::unique_ptr<StreamState> fork(std::uint32_t admitted) const override {
        if (admitted < frontier()) throw std::invalid_argument("host snapshot capacity below frontier");
        return std::make_unique<HostState>(state_.snapshot());
    }
    std::unique_ptr<KvTxn> begin() override { return state_.begin(); }
    std::uint64_t frontier() const override { return state_.frontier(); }
    std::string digest() const override { return state_.digest(); }
    std::optional<std::uint64_t> retained_bytes() const override {
        const auto bytes=state_.retained_bytes();
        if(bytes>UINT64_MAX-(sizeof(HostState)-sizeof(HostStream))) throw std::overflow_error("host retained size overflow");
        return bytes+sizeof(HostState)-sizeof(HostStream);
    }
    bool kv_view(std::uint32_t layer, const KvTxn* transaction, KvView& out) const override {
        // "No KV on this layer"/poisoned = unavailable; a foreign transaction (std::invalid_argument) propagates.
        try { out = state_.kv_view(layer,transaction); return true; } catch (const std::runtime_error&) { return false; }
    }
private:
    HostStream state_;
};
class HostDraft final : public DraftSession {
public:
    HostDraft(const dflash::Weights& weights, dflash::RopeView rope, LinearView embedding, LinearView head)
        : weights_(weights), rope_(rope), embedding_(embedding), head_(head), context_(weights.shape) {}
    void append(std::span<const float> taps, std::span<const std::uint64_t> positions) override {
        context_.append(weights_,rope_,taps,positions);
    }
    std::array<std::uint32_t,7> propose(std::uint32_t anchor, std::uint64_t position) override {
        try {
            const auto embedded = dflash::embed_block(anchor,embedding_);
            const auto block_hidden = dflash::block_forward(weights_,context_,rope_,embedded,position);
            return dflash::proposals_from_hidden(weights_,block_hidden,head_);
        } catch (const std::exception& error) {
            throw std::runtime_error("DFlash at position "+std::to_string(position)+": "+error.what());
        }
    }
private:
    const dflash::Weights& weights_;
    dflash::RopeView rope_;
    LinearView embedding_, head_;
    dflash::DraftContext context_;
};
}
HostBackend::HostBackend(const ModelManifest& manifest, RopeTables tables, std::uint32_t workers)
    : manifest_(manifest), tables_(tables), target_(bind_host_weights(manifest),tables,workers), workers_(std::clamp(workers,1U,16U)) {
    const auto* setting = std::getenv("STRIX_G4_HOST_PARALLEL");
    if (setting && std::string(setting) == "0") workers_ = 1;
}
std::unique_ptr<StreamState> HostBackend::create_stream() { return std::make_unique<HostState>(manifest_.config); }
ForwardResult HostBackend::forward_rows(std::span<const RowPlan> rows) { return target_.forward_rows(rows); }
strix::serve::Json HostBackend::identity() const {
    auto j = strix::serve::Json::object();
    j.set("backend","host-reference"); j.set("identity",manifest_.identity);
    j.set("model_hash",manifest_.model_hash); j.set("rope_hash",tables_.hash); j.set("cache_dtype",manifest_.cache_dtype);
    j.set("workers",std::uint64_t(workers_)); j.set("row_cap",8);
    auto switches = strix::serve::Json::object();
    switches.set("STRIX_G4_HOST_PARALLEL",workers_ > 1);
    for (const auto* name : {"STRIX_G4_GRAPHS","STRIX_G4_FUSION","STRIX_G4_SHARED_WEIGHTS"}) switches.set(name,"not implemented by this backend");
    j.set("off_switches",switches);
    return j;
}
DraftFactory host_drafts(const dflash::Weights& weights, dflash::RopeView rope, LinearView embedding, LinearView head) {
    return [&weights,rope,embedding,head]() { return std::make_unique<HostDraft>(weights,rope,embedding,head); };
}
} // namespace strix::gemma4::engine
