#pragma once
#include "strix/deploy/binding.hpp"
#include <filesystem>

namespace strix::minimax::engine {
// Values are read from safetensors headers only; no payload reads or HIP calls.
struct DeploymentModel {
    std::uint64_t weights = 0;
    std::uint32_t layers = 60, hidden = 6144, kv_width = 256, index_width = 128;
    std::uint32_t moe_layers = 57, experts = 128, expert_inter = 3072;
};
DeploymentModel deployment_model(const std::filesystem::path& directory);
class FootprintProvider final : public deploy::FootprintProvider {
public:
    // int4_runtime: deploy::int4_moe_formats hook. Pass true only from a build whose
    // engine links the FAST_I4 runtime and maps the chosen resident layers / JIT slots
    // onto its controls; the INT4 byte accounting below is kept ready for that.
    FootprintProvider(DeploymentModel model, std::uint32_t rows = 4096,
                      std::string id = "minimax-m3.1-flash", std::uint64_t scratch_extra = 0,
                      bool int4_runtime = false)
        : m_(model), rows_(rows), id_(std::move(id)), scratch_extra_(scratch_extra), int4_runtime_(int4_runtime) {}
    std::string model() const override { return id_; }
    std::vector<std::string> topologies() const override { return {"tp2"}; }
    std::vector<deploy::Format> formats(const std::string&) const override {
        if (!int4_runtime_) return {{"exl3", "exl3"}};
        auto out = deploy::int4_moe_formats("exl3", m_.moe_layers, 2);
        out.push_back({"exl3", "exl3"});
        return out;
    }
    std::uint32_t max_slots() const override { return 8; }
    deploy::Footprint footprint(const deploy::Spec&, const deploy::Candidate&, std::uint32_t) const override;
private:
    DeploymentModel m_; std::uint32_t rows_; std::string id_; std::uint64_t scratch_extra_;
    bool int4_runtime_;
};
} // namespace strix::minimax::engine
