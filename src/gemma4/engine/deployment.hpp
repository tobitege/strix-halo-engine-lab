#pragma once
#include "strix/deploy/binding.hpp"
#include <filesystem>

namespace strix::gemma4::engine {
struct DeploymentModel {
    std::uint64_t weights = 0, largest_tensor = 0, prefill_weights_upper_bound = 0;
    std::uint32_t hidden = 2560, inter = 10240, vocab = 262144;
    std::uint32_t layers = 42, kv_heads = 2, window = 512, max_context = 65536;
    std::uint32_t sliding_producers = 20, global_producers = 4;
};
DeploymentModel deployment_model(const std::filesystem::path& gguf);
class FootprintProvider final : public deploy::FootprintProvider {
public:
    explicit FootprintProvider(DeploymentModel m, bool replicas = false,
        std::uint64_t prefix_bytes = 4ULL<<30, bool prepared = false)
        : m_(m), replicas_(replicas), prefix_bytes_(prefix_bytes), prepared_(prepared) {}
    std::string model() const override { return "nemotron-bolt-e4b"; }
    std::vector<std::string> topologies() const override { return replicas_ ? std::vector<std::string>{"replica"} : std::vector<std::string>{"single"}; }
    std::vector<deploy::Format> formats(const std::string&) const override { return {{"q4_k_m","q4_k_m"}}; }
    std::uint32_t max_slots() const override { return 8; }
    deploy::Footprint footprint(const deploy::Spec&, const deploy::Candidate&, std::uint32_t) const override;
private:
    DeploymentModel m_; bool replicas_; std::uint64_t prefix_bytes_; bool prepared_;
};
} // namespace strix::gemma4::engine
