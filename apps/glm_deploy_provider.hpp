#pragma once
#include "strix/deploy/binding.hpp"
#include "glm/engine/glm_kvshard_layout.hpp"
#include <functional>

namespace strix::glm::deployment {
// Native adapter obtains these bounds from safetensors headers and the existing
// bounded prefill calculator. Tests inject them without HIP or model payloads.
struct Owners {
    std::uint64_t weights = 0, staging = 0;
};
struct Options {
    unsigned slots = 1;
    bool mtp = true, spec = false;
    std::string kv_shard = "0", profile = "default", refusal, configuration_id;
    // deploy::int4_moe_formats hook: true only once the FAST_I4 stored/JIT loader
    // and its controls are linked; footprint() then needs its header-only sizing.
    bool int4_runtime = false;
};
class Provider final : public deploy::FootprintProvider {
public:
    using Workspace = std::function<std::uint64_t(std::uint32_t, bool)>;
    Provider(std::array<Owners, 2> owners, Options options, Workspace workspace)
        : owners_(owners), options_(std::move(options)), workspace_(std::move(workspace)) {}
    std::string model() const override { return "glm-5.3"; }
    std::vector<std::string> topologies() const override { return {"tp2"}; }
    std::vector<deploy::Format> formats(const std::string&) const override {
        // Only executable formats: INT4 candidates come from the shared hook.
        if (!options_.int4_runtime) return {{"exl3", "exl3"}};
        auto out = deploy::int4_moe_formats("exl3", 42, 2);
        out.push_back({"exl3", "exl3"});
        return out;
    }
    std::uint32_t max_slots() const override { return options_.slots; }
    deploy::Footprint footprint(const deploy::Spec& s, const deploy::Candidate& c, std::uint32_t rank) const override {
        if (!options_.refusal.empty()) throw std::invalid_argument(options_.refusal);
        if (s.profile != "auto" && s.profile != options_.profile) throw std::invalid_argument("profile pin differs from active profile");
        if (c.format.prefill != "exl3")
            throw std::invalid_argument("GLM INT4 header-only sizing not bound yet (FAST_I4 hook)");
        if (rank > 1 || !c.context || c.context % 4) throw std::invalid_argument("GLM context must be a positive multiple of 4");
        if (options_.kv_shard != "0" && options_.kv_shard != "1" && options_.kv_shard != "auto")
            throw std::invalid_argument("STRIX_GLM_KV_SHARD must be 0, 1 or auto");
        const bool sharded = options_.kv_shard == "1";
        if (options_.kv_shard != "0" && (options_.mtp || options_.spec || c.slots != 1))
            throw std::invalid_argument("KV_SHARD requires one slot and MTP/spec OFF");
        // Auto may retain replicated ownership at runtime. Reserve that upper
        // bound; only explicit 1 can claim the smaller token-sharded cache.
        const auto cap = sharded ? kvshard::storage_tokens(c.context) : c.context;
        deploy::Footprint f;
        const auto add = deploy::checked_add, mul = deploy::checked_mul;
        const std::uint64_t dsa_layers = 15 + (options_.mtp ? 1U : 0U);
        f.kv_per_token = mul(dsa_layers, 1664); // latent, key, gate, pooled FP32/4
        f.kv_total = mul(mul(cap, c.slots), f.kv_per_token);
        // 30 recurrent layers: FP32 32x128x128 state + three conv histories;
        // optional two sets of eight deferred verify rows.
        auto recurrent = mul(30, 32ULL*128*128*4 + 3ULL*4096*3*4);
        if (options_.spec) recurrent = add(recurrent, mul(30, 16ULL*(3*4096 + 32*4*128 + 32)*4));
        f.kv_total = add(f.kv_total, mul(c.slots, recurrent));
        f.weights = owners_[rank].weights; f.staging = owners_[rank].staging;
        f.workspace = workspace_(c.context, sharded);
        if (!options_.configuration_id.empty()) f.extra.emplace_back("configuration:" + options_.configuration_id, 0);
        return f;
    }
private:
    std::array<Owners, 2> owners_;
    Options options_;
    Workspace workspace_;
};
} // namespace strix::glm::deployment
