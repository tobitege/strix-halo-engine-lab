#pragma once
// ONE RESOLVER of the Qwen CUDA engine's product profile and TURBO weight toggle, called by the serve front end
// (apps/cuda_qwen_serve_main.cu) and the CLI (tests/cuda/qwen_decode_e2e.cu) alike: the same request resolves to the
// same effective values on both. It is the shared Qwen profile resolver (strix/profiles/qwen.hpp: qwen-turbo-v1,
// --turbo-weights native|onfly|converted, --turbo-weight-bits, cache namespace) with the CUDA weight caps
// (kCudaWeightCaps) and no HIP lever rows: the HIP levers of the turbo column have no CUDA binding, so a lever option
// is refused here rather than accepted and ignored. Under turbo an unloadable weight source refuses the start (never
// a silent native fallback).
#include <optional>
#include <stdexcept>
#include <string>

#include "strix/cuda/qwen_prefill.hpp"
#include "strix/profiles/qwen.hpp"

namespace strix::cuda::qwen_turbo {

namespace pq = ::strix::qwen::profile;
namespace tb = ::strix::profile::turbo;

struct Request {
    std::optional<pq::Name> profile;
    pq::Inputs inputs;
    bool print_effective_config = false;
};

// Consumes argv[i] (and its value) when it is a profile / TURBO weight option; false: not ours.
inline bool take_option(int argc, char** argv, int& i, Request& r) {
    const std::string_view arg = argv[i];
    if (arg == "--print-effective-config") { r.print_effective_config = true; return true; }
    const auto before = r.inputs.cli.size();
    if (!pq::take_option(argc, argv, i, r.profile, r.inputs)) return false;
    if (r.inputs.cli.size() != before)
        throw std::invalid_argument("Qwen CUDA: lever option " + std::string(arg) +
                                    " has no CUDA binding (profile levers of the HIP engine)");
    return true;
}

struct Resolved {
    pq::Resolution resolution;
    bool onfly_nvfp4 = false;
    std::string cache_namespace;

    // --print-effective-config text: profile, requested / effective weights with the reason, namespace, arithmetic.
    [[nodiscard]] std::string effective_config() const {
        std::string out = "qwen-cuda profile " + resolution.versioned() + "\n";
        if (resolution.is_turbo()) {
            out += resolution.weights.describe();
            out += "turbo cache_namespace " + cache_namespace + "\n";
            out += std::string("turbo onfly ") + (onfly_nvfp4 ?
                "dense Q8_0 roles (gdn qkv/z/out, qsa q/k/v/o, shared expert) -> nvfp4 at load, arithmetic nvfp4-a8-v1 in "
                "decode windows; prefill blocks, routed experts, LM head, hyper mixes, router, MTP draft: pack form" :
                "off (native pack)") + "\n";
            out += tb::describe_facts({{"precision", onfly_nvfp4 ? "weights nvfp4 e2m1/e4m3 g16 x a8 int8 g32" : "pack"},
                                       {"hardware", "sm_120"}, {"backend", "cuda"},
                                       {"chunks", "prefill unchanged (Q8_0 >= 16 rows)"}});
        }
        for (const auto& n : resolution.notices()) out += "notice " + n + "\n";
        return out;
    }
};

// Resolves and validates (throws when a turbo weight request cannot be honoured).
inline Resolved resolve(const Request& r) {
    pq::Inputs inputs = r.inputs;
    inputs.caps = &pq::kCudaWeightCaps;
    Resolved out{pq::resolve(r.profile.value_or(pq::Name::standard), inputs, {}), false, {}};
    out.resolution.weights.require_honoured();
    out.onfly_nvfp4 = out.resolution.is_turbo() && out.resolution.weights.effective == tb::Source::onfly;
    out.cache_namespace = out.resolution.cache_namespace();
    return out;
}

inline void apply(const Resolved& r, qwen_prefill::Config& config) {
    config.turbo_onfly_nvfp4 = r.onfly_nvfp4;
    config.turbo_namespace = r.cache_namespace;
}

}  // namespace strix::cuda::qwen_turbo
