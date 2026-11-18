#pragma once
// GLM levers of the shared route receipt (strix/route_receipt.hpp). The rank CLI and the server record the same keys
// through this one function, so their requested and effective values can be compared entry point against entry point.

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "strix/env_flags.hpp"
#include "strix/route_receipt.hpp"
#include "strix/serve/prefill_interleave.hpp"
#include "glm/prefill/glm_pf_dense_policy.hpp"
#include "glm/prefill/glm_pf_attn_wmma.hpp"
#include "glm/prefill/glm_pf_exact.hpp"
#include "glm/prefill/glm_pf_wmma.hpp"

namespace strix::glm {

// Records the levers that decide the prefill arithmetic and returns the numerical-route fingerprint.
// entry: "cli" or "http" (informative, outside the fingerprint, like the chunk rows).
inline std::uint64_t record_route_levers(std::string_view entry, unsigned rank, std::string_view prefill_requested,
                                         std::string_view prefill_effective, std::string_view identity, unsigned max_rows) {
    namespace rr = strix::route;
    rr::set_engine("glm");
    rr::set_rank(static_cast<int>(rank));
    const auto asked = [](const char* name) {
        const char* v = strix::envcfg::text(name);
        return std::string(v != nullptr && *v != 0 ? v : "default");
    };
    const std::string wmma = asked("STRIX_GLM_PF_WMMA"), x = asked("STRIX_GLM_PF_EXACT_X"), kda = asked("STRIX_GLM_PF_KDA_WMMA");
    const bool wmma_on = pf::wmma_x::enabled();
    // Resolve at both ends of the supported threshold range. Only auto has two routes;
    // forced routes must never read the unused auto threshold.
    const auto low = pf::exact::route(0), high = pf::exact::route(65536);
    const std::string x_effective = low == high ? std::to_string(static_cast<unsigned>(low)) :
        "auto:min_rows=" + std::to_string(pf::exact::x_auto_rows());
    rr::lever("glm.entry", entry, entry, rr::Reason::selected, false);
    rr::lever("glm.prefill", prefill_requested, prefill_effective);
    rr::lever("glm.identity", identity, identity);
    rr::lever("glm.pf_wmma", wmma, wmma_on ? "f16_wmma" : "exact", wmma_on ? rr::Reason::selected : rr::Reason::disabled);
    rr::lever("glm.pf_kda_wmma", kda, pf::kda_wmma::enabled() ? "1" : "0");
    rr::lever("glm.pf_mla_wmma", asked("STRIX_GLM_PF_MLA_WMMA"), pf::mla_wmma::enabled() ? "1" : "0");
    rr::lever("glm.pf_mla_gemm_heads", asked("STRIX_GLM_PF_MLA_GEMM_HEADS"), pf::mla_gemm::heads_mid() ? "1" : "0");
    // The fused MLA/DSA core (glm-pf-attn-wmma, its own bits; the key block orders the online softmax) and the FP16 o-bar
    // image it can store for the WMMA out GEMM (STRIX_GLM_PF_MLA_OB16, measured not bitwise) change the prefill
    // arithmetic: the server's hello pairs on this fingerprint alone (the rank CLI also votes kPfAttnWmmaConfig).
    const bool attn_on = pf::attn_wmma::enabled();
    rr::lever("glm.pf_attn_wmma", asked("STRIX_GLM_PF_ATTN_WMMA") + ":t=" + asked("STRIX_GLM_PF_ATTN_WMMA_T"),
              attn_on ? "1:t=" + std::to_string(pf::attn_wmma::block_keys()) : std::string("0"));
    rr::lever("glm.pf_mla_ob16", asked("STRIX_GLM_PF_MLA_OB16"),
              attn_on && pf::mla_wmma::enabled() && pf::mla_wmma::ob16() ? "1" : "0");
    rr::lever("glm.pf_idxproj_wmma", asked("STRIX_GLM_PF_IDXPROJ_WMMA"), pf::idxproj_wmma::enabled() ? "1" : "0");
    rr::lever("glm.pf_x16", asked("STRIX_GLM_PF_X16"), pf::kda_wmma::x16_enabled() ? "1" : "0", rr::Reason::selected, false);
    rr::lever("glm.pf_router_wmma", asked("STRIX_GLM_PF_ROUTER_WMMA"),
              pf::router_wmma::enabled() ? "1" : "0", rr::Reason::selected, true);
    rr::lever("glm.pf_exact_x", x + ":min_rows=" + asked("STRIX_GLM_PF_EXACT_X_MIN_ROWS"), x_effective, rr::Reason::selected, true,
              "auto takes the X3 route (and the WMMA identity when on) only for calls of at least min_rows rows");
    rr::lever("glm.max_rows", std::to_string(max_rows), std::to_string(max_rows), rr::Reason::selected, false);
    return rr::fingerprint();
}

// The serving prefill chunk and the dense route its calls take (pf::dense_policy::resolve_chunk). The server and the
// rank CLI both call this with the process environment: the server admits prompts with the result, the CLI only
// reports it (it has no concurrent decode, so it always prefills whole prompts). Informative lever, outside the
// fingerprint: the chunk changes which rows share a call, never the arithmetic a selected route computes.
inline pf::dense_policy::Chunk record_prefill_chunk(std::string_view entry, bool explicit_request, unsigned requested,
                                                    unsigned world, unsigned max_rows, std::string_view prefill) {
    namespace rr = strix::route;
    namespace dp = pf::dense_policy;
    const dp::Chunk c = dp::resolve_chunk(explicit_request, requested, world, max_rows, prefill, strix::envcfg::text);
    const std::string core = !c.applicable ? "inapplicable" : c.tokens == 0 ? "prompt-dependent" : c.mixed ? "mixed" :
        c.route == dp::kX3r8 ? (c.wmma ? "x3r8+f16_wmma" : "x3r8") : c.route == dp::kTm ? "tm" :
        "x" + std::to_string(c.route);
    rr::lever("glm.prefill_chunk", explicit_request ? std::to_string(requested) : std::string("default"),
              std::to_string(c.tokens) + ":rows=" + std::to_string(c.rows) + ":dense=" + core,
              !c.applicable ? rr::Reason::wrong_route : c.tokens == 0 ? rr::Reason::no_match :
              !c.declined ? rr::Reason::selected : c.route == dp::kTm && c.reason.find("short_call") != std::string::npos ?
                  rr::Reason::short_call : rr::Reason::wrong_route,
              false, std::string(entry == "cli" ? "serving-only lever (the CLI prefills whole prompts); " : "") + c.reason);
    return c;
}


// Dry run (--print-effective-config of either entry point): an environment that is resolved but not applied.
// `apply_pending(set)` calls set(name, value) for every value the app would set if unset; lookups see those values
// first, then the process environment (an explicit value always wins, as in the applied run).
class PendingEnv {
public:
    template<class ApplyPending>
    explicit PendingEnv(ApplyPending&& apply_pending) {
        apply_pending([this](const char* name, const char* value) { if (std::getenv(name) == nullptr) pending_[name] = value; });
    }
    const char* operator()(const char* name) const {
        const auto it = pending_.find(name);
        return it != pending_.end() ? it->second.c_str() : static_cast<const char*>(std::getenv(name));
    }

private:
    std::map<std::string, std::string> pending_;
};

template<class ApplyPending>
inline pf::dense_policy::Chunk pending_prefill_chunk(bool explicit_request, unsigned requested, unsigned world, unsigned max_rows,
                                                     std::string_view prefill, ApplyPending&& apply_pending) {
    const PendingEnv env(std::forward<ApplyPending>(apply_pending));
    return pf::dense_policy::resolve_chunk(explicit_request, requested, world, max_rows, prefill, env);
}

// The server's prefill interleave (strix/serve/prefill_interleave.hpp) with the GLM route identity: on the chunked exact
// route, grouping base chunks into one call is admitted only when every dense call keeps the base calls' core
// (pf::dense_policy::choose / call_rows). One definition for the live server and its --print-effective-config dry run,
// so the printed policy is the one that admits prompts. `env` is read once here, never inside the callback.
template<class Env>
inline strix::serve::PrefillInterleave serve_prefill_interleave(std::string_view mode, unsigned max_rows, unsigned pipeline,
                                                                Env&& env) {
    auto policy = strix::serve::PrefillInterleave::from_env(env);
    if (mode == "gemv") {
        namespace dp = pf::dense_policy;
        const auto choice = dp::choose(0, env);
        policy.route_identity = [choice, max_rows, pipeline](std::size_t tokens) -> std::optional<unsigned> {
            const auto rows = dp::call_rows(static_cast<unsigned>(tokens), max_rows, pipeline);
            if (!choice.automatic) return choice.route;
            if (rows.min < choice.min_rows && rows.max >= choice.min_rows) return std::nullopt;
            return rows.min >= choice.min_rows ? dp::kX3r8 : dp::kTm;
        };
    }
    return policy;
}
inline strix::serve::PrefillInterleave serve_prefill_interleave(std::string_view mode, unsigned max_rows, unsigned pipeline) {
    return serve_prefill_interleave(mode, max_rows, pipeline, strix::envcfg::text);
}

}  // namespace strix::glm
