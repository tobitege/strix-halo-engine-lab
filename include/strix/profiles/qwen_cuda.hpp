#pragma once
// CUDA Qwen profile rows over the SAME resolver as HIP (strix::qwen::profile::resolve / take_option, rows = this table).
// Host-only: no CUDA types. Same lever names where the arithmetic is the same (moe_route_bf16); the HIP-only rows
// (qsa_budget, gdn_fp16_chunk, output_head) are listed as pending in the fast column and never used on CUDA; the FAST
// prefill forms hc_mma / mtp_mma / qsa_mma are CUDA numerical identities (FP16 tensor cores, FP32 accumulation).
#include "strix/profiles/qwen.hpp"

#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::qwen::cuda_profile {
namespace pr = ::strix::qwen::profile;
using pr::Lever;
using pr::Kind;

// Columns: default, fast, quality. A form is provisional in fast until its gate receipt flips `ready`.
inline constexpr Lever table[] = {
    // Pending in fast: the engine refuses this storage with the MTP sidecar, graphs, cohorts, node hand-off and the prefix
    // cache (qwen_route::require_composition), i.e. in every shipped CUDA composition. Runnable only by an explicit
    // option/env or --allow-pending (then unqualified), never provisional.
    {.name = "moe_route_bf16", .env = "STRIX_QPF2K_MOE_ROUTE_BF16", .option = "--prefill-moe-route-bf16",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {true, true, true},
     .target = "BF16 MoE route storage (cuda-qwen-a8-prefill+moeroute-bf16-rne-v1)", .off = "0",
     .source = "same arithmetic as the HIP row; unqualified; the engine declines it with MTP sidecar/graphs/cohort/hand-off/prefix cache; changes logits"},
    {.name = "qsa_budget", .env = "STRIX_QPF2K_QSA_BUDGET", .option = "--qsa-budget",
     .value = {"", "1536", ""}, .builtin = "default",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, false, false},
     .target = "1536", .off = "0", .source = "HIP prefill ranked-block budget; not implemented on CUDA",
     .provisional = {false, true, false}},
    {.name = "gdn_fp16_chunk", .env = "STRIX_QPF2K_GDN_FP16_CHUNK", .option = "--gdn-fp16-chunk",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, false, false},
     .target = "qwen-gdn-fp16-chunk", .off = "0", .source = "HIP chunked GDN prefill on FP16 WMMA; not implemented on CUDA",
     .provisional = {false, true, false}},
    {.name = "output_head", .env = "", .option = "--output-head-profile",
     .value = {"", "fp16-output-head", ""}, .builtin = "default",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, false, false},
     .target = "fp16 output-head projection", .off = "off", .source = "pending output-head implementation and qualification"},
    {.name = "hc_mma", .env = "STRIX_QRTX_HC_MMA", .option = "--hc-mma",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, true, false},
     .target = "qwen-cuda-hc-f16mma-v1", .off = "0",
     .source = "hyper-connection projection/output on FP16 tensor cores, FP32 accumulation; qualification in progress; changes logits",
     .provisional = {false, true, false}},
    {.name = "mtp_mma", .env = "STRIX_QRTX_MTP_MMA", .option = "--mtp-mma",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, true, false},
     .target = "qwen-cuda-mtp-f16mma-v1", .off = "0",
     .source = "MTP fused projection on FP16 tensor cores, FP32 accumulation; qualification in progress; changes logits",
     .provisional = {false, true, false}},
    {.name = "qsa_mma", .env = "STRIX_QRTX_QSA_MMA", .option = "--qsa-mma",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, true, false},
     .target = "qwen-cuda-qsa-f16mma-v1", .off = "0",
     .source = "QSA prefill attention on FP16 tensor cores, FP32 accumulation; qualification in progress; changes logits",
     .provisional = {false, true, false}},
    {.name = "moe_mma", .env = "STRIX_QRTX_MOE_MMA", .option = "--moe-mma",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, true, false},
     .target = "qwen-cuda-moe-i8fma-v1", .off = "0",
     .source = "routed-MoE prefill gate/up and down on int8 tensor cores with one FP32 FMA chain per output; qualification in progress; changes logits",
     .provisional = {false, true, false}},
};
inline std::span<const Lever> levers() { return table; }

inline constexpr std::string_view kHcIdentity = "qwen-cuda-hc-f16mma-v1";
inline constexpr std::string_view kMtpIdentity = "qwen-cuda-mtp-f16mma-v1";
inline constexpr std::string_view kQsaIdentity = "qwen-cuda-qsa-f16mma-v1";
inline constexpr std::string_view kMoeIdentity = "qwen-cuda-moe-i8fma-v1";

// The runnable levers a resolution selects. Pending, off and builtin-off rows select nothing.
struct Choice {
    bool moe_route_bf16 = false, hc_mma = false, mtp_mma = false, qsa_mma = false, moe_mma = false;
    bool fast() const { return moe_route_bf16 || hc_mma || mtp_mma || qsa_mma || moe_mma; }
    bool operator==(const Choice&) const = default;
};

inline std::string normalise(std::string_view lever, std::string_view value) {
    if (value == "1" || value == "on" || value == "true") return "1";
    if (value == "0" || value == "off" || value == "false") return "0";
    throw std::invalid_argument(std::string(lever) + " is on or off (got '" + std::string(value) + "')");
}
// Explicit values of the boolean levers are validated before use; non-boolean rows (pending) are left as they are.
inline bool lever_on(const pr::Resolution& r, std::string_view name) {
    const auto* entry = r.find(name);
    if (!entry || entry->status == pr::Status::pending || entry->status == pr::Status::off) return false;
    return normalise(name, entry->value) == "1";
}
inline Choice choose(const pr::Resolution& r) {
    Choice c;
    c.moe_route_bf16 = lever_on(r, "moe_route_bf16");
    c.hc_mma = lever_on(r, "hc_mma");
    c.mtp_mma = lever_on(r, "mtp_mma");
    c.qsa_mma = lever_on(r, "qsa_mma");
    c.moe_mma = lever_on(r, "moe_mma");
    return c;
}
inline std::string forms_list(const Choice& c) {
    std::string out;
    for (const auto& [on, name] : {std::pair{c.hc_mma, "hc_mma"}, std::pair{c.mtp_mma, "mtp_mma"}, std::pair{c.qsa_mma, "qsa_mma"}, std::pair{c.moe_mma, "moe_mma"}})
        if (on) out += (out.empty() ? "" : ",") + std::string(name);
    return out;
}
// The CUDA arithmetic of the prefill, as a stable string (receipt, hand-off producer id, prefix identity).
inline std::string arithmetic_contract(const Choice& c) {
    if (!c.fast()) return "qwen-cuda;prefill=exact";
    const auto forms = forms_list(c);
    return "qwen-cuda;prefill=fast;forms=" + (forms.empty() ? std::string("none") : forms) +
        ";moe_route_bf16=" + (c.moe_route_bf16 ? "1" : "0");
}
// Identities of the selected forms, e.g. "qwen-cuda-hc-f16mma-v1,qwen-cuda-qsa-f16mma-v1".
inline std::string form_identities(const Choice& c) {
    std::string out;
    const auto add = [&](bool on, std::string_view id) { if (on) out += (out.empty() ? "" : ",") + std::string(id); };
    add(c.hc_mma, kHcIdentity);
    add(c.mtp_mma, kMtpIdentity);
    add(c.qsa_mma, kQsaIdentity);
    add(c.moe_mma, kMoeIdentity);
    return out;
}
inline ::strix::profile::ExecutionReceipt receipt(const Choice& c, std::string_view prefill_lineage = {}) {
    ::strix::profile::ExecutionReceipt out;
    out.engine = "qwen-cuda";
    out.identity.arithmetic_contract = arithmetic_contract(c);
    out.identity.prefill_lineage = prefill_lineage;
    return out;
}
inline pr::Resolution resolve(pr::Name profile, const pr::Inputs& inputs = {}) {
    auto out = pr::resolve(profile, inputs, levers());
    (void)choose(out);  // refuses malformed explicit values here, not at first use
    return out;
}
inline bool take_option(int argc, char** argv, int& i, std::optional<pr::Name>& profile, pr::Inputs& inputs) {
    return pr::take_option(argc, argv, i, profile, inputs, levers());
}
}  // namespace strix::qwen::cuda_profile
