#pragma once
// Arm labels of the prefill quality tools (deepseek41_prefill_quality_main.cpp, deepseek41_prefill_continue_main.cpp).
#include "strix/deepseek41_prefill.hpp"
#include "strix/model_content_hash.hpp"
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::deepseek41::quality {

// Arm labels: exact, dense (FP16 all projections), attention (BF16 attention), fattn / fmoe / fast (the FAST levers
// STRIX_DS41_FAST_ATTN / STRIX_DS41_FAST_MOE / both), or a diagnostic spec of '_' tokens:
// kind d (FP16 dense), a (BF16 attention), da (both), n (null: one-ulp flip of the stream after layer l<L>), then
// qa kv qb oa ob (projections; none = all), l<first>t<end> (layers), a8 (FP16 operands from the A8 codes), r<rows>
// (FP16 only at calls of >= rows rows), f (FP16 output form: STRIX_DS41_PREFILL_FP16_FAST), s<seed> (null: +-1 ulp by a
// hash of the seed instead of the last-bit flip, so seeds sample the noise floor).
// Prefix eng_ (any arm, e.g. eng_fast3, eng_exact; labels are [a-z0-9_]): adds the FAST lever ds41-engram-f16wmma-v1 (Engram projection on WMMA).
inline PrefillOptions arm_precision(const std::string& label) {
    if (label.rfind("m16_", 0) == 0 && label.size() > 4) {  // + ds41-moe-f16wmma-v1 (relaxed MoE FP16 operands)
        PrefillOptions e = arm_precision(label.substr(4));
        e.fast_moe_f16 = true;
        return e;
    }
    if (label.rfind("sh_", 0) == 0 && label.size() > 3) {  // + ds41-shared-f16-v1 (shared expert on the FAST dense core)
        PrefillOptions e = arm_precision(label.substr(3));
        e.fast_shared = true;
        return e;
    }
    if (label.rfind("eng_", 0) == 0 && label.size() > 4) {
        PrefillOptions e = arm_precision(label.substr(4));
        e.fast_engram = true;
        return e;
    }
    PrefillOptions o;
    o.decoder_replay_override = true;  // every named arm is explicit, including exact/null OFF
    if (label == "exact") return o;
    if (label == "dense") { o.fp16_dense = true; return o; }
    if (label == "attention") { o.bf16_attention = true; return o; }
    // the FAST levers: f16 WMMA attention, relaxed-order MoE, both (the composed DS FAST prefill)
    if (label == "fattn") { o.fast_attention = true; return o; }
    if (label == "fattn2") { o.fast_attention = true; o.fast_attention_flash = true; return o; }  // ds41-attn-bf16flash-v1
    if (label == "fmoe") { o.fast_moe = true; return o; }
    if (label == "fast") { o.fast_attention = true; o.fast_moe = true; return o; }
    if (label == "fdense") { o.fast_dense = true; return o; }  // ds41-dense-f16-v1
    if (label == "fast3") { o.fast_attention = true; o.fast_attention_flash = true; o.fast_moe = true; o.fast_dense = true; return o; }
    if (label == "fast2") { o.fast_attention = true; o.fast_attention_flash = true; o.fast_moe = true; return o; }
    // fast3b<R>: fast3 + FAST policy ds41-decoder-replay-bounded-v1 with the bound R (rows, a multiple of 128)
    if (label.rfind("fast3b", 0) == 0 && label.size() > 6 && label.find_first_not_of("0123456789", 6) == std::string::npos) {
        o.fast_attention = true; o.fast_attention_flash = true; o.fast_moe = true; o.fast_dense = true;
        o.decoder_replay_rows = static_cast<std::uint32_t>(std::stoul(label.substr(6)));
        if (o.decoder_replay_rows == 0U || o.decoder_replay_rows % 128U != 0U) throw std::runtime_error("arm bound " + label);
        return o;
    }
    std::vector<std::string> tok;
    std::stringstream ss(label);
    for (std::string t; std::getline(ss, t, '_');) tok.push_back(t);
    if (tok.empty()) throw std::runtime_error("arm label");
    const std::string kind = tok[0];
    if (kind == "d" || kind == "da") o.fp16_dense = true;
    if (kind == "a" || kind == "da") o.bf16_attention = true;
    if (kind != "d" && kind != "a" && kind != "da" && kind != "n") throw std::runtime_error("arm kind " + label);
    std::uint32_t proj = 0;
    for (std::size_t i = 1; i < tok.size(); ++i) {
        const auto& t = tok[i];
        if (t == "qa") proj |= 1U;
        else if (t == "kv") proj |= 2U;
        else if (t == "qb") proj |= 4U;
        else if (t == "oa") proj |= 8U;
        else if (t == "ob") proj |= 16U;
        else if (t == "a8") o.fp16_keep_a8 = true;
        else if (t == "f") o.fp16_fast = true;
        else if (t == "w") o.fp16_wsplit = true;  // with f: weights as hi + lo FP16 halves
        else if (t[0] == 's' && kind == "n") o.null_seed = static_cast<std::uint32_t>(std::stoul(t.substr(1)));
        else if (t[0] == 'r') o.fp16_min_rows = static_cast<std::uint32_t>(std::stoul(t.substr(1)));
        else if (t[0] == 'l') {
            const auto at = t.find('t');
            if (kind == "n") o.null_ulp_layer = std::stoi(t.substr(1));
            else if (at == std::string::npos) throw std::runtime_error("layer token " + label);
            else {
                o.approx_layer_first = static_cast<std::uint32_t>(std::stoul(t.substr(1, at - 1)));
                o.approx_layer_end = static_cast<std::uint32_t>(std::stoul(t.substr(at + 1)));
            }
        } else throw std::runtime_error("arm token " + label);
    }
    if (proj) o.fp16_projections = proj;
    if (kind == "n" && o.null_ulp_layer < 0) throw std::runtime_error("null arm needs l<L>");
    return o;
}

// Profile of an arm for the gate's execution witness: "exact" (incumbent arithmetic; the null controls perturb it),
// "fast-prefill" (the FAST prefill levers fast3 / fast3b<R>), else "diagnostic" (no floor-rule role).
inline PrefillOptions executed_precision(PrefillOptions o, const PrefillResult& run) {
    o.decoder_replay_rows = run.decoder_replay_rows;
    o.decoder_replay_override = true;
    return o;
}

inline std::string arm_profile(const PrefillOptions& o) {
    if (!o.decoder_replay_override && o.decoder_replay_rows == 0U) return "diagnostic";  // unresolved inheritance
    const bool fast3 = o.fast_attention && o.fast_attention_flash && o.fast_moe && o.fast_dense;
    const bool other = o.fp16_dense || o.bf16_attention || o.fp16_fast || o.fp16_keep_a8 || o.fp16_wsplit ||
                       o.fp16_projections != 31U || o.approx_layer_first != 0U || o.approx_layer_end != 1000U || o.fp16_min_rows != 0U;
    if (other) return "diagnostic";
    if (fast3 && o.null_ulp_layer < 0) return "fast-prefill";
    if (!o.fast_attention && !o.fast_attention_flash && !o.fast_moe && !o.fast_dense && !o.fast_engram && o.decoder_replay_rows == 0U) return "exact";
    return "diagnostic";
}

// Full content reads, independent of paths and pinned metadata. The caller verifies
// the loader snapshot before/after hashing so these are the files actually loaded.
inline std::string model_witness(const std::filesystem::path& target, const std::filesystem::path& overlay) {
    const auto hash = [](const std::filesystem::path& path) {
        return model_content_identity_hex(sha256_file_content(path, FileHashPolicy::full));
    };
    const std::string content = "ds41-quality-model-v2\ntarget_sha256=" + hash(target) +
        "\noverlay_sha256=" + (overlay.empty() ? "none" : hash(overlay)) + "\n";
    return model_content_identity_hex(sha256_bytes(std::as_bytes(std::span(content))));
}

}  // namespace strix::deepseek41::quality
