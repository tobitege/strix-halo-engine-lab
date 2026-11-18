#pragma once

#include "strix/device/eager_code_load.hpp"
#include "strix/prefill_identity.hpp"
#include "strix/qpf2k_defaults.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <set>
#include <string_view>
#include <vector>

namespace strix::decode_cli {

// Eager code loading lives in the shared start gate home (strix/device/eager_code_load.hpp); this engine resolves its
// own policy (CodeLoad::engine) and calls the same functions.
using ::strix::device::apply_eager_code_loading;
using ::strix::device::split_process_arguments;

// Shared by CLI receipts, host inspection and HTTP capability discovery.
inline std::string prefill_identity_fields(unsigned mask, std::uint32_t qsa_budget = qsa_prefill_budget_active()) {
    std::ostringstream out;
    out << "\"prefill_identity\":\"" << prefill_identity_name(mask, qsa_budget)
        << "\",\"prefill_identity_hash\":\"" << std::hex << std::setw(16) << std::setfill('0')
        << prefill_identity_hash(mask, qsa_budget) << "\",\"prefill_options_active\":[";
    bool first = true;
    for (unsigned i = 0; i < kPrefillOptions.size(); ++i) {
        if (!(mask & (1U << i))) continue;
        out << (first ? "" : ",") << '"' << kPrefillOptions[i].name << '"';
        first = false;
    }
    if (qsa_budget) out << (first ? "" : ",") << "\"qsa_budget\"";
    out << ']';
    if (qsa_budget) out << ",\"qsa_prefill_budget\":" << std::dec << qsa_budget
        << ",\"qsa_prefill_budget_local_tail\":\"unchanged\",\"qsa_prefill_budget_qualification\":\"in-progress\"";
    return out.str();
}

// HTTP excludes plain-CLI early setup, automatic T16K and compact images.
// Report effective HTTP policy separately from bit-changing prefill identity.
template<class Get>
inline std::string http_exact_prefill_fields(Get get, bool baseline, bool gsq) {
    const auto* low = get("STRIX_QPF2K_HC_LOWBAND");
    const auto* hip = get("HIP_ENABLE_DEFERRED_LOADING");
    std::ostringstream out;
    out << "\"prefill_exact_items\":{\"early_setup\":false,\"automatic_t16k\":false,"
        << "\"compact_stage_images\":false,\"index_batch\":"
        << (qpf2k::index_batch_enabled(get, baseline, gsq) ? "true" : "false")
        << ",\"hc_lowband_policy\":" << (baseline || gsq || !low ? 0U : std::strtoul(low, nullptr, 0))
        << ",\"hc_lowband_guard\":\"exact-fp16-output-only\",\"ple_staged_acceptance_fix\":true,"
        << "\"hip_deferred_loading_disabled\":" << (hip && std::string_view(hip) == "0" ? "true" : "false") << '}';
    return out.str();
}
inline std::string http_exact_prefill_fields() {
    return http_exact_prefill_fields([](const char* name) { return std::getenv(name); }, qpf2k::baseline, qpf2k::gsq_model);
}

inline std::string_view prefill_trim(std::string_view value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string_view::npos) return {};
    return value.substr(begin, value.find_last_not_of(" \t\r\n") - begin + 1U);
}

// There is no general Qwen server config loader. This deliberately small startup
// format accepts arithmetic and exact startup keys, one assignment per line, with # comments.
// A false value leaves the environment in force; CLI switches can disable more.
inline void read_prefill_config(std::istream& input, bool& exact, unsigned& disabled,
    bool* early_setup = nullptr, bool* eager_code_load = nullptr) {
    std::set<std::string> seen;
    std::string line;
    while (std::getline(input, line)) {
        auto text = prefill_trim(std::string_view(line).substr(0, line.find('#')));
        if (text.empty()) continue;
        const auto equal = text.find('=');
        if (equal == std::string_view::npos) throw std::invalid_argument("prefill config requires key = value");
        const auto key = prefill_trim(text.substr(0, equal));
        const auto value = prefill_trim(text.substr(equal + 1U));
        if (!seen.emplace(key).second) throw std::invalid_argument("duplicate prefill config key");
        if (key == "qpf2k_early_setup" || key == "eager_code_load") {
            if (value != "true" && value != "false") throw std::invalid_argument("startup setting requires true or false");
            auto* target = key == "qpf2k_early_setup" ? early_setup : eager_code_load;
            if (!target) throw std::invalid_argument("startup setting requires engine options");
            *target = *target && value == "true";
        } else if (key == "prefill_exact_fp16") {
            if (value != "true" && value != "false") throw std::invalid_argument("prefill_exact_fp16 requires true or false");
            exact = exact || value == "true";
        } else if (key == "prefill_disable") {
            if (value.size() < 2U || value.front() != '[' || value.back() != ']')
                throw std::invalid_argument("prefill_disable requires an array of quoted option names");
            auto rest = prefill_trim(value.substr(1, value.size() - 2U));
            while (!rest.empty()) {
                if (rest.front() != '"') throw std::invalid_argument("prefill_disable requires quoted option names");
                const auto end = rest.find('"', 1U);
                if (end == std::string_view::npos) throw std::invalid_argument("unterminated prefill option name");
                const auto name = rest.substr(1U, end - 1U);
                if (name.empty() || name.find(',') != std::string_view::npos)
                    throw std::invalid_argument("prefill_disable requires individual option names");
                disabled |= parse_prefill_option_list(name);
                rest = prefill_trim(rest.substr(end + 1U));
                if (rest.empty()) break;
                if (rest.front() != ',') throw std::invalid_argument("prefill_disable requires comma-separated names");
                rest = prefill_trim(rest.substr(1U));
                if (rest.empty()) throw std::invalid_argument("prefill_disable has a trailing comma");
            }
        } else throw std::invalid_argument("unknown prefill config key '" + std::string(key) + "'");
    }
    if (input.bad()) throw std::runtime_error("cannot read prefill config");
}

inline void load_prefill_config(const std::string& path, bool& exact, unsigned& disabled,
    bool* early_setup = nullptr, bool* eager_code_load = nullptr) {
    std::ifstream input(path);
    if (!input) throw std::invalid_argument("cannot open prefill config");
    read_prefill_config(input, exact, disabled, early_setup, eager_code_load);
}

}  // namespace strix::decode_cli
