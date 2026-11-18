#include "strix/memory_plan.hpp"

#include <charconv>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

std::uint64_t parse_u64(std::string_view value, std::string_view name) {
    std::uint64_t parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
        throw std::invalid_argument(std::string(name) + " must be an unsigned integer");
    }
    return parsed;
}

void print_field(std::string_view name, std::uint64_t value, bool comma = true) {
    constexpr double bytes_per_gib = 1024.0 * 1024.0 * 1024.0;
    std::cout << "    \"" << name << "\": {\"bytes\": " << value
              << ", \"gib\": " << std::fixed << std::setprecision(3)
              << static_cast<double>(value) / bytes_per_gib << "}";
    std::cout << (comma ? ",\n" : "\n");
}

}  // namespace

int main(int argc, char** argv) try {
    strix::MemoryPlanRequest request;
    strix::HostBudget host{strix::gib(128), strix::gib(2), strix::gib(18)};
    request.resident_model_bytes = strix::gib(68);

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        const auto need_value = [&](std::string_view name) -> std::string_view {
            if (++index >= argc) {
                throw std::invalid_argument(std::string(name) + " needs a value");
            }
            return argv[index];
        };
        if (argument == "--tokens") {
            request.context_tokens = parse_u64(need_value(argument), argument);
        } else if (argument == "--sessions") {
            request.sessions = parse_u64(need_value(argument), argument);
        } else if (argument == "--draft-tokens") {
            request.draft_tokens = parse_u64(need_value(argument), argument);
        } else if (argument == "--resident-gib") {
            request.resident_model_bytes = strix::gib(parse_u64(need_value(argument), argument));
        } else if (argument == "--cache-gib") {
            request.bounded_cache_bytes = strix::gib(parse_u64(need_value(argument), argument));
        } else if (argument == "--host-gib") {
            host.mem_total_bytes = strix::gib(parse_u64(need_value(argument), argument));
        } else if (argument == "--baseline-gib") {
            host.baseline_without_engine_bytes = strix::gib(parse_u64(need_value(argument), argument));
        } else if (argument == "--reserve-gib") {
            host.recovery_reserve_bytes = strix::gib(parse_u64(need_value(argument), argument));
        } else if (argument == "--rollback-full") {
            request.rollback = strix::RollbackPolicy::full_gdn_snapshots;
        } else if (argument == "--mtp-draft-state-bytes") {
            request.mtp_draft_state_element_bytes =
                parse_u64(need_value(argument), argument);
        } else if (argument == "--no-mtp-draft-state") {
            request.include_mtp_draft_state = false;
        } else if (argument == "--no-mtp-indexer") {
            request.include_mtp_indexer = false;
        } else if (argument == "--no-indexer") {
            request.include_indexer = false;
        } else if (argument == "--no-mtp-kv") {
            request.include_mtp_kv = false;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }

    const auto model = strix::ModelContract::qwen38_flash_next();
    const auto plan = strix::build_memory_plan(model, request);
    const auto decision = strix::decide_admission(host, plan);

    std::cout << "{\n  \"recurrent_state\": \"" << strix::recurrent_state_contract(model.recurrent_state)
              << "\",\n  \"breakdown\": {\n";
    print_field("base_kv", plan.base_kv);
    print_field("mtp_kv", plan.mtp_kv);
    print_field("mtp_draft_state", plan.mtp_draft_state);
    print_field("mtp_indexer_raw_keys", plan.mtp_indexer);
    print_field("indexer_raw_keys", plan.indexer);
    print_field("gdn_matrix_state", plan.gdn_matrix_state);
    print_field("gdn_conv_state", plan.gdn_conv_state);
    print_field("gdn_state", plan.gdn_state);
    print_field("rollback", plan.rollback);
    print_field("resident_model", plan.resident_model);
    print_field("bounded_cache", plan.bounded_cache);
    print_field("total", plan.total, false);
    std::cout << "  },\n  \"admission\": {\n"
              << "    \"admitted\": " << (decision.admitted ? "true" : "false") << ",\n"
              << "    \"engine_limit_bytes\": " << decision.engine_limit << ",\n"
              << "    \"headroom_bytes\": " << decision.headroom << ",\n"
              << "    \"reason\": \"" << decision.reason << "\"\n"
              << "  }\n}\n";
    return decision.admitted ? 0 : 2;
} catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
}
