// Host regression for allocation/identity agreement and fail-closed composition.
#include "../src/hip_gdn_identity.hpp"
#include "attn_core/recurrent_state.hpp"
#include "strix/memory_plan.hpp"
#include "strix/recurrent_state_policy.hpp"
#include "strix/rtx_state_import.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void set_policy(const char* value) {
#ifdef _WIN32
    require(_putenv_s(strix::kRecurrentStateEnv, value) == 0, "set environment");
#else
    require(setenv(strix::kRecurrentStateEnv, value, 1) == 0, "set environment");
#endif
}

void model_identity() {
    using namespace strix;
    // This is the runner's factory pattern: assign only the model, then copy to
    // the deferred config. No second dispatch type may need manual propagation.
    HipGdnLauncherConfig config;
    ExecutionBindingRequest request;
    for (const auto type : {RecurrentStateType::f32, RecurrentStateType::f16, RecurrentStateType::bf16}) {
        auto model = ModelContract::qwen38_flash_next();
        model.recurrent_state = type;
        config.model = model;
        const auto deferred = config;
        const auto id = hip_gdn_identity::kernel_id(request, deferred);
        const auto marker = std::string(".state-") + recurrent_state_type_name(type) + "-rne-v1";
        require(type == RecurrentStateType::f32 ? id.find(".state-") == std::string::npos :
            id.find(marker) != std::string::npos, "runner model did not select the kernel's state contract");
        const auto bytes = gdn_matrix_state_bytes_per_layer(deferred.model);
        require(bytes == (type == RecurrentStateType::f32 ? 3145728U : 1572864U), "layer allocation size");
        const std::vector<std::byte> matrix(static_cast<std::size_t>(bytes));
        require(matrix.size() == bytes, "exactly sized layer view");
    }
}

void import_admission(strix::RecurrentStateType receiver) {
    using namespace strix;
    // The v1 begin message echoes the receiver identity and has no producer
    // codec. Try both equal-size encodings, including both cross-type directions.
    for (const auto producer : {RecurrentStateType::f16, RecurrentStateType::bf16}) {
        auto layout = std::make_shared<RtxStateLayout>();
        layout->layout_identity = 123;
        layout->context_tokens = 8;
        layout->fixed_bytes = recurrent_state_bytes(1, producer);
        layout->fixed = {{"recurrent_matrix", layout->fixed_bytes, 0}};
        RtxStateBeginRequest request;
        request.layout_identity = layout->layout_identity;
        request.position = 1;
        request.tokens = {1};
        request.producer = {1, 2, 3};
        bool allocated = false, refused = false;
        try {
            RtxStateStaging staging(layout, request, "host", [&](std::uint64_t bytes) -> PrefixCacheBuffer {
                allocated = true;
                return {std::shared_ptr<std::byte>(new std::byte[static_cast<std::size_t>(bytes)],
                    std::default_delete<std::byte[]>()), bytes};
            });
        } catch (const std::invalid_argument& error) {
            refused = std::string(error.what()).find("no producer state contract") != std::string::npos;
        }
        require(receiver == RecurrentStateType::f32 ? !refused && allocated : refused && !allocated,
            "typed v1 import must refuse before allocation even when identities and sizes match");
    }
}

void partition_counterexample() {
    using namespace strix::attn_core::rstate;
    for (const bool half : {true, false}) {
        const auto round = [half](float v) { return half ? f32_from_f16_host(f16_from_f32_host(v)) :
            f32_from_bf16(bf16_from_f32(v)); };
        const float value = 1.0F + (half ? 0x1p-10F : 0x1p-7F);
        const auto row = [value](float state) { return state + 0.5F * (value - state); };
        const float cold = round(row(row(1.0F)));
        const float warm_prefix = round(row(1.0F));
        const float partitioned = round(row(warm_prefix));
        require(cold != partitioned, "counterexample must expose call-boundary rounding");
        require(row(row(1.0F)) != row(warm_prefix), "second-row core must also differ");
    }
}
} // namespace

int main(int argc, char** argv) try {
    using namespace strix;
    require(argc == 2, "provide f32, f16, bf16, empty, or invalid");
    const std::string input(argv[1]);
    set_policy(input == "empty" ? "" : input.c_str());
    if (input == "invalid") {
        try { (void)recurrent_state_policy(); }
        catch (const std::invalid_argument&) { return 0; }
        throw std::runtime_error("unknown environment value accepted");
    }
    const auto policy = recurrent_state_policy();
    require(std::string(recurrent_state_type_name(policy.type)) == (input == "empty" ? "f32" : input), "resolved value");
    require(std::string(policy.origin) == (input == "empty" ? "default" : "explicit"), "origin");
    bool recorded = false;
    for (const auto& entry : envcfg::registered())
        if (entry.name == kRecurrentStateEnv)
            recorded = entry.value == recurrent_state_type_name(policy.type) &&
                (input == "empty" || entry.set);
    require(recorded, "resolved policy must be recorded");
    set_policy("invalid");
    require(recurrent_state_policy().type == policy.type, "policy must stay cached");
    model_identity();
    import_admission(policy.type);
    partition_counterexample();
    bool refused = false;
    try { require_f32_recurrent_state(policy.type, "CUDA application"); }
    catch (const std::invalid_argument&) { refused = true; }
    require(refused == policy.fast, "unqualified application must reject typed state");
    std::cout << "recurrent state admission passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
