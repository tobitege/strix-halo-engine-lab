#include "core/iu4_plan.hpp"
#include "../apps/hip_decode_options.hpp"
#include "strix/qwen_onfly_i4.hpp"
#include "strix/q6p_registry.hpp"

#include <atomic>
#include <future>
#include <iostream>
#include <type_traits>

namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> void refused(F&& f, const char* message) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, message);
}
void admission() {
    using namespace strix;
    for (unsigned rows = 1; rows <= 8; ++rows) {
        onfly_i4::require_serial_projection(rows, false);
        refused([&] { onfly_i4::require_serial_projection(rows, true); }, "A8 source parity gate");
        for (unsigned groups : {32U, 40U, 64U, 96U}) {
            const auto plan = core::iu4::row_plan(rows, groups);
            const auto serial = core::iu4::row_plan(1U, groups);
            check(plan.S == serial.S && plan.GC == serial.GC, "row partition differs");
            check(plan.S == (groups < 64U ? 4U : 8U) &&
                plan.GC == (groups == 40U ? 40U : 32U), "fixed partition contract");
            check(!plan.from_a8_slot, "IU4 must stage original rows even with an A8 slot");
            check(groups % plan.GC == 0 && plan.GC % plan.S == 0 &&
                rows * plan.GC * 128U <= 40U * 1024U, "partition/LDS bounds");
            for (unsigned pass = 0; pass < groups / plan.GC; ++pass)
                for (unsigned wave = 0; wave < plan.S; ++wave)
                    for (unsigned g = 0; g < plan.GC / plan.S; ++g)
                        check(pass * plan.GC + wave * (plan.GC / plan.S) + g ==
                            pass * serial.GC + wave * (serial.GC / serial.S) + g,
                            "per-pass group order differs");
        }
    }
    refused([] { onfly_i4::require_serial_projection(0U, false); }, "empty projection admitted");
    refused([] { onfly_i4::require_serial_projection(9U, false); }, "oversized projection admitted");
    decode_cli::Options o;
    o.turbo.source = onfly_i4::Source::onfly;
    decode_cli::validate_onfly_i4_routes(o);
    for (int route = 0; route < 6; ++route) {
        auto candidate = o;
        if (route == 0) candidate.mtp.enabled = true;
        if (route == 1) candidate.known_prefix_window = true;
        if (route == 2) candidate.prompt_lookup = true;
        if (route == 3) candidate.mtp.cohort.emplace_back();
        if (route == 4) candidate.http_serve = true; // default policy selection
        if (route == 5) {
            candidate.http_serve = true;
            candidate.http_selection_mtp = false;
            candidate.http_batch = 2;
        }
        if (route == 3 || route == 5)
            refused([&] { decode_cli::validate_onfly_i4_routes(candidate); }, "unqualified batch route admitted");
        else decode_cli::validate_onfly_i4_routes(candidate);
        candidate.turbo.source = onfly_i4::Source::native;
        decode_cli::validate_onfly_i4_routes(candidate);
    }
    o.http_serve = true;
    o.http_selection_mtp = false;
    decode_cli::validate_onfly_i4_routes(o);
}
void registry_lifetime() {
    using namespace strix::q6p;
    int original = 0;
    std::atomic<bool> freed{false};
    auto backing = std::shared_ptr<const void>(new unsigned char[8]{}, [&](const void* p) {
        delete[] static_cast<const unsigned char*>(p);
        freed.store(true);
    });
    const auto* base = static_cast<const unsigned char*>(backing.get());
    iu4::register_tensor(&original, {base, 70, 4096, backing});
    auto snapshot = iu4::find(&original);
    static_assert(std::is_const_v<std::remove_reference_t<decltype(*snapshot)>>);
    backing.reset();
    // Replacement runs on another thread; the old metadata and allocation stay live.
    auto replacement = std::make_shared<unsigned char>(static_cast<unsigned char>(0));
    auto replace = std::async(std::launch::async, [&] {
        iu4::register_tensor(&original, {replacement.get(), 128, 2048, replacement});
    });
    replace.get();
    check(snapshot->rows == 70 && snapshot->cols == 4096 && !freed.load(), "replacement invalidated snapshot");
    iu4::unregister_tensor(&original, base);
    check(iu4::find(&original)->rows == 128, "stale owner retired replacement");
    iu4::unregister_tensor(&original, replacement.get());
    check(!iu4::find(&original) && iu4::count() == 0 && !freed.load(), "retirement lost backing lease");
    snapshot.reset();
    check(freed.load(), "snapshot release leaked backing");
    refused([&] { iu4::register_tensor(&original, {nullptr, 1, 2048, {}}); }, "unowned backing accepted");
}
void identities_and_census() {
    using namespace strix;
    const auto no_env = [](const char*) -> const char* { return nullptr; };
    std::string previous;
    for (const char* rule : {"legacy", "full", "mse"}) {
        const auto resolved = onfly_i4::resolve({"gdn_in,gdn_out", rule}, no_env, true);
        const auto receipt = serve::parse_json(resolved.json());
        check(receipt.find("identity")->as_string() == resolved.identity(), "receipt identity");
        check(receipt.find("group_mask")->as_int() == 3, "receipt group mask");
        check(receipt.find("scale_rule")->as_string() == rule, "receipt scale rule");
        qwen::profile::Resolution profile;
        profile.profile = qwen::profile::Name::turbo;
        profile.onfly = resolved;
        const auto capabilities = serve::parse_json(profile.json());
        const auto& turbo = *capabilities.find("turbo");
        check(turbo.find("weights")->as_string() == resolved.identity(), "capability weight identity");
        check(turbo.find("onfly")->dump() == receipt.dump(), "capability/receipt agreement");
        check(previous != resolved.identity(), "scale rules collapsed");
        previous = resolved.identity();
    }
    const auto resolved = onfly_i4::resolve({}, no_env, true);
    for (const std::uint64_t k : {192U, 2048U, 2560U, 4096U, 6144U, 8192U}) {
        ManifestTensor tensor;
        tensor.name = "blk.0.ssm_out.weight";
        tensor.ggml_type = 8;
        tensor.dimensions = {k, 70U};
        tensor.byte_size = 70U * (k / 32U) * 34U;
        const bool supported = k == 2048U || k == 2560U || k == 4096U || k == 6144U;
        check(qwen_onfly_i4_candidate(tensor, resolved) == supported, "census/dispatch eligibility");
    }
}
}
int main() {
    try {
        admission();
        registry_lifetime();
        identities_and_census();
        std::cout << "onfly route host regressions passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
