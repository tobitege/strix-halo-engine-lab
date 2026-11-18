// DAYAUDIT 2 (P0): the precision of one DevicePrefill must not change another's in the same process.
// The production dispatch switches (fast_dense_on in src/hip_ds41_layer.hip, fast_moe in src/hip_ds41_prefill.hip) are
// extracted verbatim; extract.py also asserts that DevicePrefill::run_from installs its own options as the scope below.
#include "strix/ds41_prefill_precision.hpp"
#include "strix/lever_cache.hpp"
#include "strix/prefill_dense_plan.hpp"

#include <cstdio>
#include <stdexcept>
#include <thread>

namespace strix::ds41 {
namespace fast_profile {
bool dense_on() { return true; }  // the process environment asks for FAST (STRIX_DS41_FAST_PREFILL=1)
bool moe_on() { return true; }
}  // namespace fast_profile
int fast_dense_mode = -1;
int fast_moe_mode = -1;
#include "fast_dense_on.inc"
#include "fast_moe.inc"
struct DispatchView { bool exact; unsigned rows; };
bool dense_launched = false;
bool enqueue_fast_dense(const DispatchView& v, void*) {
    dense_launched = v.rows >= 128U;  // supported shape; launches are stubbed
    return dense_launched;
}
bool dispatch(const DispatchView& v) {
    void* stream = nullptr;
    dense_launched = false;
#include "dense_dispatch.inc"
    return true;
}
}  // namespace strix::ds41

namespace {
namespace ds41 = strix::ds41;
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
// what DevicePrefill::run_from does with its options (asserted textually by extract.py), and the routes it then takes
struct Object {
    bool fast_dense, fast_moe;
    struct Routes { bool dense, moe; };
    Routes run() const {
        const struct { ds41::PrefillPrecision options; } m{{fast_dense, fast_moe}};
#include "execution_precision.inc"
        return {ds41::fast_dense_on(), ds41::fast_moe()};
    }
    bool planned_fast() const {
        const ds41::PrefillPrecision options{fast_dense, fast_moe};
#include "planner_fast.inc"
    }
    bool dispatch(bool grouped) const {
        const struct { ds41::PrefillPrecision options; } m{{fast_dense, fast_moe}};
#include "execution_precision.inc"
        strix::prefill_plan::DenseRule rule;
        rule.fast = planned_fast();
        rule.fast_min_rows = 128U;
        const bool exact = grouped && rule.route(160U) != strix::prefill_plan::DenseRoute::dense_fast;
        (void)ds41::dispatch({exact, grouped ? 320U : 160U});
        return ds41::dense_launched;
    }
};
}  // namespace

int main() try {
    Object first{false, false}, second{false, false};
    for (const int mode : {-1, 0, 1}) {
        ds41::fast_dense_mode = mode;
        first.fast_dense = true;  // serialized OFF-to-ON of A leaves B's options alone
        require(first.dispatch(false) && first.dispatch(true), "A solo/group take FAST despite override");
        require(!second.dispatch(false) && !second.dispatch(true), "B solo/group stay exact after A changes");
        second.fast_dense = true;
        require(second.dispatch(false) && second.dispatch(true), "B solo/group switch together to FAST");
        second.fast_dense = false;
    }
    const Object a{true, true}, b{false, false};
    // the old set_precision_for_new_sequence of A wrote the process-wide modes; leave them as A left them
    ds41::fast_dense_mode = ds41::fast_moe_mode = 1;
    const auto ra = a.run();
    const auto rb = b.run();
    require(ra.dense && ra.moe, "A (FAST) runs FAST");
    require(!rb.dense && !rb.moe, "B (exact) after A runs exact: its own precision, not A's nor the environment's");
    ds41::fast_dense_mode = ds41::fast_moe_mode = 0;  // the reverse: A switched FAST off process-wide
    const auto ra2 = a.run();
    require(ra2.dense && ra2.moe, "A keeps FAST when another owner turned the process-wide modes off");
    {   // nesting restores the outer execution; no scope = the harness override / environment rule
        const ds41::PrefillPrecisionScope outer(ds41::PrefillPrecision{true, false});
        require(!b.run().dense && ds41::fast_dense_on() && !ds41::fast_moe(), "inner scope ends, outer precision back");
    }
    require(!ds41::fast_dense_on() && !ds41::fast_moe(), "outside a run: override 0 wins over the environment");
    ds41::fast_dense_mode = ds41::fast_moe_mode = -1;
    require(ds41::fast_dense_on() && ds41::fast_moe(), "outside a run: the environment");
    bool other_thread = false;
    {
        const ds41::PrefillPrecisionScope mine(ds41::PrefillPrecision{false, false});
        std::thread([&] { other_thread = ds41::fast_dense_on(); }).join();  // a concurrent run elsewhere: not mine
        require(!ds41::fast_dense_on(), "own thread: own scope");
    }
    require(other_thread, "the scope is per thread of execution");
    std::puts("dayaudit2 precision per execution: ok");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
}
