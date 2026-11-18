#pragma once
// Precision of the DeepSeek V4.1 prefill execution in progress on the calling thread (host only, no HIP).
//
// The FAST dense / MoE dispatchers (enqueue_q8_0_a8_dense_prefill, PrefillMoe) are free functions shared by every
// caller. Their switches used to be process-wide (set_fast_dense / set_fast_moe, written by
// DevicePrefill::set_precision_for_new_sequence), so one DevicePrefill's precision changed what another one computed,
// while its group admission read its own options. Now each execution owns its precision: DevicePrefill::run_from
// installs a PrefillPrecisionScope with its options for the time of the run (on the thread that enqueues, the task
// worker included), and the dispatchers read that first. Outside any scope (kernel harnesses, decode callers) the old
// rule is unchanged: the harness override (set_fast_*: 1 / 0) or else the environment.
namespace strix::ds41 {

struct PrefillPrecision {
    bool fast_dense = false;  // ds41-dense-f16-v1
    bool fast_moe = false;    // ds41-moe-relaxed-v1
    bool fast_moe_f16 = false;  // ds41-moe-f16wmma-v1 (with fast_moe: the relaxed kernels' FP16-operand form)
};

namespace prefill_precision_detail {
inline thread_local const PrefillPrecision* active = nullptr;
}

class PrefillPrecisionScope {
public:
    explicit PrefillPrecisionScope(const PrefillPrecision& p) noexcept : own_(p), outer_(prefill_precision_detail::active) {
        prefill_precision_detail::active = &own_;
    }
    ~PrefillPrecisionScope() { prefill_precision_detail::active = outer_; }
    PrefillPrecisionScope(const PrefillPrecisionScope&) = delete;
    PrefillPrecisionScope& operator=(const PrefillPrecisionScope&) = delete;

private:
    PrefillPrecision own_;
    const PrefillPrecision* outer_;
};

// The precision of the execution on this thread, nullptr when none runs.
[[nodiscard]] inline const PrefillPrecision* active_prefill_precision() noexcept { return prefill_precision_detail::active; }

// The effective switch of a dispatcher: the running execution's own member, else mode (>= 0: harness override) or env.
[[nodiscard]] inline bool effective_fast(bool PrefillPrecision::*member, int mode, bool env) noexcept {
    if (const auto* p = prefill_precision_detail::active) return p->*member;
    return mode >= 0 ? mode != 0 : env;
}

}  // namespace strix::ds41
