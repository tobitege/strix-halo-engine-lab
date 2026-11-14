#pragma once

#include "strix/hip_check.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/execution_resources.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/serial_host_state_shadow.hpp"

#include <memory>

// Owned for the complete request. A token's source owners, device record,
// host D2H record and shadow histories survive every async operation and drain.
class HipDeferredMoeStatus final {
public:
    HipDeferredMoeStatus(const strix::ExecutionRuntimeResources& resources,
        std::uint64_t execution_identity, std::uint64_t status_generation,
        std::uint64_t shadow_generation, std::shared_ptr<const void> resource_owner,
        std::shared_ptr<const strix::HipSetupQuiescenceGuard> guard)
        : resource_owner_(std::move(resource_owner)), guard_(std::move(guard)),
          shadow_(layout(resources), state(resources), shadow_generation), resources_(resources) {
        if (!resource_owner_ || !guard_ || !execution_identity || !status_generation)
            throw std::invalid_argument("deferred MoE requires complete retained owners and generations");
        resources_.session_state_memory=&shadow_.binding();
        try {
            guard_->require_original();
            strix::hip_check(hipStreamCreateWithFlags(&scope_.stream,hipStreamNonBlocking),"create deferred MoE stream");
            strix::hip_check(hipMalloc(&device_,sizeof(host_)),"allocate deferred MoE first-error record");
            scope_.deferred_moe_status={execution_identity,reinterpret_cast<std::uintptr_t>(device_),sizeof(host_),status_generation};
        } catch (...) {
            guard_->recover_or_fail_stop();
            if(device_)(void)hipFree(device_);
            if(scope_.stream)(void)hipStreamDestroy(scope_.stream);
            throw;
        }
    }
    ~HipDeferredMoeStatus() {
        abort();
        // Teardown may run after an exception changed the active device.
        try { guard_->require_original(); } catch (...) { guard_->recover_or_fail_stop(); }
        if(device_)(void)hipFree(device_);
        if(scope_.stream)(void)hipStreamDestroy(scope_.stream);
    }
    HipDeferredMoeStatus(const HipDeferredMoeStatus&)=delete;
    HipDeferredMoeStatus& operator=(const HipDeferredMoeStatus&)=delete;
    void begin() {
        if(pending_)throw std::logic_error("deferred MoE token already pending");
        guard_->require_original();
        shadow_.begin(); host_={};pending_=true;
        // This conservative reset still precedes serial preflight. Moving it to
        // construction plus a dirty-state protocol needs owner-level recovery
        // injection coverage; do not infer zero queue work from a preflight error.
        try { strix::hip_check(hipMemsetAsync(device_,0,sizeof(host_),scope_.stream),"clear deferred MoE first-error record"); }
        catch (...) { abort();throw; }
    }
    void complete_before_commit() {
        if(!pending_)throw std::logic_error("deferred MoE completion without active token");
        try {
            strix::hip_check(hipMemcpyAsync(&host_,device_,sizeof(host_),hipMemcpyDeviceToHost,scope_.stream),"read deferred MoE first-error record");
            strix::hip_check(hipStreamSynchronize(scope_.stream),"drain deferred MoE token");
            guard_->require_original();pending_=false;
            strix::check_deferred_moe_status(host_);
            shadow_.publish();++completed_tokens_;
        } catch (...) {
            abort();throw;
        }
    }
    void abort() noexcept {
        // This recovery must precede destruction of host_, shadow_ and all pins.
        if(pending_)guard_->recover_or_fail_stop();
        pending_=false;shadow_.cancel();
    }
    [[nodiscard]] const strix::HipLaunchContext& scope() const noexcept { return scope_; }
    [[nodiscard]] const strix::ExecutionRuntimeResources& resources() const noexcept { return resources_; }
    [[nodiscard]] std::uint64_t completed_tokens() const noexcept { return completed_tokens_; }
private:
    static const strix::SerialSessionStateLayout& layout(const strix::ExecutionRuntimeResources& r) {
        if(!r.session_state_layout)throw std::invalid_argument("deferred MoE missing state layout");
        return *r.session_state_layout;
    }
    static const strix::SessionStateMemoryBinding& state(const strix::ExecutionRuntimeResources& r) {
        if(!r.session_state_memory)throw std::invalid_argument("deferred MoE missing state memory");
        return *r.session_state_memory;
    }
    std::shared_ptr<const void> resource_owner_;
    std::shared_ptr<const strix::HipSetupQuiescenceGuard> guard_;
    strix::SerialHostStateShadow shadow_;
    strix::ExecutionRuntimeResources resources_;
    strix::HipLaunchContext scope_;
    void* device_=nullptr;
    strix::DeferredMoeFirstError host_;
    bool pending_=false;
    std::uint64_t completed_tokens_=0;
};
