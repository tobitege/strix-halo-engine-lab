#pragma once

#include "strix/compiled_execution.hpp"
#include "strix/compiled_speculative_verify.hpp"
#include "strix/execution_runtime.hpp"
#include "strix/execution_telemetry.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace strix {

using ExecutionLaunchFunction = void (*)(
    const ExecutionDispatchContext& context,
    void* user_data);

struct ExecutionLauncher {
    std::string key;
    std::uint64_t implementation_identity = 0;
    ExecutionLaunchFunction function = nullptr;
    void* user_data = nullptr;
    // Optional metadata-only admission; ordinary dispatch does not invoke it.
    ExecutionLaunchFunction preflight = nullptr;
};

class ExecutionLauncherRegistry {
public:
    void add_kernel_launcher(
        ExecutionLauncher launcher);

    void add_runtime_launcher(
        SerialOp op,
        ExecutionLauncher launcher);

    [[nodiscard]] const ExecutionLauncher*
    find_kernel_launcher(
        std::string_view kernel_id) const noexcept;

    [[nodiscard]] const ExecutionLauncher*
    find_runtime_launcher(
        SerialOp op) const noexcept;

private:
    struct RuntimeLauncherEntry {
        SerialOp op = SerialOp::position_prepare;
        ExecutionLauncher launcher;
    };

    std::vector<ExecutionLauncher> kernel_launchers_;
    std::vector<RuntimeLauncherEntry> runtime_launchers_;
};

struct CompiledStepLauncher {
    std::uint32_t step_id = 0;
    SerialOp op = SerialOp::token_embedding;
    bool runtime_step = false;
    std::uint64_t implementation_identity = 0;
    ExecutionLaunchFunction function = nullptr;
    void* user_data = nullptr;
    // Optional metadata-only admission; ordinary dispatch does not invoke it.
    ExecutionLaunchFunction preflight = nullptr;
};

struct MissingLauncher {
    std::uint32_t step_id = 0;
    SerialOp op = SerialOp::token_embedding;
    bool runtime_step = false;
    std::string key;
};

struct LauncherCoverage {
    std::uint32_t covered_steps = 0;
    std::uint32_t missing_steps = 0;
    std::vector<MissingLauncher> missing;
};

[[nodiscard]] LauncherCoverage audit_launcher_coverage(
    const CompiledSerialExecution& compiled,
    const ExecutionLauncherRegistry& launchers);

[[nodiscard]] LauncherCoverage audit_launcher_coverage(
    const CompiledSpeculativeVerifyExecution& compiled,
    const ExecutionLauncherRegistry& launchers);

struct CompiledLaunchPlan {
    std::uint64_t execution_identity = 0;
    std::uint64_t plan_identity = 0;
    std::vector<CompiledStepLauncher> steps;
};

[[nodiscard]] CompiledLaunchPlan compile_launch_plan(
    const CompiledSerialExecution& compiled,
    const ExecutionLauncherRegistry& launchers);

[[nodiscard]] CompiledLaunchPlan compile_launch_plan(
    const CompiledSpeculativeVerifyExecution& compiled,
    const ExecutionLauncherRegistry& launchers);

class DirectLaunchDispatcher final
    : public EagerExecutionDispatcher {
public:
    explicit DirectLaunchDispatcher(
        const CompiledLaunchPlan& plan,
        ExecutionTelemetry* telemetry = nullptr)
        : plan_(plan), telemetry_(telemetry) {}

    void dispatch_kernel(
        const ExecutionDispatchContext& context) override;

    void dispatch_runtime_step(
        const ExecutionDispatchContext& context) override;

    // The validated kernel launcher dispatch_kernel(context) would call, for a caller that launches several rows
    // of one step at once (rows_launch_for); null when this dispatcher reports telemetry per dispatch.
    [[nodiscard]] const CompiledStepLauncher* kernel_launcher(const ExecutionDispatchContext& context) const {
        return telemetry_ ? nullptr : &launcher_for(context, false);
    }

private:
    void dispatch(const ExecutionDispatchContext&, bool runtime_step);
    [[nodiscard]] const CompiledStepLauncher&
    launcher_for(
        const ExecutionDispatchContext& context,
        bool runtime_step) const;

    const CompiledLaunchPlan& plan_;
    ExecutionTelemetry* telemetry_ = nullptr;
};

}  // namespace strix
