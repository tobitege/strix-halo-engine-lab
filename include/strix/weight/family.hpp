#pragma once
#include "levers.hpp"
namespace strix::weight {
struct FamilyPlanInputs {
    const Registry* registry = nullptr;
    WeightMap weights;
    Hardware hardware{};
    Workload workload{};
    std::vector<Candidate> candidates;
    std::vector<Qualification> qualifications;
    std::vector<ArithmeticId> arithmetic;
    std::vector<KernelId> incumbents;
    Exactness permission = Exactness::legacy_bitwise;
    bool allow_fast = false;
    const char* measured_decisions_override = nullptr;
    // Verified identity of the current executable; zero keeps incumbent-only planning.
    Digest256 build_identity{};
};
struct FamilyPlans {
    WeightMap weights;
    std::vector<Plan> plans;
    std::vector<Status> status;
    std::string description;
};
Status resolve_family_weights(const FamilyPlanInputs&, FamilyPlans&);
inline std::string family_plan_notice(const std::shared_ptr<const FamilyPlans>& plans) {
    return plans ? plans->description : "weight-plans=pending tensor admission; incumbent launches retained; dependency=loaded WeightMap and backend registrations; fixture=legacy byte captures; removal=bind admitted tensors\n";
}
// Attach to the canonical family configuration shared by CLI and serving. No defaults
// are repeated here. Declaration-only inspection has no tensor index and stays pending.
template<class Config> Status attach_weight_plans(Config& config, const FamilyPlanInputs& input) {
    auto result=std::make_shared<FamilyPlans>();
    auto status=resolve_family_weights(input,*result);
    if (!status && result->description.empty()) result->description="weight-plans=unsupported reason="+status.detail+"\n";
    config.weight_plans=std::move(result);
    return status;
}
}
