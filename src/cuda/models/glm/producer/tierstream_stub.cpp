// Built only when no tierstream source is configured (GLM_TIERSTREAM_SRC): the
// tierstream backend and the whole-device budgets are refused at startup instead
// of silently degrading (the planner and the ledger live in tierstream only).
#include "tierstream_binding.hpp"
#include <stdexcept>
namespace strix::glm::rtxpf::producer {
namespace {[[noreturn]] void absent(){throw std::runtime_error("this build has no tierstream library (configure GLM_TIERSTREAM_SRC)");}}
std::unique_ptr<TierCacheApi> open_tierstream(const TierOpenConfig&){absent();}
std::pair<int,std::string> tierstream_admission_probe(const TierOpenConfig&){return {-1,"no tierstream library"};}
TierBudget tier_budget(const TierBudgetIn&){absent();}
WholeDevicePlan plan_whole_device(const WholeDeviceIn&){absent();}
bool budget_library_available(){return false;}
const char* tierstream_commit(){return "none";}
bool tierstream_supports_make_resident(){return false;}
BudgetLedger::BudgetLedger(std::uint64_t,std::uint64_t){absent();}
BudgetLedger::~BudgetLedger()=default;
bool BudgetLedger::add(MemKind,const std::string&,std::uint64_t,std::string*){absent();}
void BudgetLedger::sub(MemKind,const std::string&,std::uint64_t)noexcept{}
void BudgetLedger::observe(const std::string&,std::uint64_t,std::uint64_t){absent();}
bool BudgetLedger::within_caps()const{absent();}
std::uint64_t BudgetLedger::current(MemKind)const{absent();}
Json BudgetLedger::stats()const{absent();}
}
