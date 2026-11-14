#pragma once
// Load-cohort schedules of a staged CTA tile pipeline.
//
// A cohort is a set of raw global loads issued together. Its destination registers are not read
// until the cohort is published (decoded if the format needs it, then stored to the stage's LDS
// buffer). A schedule is data: it says which cohorts a staged kernel issues ahead of its matrix
// work and how many stages and LDS buffers that keeps live. The kernel keeps three operations,
//   issue(stage, Set<c>{})            raw loads of that stage, no use of their destinations
//   publish(stage, buffer, Set<c>{})  consume the cohorts' registers into LDS buffer `buffer`
//   sync                              the CTA barrier that hands a published buffer to all waves
// and orders them as the schedule says. A schedule never changes an operand, an address or the
// arithmetic: every schedule of one kernel must produce the same bits. The kernel names its
// schedule through its contract (`Contract::Schedule`), so the load order is a trait of the
// instantiation and not code inside one kernel body.
#include <type_traits>

namespace strix::core::pipeline {
enum Cohort : unsigned { cohort_a = 1U, cohort_b = 2U, cohort_all = cohort_a | cohort_b };
// A cohort set as a type: the kernel selects its loads with `if constexpr`, so a schedule adds no
// branch and no operand to the kernel body.
template<unsigned Mask> struct Set { static constexpr unsigned mask = Mask; };
template<class S> constexpr bool has(unsigned cohort) { return (S::mask & cohort) != 0U; }

// One stage of register lookahead, two LDS stage buffers, one barrier per stage:
//   prologue  issue(0) -> publish(0, 0) -> sync
//   stage s   issue(s+1, early) -> matrix work on buffer s&1 -> issue(s+1, late) -> publish(s+1, (s&1)^1) -> sync
// The early cohorts are in flight across the whole matrix work of the current stage; the publish
// is their first use, so the only memory wait of a steady-state stage sits after its matrix work.
// cohort_all (everything early) is the production schedule. The others expose a cohort's memory
// latency once per stage and exist so a fixture can prove schedule independence and measure what
// the overlap is worth.
template<unsigned Early = cohort_all> struct CtaStageAhead {
    static_assert((Early & ~unsigned(cohort_all)) == 0U);
    static constexpr int lookahead = 1;       // stages held in registers ahead of their LDS publication
    static constexpr int lds_stages = 2;      // simultaneously live LDS stage buffers
    static constexpr unsigned early = Early;                              // issued before the matrix work
    static constexpr unsigned late = unsigned(cohort_all) & ~Early;       // issued after it, before the publish
};

// Contract trait: the contract's `Schedule`, else the stage-ahead schedule with every cohort early.
template<class C, class = void> struct contract_schedule { using type = CtaStageAhead<>; };
template<class C> struct contract_schedule<C, std::void_t<typename C::Schedule>> { using type = typename C::Schedule; };
}
