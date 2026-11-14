#include "strix/mtp_cohort_partition.hpp"
#include "strix/mtp_window_controller.hpp"
#include "strix/mtp_batch_window_controller.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {
int failures=0;
void expect(bool pass,const char* message) {
    if(!pass) {std::cerr<<"FAIL: "<<message<<'\n';++failures;}
}
using strix::MtpMinPWindowController;
using strix::MtpWindowCostEstimate;
using Model=MtpWindowCostEstimate::Model;

struct Run {std::map<std::uint32_t,std::uint64_t> rows;std::uint64_t tokens=0,probes=0;double ms=0.0;};
// One synthetic decode against the controller, driven like HipMtpGreedyRuntime + the verifier:
// - a chain of up to 7 drafts stops after the first draft below min-p or when worth_drafting says no;
// - prefix windows of 2/4/8 rows (padding allowed), every draft at or above min-p is right;
// - cycle wall time = owner cost + 1 ms per draft; serial rows cost 45 ms each.
// script[c] gives the draft probabilities of cycle c (the last entry repeats); spike replaces the n-th cost
// sample of R4 (the second R4 window: the first one is the warm-up).
Run simulate(Model model,const std::vector<std::vector<float>>& script,std::size_t cycles,double r4_spike) {
    MtpMinPWindowController c({2U,4U,8U},0.8F,model);
    const std::array<double,3> cost{45.0,55.0,75.0};
    Run out;std::uint32_t r4_windows=0;
    for(std::size_t cycle=0;cycle<cycles;++cycle) {
        const auto& p_all=script[std::min(cycle,script.size()-1U)];
        std::vector<float> p;
        for(std::size_t k=0;k<7U;++k) {
            p.push_back(p_all[std::min(k,p_all.size()-1U)]);
            if(k+1U<7U && (p.back()<0.8F || !c.worth_drafting(p))) break;
        }
        std::size_t accepted=0;while(accepted<p.size() && p[accepted]>=0.8F) ++accepted;
        const std::array<std::uint8_t,3> fits{1U,1U,1U};
        const auto chosen=c.choose(p,p.size(),fits);
        std::size_t emitted;double ms;
        if(chosen==MtpMinPWindowController::serial) {
            emitted=accepted+1U;ms=45.0*static_cast<double>(emitted)+static_cast<double>(p.size());
            c.calibrate(p,accepted,emitted);
            if(cycle>=100U) ++out.rows[1];
        } else {
            const std::uint32_t rows=chosen==0?2U:chosen==1?4U:8U;
            emitted=std::min<std::size_t>(accepted,rows-1U)+1U;
            ms=cost[chosen]+static_cast<double>(p.size());
            if(rows==4U && ++r4_windows==2U) ms=r4_spike;
            c.calibrate(p,emitted-1U,rows-1U);
            if(cycle>=100U) ++out.rows[rows];
        }
        c.observe(chosen,emitted,ms);
        if(cycle>=100U) {out.tokens+=emitted;out.ms+=ms;}
    }
    for(const auto& s:c.owner_stats()) out.probes+=s.probes;
    return out;
}
double share(const Run& r,std::uint32_t rows) {
    std::uint64_t total=0;for(const auto& [k,v]:r.rows) total+=v;
    return total?static_cast<double>(r.rows.count(rows)?r.rows.at(rows):0U)/static_cast<double>(total):0.0;
}
} // namespace

int main() {
    for (const std::size_t count : {0U, 1U, 2U, 8U, 9U, 16U, 19U}) {
        for (std::size_t round = 0; round < count + 2U; ++round) {
            std::vector<unsigned> seen(count, 0U);
            strix::MtpBatchWindowController controller(count);
            for (const auto& group : strix::mtp_cohort_partition(count, round)) {
                expect(!group.empty() && group.size() <= 8U, "bounded joint group");
                std::vector<strix::MtpBatchWindowController::Request> requests;
                for (const auto slot : group) {
                    ++seen.at(slot);
                    requests.push_back({static_cast<std::uint32_t>(slot), 2U, {}});
                }
                const auto plan = controller.plan(requests, 32U);
                expect(plan.size() == group.size(), "controller accepts partition with stable global slot IDs");
            }
            expect(std::all_of(seen.begin(), seen.end(), [](unsigned n) { return n == 1U; }), "round covers every live slot once");
        }
    }

    {
        MtpWindowCostEstimate legacy(Model::legacy);
        legacy.add(100.0);legacy.add(300.0);
        expect(std::abs(legacy.value()-125.0)<1e-9,"legacy estimate: first sample, then EMA 0.25 with the 2x cap");
        MtpWindowCostEstimate robust(Model::robust);
        robust.add(111.0);expect(robust.provisional() && robust.value()==111.0,"robust: one sample is provisional");
        robust.add(55.0);expect(robust.value()==55.0,"robust: provisional estimate is the minimum, a spike cannot inflate it");
        robust.add(56.0);robust.add(54.0);robust.add(500.0);
        expect(!robust.provisional() && robust.value()==56.0,"robust: median of recent samples ignores one outlier");
        for(int i=0;i<8;++i) robust.add(60.0);
        expect(robust.value()==60.0,"robust: the window of 8 samples follows a lasting change");
    }
    // The stall seen on hardware (code prompt, min-p controller): R4's first cost sample is a 111 ms spike, R8
    // gets only its warm-up window, R2 is measured next; from then on worth_drafting stops every chain after
    // one draft because R4 looks too slow, R8 is never reached and every cycle is R2.
    std::vector<std::vector<float>> script{
        {0.99F,0.99F,0.99F,0.5F}, {0.99F,0.99F,0.99F,0.5F}, // R4 warm-up, then the 111 ms sample
        {0.99F},                                           // full chain: R8 warm-up (not a sample)
        {0.5F}, {0.5F},                                    // weak first drafts: R2 warm-up and sample
        {0.99F}};                                          // then a confident workload
    const auto stuck=simulate(Model::legacy,script,400,111.0);
    expect(share(stuck,2)>0.95,"legacy: the spike keeps every later cycle in R2 (stall reproduced)");
    const auto fixed=simulate(Model::robust,script,400,111.0);
    expect(share(fixed,2)<0.05 && share(fixed,8)>0.9,"robust: re-probes recover R4/R8 and the workload runs in R8");
    expect(fixed.probes>=1 && fixed.probes<=8,"robust: probes are few and bounded");
    const double stuck_rate=static_cast<double>(stuck.tokens)/stuck.ms,fixed_rate=static_cast<double>(fixed.tokens)/fixed.ms;
    expect(fixed_rate>1.5*stuck_rate,"robust: tokens per ms recover");
    // Without the spike both cost models settle on the same windows.
    const auto clean_legacy=simulate(Model::legacy,script,400,55.0),clean_robust=simulate(Model::robust,script,400,55.0);
    expect(share(clean_legacy,8)>0.9 && share(clean_robust,8)>0.9,"no spike: both models run the confident workload in R8");
    // A probe never picks an owner that does not fit.
    {
        MtpMinPWindowController c({2U,4U,8U},0.8F,Model::robust,4);
        const std::vector<float> p{0.5F};
        const std::array<std::uint8_t,3> only_r2{1U,0U,0U};
        for(int i=0;i<40;++i) {
            const auto o=c.choose(p,1U,only_r2);
            expect(o==0U || o==MtpMinPWindowController::serial,"probe respects the fits mask");
            c.observe(o,1U,45.0);
        }
    }
    // A slot first observes cheap R2 cycles and slow deeper windows (for example
    // while other requests are scheduled), then serves a confident request.
    // Those individually plausible costs can suppress all deeper drafting: a
    // larger owner's fit-mask re-probe cannot help if the chain never reaches it.
    for(const auto model:{Model::legacy,Model::robust}) {
        MtpMinPWindowController reused({2U,4U,8U},0.7F,model,4);
        for(int k=0;k<10;++k) {
            reused.observe(0,2,20.0);reused.observe(1,4,200.0);reused.observe(2,8,300.0);
        }
        const std::vector<float> p{0.99F};
        expect(!reused.worth_drafting(p),"retained workload costs stop the next request at one draft");
        for(int k=0;k<200;++k)reused.calibrate(p,0,1);
        reused.set_external_rows(0,std::array<double,1>{0.01});
        reused.begin_session();
        MtpMinPWindowController fresh({2U,4U,8U},0.7F,model,4);
        expect(reused.worth_drafting(p),"request reset reopens deeper drafting from the prior");
        for(const auto& s:reused.owner_stats())
            expect(!s.cycles && !s.probes && s.cycle_ms==0.0,"reset clears costs and exploration history");
        const std::array<std::uint8_t,3> all{1U,1U,1U};
        const std::vector<float> chain{0.99F,0.85F,0.72F,0.4F};
        for(int k=0;k<80;++k) {
            const auto a=reused.choose(chain,chain.size(),all),b=fresh.choose(chain,chain.size(),all);
            expect(a==b,"reused request choices equal a fresh controller after reset");
            expect(reused.worth_drafting(p)==fresh.worth_drafting(p),"reset restores acceptance and external-row priors");
            reused.calibrate(chain,2,3);fresh.calibrate(chain,2,3);
            reused.observe(a,3,55.0);fresh.observe(b,3,55.0);
        }
    }
    // An implausible middle owner can also stay trapped: one fast re-probe
    // cannot displace a full median window of slow samples, and the larger
    // owner is never drafted even though its retained cost is competitive.
    {
        MtpMinPWindowController reused({2U,4U,8U},0.7F,Model::robust,64);
        for(int k=0;k<10;++k) {
            reused.observe(0,2,20.0);reused.observe(1,4,90.0);reused.observe(2,8,35.0);
        }
        const std::vector<float> p{0.99F};
        const std::array<std::uint8_t,3> all{1U,1U,1U};
        std::size_t probes=0,large=0;
        for(int k=0;k<128;++k) {
            const auto o=reused.choose(p,1,all);
            if(o==1U)++probes;
            if(o==2U)++large;
            reused.observe(o,2,o==0U?20.0:o==1U?25.0:35.0);
        }
        expect(probes==1U && large==0U,"a short reused session only gets one middle-owner re-probe");
        expect(reused.owner_stats()[2].cycle_ms==90.0 && !reused.worth_drafting(p),
            "one fast probe leaves the slow median and the deeper draft barrier intact");
        reused.begin_session();
        expect(reused.worth_drafting(p),"a new request clears the contaminated median without waiting for more probes");
    }
    {
        strix::MtpBatchWindowController joint(4);
        const std::vector<float> high(7,0.99F),low(7,0.01F);
        std::vector<strix::MtpBatchWindowController::Request> q{{0,8,high},{1,8,high}};
        const auto good=joint.choose(q,16);
        expect(good==std::vector<std::uint32_t>({8,8}),"joint selection retains deep, high acceptance windows");
        q[0].probabilities=low;q[1].probabilities=low;
        expect(joint.choose(q,16)==std::vector<std::uint32_t>({2,2}),"joint selection avoids low acceptance padding");
        q[0].limit=4;q[1].limit=2;q[0].probabilities=high;
        expect(joint.choose(q,16)==std::vector<std::uint32_t>({2,2}),"six physical rows cannot fall through to serial chunks");
        const std::vector<strix::MtpBatchWindowController::Request> tails{{0,2,{}},{1,2,{}},{2,2,{}}};
        expect(joint.choose(tails,16)==std::vector<std::uint32_t>({2,2,2}),"three R2-only tails retain admissible chunks");
        const std::vector<strix::MtpBatchWindowController::Request> triple{{0,8,{}},{1,8,{}},{2,8,{}}};
        const auto probe=joint.plan(triple,16);
        expect(probe==std::vector<std::uint32_t>({4,2,2}),"three-request exploration pads a valid eight-row cohort");
        q={{0,8,high},{1,8,high},{2,8,high},{3,8,high}};
        const auto bounded=joint.choose(q,16);
        std::uint32_t total=0;for(auto r:bounded) total+=r;
        expect(total<=16U,"joint exhaustive search respects the global row budget");
        q.resize(2);q[0].probabilities=low;q[1].probabilities=low;
        const std::array<std::uint32_t,2> rows{8,8},miss{0,0};
        for(int n=0;n<100;++n) joint.observe(q,rows,miss,200.0,30.0,14U);
        for(int k=0;k<9;++k) static_cast<void>(joint.plan(q,16)); // bounded exploration
        const auto no_draft=joint.plan(q,16);
        expect(no_draft==std::vector<std::uint32_t>({0,0}),"joint planning skips expensive unproductive drafts");
        joint.reset(0);joint.reset(1);
    }
    {
        // Plan rules (STRIX_MTP_JOINT_PLAN): a cap bounds every planned owner, probes included; the pooled
        // prior follows the request's own measured acceptance where deep history is thin.
        strix::MtpBatchWindowController::Rule capped;capped.cap=4;
        strix::MtpBatchWindowController cap(4,capped),production(4);
        const std::vector<strix::MtpBatchWindowController::Request> q{{0,8,{}},{1,8,{}},{2,8,{}},{3,8,{}}};
        for(int k=0;k<40;++k) {
            const auto planned=cap.plan(q,32);
            expect(std::all_of(planned.begin(),planned.end(),[](std::uint32_t r){return r<=4U;}),"cap4 plans no R8 owner");
        }
        expect(production.plan(q,32)==std::vector<std::uint32_t>({2,2,2,2}) && production.last_plan().probe,
            "the first production plan is the R2 exploration probe");
        strix::MtpBatchWindowController::Rule pool;pool.pooled=true;
        strix::MtpBatchWindowController pooled(2,pool),flat(2);
        const std::vector<strix::MtpBatchWindowController::Request> two{{0,8,{}},{1,8,{}}};
        const std::array<std::uint32_t,2> r2{2,2};
        const std::vector<float> one(1,0.5F);
        const std::vector<strix::MtpBatchWindowController::Request> seen{{0,8,one},{1,8,one}};
        // Depth 0 is seen 60 times and accepted 30 % of the time; deeper depths are never reached.
        for(int n=0;n<60;++n) {
            const std::array<std::uint32_t,2> acc{n%10<3?1U:0U,n%10<3?1U:0U};
            pooled.observe(seen,r2,acc,2.0,48.0,2U);flat.observe(seen,r2,acc,2.0,48.0,2U);
        }
        for(int k=0;k<9;++k) {static_cast<void>(pooled.plan(two,16));static_cast<void>(flat.plan(two,16));}
        const auto a=pooled.plan(two,16),b=flat.plan(two,16);
        expect(pooled.last_plan().expected<flat.last_plan().expected,"the pooled prior is below 0.8 at 30 % acceptance");
        expect(a[0]<=b[0] && a[1]<=b[1],"the pooled prior never plans deeper than the 0.8 prior");
    }
    {
        // Calibrated choose: confident-looking drafts (p 0.95) that the run keeps rejecting stop earning rows; the
        // production blend follows the per-depth history instead. Rule text round trip and in-process switch.
        const auto parsed=strix::MtpBatchWindowController::parse_rule("calibrated,cap4");
        expect(parsed.calibrated && parsed.cap==4U && !parsed.pooled,"rule text parses");
        const auto none=strix::MtpBatchWindowController::parse_rule("");
        expect(!none.calibrated && !none.pooled && none.cap==8U,"empty rule text = production");
        strix::MtpBatchWindowController c(2);
        const std::vector<float> sure(7,0.95F);
        const std::vector<strix::MtpBatchWindowController::Request> q{{0,8,sure},{1,8,sure}};
        const std::array<std::uint32_t,2> r8{8,8},deep{6,6},none_acc{0,0};
        // 60 cycles of p 0.5 drafts accepted deep (6 of 7) make the depth history favourable; 40 cycles where every
        // p 0.95 draft fails make the 0.95 bin bad.
        const std::vector<float> half(7,0.5F);
        const std::vector<strix::MtpBatchWindowController::Request> h{{0,8,half},{1,8,half}};
        for(int n=0;n<60;++n) c.observe(h,r8,deep,1.0,60.0,14U);
        for(int n=0;n<40;++n) c.observe(q,r8,none_acc,1.0,60.0,14U);
        const auto blended=c.choose(q,16);
        strix::MtpBatchWindowController::Rule cal;cal.calibrated=true;c.set_rule(cal);
        const auto calibrated=c.choose(q,16);
        expect(blended==std::vector<std::uint32_t>({8,8}),"the production blend follows the favourable depth history");
        expect(calibrated==std::vector<std::uint32_t>({2,2}),"calibrated choose verifies R2 for drafts that keep failing");
        std::vector<strix::MtpBatchWindowController::Request> ctx{{0,8,sure},{1,8,sure}};
        ctx[0].external_first=0;ctx[1].external_first=0;
        expect(c.choose(ctx,16)==blended,"context rows keep the production estimate under the calibrated rule");
    }
    {
        using Controller=strix::MtpBatchWindowController;
        const char* previous=std::getenv("STRIX_MTP_JOINT_PLAN");
        const bool had_previous=previous!=nullptr;
        const std::string saved=previous?previous:"";
        const auto set_env=[](const char* value) {
#ifdef _WIN32
            _putenv_s("STRIX_MTP_JOINT_PLAN",value?value:"");
#else
            if(value) setenv("STRIX_MTP_JOINT_PLAN",value,1);
            else unsetenv("STRIX_MTP_JOINT_PLAN");
#endif
        };
        set_env(nullptr);
        const auto unset_default=Controller::rule_from_env();
        expect(unset_default.budget && !unset_default.calibrated,"unset env defaults to the row budget");
        const auto automatic=Controller::parse_rule("calibrated-c3");
        expect(automatic.calibrated && automatic.calibrated_min_members==3U && !automatic.budget,"calibrated-c3 parses to three plan members");
        expect(!Controller::rule_from_env(true).calibrated,"baseline defaults to production");
        for(const auto* text:{"production","0","calibrated","calibrated-c3"}) {
            set_env(text);
            const auto rule=Controller::rule_from_env(true);
            const auto parsed=Controller::parse_rule(text);
            expect(rule.calibrated==parsed.calibrated && rule.calibrated_min_members==parsed.calibrated_min_members,
                "explicit env rule wins over baseline and uses the rule-file parser");
        }
        set_env(had_previous?saved.c_str():nullptr);
        const auto combined=Controller::parse_rule("calibrated-c3,cap4 wave=3 restore");
        expect(combined.calibrated && combined.calibrated_min_members==3U && combined.cap==4U,
            "gated rule parses with cap and rule-file wave metadata");
        for(const auto* text:{"production","0"})
            expect(!Controller::parse_rule(text).calibrated,"production and 0 restore the old rule");

        // Identical trained histories make production and calibrated choose different owners.
        // Exercise group sizes on one controller, including a return below the threshold.
        Controller c(8,automatic);
        const std::vector<float> half(7,0.5F),sure(7,0.95F);
        std::vector<Controller::Request> q,h;
        for(std::uint32_t slot=0;slot<8U;++slot) {q.push_back({slot,8,sure});h.push_back({slot,8,half});}
        const std::vector<std::uint32_t> r8(8,8),deep(8,6),zero(8,0);
        for(int n=0;n<60;++n) c.observe(h,r8,deep,1.0,60.0,56U);
        for(int n=0;n<40;++n) c.observe(q,r8,zero,1.0,60.0,56U);
        for(const std::size_t count:{1U,2U,3U,4U,8U,2U}) {
            const auto group=std::span<const Controller::Request>(q).first(count);
            // Leave initial/periodic probes before checking acceptance-based choose.
            do {static_cast<void>(c.plan(group,64));} while(c.last_plan().probe);
            c.set_rule(Controller::parse_rule("production"));
            const auto production=c.choose(group,64);
            c.set_rule(Controller::parse_rule("calibrated"));
            const auto calibrated=c.choose(group,64);
            expect(production!=calibrated,"fixture distinguishes production and calibrated");
            c.set_rule(automatic);
            expect(c.choose(group,64)==(count>=3U?calibrated:production),"calibrated-c3 switches at three plan members");
            expect(c.choose(group,64)==c.choose(group,64),"choose repeats deterministically within the plan");
        }
    }
    {
        // Shared-state snapshot: plans after a restore repeat the plans after the save (costs, counters, bins).
        strix::MtpBatchWindowController c(2);
        const std::vector<strix::MtpBatchWindowController::Request> q{{0,8,{}},{1,8,{}}};
        expect(!c.restore_shared(),"restore without a snapshot is refused");
        for(int k=0;k<12;++k) static_cast<void>(c.plan(q,16));
        c.save_shared();
        std::vector<std::vector<std::uint32_t>> first,second;
        for(int k=0;k<40;++k) first.push_back(c.plan(q,16));
        expect(c.restore_shared(),"restore after a snapshot");
        for(int k=0;k<40;++k) second.push_back(c.plan(q,16));
        expect(first==second,"a restored controller replays the same plans");
    }
    {
        strix::MtpBatchWindowController growing(1), eager(5);
        const std::vector<float> probabilities(7, 0.8F);
        const std::vector<strix::MtpBatchWindowController::Request> initial{{0,8,probabilities}};
        const std::array<std::uint32_t,1> rows{8}, accepted{5};
        growing.observe(initial,rows,accepted,1.0,5.0,7);
        eager.observe(initial,rows,accepted,1.0,5.0,7);
        growing.resize_slots(5);
        growing.reset(4);eager.reset(4);
        const std::vector<strix::MtpBatchWindowController::Request> q{{0,8,probabilities},{4,8,probabilities}};
        expect(growing.choose(q,16)==eager.choose(q,16),"growth preserves old history and initializes appended slots like eager slots");
        expect(growing.plan(q,16)==eager.plan(q,16),"growth preserves shared planning state");
    }
    {
        // Budget rule: rows come from the shared row budget. Rows per lane shrink as lanes grow, every plan is
        // made of admitted owners, small totals are 4 or 8, and limits hold.
        using Controller=strix::MtpBatchWindowController;
        const auto rule=Controller::parse_rule("budget");
        expect(rule.budget && !Controller::parse_rule("calibrated-c3").budget,"budget rule name parses");
        Controller c(8,rule);
        const auto model=c.budget_cost();
        expect(!model.fitted && model.fixed_ms+model.lane_ms==34.0 && model.row_ms==6.0,"budget cost starts from the 34 + 6 x rows prior of one lane");
        const std::vector<float> sure(7,0.95F),poor(7,0.05F);
        std::uint32_t previous=UINT32_MAX;
        for(std::uint32_t lanes=1;lanes<=8U;++lanes) {
            std::vector<Controller::Request> q;
            for(std::uint32_t k=0;k<lanes;++k) q.push_back({k,8,sure});
            const auto rows=c.choose(q,32);
            std::uint32_t total=0,widest=0;
            for(const auto r:rows) {expect(r==2U || r==4U || r==8U,"budget rows are admitted owners");total+=r;widest=std::max(widest,r);}
            expect(total<=32U,"budget plan stays inside the row budget");
            expect(total>8U || total==4U || total==8U,"small budget totals are four or eight rows");
            expect(widest<=previous,"rows per lane do not grow with the lanes of a cycle");
            previous=widest;
            if(lanes==6U) expect(total>12U,"six confident lanes get more than the fixed table");
            if(lanes>6U) expect(total==2U*lanes && widest==2U,"above the lane bound every request gets the fixed two rows");
        }
        const std::vector<Controller::Request> one{{0,8,sure}},eight{{0,8,sure},{1,8,sure},{2,8,sure},{3,8,sure},{4,8,sure},
            {5,8,sure},{6,8,sure},{7,8,sure}};
        expect(c.choose(one,32).front()>c.choose(eight,32).front(),"one lane drafts deeper than eight lanes");
        const std::vector<Controller::Request> mixed{{0,8,sure},{1,8,poor},{2,8,sure},{3,8,poor}};
        const auto split=c.choose(mixed,32);
        expect(split[0]>=split[1] && split[2]>=split[3] && split[1]==2U && split[3]==2U,"rows go to the lanes whose drafts are likely");
        const std::vector<Controller::Request> limited{{0,2,sure},{1,4,sure},{2,8,sure}};
        const auto bounded=c.choose(limited,32);
        expect(bounded[0]==2U && bounded[1]<=4U,"budget rows respect each request's owner limit");
        const std::vector<Controller::Request> tails{{0,2,{}},{1,2,{}},{2,2,{}}};
        expect(c.choose(tails,32)==std::vector<std::uint32_t>({2,2,2}),"three two-row tails keep their small chunks");
        // Measured topologies replace the prior once rows and lanes moved apart: 20 + 10 x rows + 3 x lanes.
        Controller measured(8,rule);
        std::vector<Controller::Request> four,two;
        for(std::uint32_t k=0;k<4U;++k) four.push_back({k,8,sure});
        for(std::uint32_t k=0;k<2U;++k) two.push_back({k,8,sure});
        const auto before=measured.choose(four,32);
        const std::array<std::uint32_t,4> r2{2,2,2,2},r4{4,4,4,4},none{0,0,0,0};
        const std::array<std::uint32_t,2> p4{4,4},p8{8,8},none2{0,0};
        const auto ms=[](double base,double per_row,double per_lane,double row_count,double lane_count){return base+per_row*row_count+per_lane*lane_count;};
        for(int k=0;k<8;++k) {measured.observe(four,r2,none,0.0,ms(20,10,3,8,4),0);measured.observe(four,r4,none,0.0,ms(20,10,3,16,4),0);}
        expect(!measured.budget_cost().fitted,"one lane count does not identify the lane term: the prior stays");
        for(int k=0;k<8;++k) {measured.observe(two,p4,none2,0.0,ms(20,10,3,8,2),0);measured.observe(two,p8,none2,0.0,ms(20,10,3,16,2),0);}
        const auto fit=measured.budget_cost();
        expect(fit.fitted && std::abs(fit.fixed_ms-20.0)<1.0 && std::abs(fit.row_ms-10.0)<0.5 && std::abs(fit.lane_ms-3.0)<0.5,
            "budget cost follows the measured topologies");
        Controller dear(8,rule);
        for(int k=0;k<8;++k) {dear.observe(four,r2,none,0.0,ms(2,40,1,8,4),0);dear.observe(four,r4,none,0.0,ms(2,40,1,16,4),0);
            dear.observe(two,p4,none2,0.0,ms(2,40,1,8,2),0);dear.observe(two,p8,none2,0.0,ms(2,40,1,16,2),0);}
        const std::vector<float> half(7,0.5F);
        std::vector<Controller::Request> unsure;
        for(std::uint32_t k=0;k<4U;++k) unsure.push_back({k,8,half});
        std::uint32_t cheap_total=0,dear_total=0;
        for(const auto r:c.choose(unsure,32)) cheap_total+=r;
        for(const auto r:dear.choose(unsure,32)) dear_total+=r;
        expect(dear_total<=cheap_total && before.size()==4U,"dearer rows never widen the plan");
        // plan() before drafting: 0 = no draft, else an owner; the probe schedule still explores.
        Controller planner(4,rule);
        for(int k=0;k<40;++k) for(const auto r:planner.plan(four,32)) expect(r==0U || r==2U || r==4U || r==8U,"budget plan rows are owners or no draft");
    }
    if(failures) {std::cerr<<failures<<" failure(s)\n";return 1;}
    std::cout<<"mtp window controller: legacy R2 share "<<share(stuck,2)<<", robust R8 share "<<share(fixed,8)
             <<" with "<<fixed.probes<<" probes; tok/ms "<<stuck_rate<<" -> "<<fixed_rate<<'\n';
    return 0;
}
