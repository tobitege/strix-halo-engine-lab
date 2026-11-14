#include "strix/mtp_window_controller.hpp"
#include "strix/mtp_batch_window_controller.hpp"
#include "strix/serve/row_budget.hpp"
#include "strix/env_flags.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace strix {
void MtpWindowCostEstimate::add(double ms) {
    if(model_==Model::legacy) {
        constexpr double alpha=0.25;
        value_=samples_?value_+alpha*(std::min(ms,2.0*value_)-value_):ms;
        ++samples_;return;
    }
    recent_[samples_%recent_.size()]=ms;++samples_;
    const auto n=static_cast<std::size_t>(std::min<std::uint64_t>(samples_,recent_.size()));
    std::array<double,8> sorted{};
    for(std::size_t i=0;i<n;++i) { // insertion sort of at most 8 samples
        std::size_t j=i;
        while(j && sorted[j-1U]>recent_[i]) {sorted[j]=sorted[j-1U];--j;}
        sorted[j]=recent_[i];
    }
    value_=provisional()?sorted[0]:(n%2U?sorted[n/2U]:0.5*(sorted[n/2U-1U]+sorted[n/2U]));
}

MtpMinPWindowController::MtpMinPWindowController(std::vector<std::uint32_t> owner_rows,float min_probability,
    MtpWindowCostEstimate::Model model,std::uint64_t reprobe_cycles,bool timing_independent)
    :serial_(model),model_(model),min_probability_(min_probability),timing_independent_(timing_independent),
     reprobe_cycles_(reprobe_cycles) {
    if(owner_rows.empty() || !std::is_sorted(owner_rows.begin(),owner_rows.end()) || owner_rows.front()<2U ||
        std::adjacent_find(owner_rows.begin(),owner_rows.end())!=owner_rows.end() ||
        !(min_probability>=0.0F && min_probability<=1.0F))
        throw std::invalid_argument("MTP window controller requires distinct ascending owners and a probability in [0, 1]");
    for(const auto rows:owner_rows) {Owner o;o.rows=rows;o.cost=MtpWindowCostEstimate(model);o.interval=reprobe_cycles;owners_.push_back(o);}
}
namespace {
std::size_t bin(float p) {return std::min<std::size_t>(9U,static_cast<std::size_t>(std::max(p,0.0F)*10.0F));}
}
void MtpMinPWindowController::begin_session() {
    std::vector<std::uint32_t> rows;
    for(const auto& o:owners_) rows.push_back(o.rows);
    *this=MtpMinPWindowController(std::move(rows),min_probability_,model_,reprobe_cycles_,timing_independent_);
}
void MtpMinPWindowController::seed_costs(std::span<const CostPrior> priors) {
    const auto set=[](const MtpWindowCostEstimate& c,double& prior,double ms){
        if(!c.samples() && std::isfinite(ms) && ms>0.0) prior=ms;};
    for(const auto& p:priors) {
        if(p.rows==1U) {set(serial_,serial_prior_,p.ms);continue;}
        for(auto& o:owners_) if(o.rows==p.rows) set(o.cost,o.prior,p.ms);
    }
    prior_age_=0;
}
std::size_t MtpMinPWindowController::active_priors() const noexcept {
    std::size_t n=!serial_.samples() && serial_prior_>0.0 ? 1U : 0U;
    for(const auto& o:owners_) if(!o.cost.samples() && o.prior>0.0) ++n;
    return n;
}
std::vector<MtpMinPWindowController::CostPrior> MtpMinPWindowController::measured_costs(std::uint64_t min_samples) const {
    std::vector<CostPrior> out;
    const auto take=[&](std::uint32_t rows,const MtpWindowCostEstimate& c){
        if(c.samples() && c.samples()>=min_samples) out.push_back({rows,c.value(),c.samples()});};
    take(1U,serial_);
    for(const auto& o:owners_) take(o.rows,o.cost);
    return out;
}
void MtpMinPWindowController::reset_costs() {
    serial_=MtpWindowCostEstimate(model_);serial_prior_=0.0;
    for(auto& o:owners_) {o.cost=MtpWindowCostEstimate(model_);o.prior=0.0;}
}
bool MtpMinPWindowController::fresh() const noexcept {
    if(choices_ || serial_.samples()) return false;
    for(const auto& o:owners_) if(o.cost.samples() || o.warmed) return false;
    for(const auto t:tested_) if(t!=0.0) return false;
    return true;
}
// Calibrated acceptance: the bin's observed rate with a two-draft prior at p.
double MtpMinPWindowController::acceptance(float p) const {const auto b=bin(p);return (accepted_[b]+2.0*p)/(tested_[b]+2.0);}
void MtpMinPWindowController::calibrate(std::span<const float> p,std::size_t accepted_drafts,std::size_t verified_rows) {
    for(std::size_t j=0;j<p.size() && j<external_first_ && j<verified_rows && j<=accepted_drafts;++j){
        const auto b=bin(p[j]);tested_[b]+=1.0;if(j<accepted_drafts)accepted_[b]+=1.0;}
}
void MtpMinPWindowController::set_external_rows(std::size_t first,std::span<const double> acceptance) {
    external_first_=first;external_.assign(acceptance.begin(),acceptance.end());
}
// Expected emitted tokens of an R-row prefix window when draft j is accepted
// with its calibrated probability, given that every earlier draft was accepted.
double MtpMinPWindowController::expected_tokens(std::uint32_t rows,std::span<const float> p) const {
    double expected=1.0,run=1.0;
    for(std::uint32_t j=1;j<rows && j<=p.size();++j){
        const std::size_t r=j-1U;
        run*=r>=external_first_ && r-external_first_<external_.size()?external_[r-external_first_]:acceptance(p[r]);
        expected+=run;}
    return expected;
}
bool MtpMinPWindowController::worth_drafting(std::span<const float> p) const {
    if(timing_independent_ || min_probability_<=0.0F || p.empty()) return true;
    const auto usable=p.size()+1U;
    double best=known(serial_,serial_prior_)?1.0/cost(serial_,serial_prior_):0.0;
    const Owner* next=nullptr;
    for(const auto& o:owners_) {
        if(o.rows<=usable) {if(known(o.cost,o.prior)) best=std::max(best,expected_tokens(o.rows,p)/cost(o.cost,o.prior));}
        else if(!next) next=&o;
    }
    if(!next) return false;
    if(!known(next->cost,next->prior)) return true;
    std::vector<float> assumed(p.begin(),p.end());assumed.resize(next->rows-1U,p.back());
    return expected_tokens(next->rows,assumed)/cost(next->cost,next->prior)>best;
}
bool MtpMinPWindowController::implausible(const Owner& o) const {
    if(!o.cost.samples() || o.cost.provisional()) return true;
    return std::any_of(owners_.begin(),owners_.end(),[&](const Owner& m){
        return m.rows>o.rows && m.cost.samples() && !m.cost.provisional() && m.cost.value()<o.cost.value();});
}
std::size_t MtpMinPWindowController::choose(std::span<const float> p,std::size_t candidates,std::span<const std::uint8_t> fits) {
    if(fits.size()!=owners_.size()) throw std::invalid_argument("MTP window controller fits mask differs from its owners");
    const auto n=owners_.size();
    const auto largest=[&]()->std::size_t{for(std::size_t o=n;o-->0;) if(fits[o]) return o;return serial;};
    if(min_probability_<=0.0F || p.size()!=candidates) return largest();
    if(timing_independent_) {
        // The smallest owner covering the confident drafts, else the largest that fits.
        std::uint32_t want=1;
        while(want-1U<p.size() && p[want-1U]>=min_probability_) ++want;
        if(want==1U) return serial;
        for(std::size_t o=0;o<n;++o) if(owners_[o].rows>=want && fits[o]) return o;
        return largest();
    }
    ++choices_;
    const auto use=[&](std::size_t o){if(o!=serial) owners_[o].last_used=choices_;return o;};
    if(model_==MtpWindowCostEstimate::Model::robust && reprobe_cycles_) {
        for(std::size_t o=n;o-->0;) {
            auto& owner=owners_[o];
            if(fits[o] && choices_-owner.last_used>owner.interval && implausible(owner)) {
                owner.interval*=2U;++owner.probes;return use(o);
            }
        }
    }
    std::uint32_t want=1;
    while(want-1U<p.size() && p[want-1U]>=min_probability_) ++want;
    std::size_t cap=n-1U;
    for(std::size_t o=0;o<n;++o) if(owners_[o].rows>=want){cap=o;break;}
    const bool serial_only=want==1U;
    // Unmeasured options (without a prior) are tried once before the cost model ranks them.
    if(fits[cap] && !known(owners_[cap].cost,owners_[cap].prior)) return use(cap);
    if(serial_only && !known(serial_,serial_prior_)) return serial;
    std::size_t best=serial;double best_rate=0.0;
    for(std::size_t o=0;o<n;++o) {
        const auto& owner=owners_[o];
        if(owner.rows>owners_[cap].rows || !fits[o] || !known(owner.cost,owner.prior)) continue;
        const auto rate=expected_tokens(owner.rows,p)/cost(owner.cost,owner.prior);
        if(best==serial || rate>best_rate){best=o;best_rate=rate;}
    }
    if(serial_only && (best==serial || 1.0/cost(serial_,serial_prior_)>best_rate)) return serial;
    return use(best!=serial?best:largest());
}
void MtpMinPWindowController::observe(std::size_t owner,std::size_t emitted,double ms) {
    if(++prior_age_>=prior_cycles) { // bootstrap only: every prior still in use expires
        serial_prior_=0.0;
        for(auto& o:owners_) o.prior=0.0;
    }
    if(!emitted) return;
    if(owner==serial) {serial_.add(ms/static_cast<double>(emitted));return;}
    auto& o=owners_.at(owner);
    if(o.warmed) o.cost.add(ms);
    o.warmed=true;
}
std::vector<MtpMinPWindowController::OwnerStats> MtpMinPWindowController::owner_stats() const {
    std::vector<OwnerStats> out{{1U,serial_.samples(),cost(serial_,serial_prior_),0U}};
    for(const auto& o:owners_) out.push_back({o.rows,o.cost.samples(),cost(o.cost,o.prior),o.probes});
    return out;
}
} // namespace strix

namespace strix {
namespace {
std::array<std::uint32_t,3> batch_key(std::span<const std::uint32_t> rows) {
    std::array<std::uint32_t,3> key{};
    for(auto r:rows) ++key[r==8U?2U:r==4U?1U:0U];
    return key;
}
}
MtpBatchWindowController::Rule MtpBatchWindowController::rule_from_env(bool baseline) {
    const char* v=std::getenv("STRIX_MTP_JOINT_PLAN");
    if(!v && !baseline && mtp_serve_row_budget_enabled()) return parse_rule("budget");
    return v?parse_rule(v):parse_rule(baseline?"production":"calibrated-c3");
}
std::uint32_t mtp_serve_row_budget_max_lanes() noexcept {
    static const std::uint32_t lanes=[]{
        try {return envcfg::number<std::uint32_t>("STRIX_SERVE_ROW_BUDGET_MAX_LANES",6U,0U,8U);} catch(...) {return 6U;}}();
    return lanes;
}
bool mtp_serve_row_budget_enabled() noexcept {
    static const bool on=envcfg::flag("STRIX_SERVE_ROW_BUDGET",true,EnvFlagMatch::exact);
    return on;
}
MtpBatchWindowController::Rule MtpBatchWindowController::parse_rule(std::string_view text) {
    Rule rule;
    if(text.find("pooled")!=std::string_view::npos) rule.pooled=true;
    if(text.find("calibrated")!=std::string_view::npos) rule.calibrated=true;
    if(text.find("calibrated-c3")!=std::string_view::npos) rule.calibrated_min_members=3;
    if(text.find("cap4")!=std::string_view::npos) rule.cap=4;
    if(text.find("cap2")!=std::string_view::npos) rule.cap=2;
    if(text.find("budget")!=std::string_view::npos) rule.budget=true;
    return rule;
}
std::size_t MtpBatchWindowController::calibration_bin(double p) {
    return std::min<std::size_t>(9U,static_cast<std::size_t>(std::max(p,0.0)*10.0));
}
double MtpBatchWindowController::expected(const Request& q,std::uint32_t rows,bool prior) const {
    const auto& h=history_.at(q.slot);
    double e=1.0,s=1.0,base=0.8;
    if(prior && rule_.pooled) {
        double seen=0.0,hits=0.0;
        for(std::size_t d=0;d<h.seen.size();++d) {seen+=h.seen[d];hits+=h.hits[d];}
        base=(hits+4.0*0.8)/(seen+4.0);
    }
    for(std::uint32_t d=0;d+1U<rows;++d) {
        if(!prior && d>=q.probabilities.size()) break;
        const double p=prior?base:std::clamp(static_cast<double>(q.probabilities[d]),0.0,1.0);
        if(!prior && rule_.calibrated && plan_members_>=rule_.calibrated_min_members && d<q.external_first) {
            const auto b=calibration_bin(p);
            const double global=(accepted_[b]+2.0*p)/(tested_[b]+2.0);
            s*=(h.accepted[b]+4.0*global)/(h.tested[b]+4.0);
        } else s*=(h.hits[d]+4.0*p)/(h.seen[d]+4.0);
        e+=s;
    }
    return e;
}
double MtpBatchWindowController::cost(std::span<const std::uint32_t> rows) const {
    const auto key=batch_key(rows);
    const auto found=costs_.find(key);
    if(found!=costs_.end() && found->second.samples()) return found->second.value();
    std::uint32_t total=0;for(auto r:rows) total+=r?r:2U;
    // Byte-shaped prior: shared weights plus incremental union/KV rows. Once a
    // topology is measured, extrapolate from the closest same-concurrency one.
    double estimate=28.0+2.0*total+3.0*static_cast<double>(rows.size());
    std::uint32_t distance=UINT32_MAX;
    for(const auto& [k,v]:costs_) {
        if(!v.samples() || k[0]+k[1]+k[2]!=rows.size()) continue;
        const auto t=2U*k[0]+4U*k[1]+8U*k[2];
        const auto delta=total>t?total-t:t-total;
        if(delta<distance) {distance=delta;estimate=v.value()*(28.0+2.0*total)/(28.0+2.0*t);}
    }
    return estimate;
}
std::vector<std::uint32_t> MtpBatchWindowController::search(std::span<const Request> requests,std::uint32_t budget,bool prior,
    PlanInfo* info) const {
    if(requests.empty() || requests.size()>8U || budget<2U*requests.size())
        throw std::invalid_argument("joint MTP windows require 1..8 requests and two physical rows each");
    for(const auto& q:requests) if(q.limit!=2U && q.limit!=4U && q.limit!=8U)
        throw std::invalid_argument("joint MTP window limit must be an admitted owner");
    std::vector<std::uint32_t> current(requests.size(),0U),best;
    double best_rate=-1.0,best_e=0.0,best_ms=0.0;
    const auto visit=[&](auto&& self,std::size_t at,std::uint32_t used,double e,std::uint32_t drafts)->void {
        if(at==requests.size()) {
            // Small primaries have exactly four or eight rows. A six-row plan
            // would split into two serial chunks and reread every weight.
            if(used<9U && used!=4U && used!=8U) return;
            const auto ms=cost(current)+(prior?(draft_.samples()?draft_.value():1.2)*drafts:0.0);
            const auto rate=e/ms;
            if(rate>best_rate) {best_rate=rate;best=current;best_e=e;best_ms=ms;}
            return;
        }
        const auto& q=requests[at];
        for(const auto r:{0U,2U,4U,8U}) {
            if(!prior && r==0U) continue;
            const auto physical=r?r:2U;
            if(physical>q.limit || used+physical+2U*(requests.size()-at-1U)>budget) continue;
            if(prior && r>rule_.cap) continue;
            current[at]=r;
            self(self,at+1U,used+physical,e+(r?expected(q,r,prior):1.0),drafts+(r?r-1U:0U));
        }
    };
    visit(visit,0U,0U,0.0,0U);
    if(info) *info={best_e,best_ms,false};
    // Three context/budget tails can each fit only R2. They cannot be padded to
    // eight rows: retain their admissible small chunks rather than fail a request.
    if(best.empty()) {
        best.assign(requests.size(),prior?0U:2U);
        if(info) *info=estimate(requests,best,prior);
    }
    return best;
}
MtpBatchWindowController::BudgetCost MtpBatchWindowController::budget_cost() const {
    static const BudgetCost pinned=[]{
        BudgetCost c;const char* v=envcfg::text("STRIX_SERVE_ROW_COST");
        if(v) {char* end=nullptr;const double f=std::strtod(v,&end);
            if(end && *end==',') {char* next=nullptr;const double r=std::strtod(end+1,&next);
                const double l=next && *next==','?std::strtod(next+1,nullptr):0.0;
                if(f>0.0 && r>0.0 && l>=0.0 && std::isfinite(f) && std::isfinite(r) && std::isfinite(l)) {
                    c.fixed_ms=f;c.row_ms=r;c.lane_ms=l;c.fitted=true;}}}
        return c;}();
    if(pinned.fitted) return {pinned.fixed_ms,pinned.row_ms,pinned.lane_ms,false};
    // Least squares of the measured topologies' cycle ms on (1, physical rows, lanes).
    double a[3][4]{};
    for(const auto& [k,v]:costs_) {
        if(!v.samples()) continue;
        const double x[3]{1.0,2.0*k[0]+4.0*k[1]+8.0*k[2],static_cast<double>(k[0]+k[1]+k[2])},y=v.value();
        for(int i=0;i<3;++i) {for(int j=0;j<3;++j) a[i][j]+=x[i]*x[j];a[i][3]+=x[i]*y;}
    }
    // Gaussian elimination with partial pivoting; a pivot that vanishes relative to its column's scale
    // means rows and lanes did not move independently, and the priors stay.
    for(int i=0;i<3;++i) {
        int pivot=i;
        for(int r=i+1;r<3;++r) if(std::abs(a[r][i])>std::abs(a[pivot][i])) pivot=r;
        const double scale=std::max({std::abs(a[0][i]),std::abs(a[1][i]),std::abs(a[2][i]),1.0});
        if(!(std::abs(a[pivot][i])>1e-9*scale)) return {};
        for(int j=0;j<4;++j) std::swap(a[i][j],a[pivot][j]);
        for(int r=0;r<3;++r) if(r!=i) {
            const double f=a[r][i]/a[i][i];
            for(int j=i;j<4;++j) a[r][j]-=f*a[i][j];
        }
    }
    const double fixed=a[0][3]/a[0][0],row=a[1][3]/a[1][1],lane=a[2][3]/a[2][2];
    if(!(row>0.0) || !(fixed>0.0) || !(lane>=0.0) || !std::isfinite(row) || !std::isfinite(fixed) || !std::isfinite(lane)) return {};
    return {fixed,row,lane,true};
}
std::vector<std::uint32_t> MtpBatchWindowController::budget_rows(std::span<const Request> q,std::uint32_t budget,bool prior,
    PlanInfo* info) const {
    if(q.empty() || q.size()>8U || budget<2U*q.size())
        throw std::invalid_argument("joint MTP windows require 1..8 requests and two physical rows each");
    for(const auto& r:q) if(r.limit!=2U && r.limit!=4U && r.limit!=8U)
        throw std::invalid_argument("joint MTP window limit must be an admitted owner");
    // Regime selection: above the lane bound the measured cycle is faster with the fixed table's two rows per
    // request (the cost of a deeper row exceeds its yield there), so the budget plans only up to the bound.
    if(q.size()>mtp_serve_row_budget_max_lanes()) {
        std::vector<std::uint32_t> fixed(q.size(),2U);
        if(info) *info=estimate(q,fixed,prior);
        return fixed;
    }
    // accept[0] is the committed target row; row d + 2 is useful when draft d is accepted. Before drafting the
    // draft's chance is the 0.8 prior, afterwards its own probability, both blended with the slot's history
    // (the production rule's blend).
    std::vector<std::array<float,8>> accept(q.size());
    std::vector<serve::LaneDemand> lanes(q.size());
    for(std::size_t k=0;k<q.size();++k) {
        const auto& h=history_.at(q[k].slot);
        std::uint32_t most=prior?std::min(q[k].limit,std::max(rule_.cap,2U)):q[k].limit;
        accept[k][0]=1.0F;
        for(std::size_t d=0;d<7U;++d) {
            const double p=prior?0.8:d<q[k].probabilities.size()?std::clamp(static_cast<double>(q[k].probabilities[d]),0.0,1.0):0.0;
            accept[k][d+1U]=static_cast<float>((h.hits[d]+4.0*p)/(h.seen[d]+4.0));
        }
        lanes[k]={q[k].slot,most,2U,most,std::span<const float>(accept[k]),0.0};
    }
    const auto model=budget_cost();
    const double draft_ms=prior?(draft_.samples()?draft_.value():1.2):0.0;
    const serve::CycleCost cost{model.fixed_ms,model.row_ms,model.lane_ms,draft_ms};
    const serve::RowLimits limits{budget,{},static_cast<std::uint32_t>(q.size()),0.0};
    const auto plan=serve::plan_rows(lanes,cost,limits);
    // Owners hold 2, 4 or 8 rows: each lane keeps the largest owner inside its planned rows.
    std::vector<std::uint32_t> out(q.size(),2U);
    std::uint32_t used=0;
    for(std::size_t k=0;k<q.size();++k) {
        const auto r=k<plan.rows.size()?plan.rows[k]:2U;
        out[k]=r>=8U?8U:r>=4U?4U:2U;
        used+=out[k];
    }
    // Small primaries have exactly four or eight rows (see search): pad the most promising two-row lane, or
    // keep the two-row chunks when no lane can take four.
    if(used<9U && used!=4U && used!=8U) {
        std::size_t best=q.size();
        for(std::size_t k=0;k<q.size();++k) if(out[k]==2U && q[k].limit>=4U && (!prior || rule_.cap>=4U) && used+2U<=budget &&
            (best==q.size() || accept[k][1]>accept[best][1])) best=k;
        if(best!=q.size() && (used+2U==4U || used+2U==8U)) {out[best]=4U;used+=2U;}
        else {std::fill(out.begin(),out.end(),2U);used=2U*static_cast<std::uint32_t>(q.size());}
    }
    // Rows the owner sizes left unused: a lane moves to its next owner (2 -> 4 -> 8) while the budget holds it
    // and the added rows yield more tokens per millisecond than the cycle does; most promising lane first.
    const auto value=[&](std::size_t k,std::uint32_t from,std::uint32_t to) {
        double chance=1.0,tokens=0.0;
        for(std::uint32_t r=0;r<to;++r) {chance*=static_cast<double>(accept[k][r]);if(r>=from) tokens+=chance;}
        return tokens;
    };
    double tokens=0.0;
    for(std::size_t k=0;k<q.size();++k) tokens+=value(k,0U,out[k]);
    while(used>=8U) {
        const double ms=model.fixed_ms+model.row_ms*used+model.lane_ms*static_cast<double>(q.size())+draft_ms*(used-static_cast<double>(q.size()));
        std::size_t best=q.size();double best_gain=0.0;
        for(std::size_t k=0;k<q.size();++k) {
            const auto next=out[k]*2U;
            if(next>lanes[k].max_rows || used+out[k]>budget) continue;
            const double gain=value(k,out[k],next)/((model.row_ms+draft_ms)*out[k]);
            if(gain>tokens/ms && gain>best_gain) {best=k;best_gain=gain;}
        }
        if(best==q.size()) break;
        tokens+=value(best,out[best],out[best]*2U);used+=out[best];out[best]*=2U;
    }
    if(prior) {
        // Two physical rows are the minimum either way; the one draft that fills the second row is bought only
        // when its chance beats its cost at the plan's rate.
        const double rate=plan.expected_ms>0.0?plan.expected_tokens/plan.expected_ms:0.0;
        for(std::size_t k=0;k<q.size();++k) if(out[k]==2U && static_cast<double>(accept[k][1])<=draft_ms*rate) out[k]=0U;
    }
    if(info) *info=estimate(q,out,prior);
    return out;
}
MtpBatchWindowController::PlanInfo MtpBatchWindowController::estimate(std::span<const Request> q,
    std::span<const std::uint32_t> rows,bool prior) const {
    double e=0.0;std::uint32_t drafts=0;
    for(std::size_t k=0;k<q.size();++k) {e+=rows[k]?expected(q[k],rows[k],prior):1.0;drafts+=rows[k]?rows[k]-1U:0U;}
    return {e,cost(rows)+(prior?(draft_.samples()?draft_.value():1.2)*drafts:0.0),false};
}
std::vector<std::uint32_t> MtpBatchWindowController::plan(std::span<const Request> q,std::uint32_t budget) {
    plan_members_=q.size();
    auto out=rule_.budget?budget_rows(q,budget,true,&last_):search(q,budget,true,&last_);
    probe_.clear();
    if(rule_.budget && q.size()>mtp_serve_row_budget_max_lanes()) return out;
    auto& plans=plans_.at(q.size());
    // Bounded exploration samples deeper windows even after a low-acceptance
    // region. It does not borrow another request's acceptance history.
    if(++plans<=9U || plans%32U==1U) {
        const auto depth=plans<=9U?(2U<<((plans-1U)%3U)):plans%96U<33U?8U:plans%96U<65U?4U:2U;
        std::uint32_t used=2U*static_cast<std::uint32_t>(q.size());
        std::fill(out.begin(),out.end(),2U);
        for(std::size_t k=0;k<q.size();++k) {
            auto r=std::min(depth,rule_.cap);while(r>q[k].limit || used+r-2U>budget) r/=2U;
            if(r>=2U) {out[k]=r;used+=r-2U;}
        }
        // Avoid a six-row chunk even during exploration (e.g. a context tail).
        if(used==6U) {
            bool padded=false;
            for(std::size_t k=0;k<q.size();++k) if(out[k]==2U && q[k].limit>=4U && rule_.cap>=4U && budget>=8U) {
                out[k]=4U;padded=true;break;
            }
            if(!padded) std::fill(out.begin(),out.end(),2U);
        }
        probe_=out;
        last_=estimate(q,out,true);
        last_.probe=true;
    }
    return out;
}
std::vector<std::uint32_t> MtpBatchWindowController::choose(std::span<const Request> q,std::uint32_t budget) const {
    if(!probe_.empty()) return probe_;
    return rule_.budget?budget_rows(q,budget,false):search(q,budget,false);
}
void MtpBatchWindowController::observe(std::span<const Request> q,std::span<const std::uint32_t> rows,
    std::span<const std::uint32_t> accepted,double draft_ms,double target_ms,std::uint32_t steps) {
    if(q.size()!=rows.size() || rows.size()!=accepted.size() || !(target_ms>0.0) || draft_ms<0.0)
        throw std::invalid_argument("joint MTP observation geometry or timing differs");
    costs_[batch_key(rows)].add(target_ms);
    if(steps && draft_ms>0.0) draft_.add(draft_ms/steps);
    for(std::size_t k=0;k<q.size();++k) {
        auto& h=history_.at(q[k].slot);
        for(std::size_t d=0;d<q[k].probabilities.size() && d+1U<rows[k] && d<=accepted[k];++d) {
            h.seen[d]+=1.0;if(d<accepted[k]) h.hits[d]+=1.0;
            // Calibration bins (read only by the calibrated rule): every verified predictor draft that was reached.
            const auto b=calibration_bin(std::clamp(static_cast<double>(q[k].probabilities[d]),0.0,1.0));
            h.tested[b]+=1.0;tested_[b]+=1.0;
            if(d<accepted[k]) {h.accepted[b]+=1.0;accepted_[b]+=1.0;}
        }
    }
}
} // namespace strix
