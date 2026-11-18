#include "strix/mtp_depth_window_controller.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix {
namespace {
void require(bool ok,const char* message) {if(!ok)throw std::invalid_argument(message);}
std::size_t bin(float p) {return std::min<std::size_t>(9U,static_cast<std::size_t>(p*10));}
const char* domain_suffix() {
    const std::string_view d=mtp_depth_domain();
    if(d=="code")return "CODE";
    if(d=="prose")return "PROSE";
    if(d=="copy")return "COPY";
    return "GENERAL";
}
}
MtpDepthWindowController::MtpDepthWindowController(std::vector<std::uint32_t> rows,float minp,Prior prior)
    :prior_(prior),min_probability_(minp) {
    require(!rows.empty() && rows.front()>=2 && rows.back()<=8 && std::is_sorted(rows.begin(),rows.end()) &&
        std::adjacent_find(rows.begin(),rows.end())==rows.end(),"depth controller needs ascending distinct owners in [2,8]");
    require(std::isfinite(minp) && minp>=0 && minp<=1,"depth min-p must be in [0,1]");
    for(double p:prior_)require(std::isfinite(p) && p>=0 && p<=1,"depth prior must contain seven probabilities");
    for(auto r:rows) {Owner o;o.rows=r;o.interval.fill(64);owners_.push_back(o);}
}
void MtpDepthWindowController::validate(std::span<const float> p) {
    require(p.size()<=7,"depth controller supports at most seven proposals");
    for(float x:p)require(std::isfinite(x) && x>=0 && x<=1,"depth confidence must be a probability");
}
void MtpDepthWindowController::note_draft(std::uint32_t steps,double ms) {
    require(steps<=7 && std::isfinite(ms) && ms>=0,"invalid depth draft timing sample");
    drafted_=steps;current_draft_ms_=steps?ms:0;noted_=true;
}
void MtpDepthWindowController::set_external_rows(std::size_t first,std::span<const double> p) {
    require(first<=7 && p.size()<=7-first,"external rows exceed the shared draft window");
    for(double x:p)require(std::isfinite(x) && x>=0 && x<=1,"external acceptance must be a probability");
    external_first_=first;external_.assign(p.begin(),p.end());
}
double MtpDepthWindowController::acceptance(std::size_t k,float p) const {
    const auto b=bin(p);return (accepted_[k][b]+2*p)/(tested_[k][b]+2);
}
double MtpDepthWindowController::expected(std::uint32_t rows,std::span<const float> p,bool lookahead) const {
    double sum=1,run=1;
    for(std::size_t k=0;k+1<rows && k<7;++k) {
        double a=0;
        if(k<external_first_ && k<p.size())a=acceptance(k,p[k]);
        else if(k>=external_first_ && k-external_first_<external_.size())a=external_[k-external_first_];
        else if(lookahead)a=prior_[k];
        else break;
        run*=a;sum+=run;
    }
    return sum;
}
double MtpDepthWindowController::cost(const Owner& o,std::size_t filled) const {
    const auto& c=o.residual[std::min<std::size_t>(filled,o.rows-1U)];
    return c.samples()?c.value():0;
}
bool MtpDepthWindowController::uncertain(const Owner& o,std::size_t filled) const {
    const auto& c=o.residual[std::min<std::size_t>(filled,o.rows-1U)];
    if(c.provisional())return true;
    return std::any_of(owners_.begin(),owners_.end(),[&](const Owner& bigger) {
        const auto& other=bigger.residual[std::min<std::size_t>(filled,bigger.rows-1U)];
        return bigger.rows>o.rows && !other.provisional() && other.value()<c.value();
    });
}
bool MtpDepthWindowController::worth_drafting(std::span<const float> p) const {
    validate(p);if(p.empty() || min_probability_<=0)return true;
    // A periodic eligible full chain prevents an early stop from starving a
    // provisional future cell. The ordinary min-p and depth budget still apply.
    if((choices_+1U)%64U==0U && std::any_of(owners_.begin(),owners_.end(),[&](const Owner& o) {
        return o.rows>p.size()+1 && uncertain(o,o.rows-1U);
    }))return true;
    if(!draft_[p.size()].samples())return true;
    const double now_draft=draft_[p.size()].value();
    double best=0;
    const auto serial_tokens=expected(static_cast<std::uint32_t>(p.size()+1),p);
    if(serial_.samples())best=serial_tokens/(serial_.value()*serial_tokens+now_draft);
    for(const auto& o:owners_)if(o.rows<=p.size()+1 && cost(o,p.size())>0)
        best=std::max(best,expected(o.rows,p)/(cost(o,p.size())+now_draft));
    for(const auto& o:owners_)if(o.rows>p.size()+1) {
        const auto depth=o.rows-1U;
        const auto residual=cost(o,depth);
        // A missing depth/owner is measured; it is never inferred from an easier owner.
        if(!static_cast<bool>(residual) || !draft_[depth].samples())return true;
        if(expected(o.rows,p,true)/(residual+draft_[depth].value())>best)return true;
    }
    return false;
}
std::size_t MtpDepthWindowController::choose(std::span<const float> p,std::size_t candidates,
    std::span<const std::uint8_t> fits) {
    validate(p);require(fits.size()==owners_.size(),"depth fits mask differs from owners");
    filled_=candidates;
    ++choices_;
    const auto use=[&](std::size_t n) {
        if(n!=serial)owners_[n].last_used[std::min<std::size_t>(candidates,owners_[n].rows-1U)]=choices_;
        return n;
    };
    const auto largest=[&] {for(std::size_t n=owners_.size();n-->0;)if(fits[n])return n;return serial;};
    if(min_probability_<=0 || p.size()!=candidates)return use(largest());
    std::size_t want=1;
    while(want-1<p.size()) {
        const auto k=want-1;
        if(k<external_first_ && p[k]<min_probability_)break;
        if(k>=external_first_ && k-external_first_>=external_.size())break;
        ++want;
    }
    std::size_t cap=owners_.size()-1;
    for(std::size_t n=0;n<owners_.size();++n)if(owners_[n].rows>=want){cap=n;break;}
    for(std::size_t n=cap+1;n-->0;)if(fits[n]) {
        auto& o=owners_[n];const auto k=std::min<std::size_t>(candidates,o.rows-1U);
        if(o.residual[k].samples() && choices_-o.last_used[k]>=o.interval[k] && uncertain(o,candidates)) {
            o.interval[k]*=2U;++o.probes[k];return use(n);
        }
    }
    // Cold exploration is local to the actual number of filled rows.
    if(fits[cap] && !static_cast<bool>(cost(owners_[cap],candidates)))return use(cap);
    for(std::size_t n=0;n<=cap;++n)if(fits[n] && !static_cast<bool>(cost(owners_[n],candidates)))return use(n);
    if(want==1 && !serial_.samples())return serial;
    const auto d=noted_?current_draft_ms_:(draft_[p.size()].samples()?draft_[p.size()].value():0);
    auto best=serial;double rate=0;
    if(serial_.samples()) {
        const auto e=expected(static_cast<std::uint32_t>(p.size()+1),p);
        rate=e/(serial_.value()*e+d);
    }
    for(std::size_t n=0;n<=cap;++n)if(fits[n] && cost(owners_[n],candidates)>0) {
        const auto r=expected(owners_[n].rows,p)/(cost(owners_[n],candidates)+d);
        if(r>rate){rate=r;best=n;}
    }
    return use(best);
}
void MtpDepthWindowController::calibrate(std::span<const float> p,std::size_t accepted,std::size_t verified) {
    validate(p);
    for(std::size_t k=0;k<p.size() && k<external_first_ && k<verified && k<=accepted;++k) {
        const auto b=bin(p[k]);++tested_[k][b];
        if(k<accepted)++accepted_[k][b];
    }
}
void MtpDepthWindowController::observe(std::size_t owner,std::size_t emitted,double ms) {
    if(!emitted || !std::isfinite(ms) || ms<=0){noted_=false;return;}
    const auto d=noted_?current_draft_ms_:0;
    if(noted_ && drafted_ && d>0)draft_[drafted_].add(d);
    const auto residual=ms-d;noted_=false;
    if(residual<=0)return;
    if(owner==serial){serial_.add(residual/static_cast<double>(emitted));return;}
    auto& o=owners_.at(owner);
    if(o.warmed)o.residual[std::min<std::size_t>(filled_,o.rows-1U)].add(residual);
    o.warmed=true;
}
std::uint32_t MtpDepthWindowController::plan_depth(std::uint32_t max_depth) {
    max_depth=std::min<std::uint32_t>(max_depth,7U);
    if(max_depth<=1)return max_depth;
    ++plans_;
    if(plans_%64U==0U)return max_depth;
    bool draft_known=false;
    for(std::size_t n=1;n<draft_.size();++n)draft_known=draft_known || draft_[n].samples();
    double a[7];
    for(std::size_t k=0;k<7;++k) {
        double t=0,acc=0;
        for(std::size_t b=0;b<10;++b){t+=tested_[k][b];acc+=accepted_[k][b];}
        a[k]=t>=8?acc/t:prior_[k];
    }
    std::uint32_t best_d=max_depth,missing=0;double best=0;
    for(std::uint32_t d=max_depth;d>=1;--d) {
        const Owner* o=nullptr;for(const auto& x:owners_)if(x.rows==d+1U)o=&x;
        if(!o)continue;
        const auto& cell=o->residual[d];
        if(!cell.samples() || (draft_known && !draft_[d].samples())) {
            if(!missing && explored_[d]<4U)missing=d;
            continue;
        }
        double sum=1,run=1;for(std::uint32_t k=0;k<d;++k){run*=a[k];sum+=run;}
        const double ms=cell.value()+(draft_known?draft_[d].value():0.0);
        if(ms>0 && sum/ms>best){best=sum/ms;best_d=d;}
    }
    if(missing){++explored_[missing];return missing;}
    return best_d;
}
std::vector<MtpDepthWindowController::CostStats> MtpDepthWindowController::cost_stats() const {
    std::vector<CostStats> out{{1,1,serial_.samples(),serial_.value()}};
    for(const auto& o:owners_)for(std::uint32_t n=1;n<o.rows;++n)
        if(o.residual[n].samples())out.push_back({o.rows,n,o.residual[n].samples(),o.residual[n].value(),o.probes[n]});
    return out;
}
std::vector<MtpDepthWindowController::CostStats> MtpDepthWindowController::draft_stats() const {
    std::vector<CostStats> out;
    for(std::uint32_t n=1;n<=7;++n)if(draft_[n].samples())out.push_back({0,n,draft_[n].samples(),draft_[n].value()});
    return out;
}
bool mtp_depth_controller_enabled() noexcept {
    const char* v=std::getenv("STRIX_MTP_CONTROLLER");return v && std::string_view(v)=="depth";
}
const char* mtp_depth_domain() {
    const char* v=std::getenv("STRIX_MTP_DOMAIN");if(!v || !*v)return "general";
    const std::string_view d=v;
    require(d=="general" || d=="code" || d=="prose" || d=="copy","MTP domain must be general, code, prose or copy");
    return v;
}
float mtp_depth_domain_min_probability(float fallback) {
    if(!mtp_depth_controller_enabled())return fallback;
    const auto key=std::string("STRIX_MTP_MINP_")+domain_suffix();const char* v=std::getenv(key.c_str());
    if(!v || !*v)return fallback;
    char* end=nullptr;const float p=std::strtof(v,&end);
    require(end!=v && !*end && std::isfinite(p) && p>=0 && p<=1,"domain min-p must be a probability");
    return p;
}
MtpDepthWindowController::Prior mtp_depth_domain_prior() {
    MtpDepthWindowController::Prior out{.5,.5,.5,.5,.5,.5,.5};
    const auto key=std::string("STRIX_MTP_DEPTH_PRIOR_")+domain_suffix();const char* v=std::getenv(key.c_str());
    if(!v || !*v)return out;
    for(std::size_t k=0;k<out.size();++k) {
        char* end=nullptr;out[k]=std::strtod(v,&end);
        require(end!=v && std::isfinite(out[k]) && out[k]>=0 && out[k]<=1,"invalid domain depth prior");
        require(k+1<out.size()?*end==',':!*end,"domain prior requires seven comma-separated probabilities");
        v=end+1;
    }
    return out;
}
std::string mtp_depth_state_line(float cli_min_p) {
    const char* v=std::getenv("STRIX_MTP_CONTROLLER");
    std::string out="mtp-depth-controller=";
    if(mtp_depth_controller_enabled())
        out+="on (STRIX_MTP_CONTROLLER=depth) domain="+std::string(mtp_depth_domain())+" min_p="+std::to_string(mtp_depth_domain_min_probability(cli_min_p));
    else out+=std::string("off (")+(v && *v?"STRIX_MTP_CONTROLLER="+std::string(v)+" is not 'depth'":"STRIX_MTP_CONTROLLER unset; default off")+") fixed depth, min_p="+std::to_string(cli_min_p);
    return out;
}
}
