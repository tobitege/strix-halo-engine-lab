// Host test: the CUDA adapter (MtpDepthHost) and the HIP binding (raw MtpDepthWindowController with the HIP mask)
// pick the same verified rows for the same inputs; the planner finds depth ~4 at p 0.65 and 7 at p 0.9 on measured costs.
#include "strix/mtp_depth_host.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
using namespace strix;
static int fails=0,checks=0;
#define CHECK(c,...) do{++checks;if(!(c)){++fails;std::printf("FAIL %s:%d ",__FILE__,__LINE__);std::printf(__VA_ARGS__);std::printf("\n");}}while(0)
static double verify_ms(unsigned rows){return 14.3+0.6*rows;}      // fixed part + MoE bytes per extra row
static double draft_ms(unsigned steps){return 0.65*steps;}
int main(){
    _putenv("STRIX_MTP_CONTROLLER=depth");_putenv("STRIX_MTP_MINP_GENERAL=0.3");
    CHECK(mtp_depth_controller_enabled(),"controller enabled");
    const auto line=mtp_depth_state_line(0.7F);
    std::printf("%s\n",line.c_str());
    CHECK(line.find("on (STRIX_MTP_CONTROLLER=depth)")!=std::string::npos && line.find("min_p=0.3")!=std::string::npos,"state line");
    // 1. same rows: adapter vs HIP-style raw controller, identical probabilities and timings
    std::mt19937 rng(7);std::uniform_real_distribution<float> U(0,1);
    MtpDepthHost host(7,0.7F,true,0);
    std::vector<std::uint32_t> owners{2,3,4,5,6,7,8};
    MtpDepthWindowController raw(owners,0.3F,mtp_depth_domain_prior());
    int same=0,windows=2000;
    for(int w=0;w<windows;++w){
        const unsigned d=1+rng()%7;std::vector<float> p(d);for(auto& x:p)x=0.2F+0.8F*U(rng);
        const double dms=draft_ms(d);
        host.note_draft(d,dms);raw.note_draft(d,dms);
        const auto rows=host.select(p);
        std::array<std::uint8_t,8> mask{};for(std::size_t n=0;n<owners.size();++n)mask[n]=owners[n]<=d+1?1:0;  // HIP fits()
        const auto c=raw.choose(p,p.size(),std::span<const std::uint8_t>(mask.data(),owners.size()));
        const unsigned rrows=c==MtpDepthWindowController::serial?1U:owners[c];
        if(rows==rrows)++same;
        std::size_t acc=0;while(acc+1<rows && U(rng)<p[acc])++acc;
        const double ms=dms+verify_ms(rows);
        host.finish(rows,acc,ms);raw.calibrate(p,acc,rows-1U);raw.observe(c,acc+1,ms);
    }
    CHECK(same==windows,"rows equal in %d/%d windows",same,windows);
    // 2. planner on measured costs
    for(double pr:{0.65,0.9}){
        MtpDepthHost h(7,0.7F,true,3);
        double ms_total=0,tok=0;unsigned dsum=0,dn=0;
        for(int w=0;w<1500;++w){
            const auto d=h.plan(7);
            std::vector<float> p(d,static_cast<float>(pr));
            h.note_draft(d,draft_ms(d));
            const auto rows=h.select(p);
            std::size_t acc=0;while(acc+1<rows && U(rng)<pr)++acc;
            const double ms=draft_ms(d)+verify_ms(rows);
            h.finish(rows,acc,ms);
            if(w>=1000){ms_total+=ms;tok+=acc+1;dsum+=d;++dn;}
        }
        double fixed=0,ft=0;for(int w=0;w<200000;++w){std::size_t acc=0;while(acc<7 && U(rng)<pr)++acc;fixed+=draft_ms(7)+verify_ms(8);ft+=acc+1;}
        const double mpt=ms_total/tok,fpt=fixed/ft;
        std::printf("p=%.2f mean planned depth %.2f ms/token %.3f fixed-7 %.3f (%.1f%%)\n",pr,double(dsum)/dn,mpt,fpt,100*(mpt/fpt-1));
        if(pr<0.7)CHECK(double(dsum)/dn<=5.0 && mpt<fpt*0.97,"p0.65 depth/ms");
        else CHECK(double(dsum)/dn>=6.0 && mpt<fpt*1.02,"p0.9 depth/ms");
    }
    std::printf("%s checks=%d fails=%d\n",fails?"FAIL":"PASS",checks,fails);
    return fails?1:0;
}
