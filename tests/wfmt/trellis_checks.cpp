#include "trellis_checks.hpp"
#include "../../src/hip/weight/registry.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace {
void expect(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
void finite_oracle() {
    using strix::wfmt::coefficient_matches;
    expect(coefficient_matches(1.0001,1.0,1.0),"finite tolerance rejected");
    expect(coefficient_matches(0.0,0.0,0.0),"zero rejected");
    expect(!coefficient_matches(1.1,1.0,1.0),"finite mismatch accepted");
    for(double bad:{std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}) {
        expect(!coefficient_matches(bad,1.0,1.0),"nonfinite output accepted");
        expect(!coefficient_matches(1.0,bad,1.0),"nonfinite reference accepted");
        expect(!coefficient_matches(1.0,1.0,bad),"nonfinite magnitude accepted");
        expect(!coefficient_matches(bad,bad,bad),"matching nonfinite values accepted");
    }
}
void coverage() {
    using strix::wfmt::TrellisCoverage;
    for(bool glm:{false,true}) for(bool gemma:{false,true})
        for(bool only:{false,true}) for(bool required:{false,true}) {
            const TrellisCoverage c{glm,gemma,only,required};
            const bool valid=(!only || gemma) && (!required || (glm && gemma && !only));
            expect((c.error()==nullptr)==valid,"engine requirement admission");
            expect(c.run_glm()==(glm && !only),"GLM substitute must not execute");
            expect(c.run_gemma()==gemma,"Gemma availability");
        }
}
void lookup() {
    using namespace strix::weight;
    using namespace strix::weight::hip;
    const SourceRegistration rows[]{
        {{"exl3.gemma.transport",1},"trellis-tile","v1",16,2,0,0,true,true,nullptr},
        {{"exl3.mul1.tiles16",1},"trellis-tile","mul1",16,4,0,0,true,true,nullptr},
        {{"exl3.gemma.transport",1},"trellis-tile","v2",16,2,0,0,true,true,nullptr}};
    expect(!find_unique_source(rows,{"exl3.gemma.transport",1},"trellis-tile"),"ambiguous lookup accepted");
    expect(find_unique_source(rows,{"exl3.mul1.tiles16",1},"trellis-tile")==&rows[1],"unique lookup rejected");
    expect(!find_unique_source(rows,{"exl3.mul1.tiles16",1},"other"),"missing lookup accepted");
}
}
int main(int argc,char** argv) try {
    if(argc!=2) throw std::runtime_error("expected oracle, coverage or lookup");
    const std::string_view name=argv[1];
    if(name=="oracle") finite_oracle();
    else if(name=="coverage") coverage();
    else if(name=="lookup") lookup();
    else throw std::runtime_error("unknown test");
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
