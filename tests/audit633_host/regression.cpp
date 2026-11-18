#include "strix/mtp_payload_geometry.hpp"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "rotary.inc"

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
struct Options {
    bool baseline=false, context_explicit=false, rotary_automatic=false;
    std::uint64_t context_tokens=8192, generate_tokens=1;
    std::vector<std::uint32_t> tokens{1};
    strix::RotaryDescriptor rotary=strix::RotaryDescriptor::native();
};
Options admit(Options options, std::uint32_t yarn_factor=0, bool off=false) {
    // Legacy inputs remain available when compiling the original PR's block.
    [[maybe_unused]] const std::optional<std::string> yarn_cli=off ? std::optional<std::string>("off") : std::nullopt;
    [[maybe_unused]] const std::optional<std::string> yarn_env;
#include "admission.inc"
    return options;
}
void refuses(const Options& o, std::string_view recommendation, bool off=false) {
    try {(void)admit(o,0,off);}
    catch (const std::invalid_argument& e) {
        require(std::string_view(e.what()).find(recommendation)!=std::string_view::npos,e.what());
        return;
    }
    throw std::runtime_error("native overflow implicitly selected YaRN");
}
template<class Model>
std::uint64_t budget(const Model& model, std::uint64_t capacity, std::uint64_t requested, bool automatic) {
    // Compile the same tests against the original three-argument helper as well.
    if constexpr (requires {strix::mtp_payload::runtime_budget(model,capacity,requested,automatic);})
        return strix::mtp_payload::runtime_budget(model,capacity,requested,automatic);
    else return strix::mtp_payload::runtime_budget(model,capacity,requested);
}
}
int main(int argc, char** argv) try {
    require(argc==2,"one test case required");
    const std::string_view test=argv[1];
    Options o;
    if (test=="implicit-context") {
        o.tokens.resize(262144);
        refuses(o,"--rope-yarn 2 --context 262145");
        refuses(o,"native rotary context",true);
        o.baseline=true;refuses(o,"native rotary context");
        o.baseline=false;o.tokens.resize(524288);
        refuses(o,"--rope-yarn 4 --context 524289");
    } else if (test=="explicit-context") {
        o.context_explicit=true;
        for (const auto c : {262145U,300000U,524288U,524289U,1048576U}) {
            o.context_tokens=c;
            refuses(o,c<=524288U ? "--rope-yarn 2" : "--rope-yarn 4");
        }
    } else if (test=="explicit-yarn") {
        o.context_explicit=true;
        for (const auto factor : {2U,4U}) {
            o.context_tokens=262144U*factor;
            const auto admitted=admit(o,factor);
            require(admitted.rotary.factor()==factor && admitted.context_tokens==o.context_tokens,
                "explicit opt-in preserves the requested descriptor and capacity");
        }
    } else if (test=="native-boundary") {
        o.tokens.resize(262143);
        const auto native=admit(o);
        require(native.rotary==strix::RotaryDescriptor::native() && !native.context_explicit,
            "native admission and automatic storage selection stay unchanged");
    } else if (test=="runtime-ceiling" || test=="runtime-automatic") {
        const strix::ModelContract model;
        constexpr std::uint64_t requested=134217728, capacity=131200;
        const auto required=strix::mtp_payload::runtime(model,capacity).retained;
        require(required>capacity*2304U && required>requested,"native long-context fixture exceeds 128 MiB");
        if (test=="runtime-ceiling") {
            const auto explicit_cap=budget(model,capacity,requested,false);
            require(explicit_cap==requested && required>explicit_cap,
                "explicit runtime ceiling must remain enforceable by HTTP admission and native owner");
            require(budget(model,capacity,required,false)==required,"exact explicit ceiling admitted");
            require(budget(model,capacity,required+4096,false)==required+4096,"larger explicit ceiling preserved");
        } else {
            require(budget(model,capacity,requested,true)==required,"automatic budget grows to retained payload");
            require(budget(model,8192,requested,true)==requested,"automatic small context retains default budget");
            require(budget(model,0,requested,true)==requested,"zero capacity left for owner refusal");
            require(budget(model,262145,requested,true)==requested,"out-of-range capacity left for owner refusal");
        }
    } else throw std::runtime_error("unknown test case");
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
