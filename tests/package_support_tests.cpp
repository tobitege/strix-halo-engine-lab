// Compare the packaging lists with the explicitly enumerated HIP targets below.
// This does not discover new enum members or validate the installation pages.
// Platform qualification is separately covered by tests/packaging.
#include "strix/hip/arch.hpp"

#include <cstdio>
#include <set>
#include <sstream>
#include <string>

#ifndef STRIX_PKG_HIP_KNOWN
#error "STRIX_PKG_HIP_KNOWN is set by packaging/StrixPackaging.cmake"
#endif

namespace {

std::set<std::string> split(const char* csv) {
    std::set<std::string> out;
    std::stringstream in(csv);
    for (std::string item; std::getline(in, item, ',');)
        if (!item.empty()) out.insert(item);
    return out;
}

}  // namespace

int main() {
    namespace arch = strix::hip::arch;
    const auto known = split(STRIX_PKG_HIP_KNOWN);
    const auto qualified = split(STRIX_PKG_HIP_QUALIFIED);
    int failures = 0;
    std::set<std::string> seen;
    for (const auto t : {arch::Target::Gfx1151, arch::Target::Gfx1200, arch::Target::Gfx1201, arch::Target::Gfx942,
                         arch::Target::Gfx950}) {
        const auto& c = arch::caps(t);
        const std::string name = c.name;
        seen.insert(name);
        if (!known.count(name)) {
            std::fprintf(stderr, "FAIL %s is in arch.hpp but not in STRIX_HIP_KNOWN_ARCHS\n", name.c_str());
            ++failures;
        }
        if (c.qualified != (qualified.count(name) != 0)) {
            std::fprintf(stderr, "FAIL %s: arch.hpp qualified=%d, STRIX_HIP_QUALIFIED_ARCHS says %d\n", name.c_str(),
                         c.qualified ? 1 : 0, qualified.count(name) ? 1 : 0);
            ++failures;
        }
    }
    for (const auto& name : known) {
        if (!seen.count(name)) {
            std::fprintf(stderr, "FAIL %s is in STRIX_HIP_KNOWN_ARCHS but has no arch.hpp Target\n", name.c_str());
            ++failures;
        }
    }
    if (failures == 0) std::printf("package support matrix: %zu HIP targets, %zu qualified\n", seen.size(),
                                   qualified.size());
    return failures == 0 ? 0 : 1;
}
