#include <strix/weight/types.hpp>
#include <strix/weight/source.hpp>
#include <set>
#include <string>
bool wfmt_contracts_test() {
    using namespace strix::weight;
    if (!Status{Error::ok, ""} || Status{Error::malformed, "bad"}) return false;
    std::string owned = "same";
    if (!(Id{owned, 2} == Id{"same", 2}) || Id{owned, 2} == Id{"same", 1}) return false;
    if (serialize_id({owned, 2}) != "same@2") return false;
    std::set<std::string> ids;
    for (const auto& c : legacy_arithmetic()) {
        if (c.id.name.empty() || c.id.revision != 1 || !ids.insert(serialize_id(c.id)).second ||
            c.activation.empty() || c.weight_rounding.empty() || c.reduction.empty()) return false;
    }
    return ids.contains("cpu-q8a8-tree32@1") && ids.contains("intel-q8a8-f16-ascending@1") &&
        ids.contains("dense-q8a8-f16-v1@1") && ids.contains("trellis-quality@1");
}
