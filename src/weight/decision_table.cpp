#include <strix/weight/resolve.hpp>
#include <cmath>
#include <algorithm>
#include <limits>
#include <strix/serve/json.hpp>
#include <charconv>
#include <set>
namespace strix::weight {
Status validate_qualification(const Qualification& q) {
    if (q.receipt == Digest256{} || q.device_class.empty() || q.compiler_id.empty() || q.shape_regime.empty() ||
        q.build_identity == Digest256{} || q.incumbent_implementation == Digest256{} ||
        q.replacement_implementation == Digest256{} || !q.byte_equal || !q.faster || q.arithmetic.name.empty() || !q.arithmetic.revision)
        return {Error::unqualified, "measured decision requires content receipt, compiler, device, regime and byte-equal faster evidence"};
    return {Error::ok, {}};
}
Status estimate_cost(const Candidate& c, const Hardware& h, const Workload& w, double& us) {
    us = std::numeric_limits<double>::quiet_NaN();
    auto rational = [](Rational r) { return r.denominator ? static_cast<double>(r.numerator) / static_cast<double>(r.denominator) : 0.; };
    if (!c.cost.packed_bytes_per_weight.denominator || !c.cost.image_bytes_per_weight.denominator)
        return {Error::unqualified, "missing exact byte geometry"};
    const double weights = static_cast<double>(w.n) * static_cast<double>(w.k);
    const double bytes = weights * (w.warm_image ? rational(c.cost.image_bytes_per_weight) : rational(c.cost.packed_bytes_per_weight)) +
        static_cast<double>(c.cost.fixed_metadata_bytes);
    if (!std::isfinite(h.memory_bytes_per_second) || h.memory_bytes_per_second <= 0)
        return {Error::unqualified, "missing memory throughput calibration"};
    double seconds = bytes / h.memory_bytes_per_second;
    const auto& d = c.cost.decode;
    for (auto op : {d.bit_extracts,d.table_lookups,d.sign_applications,d.packs,d.int_to_float,d.float_multiplies,d.rounds,d.integer_dots})
        if (!op.denominator) return {Error::unqualified, "missing exact decode/product operation counts"};
    const double lookups = rational(d.table_lookups);
    const double dots = rational(d.integer_dots);
    const double valu = rational(d.bit_extracts) + rational(d.sign_applications) + rational(d.packs) +
        rational(d.int_to_float) + rational(d.float_multiplies) + rational(d.rounds);
    if ((lookups && (!std::isfinite(h.lookup_per_second) || h.lookup_per_second <= 0)) ||
        (dots && (!std::isfinite(h.dot4_per_second) || h.dot4_per_second <= 0)) ||
        (valu && (!std::isfinite(h.valu_ops_per_second) || h.valu_ops_per_second <= 0)))
        return {Error::unqualified, "missing decode/product throughput calibration"};
    double compute = (lookups ? lookups / h.lookup_per_second : 0) + (dots ? dots / h.dot4_per_second : 0) +
        (valu ? valu / h.valu_ops_per_second : 0);
    compute *= weights;
    // Integer dots above already describe product work, and must not be counted twice.
    if (!dots) {
        const bool matrix = c.cost.operand == Scalar::f16 || c.cost.operand == Scalar::bf16 || c.cost.operand == Scalar::e4m3fn || c.cost.operand == Scalar::e2m1;
        const double rate = matrix ? h.matrix_ops_per_second : h.valu_ops_per_second;
        if (!std::isfinite(rate) || rate <= 0) return {Error::unqualified, "missing product throughput calibration"};
        compute += 2. * weights * static_cast<double>(w.m) / rate;
    } else if (w.m > 1) compute += weights * dots * static_cast<double>(w.m - 1) / h.dot4_per_second;
    seconds = h.valu_matrix_serialize ? seconds + compute : std::max(seconds, compute);
    us = seconds * 1.e6;
    if (!std::isfinite(us)) return {Error::malformed, "cost overflow"};
    return {Error::ok, {}};
}
Status import_decisions(std::string_view json, DecisionTable& out) {
    if (json.size()>1024*1024) return {Error::malformed,"decision table exceeds bounded metadata limit"};
    try {
        const auto root=strix::serve::parse_json_unique(json);
        const auto* schema=root.find("schema"); const auto* rows=root.find("decisions");
        if (!schema || schema->as_string()!="wfmt.decisions@2" || !rows || !rows->is_array())
            return {Error::malformed,"unknown decision schema"};
        for (const auto& [key,value]:root.as_object()) { (void)value; if (key!="schema" && key!="decisions") return {Error::malformed,"unknown decision table key"}; }
        DecisionTable table; std::set<std::string> keys;
        for (const auto& row:rows->as_array()) {
            auto field=[&](const char* name) -> const strix::serve::Json& { const auto* v=row.find(name); if (!v) throw std::runtime_error(std::string("missing ")+name); return *v; };
            const std::set<std::string> allowed{"receipt","build_identity","device_class","compiler_id","compiler_flags","source_commit","shape_regime","incumbent","replacement","arithmetic","byte_equal","faster","quality_passed"};
            for (const auto& [key,value]:row.as_object()) { (void)value; if (!allowed.contains(key)) throw std::runtime_error("unknown decision semantic key"); }
            Qualification q{};
            auto hex=[](char c) -> int { if(c>='0' && c<='9') return c-'0'; if(c>='a' && c<='f') return c-'a'+10; if(c>='A' && c<='F') return c-'A'+10; return -1; };
            auto digest=[&](const char* name) {
                Digest256 result{}; const auto& value=field(name).as_string();
                if(value.size()!=64) throw std::runtime_error("invalid content digest length");
                for(std::size_t i=0;i<32;++i) { int a=hex(value[2*i]),b=hex(value[2*i+1]); if(a<0 || b<0) throw std::runtime_error("invalid content digest hex"); result.bytes[i]=static_cast<std::uint8_t>(a*16+b); }
                return result;
            };
            q.receipt=digest("receipt"); q.build_identity=digest("build_identity");
            q.incumbent_implementation=digest("incumbent"); q.replacement_implementation=digest("replacement");
            const auto& commit=field("source_commit").as_string();
            if ((commit.size()!=40 && commit.size()!=64) || !std::all_of(commit.begin(),commit.end(),[&](char c){return hex(c)>=0;})) throw std::runtime_error("missing source commit identity");
            const auto& flags=field("compiler_flags").as_string(); if(flags.empty()) throw std::runtime_error("missing compiler flags");
            q.device_class=field("device_class").as_string(); q.compiler_id=field("compiler_id").as_string()+"|"+flags;
            q.shape_regime=field("shape_regime").as_string();
            const auto& arithmetic=field("arithmetic").as_string(); const auto at=arithmetic.rfind('@');
            if(at==std::string::npos || at==0) throw std::runtime_error("unversioned arithmetic id");
            std::uint32_t revision=0; const auto parsed=std::from_chars(arithmetic.data()+at+1,arithmetic.data()+arithmetic.size(),revision);
            if(parsed.ec!=std::errc{} || parsed.ptr!=arithmetic.data()+arithmetic.size() || !revision) throw std::runtime_error("invalid arithmetic revision");
            table.owned_arithmetic.push_back(std::make_shared<const std::string>(arithmetic.substr(0,at)));
            q.arithmetic={*table.owned_arithmetic.back(),revision};
            q.byte_equal=field("byte_equal").as_bool(); q.faster=field("faster").as_bool(); q.quality_passed=field("quality_passed").as_bool();
            auto status=validate_qualification(q); if(!status) return status;
            const auto key=q.device_class+"/"+q.compiler_id+"/"+q.shape_regime+"/"+field("build_identity").as_string()+"/"+field("incumbent").as_string()+"/"+field("replacement").as_string()+"/"+serialize_id(q.arithmetic);
            if(!keys.insert(key).second) throw std::runtime_error("duplicate measured decision");
            table.entries.push_back(q); table.source_commits.push_back(commit); table.compiler_flags.push_back(flags);
        }
        out=std::move(table); return {Error::ok,{}};
    } catch(const std::exception& e) { return {Error::malformed,e.what()}; }
}

}
