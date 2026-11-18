#include <strix/weight/resolve.hpp>
#include <strix/weight/levers.hpp>
#include <strix/weight/source.hpp>
#include <cmath>
#include <cstring>
namespace {
struct Memory final : strix::weight::Source {
    std::array<strix::weight::Byte, 16> bytes{};
    strix::weight::Status read(strix::weight::ByteRange r, std::span<strix::weight::Byte> out) const override {
        if (r.offset > bytes.size() || r.bytes > bytes.size() - r.offset || out.size() != r.bytes)
            return {strix::weight::Error::malformed, "range"};
        std::memcpy(out.data(), bytes.data() + r.offset, out.size()); return {strix::weight::Error::ok, {}};
    }
    std::uint64_t size() const noexcept override { return bytes.size(); }
    strix::weight::Digest256 identity() const override { return strix::weight::digest_bytes(bytes); }
};
}
bool wfmt_resolve_test() {
    using namespace strix::weight;
    auto src = std::make_shared<Memory>();
    WeightMapEntry t{}; t.binding.tensor = t.weight.tensor = 7; t.binding.role = Role::dense_up;
    t.binding.logical_shape = t.weight.shape = {{1,4,0,0},2}; t.binding.expert_axis = UINT32_MAX;
    t.weight.storage = {"plain.f32",1}; t.binding.admitted_formats.push_back(t.weight.storage);
    t.weight.payload = {src,{0,16}};
    const auto& reg = builtin_registry(); const auto* f = find_format(reg,t.weight.storage); if (!f) return false;
    if (!f->geometry(t.weight.parameters,t.weight.geometry)) return false;
    Candidate old{}; old.kernel=1; old.storage=t.weight.storage; old.arithmetic={"cpu-q8a8-tree32",1};
    old.image={"image.raw",1}; old.backend=Backend::cpu; old.strategy=Strategy::native; old.exactness=Exactness::legacy_bitwise;
    old.cost.packed_bytes_per_weight={4,1}; old.cost.image_bytes_per_weight={4,1};
    old.cost.decode={{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1}};
    old.implementation.bytes[0]=1;
    Candidate next=old; next.kernel=2; next.implementation.bytes[0]=2;
    std::array candidates{old,next};
    ResolveRequest r{}; r.tensor=&t; r.hardware.backend=Backend::cpu; r.hardware.compiler_id="test-compiler";
    r.hardware.device_class="test-device"; r.hardware.free_bytes=1024; r.hardware.cache_budget_bytes=1024;
    r.required_arithmetic=old.arithmetic; r.incumbent=1; r.workload={1,1,4,0,0,{},1,false,false};
    Plan out{};
    if (!resolve(reg,r,candidates,{},out) || out.selected.kernel!=1 || !std::isnan(out.predicted_us)) return false;
    Qualification q{}; q.receipt.bytes[0]=1; q.compiler_id=r.hardware.compiler_id; q.device_class=r.hardware.device_class;
    q.shape_regime=shape_regime(r.workload); q.incumbent_implementation=old.implementation; q.replacement_implementation=next.implementation; q.arithmetic=old.arithmetic;
    r.build_identity.bytes[0]=42; q.build_identity=r.build_identity;
    q.byte_equal=q.faster=true;
    std::array qs{q};
    if (!resolve(reg,r,candidates,qs,out) || out.selected.kernel!=2) return false;
    r.build_identity.bytes[0]=43;
    if (!resolve(reg,r,candidates,qs,out) || out.selected.kernel!=1) return false;
    r.build_identity=q.build_identity; candidates[1].implementation.bytes[0]=3;
    if (!resolve(reg,r,candidates,qs,out) || out.selected.kernel!=1) return false;
    candidates[1]=next; candidates[1].kernel=20; // Persistent identity maps to this build's dispatch ID.
    if (!resolve(reg,r,candidates,qs,out) || out.selected.kernel!=20) return false;
    candidates[1]=next; r.build_identity={};
    if (!resolve(reg,r,candidates,qs,out) || out.selected.kernel!=1) return false;
    r.build_identity=q.build_identity;
    r.workload.m=2; if (!resolve(reg,r,candidates,qs,out) || out.selected.kernel!=1) return false;
    r.workload.m=1; apply_decision_override(r,"0");
    if (!resolve(reg,r,candidates,qs,out) || out.selected.kernel!=1) return false;
    apply_decision_override(r,"1"); qs[0].compiler_id="other";
    if (!resolve(reg,r,candidates,qs,out) || out.selected.kernel!=1) return false;
    qs[0]=q; qs[0].byte_equal=false; if (validate_qualification(qs[0])) return false;
    candidates[0].resources.image_bytes=2048; candidates[1].arithmetic={"different",1};
    if (resolve(reg,r,candidates,{},out).code!=Error::memory_budget) return false;
    candidates={old,next}; r.override_strategy=true; r.forced_strategy=Strategy::converted;
    if (resolve(reg,r,candidates,{},out)) return false;
    r.override_strategy=false; candidates[0].exactness=Exactness::fast;
    if (resolve(reg,r,std::span(candidates.data(),1),{},out)) return false;
    r.maximum_permission=Exactness::fast; r.allow_fast=true;
    if (!resolve(reg,r,std::span(candidates.data(),1),{},out)) return false;
    // A specifically requested historical FAST route is still its own migration baseline.
    r.maximum_permission=Exactness::legacy_bitwise;
    if (!resolve(reg,r,std::span(candidates.data(),1),{},out) || out.selected.exactness!=Exactness::fast) return false;
    r.maximum_permission=Exactness::operand_exact;
    if (resolve(reg,r,std::span(candidates.data(),1),{},out)) return false;
    double us=0; if (estimate_cost(old,r.hardware,r.workload,us) || !std::isnan(us)) return false;
    r.hardware.memory_bytes_per_second=1.e9; r.hardware.valu_ops_per_second=1.e12;
    if (!estimate_cost(old,r.hardware,r.workload,us) || std::abs(us-0.016)>1.e-12) return false;
    DecisionTable table;
    const std::string receipt(64,'1'), commit(40,'a');
    const auto document=std::string("{\"schema\":\"wfmt.decisions@2\",\"decisions\":[{\"receipt\":\"")+receipt+
        "\",\"build_identity\":\""+receipt+"\",\"source_commit\":\""+commit+"\",\"device_class\":\"cpu\",\"compiler_id\":\"test\",\"compiler_flags\":\"strict\","
        "\"shape_regime\":\"1:1:4\",\"incumbent\":\""+std::string(64,'2')+"\",\"replacement\":\""+std::string(64,'3')+"\",\"arithmetic\":\"cpu-q8a8-tree32@1\",\"byte_equal\":true,\"faster\":true,\"quality_passed\":false}]}";
    if(!import_decisions(document,table) || table.entries.size()!=1 || serialize_id(table.entries[0].arithmetic)!="cpu-q8a8-tree32@1" || table.entries[0].compiler_id!="test|strict") return false;
    auto imported=table.entries[0]; imported.shape_regime=shape_regime(r.workload);
    r.hardware.compiler_id=imported.compiler_id; r.hardware.device_class=imported.device_class;
    r.build_identity=imported.build_identity; r.maximum_permission=Exactness::legacy_bitwise;
    candidates={old,next}; candidates[0].implementation=imported.incumbent_implementation;
    candidates[1].implementation=imported.replacement_implementation;
    if (!resolve(reg,r,candidates,std::span(&imported,1),out) || out.selected.kernel!=2) return false;
    r.build_identity.bytes[0]^=1;
    if (!resolve(reg,r,candidates,std::span(&imported,1),out) || out.selected.kernel!=1) return false;
    auto obsolete=document; obsolete.replace(obsolete.find("wfmt.decisions@2"),16,"wfmt.decisions@1");
    if (import_decisions(obsolete,table)) return false;
    if(import_decisions("{}",table) || import_decisions("{\"schema\":\"wfmt.decisions@2\",\"schema\":0}",table)) return false;
    auto profile = strix::DeviceProfile{}; profile.arch_name="gfx1151"; profile.wave_size=32; profile.free_memory_bytes=100;
    auto hardware=hardware_from_profile(profile,"compiler");
    return hardware.backend==Backend::hip && hardware.arch=="gfx1151" && hardware.free_bytes==100;
}
