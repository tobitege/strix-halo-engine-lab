#include <strix/weight/resolve.hpp>
#include <strix/weight/admission.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
namespace strix::weight {
bool operator==(const Digest256& a, const Digest256& b) noexcept { return a.bytes == b.bytes; }
bool operator==(const ImageKey& a, const ImageKey& b) noexcept {
    return a.source == b.source && a.sidecars == b.sidecars && a.metadata == b.metadata &&
        a.tensor == b.tensor && a.payload.offset == b.payload.offset && a.payload.bytes == b.payload.bytes &&
        a.byte_strides == b.byte_strides && a.has_byte_strides == b.has_byte_strides &&
        a.shape.dims == b.shape.dims && a.shape.rank == b.shape.rank && a.storage == b.storage &&
        a.image == b.image && a.arithmetic == b.arithmetic && a.backend == b.backend &&
        a.arch == b.arch && a.kernel_abi == b.kernel_abi;
}
std::string shape_regime(const Workload& w) {
    std::ostringstream out;
    out << w.m << ':' << w.n << ':' << w.k << ':' << w.experts << ':' << w.active_experts
        << ':' << w.concurrent_requests << ':' << w.warm_image << ':' << w.warm_file;
    for (auto rows : w.rows_per_expert) out << ':' << rows;
    return out.str();
}
Hardware hardware_from_profile(const strix::DeviceProfile& p, std::string compiler) {
    Hardware h{};
    h.backend = Backend::hip; h.arch = p.arch_name; h.compiler_id = std::move(compiler);
    h.device_class = p.arch_name; h.free_bytes = p.free_memory_bytes;
    h.warp_or_wave = static_cast<std::uint32_t>(std::max(p.wave_size, 0));
    h.shared_bytes_per_sm = static_cast<std::uint32_t>(p.lds_bytes);
    h.isa_mask = (p.isa.rdna35_wmma ? 1u : 0u) | (p.isa.wmma_bf16_w32 ? 2u : 0u) |
        (p.isa.dpp_row ? 4u : 0u) | (p.isa.sudot4 ? 8u : 0u);
    return h;
}
namespace {
Status eligible(const ResolveRequest& r, const Candidate& c) {
    if (c.validate) { auto s = c.validate(r.tensor->weight, r.workload, r.hardware); if (!s) return s; }
    if (c.backend != r.hardware.backend) return {Error::unsupported_backend, "candidate backend differs from execution context"};
    if (!(c.storage == r.tensor->weight.storage)) return {Error::unknown_format, "candidate storage differs from admitted tensor"};
    if (!(c.arithmetic == r.required_arithmetic)) return {Error::unsupported_arithmetic, "no memory fallback to a different arithmetic contract"};
    if (r.maximum_permission == Exactness::operand_exact && c.exactness != Exactness::operand_exact)
        return {Error::unsupported_arithmetic, "operand-exact permission is restricted to operand tests"};
    if (c.exactness == Exactness::operand_exact && r.maximum_permission != Exactness::operand_exact)
        return {Error::unsupported_arithmetic, "operand tests cannot become a model execution path"};
    if (c.exactness == Exactness::fast && (!r.allow_fast || r.maximum_permission == Exactness::operand_exact))
        return {Error::unqualified, "FAST requires explicit permission and named arithmetic"};
    if ((c.required_isa & r.hardware.isa_mask) != c.required_isa) return {Error::unavailable_isa, "required ISA is unavailable"};
    if (c.resources.registers > r.hardware.registers_per_thread_limit && r.hardware.registers_per_thread_limit)
        return {Error::unavailable_isa, "register requirement exceeds device limit"};
    if (c.resources.shared_bytes > r.hardware.shared_bytes_per_sm && r.hardware.shared_bytes_per_sm)
        return {Error::unavailable_isa, "shared memory requirement exceeds device limit"};
    if (c.resources.image_bytes > r.hardware.free_bytes ||
        c.resources.workspace_bytes > r.hardware.free_bytes - c.resources.image_bytes ||
        (c.resources.image_bytes && c.resources.image_bytes > r.hardware.cache_budget_bytes))
        return {Error::memory_budget, "insufficient reserved image/workspace budget; arithmetic fallback forbidden"};
    if (r.override_strategy && c.strategy != r.forced_strategy) return {Error::unqualified, "forced image strategy has no qualified candidate"};
    return {Error::ok, {}};
}
}
static Status resolve_impl(const Registry& registry, const ResolveRequest& r, std::span<const Candidate> candidates,
               std::span<const Qualification> qualifications, Plan& out) {
    if (!r.tensor) return {Error::malformed, "missing admitted tensor"};
    if (std::none_of(registry.arithmetic.begin(), registry.arithmetic.end(), [&](const auto& a) { return a.id == r.required_arithmetic; }))
        return {Error::unsupported_arithmetic, "unknown arithmetic descriptor"};
    const auto* format = find_format(registry, r.tensor->weight.storage);
    if (!format) return {Error::unknown_format, "unknown storage descriptor"};
    if (!format->semantic_decode_supported) return {Error::unsupported_arithmetic, "parse-only or typed-only descriptor has no compute route"};
    { auto s = validate_binding(registry, *r.tensor); if (!s) return s; }
    const Candidate* selected = nullptr;
    Digest256 receipt{};
    Status decline{Error::unqualified, "no qualified incumbent registration; dependency=backend source; fixture=legacy byte capture; removal=register and byte-qualify the existing route"};
    for (const auto& c : candidates) {
        auto status = eligible(r, c);
        if (!status) { if (c.kernel == r.incumbent || r.override_strategy) decline = status; continue; }
        if (c.kernel == r.incumbent) { selected = &c; break; }
    }
    // A measured decision is valid only for this exact compiler/device/regime and arithmetic.
    // Iterate in registration order to keep ties deterministic.
    const auto incumbent = std::find_if(candidates.begin(), candidates.end(), [&](const auto& c) { return c.kernel == r.incumbent; });
    if (r.use_qualified_decisions && !(r.build_identity == Digest256{}) && incumbent != candidates.end()) for (const auto& c : candidates) {
        if (!eligible(r, c)) continue;
        for (const auto& q : qualifications) {
            if (!validate_qualification(q) || !(q.build_identity == r.build_identity) ||
                !(q.incumbent_implementation == incumbent->implementation) || !(q.replacement_implementation == c.implementation) ||
                !(q.arithmetic == r.required_arithmetic) || q.compiler_id != r.hardware.compiler_id ||
                q.device_class != r.hardware.device_class || q.shape_regime != shape_regime(r.workload)) continue;
            // Persistent content IDs become local dispatch IDs only after matching
            // the loaded build and an unambiguous implementation registration.
            if (std::count_if(candidates.begin(), candidates.end(), [&](const auto& candidate) { return candidate.implementation == c.implementation; }) != 1) continue;
            if (c.exactness == Exactness::fast && !q.quality_passed) continue;
            selected = &c; receipt = q.receipt; break;
        }
        if (!(receipt == Digest256{})) break;
    }
    if (!selected) return decline;
    Plan plan{}; plan.selected = *selected; plan.workspace_bytes = selected->resources.workspace_bytes;
    plan.qualification = receipt; plan.predicted_us = std::numeric_limits<double>::quiet_NaN();
    const auto cost = estimate_cost(*selected, r.hardware, r.workload, plan.predicted_us);
    plan.reason = receipt == Digest256{} ? "legacy incumbent; no measured replacement for exact regime" : "byte-qualified measured decision";
    if (!cost) plan.reason += "; cost unavailable: " + cost.detail;
    out = std::move(plan);
    return {Error::ok, {}};
}
Status resolve(const Registry& registry, const ResolveRequest& request, std::span<const Candidate> candidates,
               std::span<const Qualification> qualifications, Plan& out) {
    auto status=resolve_impl(registry,request,candidates,qualifications,out);
    if (!status) status.detail += "; dependency=descriptor/source eligibility (src/weight/resolve.cpp:eligible); fixture=legacy bytes for this request; removal=satisfy the named constraint and qualify matching arithmetic";
    return status;
}

std::string effective_config(const WeightMap& map, std::span<const Plan> plans) {
    std::ostringstream out;
    for (std::size_t i = 0; i < map.size(); ++i) {
        out << "weight tensor=" << map[i].binding.tensor << " storage=" << serialize_id(map[i].weight.storage);
        if (i < plans.size()) {
            const auto& p = plans[i];
            out << " arithmetic=" << serialize_id(p.selected.arithmetic) << " image=" << serialize_id(p.selected.image)
                << " kernel=" << p.selected.kernel << " strategy=" << static_cast<unsigned>(p.selected.strategy)
                << " workspace=" << p.workspace_bytes << " reason=" << p.reason;
        } else out << " reason=unresolved: backend registration and byte fixture required";
        out << '\n';
    }
    return out.str();
}
}
