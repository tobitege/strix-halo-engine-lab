// Included in the serve app's anonymous namespace, after its options and host sizing helpers.
// Needs strix/deploy/binding.hpp included before this file (outside any namespace).
// No device allocation or weight loading is permitted in this provider.
class Ds41FootprintProvider final : public strix::deploy::FootprintProvider {
public:
    explicit Ds41FootprintProvider(const Args& a) : a_(a) {}
    std::string model() const override { return a_.model_id; }
    std::vector<std::string> topologies() const override { return {"tp2"}; }
    std::vector<strix::deploy::Format> formats(const std::string&) const override { return {{"native", "native"}}; }
    std::uint32_t max_slots() const override { return BatchForward::configured_row_limit(); }
    BatchForward::Footprint native_footprint(std::uint32_t context, std::uint32_t slots, std::uint32_t rank) const {
        Args selected = a_;
        selected.context = context; selected.slots = slots;
        return BatchForward::footprint(a_.model, forward_options(selected, nullptr, rank));
    }
    strix::deploy::Footprint footprint(const strix::deploy::Spec& spec, const strix::deploy::Candidate& c,
                                     std::uint32_t rank) const override {
        const auto profile = strix::profile::name(strix::deepseek41::pair_defaults::profile_resolution().profile);
        if (spec.profile != "auto" && spec.profile != profile)
            throw std::invalid_argument("profile pin differs from active --profile; select it explicitly at launch");
        // These routes need header-only sizing before they can be admitted by deployment.
        // Existing launches without --deploy retain their established runtime admission.
        if (!a_.dspark.empty()) throw std::invalid_argument("DSpark: header-only peak footprint not implemented");
        if (!a_.hetero_listen.empty()) throw std::invalid_argument("heterogeneous stage: footprint not implemented");
        if (a_.prefix_cache_mb || a_.session_ram_bytes)
            throw std::invalid_argument("prefix/session cache: peak footprint not implemented; use --cache-profile off and no session RAM");
        const auto f = native_footprint(c.context, c.slots, rank);
        if (f.trellis) throw std::invalid_argument("trellis: pre-load library workspace sizing unavailable");
        strix::deploy::Footprint out;
        out.weights = f.weights;
        // BatchForward's state includes KV rings/carries. Charge the complete state to
        // KV first (never a linear extrapolation across compressed-cache boundaries).
        out.kv_total = strix::deploy::checked_add(strix::deploy::checked_mul(f.state_per_slot, c.slots), f.pool);
        out.workspace = f.workspace;
        if (a_.device_prefill) {
            namespace sc = strix::deepseek41::sched;
            if (!sc::auto_wide_pieces(c.context, [](const char* key) {
                    const char* value = std::getenv(key); return value ? std::string(value) : std::string();
                })) throw std::invalid_argument("prefill workspace estimate unqualified for this context/options; use --prefill-path rows");
            strix::deepseek41::PrefillOptions po;
            po.layers = a_.layers; po.max_positions = c.context; po.rank = rank; po.world = 2;
            po.engram_direct_io = a_.engram_direct;
            po.expert_tp = strix::ds41::expert_tp::enabled_from_env();
            po.max_rows = a_.prefill_rows_auto ? std::min(sc::kAutoPrefillRows[0], c.context) : std::min(a_.prefill_rows, c.context);
            if (po.max_rows > sc::kAutoPrefillRows[0]) throw std::invalid_argument("prefill rows exceed measured workspace estimate range");
            const auto scratch = sc::prefill_piece_bytes(po.max_rows, c.context);
            out.workspace = strix::deploy::checked_add(out.workspace, scratch);
            // The live serve passes BatchForward::file_form_weights() as shared_weights: the
            // prefill reads those in place and loads only the rest (the same rule, host-side).
            Args selected = a_;
            selected.context = c.context; selected.slots = c.slots;
            for (const auto& [name, bytes] : BatchForward::file_form_weight_sizes(a_.model, forward_options(selected, nullptr, rank)))
                po.shared_weights.emplace(name, std::make_pair(static_cast<const void*>(nullptr), bytes));
            out.extra.emplace_back("prefill_own_weights", strix::deepseek41::DevicePrefill::owned_weight_bytes(a_.model, po));
        }
        return out;
    }
private:
    const Args& a_;
};

std::string deployment_link(const std::string& transport) {
    // Never put TCP addresses or ports into a public plan.
    if (transport.starts_with("tcp-")) return "tcp";
    return transport;
}
void prepare_deployment(Args& a) {
    if (a.transport == "emulate") throw std::invalid_argument("deployment: emulated ranks do not provide two measured nodes");
    auto spec = strix::deploy::load_spec(a.deploy);
    if (!strix::deploy::pinned(spec, "context")) spec.context = a.context;
    if (!strix::deploy::pinned(spec, "slots")) spec.slots = a.slots;
    if (!spec.nodes) spec.nodes = 2;
    if (spec.link == "auto") spec.link = deployment_link(a.transport);
    else if (spec.link != deployment_link(a.transport))
        throw std::invalid_argument("deployment link pin differs from --transport");
    if (spec.model.empty()) spec.model = a.model_id;
    a.context = spec.context; a.slots = spec.slots;
    a.deploy_spec = std::make_shared<const strix::deploy::Spec>(std::move(spec));
}

void resolve_live_deployment(Args& a, std::uint32_t rank, strix::dist::Transport& transport) {
    // Host transport exchange before Communicator/BatchForward construction. Both ranks
    // resolve the same rank-ordered, freshly measured inventory, including mixed memory sizes.
    auto plan = strix::deploy::resolve_pair(*a.deploy_spec, Ds41FootprintProvider(a), rank, strix::deploy::probe_device(),
        [&](auto out, auto in) { transport.exchange(out, in); });
    strix::deploy::require_plan(plan);
    const auto id = plan.hash();
    a.context = plan.chosen.context; a.slots = plan.chosen.slots;
    a.deployment_plan = std::make_shared<const strix::deploy::Plan>(std::move(plan));
    std::fprintf(stderr, "deploy: id=%s topology=tp2 context=%u slots=%u\n", id.c_str(), a.context, a.slots);
}
