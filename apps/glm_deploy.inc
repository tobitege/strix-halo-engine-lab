// Shared native binding for the rank and serve entry points. All work before
// resolve_live is host-only; the safetensors Index reads headers, not payloads.
struct GlmDeployment {
    std::string file;
    bool check = false;
    strix::deploy::Spec spec;
    std::unique_ptr<strix::glm::deployment::Provider> provider;
    std::shared_ptr<const strix::deploy::Plan> plan;
    std::uint64_t prefix_keep = 0;  // requested kept-prefix bytes per rank; resolve_live lowers it to the plan grant
    bool prefix_keep_pinned = false;
    static strix::deploy::DeviceFacts observe() {
        const char* selected = std::getenv("STRIX_GLM_DRM_DEVICE");
        return strix::deploy::probe_device("", selected ? selected : strix::deploy::first_drm_device());
    }
    void prepare(ge::Options& opt, const rd::Effective& eff, const std::string& transport, std::string refusal = {}) {
        namespace dp = strix::deploy;
        if (file.empty()) file = "auto";
        spec = dp::load_spec(file);
        if (!spec.nodes) spec.nodes = 2;
        const auto link = transport.starts_with("tcp-") ? std::string("tcp") : transport;
        if (spec.link == "auto") spec.link = link.empty() ? "usb4-auto" : link;
        else if (!link.empty() && spec.link != link) refusal = "link pin differs from --transport";
        if (!strix::deploy::pinned(spec, "context")) {
            if (opt.capacity > UINT32_MAX) throw std::invalid_argument("deployment context exceeds u32");
            spec.context = static_cast<std::uint32_t>(opt.capacity);
        }
        if (!strix::deploy::pinned(spec, "slots")) spec.slots = opt.slots;
        opt.capacity = spec.context; opt.slots = spec.slots ? spec.slots : opt.slots;
        if (opt.world != 2 || opt.layers != 45 || opt.external_exchange) refusal = "GLM deployment requires the complete TP2 communicator route";
        opt.pf_pipeline = strix::glm::xfer::prefill_pipeline(opt.pf_pipeline,
            eff.env.at("STRIX_GLM_PF_PIPELINE").value.c_str());
        if (opt.slots > 65536) throw std::invalid_argument("deployment slots exceed provider limit");
        strix::glm::load::Index index(opt.model_dir);
        // The loader's routes (engine.hip moe_fast / attn_tierb, Options::raw_linears).
        strix::glm::deployment::Routes routes;
        routes.fast_moe = eff.env.at("STRIX_GLM_MOE").value == "fast";
        routes.attn_raw = eff.env.at("STRIX_GLM_ATTN").value != "ref" && eff.env.at("STRIX_GLM_TIERB").value != "lib";
        routes.keep_raw = opt.raw_linears;
        const auto owners = strix::glm::deployment::header_owners(index.tensors(), opt.mtp, routes);
        strix::glm::deployment::Options po;
        po.slots = opt.slots; po.mtp = opt.mtp; po.spec = opt.spec; po.refusal = std::move(refusal);
        po.prefix_keep_bytes = prefix_keep; po.prefix_keep_pinned = prefix_keep_pinned;
        std::uint64_t configuration = 14695981039346656037ULL;
        for (const char c : rd::print(eff)) configuration = (configuration ^ static_cast<unsigned char>(c)) * 1099511628211ULL;
        po.configuration_id = std::to_string(configuration);
        if (const char* p = std::getenv("STRIX_GLM_KV_SHARD")) po.kv_shard = p;
        const unsigned rows = static_cast<unsigned>(std::stoul(eff.max_rows.value));
        const bool mtp = opt.mtp;
        const bool chunked = eff.prefill != "ref";
        const auto tier = eff.prefill == "wmma" ? strix::glm::pf::TierB::wmma : strix::glm::pf::TierB::gemv;
        provider = std::make_unique<strix::glm::deployment::Provider>(std::array{owners, owners}, po,
            [rows, mtp, chunked, tier](std::uint32_t context, bool sharded) {
                // Existing bounded workspace calculator, no HIP calls. Its full
                // scratch layout bounds later capacity fallback layouts too.
                const auto b = strix::glm::pf::prefill_budget(32, 32, context, rows, mtp, tier, sharded);
                auto bytes = dp::checked_add(b.engine_rows, chunked ? dp::checked_add(b.path, b.workspace) : 0);
                bytes = dp::checked_add(bytes, chunked ? b.pipeline : 0);
                // Fixed decode scratch, per-allocation rounding and first-use runtime reserve.
                return dp::checked_add(bytes, 1ULL << 30);
            });
    }
    int inspect(unsigned rank) {
        auto p = strix::deploy::resolve(spec,
            strix::deploy::inspection_devices(spec, rank, observe()), *provider);
        std::puts(p.to_json().c_str());
        return p.refused ? 2 : 0;
    }
    void resolve_live(ge::Options& opt, strix::dist::Transport& transport) {
        auto p = strix::deploy::resolve_pair(spec, *provider, opt.rank, observe(),
            [&](auto out, auto in) { transport.exchange(out, in); });
        strix::deploy::require_plan(p);
        opt.capacity = p.chosen.context; opt.slots = p.chosen.slots;
        // Both ranks keep the same snapshots: the pair budget is the smaller rank grant.
        std::uint64_t grant = prefix_keep;
        for (const auto& r : p.ranks) {
            std::uint64_t got = 0;
            for (const auto& [name, bytes] : r.footprint.extra) if (name == "prefix_keep") got = bytes;
            grant = std::min(grant, got);
        }
        prefix_keep = grant;
        plan = std::make_shared<const strix::deploy::Plan>(std::move(p));
    }
};
