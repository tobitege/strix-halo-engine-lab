// Included after the Qwen host compiler and HTTP bindings.
struct QwenDeployment {
    std::shared_ptr<const strix::deploy::Plan> plan;
    std::string effective_json;
    std::unique_ptr<strix::dist::Transport> pair_link;
    static unsigned rank() {
        const char* r = std::getenv("STRIX_QWEN_PF_PIPE_RANK");
        return r && std::string_view(r) == "1" ? 1U : 0U;
    }
    void resolve(strix::decode_cli::EffectiveConfig& effective, const strix::ModelContract& model,
                 const strix::ModelManifest& manifest, std::uint64_t model_bytes,
                 const strix::decode_cli::PrefillMemory& memory) {
        namespace dp = strix::deploy;
        namespace dc = strix::decode_cli;
        auto& o = effective.options;
        auto spec = *o.deployment_spec;
        if (!strix::deploy::pinned(spec, "context")) {
            if (o.context_tokens > UINT32_MAX) throw std::invalid_argument("deployment context exceeds u32");
            spec.context = static_cast<std::uint32_t>(o.context_tokens);
        }
        const bool pipeline = strix::qwen_pp::pipe_enabled();
        const auto topology = pipeline ? std::string("pipeline") :
            (spec.topology == "single" ? "single" :
             ((spec.nodes == 2 || spec.devices.size() == 2 || spec.topology == "replica" || spec.topology == "replicas") ? "replica" : "single"));
        if (!spec.nodes) spec.nodes = spec.devices.empty() ? (topology == "single" ? 1 : 2) : static_cast<std::uint32_t>(spec.devices.size());
        std::string refusal = dc::deployment_startup_refusal(effective);
        if (topology == "single" && (spec.nodes != 1 || spec.devices.size() > 1))
            refusal = "Qwen single binding requires one local measured node; supply that node's inventory";
        if (o.rotary.mode() != strix::RotaryMode::native || o.residency != strix::HipModelResidencyMode::registered_file_mapping ||
            o.execution.mode != strix::DecodeExecutionMode::eager)
            refusal = "Qwen deployment: rotary/copy/graph owner envelope not bound; use native registered eager route";
        if (pipeline && o.http_serve) refusal = "Qwen pipeline is a CLI prefill/handoff route; HTTP pipeline is unavailable";
        if (refusal.empty() && !o.known_prefix_prefill_rows) refusal = "Qwen deployment requires the existing tiled prefill owner calculator";
        dp::Footprint f;
        std::uint64_t slot_retained = 0, slot_peak = 0, slot_state = 0;
        if (refusal.empty()) try {
            // Reuse the compiler's owner registry (also used by effective-config
            // admission), including the closed retained/staging census. This is
            // an upper bound across phases, not another parameter-size estimate.
            const auto req = dc::compile_prompt_requirements(o, model, manifest, model_bytes, nullptr, true);
            if (!req.owner_error.empty()) refusal = req.owner_error;
            for (const auto& [name, bytes] : req.owners) {
                // HTTP owns a different bridge envelope, already sized by its
                // existing asset-verified admission callback below.
                if (o.http_serve && name.starts_with("mtp_") && name != "mtp_m1_keys" && name != "mtp_m1_scratch_metadata") continue;
                if (name == "model_backing") f.weights = dp::checked_add(f.weights, bytes);
                else if (name == "target_kv" || name == "gdn_matrix" || name == "gdn_convolution" ||
                         name == "ple_convolution" || name == "target_raw_index" || name == "ngram_history" ||
                         name == "position_history" || name == "state_alignment_and_d1_shadow" ||
                         name == "qsa_decode_pool" || name == "mtp_m1_keys" || name == "mtp_m1_scratch_metadata")
                    f.kv_total = dp::checked_add(f.kv_total, bytes);
                else if (name.find("staging") != name.npos) f.staging = dp::checked_add(f.staging, bytes);
                else f.extra.emplace_back(name, bytes);
            }
            if (!f.weights || !f.kv_total) refusal = "Qwen compiler returned an incomplete owner inventory";
            // Reclassify compiled workspace once, leaving every other owner named.
            for (auto& [name, bytes] : f.extra)
                if (name == "prefill_workspace_and_status" || name == "tail_workspaces_and_status" ||
                    name == "serial_scratch" || name == "qsa_score_ring" || name == "terminal_logits_stripes") {
                    f.workspace = dp::checked_add(f.workspace, bytes); bytes = 0;
                }
            // external_host/device describe aliases of owners already in this
            // registry; do not charge their aggregate again.
            if (o.http_serve) {
                dc::validate_http_serving_memory(o,memory);
                const auto serving = memory.serving ? memory.serving(o) : dc::HttpServingMemory{};
                f.extra.emplace_back("http_bridge_shared", std::max(serving.shared_host, serving.shared_device));
                auto slot = std::max(serving.slot_host, serving.slot_device);
                const auto pool = dc::qsa_mtp_pool_bytes(o, model, strix::qpf2k::has_gsq_gate(manifest));
                if (slot < pool) throw std::invalid_argument("HTTP bridge inventory smaller than shared QSA pool");
                slot_peak = slot-pool;
                slot_retained = req.http_slot_retained;
                slot_state = req.http_slot_state;
                if (!slot_retained) throw std::invalid_argument("HTTP retained stack inventory unavailable");
                strix::SerialSessionStateLayoutRequest state_request;
                state_request.context_tokens = o.context_tokens;
                state_request.recurrent_storage = o.gdn_recurrent_storage;
                f.staging = dp::checked_add(f.staging, dc::qwen_http_batch_staging_reserve(o, model,
                    strix::build_serial_session_state_layout(model, state_request)));
            }
            const auto cache = o.http_serve && o.prefix_cache ? std::max<std::uint64_t>(o.prefix_cache_host_bytes, 4ULL<<30) : o.prefix_cache_host_bytes;
            f.extra.emplace_back("prefix_cache_upper_bound", cache);
            // Bound capture/replacement staging by complete existing state
            // reservations; this includes both checkpoint images and controls.
            if (cache) f.staging = dp::checked_add(f.staging, dp::checked_mul(f.kv_total, 4));
            if (o.supplied_options.contains("--session-ram-bytes")) f.extra.emplace_back("session_ram", o.session_ram_bytes);
            else f.elastic_extra.emplace_back("session_ram", o.session_ram_bytes);
            if (pipeline) f.staging = dp::checked_add(f.staging,
                dp::checked_mul(2ULL*o.known_prefix_prefill_rows, strix::qwen_pp::Seam::row_bytes));
            for (const char* identity : {"serial_execution_identity", "serial_kernel_plan_identity",
                                       "prefill_execution_identity", "prefill_kernel_plan_identity"})
                f.extra.emplace_back(std::string(identity) + ":" + std::to_string(req.organization.at(identity)), 0);
        } catch (const std::exception& e) { refusal = e.what(); }
        const auto local = dp::probe_device();
        std::uint32_t max_slots = 1;
        if (dc::ordinary_http_batch_slots(o) && slot_retained) {
            // Loose finite search ceiling; resolve still charges every owner and
            // checks each measured rank. CLI --http-batch N remains a cap.
            const auto per_slot = dp::checked_add(slot_retained, slot_peak);
            max_slots = static_cast<std::uint32_t>(std::clamp<std::uint64_t>(local.mem_free_bytes/per_slot, 1, 65536));
            if (o.http_batch_cap) max_slots = std::min(max_slots, o.http_batch_cap);
        }
        strix::qwen::deployment::Provider provider(std::move(f), static_cast<std::uint32_t>(o.context_tokens), max_slots, topology,
            std::string(strix::profile::name(o.product_profile.profile)), refusal, slot_retained, slot_peak, slot_state);
        const auto* transport_env = std::getenv("STRIX_QWEN_PF_PIPE_TRANSPORT");
        const std::string transport = transport_env ? transport_env : "usb4auto";
        if (spec.link == "auto") spec.link = pipeline ? transport : "none";
        else if (pipeline && spec.link != transport && !(spec.link == "usb4-auto" && transport == "usb4auto"))
            throw std::invalid_argument("deployment link pin differs from pair transport");
        dp::Plan p;
        if (pipeline && !o.deploy_check && !o.print_effective_config) {
            if (transport == "usb4auto" || transport == "usb4-auto") {
                strix::dist::Usb4AutoOptions usb; usb.rank = rank();
                pair_link = strix::dist::usb4_host_auto(usb);
            } else if (transport == "tcp") {
                const char* peer = std::getenv("STRIX_QWEN_PF_PIPE_PEER");
                if (rank() && !peer) throw std::invalid_argument("pipeline rank 1 requires --peer");
                const auto timeout = std::chrono::milliseconds(120000);
                pair_link = rank() ? strix::dist::tcp_connect(peer, 29872, timeout) : strix::dist::tcp_listen(29872, timeout);
            } else throw std::invalid_argument("deployment: unsupported pipeline transport");
            p = dp::resolve_pair(spec, provider, rank(), local, [&](auto out, auto in) { pair_link->exchange(out, in); });
        } else p = dp::resolve(spec, dp::inspection_devices(spec, rank(), local), provider);
        if (!p.refused) {
            const auto node = topology == "single" ? 0U : rank();
            for (const auto& [name, bytes] : p.ranks.at(node).footprint.extra)
                if (name == "session_ram") o.session_ram_bytes = bytes;
            if (dc::ordinary_http_batch_slots(o)) {
                o.http_batch_cap = p.chosen.slots;
                o.http_batch = p.chosen.slots; // applied count; preserve explicit/automatic provenance
            }
        }
        plan = std::make_shared<const dp::Plan>(std::move(p));
        effective_json = dc::effective_config_json(effective);
    }
    std::string json() const {
        auto out = plan->to_json(); out.pop_back();
        return out + ",\"effective_config\":" + effective_json + "}";
    }
    strix::serve::DeploymentCallbacks callbacks() const {
        if (!plan) return {};
        strix::serve::DeploymentCallbacks out;
        out.deployment_json = [body = json()] { return body; };
        out.deployment_id = [p = plan] { return p->hash(); };
        return out;
    }
};
