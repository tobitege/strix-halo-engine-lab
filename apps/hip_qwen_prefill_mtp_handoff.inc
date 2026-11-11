// Included inside strix::decode_cli after TiledMtpPrompt. This opt-in runner
// keeps the ordinary single-node and plain pair handoff prompt paths unchanged.
inline HipMtpTargetHiddenTicket run_pair_tiled_mtp_prompt(std::span<const std::uint32_t> tokens,
    const ModelContract& model,const CompiledSerialExecution& compiled,MtpDecodeTargetParents::TiledPrompt& tiled,
    SerialSessionContext& session,HipMtpTargetHidden& target,HipMtpRuntime& runtime,
    const QualifiedMtpDraftPlan& qualified,const ActivatedMtpDraftPlan& activated,
    const ExecutionRuntimeResources& resources,DirectLaunchDispatcher& dispatcher,std::byte* held,std::string& receipt) {
    if(!qwen_pp::decode_enabled() || model.hidden_layers!=48U || !tiled.rows || tokens.empty() ||
       session.next_position || runtime.next_source_position() ||
       tiled.cache || tiled.rtx_import || !resources.session_state_layout || !resources.session_state_memory ||
       !resources.model_weights || !tiled.block_owner ||
       tiled.block_owner->output_policy()==KnownWindowOutputPolicy::state_only)
        throw std::invalid_argument("pair handoff MTP requires fresh tiled prefill without prefix/session restore");
    const char* tiled_start=std::getenv("STRIX_MTP_TILED_START");
    if(tiled_start && tiled_start[0]=='0') throw std::invalid_argument("pair handoff MTP requires tiled start");
    const unsigned rank=std::getenv("STRIX_QWEN_PF_PIPE_RANK")[0]=='0'?0U:1U;
    qwen_pp::LegalCuts legal{};
    for(std::uint32_t layer=1;layer<48U;++layer) {
        const auto first=std::find_if(compiled.execution.steps.begin(),compiled.execution.steps.end(),
            [&](const auto& step){return step.layer==layer;});
        legal[layer]=first!=compiled.execution.steps.end() && first->op==SerialOp::attn_hyper_mix &&
            std::any_of(compiled.execution.steps.begin(),compiled.execution.steps.end(),
                [&](const auto& step){return step.layer==layer && step.op==SerialOp::attn_inject;});
    }
    const char* cut_env=std::getenv("STRIX_QWEN_PF_PIPE_CUT");
    const char* costs_env=std::getenv("STRIX_QWEN_PF_PIPE_COSTS");
    qwen_pp::CutPrediction prediction;
    std::optional<qwen_pp::LayerCosts> route_costs;
    if(costs_env) {
        const auto costs=qwen_pp::load_costs(costs_env,tokens.size());
        route_costs=costs;
        prediction=cut_env?qwen_pp::predict_cut(costs,qwen_pp::parse_cut(cut_env,legal)):qwen_pp::choose_cut(costs,legal);
    } else prediction.cut=qwen_pp::parse_cut(cut_env,legal);
    // Respect the ordinary MTP owner's tail ladder, rather than inventing a
    // partial-T numerical route for rows that single-node MTP runs serially.
    auto route_plan=tiled.numerical_plan;
    route_plan.rows=tiled.rows;
    route_plan.schedule=qwen_bridge_plan_schedule(std::getenv("STRIX_QPF2K_SCHED"));
    route_plan.suppressed=qpf2k::suppressed(qpf2k::Item::sched,tiled.blocks->gsq_model());
    route_plan.tiled_start=true;
    route_plan.ladder=tiled.ladder_tail_rows;
    const auto route_policy=[](KnownWindowOutputPolicy policy) {
        return policy==KnownWindowOutputPolicy::state_only?QwenPlanOutput::state_only:
            policy==KnownWindowOutputPolicy::all_rows?QwenPlanOutput::all_rows:QwenPlanOutput::terminal;
    };
    route_plan.block.rows=tiled.rows;
    route_plan.block.policy=route_policy(tiled.blocks->output_policy());
    for(std::size_t index=0;index<tiled.tails.size();++index) {
        route_plan.tails.at(index).rows=tiled.tails[index].rows;
        route_plan.tails.at(index).policy=route_policy(tiled.tails[index].backend->output_policy());
    }
    QwenPlanCursor route_cursor;
    std::uint64_t legal_rows=0U;
    while(const auto unit=qwen_prompt_next(route_plan,route_cursor,legal_rows,tokens.size())) {
        if(unit->kind==QwenPlanKind::serial || unit->kind==QwenPlanKind::window) break;
        legal_rows=unit->end;
    }
    auto route=qwen_pp::choose_prompt_route(tokens.size(),tiled.rows,
        route_costs?&*route_costs:nullptr,prediction);
    route.prefix=std::min(route.prefix,legal_rows);
    if(route.prefix<=tiled.rows) route.prefix=0U;
    const auto all_tokens=tokens;
    if(!route.prefix) {
        if(rank==1U) return {};
        return run_tiled_mtp_prompt(tokens,model,compiled,tiled,session,target,runtime,
            qualified,activated,resources,dispatcher,held,receipt);
    }
    tokens=tokens.first(static_cast<std::size_t>(route.prefix));
    const auto range=rank==0U?"0:"+std::to_string(prediction.cut):std::to_string(prediction.cut)+":48";
    // Window verification after handoff must never inherit the stage-only range.
    struct RestoreRange {
        std::string previous;
        bool present;
        ~RestoreRange() { reset(); }
        void reset() {
            qwen_pp::pipe_context={};
            if(present) (void)process_environment::set("STRIX_QWEN_PF_LAYER_RANGE",previous.c_str(),true);
            else (void)process_environment::unset("STRIX_QWEN_PF_LAYER_RANGE");
        }
    };
    const char* previous_range=std::getenv("STRIX_QWEN_PF_LAYER_RANGE");
    RestoreRange restore{previous_range?previous_range:"",previous_range!=nullptr};
    if(!process_environment::set("STRIX_QWEN_PF_LAYER_RANGE",range.c_str(),true))
        throw std::runtime_error("pair handoff MTP layer range update failed");
    auto link=std::make_unique<qwen_pp::PairTransport>(rank);
    qwen_pp::Pipe pipe(rank,(tokens.size()+tiled.rows-1U)/tiled.rows,
        [&](std::span<const std::byte> data){link->send(data);},
        [&](std::span<std::byte> data){link->receive(data);},[&]{link->abort("pair handoff MTP stopped");},prediction.cut,
        [&]{return qwen_pp::make_hip_pipe_staging(static_cast<std::size_t>(tiled.rows)*qwen_pp::Seam::row_bytes);},
        [](std::span<const std::byte> data){return dist::crc32c(data);},
        [&](bool sending, qwen_pp::PipeStaging& staging, std::size_t offset, std::size_t bytes) {
            return link->piece(sending, staging, offset, bytes);
        });
    auto& context=qwen_pp::pipe_context;
    context={&pipe,rank,0U,compiled.execution_identity^tiled.numerical_plan.block.execution^
        tiled.numerical_plan.block.kernel,prediction.cut};
    const auto bind_content=[&](const ModelContentIdentity& content) {
        for(const char byte:model_content_identity_hex(content))
            context.identity=(context.identity^static_cast<unsigned char>(byte))*1099511628211ULL;
    };
    bind_content(resources.model_weights->model_content_identity);
    bind_content(runtime.resources().binding().model_content_identity);
    for(const auto& token:all_tokens)
        for(unsigned shift=0;shift<32U;shift+=8U)
            context.identity=(context.identity^((token>>shift)&255U))*1099511628211ULL;
    const auto start=std::chrono::steady_clock::now();
    TiledMtpPrompt prompt(tokens,model,compiled,tiled,session,target,runtime,qualified,activated,resources,dispatcher,held);
    while(!prompt.done()) {
        prompt.step_pair();
        if(rank==1U) pipe.release(context.sequence);
        ++context.sequence;
    }
    pipe.finish(context.sequence); // join the seam worker before one-shot state transfer
    link->finish();
    HipMtpTargetHiddenTicket pending;
    if(rank==1U) pending=prompt.finish(receipt); // drains all tap/predictor work
    const auto check_copy=[](hipError_t status) {
        if(status!=hipSuccess) throw std::runtime_error(std::string("pair handoff MTP state copy/fence: ")+hipGetErrorString(status));
    };
    check_copy(hipDeviceSynchronize());
    const auto handoff_start=std::chrono::steady_clock::now();
    auto logits=prompt.take_pair_logits();
    if(rank==0U) logits.resize(static_cast<std::size_t>(model.vocab_size));
    if(logits.size()!=model.vocab_size) throw std::runtime_error("pair handoff MTP has no terminal logits");
    const auto& draft_model=runtime.resources().model();
    const auto hidden_bytes=mtp_cli_mul(mtp_cli_mul(model.hidden_size,model.hc_count),2U);
    const qwen_pp::MtpStateDescriptor descriptor{qwen_pp::mtp_state_magic,1U,tokens.size()-1U,tokens.back(),
        hidden_bytes,mtp_cli_mul(mtp_cli_mul(draft_model.kv_heads,draft_model.head_dim),4U),
        mtp_cli_mul(draft_model.indexer_head_dim,2U),runtime.numerical_profile_identity(),
        runtime.resources().binding().compatibility_identity};
    const auto extra=qwen_pp::mtp_state_parts(session.next_position,descriptor);
    auto kv=runtime.resources().view(HipMtpBuffer::kv),raw=runtime.resources().view(HipMtpBuffer::raw_index);
    if(extra[2].bytes>kv.bytes || extra[3].bytes>raw.bytes)
        throw std::runtime_error("pair handoff MTP predictor rows exceed retained backing");
    if(rank==1U) {
        const auto cache=runtime.cache_state();
        if(cache.logical_rows!=descriptor[2] || cache.kv.address!=kv.address || cache.raw_index.address!=raw.address ||
           cache.kv.bytes!=extra[2].bytes || cache.raw_index.bytes!=extra[3].bytes)
            throw std::runtime_error("pair handoff MTP predictor frontier mismatch");
    }
    std::vector<qwen_pp::StatePart> parts(extra.begin(),extra.end());
    std::vector<std::uintptr_t> addresses{0U,reinterpret_cast<std::uintptr_t>(held),kv.address,raw.address};
    const auto& layout=*resources.session_state_layout;
    const auto& memory=*resources.session_state_memory;
    const auto append=[&](qwen_pp::StatePart part,std::uintptr_t address) {
        parts.push_back(part);addresses.push_back(address);
    };
    for(std::uint32_t layer=prediction.cut;layer<48U;++layer) {
        const bool attention=(layer+1U)%model.full_attention_interval==0U;
        for(const auto component:{SessionStateComponent::attention_kv,SessionStateComponent::recurrent_matrix,
            SessionStateComponent::gdn_convolution,SessionStateComponent::sparse_index}) {
            const bool indexed=component==SessionStateComponent::attention_kv || component==SessionStateComponent::sparse_index;
            if(indexed!=attention) continue;
            const auto address=indexed?resolve_indexed_attention_layer_state(layout,memory,model,layer,component):
                resolve_gdn_layer_state(layout,memory,model,layer,component);
            append({layer,static_cast<std::uint32_t>(component),address.bytes},address.address);
        }
    }
    // Stage A owns the input PLE history; stage B skips that prelude. Keep
    // rank 0's history instead of installing stage B's unwritten allocation.
    const auto logits_index=parts.size();
    append({48U,0xffffffffU,model.vocab_size*sizeof(float)},0U);
    auto received_descriptor=descriptor;
    auto state_staging=qwen_pp::make_hip_state_staging();
    const auto bytes=qwen_pp::handoff_overlap(rank==1U,parts,
        qwen_pp::mtp_state_identity(context.identity^session.state_layout_identity,descriptor),
        session.next_position,prediction.cut,*state_staging,
        [&](std::span<const std::byte> data){link->send(data);},
        [&](std::span<std::byte> data){link->receive(data);},
        [&](bool save,std::size_t index,std::uint64_t offset,std::span<std::byte> data) {
            if(index==0U || index==logits_index) {
                auto host=index==0U?std::as_writable_bytes(std::span(received_descriptor)):
                    std::as_writable_bytes(std::span(logits));
                host=host.subspan(static_cast<std::size_t>(offset),data.size());
                if(save) std::copy(host.begin(),host.end(),data.begin());
                else {
                    std::copy(data.begin(),data.end(),host.begin());
                    if(index==0U && received_descriptor!=descriptor)
                        throw std::runtime_error("pair handoff MTP descriptor payload mismatch");
                }
            } else {
                auto* device=reinterpret_cast<void*>(addresses[index]+offset);
                state_staging->copy_device(save,device,data);
            }
        },[](std::span<const std::byte> data){return dist::crc32c(data);},
        [&]{link->abort("pair handoff MTP state transfer failed");});
    check_copy(hipDeviceSynchronize());
    if(rank==0U) {
        runtime.restore_cache_frontier(qualified,activated,descriptor[2]);
        MtpDeviceView terminal{reinterpret_cast<std::uintptr_t>(held),hidden_bytes,0U};
        pending=target.adopt_prefill_row(session,tokens.back(),terminal,logits,model.rotary_identity());
        receipt="\"pair_mtp_handoff\":true,\"predictor_rows\":"+std::to_string(descriptor[2]);
    }
    std::fprintf(stderr,"qwen-handoff-mtp rank=%u cut=%u handoff-bytes=%llu handoff-seconds=%.6f prefill-handoff-seconds=%.6f\n",
        rank,prediction.cut,static_cast<unsigned long long>(bytes),
        std::chrono::duration<double>(std::chrono::steady_clock::now()-handoff_start).count(),
        std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
    restore.reset();
    if(rank==0U && tokens.size()<all_tokens.size()) {
        // The restored predictor has prefix-1 source rows and the held terminal
        // hidden row, exactly the existing session-resume frontier contract.
        TiledMtpPrompt suffix(all_tokens,model,compiled,tiled,session,target,runtime,qualified,activated,
            resources,dispatcher,held,{}, {}, {},true);
        while(!suffix.done()) suffix.step();
        pending=suffix.finish(receipt);
    }
    return pending;
}
