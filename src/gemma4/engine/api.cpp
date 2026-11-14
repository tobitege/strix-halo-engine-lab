#include "api.hpp"
#include "strix/serve/api.hpp"
#include "strix/serve/request_fields.hpp"
#include "native_memory.hpp"
#include "session_engine.hpp"
#include "gemma4/features/features.hpp"
#include "strix/env_flags.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <limits>
#include <thread>

namespace strix::gemma4::engine {
namespace {
using Json = strix::serve::Json;
using strix::serve::HttpRequest;
using strix::serve::HttpResponseWriter;
const Json& get(const Json& j, const char* name) {
    static const Json empty;
    const auto* p = j.find(name); return p ? *p : empty;
}
std::uint64_t integer(const Json& j, const char* field, std::uint64_t maximum = UINT32_MAX) {
    const auto n = j.as_int();
    if (n < 0 || std::uint64_t(n) > maximum) throw std::invalid_argument(std::string("invalid ")+field);
    return std::uint64_t(n);
}
std::size_t env_count(const char* name, std::size_t fallback, std::size_t maximum) {
    return std::min(strix::env_number<std::size_t>(name, fallback), maximum);
}
bool serve_batched() { return strix::env_flag_default_on("STRIX_G4_SERVE_BATCH"); }
}
std::size_t serve_slots(std::size_t requested) {
    if (!serve_batched()) return 1;
    return std::clamp<std::size_t>(requested != 0 ? requested : env_count("STRIX_G4_SERVE_SLOTS",8,8),1,8);
}
ChatRequest parse_chat_request(const Json& j, DraftMode default_mode, const Request& defaults) {
    if (!j.is_object()) throw std::invalid_argument("chat request must be an object");
    for (const auto& [key,value] : j.as_object()) {
        (void)value;
        bool known = false;
        for (const auto* name : {"model","messages","tools","tool_choice","stream","stream_options","max_tokens",
                "max_completion_tokens","temperature","top_p","top_k","seed","enable_thinking","preserve_thinking",
                "conversational_mode","thinking_budget","chat_template_kwargs","dflash","n"}) known |= key == name;
        if (!known) throw std::invalid_argument("unsupported field: "+key);
    }
    ChatRequest r;
    r.generation = defaults;
    r.generation.dflash = default_mode;
    auto messages = Json(get(j,"messages").as_array());
    // Deep-copy modified message/function objects; Json containers share storage.
    for (auto& m : messages.as_array()) {
        m = Json(m.as_object());
        if (const auto* calls = m.find("tool_calls")) {
            auto converted = Json(calls->as_array());
            for (auto& call : converted.as_array()) {
                call = Json(call.as_object());
                auto fn = Json(get(call,"function").as_object());
                if (get(fn,"arguments").is_string()) fn.set("arguments",strix::serve::parse_json_unique(get(fn,"arguments").as_string()));
                call.set("function",fn);
            }
            m.set("tool_calls",converted);
        }
    }
    Json kwargs = get(j,"chat_template_kwargs").is_null() ? Json::object() : Json(get(j,"chat_template_kwargs").as_object());
    for (const auto* name : {"enable_thinking","preserve_thinking","conversational_mode"}) if (j.find(name)) {
        if (kwargs.find(name)) throw std::invalid_argument(std::string("duplicate template option: ")+name);
        (void)get(j,name).as_bool(); kwargs.set(name,get(j,name));
    }
    r.input = gemma4::serve::chat_input(messages,kwargs);
    if (j.find("tools")) r.input.tools = get(j,"tools");
    if (j.find("tool_choice")) {
        const auto& choice = get(j,"tool_choice").as_string();
        if (choice == "none") r.input.tools = Json();
        else if (choice != "auto") throw std::invalid_argument("unsupported tool_choice");
    }
    if (j.find("n") && integer(get(j,"n"),"n") != 1) throw std::invalid_argument("n must be 1");
    if (j.find("stream")) (void)get(j,"stream").as_bool();
    if (j.find("stream_options")) {
        for (const auto& [key,value] : get(j,"stream_options").as_object())
            if (key != "include_usage" || !value.is_bool()) throw std::invalid_argument("unsupported stream_options");
    }
    if (j.find("max_tokens") && j.find("max_completion_tokens")) throw std::invalid_argument("choose one max token field");
    for (const auto* name : {"max_tokens","max_completion_tokens"}) if (j.find(name)) r.generation.max_tokens = integer(get(j,name),name);
    if (!r.generation.max_tokens) throw std::invalid_argument("max_tokens must be positive");
    if (j.find("temperature")) r.generation.sampling.temperature = float(get(j,"temperature").as_double());
    if (j.find("top_p")) r.generation.sampling.top_p = float(get(j,"top_p").as_double());
    if (j.find("top_k")) r.generation.sampling.top_k = std::uint32_t(integer(get(j,"top_k"),"top_k"));
    if (j.find("seed")) r.generation.sampling.seed = integer(get(j,"seed"),"seed",INT64_MAX);
    r.generation.sampling.validate();
    if (j.find("thinking_budget")) r.thinking_budget = std::uint32_t(integer(get(j,"thinking_budget"),"thinking_budget"));
    if (r.thinking_budget && (!r.input.enable_thinking || r.thinking_budget >= r.generation.max_tokens))
        throw std::invalid_argument("thinking_budget requires thinking and must be below max_tokens");
    if (j.find("dflash")) {
        const auto& mode = get(j,"dflash").as_string();
        if (mode != "on" && mode != "off" && mode != "auto") throw std::invalid_argument("dflash must be on/off/auto");
        r.generation.dflash = mode == "on" ? DraftMode::on : mode == "off" ? DraftMode::off : DraftMode::automatic;
    }
    return r;
}
struct Api::Impl {
    struct Task final : strix::serve::GenerationContext {
        ChatRequest request;
        ChatResult result;
        ChatSink sink;
        std::unique_ptr<ChatJob> chat;
        Request phase;
        std::optional<gemma4::serve::PreparedPrompt> prepared;
        session::KvHandle handle;
        std::uint64_t admitted=0, steps=0, width_sum=0, width_max=0, shared_prefill=0;
        bool prompt_kept=false, ended=false;
        std::optional<TurnCacheRequest> turn;
        std::atomic<bool> cancelled{false};
        Cancelled client_gone;
        std::shared_ptr<strix::serve::GenerationOutput> output = std::make_shared<strix::serve::GenerationOutput>();
    };
    Backend& backend;
    gemma4::serve::Adapter adapter;
    DraftFactory drafts;
    DraftMode mode;
    Request defaults;
    TurnCacheConfig turn_config;
    bool turns=false;
    std::string model;
    strix::serve::GenerationQueue queue;
    strix::serve::DeploymentCallbacks deployment;
    std::shared_ptr<CostModel> costs = std::make_shared<CostModel>();
    std::mutex backend_mutex;
    std::unique_ptr<features::Exporter> feature_exporter;
    std::unique_ptr<strix::serve::ApiService> frontend;
    // STRIX_G4_SERVE_BATCH=0: one chat job at a time (queue.run + run_chat). Otherwise up to STRIX_G4_SERVE_SLOTS
    // (default 8) jobs share one Coordinator and decode together; STRIX_G4_SERVE_PREFILL_PER_STEP (default 1, 0 = all)
    // bounds the prompt chunks one step may prefill. Each job's ids equal the job run alone.
    const bool batched = serve_batched();
    const std::size_t slots;
    const std::size_t prefill_per_step = env_count("STRIX_G4_SERVE_PREFILL_PER_STEP",1,8);
    // Prompt rows per prefill chunk while other requests decode (0 = whole 512-row chunks).
    const std::size_t prefill_rows = env_count("STRIX_G4_SERVE_PREFILL_ROWS",64,512);
    // Prompt prefix cache (STRIX_G4_SERVE_PREFIX=0 off): committed states at the end of each prompt and of each
    // finished request, reused by a later prompt that starts with exactly those tokens (a conversation's next turn).
    // LRU within STRIX_G4_SERVE_PREFIX_MB (default 4096) of KV and 16 entries.
    const bool prefix_cache = batched && strix::env_flag_default_on("STRIX_G4_SERVE_PREFIX");
    const std::size_t prefix_bytes = env_count("STRIX_G4_SERVE_PREFIX_MB",4096,1U << 20U) << 20U;
    std::uint64_t steps=0;
    std::unique_ptr<SessionEngine> session_engine;
    std::unique_ptr<strix::serve::SessionRuntime> session_runtime;
    std::thread worker;
    static std::shared_ptr<Task> task(const strix::serve::GenerationJob& j) {
        auto t=std::dynamic_pointer_cast<Task>(j.family_context);
        if(!t || !t->chat) throw std::invalid_argument("Gemma4 requires typed chat context");
        return t;
    }
    Json phase_receipt(const Task& t) const {
        auto r=session_engine->lane_receipt(t.handle);
        if(batched) r.set("serve_batch",Json(Json::Object{{"mode","continuous"},{"slots",std::uint64_t(slots)},
            {"prefill_per_step",std::uint64_t(prefill_per_step)},{"prefill_rows",std::uint64_t(prefill_rows)},
            {"slot",std::uint64_t(t.handle.slot)},{"admitted_at_step",t.admitted},{"steps",t.steps},
            {"max_batch",t.width_max},{"mean_batch",t.steps?double(t.width_sum)/double(t.steps):0.0},
            {"steps_sharing_prefill",t.shared_prefill}}));
        if(prefix_cache || turns) {const auto c=session_runtime->cache_stats();
            r.set("serve_prefix",Json(Json::Object{{"cached_tokens",session_engine->result(t.handle).cached_tokens},
                {"entries",c.entries},{"bytes",c.bytes},{"hits",c.hits},{"misses",c.misses}}));}
        return r;
    }
    strix::serve::SessionHooks hooks() {
        using namespace strix::serve;
        SessionHooks h;
        h.identity={backend.identity().dump(),effective_config(backend,defaults,bool(drafts)).dump(),"gemma4-prompt-state-v1",1};
        if(turns) {
            const auto identity=backend.identity();
            const auto known_string=[&](const char* key) {const auto* p=identity.find(key);return p && p->is_string() && !p->as_string().empty();};
            // Identity comes from the loaded backend/assets and effective settings.
            // Missing asset identity must never become an invented cache key.
            if(!known_string("model_hash") || !known_string("rope_hash") || !known_string("identity") ||
                !known_string("backend") ||
                (identity.find("backend")->as_string()!="host-reference" && !backend.native_turn_work_bound()) ||
                identity.find("identity")->as_string()!=arithmetic_id) h.identity={};
            else h.identity.state_format="gemma4-complete-turn-state-v1";
            h.turn_cache=[this,bound=backend.identity().dump()](const SessionView& v)->std::optional<SessionTurnCache>{
                const auto t=task(v.job);
                if(backend.identity().dump()!=bound || !t->turn || !t->turn->metadata.known() || !turn_headroom()) return {};
                return SessionTurnCache{t->turn->metadata,turn_config.max_state_bytes};
            };
            h.turn_checkpoint=[this,bound=backend.identity().dump()](const SessionView& v,session::FinishReason reason){
                const auto t=task(v.job);
                if(backend.identity().dump()!=bound || reason==session::FinishReason::none || reason==session::FinishReason::cancelled ||
                    !t->ended || !t->chat->complete_turn() || t->cancelled || t->chat->disconnected() ||
                    v.job.cache_retention==CacheRetention::no_store || !turn_headroom()) return false;
                const auto payload_bound=session_engine->turn_retained_bound(v.state);
                return payload_bound && *payload_bound && *payload_bound<=turn_config.max_state_bytes;
            };
            // This sealed-only producer MUST run after commit, never in the
            // pre-commit turn_checkpoint admission hook above.
            h.turn_snapshot=[this,bound=backend.identity().dump()](const SessionView& v){
                return backend.identity().dump()==bound && session_engine->turn_cache_candidate(v.state);};
        }
        h.validate=[](const GenerationJob& j){const auto t=task(j);
            if(j.prompt!=t->phase.prompt || j.stop_tokens!=t->phase.stop_ids)
                throw std::invalid_argument("Gemma4 context differs from admitted prompt");};
        h.initial_allowance=[](const GenerationJob& j){return std::uint32_t(task(j)->phase.max_tokens);};
        h.opened=[this](const SessionView& v){auto t=task(v.job);t->handle=v.state;t->admitted=steps;
            t->steps=t->width_sum=t->width_max=t->shared_prefill=0;t->prompt_kept=false;
            session_engine->prepare(v.state,t->phase);};
        h.ready=[this](const SessionView& v){session_engine->start(v.state);};
        h.initial_cursor=[this](const SessionView& v){return session_engine->initial_cursor(v.state);};
        h.cancelled=[](const SessionView& v){auto t=task(v.job);return t->cancelled.load() || t->chat->disconnected() || (t->client_gone && t->client_gone());};
        h.prompt_restart_cache=true;
        h.schedule=[this](std::span<const SessionView> views){
            std::size_t prefills=0;for(const auto& v:views) if(session_engine->result(v.state).frontier+1<v.job.prompt.size())++prefills;
            SessionWork work{SessionPhase::mixed,{},{}};
            for(std::size_t i=0;i<views.size();++i){const auto& v=views[i];auto t=task(v.job);
                ++t->steps;t->width_sum+=views.size();t->width_max=std::max<std::uint64_t>(t->width_max,views.size());
                const bool prefill=session_engine->result(v.state).frontier+1<v.job.prompt.size();
                t->shared_prefill+=prefills>(prefill?1U:0U)?1:0;work.lanes.push_back(i);}
            ++steps;return std::vector<SessionWork>{std::move(work)};
        };
        h.prompt_checkpoint=[this](const SessionView& v,SessionCheckpoint boundary){auto t=task(v.job);
            const auto& r=session_engine->result(v.state);const bool same=session_engine->matches_decode();
            if(boundary==SessionCheckpoint::progress){const auto chunk=session_engine->chunk_rows();
                if(t->prompt_kept || r.status==Status::failed || r.frontier<=r.cached_tokens ||
                    !(same?r.frontier+1>=v.job.prompt.size():chunk && r.prefill_frontier==r.frontier && r.frontier%chunk==0 &&
                        v.job.prompt.size()-1-r.frontier<chunk))return false;
                t->prompt_kept=true;
            } else if(!same || (r.status!=Status::stopped && r.status!=Status::length))return false;
            return session_engine->cache_candidate(v.state);
        };
        h.continuation=[this](const SessionView& v,session::FinishReason)->std::optional<SessionContinuation>{auto t=task(v.job);
            const auto& r=session_engine->result(v.state);(void)t->chat->absorb(r);
            if(t->chat->end_phase(r,phase_receipt(*t))){t->phase=t->chat->phase_request();
                return SessionContinuation{t->phase.prompt,t->phase.stop_ids,std::uint32_t(t->phase.max_tokens),SessionContinuation::Mode::restart};}
            t->result=t->chat->take();t->ended=true;return std::nullopt;
        };
        h.receipt=[this](const SessionView& v,GenerationStats& stats){auto t=task(v.job);
            if(!t->ended){const auto& r=session_engine->result(v.state);(void)t->chat->absorb(r);
                (void)t->chat->end_phase(r,phase_receipt(*t));t->result=t->chat->take();t->ended=true;}
            stats.prompt_tokens=t->result.prompt_tokens;stats.generated_tokens=t->result.ids.size();stats.finish=t->result.finish;
            t->output->parsed=gemma4::serve::shared_output(t->result.output);
            auto phases=Json::array();for(const auto& phase:t->result.receipts)phases.push(phase);
            auto ids=Json::array();for(const auto id:t->result.ids)ids.push(std::uint64_t(id));
            t->output->receipt=Json(Json::Object{{"phases",phases},{"budget_continued",t->result.budget_continued},{"generated_ids",ids}});
            stats.cached_tokens=0;for(const auto& phase:t->result.receipts)
                if(const auto* cached=phase.find("serve_prefix"))stats.cached_tokens+=std::uint64_t(cached->find("cached_tokens")->as_int());
        };
        return h;
    }
    bool turn_headroom() const noexcept {
        try {
            const auto payload=turn_config.max_state_bytes;
            const auto work=session_engine->turn_work_bound();
            // All slots may retain a candidate until release. Four extra copies
            // cover rollback/restore installation, candidate and index staging.
            // Cache retained charge is a SEPARATE finite policy budget.
            const auto copies=std::uint64_t(slots)+4;
            return memory::fits(work,payload,copies,turn_config.max_transient_bytes);
        } catch(...) {return false;}
    }
    Impl(Backend& b, std::shared_ptr<const gemma4::serve::Tokenizer> t, DraftFactory d, DraftMode m, std::string name, Request settings,
        strix::serve::DeploymentCallbacks callbacks, Api::Owner owner, TurnCacheConfig cache, std::size_t requested_slots)
        : backend(b), adapter(t), drafts(std::move(d)), mode(m), defaults(std::move(settings)), turn_config(std::move(cache)), model(std::move(name)),
          deployment(std::move(callbacks)), slots(serve_slots(requested_slots)) {
        const auto shared_tokenizer=gemma4::serve::api_tokenizer(t);
        if (!deployment.capacity_provider) deployment.capacity_provider = [this] {
            const auto q = queue.snapshot();return strix::serve::ServeCapacity{slots,q.running,std::nullopt,q.queued};};
        session_engine=std::make_unique<SessionEngine>(backend,drafts,costs,backend_mutex,slots,prefill_per_step,prefill_rows);
        session::Policy policy;policy.enabled=prefix_cache;policy.max_entries=16;policy.max_bytes=prefix_bytes;
        policy.leave_one_token=true;policy.replace_duplicates=true;policy.retain_one_oversize=true;policy.charge_token_bytes=false;
        turns=turn_config.policy.enabled && turn_config.policy.completed_turns;
        if(turns) {
            policy=turn_config.policy;
            // Invalid bounded configuration safely declines the turn route.
            // Do not fall back to the old environment-controlled prompt cache.
            if(policy.max_entries==UINT64_MAX || !turn_config.request || !turn_headroom()) policy.enabled=false;
        }
        session_runtime=std::make_unique<strix::serve::SessionRuntime>(queue,*session_engine,session_engine->store(),policy,hooks());
        frontend=std::make_unique<strix::serve::ApiService>(api_config(),strix::serve::ApiTokenizerBinding(shared_tokenizer),queue,backend.config().max_positions,true);
        if(owner==Api::Owner::standalone) worker=std::thread([this]{
            try {session_runtime->run();} catch(...) {queue.shutdown();}
        });
    }
    ~Impl() {
        session_runtime->shutdown();if(worker.joinable())worker.join();
    }
    strix::serve::ApiConfig api_config() {
        using namespace strix::serve;
        ApiConfig c;
        c.deployment=deployment;c.model_id=model;c.default_thinking=false;
        c.chat_extension_fields={"dflash","thinking_budget","chat_template_kwargs","enable_thinking","preserve_thinking","conversational_mode"};
        c.default_max_tokens=std::uint32_t(defaults.max_tokens);
        c.maximum_temperature=std::numeric_limits<float>::max();c.float_sampling_parameters=true;
        c.default_temperature=defaults.sampling.temperature;c.default_top_p=defaults.sampling.top_p;
        c.default_top_k=defaults.sampling.top_k;c.default_seed=defaults.sampling.seed;
        c.strict_context_limit=true;c.strict_request_json=true;c.compact_chat_envelope=true;c.chat_id_prefix="chatcmpl-g4-";
        c.sse_keepalive_seconds=0;c.sse_progress_seconds=0;
        c.capabilities_extra=[this](Json& out) {
            out.set("identity",*effective_config(backend,defaults,bool(drafts)).find("decode_identity"));
            out.set("dflash",effective_config(backend,defaults,bool(drafts)));
            out.set("features",backend.identity().find("feature_identity")?*backend.identity().find("feature_identity"):Json());
            out.set("session_sampling",Json(Json::Object{{"temperature",true},{"top_p",true},{"top_k",true},{"min_p",false},{"seed",true}}));
            out.set("thinking_budget",true);out.set("thinking_budget_policy","checked-two-phase-continuation");
            // min-p and shared constrained host-selection policies do not run here.
            auto sampling=Json(out.find("sampling")->as_object());sampling.set("min_p_supported",false);out.set("sampling",sampling);
        };
        c.validate_request=[this](const HttpRequest& request,const Json& body) {
            if(body.find("model") && get(body,"model").as_string()!=model)
                throw RequestFieldError("model","unknown model");
            if(request.path=="/detokenize" || request.path=="/tokenize")return;
            if(request.path!="/v1/chat/completions" && request.path!="/v1/completions" &&
                request.path!="/v1/messages" && request.path!="/v1/responses")return;
            refuse_unexecuted(body,false,false);
            for(const auto& [key,value]:body.as_object()) {
                (void)value;
                bool known=false;
                for(const auto* name:{"model","messages","tools","tool_choice","stream","stream_options","max_tokens",
                    "max_completion_tokens","temperature","top_p","top_k","seed","enable_thinking","preserve_thinking",
                    "conversational_mode","thinking_budget","chat_template_kwargs","dflash","n","stop","state_cache"})known|=key==name;
                if(request.path=="/v1/completions")for(const auto* name:{"prompt","echo","best_of","suffix"})known|=key==name;
                if(request.path=="/v1/messages")for(const auto* name:{"system","thinking","stop_sequences","output_config"})known|=key==name;
                if(request.path=="/v1/responses")for(const auto* name:{"input","instructions","max_output_tokens","store","background",
                    "previous_response_id","conversation","include","truncation","max_tool_calls","text"})known|=key==name;
                if(!known)throw RequestFieldError(key,"unsupported field: "+key);
            }
            if(body.find("max_tokens") && body.find("max_completion_tokens"))
                throw RequestFieldError("max_tokens","choose one max token field");
            for(const auto* name:{"max_tokens","max_completion_tokens","max_output_tokens","n"})if(const auto* v=body.find(name)) {
                const auto n=unsigned_integer(*v,name,UINT32_MAX);
                if(!n || (std::string(name)=="n" && n!=1))throw RequestFieldError(name,std::string("invalid ")+name);
            }
            if(const auto* t=body.find("thinking")) {
                if(!t->is_object())throw RequestFieldError("thinking","thinking must be an object");
                for(const auto& [key,value]:t->as_object()) {
                    if(key!="type")throw RequestFieldError("thinking."+key,"unsupported thinking field: "+key);
                    if(!value.is_string() || (value.as_string()!="enabled" && value.as_string()!="disabled"))
                        throw RequestFieldError("thinking.type","thinking.type must be enabled or disabled");
                }
            }
            if(const auto* stream=body.find("stream");stream && !stream->is_bool())throw RequestFieldError("stream","stream must be a boolean");
            if(request.path=="/v1/completions")for(const auto* key:{"enable_thinking","preserve_thinking","conversational_mode","thinking_budget","chat_template_kwargs","tools","tool_choice","messages"})
                if(body.find(key))throw RequestFieldError(key,std::string("unsupported completion field: ")+key);
            if(const auto* options=body.find("stream_options"))for(const auto& [key,value]:options->as_object())
                if(key!="include_usage" || !value.is_bool())throw RequestFieldError("stream_options."+key,"unsupported stream_options");
        };
        c.prepare_chat=[this](const Json& body) {
            // Shared protocol normalizers already own their envelopes. Only the
            // normalized family chat/template values enter the typed context.
            Json family=Json::object();
            for(const auto* name:{"messages","tools","tool_choice","enable_thinking","preserve_thinking",
                "conversational_mode","thinking_budget","chat_template_kwargs","dflash","max_tokens","max_completion_tokens"})
                if(const auto* v=body.find(name))family.set(name,*v);
            auto t=std::make_shared<Task>();t->request=parse_chat_request(family,mode,defaults);
            t->prepared=adapter.prepare(t->request.input);
            const auto& prepared=*t->prepared;
            ApiChatPrompt out;out.tokens=prepared.ids;out.thinking=t->request.input.enable_thinking;
            out.tools=t->request.input.tools;out.context=t;return out;
        };
        c.prepare_job=[this](const HttpRequest& request,const Json& body,GenerationJob& job) {
            auto t=std::dynamic_pointer_cast<Task>(job.family_context);
            const bool chat_context=bool(t);
            std::optional<gemma4::serve::PreparedPrompt> prepared;
            if(!t) {
                t=std::make_shared<Task>();t->request.generation=defaults;
                t->request.generation.dflash=mode;
                if(const auto* d=body.find("dflash")) {
                    const auto& value=d->as_string();
                    if(value!="on" && value!="off" && value!="auto")throw RequestFieldError("dflash","dflash must be on/off/auto");
                    t->request.generation.dflash=value=="on"?DraftMode::on:value=="off"?DraftMode::off:DraftMode::automatic;
                }
                prepared.emplace();prepared->ids=job.prompt;prepared->stop_ids={adapter.tokenizer().find("<eos>"),adapter.tokenizer().find("<turn|>")};
            } else {
                prepared=std::move(t->prepared);
            }
            t->request.generation.max_tokens=job.max_tokens;
            t->request.generation.sampling={float(job.sampling.temperature),float(job.sampling.top_p),job.sampling.top_k,job.sampling.seed,defaults.sampling.draws};
            t->request.generation.sampling.validate();
            if(t->request.generation.dflash==DraftMode::on && !drafts && dflash::enabled_from_environment() && job.sampling.temperature==0)
                throw RequestFieldError("dflash","DFlash on requires a drafter");
            t->client_gone=turns?job.client_gone:Cancelled{}; // preserve the opt-in cache probe boundary
            if(turns && chat_context && turn_config.request) {
                try {t->turn=turn_config.request(request,t->request);} catch(...) {t->turn.reset();}
            }
            if(t->turn && job.cache_retention!=CacheRetention::no_store)job.cache_retention=t->turn->retention;
            // A host stop string may end presentation before a queued phase is
            // observed; never retain that partial presentation as a full turn.
            if(!chat_context || body.find("stop") || body.find("stop_sequences"))job.cache_retention=CacheRetention::no_store;
            auto output=t->output;
            t->chat=std::make_unique<ChatJob>(adapter,t->request,[output](const gemma4::serve::OutputDelta& delta) {
                output->publish({delta.content,delta.reasoning,delta.tool_calls.size()?gemma4::serve::shared_calls(delta.tool_calls):Json()});return true;
            },std::move(prepared));
            output->consume_tokens=[this, context=std::weak_ptr<Task>(t)](std::span<const std::uint32_t>) {
                const auto task=context.lock();
                return task && task->chat->absorb(session_engine->result(task->handle));
            };
            t->phase=t->chat->phase_request();job.prompt=t->phase.prompt;job.stop_tokens=t->phase.stop_ids;
            job.family_context=t;job.output=output;
        };
        c.engine_routes=[this](const HttpRequest& request,HttpResponseWriter& writer) {
            if(request.method!="POST" || request.path!="/v1/gemma4/features")return false;
            try {
            QueueExternalWork work(queue);
            if(!work)throw ApiRouteError(503,"the server is draining or shutting down");
            const auto parsed=features::parse_request(parse_json_unique(request.body),&adapter);
            features::Result result;
            {std::lock_guard lock(backend_mutex);
                if(!feature_exporter)feature_exporter=std::make_unique<features::Exporter>(backend,features::backend_identity(backend));
                result=feature_exporter->export_features(parsed);}
            writer.send(200,"application/json",result.json().dump());return true;
            } catch(const JsonError& error) {throw ApiRouteError(400,error.what());}
              catch(const std::invalid_argument& error) {throw ApiRouteError(400,error.what());}
        };
        return c;
    }

};
Api::Api(Backend& b, std::shared_ptr<const gemma4::serve::Tokenizer> t, DraftFactory d, DraftMode m, std::string name, Request defaults,
    strix::serve::DeploymentCallbacks deployment, Owner owner, TurnCacheConfig turn_cache, std::size_t slots)
    : impl_(std::make_unique<Impl>(b,std::move(t),std::move(d),m,std::move(name),std::move(defaults),std::move(deployment),owner,std::move(turn_cache),slots)) {}
Api::~Api() = default;
strix::serve::SessionRuntime& Api::runtime(){return *impl_->session_runtime;}
void Api::handle(const HttpRequest& r, HttpResponseWriter& w) { impl_->frontend->handle(r,w); }
} // namespace strix::gemma4::engine
