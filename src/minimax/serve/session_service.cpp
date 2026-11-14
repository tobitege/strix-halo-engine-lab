#include "session_service.hpp"
#include "serve_runtime.hpp"
#include "minimax/model_graph.hpp"
#include "strix/minimax/model_description.hpp"
#include "minimax/engine/mm_defaults.hpp"
#include "strix/serve/session_api.hpp"
#include "strix/session/speculation_default.hpp"
#include <charconv>
#include <filesystem>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <sstream>

namespace strix::minimax::serve {
namespace mm = strix::minimax::engine;
namespace sv = strix::serve;
namespace ms = strix::minimax::serve;
namespace ds = strix::minimax::dspark;
using lead::Shared;
using lead::Policy;
using lead::Counters;
using lead::Extra;
using lead::Sched;
namespace {
std::uint64_t number(const std::string& s, std::uint64_t max, const char* name) {
    std::uint64_t n = 0;
    const auto r = std::from_chars(s.data(), s.data()+s.size(), n);
    if (r.ec != std::errc{} || r.ptr != s.data()+s.size() || n > max)
        throw std::invalid_argument(std::string(name) + ": invalid integer");
    return n;
}
// Retain the native session codec's arguments in order. Its execution parser
// runs before transport construction; host preparation validates common inputs.
struct SessionInputs { std::string save, load, prompt; };
bool session_option(ServePreparation& p, SessionInputs& o, const std::string& a,
                    const std::function<std::string()>& next) {
    if (a == "--baseline" || a == "--session-dspark") { p.session_arguments.push_back(a); return true; }
    const std::vector<std::string> values{
        "--session-cache", "--session-disk", "--session-ram-bytes", "--session-max-parked",
        "--session-dir", "--session-disk-bytes", "--session-host-budget", "--session-cache-dir",
        "--session-id", "--session-save", "--session-load", "--session-prompt", "--session-suffix",
        "--session-control-output", "--session-output", "--session-generate", "--session-suffix-generate"};
    if (std::find(values.begin(), values.end(), a) == values.end()) return false;
    auto v = next();
    if (a == "--session-save") o.save = v;
    if (a == "--session-load") o.load = v;
    if (a == "--session-prompt") o.prompt = v;
    if (a == "--session-host-budget") (void)number(v, UINT64_MAX, a.c_str());
    if (a == "--session-generate" || a == "--session-suffix-generate") (void)number(v, UINT32_MAX, a.c_str());
    p.session_arguments.insert(p.session_arguments.end(), {a, v});
    return true;
}
sv::Json policy_json(Shared& shared, bool drafter) {
    std::lock_guard<std::mutex> l(shared.m);
    const Policy& p = shared.policy;
    const Counters& c = shared.counters;
    sv::Json b = sv::Json::object();
    b.set("prefix_cache", sv::Json(shared.prefix_reuse));
    b.set("sched", sv::Json(p.sched == Sched::interleave ? "interleave" : "off"));
    b.set("effective_sched", sv::Json(p.sched == Sched::off ? "off" : shared.unlimited_reason.empty() ? "interleave" : "unlimited"));
    b.set("nest", sv::Json(p.nest));
    b.set("effective_nest", sv::Json(p.nest && p.sched == Sched::interleave && shared.nest_rows > 0 && shared.unlimited_reason.empty()));
    b.set("nest_reason", sv::Json(!p.nest ? "disabled" : p.sched == Sched::off ? "sched_off" :
        !shared.unlimited_reason.empty() ? shared.unlimited_reason : shared.nest_reason));
    b.set("nest_rows", sv::Json(static_cast<std::int64_t>(shared.nest_rows)));
    b.set("nest_poll_ms", sv::Json(std::int64_t{0}));  // Phase A gap-budget units; no extra arrival cap
    b.set("nest_max_per_parent", sv::Json(std::int64_t{2}));
    b.set("nest_bank_bytes", sv::Json(static_cast<std::int64_t>((shared.nest_rows == 0 ? 0U : std::max(8U, shared.nest_rows)) * 6144ULL * 8)));
    b.set("nest_tap_bytes", sv::Json(static_cast<std::int64_t>(shared.nest_taps && shared.nest_rows != 0 ? std::max(8U, shared.nest_rows) * 6144ULL * 20 : 0)));
    b.set("active_task", sv::Json(static_cast<std::int64_t>(shared.active_id)));
    b.set("parked_task", sv::Json(static_cast<std::int64_t>(shared.parked_id)));
    b.set("nest_qualification", sv::Json("pending extended resume-check"));
    b.set("service_ms", sv::Json(p.service_ms));
    b.set("short_first", sv::Json(p.sched == Sched::interleave && p.short_first));
    b.set("fallback_reason", sv::Json(p.sched == Sched::off ? "sched_off" : shared.unlimited_reason));
    b.set("max_unit_layers", sv::Json(static_cast<std::int64_t>(c.max_unit_layers)));
    b.set("qualification", sv::Json("pending"));
    b.set("gap_ms", sv::Json(p.gap_ms));
    b.set("dspark", sv::Json(drafter && (p.dspark_mode == ds::Mode::on || shared.dspark_contexts != 0)));
    b.set("dspark_mode", sv::Json(ds::mode_name(p.dspark_mode)));
    b.set("lookup", sv::Json(p.lookup));
    b.set("dspark_effective", sv::Json(p.dspark_mode == ds::Mode::on && drafter ? "on" : p.dspark_mode == ds::Mode::automatic && p.dspark_prior == ds::Prior::structural_v1 && drafter ? "per_request" : "plain"));
    b.set("dspark_assets", sv::Json(drafter));
    b.set("dspark_live", sv::Json(static_cast<std::int64_t>(shared.dspark_live)));
    b.set("dspark_contexts", sv::Json(static_cast<std::int64_t>(shared.dspark_contexts)));
    b.set("dspark_concurrency_eligible", sv::Json(shared.dspark_live == 1 && shared.active_id == 0 && shared.parked_id == 0));
    b.set("dspark_prior", sv::Json(ds::prior_name(p.dspark_prior)));
    b.set("dspark_reason", sv::Json(p.dspark_mode == ds::Mode::off ? "plain:off" : !drafter ? "plain:assets_missing" :
                                     p.dspark_mode == ds::Mode::automatic ? (p.dspark_prior == ds::Prior::unqualified_v1 ? "plain:prior_unqualified" : "auto:per_request") : "legacy"));
    b.set("dspark_inflight", sv::Json(static_cast<std::int64_t>(p.inflight)));
    b.set("draft_slots", sv::Json(static_cast<std::int64_t>(p.draft_slots)));
    b.set("epoch", sv::Json(static_cast<std::int64_t>(p.epoch)));
    sv::Json k = sv::Json::object();
    k.set("step_s", sv::Json(c.step_s));
    k.set("prefill_s", sv::Json(c.prefill_s));
    k.set("draft_s", sv::Json(c.draft_s));
    k.set("idle_s", sv::Json(c.idle_s));
    k.set("stall_s", sv::Json(c.stall_s));
    k.set("steps", sv::Json(static_cast<std::int64_t>(c.steps)));
    k.set("step_rows", sv::Json(static_cast<std::int64_t>(c.step_rows)));
    k.set("units", sv::Json(static_cast<std::int64_t>(c.units)));
    k.set("prefills", sv::Json(static_cast<std::int64_t>(c.prefills)));
    k.set("cancels", sv::Json(static_cast<std::int64_t>(c.cancels)));
    k.set("multi_windows", sv::Json(static_cast<std::int64_t>(c.multi_windows)));
    k.set("unequal_windows", sv::Json(static_cast<std::int64_t>(c.unequal_windows)));
    k.set("rejections", sv::Json(static_cast<std::int64_t>(c.rejections)));
    k.set("draft_cycles", sv::Json(static_cast<std::int64_t>(c.draft_cycles)));
    k.set("plain_cycles", sv::Json(static_cast<std::int64_t>(c.plain_cycles)));
    k.set("drafts", sv::Json(static_cast<std::int64_t>(c.drafts)));
    k.set("drafted", sv::Json(static_cast<std::int64_t>(c.drafted)));
    k.set("accepted", sv::Json(static_cast<std::int64_t>(c.accepted)));
    k.set("lookup_windows", sv::Json(static_cast<std::int64_t>(c.lookup_windows)));
    k.set("lookup_drafted", sv::Json(static_cast<std::int64_t>(c.lookup_drafted)));
    k.set("lookup_accepted", sv::Json(static_cast<std::int64_t>(c.lookup_accepted)));
    k.set("lookup_s", sv::Json(c.lookup_s));
    sv::Json h = sv::Json::array();
    for (auto v : c.rows_hist) h.as_array().push_back(sv::Json(static_cast<std::int64_t>(v)));
    k.set("rows_hist", std::move(h));
    b.set("counters", std::move(k));
    return b;
}

// The mm_* value lines of GET /metrics: the queue's (from the snapshot the observer renders too) and the scheduler's
// counters (the observer adds their HELP/TYPE).
std::string mm_metric_lines(const sv::QueueSnapshot& s, Shared& shared) {
    Counters c;
    {
        std::lock_guard<std::mutex> l(shared.m);
        c = shared.counters;
    }
    char buf[1024];
    std::snprintf(buf, sizeof buf,
                  "mm_queued %llu\nmm_running %llu\nmm_completed %llu\nmm_failed %llu\nmm_prompt_tokens %llu\n"
                  "mm_generated_tokens %llu\nmm_busy_seconds %.3f\nmm_step_seconds %.3f\nmm_prefill_seconds %.3f\n"
                  "mm_draft_seconds %.3f\nmm_idle_seconds %.3f\nmm_decode_stall_seconds %.3f\nmm_steps %llu\nmm_step_rows %llu\n"
                  "mm_prefill_units %llu\nmm_draft_cycles %llu\nmm_drafted %llu\nmm_accepted %llu\n",
                  static_cast<unsigned long long>(s.queued), static_cast<unsigned long long>(s.running),
                  static_cast<unsigned long long>(s.completed), static_cast<unsigned long long>(s.failed),
                  static_cast<unsigned long long>(s.prompt_tokens), static_cast<unsigned long long>(s.generated_tokens),
                  s.busy_seconds, c.step_s, c.prefill_s, c.draft_s, c.idle_s, c.stall_s, static_cast<unsigned long long>(c.steps),
                  static_cast<unsigned long long>(c.step_rows), static_cast<unsigned long long>(c.units),
                  static_cast<unsigned long long>(c.draft_cycles), static_cast<unsigned long long>(c.drafted),
                  static_cast<unsigned long long>(c.accepted));
    return buf;
}

// GET / POST /admin/sched (with --admin): the policy and counters; POST changes them (a bad value throws: a 400).
sv::Json admin_sched(const sv::HttpRequest& req, Shared& shared, bool drafter) {
    if (req.method == "POST") {
        const sv::Json body = sv::parse_request_json(req.body);
        std::lock_guard<std::mutex> l(shared.m);
        Policy p = shared.policy;
        if (const auto* v = body.find("sched"); v && v->is_string()) {
            if (v->as_string() == "off") p.sched = Sched::off;
            else if (v->as_string() == "interleave") p.sched = Sched::interleave;
            else throw std::invalid_argument("sched: off | interleave");
        }
        if (const auto* v = body.find("gap_ms"); v && v->is_number()) p.gap_ms = ms::gap_ms(v->dump().c_str());
        if (const auto* v = body.find("service_ms"); v && v->is_number()) p.service_ms = ms::service_ms(v->dump().c_str());
        if (const auto* v = body.find("nest")) {
            if (!v->is_bool()) throw std::invalid_argument("nest: boolean required");
            p.nest = v->as_bool();
        }
        if (const auto* v = body.find("short_first"); v && v->is_bool()) p.short_first = v->as_bool();
        p.dspark_mode = ms::dspark_mode_update(p.dspark_mode, body);
        if (const auto* v = body.find("dspark_inflight")) {
            if (!v->is_number()) throw std::invalid_argument("dspark_inflight: 0..8");
            p.inflight = ds::parse_inflight(v->dump().c_str());
        }
        if (const auto* v = body.find("lookup")) {
            if (!v->is_bool()) throw std::invalid_argument("lookup: boolean required");
            p.lookup = v->as_bool();
        }
        if (const auto* v = body.find("draft_slots"); v && v->is_number()) p.draft_slots = static_cast<std::uint32_t>(std::clamp<std::int64_t>(v->as_int(), 1, 8));
        if (body.find("reset_counters")) shared.counters = Counters{};
        ++p.epoch;
        shared.policy = p;
    }
    return policy_json(shared, drafter);
}

} // namespace
ServePreparation prepare_serve(const std::vector<std::string>& arguments) {
    ServePreparation p;
    std::vector<std::string> args{"strix-minimax-serve"};
    args.insert(args.end(), arguments.begin(), arguments.end());
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    const auto argc = static_cast<int>(argv.size());
    // Preserve resolver precedence and its pre-device effective-config exit.
    if (defaults::print_or_apply("serve", argc, argv.data())) { p.exit_code = 0; return p; }
    if (arguments.empty() || arguments[0] == "--help" || arguments[0] == "-h") {
        p.output = "usage: strix-minimax-serve MODEL_DIR --rank R --transport T [--port P] [--bind ADDR] [--slots S] [--capacity C] [--max-rows R] [--model-id ID] [--max-tokens N] [--dspark-mode off|auto|on] [--dspark DIR] [--admin] [--resume-check P.u32] [--print-model-graph] [--print-plan] [--deploy FILE|auto] [--deploy-check]\n";
        p.exit_code = arguments.empty() ? 2 : 0; return p;
    }
    SessionInputs session_options;
    auto& opt = p.opt;
    opt.model_dir = argv[1]; opt.slots = 8;
    std::string deploy_file;
    bool deploy_check = false, slots_given = false;
    auto& transport_spec=p.transport_spec; auto& bind=p.bind; auto& model_id=p.model_id;
    auto& dspark_dir=p.dspark_dir; auto& dspark_exl3=p.dspark_exl3; auto& check=p.check; auto& check_nested=p.check_nested;
    auto& port=p.port; auto& max_tokens=p.max_tokens; auto& check_tokens=p.check_tokens; auto& check_child_layers=p.check_child_layers;
    auto& admin=p.admin; auto& check_only=p.check_only;
        std::string dspark_mode_value, dspark_prior_value;
        bool no_mtp = false;
        bool print_graph = strix::minimax::graph::requested("STRIX_PRINT_MODEL_GRAPH");
        bool print_plan = strix::minimax::graph::requested("STRIX_PRINT_EXEC_PLAN");
        const auto dspark_inflight = ds::parse_inflight(std::getenv("STRIX_MM_SERVE_DSPARK_INFLIGHT"));
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
                return argv[++i];
            };
            if (a == "--print-model-graph") print_graph = true;
            else if (a == "--deploy") deploy_file = next();
            else if (a == "--deploy-check") deploy_check = true;
            else if (a == "--print-plan") print_plan = true;
            else if (a == "--rank") opt.rank = static_cast<std::uint32_t>(number(next(), UINT32_MAX, a.c_str()));
            else if (a == "--transport") transport_spec = next();
            else if (a == "--port") { port = static_cast<std::uint16_t>(number(next(), 65535, "--port")); p.port_given = true; }
            else if (a == "--bind") { bind = next(); p.bind_given = true; }
            else if (a == "--slots") { opt.slots = static_cast<std::uint32_t>(number(next(), UINT32_MAX, a.c_str())); slots_given = true; }
            else if (a == "--capacity") opt.capacity = number(next(), UINT64_MAX, a.c_str());
            else if (a == "--max-rows") opt.max_rows = static_cast<std::uint32_t>(number(next(), UINT32_MAX, a.c_str()));
            else if (a == "--model-id") model_id = next();
            else if (a == "--max-tokens") max_tokens = static_cast<std::uint32_t>(number(next(), UINT32_MAX, a.c_str()));
            else if (a == "--dspark") dspark_dir = next();
            else if (a == "--dspark-prior") dspark_prior_value = next();
            else if (a == "--dspark-mode") dspark_mode_value = next();
            else if (strix::session::speculation_opt_out_argument(a)) no_mtp = true;
            else if (a == "--dspark-exl3") dspark_exl3 = next();
            else if (a == "--admin") admin = true;
            else if (a == "--resume-check-child-layers") check_child_layers = static_cast<unsigned>(number(next(), UINT32_MAX, a.c_str()));
            else if (a == "--resume-check-nested") {
                check_nested = next();
                if (check_nested.empty()) throw std::invalid_argument("--resume-check-nested requires a fixture");
            }
            else if (a == "--resume-check") check = next();
            else if (a == "--resume-check-tokens") check_tokens = static_cast<std::uint32_t>(number(next(), UINT32_MAX, a.c_str()));
            else if (a == "--check-only") check_only = true;
            else if (session_option(p, session_options, a, next)) {}
            else throw std::invalid_argument("unknown option " + a);
        }
        if (no_mtp && dspark_mode_value.empty()) dspark_mode_value = "off";  // shared opt-out; --dspark-mode wins
        if (opt.rank > 1) throw std::invalid_argument("--rank: 0 or 1");
        if (opt.capacity == 0 || opt.max_rows == 0) throw std::invalid_argument("capacity and max-rows must be positive");
        const auto dspark_prior = ds::parse_prior(dspark_prior_value.empty() ? std::getenv("STRIX_MM_DSPARK_PRIOR") : dspark_prior_value.c_str());
        const auto dspark_mode = ds::parse_mode(dspark_mode_value.empty() ? std::getenv("STRIX_MM_SERVE_DSPARK") : dspark_mode_value.c_str());
        if (check_child_layers == 0 || (check_child_layers != ~0U && check_child_layers >= 60))
            throw std::invalid_argument("--resume-check-child-layers: 1..59");
        if ((!check_nested.empty() || check_child_layers != ~0U) && check.empty())
            throw std::invalid_argument("nested qualification requires --resume-check");
        if (check_child_layers != ~0U && check_nested.empty())
            throw std::invalid_argument("bounded child requires --resume-check-nested");
        if (check_child_layers != ~0U && check_tokens < 2 + 59 / check_child_layers)
            throw std::invalid_argument("bounded child requires enough continuation tokens for every background step");
        if(!session_options.save.empty() && !session_options.load.empty()) throw std::invalid_argument("choose session save or load");
        if (opt.slots == 0 || opt.slots > 8) throw std::invalid_argument("--slots 1..8");
        if (!dspark_exl3.empty() && dspark_dir.empty()) dspark_dir = opt.model_dir + "/dspark";
        if (dspark_dir.empty() && dspark_mode != ds::Mode::off && std::filesystem::is_directory(opt.model_dir + "/dspark"))
            for (const auto& entry : std::filesystem::directory_iterator(opt.model_dir + "/dspark"))
                if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                    dspark_dir = opt.model_dir + "/dspark";
                    break;
                }
        const strix::minimax::graph::Attachments graph_attachments{!dspark_dir.empty(), !dspark_exl3.empty(), 0};
        if (strix::minimax::graph::inspect_or_check(opt.model_dir, opt.rank, graph_attachments, print_graph, print_plan)) { p.exit_code = 0; return p; }
        auto& deploy_spec = p.deploy_spec;
        auto& deploy_provider = p.deploy_provider;
        if (deploy_check && deploy_file.empty()) deploy_file = "auto";
        if (!deploy_file.empty()) {
            auto spec = strix::deploy::load_spec(deploy_file);
            if (opt.capacity > UINT32_MAX) throw std::invalid_argument("deployment capacity exceeds uint32 range");
            if (!strix::deploy::pinned(spec,"context")) spec.context = static_cast<std::uint32_t>(opt.capacity);
            if (!strix::deploy::pinned(spec,"slots") && slots_given) spec.slots = opt.slots;
            if (!spec.nodes) spec.nodes = 2;
            const auto link = transport_spec.starts_with("tcp") ? "tcp" : transport_spec;
            if (spec.link == "auto") spec.link = link;
            else if (spec.link != link) throw std::invalid_argument("deployment link differs from --transport");
            if (spec.profile != "auto" && spec.profile != "default")
                throw std::invalid_argument("MiniMax deployment profile pin unavailable in this build");
            if (!dspark_dir.empty()) throw std::invalid_argument("deployment DSpark peak sizing unavailable; disable DSpark");
            const auto* session_cache = std::getenv("STRIX_SESSION_CACHE");
            if (!session_cache || (std::string_view(session_cache) != "0" && std::string_view(session_cache) != "off") ||
                !session_options.load.empty() || !session_options.save.empty() || !session_options.prompt.empty())
                throw std::invalid_argument("deployment session-cache peak sizing unavailable; use --session-cache off and no session restore/save");
            std::uint64_t scratch_extra = 0;
            if (const auto* cap = std::getenv("STRIX_MM_PF_MSA_SCRATCH_MB")) {
                std::size_t end = 0; const auto mb = std::stoull(cap,&end);
                if (end != std::string_view(cap).size() || mb < 64 || mb > 8192)
                    throw std::invalid_argument("deployment invalid MSA scratch cap");
                if (mb > 1152) scratch_extra = (mb-1152)<<20;
            }
            deploy_provider = std::make_shared<mm::FootprintProvider>(mm::deployment_model(opt.model_dir),opt.max_rows,model_id,scratch_extra);
            if (deploy_check) {
                // Pair inspection needs the orchestrator's measured peer entry; never invented.
                auto plan = strix::deploy::resolve(spec,
                    strix::deploy::inspection_devices(spec,opt.rank,strix::deploy::probe_device()),*deploy_provider);
                p.output = plan.to_json() + '\n'; p.exit_code = plan.refused ? 2 : 0; return p;
            }
            deploy_spec = std::move(spec);
        }
    p.dspark_mode=dspark_mode; p.dspark_prior=dspark_prior; p.dspark_inflight=dspark_inflight;
    return p;
}

namespace {
const ServePreparation& prepared(const sv::ServiceSelection& s) {
    const auto p=std::dynamic_pointer_cast<const ServePreparation>(s.preparation);
    if (!p) throw std::invalid_argument("MiniMax service has not passed validation");
    return *p;
}
class Service final : public sv::SessionService {
    ServePreparation settings_;
    std::unique_ptr<ServiceAssets> assets_;
    std::optional<int> startup_;
    std::unique_ptr<Observer> observer_;
    std::unique_ptr<MinimaxApi> api_;
public:
    Service(const ServePreparation& p, std::unique_ptr<ServiceAssets> assets)
        : settings_(p), assets_(std::move(assets)) {
        if (!assets_) throw std::invalid_argument("incomplete MiniMax service assets");
        try {
            // Follower and qualification-only lifecycles never construct HTTP.
            startup_=assets_->startup();
            if (p.opt.rank != 0 || startup_) return;
            if (!assets_->tokenizer.encode || !assets_->tokenizer.decode || !assets_->tokenizer.stream_bytes)
                throw std::invalid_argument("incomplete MiniMax tokenizer assets");
            observer_=std::make_unique<Observer>(assets_->info);
            MinimaxApiOptions ao;
            ao.http_slots=assets_->info.slots; ao.model_id=p.model_id;
            ao.default_max_tokens=p.max_tokens; ao.context_tokens=assets_->info.context_tokens;
            ao.session_enabled=assets_->session_enabled; ao.deployment=assets_->deployment;
            ao.deployment.capacity_provider=[this] {
                const auto q=assets_->queue.snapshot();
                return sv::ServeCapacity{assets_->info.slots,q.running,std::nullopt,q.queued};
            };
            ao.take_extra=[this](const sv::GenerationTicket& t)->std::optional<Extra> {
                auto& shared=assets_->shared; std::lock_guard<std::mutex> lock(shared.m);
                const auto it=shared.extras.find(&t); if(it==shared.extras.end())return {};
                Extra x=it->second; shared.extras.erase(it); return x;
            };
            ao.legacy_metrics=[this](const sv::QueueSnapshot& q){return mm_metric_lines(q,assets_->shared);};
            if(p.admin)ao.admin_sched=[this](const sv::HttpRequest& r){return admin_sched(r,assets_->shared,assets_->info.draft);};
            api_=std::make_unique<MinimaxApi>(assets_->tokenizer,assets_->queue,std::move(ao),*observer_);
        } catch (...) { assets_->fail(); throw; }
    }
    ~Service() override { assets_->close(); }
    sv::ServiceRole role() const noexcept override { return settings_.opt.rank ? sv::ServiceRole::follower : sv::ServiceRole::leader; }
    std::optional<int> startup() override { return startup_; }
    int follow() override { return assets_->follow(); }
    sv::SessionRuntime& runtime() override { return assets_->runtime(); }
    void failed() noexcept override { assets_->fail(); }
    void shutdown() noexcept override { assets_->close(); }
    void configure_listener(sv::HttpServerConfig& config) const override {
        if(settings_.bind_given)config.bind_address=settings_.bind;
        if(settings_.port_given)config.port=settings_.port;
    }
    void listening(const sv::HttpServerConfig& config, std::uint16_t port) override {
        auto& shared=assets_->shared;
        std::printf("minimax serve policy: %s\n",policy_json(shared,assets_->info.draft).dump().c_str());
        if (shared.prefix_reuse) std::printf("minimax serve: prefix reuse on (STRIX_MM_SERVE_PREFIX=1 on both ranks)%s\n",
            shared.conversation ? "; conversation cache on (STRIX_MM_SERVE_CONVERSATION=1 on both ranks)" : "");
        std::printf("READY rank 0 http://%s:%u model %s slots %u context %llu sched %s gap %.0f ms dspark %s\n",
            config.bind_address.c_str(),static_cast<unsigned>(port),settings_.model_id.c_str(),static_cast<unsigned>(assets_->info.slots),
            static_cast<unsigned long long>(assets_->info.context_tokens),
            shared.policy.sched==Sched::interleave ? "interleave" : "off", shared.policy.gap_ms,ds::mode_name(shared.policy.dspark_mode));
        std::fflush(stdout);
    }
    void stopped() override { std::printf("rank 0: STOPPED\n"); }
    void handle(const sv::HttpRequest& r, sv::HttpResponseWriter& w) override {
        if(assets_->session_enabled && (r.path.starts_with("/v1/sessions") || r.path=="/v1/cache/stats")) {
            try {
                const auto command=sv::session_command(r.method,r.path,r.body);
                auto reply=assets_->sessions.tasks.submit(command);
                int status=200;
                if(const auto* v=reply.find("_status"))status=static_cast<int>(v->as_int());
                if(const auto* error=reply.find("error")) {
                    if(status==200 || status==503)w.add_response_header("Retry-After","1");
                    w.send(status==200 ? 503 : status,"application/json",sv::session_error(error->find("code")->as_string(),error->find("message")->as_string()).dump());
                } else w.send(status,"application/json",reply.dump());
            } catch(const sv::SessionApiError& e) {
                if(e.status==503)w.add_response_header("Retry-After","1");
                w.send(e.status,"application/json",sv::session_error(e.code,e.what()).dump());
            } catch(const std::exception& e) {w.send(400,"application/json",sv::session_error("invalid_request",e.what()).dump());}
            return;
        }
        if(!api_)throw std::logic_error("MiniMax lifecycle has no HTTP handler");
        api_->handle(r,w);
    }
};
}
void install_session_service(sv::ServiceFactories& factories) { install_session_service(factories,{}); }
void install_session_service(sv::ServiceFactories& factories, SessionAssetLoader loader) {
    if(!loader)loader=load_native_service;
    sv::ServiceFactory f;
    f.validate=[](const sv::ServiceSelection& s) {
        description::validate(s.description);
        s.preparation=std::make_shared<ServePreparation>(prepare_serve(s.arguments));
    };
    f.early_exit=[](const sv::ServiceSelection& s,std::ostream& out)->std::optional<int> {
        const auto& p=prepared(s); if(p.exit_code) {out<<p.output;return p.exit_code;}return {};
    };
    f.inspect=[](const sv::ServiceSelection& s,std::ostream& out) {
        const auto& p=prepared(s);
        defaults::Cli cli;cli.app="serve";cli.transport=p.transport_spec;cli.slots=p.opt.slots;cli.capacity=p.opt.capacity;cli.max_rows=p.opt.max_rows;
        out<<defaults::effective_config(cli,[](const char* k){return std::getenv(k);});
    };
    f.create=[loader](const sv::ServiceSelection& s) {const auto& p=prepared(s);return std::make_unique<Service>(p,loader(p));};
    factories.install("minimax",std::move(f));
}
int legacy_serve(int argc,char** argv) {
    // Preserve MiniMax's family --model-id spelling behind the shell delimiter.
#ifdef STRIX_SERVE_REGISTRY_ROOT
    const std::string registry_root=STRIX_SERVE_REGISTRY_ROOT;
#else
    const std::string registry_root=".";
#endif
    std::vector<std::string> args{argc ? argv[0] : "strix-minimax-serve","--registry-root",registry_root,"--model","minimax","--"};
    for(int i=1;i<argc;++i)args.emplace_back(argv[i]);
    std::vector<char*> pointers;for(auto& a:args)pointers.push_back(a.data());
    return sv::run(static_cast<int>(pointers.size()),pointers.data());
}
} // namespace strix::minimax::serve
