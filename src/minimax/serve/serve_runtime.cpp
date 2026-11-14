#include "strix/route_receipt.hpp"
#include "serve_runtime.hpp"
#include <stdexcept>
#ifdef STRIX_MINIMAX_NATIVE
#include "serve_runner.hpp"
#include "minimax/serve/minimax_tokenizer.hpp"
#include "strix/dist/open_transport.hpp"
#include "xport/borrowed_transport.hpp"
#include "minimax/xfer/minimax_xfer.hpp"
#include "strix/minimax/session_startup.hpp"
#include <csignal>
#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace strix::minimax::serve {
namespace {
namespace mm = strix::minimax::engine;
namespace sv = strix::serve;
namespace ms = strix::minimax::serve;
namespace ds = strix::minimax::dspark;
using native::Runner;
using native::env_num;
using lead::Sched;
using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b-a).count(); }
// Signal storage outlives every Lead and transport; no monitor thread.
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }
void default_env(const char* name, const char* value) { (void)::setenv(name,value,0); }
std::unique_ptr<strix::dist::Transport> open_transport(const std::string& spec, std::uint32_t rank, std::string& receipt,
    std::unique_ptr<strix::dist::DeviceExchange>& gpu_direct) {
    // transport_capabilities 0 preserves the serving capability contract.
    auto opened = strix::dist::open_pair(spec, rank, {std::chrono::seconds(300), 0});
    gpu_direct = std::move(opened.direct);
    receipt = std::move(opened.receipt);
    return std::move(opened.host);
}

// The transport shared in turn by the communicator and the exchange module (as strix-minimax-rank).
using Borrowed = strix::xport::BorrowedTransport<
    strix::xport::BorrowedAbort::forward, strix::xport::BorrowedState::base_default>;


class NativeAssets final : public ServiceAssets {
    // Reverse destruction: Lead -> sessions -> Runner/drafter -> engine ->
    // communicator -> exchanger -> host transport. Base queue/Shared live longer.
    ServePreparation settings;
    std::unique_ptr<strix::dist::DeviceExchange> gpu_direct;
    std::unique_ptr<strix::dist::Transport> transport;
    std::unique_ptr<strix::minimax::xfer::Exchanger> exchanger;
    std::unique_ptr<strix::dist::Communicator> comm_owner;
    std::unique_ptr<mm::Engine> engine_owner;
    std::unique_ptr<ds::DraftFile> draft_file;
    std::unique_ptr<ds::DeviceDrafter> drafter;
    std::unique_ptr<Runner> runner_owner;
    std::unique_ptr<strix::minimax::session::Runtime> session_runtime;
    ms::SessionCounts session_counts;
    std::unique_ptr<lead::Lead<mm::Engine,Runner>> lead_owner;
    std::optional<int> exit_code;
public:
    explicit NativeAssets(const ServePreparation& p) : settings(p) {
        try {
        mm::Options opt{p.opt.model_dir,p.opt.rank,p.opt.slots,p.opt.capacity,p.opt.max_rows};
        const auto& transport_spec=p.transport_spec;
        const auto& dspark_dir=p.dspark_dir; const auto& dspark_exl3=p.dspark_exl3;
        const auto& model_id=p.model_id; const auto& check=p.check; const auto& check_nested=p.check_nested;
        const auto check_tokens=p.check_tokens, check_child_layers=p.check_child_layers;
        const auto check_only=p.check_only;
        const auto dspark_mode=p.dspark_mode; const auto dspark_prior=p.dspark_prior;
        const auto dspark_inflight=p.dspark_inflight;
        auto deploy_spec=p.deploy_spec; auto deploy_provider=p.deploy_provider;
        std::shared_ptr<const strix::deploy::Plan> deployment_plan;
        // Parse the retained native session CLI before constructing transport/device.
        strix::minimax::session::RunOptions session_options;
        for(std::size_t i=0;i<p.session_arguments.size();++i) {
            const auto& a=p.session_arguments[i];
            if(!strix::minimax::session::parse_option(session_options,a,[&] {return p.session_arguments.at(++i);}))
                throw std::invalid_argument("unknown session option " + a);
        }
        const strix::minimax::defaults::Lookup lookup=[](const char* key){return std::getenv(key);};
        strix::minimax::defaults::Cli cli;cli.app="serve";cli.transport=transport_spec;
        cli.slots=opt.slots;cli.capacity=opt.capacity;cli.max_rows=opt.max_rows;
        strix::minimax::defaults::SessionConfig session_config;
        default_env("STRIX_DIST_USB4_TIMEOUT_MS","60000");
        default_env("STRIX_DIST_WAIT_TIMEOUT_MS","60000");
        default_env("STRIX_DIST_CRC_PARALLEL","1");
        default_env("STRIX_DIST_HOST_CRC_OVERLAP","1");
        default_env("STRIX_DIST_HOST_CHUNK_BYTES","1048576");
        default_env("STRIX_MMX_CRC","deferred");
        g_stop.store(false);
        std::signal(SIGINT,on_signal); std::signal(SIGTERM,on_signal);
        std::string receipt;
        transport = open_transport(transport_spec, opt.rank, receipt, gpu_direct);
        if (deploy_spec) {
            auto plan = strix::deploy::resolve_pair(*deploy_spec,*deploy_provider,opt.rank,strix::deploy::probe_device(),
                [&](auto out, auto in) { transport->exchange(out,in); });
            strix::deploy::require_plan(plan);
            opt.capacity = plan.chosen.context; opt.slots = plan.chosen.slots;
            cli.capacity = opt.capacity; cli.slots = opt.slots;
            deployment_plan = std::make_shared<const strix::deploy::Plan>(std::move(plan));
        }
        strix::dist::Group group;
        group.transport_capabilities = 0;
        group.rank = opt.rank;
        group.world = 2;
        group.plan = 0x4d4d5331ULL;  // "MMS1"
        if (deployment_plan) group.plan ^= std::stoull(deployment_plan->hash(),nullptr,16);
        const std::size_t max_bytes = std::max<std::size_t>(std::size_t{opt.max_rows} * 6144, 256 * 782 * 128) * sizeof(float);
        // exchange module for the decode steps (default on, STRIX_MM_XFER=off disables)
        const char* xfer_env = std::getenv("STRIX_MM_XFER");

        if (xfer_env == nullptr || std::string(xfer_env) != "off") {
            strix::minimax::xfer::Options xo;
            xo.rank = opt.rank;
            xo.capacity = strix::minimax::xfer::MmSchedule::row_bytes(8);
            xo.stream = nullptr;
            xo.from_env();
            exchanger = std::make_unique<strix::minimax::xfer::Exchanger>(std::make_unique<Borrowed>(*transport), xo);
        }
        comm_owner = std::make_unique<strix::dist::Communicator>(group, std::make_unique<Borrowed>(*transport),
                                                                 std::move(gpu_direct), max_bytes);
        auto& comm = *comm_owner;
        cli.transport = strix::dist::negotiated_transport(transport_spec, comm.device_exchange() != nullptr,
                                                          strix::minimax::defaults::kRouteHost, strix::minimax::defaults::kRouteGpu);
        if (!receipt.empty()) std::printf("transport: %s\n", receipt.c_str());
        std::printf("connected: rank %u transport %s\n", opt.rank, comm.transport_name().c_str());
        std::fflush(stdout);
        engine_owner = std::make_unique<mm::Engine>(opt, comm);
        auto& eng = *engine_owner;
        if (exchanger) eng.set_exchanger(exchanger.get(), true);
        const auto lr = eng.load();
        const std::uint32_t dspark_config = 0x4d4d0400U | (static_cast<std::uint32_t>(dspark_prior) << 7) | (static_cast<std::uint32_t>(dspark_mode) << 5) |
                                            (dspark_dir.empty() ? 0U : 16U) | dspark_inflight;
        if (eng.peer(dspark_config) != dspark_config) {
            eng.abort("DSpark startup mode / assets / protocol mismatch");
            throw std::runtime_error("DSpark startup mode / assets / protocol mismatch");
        }
        // Shared route receipt: both ranks must have resolved the same numerical route before any product runs.
        try {
            const auto route_fp = strix::route::compare_ranks([&](const void* o, void* i, std::size_t n) { eng.exchange_bytes(o, i, n); });
            std::printf("rank %u: route fingerprint %016llx agreed with the peer\n", opt.rank, static_cast<unsigned long long>(route_fp));
        } catch (...) { eng.abort("route receipt: numerical-route fingerprint mismatch"); throw; }
        std::printf("rank %u: load %.1f s, device used %.1f GiB\n", opt.rank, lr.seconds, lr.device_gib);
        std::fflush(stdout);
        // DSpark drafter on both ranks (both run every draft command; the target's residual taps are replicated)
        runner_owner = std::make_unique<Runner>(eng, comm);
        auto& run = *runner_owner;
        if (!dspark_dir.empty()) {
            const auto t0 = Clock::now();
            draft_file = ds::DraftFile::open(dspark_dir, dspark_exl3.empty());
            ds::DeviceDrafter::Options dopt;
            dopt.slots = opt.slots;
            dopt.capacity = opt.capacity;
            if (const char* e = std::getenv("STRIX_MM_DSPARK_ATTN")) dopt.attn_variant = static_cast<std::uint32_t>(std::atoi(e));  // as the rank app
            drafter = std::make_unique<ds::DeviceDrafter>(draft_file->weights(), dopt);
            if (!dspark_exl3.empty()) eng.attach_draft_exl3(dspark_exl3, *drafter);
            eng.set_taps(true);  // for the drafter-context gate; then the startup policy below (both ranks)
            run.dd = drafter.get();
            std::tie(run.v0, run.vn) = eng.vocab_range();
            std::printf("rank %u: dspark %s%s loaded in %.1f s (%.2f GB device, rings %.0f MB, %u slots)\n", opt.rank, dspark_dir.c_str(),
                        dspark_exl3.empty() ? "" : (" + EXL3 " + dspark_exl3).c_str(), secs(t0, Clock::now()),
                        static_cast<double>(drafter->weight_bytes()) / 1e9, static_cast<double>(drafter->ring_bytes()) / 1e6, opt.slots);
            std::fflush(stdout);
        }
        (void)eng.reserve_nested(drafter != nullptr);
        // A protocol/capacity handshake before any check or READY, independent of rank-1 environment policy.
        const auto peer_capacity = eng.peer(eng.nested_rows());
        if (peer_capacity != eng.nested_rows()) {
            eng.abort("nested capacity mismatch");
            throw std::runtime_error("nested capacity mismatch");
        }
        const bool session_cli = !session_options.prompt.empty() || !session_options.load.empty();
        try { session_config = strix::minimax::defaults::session_policy(cli, lookup, session_cli); }
        catch (const std::exception& e) { session_config.enabled = false; session_config.reason = e.what(); }
        const char* session_switch = std::getenv("STRIX_SESSION_CACHE");
        const bool session_explicit = session_cli || (session_switch &&
            (std::string(session_switch) == "1" || std::string(session_switch) == "on"));
        bool prefix_agreed = false;
        const bool prefix_requested = session_config.reason != "baseline" && lead::prefix_cache_requested();
        bool conversation_agreed = false;
        const bool conversation_requested = prefix_requested && lead::conversation_cache_requested();
        const bool session_vote = ms::session_hello(eng, session_config.reason != "baseline" &&
            (session_config.enabled || session_explicit), prefix_requested, &prefix_agreed, eng.prefill_reuse_rows(),
            conversation_requested, &conversation_agreed);
        if (prefix_requested && !prefix_agreed)
            std::printf("minimax serve: prefix reuse off (peer vote or prefill alignment differs)\n");
        if (conversation_requested && !conversation_agreed)
            std::printf("minimax serve: conversation cache off (peer vote differs); chunk-aligned reuse stays\n");
        run.prefix_reuse = prefix_agreed;
        shared.prefix_reuse = prefix_agreed;
        shared.prefill_int4_regime = [&eng](std::uint64_t rows) { return eng.prefill_int4_regime(rows); };
        run.conversation = conversation_agreed;
        shared.conversation = conversation_agreed;
        eng.set_unaligned_resume(conversation_agreed);
        if (prefix_agreed) {
            const auto snapshot = eng.loader_snapshot();
            strix::verify_loader_snapshot(snapshot);
            const auto record = strix::minimax::session::compatibility(
                {{"world","2"},{"capacity",std::to_string(opt.capacity)},{"slots",std::to_string(opt.slots)},
                 {"max_rows",std::to_string(opt.max_rows)},{"producer","resident-prefill-v1"}}, snapshot, "/proc/self/exe");
            const auto digest = strix::minimax::session::sha(std::span(
                reinterpret_cast<const std::uint8_t*>(record.data()), record.size()));
            shared.prefix_identity = {digest, digest, "minimax-resident-prefill-tp2-v1", 2};
        }
        if (session_vote) {
            try {
                if (session_cli) {
                    session_config = strix::minimax::defaults::session_config(cli, lookup, UINT64_MAX, true,
                        !session_options.save.empty() ? session_options.save : !session_options.load.empty() ? session_options.load : session_options.directory);
                    if ((!session_options.save.empty() || !session_options.load.empty()) && !session_config.disk) {
                        session_config.enabled = false; session_config.reason = session_config.disk_reason;
                    }
                } else session_config = strix::minimax::defaults::session_config(cli, lookup);
            } catch (const std::exception& e) { session_config.enabled = false; session_config.reason = e.what(); }
            if (!session_options.budget) session_options.budget = session_config.ram_bytes;
            if (session_cli || session_options.directory.empty()) session_options.directory = session_config.directory;
            session_options.max_parked = session_config.max_parked;
            session_options.disk_budget = session_config.disk_bytes;
            session_options.disk_enabled = session_config.disk;
            const bool session_on = session_config.enabled;
            prepare_cache_owner(session_options.budget ? session_options.budget : UINT64_MAX);

            // Named sessions decode without the drafter by default: its taps force the ungrouped prefill (a 6.9K-token
            // first turn took 22 s instead of 13 s) and fixed-width-one drafting gave no decode gain there; output ids
            // are the same either way (target greedy). STRIX_MM_SESSION_DRAFT=1 keeps the drafter for named sessions.
            if (!session_cli) {
                const char* draft_env = std::getenv("STRIX_MM_SESSION_DRAFT");
                session_options.draft = drafter != nullptr && draft_env != nullptr && std::string_view(draft_env) == "1";
            }
            session_runtime = strix::minimax::session::admit_runtime_pair(
                {session_on, session_explicit, session_config.reason}, eng.rank(),
                [&](const strix::minimax::session::StartupVote& vote) { return strix::minimax::session::exchange_startup(eng, vote); }, [&] {
                    std::fprintf(stderr, "session cache admission: ram_bytes=%llu disk_bytes=%llu disk=%s\n",
                        static_cast<unsigned long long>(session_options.budget), static_cast<unsigned long long>(session_options.disk_budget),
                        session_options.disk_enabled ? "on" : "off");
                    return with_cache_owner(session_options.budget, [&](const auto& owner) {
                        return std::make_unique<strix::minimax::session::Runtime>(eng,drafter.get(),opt,"/proc/self/exe",session_options,dspark_dir,dspark_exl3,draft_file ? draft_file->loader_snapshot() : strix::LoaderSnapshot{}, owner);
                    });
                }, [](const std::string& line) {
                    std::fprintf(stderr, "%s\n", line.c_str());
                });
        } else {
            std::fprintf(stderr, "session cache: off (%s)\n", session_config.reason.c_str());
        }
        if (session_runtime && session_cli) {
            strix::minimax::session::run_cli(*session_runtime,session_options); exit_code = 0; return;
        }

        if (!check.empty()) {
            const int bad = native::resume_check(eng, run, check, check_tokens, check_nested, check_child_layers);
            std::printf("rank %u resume-check: %s\n", opt.rank, bad == 0 ? "ALL IDENTICAL" : "MISMATCH");
            std::fflush(stdout);
            if (bad != 0) { exit_code = 4; return; }
            if (check_only) {  // structural gates only (e.g. a small --max-rows: groups then a single chunk), no serving
                std::printf("rank %u: check only, STOPPED\n", opt.rank);
                exit_code = 0; return;
            }
        }
        // taps only while DSpark is on (both ranks read the same STRIX_MM_SERVE_DSPARK; runtime switches via Command::taps)
        const bool dspark_on = drafter != nullptr && dspark_mode == ds::Mode::on;
        if (drafter != nullptr) eng.set_taps(dspark_on);
        if (opt.rank != 0) {
            return;
        }
        {
            shared.policy.nest = env_num("STRIX_MM_SERVE_NEST", 1) != 0;
            shared.nest_rows = eng.nested_rows();
            shared.nest_reason = eng.nested_reason();
            shared.nest_taps = drafter != nullptr;
            shared.policy.sched = ms::sched_mode(std::getenv("STRIX_MM_SERVE_SCHED"));
            shared.policy.service_ms = ms::service_ms(std::getenv("STRIX_MM_SERVE_SERVICE_MS"));
            shared.policy.short_first = env_num("STRIX_MM_SERVE_SHORT_FIRST", 1) != 0;
            shared.unlimited_reason = eng.prefill_unlimited_reason();
            shared.policy.gap_ms = ms::gap_ms(std::getenv("STRIX_MM_SERVE_GAP_MS"));
            shared.policy.dspark_mode = dspark_mode;
            const char* lookup_env = std::getenv("STRIX_MM_LOOKUP");
            shared.policy.lookup = lookup_env == nullptr || std::strcmp(lookup_env, "0") != 0;
            shared.policy.dspark_prior = dspark_prior;
            shared.policy.inflight = dspark_inflight;
            shared.policy.draft_slots = static_cast<std::uint32_t>(std::clamp(env_num("STRIX_MM_SERVE_DRAFT_SLOTS", 8), 1.0, 8.0));
        }
        auto& session_service = sessions;
        auto refresh_session_metadata = [this] {
            try {
                sessions.tasks.publish_metadata(ms::session_metadata(*session_runtime, session_counts, true),
                                                       ms::session_metadata(*session_runtime, session_counts, false));
            } catch (const std::exception& e) {
                auto error = sv::session_error("unavailable", std::string(e.what()) + "; retry after 1 second", 503);
                sessions.tasks.publish_metadata(error, error);
            }
        };
        if (session_runtime) {
            refresh_session_metadata();
            session_service.execute = [this, refresh_session_metadata](const std::string& body, const ms::SessionService::Publish& publish) {
                auto reply = ms::session_execute(*engine_owner, *session_runtime, body, session_counts, publish);
                refresh_session_metadata();
                return reply;
            };
        }
        auto& oi = info;
        oi.model = model_id;
        oi.context_tokens = opt.capacity;
        oi.slots = opt.slots;
        oi.transport = comm.transport_name();
        oi.load_seconds = lr.seconds;
        oi.device_used_bytes = static_cast<std::uint64_t>(std::max(0.0, lr.device_gib) * 1073741824.0);
        oi.draft = drafter != nullptr;
        oi.prefix_cache = prefix_agreed;

        session_enabled = session_runtime != nullptr;
        if(deployment_plan) {
            deployment.deployment_json=[deployment_plan]{return deployment_plan->to_json();};
            deployment.deployment_id=[deployment_plan]{return deployment_plan->hash();};
        }
        const auto tok=MinimaxTokenizer::load(opt.model_dir + "/tokenizer.json");
        tokenizer=api_tokenizer(tok);
        lead_owner=make_lead<mm::Engine,Runner>(eng,run,opt.max_rows,dspark_on,g_stop,
            [tok](const std::vector<std::uint32_t>& ids){return tok->decode(ids,true);},
            session_runtime ? &sessions : nullptr);
        } catch (...) { fail(); throw; }
    }
    ~NativeAssets() override { close(); }
    std::optional<int> startup() override {return exit_code;}
    sv::SessionRuntime& runtime() override {
        if(!lead_owner)throw std::logic_error("MiniMax follower/qualification has no leader runtime");
        return lead_owner->runtime();
    }
    int follow() override {
        native::follow(*engine_owner,*runner_owner,session_runtime.get());
        std::printf("rank 1: STOPPED\n"); return 0;
    }
    void close() noexcept override {
        g_stop.store(true); ServiceAssets::close();
        if(lead_owner)lead_owner->runtime().shutdown();
    }
    void fail() noexcept override {
        close();
        try {if(engine_owner)engine_owner->abort("serve lifecycle failure");} catch(...) {}
        // Setup can fail before Engine construction. Wake the peer in that case,
        // and still close the host channel if the engine's abort itself failed.
        try {if(transport)transport->abort("serve lifecycle failure");} catch(...) {}
    }
};
} // namespace
std::unique_ptr<ServiceAssets> load_native_service(const ServePreparation& p) {
    return std::make_unique<NativeAssets>(p);
}
} // namespace strix::minimax::serve
#else
namespace strix::minimax::serve {
std::unique_ptr<ServiceAssets> load_native_service(const ServePreparation&) {
    throw std::runtime_error("MiniMax native execution unavailable: build the native engine and required local assets; CPU reference operations do not execute this model");
}
} // namespace strix::minimax::serve
#endif
