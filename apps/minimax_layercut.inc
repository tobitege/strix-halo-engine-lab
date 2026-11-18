// MiniMax adapter: shared layercut policy/order validation, handoff sessions and engine CutFetch.
namespace lc = strix::pair::layercut;
namespace mlc = strix::minimax::layercut;
void cut_hip(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
struct CutPinned {
    std::uint8_t* data = nullptr;
    explicit CutPinned(std::size_t bytes) { cut_hip(hipHostMalloc(reinterpret_cast<void**>(&data), bytes), "layercut pinned"); }
    ~CutPinned() { (void)hipHostFree(data); }
    CutPinned(const CutPinned&) = delete;
};
// The pair part's identity. Production bits depend on the chunk boundaries (pair::layercut::identity_grain), so the
// default identity names its chunk rows (the pair-alone --max-rows); FP32 is row-invariant (gate T1).
std::string cut_pair_identity(std::uint32_t chunk_rows) {
    const char* p = strix::envcfg::text("STRIX_MM_PREFILL_IDENTITY");
    std::string id = p && std::string_view(p) == "fp32" ? "fp32" : "default/c" + std::to_string(chunk_rows);
    const auto policies = strix::minimax::defaults::prefill_identity([](const char* k) { return strix::envcfg::text(k); });
    if (!policies.empty()) id += ";" + policies;
    return id;
}
std::vector<std::string> cut_fields(const std::string& s) {
    std::vector<std::string> f;
    for (std::size_t p = 0, q; p <= s.size(); p = q + 1) {
        q = s.find(':', p); if (q == std::string::npos) q = s.size();
        f.push_back(s.substr(p, q - p));
    }
    if (f.size() != 5) throw std::invalid_argument("--layercut-listen NAME:PROMPT.u32:PORT:STEPS:COUNT");
    return f;
}
struct CutArguments {
    lc::Resolved config;
    std::vector<std::string> listens;
    std::string relay;
    unsigned relay_port = 0;
    std::uint32_t chunk_rows = mm::Options{}.max_rows;  // the pair-alone chunk (--max-rows)
};
CutArguments cut_arguments(int& argc, char** argv) {
    CutArguments args;
    auto cli = lc::take_cli(argc, argv, 5);
    for (int i = 5; i < argc; ++i) {
        const std::string flag = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + flag);
            return argv[++i];
        };
        if (flag == "--layercut-listen") args.listens.push_back(next());
        else if (flag == "--cut-relay") args.relay = next();
        else if (flag == "--cut-relay-listen") args.relay_port = lc::number(next(), "--cut-relay-listen");
        else if (flag == "--max-rows") args.chunk_rows = lc::number(next(), "--max-rows");
    }
    lc::Inputs input{mlc::chain_shape()};
    if (!args.listens.empty()) {
        if (!cli.split) cli.split = "layer-cut";
        input.rows = read_u32(cut_fields(args.listens.front())[1]).size();
    }
    input.pair_grain = lc::route_identity_grain(cli, cut_pair_identity(args.chunk_rows));
    if (input.pair_grain == mlc::pair_chunk_rows) input.pair_unit_scale = mlc::pair_unit_scale();
    input.rtx_rows_s = strix::envcfg::number<double>("STRIX_SPLIT_RTX_ROWS_S", 0, 0);
    input.pair_rows_s = strix::envcfg::number<double>("STRIX_SPLIT_PAIR_ROWS_S", 0, 0);
    input.link_mbps = strix::envcfg::number<double>("STRIX_SPLIT_LINK_MBPS", 0, 0);
    args.config = lc::resolve(cli, input);
    return args;
}
std::vector<std::uint32_t> layercut_session(mm::Engine& eng, strix::handoff::Listener& listener,
    strix::handoff::Listener* relay_listener, const std::string& relay_to, const lc::Resolved& cfg,
    const std::vector<std::uint32_t>& prompt, int steps, const std::string& name, const std::string& out_dir, unsigned req) {
    if (prompt.empty() || steps < 0) throw std::invalid_argument("layercut: empty prompt / negative steps");
    const bool relay = cfg.act == lc::Act::relay;      // rank 0 forwards ACT over TCP (--cut-relay)
    const bool link = cfg.act == lc::Act::link;        // the engine forwards rank 0's ACT over the pair link
    const bool replicated = cfg.act != lc::Act::direct; // ACT reaches rank 0 only, flagged kRelay
    if (relay && ((eng.rank() == 0 && relay_to.empty()) || (eng.rank() == 1 && !relay_listener)))
        throw std::invalid_argument("layercut relay needs --cut-relay on rank 0 and --cut-relay-listen on rank 1");
    std::printf("rank %u layercut-listen %s #%u: waiting\n", eng.rank(), name.c_str(), req); std::fflush(stdout);
    auto session = listener.accept();
    lc::StartupScope startup([&]() noexcept {
        session->cancel("layercut startup failed");
        if (relay && relay_listener) relay_listener->shutdown();
    });
    std::unique_ptr<strix::handoff::Session> relay_session;
    std::unique_ptr<strix::handoff::Sender> relay_sender;
    const auto start = Clock::now();
    const std::string id = mlc::identity(cfg.cut, cut_pair_identity(cfg.grain != 0 ? cfg.grain : mm::Options{}.max_rows));
    lc::Meta expected{0, cfg.cut, cfg.block, mlc::hidden, prompt.size(), lc::token_fnv(prompt), id};
    const auto check_hello = [&](const strix::handoff::Hello& h) {
        if (h.model != mlc::format || h.rank != eng.rank() || h.world != 2)
            throw std::invalid_argument("layercut hello: model/rank/world mismatch");
        lc::validate(lc::unpack(h.meta), expected, mlc::layers);
    };
    try {
        check_hello(session->hello());
        // the producer picks the contract (v1 fifo / v2 act-first); the relay session must then carry the same version
        expected.version = lc::unpack(session->hello().meta).version;
    } catch (...) { session->abort("layercut hello/meta mismatch"); throw; }
    std::printf("identity %s\n", id.c_str());
    // Full prompt staging is intentionally retained until HIP is done reading it. KV layout is component-major.
    if (prompt.size() > std::numeric_limits<std::size_t>::max() / mlc::seam_bytes)
        throw std::invalid_argument("layercut prompt staging overflow");
    CutPinned act(prompt.size() * mlc::seam_bytes);
    std::vector<std::vector<std::uint8_t>> kv(cfg.cut, std::vector<std::uint8_t>(prompt.size() * 1280));
    std::mutex mutex;
    std::condition_variable cv;
    std::uint64_t act_rows = 0;
    std::exception_ptr error;
    double first = -1, last = -1, wait_act = 0;
    // Timeline (rank clock, s after accept): per block ACT stored, ACT relayed (rank 0, released by the relay Sender),
    // last KV piece stored; per fetch call its start and wait. Printed after the reply, outside TTFT.
    const std::size_t n_blocks = (prompt.size() + cfg.block - 1) / cfg.block;
    std::vector<double> tl_act(n_blocks, -1), tl_kv(n_blocks, -1);
    struct RelayTimes { std::mutex m; Clock::time_point start; std::vector<double> at; };
    auto tl_relay = std::make_shared<RelayTimes>();  // the relay Sender may release after this frame on a failure
    tl_relay->start = start; tl_relay->at.assign(n_blocks, -1);
    struct FetchTimes { std::uint64_t row0; double at, wait; };
    std::vector<FetchTimes> tl_fetch;
    // Cancellation wakes both the transport reader and a fetch blocked on missing rows.
    const auto fail = [&](std::exception_ptr e) {
        startup.fail(); // wake relay accept before any peer/engine teardown can wait
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!error) error = e;
            if (relay_session) relay_session->cancel("layercut peer failed");
            cv.notify_all();
        }
        try { eng.abort("layercut receive/compute failed"); } catch (...) {}
        session->cancel("layercut receive/compute failed");
        if (relay_sender) relay_sender->abort("layercut peer failed");
    };
    const auto check_error = [&] {
        std::exception_ptr e;
        { std::lock_guard<std::mutex> lock(mutex); e = error; }
        if (e) std::rethrow_exception(e);
    };
    std::thread relay_finisher;
    auto receive = [&](strix::handoff::Session& source, bool acts, bool states, bool forward) {
        try {
            lc::Pieces order(expected, acts, states, replicated && acts);
            strix::handoff::Piece piece;
            while (source.next(piece)) {
                order.accept(piece.header, piece.plain.size(), 1280);
                const auto& h = piece.header;
                if (h.kind == lc::kAct) {
                    if (forward) {
                        // The relay owns its copy; enqueue before publishing local availability.
                        auto owned = std::make_shared<std::vector<std::uint8_t>>(piece.plain);
                        const strix::bp_codec::Region region{owned->data(), owned->size(), 4};
                        const std::size_t b = static_cast<std::size_t>(h.row0 / cfg.block);
                        relay_sender->submit(h, {&region, 1}, [owned, b, tl = tl_relay]() mutable {
                            owned.reset();
                            std::lock_guard<std::mutex> lock(tl->m);
                            if (b < tl->at.size()) tl->at[b] = secs(tl->start, Clock::now());
                        });
                    }
                    std::memcpy(act.data + h.row0 * mlc::seam_bytes, piece.plain.data(), piece.plain.size());
                } else {
                    lc::scatter_state(kv[h.index], piece.plain, prompt.size(), h.row0, h.rows, mlc::rank_state_widths);
                }
                std::lock_guard<std::mutex> lock(mutex);
                if (first < 0) first = secs(start, Clock::now());
                last = secs(start, Clock::now());
                const std::size_t b = static_cast<std::size_t>(h.row0 / cfg.block);
                if (b < n_blocks) (h.kind == lc::kAct ? tl_act[b] : tl_kv[b]) = last;
                if (h.kind == lc::kAct) { act_rows = h.row0 + h.rows; cv.notify_all(); }
            }
            order.finish();
            if (forward) relay_finisher = std::thread([&] {
                try { if (relay_sender->finish().status) throw std::runtime_error("layercut relay rejected"); }
                catch (...) { fail(std::current_exception()); }
            });
        } catch (...) { fail(std::current_exception()); }
    };
    std::thread main_reader, act_reader;
    std::uint32_t next = 0;
    double prefill = 0, imported = 0, agree_s = 0, reply_at = 0;
    try {
        if (relay && eng.rank() == 0) {
            const auto paths = strix::handoff::parse_path_map(relay_to);
            if (paths.size() != 1) throw std::invalid_argument("--cut-relay needs one HOST:PORT");
            auto h = session->hello(); h.rank = 1;
            strix::handoff::SenderOptions options;
            options.codec = strix::handoff::Codec::identity;
            options.connect_timeout = std::chrono::seconds(120);
            options.reply_timeout = std::chrono::seconds(120);
            options.max_pending_plain_bytes = std::size_t{2} * cfg.block * mlc::seam_bytes;
            relay_sender = std::make_unique<strix::handoff::Sender>(paths[0], h, options);
        }
        main_reader = std::thread([&] { receive(*session, !replicated || eng.rank() == 0, true, relay && eng.rank() == 0); });
        if (relay && eng.rank() == 1) {
            auto accepted = relay_listener->accept();
            { std::lock_guard<std::mutex> lock(mutex); relay_session = std::move(accepted); }
            check_hello(relay_session->hello());
            if (relay_session->hello().session != session->hello().session)
                throw std::invalid_argument("layercut relay: session mismatch");
        }
        check_error();
        startup.complete();
        if (relay_session) act_reader = std::thread([&] { receive(*relay_session, true, false, false); });
        eng.prefill_begin_cut(0, prompt, 0, cfg.cut, [&](std::uint64_t row0, std::size_t rows, float* dst, hipStream_t stream) {
            const auto t = Clock::now();
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return error || act_rows >= row0 + rows; });
            const double waited = secs(t, Clock::now());
            wait_act += waited;
            tl_fetch.push_back({row0, secs(start, t), waited});
            if (error) std::rethrow_exception(error);
            lock.unlock();
            cut_hip(hipMemcpyAsync(dst, act.data + row0 * mlc::seam_bytes, rows * mlc::seam_bytes, hipMemcpyHostToDevice, stream), "layercut fetch");
        }, link);
        while (!eng.prefill_resume(~0U, &next)) {}
        prefill = secs(start, Clock::now());
        // STRIX_MM_QUALITY_OUT=PREFIX (+ STRIX_MM_QUALITY_ROWS=N): the same raw FP32 dumps as a --session prefill, named
        // PREFIX-cut-<name>-{logits,row,tf}-r<rank>.f32, so the mixed identity is compared row by row with the pair alone.
        if (const char* qo = strix::envcfg::text("STRIX_MM_QUALITY_OUT"); qo != nullptr && *qo != '\0') {
            const std::string base = std::string(qo) + "-cut-" + name + "-";
            const auto put = [&](const std::string& what, const float* p, std::size_t n) {
                std::ofstream f(base + what + "-r" + std::to_string(eng.rank()) + ".f32", std::ios::binary);
                f.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n * sizeof(float)));
                if (!f) throw std::runtime_error("STRIX_MM_QUALITY_OUT: cannot write " + base + what);
            };
            const auto lg = eng.last_logits(1);
            put("logits", lg.data(), lg.size());
            const auto out_rows = eng.prefill_out_rows();
            if (out_rows.size() >= mlc::hidden) put("row", out_rows.data() + out_rows.size() - mlc::hidden, mlc::hidden);
            if (const char* qr = strix::envcfg::text("STRIX_MM_QUALITY_ROWS"); qr != nullptr && *qr != '\0') {
                const auto tf = eng.prefill_out_logits(std::stoul(qr));
                put("tf", tf.data(), tf.size());
            }
        }
        main_reader.join(); if (act_reader.joinable()) act_reader.join();
        check_error();
        const auto t = Clock::now();
        for (unsigned layer = 0; layer < cfg.cut; ++layer) eng.import_layer(0, layer, 0, prompt.size(), kv[layer].data());
        cut_hip(hipDeviceSynchronize(), "layercut import");
        imported = secs(t, Clock::now());
        // Both ranks finish state import before the token reply permits the producer to start another request.
        const auto t_agree = Clock::now();
        (void)eng.agree(cfg.cut);
        agree_s = secs(t_agree, Clock::now());
        if (relay_session) relay_session->reply(0);
        if (relay_finisher.joinable()) relay_finisher.join();
        check_error();
        const std::span<const std::uint8_t> token_bytes(reinterpret_cast<const std::uint8_t*>(&next), 4);
        session->reply(0, eng.rank() == 0 ? token_bytes : std::span<const std::uint8_t>{});
        reply_at = secs(start, Clock::now());
        // After the reply (outside TTFT): the imported + computed state digest, the same hash as the session's
        // "state fnv" line (N positions) and over N - 1 positions (the full hand-off imports rows [0, N - 1)).
        // STRIX_MM_STATE_FNV=0 skips it, as for the session prints.
        if (const char* v = strix::envcfg::text("STRIX_MM_STATE_FNV"); !(v != nullptr && std::string_view(v) == "0"))
            std::printf("rank %u layercut %s: state fnv %016llx (slot 0, %zu positions, K/V/index key, 60 layers), first %zu positions %016llx\n",
                        eng.rank(), name.c_str(), static_cast<unsigned long long>(eng.state_fnv(0, prompt.size())), prompt.size(),
                        prompt.size() - 1, static_cast<unsigned long long>(eng.state_fnv(0, prompt.size() - 1)));
    } catch (...) {
        const auto e = std::current_exception();
        fail(e);
        if (main_reader.joinable()) main_reader.join();
        if (act_reader.joinable()) act_reader.join();
        if (relay_finisher.joinable()) relay_finisher.join();
        try { eng.prefill_cancel(); } catch (...) {}
        (void)hipDeviceSynchronize(); // act staging survives every submitted H2D, including a failed compute
        session->abort("layercut request failed");
        if (relay_session) relay_session->abort("layercut request failed");
        std::rethrow_exception(e);
    }
    std::vector<std::uint32_t> generated{next};
    for (int s = 0; s < steps; ++s) {
        const mm::Row row{0, generated.back(), prompt.size() + generated.size() - 1};
        generated.push_back(eng.step({&row, 1})[0]);
    }
    std::printf("rank %u layercut: prefill %.6f s, kv import %.6f s, wait-for-act %.6f s, first piece %.6f s / last piece %.6f s; %s #%u %s tokens fnv %016llx ids:",
        eng.rank(), prefill, imported, wait_act, first, last, name.c_str(), req, req == 0 ? "cold" : "warm",
        static_cast<unsigned long long>(fnv(generated.data(), generated.size() * 4)));
    for (auto token : generated) std::printf(" %u", token);
    std::printf("\n");
    {
        std::lock_guard<std::mutex> lock(mutex);
        std::lock_guard<std::mutex> relay_lock(tl_relay->m);
        for (std::size_t b = 0; b < n_blocks; ++b) {
            double fetch_at = -1, fetch_wait = 0;
            for (const auto& f : tl_fetch)
                if (f.row0 / cfg.block == b) { if (fetch_at < 0) fetch_at = f.at; fetch_wait += f.wait; }
            std::printf("layercut-tl r%u req %u block %zu act %.3f relay %.3f kv %.3f fetch %.3f wait %.3f\n", eng.rank(), req, b,
                        tl_act[b], tl_relay->at[b], tl_kv[b], fetch_at, fetch_wait);
        }
        const auto rs = session->stats();
        // exactness witness: XXH64 of every ACT byte this rank staged (rank 0, or both ranks with direct)
        char act_digest[32] = "none";
        if (!replicated || eng.rank() == 0) {
            strix::handoff::Xxh64 x(0);
            x.update(act.data, prompt.size() * mlc::seam_bytes);
            std::snprintf(act_digest, sizeof act_digest, "%016llx", static_cast<unsigned long long>(x.digest()));
        }
        std::printf("layercut-act r%u req %u act xxh64 %s rows %zu\n", eng.rank(), req, act_digest, prompt.size());
        std::printf("layercut-rank r%u req %u contract v%u prefill %.3f import %.3f agree %.3f reply %.3f recv wait %.3f decode %.3f end %.3f%s\n",
                    eng.rank(), req, expected.version, prefill, imported, agree_s, reply_at, rs.wait_seconds, rs.decode_seconds, rs.end,
                    relay_session ? " (ACT on the relay session)" : "");
    }
    std::fflush(stdout);
    std::ofstream out(out_dir + "/gen-layercut-" + name + "-" + std::to_string(req) + "-rank" + std::to_string(eng.rank()) + ".u32", std::ios::binary);
    out.write(reinterpret_cast<const char*>(generated.data()), static_cast<std::streamsize>(generated.size() * 4));
    return generated;
}
