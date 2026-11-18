// Included after Runner/Store. Only the explicitly selected layer-cut route enters here.
namespace lc = strix::pair::layercut;
namespace mlc = strix::minimax::layercut;

struct CutSlot {
    std::uint8_t* act = nullptr;
    std::uint8_t* kv[2]{};
    cudaEvent_t ready{}, copied{}, layer_ready{};
    std::mutex mutex;
    std::condition_variable cv;
    unsigned pending = 0;
    cudaStream_t copy = nullptr;
    void alloc(unsigned cut, unsigned block, cudaStream_t stream) {
        copy = stream;
        ck(cudaHostAlloc(reinterpret_cast<void**>(&act), std::size_t{block} * mlc::seam_bytes, cudaHostAllocDefault), "cut activation ring");
        for (auto& p : kv) ck(cudaHostAlloc(reinterpret_cast<void**>(&p), std::size_t{cut} * block * 1280, cudaHostAllocDefault), "cut KV ring");
        ck(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "cut ready");
        ck(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming), "cut copied");
        ck(cudaEventCreateWithFlags(&layer_ready, cudaEventDisableTiming), "cut layer ready");
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!cv.wait_for(lock, std::chrono::seconds(120), [&] { return pending == 0; }))
            throw std::runtime_error("layercut: ring stalled waiting for Sender release");
    }
    void release() { std::lock_guard<std::mutex> lock(mutex); --pending; cv.notify_all(); }
    ~CutSlot() {
        if (copy) cudaStreamSynchronize(copy); // also retire a partially enqueued D2H on exceptions
        cudaFreeHost(act); for (auto* p : kv) cudaFreeHost(p);
        cudaEventDestroy(ready); cudaEventDestroy(copied); cudaEventDestroy(layer_ready);
    }
};

// act-first (Meta v2): the ring holds ACT only; [0,k) KV of the whole prompt is resident in seq anyway, so finished
// blocks are exported later in kv_frag-row fragments into a bounded pinned pool and handed to Lanes as bulk.
struct ActSlot {
    std::uint8_t* act = nullptr;
    cudaEvent_t ready{}, copied{};
    std::mutex mutex;
    std::condition_variable cv;
    unsigned pending = 0;
    cudaStream_t copy = nullptr;
    void alloc(unsigned block, cudaStream_t stream) {
        copy = stream;
        ck(cudaHostAlloc(reinterpret_cast<void**>(&act), std::size_t{block} * mlc::seam_bytes, cudaHostAllocDefault), "cut activation ring");
        ck(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "cut ready");
        ck(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming), "cut copied");
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!cv.wait_for(lock, std::chrono::seconds(120), [&] { return pending == 0; }))
            throw std::runtime_error("layercut: ACT ring stalled waiting for Sender release");
    }
    void release() { std::lock_guard<std::mutex> lock(mutex); --pending; cv.notify_all(); }
    ~ActSlot() {
        if (copy) cudaStreamSynchronize(copy);
        cudaFreeHost(act); cudaEventDestroy(ready); cudaEventDestroy(copied);
    }
};
struct KvPool {  // fixed pinned fragments of one rank; get() blocks until one is released by the transport
    std::vector<std::uint8_t*> all, free;
    std::mutex m;
    std::condition_variable cv;
    bool dead = false;
    void alloc(std::size_t count, std::size_t bytes) {
        for (std::size_t i = 0; i < count; ++i) {
            std::uint8_t* p = nullptr;
            ck(cudaHostAlloc(reinterpret_cast<void**>(&p), bytes, cudaHostAllocDefault), "cut KV pool");
            all.push_back(p); free.push_back(p);
        }
    }
    std::uint8_t* get() {
        std::unique_lock<std::mutex> lock(m);
        if (!cv.wait_for(lock, std::chrono::seconds(120), [&] { return dead || !free.empty(); }))
            throw std::runtime_error("layercut: KV pool stalled waiting for Sender release");
        if (dead) throw std::runtime_error("layercut: KV pump cancelled");
        auto* p = free.back(); free.pop_back();
        return p;
    }
    void put(std::uint8_t* p) { { std::lock_guard<std::mutex> lock(m); free.push_back(p); } cv.notify_all(); }
    void kill() { { std::lock_guard<std::mutex> lock(m); dead = true; } cv.notify_all(); }
    ~KvPool() { for (auto* p : all) cudaFreeHost(p); }
};
// Per block, seconds after the run start on the PC clock (-1 = none): ring wait, GPU+D2H, ACT handed, ACT and last KV
// piece released per rank (identity: last byte to the socket; bp-v1: encoded).
struct BlockTimes { double buf_wait = 0, gpu = 0, act_q = -1, act_sent[2]{-1, -1}, kv_sent[2]{-1, -1}; };

void run_layercut(const lc::Resolved& cfg, Store& store, const std::vector<std::uint16_t>& table,
                  const float* w_in0, const cu::RopeRow* rope, const std::vector<std::uint32_t>& tokens,
                  const std::string& destinations, const std::string& pair_identity, unsigned repeat, bool bp,
                  bool abab_xport = false) {
    const auto endpoints = strix::handoff::parse_path_map(destinations);
    if (endpoints.size() != 2 || tokens.empty() || tokens.size() > UINT32_MAX || repeat == 0)
        throw std::invalid_argument("layercut: needs one nonempty prompt, two --send endpoints and --repeat >= 1");
    const unsigned cut = cfg.cut, block = cfg.block;
    const auto schedule = lc::blocks(tokens.size(), block);
    cu::SeqView seq{};
    seq.cap = static_cast<std::uint32_t>(tokens.size());
    seq.k = dalloc<std::uint16_t>(std::size_t{cut} * seq.cap * 512);
    seq.v = dalloc<std::uint16_t>(std::size_t{cut} * seq.cap * 512);
    seq.ik = dalloc<std::uint16_t>(std::size_t{cut} * seq.cap * 128);
    cu::SeqView* seq_d = dalloc<cu::SeqView>(1);
    Runner run;
    // A separate scratch runner for the short tail keeps R-dependent strides/maps correct.
    run.alloc(std::min<std::size_t>(block, tokens.size()), (seq.cap + 127) / 128);
    run.row_seq = dupload(std::vector<std::uint32_t>(run.R, 0));
    struct CopyStream {
        cudaStream_t value = nullptr;
        CopyStream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "cut copy stream"); }
        ~CopyStream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
    } copy_owner;
    const auto copy = copy_owner.value;
    // --abab-xport (bench): runs alternate the resolved xport and the other one, both rings allocated once
    const bool first_v2 = cfg.xport == lc::Xport::act_first;
    const bool need_v2 = first_v2 || abab_xport, need_v1 = !first_v2 || abab_xport;
    const unsigned frag = std::min(cfg.kv_frag, block);
    std::unique_ptr<CutSlot[]> ring;
    std::unique_ptr<ActSlot[]> act_ring;
    KvPool pool[2];
    struct KvStream {
        cudaStream_t value = nullptr;
        KvStream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "cut KV stream"); }
        ~KvStream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
    };
    std::unique_ptr<KvStream[]> kv_streams;
    if (need_v2) {
        act_ring = std::make_unique<ActSlot[]>(2);
        for (unsigned i = 0; i < 2; ++i) act_ring[i].alloc(block, copy);
        kv_streams = std::make_unique<KvStream[]>(2);
        const std::size_t frag_bytes = std::size_t{frag} * 1280;
        for (auto& p : pool) p.alloc(std::max<std::size_t>(8, (std::size_t{32} << 20) / frag_bytes), frag_bytes);
    }
    if (need_v1) {
        ring = std::make_unique<CutSlot[]>(2);
        for (unsigned i = 0; i < 2; ++i) ring[i].alloc(cut, block, copy);
    }
    const cu::SeqView kv_seq = seq;  // the pump reads k/v/ik/cap only; the block loop rewrites rows/pos0
    const std::string id = mlc::identity(cut, pair_identity);
    std::printf("identity %s\n", id.c_str());
    struct Times { std::mutex mutex; Clock::time_point start; double act = 0, kv = 0; std::vector<BlockTimes> blocks; };
    for (unsigned it = 0; it < repeat; ++it) {
        const bool act_first = abab_xport ? ((it % 2 == 0) == first_v2) : first_v2;
        strix::handoff::SenderOptions options;
        options.codec = bp ? strix::handoff::Codec::bp_v1 : strix::handoff::Codec::identity;
        if (act_first) {
            // Lanes and the rings bound the caller bytes; small kernel/encoded backlogs keep a new ACT close to the wire.
            options.max_pending_plain_bytes = 0;
            options.socket_buffer = 4 << 20;
            options.max_queued_wire_bytes = std::size_t{8} << 20;
        } else {
            options.max_pending_plain_bytes = std::size_t{2} * block * (cut * 1280 + mlc::seam_bytes);
        }
        lc::Meta meta{act_first ? lc::kMetaV2 : 1U, cut, block, mlc::hidden, tokens.size(), lc::token_fnv(tokens), id};
        std::unique_ptr<strix::handoff::Sender> sender[2];
        for (unsigned rank = 0; rank < 2; ++rank)
            sender[rank] = std::make_unique<strix::handoff::Sender>(endpoints[rank],
                strix::handoff::Hello{it + 1ULL, mlc::format, rank, 2, lc::pack(meta)}, options);
        std::unique_ptr<strix::handoff::Lanes> lanes;
        if (act_first)
            lanes = std::make_unique<strix::handoff::Lanes>(std::vector<strix::handoff::Sender*>{sender[0].get(), sender[1].get()},
                                                            strix::handoff::Lanes::Options{});
        const auto start = Clock::now();
        double compute = 0;
        auto times = std::make_shared<Times>(); times->start = start; times->blocks.resize(schedule.size());
        // KV pumps (act-first): block b is exported once the host saw its ACT copy complete (all of its layers ran).
        std::mutex done_mutex;
        std::condition_variable done_cv;
        std::size_t done_blocks = 0;
        bool cancel = false;
        std::exception_ptr pump_error[2];
        std::thread pumps[2];
        const auto stop_pumps = [&] {
            { std::lock_guard<std::mutex> lock(done_mutex); cancel = true; }
            done_cv.notify_all();
            if (lanes) lanes->abort();
            for (auto& p : pool) p.kill();
            for (auto& t : pumps) if (t.joinable()) t.join();
        };
        try {
        if (act_first) for (unsigned rank = 0; rank < 2; ++rank) pumps[rank] = std::thread([&, rank] {
            try {
                for (std::size_t b = 0; b < schedule.size(); ++b) {
                    {
                        std::unique_lock<std::mutex> lock(done_mutex);
                        done_cv.wait(lock, [&] { return cancel || done_blocks > b; });
                        if (cancel) return;
                    }
                    const auto desc = schedule[b];
                    for (unsigned layer = 0; layer < cut; ++layer)
                        for (std::uint32_t f = 0; f < desc.rows; f += frag) {
                            const std::uint32_t n = std::min(frag, desc.rows - f);
                            std::uint8_t* buf = pool[rank].get();
                            cu::export_layer(kv_seq, layer, rank, desc.row0 + f, n, buf, kv_streams[rank].value);
                            ck(cudaStreamSynchronize(kv_streams[rank].value), "cut KV export");
                            const bool last = layer + 1 == cut && f + n == desc.rows;
                            lanes->bulk(rank, {lc::kKv, layer, 0, desc.row0 + f, n},
                                {strix::bp_codec::Region{buf, std::size_t{n} * 1280, 2}},
                                [times, poolp = &pool[rank], buf, rank, b, last] {
                                    poolp->put(buf);
                                    std::lock_guard<std::mutex> lock(times->mutex);
                                    const double t = secs(times->start, Clock::now());
                                    times->kv = std::max(times->kv, t);
                                    if (last) times->blocks[b].kv_sent[rank] = std::max(times->blocks[b].kv_sent[rank], t);
                                });
                        }
                }
            } catch (...) {
                pump_error[rank] = std::current_exception();
                for (auto& s : sender) s->abort("layercut KV pump failed");
            }
        });
        for (std::size_t b = 0; b < schedule.size(); ++b) {
            const auto w0 = Clock::now();
            std::mutex* slot_mutex = nullptr;
            std::condition_variable* slot_cv = nullptr;
            unsigned* slot_pending = nullptr;
            std::uint8_t* act_host = nullptr;
            cudaEvent_t ev_ready{}, ev_copied{};
            if (act_first) {
                auto& slot = act_ring[b % 2];
                slot.wait(); // only ACT sends hold this slot
                slot_mutex = &slot.mutex; slot_cv = &slot.cv; slot_pending = &slot.pending; act_host = slot.act;
                ev_ready = slot.ready; ev_copied = slot.copied;
            } else {
                auto& slot = ring[b % 2];
                slot.wait(); // bounded backpressure; callbacks return every submitted buffer
                slot_mutex = &slot.mutex; slot_cv = &slot.cv; slot_pending = &slot.pending; act_host = slot.act;
                ev_ready = slot.ready; ev_copied = slot.copied;
            }
            times->blocks[b].buf_wait = secs(w0, Clock::now());
            const auto desc = schedule[b];
            if (run.R != desc.rows) {
                run.release(); run.tiles_d = nullptr;
                run.alloc(desc.rows, (seq.cap + 127) / 128);
                run.row_seq = dupload(std::vector<std::uint32_t>(run.R, 0));
            }
            seq.rows = desc.rows; seq.pos0 = static_cast<std::uint32_t>(desc.row0);
            ck(cudaMemcpy(seq_d, &seq, sizeof(seq), cudaMemcpyHostToDevice), "cut sequence");
            cudaFree(run.tiles_d);
            std::vector<uint2> tiles;
            for (unsigned r = 0; r < desc.rows; r += 32) tiles.push_back(make_uint2(r, std::min(32U, desc.rows - r)));
            run.tiles_d = dupload(tiles); run.n_tiles = static_cast<unsigned>(tiles.size());
            std::vector<std::uint32_t> embedding(std::size_t{desc.rows} * kHidden);
            for (unsigned r = 0; r < desc.rows; ++r) {
                const auto token = tokens[static_cast<std::size_t>(desc.row0) + r];
                if (token >= table.size() / kHidden) throw std::invalid_argument("layercut: token outside embedding");
                for (unsigned j = 0; j < kHidden; ++j)
                    embedding[std::size_t{r} * kHidden + j] = std::uint32_t{table[std::size_t{token} * kHidden + j]} << 16;
            }
            const auto begin = Clock::now();
            ck(cudaMemcpy(run.h, embedding.data(), embedding.size() * 4, cudaMemcpyHostToDevice), "cut embed");
            cu::rmsnorm(run.h, w_in0, run.hn, desc.rows, nullptr);
            for (unsigned layer = 0; layer < cut; ++layer) {
                run.layer(hx::bind(store.meta[layer], store.dev[layer]), layer, seq_d, rope);
                if (!act_first) {
                    auto& slot = ring[b % 2];
                    ck(cudaEventRecord(slot.layer_ready, nullptr), "cut layer event");
                    ck(cudaStreamWaitEvent(copy, slot.layer_ready, 0), "cut copy waits layer");
                    for (unsigned rank = 0; rank < 2; ++rank)
                        cu::export_layer(seq, layer, rank, desc.row0, desc.rows,
                            slot.kv[rank] + std::size_t{layer} * block * 1280, copy);
                }
            }
            ck(cudaEventRecord(ev_ready, nullptr), "cut residual ready");
            ck(cudaStreamWaitEvent(copy, ev_ready, 0), "cut waits residual");
            ck(cudaMemcpyAsync(act_host, run.h, std::size_t{desc.rows} * mlc::seam_bytes, cudaMemcpyDeviceToHost, copy), "cut residual copy");
            ck(cudaEventRecord(ev_copied, copy), "cut copied");
            // Host waits for D2H, never a compute-stream wait on the network. Next block overlaps send and pair compute.
            ck(cudaEventSynchronize(ev_copied), "cut copy complete");
            const double elapsed = secs(begin, Clock::now());
            compute += elapsed;
            times->blocks[b].gpu = elapsed;
            {
                std::lock_guard<std::mutex> lock(*slot_mutex);
                *slot_pending = (act_first ? 0U : 2 * cut) + (cfg.act == lc::Act::direct ? 2 : 1);
            }
            if (act_first) {
                { std::lock_guard<std::mutex> lock(done_mutex); done_blocks = b + 1; }
                done_cv.notify_all();
            }
            times->blocks[b].act_q = secs(start, Clock::now());
            for (unsigned rank = 0; rank < 2; ++rank) {
                if (rank == 0 || cfg.act == lc::Act::direct) {
                    const strix::bp_codec::Region region{act_host, std::size_t{desc.rows} * mlc::seam_bytes, 4};
                    const strix::handoff::PieceHeader header{lc::kAct, static_cast<std::uint32_t>(b),
                        cfg.act != lc::Act::direct ? strix::handoff::kRelay : 0U, desc.row0, desc.rows};
                    auto done = [times, slot_mutex, slot_cv, slot_pending, rank, b] {
                        {
                            std::lock_guard<std::mutex> lock(times->mutex);
                            const double t = secs(times->start, Clock::now());
                            times->act = std::max(times->act, t);
                            times->blocks[b].act_sent[rank] = std::max(times->blocks[b].act_sent[rank], t);
                        }
                        std::lock_guard<std::mutex> lock(*slot_mutex);
                        --*slot_pending;
                        slot_cv->notify_all();
                    };
                    // act-first: ACT is urgent over both links and always sent uncoded (bp-v1, if on, is for KV only)
                    if (act_first) lanes->urgent(rank, header, {&region, 1}, std::move(done));
                    else sender[rank]->submit(header, {&region, 1}, std::move(done));
                }
                if (!act_first) for (unsigned layer = 0; layer < cut; ++layer) {
                    auto& slot = ring[b % 2];
                    const strix::bp_codec::Region region{slot.kv[rank] + std::size_t{layer} * block * 1280, std::size_t{desc.rows} * 1280, 2};
                    const bool last = layer + 1 == cut;
                    sender[rank]->submit({lc::kKv, layer, 0, desc.row0, desc.rows}, {&region, 1}, [times, slotp = &slot, rank, b, last] {
                        {
                            std::lock_guard<std::mutex> lock(times->mutex);
                            const double t = secs(times->start, Clock::now());
                            times->kv = std::max(times->kv, t);
                            if (last) times->blocks[b].kv_sent[rank] = std::max(times->blocks[b].kv_sent[rank], t);
                        }
                        slotp->release();
                    });
                }
            }
            std::printf("layercut: run %u block %zu row0 %llu rows %u rtx %.6f s submitted %.6f s\n", it, b,
                static_cast<unsigned long long>(desc.row0), desc.rows, elapsed, secs(start, Clock::now()));
        }
        for (auto& t : pumps) if (t.joinable()) t.join();
        for (auto& e : pump_error) if (e) std::rethrow_exception(e);
        if (lanes) lanes->drain();
        } catch (...) {
            for (auto& s : sender) s->abort("layercut producer failed");
            stop_pumps();
            throw;
        }
        // Both ranks must see END before either can finish the pair prefill/import collective.
        strix::handoff::Reply replies[2];
        std::exception_ptr failures[2];
        double ttft = 0;
        std::thread finishers[2];
        try {
        for (unsigned rank = 0; rank < 2; ++rank) finishers[rank] = std::thread([&, rank] {
            try { replies[rank] = sender[rank]->finish(); if (rank == 0) ttft = secs(start, Clock::now()); }
            catch (...) { failures[rank] = std::current_exception(); sender[1 - rank]->abort("peer finish failed"); }
        });
        } catch (...) {
            for (auto& s : sender) s->abort("layercut cannot start finish thread");
            for (auto& t : finishers) if (t.joinable()) t.join();
            throw;
        }
        for (auto& t : finishers) t.join();
        for (auto& e : failures) if (e) std::rethrow_exception(e);
        if (replies[0].status || replies[1].status || replies[0].payload.size() != 4)
            throw std::runtime_error("layercut: rejected or malformed token reply");
        std::uint32_t token = 0; std::memcpy(&token, replies[0].payload.data(), 4);
        double act_last = times->act, kv_last = times->kv;
        if (bp) { // bp release means encoded, not sent. The aggregate last-byte time bounds KV (and ACT under fifo).
            kv_last = std::max(sender[0]->stats().last_byte, sender[1]->stats().last_byte);
            if (!act_first) act_last = kv_last;
        }
        std::printf("layercut: TTFT %.6f s, rtx compute %.6f s, act last byte %.6f s, kv last byte %.6f s, cut %u block %u; run %u %s token %u%s\n",
            ttft, compute, act_last, kv_last, cut, block, it, it == 0 ? "cold" : "warm", token,
            bp ? " (bp last-byte values are aggregate upper bounds from session open)" : "");
        // Timeline (PC clock, s after run start; bp-v1 KV 'sent' = encoded). Correlate with the ranks' layercut-tl lines by block.
        {
            std::lock_guard<std::mutex> lock(times->mutex);
            for (std::size_t b = 0; b < schedule.size(); ++b) {
                const auto& t = times->blocks[b];
                std::printf("layercut-tl pc run %u block %zu rows %u buf_wait %.3f gpu %.3f act_q %.3f act_sent %.3f %.3f kv_sent %.3f %.3f\n",
                    it, b, schedule[b].rows, t.buf_wait, t.gpu, t.act_q, t.act_sent[0], t.act_sent[1], t.kv_sent[0], t.kv_sent[1]);
            }
        }
        const auto s0 = sender[0]->stats(), s1 = sender[1]->stats();
        const auto ls = lanes ? lanes->stats() : strix::handoff::Lanes::Stats{};
        std::printf("layercut-xport pc run %u %s kv_frag %u codec %s wire %.3f/%.3f GB encode %.3f/%.3f s last_byte %.3f/%.3f s bulk_held %.3f s\n",
            it, act_first ? "act-first" : "fifo", act_first ? frag : block, bp ? "bp-v1" : "identity", s0.wire_bytes / 1e9, s1.wire_bytes / 1e9,
            s0.encode_seconds, s1.encode_seconds, s0.last_byte, s1.last_byte, ls.bulk_held_s);
        std::fflush(stdout);
    }
    ck(cudaStreamSynchronize(copy), "cut cleanup");
    run.release(); cudaFree(seq_d); cudaFree(seq.k); cudaFree(seq.v); cudaFree(seq.ik);
}
