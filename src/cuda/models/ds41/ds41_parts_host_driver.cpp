// deepseek41 CUDA parts-host layer driver (cuda_ds41_parts_host_driver.hpp).

#include "strix/cuda/ds41_parts_host_driver.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>

namespace strix::ds41::cuda {

namespace {

using clock_type = std::chrono::steady_clock;
double us_since(clock_type::time_point a, clock_type::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }

template <class T>
class Pinned {
public:
    Pinned() = default;
    explicit Pinned(std::size_t n) {
        if (n) STRIX_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&p_), n * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable));
    }
    ~Pinned() { if (p_) cudaFreeHost(p_); }
    Pinned(const Pinned&) = delete;
    Pinned& operator=(const Pinned&) = delete;
    Pinned(Pinned&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    Pinned& operator=(Pinned&& o) noexcept { std::swap(p_, o.p_); return *this; }
    [[nodiscard]] T* get() const noexcept { return p_; }
    // the device-side address of the mapped allocation
    [[nodiscard]] T* device() const {
        void* d = nullptr;
        if (p_) STRIX_CUDA_CHECK(cudaHostGetDevicePointer(&d, p_, 0));
        return static_cast<T*>(d);
    }

private:
    T* p_ = nullptr;
};

float elapsed_us(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0F;
    STRIX_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms * 1000.0F;
}

}  // namespace

struct PartsHost::Impl {
    PartsHostOptions o;
    std::vector<std::unique_ptr<xrpc::ExpertClient>> clients;
    cudaStream_t copy = nullptr;
    cudaEvent_t ev_d2h = nullptr, ev_local = nullptr, ev_h2d_start = nullptr, ev_h2d = nullptr, ev_comb = nullptr, ev_done = nullptr;
    Pinned<std::uint32_t> h_ids;
    Pinned<float> h_weights, h_deltas, h_remote;
    Pinned<std::int8_t> h_codes;
    Pinned<std::int32_t> h_tables;
    Pinned<std::uint32_t> h_flag;
    std::uint32_t sequence = 0;
    DeviceBuffer<std::uint8_t> workspace;
    DeviceBuffer<float> d_remote;
    DeviceBuffer<std::int32_t> d_tables;
    struct Payload {
        std::vector<std::uint8_t> bytes;
        const std::uint8_t* registered = nullptr;
        const std::uint8_t* device = nullptr;  // mapped address of `registered` (zero-copy reads)
    };
    std::vector<Payload> payloads;

    explicit Impl(const PartsHostOptions& opt) : o(opt) {
        const auto& s = o.shape;
        const std::size_t slots = std::size_t{o.capacity} * s.top_k;
        STRIX_CUDA_CHECK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
        for (auto* e : {&ev_d2h, &ev_local, &ev_h2d_start, &ev_h2d, &ev_comb, &ev_done}) STRIX_CUDA_CHECK(cudaEventCreate(e));
        STRIX_CUDA_CHECK(cudaEventRecord(ev_h2d, copy));  // valid for the first wait
        STRIX_CUDA_CHECK(cudaEventRecord(ev_done, copy));
        h_flag = Pinned<std::uint32_t>(1);
        *h_flag.get() = 0U;
        h_ids = Pinned<std::uint32_t>(slots);
        h_weights = Pinned<float>(slots);
        h_codes = Pinned<std::int8_t>(std::size_t{o.capacity} * s.hidden);
        h_deltas = Pinned<float>(std::size_t{o.capacity} * s.hidden / 32U);
        h_tables = Pinned<std::int32_t>(seeded_combine_table_ints(o.capacity, s.top_k));
        workspace.resize(moe_layer_workspace_bytes(o.capacity, s));
        d_tables.resize(seeded_combine_table_ints(o.capacity, s.top_k));
        if (!o.endpoints.empty()) {
            h_remote = Pinned<float>(slots * s.hidden);
            d_remote.resize(slots * s.hidden);
        }
    }
    ~Impl() {
        for (auto& p : payloads)
            if (p.registered) cudaHostUnregister(const_cast<std::uint8_t*>(p.registered));
        for (auto* e : {ev_d2h, ev_local, ev_h2d_start, ev_h2d, ev_comb, ev_done}) if (e) cudaEventDestroy(e);
        if (copy) cudaStreamDestroy(copy);
    }
};

PartsHost::PartsHost(const PartsHostOptions& options) : impl_(std::make_unique<Impl>(options)) {
    auto& o = impl_->o;
    if (o.capacity == 0U || o.shape.top_k == 0U || o.shape.top_k > xrpc::kMaxTopK) throw std::invalid_argument("ds41 parts-host: bad options");
    if (o.hello.client.empty()) o.hello.client = "cuda-parts-host";
    if (o.hello.numeric_identity.empty()) o.hello.numeric_identity = "ds41-exact-v1";
    const std::size_t result_cap = xrpc::kResultHeadBytes + std::size_t{o.capacity} * o.shape.top_k * o.shape.hidden * 4U;
    for (const auto& ep : o.endpoints) {
        impl_->clients.push_back(std::make_unique<xrpc::ExpertClient>(ep, o.hello, o.connect_wait, o.io_timeout, o.busy_poll));
        acks_.push_back(impl_->clients.back()->ack());
        if (acks_.back().max_rows < o.capacity)
            throw std::runtime_error("ds41 parts-host: server " + ep + " takes fewer rows than the window capacity");
        // answers land in a page-locked buffer (reused: receive_into keeps the allocation) so the H2D reads them in place
        Impl::Payload p;
        p.bytes.resize(result_cap);
        p.bytes.clear();
        if (cudaHostRegister(p.bytes.data(), result_cap, cudaHostRegisterMapped | cudaHostRegisterPortable) == cudaSuccess) {
            p.registered = p.bytes.data();
            void* d = nullptr;
            if (cudaHostGetDevicePointer(&d, p.bytes.data(), 0) == cudaSuccess) p.device = static_cast<const std::uint8_t*>(d);
            else (void)cudaGetLastError();
        } else {
            (void)cudaGetLastError();
        }
        impl_->payloads.push_back(std::move(p));
    }
}

PartsHost::~PartsHost() = default;

const std::uint32_t* PartsHost::last_host_ids() const noexcept { return impl_->h_ids.get(); }
cudaEvent_t PartsHost::last_done_event() const noexcept { return impl_->ev_done; }

PartsHostTiming PartsHost::run_layer(const ResidentLayer& layer, const Window& w, cudaStream_t stream, bool wait,
                                     const std::function<void(cudaStream_t)>& extra_local) {
    auto& m = *impl_;
    const auto& o = m.o;
    const auto& s = o.shape;
    const std::uint32_t R = w.rows, K = s.top_k, H = s.hidden, G = H / 32U;
    if (R == 0U || R > o.capacity) throw std::invalid_argument("ds41 parts-host: window rows outside the capacity");
    if (layer.host_slot_of.size() != s.experts || !layer.slot_of) throw std::invalid_argument("ds41 parts-host: residency table size");
    PartsHostTiming t;
    t.rows = R;
    const auto t_entry = clock_type::now();
    // the previous layer's combine / H2D read the pinned tables and the answer payloads: done before they are rewritten
    STRIX_CUDA_CHECK(cudaEventSynchronize(o.answers_in_place ? m.ev_done : m.ev_h2d));

    const std::uint32_t seq = ++m.sequence;
    if (o.export_flag) {  // stream: a small kernel writes the window to mapped host memory and raises the flag
        ExportWindowView ev;
        ev.ids = w.ids;
        ev.weights = w.weights;
        ev.codes = w.x.codes;
        ev.deltas = w.x.deltas;
        ev.rows = R;
        ev.top_k = K;
        ev.hidden = H;
        ev.host_ids = m.h_ids.device();
        ev.host_weights = m.h_weights.device();
        ev.host_codes = m.h_codes.device();
        ev.host_deltas = m.h_deltas.device();
        ev.flag = m.h_flag.device();
        ev.sequence = seq;
        enqueue_export_window(ev, stream);
        STRIX_CUDA_CHECK(cudaEventRecord(m.ev_d2h, stream));
        (void)cudaStreamQuery(stream);  // flush the submission now (batched submission would delay the export)
    } else {  // copy stream: the router's ids and weights and the A8 input to the host
        STRIX_CUDA_CHECK(cudaStreamWaitEvent(m.copy, w.router_ready, 0));
        STRIX_CUDA_CHECK(cudaMemcpyAsync(m.h_ids.get(), w.ids, std::size_t{R} * K * 4U, cudaMemcpyDeviceToHost, m.copy));
        STRIX_CUDA_CHECK(cudaMemcpyAsync(m.h_weights.get(), w.weights, std::size_t{R} * K * 4U, cudaMemcpyDeviceToHost, m.copy));
        STRIX_CUDA_CHECK(cudaMemcpyAsync(m.h_codes.get(), w.x.codes, std::size_t{R} * H, cudaMemcpyDeviceToHost, m.copy));
        STRIX_CUDA_CHECK(cudaMemcpyAsync(m.h_deltas.get(), w.x.deltas, std::size_t{R} * G * 4U, cudaMemcpyDeviceToHost, m.copy));
        STRIX_CUDA_CHECK(cudaEventRecord(m.ev_d2h, m.copy));
    }

    // stream: the resident experts of the window's local slots
    const MoeLayerScratch scratch = moe_layer_workspace_views(m.workspace.get(), o.capacity, s);
    ResidentRoutesView rv;
    rv.ids = w.ids;
    rv.slot_of = layer.slot_of;
    rv.rows = R;
    rv.top_k = K;
    rv.experts = s.experts;
    rv.split = o.split;
    rv.out = scratch.routes;
    enqueue_resident_routes(rv, stream);
    const auto resident = static_cast<std::uint32_t>(std::count_if(layer.host_slot_of.begin(), layer.host_slot_of.end(), [](std::int32_t v) { return v >= 0; }));
    const std::uint32_t max_union = std::min(R * K, resident);
    if (max_union) {
        RoutedGateUpView g;
        g.shape = s;
        g.weights = layer.weights;
        g.routes = scratch.routes;
        g.max_union = max_union;
        g.max_rows = std::min(R, 8U);
        g.x = w.x;
        g.act = scratch.act;
        RoutedDownView d;
        d.shape = s;
        d.weights = layer.weights;
        d.routes = scratch.routes;
        d.max_union = max_union;
        d.max_rows = g.max_rows;
        d.act = scratch.act;
        d.raw = w.raw;
        if (o.grouped_min_rows != 0U && R >= o.grouped_min_rows) {
            enqueue_routed_gate_up_grouped(g, R, scratch.grouped, stream);
            enqueue_routed_down_grouped(d, R, scratch.grouped, stream);
        } else {
            enqueue_routed_gate_up(g, stream);
            enqueue_routed_down(d, stream);
        }
    }
    if (extra_local) extra_local(stream);
    STRIX_CUDA_CHECK(cudaEventRecord(m.ev_local, stream));
    (void)cudaStreamQuery(stream);  // flush
    t.enqueue_us = us_since(t_entry, clock_type::now());

    // host: the ids (spin: no sleep between the copy's end and the requests)
    if (o.export_flag) {
        const volatile std::uint32_t* flag = m.h_flag.get();
        std::uint64_t spins = 0;
        while (*flag != seq)
            if ((++spins & 0xffffU) == 0U) {  // flush the submission and keep the device's errors visible while spinning
                const auto st = cudaStreamQuery(stream);
                if (st != cudaSuccess && st != cudaErrorNotReady) STRIX_CUDA_CHECK(st);
            }
        std::atomic_thread_fence(std::memory_order_acquire);
    } else {
        for (;;) {
            const auto st = cudaEventQuery(m.ev_d2h);
            if (st == cudaSuccess) break;
            if (st != cudaErrorNotReady) STRIX_CUDA_CHECK(st);
        }
    }
    const auto t_host = clock_type::now();
    t.host_wait_us = us_since(t_entry, t_host);
    const std::size_t servers = m.clients.size();
    std::vector<std::uint32_t> remote(R, 0U);
    std::vector<std::vector<std::uint32_t>> mask(servers, std::vector<std::uint32_t>(R, 0U));
    for (std::uint32_t r = 0; r < R; ++r)
        for (std::uint32_t k = 0; k < K; ++k) {
            const std::uint32_t e = m.h_ids.get()[r * K + k];
            if (e >= s.experts) throw std::runtime_error("ds41 parts-host: expert id outside the layer");
            if (layer.host_slot_of[e] >= 0) { ++t.local_slots; continue; }
            std::size_t srv = 0;
            while (srv < servers && !acks_[srv].residency.serves(layer.layer, e)) ++srv;
            if (srv == servers)
                throw std::runtime_error("ds41 parts-host: expert " + std::to_string(e) + " of layer " + std::to_string(layer.layer) +
                                         " is neither resident nor served");
            mask[srv][r] |= 1U << k;
            remote[r] |= 1U << k;
            ++t.remote_slots;
        }

    MoeCombineView plain;
    plain.raw = w.raw;
    plain.shared = w.shared;
    plain.weights = w.weights;
    plain.order = scratch.routes.order;
    plain.low_counts = scratch.routes.low_counts;
    plain.rows = R;
    plain.top_k = K;
    plain.hidden = H;
    plain.out = w.y;
    if (t.remote_slots == 0U) {  // everything local: no RPC, the one-device combine
        t.prep_us = us_since(t_host, clock_type::now());
        STRIX_CUDA_CHECK(cudaEventRecord(m.ev_comb, stream));
        enqueue_moe_combine(plain, stream);
        STRIX_CUDA_CHECK(cudaEventRecord(m.ev_done, stream));
        t.local_done_before_answers = true;
    } else {
        // one request per server with remote slots: its rows only
        std::vector<xrpc::ExpertsRequest> reqs(servers);
        std::vector<std::vector<std::uint32_t>> row_maps(servers);
        std::vector<std::size_t> used;
        for (std::size_t srv = 0; srv < servers; ++srv) {
            auto& q = reqs[srv];
            for (std::uint32_t r = 0; r < R; ++r) {
                if (!mask[srv][r]) continue;
                row_maps[srv].push_back(r);
                q.ids.insert(q.ids.end(), m.h_ids.get() + r * K, m.h_ids.get() + (r + 1U) * K);
                q.weights.insert(q.weights.end(), m.h_weights.get() + r * K, m.h_weights.get() + (r + 1U) * K);
                q.remote.push_back(mask[srv][r]);
                q.codes.insert(q.codes.end(), m.h_codes.get() + std::size_t{r} * H, m.h_codes.get() + std::size_t{r + 1U} * H);
                q.deltas.insert(q.deltas.end(), m.h_deltas.get() + std::size_t{r} * G, m.h_deltas.get() + std::size_t{r + 1U} * G);
            }
            if (row_maps[srv].empty()) continue;
            q.layer = layer.layer;
            q.rows = static_cast<std::uint32_t>(row_maps[srv].size());
            q.top_k = K;
            q.flags = o.request_flags;
            (void)m.clients[srv]->submit(q);
            t.request_bytes += xrpc::request_bytes(q.rows, K, o.hello.geometry);
            used.push_back(srv);
        }
        const auto t_sent = clock_type::now();
        t.prep_us = us_since(t_host, t_sent);
        t.servers = static_cast<std::uint32_t>(used.size());
        std::vector<xrpc::ResponsePlan> plans(servers);
        for (const auto srv : used) plans[srv] = xrpc::plan_response(reqs[srv], o.hello.geometry, acks_[srv].residency);
        std::vector<xrpc::ResultView> views(servers);
        for (const auto srv : used) {
            auto& p = m.payloads[srv];
            views[srv] = m.clients[srv]->collect(p.bytes);
            if (views[srv].items != plans[srv].items.size() || views[srv].hidden != H || views[srv].layer != layer.layer)
                throw std::runtime_error("ds41 parts-host: the answer does not match its plan");
            t.server_us = std::max(t.server_us, static_cast<double>(views[srv].timing.server_ns) / 1000.0);
            t.answer_bytes += xrpc::result_bytes(views[srv].items, o.hello.geometry);
        }
        const auto t_recv = clock_type::now();
        t.rtt_us = us_since(t_sent, t_recv);
        t.local_done_before_answers = cudaEventQuery(m.ev_local) == cudaSuccess;
        if (!t.local_done_before_answers) (void)cudaGetLastError();

        std::vector<xrpc::Answer> answers;
        std::vector<std::int32_t> base;
        std::int32_t vectors = 0;
        for (const auto srv : used) {
            answers.push_back({&plans[srv], views[srv].vectors, row_maps[srv]});
            base.push_back(vectors);
            vectors += static_cast<std::int32_t>(views[srv].items);
        }
        t.items = static_cast<std::uint32_t>(vectors);
        const auto cp = xrpc::plan_combine(R, K, std::span<const std::uint32_t>(m.h_ids.get(), std::size_t{R} * K), remote, o.split, answers);
        if (used.size() > kMaxAnswers) throw std::runtime_error("ds41 parts-host: more answers than the combine takes");
        const auto index = [](std::int64_t src) { return static_cast<std::int32_t>(src); };  // -1 or answer << 24 | item
        std::int32_t* tab = m.h_tables.get();
        for (std::uint32_t i = 0; i < 2U * R; ++i) {
            tab[i] = index(cp.seed[i]);
            tab[2U * R + i] = static_cast<std::int32_t>(cp.start[i]);
        }
        for (std::uint32_t i = 0; i < R * K; ++i) tab[4U * R + i] = index(cp.source[i]);

        t.pack_us = us_since(t_recv, clock_type::now());
        SeededCombineView c;
        bool all_mapped = o.answers_in_place;
        for (const auto srv : used) all_mapped = all_mapped && m.payloads[srv].device && m.payloads[srv].bytes.data() == m.payloads[srv].registered;
        t.pinned_answers = all_mapped;
        if (all_mapped) {  // the combine reads the tables and the answers in place (mapped host memory)
            for (std::size_t a = 0; a < used.size(); ++a) {
                const auto& p = m.payloads[used[a]];
                const auto offset = reinterpret_cast<const std::uint8_t*>(views[used[a]].vectors) - p.bytes.data();
                c.remote[a] = reinterpret_cast<const float*>(p.device + offset);
            }
            c.tables = m.h_tables.device();
            STRIX_CUDA_CHECK(cudaEventRecord(m.ev_comb, stream));
        } else {
            // local work already done: copy on the stream itself (no cross-stream event); else overlap its tail
            const cudaStream_t hs = t.local_done_before_answers ? stream : m.copy;
            STRIX_CUDA_CHECK(cudaEventRecord(m.ev_h2d_start, hs));
            STRIX_CUDA_CHECK(cudaMemcpyAsync(m.d_tables.get(), tab, seeded_combine_table_ints(R, K) * 4U, cudaMemcpyHostToDevice, hs));
            for (std::size_t a = 0; a < used.size(); ++a) {
                const auto srv = used[a];
                const std::size_t bytes = std::size_t{views[srv].items} * H * 4U;
                float* dst = m.d_remote.get() + std::size_t(base[a]) * H;
                const auto* src = reinterpret_cast<const std::uint8_t*>(views[srv].vectors);
                const auto& p = m.payloads[srv];
                if (p.registered && p.bytes.data() == p.registered) {
                    STRIX_CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, hs));
                } else {  // not page-locked: stage through the pinned buffer
                    auto* stage = m.h_remote.get() + std::size_t(base[a]) * H;
                    std::memcpy(stage, src, bytes);
                    STRIX_CUDA_CHECK(cudaMemcpyAsync(dst, stage, bytes, cudaMemcpyHostToDevice, hs));
                }
                c.remote[a] = dst;
            }
            c.tables = m.d_tables.get();
            STRIX_CUDA_CHECK(cudaEventRecord(m.ev_h2d, hs));
            if (hs != stream) STRIX_CUDA_CHECK(cudaStreamWaitEvent(stream, m.ev_h2d, 0));
            STRIX_CUDA_CHECK(cudaEventRecord(m.ev_comb, stream));
        }
        c.raw = w.raw;
        c.shared = w.shared;
        c.weights = w.weights;
        c.order = scratch.routes.order;
        c.low_counts = scratch.routes.low_counts;
        c.rows = R;
        c.top_k = K;
        c.hidden = H;
        c.out = w.y;
        enqueue_seeded_combine(c, stream);
        STRIX_CUDA_CHECK(cudaEventRecord(m.ev_done, stream));
        (void)cudaStreamQuery(stream);  // flush
        t.plan_us = us_since(t_recv, clock_type::now());
        if (wait) {
            STRIX_CUDA_CHECK(cudaEventSynchronize(m.ev_done));
            if (!all_mapped) t.h2d_us = elapsed_us(m.ev_h2d_start, m.ev_h2d);
        }
        t.local_overlap_us = -1.0;  // filled below once the GPU spans are known
        if (wait) {
            const double d2h = elapsed_us(w.router_ready, m.ev_d2h), local = elapsed_us(w.router_ready, m.ev_local);
            t.local_overlap_us = std::min(local, d2h + us_since(t_host, t_recv));
        }
    }
    if (wait) {
        STRIX_CUDA_CHECK(cudaEventSynchronize(m.ev_done));
        t.d2h_us = elapsed_us(w.router_ready, m.ev_d2h);
        t.local_us = elapsed_us(w.router_ready, m.ev_local);
        t.combine_us = elapsed_us(m.ev_comb, m.ev_done);
        t.total_us = elapsed_us(w.router_ready, m.ev_done);
        if (t.remote_slots == 0U) t.local_overlap_us = 0.0;
    }
    return t;
}

}  // namespace strix::ds41::cuda
