// deepseek41 CUDA parts-host MoE site for graph-captured R1 decode steps (cuda_ds41_hot_site.hpp).

#include "strix/cuda/ds41_hot_site.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <stdexcept>
#include <string>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <avrt.h>
#include <timeapi.h>
#pragma comment(lib, "avrt.lib")
#pragma comment(lib, "winmm.lib")
#endif

namespace strix::ds41::cuda {

namespace {

constexpr std::uint32_t kK = 8;           // ids / weights slots per layer in the export area (top_k <= 8)
constexpr std::uint32_t kTableInts = 16;  // packed tables of one R1 window (4 + top_k <= 16)
constexpr unsigned long long kTimeoutNs = 10ULL * 1000ULL * 1000ULL * 1000ULL;

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

struct ExportArgs {
    std::uint32_t layer = 0, top_k = 0, hidden = 0;
    const std::uint32_t* ids = nullptr;
    const float* weights = nullptr;
    const std::int8_t* codes = nullptr;
    const float* deltas = nullptr;
    const std::int32_t* slot_of = nullptr;
    const volatile std::uint32_t* required = nullptr;     // mapped [layers]
    const volatile std::uint32_t* refill_done = nullptr;  // mapped [layers]
    const std::uint32_t* seq = nullptr;                   // device
    std::uint32_t* host_ids = nullptr;                    // mapped [kK]
    float* host_weights = nullptr;
    std::int8_t* host_codes = nullptr;
    float* host_deltas = nullptr;
    std::uint32_t* host_mask = nullptr;
    volatile std::uint32_t* flag = nullptr;
    std::uint32_t* any_remote = nullptr;  // device [layers]
    std::uint32_t* errors = nullptr;      // device
};

__global__ void site_begin_kernel(std::uint32_t* seq) { *seq = *seq + 1U; }

__global__ __launch_bounds__(256) void site_export_kernel(ExportArgs a) {
    __shared__ std::uint32_t mask;
    const std::uint32_t t = threadIdx.x;
    if (t == 0U) {
        // a dynamic-slot refill of this layer must be in before its table is read
        const unsigned long long t0 = gtimer();
        while (static_cast<std::int32_t>(a.refill_done[a.layer] - a.required[a.layer]) < 0)
            if (gtimer() - t0 > kTimeoutNs) { atomicAdd(a.errors, 1U); break; }
        mask = 0U;
    }
    __syncthreads();
    if (t < a.top_k) {
        const std::uint32_t id = a.ids[t];
        a.host_ids[t] = id;
        a.host_weights[t] = a.weights[t];
        if (a.slot_of[id] >= 0) atomicOr(&mask, 1U << t);
    }
    const std::uint32_t words = a.hidden / 16U, groups = a.hidden / 32U;
    const auto* src = reinterpret_cast<const int4*>(a.codes);
    auto* dst = reinterpret_cast<int4*>(a.host_codes);
    for (std::uint32_t i = t; i < words; i += 256U) dst[i] = src[i];
    for (std::uint32_t i = t; i < groups; i += 256U) a.host_deltas[i] = a.deltas[i];
    __syncthreads();
    if (t == 0U) {
        const std::uint32_t full = (1U << a.top_k) - 1U;
        *a.host_mask = mask;
        a.any_remote[a.layer] = mask != full ? 1U : 0U;
        __threadfence_system();
        *a.flag = *a.seq;
    }
}

__global__ void site_wait_kernel(const std::uint32_t* any_remote, std::uint32_t layer, const volatile std::uint32_t* ans_flag, const std::uint32_t* seq,
                                 std::uint32_t* errors) {
    if (!any_remote[layer]) return;
    const std::uint32_t s = *seq;
    const unsigned long long t0 = gtimer();
    while (static_cast<std::int32_t>(*ans_flag - s) < 0)
        if (gtimer() - t0 > kTimeoutNs) { atomicAdd(errors, 1U); break; }
    __threadfence_system();
}

__global__ void site_select_kernel(const std::uint32_t* any_remote, std::uint32_t layer, const std::int32_t* staged, std::int32_t* tables, std::uint32_t n) {
    const std::uint32_t i = threadIdx.x;
    if (i >= n) return;
    // all local: no seed (-1), start 0, every source local (-1) = the one-device combine
    tables[i] = any_remote[layer] ? staged[i] : (i >= 2U && i < 4U ? 0 : -1);
}

__global__ void site_done_kernel(volatile std::uint32_t* done, const std::uint32_t* seq) {
    __threadfence_system();
    *done = *seq;
}

template <class T>
T* mapped(T* host) {
    void* d = nullptr;
    STRIX_CUDA_CHECK(cudaHostGetDevicePointer(&d, host, 0));
    return static_cast<T*>(d);
}

using clock_type = std::chrono::steady_clock;
double us_since(clock_type::time_point a, clock_type::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }

}  // namespace

struct HotSite::Impl {
    HotResidency& res;
    PartsHostOptions o;
    std::uint32_t L = 0, H = 0, G = 0, K = 0;
    std::size_t stage_floats = 0;
    // mapped pinned host memory
    std::uint32_t *h_ids = nullptr, *h_mask = nullptr, *h_flag = nullptr, *h_ans = nullptr, *h_done = nullptr;
    float *h_w = nullptr, *h_deltas = nullptr, *h_stage = nullptr;
    std::int8_t* h_codes = nullptr;
    // device
    DeviceBuffer<std::uint32_t> d_seq, d_any, d_err;
    DeviceBuffer<float> d_stage;
    DeviceBuffer<std::int32_t> d_tables;
    DeviceBuffer<std::uint8_t> workspace;
    std::vector<std::unique_ptr<xrpc::ExpertClient>> clients;
    std::vector<std::vector<std::uint8_t>> payloads;
    // service thread
    std::thread th;
    std::atomic<bool> stopping{false};
    std::atomic<std::uint32_t> served{0};
    std::uint32_t first = 0, count = 0;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<HotSiteLayerTiming> last;
    std::string error;

    Impl(HotResidency& r, const PartsHostOptions& opt, std::uint32_t layers) : res(r), o(opt), L(layers) {
        H = o.shape.hidden;
        G = H / 32U;
        K = o.shape.top_k;
        if (K > kK || 4U + K > kTableInts) throw std::invalid_argument("ds41 hot site: top_k");
        stage_floats = kTableInts + std::size_t{K} * H;
        const auto alloc = [](auto*& p, std::size_t n) {
            STRIX_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&p), n * sizeof(*p), cudaHostAllocMapped | cudaHostAllocPortable));
            std::memset(p, 0, n * sizeof(*p));
        };
        alloc(h_ids, std::size_t{L} * kK);
        alloc(h_w, std::size_t{L} * kK);
        alloc(h_mask, L);
        alloc(h_flag, L);
        alloc(h_ans, L);
        alloc(h_done, L);
        alloc(h_codes, std::size_t{L} * H);
        alloc(h_deltas, std::size_t{L} * G);
        alloc(h_stage, std::size_t{L} * stage_floats);
        d_seq.resize(1);
        d_seq.zero();
        d_any.resize(L);
        d_any.zero();
        d_err.resize(1);
        d_err.zero();
        d_stage.resize(stage_floats);
        d_tables.resize(kTableInts);
        workspace.resize(moe_layer_workspace_bytes(1, o.shape));
    }
    ~Impl() {
        for (auto* p : {static_cast<void*>(h_ids), static_cast<void*>(h_w), static_cast<void*>(h_mask), static_cast<void*>(h_flag), static_cast<void*>(h_ans),
                        static_cast<void*>(h_done), static_cast<void*>(h_codes), static_cast<void*>(h_deltas), static_cast<void*>(h_stage)})
            if (p) cudaFreeHost(p);
    }

    static void boost_thread() {
#ifdef _WIN32
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
        timeBeginPeriod(1);
        DWORD task = 0;
        (void)AvSetMmThreadCharacteristicsW(L"Pro Audio", &task);
#endif
    }
    void serve() {
        boost_thread();
        std::vector<HotSiteLayerTiming> tl(count);
        std::uint32_t seq = 0;
        const std::size_t servers = clients.size();
        // heartbeat: a PING to every server every heartbeat_us while waiting (keeps the NIC / CPU paths warm); the
        // PONGs are read just before the next RESULT of that connection (answers come in order), at most 8 in flight
        std::vector<std::uint32_t> pongs(servers, 0U);
        std::vector<clock_type::time_point> last_send(servers, clock_type::now());
        const auto heartbeat = [&]() {
            if (heartbeat_us <= 0) return;
            const auto now = clock_type::now();
            for (std::size_t s = 0; s < servers; ++s)
                if (pongs[s] < 8U && us_since(last_send[s], now) >= heartbeat_us) {
                    clients[s]->connection().send(xrpc::MsgType::ping, 0, {});
                    ++pongs[s];
                    last_send[s] = now;
                }
        };
        const auto drain = [&](std::size_t s) {
            for (; pongs[s] > 0U; --pongs[s]) {
                const auto m = clients[s]->connection().receive(1U << 20U);
                if (m.header.type != xrpc::MsgType::pong) throw std::runtime_error("ds41 hot site: expected PONG");
            }
        };
        while (!stopping.load()) {
            ++seq;
            res.begin_token();
            auto t_prev = clock_type::now();
            for (std::uint32_t i = 0; i < count; ++i) {
                const std::uint32_t l = first + i;
                HotSiteLayerTiming t;
                const volatile std::uint32_t* f = h_flag + l;
                std::uint32_t spins = 0;
                while (*f != seq) {
                    if (stopping.load()) return;
                    cpu_pause();
                    if ((++spins & 255U) == 0U) heartbeat();
                }
                std::atomic_thread_fence(std::memory_order_acquire);
                const auto t_seen = clock_type::now();
                t.export_seen_us = us_since(t_prev, t_seen);
                const std::uint32_t full = (1U << K) - 1U, local = h_mask[l] & full, remote = full & ~local;
                const std::uint32_t* ids = h_ids + std::size_t{l} * kK;
                t.local = static_cast<std::uint32_t>(popcount32(local));
                t.remote = K - t.local;
                if (remote) {
                    std::vector<xrpc::ExpertsRequest> reqs(servers);
                    std::vector<std::uint32_t> masks(servers, 0U);
                    for (std::uint32_t k = 0; k < K; ++k) {
                        if (!(remote >> k & 1U)) continue;
                        std::size_t s = 0;
                        while (s < servers && !acks[s].residency.serves(l, ids[k])) ++s;
                        if (s == servers) throw std::runtime_error("ds41 hot site: expert neither resident nor served");
                        masks[s] |= 1U << k;
                    }
                    std::vector<std::size_t> used;
                    const auto t_prep = clock_type::now();
                    for (std::size_t s = 0; s < servers; ++s) {
                        if (!masks[s]) continue;
                        auto& q = reqs[s];
                        q.layer = l;
                        q.rows = 1;
                        q.top_k = K;
                        q.flags = o.request_flags;
                        q.ids.assign(ids, ids + K);
                        q.weights.assign(h_w + std::size_t{l} * kK, h_w + std::size_t{l} * kK + K);
                        q.remote.assign(1, masks[s]);
                        q.codes.assign(h_codes + std::size_t{l} * H, h_codes + std::size_t{l + 1U} * H);
                        q.deltas.assign(h_deltas + std::size_t{l} * G, h_deltas + std::size_t{l + 1U} * G);
                        (void)clients[s]->submit(q);
                        last_send[s] = clock_type::now();
                        used.push_back(s);
                    }
                    const auto t_sent = clock_type::now();
                    t.prep_us = us_since(t_prep, t_sent);
                    t.servers = static_cast<std::uint32_t>(used.size());
                    std::vector<xrpc::ResponsePlan> plans(servers);
                    for (const auto s : used) plans[s] = xrpc::plan_response(reqs[s], o.hello.geometry, acks[s].residency);
                    std::vector<xrpc::ResultView> views(servers);
                    for (const auto s : used) {
                        drain(s);
                        views[s] = clients[s]->collect(payloads[s]);
                        if (views[s].items != plans[s].items.size() || views[s].hidden != H || views[s].layer != l)
                            throw std::runtime_error("ds41 hot site: the answer does not match its plan");
                        t.server_us = std::max(t.server_us, static_cast<double>(views[s].timing.server_ns) / 1000.0);
                        t.answer_bytes += xrpc::result_bytes(views[s].items, o.hello.geometry);
                    }
                    const auto t_recv = clock_type::now();
                    t.rtt_us = us_since(t_sent, t_recv);
                    std::vector<xrpc::Answer> answers;
                    std::vector<std::int32_t> base;
                    std::int32_t vectors = 0;
                    const std::vector<std::uint32_t> row_map{0U};
                    for (const auto s : used) {
                        answers.push_back({&plans[s], views[s].vectors, row_map});
                        base.push_back(vectors);
                        vectors += static_cast<std::int32_t>(views[s].items);
                    }
                    if (static_cast<std::uint32_t>(vectors) > K) throw std::runtime_error("ds41 hot site: more answer vectors than slots");
                    t.items = static_cast<std::uint32_t>(vectors);
                    const std::vector<std::uint32_t> rem{remote};
                    const auto cp = xrpc::plan_combine(1, K, std::span<const std::uint32_t>(ids, K), rem, o.split, answers);
                    // one answer area: (a, i) -> base[a] + i
                    const auto remap = [&](std::int64_t src) -> std::int32_t {
                        if (src < 0) return -1;
                        return base[static_cast<std::size_t>(src >> 24)] + static_cast<std::int32_t>(src & 0xffffff);
                    };
                    float* stage = h_stage + std::size_t{l} * stage_floats;
                    auto* tab = reinterpret_cast<std::int32_t*>(stage);
                    tab[0] = remap(cp.seed[0]);
                    tab[1] = remap(cp.seed[1]);
                    tab[2] = static_cast<std::int32_t>(cp.start[0]);
                    tab[3] = static_cast<std::int32_t>(cp.start[1]);
                    for (std::uint32_t k = 0; k < K; ++k) tab[4U + k] = remap(cp.source[k]);
                    for (std::size_t a = 0; a < used.size(); ++a)
                        std::memcpy(stage + kTableInts + std::size_t(base[a]) * H, views[used[a]].vectors, std::size_t{views[used[a]].items} * H * 4U);
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    reinterpret_cast<volatile std::uint32_t*>(h_ans)[l] = seq;
                    t.plan_us = us_since(t_recv, clock_type::now());
                }
                const auto before = res.token_stats();
                res.after_layer_mapped(l, ids, K, mapped(h_done) + l, seq);
                t.copies = res.token_stats().copies - before.copies;
                t.hits = res.token_stats().dynamic_hits - before.dynamic_hits;
                tl[i] = t;
                t_prev = clock_type::now();
            }
            {
                std::lock_guard<std::mutex> g(mu);
                last = tl;
                served.store(seq);
            }
            cv.notify_all();
        }
    }
    static void cpu_pause() {
#if defined(_MSC_VER)
        _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
    }
    static int popcount32(std::uint32_t v) {
        int n = 0;
        while (v) { n += static_cast<int>(v & 1U); v >>= 1U; }
        return n;
    }
    std::vector<xrpc::HelloAck> acks;
    double heartbeat_us = 200.0;
};

HotSite::HotSite(HotResidency& residency, const PartsHostOptions& options, std::uint32_t layers)
    : impl_(std::make_unique<Impl>(residency, options, layers)) {
    auto& m = *impl_;
    if (m.o.hello.client.empty()) m.o.hello.client = "cuda-hot-site";
    if (m.o.hello.numeric_identity.empty()) m.o.hello.numeric_identity = "ds41-exact-v1";
    const std::size_t result_cap = xrpc::kResultHeadBytes + std::size_t{m.K} * m.H * 4U;
    for (const auto& ep : m.o.endpoints) {
        m.clients.push_back(std::make_unique<xrpc::ExpertClient>(ep, m.o.hello, m.o.connect_wait, m.o.io_timeout, m.o.busy_poll));
        acks_.push_back(m.clients.back()->ack());
        m.payloads.emplace_back();
        m.payloads.back().reserve(result_cap);
    }
    m.acks = acks_;
}

HotSite::~HotSite() { stop(); }

void HotSite::enqueue(std::uint32_t l, const A8Rows& x, const std::uint32_t* ids, const float* weights, float* shared, float* raw, float* y,
                      const std::function<void(cudaStream_t)>& shared_expert, cudaStream_t st, bool begin_token) {
    auto& m = *impl_;
    if (x.rows != 1U) throw std::invalid_argument("ds41 hot site: R1 windows only");
    const auto& s = m.o.shape;
    const ResidentLayer& rl = m.res.layer(l);
    if (begin_token) {
        site_begin_kernel<<<1, 1, 0, st>>>(m.d_seq.get());
        STRIX_CUDA_CHECK(cudaGetLastError());
    }
    ExportArgs a;
    a.layer = l;
    a.top_k = m.K;
    a.hidden = m.H;
    a.ids = ids;
    a.weights = weights;
    a.codes = x.codes;
    a.deltas = x.deltas;
    a.slot_of = rl.slot_of;
    a.required = m.res.required_dev();
    a.refill_done = m.res.refill_done_dev();
    a.seq = m.d_seq.get();
    a.host_ids = mapped(m.h_ids) + std::size_t{l} * kK;
    a.host_weights = mapped(m.h_w) + std::size_t{l} * kK;
    a.host_codes = mapped(m.h_codes) + std::size_t{l} * m.H;
    a.host_deltas = mapped(m.h_deltas) + std::size_t{l} * m.G;
    a.host_mask = mapped(m.h_mask) + l;
    a.flag = mapped(m.h_flag) + l;
    a.any_remote = m.d_any.get();
    a.errors = m.d_err.get();
    site_export_kernel<<<1, 256, 0, st>>>(a);
    STRIX_CUDA_CHECK(cudaGetLastError());
    // local slots (the device table decides) + the shared expert, overlapping the host's RPC
    const MoeLayerScratch scratch = moe_layer_workspace_views(m.workspace.get(), 1, s);
    ResidentRoutesView rv;
    rv.ids = ids;
    rv.slot_of = rl.slot_of;
    rv.rows = 1;
    rv.top_k = m.K;
    rv.experts = s.experts;
    rv.split = m.o.split;
    rv.out = scratch.routes;
    enqueue_resident_routes(rv, st);
    RoutedGateUpView g;
    g.shape = s;
    g.weights = rl.weights;
    g.routes = scratch.routes;
    g.max_union = m.K;
    g.max_rows = 1;
    g.x = x;
    g.act = scratch.act;
    enqueue_routed_gate_up(g, st);
    RoutedDownView d;
    d.shape = s;
    d.weights = rl.weights;
    d.routes = scratch.routes;
    d.max_union = m.K;
    d.max_rows = 1;
    d.act = scratch.act;
    d.raw = raw;
    enqueue_routed_down(d, st);
    if (shared_expert) shared_expert(st);
    // the answers (only at layers with remote slots), then the combine
    site_wait_kernel<<<1, 1, 0, st>>>(m.d_any.get(), l, mapped(m.h_ans) + l, m.d_seq.get(), m.d_err.get());
    STRIX_CUDA_CHECK(cudaGetLastError());
    STRIX_CUDA_CHECK(cudaMemcpyAsync(m.d_stage.get(), m.h_stage + std::size_t{l} * m.stage_floats, m.stage_floats * 4U, cudaMemcpyHostToDevice, st));
    site_select_kernel<<<1, 32, 0, st>>>(m.d_any.get(), l, reinterpret_cast<const std::int32_t*>(m.d_stage.get()), m.d_tables.get(), 4U + m.K);
    STRIX_CUDA_CHECK(cudaGetLastError());
    SeededCombineView c;
    c.raw = raw;
    c.remote[0] = m.d_stage.get() + kTableInts;
    c.shared = shared;
    c.weights = weights;
    c.order = scratch.routes.order;
    c.low_counts = scratch.routes.low_counts;
    c.tables = m.d_tables.get();
    c.rows = 1;
    c.top_k = m.K;
    c.hidden = m.H;
    c.out = y;
    enqueue_seeded_combine(c, st);
    site_done_kernel<<<1, 1, 0, st>>>(mapped(m.h_done) + l, m.d_seq.get());
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void HotSite::start(std::uint32_t first, std::uint32_t layers_per_token) {
    auto& m = *impl_;
    stop();
    m.first = first;
    m.count = layers_per_token;
    m.stopping = false;
    m.th = std::thread([&m] {
        try {
            m.serve();
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> g(m.mu);
            m.error = e.what();
            m.cv.notify_all();
        }
    });
}

void HotSite::set_heartbeat_us(double us) { impl_->heartbeat_us = us; }

void HotSite::stop() {
    auto& m = *impl_;
    m.stopping = true;
    if (m.th.joinable()) m.th.join();
}

std::vector<HotSiteLayerTiming> HotSite::last_token() {
    auto& m = *impl_;
    std::uint32_t want = 0;
    STRIX_CUDA_CHECK(cudaMemcpy(&want, m.d_seq.get(), 4, cudaMemcpyDeviceToHost));
    std::unique_lock<std::mutex> g(m.mu);
    m.cv.wait_for(g, std::chrono::seconds(30), [&] { return m.served.load() >= want || !m.error.empty(); });
    if (!m.error.empty()) throw std::runtime_error("ds41 hot site service: " + m.error);
    return m.last;
}

std::uint32_t HotSite::errors() const noexcept {
    std::uint32_t e = 0;
    (void)cudaMemcpy(&e, impl_->d_err.get(), 4, cudaMemcpyDeviceToHost);
    return e;
}

}  // namespace strix::ds41::cuda
