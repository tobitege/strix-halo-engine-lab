// Device side of ds41_trellis.hpp: the trellis library context, prepared sites, per-stream workspaces and the linear.
#include "strix/ds41_trellis.hpp"
#include "strix/hip_check.hpp"
#include "strix/ds41_exact_flags.hpp"

#include "strix/ds41_moe.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#if defined(STRIX_DS41_TRELLIS_LIB)
#include <strix_trellis.h>
#if defined(STRIX_DS41_TRELLIS_PREPARED) && (!defined(STRIX_TRELLIS_VERSION_MINOR) || STRIX_TRELLIS_VERSION_MINOR < 3)
#error "Prepared TP adapter requires the 0.3 trellis header and runtime"
#endif
#endif

namespace strix::ds41 {
namespace {

void check(hipError_t e, const char* what) {
    ::strix::hip_check_prefixed("ds41 trellis: ", e, what);
}
hipStream_t as_stream(void* s) { return static_cast<hipStream_t>(s); }
unsigned blocks_for(std::size_t n, unsigned block) { return static_cast<unsigned>((n + block - 1U) / block); }

__global__ void swiglu_f32_kernel(const float* __restrict__ gate, const float* __restrict__ up, std::size_t n, float limit,
                                  float* __restrict__ act) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < n) act[i] = swiglu_clamped(gate[i], up[i], limit);
}
#if defined(STRIX_DS41_TRELLIS_LIB)
// y[r][i] = y[r][i] + t[r][i] (t contiguous rows of n values)
__global__ void accumulate_rows_kernel(float* __restrict__ y, std::size_t y_stride, const float* __restrict__ t, std::uint32_t n,
                                       std::uint32_t rows) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i >= std::size_t{n} * rows) return;
    const std::size_t r = i / n, c = i % n;
    y[r * y_stride + c] = y[r * y_stride + c] + t[i];
}
#endif

}  // namespace

void enqueue_swiglu_f32(const float* gate, const float* up, std::uint32_t rows, std::uint32_t width, float limit, float* act,
                        void* stream) {
    const std::size_t n = std::size_t{rows} * width;
    if (n == 0U) return;
    hipLaunchKernelGGL(swiglu_f32_kernel, dim3(blocks_for(n, 256)), dim3(256), 0, as_stream(stream), gate, up, n, limit, act);
    check(hipGetLastError(), "swiglu f32");
}

// Identity table (v12 library), keyed on (I, N, K): measured R classes on real ours-K3 / K4 / K5 tensors. Only K5 has
// R-dependent shapes; every K3 and K4 shape of every class (q_a, kv, q_b rank rows, o_b K halves, o_a groups, shared
// w1/w3/w2 and their 1152 TP slices, head rank slices) is one class R1..R8 (R-class run). Shapes not listed take
// every count 1..8.
std::uint32_t trellis_submit_sizes(std::uint32_t columns, std::uint32_t rows, std::uint32_t bits) noexcept {
    const auto set = [](std::initializer_list<std::uint32_t> counts) {
        std::uint32_t m = 0;
        for (const auto c : counts) m |= 1U << (c - 1U);
        return m;
    };
    if (bits == 5U && columns == 4096U && rows == 1024U) return set({1, 4, 8});  // wo_a group: split 16 at R 1, 4, 8 (32 else)
    if (bits == 5U && columns == 5120U && rows == 2304U) return set({2, 3, 4, 5, 6, 7});  // shared w1 / w3: R1 | R2..R7 | R8
    if (bits == 5U && columns == 2304U && rows == 5120U) return set({2, 3, 4, 5, 6, 7, 8});  // shared w2: R1 | R2..R8
    return 0xffU;
}
bool trellis_headout_enabled() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("STRIX_DS41_TRELLIS_HEADOUT");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}
bool trellis_dense_full_enabled() noexcept {
    static const bool on = [] { const auto* e = std::getenv("STRIX_DS41_TRELLIS_DENSE_FULL"); return e && e[0] == '1'; }();
    return on;
}
bool trellis_dense_row_slices_enabled() noexcept {
#if defined(STRIX_DS41_TRELLIS_SPLITK)
    return trellis_dense_full_enabled();
#else
    return false;
#endif
}
bool trellis_qakv_pair_enabled() noexcept {
    // default on (=0 off): the pair has the bits of the individual q_a / kv submissions
    static const bool on = [] { const auto* e = std::getenv("STRIX_DS41_TRELLIS_QAKV_PAIR"); return !(e && e[0] == '0'); }();
    return on;
}
bool trellis_row_views_enabled() noexcept {
#if defined(STRIX_DS41_TRELLIS_DENSE_R1)
    return trellis_row_views();  // default on, STRIX_DS41_TRELLIS_ROW_VIEWS=0 off (ds41_exact_flags.hpp)
#else
    return false;
#endif
}
namespace {
// STRIX_DS41_SHARED_GROUPED read once: -1 unset/empty (implicit default), 1 explicit on, 0 any other value (off).
int shared_grouped_env() noexcept {
    static const int v = [] {
        const auto* e = std::getenv("STRIX_DS41_SHARED_GROUPED");
        if (e == nullptr || e[0] == '\0') return -1;
        return e[0] == '1' ? 1 : 0;
    }();
    return v;
}
// Implicit default: on only after a startup admission found the configuration admissible; any inadmissible admission
// in the process switches it off for good (both routes are bit-identical, so this only selects the faster one).
std::atomic<bool> shared_grouped_implicit{false};
std::atomic<bool> shared_grouped_implicit_refused{false};
} // namespace
bool trellis_shared_grouped_enabled() noexcept {
    // explicit =1 on; unset = on where admitted (trellis_shared_grouped_admit); any other value off. Only a dense-r1
    // build linking a library with the declared SwiGLU pair epilogue can run it.
#if defined(STRIX_DS41_TRELLIS_LIB) && defined(STRIX_DS41_TRELLIS_DENSE_R1) && defined(STRIX_TP_SWIGLU_CLAMPED_DECLARED)
    static const bool explicit_on = [] {
        const bool v = shared_grouped_env() == 1;
        if (v) std::fprintf(stderr, "ds41 trellis: shared gate/up as SwiGLU pair launches (STRIX_DS41_SHARED_GROUPED=1)\n");
        return v;
    }();
    return explicit_on || shared_grouped_implicit.load(std::memory_order_relaxed);
#else
    return false;
#endif
}
void trellis_shared_grouped_admit(bool shexp_tp, bool halves_route, std::uint32_t row_limit) {
    const int env = shared_grouped_env();
    if (env == 0) return;
    if (env < 0) {
#if defined(STRIX_DS41_TRELLIS_LIB) && defined(STRIX_DS41_TRELLIS_DENSE_R1) && defined(STRIX_TP_SWIGLU_CLAMPED_DECLARED)
        const char* why = shexp_tp ? "STRIX_DS41_SHEXP_TP=1"
                        : !halves_route ? "no K4 shared row-halves route"
                        : row_limit > 8U ? "more than 8 rows per step" : nullptr;
#else
        const char* why = "build without the declared SwiGLU pair";
#endif
        if (why != nullptr) {
            shared_grouped_implicit.store(false, std::memory_order_relaxed);
            if (!shared_grouped_implicit_refused.exchange(true))
                std::fprintf(stderr, "ds41 trellis: shared gate/up SwiGLU pair launches not admitted by default (%s): old path\n", why);
        } else if (!shared_grouped_implicit_refused.load() && !shared_grouped_implicit.exchange(true)) {
            std::fprintf(stderr, "ds41 trellis: shared gate/up as SwiGLU pair launches (default; STRIX_DS41_SHARED_GROUPED=0 off)\n");
        }
        return;
    }
    if (!trellis_shared_grouped_enabled())
        throw std::invalid_argument("ds41 trellis: STRIX_DS41_SHARED_GROUPED=1 needs a dense-r1 build linking the declared SwiGLU pair");
    if (shexp_tp) throw std::invalid_argument("ds41 trellis: STRIX_DS41_SHARED_GROUPED=1 is refused with STRIX_DS41_SHEXP_TP=1");
    if (!halves_route)
        throw std::invalid_argument("ds41 trellis: STRIX_DS41_SHARED_GROUPED=1 needs the K4 shared row-halves route "
                                    "(trellis shexp K4, STRIX_DS41_TRELLIS_DENSE_FULL=1)");
    if (row_limit > 8U)
        throw std::invalid_argument("ds41 trellis: STRIX_DS41_SHARED_GROUPED=1 is not qualified with more than 8 rows per step "
                                    "(STRIX_DS41_BATCH_ROWS16)");
}
bool trellis_site_span(const TrellisSite& site, const void** data, std::uint64_t* bytes) noexcept {
#if defined(STRIX_DS41_TRELLIS_LIB) && defined(STRIX_TRELLIS_HAS_WEIGHTS_SPAN)
    std::uint64_t n = 0;
    if (site.weights == nullptr || strix_weights_device_span(static_cast<strix_weights>(site.weights), data, &n) != STRIX_SUCCESS) return false;
    *bytes = n;
    return true;
#else
    (void)site;
    (void)data;
    (void)bytes;
    return false;
#endif
}
std::uint32_t trellis_submit_sizes(const TrellisSite& site) noexcept { return trellis_submit_sizes(site.columns, site.rows, site.bits); }
std::uint32_t trellis_chunk_rows(const TrellisSite& site) noexcept {
    const auto m = trellis_submit_sizes(site);
    std::uint32_t c = 1;
    for (std::uint32_t k = 1; k <= 8U; ++k)
        if ((m >> (k - 1U)) & 1U) c = k;
    return c;
}
std::uint32_t trellis_padded_rows(const TrellisSite& site, std::uint32_t n) noexcept {
    const auto m = trellis_submit_sizes(site);
    for (std::uint32_t k = n; k <= 8U; ++k)
        if ((m >> (k - 1U)) & 1U) return k;
    return trellis_chunk_rows(site);
}

#if defined(STRIX_DS41_TRELLIS_LIB)

namespace {
void lib_check(strix_status s, const std::string& what) {
    if (s != STRIX_SUCCESS) throw std::runtime_error("ds41 trellis: " + what + ": " + strix_status_string(s));
}
strix_stream lib_stream(hipStream_t s) {
    strix_stream st{};
    st.struct_size = sizeof(st);
    st.abi_version = STRIX_TRELLIS_ABI_VERSION;
    st.hip_stream = s;
    return st;
}
void set_tensor(strix_tensor_desc& t, void* data, std::uint32_t rows, std::uint32_t width, std::size_t stride = 0) {
    if (stride == 0) stride = width;
    t.data = data;
    t.buffer_bytes = (std::uint64_t{rows - 1U} * stride + width) * sizeof(float);
    t.dtype = STRIX_DTYPE_F32;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = width;
    t.strides[0] = stride;
    t.strides[1] = 1;
}
}  // namespace

struct TrellisContext::Impl {
    strix_context context = nullptr;
    std::deque<TrellisSite> sites;
#if defined(STRIX_DS41_TRELLIS_PREPARED)
    using PlanKey = std::tuple<std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uint32_t, std::size_t,
                               std::size_t>;  // ..., rows, x row stride, y row stride
    struct Plan { strix_linear_plan linear = nullptr; strix_tp_dense_plan tp = nullptr; };
    std::map<PlanKey, Plan> plans;
#endif
#if defined(STRIX_DS41_TRELLIS_SPLITK)
    using SplitKey = std::tuple<std::uintptr_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>;
    struct HipFree {
        void operator()(void* p) const noexcept { if (p != nullptr) (void)hipFree(p); }
    };
    struct SplitWorkspace {
        std::unique_ptr<void, HipFree> data;
        std::uint64_t bytes = 0;
    };
    std::map<SplitKey, SplitWorkspace> split_workspaces;
    using PairKey = std::tuple<std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t,
                               std::uintptr_t, std::uintptr_t, std::uint32_t>;
    std::map<PairKey, strix_tp_dense_pair_plan> pair_plans;
#if defined(STRIX_DS41_TRELLIS_DENSE_R1)
    std::map<PairKey, strix_tp_dense_plan> pair_r1_plans;  // dense-r1 q_a + kv in one dispatch (LEGACY partitions)
#endif
    using PairWorkspaceKey = std::tuple<std::uintptr_t, std::uint32_t, std::uint32_t, std::uint32_t>;
    std::map<PairWorkspaceKey, SplitWorkspace> pair_workspaces;
#if defined(STRIX_DS41_TRELLIS_DENSE_R1) && defined(STRIX_TP_SWIGLU_CLAMPED_DECLARED)
    // STRIX_DS41_SHARED_GROUPED: shared gate/up SwiGLU pair plans keyed like the q_a+kv pair (gate, up, x, act half,
    // 0, stream, rows); their counter workspaces per (stream, gate bits, up bits, rows), separate from the q_a+kv ones.
    std::map<PairKey, strix_tp_swiglu_plan> swiglu_plans;
    std::map<PairWorkspaceKey, SplitWorkspace> swiglu_workspaces;
#endif
#endif
#if defined(STRIX_LINEAR_TRANSFORMS_FUSED_DECLARED)
    // STRIX_DS41_SHARED_DOWN_FUSED: declared single-launch shared down plans (weights, act, out, workspace, stream, rows)
    using SharedDownKey = std::tuple<std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uint32_t>;
    std::map<SharedDownKey, strix_linear_plan> shared_down_plans;
#endif
#if defined(STRIX_DS41_TRELLIS_SPLITK)
    const SplitWorkspace& split_workspace(const TrellisSite& site, std::uint32_t rows, hipStream_t stream, std::uint64_t bytes) {
        const SplitKey key{reinterpret_cast<std::uintptr_t>(stream), site.columns, site.rows, site.bits, rows};
        if (const auto it = split_workspaces.find(key); it != split_workspaces.end()) {
            if (it->second.bytes != bytes) throw std::logic_error("ds41 trellis: inconsistent split-K workspace layout");
            return it->second;
        }
        void* p = nullptr;
        check(hipMalloc(&p, bytes), "split-K persistent workspace");
        SplitWorkspace owned{std::unique_ptr<void, HipFree>(p), bytes};
        return split_workspaces.emplace(key, std::move(owned)).first->second;
    }
#endif
    struct Workspace {
        hipStream_t stream = nullptr;
        void* lib = nullptr;
        std::uint64_t lib_bytes = 0;
        float* xs = nullptr;  // 8 rows x max I (strided or padded input rows)
        float* ys = nullptr;  // 8 rows x max N (strided or padded output rows, accumulate)
        std::uint64_t staging = 0;  // bytes of xs + ys
    };
    std::vector<Workspace> workspaces;
    const Workspace& workspace(hipStream_t s) const {
        for (const auto& w : workspaces)
            if (w.stream == s) return w;
        throw std::logic_error("ds41 trellis: no workspace reserved for this stream");
    }
};

TrellisContext::TrellisContext() : impl_(std::make_unique<Impl>()) {
    int device = 0;
    check(hipGetDevice(&device), "device");
    strix_context_desc d{};
    d.struct_size = sizeof(d);
    d.abi_version = STRIX_TRELLIS_ABI_VERSION;
    d.device_ordinal = device;
    lib_check(strix_context_create(&d, &impl_->context), "context create");
}

TrellisContext::~TrellisContext() {
    if (!impl_) return;
    (void)hipDeviceSynchronize();
#if defined(STRIX_LINEAR_TRANSFORMS_FUSED_DECLARED)
    for (const auto& [key, plan] : impl_->shared_down_plans) {
        (void)key;
        (void)strix_linear_plan_destroy(plan);
    }
#endif
#if defined(STRIX_DS41_TRELLIS_PREPARED)
    for (const auto& [key, p] : impl_->plans) {
        (void)key;
        if (p.tp) (void)strix_tp_dense_plan_destroy(p.tp);
        if (p.linear) (void)strix_linear_plan_destroy(p.linear);
    }
#endif
#if defined(STRIX_DS41_TRELLIS_SPLITK)
    for (const auto& [key, plan] : impl_->pair_plans) {
        (void)key;
        (void)strix_tp_dense_pair_plan_destroy(plan);
    }
#if defined(STRIX_DS41_TRELLIS_DENSE_R1)
    for (const auto& [key, plan] : impl_->pair_r1_plans) {
        (void)key;
        (void)strix_tp_dense_plan_destroy(plan);
    }
#endif
#if defined(STRIX_DS41_TRELLIS_DENSE_R1) && defined(STRIX_TP_SWIGLU_CLAMPED_DECLARED)
    for (const auto& [key, plan] : impl_->swiglu_plans) {
        (void)key;
        (void)strix_tp_swiglu_plan_destroy(plan);
    }
    impl_->swiglu_workspaces.clear();
#endif
    impl_->pair_workspaces.clear();
    impl_->split_workspaces.clear();
#endif
    for (auto& w : impl_->workspaces) {
        (void)hipFree(w.lib);
        (void)hipFree(w.xs);
        (void)hipFree(w.ys);
    }
    for (auto& s : impl_->sites) (void)strix_weights_destroy(static_cast<strix_weights>(s.weights));
    if (impl_->context) (void)strix_context_destroy(impl_->context);
}

bool TrellisContext::available() noexcept { return true; }
const char* TrellisContext::library_version() noexcept {
#if defined(STRIX_DS41_TRELLIS_DENSE_R1)
    return "strix_trellis dense-r1 LEGACY (C ABI 1)";
#elif defined(STRIX_DS41_TRELLIS_SPLITK)
    return "strix_trellis 0.3 + split-K 7b5b9b3 (C ABI 1)";
#elif defined(STRIX_DS41_TRELLIS_PREPARED)
    return "strix_trellis 0.3 ef48b33 prepared TP (C ABI 1)";
#else
    return "strix_trellis fused v12 (C ABI 1)";
#endif
}
void* TrellisContext::library_context() const noexcept { return impl_->context; }

namespace {
[[maybe_unused]] bool tp_shape(const TrellisSite& s) {
    if (s.bits != 4U && s.bits != 5U) return false;
    return (s.columns == 1280U && s.rows == 16384U) || (s.columns == 4096U && s.rows == 5120U) ||
           (s.columns == 5120U && (s.rows == 1152U || s.rows == 1280U || s.rows == 512U)) ||
           (s.columns == 1152U && s.rows == 5120U)
#if defined(STRIX_DS41_TRELLIS_DENSE_R1)
           || (s.columns == 5120U && (s.rows == 64640U || s.rows == 129280U))
#endif
           ;
}
}

void TrellisContext::submit(const TrellisSite& site, const float* x, float* y, std::uint32_t rows, void* workspace,
                            std::uint64_t bytes, void* stream) {
    submit(site, x, 0, y, 0, rows, workspace, bytes, stream);
}
void TrellisContext::submit(const TrellisSite& site, const float* x, std::size_t x_stride, float* y, std::size_t y_stride,
                            std::uint32_t rows, void* workspace, std::uint64_t bytes, void* stream) {
    if (x_stride == site.columns) x_stride = 0;
    if (y_stride == site.rows) y_stride = 0;
    strix_matmul_desc d{};
    strix_matmul_desc_init(&d);
    d.weights = static_cast<strix_weights>(site.weights);
    set_tensor(d.input, const_cast<float*>(x), rows, site.columns, x_stride);
    set_tensor(d.output, y, rows, site.rows, y_stride);
    d.workspace = workspace;
    d.workspace_bytes = bytes;
    const auto st = lib_stream(as_stream(stream));
#if defined(STRIX_DS41_TRELLIS_PREPARED)
    const Impl::PlanKey key{reinterpret_cast<std::uintptr_t>(site.weights), reinterpret_cast<std::uintptr_t>(x),
        reinterpret_cast<std::uintptr_t>(y), reinterpret_cast<std::uintptr_t>(workspace),
        reinterpret_cast<std::uintptr_t>(stream), rows, x_stride, y_stride};
    auto it = impl_->plans.find(key);
    if (it == impl_->plans.end()) {
        hipStreamCaptureStatus capture{};
        check(hipStreamIsCapturing(as_stream(stream), &capture), "plan capture check");
        if (capture != hipStreamCaptureStatusNone)
            throw std::logic_error("ds41 trellis: warm this site/buffer/row plan before capture: " + site.key);
        const auto created = std::chrono::steady_clock::now();
        Impl::Plan p;
        bool tp = tp_shape(site);
#if defined(STRIX_DS41_TRELLIS_DENSE_R1)
        if (tp) {
            std::uint64_t required = 0;
            lib_check(strix_tp_dense_r1_workspace_bytes(impl_->context, &d, 0, STRIX_DENSE_R1_LEGACY, &required), "dense-r1 workspace " + site.key);
            const auto& pool = impl_->split_workspace(site, rows, as_stream(stream), required);
            d.workspace = pool.data.get(); d.workspace_bytes = pool.bytes;
            lib_check(strix_tp_dense_plan_create_r1(impl_->context, &d, &st, 0, STRIX_DENSE_R1_LEGACY, &p.tp), "dense-r1 plan " + site.key);
        }
#elif defined(STRIX_DS41_TRELLIS_SPLITK)
        // Frozen per-shape selection: q_b K5 R8 retains the linear plan.
        // shared_down retains the qualified 0.3 kernel (zero workspace query).
        tp = tp && !(site.columns == 1280U && site.rows == 16384U && site.bits == 5U && rows == 8U);
        if (tp) {
            std::uint64_t required = 0;
            lib_check(strix_tp_dense_workspace_bytes(impl_->context, &d, 0, &required), "split-K plan workspace " + site.key);
            if (required == 0U) {
                d.workspace = nullptr;
                d.workspace_bytes = 0;
            } else {
                // Counter storage persists between submits. Only identical layouts on one stream may share it;
                // generic GEMV and other row counts must never overwrite this allocation.
                const auto& pool = impl_->split_workspace(site, rows, as_stream(stream), required);
                d.workspace = pool.data.get();
                d.workspace_bytes = pool.bytes;
            }
            lib_check(strix_tp_dense_plan_create_splitk(impl_->context, &d, &st, 0, STRIX_TP_SPLITK_SINGLE, &p.tp),
                      "split-K plan " + site.key);
        }
#else
        if (tp) {
            d.workspace = nullptr;
            d.workspace_bytes = 0;
            lib_check(strix_tp_dense_plan_create(impl_->context, &d, &st, &p.tp), "TP plan " + site.key);
        }
#endif
        if (!tp) lib_check(strix_linear_plan_create(impl_->context, &d, &st, STRIX_OPERATION_GEMV, &p.linear),
                            "linear plan " + site.key);
        it = impl_->plans.emplace(key, p).first;
        note_trellis_plan_created(std::chrono::duration<double>(std::chrono::steady_clock::now() - created).count());
    }
    if (it->second.tp) lib_check(strix_tp_dense_plan_submit(it->second.tp), "TP submit " + site.key);
    else lib_check(strix_linear_plan_submit(it->second.linear), "linear submit " + site.key);
#else
    lib_check(strix_gemv(impl_->context, &d, &st), "gemv " + site.key);
#endif
}

void TrellisContext::release_plans(void* workspace) noexcept {
#if defined(STRIX_LINEAR_TRANSFORMS_FUSED_DECLARED)
    for (auto it = impl_->shared_down_plans.begin(); it != impl_->shared_down_plans.end();) {
        if (std::get<3>(it->first) == reinterpret_cast<std::uintptr_t>(workspace)) {
            (void)strix_linear_plan_destroy(it->second);
            it = impl_->shared_down_plans.erase(it);
        } else ++it;
    }
#endif
#if defined(STRIX_DS41_TRELLIS_PREPARED)
    for (auto it = impl_->plans.begin(); it != impl_->plans.end();) {
        if (std::get<3>(it->first) == reinterpret_cast<std::uintptr_t>(workspace)) {
            if (it->second.tp) (void)strix_tp_dense_plan_destroy(it->second.tp);
            if (it->second.linear) (void)strix_linear_plan_destroy(it->second.linear);
            it = impl_->plans.erase(it);
        } else ++it;
    }
#elif !defined(STRIX_LINEAR_TRANSFORMS_FUSED_DECLARED)
    (void)workspace;
#endif
}

const TrellisSite& TrellisContext::prepare(TrellisPack& pack, const std::string& pack_name, const TrellisSlice& slice) {
    if (const auto* s = find(pack_name, slice)) return *s;
    const auto host = pack.load(pack_name, slice);
    strix_exl3_metadata_v1 m{};
    m.struct_size = sizeof(m);
    m.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
    m.rows = host.rows;
    m.columns = host.columns;
    m.bits_x2 = 2U * host.bits;
    m.codebook = STRIX_CODEBOOK_MUL1;
    m.marker = host.marker;
    m.trellis_offset = host.trellis_offset;
    m.trellis_bytes = host.trellis_bytes;
    m.suh_offset = host.suh_offset;
    m.suh_bytes = host.suh_bytes;
    m.svh_offset = host.svh_offset;
    m.svh_bytes = host.svh_bytes;
    strix_exl3_transport_desc t{};
    t.struct_size = sizeof(t);
    t.abi_version = STRIX_TRELLIS_ABI_VERSION;
    t.format = STRIX_TRANSPORT_EXL3_CANONICAL;
    t.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
    t.data = host.blob.data();
    t.data_bytes = host.blob.size();
    t.metadata = &m;
    t.metadata_bytes = sizeof(m);
    t.rows = host.rows;
    t.columns = host.columns;
    const auto st = lib_stream(nullptr);
    strix_weights w = nullptr;
    lib_check(strix_weights_prepare(impl_->context, &t, &st, &w), "prepare " + pack_name);
    check(hipDeviceSynchronize(), "prepare sync");
    TrellisSite s;
    s.weights = w;
    s.rows = host.rows;
    s.columns = host.columns;
    s.bits = host.bits;
    s.device_bytes = host.blob.size();
    s.key = pack_name + "@" + slice.describe();
    s.context = this;
    impl_->sites.push_back(std::move(s));
    return impl_->sites.back();
}

const TrellisSite* TrellisContext::find(const std::string& pack_name, const TrellisSlice& slice) const {
    const auto key = pack_name + "@" + slice.describe();
    for (const auto& s : impl_->sites)
        if (s.key == key) return &s;
    return nullptr;
}

void TrellisContext::reserve(void* stream, std::uint32_t max_rows) {
    const auto hs = as_stream(stream);
    for (const auto& w : impl_->workspaces)
        if (w.stream == hs) throw std::logic_error("ds41 trellis: stream reserved twice");
    (void)max_rows;  // every chunk has at most 8 rows (padding may exceed a smaller max_rows)
    Impl::Workspace w;
    w.stream = hs;
    std::uint64_t max_i = 128, max_n = 128;
    for (const auto& s : impl_->sites) {
        max_i = std::max<std::uint64_t>(max_i, s.columns);
        max_n = std::max<std::uint64_t>(max_n, s.rows);
        for (std::uint32_t r = 1; r <= 8U; ++r) {
            strix_matmul_desc d{};
            strix_matmul_desc_init(&d);
            d.weights = static_cast<strix_weights>(s.weights);
            // query only: host descriptors are validated, device memory is not touched (addresses must not overlap)
            set_tensor(d.input, reinterpret_cast<void*>(std::uintptr_t{1} << 40U), r, s.columns);
            set_tensor(d.output, reinterpret_cast<void*>(std::uintptr_t{1} << 41U), r, s.rows);
            std::uint64_t bytes = 0;
            lib_check(strix_matmul_workspace_bytes(impl_->context, &d, &bytes), "workspace query " + s.key);
            w.lib_bytes = std::max(w.lib_bytes, bytes);
#if defined(STRIX_DS41_TRELLIS_DENSE_R1)
            if (tp_shape(s)) {
                lib_check(strix_tp_dense_r1_workspace_bytes(impl_->context, &d, 0, STRIX_DENSE_R1_LEGACY, &bytes), "dense-r1 workspace " + s.key);
                (void)impl_->split_workspace(s, r, hs, bytes);
            }
#elif defined(STRIX_DS41_TRELLIS_SPLITK)
            if (tp_shape(s) && !(s.columns == 1280U && s.rows == 16384U && s.bits == 5U && r == 8U)) {
                lib_check(strix_tp_dense_workspace_bytes(impl_->context, &d, 0, &bytes), "split-K workspace " + s.key);
                if (bytes != 0U) (void)impl_->split_workspace(s, r, hs, bytes);
            }
#endif
        }
    }
    constexpr std::uint32_t chunk_max = 8;
#if defined(STRIX_DS41_TRELLIS_SPLITK)
    // Pair counters have their own layout and must never share the single-projection pool.
    // Reserve all R layouts before capture, just as for individual projections above.
    for (const auto& qa : impl_->sites) {
        const auto at = qa.key.find(".wq_a@");
        if (at == std::string::npos) continue;
        auto kv_key = qa.key;
        kv_key.replace(at, 6U, ".wkv@");
        const auto ki = std::find_if(impl_->sites.begin(), impl_->sites.end(), [&](const auto& s) { return s.key == kv_key; });
        if (ki == impl_->sites.end()) continue;
        for (std::uint32_t r = 1; r <= 8U; ++r) {
            strix_tp_dense_pair_desc d{};
            strix_tp_dense_pair_desc_init(&d);
            set_tensor(d.input, reinterpret_cast<void*>(std::uintptr_t{1} << 40U), r, 5120U);
            d.member[0].weights = static_cast<strix_weights>(qa.weights);
            d.member[1].weights = static_cast<strix_weights>(ki->weights);
            set_tensor(d.member[0].output, reinterpret_cast<void*>(std::uintptr_t{1} << 41U), r, 1280U);
            set_tensor(d.member[1].output, reinterpret_cast<void*>(std::uintptr_t{1} << 42U), r, 512U);
            std::uint64_t bytes = 0;
#if defined(STRIX_DS41_TRELLIS_DENSE_R1)
            lib_check(strix_tp_dense_pair_r1_workspace_bytes(impl_->context, &d, STRIX_DENSE_R1_LEGACY, &bytes), "pair workspace reserve");
#else
            lib_check(strix_tp_dense_pair_workspace_bytes(impl_->context, &d, &bytes), "pair workspace reserve");
#endif
            const Impl::PairWorkspaceKey key{reinterpret_cast<std::uintptr_t>(stream), qa.bits, ki->bits, r};
            if (!impl_->pair_workspaces.contains(key)) {
                void* p = nullptr;
                check(hipMalloc(&p, bytes), "pair workspace reserve");
                Impl::SplitWorkspace owned{std::unique_ptr<void, Impl::HipFree>(p), bytes};
                impl_->pair_workspaces.emplace(key, std::move(owned));
            }
        }
    }
#endif
    check(hipMalloc(&w.lib, std::max<std::uint64_t>(w.lib_bytes, 256U)), "workspace");
    check(hipMalloc(reinterpret_cast<void**>(&w.xs), std::size_t{chunk_max} * max_i * sizeof(float)), "x staging");
    check(hipMalloc(reinterpret_cast<void**>(&w.ys), std::size_t{chunk_max} * max_n * sizeof(float)), "y staging");
    w.staging = staging_bytes_for(static_cast<std::uint32_t>(max_i), static_cast<std::uint32_t>(max_n));
    impl_->workspaces.push_back(w);
}

std::uint64_t TrellisContext::staging_bytes() const noexcept {
    std::uint64_t b = 0;
    for (const auto& w : impl_->workspaces) b += w.staging;
    return b;
}
std::uint64_t TrellisContext::library_workspace_bytes() const noexcept {
    std::uint64_t b = 0;
    for (const auto& w : impl_->workspaces) b += std::max<std::uint64_t>(w.lib_bytes, 256U);
#if defined(STRIX_DS41_TRELLIS_SPLITK)
    for (const auto& [key, w] : impl_->split_workspaces) {
        (void)key;
        b += w.bytes;
    }
    for (const auto& [key, w] : impl_->pair_workspaces) {
        (void)key;
        b += w.bytes;
    }
#endif
    return b;
}

std::uint64_t TrellisContext::device_bytes() const noexcept {
    std::uint64_t b = 0;
    for (const auto& s : impl_->sites) b += s.device_bytes;
    return b;
}
std::size_t TrellisContext::sites() const noexcept { return impl_->sites.size(); }
std::string TrellisContext::announced_identity(const std::vector<std::string>& site_classes, bool shexp_tp) const {
    std::map<std::string, std::uint32_t> bits;  // mix class -> K (one pack per class)
    for (const auto& site : impl_->sites) {
        const auto cls = trellis_mix_class(site.key.substr(0, site.key.find('@')));
        if (const auto it = bits.find(cls); it != bits.end() && it->second != site.bits)
            throw std::logic_error("ds41 trellis: mix class " + cls + " prepared with K" + std::to_string(it->second) + " and K" +
                                   std::to_string(site.bits));
        bits[cls] = site.bits;
    }
    if (bits.empty()) return {};
    // only the classes that were prepared (a selected class without any site in this process announces nothing)
    std::vector<std::string> prepared;
    for (const auto& c : site_classes) {
        const bool any = c == "shexp" ? bits.contains("shexp_w13") || bits.contains("shexp_w2") : bits.contains(c);
        if (any) prepared.push_back(c);
    }
    return trellis_config_identity(bits, prepared, shexp_tp);
}

void enqueue_trellis_linear(const TrellisSite& site, const float* x, float* y, std::uint32_t rows, std::size_t x_stride,
                            std::size_t y_stride, bool accumulate, void* stream) {
    if (rows == 0U) return;
    if (site.context == nullptr || x_stride < site.columns || y_stride < site.rows)
        throw std::invalid_argument("ds41 trellis: bad linear call for " + site.key);
    auto& impl = site.context->impl();
    const auto hs = as_stream(stream);
    const auto& w = impl.workspace(hs);
    const std::uint32_t chunk = trellis_chunk_rows(site);
    const std::size_t xb = std::size_t{site.columns} * 4U;
    const bool headout = trellis_headout_enabled();
    const bool views = trellis_row_views_enabled() && tp_shape(site);  // strided rows straight to the dense-r1 plan
    for (std::uint32_t begin = 0; begin < rows; begin += chunk) {
        const std::uint32_t n = std::min(chunk, rows - begin);
        const std::uint32_t m = trellis_padded_rows(site, n);  // library rows of this chunk (n real + m - n copies of the last)
        const float* xin = x + std::size_t{begin} * x_stride;
        float* yrow = y + std::size_t{begin} * y_stride;
        if (views && n > 1U && m == n && !accumulate) {  // one row needs no staging: R1 keeps its plans as they are
            site.context->submit(site, xin, x_stride, yrow, y_stride, n, w.lib, std::max<std::uint64_t>(w.lib_bytes, 256U), hs);
            continue;
        }
        if ((x_stride != site.columns && !(headout && n == 1U)) || m != n) {
            check(hipMemcpy2DAsync(w.xs, xb, xin, x_stride * 4U, xb, n, hipMemcpyDeviceToDevice, hs), "x staging copy");
            for (std::uint32_t k = n; k < m; ++k)
                check(hipMemcpyAsync(w.xs + std::size_t{k} * site.columns, xin + std::size_t{n - 1U} * x_stride, xb, hipMemcpyDeviceToDevice, hs),
                      "x padding copy");
            xin = w.xs;
        }
        const bool direct = !accumulate && (y_stride == site.rows || (headout && n == 1U)) && m == n;
        float* yout = direct ? yrow : w.ys;
        site.context->submit(site, xin, yout, m, w.lib, std::max<std::uint64_t>(w.lib_bytes, 256U), hs);
        if (direct) continue;
        if (accumulate) {
            hipLaunchKernelGGL(accumulate_rows_kernel, dim3(blocks_for(std::size_t{n} * site.rows, 256)), dim3(256), 0, hs, yrow,
                               y_stride, w.ys, site.rows, n);
            check(hipGetLastError(), "accumulate");
        } else {
            check(hipMemcpy2DAsync(yrow, y_stride * 4U, w.ys, std::size_t{site.rows} * 4U, std::size_t{site.rows} * 4U, n,
                                   hipMemcpyDeviceToDevice, hs),
                  "y staging copy");
        }
    }
}

void enqueue_trellis_linear_rows(const TrellisSite& site, const float* x, float* y, std::uint32_t rows, std::size_t x_stride,
                                 std::size_t y_stride, std::uint32_t capacity, std::uint32_t pad_from, void* stream) {
    if (rows == 0U) return;
    if (site.context == nullptr || x_stride < site.columns || y_stride < site.rows || capacity < rows)
        throw std::invalid_argument("ds41 trellis: bad rows call for " + site.key);
    auto& impl = site.context->impl();
    const auto hs = as_stream(stream);
    const auto& w = impl.workspace(hs);
    const std::uint32_t chunk = trellis_chunk_rows(site);
    const std::size_t xb = std::size_t{site.columns} * 4U;
    const bool views = trellis_row_views_enabled() && tp_shape(site);  // as enqueue_trellis_linear
    for (std::uint32_t begin = 0; begin < rows; begin += chunk) {
        const std::uint32_t n = std::min(chunk, rows - begin);
        std::uint32_t m = trellis_padded_rows(site, n);
        if (pad_from != 0U && n >= pad_from && begin + chunk <= capacity) m = std::max(m, chunk);
        const bool fits = begin + m <= capacity;  // rows [n, m) of x / y are the caller's scratch rows
        const float* xin = x + std::size_t{begin} * x_stride;
        if (views && fits && m > 1U) {  // padding rows are the caller's scratch rows at the same strides; R1 as before
            site.context->submit(site, xin, x_stride, y + std::size_t{begin} * y_stride, y_stride, m, w.lib,
                                 std::max<std::uint64_t>(w.lib_bytes, 256U), hs);
            continue;
        }
        if (!(fits && (m == 1U || x_stride == site.columns))) {  // staged as enqueue_trellis_linear
            check(hipMemcpy2DAsync(w.xs, xb, xin, x_stride * 4U, xb, n, hipMemcpyDeviceToDevice, hs), "x staging copy");
            for (std::uint32_t k = n; k < m; ++k)
                check(hipMemcpyAsync(w.xs + std::size_t{k} * site.columns, xin + std::size_t{n - 1U} * x_stride, xb, hipMemcpyDeviceToDevice, hs),
                      "x padding copy");
            xin = w.xs;
        }
        float* yrow = y + std::size_t{begin} * y_stride;
        const bool direct = fits && (m == 1U || y_stride == site.rows);
        site.context->submit(site, xin, direct ? yrow : w.ys, m, w.lib, std::max<std::uint64_t>(w.lib_bytes, 256U), hs);
        if (!direct)
            check(hipMemcpy2DAsync(yrow, y_stride * 4U, w.ys, std::size_t{site.rows} * 4U, std::size_t{site.rows} * 4U, n,
                                   hipMemcpyDeviceToDevice, hs),
                  "y staging copy");
    }
}

#else  // built without the trellis library: the host side works, the device side refuses

struct TrellisContext::Impl {};
TrellisContext::TrellisContext() {
    throw std::runtime_error("ds41 trellis: this build has no trellis library (configure with STRIX_DS41_TRELLIS=ON)");
}
TrellisContext::~TrellisContext() = default;
bool TrellisContext::available() noexcept { return false; }
const char* TrellisContext::library_version() noexcept { return "none"; }
void* TrellisContext::library_context() const noexcept { return nullptr; }
void TrellisContext::submit(const TrellisSite&, const float*, float*, std::uint32_t, void*, std::uint64_t, void*) {
    throw std::logic_error("ds41 trellis: no library");
}
void TrellisContext::release_plans(void*) noexcept {}
const TrellisSite& TrellisContext::prepare(TrellisPack&, const std::string&, const TrellisSlice&) {
    throw std::logic_error("ds41 trellis: no library");
}
const TrellisSite* TrellisContext::find(const std::string&, const TrellisSlice&) const { return nullptr; }
void TrellisContext::reserve(void*, std::uint32_t) { throw std::logic_error("ds41 trellis: no library"); }
std::uint64_t TrellisContext::device_bytes() const noexcept { return 0; }
std::uint64_t TrellisContext::staging_bytes() const noexcept { return 0; }
std::uint64_t TrellisContext::library_workspace_bytes() const noexcept { return 0; }
std::size_t TrellisContext::sites() const noexcept { return 0; }
std::string TrellisContext::announced_identity(const std::vector<std::string>&, bool) const { return {}; }
void enqueue_trellis_linear(const TrellisSite&, const float*, float*, std::uint32_t, std::size_t, std::size_t, bool, void*) {
    throw std::logic_error("ds41 trellis: no library");
}
void enqueue_trellis_linear_rows(const TrellisSite&, const float*, float*, std::uint32_t, std::size_t, std::size_t, std::uint32_t,
                                 std::uint32_t, void*) {
    throw std::logic_error("ds41 trellis: no library");
}

#endif

std::uint64_t TrellisContext::staging_bytes_for(std::uint32_t max_columns, std::uint32_t max_rows) noexcept {
    // reserve(): x / y staging of 8 rows (every chunk has at most 8), widths at least 128
    return 8U * (std::uint64_t{std::max(max_columns, 128U)} + std::max(max_rows, 128U)) * sizeof(float);
}

void enqueue_trellis_qakv(const TrellisSite& qa, const TrellisSite& kv, const float* x, float* q, float* k,
                          std::uint32_t rows, void* stream) {
    if (qa.context == nullptr || qa.context != kv.context || qa.columns != 5120U || kv.columns != 5120U ||
        qa.rows != 1280U || kv.rows != 512U)
        throw std::invalid_argument("ds41 trellis: invalid q_a+kv pair");
    for (std::uint32_t first = 0; first < rows; first += 8U) {
        const auto n = std::min(8U, rows - first);
        const float* in = x + std::size_t{first} * 5120U;
        float* qo = q + std::size_t{first} * 1280U;
        float* ko = k + std::size_t{first} * 512U;
#if defined(STRIX_DS41_TRELLIS_SPLITK)
        auto& impl = qa.context->impl();
        const TrellisContext::Impl::PairKey key{reinterpret_cast<std::uintptr_t>(qa.weights),
            reinterpret_cast<std::uintptr_t>(kv.weights), reinterpret_cast<std::uintptr_t>(in),
            reinterpret_cast<std::uintptr_t>(qo), reinterpret_cast<std::uintptr_t>(ko),
            reinterpret_cast<std::uintptr_t>(stream), n};
#if defined(STRIX_DS41_TRELLIS_DENSE_R1)
        auto it = impl.pair_r1_plans.find(key);
        if (it == impl.pair_r1_plans.end()) {
#else
        auto it = impl.pair_plans.find(key);
        if (it == impl.pair_plans.end()) {
#endif
            hipStreamCaptureStatus capture{};
            check(hipStreamIsCapturing(as_stream(stream), &capture), "pair capture check");
            if (capture != hipStreamCaptureStatusNone)
                throw std::logic_error("ds41 trellis: warm q_a+kv buffers/rows before capture");
            const auto created = std::chrono::steady_clock::now();
            strix_tp_dense_pair_desc d{};
            strix_tp_dense_pair_desc_init(&d);
            set_tensor(d.input, const_cast<float*>(in), n, 5120U);
            d.member[0].weights = static_cast<strix_weights>(qa.weights);
            d.member[1].weights = static_cast<strix_weights>(kv.weights);
            set_tensor(d.member[0].output, qo, n, 1280U);
            set_tensor(d.member[1].output, ko, n, 512U);
            std::uint64_t bytes = 0;
#if defined(STRIX_DS41_TRELLIS_DENSE_R1)
            lib_check(strix_tp_dense_pair_r1_workspace_bytes(impl.context, &d, STRIX_DENSE_R1_LEGACY, &bytes), "pair workspace");
#else
            lib_check(strix_tp_dense_pair_workspace_bytes(impl.context, &d, &bytes), "pair workspace");
#endif
            const TrellisContext::Impl::PairWorkspaceKey wk{reinterpret_cast<std::uintptr_t>(stream), qa.bits, kv.bits, n};
            auto wi = impl.pair_workspaces.find(wk);
            if (wi == impl.pair_workspaces.end()) {
                void* p = nullptr;
                check(hipMalloc(&p, bytes), "pair persistent workspace");
                TrellisContext::Impl::SplitWorkspace owned{std::unique_ptr<void, TrellisContext::Impl::HipFree>(p), bytes};
                wi = impl.pair_workspaces.emplace(wk, std::move(owned)).first;
            }
            if (wi->second.bytes != bytes) throw std::logic_error("ds41 trellis: inconsistent pair workspace");
            d.workspace = wi->second.data.get();
            d.workspace_bytes = bytes;
            const auto st = lib_stream(as_stream(stream));
#if defined(STRIX_DS41_TRELLIS_DENSE_R1)
            strix_tp_dense_plan plan = nullptr;
            lib_check(strix_tp_dense_pair_plan_create_r1(impl.context, &d, &st, STRIX_DENSE_R1_LEGACY, &plan), "dense-r1 q_a+kv plan");
            it = impl.pair_r1_plans.emplace(key, plan).first;
            note_trellis_plan_created(std::chrono::duration<double>(std::chrono::steady_clock::now() - created).count());
        }
        lib_check(strix_tp_dense_plan_submit(it->second), "dense-r1 q_a+kv submit");
#else
            d.strategy = STRIX_TP_SPLITK_SINGLE;
            strix_tp_dense_pair_plan plan = nullptr;
            lib_check(strix_tp_dense_pair_plan_create(impl.context, &d, &st, &plan), "q_a+kv plan");
            it = impl.pair_plans.emplace(key, plan).first;
            note_trellis_plan_created(std::chrono::duration<double>(std::chrono::steady_clock::now() - created).count());
        }
        lib_check(strix_tp_dense_pair_plan_submit(it->second), "q_a+kv submit");
#endif
#else
        enqueue_trellis_linear(qa, in, qo, n, 5120U, 1280U, false, stream);
        enqueue_trellis_linear(kv, in, ko, n, 5120U, 512U, false, stream);
#endif
    }
}

void enqueue_trellis_shared_swiglu(const TrellisSite* const halves[4], const float* x, float* act, std::uint32_t rows,
                                   float limit, void* stream) {
#if defined(STRIX_DS41_TRELLIS_LIB) && defined(STRIX_DS41_TRELLIS_DENSE_R1) && defined(STRIX_TP_SWIGLU_CLAMPED_DECLARED)
    constexpr std::uint32_t H = 5120U, HALF = 1152U, I = 2U * HALF;
    for (int k = 0; k < 4; ++k)
        if (halves[k] == nullptr || halves[k]->context == nullptr || halves[k]->context != halves[0]->context ||
            halves[k]->columns != H || halves[k]->rows != HALF || halves[k]->bits != halves[0]->bits)
            throw std::invalid_argument("ds41 trellis: invalid shared gate/up halves for the SwiGLU pair");
    auto& impl = halves[0]->context->impl();
    const auto hs = as_stream(stream);
    for (std::uint32_t first = 0; first < rows; first += 8U) {
        const auto n = std::min(8U, rows - first);
        const float* in = x + std::size_t{first} * H;
        for (std::uint32_t h = 0; h < 2U; ++h) {
            const TrellisSite& gate = *halves[h];
            const TrellisSite& up = *halves[2U + h];
            float* out = act + std::size_t{first} * I + std::size_t{h} * HALF;
            const TrellisContext::Impl::PairKey key{reinterpret_cast<std::uintptr_t>(gate.weights),
                reinterpret_cast<std::uintptr_t>(up.weights), reinterpret_cast<std::uintptr_t>(in),
                reinterpret_cast<std::uintptr_t>(out), std::uintptr_t{__builtin_bit_cast(std::uint32_t, limit)},
                reinterpret_cast<std::uintptr_t>(stream), n};  // the activation limit is part of the key
            auto it = impl.swiglu_plans.find(key);
            if (it == impl.swiglu_plans.end()) {
                hipStreamCaptureStatus capture{};
                check(hipStreamIsCapturing(hs, &capture), "shared swiglu capture check");
                if (capture != hipStreamCaptureStatusNone)
                    throw std::logic_error("ds41 trellis: warm the shared SwiGLU pair buffers/rows before capture");
                const auto created = std::chrono::steady_clock::now();
                strix_tp_swiglu_desc d{};
                strix_tp_swiglu_desc_init(&d);
                set_tensor(d.input, const_cast<float*>(in), n, H);
                d.gate = static_cast<strix_weights>(gate.weights);
                d.up = static_cast<strix_weights>(up.weights);
                // [n][1152] at row stride 2304 inside act (the rows of this half only)
                d.hidden_output.data = out;
                d.hidden_output.buffer_bytes = (std::uint64_t{n - 1U} * I + HALF) * sizeof(float);
                d.hidden_output.dtype = STRIX_DTYPE_F32;
                d.hidden_output.rank = 2;
                d.hidden_output.shape[0] = n;
                d.hidden_output.shape[1] = HALF;
                d.hidden_output.strides[0] = I;
                d.hidden_output.strides[1] = 1;
                d.strategy = STRIX_TP_SPLITK_SINGLE;
                d.activation = STRIX_TP_SWIGLU_CLAMPED_DECLARED;
                d.limit = limit;
                std::uint64_t bytes = 0;
                lib_check(strix_tp_swiglu_workspace_bytes(impl.context, &d, &bytes), "shared swiglu workspace");
                const TrellisContext::Impl::PairWorkspaceKey wk{reinterpret_cast<std::uintptr_t>(stream), gate.bits, up.bits, n};
                auto wi = impl.swiglu_workspaces.find(wk);
                if (wi == impl.swiglu_workspaces.end()) {
                    void* p = nullptr;
                    check(hipMalloc(&p, bytes), "shared swiglu persistent workspace");
                    TrellisContext::Impl::SplitWorkspace owned{std::unique_ptr<void, TrellisContext::Impl::HipFree>(p), bytes};
                    wi = impl.swiglu_workspaces.emplace(wk, std::move(owned)).first;
                }
                if (wi->second.bytes != bytes) throw std::logic_error("ds41 trellis: inconsistent shared swiglu workspace");
                d.workspace = wi->second.data.get();
                d.workspace_bytes = bytes;
                const auto st = lib_stream(hs);
                strix_tp_swiglu_plan plan = nullptr;
                lib_check(strix_tp_swiglu_plan_create(impl.context, &d, &st, &plan), "shared swiglu plan");
                it = impl.swiglu_plans.emplace(key, plan).first;
                note_trellis_plan_created(std::chrono::duration<double>(std::chrono::steady_clock::now() - created).count());
                static bool announced = false;  // diagnostic evidence that the grouped route really executes
                if (!announced) {
                    announced = true;
                    std::fprintf(stderr, "ds41 trellis: shared SwiGLU pair plan created (rows %u, %u-bit)\n", n, gate.bits);
                }
            }
            lib_check(strix_tp_swiglu_plan_submit(it->second), "shared swiglu submit");
        }
    }
#else
    (void)halves; (void)x; (void)act; (void)rows; (void)limit; (void)stream;
    throw std::logic_error("ds41 trellis: STRIX_DS41_SHARED_GROUPED needs a dense-r1 build with the declared SwiGLU pair");
#endif
}


// ---- STRIX_DS41_SHARED_DOWN_FUSED: the shared down as one declared library launch ----------------------------------
#if defined(STRIX_DS41_TRELLIS_LIB) && defined(STRIX_LINEAR_TRANSFORMS_FUSED_DECLARED)
#define STRIX_DS41_SHARED_DOWN_ROUTE 1
#endif
namespace {
#if defined(STRIX_DS41_SHARED_DOWN_ROUTE)
constexpr std::uint32_t shared_down_input = 2304U, shared_down_output = 5120U, shared_down_bits = 4U;
#endif
// read once: -1 unset/empty (implicit default), 1 explicit on, 0 any other value (off)
int shared_down_env() noexcept {
    static const int v = [] {
        const auto* e = std::getenv("STRIX_DS41_SHARED_DOWN_FUSED");
        if (e == nullptr || e[0] == '\0') return -1;
        return e[0] == '1' ? 1 : 0;
    }();
    return v;
}
// Off until an admission admits it; an implicit refusal holds for the process. Once the route ran (`used`) the
// selection is frozen: a later admission that would change it throws instead.
std::atomic<bool> shared_down_on{false};
std::atomic<bool> shared_down_refused{false};
std::atomic<bool> shared_down_used{false};
}  // namespace

bool trellis_shared_down_fused_enabled() noexcept {
#if defined(STRIX_DS41_SHARED_DOWN_ROUTE)
    return shared_down_on.load(std::memory_order_relaxed);
#else
    return false;
#endif
}

void trellis_shared_down_fused_admit(const std::vector<const TrellisSite*>& downs, std::size_t layers, bool shexp_tp,
                                     std::uint32_t row_limit) {
    const int env = shared_down_env();
    if (env == 0) return;
    std::string why;
#if defined(STRIX_DS41_SHARED_DOWN_ROUTE)
    if (shexp_tp) why = "STRIX_DS41_SHEXP_TP=1";
    else if (downs.empty()) why = "no trellis shared down site";
    else if (downs.size() != layers)
        why = std::to_string(downs.size()) + " trellis shared down sites for " + std::to_string(layers) + " shared experts";
    for (const auto* d : downs) {
        if (!why.empty()) break;
        if (d == nullptr || d->context == nullptr || d->context != downs.front()->context)
            why = "shared down sites of different contexts";
        else if (d->bits != shared_down_bits || d->columns != shared_down_input || d->rows != shared_down_output)
            why = "shared down site " + d->key + " is K" + std::to_string(d->bits) + " [" + std::to_string(d->rows) + ", " +
                  std::to_string(d->columns) + "], not K4 [5120, 2304]";
    }
    // Startup probe of the linked library: a valid declared workspace query per site and row count. The staged query
    // of the same descriptor must succeed, so INVALID_ARGUMENT from the declared one is a confirmed rejection of the
    // selector (an older library, another geometry or a context whose kernels failed the creation-time check).
    const std::uint32_t probe_rows = std::min(8U, std::max(row_limit, 1U));
    for (std::size_t k = 0; why.empty() && k < downs.size(); ++k) {
        const auto& site = *downs[k];
        for (std::uint32_t r = 1; why.empty() && r <= probe_rows; ++r) {
            strix_matmul_desc d{};
            strix_matmul_desc_init(&d);
            d.weights = static_cast<strix_weights>(site.weights);
            // query only: host descriptors are validated, device memory is not touched (addresses must not overlap)
            set_tensor(d.input, reinterpret_cast<void*>(std::uintptr_t{1} << 40U), r, site.columns);
            set_tensor(d.output, reinterpret_cast<void*>(std::uintptr_t{1} << 41U), r, site.rows);
            std::uint64_t staged = 0, declared = 0;
            lib_check(strix_matmul_workspace_bytes(site.context->impl().context, &d, &staged), "shared down staged probe " + site.key);
            d.reserved[0] = STRIX_LINEAR_TRANSFORMS_FUSED_DECLARED;
            const strix_status s = strix_matmul_workspace_bytes(site.context->impl().context, &d, &declared);
            if (s == STRIX_ERROR_INVALID_ARGUMENT)
                why = "the linked library rejects the declared route (" + site.key + ", " + std::to_string(r) + " rows)";
            else if (s != STRIX_SUCCESS)
                throw std::runtime_error("ds41 trellis: shared down declared probe " + site.key + ": " + strix_status_string(s));
            else if (declared != staged)
                throw std::runtime_error("ds41 trellis: shared down declared probe " + site.key + ": workspace size differs");
        }
    }
#else
    (void)downs;
    (void)layers;
    (void)shexp_tp;
    (void)row_limit;
    why = "build without the declared shared down route";
#endif
    if (env == 1 && !why.empty()) throw std::invalid_argument("ds41 trellis: STRIX_DS41_SHARED_DOWN_FUSED=1: " + why);
    if (!why.empty()) {
        if (shared_down_used.load() && shared_down_on.load())
            throw std::logic_error("ds41 trellis: shared down fused route already in use, a later admission refuses it (" + why + ")");
        shared_down_on.store(false);
        if (!shared_down_refused.exchange(true))
            std::fprintf(stderr, "ds41 trellis: shared down fused launch not admitted by default (%s): old path\n", why.c_str());
        return;
    }
    if (shared_down_refused.load()) return;  // an earlier implicit refusal holds for the process
    if (!shared_down_on.exchange(true))
        std::fprintf(stderr, "ds41 trellis: shared down as one fused launch (%zu K4 sites, probe R1..R%u; %s)\n", downs.size(),
                     std::min(8U, std::max(row_limit, 1U)),
                     env == 1 ? "STRIX_DS41_SHARED_DOWN_FUSED=1" : "default; STRIX_DS41_SHARED_DOWN_FUSED=0 off");
}

void enqueue_trellis_shared_down(const TrellisSite& site, const float* act, std::size_t act_stride, float* out,
                                 std::size_t out_stride, std::uint32_t rows, void* stream) {
    if (!trellis_shared_down_fused_enabled()) {
        enqueue_trellis_linear(site, act, out, rows, act_stride, out_stride, false, stream);
        return;
    }
#if defined(STRIX_DS41_SHARED_DOWN_ROUTE)
    if (rows == 0U) return;
    if (site.context == nullptr || site.bits != shared_down_bits || site.columns != shared_down_input ||
        site.rows != shared_down_output || act_stride != shared_down_input || out_stride != shared_down_output)
        throw std::invalid_argument("ds41 trellis: shared down fused call outside the admitted geometry: " + site.key);
    shared_down_used.store(true, std::memory_order_relaxed);
    auto& impl = site.context->impl();
    const auto hs = as_stream(stream);
    const auto& w = impl.workspace(hs);
    const std::uint64_t ws_bytes = std::max<std::uint64_t>(w.lib_bytes, 256U);
    for (std::uint32_t first = 0; first < rows; first += 8U) {
        const auto n = std::min(8U, rows - first);
        const float* in = act + std::size_t{first} * shared_down_input;
        float* o = out + std::size_t{first} * shared_down_output;
        const TrellisContext::Impl::SharedDownKey key{reinterpret_cast<std::uintptr_t>(site.weights), reinterpret_cast<std::uintptr_t>(in),
            reinterpret_cast<std::uintptr_t>(o), reinterpret_cast<std::uintptr_t>(w.lib), reinterpret_cast<std::uintptr_t>(stream), n};
        auto it = impl.shared_down_plans.find(key);
        if (it == impl.shared_down_plans.end()) {
            hipStreamCaptureStatus capture{};
            check(hipStreamIsCapturing(hs, &capture), "shared down capture check");
            if (capture != hipStreamCaptureStatusNone)
                throw std::logic_error("ds41 trellis: warm the shared down fused plans (buffers/rows) before capture: " + site.key);
            const auto created = std::chrono::steady_clock::now();
            strix_matmul_desc d{};
            strix_matmul_desc_init(&d);
            d.weights = static_cast<strix_weights>(site.weights);
            set_tensor(d.input, const_cast<float*>(in), n, shared_down_input);
            set_tensor(d.output, o, n, shared_down_output);
            d.workspace = w.lib;
            d.workspace_bytes = ws_bytes;
            d.reserved[0] = STRIX_LINEAR_TRANSFORMS_FUSED_DECLARED;
            const auto st = lib_stream(hs);
            strix_linear_plan plan = nullptr;
            lib_check(strix_linear_plan_create(impl.context, &d, &st, STRIX_OPERATION_GEMV, &plan), "shared down fused plan " + site.key);
            it = impl.shared_down_plans.emplace(key, plan).first;
            note_trellis_plan_created(std::chrono::duration<double>(std::chrono::steady_clock::now() - created).count());
            static std::atomic<bool> announced{false};  // diagnostic evidence that the fused route really executes
            if (!announced.exchange(true))
                std::fprintf(stderr, "ds41 trellis: shared down fused plan created (rows %u, %u-bit)\n", n, site.bits);
        }
        lib_check(strix_linear_plan_submit(it->second), "shared down fused submit " + site.key);
    }
#else
    throw std::logic_error("ds41 trellis: STRIX_DS41_SHARED_DOWN_FUSED needs a library with the declared route");
#endif
}

}  // namespace strix::ds41
