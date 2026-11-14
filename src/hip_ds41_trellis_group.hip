// Grouped trellis linears with strided input/output (ds41_trellis_group.hpp): the o_a groups of a layer.
#include "strix/ds41_trellis_group.hpp"
#include "strix/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(STRIX_DS41_TRELLIS_LIB)
#include <strix_trellis.h>
#endif

namespace strix::ds41 {

int trellis_oa_group_mode() {
    // default 1 (=0 off): same bits as the per-group calls; with a grouped library GEMV it is one dispatch per layer
    const char* e = std::getenv("STRIX_DS41_TRELLIS_OA_GROUP");
    if (e == nullptr || *e == 0) return 1;
    if (std::strcmp(e, "0") == 0) return 0;
    if (std::strcmp(e, "1") == 0) return 1;
    if (std::strcmp(e, "2") == 0) return 2;
    throw std::invalid_argument(std::string("STRIX_DS41_TRELLIS_OA_GROUP must be 0, 1 or 2, got ") + e);
}

#if defined(STRIX_DS41_TRELLIS_LIB)

namespace {

void check(hipError_t e, const char* what) {
    ::strix::hip_check_prefixed("ds41 trellis group: ", e, what);
}
void lib_check(strix_status s, const std::string& what) {
    if (s != STRIX_SUCCESS) throw std::runtime_error("ds41 trellis group: " + what + ": " + strix_status_string(s));
}
hipStream_t as_stream(void* s) { return static_cast<hipStream_t>(s); }
[[maybe_unused]] strix_stream lib_stream(hipStream_t s) {
    strix_stream st{};
    st.struct_size = sizeof(st);
    st.abi_version = STRIX_TRELLIS_ABI_VERSION;
    st.hip_stream = s;
    return st;
}
void set_tensor(strix_tensor_desc& t, const float* data, std::uint32_t rows, std::uint32_t width) {
    t.data = const_cast<float*>(data);
    t.buffer_bytes = std::uint64_t{rows} * width * sizeof(float);
    t.dtype = STRIX_DTYPE_F32;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = width;
    t.strides[0] = width;
    t.strides[1] = 1;
}

// xs[g][k][i] = x[min(k, n - 1)][g I + i] for g < G, k < m, i < I (I multiple of 4, x rows 16-byte aligned columns)
__global__ void stage_groups_kernel(const float* __restrict__ x, std::size_t x_stride, float* __restrict__ xs, std::uint32_t G,
                                    std::uint32_t I, std::uint32_t n, std::uint32_t m) {
    const std::size_t q = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;  // one float4 of xs
    const std::size_t i4 = I / 4U, total = std::size_t{G} * m * i4;
    if (q >= total) return;
    const std::size_t i = (q % i4) * 4U, k = (q / i4) % m, g = q / (i4 * m);
    const std::size_t src = std::min<std::size_t>(k, n - 1U);
    const float* s = x + src * x_stride + g * I + i;
    float* d = xs + (g * m + k) * I + i;
    d[0] = s[0];
    d[1] = s[1];
    d[2] = s[2];
    d[3] = s[3];
}
// y[k][g N + j] = ys[g][k][j] for g < G, k < n (the real rows of m), j < N
__global__ void scatter_groups_kernel(const float* __restrict__ ys, float* __restrict__ y, std::size_t y_stride, std::uint32_t G,
                                      std::uint32_t N, std::uint32_t n, std::uint32_t m) {
    const std::size_t q = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const std::size_t j4 = N / 4U, total = std::size_t{G} * n * j4;
    if (q >= total) return;
    const std::size_t j = (q % j4) * 4U, k = (q / j4) % n, g = q / (j4 * n);
    const float* s = ys + (g * m + k) * N + j;
    float* d = y + k * y_stride + g * N + j;
    d[0] = s[0];
    d[1] = s[1];
    d[2] = s[2];
    d[3] = s[3];
}
unsigned blocks_for(std::size_t n, unsigned block) { return static_cast<unsigned>((n + block - 1U) / block); }

}  // namespace

struct TrellisGroupRunner::Impl {
    int mode = 1;
    strix_context context = nullptr;
    TrellisContext* owner = nullptr;
    std::uint32_t groups = 0, I = 0, N = 0, bits = 0;
    hipStream_t stream = nullptr;
    float* xs = nullptr;  // [G][8][I]
    float* ys = nullptr;  // [G][8][N]
    std::uint64_t lib_bytes = 0;
    std::vector<void*> lib;            // one library workspace per concurrent lane (mode 2: per group; else 1)
    std::vector<hipStream_t> lanes;    // mode 2: one stream per group
    std::vector<hipEvent_t> events;    // mode 2: fork + one join event per group
    bool grouped_lib = false;          // the library's grouped strided GEMV passed the self-check
    void* grouped_ws = nullptr;
    std::uint64_t grouped_bytes = 0;
    std::string why_not_grouped = "library without a grouped strided GEMV";

    void gemv(const TrellisSite& site, const float* x, float* y, std::uint32_t m, void* ws, hipStream_t s) const {
#if defined(STRIX_DS41_TRELLIS_PREPARED)
        site.context->submit(site, x, y, m, ws, lib_bytes, s);
#else
        strix_matmul_desc d{};
        strix_matmul_desc_init(&d);  // alpha 1, beta 0
        d.weights = static_cast<strix_weights>(site.weights);
        set_tensor(d.input, x, m, site.columns);
        set_tensor(d.output, y, m, site.rows);
        d.workspace = ws;
        d.workspace_bytes = lib_bytes;
        const auto st = lib_stream(s);
        lib_check(strix_gemv(context, &d, &st), "gemv " + site.key);
#endif
    }
    // the groups' library calls of one chunk (x / y: group g at x + g gx, y + g gy, m rows of stride xr / yr = I / N
    // contiguous, or one row)
    void calls(const std::vector<const TrellisSite*>& sites, const float* x, std::size_t gx, float* y, std::size_t gy,
               std::uint32_t m) const {
        if (mode != 2) {
            for (std::size_t g = 0; g < sites.size(); ++g) gemv(*sites[g], x + g * gx, y + g * gy, m, lib[0], stream);
            return;
        }
        check(hipEventRecord(events[0], stream), "fork record");
        for (std::size_t g = 0; g < sites.size(); ++g) {
            check(hipStreamWaitEvent(lanes[g], events[0], 0), "fork wait");
            gemv(*sites[g], x + g * gx, y + g * gy, m, lib[g], lanes[g]);
            check(hipEventRecord(events[1 + g], lanes[g]), "join record");
        }
        for (std::size_t g = 0; g < sites.size(); ++g) check(hipStreamWaitEvent(stream, events[1 + g], 0), "join wait");
    }
    void staged(const std::vector<const TrellisSite*>& sites, const float* x, std::size_t x_stride, float* y, std::size_t y_stride,
                std::uint32_t rows) const {
        const auto& s0 = *sites[0];
        const std::uint32_t chunk = trellis_chunk_rows(s0), G = static_cast<std::uint32_t>(sites.size());
        for (std::uint32_t begin = 0; begin < rows; begin += chunk) {
            const std::uint32_t n = std::min(chunk, rows - begin), m = trellis_padded_rows(s0, n);
            const float* xb = x + std::size_t{begin} * x_stride;
            float* yb = y + std::size_t{begin} * y_stride;
            if (m == 1U) {  // one row: contiguous in place
                calls(sites, xb, I, yb, N, 1);
                continue;
            }
            const std::size_t sx = std::size_t{G} * m * (I / 4U), sy = std::size_t{G} * n * (N / 4U);
            hipLaunchKernelGGL(stage_groups_kernel, dim3(blocks_for(sx, 256)), dim3(256), 0, stream, xb, x_stride, xs, G, I, n, m);
            check(hipGetLastError(), "stage");
            calls(sites, xs, std::size_t{m} * I, ys, std::size_t{m} * N, m);
            hipLaunchKernelGGL(scatter_groups_kernel, dim3(blocks_for(sy, 256)), dim3(256), 0, stream, ys, yb, y_stride, G, N, n, m);
            check(hipGetLastError(), "scatter");
        }
    }
#if defined(STRIX_TRELLIS_HAS_GROUPED_GEMV)
    // the library's grouped strided GEMV plans: rows of every group in one dispatch, each row the bits of strix_gemv R1 of
    // that row alone; chunks of up to 8 rows, no padding, no staging. A plan freezes the weights, x / y, rows and stream;
    // plans are cached per (layer weights, x, y, rows) and share grouped_ws serially on the runner's stream.
    struct GroupKey {
        std::uintptr_t w0, x, y;
        std::size_t xs, ys;
        std::uint32_t rows;
        auto operator<=>(const GroupKey&) const = default;
    };
    mutable std::map<GroupKey, strix_grouped_gemv_plan> group_plans;
    strix_status grouped(const std::vector<const TrellisSite*>& sites, const float* x, std::size_t x_stride, float* y,
                         std::size_t y_stride, std::uint32_t rows, bool query, std::uint64_t* bytes) const {
        std::vector<strix_weights> w(sites.size());
        for (std::size_t g = 0; g < sites.size(); ++g) w[g] = static_cast<strix_weights>(sites[g]->weights);
        for (std::uint32_t begin = 0; begin < rows; begin += 8U) {
            const std::uint32_t n = std::min(8U, rows - begin);
            strix_grouped_gemv_desc d{};
            d.struct_size = sizeof(d);
            d.abi_version = STRIX_TRELLIS_ABI_VERSION;
            d.weights = w.data();
            d.group_count = w.size();
            d.rows = n;
            d.input = x + std::size_t{begin} * x_stride;
            d.input_row_stride = x_stride;
            d.input_group_stride = I;
            d.input_buffer_bytes = ((std::uint64_t{n} - 1U) * x_stride + std::uint64_t{groups} * I) * 4U;
            d.output = y + std::size_t{begin} * y_stride;
            d.output_row_stride = y_stride;
            d.output_group_stride = N;
            d.output_buffer_bytes = ((std::uint64_t{n} - 1U) * y_stride + std::uint64_t{groups} * N) * 4U;
            d.flags = STRIX_GEMV_ROW_INVARIANT_R1;
            if (query) {
                std::uint64_t b = 0;
                const auto s = strix_grouped_gemv_workspace_bytes(context, &d, &b);
                if (s != STRIX_SUCCESS) return s;
                *bytes = std::max(*bytes, b);
                continue;
            }
            const GroupKey key{reinterpret_cast<std::uintptr_t>(w[0]), reinterpret_cast<std::uintptr_t>(d.input),
                               reinterpret_cast<std::uintptr_t>(d.output), x_stride, y_stride, n};
            auto it = group_plans.find(key);
            if (it == group_plans.end()) {
                hipStreamCaptureStatus capture{};
                check(hipStreamIsCapturing(stream, &capture), "grouped plan capture check");
                if (capture != hipStreamCaptureStatusNone)
                    throw std::logic_error("ds41 trellis group: warm this grouped o_a plan before capture: " + sites[0]->key);
                const auto created = std::chrono::steady_clock::now();
                d.workspace = grouped_ws;
                d.workspace_bytes = grouped_bytes;
                const auto st = lib_stream(stream);
                strix_grouped_gemv_plan plan = nullptr;
                const auto s = strix_grouped_gemv_plan_create(context, &d, &st, &plan);
                if (s != STRIX_SUCCESS) return s;
                it = group_plans.emplace(key, plan).first;
                note_trellis_plan_created(std::chrono::duration<double>(std::chrono::steady_clock::now() - created).count());
            }
            const auto s = strix_grouped_gemv_plan_submit(it->second);
            if (s != STRIX_SUCCESS) return s;
        }
        return STRIX_SUCCESS;
    }
#endif
};

TrellisGroupRunner::TrellisGroupRunner(const std::vector<const TrellisSite*>& sites, int mode, void* stream)
    : impl_(std::make_unique<Impl>()) {
    if (sites.empty() || mode < 1 || mode > 2) throw std::invalid_argument("ds41 trellis group: no sites or bad mode");
    auto& m = *impl_;
    const auto& s0 = *sites[0];
    m.mode = mode;
    m.groups = static_cast<std::uint32_t>(sites.size());
    m.I = s0.columns;
    m.N = s0.rows;
    m.bits = s0.bits;
    m.stream = as_stream(stream);
    m.context = static_cast<strix_context>(s0.context->library_context());
    m.owner = s0.context;
    for (const auto* s : sites)
        if (s->columns != m.I || s->rows != m.N || s->bits != m.bits || s->context != s0.context)
            throw std::invalid_argument("ds41 trellis group: groups of different shapes");
    if (m.I % 4U != 0U || m.N % 4U != 0U) throw std::invalid_argument("ds41 trellis group: I and N must be multiples of 4");
    for (std::uint32_t r = 1; r <= 8U; ++r) {
        strix_matmul_desc d{};
        strix_matmul_desc_init(&d);
        d.weights = static_cast<strix_weights>(s0.weights);
        set_tensor(d.input, reinterpret_cast<const float*>(std::uintptr_t{1} << 40U), r, m.I);  // query only
        set_tensor(d.output, reinterpret_cast<float*>(std::uintptr_t{1} << 41U), r, m.N);
        std::uint64_t b = 0;
        lib_check(strix_matmul_workspace_bytes(m.context, &d, &b), "workspace query " + s0.key);
        m.lib_bytes = std::max(m.lib_bytes, b);
    }
    m.lib_bytes = std::max<std::uint64_t>((m.lib_bytes + 255U) & ~std::uint64_t{255}, 256U);
    m.lib.assign(mode == 2 ? m.groups : 1U, nullptr);
    for (auto& p : m.lib) check(hipMalloc(&p, m.lib_bytes), "workspace");
    check(hipMalloc(reinterpret_cast<void**>(&m.xs), std::size_t{m.groups} * 8U * m.I * sizeof(float)), "x staging");
    check(hipMalloc(reinterpret_cast<void**>(&m.ys), std::size_t{m.groups} * 8U * m.N * sizeof(float)), "y staging");
    if (mode == 2) {
        m.lanes.assign(m.groups, nullptr);
        for (auto& l : m.lanes) check(hipStreamCreateWithFlags(&l, hipStreamNonBlocking), "lane stream");
        m.events.assign(1U + m.groups, nullptr);
        for (auto& e : m.events) check(hipEventCreateWithFlags(&e, hipEventDisableTiming), "event");
    }
#if defined(STRIX_TRELLIS_HAS_GROUPED_GEMV)
    if (const char* e = std::getenv("STRIX_DS41_TRELLIS_OA_GROUP_LIB"); e != nullptr && std::strcmp(e, "0") == 0) {
        m.why_not_grouped = "STRIX_DS41_TRELLIS_OA_GROUP_LIB=0";
    } else {
        // feature check: the query must accept the shape at R = 1..8, then the grouped rows must equal the staged rows
        // bitwise (the identity) at every R = 1..8 and again at R = 8..1 on these sites with a deterministic input
        std::uint64_t bytes = 0;
        const std::size_t xw = std::size_t{m.groups} * m.I, yw = std::size_t{m.groups} * m.N;
        const auto q = m.grouped(sites, reinterpret_cast<const float*>(std::uintptr_t{1} << 40U), xw, reinterpret_cast<float*>(std::uintptr_t{1} << 41U),
                                 yw, 8, true, &bytes);
        if (q != STRIX_SUCCESS) {
            m.why_not_grouped = std::string("grouped GEMV query: ") + strix_status_string(q);
        } else {
            m.grouped_bytes = std::max<std::uint64_t>((bytes + 255U) & ~std::uint64_t{255}, 256U);
            check(hipMalloc(&m.grouped_ws, m.grouped_bytes), "grouped workspace");
            std::vector<float> hx(8U * xw);
            std::uint64_t seed = 0x5eed0a9ULL;
            for (auto& v : hx) {
                seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                v = static_cast<float>(static_cast<std::int32_t>(seed >> 40U) % 20001) / 5000.0F;
            }
            float *dx = nullptr, *ya = nullptr, *yb = nullptr;
            check(hipMalloc(reinterpret_cast<void**>(&dx), hx.size() * 4U), "self-check x");
            check(hipMalloc(reinterpret_cast<void**>(&ya), 8U * yw * 4U), "self-check y");
            check(hipMalloc(reinterpret_cast<void**>(&yb), 8U * yw * 4U), "self-check y");
            check(hipMemcpyAsync(dx, hx.data(), hx.size() * 4U, hipMemcpyHostToDevice, m.stream), "self-check upload");
            bool same = true;
            std::uint32_t bad_r = 0;
            // two passes: the second reuses the plans of the first after every other row count ran on the workspace
            for (std::uint32_t pass = 0; pass < 16U && same; ++pass) {
                const std::uint32_t r = pass < 8U ? pass + 1U : 16U - pass;
                check(hipMemsetAsync(ya, 0xff, 8U * yw * 4U, m.stream), "self-check clear");
                check(hipMemsetAsync(yb, 0xfe, 8U * yw * 4U, m.stream), "self-check clear");
                m.staged(sites, dx, xw, ya, yw, r);
                const auto s = m.grouped(sites, dx, xw, yb, yw, r, false, nullptr);
                if (s != STRIX_SUCCESS) {
                    m.why_not_grouped = std::string("grouped GEMV submit: ") + strix_status_string(s);
                    same = false;
                    break;
                }
                std::vector<float> a(std::size_t{r} * yw), b(std::size_t{r} * yw);
                check(hipMemcpyAsync(a.data(), ya, a.size() * 4U, hipMemcpyDeviceToHost, m.stream), "self-check download");
                check(hipMemcpyAsync(b.data(), yb, b.size() * 4U, hipMemcpyDeviceToHost, m.stream), "self-check download");
                check(hipStreamSynchronize(m.stream), "self-check sync");
                if (std::memcmp(a.data(), b.data(), a.size() * 4U) != 0) {
                    same = false;
                    bad_r = r;
                }
            }
            (void)hipFree(dx);
            (void)hipFree(ya);
            (void)hipFree(yb);
            if (same) {
                m.grouped_lib = true;
            } else if (bad_r != 0U) {
                m.why_not_grouped = "grouped GEMV rows differ from the staged path at R" + std::to_string(bad_r);
            }
            for (const auto& [key, plan] : m.group_plans) (void)strix_grouped_gemv_plan_destroy(plan);
            m.group_plans.clear();  // self-check plans point at the self-check buffers
            if (!m.grouped_lib) {
                (void)hipFree(m.grouped_ws);
                m.grouped_ws = nullptr;
                std::fprintf(stderr, "ds41 trellis group: grouped library GEMV not used (%s); staged path\n", m.why_not_grouped.c_str());
            }
        }
    }
#endif
}

TrellisGroupRunner::~TrellisGroupRunner() {
    if (!impl_) return;
    auto& m = *impl_;
    (void)hipDeviceSynchronize();
    for (auto* p : m.lib) {
        m.owner->release_plans(p);
        (void)hipFree(p);
    }
#if defined(STRIX_TRELLIS_HAS_GROUPED_GEMV)
    for (const auto& [key, plan] : m.group_plans) (void)strix_grouped_gemv_plan_destroy(plan);
    m.group_plans.clear();
#endif
    (void)hipFree(m.xs);
    (void)hipFree(m.ys);
    (void)hipFree(m.grouped_ws);
    for (auto e : m.events) (void)hipEventDestroy(e);
    for (auto l : m.lanes) (void)hipStreamDestroy(l);
}

void TrellisGroupRunner::enqueue(const std::vector<const TrellisSite*>& sites, const float* x, std::size_t x_stride, float* y,
                                 std::size_t y_stride, std::uint32_t rows, void* stream) const {
    const auto& m = *impl_;
    if (rows == 0U) return;
    if (as_stream(stream) != m.stream || sites.size() != m.groups || x_stride < std::size_t{m.groups} * m.I ||
        y_stride < std::size_t{m.groups} * m.N)
        throw std::invalid_argument("ds41 trellis group: bad call");
    for (const auto* s : sites)
        if (s->columns != m.I || s->rows != m.N || s->bits != m.bits) throw std::invalid_argument("ds41 trellis group: shape");
#if defined(STRIX_TRELLIS_HAS_GROUPED_GEMV)
    if (m.grouped_lib) {
        lib_check(m.grouped(sites, x, x_stride, y, y_stride, rows, false, nullptr), "grouped gemv");
        return;
    }
#endif
    m.staged(sites, x, x_stride, y, y_stride, rows);
}

std::uint64_t TrellisGroupRunner::staging_bytes() const noexcept { return staging_bytes_for(impl_->groups, impl_->I, impl_->N); }
std::uint64_t TrellisGroupRunner::library_workspace_bytes() const noexcept {
    return impl_->lib_bytes * impl_->lib.size() + (impl_->grouped_ws != nullptr ? impl_->grouped_bytes : 0U);
}
std::string TrellisGroupRunner::describe() const {
    const auto& m = *impl_;
    return "o_a group mode " + std::to_string(m.mode) + " (" + std::to_string(m.groups) + " x " + std::to_string(m.I) + "->" +
           std::to_string(m.N) + ", " + (m.grouped_lib ? std::string("grouped library GEMV") : "staged: " + m.why_not_grouped) + ")";
}

#else  // no library

struct TrellisGroupRunner::Impl {};
TrellisGroupRunner::TrellisGroupRunner(const std::vector<const TrellisSite*>&, int, void*) {
    throw std::logic_error("ds41 trellis group: no library");
}
TrellisGroupRunner::~TrellisGroupRunner() = default;
void TrellisGroupRunner::enqueue(const std::vector<const TrellisSite*>&, const float*, std::size_t, float*, std::size_t, std::uint32_t,
                                 void*) const {
    throw std::logic_error("ds41 trellis group: no library");
}
std::string TrellisGroupRunner::describe() const { return "none"; }
std::uint64_t TrellisGroupRunner::staging_bytes() const noexcept { return 0; }
std::uint64_t TrellisGroupRunner::library_workspace_bytes() const noexcept { return 0; }

#endif

std::uint64_t TrellisGroupRunner::staging_bytes_for(std::uint32_t groups, std::uint32_t columns, std::uint32_t rows) noexcept {
    return std::uint64_t{groups} * 8U * (std::uint64_t{columns} + rows) * sizeof(float);  // xs [G][8][I], ys [G][8][N]
}

}  // namespace strix::ds41
