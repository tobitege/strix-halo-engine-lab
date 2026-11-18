#pragma once
// TURBO+ onfly INT4 g64 decode projections for the q8_rows sites (rows <= 8). A site opts in with
// `static constexpr unsigned turbo_group` (one bit of strix::turbo_onfly_groups); the launch uses the INT4 copies
// (q6p::iu4 registry, built at load by hip_qwen_onfly_i4.cpp) when EVERY segment of the site has one, else the
// Q8_0 kernels run unchanged. Shared kernel over the one W4A16 small-M core (core/w4a16.hpp): activations (BF16 / F32
// rows) are staged once per block as FP16, F32 accumulation, a block of S waves
// splits K and reduces its partial sums in LDS in a fixed order (deterministic), Site::store is the epilogue.
// Included from hip_q8_rows_gemv.hpp (namespace strix::q8_rows).


inline std::atomic<std::uint64_t>& iu4_dispatches() { static std::atomic<std::uint64_t> n{0}; return n; }
template <class Site> constexpr unsigned iu4_group_of() {
    if constexpr (requires { Site::turbo_group; }) return Site::turbo_group; else return 0U;
}
constexpr std::uint32_t kIu4StageBytes = 40U * 1024U;

struct Iu4Weights { const unsigned char* base[kMaxSegments]; };

// Plan per (rows R, groups G): S waves split the K groups of a pass, GC groups are staged per pass (G % GC == 0). Rows
// The control uses two 4-row halves for rows 7-8. Staged-dot single decode is
// selected for the measured G96 output shape on gfx1151; force is diagnostic.
using core::iu4_rows::Iu4Plan;
using core::iu4_rows::iu4_plan;

// Properties are cached per calling thread and current device, including device
// changes. The hot path only reads HIP's current-device index, not properties.
inline bool iu4_single_decode_device_qualified() {
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess) return false;
    static thread_local int cached_device = -1;
    static thread_local bool qualified = false;
    if (cached_device != device) {
        hipDeviceProp_t properties{};
        if (hipGetDeviceProperties(&properties, device) != hipSuccess) return false;
        qualified = core::iu4_rows::measured_single_decode_device(properties.gcnArchName,
                                                                 static_cast<unsigned>(properties.warpSize));
        cached_device = device;
    }
    return qualified;
}

// Grid: one block per (segment, 64 output columns); G = K / 64 groups in G / GC passes, S waves split each pass.
template <class Site, unsigned S, unsigned G, unsigned GC, bool FromSlot, bool SingleDecode = false>
__global__ __launch_bounds__(32 * S) void project_rows_iu4_kernel(Site site, Iu4Weights weights, const std::int8_t* codes,
                                                                  const float* deltas) {
    constexpr unsigned R = Site::rows, K = G * 64U, Kc = GC * 64U, MaxG = GC / S, Threads = 32U * S;
    constexpr bool StoreDots = SingleDecode;
    static_assert(G % GC == 0U && GC % S == 0U && MaxG % 2U == 0U && R >= 1U && R <= 8U && R * Kc * 2U <= kIu4StageBytes);
    static_assert(!SingleDecode || !StoreDots || R * Kc * 2U + R * 32U * S * sizeof(core::w4a16::Float2) <= 64U * 1024U);
    static_assert(Kc * 2U >= S * 256U, "partial sums alias the staged activations");
    __shared__ __attribute__((aligned(16))) _Float16 xs[R * Kc];
    __shared__ core::w4a16::Float2 staged_dot[SingleDecode && StoreDots && R >= 7U ? R * 32U * S : 1U];
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid / 32U;
    // Block -> (segment m, 64-column quad).
    unsigned m = 0, quad = blockIdx.x;
    for (;; ++m) {
        const unsigned q = ((site.segments.first[m + 1U] - site.segments.first[m]) + 63U) / 64U;
        if (quad < q || m + 1U == site.segments.count) break;
        quad -= q;
    }
    const std::uint32_t n_real = site.segments.first[m + 1U] - site.segments.first[m], n_pad = (n_real + 63U) / 64U * 64U;
    const auto* wq = weights.base[m];
    const auto* ws = reinterpret_cast<const std::uint16_t*>(wq + core::onfly::packed_bytes(n_pad, K));
    float res[R][2] = {};
    for (unsigned pass = 0; pass < G / GC; ++pass) {
        const unsigned c0 = pass * Kc;
        if (pass) __syncthreads();  // the previous pass's waves are done reading the staged rows
        // Stage this pass's activations as FP16 (finite values saturate, NaN propagates).
        for (unsigned r = 0; r < R; ++r)
            for (unsigned i = tid * 8U; i < Kc; i += Threads * 8U) {
                float x[8];
                if constexpr (FromSlot) {
                    const float delta = deltas[((c0 + i) / 32U) * dense_a8_pitch<R>() + r];
                    const std::int8_t* c = codes + static_cast<std::uint64_t>(r) * K + c0 + i;
#pragma unroll
                    for (unsigned j = 0; j < 8U; ++j) x[j] = static_cast<float>(c[j]) * delta;
                } else load8(row_input(site, r) + c0 + i, x);
                _Float16 h[8];
#pragma unroll
                for (unsigned j = 0; j < 8U; ++j) h[j] = static_cast<_Float16>(x[j] != x[j] ? x[j] : fminf(65504.0F, fmaxf(-65504.0F, x[j])));
                *reinterpret_cast<uint4*>(xs + r * Kc + i) = *reinterpret_cast<const uint4*>(h);
            }
        __syncthreads();
        const unsigned kl = wave * MaxG * 4U, k0 = pass * GC * 4U + kl;
        // Keep each row's dot/group/reduction order unchanged. The full-row form
        // reuses the same nibble decode across both halves; the control limits VGPR use.
        constexpr unsigned Rc = R >= 7U && !SingleDecode ? 4U : R;
#pragma unroll
        for (unsigned h = 0; h * Rc < R; ++h)
            core::w4a16::wave_rows<Rc, MaxG * 4U, MaxG, SingleDecode, StoreDots>(wq, ws, n_pad, K, k0, k0 + MaxG * 4U, quad * 4U, xs + h * Rc * Kc + kl * 16U, Kc,
                lane, R - h * Rc < Rc ? R - h * Rc : Rc,
                [&](unsigned, unsigned r, unsigned column, float value) { res[h * Rc + r][column & 1U] += value; },
                SingleDecode && StoreDots ? staged_dot + wave * R * 32U : nullptr);
    }
    __syncthreads();  // every wave is done reading the staged rows: reuse them for the partial sums
    float* part = reinterpret_cast<float*>(xs);
    const unsigned rel = (lane >> 3U) * 16U + 2U * (lane & 7U);
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
        part[(r * S + wave) * 64U + rel] = res[r][0];
        part[(r * S + wave) * 64U + rel + 1U] = res[r][1];
    }
    __syncthreads();
    for (unsigned t = tid; t < R * 64U; t += Threads) {
        const unsigned r = t / 64U, c = t % 64U;
        float sum = part[(r * S) * 64U + c];
#pragma unroll
        for (unsigned s = 1; s < S; ++s) sum += part[(r * S + s) * 64U + c];
        if (quad * 64U + c < n_real) site.store(site.segments.first[m] + quad * 64U + c, r, sum);
    }
}

// True when INT4 launched. A8 slots remain exclusively for the Q8 fallback.
// Even with a slot, use the original site rows and the serial FP16 conversion.
template <class Site> bool launch_iu4_rows(const Site& site, hipStream_t stream, bool any_order, const DenseA8Slot* slot) {
    if constexpr (Site::rows > 8U || !iu4_group_of<Site>()) return false;
    else {
        if (!q6p::iu4::count() || q6p::iu4::suspended()) return false;
        const unsigned width = site.width;
        if (!onfly_i4::dispatch_width(width) || site.segments.count == 0U ) return false;
        Iu4Weights w{};
        std::shared_ptr<const q6p::iu4::Tensor> leases[kMaxSegments];
        unsigned blocks = 0;
        for (unsigned m = 0; m < site.segments.count; ++m) {
            const auto t = q6p::iu4::find(site.segments.weights[m]);
            const unsigned n = site.segments.first[m + 1U] - site.segments.first[m];
            if (!t || t->cols != width || t->rows != n || site.segments.types[m]) return false;
            w.base[m] = t->base;
            leases[m] = t;
            blocks += (n + 63U) / 64U;
        }
        onfly_i4::require_serial_projection(Site::rows, false);
        const auto& execution = core::iu4_rows::execution_policy();
        const auto reduction = execution.reduction;
        const auto run = [&]<unsigned G, core::iu4_rows::RowReduction Reduction>() {
            constexpr auto plan = iu4_plan(Site::rows, G, Reduction);
            constexpr unsigned S = plan.S, GC = plan.GC;
            const auto go = [&](auto kernel, const std::int8_t* codes, const float* deltas) {
                if (any_order && !slot)
                    hipExtLaunchKernelGGL(kernel, dim3(blocks), dim3(32U * S), 0, stream, nullptr, nullptr, hipExtAnyOrderLaunch, site, w,
                        codes, deltas);
                else hipLaunchKernelGGL(kernel, dim3(blocks), dim3(32U * S), 0, stream, site, w, codes, deltas);
            };
            if constexpr (Site::rows * GC * 128U <= kIu4StageBytes) {  // staged FP16 rows must fit the LDS budget
                if constexpr (Site::rows >= 7U) {
                    const auto policy = execution.single_decode;
                    const unsigned outputs = site.segments.first[site.segments.count] - site.segments.first[0];
                    const bool device_qualified = policy == core::iu4_rows::SingleDecodePolicy::automatic &&
                        core::iu4_rows::measured_single_decode_admitted(Site::rows, G, outputs) && iu4_single_decode_device_qualified();
                    if (core::iu4_rows::single_decode_applies(Site::rows, G, policy, outputs, device_qualified)) {
                        core::iu4_rows::single_decode_launches.fetch_add(1, std::memory_order_relaxed);
                        if (slot) go(project_rows_iu4_kernel<Site, S, G, GC, true, true>, slot->codes, slot->deltas);
                        else go(project_rows_iu4_kernel<Site, S, G, GC, false, true>, nullptr, nullptr);
                    } else {
                        if (slot) go(project_rows_iu4_kernel<Site, S, G, GC, true>, slot->codes, slot->deltas);
                        else go(project_rows_iu4_kernel<Site, S, G, GC, false>, nullptr, nullptr);
                    }
                } else {
                    if (slot) go(project_rows_iu4_kernel<Site, S, G, GC, true>, slot->codes, slot->deltas);
                    else go(project_rows_iu4_kernel<Site, S, G, GC, false>, nullptr, nullptr);
                }
            }
        };
        const auto dispatch = [&]<unsigned G>() {
            if constexpr (iu4_plan(Site::rows, G).GC != iu4_plan(Site::rows, G, core::iu4_rows::RowReduction::canonical).GC) {
                if (reduction == core::iu4_rows::RowReduction::canonical)
                    run.template operator()<G, core::iu4_rows::RowReduction::canonical>();
                else run.template operator()<G, core::iu4_rows::RowReduction::legacy>();
            } else run.template operator()<G, core::iu4_rows::RowReduction::legacy>();
        };
        switch (width / 64U) {
        case 32U: dispatch.template operator()<32U>(); break;
        case 40U: dispatch.template operator()<40U>(); break;
        case 64U: dispatch.template operator()<64U>(); break;
        case 96U: dispatch.template operator()<96U>(); break;
        default: return false;
        }
        iu4_dispatches().fetch_add(1U, std::memory_order_relaxed);
        q6p::iu4::launches.fetch_add(1U, std::memory_order_relaxed);
        return true;
    }
}


