// GLM-5.3-Flash attention Tier B fused dispatches (identity and structure: glm_attn_tierb.hpp, design
// design notes). The CTA body is MoE gate_up_body / wide_core (src/glm/moe/glm_moe.hip, qualified
// bitwise against strix_gemv in S7), generalized to the attention shapes: compile-time geometry per member, any SG,
// optional weight-ring prime before the input transform, outputs at a row stride.
#include <atomic>
#include "glm/attn/glm_attn_tierb.hpp"

#include "glm/moe/glm_trellis_device.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace strix::glm::attn_tierb {
namespace {

namespace d = strix::glm::moe::dev;

constexpr unsigned kBits = 6;
constexpr unsigned kTileU = 2U * kBits;             // uint4 per 16x16 K6 tile (192 B)
constexpr unsigned kDepth = 4;                      // k-rows in flight per wave
// TPW = 16-column tiles per wave (4 or 2): the wave's k-row run is TPW contiguous tiles (TPW x 192 B), the CTA (4 waves)
// covers 4 TPW tiles = 64 TPW outputs = TPW / 2 H128 blocks.
template <unsigned TPW> constexpr unsigned run_u() { return kTileU * TPW; }
template <unsigned TPW> constexpr unsigned upl() { return (run_u<TPW>() + 31U) / 32U; }
using V4 = std::uint32_t __attribute__((ext_vector_type(4)));

// Member geometry: library split count S (5acf090 measured_vector_splits; no GLM attention shape is special-cased),
// k-tiles per split, 256-column CTA groups, H128 output blocks.
template <unsigned I, unsigned N> struct Geo {
    static constexpr unsigned In = I, Out = N;
    static constexpr unsigned S = d::vector_splits(I, N);
    static constexpr unsigned Kt = I / 16U, Kps = Kt / S, Cols = N / 16U, Blocks = N / 128U;
    static_assert(Kt % S == 0 && Kps % kDepth == 0 && N % 256U == 0 && (Kps * 16U) % 128U == 0, "attention K6 geometry");
    template <unsigned TPW> static constexpr unsigned cg() { return N / (64U * TPW); }  // column groups
    template <unsigned SG, unsigned TPW> static constexpr unsigned ctas() { return cg<TPW>() * (S / SG); }
};
using GQkv = Geo<4096, 4096>;   // KDA q / k / v / o
using GQa = Geo<4096, 1536>;    // DSA q_a
using GKva = Geo<4096, 512>;    // DSA kv_a
using GQb = Geo<1536, 8192>;    // DSA q_b
using GWqb = Geo<1536, 4096>;   // DSA wq_b (indexer)
using GDo = Geo<8192, 4096>;    // DSA o_proj
static_assert(GQkv::S == 16 && GQa::S == 16 && GKva::S == 32 && GQb::S == 4 && GWqb::S == 4 && GDo::S == 16,
              "strix-trellis 5acf090 split policy for the GLM attention shapes");

// Workspace: per member slot the raw split partials [S][kMaxRows][N] F32 (2 MiB covers every member), then counters
// (64 H128 blocks per member slot).
constexpr std::size_t kSlotBytes = std::size_t{2} << 20U;
constexpr unsigned kSlots = 3, kCntPerSlot = 64;
constexpr std::size_t kCntOff = kSlots * kSlotBytes;
constexpr std::size_t kWsBytes = kCntOff + std::size_t{kSlots} * kCntPerSlot * 4U;
static_assert(std::size_t{GQkv::S} * kMaxRows * GQkv::Out * 4U <= kSlotBytes && std::size_t{GDo::S} * kMaxRows * GDo::Out * 4U <= kSlotBytes &&
                  std::size_t{GQb::S} * kMaxRows * GQb::Out * 4U <= kSlotBytes && std::size_t{GKva::S} * kMaxRows * GKva::Out * 4U <= kSlotBytes &&
                  GQb::Blocks <= kCntPerSlot,
              "workspace slots");

struct MArgs {
    const std::uint32_t* trellis;
    const std::uint16_t* suh;
    const std::uint16_t* svh;
    const float* x;
    float* y;
    unsigned x_stride, y_stride;
    float* part;            // [S][kMaxRows][N]
    std::uint32_t* cnt;     // [N / 128]
};
struct Args {
    MArgs m0, m1, m2;
    unsigned rows;
    unsigned pub_blocks;               // 128-output blocks of member 0 (the whole partial) when pub is on
    strix::pair::ProducerPublish pub;  // o_proj launches: member 0 publishes its output (producer_publish.hpp)
};

__device__ __forceinline__ float load_agent(const float* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }

// True in every thread of the CTA that arrives last at `counter` (expected arrivals); that CTA resets the counter.
// Publication: every thread fences its own partial stores at agent scope (release), the barrier orders them before
// thread 0's agent-scope acq_rel RMW (release of the CTA's partials, acquire of every earlier arrival's release
// sequence); the result is broadcast through LDS + barrier, and every thread of the last CTA executes an agent-scope
// acquire fence before it reads the other CTAs' partials (agent-scope atomic loads, never stale non-coherent cache
// lines). The reset needs no ordering: the workspace is reused only by a later, stream-ordered launch.
__device__ __forceinline__ bool last_arrival(std::uint32_t* counter, unsigned expected) {
    __shared__ unsigned last;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned old = __hip_atomic_fetch_add(counter, 1U, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        last = old + 1U == expected ? 1U : 0U;
        if (last) __hip_atomic_store(counter, 0U, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    const bool is_last = last != 0;
    if (is_last) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    return is_last;
}

using GV4 = const V4 __attribute__((address_space(1)));

// Input transform of one H128 block of one row (strix-trellis input_transform = d::transform_block_at, same ops),
// split into its loads and its arithmetic so the four x / suh loads per lane are issued together (the materialize
// asm statements would otherwise serialize them, one vmcnt(0) each).
struct XLoad {
    float x[4];
    std::uint16_t s[4];
};
__device__ __forceinline__ XLoad xload(const float* x, const std::uint16_t* suh, unsigned begin, unsigned lane) {
    XLoad l;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = begin + lane + 32U * static_cast<unsigned>(j);
        l.x[j] = x[c];
        l.s[j] = suh[c];
    }
    return l;
}
__device__ __forceinline__ void xstore(const XLoad& l, _Float16* out, unsigned obegin, unsigned lane) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float scaled = d::materialize(l.x[j] * d::half_bits(l.s[j]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    d::hadamard128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) out[obegin + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
}
// CTA barrier for LDS data only: waits for this wave's LDS operations, not for its outstanding global loads (the
// weight ring stays in flight; __syncthreads would emit s_waitcnt vmcnt(0)).
__device__ __forceinline__ void lds_barrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }

// Wave-private LDS row: 32 * upl uint4 (every lane stores unconditionally; slots past the run are never read).
template <unsigned TPW> constexpr unsigned rowbuf_u() { return 32U * upl<TPW>(); }
template <class G, int R, unsigned SG, unsigned TPW> constexpr std::size_t smem_bytes() {
    return std::size_t{static_cast<unsigned>(R)} * SG * G::Kps * 16U * 2U + 4U * rowbuf_u<TPW>() * 16U;
}

// One CTA of member geometry G: 4 waves x TPW tiles (64 TPW outputs, H128 blocks cg * TPW / 2 + h) over the SG library
// splits [sgi * SG, (sgi + 1) * SG). Core = glm_moe.hip wide_core (Store form), generalized: per k-row the wave's
// TPW x 192 contiguous bytes go through a kDepth-deep register ring and a wave-private LDS row; lane chains in the
// library order; at every split end the 4-lane reduction and the split partial store at its GLOBAL split index
// (plane stride kMaxRows * N, row stride N). Finish = output_transform of the block.
template <class G, int R, unsigned SG, bool Prime, unsigned TPW>
__device__ __forceinline__ unsigned member_cta(const MArgs a, unsigned local, unsigned rows, unsigned char* smem) {
    constexpr unsigned kRunU = run_u<TPW>(), kUpl = upl<TPW>(), Cg = G::template cg<TPW>();
    constexpr unsigned Ch = SG * G::Kps * 16U;  // channels of the CTA (whole H128 blocks)
    constexpr unsigned Xb = Ch / 128U;
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    auto* xs = reinterpret_cast<_Float16*>(smem);
    V4* rowbuf = reinterpret_cast<V4*>(smem + std::size_t{static_cast<unsigned>(R)} * Ch * 2U) + wave * rowbuf_u<TPW>();
    const unsigned cg = local % Cg, sgi = local / Cg;
    const unsigned k0 = sgi * SG * G::Kps;
    const unsigned col0 = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>((cg * 4U + wave) * TPW)));
    // Global address space explicitly: the readfirstlane-rebuilt pointer is generic and would give flat loads, which
    // also count on lgkmcnt, so every LDS wait of the loop would drain the weight ring.
    const auto* gw = (GV4*)(reinterpret_cast<const V4*>(d::uniform(a.trellis)));
    constexpr std::size_t kstride = std::size_t{G::Cols} * 2U * kBits;  // uint4 per k-row of the matrix
    const std::size_t cbase = static_cast<std::size_t>(col0) * 2U * kBits;
    V4 ring[kDepth][kUpl];
    // Branch-free: lanes past the run re-load its last 16 bytes (same cache line, never stored). A predicated load sits
    // behind s_cbranch_execz and the waitcnt pass then assumes it may not have been issued: it waited vmcnt(3) instead
    // of vmcnt(6) at every step, i.e. the 4-deep ring ran ~1.5 rows ahead.
    auto load = [&](unsigned k, auto slot) __attribute__((always_inline)) {
        constexpr unsigned q = decltype(slot)::value;
#pragma unroll
        for (unsigned u = 0; u < kUpl; ++u) {
            const unsigned e = lane + 32U * u;
            ring[q][u] = gw[static_cast<std::size_t>(k) * kstride + cbase + (e < kRunU ? e : kRunU - 1U)];
        }
    };
    using S0 = std::integral_constant<unsigned, 0>;
    using S1 = std::integral_constant<unsigned, 1>;
    using S2 = std::integral_constant<unsigned, 2>;
    using S3 = std::integral_constant<unsigned, 3>;
    // Issue order pinned (sched barriers): the scheduler otherwise reordered the four rows (slot 0 last); with in-order
    // vmcnt the loop header then had to wait for nearly everything, and the waitcnt merge carried vmcnt(1)/vmcnt(0)
    // into every iteration of the loop.
    auto prime = [&]() __attribute__((always_inline)) {
        __builtin_amdgcn_sched_barrier(0);
        load(k0 + 0, S0{});
        __builtin_amdgcn_sched_barrier(0);
        load(k0 + 1, S1{});
        __builtin_amdgcn_sched_barrier(0);
        load(k0 + 2, S2{});
        __builtin_amdgcn_sched_barrier(0);
        load(k0 + 3, S3{});
        __builtin_amdgcn_sched_barrier(0);
    };
    // Input transform of the CTA's channels [k0 * 16, k0 * 16 + Ch) of every active row (library input_transform ops).
    // The wave's first job loads its x / suh values before the weight ring is primed: vector-memory counters retire in
    // issue order, so the transform then waits only for its own (older) loads, not for the primed weight rows.
    // Early prime only when no wave has a second transform job (R * Xb <= 4, static): a later job's x loads are younger
    // than the primed rows, so waiting for them would drain the ring anyway; otherwise prime after the transform.
    constexpr bool kEarly = Prime && static_cast<unsigned>(R) * Xb <= 4U;
    const unsigned njobs = rows * Xb;
    XLoad first{};
    if (wave < njobs) first = xload(a.x + static_cast<std::size_t>(wave / Xb) * a.x_stride, a.suh, k0 * 16U + (wave % Xb) * 128U, lane);
    if constexpr (kEarly) prime();
    if (wave < njobs) xstore(first, xs + (wave / Xb) * Ch, (wave % Xb) * 128U, lane);
    for (unsigned job = wave + 4U; job < njobs; job += 4U) {
        const unsigned lr = job / Xb, b = job % Xb;
        const XLoad xl = xload(a.x + static_cast<std::size_t>(lr) * a.x_stride, a.suh, k0 * 16U + b * 128U, lane);
        xstore(xl, xs + lr * Ch, b * 128U, lane);
    }
    lds_barrier();
    if constexpr (!kEarly) prime();

    const d::LaneGeom<kBits> geo(lane);
    float sums[TPW][R][2];
#pragma unroll
    for (unsigned c = 0; c < TPW; ++c)
#pragma unroll
        for (int r = 0; r < R; ++r) sums[c][r][0] = sums[c][r][1] = 0.0F;
    const auto* rl = (d::LocalU32*)(reinterpret_cast<const std::uint32_t*>(rowbuf));
    const auto* xw = (d::LocalU32*)(reinterpret_cast<const std::uint32_t*>(xs)) + (lane & 3U);
    auto split_end = [&](unsigned split) __attribute__((always_inline)) {
        float* const ps = a.part + static_cast<std::size_t>(sgi * SG + split) * kMaxRows * G::Out;  // wave-uniform
#pragma unroll
        for (unsigned c = 0; c < TPW; ++c)
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float v0 = d::reduce4(sums[c][r][0]), v1 = d::reduce4(sums[c][r][1]);
                if ((lane & 3U) == 0 && static_cast<unsigned>(r) < rows) {
                    float* pr = ps + static_cast<std::size_t>(r) * G::Out + (col0 + c) * 16U + lane / 4U;
                    pr[0] = v0;
                    pr[8] = v1;
                }
                sums[c][r][0] = sums[c][r][1] = 0.0F;
            }
    };
    // Unconditional LDS store (padded row) and a compile-time Load flag instead of `if (k + kDepth < k1)`: a divergent
    // store block or a conditionally issued load makes the waitcnt pass fall back to vmcnt(0) inside the loop.
    auto step = [&](unsigned k, auto slot, auto with_load) __attribute__((always_inline)) {
        constexpr unsigned q = decltype(slot)::value;
#pragma unroll
        for (unsigned u = 0; u < kUpl; ++u) rowbuf[lane + 32U * u] = ring[q][u];
        if constexpr (decltype(with_load)::value) load(k + kDepth, slot);
        d::Half2 act[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const auto* pr = xw + (static_cast<unsigned>(r) * Ch) / 2U + (k - k0) * 8U;
            act[r][0] = __builtin_bit_cast(d::Half2, pr[0]);
            act[r][1] = __builtin_bit_cast(d::Half2, pr[4]);
        }
#pragma unroll
        for (unsigned c = 0; c < TPW; ++c) {
            d::Slot<kBits, R> cur;
            const unsigned tw = c * kBits * 8U;
            cur.a = rl[tw + geo.oa / 4U];
            cur.b = rl[tw + geo.ob / 4U];
            cur.c = rl[tw + geo.oc / 4U];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                cur.act[r][0] = act[r][0];
                cur.act[r][1] = act[r][1];
            }
            d::compute_slot<kBits, R>(cur, geo, static_cast<unsigned>(R), sums[c]);
        }
        // Keep the step boundaries: without this the scheduler hoisted the four steps' LDS stores to the loop top and
        // sank all four refill loads to the bottom, so every iteration waited a full memory latency for its rows.
        __builtin_amdgcn_sched_barrier(0);
    };
    // k-rows [k0, k0 + kRows) in groups of kDepth; split ends fall on group boundaries (Kps % kDepth == 0). The last
    // group issues no loads (peeled at compile time).
    constexpr unsigned kRows = SG * G::Kps;
    using Ld = std::true_type;
    using NoLd = std::false_type;
    for (unsigned kk = 0; kk + kDepth < kRows; kk += kDepth) {
        const unsigned k = k0 + kk;
        step(k, S0{}, Ld{});
        step(k + 1, S1{}, Ld{});
        step(k + 2, S2{}, Ld{});
        step(k + 3, S3{}, Ld{});
        if ((kk + kDepth) % G::Kps == 0) split_end((kk + kDepth) / G::Kps - 1U);
    }
    {
        const unsigned k = k0 + kRows - kDepth;
        step(k, S0{}, NoLd{});
        step(k + 1, S1{}, NoLd{});
        step(k + 2, S2{}, NoLd{});
        step(k + 3, S3{}, NoLd{});
        split_end(SG - 1U);
    }

    // Output transform of a finished H128 block: acc = ((+0 + p_0) + p_1) + ... (ascending splits), H32, x svh.
    auto finish = [&](unsigned block) {
        for (unsigned lr = wave; lr < rows; lr += 4U) {
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = 0.0F;
            for (unsigned sp = 0; sp < G::S; ++sp) {
                const float* p = a.part + (static_cast<std::size_t>(sp) * kMaxRows + lr) * G::Out + block * 128U + lane;
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] += load_agent(p + 32 * j);
            }
            d::hadamard128(v, lane);
            float* out = a.y + static_cast<std::size_t>(lr) * a.y_stride + block * 128U + lane;
#pragma unroll
            for (int j = 0; j < 4; ++j) out[32 * j] = v[j] * d::half_bits(a.svh[block * 128U + lane + 32U * static_cast<unsigned>(j)]);
        }
    };
    unsigned finished = 0;  // output blocks completed by this CTA (uniform over its threads)
#pragma unroll
    for (unsigned h = 0; h < TPW / 2U; ++h)
        if (last_arrival(a.cnt + cg * (TPW / 2U) + h, G::S / SG)) {
            finish(cg * (TPW / 2U) + h);
            ++finished;
        }
    return finished;
}

// By value, explicit branches: a runtime-indexed member array would put Args in scratch.
__device__ __forceinline__ MArgs member(const Args& a, unsigned m) { return m == 0 ? a.m0 : (m == 1 ? a.m1 : a.m2); }

// Grid: M0 members of geometry G0 then M1 members of G1, each member's CTAs consecutive along x.
template <class G0, class G1, int R, unsigned SG, unsigned TPW>
constexpr std::size_t kGroupSmem = smem_bytes<G0, R, SG, TPW>() > smem_bytes<G1, R, SG, TPW>() ? smem_bytes<G0, R, SG, TPW>()
                                                                                              : smem_bytes<G1, R, SG, TPW>();
template <class G0, unsigned M0, class G1, unsigned M1, int R, unsigned SG, bool Prime, unsigned TPW>
__global__ __launch_bounds__(128) void group_kernel(Args a) {
    __shared__ alignas(16) unsigned char smem[kGroupSmem<G0, G1, R, SG, TPW>];
    constexpr unsigned n0 = G0::template ctas<SG, TPW>(), n1 = G1::template ctas<SG, TPW>();
    const unsigned b = blockIdx.x;
    if (b < M0 * n0) {
        const unsigned finished = member_cta<G0, R, SG, Prime, TPW>(member(a, b / n0), b % n0, a.rows, smem);
        if constexpr (M0 == 1 && M1 == 0) {  // single-member launches (o_proj): the producer may publish its partial
            if (a.pub.on() && finished != 0) strix::pair::producer_publish(a.pub, a.pub_blocks, finished);
        } else {
            (void)finished;
        }
    } else if constexpr (M1 > 0) {
        const unsigned b1 = b - M0 * n0;
        (void)member_cta<G1, R, SG, Prime, TPW>(member(a, M0 + b1 / n1), b1 % n1, a.rows, smem);
    }
}

void check_launch(const char* what) {
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("glm attn tierb: launch failed: ") + what + ": " + hipGetErrorString(e));
}

// query != nullptr: no launch; *query = the instantiation's private (scratch) bytes per lane.
template <class G0, unsigned M0, class G1, unsigned M1, int R, unsigned SG, bool Prime, unsigned TPW>
void launch4(const Args& a, hipStream_t s, std::size_t* query) {
    if (query != nullptr) {
        hipFuncAttributes fa{};
        const hipError_t e = hipFuncGetAttributes(&fa, reinterpret_cast<const void*>(&group_kernel<G0, M0, G1, M1, R, SG, Prime, TPW>));
        if (e != hipSuccess) throw std::runtime_error(std::string("glm attn tierb: attributes: ") + hipGetErrorString(e));
        *query = fa.localSizeBytes;
        return;
    }
    const unsigned grid = M0 * G0::template ctas<SG, TPW>() + M1 * G1::template ctas<SG, TPW>();
    group_kernel<G0, M0, G1, M1, R, SG, Prime, TPW><<<grid, 128, 0, s>>>(a);
    check_launch("group");
}
template <class G0, unsigned M0, class G1, unsigned M1, int R, unsigned SG>
void launch3(const Args& a, Knobs k, hipStream_t s, std::size_t* q) {
    const bool p = k.prime != 0;
    if (k.tpw == 2) p ? launch4<G0, M0, G1, M1, R, SG, true, 2>(a, s, q) : launch4<G0, M0, G1, M1, R, SG, false, 2>(a, s, q);
    else p ? launch4<G0, M0, G1, M1, R, SG, true, 4>(a, s, q) : launch4<G0, M0, G1, M1, R, SG, false, 4>(a, s, q);
}
template <class G0, unsigned M0, class G1, unsigned M1, int R>
void launch2(const Args& a, Knobs k, hipStream_t s, std::size_t* q) {
    switch (k.sg) {
    case 2: launch3<G0, M0, G1, M1, R, 2>(a, k, s, q); break;
    case 4: launch3<G0, M0, G1, M1, R, 4>(a, k, s, q); break;
    default: launch3<G0, M0, G1, M1, R, 1>(a, k, s, q); break;
    }
}
template <class G0, unsigned M0, class G1, unsigned M1>
void launch(const Args& a, Knobs k, hipStream_t s, std::size_t* q = nullptr) {
    if (k.sg != 1 && k.sg != 2 && k.sg != 4) throw std::invalid_argument("glm attn tierb: SG must be 1, 2 or 4");
    if (k.tpw != 2 && k.tpw != 4) throw std::invalid_argument("glm attn tierb: TPW must be 2 or 4");
    if (a.rows == 1) launch2<G0, M0, G1, M1, 1>(a, k, s, q);
    else if (a.rows == 2) launch2<G0, M0, G1, M1, 2>(a, k, s, q);
    else if (a.rows <= 4) launch2<G0, M0, G1, M1, 4>(a, k, s, q);
    else launch2<G0, M0, G1, M1, 8>(a, k, s, q);
}

bool shape_ok(const Mat& m, unsigned in, unsigned out) {
    return m.trellis != nullptr && m.suh != nullptr && m.svh != nullptr && m.bits == kBits && m.input == in && m.output == out &&
           (reinterpret_cast<std::uintptr_t>(m.trellis) & 15U) == 0 && (reinterpret_cast<std::uintptr_t>(m.suh) & 1U) == 0 &&
           (reinterpret_cast<std::uintptr_t>(m.svh) & 1U) == 0;
}
template <class G> bool geo_ok(const Mat& m) { return shape_ok(m, G::In, G::Out); }

MArgs margs(const Mat& m, const float* x, unsigned xs, float* y, unsigned ys, void* ws, unsigned slot) {
    auto* base = static_cast<unsigned char*>(ws);
    return MArgs{m.trellis, m.suh, m.svh, x, y, xs, ys, reinterpret_cast<float*>(base + slot * kSlotBytes),
                 reinterpret_cast<std::uint32_t*>(base + kCntOff) + slot * kCntPerSlot};
}
void check_rows(unsigned rows, const float* x, unsigned xs, unsigned in, void* ws) {
    if (rows < 1 || rows > kMaxRows) throw std::invalid_argument("glm attn tierb: rows must be 1..8");
    if (x == nullptr || (reinterpret_cast<std::uintptr_t>(x) & 3U) != 0 || (rows > 1 && xs < in))
        throw std::invalid_argument("glm attn tierb: input");
    if (ws == nullptr || (reinterpret_cast<std::uintptr_t>(ws) & 255U) != 0)
        throw std::invalid_argument("glm attn tierb: workspace (workspace_bytes(), 256-byte aligned)");
}
void check_out(const float* y, unsigned ys, unsigned out, unsigned rows) {
    if (y == nullptr || (reinterpret_cast<std::uintptr_t>(y) & 3U) != 0 || (rows > 1 && ys < out))
        throw std::invalid_argument("glm attn tierb: output");
}
// Default schedule per (group, rows): measured table (measurement node 0 + measurement node 1 benches, design notes,
// SCREEN of both nodes summed; implementation review). Every entry is one of the 12 gated knob combinations, so the bits are
// those of strix_gemv whatever the entry; the table only changes speed. Entries whose measured gain over the legacy
// default (SG 1, prime, TPW 4; DSA a R > 4 TPW 2) was < 0.5 % keep the legacy default. No entry has scratch (TPW 4 is
// never chosen for DSA a at R > 4). STRIX_GLM_TIERB_TABLE=0 restores the legacy defaults; explicit knobs (env or
// argument) override per field as before and then disable the table for that call.
struct KnobRow { unsigned char sg, prime, tpw; };
constexpr KnobRow kTable[5][8] = {
    // KdaQkv R1..8
    {{1, 1, 2}, {1, 1, 2}, {1, 1, 4}, {1, 1, 4}, {1, 1, 2}, {1, 1, 2}, {1, 0, 2}, {1, 1, 2}},
    // KdaO
    {{2, 1, 2}, {4, 0, 2}, {2, 0, 2}, {2, 1, 2}, {1, 1, 4}, {1, 1, 2}, {1, 1, 4}, {1, 1, 2}},
    // DsaA
    {{4, 1, 2}, {4, 1, 2}, {4, 1, 2}, {4, 0, 2}, {2, 0, 2}, {2, 0, 2}, {2, 1, 2}, {2, 0, 2}},
    // DsaB
    {{4, 1, 2}, {4, 1, 2}, {4, 1, 2}, {4, 0, 2}, {1, 1, 2}, {1, 1, 2}, {1, 1, 2}, {1, 1, 2}},
    // DsaO
    {{1, 1, 2}, {4, 1, 2}, {1, 1, 4}, {1, 1, 4}, {1, 1, 4}, {1, 1, 4}, {1, 1, 4}, {1, 1, 4}},
};
std::atomic<int> g_table{-1};  // -1 not read yet; 0 legacy defaults; 1 table (set_table_enabled overrides the env)
bool table_enabled() {
    int v = g_table.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* s = std::getenv("STRIX_GLM_TIERB_TABLE");
        const int want = (s == nullptr || *s == '\0' || std::stoi(s) != 0) ? 1 : 0;
        g_table.compare_exchange_strong(v, want);
        v = g_table.load(std::memory_order_relaxed);
    }
    return v == 1;
}
Knobs resolve(Knobs k, Group g, unsigned rows) {
    if (k.sg == 0 && k.prime < 0 && k.tpw == 0 && rows >= 1 && rows <= kMaxRows && table_enabled()) {
        const KnobRow& t = kTable[static_cast<unsigned>(g)][rows - 1];
        return Knobs{t.sg, static_cast<int>(t.prime), t.tpw};
    }
    if (k.sg == 0) k.sg = 1;
    if (k.prime < 0) k.prime = 1;
    if (k.tpw == 0) k.tpw = g == Group::DsaA && rows > 4 ? 2 : 4;
    return k;
}

}  // namespace

bool supported(Group g, const Mat* m, unsigned count) {
    if (m == nullptr) return false;
    switch (g) {
    case Group::KdaQkv: return count == 3 && geo_ok<GQkv>(m[0]) && geo_ok<GQkv>(m[1]) && geo_ok<GQkv>(m[2]);
    case Group::KdaO: return count == 1 && geo_ok<GQkv>(m[0]);
    case Group::DsaA: return count == 2 && geo_ok<GQa>(m[0]) && geo_ok<GKva>(m[1]);
    case Group::DsaB: return count == 2 && geo_ok<GQb>(m[0]) && geo_ok<GWqb>(m[1]);
    case Group::DsaO: return count == 1 && geo_ok<GDo>(m[0]);
    }
    return false;
}

std::size_t workspace_bytes() { return kWsBytes; }

std::size_t scratch_bytes(Group g, unsigned rows, Knobs k) {
    if (rows < 1 || rows > kMaxRows) throw std::invalid_argument("glm attn tierb: rows must be 1..8");
    Args a{};
    a.rows = rows;
    std::size_t q = 0;
    const Knobs r = resolve(k, g, rows);
    switch (g) {
    case Group::KdaQkv: launch<GQkv, 3, GQkv, 0>(a, r, nullptr, &q); break;
    case Group::KdaO: launch<GQkv, 1, GQkv, 0>(a, r, nullptr, &q); break;
    case Group::DsaA: launch<GQa, 1, GKva, 1>(a, r, nullptr, &q); break;
    case Group::DsaB: launch<GQb, 1, GWqb, 1>(a, r, nullptr, &q); break;
    case Group::DsaO: launch<GDo, 1, GDo, 0>(a, r, nullptr, &q); break;
    }
    return q;
}
std::size_t workspace_counters_offset() { return kCntOff; }
unsigned workspace_counters() { return kSlots * kCntPerSlot; }

Knobs env_knobs() {
    static const Knobs k = [] {
        Knobs r;
        if (const char* s = std::getenv("STRIX_GLM_TIERB_SG"); s != nullptr && *s != '\0') r.sg = static_cast<unsigned>(std::stoul(s));
        if (const char* s = std::getenv("STRIX_GLM_TIERB_PRIME"); s != nullptr && *s != '\0') r.prime = std::stoi(s) != 0 ? 1 : 0;
        if (const char* s = std::getenv("STRIX_GLM_TIERB_TPW"); s != nullptr && *s != '\0') r.tpw = static_cast<unsigned>(std::stoul(s));
        return r;
    }();
    return k;
}

void kda_qkv(const Mat (&m)[3], const float* x, unsigned x_stride, unsigned rows, float* y, unsigned y_stride, void* ws, hipStream_t s,
             Knobs k) {
    if (!supported(Group::KdaQkv, m, 3)) throw std::invalid_argument("glm attn tierb: kda_qkv shapes");
    check_rows(rows, x, x_stride, GQkv::In, ws);
    check_out(y, y_stride, 3 * GQkv::Out, rows);
    Args a{};
    a.m0 = margs(m[0], x, x_stride, y, y_stride, ws, 0);
    a.m1 = margs(m[1], x, x_stride, y + GQkv::Out, y_stride, ws, 1);
    a.m2 = margs(m[2], x, x_stride, y + 2 * GQkv::Out, y_stride, ws, 2);
    a.rows = rows;
    launch<GQkv, 3, GQkv, 0>(a, resolve(k, Group::KdaQkv, rows), s);
}
void kda_o(const Mat& m, const float* x, unsigned x_stride, unsigned rows, float* y, unsigned y_stride, void* ws, hipStream_t s, Knobs k,
           const strix::pair::ProducerPublish* pub) {
    if (!supported(Group::KdaO, &m, 1)) throw std::invalid_argument("glm attn tierb: kda_o shape");
    check_rows(rows, x, x_stride, GQkv::In, ws);
    check_out(y, y_stride, GQkv::Out, rows);
    Args a{};
    a.m0 = margs(m, x, x_stride, y, y_stride, ws, 0);
    a.m1 = a.m2 = a.m0;
    a.rows = rows;
    if (pub != nullptr && pub->on()) {
        a.pub = *pub;
        a.pub_blocks = GQkv::Out / 128U;
    }
    launch<GQkv, 1, GQkv, 0>(a, resolve(k, Group::KdaO, rows), s);
}
void dsa_a(const Mat& q_a, const Mat& kv_a, const float* x, unsigned x_stride, unsigned rows, float* qa, unsigned qa_stride, float* kva,
           unsigned kva_stride, void* ws, hipStream_t s, Knobs k) {
    const Mat m[2] = {q_a, kv_a};
    if (!supported(Group::DsaA, m, 2)) throw std::invalid_argument("glm attn tierb: dsa_a shapes");
    check_rows(rows, x, x_stride, GQa::In, ws);
    check_out(qa, qa_stride, GQa::Out, rows);
    check_out(kva, kva_stride, GKva::Out, rows);
    Args a{};
    a.m0 = margs(q_a, x, x_stride, qa, qa_stride, ws, 0);
    a.m1 = margs(kv_a, x, x_stride, kva, kva_stride, ws, 1);
    a.m2 = a.m1;
    a.rows = rows;
    launch<GQa, 1, GKva, 1>(a, resolve(k, Group::DsaA, rows), s);
}
void dsa_b(const Mat& q_b, const Mat& wq_b, const float* qr, unsigned qr_stride, unsigned rows, float* q, unsigned q_stride, float* qI,
           unsigned qI_stride, void* ws, hipStream_t s, Knobs k) {
    const Mat m[2] = {q_b, wq_b};
    if (!supported(Group::DsaB, m, 2)) throw std::invalid_argument("glm attn tierb: dsa_b shapes");
    check_rows(rows, qr, qr_stride, GQb::In, ws);
    check_out(q, q_stride, GQb::Out, rows);
    check_out(qI, qI_stride, GWqb::Out, rows);
    Args a{};
    a.m0 = margs(q_b, qr, qr_stride, q, q_stride, ws, 0);
    a.m1 = margs(wq_b, qr, qr_stride, qI, qI_stride, ws, 1);
    a.m2 = a.m1;
    a.rows = rows;
    launch<GQb, 1, GWqb, 1>(a, resolve(k, Group::DsaB, rows), s);
}
void dsa_o(const Mat& m, const float* x, unsigned x_stride, unsigned rows, float* y, unsigned y_stride, void* ws, hipStream_t s, Knobs k,
           const strix::pair::ProducerPublish* pub) {
    if (!supported(Group::DsaO, &m, 1)) throw std::invalid_argument("glm attn tierb: dsa_o shape");
    check_rows(rows, x, x_stride, GDo::In, ws);
    check_out(y, y_stride, GDo::Out, rows);
    Args a{};
    a.m0 = margs(m, x, x_stride, y, y_stride, ws, 0);
    a.m1 = a.m2 = a.m0;
    a.rows = rows;
    if (pub != nullptr && pub->on()) {
        a.pub = *pub;
        a.pub_blocks = GDo::Out / 128U;
    }
    launch<GDo, 1, GDo, 0>(a, resolve(k, Group::DsaO, rows), s);
}

void set_table_enabled(bool on) { g_table.store(on ? 1 : 0); }

}  // namespace strix::glm::attn_tierb
