// Synthetic arithmetic/selection gate, not model qualification. Standalone:
// amdclang++ -O3 -std=gnu++20 --offload-arch=gfx1151 -Iinclude -Isrc -fno-fast-math
// -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee
// -x hip tests/qsa_index_band.hip -o qsa_index_band
#include "qsa_flash_prod.inc"
#include <bit>
#include <array>
namespace strix { namespace {
#include "../src/hip_qsa_stage_score.inc"
} }
using namespace strix;
namespace {
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
void require(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
template <class T> struct Device {
    T* p = nullptr;
    std::size_t n;
    explicit Device(std::size_t count) : n(count) { check(hipMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    ~Device() { if (p) (void)hipFree(p); }
    Device(const Device&) = delete;
    void put(const std::vector<T>& v) { require(v.size() == n, "upload size"); check(hipMemcpy(p, v.data(), n * sizeof(T), hipMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> v(n); check(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return v; }
};
float bf(float x) {
    auto u = std::bit_cast<unsigned>(x);
    u = (u + 0x7fffU + ((u >> 16U) & 1U)) & 0xffff0000U;
    return std::bit_cast<float>(u);
}
template <unsigned Rows, bool Batch>
void run(unsigned blocks, unsigned topk, bool packed, unsigned edge, bool preserve) {
    constexpr unsigned groups = Batch ? 2U : 1U, rows = groups * Rows, dim = 128U, heads = 4U;
    constexpr unsigned guard = 8U, poison = 0x4e123456U;
    const unsigned stride = blocks + guard, ms = (blocks + 3U) / 4U + guard, ss = topk * 4U + 3U + guard;
    std::uint64_t seed = 17U;
    const auto sample = [&] {
        seed ^= seed << 13U; seed ^= seed >> 7U; seed ^= seed << 17U;
        return bf(static_cast<float>(static_cast<int>(seed % 4097U) - 2048) * 0.0009765625F);
    };
    std::vector<float> hk(std::size_t{blocks} * dim), hq(std::size_t{rows} * heads * dim);
    for (auto& x : hk) x = edge == 1U ? 0.0F : sample();
    for (auto& x : hq) x = sample();
    // Repeated keys, zero queries and subnormal query elements exercise ties and
    // the separate tree. Row visibility and group score prefixes differ at tails.
    for (unsigned b = 1U; b < blocks; b += 127U) std::copy_n(hk.data(), dim, hk.data() + std::size_t{b} * dim);
    std::fill_n(hq.data() + heads * dim, heads * dim, 0.0F);
    hq[2U * heads * dim] = 0x1p-127F;
    if (edge == 2U) hq[3U * heads * dim] = 65536.0F;
    if (edge == 3U) hk[(std::size_t{blocks} - 1U) * dim] = INFINITY;
    Device<float> keys(hk.size()), query(hq.size()), norms(blocks), bound(1U);
    keys.put(hk); query.put(hq);
    std::vector<std::uint16_t> hp(hk.size());
    for (std::size_t i = 0; i < hk.size(); ++i) hp[i] = static_cast<std::uint16_t>(std::bit_cast<unsigned>(hk[i]) >> 16U);
    Device<std::uint16_t> pk(hp.size()); pk.put(hp);
    Device<uint2> ranges(blocks);
    // Incremental range/norm publication must produce the same complete bound.
    const unsigned first = blocks / 2U;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::key_range_kernel<>), dim3((first + 255U) / 256U), dim3(256), 0, nullptr,
        keys.p, 0U, first, ranges.p, nullptr, 0U, norms.p);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::key_range_kernel<>), dim3((blocks - first + 255U) / 256U), dim3(256), 0, nullptr,
        keys.p, first, blocks - first, ranges.p, nullptr, 0U, norms.p);
    hipLaunchKernelGGL(qsa_flash::key_norm_max_kernel, dim3(1), dim3(1024), 0, nullptr, static_cast<const void*>(keys.p), false, blocks, bound.p);
    using Group = qsa_flash::ScoreGroup<Rows>;
    std::array<Device<float>, 2> scores{Device<float>(std::size_t{rows} * stride), Device<float>(std::size_t{rows} * stride)};
    std::array<Device<float>, 2> maxima{Device<float>(std::size_t{rows} * ms), Device<float>(std::size_t{rows} * ms)};
    std::array<Device<unsigned>, 2> selected{Device<unsigned>(std::size_t{rows} * ss), Device<unsigned>(std::size_t{rows} * ss)};
    std::array<Device<Group>, 2> desc{Device<Group>(groups), Device<Group>(groups)};
    std::array<Device<qsa_flash::SelectRow>, 2> sr{Device<qsa_flash::SelectRow>(rows), Device<qsa_flash::SelectRow>(rows)};
    Device<qsa_flash::BandRow> bands(rows);
    Device<unsigned> stats(qsa_flash::kStats); stats.put(std::vector<unsigned>(qsa_flash::kStats, 0U));
    std::vector<qsa_flash::BandRow> hb(rows);
    std::array<std::vector<Group>, 2> hg;
    const void* operand = packed ? static_cast<const void*>(pk.p) : keys.p;
    for (unsigned arm = 0; arm < 2U; ++arm) {
        scores[arm].put(std::vector<float>(scores[arm].n, std::bit_cast<float>(poison)));
        maxima[arm].put(std::vector<float>(maxima[arm].n, std::bit_cast<float>(poison)));
        selected[arm].put(std::vector<unsigned>(selected[arm].n, poison));
        hg[arm].resize(groups);
        std::vector<qsa_flash::SelectRow> hs(rows);
        for (unsigned g = 0; g < groups; ++g) {
            auto& sc = hg[arm][g];
            sc.blocks = blocks - (g + 1U == groups ? 0U : 37U);
            sc.maxima = maxima[arm].p + std::size_t{g * Rows} * ms; sc.maxima_stride = ms;
            for (unsigned r = 0; r < Rows; ++r) {
                const unsigned row = g * Rows + r, complete = r ? sc.blocks - (Rows - 1U - r) % 19U : topk;
                sc.views.index_query[r] = query.p + std::size_t{row} * heads * dim;
                sc.views.block_scores[r] = scores[arm].p + std::size_t{row} * stride;
                sc.views.blocks[r] = complete > topk ? complete : 0U;
                hs[row] = {sc.views.block_scores[r], selected[arm].p + std::size_t{row} * ss,
                    sc.maxima + std::size_t{r} * ms, complete * 4U + row % 4U, preserve ? 1U : 0U};
                hb[row] = {sc.views.index_query[r], operand, ranges.p, bound.p, sc.blocks, dim, packed};
                if (arm && !sc.views.blocks[r]) sc.views.index_query[r] = nullptr;
            }
        }
        desc[arm].put(hg[arm]); sr[arm].put(hs);
    }
    bands.put(hb);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::score_kernel<Rows, 16U, 8U, false, false, true, 1U, true>),
        dim3(((blocks + 31U) / 32U + 7U) / 8U, Rows / 16U, groups), dim3(256), 0, nullptr,
        QsaRowsScoreViews<Rows>{}, keys.p, blocks, dim, nullptr, 0U, ranges.p, nullptr, desc[0].p);
    if constexpr (Batch) {
        // One descriptor per slice tests shared-core group addressing as well as
        // score tiles clipped to each descriptor's scored prefix.
        for (unsigned g = 0; g < groups; ++g)
            qpf2k_stage_score_launch(desc[1].p, operand, packed, ranges.p, dim, blocks, g, 1U, nullptr);
    } else {
        qsa_flash::launch_index_band<Rows, false>(hg[1][0].views, operand, packed, blocks, dim,
            maxima[1].p, ms, nullptr, 1U, nullptr);
    }
    check(hipGetLastError()); check(hipDeviceSynchronize());
    const auto exact = scores[0].get(), approximate = scores[1].get(), hm = maxima[1].get();
    const float kn = bound.get()[0];
    for (unsigned row = 0; row < rows; ++row) {
        float error = 0.0F;
        bool safe = std::isfinite(kn);
        for (unsigned h = 0; h < heads; ++h) {
            double q2 = 0.0;
            for (unsigned d = 0; d < dim; ++d) {
                const float x = hq[(std::size_t{row} * heads + h) * dim + d];
                safe = safe && std::fabs(x) < attn_core::idx::kF16Safe;
                q2 += double{x} * double{x};
            }
            error += attn_core::idx::f16_dot_error(static_cast<float>(std::sqrt(q2)), kn);
        }
        const float delta = 1.01F * error / std::sqrt(float{dim});
        const unsigned n = hg[1][row / Rows].views.blocks[row % Rows];
        if (safe) for (unsigned b = 0; b < n; ++b) {
            const auto at = std::size_t{row} * stride + b;
            require(std::fabs(approximate[at] - exact[at]) <= delta, "f16 score error exceeds row bound");
        }
        for (unsigned b = n; b < stride; ++b)
            require(std::bit_cast<unsigned>(approximate[std::size_t{row} * stride + b]) == poison, "score tail overwrite");
        for (unsigned g = 0; g < ms; ++g) {
            const float m = hm[std::size_t{row} * ms + g];
            if (g * 4U >= n) require(std::bit_cast<unsigned>(m) == poison, "maxima tail overwrite");
            else {
                float want = -INFINITY;
                for (unsigned b = g * 4U; b < std::min(n, g * 4U + 4U); ++b) want = std::fmax(want, approximate[std::size_t{row} * stride + b]);
                require(m == want, "WMMA group maximum mapping");
            }
        }
    }
    hipLaunchKernelGGL(qsa_flash::select_batch_kernel, dim3(rows), dim3(kSelectThreads), 0, nullptr, sr[0].p, 4U, topk);
    if constexpr (Batch) {
        hipLaunchKernelGGL(qsa_flash::select_band_batch_kernel, dim3(rows), dim3(kSelectThreads), 0, nullptr, sr[1].p, bands.p, 4U, topk);
    } else {
        QsaRowsSelectViews<Rows> v{}; v.preserve_scores = preserve;
        for (unsigned r = 0; r < Rows; ++r) {
            v.block_scores[r] = hg[1][0].views.block_scores[r];
            v.selected[r] = selected[1].p + std::size_t{r} * ss;
            const unsigned complete = r ? blocks - (Rows - 1U - r) % 19U : topk;
            v.visible[r] = complete * 4U + r % 4U;
        }
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::select_band_kernel<Rows, true>), dim3(Rows), dim3(kSelectThreads), 0, nullptr,
            v, hg[1][0].views, maxima[1].p, ms, 4U, topk, hb[0], stats.p);
    }
    check(hipGetLastError()); check(hipDeviceSynchronize());
    if constexpr (!Batch) if (blocks > 4U * topk && edge != 3U)
        require(stats.get()[qsa_flash::kBandSize] != 0U, "band refinement path was not exercised");
    const auto want = selected[0].get(), got = selected[1].get();
    const auto post = scores[1].get();
    require(want == got, "band selected tokens/ties/tails/canaries differ from exact");
    for (unsigned row = 0; row < rows; ++row) {
        const unsigned n = hg[1][row / Rows].views.blocks[row % Rows];
        for (unsigned j = 0; n && j < topk; ++j) {
            const unsigned b = got[std::size_t{row} * ss + j * 4U] / 4U;
            const auto at = std::size_t{row} * stride + b;
            require(preserve ? std::bit_cast<unsigned>(post[at]) == std::bit_cast<unsigned>(exact[at]) : post[at] == -INFINITY,
                "selected score preservation/-inf write differs");
        }
    }
    std::printf("PASS band Rows=%u batch=%u blocks=%u topk=%u packed=%u edge=%u preserve=%u\n",
        Rows, unsigned{Batch}, blocks, topk, unsigned{packed}, edge, unsigned{preserve});
}
} // namespace
int main() try {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || !count) return 77;
    if (!qsa_flash::index_band()) return 77;
    for (bool packed : {false, true}) for (bool preserve : {false, true}) {
        run<16U, false>(4099U, 384U, packed, 0U, preserve);
        run<32U, false>(4099U, 512U, packed, 0U, preserve);
        run<64U, true>(4099U, 384U, packed, 0U, preserve);
        run<64U, true>(8193U, 512U, packed, 1U, preserve); // capacity overflow
        run<16U, false>(1537U, 512U, packed, 0U, preserve); // fewer group maxima
        run<16U, false>(4099U, 384U, packed, 2U, preserve); // unsafe query
        run<64U, true>(4099U, 512U, packed, 3U, preserve); // unsafe key in final tile
    }
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
