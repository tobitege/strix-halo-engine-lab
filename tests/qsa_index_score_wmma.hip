// Standalone index scorer/selector diagnostic; no model or full engine linkage.
#include "qsa_flash_prod.inc"
#include "qsa_index_score_fixture.hpp"
#include <array>
#include <iostream>
#include <iomanip>
#include "row_fixture_stride.hpp"
namespace strix { namespace {
#include "../src/hip_qsa_index_score_wmma.inc"
#include "../src/hip_qsa_stage_score.inc"
} }

namespace {
using namespace strix;
using namespace index_fixture;
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
template <class T> struct Device {
    T* p = nullptr;
    std::size_t n;
    explicit Device(std::size_t size) : n(size) { check(hipMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    ~Device() { if (p) (void)hipFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void put(const std::vector<T>& h) { require(h.size() == n, "upload extent"); check(hipMemcpy(p, h.data(), n * sizeof(T), hipMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> h(n); check(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
};
constexpr unsigned T = 16384U, Guard = 16U;
constexpr std::uint32_t Poison = 0x4e123456U;
template <unsigned Rows> void tails() {
    constexpr unsigned blocks = 67U, stride = blocks + Guard;
    Device<float> keys(blocks * dim), query(Rows * heads * dim), maxima(Rows * 32U);
    Device<std::uint16_t> packed(keys.n);
    Device<uint2> ranges(blocks);
    std::vector<float> hk(keys.n), hq(query.n);
    std::vector<std::uint16_t> hp(keys.n);
    for (unsigned i = 0U; i < hk.size(); ++i) { hk[i] = sample(i + 7U); hp[i] = static_cast<std::uint16_t>(std::bit_cast<unsigned>(hk[i]) >> 16U); }
    for (unsigned i = 0U; i < hq.size(); ++i) hq[i] = sample(i + 977U);
    keys.put(hk); packed.put(hp); query.put(hq);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::key_range_kernel<>), dim3(1), dim3(256), 0, nullptr, keys.p, 0U, blocks, ranges.p);
    check(hipGetLastError());
    std::vector<float> reference;
    for (unsigned arm = 0U; arm < 3U; ++arm) {
        Device<float> out(Rows * stride);
        out.put(std::vector<float>(out.n, std::bit_cast<float>(Poison)));
        QsaRowsScoreViews<Rows> v{};
        for (unsigned r = 0U; r < Rows; ++r) {
            v.index_query[r] = query.p + r * heads * dim;
            v.block_scores[r] = out.p + r * stride;
            v.blocks[r] = r ? 51U + r % 17U : 0U;
        }
        if (!arm)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::score_kernel<Rows, 16U, 8U>), dim3(1, Rows / 16U),
                dim3(256), 0, nullptr, v, keys.p, blocks, dim, maxima.p, 32U, ranges.p);
        else {
            v.index_query[0] = nullptr; // skipped query must not be consumed
            qsa_flash::launch_index_score_wmma<Rows, false>(v, arm == 2U ? static_cast<const void*>(packed.p) : keys.p,
                arm == 2U, blocks, maxima.p, 32U, nullptr, 1U, nullptr);
        }
        check(hipGetLastError()); check(hipDeviceSynchronize());
        const auto got = out.get();
        if (!arm) reference = got;
        else for (unsigned r = 0U; r < Rows; ++r) for (unsigned b = 0U; b < stride; ++b) {
            const auto at = r * stride + b;
            if (b >= v.blocks[r]) require(std::bit_cast<unsigned>(got[at]) == Poison, "small-tail guard");
            else require(std::fabs(got[at] - reference[at]) < 1e-3F, "single-group score mismatch");
        }
    }
    std::cout << "INDEX_TAIL_PASS rows=" << Rows << " blocks=67 packed_and_fp32_keys=1 skipped_query=1\n";
}
void run(unsigned position, bool timing, bool packed) {
    const unsigned blocks = (position + T) / ratio, tight_stride = blocks + Guard;
    const unsigned stride = static_cast<unsigned>(fixture_row_stride_bytes(tight_stride * 4ULL) / 4U);
    std::cout << "INDEX_ROWS P=" << position << " T=" << T << " tight_bytes=" << tight_stride * 4U << " stride_bytes=" << stride * 4U << '\n';
    const unsigned max_stride = (blocks + 3U) / 4U + Guard, selected_stride = budget + 3U + Guard;
    const std::size_t elements = static_cast<std::size_t>(T) * stride;
    Device<float> keys(static_cast<std::size_t>(blocks) * dim), query(T * heads * dim);
    Device<std::uint16_t> packed_keys(keys.n);
    std::vector<float> hk(keys.n), hq(query.n);
    for (unsigned i = 0U; i < hk.size(); ++i) hk[i] = sample(i + 889U);
    // Repeated keys exercise score ties. Zero query rows exercise dense zero ties.
    for (unsigned b = 1U; b < blocks; b += 128U) std::copy_n(hk.data(), dim, hk.data() + b * dim);
    for (unsigned i = 0U; i < hq.size(); ++i) hq[i] = sample(i + 1234567U + position);
    for (unsigned r = 0U; r < T; r += 997U) std::fill_n(hq.data() + r * heads * dim, heads * dim, 0.0F);
    keys.put(hk); query.put(hq);
    std::vector<std::uint16_t> hp(keys.n);
    for (unsigned i = 0U; i < hp.size(); ++i) hp[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(hk[i]) >> 16U);
    packed_keys.put(hp);
    Device<uint2> ranges(blocks);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::key_range_kernel<>), dim3((blocks + 255U) / 256U),
        dim3(256), 0, nullptr, keys.p, 0U, blocks, ranges.p);
    check(hipGetLastError());
    std::array<Device<float>, 2> scores{Device<float>(elements), Device<float>(elements)};
    std::array<Device<float>, 2> maxima{Device<float>(static_cast<std::size_t>(T) * max_stride), Device<float>(static_cast<std::size_t>(T) * max_stride)};
    std::array<Device<unsigned>, 2> selected{Device<unsigned>(static_cast<std::size_t>(T) * selected_stride), Device<unsigned>(static_cast<std::size_t>(T) * selected_stride)};
    using Group = qsa_flash::ScoreGroup<64U>;
    std::array<Device<Group>, 2> descriptors{Device<Group>(T / 64U), Device<Group>(T / 64U)};
    std::array<Device<qsa_flash::SelectRow>, 2> selections{Device<qsa_flash::SelectRow>(T), Device<qsa_flash::SelectRow>(T)};
    for (unsigned arm = 0U; arm < 2U; ++arm) {
        std::vector<Group> groups(T / 64U);
        std::vector<qsa_flash::SelectRow> sr(T);
        std::vector<float> poison(elements, std::bit_cast<float>(Poison)); scores[arm].put(poison);
        poison.assign(maxima[arm].n, std::bit_cast<float>(Poison)); maxima[arm].put(poison);
        check(hipMemset(selected[arm].p, 0xa5, selected[arm].n * sizeof(unsigned)));
        for (unsigned r = 0U; r < T; ++r) {
            auto& g = groups[r / 64U]; const auto i = r % 64U;
            const unsigned complete = (position + r + 1U) / ratio;
            // The exact range prologue reads every row in an active group.
            // Only the candidate may use null for skipped dense queries.
            g.views.index_query[i] = query_for_arm(arm, complete, query.p + r * heads * dim);
            g.views.block_scores[i] = scores[arm].p + static_cast<std::size_t>(r) * stride;
            g.views.blocks[i] = complete > topk ? complete : 0U;
            g.blocks = std::max(g.blocks, g.views.blocks[i]);
            g.maxima = maxima[arm].p + static_cast<std::size_t>(r / 64U * 64U) * max_stride;
            g.maxima_stride = max_stride;
            sr[r] = {g.views.block_scores[i], selected[arm].p + static_cast<std::size_t>(r) * selected_stride,
                maxima[arm].p + static_cast<std::size_t>(r) * max_stride, position + r + 1U, 0U};
        }
        descriptors[arm].put(groups); selections[arm].put(sr);
    }
    const void* operand = packed ? static_cast<const void*>(packed_keys.p) : keys.p;
    const auto score = [&](unsigned arm) {
        if (arm == 0U) qpf2k_stage_score_launch(descriptors[arm].p, operand, packed, ranges.p, dim, blocks, 0U, T / 64U, nullptr);
        else qsa_flash::launch_index_score_wmma<64U, true>(QsaRowsScoreViews<64U>{}, operand, packed, blocks,
            nullptr, 0U, descriptors[arm].p, T / 64U, nullptr);
        check(hipGetLastError());
    };
    const auto select = [&](unsigned arm) {
        hipLaunchKernelGGL(qsa_flash::select_batch_kernel, dim3(T), dim3(kSelectThreads), 0, nullptr,
            selections[arm].p, ratio, topk);
        check(hipGetLastError());
    };
    score(0U); score(1U); check(hipDeviceSynchronize());
    auto a = scores[0].get(), b = scores[1].get();
    score(1U); score(1U); check(hipDeviceSynchronize());
    require(scores[1].get() == b, "candidate score nondeterminism");
    SelectionChanges changes;
    float maximum = 0.0F, scale = 0.0F, oracle_error = 0.0F;
    std::array<std::vector<unsigned>, 2> want;
    for (auto& w : want) w.resize(static_cast<std::size_t>(T) * topk);
    for (unsigned r = 0U; r < T; ++r) {
        const unsigned complete = (position + r + 1U) / ratio, n = complete > topk ? complete : 0U;
        const auto at = static_cast<std::size_t>(r) * stride;
        for (unsigned j = 0U; j < stride; ++j) {
            if (j >= n) require(std::bit_cast<unsigned>(a[at + j]) == Poison && std::bit_cast<unsigned>(b[at + j]) == Poison, "score canary overwritten");
            else {
                require(std::isfinite(a[at + j]) && std::isfinite(b[at + j]), "nonfinite score");
                maximum = std::max(maximum, std::fabs(a[at + j] - b[at + j]));
                scale = std::max(scale, std::fabs(a[at + j]));
            }
        }
        // Independent sampled CPU trees/panels at both column ends and across K16.
        if (n && r % 257U == 0U) for (unsigned j : {0U, n / 2U, n - 1U}) {
            const auto* q = hq.data() + r * heads * dim; const auto* k = hk.data() + j * dim;
            require(std::fabs(fp32_score(q, k) - a[at + j]) < 1e-4F, "CPU FP32 oracle mismatch");
            oracle_error = std::max(oracle_error, std::fabs(panel_score(q, k) - b[at + j]));
        }
        auto x = selected_blocks(a.data() + at, complete), y = selected_blocks(b.data() + at, complete);
        changes.add(x, y);
        for (unsigned arm = 0U; arm < 2U; ++arm) {
            const auto& ids = arm ? y : x;
            std::copy(ids.begin(), ids.end(), want[arm].begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(r) * topk));
        }
    }
    require(maximum < 1e-3F * (1.0F + scale) && oracle_error < 1e-3F, "WMMA arithmetic/mapping sanity bound");
    // Validate every maxima value and untouched tail before the selector consumes it.
    for (unsigned arm = 0U; arm < 2U; ++arm) {
        const auto hm = maxima[arm].get(); const auto& raw = arm ? b : a;
        for (unsigned r = 0U; r < T; ++r) {
            const unsigned complete = (position + r + 1U) / ratio, n = complete > topk ? complete : 0U;
            for (unsigned g = 0U; g < max_stride; ++g) {
                const auto value = hm[static_cast<std::size_t>(r) * max_stride + g];
                if (g * 4U >= n) require(std::bit_cast<unsigned>(value) == Poison, "maxima canary overwritten");
                else {
                    float m = -INFINITY;
                    for (unsigned j = g * 4U; j < std::min(n, g * 4U + 4U); ++j) m = std::fmax(m, raw[static_cast<std::size_t>(r) * stride + j]);
                    require(value == m, "maxima mismatch");
                }
            }
        }
        select(arm); check(hipDeviceSynchronize());
        const auto got = selected[arm].get();
        for (unsigned r = 0U; r < T; ++r) {
            const unsigned visible = position + r + 1U, complete = visible / ratio;
            const unsigned count = std::min(complete, topk) * ratio + visible % ratio;
            for (unsigned j = 0U; j < selected_stride; ++j) {
                unsigned expected = 0xa5a5a5a5U;
                if (j < std::min(complete, topk) * ratio)
                    expected = want[arm][static_cast<std::size_t>(r) * topk + j / ratio] * ratio + j % ratio;
                else if (j < count) expected = complete * ratio + j - std::min(complete, topk) * ratio;
                require(got[static_cast<std::size_t>(r) * selected_stride + j] == expected, "selector set/tie/tail/canary mismatch");
            }
        }
    }
    std::cout << "INDEX_SCORE T=" << T << " P=" << position << " budget=" << budget << " packed=" << packed
        << " fp32_rows_per_cta=" << qsa_flash::score_rows(blocks)
        << " max_abs_score_diff=" << maximum << " max_panel_oracle_diff=" << oracle_error
        << " rows=" << changes.rows << " rows_with_selection_change=" << changes.changed
        << " mean_symmetric_difference_blocks=" << changes.mean() << " max_symmetric_difference_blocks=" << changes.maximum
        << " mean_symmetric_difference_tokens=" << ratio * changes.mean()
        << " max_symmetric_difference_tokens=" << ratio * changes.maximum << '\n';
    if (!timing) return;
    hipEvent_t begin{}, end{}; check(hipEventCreate(&begin)); check(hipEventCreate(&end));
    for (bool with_select : {false, true}) {
        // Two warmups of each arm, then four complete ABBA quartets.
        for (unsigned i = 0U; i < 2U; ++i) for (unsigned arm = 0U; arm < 2U; ++arm) { score(arm); if (with_select) select(arm); }
        check(hipDeviceSynchronize());
        std::array<std::vector<float>, 2> ms;
        for (unsigned pair = 0U; pair < 8U; ++pair) for (unsigned j = 0U; j < 2U; ++j) {
            const unsigned arm = (pair + j) % 2U;
            check(hipEventRecord(begin)); score(arm); if (with_select) select(arm);
            check(hipEventRecord(end)); check(hipEventSynchronize(end));
            float value = 0.0F; check(hipEventElapsedTime(&value, begin, end)); ms[arm].push_back(value);
            std::cout << "INDEX_PAIR P=" << position << " packed=" << packed << " select=" << with_select
                << " pair=" << pair << " order=" << j << " arm=" << arm << " ms=" << value << '\n';
        }
        std::cout << "INDEX_TIME P=" << position << " packed=" << packed << " select=" << with_select
            << " warmups=2 pairs=8 fp32_ms=" << median(ms[0]) << " wmma_ms=" << median(ms[1])
            << " saving_ms=" << median(ms[0]) - median(ms[1]) << '\n';
    }
    check(hipEventDestroy(begin)); check(hipEventDestroy(end));
}
} // namespace
int main(int argc, char** argv) try {
    bool timing = false, packed = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--time") timing = true;
        else if (arg == "--packed") packed = true;
        else throw std::invalid_argument("usage: qsa-index-score-wmma [--time] [--packed]");
    }
    hipDeviceProp_t p{}; check(hipGetDeviceProperties(&p, 0));
    const std::string arch(p.gcnArchName);
    require((arch == "gfx1151" || arch.starts_with("gfx1151:")) && p.warpSize == 32, "requires gfx1151 wave32");
    std::cout << std::setprecision(9) << "synthetic score-only arithmetic screen; model qualification pending\n";
    tails<16U>(); tails<32U>();
    run(0U, timing, packed); run(16384U, timing, packed);
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
