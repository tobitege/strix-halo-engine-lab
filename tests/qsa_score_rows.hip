// F12 (Package C) fixture + microbench: the stage-batch scorer (score_kernel<64, R, 8, ..., Batch>) with R = 16
// (default), 32 and 64 query rows per CTA (STRIX_QPF2K_SCORE_ROWS). Standalone; build with the engine TU's flags:
// amdclang++ -O3 -std=gnu++20 --offload-arch=gfx1151 -Iinclude -x hip tests/qsa_score_rows.hip -o qsa_score_rows
//
// One layer-chunk of T rows (64-row groups, one batched launch over all groups, as hip_qsa_prefill_batch.inc
// launches it) at prompt position P. Exact gate first: for every P, R = 32 and R = 64 must reproduce the R = 16
// scores, group maxima and guard counters byte for byte, including canaries past every row's blocks and past
// every group's maxima. Then timing (no timing after a mismatch): A/A of R = 16 against itself, then rounds
// of 16, 32, 64, 64, 32, 16 (ABBA), hipEvent per launch, median / spread per arm.
// Data: BF16-valued pooled keys and index queries (pseudo-normal), with edge values outside the fused-form
// guard (tiny, huge, subnormal, inf, NaN, -0) in some query rows and some key blocks, all-zero query rows and
// all-zero key blocks, so both guard forms run.
// Packed arms (STRIX_QSA_POOL_PACKED): the same scorer reading the pool's packed BF16 copy (score_kernel<..., Packed>),
// R = 16 / 32 / 64, must equal the FP32-key R = 16 arm byte for byte too, and join the timing rounds.
// Each arm also runs with SplitLoads=true (S suffix), comparing scores, maxima,
// canaries and both guard counters to the original R16 form.
#include "qsa_flash_prod.inc"
using namespace strix;
static void check(hipError_t e, const char* what) { if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(2); } }
template <class T> static T* dalloc(std::size_t n) { void* p = nullptr; check(hipMalloc(&p, n * sizeof(T)), "hipMalloc"); return static_cast<T*>(p); }
static std::uint16_t host_bf16(float v) { std::uint32_t b; std::memcpy(&b, &v, 4); return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U); }
static float host_from_bf16(std::uint16_t h) { std::uint32_t b = static_cast<std::uint32_t>(h) << 16U; float v; std::memcpy(&v, &b, 4); return v; }
static float bf(float v) { return host_from_bf16(host_bf16(v)); }

int main(int argc, char** argv) {
    std::vector<std::uint64_t> positions{0, 24576, 122880, 253952};
    std::uint64_t T = 8192; unsigned rounds = 4, seed = 1; bool timing = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--pos" && i + 1 < argc) { positions.clear(); for (const char* p = argv[++i]; *p;) { char* e; positions.push_back(std::strtoull(p, &e, 10)); p = *e ? e + 1 : e; } }
        else if (a == "--rows" && i + 1 < argc) T = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--rounds" && i + 1 < argc) rounds = static_cast<unsigned>(std::atoi(argv[++i]));
        else if (a == "--seed" && i + 1 < argc) seed = static_cast<unsigned>(std::atoi(argv[++i]));
        else if (a == "--no-timing") timing = false;
    }
    if (T % 64U) { std::fprintf(stderr, "rows must be a multiple of 64\n"); return 2; }
    constexpr unsigned ratio = 4, topk = 512, dim = qsa_flash::kDim, heads = qsa_flash::kHeads, W = qsa_flash::kScoreWaves;
    const std::uint64_t groups = T / 64U;
    hipDeviceProp_t prop{}; check(hipGetDeviceProperties(&prop, 0), "props");
    std::printf("device %s %s CUs %d  T %llu groups %llu rounds %u seed %u\n", prop.name, prop.gcnArchName, prop.multiProcessorCount,
        (unsigned long long)T, (unsigned long long)groups, rounds, seed);
    struct Fast { std::uint64_t x; float operator()() { float t = 0.0F; for (int i = 0; i < 4; ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        t += static_cast<float>(x >> 40) * (1.0F / 16777216.0F); } return (t - 2.0F) * 1.7320508F; } };
    const float edge[] = {std::ldexp(1.0F, -110), std::ldexp(1.0F, 124), 1e-40F, std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(), -0.0F, std::ldexp(1.0F, -60), std::ldexp(1.5F, 63)};
    hipStream_t s; check(hipStreamCreate(&s), "stream");
    hipEvent_t e0, e1; check(hipEventCreate(&e0), "event"); check(hipEventCreate(&e1), "event");
    unsigned failures = 0;
    for (const auto P : positions) {
        Fast nd{0x9e3779b97f4a7c15ULL * (seed + 1U) ^ P};
        const std::uint64_t L = P + T, blocks_max = L / ratio;
        const std::uint64_t stride = (blocks_max + 8U + 63U) / 64U * 64U; // floats per row's score slot (canaries after)
        std::vector<float> keys(blocks_max * dim);
        for (auto& x : keys) x = bf(nd());
        // key edge blocks: every 997th block one edge value, every 4093rd block all zero
        for (std::uint64_t b = 0; b < blocks_max; b += 997U) keys[b * dim + (b * 13U) % dim] = bf(edge[(b / 997U) % 8U]);
        for (std::uint64_t b = 1; b < blocks_max; b += 4093U) std::fill_n(keys.begin() + static_cast<std::ptrdiff_t>(b * dim), dim, 0.0F);
        std::vector<float> q(T * heads * dim);
        for (auto& x : q) x = bf(nd());
        for (std::uint64_t g = 0; g < groups; ++g) {
            for (unsigned e = 0; e < 8U; ++e) q[(g * 64U + 5U + 7U * e) * heads * dim + (e * 37U) % (heads * dim)] = bf(edge[e]);
            if (g % 8U == 0U) std::fill_n(q.begin() + static_cast<std::ptrdiff_t>((g * 64U + 3U) * heads * dim), heads * dim, 0.0F);
        }
        auto* d_k = dalloc<float>(keys.size()); check(hipMemcpy(d_k, keys.data(), keys.size() * 4, hipMemcpyHostToDevice), "h2d keys");
        std::vector<std::uint16_t> packed(keys.size());
        for (std::size_t i = 0; i < keys.size(); ++i) { std::uint32_t b; std::memcpy(&b, &keys[i], 4); packed[i] = static_cast<std::uint16_t>(b >> 16U); }
        auto* d_p = dalloc<std::uint16_t>(packed.size()); check(hipMemcpy(d_p, packed.data(), packed.size() * 2, hipMemcpyHostToDevice), "h2d packed");
        auto* d_q = dalloc<float>(q.size()); check(hipMemcpy(d_q, q.data(), q.size() * 4, hipMemcpyHostToDevice), "h2d q");
        auto* d_r = dalloc<uint2>(blocks_max);
        if (blocks_max)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::key_range_kernel<>), dim3(static_cast<std::uint32_t>((blocks_max + 255U) / 256U)), dim3(256), 0, s,
                d_k, 0U, static_cast<std::uint32_t>(blocks_max), d_r);
        // Descriptors as hip_qsa_prefill_batch.inc builds them (blocks[r] = complete if > topk else 0).
        std::vector<qsa_flash::ScoreGroup<64U>> desc(groups);
        std::uint64_t maxima_count = 0, pairs = 0; std::uint32_t max_blocks = 0;
        for (std::uint64_t g = 0; g < groups; ++g) {
            auto& sc = desc[g]; sc.blocks = 0;
            for (unsigned r = 0; r < 64U; ++r) {
                const std::uint64_t row = g * 64U + r, complete = (P + row + 1U) / ratio;
                sc.views.index_query[r] = d_q + row * heads * dim;
                sc.views.blocks[r] = complete > topk ? static_cast<std::uint32_t>(complete) : 0U;
                sc.blocks = std::max(sc.blocks, sc.views.blocks[r]); pairs += sc.views.blocks[r];
            }
            sc.maxima_stride = (sc.blocks + qsa_flash::kGroup - 1U) / qsa_flash::kGroup + 3U; // +3 canary floats per row
            maxima_count += 64U * sc.maxima_stride; max_blocks = std::max(max_blocks, sc.blocks);
        }
        std::printf("P %llu L %llu blocks_max %llu max_blocks %u pairs %.4e\n", (unsigned long long)P, (unsigned long long)L,
            (unsigned long long)blocks_max, max_blocks, static_cast<double>(pairs));
        if (!max_blocks) { std::printf("  no scored rows\n"); check(hipFree(d_k), "free"); check(hipFree(d_p), "free"); check(hipFree(d_q), "free"); check(hipFree(d_r), "free"); continue; }
        struct Arm { float* scores; float* maxima; std::uint32_t* stats; qsa_flash::ScoreGroup<64U>* d; };
        const auto make = [&] {
            Arm a{dalloc<float>(T * stride), dalloc<float>(maxima_count + 1U), dalloc<std::uint32_t>(qsa_flash::kStats), nullptr};
            auto d = desc; std::uint64_t off = 0;
            for (std::uint64_t g = 0; g < groups; ++g) {
                for (unsigned r = 0; r < 64U; ++r) d[g].views.block_scores[r] = a.scores + (g * 64U + r) * stride;
                d[g].maxima = d[g].blocks ? a.maxima + off : nullptr; off += 64U * d[g].maxima_stride;
            }
            a.d = dalloc<qsa_flash::ScoreGroup<64U>>(groups);
            check(hipMemcpy(a.d, d.data(), groups * sizeof(d[0]), hipMemcpyHostToDevice), "h2d desc");
            return a;
        };
        constexpr unsigned kArms = 12U;
        const char* names[kArms] = {"R16", "R32", "R64", "R16P", "R32P", "R64P",
            "R16S", "R32S", "R64S", "R16PS", "R32PS", "R64PS"};
        Arm arm[kArms] = {make(), make(), make(), make(), make(), make(),
            make(), make(), make(), make(), make(), make()};
        const auto launch = [&]<unsigned R, bool Stats, bool Packed, bool SplitLoads>(const Arm& a) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::score_kernel<64U, R, W, Stats, false, true, 1U, true, Packed, SplitLoads>),
                dim3(((max_blocks + 31U) / 32U + W - 1U) / W, 64U / R, static_cast<std::uint32_t>(groups)), dim3(W * 32U), 0, s,
                QsaRowsScoreViews<64U>{}, Packed ? static_cast<const void*>(d_p) : static_cast<const void*>(d_k), 0U, dim, nullptr, 0U, d_r,
                Stats ? a.stats : nullptr, a.d);
        };
        const auto run = [&]<bool Stats>(unsigned which) {
            switch (which) {
            case 0: launch.template operator()<16U, Stats, false, false>(arm[0]); break;
            case 1: launch.template operator()<32U, Stats, false, false>(arm[1]); break;
            case 2: launch.template operator()<64U, Stats, false, false>(arm[2]); break;
            case 3: launch.template operator()<16U, Stats, true, false>(arm[3]); break;
            case 4: launch.template operator()<32U, Stats, true, false>(arm[4]); break;
            case 5: launch.template operator()<64U, Stats, true, false>(arm[5]); break;
            case 6: launch.template operator()<16U, Stats, false, true>(arm[6]); break;
            case 7: launch.template operator()<32U, Stats, false, true>(arm[7]); break;
            case 8: launch.template operator()<64U, Stats, false, true>(arm[8]); break;
            case 9: launch.template operator()<16U, Stats, true, true>(arm[9]); break;
            case 10: launch.template operator()<32U, Stats, true, true>(arm[10]); break;
            default: launch.template operator()<64U, Stats, true, true>(arm[11]); break;
            }
        };
        // exact gate
        for (auto& a : arm) {
            check(hipMemsetAsync(a.scores, 0xA5, T * stride * 4, s), "canary"); check(hipMemsetAsync(a.maxima, 0xA5, (maxima_count + 1U) * 4, s), "canary");
            check(hipMemsetAsync(a.stats, 0, qsa_flash::kStats * 4, s), "stats");
        }
        for (unsigned w = 0; w < kArms; ++w) run.template operator()<true>(w);
        check(hipGetLastError(), "launch"); check(hipStreamSynchronize(s), "sync");
        std::vector<std::uint32_t> base_s(T * stride), base_m(maxima_count + 1U), base_st(qsa_flash::kStats), x;
        check(hipMemcpy(base_s.data(), arm[0].scores, base_s.size() * 4, hipMemcpyDeviceToHost), "d2h");
        check(hipMemcpy(base_m.data(), arm[0].maxima, base_m.size() * 4, hipMemcpyDeviceToHost), "d2h");
        check(hipMemcpy(base_st.data(), arm[0].stats, base_st.size() * 4, hipMemcpyDeviceToHost), "d2h");
        std::uint64_t written = 0, nan_scores = 0;
        for (std::uint64_t row = 0; row < T; ++row)
            for (std::uint64_t b = 0; b < stride; ++b) {
                const auto v = base_s[row * stride + b];
                written += v != 0xA5A5A5A5U; nan_scores += (v & 0x7fffffffU) > 0x7f800000U && v != 0xA5A5A5A5U;
            }
        std::printf("  R16 written scores %llu (pairs %.0f) nan %llu  guard fused %u separate %u\n", (unsigned long long)written,
            static_cast<double>(pairs), (unsigned long long)nan_scores, base_st[qsa_flash::kFused], base_st[qsa_flash::kSeparate]);
        bool ok = written == pairs && base_st[qsa_flash::kFused] > 0U && base_st[qsa_flash::kSeparate] > 0U;
        if (!ok) std::printf("  FAIL baseline witness (written != pairs or a guard form never ran)\n");
        for (unsigned w = 1; w < kArms; ++w) {
            std::uint64_t ds = 0, dm = 0, dst = 0;
            x.resize(base_s.size()); check(hipMemcpy(x.data(), arm[w].scores, x.size() * 4, hipMemcpyDeviceToHost), "d2h");
            for (std::size_t i = 0; i < x.size(); ++i) ds += x[i] != base_s[i];
            x.resize(base_m.size()); check(hipMemcpy(x.data(), arm[w].maxima, x.size() * 4, hipMemcpyDeviceToHost), "d2h");
            for (std::size_t i = 0; i < x.size(); ++i) dm += x[i] != base_m[i];
            x.resize(base_st.size()); check(hipMemcpy(x.data(), arm[w].stats, x.size() * 4, hipMemcpyDeviceToHost), "d2h");
            for (std::size_t i = 0; i < x.size(); ++i) dst += x[i] != base_st[i];
            std::printf("  %s vs R16: score words differ %llu of %llu, maxima words differ %llu of %llu, guard counters differ %llu\n",
                names[w], (unsigned long long)ds, (unsigned long long)base_s.size(), (unsigned long long)dm,
                (unsigned long long)base_m.size(), (unsigned long long)dst);
            ok = ok && ds == 0 && dm == 0 && dst == 0;
        }
        std::printf("  EXACT %s\n", ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
        if (ok && timing) {
            const auto timed = [&](unsigned which) {
                check(hipEventRecord(e0, s), "rec"); run.template operator()<false>(which); check(hipEventRecord(e1, s), "rec");
                check(hipEventSynchronize(e1), "sync"); float ms = 0; check(hipEventElapsedTime(&ms, e0, e1), "elapsed"); return ms;
            };
            for (unsigned w = 0; w < kArms; ++w) { timed(w); timed(w); } // warm
            std::vector<float> aa0, aa1, t[kArms];
            for (unsigned r = 0; r < rounds; ++r) { aa0.push_back(timed(0)); aa1.push_back(timed(0)); }
            const unsigned order[2U * kArms] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
            for (unsigned r = 0; r < rounds; ++r) for (unsigned o : order) t[o].push_back(timed(o));
            check(hipGetLastError(), "timed launch");
            const auto med = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
            const auto lo = [](const std::vector<float>& v) { return *std::min_element(v.begin(), v.end()); };
            const auto hi = [](const std::vector<float>& v) { return *std::max_element(v.begin(), v.end()); };
            const float m0 = med(t[0]);
            std::printf("  A/A R16 med %.3f / %.3f ms (%.2f %%)\n", med(aa0), med(aa1), 100.0 * (med(aa1) - med(aa0)) / med(aa0));
            for (unsigned w = 0; w < kArms; ++w)
                std::printf("  %-4s med %8.3f ms  min %8.3f  max %8.3f  vs R16 %+6.2f %%  (n %zu)\n", names[w],
                    med(t[w]), lo(t[w]), hi(t[w]), 100.0 * (med(t[w]) - m0) / m0, t[w].size());
        }
        for (auto& a : arm) { check(hipFree(a.scores), "free"); check(hipFree(a.maxima), "free"); check(hipFree(a.stats), "free"); check(hipFree(a.d), "free"); }
        check(hipFree(d_k), "free"); check(hipFree(d_p), "free"); check(hipFree(d_q), "free"); check(hipFree(d_r), "free");
    }
    std::printf("%s\n", failures ? "RESULT FAIL" : "RESULT PASS");
    return failures ? 1 : 0;
}
