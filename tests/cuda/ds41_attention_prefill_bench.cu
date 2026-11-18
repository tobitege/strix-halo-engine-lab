// Timing of the deepseek41 attention ops at prefill shapes (not a test): a prompt of T rows from position 0 in
// windows of 64 rows, synthetic caches of the right sizes, per layer kind: index scores, selection and attention
// (device ms summed over the windows). Kinds: win (window only), r2 (ratio-2 index source, plain pool), r1src (layer
// 20: ratio 1, candidate source), r1user (ratio-1 candidate user). Users of a source's selection cost its attention.
//   strix-cuda-ds41-attention-prefill-bench [T=8192] [reps=3]   (per window and phase the minimum over the repetitions)
#include "strix/cuda/ds41_attention.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace a = strix::ds41::attn;
namespace ca = strix::ds41::cuda::attn;
using strix::ds41::cuda::DeviceBuffer;

namespace {

std::mt19937 rng(77);
template <class T> DeviceBuffer<T> dev(const std::vector<T>& h) {
    DeviceBuffer<T> d(std::max<std::size_t>(h.size(), 1));
    d.upload(h.data(), h.size());
    return d;
}
std::vector<float> normal(std::size_t n, float sd) {
    std::normal_distribution<float> d(0.0F, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}
std::vector<std::uint8_t> bytes(std::size_t n, std::size_t entry, std::size_t codes, int slo, int shi, bool no_nan) {
    std::vector<std::uint8_t> e(n * entry);
    std::uniform_int_distribution<int> byte(0, 255), sc(slo, shi);
    for (std::size_t t = 0; t < n; ++t) {
        for (std::size_t i = 0; i < codes; ++i) { int b = byte(rng); if (no_nan && (b & 0x7F) == 0x7F) b ^= 1; e[t * entry + i] = static_cast<std::uint8_t>(b); }
        for (std::size_t i = codes; i < entry; ++i) e[t * entry + i] = static_cast<std::uint8_t>(sc(rng));
    }
    return e;
}

struct Ev {
    std::vector<cudaEvent_t> e;
    std::size_t used = 0;
    cudaEvent_t next() {
        if (used == e.size()) { cudaEvent_t x; STRIX_CUDA_CHECK(cudaEventCreate(&x)); e.push_back(x); }
        return e[used++];
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const unsigned T = argc > 1 ? static_cast<unsigned>(std::atoi(argv[1])) : 8192U;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 3;
    const unsigned W = 64, cap = a::kWindow + T;
    try {
        strix::ds41::cuda::select_device(0);
        ca::RopeTables rope(T + 1);
        const auto ring = dev(bytes(cap, a::kWindowEntryBytes, 512, 122, 130, true));
        const auto kv = dev(bytes(T + 1, a::kKvEntryBytes, 256, 0x20, 0x50, false));
        const auto idxc = dev(bytes(T + 1, a::kIdxEntryBytes, 64, 118, 128, false));
        std::vector<std::uint64_t> pos(T);
        for (unsigned i = 0; i < T; ++i) pos[i] = i;
        const auto dpos = dev(pos);
        const auto q = dev(normal(std::size_t{W} * 64 * 512, 1.0F));
        const auto iq = dev(normal(std::size_t{W} * 32 * 128, 1.0F));
        const auto iw = dev(normal(std::size_t{W} * 32, 1.0F));
        const auto sink = dev(normal(64, 1.0F));
        const unsigned stride = (T + a::kCandBlockSize - 1) / a::kCandBlockSize + 1;
        DeviceBuffer<float> scores(std::size_t{W} * (T + 1)), partial(std::size_t{W} * 64 * ca::kPartialFloats), o(std::size_t{W} * 64 * 512);
        DeviceBuffer<std::uint8_t> qprep(std::size_t{W} * ca::kIdxPrepBytes), cand(std::size_t{T} * stride);
        DeviceBuffer<std::uint32_t> bkeys(std::size_t{W} * stride), idx(std::size_t{T} * a::kTopK), cnt(T);
        cand.zero();
        DeviceBuffer<std::uint32_t> surv(std::max<std::size_t>(1, ca::select_survivor_words(W, T + 1, a::kTopK)));

        struct Kind { const char* name; unsigned ratio; a::Pool pool; bool index; };
        const Kind kinds[] = {{"win", 0, a::Pool::plain, false}, {"r2", 2, a::Pool::plain, true},
                              {"r1src", 1, a::Pool::candidate_source, true}, {"r1user", 1, a::Pool::candidate_user, true}};
        // per window and phase the minimum over the repetitions (robust to other work on a shared device)
        const unsigned nwin = (T + W - 1) / W;
        for (const auto& k : kinds) {
            std::vector<std::array<double, 3>> best;
            for (int pass = 0; pass <= reps; ++pass)  // pass 0: warm-up (a few windows)
            {
                Ev ev;
                std::vector<std::array<cudaEvent_t, 4>> marks;
                for (unsigned w0 = 0; w0 < (pass == 0 ? std::min(T, 8 * W) : T); w0 += W) {
                    const unsigned n = std::min(W, T - w0);
                    std::array<cudaEvent_t, 4> m{};
                    m[0] = ev.next();
                    STRIX_CUDA_CHECK(cudaEventRecord(m[0], nullptr));
                    const unsigned max_count = k.ratio ? (w0 + n) / k.ratio : 0U;
                    ca::SelectView sv{};
                    if (k.index && max_count > 0) {
                        ca::IndexView iv{n, k.ratio, max_count, iq.get(), iw.get(), rope.table(a::RopeKind::compressed), dpos.get() + w0, idxc.get(),
                                         scores.get(), qprep.get()};
                        sv.rows = n; sv.ratio = k.ratio; sv.max_count = max_count; sv.topk = a::kTopK; sv.cand_blocks = a::kCandBlocks;
                        sv.block_size = a::kCandBlockSize; sv.pos = dpos.get() + w0; sv.scores = scores.get();
                        sv.candidates = cand.get() + std::size_t{w0} * stride; sv.cand_stride = stride; sv.block_keys = bkeys.get();
                        sv.idx = idx.get() + std::size_t{w0} * a::kTopK; sv.idx_count = cnt.get() + w0;
                        sv.survivors = surv.get();
                        if (k.pool == a::Pool::candidate_user) {  // the source's candidates of these rows (untimed)
                            sv.pool = a::Pool::candidate_source;
                            ca::enqueue_index_scores(iv, nullptr);
                            ca::enqueue_select(sv, nullptr);
                        }
                        sv.pool = k.pool;
                        m[0] = ev.next();
                        STRIX_CUDA_CHECK(cudaEventRecord(m[0], nullptr));
                        ca::enqueue_index_scores(iv, nullptr);
                        m[1] = ev.next();
                        STRIX_CUDA_CHECK(cudaEventRecord(m[1], nullptr));
                        ca::enqueue_select(sv, nullptr);
                    } else {
                        m[1] = m[0];
                    }
                    m[2] = ev.next();
                    STRIX_CUDA_CHECK(cudaEventRecord(m[2], nullptr));
                    ca::AttentionView av{};
                    av.rows = n; av.capacity = cap; av.topk = a::kTopK; av.q = q.get(); av.sink = sink.get();
                    av.rope = rope.table(k.ratio ? a::RopeKind::compressed : a::RopeKind::window); av.pos = dpos.get() + w0; av.ring = ring.get();
                    if (k.ratio != 0U) {
                        av.kv_cache = kv.get();
                        av.idx = idx.get() + std::size_t{w0} * a::kTopK;
                        av.idx_count = cnt.get() + w0;
                        if (!k.index || max_count == 0) STRIX_CUDA_CHECK(cudaMemsetAsync(cnt.get() + w0, 0, n * 4U, nullptr));
                    }
                    av.partial = partial.get(); av.o = o.get(); av.heads = 64;
                    ca::enqueue_attention(av, nullptr);
                    m[3] = ev.next();
                    STRIX_CUDA_CHECK(cudaEventRecord(m[3], nullptr));
                    marks.push_back(m);
                }
                STRIX_CUDA_CHECK(cudaDeviceSynchronize());
                if (pass > 0) {
                    auto& b = best;
                    if (b.empty()) b.assign(nwin, std::array<double, 3>{1e30, 1e30, 1e30});
                    for (std::size_t i = 0; i < marks.size(); ++i) {
                        const auto& m = marks[i];
                        float x = 0;
                        if (m[1] != m[0]) {
                            STRIX_CUDA_CHECK(cudaEventElapsedTime(&x, m[0], m[1])); b[i][0] = std::min<double>(b[i][0], x);
                            STRIX_CUDA_CHECK(cudaEventElapsedTime(&x, m[1], m[2])); b[i][1] = std::min<double>(b[i][1], x);
                        } else {
                            b[i][0] = b[i][1] = 0;
                        }
                        STRIX_CUDA_CHECK(cudaEventElapsedTime(&x, m[2], m[3])); b[i][2] = std::min<double>(b[i][2], x);
                    }
                }
                for (auto e : ev.e) cudaEventDestroy(e);
            }
            double t[3] = {0, 0, 0};
            for (const auto& x : best) for (int j = 0; j < 3; ++j) t[j] += x[j];
            std::printf("T %5u %-6s scores %8.2f ms  select %8.2f ms  attention %8.2f ms  total %8.2f ms\n", T, k.name, t[0], t[1], t[2],
                        t[0] + t[1] + t[2]);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
