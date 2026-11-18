// mm-rtx-prefill: MiniMax M3.1 Flash prefill on the RTX PRO in the FP32 prefill identity, both
// TP halves computed here exactly as the two nodes do (rank slices, rank partials, p0 + p1), layer-major over every
// row of every prompt. Writes each prompt's state rows [0, P-1) in the Engine::export_rows layout, one file per rank:
// OUT_DIR/state-NAME-rank<r>.bin. The node imports them (strix-minimax-rank --offload-import) and runs row P-1 itself.
//
// usage: mm-rtx-prefill MODEL_DIR ROPE.bin OUT_DIR NAME:PROMPT.u32 [NAME:PROMPT.u32 ...] [options]
//   (default)        gate mode: each layer read from MODEL_DIR when it is needed (low memory)
//   --resident N     timing mode: layers [0, N) in device memory, the rest in pinned host memory, each streamed into
//                    one of two device slots (copy stream, overlapped with the previous layer's compute)
//   --pinned H       timing mode, three tiers (spec-minimax 3.1): the --resident N device layers are the dense layers
//                    plus MoE layers spread evenly over the depth; of the other layers H are pinned host copies and the
//                    rest are read per prompt from --blob-cache DIR (one file per layer blob, written at load, named
//                    by the blob's hash) with unbuffered reads into a 4-slot pinned staging ring. After every prompt
//                    the run idles as long as that prompt's disk reads took (D: duty rule).
//   --repeat K       run the prompt set K times after loading (timing mode); state files from the last run
//   --layers N       first N layers only (profiling)
//   --send IP0:PORT0,IP1:PORT1  timing mode, one prompt: every layer's state rows go to rank r's
//                    strix-minimax-rank --offload-listen as soon as the layer is done (one sender thread per rank);
//                    rank 0 returns the last row's token, which ends the TTFT clock
//   --layer-cut K    prefix-resident block pipeline to --layercut-listen (one full prompt)
//   --cut-block C --cut-act relay|direct --cut-xport act-first|fifo --cut-kv-frag ROWS --cut-pair-identity ID; --split/--cut share env resolution
//   --print-effective-config   inspect split policy without CUDA work
//   --codec bp       hand-off frames bp-v1 encoded (lossless; the ranks decode before importing)
//   --replay-state DIR --send ...  no GPU: stream DIR/state-NAME-rank<r>.bin (a finished prefill) layer by layer to the
//                    listeners, --repeat K times per prompt alternating identity / bp-v1 frames (ABA...): transport,
//                    codec, import and token reply on the real link, plus its rate
//   MM_RTX_PROFILE=1 synchronised per-phase times
#include "strix/pair/state_decls.hpp"
#include "mm_rtx_kernels.cuh"
#include "mm_rtx_pack.hpp"
#include "mm_rtx_weights.hpp"
#include "minimax/rtxpf/fp32_identity.hpp"
#include "strix/bp_codec.hpp"
#include "strix/minimax_layercut.hpp"
#include "strix/handoff_lanes.hpp"

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace cu = strix::minimax::rtxpf::cuda;
namespace hx = strix::minimax::rtxpf::host;
namespace fp = strix::minimax::rtxpf;
using Clock = std::chrono::steady_clock;

namespace {

constexpr unsigned kHidden = 6144, kQ = 8192, kQLocal = 4096, kKV = 512, kKVLocal = 256, kIdxLocal = 256;
constexpr unsigned kDenseLocal = 6144, kExpertLocal = 1536, kLayers = 60, kExperts = 128, kTopE = 4;
const std::string kP = "language_model.model.";

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }
// MM_RTX_PROFILE=1: synchronised per-phase times (seconds, summed over layers).
struct Phases {
    bool on = false;
    Clock::time_point t;
    std::array<double, 10> s{};
    void mark() { if (on) { cudaDeviceSynchronize(); t = Clock::now(); } }
    void lap(int i) { if (on) { cudaDeviceSynchronize(); const auto n = Clock::now(); s[i] += secs(t, n); t = n; } }
};
const char* const kPhase[10] = {"qkv", "index_proj", "msa_prep", "msa_attend", "o_proj", "post+route", "experts", "shared+combine", "dense_mlp", "export"};
Phases g_ph;
template <class T> T* dalloc(std::size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "malloc");
    return static_cast<T*>(p);
}
template <class T> T* dupload(const std::vector<T>& h) {
    T* d = dalloc<T>(h.size());
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
std::vector<std::uint32_t> read_u32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const auto n = static_cast<std::size_t>(f.tellg());
    std::vector<std::uint32_t> v(n / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
    return v;
}

fp::F32Job job(const hx::DevMat& d, std::uint32_t rows) { return {d.trellis, d.suh, d.svh, d.input, d.output, d.bits, rows, 0, 0}; }

struct Runner {
    std::size_t R = 0;  // total rows
    float *h, *hn, *q, *k, *v, *iq, *ik, *attn, *part[2], *g, *u, *ye, *ys, *xt, *raw, *scratch, *scores;
    std::uint32_t *sel, *sel_count, *ids_d, *maps_d, *row_seq;
    uint2* tiles_d = nullptr;  // 32-row tiles of one sequence each (tiled index scores)
    unsigned n_tiles = 0;
    bool attn_legacy = std::getenv("MM_RTX_ATTN_LEGACY") != nullptr;
    bool no_tail = std::getenv("MM_RTX_NO_TAIL") != nullptr;  // A/B: 128-row tiles only (no 64-row tail core)
    // job tables: a pinned ring copied asynchronously (a blocking cudaMemcpy per call stalled the GPU ~20x per layer);
    // a region is reused only after the device drains at the wrap
    static constexpr std::uint32_t kJobRing = 8192;
    fp::F32Job* jobs_d;
    std::uint32_t* kv_map = nullptr;  // k/v grouped call: xmap [4R] then ymap [4R]
    int sms = [] { int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0); return n; }();
    fp::F32Job* jobs_h = nullptr;
    std::uint32_t ring_at = 0;
    std::vector<std::uint32_t> ids_h, maps_h;
    std::vector<float> wts_h;
    unsigned max_blocks = 1;

    void release() {
        for (void* p : {static_cast<void*>(h), static_cast<void*>(hn), static_cast<void*>(q), static_cast<void*>(k), static_cast<void*>(kv_map),
                        static_cast<void*>(iq), static_cast<void*>(ik), static_cast<void*>(attn), static_cast<void*>(part[0]),
                        static_cast<void*>(part[1]), static_cast<void*>(g), static_cast<void*>(u), static_cast<void*>(ye), static_cast<void*>(ys),
                        static_cast<void*>(xt), static_cast<void*>(raw), static_cast<void*>(scratch), static_cast<void*>(scores),
                        static_cast<void*>(sel), static_cast<void*>(sel_count), static_cast<void*>(ids_d), static_cast<void*>(maps_d),
                        static_cast<void*>(row_seq), static_cast<void*>(tiles_d), static_cast<void*>(jobs_d)})
            cudaFree(p);
        cudaFreeHost(jobs_h);
        jobs_h = nullptr;
    }
    void alloc(std::size_t rows, unsigned mb) {
        R = rows;
        max_blocks = mb;
        const std::size_t pairs = R * kTopE;
        h = dalloc<float>(R * kHidden);
        hn = dalloc<float>(R * kHidden);
        q = dalloc<float>(R * kQ);
        k = dalloc<float>(2 * R * kKV);  // k then v, contiguous: one grouped call writes both (kv_map)
        v = k + R * kKV;
        {  // the k/v call: jobs k r0, k r1, v r0, v r1 over R rows each; source row t, destination row of 256 floats
            std::vector<std::uint32_t> m(8 * R);
            for (std::size_t j = 0; j < 4; ++j)
                for (std::size_t t = 0; t < R; ++t) {
                    m[j * R + t] = static_cast<std::uint32_t>(t);
                    m[4 * R + j * R + t] = static_cast<std::uint32_t>((j / 2) * 2 * R + 2 * t + (j % 2));
                }
            kv_map = dupload(m);
        }
        iq = dalloc<float>(R * 512);
        ik = dalloc<float>(R * 128);
        attn = dalloc<float>(R * kQ);
        part[0] = dalloc<float>(R * kHidden);
        part[1] = dalloc<float>(R * kHidden);
        g = dalloc<float>(std::max(pairs * kExpertLocal, R * kDenseLocal));
        u = dalloc<float>(std::max(pairs * kExpertLocal, R * kDenseLocal));
        ye = dalloc<float>(pairs * kHidden);
        ys = dalloc<float>(R * kHidden);
        xt = dalloc<float>(pairs * kHidden);
        raw = dalloc<float>(pairs * kHidden);
        scratch = dalloc<float>(R * kExperts * 2);
        scores = dalloc<float>(R * 4 * mb);
        sel = dalloc<std::uint32_t>(R * 4 * 16);
        sel_count = dalloc<std::uint32_t>(R * 4);
        ids_d = dalloc<std::uint32_t>(R * kTopE * 2);  // ids then weights
        maps_d = dalloc<std::uint32_t>(3 * pairs);
        jobs_d = dalloc<fp::F32Job>(kJobRing);
        ck(cudaHostAlloc(reinterpret_cast<void**>(&jobs_h), kJobRing * sizeof(fp::F32Job), cudaHostAllocDefault), "job ring");
        ring_at = 0;
        ids_h.resize(R * kTopE);
        wts_h.resize(R * kTopE);
        maps_h.resize(3 * pairs);
    }
    void grouped(std::vector<fp::F32Job>& jobs, const float* x, std::uint32_t xs, const std::uint32_t* xmap, float* y, std::uint32_t ys_,
                 const std::uint32_t* ymap = nullptr) {
        std::uint64_t total = 0, tiles128 = 0;
        for (const auto& j : jobs) {
            total += j.rows;
            tiles128 += (j.rows + 127) / 128 * (j.output / 128);
        }
        std::uint32_t tr = fp::tile_rows_for(total, jobs.size());
        if (tr == 128 && tiles128 < static_cast<std::uint64_t>(sms)) tr = 64;  // a grid below one CTA per SM: twice the CTAs
        auto ext = fp::layout_jobs(jobs, tr);
        if (ext.entries == 0) return;
        const auto n = static_cast<std::uint32_t>(jobs.size());
        std::vector<fp::F32Job> tail;
        std::uint32_t tail_tiles = 0;
        if (tr == 128 && !no_tail) {  // remainders <= 64 rows to the 64-row tail core (same bits)
            tail_tiles = fp::split_tails(jobs, tail, ext.tiles);
            jobs.insert(jobs.end(), tail.begin(), tail.end());
        }
        const auto nj = static_cast<std::uint32_t>(jobs.size());
        if (ring_at + nj > kJobRing) {
            ck(cudaDeviceSynchronize(), "job ring wrap");
            ring_at = 0;
        }
        std::memcpy(jobs_h + ring_at, jobs.data(), nj * sizeof(fp::F32Job));
        ck(cudaMemcpyAsync(jobs_d + ring_at, jobs_h + ring_at, nj * sizeof(fp::F32Job), cudaMemcpyHostToDevice, nullptr), "jobs");
        fp::GroupedCall c;
        c.jobs_d = jobs_d + ring_at;
        ring_at += nj;
        c.jobs = n;
        c.tiles = ext.tiles;
        c.tile_rows = tr;
        c.tail_jobs = static_cast<std::uint32_t>(tail.size());
        c.tail_tiles = tail_tiles;
        c.input = jobs[0].input;
        c.output = jobs[0].output;
        c.entries = ext.entries;
        c.x = x;
        c.x_stride = xs;
        c.xmap = xmap;
        c.y = y;
        c.y_stride = ys_;
        c.ymap = ymap;
        fp::enqueue_grouped(c, xt, raw, nullptr);
    }
    void linear(const hx::DevMat& w, const float* x, std::uint32_t xs, float* y, std::uint32_t ys_) {
        std::vector<fp::F32Job> j{job(w, static_cast<std::uint32_t>(R))};
        grouped(j, x, xs, nullptr, y, ys_);
    }
    void mlp(const hx::DevMlp& m, unsigned inter, float* out) {  // on hn, all rows
        linear(m.gate, hn, kHidden, g, inter);
        linear(m.up, hn, kHidden, u, inter);
        fp::enqueue_swiglu(g, u, g, R * inter, nullptr);
        linear(m.down, g, inter, out, kHidden);
    }
    // The engine's f32_moe for rank r (routes in ids_h / wts_h).
    void moe(const hx::DevRank& rw, float* out) {
        const std::size_t n = R, pairs = n * kTopE;
        std::array<std::uint32_t, kExperts + 1> first{};
        for (std::size_t i = 0; i < pairs; ++i) ++first[ids_h[i] + 1];
        for (unsigned e = 0; e < kExperts; ++e) first[e + 1] += first[e];
        std::array<std::uint32_t, kExperts> fill{};
        std::uint32_t* xmap = maps_h.data();
        std::uint32_t* slot = maps_h.data() + pairs;
        float* wt = reinterpret_cast<float*>(maps_h.data() + 2 * pairs);
        for (std::size_t t = 0; t < n; ++t) {
            std::array<std::pair<std::uint32_t, float>, kTopE> r;
            for (unsigned i = 0; i < kTopE; ++i) r[i] = {ids_h[t * kTopE + i], wts_h[t * kTopE + i]};
            std::sort(r.begin(), r.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (unsigned i = 0; i < kTopE; ++i) {
                const std::uint32_t at = first[r[i].first] + fill[r[i].first]++;
                xmap[at] = static_cast<std::uint32_t>(t);
                slot[t * kTopE + i] = at;
                wt[t * kTopE + i] = r[i].second;
            }
        }
        ck(cudaMemcpy(maps_d, maps_h.data(), 3 * pairs * 4, cudaMemcpyHostToDevice), "maps");
        std::vector<fp::F32Job> jg, ju, jd;
        for (unsigned e = 0; e < kExperts; ++e) {
            const std::uint32_t m = first[e + 1] - first[e];
            if (m == 0) continue;
            jg.push_back(job(rw.experts[e].gate, m));
            ju.push_back(job(rw.experts[e].up, m));
            jd.push_back(job(rw.experts[e].down, m));
        }
        grouped(jg, hn, kHidden, maps_d, g, kExpertLocal);
        grouped(ju, hn, kHidden, maps_d, u, kExpertLocal);
        fp::enqueue_swiglu(g, u, g, pairs * kExpertLocal, nullptr);
        grouped(jd, g, kExpertLocal, nullptr, ye, kHidden);
        g_ph.lap(6);
        mlp(rw.shared, kExpertLocal, ys);  // g / u reused: the routed down above is earlier on the same stream
        fp::enqueue_combine(ye, ys, maps_d + pairs, reinterpret_cast<const float*>(maps_d + 2 * pairs), static_cast<std::uint32_t>(n), out, nullptr);
        g_ph.lap(7);
    }
    // One layer over all rows (weights bound in device memory).
    void layer(const hx::DevLayer& w, unsigned L, const cu::SeqView* seqs_d, const cu::RopeRow* rope_d) {
        g_ph.mark();
        for (unsigned r = 0; r < 2; ++r) linear(w.r[r].q, hn, kHidden, q + r * kQLocal, kQ);
        {  // k and v of both ranks in one call (4 x 16 CTAs at 1K instead of 4 launches of 16)
            const auto n = static_cast<std::uint32_t>(R);
            std::vector<fp::F32Job> j{job(w.r[0].k, n), job(w.r[1].k, n), job(w.r[0].v, n), job(w.r[1].v, n)};
            grouped(j, hn, kHidden, kv_map, k, kKVLocal, kv_map + 4 * R);
        }
        g_ph.lap(0);
        for (unsigned r = 0; r < 2; ++r) cu::dense_t(hn, w.r[r].wiq, iq + r * kIdxLocal, R, kIdxLocal, 512, nullptr);
        cu::dense_t(hn, w.wik, ik, R, 128, 128, nullptr);
        g_ph.lap(1);
        const cu::MsaLayer ml{L, w.q_w, w.k_w, w.iq_w, w.ik_w};
        cu::msa_prep(seqs_d, row_seq, R, ml, q, k, v, iq, ik, rope_d, nullptr, nullptr);
        g_ph.lap(2);
        cu::msa_attend(seqs_d, row_seq, R, max_blocks, L, q, iq, scores, sel, sel_count, attn, nullptr, tiles_d, n_tiles, attn_legacy);
        g_ph.lap(3);
        for (unsigned r = 0; r < 2; ++r) linear(w.r[r].o, attn + r * kQLocal, kQ, part[r], kHidden);
        g_ph.lap(4);
        cu::post_exchange(h, part[0], part[1], w.w_post, hn, R, nullptr);
        if (w.moe) {
            cu::route(hn, w.router, w.bias, R, scratch, ids_d, reinterpret_cast<float*>(ids_d + R * kTopE), nullptr);
            ck(cudaMemcpy(ids_h.data(), ids_d, R * kTopE * 4, cudaMemcpyDeviceToHost), "ids");
            ck(cudaMemcpy(wts_h.data(), ids_d + R * kTopE, R * kTopE * 4, cudaMemcpyDeviceToHost), "wts");
            g_ph.lap(5);
            for (unsigned r = 0; r < 2; ++r) moe(w.r[r], part[r]);
        } else {
            g_ph.lap(5);
            for (unsigned r = 0; r < 2; ++r) mlp(w.r[r].dense, kDenseLocal, part[r]);
            g_ph.lap(8);
        }
        cu::post_exchange(h, part[0], part[1], w.w_next, hn, R, nullptr);
        g_ph.lap(5);
    }
};

// Live hand-off (strix-minimax-rank --offload-listen): frame {u32 'MMRX', u32 layer, u64 rows, u64 bytes} + bytes.
struct Sender {
    SOCKET s = INVALID_SOCKET;
    void open(const std::string& hp) {
        const auto c = hp.find(':');
        s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(static_cast<u_short>(std::stoi(hp.substr(c + 1))));
        inet_pton(AF_INET, hp.substr(0, c).c_str(), &a.sin_addr);
        int buf = 32 << 20;
        ::setsockopt(s, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<const char*>(&buf), sizeof(buf));
        BOOL one = TRUE;
        ::setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));
        for (int tries = 0;; ++tries) {  // the ranks may still be loading: retry for up to 15 minutes
            if (::connect(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0) break;
            if (tries >= 900) throw std::runtime_error("send: cannot connect to " + hp);
            ::closesocket(s);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            ::setsockopt(s, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<const char*>(&buf), sizeof(buf));
            ::setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));
        }
    }
    void all(const void* p, std::size_t n) {
        const char* b = static_cast<const char*>(p);
        while (n) {
            const int k = ::send(s, b, static_cast<int>(std::min<std::size_t>(n, std::size_t{1} << 24)), 0);
            if (k <= 0) throw std::runtime_error("send failed");
            b += k;
            n -= static_cast<std::size_t>(k);
        }
    }
    // Magic: 'MMRX' plain / 'MMRB' bp-v1; with index keys once 'MMRI' / 'MMRJ' (rank 0: full K/V/IK block, it relays IK;
    // rank 1: the K/V head of the block only).
    void frame(std::uint32_t layer, std::uint64_t rows, const void* p, std::uint64_t bytes, bool bp = false, bool once = false) {
        const std::uint32_t h32[2] = {once ? (bp ? 0x4A524d4dU : 0x49524d4dU) : (bp ? 0x42524d4dU : 0x58524d4dU), layer};
        const std::uint64_t h64[2] = {rows, bytes};
        all(h32, 8);
        all(h64, 16);
        if (bytes) all(p, bytes);
    }
    std::uint32_t reply() {
        std::uint32_t t = 0;
        char* b = reinterpret_cast<char*>(&t);
        for (int got = 0; got < 4;) {
            const int k = ::recv(s, b + got, 4 - got, 0);
            if (k <= 0) throw std::runtime_error("send: no reply");
            got += k;
        }
        return t;
    }
    void close() {
        if (s != INVALID_SOCKET) ::closesocket(s);
        s = INVALID_SOCKET;
    }
};

// Weights of every layer for the timing mode: resident layers in one device buffer each; streamed layers as pinned
// host blobs, copied into two alternating device slots on a copy stream.
struct Store {
    std::vector<hx::LayerBlob> meta;            // offsets (bytes cleared after placement)
    std::vector<std::uint8_t*> dev;             // resident layers: device copy
    std::vector<std::uint8_t*> pinned;          // streamed layers: pinned host copy
    std::vector<std::size_t> size;
    std::uint8_t* slot[2] = {nullptr, nullptr};
    cudaStream_t copy = nullptr;
    cudaEvent_t ready[2]{}, free_[2]{};
    // disk tier
    static constexpr unsigned kStage = 4;
    std::vector<char> tier;                     // 'd' device, 'p' pinned, 'f' file
    std::vector<std::string> file;              // 'f': blob cache file (size padded to 4096)
    std::vector<std::uint64_t> padded;
    std::uint8_t* stage[kStage]{};
    cudaEvent_t stage_free[kStage]{};
};

// Unbuffered whole-file read into a sector-aligned (pinned) buffer.
void read_unbuffered(const std::string& path, std::uint8_t* dst, std::uint64_t bytes) {
    HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
                           nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error("open " + path);
    std::uint64_t at = 0;
    while (at < bytes) {
        const DWORD want = static_cast<DWORD>(std::min<std::uint64_t>(bytes - at, std::uint64_t{64} << 20));
        DWORD got = 0;
        if (!ReadFile(f, dst + at, want, &got, nullptr) || got == 0) {
            CloseHandle(f);
            throw std::runtime_error("read " + path);
        }
        at += got;
    }
    CloseHandle(f);
}

std::uint64_t hash_bytes(const std::vector<std::uint8_t>& b) {  // cache-file name: changes whenever the blob does
    std::uint64_t h = 1469598103934665603ULL ^ b.size();
    const std::size_t words = b.size() / 8;
    for (std::size_t i = 0; i < words; ++i) {
        std::uint64_t w;
        std::memcpy(&w, b.data() + 8 * i, 8);
        h = (h ^ w) * 1099511628211ULL;
    }
    for (std::size_t i = words * 8; i < b.size(); ++i) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}

#include "mm_rtx_layercut.inc"

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5) {
            std::fprintf(stderr, "usage: mm-rtx-prefill MODEL_DIR ROPE.bin OUT_DIR NAME:PROMPT.u32 [...] [--resident N] [--repeat K] [--layers N]\n");
            return 2;
        }
        const std::string model = argv[1], rope_path = argv[2], out_dir = argv[3];
        unsigned n_layers = kLayers, repeat = 1;
        lc::Cli cut_cli;
        bool print_effective = false;
        std::string cut_pair_identity = mlc::default_pair_identity();
        std::string send_to;
        bool codec_bp = false, abab_xport = false;
        strix::pair::state::SharedOnceMode ik_mode;  // off / on / alternating replay; last option wins
        int resident = -1, pinned_n = -1;
        std::string replay_dir, blob_cache;
        struct Prompt {
            std::string name;
            std::vector<std::uint32_t> tokens;
        };
        std::vector<Prompt> prompts;
        for (int i = 4; i < argc; ++i) {
            const std::string a = argv[i];
            auto cut_value = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
                return argv[++i];
            };
            if (a == "--layer-cut") { cut_cli.split = "layer-cut"; cut_cli.cut = cut_value(); continue; }
            if (a == "--split") { cut_cli.split = cut_value(); continue; }
            if (a == "--cut") { cut_cli.cut = cut_value(); continue; }
            if (a == "--cut-block") { cut_cli.block = cut_value(); continue; }
            if (a == "--cut-act") { cut_cli.act = cut_value(); continue; }
            if (a == "--cut-xport") { cut_cli.xport = cut_value(); continue; }
            if (a == "--cut-kv-frag") { cut_cli.kv_frag = cut_value(); continue; }
            if (a == "--cut-pair-identity") { cut_pair_identity = cut_value(); continue; }
            if (a == "--print-effective-config") { print_effective = true; continue; }
            if (a == "--layers") { n_layers = static_cast<unsigned>(std::stoul(argv[++i])); continue; }
            if (a == "--resident") { resident = std::stoi(argv[++i]); continue; }
            if (a == "--repeat") { repeat = static_cast<unsigned>(std::stoul(argv[++i])); continue; }
            if (a == "--send") { send_to = argv[++i]; continue; }
            if (a == "--codec") { codec_bp = std::string(argv[++i]) == "bp"; continue; }
            if (a == "--ik-once") {
                const bool ab = i + 1 < argc && std::string(argv[i + 1]) == "ab";
                if (ab) ++i;
                ik_mode.set(true, ab);
                continue;
            }
            if (a == "--no-ik-once") { ik_mode.set(false); continue; }
            if (a == "--abab-xport") { abab_xport = true; continue; }  // bench: layer-cut runs alternate xport (the resolved one first)
            if (a == "--replay-state") { replay_dir = argv[++i]; continue; }
            if (a == "--pinned") { pinned_n = std::stoi(argv[++i]); continue; }
            if (a == "--blob-cache") { blob_cache = argv[++i]; continue; }
            const auto c = a.find(':');
            prompts.push_back({a.substr(0, c), read_u32(a.substr(c + 1))});
        }
        // One resolver for the hand-off owners (shared with the ranks): flag, then STRIX_HANDOFF_SHARED_ONCE, then the
        // MiniMax default (per-rank copies until its pair receipt). The ranks follow the frame magic.
        namespace ps = strix::pair::state;
        const bool ik_once = ps::resolve(ps::decl::minimax, ik_mode.cli, -1).shared_once;
        if (ik_mode.ab && replay_dir.empty()) throw std::runtime_error("--ik-once ab needs --replay-state");
        if (!send_to.empty())
            std::printf("mm-rtx-prefill: hand-off owners %s (STRIX_HANDOFF_SHARED_ONCE / --ik-once): %.0f B per row\n",
                        ik_mode.ab ? "ABAB" : ik_once ? "ik-once" : "per-rank copies",
                        static_cast<double>(ps::host_bytes(ps::decl::minimax, 1, 0, ik_once && !ik_mode.ab)));
        lc::Inputs cut_input{mlc::chain_shape(), prompts.empty() ? 0 : prompts.front().tokens.size()};
        cut_input.rtx_rows_s = strix::envcfg::number<double>("STRIX_SPLIT_RTX_ROWS_S", 0, 0);
        cut_input.pair_rows_s = strix::envcfg::number<double>("STRIX_SPLIT_PAIR_ROWS_S", 0, 0);
        cut_input.link_mbps = strix::envcfg::number<double>("STRIX_SPLIT_LINK_MBPS", 0, 0);
        // Same resolver as the pair listener: the declared pair identity names its chunk rows (identity grain).
        cut_input.pair_grain = lc::identity_grain(cut_pair_identity);
        if (cut_input.pair_grain == mlc::pair_chunk_rows) cut_input.pair_unit_scale = mlc::pair_unit_scale();
        const auto cut_cfg = lc::resolve(cut_cli, cut_input);
        if (print_effective || cut_cfg.split == lc::chain::Split::layer_cut) std::printf("%s", cut_cfg.lines.c_str());
        if (print_effective) return 0;
        if (cut_cfg.planned && cut_cfg.split == lc::chain::Split::device_only && cut_cfg.first != 0)
            throw std::invalid_argument("--split auto chose pair device-only: run the prompt on the pair without the CUDA producer");
        const bool layer_cut = cut_cfg.split == lc::chain::Split::layer_cut;
        if (layer_cut) {
            if (prompts.size() != 1 || prompts[0].tokens.empty() || send_to.empty() || !replay_dir.empty() || pinned_n >= 0)
                throw std::invalid_argument("--layer-cut needs one prompt, --send and resident timing mode (no replay/pinned tier)");
            if (resident >= 0 && resident != static_cast<int>(cut_cfg.cut)) throw std::invalid_argument("--resident must equal --cut");
            resident = static_cast<int>(cut_cfg.cut); n_layers = cut_cfg.cut;
        }
        if (cut_cfg.split == lc::chain::Split::handoff && send_to.empty())
            throw std::invalid_argument("--split handoff-full requires --send");
        if (!replay_dir.empty()) {  // transport replay, no GPU
            WSADATA wsa;
            if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) throw std::runtime_error("WSAStartup");
            static strix::bp_codec::WorkerPool pool(8);
            for (const auto& pr : prompts) {
                const std::uint64_t rows = pr.tokens.size() - 1, bytes = rows * 1280;
                std::vector<std::uint8_t> st[2];
                for (unsigned r = 0; r < 2; ++r) {
                    std::ifstream f(replay_dir + "/state-" + pr.name + "-rank" + std::to_string(r) + ".bin", std::ios::binary | std::ios::ate);
                    if (!f || static_cast<std::uint64_t>(f.tellg()) != bytes * n_layers) throw std::runtime_error("replay: bad state file for " + pr.name);
                    st[r].resize(bytes * n_layers);
                    f.seekg(0);
                    f.read(reinterpret_cast<char*>(st[r].data()), static_cast<std::streamsize>(st[r].size()));
                }
                for (unsigned it = 0; it < repeat; ++it) {
                    const bool bp = codec_bp && !ik_mode.ab ? (it % 2 == 1) : codec_bp;  // --codec bp: identity, bp, identity, ...
                    Sender snd[2];
                    const auto c = send_to.find(',');
                    snd[0].open(send_to.substr(0, c));
                    snd[1].open(send_to.substr(c + 1));
                    const auto c0 = Clock::now();
                    double t_sent[2] = {0, 0}, ttft = 0;
                    std::uint64_t wire[2] = {0, 0};
                    std::uint32_t token = 0;
                    std::thread th[2];
                    for (unsigned r = 0; r < 2; ++r)
                        th[r] = std::thread([&, r]() {
                            // --ik-once ab: odd repeats send the index keys once (ABAB in one process, same state files)
                            const bool once = ik_mode.ab ? (it % 2 == 1) : ik_once;
                            const std::uint64_t sent = r == 1 && once ? rows * 1024 : bytes;
                            for (unsigned L = 0; L < n_layers; ++L) {
                                const std::uint8_t* p = st[r].data() + static_cast<std::size_t>(L) * bytes;
                                if (bp) {
                                    const strix::bp_codec::Region reg{p, sent, 2};
                                    const auto f = strix::bp_codec::encode(std::span<const strix::bp_codec::Region>(&reg, 1), &pool);
                                    snd[r].frame(L, rows, f.data(), f.size(), true, once);
                                    wire[r] += f.size();
                                } else {
                                    snd[r].frame(L, rows, p, sent, false, once);
                                    wire[r] += sent;
                                }
                            }
                            snd[r].frame(0xffffffffU, rows, nullptr, 0);
                            t_sent[r] = secs(c0, Clock::now());
                            if (r == 0) {
                                token = snd[r].reply();
                                ttft = secs(c0, Clock::now());
                            }
                        });
                    for (auto& t : th) t.join();
                    snd[0].close();
                    snd[1].close();
                    const double mb = static_cast<double>(wire[0] + wire[1]) / 1e6;
                    std::printf("mm-rtx-prefill: replay %s #%u %s%s: sent rank0 %.3f s rank1 %.3f s, wire %.1f MB (%.3f of %.1f MB plain), "
                                "%.1f MB/s plain-equivalent, reply (token %u) at %.3f s\n",
                                pr.name.c_str(), it, bp ? "bp-v1" : "identity", (ik_mode.ab ? (it % 2 == 1) : ik_once) ? " ik-once" : "", t_sent[0], t_sent[1], mb,
                                mb / (2.0 * static_cast<double>(bytes) * n_layers / 1e6), 2.0 * static_cast<double>(bytes) * n_layers / 1e6,
                                2.0 * static_cast<double>(bytes) * n_layers / 1e6 / std::max(t_sent[0], t_sent[1]), token, ttft);
                    std::fflush(stdout);
                }
            }
            return 0;
        }
        if (!send_to.empty() && resident < 0) throw std::runtime_error("--send needs the timing mode (--resident N)");
        const auto t0 = Clock::now();
        g_ph.on = std::getenv("MM_RTX_PROFILE") != nullptr;
        hx::Pack pack(model, layer_cut ? cut_cfg.cut : 0);
        std::vector<cu::RopeRow> rope;
        {
            std::size_t max_rows = 0;
            for (const auto& p : prompts) max_rows = std::max(max_rows, p.tokens.size());
            rope.resize(max_rows);
            std::ifstream f(rope_path, std::ios::binary);
            if (!f.read(reinterpret_cast<char*>(rope.data()), static_cast<std::streamsize>(rope.size() * sizeof(cu::RopeRow))))
                throw std::runtime_error("rope table too short");
        }
        cu::RopeRow* rope_d = dupload(rope);
        // embedding rows gathered on the host (BF16 -> FP32 is exact; keeps 2.5 GB of device memory for weights)
        std::vector<std::uint16_t> table;
        {
            const hx::TensorInfo& e = pack.info(kP + "embed_tokens.weight");
            table.resize(e.bytes / 2);
            pack.read(e, 0, e.bytes, table.data());
        }
        float* w_in0 = dupload(pack.floats(kP + "layers.0.input_layernorm.weight"));
        std::vector<cudaEvent_t> layer_done(kLayers);
        for (auto& e : layer_done) ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming | cudaEventBlockingSync), "event");

        // timing mode: every layer loaded once (resident or pinned), before any prompt
        Store st;
        std::vector<unsigned> streamed;
        if (resident >= 0) {
            st.meta.resize(n_layers);
            st.dev.assign(n_layers, nullptr);
            st.pinned.assign(n_layers, nullptr);
            st.size.assign(n_layers, 0);
            st.tier.assign(n_layers, 'p');
            st.file.assign(n_layers, std::string());
            st.padded.assign(n_layers, 0);
            if (pinned_n < 0) {
                for (unsigned L = 0; L < n_layers; ++L) st.tier[L] = static_cast<int>(L) < resident ? 'd' : 'p';
            } else {
                if (blob_cache.empty()) throw std::runtime_error("--pinned needs --blob-cache DIR");
                const unsigned dense = std::min(3U, n_layers), moe_n = n_layers - dense;
                const unsigned rm = static_cast<unsigned>(std::clamp(resident - static_cast<int>(dense), 0, static_cast<int>(moe_n)));
                for (unsigned L = 0; L < dense; ++L) st.tier[L] = 'd';
                for (unsigned i = 0; i < moe_n; ++i)  // MoE device layers spread evenly over the depth
                    st.tier[dense + i] = (static_cast<std::uint64_t>(i + 1) * rm / moe_n > static_cast<std::uint64_t>(i) * rm / moe_n) ? 'd' : 'p';
                std::vector<unsigned> rest;
                for (unsigned L = 0; L < n_layers; ++L)
                    if (st.tier[L] != 'd') rest.push_back(L);
                const std::size_t S = rest.size();
                const std::size_t F = S > static_cast<std::size_t>(pinned_n) ? S - static_cast<std::size_t>(pinned_n) : 0;
                if (F > 0 && S <= 2) throw std::runtime_error("disk tier needs more than 2 streamed layers");
                for (std::size_t j = 0; F > 0 && j + 2 < S; ++j)  // file layers spread over the streamed ones, never the first two
                    if ((j + 1) * F / (S - 2) > j * F / (S - 2)) st.tier[rest[j + 2]] = 'f';
            }
            {
                unsigned nd = 0, np = 0, nf = 0;
                std::string map;
                for (unsigned L = 0; L < n_layers; ++L) {
                    nd += st.tier[L] == 'd';
                    np += st.tier[L] == 'p';
                    nf += st.tier[L] == 'f';
                    map += st.tier[L];
                }
                std::printf("mm-rtx-prefill: tiers device %u, pinned %u, file %u: %s\n", nd, np, nf, map.c_str());
            }
            std::size_t slot_bytes = 0, stage_bytes = 0;
            const auto a = Clock::now();
            for (unsigned L = 0; L < n_layers; ++L) {
                st.meta[L] = hx::build_layer(pack, L, layer_cut && L + 1 == cut_cfg.cut);
                auto& bytes = st.meta[L].bytes;
                st.size[L] = bytes.size();
                if (st.tier[L] == 'd') {
                    st.dev[L] = dalloc<std::uint8_t>(bytes.size());
                    ck(cudaMemcpy(st.dev[L], bytes.data(), bytes.size(), cudaMemcpyHostToDevice), "resident upload");
                } else if (st.tier[L] == 'f') {
                    char name[64];
                    std::snprintf(name, sizeof(name), "/layer-%02u-%016llx.bin", L, static_cast<unsigned long long>(hash_bytes(bytes)));
                    st.file[L] = blob_cache + name;
                    st.padded[L] = (bytes.size() + 4095) / 4096 * 4096;
                    std::ifstream probe(st.file[L], std::ios::binary | std::ios::ate);
                    if (!probe || static_cast<std::uint64_t>(probe.tellg()) != st.padded[L]) {
                        probe.close();
                        std::ofstream o(st.file[L], std::ios::binary | std::ios::trunc);
                        o.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
                        const std::vector<char> pad(st.padded[L] - bytes.size(), 0);
                        o.write(pad.data(), static_cast<std::streamsize>(pad.size()));
                        if (!o) throw std::runtime_error("write " + st.file[L]);
                        std::printf("mm-rtx-prefill: wrote blob cache %s\n", st.file[L].c_str());
                    }
                    slot_bytes = std::max(slot_bytes, bytes.size());
                    stage_bytes = std::max<std::size_t>(stage_bytes, st.padded[L]);
                    streamed.push_back(L);
                } else {
                    ck(cudaHostAlloc(reinterpret_cast<void**>(&st.pinned[L]), bytes.size(), cudaHostAllocDefault), "pinned");
                    std::memcpy(st.pinned[L], bytes.data(), bytes.size());
                    slot_bytes = std::max(slot_bytes, bytes.size());
                    streamed.push_back(L);
                }
                std::vector<std::uint8_t>().swap(bytes);
                if (L % 5 == 4 || L + 1 == n_layers) {
                    std::size_t fr = 0, tot = 0;
                    ck(cudaMemGetInfo(&fr, &tot), "meminfo");
                    PROCESS_MEMORY_COUNTERS_EX pm{};
                    GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pm), sizeof(pm));
                    std::size_t pinned_bytes = 0;
                    for (unsigned q = 0; q <= L; ++q) pinned_bytes += st.pinned[q] ? st.size[q] : 0;
                    std::printf("mm-rtx-prefill: loaded layers 0-%u, %.1f s, read %.1f GiB, device used %.1f GiB, %zu streamed so far "
                                "(pinned %.1f GiB; process private %.1f GiB, working set %.1f GiB)\n", L,
                                secs(a, Clock::now()), pack.gib_read(), static_cast<double>(tot - fr) / 1073741824.0, streamed.size(),
                                static_cast<double>(pinned_bytes) / 1073741824.0, static_cast<double>(pm.PrivateUsage) / 1073741824.0,
                                static_cast<double>(pm.WorkingSetSize) / 1073741824.0);
                    std::fflush(stdout);
                }
            }
            if (slot_bytes) {
                st.slot[0] = dalloc<std::uint8_t>(slot_bytes);
                st.slot[1] = dalloc<std::uint8_t>(slot_bytes);
            }
            if (stage_bytes)
                for (unsigned i = 0; i < Store::kStage; ++i) {
                    ck(cudaHostAlloc(reinterpret_cast<void**>(&st.stage[i]), stage_bytes, cudaHostAllocDefault), "staging ring");
                    ck(cudaEventCreateWithFlags(&st.stage_free[i], cudaEventDisableTiming | cudaEventBlockingSync), "event");
                }
            ck(cudaStreamCreateWithFlags(&st.copy, cudaStreamNonBlocking), "copy stream");
            for (int i = 0; i < 2; ++i) {
                ck(cudaEventCreateWithFlags(&st.ready[i], cudaEventDisableTiming), "event");
                ck(cudaEventCreateWithFlags(&st.free_[i], cudaEventDisableTiming), "event");
            }
            WSADATA wsa;
            if (!send_to.empty() && WSAStartup(MAKEWORD(2, 2), &wsa) != 0) throw std::runtime_error("WSAStartup");
        }

        if (layer_cut) {
            run_layercut(cut_cfg, st, table, w_in0, rope_d, prompts[0].tokens, send_to, cut_pair_identity, repeat, codec_bp, abab_xport);
            return 0;
        }
        // groups: gate mode = all prompts batched in one pass; timing mode = one prompt at a time
        std::vector<std::vector<std::size_t>> groups;
        if (resident < 0) {
            groups.emplace_back();
            for (std::size_t i = 0; i < prompts.size(); ++i) groups.back().push_back(i);
        } else {
            for (std::size_t i = 0; i < prompts.size(); ++i) groups.push_back({i});
        }
        for (const auto& grp : groups) {
            // sequences: rows [0, P-1) of each prompt of the group
            std::vector<cu::SeqView> seqs;
            std::vector<std::uint32_t> tokens, row_seq;
            std::size_t max_rows = 0;
            for (std::size_t s = 0; s < grp.size(); ++s) {
                const auto& pt = prompts[grp[s]].tokens;
                const std::size_t rows = pt.size() - 1;
                cu::SeqView v{};
                v.row0 = static_cast<std::uint32_t>(tokens.size());
                v.rows = static_cast<std::uint32_t>(rows);
                v.cap = static_cast<std::uint32_t>(rows);
                v.k = dalloc<std::uint16_t>(std::size_t{kLayers} * rows * 4 * 128);
                v.v = dalloc<std::uint16_t>(std::size_t{kLayers} * rows * 4 * 128);
                v.ik = dalloc<std::uint16_t>(std::size_t{kLayers} * rows * 128);
                seqs.push_back(v);
                tokens.insert(tokens.end(), pt.begin(), pt.end() - 1);
                for (std::size_t r = 0; r < rows; ++r) row_seq.push_back(static_cast<std::uint32_t>(s));
                max_rows = std::max(max_rows, rows);
            }
            const std::size_t R = tokens.size();
            Runner run;
            run.alloc(R, static_cast<unsigned>((max_rows + 127) / 128));
            cu::SeqView* seqs_d = dupload(seqs);
            run.row_seq = dupload(row_seq);
            {
                std::vector<uint2> tiles;
                for (const auto& s : seqs)
                    for (std::uint32_t r = 0; r < s.rows; r += 32) tiles.push_back(make_uint2(s.row0 + r, std::min<std::uint32_t>(32, s.rows - r)));
                run.tiles_d = dupload(tiles);
                run.n_tiles = static_cast<unsigned>(tiles.size());
            }
            // state blobs (pinned: the per-layer copies are asynchronous; a sender thread reads them after the layer's event)
            std::vector<std::uint8_t*> blobs[2];
            for (unsigned r = 0; r < 2; ++r)
                for (const auto& s : seqs) {
                    std::uint8_t* b = nullptr;
                    ck(cudaHostAlloc(reinterpret_cast<void**>(&b), static_cast<std::size_t>(s.rows) * kLayers * 1280, cudaHostAllocDefault), "pinned blob");
                    blobs[r].push_back(b);
                }
            std::uint32_t* emb_h = nullptr;
            ck(cudaHostAlloc(reinterpret_cast<void**>(&emb_h), R * kHidden * 4, cudaHostAllocDefault), "embed staging");
            // layer_done[L] is recorded by this thread; a sender waits on the host counter first (synchronizing on an event
            // that is not recorded yet returns at once: the 18:47 measurement senders shipped unwritten blobs)
            std::mutex rec_m;
            std::condition_variable rec_cv;
            unsigned recorded = 0;
            auto export_all = [&](unsigned L) {
                for (std::size_t s = 0; s < seqs.size(); ++s)
                    for (unsigned r = 0; r < 2; ++r)
                        cu::export_layer(seqs[s], L, r, seqs[s].rows, blobs[r][s] + static_cast<std::size_t>(L) * seqs[s].rows * 1280, nullptr);
                ck(cudaEventRecord(layer_done[L], nullptr), "layer event");
                std::lock_guard<std::mutex> lk(rec_m);
                recorded = L + 1;
                rec_cv.notify_all();
            };
            auto start = [&]() {
                for (std::size_t r = 0; r < R; ++r) {
                    const std::uint16_t* src = table.data() + static_cast<std::size_t>(tokens[r]) * kHidden;
                    for (unsigned i = 0; i < kHidden; ++i) emb_h[r * kHidden + i] = static_cast<std::uint32_t>(src[i]) << 16U;
                }
                ck(cudaMemcpy(run.h, emb_h, R * kHidden * 4, cudaMemcpyHostToDevice), "embed rows");
                cu::rmsnorm(run.h, w_in0, run.hn, R, nullptr);
            };
            std::printf("mm-rtx-prefill: group of %zu prompt(s), %zu rows, %.1f s since start\n", grp.size(), R, secs(t0, Clock::now()));
            std::fflush(stdout);
            if (resident < 0) {  // gate mode: each layer read when needed
                double t_load = 0, t_comp = 0;
                start();
                for (unsigned L = 0; L < n_layers; ++L) {
                    const auto a = Clock::now();
                    const hx::LayerBlob blob = hx::build_layer(pack, L);
                    std::uint8_t* d = dalloc<std::uint8_t>(blob.bytes.size());
                    ck(cudaMemcpy(d, blob.bytes.data(), blob.bytes.size(), cudaMemcpyHostToDevice), "layer upload");
                    const auto b = Clock::now();
                    t_load += secs(a, b);
                    run.layer(hx::bind(blob, d), L, seqs_d, rope_d);
                    export_all(L);
                    ck(cudaDeviceSynchronize(), "layer");
                    g_ph.lap(9);
                    cudaFree(d);
                    t_comp += secs(b, Clock::now());
                    if (L % 10 == 9 || L + 1 == n_layers)
                        std::printf("mm-rtx-prefill: layer %u done, load %.1f s, compute %.1f s, read %.1f GiB\n", L, t_load, t_comp, pack.gib_read());
                    std::fflush(stdout);
                }
                std::printf("mm-rtx-prefill: weights load %.1f s, compute %.1f s\n", t_load, t_comp);
            } else {  // timing mode
                for (unsigned it = 0; it < repeat; ++it) {
                    ck(cudaDeviceSynchronize(), "sync");
                    Sender snd[2];
                    if (!send_to.empty()) {
                        const auto c = send_to.find(',');
                        snd[0].open(send_to.substr(0, c));
                        snd[1].open(send_to.substr(c + 1));
                    }
                    const auto c0 = Clock::now();
                    std::thread th[2];
                    double t_sent[2] = {0, 0}, ttft = 0, t_enc[2] = {0, 0};
                    std::uint64_t wire[2] = {0, 0};
                    static strix::bp_codec::WorkerPool pool(8);
                    std::uint32_t token = 0;
                    {
                        std::lock_guard<std::mutex> lk(rec_m);
                        recorded = 0;
                    }
                    if (!send_to.empty())
                        for (unsigned r = 0; r < 2; ++r)
                            th[r] = std::thread([&, r]() {
                                // IK once (shared piece): rank 1's frame is the block's K/V head [0, rows * 1024); rank 0 relays IK.
                                const std::uint64_t rows = seqs[0].rows, block = rows * 1280,
                                                    bytes = r == 1 && ik_once ? rows * 1024 : block;
                                for (unsigned L = 0; L < n_layers; ++L) {
                                    {
                                        std::unique_lock<std::mutex> lk(rec_m);
                                        rec_cv.wait(lk, [&] { return recorded > L; });
                                    }
                                    ck(cudaEventSynchronize(layer_done[L]), "layer done");
                                    const std::uint8_t* p = blobs[r][0] + static_cast<std::size_t>(L) * block;
                                    if (codec_bp) {
                                        const auto e0 = Clock::now();
                                        const strix::bp_codec::Region reg{p, bytes, 2};  // BF16 rows
                                        const auto f = strix::bp_codec::encode(std::span<const strix::bp_codec::Region>(&reg, 1), &pool);
                                        t_enc[r] += secs(e0, Clock::now());
                                        snd[r].frame(L, rows, f.data(), f.size(), true, ik_once);
                                        wire[r] += f.size();
                                    } else {
                                        snd[r].frame(L, rows, p, bytes, false, ik_once);
                                        wire[r] += bytes;
                                    }
                                }
                                snd[r].frame(0xffffffffU, rows, nullptr, 0);
                                t_sent[r] = secs(c0, Clock::now());
                                if (r == 0) {
                                    token = snd[r].reply();
                                    ttft = secs(c0, Clock::now());
                                }
                            });
                    std::size_t next = 0;
                    // disk tier: a reader thread fills staging slot k % 4 with the k-th file layer (in stream order)
                    // once the H2D copy of file layer k - 4 has been issued and has finished
                    std::vector<int> ford(streamed.size(), -1);
                    std::vector<unsigned> flayers;
                    for (std::size_t i = 0; i < streamed.size(); ++i)
                        if (st.tier[streamed[i]] == 'f') {
                            ford[i] = static_cast<int>(flayers.size());
                            flayers.push_back(streamed[i]);
                        }
                    std::mutex fm;
                    std::condition_variable fcv;
                    std::size_t f_read = 0, f_issued = 0;
                    std::string f_err;
                    double t_read = 0;
                    std::thread reader;
                    if (!flayers.empty())
                        reader = std::thread([&]() {
                            const auto r0 = Clock::now();
                            try {
                                for (std::size_t k = 0; k < flayers.size(); ++k) {
                                    const unsigned s_ = static_cast<unsigned>(k % Store::kStage);
                                    if (k >= Store::kStage) {
                                        std::unique_lock<std::mutex> lk(fm);
                                        fcv.wait(lk, [&] { return f_issued > k - Store::kStage; });
                                        lk.unlock();
                                        ck(cudaEventSynchronize(st.stage_free[s_]), "staging free");
                                    }
                                    read_unbuffered(st.file[flayers[k]], st.stage[s_], st.padded[flayers[k]]);
                                    std::lock_guard<std::mutex> lk(fm);
                                    f_read = k + 1;
                                    fcv.notify_all();
                                }
                            } catch (const std::exception& e) {
                                std::lock_guard<std::mutex> lk(fm);
                                f_err = e.what();
                                f_read = flayers.size() + 1;  // wake the issuer
                                fcv.notify_all();
                            }
                            t_read = secs(r0, Clock::now());
                        });
                    auto issue = [&](std::size_t i) {  // streamed layer streamed[i] -> slot i % 2
                        const int sl = static_cast<int>(i % 2);
                        const std::uint8_t* src = st.pinned[streamed[i]];
                        if (ford[i] >= 0) {
                            const auto k = static_cast<std::size_t>(ford[i]);
                            std::unique_lock<std::mutex> lk(fm);
                            fcv.wait(lk, [&] { return f_read > k; });
                            if (!f_err.empty()) throw std::runtime_error("disk tier: " + f_err);
                            src = st.stage[k % Store::kStage];
                        }
                        ck(cudaStreamWaitEvent(st.copy, st.free_[sl], 0), "wait free");
                        ck(cudaMemcpyAsync(st.slot[sl], src, st.size[streamed[i]], cudaMemcpyHostToDevice, st.copy), "stream layer");
                        ck(cudaEventRecord(st.ready[sl], st.copy), "ready");
                        if (ford[i] >= 0) {
                            const auto k = static_cast<std::size_t>(ford[i]);
                            ck(cudaEventRecord(st.stage_free[k % Store::kStage], st.copy), "staging used");
                            std::lock_guard<std::mutex> lk(fm);
                            f_issued = k + 1;
                            fcv.notify_all();
                        }
                    };
                    for (; next < streamed.size() && next < 2; ++next) issue(next);  // the first two streamed layers
                    start();
                    std::size_t si = 0;
                    for (unsigned L = 0; L < n_layers; ++L) {
                        const std::uint8_t* base = st.dev[L];
                        int sl = -1;
                        if (!base) {
                            sl = static_cast<int>(si % 2);
                            ck(cudaStreamWaitEvent(nullptr, st.ready[sl], 0), "wait ready");
                            base = st.slot[sl];
                        }
                        run.layer(hx::bind(st.meta[L], base), L, seqs_d, rope_d);
                        export_all(L);
                        if (sl >= 0) {
                            ck(cudaEventRecord(st.free_[sl], nullptr), "free");
                            ++si;
                            if (next < streamed.size()) issue(next++);
                        }
                        g_ph.lap(9);
                    }
                    ck(cudaDeviceSynchronize(), "run");
                    const double t_comp = secs(c0, Clock::now());
                    std::printf("mm-rtx-prefill: %s run %u: %zu rows in %.3f s (%.1f tok/s), %zu streamed layers\n", prompts[grp[0]].name.c_str(), it, R,
                                t_comp, static_cast<double>(R) / t_comp, streamed.size());
                    if (reader.joinable()) {
                        reader.join();
                        double gb = 0;
                        for (unsigned L : flayers) gb += static_cast<double>(st.padded[L]) / 1e9;
                        std::printf("mm-rtx-prefill: %s run %u disk tier: %zu layers, %.1f GB in %.2f s (%.2f GB/s); idling %.2f s (D: duty rule)\n",
                                    prompts[grp[0]].name.c_str(), it, flayers.size(), gb, t_read, gb / t_read, t_read);
                    }
                    if (!send_to.empty()) {
                        for (auto& t : th) t.join();
                        snd[0].close();
                        snd[1].close();
                        std::printf("mm-rtx-prefill: %s run %u hand-off: compute %.3f s, sent rank0 %.3f s rank1 %.3f s (wire %.1f / %.1f MB of %.1f, "
                                    "encode %.3f / %.3f s), TTFT %.3f s (prompt %zu tokens, first token %u)\n",
                                    prompts[grp[0]].name.c_str(), it, t_comp, t_sent[0], t_sent[1], static_cast<double>(wire[0]) / 1e6,
                                    static_cast<double>(wire[1]) / 1e6, static_cast<double>(seqs[0].rows) * kLayers * 1280 / 1e6, t_enc[0], t_enc[1],
                                    ttft, R + 1, token);
                    }
                    std::fflush(stdout);
                    if (!flayers.empty()) std::this_thread::sleep_for(std::chrono::duration<double>(t_read));
                }
            }
            for (std::size_t s = 0; s < seqs.size(); ++s)
                for (unsigned r = 0; r < 2; ++r) {
                    const std::string p = out_dir + "/state-" + prompts[grp[s]].name + "-rank" + std::to_string(r) + ".bin";
                    std::ofstream o(p, std::ios::binary);
                    o.write(reinterpret_cast<const char*>(blobs[r][s]), static_cast<std::streamsize>(static_cast<std::size_t>(seqs[s].rows) * kLayers * 1280));
                    std::printf("mm-rtx-prefill: wrote %s (%.1f MB)%s\n", p.c_str(), static_cast<double>(seqs[s].rows) * kLayers * 1280 / 1e6,
                                o ? "" : " FAILED");
                }
            // release the group's buffers (the next group may be larger)
            ck(cudaDeviceSynchronize(), "group");
            run.release();
            for (auto& s : seqs) {
                cudaFree(s.k);
                cudaFree(s.v);
                cudaFree(s.ik);
            }
            cudaFree(seqs_d);
            for (unsigned r = 0; r < 2; ++r)
                for (auto* b : blobs[r]) cudaFreeHost(b);
            cudaFreeHost(emb_h);
        }
        if (g_ph.on) {
            std::printf("mm-rtx-prefill: phases (s):");
            for (int i = 0; i < 10; ++i) std::printf(" %s %.3f", kPhase[i], g_ph.s[i]);
            std::printf(" | msa scores %.3f select %.3f attend %.3f\n", cu::g_attn_prof[0], cu::g_attn_prof[1], cu::g_attn_prof[2]);
        }
        std::printf("mm-rtx-prefill: total %.1f s\n", secs(t0, Clock::now()));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mm-rtx-prefill: %s\n", e.what());
        return 1;
    }
}
