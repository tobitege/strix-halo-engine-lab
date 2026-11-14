// Device gate and operator bench of the shared trellis codec (core/trellis_codec.hpp).
// Every reader / codebook combination decodes the same random packed tensors on the device and must reproduce the
// host reference image (Format::state + Mul1::value) bit for bit, for Bits 1..8, including single-tile and odd tile
// counts. STRIX_TRELLIS_CODEC_BENCH=<repeats> additionally times each decoder on a larger tensor (ms, best of N).
#include "core/tiled_core.hpp"
#include "core/trellis_codec.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace kernels {
using namespace strix::core::trellis;

// Sink = false: out[tile * 256 + position] = FP16 bits of state(position). Sink = true (bench): every lane decodes,
// lane 0 (or column 0) stores the XOR of its halves at out[tile], so the time is the decode and not the store.
// Bench launches read a cache-resident window of the tensor (kBenchTiles tiles), so the time is the decode work and
// not the packed stream's memory bandwidth.
constexpr unsigned kBenchTiles = 2048;
template <bool Sink> __host__ __device__ constexpr unsigned source_tile(unsigned tile) { return Sink ? tile % kBenchTiles : tile; }
__device__ __forceinline__ std::uint16_t fold(std::uint32_t w) { return static_cast<std::uint16_t>(w ^ (w >> 16U)); }
template <bool Sink> __device__ __forceinline__ void finish_lane(const std::uint32_t (&words)[4], std::uint16_t* out, unsigned tile,
                                                               unsigned lane) {
    if constexpr (Sink) {
        if (lane == 0) out[tile] = fold(words[0] ^ words[1] ^ words[2] ^ words[3]);
    } else {
        std::uint16_t* dst = out + static_cast<std::size_t>(tile) * 256U + lane * 8U;
#pragma unroll
        for (int j = 0; j < 4; ++j) *reinterpret_cast<std::uint32_t*>(dst + 2 * j) = words[j];
    }
}
template <bool Sink> __device__ __forceinline__ void finish_fragment(const std::uint32_t (&frag)[8], std::uint16_t* out, unsigned tile,
                                                                   unsigned column) {
    if constexpr (Sink) {
        if (column == 0) out[tile] = fold(frag[0] ^ frag[1] ^ frag[2] ^ frag[3] ^ frag[4] ^ frag[5] ^ frag[6] ^ frag[7]);
    } else {
        std::uint16_t* dst = out + static_cast<std::size_t>(tile) * 256U;
#pragma unroll
        for (unsigned k = 0; k < 16; ++k) {
            const unsigned position = ((column & 7U) << 5U) | ((k & 6U) << 2U) | (k & 1U) | ((k & 8U) >> 2U) | ((column & 8U) >> 1U);
            dst[position] = static_cast<std::uint16_t>(frag[k / 2U] >> (16U * (k & 1U)));
        }
    }
}
enum class Lane { pair, pair_masked, half, half16, value, wave, tiled };

template <int Bits, Lane Mode, bool Sink> __global__ void lane_kernel(const std::uint32_t* trellis, std::uint16_t* out, unsigned tiles) {
    const unsigned lane = threadIdx.x & 31U;
    const unsigned tile = uni(blockIdx.x * (blockDim.x / 32U) + threadIdx.x / 32U);
    if (tile >= tiles) return;
    const std::uint32_t* ring = uniform(trellis) + static_cast<std::size_t>(source_tile<Sink>(tile)) * (8U * Bits);
    std::uint32_t st[8];
    if constexpr (Mode == Lane::wave) {
        const WaveTile<Bits> wave(ring);
#pragma unroll
        for (int j = 0; j < 8; ++j) st[j] = wave.state(static_cast<int>(lane) * 8 + j);
    } else {
        const LaneGeom<Bits> geom(lane);
        const TileStates<Bits> t = align_words<Bits>(LaneWords<Bits>::issue(ring, geom), geom.align);
        st[0] = t.template state<0>(); st[1] = t.template state<1>(); st[2] = t.template state<2>(); st[3] = t.template state<3>();
        st[4] = t.template state<4>(); st[5] = t.template state<5>(); st[6] = t.template state<6>(); st[7] = t.template state<7>();
    }
    std::uint32_t words[4];
    if constexpr (Mode == Lane::pair) {
        // States taken as extracted (State16): the production idiom mul1_pair(t.state<A>(), t.state<B>()).
        const LaneGeom<Bits> geom(lane);
        const TileStates<Bits> t = align_words<Bits>(LaneWords<Bits>::issue(ring, geom), geom.align);
        words[0] = __builtin_bit_cast(std::uint32_t, mul1_pair(t.template state<0>(), t.template state<1>()));
        words[1] = __builtin_bit_cast(std::uint32_t, mul1_pair(t.template state<2>(), t.template state<3>()));
        words[2] = __builtin_bit_cast(std::uint32_t, mul1_pair(t.template state<4>(), t.template state<5>()));
        words[3] = __builtin_bit_cast(std::uint32_t, mul1_pair(t.template state<6>(), t.template state<7>()));
    } else if constexpr (Mode == Lane::tiled) {
        // The tiled core's names for the lane reader (core/tiled_core.hpp RawLane / Stream / decode_pair).
        strix::core::wmma_tiled::RawLane<Bits> raw;
        raw.load(ring);
        const strix::core::wmma_tiled::Stream<Bits> s(raw);
        words[0] = strix::core::wmma_tiled::decode_pair<2>(s.template state<0>(), s.template state<1>());
        words[1] = strix::core::wmma_tiled::decode_pair<2>(s.template state<2>(), s.template state<3>());
        words[2] = strix::core::wmma_tiled::decode_pair<2>(s.template state<4>(), s.template state<5>());
        words[3] = strix::core::wmma_tiled::decode_pair<2>(s.template state<6>(), s.template state<7>());
    } else if constexpr (Mode == Lane::pair_masked) {
#pragma unroll
        for (int j = 0; j < 4; ++j) words[j] = __builtin_bit_cast(std::uint32_t, Mul1::pair(st[2 * j], st[2 * j + 1]));
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            std::uint16_t v[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                _Float16 h;
                if constexpr (Mode == Lane::half) h = Mul1::half(st[2 * j + i]);
                else if constexpr (Mode == Lane::half16) h = Mul1::half16(st[2 * j + i]);
                else h = static_cast<_Float16>(Mul1::value(st[2 * j + i]));
                v[i] = __builtin_bit_cast(std::uint16_t, h);
            }
            words[j] = static_cast<std::uint32_t>(v[0]) | (static_cast<std::uint32_t>(v[1]) << 16U);
        }
    }
    finish_lane<Sink>(words, out, tile, lane);
}

// WMMA fragment geometry: lanes 0..15 = output columns, 16 input coordinates each.
enum class Frag { scalar, pair, decode16, pipe, v2 };

template <int Bits, Frag Mode, bool Sink> __global__ void fragment_kernel(const std::uint32_t* trellis, std::uint16_t* out, unsigned tiles) {
    using Reader = Fragment<Bits>;
    const unsigned lane = threadIdx.x & 31U, column = lane & 15U;
    // Two tiles per wave (lanes 0..15 and 16..31), as a matrix kernel with two B fragments per wave would hold them.
    const unsigned tile = 2U * (blockIdx.x * (blockDim.x / 32U) + threadIdx.x / 32U) + (lane >> 4U);
    if (tile >= tiles) return;
    const Reader reader(static_cast<int>(column));
    std::uint32_t raw[Reader::kLoads], nrm[Reader::kNormalized], frag[8];
    reader.issue(trellis + static_cast<std::size_t>(source_tile<Sink>(tile)) * (8U * Bits), raw);
    reader.normalize(raw, nrm);
    if constexpr (Mode == Frag::decode16) {
        Reader::template decode16<Mul1>(nrm, frag);
    } else {
        std::uint32_t st[16];
        st[0] = Reader::template state<0>(nrm); st[1] = Reader::template state<1>(nrm); st[2] = Reader::template state<2>(nrm);
        st[3] = Reader::template state<3>(nrm); st[4] = Reader::template state<4>(nrm); st[5] = Reader::template state<5>(nrm);
        st[6] = Reader::template state<6>(nrm); st[7] = Reader::template state<7>(nrm); st[8] = Reader::template state<8>(nrm);
        st[9] = Reader::template state<9>(nrm); st[10] = Reader::template state<10>(nrm); st[11] = Reader::template state<11>(nrm);
        st[12] = Reader::template state<12>(nrm); st[13] = Reader::template state<13>(nrm); st[14] = Reader::template state<14>(nrm);
        st[15] = Reader::template state<15>(nrm);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if constexpr (Mode == Frag::pair) {
                frag[i] = __builtin_bit_cast(std::uint32_t, Mul1::pair(st[2 * i], st[2 * i + 1]));
            } else {
                std::uint16_t v[2];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    float f;
                    if constexpr (Mode == Frag::pipe) f = Mul1::pipe_value(st[2 * i + j]);
                    else if constexpr (Mode == Frag::v2) f = Mul1::value_v2(st[2 * i + j]);
                    else f = Mul1::value(st[2 * i + j]);
                    v[j] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(f));
                }
                frag[i] = static_cast<std::uint32_t>(v[0]) | (static_cast<std::uint32_t>(v[1]) << 16U);
            }
        }
    }
    finish_fragment<Sink>(frag, out, tile, column);
}

// 16-byte lane windows: lane l decodes tile (base + l); whole-tile window, first 32 ring positions of each group of 8.
template <int Bits, bool Sink> __global__ void lane_tile_kernel(const std::uint32_t* trellis, std::uint16_t* out, unsigned tiles) {
    const unsigned tile = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile >= tiles) return;
    LaneTile<static_cast<unsigned>(Bits), 8, 0> window;
    window.load(trellis + static_cast<std::size_t>(source_tile<Sink>(tile)) * (8U * Bits));
    std::uint16_t* dst = out + static_cast<std::size_t>(tile) * 256U;
    std::uint16_t sum = 0;
#define STRIX_TRELLIS_LT(P) { const std::uint16_t b = __builtin_bit_cast(std::uint16_t, Mul1::half(window.template state<P>())); if constexpr (Sink) sum ^= b; else dst[P] = b; }
#define STRIX_TRELLIS_LT8(P) STRIX_TRELLIS_LT(P) STRIX_TRELLIS_LT(P + 1) STRIX_TRELLIS_LT(P + 2) STRIX_TRELLIS_LT(P + 3) \
    STRIX_TRELLIS_LT(P + 4) STRIX_TRELLIS_LT(P + 5) STRIX_TRELLIS_LT(P + 6) STRIX_TRELLIS_LT(P + 7)
    STRIX_TRELLIS_LT8(0) STRIX_TRELLIS_LT8(8) STRIX_TRELLIS_LT8(56) STRIX_TRELLIS_LT8(120) STRIX_TRELLIS_LT8(184) STRIX_TRELLIS_LT8(248)
#undef STRIX_TRELLIS_LT8
#undef STRIX_TRELLIS_LT
    if constexpr (Sink) out[tile] = sum;
}
constexpr bool lane_tile_covers(unsigned p) {
    return p < 16 || (p >= 56 && p < 64) || (p >= 120 && p < 128) || (p >= 184 && p < 192) || p >= 248;
}

// Every codebook form for all 65,536 states: out[form * 65536 + state]. The forms that read only the low 16 bits of
// the container (hash16) also get a container with nonzero high bits.
constexpr unsigned kBookForms = 9;
__global__ void codebook_kernel(std::uint16_t* out) {
    const std::uint32_t s = blockIdx.x * blockDim.x + threadIdx.x, dirty = s | 0xa5c30000U;
    const auto bits = [](_Float16 v) { return __builtin_bit_cast(std::uint16_t, v); };
    const auto word = [](Half2 v) { return __builtin_bit_cast(std::uint32_t, v); };
    out[0U * 65536U + s] = bits(static_cast<_Float16>(Mul1::value(s)));
    out[1U * 65536U + s] = bits(static_cast<_Float16>(Mul1::value_v2(s)));
    out[2U * 65536U + s] = bits(static_cast<_Float16>(Mul1::pipe_value(s)));
    out[3U * 65536U + s] = bits(Mul1::half(s));
    out[4U * 65536U + s] = bits(Mul1::half16(s));
    out[5U * 65536U + s] = bits(Mul1::half16(dirty));
    out[6U * 65536U + s] = static_cast<std::uint16_t>(word(Mul1::pair(s, s ^ 0x5a5aU)));
    out[7U * 65536U + s] = static_cast<std::uint16_t>(word(Mul1::pair(s ^ 0xa5a5U, dirty)) >> 16U);
    out[8U * 65536U + s] = bits(static_cast<_Float16>(codebook_value(s, 2)));
}
}  // namespace kernels

// ---- driver (shared with the pre-unification variant of this fixture)
namespace {
void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::fprintf(stderr, "HIP error %s: %s\n", what, hipGetErrorString(e)); std::exit(2); }
}
std::uint64_t splitmix(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}
std::uint64_t fnv(const void* p, std::size_t n, std::uint64_t h = 1469598103934665603ULL) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

template <int Bits> struct Suite {
    template <bool Sink> static void launch(int v, const std::uint32_t* in, std::uint16_t* out, unsigned tiles) {
        using namespace kernels;
        const dim3 block(256);
        const dim3 lane_grid((tiles + 7U) / 8U), frag_grid((tiles + 15U) / 16U), tile_grid((tiles + 255U) / 256U);
        switch (v) {
        case 0: hipLaunchKernelGGL((lane_kernel<Bits, Lane::pair, Sink>), lane_grid, block, 0, nullptr, in, out, tiles); break;
        case 1: hipLaunchKernelGGL((lane_kernel<Bits, Lane::pair_masked, Sink>), lane_grid, block, 0, nullptr, in, out, tiles); break;
        case 2: hipLaunchKernelGGL((lane_kernel<Bits, Lane::half, Sink>), lane_grid, block, 0, nullptr, in, out, tiles); break;
        case 3: hipLaunchKernelGGL((lane_kernel<Bits, Lane::half16, Sink>), lane_grid, block, 0, nullptr, in, out, tiles); break;
        case 4: hipLaunchKernelGGL((lane_kernel<Bits, Lane::value, Sink>), lane_grid, block, 0, nullptr, in, out, tiles); break;
        case 5: hipLaunchKernelGGL((lane_kernel<Bits, Lane::wave, Sink>), lane_grid, block, 0, nullptr, in, out, tiles); break;
        case 6: hipLaunchKernelGGL((fragment_kernel<Bits, Frag::scalar, Sink>), frag_grid, block, 0, nullptr, in, out, tiles); break;
        case 7: hipLaunchKernelGGL((fragment_kernel<Bits, Frag::pair, Sink>), frag_grid, block, 0, nullptr, in, out, tiles); break;
        case 8: hipLaunchKernelGGL((fragment_kernel<Bits, Frag::decode16, Sink>), frag_grid, block, 0, nullptr, in, out, tiles); break;
        case 9: hipLaunchKernelGGL((fragment_kernel<Bits, Frag::pipe, Sink>), frag_grid, block, 0, nullptr, in, out, tiles); break;
        case 10: hipLaunchKernelGGL((fragment_kernel<Bits, Frag::v2, Sink>), frag_grid, block, 0, nullptr, in, out, tiles); break;
        case 11: hipLaunchKernelGGL((lane_tile_kernel<Bits, Sink>), tile_grid, block, 0, nullptr, in, out, tiles); break;
        default: hipLaunchKernelGGL((lane_kernel<Bits, Lane::tiled, Sink>), lane_grid, block, 0, nullptr, in, out, tiles); break;
        }
    }
};
// Checksum a bench (Sink) launch leaves at out[tile]: XOR of the halves one lane (or column 0) decodes.
std::uint16_t sink_sum(int v, const std::uint16_t* tile) {
    std::uint16_t sum = 0;
    if (v <= 5 || v == 12) { for (unsigned p = 0; p < 8; ++p) sum ^= tile[p]; }
    else if (v <= 10) { for (unsigned k = 0; k < 16; ++k) sum ^= tile[strix::core::trellis::Format::position(k, 0)]; }
    else { for (unsigned p = 0; p < 256; ++p) if (kernels::lane_tile_covers(p)) sum ^= tile[p]; }
    return sum;
}
const char* const kNames[] = {"lane.pair", "lane.pair_masked", "lane.half", "lane.half16", "lane.value", "wave.value",
                              "fragment.scalar", "fragment.pair", "fragment.decode16", "fragment.pipe", "fragment.v2", "lane_tile.half", "tiled.decode_pair"};
constexpr int kVariants = 13;
constexpr int kLaneTile = 11;  // covers a subset of positions

bool codebooks() {
    constexpr std::size_t n = kernels::kBookForms * 65536U;
    std::uint16_t* dout = nullptr;
    ok(hipMalloc(&dout, n * 2U), "malloc book");
    hipLaunchKernelGGL(kernels::codebook_kernel, dim3(256), dim3(256), 0, nullptr, dout);
    ok(hipDeviceSynchronize(), "sync");
    std::vector<std::uint16_t> got(n);
    ok(hipMemcpy(got.data(), dout, n * 2U, hipMemcpyDeviceToHost), "download book");
    ok(hipFree(dout), "free");
    std::size_t bad = 0;
    std::vector<std::uint16_t> table(65536U);
    for (std::uint32_t s = 0; s < 65536U; ++s) {
        table[s] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(strix::core::trellis::Mul1::value(s)));
        for (unsigned f = 0; f < kernels::kBookForms; ++f) bad += got[f * 65536U + s] != table[s];
    }
    std::printf("codebook mul1: %u device forms x 65536 states vs host: %zu mismatches, table fnv %016llx\n", kernels::kBookForms, bad,
                static_cast<unsigned long long>(fnv(table.data(), table.size() * 2U)));
    return bad == 0;
}

template <int Bits> bool run(unsigned tiles, int repeats, bool report) {
    const std::size_t words = static_cast<std::size_t>(tiles) * 8U * Bits, halves = static_cast<std::size_t>(tiles) * 256U;
    std::vector<std::uint32_t> packed(words);
    std::uint64_t seed = 0x7472656c6c6973ULL + static_cast<std::uint64_t>(Bits) * 131U + tiles;
    // One tile: all-zero ring; three tiles: all-one ring; otherwise random.
    for (auto& w : packed) w = tiles == 1U ? 0U : tiles == 3U ? 0xffffffffU : static_cast<std::uint32_t>(splitmix(seed));
    std::vector<std::uint16_t> reference(halves), got(halves);
    for (unsigned t = 0; t < tiles; ++t)
        for (unsigned p = 0; p < 256; ++p) {
            const float v = strix::core::trellis::Mul1::value(
                strix::core::trellis::Format::state(packed.data() + static_cast<std::size_t>(t) * 8U * Bits, Bits, p));
            reference[static_cast<std::size_t>(t) * 256U + p] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(v));
        }
    const std::uint64_t want = fnv(reference.data(), halves * 2U);
    std::uint32_t* din = nullptr;
    std::uint16_t* dout = nullptr;
    ok(hipMalloc(&din, words * 4U), "malloc in");
    ok(hipMalloc(&dout, halves * 2U), "malloc out");
    ok(hipMemcpy(din, packed.data(), words * 4U, hipMemcpyHostToDevice), "upload");
    bool pass = true;
    for (int v = 0; v < kVariants; ++v) {
        ok(hipMemset(dout, 0xff, halves * 2U), "memset");
        Suite<Bits>::template launch<false>(v, din, dout, tiles);
        ok(hipDeviceSynchronize(), "sync");
        ok(hipMemcpy(got.data(), dout, halves * 2U, hipMemcpyDeviceToHost), "download");
        std::size_t bad = 0;
        if (v == kLaneTile) {
            for (std::size_t i = 0; i < halves; ++i)
                if (kernels::lane_tile_covers(static_cast<unsigned>(i % 256U)) && got[i] != reference[i]) ++bad;
        } else {
            for (std::size_t i = 0; i < halves; ++i) bad += got[i] != reference[i];
        }
        double best = 0.0;
        if (repeats > 0) {
            // Bench mode: the same decode with one checksum store per tile.
            best = 1e30;
            for (int r = 0; r < repeats + 1; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                Suite<Bits>::template launch<true>(v, din, dout, tiles);
                ok(hipDeviceSynchronize(), "sync");
                if (r > 0) best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            std::vector<std::uint16_t> sums(tiles);
            ok(hipMemcpy(sums.data(), dout, static_cast<std::size_t>(tiles) * 2U, hipMemcpyDeviceToHost), "download sums");
            for (unsigned t = 0; t < tiles; ++t)
                bad += sums[t] != sink_sum(v, reference.data() + static_cast<std::size_t>(kernels::source_tile<true>(t)) * 256U);
        }
        if (bad || report)
            std::printf("K%d tiles %u %-18s %s fnv %016llx mismatches %zu%s", Bits, tiles, kNames[v], bad ? "FAIL" : "ok",
                        static_cast<unsigned long long>(v == kLaneTile ? want : fnv(got.data(), halves * 2U)), bad,
                        repeats > 0 ? "" : "\n");
        if (repeats > 0 && (bad || report)) std::printf(" ms %.3f\n", best);
        pass = pass && bad == 0;
    }
    if (report) std::printf("K%d tiles %u reference fnv %016llx\n", Bits, tiles, static_cast<unsigned long long>(want));
    ok(hipFree(din), "free");
    ok(hipFree(dout), "free");
    return pass;
}
template <int Bits> bool bits(int repeats) {
    bool pass = true;
    for (const unsigned tiles : {1U, 3U, 17U, 255U, 4099U}) pass = run<Bits>(tiles, 0, tiles == 4099U) && pass;
    if (repeats > 0) pass = run<Bits>(262144U, repeats, true) && pass;
    return pass;
}
}  // namespace

int main() {
    const char* env = std::getenv("STRIX_TRELLIS_CODEC_BENCH");
    const int repeats = env ? std::atoi(env) : 0;
    bool pass = codebooks();
    pass = bits<1>(0) && pass; pass = bits<2>(0) && pass;
    pass = bits<3>(repeats) && pass; pass = bits<4>(repeats) && pass; pass = bits<5>(repeats) && pass;
    pass = bits<6>(repeats) && pass; pass = bits<7>(0) && pass; pass = bits<8>(repeats) && pass;
    std::printf(pass ? "PASS: device decoders equal the host reference\n" : "FAIL\n");
    return pass ? 0 : 1;
}
