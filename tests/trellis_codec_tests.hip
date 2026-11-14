// CPU-only gate of the shared trellis codec (core/trellis_codec.hpp, core/trellis_format.hpp). Requires the HIP
// headers/compiler, never a device.
// The same random packed tensors (Bits 1..8; single tiles, odd tile counts, ring wrap in every tile) are decoded
// through the codec's host reference, its codebook forms and its reader geometry tables (Window, Norm, the lane
// window), and through frozen transcriptions of the decoders this codec replaced (the former core WaveTile / group /
// DirectLane / AlignedLane readers, the former moe_core lane and fragment readers, the former Gemma column, lane-tile
// and cooperative readers, and each one's codebook arithmetic). All images must be bit-identical, and their hashes
// must equal the values recorded from the replaced implementations before they were removed.
#include "core/trellis_codec.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Frozen transcriptions of the pre-unification decoders' bit addressing and codebook arithmetic (host form).
// Each function names the declaration it was transcribed from (tree d0a45c4ce). Device-only originals read
// threadIdx / shuffle banks; here the lane and the word bank are explicit arguments, the arithmetic is unchanged.
namespace legacy {
inline std::uint32_t alignbit(std::uint32_t hi, std::uint32_t lo, unsigned s) {  // v_alignbit_b32: ({hi, lo} >> s)[31:0]
    return static_cast<std::uint32_t>(((static_cast<std::uint64_t>(hi) << 32U) | lo) >> (s & 31U));
}
inline unsigned byte_sum(std::uint32_t p) { return (p & 255U) + ((p >> 8U) & 255U) + ((p >> 16U) & 255U) + (p >> 24U); }
inline float f16(std::uint16_t h) { return static_cast<float>(__builtin_bit_cast(_Float16, h)); }
inline std::uint16_t rn16(float v) { return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(v)); }

// ---- src/core/trellis_device.hpp
inline std::uint16_t core_codebook(std::uint32_t state, int cb) {  // codebook_value
    if (cb == 2) return rn16(std::fmaf(static_cast<float>(1024U + byte_sum(state * 0x83dcd12dU)), f16(0x1eee), f16(0xc931)));
    const std::uint32_t mixed = cb == 1 ? state * 0xcbac1fedU : state * 89226354U + 64248484U;
    const std::uint32_t halves = (mixed & 0x8fff8fffU) ^ 0x3b603b60U;
    return rn16(f16(static_cast<std::uint16_t>(halves & 65535U)) + f16(static_cast<std::uint16_t>(halves >> 16U)));
}
inline std::uint32_t core_wave_state(const std::uint32_t* data, int bits, int position) {  // WaveTile::state
    const int words = bits * 8, end = (position + 1) * bits + 256 * bits, start = end - 16;
    const std::uint64_t pair = (static_cast<std::uint64_t>(data[(start / 32) % words]) << 32U) | data[((end - 1) / 32) % words];
    return static_cast<std::uint32_t>(pair >> ((32 - (end & 31)) & 31)) & 65535U;
}
template <int Group> std::uint32_t core_group_state(const std::uint32_t* data, int bits, int first, int j) {  // WaveTile::group + group_states
    const int words = bits * 8, width = 16 + (Group - 1) * bits, end = (first + Group) * bits + 256 * bits, start = end - width;
    const std::uint64_t pair = (static_cast<std::uint64_t>(data[(start / 32) % words]) << 32U) | data[((end - 1) / 32) % words];
    const std::uint64_t window = pair >> ((32 - (end & 31)) & 31);
    return static_cast<std::uint32_t>(window >> ((Group - 1 - j) * bits)) & 65535U;
}
// ---- src/core/trellis_optimized.hpp
inline std::uint32_t core_direct_state(const std::uint32_t* data, int bits, int lane, int j) {  // DirectLane
    const int ring = bits * 8, start = (lane * 8 + 1) * bits + 256 * bits - 16, index = start / 32;
    std::uint32_t words[3];
    for (int i = 0; i < 3; ++i) words[i] = data[(index + i) % ring];
    const int offset = (start & 31) + j * bits, at = offset / 32, shift = offset & 31;
    const std::uint64_t window = (static_cast<std::uint64_t>(words[at]) << 32U) | words[at + (shift > 16)];
    return static_cast<std::uint32_t>(window >> (48 - shift)) & 65535U;
}
struct Aligned { std::uint32_t high, low, tail; };
inline std::uint32_t aligned_state(const Aligned& w, int bits, int j) {  // AlignedLane / SaddrLane / Stream / TileStates / CoopLane ::state<J>
    const int offset = j * bits;
    if (offset <= 16) return (w.high >> (16 - offset)) & 65535U;
    if (offset < 32) return ((w.high << (offset - 16)) | (w.low >> (48 - offset))) & 65535U;
    if (offset <= 48) return (w.low >> (48 - offset)) & 65535U;
    return ((w.low << (offset - 48)) | (w.tail >> (80 - offset))) & 65535U;
}
inline Aligned core_aligned(const std::uint32_t* data, int bits, int lane, bool fast_wrap) {  // AlignedLane ctor
    const int count = bits * 8, start = (lane * 8 + 1) * bits + 256 * bits - 16, shift = start & 31;
    int index = start / 32;
    std::uint32_t a, b, c;
    if (fast_wrap) {
        index -= count;
        a = data[index < 0 ? index + count : index];
        b = data[index + 1 >= count ? index + 1 - count : index + 1];
        c = data[index + 2 >= count ? index + 2 - count : index + 2];
    } else {
        a = data[index % count]; b = data[(index + 1) % count]; c = data[(index + 2) % count];
    }
    return Aligned{(a << shift) | (b >> (32 - shift)), (b << shift) | (c >> (32 - shift)), c << shift};
}
// ---- src/moe_core/exl3_trellis.hpp
inline Aligned moe_tile_states(const std::uint32_t* data, int bits, int lane) {  // LaneGeom + TileStates ctor
    const int count = bits * 8, start = (lane * 8 + 1) * bits + 256 * bits - 16;
    const int index = start / 32 - count, shift = start & 31;
    const int ia = index < 0 ? index + count : index;
    const int ib = index + 1 >= count ? index + 1 - count : index + 1;
    const int ic = index + 2 >= count ? index + 2 - count : index + 2;
    const auto* base = reinterpret_cast<const unsigned char*>(data);
    std::uint32_t a, b, c;
    std::memcpy(&a, base + static_cast<std::uint32_t>(ia) * 4U, 4);
    std::memcpy(&b, base + static_cast<std::uint32_t>(ib) * 4U, 4);
    std::memcpy(&c, base + static_cast<std::uint32_t>(ic) * 4U, 4);
    const unsigned align = 32U - static_cast<unsigned>(shift);
    return Aligned{alignbit(a, b, align), alignbit(b, c, align), c << (32U - align)};
}
inline std::uint16_t moe_pipe_codebook(std::uint32_t state) {  // Exl3Trellis::pipe_product + pipe_codebook
    const std::uint32_t lo = (state & 0xffffffU) * 0xdcd12dU, hi = (state & 0xffffffU) * 0x83U;  // v_mul_u32_u24
    const std::uint32_t product = (hi << 24U) + lo;
    return rn16(std::fmaf(static_cast<float>(byte_sum(product)), 887.0f / 131072.0f, -3.453125f));
}
inline std::uint16_t moe_codebook_v2(std::uint32_t state) {  // Exl3Trellis::codebook_value_v2
    const unsigned sum = byte_sum(state * 0x83dcd12dU);
    return rn16(std::fmaf(__builtin_bit_cast(float, 0x44800000U | (sum << 13U)), f16(0x1eee), f16(0xc931)));
}
// ---- src/gemma4/exl3/device.hip
inline std::uint16_t gemma_native_codebook(std::uint32_t s) {  // native_codebook: separate multiply and add, RN32 then RN16
    volatile float product = static_cast<float>(1024U + byte_sum(s * 0x83dcd12dU)) * f16(0x1eee);
    volatile float value = product + f16(0xc931);
    return rn16(value);
}
inline std::uint16_t gemma_codebook_half(std::uint32_t s) {  // codebook_half / codebook_half16: one FP16 FMA
    const auto in = static_cast<double>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(0x6400U + byte_sum(s * 0x83dcd12dU))));
    // Exact in double (11-bit x 11-bit product + 11-bit addend), then one RN16: the FP16 FMA result.
    const double exact = in * static_cast<double>(f16(0x1eee)) + static_cast<double>(f16(0xc931));
    return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(exact));
}
inline std::uint32_t gemma_column_state(const std::uint32_t* tile, unsigned k, unsigned output, unsigned e) {  // Column<K, High>::load + at
    const bool high = (output & 8U) != 0;
    std::uint32_t words[9];
    for (unsigned j = 0; j <= k; ++j) {
        const int index = static_cast<int>((output & 7U) * k + j) - (high ? 0 : 1);
        const auto wrapped = static_cast<unsigned>(index < 0 ? index + static_cast<int>(8 * k) : index >= static_cast<int>(8 * k) ? index - static_cast<int>(8 * k) : index);
        words[j] = tile[wrapped];
    }
    const auto position = ((e & 6U) << 2U) | (e & 1U) | ((e & 8U) >> 2U);
    const auto start = (high ? 5 * k - 16 : k + 16) + position * k;
    const auto pair = (static_cast<std::uint64_t>(words[start / 32]) << 32U) | words[(start + 15) / 32];
    return static_cast<std::uint32_t>(pair >> ((32U - ((start + 16) & 31U)) & 31U)) & 65535U;
}
inline std::uint32_t gemma_lane_tile_state(const std::uint32_t* tile, unsigned k, unsigned g, unsigned g0, unsigned p) {  // LaneTile<K, G, G0>::load + state<P>
    const unsigned ring = 8 * k, base = g == 8 ? 0 : (g0 * k + ring - 4) % ring, vectors = g == 8 ? 2 * k : g * k / 4 + 1;
    std::uint32_t v[4 * 17];
    for (unsigned i = 0; i < vectors; ++i)
        for (unsigned j = 0; j < 4; ++j) v[4 * i + j] = tile[(base + 4 * i) % ring + j];  // one 16-byte load
    const unsigned start = ((p + 1) * k + 256 * k - 16) % (256 * k), off = start % 32;
    const unsigned first = (start / 32 + ring - base) % ring;
    if (off <= 16) return (v[first] >> (16 - off)) & 65535U;
    const unsigned second = (start / 32 + 1 + ring - base) % ring;
    return ((v[first] << (off - 16)) | (v[second] >> (48 - off))) & 65535U;
}
inline Aligned gemma_coop(const std::uint32_t* data, unsigned k, unsigned lane) {  // CoopLane::load / CoopWords + align
    const unsigned count = 8 * k, start = (lane * 8 + 1) * k + 256 * k - 16, index = start / 32, shift = start & 31U;
    const auto a = data[index % count], b = data[(index + 1) % count], c = data[(index + 2) % count];
    return Aligned{(a << shift) | (b >> (32 - shift)), (b << shift) | (c >> (32 - shift)), c << shift};
}
}  // namespace legacy

namespace {
struct Fnv {
    std::uint64_t h = 1469598103934665603ULL;
    void add(const void* p, std::size_t n) {
        const auto* b = static_cast<const unsigned char*>(p);
        for (std::size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    }
};
std::uint64_t splitmix(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}
std::uint16_t lane_bits(strix::core::trellis::Half2 v, int lane) {
    const auto word = __builtin_bit_cast(std::uint32_t, v);
    return static_cast<std::uint16_t>(lane == 0 ? (word & 65535U) : (word >> 16U));
}
enum Impl {
    kReference, kReferenceBytes, kGemmaColumn, kGemmaLaneTile8, kGemmaLaneTile4, kGemmaCoop,
    kCoreWave, kCoreGroup2, kCoreDirect, kCoreAligned, kCoreAlignedFast,
    kMoeLanePair, kMoeLaneScalar, kMoeNormScalar, kMoeNormPair, kMoeNormPipe, kMoeNormV2, kMoeWindow, kImpls
};
const char* const kNames[kImpls] = {
    "codec.reference_bits", "codec.state_bytes", "gemma.Column+native_codebook", "gemma.LaneTile<8>+codebook_half", "gemma.LaneTile<4>+codebook_half",
    "gemma.CoopLane+codebook_half", "core.WaveTile+codebook_value", "core.group<2>+codebook_value", "core.DirectLane+mul1_half",
    "core.AlignedLane+mul1_half2", "core.AlignedLane<fast>+mul1_half2", "codec.lane+Mul1::pair", "codec.lane+Mul1::value",
    "codec.Norm+ScalarDecode8", "codec.Norm+PairDecode8", "codec.Norm+pipe", "codec.Norm+value_v2", "codec.Window+Mul1::value"};

template <int Bits> bool tensor(unsigned inputs, unsigned outputs, std::uint64_t seed, Fnv (&h16)[kImpls], Fnv (&h32)[kImpls]) {
    using namespace strix::core::trellis;
    using NW = Norm<Bits>;
    using W = Window<Bits>;
    const unsigned tiles_k = inputs / 16U, tiles_n = outputs / 16U, kw = 8U * Bits;
    std::vector<std::uint32_t> trellis(static_cast<std::size_t>(tiles_k) * tiles_n * kw);
    for (auto& w : trellis) w = static_cast<std::uint32_t>(splitmix(seed));
    std::vector<std::uint8_t> bytes(trellis.size() * 4U);
    std::memcpy(bytes.data(), trellis.data(), bytes.size());
    unsigned inverse_e[256], inverse_c[256];
    for (unsigned e = 0; e < 16; ++e)
        for (unsigned c = 0; c < 16; ++c) {
            const unsigned p = Format::position(e, c);
            inverse_e[p] = e; inverse_c[p] = c;
        }
    std::vector<std::uint16_t> image[kImpls];
    for (auto& i : image) i.assign(static_cast<std::size_t>(inputs) * outputs, 0xffffU);
    for (unsigned tk = 0; tk < tiles_k; ++tk)
        for (unsigned tn = 0; tn < tiles_n; ++tn) {
            const std::uint32_t* tile = trellis.data() + (static_cast<std::size_t>(tk) * tiles_n + tn) * kw;
            const auto at = [&](unsigned e, unsigned c) { return static_cast<std::size_t>(tk * 16U + e) * outputs + tn * 16U + c; };
            for (unsigned e = 0; e < 16; ++e)
                for (unsigned c = 0; c < 16; ++c) {
                    const unsigned p = Format::position(e, c);
                    image[kReference][at(e, c)] = reference_bits<Mul1>(trellis.data(), Bits, outputs, tk * 16U + e, tn * 16U + c);
                    image[kReferenceBytes][at(e, c)] = legacy::rn16(Mul1::value(Format::state_bytes(
                        bytes.data() + (static_cast<std::size_t>(tk) * tiles_n + tn) * kw * 4U, Bits, p)));
                    // The Gemma column window starts 5K - 16 bits into its first word: defined for K >= 4 only (Gemma packs are 4/5/6/8).
                    image[kGemmaColumn][at(e, c)] = Bits >= 4 ? legacy::gemma_native_codebook(legacy::gemma_column_state(tile, Bits, c, e))
                                                              : image[kReference][at(e, c)];
                    image[kGemmaLaneTile8][at(e, c)] = legacy::gemma_codebook_half(legacy::gemma_lane_tile_state(tile, Bits, 8, 0, p));
                    image[kGemmaLaneTile4][at(e, c)] =
                        legacy::gemma_codebook_half(legacy::gemma_lane_tile_state(tile, Bits, 4, (c & 7U) < 4 ? 0 : 4, p));
                    image[kCoreWave][at(e, c)] = legacy::core_codebook(legacy::core_wave_state(tile, Bits, static_cast<int>(p)), 2);
                    image[kCoreGroup2][at(e, c)] = legacy::core_codebook(
                        legacy::core_group_state<2>(tile, Bits, static_cast<int>(p & ~1U), static_cast<int>(p & 1U)), 2);
                }
            // Lane readers: lane t owns positions 8t .. 8t + 7.
            for (unsigned t = 0; t < 32; ++t) {
                const auto coop = legacy::gemma_coop(tile, Bits, t);
                const auto aligned = legacy::core_aligned(tile, Bits, static_cast<int>(t), false);
                const auto fast = legacy::core_aligned(tile, Bits, static_cast<int>(t), true);
                const auto moe = legacy::moe_tile_states(tile, Bits, static_cast<int>(t));
                std::uint32_t st[8];
                for (int j = 0; j < 8; ++j) st[j] = legacy::aligned_state(moe, Bits, j);
                for (int j = 0; j < 8; j += 2) {
                    const Half2 pair = mul1_pair(st[j], st[j + 1]);
                    for (int i = 0; i < 2; ++i) {
                        const unsigned p = t * 8U + static_cast<unsigned>(j + i);
                        const std::size_t o = at(inverse_e[p], inverse_c[p]);
                        image[kMoeLanePair][o] = lane_bits(pair, i);
                        image[kMoeLaneScalar][o] = legacy::rn16(Codec<Bits>::codebook_value(st[j + i]));
                        image[kGemmaCoop][o] = legacy::gemma_codebook_half(legacy::aligned_state(coop, Bits, j + i));
                        // mul1_half / mul1_half2 are one FP16 FMA of (0x6400 + sum): the same operation as codebook_half.
                        image[kCoreDirect][o] = legacy::gemma_codebook_half(legacy::core_direct_state(tile, Bits, static_cast<int>(t), j + i));
                        image[kCoreAligned][o] = legacy::gemma_codebook_half(legacy::aligned_state(aligned, Bits, j + i));
                        image[kCoreAlignedFast][o] = legacy::gemma_codebook_half(legacy::aligned_state(fast, Bits, j + i));
                    }
                }
            }
            // WMMA fragment readers: lane c = output column, element r = input coordinate.
            for (unsigned c = 0; c < 16; ++c) {
                const int base = 32 * static_cast<int>(c & 7U) + 4 * static_cast<int>(c >> 3U);
                const int S = (base + 1) * Bits - 16 + 256 * Bits - 32;
                const int d = S & 31, wb = (S >> 5) - (d == 0 ? 1 : 0);
                const unsigned sh = static_cast<unsigned>((32 - d) & 31);
                std::uint32_t lc[NW::kNL], nrm[NW::kNL];
                for (int i = 0; i < NW::kNL; ++i) lc[i] = tile[static_cast<unsigned>(wb + i) % kw];
                for (int i = 0; i < NW::kNL - 1; ++i) nrm[i] = legacy::alignbit(lc[i], lc[i + 1], sh);
                std::uint32_t st[16];
                for (int r = 0; r < 16; ++r) st[r] = legacy::alignbit(nrm[NW::g(r) - 1], nrm[NW::g(r)], NW::shift(r)) & 65535U;
                for (int half = 0; half < 2; ++half) {
                    std::uint32_t s8[8];
                    Half2 packed[4], scalar[4];
                    for (int i = 0; i < 8; ++i) s8[i] = st[half * 8 + i];
                    PairDecode8<Bits>::decode8(s8, packed);
                    ScalarDecode8<Bits>::decode8(s8, scalar);
                    for (int i = 0; i < 8; ++i) {
                        const std::size_t o = at(static_cast<unsigned>(half * 8 + i), c);
                        image[kMoeNormPair][o] = lane_bits(packed[i / 2], i % 2);
                        image[kMoeNormScalar][o] = lane_bits(scalar[i / 2], i % 2);
                        image[kMoeNormPipe][o] = legacy::moe_pipe_codebook(s8[i]);
                        image[kMoeNormV2][o] = legacy::moe_codebook_v2(s8[i]);
                    }
                }
                // v2 window (grouped_core.hpp load_window + decode_fragment).
                const int w0 = (((base + 1) * Bits - 16 + 256 * Bits) >> 5) - 1, h = static_cast<int>(c >> 3U);
                std::uint32_t w[W::kNW];
                for (int i = 0; i < W::kNW; ++i) {
                    int idx = w0 + i;
                    while (idx >= W::kWords) idx -= W::kWords;
                    w[i] = tile[idx];
                }
                for (int r = 0; r < 16; ++r) {
                    const auto state = legacy::alignbit(w[W::last_word(h, r) - 1], w[W::last_word(h, r)], W::shift(h, r)) & 65535U;
                    image[kMoeWindow][at(static_cast<unsigned>(r), c)] = legacy::rn16(Codec<Bits>::codebook_value(state));
                }
            }
        }
    bool ok = true;
    for (int i = 0; i < kImpls; ++i) {
        std::size_t bad = 0, first = 0;
        for (std::size_t j = 0; j < image[i].size(); ++j)
            if (image[i][j] != image[kReference][j] && bad++ == 0) first = j;
        if (bad) {
            ok = false;
            std::printf("  DIFF K%d I%u O%u %s: %zu of %zu differ, first k=%zu n=%zu got=%04x ref=%04x\n", Bits, inputs, outputs, kNames[i],
                        bad, image[i].size(), first / outputs, first % outputs, image[i][first], image[kReference][first]);
        }
        h16[i].add(image[i].data(), image[i].size() * 2U);
        for (const auto v : image[i]) {
            const float f = legacy::f16(v);
            h32[i].add(&f, 4);
        }
    }
    return ok;
}

bool codebooks() {
    using namespace strix::core::trellis;
    std::size_t bad = 0;
    Fnv table;
    for (std::uint32_t s = 0; s < 65536U; ++s) {
        const std::uint16_t ref = legacy::rn16(Mul1::value(s));
        const std::uint16_t all[] = {legacy::core_codebook(s, 2), legacy::gemma_native_codebook(s), legacy::gemma_codebook_half(s),
                                     legacy::rn16(codebook_value(s, 2)), lane_bits(mul1_pair(s, s ^ 0x5a5aU), 0),
                                     lane_bits(mul1_pair(s ^ 0xa5a5U, s), 1), legacy::moe_pipe_codebook(s), legacy::moe_codebook_v2(s)};
        for (const auto v : all) bad += v != ref;
        table.add(&ref, 2);
    }
    std::printf("codebook mul1: 65536 states x 8 formulations vs Mul1::value: %zu mismatches, table fnv %016llx\n", bad,
                static_cast<unsigned long long>(table.h));
    return bad == 0;
}

template <int Bits> bool bits(Fnv (&total16)[kImpls], Fnv (&total32)[kImpls]) {
    static constexpr unsigned shapes[][2] = {{16, 16}, {48, 80}, {144, 48}, {128, 128}, {256, 144}, {16, 272}, {368, 16}};
    Fnv h16[kImpls], h32[kImpls];
    bool ok = true;
    std::uint64_t seed = 0x7265'6c6c'6973ULL + static_cast<std::uint64_t>(Bits);
    for (const auto& s : shapes) ok = tensor<Bits>(s[0], s[1], seed += 977, h16, h32) && ok;
    bool same = true;
    for (int i = 1; i < kImpls; ++i) same = same && h16[i].h == h16[0].h && h32[i].h == h32[0].h;
    // Hashes of the same tensors decoded by the three replaced implementations (recorded at d0a45c4ce).
    static constexpr std::uint64_t golden16[8] = {0xa56ffddae5327144ULL, 0xa85d585aa1162ac9ULL, 0xe0a34a0fadbc9c6bULL, 0x9d372912e771b993ULL,
                                                       0x5b7e8a0e6da6304aULL, 0x2b7ed6673f3cf41cULL, 0xc2c56953c6b21ec5ULL, 0x64126014431cb605ULL};
    static constexpr std::uint64_t golden32[8] = {0x95c8f0900d7155fcULL, 0x30efa9e030e2d9ddULL, 0x12146503f1a0c687ULL, 0x0e32944c672fe086ULL,
                                                       0x49e25135ea3a2583ULL, 0x49432395c80093b9ULL, 0x99cf2cd084246d3aULL, 0xa1ab2b20451a11feULL};
    const bool recorded = h16[0].h == golden16[Bits - 1] && h32[0].h == golden32[Bits - 1];
    std::printf("K%d: %d decoders, fp16 %016llx fp32 %016llx %s%s\n", Bits, static_cast<int>(kImpls),
                static_cast<unsigned long long>(h16[0].h), static_cast<unsigned long long>(h32[0].h), same && ok ? "ALL EQUAL" : "DIFFER",
                recorded ? "" : " (not the recorded hash)");
    ok = ok && recorded;
    for (int i = 0; i < kImpls; ++i) {
        if (h16[i].h != h16[0].h) std::printf("  %s fp16 %016llx\n", kNames[i], static_cast<unsigned long long>(h16[i].h));
        total16[i].add(&h16[i].h, 8); total32[i].add(&h32[i].h, 8);
    }
    return same && ok;
}
}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Fnv total16[kImpls], total32[kImpls];
    bool ok = codebooks();
    ok = bits<1>(total16, total32) && ok; ok = bits<2>(total16, total32) && ok; ok = bits<3>(total16, total32) && ok;
    ok = bits<4>(total16, total32) && ok; ok = bits<5>(total16, total32) && ok; ok = bits<6>(total16, total32) && ok;
    ok = bits<7>(total16, total32) && ok; ok = bits<8>(total16, total32) && ok;
    for (int i = 0; i < kImpls; ++i)
        std::printf("%-36s fp16 %016llx fp32 %016llx\n", kNames[i], static_cast<unsigned long long>(total16[i].h),
                    static_cast<unsigned long long>(total32[i].h));
    std::printf(ok ? "PASS: codec host reference, tables and replaced decoders are bitwise interchangeable\n" : "FAIL\n");
    return ok ? 0 : 1;
}
