#include "strix/ds41_expert_tp.hpp"
#include "strix/env_flags.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strix::ds41::expert_tp {
namespace {

constexpr std::uint32_t tail = 128U;  // rank 1's elements of the boundary block (its second half)
constexpr std::uint32_t lanes = 32U;

float butterfly(std::array<float, lanes> v) noexcept {
    for (unsigned d = 16U; d; d /= 2U) {
        std::array<float, lanes> next{};
        for (unsigned l = 0; l < lanes; ++l) next[l] = v[l] + v[l ^ d];
        v = next;
    }
    return v[0];
}

}  // namespace

bool enabled_from_env() {
    static const bool on = [] {
        const char* v = envcfg::text("STRIX_DS41_EXPERT_TP");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

namespace {
bool decode_default = true;
}  // namespace

void set_decode_default(bool on) { decode_default = on; }

bool decode_enabled(std::uint32_t world, std::uint32_t moe_inter, bool stream_experts) {
    static const int env = [] {
        const char* v = envcfg::text("STRIX_DS41_EXPERT_TP");
        return v == nullptr || v[0] == 0 ? -1 : v[0] == '1' ? 1 : 0;
    }();
    if (env >= 0) return env == 1;
    return decode_default && world == ranks && moe_inter == inter && !stream_experts;
}

namespace {
int packed_override = -1;
}  // namespace

void set_packed(int mode) { packed_override = mode; }

bool packed_from_env() {
    if (packed_override >= 0) return packed_override == 1;
    static const bool packed = [] {
        const char* v = envcfg::text("STRIX_DS41_EXPERT_TP_LAYOUT");
        return !(v != nullptr && std::strcmp(v, "dup") == 0);
    }();
    return packed;
}

std::uint32_t inter_index(unsigned rank, std::uint32_t j) {
    if (j >= half || rank >= ranks) throw std::invalid_argument("ds41 expert tp: local index out of range");
    if (rank == 0U) return j;
    return j < half - tail ? half + tail + j : half + (j - (half - tail));
}
std::uint32_t group_index(unsigned rank, std::uint32_t g) {
    if (g >= half_groups || rank >= ranks) throw std::invalid_argument("ds41 expert tp: local group out of range");
    return inter_index(rank, 32U * g) / 32U;
}

void pack_gate_rows(std::span<const std::uint8_t> expert, std::uint64_t row_bytes, unsigned rank, std::span<std::uint8_t> out) {
    if (expert.size() != row_bytes * inter || out.size() != row_bytes * half) throw std::invalid_argument("ds41 expert tp: gate rows");
    // runs of consecutive rows: rank 0 [0, 1152); rank 1 [1280, 2304) then [1152, 1280)
    for (std::uint32_t j = 0; j < half;) {
        const std::uint32_t g = inter_index(rank, j);
        std::uint32_t n = 1;
        while (j + n < half && inter_index(rank, j + n) == g + n) ++n;
        std::memcpy(out.data() + std::uint64_t{j} * row_bytes, expert.data() + std::uint64_t{g} * row_bytes, std::uint64_t{n} * row_bytes);
        j += n;
    }
}

void pack_down_row(std::span<const std::uint8_t> row, unsigned rank, std::span<std::uint8_t> out) {
    constexpr std::uint64_t B = q2_k_block_bytes;
    if (row.size() != inter / 256U * B || out.size() != down_row_bytes || rank >= ranks)
        throw std::invalid_argument("ds41 expert tp: down row");
    if (rank == 0U) {
        std::memcpy(out.data(), row.data(), down_row_bytes);  // blocks 0..4 (block 4: groups 0..3 used)
        return;
    }
    std::memcpy(out.data(), row.data() + 5U * B, 4U * B);  // blocks 5..8
    std::uint8_t* last = out.data() + 4U * B;
    std::memcpy(last, row.data() + 4U * B, B);              // block 4 ...
    std::memcpy(last, row.data() + 4U * B + 8U, 8U);        // ... scales 8..15 over scales 0..7
    std::memcpy(last + 16U, row.data() + 4U * B + 48U, 32U);  // ... codes of groups 4..7 over those of groups 0..3
}

void pack_down_row_packed(std::span<const std::uint8_t> row, unsigned rank, std::span<std::uint8_t> out) {
    constexpr std::uint64_t B = q2_k_block_bytes;
    if (row.size() != inter / 256U * B || out.size() != down_row_bytes_packed || rank >= ranks)
        throw std::invalid_argument("ds41 expert tp: packed down row");
    const std::uint8_t* whole = row.data() + (rank == 0U ? 0U : 5U * B);
    std::memcpy(out.data(), whole, 4U * B);
    const std::uint8_t* b4 = row.data() + 4U * B;
    std::uint8_t* half_block = out.data() + 4U * B;
    std::memcpy(half_block, b4 + 8U * rank, 8U);              // scales 0..7 or 8..15
    std::memcpy(half_block + 8U, b4 + 16U + 32U * rank, 32U);  // codes of groups 0..3 or 4..7
    std::memcpy(half_block + 40U, b4 + 80U, 4U);               // d, dmin
}

Q2kGroup packed_group(std::span<const std::uint8_t> row, std::uint32_t g) {
    if (row.size() < down_row_bytes_packed || g >= half_groups) throw std::invalid_argument("ds41 expert tp: packed group");
    if (g < 32U)
        return q2_k_group(std::span<const std::uint8_t, q2_k_block_bytes>(row.data() + g / 8U * q2_k_block_bytes, q2_k_block_bytes), g % 8U);
    const std::uint8_t* h = row.data() + 4U * q2_k_block_bytes;
    Q2kGroup out;
    const unsigned group = g % 4U, shift = 2U * group;
    for (unsigned l = 0; l < 32U; ++l) out.codes[l] = static_cast<std::uint8_t>((h[8U + l] >> shift) & 3U);
    out.scale[0] = h[2U * group];
    out.scale[1] = h[2U * group + 1U];
    // d and dmin through the unsplit block decoder: build a block with the same d/dmin bytes
    std::array<std::uint8_t, q2_k_block_bytes> block{};
    std::memcpy(block.data() + 80U, h + 40U, 4U);
    const auto ref = q2_k_group(block, 0U);
    out.d = ref.d;
    out.dmin = ref.dmin;
    return out;
}

float product_q2_k_groups(std::span<const std::uint8_t> row, const A8Activation& x, std::uint32_t groups) {
    if (groups == 0U || row.size() < (groups + 7U) / 8U * q2_k_block_bytes || x.codes.size() < std::size_t{groups} * 32U ||
        x.deltas.size() < groups)
        throw std::invalid_argument("ds41 expert tp: K-half product row or activation");
    std::array<float, lanes> acc{};
    for (std::uint32_t g = 0; g < groups; ++g) {
        const auto q = q2_k_group(std::span<const std::uint8_t, q2_k_block_bytes>(row.data() + g / 8U * q2_k_block_bytes, q2_k_block_bytes),
                                  g % 8U);
        std::int32_t a = 0, b = 0;
        for (unsigned h = 0; h < 2U; ++h) {
            std::int32_t qx = 0, sx = 0;
            for (unsigned j = 16U * h; j < 16U * h + 16U; ++j) {
                qx += q.codes[j] * x.codes[g * 32U + j];
                sx += x.codes[g * 32U + j];
            }
            a += static_cast<std::int32_t>(q.scale[h] & 15U) * qx;
            b += static_cast<std::int32_t>(q.scale[h] >> 4U) * sx;
        }
        const float scaled = q.d * static_cast<float>(a);
        const float offset = q.dmin * static_cast<float>(b);
        acc[g % lanes] = acc[g % lanes] + (scaled - offset) * x.deltas[g];
    }
    return butterfly(acc);
}

WindowTp reference_window(const ExpertTensors& t, const std::vector<A8Activation>& x, const WindowRoutes& routes,
                          std::span<const float> weights, std::span<const float> shared, std::uint32_t top_k) {
    const auto slots = weights.size(), rows = slots / top_k;
    if (t.inter != inter || t.gate_up_format != WeightFormat::iq2_xxs || t.down_format != WeightFormat::q2_k)
        throw std::invalid_argument("ds41 expert tp: the identity is declared for IQ2_XXS / Q2_K experts of intermediate 2304");
    if (rows == 0U || rows * top_k != slots || x.size() < rows || shared.size() != rows * t.hidden)
        throw std::invalid_argument("ds41 expert tp: window reference inputs are inconsistent");
    const auto gate_row = t.gate_row_bytes(), down_row = t.down_row_bytes();
    const auto gate_stride = gate_row * t.inter, down_stride = down_row * t.hidden;
    WindowTp out;
    std::vector<std::uint8_t> packed(down_row_bytes);
    for (unsigned p = 0; p < ranks; ++p) {
        out.partial[p].assign(slots * t.hidden, 0.0F);
        for (std::size_t j = 0; j + 1U < routes.begins.size(); ++j) {
            const std::uint64_t expert = routes.experts[j];
            for (auto i = routes.begins[j]; i < routes.begins[j + 1U]; ++i) {
                const auto row = routes.rows[i], slot = routes.slots[i];
                std::vector<float> act(half);
                for (std::uint32_t o = 0; o < half; ++o) {  // local order of the rank's intermediate rows
                    const auto at = expert * gate_stride + std::uint64_t{inter_index(p, o)} * gate_row;
                    const float g = product_iq2_xxs(t.gate.subspan(at, gate_row), x[row], t.hidden);
                    const float u = product_iq2_xxs(t.up.subspan(at, gate_row), x[row], t.hidden);
                    act[o] = swiglu_clamped(g, u, t.limit);
                }
                const auto a8 = quantize_a8(act);
                for (std::uint32_t o = 0; o < t.hidden; ++o) {
                    pack_down_row(t.down.subspan(expert * down_stride + std::uint64_t{o} * down_row, down_row), p, packed);
                    out.partial[p][slot * t.hidden + o] = product_q2_k_groups(packed, a8, half_groups);
                }
            }
        }
        out.sum[p].assign(rows * t.hidden, 0.0F);
        for (std::size_t r = 0; r < rows; ++r)
            for (std::uint32_t o = 0; o < t.hidden; ++o) {
                float s = 0.0F;
                for (std::uint32_t k = 0; k < top_k; ++k) {
                    const std::size_t slot = routes.order[r * top_k + k];
                    s = s + weights[slot] * out.partial[p][slot * t.hidden + o];
                }
                out.sum[p][r * t.hidden + o] = s;
            }
    }
    out.y.assign(rows * t.hidden, 0.0F);
    for (std::size_t i = 0; i < out.y.size(); ++i) out.y[i] = (shared[i] + out.sum[0][i]) + out.sum[1][i];
    return out;
}

}  // namespace strix::ds41::expert_tp
