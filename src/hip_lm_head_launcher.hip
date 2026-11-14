#include "strix/env_flags.hpp"
#include "strix/hip_qwen_dpp_reduce.hpp"
#include "strix/qpf2k_defaults.hpp"
#include "strix/hip_mtp_native_validation.hpp"
#include "strix/hip_ggml_weight_formats.hpp"
#include "strix/known_window_cohort.hpp"
#ifdef STRIX_BUILD_GDN_LM_WAVE8_DIAGNOSTIC
#include "strix/hip_gdn_lm_wave8_diagnostic.hpp"
#endif
#include "strix/hip_launch_context.hpp"
#include "strix/hip_lm_head_launcher.hpp"
#include "strix/known_window_rows_binding.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/hip_argmax_core.hpp"
#include "strix/q8_projection_organization.hpp"

#include "strix/dispatch_resources.hpp"
#include "strix/q6p_registry.hpp"
#include "hip_q6p_weights.hpp"

#include <cstdlib>
#include <string_view>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <type_traits>

namespace strix {
namespace {

constexpr std::uint32_t kThreads = 256;
constexpr std::uint64_t kLauncherTag =
    0x4c4d484541440001ULL;
constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

struct Q8Block {
    __half scale;
    std::int8_t values[32];
};

struct IQ4NLBlock {
    __half d;
    std::uint8_t qs[16];
};

struct IQ4XSBlock {
    __half d;
    std::uint16_t scales_h;
    std::uint8_t scales_l[4];
    std::uint8_t qs[128];
};

static_assert(sizeof(Q8Block) == 34);
static_assert(sizeof(IQ4NLBlock) == 18);
static_assert(sizeof(IQ4XSBlock) == 136);

__device__ __constant__
std::int8_t iq4_codebook[16] = {
    -127, -104, -83, -65,
    -49, -35, -22, -10,
       1,   13,  25,  38,
      53,   69,  89, 113,
};

enum class WeightKind {
    f32,
    f16,
    bf16,
    q8_0,
    iq4_nl,
    iq4_xs,
#define STRIX_GGML_KIND_ENUM(kind, type) kind,
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_ENUM) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_ENUM)
#undef STRIX_GGML_KIND_ENUM
};

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

void check_hip(
    hipError_t error,
    const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            ": " +
            hipGetErrorString(error));
    }
}

void validate_config(
    const HipLmHeadLauncherConfig& config) {
    q8_projection_organization::validate_modes(config.virtual_q8_wave,
        config.prefetch_q8_wave, config.wave8_q8_wave,
        config.input == HipActivationStorage::bf16, config.hidden_size, config.vocab_size);
    if (config.hidden_size == 0 ||
        config.vocab_size == 0) {
        throw std::invalid_argument(
            "HIP LM-head geometry must be non-zero");
    }
    if (config.input !=
            HipActivationStorage::bf16 &&
        config.input !=
            HipActivationStorage::fp32) {
        throw std::invalid_argument(
            "HIP LM-head input storage is unsupported");
    }
}

const char* storage_name(
    HipActivationStorage storage) noexcept {
    switch (storage) {
    case HipActivationStorage::bf16:
        return "bf16";
    case HipActivationStorage::fp32:
        return "fp32";
    }
    return "unknown";
}

WeightKind weight_kind(
    std::uint32_t ggml_type) {
    switch (ggml_type) {
    case 0: return WeightKind::f32;
    case 1: return WeightKind::f16;
    case 8: return WeightKind::q8_0;
    case 20: return WeightKind::iq4_nl;
    case 23: return WeightKind::iq4_xs;
#define STRIX_GGML_KIND_MAP(kind, type) case type: return WeightKind::kind;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_MAP) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_MAP)
#undef STRIX_GGML_KIND_MAP
    case 30: return WeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP LM head supports F32, F16, BF16, Q8_0, IQ4_NL and IQ4_XS");
    }
}

__device__ float bf16_to_float(
    std::uint16_t bits) {
    return __uint_as_float(
        static_cast<std::uint32_t>(
            bits) << 16U);
}

__device__ float load_activation(
    const float* value) {
    return *value;
}

__device__ float load_activation(
    const std::uint16_t* value) {
    return bf16_to_float(*value);
}

__device__ std::uint8_t iq4_local_scale(
    const IQ4XSBlock& block,
    std::uint32_t subblock) {
    const auto low_byte =
        block.scales_l[subblock / 2U];
    const auto low =
        static_cast<std::uint8_t>(
            (low_byte >>
             (4U * (subblock % 2U))) &
            0x0fU);
    const auto high =
        static_cast<std::uint8_t>(
            (block.scales_h >>
             (2U * subblock)) &
            0x03U);
    return static_cast<std::uint8_t>(
        low |
        static_cast<std::uint8_t>(
            high << 4U));
}

template <WeightKind Kind>
__device__ float load_weight(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature);

template <>
__device__ float load_weight<WeightKind::f32>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const float*>(weights);
    return values[
        static_cast<std::uint64_t>(row) *
            row_width +
        feature];
}

template <>
__device__ float load_weight<WeightKind::f16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const __half*>(weights);
    return __half2float(
        values[
            static_cast<std::uint64_t>(row) *
                row_width +
            feature]);
}

template <>
__device__ float load_weight<WeightKind::bf16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const std::uint16_t*>(weights);
    return bf16_to_float(
        values[
            static_cast<std::uint64_t>(row) *
                row_width +
            feature]);
}

template <>
__device__ float load_weight<WeightKind::q8_0>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const Q8Block*>(weights);
    const auto blocks_per_row =
        row_width / 32U;
    const auto& block =
        blocks[
            static_cast<std::uint64_t>(row) *
                blocks_per_row +
            feature / 32U];
    return __half2float(block.scale) *
           static_cast<float>(
               block.values[
                   feature % 32U]);
}

template <>
__device__ float load_weight<WeightKind::iq4_nl>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const IQ4NLBlock*>(weights);
    const auto blocks_per_row =
        row_width / 32U;
    const auto& block =
        blocks[
            static_cast<std::uint64_t>(row) *
                blocks_per_row +
            feature / 32U];
    const auto within =
        feature % 32U;
    const auto packed =
        block.qs[within % 16U];
    const auto code =
        within < 16U
            ? static_cast<std::uint8_t>(
                  packed & 0x0fU)
            : static_cast<std::uint8_t>(
                  packed >> 4U);
    return __half2float(block.d) *
           static_cast<float>(
               iq4_codebook[code]);
}

template <>
__device__ float load_weight<WeightKind::iq4_xs>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const IQ4XSBlock*>(weights);
    const auto blocks_per_row =
        row_width / 256U;
    const auto& block =
        blocks[
            static_cast<std::uint64_t>(row) *
                blocks_per_row +
            feature / 256U];

    const auto within =
        feature % 256U;
    const auto subblock =
        within / 32U;
    const auto suboffset =
        within % 32U;
    const auto packed =
        block.qs[
            subblock * 16U +
            suboffset % 16U];
    const auto code =
        suboffset < 16U
            ? static_cast<std::uint8_t>(
                  packed & 0x0fU)
            : static_cast<std::uint8_t>(
                  packed >> 4U);
    const int local =
        static_cast<int>(
            iq4_local_scale(
                block,
                subblock)) -
        32;

    return __half2float(block.d) *
           static_cast<float>(local) *
           static_cast<float>(
               iq4_codebook[code]);
}

#define STRIX_GGML_KIND_LOAD(kind, type)                                                          \
    template <>                                                                                   \
    __device__ float load_weight<WeightKind::kind>(                                               \
        const void* weights, std::uint32_t row_width, std::uint32_t row, std::uint32_t feature) { \
        return hip_ggml::element<type>(weights, row_width, row, feature);                        \
    }
STRIX_GGML_KINDS_K(STRIX_GGML_KIND_LOAD) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_LOAD)
#undef STRIX_GGML_KIND_LOAD

std::uint32_t ggml_type_of(WeightKind kind) {
    switch (kind) {
#define STRIX_GGML_KIND_TYPE(k, type) case WeightKind::k: return type;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_TYPE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_TYPE)
#undef STRIX_GGML_KIND_TYPE
    default: return 0;
    }
}

std::uint64_t required_weight_bytes(
    WeightKind kind,
    std::uint32_t row_width,
    std::uint64_t rows) {
    switch (kind) {
    case WeightKind::f32:
        return
            static_cast<std::uint64_t>(
                row_width) *
            rows * 4ULL;
    case WeightKind::f16:
    case WeightKind::bf16:
        return
            static_cast<std::uint64_t>(
                row_width) *
            rows * 2ULL;
    case WeightKind::q8_0:
        if (row_width % 32U != 0) {
            throw std::invalid_argument(
                "Q8_0 LM-head row width must be divisible by 32");
        }
        return
            static_cast<std::uint64_t>(
                row_width / 32U) *
            rows *
            sizeof(Q8Block);
    case WeightKind::iq4_nl:
        if (row_width % 32U != 0) {
            throw std::invalid_argument(
                "IQ4_NL LM-head row width must be divisible by 32");
        }
        return
            static_cast<std::uint64_t>(
                row_width / 32U) *
            rows *
            sizeof(IQ4NLBlock);
    case WeightKind::iq4_xs:
        if (row_width % 256U != 0) {
            throw std::invalid_argument(
                "IQ4_XS LM-head row width must be divisible by 256");
        }
        return
            static_cast<std::uint64_t>(
                row_width / 256U) *
            rows *
            sizeof(IQ4XSBlock);
    default: {
        const auto g = hip_ggml::geometry(ggml_type_of(kind));
        if (g.elements == 0 || row_width % g.elements != 0) {
            throw std::invalid_argument("weight width is not a multiple of its GGUF block");
        }
        return static_cast<std::uint64_t>(row_width / g.elements) * rows * g.bytes;
    }
    }
    throw std::logic_error(
        "unreachable LM-head weight kind");
}

template <
    typename Input,
    WeightKind Kind>
__global__ void lm_head_kernel(
    const Input* hidden,
    const void* weights,
    std::uint32_t hidden_size,
    std::uint32_t vocab_size,
    float* logits) {
    const auto vocab_row =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (vocab_row >= vocab_size) {
        return;
    }

    __shared__ float scratch[kThreads];
    float sum = 0.0F;
    for (std::uint32_t feature =
             static_cast<std::uint32_t>(
                 threadIdx.x);
         feature < hidden_size;
         feature += kThreads) {
        sum +=
            load_activation(
                &hidden[feature]) *
            load_weight<Kind>(
                weights,
                hidden_size,
                vocab_row,
                feature);
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] +=
                scratch[
                    threadIdx.x +
                    stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        logits[vocab_row] =
            scratch[0];
    }
}

// Emulate all 256 original leaves and their tree inside one 32-lane wave.
// Local register reduction MUST follow the five cross-lane stages.
template <typename Input>
__global__ __launch_bounds__(32) void virtual_q8_lm_head_kernel(
    const Input* hidden, const void* weights, std::uint32_t hidden_size,
    std::uint32_t vocab_size, float* logits) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x);
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    if (row >= vocab_size) return;
    float partial[8] = {};
    for (std::uint64_t chunk = 0; chunk < hidden_size; chunk += kThreads) {
#pragma unroll
        for (std::uint32_t local = 0; local < 8; ++local) {
            const auto feature = chunk + lane * 8ULL + local;
            if (feature < hidden_size)
                partial[local] += load_activation(&hidden[feature]) *
                    load_weight<WeightKind::q8_0>(weights, hidden_size, row,
                        static_cast<std::uint32_t>(feature));
        }
    }
    for (std::uint32_t distance = 16; distance > 0; distance >>= 1U) {
#pragma unroll
        for (std::uint32_t local = 0; local < 8; ++local) {
            const float partner = strix::qwen_dpp::down_tree32(partial[local], distance);
            if (lane < distance) partial[local] += partner;
        }
    }
    if (lane == 0) {
        partial[0] += partial[4];
        partial[1] += partial[5];
        partial[2] += partial[6];
        partial[3] += partial[7];
        partial[0] += partial[2];
        partial[1] += partial[3];
        partial[0] += partial[1];
        logits[row] = partial[0];
    }
}

// Project-owned GDN prefetch8 load body, preserving the LM coefficient/leaf DAG.
template <typename Input, unsigned Waves>
__global__ __launch_bounds__(32U * Waves)
void prefetch8_wave_q8_lm_head_kernel(
    const Input* input, const void* weights,
    std::uint32_t input_width, std::uint32_t output_width, float* output) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x) * Waves + static_cast<std::uint32_t>(threadIdx.x) / 32U;
    if (row >= output_width) return;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) % 32U;
    float partial[8]{};
    const auto* blocks = static_cast<const Q8Block*>(weights) +
        static_cast<std::uint64_t>(row) * (input_width / 32U);
    for (std::uint64_t chunk = 0; chunk < input_width; chunk += 256U) {
        const auto base = chunk + 8U * lane;
        // The prelaunch Q8 width%32 gate proves all eight leaves are valid
        // together and remain inside one block. No tail load is speculative.
        if (base < input_width) {
            const auto feature = static_cast<std::uint32_t>(base);
            const auto& block = blocks[feature / 32U];
            Input bits[8];
            std::int8_t codes[8];
#pragma unroll
            for (unsigned local = 0; local < 8; ++local) {
                bits[local] = input[feature + local];
                codes[local] = block.values[feature % 32U + local];
            }
            const float scale = __half2float(block.scale);
#pragma unroll
            for (unsigned local = 0; local < 8; ++local)
                partial[local] += load_activation(&bits[local]) *
                    (scale * static_cast<float>(codes[local]));
        }
    }
#pragma unroll
    for (unsigned distance = 16; distance > 0; distance /= 2) {
#pragma unroll
        for (unsigned local = 0; local < 8; ++local) {
            const auto partner = strix::qwen_dpp::down_tree32(partial[local], distance);
            if (lane < distance) partial[local] += partner;
        }
    }
    if (lane == 0) {
        partial[0] += partial[4]; partial[1] += partial[5];
        partial[2] += partial[6]; partial[3] += partial[7];
        partial[0] += partial[2]; partial[1] += partial[3];
        partial[0] += partial[1];
        output[row] = partial[0];
    }
}

struct LmHeadRowsViews {
    const std::uint16_t* input[lm_head_cohort_max_rows];
    float* output[lm_head_cohort_max_rows];
};

// Wave32 value of lane + Distance for the low lanes that consume it.
template <unsigned Distance>
__device__ float lane_down(float value) {
    if constexpr (Distance == 16) {
        const auto bits = __float_as_uint(value);
        return __uint_as_float(__builtin_amdgcn_permlanex16(bits, bits, 0x76543210U, 0xfedcba98U, false, false));
    }
    const auto bits = __float_as_int(value);
    if constexpr (Distance != 16)
        return __int_as_float(__builtin_amdgcn_mov_dpp(bits, 0x100 + static_cast<int>(Distance), 0xf, 0xf, true));
}

// prefetch8 body with one accumulator set per window row: each Q8 block and
// its scale*code products are read once, then every row repeats the exact
// R1 leaf, lane-tree and local-tree DAG of prefetch8_wave_q8_lm_head_kernel.
// Groups > 1 stages all rows' BF16 inputs in LDS once per block, which then
// covers Groups consecutive Waves-row tiles; it changes no arithmetic.
// Q6: weights is the Q6P32 copy of the Q8_0 head; it yields the same codes and
// scales (decoded where the Q8_0 path unpacks its prefetched codes).
using LeanGreedy = strix::argmax::Partial;
// Greedy (staged form, one block per hip_greedy_rows_partial_logits vocabulary rows): lane 0 of
// each wave also folds its logits into one greedy-rows partial record per window row, so the
// separate partials launch (a second read of every logit) is not needed. The records hold the
// same logits and the reduce is a total order, so the finish reply is unchanged.
struct LmHeadRowsRecords { LeanGreedy* row[16]; };
template <unsigned Waves, unsigned Rows, unsigned Groups, bool Q6, bool Greedy>
__device__ __forceinline__ void lm_head_rows_body(uint4* staged, const LmHeadRowsViews& views, const void* weights,
    std::uint32_t input_width, std::uint32_t output_width, const LmHeadRowsRecords* records) {
    const auto* lds = reinterpret_cast<const std::uint16_t*>(staged);
    [[maybe_unused]] auto* best = reinterpret_cast<LeanGreedy*>(staged + Rows * (input_width / 8U));
    if constexpr (Groups > 1) {
        const auto words = input_width / 8U;
        for (auto i = static_cast<std::uint32_t>(threadIdx.x); i < Rows * words; i += 32U * Waves)
            staged[i] = reinterpret_cast<const uint4*>(views.input[i / words])[i % words];
        if constexpr (Greedy) {
            if (threadIdx.x < Waves * Rows) best[threadIdx.x] = LeanGreedy{0.0F, 0U, 0U, 0U};
        }
        __syncthreads();
    }
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) % 32U;
    for (unsigned group = 0; group < Groups; ++group) {
    const auto row = (static_cast<std::uint32_t>(blockIdx.x) * Groups + group) * Waves +
        static_cast<std::uint32_t>(threadIdx.x) / 32U;
    if (row >= output_width) { if constexpr (Greedy) break; else return; }
    float partial[Rows][8]{};
    const auto* blocks = static_cast<const Q8Block*>(weights) +
        static_cast<std::uint64_t>(row) * (input_width / 32U);
    // One chunk of Q8 codes/scale is prefetched ahead; lanes past the width
    // stop exactly where the R1 base < width guard stops accumulating.
    std::uint64_t next_codes = 0;
    __half next_scale{};
    q6p::Group next_group{};
    const auto q6_row = q6p::uniform(q6p::row_of(static_cast<const unsigned char*>(weights), input_width, row));
    const auto fetch = [&](std::uint32_t feature) {
        if constexpr (Q6) {
            next_group = q6p::load_group(q6_row, feature);
        } else {
            const auto& block = blocks[feature / 32U];
            __builtin_memcpy(&next_codes, block.values + feature % 32U, 8U);
            next_scale = block.scale;
        }
    };
    if (8U * lane < input_width) fetch(8U * lane);
    for (std::uint64_t chunk = 0; chunk < input_width; chunk += 256U) {
        const auto base = chunk + 8U * lane;
        if (base >= input_width) break;
        const auto feature = static_cast<std::uint32_t>(base);
        if constexpr (Q6) {
            next_codes = q6p::codes8(next_group.qs, next_group.qh, q6p::shift_of(feature));
            next_scale = __ushort_as_half(static_cast<unsigned short>(next_group.scale));
        }
        const auto packed = next_codes;
        const float scale = __half2float(next_scale);
        if (base + 256U < input_width) fetch(feature + 256U);
        float weight[8];
#pragma unroll
        for (unsigned local = 0; local < 8; ++local)
            weight[local] = scale * static_cast<float>(static_cast<std::int8_t>(packed >> (8U * local)));
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) {
            const auto* input = Groups > 1 ? lds + static_cast<std::uint64_t>(r) * input_width : views.input[r];
            std::uint16_t bits[8];
#pragma unroll
            for (unsigned local = 0; local < 8; ++local) bits[local] = input[feature + local];
#pragma unroll
            for (unsigned local = 0; local < 8; ++local)
                partial[r][local] += load_activation(&bits[local]) * weight[local];
        }
    }
    // Same lane tree as the R1 __shfl_down stages: lane i < d adds lane i + d.
    // Lanes >= d are never read again, so they may add unconditionally.
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r) {
#pragma unroll
        for (unsigned local = 0; local < 8; ++local) {
            auto& v = partial[r][local];
            v += lane_down<16>(v); v += lane_down<8>(v); v += lane_down<4>(v);
            v += lane_down<2>(v); v += lane_down<1>(v);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) {
            auto* p = partial[r];
            p[0] += p[4]; p[1] += p[5]; p[2] += p[6]; p[3] += p[7];
            p[0] += p[2]; p[1] += p[3];
            p[0] += p[1];
            views.output[r][row] = p[0];
            if constexpr (Greedy) {
                auto& slot = best[(static_cast<std::uint32_t>(threadIdx.x) / 32U) * Rows + r];
                slot = strix::argmax::choose(slot, strix::argmax::load(p[0], row));
            }
        }
    }
    }
    if constexpr (Greedy) {
        __syncthreads();
        if (threadIdx.x < Rows) {
            LeanGreedy total{0.0F, 0U, 0U, 0U};
            for (unsigned w = 0; w < Waves; ++w) total = strix::argmax::choose(total, best[w * Rows + threadIdx.x]);
            records->row[threadIdx.x][blockIdx.x] = total;
        }
    }
}
template <unsigned Waves, unsigned Rows, unsigned Groups, bool Q6 = false>
__global__ __launch_bounds__(32U * Waves)
void prefetch8_wave_q8_lm_head_rows_kernel(LmHeadRowsViews views, const void* weights,
    std::uint32_t input_width, std::uint32_t output_width) {
    extern __shared__ uint4 staged[];
    lm_head_rows_body<Waves, Rows, Groups, Q6, false>(staged, views, weights, input_width, output_width, nullptr);
}
template <unsigned Waves, unsigned Rows, unsigned Groups>
__global__ __launch_bounds__(32U * Waves)
void prefetch8_wave_q8_lm_head_rows_greedy_kernel(LmHeadRowsViews views, LmHeadRowsRecords records, const void* weights,
    std::uint32_t input_width, std::uint32_t output_width) {
    extern __shared__ uint4 staged[];
    lm_head_rows_body<Waves, Rows, Groups, false, true>(staged, views, weights, input_width, output_width, &records);
}

// Wide windows are VALU-heavy; staging their inputs in LDS also keeps the
// direct per-wave input loads from overflowing the vector cache.
constexpr unsigned lm_head_rows_groups = 16;
constexpr std::uint64_t lm_head_rows_lds_limit = 48U * 1024U;
// Lean rows head: every logit keeps the leaf, lane-tree and local-tree DAG of
// prefetch8_wave_q8_lm_head_rows_kernel (and so of the R1 head) with fewer instructions around
// it. Inputs are staged in LDS and read as 16-byte words (one shift or mask per BF16 value); a
// wave runs Cols consecutive logits that share those reads; and the trees run two partial sums
// per register: an add pairs the same two values as the R1 shuffle tree and only the lane that
// keeps the result changes, so each sum is the same IEEE add with its operands commuted.
// The optional direct-input specialization keeps the same vector reads and trees
// but reads the BF16 words from the input rows instead of an LDS copy.
template <unsigned D>
__device__ __forceinline__ float lane_xor(float value) {
    if constexpr (D == 16) {
        const auto bits = __float_as_uint(value);
        return __uint_as_float(__builtin_amdgcn_permlanex16(bits, bits, 0x76543210U, 0xfedcba98U, false, false));
    } else {
        return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(value), 0x160 + static_cast<int>(D), 0xf, 0xf, false));
    }
}
// v[0..M/2) <- pairs (v[2i], v[2i+1]): lanes without bit D add v[2i] at lane ^ D, lanes with it
// add v[2i+1] at lane ^ D, so each value's partial sums keep their positions in a half-size segment.
template <unsigned D, unsigned M>
__device__ __forceinline__ void lean_pack(float* v, bool upper) {
#pragma unroll
    for (unsigned i = 0; i < M / 2U; ++i) {
        const float a = v[2U * i], b = v[2U * i + 1U];
        v[i] = (upper ? b : a) + lane_xor<D>(upper ? a : b);
    }
}
constexpr unsigned lm_head_lean_waves = 8;
template <unsigned Rows, unsigned Cols, bool Greedy, bool Staged = true>
__global__ __launch_bounds__(32U * lm_head_lean_waves)
void lean_q8_lm_head_rows_kernel(LmHeadRowsViews views, const void* weights, std::uint32_t input_width,
    std::uint32_t output_width, LeanGreedy* partials, std::uint32_t partial_count) {
    constexpr unsigned groups = hip_greedy_rows_partial_logits / (lm_head_lean_waves * Cols);
    constexpr unsigned slots = (Rows * Cols + 3U) / 4U * 4U, regs = slots / 4U;
    extern __shared__ uint4 staged[];
    __shared__ LeanGreedy best_lds[Greedy ? lm_head_lean_waves * slots : 1U];
    const auto words = input_width / 8U;
    if constexpr (Staged) {
        for (auto i = static_cast<std::uint32_t>(threadIdx.x); i < Rows * words; i += 32U * lm_head_lean_waves)
            staged[i] = reinterpret_cast<const uint4*>(views.input[i / words])[i % words];
        __syncthreads();
    }
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) % 32U, wave = static_cast<std::uint32_t>(threadIdx.x) / 32U;
    LeanGreedy best[regs]{};
    for (unsigned group = 0; group < groups; ++group) {
        const auto first = ((static_cast<std::uint32_t>(blockIdx.x) * groups + group) * lm_head_lean_waves + wave) * Cols;
        if (first >= output_width) break;
        float partial[Cols][Rows][8]{};
        const Q8Block* blocks[Cols];
        std::uint64_t next_codes[Cols]{};
        __half next_scale[Cols]{};
#pragma unroll
        for (unsigned c = 0; c < Cols; ++c)
            blocks[c] = static_cast<const Q8Block*>(weights) +
                static_cast<std::uint64_t>(min(first + c, output_width - 1U)) * (input_width / 32U);
        const auto fetch = [&](std::uint32_t feature) {
#pragma unroll
            for (unsigned c = 0; c < Cols; ++c) {
                const auto& block = blocks[c][feature / 32U];
                __builtin_memcpy(&next_codes[c], block.values + feature % 32U, 8U);
                next_scale[c] = block.scale;
            }
        };
        if (8U * lane < input_width) fetch(8U * lane);
        for (std::uint64_t chunk = 0; chunk < input_width; chunk += 256U) {
            const auto base = chunk + 8U * lane;
            if (base >= input_width) break;
            const auto feature = static_cast<std::uint32_t>(base);
            float weight[Cols][8];
#pragma unroll
            for (unsigned c = 0; c < Cols; ++c) {
                const auto packed = next_codes[c];
                const float scale = __half2float(next_scale[c]);
#pragma unroll
                for (unsigned local = 0; local < 8; ++local)
                    weight[c][local] = scale * static_cast<float>(static_cast<std::int8_t>(packed >> (8U * local)));
            }
            if (base + 256U < input_width) fetch(feature + 256U);
#pragma unroll
            for (unsigned r = 0; r < Rows; ++r) {
                const uint4 word = Staged ? staged[(r * input_width + feature) / 8U] :
                    reinterpret_cast<const uint4*>(views.input[r])[feature / 8U];
                const std::uint32_t pair[4] = {word.x, word.y, word.z, word.w};
                float act[8];
#pragma unroll
                for (unsigned q = 0; q < 4; ++q) {
                    act[2U * q] = __uint_as_float(pair[q] << 16U);
                    act[2U * q + 1U] = __uint_as_float(pair[q] & 0xffff0000U);
                }
#pragma unroll
                for (unsigned c = 0; c < Cols; ++c)
#pragma unroll
                    for (unsigned local = 0; local < 8; ++local)
                        partial[c][r][local] += act[local] * weight[c][local];
            }
        }
        // Value (slot, local) starts at list index n such that after the five lane stages it sits in
        // register slot / 4 at lane 8 * (slot % 4) + local.
        float v[slots * 8U];
#pragma unroll
        for (unsigned slot = 0; slot < slots; ++slot)
#pragma unroll
            for (unsigned local = 0; local < 8; ++local) {
                const unsigned n = ((slot >> 1U) & 1U) | ((slot & 1U) << 1U) | (((local >> 2U) & 1U) << 2U) |
                    (((local >> 1U) & 1U) << 3U) | ((local & 1U) << 4U) | ((slot >> 2U) << 5U);
                v[n] = slot < Rows * Cols ? partial[slot / Rows][slot % Rows][local] : 0.0F;
            }
        lean_pack<16, slots * 8U>(v, (lane & 16U) != 0U);
        lean_pack<8, slots * 4U>(v, (lane & 8U) != 0U);
        lean_pack<4, slots * 2U>(v, (lane & 4U) != 0U);
        lean_pack<2, slots>(v, (lane & 2U) != 0U);
        lean_pack<1, slots / 2U>(v, (lane & 1U) != 0U);
        // Local tree p0 += p4 ...; p0 += p2, p1 += p3; p0 += p1 over lanes 8q + local.
#pragma unroll
        for (unsigned j = 0; j < regs; ++j) {
            v[j] += lane_xor<4>(v[j]);
            v[j] += lane_xor<2>(v[j]);
            v[j] += lane_xor<1>(v[j]);
        }
        if ((lane & 7U) == 0U) {
#pragma unroll
            for (unsigned j = 0; j < regs; ++j) {
                const auto slot = 4U * j + lane / 8U;
                if (slot >= Rows * Cols) continue;
                const auto row = first + slot / Rows;
                if (row >= output_width) continue;
                views.output[slot % Rows][row] = v[j];
                if constexpr (Greedy) {
                    best[j] = strix::argmax::choose(best[j], strix::argmax::load(v[j], row));
                }
            }
        }
    }
    if constexpr (Greedy) {
        if ((lane & 7U) == 0U)
#pragma unroll
            for (unsigned j = 0; j < regs; ++j) best_lds[wave * slots + 4U * j + lane / 8U] = best[j];
        __syncthreads();
        if (threadIdx.x < Rows) {
            LeanGreedy total{0.0F, 0U, 0U, 0U};
            for (unsigned w = 0; w < lm_head_lean_waves; ++w)
                for (unsigned c = 0; c < Cols; ++c) total = strix::argmax::choose(total, best_lds[w * slots + c * Rows + threadIdx.x]);
            partials[static_cast<std::uint64_t>(threadIdx.x) * partial_count + blockIdx.x] = total;
        }
    }
}

template <unsigned Rows, unsigned Cols>
void launch_lean_lm_head_rows(const LmHeadRowsViews& views, const void* weights, std::uint32_t hidden,
    std::uint32_t vocab, LeanGreedy* partials, hipStream_t stream) {
    static const bool direct = strix::envcfg::flag("STRIX_QKERN_HEAD_DIRECT", false);
    const auto lds = static_cast<std::uint32_t>(Rows * hidden * 2U);
    const auto blocks = (vocab + hip_greedy_rows_partial_logits - 1U) / hip_greedy_rows_partial_logits;
    if (direct) {
        if (partials)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(lean_q8_lm_head_rows_kernel<Rows, Cols, true, false>), dim3(blocks),
                dim3(32U * lm_head_lean_waves), 0, stream, views, weights, hidden, vocab, partials, blocks);
        else
            hipLaunchKernelGGL(HIP_KERNEL_NAME(lean_q8_lm_head_rows_kernel<Rows, Cols, false, false>), dim3(blocks),
                dim3(32U * lm_head_lean_waves), 0, stream, views, weights, hidden, vocab, partials, blocks);
        return;
    }
    if (partials)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(lean_q8_lm_head_rows_kernel<Rows, Cols, true>), dim3(blocks),
            dim3(32U * lm_head_lean_waves), lds, stream, views, weights, hidden, vocab, partials, blocks);
    else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(lean_q8_lm_head_rows_kernel<Rows, Cols, false>), dim3(blocks),
            dim3(32U * lm_head_lean_waves), lds, stream, views, weights, hidden, vocab, partials, blocks);
}

int lm_head_rows_wide_mode() {
    static const int wide = [] {
        const char* v = std::getenv("STRIX_LM_HEAD_ROWS_WIDE");
        return !v ? 0 : std::string_view(v) == "w16" ? 1 : std::string_view(v) == "direct" ? 2 : 0;
    }();
    return wide;
}
template <unsigned Rows, bool Q6>
void launch_lm_head_rows_format(const LmHeadRowsViews& views, const void* weights,
    std::uint32_t hidden, std::uint32_t vocab, hipStream_t stream) {
    const auto lds = static_cast<std::uint64_t>(Rows) * hidden * 2U;
    // STRIX_LM_HEAD_ROWS_WIDE=w16|direct (wide windows): 16 waves share one staged copy
    // (twice the resident waves per CU), or no staging; same per-row arithmetic.
    const int wide = lm_head_rows_wide_mode();
    if (Rows > 4U && lds <= lm_head_rows_lds_limit && wide == 1) {
        constexpr auto rows_per_block = 16U * (lm_head_rows_groups / 2U);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(prefetch8_wave_q8_lm_head_rows_kernel<16, Rows, lm_head_rows_groups / 2U, Q6>),
            dim3((vocab + rows_per_block - 1U) / rows_per_block), dim3(512), static_cast<std::uint32_t>(lds), stream,
            views, weights, hidden, vocab);
    } else if (Rows > 4U && lds <= lm_head_rows_lds_limit && wide == 0) {
        constexpr auto rows_per_block = 8U * lm_head_rows_groups;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(prefetch8_wave_q8_lm_head_rows_kernel<8, Rows, lm_head_rows_groups, Q6>),
            dim3((vocab + rows_per_block - 1U) / rows_per_block), dim3(256), static_cast<std::uint32_t>(lds), stream,
            views, weights, hidden, vocab);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(prefetch8_wave_q8_lm_head_rows_kernel<8, Rows, 1, Q6>),
            dim3((vocab + 7U) / 8U), dim3(256), 0, stream, views, weights, hidden, vocab);
    }
}
// Fused head + greedy partials for a staged piece (Q8_0 head, more than four rows). The grid is
// one block per hip_greedy_rows_partial_logits vocabulary rows, the record count the finish reads.
bool lm_head_rows_greedy_fusable(unsigned rows, std::uint32_t hidden) {
    return rows > 4U && rows <= 16U && lm_head_rows_wide_mode() != 2 &&
        static_cast<std::uint64_t>(rows) * hidden * 2U + 16U * rows * sizeof(LeanGreedy) <= lm_head_rows_lds_limit;
}
template <unsigned Rows>
void launch_lm_head_rows_greedy(const LmHeadRowsViews& views, const LmHeadRowsRecords& records, const void* weights,
    std::uint32_t hidden, std::uint32_t vocab, hipStream_t stream) {
    constexpr auto rows_per_block = static_cast<std::uint32_t>(hip_greedy_rows_partial_logits);
    const auto grid = dim3((vocab + rows_per_block - 1U) / rows_per_block);
    const auto staged = static_cast<std::uint32_t>(static_cast<std::uint64_t>(Rows) * hidden * 2U);
    if (lm_head_rows_wide_mode() == 1)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(prefetch8_wave_q8_lm_head_rows_greedy_kernel<16, Rows, rows_per_block / 16U>),
            grid, dim3(512), staged + 16U * Rows * static_cast<std::uint32_t>(sizeof(LeanGreedy)), stream,
            views, records, weights, hidden, vocab);
    else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(prefetch8_wave_q8_lm_head_rows_greedy_kernel<8, Rows, rows_per_block / 8U>),
            grid, dim3(256), staged + 8U * Rows * static_cast<std::uint32_t>(sizeof(LeanGreedy)), stream,
            views, records, weights, hidden, vocab);
}
void launch_lm_head_rows_greedy_up_to_16(unsigned rows, const LmHeadRowsViews& views, const LmHeadRowsRecords& records,
    const void* weights, std::uint32_t hidden, std::uint32_t vocab, hipStream_t stream) {
    const auto run = [&]<unsigned N>() { launch_lm_head_rows_greedy<N>(views, records, weights, hidden, vocab, stream); };
    switch (rows) {
    case 5: run.template operator()<5>(); break;   case 6: run.template operator()<6>(); break;
    case 7: run.template operator()<7>(); break;   case 8: run.template operator()<8>(); break;
    case 9: run.template operator()<9>(); break;   case 10: run.template operator()<10>(); break;
    case 11: run.template operator()<11>(); break; case 12: run.template operator()<12>(); break;
    case 13: run.template operator()<13>(); break; case 14: run.template operator()<14>(); break;
    case 15: run.template operator()<15>(); break; default: run.template operator()<16>(); break;
    }
}
// The registered Q6P32 copy of the Q8_0 head replaces it when its shape matches.
const q6p::Tensor* q6_head(const void* weights, std::uint32_t hidden, std::uint32_t vocab) {
    const auto* q6 = q6p::find(weights);
    return q6 && q6->kind == q6p::Kind::q6p32 && q6->rows == vocab && q6->cols == hidden ? q6 : nullptr;
}
// Wide windows (staged, more than four rows) are VALU bound and the decode makes the Q6P32 head
// slower (8 rows: 4.36 -> 4.51 ms in the window profile); they keep the Q8_0 head, same bits.
template <unsigned Rows>
void launch_lm_head_rows(const LmHeadRowsViews& views, const void* weights,
    std::uint32_t hidden, std::uint32_t vocab, hipStream_t stream) {
    if (const auto* q6 = Rows <= 4U ? q6_head(weights, hidden, vocab) : nullptr)
        launch_lm_head_rows_format<Rows, true>(views, q6->base, hidden, vocab, stream);
    else launch_lm_head_rows_format<Rows, false>(views, weights, hidden, vocab, stream);
}
// 1..16 rows (pieces of cohort heads).
void launch_lm_head_rows_up_to_16(unsigned rows, const LmHeadRowsViews& views, const void* weights,
    std::uint32_t hidden, std::uint32_t vocab, hipStream_t stream) {
    const auto run = [&]<unsigned N>() { launch_lm_head_rows<N>(views, weights, hidden, vocab, stream); };
    switch (rows) {
    case 1: run.template operator()<1>(); break;   case 2: run.template operator()<2>(); break;
    case 3: run.template operator()<3>(); break;   case 4: run.template operator()<4>(); break;
    case 5: run.template operator()<5>(); break;   case 6: run.template operator()<6>(); break;
    case 7: run.template operator()<7>(); break;   case 8: run.template operator()<8>(); break;
    case 9: run.template operator()<9>(); break;   case 10: run.template operator()<10>(); break;
    case 11: run.template operator()<11>(); break; case 12: run.template operator()<12>(); break;
    case 13: run.template operator()<13>(); break; case 14: run.template operator()<14>(); break;
    case 15: run.template operator()<15>(); break; default: run.template operator()<16>(); break;
    }
}
// R1 prefetch8 wave8 head over a registered Q6P32 copy: the one-row rows kernel repeats the
// R1 leaf, lane-tree and local-tree DAG. False when the head has no copy.
bool launch_q6_head_r1(const std::uint16_t* hidden, const void* weights, std::uint32_t hidden_size,
    std::uint32_t vocab, float* logits, hipStream_t stream) {
    const auto* q6 = q6_head(weights, hidden_size, vocab);
    if (!q6) return false;
    LmHeadRowsViews views{};
    views.input[0] = hidden;
    views.output[0] = logits;
    launch_lm_head_rows_format<1, true>(views, q6->base, hidden_size, vocab, stream);
    return true;
}

template <
    typename Input,
    WeightKind Kind>
void launch_typed(
    const RuntimeTensorBinding& weights,
    const HipLmHeadLauncherConfig& config,
    const ExecutionBufferAddress& hidden,
    const ExecutionBufferAddress& logits) {
    if (config.prefetch_q8_wave) {
        if constexpr (Kind == WeightKind::q8_0 && std::is_same_v<Input, std::uint16_t>) {
            if (config.wave8_q8_wave && std::is_same_v<Input, std::uint16_t> &&
                launch_q6_head_r1(reinterpret_cast<const std::uint16_t*>(hidden.address), reinterpret_cast<const void*>(weights.address),
                    config.hidden_size, config.vocab_size, reinterpret_cast<float*>(logits.address), hip_execution_stream())) {
            } else if (config.wave8_q8_wave) {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(prefetch8_wave_q8_lm_head_kernel<Input, 8>),
                    dim3((config.vocab_size + 7U) / 8U), dim3(256), 0, hip_execution_stream(),
                    reinterpret_cast<const Input*>(hidden.address),
                    reinterpret_cast<const void*>(weights.address), config.hidden_size,
                    config.vocab_size, reinterpret_cast<float*>(logits.address));
            } else {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(prefetch8_wave_q8_lm_head_kernel<Input, 1>),
                    dim3(config.vocab_size), dim3(32), 0, hip_execution_stream(),
                    reinterpret_cast<const Input*>(hidden.address),
                    reinterpret_cast<const void*>(weights.address), config.hidden_size,
                    config.vocab_size, reinterpret_cast<float*>(logits.address));
            }
            check_hip(hipGetLastError(), "HIP prefetch Q8 LM-head launch");
            return;
        } else {
            throw std::invalid_argument("prefetch LM head supports BF16/Q8_0 only");
        }
    }
    if (config.virtual_q8_wave) {
        if constexpr (Kind != WeightKind::q8_0) {
            throw std::invalid_argument("virtual-wave LM head supports Q8_0 weights only");
        } else {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(virtual_q8_lm_head_kernel<Input>),
                dim3(config.vocab_size), dim3(32), 0, hip_execution_stream(),
                reinterpret_cast<const Input*>(hidden.address),
                reinterpret_cast<const void*>(weights.address), config.hidden_size,
                config.vocab_size, reinterpret_cast<float*>(logits.address));
            check_hip(hipGetLastError(), "HIP virtual-wave Q8 LM-head launch");
            return;
        }
    }
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            lm_head_kernel<
                Input,
                Kind>),
        dim3(config.vocab_size),
        dim3(kThreads),
        0,
        hip_execution_stream(),
        reinterpret_cast<const Input*>(
            hidden.address),
        reinterpret_cast<const void*>(
            weights.address),
        config.hidden_size,
        config.vocab_size,
        reinterpret_cast<float*>(
            logits.address));
    check_hip(
        hipGetLastError(),
        "HIP LM-head launch");
}

template <typename Input>
void launch_by_weight(
    WeightKind kind,
    const RuntimeTensorBinding& weights,
    const HipLmHeadLauncherConfig& config,
    const ExecutionBufferAddress& hidden,
    const ExecutionBufferAddress& logits) {
    switch (kind) {
    case WeightKind::f32:
        launch_typed<Input, WeightKind::f32>(
            weights, config, hidden, logits);
        return;
    case WeightKind::f16:
        launch_typed<Input, WeightKind::f16>(
            weights, config, hidden, logits);
        return;
    case WeightKind::bf16:
        launch_typed<Input, WeightKind::bf16>(
            weights, config, hidden, logits);
        return;
    case WeightKind::q8_0:
        launch_typed<Input, WeightKind::q8_0>(
            weights, config, hidden, logits);
        return;
    case WeightKind::iq4_nl:
        launch_typed<Input, WeightKind::iq4_nl>(
            weights, config, hidden, logits);
        return;
    case WeightKind::iq4_xs:
        launch_typed<Input, WeightKind::iq4_xs>(
            weights, config, hidden, logits);
        return;
#define STRIX_GGML_KIND_CASE(KIND, type) \
    case WeightKind::KIND: \
        launch_typed<Input, WeightKind::KIND>( \
            weights, config, hidden, logits); \
        return;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_CASE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_CASE)
#undef STRIX_GGML_KIND_CASE
    }
    throw std::logic_error(
        "unreachable LM-head weight kind");
}

struct PreparedLmHead {
    const HipLmHeadLauncherConfig& config;
    ExecutionBufferAddress hidden, logits;
    const RuntimeTensorBinding& runtime_tensor;
    WeightKind kind;
};
PreparedLmHead prepare_lm_head(
    const ExecutionDispatchContext& context,
    void* user_data) {
    if (context.op !=
        SerialOp::lm_head) {
        throw std::invalid_argument(
            "HIP LM-head launcher attached to wrong operator");
    }
    if (context.reads.size() != 1 ||
        context.writes.size() != 1) {
        throw std::invalid_argument(
            "HIP LM-head expects one read and one write");
    }
    if (context.logical_rows != 1 ||
        context.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP LM-head v1 supports only C1 one-row execution");
    }
    if (user_data == nullptr) {
        throw std::invalid_argument(
            "HIP LM-head requires configuration");
    }

    const auto& config =
        *static_cast<
            HipLmHeadLauncherConfig*>(
                user_data);
    validate_config(config);

    if (config.prefetch_q8_wave) {
        if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture)
            throw std::invalid_argument("prefetch LM head supports eager execution only");
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        check_hip(hipStreamIsCapturing(hip_execution_stream(), &capture), "query prefetch LM-head capture");
        if (capture != hipStreamCaptureStatusNone)
            throw std::invalid_argument("prefetch LM head cannot be captured");
    }

    const auto hidden =
        resolve_dispatch_read(
            context, 0);
    const auto logits =
        resolve_dispatch_write(
            context, 0);
    const auto& access =
        resolve_dispatch_tensor_access(
            context, 0);
    if (access.tensor_indices.size() != 1U) {
        throw std::invalid_argument(
            "HIP LM-head requires exactly one output tensor");
    }
    if (context.resources == nullptr ||
        context.resources->model_manifest ==
            nullptr) {
        throw std::invalid_argument(
            "HIP LM-head requires model manifest resources");
    }

    const auto tensor_index =
        access.tensor_indices.front();
    if (tensor_index >=
        context.resources->
            model_manifest->
            tensors.size()) {
        throw std::out_of_range(
            "HIP LM-head tensor index exceeds manifest");
    }

    const auto& manifest_tensor =
        context.resources->
            model_manifest->
            tensors[tensor_index];
    const auto& runtime_tensor =
        resolve_dispatch_tensor(
            context,
            tensor_index);

    if (manifest_tensor.role !=
            TensorRole::output_head ||
        manifest_tensor.dimensions.size() < 2 ||
        manifest_tensor.dimensions[0] !=
            config.hidden_size ||
        manifest_tensor.dimensions[1] !=
            config.vocab_size) {
        throw std::invalid_argument(
            "HIP LM-head tensor geometry does not match contract");
    }
    if (!runtime_tensor_gpu_visible(runtime_tensor.storage) ||
        runtime_tensor.address == 0) {
        throw std::invalid_argument(
            "HIP LM-head requires resident or repacked GPU-visible weights");
    }

    const auto kind =
        weight_kind(
            manifest_tensor.ggml_type);
    const auto required =
        required_weight_bytes(
            kind,
            config.hidden_size,
            config.vocab_size);
    if (runtime_tensor.bytes < required) {
        throw std::invalid_argument(
            "HIP LM-head runtime weight binding is undersized");
    }

    const auto input_required =
        static_cast<std::uint64_t>(
            config.hidden_size) *
        static_cast<std::uint32_t>(
            config.input);
    const auto output_required =
        static_cast<std::uint64_t>(
            config.vocab_size) *
        sizeof(float);
    if (hidden.bytes < input_required ||
        logits.bytes < output_required) {
        throw std::invalid_argument(
            "HIP LM-head scratch buffers are undersized");
    }

    if (config.prefetch_q8_wave) {
        if (kind != WeightKind::q8_0)
            throw std::invalid_argument("prefetch LM head supports BF16/Q8_0 only");
        q8_projection_organization::validate_views(config.hidden_size, config.vocab_size,
            {hidden.address, hidden.bytes}, {runtime_tensor.address, runtime_tensor.bytes},
            {logits.address, logits.bytes});
    }

    // The virtual-wave branch checks this again at dispatch. Keep its static
    // type requirement in the shared preparation before a window can enqueue.
    if (config.virtual_q8_wave && kind != WeightKind::q8_0)
        throw std::invalid_argument("virtual-wave LM head supports Q8_0 weights only");
    return {config, hidden, logits, runtime_tensor, kind};
}

void preflight_lm_head(const ExecutionDispatchContext& context, void* user_data) {
    if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture)
        throw std::invalid_argument("known-window launcher preflight requires eager scope");
    static_cast<void>(prepare_lm_head(context, user_data));
}

void launch_lm_head(
    const ExecutionDispatchContext& context,
    void* user_data) {
    const auto prepared = prepare_lm_head(context, user_data);
    const auto& config = prepared.config;
    const auto& hidden = prepared.hidden;
    const auto& logits = prepared.logits;
    const auto& runtime_tensor = prepared.runtime_tensor;
    const auto kind = prepared.kind;

    switch (config.input) {
    case HipActivationStorage::bf16:
        launch_by_weight<
            std::uint16_t>(
                kind,
                runtime_tensor,
                config,
                hidden,
                logits);
        return;
    case HipActivationStorage::fp32:
        launch_by_weight<float>(
            kind,
            runtime_tensor,
            config,
            hidden,
            logits);
        return;
    }
    throw std::logic_error(
        "unreachable LM-head input storage");
}

std::string kernel_id(
    const ExecutionBindingRequest& request,
    const HipLmHeadLauncherConfig& config) {
    return
        "hip.lm_head." +
        std::string(
            storage_name(
                config.input)) +
        ".h" +
        std::to_string(
            config.hidden_size) +
        ".v" +
        std::to_string(
            config.vocab_size) +
        ".r" +
        std::to_string(
            request.logical_rows) +
        ".c" +
        std::to_string(
            request.request_concurrency) +
        (config.wave8_q8_wave ? std::string(q8_projection_organization::lm_wave8_marker)
            : config.prefetch_q8_wave ? std::string(q8_projection_organization::lm_prefetch_marker) : "") +
        (config.virtual_q8_wave ? ".virtual256.q8" : "");
}

std::uint64_t implementation_identity(
    const HipLmHeadLauncherConfig& config) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, kLauncherTag);
    hash_u64(hash, config.hidden_size);
    hash_u64(hash, config.vocab_size);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            config.input));
    if (config.virtual_q8_wave) hash_u64(hash, 0x5649525451323536ULL);
    if (config.prefetch_q8_wave) hash_u64(hash, 0x4c4d513850524631ULL);
    if (config.wave8_q8_wave) hash_u64(hash, 0x4c4d513857415638ULL);
    return hash;
}

}  // namespace

void register_hip_lm_head_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipLmHeadLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification) {
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    validate_config(config);
    if (request.prefetch_q8_lm_head != config.prefetch_q8_wave ||
        request.wave8_q8_lm_head != config.wave8_q8_wave)
        throw std::invalid_argument("LM-head organization request/config mismatch");
    if (config.prefetch_q8_wave && (request.device_graph_inputs || request.require_graph_safe))
        throw std::invalid_argument("prefetch LM head registers eager variants only");
    if (config.hidden_size !=
            model.hidden_size ||
        config.vocab_size !=
            model.vocab_size) {
        throw std::invalid_argument(
            "HIP LM-head config does not match model contract");
    }
    if (request.logical_rows != 1 ||
        request.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP LM-head v1 registers only C1 one-row variants");
    }
    if (qualification ==
            kernels::QualificationStatus::target_qualified ||
        qualification ==
            kernels::QualificationStatus::production ||
        qualification ==
            kernels::QualificationStatus::rejected) {
        throw std::invalid_argument(
            "unmeasured HIP LM-head may only be experimental or oracle-matched");
    }

    kernels::KernelVariant variant;
    variant.id =
        kernel_id(
            request,
            config);
    variant.family =
        kernels::OperatorFamily::lm_head;
    variant.phase =
        kernels::ExecutionPhase::decode;
    variant.shape =
        execution_shape_for(
            model,
            SerialOp::lm_head,
            request);
    variant.wave =
        kernels::WaveSize::wave32;
    variant.resources.workgroup_threads =
        config.virtual_q8_wave && !config.wave8_q8_wave ? 32U : kThreads;
    variant.resources.lds_bytes =
        config.virtual_q8_wave ? 0U : kThreads * sizeof(float);
    variant.numerics.input_precision =
        storage_name(
            config.input);
    variant.numerics.accumulator_precision =
        "fp32";
    variant.numerics.output_precision =
        "fp32";
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        qualification;
    variant.graph_safe = !config.prefetch_q8_wave;
    variant.input_layout =
        "row-major";
    variant.output_layout =
        "fp32-logits";
    variant.compiler_identity =
        config.virtual_q8_wave ? "hip-gfx1151-lm-head-v2-virtual256-q8" : "hip-gfx1151-lm-head-v1";
    if (config.prefetch_q8_wave) variant.compiler_identity += ":prefetch8-load-v1";
    if (config.wave8_q8_wave) variant.compiler_identity += ":wave8-rows-v1";
    variant.measured_seconds = 0.0;

    registry.add(
        std::move(variant));
}

ExecutionLauncher
make_hip_lm_head_launcher(
    std::string kernel_id_value,
    HipLmHeadLauncherConfig* config) {
    if (kernel_id_value.empty() ||
        config == nullptr) {
        throw std::invalid_argument(
            "HIP LM-head launcher requires id and config");
    }
    validate_config(*config);
    if (config->prefetch_q8_wave ||
        kernel_id_value.find(q8_projection_organization::lm_prefetch_marker) != std::string::npos ||
        kernel_id_value.find(q8_projection_organization::lm_wave8_marker) != std::string::npos) {
        ExecutionBindingRequest request;
        if (kernel_id_value != kernel_id(request, *config))
            throw std::invalid_argument("prefetch LM-head launcher requires its exact eager C1 kernel id");
    }
    return ExecutionLauncher{
        std::move(kernel_id_value),
        implementation_identity(
            *config),
        &launch_lm_head,
        config,
        &preflight_lm_head,
    };
}

namespace {
struct PreparedLmHeadRows { LmHeadRowsViews views{}; const void* weights = nullptr; };
// Lean rows (config.rows_lean 1 or 2) stage every row's input in LDS next to its greedy slots.
// Windows of up to four rows keep the unstaged rows kernel, which is faster there.
bool lean_rows_admitted(const HipLmHeadLauncherConfig& config, std::size_t rows) {
    static const bool direct = strix::envcfg::flag("STRIX_QKERN_HEAD_DIRECT", false);
    return (direct || config.rows_lean == 1U || config.rows_lean == 2U) && rows > 4U && rows <= 8U && config.hidden_size % 256U == 0U &&
        rows * config.hidden_size * 2U <= lm_head_rows_lds_limit;
}
template <unsigned Cols>
void launch_lean_rows_cols(std::size_t rows, const LmHeadRowsViews& views, const void* weights, std::uint32_t h,
    std::uint32_t v, LeanGreedy* partials, hipStream_t stream) {
    switch (rows) {
    case 2: launch_lean_lm_head_rows<2, Cols>(views, weights, h, v, partials, stream); break;
    case 3: launch_lean_lm_head_rows<3, Cols>(views, weights, h, v, partials, stream); break;
    case 4: launch_lean_lm_head_rows<4, Cols>(views, weights, h, v, partials, stream); break;
    case 5: launch_lean_lm_head_rows<5, Cols>(views, weights, h, v, partials, stream); break;
    case 6: launch_lean_lm_head_rows<6, Cols>(views, weights, h, v, partials, stream); break;
    case 7: launch_lean_lm_head_rows<7, Cols>(views, weights, h, v, partials, stream); break;
    default: launch_lean_lm_head_rows<8, Cols>(views, weights, h, v, partials, stream); break;
    }
}
void launch_lean_rows(std::uint32_t cols, std::size_t rows, const LmHeadRowsViews& views, const void* weights,
    std::uint32_t h, std::uint32_t v, LeanGreedy* partials, hipStream_t stream) {
    if (cols == 2U) launch_lean_rows_cols<2>(rows, views, weights, h, v, partials, stream);
    else launch_lean_rows_cols<1>(rows, views, weights, h, v, partials, stream);
}

PreparedLmHeadRows prepare_lm_head_rows(std::span<const ExecutionDispatchContext> rows,
    const HipLmHeadLauncherConfig& config) {
    if (rows.size() < 2U || rows.size() > lm_head_cohort_max_rows)
        throw std::invalid_argument("known-window LM head supports two to eight rows (cohorts: up to 32)");
    validate_config(config);
    if (!config.prefetch_q8_wave || config.input != HipActivationStorage::bf16)
        throw std::invalid_argument("known-window LM head rows require the eager prefetch BF16/Q8_0 head");
    const auto expected = kernel_id(ExecutionBindingRequest{}, config);
    PreparedLmHeadRows out;
    std::array<ExecutionBufferAddress, lm_head_cohort_max_rows> inputs{}, outputs{};
    const RuntimeTensorBinding* tensor = nullptr;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        if (!rows[r].kernel_id || *rows[r].kernel_id != expected)
            throw std::invalid_argument("known-window LM head row kernel differs from its R1 source");
        const auto p = prepare_lm_head(rows[r], const_cast<HipLmHeadLauncherConfig*>(&config));
        if (p.kind != WeightKind::q8_0) throw std::invalid_argument("known-window LM head rows support Q8_0 only");
        if (tensor && (p.runtime_tensor.address != tensor->address || p.runtime_tensor.bytes != tensor->bytes))
            throw std::invalid_argument("known-window LM head rows must share one head tensor");
        tensor = &p.runtime_tensor;
        inputs[r] = p.hidden; outputs[r] = p.logits;
        out.views.input[r] = reinterpret_cast<const std::uint16_t*>(p.hidden.address);
        out.views.output[r] = reinterpret_cast<float*>(p.logits.address);
    }
    const auto overlap = [](const ExecutionBufferAddress& a, const ExecutionBufferAddress& b) {
        return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
    };
    for (std::size_t r = 0; r < rows.size(); ++r)
        for (std::size_t q = 0; q < rows.size(); ++q)
            if ((q != r && overlap(outputs[r], outputs[q])) || overlap(outputs[r], inputs[q]))
                throw std::invalid_argument("known-window LM head row output aliases another row");
    out.weights = reinterpret_cast<const void*>(tensor->address);
    return out;
}
} // namespace

void lm_head_known_window_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipLmHeadLauncherConfig& config) {
    if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture)
        throw std::invalid_argument("known-window launcher preflight requires eager scope");
    std::optional<PreparedLmHeadRows> local;
    static_cast<void>(KnownWindowRowsBindingScope::get(local, [&] { return prepare_lm_head_rows(rows, config); }));
}

bool lm_head_cohort_lean_enabled() {
    static const bool cohort_lean = strix::envcfg::flag("STRIX_LM_HEAD_COHORT_LEAN", false);
    return !qpf2k::suppressed(qpf2k::Item::cohort_lean) && cohort_lean;
}

void lm_head_known_window_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipLmHeadLauncherConfig& config) {
    std::optional<PreparedLmHeadRows> local;
    const auto& p = KnownWindowRowsBindingScope::get(local, [&] { return prepare_lm_head_rows(rows, config); });
    const auto stream = hip_execution_stream();
    const auto h = config.hidden_size, v = config.vocab_size;
    check_hip(hipPeekAtLastError(), "known-window LM head rows stale launch error");
    if (lean_rows_admitted(config, rows.size())) {
        launch_lean_rows(config.rows_lean, rows.size(), p.views, p.weights, h, v, nullptr, stream);
        check_hip(hipGetLastError(), "known-window lean LM head rows launch");
        return;
    }
    // STRIX_LM_HEAD_COHORT_LEAN=1 (default off): cohorts of several sessions (9..32 rows) run as pieces of eight rows
    // on the lean rows head (two logits per wave unless STRIX_LM_HEAD_ROWS_LEAN=1), a last piece of fewer than five
    // rows on the rows head; every logit is bit-identical to the R1 head either way.
    if (lm_head_cohort_lean_enabled() && rows.size() > lm_head_window_max_rows) {
        auto lean = config;
        lean.rows_lean = config.rows_lean ? config.rows_lean : 2U;
        for (std::size_t first = 0; first < rows.size(); first += 8U) {
            const auto n = static_cast<unsigned>(std::min<std::size_t>(8U, rows.size() - first));
            LmHeadRowsViews part{};
            for (unsigned r = 0; r < n; ++r) { part.input[r] = p.views.input[first + r]; part.output[r] = p.views.output[first + r]; }
            if (lean_rows_admitted(lean, n)) launch_lean_rows(lean.rows_lean, n, part, p.weights, h, v, nullptr, stream);
            else launch_lm_head_rows_up_to_16(n, part, p.weights, h, v, stream);
        }
        check_hip(hipGetLastError(), "known-window lean LM head cohort rows launch");
        return;
    }
    // Cohorts of several sessions (9..32 rows): unstaged launches of at most 16 rows (per row the same DAG).
    if (rows.size() > lm_head_window_max_rows) {
        for (std::size_t first = 0; first < rows.size(); first += 16U) {
            const auto n = static_cast<unsigned>(std::min<std::size_t>(16U, rows.size() - first));
            LmHeadRowsViews part{};
            for (unsigned r = 0; r < n; ++r) { part.input[r] = p.views.input[first + r]; part.output[r] = p.views.output[first + r]; }
            launch_lm_head_rows_up_to_16(n, part, p.weights, h, v, stream);
        }
        check_hip(hipGetLastError(), "known-window LM head cohort rows launch");
        return;
    }
    switch (rows.size()) {
    case 2: launch_lm_head_rows<2>(p.views, p.weights, h, v, stream); break;
    case 3: launch_lm_head_rows<3>(p.views, p.weights, h, v, stream); break;
    case 4: launch_lm_head_rows<4>(p.views, p.weights, h, v, stream); break;
    case 5: launch_lm_head_rows<5>(p.views, p.weights, h, v, stream); break;
    case 6: launch_lm_head_rows<6>(p.views, p.weights, h, v, stream); break;
    case 7: launch_lm_head_rows<7>(p.views, p.weights, h, v, stream); break;
    default: launch_lm_head_rows<8>(p.views, p.weights, h, v, stream); break;
    }
    check_hip(hipGetLastError(), "known-window LM head rows launch");
}

// One launch per head piece writes the logits and the greedy partial records (default; the
// rows head and the greedy partials launch stay separate with STRIX_LM_HEAD_FUSED_GREEDY=0, for
// pieces of fewer than five rows, unstaged pieces and the lean heads). The greedy segments must
// cover the head rows in order; anything else takes the separate launches.
bool lm_head_rows_with_greedy_fused(std::span<const ExecutionDispatchContext> rows,
    const HipLmHeadLauncherConfig& config, std::span<const HipLmHeadGreedyRows> greedy) {
    static const bool enabled = [] { const char* v = std::getenv("STRIX_LM_HEAD_FUSED_GREEDY"); return !v || v[0] != '0'; }();
    const auto h = config.hidden_size, v = config.vocab_size;
    // In-process A/B: STRIX_COHORT_AB_OFF=headgreedy.
    if (!enabled || known_window_cohort_feature_off("headgreedy") || rows.size() < 5U || rows.size() > lm_head_cohort_max_rows || lean_rows_admitted(config, rows.size())) return false;
    const bool cohort = rows.size() > lm_head_window_max_rows;
    if (cohort && lm_head_cohort_lean_enabled()) return false;
    const std::size_t piece = cohort ? 16U : lm_head_window_max_rows;
    for (std::size_t first = 0; first < rows.size(); first += piece)
        if (!lm_head_rows_greedy_fusable(static_cast<unsigned>(std::min(piece, rows.size() - first)), h)) return false;
    const std::uint64_t count = (static_cast<std::uint64_t>(v) + hip_greedy_rows_partial_logits - 1U) / hip_greedy_rows_partial_logits;
    std::array<LeanGreedy*, lm_head_cohort_max_rows> records{};
    std::size_t covered = 0;
    for (const auto& segment : greedy) {
        if (segment.rows.empty() || segment.rows.size() > hip_greedy_rows_max || covered + segment.rows.size() > rows.size() ||
            segment.partials.address == 0 ||
            segment.partials.bytes < hip_greedy_rows_scratch_bytes(static_cast<std::uint32_t>(segment.rows.size()), v)) return false;
        for (std::size_t r = 0; r < segment.rows.size(); ++r, ++covered) {
            if (resolve_dispatch_write(segment.rows[r], 0).address != resolve_dispatch_write(rows[covered], 0).address) return false;
            records[covered] = reinterpret_cast<LeanGreedy*>(segment.partials.address) + r * count;
        }
    }
    if (covered != rows.size()) return false;
    const auto p = prepare_lm_head_rows(rows, config);
    const auto stream = hip_execution_stream();
    check_hip(hipPeekAtLastError(), "known-window fused LM head rows stale launch error");
    for (std::size_t first = 0; first < rows.size(); first += piece) {
        const auto n = static_cast<unsigned>(std::min(piece, rows.size() - first));
        LmHeadRowsViews part{};
        LmHeadRowsRecords part_records{};
        for (unsigned r = 0; r < n; ++r) {
            part.input[r] = p.views.input[first + r]; part.output[r] = p.views.output[first + r];
            part_records.row[r] = records[first + r];
        }
        launch_lm_head_rows_greedy_up_to_16(n, part, part_records, p.weights, h, v, stream);
    }
    check_hip(hipGetLastError(), "known-window fused LM head rows launch");
    for (const auto& segment : greedy)
        enqueue_hip_greedy_rows_finish(segment.partials, static_cast<std::uint32_t>(segment.rows.size()), v, segment.replies, stream);
    return true;
}

void lm_head_rows_with_greedy(std::span<const ExecutionDispatchContext> rows,
    const HipLmHeadLauncherConfig& config, std::span<const HipLmHeadGreedyRows> greedy,
    bool window_rows_only) {
    if (window_rows_only && rows.size() > lm_head_window_max_rows)
        throw std::invalid_argument("known-window greedy LM head supports two to eight rows");
    if (lm_head_rows_with_greedy_fused(rows, config, greedy)) return;
    lm_head_known_window_rows_launch(rows, config);
    for (const auto& segment : greedy) {
        std::array<ExecutionBufferAddress, hip_greedy_rows_max> logits{};
        for (std::size_t r = 0; r < segment.rows.size(); ++r) logits[r] = resolve_dispatch_write(segment.rows[r], 0);
        enqueue_hip_greedy_rows(std::span<const ExecutionBufferAddress>(logits.data(), segment.rows.size()), config.vocab_size,
            segment.partials, segment.replies, hip_execution_stream());
    }
}

void lm_head_known_window_rows_launch_greedy(std::span<const ExecutionDispatchContext> rows,
    const HipLmHeadLauncherConfig& config, ExecutionBufferAddress partials, HipGreedyRowsReply* replies) {
    if (!lean_rows_admitted(config, rows.size())) {
        const HipLmHeadGreedyRows greedy{rows, partials, replies};
        lm_head_rows_with_greedy(rows, config, std::span<const HipLmHeadGreedyRows>(&greedy, 1), true);
        return;
    }
    std::optional<PreparedLmHeadRows> local;
    const auto& p = KnownWindowRowsBindingScope::get(local, [&] { return prepare_lm_head_rows(rows, config); });
    const auto rows_count = static_cast<std::uint32_t>(rows.size());
    const auto bytes = hip_greedy_rows_scratch_bytes(rows_count, config.vocab_size);
    if (!partials.address || partials.address % alignof(LeanGreedy) != 0 || partials.bytes < bytes || !replies)
        throw std::invalid_argument("known-window fused greedy LM head scratch/reply views are invalid");
    const auto stream = hip_execution_stream();
    check_hip(hipPeekAtLastError(), "known-window LM head rows stale launch error");
    launch_lean_rows(config.rows_lean, rows.size(), p.views, p.weights, config.hidden_size, config.vocab_size,
        reinterpret_cast<LeanGreedy*>(partials.address), stream);
    check_hip(hipGetLastError(), "known-window fused greedy LM head rows launch");
    enqueue_hip_greedy_rows_finish(partials, rows_count, config.vocab_size, replies, stream);
}

void HipLmHeadLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry) {
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP LM-head bundle requires valid compiled execution");
    }

    for (std::size_t index = 0;
         index < compiled.execution.steps.size();
         ++index) {
        if (compiled.execution.steps[index].op !=
            SerialOp::lm_head) {
            continue;
        }
        const auto& binding =
            compiled.bindings.bindings[index];
        if (binding.host_or_runtime_step ||
            !binding.kernel_id.has_value() ||
            binding.kernel_id->empty()) {
            throw std::invalid_argument(
                "HIP LM-head step has no kernel binding");
        }
        ExecutionBindingRequest request;
        request.logical_rows = compiled.logical_rows;
        request.request_concurrency = compiled.request_concurrency;
        if (*binding.kernel_id != kernel_id(request, config_)) {
            throw std::invalid_argument(
                "default HIP LM-head cannot execute another numerical kernel/profile");
        }
        registry.add_kernel_launcher(
            make_hip_lm_head_launcher(
                *binding.kernel_id,
                &config_));
        return;
    }
    throw std::invalid_argument(
        "compiled execution has no LM-head step");
}


#ifdef STRIX_BUILD_GDN_LM_WAVE8_DIAGNOSTIC
namespace gdn_lm_wave8_diagnostic::detail {
const void* lm_symbol(Kind kind, Arm arm) {
    if (kind != Kind::lm_bf16) throw std::invalid_argument("invalid LM diagnostic kind");
    switch (arm) {
    case Arm::reference:
        return reinterpret_cast<const void*>(virtual_q8_lm_head_kernel<std::uint16_t>);
    case Arm::prefetch1:
        return reinterpret_cast<const void*>(prefetch8_wave_q8_lm_head_kernel<std::uint16_t, 1>);
    case Arm::wave8:
        return reinterpret_cast<const void*>(prefetch8_wave_q8_lm_head_kernel<std::uint16_t, 8>);
    }
    throw std::invalid_argument("invalid LM diagnostic arm");
}
void enqueue_lm(Shape shape, Span input, Span weights, Span output, Kind kind, Arm arm, hipStream_t stream) {
    const auto launch = validate(shape, input, weights, output, kind, arm);
    if (kind != Kind::lm_bf16) throw std::invalid_argument("invalid LM diagnostic kind");
    check_hip(hipPeekAtLastError(), "LM wave8 diagnostic stale error");
    if (arm == Arm::reference) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(virtual_q8_lm_head_kernel<std::uint16_t>),
            dim3(launch.grid_x), dim3(launch.block_x), 0, stream,
            reinterpret_cast<const std::uint16_t*>(input.address),
            reinterpret_cast<const void*>(weights.address), shape.width, shape.outputs,
            reinterpret_cast<float*>(output.address));
    } else if (arm == Arm::prefetch1) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(prefetch8_wave_q8_lm_head_kernel<std::uint16_t, 1>),
            dim3(launch.grid_x), dim3(launch.block_x), 0, stream,
            reinterpret_cast<const std::uint16_t*>(input.address),
            reinterpret_cast<const void*>(weights.address), shape.width, shape.outputs,
            reinterpret_cast<float*>(output.address));
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(prefetch8_wave_q8_lm_head_kernel<std::uint16_t, 8>),
            dim3(launch.grid_x), dim3(launch.block_x), 0, stream,
            reinterpret_cast<const std::uint16_t*>(input.address),
            reinterpret_cast<const void*>(weights.address), shape.width, shape.outputs,
            reinterpret_cast<float*>(output.address));
    }
    check_hip(hipGetLastError(), "LM wave8 diagnostic launch");
}
} // namespace gdn_lm_wave8_diagnostic::detail
#endif

HipMtpLmHeadRecipe select_hip_mtp_lm_head_recipe(const ModelContract& model,const MtpTensorView& tensor) {
    const auto h=model.hidden_size,v=model.vocab_size;
    if(!h || h>std::numeric_limits<std::uint32_t>::max() ||
       !v || v>std::numeric_limits<std::uint32_t>::max() || !tensor.manifest)
        throw std::invalid_argument("MTP head geometry or tensor exceeds native launch ABI");
    const auto kind=weight_kind(tensor.manifest->ggml_type);
    mtp_native::tensor(tensor,{h,v},required_weight_bytes(kind,static_cast<std::uint32_t>(h),v));
    // Match the incumbent target overlay's closed geometry and view contract.
    // Other valid original formats/geometries retain their generic recipe.
    if(kind==WeightKind::q8_0 && h<=10240U && h%32U==0U && v<=248320U &&
       tensor.runtime->address%16U==0U)
        return HipMtpLmHeadRecipe::prefetch8_wave8_q8;
    return HipMtpLmHeadRecipe::generic256;
}
std::uint64_t hip_mtp_lm_head_profile_identity(const ModelContract& model,const MtpTensorView& tensor) {
    const auto recipe=select_hip_mtp_lm_head_recipe(model,tensor);
    // The v2 domain explicitly separates this auto-selected MTP-only recipe
    // from the previous generic MTP profile. Never infer numeric equivalence.
    std::uint64_t hash=fnv_offset;
    for(const auto word:{std::uint64_t{0x4d54504c4d483032ULL},static_cast<std::uint64_t>(recipe),
        static_cast<std::uint64_t>(tensor.manifest->ggml_type),model.hidden_size,model.vocab_size,
        std::uint64_t{16U},std::uint64_t{32U},std::uint64_t{256U}})hash_u64(hash,word);
    return hash?hash:1U;
}
void preflight_hip_mtp_lm_head(const MtpNativeContext& c) {
    mtp_native::context(c);
    const auto h=c.model->hidden_size,v=c.model->vocab_size;
    if(!h || h>std::numeric_limits<std::uint32_t>::max() ||
       !v || v>std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("MTP head geometry exceeds native launch ABI");
    const auto& t=c.tensors->lm_head;
    const auto recipe=select_hip_mtp_lm_head_recipe(*c.model,t);
    auto input=mtp_native::buffer(c,HipMtpBuffer::hidden0,mtp_native::product(h,2));
    auto out=mtp_native::buffer(c,HipMtpBuffer::logits,mtp_native::product(v,4));
    mtp_native::disjoint({input,out,{t.runtime->address,t.runtime->bytes,t.runtime->allocation_generation}});
    if(recipe==HipMtpLmHeadRecipe::prefetch8_wave8_q8)
        q8_projection_organization::validate_views(static_cast<std::uint32_t>(h),static_cast<std::uint32_t>(v),
            {input.address,input.bytes},{t.runtime->address,t.runtime->bytes},{out.address,out.bytes});
}
void launch_hip_mtp_lm_head(const MtpNativeContext& c,void* stream) {
    preflight_hip_mtp_lm_head(c); mtp_native::stream(c,stream);
    check_hip(hipPeekAtLastError(),"MTP head pending HIP error");
    if(select_hip_mtp_lm_head_recipe(*c.model,c.tensors->lm_head)==HipMtpLmHeadRecipe::prefetch8_wave8_q8) {
        const auto outputs=static_cast<std::uint32_t>(c.model->vocab_size);
        if(launch_q6_head_r1(reinterpret_cast<const std::uint16_t*>(c.resources->view(HipMtpBuffer::hidden0).address),
            reinterpret_cast<const void*>(c.tensors->lm_head.runtime->address),static_cast<std::uint32_t>(c.model->hidden_size),outputs,
            reinterpret_cast<float*>(c.resources->view(HipMtpBuffer::logits).address),reinterpret_cast<hipStream_t>(stream))) {
            check_hip(hipGetLastError(),"MTP Q6P32 LM head launch");
            return;
        }
        hipLaunchKernelGGL(HIP_KERNEL_NAME(prefetch8_wave_q8_lm_head_kernel<std::uint16_t,8>),
            dim3(outputs/8U+static_cast<std::uint32_t>(outputs%8U!=0U)),dim3(256U),0,
            reinterpret_cast<hipStream_t>(stream),
            reinterpret_cast<const std::uint16_t*>(c.resources->view(HipMtpBuffer::hidden0).address),
            reinterpret_cast<const void*>(c.tensors->lm_head.runtime->address),
            static_cast<std::uint32_t>(c.model->hidden_size),outputs,
            reinterpret_cast<float*>(c.resources->view(HipMtpBuffer::logits).address));
        check_hip(hipGetLastError(),"MTP prefetch8 wave8 LM head launch");
        return;
    }
    auto launch=[&]<WeightKind Kind>() {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(lm_head_kernel<std::uint16_t,Kind>),
            dim3(static_cast<std::uint32_t>(c.model->vocab_size)),dim3(kThreads),0,
            reinterpret_cast<hipStream_t>(stream),
            reinterpret_cast<const std::uint16_t*>(c.resources->view(HipMtpBuffer::hidden0).address),
            reinterpret_cast<const void*>(c.tensors->lm_head.runtime->address),
            static_cast<std::uint32_t>(c.model->hidden_size),
            static_cast<std::uint32_t>(c.model->vocab_size),
            reinterpret_cast<float*>(c.resources->view(HipMtpBuffer::logits).address));
    };
    switch(weight_kind(c.tensors->lm_head.manifest->ggml_type)) {
    case WeightKind::f32:launch.template operator()<WeightKind::f32>();break;
    case WeightKind::f16:launch.template operator()<WeightKind::f16>();break;
    case WeightKind::bf16:launch.template operator()<WeightKind::bf16>();break;
    case WeightKind::q8_0:launch.template operator()<WeightKind::q8_0>();break;
    case WeightKind::iq4_nl:launch.template operator()<WeightKind::iq4_nl>();break;
    case WeightKind::iq4_xs:launch.template operator()<WeightKind::iq4_xs>();break;
#define STRIX_GGML_KIND_CASE(KIND, type) case WeightKind::KIND:launch.template operator()<WeightKind::KIND>();break;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_CASE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_CASE)
#undef STRIX_GGML_KIND_CASE
    }
    check_hip(hipGetLastError(),"MTP native LM head launch");
}
void launch_hip_q8_head_rows(const std::uint16_t* hidden,const void* q8_rows,std::uint32_t hidden_size,
    std::uint32_t rows,float* logits,void* stream) {
    if(!hidden || !q8_rows || !logits || !stream || !rows || !hidden_size || hidden_size%32U || hidden_size>10240U ||
       reinterpret_cast<std::uintptr_t>(q8_rows)%2U)
        throw std::invalid_argument("Q8 draft head rows geometry/views invalid");
    if(launch_q6_head_r1(hidden,q8_rows,hidden_size,rows,logits,reinterpret_cast<hipStream_t>(stream))) {
        check_hip(hipGetLastError(),"Q6P32 draft head rows launch");
        return;
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(prefetch8_wave_q8_lm_head_kernel<std::uint16_t,8>),
        dim3(rows/8U+static_cast<std::uint32_t>(rows%8U!=0U)),dim3(256U),0,reinterpret_cast<hipStream_t>(stream),
        hidden,q8_rows,hidden_size,rows,logits);
    check_hip(hipGetLastError(),"Q8 draft head rows launch");
}

}  // namespace strix
