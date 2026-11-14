#include "strix/env_flags.hpp"
#include "core/row_layout.hpp"
#include "strix/hip_qwen_dpp_reduce.hpp"
#include "strix/moe_route_bf16.hpp"
#include "strix/hip_mtp_fast_draft.hpp"
#include "strix/hip_mtp_projection.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_moe_experts_launcher.hpp"
#include "strix/hip_qwen_moe_prepared.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/q51_slot_parallel_contract.hpp"
#include "strix/moe_a8_down_contract.hpp"

#include "strix/dispatch_resources.hpp"
#include "strix/known_window_cohort.hpp"
#include "strix/prepared_tensor_views.hpp"
#include "strix/hip_setup_quiescence.hpp"
#ifdef STRIX_BUILD_MOE_BATCHED_ACTIVATION_DIAGNOSTIC
#include "strix/moe_batched_activation_diagnostic.hpp"
#endif
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
#include "strix/hip_moe_q4k_a8_probe.hpp"
#include "strix/hip_moe_q4k_a8_known_rows.hpp"
#include "strix/hip_q51_grouped_down_probe.hpp"
#include "strix/hip_moe_a8_down.hpp"
#endif
// Self-contained format headers (STRIX_GGML_KINDS_GSQ, hip_ggml, gsq_a8) are needed with the A8 runtime OFF too.
#include "strix/hip_ggml_weight_formats.hpp"
#include "strix/gsq_a8_groups.hpp"
#include <array>
#include <functional>
#include <algorithm>
#include <limits>
#include <optional>
#include <mutex>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace strix {
namespace {

constexpr std::uint32_t kThreads = 256;
constexpr std::uint64_t kLauncherTag =
    0x4d4f454558505255ULL;
constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

struct Q8Block {
    __half scale;
    std::int8_t values[32];
};

struct IQ4XSBlock {
    __half d;
    std::uint16_t scales_h;
    std::uint8_t scales_l[4];
    std::uint8_t qs[128];
};

struct IQ4NLBlock {
    __half d;
    std::uint8_t qs[16];
};

struct Q51Block {
    __half d;
    __half m;
    std::uint8_t qh[4];
    std::uint8_t qs[16];
};

struct Q4KBlock {
    __half d;
    __half dmin;
    std::uint8_t scales[12];
    std::uint8_t qs[128];
};

struct Q5KBlock {
    __half d;
    __half dmin;
    std::uint8_t scales[12];
    std::uint8_t qh[32];
    std::uint8_t qs[128];
};

static_assert(sizeof(Q8Block) == 34);
static_assert(sizeof(IQ4XSBlock) == 136);
static_assert(sizeof(IQ4NLBlock) == 18);
static_assert(sizeof(Q51Block) == 24);
static_assert(sizeof(Q4KBlock) == 144);
static_assert(sizeof(Q5KBlock) == 176);

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
    q5_1,
    q4_k,
    q5_k,
#define STRIX_GGML_KIND_ENUM(kind, type) kind,
    STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_ENUM)
#undef STRIX_GGML_KIND_ENUM
};

struct TensorPair {
    const ManifestTensor* manifest = nullptr;
    const RuntimeTensorBinding* runtime = nullptr;
};

struct RoutedTensors {
    TensorPair gate_up;
    TensorPair gate;
    TensorPair up;
    TensorPair down;
    bool fused_gate_up = false;
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
    const HipMoeExpertsLauncherConfig& config) {
    if (config.q5k_a8g32_gate_up &&
        (!config.q4k_a8g32_gate_up || !config.a8_down_q51 || !config.a8_down_q80))
        throw std::invalid_argument("Q5_K A8 gate/up requires the complete Q5_1/Q8_0 A8 down profile");
    if (config.a8_down_q80 && !config.a8_down_q51)
        throw std::invalid_argument("Q8_0 A8 down phase b requires Q5_1 A8 down");
    if (config.a8_down_q51 || config.a8_down_q80) {
        if (!config.q4k_a8g32_gate_up || !config.grouped_exact_q51_down ||
            !config.q51_slot_parallel_down || !config.batched_slot_activation || !config.deferred_moe_status)
            throw std::invalid_argument("A8 down requires explicit A8 batched slot-parallel eager D1 configuration");
        if (config.hidden_size != 2560U || config.intermediate_size != 640U ||
            config.expert_count != 512U || config.top_k != 10U || config.activation != HipActivationStorage::bf16)
            throw std::invalid_argument("A8 down requires canonical H/I/E/K geometry");
    }
    if (config.batched_slot_activation && !config.q4k_a8g32_gate_up)
        throw std::invalid_argument("batched routed activation requires Q4K A8 independent slot workspace");
    if(config.grouped_exact_q51_down && !config.q4k_a8g32_gate_up)
        throw std::invalid_argument("grouped exact Q5_1 down requires Q4K A8 slot workspace");
    if (config.q51_slot_parallel_down && !config.grouped_exact_q51_down)
        throw std::invalid_argument("slot-parallel Q5_1 down requires grouped exact Q5_1 workspace");
    if (config.q4k_a8g32_gate_up) {
#ifndef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
        throw std::invalid_argument("Q4K A8 runtime capability is not built");
#else
        (void)moe_q4k_a8::build_runtime_workspace(config.hidden_size,
            config.intermediate_size, config.expert_count, config.top_k, config.q51_slot_parallel_down,
            config.a8_down_q51 || config.a8_down_q80);
#endif
    }
    if (config.hidden_size == 0 ||
        config.intermediate_size == 0 ||
        config.expert_count == 0 ||
        config.top_k == 0 ||
        config.top_k > config.expert_count) {
        throw std::invalid_argument(
            "HIP routed-expert geometry is invalid");
    }
    if (config.activation !=
            HipActivationStorage::bf16 &&
        config.activation !=
            HipActivationStorage::fp32) {
        throw std::invalid_argument(
            "HIP routed-expert activation storage is unsupported");
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
    case 7: return WeightKind::q5_1;
    case 12: return WeightKind::q4_k;
    case 13: return WeightKind::q5_k;
    case 20: return WeightKind::iq4_nl;
    case 23: return WeightKind::iq4_xs;
    case 30: return WeightKind::bf16;
#define STRIX_GGML_KIND_MAP(kind, type) case type: return WeightKind::kind;
    STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_MAP)
#undef STRIX_GGML_KIND_MAP
    default:
        throw std::invalid_argument(
            "HIP routed experts support F32, F16, BF16, Q8_0, IQ4_NL, IQ4_XS, Q5_1, Q4_K, Q5_K and the GSQ-RCO formats");
    }
}

__device__ float bf16_to_float(
    std::uint16_t bits) {
    return __uint_as_float(
        static_cast<std::uint32_t>(bits) << 16U);
}

__device__ float load_activation(
    const float* value) {
    return *value;
}

__device__ float load_activation(
    const std::uint16_t* value) {
    return bf16_to_float(*value);
}

__device__ void store_activation(
    float* destination,
    float value) {
    *destination = value;
}

__device__ void store_activation(
    std::uint16_t* destination,
    float value) {
    const auto bits =
        __float_as_uint(value);
    const auto rounded =
        bits +
        0x7fffU +
        ((bits >> 16U) & 1U);
    *destination =
        static_cast<std::uint16_t>(
            rounded >> 16U);
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
    std::uint64_t row,
    std::uint32_t feature);

template <>
__device__ float load_weight<WeightKind::f32>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    return static_cast<const float*>(weights)[
        row * row_width + feature];
}

template <>
__device__ float load_weight<WeightKind::f16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const __half*>(weights);
    return __half2float(
        values[row * row_width + feature]);
}

template <>
__device__ float load_weight<WeightKind::bf16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const std::uint16_t*>(weights);
    return bf16_to_float(
        values[row * row_width + feature]);
}

template <>
__device__ float load_weight<WeightKind::q8_0>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const Q8Block*>(weights);
    const auto blocks_per_row =
        row_width / 32U;
    const auto& block =
        blocks[
            row * blocks_per_row +
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
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks = static_cast<const IQ4NLBlock*>(weights);
    const auto& block = blocks[row * (row_width / 32U) + feature / 32U];
    const auto within = feature % 32U;
    const auto packed = block.qs[within % 16U];
    const auto code = within < 16U
        ? static_cast<std::uint8_t>(packed & 0x0fU)
        : static_cast<std::uint8_t>(packed >> 4U);
    return __half2float(block.d) * static_cast<float>(iq4_codebook[code]);
}

template <>
__device__ float load_weight<WeightKind::iq4_xs>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const IQ4XSBlock*>(weights);
    const auto blocks_per_row =
        row_width / 256U;
    const auto& block =
        blocks[
            row * blocks_per_row +
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

template <>
__device__ float load_weight<WeightKind::q5_1>(
    const void* weights, std::uint32_t row_width,
    std::uint64_t row, std::uint32_t feature) {
    const auto* blocks = static_cast<const Q51Block*>(weights);
    const auto& block = blocks[row * (row_width / 32U) + feature / 32U];
    const auto lane = feature % 32U;
    const auto byte = block.qs[lane % 16U];
    const auto low = lane < 16U ? byte & 15U : byte >> 4U;
    const auto high = (block.qh[lane / 8U] >> (lane % 8U)) & 1U;
    return __half2float(block.d) * static_cast<float>(low | (high << 4U)) +
           __half2float(block.m);
}

// K-block groups carry unsigned six-bit scale and minimum factors in twelve
// bytes. The last four groups use the high bits of the first eight bytes.
__device__ void k_group_factors(
    const std::uint8_t* packed, std::uint32_t group,
    std::uint32_t& scale, std::uint32_t& minimum) {
    if (group < 4U) {
        scale = packed[group] & 63U;
        minimum = packed[group + 4U] & 63U;
    } else {
        scale = (packed[group + 4U] & 15U) | ((static_cast<std::uint32_t>(packed[group - 4U]) >> 6U) << 4U);
        minimum = (static_cast<std::uint32_t>(packed[group + 4U]) >> 4U) | ((static_cast<std::uint32_t>(packed[group]) >> 6U) << 4U);
    }
}

template <typename Block>
__device__ float load_k_weight(
    const Block* blocks, std::uint32_t row_width,
    std::uint64_t row, std::uint32_t feature) {
    const auto& block = blocks[row * (row_width / 256U) + feature / 256U];
    const auto lane = feature % 256U;
    const auto group = lane / 32U;
    const auto byte = block.qs[(group / 2U) * 32U + lane % 32U];
    auto code = group % 2U == 0U ? byte & 15U : byte >> 4U;
    if constexpr (sizeof(Block) == sizeof(Q5KBlock)) {
        code |= ((block.qh[lane % 32U] >> group) & 1U) << 4U;
    }
    std::uint32_t scale = 0, minimum = 0;
    k_group_factors(block.scales, group, scale, minimum);
    return (__half2float(block.d) * static_cast<float>(scale)) * static_cast<float>(code) -
           __half2float(block.dmin) * static_cast<float>(minimum);
}

template <>
__device__ float load_weight<WeightKind::q4_k>(
    const void* weights, std::uint32_t row_width,
    std::uint64_t row, std::uint32_t feature) {
    return load_k_weight(static_cast<const Q4KBlock*>(weights), row_width, row, feature);
}

template <>
__device__ float load_weight<WeightKind::q5_k>(
    const void* weights, std::uint32_t row_width,
    std::uint64_t row, std::uint32_t feature) {
    return load_k_weight(static_cast<const Q5KBlock*>(weights), row_width, row, feature);
}

#define STRIX_GGML_KIND_LOAD(kind, type)                                                              template <>                                                                                       __device__ float load_weight<WeightKind::kind>(                                                       const void* weights, std::uint32_t row_width, std::uint64_t row, std::uint32_t feature) {         return hip_ggml::element<type>(weights, row_width, row, feature);                            }
STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_LOAD)
#undef STRIX_GGML_KIND_LOAD

std::uint32_t ggml_type_of(WeightKind kind) {
    switch (kind) {
#define STRIX_GGML_KIND_TYPE(k, type) case WeightKind::k: return type;
    STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_TYPE)
#undef STRIX_GGML_KIND_TYPE
    default: return 0;
    }
}

std::uint64_t required_weight_bytes(
    WeightKind kind,
    std::uint32_t input,
    std::uint64_t output_rows,
    std::uint32_t experts) {
    std::uint64_t row_bytes = 0;
    switch (kind) {
    case WeightKind::f32:
        row_bytes =
            static_cast<std::uint64_t>(input) * 4ULL;
        break;
    case WeightKind::f16:
    case WeightKind::bf16:
        row_bytes =
            static_cast<std::uint64_t>(input) * 2ULL;
        break;
    case WeightKind::q8_0:
        if (input % 32U != 0) {
            throw std::invalid_argument(
                "Q8_0 routed-expert input width must be divisible by 32");
        }
        row_bytes =
            static_cast<std::uint64_t>(
                input / 32U) *
            sizeof(Q8Block);
        break;
    case WeightKind::iq4_xs:
        if (input % 256U != 0) {
            throw std::invalid_argument(
                "IQ4_XS routed-expert input width must be divisible by 256");
        }
        row_bytes =
            static_cast<std::uint64_t>(
                input / 256U) *
            sizeof(IQ4XSBlock);
        break;
    case WeightKind::iq4_nl:
        if (input % 32U != 0) {
            throw std::invalid_argument(
                "IQ4_NL routed-expert input width must be divisible by 32");
        }
        row_bytes = static_cast<std::uint64_t>(input / 32U) * sizeof(IQ4NLBlock);
        break;
    case WeightKind::q5_1:
        if (input % 32U != 0) {
            throw std::invalid_argument("Q5_1 routed-expert input width must be divisible by 32");
        }
        row_bytes = static_cast<std::uint64_t>(input / 32U) * sizeof(Q51Block);
        break;
    case WeightKind::q4_k:
    case WeightKind::q5_k:
        if (input % 256U != 0) {
            throw std::invalid_argument("K-quant routed-expert input width must be divisible by 256");
        }
        row_bytes = static_cast<std::uint64_t>(input / 256U) *
            (kind == WeightKind::q4_k ? sizeof(Q4KBlock) : sizeof(Q5KBlock));
        break;
    default: {
        const auto g = hip_ggml::geometry(ggml_type_of(kind));
        if (g.elements == 0 || input % g.elements != 0) {
            throw std::invalid_argument("routed-expert input width is not a multiple of its GGUF block");
        }
        row_bytes = static_cast<std::uint64_t>(input / g.elements) * g.bytes;
        break;
    }
    }
    return row_bytes *
           output_rows *
           experts;
}

template <
    typename Input,
    WeightKind Kind>
__global__ void selected_projection_kernel(
    const Input* input,
    const std::uint32_t* route_indices,
    std::uint32_t slot,
    const void* weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    std::uint32_t expert_count,
    float* output, const std::uint32_t* status) {
    if (*status != 0U) return;
    const auto out =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (out >= output_width) {
        return;
    }
    const auto expert =
        route_indices[slot];
    if (expert >= expert_count) {
        return;
    }

    __shared__ float scratch[kThreads];
    float sum = 0.0F;
    const auto row =
        static_cast<std::uint64_t>(expert) *
            output_width +
        out;

    for (std::uint32_t feature =
             static_cast<std::uint32_t>(
                 threadIdx.x);
         feature < input_width;
         feature += kThreads) {
        sum +=
            load_activation(
                &input[feature]) *
            load_weight<Kind>(
                weights,
                input_width,
                row,
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
        output[out] =
            scratch[0];
    }
}

__global__ void routed_activation_kernel(
    float* workspace,
    std::uint32_t intermediate_size, const std::uint32_t* status) {
    if (*status != 0U) return;
    const auto index =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (index >= intermediate_size) {
        return;
    }
    const float gate =
        workspace[index];
    const float up =
        workspace[
            intermediate_size +
            index];
    workspace[index] =
        (gate /
         (1.0F + expf(-gate))) *
        up;
}

#if defined(STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME) || defined(STRIX_BUILD_MOE_BATCHED_ACTIVATION_DIAGNOSTIC)
// Each y block owns one pre-existing [gate,up] slot. Arithmetic and x indexing
// remain the original routed_activation_kernel's; status is immutable here.
__global__ void batched_routed_activation_kernel(
    float* slots,
    std::uint32_t intermediate_size, const std::uint32_t* status) {
    if (*status != 0U) return;
    const auto index =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (index >= intermediate_size) {
        return;
    }
    auto* workspace = slots + 2ULL * blockIdx.y * intermediate_size;
    const float gate =
        workspace[index];
    const float up =
        workspace[
            intermediate_size +
            index];
    workspace[index] =
        (gate /
         (1.0F + expf(-gate))) *
        up;
}
#endif

__global__ void validate_routes_kernel(
    const std::uint32_t* routes, const float* weights,
    std::uint32_t count, std::uint32_t experts,
    std::uint32_t* status, std::uint32_t* graph_status) {
    for (std::uint32_t slot = threadIdx.x; slot < count; slot += blockDim.x) {
        if (routes[slot] >= experts || !isfinite(weights[slot])) {
            atomicOr(status, 1U);
            if (graph_status) atomicOr(graph_status, 16U);
        }
    }
}

template <typename Storage>
__global__ void commit_experts_kernel(
    const float* accumulation, Storage* output,
    std::uint32_t count, const std::uint32_t* status) {
    if (*status != 0U) return;
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) store_activation(&output[index], accumulation[index]);
}

template <
    typename Output,
    WeightKind Kind>
__global__ void selected_down_accumulate_kernel(
    const float* intermediate,
    const std::uint32_t* route_indices,
    const float* route_weights,
    std::uint32_t slot,
    const void* weights,
    std::uint32_t intermediate_size,
    std::uint32_t hidden_size,
    std::uint32_t expert_count,
    Output* output, const std::uint32_t* status) {
    if (*status != 0U) return;
    const auto out =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (out >= hidden_size) {
        return;
    }
    const auto expert =
        route_indices[slot];
    if (expert >= expert_count) {
        return;
    }

    __shared__ float scratch[kThreads];
    float sum = 0.0F;
    const auto row =
        static_cast<std::uint64_t>(expert) *
            hidden_size +
        out;

    for (std::uint32_t feature =
             static_cast<std::uint32_t>(
                 threadIdx.x);
         feature < intermediate_size;
         feature += kThreads) {
        sum +=
            intermediate[feature] *
            load_weight<Kind>(
                weights,
                intermediate_size,
                row,
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
        const float current =
            load_activation(
                &output[out]);
        store_activation(
            &output[out],
            current +
                scratch[0] *
                    route_weights[slot]);
    }
}

bool contains(
    std::string_view text,
    std::string_view needle) {
    return text.find(needle) !=
        std::string_view::npos;
}

TensorPair tensor_pair(
    const ExecutionDispatchContext& context,
    std::uint32_t index) {
    if (context.resources == nullptr ||
        context.resources->model_manifest ==
            nullptr) {
        throw std::invalid_argument(
            "HIP routed experts require model manifest");
    }
    if (index >=
        context.resources->
            model_manifest->
            tensors.size()) {
        throw std::out_of_range(
            "routed-expert tensor index exceeds manifest");
    }
    return TensorPair{
        &context.resources->
            model_manifest->
            tensors[index],
        &resolve_dispatch_tensor(
            context,
            index),
    };
}

RoutedTensors identify_tensors(
    const ExecutionDispatchContext& context,
    const ExecutionTensorAccess& access,
    const HipMoeExpertsLauncherConfig& config) {
    if(const auto* cached=context.prepared_tensor_views) {
        if(!context.resources) throw std::invalid_argument("prepared routed roles require retained resources");
        cached->validate_stage_scope(context.resources->distributed_stage,context.step_id);
        cached->validate_source_handles(context.resources->model_manifest,context.resources->model_weights,context.resources->tensor_access);
        const auto& r=cached->routed_roles(context.execution_identity,context.step_id,config.hidden_size,config.intermediate_size,config.expert_count);
        RoutedTensors result;result.fused_gate_up=r.fused_gate_up;result.down=tensor_pair(context,r.down);
        if(r.fused_gate_up) result.gate_up=tensor_pair(context,r.gate_up);
        else {result.gate=tensor_pair(context,r.gate);result.up=tensor_pair(context,r.up);}
        return result;
    }
    RoutedTensors result;
    std::uint32_t fused = 0;
    std::uint32_t gate = 0;
    std::uint32_t up = 0;
    std::uint32_t down = 0;

    for (const auto index :
         access.tensor_indices) {
        const auto pair =
            tensor_pair(
                context,
                index);
        const auto& tensor =
            *pair.manifest;
        if (tensor.role !=
            TensorRole::moe_expert ||
            tensor.dimensions.size() < 3 ||
            tensor.dimensions[2] !=
                config.expert_count) {
            continue;
        }

        if (tensor.dimensions[0] ==
                config.hidden_size &&
            tensor.dimensions[1] ==
                2ULL *
                    config.intermediate_size &&
            contains(
                tensor.name,
                "gate_up")) {
            result.gate_up = pair;
            ++fused;
            continue;
        }

        if (tensor.dimensions[0] ==
                config.hidden_size &&
            tensor.dimensions[1] ==
                config.intermediate_size) {
            if (contains(
                    tensor.name,
                    "gate") &&
                !contains(
                    tensor.name,
                    "gate_up")) {
                result.gate = pair;
                ++gate;
                continue;
            }
            if (contains(
                    tensor.name,
                    "up")) {
                result.up = pair;
                ++up;
                continue;
            }
        }

        if (tensor.dimensions[0] ==
                config.intermediate_size &&
            tensor.dimensions[1] ==
                config.hidden_size &&
            contains(
                tensor.name,
                "down")) {
            result.down = pair;
            ++down;
        }
    }

    if (down != 1) {
        throw std::invalid_argument(
            "routed expert down tensor is missing or ambiguous");
    }
    if (fused == 1 &&
        gate == 0 &&
        up == 0) {
        result.fused_gate_up = true;
        return result;
    }
    if (fused == 0 &&
        gate == 1 &&
        up == 1) {
        result.fused_gate_up = false;
        return result;
    }

    throw std::invalid_argument(
        "routed expert gate/up tensor family is missing or ambiguous");
}

void validate_tensor(
    const TensorPair& pair,
    std::uint32_t input,
    std::uint64_t output,
    std::uint32_t experts,
    const char* name) {
    if (pair.manifest == nullptr ||
        pair.runtime == nullptr) {
        throw std::invalid_argument(
            std::string(name) +
            " tensor is missing");
    }
    if (!runtime_tensor_gpu_visible(pair.runtime->storage) ||
        pair.runtime->address == 0) {
        throw std::invalid_argument(
            std::string(name) +
            " tensor is not GPU-visible");
    }

    const auto required =
        required_weight_bytes(
            weight_kind(
                pair.manifest->ggml_type),
            input,
            output,
            experts);
    if (pair.runtime->bytes <
        required) {
        throw std::invalid_argument(
            std::string(name) +
            " tensor binding is undersized");
    }
}

template <typename Input>
void launch_projection_by_kind(
    WeightKind kind,
    const Input* input,
    const std::uint32_t* routes,
    std::uint32_t slot,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    std::uint32_t experts,
    float* output, const std::uint32_t* status) {
#define STRIX_PROJECT_CASE(kind_name)     hipLaunchKernelGGL(         HIP_KERNEL_NAME(             selected_projection_kernel<                 Input, WeightKind::kind_name>),         dim3(output_width), dim3(kThreads), 0, hip_execution_stream(),         input, routes, slot,         reinterpret_cast<const void*>(weights.address),         input_width, output_width, experts, output, status)
    switch (kind) {
    case WeightKind::f32: STRIX_PROJECT_CASE(f32); break;
    case WeightKind::f16: STRIX_PROJECT_CASE(f16); break;
    case WeightKind::bf16: STRIX_PROJECT_CASE(bf16); break;
    case WeightKind::q8_0: STRIX_PROJECT_CASE(q8_0); break;
    case WeightKind::iq4_nl: STRIX_PROJECT_CASE(iq4_nl); break;
    case WeightKind::iq4_xs: STRIX_PROJECT_CASE(iq4_xs); break;
    case WeightKind::q5_1: STRIX_PROJECT_CASE(q5_1); break;
    case WeightKind::q4_k: STRIX_PROJECT_CASE(q4_k); break;
    case WeightKind::q5_k: STRIX_PROJECT_CASE(q5_k); break;
#define STRIX_GGML_KIND_CASE(k, type) case WeightKind::k: STRIX_PROJECT_CASE(k); break;
    STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_CASE)
#undef STRIX_GGML_KIND_CASE
    }
#undef STRIX_PROJECT_CASE
    check_hip(
        hipGetLastError(),
        "HIP routed-expert projection launch");
}

template <typename Output>
void launch_down_by_kind(
    WeightKind kind,
    const float* intermediate,
    const std::uint32_t* routes,
    const float* route_weights,
    std::uint32_t slot,
    const RuntimeTensorBinding& weights,
    std::uint32_t intermediate_size,
    std::uint32_t hidden_size,
    std::uint32_t experts,
    Output* output, const std::uint32_t* status) {
#define STRIX_DOWN_CASE(kind_name)     hipLaunchKernelGGL(         HIP_KERNEL_NAME(             selected_down_accumulate_kernel<                 Output, WeightKind::kind_name>),         dim3(hidden_size), dim3(kThreads), 0, hip_execution_stream(),         intermediate, routes, route_weights, slot,         reinterpret_cast<const void*>(weights.address),         intermediate_size, hidden_size, experts, output, status)
    switch (kind) {
    case WeightKind::f32: STRIX_DOWN_CASE(f32); break;
    case WeightKind::f16: STRIX_DOWN_CASE(f16); break;
    case WeightKind::bf16: STRIX_DOWN_CASE(bf16); break;
    case WeightKind::q8_0: STRIX_DOWN_CASE(q8_0); break;
    case WeightKind::iq4_nl: STRIX_DOWN_CASE(iq4_nl); break;
    case WeightKind::iq4_xs: STRIX_DOWN_CASE(iq4_xs); break;
    case WeightKind::q5_1: STRIX_DOWN_CASE(q5_1); break;
    case WeightKind::q4_k: STRIX_DOWN_CASE(q4_k); break;
    case WeightKind::q5_k: STRIX_DOWN_CASE(q5_k); break;
#define STRIX_GGML_KIND_CASE(k, type) case WeightKind::k: STRIX_DOWN_CASE(k); break;
    STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_CASE)
#undef STRIX_GGML_KIND_CASE
    }
#undef STRIX_DOWN_CASE
    check_hip(
        hipGetLastError(),
        "HIP routed-expert down launch");
}

__global__ void publish_deferred_moe_first_error(const std::uint32_t* local,
    std::uint32_t* record, std::uint32_t domain) {
    // One thread, one stream: completion of this node precedes the next layer's
    // workspace reset. Existing arithmetic kernels and status words are intact.
    merge_deferred_moe_first_error(*reinterpret_cast<DeferredMoeFirstError*>(record),
        *local, static_cast<DeferredMoeDomain>(domain));
}
void enqueue_deferred_moe_status(const std::uint32_t* local, DeferredMoeDomain domain) {
    hipLaunchKernelGGL(publish_deferred_moe_first_error,dim3(1),dim3(1),0,hip_execution_stream(),
        local,hip_deferred_moe_status(),static_cast<std::uint32_t>(domain));
    check_hip(hipGetLastError(),"publish deferred MoE first-error record");
}

template <typename Storage>
void launch_typed(
    const HipMoeExpertsLauncherConfig& config,
    const ExecutionBufferAddress& hidden,
    const ExecutionBufferAddress& route_indices,
    const ExecutionBufferAddress& route_weights,
    const ExecutionBufferAddress& output,
    const ExecutionBufferAddress& temporary,
    const RoutedTensors& tensors) {
    const auto* input =
        reinterpret_cast<const Storage*>(
            hidden.address);
    const auto* routes =
        reinterpret_cast<const std::uint32_t*>(
            route_indices.address);
    const auto* weights =
        reinterpret_cast<const float*>(
            route_weights.address);
    auto* workspace =
        reinterpret_cast<float*>(
            temporary.address);
    auto* accumulation = workspace + 2ULL * config.intermediate_size;
    auto* status = reinterpret_cast<std::uint32_t*>(accumulation + config.hidden_size);

    check_hip(
        hipMemsetAsync(
            accumulation,
            0,
            static_cast<std::size_t>(config.hidden_size) * sizeof(float) + sizeof(std::uint32_t),
            hip_execution_stream()),
        "clear routed-expert FP32 accumulation and status");
    hipLaunchKernelGGL(validate_routes_kernel, dim3(1), dim3(kThreads), 0,
        hip_execution_stream(), routes, weights, config.top_k,
        config.expert_count, status, hip_graph_status());
    check_hip(hipGetLastError(), "validate routed-expert routes");

    for (std::uint32_t slot = 0;
         slot < config.top_k;
         ++slot) {
        if (tensors.fused_gate_up) {
            launch_projection_by_kind(
                weight_kind(
                    tensors.gate_up.
                        manifest->ggml_type),
                input,
                routes,
                slot,
                *tensors.gate_up.runtime,
                config.hidden_size,
                2U *
                    config.intermediate_size,
                config.expert_count,
                workspace, status);
        } else {
            launch_projection_by_kind(
                weight_kind(
                    tensors.gate.
                        manifest->ggml_type),
                input,
                routes,
                slot,
                *tensors.gate.runtime,
                config.hidden_size,
                config.intermediate_size,
                config.expert_count,
                workspace, status);
            launch_projection_by_kind(
                weight_kind(
                    tensors.up.
                        manifest->ggml_type),
                input,
                routes,
                slot,
                *tensors.up.runtime,
                config.hidden_size,
                config.intermediate_size,
                config.expert_count,
                workspace +
                    config.intermediate_size, status);
        }

        const auto blocks =
            (config.intermediate_size +
             kThreads - 1U) /
            kThreads;
        hipLaunchKernelGGL(
            routed_activation_kernel,
            dim3(blocks),
            dim3(kThreads),
            0, hip_execution_stream(),
            workspace,
            config.intermediate_size, status);
        check_hip(
            hipGetLastError(),
            "HIP routed-expert activation launch");

        launch_down_by_kind(
            weight_kind(
                tensors.down.manifest->
                    ggml_type),
            workspace,
            routes,
            weights,
            slot,
            *tensors.down.runtime,
            config.intermediate_size,
            config.hidden_size,
            config.expert_count,
            accumulation, status);
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(commit_experts_kernel<Storage>),
        dim3((config.hidden_size + kThreads - 1U) / kThreads), dim3(kThreads),
        0, hip_execution_stream(), accumulation,
        reinterpret_cast<Storage*>(output.address), config.hidden_size, status);
    check_hip(hipGetLastError(), "commit routed-expert output");
    // Graph replay consumes the existing aggregate status at its completion
    // boundary. Eager C1 retains scratch until this stream has drained.
    if (config.deferred_moe_status) {
        enqueue_deferred_moe_status(status,DeferredMoeDomain::legacy);
    } else if (hip_graph_status() == nullptr) {
        std::uint32_t host_status = 0;
        check_hip(hipMemcpyAsync(&host_status, status, sizeof(host_status),
            hipMemcpyDeviceToHost, hip_execution_stream()), "read routed-expert status");
        check_hip(hipStreamSynchronize(hip_execution_stream()), "drain routed-expert validation");
        if (host_status != 0U) throw std::invalid_argument("HIP routed experts rejected invalid routes or nonfinite weights");
    }
}


#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
// Integer classification avoids finite-math assumptions and never mutates the
// graph aggregate except by OR. Each CTA snapshots the sticky status atomically
// before its barrier; bad producers cannot expose indeterminate scratch values.
__global__ void q4k_a8_validate_finite_kernel(const float* values, std::uint32_t count,
    std::uint32_t* status, std::uint32_t* aggregate, bool bf16_commit) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ std::uint32_t initial_status;
    if (threadIdx.x == 0U) initial_status = atomicOr(status,0U);
    __syncthreads();
    if (initial_status != 0U) {
        if (index == 0U && aggregate != nullptr) atomicOr(aggregate, 16U);
        return;
    }
    if (index < count && (__float_as_uint(values[index]) & 0x7fffffffU) >=
        (bf16_commit ? 0x7f7f8000U : 0x7f800000U)) {
        atomicOr(status, 8U);
        if (aggregate != nullptr) atomicOr(aggregate, 16U);
    }
}

// One complete wave owns an output. Eight register leaves represent the
// ORIGINAL256 threads, not a32-leaf reduction. All shuffle participants stay
// active; only complete output waves can return at the output-tail guard.
__global__ void q51_grouped_virtual256_down_probe_kernel(const float* intermediates,
    const std::uint32_t* routes,const float* route_weights,const void* weights,
    std::uint32_t intermediate_size,std::uint32_t hidden_size,std::uint32_t experts,
    std::uint32_t top_k,float* accumulation,const std::uint32_t* status) {
    if(*status!=0U) return; // No writer changes local status during this kernel.
    const auto lane=static_cast<std::uint32_t>(threadIdx.x)%32U;
    const auto wave=static_cast<std::uint32_t>(threadIdx.x)/32U;
    const auto out=static_cast<std::uint32_t>(blockIdx.x)*8U+wave;
    if(out>=hidden_size) return;
    float current=0.0F;
    for(std::uint32_t slot=0;slot<top_k;++slot) {
        const auto expert=routes[slot];
        if(expert>=experts) return; // Same-stream validator already sets bit1.
        const auto row=static_cast<std::uint64_t>(expert)*hidden_size+out;
        const auto input_base=static_cast<std::uint64_t>(slot)*2U*intermediate_size;
        float partial[8]={};
        for(std::uint32_t chunk=0;chunk<intermediate_size;chunk+=256U) {
#pragma unroll
            for(unsigned j=0;j<8;++j) {
                const auto feature=chunk+8U*lane+j;
                if(feature<intermediate_size)
                    partial[j]+=intermediates[input_base+feature]*
                        load_weight<WeightKind::q5_1>(weights,intermediate_size,row,feature);
            }
        }
#pragma unroll
        for(unsigned distance=16;distance>0;distance>>=1U) {
#pragma unroll
            for(unsigned j=0;j<8;++j) {
                const auto neighbor=strix::qwen_dpp::down_tree32(partial[j], distance);
                if(lane<distance) partial[j]+=neighbor;
            }
        }
        if(lane==0U) {
#pragma unroll
            for(unsigned j=0;j<4;++j) partial[j]+=partial[j+4];
#pragma unroll
            for(unsigned j=0;j<2;++j) partial[j]+=partial[j+2];
            partial[0]+=partial[1];
            current=current+partial[0]*route_weights[slot];
        }
    }
    if(lane==0U) accumulation[out]=current;
}

// Each complete wave owns exactly the incumbent's eight virtual leaves for
// one (slot, output). Store the unweighted dot without arithmetic conversion.
__global__ void q51_slot_parallel_dot_kernel(const float* intermediates,
    const std::uint32_t* routes,const void* weights,
    std::uint32_t intermediate_size,std::uint32_t hidden_size,std::uint32_t experts,
    float* rawdots,const std::uint32_t* status) {
    if(*status!=0U) return;
    const auto lane=static_cast<std::uint32_t>(threadIdx.x)%32U;
    const auto wave=static_cast<std::uint32_t>(threadIdx.x)/32U;
    const auto out=static_cast<std::uint32_t>(blockIdx.x)*8U+wave;
    if(out>=hidden_size) return;
    const auto slot=static_cast<std::uint32_t>(blockIdx.y);
    const auto expert=routes[slot];
    if(expert>=experts) return;
    const auto row=static_cast<std::uint64_t>(expert)*hidden_size+out;
    const auto input_base=static_cast<std::uint64_t>(slot)*2U*intermediate_size;
    float partial[8]={};
    for(std::uint32_t chunk=0;chunk<intermediate_size;chunk+=256U) {
#pragma unroll
        for(unsigned j=0;j<8;++j) {
            const auto feature=chunk+8U*lane+j;
            if(feature<intermediate_size)
                partial[j]+=intermediates[input_base+feature]*
                    load_weight<WeightKind::q5_1>(weights,intermediate_size,row,feature);
        }
    }
#pragma unroll
    for(unsigned distance=16;distance>0;distance>>=1U) {
#pragma unroll
        for(unsigned j=0;j<8;++j) {
            const auto neighbor=strix::qwen_dpp::down_tree32(partial[j], distance);
            if(lane<distance) partial[j]+=neighbor;
        }
    }
    if(lane==0U) {
#pragma unroll
        for(unsigned j=0;j<4;++j) partial[j]+=partial[j+4];
#pragma unroll
        for(unsigned j=0;j<2;++j) partial[j]+=partial[j+2];
        partial[0]+=partial[1];
        rawdots[static_cast<std::uint64_t>(slot)*hidden_size+out]=partial[0];
    }
}

__global__ void q51_slot_parallel_merge_kernel(const float* rawdots,
    const std::uint32_t* routes,const float* route_weights,
    std::uint32_t hidden_size,std::uint32_t experts,std::uint32_t top_k,
    float* accumulation,const std::uint32_t* status) {
    if(*status!=0U) return;
    const auto out=static_cast<std::uint32_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(out>=hidden_size) return;
    float current=0.0F;
    for(std::uint32_t slot=0;slot<top_k;++slot) {
        if(routes[slot]>=experts) return;
        const float weight=route_weights[slot];
        const float dot=rawdots[static_cast<std::uint64_t>(slot)*hidden_size+out];
        // Actual incumbent merge roles: weight src0, dot src1, accumulator src2.
        // The read/write operand also preserves the positive-zero initial seed.
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(current) : "s"(weight), "v"(dot));
    }
    accumulation[out]=current;
}

// One serial row with A8 down (STRIX_MOE_R1_GLUE_FUSE): q51_slot_parallel_merge_kernel, q4k_a8_validate_finite_kernel,
// commit_experts_kernel and publish_deferred_moe_first_error in one CTA. Per output the same ordered v_fmac merge (the
// accumulation word written only where the merge kernel writes it), the same finite threshold over the accumulation,
// status |= 8 when any output fails it (only from a clean status), the commit only from a clean final status, and the
// first-error record merged from the final status. Eager only (no graph aggregate).
template <typename Storage>
__global__ __launch_bounds__(1024) void q4k_a8_serial_tail_kernel(const float* rawdots, const std::uint32_t* routes,
    const float* route_weights, std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k, float* accumulation,
    Storage* output, std::uint32_t* status, bool bf16_commit, DeferredMoeFirstError* record, std::uint32_t domain) {
    __shared__ std::uint32_t initial, nonfinite;
    if (threadIdx.x == 0U) { initial = *status; nonfinite = 0U; }
    __syncthreads();
    if (initial == 0U) {
        for (auto out = static_cast<std::uint32_t>(threadIdx.x); out < hidden_size; out += blockDim.x) {
            float current = 0.0F;
            bool written = true;
            for (std::uint32_t slot = 0; slot < top_k; ++slot) {
                if (routes[slot] >= experts) { written = false; break; }
                const float weight = route_weights[slot];
                const float dot = rawdots[static_cast<std::uint64_t>(slot) * hidden_size + out];
                asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(current) : "s"(weight), "v"(dot));
            }
            if (written) accumulation[out] = current;
            const float value = written ? current : accumulation[out];
            if ((__float_as_uint(value) & 0x7fffffffU) >= (bf16_commit ? 0x7f7f8000U : 0x7f800000U)) atomicOr(&nonfinite, 1U);
        }
    }
    __syncthreads();
    const auto final_status = initial | (nonfinite ? 8U : 0U);
    if (final_status == 0U)
        for (auto out = static_cast<std::uint32_t>(threadIdx.x); out < hidden_size; out += blockDim.x)
            store_activation(&output[out], accumulation[out]);
    if (threadIdx.x == 0U) {
        if (nonfinite) atomicOr(status, 8U);
        merge_deferred_moe_first_error(*record, final_status, static_cast<DeferredMoeDomain>(domain));
    }
}
// STRIX_MOE_R1_GLUE_FUSE (default on): the serial Q4K A8 experts step with A8 down runs its glue in two launches
// (moe_q4k_a8::enqueue_serial_prologue_probe and q4k_a8_serial_tail_kernel) instead of nine (accumulation/status clear,
// validate_routes_kernel, the probe's route check and quantizer, merge, finite scan, commit and the deferred status
// publication). Eager deferred-status execution only. Off with STRIX_MOE_R1_GLUE_FUSE=0, under the baseline and
// STRIX_QPF2K_DEFAULTS=0.
bool moe_r1_glue_fuse_enabled() {
    static const bool on = qpf2k::long_context_item_enabled("STRIX_MOE_R1_GLUE_FUSE", false);
    return on && !qpf2k::baseline;
}
void enqueue_serial_tail_for_test(bool fused, const float* rawdots, const std::uint32_t* routes,
    const float* route_weights, std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k,
    float* accumulation, std::uint16_t* output, std::uint32_t* status, void* record, hipStream_t stream) {
    auto* first_error = reinterpret_cast<DeferredMoeFirstError*>(record);
    if (fused) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(q4k_a8_serial_tail_kernel<std::uint16_t>), dim3(1), dim3(1024), 0, stream, rawdots, routes,
            route_weights, hidden_size, experts, top_k, accumulation, output, status, true, first_error,
            static_cast<std::uint32_t>(DeferredMoeDomain::q4k_a8));
    } else {
        hipLaunchKernelGGL(q51_slot_parallel_merge_kernel, dim3((hidden_size + kThreads - 1U) / kThreads), dim3(kThreads), 0, stream,
            rawdots, routes, route_weights, hidden_size, experts, top_k, accumulation, status);
        hipLaunchKernelGGL(q4k_a8_validate_finite_kernel, dim3((hidden_size + kThreads - 1U) / kThreads), dim3(kThreads), 0, stream,
            accumulation, hidden_size, status, nullptr, true);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(commit_experts_kernel<std::uint16_t>), dim3((hidden_size + kThreads - 1U) / kThreads),
            dim3(kThreads), 0, stream, accumulation, output, hidden_size, status);
        hipLaunchKernelGGL(publish_deferred_moe_first_error, dim3(1), dim3(1), 0, stream, status,
            reinterpret_cast<std::uint32_t*>(record), static_cast<std::uint32_t>(DeferredMoeDomain::q4k_a8));
    }
    check_hip(hipGetLastError(), "Q4K A8 serial tail test enqueue");
}
void require_q51_slot_eager_stream(hipStream_t stream) {
    if (hip_graph_status()!=nullptr)
        throw std::invalid_argument("slot-parallel Q5_1 down admits eager execution only");
    hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
    check_hip(hipStreamIsCapturing(stream,&capture),"query slot-parallel Q5_1 capture status");
    if (capture!=hipStreamCaptureStatusNone)
        throw std::invalid_argument("slot-parallel Q5_1 down admits eager execution only");
}

struct Q4kA8Range { std::uintptr_t address; std::uint64_t bytes; };
void validate_q4k_a8_ranges(Q4kA8Range temporary, Q4kA8Range output,
    const std::array<Q4kA8Range, 6>& readonly) {
    const auto end = [](Q4kA8Range r) {
        if (r.address == 0 || r.bytes == 0 || r.bytes > std::numeric_limits<std::uintptr_t>::max() - r.address)
            throw std::invalid_argument("Q4K A8 invalid runtime address range");
        return r.address + static_cast<std::uintptr_t>(r.bytes);
    };
    const auto disjoint = [&](Q4kA8Range a, Q4kA8Range c) {
        if (a.address < end(c) && c.address < end(a))
            throw std::invalid_argument("Q4K A8 runtime writable range aliases another owner");
    };
    (void)end(temporary); (void)end(output);
    if (temporary.address % 256 != 0)
        throw std::invalid_argument("Q4K A8 runtime temporary requires absolute 256-byte alignment");
    disjoint(temporary,output);
    for (const auto r : readonly) { disjoint(temporary,r); disjoint(output,r); }
}

struct PreparedQ4kA8 {
    moe_q4k_a8::RuntimeWorkspace layout;
    moe_q4k_a8::Request request;
    moe_q4k_a8::Bindings bindings;
    bool grouped_q51, slot_parallel_q51;
    std::optional<moe_q4k_a8::DownRequest> a8_down_request;
    moe_q4k_a8::DownBindings a8_down_bindings;
};

PreparedQ4kA8 prepare_q4k_a8(const HipMoeExpertsLauncherConfig& config,
    const ExecutionBufferAddress& hidden, const ExecutionBufferAddress& route_indices,
    const ExecutionBufferAddress& route_weights, const ExecutionBufferAddress& output,
    const ExecutionBufferAddress& temporary, const RoutedTensors& tensors) {
    if (!config.q4k_a8_quiescence)
        throw std::invalid_argument("Q4K A8 runtime requires setup completion guard");
    const auto layout = moe_q4k_a8::build_runtime_workspace(config.hidden_size,
        config.intermediate_size, config.expert_count, config.top_k, config.q51_slot_parallel_down,
        config.a8_down_q51 || config.a8_down_q80);
    if (temporary.bytes < layout.total_bytes)
        throw std::invalid_argument("Q4K A8 runtime temporary is undersized");
    const auto gate_type = tensors.gate.manifest->ggml_type;
    if (gate_type == 13U && !config.q5k_a8g32_gate_up)
        throw std::invalid_argument("Q5_K A8 gate/up requires its explicit candidate");
    const auto gate_bytes = required_weight_bytes(weight_kind(gate_type), config.hidden_size,
        config.intermediate_size, config.expert_count);
    // Validate entire runtime owners, not just the exact probe slices. Read-only
    // owners can alias each other; scratch and semantic output cannot alias any.
    validate_q4k_a8_ranges({temporary.address, temporary.bytes},
        {output.address, output.bytes}, {{{hidden.address,hidden.bytes},
        {route_indices.address,route_indices.bytes}, {route_weights.address,route_weights.bytes},
        {tensors.gate.runtime->address,tensors.gate.runtime->bytes}, {tensors.up.runtime->address,tensors.up.runtime->bytes},
        {tensors.down.runtime->address,tensors.down.runtime->bytes}}});
    moe_q4k_a8::Request request;
    request.enabled = true;
    request.approved_numerical_identity = moe_q4k_a8::numerical_identity_for_gate_type(gate_type);
    request.input = config.activation == HipActivationStorage::bf16 ?
        moe_q4k_a8::InputKind::bf16 : moe_q4k_a8::InputKind::fp32;
    request.shape = {config.hidden_size, config.intermediate_size, config.expert_count, config.top_k, 1};
    request.gate_type = tensors.gate.manifest->ggml_type;
    request.up_type = tensors.up.manifest->ggml_type;
    request.fused_gate_up = tensors.fused_gate_up;
    const auto slice = [&](std::uint64_t offset, std::uint64_t bytes) {
        return moe_q4k_a8::DeviceBytes{temporary.address + static_cast<std::uintptr_t>(offset), bytes};
    };
    const moe_q4k_a8::Bindings bindings{
        {hidden.address, static_cast<std::uint64_t>(config.hidden_size) * static_cast<std::uint32_t>(config.activation)},
        {tensors.gate.runtime->address,gate_bytes}, {tensors.up.runtime->address,gate_bytes},
        {route_indices.address, static_cast<std::uint64_t>(config.top_k) * sizeof(std::uint32_t)},
        slice(layout.codes_offset,layout.codes_bytes), slice(layout.deltas_offset,layout.deltas_bytes),
        slice(layout.sums_offset,layout.sums_bytes), slice(layout.status_offset,4),
        slice(layout.gate_up_offset,layout.gate_up_bytes)};
    // All admission is complete before even the initial clear is submitted.
    moe_q4k_a8::validate(request,bindings);
    const auto down_type = tensors.down.manifest->ggml_type;
    const bool integer_down = (config.a8_down_q51 && down_type == 7U) ||
        (config.a8_down_q80 && down_type == 8U) ||
        (config.a8_down_q80 && gsq_a8::down_type(down_type) && moe_q4k_a8::q20_a8_enabled());
    if (gsq_a8::gate_type(gate_type) && !integer_down)
        throw std::invalid_argument("Q2_0 A8 gate/up requires the A8 down profile");
    const bool grouped_q51=config.grouped_exact_q51_down && down_type==7U && !integer_down;
    const bool slot_parallel_q51=grouped_q51 && config.q51_slot_parallel_down;
    if(grouped_q51) {
        const q51_grouped_down::Request q51_request{true,q51_grouped_down::numerical_identity,
            {config.hidden_size,config.intermediate_size,config.expert_count,config.top_k}};
        const auto n=q51_grouped_down::required_bytes(q51_request.shape);
        q51_grouped_down::validate(q51_request,{
            {bindings.output.address,n.intermediates},{tensors.down.runtime->address,n.weights},
            {route_indices.address,n.routes},{route_weights.address,n.route_weights},
            {bindings.status.address,n.status},{temporary.address+layout.accumulation_offset,n.accumulation}});
        if (slot_parallel_q51) {
            const q51_slot_parallel::Request parallel_request{true,q51_slot_parallel::numerical_identity,q51_request.shape};
            q51_slot_parallel::validate(parallel_request,{
                {bindings.output.address,n.intermediates},{tensors.down.runtime->address,n.weights},
                {route_indices.address,n.routes},{route_weights.address,n.route_weights},
                {bindings.status.address,n.status},{temporary.address+layout.accumulation_offset,n.accumulation},
                {temporary.address+layout.rawdots_offset,layout.rawdots_bytes}});
        }
    }
    PreparedQ4kA8 result{layout, request, bindings, grouped_q51, slot_parallel_q51, std::nullopt, {}};
    if (integer_down) {
        const moe_q4k_a8::DownRequest down_request{true,moe_q4k_a8::down_numerical_identity(down_type),
            {config.hidden_size,config.intermediate_size,config.expert_count,config.top_k},down_type};
        const moe_q4k_a8::DownBindings down_bindings{
            bindings.output, {tensors.down.runtime->address,
                required_weight_bytes(weight_kind(down_type),config.intermediate_size,config.hidden_size,config.expert_count)},
            bindings.routes, slice(layout.down_codes_offset,layout.down_codes_bytes),
            slice(layout.down_deltas_offset,layout.down_deltas_bytes),
            slice(layout.down_sums_offset,layout.down_sums_bytes),
            slice(layout.rawdots_offset,layout.rawdots_bytes), bindings.status, {}};
        // Full producer/consumer admission precedes even the local status clear.
        moe_q4k_a8::validate_down(down_request,down_bindings);
        result.a8_down_request = down_request;
        result.a8_down_bindings = down_bindings;
    }
    return result;
}

template <typename Storage>
void launch_q4k_a8_typed(const HipMoeExpertsLauncherConfig& config,
    const ExecutionBufferAddress& route_indices,
    const ExecutionBufferAddress& route_weights, const ExecutionBufferAddress& output,
    const ExecutionBufferAddress& temporary, const RoutedTensors& tensors,
    const PreparedQ4kA8& prepared) {
    const auto& layout = prepared.layout;
    const auto& request = prepared.request;
    const auto& bindings = prepared.bindings;
    const bool grouped_q51 = prepared.grouped_q51;
    const bool slot_parallel_q51 = prepared.slot_parallel_q51;
    auto* workspace = reinterpret_cast<float*>(bindings.output.address);
    auto* accumulation = reinterpret_cast<float*>(temporary.address + layout.accumulation_offset);
    auto* status = reinterpret_cast<std::uint32_t*>(bindings.status.address);
    const auto* routes = reinterpret_cast<const std::uint32_t*>(route_indices.address);
    const auto* weights = reinterpret_cast<const float*>(route_weights.address);
    auto* aggregate = hip_graph_status();
    check_hip(hipPeekAtLastError(), "Q4K A8 runtime pre-enqueue stale error");
    // The accumulation's status word directly follows it (moe_q4k_a8_runtime.cpp layout), so the fused prologue's
    // accumulation clear plus its status write cover the bytes of the clear below.
    const bool glue_fuse = prepared.a8_down_request && config.deferred_moe_status && aggregate == nullptr &&
        layout.status_offset == layout.accumulation_offset + layout.accumulation_bytes && layout.accumulation_bytes % 4U == 0U &&
        moe_r1_glue_fuse_enabled();
    if (glue_fuse) {
        moe_q4k_a8::enqueue_serial_prologue_probe(request,bindings,accumulation,
            static_cast<unsigned>(layout.accumulation_bytes / 4U),weights,hip_execution_stream());
    } else {
    check_hip(hipMemsetAsync(accumulation,0,
        static_cast<std::size_t>(layout.accumulation_bytes + 4),hip_execution_stream()),
        "clear Q4K A8 runtime accumulation/status");
    hipLaunchKernelGGL(validate_routes_kernel,dim3(1),dim3(kThreads),0,hip_execution_stream(),
        routes,weights,config.top_k,config.expert_count,status,aggregate);
    check_hip(hipGetLastError(),"Q4K A8 route/weight validation");
    moe_q4k_a8::enqueue_probe(request,bindings,hip_execution_stream());
    }
    const auto scan = [&](const float* values,std::uint32_t count,bool bf16_commit=false) {
        hipLaunchKernelGGL(q4k_a8_validate_finite_kernel,
            dim3((count+kThreads-1U)/kThreads),dim3(kThreads),0,hip_execution_stream(),
            values,count,status,aggregate,bf16_commit);
        check_hip(hipGetLastError(),"Q4K A8 finite pipeline validation");
    };
    if (prepared.a8_down_request) {
        moe_q4k_a8::enqueue_down_activation_quantize(*prepared.a8_down_request,
            prepared.a8_down_bindings,hip_execution_stream());
    } else {
    scan(workspace,2U*config.intermediate_size*config.top_k);
    // Slot buffers are independent. Finish all original SiLU kernels before
    // checking their FP32 values, then preserve ascending down accumulation.
    if (config.batched_slot_activation) {
        hipLaunchKernelGGL(batched_routed_activation_kernel,
            dim3((config.intermediate_size+kThreads-1U)/kThreads,config.top_k),dim3(kThreads),0,
            hip_execution_stream(),workspace,config.intermediate_size,status);
        check_hip(hipGetLastError(),"Q4K A8 batched slot SiLU enqueue");
        if (config.q4k_a8_stats) {
            config.q4k_a8_stats->batched_activation_launches.fetch_add(1,std::memory_order_relaxed);
            config.q4k_a8_stats->batched_activation_slots.fetch_add(config.top_k,std::memory_order_relaxed);
        }
    } else {
        for (std::uint32_t slot=0;slot<config.top_k;++slot) {
            auto* slot_workspace=workspace+2ULL*slot*config.intermediate_size;
            hipLaunchKernelGGL(routed_activation_kernel,
                dim3((config.intermediate_size+kThreads-1U)/kThreads),dim3(kThreads),0,
                hip_execution_stream(),slot_workspace,config.intermediate_size,status);
            check_hip(hipGetLastError(),"Q4K A8 original SiLU enqueue");
        }
    }
    scan(workspace,2U*config.intermediate_size*config.top_k);
    }
    if (glue_fuse) {
        moe_q4k_a8::enqueue_down_rawdots(*prepared.a8_down_request,
            prepared.a8_down_bindings,hip_execution_stream());
        hipLaunchKernelGGL(HIP_KERNEL_NAME(q4k_a8_serial_tail_kernel<Storage>),dim3(1),dim3(1024),0,hip_execution_stream(),
            reinterpret_cast<const float*>(prepared.a8_down_bindings.rawdots.address),routes,weights,config.hidden_size,
            config.expert_count,config.top_k,accumulation,reinterpret_cast<Storage*>(output.address),status,
            config.activation == HipActivationStorage::bf16,reinterpret_cast<DeferredMoeFirstError*>(hip_deferred_moe_status()),
            static_cast<std::uint32_t>(DeferredMoeDomain::q4k_a8));
        check_hip(hipGetLastError(),"Q4K A8 serial fused merge/commit");
        if (config.q4k_a8_stats) {
            auto& counter = prepared.a8_down_request->weight_type == 7U ?
                config.q4k_a8_stats->a8_down_q51_submissions : config.q4k_a8_stats->a8_down_q80_submissions;
            counter.fetch_add(1,std::memory_order_relaxed);
        }
    } else {
    if (prepared.a8_down_request) {
        moe_q4k_a8::enqueue_down_rawdots(*prepared.a8_down_request,
            prepared.a8_down_bindings,hip_execution_stream());
        // Same kernel body and operand order as the incumbent Q5_1 merge.
        hipLaunchKernelGGL(q51_slot_parallel_merge_kernel,
            dim3((config.hidden_size+kThreads-1U)/kThreads),dim3(kThreads),0,hip_execution_stream(),
            reinterpret_cast<const float*>(prepared.a8_down_bindings.rawdots.address),routes,weights,
            config.hidden_size,config.expert_count,config.top_k,accumulation,status);
        check_hip(hipGetLastError(),"A8 down incumbent ordered slot merge");
        if (config.q4k_a8_stats) {
            auto& counter = prepared.a8_down_request->weight_type == 7U ?
                config.q4k_a8_stats->a8_down_q51_submissions : config.q4k_a8_stats->a8_down_q80_submissions;
            counter.fetch_add(1,std::memory_order_relaxed);
        }
    } else if(slot_parallel_q51) {
        auto* rawdots=reinterpret_cast<float*>(temporary.address+layout.rawdots_offset);
        hipLaunchKernelGGL(q51_slot_parallel_dot_kernel,
            dim3((config.hidden_size+7U)/8U,config.top_k),dim3(kThreads),0,hip_execution_stream(),
            workspace,routes,reinterpret_cast<const void*>(tensors.down.runtime->address),
            config.intermediate_size,config.hidden_size,config.expert_count,rawdots,status);
        check_hip(hipGetLastError(),"native slot-parallel Q5_1 raw dots");
        hipLaunchKernelGGL(q51_slot_parallel_merge_kernel,
            dim3((config.hidden_size+kThreads-1U)/kThreads),dim3(kThreads),0,hip_execution_stream(),
            rawdots,routes,weights,config.hidden_size,config.expert_count,config.top_k,accumulation,status);
        check_hip(hipGetLastError(),"native slot-parallel Q5_1 ordered merge");
        if (config.q4k_a8_stats)
            config.q4k_a8_stats->slot_parallel_q51_submissions.fetch_add(1,std::memory_order_relaxed);
    } else if(grouped_q51) {
        hipLaunchKernelGGL(q51_grouped_virtual256_down_probe_kernel,
            dim3((config.hidden_size+7U)/8U),dim3(kThreads),0,hip_execution_stream(),
            workspace,routes,weights,reinterpret_cast<const void*>(tensors.down.runtime->address),
            config.intermediate_size,config.hidden_size,config.expert_count,config.top_k,accumulation,status);
        check_hip(hipGetLastError(),"native grouped exact Q5_1 down");
    } else {
        for (std::uint32_t slot=0;slot<config.top_k;++slot)
            launch_down_by_kind(weight_kind(tensors.down.manifest->ggml_type),
                workspace+2ULL*slot*config.intermediate_size,routes,weights,slot,*tensors.down.runtime,
                config.intermediate_size,config.hidden_size,config.expert_count,accumulation,status);
    }
    scan(accumulation,config.hidden_size,config.activation == HipActivationStorage::bf16);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(commit_experts_kernel<Storage>),
        dim3((config.hidden_size+kThreads-1U)/kThreads),dim3(kThreads),0,hip_execution_stream(),
        accumulation,reinterpret_cast<Storage*>(output.address),config.hidden_size,status);
    check_hip(hipGetLastError(),"Q4K A8 semantic output commit");
    if (config.deferred_moe_status) {
        enqueue_deferred_moe_status(status,DeferredMoeDomain::q4k_a8);
    } else if (aggregate == nullptr) {
        std::uint32_t host_status = 0;
        try {
            check_hip(hipMemcpyAsync(&host_status,status,sizeof(host_status),hipMemcpyDeviceToHost,
                hip_execution_stream()),"Q4K A8 read status");
            check_hip(hipStreamSynchronize(hip_execution_stream()),"Q4K A8 drain status");
        } catch (...) {
            // host_status is still alive until the original device is drained.
            config.q4k_a8_quiescence->recover_or_fail_stop();
            throw;
        }
        if (host_status != 0)
            throw std::invalid_argument("Q4K A8 rejected status bits (route/weight=1,input=2,delta=4,pipeline=8): " +
                std::to_string(host_status));
    }
    }
    if (config.q4k_a8_stats) config.q4k_a8_stats->quantized_submissions.fetch_add(1,std::memory_order_relaxed);
    if (request.gate_type == 13U && config.q4k_a8_stats)
        config.q4k_a8_stats->q5k_a8_gate_up_submissions.fetch_add(1,std::memory_order_relaxed);
    if(grouped_q51 && config.q4k_a8_stats)
        config.q4k_a8_stats->grouped_q51_submissions.fetch_add(1,std::memory_order_relaxed);
    if ((config.a8_down_q51 || config.a8_down_q80) && !prepared.a8_down_request &&
        tensors.down.manifest->ggml_type == 8U && config.q4k_a8_stats)
        config.q4k_a8_stats->a8_down_q80_legacy_submissions.fetch_add(1,std::memory_order_relaxed);
}
#endif

struct PreparedMoeExperts {
    const HipMoeExpertsLauncherConfig* config;
    ExecutionBufferAddress hidden, route_indices, route_weights, output, temporary;
    RoutedTensors tensors;
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    std::optional<PreparedQ4kA8> quantized;
#endif
};

void validate_q51_slot_kernel_id(std::string_view id, const HipMoeExpertsLauncherConfig& config);

PreparedMoeExperts prepare_moe_experts(
    const ExecutionDispatchContext& context,
    void* user_data) {
    if (context.op !=
        SerialOp::moe_experts) {
        throw std::invalid_argument(
            "HIP routed-expert launcher attached to wrong operator");
    }
    if (context.reads.size() != 3 ||
        context.writes.size() != 1 ||
        context.temporaries.size() != 1) {
        throw std::invalid_argument(
            "HIP routed experts expect hidden/routes/weights, one output and one temporary");
    }
    if (context.logical_rows != 1 ||
        context.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP routed-expert v1 supports only C1 one-row execution");
    }
    if (user_data == nullptr) {
        throw std::invalid_argument(
            "HIP routed experts require configuration");
    }

    const auto& config =
        *static_cast<
            HipMoeExpertsLauncherConfig*>(
                user_data);
    validate_config(config);
    if (config.a8_down_q51 || config.a8_down_q80) {
        if (!config.q4k_a8_quiescence)
            throw std::invalid_argument("A8 down requires retained setup completion guard on every layer");
        if (!context.execution_identity || !context.kernel_id)
            throw std::invalid_argument("A8 down requires a compiled arithmetic selection");
        validate_q51_slot_kernel_id(*context.kernel_id,config);
        const auto* scope = current_hip_launch_context();
        if (!scope || scope->graph_capture || hip_graph_status() != nullptr ||
            scope->status.arena_base || scope->arguments.memory.arena_base ||
            !scope->stream || scope->stream == hipStreamLegacy || scope->stream == hipStreamPerThread)
            throw std::invalid_argument("A8 down admits owned eager C1 execution only");
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        check_hip(hipStreamIsCapturing(scope->stream,&capture),"query A8 down capture status");
        if (capture != hipStreamCaptureStatusNone)
            throw std::invalid_argument("A8 down rejects stream capture");
        if (!context.layer || *context.layer >= 48U || !context.resources ||
            !context.resources->model_weights || context.resources->distributed_stage)
            throw std::invalid_argument("A8 down requires canonical unpartitioned model resources");
        // This applies to every layer, including the literal Q5_K fallback.
        moe_q4k_a8::validate_a8_down_content(context.resources->model_weights->model_content_identity);
    }
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    if (config.q51_slot_parallel_down) require_q51_slot_eager_stream(hip_execution_stream());
#endif
    if (config.deferred_moe_status != (hip_deferred_moe_status()!=nullptr))
        throw std::invalid_argument("deferred MoE launcher requires matching owned eager completion scope");

    const auto hidden =
        resolve_dispatch_read(
            context, 0);
    const auto route_indices =
        resolve_dispatch_read(
            context, 1);
    const auto route_weights =
        resolve_dispatch_read(
            context, 2);
    const auto output =
        resolve_dispatch_write(
            context, 0);
    const auto temporary =
        resolve_dispatch_temporary(
            context, 0);

    const auto activation_bytes =
        static_cast<std::uint32_t>(
            config.activation);
    const auto hidden_required =
        static_cast<std::uint64_t>(
            config.hidden_size) *
        activation_bytes;
    const auto route_index_required =
        static_cast<std::uint64_t>(
            config.top_k) *
        sizeof(std::uint32_t);
    const auto route_weight_required =
        static_cast<std::uint64_t>(
            config.top_k) *
        sizeof(float);
    const auto temporary_required = config.q4k_a8g32_gate_up ?
        moe_q4k_a8::build_runtime_workspace(config.hidden_size, config.intermediate_size,
            config.expert_count, config.top_k, config.q51_slot_parallel_down,
            config.a8_down_q51 || config.a8_down_q80).total_bytes :
        (2ULL * config.intermediate_size + config.hidden_size) * sizeof(float) + sizeof(std::uint32_t);

    if (hidden.bytes <
            hidden_required ||
        output.bytes <
            hidden_required ||
        route_indices.bytes <
            route_index_required ||
        route_weights.bytes <
            route_weight_required ||
        temporary.bytes <
            temporary_required) {
        throw std::invalid_argument(
            "HIP routed-expert buffers are undersized");
    }

    const auto& access =
        resolve_dispatch_tensor_access(
            context, 0);
    const auto tensors =
        identify_tensors(
            context,
            access,
            config);

    if (tensors.fused_gate_up) {
        validate_tensor(
            tensors.gate_up,
            config.hidden_size,
            2ULL *
                config.intermediate_size,
            config.expert_count,
            "routed gate-up");
    } else {
        validate_tensor(
            tensors.gate,
            config.hidden_size,
            config.intermediate_size,
            config.expert_count,
            "routed gate");
        validate_tensor(
            tensors.up,
            config.hidden_size,
            config.intermediate_size,
            config.expert_count,
            "routed up");
    }
    validate_tensor(
        tensors.down,
        config.intermediate_size,
        config.hidden_size,
        config.expert_count,
        "routed down");

    if (config.a8_down_q51 || config.a8_down_q80) {
        const auto layer = *context.layer;
        const bool q80_layer = layer == 2U || layer == 4U || layer == 30U || layer == 46U || layer == 47U;
        const auto gate_type = layer == 2U ? 13U : 12U;
        const bool q20_layer = moe_q4k_a8::q20_a8_enabled() && !tensors.fused_gate_up &&
            gsq_a8::gate_type(tensors.gate.manifest->ggml_type) &&
            tensors.up.manifest->ggml_type == tensors.gate.manifest->ggml_type &&
            (gsq_a8::down_type(tensors.down.manifest->ggml_type) || tensors.down.manifest->ggml_type == 7U ||
             tensors.down.manifest->ggml_type == 8U);
        if (!q20_layer && (tensors.fused_gate_up || tensors.gate.manifest->ggml_type != gate_type ||
            tensors.up.manifest->ggml_type != gate_type ||
            tensors.down.manifest->ggml_type != (q80_layer ? 8U : 7U)))
            throw std::invalid_argument("A8 down layer types differ from canonical fallback census");
    }

    if (config.deferred_moe_status) {
        const auto& record=current_hip_launch_context()->deferred_moe_status;
        // A cohort's shared union reports to the record of the owner that enqueues it for all
        // sessions (its completion checks it); every other launch to its own execution's record.
        if (!context.memory || !record.execution_identity || (!known_window_cohort_active() &&
            (record.execution_identity!=context.memory->execution_identity ||
             (context.execution_identity && record.execution_identity!=context.execution_identity))))
            throw std::invalid_argument("deferred MoE record belongs to a different execution");
        const auto disjoint=[&](std::uintptr_t address,std::uint64_t bytes) {
            if (bytes>std::numeric_limits<std::uintptr_t>::max()-address ||
                (record.arena_base<address+bytes && address<record.arena_base+record.arena_bytes))
                throw std::invalid_argument("deferred MoE record aliases execution storage");
        };
        disjoint(context.memory->arena_base,context.memory->arena_bytes);
        for (const auto* pair : {&tensors.gate,&tensors.up,&tensors.gate_up,&tensors.down})
            if (pair->runtime) disjoint(pair->runtime->address,pair->runtime->bytes);
        if (context.session_state_memory) {
            disjoint(context.session_state_memory->device_base,context.session_state_memory->device_bytes);
            disjoint(context.session_state_memory->host_base,context.session_state_memory->host_bytes);
        }
    }

#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    std::optional<PreparedQ4kA8> quantized;
#endif
    if (config.q4k_a8g32_gate_up) {
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
        const bool q20 = gsq_a8::gate_type(tensors.gate.manifest->ggml_type) && moe_q4k_a8::q20_a8_enabled();
        if (tensors.fused_gate_up || tensors.gate.manifest->ggml_type != tensors.up.manifest->ggml_type ||
            (tensors.gate.manifest->ggml_type != 12 && tensors.gate.manifest->ggml_type != 13 && !q20))
            throw std::invalid_argument("Q4K A8 runtime requires separate paired Q4K or Q5K gate/up");
        if (tensors.gate.manifest->ggml_type == 12 || config.q5k_a8g32_gate_up || q20)
            quantized = prepare_q4k_a8(config, hidden, route_indices, route_weights, output, temporary, tensors);
#endif
    }
    return {&config, hidden, route_indices, route_weights, output, temporary, tensors
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
        , quantized
#endif
    };
}

void preflight_moe_experts(const ExecutionDispatchContext& context, void* user_data) {
    (void)prepare_moe_experts(context, user_data);
}

void launch_moe_experts(const ExecutionDispatchContext& context, void* user_data) {
    const auto prepared = prepare_moe_experts(context, user_data);
    const auto& config = *prepared.config;
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    if (prepared.quantized) {
        if (config.activation == HipActivationStorage::bf16)
            launch_q4k_a8_typed<std::uint16_t>(config, prepared.route_indices,
                prepared.route_weights, prepared.output, prepared.temporary, prepared.tensors, *prepared.quantized);
        else launch_q4k_a8_typed<float>(config, prepared.route_indices,
                prepared.route_weights, prepared.output, prepared.temporary, prepared.tensors, *prepared.quantized);
        return;
    }
    if (config.q4k_a8g32_gate_up && config.q4k_a8_stats)
        config.q4k_a8_stats->legacy_q5k_submissions.fetch_add(1,std::memory_order_relaxed);
#endif
    const auto invoke = [&]<typename Storage>() {
        launch_typed<Storage>(config, prepared.hidden, prepared.route_indices,
            prepared.route_weights, prepared.output, prepared.temporary, prepared.tensors);
    };
    switch (config.activation) {
    case HipActivationStorage::bf16: invoke.template operator()<std::uint16_t>(); return;
    case HipActivationStorage::fp32: invoke.template operator()<float>(); return;
    }
    throw std::logic_error("unreachable routed-expert activation storage");
}

std::string kernel_id(
    const ExecutionBindingRequest& request,
    const HipMoeExpertsLauncherConfig& config) {
    return
        "hip.moe_experts." +
        std::string(
            storage_name(
                config.activation)) +
        ".h" +
        std::to_string(
            config.hidden_size) +
        ".m" +
        std::to_string(
            config.intermediate_size) +
        ".e" +
        std::to_string(
            config.expert_count) +
        ".k" +
        std::to_string(
            config.top_k) +
        ".r" +
        std::to_string(
            request.logical_rows) +
        ".c" +
        std::to_string(
            request.request_concurrency) +
        (config.batched_slot_activation ? ".batched-slot-activation-v1" : "") +
        (config.deferred_moe_status ? ".deferred-status-v1" : "") +
        (config.q5k_a8g32_gate_up ? std::string(moe_q4k_a8::q5k_gate_up_kernel_marker) : "") +
        (config.a8_down_q51 ? std::string(moe_q4k_a8::q51_a8_down_kernel_marker) : "") +
        (config.a8_down_q80 ? std::string(moe_q4k_a8::q80_a8_down_kernel_marker) : "") +
        ((config.a8_down_q51 || config.a8_down_q80) ? std::string(moe_q4k_a8::a8_down_kernel_ending) :
            std::string(config.q51_slot_parallel_down ? ".q51-slot-parallel-v1" : "") +
            (config.grouped_exact_q51_down ? ".grouped-exact-q51-down-v1" : "") +
            (config.q4k_a8g32_gate_up ? std::string(moe_q4k_a8::runtime_kernel_suffix) : ""));
}

std::uint64_t implementation_identity(
    const HipMoeExpertsLauncherConfig& config) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, kLauncherTag);
    hash_u64(hash, config.hidden_size);
    hash_u64(hash, config.intermediate_size);
    hash_u64(hash, config.expert_count);
    hash_u64(hash, config.top_k);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            config.activation));
    if (config.q4k_a8g32_gate_up) hash_u64(hash,0x51344b4138473332ULL);
    if (config.grouped_exact_q51_down) hash_u64(hash,0x5135314752505631ULL);
    if (config.q51_slot_parallel_down) hash_u64(hash,0x5135315350415231ULL);
    if (config.deferred_moe_status) hash_u64(hash,0x4445464d4f453031ULL);
    if (config.batched_slot_activation) hash_u64(hash,0x42415453494c5531ULL);
    if (config.a8_down_q51) hash_u64(hash,0x5135314138445631ULL);
    if (config.a8_down_q80) hash_u64(hash,0x5138304138445631ULL);
    if (config.q5k_a8g32_gate_up) hash_u64(hash,0x51354b4138475531ULL);
    return hash;
}

void validate_q51_slot_kernel_id(std::string_view id, const HipMoeExpertsLauncherConfig& config) {
    const auto q5k_marker = moe_q4k_a8::q5k_gate_up_kernel_marker;
    const auto q5k_position = id.find(q5k_marker);
    if ((q5k_position != std::string_view::npos) != config.q5k_a8g32_gate_up ||
        (config.q5k_a8g32_gate_up && id.find(q5k_marker,q5k_position+q5k_marker.size()) != std::string_view::npos))
        throw std::invalid_argument("Q5_K A8 id has missing, foreign or duplicate gate/up marker");
    if (config.a8_down_q51 || config.a8_down_q80) {
        const auto tail =
            (config.q5k_a8g32_gate_up ? std::string(q5k_marker) : "") +
            (config.a8_down_q51 ? std::string(moe_q4k_a8::q51_a8_down_kernel_marker) : "") +
            (config.a8_down_q80 ? std::string(moe_q4k_a8::q80_a8_down_kernel_marker) : "") +
            std::string(moe_q4k_a8::a8_down_kernel_ending);
        if (!id.ends_with(tail) || id.find(".q51-slot-parallel-v1") != std::string_view::npos ||
            id.find(".grouped-exact-q51-down-v1") != std::string_view::npos)
            throw std::invalid_argument("A8 down id differs from explicit mixed arithmetic configuration");
        for (const auto marker : {moe_q4k_a8::q51_a8_down_kernel_marker,moe_q4k_a8::q80_a8_down_kernel_marker}) {
            const auto pos = id.find(marker);
            const bool expected = marker == moe_q4k_a8::q51_a8_down_kernel_marker ? config.a8_down_q51 : config.a8_down_q80;
            if ((pos != std::string_view::npos) != expected || (expected &&
                id.find(marker,pos+marker.size()) != std::string_view::npos))
                throw std::invalid_argument("A8 down id has missing, foreign or duplicate arithmetic markers");
        }
        return;
    }
    if (id.find(moe_q4k_a8::q51_a8_down_kernel_marker) != std::string_view::npos ||
        id.find(moe_q4k_a8::q80_a8_down_kernel_marker) != std::string_view::npos ||
        id.find(".mixed-a8-down-v1") != std::string_view::npos)
        throw std::invalid_argument("A8 down id cannot select an exact configuration");
    const bool enabled = config.q51_slot_parallel_down;
    constexpr std::string_view marker=".q51-slot-parallel-v1";
    const auto position=id.find(marker);
    const bool present=position!=std::string_view::npos;
    const auto ending=std::string(marker)+std::string(moe_q4k_a8::grouped_q51_kernel_ending);
    if (present!=enabled || (enabled && (!id.ends_with(ending) ||
        id.find(marker,position+marker.size())!=std::string_view::npos)))
        throw std::invalid_argument("slot-parallel Q5_1 id differs from config or marker placement");
}

}  // namespace

void register_hip_moe_experts_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipMoeExpertsLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification) {
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    validate_config(config);
    if (request.batched_moe_activation != config.batched_slot_activation)
        throw std::invalid_argument("batched routed activation config differs from binding request");
    if (request.batched_moe_activation && (request.device_graph_inputs || request.require_graph_safe))
        throw std::invalid_argument("batched routed activation variant admits eager execution only");
    if(request.grouped_exact_q51_down != config.grouped_exact_q51_down)
        throw std::invalid_argument("grouped exact Q5_1 config differs from binding request");
    if (request.q51_slot_parallel_down != config.q51_slot_parallel_down)
        throw std::invalid_argument("slot-parallel Q5_1 config differs from binding request");
    if (request.q51_slot_parallel_down && (request.device_graph_inputs || request.require_graph_safe))
        throw std::invalid_argument("slot-parallel Q5_1 variant admits eager execution only");
    if (request.deferred_moe_status != config.deferred_moe_status ||
        (config.deferred_moe_status && (request.device_graph_inputs || request.require_graph_safe)))
        throw std::invalid_argument("deferred MoE variant requires matching eager request");
    if (request.q4k_a8g32_gate_up != config.q4k_a8g32_gate_up)
        throw std::invalid_argument("Q4K A8 variant config differs from binding request");
    if (request.a8_down_q51 != config.a8_down_q51 || request.a8_down_q80 != config.a8_down_q80)
        throw std::invalid_argument("A8 down config differs from binding request");
    if (request.q5k_a8g32_gate_up != config.q5k_a8g32_gate_up)
        throw std::invalid_argument("Q5_K A8 gate/up config differs from binding request");
    if ((config.a8_down_q51 || config.a8_down_q80) &&
        (model.hidden_layers != 48U || model.bytes_fp32 != 4U))
        throw std::invalid_argument("A8 down registers canonical 48-layer FP32 control geometry only");
    if ((config.a8_down_q51 || config.a8_down_q80) && (request.device_graph_inputs || request.require_graph_safe))
        throw std::invalid_argument("A8 down registers eager C1 variants only");
    if (config.hidden_size !=
            model.hidden_size ||
        config.intermediate_size !=
            model.moe_intermediate_size ||
        config.expert_count !=
            model.num_experts ||
        config.top_k !=
            model.num_experts_per_token) {
        throw std::invalid_argument(
            "HIP routed-expert config does not match model contract");
    }
    if (request.logical_rows != 1 ||
        request.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP routed-expert v1 registers only C1 one-row variants");
    }
    if (qualification ==
            kernels::QualificationStatus::target_qualified ||
        qualification ==
            kernels::QualificationStatus::production ||
        qualification ==
            kernels::QualificationStatus::rejected) {
        throw std::invalid_argument(
            "unmeasured HIP routed experts may only be experimental or oracle-matched");
    }

    kernels::KernelVariant variant;
    variant.id =
        kernel_id(
            request,
            config);
    variant.family =
        kernels::OperatorFamily::moe_expert;
    variant.phase =
        kernels::ExecutionPhase::decode;
    variant.shape =
        execution_shape_for(
            model,
            SerialOp::moe_experts,
            request);
    variant.wave =
        kernels::WaveSize::wave32;
    variant.resources.workgroup_threads =
        kThreads;
    variant.resources.lds_bytes =
        kThreads * sizeof(float);
    variant.resources.scratch_bytes =
        (2ULL * config.intermediate_size + config.hidden_size) * sizeof(float) + sizeof(std::uint32_t);
    variant.numerics.input_precision =
        storage_name(
            config.activation);
    variant.numerics.accumulator_precision =
        "fp32";
    variant.numerics.output_precision =
        storage_name(
            config.activation);
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        qualification;
    variant.graph_safe = !config.deferred_moe_status && !config.q51_slot_parallel_down;
    variant.input_layout =
        "row-major+device-topk";
    variant.output_layout =
        "row-major";
    variant.compiler_identity =
        "hip-gfx1151-moe-experts-v2";
    if (config.q4k_a8g32_gate_up) {
        variant.resources.lds_bytes = 3328;
        variant.resources.scratch_bytes = moe_q4k_a8::build_runtime_workspace(config.hidden_size,
            config.intermediate_size,config.expert_count,config.top_k,config.q51_slot_parallel_down,
            config.a8_down_q51 || config.a8_down_q80).total_bytes;
        variant.numerics.accumulator_precision = std::string(moe_q4k_a8::runtime_numerical_identity);
        variant.compiler_identity = "hip-gfx1151-moe-q4k-a8g32-native-v2-probe-strict-rndiv-legacy-down";
    }
    if(config.grouped_exact_q51_down && !config.a8_down_q51) {
        variant.numerics.accumulator_precision += ";" + std::string(moe_q4k_a8::grouped_q51_numerical_identity);
        variant.compiler_identity += ":grouped-exact-q51-down-v1";
    }
    if(config.deferred_moe_status) variant.compiler_identity += ":deferred-first-error-v1";
    if(config.q51_slot_parallel_down && !config.a8_down_q51) {
        variant.numerics.accumulator_precision += ";" + std::string(q51_slot_parallel::numerical_identity);
        variant.compiler_identity += ":q51-slot-parallel-rawdot-v1";
    }
    if(config.batched_slot_activation) variant.compiler_identity += ":batched-slot-activation-v1";
    if (config.a8_down_q51 || config.a8_down_q80) {
        variant.numerics.accumulator_precision = std::string(moe_q4k_a8::a8_down_gate_up_numerical_identity);
        if (config.a8_down_q51)
            variant.numerics.accumulator_precision += ";" + std::string(moe_q4k_a8::q51_a8_down_numerical_identity);
        else
            variant.numerics.accumulator_precision += ";" + std::string(q51_slot_parallel::numerical_identity);
        if (config.a8_down_q80)
            variant.numerics.accumulator_precision += ";" + std::string(moe_q4k_a8::q80_a8_down_numerical_identity);
        else
            variant.numerics.accumulator_precision += ";four-q4k-q80-legacy-slot-down";
        if (config.q5k_a8g32_gate_up)
            variant.numerics.accumulator_precision += ";" + std::string(moe_q4k_a8::q5k_numerical_identity) +
                ";layer2-q5k-q80-a8:final-scan-bf16-commit-d1";
        else
            variant.numerics.accumulator_precision += ";layer2-q5k-q80-literal-fallback:final-scan-bf16-commit-d1";
        variant.compiler_identity = "hip-gfx1151-moe-a8-down-strict-v1:activation-quantize:integer-rawdots:incumbent-slot-merge:final-scan-commit-d1";
        if (config.a8_down_q51) variant.compiler_identity += ":q51-iu5";
        if (config.a8_down_q80) variant.compiler_identity += ":q80-is8";
        if (config.q5k_a8g32_gate_up) variant.compiler_identity += ":q5k-a8g32-gateup-v1";
        variant.graph_safe = false;
    }
    variant.measured_seconds = 0.0;

    registry.add(
        std::move(variant));
}

ExecutionLauncher
make_hip_moe_experts_launcher(
    std::string kernel_id_value,
    HipMoeExpertsLauncherConfig* config, bool perform_device_setup) {
    if (kernel_id_value.empty() ||
        config == nullptr) {
        throw std::invalid_argument(
            "HIP routed-expert launcher requires id and config");
    }
    validate_config(*config);
    validate_q51_slot_kernel_id(kernel_id_value,*config);
    if ((kernel_id_value.find(".batched-slot-activation-v1")!=std::string::npos)!=config->batched_slot_activation)
        throw std::invalid_argument("batched routed activation launcher id differs from config");
    if ((kernel_id_value.find(".deferred-status-v1")!=std::string::npos)!=config->deferred_moe_status)
        throw std::invalid_argument("deferred MoE launcher id differs from config");
    if(!(config->a8_down_q51 || config->a8_down_q80) &&
        kernel_id_value.ends_with(moe_q4k_a8::grouped_q51_kernel_ending)!=config->grouped_exact_q51_down)
        throw std::invalid_argument("grouped exact Q5_1 launcher id differs from config");
    if (kernel_id_value.ends_with(moe_q4k_a8::runtime_kernel_suffix) != config->q4k_a8g32_gate_up)
        throw std::invalid_argument("Q4K A8 launcher id differs from config");
    if (perform_device_setup && config->q4k_a8g32_gate_up && !config->q4k_a8_quiescence)
        config->q4k_a8_quiescence = std::make_shared<HipSetupQuiescenceGuard>();
    return ExecutionLauncher{
        std::move(kernel_id_value),
        implementation_identity(
            *config),
        &launch_moe_experts,
        config,
        &preflight_moe_experts,
    };
}

void HipMoeExpertsLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry, bool perform_device_setup) {
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP routed-expert bundle requires valid compiled execution");
    }

    std::string registered_id;
    for (std::size_t index = 0;
         index < compiled.execution.steps.size();
         ++index) {
        if (compiled.execution.steps[index].op !=
            SerialOp::moe_experts) {
            continue;
        }
        const auto& binding =
            compiled.bindings.bindings[index];
        if (binding.host_or_runtime_step ||
            !binding.kernel_id.has_value() ||
            binding.kernel_id->empty()) {
            throw std::invalid_argument(
                "HIP routed-expert step has no kernel binding");
        }
        validate_q51_slot_kernel_id(*binding.kernel_id,config_);
        if ((binding.kernel_id->find(".deferred-status-v1")!=std::string::npos)!=config_.deferred_moe_status)
            throw std::invalid_argument("deferred MoE bundle mode differs from compiled binding");
        if ((binding.kernel_id->find(".batched-slot-activation-v1")!=std::string::npos)!=config_.batched_slot_activation)
            throw std::invalid_argument("batched routed activation bundle differs from compiled selection");
        if(!(config_.a8_down_q51 || config_.a8_down_q80) &&
            binding.kernel_id->ends_with(moe_q4k_a8::grouped_q51_kernel_ending)!=config_.grouped_exact_q51_down)
            throw std::invalid_argument("grouped exact Q5_1 bundle differs from compiled selection");
        if (binding.kernel_id->ends_with(moe_q4k_a8::runtime_kernel_suffix) != config_.q4k_a8g32_gate_up)
            throw std::invalid_argument("Q4K A8 bundle mode differs from compiled selection");
        if (registered_id.empty()) {
            registered_id =
                *binding.kernel_id;
            registry.add_kernel_launcher(
                make_hip_moe_experts_launcher(
                    registered_id,
                    &config_, perform_device_setup));
        } else if (registered_id !=
                   *binding.kernel_id) {
            throw std::invalid_argument(
                "HIP routed-expert bundle requires one selected kernel id");
        }
    }
}


#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
namespace q51_grouped_down {
namespace {
void enqueue_preflight(const Request& request,const Bindings& b,hipStream_t stream) {
    validate(request,b);
    check_hip(hipPeekAtLastError(),"Q5_1 grouped-down pre-enqueue stale error");
    hipLaunchKernelGGL(validate_routes_kernel,dim3(1),dim3(kThreads),0,stream,
        reinterpret_cast<const std::uint32_t*>(b.routes.address),
        reinterpret_cast<const float*>(b.route_weights.address),request.shape.top_k,
        request.shape.experts,reinterpret_cast<std::uint32_t*>(b.status.address),nullptr);
    check_hip(hipGetLastError(),"Q5_1 grouped-down route validation");
}
}
void enqueue_grouped_probe(const Request& request,const Bindings& b,hipStream_t stream) {
    enqueue_preflight(request,b,stream);
    hipLaunchKernelGGL(q51_grouped_virtual256_down_probe_kernel,
        dim3((request.shape.hidden+7U)/8U),dim3(kThreads),0,stream,
        reinterpret_cast<const float*>(b.intermediates.address),
        reinterpret_cast<const std::uint32_t*>(b.routes.address),
        reinterpret_cast<const float*>(b.route_weights.address),reinterpret_cast<const void*>(b.weights.address),
        request.shape.intermediate,request.shape.hidden,request.shape.experts,request.shape.top_k,
        reinterpret_cast<float*>(b.accumulation.address),reinterpret_cast<const std::uint32_t*>(b.status.address));
    check_hip(hipGetLastError(),"Q5_1 grouped virtual256 down probe");
}
void enqueue_original_receipt(const Request& request,const Bindings& b,hipStream_t stream) {
    enqueue_preflight(request,b,stream);
    for(std::uint32_t slot=0;slot<request.shape.top_k;++slot) {
        const auto address=b.intermediates.address+static_cast<std::uintptr_t>(slot)*2U*request.shape.intermediate*sizeof(float);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(selected_down_accumulate_kernel<float,WeightKind::q5_1>),
            dim3(request.shape.hidden),dim3(kThreads),0,stream,reinterpret_cast<const float*>(address),
            reinterpret_cast<const std::uint32_t*>(b.routes.address),
            reinterpret_cast<const float*>(b.route_weights.address),slot,reinterpret_cast<const void*>(b.weights.address),
            request.shape.intermediate,request.shape.hidden,request.shape.experts,
            reinterpret_cast<float*>(b.accumulation.address),reinterpret_cast<const std::uint32_t*>(b.status.address));
        check_hip(hipGetLastError(),"original Q5_1 down receipt");
    }
}
} // namespace q51_grouped_down

namespace q51_slot_parallel {
void enqueue_slot_parallel_probe(const Request& request,const Bindings& b,hipStream_t stream) {
    validate(request,b);
    require_q51_slot_eager_stream(stream);
    const q51_grouped_down::Request original_request{true,q51_grouped_down::numerical_identity,request.shape};
    const q51_grouped_down::Bindings original{b.intermediates,b.weights,b.routes,b.route_weights,b.status,b.accumulation};
    q51_grouped_down::enqueue_preflight(original_request,original,stream);
    hipLaunchKernelGGL(q51_slot_parallel_dot_kernel,
        dim3((request.shape.hidden+7U)/8U,request.shape.top_k),dim3(kThreads),0,stream,
        reinterpret_cast<const float*>(b.intermediates.address),
        reinterpret_cast<const std::uint32_t*>(b.routes.address),reinterpret_cast<const void*>(b.weights.address),
        request.shape.intermediate,request.shape.hidden,request.shape.experts,
        reinterpret_cast<float*>(b.rawdots.address),reinterpret_cast<const std::uint32_t*>(b.status.address));
    check_hip(hipGetLastError(),"Q5_1 slot-parallel raw dots probe");
    hipLaunchKernelGGL(q51_slot_parallel_merge_kernel,
        dim3((request.shape.hidden+kThreads-1U)/kThreads),dim3(kThreads),0,stream,
        reinterpret_cast<const float*>(b.rawdots.address),
        reinterpret_cast<const std::uint32_t*>(b.routes.address),reinterpret_cast<const float*>(b.route_weights.address),
        request.shape.hidden,request.shape.experts,request.shape.top_k,
        reinterpret_cast<float*>(b.accumulation.address),reinterpret_cast<const std::uint32_t*>(b.status.address));
    check_hip(hipGetLastError(),"Q5_1 slot-parallel ordered merge probe");
}
} // namespace q51_slot_parallel
#endif

#ifdef STRIX_BUILD_MOE_BATCHED_ACTIVATION_DIAGNOSTIC
namespace moe_batched_activation_diagnostic {
const void* kernel_symbol(Mode mode) {
    switch (mode) {
    case Mode::reference: return reinterpret_cast<const void*>(routed_activation_kernel);
    case Mode::batched: return reinterpret_cast<const void*>(batched_routed_activation_kernel);
    }
    throw std::invalid_argument("invalid routed activation diagnostic mode");
}
void enqueue(Shape shape, const Bindings& bindings, Mode mode, hipStream_t stream) {
    const auto launch = validate(shape, bindings, mode);
    if (!stream || stream == hipStreamPerThread || stream == hipStreamLegacy)
        throw std::invalid_argument("routed activation diagnostic requires an explicitly owned stream");
    check_hip(hipPeekAtLastError(), "routed activation diagnostic stale error");
    auto* workspace = reinterpret_cast<float*>(bindings.workspace.address);
    const auto* status = reinterpret_cast<const std::uint32_t*>(bindings.status.address);
    if (mode == Mode::batched) {
        hipLaunchKernelGGL(batched_routed_activation_kernel,
            dim3(launch.grid_x,launch.grid_y),dim3(launch.block_x),0,stream,
            workspace,shape.intermediate,status);
        check_hip(hipGetLastError(), "routed activation diagnostic batched enqueue");
    } else {
        for (std::uint32_t slot = 0; slot < shape.slots; ++slot) {
            hipLaunchKernelGGL(routed_activation_kernel,
                dim3(launch.grid_x),dim3(launch.block_x),0,stream,
                workspace+2ULL*slot*shape.intermediate,shape.intermediate,status);
            check_hip(hipGetLastError(), "routed activation diagnostic original enqueue");
        }
    }
}
} // namespace moe_batched_activation_diagnostic
#endif

}  // namespace strix

namespace strix {
namespace {
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
// Window unions of 2/4/8 rows; cohort unions of several sessions' windows of 16 or 32 rows.
constexpr std::uint32_t kKnownMoeMaxRows = moe_q4k_a8::known_rows::max_union_rows;
struct KnownMoeRowsDevice {
    const float* intermediates[kKnownMoeMaxRows]{};
    const std::uint32_t* routes[kKnownMoeMaxRows]{};
    const float* route_weights[kKnownMoeMaxRows]{};
    std::uint32_t* status[kKnownMoeMaxRows]{};
    float* rawdots[kKnownMoeMaxRows]{};
    float* accumulation[kKnownMoeMaxRows]{};
    std::uint16_t* output[kKnownMoeMaxRows]{};
};
struct KnownMoeCsrDevice {
    std::uint32_t* header; // union status, touched count, batch count
    std::uint32_t* offsets;
    std::uint32_t* touched;
    std::uint32_t* rows;
    std::uint32_t* slots;
    std::uint32_t* batch_experts;
    std::uint32_t* batch_begins;
};

// The incumbent per-output reduction of eight lane leaves (__shfl_down tree
// over 32 lanes, then ((s0+s4)+(s2+s6))+((s1+s5)+(s3+s7)) in lane 0) with
// the same operand pairs, packing two then four leaves per register in DPP
// bank masks. Result in lane 0; every lane must be active.
#define STRIX_KNOWN_MOE_DPP(out, in, control, banks) \
    asm("v_add_f32_dpp %0, %1, %1 " control " row_mask:0xf bank_mask:" banks : "=v"(out) : "v"(in))
#define STRIX_KNOWN_MOE_DPP_INTO(out, in, control, banks) \
    asm("v_add_f32_dpp %0, %1, %1 " control " row_mask:0xf bank_mask:" banks : "+v"(out) : "v"(in))
__device__ __forceinline__ float known_moe_reduce_leaves(float (&leaf)[8]) {
#pragma unroll
    for (unsigned j = 0; j < 8U; ++j) {
        const auto bits = __float_as_uint(leaf[j]);
        leaf[j] += __uint_as_float(__builtin_amdgcn_permlanex16(bits, bits, 0x76543210U, 0xfedcba98U, false, false));
    }
    float pair[4], quad[2];
    // Step 8: pair[p] lanes 0-7 hold leaf first[p], lanes 8-15 leaf first[p]+2.
    constexpr unsigned first[4] = {0U, 1U, 4U, 5U};
#pragma unroll
    for (unsigned p = 0; p < 4U; ++p) {
        STRIX_KNOWN_MOE_DPP(pair[p], leaf[first[p]], "row_shl:8", "0x3");
        STRIX_KNOWN_MOE_DPP_INTO(pair[p], leaf[first[p] + 2U], "row_shr:8", "0xc");
    }
    // Step 4: banks 0..3 of quad[h] hold leaves 4h..4h+3; then steps 2 and 1.
#pragma unroll
    for (unsigned h = 0; h < 2U; ++h) {
        STRIX_KNOWN_MOE_DPP(quad[h], pair[2U * h], "row_shl:4", "0x5");
        STRIX_KNOWN_MOE_DPP_INTO(quad[h], pair[2U * h + 1U], "row_shr:4", "0xa");
        float step;
        STRIX_KNOWN_MOE_DPP(step, quad[h], "row_shl:2", "0xf");
        STRIX_KNOWN_MOE_DPP(quad[h], step, "row_shl:1", "0xf");
    }
    float sum = quad[0] + quad[1], step;
    STRIX_KNOWN_MOE_DPP(step, sum, "row_shl:8", "0xf");
    STRIX_KNOWN_MOE_DPP(sum, step, "row_shl:4", "0xf");
    return sum;
}
#undef STRIX_KNOWN_MOE_DPP_INTO
#undef STRIX_KNOWN_MOE_DPP

// One block per (CSR batch, 8*Outs outputs, group of G batch rows); row groups
// are the fastest grid index so siblings re-reading a slice hit L2. Each wave
// loads its Outs packed Q5_1 rows once, decodes every coefficient once and
// applies it to the G rows staged in LDS. Per row/output the lane leaves,
// x*coefficient expression, chunk order, +0 seed and reduction tree are the
// incumbent slot-parallel kernel's. A failing block (0,0,0) folds every
// projection-stage status into the summary (the incumbent status kernel); the
// success path stores nothing before its loads, so batch metadata stays
// scalar and the staged rows are requested ahead of the weights.
template <unsigned G, unsigned Outs>
__global__ __launch_bounds__(256) void known_moe_q51_union_down_kernel(KnownMoeRowsDevice view, KnownMoeCsrDevice csr,
    std::uint32_t row_count, const void* weights, std::uint32_t intermediate_size, std::uint32_t hidden_size) {
    std::uint32_t failure = csr.header[0];
    for (std::uint32_t r = 0; r < row_count; ++r) failure |= *view.status[r];
    if (failure) {
        if (blockIdx.x == 0U && blockIdx.y == 0U && threadIdx.x == 0U) csr.header[0] = failure;
        return;
    }
    if (blockIdx.y >= csr.header[2]) return;
    const auto expert = csr.batch_experts[blockIdx.y];
    const auto groups = (row_count + G - 1U) / G, tile = static_cast<std::uint32_t>(blockIdx.x) / groups;
    const auto group = static_cast<std::uint32_t>(blockIdx.x) % groups * G;
    const auto begin = csr.batch_begins[blockIdx.y] + group, end = csr.batch_begins[blockIdx.y + 1U];
    if (begin >= end) return;
    const auto count = end - begin;
    __shared__ __align__(16) float staged[G][768]; // intermediate_size <= 768, admitted by the host
    // One float4 per thread and staged row, requested before the weights.
    const auto vector = 4U * static_cast<std::uint32_t>(threadIdx.x);
    float4 row_vector[G];
#pragma unroll
    for (unsigned a = 0; a < G; ++a) if (a < count && vector < intermediate_size)
        row_vector[a] = *reinterpret_cast<const float4*>(view.intermediates[csr.rows[begin + a]] +
            static_cast<std::uint64_t>(csr.slots[begin + a]) * 2U * intermediate_size + vector);
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) % 32U;
    // hidden_size % (8*Outs) == 0 is admitted by the host: no wave leaves before the barrier.
    const auto first = (tile * 8U + static_cast<std::uint32_t>(threadIdx.x) / 32U) * Outs;
    constexpr unsigned chunks = 3U;
    const auto blocks_per_row = intermediate_size / 32U, quarter = lane % 4U, shift = quarter >= 2U ? 4U : 0U;
    const auto* base = static_cast<const unsigned char*>(weights);
    uint2 header[Outs][chunks], packed[Outs][chunks];
#pragma unroll
    for (unsigned o = 0; o < Outs; ++o)
#pragma unroll
        for (unsigned k = 0; k < chunks; ++k) if (256U * k + 8U * lane < intermediate_size) {
            const auto* block = base + ((static_cast<std::uint64_t>(expert) * hidden_size + first + o) * blocks_per_row + 8U * k + lane / 4U) * 24U;
            header[o][k] = *reinterpret_cast<const uint2*>(block);
            packed[o][k] = *reinterpret_cast<const uint2*>(block + 8U + 8U * (quarter & 1U));
        }
#pragma unroll
    for (unsigned a = 0; a < G; ++a) if (a < count && vector < intermediate_size)
        *reinterpret_cast<float4*>(&staged[a][vector]) = row_vector[a];
    __syncthreads();
    float partial[G][Outs][8] = {};
#pragma unroll
    for (unsigned k = 0; k < chunks; ++k) {
        const auto feature = 256U * k + 8U * lane;
        if (feature >= intermediate_size) continue;
        float xs[G][8];
#pragma unroll
        for (unsigned a = 0; a < G; ++a) if (a < count) {
            const float4 x0 = *reinterpret_cast<const float4*>(&staged[a][feature]);
            const float4 x1 = *reinterpret_cast<const float4*>(&staged[a][feature + 4U]);
            xs[a][0] = x0.x; xs[a][1] = x0.y; xs[a][2] = x0.z; xs[a][3] = x0.w;
            xs[a][4] = x1.x; xs[a][5] = x1.y; xs[a][6] = x1.z; xs[a][7] = x1.w;
        }
#pragma unroll
        for (unsigned o = 0; o < Outs; ++o) {
            const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(header[o][k].x & 65535U)));
            const float m = __half2float(__ushort_as_half(static_cast<unsigned short>(header[o][k].x >> 16U)));
            const auto high_bits = (header[o][k].y >> (8U * quarter)) & 255U;
            float coefficient[8];
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) {
                const auto byte = ((j < 4U ? packed[o][k].x : packed[o][k].y) >> (8U * (j % 4U))) & 255U;
                const auto low = (byte >> shift) & 15U, high = (high_bits >> j) & 1U;
                coefficient[j] = d * static_cast<float>(low | (high << 4U)) + m;
            }
#pragma unroll
            for (unsigned a = 0; a < G; ++a) if (a < count)
#pragma unroll
                for (unsigned j = 0; j < 8U; ++j) partial[a][o][j] += xs[a][j] * coefficient[j];
        }
    }
#pragma unroll
    for (unsigned a = 0; a < G; ++a) if (a < count) {
        auto* target = view.rawdots[csr.rows[begin + a]] + static_cast<std::uint64_t>(csr.slots[begin + a]) * hidden_size + first;
#pragma unroll
        for (unsigned o = 0; o < Outs; ++o) {
            const float sum = known_moe_reduce_leaves(partial[a][o]);
            if (lane == 0U) target[o] = sum;
        }
    }
}

// Incumbent ordered merge and BF16 finite scan for every row in one launch.
// header[0] already holds every pre-merge status (down kernel block (0,0)).
__global__ void known_moe_rows_merge_kernel(KnownMoeRowsDevice view, KnownMoeCsrDevice csr,
    std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k) {
    const auto row = blockIdx.y;
    if (csr.header[0] || *view.status[row]) return;
    const auto out = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out >= hidden_size) return;
    for (std::uint32_t slot = 0; slot < top_k; ++slot) if (view.routes[row][slot] >= experts) return;
    float dots[10];
#pragma unroll
    for (std::uint32_t slot = 0; slot < 10U; ++slot) if (slot < top_k) dots[slot] = view.rawdots[row][static_cast<std::uint64_t>(slot) * hidden_size + out];
    float current = 0.0F;
#pragma unroll
    for (std::uint32_t slot = 0; slot < 10U; ++slot) if (slot < top_k) {
        const float weight = view.route_weights[row][slot];
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(current) : "s"(weight), "v"(dots[slot]));
    }
    view.accumulation[row][out] = current;
    if ((__float_as_uint(current) & 0x7fffffffU) >= 0x7f7f8000U) atomicOr(view.status[row], 8U);
}
// All-row commit admission, summary status and D1 publication in row order.
// After a route/quantizer failure (bits 1|2|4, nothing projected) the summary
// is the planner's fail-closed form: no touched experts, batches or histograms.
__global__ void known_moe_rows_commit_kernel(KnownMoeRowsDevice view, KnownMoeCsrDevice csr,
    std::uint32_t row_count, std::uint32_t hidden_size, std::uint32_t* record) {
    std::uint32_t failure = csr.header[0];
    for (std::uint32_t r = 0; r < row_count; ++r) failure |= *view.status[r];
    if (blockIdx.x == 0U && blockIdx.y == 0U && threadIdx.x == 0U) {
        csr.header[0] = failure;
        if (failure & 7U) for (const auto word : {1U, 2U, 4U, 5U}) csr.header[word] = 0U;
        if (failure & 7U) for (std::uint32_t word = 8U; word < 24U; ++word) csr.header[word] = 0U;
        for (std::uint32_t r = 0; r < row_count; ++r)
            merge_deferred_moe_first_error(*reinterpret_cast<DeferredMoeFirstError*>(record), *view.status[r], DeferredMoeDomain::q4k_a8);
    }
    if (failure) return;
    const auto out = blockIdx.x * blockDim.x + threadIdx.x;
    if (out < hidden_size) store_activation(view.output[blockIdx.y] + out, view.accumulation[blockIdx.y][out]);
}

// The MoE combine folded into the commit (STRIX_WINDOW_COMMIT_COMBINE=1): each output element's thread also
// stores shared + routed (the routed value just committed, or the existing one after a failure, as the
// separate combine launch would read it) in the combine's output row. The sum is operand-pinned to
// combine_rows_kernel's v_add_f32 shared, routed (src0 shared), so two NaN operands keep the shared payload
// there too; the commit store and the reload are the commit kernel's BF16 transform.
struct KnownMoeCombineDevice {
    const std::uint16_t* shared[kKnownMoeMaxRows]{};
    std::uint16_t* combined[kKnownMoeMaxRows]{};
};
template <bool Combine>
__global__ void known_moe_rows_commit_combine_kernel(KnownMoeRowsDevice view, KnownMoeCsrDevice csr,
    std::uint32_t row_count, std::uint32_t hidden_size, std::uint32_t* record, KnownMoeCombineDevice combine) {
    std::uint32_t failure = csr.header[0];
    for (std::uint32_t r = 0; r < row_count; ++r) failure |= *view.status[r];
    if (blockIdx.x == 0U && blockIdx.y == 0U && threadIdx.x == 0U) {
        csr.header[0] = failure;
        if (failure & 7U) for (const auto word : {1U, 2U, 4U, 5U}) csr.header[word] = 0U;
        if (failure & 7U) for (std::uint32_t word = 8U; word < 24U; ++word) csr.header[word] = 0U;
        for (std::uint32_t r = 0; r < row_count; ++r)
            merge_deferred_moe_first_error(*reinterpret_cast<DeferredMoeFirstError*>(record), *view.status[r], DeferredMoeDomain::q4k_a8);
    }
    const auto out = blockIdx.x * blockDim.x + threadIdx.x;
    if (out >= hidden_size) return;
    if (!failure) store_activation(view.output[blockIdx.y] + out, view.accumulation[blockIdx.y][out]);
    if constexpr (Combine) {
        const float shared = load_activation(combine.shared[blockIdx.y] + out), routed = load_activation(view.output[blockIdx.y] + out);
        float sum;
        asm volatile("v_add_f32 %0, %1, %2" : "=v"(sum) : "v"(shared), "v"(routed));
        store_activation(combine.combined[blockIdx.y] + out, sum);
    }
}

struct KnownMoeRowsPrepared {
    std::array<PreparedMoeExperts, kKnownMoeMaxRows> rows{};
    std::array<moe_q4k_a8::Bindings, kKnownMoeMaxRows> quantized_bindings{};
    std::array<moe_q4k_a8::DownBindings, kKnownMoeMaxRows> a8_down{};
    q51_known_rows::Workspace workspace;
    KnownMoeRowsDevice device;
    KnownMoeCsrDevice csr{};
};
struct KnownMoeRange { std::uintptr_t address; std::uint64_t bytes; };
void known_moe_range(KnownMoeRange r) {
    if (!r.address || !r.bytes || r.bytes > UINTPTR_MAX - r.address)
        throw std::invalid_argument("known-window routed rows owner range is invalid");
}
bool known_moe_overlap(KnownMoeRange a, KnownMoeRange b) {
    return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
hipStream_t known_moe_rows_stream() {
    const auto* scope = current_hip_launch_context();
    if (!scope || scope->graph_capture || !scope->stream || scope->stream == hipStreamLegacy ||
        scope->stream == hipStreamPerThread || !scope->deferred_moe_status.arena_base)
        throw std::invalid_argument("known-window routed rows require explicit eager D1 scope");
    require_q51_slot_eager_stream(scope->stream);
    return scope->stream;
}
// striped_weights: the caller checks the row stripes against every weight tensor once.
KnownMoeRowsPrepared prepare_known_moe_rows(std::span<const ExecutionDispatchContext> rows,
    const HipMoeExpertsLauncherConfig& config, const q51_known_rows::Workspace& extra, bool striped_weights = false) {
    // 9..32 rows: a cohort union of several sessions' windows (A8 profile only).
    if (rows.size() != 2U && rows.size() != 4U && rows.size() != 8U &&
        !(rows.size() >= 9U && rows.size() <= kKnownMoeMaxRows && known_window_cohort_active() && (config.a8_down_q51 || config.a8_down_q80)))
        throw std::invalid_argument("known-window routed rows require R2/R4/R8 (9..32: an A8 cohort union)");
    const auto layout = q51_known_rows::scratch_layout(static_cast<std::uint32_t>(rows.size()),
        config.expert_count, config.top_k);
    validate_config(config);
    if (!config.q4k_a8g32_gate_up || !config.grouped_exact_q51_down || !config.q51_slot_parallel_down ||
        !config.deferred_moe_status || config.activation != HipActivationStorage::bf16)
        throw std::invalid_argument("known-window routed rows require BF16 A8 slot-parallel Q5_1 D1 configuration");
    if (config.intermediate_size % 32U || config.intermediate_size > 768U || config.hidden_size % 64U || config.top_k > 10U)
        throw std::invalid_argument("known-window routed rows union down geometry is unsupported");
    const bool a8 = config.a8_down_q51 || config.a8_down_q80;
    if (a8 && !config.q4k_a8_quiescence)
        throw std::invalid_argument("known-window A8 routed rows require retained setup completion guard");
    (void)known_moe_rows_stream();
    const auto& first = rows.front();
    if (!first.execution_identity || !first.layer || !first.kernel_id || !first.resources ||
        !first.resources->scratch || !first.arena || !first.resources->model_manifest ||
        !first.resources->model_weights || !first.resources->tensor_access)
        throw std::invalid_argument("known-window routed rows require retained execution resources");
    const auto& memory = *first.resources->scratch;
    validate_execution_memory_binding(first.execution_identity, *first.arena, memory);
    // A cohort window's rows belong to the windows of several sessions; the union workspace
    // belongs to the owner that enqueues it, so only its extent is bound to these rows.
    const bool cohort = known_window_cohort_active();
    if (!extra.owner || (!cohort && extra.memory.execution_identity != first.execution_identity) ||
        !extra.memory.allocation_generation || extra.memory.arena_base % 256U || extra.memory.arena_bytes != layout.bytes)
        throw std::invalid_argument("known-window routed rows CSR owner identity/extent differs");
    known_moe_range({extra.memory.arena_base, extra.memory.arena_bytes});
    if (first.reads.size() != 3U || first.writes.size() != 1U || first.temporaries.size() != 1U ||
        first.reads_state != PersistentState::none || first.writes_state != PersistentState::none ||
        first.session_state_layout != first.resources->session_state_layout ||
        first.session_state_memory != first.resources->session_state_memory ||
        *first.kernel_id != kernel_id(ExecutionBindingRequest{}, config))
        throw std::invalid_argument("known-window routed rows selected serial kernel/IO differs");
    validate_model_runtime_binding_unless_scoped(*first.resources->model_manifest, *first.resources->model_weights);
    if (a8) {
        // Same admission as the serial A8 launcher on every layer.
        if (*first.layer >= 48U || first.resources->distributed_stage)
            throw std::invalid_argument("known-window A8 routed rows require canonical unpartitioned model resources");
        moe_q4k_a8::validate_a8_down_content(first.resources->model_weights->model_content_identity);
    }
    KnownMoeRowsPrepared out;
    std::array<KnownMoeRange, 2U * kKnownMoeMaxRows> writes{};
    std::array<KnownMoeRange, 3U * kKnownMoeMaxRows> reads{};
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const auto& row = rows[r];
        // Per session segment (the whole window outside a cohort): one scratch arena, state and
        // lookup owner at consecutive positions.
        const auto& head = rows[known_window_segment_start(rows, r) ? r : r - 1U];
        if (!row.resources || !row.resources->scratch || row.resources->scratch != head.resources->scratch ||
            row.resources->model_manifest != first.resources->model_manifest ||
            row.resources->model_weights != first.resources->model_weights ||
            row.resources->tensor_access != first.resources->tensor_access ||
            row.resources->session_state_layout != first.resources->session_state_layout ||
            row.resources->session_state_memory != head.resources->session_state_memory ||
            row.resources->ple_lookup != head.resources->ple_lookup ||
            row.resources->ple_hash_contract != head.resources->ple_hash_contract ||
            row.resources->ple_prefetch != head.resources->ple_prefetch ||
            row.session_state_layout != first.session_state_layout || row.session_state_memory != head.session_state_memory ||
            row.session_state_memory != row.resources->session_state_memory ||
            (row.memory && row.memory != row.resources->scratch) || row.prepared_tensor_views || row.prepared_arena_views ||
            row.execution_identity != head.execution_identity || row.step_id != first.step_id || row.op != SerialOp::moe_experts ||
            row.layer != first.layer || row.kernel_id != first.kernel_id || row.logical_rows != 1U || row.request_concurrency != 1U ||
            row.reads_state != first.reads_state || row.writes_state != first.writes_state ||
            !known_window_row_position_follows(rows, r) ||
            row.reads.size() != 3U || row.writes.size() != 1U || row.temporaries.size() != 1U ||
            !std::equal(row.reads.begin(), row.reads.end(), first.reads.begin()) ||
            row.writes[0] != first.writes[0] || row.temporaries[0] != first.temporaries[0] || !row.arena)
            throw std::invalid_argument("known-window routed rows source/position/owner differs");
        const auto& row_memory = *row.resources->scratch;
        validate_execution_memory_binding(row.execution_identity, *row.arena, row_memory);
        out.rows[r] = prepare_moe_experts(row, const_cast<HipMoeExpertsLauncherConfig*>(&config));
        const auto& p = out.rows[r];
        const auto gate_type = p.tensors.gate.manifest->ggml_type, down_type = p.tensors.down.manifest->ggml_type;
        // Float profile: Q4K gate/up + slot-parallel Q5_1 down. A8 profile:
        // Q4K (or Q5K) gate/up + the serial Q5_1/Q8_0 A8G32 down request.
        if (!p.quantized || p.tensors.fused_gate_up || p.tensors.up.manifest->ggml_type != gate_type ||
            (a8 ? !p.quantized->a8_down_request || p.quantized->a8_down_request->weight_type != down_type ||
                  (gate_type != 12U && !(gate_type == 13U && config.q5k_a8g32_gate_up) && !(gsq_a8::gate_type(gate_type) && moe_q4k_a8::q20_a8_enabled())) :
                  !p.quantized->slot_parallel_q51 || gate_type != 12U || down_type != 7U) ||
            gate_type != out.rows[0].tensors.gate.manifest->ggml_type || down_type != out.rows[0].tensors.down.manifest->ggml_type)
            throw std::invalid_argument("known-window routed rows require the serial Q4K/Q5K A8 gate/up and down formats");
        const std::array views{p.hidden, p.route_indices, p.route_weights, p.output, p.temporary};
        const auto& p0 = out.rows[0];
        const std::array base{p0.hidden, p0.route_indices, p0.route_weights, p0.output, p0.temporary};
        for (unsigned v = 0; v < views.size(); ++v) {
            const auto current = views[v]; known_moe_range({current.address, current.bytes});
            if (current.address < row_memory.arena_base || current.address - row_memory.arena_base > row_memory.arena_bytes ||
                current.bytes > row_memory.arena_bytes - (current.address - row_memory.arena_base) ||
                current.allocation_generation != row_memory.allocation_generation || base[v].bytes > UINT64_MAX - 255U)
                throw std::invalid_argument("known-window routed rows view exceeds retained physical allocation");
            // Validate the retained arena's stride, including explicit per-arena policies.
            const auto tight = (base[v].bytes + 255U) / 256U * 256U;
            const std::array row1{out.rows[r ? 1U : 0U].hidden, out.rows[r ? 1U : 0U].route_indices, out.rows[r ? 1U : 0U].route_weights,
                out.rows[r ? 1U : 0U].output, out.rows[r ? 1U : 0U].temporary};
            const auto stride = r && !cohort ? core::retained_row_stride(
                base[v].address, base[v].bytes, row1[v].address, row1[v].bytes) : tight;
            // Rows of different cohort sessions live in different arenas; every union kernel
            // addresses rows through per-row pointers, so only one session's window keeps the stripe.
            if (current.bytes != base[v].bytes || (!cohort && (stride > (UINTPTR_MAX - base[v].address) / rows.size() ||
                current.address != base[v].address + r * stride)))
                throw std::invalid_argument("known-window routed rows stripe differs from serial allocation");
        }
        const std::array tensor_pairs{p.tensors.gate, p.tensors.up, p.tensors.down};
        for (unsigned t = 0; t < tensor_pairs.size(); ++t) {
            const auto& pair = tensor_pairs[t]; const auto& dims = pair.manifest->dimensions;
            const auto width = t == 2U ? config.intermediate_size : config.hidden_size;
            const auto outputs = t == 2U ? config.hidden_size : config.intermediate_size;
            const auto required = required_weight_bytes(weight_kind(pair.manifest->ggml_type),
                width, outputs, config.expert_count);
            if (pair.manifest->layer != row.layer || dims.size() != 3U || dims[0] != width || dims[1] != outputs ||
                dims[2] != config.expert_count || pair.manifest->byte_size != required || !pair.runtime->allocation_generation ||
                pair.runtime->address % 16U)
                throw std::invalid_argument("known-window routed rows selected tensor metadata differs");
        }
        reads[3U*r] = {p.hidden.address, p.hidden.bytes};
        reads[3U*r+1U] = {p.route_indices.address, p.route_indices.bytes};
        reads[3U*r+2U] = {p.route_weights.address, p.route_weights.bytes};
        writes[2U*r] = {p.output.address, p.output.bytes}; writes[2U*r+1U] = {p.temporary.address, p.temporary.bytes};
        const auto& q = *p.quantized;
        out.quantized_bindings[r]=q.bindings;
        out.device.intermediates[r] = reinterpret_cast<const float*>(q.bindings.output.address);
        out.device.routes[r] = reinterpret_cast<const std::uint32_t*>(p.route_indices.address);
        out.device.route_weights[r] = reinterpret_cast<const float*>(p.route_weights.address);
        out.device.status[r] = reinterpret_cast<std::uint32_t*>(q.bindings.status.address);
        out.device.rawdots[r] = reinterpret_cast<float*>(p.temporary.address + q.layout.rawdots_offset);
        out.device.accumulation[r] = reinterpret_cast<float*>(p.temporary.address + q.layout.accumulation_offset);
        out.device.output[r] = reinterpret_cast<std::uint16_t*>(p.output.address);
        if (a8) out.a8_down[r] = q.a8_down_bindings;
    }
    const auto disjoint = [](KnownMoeRange a, KnownMoeRange b) {
        known_moe_range(a); known_moe_range(b);
        if (known_moe_overlap(a,b)) throw std::invalid_argument("known-window routed rows writable owner aliases storage");
    };
    const KnownMoeRange csr_range{extra.memory.arena_base, extra.memory.arena_bytes};
    for (const auto& row : rows) disjoint(csr_range, {row.resources->scratch->arena_base, row.resources->scratch->arena_bytes});
    const auto& d1 = current_hip_launch_context()->deferred_moe_status;
    disjoint(csr_range, {d1.arena_base, d1.arena_bytes});
    if (!striped_weights) for (const auto& weight : first.resources->model_weights->tensors) {
        if (!weight.address || !weight.bytes) continue;
        known_moe_range({weight.address, weight.bytes});
        disjoint(csr_range, {weight.address, weight.bytes});
        for (std::size_t w = 0; w < 2U*rows.size(); ++w) disjoint(writes[w], {weight.address, weight.bytes});
    }
    for (std::size_t r = 0; r < rows.size(); ++r) {
        if (!known_window_segment_start(rows, r)) continue;
        if (const auto* state = rows[r].resources->session_state_memory) {
            for (const auto s : {KnownMoeRange{state->host_base,state->host_bytes}, KnownMoeRange{state->device_base,state->device_bytes}})
                if (s.bytes) { disjoint(csr_range,s); for (std::size_t w=0;w<2U*rows.size();++w) disjoint(writes[w],s); }
        }
    }
    for (std::size_t w = 0; w < 2U*rows.size(); ++w) {
        disjoint(writes[w], {d1.arena_base,d1.arena_bytes});
        for (std::size_t r = 0; r < 3U*rows.size(); ++r) disjoint(writes[w], reads[r]);
        for (std::size_t other = 0; other < w; ++other) disjoint(writes[w], writes[other]);
    }
    const auto ptr = [&](std::uint64_t offset) { return reinterpret_cast<std::uint32_t*>(extra.memory.arena_base+offset); };
    out.csr = {ptr(layout.header),ptr(layout.offsets),ptr(layout.touched),ptr(layout.rows),ptr(layout.slots),
        ptr(layout.batch_experts),ptr(layout.batch_begins)};
    out.workspace=extra;
    moe_q4k_a8::known_rows::preflight(out.rows[0].quantized->request,
        std::span<const moe_q4k_a8::Bindings>(out.quantized_bindings.data(),rows.size()),extra,hip_execution_stream());
    return out;
}

// active (0 = every row): rows from active on stay out of the expert union and are not merged, so their
// routed output is zero (shared expert only); the caller never accepts them.
void known_moe_rows_enqueue(std::span<const ExecutionDispatchContext> rows,
    const HipMoeExpertsLauncherConfig& config, const KnownMoeRowsPrepared& p,
    const KnownMoeCombineDevice* combine = nullptr, const std::function<void()>* before_commit = nullptr,
    std::uint32_t active = 0) {
    const auto stream = hip_execution_stream();
    const auto count = static_cast<std::uint32_t>(rows.size());
    if (!active) active = count;
    if (active > count) throw std::invalid_argument("known-window routed rows active count exceeds the window");
    check_hip(hipPeekAtLastError(), "known-window routed rows stale error");
    moe_q4k_a8::known_rows::WindowRows window;
    for (unsigned r = 0; r < count; ++r) {
        const auto& q = *p.rows[r].quantized;
        window.route_weights[r] = {p.rows[r].route_weights.address, p.rows[r].route_weights.bytes};
        window.clear[r] = {reinterpret_cast<std::uintptr_t>(p.device.accumulation[r]), q.layout.accumulation_bytes};
        if (q.a8_down_request) {
            const auto& d = p.a8_down[r];
            window.down_codes[r] = d.codes; window.down_deltas[r] = d.deltas; window.down_sums[r] = d.sums; window.rawdots[r] = d.rawdots;
        }
    }
    const auto& first = *p.rows[0].quantized;
    if (first.a8_down_request) { window.down_type = first.a8_down_request->weight_type; window.down_weights = p.a8_down[0].weights; }
    moe_q4k_a8::known_rows::enqueue_window(first.request,
        std::span<const moe_q4k_a8::Bindings>(p.quantized_bindings.data(), count), p.workspace, window, stream, active);
    const auto weights = reinterpret_cast<const void*>(p.rows[0].tensors.down.runtime->address);
    if (!first.a8_down_request) {
    constexpr unsigned down_group = 2U, down_outputs = 4U;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(known_moe_q51_union_down_kernel<down_group, down_outputs>),
        dim3(config.hidden_size / (8U * down_outputs) * (count / down_group), count * config.top_k), dim3(256), 0, stream,
        p.device, p.csr, count, weights, config.intermediate_size, config.hidden_size);
    check_hip(hipGetLastError(), "known-window routed rows shared-decode Q5_1 down");
    }
    hipLaunchKernelGGL(known_moe_rows_merge_kernel, dim3((config.hidden_size + 255U) / 256U, active), dim3(256), 0, stream,
        p.device, p.csr, config.hidden_size, config.expert_count, config.top_k);
    check_hip(hipGetLastError(), "known-window routed rows ordered merge");
    if (combine) {
        if (before_commit && *before_commit) (*before_commit)();
        hipLaunchKernelGGL(known_moe_rows_commit_combine_kernel<true>, dim3((config.hidden_size + 255U) / 256U, count), dim3(256), 0,
            stream, p.device, p.csr, count, config.hidden_size, hip_deferred_moe_status(), *combine);
        check_hip(hipGetLastError(), "known-window routed rows output commit + MoE combine");
    } else {
        hipLaunchKernelGGL(known_moe_rows_commit_kernel, dim3((config.hidden_size + 255U) / 256U, count), dim3(256), 0, stream,
            p.device, p.csr, count, config.hidden_size, hip_deferred_moe_status());
        check_hip(hipGetLastError(), "known-window routed rows output commit");
    }
    if (config.q4k_a8_stats) {
        auto& stats = *config.q4k_a8_stats;
        if (config.batched_slot_activation) {
            stats.batched_activation_launches.fetch_add(count, std::memory_order_relaxed);
            stats.batched_activation_slots.fetch_add(count * config.top_k, std::memory_order_relaxed);
        }
        stats.quantized_submissions.fetch_add(count, std::memory_order_relaxed);
        if (first.a8_down_request) {
            (first.a8_down_request->weight_type == 7U ? stats.a8_down_q51_submissions : stats.a8_down_q80_submissions)
                .fetch_add(count, std::memory_order_relaxed);
            if (first.request.gate_type == 13U) stats.q5k_a8_gate_up_submissions.fetch_add(count, std::memory_order_relaxed);
        } else {
            stats.grouped_q51_submissions.fetch_add(count, std::memory_order_relaxed);
            stats.slot_parallel_q51_submissions.fetch_add(count, std::memory_order_relaxed);
        }
    }
}

// Known-prompt prefill over T strided rows (A8 profile).
using moe_q4k_a8::known_rows::StridedRows;
struct KnownMoePrefillDevice {
    StridedRows<const std::uint32_t> routes;
    StridedRows<const float> route_weights;
    StridedRows<std::uint32_t> status;
    StridedRows<float> rawdots, accumulation;
    StridedRows<std::uint16_t> output;
};
template <class T> __device__ __forceinline__ T* known_moe_prefill_row(StridedRows<T> v, std::uint32_t r) {
    return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(v.base) + static_cast<std::uint64_t>(r) * v.stride);
}
// known_moe_rows_merge_kernel per strided row. header[0] already holds every
// row status of the projection stages; a nonfinite sum also lands there.
__global__ void known_moe_prefill_merge_kernel(KnownMoePrefillDevice view, std::uint32_t* header,
    std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k) {
    const auto row = blockIdx.y;
    auto* status = known_moe_prefill_row(view.status, row);
    if (header[0] || *status) return;
    const auto out = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out >= hidden_size) return;
    const auto* routes = known_moe_prefill_row(view.routes, row);
    for (std::uint32_t slot = 0; slot < top_k; ++slot) if (routes[slot] >= experts) return;
    const auto* rawdots = known_moe_prefill_row(view.rawdots, row);
    const auto* route_weights = known_moe_prefill_row(view.route_weights, row);
    float dots[10];
#pragma unroll
    for (std::uint32_t slot = 0; slot < 10U; ++slot) if (slot < top_k) dots[slot] = rawdots[static_cast<std::uint64_t>(slot) * hidden_size + out];
    float current = 0.0F;
#pragma unroll
    for (std::uint32_t slot = 0; slot < 10U; ++slot) if (slot < top_k) {
        const float weight = route_weights[slot];
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(current) : "s"(weight), "v"(dots[slot]));
    }
    known_moe_prefill_row(view.accumulation, row)[out] = current;
    if ((__float_as_uint(current) & 0x7fffffffU) >= 0x7f7f8000U) { atomicOr(status, 8U); atomicOr(header, 8U); }
}
// qpf2k/moe-mi (STRIX_QPF2K_MOE_ROUTE bit 8): known_moe_prefill_merge_kernel when the routed prologue did not zero the
// accumulation view: the same kernel, plus a zero store on each path where the base returns without a store (the zero the
// prologue would have left there; the prologue clears exactly hidden words per row). Successful rows: identical.
#include "hip_moe_route_bf16.inc"
#include "hip_moe_prefill_commit.inc"
__global__ void known_moe_prefill_merge_kernel_z(KnownMoePrefillDevice view, std::uint32_t* header,
    std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k) {
    moe_prefill_merge_z_body(view, header, hidden_size, experts, top_k,
        [](auto v, unsigned r) { return known_moe_prefill_row(v, r); });
}
__global__ void known_moe_prefill_commit_kernel(KnownMoePrefillDevice view, std::uint32_t* header,
    std::uint32_t row_count, std::uint32_t hidden_size, std::uint32_t* record) {
    moe_prefill_commit_body(view, header, row_count, hidden_size, record, blockIdx.x, blockIdx.y,
        [](auto v, unsigned r) { return known_moe_prefill_row(v, r); },
        [](auto* p, float v) { store_activation(p, v); });
}
// MC entry points only: base wrappers above inline the original statement order.
template <bool Acc>
__global__ void known_moe_prefill_merge_commit_kernel(KnownMoePrefillDevice view, std::uint32_t* header,
    std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k, const unsigned* unsafe) {
    moe_prefill_merge_z_body<true, Acc>(view, header, hidden_size, experts, top_k,
        [](auto v, unsigned r) { return known_moe_prefill_row(v, r); }, unsafe);
}
template <bool Acc>
__global__ void known_moe_prefill_merge_commit_vec4_kernel(KnownMoePrefillDevice view, std::uint32_t* header,
    std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k, const unsigned* unsafe) {
    moe_prefill_merge_z_body_vec4<Acc>(view, header, hidden_size, experts, top_k,
        [](auto v, unsigned r) { return known_moe_prefill_row(v, r); }, unsafe);
}
// STRIX_PREFILL_MOE_MERGE_VEC (default on; "0", the baseline, STRIX_QPF2K_DEFAULTS=0 or a GSQ model turn it off): the MC
// merge with four outputs per thread when every row view is aligned for 16-byte FP32 / 8-byte BF16 vectors.
bool prefill_merge_vec4(const KnownMoePrefillDevice& d, std::uint32_t hidden_size, bool gsq) {
    static const bool enabled = qpf2k::long_context_item_enabled("STRIX_PREFILL_MOE_MERGE_VEC", false);
    if (!enabled || gsq || hidden_size % 4U) return false;
    const auto aligned = [](const auto& v, std::uint64_t a) { return reinterpret_cast<std::uintptr_t>(v.base) % a == 0U && v.stride % a == 0U; };
    return aligned(d.rawdots, 16U) && aligned(d.accumulation, 16U) && aligned(d.output, 8U);
}
__global__ void known_moe_prefill_merge_commit_bf16_kernel(KnownMoePrefillDevice view, std::uint32_t* header,
    std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k, const unsigned* unsafe) {
    moe_prefill_merge_z_body<true, false, true>(view, header, hidden_size, experts, top_k,
        [](auto v, unsigned r) { return known_moe_prefill_row(v, r); }, unsafe);
}
__global__ void known_moe_prefill_merge_commit_vec4_bf16_kernel(KnownMoePrefillDevice view, std::uint32_t* header,
    std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k, const unsigned* unsafe) {
    moe_prefill_merge_z_body_vec4<false, true>(view, header, hidden_size, experts, top_k,
        [](auto v, unsigned r) { return known_moe_prefill_row(v, r); }, unsafe);
}
__global__ void known_moe_prefill_merge_commit_i8_kernel(KnownMoePrefillDevice view, std::uint32_t* header,
    std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k, const unsigned* unsafe) {
    moe_prefill_merge_z_body<true, false, true, true>(view, header, hidden_size, experts, top_k,
        [](auto v, unsigned r) { return known_moe_prefill_row(v, r); }, unsafe);
}
__global__ void known_moe_prefill_merge_commit_vec4_i8_kernel(KnownMoePrefillDevice view, std::uint32_t* header,
    std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k, const unsigned* unsafe) {
    moe_prefill_merge_z_body_vec4<false, true, true>(view, header, hidden_size, experts, top_k,
        [](auto v, unsigned r) { return known_moe_prefill_row(v, r); }, unsafe);
}
// RoutePrune merge+commit (MC mode 1): the kernels above with zero-weight slots skipped.
template <bool Vec4, bool RouteBF16, bool RouteI8>
__global__ void known_moe_prefill_merge_commit_pruned_kernel(KnownMoePrefillDevice view, std::uint32_t* header,
    std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k, const unsigned* unsafe) {
    const auto row_at = [](auto v, unsigned r) { return known_moe_prefill_row(v, r); };
    if constexpr (Vec4) moe_prefill_merge_z_body_vec4<false, RouteBF16, RouteI8, true>(view, header, hidden_size, experts, top_k, row_at, unsafe);
    else moe_prefill_merge_z_body<true, false, RouteBF16, RouteI8, true>(view, header, hidden_size, experts, top_k, row_at, unsafe);
}
template <unsigned CTAs>
__global__ void known_moe_prefill_finish_kernel(KnownMoePrefillDevice view, std::uint32_t* header,
    std::uint32_t row_count, std::uint32_t hidden, std::uint32_t* record, const unsigned* unsafe) {
    moe_prefill_finish_body<CTAs>(view, header, row_count, hidden, record, unsafe,
        [](auto v, unsigned r) { return known_moe_prefill_row(v, r); },
        [](auto* p, float v) { store_activation(p, v); });
}
// STRIX_QPF2K_MOE_MC: 0/unset base merge + commit, 1 merge+commit (decoder default), 2 also writes the accumulation.
unsigned prefill_merge_commit(bool gsq) {
    static const unsigned mode = [] {
        const char* v = std::getenv("STRIX_QPF2K_MOE_MC");
        const unsigned m = v && (v[0] == '1' || v[0] == '2') && v[1] == '\0' ? static_cast<unsigned>(v[0] - '0') : 0U;
        return m;
    }();
    return qpf2k::suppressed(qpf2k::Item::moe_mc, gsq) ? 0U : mode;
}
// Retained allocations (never freed, so queued work never sees a released word). Different devices/explicit streams
// never share admission state; reuse on one stream is ordered through finish before the next prologue. The mutex only
// guards lookup/allocation: calls on one stream must be submitted serially (the prefill scheduler contract; graph
// capture and per-thread/default streams are rejected before this point).
unsigned* prefill_admission_word(hipStream_t stream) {
    struct Word { int device; hipStream_t stream; unsigned* ptr; };
    static std::mutex mutex;
    static std::vector<Word> words;
    int device=0; check_hip(hipGetDevice(&device), "MC admission device");
    const std::lock_guard<std::mutex> lock(mutex);
    for (const auto& w : words) if (w.device==device && w.stream==stream) return w.ptr;
    void* ptr=nullptr; check_hip(hipMalloc(&ptr, sizeof(unsigned)), "MC admission allocation");
    words.push_back({device,stream,static_cast<unsigned*>(ptr)});
    return static_cast<unsigned*>(ptr);
}
struct KnownMoePrefillPrepared {
    std::shared_ptr<const qwen_moe_prepared::Model> direct_b;
    moe_q4k_a8::known_rows::DirectBOperands direct;
    moe_q4k_a8::Request request;
    moe_q4k_a8::known_rows::PrefillRows rows;
    KnownMoePrefillDevice device;
    std::uint32_t clear_words = 0, down_type = 0;
    const unsigned char *gate = nullptr, *up = nullptr, *down = nullptr;
};
KnownMoePrefillPrepared prepare_known_moe_prefill(std::span<const ExecutionDispatchContext> rows,
    const HipMoeExpertsLauncherConfig& config, const q51_known_rows::Workspace& window,
    const q51_known_rows::Workspace& prefill, bool shared_admission = true) {
    const auto count = static_cast<std::uint32_t>(rows.size());
    if (count < 16U || count % 8U || count > q51_known_rows::prefill_max_rows)
        throw std::invalid_argument("known prefill routed rows require T in [16, " + std::to_string(q51_known_rows::prefill_max_rows) +
            "], a multiple of 8");
    if (!config.a8_down_q51 && !config.a8_down_q80)
        throw std::invalid_argument("known prefill routed rows require the A8 down profile");
    // Every view a kernel touches, per row; checked to be row 0 plus r * stride.
    enum View : unsigned { input, routes, route_weights, codes, deltas, sums, status, intermediates, accumulation,
        down_codes, down_deltas, down_sums, rawdots, output, temporary, view_count };
    std::vector<std::array<std::uintptr_t, view_count>> address(count);
    std::array<std::uint64_t, view_count> extent{};
    KnownMoePrefillPrepared out;
    // Interior chunks share the window admission of the first chunk (STRIX_PREFILL_MOE_SHARED_ADMISSION=0
    // admits every chunk): each row is checked to be the same step, source and serial IO as row 0,
    // its five arena views are resolved and checked like the admission does, and its derived views
    // (codes, deltas, sums, status, intermediates, accumulation, A8 down image, raw dots) are row 0's
    // offsets inside its own temporary. The first and last chunks keep the full admission, every view
    // is then checked row-striped below, and the striped writes are checked against the striped reads.
    static const bool shared_env = strix::envcfg::flag("STRIX_PREFILL_MOE_SHARED_ADMISSION", true);
    const bool shared = shared_admission && shared_env;
    enum Derived : unsigned { d_codes, d_deltas, d_sums, d_status, d_intermediates, d_accumulation,
        d_down_codes, d_down_deltas, d_down_sums, d_rawdots, derived_count };
    std::array<std::uint64_t, derived_count> derived_offset{};
    const auto& memory0 = *rows.front().resources->scratch;
    // Segmented prefill (inside a cohort scope): consecutive rows of one session form a segment over that
    // session's own resources and positions; every segment uses the same block arena (memory0).
    std::vector<std::uint32_t> segment_head(count);
    for (std::uint32_t r = 0; r < count; ++r) segment_head[r] = known_window_segment_start(rows, r) ? r : segment_head[r - 1U];
    // Row views of the compiled prefill arena list the same unique uses in the same order:
    // each row's allocation indices are the previous row's (hinted lookup, no arena search).
    std::array<std::size_t, 5> hint{};
    const auto own_views = [&](std::uint32_t r) {
        const auto& row = rows[r]; const auto& first = rows.front(); const auto& head = rows[segment_head[r]];
        if (!row.resources || row.resources != head.resources || row.resources->scratch != &memory0 ||
            (row.memory && row.memory != row.resources->scratch) || row.prepared_tensor_views || row.prepared_arena_views ||
            row.session_state_layout != first.session_state_layout || row.session_state_memory != head.session_state_memory ||
            row.execution_identity != first.execution_identity || row.step_id != first.step_id || row.op != SerialOp::moe_experts ||
            row.layer != first.layer || row.kernel_id != first.kernel_id || row.logical_rows != 1U || row.request_concurrency != 1U ||
            row.reads_state != first.reads_state || row.writes_state != first.writes_state ||
            row.position != head.position + (r - segment_head[r]) || row.reads.size() != 3U || row.writes.size() != 1U || row.temporaries.size() != 1U ||
            !std::equal(row.reads.begin(), row.reads.end(), first.reads.begin()) ||
            row.writes[0] != first.writes[0] || row.temporaries[0] != first.temporaries[0] || !row.arena)
            throw std::invalid_argument("known prefill routed rows source/position/owner differs");
        validate_execution_memory_binding(row.execution_identity, *row.arena, memory0);
        const std::array base{resolve_dispatch_read_hinted(row, 0, hint[0]), resolve_dispatch_read_hinted(row, 1, hint[1]),
            resolve_dispatch_read_hinted(row, 2, hint[2]), resolve_dispatch_write_hinted(row, 0, hint[3]), resolve_dispatch_temporary_hinted(row, 0, hint[4])};
        const std::array index{input, routes, route_weights, output, temporary};
        for (unsigned v = 0; v < base.size(); ++v) {
            const auto& current = base[v]; known_moe_range({current.address, current.bytes});
            if (current.address < memory0.arena_base || current.address - memory0.arena_base > memory0.arena_bytes ||
                current.bytes > memory0.arena_bytes - (current.address - memory0.arena_base) ||
                current.allocation_generation != memory0.allocation_generation || current.bytes != extent[index[v]])
                throw std::invalid_argument("known prefill routed row view exceeds retained physical allocation");
            address[r][index[v]] = current.address;
        }
        const std::array derived{codes, deltas, sums, status, intermediates, accumulation, down_codes, down_deltas, down_sums, rawdots};
        for (unsigned d = 0; d < derived_count; ++d) address[r][derived[d]] = base[4].address + derived_offset[d];
    };
    for (std::uint32_t chunk = 0; chunk < count; chunk += 8U) {
        if (shared && chunk && chunk + 8U < count) {
            if (!known_window_row_position_follows(rows, chunk))
                throw std::invalid_argument("known prefill routed rows positions are not consecutive");
            for (std::uint32_t i = 0; i < 8U; ++i) own_views(chunk + i);
            continue;
        }
        // Window admission of each 8-row chunk (positions consecutive inside the chunk).
        const auto p = prepare_known_moe_rows(rows.subspan(chunk, 8U), config, window, true);
        if (chunk && !known_window_row_position_follows(rows, chunk))
            throw std::invalid_argument("known prefill routed rows positions are not consecutive");
        for (std::uint32_t i = 0; i < 8U; ++i) {
            const auto& e = p.rows[i];
            const auto& q = *e.quantized;
            const auto& b = p.quantized_bindings[i];
            const auto& d = p.a8_down[i];
            if (!q.a8_down_request || d.rawdots.address != reinterpret_cast<std::uintptr_t>(p.device.rawdots[i]))
                throw std::invalid_argument("known prefill routed rows require the A8 down image");
            const std::array<moe_q4k_a8::DeviceBytes, view_count> views{
                moe_q4k_a8::DeviceBytes{e.hidden.address, e.hidden.bytes}, {e.route_indices.address, e.route_indices.bytes},
                {e.route_weights.address, e.route_weights.bytes}, b.codes, b.deltas, b.sums, b.status, b.output,
                {reinterpret_cast<std::uintptr_t>(p.device.accumulation[i]), q.layout.accumulation_bytes},
                d.codes, d.deltas, d.sums, d.rawdots, {e.output.address, e.output.bytes},
                {e.temporary.address, e.temporary.bytes}};
            for (unsigned v = 0; v < view_count; ++v) {
                if (chunk + i && views[v].bytes != extent[v])
                    throw std::invalid_argument("known prefill routed rows view extent differs across rows");
                extent[v] = views[v].bytes; address[chunk + i][v] = views[v].address;
            }
        }
        if (!chunk) {
            // Row 0's derived views as offsets inside its own temporary (the layout is per step).
            const std::array derived{codes, deltas, sums, status, intermediates, accumulation, down_codes, down_deltas, down_sums, rawdots};
            const auto temp = address[0][temporary];
            for (unsigned d = 0; d < derived_count; ++d) {
                const auto at = address[0][derived[d]];
                if (at < temp || at - temp > extent[temporary] || extent[derived[d]] > extent[temporary] - (at - temp))
                    throw std::invalid_argument("known prefill routed row derived view lies outside its temporary");
                derived_offset[d] = at - temp;
            }
            const auto& q = *p.rows[0].quantized;
            out.request = q.request;
            out.clear_words = static_cast<std::uint32_t>(q.layout.accumulation_bytes / 4U);
            out.down_type = q.a8_down_request->weight_type;
            out.gate = reinterpret_cast<const unsigned char*>(p.quantized_bindings[0].gate.address);
            out.up = reinterpret_cast<const unsigned char*>(p.quantized_bindings[0].up.address);
            out.down = reinterpret_cast<const unsigned char*>(p.a8_down[0].weights.address);
        } else if (p.quantized_bindings[0].gate.address != reinterpret_cast<std::uintptr_t>(out.gate) ||
            p.quantized_bindings[0].up.address != reinterpret_cast<std::uintptr_t>(out.up) ||
            p.a8_down[0].weights.address != reinterpret_cast<std::uintptr_t>(out.down) ||
            p.rows[0].quantized->request.gate_type != out.request.gate_type ||
            p.rows[0].quantized->a8_down_request->weight_type != out.down_type)
            throw std::invalid_argument("known prefill routed rows weights differ across chunks");
    }
    std::array<std::uint64_t, view_count> stride{};
    for (unsigned v = 0; v < view_count; ++v) {
        stride[v] = address[1][v] - address[0][v];
        if (address[1][v] <= address[0][v] || stride[v] < extent[v])
            throw std::invalid_argument("known prefill routed rows views are not row-striped");
        for (std::uint32_t r = 0; r < count; ++r)
            if (address[r][v] != address[0][v] + static_cast<std::uint64_t>(r) * stride[v])
                throw std::invalid_argument("known prefill routed rows view stride differs");
    }
    if (shared && count > 16U) {
        // Shared rows skipped the per-chunk cross-view checks: the striped writes (output, temporary)
        // must not meet each other or any striped read (input, routes, route weights); interleaved
        // stripes fall back to the per-chunk admission.
        const auto range = [&](View v) { return KnownMoeRange{address[0][v], (count - 1U) * stride[v] + extent[v]}; };
        const std::array<KnownMoeRange, 2> written{range(output), range(temporary)};
        bool apart = !known_moe_overlap(written[0], written[1]);
        for (const auto& w : written)
            for (const auto read : {range(input), range(routes), range(route_weights)}) apart = apart && !known_moe_overlap(w, read);
        if (!apart) return prepare_known_moe_prefill(rows, config, window, prefill, false);
    }
    const auto layout = q51_known_rows::scratch_layout(count, config.expert_count, config.top_k, true);
    const auto& first = rows.front();
    const auto& memory = *first.resources->scratch;
    if (!prefill.owner || prefill.memory.execution_identity != first.execution_identity ||
        !prefill.memory.allocation_generation || prefill.memory.arena_base % 256U ||
        prefill.memory.arena_bytes != layout.bytes || prefill.memory.arena_bytes > UINTPTR_MAX - prefill.memory.arena_base)
        throw std::invalid_argument("known prefill routed rows CSR owner identity/extent differs");
    const KnownMoeRange csr_range{prefill.memory.arena_base, prefill.memory.arena_bytes};
    const auto disjoint = [&](KnownMoeRange other) {
        if (other.bytes && known_moe_overlap(csr_range, other))
            throw std::invalid_argument("known prefill routed rows CSR aliases storage");
    };
    disjoint({memory.arena_base, memory.arena_bytes});
    disjoint({window.memory.arena_base, window.memory.arena_bytes});
    const auto& d1 = current_hip_launch_context()->deferred_moe_status;
    disjoint({d1.arena_base, d1.arena_bytes});
    // The chunks skipped their weight scan: the writable stripes (BF16 output and the
    // temporary holding every other written view) of all T rows against each tensor.
    const std::array<KnownMoeRange, 2> stripes{
        KnownMoeRange{address[0][output], (count - 1U) * stride[output] + extent[output]},
        KnownMoeRange{address[0][temporary], (count - 1U) * stride[temporary] + extent[temporary]}};
    const KnownMoeRange window_csr{window.memory.arena_base, window.memory.arena_bytes};
    for (const auto& weight : first.resources->model_weights->tensors) {
        if (!weight.address || !weight.bytes) continue;
        const KnownMoeRange w{weight.address, weight.bytes};
        known_moe_range(w); disjoint(w);
        if (known_moe_overlap(window_csr, w)) throw std::invalid_argument("known prefill routed rows window CSR aliases weights");
        for (const auto& stripe : stripes) if (known_moe_overlap(stripe, w))
            throw std::invalid_argument("known prefill routed rows writable stripe aliases weights");
    }
    if (const auto* state = first.resources->session_state_memory) {
        disjoint({state->host_base, state->host_bytes}); disjoint({state->device_base, state->device_bytes});
    }
    const auto strided = [&]<class T>(View v) { return StridedRows<T>{reinterpret_cast<T*>(address[0][v]), stride[v]}; };
    auto& r = out.rows;
    r.input = strided.template operator()<const void>(input);
    r.routes = strided.template operator()<const unsigned>(routes);
    r.route_weights = strided.template operator()<const float>(route_weights);
    r.codes = strided.template operator()<signed char>(codes);
    r.deltas = strided.template operator()<float>(deltas);
    r.sums = strided.template operator()<int>(sums);
    r.status = strided.template operator()<unsigned>(status);
    r.output = strided.template operator()<float>(intermediates);
    r.clear = strided.template operator()<float>(accumulation);
    r.down_codes = strided.template operator()<signed char>(down_codes);
    r.down_deltas = strided.template operator()<float>(down_deltas);
    r.down_sums = strided.template operator()<int>(down_sums);
    r.rawdots = strided.template operator()<float>(rawdots);
    const auto ptr = [&](std::uint64_t offset) { return reinterpret_cast<unsigned*>(prefill.memory.arena_base + offset); };
    r.csr = {ptr(layout.header), ptr(layout.offsets), ptr(layout.touched), ptr(layout.rows), ptr(layout.slots),
        ptr(layout.batch_experts), ptr(layout.batch_begins)};
    if (const auto& prepared=first.resources->model_weights->qwen_moe_prepared;
        prepared && prepared->model_hash==first.resources->model_weights->model_content_identity) {
        out.direct_b=prepared;
        namespace qp=qwen_moe_prepared;
        const qp::Spec spec{qp::Format::q4_k,qp::Role::gate,out.request.shape.experts,out.request.shape.rows,out.request.shape.hidden};
        if(out.request.gate_type==12U && out.request.up_type==12U) {
            const auto* g=prepared->find(reinterpret_cast<std::uintptr_t>(out.gate),qp::Role::gate,spec);
            const auto* u=prepared->find(reinterpret_cast<std::uintptr_t>(out.up),qp::Role::up,spec);
            if(g&&u) {
                out.direct.gate=reinterpret_cast<const unsigned char*>(g->allocation->address());
                out.direct.up=reinterpret_cast<const unsigned char*>(u->allocation->address());
            }
        }
    }
    out.device = {strided.template operator()<const std::uint32_t>(routes), strided.template operator()<const float>(route_weights),
        strided.template operator()<std::uint32_t>(status), strided.template operator()<float>(rawdots),
        strided.template operator()<float>(accumulation), strided.template operator()<std::uint16_t>(output)};
    return out;
}
// Measurement only (STRIX_MOE_ROUTE_CENSUS=<file>): appends each prefill MoE call's routes and route weights
// (u32 magic, rows, top_k, experts; then rows*top_k f32 weights and u32 expert ids). Synchronizes the stream.
static void known_moe_route_census(std::uint32_t count, const HipMoeExpertsLauncherConfig& config, const KnownMoePrefillPrepared& p,
                                   hipStream_t stream) {
    static const char* const path = strix::envcfg::text("STRIX_MOE_ROUTE_CENSUS");
    if (!path || !*path || !count) return;
    check_hip(hipStreamSynchronize(stream), "route census sync");
    const std::uint32_t k = config.top_k;
    std::vector<float> w(std::size_t{count} * k); std::vector<std::uint32_t> e(w.size());
    const auto pull = [&](const void* base, std::uint64_t stride, void* out) {
        const std::size_t span = std::size_t{count - 1U} * stride + k * 4U;
        if (stride == k * 4U) { check_hip(hipMemcpy(out, base, span, hipMemcpyDeviceToHost), "route census copy"); return; }
        std::vector<unsigned char> tmp(span);
        check_hip(hipMemcpy(tmp.data(), base, span, hipMemcpyDeviceToHost), "route census copy");
        for (std::uint32_t r = 0; r < count; ++r) std::memcpy(static_cast<char*>(out) + std::size_t{r} * k * 4U, tmp.data() + r * stride, k * 4U);
    };
    pull(p.rows.route_weights.base, p.rows.route_weights.stride, w.data());
    pull(p.rows.routes.base, p.rows.routes.stride, e.data());
    static moe_core::RouteCensusWriter<> writer(path);
    writer.append(count, k, static_cast<std::uint32_t>(config.expert_count), w.data(), e.data());
}
void known_moe_prefill_enqueue(std::uint32_t count, const HipMoeExpertsLauncherConfig& config, const KnownMoePrefillPrepared& p) {
    const auto stream = hip_execution_stream();
    known_moe_route_census(count, config, p, stream);
    const bool gsq = qpf2k::gsq_model || gsq_a8::gate_type(p.request.gate_type);
    check_hip(hipPeekAtLastError(), "known prefill routed rows stale error");
    bool committed = false;
    auto rows = p.rows;
    rows.route_bf16 = config.prefill_moe_route_bf16;
    rows.route_i8 = rows.route_bf16 && strix::moe_route_i8_enabled();
    if (rows.route_bf16 && (!config.prefill_f16wmma || gsq || prefill_merge_commit(gsq)!=1U))
        throw std::invalid_argument("MoE route BF16 requires fast DA16 prefill and MC=1");
    if (prefill_merge_commit(gsq)) rows.unsafe = prefill_admission_word(stream);
    // RoutePrune (FAST, default off): supported on the DA16 merge+commit tail only (MC 1), whose merge skips pruned slots.
    const bool prune = config.prefill_moe_route_prune.active();
    if (prune && (!config.prefill_f16wmma || gsq || prefill_merge_commit(gsq) != 1U))
        throw std::invalid_argument("MoE route prune requires fast DA16 prefill and MC=1");
    moe_q4k_a8::known_rows::set_prefill_route_prune(prune ? config.prefill_moe_route_prune : strix::moe_core::RoutePrune{});
    try {
        moe_q4k_a8::known_rows::enqueue_prefill(p.request, rows, count, p.clear_words, p.gate, p.up, p.down_type, p.down, stream,
            config.prefill_f16wmma ? 3 : -1, p.direct);
        // STRIX_QPF2K_SCHED bit 64: a stage hook the call's path did not reach runs here, before the merge.
        moe_q4k_a8::known_rows::prefill_stage_reached(3U, stream);
        if(p.direct_b) {
            const bool ran=moe_q4k_a8::known_rows::prefill_direct_b_ran();
            if(ran)++p.direct_b->engaged;else ++p.direct_b->declined;
            p.direct_b->reason=moe_q4k_a8::known_rows::prefill_direct_b_reason();
        }
    } catch (...) {
        moe_q4k_a8::known_rows::set_prefill_stage_hook(nullptr, nullptr, 0U); // never leave it armed for a later call
        throw;
    }
    // qpf2k core2 bit 9 (STRIX_QPF2K_MOE): the down already ran the ordered merge (last arrival); its guards and non-finite
    // flag run here instead (same accumulation, status and header bits).
    if (prune && (moe_q4k_a8::known_rows::prefill_fused_merge() || !moe_q4k_a8::known_rows::prefill_certificate_produced()))
        throw std::invalid_argument("MoE route prune: this call did not take the certified merge+commit tail");
    if (moe_q4k_a8::known_rows::prefill_fused_merge()) {
        // The fused tail writes PrefillRows::clear; the commit reads the accumulation view: they must be the same rows.
        if (reinterpret_cast<std::uintptr_t>(p.rows.clear.base) != reinterpret_cast<std::uintptr_t>(p.device.accumulation.base) ||
            p.rows.clear.stride != p.device.accumulation.stride)
            throw std::logic_error("known prefill fused merge: accumulation view differs from the prologue clear view");
        moe_q4k_a8::known_rows::enqueue_prefill_merge_check(p.rows, count, config.hidden_size, config.expert_count, config.top_k, stream);
    }
    // Only on the default chain whose merge is merge_z (ROUTE bit 8: the prologue left the accumulation unzeroed).
    else if (const auto mc = moe_q4k_a8::known_rows::prefill_certificate_produced() ? prefill_merge_commit(gsq) : 0U) {
        static const bool witness = [mc] { std::fprintf(stderr, "qpf2k moe merge+commit active (mode %u)\n", mc); return true; }();
        (void)witness;
        // The fallback choice is device-side and per call. No host fallback count/witness can observe it
        // without a readback; do not add synchronization or claim a host-observed fallback.
        if (prune) {
            const bool vec4 = prefill_merge_vec4(p.device, config.hidden_size, gsq);
            const dim3 grid(vec4 ? (config.hidden_size / 4U + 127U) / 128U : (config.hidden_size + 255U) / 256U, count), block(vec4 ? 128 : 256);
#define STRIX_PRUNED_MERGE(V, B, I) hipLaunchKernelGGL(HIP_KERNEL_NAME(known_moe_prefill_merge_commit_pruned_kernel<V, B, I>), grid, block, 0, \
                stream, p.device, p.rows.csr.summary, config.hidden_size, config.expert_count, config.top_k, rows.unsafe)
            if (rows.route_i8) { if (vec4) STRIX_PRUNED_MERGE(true, true, true); else STRIX_PRUNED_MERGE(false, true, true); }
            else if (rows.route_bf16) { if (vec4) STRIX_PRUNED_MERGE(true, true, false); else STRIX_PRUNED_MERGE(false, true, false); }
            else { if (vec4) STRIX_PRUNED_MERGE(true, false, false); else STRIX_PRUNED_MERGE(false, false, false); }
#undef STRIX_PRUNED_MERGE
        }
        else if (rows.route_i8) {
            if (prefill_merge_vec4(p.device, config.hidden_size, gsq))
                hipLaunchKernelGGL(known_moe_prefill_merge_commit_vec4_i8_kernel, dim3((config.hidden_size / 4U + 127U) / 128U, count), dim3(128), 0,
                    stream, p.device, p.rows.csr.summary, config.hidden_size, config.expert_count, config.top_k, rows.unsafe);
            else
                hipLaunchKernelGGL(known_moe_prefill_merge_commit_i8_kernel, dim3((config.hidden_size + 255U) / 256U, count), dim3(256), 0,
                    stream, p.device, p.rows.csr.summary, config.hidden_size, config.expert_count, config.top_k, rows.unsafe);
        }
        else if (rows.route_bf16) {
            if (prefill_merge_vec4(p.device, config.hidden_size, gsq))
                hipLaunchKernelGGL(known_moe_prefill_merge_commit_vec4_bf16_kernel, dim3((config.hidden_size / 4U + 127U) / 128U, count), dim3(128), 0,
                    stream, p.device, p.rows.csr.summary, config.hidden_size, config.expert_count, config.top_k, rows.unsafe);
            else
                hipLaunchKernelGGL(known_moe_prefill_merge_commit_bf16_kernel, dim3((config.hidden_size + 255U) / 256U, count), dim3(256), 0,
                    stream, p.device, p.rows.csr.summary, config.hidden_size, config.expert_count, config.top_k, rows.unsafe);
        }
        else if (prefill_merge_vec4(p.device, config.hidden_size, gsq)) {
            static const bool vec_witness = [] { std::fprintf(stderr, "qwen moe merge vec4 active\n"); return true; }();
            (void)vec_witness;
            const dim3 grid((config.hidden_size / 4U + 127U) / 128U, count);
            if (mc == 2U)
                hipLaunchKernelGGL(known_moe_prefill_merge_commit_vec4_kernel<true>, grid, dim3(128), 0,
                    stream, p.device, p.rows.csr.summary, config.hidden_size, config.expert_count, config.top_k, rows.unsafe);
            else
                hipLaunchKernelGGL(known_moe_prefill_merge_commit_vec4_kernel<false>, grid, dim3(128), 0,
                    stream, p.device, p.rows.csr.summary, config.hidden_size, config.expert_count, config.top_k, rows.unsafe);
        }
        else if (mc == 2U)
            hipLaunchKernelGGL(known_moe_prefill_merge_commit_kernel<true>, dim3((config.hidden_size + 255U) / 256U, count), dim3(256), 0,
                stream, p.device, p.rows.csr.summary, config.hidden_size, config.expert_count, config.top_k, rows.unsafe);
        else
            hipLaunchKernelGGL(known_moe_prefill_merge_commit_kernel<false>, dim3((config.hidden_size + 255U) / 256U, count), dim3(256), 0,
                stream, p.device, p.rows.csr.summary, config.hidden_size, config.expert_count, config.top_k, rows.unsafe);
        check_hip(hipGetLastError(), "known prefill routed rows ordered merge + commit");
        hipLaunchKernelGGL(known_moe_prefill_finish_kernel<32>, dim3(32), dim3(256), 0, stream,
            p.device, p.rows.csr.summary, count, config.hidden_size, hip_deferred_moe_status(), rows.unsafe);
        check_hip(hipGetLastError(), "known prefill routed rows commit record");
        committed = true;
    }
    else {
    if (prefill_merge_commit(gsq)) {
        static const bool witness=[] { std::fprintf(stderr,"qpf2k moe MC base tail (certificate not produced)\n"); return true; }();
        (void)witness;
    }
    if (moe_q4k_a8::known_rows::prefill_clear_skipped()) // qpf2k/moe-mi ROUTE bit 8: the zeros move into the merge
        hipLaunchKernelGGL(known_moe_prefill_merge_kernel_z, dim3((config.hidden_size + 255U) / 256U, count), dim3(256), 0, stream,
            p.device, p.rows.csr.summary, config.hidden_size, config.expert_count, config.top_k);
    else
    hipLaunchKernelGGL(known_moe_prefill_merge_kernel, dim3((config.hidden_size + 255U) / 256U, count), dim3(256), 0, stream,
        p.device, p.rows.csr.summary, config.hidden_size, config.expert_count, config.top_k);
    check_hip(hipGetLastError(), "known prefill routed rows ordered merge");
    }
    if (!committed) {
    hipLaunchKernelGGL(known_moe_prefill_commit_kernel, dim3((config.hidden_size + 255U) / 256U, count), dim3(256), 0, stream,
        p.device, p.rows.csr.summary, count, config.hidden_size, hip_deferred_moe_status());
    check_hip(hipGetLastError(), "known prefill routed rows output commit");
    }
    if (config.q4k_a8_stats) {
        auto& stats = *config.q4k_a8_stats;
        if (config.batched_slot_activation) {
            stats.batched_activation_launches.fetch_add(count, std::memory_order_relaxed);
            stats.batched_activation_slots.fetch_add(count * config.top_k, std::memory_order_relaxed);
        }
        stats.quantized_submissions.fetch_add(count, std::memory_order_relaxed);
        (p.down_type == 7U ? stats.a8_down_q51_submissions : stats.a8_down_q80_submissions).fetch_add(count, std::memory_order_relaxed);
        if (p.request.gate_type == 13U) stats.q5k_a8_gate_up_submissions.fetch_add(count, std::memory_order_relaxed);
        using Form = moe_q4k_a8::known_rows::PrefillKernelForm;
        const auto form = moe_q4k_a8::known_rows::prefill_kernel_form(p.request, p.down_type, config.prefill_f16wmma ? 3 : -1);
        if (form == Form::f16_wmma) stats.prefill_f16wmma_submissions.fetch_add(count, std::memory_order_relaxed);
        if (gsq_a8::gate_type(p.request.gate_type)) {
            (form == Form::f16_wmma ? stats.prefill_gsq_f16wmma_submissions : form == Form::iu8_wmma ? stats.prefill_gsq_iu8_submissions
                : stats.prefill_gsq_dot4_submissions).fetch_add(count, std::memory_order_relaxed);
            const bool batch128 = form == Form::f16_wmma && (moe_q4k_a8::known_rows::prefill_effective_tune(p.request, count, p.down_type, 3) &
                moe_q4k_a8::known_rows::gsq_f16_tune_mask & 2U) != 0U;
            if (batch128) stats.prefill_gsq_f16wmma_batch128_submissions.fetch_add(count, std::memory_order_relaxed);
            auto& global = moe_q4k_a8::gsq_prefill_rows();
            (form == Form::f16_wmma ? global.f16wmma : form == Form::iu8_wmma ? global.iu8wmma : global.dot4).fetch_add(count, std::memory_order_relaxed);
            if (batch128) global.f16wmma_batch128.fetch_add(count, std::memory_order_relaxed);
        }
    }
}
#endif
} // namespace

void moe_experts_known_window_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipMoeExpertsLauncherConfig& config, const q51_known_rows::Workspace& workspace) {
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    (void)prepare_known_moe_rows(rows,config,workspace);
#else
    (void)rows; (void)config; (void)workspace;
    throw std::invalid_argument("known-window routed rows require built A8 runtime");
#endif
}
void moe_experts_known_window_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipMoeExpertsLauncherConfig& config, const q51_known_rows::Workspace& workspace, std::uint32_t active_rows) {
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    const auto prepared=prepare_known_moe_rows(rows,config,workspace);
    known_moe_rows_enqueue(rows,config,prepared,nullptr,nullptr,active_rows);
#else
    (void)rows; (void)config; (void)workspace; (void)active_rows;
    throw std::invalid_argument("known-window routed rows require built A8 runtime");
#endif
}
bool moe_experts_known_window_rows_launch_combined(std::span<const ExecutionDispatchContext> rows,
    const HipMoeExpertsLauncherConfig& config, const q51_known_rows::Workspace& workspace,
    std::span<const void* const> combine_shared, std::span<const void* const> combine_routed,
    std::span<void* const> combine_output, const std::function<void()>& before_commit, std::uint32_t active_rows) {
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    const auto prepared=prepare_known_moe_rows(rows,config,workspace);
    const auto count=static_cast<std::uint32_t>(rows.size());
    const auto decline=[](const char* why) {
        if (std::getenv("STRIX_WINDOW_FOLD_VERBOSE")) std::fprintf(stderr, "window commit + combine fold declined: %s\n", why);
        return false;
    };
    if (combine_shared.size()!=count || combine_routed.size()!=count || combine_output.size()!=count)
        return decline("row counts differ");
    const auto at=[](const void* x){return reinterpret_cast<std::uintptr_t>(x);};
    const std::uint64_t row_bytes=static_cast<std::uint64_t>(config.hidden_size)*2U;
    const auto overlap=[](std::uintptr_t a,std::uint64_t an,std::uintptr_t b,std::uint64_t bn){return a<b+bn && b<a+an;};
    KnownMoeCombineDevice combine{};
    for (std::uint32_t t=0;t<count;++t) {
        // The combine reads this step's routed output rows; outputs are BF16 rows.
        if (combine_routed[t]!=prepared.device.output[t]) return decline("the combine does not read the routed output");
        combine.shared[t]=static_cast<const std::uint16_t*>(combine_shared[t]);
        combine.combined[t]=static_cast<std::uint16_t*>(combine_output[t]);
    }
    // Each element's thread reads its shared and routed values and the accumulation before storing the
    // combined value: a combined row may replace its own row's shared or routed row exactly, nothing else.
    for (std::uint32_t t=0;t<count;++t) {
        const auto c=at(combine.combined[t]);
        for (std::uint32_t o=0;o<count;++o) {
            const auto acc=at(prepared.device.accumulation[o]);
            if (overlap(c,row_bytes,acc,static_cast<std::uint64_t>(config.hidden_size)*4U)) return decline("combined row aliases an accumulation");
            for (const auto x:{at(combine.shared[o]),at(prepared.device.output[o]),at(combine.combined[o])}) {
                if (o==t && x==c) continue;
                if (overlap(c,row_bytes,x,row_bytes)) return decline("combined row aliases another row");
            }
        }
    }
    known_moe_rows_enqueue(rows,config,prepared,&combine,&before_commit,active_rows);
    return true;
#else
    (void)rows; (void)config; (void)workspace; (void)combine_shared; (void)combine_routed; (void)combine_output; (void)before_commit; (void)active_rows;
    throw std::invalid_argument("known-window routed rows require built A8 runtime");
#endif
}
void moe_experts_known_prefill_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipMoeExpertsLauncherConfig& config, const q51_known_rows::Workspace& window, const q51_known_rows::Workspace& prefill) {
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    (void)prepare_known_moe_prefill(rows,config,window,prefill);
#else
    (void)rows; (void)config; (void)window; (void)prefill;
    throw std::invalid_argument("known prefill routed rows require built A8 runtime");
#endif
}
void moe_experts_known_prefill_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipMoeExpertsLauncherConfig& config, const q51_known_rows::Workspace& window, const q51_known_rows::Workspace& prefill) {
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    const auto prepared=prepare_known_moe_prefill(rows,config,window,prefill);
    known_moe_prefill_enqueue(static_cast<std::uint32_t>(rows.size()),config,prepared);
#else
    (void)rows; (void)config; (void)window; (void)prefill;
    throw std::invalid_argument("known prefill routed rows require built A8 runtime");
#endif
}
std::array<double, 2> known_moe_commit_combine_selftest(std::uint32_t rows, std::uint32_t flavor, std::uint32_t seed) {
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    constexpr std::uint32_t hidden = 2560U, header_words = 64U, half = hidden / 2U;
    if (rows == 0U || rows > 8U || flavor > 8U) throw std::invalid_argument("commit + combine self-test: rows or flavor");
    std::uint64_t state = 0x9e3779b97f4a7c15ULL ^ (static_cast<std::uint64_t>(seed) << 19U) ^ (static_cast<std::uint64_t>(flavor) << 3U);
    const auto next = [&] { state ^= state << 13U; state ^= state >> 7U; state ^= state << 17U; return state; };
    // NaNs keep payload bits above bit 16 (they survive the BF16 store): quiet and signalling, both signs.
    const auto nan32 = [&] {
        const auto payload = static_cast<std::uint32_t>(next() % 0x7fU + 1U) << 16U | static_cast<std::uint32_t>(next() & 0xffffU);
        return (next() % 2U ? 0x80000000U : 0U) | 0x7f800000U | payload;
    };
    const auto special32 = [&]() -> std::uint32_t {
        switch (next() % 8U) {
        case 0: return 0x7f800000U; case 1: return 0xff800000U; case 2: return static_cast<std::uint32_t>(next() % 0x7fffffU + 1U);
        case 3: return 0x80000000U; case 4: return 0x7f7fffffU; case 5: return 0x7f7f8000U; default: return nan32();
        }
    };
    const auto finite32 = [&] {
        const float v = static_cast<float>(static_cast<double>(next() >> 40U) / 16777216.0 * 16.0 - 8.0);
        std::uint32_t u; std::memcpy(&u, &v, 4); return u;
    };
    const bool specials = flavor != 0U;
    const auto value32 = [&] { return specials && next() % 2U ? special32() : finite32(); };
    const auto value16 = [&] { return static_cast<std::uint16_t>(value32() >> 16U); };
    // Per row: status, accumulation (F32), routed output, shared and combined rows (BF16); then the CSR header and D1.
    const std::uint64_t row_words = 1U + hidden + 3U * half;
    const std::uint64_t words = row_words * rows + header_words + 2U;
    std::vector<std::uint32_t> image(words);
    const auto bad = static_cast<std::uint32_t>(next() % rows);
    for (std::uint32_t r = 0; r < rows; ++r) {
        auto* row = image.data() + r * row_words;
        row[0] = flavor == 3U && r == bad ? 2U : flavor == 5U && r == bad ? 8U : 0U;
        for (std::uint32_t i = 0; i < hidden; ++i) row[1U + i] = value32();
        // Flavor 2: BF16 rounding ties, carries into the exponent and overflow to Inf.
        if (flavor == 2U)
            for (std::uint32_t i = 0; i < hidden; ++i) {
                const auto pick = next() % 4U;
                row[1U + i] = (next() % 2U ? 0x80000000U : 0U) | (pick == 0U ? 0x7f7f8000U : pick == 1U ? 0x3f7fffffU :
                    0x3f808000U + static_cast<std::uint32_t>(next() % 3U) * 0x8000U);
            }
        for (std::uint32_t i = 0; i < 3U * half; ++i) row[1U + hidden + i] = static_cast<std::uint32_t>(value16()) | static_cast<std::uint32_t>(value16()) << 16U;
    }
    auto* header = image.data() + row_words * rows;
    for (std::uint32_t i = 0; i < 24U; ++i) header[i] = i == 0U ? 0U : static_cast<std::uint32_t>(next() % 97U);
    if (flavor == 4U) header[0] = 1U;
    // Flavor 8: every shared and routed element is a NaN pair (shared payload vs routed payload).
    if (flavor == 8U)
        for (std::uint32_t r = 0; r < rows; ++r) {
            auto* row = image.data() + r * row_words;
            for (std::uint32_t i = 0; i < hidden; ++i) row[1U + i] = nan32();
            for (std::uint32_t i = 0; i < half; ++i)
                row[1U + hidden + half + i] = (nan32() >> 16U) | (nan32() & 0xffff0000U);
        }
    std::uint32_t* device[2] = {};
    for (auto& d : device) {
        check_hip(hipMalloc(&d, words * 4U), "commit + combine self-test allocation");
        check_hip(hipMemcpy(d, image.data(), words * 4U, hipMemcpyHostToDevice), "commit + combine self-test upload");
    }
    struct Arm { KnownMoeRowsDevice view{}; KnownMoeCsrDevice csr{}; KnownMoeCombineDevice combine{}; ElementwiseRowsViews views{}; std::uint32_t* record; };
    const auto arm = [&](std::uint32_t* base) {
        Arm a{};
        for (std::uint32_t r = 0; r < rows; ++r) {
            auto* row = base + r * row_words;
            a.view.status[r] = row;
            a.view.accumulation[r] = reinterpret_cast<float*>(row + 1U);
            a.view.output[r] = reinterpret_cast<std::uint16_t*>(row + 1U + hidden);
            auto* shared = reinterpret_cast<std::uint16_t*>(row + 1U + hidden + half);
            // Flavors 6 / 7: the combined row replaces the row's shared / routed row in place.
            auto* combined = flavor == 6U ? shared : flavor == 7U ? a.view.output[r] : reinterpret_cast<std::uint16_t*>(row + 1U + hidden + 2U * half);
            a.combine.shared[r] = shared; a.combine.combined[r] = combined;
            a.views.first[r] = shared; a.views.second[r] = a.view.output[r]; a.views.output[r] = combined;
        }
        a.views.count = rows;
        a.csr.header = base + row_words * rows;
        a.record = a.csr.header + header_words;
        return a;
    };
    const auto off = arm(device[0]), on = arm(device[1]);
    const dim3 grid((hidden + 255U) / 256U, rows);
    hipLaunchKernelGGL(known_moe_rows_commit_kernel, grid, dim3(256), 0, nullptr, off.view, off.csr, rows, hidden, off.record);
    check_hip(hipGetLastError(), "commit + combine self-test commit");
    moe_combine_known_window_testing_launch(off.views, hidden);   // no launch scope: the null stream, as the kernels above
    hipLaunchKernelGGL(known_moe_rows_commit_combine_kernel<true>, grid, dim3(256), 0, nullptr, on.view, on.csr, rows, hidden, on.record,
        on.combine);
    check_hip(hipGetLastError(), "commit + combine self-test launch");
    check_hip(hipDeviceSynchronize(), "commit + combine self-test sync");
    std::vector<std::uint32_t> x(words), y(words);
    check_hip(hipMemcpy(x.data(), device[0], words * 4U, hipMemcpyDeviceToHost), "read");
    check_hip(hipMemcpy(y.data(), device[1], words * 4U, hipMemcpyDeviceToHost), "read");
    for (auto* d : device) check_hip(hipFree(d), "free");
    double differing = 0;
    for (std::uint64_t i = 0; i < words; ++i) if (x[i] != y[i]) differing += 1.0;
    return {static_cast<double>(words), differing};
#else
    (void)rows; (void)flavor; (void)seed;
    throw std::invalid_argument("commit + combine self-test requires built A8 runtime");
#endif
}
} // namespace strix

namespace strix {
namespace {
void mtp_moe_range(MtpDeviceView v,std::uint64_t required,unsigned alignment) {
    if(!v.address || !v.allocation_generation || v.bytes<required || v.address%alignment ||
       v.bytes>std::numeric_limits<std::uintptr_t>::max()-v.address)
        throw std::invalid_argument("MTP routed device view is invalid");
}
bool mtp_moe_overlap(MtpDeviceView a,MtpDeviceView b){return a.address<b.address+b.bytes && b.address<a.address+a.bytes;}
void mtp_moe_prepare(const MtpNativeContext& c) {
    if(!c.model || !c.plan || !c.tensors || !c.resources)throw std::invalid_argument("MTP routed context is incomplete");
    c.resources->validate_stage();
    if(c.model!=&c.resources->model() || c.plan!=&c.resources->plan() || c.tensors!=&c.resources->tensors() ||
       c.position>=c.resources->context_capacity() || c.model->hidden_size!=2560U || c.model->moe_intermediate_size!=640U ||
       c.model->num_experts!=512U || c.model->num_experts_per_token!=10U)
        throw std::invalid_argument("MTP routed context is not bound to its actual canonical owner");
    const auto& owner=*c.resources;const auto& t=*c.tensors;
    const auto input=owner.view(HipMtpBuffer::hidden0),out=owner.view(HipMtpBuffer::hidden2),
        temp=owner.view(HipMtpBuffer::moe_workspace),routes=owner.view(HipMtpBuffer::routes),
        rw=owner.view(HipMtpBuffer::route_weights),status=owner.view(HipMtpBuffer::status);
    mtp_moe_range(input,5120U,2U);mtp_moe_range(out,5120U,2U);mtp_moe_range(temp,(1280U+2560U)*4U,4U);
    mtp_moe_range(routes,40U,4U);mtp_moe_range(rw,40U,4U);mtp_moe_range(status,4U,4U);
    std::array<MtpDeviceView,3> weights{};const std::array tensors{t.expert_gate,t.expert_up,t.expert_down};
    for(std::size_t i=0;i<tensors.size();++i){const auto pair=tensors[i];const bool down=i==2U;
        const std::vector<std::uint64_t> dims{down?640U:2560U,down?2560U:640U,512U};
        const std::uint64_t n=dims[0]/32U*34U*dims[1]*dims[2];
        if(!pair.manifest || !pair.runtime || pair.manifest->ggml_type!=8U || pair.manifest->dimensions!=dims ||
           pair.manifest->byte_size!=n || !runtime_tensor_gpu_visible(pair.runtime->storage))
            throw std::invalid_argument("MTP routed tensors must be separate Q8 gate/up/down with exact expert geometry");
        weights[i]={pair.runtime->address,pair.runtime->bytes,pair.runtime->allocation_generation};mtp_moe_range(weights[i],n,16U);
    }
    const std::array reads{input,routes,rw};const std::array writes{out,temp,status};
    for(std::size_t i=0;i<writes.size();++i){
        for(auto read:reads)if(mtp_moe_overlap(writes[i],read))throw std::invalid_argument("MTP routed writable view aliases input");
        for(auto w:weights)if(mtp_moe_overlap(writes[i],w))throw std::invalid_argument("MTP routed writable view aliases weights");
        for(std::size_t j=0;j<i;++j)if(mtp_moe_overlap(writes[i],writes[j]))throw std::invalid_argument("MTP routed writable views overlap");
    }
}
}
void preflight_hip_mtp_routed_experts(const MtpNativeContext& c){mtp_moe_prepare(c);}
void launch_hip_mtp_routed_experts(const MtpNativeContext& c,void* raw_stream) {
    mtp_moe_prepare(c);
    if(!raw_stream || raw_stream!=c.resources->stream())throw std::invalid_argument("MTP routed stream differs from retained owner");
    const auto stream=static_cast<hipStream_t>(raw_stream);hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
    check_hip(hipStreamIsCapturing(stream,&capture),"MTP routed capture query");
    if(capture!=hipStreamCaptureStatusNone)throw std::invalid_argument("MTP routed is eager only");
    check_hip(hipPeekAtLastError(),"MTP routed prior HIP error");
    const auto& o=*c.resources;const auto& t=*c.tensors;
    enqueue_hip_mtp_routed_experts_q8(reinterpret_cast<const std::uint16_t*>(o.view(HipMtpBuffer::hidden0).address),
        reinterpret_cast<const std::uint32_t*>(o.view(HipMtpBuffer::routes).address),
        reinterpret_cast<const float*>(o.view(HipMtpBuffer::route_weights).address),
        reinterpret_cast<const void*>(t.expert_gate.runtime->address),reinterpret_cast<const void*>(t.expert_up.runtime->address),
        reinterpret_cast<const void*>(t.expert_down.runtime->address),reinterpret_cast<float*>(o.view(HipMtpBuffer::moe_workspace).address),
        reinterpret_cast<std::uint32_t*>(o.view(HipMtpBuffer::status).address),
        reinterpret_cast<std::uint16_t*>(o.view(HipMtpBuffer::hidden2).address),stream);
}
void enqueue_hip_mtp_routed_experts_q8(const std::uint16_t* input,const std::uint32_t* routes,const float* rw,
    const void* gate,const void* up,const void* down,float* temp,std::uint32_t* status,std::uint16_t* output,void* raw_stream) {
    const auto stream=static_cast<hipStream_t>(raw_stream);auto* accumulation=temp+1280U;
    check_hip(hipMemsetAsync(accumulation,0,2560U*4U,stream),"MTP routed clear local accumulation");
    hipLaunchKernelGGL(validate_routes_kernel,dim3(1),dim3(kThreads),0,stream,routes,rw,10U,512U,status,nullptr);
    check_hip(hipGetLastError(),"MTP routed dynamic route validation");
    for(unsigned slot=0;slot<10U;++slot){
        hipLaunchKernelGGL(HIP_KERNEL_NAME(selected_projection_kernel<std::uint16_t,WeightKind::q8_0>),dim3(640U),dim3(kThreads),0,stream,
            input,routes,slot,gate,2560U,640U,512U,temp,status);
        check_hip(hipGetLastError(),"MTP routed Q8 gate");
        hipLaunchKernelGGL(HIP_KERNEL_NAME(selected_projection_kernel<std::uint16_t,WeightKind::q8_0>),dim3(640U),dim3(kThreads),0,stream,
            input,routes,slot,up,2560U,640U,512U,temp+640U,status);
        check_hip(hipGetLastError(),"MTP routed Q8 up");
        hipLaunchKernelGGL(routed_activation_kernel,dim3(3U),dim3(kThreads),0,stream,temp,640U,status);
        check_hip(hipGetLastError(),"MTP routed activation");
        hipLaunchKernelGGL(HIP_KERNEL_NAME(selected_down_accumulate_kernel<float,WeightKind::q8_0>),dim3(2560U),dim3(kThreads),0,stream,
            temp,routes,rw,slot,down,640U,2560U,512U,accumulation,status);
        check_hip(hipGetLastError(),"MTP routed ascending slot down");
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(commit_experts_kernel<std::uint16_t>),dim3(10U),dim3(kThreads),0,stream,
        accumulation,output,2560U,status);
    check_hip(hipGetLastError(),"MTP routed BF16 final commit");
}
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
void enqueue_q4k_a8_serial_tail_for_test(bool fused, const float* rawdots, const std::uint32_t* routes,
    const float* route_weights, std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k,
    float* accumulation, std::uint16_t* output, std::uint32_t* status, void* record, void* stream) {
    enqueue_serial_tail_for_test(fused, rawdots, routes, route_weights, hidden_size, experts, top_k, accumulation, output, status,
        record, static_cast<hipStream_t>(stream));
}
#endif
} // namespace strix
