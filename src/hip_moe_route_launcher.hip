#include "strix/env_flags.hpp"
#include "moe_core/route.hpp"
#include "strix/hip_qwen_dpp_reduce.hpp"
#include "strix/hip_mtp_native_validation.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_moe_route_launcher.hpp"
#include "strix/known_window_cohort.hpp"
#include "strix/q6p_registry.hpp"
#include "strix/wave_router_contract.hpp"

#include "strix/dispatch_resources.hpp"

#include <cstdio>
#include <cstdlib>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix {
namespace {

constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kMaxExperts = 512;
constexpr std::uint32_t kMaxTopK = 16;
constexpr std::uint32_t kWaveMaxTopK = 32;
constexpr std::uint32_t kRouterWave = 32;
constexpr std::uint32_t kWaveCandidates = kMaxExperts / kRouterWave;
static_assert(kWaveCandidates <= 32);
constexpr std::uint32_t kRouterMaxRows = 8;
struct RouterRowsArgs {
    const void* hidden[kRouterMaxRows];
    float* workspace[kRouterMaxRows];
    std::uint32_t* indices[kRouterMaxRows];
    float* weights[kRouterMaxRows];
    // Prefill tile strides (read only by RouterTileRowsArgs kernels).
    std::uint64_t tile_hidden, tile_workspace, tile_indices, tile_weights;
};
// Prefill tiles: blockIdx.y selects the 8-row tile, each view shifted by its own stride.
// Window kernels take RouterRowsArgs and compile without tile arithmetic.
struct RouterTileRowsArgs : RouterRowsArgs {};
struct RouterCohortRowsArgs {
    const void* hidden[32];float* workspace[32];
    std::uint32_t* indices[32];float* weights[32];std::uint32_t count=0;
    std::uint64_t tile_hidden=0,tile_workspace=0,tile_indices=0,tile_weights=0;
};
template<typename Pointer> __device__ __forceinline__ Pointer route_tiled(Pointer pointer,std::uint64_t,const RouterCohortRowsArgs&) {
    return pointer;
}
template <typename Pointer> __device__ __forceinline__ Pointer route_tiled(Pointer pointer, std::uint64_t, const RouterRowsArgs&) {
    return pointer;
}
template <typename Pointer> __device__ __forceinline__ Pointer route_tiled(Pointer pointer, std::uint64_t stride, const RouterTileRowsArgs&) {
    return reinterpret_cast<Pointer>(reinterpret_cast<std::uintptr_t>(pointer) + static_cast<std::uint64_t>(blockIdx.y) * stride);
}
#include "hip_router_fast_prefill.inc"
constexpr std::uint64_t kLauncherTag =
    0x4d4f45524f555432ULL;
constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

struct Q8Block {
    __half scale;
    std::int8_t values[32];
};

static_assert(sizeof(Q8Block) == 34);

enum class RouterWeightKind {
    f32,
    f16,
    bf16,
    q8_0,
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

const char* activation_name(
    HipActivationStorage storage) noexcept {
    switch (storage) {
    case HipActivationStorage::bf16:
        return "bf16";
    case HipActivationStorage::fp32:
        return "fp32";
    }
    return "unknown";
}

void validate_config(
    const HipMoeRouteLauncherConfig& config) {
    if (config.staged_logits && !config.parallel_projection)
        throw std::invalid_argument("staged router logits requires parallel projection");
    if (config.wave_topk)
        validate_wave_router_selection_contract(config.expert_count, config.top_k, config.parallel_projection, config.ranked_topk);
    if (config.ranked_topk && !config.parallel_projection)
        throw std::invalid_argument("ranked router top-k requires parallel projection");
    if (config.hidden_size == 0 ||
        config.expert_count == 0 ||
        config.expert_count > kMaxExperts ||
        config.top_k == 0 ||
        config.top_k > config.expert_count ||
        config.top_k > (config.wave_topk ? kWaveMaxTopK : kMaxTopK)) {
        throw std::invalid_argument(
            "HIP MoE route geometry is invalid");
    }
    if (config.activation !=
            HipActivationStorage::bf16 &&
        config.activation !=
            HipActivationStorage::fp32) {
        throw std::invalid_argument(
            "HIP MoE route activation storage is unsupported");
    }
}

RouterWeightKind weight_kind(
    std::uint32_t ggml_type) {
    switch (ggml_type) {
    case 0:
        return RouterWeightKind::f32;
    case 1:
        return RouterWeightKind::f16;
    case 8:
        return RouterWeightKind::q8_0;
    case 30:
        return RouterWeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP MoE route currently supports only F32, F16, BF16 and Q8_0 router weights");
    }
}

__device__ float bf16_to_float(
    std::uint16_t bits) {
    return __uint_as_float(
        static_cast<std::uint32_t>(
            bits) << 16U);
}

template <typename Activation>
__device__ float load_activation(
    const Activation* values,
    std::uint64_t index);

template <>
__device__ float load_activation<float>(
    const float* values,
    std::uint64_t index) {
    return values[index];
}

template <>
__device__ float load_activation<std::uint16_t>(
    const std::uint16_t* values,
    std::uint64_t index) {
    return bf16_to_float(
        values[index]);
}

template <RouterWeightKind Kind>
__device__ float load_weight(
    const void* weights,
    std::uint32_t hidden_size,
    std::uint32_t expert,
    std::uint32_t feature);

template <>
__device__ float load_weight<
    RouterWeightKind::f32>(
    const void* weights,
    std::uint32_t hidden_size,
    std::uint32_t expert,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const float*>(
            weights);
    return values[
        static_cast<std::uint64_t>(
            expert) *
            hidden_size +
        feature];
}

template <>
__device__ float load_weight<
    RouterWeightKind::f16>(
    const void* weights,
    std::uint32_t hidden_size,
    std::uint32_t expert,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const __half*>(
            weights);
    return __half2float(
        values[
            static_cast<std::uint64_t>(
                expert) *
                hidden_size +
            feature]);
}

template <>
__device__ float load_weight<
    RouterWeightKind::bf16>(
    const void* weights,
    std::uint32_t hidden_size,
    std::uint32_t expert,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const std::uint16_t*>(
            weights);
    return bf16_to_float(
        values[
            static_cast<std::uint64_t>(
                expert) *
                hidden_size +
            feature]);
}

template <>
__device__ float load_weight<
    RouterWeightKind::q8_0>(
    const void* weights,
    std::uint32_t hidden_size,
    std::uint32_t expert,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const Q8Block*>(
            weights);
    const auto blocks_per_row =
        hidden_size / 32U;
    const auto& block =
        blocks[
            static_cast<std::uint64_t>(
                expert) *
                blocks_per_row +
            feature / 32U];
    return __half2float(
               block.scale) *
           static_cast<float>(
               block.values[
                   feature % 32U]);
}

__device__ bool better(
    float left_value,
    std::uint32_t left_index,
    float right_value,
    std::uint32_t right_index) {
    return left_value > right_value ||
           (left_value == right_value &&
            left_index < right_index);
}

struct InsertRoutePolicy {
    static constexpr std::uint32_t max_top_k = kMaxTopK;
    __device__ static bool better(float lv, std::uint32_t li, float rv, std::uint32_t ri) {
        return strix::better(lv, li, rv, ri);
    }
};

struct WaveRoutePolicy : InsertRoutePolicy {
    static constexpr std::uint32_t candidates = kWaveCandidates, wave = kRouterWave;
    template <class T>
    __device__ static T xor32(T value, int mask) { return strix::qwen_dpp::xor32(value, mask); }
};

struct BlockWaveRoutePolicy : WaveRoutePolicy {
    std::uint32_t lane, expert_count;
    float* best_values;
    std::uint32_t* best_indices;
    __device__ void record(std::uint32_t slot, bool has, float best, std::uint32_t index) const {
        if (lane == 0) { best_values[slot] = has ? best : -INFINITY; best_indices[slot] = has ? index : expert_count; }
    }
};

struct LaneWaveRoutePolicy : WaveRoutePolicy {
    std::uint32_t lane, expert_count;
    float& mine;
    std::uint32_t& mine_index;
    __device__ void record(std::uint32_t slot, bool has, float best, std::uint32_t index) const {
        if (lane == slot) { mine = has ? best : -INFINITY; mine_index = has ? index : expert_count; }
    }
};

template <
    typename Activation,
    RouterWeightKind WeightKind>
__global__ void moe_route_kernel(
    const Activation* hidden,
    const void* weights,
    std::uint32_t hidden_size,
    std::uint32_t expert_count,
    std::uint32_t top_k,
    bool normalize_selected,
    std::uint32_t* selected_indices,
    float* selected_weights) {
    const auto row =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto lane =
        static_cast<std::uint32_t>(
            threadIdx.x);

    __shared__ float logits[
        kMaxExperts];
    __shared__ float scratch[
        kThreads];

    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        float sum = 0.0F;
        const auto hidden_base =
            static_cast<std::uint64_t>(
                row) *
            hidden_size;
        for (std::uint32_t feature = 0;
             feature < hidden_size;
             ++feature) {
            sum +=
                load_activation(
                    hidden,
                    hidden_base +
                        feature) *
                load_weight<WeightKind>(
                    weights,
                    hidden_size,
                    expert,
                    feature);
        }
        logits[expert] = sum;
    }
    __syncthreads();

    moe_core::route_softmax_insert<InsertRoutePolicy>(logits, scratch, row, lane,
        expert_count, top_k, normalize_selected, selected_indices, selected_weights);
}

constexpr std::uint32_t kRouterExpertsPerBlock = 8;
constexpr std::uint32_t kRouterFeaturesPerTile = 256;
constexpr std::uint32_t kRouterProjectionThreads = 128;
constexpr std::uint32_t kRouterCoefficientPitch = kRouterFeaturesPerTile + 4;
constexpr std::uint32_t kRouterProjectionLdsBytes =
    (kRouterExpertsPerBlock * kRouterCoefficientPitch + kRouterFeaturesPerTile) * sizeof(float);
constexpr std::uint32_t kRouterDeclaredLdsBytes =
    (kRouterProjectionLdsBytes + 255U) & ~255U;
static_assert(kRouterDeclaredLdsBytes == 9472U);

__device__ __forceinline__ float router_dot_ascending(
    const float* input, const float* coefficients, std::uint32_t count, float sum) {
    for (std::uint32_t feature = 0; feature < count; ++feature)
        sum = __builtin_fmaf(input[feature], coefficients[feature], sum);
    return sum;
}

template <typename Activation, RouterWeightKind WeightKind>
__global__ void parallel_moe_route_logits_kernel(
    const Activation* hidden, const void* weights, std::uint32_t hidden_size,
    std::uint32_t expert_count, float* workspace) {
    const auto row = static_cast<std::uint32_t>(blockIdx.y);
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    const auto expert_base = static_cast<std::uint32_t>(blockIdx.x) * kRouterExpertsPerBlock;
    __shared__ float tile_input[kRouterFeaturesPerTile];
    __shared__ float tile_weights[kRouterExpertsPerBlock][kRouterCoefficientPitch];
    float sum = 0.0F;
    const auto hidden_base = static_cast<std::uint64_t>(row) * hidden_size;
    for (std::uint32_t tile = 0; tile < hidden_size; tile += kRouterFeaturesPerTile) {
        const auto remaining = hidden_size - tile;
        const auto count = remaining < kRouterFeaturesPerTile ? remaining : kRouterFeaturesPerTile;
        // Separate global load and LDS publication lets independent coefficient
        // loads stay in flight. The inactive expert/tail lanes still synchronize.
        float coefficients[kRouterExpertsPerBlock * kRouterFeaturesPerTile / kRouterProjectionThreads];
        #pragma unroll
        for (std::uint32_t item = 0; item < kRouterExpertsPerBlock * kRouterFeaturesPerTile / kRouterProjectionThreads; ++item) {
            const auto flat = lane + item * kRouterProjectionThreads;
            const auto expert = expert_base + flat / kRouterFeaturesPerTile;
            const auto feature = flat % kRouterFeaturesPerTile;
            coefficients[item] = expert < expert_count && feature < count
                ? load_weight<WeightKind>(weights, hidden_size, expert, tile + feature) : 0.0F;
        }
        for (std::uint32_t feature = lane; feature < count; feature += kRouterProjectionThreads)
            tile_input[feature] = load_activation(hidden, hidden_base + tile + feature);
        #pragma unroll
        for (std::uint32_t item = 0; item < kRouterExpertsPerBlock * kRouterFeaturesPerTile / kRouterProjectionThreads; ++item) {
            const auto flat = lane + item * kRouterProjectionThreads;
            tile_weights[flat / kRouterFeaturesPerTile][flat % kRouterFeaturesPerTile] = coefficients[item];
        }
        __syncthreads();
        if (lane < kRouterExpertsPerBlock && expert_base + lane < expert_count)
            sum = router_dot_ascending(tile_input, tile_weights[lane], count, sum);
        __syncthreads();
    }
    if (lane < kRouterExpertsPerBlock && expert_base + lane < expert_count)
        workspace[static_cast<std::uint64_t>(row) * expert_count + expert_base + lane] = sum;
}

// Full-row staging changes data movement only. One lane in each of two
// waves retains the exact ascending fused accumulation chain used above.
template <typename Activation>
__global__ void staged_moe_route_logits_kernel(const Activation* hidden, const void* weights,
    float* workspace) {
    constexpr std::uint32_t width = 2560;
    constexpr std::uint32_t experts = 512;
    constexpr std::uint32_t threads = 256;
    const auto row = static_cast<std::uint32_t>(blockIdx.y);
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    const auto expert = static_cast<std::uint32_t>(blockIdx.x) * 2U;
    __shared__ float input[width];
    __shared__ float coefficients[2][width];
    for (std::uint32_t feature = lane; feature < width; feature += threads)
        input[feature] = load_activation(hidden, static_cast<std::uint64_t>(row) * width + feature);
    // Eligibility proves every row starts at a real 16-byte-aligned address.
    const auto* packed = reinterpret_cast<const float4*>(weights);
    float4 values[5];
    #pragma unroll
    for (std::uint32_t item = 0; item < 5; ++item)
        values[item] = packed[(static_cast<std::uint64_t>(expert) * width) / 4U + lane + item * threads];
    #pragma unroll
    for (std::uint32_t item = 0; item < 5; ++item) {
        const auto flat = lane + item * threads;
        const auto group = flat / (width / 4U);
        const auto feature = (flat % (width / 4U)) * 4U;
        coefficients[group][feature] = values[item].x;
        coefficients[group][feature + 1U] = values[item].y;
        coefficients[group][feature + 2U] = values[item].z;
        coefficients[group][feature + 3U] = values[item].w;
    }
    __syncthreads();
    if (lane == 0U || lane == 32U) {
        const auto group = lane / 32U;
        const float sum = router_dot_ascending(input, coefficients[group], width, 0.0F);
        workspace[static_cast<std::uint64_t>(row) * experts + expert + group] = sum;
    }
}

__device__ __forceinline__ void parallel_moe_route_select_row(
    const float* workspace, std::uint32_t expert_count, std::uint32_t top_k,
    bool normalize_selected, std::uint32_t* selected_indices, float* selected_weights) {
    constexpr std::uint32_t row = 0; // Callers pass this row's own buffers.
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    __shared__ float logits[kMaxExperts];
    __shared__ float scratch[kThreads];
    for (std::uint32_t expert = lane; expert < expert_count; expert += blockDim.x)
        logits[expert] = workspace[static_cast<std::uint64_t>(row) * expert_count + expert];
    __syncthreads();

    moe_core::route_softmax_insert<InsertRoutePolicy>(logits, scratch, row, lane,
        expert_count, top_k, normalize_selected, selected_indices, selected_weights);
}

__device__ __forceinline__ void ranked_parallel_moe_route_select_row(
    const float* workspace, std::uint32_t expert_count, std::uint32_t top_k,
    bool normalize_selected, std::uint32_t* selected_indices, float* selected_weights) {
    constexpr std::uint32_t row = 0; // Callers pass this row's own buffers.
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    __shared__ float logits[kMaxExperts];
    __shared__ float scratch[kThreads];
    __shared__ float best_values[kMaxTopK];
    __shared__ std::uint32_t best_indices[kMaxTopK];
    for (std::uint32_t expert = lane; expert < expert_count; expert += blockDim.x)
        logits[expert] = workspace[static_cast<std::uint64_t>(row) * expert_count + expert];
    __syncthreads();

    float local_max =
        -INFINITY;
    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        local_max =
            fmaxf(
                local_max,
                logits[expert]);
    }
    scratch[lane] = local_max;
    __syncthreads();

    for (std::uint32_t stride =
             blockDim.x / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] =
                fmaxf(
                    scratch[lane],
                    scratch[
                        lane + stride]);
        }
        __syncthreads();
    }
    const float maximum =
        scratch[0];
    // Every wave must retain the maximum before lane zero reuses scratch[0]
    // for its contribution to the denominator reduction.
    __syncthreads();

    float local_sum = 0.0F;
    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        const float probability =
            expf(
                logits[expert] -
                maximum);
        logits[expert] =
            probability;
        local_sum += probability;
    }
    scratch[lane] = local_sum;
    __syncthreads();

    for (std::uint32_t stride =
             blockDim.x / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] +=
                scratch[lane + stride];
        }
        __syncthreads();
    }
    const float denominator =
        scratch[0];

    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        logits[expert] /=
            denominator;
    }
    // Separate slots do not overwrite the denominator scratch still read by waves.
    if (lane < top_k) {
        best_values[lane] = -INFINITY;
        best_indices[lane] = expert_count;
    }
    __syncthreads();
    for (std::uint32_t expert = lane; expert < expert_count; expert += blockDim.x) {
        const float probability = logits[expert];
        // NaN compares better than nothing: explicitly exclude it before any write.
        if (probability == probability) {
            std::uint32_t rank = 0;
            for (std::uint32_t other = 0; other < expert_count; ++other) {
                if (better(logits[other], other, probability, expert)) ++rank;
                if (rank >= top_k) break;
            }
            // Non-NaN value/index pairs have a strict total order, so selected ranks
            // are unique. No barrier or early return exists inside this owner loop.
            if (rank < top_k) {
                best_values[rank] = probability;
                best_indices[rank] = expert;
            }
        }
    }
    __syncthreads();
    if (lane == 0) {
        float selected_sum = 0.0F;
        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            selected_sum +=
                best_values[slot];
        }
        if (!normalize_selected) {
            selected_sum = 1.0F;
        }

        const auto output_base =
            static_cast<std::uint64_t>(
                row) *
            top_k;
        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            selected_indices[
                output_base + slot] =
                best_indices[slot];
            selected_weights[
                output_base + slot] =
                best_values[slot] /
                selected_sum;
        }
    }
}

__device__ __forceinline__ void wave_parallel_moe_route_select_row(
    const float* workspace, std::uint32_t expert_count, std::uint32_t top_k,
    bool normalize_selected, std::uint32_t* selected_indices, float* selected_weights) {
    constexpr std::uint32_t row = 0; // Callers pass this row's own buffers.
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    __shared__ float logits[kMaxExperts];
    __shared__ float scratch[kThreads];
    __shared__ float best_values[kWaveMaxTopK];
    __shared__ std::uint32_t best_indices[kWaveMaxTopK];
    for (std::uint32_t expert = lane; expert < expert_count; expert += blockDim.x)
        logits[expert] = workspace[static_cast<std::uint64_t>(row) * expert_count + expert];
    __syncthreads();

    float local_max =
        -INFINITY;
    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        local_max =
            fmaxf(
                local_max,
                logits[expert]);
    }
    scratch[lane] = local_max;
    __syncthreads();

    for (std::uint32_t stride =
             blockDim.x / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] =
                fmaxf(
                    scratch[lane],
                    scratch[
                        lane + stride]);
        }
        __syncthreads();
    }
    const float maximum =
        scratch[0];
    // Every wave must retain the maximum before lane zero reuses scratch[0]
    // for its contribution to the denominator reduction.
    __syncthreads();

    float local_sum = 0.0F;
    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        const float probability =
            expf(
                logits[expert] -
                maximum);
        logits[expert] =
            probability;
        local_sum += probability;
    }
    scratch[lane] = local_sum;
    __syncthreads();

    for (std::uint32_t stride =
             blockDim.x / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] +=
                scratch[lane + stride];
        }
        __syncthreads();
    }
    const float denominator =
        scratch[0];

    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        logits[expert] /=
            denominator;
    }
    // Publish all divided probabilities before wave zero reads other waves' stores.
    __syncthreads();
    if (lane < kRouterWave) {
        float values[kWaveCandidates];
        std::uint32_t available = 0;
#pragma unroll
        for (std::uint32_t i = 0; i < kWaveCandidates; ++i) {
            const auto expert = lane + kRouterWave * i;
            values[i] = expert < expert_count ? logits[expert] : 0.0F;
            if (expert < expert_count && values[i] == values[i]) available |= 1U << i;
        }
        moe_core::route_wave_topk(values, available, expert_count, top_k, lane,
            BlockWaveRoutePolicy{{}, lane, expert_count, best_values, best_indices});
    }
    // All eight waves participate; no early return may bypass this barrier.
    __syncthreads();
    if (lane == 0) {
        float selected_sum = 0.0F;
        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            selected_sum +=
                best_values[slot];
        }
        if (!normalize_selected) {
            selected_sum = 1.0F;
        }

        const auto output_base =
            static_cast<std::uint64_t>(
                row) *
            top_k;
        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            selected_indices[
                output_base + slot] =
                best_indices[slot];
            selected_weights[
                output_base + slot] =
                best_values[slot] /
                selected_sum;
        }
    }
}

// Serial rows offset one buffer; window rows bring their own resolved
// buffers. Both forms run the same per-row selection body.
__global__ void parallel_moe_route_select_kernel(const float* workspace, std::uint32_t expert_count,
    std::uint32_t top_k, bool normalize_selected, std::uint32_t* selected_indices, float* selected_weights) {
    const auto row = static_cast<std::uint64_t>(blockIdx.x);
    parallel_moe_route_select_row(workspace + row * expert_count, expert_count, top_k, normalize_selected,
        selected_indices + row * top_k, selected_weights + row * top_k);
}
__global__ void ranked_parallel_moe_route_select_kernel(const float* workspace, std::uint32_t expert_count,
    std::uint32_t top_k, bool normalize_selected, std::uint32_t* selected_indices, float* selected_weights) {
    const auto row = static_cast<std::uint64_t>(blockIdx.x);
    ranked_parallel_moe_route_select_row(workspace + row * expert_count, expert_count, top_k, normalize_selected,
        selected_indices + row * top_k, selected_weights + row * top_k);
}
__global__ void wave_parallel_moe_route_select_kernel(const float* workspace, std::uint32_t expert_count,
    std::uint32_t top_k, bool normalize_selected, std::uint32_t* selected_indices, float* selected_weights) {
    const auto row = static_cast<std::uint64_t>(blockIdx.x);
    wave_parallel_moe_route_select_row(workspace + row * expert_count, expert_count, top_k, normalize_selected,
        selected_indices + row * top_k, selected_weights + row * top_k);
}
enum class RouterSelect { sorted, ranked, wave };
template <RouterSelect Mode, class Args = RouterRowsArgs>
__global__ void rows_moe_route_select_kernel(Args rows, std::uint32_t expert_count,
    std::uint32_t top_k, bool normalize_selected) {
    const auto r = blockIdx.x;
    if constexpr (Mode == RouterSelect::wave)
        wave_parallel_moe_route_select_row(route_tiled(rows.workspace[r], rows.tile_workspace, rows), expert_count, top_k, normalize_selected, route_tiled(rows.indices[r], rows.tile_indices, rows), route_tiled(rows.weights[r], rows.tile_weights, rows));
    else if constexpr (Mode == RouterSelect::ranked)
        ranked_parallel_moe_route_select_row(route_tiled(rows.workspace[r], rows.tile_workspace, rows), expert_count, top_k, normalize_selected, route_tiled(rows.indices[r], rows.tile_indices, rows), route_tiled(rows.weights[r], rows.tile_weights, rows));
    else
        parallel_moe_route_select_row(route_tiled(rows.workspace[r], rows.tile_workspace, rows), expert_count, top_k, normalize_selected, route_tiled(rows.indices[r], rows.tile_indices, rows), route_tiled(rows.weights[r], rows.tile_weights, rows));
}

// Prefill selection, one wave per row (STRIX_PREFILL_ROUTER_SELECT=1): bit-identical to
// rows_moe_route_select_kernel<wave> (one 256-thread block per row). Lane l holds experts l + 32 i in
// registers. The maximum is order-free (fmaxf over the non-NaN logits, -INFINITY start; its zero sign
// cannot change exp(x - maximum)). The denominator replays the block's sum exactly: thread t = l + 32 j
// held (0 + p[t]) + p[t + 256], then the tree strides 128 / 64 / 32 (registers j + 4, j + 2, j + 1) and
// 16..1 (shuffle down; lane 0 ends with scratch[0]). Same division, the same wave top-k (better(),
// NaN excluded), the same serial selected_sum in slot order. 8 rows (one tile) per 256-thread block.
__device__ __forceinline__ void wave_route_select_row(const float* workspace, std::uint32_t* indices, float* weights,
    std::uint32_t expert_count, std::uint32_t top_k, bool normalize_selected, std::uint32_t lane) {
    float p[kWaveCandidates];
#pragma unroll
    for (std::uint32_t i = 0; i < kWaveCandidates; ++i) {
        const auto expert = lane + kRouterWave * i;
        p[i] = expert < expert_count ? workspace[expert] : 0.0F;
    }
    float maximum = -INFINITY;
#pragma unroll
    for (std::uint32_t i = 0; i < kWaveCandidates; ++i)
        if (lane + kRouterWave * i < expert_count) maximum = fmaxf(maximum, p[i]);
#pragma unroll
    for (std::uint32_t mask = kRouterWave / 2U; mask > 0; mask >>= 1U)
        maximum = fmaxf(maximum, strix::qwen_dpp::xor32(maximum, static_cast<int>(mask)));
#pragma unroll
    for (std::uint32_t i = 0; i < kWaveCandidates; ++i)
        if (lane + kRouterWave * i < expert_count) p[i] = expf(p[i] - maximum);
    constexpr std::uint32_t half = kThreads / kRouterWave;   // 8 partials per lane: threads l + 32 j
    float partial[half];
#pragma unroll
    for (std::uint32_t j = 0; j < half; ++j) {
        float sum = 0.0F;
        if (lane + kRouterWave * j < expert_count) sum += p[j];
        if (lane + kRouterWave * (j + half) < expert_count) sum += p[j + half];
        partial[j] = sum;
    }
#pragma unroll
    for (std::uint32_t j = 0; j < 4U; ++j) partial[j] += partial[j + 4U];   // stride 128
#pragma unroll
    for (std::uint32_t j = 0; j < 2U; ++j) partial[j] += partial[j + 2U];   // stride 64
    partial[0] += partial[1];                                                 // stride 32
    float total = partial[0];
#pragma unroll
    for (std::uint32_t stride = kRouterWave / 2U; stride > 0; stride >>= 1U)
        total += __shfl_down(total, stride, kRouterWave);
    const float denominator = __shfl(total, 0, kRouterWave);
    float values[kWaveCandidates];
    std::uint32_t available = 0;
#pragma unroll
    for (std::uint32_t i = 0; i < kWaveCandidates; ++i) {
        const auto expert = lane + kRouterWave * i;
        values[i] = expert < expert_count ? p[i] / denominator : 0.0F;
        if (expert < expert_count && values[i] == values[i]) available |= 1U << i;
    }
    float mine = -INFINITY;
    std::uint32_t mine_index = expert_count;
    moe_core::route_wave_topk(values, available, expert_count, top_k, lane,
        LaneWaveRoutePolicy{{}, lane, expert_count, mine, mine_index});
    float selected_sum = 0.0F;
    for (std::uint32_t slot = 0; slot < top_k; ++slot) selected_sum += __shfl(mine, static_cast<int>(slot), kRouterWave);
    if (!normalize_selected) selected_sum = 1.0F;
    if (lane < top_k) {
        indices[lane] = mine_index;
        weights[lane] = mine / selected_sum;
    }
}
template <class Args>
__global__ void __launch_bounds__(256) wave_rows_route_select_kernel(Args rows, std::uint32_t expert_count,
    std::uint32_t top_k, bool normalize_selected, std::uint32_t row_count) {
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) % kRouterWave;
    // Row = wave of the block, plus blockDim/32 rows per block in x (the window form launches one 32-thread block per row,
    // spread over the CUs: 8 rows in one block were 12.8 vs 11.4 us in the engine).
    const auto r = static_cast<std::uint32_t>(blockIdx.x) * (blockDim.x / kRouterWave) + static_cast<std::uint32_t>(threadIdx.x) / kRouterWave;
    if (r >= row_count) return;   // waves without a row; no barrier follows
    wave_route_select_row(route_tiled(rows.workspace[r], rows.tile_workspace, rows), route_tiled(rows.indices[r], rows.tile_indices, rows),
        route_tiled(rows.weights[r], rows.tile_weights, rows), expert_count, top_k, normalize_selected, lane);
}
// STRIX_PREFILL_ROUTER_FUSE=1: the logits kernel's tail. Every block of a 128-row tile releases its stores and counts in;
// the last of the gridDim.y blocks (acquire) runs wave_route_select_row for the tile's rows (8 waves, 16 rows each) from
// the just-written logits and resets the tile counter. Same logits, same selection code: same bits as logits + select.
struct RouterSelectTail {
    unsigned* counters; std::uint32_t expert_count, top_k; bool normalize;
    __device__ void operator()(const RouterTileRowsArgs& rows, unsigned m0, unsigned total_rows) const {
        __shared__ unsigned last;
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0U) {
            const unsigned seen = atomicAdd(&counters[blockIdx.x], 1U);
            last = seen + 1U == gridDim.y ? 1U : 0U;
            if (last) atomicExch(&counters[blockIdx.x], 0U);
        }
        __syncthreads();
        if (!last) return;
        __threadfence();
        const unsigned lane = threadIdx.x % kRouterWave, wave = threadIdx.x / kRouterWave;
        for (unsigned r = wave; r < router_fast::kM; r += blockDim.x / kRouterWave) {
            const unsigned g = m0 + r;
            if (g >= total_rows) break;
            const auto at = [&](auto* base, std::uint64_t stride) {
                return reinterpret_cast<decltype(base)>(reinterpret_cast<std::uintptr_t>(base) + static_cast<std::uint64_t>(g / 8U) * stride);
            };
            wave_route_select_row(at(rows.workspace[g % 8U], rows.tile_workspace), at(rows.indices[g % 8U], rows.tile_indices),
                at(rows.weights[g % 8U], rows.tile_weights), expert_count, top_k, normalize, lane);
        }
    }
};

// Window rows share each staged weight tile; lane (expert, row) keeps the
// serial chain: fused, ascending from feature zero, carried across tiles.
constexpr std::uint32_t kRowsExpertsPerBlock = 2;
constexpr std::uint32_t kRowsTile = 512;
constexpr std::uint32_t kRowsPitch = kRowsTile + 4;
// Same fused ascending chain as router_dot_ascending; vector LDS reads are
// issued ahead of the dependent FMAs (the chain is LDS-latency bound).
__device__ __forceinline__ float router_dot_tile(const float* input, const float* coefficients, float sum) {
    const auto* x = reinterpret_cast<const float4*>(input);
    const auto* c = reinterpret_cast<const float4*>(coefficients);
    #pragma unroll
    for (std::uint32_t q = 0; q < kRowsTile / 4U; ++q) {
        const auto a = x[q], b = c[q];
        sum = __builtin_fmaf(a.x, b.x, sum); sum = __builtin_fmaf(a.y, b.y, sum);
        sum = __builtin_fmaf(a.z, b.z, sum); sum = __builtin_fmaf(a.w, b.w, sum);
    }
    return sum;
}
// F32 weights: one wave per expert, lane = row. The expert's coefficients are
// wave-uniform scalar loads and each lane streams its own row, so the chain
// never waits on LDS. Per (row, expert) it is the serial chain: fused,
// ascending from feature zero.
constexpr std::uint32_t kWaveRowsChunk = 64;
template <typename Activation, std::uint32_t Rows,class Args=RouterRowsArgs>
__global__ void __launch_bounds__(256) wave_rows_moe_route_logits_kernel(Args rows,
    const float* __restrict__ weights, std::uint32_t hidden_size, std::uint32_t expert_count) {
    constexpr std::uint32_t per_vector = 16U / sizeof(Activation);
    const auto wave = static_cast<std::uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / 32U)));
    const auto lane = static_cast<std::uint32_t>(threadIdx.x % 32U);
    const auto expert = static_cast<std::uint32_t>(blockIdx.x) * 8U + wave;
    if (expert >= expert_count) return;
    const float* __restrict__ coefficients = weights + static_cast<std::uint64_t>(expert) * hidden_size;
    const auto active=[&] {if constexpr(Rows==32U) return rows.count;else return Rows;}();
    const auto* hidden = static_cast<const Activation*>(rows.hidden[lane < active ? lane : 0U]);
    float sum = 0.0F;
    for (std::uint32_t feature = 0; feature < hidden_size; feature += kWaveRowsChunk) {
        float input[kWaveRowsChunk];
        #pragma unroll
        for (std::uint32_t v = 0; v < kWaveRowsChunk / per_vector; ++v) {
            const auto packed = *reinterpret_cast<const uint4*>(hidden + feature + v * per_vector);
            const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
            #pragma unroll
            for (std::uint32_t k = 0; k < per_vector; ++k) {
                if constexpr (sizeof(Activation) == 2U)
                    input[v * per_vector + k] = bf16_to_float(static_cast<std::uint16_t>(k % 2U ? words[k / 2U] >> 16U : words[k / 2U] & 0xffffU));
                else input[v * per_vector + k] = __uint_as_float(words[k]);
            }
        }
        #pragma unroll
        for (std::uint32_t k = 0; k < kWaveRowsChunk; ++k)
            sum = __builtin_fmaf(input[k], coefficients[feature + k], sum);
    }
    if (lane < active) rows.workspace[lane][expert] = sum;
}
// F32 weights, every lane busy: a wave covers 32 / Rows experts x Rows rows and
// lane (expert, row) runs that pair's serial chain (fused, ascending from
// feature zero) on its own coefficients. The one-expert-per-wave form above
// left 32 - Rows lanes idle and needed 8x more waves for the same chains.
// Chunk = coefficients each lane has in flight per step: the 2048 chains of a 4-row window
// are only 64 waves, so the stream is latency bound and a longer step overlaps more of it.
// Bf16: weights is the BF16 copy of F32 router weights that are exact BF16 values (a
// registered strix::q6p copy); the expanded coefficients and the chain are unchanged.
template <typename Activation, std::uint32_t Rows, std::uint32_t Chunk = kWaveRowsChunk, class Args = RouterRowsArgs, bool Bf16 = false>
__global__ void __launch_bounds__(256) lanes_rows_moe_route_logits_kernel(Args rows,
    const void* __restrict__ weights, std::uint32_t hidden_size, std::uint32_t expert_count) {
    static_assert(Rows == 2U || Rows == 4U || Rows == 8U);
    constexpr std::uint32_t per_vector = 16U / sizeof(Activation);
    constexpr std::uint32_t experts_per_wave = 32U / Rows;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x % 32U);
    const auto wave = static_cast<std::uint32_t>(blockIdx.x) * 8U + static_cast<std::uint32_t>(threadIdx.x / 32U);
    const auto expert = wave * experts_per_wave + lane / Rows;
    const auto row = lane % Rows;
    if (expert >= expert_count) return;
    const auto* __restrict__ coefficients = reinterpret_cast<const float4*>(static_cast<const float*>(weights) +
        static_cast<std::uint64_t>(expert) * hidden_size);
    const auto* __restrict__ halves = reinterpret_cast<const uint4*>(static_cast<const std::uint16_t*>(weights) +
        static_cast<std::uint64_t>(expert) * hidden_size);
    const auto* hidden = static_cast<const Activation*>(route_tiled(rows.hidden[row], rows.tile_hidden, rows));
    float sum = 0.0F;
    for (std::uint32_t feature = 0; feature < hidden_size; feature += Chunk) {
        float4 coefficient[Chunk / 4U];
        if constexpr (Bf16) {
            uint4 packed[Chunk / 8U];
            #pragma unroll
            for (std::uint32_t q = 0; q < Chunk / 8U; ++q) packed[q] = halves[(feature >> 3U) + q];
            #pragma unroll
            for (std::uint32_t q = 0; q < Chunk / 8U; ++q) {
                const std::uint32_t w[4] = {packed[q].x, packed[q].y, packed[q].z, packed[q].w};
                coefficient[2U * q] = make_float4(__uint_as_float(w[0] << 16U), __uint_as_float(w[0] & 0xffff0000U),
                    __uint_as_float(w[1] << 16U), __uint_as_float(w[1] & 0xffff0000U));
                coefficient[2U * q + 1U] = make_float4(__uint_as_float(w[2] << 16U), __uint_as_float(w[2] & 0xffff0000U),
                    __uint_as_float(w[3] << 16U), __uint_as_float(w[3] & 0xffff0000U));
            }
        } else {
            #pragma unroll
            for (std::uint32_t q = 0; q < Chunk / 4U; ++q) coefficient[q] = coefficients[(feature >> 2U) + q];
        }
        float input[Chunk];
        #pragma unroll
        for (std::uint32_t v = 0; v < Chunk / per_vector; ++v) {
            const auto packed = *reinterpret_cast<const uint4*>(hidden + feature + v * per_vector);
            const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
            #pragma unroll
            for (std::uint32_t k = 0; k < per_vector; ++k) {
                if constexpr (sizeof(Activation) == 2U)
                    input[v * per_vector + k] = bf16_to_float(static_cast<std::uint16_t>(k % 2U ? words[k / 2U] >> 16U : words[k / 2U] & 0xffffU));
                else input[v * per_vector + k] = __uint_as_float(words[k]);
            }
        }
        #pragma unroll
        for (std::uint32_t q = 0; q < Chunk / 4U; ++q) {
            const auto c = coefficient[q];
            sum = __builtin_fmaf(input[4U * q + 0U], c.x, sum);
            sum = __builtin_fmaf(input[4U * q + 1U], c.y, sum);
            sum = __builtin_fmaf(input[4U * q + 2U], c.z, sum);
            sum = __builtin_fmaf(input[4U * q + 3U], c.w, sum);
        }
    }
    route_tiled(rows.workspace[row], rows.tile_workspace, rows)[expert] = sum;
}
// Staged form of lanes_rows_moe_route_logits_kernel (Chunk 128) for Steps x 128 features: the first chain
// waves of a block run the same lane (expert, row) chains for its Experts experts (fused,
// ascending from feature zero, the same float4 order; for one row also the order of router_dot_ascending in
// parallel_moe_route_logits_kernel), but read the coefficients from LDS; the Loaders waves after them stream the
// block's coefficients chunk by chunk, depth chunks ahead in registers, into a two-slot LDS ring. The lanes form
// keeps only one chunk per chain in flight, 512 x 512 B for the whole router, so every chunk waited a
// DRAM round trip; here 64 KB per block (8 chunks of its 16 experts) are in flight.
// The ring barrier orders LDS only: __syncthreads() would also wait for the loaders' outstanding loads.
// Bf16: weights is the registered BF16 copy of BF16-exact F32 weights; the loaders read 4 halves per float4
// and widen them before the LDS store, so the chains read the same coefficients.
__device__ __forceinline__ void router_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
template <std::uint32_t Rows, std::uint32_t Experts = 16U> constexpr std::uint32_t staged_router_chain_waves() {
    return (Rows * Experts + 31U) / 32U;
}
// StageAct (STRIX_ROUTER_STAGE_ACT=1): the block first copies its rows' activations into LDS (one barrier), so the chains
// read each chunk's activations from LDS instead of one L2/DRAM round trip per chunk (the chunk loop was paced by that
// round trip: ~1.9 us x 20 chunks at one row). Same values, same FMA order: bit-identical logits.
template <typename Activation, std::uint32_t Rows, std::uint32_t Steps, std::uint32_t Loaders, bool Bf16 = false, std::uint32_t Experts = 16U,
    bool StageAct = false, std::uint32_t Chunk = 128U>
__global__ void __launch_bounds__(32U * (staged_router_chain_waves<Rows, Experts>() + Loaders)) staged_rows_moe_route_logits_kernel(
    RouterRowsArgs rows, const void* __restrict__ weights, std::uint32_t expert_count) {
    static_assert(Rows == 1U || Rows == 2U || Rows == 4U || Rows == 8U);
    // Chunk 256 (STRIX_ROUTER_CHUNK256=1, needs StageAct): half the staging rendezvous; the chains stream their activations
    // from the LDS copy per float4 (no register pack arrays). Same coefficients, same ascending fused chain.
    static_assert(Chunk == 128U || (Chunk == 256U && StageAct));
    constexpr std::uint32_t chunk = Chunk, hidden_size = Steps * chunk, experts = Experts, vectors = chunk / 4U;
    constexpr std::uint32_t chains = staged_router_chain_waves<Rows, Experts>();
    // 16 float4 of ring per loader thread (a deeper ring, 32, measured slower).
    constexpr std::uint32_t per_loader = experts * vectors / (32U * Loaders), depth = 16U / per_loader, pitch = vectors + 1U;
    static_assert(per_loader * 32U * Loaders == experts * vectors && depth * per_loader == 16U);
    constexpr std::uint32_t per_vector = 16U / sizeof(Activation);
    __shared__ float4 stage[2][experts * pitch];
    constexpr std::uint32_t packs_per_row = hidden_size / per_vector;
    static_assert(!StageAct || Rows * packs_per_row * 16U <= 48U * 1024U, "staged activations exceed the LDS budget");
    __shared__ uint4 act[StageAct ? Rows : 1U][StageAct ? packs_per_row : 1U];
    if constexpr (StageAct) {
        for (std::uint32_t i = threadIdx.x; i < Rows * packs_per_row; i += blockDim.x)
            act[i / packs_per_row][i % packs_per_row] = reinterpret_cast<const uint4*>(rows.hidden[i / packs_per_row])[i % packs_per_row];
        __syncthreads();
    }
    const auto wave = threadIdx.x / 32U, lane = threadIdx.x % 32U;
    const auto first_expert = static_cast<std::uint32_t>(blockIdx.x) * experts;
    if (wave < chains) {
        // One row: lanes 16-31 run a copy of lane 15's chain and store nothing.
        const auto chain = (wave * 32U + lane) / Rows, local = chain < experts ? chain : experts - 1U;
        const auto row = lane % Rows, expert = first_expert + local;
        const auto* hidden = static_cast<const Activation*>(rows.hidden[row]);
        float sum = 0.0F;
        if constexpr (Chunk != 128U) {
            (void)hidden;
            constexpr std::uint32_t packs = chunk / per_vector;
#pragma unroll 1
            for (std::uint32_t step = 0; step < Steps; ++step) {
                router_lds_barrier();
                const float4* coefficients = stage[step % 2U] + local * pitch;
                const uint4* packed_row = act[row] + step * packs;
#pragma unroll
                for (std::uint32_t q = 0; q < vectors; ++q) {
                    const auto c = coefficients[q];
                    const uint4 packed = packed_row[(4U * q) / per_vector];
                    const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
                    const auto value = [&](std::uint32_t k) {
                        const auto i = k % per_vector;
                        if constexpr (sizeof(Activation) == 2U)
                            return bf16_to_float(static_cast<std::uint16_t>(i % 2U ? words[i / 2U] >> 16U : words[i / 2U] & 0xffffU));
                        else return __uint_as_float(words[i]);
                    };
                    sum = __builtin_fmaf(value(4U * q + 0U), c.x, sum);
                    sum = __builtin_fmaf(value(4U * q + 1U), c.y, sum);
                    sum = __builtin_fmaf(value(4U * q + 2U), c.z, sum);
                    sum = __builtin_fmaf(value(4U * q + 3U), c.w, sum);
                }
            }
        } else {
        // The next chunk's activations are requested before this chunk's FMAs (L2 round trip per chunk).
        constexpr std::uint32_t packs = chunk / per_vector;
        uint4 next[packs];
#pragma unroll
        for (std::uint32_t v = 0; v < packs; ++v)
            next[v] = StageAct ? act[StageAct ? row : 0U][v] : *reinterpret_cast<const uint4*>(hidden + v * per_vector);
#pragma unroll 1
        for (std::uint32_t step = 0; step < Steps; ++step) {
            uint4 raw[packs];
#pragma unroll
            for (std::uint32_t v = 0; v < packs; ++v) raw[v] = next[v];
            if (step + 1U < Steps)
#pragma unroll
                for (std::uint32_t v = 0; v < packs; ++v)
                    next[v] = StageAct ? act[StageAct ? row : 0U][(step + 1U) * packs + v]
                                       : *reinterpret_cast<const uint4*>(hidden + (step + 1U) * chunk + v * per_vector);
            router_lds_barrier();
            const float4* coefficients = stage[step % 2U] + local * pitch;
            // Activations decoded at their FMA (no 128-float staging array: register budget), same values and order.
            const auto activation = [&](std::uint32_t k) {
                const auto& packed = raw[k / per_vector];
                const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
                const auto i = k % per_vector;
                if constexpr (sizeof(Activation) == 2U)
                    return bf16_to_float(static_cast<std::uint16_t>(i % 2U ? words[i / 2U] >> 16U : words[i / 2U] & 0xffffU));
                else return __uint_as_float(words[i]);
            };
#pragma unroll
            for (std::uint32_t q = 0; q < vectors; ++q) {
                const auto c = coefficients[q];
                sum = __builtin_fmaf(activation(4U * q + 0U), c.x, sum);
                sum = __builtin_fmaf(activation(4U * q + 1U), c.y, sum);
                sum = __builtin_fmaf(activation(4U * q + 2U), c.z, sum);
                sum = __builtin_fmaf(activation(4U * q + 3U), c.w, sum);
            }
        }
        }
        if (chain < experts && expert < expert_count) rows.workspace[row][expert] = sum;
    } else {
        // Loader thread t copies vectors t + 32 Loaders p (p < per_loader) of every chunk: expert j / vectors, vector j % vectors.
        const auto t = threadIdx.x - 32U * chains;
        using Vector = std::conditional_t<Bf16, uint2, float4>;
        const Vector* source[per_loader];
        std::uint32_t slot[per_loader];
#pragma unroll
        for (std::uint32_t p = 0; p < per_loader; ++p) {
            const auto j = t + 32U * Loaders * p, e = j / vectors, q = j % vectors;
            const auto expert = first_expert + e < expert_count ? first_expert + e : expert_count - 1U;
            if constexpr (Bf16)
                source[p] = reinterpret_cast<const uint2*>(static_cast<const std::uint16_t*>(weights) + static_cast<std::uint64_t>(expert) * hidden_size) + q;
            else source[p] = reinterpret_cast<const float4*>(static_cast<const float*>(weights) + static_cast<std::uint64_t>(expert) * hidden_size) + q;
            slot[p] = e * pitch + q;
        }
        const auto widen = [](const Vector& v) {
            if constexpr (Bf16)
                return make_float4(__uint_as_float(v.x << 16U), __uint_as_float(v.x & 0xffff0000U), __uint_as_float(v.y << 16U),
                    __uint_as_float(v.y & 0xffff0000U));
            else return v;
        };
        Vector ring[depth][per_loader];
#pragma unroll
        for (std::uint32_t d = 0; d < depth; ++d)
#pragma unroll
            for (std::uint32_t p = 0; p < per_loader; ++p) if (d < Steps) ring[d][p] = source[p][d * vectors];
        // The ring shifts by one chunk per step (constant indices: an indexed ring would live in scratch).
        for (std::uint32_t step = 0; step < Steps; ++step) {
#pragma unroll
            for (std::uint32_t p = 0; p < per_loader; ++p) stage[step % 2U][slot[p]] = widen(ring[0][p]);
#pragma unroll
            for (std::uint32_t d = 1; d < depth; ++d)
#pragma unroll
                for (std::uint32_t p = 0; p < per_loader; ++p) ring[d - 1U][p] = ring[d][p];
            if (step + depth < Steps)
#pragma unroll
                for (std::uint32_t p = 0; p < per_loader; ++p) ring[depth - 1U][p] = source[p][(step + depth) * vectors];
            router_lds_barrier();
        }
    }
}
// Prefill tiles: lanes_rows_moe_route_logits_kernel's lane (expert, row) chains for Tiles 8-row tiles
// at once (blockIdx.y covers Tiles tiles). Each coefficient chunk is loaded once and used by the
// lane's Tiles chains, each chain the same fused ascending FMA sequence over its own row.
template <typename Activation, std::uint32_t Tiles, std::uint32_t Chunk>
__global__ void __launch_bounds__(256) lanes_tiles_moe_route_logits_kernel(RouterTileRowsArgs rows,
    const float* __restrict__ weights, std::uint32_t hidden_size, std::uint32_t expert_count, std::uint32_t tiles) {
    constexpr std::uint32_t Rows = 8U, per_vector = 16U / sizeof(Activation), experts_per_wave = 32U / Rows;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x % 32U);
    const auto wave = static_cast<std::uint32_t>(blockIdx.x) * 8U + static_cast<std::uint32_t>(threadIdx.x / 32U);
    const auto expert = wave * experts_per_wave + lane / Rows;
    const auto row = lane % Rows;
    if (expert >= expert_count) return;
    const auto first_tile = static_cast<std::uint32_t>(blockIdx.y) * Tiles;
    const auto* __restrict__ coefficients = reinterpret_cast<const float4*>(weights + static_cast<std::uint64_t>(expert) * hidden_size);
    const Activation* hidden[Tiles];
    float sum[Tiles];
#pragma unroll
    for (std::uint32_t t = 0; t < Tiles; ++t) {
        const auto tile = first_tile + t < tiles ? first_tile + t : first_tile;
        hidden[t] = reinterpret_cast<const Activation*>(reinterpret_cast<std::uintptr_t>(rows.hidden[row]) +
            static_cast<std::uint64_t>(tile) * rows.tile_hidden);
        sum[t] = 0.0F;
    }
    for (std::uint32_t feature = 0; feature < hidden_size; feature += Chunk) {
        float4 coefficient[Chunk / 4U];
#pragma unroll
        for (std::uint32_t q = 0; q < Chunk / 4U; ++q) coefficient[q] = coefficients[(feature >> 2U) + q];
#pragma unroll
        for (std::uint32_t t = 0; t < Tiles; ++t) {
            float input[Chunk];
#pragma unroll
            for (std::uint32_t v = 0; v < Chunk / per_vector; ++v) {
                const auto packed = *reinterpret_cast<const uint4*>(hidden[t] + feature + v * per_vector);
                const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
                for (std::uint32_t k = 0; k < per_vector; ++k) {
                    if constexpr (sizeof(Activation) == 2U)
                        input[v * per_vector + k] = bf16_to_float(static_cast<std::uint16_t>(k % 2U ? words[k / 2U] >> 16U : words[k / 2U] & 0xffffU));
                    else input[v * per_vector + k] = __uint_as_float(words[k]);
                }
            }
#pragma unroll
            for (std::uint32_t q = 0; q < Chunk / 4U; ++q) {
                const auto c = coefficient[q];
                sum[t] = __builtin_fmaf(input[4U * q + 0U], c.x, sum[t]);
                sum[t] = __builtin_fmaf(input[4U * q + 1U], c.y, sum[t]);
                sum[t] = __builtin_fmaf(input[4U * q + 2U], c.z, sum[t]);
                sum[t] = __builtin_fmaf(input[4U * q + 3U], c.w, sum[t]);
            }
        }
    }
#pragma unroll
    for (std::uint32_t t = 0; t < Tiles; ++t)
        if (first_tile + t < tiles)
            reinterpret_cast<float*>(reinterpret_cast<std::uintptr_t>(rows.workspace[row]) +
                static_cast<std::uint64_t>(first_tile + t) * rows.tile_workspace)[expert] = sum[t];
}
template <typename Activation, RouterWeightKind WeightKind, std::uint32_t Rows>
__global__ void __launch_bounds__(256) rows_moe_route_logits_kernel(RouterRowsArgs rows, const void* weights,
    std::uint32_t hidden_size, std::uint32_t expert_count) {
    constexpr std::uint32_t threads = 256;
    constexpr std::uint32_t weight_items = kRowsExpertsPerBlock * kRowsTile / threads;
    constexpr std::uint32_t input_items = Rows * kRowsTile / threads;
    static_assert(kRowsTile % threads == 0);
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    const auto expert_base = static_cast<std::uint32_t>(blockIdx.x) * kRowsExpertsPerBlock;
    __shared__ __align__(16) float coefficients[kRowsExpertsPerBlock][kRowsTile];
    __shared__ __align__(16) float input[Rows][kRowsPitch];
    float w[weight_items], x[input_items];
    const Activation* hidden[Rows];
    #pragma unroll
    for (std::uint32_t r = 0; r < Rows; ++r) hidden[r] = static_cast<const Activation*>(rows.hidden[r]);
    const auto fetch = [&](std::uint32_t tile) {
        const auto count = hidden_size - tile < kRowsTile ? hidden_size - tile : kRowsTile;
        #pragma unroll
        for (std::uint32_t item = 0; item < weight_items; ++item) {
            const auto flat = lane + item * threads;
            const auto expert = expert_base + flat / kRowsTile;
            const auto feature = flat % kRowsTile;
            w[item] = expert < expert_count && feature < count
                ? load_weight<WeightKind>(weights, hidden_size, expert, tile + feature) : 0.0F;
        }
        // Row and feature are compile-time per item: no per-load pointer fetch.
        #pragma unroll
        for (std::uint32_t item = 0; item < input_items; ++item) {
            const auto feature = lane + (item % (kRowsTile / threads)) * threads;
            x[item] = feature < count ? load_activation(hidden[item / (kRowsTile / threads)],
                static_cast<std::uint64_t>(tile) + feature) : 0.0F;
        }
    };
    const auto chain_expert = lane / 32U, chain_row = lane % 32U;
    const bool chain = chain_expert < kRowsExpertsPerBlock && chain_row < Rows &&
        expert_base + chain_expert < expert_count;
    float sum = 0.0F;
    fetch(0);
    for (std::uint32_t tile = 0; tile < hidden_size; tile += kRowsTile) {
        const auto count = hidden_size - tile < kRowsTile ? hidden_size - tile : kRowsTile;
        __syncthreads();
        #pragma unroll
        for (std::uint32_t item = 0; item < weight_items; ++item) {
            const auto flat = lane + item * threads;
            coefficients[flat / kRowsTile][flat % kRowsTile] = w[item];
        }
        #pragma unroll
        for (std::uint32_t item = 0; item < input_items; ++item)
            input[item / (kRowsTile / threads)][lane + (item % (kRowsTile / threads)) * threads] = x[item];
        __syncthreads();
        if (tile + kRowsTile < hidden_size) fetch(tile + kRowsTile);
        if (chain) sum = count == kRowsTile ? router_dot_tile(input[chain_row], coefficients[chain_expert], sum) :
            router_dot_ascending(input[chain_row], coefficients[chain_expert], count, sum);
    }
    if (chain) rows.workspace[chain_row][expert_base + chain_expert] = sum;
}

std::uint64_t required_router_weight_bytes(
    RouterWeightKind kind,
    std::uint32_t hidden_size,
    std::uint32_t expert_count) {
    switch (kind) {
    case RouterWeightKind::f32:
        return
            static_cast<std::uint64_t>(
                hidden_size) *
            expert_count *
            4ULL;
    case RouterWeightKind::f16:
    case RouterWeightKind::bf16:
        return
            static_cast<std::uint64_t>(
                hidden_size) *
            expert_count *
            2ULL;
    case RouterWeightKind::q8_0:
        if (hidden_size % 32U != 0) {
            throw std::invalid_argument(
                "Q8_0 router hidden size must be divisible by 32");
        }
        return
            static_cast<std::uint64_t>(
                hidden_size / 32U) *
            expert_count *
            sizeof(Q8Block);
    }
    throw std::logic_error(
        "unreachable router weight kind");
}

template <
    typename Activation,
    RouterWeightKind WeightKind>
void launch_default_typed(
    const ExecutionDispatchContext& context,
    const HipMoeRouteLauncherConfig& config,
    const ExecutionBufferAddress& hidden,
    const RuntimeTensorBinding& weight,
    const ExecutionBufferAddress& indices,
    const ExecutionBufferAddress& weights) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            moe_route_kernel<
                Activation,
                WeightKind>),
        dim3(context.logical_rows),
        dim3(kThreads),
        0,
        hip_execution_stream(),
        reinterpret_cast<
            const Activation*>(
                hidden.address),
        reinterpret_cast<
            const void*>(
                weight.address),
        config.hidden_size,
        config.expert_count,
        config.top_k,
        config.normalize_selected,
        reinterpret_cast<
            std::uint32_t*>(
                indices.address),
        reinterpret_cast<float*>(
            weights.address));
    check_hip(
        hipGetLastError(),
        "HIP MoE route kernel launch");
}

ExecutionBufferAddress prepare_parallel_router_workspace(
    const ExecutionDispatchContext& context, const HipMoeRouteLauncherConfig& config) {
    if (context.logical_rows == 0 || context.logical_rows > 65535U)
        throw std::invalid_argument("parallel router row grid exceeds HIP launch limits");
    const auto workspace = resolve_dispatch_temporary(context, 0);
    const auto* workspace_allocation = find_arena_allocation(*context.arena, workspace.use);
    if (workspace_allocation == nullptr || workspace_allocation->first_step != context.step_id ||
        workspace_allocation->last_step != context.step_id)
        throw std::invalid_argument("parallel router workspace does not belong to this step");
    const auto required = static_cast<std::uint64_t>(context.logical_rows) * config.expert_count * sizeof(float);
    if (workspace.use.buffer.buffer_class != ExecutionBufferClass::router_workspace ||
        workspace.address % 16U != 0 || workspace.bytes < required || workspace.allocation_generation == 0)
        throw std::invalid_argument("parallel router requires admitted aligned FP32 logits workspace");
    return workspace;
}

template <typename Activation, RouterWeightKind WeightKind>
void launch_typed(
    const ExecutionDispatchContext& context, const HipMoeRouteLauncherConfig& config,
    const ExecutionBufferAddress& hidden, const RuntimeTensorBinding& weight,
    const ExecutionBufferAddress& indices, const ExecutionBufferAddress& weights) {
    if (!config.parallel_projection) {
        launch_default_typed<Activation, WeightKind>(context, config, hidden, weight, indices, weights);
        return;
    }
    const auto workspace = prepare_parallel_router_workspace(context, config);
    const auto* bf16 = WeightKind == RouterWeightKind::f32 ? q6p::find(reinterpret_cast<const void*>(weight.address)) : nullptr;
    const bool half = bf16 && bf16->kind == q6p::Kind::bf16 && bf16->rows == config.expert_count && bf16->cols == config.hidden_size;
    const bool staged = config.staged_logits && WeightKind == RouterWeightKind::f32 &&
        !half && staged_router_logits_eligible(config.hidden_size, config.expert_count, 0, weight.address);
    if (staged) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_moe_route_logits_kernel<Activation>),
            dim3(256U, context.logical_rows), dim3(256U), 0, hip_execution_stream(),
            reinterpret_cast<const Activation*>(hidden.address), reinterpret_cast<const void*>(weight.address),
            reinterpret_cast<float*>(workspace.address));
    } else if (half) {
        // Registered BF16 copy of BF16-exact F32 weights: load_weight<bf16> yields the same coefficients.
        hipLaunchKernelGGL(HIP_KERNEL_NAME(parallel_moe_route_logits_kernel<Activation, RouterWeightKind::bf16>),
            dim3((config.expert_count + kRouterExpertsPerBlock - 1U) / kRouterExpertsPerBlock, context.logical_rows),
            dim3(kRouterProjectionThreads), 0, hip_execution_stream(),
            reinterpret_cast<const Activation*>(hidden.address), reinterpret_cast<const void*>(bf16->base),
            config.hidden_size, config.expert_count, reinterpret_cast<float*>(workspace.address));
    } else {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(parallel_moe_route_logits_kernel<Activation, WeightKind>),
        dim3((config.expert_count + kRouterExpertsPerBlock - 1U) / kRouterExpertsPerBlock, context.logical_rows),
        dim3(kRouterProjectionThreads), 0, hip_execution_stream(),
        reinterpret_cast<const Activation*>(hidden.address), reinterpret_cast<const void*>(weight.address),
        config.hidden_size, config.expert_count, reinterpret_cast<float*>(workspace.address));
    }
    check_hip(hipGetLastError(), "HIP parallel MoE route logits launch");
    if (config.wave_topk) {
        hipLaunchKernelGGL(wave_parallel_moe_route_select_kernel, dim3(context.logical_rows), dim3(kThreads),
            0, hip_execution_stream(), reinterpret_cast<const float*>(workspace.address),
            config.expert_count, config.top_k, config.normalize_selected,
            reinterpret_cast<std::uint32_t*>(indices.address), reinterpret_cast<float*>(weights.address));
    } else if (config.ranked_topk) {
        hipLaunchKernelGGL(ranked_parallel_moe_route_select_kernel, dim3(context.logical_rows), dim3(kThreads),
            0, hip_execution_stream(), reinterpret_cast<const float*>(workspace.address),
            config.expert_count, config.top_k, config.normalize_selected,
            reinterpret_cast<std::uint32_t*>(indices.address), reinterpret_cast<float*>(weights.address));
    } else {
    hipLaunchKernelGGL(parallel_moe_route_select_kernel, dim3(context.logical_rows), dim3(kThreads),
        0, hip_execution_stream(), reinterpret_cast<const float*>(workspace.address),
        config.expert_count, config.top_k, config.normalize_selected,
        reinterpret_cast<std::uint32_t*>(indices.address), reinterpret_cast<float*>(weights.address));
    }
    check_hip(hipGetLastError(), "HIP parallel MoE route selection launch");
}

template <typename Activation>
void launch_by_weight(
    RouterWeightKind kind,
    const ExecutionDispatchContext& context,
    const HipMoeRouteLauncherConfig& config,
    const ExecutionBufferAddress& hidden,
    const RuntimeTensorBinding& weight,
    const ExecutionBufferAddress& indices,
    const ExecutionBufferAddress& weights) {
    switch (kind) {
    case RouterWeightKind::f32:
        launch_typed<
            Activation,
            RouterWeightKind::f32>(
                context,
                config,
                hidden,
                weight,
                indices,
                weights);
        return;
    case RouterWeightKind::f16:
        launch_typed<
            Activation,
            RouterWeightKind::f16>(
                context,
                config,
                hidden,
                weight,
                indices,
                weights);
        return;
    case RouterWeightKind::bf16:
        launch_typed<
            Activation,
            RouterWeightKind::bf16>(
                context,
                config,
                hidden,
                weight,
                indices,
                weights);
        return;
    case RouterWeightKind::q8_0:
        launch_typed<
            Activation,
            RouterWeightKind::q8_0>(
                context,
                config,
                hidden,
                weight,
                indices,
                weights);
        return;
    }
    throw std::logic_error(
        "unreachable router weight kind");
}

struct PreparedRouterLaunch {
    const HipMoeRouteLauncherConfig* config;
    ExecutionBufferAddress hidden, indices, route_weights;
    const RuntimeTensorBinding* runtime_tensor;
    RouterWeightKind kind;
};
PreparedRouterLaunch prepare_router_launch(const ExecutionDispatchContext& context, void* user_data) {
    if (context.op !=
        SerialOp::moe_route) {
        throw std::invalid_argument(
            "HIP MoE route launcher attached to wrong operator");
    }
    if (context.reads.size() != 1 ||
        context.writes.size() != 2) {
        throw std::invalid_argument(
            "HIP MoE route expects one read and two write buffers");
    }
    if (user_data == nullptr) {
        throw std::invalid_argument(
            "HIP MoE route launcher requires configuration");
    }

    const auto& config =
        *static_cast<
            HipMoeRouteLauncherConfig*>(
                user_data);
    validate_config(config);
    if (context.temporaries.size() != (config.parallel_projection ? 1U : 0U))
        throw std::invalid_argument("router temporary IO differs from launcher topology");
    if (config.parallel_projection &&
        context.temporaries.front().buffer.buffer_class != ExecutionBufferClass::router_workspace)
        throw std::invalid_argument("parallel router temporary must use router_workspace");

    const auto hidden =
        resolve_dispatch_read(
            context, 0);
    const auto indices =
        resolve_dispatch_write(
            context, 0);
    const auto route_weights =
        resolve_dispatch_write(
            context, 1);

    const auto& access =
        resolve_dispatch_tensor_access(
            context, 0);
    if (access.tensor_indices.size() != 1U) {
        throw std::invalid_argument(
            "HIP MoE route requires exactly one router tensor");
    }
    if (context.resources == nullptr ||
        context.resources->model_manifest == nullptr) {
        throw std::invalid_argument(
            "HIP MoE route requires model manifest resources");
    }

    if (config.wave_topk && (!config.wave_manifest_admission ||
        context.resources->model_manifest != &config.wave_manifest_admission->manifest()))
        throw std::invalid_argument("wave router requires the exact owned admitted manifest");
    const auto tensor_index =
        access.tensor_indices.front();
    if (tensor_index >=
        context.resources->
            model_manifest->
            tensors.size()) {
        throw std::out_of_range(
            "HIP MoE route tensor index exceeds manifest");
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
            TensorRole::moe_router ||
        manifest_tensor.dimensions.size() < 2 ||
        manifest_tensor.dimensions[0] !=
            config.hidden_size ||
        manifest_tensor.dimensions[1] !=
            config.expert_count) {
        throw std::invalid_argument(
            "HIP MoE route tensor geometry does not match router contract");
    }
    if (!runtime_tensor_gpu_visible(runtime_tensor.storage) ||
        runtime_tensor.address == 0) {
        throw std::invalid_argument(
            "HIP MoE route requires resident or repacked GPU-visible router weights");
    }

    if (config.wave_topk && (hidden.address % (config.activation == HipActivationStorage::bf16 ? 2U : 4U) != 0 ||
        indices.address % 4U != 0 || route_weights.address % 4U != 0))
        throw std::invalid_argument("wave router requires aligned runtime inputs and outputs");
    const auto kind =
        weight_kind(
            manifest_tensor.ggml_type);
    const auto required_weights =
        required_router_weight_bytes(
            kind,
            config.hidden_size,
            config.expert_count);
    if (runtime_tensor.bytes <
        required_weights) {
        throw std::invalid_argument(
            "HIP MoE route runtime tensor is smaller than required packed weights");
    }

    const auto activation_bytes =
        config.activation ==
                HipActivationStorage::bf16
            ? 2ULL
            : 4ULL;
    const auto hidden_required =
        static_cast<std::uint64_t>(
            context.logical_rows) *
        config.hidden_size *
        activation_bytes;
    const auto indices_required =
        static_cast<std::uint64_t>(
            context.logical_rows) *
        config.top_k *
        sizeof(std::uint32_t);
    const auto route_weights_required =
        static_cast<std::uint64_t>(
            context.logical_rows) *
        config.top_k *
        sizeof(float);

    if (hidden.bytes <
            hidden_required ||
        indices.bytes <
            indices_required ||
        route_weights.bytes <
            route_weights_required) {
        throw std::invalid_argument(
            "HIP MoE route scratch buffers are undersized");
    }

    return {&config, hidden, indices, route_weights, &runtime_tensor, kind};
}
void preflight_moe_route(const ExecutionDispatchContext& context, void* user_data) {
    const auto prepared = prepare_router_launch(context, user_data);
    if (prepared.config->parallel_projection)
        (void)prepare_parallel_router_workspace(context, *prepared.config);
}

void launch_moe_route(const ExecutionDispatchContext& context, void* user_data) {
    const auto prepared = prepare_router_launch(context, user_data);
    const auto& config = *prepared.config;
    const auto& hidden = prepared.hidden;
    const auto& indices = prepared.indices;
    const auto& route_weights = prepared.route_weights;
    const auto& runtime_tensor = *prepared.runtime_tensor;
    const auto kind = prepared.kind;
    switch (config.activation) {
    case HipActivationStorage::bf16:
        launch_by_weight<
            std::uint16_t>(
                kind,
                context,
                config,
                hidden,
                runtime_tensor,
                indices,
                route_weights);
        return;
    case HipActivationStorage::fp32:
        launch_by_weight<float>(
            kind,
            context,
            config,
            hidden,
            runtime_tensor,
            indices,
            route_weights);
        return;
    }
    throw std::logic_error(
        "unreachable router activation storage");
}

std::string route_kernel_id(
    const ExecutionBindingRequest& request,
    const HipMoeRouteLauncherConfig& config) {
    return
        "hip.moe_route." +
        std::string(
            activation_name(
                config.activation)) +
        ".h" +
        std::to_string(
            config.hidden_size) +
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
            request.request_concurrency) + (config.staged_logits ? ".staged_logits" : "") + (config.wave_topk ? ".wave.parallel" : config.ranked_topk ? ".ranked.parallel" :
            config.parallel_projection ? ".parallel" : "");
}

std::uint64_t launcher_identity(
    const HipMoeRouteLauncherConfig& config) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, kLauncherTag);
    hash_u64(hash, config.hidden_size);
    hash_u64(hash, config.expert_count);
    hash_u64(hash, config.top_k);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            config.activation));
    hash_u64(
        hash,
        config.normalize_selected
            ? 1ULL
            : 0ULL);
    if (config.parallel_projection) hash_u64(hash, 0x524f555445525057ULL);
    if (config.staged_logits) hash_u64(hash, 0x5354474c4f474954ULL);
    if (config.ranked_topk) hash_u64(hash, 0x52414e4b544f504bULL);
    if (config.wave_topk) hash_u64(hash, 0x57415645544f504bULL);
    return hash;
}

struct PreparedRouterRows {
    RouterRowsArgs args{};
    std::uint32_t rows = 0;
    const RuntimeTensorBinding* tensor = nullptr;
    RouterWeightKind kind = RouterWeightKind::f32;
    bool wave_logits = false;
};
// Every row passes the complete serial admission; rows add only the window
// shape (one step, consecutive positions, one weight) and cross-row aliasing.
PreparedRouterRows prepare_router_rows(std::span<const ExecutionDispatchContext> rows,
    const HipMoeRouteLauncherConfig& config) {
    if (rows.size() != 2U && rows.size() != 4U && rows.size() != 8U)
        throw std::invalid_argument("known-window router rows count must be 2, 4 or 8");
    if (!config.parallel_projection)
        throw std::invalid_argument("known-window router rows require the parallel router topology");
    PreparedRouterRows out;
    out.rows = static_cast<std::uint32_t>(rows.size());
    struct Range { std::uintptr_t address; std::uint64_t bytes; };
    std::array<Range, 3U * kRouterMaxRows> writes{};
    std::array<Range, kRouterMaxRows> reads{};
    const auto& first = rows.front();
    auto* user = const_cast<HipMoeRouteLauncherConfig*>(&config);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const auto& row = rows[r];
        // A cohort window starts a new session segment (its own window identity) at row r.
        const auto& head = rows[known_window_segment_start(rows, r) ? r : r - 1U];
        if (row.logical_rows != 1U || row.request_concurrency != 1U ||
            row.execution_identity != head.execution_identity || row.step_id != first.step_id ||
            row.layer != first.layer || row.kernel_id != first.kernel_id || !known_window_row_position_follows(rows, r))
            throw std::invalid_argument("known-window router rows identity/position differs");
        const auto prepared = prepare_router_launch(row, user);
        const auto workspace = prepare_parallel_router_workspace(row, config);
        if (r && (prepared.runtime_tensor->address != out.tensor->address || prepared.kind != out.kind))
            throw std::invalid_argument("known-window router rows selected weights differ");
        out.tensor = prepared.runtime_tensor; out.kind = prepared.kind;
        out.args.hidden[r] = reinterpret_cast<const void*>(prepared.hidden.address);
        out.args.workspace[r] = reinterpret_cast<float*>(workspace.address);
        out.args.indices[r] = reinterpret_cast<std::uint32_t*>(prepared.indices.address);
        out.args.weights[r] = reinterpret_cast<float*>(prepared.route_weights.address);
        reads[r] = {prepared.hidden.address, static_cast<std::uint64_t>(config.hidden_size) *
            (config.activation == HipActivationStorage::bf16 ? 2U : 4U)};
        writes[3U * r] = {prepared.indices.address, config.top_k * 4ULL};
        writes[3U * r + 1U] = {prepared.route_weights.address, config.top_k * 4ULL};
        writes[3U * r + 2U] = {workspace.address, config.expert_count * 4ULL};
    }
    const auto per_vector = config.activation == HipActivationStorage::bf16 ? 8U : 4U;
    out.wave_logits = out.kind == RouterWeightKind::f32 && config.hidden_size % kWaveRowsChunk == 0U &&
        out.tensor->address % 4U == 0U && kWaveRowsChunk % per_vector == 0U &&
        std::all_of(reads.begin(), reads.begin() + rows.size(), [](Range r) { return r.address % 16U == 0U; });
    const auto overlap = [](Range a, Range b) { return a.address < b.address + b.bytes && b.address < a.address + a.bytes; };
    const Range weight{out.tensor->address, out.tensor->bytes};
    for (std::size_t w = 0; w < 3U * rows.size(); ++w) {
        if (overlap(writes[w], weight)) throw std::invalid_argument("known-window router rows output aliases weights");
        for (std::size_t r = 0; r < rows.size(); ++r)
            if (overlap(writes[w], reads[r])) throw std::invalid_argument("known-window router rows output aliases an input row");
        for (std::size_t o = 0; o < w; ++o)
            if (overlap(writes[w], writes[o])) throw std::invalid_argument("known-window router rows outputs alias");
    }
    return out;
}

template <typename Activation, RouterWeightKind WeightKind>
void launch_router_rows_typed(const PreparedRouterRows& p, const HipMoeRouteLauncherConfig& config) {
    const dim3 grid((config.expert_count + kRowsExpertsPerBlock - 1U) / kRowsExpertsPerBlock);
    const auto* weights = reinterpret_cast<const void*>(p.tensor->address);
    const auto stream = hip_execution_stream();
    if (WeightKind == RouterWeightKind::f32 && p.wave_logits) {
        const auto* f32 = static_cast<const float*>(weights);
        // 8 waves per block; each wave holds 32 / rows experts.
        const auto blocks = [&](std::uint32_t rows) { return dim3((config.expert_count * rows + 255U) / 256U); };
        const bool aligned = config.hidden_size % kWaveRowsChunk == 0U;
        // STRIX_ROUTER_ROWS_CHUNK=64|128|256 (the hidden size must be a multiple): same chains; 128 measured best.
        static const std::uint32_t chunk = [] {
            const char* v = std::getenv("STRIX_ROUTER_ROWS_CHUNK");
            const auto n = v ? static_cast<std::uint32_t>(std::atoi(v)) : 128U;
            return n == 64U || n == 256U ? n : 128U;
        }();
        // STRIX_ROUTER_STAGED=0: the lanes form below for the 2560-wide router as well (same chains).
        static const bool staged = strix::envcfg::flag("STRIX_ROUTER_STAGED", true);
        // A registered BF16 copy of the (BF16-exact) F32 weights replaces them: same coefficients and chains.
        const auto* bf16 = q6p::find(weights);
        const bool half = bf16 && bf16->kind == q6p::Kind::bf16 && bf16->rows == config.expert_count &&
            bf16->cols == config.hidden_size;
        const auto lanes = [&]<std::uint32_t R>() {
            // Twice as many independent blocks for the 512-expert single-session router.
            // Each (row, expert) retains the ascending fused chain and the same loader ring depth.
            static const bool small = strix::envcfg::flag("STRIX_QKERN_ROUTER8", false);
            if (small && staged && config.hidden_size == 20U * 128U) {
                constexpr unsigned experts = 8U, loaders = 4U;
                const auto grid8 = dim3((config.expert_count + experts - 1U) / experts);
                const auto threads8 = dim3(32U * (staged_router_chain_waves<R, experts>() + loaders));
                if (half)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_rows_moe_route_logits_kernel<Activation, R, 20U, loaders, true, experts>),
                        grid8, threads8, 0, stream, p.args, bf16->base, config.expert_count);
                else
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_rows_moe_route_logits_kernel<Activation, R, 20U, loaders, false, experts>),
                        grid8, threads8, 0, stream, p.args, f32, config.expert_count);
                return;
            }
            if (staged && config.hidden_size == 20U * 128U) {
                // 16 experts per block: R / 2 chain waves and 8 loader waves keeping 8 chunks in flight.
                constexpr std::uint32_t experts = 16U, loaders = 8U;
                // STRIX_ROUTER_STAGE_ACT=1: one-row launches; =2: also the 2- and 4-row window launches (in-situ test).
                // Default (unset) = 2 (measured in the engine: R1 38.9 -> 30.0 us, R2 35.9 -> 32.6, R4 36.5 -> 35.8; R8 stays unstaged); =0 off.
                static const unsigned stage_mode = [] { const char* v = std::getenv("STRIX_ROUTER_STAGE_ACT"); return !v || !*v ? 2U :
                    v[0] == '1' || v[0] == '2' ? static_cast<unsigned>(v[0] - '0') : 0U; }();
                const bool stage_act = stage_mode == 1U ? R == 1U : stage_mode == 2U && R <= 4U;
                static const bool chunk256 = strix::envcfg::flag("STRIX_ROUTER_CHUNK256", false);
                if constexpr (R == 1U) if (stage_act && chunk256) {   // fixture: the staged forms win at one row only
                    if (half)
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_rows_moe_route_logits_kernel<Activation, R, 10U, loaders, true, experts, true, 256U>),
                            dim3((config.expert_count + experts - 1U) / experts), dim3(32U * (staged_router_chain_waves<R>() + loaders)),
                            0, stream, p.args, bf16->base, config.expert_count);
                    else
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_rows_moe_route_logits_kernel<Activation, R, 10U, loaders, false, experts, true, 256U>),
                            dim3((config.expert_count + experts - 1U) / experts), dim3(32U * (staged_router_chain_waves<R>() + loaders)),
                            0, stream, p.args, f32, config.expert_count);
                    return;
                }
                if constexpr (R <= 4U && (sizeof(Activation) == 2U || R <= 2U)) if (stage_act) {
                    if (half)
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_rows_moe_route_logits_kernel<Activation, R, 20U, loaders, true, experts, true>),
                            dim3((config.expert_count + experts - 1U) / experts), dim3(32U * (staged_router_chain_waves<R>() + loaders)),
                            0, stream, p.args, bf16->base, config.expert_count);
                    else
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_rows_moe_route_logits_kernel<Activation, R, 20U, loaders, false, experts, true>),
                            dim3((config.expert_count + experts - 1U) / experts), dim3(32U * (staged_router_chain_waves<R>() + loaders)),
                            0, stream, p.args, f32, config.expert_count);
                    return;
                }
                if (half)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_rows_moe_route_logits_kernel<Activation, R, 20U, loaders, true>),
                        dim3((config.expert_count + experts - 1U) / experts), dim3(32U * (staged_router_chain_waves<R>() + loaders)),
                        0, stream, p.args, bf16->base, config.expert_count);
                else
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_rows_moe_route_logits_kernel<Activation, R, 20U, loaders>),
                        dim3((config.expert_count + experts - 1U) / experts), dim3(32U * (staged_router_chain_waves<R>() + loaders)),
                        0, stream, p.args, f32, config.expert_count);
                return;
            }
            if (half && chunk == 128U && config.hidden_size % 128U == 0U)
                hipLaunchKernelGGL(HIP_KERNEL_NAME(lanes_rows_moe_route_logits_kernel<Activation, R, 128U, RouterRowsArgs, true>),
                    blocks(R), dim3(256U), 0, stream, p.args, bf16->base, config.hidden_size, config.expert_count);
            else if (chunk == 256U && config.hidden_size % 256U == 0U)
                hipLaunchKernelGGL(HIP_KERNEL_NAME(lanes_rows_moe_route_logits_kernel<Activation, R, 256U>),
                    blocks(R), dim3(256U), 0, stream, p.args, f32, config.hidden_size, config.expert_count);
            else if (chunk == 128U && config.hidden_size % 128U == 0U)
                hipLaunchKernelGGL(HIP_KERNEL_NAME(lanes_rows_moe_route_logits_kernel<Activation, R, 128U>),
                    blocks(R), dim3(256U), 0, stream, p.args, f32, config.hidden_size, config.expert_count);
            else hipLaunchKernelGGL(HIP_KERNEL_NAME(lanes_rows_moe_route_logits_kernel<Activation, R>),
                    blocks(R), dim3(256U), 0, stream, p.args, f32, config.hidden_size, config.expert_count);
        };
        if (!aligned) {
            const dim3 waves((config.expert_count + 7U) / 8U);
            if (p.rows == 2U) hipLaunchKernelGGL(HIP_KERNEL_NAME(wave_rows_moe_route_logits_kernel<Activation, 2U>),
                waves, dim3(256U), 0, stream, p.args, f32, config.hidden_size, config.expert_count);
            else if (p.rows == 4U) hipLaunchKernelGGL(HIP_KERNEL_NAME(wave_rows_moe_route_logits_kernel<Activation, 4U>),
                waves, dim3(256U), 0, stream, p.args, f32, config.hidden_size, config.expert_count);
            else hipLaunchKernelGGL(HIP_KERNEL_NAME(wave_rows_moe_route_logits_kernel<Activation, 8U>),
                waves, dim3(256U), 0, stream, p.args, f32, config.hidden_size, config.expert_count);
        } else if (p.rows == 2U) lanes.template operator()<2U>();
        else if (p.rows == 4U) lanes.template operator()<4U>();
        else lanes.template operator()<8U>();
    } else if (p.rows == 2U) hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_moe_route_logits_kernel<Activation, WeightKind, 2U>),
        grid, dim3(256U), 0, stream, p.args, weights, config.hidden_size, config.expert_count);
    else if (p.rows == 4U) hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_moe_route_logits_kernel<Activation, WeightKind, 4U>),
        grid, dim3(256U), 0, stream, p.args, weights, config.hidden_size, config.expert_count);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_moe_route_logits_kernel<Activation, WeightKind, 8U>),
        grid, dim3(256U), 0, stream, p.args, weights, config.hidden_size, config.expert_count);
    check_hip(hipGetLastError(), "known-window router rows logits launch");
    // STRIX_WINDOW_ROUTER_SELECT=1: the window's rows in one block, one wave per row (wave_rows_route_select_kernel, same bits
    // as the block-per-row wave form).
    static const bool window_wave_rows = strix::envcfg::flag("STRIX_WINDOW_ROUTER_SELECT", false);
    if (config.wave_topk && window_wave_rows && p.rows <= 8U && config.expert_count <= kMaxExperts && config.top_k <= kRouterWave)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(wave_rows_route_select_kernel<RouterRowsArgs>),
            dim3(p.rows), dim3(kRouterWave), 0, stream, p.args, config.expert_count, config.top_k, config.normalize_selected, p.rows);
    else if (config.wave_topk) hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_moe_route_select_kernel<RouterSelect::wave>),
        dim3(p.rows), dim3(kThreads), 0, stream, p.args, config.expert_count, config.top_k, config.normalize_selected);
    else if (config.ranked_topk) hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_moe_route_select_kernel<RouterSelect::ranked>),
        dim3(p.rows), dim3(kThreads), 0, stream, p.args, config.expert_count, config.top_k, config.normalize_selected);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_moe_route_select_kernel<RouterSelect::sorted>),
        dim3(p.rows), dim3(kThreads), 0, stream, p.args, config.expert_count, config.top_k, config.normalize_selected);
    check_hip(hipGetLastError(), "known-window router rows selection launch");
}
template <typename Activation>
void launch_router_rows_by_weight(const PreparedRouterRows& p, const HipMoeRouteLauncherConfig& config) {
    switch (p.kind) {
    case RouterWeightKind::f32: launch_router_rows_typed<Activation, RouterWeightKind::f32>(p, config); return;
    case RouterWeightKind::f16: launch_router_rows_typed<Activation, RouterWeightKind::f16>(p, config); return;
    case RouterWeightKind::bf16: launch_router_rows_typed<Activation, RouterWeightKind::bf16>(p, config); return;
    case RouterWeightKind::q8_0: launch_router_rows_typed<Activation, RouterWeightKind::q8_0>(p, config); return;
    }
    throw std::logic_error("unreachable router weight kind");
}

}  // namespace

void moe_route_known_window_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipMoeRouteLauncherConfig& config) {
    static_cast<void>(prepare_router_rows(rows, config));
}
void moe_route_known_window_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipMoeRouteLauncherConfig& config) {
    const auto prepared = prepare_router_rows(rows, config);
    if (config.activation == HipActivationStorage::bf16) launch_router_rows_by_weight<std::uint16_t>(prepared, config);
    else launch_router_rows_by_weight<float>(prepared, config);
}
void moe_route_known_window_cohort_launch(std::span<const std::span<const ExecutionDispatchContext>> segments,
    const HipMoeRouteLauncherConfig& config) {
    if(segments.size()<2U || segments.size()>8U || config.hidden_size%kWaveRowsChunk || !config.wave_topk)
        throw std::invalid_argument("wide router cohort geometry or topology differs");
    RouterCohortRowsArgs args{};
    const auto lead=prepare_router_rows(segments[0],config);
    if(lead.kind!=RouterWeightKind::f32) throw std::invalid_argument("wide router cohort requires F32 weights");
    for(const auto segment:segments) {
        const auto p=prepare_router_rows(segment,config);
        if(args.count+p.rows>32U || p.kind!=lead.kind || p.tensor->address!=lead.tensor->address)
            throw std::invalid_argument("wide router cohort rows or weights differ");
        for(std::uint32_t r=0;r<p.rows;++r,++args.count) {
            args.hidden[args.count]=p.args.hidden[r];args.workspace[args.count]=p.args.workspace[r];
            args.indices[args.count]=p.args.indices[r];args.weights[args.count]=p.args.weights[r];
        }
    }
    const auto run=[&]<typename Activation>() {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(wave_rows_moe_route_logits_kernel<Activation,32U,RouterCohortRowsArgs>),
            dim3((config.expert_count+7U)/8U),dim3(256U),0,hip_execution_stream(),args,
            reinterpret_cast<const float*>(lead.tensor->address),config.hidden_size,config.expert_count);
    };
    if(config.activation==HipActivationStorage::bf16) run.template operator()<std::uint16_t>();else run.template operator()<float>();
    check_hip(hipGetLastError(),"wide router cohort logits launch");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_moe_route_select_kernel<RouterSelect::wave,RouterCohortRowsArgs>),
        dim3(args.count),dim3(kThreads),0,hip_execution_stream(),args,config.expert_count,config.top_k,config.normalize_selected);
    check_hip(hipGetLastError(),"wide router cohort select launch");
}

namespace {
// All T rows of a router step: every 8-row tile is admitted like a window call and the tiles
// must be one stride apart in each view. The f32 lanes logits and the wave top-k run all tiles
// in one launch (the 5 MB router matrix is read from DRAM once per step instead of once per
// tile); other router forms launch tile by tile.
void route_prefill(std::span<const ExecutionDispatchContext> rows, const HipMoeRouteLauncherConfig& config, bool launch) {
    if (rows.size() < 16U || rows.size() % 8U) throw std::invalid_argument("router prefill rows require 8-row tiles");
    const auto tiles = static_cast<std::uint32_t>(rows.size() / 8U);
    if (tiles > 65535U) throw std::invalid_argument("router prefill rows exceed the grid");
    const auto first = prepare_router_rows(rows.first(8U), config);
    const auto second = prepare_router_rows(rows.subspan(8U, 8U), config);
    const auto at = [](const auto* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); };
    const auto stride = [&](const auto* a, const auto* b) -> std::uint64_t {
        if (at(b) <= at(a) || (at(b) - at(a)) % 16U) throw std::invalid_argument("router prefill tiles are not strided");
        return at(b) - at(a);
    };
    RouterTileRowsArgs args;
    static_cast<RouterRowsArgs&>(args) = first.args;
    args.tile_hidden = stride(first.args.hidden[0], second.args.hidden[0]);
    args.tile_workspace = stride(first.args.workspace[0], second.args.workspace[0]);
    args.tile_indices = stride(first.args.indices[0], second.args.indices[0]);
    args.tile_weights = stride(first.args.weights[0], second.args.weights[0]);
    bool together = first.kind == RouterWeightKind::f32 && first.wave_logits && config.wave_topk &&
        config.hidden_size % 128U == 0U;
    // Every tile is admitted (and the batched form decided) before anything is launched.
    std::vector<PreparedRouterRows> prepared{first};
    prepared.reserve(tiles);
    for (std::uint32_t t = 1; t < tiles; ++t) {
        prepared.push_back(t < 2U ? second : prepare_router_rows(rows.subspan(t * 8U, 8U), config));
        const auto& tile = prepared.back();
        together = together && tile.wave_logits;
        const auto offset = static_cast<std::uint64_t>(t);
        for (std::uint32_t r = 0; r < 8U; ++r)
            if (at(tile.args.hidden[r]) != at(first.args.hidden[r]) + offset * args.tile_hidden ||
                at(tile.args.workspace[r]) != at(first.args.workspace[r]) + offset * args.tile_workspace ||
                at(tile.args.indices[r]) != at(first.args.indices[r]) + offset * args.tile_indices ||
                at(tile.args.weights[r]) != at(first.args.weights[r]) + offset * args.tile_weights)
                throw std::invalid_argument("router prefill tile views are not strided");
    }
    // Inputs of every row vs outputs of every row (tiles run concurrently in the batched form).
    const auto span = [&](const auto* const* views, std::uint64_t stride, std::uint64_t bytes) {
        std::uintptr_t low = UINTPTR_MAX, high = 0;
        for (std::uint32_t r = 0; r < 8U; ++r) {
            low = std::min(low, at(views[r]));
            high = std::max(high, at(views[r]) + (tiles - 1U) * stride + bytes);
        }
        return std::pair<std::uintptr_t, std::uintptr_t>{low, high};
    };
    const auto hidden_bytes = static_cast<std::uint64_t>(config.hidden_size) *
        (config.activation == HipActivationStorage::bf16 ? 2U : 4U);
    const auto in = span(first.args.hidden, args.tile_hidden, hidden_bytes);
    for (const auto& out : {span(first.args.workspace, args.tile_workspace, config.expert_count * 4ULL),
             span(first.args.indices, args.tile_indices, config.top_k * 4ULL),
             span(first.args.weights, args.tile_weights, config.top_k * 4ULL)})
        if (out.first < in.second && in.first < out.second) throw std::invalid_argument("router prefill outputs alias the input rows");
    if (launch && !together)
        for (const auto& tile : prepared) {
            if (config.activation == HipActivationStorage::bf16) launch_router_rows_by_weight<std::uint16_t>(tile, config);
            else launch_router_rows_by_weight<float>(tile, config);
        }
    if (!launch || !together) return;
    const auto stream = hip_execution_stream();
    const auto* f32 = reinterpret_cast<const float*>(first.tensor->address);
    const dim3 logits((config.expert_count * 8U + 255U) / 256U, tiles);
    // STRIX_ROUTER_PREFILL_TILES=1|2|4|8: 8-row tiles per lane sharing each coefficient load
    // (1: the window kernel per tile). STRIX_ROUTER_PREFILL_CHUNK=32|64: coefficients in flight.
    // Measured at T = 256: 359 us per call (one tile per lane, 128 in flight), 323 us (2 tiles, 64);
    // 4 or 8 tiles and an LDS-staged 32 x 32 block were slower (latency, not bandwidth, bound).
    static const unsigned tiles_per_lane = [] { const char* v = std::getenv("STRIX_ROUTER_PREFILL_TILES");
        const unsigned n = v ? static_cast<unsigned>(std::atoi(v)) : 2U; return n == 1U || n == 4U || n == 8U ? n : 2U; }();
    static const unsigned chunk = [] { const char* v = std::getenv("STRIX_ROUTER_PREFILL_CHUNK");
        const unsigned n = v ? static_cast<unsigned>(std::atoi(v)) : 64U; return n == 32U ? 32U : 64U; }();
    const bool hidden_aligned = config.hidden_size % 64U == 0U;
    const auto tiled = [&]<typename Activation, unsigned T, unsigned C>() {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(lanes_tiles_moe_route_logits_kernel<Activation, T, C>),
            dim3((config.expert_count * 8U + 255U) / 256U, (tiles + T - 1U) / T), dim3(256U), 0, stream, args, f32,
            config.hidden_size, config.expert_count, tiles);
    };
    const auto by_tiles = [&]<typename Activation>() {
        const auto by_chunk = [&]<unsigned T>() {
            if (chunk == 64U) tiled.template operator()<Activation, T, 64U>(); else tiled.template operator()<Activation, T, 32U>();
        };
        if (tiles_per_lane == 2U) by_chunk.template operator()<2U>();
        else if (tiles_per_lane == 8U) by_chunk.template operator()<8U>();
        else by_chunk.template operator()<4U>();
    };
    // Fast prefill profile, router component: BF16 WMMA logits with two-term weights (hip_router_fast_prefill.inc).
    if (config.router_fast_prefill && config.activation == HipActivationStorage::bf16 && config.hidden_size % router_fast::kK == 0U &&
        config.expert_count % router_fast::kN == 0U) {
        // STRIX_PREFILL_ROUTER_TUNE: pre-split weights and K steps of 64 (router_fast::logits_presplit_kernel, same bits), 1 with
        // 64 experts per workgroup, 2 (default) with 128 from 2048 rows (32K prefill: 3.56 -> 1.21 ms per 8192-row call) and
        // 64 below (as many workgroups as logits_kernel<2>); 0 runs logits_kernel<2>.
        // STRIX_PREFILL_ROUTER_CHECK=1 (diagnostic, synchronizes): also runs logits_kernel<2> on the same rows and prints the
        // number of logits whose bits differ.
        static const unsigned router_tune = [] { const char* v = std::getenv("STRIX_PREFILL_ROUTER_TUNE"); return v ? static_cast<unsigned>(std::atoi(v)) : 2U; }();
        static const bool router_check = strix::envcfg::flag("STRIX_PREFILL_ROUTER_CHECK", false);
        const std::uint32_t total = tiles * 8U, experts = config.expert_count;
        const dim3 grid((total + router_fast::kM - 1U) / router_fast::kM, experts / router_fast::kN);
        const auto original = [&] {
            hipLaunchKernelGGL(router_fast::logits_kernel<2U>, grid, dim3(256U), 0, stream, args, f32, config.hidden_size, total);
            check_hip(hipGetLastError(), "router prefill BF16 WMMA logits launch");
        };
        const unsigned presplit_n = router_tune == 2U && experts % 128U == 0U && total >= 2048U ? 128U : 64U;
        if ((router_tune == 1U || router_tune == 2U) && config.hidden_size % router_fast::kK2 == 0U) {
            const std::size_t weights = static_cast<std::size_t>(experts) * config.hidden_size;
            static std::uint16_t* split = nullptr;
            static std::size_t split_capacity = 0;
            if (split_capacity < weights) {
                if (split) check_hip(hipFree(split), "router split weights release");
                split = nullptr; split_capacity = 0;
                check_hip(hipMalloc(&split, 2U * weights * sizeof(std::uint16_t)), "router split weights allocation");
                split_capacity = weights;
            }
            static float* copy = nullptr;
            static std::size_t copy_capacity = 0;
            static unsigned* mismatches = nullptr;
            const std::size_t logits_count = static_cast<std::size_t>(total) * experts;
            if (router_check) {
                if (copy_capacity < logits_count) {
                    if (copy) check_hip(hipFree(copy), "router check copy release");
                    copy = nullptr; copy_capacity = 0;
                    check_hip(hipMalloc(&copy, logits_count * sizeof(float)), "router check copy allocation");
                    copy_capacity = logits_count;
                }
                if (!mismatches) check_hip(hipMalloc(&mismatches, sizeof(unsigned)), "router check counter allocation");
                check_hip(hipMemsetAsync(mismatches, 0, sizeof(unsigned), stream), "router check counter clear");
                original();
                hipLaunchKernelGGL(router_fast::check_logits, dim3(total), dim3(256U), 0, stream, args, copy, mismatches, experts, total, 0U);
                check_hip(hipGetLastError(), "router check copy launch");
            }
            hipLaunchKernelGGL(router_fast::split_weights, dim3(static_cast<unsigned>(std::min<std::size_t>((weights + 255U) / 256U, 4096U))),
                dim3(256U), 0, stream, f32, split, split + weights, static_cast<std::uint32_t>(weights));
            check_hip(hipGetLastError(), "router weight split launch");
            static const bool fuse = strix::envcfg::flag("STRIX_PREFILL_ROUTER_FUSE", false);
            if (fuse && presplit_n == 128U && !router_check && experts <= kMaxExperts && config.top_k <= kRouterWave) {
                static unsigned* counters = nullptr;
                static std::uint32_t counter_capacity = 0;
                if (counter_capacity < grid.x) {
                    if (counters) check_hip(hipFree(counters), "router fuse counters release");
                    counters = nullptr; counter_capacity = 0;
                    check_hip(hipMalloc(&counters, sizeof(unsigned) * grid.x), "router fuse counters allocation");
                    check_hip(hipMemsetAsync(counters, 0, sizeof(unsigned) * grid.x, stream), "router fuse counters clear");
                    counter_capacity = grid.x;
                }
                const RouterSelectTail tail{counters, experts, config.top_k, config.normalize_selected};
                hipLaunchKernelGGL(HIP_KERNEL_NAME(router_fast::logits_presplit_kernel<128U, RouterSelectTail>), dim3(grid.x, experts / 128U),
                    dim3(256U), 0, stream, args, split, split + weights, config.hidden_size, total, tail);
                check_hip(hipGetLastError(), "router prefill fused logits + select launch");
                return;
            }
            if (presplit_n == 128U)
                hipLaunchKernelGGL(router_fast::logits_presplit_kernel<128U>, dim3(grid.x, experts / 128U), dim3(256U), 0, stream, args,
                    split, split + weights, config.hidden_size, total);
            else
                hipLaunchKernelGGL(router_fast::logits_presplit_kernel<64U>, grid, dim3(256U), 0, stream, args, split, split + weights,
                    config.hidden_size, total);
            check_hip(hipGetLastError(), "router prefill pre-split logits launch");
            if (router_check) {
                hipLaunchKernelGGL(router_fast::check_logits, dim3(total), dim3(256U), 0, stream, args, copy, mismatches, experts, total, 1U);
                check_hip(hipGetLastError(), "router check compare launch");
                unsigned bad = 0;
                check_hip(hipMemcpyAsync(&bad, mismatches, sizeof(bad), hipMemcpyDeviceToHost, stream), "router check read");
                check_hip(hipStreamSynchronize(stream), "router check sync");
                std::fprintf(stderr, "router-presplit-check rows=%u experts=%u mismatched_logits=%u\n", total, experts, bad);
            }
        } else original();
    } else if (tiles_per_lane > 1U && hidden_aligned) {
        if (config.activation == HipActivationStorage::bf16) by_tiles.template operator()<std::uint16_t>();
        else by_tiles.template operator()<float>();
    } else if (config.activation == HipActivationStorage::bf16)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(lanes_rows_moe_route_logits_kernel<std::uint16_t, 8U, 128U, RouterTileRowsArgs>),
            logits, dim3(256U), 0, stream, args, f32, config.hidden_size, config.expert_count);
    else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(lanes_rows_moe_route_logits_kernel<float, 8U, 128U, RouterTileRowsArgs>),
            logits, dim3(256U), 0, stream, args, f32, config.hidden_size, config.expert_count);
    check_hip(hipGetLastError(), "router prefill rows logits launch");
    // STRIX_PREFILL_ROUTER_SELECT=1: one wave per row, 8 rows per block (wave_rows_route_select_kernel, same bits).
    // Default ON (bitwise; =0 restores the block-per-row form).
    static const bool wave_rows_select = strix::envcfg::flag("STRIX_PREFILL_ROUTER_SELECT", true);
    if (wave_rows_select && config.expert_count <= kMaxExperts && config.top_k <= kRouterWave)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(wave_rows_route_select_kernel<RouterTileRowsArgs>),
            dim3(1U, tiles), dim3(kThreads), 0, stream, args, config.expert_count, config.top_k, config.normalize_selected, 8U);
    else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_moe_route_select_kernel<RouterSelect::wave, RouterTileRowsArgs>),
            dim3(8U, tiles), dim3(kThreads), 0, stream, args, config.expert_count, config.top_k, config.normalize_selected);
    check_hip(hipGetLastError(), "router prefill rows selection launch");
}
}  // namespace

void moe_route_known_prefill_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipMoeRouteLauncherConfig& config) {
    route_prefill(rows, config, false);
}
void moe_route_known_prefill_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipMoeRouteLauncherConfig& config) {
    route_prefill(rows, config, true);
}

void register_hip_moe_route_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipMoeRouteLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    validate_config(config);
    if (config.staged_logits != request.staged_moe_router_logits)
        throw std::invalid_argument("staged router variant mode differs from binding request");
    if (config.wave_topk != request.wave_moe_router_topk)
        throw std::invalid_argument("wave router variant mode differs from binding request");
    if (config.ranked_topk != request.ranked_moe_router_topk)
        throw std::invalid_argument("ranked router variant mode differs from binding request");
    if (config.parallel_projection != request.parallel_moe_router)
        throw std::invalid_argument("router variant mode differs from binding request");
    if (config.hidden_size !=
            model.hidden_size ||
        config.expert_count !=
            model.num_experts ||
        config.top_k !=
            model.num_experts_per_token) {
        throw std::invalid_argument(
            "HIP MoE route config does not match model contract");
    }
    if (request.logical_rows == 0 ||
        request.request_concurrency == 0) {
        throw std::invalid_argument(
            "HIP MoE route variant requires non-zero rows and concurrency");
    }
    if (qualification ==
            kernels::QualificationStatus::target_qualified ||
        qualification ==
            kernels::QualificationStatus::production ||
        qualification ==
            kernels::QualificationStatus::rejected) {
        throw std::invalid_argument(
            "unmeasured HIP MoE route may only be experimental or oracle-matched");
    }

    kernels::KernelVariant variant;
    variant.id =
        route_kernel_id(
            request,
            config);
    variant.family =
        kernels::OperatorFamily::moe_route;
    variant.phase =
        kernels::ExecutionPhase::decode;
    variant.shape =
        execution_shape_for(
            model,
            SerialOp::moe_route,
            request);
    variant.wave =
        kernels::WaveSize::wave32;
    variant.resources.workgroup_threads =
        kThreads;
    variant.resources.lds_bytes =
        config.staged_logits ? staged_router_logits_lds_bytes :
        config.parallel_projection ? kRouterDeclaredLdsBytes :
        (kMaxExperts + kThreads) * sizeof(float);
    variant.numerics.input_precision =
        activation_name(
            config.activation);
    variant.numerics.accumulator_precision =
        "fp32";
    variant.numerics.output_precision =
        "u32+fp32";
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        qualification;
    variant.graph_safe = true;
    variant.input_layout =
        "row-major+ggml-router";
    variant.output_layout =
        "topk-indices+fp32-weights";
    variant.compiler_identity = config.staged_logits ? "hip-gfx1151-moe-route-v7-staged-logits" : config.wave_topk ? "hip-gfx1151-moe-route-v6-wave-parallel" : config.ranked_topk ? "hip-gfx1151-moe-route-v5-ranked-parallel" : config.parallel_projection
        ? "hip-gfx1151-moe-route-v4-parallel" : "hip-gfx1151-moe-route-v2";
    variant.measured_seconds = 0.0;

    registry.add(
        std::move(variant));
}

ExecutionLauncher
make_hip_moe_route_launcher(
    std::string kernel_id,
    HipMoeRouteLauncherConfig* config) {
    if (kernel_id.empty() ||
        config == nullptr) {
        throw std::invalid_argument(
            "HIP MoE route launcher requires id and config");
    }
    validate_config(*config);
    if(config->wave_topk && !config->wave_manifest_admission)
        throw std::invalid_argument("wave router launcher requires owned manifest preflight admission");
    return ExecutionLauncher{
        std::move(kernel_id),
        launcher_identity(*config),
        &launch_moe_route,
        config,
        &preflight_moe_route,
    };
}

void HipMoeRouteLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry) {
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP MoE route bundle requires valid compiled execution");
    }

    // Check the entire router registration contract before publishing launchers.
    if (compiled.io.steps.size() != compiled.execution.steps.size() ||
        compiled.bindings.bindings.size() != compiled.execution.steps.size())
        throw std::invalid_argument("router launcher preflight requires complete compiled IO and bindings");
    if (compiled.io.router_workspace_slots != (config_.parallel_projection ? 1U : 0U))
        throw std::invalid_argument("router launcher mode differs from compiled workspace slots");
    for (std::size_t index = 0; index < compiled.execution.steps.size(); ++index) {
        if (compiled.execution.steps[index].op != SerialOp::moe_route) continue;
        const auto& binding = compiled.bindings.bindings[index];
        const auto& io = compiled.io.steps[index];
        if (!binding.kernel_id.has_value() ||
            ((binding.kernel_id->find(".staged_logits") != std::string::npos) != config_.staged_logits ||
             binding.kernel_id->ends_with(".parallel") != config_.parallel_projection ||
             binding.kernel_id->ends_with(".ranked.parallel") != config_.ranked_topk ||
             binding.kernel_id->ends_with(".wave.parallel") != config_.wave_topk))
            throw std::invalid_argument("router launcher mode differs from compiled kernel binding");
        if (io.step_id != compiled.execution.steps[index].id || io.op != SerialOp::moe_route ||
            io.temporaries.size() != (config_.parallel_projection ? 1U : 0U) ||
            (config_.parallel_projection && io.temporaries.front().buffer.buffer_class != ExecutionBufferClass::router_workspace))
            throw std::invalid_argument("router launcher mode differs from compiled temporary IO");
    }

    std::string registered_id;
    for (std::size_t index = 0;
         index <
            compiled.execution.steps.size();
         ++index) {
        const auto& step =
            compiled.execution.steps[index];
        if (step.op !=
            SerialOp::moe_route) {
            continue;
        }

        const auto& binding =
            compiled.bindings.bindings[index];
        if (binding.host_or_runtime_step ||
            !binding.kernel_id.has_value() ||
            binding.kernel_id->empty()) {
            throw std::invalid_argument(
                "HIP MoE route step has no kernel binding");
        }

        if (registered_id.empty()) {
            registered_id =
                *binding.kernel_id;
            registry.add_kernel_launcher(
                make_hip_moe_route_launcher(
                    registered_id,
                    &config_));
        } else if (registered_id !=
                   *binding.kernel_id) {
            throw std::invalid_argument(
                "HIP MoE route bundle currently requires one route kernel id for all serial layers");
        }
    }
}

void preflight_hip_mtp_route(const MtpNativeContext& c) {
    mtp_native::context(c);
    const auto h=c.model->hidden_size,e=c.model->num_experts,k=c.model->num_experts_per_token;
    if(!h || h>std::numeric_limits<std::uint32_t>::max() || !e || e>kMaxExperts || !k || k>e || k>kWaveMaxTopK)
        throw std::invalid_argument("MTP native router geometry unsupported");
    const auto& t=c.tensors->router;
    mtp_native::tensor(t,{h,e},mtp_native::product(mtp_native::product(h,e),4));
    if(t.manifest->ggml_type!=0) throw std::invalid_argument("MTP native router requires F32 weights");
    auto input=mtp_native::buffer(c,HipMtpBuffer::hidden0,mtp_native::product(h,2));
    auto indices=mtp_native::buffer(c,HipMtpBuffer::routes,mtp_native::product(k,4));
    auto weights=mtp_native::buffer(c,HipMtpBuffer::route_weights,mtp_native::product(k,4));
    auto workspace=mtp_native::buffer(c,HipMtpBuffer::router_workspace,mtp_native::product(e,4));
    mtp_native::disjoint({input,indices,weights,workspace,
        {t.runtime->address,t.runtime->bytes,t.runtime->allocation_generation}});
}
void launch_hip_mtp_route(const MtpNativeContext& c,void* stream) {
    preflight_hip_mtp_route(c); mtp_native::stream(c,stream);
    check_hip(hipGetLastError(),"MTP router pending HIP error");
    const auto h=static_cast<std::uint32_t>(c.model->hidden_size);
    const auto e=static_cast<std::uint32_t>(c.model->num_experts);
    const auto k=static_cast<std::uint32_t>(c.model->num_experts_per_token);
    auto workspace=reinterpret_cast<float*>(c.resources->view(HipMtpBuffer::router_workspace).address);
    const auto hidden0=c.resources->view(HipMtpBuffer::hidden0).address;
    // The one-row staged chains (same bits as parallel_moe_route_logits_kernel); STRIX_ROUTER_STAGED=0: off.
    static const bool staged=strix::envcfg::flag("STRIX_ROUTER_STAGED", true);
    const auto* bf16 = q6p::find(reinterpret_cast<const void*>(c.tensors->router.runtime->address));
    if (bf16 && bf16->kind == q6p::Kind::bf16 && bf16->rows == e && bf16->cols == h) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(parallel_moe_route_logits_kernel<std::uint16_t,RouterWeightKind::bf16>),
            dim3((e+kRouterExpertsPerBlock-1)/kRouterExpertsPerBlock),dim3(kRouterProjectionThreads),0,
            reinterpret_cast<hipStream_t>(stream),reinterpret_cast<const std::uint16_t*>(hidden0),
            bf16->base,h,e,workspace);
    } else if(staged && h==20U*128U && hidden0%16U==0U && c.tensors->router.runtime->address%16U==0U) {
        RouterRowsArgs args{};args.hidden[0]=reinterpret_cast<const void*>(hidden0);args.workspace[0]=workspace;
        static const bool stage_act=strix::envcfg::flag("STRIX_ROUTER_STAGE_ACT", true);   // default ON
        static const bool chunk256=strix::envcfg::flag("STRIX_ROUTER_CHUNK256", false);
        // STRIX_ROUTER_LOADERS16=1 (with STAGE_ACT=1): 16 loader waves, ring depth 16 (4 MiB lookahead).
        static const bool loaders16=strix::envcfg::flag("STRIX_ROUTER_LOADERS16", false);
        if(stage_act&&loaders16)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_rows_moe_route_logits_kernel<std::uint16_t,1U,20U,16U,false,16U,true>),dim3((e+15U)/16U),
            dim3(32U*(staged_router_chain_waves<1U>()+16U)),0,reinterpret_cast<hipStream_t>(stream),args,
            reinterpret_cast<const float*>(c.tensors->router.runtime->address),e);
        else if(stage_act&&chunk256)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_rows_moe_route_logits_kernel<std::uint16_t,1U,10U,8U,false,16U,true,256U>),dim3((e+15U)/16U),
            dim3(32U*(staged_router_chain_waves<1U>()+8U)),0,reinterpret_cast<hipStream_t>(stream),args,
            reinterpret_cast<const float*>(c.tensors->router.runtime->address),e);
        else if(stage_act)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_rows_moe_route_logits_kernel<std::uint16_t,1U,20U,8U,false,16U,true>),dim3((e+15U)/16U),
            dim3(32U*(staged_router_chain_waves<1U>()+8U)),0,reinterpret_cast<hipStream_t>(stream),args,
            reinterpret_cast<const float*>(c.tensors->router.runtime->address),e);
        else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_rows_moe_route_logits_kernel<std::uint16_t,1U,20U,8U>),dim3((e+15U)/16U),
            dim3(32U*(staged_router_chain_waves<1U>()+8U)),0,reinterpret_cast<hipStream_t>(stream),args,
            reinterpret_cast<const float*>(c.tensors->router.runtime->address),e);
    } else
    hipLaunchKernelGGL(HIP_KERNEL_NAME(parallel_moe_route_logits_kernel<std::uint16_t,RouterWeightKind::f32>),
        dim3((e+kRouterExpertsPerBlock-1)/kRouterExpertsPerBlock),dim3(kRouterProjectionThreads),0,
        reinterpret_cast<hipStream_t>(stream),
        reinterpret_cast<const std::uint16_t*>(hidden0),
        reinterpret_cast<const void*>(c.tensors->router.runtime->address),h,e,workspace);
    check_hip(hipGetLastError(),"MTP native parallel route projection launch");
    hipLaunchKernelGGL(wave_parallel_moe_route_select_kernel,dim3(1),dim3(kThreads),0,
        reinterpret_cast<hipStream_t>(stream),workspace,e,k,c.model->norm_topk_probability,
        reinterpret_cast<std::uint32_t*>(c.resources->view(HipMtpBuffer::routes).address),
        reinterpret_cast<float*>(c.resources->view(HipMtpBuffer::route_weights).address));
    check_hip(hipGetLastError(),"MTP native parallel route selection launch");
}
// Multi-session draft batch: launch_hip_mtp_route of every context on `stream`. The staged logits kernel exists
// for 1, 2, 4 and 8 rows (same ascending FMA chain per expert and row at every row count; rows never reduce
// across each other), so the live rows run as groups of 8, 4, 2 and 1; the selection runs once per row with
// the one-token kernel's row body (rows_moe_route_select_kernel<wave>). Only the staged production branch.
bool hip_mtp_route_rows_supported() {
    static const bool staged=strix::envcfg::flag("STRIX_ROUTER_STAGED", true);
    return staged;
}
void launch_hip_mtp_route_rows(std::span<const MtpNativeContext> contexts,void* raw_stream) {
    if(contexts.empty() || contexts.size()>kRouterMaxRows || !raw_stream)
        throw std::invalid_argument("MTP router batch size or stream is invalid");
    const bool staged=hip_mtp_route_rows_supported();
    const auto& c0=contexts.front();
    const auto h=static_cast<std::uint32_t>(c0.model->hidden_size);
    const auto e=static_cast<std::uint32_t>(c0.model->num_experts);
    const auto k=static_cast<std::uint32_t>(c0.model->num_experts_per_token);
    const auto* router=reinterpret_cast<const float*>(c0.tensors->router.runtime->address);
    RouterRowsArgs args{};
    for(std::size_t t=0;t<contexts.size();++t) {
        const auto& c=contexts[t];
        preflight_hip_mtp_route(c);
        const auto hidden0=c.resources->view(HipMtpBuffer::hidden0).address;
        if(!staged || h!=20U*128U || hidden0%16U || c.tensors->router.runtime->address%16U ||
           reinterpret_cast<const float*>(c.tensors->router.runtime->address)!=router ||
           c.model->hidden_size!=h || c.model->num_experts!=e || c.model->num_experts_per_token!=k ||
           c.model->norm_topk_probability!=c0.model->norm_topk_probability)
            throw std::invalid_argument("MTP router batch needs the staged branch and one shared router");
        args.hidden[t]=reinterpret_cast<const void*>(hidden0);
        args.workspace[t]=reinterpret_cast<float*>(c.resources->view(HipMtpBuffer::router_workspace).address);
        args.indices[t]=reinterpret_cast<std::uint32_t*>(c.resources->view(HipMtpBuffer::routes).address);
        args.weights[t]=reinterpret_cast<float*>(c.resources->view(HipMtpBuffer::route_weights).address);
    }
    const auto stream=reinterpret_cast<hipStream_t>(raw_stream);
    check_hip(hipGetLastError(),"MTP router batch pending HIP error");
    const auto* bf16 = q6p::find(router);
    const bool half = bf16 && bf16->kind == q6p::Kind::bf16 && bf16->rows == e && bf16->cols == h;
    std::uint32_t first=0;const auto count=static_cast<std::uint32_t>(contexts.size());
    const auto group=[&]<std::uint32_t R>() {
        RouterRowsArgs g{};
        for(std::uint32_t r=0;r<R;++r){g.hidden[r]=args.hidden[first+r];g.workspace[r]=args.workspace[first+r];}
        if (half) {
            if constexpr (R == 1U)
                hipLaunchKernelGGL(HIP_KERNEL_NAME(parallel_moe_route_logits_kernel<std::uint16_t,RouterWeightKind::bf16>),
                    dim3((e+kRouterExpertsPerBlock-1)/kRouterExpertsPerBlock),dim3(kRouterProjectionThreads),0,stream,
                    static_cast<const std::uint16_t*>(g.hidden[0]),bf16->base,h,e,g.workspace[0]);
            else
                hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_rows_moe_route_logits_kernel<std::uint16_t,R,20U,8U,true>),dim3((e+15U)/16U),
                    dim3(32U*(staged_router_chain_waves<R>()+8U)),0,stream,g,bf16->base,e);
        } else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_rows_moe_route_logits_kernel<std::uint16_t,R,20U,8U>),dim3((e+15U)/16U),
            dim3(32U*(staged_router_chain_waves<R>()+8U)),0,stream,g,router,e);
        check_hip(hipGetLastError(),"MTP router batch logits launch");
        first+=R;
    };
    while(first<count) {
        const auto left=count-first;
        if(left>=8U) group.template operator()<8U>();
        else if(left>=4U) group.template operator()<4U>();
        else if(left>=2U) group.template operator()<2U>();
        else group.template operator()<1U>();
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_moe_route_select_kernel<RouterSelect::wave>),dim3(count),dim3(kThreads),0,stream,
        args,e,k,c0.model->norm_topk_probability);
    check_hip(hipGetLastError(),"MTP router batch selection launch");
}

std::array<double, 4> router_prefill_select_selftest(std::uint32_t flavor, std::uint32_t tiles, std::uint32_t experts,
    std::uint32_t top_k, bool normalize, std::uint32_t seed, std::uint32_t bench_reps) {
    if (experts == 0U || experts > kMaxExperts || top_k == 0U || top_k > kRouterWave || top_k > experts || tiles == 0U || tiles > 65535U)
        throw std::invalid_argument("router select self-test: bad shape");
    const std::uint64_t rows = 8ULL * tiles, n = rows * experts;
    std::vector<std::uint32_t> host(n);
    std::uint64_t state = 0x9e3779b97f4a7c15ULL ^ (static_cast<std::uint64_t>(seed) << 17U) ^ flavor;
    const auto next = [&] { state ^= state << 13U; state ^= state >> 7U; state ^= state << 17U; return state; };
    const auto bits = [](float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; };
    const auto uniform = [&](float lo, float hi) { return lo + (hi - lo) * static_cast<float>(next() >> 40U) / 16777216.0F; };
    const std::uint32_t specials[] = {0x7fc00000U, 0xffc12345U, 0x7f800001U, 0x7f800000U, 0xff800000U, 0x00000000U, 0x80000000U,
        0x00000001U, 0x807fffffU, 0x7f7fffffU, 0xff7fffffU, 0x3f800000U};
    for (std::uint64_t r = 0; r < rows; ++r) {
        const std::uint32_t kind = flavor == 4U ? static_cast<std::uint32_t>(r % 5U) : 0U;
        for (std::uint32_t e = 0; e < experts; ++e) {
            float v = uniform(-8.0F, 8.0F);
            std::uint32_t u = bits(v);
            if (flavor == 1U) u = bits(static_cast<float>(static_cast<int>(next() % 7U)) * 0.5F - 1.0F);          // ties
            else if (flavor == 2U && next() % 64U == 0U) u = specials[next() % 3U];                              // NaN payloads
            else if (flavor == 3U && next() % 8U == 0U) u = specials[next() % 12U];                              // +-Inf, +-0, denormals, max
            else if (flavor == 4U) u = kind == 0U ? 0x7fc00000U : kind == 1U ? 0xff800000U : kind == 2U ? 0x40400000U :
                kind == 3U ? (e < top_k / 2U ? 0x7fc00000U : u) : (e % 3U == 0U ? 0x7f800000U : u);              // all NaN / -Inf / equal / few NaN / +Inf
            else if (flavor == 5U) u = bits(uniform(-120.0F, 120.0F));                                           // exp underflow ties at 0
            host[r * experts + e] = u;
        }
    }
    float* workspace = nullptr; std::uint32_t* indices[2] = {}; float* weights[2] = {};
    const std::uint64_t outs = rows * top_k;
    check_hip(hipMalloc(&workspace, n * 4U), "select self-test workspace");
    for (int a = 0; a < 2; ++a) {
        check_hip(hipMalloc(&indices[a], outs * 4U), "select self-test indices");
        check_hip(hipMalloc(&weights[a], outs * 4U), "select self-test weights");
        check_hip(hipMemset(indices[a], 0xab, outs * 4U), "select self-test poison");
        check_hip(hipMemset(weights[a], 0xcd, outs * 4U), "select self-test poison");
    }
    check_hip(hipMemcpy(workspace, host.data(), n * 4U, hipMemcpyHostToDevice), "select self-test upload");
    const auto make = [&](int a) {
        RouterTileRowsArgs args{};
        for (std::uint32_t r = 0; r < 8U; ++r) {
            args.hidden[r] = nullptr;
            args.workspace[r] = workspace + static_cast<std::uint64_t>(r) * experts;
            args.indices[r] = indices[a] + static_cast<std::uint64_t>(r) * top_k;
            args.weights[r] = weights[a] + static_cast<std::uint64_t>(r) * top_k;
        }
        args.tile_hidden = 0; args.tile_workspace = 32ULL * experts; args.tile_indices = 32ULL * top_k; args.tile_weights = 32ULL * top_k;
        return args;
    };
    const auto a0 = make(0), a1 = make(1);
    const auto old_form = [&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_moe_route_select_kernel<RouterSelect::wave, RouterTileRowsArgs>),
        dim3(8U, tiles), dim3(kThreads), 0, nullptr, a0, experts, top_k, normalize); };
    const auto new_form = [&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(wave_rows_route_select_kernel<RouterTileRowsArgs>),
        dim3(1U, tiles), dim3(kThreads), 0, nullptr, a1, experts, top_k, normalize, 8U); };
    old_form(); new_form();
    check_hip(hipGetLastError(), "select self-test launch");
    check_hip(hipDeviceSynchronize(), "select self-test sync");
    std::vector<std::uint32_t> i0(outs), i1(outs), w0(outs), w1(outs);
    check_hip(hipMemcpy(i0.data(), indices[0], outs * 4U, hipMemcpyDeviceToHost), "read");
    check_hip(hipMemcpy(i1.data(), indices[1], outs * 4U, hipMemcpyDeviceToHost), "read");
    check_hip(hipMemcpy(w0.data(), weights[0], outs * 4U, hipMemcpyDeviceToHost), "read");
    check_hip(hipMemcpy(w1.data(), weights[1], outs * 4U, hipMemcpyDeviceToHost), "read");
    double differing = 0;
    for (std::uint64_t r = 0; r < rows; ++r) {
        bool same = true;
        for (std::uint32_t k = 0; k < top_k; ++k) same = same && i0[r * top_k + k] == i1[r * top_k + k] && w0[r * top_k + k] == w1[r * top_k + k];
        differing += same ? 0.0 : 1.0;
    }
    double us[2] = {0.0, 0.0};
    if (bench_reps > 0U) {
        hipEvent_t start, stop;
        check_hip(hipEventCreate(&start), "event"); check_hip(hipEventCreate(&stop), "event");
        for (int round = 0; round < 3; ++round) for (int a = 0; a < 2; ++a) {   // A B A B A B: the last two rounds count
            check_hip(hipEventRecord(start, nullptr), "event");
            for (std::uint32_t k = 0; k < bench_reps; ++k) { if (a == 0) old_form(); else new_form(); }
            check_hip(hipEventRecord(stop, nullptr), "event");
            check_hip(hipEventSynchronize(stop), "event");
            float ms = 0.0F; check_hip(hipEventElapsedTime(&ms, start, stop), "event");
            if (round > 0) us[a] += 1000.0 * ms / bench_reps / 2.0;
        }
        check_hip(hipEventDestroy(start), "event"); check_hip(hipEventDestroy(stop), "event");
    }
    check_hip(hipFree(workspace), "free");
    for (int a = 0; a < 2; ++a) { check_hip(hipFree(indices[a]), "free"); check_hip(hipFree(weights[a]), "free"); }
    return {static_cast<double>(rows), differing, us[0], us[1]};
}

std::array<double, 4> router_staged_logits_selftest(std::uint32_t rows_count, bool float_activation, std::uint32_t flavor,
    std::uint32_t seed, std::uint32_t bench_reps) {
    constexpr std::uint32_t hidden = 2560U, experts = 512U;
    if (rows_count != 1U && rows_count != 2U && rows_count != 4U && rows_count != 8U) throw std::invalid_argument("staged self-test: rows");
    if (float_activation && rows_count > 2U) throw std::invalid_argument("staged self-test: float activations take <= 2 rows");
    if ((flavor & 4U) != 0U && rows_count > 4U) throw std::invalid_argument("staged self-test: the chunk-256 form takes <= 4 rows");
    if ((flavor & 8U) != 0U && rows_count != 1U) throw std::invalid_argument("staged self-test: the 16-loader form takes 1 row");
    std::uint64_t state = 0x2545f4914f6cdd1dULL ^ (static_cast<std::uint64_t>(seed) << 21U) ^ flavor;
    const auto next = [&] { state ^= state << 13U; state ^= state >> 7U; state ^= state << 17U; return state; };
    const auto uniform = [&] { return static_cast<float>(static_cast<double>(next() >> 40U) / 16777216.0 * 2.0 - 1.0); };
    const auto f2u = [](float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; };
    std::vector<float> w(static_cast<std::size_t>(experts) * hidden);
    for (auto& x : w) {
        x = uniform() * 0.05F;
        if ((flavor & 3U) == 1U && next() % 4096U == 0U) { const std::uint32_t u = (next() & 1U) ? 0x7fc00000U : 0x7f800000U; std::memcpy(&x, &u, 4); }
    }
    const std::size_t act_bytes = static_cast<std::size_t>(hidden) * (float_activation ? 4U : 2U);
    std::vector<unsigned char> act(act_bytes * rows_count);
    for (std::uint32_t r = 0; r < rows_count; ++r)
        for (std::uint32_t k = 0; k < hidden; ++k) {
            float v = uniform() * 4.0F;
            if ((flavor & 3U) == 1U && next() % 2048U == 0U) { const std::uint32_t u = 0xff800000U; std::memcpy(&v, &u, 4); }
            if (float_activation) std::memcpy(act.data() + r * act_bytes + 4U * k, &v, 4);
            else { const std::uint16_t h = static_cast<std::uint16_t>(f2u(v) >> 16U); std::memcpy(act.data() + r * act_bytes + 2U * k, &h, 2); }
        }
    float* weights = nullptr; unsigned char* hidden_rows = nullptr; float* out[2] = {};
    check_hip(hipMalloc(&weights, w.size() * 4U), "staged self-test weights");
    check_hip(hipMalloc(&hidden_rows, act.size()), "staged self-test rows");
    for (auto& o : out) {
        check_hip(hipMalloc(&o, 4ULL * experts * rows_count), "staged self-test out");
        check_hip(hipMemset(o, 0xa5, 4ULL * experts * rows_count), "poison");
    }
    check_hip(hipMemcpy(weights, w.data(), w.size() * 4U, hipMemcpyHostToDevice), "upload");
    check_hip(hipMemcpy(hidden_rows, act.data(), act.size(), hipMemcpyHostToDevice), "upload");
    const auto make = [&](int a) {
        RouterRowsArgs args{};
        for (std::uint32_t r = 0; r < rows_count; ++r) {
            args.hidden[r] = hidden_rows + r * act_bytes;
            args.workspace[r] = out[a] + static_cast<std::uint64_t>(r) * experts;
        }
        return args;
    };
    const auto a0 = make(0), a1 = make(1);
    const dim3 grid(experts / 16U);
    const auto run = [&]<typename A, std::uint32_t R>(bool staged) {
        const dim3 threads(32U * (staged_router_chain_waves<R>() + 8U));
        if (staged && (flavor & 8U) != 0U) {
            if constexpr (R == 1U) hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_rows_moe_route_logits_kernel<A, R, 20U, 16U, false, 16U, true>), grid,
                dim3(32U * (staged_router_chain_waves<R>() + 16U)), 0, nullptr, a1, weights, experts);
        } else if (staged && (flavor & 4U) != 0U) {
            if constexpr (R <= 4U) hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_rows_moe_route_logits_kernel<A, R, 10U, 8U, false, 16U, true, 256U>), grid, threads, 0, nullptr, a1, weights, experts);
        } else if (staged) hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_rows_moe_route_logits_kernel<A, R, 20U, 8U, false, 16U, true>), grid, threads, 0, nullptr, a1, weights, experts);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_rows_moe_route_logits_kernel<A, R, 20U, 8U>), grid, threads, 0, nullptr, a0, weights, experts);
    };
    const auto both = [&](bool staged) {
        if (float_activation) { if (rows_count == 1U) run.template operator()<float, 1U>(staged); else run.template operator()<float, 2U>(staged); }
        else if (rows_count == 1U) run.template operator()<std::uint16_t, 1U>(staged);
        else if (rows_count == 2U) run.template operator()<std::uint16_t, 2U>(staged);
        else if (rows_count == 4U) run.template operator()<std::uint16_t, 4U>(staged);
        else run.template operator()<std::uint16_t, 8U>(staged);
    };
    both(false); both(true);
    check_hip(hipGetLastError(), "staged self-test launch");
    check_hip(hipDeviceSynchronize(), "sync");
    std::vector<std::uint32_t> o0(experts * rows_count), o1(experts * rows_count);
    check_hip(hipMemcpy(o0.data(), out[0], o0.size() * 4U, hipMemcpyDeviceToHost), "read");
    check_hip(hipMemcpy(o1.data(), out[1], o1.size() * 4U, hipMemcpyDeviceToHost), "read");
    double differing = 0;
    for (std::size_t i = 0; i < o0.size(); ++i) differing += o0[i] != o1[i] ? 1.0 : 0.0;
    double us[2] = {0.0, 0.0};
    if (bench_reps > 0U) {
        hipEvent_t start, stop;
        check_hip(hipEventCreate(&start), "event"); check_hip(hipEventCreate(&stop), "event");
        for (int round = 0; round < 3; ++round) for (int a = 0; a < 2; ++a) {
            check_hip(hipEventRecord(start, nullptr), "event");
            for (std::uint32_t k = 0; k < bench_reps; ++k) both(a == 1);
            check_hip(hipEventRecord(stop, nullptr), "event"); check_hip(hipEventSynchronize(stop), "event");
            float ms = 0.0F; check_hip(hipEventElapsedTime(&ms, start, stop), "event");
            if (round > 0) us[a] += 1000.0 * ms / bench_reps / 2.0;
        }
        check_hip(hipEventDestroy(start), "event"); check_hip(hipEventDestroy(stop), "event");
    }
    check_hip(hipFree(weights), "free"); check_hip(hipFree(hidden_rows), "free");
    for (auto& o : out) check_hip(hipFree(o), "free");
    return {static_cast<double>(o0.size()), differing, us[0], us[1]};
}

}  // namespace strix
