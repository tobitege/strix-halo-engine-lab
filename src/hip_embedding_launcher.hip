#include "strix/hip_mtp_prime_rows.hpp"
#include <type_traits>
#include "strix/hip_mtp_native_validation.hpp"
#include "strix/hip_ggml_weight_formats.hpp"
#include "strix/hip_embedding_launcher.hpp"
#include "strix/hip_launch_context.hpp"

#include "strix/dispatch_resources.hpp"
#include "strix/rows_launch_registry.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <span>

#include <algorithm>
#include <array>
#include <span>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix {
namespace {

constexpr std::uint32_t kThreads = 256;
constexpr std::uint64_t kLauncherTag =
    0x454d42454444494eULL;
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

static_assert(sizeof(Q8Block) == 34);
static_assert(sizeof(IQ4XSBlock) == 136);

__device__ __constant__
std::int8_t iq4_codebook[16] = {
    -127, -104, -83, -65,
    -49, -35, -22, -10,
       1,   13,  25,  38,
      53,   69,  89, 113,
};

enum class EmbeddingWeightKind {
    f32,
    f16,
    bf16,
    q8_0,
    iq4_xs,
#define STRIX_GGML_KIND_ENUM(kind, type) kind,
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_ENUM) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_ENUM) STRIX_GGML_KIND_ENUM(iq4_nl, 20)
#undef STRIX_GGML_KIND_ENUM
};

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^=
            (value >> shift) &
            0xffULL;
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
    const HipEmbeddingLauncherConfig& config) {
    if (config.hidden_size == 0) {
        throw std::invalid_argument(
            "HIP embedding hidden size must be non-zero");
    }
    if (config.output !=
            HipActivationStorage::bf16 &&
        config.output !=
            HipActivationStorage::fp32) {
        throw std::invalid_argument(
            "HIP embedding output storage is unsupported");
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

EmbeddingWeightKind weight_kind(
    std::uint32_t ggml_type) {
    switch (ggml_type) {
    case 0:
        return EmbeddingWeightKind::f32;
    case 1:
        return EmbeddingWeightKind::f16;
    case 8:
        return EmbeddingWeightKind::q8_0;
    case 23:
        return EmbeddingWeightKind::iq4_xs;
#define STRIX_GGML_KIND_MAP(kind, type) case type: return EmbeddingWeightKind::kind;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_MAP) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_MAP) STRIX_GGML_KIND_MAP(iq4_nl, 20)
#undef STRIX_GGML_KIND_MAP
    case 30:
        return EmbeddingWeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP embedding currently supports only F32, F16, BF16, Q8_0 and IQ4_XS");
    }
}

__device__ __forceinline__ float bf16_to_float(
    std::uint16_t bits) {
    return __uint_as_float(
        static_cast<std::uint32_t>(
            bits) << 16U);
}

__device__ __forceinline__ void store_output(
    float* destination,
    float value) {
    *destination = value;
}

__device__ __forceinline__ void store_output(
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

__device__ __forceinline__ std::uint8_t iq4_local_scale(
    const IQ4XSBlock& block,
    std::uint32_t subblock) {
    const auto low_byte =
        block.scales_l[
            subblock / 2U];
    const auto low =
        static_cast<std::uint8_t>(
            (low_byte >>
             (4U *
              (subblock % 2U))) &
            0x0fU);
    const auto high =
        static_cast<std::uint8_t>(
            (block.scales_h >>
             (2U * subblock)) &
            0x03U);
    return
        static_cast<std::uint8_t>(
            low |
            static_cast<std::uint8_t>(
                high << 4U));
}

template <EmbeddingWeightKind Kind>
__device__ __forceinline__ float load_weight(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature);

template <>
__device__ __forceinline__ float load_weight<
    EmbeddingWeightKind::f32>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const float*>(
            weights);
    return values[
        static_cast<std::uint64_t>(
            row) *
            row_width +
        feature];
}

template <>
__device__ __forceinline__ float load_weight<
    EmbeddingWeightKind::f16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const __half*>(
            weights);
    return __half2float(
        values[
            static_cast<std::uint64_t>(
                row) *
                row_width +
            feature]);
}

template <>
__device__ __forceinline__ float load_weight<
    EmbeddingWeightKind::bf16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const std::uint16_t*>(
            weights);
    return bf16_to_float(
        values[
            static_cast<std::uint64_t>(
                row) *
                row_width +
            feature]);
}

template <>
__device__ __forceinline__ float load_weight<
    EmbeddingWeightKind::q8_0>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const Q8Block*>(
            weights);
    const auto blocks_per_row =
        row_width / 32U;
    const auto& block =
        blocks[
            static_cast<std::uint64_t>(
                row) *
                blocks_per_row +
            feature / 32U];
    return __half2float(
               block.scale) *
           static_cast<float>(
               block.values[
                   feature % 32U]);
}

template <>
__device__ __forceinline__ float load_weight<
    EmbeddingWeightKind::iq4_xs>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const IQ4XSBlock*>(
            weights);
    const auto blocks_per_row =
        row_width / 256U;
    const auto& block =
        blocks[
            static_cast<std::uint64_t>(
                row) *
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
            (suboffset % 16U)];
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

    return
        __half2float(block.d) *
        static_cast<float>(local) *
        static_cast<float>(
            iq4_codebook[code]);
}

#define STRIX_GGML_KIND_LOAD(kind, type)                                                          \
    template <>                                                                                   \
    __device__ __forceinline__ float load_weight<EmbeddingWeightKind::kind>(                                               \
        const void* weights, std::uint32_t row_width, std::uint32_t row, std::uint32_t feature) { \
        return hip_ggml::element<type>(weights, row_width, row, feature);                        \
    }
STRIX_GGML_KINDS_K(STRIX_GGML_KIND_LOAD) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_LOAD) STRIX_GGML_KIND_LOAD(iq4_nl, 20)
#undef STRIX_GGML_KIND_LOAD

template <
    typename Output,
    EmbeddingWeightKind Kind>
__global__ void embedding_kernel(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t token_id,
    Output* output) {
    for (std::uint64_t feature =
             static_cast<std::uint64_t>(
                 blockIdx.x) *
                 blockDim.x +
             threadIdx.x;
         feature < row_width;
         feature +=
             static_cast<std::uint64_t>(
                 gridDim.x) *
             blockDim.x) {
        store_output(
            &output[feature],
            load_weight<Kind>(
                weights,
                row_width,
                token_id,
                static_cast<std::uint32_t>(feature)));
    }
}

// Rows of a tiled prefill block in one launch (STRIX_PREFILL_PIPE bit 16): blockIdx.y selects the row, whose
// output starts pitch elements after the previous one; each element is embedding_kernel's.
constexpr std::uint32_t kEmbeddingRowsMax = 256;
struct EmbeddingRowsTokens {
    std::uint32_t id[kEmbeddingRowsMax];
};
template <typename Output, EmbeddingWeightKind Kind>
__global__ void embedding_rows_kernel(const void* weights, std::uint32_t row_width, EmbeddingRowsTokens tokens,
    Output* output, std::uint64_t pitch) {
    Output* row = output + static_cast<std::uint64_t>(blockIdx.y) * pitch;
    for (std::uint64_t feature = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; feature < row_width;
         feature += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        store_output(&row[feature], load_weight<Kind>(weights, row_width, tokens.id[blockIdx.y], static_cast<std::uint32_t>(feature)));
}

template <typename Output, EmbeddingWeightKind Kind>
__global__ void device_token_embedding_kernel(
    const void* weights, std::uint32_t row_width, std::uint32_t vocabulary_size,
    const std::uint32_t* token, Output* output) {
    const auto id = *token < vocabulary_size ? *token : 0U;
    for (std::uint64_t feature = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         feature < row_width; feature += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        store_output(&output[feature], load_weight<Kind>(weights, row_width, id, static_cast<std::uint32_t>(feature)));
}

// Multi-session draft batch: row blockIdx.y loads its token's row as the one-token kernels do (a device token
// word with its out-of-vocabulary row-0 rule, or a host token already admitted by preflight).
struct MtpEmbeddingRows { const std::uint32_t* token[8]; std::uint32_t value[8]; std::uint16_t* output[8]; };
template <typename Output, EmbeddingWeightKind Kind>
__device__ __forceinline__ void mtp_embedding_row_body(const void* weights,std::uint32_t row_width,
    std::uint32_t id,Output* output) {
    for (std::uint64_t feature = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         feature < row_width; feature += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        store_output(&output[feature], load_weight<Kind>(weights, row_width, id, static_cast<std::uint32_t>(feature)));
}
template <typename Output, EmbeddingWeightKind Kind>
__global__ void mtp_embedding_rows_kernel(const void* weights, std::uint32_t row_width, std::uint32_t vocabulary_size,
    MtpEmbeddingRows rows) {
    const auto r = blockIdx.y;
    const auto id = rows.token[r] ? (*rows.token[r] < vocabulary_size ? *rows.token[r] : 0U) : rows.value[r];
    mtp_embedding_row_body<Output,Kind>(weights,row_width,id,rows.output[r]);
}
template <EmbeddingWeightKind Kind>
__global__ void mtp_embedding_prime_rows_kernel(const void* weights,std::uint32_t row_width,std::uint32_t,
    HipMtpEmbeddingPrimeRows rows) {
    mtp_embedding_row_body<std::uint16_t,Kind>(weights,row_width,rows.value[blockIdx.y],
        reinterpret_cast<std::uint16_t*>(rows.output[blockIdx.y]));
}

template <typename Output, EmbeddingWeightKind Kind>
__global__ void graph_embedding_kernel(
    const void* weights, std::uint32_t width, std::uint32_t vocabulary_size,
    std::uint64_t context_tokens, const std::uint32_t* token,
    const std::uint64_t* position, Output* output, std::uint32_t* status) {
    const auto token_id = *token;
    const auto absolute_position = *position;
    const std::uint32_t flags = (token_id >= vocabulary_size ? 1U : 0U) |
        (absolute_position >= context_tokens ? 2U : 0U);
    if (blockIdx.x == 0 && threadIdx.x == 0) *status = flags;
    for (std::uint64_t feature = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         feature < width; feature += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        store_output(&output[feature], flags == 0 ? load_weight<Kind>(weights, width, token_id,
            static_cast<std::uint32_t>(feature)) : 0.0F);
    }
}

template <typename Output, EmbeddingWeightKind Kind>
void launch_graph_typed(
    const HipEmbeddingLauncherConfig& config, const RuntimeTensorBinding& weights,
    std::uint32_t vocabulary_size, std::uint64_t context_tokens,
    const GraphDynamicArguments& arguments, const ExecutionBufferAddress& output,
    const ExecutionMemoryBinding& status, hipStream_t stream) {
    const auto blocks = std::min<std::uint32_t>(32U, (config.hidden_size - 1U) / kThreads + 1U);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(graph_embedding_kernel<Output, Kind>), dim3(blocks),
        dim3(kThreads), 0, stream, reinterpret_cast<const void*>(weights.address), config.hidden_size,
        vocabulary_size, context_tokens,
        reinterpret_cast<const std::uint32_t*>(arguments.memory.arena_base + arguments.token_offset),
        reinterpret_cast<const std::uint64_t*>(arguments.memory.arena_base + arguments.position_offset),
        reinterpret_cast<Output*>(output.address), reinterpret_cast<std::uint32_t*>(status.arena_base));
    check_hip(hipGetLastError(), "HIP device-input graph embedding launch");
}

template <typename Output>
void launch_graph_by_kind(
    EmbeddingWeightKind kind, const HipEmbeddingLauncherConfig& config,
    const RuntimeTensorBinding& weights, std::uint32_t vocabulary_size,
    std::uint64_t context_tokens, const GraphDynamicArguments& arguments,
    const ExecutionBufferAddress& output, const ExecutionMemoryBinding& status, hipStream_t stream) {
    switch (kind) {
    case EmbeddingWeightKind::f32:
        launch_graph_typed<Output, EmbeddingWeightKind::f32>(config, weights, vocabulary_size, context_tokens, arguments, output, status, stream); break;
    case EmbeddingWeightKind::f16:
        launch_graph_typed<Output, EmbeddingWeightKind::f16>(config, weights, vocabulary_size, context_tokens, arguments, output, status, stream); break;
    case EmbeddingWeightKind::bf16:
        launch_graph_typed<Output, EmbeddingWeightKind::bf16>(config, weights, vocabulary_size, context_tokens, arguments, output, status, stream); break;
    case EmbeddingWeightKind::q8_0:
        launch_graph_typed<Output, EmbeddingWeightKind::q8_0>(config, weights, vocabulary_size, context_tokens, arguments, output, status, stream); break;
    case EmbeddingWeightKind::iq4_xs:
        launch_graph_typed<Output, EmbeddingWeightKind::iq4_xs>(config, weights, vocabulary_size, context_tokens, arguments, output, status, stream); break;
#define STRIX_GGML_KIND_CASE(KIND, type) \
    case EmbeddingWeightKind::KIND: \
        launch_graph_typed<Output, EmbeddingWeightKind::KIND>(config, weights, vocabulary_size, context_tokens, arguments, output, status, stream); break;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_CASE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_CASE) STRIX_GGML_KIND_CASE(iq4_nl, 20)
#undef STRIX_GGML_KIND_CASE
    }
}


std::uint32_t ggml_type_of(EmbeddingWeightKind kind) {
    switch (kind) {
#define STRIX_GGML_KIND_TYPE(k, type) case EmbeddingWeightKind::k: return type;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_TYPE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_TYPE) STRIX_GGML_KIND_TYPE(iq4_nl, 20)
#undef STRIX_GGML_KIND_TYPE
    default: return 0;
    }
}

std::uint64_t required_weight_bytes(
    EmbeddingWeightKind kind,
    std::uint32_t row_width,
    std::uint64_t rows) {
    switch (kind) {
    case EmbeddingWeightKind::f32:
        return
            static_cast<std::uint64_t>(
                row_width) *
            rows *
            4ULL;
    case EmbeddingWeightKind::f16:
    case EmbeddingWeightKind::bf16:
        return
            static_cast<std::uint64_t>(
                row_width) *
            rows *
            2ULL;
    case EmbeddingWeightKind::q8_0:
        if (row_width % 32U != 0) {
            throw std::invalid_argument(
                "Q8_0 embedding row width must be divisible by 32");
        }
        return
            static_cast<std::uint64_t>(
                row_width / 32U) *
            rows *
            sizeof(Q8Block);
    case EmbeddingWeightKind::iq4_xs:
        if (row_width % 256U != 0) {
            throw std::invalid_argument(
                "IQ4_XS embedding row width must be divisible by 256");
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
        "unreachable embedding weight kind");
}

template <
    typename Output,
    EmbeddingWeightKind Kind>
void launch_typed(
    const RuntimeTensorBinding& weight,
    const HipEmbeddingLauncherConfig& config,
    std::uint32_t token_id,
    const ExecutionBufferAddress& output) {
    const auto blocks =
        std::min<std::uint32_t>(
            32U,
            (config.hidden_size - 1U) / kThreads + 1U);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            embedding_kernel<
                Output,
                Kind>),
        dim3(blocks),
        dim3(kThreads),
        0,
        hip_execution_stream(),
        reinterpret_cast<const void*>(
            weight.address),
        config.hidden_size,
        token_id,
        reinterpret_cast<Output*>(
            output.address));
    check_hip(
        hipGetLastError(),
        "HIP embedding kernel launch");
}

template <typename Output>
void launch_by_weight(
    EmbeddingWeightKind kind,
    const RuntimeTensorBinding& weight,
    const HipEmbeddingLauncherConfig& config,
    std::uint32_t token_id,
    const ExecutionBufferAddress& output) {
    switch (kind) {
    case EmbeddingWeightKind::f32:
        launch_typed<
            Output,
            EmbeddingWeightKind::f32>(
                weight,
                config,
                token_id,
                output);
        return;
    case EmbeddingWeightKind::f16:
        launch_typed<
            Output,
            EmbeddingWeightKind::f16>(
                weight,
                config,
                token_id,
                output);
        return;
    case EmbeddingWeightKind::bf16:
        launch_typed<
            Output,
            EmbeddingWeightKind::bf16>(
                weight,
                config,
                token_id,
                output);
        return;
    case EmbeddingWeightKind::q8_0:
        launch_typed<
            Output,
            EmbeddingWeightKind::q8_0>(
                weight,
                config,
                token_id,
                output);
        return;
    case EmbeddingWeightKind::iq4_xs:
        launch_typed<
            Output,
            EmbeddingWeightKind::iq4_xs>(
                weight,
                config,
                token_id,
                output);
        return;
#define STRIX_GGML_KIND_CASE(KIND, type) \
    case EmbeddingWeightKind::KIND: \
        launch_typed< \
            Output, \
            EmbeddingWeightKind::KIND>( \
                weight, \
                config, \
                token_id, \
                output); \
        return;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_CASE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_CASE) STRIX_GGML_KIND_CASE(iq4_nl, 20)
#undef STRIX_GGML_KIND_CASE
    }
    throw std::logic_error(
        "unreachable embedding weight kind");
}

struct PreparedEmbedding {
    const HipEmbeddingLauncherConfig& config;
    ExecutionBufferAddress output;
    const ManifestTensor& manifest_tensor;
    const RuntimeTensorBinding& runtime_tensor;
    EmbeddingWeightKind kind;
};
PreparedEmbedding prepare_embedding(
    const ExecutionDispatchContext& context,
    void* user_data) {
    if (context.op !=
        SerialOp::token_embedding) {
        throw std::invalid_argument(
            "HIP embedding launcher attached to wrong operator");
    }
    if (!context.reads.empty() ||
        context.writes.size() != 1) {
        throw std::invalid_argument(
            "HIP embedding expects no scratch reads and one write");
    }
    if (context.logical_rows != 1 ||
        context.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP embedding v1 supports only C1 one-row execution");
    }
    if (user_data == nullptr) {
        throw std::invalid_argument(
            "HIP embedding launcher requires configuration");
    }

    const auto& config =
        *static_cast<
            HipEmbeddingLauncherConfig*>(
                user_data);
    validate_config(config);

    const auto output =
        resolve_dispatch_write(
            context, 0);
    const auto& access =
        resolve_dispatch_tensor_access(
            context, 0);
    if (access.tensor_indices.size() != 1U) {
        throw std::invalid_argument(
            "HIP embedding requires exactly one embedding tensor");
    }
    if (context.resources == nullptr ||
        context.resources->model_manifest ==
            nullptr) {
        throw std::invalid_argument(
            "HIP embedding requires model manifest resources");
    }

    const auto tensor_index =
        access.tensor_indices.front();
    if (tensor_index >=
        context.resources->
            model_manifest->
            tensors.size()) {
        throw std::out_of_range(
            "HIP embedding tensor index exceeds manifest");
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
            TensorRole::embedding ||
        manifest_tensor.dimensions.size() < 2 ||
        manifest_tensor.dimensions[0] !=
            config.hidden_size) {
        throw std::invalid_argument(
            "HIP embedding tensor geometry does not match contract");
    }
    if (context.input_token_id >=
        manifest_tensor.dimensions[1]) {
        throw std::out_of_range(
            "HIP embedding token id exceeds embedding rows");
    }
    if (!runtime_tensor_gpu_visible(runtime_tensor.storage) ||
        runtime_tensor.address == 0) {
        throw std::invalid_argument(
            "HIP embedding requires resident or repacked GPU-visible weights");
    }

    const auto kind =
        weight_kind(
            manifest_tensor.ggml_type);
    const auto required =
        required_weight_bytes(
            kind,
            config.hidden_size,
            manifest_tensor.dimensions[1]);
    if (runtime_tensor.bytes < required) {
        throw std::invalid_argument(
            "HIP embedding runtime weight binding is undersized");
    }

    const auto output_bytes =
        static_cast<std::uint64_t>(
            config.hidden_size) *
        static_cast<std::uint32_t>(
            config.output);
    if (output.bytes <
        output_bytes) {
        throw std::invalid_argument(
            "HIP embedding output buffer is undersized");
    }

    return {config, output, manifest_tensor, runtime_tensor, kind};
}

void preflight_embedding(const ExecutionDispatchContext& context, void* user_data) {
    if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture)
        throw std::invalid_argument("known-window launcher preflight requires eager scope");
    static_cast<void>(prepare_embedding(context, user_data));
}

void launch_embedding(
    const ExecutionDispatchContext& context,
    void* user_data) {
    const auto prepared = prepare_embedding(context, user_data);
    const auto& config = prepared.config;
    const auto& output = prepared.output;
    const auto& manifest_tensor = prepared.manifest_tensor;
    const auto& runtime_tensor = prepared.runtime_tensor;
    const auto kind = prepared.kind;

    if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture) {
        if (!config.enable_device_graph_inputs || manifest_tensor.dimensions[1] != scope->vocabulary_size) {
            throw std::invalid_argument("embedding graph mode requires an exact opt-in vocabulary");
        }
        launch_hip_graph_embedding(config, manifest_tensor.ggml_type, runtime_tensor,
            scope->vocabulary_size, scope->context_tokens, scope->arguments, output,
            scope->status, scope->stream);
        return;
    }

    switch (config.output) {
    case HipActivationStorage::bf16:
        launch_by_weight<
            std::uint16_t>(
                kind,
                runtime_tensor,
                config,
                context.input_token_id,
                output);
        return;
    case HipActivationStorage::fp32:
        launch_by_weight<float>(
            kind,
            runtime_tensor,
            config,
            context.input_token_id,
            output);
        return;
    }
    throw std::logic_error(
        "unreachable embedding output storage");
}

template <typename Output, EmbeddingWeightKind Kind>
void launch_rows_typed(const RuntimeTensorBinding& weight, const HipEmbeddingLauncherConfig& config,
    std::span<const ExecutionDispatchContext> rows, std::uintptr_t output, std::uint64_t pitch) {
    const auto blocks = std::min<std::uint32_t>(32U, (config.hidden_size - 1U) / kThreads + 1U);
    for (std::size_t first = 0; first < rows.size(); first += kEmbeddingRowsMax) {
        const auto count = static_cast<std::uint32_t>(std::min<std::size_t>(kEmbeddingRowsMax, rows.size() - first));
        EmbeddingRowsTokens tokens{};
        for (std::uint32_t r = 0; r < count; ++r) tokens.id[r] = rows[first + r].input_token_id;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(embedding_rows_kernel<Output, Kind>), dim3(blocks, count), dim3(kThreads), 0,
            hip_execution_stream(), reinterpret_cast<const void*>(weight.address), config.hidden_size, tokens,
            reinterpret_cast<Output*>(output) + first * pitch, pitch);
        check_hip(hipGetLastError(), "HIP embedding rows kernel launch");
    }
}
template <typename Output>
void launch_rows_by_weight(EmbeddingWeightKind kind, const RuntimeTensorBinding& weight, const HipEmbeddingLauncherConfig& config,
    std::span<const ExecutionDispatchContext> rows, std::uintptr_t output, std::uint64_t pitch) {
    switch (kind) {
    case EmbeddingWeightKind::f32: launch_rows_typed<Output, EmbeddingWeightKind::f32>(weight, config, rows, output, pitch); return;
    case EmbeddingWeightKind::f16: launch_rows_typed<Output, EmbeddingWeightKind::f16>(weight, config, rows, output, pitch); return;
    case EmbeddingWeightKind::bf16: launch_rows_typed<Output, EmbeddingWeightKind::bf16>(weight, config, rows, output, pitch); return;
    case EmbeddingWeightKind::q8_0: launch_rows_typed<Output, EmbeddingWeightKind::q8_0>(weight, config, rows, output, pitch); return;
    case EmbeddingWeightKind::iq4_xs: launch_rows_typed<Output, EmbeddingWeightKind::iq4_xs>(weight, config, rows, output, pitch); return;
#define STRIX_GGML_KIND_CASE(KIND, type)     case EmbeddingWeightKind::KIND: launch_rows_typed<Output, EmbeddingWeightKind::KIND>(weight, config, rows, output, pitch); return;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_CASE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_CASE) STRIX_GGML_KIND_CASE(iq4_nl, 20)
#undef STRIX_GGML_KIND_CASE
    }
    throw std::logic_error("unreachable embedding weight kind");
}
// Every row is admitted as launch_embedding admits it; false (nothing enqueued) unless all rows read one table
// and their outputs sit at one pitch.
bool launch_embedding_rows(std::span<const ExecutionDispatchContext> rows, void* user_data) {
    if (rows.size() < 2U) return false;
    if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture) return false;
    const auto first = prepare_embedding(rows[0], user_data);
    const auto element = static_cast<std::uint64_t>(static_cast<std::uint32_t>(first.config.output));
    const auto second = prepare_embedding(rows[1], user_data);
    if (second.output.address <= first.output.address) return false;
    const auto pitch_bytes = static_cast<std::uint64_t>(second.output.address - first.output.address);
    if (pitch_bytes % element != 0U || pitch_bytes < static_cast<std::uint64_t>(first.config.hidden_size) * element) return false;
    for (std::size_t r = 1; r < rows.size(); ++r) {
        const auto row = r == 1U ? second : prepare_embedding(rows[r], user_data);
        if (row.output.address != first.output.address + r * pitch_bytes || row.runtime_tensor.address != first.runtime_tensor.address ||
            row.kind != first.kind || &row.config != &first.config)
            return false;
    }
    switch (first.config.output) {
    case HipActivationStorage::bf16:
        launch_rows_by_weight<std::uint16_t>(first.kind, first.runtime_tensor, first.config, rows, first.output.address, pitch_bytes / element);
        return true;
    case HipActivationStorage::fp32:
        launch_rows_by_weight<float>(first.kind, first.runtime_tensor, first.config, rows, first.output.address, pitch_bytes / element);
        return true;
    }
    throw std::logic_error("unreachable embedding output storage");
}
[[maybe_unused]] const bool embedding_rows_registered = register_rows_launch(&launch_embedding, &launch_embedding_rows);

std::string kernel_id(
    const ExecutionBindingRequest& request,
    const HipEmbeddingLauncherConfig& config) {
    return
        "hip.embedding." +
        std::string(
            storage_name(
                config.output)) +
        ".h" +
        std::to_string(
            config.hidden_size) +
        ".r" +
        std::to_string(
            request.logical_rows) +
        ".c" +
        std::to_string(
            request.request_concurrency) + (config.enable_device_graph_inputs ? ".device-graph" : "");
}

std::uint64_t implementation_identity(
    const HipEmbeddingLauncherConfig& config) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, kLauncherTag);
    hash_u64(hash, config.hidden_size);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            config.output));
    hash_u64(hash, config.enable_device_graph_inputs ? 1 : 0);
    return hash;
}

}  // namespace

void register_hip_embedding_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipEmbeddingLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification) {
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    validate_config(config);
    if (config.hidden_size !=
        model.hidden_size) {
        throw std::invalid_argument(
            "HIP embedding config does not match model hidden size");
    }
    if (request.logical_rows != 1 ||
        request.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP embedding v1 registers only C1 one-row variants");
    }
    if (qualification ==
            kernels::QualificationStatus::target_qualified ||
        qualification ==
            kernels::QualificationStatus::production ||
        qualification ==
            kernels::QualificationStatus::rejected) {
        throw std::invalid_argument(
            "unmeasured HIP embedding may only be experimental or oracle-matched");
    }

    kernels::KernelVariant variant;
    variant.id =
        kernel_id(
            request,
            config);
    variant.family =
        kernels::OperatorFamily::
            embedding_lookup;
    variant.phase =
        kernels::ExecutionPhase::decode;
    variant.shape =
        execution_shape_for(
            model,
            SerialOp::token_embedding,
            request);
    variant.wave =
        kernels::WaveSize::wave32;
    variant.resources.workgroup_threads =
        kThreads;
    variant.numerics.input_precision =
        "ggml-packed";
    variant.numerics.accumulator_precision =
        "fp32";
    variant.numerics.output_precision =
        storage_name(
            config.output);
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        qualification;

    // token_id is currently a by-value launch parameter. Do not mark this
    // launcher graph-safe until graph parameter update/stable token storage is
    // part of the runtime contract.
    variant.graph_safe = config.enable_device_graph_inputs;
    variant.input_layout =
        "ggml-row";
    variant.output_layout =
        "row-major";
    variant.compiler_identity =
        "hip-gfx1151-embedding-v1";
    variant.measured_seconds = 0.0;

    registry.add(
        std::move(variant));
}

ExecutionLauncher
make_hip_embedding_launcher(
    std::string kernel_id_value,
    HipEmbeddingLauncherConfig* config) {
    if (kernel_id_value.empty() ||
        config == nullptr) {
        throw std::invalid_argument(
            "HIP embedding launcher requires id and config");
    }
    validate_config(*config);
    return ExecutionLauncher{
        std::move(kernel_id_value),
        implementation_identity(
            *config),
        &launch_embedding,
        config,
        &preflight_embedding,
    };
}

void HipEmbeddingLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry) {
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP embedding bundle requires valid compiled execution");
    }

    for (std::size_t index = 0;
         index <
            compiled.execution.steps.size();
         ++index) {
        if (compiled.execution.steps[index].op !=
            SerialOp::token_embedding) {
            continue;
        }

        const auto& binding =
            compiled.bindings.bindings[index];
        if (binding.host_or_runtime_step ||
            !binding.kernel_id.has_value() ||
            binding.kernel_id->empty()) {
            throw std::invalid_argument(
                "HIP embedding step has no kernel binding");
        }

        registry.add_kernel_launcher(
            make_hip_embedding_launcher(
                *binding.kernel_id,
                &config_));
        return;
    }

    throw std::invalid_argument(
        "compiled execution has no token embedding step");
}

void launch_hip_graph_embedding(
    const HipEmbeddingLauncherConfig& config, std::uint32_t ggml_type,
    const RuntimeTensorBinding& weights, std::uint32_t vocabulary_size,
    std::uint64_t context_tokens, const GraphDynamicArguments& arguments,
    const ExecutionBufferAddress& output, const ExecutionMemoryBinding& status, void* stream) {
    validate_config(config);
    if (stream == nullptr) throw std::invalid_argument("graph embedding requires an explicitly owned stream");
    validate_graph_embedding_bindings(config.hidden_size, ggml_type, vocabulary_size, context_tokens,
        config.output == HipActivationStorage::fp32 ? 4U : 2U, weights, arguments, output, status);
    const auto kind = weight_kind(ggml_type);
    const auto native_stream = static_cast<hipStream_t>(stream);
    if (config.output == HipActivationStorage::fp32) {
        launch_graph_by_kind<float>(kind, config, weights, vocabulary_size, context_tokens, arguments, output, status, native_stream);
    } else {
        launch_graph_by_kind<std::uint16_t>(kind, config, weights, vocabulary_size, context_tokens, arguments, output, status, native_stream);
    }
}

void preflight_hip_mtp_embedding(const MtpNativeContext& c,std::uint32_t token) {
    mtp_native::context(c);
    const auto h=c.model->hidden_size,v=c.model->vocab_size;
    if(!h || h>std::numeric_limits<std::uint32_t>::max() || !v || token>=v)
        throw std::invalid_argument("MTP native embedding geometry/token invalid");
    const auto& t=c.tensors->embedding;
    if(!t.manifest) throw std::invalid_argument("MTP actual shared embedding is missing");
    mtp_native::tensor(t,{h,v},required_weight_bytes(weight_kind(t.manifest->ggml_type),static_cast<std::uint32_t>(h),v));
    auto output=mtp_native::buffer(c,HipMtpBuffer::embedding,mtp_native::product(h,2));
    mtp_native::disjoint({output,{t.runtime->address,t.runtime->bytes,t.runtime->allocation_generation}});
}
void launch_hip_mtp_embedding(const MtpNativeContext& c,std::uint32_t token,void* stream) {
    preflight_hip_mtp_embedding(c,token);mtp_native::stream(c,stream);
    check_hip(hipGetLastError(),"MTP embedding pending HIP error");
    auto launch=[&]<EmbeddingWeightKind Kind>() {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(embedding_kernel<std::uint16_t,Kind>),
            dim3((static_cast<std::uint32_t>(c.model->hidden_size)+kThreads-1)/kThreads),dim3(kThreads),0,
            reinterpret_cast<hipStream_t>(stream),reinterpret_cast<const void*>(c.tensors->embedding.runtime->address),
            static_cast<std::uint32_t>(c.model->hidden_size),token,
            reinterpret_cast<std::uint16_t*>(c.resources->view(HipMtpBuffer::embedding).address));
    };
    switch(weight_kind(c.tensors->embedding.manifest->ggml_type)) {
    case EmbeddingWeightKind::f32:launch.template operator()<EmbeddingWeightKind::f32>();break;
    case EmbeddingWeightKind::f16:launch.template operator()<EmbeddingWeightKind::f16>();break;
    case EmbeddingWeightKind::bf16:launch.template operator()<EmbeddingWeightKind::bf16>();break;
    case EmbeddingWeightKind::q8_0:launch.template operator()<EmbeddingWeightKind::q8_0>();break;
    case EmbeddingWeightKind::iq4_xs:launch.template operator()<EmbeddingWeightKind::iq4_xs>();break;
#define STRIX_GGML_KIND_CASE(KIND, type) case EmbeddingWeightKind::KIND:launch.template operator()<EmbeddingWeightKind::KIND>();break;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_CASE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_CASE) STRIX_GGML_KIND_CASE(iq4_nl, 20)
#undef STRIX_GGML_KIND_CASE
    }
    check_hip(hipGetLastError(),"MTP native embedding launch");
}

void launch_hip_mtp_embedding_rows(std::span<const MtpNativeContext> contexts,std::span<const std::uint32_t* const> tokens,
    std::span<const std::uint32_t> values,void* stream) {
    if(contexts.empty() || contexts.size()>8U || tokens.size()!=contexts.size() || values.size()!=contexts.size() || !stream)
        throw std::invalid_argument("MTP embedding batch is invalid");
    const auto& c0=contexts.front();
    MtpEmbeddingRows rows{};
    for(std::size_t t=0;t<contexts.size();++t) {
        const auto& c=contexts[t];
        preflight_hip_mtp_embedding(c,tokens[t]?0U:values[t]);
        if(c.tensors->embedding.runtime->address!=c0.tensors->embedding.runtime->address || c.model->hidden_size!=c0.model->hidden_size ||
           c.model->vocab_size!=c0.model->vocab_size)
            throw std::invalid_argument("MTP embedding batch members do not share one embedding table");
        rows.token[t]=tokens[t];rows.value[t]=values[t];
        rows.output[t]=reinterpret_cast<std::uint16_t*>(c.resources->view(HipMtpBuffer::embedding).address);
    }
    check_hip(hipGetLastError(),"MTP embedding batch pending HIP error");
    auto launch=[&]<EmbeddingWeightKind Kind>() {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_embedding_rows_kernel<std::uint16_t,Kind>),
            dim3((static_cast<std::uint32_t>(c0.model->hidden_size)+kThreads-1)/kThreads,static_cast<unsigned>(contexts.size())),
            dim3(kThreads),0,reinterpret_cast<hipStream_t>(stream),reinterpret_cast<const void*>(c0.tensors->embedding.runtime->address),
            static_cast<std::uint32_t>(c0.model->hidden_size),static_cast<std::uint32_t>(c0.model->vocab_size),rows);
    };
    switch(weight_kind(c0.tensors->embedding.manifest->ggml_type)) {
    case EmbeddingWeightKind::f32:launch.template operator()<EmbeddingWeightKind::f32>();break;
    case EmbeddingWeightKind::f16:launch.template operator()<EmbeddingWeightKind::f16>();break;
    case EmbeddingWeightKind::bf16:launch.template operator()<EmbeddingWeightKind::bf16>();break;
    case EmbeddingWeightKind::q8_0:launch.template operator()<EmbeddingWeightKind::q8_0>();break;
    case EmbeddingWeightKind::iq4_xs:launch.template operator()<EmbeddingWeightKind::iq4_xs>();break;
#define STRIX_GGML_KIND_CASE(KIND, type) case EmbeddingWeightKind::KIND:launch.template operator()<EmbeddingWeightKind::KIND>();break;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_CASE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_CASE) STRIX_GGML_KIND_CASE(iq4_nl, 20)
#undef STRIX_GGML_KIND_CASE
    }
    check_hip(hipGetLastError(),"MTP embedding batch launch");
}
namespace {
template<class Rows> void enqueue_mtp_embedding_rows_to(const MtpNativeContext& c,const Rows& rows,unsigned count,void* stream) {
    auto launch=[&]<EmbeddingWeightKind Kind>() {
        if constexpr (std::is_same_v<Rows,MtpEmbeddingRows>) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_embedding_rows_kernel<std::uint16_t,Kind>),
                dim3((static_cast<std::uint32_t>(c.model->hidden_size)+kThreads-1)/kThreads,count),
                dim3(kThreads),0,reinterpret_cast<hipStream_t>(stream),reinterpret_cast<const void*>(c.tensors->embedding.runtime->address),
                static_cast<std::uint32_t>(c.model->hidden_size),static_cast<std::uint32_t>(c.model->vocab_size),rows);
        } else {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_embedding_prime_rows_kernel<Kind>),
                dim3((static_cast<std::uint32_t>(c.model->hidden_size)+kThreads-1)/kThreads,count),
                dim3(kThreads),0,reinterpret_cast<hipStream_t>(stream),reinterpret_cast<const void*>(c.tensors->embedding.runtime->address),
                static_cast<std::uint32_t>(c.model->hidden_size),static_cast<std::uint32_t>(c.model->vocab_size),rows);
        }
    };
    switch(weight_kind(c.tensors->embedding.manifest->ggml_type)) {
    case EmbeddingWeightKind::f32:launch.template operator()<EmbeddingWeightKind::f32>();break;
    case EmbeddingWeightKind::f16:launch.template operator()<EmbeddingWeightKind::f16>();break;
    case EmbeddingWeightKind::bf16:launch.template operator()<EmbeddingWeightKind::bf16>();break;
    case EmbeddingWeightKind::q8_0:launch.template operator()<EmbeddingWeightKind::q8_0>();break;
    case EmbeddingWeightKind::iq4_xs:launch.template operator()<EmbeddingWeightKind::iq4_xs>();break;
#define STRIX_GGML_KIND_CASE(KIND, type) case EmbeddingWeightKind::KIND:launch.template operator()<EmbeddingWeightKind::KIND>();break;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_CASE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_CASE) STRIX_GGML_KIND_CASE(iq4_nl, 20)
#undef STRIX_GGML_KIND_CASE
    }
}
}
void launch_hip_mtp_embedding_rows_to(const MtpNativeContext& c,std::span<const std::uint32_t> tokens,
    std::span<std::uint16_t* const> outputs,void* stream) {
    if(tokens.empty() || tokens.size()>8U || outputs.size()!=tokens.size() || !stream)
        throw std::invalid_argument("MTP embedding row batch is invalid");
    MtpEmbeddingRows rows{};
    for(std::size_t t=0;t<tokens.size();++t) {
        preflight_hip_mtp_embedding(c,tokens[t]);
        if(!outputs[t]) throw std::invalid_argument("MTP embedding row batch output is missing");
        rows.token[t]=nullptr;rows.value[t]=tokens[t];rows.output[t]=outputs[t];
    }
    mtp_native::stream(c,stream);
    check_hip(hipGetLastError(),"MTP embedding row batch pending HIP error");
    enqueue_mtp_embedding_rows_to(c,rows,static_cast<unsigned>(tokens.size()),stream);
    check_hip(hipGetLastError(),"MTP embedding row batch launch");
}
void launch_hip_mtp_embedding_prime_rows(const MtpNativeContext& c,std::span<const std::uint32_t> tokens,
    HipMtpPrimeStride output,void* stream) {
    if(tokens.empty() || tokens.size()>hip_mtp_prime_rows_max || !output.base || !output.stride || !stream)
        throw std::invalid_argument("MTP embedding prime batch is invalid");
    HipMtpEmbeddingPrimeRows rows{};rows.output=output;
    for(std::size_t t=0;t<tokens.size();++t) {preflight_hip_mtp_embedding(c,tokens[t]);rows.value[t]=tokens[t];}
    mtp_native::stream(c,stream);
    check_hip(hipGetLastError(),"MTP embedding prime batch pending HIP error");
    enqueue_mtp_embedding_rows_to(c,rows,static_cast<unsigned>(tokens.size()),stream);
    check_hip(hipGetLastError(),"MTP embedding prime batch launch");
}
void launch_hip_mtp_embedding_device(const MtpNativeContext& c,const std::uint32_t* token,void* stream) {
    if(!token) throw std::invalid_argument("MTP device-token embedding requires a token word");
    preflight_hip_mtp_embedding(c,0U);mtp_native::stream(c,stream);
    check_hip(hipGetLastError(),"MTP embedding pending HIP error");
    auto launch=[&]<EmbeddingWeightKind Kind>() {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(device_token_embedding_kernel<std::uint16_t,Kind>),
            dim3((static_cast<std::uint32_t>(c.model->hidden_size)+kThreads-1)/kThreads),dim3(kThreads),0,
            reinterpret_cast<hipStream_t>(stream),reinterpret_cast<const void*>(c.tensors->embedding.runtime->address),
            static_cast<std::uint32_t>(c.model->hidden_size),static_cast<std::uint32_t>(c.model->vocab_size),token,
            reinterpret_cast<std::uint16_t*>(c.resources->view(HipMtpBuffer::embedding).address));
    };
    switch(weight_kind(c.tensors->embedding.manifest->ggml_type)) {
    case EmbeddingWeightKind::f32:launch.template operator()<EmbeddingWeightKind::f32>();break;
    case EmbeddingWeightKind::f16:launch.template operator()<EmbeddingWeightKind::f16>();break;
    case EmbeddingWeightKind::bf16:launch.template operator()<EmbeddingWeightKind::bf16>();break;
    case EmbeddingWeightKind::q8_0:launch.template operator()<EmbeddingWeightKind::q8_0>();break;
    case EmbeddingWeightKind::iq4_xs:launch.template operator()<EmbeddingWeightKind::iq4_xs>();break;
#define STRIX_GGML_KIND_CASE(KIND, type) case EmbeddingWeightKind::KIND:launch.template operator()<EmbeddingWeightKind::KIND>();break;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_CASE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_CASE) STRIX_GGML_KIND_CASE(iq4_nl, 20)
#undef STRIX_GGML_KIND_CASE
    }
    check_hip(hipGetLastError(),"MTP device-token embedding launch");
}

}  // namespace strix
