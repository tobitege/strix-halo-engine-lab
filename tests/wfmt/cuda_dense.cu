#include "../../src/cuda/weight/ggml_sources.cuh"
#include "strix/weight/source.hpp"

#include <cuda_runtime.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace strix::weight::test {
namespace {
struct DecodeArch {};
struct SemanticF32 {};
template <class T> struct DeviceBytes {
    T* data{};
    ~DeviceBytes() { if (data) cudaFree(data); }
    bool allocate(std::size_t count) { return cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)) == cudaSuccess; }
};
template <std::uint32_t Type>
using ScalarSource = ::strix::cuda::weight::OperandSource<::strix::cuda::weight::GgmlTag<Type>,
    ::strix::cuda::weight::NativeScalarTag, DecodeArch, SemanticF32>;

template <std::uint32_t Type>
__global__ void decode_groups(typename ScalarSource<Type>::View view, float* output, std::uint32_t groups) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= groups) return;
    const unsigned row = i / (view.k / 32), group = i % (view.k / 32);
    float decoded[32];
    ScalarSource<Type>::group(view, row, group, decoded);
#pragma unroll
    for (unsigned j = 0; j < 32; ++j) output[i * 32 + j] = decoded[j];
}

void half_at(std::vector<Byte>& bytes, std::size_t offset, unsigned short value) {
    bytes[offset] = Byte(value & 255);
    bytes[offset + 1] = Byte(value >> 8);
}
template <std::uint32_t Type>
bool decode_format(std::uint32_t seed) {
    namespace cw = ::strix::cuda::weight;
    namespace quant = ::strix::cuda::quant;
    const auto wire = quant::geometry(Type);
    const std::uint32_t rows = 3, width = wire.block_elements * 2, count = rows * width;
    std::vector<Byte> raw(std::size_t(count / wire.block_elements) * wire.block_bytes);
    std::uint32_t rng = seed + Type * 31337;
    for (auto& byte : raw) { rng = rng * 1664525U + 1013904223U; byte = Byte(rng >> 24); }
    if constexpr (Type == quant::t_f32) {
        constexpr std::array<std::uint32_t, 12> bits{0,0x80000000U,1,0x007fffffU,0x00800000U,0x3f800000U,
            0xbf800000U,0x7f7fffffU,0xff7fffffU,0x3eaaaaabU,0x00800001U,0x80800001U};
        for (std::uint32_t i = 0; i < count; ++i) std::memcpy(raw.data() + 4 * i, &bits[(i + seed) % bits.size()], 4);
    } else if constexpr (Type == quant::t_f16 || Type == quant::t_bf16) {
        constexpr std::array<unsigned short, 10> bits{0,0x8000,1,0x03ff,0x0400,0x3c00,0xbc00,0x7bff,0xfbff,0x3555};
        for (std::uint32_t i = 0; i < count; ++i) half_at(raw, 2 * i, bits[(i + seed) % bits.size()]);
    } else {
        constexpr std::array<unsigned short, 6> scales{0,0x8000,1,0x03ff,0x0400,0x3c00};
        for (std::size_t block = 0; block < raw.size() / wire.block_bytes; ++block) {
            const auto offset = block * wire.block_bytes;
            half_at(raw, offset + (Type == quant::t_q6_k ? 208 : 0), scales[(block + seed) % scales.size()]);
            if constexpr (Type == quant::t_q5_1 || Type == quant::t_q4_k || Type == quant::t_q5_k)
                half_at(raw, offset + 2, 0x3400);
        }
    }
    WeightView weight{};
    weight.storage = gguf_format(Type);
    weight.shape = {{{rows, width, 0, 0}}, 2};
    if (!format_geometry(weight.storage, weight.parameters, weight.geometry)) return false;
    weight.payload = {memory_source(raw), {0, raw.size()}};
    std::vector<float> expected(count), got(count);
    DecodeBuffer buffer{std::as_writable_bytes(std::span(expected)), 0};
    const auto status = decode_reference(builtin_registry(), weight, {0, count, {}, Scalar::f32}, buffer);
    if (!status || buffer.written != count * sizeof(float)) {
        std::fprintf(stderr, "CUDA source oracle type %u: %s\n", Type, status.detail.c_str());
        return false;
    }
    const cw::LaunchShape shape{rows, rows, width};
    if (!ScalarSource<Type>::validate(weight, shape)) return false;
    DeviceBytes<Byte> source;
    DeviceBytes<float> output;
    if (!source.allocate(raw.size()) || !output.allocate(count)) return false;
    if (cudaMemcpy(source.data, raw.data(), raw.size(), cudaMemcpyHostToDevice) != cudaSuccess) return false;
    typename ScalarSource<Type>::View view{};
    const cw::UploadedSlices upload{{source.data, raw.size()}, {}};
    if (!cw::make_view(weight, upload, view)) return false;
    const cw::UploadedSlices truncated{{source.data, raw.size() - 1}, {}};
    typename ScalarSource<Type>::View rejected{};
    if (cw::make_view(weight, truncated, rejected)) return false;
    auto wrong_shape = shape;
    wrong_shape.k += 32;
    if (ScalarSource<Type>::validate(weight, wrong_shape)) return false;
    decode_groups<Type><<<(count / 32 + 63) / 64, 64>>>(view, output.data, count / 32);
    if (cudaGetLastError() != cudaSuccess || cudaMemcpy(got.data(), output.data, count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    if (std::memcmp(expected.data(), got.data(), count * sizeof(float))) {
        for (std::size_t i = 0; i < count; ++i) if (std::memcmp(&expected[i], &got[i], sizeof(float))) {
            std::fprintf(stderr, "CUDA source type %u seed %u element %zu: %a != %a\n", Type, seed, i, got[i], expected[i]); break;
        }
        return false;
    }
    return true;
}
}

bool cuda_dense_case() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != 0) return false;
    for (std::uint32_t seed : {0U, 1U, 5U, 0xffffffffU}) {
#define DECODE(Type) if (!decode_format<Type>(seed)) return false
        DECODE(0); DECODE(1); DECODE(2); DECODE(6); DECODE(7); DECODE(8);
        DECODE(12); DECODE(13); DECODE(14); DECODE(18); DECODE(20); DECODE(21);
        DECODE(22); DECODE(23); DECODE(30); DECODE(42);
#undef DECODE
    }
    return true;
}
} // namespace strix::weight::test
