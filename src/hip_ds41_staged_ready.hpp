#pragma once
#include "strix/dist/staged_ready.hpp"
#include "strix/hip_ds41_layer.hpp"
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::deepseek41::ready_detail {
namespace {
inline bool enabled(const char* name) {
    const char* e = std::getenv(name);
    return e && e[0] == '1' && e[1] == '\0';
}
// Production defaults: on unless =0
inline bool enabled_default(const char* name, bool fallback = true) {
    const char* e = std::getenv(name);
    if (!e || !*e) return fallback;
    return !(e[0] == '0' && e[1] == '\0');
}
// STRIX_DS41_READY_LEAN=1: the lean releases of staged_ready.hpp (release-only lane fences, no
// per-block control-line loads, token checked by the last block) in every ds41 ready kernel.
inline bool lean() {
    static const bool on = enabled_default("STRIX_DS41_READY_LEAN");
    return on;
}
// STRIX_DS41_MOE_READY=1: the MoE half-sum exchange through the ready views (always the lean releases).
inline bool moe() {
    static const bool on = enabled_default("STRIX_DS41_MOE_READY");
    return on;
}
// The view as the kernels take it: without the counter they use the general releases.
inline dist::ProducerStage kernel_view(dist::ProducerStage stage) {
    if (!lean()) stage.counter = nullptr;
    else if (!stage.counter) throw std::logic_error("STRIX_DS41_READY_LEAN: ready view without its block counter");
    return stage;
}
__global__ void publish_ready_kernel(dist::ProducerStage stage) {
    dist::publish_staged_view(stage, 1U);
}
// STRIX_DS41_TRELLIS_OB_READY (default on, =0 off): the TP trellis o_b exchange through the ready views. The trellis plans take
// no host output, so one small kernel copies the device half into the coherent TX view and publishes it; the add reads
// the RX view in place. Replaces the SDMA copy + event of start() and the H2D copy of finish(); same adds, same bits.
inline bool trellis_ob() {
    static const bool on = enabled_default("STRIX_DS41_TRELLIS_OB_READY");
    return on;
}
__global__ void copy_publish_kernel(const float4* __restrict__ src, float4* dst, std::uint32_t n4, dist::ProducerStage stage) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) dst[i] = src[i];
    dist::publish_staged_view(stage, gridDim.x);
}
inline void copy_publish(const float* src, dist::ProducerStage stage, std::uint32_t n, hipStream_t stream) {
    if (n % 4U) throw std::logic_error("ready copy: float count not a multiple of 4");
    const std::uint32_t n4 = n / 4U;
    copy_publish_kernel<<<(n4 + 255U) / 256U, 256, 0, stream>>>(reinterpret_cast<const float4*>(src), static_cast<float4*>(stage.data), n4,
                                                                kernel_view(stage));
    const auto status = hipGetLastError();
    if (status != hipSuccess) throw std::runtime_error(std::string("ready copy: ") + hipGetErrorString(status));
}
inline void product(const ds41::Q8A8View& v, dist::ProducerStage stage, hipStream_t stream) {
    static const bool epilogue = enabled_default("STRIX_DS41_OB_READY_EPILOGUE", lean());
    stage = kernel_view(stage);
    if (epilogue && ds41::enqueue_q8_0_a8_ready(v, stage, stream)) return;
    ds41::enqueue_q8_0_a8(v, stream);
    publish_ready_kernel<<<1, 64, 0, stream>>>(stage);
    const auto status = hipGetLastError();
    if (status != hipSuccess) throw std::runtime_error(std::string("publish ready: ") + hipGetErrorString(status));
}
inline bool reads_only() {
    static const bool on = enabled_default("STRIX_DS41_READY_READS_ONLY", lean());
    return on;
}
template <bool ReadsOnly, bool Lean>
__global__ void add_halves_ready(const float* a, const float* b, float* __restrict__ out, std::uint32_t n,
                                 dist::StagedReady* flag, std::uint64_t token, unsigned* counter) {
    if constexpr (Lean) dist::staged_acquire_lean();
    else dist::staged_acquire(flag, token);
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
    // The add output stays GPU-owned; continuation runs on this same stream.
    // Every padding lane participates after its final RX read.
    if constexpr (Lean) {
        if constexpr (ReadsOnly) dist::publish_staged_reads_block_lean(flag, counter, token, gridDim.x);
        else dist::publish_staged_block_lean(flag, counter, token, gridDim.x);
    } else {
        if constexpr (ReadsOnly) dist::publish_staged_reads_block(flag, token, gridDim.x);
        else dist::publish_staged_block(flag, token, gridDim.x);
    }
}
template <bool Lean>
inline void launch_add(const float* a, const float* b, float* out, std::uint32_t n, dist::ConsumerStage peer, hipStream_t stream) {
    const dim3 grid((n + 255U) / 256U);
    if (reads_only()) add_halves_ready<true, Lean><<<grid, 256, 0, stream>>>(a, b, out, n, peer.flag, peer.token, peer.counter);
    else add_halves_ready<false, Lean><<<grid, 256, 0, stream>>>(a, b, out, n, peer.flag, peer.token, peer.counter);
}
// Strided halves through the ready views (the Engram output-row exchange): `rows` rows of `half` floats, row r at
// base[r * stride + first], packed contiguous into the TX view by the kernel that publishes it, and the peer's rows
// written back from the RX view by the kernel that releases it. Plain copies: the same bits as pack + copies + scatter.
__global__ void gather_publish_kernel(const float* __restrict__ base, float* tx, std::uint32_t rows, std::uint32_t stride, std::uint32_t half,
                                      std::uint32_t first, dist::ProducerStage stage) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * half) tx[i] = base[std::size_t{i / half} * stride + first + i % half];
    dist::publish_staged_view(stage, gridDim.x);
}
inline void gather_publish(const float* base, std::uint32_t rows, std::uint32_t stride, std::uint32_t half, std::uint32_t first,
                           dist::ProducerStage stage, hipStream_t stream) {
    gather_publish_kernel<<<(rows * half + 255U) / 256U, 256, 0, stream>>>(base, static_cast<float*>(stage.data), rows, stride, half, first,
                                                                           kernel_view(stage));
    const auto status = hipGetLastError();
    if (status != hipSuccess) throw std::runtime_error(std::string("ready gather: ") + hipGetErrorString(status));
}
template <bool ReadsOnly, bool Lean>
__global__ void scatter_ready_kernel(const float* rx, float* __restrict__ base, std::uint32_t rows, std::uint32_t stride, std::uint32_t half,
                                     std::uint32_t first, dist::StagedReady* flag, std::uint64_t token, unsigned* counter) {
    if constexpr (Lean) dist::staged_acquire_lean();
    else dist::staged_acquire(flag, token);
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * half) base[std::size_t{i / half} * stride + first + i % half] = rx[i];
    // the scattered rows stay GPU-owned on this stream, as the add's output
    if constexpr (Lean) {
        if constexpr (ReadsOnly) dist::publish_staged_reads_block_lean(flag, counter, token, gridDim.x);
        else dist::publish_staged_block_lean(flag, counter, token, gridDim.x);
    } else {
        if constexpr (ReadsOnly) dist::publish_staged_reads_block(flag, token, gridDim.x);
        else dist::publish_staged_block(flag, token, gridDim.x);
    }
}
inline void scatter(dist::ConsumerStage peer, float* base, std::uint32_t rows, std::uint32_t stride, std::uint32_t half, std::uint32_t first,
                    hipStream_t stream) {
    const dim3 grid((rows * half + 255U) / 256U);
    const auto* rx = static_cast<const float*>(peer.data);
    if (lean() && !peer.counter) throw std::logic_error("STRIX_DS41_READY_LEAN: consumer view without its block counter");
    if (lean() && reads_only())
        scatter_ready_kernel<true, true><<<grid, 256, 0, stream>>>(rx, base, rows, stride, half, first, peer.flag, peer.token, peer.counter);
    else if (lean())
        scatter_ready_kernel<false, true><<<grid, 256, 0, stream>>>(rx, base, rows, stride, half, first, peer.flag, peer.token, peer.counter);
    else if (reads_only())
        scatter_ready_kernel<true, false><<<grid, 256, 0, stream>>>(rx, base, rows, stride, half, first, peer.flag, peer.token, peer.counter);
    else
        scatter_ready_kernel<false, false><<<grid, 256, 0, stream>>>(rx, base, rows, stride, half, first, peer.flag, peer.token, peer.counter);
    const auto status = hipGetLastError();
    if (status != hipSuccess) throw std::runtime_error(std::string("ready scatter: ") + hipGetErrorString(status));
}
inline void add(const float* a, const float* b, float* out, std::uint32_t n, dist::ConsumerStage peer, hipStream_t stream) {
    if (lean()) {
        if (!peer.counter) throw std::logic_error("STRIX_DS41_READY_LEAN: consumer view without its block counter");
        launch_add<true>(a, b, out, n, peer, stream);
    } else {
        launch_add<false>(a, b, out, n, peer, stream);
    }
    const auto status = hipGetLastError();
    if (status != hipSuccess) throw std::runtime_error(std::string("add_halves_ready: ") + hipGetErrorString(status));
}
}
}
