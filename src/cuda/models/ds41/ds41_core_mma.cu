// Model adapters for the shared tiled Q8_0 x A8 core. Allocation stays with the owner.
#include "ds41_cuda_common.cuh"
#include "strix/cuda/ds41_core.hpp"
#include "strix/cuda/core/dense.hpp"

namespace strix::ds41::cuda {
namespace dense = ::strix::cuda::core::dense;
namespace core = ::strix::cuda::core;

void set_q8_pdl_prefetch(bool on) { dense::set_q8_pdl_prefetch(on); }
void set_q8_pdl_mode(int mode) { dense::set_q8_pdl_mode(mode); }
void set_q8_r1(bool on) { dense::set_q8_r1(on); }
void set_q8_r1_pad(bool on) { dense::set_q8_r1_pad(on); }
void set_q8_occupancy(int blocks) { dense::set_q8_occupancy(blocks); }
std::size_t q8_tiled_code_bytes(std::uint32_t rows, std::uint32_t width) { return dense::q8_tiled_code_bytes(rows, width); }
std::size_t q8_tiled_scale_count(std::uint32_t rows, std::uint32_t width) { return dense::q8_tiled_scale_count(rows, width); }
std::size_t q8_0_a8_split_scratch_floats(std::uint32_t count, std::uint32_t rows, std::uint32_t split) {
    return dense::q8_0_a8_split_scratch_floats(count, rows, split);
}

void Q8TiledBuffer::repack(const std::uint8_t* q8, std::uint32_t rows_, std::uint32_t width_, cudaStream_t stream) {
    detail::require(rows_ % 16U == 0 && width_ % 32U == 0 && rows_ > 0, "tiled Q8_0 geometry (rows % 16, width % 32)");
    rows = rows_;
    width = width_;
    codes.resize(q8_tiled_code_bytes(rows, width));
    scales.resize(q8_tiled_scale_count(rows, width));
    dense::enqueue_q8_tiled_repack(q8, codes.get(), scales.get(), rows, width, stream);
}

void enqueue_q8_0_a8_tiled(const Q8A8View& v, cudaStream_t stream) {
    dense::Q8A8View site;
    site.tiled = v.tiled;
    site.split = v.split;
    site.first = v.first;
    site.count = v.count;
    site.per_group = v.per_group;
    site.k_offset = v.k_offset;
    site.width = v.width;
    site.x = v.x;
    site.out = v.out;
    site.out_stride = v.out_stride;
    site.accumulate = v.accumulate;
    site.pdl = v.pdl;
    site.occupancy = v.occupancy;
    site.k_halves = v.k_halves;
    site.hc_residual = v.hc_residual;
    site.hc_mixes = v.hc_mixes;
    site.hc_out = v.hc_out;
    site.hc_hidden = v.hc_hidden;
    core::DensePlan plan;
    plan.count = v.count;
    plan.width = v.width ? v.width : v.tiled.width;
    plan.rows = v.x.rows;
    plan.mode = v.x.rows <= 8U ? core::DenseMode::decode : core::DenseMode::prefill;
    plan.split_scratch = v.split_scratch;
    plan.split_scratch_floats = v.split_scratch_floats;
    plan.stream = stream;
    dense::enqueue_q8_0_a8_tiled(site, plan);
}

void enqueue_swiglu_a8_down(const float* gate_up, std::uint32_t rows, float limit, Q8Tiled down,
                          float* out, std::uint32_t out_stride, A8Rows tap, cudaStream_t stream) {
    core::DensePlan plan;
    plan.count = down.rows;
    plan.width = down.width;
    plan.rows = rows;
    plan.mode = rows <= 8U ? core::DenseMode::decode : core::DenseMode::prefill;
    plan.stream = stream;
    dense::enqueue_swiglu_a8_down(gate_up, limit, down, out, out_stride, tap, plan);
}

}  // namespace strix::ds41::cuda
