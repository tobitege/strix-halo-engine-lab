#include <strix/weight/types.hpp>
#include <array>
namespace strix::weight {
namespace {
// Descriptive migration inventory, not complete frozen arithmetic specifications.
// Unspecified boundaries must be frozen and qualified before implementation migration.
// These records never select a production kernel.
constexpr auto contracts = std::to_array<ArithmeticDescriptor>({
  {{"cpu-q8a8-tree32", 1}, Exactness::legacy_bitwise, "groups32; amax=max(abs(x)); delta=1 if amax<2^-119 else amax/127; tiny 0<amax<2^-119 -> codes0; otherwise nearbyint(x/delta), clamp[-127,127] (RNE environment)", "raw Q8_0 int8", "int32 dot32 then float(dot)*(dw*dx)", "+0 lanes; g ascending into lane g%32", "32 lanes", "xor butterfly 16,8,4,2,1; output=p or old_output+p when accumulate", "none", "none", false, false},
  {{"intel-q8a8-f16-ascending", 1}, Exactness::legacy_bitwise, "RNE signed A8, groups32", "F16(code*scale)", "F16 operands -> FP32 product", "k ascending", "one output", "ascending from +0", "none", "none", false, false},
  {{"dense-q8a8-f16-v1", 1}, Exactness::fast, "A8 -> F16", "Q8_0 -> F16", "HIP WMMA FP32", "CTA fixed K partitions", "CTA", "ascending partitions", "none", "none", false, false},
  {{"ds41-group-product", 1}, Exactness::legacy_bitwise, "existing group operand", "format delegate", "ordered group product", "format-defined", "format-defined", "existing decoder order", "none", "none", false, false},
  {{"ds41-q2k-affine", 1}, Exactness::legacy_bitwise, "existing group operand", "Q2_K signed groups", "integer dot plus affine correction", "format-defined", "format-defined", "existing decoder order", "none", "none", false, false},
  {{"cuda-qwen-imma-four-chain", 1}, Exactness::legacy_bitwise, "signed A8", "raw quantized groups", "integer IMMA", "four chains", "IMMA tile", "pair fold", "none", "none", false, false},
  {{"cuda-qwen-iq-native", 1}, Exactness::fast, "BF16 input", "native IQ FP32", "FP32 FMA", "native IQ tile", "native IQ tile", "existing CUDA order", "none", "none", true, false},
  {{"exl3-linear-v1", 1}, Exactness::legacy_bitwise, "existing staged operand", "trellis transformed-domain", "WMMA FP32", "rows-dependent K16 buckets", "K16 splits", "ascending partitions", "EXL3 input transform", "EXL3 output transform", false, false},
  {{"exl3-linear-v2", 1}, Exactness::legacy_bitwise, "existing staged operand", "trellis transformed-domain", "WMMA FP32", "fixed rows-free K16 splits", "K16 splits", "ascending partitions", "EXL3 input transform", "EXL3 output transform", false, false},
  {{"g4-exl3-v1", 1}, Exactness::legacy_bitwise, "RN16(H128(RN16(RN32(x*suh))))", "decoded FP16", "separate FP32 multiply/add", "chunk16", "chunk16", "ascending chunks", "EXL3 input transform", "EXL3 output transform", false, false},
  {{"g4-exl3-v2", 1}, Exactness::legacy_bitwise, "RN16(H128(RN16(RN32(x*suh))))", "decoded FP16", "exact FP16 product -> FP32 add", "contiguous I/16 split ranges; S<=4", "four lanes: 2a,2a+1,2a+8,2a+9", "(acc0+acc2)+(acc1+acc3), ascending splits", "EXL3 input transform", "EXL3 output transform", false, false},
  {{"g4-exl3-v3", 1}, Exactness::legacy_bitwise, "RN16(H128(RN16(RN32(x*suh))))", "decoded FP16", "exact FP16 product -> FP32 add", "contiguous I/16 split ranges; S<=8", "four lanes: 2a,2a+1,2a+8,2a+9", "(acc0+acc2)+(acc1+acc3), ascending splits", "EXL3 input transform", "EXL3 output transform", false, false},
  {{"g4-exl3-wmma-pf-v1", 1}, Exactness::legacy_bitwise, "F16", "F16", "WMMA FP32", "existing prefill partition", "WMMA tile", "existing prefill order", "EXL3 input transform", "EXL3 output transform", false, false},
  {{"trellis-bitwise", 1}, Exactness::legacy_bitwise, "existing operand", "trellis reference_bits", "existing bitwise product", "existing bitwise partition", "existing bitwise partition", "existing bitwise order", "trellis input transform", "trellis output transform", false, false},
  {{"trellis-quality", 1}, Exactness::fast, "existing operand", "trellis quality", "existing quality product", "existing quality partition", "existing quality partition", "existing quality order", "trellis input transform", "trellis output transform", true, false},
  {{"h3-bf16", 1}, Exactness::legacy_bitwise, "BF16", "BF16", "existing H3 product", "existing H3 partition", "existing H3 partition", "existing H3 order", "none", "none", true, false},
  {{"int8_tensorwise-convrot256", 1}, Exactness::legacy_bitwise, "RNE/clamp signed A8", "tensorwise I8", "existing H3 int8 product", "groups256", "existing H3 partition", "existing H3 order", "convrot256", "channel scales", false, false},
  {{"asym_w4a8_int8", 1}, Exactness::legacy_bitwise, "RNE/clamp signed A8", "asymmetric W4", "existing H3 int8 product", "declared group size", "existing H3 partition", "existing H3 order", "none", "group scales", false, false},
  {{"nvfp4-mma-f32-v1", 1}, Exactness::fast, "E2M1/E4M3 scaled", "E2M1/E4M3 scaled", "native NVFP4 MMA FP32", "native MMA tile", "native MMA tile", "tolerance-qualified MMA", "none", "tensor scale", true, false},
});
}
std::span<const ArithmeticDescriptor> legacy_arithmetic() noexcept { return contracts; }
}
