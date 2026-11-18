#pragma once
#include "native.cuh"
namespace strix::glm::rtxpf {
// Root-level lm_head (K6 EXL3, mul1 0x83dcd12d): trellis I16 [256,9680,96],
// suh F16 [4096], svh F16 [154880]. Decoded once to FP16 [4096,154880]
// (K-major, the layer decoder's layout). Projection keeps the H128 input and
// output transforms and RN16 boundaries; the K chain is one ascending scalar
// FP32 __fmaf_rn chain per logit (the projection reference semantics).
inline constexpr const char* head_identity="glm-dec-rtx-native-v1-head-k6-f32seq";
inline constexpr unsigned head_vocab=154880,head_hidden=4096;
struct HeadWeights {const U16 *decoded,*suh,*svh;};
std::size_t head_decoded_bytes();
std::size_t head_workspace_bytes(unsigned rows);
// No public decode() bound applies: the output width exceeds the FFN diagnostic decoder.
cudaError_t decode_head(Matrix packed,U16* decoded,cudaStream_t);
// hidden [rows,4096] final-normalized FP32 -> logits [rows,154880] FP32.
cudaError_t enqueue_head(const HeadWeights&,const float* hidden,float* logits,unsigned rows,void* workspace,std::size_t bytes,cudaStream_t);
// result[0]=token (0xffffffff if none), result[1]=nonfinite count, result[2]=winning logit bits.
// Order: logit descending, token ascending; +0==-0; nonfinite never wins.
cudaError_t enqueue_argmax(const float* logits,unsigned n,U32* result,cudaStream_t);
}
