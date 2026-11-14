// GLM-5.3-Flash indexer + absorbed MLA attention device kernels (see glm_dsa.hpp). Operation order = host emulation
// (glm_host.cpp dsa_append / indexer_select / mla_attention). Build with -ffp-contract=off -fno-fast-math
// -fno-gpu-flush-denormals-to-zero.
#include "glm/kernels/glm_dsa.hpp"
#include "attn_core/select.hpp"

#include <stdexcept>
#include <string>

namespace strix::glm::dev {

namespace {

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

template <unsigned L> __device__ __forceinline__ float lds_tree(float* s, float part) {
    const unsigned l = threadIdx.x;
    s[l] = part;
    __syncthreads();
    for (unsigned h = L / 2; h > 0; h >>= 1U) {
        if (l < h) s[l] = s[l] + s[l + h];
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}

// ---------------------------------------------------------------- append (R4 LayerNorm, cache writes, R12 pooling)
__global__ __launch_bounds__(128) void dsa_store_kernel(const DsaSlot* slots, const unsigned* tpos, const float* latent,
                                                        IndexerDev w, const float* key, const float* gs, unsigned* flag) {
    __shared__ float s[128];
    const unsigned r = blockIdx.x, i = threadIdx.x;
    const DsaSlot sl = slots[r];
    const unsigned t = tpos[r];
    const float k = key[static_cast<std::size_t>(r) * kIdxDim + i];
    const float mu = lds_tree<128>(s, 0.0F + k) / static_cast<float>(kIdxDim);
    const float d = k - mu;
    const float var = lds_tree<128>(s, __builtin_fmaf(d, d, 0.0F)) / static_cast<float>(kIdxDim);
    const float rr = inv_sqrt(var + kLnEps);
    const std::uint16_t kb = float_to_bf16((d * rr) * f16_to_float(w.k_norm_w[i]) + f16_to_float(w.k_norm_b[i]));
    const std::uint16_t gb = float_to_bf16(gs[static_cast<std::size_t>(r) * kIdxDim + i]);
    if (!bf16_is_finite(kb) || !bf16_is_finite(gb)) atomicOr(flag, 1U);
    sl.ikey[static_cast<std::size_t>(t) * kIdxDim + i] = kb;
    sl.igate[static_cast<std::size_t>(t) * kIdxDim + i] = gb;
    for (unsigned l = i; l < kKvLora; l += 128) {
        const std::uint16_t lb = float_to_bf16(latent[static_cast<std::size_t>(r) * kKvLora + l]);
        if (!bf16_is_finite(lb)) atomicOr(flag, 1U);
        sl.latent[static_cast<std::size_t>(t) * kKvLora + l] = lb;
    }
    __syncthreads();
    if ((t + 1) % kKpool == 0) {  // pool (t + 1) / 4 - 1 completed: R12 for channel i
        const unsigned t0 = t + 1 - kKpool;
        float l4[kKpool], kv[kKpool], e[kKpool], p[kKpool];
        for (unsigned j = 0; j < kKpool; ++j) {
            l4[j] = bf16_to_float(sl.igate[static_cast<std::size_t>(t0 + j) * kIdxDim + i]) + w.ape[j * kIdxDim + i];
            kv[j] = bf16_to_float(sl.ikey[static_cast<std::size_t>(t0 + j) * kIdxDim + i]);
        }
        float m = l4[0];
        for (unsigned j = 1; j < kKpool; ++j) m = l4[j] > m ? l4[j] : m;
        for (unsigned j = 0; j < kKpool; ++j) e[j] = declared_exp(l4[j] - m);
        const float sum = ((e[0] + e[1]) + e[2]) + e[3];
        for (unsigned j = 0; j < kKpool; ++j) p[j] = e[j] / sum;
        const float K = ((p[0] * kv[0] + p[1] * kv[1]) + p[2] * kv[2]) + p[3] * kv[3];
        if (!is_finite(K)) atomicOr(flag, 1U);
        sl.pooled[static_cast<std::size_t>(t / kKpool) * kIdxDim + i] = K;
    }
}

// ---------------------------------------------------------------- scores (R11)
__global__ void scale_kernel(float* v, unsigned n, float c) {
    const unsigned i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) v[i] = v[i] * c;
}
__global__ void idx_score_kernel(const float* pooled, unsigned P, const float* qI, const float* wh, float* score,
                                 std::uint32_t* keys) {
    const unsigned p = blockIdx.x * 256 + threadIdx.x;
    if (p >= P) return;
    const float* K = pooled + static_cast<std::size_t>(p) * kIdxDim;
    float acc = 0.0F;
    for (unsigned h = 0; h < kIdxHeads; ++h) {
        float dot = 0.0F;
        for (unsigned i = 0; i < kIdxDim; ++i) dot = __builtin_fmaf(qI[h * kIdxDim + i], K[i], dot);
        const float sc = dot * kInvSqrt128;
        acc = __builtin_fmaf(wh[h], sc > 0.0F ? sc : 0.0F, acc);
    }
    score[p] = acc;
    keys[p] = ordered_key(acc);
}

// ---------------------------------------------------------------- exact top-k (R5) + canonical token list (section 1.4)
// One block of 1024 threads. Radix select on the 32-bit ordered keys finds the k-th largest key K*; selected = key > K*
// plus the lowest-index pools with key == K* to fill k. Output in ascending pool order by an ordered block scan.
__global__ __launch_bounds__(1024) void idx_select_kernel(const std::uint32_t* keys, unsigned P, unsigned t,
                                                          std::uint32_t* tokens, std::uint32_t* n_out) {
    __shared__ unsigned hist[256];
    __shared__ unsigned sh_prefix, sh_mask, sh_remaining, sh_carry_eq, sh_carry_out;
    __shared__ unsigned scan_eq[1024], scan_sel[1024];
    const unsigned l = threadIdx.x;
    const unsigned k = P < kIdxPools ? P : kIdxPools;
    strix::attn_core::select::select_row<strix::attn_core::select::ChunkPolicy<kKpool, true>>(
        keys, P, t, k, l, tokens, n_out,
        strix::attn_core::select::ChunkStorage{hist, sh_prefix, sh_mask, sh_remaining, sh_carry_eq, sh_carry_out, scan_eq, scan_sel});
}

// ---------------------------------------------------------------- MLA (R13)
__global__ void mla_absorb_kernel(const float* q, const std::uint16_t* kv_b, unsigned heads, float* qt) {
    const unsigned l = blockIdx.x * 256 + threadIdx.x, h = blockIdx.y;
    if (l >= kKvLora || h >= heads) return;
    float acc = 0.0F;
    for (unsigned d = 0; d < kMlaDim; ++d)
        acc = __builtin_fmaf(q[h * kMlaDim + d], f16_to_float(kv_b[static_cast<std::size_t>(h * 2 * kMlaDim + d) * kKvLora + l]), acc);
    qt[h * kKvLora + l] = acc;
}

constexpr unsigned kMaxTok = kIdxTopk + kKpool - 1;  // 2051
__global__ __launch_bounds__(256) void mla_attn_kernel(DsaSlot sl, const std::uint32_t* tokens, const std::uint32_t* n_ptr,
                                                       const float* qt, const std::uint16_t* kv_b, float* o) {
    __shared__ float e[kMaxTok];
    __shared__ float s[256];
    __shared__ float ob[kKvLora];
    const unsigned h = blockIdx.x, l = threadIdx.x, n = *n_ptr;
    const float* q = qt + h * kKvLora;
    float mx = -__builtin_huge_valf();
    for (unsigned t = l; t < n; t += 256) {
        const std::uint16_t* lat = sl.latent + static_cast<std::size_t>(tokens[t]) * kKvLora;
        float acc = 0.0F;
        for (unsigned i = 0; i < kKvLora; ++i) acc = __builtin_fmaf(q[i], bf16_to_float(lat[i]), acc);
        e[t] = acc * kMlaScale;
        mx = e[t] > mx ? e[t] : mx;
    }
    s[l] = mx;  // max tree (exact in any order)
    __syncthreads();
    for (unsigned hh = 128; hh > 0; hh >>= 1U) {
        if (l < hh) s[l] = s[l + hh] > s[l] ? s[l + hh] : s[l];
        __syncthreads();
    }
    const float m = s[0];
    __syncthreads();
    float a = 0.0F;
    for (unsigned t = l; t < n; t += 256) {
        e[t] = declared_exp(e[t] - m);
        a = a + e[t];
    }
    const float sum = lds_tree<256>(s, a);
    for (unsigned t = l; t < n; t += 256) e[t] = e[t] / sum;
    __syncthreads();
    for (unsigned li = 0; li < kKvLora; ++li) {  // o-bar: lane-strided fmaf over tokens, then the tree
        float b = 0.0F;
        for (unsigned t = l; t < n; t += 256)
            b = __builtin_fmaf(e[t], bf16_to_float(sl.latent[static_cast<std::size_t>(tokens[t]) * kKvLora + li]), b);
        const float v = lds_tree<256>(s, b);
        if (l == 0) ob[li] = v;
    }
    __syncthreads();
    const unsigned d = l;  // 256 lanes = 256 output dims
    float acc = 0.0F;
    for (unsigned i = 0; i < kKvLora; ++i)
        acc = __builtin_fmaf(ob[i], f16_to_float(kv_b[static_cast<std::size_t>(h * 2 * kMlaDim + kMlaDim + d) * kKvLora + i]), acc);
    o[h * kMlaDim + d] = acc;
}

}  // namespace

void dsa_append(const DsaSlot* slots, const unsigned* t, const float* latent, const float* x, const IndexerDev& w,
                float* key_scratch, float* gate_scratch, unsigned* flag, unsigned rows, hipStream_t s) {
    dense(x, w.wk, WType::F16, kHidden, 0, kIdxDim, key_scratch, rows, s);
    dense(x, w.gate, WType::F16, kHidden, 0, kIdxDim, gate_scratch, rows, s);
    dsa_store_kernel<<<rows, 128, 0, s>>>(slots, t, latent, w, key_scratch, gate_scratch, flag);
    ok(hipGetLastError(), "glm dsa_store");
}
void indexer_select(const DsaSlot& slot, unsigned t, const float* qI, const float* x, const IndexerDev& w, float* wh,
                    float* scores, std::uint32_t* work, std::uint32_t* tokens, std::uint32_t* n, hipStream_t s) {
    const unsigned P = (t + 1) / kKpool;
    if (P > 0) {
        dense(x, w.weights_proj, WType::F16, kHidden, 0, kIdxHeads, wh, 1, s);
        scale_kernel<<<1, 256, 0, s>>>(wh, kIdxHeads, kInvSqrt32);
        idx_score_kernel<<<(P + 255) / 256, 256, 0, s>>>(slot.pooled, P, qI, wh, scores, work);
        ok(hipGetLastError(), "glm idx_score");
    }
    idx_select_kernel<<<1, 1024, 0, s>>>(work, P, t, tokens, n);
    ok(hipGetLastError(), "glm idx_select");
}
void select_tokens(const std::uint32_t* keys, unsigned P, unsigned t, std::uint32_t* tokens, std::uint32_t* n, hipStream_t s) {
    idx_select_kernel<<<1, 1024, 0, s>>>(keys, P, t, tokens, n);
    ok(hipGetLastError(), "glm idx_select");
}
void mla_attention(const DsaSlot& slot, const std::uint32_t* tokens, const std::uint32_t* n, const float* q,
                   const std::uint16_t* kv_b, unsigned heads, float* qt, float* o, hipStream_t s) {
    mla_absorb_kernel<<<dim3(kKvLora / 256, heads), 256, 0, s>>>(q, kv_b, heads, qt);
    ok(hipGetLastError(), "glm mla_absorb");
    mla_attn_kernel<<<heads, 256, 0, s>>>(slot, tokens, n, qt, kv_b, o);
    ok(hipGetLastError(), "glm mla_attn");
}

}  // namespace strix::glm::dev
