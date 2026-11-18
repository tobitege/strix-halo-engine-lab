// Qwen3.8-27B dense family (`qwen35`) CUDA engine over the shared cuda_core. See include/strix/qwen35/cuda_engine.hpp.
// Home: specific (model layer composition); all arithmetic is in src/cuda_core (shared).
#include "strix/qwen35/cuda_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <unordered_map>

#include "strix/cuda_core/attention.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/gdn.hpp"
#include "strix/cuda_core/nvfp4.hpp"
#include "strix/cuda_core/norms.hpp"
#include "strix/cuda_core/dense_big.hpp"
#include "strix/cuda_core/rope.hpp"
#include "strix/cuda_core/submit_window.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/gguf_writer.hpp"
#include "strix/wave_fit.hpp"
#include "../../core/dense_linear.hpp"

namespace cc = strix::cuda_core;

#define CK(x)                                                                                                  \
    do {                                                                                                       \
        const cudaError_t e__ = (x);                                                                           \
        if (e__ != cudaSuccess) {                                                                              \
            throw std::runtime_error(std::string(#x) + " failed: " + cudaGetErrorString(e__) + " at line " + \
                                     std::to_string(__LINE__));                                                \
        }                                                                                                      \
    } while (0)

namespace strix::qwen35 {
namespace {

// ---------------------------------------------------------------- config from GGUF (until config_from_gguf lands here)
// ---------------------------------------------------------------- tensors
struct PackTensorSpec {
    std::string name;
    int type = 0;  // ggml: 30 bf16, 8 q8_0, 0 f32
    std::uint64_t in = 0, out = 1, bytes = 0, file_off = 0;
    // TURBO onfly: the source is BF16 (src_bytes) and the stored copy is NVFP4 (bytes); gscale is set at conversion.
    // TURBO converted: fp4 with src_bytes == 0: the file already holds the NVFP4 tile bytes, gscale comes from the pack's table.
    bool fp4 = false;
    std::uint64_t src_bytes = 0;
    float gscale = 1.0f;
};

// Tensors that take the NVFP4 path: 2-D BF16 weights (every projection, the MTP eh_proj, optional lm_head), never the
// embedding table (row gather) nor norms / conv / state tensors.
bool fp4_eligible(const EngineOptions& o, const PackTensorSpec& t) {
    if (o.turbo_weights != TurboWeights::onfly || t.type != 30 || t.out <= 1) return false;
    if (t.name == "token_embd.weight") return false;
    if (t.name == "output.weight" && !o.turbo_lm_head) return false;
    if (t.name.size() < 7 || t.name.compare(t.name.size() - 7, 7, ".weight") != 0) return false;
    return cc::nvfp4_shape_ok(static_cast<std::uint32_t>(t.out), static_cast<std::uint32_t>(t.in));
}
void mark_fp4(const EngineOptions& o, std::vector<PackTensorSpec>& v) {
    for (auto& t : v) {
        if (!fp4_eligible(o, t)) continue;
        t.fp4 = true;
        t.src_bytes = t.bytes;
        t.bytes = cc::nvfp4_bytes(static_cast<std::uint32_t>(t.out), static_cast<std::uint32_t>(t.in));
    }
}

// Converted pack: per-tensor global scales of the NVFP4 tensors, in table order, in one F32 tensor.
constexpr const char* kPackGscaleName = "strix.nvfp4.gscale";

std::uint64_t bytes_of(int type, std::uint64_t in, std::uint64_t out) {
    if (type == 30) return in * out * 2;
    if (type == 8) return out * (in / 32) * 34;
    return in * out * 4;
}

// Synthetic fill: counter-based RNG (splitmix64), deterministic in (seed, tensor, index).
enum class Fill : int { weight = 0, affine = 1 };
struct FillSpec {
    Fill kind = Fill::weight;
    float base = 0.0f, amp = 1.0f;  // value = base + amp * u, u in [-1, 1)
};

__device__ __forceinline__ std::uint64_t mix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
__device__ __forceinline__ float urand(std::uint64_t seed, std::uint64_t idx) {
    const std::uint64_t h = mix64(seed ^ mix64(idx));
    return static_cast<float>(static_cast<std::int32_t>(h >> 40)) * (1.0f / 8388608.0f) - 1.0f;  // exact in float: 24-bit / 2^23 - 1
}

__global__ void fill_bf16_kernel(std::uint16_t* p, std::uint64_t n, std::uint64_t seed, float base, float amp) {
    for (std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        p[i] = __bfloat16_as_ushort(__float2bfloat16_rn(base + amp * urand(seed, i)));
}
__global__ void fill_f32_kernel(float* p, std::uint64_t n, std::uint64_t seed, float base, float amp) {
    for (std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        p[i] = base + amp * urand(seed, i);
}
// q8_0: block = {f16 d, int8 q[32]}; value ~ amp * u: q = round(127 * u), d = amp / 127 (x (0.75 + 0.25 u2) per block).
__global__ void fill_q8_kernel(unsigned char* p, std::uint64_t nblocks, std::uint64_t seed, float amp) {
    for (std::uint64_t b = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x; b < nblocks;
         b += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        unsigned char* q = p + b * 34;
        const float d = amp / 127.0f * (0.75f + 0.25f * urand(seed ^ 0x5555ull, b));
        const std::uint16_t hb = __half_as_ushort(__float2half_rn(d));
        q[0] = static_cast<unsigned char>(hb & 0xFF);
        q[1] = static_cast<unsigned char>(hb >> 8);
        for (int j = 0; j < 32; ++j) q[2 + j] = static_cast<unsigned char>(static_cast<std::int8_t>(std::lrintf(127.0f * urand(seed, b * 32 + j))));
    }
}

std::uint64_t fnv(const std::string& s) {
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char ch : s) h = (h ^ ch) * 1099511628211ull;
    return h;
}

struct WeightStore {
    cc::UniqueHandle<void*, cudaFree> owner;
    unsigned char* base = nullptr;
    std::uint64_t total = 0, trunk_bytes = 0;
    std::map<std::string, PackTensorSpec> spec;
    std::map<std::string, std::uint64_t> off;
    [[nodiscard]] bool has(const std::string& n) const { return off.count(n) != 0; }
    [[nodiscard]] const void* ptr(const std::string& n) const { return base + off.at(n); }
    [[nodiscard]] cc::DenseWeight dense(const std::string& n) const {
        const auto& t = spec.at(n);
        cc::DenseWeight w;
        w.data = ptr(n);
        w.in = static_cast<std::uint32_t>(t.in);
        w.out = static_cast<std::uint32_t>(t.out);
        if (t.fp4) {
            w.format = cc::WeightFormat::nvfp4;
            w.gscale = t.gscale;
            return w;
        }
        switch (t.type) {
            case 30: w.format = cc::WeightFormat::bf16; break;
            case 8: w.format = cc::WeightFormat::q8_0; break;
            case 0: w.format = cc::WeightFormat::f32; break;
            default: throw std::runtime_error("unsupported ggml type for " + n);
        }
        return w;
    }
    [[nodiscard]] const float* f32(const std::string& n) const {
        if (spec.at(n).type != 0) throw std::runtime_error(n + " must be F32");
        return static_cast<const float*>(ptr(n));
    }
};

void layout(WeightStore& W, const std::vector<PackTensorSpec>& specs, std::uint32_t layers) {
    std::uint64_t total = 0;
    for (const auto& t : specs) {
        W.spec[t.name] = t;
        W.off[t.name] = total;
        core::require_capacity(t.bytes, 255, std::numeric_limits<std::size_t>::max());
        const auto padded = (t.bytes + 255) & ~255ULL;
        core::require_capacity(total, padded, std::numeric_limits<std::size_t>::max());
        total += padded;
        const bool mtp = t.name.rfind("blk." + std::to_string(layers) + ".", 0) == 0;
        if (t.name != "token_embd.weight" && !mtp) W.trunk_bytes += t.bytes;
    }
    W.total = total;
    CK(cudaMalloc(W.owner.put(), total));
    W.base = static_cast<unsigned char*>(W.owner.get());
}

std::vector<PackTensorSpec> specs_from_gguf(const gguf::FileInfo& f, std::uint32_t layers, bool load_mtp) {
    std::vector<PackTensorSpec> v;
    for (const auto& t : f.tensors) {
        const std::string pre = "blk." + std::to_string(layers) + ".";
        if (t.name.rfind(pre, 0) == 0 && !load_mtp) continue;
        if (t.name.rfind("blk.", 0) == 0) {
            const std::uint32_t idx = static_cast<std::uint32_t>(std::stoul(t.name.substr(4)));
            if (idx > layers) continue;
        }
        PackTensorSpec s;
        s.name = t.name;
        s.type = static_cast<int>(t.ggml_type);
        s.in = t.dimensions.at(0);
        s.out = t.dimensions.size() > 1 ? t.dimensions[1] : 1;
        s.bytes = t.byte_size;
        s.file_off = t.absolute_offset;
        v.push_back(s);
    }
    return v;
}

// Tensor list of the qwen35 contract (design section 2) plus the fill rule of each tensor.
std::vector<std::pair<PackTensorSpec, FillSpec>> synth_specs(const Config& c, PackFormat pack, bool with_mtp) {
    std::vector<std::pair<PackTensorSpec, FillSpec>> v;
    const int wt = pack == PackFormat::bf16 ? 30 : 8;
    auto W = [&](const std::string& n, std::uint64_t in, std::uint64_t out, float gain) {
        PackTensorSpec s;
        s.name = n;
        s.type = wt;
        s.in = in;
        s.out = out;
        s.bytes = bytes_of(wt, in, out);
        v.push_back({s, FillSpec{Fill::weight, 0.0f, gain * std::sqrt(3.0f / static_cast<float>(in))}});
    };
    auto F = [&](const std::string& n, std::uint64_t cnt, std::uint64_t cols, float base, float amp) {
        PackTensorSpec s;
        s.name = n;
        s.type = 0;
        s.in = cols;
        s.out = cnt;
        s.bytes = cnt * cols * 4;
        v.push_back({s, FillSpec{Fill::affine, base, amp}});
    };
    const std::uint64_t H = c.hidden;
    auto attn_block = [&](const std::string& p) {
        W(p + "attn_q.weight", H, c.q_gate_width(), 1.0f);
        W(p + "attn_k.weight", H, c.kv_width(), 1.0f);
        W(p + "attn_v.weight", H, c.kv_width(), 1.0f);
        W(p + "attn_output.weight", c.heads * c.head_dim, H, 0.5f);
        F(p + "attn_q_norm.weight", 1, c.head_dim, 1.0f, 0.1f);
        F(p + "attn_k_norm.weight", 1, c.head_dim, 1.0f, 0.1f);
    };
    auto ffn_block = [&](const std::string& p) {
        W(p + "ffn_gate.weight", H, c.intermediate, 1.0f);
        W(p + "ffn_up.weight", H, c.intermediate, 1.0f);
        W(p + "ffn_down.weight", c.intermediate, H, 0.5f);
    };
    W("token_embd.weight", H, c.vocab, 1.0f / std::sqrt(3.0f / static_cast<float>(H)));
    F("output_norm.weight", 1, H, 1.0f, 0.1f);
    W("output.weight", H, c.vocab, 1.0f);
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        F(p + "attn_norm.weight", 1, H, 1.0f, 0.1f);
        F(p + "post_attention_norm.weight", 1, H, 1.0f, 0.1f);
        if (c.kind(l) == LayerKind::gated_delta_net) {
            W(p + "attn_qkv.weight", H, c.gdn_conv_channels(), 1.0f);
            W(p + "attn_gate.weight", H, c.gdn_v_width(), 1.0f);
            W(p + "ssm_beta.weight", H, c.gdn_value_heads, 1.0f);
            W(p + "ssm_alpha.weight", H, c.gdn_value_heads, 1.0f);
            F(p + "ssm_conv1d.weight", c.gdn_conv_channels(), c.gdn_conv_kernel, 0.0f, 0.5f);
            F(p + "ssm_a", 1, c.gdn_value_heads, -1.0f, 0.5f);
            F(p + "ssm_dt.bias", 1, c.gdn_value_heads, 0.0f, 0.5f);
            F(p + "ssm_norm.weight", 1, c.gdn_value_dim, 1.0f, 0.1f);
            W(p + "ssm_out.weight", c.gdn_v_width(), H, 0.5f);
        } else {
            attn_block(p);
        }
        ffn_block(p);
    }
    if (with_mtp && c.mtp_layers > 0) {
        const std::string p = "blk." + std::to_string(c.layers) + ".";
        F(p + "attn_norm.weight", 1, H, 1.0f, 0.1f);
        F(p + "post_attention_norm.weight", 1, H, 1.0f, 0.1f);
        F(p + "nextn.enorm.weight", 1, H, 1.0f, 0.1f);
        F(p + "nextn.hnorm.weight", 1, H, 1.0f, 0.1f);
        W(p + "nextn.eh_proj.weight", 2 * H, H, 1.0f);
        F(p + "nextn.shared_head_norm.weight", 1, H, 1.0f, 0.1f);
        attn_block(p);
        ffn_block(p);
    }
    return v;
}

struct LayerW {
    bool attn = false;
    cc::DenseWeight qkv, gate, beta, alpha, out;  // GDN (out = ssm_out)
    cc::DenseWeight q, k, v, o;                    // attention
    cc::DenseWeight ffn_gate, ffn_up, ffn_down;
    const float *attn_norm = nullptr, *post_norm = nullptr;
    const float *conv = nullptr, *ssm_a = nullptr, *dt = nullptr, *ssm_norm = nullptr, *qn = nullptr, *kn = nullptr;
};

struct InputPack {
    std::uint32_t pos0;
    std::uint32_t ids[cc::kMaxRows];
};
__global__ void set_inputs_kernel(InputPack in, std::uint32_t* pos, std::uint32_t* ids) {
    if (threadIdx.x == 0) *pos = in.pos0;
    if (threadIdx.x < cc::kMaxRows) ids[threadIdx.x] = in.ids[threadIdx.x];
}
__global__ void zero_u32_kernel(std::uint32_t* p) { *p = 0; }

// Multi-session batch tables (one small by-value pack, scattered to device arrays by one kernel before the graph).
struct SlotPack {
    std::uint32_t ids[cc::kMaxRows], pos[cc::kMaxRows], slot[cc::kMaxRows];
    std::uint32_t g_row0[cc::kMaxRows], g_rows[cc::kMaxRows], g_slot[cc::kMaxRows];
};
struct SlotDev {
    std::uint32_t *ids, *pos, *slot, *g_row0, *g_rows, *g_slot;
};
__global__ void set_slot_inputs_kernel(SlotPack in, SlotDev d) {
    const std::uint32_t i = threadIdx.x;
    if (i < cc::kMaxRows) {
        d.ids[i] = in.ids[i];
        d.pos[i] = in.pos[i];
        d.slot[i] = in.slot[i];
        d.g_row0[i] = in.g_row0[i];
        d.g_rows[i] = in.g_rows[i];
        d.g_slot[i] = in.g_slot[i];
    }
}

}  // namespace

// ================================================================ Impl
struct CudaEngine::Impl {
    Config c;
    EngineOptions opt;
    WeightStore W;
    std::vector<LayerW> L;
    LayerW M;  // MTP block
    bool mtp_ok = false;
    cc::DenseWeight head, embd;
    const float* out_norm = nullptr;
    const float *m_enorm = nullptr, *m_hnorm = nullptr, *m_shnorm = nullptr;
    cc::DenseWeight m_eh;

    static constexpr std::uint32_t RM = cc::kMaxRows;
    int device = 0;
    std::vector<cc::UniqueHandle<void*, cudaFree>> allocations;
    cudaStream_t s = nullptr;
    cc::GdnDims gd;
    cc::AttentionDims ad;

    // activations
    float *x = nullptr, *xn = nullptr, *z = nullptr, *gout = nullptr, *qg = nullptr, *q = nullptr, *k = nullptr, *v = nullptr,
          *ao = nullptr, *hid = nullptr, *logits = nullptr, *scratch = nullptr, *cos_t = nullptr, *sin_t = nullptr;
    float *mx = nullptr, *mxn = nullptr, *mcat = nullptr, *mtmp = nullptr, *mlogits = nullptr;
    std::uint32_t *d_ids = nullptr, *d_pos = nullptr, *d_top = nullptr, *d_parity = nullptr;
    void* d_argws = nullptr;
    std::uint32_t *d_mids = nullptr, *d_mpos = nullptr, *d_mtop = nullptr;
    // per-layer state / window buffers
    std::vector<float*> conv, delta, raw, qkvc, ar, br;
    std::vector<__nv_bfloat16*> kc, vc;
    __nv_bfloat16 *mkc = nullptr, *mvc = nullptr;
    std::vector<cc::GdnCommitLayer> commit_layers;

    std::uint32_t committed = 0, last_rows = 0;
    std::vector<cudaEvent_t> last_chunk_ends;  // prefill(): end event of each queued chunk (SubmitWindow input)
    std::uint32_t last_window_waits = 0;       // host waits the bounded submission issued in the last prefill
    bool graphs_on = true;
    struct Graph {
        cc::UniqueHandle<cudaGraphExec_t, cudaGraphExecDestroy> exec;
        std::uint64_t nodes = 0;
    };
    std::unordered_map<std::uint32_t, Graph> graphs;
    std::uint64_t last_nodes = 0;

    template <class T>
    T* alloc(std::size_t n, bool zero = false) {
        cc::UniqueHandle<void*, cudaFree> owner;
        CK(cudaMalloc(owner.put(), core::checked_product({n, sizeof(T)})));
        T* p = static_cast<T*>(owner.get());
        allocations.push_back(std::move(owner));
        if (zero) CK(cudaMemset(p, 0, core::checked_product({n, sizeof(T)})));
        return p;
    }
    [[nodiscard]] std::string nm(std::uint32_t l, const char* suffix) const { return "blk." + std::to_string(l) + "." + suffix; }

    LayerW resolve(std::uint32_t l, bool attn) const {
        LayerW w;
        w.attn = attn;
        w.attn_norm = W.f32(nm(l, "attn_norm.weight"));
        w.post_norm = W.f32(nm(l, "post_attention_norm.weight"));
        if (!attn) {
            w.qkv = W.dense(nm(l, "attn_qkv.weight"));
            w.gate = W.dense(nm(l, "attn_gate.weight"));
            w.beta = W.dense(nm(l, "ssm_beta.weight"));
            w.alpha = W.dense(nm(l, "ssm_alpha.weight"));
            w.out = W.dense(nm(l, "ssm_out.weight"));
            w.conv = W.f32(nm(l, "ssm_conv1d.weight"));
            w.ssm_a = W.f32(nm(l, "ssm_a"));
            w.dt = W.f32(nm(l, "ssm_dt.bias"));
            w.ssm_norm = W.f32(nm(l, "ssm_norm.weight"));
        } else {
            w.q = W.dense(nm(l, "attn_q.weight"));
            w.k = W.dense(nm(l, "attn_k.weight"));
            w.v = W.dense(nm(l, "attn_v.weight"));
            w.o = W.dense(nm(l, "attn_output.weight"));
            w.qn = W.f32(nm(l, "attn_q_norm.weight"));
            w.kn = W.f32(nm(l, "attn_k_norm.weight"));
        }
        w.ffn_gate = W.dense(nm(l, "ffn_gate.weight"));
        w.ffn_up = W.dense(nm(l, "ffn_up.weight"));
        w.ffn_down = W.dense(nm(l, "ffn_down.weight"));
        return w;
    }

    void init() {
        cc::set_programmatic_launch(opt.pdl);
        gd = {c.gdn_key_heads, c.gdn_key_dim, c.gdn_value_heads, c.gdn_value_dim, c.gdn_conv_kernel};
        ad.heads = c.heads;
        ad.kv_heads = c.kv_heads;
        ad.head_dim = c.head_dim;
        ad.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(c.head_dim)));
        ad.max_positions = opt.max_positions;
        CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        {   // NVFP4 split-K workspace: grow-only, must exist before graph capture. Bound = rows 32, all jobs of the largest layer.
            std::size_t need = 0, layer_sum = 0;
            for (const auto& [name, t] : W.spec) {
                if (!t.fp4) continue;
                const std::size_t ks = cc::nvfp4_split_k(static_cast<std::uint32_t>(t.out), static_cast<std::uint32_t>(t.in));
                const std::size_t b = 2 * ks * cc::kNvfp4MaxRows * t.out * 4;
                layer_sum += b;
                need = std::max(need, b);
            }
            if (layer_sum) CK(cc::nvfp4_reserve_workspace(std::min<std::size_t>(layer_sum, std::max<std::size_t>(need * 4, 64u << 20)), s));
        }
        for (std::uint32_t l = 0; l < c.layers; ++l) L.push_back(resolve(l, c.kind(l) == LayerKind::full_attention));
        head = W.has("output.weight") ? W.dense("output.weight") : W.dense("token_embd.weight");
        embd = W.dense("token_embd.weight");
        out_norm = W.f32("output_norm.weight");
        mtp_ok = opt.load_mtp && c.mtp_layers > 0 && W.has(nm(c.layers, "nextn.eh_proj.weight"));
        if (mtp_ok) {
            M = resolve(c.layers, true);
            m_enorm = W.f32(nm(c.layers, "nextn.enorm.weight"));
            m_hnorm = W.f32(nm(c.layers, "nextn.hnorm.weight"));
            m_shnorm = W.f32(nm(c.layers, "nextn.shared_head_norm.weight"));
            m_eh = W.dense(nm(c.layers, "nextn.eh_proj.weight"));
        }
        const std::uint32_t H = c.hidden;
        x = alloc<float>(RM * H);
        xn = alloc<float>(RM * H);
        z = alloc<float>(RM * gd.v_width());
        gout = alloc<float>(RM * gd.v_width());
        qg = alloc<float>(static_cast<std::size_t>(RM) * c.q_gate_width());
        q = alloc<float>(static_cast<std::size_t>(RM) * c.heads * c.head_dim);
        k = alloc<float>(RM * c.kv_width());
        v = alloc<float>(RM * c.kv_width());
        ao = alloc<float>(static_cast<std::size_t>(RM) * c.heads * c.head_dim);
        hid = alloc<float>(static_cast<std::size_t>(RM) * c.intermediate);
        logits = alloc<float>(static_cast<std::size_t>(RM) * c.vocab);
        scratch = alloc<float>(cc::attention_scratch_floats(ad, RM));
        d_ids = alloc<std::uint32_t>(RM);
        d_pos = alloc<std::uint32_t>(1);
        d_top = alloc<std::uint32_t>(RM);
        d_parity = alloc<std::uint32_t>(1, true);
        d_argws = alloc<unsigned char>(cc::kArgmaxWorkspaceBytes);
        if (mtp_ok) {
            mx = alloc<float>(RM * H);
            mxn = alloc<float>(RM * H);
            mcat = alloc<float>(2 * RM * H);
            mtmp = alloc<float>(RM * H);
            mlogits = alloc<float>(static_cast<std::size_t>(RM) * c.vocab);
            d_mids = alloc<std::uint32_t>(RM);
            d_mpos = alloc<std::uint32_t>(1);
            d_mtop = alloc<std::uint32_t>(RM);
            const std::size_t n = static_cast<std::size_t>(c.kv_heads) * opt.max_positions * c.head_dim;
            mkc = alloc<__nv_bfloat16>(n, true);
            mvc = alloc<__nv_bfloat16>(n, true);
        }
        std::vector<float> ct, st;
        cc::rope_table_host(opt.max_positions, c.rotary_dim, c.rope_theta, ct, st);
        cos_t = alloc<float>(ct.size());
        sin_t = alloc<float>(st.size());
        CK(cudaMemcpy(cos_t, ct.data(), ct.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(sin_t, st.data(), st.size() * 4, cudaMemcpyHostToDevice));
        const std::size_t kvn = static_cast<std::size_t>(c.kv_heads) * opt.max_positions * c.head_dim;
        const std::size_t dside = static_cast<std::size_t>(gd.value_heads) * gd.key_dim * gd.value_dim;
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            if (L[l].attn) {
                kc.push_back(alloc<__nv_bfloat16>(kvn, true));
                vc.push_back(alloc<__nv_bfloat16>(kvn, true));
                conv.push_back(nullptr);
                delta.push_back(nullptr);
                raw.push_back(nullptr);
                qkvc.push_back(nullptr);
                ar.push_back(nullptr);
                br.push_back(nullptr);
            } else {
                kc.push_back(nullptr);
                vc.push_back(nullptr);
                conv.push_back(alloc<float>(static_cast<std::size_t>(gd.conv_channels()) * 3, true));
                delta.push_back(alloc<float>(2 * dside, true));
                raw.push_back(alloc<float>(static_cast<std::size_t>(RM) * gd.conv_channels()));
                qkvc.push_back(alloc<float>(static_cast<std::size_t>(RM) * gd.conv_channels()));
                ar.push_back(alloc<float>(RM * gd.value_heads));
                br.push_back(alloc<float>(RM * gd.value_heads));
                commit_layers.push_back({{L[l].conv, L[l].ssm_a, L[l].dt, L[l].ssm_norm}, {conv[l], delta[l]},
                                         {raw[l], qkvc[l], ar[l], br[l]}});
            }
        }
        CK(cudaDeviceSynchronize());
    }

    void reset() {
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            if (L[l].attn) continue;
            CK(cudaMemsetAsync(conv[l], 0, static_cast<std::size_t>(gd.conv_channels()) * 3 * 4, s));
            CK(cudaMemsetAsync(delta[l], 0, 2 * static_cast<std::size_t>(gd.value_heads) * gd.key_dim * gd.value_dim * 4, s));
        }
        CK(cudaMemsetAsync(d_parity, 0, 4, s));
        CK(cudaStreamSynchronize(s));
        committed = 0;
        last_rows = 0;
    }

    // ---- one attention layer body (shared by the trunk and the MTP block): mixer + MLP on rows of xa (residual stream).
    struct Bufs {
        float *x, *xn;
        const std::uint32_t* pos;
        __nv_bfloat16 *kc, *vc;
        cc::RowMap rm{};
    };

    // GEMVs that share x: one launch when format and in agree (bitwise equal either way), else separate launches.
    void multi(const cc::DenseWeight* ws, float* const* ys, int n, const float* xin, std::uint32_t rows) {
        bool ok = true;
        for (int j = 1; j < n; ++j) ok = ok && ws[j].format == ws[0].format && ws[j].in == ws[0].in;
        if (ok && opt.multi_gemv) {
            CK(cc::dense_rows_multi(ws, ys, n, xin, rows, nullptr, s, opt.dense_arithmetic));
        } else {
            for (int j = 0; j < n; ++j) CK(cc::dense_rows(ws[j], xin, ys[j], rows, nullptr, s, opt.dense_arithmetic));
        }
    }

    void attn_mixer(const LayerW& w, const Bufs& b, std::uint32_t rows) {
        const std::uint32_t D = c.head_dim;
        const std::uint32_t qgw = c.heads * D * 2;
        {
            const cc::DenseWeight ws[3] = {w.q, w.k, w.v};
            float* const ys[3] = {qg, k, v};
            multi(ws, ys, 3, b.xn, rows);
        }
        const cc::GroupedView qsrc{rows, c.heads, D, qgw, 2 * D}, qdst{rows, c.heads, D, c.heads * D, D};
        const cc::GroupedView kvv{rows, c.kv_heads, D, c.kv_heads * D, D};
        CK(cc::rmsnorm(qg, qsrc, w.qn, q, qdst, c.rms_eps, s));
        CK(cc::rmsnorm(k, kvv, w.kn, k, kvv, c.rms_eps, s));
        CK(cc::rope_apply(q, qdst, c.rotary_dim, cos_t, sin_t, b.pos, s, b.rm));
        CK(cc::rope_apply(k, kvv, c.rotary_dim, cos_t, sin_t, b.pos, s, b.rm));
        CK(cc::kv_store(ad, k, v, b.kc, b.vc, b.pos, rows, s, b.rm));
        CK(cc::attention_decode(ad, q, qdst, qg + D, qsrc, b.kc, b.vc, b.pos, rows, ao, scratch, s, b.rm));
        CK(cc::dense_rows(w.o, ao, b.x, rows, b.x, s, opt.dense_arithmetic));
    }

    void mlp(const LayerW& w, const Bufs& b, std::uint32_t rows) {
        const cc::GroupedView hv{rows, 1, c.hidden, c.hidden, c.hidden};
        CK(cc::rmsnorm(b.x, hv, w.post_norm, b.xn, hv, c.rms_eps, s));
        CK(cc::dense_gate_up_silu_rows(w.ffn_gate, w.ffn_up, b.xn, hid, rows, s, opt.dense_arithmetic));
        CK(cc::dense_rows(w.ffn_down, hid, b.x, rows, b.x, s, opt.dense_arithmetic));
    }

    void trunk(std::uint32_t rows, LogitRows want) {
        const std::uint32_t H = c.hidden;
        const cc::GroupedView hv{rows, 1, H, H, H};
        CK(cc::embedding_rows(embd, d_ids, x, rows, s));
        std::uint32_t gi = 0;
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            const LayerW& w = L[l];
            CK(cc::rmsnorm(x, hv, w.attn_norm, xn, hv, c.rms_eps, s));
            if (!w.attn) {
                {
                    const cc::DenseWeight ws[4] = {w.qkv, w.gate, w.beta, w.alpha};
                    float* const ys[4] = {raw[l], z, br[l], ar[l]};
                    multi(ws, ys, 4, xn, rows);
                }
                const auto& cl = commit_layers[gi++];
                CK(cc::gdn_window(gd, cl.layer, cl.state, cl.win, z, gout, rows, c.rms_eps, d_parity, s));
                CK(cc::dense_rows(w.out, gout, x, rows, x, s, opt.dense_arithmetic));
            } else {
                attn_mixer(w, Bufs{x, xn, d_pos, kc[l], vc[l]}, rows);
            }
            mlp(w, Bufs{x, xn, d_pos, nullptr, nullptr}, rows);
        }
        CK(cc::rmsnorm(x, hv, out_norm, xn, hv, c.rms_eps, s));
        if (want == LogitRows::all) {
            CK(cc::dense_rows(head, xn, logits, rows, nullptr, s, opt.dense_arithmetic));
            CK(cc::argmax_rows(logits, c.vocab, rows, d_top, d_argws, s));
        } else if (want == LogitRows::last) {
            CK(cc::dense_rows(head, xn + static_cast<std::size_t>(rows - 1) * H, logits, 1, nullptr, s, opt.dense_arithmetic));
            CK(cc::argmax_rows(logits, c.vocab, 1, d_top, d_argws, s));
        }
    }

    RowsOut forward(const std::uint32_t* tokens, std::uint32_t rows, LogitRows want) {
        if (rows == 0 || rows > RM) throw std::runtime_error("forward_rows: rows must be 1..32");
        core::require_capacity(committed, rows, opt.max_positions);
        core::validate_token_ids(tokens, rows, c.vocab);
        InputPack in{};
        in.pos0 = committed;
        for (std::uint32_t i = 0; i < rows; ++i) in.ids[i] = tokens[i];
        set_inputs_kernel<<<1, 32, 0, s>>>(in, d_pos, d_ids);
        CK(cudaGetLastError());
        const std::uint32_t key = rows * 4 + static_cast<std::uint32_t>(want);
        if (graphs_on) {
            auto it = graphs.find(key);
            if (it == graphs.end()) {
                cc::UniqueHandle<cudaGraph_t, cudaGraphDestroy> g;
                CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                try {
                    trunk(rows, want);
                } catch (...) {
                    cudaGraph_t dummy = nullptr;
                    cudaStreamEndCapture(s, &dummy);
                    if (dummy) cudaGraphDestroy(dummy);
                    throw;
                }
                CK(cudaStreamEndCapture(s, g.put()));
                Graph gr;
                CK(cudaGraphInstantiate(gr.exec.put(), g.get(), 0));
                std::size_t n = 0;
                CK(cudaGraphGetNodes(g.get(), nullptr, &n));
                gr.nodes = n;

                it = graphs.emplace(key, std::move(gr)).first;
            }
            last_nodes = it->second.nodes;
            CK(cudaGraphLaunch(it->second.exec.get(), s));
        } else {
            trunk(rows, want);
            last_nodes = 0;
        }
        last_rows = rows;
        RowsOut o;
        o.logits = want == LogitRows::none ? nullptr : logits;
        o.top = want == LogitRows::none ? nullptr : d_top;
        o.hidden_post = xn;
        o.hidden_pre = x;
        return o;
    }

    void commit_rows(std::uint32_t kk) {
        if (kk > last_rows) throw std::runtime_error("commit: k exceeds the last window");
        if (kk > 0) {
            CK(cc::gdn_commit(gd, commit_layers.data(), static_cast<std::uint32_t>(commit_layers.size()), last_rows, kk, d_parity, s));
        }
        committed += kk;
        last_rows = 0;
    }

    // ================= multi-session (slots)
    struct SlotState {
        std::uint32_t n = 0;
        std::vector<__nv_bfloat16*> kc, vc;  // per layer [slot][kvh][maxpos][D] (attention layers)
        std::vector<float*> conv, delta;     // per layer [slot][C*3], [slot][2][side] (GDN layers)
        std::uint32_t* parity = nullptr;     // [slot]
        SlotDev dev{};
        std::uint32_t *accept = nullptr;
        std::vector<std::uint32_t> len;
        std::vector<cc::GdnCommitLayer> cl;
        std::unordered_map<std::uint32_t, Graph> graphs;
        std::uint32_t last_groups = 0, last_rows = 0;
        std::vector<SlotRows> last;
        std::size_t kv_stride = 0, conv_stride = 0, delta_stride = 0;
    } ss;

    ~Impl() {
        int previous = device;
        (void)cudaGetDevice(&previous);
        (void)cudaSetDevice(device);
        if (s) (void)cudaStreamSynchronize(s);
        graphs.clear();
        ss.graphs.clear();
        if (s) cc::nvfp4_release_workspace(s);
        pf.sl.clear();  // drain slot streams before freeing shared model/state buffers
        allocations.clear();
        W.owner.reset();
        if (s) (void)cudaStreamDestroy(s);
        (void)cudaSetDevice(previous);
    }

    cc::GdnSlotTable slot_table() const {
        cc::GdnSlotTable t;
        t.row0 = ss.dev.g_row0;
        t.nrows = ss.dev.g_rows;
        t.state_slot = ss.dev.g_slot;
        t.parity = ss.parity;
        t.conv_stride = ss.conv_stride;
        t.delta_stride = ss.delta_stride;
        return t;
    }

    void configure_slots(std::uint32_t n) {
        if (n == 0) throw std::runtime_error("configure_slots: n must be >= 1");
        if (ss.n != 0) throw std::runtime_error("configure_slots: already configured");
        const auto mark = allocations.size();
        try {
        ss.n = n;
        ss.kv_stride = static_cast<std::size_t>(c.kv_heads) * opt.max_positions * c.head_dim;
        ss.conv_stride = static_cast<std::size_t>(gd.conv_channels()) * 3;
        ss.delta_stride = 2 * static_cast<std::size_t>(gd.value_heads) * gd.key_dim * gd.value_dim;
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            if (L[l].attn) {
                ss.kc.push_back(alloc<__nv_bfloat16>(core::checked_product({ss.kv_stride, n}), true));
                ss.vc.push_back(alloc<__nv_bfloat16>(core::checked_product({ss.kv_stride, n}), true));
                ss.conv.push_back(nullptr);
                ss.delta.push_back(nullptr);
            } else {
                ss.kc.push_back(nullptr);
                ss.vc.push_back(nullptr);
                ss.conv.push_back(alloc<float>(core::checked_product({ss.conv_stride, n}), true));
                ss.delta.push_back(alloc<float>(core::checked_product({ss.delta_stride, n}), true));
            }
        }
        std::uint32_t gi = 0;
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            if (L[l].attn) continue;
            auto cl = commit_layers[gi++];
            cl.state = {ss.conv[l], ss.delta[l]};
            ss.cl.push_back(cl);
        }
        ss.parity = alloc<std::uint32_t>(n, true);
        ss.accept = alloc<std::uint32_t>(RM, true);
        ss.dev.ids = d_ids;
        ss.dev.pos = alloc<std::uint32_t>(RM);
        ss.dev.slot = alloc<std::uint32_t>(RM);
        ss.dev.g_row0 = alloc<std::uint32_t>(RM, true);
        ss.dev.g_rows = alloc<std::uint32_t>(RM, true);
        ss.dev.g_slot = alloc<std::uint32_t>(RM, true);
        ss.len.assign(n, 0);
        CK(cudaDeviceSynchronize());
        } catch (...) {
            (void)cudaStreamSynchronize(s);
            ss = SlotState{};
            allocations.resize(mark);
            throw;
        }
    }

    void check_slot(std::uint32_t slot) const {
        if (slot >= ss.n) throw std::runtime_error("slot out of range (configure_slots first)");
    }

    void reset_slot(std::uint32_t slot) {
        check_slot(slot);
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            if (L[l].attn) continue;
            CK(cudaMemsetAsync(ss.conv[l] + slot * ss.conv_stride, 0, ss.conv_stride * 4, s));
            CK(cudaMemsetAsync(ss.delta[l] + slot * ss.delta_stride, 0, ss.delta_stride * 4, s));
        }
        CK(cudaMemsetAsync(ss.parity + slot, 0, 4, s));
        CK(cudaStreamSynchronize(s));
        ss.len[slot] = 0;
    }

    void copy_slot(std::uint32_t dst, std::uint32_t src) {
        check_slot(dst);
        check_slot(src);
        if (dst == src) return;
        const std::size_t pitch = static_cast<std::size_t>(opt.max_positions) * c.head_dim * sizeof(__nv_bfloat16);
        const std::size_t width = static_cast<std::size_t>(ss.len[src]) * c.head_dim * sizeof(__nv_bfloat16);
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            if (L[l].attn) {
                if (width) {
                    CK(cudaMemcpy2DAsync(ss.kc[l] + dst * ss.kv_stride, pitch, ss.kc[l] + src * ss.kv_stride, pitch, width, c.kv_heads,
                                         cudaMemcpyDeviceToDevice, s));
                    CK(cudaMemcpy2DAsync(ss.vc[l] + dst * ss.kv_stride, pitch, ss.vc[l] + src * ss.kv_stride, pitch, width, c.kv_heads,
                                         cudaMemcpyDeviceToDevice, s));
                }
            } else {
                CK(cudaMemcpyAsync(ss.conv[l] + dst * ss.conv_stride, ss.conv[l] + src * ss.conv_stride, ss.conv_stride * 4,
                                   cudaMemcpyDeviceToDevice, s));
                CK(cudaMemcpyAsync(ss.delta[l] + dst * ss.delta_stride, ss.delta[l] + src * ss.delta_stride, ss.delta_stride * 4,
                                   cudaMemcpyDeviceToDevice, s));
            }
        }
        CK(cudaMemcpyAsync(ss.parity + dst, ss.parity + src, 4, cudaMemcpyDeviceToDevice, s));
        CK(cudaStreamSynchronize(s));
        ss.len[dst] = ss.len[src];
    }

    void trunk_slots(std::uint32_t rows, std::uint32_t groups, LogitRows want, bool auto_commit) {
        const std::uint32_t H = c.hidden;
        const cc::GroupedView hv{rows, 1, H, H, H};
        const cc::RowMap rm{ss.dev.pos, ss.dev.slot, ss.kv_stride};
        const cc::GdnSlotTable tab = slot_table();
        CK(cc::embedding_rows(embd, d_ids, x, rows, s));
        std::uint32_t gi = 0;
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            const LayerW& w = L[l];
            CK(cc::rmsnorm(x, hv, w.attn_norm, xn, hv, c.rms_eps, s));
            if (!w.attn) {
                {
                    const cc::DenseWeight ws[4] = {w.qkv, w.gate, w.beta, w.alpha};
                    float* const ys[4] = {raw[l], z, br[l], ar[l]};
                    multi(ws, ys, 4, xn, rows);
                }
                const auto& cl = ss.cl[gi++];
                CK(cc::gdn_window_slots(gd, cl.layer, cl.state, cl.win, z, gout, groups, tab, c.rms_eps, s));
                CK(cc::dense_rows(w.out, gout, x, rows, x, s, opt.dense_arithmetic));
            } else {
                attn_mixer(w, Bufs{x, xn, nullptr, ss.kc[l], ss.vc[l], rm}, rows);
            }
            mlp(w, Bufs{x, xn, nullptr, nullptr, nullptr}, rows);
        }
        CK(cc::rmsnorm(x, hv, out_norm, xn, hv, c.rms_eps, s));
        if (want != LogitRows::none) {
            CK(cc::dense_rows(head, xn, logits, rows, nullptr, s, opt.dense_arithmetic));
            CK(cc::argmax_rows(logits, c.vocab, rows, d_top, d_argws, s));
        }
        if (auto_commit) CK(cc::gdn_commit_slots(gd, ss.cl.data(), static_cast<std::uint32_t>(ss.cl.size()), groups, tab, nullptr, false, s));
    }

    RowsOut forward_slots(const SlotRows* g, std::uint32_t n, const std::uint32_t* tokens, LogitRows want, bool auto_commit) {
        if (ss.n == 0) throw std::runtime_error("forward_slots: configure_slots first");
        if (n == 0 || n > RM) throw std::runtime_error("forward_slots: 1..32 groups");
        SlotPack in{};
        std::uint32_t rows = 0;
        std::vector<bool> seen(ss.n, false);
        for (std::uint32_t i = 0; i < n; ++i) {
            check_slot(g[i].slot);
            if (g[i].rows == 0 || g[i].rows > RM - rows) throw std::runtime_error("forward_slots: rows must total 1..32");
            if (seen[g[i].slot]) throw std::runtime_error("forward_slots: a slot may appear once per batch");
            seen[g[i].slot] = true;
            core::require_capacity(ss.len[g[i].slot], g[i].rows, opt.max_positions);
            core::validate_token_ids(tokens ? tokens + rows : nullptr, g[i].rows, c.vocab);
            in.g_row0[i] = rows;
            in.g_rows[i] = g[i].rows;
            in.g_slot[i] = g[i].slot;
            for (std::uint32_t r = 0; r < g[i].rows; ++r, ++rows) {
                in.ids[rows] = tokens[rows];
                in.pos[rows] = ss.len[g[i].slot] + r;
                in.slot[rows] = g[i].slot;
            }
        }
        set_slot_inputs_kernel<<<1, 32, 0, s>>>(in, ss.dev);
        CK(cudaGetLastError());
        const std::uint32_t key = rows | (n << 8) | (static_cast<std::uint32_t>(want) << 16) | (auto_commit ? 1u << 20 : 0u);
        if (graphs_on) {
            auto it = ss.graphs.find(key);
            if (it == ss.graphs.end()) {
                cc::UniqueHandle<cudaGraph_t, cudaGraphDestroy> gr;
                CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                try {
                    trunk_slots(rows, n, want, auto_commit);
                } catch (...) {
                    cudaGraph_t dummy = nullptr;
                    cudaStreamEndCapture(s, &dummy);
                    if (dummy) cudaGraphDestroy(dummy);
                    throw;
                }
                CK(cudaStreamEndCapture(s, gr.put()));
                Graph ge;
                CK(cudaGraphInstantiate(ge.exec.put(), gr.get(), 0));
                std::size_t nn = 0;
                CK(cudaGraphGetNodes(gr.get(), nullptr, &nn));
                ge.nodes = nn;

                it = ss.graphs.emplace(key, std::move(ge)).first;
            }
            last_nodes = it->second.nodes;
            CK(cudaGraphLaunch(it->second.exec.get(), s));
        } else {
            trunk_slots(rows, n, want, auto_commit);
            last_nodes = 0;
        }
        if (auto_commit) {
            for (std::uint32_t i = 0; i < n; ++i) ss.len[g[i].slot] += g[i].rows;
            ss.last_groups = 0;
        } else {
            ss.last.assign(g, g + n);
            ss.last_groups = n;
            ss.last_rows = rows;
        }
        RowsOut o;
        o.logits = want == LogitRows::none ? nullptr : logits;
        o.top = want == LogitRows::none ? nullptr : d_top;
        o.hidden_post = xn;
        o.hidden_pre = x;
        return o;
    }

    void commit_slots(const std::uint32_t* accept) {
        if (ss.last_groups == 0) throw std::runtime_error("commit_slots: no pending window");
        bool partial = false;
        std::uint32_t any = 0;
        for (std::uint32_t i = 0; i < ss.last_groups; ++i) {
            if (accept[i] > ss.last[i].rows) throw std::runtime_error("commit_slots: accept exceeds the window");
            partial = partial || (accept[i] > 0 && accept[i] < ss.last[i].rows);
            any += accept[i];
        }
        if (any) {
            CK(cudaMemcpyAsync(ss.accept, accept, ss.last_groups * 4, cudaMemcpyHostToDevice, s));
            CK(cc::gdn_commit_slots(gd, ss.cl.data(), static_cast<std::uint32_t>(ss.cl.size()), ss.last_groups, slot_table(), ss.accept,
                                    partial, s));
            CK(cudaStreamSynchronize(s));  // `accept` is pageable host memory
        }
        for (std::uint32_t i = 0; i < ss.last_groups; ++i) ss.len[ss.last[i].slot] += accept[i];
        ss.last_groups = 0;
    }

    // ============================================================ large-M prefill (see cuda_engine.hpp prefill())
    // A SLOT is the buffer set + stream + GEMM scratch of one chunk in flight. With 2 slots chunk c runs on slot c % 2; chunk c layer l
    // only waits for chunk c-1 layer l (GDN state, conv state, KV rows), so the sequential GDN delta kernel of one chunk overlaps the
    // GEMMs of the other. Every state hand-off keeps the single-chunk order, so the result is bitwise the same for any slot count.
    struct Slot {
        cudaStream_t st = nullptr;
        cc::BigCtx* bc = nullptr;
        std::vector<cc::UniqueHandle<void*, cudaFree>> allocations;
        ~Slot() {
            if (st) (void)cudaStreamSynchronize(st);
            cc::dense_big_ctx_destroy(bc);
            allocations.clear();
            if (st) (void)cudaStreamDestroy(st);
        }
        template <class T> T* alloc(std::size_t n) {
            cc::UniqueHandle<void*, cudaFree> owner;
            CK(cudaMalloc(owner.put(), core::checked_product({n, sizeof(T)})));
            T* p = static_cast<T*>(owner.get());
            allocations.push_back(std::move(owner));
            return p;
        }
        float *x = nullptr, *xn = nullptr, *z = nullptr, *gout = nullptr, *raw = nullptr, *qkvc = nullptr, *ar = nullptr, *br = nullptr,
              *qg = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *ao = nullptr, *hid = nullptr, *scr = nullptr;
        std::uint32_t *ids = nullptr, *wpos = nullptr, *pos = nullptr, *cpar = nullptr;
        // activation split cache for the current x
        const float* act_x = nullptr;
        std::uint32_t act_in = 0, act_rows = 0;
        cc::BigAct act[2];
        bool have[2] = {false, false};
    };
    struct Pf {
        std::uint32_t C = 0;
        std::vector<std::unique_ptr<Slot>> sl;
        int terms = 3;
        // e4m3 KV shadow (opt.prefill_attn8): per attention layer, allocated on first use; positions < sh_end mirror the BF16 cache.
        std::vector<cc::KvShadowE4m3> sh;
        std::uint32_t sh_end = 0, sh_from = 0, start = 0;
    } pf;
    [[nodiscard]] bool attn8_on() const { return opt.prefill_attn8 && opt.prefill_arith != PrefillArith::exact; }
    void pf_shadow_init() {
        if (!pf.sh.empty()) return;
        const std::size_t bytes = cc::kv_shadow_e4m3_bytes(ad), p8 = cc::attn8_capacity(ad), kvd = static_cast<std::size_t>(c.kv_heads) * p8 * c.head_dim;
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            cc::KvShadowE4m3 s8;
            if (L[l].attn) {
                auto* base = alloc<std::uint8_t>(bytes, true);
                s8.k8 = base;
                s8.v8t = base + kvd;
                s8.ks = reinterpret_cast<float*>(base + 2 * kvd);
                s8.vs = s8.ks + static_cast<std::size_t>(c.kv_heads) * p8;
            }
            pf.sh.push_back(s8);
        }
    }

    [[nodiscard]] const char* big_fallback() const {
        if (!opt.prefill_big) return "disabled";
        if (opt.dense_arithmetic == cc::DenseArith::fp32) return "fp32 policy";
        if (opt.prefill_chunk == 0 || opt.prefill_chunk % 128 != 0 || opt.prefill_chunk > 8192) return "chunk shape";
        for (const LayerW& w : L) {
            const cc::DenseWeight all[] = {w.qkv, w.gate, w.beta, w.alpha, w.out, w.q, w.k, w.v, w.o, w.ffn_gate, w.ffn_up, w.ffn_down};
            for (const auto& d : all)
                if (d.data != nullptr && !cc::dense_big_supported(d)) return "weight shape/format";
        }
        return c.head_dim == 256 && gd.key_dim == 128 && gd.value_dim == 128 ? nullptr : "attention/GDN shape";
    }
    [[nodiscard]] bool big_supported() const {
        return big_fallback() == nullptr;
    }

    void pf_make_slot(Slot& sl, std::uint32_t C) {
        const std::uint32_t H = c.hidden;
        CK(cudaStreamCreateWithFlags(&sl.st, cudaStreamNonBlocking));
        sl.bc = cc::dense_big_ctx_create();
        sl.x = sl.alloc<float>(static_cast<std::size_t>(C) * H);
        sl.xn = sl.alloc<float>(static_cast<std::size_t>(C) * H);
        sl.z = sl.alloc<float>(static_cast<std::size_t>(C) * gd.v_width());
        sl.gout = sl.alloc<float>(static_cast<std::size_t>(C) * gd.v_width());
        sl.raw = sl.alloc<float>(static_cast<std::size_t>(C) * gd.conv_channels());
        sl.qkvc = sl.alloc<float>(static_cast<std::size_t>(C) * gd.conv_channels());
        sl.ar = sl.alloc<float>(static_cast<std::size_t>(C) * gd.value_heads);
        sl.br = sl.alloc<float>(static_cast<std::size_t>(C) * gd.value_heads);
        sl.qg = sl.alloc<float>(static_cast<std::size_t>(C) * c.q_gate_width());
        sl.q = sl.alloc<float>(static_cast<std::size_t>(C) * c.heads * c.head_dim);
        sl.k = sl.alloc<float>(static_cast<std::size_t>(C) * c.kv_width());
        sl.v = sl.alloc<float>(static_cast<std::size_t>(C) * c.kv_width());
        sl.ao = sl.alloc<float>(static_cast<std::size_t>(C) * c.heads * c.head_dim);
        sl.hid = sl.alloc<float>(static_cast<std::size_t>(C) * c.intermediate);
        sl.ids = sl.alloc<std::uint32_t>(C);
        sl.wpos = sl.alloc<std::uint32_t>(C / 32 + 1);
        sl.pos = sl.alloc<std::uint32_t>(1);
        sl.cpar = sl.alloc<std::uint32_t>(1);
        if (opt.prefill_arith == PrefillArith::exact) sl.scr = sl.alloc<float>(cc::attention_scratch_floats(ad, RM));
        CK(cc::dense_big_reserve(C, std::max(c.intermediate, H), static_cast<std::uint64_t>(c.intermediate) * H, 3, sl.bc));
    }

    // Rows per prefill step: opt.prefill_chunk, or for prompts >= opt.prefill_wave_min_tokens the wave-aligned rows of the
    // flash grid (strix/wave_fit.hpp; <= prefill_chunk, so the slot buffers are unchanged). Scheduling only.
    std::uint32_t pf_wave_rows = 0;  // resolved once per engine (device geometry)
    std::uint32_t pf_step_rows(std::uint32_t n) {
        const std::uint32_t C = opt.prefill_chunk;
        if (opt.prefill_wave_min_tokens == 0 || n < opt.prefill_wave_min_tokens) return C;
        if (pf_wave_rows == 0) {
            cc::AttentionDims d{};
            d.heads = c.heads;
            cc::FlashGrid g{};
            int dev = 0;
            CK(cudaGetDevice(&dev));
            CK(cc::attention_prefill_flash_grid(d, opt.flash_pipe, dev, &g));
            pf_wave_rows = strix::wave_fit::rows({g.tile_rows, g.ctas_per_tile, g.sms, g.ctas_per_sm}, C, std::min<std::uint32_t>(C, 1024U), 128U);
        }
        return pf_wave_rows;
    }

    void pf_init(std::uint32_t nslots) {
        if (pf.C != opt.prefill_chunk && !pf.sl.empty()) throw std::runtime_error("prefill: prefill_chunk cannot change after the first prefill");
        pf.C = opt.prefill_chunk;
        while (pf.sl.size() < nslots) {
            auto slot = std::make_unique<Slot>();
            pf_make_slot(*slot, pf.C);
            pf.sl.push_back(std::move(slot));
        }
    }

    [[nodiscard]] int pf_terms() const {
        if (opt.prefill_arith == PrefillArith::fast || opt.prefill_arith == PrefillArith::fast8) return 1;
        if (opt.prefill_arith == PrefillArith::fast2) return 2;
        const auto a = opt.dense_arithmetic;
        return a == cc::DenseArith::mma3 ? 3 : a == cc::DenseArith::mma2 ? 2 : 1;
    }

    // the activation planes of x for weight w (split lazily, cached until x changes)
    const cc::BigAct& pf_act(Slot& sl, const float* x, std::uint32_t rows, const cc::DenseWeight& w) {
        if (sl.act_x != x || sl.act_in != w.in || sl.act_rows != rows) {
            sl.have[0] = sl.have[1] = false;
            sl.act_x = x;
            sl.act_in = w.in;
            sl.act_rows = rows;
        }
        const int cls = w.format == cc::WeightFormat::nvfp4 ? 1 : 0;
        if (!sl.have[cls]) {
            CK(cc::dense_big_split(x, rows, w.in, w.format, pf.terms, &sl.act[cls], sl.st, sl.bc, opt.prefill_arith == PrefillArith::fast8));
            sl.have[cls] = true;
        }
        return sl.act[cls];
    }
    static void pf_dirty(Slot& sl) {
        sl.act_x = nullptr;
        sl.have[0] = sl.have[1] = false;
    }

    void pf_gemm(Slot& sl, const cc::DenseWeight& w, const float* x, std::uint32_t rows, float* y, const float* aux, cc::BigEpi epi) {
        const cc::BigAct& a = pf_act(sl, x, rows, w);
        CK(cc::dense_big_gemm(w, a, y, aux, epi, sl.st, sl.bc));
    }

    void pf_mlp(Slot& sl, const LayerW& w, std::uint32_t rows) {
        const cc::GroupedView hv{rows, 1, c.hidden, c.hidden, c.hidden};
        CK(cc::rmsnorm(sl.x, hv, w.post_norm, sl.xn, hv, c.rms_eps, sl.st));
        pf_dirty(sl);
        pf_gemm(sl, w.ffn_gate, sl.xn, rows, sl.hid, nullptr, cc::BigEpi::none);
        pf_gemm(sl, w.ffn_up, sl.xn, rows, sl.hid, sl.hid, cc::BigEpi::silu_mul);  // y = silu(gate) * up, in place
        pf_dirty(sl);
        pf_gemm(sl, w.ffn_down, sl.hid, rows, sl.x, sl.x, cc::BigEpi::residual);
        pf_dirty(sl);
    }

    void pf_attn(Slot& sl, std::uint32_t l, const LayerW& w, std::uint32_t rows, std::uint32_t pos0) {
        const std::uint32_t D = c.head_dim, qgw = c.heads * D * 2;
        pf_gemm(sl, w.q, sl.xn, rows, sl.qg, nullptr, cc::BigEpi::none);
        pf_gemm(sl, w.k, sl.xn, rows, sl.k, nullptr, cc::BigEpi::none);
        pf_gemm(sl, w.v, sl.xn, rows, sl.v, nullptr, cc::BigEpi::none);
        const cc::GroupedView qsrc{rows, c.heads, D, qgw, 2 * D}, qdst{rows, c.heads, D, c.heads * D, D};
        const cc::GroupedView kvv{rows, c.kv_heads, D, c.kv_heads * D, D};
        CK(cc::rmsnorm(sl.qg, qsrc, w.qn, sl.q, qdst, c.rms_eps, sl.st));
        CK(cc::rmsnorm(sl.k, kvv, w.kn, sl.k, kvv, c.rms_eps, sl.st));
        CK(cc::rope_apply(sl.q, qdst, c.rotary_dim, cos_t, sin_t, sl.pos, sl.st));
        CK(cc::rope_apply(sl.k, kvv, c.rotary_dim, cos_t, sin_t, sl.pos, sl.st));
        CK(cc::kv_store(ad, sl.k, sl.v, kc[l], vc[l], sl.pos, rows, sl.st));
        if (opt.prefill_arith == PrefillArith::exact) {
            for (std::uint32_t r0 = 0, wi = 0; r0 < rows; r0 += RM, ++wi) {
                const std::uint32_t rw = std::min<std::uint32_t>(RM, rows - r0);
                const cc::GroupedView qs{rw, c.heads, D, qgw, 2 * D}, qd{rw, c.heads, D, c.heads * D, D};
                CK(cc::attention_decode(ad, sl.q + static_cast<std::size_t>(r0) * c.heads * D, qd,
                                        sl.qg + static_cast<std::size_t>(r0) * qgw + D, qs, kc[l], vc[l], sl.wpos + wi, rw,
                                        sl.ao + static_cast<std::size_t>(r0) * c.heads * D, sl.scr, sl.st));
            }
        } else {
            if (attn8_on()) {
                // the first chunk also re-quantises positions written since the shadow was last in sync (decode commits, truncation)
                const std::uint32_t from = pos0 == pf.start ? pf.sh_from : pos0;
                CK(cc::kv_shadow_e4m3(ad, kc[l], vc[l], pf.sh[l], from, pos0 + rows - from, sl.st));
                CK(cc::attention_prefill_flash8(ad, sl.q, qdst, sl.qg + D, qsrc, pf.sh[l], pos0, rows, sl.ao, sl.st));
            } else {
                CK(cc::attention_prefill_flash(ad, sl.q, qdst, sl.qg + D, qsrc, kc[l], vc[l], pos0, rows, sl.ao, sl.st, opt.flash_pipe, opt.flash_tail));
            }
        }
        pf_dirty(sl);
        pf_gemm(sl, w.o, sl.ao, rows, sl.x, sl.x, cc::BigEpi::residual);
        pf_dirty(sl);
    }

    // one chunk of `rows` tokens at positions pos0.. through all layers on slot sl; x ends in sl.x. prev[l] = event of the previous chunk's
    // layer l (null for the first chunk); mine[l] is recorded after this chunk's layer l.
    void pf_chunk(Slot& sl, const std::uint32_t* tokens, std::uint32_t rows, std::uint32_t pos0, std::uint32_t read_side,
                  const cudaEvent_t* prev, cudaEvent_t* mine) {
        const std::uint32_t H = c.hidden;
        const cc::GroupedView hv{rows, 1, H, H, H};
        CK(cudaMemcpyAsync(sl.ids, tokens, static_cast<std::size_t>(rows) * 4, cudaMemcpyHostToDevice, sl.st));
        CK(cudaMemcpyAsync(sl.pos, &pos0, 4, cudaMemcpyHostToDevice, sl.st));
        CK(cudaMemcpyAsync(sl.cpar, &read_side, 4, cudaMemcpyHostToDevice, sl.st));
        CK(cc::iota_u32(sl.wpos, (rows + RM - 1) / RM, pos0, RM, sl.st));
        CK(cc::embedding_rows(embd, sl.ids, sl.x, rows, sl.st));
        std::uint32_t gi = 0;
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            const LayerW& w = L[l];
            if (prev != nullptr) CK(cudaStreamWaitEvent(sl.st, prev[l], 0));
            CK(cc::rmsnorm(sl.x, hv, w.attn_norm, sl.xn, hv, c.rms_eps, sl.st));
            pf_dirty(sl);
            if (!w.attn) {
                pf_gemm(sl, w.qkv, sl.xn, rows, sl.raw, nullptr, cc::BigEpi::none);
                pf_gemm(sl, w.gate, sl.xn, rows, sl.z, nullptr, cc::BigEpi::none);
                pf_gemm(sl, w.beta, sl.xn, rows, sl.br, nullptr, cc::BigEpi::none);
                pf_gemm(sl, w.alpha, sl.xn, rows, sl.ar, nullptr, cc::BigEpi::none);
                const auto& cl = commit_layers[gi++];
                CK(cc::gdn_prefill_layer(gd, cl.layer, cl.state, cc::GdnWindow{sl.raw, sl.qkvc, sl.ar, sl.br}, sl.z, sl.gout, rows, c.rms_eps,
                                         sl.cpar, sl.st));
                pf_gemm(sl, w.out, sl.gout, rows, sl.x, sl.x, cc::BigEpi::residual);
                pf_dirty(sl);
            } else {
                pf_attn(sl, l, w, rows, pos0);
            }
            pf_mlp(sl, w, rows);
            CK(cudaEventRecord(mine[l], sl.st));
        }
    }

    RowsOut prefill(const std::uint32_t* tokens, std::uint32_t n, LogitRows want) {
        if (n == 0) throw std::runtime_error("prefill: n must be >= 1");
        core::require_capacity(committed, n, opt.max_positions);
        core::validate_token_ids(tokens, n, c.vocab);
        if (opt.prefill_arith == PrefillArith::fast8 && !cc::dense_big_fast8_supported())
            throw std::invalid_argument("prefill: FAST8 implementation unavailable for this build/device");
        const char* fallback = big_fallback();
        if (!opt.quiet) std::fprintf(stderr, "prefill route: %s; reason: %s; chunk: %u; requested slots: %u\n",
                                    fallback ? "windows" : "large", fallback ? fallback : "admitted", opt.prefill_chunk, opt.prefill_slots);
        last_rows = 0;
        if (!big_supported()) {  // windows + commit
            RowsOut o;
            for (std::uint32_t p = 0; p < n; p += RM) {
                const std::uint32_t r = std::min<std::uint32_t>(RM, n - p);
                o = forward(tokens + p, r, p + r == n ? want : LogitRows::none);
                commit_rows(r);
            }
            return o;
        }
        const std::uint32_t C = pf_step_rows(n), H = c.hidden;  // C <= opt.prefill_chunk (buffers)
        // chunk plan (optional ramp from idle) + bounded submission: cuda_core/submit_window.hpp
        const auto plan = cc::plan_chunks(n, C, opt.prefill_ramp, 128);
        const std::uint32_t nchunks = static_cast<std::uint32_t>(plan.size());
        pf_init(std::min<std::uint32_t>(std::max<std::uint32_t>(1, opt.prefill_slots), nchunks));
        pf.terms = pf_terms();
        if (attn8_on()) {
            pf_shadow_init();
            pf.start = committed;
            pf.sh_from = std::min(pf.sh_end, committed);
        }
        const std::uint32_t nslots = static_cast<std::uint32_t>(pf.sl.size());
        if (!opt.quiet)
            std::fprintf(stderr, "prefill effective slots: %u; chunks: %u (first %u rows, ramp %u); inflight: %u%s\n", std::min(nslots, nchunks),
                         nchunks, plan.front().rows, opt.prefill_ramp, opt.prefill_inflight, opt.prefill_inflight ? " chunks" : " (unbounded)");
        CK(cudaStreamSynchronize(s));
        std::uint32_t par0 = 0;
        CK(cudaMemcpy(&par0, d_parity, 4, cudaMemcpyDeviceToHost));
        struct Events {
            Pf& pf;
            cudaStream_t stream;
            cudaEvent_t start = nullptr;
            std::vector<cudaEvent_t> ev;
            ~Events() {
                for (const auto& slot : pf.sl) (void)cudaStreamSynchronize(slot->st);
                (void)cudaStreamSynchronize(stream);
                for (auto e : ev) if (e) (void)cudaEventDestroy(e);
                if (start) (void)cudaEventDestroy(start);
            }
        } events{pf, s, nullptr, {}};
        auto& start = events.start;
        auto& ev = events.ev;
        CK(cudaEventCreateWithFlags(&start, cudaEventDisableTiming));
        CK(cudaEventRecord(start, s));
        ev.resize(static_cast<std::size_t>(nchunks) * c.layers);
        for (auto& e : ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        cc::SubmitWindow window{opt.prefill_inflight};
        last_chunk_ends.clear();
        for (std::uint32_t ci = 0; ci < nchunks; ++ci) {
            Slot& sl = *pf.sl[ci % nslots];
            // at most prefill_inflight chunks queued ahead of the GPU: wait for chunk ci - N's last layer (scheduling only)
            if (ci > 0) last_chunk_ends.push_back(ev[static_cast<std::size_t>(ci) * c.layers - 1]);
            CK(window.admit(last_chunk_ends.data(), ci));
            if (ci < nslots) CK(cudaStreamWaitEvent(sl.st, start, 0));
            const std::uint32_t p = plan[ci].pos, r = plan[ci].rows;
            pf_chunk(sl, tokens + p, r, committed + p, (par0 + ci) & 1u, ci ? &ev[static_cast<std::size_t>(ci - 1) * c.layers] : nullptr,
                     &ev[static_cast<std::size_t>(ci) * c.layers]);
        }
        Slot& last = *pf.sl[(nchunks - 1) % nslots];
        CK(cudaStreamWaitEvent(s, ev[static_cast<std::size_t>(nchunks) * c.layers - 1], 0));
        const std::uint32_t rl = plan.back().rows;
        last_window_waits = window.waits;
        // last token: final norm + lm_head on its row, exported through the decode buffers (row 0)
        CK(cudaMemcpyAsync(x, last.x + static_cast<std::size_t>(rl - 1) * H, static_cast<std::size_t>(H) * 4, cudaMemcpyDeviceToDevice, s));
        const cc::GroupedView h1{1, 1, H, H, H};
        CK(cc::rmsnorm(x, h1, out_norm, xn, h1, c.rms_eps, s));
        if (want != LogitRows::none) {
            CK(cc::dense_rows(head, xn, logits, 1, nullptr, s, opt.dense_arithmetic));
            CK(cc::argmax_rows(logits, c.vocab, 1, d_top, d_argws, s));
        }
        const std::uint32_t par_end = (par0 + nchunks) & 1u;
        CK(cudaMemcpyAsync(d_parity, &par_end, 4, cudaMemcpyHostToDevice, s));
        CK(cudaStreamSynchronize(s));  // the staging copy above and the event pool are released after this point
        committed += n;
        if (attn8_on()) pf.sh_end = committed;
        RowsOut o;
        o.logits = want == LogitRows::none ? nullptr : logits;
        o.top = want == LogitRows::none ? nullptr : d_top;
        o.hidden_post = xn;
        o.hidden_pre = x;
        return o;
    }

    CudaEngine::MtpOut mtp(const std::uint32_t* next_tokens, const float* hidden_dev, std::uint32_t rows, std::uint32_t pos0, bool want_top) {
        if (!mtp_ok) throw std::runtime_error("mtp_forward: no MTP tensors loaded");
        if (rows == 0 || rows > RM) throw std::runtime_error("mtp_forward: bad rows/pos0");
        core::require_capacity(pos0, rows, opt.max_positions);
        core::validate_token_ids(next_tokens, rows, c.vocab);
        const std::uint32_t H = c.hidden;
        InputPack in{};
        in.pos0 = pos0;
        for (std::uint32_t i = 0; i < rows; ++i) in.ids[i] = next_tokens[i];
        set_inputs_kernel<<<1, 32, 0, s>>>(in, d_mpos, d_mids);
        CK(cudaGetLastError());
        const cc::GroupedView src{rows, 1, H, H, H}, dst{rows, 1, H, 2 * H, H};
        CK(cc::embedding_rows(embd, d_mids, mtmp, rows, s));
        CK(cc::rmsnorm(mtmp, src, m_enorm, mcat, dst, c.rms_eps, s));
        CK(cc::rmsnorm(hidden_dev, src, m_hnorm, mcat + H, dst, c.rms_eps, s));
        CK(cc::dense_rows(m_eh, mcat, mx, rows, nullptr, s, opt.dense_arithmetic));
        const cc::GroupedView hv{rows, 1, H, H, H};
        CK(cc::rmsnorm(mx, hv, M.attn_norm, mxn, hv, c.rms_eps, s));
        const Bufs b{mx, mxn, d_mpos, mkc, mvc};
        attn_mixer(M, b, rows);
        mlp(M, b, rows);
        // mx holds the block output (pre mtp.norm); the hidden export needs it intact, so normalise into mxn.
        CK(cc::rmsnorm(mx, hv, m_shnorm, mxn, hv, c.rms_eps, s));
        CudaEngine::MtpOut o;
        if (want_top) {
            CK(cc::dense_rows(head, mxn, mlogits, rows, nullptr, s, opt.dense_arithmetic));
            CK(cc::argmax_rows(mlogits, c.vocab, rows, d_mtop, d_argws, s));
            o.logits = mlogits;
            o.top = d_mtop;
        }
        o.hidden_pre = mx;
        o.hidden_post = mxn;
        return o;
    }
};

// ================================================================ loading
CudaEngine::CudaEngine() : p_(new Impl) {}
CudaEngine::~CudaEngine() = default;

std::unique_ptr<CudaEngine> CudaEngine::from_gguf(const std::string& path, const EngineOptions& opt) {
    std::unique_ptr<CudaEngine> e(new CudaEngine);
    auto& P = *e->p_;
    CK(cudaGetDevice(&P.device));
    if (opt.prefill_slots < 1 || opt.prefill_slots > 2) throw std::invalid_argument("prefill_slots expects 1..2");
    P.opt = opt;
    P.graphs_on = opt.use_graphs;
    if (!opt.max_positions) throw std::invalid_argument("max_positions must be positive");
    const auto info = gguf::read_file(path);
    P.c = validate(info, opt.max_positions);
    validate_backend(P.c, opt.max_positions);
    const auto t0 = std::chrono::steady_clock::now();
    auto specs = specs_from_gguf(info, P.c.layers, opt.load_mtp);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    if (opt.turbo_weights == TurboWeights::converted) {
        // converted pack: NVFP4 tile tensors (type kNvfp4TilePack) are mapped as they are; their gscales come from the table tensor.
        std::vector<float> gs;
        for (auto it = specs.begin(); it != specs.end(); ++it) {
            if (it->name != kPackGscaleName) continue;
            gs.resize(static_cast<std::size_t>(it->bytes / 4));
            in.seekg(static_cast<std::streamoff>(it->file_off));
            in.read(reinterpret_cast<char*>(gs.data()), static_cast<std::streamsize>(gs.size() * 4));
            if (!in) throw std::runtime_error("short read of the pack gscale table");
            specs.erase(it);
            break;
        }
        std::size_t k = 0;
        for (auto& t : specs) {
            if (t.type != static_cast<int>(gguf::kNvfp4TilePack)) continue;
            if (k >= gs.size() || t.bytes != cc::nvfp4_bytes(static_cast<std::uint32_t>(t.out), static_cast<std::uint32_t>(t.in)))
                throw std::runtime_error("converted pack: bad NVFP4 tensor " + t.name);
            t.fp4 = true;
            t.gscale = gs[k++];
        }
        if (k != gs.size() || k == 0) throw std::runtime_error("converted pack: gscale table does not match its NVFP4 tensors");
        P.opt.turbo_lm_head = false;
        for (const auto& t : specs) if (t.name == "output.weight") P.opt.turbo_lm_head = t.fp4;  // identity follows the pack
    } else {
        // A converted pack holds NVFP4 tile bytes: without --turbo-weights converted they would be laid out as exact weights.
        for (const auto& t : specs)
            if (t.type == static_cast<int>(gguf::kNvfp4TilePack) || t.name == kPackGscaleName)
                throw std::invalid_argument("qwen35: " + path + " is a converted NVFP4 pack; load it with --turbo-weights converted");
        mark_fp4(opt, specs);
    }
    layout(P.W, specs, P.c.layers);
    std::uint64_t tmp_bytes = 0;
    for (const auto& t : specs) if (t.fp4) tmp_bytes = std::max(tmp_bytes, t.src_bytes);
    cc::UniqueHandle<void*, cudaFree> tmp_owner;
    unsigned char* tmp = nullptr;  // BF16 staging for the on-the-fly conversion (largest eligible tensor)
    if (tmp_bytes) CK(cudaMalloc(tmp_owner.put(), tmp_bytes));
    tmp = static_cast<unsigned char*>(tmp_owner.get());
    constexpr std::size_t kChunk = 128u << 20;
    cc::UniqueHandle<void*, cudaFreeHost> pin_owner;
    CK(cudaMallocHost(pin_owner.put(), kChunk));
    auto* pin = static_cast<unsigned char*>(pin_owner.get());
    for (auto& [name, t] : P.W.spec) {
        in.seekg(static_cast<std::streamoff>(t.file_off));
        std::uint64_t done = 0;
        const bool convert = t.fp4 && t.src_bytes != 0;  // false for a converted pack: bytes go straight to their slot
        const std::uint64_t total = convert ? t.src_bytes : t.bytes;
        unsigned char* dst = convert ? tmp : P.W.base + P.W.off[name];
        while (done < total) {
            const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(kChunk, total - done));
            in.read(reinterpret_cast<char*>(pin), static_cast<std::streamsize>(n));
            if (!in) throw std::runtime_error("short read for " + name);
            CK(cudaMemcpy(dst + done, pin, n, cudaMemcpyHostToDevice));
            done += n;
        }
        if (convert) {
            CK(cc::nvfp4_quantize_bf16(tmp, static_cast<std::uint32_t>(t.out), static_cast<std::uint32_t>(t.in),
                                       P.W.base + P.W.off[name], &t.gscale, nullptr));
            CK(cudaDeviceSynchronize());
        }
    }
    pin_owner.reset();
    tmp_owner.reset();
    P.init();
    if (!opt.quiet)
        std::fprintf(stderr, "qwen35 engine: loaded %.2f GB in %.1f s, mtp %s\n", static_cast<double>(P.W.total) / 1e9,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), P.mtp_ok ? "yes" : "no");
    return e;
}

// Writes the NVFP4 tensors the onfly route builds (same kernel, same bytes) as a converted pack: a GGUF with the source key/value
// section, every non-NVFP4 tensor copied unchanged (norms, embeddings, GDN state tensors, BF16 lm_head when requested), the NVFP4
// weights as type kNvfp4TilePack, and one F32 table of their global scales (table order = tensor order).
std::uint64_t save_converted_pack(const std::string& path, const std::string& out_path, const EngineOptions& opt_in) {
    EngineOptions opt = opt_in;
    opt.turbo_weights = TurboWeights::onfly;
    const auto info = gguf::read_file(path);
    const Config c = validate(info, opt.max_positions);
    validate_backend(c, opt.max_positions);
    auto specs = specs_from_gguf(info, c.layers, opt.load_mtp);
    mark_fp4(opt, specs);
    std::map<std::string, const gguf::TensorInfo*> src;
    for (const auto& t : info.tensors) src[t.name] = &t;
    std::vector<gguf::OutTensor> outs;
    std::uint64_t max_src = 0, max_dst = 0, n4 = 0;
    for (const auto& t : specs) {
        gguf::OutTensor o;
        o.name = t.name;
        o.dimensions = src.at(t.name)->dimensions;
        o.ggml_type = t.fp4 ? gguf::kNvfp4TilePack : static_cast<std::uint32_t>(t.type);
        o.bytes = t.bytes;
        if (t.fp4) { ++n4; max_src = std::max(max_src, t.src_bytes); max_dst = std::max(max_dst, t.bytes); }
        outs.push_back(std::move(o));
    }
    if (n4 == 0) throw std::runtime_error("save_converted_pack: no NVFP4-eligible tensor (the source must be the BF16 gguf)");
    outs.push_back({kPackGscaleName, {n4}, 0, n4 * 4, 0});
    gguf::PackWriter w(out_path, path, info, outs);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    constexpr std::size_t kChunk = 128u << 20;
    cc::UniqueHandle<void*, cudaFreeHost> pin_owner;
    CK(cudaMallocHost(pin_owner.put(), kChunk));
    auto* pin = static_cast<unsigned char*>(pin_owner.get());
    cc::UniqueHandle<void*, cudaFree> tmp_owner, dst_owner;
    CK(cudaMalloc(tmp_owner.put(), max_src));
    CK(cudaMalloc(dst_owner.put(), max_dst));
    std::vector<unsigned char> host_dst(static_cast<std::size_t>(max_dst));
    std::vector<float> gscales;
    for (const auto& t : specs) {
        in.seekg(static_cast<std::streamoff>(t.file_off));
        const std::uint64_t total = t.fp4 ? t.src_bytes : t.bytes;
        std::uint64_t done = 0;
        while (done < total) {
            const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(kChunk, total - done));
            in.read(reinterpret_cast<char*>(pin), static_cast<std::streamsize>(n));
            if (!in) throw std::runtime_error("short read for " + t.name);
            if (t.fp4) CK(cudaMemcpy(static_cast<unsigned char*>(tmp_owner.get()) + done, pin, n, cudaMemcpyHostToDevice));
            else w.append(pin, n);
            done += n;
        }
        if (t.fp4) {
            float gs = 1.0f;
            CK(cc::nvfp4_quantize_bf16(tmp_owner.get(), static_cast<std::uint32_t>(t.out), static_cast<std::uint32_t>(t.in), dst_owner.get(), &gs, nullptr));
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(host_dst.data(), dst_owner.get(), static_cast<std::size_t>(t.bytes), cudaMemcpyDeviceToHost));
            w.append(host_dst.data(), t.bytes);
            gscales.push_back(gs);
        }
        w.finish_tensor();
    }
    w.append(gscales.data(), gscales.size() * 4);
    w.finish_tensor();
    return w.close();
}

std::unique_ptr<CudaEngine> CudaEngine::synthetic(const Config& cfg, PackFormat pack, const EngineOptions& opt, std::uint64_t seed) {
    std::unique_ptr<CudaEngine> e(new CudaEngine);
    auto& P = *e->p_;
    CK(cudaGetDevice(&P.device));
    if (opt.prefill_slots < 1 || opt.prefill_slots > 2) throw std::invalid_argument("prefill_slots expects 1..2");
    P.opt = opt;
    P.graphs_on = opt.use_graphs;
    P.c = cfg;
    validate_backend(P.c, opt.max_positions);
    const auto t0 = std::chrono::steady_clock::now();
    if (opt.turbo_weights == TurboWeights::onfly) pack = PackFormat::bf16;  // onfly source is BF16
    const auto sp = synth_specs(cfg, pack, opt.load_mtp);
    std::vector<PackTensorSpec> specs;
    for (const auto& [t, f] : sp) specs.push_back(t);
    mark_fp4(opt, specs);
    layout(P.W, specs, cfg.layers);
    std::uint64_t tmp_bytes = 0;
    for (const auto& t : specs) if (t.fp4) tmp_bytes = std::max(tmp_bytes, t.src_bytes);
    cc::UniqueHandle<void*, cudaFree> tmp_owner;
    unsigned char* tmp = nullptr;
    if (tmp_bytes) CK(cudaMalloc(tmp_owner.put(), tmp_bytes));
    tmp = static_cast<unsigned char*>(tmp_owner.get());
    for (const auto& [t0_, f] : sp) {
        auto& t = P.W.spec[t0_.name];
        unsigned char* dst = t.fp4 ? tmp : P.W.base + P.W.off[t.name];
        const std::uint64_t sd = seed ^ fnv(t.name);
        const std::uint64_t n = t.in * t.out;
        if (t.type == 0) {
            fill_f32_kernel<<<static_cast<unsigned>(std::min<std::uint64_t>((n + 255) / 256, 4096)), 256>>>(reinterpret_cast<float*>(dst), n, sd, f.base, f.amp);
        } else if (t.type == 30) {
            fill_bf16_kernel<<<4096, 256>>>(reinterpret_cast<std::uint16_t*>(dst), n, sd, f.base, f.amp);
        } else {
            fill_q8_kernel<<<4096, 256>>>(dst, n / 32, sd, f.amp);
        }
        CK(cudaGetLastError());
        if (t.fp4) {
            CK(cc::nvfp4_quantize_bf16(tmp, static_cast<std::uint32_t>(t.out), static_cast<std::uint32_t>(t.in),
                                       P.W.base + P.W.off[t.name], &t.gscale, nullptr));
            CK(cudaDeviceSynchronize());
        }
    }
    CK(cudaDeviceSynchronize());
    tmp_owner.reset();
    P.init();
    if (!opt.quiet)
        std::fprintf(stderr, "qwen35 engine: synthetic %s %.2f GB generated in %.1f s, mtp %s\n", pack == PackFormat::bf16 ? "bf16" : "q8_0",
                     static_cast<double>(P.W.total) / 1e9,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), P.mtp_ok ? "yes" : "no");
    return e;
}

void print_effective_config(const EngineOptions& o, const char* arith, std::FILE* f) {
    static const char* names[] = {"fp32", "mma3", "mma2", "mma1"};
    arith = names[static_cast<unsigned>(o.dense_arithmetic)];
    const std::string id = engine_identity(o);
    std::fprintf(f, "effective-config:\n  identity: %s\n  prefix-cache-key: %s (never shared across identities)\n", id.c_str(), id.c_str());
    std::fprintf(f, "  dense-arith: %s\n  turbo-weights: %s\n  turbo-bits: %u\n  turbo-lm-head: %s\n  turbo-multi-launch: %s (bitwise neutral)\n",
                 arith ? arith : "mma3", o.turbo_weights == TurboWeights::off ? "off (exact pack)" : o.turbo_weights == TurboWeights::native ? "native" : o.turbo_weights == TurboWeights::converted ? "converted (NVFP4 pack)" : "onfly",
                 o.turbo_bits, o.turbo_weights >= TurboWeights::onfly ? (o.turbo_lm_head ? "nvfp4" : "bf16") : "n/a", o.multi_gemv ? "on" : "off");
    std::fprintf(f, "  prefill-arith: %s\n  prefill-big: %s\n  prefill-chunk: %u\n  prefill-slots: %u\n  prefill-admission: pending model/device; execution reports route/reason\n",
                 prefill_arith_name(o.prefill_arith), o.prefill_big ? "on" : "off", o.prefill_chunk, o.prefill_slots);
    std::fprintf(f, "  prefill-inflight: %u%s (bounded submission, scheduling only, bitwise)\n  prefill-ramp: %u%s\n", o.prefill_inflight,
                 o.prefill_inflight ? " chunks" : " = unbounded", o.prefill_ramp, o.prefill_ramp ? " rows first chunk, doubling to prefill-chunk" : " (off)");
    if (o.turbo_weights >= TurboWeights::onfly)
        std::fprintf(f, "  format: NVFP4 e2m1 + e4m3/16 + FP32 global; RTN; GEMV f16 mma, FP32 accumulate, activations RN f16, rows 1..32 bitwise row-invariant\n");
    std::fprintf(f, "  prefill-flash-tail: %s (scheduling only, bitwise equal; split = partial last wave as fewer-warp jobs)\n",
                 o.flash_tail == cuda_core::FlashTail::split ? "split" : "off");
    if (o.prefill_wave_min_tokens == 0) std::fprintf(f, "  prefill-wave-steps: off (steps of prefill_chunk; --prefill-wave-min N turns it on)\n");
    else std::fprintf(f, "  prefill-wave-steps: prompts >= %u tokens step by wave-aligned rows <= prefill_chunk (scheduling only, bitwise)\n",
                      o.prefill_wave_min_tokens);
    std::fprintf(f, "  prefill-flash-pipe: %s (scheduling only, bitwise equal across pipes)\n  prefill-attn8: %s\n",
                 o.flash_pipe == cuda_core::FlashPipe::swz3 ? "swz3" : "pad2",
                 o.prefill_arith == PrefillArith::exact ? "n/a (exact prefill keeps attention_decode)"
                 : o.prefill_attn8 ? "on (e4m3 shadow KV + e4m3 flash, bit-changing, identity -a8)" : "off");
}

const Config& CudaEngine::config() const { return p_->c; }
const EngineOptions& CudaEngine::options() const { return p_->opt; }
std::string CudaEngine::identity() const {
    return engine_identity(p_->opt);
}
std::uint32_t CudaEngine::nvfp4_tensors() const {
    std::uint32_t n = 0;
    for (const auto& [k, t] : p_->W.spec) n += t.fp4 ? 1u : 0u;
    return n;
}
bool CudaEngine::has_mtp() const { return p_->mtp_ok; }
cudaStream_t CudaEngine::stream() const { return p_->s; }
std::uint32_t CudaEngine::committed() const { return p_->committed; }
std::uint64_t CudaEngine::weight_bytes() const { return p_->W.trunk_bytes; }
std::uint64_t CudaEngine::state_bytes_per_pass() const {
    return static_cast<std::uint64_t>(p_->commit_layers.size()) * p_->gd.value_heads * p_->gd.key_dim * p_->gd.value_dim * 4 * 2;
}
void CudaEngine::reset() { p_->reset(); }
RowsOut CudaEngine::forward_rows(const std::uint32_t* tokens, std::uint32_t rows, LogitRows want) {
    return p_->forward(tokens, rows, want);
}
void CudaEngine::commit(std::uint32_t k) { p_->commit_rows(k); }
RowsOut CudaEngine::prefill(const std::uint32_t* tokens, std::uint32_t n, LogitRows want) { return p_->prefill(tokens, n, want); }
bool CudaEngine::prefill_is_big() const { return p_->big_supported(); }
const char* prefill_arith_name(PrefillArith a) {
    static const char* n[] = {"exact", "flash", "fast", "fast2", "fast8"};
    return n[static_cast<int>(a)];
}
CudaEngine::MtpOut CudaEngine::mtp_forward(const std::uint32_t* next_tokens, const float* hidden_dev, std::uint32_t rows,
                                           std::uint32_t pos0, bool want_top) {
    return p_->mtp(next_tokens, hidden_dev, rows, pos0, want_top);
}
void CudaEngine::read_logits(const RowsOut& o, std::uint32_t row, float* host) const {
    CK(cudaMemcpyAsync(host, o.logits + static_cast<std::size_t>(row) * p_->c.vocab, static_cast<std::size_t>(p_->c.vocab) * 4,
                       cudaMemcpyDeviceToHost, p_->s));
    CK(cudaStreamSynchronize(p_->s));
}
void CudaEngine::read_top(const RowsOut& o, std::uint32_t rows, std::uint32_t* host) const {
    CK(cudaMemcpyAsync(host, o.top, rows * 4, cudaMemcpyDeviceToHost, p_->s));
    CK(cudaStreamSynchronize(p_->s));
}
void CudaEngine::read_hidden(const float* dev, std::uint32_t rows, float* host) const {
    CK(cudaMemcpyAsync(host, dev, static_cast<std::size_t>(rows) * p_->c.hidden * 4, cudaMemcpyDeviceToHost, p_->s));
    CK(cudaStreamSynchronize(p_->s));
}
void CudaEngine::configure_slots(std::uint32_t n) { p_->configure_slots(n); }
std::uint32_t CudaEngine::slot_count() const { return p_->ss.n; }
std::uint32_t CudaEngine::slot_committed(std::uint32_t slot) const {
    p_->check_slot(slot);
    return p_->ss.len[slot];
}
void CudaEngine::reset_slot(std::uint32_t slot) { p_->reset_slot(slot); }
void CudaEngine::copy_slot(std::uint32_t dst, std::uint32_t src) { p_->copy_slot(dst, src); }
RowsOut CudaEngine::forward_slots(const SlotRows* groups, std::uint32_t n, const std::uint32_t* tokens, LogitRows want, bool auto_commit) {
    return p_->forward_slots(groups, n, tokens, want, auto_commit);
}
void CudaEngine::commit_slots(const std::uint32_t* accept) { p_->commit_slots(accept); }
void CudaEngine::sync() const { CK(cudaStreamSynchronize(p_->s)); }
void CudaEngine::set_graphs(bool on) { p_->graphs_on = on; }
std::uint64_t CudaEngine::launches_last_forward() const { return p_->last_nodes; }

}  // namespace strix::qwen35
