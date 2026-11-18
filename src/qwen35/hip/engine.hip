// qwen35 HIP window engine (see include/strix/qwen35/hip_engine.hpp). Moved from apps/hip_qwen35_decode_main.hip and extended
// with the multi-token verify window (forward_rows + commit(k), deferred GDN commit by snapshot + replay).
// Binds shared cores only: core/dense_linear (gemv_rows + weight_decode), attn_core/dense_decode, ops/norm/norm_device,
// hip_gdn_decode_step_dims (R1 GDN launcher path, swish gated norm).
// Home: specific = layer composition; every kernel is shared.
#include "strix/qwen35/hip_engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/env_flags.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/hip_gdn_dims.hpp"
#include "../../attn_core/dense_decode.hpp"
#include "../../attn_core/dense_decode_flash.hpp"
#include "../../core/dense_linear.hpp"
#include "../../core/dense_linear_iu4.hpp"
#include "../../core/f16_dense_iu4.hpp"
#include "../../core/f16_dense_q8a8.hpp"
#include "../../ops/norm/norm_device.hpp"

namespace strix::qwen35 {
namespace {
using namespace strix;
namespace dl = core::dense_linear;
namespace wd = core::weight_decode;

#define HK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); } while (0)

// RMSNorm through the shared ops/norm kernel: F32 stored gains (1+w); reciprocal = 1/sqrt (as the attention prep).
struct NormTraits {
    using Input = float;
    using Output = float;
    using Scale = float;
    using ScaleFormat = int;
    static constexpr auto reduction_order = ops::norm::ReductionOrder::lane_strided_fma_lds_halving;
    template <unsigned L> __device__ __forceinline__ static float reduce(float* s, float part) {
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
    __device__ __forceinline__ static float inv_sqrt(float x) { return 1.0F / sqrtf(x); }
    __device__ __forceinline__ static float load(const float* w, int, unsigned i) { return w[i]; }
};

__global__ void embed_kernel(const std::uint8_t* table, unsigned fmt, std::size_t row_bytes, unsigned hidden, const int* ids, float* out) {
    const unsigned r = blockIdx.x;
    const std::uint8_t* row = table + std::size_t(ids[r]) * row_bytes;
    for (unsigned i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v;
        if (fmt == 30U) {
            std::uint16_t h;
            __builtin_memcpy(&h, row + 2ULL * i, 2);
            v = wd::bf16_to_float(h);
        } else if (fmt == 0U) {
            __builtin_memcpy(&v, row + 4ULL * i, 4);
        } else {
            const std::uint8_t* b = row + std::size_t(i / 32U) * 34U;
            std::uint16_t h;
            __builtin_memcpy(&h, b, 2);
            const int q = static_cast<signed char>(b[2 + i % 32U]);
            v = float(__builtin_bit_cast(_Float16, h)) * float(q);
        }
        out[std::size_t(r) * hidden + i] = v;
    }
}
__global__ void add_kernel(float* x, const float* y, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] + y[i];
}
// Lowest index wins ties; fixed tree.
__global__ void argmax_kernel(const float* logits, unsigned vocab, int* out) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    const float* row = logits + std::size_t(blockIdx.x) * vocab;
    float best = -INFINITY;
    int bidx = 0x7fffffff;
    for (unsigned i = threadIdx.x; i < vocab; i += 1024U)
        if (row[i] > best) { best = row[i]; bidx = int(i); }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = bidx;
    __syncthreads();
    for (unsigned s = 512U; s > 0U; s >>= 1U) {
        if (threadIdx.x < s) {
            const float v = bv[threadIdx.x + s];
            const int i = bi[threadIdx.x + s];
            if (v > bv[threadIdx.x] || (v == bv[threadIdx.x] && i < bi[threadIdx.x])) { bv[threadIdx.x] = v; bi[threadIdx.x] = i; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = bi[0];
}

struct Tensor {
    std::uint8_t* data = nullptr;
    unsigned inputs = 0, outputs = 0, fmt = 0;  // fmt: ggml type 0 f32, 1 f16, 8 q8_0, 30 bf16, 200 IU4 g64 (shared pack)
    std::size_t bytes = 0, row_bytes = 0;
};
constexpr unsigned kMaxRows = HipEngine::max_rows();

class Loader {
public:
    explicit Loader(const std::string& path) : info_(gguf::read_file(path)), file_(path, std::ios::binary) {
        if (!file_) throw std::runtime_error("cannot open " + path);
        for (const auto& t : info_.tensors) by_name_[t.name] = &t;
    }
    const gguf::FileInfo& info() const { return info_; }
    const gguf::TensorInfo* find(const std::string& name) const {
        for (const char* suffix : {"", ".weight", ".bias"}) {
            auto it = by_name_.find(name + suffix);
            if (it != by_name_.end()) return it->second;
        }
        return nullptr;
    }
    Tensor load(const std::string& name) {
        const auto* t = find(name);
        if (!t) throw std::runtime_error("missing tensor " + name);
        Tensor r;
        r.fmt = t->ggml_type;
        r.bytes = t->byte_size;
        r.inputs = unsigned(t->dimensions.at(0));
        r.outputs = t->dimensions.size() > 1 ? unsigned(t->dimensions[1]) : 1U;
        const auto g = gguf::tensor_type_geometry(t->ggml_type);
        if (!g) throw std::runtime_error("unsupported ggml type for " + name);
        r.row_bytes = std::size_t(r.inputs / g->block_elements) * g->block_bytes;
        HK(hipMalloc(&r.data, r.bytes));
        file_.seekg(std::streamoff(t->absolute_offset));
        std::vector<char> stage(std::min<std::size_t>(r.bytes, 64ULL << 20));
        for (std::size_t done = 0; done < r.bytes;) {
            const auto n = std::min(stage.size(), r.bytes - done);
            file_.read(stage.data(), std::streamsize(n));
            if (!file_) throw std::runtime_error("short read " + name);
            HK(hipMemcpy(r.data + done, stage.data(), n, hipMemcpyHostToDevice));
            done += n;
        }
        total_ += r.bytes;
        return r;
    }
    std::size_t total() const { return total_; }

private:
    gguf::FileInfo info_;
    std::ifstream file_;
    std::map<std::string, const gguf::TensorInfo*> by_name_;
    std::size_t total_ = 0;
};


struct Layer {
    Tensor attn_norm, post_norm, gate, up, down;
    Tensor qkv, z, beta, alpha, conv, dt, a, snorm, out;  // gdn
    Tensor q, k, v, o, qnorm, knorm;                      // attention
    float* gdn_state = nullptr;
    std::uint16_t* conv_state = nullptr;
    std::uint16_t *k_cache = nullptr, *v_cache = nullptr;
    std::uint16_t* v_t = nullptr;  // V^T copy for the wide-window flash attention (prefill_rows > 16 only)
};

dl::View view(const Tensor& t) { return {t.data, t.inputs, t.outputs, t.row_bytes}; }

// Rows 9..16: four outputs per wave (store epilogue) so each activation read feeds 4 outputs (T=1 re-reads 64 B of
// activations per weight and is L2-bound: 16 rows 827 ms, T=2 414, T=4 305-323); gate/up (two accumulator sets) keep T=2
// (T=4 spills 176-252 B/lane scratch). Rows <= 8: two outputs per wave (unchanged).
template <unsigned Rows, wd::Format F, dl::Epilogue E>
void launch_dense(const Tensor& w, const Tensor& up, const float* x, float* out, hipStream_t s) {
    constexpr unsigned T = Rows <= 8U ? 2U : (E == dl::Epilogue::silu_mul ? 2U : 4U);
    dl::kernel<Rows, T, F, E><<<(w.outputs + 4U * T - 1U) / (4U * T), 128, 0, s>>>(view(w), view(up), x, out);
}
// Rows are padded to the next of 1/2/4/8/16 over kMaxRows-sized scratch; every row is independent.
template <wd::Format F, dl::Epilogue E>
void dispatch_rows(unsigned rows, const Tensor& w, const Tensor& up, const float* x, float* out, hipStream_t s) {
    if (rows <= 1U) launch_dense<1, F, E>(w, up, x, out, s);
    else if (rows <= 2U) launch_dense<2, F, E>(w, up, x, out, s);
    else if (rows <= 4U) launch_dense<4, F, E>(w, up, x, out, s);
    else if (rows <= 8U) launch_dense<8, F, E>(w, up, x, out, s);
    else launch_dense<16, F, E>(w, up, x, out, s);
}
// ---- wide windows (rows > 16): FAST dense-q8a8-f16-v1 of the shared f16_dense_tall core --------------------------------
// One engine per process binds these scratch buffers (set by Impl); rows beyond the exact 16-row sweep go through them.
using WideScratch = core::DenseWideScratch;

// A8 of 32-value groups, one wave per group (as quantize_a8: amax / 127, rint, clamp +-127; delta 1 for zero or tiny groups).
__global__ void a8_quant_kernel(const float* x, unsigned inputs, std::int8_t* codes, float* deltas) {
    const unsigned lane = threadIdx.x % 32U, group = blockIdx.x * (blockDim.x / 32U) + threadIdx.x / 32U, groups = inputs / 32U;
    if (group >= groups) return;
    const std::size_t at = std::size_t(blockIdx.y) * inputs + std::size_t(group) * 32U + lane;
    const float y = x[at];
    float amax = fmaxf(0.0F, fabsf(y));
    for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor(amax, d, 32));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float dl = (amax == 0.0F || tiny) ? 1.0F : amax / 127.0F;
    float q = tiny ? 0.0F : __builtin_rintf(y / dl);
    q = fmaxf(-127.0F, fminf(127.0F, q));
    codes[at] = static_cast<std::int8_t>(q);
    if (lane == 0) deltas[std::size_t(blockIdx.y) * groups + group] = dl;
}
__global__ void silu_mul_kernel(const float* g, const float* u, float* out, std::size_t n) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x)
        out[i] = dl::silu(g[i]) * u[i];
}
bool wide_ok(const Tensor& t, const WideScratch& wide) { return wide.codes && t.fmt == 8U && t.inputs % 128U == 0U && t.outputs % 16U == 0U; }
// out[r][o] for rows > 16 from already quantized activations (codes / deltas in wide).
void wide_dense(const Tensor& w, float* out, unsigned rows, hipStream_t s, const WideScratch& wide) {
    namespace q = core::f16_dense_tall::q8a8;
    q::View v{};
    v.weight = w.data; v.codes = wide.codes; v.deltas = wide.deltas; v.out = out;
    v.row_bytes = std::uint32_t(w.row_bytes); v.k_bytes = 0U; v.in = w.inputs; v.count = w.outputs; v.rows = rows;
    v.x_stride = w.inputs; v.out_stride = w.outputs; v.accumulate = 0U;
    if (!q::supported(v)) throw std::runtime_error("q8a8 view not supported");
    if (q::waves_for(rows, w.inputs) == 16U) q::launch_form<16>(v, q::form_for(), s);
    else q::launch_form<8>(v, q::form_for(), s);
}
void wide_quant(const float* x, unsigned inputs, unsigned rows, hipStream_t s, const WideScratch& wide) {
    a8_quant_kernel<<<dim3((inputs / 32U + 7U) / 8U, rows), 256, 0, s>>>(x, inputs, wide.codes, wide.deltas);
}
// ---- IU4 g64 pack (ggml type 200): shared W4A16 core, third dense weight format -----------------------------------------
// FP16 activation rows and F32 gate / up rows of the 16-row windows; non-owning view, the engine Impl owns the allocations.
struct Iu4Scratch { _Float16* xh = nullptr; float *g = nullptr, *u = nullptr; };
constexpr unsigned kIu4Fmt = gguf::type_strix_iu4_g64;
void linear(const Tensor& w, const float* x, float* out, unsigned rows, hipStream_t s, const WideScratch& wide, const Iu4Scratch& iu4);
void linear16(const Tensor& w, const float* x, float* out, unsigned rows, hipStream_t s, const Iu4Scratch& iu4) {
    if (w.fmt == 30U) dispatch_rows<wd::Format::bf16, dl::Epilogue::store>(rows, w, w, x, out, s);
    else if (w.fmt == 8U) dispatch_rows<wd::Format::q8_0, dl::Epilogue::store>(rows, w, w, x, out, s);
    else if (w.fmt == kIu4Fmt) {
        if (!iu4.xh) throw std::runtime_error("iu4 scratch not allocated");
        if (core::dense_iu4::reference_flag()) return core::dense_iu4::reference(w.data, w.inputs, w.outputs, x, out, rows, s);
        core::dense_iu4::linear(w.data, w.inputs, w.outputs, x, iu4.xh, out, rows, s);
    } else throw std::runtime_error("dense projection needs bf16, q8_0 or iu4 weights");
}
void glu16(const Tensor& g, const Tensor& u, const float* x, float* out, unsigned rows, hipStream_t s, const Iu4Scratch& iu4) {
    if (g.fmt != u.fmt || g.outputs != u.outputs) throw std::runtime_error("gate/up mismatch");
    if (g.fmt == 30U) dispatch_rows<wd::Format::bf16, dl::Epilogue::silu_mul>(rows, g, u, x, out, s);
    else if (g.fmt == 8U) dispatch_rows<wd::Format::q8_0, dl::Epilogue::silu_mul>(rows, g, u, x, out, s);
    else if (g.fmt == kIu4Fmt) {  // two projections that share one FP16 conversion, then SiLU(gate) * up
        if (!iu4.xh) throw std::runtime_error("iu4 scratch not allocated");
        if (core::dense_iu4::reference_flag()) {
            core::dense_iu4::reference(g.data, g.inputs, g.outputs, x, iu4.g, rows, s);
            core::dense_iu4::reference(u.data, u.inputs, u.outputs, x, iu4.u, rows, s);
            return silu_mul_kernel<<<256, 256, 0, s>>>(iu4.g, iu4.u, out, std::size_t(rows) * g.outputs);
        }
        core::dense_iu4::prepare(x, iu4.xh, g.inputs, rows, s);
        core::dense_iu4::run(g.data, g.inputs, g.outputs, iu4.xh, iu4.g, rows, s);
        core::dense_iu4::run(u.data, u.inputs, u.outputs, iu4.xh, iu4.u, rows, s);
        silu_mul_kernel<<<256, 256, 0, s>>>(iu4.g, iu4.u, out, std::size_t(rows) * g.outputs);
    } else throw std::runtime_error("gate/up need bf16, q8_0 or iu4");
}
// rows > 16 over IU4 weights: FAST dense-iu4a16-f16-v1 (f16_dense_iu4.hpp), the same opt-in as the Q8_0 wide windows
// (prefill_rows > 16 allocates the wide scratch). The exact reference flag keeps the 16-row sweeps.
core::f16_dense_tall::iu4a16::View wide_iu4_view(const Tensor& w, const float* x, float* out, unsigned rows) {
    core::f16_dense_tall::iu4a16::View v{};
    v.weight = static_cast<const std::uint8_t*>(w.data);
    v.scale = reinterpret_cast<const std::uint16_t*>(v.weight + core::dense_iu4::packed_bytes(w.outputs, w.inputs));
    v.x = x; v.out = out; v.in = w.inputs; v.n = w.outputs; v.count = w.outputs; v.rows = rows; v.out_stride = w.outputs;
    return v;
}
bool wide_iu4_ok(const Tensor& t, const WideScratch& wide, unsigned rows) {
    return wide.codes && t.fmt == kIu4Fmt && !core::dense_iu4::reference_flag() && t.inputs % 128U == 0U && t.outputs % 16U == 0U &&
           std::uint64_t(rows) * t.inputs * 4U < (1ULL << 32U);
}
void wide_iu4(const Tensor& w, const float* x, float* out, unsigned rows, hipStream_t s) {
    core::f16_dense_tall::iu4a16::enqueue(wide_iu4_view(w, x, out, rows), s);
}
void linear(const Tensor& w, const float* x, float* out, unsigned rows, hipStream_t s, const WideScratch& wide, const Iu4Scratch& iu4) {
    if (rows > 16U && wide_ok(w, wide)) {
        wide_quant(x, w.inputs, rows, s, wide);
        wide_dense(w, out, rows, s, wide);
    } else if (rows > 16U && wide_iu4_ok(w, wide, rows)) {
        wide_iu4(w, x, out, rows, s);
    } else {  // not a Q8_0 shape the WMMA core takes: exact 16-row sweeps
        core::dense_row_sweeps(rows, [&](unsigned r, unsigned count) {
            linear16(w, x + std::size_t(r) * w.inputs, out + std::size_t(r) * w.outputs, count, s, iu4);
        });
    }
}
void glu(const Tensor& g, const Tensor& u, const float* x, float* out, unsigned rows, hipStream_t s, const WideScratch& wide, const Iu4Scratch& iu4) {
    if (rows > 16U && wide_ok(g, wide) && wide_ok(u, wide) && g.outputs == u.outputs) {
        wide_quant(x, g.inputs, rows, s, wide);
        wide_dense(g, wide.g, rows, s, wide);
        wide_dense(u, wide.u, rows, s, wide);
        silu_mul_kernel<<<1024, 256, 0, s>>>(wide.g, wide.u, out, std::size_t(rows) * g.outputs);
    } else if (rows > 16U && wide_iu4_ok(g, wide, rows) && wide_iu4_ok(u, wide, rows) && g.outputs == u.outputs) {
        wide_iu4(g, x, wide.g, rows, s);
        wide_iu4(u, x, wide.u, rows, s);
        silu_mul_kernel<<<1024, 256, 0, s>>>(wide.g, wide.u, out, std::size_t(rows) * g.outputs);
    } else {
        core::dense_row_sweeps(rows, [&](unsigned r, unsigned count) {
            glu16(g, u, x + std::size_t(r) * g.inputs, out + std::size_t(r) * g.outputs, count, s, iu4);
        });
    }
}
void rmsnorm(const float* x, const Tensor& gain, unsigned n, float eps, float* y, unsigned rows, hipStream_t s) {
    ops::norm::rmsnorm_kernel<NormTraits, 256><<<rows, 256, 0, s>>>(x, reinterpret_cast<const float*>(gain.data), 0, n, eps, y);
}

// Raw GDN inputs of a window (qkv | beta | gate projections, before the in-place conv / transform) -> replay slot.
__global__ void capture_raw_kernel(const float* qkv, const float* beta, const float* gate, float* dq, float* db, float* dg,
                                   unsigned nq, unsigned nb) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < nq + 2U * nb; i += gridDim.x * blockDim.x) {
        if (i < nq) dq[i] = qkv[i];
        else if (i < nq + nb) db[i - nq] = beta[i - nq];
        else dg[i - nq - nb] = gate[i - nq - nb];
    }
}

// MTP input: out[r] = [a[r] (H) | b[r] (H)] (eh_proj consumes the concatenation enorm(embed) | hnorm(hidden)).
__global__ void concat2_kernel(const float* a, const float* b, float* out, unsigned rows, unsigned H) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * H; i += gridDim.x * blockDim.x) {
        const unsigned r = i / H, c = i % H;
        out[std::size_t(r) * 2U * H + c] = a[i];
        out[std::size_t(r) * 2U * H + H + c] = b[i];
    }
}

}  // namespace

struct HipEngine::Impl {
    HipEngineOptions opt;
    WideScratch wide;
    Iu4Scratch iu4;  // IU4 dense weights: FP16 activations + F32 gate/up windows of kMaxRows rows
    Config cfg;
    hipStream_t stream = nullptr;
    unsigned H = 0, V = 0, D = 0, QW = 0, KW = 0, I = 0, GV = 0, GC = 0, GH = 0;
    Tensor embd, onorm, head;
    std::vector<Layer> L;
    std::vector<unsigned> gdn_index;  // layer -> index among GDN layers (or ~0u)
    unsigned n_gdn = 0;
    std::size_t sb = 0, cb = 0;       // GDN matrix bytes / conv bytes per layer
    std::size_t pass_weight_bytes = 0, kv_pos_bytes = 0;
    float *gdn_all = nullptr, *bak_state = nullptr;
    std::uint16_t *conv_all = nullptr, *bak_conv = nullptr;
    float *raw_qkv = nullptr, *raw_beta = nullptr, *raw_gate = nullptr, *core_r = nullptr;
    float *x = nullptr, *xn = nullptr, *qkv = nullptr, *z = nullptr, *beta = nullptr, *gate = nullptr, *core = nullptr, *proj = nullptr,
          *q_in = nullptr, *kk = nullptr, *vv = nullptr, *qn = nullptr, *ctx = nullptr, *gu = nullptr, *logits = nullptr;
    int *d_ids = nullptr, *d_arg = nullptr;
    unsigned committed = 0, pending = 0;
    // Wide-window attention (rows > 16, FAST): shared flash over the BF16 cache (attn_core/dense_decode_flash.hpp).
    // vt_valid = positions whose V^T copy matches the cache (any other cache write lowers it).
    unsigned vt_valid = 0, vt_stride = 0;
    float* fws = nullptr;
    std::size_t fws_floats = 0;
    unsigned* ffail = nullptr;
    bool wide_flash(unsigned n) const {
        static const bool off = strix::envcfg::choice("STRIX_Q35_WIDE_ATTN", {"flash", "decode"}, 0) == 1;
        return n > kMaxRows && vt_stride && !off;
    }

    // MTP layer (load_mtp): own tensors, own bf16 KV (max_positions), own scratch; the trunk's embedding / lm_head are shared.
    struct MtpData {
        bool ok = false;
        Layer blk;
        Tensor enorm, hnorm, shnorm, eh;
        float *tA = nullptr, *tB = nullptr, *cat = nullptr, *mx = nullptr, *mxn = nullptr, *hpost = nullptr, *mlogits = nullptr;
        int *ids = nullptr, *top = nullptr;
        ~MtpData() {
            auto fr = [](auto* p) { if (p) (void)hipFree(p); };
            for (Tensor* t : {&blk.attn_norm, &blk.post_norm, &blk.gate, &blk.up, &blk.down, &blk.q, &blk.k, &blk.v, &blk.o, &blk.qnorm,
                              &blk.knorm, &enorm, &hnorm, &shnorm, &eh}) fr(t->data);
            fr(blk.k_cache); fr(blk.v_cache);
            for (auto* p : {tA, tB, cat, mx, mxn, hpost, mlogits}) fr(p);
            fr(ids); fr(top);
        }
    } mtp;

    ~Impl() {
        if (stream) (void)hipStreamSynchronize(stream);
        auto fr = [](auto* p) { if (p) (void)hipFree(p); };
        auto fn = [&](Tensor& t) { fr(t.data); };
        fn(embd); fn(onorm); fn(head);
        for (auto& y : L)
            for (Tensor* t : {&y.attn_norm, &y.post_norm, &y.gate, &y.up, &y.down, &y.qkv, &y.z, &y.beta, &y.alpha, &y.conv, &y.dt,
                              &y.a, &y.snorm, &y.out, &y.q, &y.k, &y.v, &y.o, &y.qnorm, &y.knorm}) fn(*t);
        for (auto& y : L) { fr(y.k_cache); fr(y.v_cache); fr(y.v_t); }
        fr(fws); fr(ffail);
        for (auto* p : {gdn_all, bak_state, raw_qkv, raw_beta, raw_gate, core_r, x, xn, qkv, z, beta, gate, core, proj, q_in, kk, vv,
                        qn, ctx, gu, logits}) fr(p);
        for (auto* p : {conv_all, bak_conv}) fr(p);
        fr(d_ids); fr(d_arg);
        fr(wide.codes); fr(wide.deltas); fr(wide.g); fr(wide.u);
        fr(iu4.xh); fr(iu4.g); fr(iu4.u);
        if (stream) (void)hipStreamDestroy(stream);
    }

    template <class T> static void alloc0(T** p, std::size_t bytes) {
        HK(hipMalloc(p, bytes));
        HK(hipMemset(*p, 0, bytes));
    }

    void load(const std::string& path) {
        Loader loader(path);
        cfg = qwen35::validate(loader.info(), opt.max_positions);
        qwen35::validate_backend(cfg, opt.max_positions);
        H = cfg.hidden;
        V = unsigned(loader.find("token_embd")->dimensions.at(1));
        D = cfg.head_dim; QW = cfg.heads * D; KW = cfg.kv_heads * D; I = cfg.intermediate;
        GV = cfg.gdn_v_width(); GC = cfg.gdn_conv_channels(); GH = cfg.gdn_value_heads;
        if (D != 256U) throw std::runtime_error("only head_dim 256 is instantiated");
        HK(hipStreamCreate(&stream));
        embd = loader.load("token_embd"); onorm = loader.load("output_norm"); head = loader.load("output");
        L.assign(cfg.layers, Layer{});
        gdn_index.assign(cfg.layers, ~0U);
        for (unsigned l = 0; l < cfg.layers; ++l)
            if (cfg.kind(l) == LayerKind::gated_delta_net) gdn_index[l] = n_gdn++;
        sb = std::size_t(GH) * cfg.gdn_key_dim * cfg.gdn_value_dim * 4U;
        cb = std::size_t(GC) * cfg.gdn_conv_kernel * 2U;
        alloc0(&gdn_all, sb * n_gdn);
        alloc0(&conv_all, cb * n_gdn);
        alloc0(&bak_state, sb * n_gdn);
        alloc0(&bak_conv, cb * n_gdn);
        alloc0(&raw_qkv, std::size_t(n_gdn) * kMaxRows * GC * 4U);
        alloc0(&raw_beta, std::size_t(n_gdn) * kMaxRows * GH * 4U);
        alloc0(&raw_gate, std::size_t(n_gdn) * kMaxRows * GH * 4U);
        for (unsigned l = 0; l < cfg.layers; ++l) {
            const std::string b = "blk." + std::to_string(l) + ".";
            auto& y = L[l];
            y.attn_norm = loader.load(b + "attn_norm");
            y.post_norm = loader.load(b + "post_attention_norm");
            y.gate = loader.load(b + "ffn_gate");
            y.up = loader.load(b + "ffn_up");
            y.down = loader.load(b + "ffn_down");
            if (gdn_index[l] != ~0U) {
                y.qkv = loader.load(b + "attn_qkv");
                y.z = loader.load(b + "attn_gate");
                y.beta = loader.load(b + "ssm_beta");
                y.alpha = loader.load(b + "ssm_alpha");
                y.conv = loader.load(b + "ssm_conv1d");
                y.dt = loader.load(b + "ssm_dt");
                y.a = loader.load(b + "ssm_a");
                y.snorm = loader.load(b + "ssm_norm");
                y.out = loader.load(b + "ssm_out");
                for (const Tensor* t : {&y.conv, &y.dt, &y.a, &y.snorm})
                    if (t->fmt != 0U) throw std::runtime_error("GDN small tensors must be F32");
                y.gdn_state = gdn_all + std::size_t(gdn_index[l]) * sb / 4U;
                y.conv_state = conv_all + std::size_t(gdn_index[l]) * cb / 2U;
            } else {
                y.q = loader.load(b + "attn_q");
                y.k = loader.load(b + "attn_k");
                y.v = loader.load(b + "attn_v");
                y.o = loader.load(b + "attn_output");
                y.qnorm = loader.load(b + "attn_q_norm");
                y.knorm = loader.load(b + "attn_k_norm");
                const std::size_t kb = std::size_t(opt.max_positions) * KW * 2U;
                alloc0(&y.k_cache, kb);
                alloc0(&y.v_cache, kb);
                kv_pos_bytes += 2U * KW * 2U;
            }
        }
        pass_weight_bytes = loader.total() - embd.bytes;
        if (!opt.quiet)
            std::printf("loaded %.2f GiB (layers %u, hidden %u, vocab %u, GDN layers %u)\n", double(loader.total()) / double(1ULL << 30),
                        cfg.layers, H, V, n_gdn);
        const unsigned RS = std::max(kMaxRows, opt.prefill_rows);  // scratch rows (wide prefill windows)
        auto fm = [&](float** p, std::size_t n) { alloc0(p, n * RS * 4U); };
        fm(&x, H); fm(&xn, H); fm(&qkv, GC); fm(&z, GV); fm(&beta, GH); fm(&gate, GH); fm(&core, GV); fm(&proj, H);
        fm(&q_in, 2U * QW); fm(&kk, KW); fm(&vv, KW); fm(&qn, QW); fm(&ctx, QW); fm(&gu, I); alloc0(&logits, std::size_t(V) * kMaxRows * 4U);
        fm(&core_r, GV);
        alloc0(&d_ids, RS * 4U);
        if (RS > kMaxRows) {
            std::size_t maxin = std::max<std::size_t>(H, I);
            alloc0(&wide.codes, std::size_t(RS) * maxin);
            alloc0(&wide.deltas, std::size_t(RS) * (maxin / 32U) * 4U);
            alloc0(&wide.g, std::size_t(RS) * I * 4U);
            alloc0(&wide.u, std::size_t(RS) * I * 4U);
            // flash attention of the wide windows: geometry the shared adapter is instantiated for (Dim 256, GQA group 6)
            if (D == 256U && cfg.heads == 6U * cfg.kv_heads) {
                vt_stride = (opt.max_positions + 15U) / 16U * 16U;
                for (unsigned l = 0; l < cfg.layers; ++l)
                    if (gdn_index[l] == ~0U) alloc0(&L[l].v_t, std::size_t(KW) * vt_stride * 2U);
                fws_floats = attn_core::dense::workspace_floats(RS, cfg.heads, D, 4U);
                alloc0(&fws, fws_floats * 4U);
                alloc0(&ffail, 4U);
            }
        }
        {  // IU4 dense weights; sized for the widest dense input / gate-up output of a kMaxRows window (~3 MB)
            alloc0(&iu4.xh, std::size_t(kMaxRows) * std::max<std::size_t>(H, I) * sizeof(_Float16));
            alloc0(&iu4.g, std::size_t(kMaxRows) * I * 4U);
            alloc0(&iu4.u, std::size_t(kMaxRows) * I * 4U);
        }
        alloc0(&d_arg, kMaxRows * 4U);
    }

    void reset_state() {
        HK(hipMemsetAsync(gdn_all, 0, sb * n_gdn, stream));
        HK(hipMemsetAsync(conv_all, 0, cb * n_gdn, stream));
        committed = pending = 0;
        HK(hipStreamSynchronize(stream));
    }

    // Restore the pre-window GDN state and replay the first k rows of the window (k may be 0).
    void restore_and_replay(unsigned k) {
        HK(hipMemcpyAsync(gdn_all, bak_state, sb * n_gdn, hipMemcpyDeviceToDevice, stream));
        HK(hipMemcpyAsync(conv_all, bak_conv, cb * n_gdn, hipMemcpyDeviceToDevice, stream));
        if (k == 0U) return;
        for (unsigned l = 0; l < cfg.layers; ++l) {
            if (gdn_index[l] == ~0U) continue;
            const std::size_t gi = gdn_index[l];
            auto& y = L[l];
            HipGdnDimsStep g{};
            g.qkv = raw_qkv + gi * kMaxRows * GC; g.beta = raw_beta + gi * kMaxRows * GH; g.gate = raw_gate + gi * kMaxRows * GH;
            g.core = core_r; g.matrix = y.gdn_state; g.z = z;
            g.conv_state = y.conv_state; g.conv = y.conv.data; g.dt = y.dt.data; g.a = y.a.data; g.norm = y.snorm.data;
            g.small_type = 0; g.key_heads = cfg.gdn_key_heads; g.value_heads = GH; g.head_dim = cfg.gdn_key_dim;
            g.kernel = cfg.gdn_conv_kernel; g.rows = k; g.qk_epsilon = 1e-6F; g.rms_epsilon = cfg.rms_eps;
            g.norm_kind = HipGdnGatedNorm::swish;
            hip_gdn_decode_step_dims(g, stream);
        }
    }

    HipRowsOut run(const std::uint32_t* tokens, unsigned n, LogitRows want, bool defer) {
        if (n < 1U || n > (defer ? kMaxRows : std::max(kMaxRows, opt.prefill_rows)))
            throw std::invalid_argument("rows must be 1..16 (decode windows) or 1..prefill_rows (forward_commit)");
        if (n > kMaxRows && want == LogitRows::all) throw std::invalid_argument("wide windows return last-row or no logits");
        if (pending) { restore_and_replay(0U); pending = 0; }  // an uncommitted window is discarded
        core::require_capacity(committed, n, opt.max_positions);
        core::validate_token_ids(tokens, n, cfg.vocab);
        const unsigned pos = committed;
        if (defer) {
            HK(hipMemcpyAsync(bak_state, gdn_all, sb * n_gdn, hipMemcpyDeviceToDevice, stream));
            HK(hipMemcpyAsync(bak_conv, conv_all, cb * n_gdn, hipMemcpyDeviceToDevice, stream));
        }
        HK(hipMemcpyAsync(d_ids, tokens, n * 4U, hipMemcpyHostToDevice, stream));
        embed_kernel<<<n, 256, 0, stream>>>(embd.data, embd.fmt, embd.row_bytes, H, d_ids, x);
        const unsigned nb = (n * H + 255U) / 256U;
        const bool flash = wide_flash(n);
        const unsigned vt_from = std::min(vt_valid, pos);
        vt_valid = flash ? pos + n : vt_from;
        for (unsigned l = 0; l < cfg.layers; ++l) {
            auto& y = L[l];
            rmsnorm(x, y.attn_norm, H, cfg.rms_eps, xn, n, stream);
            if (gdn_index[l] != ~0U) {
                linear(y.qkv, xn, qkv, n, stream, wide, iu4);
                linear(y.z, xn, z, n, stream, wide, iu4);
                linear(y.beta, xn, beta, n, stream, wide, iu4);
                linear(y.alpha, xn, gate, n, stream, wide, iu4);
                if (defer) {
                    const std::size_t gi = gdn_index[l];
                    capture_raw_kernel<<<64, 256, 0, stream>>>(qkv, beta, gate, raw_qkv + gi * kMaxRows * GC, raw_beta + gi * kMaxRows * GH,
                                                               raw_gate + gi * kMaxRows * GH, n * GC, n * GH);
                }
                HipGdnDimsStep g{};
                g.qkv = qkv; g.beta = beta; g.gate = gate; g.core = core; g.matrix = y.gdn_state; g.z = z;
                g.conv_state = y.conv_state; g.conv = y.conv.data; g.dt = y.dt.data; g.a = y.a.data; g.norm = y.snorm.data;
                g.small_type = 0; g.key_heads = cfg.gdn_key_heads; g.value_heads = GH; g.head_dim = cfg.gdn_key_dim;
                g.kernel = cfg.gdn_conv_kernel; g.rows = n; g.qk_epsilon = 1e-6F; g.rms_epsilon = cfg.rms_eps;
                g.norm_kind = HipGdnGatedNorm::swish;
                hip_gdn_decode_step_dims(g, stream);
                linear(y.out, core, proj, n, stream, wide, iu4);
            } else {
                linear(y.q, xn, q_in, n, stream, wide, iu4);
                linear(y.k, xn, kk, n, stream, wide, iu4);
                linear(y.v, xn, vv, n, stream, wide, iu4);
                attn_core::dense_decode::Params p{n, cfg.heads, cfg.kv_heads, 2U * D, cfg.rotary_dim, pos, cfg.rms_eps,
                                                  1.0F / std::sqrt(float(D)), cfg.rope_theta, true};
                if (flash)
                    attn_core::dense_decode_flash::launch<256, 6>(p, q_in, kk, vv, reinterpret_cast<const float*>(y.qnorm.data),
                                                                  reinterpret_cast<const float*>(y.knorm.data), qn, y.k_cache, y.v_cache,
                                                                  y.v_t, vt_stride, vt_from, ffail, fws, fws_floats, ctx, stream);
                else
                    attn_core::dense_decode::launch<256>(p, q_in, kk, vv, reinterpret_cast<const float*>(y.qnorm.data),
                                                         reinterpret_cast<const float*>(y.knorm.data), qn, y.k_cache, y.v_cache, ctx, stream);
                linear(y.o, ctx, proj, n, stream, wide, iu4);
            }
            add_kernel<<<nb, 256, 0, stream>>>(x, proj, n * H);
            rmsnorm(x, y.post_norm, H, cfg.rms_eps, xn, n, stream);
            glu(y.gate, y.up, xn, gu, n, stream, wide, iu4);
            linear(y.down, gu, proj, n, stream, wide, iu4);
            add_kernel<<<nb, 256, 0, stream>>>(x, proj, n * H);
        }
        HipRowsOut out;
        if (mtp.ok) {  // trunk hidden of every row for the MTP chain (same final norm as the logits path)
            rmsnorm(x, onorm, H, cfg.rms_eps, mtp.hpost, n, stream);
            out.hidden_post = mtp.hpost;
            out.hidden_pre = x;
        }
        if (want != LogitRows::none) {
            const unsigned first = want == LogitRows::last ? n - 1U : 0U, m = want == LogitRows::last ? 1U : n;
            rmsnorm(x + std::size_t(first) * H, onorm, H, cfg.rms_eps, xn, m, stream);
            linear(head, xn, logits, m, stream, wide, iu4);
            argmax_kernel<<<m, 1024, 0, stream>>>(logits, V, d_arg);
            out.logits = logits;
            out.top = reinterpret_cast<const std::uint32_t*>(d_arg);
        }
        HK(hipGetLastError());
        if (defer) pending = n;
        else committed += n;
        return out;
    }

    void load_mtp(const std::string& path) {
        if (mtp.ok) throw std::logic_error("load_mtp: already loaded");
        Loader loader(path);
        const std::string b = "blk." + std::to_string(cfg.layers) + ".";
        auto& y = mtp.blk;
        y.attn_norm = loader.load(b + "attn_norm");
        y.post_norm = loader.load(b + "post_attention_norm");
        y.gate = loader.load(b + "ffn_gate");
        y.up = loader.load(b + "ffn_up");
        y.down = loader.load(b + "ffn_down");
        y.q = loader.load(b + "attn_q");
        y.k = loader.load(b + "attn_k");
        y.v = loader.load(b + "attn_v");
        y.o = loader.load(b + "attn_output");
        y.qnorm = loader.load(b + "attn_q_norm");
        y.knorm = loader.load(b + "attn_k_norm");
        mtp.enorm = loader.load(b + "nextn.enorm");
        mtp.hnorm = loader.load(b + "nextn.hnorm");
        mtp.shnorm = loader.load(b + "nextn.shared_head_norm");
        mtp.eh = loader.load(b + "nextn.eh_proj");
        for (const Tensor* t : {&y.attn_norm, &y.post_norm, &y.qnorm, &y.knorm, &mtp.enorm, &mtp.hnorm, &mtp.shnorm})
            if (t->fmt != 0U) throw std::runtime_error("MTP norms must be F32");
        if (mtp.eh.inputs != 2U * H || mtp.eh.outputs != H || y.q.outputs != 2U * QW || y.k.outputs != KW || y.gate.outputs != I)
            throw std::runtime_error("MTP tensor shapes do not match the trunk config");
        const std::size_t kb = std::size_t(opt.max_positions) * KW * 2U;
        alloc0(&y.k_cache, kb);
        alloc0(&y.v_cache, kb);
        const unsigned RS = std::max(kMaxRows, opt.prefill_rows);
        alloc0(&mtp.tA, std::size_t(kMaxRows) * H * 4U);
        alloc0(&mtp.tB, std::size_t(kMaxRows) * H * 4U);
        alloc0(&mtp.cat, std::size_t(kMaxRows) * 2U * H * 4U);
        alloc0(&mtp.mx, std::size_t(kMaxRows) * H * 4U);
        alloc0(&mtp.mxn, std::size_t(kMaxRows) * H * 4U);
        alloc0(&mtp.hpost, std::size_t(RS) * H * 4U);
        alloc0(&mtp.mlogits, std::size_t(kMaxRows) * V * 4U);
        alloc0(&mtp.ids, kMaxRows * 4U);
        alloc0(&mtp.top, kMaxRows * 4U);
        mtp.ok = true;
        if (!opt.quiet) std::printf("mtp loaded %.2f GiB (blk.%u)\n", double(loader.total()) / double(1ULL << 30), cfg.layers);
    }

    // Same math as the CUDA MTP layer: cat(enorm(embed(tok)), hnorm(hidden)) -> eh_proj -> one full-attention block (own KV) ->
    // shared_head_norm -> trunk lm_head. Scratch of the trunk window (q_in, kk, vv, qn, ctx, gu, proj) is reused: calls are
    // stream-ordered between passes.
    HipMtpOut mtp_run(const std::uint32_t* next_tokens, const float* hidden_dev, unsigned n, unsigned pos0, bool want_top) {
        if (!mtp.ok) throw std::runtime_error("mtp_forward: no MTP tensors loaded");
        if (n < 1U || n > kMaxRows) throw std::invalid_argument("mtp_forward: rows must be 1..16");
        core::require_capacity(pos0, n, opt.max_positions);
        core::validate_token_ids(next_tokens, n, cfg.vocab);
        auto& y = mtp.blk;
        HK(hipMemcpyAsync(mtp.ids, next_tokens, n * 4U, hipMemcpyHostToDevice, stream));
        embed_kernel<<<n, 256, 0, stream>>>(embd.data, embd.fmt, embd.row_bytes, H, mtp.ids, mtp.tA);
        rmsnorm(mtp.tA, mtp.enorm, H, cfg.rms_eps, mtp.mxn, n, stream);
        rmsnorm(hidden_dev, mtp.hnorm, H, cfg.rms_eps, mtp.tB, n, stream);
        concat2_kernel<<<(n * H + 255U) / 256U, 256, 0, stream>>>(mtp.mxn, mtp.tB, mtp.cat, n, H);
        linear(mtp.eh, mtp.cat, mtp.mx, n, stream, wide, iu4);
        const unsigned nb = (n * H + 255U) / 256U;
        rmsnorm(mtp.mx, y.attn_norm, H, cfg.rms_eps, mtp.mxn, n, stream);
        linear(y.q, mtp.mxn, q_in, n, stream, wide, iu4);
        linear(y.k, mtp.mxn, kk, n, stream, wide, iu4);
        linear(y.v, mtp.mxn, vv, n, stream, wide, iu4);
        attn_core::dense_decode::Params p{n, cfg.heads, cfg.kv_heads, 2U * D, cfg.rotary_dim, pos0, cfg.rms_eps,
                                          1.0F / std::sqrt(float(D)), cfg.rope_theta, true};
        attn_core::dense_decode::launch<256>(p, q_in, kk, vv, reinterpret_cast<const float*>(y.qnorm.data),
                                             reinterpret_cast<const float*>(y.knorm.data), qn, y.k_cache, y.v_cache, ctx, stream);
        linear(y.o, ctx, proj, n, stream, wide, iu4);
        add_kernel<<<nb, 256, 0, stream>>>(mtp.mx, proj, n * H);
        rmsnorm(mtp.mx, y.post_norm, H, cfg.rms_eps, mtp.mxn, n, stream);
        glu(y.gate, y.up, mtp.mxn, gu, n, stream, wide, iu4);
        linear(y.down, gu, proj, n, stream, wide, iu4);
        add_kernel<<<nb, 256, 0, stream>>>(mtp.mx, proj, n * H);
        rmsnorm(mtp.mx, mtp.shnorm, H, cfg.rms_eps, mtp.mxn, n, stream);  // mx (pre-norm block output) stays intact for 'pre'
        HipMtpOut o;
        if (want_top) {
            linear(head, mtp.mxn, mtp.mlogits, n, stream, wide, iu4);
            argmax_kernel<<<n, 1024, 0, stream>>>(mtp.mlogits, V, mtp.top);
            o.logits = mtp.mlogits;
            o.top = reinterpret_cast<const std::uint32_t*>(mtp.top);
        }
        HK(hipGetLastError());
        o.hidden_pre = mtp.mx;
        o.hidden_post = mtp.mxn;
        return o;
    }
};

HipEngine::HipEngine() : p_(std::make_unique<Impl>()) {}
HipEngine::~HipEngine() = default;

std::unique_ptr<HipEngine> HipEngine::from_gguf(const std::string& path, const HipEngineOptions& opt) {
    std::unique_ptr<HipEngine> e(new HipEngine());
    e->p_->opt = opt;
    e->p_->load(path);
    return e;
}
const Config& HipEngine::config() const { return p_->cfg; }
std::uint32_t HipEngine::vocab() const { return p_->V; }
std::uint32_t HipEngine::prefill_rows() const { return std::max(kMaxRows, p_->opt.prefill_rows); }
std::uint32_t HipEngine::committed() const { return p_->committed; }
std::uint64_t HipEngine::weight_bytes() const { return p_->pass_weight_bytes; }
std::uint64_t HipEngine::state_bytes_per_pass() const { return 2ULL * p_->sb * p_->n_gdn; }
std::uint32_t HipEngine::kv_bytes_per_position() const { return std::uint32_t(p_->kv_pos_bytes); }
void HipEngine::reset() { p_->reset_state(); }
HipRowsOut HipEngine::forward_rows(const std::uint32_t* t, std::uint32_t rows, LogitRows want) { return p_->run(t, rows, want, true); }
HipRowsOut HipEngine::forward_commit(const std::uint32_t* t, std::uint32_t rows, LogitRows want) { return p_->run(t, rows, want, false); }

void HipEngine::commit(std::uint32_t k) {
    auto& p = *p_;
    if (!p.pending) throw std::logic_error("commit without a pending window");
    if (k > p.pending) throw std::invalid_argument("commit(k): k > rows");
    if (k < p.pending) p.restore_and_replay(k);
    p.committed += k;
    p.pending = 0;
}
void HipEngine::read_top(const HipRowsOut& o, std::uint32_t rows, std::uint32_t* host) const {
    HK(hipMemcpyAsync(host, o.top, rows * 4U, hipMemcpyDeviceToHost, p_->stream));
    HK(hipStreamSynchronize(p_->stream));
}
void HipEngine::read_logits(const HipRowsOut& o, std::uint32_t rows, float* host) const {
    HK(hipMemcpyAsync(host, o.logits, std::size_t(rows) * p_->V * 4U, hipMemcpyDeviceToHost, p_->stream));
    HK(hipStreamSynchronize(p_->stream));
}
void HipEngine::bench_seek(std::uint32_t pos) {
    if (pos > p_->opt.max_positions) throw std::invalid_argument("bench_seek beyond max_positions");
    p_->committed = pos;
    p_->pending = 0;
}
void HipEngine::sync() const { HK(hipStreamSynchronize(p_->stream)); }

void HipEngine::load_mtp(const std::string& path) { p_->load_mtp(path); }
bool HipEngine::has_mtp() const { return p_->mtp.ok; }
HipMtpOut HipEngine::mtp_forward(const std::uint32_t* t, const float* hidden_dev, std::uint32_t rows, std::uint32_t pos0, bool want_top) {
    return p_->mtp_run(t, hidden_dev, rows, pos0, want_top);
}
float* HipEngine::alloc_f32(std::size_t n) const {
    float* q = nullptr;
    HK(hipMalloc(&q, n * sizeof(float)));
    return q;
}
void HipEngine::free_f32(float* q) const { (void)hipFree(q); }
void HipEngine::copy_f32_async(float* dst, const float* src, std::size_t n) const {
    HK(hipMemcpyAsync(dst, src, n * sizeof(float), hipMemcpyDeviceToDevice, p_->stream));
}

}  // namespace strix::qwen35
