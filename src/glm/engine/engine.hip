#include <atomic>
#include <limits>
#include <array>
#include <exception>
#include "xport/channel.hpp"
#include "xport/communicator_backend.hpp"
#include "glm/xfer/glm_traits.hpp"
// GLM-5.3-Flash engine (see include/strix/glm/engine.hpp, design notes).
//
// Skeleton v1: correctness first. Per layer: mHC pre (Tier A) -> RMSNorm -> attention seam -> TP reduction -> mHC
// expand -> mHC pre -> RMSNorm -> dense MLP (strix_gemv) or router (Tier A) + MoE seam -> reduction -> mHC expand. Head:
// final collapse -> RMSNorm -> lm_head vocab slice (strix_gemv) -> local argmax -> argmax exchange. The default seams
// process one row at a time with the Tier A kernels (bitwise to src/glm/host) and strix_gemv (Tier B reference); the
// Optimized implementations replace them with fused / batched paths that must reproduce these bits.
// Weights: O_DIRECT reads through one pinned staging buffer (MiniMax DirectLoader pattern), rank slices per
// tools/glm/split_plan.py (H128-aligned EXL3 slices, plain tensors by rows).
#include "strix/glm/engine.hpp"

#include "glm/engine/glm_fast_attn.hpp"
#include "glm/engine/glm_fast_moe.hpp"
#include "glm/engine/glm_fast_i4_pack.hpp"
#include "core/iu4.hpp"
#include "moe_core/int4_layer_policy.hpp"
#include "glm/prefill/glm_pf_exact_moe.hpp"
#include "glm/prefill/glm_pf_wmma.hpp"
#include "glm/engine/glm_lib.hpp"
#include "glm/engine/glm_pack.hpp"
#include "glm/engine/glm_r1b.hpp"
#include "glm/engine/glm_kvshard.hpp"
#include "glm/prefill/glm_pf_capacity_policy.hpp"
#include "glm/engine/glm_kvshard_exchange.hpp"
#include "glm/kernels/glm_dsa.hpp"
#include "../kernels/glm_norm.hpp"
#include "glm/load/glm_model_description.hpp"
#include "glm/moe/glm_moe.hpp"
#include "glm/prefill/glm_pf_moe.hpp"
#include "glm/prefill/glm_prefill.hpp"
#include "glm/spec/glm_spec_dev.hpp"
#include "glm/xfer/glm_post_fast.hpp"
#include "glm/xfer/glm_xfer_exchange.hpp"
#include "glm/xfer/glm_post_exchange.hpp"
#include "strix/dist/communicator.hpp"

#include <strix_trellis.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>

namespace strix::glm::engine {

// The explicit host adapter checks the description against the common model
// dimensions. Also bind the duplicated TP2 MoE and exchange constants here,
// without changing their kernel headers or their compiled launch schedules.
static_assert(moe::kHidden == kHidden && moe::kExperts == kExperts && moe::kTopE == kTopE && moe::kInter * 2 == kExpertI);
static_assert(pf::kMoeHidden == kHidden && pf::kMoeExperts == kExperts && pf::kMoeTop == kTopE && pf::kMoeInter * 2 == kExpertI);
static_assert(xfer::GlmSchedule::kLayers == kLayers && xfer::kRowF32 == std::size_t{kHidden} * sizeof(float));
static_assert(kEps == static_cast<float>(model::contract::rms_eps) && kHcEps == static_cast<float>(model::contract::hc_eps));
static_assert(kL2Eps == static_cast<float>(model::contract::l2_eps) && kLnEps == static_cast<float>(model::contract::ln_eps));
static_assert(kGateLow == static_cast<float>(model::contract::gate_lower_bound) && kClamp == static_cast<float>(model::contract::clamp));
static_assert(kRouteScale == static_cast<float>(model::contract::route_scale) && kRouteDenEps == static_cast<float>(model::contract::route_den_eps));
static_assert(kInvSqrt128 == static_cast<float>(model::contract::inv_sqrt128) && kInvSqrt32 == static_cast<float>(model::contract::inv_sqrt32));
static_assert(kMlaScale == static_cast<float>(model::contract::mla_scale));

namespace {

using namespace detail;
namespace D = strix::glm::dev;
using Clock = std::chrono::steady_clock;

const std::string kP = "model.language_model.";

// ---------------------------------------------------------------- small engine-only kernels
__global__ void embed4_kernel(const std::uint16_t* table, const std::uint32_t* tokens, float* X) {
    const unsigned r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d >= kHidden) return;
    const float v = bf16_to_float(table[static_cast<std::size_t>(tokens[r]) * kHidden + d]);
    for (unsigned c = 0; c < kHc; ++c) X[(static_cast<std::size_t>(r) * kHc + c) * kHidden + d] = v;  // 4 identical streams
}
__global__ void embed1_kernel(const std::uint16_t* table, const std::uint32_t* tokens, float* e) {
    const unsigned r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d < kHidden) e[static_cast<std::size_t>(r) * kHidden + d] = bf16_to_float(table[static_cast<std::size_t>(tokens[r]) * kHidden + d]);
}
__global__ void cat2_kernel(const float* a, const float* b, float* y) {  // y[r] = [a[r] | b[r]], 2 x 4096
    const unsigned r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d >= kHidden) return;
    y[static_cast<std::size_t>(r) * 2 * kHidden + d] = a[static_cast<std::size_t>(r) * kHidden + d];
    y[static_cast<std::size_t>(r) * 2 * kHidden + kHidden + d] = b[static_cast<std::size_t>(r) * kHidden + d];
}
__global__ void add_kernel(const float* a, const float* b, float* y, std::size_t n) {  // residual + branch (MTP, no mHC)
    const std::size_t i = blockIdx.x * std::size_t{256} + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}
// Drafter confidence: max M and Z = sum exp(l - M) of one row's rank slice (fixed tree order: identical per rank).
__global__ __launch_bounds__(256) void lse_kernel(const float* l, unsigned n, float* mz) {
    __shared__ float s[256];
    const unsigned t = threadIdx.x;
    float m = -__builtin_huge_valf();
    for (unsigned i = t; i < n; i += 256) m = fmaxf(m, l[i]);
    s[t] = m;
    __syncthreads();
    for (unsigned h = 128; h > 0; h >>= 1U) {
        if (t < h) s[t] = fmaxf(s[t], s[t + h]);
        __syncthreads();
    }
    const float M = s[0];
    __syncthreads();
    float z = 0.0F;
    for (unsigned i = t; i < n; i += 256) z += expf(l[i] - M);
    s[t] = z;
    __syncthreads();
    for (unsigned h = 128; h > 0; h >>= 1U) {
        if (t < h) s[t] = s[t] + s[t + h];
        __syncthreads();
    }
    if (t == 0) {
        mz[0] = M;
        mz[1] = s[0];
    }
}
__global__ void sum2_kernel(const float* p0, const float* p1, float* y, std::size_t n) {  // R17: p0 + p1, rank order
    const std::size_t i = blockIdx.x * std::size_t{256} + threadIdx.x;
    if (i < n) y[i] = p0[i] + p1[i];
}
// Staged prefill exchange: the partial goes into the communicator's coherent TX view on the compute stream.
__global__ void stage_copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, std::size_t n4) {
    const std::size_t i = blockIdx.x * std::size_t{256} + threadIdx.x;
    if (i < n4) dst[i] = src[i];
}
// STRIX_GLM_PF_STAGED (default on, =0 off): pipelined prefill exchanges through the communicator's coherent staging
// views (one copy kernel into TX, sum2 reads RX in place) instead of the async chunked D2H / H2D copies, which a side
// stream only gets between the long prefill kernels. Same p0 + p1 adds in rank order, same bits.
bool pf_staged_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRIX_GLM_PF_STAGED");
        if (v == nullptr || *v == '\0') return true;
        const std::string s(v);
        if (s != "0" && s != "1") throw std::invalid_argument("STRIX_GLM_PF_STAGED must be 0 or 1");
        return s == "1";
    }();
    return on;
}

// ---------------------------------------------------------------- default seams (bitwise references)
struct Scratch {  // per-row scratch shared by the default seams (one row at a time)
    float *qkv, *f, *g, *b, *t128a, *t128b, *kda_scr, *beta, *o, *y1;
    float *qa, *qr, *q, *kva, *lat, *qI, *wh, *scores, *qt, *key, *gate;
    std::uint32_t *work, *tokens, *ntok;
    D::DsaSlot* slot_d;
    unsigned* t_d;
    float *eo, *ge, *ue, *ae, *sh;
    std::uint64_t cap;
};

class DefaultAttn final : public AttnPath {
public:
    explicit DefaultAttn(std::shared_ptr<Scratch> s) : s_(std::move(s)) {}
    void kda(const KdaLayerWeights& w, KdaState& st, const CallCtx& c, const float* x, float* y) override {
        if (c.spec) throw std::logic_error("speculative verify runs on the FastAttn seam (STRIX_GLM_ATTN=ref is plain only)");
        const unsigned H = w.heads, C = H * kKdaDim;
        const D::KdaDev kd{H, w.conv, w.A_log, w.dt_bias, w.o_norm};
        for (std::size_t r = 0; r < c.rows.size(); ++r) {
            const float* xr = x + r * kHidden;
            const std::uint32_t slot = c.rows[r].slot;
            for (unsigned p = 0; p < 3; ++p) c.lin->run(w.qkv[p], xr, s_->qkv + p * C, 1, c.stream);
            if (strix::glm::r1b::active()) {  // R1b generated identity (default; prefill scopes R1): the K = 4096 chains blocked; f_b / g_b (K 128) unchanged
                const int selected = strix::glm::r1b::dense_variant();
                const bool packed = (selected == 5 || selected == 6 || selected == 7) && w.f_a_packed != nullptr &&
                                    w.g_a_packed != nullptr && w.b_packed != nullptr;
                const int variant = selected == 5 || selected == 6 || selected == 7 ? (packed ? selected : 2) : selected;
                strix::glm::r1b::dense_f16_k4096(xr, kHidden, packed ? w.f_a_packed : w.f_a,
                                               kKdaDim, s_->t128a, kKdaDim, 1, c.stream, variant);
                strix::glm::r1b::dense_f16_k4096(xr, kHidden, packed ? w.g_a_packed : w.g_a,
                                               kKdaDim, s_->t128b, kKdaDim, 1, c.stream, variant);
                strix::glm::r1b::dense_f16_k4096(xr, kHidden, packed ? w.b_packed : w.b,
                                               H, s_->b, H, 1, c.stream, variant);
            } else {
                D::dense(xr, w.f_a, D::WType::F16, kHidden, 0, kKdaDim, s_->t128a, 1, c.stream);
                D::dense(xr, w.g_a, D::WType::F16, kHidden, 0, kKdaDim, s_->t128b, 1, c.stream);
                D::dense(xr, w.b, D::WType::F16, kHidden, 0, H, s_->b, 1, c.stream);
            }
            D::dense(s_->t128a, w.f_b, D::WType::F16, kKdaDim, 0, C, s_->f, 1, c.stream);
            D::dense(s_->t128b, w.g_b, D::WType::F16, kKdaDim, 0, C, s_->g, 1, c.stream);
            float* S = st.S + static_cast<std::size_t>(slot) * H * kKdaDim * kKdaDim;
            float* cv = st.conv + static_cast<std::size_t>(slot) * 3 * C * 3;
            D::kda_step(kd, s_->qkv, s_->f, s_->g, s_->b, S, cv, s_->kda_scr, s_->beta, s_->o, s_->y1, c.flag, 1, c.stream);
            c.lin->run(w.o, s_->y1, y + r * kHidden, 1, c.stream);
        }
    }
    void dsa(const DsaLayerWeights& w, DsaState& st, const CallCtx& c, const float* x, float* y) override {
        if (c.spec) throw std::logic_error("speculative verify runs on the FastAttn seam (STRIX_GLM_ATTN=ref is plain only)");
        const unsigned H = w.heads;
        const D::IndexerDev iw{w.wk, w.gate, w.weights_proj, w.k_norm_w, w.k_norm_b, w.ape};
        for (std::size_t r = 0; r < c.rows.size(); ++r) {
            const float* xr = x + r * kHidden;
            const std::uint32_t slot = c.rows[r].slot;
            const auto t = static_cast<unsigned>(c.rows[r].pos);
            if (t >= st.cap) throw std::out_of_range("DSA position beyond the slot capacity");
            c.lin->run(w.q_a, xr, s_->qa, 1, c.stream);
            D::GlmNorm::run(s_->qa, w.q_a_norm, D::WType::BF16, kQLora, kEps, s_->qr, 1, c.stream);
            c.lin->run(w.q_b, s_->qr, s_->q, 1, c.stream);
            c.lin->run(w.kv_a, xr, s_->kva, 1, c.stream);
            D::GlmNorm::run(s_->kva, w.kv_a_norm, D::WType::BF16, kKvLora, kEps, s_->lat, 1, c.stream);
            c.lin->run(w.wq_b, s_->qr, s_->qI, 1, c.stream);
            const D::DsaSlot sl{st.latent + static_cast<std::size_t>(slot) * st.cap * kKvLora,
                                st.ikey + static_cast<std::size_t>(slot) * st.cap * kIdxDim,
                                st.igate + static_cast<std::size_t>(slot) * st.cap * kIdxDim,
                                st.pooled + static_cast<std::size_t>(slot) * (st.cap / kKpool) * kIdxDim, static_cast<unsigned>(st.cap)};
            hip_ok(hipMemcpyAsync(s_->slot_d, &sl, sizeof(sl), hipMemcpyHostToDevice, c.stream), "slot");
            hip_ok(hipMemcpyAsync(s_->t_d, &t, sizeof(t), hipMemcpyHostToDevice, c.stream), "t");
            D::dsa_append(s_->slot_d, s_->t_d, s_->lat, xr, iw, s_->key, s_->gate, c.flag, 1, c.stream);
            D::indexer_select(sl, t, s_->qI, xr, iw, s_->wh, s_->scores, s_->work, s_->tokens, s_->ntok, c.stream);
            D::mla_attention(sl, s_->tokens, s_->ntok, s_->q, w.kv_b, H, s_->qt, s_->o, c.stream);
            c.lin->run(w.o, s_->o, y + r * kHidden, 1, c.stream);
            hip_ok(hipStreamSynchronize(c.stream), "dsa row");  // host-to-device argument copies above are reused per row
        }
    }

private:
    std::shared_ptr<Scratch> s_;
};

class DefaultMoe final : public MoePath {
public:
    explicit DefaultMoe(std::shared_ptr<Scratch> s) : s_(std::move(s)) {}
    void forward(const MoeLayerWeights& w, const CallCtx& c, const float* x, const std::uint32_t* ids, const float* wts,
                 float* y) override {
        const std::size_t R = c.rows.size();
        std::vector<std::uint32_t> hid(R * kTopE);
        hip_ok(hipMemcpyAsync(hid.data(), ids, hid.size() * 4, hipMemcpyDeviceToHost, c.stream), "ids");
        hip_ok(hipStreamSynchronize(c.stream), "ids sync");
        const unsigned I = w.shared.down.raw.input;  // I_local
        for (std::size_t r = 0; r < R; ++r) {
            const float* xr = x + r * kHidden;
            for (unsigned k = 0; k < kTopE; ++k) mlp(w.experts.at(hid[r * kTopE + k]), xr, I, s_->eo + static_cast<std::size_t>(k) * kHidden, c);
            mlp(w.shared, xr, I, s_->sh, c);
            D::moe_combine(wts + r * kTopE, s_->eo, s_->sh, y + r * kHidden, 1, c.stream);
        }
    }
    void mlp(const Mlp& m, const float* x, unsigned I, float* out, const CallCtx& c) {
        if (m.gate.prepared == nullptr) throw std::logic_error("DefaultMoe needs prepared experts (loaded raw-only for the fast MoE seam)");
        c.lin->run(m.gate, x, s_->ge, 1, c.stream);
        c.lin->run(m.up, x, s_->ue, 1, c.stream);
        D::swiglu(s_->ge, s_->ue, s_->ae, I, c.stream);
        c.lin->run(m.down, s_->ae, out, 1, c.stream);
    }

private:
    std::shared_ptr<Scratch> s_;
};

class ReferencePrefill final : public PrefillPath {  // never called: Engine::prefill runs decode steps row by row
public:
    void kda(const KdaLayerWeights&, KdaState&, const CallCtx&, const float*, float*) override { throw std::logic_error("reference prefill = decode"); }
    void dsa(const DsaLayerWeights&, DsaState&, const CallCtx&, const float*, float*) override { throw std::logic_error("reference prefill = decode"); }
    void moe(const MoeLayerWeights&, const CallCtx&, const float*, const std::uint32_t*, const float*, float*) override {
        throw std::logic_error("reference prefill = decode");
    }
    void dense(const Mlp&, const CallCtx&, const float*, unsigned, float*) override { throw std::logic_error("reference prefill = decode"); }
};

// Placeholder for Options::external_exchange until the caller's set_exchange: any use is an error.
class MissingExchange final : public Exchange {
public:
    void reduce(float*, unsigned, hipStream_t) override { throw std::logic_error("external exchange not installed (set_exchange)"); }
    void argmax(float*, std::uint32_t*, unsigned, hipStream_t) override { throw std::logic_error("external exchange not installed (set_exchange)"); }
};

class LocalExchange final : public Exchange {
public:
    void reduce(float*, unsigned, hipStream_t) override {}
    void argmax(float*, std::uint32_t*, unsigned, hipStream_t) override {}
    void merge(MergeRow*, unsigned, MergeCtl&, hipStream_t) override {}  // one rank: already merged
};

// TP2 over the communicator: send this rank's partial, receive the peer's, y = p0 + p1 in rank order on both ranks.
// Communicator framing remains unchanged. The core owns the slot leases and
// their events; this backend supplies stream operations and the existing copies.
struct CommStagingRuntime {
    using Stream = hipStream_t;
    using Event = hipEvent_t;
    float*& storage;
    unsigned& capacity;
    unsigned depth;
    hipStream_t stream = nullptr;
    void destroy(Event event) { if (event) (void)hipEventDestroy(event); }
    void init(Event& source, Event& remote, Stream s) {
        if (!source) hip_ok(hipEventCreateWithFlags(&source, hipEventDisableTiming), "prefill source event");
        if (!remote) {
            hip_ok(hipEventCreateWithFlags(&remote, hipEventDisableTiming), "prefill remote event");
            hip_ok(hipEventRecord(remote, s), "prefill initial remote lifetime");
        }
        stream = s;
    }
    void wait_source(Event source, Stream s) {
        hip_ok(hipStreamWaitEvent(s, source, 0), "prefill source lifetime");
    }
    void reuse(Event source, Event, Stream s) {
        wait_source(source, s);
        // start_async enqueues remote_last_use on its H2D stream, as before.
    }
    void* remote(unsigned slot, unsigned offset) {
        // The first two views retain today's disjoint half regions and bytes.
        const std::size_t row = depth == 2 ? offset : std::size_t{slot} * capacity;
        return storage + row * kHidden;
    }
    void copy(void* local, void* target, std::size_t bytes, Event remote_event) {
        hip_ok(hipStreamWaitEvent(stream, remote_event, 0), "prefill staged slot lifetime");
        const std::size_t n4 = bytes / sizeof(float4);
        stage_copy_kernel<<<static_cast<unsigned>((n4 + 255) / 256), 256, 0, stream>>>(
            reinterpret_cast<const float4*>(local), static_cast<float4*>(target), n4);
        hip_ok(hipGetLastError(), "prefill stage copy");
    }
    void record_source(Event source, Stream s, bool staged) {
        hip_ok(hipEventRecord(source, s), staged ? "prefill staged source lifetime" : "prefill fallback source lifetime");
    }
    void record_remote(Event remote, Stream s) {
        hip_ok(hipEventRecord(remote, s), "prefill remote last use");
    }
};
using CommBackend = xport::CommunicatorBackend<dist::Communicator, CommStagingRuntime, dist::Tag>;
using CommStagingOps = xport::BackendStagingOps<CommBackend>;
struct CommStagingTraits : glm::xfer::GlmTraits { using Tag = dist::Tag; };
using CommStaging = xport::Channel<CommStagingTraits>::Staging<CommStagingOps>;

class CommExchange final : public Exchange {
public:
    CommExchange(dist::Communicator& comm, std::uint32_t rank, unsigned max_rows)
        : comm_(comm), rank_(rank), capacity_(max_rows), slots_(xport::staging_slots(CommStagingTraits::kStagingSlots)),
          remote_(dalloc<float>(static_cast<std::size_t>(max_rows) * kHidden * (slots_ == 2 ? 1 : slots_))),
          staging_runtime_{remote_, capacity_, slots_}, staging_backend_{comm_, staging_runtime_}, staging_ops_{staging_backend_}, staging_channel_(staging_ops_, slots_, xport::staging_trace_enabled()),
          am_l_(dalloc<std::uint64_t>(max_rows)), am_r_(dalloc<std::uint64_t>(max_rows)),
          merge_l_(dalloc<std::uint8_t>(kMergeBytes)), merge_r_(dalloc<std::uint8_t>(kMergeBytes)) {}
    ~CommExchange() override {
        if (xport::staging_trace_enabled()) {
            const auto& c = staging_channel_.counters();
            std::fprintf(stderr, "xport prefill bytes=%llu copied=%llu messages=%llu reuse_dependencies=%llu\n",
                static_cast<unsigned long long>(c.bytes), static_cast<unsigned long long>(c.staging_bytes),
                static_cast<unsigned long long>(c.exchanges), static_cast<unsigned long long>(c.reuse_dependencies));
        }
        (void)hipFree(remote_);
        (void)hipFree(am_l_);
        (void)hipFree(am_r_);
        (void)hipFree(merge_l_);
        (void)hipFree(merge_r_);
    }
    void resize_rows(unsigned rows) {
        if (rows == capacity_) return;
        hip_ok(hipDeviceSynchronize(), "exchange resize sync");
        (void)hipFree(remote_);
        (void)hipFree(am_l_);
        (void)hipFree(am_r_);
        remote_ = nullptr; am_l_ = am_r_ = nullptr;
        remote_ = dalloc<float>(static_cast<std::size_t>(rows) * kHidden * (slots_ == 2 ? 1 : slots_));
        am_l_ = dalloc<std::uint64_t>(rows);
        am_r_ = dalloc<std::uint64_t>(rows);
        capacity_ = rows;
    }
    void reduce(float* p, unsigned rows, hipStream_t s) override {
        dist::Tag tag;
        tag.kind = static_cast<std::uint32_t>(dist::TagKind::half_sum);
        tag.request = seq_++;
        const std::size_t n = static_cast<std::size_t>(rows) * kHidden;
        comm_.exchange(s, p, remote_, n * sizeof(float), tag);
        const float* p0 = rank_ == 0 ? p : remote_;
        const float* p1 = rank_ == 0 ? remote_ : p;
        sum2_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(p0, p1, p, n);
        hip_ok(hipGetLastError(), "sum2");
    }
    void start_reduce(float* p, unsigned rows, hipStream_t s) override {
        start_reduce_region(p, rows, 0, -1, s);
    }
    // STRIX_GLM_PF_STAGED: staged and chunked framings do not interoperate, so both ranks vote once (whatever their
    // environment) before the first pipelined layer and use the staging views only if both offer them.
    void agree_staged(hipStream_t s, bool async_start) {
        if (staged_vote_ >= 0) return;
        const std::size_t probe = static_cast<std::size_t>(capacity_ / 2U + 1U) * kHidden * sizeof(float);
        const std::uint64_t mine = async_start && pf_staged_enabled() && comm_.send_staging(std::min(probe, comm_.max_bytes())) != nullptr ? 1U : 0U;
        std::uint64_t peer = 0;
        dist::Tag tag;
        tag.kind = static_cast<std::uint32_t>(dist::TagKind::control);
        tag.request = seq_++;
        hip_ok(hipMemcpyAsync(am_l_, &mine, sizeof(mine), hipMemcpyHostToDevice, s), "staged vote");
        comm_.exchange(s, am_l_, am_r_, sizeof(mine), tag);
        hip_ok(hipMemcpyAsync(&peer, am_r_, sizeof(peer), hipMemcpyDeviceToHost, s), "staged vote peer");
        hip_ok(hipStreamSynchronize(s), "staged vote sync");
        staged_vote_ = mine == 1U && peer == 1U ? 1 : 0;
        std::fprintf(stderr, "rank %u prefill exchange: %s (local %u, peer %u)\n", rank_,
                     staged_vote_ == 1 ? "staging views" : "chunked copies", static_cast<unsigned>(mine), static_cast<unsigned>(peer));
    }
    // Producer buffers still have two halves; their latest ticket carries the
    // source lifetime independently of the number of channel staging slots.
    void wait_source(unsigned half, hipStream_t s) {
        if (source_pending_[half]) {
            staging_channel_.wait_source(source_fence_[half], s);
            source_pending_[half] = false;
        }
    }
    void start_reduce_region(float* p, unsigned rows, unsigned offset, int half, hipStream_t s) {
        if (pending_part_ != nullptr) throw std::logic_error("prefill reduction already in flight");
        if (held_) throw std::logic_error("prefill pair still held (release_pair before the next start)");
        dist::Tag tag;
        tag.kind = static_cast<std::uint32_t>(dist::TagKind::half_sum);
        tag.request = seq_++;
        const std::size_t bytes = static_cast<std::size_t>(rows) * kHidden * sizeof(float);
        const unsigned preferred = slots_ == 2 ? (half >= 0 ? static_cast<unsigned>(half) : 0U) : next_slot_++ % slots_;
        auto view = staging_channel_.staging(p, offset, s, preferred);
        pending_ticket_ = staging_channel_.post(tag, view, bytes, CommStagingOps::How{half >= 0 && staged_vote_ == 1, half >= 0});
        if (half >= 0) {
            source_fence_[half] = staging_channel_.source_fence(pending_ticket_);
            source_pending_[half] = true;
        }
        pending_part_ = p;
        pending_rows_ = rows;
    }
    void finish_reduce(hipStream_t s) override {
        float* p = pending_part_;
        const std::size_t n = static_cast<std::size_t>(pending_rows_) * kHidden;
        const float *p0 = nullptr, *p1 = nullptr;
        finish_reduce_pair(s, &p0, &p1);
        sum2_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(p0, p1, p, n);
        hip_ok(hipGetLastError(), "sum2 prefill pipeline");
        release_pair(s);
    }
    // SUMEXP retains the lease until the fused consumer has been enqueued.
    void finish_reduce_pair(hipStream_t s, const float** p0, const float** p1) {
        if (pending_part_ == nullptr) throw std::logic_error("prefill reduction not in flight");
        if (held_) throw std::logic_error("prefill pair already held");
        const float* p = pending_part_;
        pending_part_ = nullptr;
        const float* r;
        const auto view = staging_channel_.wait(pending_ticket_, s);
        r = static_cast<const float*>(view.remote);
        held_ = true;
        held_ticket_ = pending_ticket_;
        xport::rank_pair(rank_, p, r, p0, p1);
    }
    void release_pair(hipStream_t s) {
        if (!held_) throw std::logic_error("prefill pair not held");
        held_ = false;
        staging_channel_.release(held_ticket_, s);
    }
    void complete(hipStream_t s) override {
        if (held_) release_pair(s);
        if (pending_part_ != nullptr) finish_reduce(s);
    }
    void argmax(float* value, std::uint32_t* id, unsigned rows, hipStream_t s) override {
        std::vector<float> v(rows);
        std::vector<std::uint32_t> i(rows);
        std::vector<std::uint64_t> packed(rows), peer(rows);
        hip_ok(hipMemcpyAsync(v.data(), value, rows * 4, hipMemcpyDeviceToHost, s), "am v");
        hip_ok(hipMemcpyAsync(i.data(), id, rows * 4, hipMemcpyDeviceToHost, s), "am i");
        hip_ok(hipStreamSynchronize(s), "am sync");
        for (unsigned r = 0; r < rows; ++r) {
            std::uint32_t vb;
            std::memcpy(&vb, &v[r], 4);
            packed[r] = (static_cast<std::uint64_t>(i[r]) << 32U) | vb;
        }
        hip_ok(hipMemcpy(am_l_, packed.data(), rows * 8, hipMemcpyHostToDevice), "am up");
        dist::Tag tag;
        tag.kind = static_cast<std::uint32_t>(dist::TagKind::argmax);
        tag.request = seq_++;
        comm_.exchange(s, am_l_, am_r_, rows * 8, tag);
        hip_ok(hipMemcpyAsync(peer.data(), am_r_, rows * 8, hipMemcpyDeviceToHost, s), "am down");
        hip_ok(hipStreamSynchronize(s), "am sync2");
        for (unsigned r = 0; r < rows; ++r) {
            float pv;
            const auto pb = static_cast<std::uint32_t>(peer[r]);
            std::memcpy(&pv, &pb, 4);
            const auto pi = static_cast<std::uint32_t>(peer[r] >> 32U);
            if (pv > v[r] || (pv == v[r] && pi < i[r])) {  // R18: larger value, ties -> lower id
                v[r] = pv;
                i[r] = pi;
            }
        }
        hip_ok(hipMemcpy(value, v.data(), rows * 4, hipMemcpyHostToDevice), "am v up");
        hip_ok(hipMemcpy(id, i.data(), rows * 4, hipMemcpyHostToDevice), "am i up");
    }
    // design notes section 2: one exchange of [ctl][rows] per call; both ranks merge the same two halves in rank order.
    void merge(MergeRow* rows, unsigned n, MergeCtl& ctl, hipStream_t s) override {
        if (n > 8) throw std::invalid_argument("merge: at most eight rows");
        const std::size_t bytes = sizeof(MergeCtl) + static_cast<std::size_t>(n) * sizeof(MergeRow);
        std::vector<std::uint8_t> mine(bytes), peer(bytes);
        std::memcpy(mine.data(), &ctl, sizeof(MergeCtl));
        std::memcpy(mine.data() + sizeof(MergeCtl), rows, n * sizeof(MergeRow));
        hip_ok(hipMemcpy(merge_l_, mine.data(), bytes, hipMemcpyHostToDevice), "merge up");
        dist::Tag tag;
        tag.kind = static_cast<std::uint32_t>(dist::TagKind::argmax);
        tag.request = seq_++;
        comm_.exchange(s, merge_l_, merge_r_, bytes, tag);
        hip_ok(hipMemcpyAsync(peer.data(), merge_r_, bytes, hipMemcpyDeviceToHost, s), "merge down");
        hip_ok(hipStreamSynchronize(s), "merge sync");
        MergeCtl pc;
        std::memcpy(&pc, peer.data(), sizeof(MergeCtl));
        std::array<MergeRow, 8> peer_rows{};
        std::memcpy(peer_rows.data(), peer.data() + sizeof(MergeCtl), n * sizeof(MergeRow));
        spec::merge_rank_order({rows, n}, ctl, {peer_rows.data(), n}, pc, rank_);
    }

private:
    dist::Communicator& comm_;
    std::uint32_t rank_;
    unsigned capacity_;
    unsigned slots_;
    unsigned next_slot_ = 0;
    float* remote_;
    CommStagingRuntime staging_runtime_;
    CommBackend staging_backend_;
    CommStagingOps staging_ops_;
    CommStaging staging_channel_;
    CommStaging::Ticket pending_ticket_{}, held_ticket_{};
    CommStaging::SourceFence source_fence_[2]{};
    static_assert(sizeof(MergeRow) == 12);
    static_assert(sizeof(MergeCtl) == 72);
    static constexpr std::size_t kMergeBytes = sizeof(MergeCtl) + 8 * sizeof(MergeRow);
    std::uint64_t *am_l_, *am_r_;
    std::uint8_t *merge_l_, *merge_r_;
    std::uint64_t seq_ = 0;
    float* pending_part_ = nullptr;
    unsigned pending_rows_ = 0;
    bool source_pending_[2] = {};
    int staged_vote_ = -1;  // -1 not yet agreed, 0 chunked copies, 1 staging views (both ranks)
    bool held_ = false;  // the consumer holds a ticket until release_pair
};

// ---------------------------------------------------------------- layer weights
struct HcSite {
    float *fn = nullptr, *base = nullptr, *scale = nullptr;
};
struct LayerW {
    unsigned index = 0;
    bool kda = false, moe = false, mtp = false;
    HcSite hc_attn, hc_ffn;
    const std::uint16_t *in_norm = nullptr, *post_norm = nullptr;
    KdaLayerWeights kdaw;
    DsaLayerWeights dsaw;
    MoeLayerWeights moew;
    Mlp dense;
    KdaState kst;
    DsaState dst;
    // MTP only
    Linear eh_proj;
    const std::uint16_t *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
};

}  // namespace

void Exchange::merge(MergeRow*, unsigned, MergeCtl&, hipStream_t) {
    throw std::logic_error("this Exchange does not implement the speculative merge (design notes section 2)");
}

std::shared_ptr<PrefillPath> reference_prefill() { return std::make_shared<ReferencePrefill>(); }
std::shared_ptr<Exchange> local_exchange() { return std::make_shared<LocalExchange>(); }

namespace {
std::atomic<int> g_xb_publish{-1};
}  // namespace
void set_decode_producer_publish(int mode) { g_xb_publish.store(mode, std::memory_order_relaxed); }

struct Engine::Impl {
    // Declared first = destroyed last: the memory watchdog keeps running through the HIP teardown of a failed load.
    mem::Probe probe;
    std::unique_ptr<mem::Monitor> monitor;
    Options opt;
    dist::Communicator* comm;
    PackReader pack;
    Lib lib;
    std::shared_ptr<Scratch> scr = std::make_shared<Scratch>();
    std::shared_ptr<AttnPath> attn;
    std::shared_ptr<MoePath> moe;
    std::shared_ptr<Exchange> xch;          // active for the entire forward, including its head / merge
    std::shared_ptr<Exchange> fallback_xch; // installed seam (CommExchange unless explicitly overridden)
    std::shared_ptr<Exchange> decode_xch;   // borrowed transport; only automatic plain decode uses this
    std::shared_ptr<PrefillPath> pf;
    bool pf_seqpar = false;  // ordinary decode/verify never use it
    unsigned pf_streams = 2;  // single-stream experiment is explicit until device qualification
    bool pf_seqpar_auto_eligible = false;
    std::unique_ptr<xfer::SeqparExchange> seqpar_xch;
    struct InstallRefused : std::runtime_error { using std::runtime_error::runtime_error; };
    unsigned capacity_skip = 0, capacity_candidate = 0;
    bool loaded = false;        // load() finished
    bool pf_installed = false;  // pf came from Engine::set_prefill (not the default reference path)
    std::vector<LayerW> layers;
    std::uint16_t* embed = nullptr;
    const std::uint16_t* final_norm = nullptr;
    Linear head;
    std::uint32_t vocab0 = 0, vocab_n = 0;
    unsigned kda_heads = 0, mla_heads = 0, expert_i = 0, dense_i = 0;
    // activations
    float* hz = nullptr;                // [rows][24] mHC chains of a prefill chunk (pf::hc_pre_rows)
    float* hrr = nullptr;               // [rows] next-site norm factors of a prefill chunk (STRIX_GLM_PF_HC_FUSE)
    std::vector<unsigned char> no_mtp;  // per slot: chunk-prefilled, MTP cache not filled
    int null_layer = -1;  // set_null_flip (test hook)
    std::uint64_t null_pos = 0;
    unsigned null_elem = 0;
    float *X = nullptr, *X2 = nullptr, *F = nullptr, *mix = nullptr, *xc = nullptr, *xn = nullptr, *part = nullptr;
    float *logits = nullptr, *rlog = nullptr, *wts = nullptr, *hfin = nullptr, *hidden = nullptr, *amv = nullptr;
    float *dg = nullptr, *du = nullptr, *da = nullptr;
    std::uint32_t *ids = nullptr, *tok_d = nullptr, *ami = nullptr, *slot_d = nullptr;
    std::uint64_t* pos_d = nullptr;
    unsigned* flag = nullptr;
    hipStream_t stream = nullptr;
    Profile prof;
    bool profiling = false;
    // profiling: events at segment boundaries; seg_cat[i] = category of the segment ending at event i
    // Profile segment categories. NOT k-prefixed: an enumerator named kHc here shadowed strix::glm::kHc (= 4) inside Impl
    // as 1, so X / X2 / F were sized for 2 rows (R >= 3 memory fault (review finding)) and gate_residual copied stream 0 only.
    enum Cat : int { segStart = -1, segEmbed, segHc, segKda, segDsa, segXch, segMoe, segDense, segHead };
    static_assert(kHc == 4 && kHidden == 4096, "a member name shadows glm::kHc / kHidden inside Impl (R >= 3 fault 7a5a9381)");
    std::vector<hipEvent_t> seg_ev;
    std::vector<int> seg_cat;
    std::size_t seg_n = 0;
    void mark(int cat) {
        if (!profiling) return;
        if (seg_n == seg_ev.size()) {
            hipEvent_t e{};
            hip_ok(hipEventCreate(&e), "profile event");
            seg_ev.push_back(e);
            seg_cat.push_back(cat);
        }
        seg_cat[seg_n] = cat;
        hip_ok(hipEventRecord(seg_ev[seg_n++], stream), "profile record");
    }
    void collect() {  // after the call's stream sync
        if (!profiling) return;
        hip_ok(hipEventSynchronize(seg_ev[seg_n - 1]), "profile sync");
        for (std::size_t i = 1; i < seg_n; ++i) {
            float ms = 0;
            hip_ok(hipEventElapsedTime(&ms, seg_ev[i - 1], seg_ev[i]), "profile elapsed");
            double* f[] = {&prof.embed, &prof.hc, &prof.kda, &prof.dsa, &prof.exchange, &prof.moe, &prof.dense, &prof.head};
            if (seg_cat[i] >= 0) *f[seg_cat[i]] += ms;
        }
        prof.attn = prof.kda + prof.dsa;
        prof.ffn = prof.moe + prof.dense;
        seg_n = 0;
    }
    bool smoke_head = false;
    // STRIX_GLM_STEP_FLAG=0: a plain step syncs for the layers' checkpoints before the head (old path).
    const bool defer_layer_flag = !(std::getenv("STRIX_GLM_STEP_FLAG") != nullptr && std::string(std::getenv("STRIX_GLM_STEP_FLAG")) == "0");
    const bool boundary_fused = !(std::getenv("STRIX_GLM_BOUNDARY") != nullptr && std::string(std::getenv("STRIX_GLM_BOUNDARY")) == "0");
    // STRIX_GLM_XB_PUBLISH (default on, =0 disables): the decode producers of an exchanged partial (attention o
    // projection, MoE down) write it into the exchange TX slot and publish it themselves: no copy + publish launch, and
    // the exchange starts at the end of the producer. Same bits. Needs the fused boundary (STRIX_GLM_BOUNDARY).
    const bool xb_publish_env = !(std::getenv("STRIX_GLM_XB_PUBLISH") != nullptr && std::string(std::getenv("STRIX_GLM_XB_PUBLISH")) == "0");
    bool xb_publish_on() const {
        const int forced = g_xb_publish.load(std::memory_order_relaxed);
        return forced < 0 ? xb_publish_env : forced != 0;
    }
    std::uint64_t xb_publish_count = 0;
    // MoE kernels (TP2 rank slices, I = 1024): default for world 2; STRIX_GLM_MOE=ref = DefaultMoe.
    const bool moe_fast = opt.world == 2 && !(std::getenv("STRIX_GLM_MOE") != nullptr && std::string(std::getenv("STRIX_GLM_MOE")) == "ref");
    // STRIX_GLM_MODEL_FORMAT=exl3 (default) | fast_i4. fast_i4 (lossy FAST identity glm-fast-i4): at load the routed
    // experts of the INT4 layers are decoded from EXL3 and converted on the GPU to the IU4 g64 pack (or read from a
    // matching STRIX_GLM_FAST_I4_PACK), and their EXL3 trellis is freed; prefill runs them W4A4 on the grouped IU4 core,
    // decode / MTP verify W4A16. Shared experts, attention, dense layers and the head stay EXL3. Exact identities are refused.
    const bool model_fast_i4 = [] {
        const char* v = std::getenv("STRIX_GLM_MODEL_FORMAT");
        if (v == nullptr || *v == '\0' || std::string(v) == "exl3") return false;
        if (std::string(v) == "fast_i4") return true;
        throw std::invalid_argument("STRIX_GLM_MODEL_FORMAT must be exl3 or fast_i4");
    }();
    // Layer / scope policy (src/moe_core/int4_layer_policy.hpp; the engine only names the options and the defaults).
    // STRIX_GLM_FAST_I4_EXL3_LAYERS = none | comma list of trunk MoE ordinals / ranges a-b (0 = the first MoE layer) that
    // keep the EXL3 routed experts in prefill and decode. Default 0-14: an early INT4 layer costs a saturating KL step
    // (about 0.007 alone), late layers about 0.0003 each; INT4 on ordinals >= 15 passes the quality gate, all layers do not.
    // STRIX_GLM_FAST_I4_SCOPE = gate_up (default: the routed down stays EXL3, the largest error source) | all.
    // STRIX_GLM_FAST_I4_SCALE = fast_full (default: core/iu4.hpp fast full-range rule) | legacy (max|w| / 7).
    const strix::moe_core::Int4LayerPolicy i4_policy = [] {
        strix::moe_core::Int4LayerPolicy policy;
        const char* layers_env = std::getenv("STRIX_GLM_FAST_I4_EXL3_LAYERS");
        policy.source_mask = strix::moe_core::parse_int4_source_layers(layers_env == nullptr || *layers_env == '\0' ? "0-14" : layers_env, "STRIX_GLM_FAST_I4_EXL3_LAYERS");
        const char* scope_env = std::getenv("STRIX_GLM_FAST_I4_SCOPE");
        policy.matrices = strix::moe_core::parse_int4_scope(scope_env == nullptr || *scope_env == '\0' ? "gate_up" : scope_env, "STRIX_GLM_FAST_I4_SCOPE");
        return policy;
    }();
    const unsigned i4_matrices = i4_policy.matrices;
    const int i4_scale_rule = [] {
        const char* v = std::getenv("STRIX_GLM_FAST_I4_SCALE");
        if (v == nullptr || *v == '\0' || std::string(v) == "fast_full") return strix::core::iu4_scale_rule_fast_full;
        if (std::string(v) == "legacy") return strix::core::iu4_scale_rule_legacy;
        throw std::invalid_argument("STRIX_GLM_FAST_I4_SCALE must be fast_full or legacy");
    }();
    // INT4 layer: a trunk MoE layer whose ordinal the policy does not keep in EXL3 (the MTP layer stays EXL3).
    bool i4_layer_at(unsigned layer) const {
        if (!model_fast_i4 || layer == kMtpLayer || !is_moe_layer(layer)) return false;
        unsigned ordinal = 0;
        for (unsigned l = 0; l < layer; ++l) ordinal += l != kMtpLayer && is_moe_layer(l) ? 1U : 0U;
        return i4_policy.int4(ordinal);
    }
    fast_i4::Pack i4pack;
    double i4_convert_s = 0;
    std::uint64_t i4_bytes = 0;
    unsigned i4_layers = 0, i4_converted = 0;
    // attention fused Tier B (src/glm/attn/glm_attn_tierb, bits of strix_gemv): the FastAttn seam reads raw attention EXL3
    // linears (file layout, no strix_weights). STRIX_GLM_ATTN=ref or STRIX_GLM_TIERB=lib load them prepared and keep the
    // library calls (same bits). The MTP layer keeps prepared linears (its drafter callbacks use the library).
    // attention fused groups are built for the TP2 rank shapes (32 KDA heads / rank): world 1 keeps strix_gemv (same bits).
    const bool attn_tierb = opt.world == 2 && !(std::getenv("STRIX_GLM_ATTN") != nullptr && std::string(std::getenv("STRIX_GLM_ATTN")) == "ref") &&
                            !(std::getenv("STRIX_GLM_TIERB") != nullptr && std::string(std::getenv("STRIX_GLM_TIERB")) == "lib");
    const bool post_ref = std::getenv("STRIX_GLM_POST") != nullptr && std::string(std::getenv("STRIX_GLM_POST")) == "ref";
    // hc_pre_v3 variant: default 7 (qualified revision a7c9d6db, bitwise rows 1-8); STRIX_GLM_HC_PRE_V=4 keeps the previous one.
    const int hc_pre_variant = std::getenv("STRIX_GLM_HC_PRE_V") != nullptr ? std::atoi(std::getenv("STRIX_GLM_HC_PRE_V")) : 7;
    IdentityMode identity = opt.identity;
    // Frozen when load starts for identical load-plan and actual-load allocations.
    IdentityMode load_identity = opt.identity;
    bool r1b = false;
    // route log (identity comparisons): ids [kLayers][8] of row 0 of the last forward, device; off by default.
    bool route_log = false;
    unsigned cur_layer = 0;
    std::unique_ptr<KvShard> kv_shard;
    bool kv_shard_auto = false;
    std::uint32_t* route_log_d = nullptr;
    bool load_test = false;
    // speculative verify (Options::spec): per-row / replay flags, per-slot pending rows + current replay set
    unsigned *row_flag = nullptr, *replay_flag = nullptr;
    std::vector<unsigned> pending, cur;
    // MTP drafter buffers (Options::mtp): own flag (drafter non-finite never touches the trunk flag), h' rows
    unsigned* mtp_flag = nullptr;
    float* trunk_logits = nullptr;
    std::vector<Row> mtp_rows;
    std::vector<std::uint32_t> mtp_slots;
    std::vector<std::uint64_t> mtp_positions;
    float *m_e = nullptr, *m_en = nullptr, *m_hn = nullptr, *m_cat = nullptr, *m_z = nullptr, *m_r = nullptr, *m_y = nullptr, *m_h = nullptr, *m_hs = nullptr,
          *m_mz = nullptr;
    // load memory accounting (design notes): plan (dry run) -> calibration -> admission -> checked load
    mem::LoadPlan plan;
    mem::AllocModel amodel;
    mem::Margins margins;
    mem::Snapshot base;

    Impl(Options o, dist::Communicator* c) : opt(std::move(o)), comm(c), pack(opt.model_dir), lib(pack, opt.raw_linears) {
        const char* shard_env = std::getenv("STRIX_GLM_KV_SHARD");
        const std::string shard_policy = shard_env == nullptr ? "0" : shard_env;
        if (shard_policy != "0" && shard_policy != "1" && shard_policy != "auto")
            throw std::invalid_argument("STRIX_GLM_KV_SHARD must be 0, 1 or auto");
        kv_shard_auto = shard_policy == "auto";
        const bool shard_requested = shard_policy == "1";
        if (shard_requested || kv_shard_auto) {
            if (opt.world != 2 || opt.rank > 1 || opt.slots != 1 || opt.spec || opt.mtp ||
                opt.external_exchange || c == nullptr || opt.pf_pipeline != 1)
                throw std::invalid_argument("STRIX_GLM_KV_SHARD: draft TP2 plain decode only; requires communicator, one slot, MTP/spec OFF, pipeline 1");
            (void)kvshard::storage_tokens(opt.capacity);
            if (shard_requested) kv_shard = std::make_unique<KvShard>(*c, opt.rank);
        }
        opt.pf_pipeline = xfer::prefill_pipeline(opt.pf_pipeline, std::getenv("STRIX_GLM_PF_PIPELINE"));
        if ((kv_shard || kv_shard_auto) && opt.pf_pipeline != 1) throw std::invalid_argument("kvshard: prefill pipeline 1 required");
        // STRIX_GLM_PF_SEQPAR: auto (default) | 1 | 0. auto = on for ordinary TP2 replicated attention ownership,
        // off (declined, legacy path) for TP1, external exchange or any KV-shard policy; 1 refuses those cases.
        const char* seqpar_env = std::getenv("STRIX_GLM_PF_SEQPAR");
        const std::string seqpar_policy = seqpar_env == nullptr ? "auto" : seqpar_env;
        if (seqpar_policy != "0" && seqpar_policy != "1" && seqpar_policy != "auto")
            throw std::invalid_argument("STRIX_GLM_PF_SEQPAR must be 0, 1 or auto");
        const bool seqpar_eligible = opt.world == 2 && opt.rank <= 1 && c != nullptr && !opt.external_exchange &&
                                     !kv_shard && !kv_shard_auto;
        if (seqpar_policy == "1" && !seqpar_eligible)
            throw std::invalid_argument("glm seqpar: requires TP2 communicator and ordinary attention cache ownership");
        pf_seqpar = seqpar_policy == "1" || (seqpar_policy == "auto" && seqpar_eligible);
        pf_seqpar_auto_eligible = seqpar_policy == "auto" && seqpar_eligible;
        if (const char* value = std::getenv("STRIX_GLM_PF_STREAMS")) pf_streams = xfer::seqpar_compute_streams(value);
        if (pf_streams == 1 && !pf_seqpar)
            throw std::invalid_argument("STRIX_GLM_PF_STREAMS=1 requires seqpar TP2 prefill");
        if (seqpar_policy == "auto" && !seqpar_eligible && opt.world == 2)
            std::fprintf(stderr, "glm prefill seqpar auto declined: %s; legacy path\n",
                         kv_shard || kv_shard_auto ? "kv-shard policy" : opt.external_exchange ? "external exchange" : "no TP2 communicator");
        if (opt.world != 1 && opt.world != 2) throw std::invalid_argument("world must be 1 or 2");
        // STRIX_GLM_LOAD_TEST=1: one rank's slice loaded without a peer (load timing only; any forward throws).
        load_test = opt.world == 2 && comm == nullptr && std::getenv("STRIX_GLM_LOAD_TEST") != nullptr &&
                    std::string(std::getenv("STRIX_GLM_LOAD_TEST")) == "1";
        if (opt.world == 2 && comm == nullptr && !load_test && !opt.external_exchange)
            throw std::invalid_argument("world 2 needs a communicator (or Options::external_exchange + set_exchange)");
        if (opt.capacity % kKpool != 0) throw std::invalid_argument("capacity must be a multiple of 4");
        if (const char* env = std::getenv("STRIX_GLM_IDENTITY")) {
            const std::string mode(env);
            if (mode == "r1") identity = IdentityMode::kR1;
            else if (mode == "r1b") identity = IdentityMode::kR1b;
            else if (mode == "r1b-all") identity = IdentityMode::kR1bAll;
            else throw std::invalid_argument("STRIX_GLM_IDENTITY must be r1|r1b|r1b-all");
        }
        (void)identity_mode_name(identity);  // Reject invalid Options enum values as well.
        r1b = identity != IdentityMode::kR1;
        strix::glm::r1b::set_active(r1b);  // KDA seams follow generated forwards; prefill scopes R1.
        kda_heads = kKdaHeads / opt.world;
        mla_heads = kMlaHeads / opt.world;
        expert_i = kExpertI / opt.world;
        dense_i = kDenseI / opt.world;
        vocab_n = kVocab / opt.world;
        vocab0 = opt.rank * vocab_n;
        const char* attn_env = std::getenv("STRIX_GLM_ATTN");
        if ((kv_shard || kv_shard_auto) && attn_env != nullptr && std::string(attn_env) == "ref")
            throw std::invalid_argument("kvshard: fast attention seam required");
        if (attn_env != nullptr && std::string(attn_env) == "ref") attn = std::make_shared<DefaultAttn>(scr);
        else attn = fast_attention(kda_heads, mla_heads, opt.capacity);
        moe = moe_fast ? fast_moe() : std::make_shared<DefaultMoe>(scr);
        if (model_fast_i4) {
            const char* why = nullptr;
            if (!moe_fast) why = "needs --world 2 and the MoE kernels (STRIX_GLM_MOE=ref refused)";
            else if (!glm::pf::exact_moe::iu4::built() || !moe::fast_i4_built()) why = "this build has no IU4 / W4A16 core (shared stored INT4 core)";
            else if (std::getenv("STRIX_GLM_PF_IU4") != nullptr && std::string(std::getenv("STRIX_GLM_PF_IU4")) == "1")
                why = "replaces STRIX_GLM_PF_IU4=1 (the duplicate-copy opt-in): unset it";
            else if (kv_shard || kv_shard_auto) why = "is not qualified with STRIX_GLM_KV_SHARD";
            if (why) throw std::invalid_argument(std::string("STRIX_GLM_MODEL_FORMAT=fast_i4 ") + why);
            std::printf("glm model format: fast_i4 (routed INT4 g64 stored, scope %s, EXL3 MoE ordinals mask 0x%llx, scale %s; "
                        "W4A4 prefill / W4A16 decode; lossy)\n", i4_matrices == 3U ? "all" : "gate_up",
                        static_cast<unsigned long long>(i4_policy.source_mask),
                        i4_scale_rule == strix::core::iu4_scale_rule_legacy ? "legacy" : "fast_full");
        }
        xch = opt.world == 1 || load_test ? local_exchange()
              : opt.external_exchange   ? std::shared_ptr<Exchange>(std::make_shared<MissingExchange>())
                                        : std::shared_ptr<Exchange>(std::make_shared<CommExchange>(*comm, opt.rank, std::max(8U, opt.exchange_rows ? opt.exchange_rows : opt.max_rows)));  // decode / verify calls carry up to 8 rows
        fallback_xch = xch;
        if (const char* v = std::getenv("STRIX_GLM_DECODE_XCH"); v != nullptr) {
            const std::string kind(v);
            if (kind != "comm" && kind != "xfer") throw std::invalid_argument("STRIX_GLM_DECODE_XCH must be comm or xfer");
            opt.decode_xfer = kind == "xfer";
        }
        // Attention can borrow an idle Channel while TP uses the communicator.
        if (opt.world == 2 && comm != nullptr && !opt.external_exchange && opt.decode_xfer)
            decode_xch = std::make_shared<xfer::XferExchange>(comm->transport(), opt.rank, 8U);
        if (kv_shard) bind_shard_exchange();
        pf = reference_prefill();
    }

    void bind_shard_exchange() {
        auto shared = opt.decode_xfer ? std::dynamic_pointer_cast<xfer::XferExchange>(decode_xch)
                                     : std::make_shared<xfer::XferExchange>(comm->transport(), opt.rank, 8U);
        if (!shared) throw std::logic_error("kvshard: attention Channel required");
        shared->reserve_attention_bytes(kvshard::decode_capacity(opt.capacity));
        kv_shard->bind_exchange(std::move(shared), opt.decode_xfer);
    }

    // Never share transport progress between implementations. complete() joins Xfer jobs and seals any deferred
    // CRC; the stream sync also retires device consumers of either implementation's receive buffers / flags.
    void switch_exchange(const std::shared_ptr<Exchange>& next) {
        if (xch == next) return;
        xch->complete(stream);
        hip_ok(hipStreamSynchronize(stream), "exchange switch sync");
        xch = next;
    }

    enum class ForwardKind { Plain, Verify, Mtp, Gate, Prefill };
    void select_exchange(ForwardKind kind, unsigned rows) {
        const char* env = std::getenv("STRIX_GLM_SPEC_XCH");
        const std::string mode = env == nullptr || *env == '\0' ? "xfer" : env;
        if (mode != "xfer" && mode != "comm") throw std::invalid_argument("STRIX_GLM_SPEC_XCH must be xfer or comm");
        const bool fast = kind == ForwardKind::Plain ||
                          ((kind == ForwardKind::Verify || kind == ForwardKind::Mtp) && mode == "xfer");
        switch_exchange(decode_xch && fast && rows > 0 && rows <= 8 ? decode_xch : fallback_xch);
    }

    std::string L(unsigned l) const { return kP + "layers." + std::to_string(l) + "."; }

    void load_hc(LayerW& w, unsigned l) {
        for (int s = 0; s < 2; ++s) {
            HcSite& h = s == 0 ? w.hc_attn : w.hc_ffn;
            const std::string n = L(l) + (s == 0 ? "hc_attn_" : "hc_ffn_");
            if (lib.dtype(n + "fn") != "F32") throw std::runtime_error("hc fn must be F32 (identity): " + n + "fn");
            h.fn = static_cast<float*>(lib.upload_rows(n + "fn"));
            h.base = static_cast<float*>(lib.upload_rows(n + "base"));
            h.scale = static_cast<float*>(lib.upload_rows(n + "scale"));
        }
    }
    void load_dsa(LayerW& w, const std::string& a) {
        const std::uint64_t r = opt.rank, H = mla_heads;
        auto& d = w.dsaw;
        d.heads = mla_heads;
        const bool raw = attn_tierb && !w.mtp;
        d.q_a = raw ? lib.whole_raw(a + "q_a_proj") : lib.whole(a + "q_a_proj");
        d.kv_a = raw ? lib.whole_raw(a + "kv_a_proj_with_mqa") : lib.whole(a + "kv_a_proj_with_mqa");
        d.wq_b = raw ? lib.whole_raw(a + "indexer.wq_b") : lib.whole(a + "indexer.wq_b");
        d.q_b = raw ? lib.out_slice_raw(a + "q_b_proj", r * H * kMlaDim, H * kMlaDim) : lib.out_slice(a + "q_b_proj", r * H * kMlaDim, H * kMlaDim);
        d.o = raw ? lib.in_slice_raw(a + "o_proj", r * H * kMlaDim, H * kMlaDim) : lib.in_slice(a + "o_proj", r * H * kMlaDim, H * kMlaDim);
        d.q_a_norm = static_cast<const std::uint16_t*>(lib.upload_rows(a + "q_a_layernorm.weight"));
        d.kv_a_norm = static_cast<const std::uint16_t*>(lib.upload_rows(a + "kv_a_layernorm.weight"));
        d.kv_b = static_cast<const std::uint16_t*>(lib.upload_rows(a + "kv_b_proj.weight", r * H * 2 * kMlaDim, H * 2 * kMlaDim));
        d.wk = static_cast<const std::uint16_t*>(lib.upload_rows(a + "indexer.wk.weight"));
        d.gate = static_cast<const std::uint16_t*>(lib.upload_rows(a + "indexer.index_kpool_compress_gate"));
        d.weights_proj = static_cast<const std::uint16_t*>(lib.upload_rows(a + "indexer.weights_proj.weight"));
        d.k_norm_w = static_cast<const std::uint16_t*>(lib.upload_rows(a + "indexer.k_norm.weight"));
        d.k_norm_b = static_cast<const std::uint16_t*>(lib.upload_rows(a + "indexer.k_norm.bias"));
        d.ape = static_cast<const float*>(lib.upload_rows(a + "indexer.index_kpool_compress_ape"));
        for (const char* n : {"indexer.wk.weight", "indexer.index_kpool_compress_gate", "indexer.weights_proj.weight", "indexer.k_norm.weight",
                              "indexer.k_norm.bias", "kv_b_proj.weight"})
            if (lib.dtype(a + n) != "F16") throw std::runtime_error(std::string("expected F16: ") + a + n);
        if (lib.dtype(a + "indexer.index_kpool_compress_ape") != "F32") throw std::runtime_error("ape must be F32");
        const std::size_t S = opt.slots, cap = static_cast<std::size_t>(kv_shard ? kvshard::storage_tokens(opt.capacity) : opt.capacity);
        w.dst.cap = opt.capacity;
        w.dst.shard = kv_shard.get();
        w.dst.latent = lib.state<std::uint16_t>(S * cap * kKvLora);
        w.dst.ikey = lib.state<std::uint16_t>(S * cap * kIdxDim);
        w.dst.igate = lib.state<std::uint16_t>(S * cap * kIdxDim);
        w.dst.pooled = lib.state<float>(S * (cap / kKpool) * kIdxDim);
    }
    const std::uint16_t* pack_r1b(const std::uint16_t* w, unsigned n) {
        const int variant = strix::glm::r1b::dense_variant();
        if (load_identity == IdentityMode::kR1 || (variant != 5 && variant != 6 && variant != 7)) return nullptr;
        // Identical load-time condition and lib.buffer allocation in both planning/loading passes.
        auto* packed = lib.buffer<std::uint16_t>(static_cast<std::size_t>(n) * kHidden);
        if (!lib.planning()) {
            strix::glm::r1b::pack_k4096(w, n, packed, stream);
            hip_ok(hipStreamSynchronize(stream), "r1b load pack");
        }
        return packed;
    }
    void load_moe(LayerW& w, const std::string& m) {
        const std::uint64_t r = opt.rank, I = expert_i;
        w.moew.router = static_cast<const std::uint16_t*>(lib.upload_rows(m + "gate.weight"));
        if (lib.dtype(m + "gate.weight") != "F16") throw std::runtime_error("router must be F16: " + m);
        w.moew.router_packed = pack_r1b(w.moew.router, kExperts);
        w.moew.bias = static_cast<const float*>(lib.upload_rows(m + "gate.e_score_correction_bias"));
        auto mlp = [&](const std::string& p) {
            if (moe_fast)  // MoE kernels read the raw file layout: no strix_weights per expert (no 2 MiB granules)
                return Mlp{lib.out_slice_raw(p + "gate_proj", r * I, I), lib.out_slice_raw(p + "up_proj", r * I, I),
                           lib.in_slice_raw(p + "down_proj", r * I, I)};
            return Mlp{lib.out_slice(p + "gate_proj", r * I, I), lib.out_slice(p + "up_proj", r * I, I), lib.in_slice(p + "down_proj", r * I, I)};
        };
        w.moew.experts.reserve(kExperts);
        const bool i4_layer = i4_layer_at(w.index);
        const bool i4_hit = i4_layer && !lib.planning() && i4pack.has(w.index);
        if (i4_layer)
            lib.i4_routed_begin(std::uint64_t{kExperts} * i4_matrices * (kHidden / 16U) * (I / 16U) * 4U * 32U, i4_hit);
        auto routed = [&](const std::string& p) {  // scope gate_up: the routed down stays a resident EXL3 raw slice
            if (!i4_layer || i4_matrices == 3U) return mlp(p);
            Mlp x;
            x.gate = lib.out_slice_raw(p + "gate_proj", r * I, I);
            x.up = lib.out_slice_raw(p + "up_proj", r * I, I);
            lib.i4_suspend(true);
            x.down = lib.in_slice_raw(p + "down_proj", r * I, I);
            lib.i4_suspend(false);
            return x;
        };
        for (unsigned e = 0; e < kExperts; ++e) w.moew.experts.push_back(routed(m + "experts." + std::to_string(e) + "."));
        if (i4_layer) lib.i4_routed_end();
        w.moew.shared = mlp(m + "shared_experts.");
        if (i4_layer) install_i4(w, i4_hit);
        auto* ed = static_cast<Mlp*>(lib.arena(sizeof(Mlp) * kExperts));
        if (!lib.planning()) hip_ok(hipMemcpy(ed, w.moew.experts.data(), sizeof(Mlp) * kExperts, hipMemcpyHostToDevice), "experts_d");
        w.moew.experts_d = ed;
    }
    // fast_i4: one dedicated allocation per MoE layer: per matrix the nibbles, the FP16 scales and the [288] svh table.
    void install_i4(LayerW& w, bool hit) {
        namespace X = glm::pf::exact_moe::iu4;
        const std::uint64_t wb = X::weight_bytes(), sb = X::scale_bytes(0), tb = (kExperts * sizeof(void*) + 255) & ~std::size_t{255};
        const unsigned nm = i4_matrices;
        auto* blk = static_cast<std::uint8_t*>(lib.dedicated(nm * (wb + sb + tb), "fast_i4 layer pack"));
        if (lib.planning()) return;
        auto& q = w.moew.i4;
        q.base = blk;
        q.bytes = nm * (wb + sb);
        X::Layer v;
        std::uint8_t* p = blk;
        for (unsigned m = 0; m < nm; ++m) {
            q.wq[m] = p;
            q.ws[m] = reinterpret_cast<std::uint16_t*>(p + wb);
            q.svh[m] = reinterpret_cast<const std::uint16_t* const*>(p + wb + sb);
            p += wb + sb + tb;
            v.wq[m] = q.wq[m];
            v.ws[m] = q.ws[m];
            v.svh[m] = q.svh[m];
        }
        auto ex = [](const Linear& l) { return moe::Exl3{l.raw.trellis, l.raw.suh, l.raw.svh, l.raw.input, l.raw.output, l.raw.bits}; };
        std::vector<moe::ExpertSlices> host(kExperts);
        for (unsigned e = 0; e < kExperts; ++e) {
            const Mlp& x = w.moew.experts[e];
            host[e] = {ex(x.gate), ex(x.up), ex(x.down)};
        }
        if (hit) {
            i4_pack_read(q, w.index, w.moew.experts);
            std::vector<const std::uint16_t*> cols(kExperts);
            for (unsigned m = 0; m < nm; ++m) {
                for (unsigned e = 0; e < kExperts; ++e) cols[e] = (&host[e].gate)[m].svh;
                hip_ok(hipMemcpy(const_cast<const std::uint16_t**>(q.svh[m]), cols.data(), kExperts * sizeof(void*), hipMemcpyHostToDevice),
                       "fast_i4 svh table");
            }
        } else {
            double s = 0;
            auto conv = [&] { s = X::convert_into(host.data(), v, stream, i4_scale_rule); };
            if (i4_converted == 0) i4_receipt(w, host, conv);  // first converted layer: the MoE block, EXL3 vs fast_i4
            else conv();
            i4_convert_s += s;
            ++i4_converted;
            if (!i4pack.dir.empty()) i4_pack_write(q, w.index, w.moew.experts);
        }
        for (Mlp& x : w.moew.experts) {
            x.gate.raw.trellis = x.up.raw.trellis = nullptr;
            if (nm == 3U) x.down.raw.trellis = nullptr;
        }
        ++i4_layers;
        i4_bytes += q.bytes;
        if (i4_layers % 8 == 1)
            std::printf("glm fast_i4: rank %u layer %u %s (%u layers, %.2f GiB INT4, convert %.2f s, pack read %.2f s)\n", opt.rank, w.index,
                        hit ? "read from pack" : "converted", i4_layers, static_cast<double>(i4_bytes) / double(1ULL << 30), i4_convert_s,
                        i4pack.read_seconds);
    }
    // Layer receipt (xphalf-anomaly rule): the first converted MoE layer's decode MoE block (routed + shared + R16) on
    // R = 4 deterministic rows (unit-RMS pseudo-random hidden states, 8 distinct experts per row, weights 1/8 * 2.5)
    // with the EXL3 routed weights (still in the layer buffer) and with the fast_i4 pack: rel-L2 of the outputs. A
    // conversion / layout bug shows as rel-L2 ~1; the expected INT4 error is a few 1e-2.
    template <class Convert> void i4_receipt(const LayerW& w, const std::vector<moe::ExpertSlices>& host, Convert&& convert) {
        constexpr unsigned R = 4;
        std::vector<float> x(std::size_t{R} * kHidden), rwt(std::size_t{R} * kTopE, 2.5F / kTopE), y0(x.size()), y1(x.size());
        std::vector<std::uint32_t> rid(std::size_t{R} * kTopE);
        std::uint64_t st = 0x9e3779b97f4a7c15ULL;
        for (auto& v : x) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            v = (static_cast<float>(static_cast<std::uint32_t>(st >> 40)) / 16777216.0F - 0.5F) * 3.4641F;  // uniform, unit RMS
        }
        for (unsigned r = 0; r < R; ++r)
            for (unsigned k = 0; k < kTopE; ++k) rid[r * kTopE + k] = (r * 7U + k * 36U) % kExperts;  // distinct, ascending per row
        float *dx = nullptr, *dw = nullptr, *dy = nullptr;
        std::uint32_t* di = nullptr;
        moe::ExpertSlices* dt = nullptr;
        void* ws = nullptr;
        hip_ok(hipMalloc(&dx, x.size() * 4), "receipt");
        hip_ok(hipMalloc(&dw, rwt.size() * 4), "receipt");
        hip_ok(hipMalloc(&dy, x.size() * 4), "receipt");
        hip_ok(hipMalloc(&di, rid.size() * 4), "receipt");
        hip_ok(hipMalloc(&dt, host.size() * sizeof(moe::ExpertSlices)), "receipt");
        hip_ok(hipMalloc(&ws, moe::moe_workspace_bytes()), "receipt");
        hip_ok(hipMemset(ws, 0, moe::moe_workspace_bytes()), "receipt");
        hip_ok(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice), "receipt");
        hip_ok(hipMemcpy(dw, rwt.data(), rwt.size() * 4, hipMemcpyHostToDevice), "receipt");
        hip_ok(hipMemcpy(di, rid.data(), rid.size() * 4, hipMemcpyHostToDevice), "receipt");
        hip_ok(hipMemcpy(dt, host.data(), host.size() * sizeof(moe::ExpertSlices), hipMemcpyHostToDevice), "receipt");
        auto ex = [](const Linear& l) { return moe::Exl3{l.raw.trellis, l.raw.suh, l.raw.svh, l.raw.input, l.raw.output, l.raw.bits}; };
        moe::MoeView v;
        v.rows = R;
        v.y = dx;
        v.ids = di;
        v.weights = dw;
        v.experts = dt;
        v.shared = {ex(w.moew.shared.gate), ex(w.moew.shared.up), ex(w.moew.shared.down)};
        v.out = dy;
        v.ws.base = ws;
        moe::enqueue_experts(v, stream);  // EXL3 reference before load-time conversion
        hip_ok(hipStreamSynchronize(stream), "receipt");  // the engine stream is non-blocking: no implicit order with hipMemcpy
        hip_ok(hipMemcpy(y0.data(), dy, y0.size() * 4, hipMemcpyDeviceToHost), "receipt");
        convert();
        for (unsigned m = 0; m < i4_matrices; ++m) {
            v.i4.wq[m] = w.moew.i4.wq[m];
            v.i4.ws[m] = w.moew.i4.ws[m];
        }
        moe::enqueue_experts(v, stream);
        hip_ok(hipStreamSynchronize(stream), "receipt");
        hip_ok(hipMemcpy(y1.data(), dy, y1.size() * 4, hipMemcpyDeviceToHost), "receipt");
        double num = 0, den = 0;
        for (std::size_t i = 0; i < y0.size(); ++i) {
            num += (static_cast<double>(y1[i]) - y0[i]) * (static_cast<double>(y1[i]) - y0[i]);
            den += static_cast<double>(y0[i]) * y0[i];
        }
        for (void* p : {static_cast<void*>(dx), static_cast<void*>(dw), static_cast<void*>(dy), static_cast<void*>(di), static_cast<void*>(dt), ws})
            (void)hipFree(p);
        std::printf("glm fast_i4 receipt: rank %u layer %u decode MoE block (R=%u synthetic rows, routed+shared) rel-L2 fast_i4 vs EXL3 %.4e "
                    "(|y| rms %.4e)\n", opt.rank, w.index, R, std::sqrt(num / std::max(den, 1e-30)), std::sqrt(den / static_cast<double>(y0.size())));
    }
    bool i4_pack_suh() const { return false; }
    void i4_pack_read(const I4Routed& q, unsigned layer, const std::vector<Mlp>& experts) {
        namespace X = glm::pf::exact_moe::iu4;
        const auto t0 = Clock::now();
        const std::string f = i4pack.dir + "/L" + std::to_string(layer) + ".i4";
        std::FILE* in = std::fopen(f.c_str(), "rb");
        if (!in) throw std::runtime_error("glm fast_i4 pack: cannot open " + f);
        std::uint64_t h = 0;
        try {
            for (unsigned r = 0; r < 2 * i4_matrices; ++r) {
                auto* dst = r % 2 == 0 ? q.wq[r / 2] : reinterpret_cast<std::uint8_t*>(q.ws[r / 2]);
                const std::uint64_t n = r % 2 == 0 ? X::weight_bytes() : X::scale_bytes(r / 2);
                for (std::uint64_t o = 0; o < n; o += fast_i4::kChunk) {
                    const std::uint64_t c = std::min(fast_i4::kChunk, n - o);
                    std::uint8_t* s = lib.staging(c);
                    if (std::fread(s, 1, c, in) != c) throw std::runtime_error("glm fast_i4 pack: short read " + f);
                    if (i4pack.verify) h = fast_i4::fold(h, fast_i4::chunk_hash(s, c));
                    hip_ok(hipMemcpy(dst + o, s, c, hipMemcpyHostToDevice), "fast_i4 pack upload");
                }
            }
            if (i4_pack_suh()) {
                const std::uint64_t c = std::uint64_t{kExperts} * expert_i * 2U;
                std::uint8_t* s = lib.staging(c);
                if (std::fread(s, 1, c, in) != c) throw std::runtime_error("glm fast_i4 pack: short read " + f);
                if (i4pack.verify) h = fast_i4::fold(h, fast_i4::chunk_hash(s, c));
                for (unsigned e = 0; e < kExperts; ++e)
                    hip_ok(hipMemcpy(const_cast<std::uint16_t*>(experts[e].down.raw.suh), s + std::size_t{e} * expert_i * 2U, expert_i * 2U,
                                     hipMemcpyHostToDevice), "fast_i4 pack suh");
            }
        } catch (...) {
            std::fclose(in);
            throw;
        }
        std::fclose(in);
        if (i4pack.verify && h != i4pack.layers.at(layer).second)
            throw std::runtime_error("glm fast_i4 pack: checksum mismatch in " + f + " (pack corrupt: rewrite it or unset STRIX_GLM_FAST_I4_PACK)");
        i4pack.read_seconds += std::chrono::duration<double>(Clock::now() - t0).count();
        ++i4pack.read_layers;
    }
    void i4_pack_write(const I4Routed& q, unsigned layer, const std::vector<Mlp>& experts) {
        namespace X = glm::pf::exact_moe::iu4;
        const std::uint64_t total = i4_matrices * (X::weight_bytes() + X::scale_bytes(0)) + (i4_pack_suh() ? std::uint64_t{kExperts} * expert_i * 2U : 0U);
        if (!i4pack.want_write(total)) return;
        const auto t0 = Clock::now();
        const std::string f = i4pack.dir + "/L" + std::to_string(layer) + ".i4", tmp = f + ".tmp";
        std::FILE* out = std::fopen(tmp.c_str(), "wb");
        if (!out) throw std::runtime_error("glm fast_i4 pack: cannot create " + tmp);
        std::uint64_t h = 0;
        try {
            for (unsigned r = 0; r < 2 * i4_matrices; ++r) {
                const auto* src = r % 2 == 0 ? q.wq[r / 2] : reinterpret_cast<const std::uint8_t*>(q.ws[r / 2]);
                const std::uint64_t n = r % 2 == 0 ? X::weight_bytes() : X::scale_bytes(r / 2);
                for (std::uint64_t o = 0; o < n; o += fast_i4::kChunk) {
                    const std::uint64_t c = std::min(fast_i4::kChunk, n - o);
                    std::uint8_t* s = lib.staging(c);
                    hip_ok(hipMemcpy(s, src + o, c, hipMemcpyDeviceToHost), "fast_i4 pack download");
                    h = fast_i4::fold(h, fast_i4::chunk_hash(s, c));
                    if (std::fwrite(s, 1, c, out) != c) throw std::runtime_error("glm fast_i4 pack: short write " + tmp);
                }
            }
            if (i4_pack_suh()) {
                const std::uint64_t c = std::uint64_t{kExperts} * expert_i * 2U;
                std::uint8_t* s = lib.staging(c);
                for (unsigned e = 0; e < kExperts; ++e)
                    hip_ok(hipMemcpy(s + std::size_t{e} * expert_i * 2U, experts[e].down.raw.suh, expert_i * 2U, hipMemcpyDeviceToHost),
                           "fast_i4 pack suh");
                h = fast_i4::fold(h, fast_i4::chunk_hash(s, c));
                if (std::fwrite(s, 1, c, out) != c) throw std::runtime_error("glm fast_i4 pack: short write " + tmp);
            }
        } catch (...) {
            std::fclose(out);
            std::remove(tmp.c_str());
            throw;
        }
        if (std::fclose(out) != 0) throw std::runtime_error("glm fast_i4 pack: close failed " + tmp);
        std::filesystem::rename(tmp, f);
        i4pack.layers[layer] = {total, h};
        i4pack.save_manifest();
        if (i4pack.write_left != ~0U) --i4pack.write_left;
        ++i4pack.written_layers;
        i4pack.write_seconds += std::chrono::duration<double>(Clock::now() - t0).count();
    }
    void load_layer(unsigned l) {
        LayerW w;
        w.index = l;
        w.mtp = l == kMtpLayer;
        w.kda = !w.mtp && is_kda_layer(l);
        w.moe = is_moe_layer(l);
        const std::string p = L(l), a = p + "self_attn.";
        if (!w.mtp) load_hc(w, l);
        w.in_norm = static_cast<const std::uint16_t*>(lib.upload_rows(p + "input_layernorm.weight"));
        w.post_norm = static_cast<const std::uint16_t*>(lib.upload_rows(p + "post_attention_layernorm.weight"));
        if (w.kda) {
            const std::uint64_t r = opt.rank, H = kda_heads, C = H * kKdaDim;
            auto& k = w.kdaw;
            k.heads = kda_heads;
            for (unsigned c = 0; c < 3; ++c)
                k.qkv[c] = attn_tierb ? lib.out_slice_raw(a + "qkv_proj", c * kKdaQkv + r * C, C) : lib.out_slice(a + "qkv_proj", c * kKdaQkv + r * C, C);
            k.o = attn_tierb ? lib.in_slice_raw(a + "o_proj", r * C, C) : lib.in_slice(a + "o_proj", r * C, C);
            k.f_a = static_cast<const std::uint16_t*>(lib.upload_rows(a + "f_a_proj.weight"));
            k.g_a = static_cast<const std::uint16_t*>(lib.upload_rows(a + "g_a_proj.weight"));
            k.f_b = static_cast<const std::uint16_t*>(lib.upload_rows(a + "f_b_proj.weight", r * C, C));
            k.g_b = static_cast<const std::uint16_t*>(lib.upload_rows(a + "g_b_proj.weight", r * C, C));
            k.b = static_cast<const std::uint16_t*>(lib.upload_rows(a + "b_proj.weight", r * H, H));
            k.f_a_packed = pack_r1b(k.f_a, kKdaDim);
            k.g_a_packed = pack_r1b(k.g_a, kKdaDim);
            k.b_packed = pack_r1b(k.b, kda_heads);
            k.conv = static_cast<const std::uint16_t*>(
                lib.upload_row_ranges(a + "conv1d.weight", {{r * C, C}, {kKdaQkv + r * C, C}, {2 * kKdaQkv + r * C, C}}));
            k.A_log = static_cast<const float*>(lib.upload_rows(a + "A_log", r * H, H));
            k.dt_bias = static_cast<const float*>(lib.upload_rows(a + "dt_bias", r * C, C));
            k.o_norm = static_cast<const std::uint16_t*>(lib.upload_rows(a + "o_norm.weight"));
            for (const char* n : {"f_a_proj.weight", "g_a_proj.weight", "f_b_proj.weight", "g_b_proj.weight", "b_proj.weight"})
                if (lib.dtype(a + n) != "F16") throw std::runtime_error(std::string("expected F16: ") + a + n);
            if (lib.dtype(a + "conv1d.weight") != "BF16" || lib.dtype(a + "o_norm.weight") != "BF16")
                throw std::runtime_error("expected BF16 conv1d / o_norm: " + a);
            w.kst.S = lib.state<float>(static_cast<std::size_t>(opt.slots) * H * kKdaDim * kKdaDim);
            w.kst.conv = lib.state<float>(static_cast<std::size_t>(opt.slots) * 3 * C * 3);
            if (opt.spec) {  // deferred-replay sets (design notes section 1): [slots][2][8] rows
                const std::size_t sets = static_cast<std::size_t>(opt.slots) * 2 * 8;
                w.kst.rep_qkv = lib.state<float>(sets * 3 * C);
                w.kst.rep_scr = lib.state<float>(sets * H * 4 * kKdaDim);
                w.kst.rep_beta = lib.state<float>(sets * H);
            }
        } else {
            load_dsa(w, a);
            w.dst.shard_layer = l;
            w.dst.dump_rank = opt.rank;
        }
        if (w.moe) {
            load_moe(w, p + "mlp.");
        } else {
            const std::uint64_t r = opt.rank, I = dense_i;
            const std::string m = p + "mlp.";
            w.dense = Mlp{lib.out_slice(m + "gate_proj", r * I, I), lib.out_slice(m + "up_proj", r * I, I), lib.in_slice(m + "down_proj", r * I, I)};
        }
        if (w.mtp) {
            w.eh_proj = lib.whole(p + "eh_proj");
            w.enorm = static_cast<const std::uint16_t*>(lib.upload_rows(p + "enorm.weight"));
            w.hnorm = static_cast<const std::uint16_t*>(lib.upload_rows(p + "hnorm.weight"));
            w.head_norm = static_cast<const std::uint16_t*>(lib.upload_rows(p + "shared_head.norm.weight"));
        }
        layers.push_back(std::move(w));
    }

    // The load sequence, run twice: once as a dry run (lib.planning(): records every allocation, nothing read or
    // allocated) and once for real with a memory check after every step. `step` is called after each step.
    template <class Step> void load_steps(bool want_head, Step&& step) {
        embed = static_cast<std::uint16_t*>(lib.upload_rows(kP + "embed_tokens.weight"));
        if (lib.dtype(kP + "embed_tokens.weight") != "BF16") throw std::runtime_error("embedding must be BF16");
        step("embed");
        for (unsigned l = 0; l < opt.layers; ++l) {
            load_layer(l);
            step("layer " + std::to_string(l));
        }
        if (opt.mtp) {
            load_layer(kMtpLayer);
            step("mtp");
        }
        if (want_head) {
            final_norm = static_cast<const std::uint16_t*>(lib.upload_rows(kP + "norm.weight"));
            head = lib.out_slice("lm_head", vocab0, vocab_n);
            step("head");
        }
    }

    // STRIX_GLM_MEMTRACE=FILE ('-' = stderr): one memory line per second for the engine's lifetime (load + forwards).
    // Watchdog (default ON, STRIX_GLM_WATCHDOG=0 off): aborts the process (exit 86) near the OOM line (glm_mem.hpp).
    void start_monitor() {
        const char* tp = std::getenv("STRIX_GLM_MEMTRACE");
        monitor = std::make_unique<mem::Monitor>(probe, tp ? std::string(tp) : std::string(), mem::Watchdog::from_env(opt.host_budget));
        monitor->phase("plan");
    }

    // Physical bytes of one hipMalloc of the plan's dominant size (the expert linear): K touched allocations, delta of
    // the driver's device counters (amdgpu GTT + VRAM used; else /proc/meminfo GPUActive). STRIX_GLM_ALLOC_GRANULE=BYTES
    // skips the measurement; no counters -> 2 MiB assumed.
    void calibrate() {
        const std::uint64_t size = plan.dominant_size();
        if (const char* e = std::getenv("STRIX_GLM_ALLOC_GRANULE"); e != nullptr && *e) {
            amodel = mem::AllocModel{};
            amodel.granule = std::strtoull(e, nullptr, 10);
            if (amodel.granule == 0) throw std::invalid_argument("STRIX_GLM_ALLOC_GRANULE must be > 0 bytes");
            return;
        }
        const mem::Snapshot a = probe.read();
        if (!a.dev_counters() || size == 0) {
            amodel = mem::infer_alloc_model(size, 0.0);
            std::fprintf(stderr, "mem: no device memory counters: allocation cost assumed %s\n", amodel.str().c_str());
            return;
        }
        // Retained batches (a repeatable slope, not one global delta): kBatches x K touched allocations, all kept until
        // the end; per batch the larger of (GTT + VRAM used) and GPUActive deltas. Two attempts, then the conservative model.
        constexpr int K = 16, kBatches = 4;
        mem::admit("allocation calibration",
                   {{"calibration buffers (2x worst case)", 2 * K * kBatches * mem::round_up(size, 2 * mem::kMiB), 0}}, a, margins);
        struct Bufs {  // freed on every exit path (a HIP error included)
            std::vector<void*> p;
            ~Bufs() {
                for (void* q : p) (void)hipFree(q);
            }
        };
        bool consistent = false;
        for (int attempt = 0; attempt < 2 && !consistent; ++attempt) {
            Bufs bufs;
            bufs.p.reserve(static_cast<std::size_t>(K) * kBatches);  // push_back after hipMalloc cannot throw
            std::vector<double> per;
            std::string detail;
            mem::Snapshot prev = probe.read();
            for (int b = 0; b < kBatches; ++b) {
                for (int i = 0; i < K; ++i) {
                    void* p = nullptr;
                    hip_ok(hipMalloc(&p, size), "calibration malloc");
                    bufs.p.push_back(p);
                    hip_ok(hipMemset(p, 0, size), "calibration memset");
                }
                hip_ok(hipDeviceSynchronize(), "calibration sync");
                const mem::Snapshot now_s = probe.read();
                const double dev = now_s.dev_used() > prev.dev_used() ? static_cast<double>(now_s.dev_used() - prev.dev_used()) / K : 0.0;
                const double act = now_s.gpu_active > prev.gpu_active ? static_cast<double>(now_s.gpu_active - prev.gpu_active) / K : 0.0;
                per.push_back(std::max(dev, act));
                detail += " " + std::to_string(static_cast<long long>(dev)) + "/" + std::to_string(static_cast<long long>(act));
                prev = now_s;
            }
            amodel = mem::model_from_batches(size, per, consistent);
            std::fprintf(stderr, "mem: calibration attempt %d: %d x %d x hipMalloc(%llu B), B each per batch (GTT+VRAM / GPUActive):%s -> %s%s\n",
                         attempt + 1, kBatches, K, static_cast<unsigned long long>(size), detail.c_str(), amodel.str().c_str(),
                         consistent ? "" : " (batches disagree)");
        }
    }

    LoadReport load(const std::function<unsigned(const mem::LoadAdmission&)>& select_max_rows,
                    const std::function<void(const mem::LoadAdmission&)>& admit_prefill,
                    const std::function<std::shared_ptr<PrefillPath>(bool)>& make_prefill) {
        try {
            return load_checked(select_max_rows, admit_prefill, make_prefill);
        } catch (const InstallRefused&) {
            throw;  // a coordinated capacity fallback, not a failed load
        } catch (const std::exception& e) {  // printed before the unwinding (HIP teardown can stall)
            std::fprintf(stderr, "load: rank %u FAILED: %s\n", opt.rank, e.what());
            if (monitor) monitor->phase("failed");
            throw;
        }
    }

    LoadReport load_checked(const std::function<unsigned(const mem::LoadAdmission&)>& select_max_rows,
                            const std::function<void(const mem::LoadAdmission&)>& admit_prefill,
                    const std::function<std::shared_ptr<PrefillPath>(bool)>& make_prefill) {
        model::validate_model_directory(opt.model_dir);
        load_identity = identity;
        const auto t0 = Clock::now();
        start_monitor();
        hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
        const auto& pf_policy = pf::capacity_policy();
        const auto initial_pf_layout = pf::capacity_candidates(pf_policy, kv_shard != nullptr).front();
        pf::capacity_layout() = initial_pf_layout;
        const bool capacity_peer = opt.world == 2 && comm != nullptr && !opt.external_exchange && !load_test;
        if (opt.world == 2 && comm != nullptr && !opt.external_exchange && !load_test) {
            // All policies participate, so mismatched auto/explicit peers fail
            // before only one rank can enter the memory-plan vote.
            const std::array<std::uint32_t, 5> policy{kv_shard_auto ? 2U : kv_shard ? 1U : 0U,
                static_cast<std::uint32_t>(pf_policy.scratch), static_cast<std::uint32_t>(pf_policy.query_window + 1),
                pf_seqpar ? 4U | (opt.pf_pipeline << 3U) | (pf_streams == 1 ? 1U << 5U : 0U) : 0U,
                model_fast_i4 ? 0x46493400U ^ i4_policy.tag(static_cast<unsigned>(i4_scale_rule))
                              : 0x45584c33U /* model format + layer mask + scope + scale rule */};
            std::array<std::uint32_t, 5> peer_policy{};
            comm->exchange_host({reinterpret_cast<const std::byte*>(policy.data()), sizeof(policy)},
                                {reinterpret_cast<std::byte*>(peer_policy.data()), sizeof(peer_policy)},
                                {static_cast<std::uint32_t>(dist::TagKind::raw_slots), 0xFFFFFFFDU, 0});
            if (peer_policy != policy)
                throw std::runtime_error(peer_policy[4] != policy[4]
                                             ? "glm: rank policy mismatch: STRIX_GLM_MODEL_FORMAT differ between the ranks"
                                             : "glm: rank policy mismatch (KV_SHARD/PF_SCRATCH/PF_ATTN_QWIN/PF_SEQPAR)");
            if (model_fast_i4) std::printf("rank %u model format agreed: fast_i4 on both ranks\n", opt.rank);
        }
        // STRIX_GLM_SMOKE_HEAD=1 (test only): the head on a truncated stack, allowed only for a synthetic model
        // (tools/glm/make_synth_model.py writes "synthetic": true into the index metadata); refused for any other model.
        if (const char* e = std::getenv("STRIX_GLM_SMOKE_HEAD"); e != nullptr && std::string(e) == "1" && opt.layers != kLayers) {
            std::ifstream f(opt.model_dir + "/model.safetensors.index.json");
            const std::string idx((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            if (idx.find("\"synthetic\": true") == std::string::npos)
                throw std::invalid_argument("STRIX_GLM_SMOKE_HEAD=1 is only allowed on a synthetic model");
            smoke_head = true;
        }
        const bool want_head = opt.layers == kLayers || smoke_head;

        // 1. Plan: the exact list of device allocations, staging and file bytes, before the first byte is allocated.
        margins = mem::Margins::from_env();
        std::uint64_t prefill_install = 0;
        auto make_plan = [&]() {
            plan = {};
            lib.set_plan(&plan);
            load_steps(want_head, [&](const std::string& n) { plan.step(n); });
            alloc();
            plan.step("buffers");
            lib.set_plan(nullptr);
            // Planned handles are null: conservatively include every shared
            // expert, using the SAME accounting as the later install guard.
            prefill_install = prefill_install_bytes(true);
            if (pf_seqpar) prefill_install += xfer::SeqparExchange::workspace_bytes(opt.max_rows);
            layers.clear();
        };
        make_plan();
        // 2. Physical cost of the dominant allocation, 3. admission: fail closed before loading anything.
        monitor->phase("calibrate");
        calibrate();
        auto load_terms = [&]() -> std::vector<mem::Term> {
            return {{"device plan (physical)", plan.physical(amodel), 0},
                    {"first-forward reserve (strix workspace, runtime)", fwd_reserve(), 0},
                    {"pinned staging", mem::round_up(std::max<std::uint64_t>(plan.stage_max, 1), 2 * mem::kMiB), 0},
                    {"pageable whole-tensor reads", 0, plan.whole_max}, {"O_DIRECT bounce", 0, plan.bounce}};
        };
        auto select_rows = [&](bool recheck) {
            std::size_t dev_free = 0, dev_total = 0;
            hip_ok(hipMemGetInfo(&dev_free, &dev_total), "row admission mem info");
            const unsigned rows = select_max_rows({load_terms(), probe.read(), margins, prefill_install, dev_free, recheck, amodel,
                                                   plan.logical, plan.physical(amodel), kv_shard != nullptr, false, fwd_reserve()});
            if (rows == 0 || rows > opt.max_rows) throw std::invalid_argument("load row selector must reduce max_rows within 1..max_rows");
            if (rows != opt.max_rows) {
                opt.max_rows = rows;
                make_plan();
            }
            if (auto* cx = dynamic_cast<CommExchange*>(fallback_xch.get())) cx->resize_rows(std::max(8U, rows));
        };
        auto prefill_admission = [&](bool candidate) {
            std::size_t dev_free = 0, dev_total = 0;
            hip_ok(hipMemGetInfo(&dev_free, &dev_total), "prefill layout mem info");
            return mem::LoadAdmission{load_terms(), probe.read(), margins, prefill_install, dev_free,
                                      opt.auto_rows, amodel, plan.logical, plan.physical(amodel),
                                      kv_shard != nullptr, candidate, fwd_reserve()};
        };
        if (select_max_rows) {
            select_rows(false);
            // Only negotiated auto peers enter this second round. It sees actual communication allocations,
            // so an admission refusal at the larger size becomes a shared fallback before the first weight.
            if (opt.auto_rows) select_rows(true);
        }
        if (model_fast_i4) {
            // Context, slots, KV, state and prefill workspace keep their planned sizes.
            std::size_t dev_free = 0, dev_total = 0;
            hip_ok(hipMemGetInfo(&dev_free, &dev_total), "fast_i4 budget");
            unsigned n = 0, total = 0;
            for (unsigned l = 0; l < opt.layers; ++l) {
                n += i4_layer_at(l) ? 1U : 0U;
                total += l != kMtpLayer && is_moe_layer(l) ? 1U : 0U;
            }
            auto a = prefill_admission(false);
            std::printf("glm fast_i4 budget: rank %u context %llu x %u slots, max_rows %u (KV/state/workspace planned first); "
                        "%u of %u routed layers INT4 (%s), their routed EXL3 released; device plan %.2f GiB of %.2f GiB free; decode W4A16\n",
                        opt.rank, static_cast<unsigned long long>(opt.capacity), opt.slots, opt.max_rows, n, total,
                        i4_matrices == 3U ? "gate, up, down" : "gate, up",
                        static_cast<double>(plan.physical(amodel)) / double(1ULL << 30), static_cast<double>(dev_free) / double(1ULL << 30));
            if (!mem::fits(a.terms, a.snapshot, a.margins, a.device_free))
                throw std::runtime_error("fast_i4: the routed INT4 layers do not fit after configured KV/state/workspace");
            if (admit_prefill) admit_prefill(a);
        }
        if (admit_prefill || kv_shard_auto) {
            // Exhaust replicated capacity levers BEFORE considering sharding.
            // The same candidate must pass on both ranks; candidate checks never
            // install a path. The post-load guard keeps the chosen layout fixed.
            struct Vote { std::uint32_t fits = 0; std::array<char, 768> reason{}; };
            std::array<Vote, 2> last_votes{};
            unsigned phase = capacity_skip;
            pf::CapacityLayout selected;
            const bool initial_sharded = kv_shard != nullptr;
            const bool admitted = pf::select_capacity_layout(pf_policy, initial_sharded, kv_shard_auto,
                [&](pf::CapacityLayout layout) {
                    last_votes = {};
                    std::exception_ptr install_error;
                    try {
                        if (layout.sharded && !kv_shard) {
                            kv_shard = std::make_unique<KvShard>(*comm, opt.rank);
                            bind_shard_exchange();
                            make_plan();
                        }
                        pf::capacity_layout() = layout;
                        if (opt.world == 2 && !capacity_peer && !load_test &&
                            (layout.routed != initial_pf_layout.routed || layout.query_window != initial_pf_layout.query_window))
                            throw std::runtime_error("automatic scratch/window fallback requires the internal two-rank startup policy vote");
                        const auto a = prefill_admission(true);
                        if (!mem::fits(a.terms, a.snapshot, a.margins, a.device_free))
                            throw std::runtime_error("load plan / first-forward reserve exceeds RAM, GTT or device headroom");
                        if (!admit_prefill) throw std::runtime_error("unified prefill admission callback required");
                        admit_prefill(a);
                        last_votes[0].fits = 1;
                        std::snprintf(last_votes[0].reason.data(), last_votes[0].reason.size(), "admitted");
                    } catch (const std::runtime_error& e) {
                        std::snprintf(last_votes[0].reason.data(), last_votes[0].reason.size(), "%s", e.what());
                    } catch (...) {
                        install_error = std::current_exception();
                        last_votes[0].fits = 2;
                        std::snprintf(last_votes[0].reason.data(), last_votes[0].reason.size(), "install failed");
                    }
                    if (capacity_peer) {
                        comm->exchange_host({reinterpret_cast<const std::byte*>(&last_votes[0]), sizeof(Vote)},
                                            {reinterpret_cast<std::byte*>(&last_votes[1]), sizeof(Vote)},
                                            {static_cast<std::uint32_t>(dist::TagKind::raw_slots), 0xFFFFFFFEU, phase});
                        if (last_votes[1].fits > 2U || last_votes[1].reason.back() != '\0')
                            throw std::runtime_error("glm capacity: invalid auto admission vote");
                    } else {
                        last_votes[1] = last_votes[0];
                    }
                    if (install_error) std::rethrow_exception(install_error);
                    if (last_votes[1].fits == 2U) throw std::runtime_error("peer install failed");
                    std::fprintf(stderr, "glm capacity candidate: rank=%u kv=%s scratch=%s qwin=%u local=%s peer=%s\n",
                        opt.rank, layout.sharded ? "sharded" : "replicated", layout.routed ? "routed" : "full",
                        layout.query_window, last_votes[0].reason.data(), last_votes[1].reason.data());
                    capacity_candidate = phase++;
                    return std::array<bool, 2>{last_votes[0].fits != 0, last_votes[1].fits != 0};
                }, selected, capacity_skip);
            if (!admitted)
                throw std::runtime_error(std::string("glm capacity: no layout admitted; local: ") +
                    last_votes[0].reason.data() + "; peer: " + last_votes[1].reason.data());
            pf::capacity_layout() = selected;
            const std::string requested_window = pf_policy.query_window < 0 ? "auto" : std::to_string(pf_policy.query_window);
            std::fprintf(stderr, "glm capacity: rank=%u context=%llu scratch requested=%s resolved=%s; qwin requested=%s resolved=%u; kv=%s reason=%s\n",
                opt.rank, static_cast<unsigned long long>(opt.capacity), pf::scratch_policy_name(pf_policy.scratch),
                selected.routed ? "routed" : "full", requested_window.c_str(), selected.query_window,
                selected.sharded ? "sharded" : "replicated",
                initial_sharded ? "explicit-sharding-first-admitted-scratch-window" : pf::capacity_layout_reason(selected));
        }
        if (admit_prefill) vote_prefill_install([&] { admit_prefill(prefill_admission(false)); }, 0xFFFFFFFDU);
        const std::uint64_t stage = mem::round_up(std::max<std::uint64_t>(plan.stage_max, 1), 2 * mem::kMiB);
        lib.set_stage_bytes(stage);
        const std::uint64_t phys = plan.physical(amodel);
        const mem::Snapshot s0 = probe.read();
        std::fprintf(stderr, "mem: rank %u %s\nmem: node %s\n", opt.rank, plan.summary(amodel).c_str(), mem::describe(s0).c_str());
        if (const char* e = std::getenv("STRIX_GLM_MEM_ADMIT"); e != nullptr && std::string(e) == "0") {
            std::fprintf(stderr, "mem: WARNING admission check disabled (STRIX_GLM_MEM_ADMIT=0); the watchdog stays on\n");
        } else {
            mem::admit("rank " + std::to_string(opt.rank) + " load (" + std::to_string(opt.layers) + " layers)",
                       load_terms(),
                       s0, margins);
        }
        if (const char* e = std::getenv("STRIX_GLM_MEM_PLAN_ONLY"); e != nullptr && std::string(e) == "1") {  // --mem-plan
            monitor->phase("plan-only");
            LoadReport rep;
            rep.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
            return rep;
        }
        lib.reserve_staging();
        base = probe.read();
        const bool load_limit = base.dev_counters() && mem::Watchdog::from_env().max_gtt_used == 0;
        if (load_limit) monitor->set_max_gtt_used(base.dev_used() + phys + 2 * mem::kGiB);

        // 4. The load, checked after every step against the plan (measured gtt_used growth) and the node's headroom.
        monitor->phase("load");
        if (model_fast_i4) {
            i4pack = fast_i4::Pack::open(opt.model_dir, opt.rank, opt.world, expert_i, i4_matrices, "none",
                                         i4_scale_rule == strix::core::iu4_scale_rule_legacy ? "" : "fast_full");
        }
        const auto t_i4 = Clock::now();
        std::size_t idx = 0;
        load_steps(want_head, [&](const std::string& n) { check_step(n, idx++, t0); });
        if (model_fast_i4) {
            const auto freed = lib.i4_temp_release();
            std::printf("glm fast_i4 startup: rank %u %u routed layers stored INT4, %.3f GiB; "
                        "routed EXL3 released (scratch %.3f GiB); load+pack %.2f s; decode W4A16\n",
                        opt.rank, i4_layers, static_cast<double>(i4_bytes) / double(1ULL << 30),
                        static_cast<double>(freed) / double(1ULL << 30), std::chrono::duration<double>(Clock::now() - t_i4).count());
        }
        lib.release_staging();
        alloc();
        check_step("buffers", idx++, t0);
        // parity: the real pass made exactly the planned allocations, in order (covers raw linears, MTP, spec, options)
        const auto& real = lib.recorded();
        if (real != plan.device) {
            std::size_t i = 0;
            while (i < real.size() && i < plan.device.size() && real[i] == plan.device[i]) ++i;
            throw std::logic_error("memory plan diverged from the load at allocation " + std::to_string(i) + " (plan " +
                                   std::to_string(plan.device.size()) + " allocations, real " + std::to_string(real.size()) + ")");
        }
        // the plan-derived device limit covers the load only: later owners (prefill path, strix workspaces) admit their own
        // allocations; the RAM headroom / swap / PSI / RSS thresholds stay on
        if (load_limit) monitor->set_max_gtt_used(0);
        monitor->phase("ready");
        loaded = true;
        if (make_prefill || (pf_installed && pf)) vote_prefill_install([&] {
            auto path = make_prefill ? make_prefill(kv_shard != nullptr) : pf;
            if (!path) return;
            install_prefill(*path);
            pf = std::move(path);
            pf_installed = true;
        }, 0xFFFFFFFCU);
        LoadReport rep;
        rep.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
        rep.gib_read = pack.gib_read();
        rep.device_gib = lib.device_gib();
        return rep;
    }

    // Headroom kept for allocations made after the load by their owners' first use (strix matmul workspace, HIP runtime);
    // STRIX_GLM_MEM_FWD_RESERVE_GIB (default 0.5). A prefill path admits its own buffers (make_prefill_path).
    static std::uint64_t fwd_reserve() {
        const char* e = std::getenv("STRIX_GLM_MEM_FWD_RESERVE_GIB");
        return e != nullptr && *e ? static_cast<std::uint64_t>(std::strtod(e, nullptr) * static_cast<double>(mem::kGiB)) : mem::kGiB / 2;
    }

    // After load step `i`: measured gtt_used growth must follow the plan (a model error fails closed long before the
    // OOM line) and the rest of the plan must still fit the node's current headroom (other tenants can grow).
    void check_step(const std::string& name, std::size_t i, Clock::time_point t0) {
        if (i >= plan.mark.size() || plan.step_name[i] != name) throw std::logic_error("load diverged from its plan at " + name);
        monitor->counters(lib.allocated_bytes(), pack.bytes_read());
        const mem::Snapshot s = probe.read();
        const std::uint64_t pred = plan.physical(amodel, plan.mark[i]), total = plan.physical(amodel);
        const std::uint64_t measured = s.dev_counters() && base.dev_counters() && s.dev_used() > base.dev_used() ? s.dev_used() - base.dev_used() : 0;
        const bool report = name.rfind("layer ", 0) != 0 || (std::stoul(name.substr(6)) % 5 == 4) || i + 2 >= plan.mark.size();
        if (report)
            std::fprintf(stderr, "load: rank %u %s %.1f s %.1f GiB read %.2f GiB device (plan %.2f, measured +%.2f GiB) headroom ram %.2f gtt %.2f GiB\n",
                         opt.rank, name.c_str(), std::chrono::duration<double>(Clock::now() - t0).count(), pack.gib_read(),
                         mem::gib(lib.allocated_bytes()), mem::gib(pred), mem::gib(measured), mem::gib(s.ram_headroom()),
                         mem::gib(s.gtt_headroom()));
        if (s.dev_counters() && base.dev_counters() && measured > pred + pred / 32 + 512 * mem::kMiB)
            throw mem::BudgetError("memory budget: STOPPED at " + name + ": device memory used grew " + std::to_string(measured >> 20) +
                                   " MiB but the plan predicts " + std::to_string(pred >> 20) + " MiB (" + amodel.str() +
                                   "): the allocation model is wrong; nothing more is allocated. node: " + mem::describe(s));
        if (std::getenv("STRIX_GLM_MEM_ADMIT") == nullptr || std::string(std::getenv("STRIX_GLM_MEM_ADMIT")) != "0")
            mem::admit("rest of the load after " + name, {{"device plan still to allocate", total > pred ? total - pred : 0, 0},
                                                          {"first-forward reserve", fwd_reserve(), 0},
                                                          {"pageable whole-tensor reads", 0, plan.whole_max}},
                       s, margins);
    }

    // Everything a chunked prefill through `p` needs, allocated now so a prefill allocates no device memory: prepared
    // copies of raw-only shared experts (needs_prepared_shared), the MoE seam's per-layer tables (used per row when
    // moe_by_row, and by decode), and the LinearRunner workspace for the path's calls and every GEMV call. Each step
    // is admitted against hipMemGetInfo free and MemAvailable with the path's margin before it allocates.
    template<class Install>
    void vote_prefill_install(Install&& install, std::uint32_t tag) {
        struct Vote { std::uint32_t fits = 0, seqpar = 0; std::array<char, 768> reason{}; };
        std::array<Vote, 2> votes{};
        std::exception_ptr install_error;
        try {
            install();
            votes[0].fits = 1;
            std::snprintf(votes[0].reason.data(), votes[0].reason.size(), "installed");
        } catch (const std::runtime_error& e) {
            // Retry only an admission refusal. Compiler/runtime and model errors
            // are not evidence that a smaller layout is safe to execute.
            votes[0].fits = std::string(e.what()).find("refusing") != std::string::npos ? 0U : 2U;
            std::snprintf(votes[0].reason.data(), votes[0].reason.size(), "%s", e.what());
        } catch (...) {
            install_error = std::current_exception();
            votes[0].fits = 3;
            std::snprintf(votes[0].reason.data(), votes[0].reason.size(), "install failed");
        }
        votes[0].seqpar = pf_seqpar ? 1U : 0U;
        if (opt.world == 2 && comm && !load_test && !opt.external_exchange) {
            comm->exchange_host({reinterpret_cast<const std::byte*>(&votes[0]), sizeof(Vote)},
                                {reinterpret_cast<std::byte*>(&votes[1]), sizeof(Vote)},
                                {static_cast<std::uint32_t>(dist::TagKind::raw_slots), tag, capacity_candidate});
            if (votes[1].fits > 3U || votes[1].seqpar > 1U || votes[1].reason.back() != '\0')
                throw std::runtime_error("glm capacity: invalid install vote");
        } else votes[1] = votes[0];
        if (install_error) std::rethrow_exception(install_error);
        if (votes[1].fits == 3U) throw std::runtime_error("peer install failed");
        if (votes[0].fits == 1U && votes[1].fits == 1U && votes[0].seqpar != votes[1].seqpar)
            throw std::runtime_error("glm: rank policy mismatch (PF_SEQPAR install decision)");
        if (votes[0].fits == 1U && votes[1].fits == 1U) return;
        const std::string reason = std::string("glm capacity install: local: ") + votes[0].reason.data() +
            "; peer: " + votes[1].reason.data();
        std::fprintf(stderr, "%s; %s\n", reason.c_str(),
            votes[0].fits == 2U || votes[1].fits == 2U ? "fatal install error" : "trying next permitted layout");
        if (votes[0].fits == 2U || votes[1].fits == 2U) throw std::runtime_error(reason);
        throw InstallRefused(reason);
    }

    std::uint64_t prefill_install_bytes(bool prepared_shared) const {
        constexpr std::uint64_t kGranule = std::uint64_t{2} << 20;
        auto granules = [&](std::uint64_t b) { return (b + kGranule - 1) / kGranule * kGranule; };
        std::uint64_t need = 0, staging = 0;
        for (auto& w : layers) {
            if (!w.moe) continue;
            need += moe->warm_bytes(w.moew);
            if (!prepared_shared) continue;
            for (const Linear* l : {&w.moew.shared.gate, &w.moew.shared.up, &w.moew.shared.down}) {
                if (l->prepared != nullptr) continue;
                const std::uint64_t parts = Lib::raw_trellis_bytes(l->raw) + std::uint64_t{l->raw.input} * 2 + std::uint64_t{l->raw.output} * 2;
                need += granules(parts + 3 * 256);  // one allocation per prepared matrix, regions 256-byte aligned
                staging = std::max(staging, parts);  // prepare_from_raw's host copy, one matrix at a time
            }
        }
        return need + staging;
    }

    void install_prefill(PrefillPath& p) {
        if (pf_streams == 1 && (!p.supports_pipeline() || p.moe_by_row() || opt.max_rows < 2))
            throw std::invalid_argument("STRIX_GLM_PF_STREAMS=1 requires row-invariant grouped prefill");
        if (pf_seqpar_auto_eligible) {
            pf_seqpar = p.supports_pipeline() && opt.max_rows >= 2;
            if (!pf_seqpar)
                std::fprintf(stderr, "glm prefill seqpar auto declined: %s; legacy path\n",
                             !p.supports_pipeline() ? "prefill path does not support pipeline" : "max_rows < 2");
        }
        const auto need = prefill_install_bytes(p.needs_prepared_shared());
        if (need != 0) {
            hip_ok(hipDeviceSynchronize(), "sync");
            Lib::admit(need, p.admission_margin(), "prefill install (prepared shared experts, MoE tables)");
            for (std::size_t i = 0; i < layers.size(); ++i) {
                auto& w = layers[i];
                if (!w.moe) continue;
                moe->warm(w.moew);
                if (!p.needs_prepared_shared()) continue;
                const std::string s = "layer " + std::to_string(i) + " shared ";
                lib.prepare_from_raw(w.moew.shared.gate, s + "gate");
                lib.prepare_from_raw(w.moew.shared.up, s + "up");
                lib.prepare_from_raw(w.moew.shared.down, s + "down");
            }
        }
        // No workspace growth during a prefill: the path's own calls and every GEMV call (gemv-mode Tier B, the head)
        // fit. The reservation fails closed (device free and MemAvailable must cover it + the margin while the old
        // workspace still exists).
        lib.reserve(std::max(p.linear_workspace_bytes(), lib.gemv_workspace_bound(8, p.admission_margin())), p.admission_margin());
        if (pf_seqpar && p.supports_pipeline() && opt.max_rows >= 2 && !seqpar_xch) {
            Lib::admit(xfer::SeqparExchange::workspace_bytes(opt.max_rows), p.admission_margin(), "prefill seqpar channel");
            xch->complete(stream);
            hip_ok(hipStreamSynchronize(stream), "seqpar install");
            seqpar_xch = std::make_unique<xfer::SeqparExchange>(comm->transport(), opt.rank, opt.max_rows, stream, pf_streams);
            std::fprintf(stderr, "glm prefill seqpar: on even chunks=owner-mhc-norm odd/singleton=legacy pipeline=%u\n",
                         opt.pf_pipeline);
        }
    }

    void warm_moe(MoePath& m, std::uint64_t margin) {
        std::uint64_t need = 0;
        for (auto& w : layers)
            if (w.moe) need += m.warm_bytes(w.moew);
        if (need == 0) return;
        hip_ok(hipDeviceSynchronize(), "sync");
        Lib::admit(need, margin, "MoE seam tables");
        for (auto& w : layers)
            if (w.moe) m.warm(w.moew);
    }

    void alloc() {
        const std::size_t R = 8;  // decode rows per call
        const std::size_t RC = std::max<std::size_t>(R, opt.max_rows);  // chunked prefill rows (PrefillPath)
        X = lib.buffer<float>(RC * kHc * kHidden);
        X2 = lib.buffer<float>(RC * kHc * kHidden);
        F = lib.buffer<float>(RC * kHc * kHidden);
        hz = lib.buffer<float>(RC * kHcMix);
        hrr = lib.buffer<float>(RC);
        mix = lib.buffer<float>(RC * kHcMix);
        xc = lib.buffer<float>(RC * kHidden);
        xn = lib.buffer<float>(RC * kHidden);
        part = lib.buffer<float>(RC * kHidden);
        hfin = lib.buffer<float>(RC * kHidden);
        hidden = lib.buffer<float>(RC * kHidden);
        logits = lib.buffer<float>(R * std::max<std::size_t>(vocab_n, kExperts));
        rlog = lib.buffer<float>(RC * kExperts);
        wts = lib.buffer<float>(RC * kTopE);
        ids = lib.buffer<std::uint32_t>(RC * kTopE);
        tok_d = lib.buffer<std::uint32_t>(RC);
        slot_d = lib.buffer<std::uint32_t>(RC);
        pos_d = lib.buffer<std::uint64_t>(RC);
        amv = lib.buffer<float>(R);
        ami = lib.buffer<std::uint32_t>(R);
        flag = lib.state<unsigned>(1);
        if (opt.spec) {
            row_flag = lib.state<unsigned>(R);
            replay_flag = lib.state<unsigned>(1);
            pending.assign(opt.slots, 0);
            cur.assign(opt.slots, 0);
        }
        if (opt.mtp) {
            mtp_flag = lib.state<unsigned>(1);
            trunk_logits = lib.buffer<float>(vocab_n);
            mtp_rows.reserve(RC);
            mtp_slots.reserve(RC);
            mtp_positions.reserve(RC);
            m_e = lib.buffer<float>(RC * kHidden);
            m_en = lib.buffer<float>(RC * kHidden);
            m_hn = lib.buffer<float>(RC * kHidden);
            m_cat = lib.buffer<float>(RC * 2 * kHidden);
            m_z = lib.buffer<float>(RC * kHidden);
            m_r = lib.buffer<float>(RC * kHidden);
            m_y = lib.buffer<float>(RC * kHidden);
            m_h = lib.buffer<float>(RC * kHidden);
            m_hs = lib.buffer<float>(static_cast<std::size_t>(opt.slots) * 8 * kHidden);
            m_mz = lib.buffer<float>(2);
        }
        dg = lib.buffer<float>(R * dense_i);
        du = lib.buffer<float>(R * dense_i);
        da = lib.buffer<float>(R * dense_i);
        no_mtp.assign(opt.slots, 0);
        Scratch& s = *scr;
        const unsigned C = kda_heads * kKdaDim;
        s.qkv = lib.buffer<float>(3 * C);
        s.f = lib.buffer<float>(C);
        s.g = lib.buffer<float>(C);
        s.b = lib.buffer<float>(kda_heads);
        s.t128a = lib.buffer<float>(kKdaDim);
        s.t128b = lib.buffer<float>(kKdaDim);
        s.kda_scr = lib.buffer<float>(static_cast<std::size_t>(kda_heads) * 4 * kKdaDim);
        s.beta = lib.buffer<float>(kda_heads);
        s.o = lib.buffer<float>(std::max<std::size_t>(C, static_cast<std::size_t>(mla_heads) * kMlaDim));
        s.y1 = lib.buffer<float>(C);
        s.qa = lib.buffer<float>(kQLora);
        s.qr = lib.buffer<float>(kQLora);
        s.q = lib.buffer<float>(static_cast<std::size_t>(mla_heads) * kMlaDim);
        s.kva = lib.buffer<float>(kKvLora);
        s.lat = lib.buffer<float>(kKvLora);
        s.qI = lib.buffer<float>(kIdxHeads * kIdxDim);
        s.wh = lib.buffer<float>(kIdxHeads);
        s.cap = opt.capacity;
        s.scores = lib.buffer<float>(opt.capacity / kKpool);
        s.work = lib.buffer<std::uint32_t>(opt.capacity / kKpool);
        s.tokens = lib.buffer<std::uint32_t>(kIdxTopk + kKpool);
        s.ntok = lib.buffer<std::uint32_t>(1);
        s.qt = lib.buffer<float>(static_cast<std::size_t>(mla_heads) * kKvLora);
        s.key = lib.buffer<float>(kIdxDim);
        s.gate = lib.buffer<float>(kIdxDim);
        s.slot_d = lib.buffer<D::DsaSlot>(1);
        s.t_d = lib.buffer<unsigned>(1);
        s.eo = lib.buffer<float>(static_cast<std::size_t>(kTopE) * kHidden);
        s.ge = lib.buffer<float>(expert_i);
        s.ue = lib.buffer<float>(expert_i);
        s.ae = lib.buffer<float>(expert_i);
        s.sh = lib.buffer<float>(kHidden);
        if (!lib.planning()) {  // STRIX_GLM_ALLOC_LOG=1: names of the per-row buffers (the dalloc lines carry only sizes)
            const std::pair<const char*, const void*> named[] = {{"X", X}, {"X2", X2}, {"F", F}, {"mix", mix}, {"xc", xc}, {"xn", xn},
                                                                 {"part", part}, {"logits", logits}, {"tok_d", tok_d}, {"slot_d", slot_d},
                                                                 {"pos_d", pos_d}, {"flag", flag}, {"dg", dg}, {"da", da}};
            for (const auto& [n, q] : named) alloc_log(q, 0, n);
        }
    }

    void check_flag(const char* where) {
        unsigned f = 0;
        hip_ok(hipMemcpyAsync(&f, flag, 4, hipMemcpyDeviceToHost, stream), "flag");
        hip_ok(hipStreamSynchronize(stream), "flag sync");
        if (f) throw std::runtime_error(std::string("non-finite value (checkpoints C1-C5) at ") + where);
    }

    // One forward of R <= 8 rows through layers [0, n_layers); X holds the 4-stream residual afterwards.
    // Post-exchange consumers: the exact fast forms (glm_post_fast, gated bitwise vs Tier A) by
    // default; STRIX_GLM_POST=ref keeps the Tier A kernels (dev::hc_pre / dense / route).
    void hc_pre(const float* x, const float* fn, const float* hc_base, const float* scale, float* f, float* m, float* c, unsigned R) {
        if (r1b) strix::glm::r1b::hc_pre(x, fn, hc_base, scale, f, m, c, R, stream);  // default generated identity (blocked fn chains; prompts use R1)
        else if (post_ref) D::hc_pre(x, fn, hc_base, scale, f, m, c, R, stream);
        else xfer::hc_pre_v3(x, fn, hc_base, scale, f, m, c, R, stream, hc_pre_variant);
    }
    void router_route(const float* x, const MoeLayerWeights& w, unsigned R) {
        router_route_impl(x, w, R);
        if (route_log && route_log_d != nullptr && cur_layer < kLayers + 1)
            hip_ok(hipMemcpyAsync(route_log_d + static_cast<std::size_t>(cur_layer) * kTopE, ids, kTopE * 4, hipMemcpyDeviceToDevice, stream), "route log");
    }
    void router_route_impl(const float* x, const MoeLayerWeights& w, unsigned R) {
        if (r1b) {  // default generated identity: blocked router logits; selection / weights (R5 / R14) unchanged
            const int variant = strix::glm::r1b::dense_variant();
            const bool packed = (variant == 7 || variant == 6 || (variant == 5 && R == 1)) && w.router_packed != nullptr;
            strix::glm::r1b::router_dense(x, packed ? w.router_packed : w.router, rlog, R, stream,
                                         variant == 5 || variant == 6 || variant == 7 ? (packed ? variant : 2) : variant);
            xfer::route_v2(rlog, w.bias, ids, wts, R, stream);
        } else if (post_ref) {
            D::dense(x, w.router, D::WType::F16, kHidden, 0, kExperts, rlog, R, stream);
            D::route(rlog, w.bias, ids, wts, R, stream);
        } else {
            xfer::router_dense(x, w.router, rlog, R, stream, -1);
            xfer::route_v2(rlog, w.bias, ids, wts, R, stream);
        }
    }

    // Sublayer boundary: TP reduce of `part` + R7 expand into X. The exchange's fused form when it has one and the call
    // supports per-row flags when the exchange provides them (STRIX_GLM_BOUNDARY=0 forces the split form).
    // pfc (chunked prefill rows, up to max_rows): always the split form (the fused kernels take decode rows only).
    // `produced`: the partial already written to (and published from) the exchange's TX slot by its producer.
    void boundary(unsigned* c1, unsigned c1_stride, unsigned R, bool pfc = false, const float* produced = nullptr) {
        if (!pfc && boundary_fused && xch->fused_boundary_rows(produced != nullptr ? produced : part, X, mix, X2, c1, R, c1_stride, stream)) {
            mark(segXch);  // the fused kernel: exchange wait + R17 + R7
        } else {
            xch->reduce(part, R, stream);
            mark(segXch);
            D::hc_expand(X, mix, part, X2, c1, R, stream, c1_stride);
            mark(segHc);
        }
        std::swap(X, X2);
    }

    // N=2 whole-layer schedule. Only transport progresses off-stream; every seam and boundary is
    // enqueued on stream. Each half swaps its own residual pointers twice per layer, ending in X.
    void layers_forward_pipeline(std::span<const Row> rows, unsigned n_layers) {
        const unsigned R = static_cast<unsigned>(rows.size()), A = (R + 1) / 2;
        struct Half {
            unsigned offset, count;
            float *x, *out;
            bool normed = false;  // the half's F already holds its next site's norm (STRIX_GLM_PF_HC_FUSE)
        };
        Half a{0, A, X, X2};
        Half b{A, R - A, X + static_cast<std::size_t>(A) * kHc * kHidden,
               X2 + static_cast<std::size_t>(A) * kHc * kHidden};
        const bool hc_fuse = glm::pf::hc_expand_norm_enabled();
        const bool mixnorm = glm::pf::hc_mix_norm_enabled();
        CommExchange* async_xch = nullptr;
        if (const char* v = std::getenv("STRIX_DIST_ASYNC_START"); v && std::string(v) == "1")
            async_xch = dynamic_cast<CommExchange*>(xch.get());
        // Every Comm rank votes (the async start is part of the vote: only its half regions use the staging views).
        if (auto* cx = dynamic_cast<CommExchange*>(xch.get())) cx->agree_staged(stream, async_xch != nullptr);
        auto wait_source = [&](const Half& h) {
            if (async_xch) async_xch->wait_source(h.offset == 0 ? 0U : 1U, stream);
        };
        auto context = [&](const Half& h) {
            return CallCtx{&lib, rows.subspan(h.offset, h.count), slot_d + h.offset, pos_d + h.offset,
                           flag, stream, false, nullptr};
        };
        auto partial = [&](const Half& h) { return part + static_cast<std::size_t>(h.offset) * kHidden; };
        auto pre = [&](Half& h, const HcSite& site, const std::uint16_t* norm) {
            const std::size_t q = h.offset;
            if (mixnorm) {
                if (h.normed) glm::pf::hc_pre_rows_normed(h.x, site.fn, site.base, site.scale, hrr + q,
                                                          hz + q * kHcMix, mix + q * kHcMix, norm, xn + q * kHidden, h.count, stream);
                else glm::pf::hc_pre_rows_v2(h.x, site.fn, site.base, site.scale, F + q * kHc * kHidden,
                                             hz + q * kHcMix, mix + q * kHcMix, norm, xn + q * kHidden, h.count, stream);
            } else {
                if (h.normed) glm::pf::hc_pre_rows_normed(h.x, site.fn, site.base, site.scale, hrr + q,
                                                          hz + q * kHcMix, mix + q * kHcMix, xc + q * kHidden, h.count, stream);
                else glm::pf::hc_pre_rows(h.x, site.fn, site.base, site.scale, F + q * kHc * kHidden,
                                          hz + q * kHcMix, mix + q * kHcMix, xc + q * kHidden, h.count, stream);
                D::GlmNorm::run(xc + q * kHidden, norm, D::WType::BF16, kHidden, kEps, xn + q * kHidden, h.count, stream);
            }
            h.normed = false;
            mark(segHc);
        };
        auto attention = [&](Half& h, LayerW& w) {
            wait_source(h);  // this half's seam may overwrite part; the other half remains independent
            pre(h, w.hc_attn, w.in_norm);
            const CallCtx c = context(h);
            const float* x = xn + static_cast<std::size_t>(h.offset) * kHidden;
            if (w.kda) pf->kda(w.kdaw, w.kst, c, x, partial(h));
            else if (kv_shard) kv_shard->prefill(*pf, w.dsaw, w.dst, c, x, partial(h));
            else pf->dsa(w.dsaw, w.dst, c, x, partial(h));
            mark(w.kda ? segKda : segDsa);
        };
        auto ffn = [&](Half& h, LayerW& w) {
            wait_source(h);
            pre(h, w.hc_ffn, w.post_norm);
            const CallCtx c = context(h);
            const std::size_t q = h.offset;
            const float* x = xn + q * kHidden;
            if (w.moe) {
                float* log = rlog + q * kExperts;
                std::uint32_t* id = ids + q * kTopE;
                float* wt = wts + q * kTopE;
                glm::pf::dense_rows(x, w.moew.router, D::WType::F16, kHidden, 0, kExperts, log, h.count, stream);
                D::route(log, w.moew.bias, id, wt, h.count, stream);
                if (pf->moe_by_row()) {
                    const unsigned B = std::clamp(pf->moe_row_block(), 1U, 8U);
                    for (unsigned r = 0; r < h.count; r += B) {
                        const CallCtx cr{&lib, c.rows.subspan(r, std::min(B, h.count - r)), slot_d + q + r,
                                         pos_d + q + r, flag, stream, false, nullptr};
                        moe->forward(w.moew, cr, x + static_cast<std::size_t>(r) * kHidden,
                                     id + static_cast<std::size_t>(r) * kTopE, wt + static_cast<std::size_t>(r) * kTopE,
                                     partial(h) + static_cast<std::size_t>(r) * kHidden);
                    }
                } else pf->moe(w.moew, c, x, id, wt, partial(h));
            } else pf->dense(w.dense, c, x, dense_i, partial(h));
            mark(w.moe ? segMoe : segDense);
        };
        auto start = [&](const Half& h) {
            if (async_xch) async_xch->start_reduce_region(partial(h), h.count, h.offset, h.offset == 0 ? 0 : 1, stream);
            else xch->start_reduce(partial(h), h.count, stream);
        };
        auto finish = [&] {
            xch->finish_reduce(stream);
            mark(segXch);
        };
        auto expand = [&](Half& h) {
            if (hc_fuse && glm::pf::hc_expand_norm_split()) {  // split control: unfused expand, then rr alone
                D::hc_expand(h.x, mix + static_cast<std::size_t>(h.offset) * kHcMix, partial(h), h.out, flag, h.count, stream);
                glm::pf::hc_norm_factor_rows(h.out, hrr + h.offset, h.count, stream);
                h.normed = true;
            } else if (hc_fuse) {  // also this half's next-site norm factors (its next pre skips hc_norm_rows and F)
                glm::pf::hc_expand_norm_rows(h.x, mix + static_cast<std::size_t>(h.offset) * kHcMix, partial(h), h.out,
                                             hrr + h.offset, flag, h.count, stream);
                h.normed = true;
            } else {
                D::hc_expand(h.x, mix + static_cast<std::size_t>(h.offset) * kHcMix, partial(h), h.out, flag, h.count, stream);
            }
            mark(segHc);
            std::swap(h.x, h.out);
        };
        // STRIX_GLM_PF_SUMEXP (async half regions + the fused expand only): the seam hands both partials to the expand,
        // which sums them on load, and frees RX / the remote region before the next start (one staging RX buffer).
        // Otherwise the exchange sums into part and the next start precedes the expand, as before.
        CommExchange* sum_xch = async_xch != nullptr && glm::pf::hc_sum_expand_enabled() ? async_xch : nullptr;
        auto seam = [&](Half& done, Half* next) {
            if (sum_xch == nullptr) {
                finish();
                if (next) start(*next);
                expand(done);
                return;
            }
            const float *p0 = nullptr, *p1 = nullptr;
            sum_xch->finish_reduce_pair(stream, &p0, &p1);
            mark(segXch);
            const std::size_t q = done.offset;
            // p0 / p1 already point at this half's first row (its part region and its RX / remote region).
            glm::pf::hc_sum_expand_norm_rows(done.x, mix + q * kHcMix, p0, p1, done.out, hrr + q, flag, done.count, stream);
            done.normed = true;
            sum_xch->release_pair(stream);
            mark(segHc);
            std::swap(done.x, done.out);
            if (next) start(*next);
        };
        try {
            for (unsigned li = 0; li < n_layers; ++li) {
                LayerW& w = layers[li];
                cur_layer = li;
                attention(a, w);
                // B's previous FFN may overlap next-layer attention A. Retire before the next start.
                if (li != 0 && b.count != 0) seam(b, nullptr);
                start(a);
                if (b.count != 0) attention(b, w);  // A's recurrence / cache commits precede B on stream
                seam(a, b.count != 0 ? &b : nullptr);
                ffn(a, w);
                if (b.count != 0) seam(b, &a);
                else start(a);
                if (b.count != 0) ffn(b, w);
                seam(a, b.count != 0 ? &b : nullptr);
            }
            if (n_layers != 0 && b.count != 0) seam(b, nullptr);
        } catch (...) {
            // Do not leave a producer / host transfer borrowing row buffers after a seam throws.
            try { xch->complete(stream); } catch (...) {}
            throw;
        }
        check_flag("layers");
    }

    // TP2 SP island: residual/mHC/norm are owner-local, column/head/K-half work
    // still consumes every row. Pipeline halves each contain both owners, so
    // their independently queued RS/AG can overlap the other half's column work.
    void layers_forward_seqpar(std::span<const Row> rows, unsigned n_layers) {
        const unsigned R = static_cast<unsigned>(rows.size());
        const unsigned tiles = xfer::seqpar_tiles(R, opt.pf_pipeline);
        const unsigned block = opt.pf_pipeline == 2 ? R / 2U : R;
        std::array<float*, 2> tile_x{X, X}, tile_out{X2, X2};
        if (!pf->supports_pipeline() || layer_dump != nullptr || null_layer >= 0)
            throw std::logic_error("glm seqpar: requires row-invariant chunked prefill without diagnostic mutation hooks");
        // No two progress owners may borrow the transport at once. CRC/seal and
        // device consumers retire before entry and before the unchanged head.
        xch->complete(stream);
        hip_ok(hipStreamSynchronize(stream), "seqpar entry");
        if (!seqpar_xch) throw std::logic_error("glm seqpar: channel must be admitted by set_prefill before execution");
        const bool fuse = opt.pf_pipeline == 2 && glm::pf::hc_expand_norm_enabled();
        float* const sp_xn = seqpar_xch->normalized(); // coherent direct AG consumer
        const bool mixnorm = glm::pf::hc_mix_norm_enabled();
        std::array<bool, 2> normed{};
        auto pre = [&](unsigned i, const HcSite& site, const std::uint16_t* norm) {
            const auto half_stream = seqpar_xch->compute(i);
            const auto tile = xfer::seqpar_tile(R, opt.pf_pipeline, i);
            const auto own = xfer::seqpar_owner(tile, opt.rank);
            const std::size_t q = own.first, hidden_offset = own.elements(), residual_offset = own.elements(kHc * kHidden);
            float* local = tile_x[i] + residual_offset;
            if (normed[i]) {
                if (mixnorm)
                    glm::pf::hc_pre_rows_normed(local, site.fn, site.base, site.scale, hrr + q, hz + q * kHcMix,
                                                mix + q * kHcMix, norm, sp_xn + hidden_offset, own.count, half_stream);
                else
                    glm::pf::hc_pre_rows_normed(local, site.fn, site.base, site.scale, hrr + q, hz + q * kHcMix,
                                                mix + q * kHcMix, xc + hidden_offset, own.count, half_stream);
            } else if (mixnorm) {
                glm::pf::hc_pre_rows_v2(local, site.fn, site.base, site.scale, F + residual_offset, hz + q * kHcMix,
                                       mix + q * kHcMix, norm, sp_xn + hidden_offset, own.count, half_stream);
            } else {
                glm::pf::hc_pre_rows(local, site.fn, site.base, site.scale, F + residual_offset, hz + q * kHcMix,
                                    mix + q * kHcMix, xc + hidden_offset, own.count, half_stream);
            }
            if (!mixnorm)
                D::GlmNorm::run(xc + hidden_offset, norm, D::WType::BF16, kHidden, kEps, sp_xn + hidden_offset, own.count, half_stream);
            normed[i] = false;
            // Restore all rows before KDA recurrence, MLA/KV writes, router or
            // any TP column projection. Absolute positions/cache layouts stay intact.
            seqpar_xch->start_gather(sp_xn, tile, i, half_stream);
        };
        auto post = [&](unsigned i) {
            const auto half_stream = seqpar_xch->compute(i);
            const float* peer = seqpar_xch->finish_reduce_scatter(i, half_stream);
            const auto own = xfer::seqpar_owner(xfer::seqpar_tile(R, opt.pf_pipeline, i), opt.rank);
            const std::size_t q = own.first, hidden_offset = own.elements(), residual_offset = own.elements(kHc * kHidden);
            if (fuse && !glm::pf::hc_expand_norm_split()) {
                const float* local = part + hidden_offset;
                // Reuse the existing mHC input path: rank-ordered FP32 sum on
                // load, followed by the unchanged expand/norm expressions.
                glm::pf::hc_sum_expand_norm_rows(tile_x[i] + residual_offset, mix + q * kHcMix,
                                                 opt.rank == 0 ? local : peer, opt.rank == 0 ? peer : local,
                                                 tile_out[i] + residual_offset, hrr + q, flag, own.count, half_stream);
            } else {
                xfer::post_expand(tile_x[i] + residual_offset, mix + q * kHcMix, part + hidden_offset,
                                  peer, xfer::RxView{}, opt.rank, tile_out[i] + residual_offset, flag, own.count, half_stream);
                if (fuse) glm::pf::hc_norm_factor_rows(tile_out[i] + residual_offset, hrr + q, own.count, half_stream);
            }
            seqpar_xch->release_reduce_scatter(i, half_stream);
            normed[i] = fuse;
            std::swap(tile_x[i], tile_out[i]);
        };
        auto column_range = [&](unsigned i, LayerW& w, bool ffn, xfer::SeqparRows tile, unsigned piece_rows) {
            const auto half_stream = seqpar_xch->compute(i);
            seqpar_xch->column_begin(i);
            for (unsigned first = tile.first; first < tile.first + tile.count; first += piece_rows) {
                const unsigned count = std::min(piece_rows, tile.first + tile.count - first);
                const CallCtx c{&lib, rows.subspan(first, count), slot_d + first, pos_d + first, flag, half_stream, false, nullptr};
                const auto off = std::size_t{first} * kHidden;
                const float* x = sp_xn + off;
                if (!ffn) {
                    if (w.kda) pf->kda(w.kdaw, w.kst, c, x, part + off);
                    else pf->dsa(w.dsaw, w.dst, c, x, part + off);
                } else if (w.moe) {
                    auto* id = ids + std::size_t{first} * kTopE;
                    auto* wt = wts + std::size_t{first} * kTopE;
                    auto* log = rlog + std::size_t{first} * kExperts;
                    // Router and matrix-specific MoE input transforms stay replicated.
                    glm::pf::dense_rows(x, w.moew.router, D::WType::F16, kHidden, 0, kExperts, log, count, half_stream);
                    D::route(log, w.moew.bias, id, wt, count, half_stream);
                    if (pf->moe_by_row()) {
                        const unsigned B = std::clamp(pf->moe_row_block(), 1U, 8U);
                        for (unsigned r = 0; r < count; r += B) {
                            const CallCtx cr{&lib, c.rows.subspan(r, std::min(B, count - r)), slot_d + first + r,
                                             pos_d + first + r, flag, half_stream, false, nullptr};
                            moe->forward(w.moew, cr, x + std::size_t{r} * kHidden, id + std::size_t{r} * kTopE,
                                         wt + std::size_t{r} * kTopE, part + off + std::size_t{r} * kHidden);
                        }
                    } else pf->moe(w.moew, c, x, id, wt, part + off);
                } else pf->dense(w.dense, c, x, dense_i, part + off);
            }
            seqpar_xch->column_end(i);
        };
        auto column = [&](unsigned i, LayerW& w, bool ffn) {
            column_range(i, w, ffn, xfer::seqpar_tile(R, opt.pf_pipeline, i), block);
        };
        auto consume_column = [&](unsigned i, LayerW& w, bool ffn) {
            seqpar_xch->finish(i, seqpar_xch->compute(i));  // only this half's xn must be present
            column(i, w, ffn);
        };
        auto start_rs = [&](unsigned i) {
            seqpar_xch->start_reduce_scatter(part, xfer::seqpar_tile(R, opt.pf_pipeline, i), i, seqpar_xch->compute(i));
        };
        try {
            for (unsigned li = 0; li < n_layers; ++li) {
                LayerW& w = layers[li];
                cur_layer = li;
                if (pf_streams == 1) {
                    xfer::seqpar_single_layer(tiles, [&](xfer::SeqparStep step, unsigned i) {
                        switch (step) {
                        case xfer::SeqparStep::pre_attn: pre(i, w.hc_attn, w.in_norm); break;
                        case xfer::SeqparStep::attn: consume_column(i, w, false); break;
                        case xfer::SeqparStep::reduce_attn:
                        case xfer::SeqparStep::reduce_ffn: start_rs(i); break;
                        case xfer::SeqparStep::post_attn:
                        case xfer::SeqparStep::post_ffn: post(i); break;
                        case xfer::SeqparStep::pre_ffn: pre(i, w.hc_ffn, w.post_norm); break;
                        case xfer::SeqparStep::ffn:
                            // Both normalized halves must arrive before the one grouped call.
                            // Attention/dense call sizes and recurrence order stay legacy.
                            for (unsigned lane = 0; lane < tiles; ++lane)
                                seqpar_xch->finish(lane, seqpar_xch->compute(lane));
                            if (w.moe)
                                column_range(0, w, true, {0, R}, xfer::seqpar_moe_piece(R, 0).count);
                            else
                                for (unsigned lane = 0; lane < tiles; ++lane) column(lane, w, true);
                            break;
                        }
                    });
                    continue;
                }
                pre(0, w.hc_attn, w.in_norm);
                if (tiles == 1) {
                    consume_column(0, w, false);
                    start_rs(0);
                    post(0);
                    pre(0, w.hc_ffn, w.post_norm);
                    consume_column(0, w, true);
                    start_rs(0);
                    post(0);
                    continue;
                }
                if (li == 0) pre(1, w.hc_attn, w.in_norm);
                else start_rs(1); // previous FFN B; AG A precedes it on the wire
                consume_column(0, w, false);
                if (li != 0) {
                    post(1);
                    pre(1, w.hc_attn, w.in_norm);
                }
                start_rs(0); // attention A overlaps attention B
                consume_column(1, w, false);
                post(0);
                pre(0, w.hc_ffn, w.post_norm);
                start_rs(1); // attention B overlaps FFN A, after its AG
                consume_column(0, w, true);
                post(1);
                pre(1, w.hc_ffn, w.post_norm);
                start_rs(0); // FFN A overlaps FFN B, after its AG
                consume_column(1, w, true);
                post(0);
                // Defer B's RS until AFTER next-layer AG A, so that a side
                // stream RX wait cannot sit in front of A's next column call.
            }
            if (pf_streams == 2 && tiles == 2 && n_layers != 0) {
                start_rs(1);
                post(1);
            }
            seqpar_xch->join();
            seqpar_xch->materialize_last(X, R);
            seqpar_xch->complete();
            hip_ok(hipStreamSynchronize(stream), "seqpar exit");
        } catch (...) {
            try { seqpar_xch->complete(); } catch (...) {}
            throw;
        }
        // Legacy segment events are single-stream intervals. Attribute this
        // multi-queue island as one exchange interval; use a queue trace for
        // per-kernel/overlap accounting, never sum cross-stream intervals.
        mark(segXch);
        check_flag("seqpar layers");
    }

    // sp != nullptr: speculative verify rows (one slot): checkpoints go to sp->row_flag[r], nothing throws here.
    // pfc: a chunk of a chunked prefill (rows of one slot, any count <= max_rows): attention, MoE and the dense FFN go
    // through the PrefillPath seam; the row-batched Tier A sites use pf::hc_pre_rows / pf::dense_rows (== decode bits).
    void layers_forward(std::span<const Row> rows, unsigned n_layers, ForwardKind kind, const SpecRows* sp = nullptr) {
        const bool pfc = kind == ForwardKind::Prefill;
        const auto R = static_cast<unsigned>(rows.size());
        const bool seqpar = pfc && pf_seqpar && xfer::seqpar_admits(R);
        if (pfc && pf_seqpar && !seqpar)
            std::fprintf(stderr, "glm prefill seqpar: rows=%u declined (requires even rows >= 2); legacy path\n", R);
        if (pfc && opt.pf_pipeline == 2) {
            if (sp != nullptr || layer_dump != nullptr || null_layer >= 0)
                throw std::logic_error("prefill pipeline 2 does not support spec / layer_dump / null_layer hooks");
            if (!pf->supports_pipeline()) throw std::logic_error("prefill pipeline 2 needs an exact half-row PrefillPath");
            if (rows.empty()) throw std::invalid_argument("prefill pipeline 2 needs nonempty rows");
            for (unsigned r = 1; r < R; ++r)
                if (rows[r].slot != rows[0].slot || rows[r].pos != rows[0].pos + r)
                    throw std::invalid_argument("prefill pipeline 2 needs consecutive positions of one slot");
        }
        std::vector<std::uint32_t> tk(R), sl(R);
        std::vector<std::uint64_t> ps(R);
        for (unsigned r = 0; r < R; ++r) {
            tk[r] = rows[r].token;
            sl[r] = rows[r].slot;
            ps[r] = rows[r].pos;
            if (rows[r].slot >= opt.slots || rows[r].pos >= opt.capacity) throw std::out_of_range("row slot / position");
            if (rows[r].token >= kVocab) throw std::out_of_range("token id");
            // Rows of one slot: consecutive ascending positions in row order (the seams append before they read).
            for (unsigned q = r; q-- > 0;)
                if (rows[q].slot == rows[r].slot) {
                    if (rows[r].pos != rows[q].pos + 1) throw std::invalid_argument("rows of one slot need consecutive ascending positions");
                    break;
                }
        }
        hip_ok(hipMemcpyAsync(tok_d, tk.data(), R * 4, hipMemcpyHostToDevice, stream), "tokens");
        hip_ok(hipMemcpyAsync(slot_d, sl.data(), R * 4, hipMemcpyHostToDevice, stream), "slots");
        hip_ok(hipMemcpyAsync(pos_d, ps.data(), R * 8, hipMemcpyHostToDevice, stream), "pos");
        seg_n = 0;  // only step() collects; a verify / gate forward's marks are dropped here
        mark(segStart);
        if (seqpar) {
            for (unsigned i = 0; i < xfer::seqpar_tiles(R, opt.pf_pipeline); ++i) {
                const auto embed_rows = xfer::seqpar_owner(xfer::seqpar_tile(R, opt.pf_pipeline, i), opt.rank);
                embed4_kernel<<<dim3(kHidden / 256, embed_rows.count), 256, 0, stream>>>(
                    embed, tok_d + embed_rows.first, X + embed_rows.elements(kHc * kHidden));
            }
        } else {
            embed4_kernel<<<dim3(kHidden / 256, R), 256, 0, stream>>>(embed, tok_d, X);
        }
        hip_ok(hipGetLastError(), "embed");
        mark(segEmbed);
        if (!pfc && R > 8) throw std::invalid_argument("layers_forward: decode calls take 1..8 rows");
        if (R > std::max<unsigned>(8U, opt.max_rows)) throw std::invalid_argument("layers_forward: rows > max_rows");
        if (pfc && r1b) throw std::logic_error("r1b-all needs the reference prefill; the chunked prefill implements R1 prompts");
        select_exchange(kind, R);
        if (seqpar) {
            layers_forward_seqpar(rows, n_layers);
            return;
        }
        if (pfc && opt.pf_pipeline == 2) {
            layers_forward_pipeline(rows, n_layers);
            return;
        }
        const CallCtx c{&lib, rows, slot_d, pos_d, flag, stream, !pfc, sp};
        unsigned* const c1 = sp ? sp->row_flag : flag;  // C1 per row in spec mode
        const unsigned c1_stride = sp ? 1U : 0U;
        const bool mixnorm = pfc && glm::pf::hc_mix_norm_enabled();
        for (unsigned li = 0; li < n_layers; ++li) {
            LayerW& w = layers[li];
            cur_layer = li;
            const float* attn_produced = nullptr;  // the attention partial when its producer wrote the TX slot
            if (!pfc && r1b && strix::glm::r1b::hc_norm_enabled()) {
                strix::glm::r1b::hc_pre_norm(X, w.hc_attn.fn, w.hc_attn.base, w.hc_attn.scale, F, mix, xc,
                                          w.in_norm, xn, R, stream);
            } else if (mixnorm) {
                glm::pf::hc_pre_rows_v2(X, w.hc_attn.fn, w.hc_attn.base, w.hc_attn.scale, F, hz, mix, w.in_norm, xn, R, stream);
            } else {
                if (pfc) glm::pf::hc_pre_rows(X, w.hc_attn.fn, w.hc_attn.base, w.hc_attn.scale, F, hz, mix, xc, R, stream);
                else hc_pre(X, w.hc_attn.fn, w.hc_attn.base, w.hc_attn.scale, F, mix, xc, R);
                D::GlmNorm::run(xc, w.in_norm, D::WType::BF16, kHidden, kEps, xn, R, stream);
            }
            mark(segHc);
            if (pfc) {
                if (w.kda) pf->kda(w.kdaw, w.kst, c, xn, part);
                else if (kv_shard) kv_shard->prefill(*pf, w.dsaw, w.dst, c, xn, part);
                else pf->dsa(w.dsaw, w.dst, c, xn, part);
            } else {
                CallCtx ca = c;
                float* out = part;
                if (boundary_fused && xb_publish_on() && attn->publishes()) {  // the o projection writes and publishes the TX slot
                    const auto slot = xch->producer_slot(R, stream);
                    if (slot.data != nullptr) {
                        ca.publish = slot.publish;
                        out = slot.data;
                        attn_produced = slot.data;
                        ++xb_publish_count;
                    }
                }
                if (w.kda) attn->kda(w.kdaw, w.kst, ca, xn, out);
                else attn->dsa(w.dsaw, w.dst, ca, xn, out);
            }
            mark(w.kda ? segKda : segDsa);
            boundary(c1, c1_stride, R, pfc, attn_produced);
            if (!pfc && r1b && strix::glm::r1b::hc_norm_enabled()) {
                strix::glm::r1b::hc_pre_norm(X, w.hc_ffn.fn, w.hc_ffn.base, w.hc_ffn.scale, F, mix, xc,
                                          w.post_norm, xn, R, stream);
            } else if (mixnorm) {
                glm::pf::hc_pre_rows_v2(X, w.hc_ffn.fn, w.hc_ffn.base, w.hc_ffn.scale, F, hz, mix, w.post_norm, xn, R, stream);
            } else {
                if (pfc) glm::pf::hc_pre_rows(X, w.hc_ffn.fn, w.hc_ffn.base, w.hc_ffn.scale, F, hz, mix, xc, R, stream);
                else hc_pre(X, w.hc_ffn.fn, w.hc_ffn.base, w.hc_ffn.scale, F, mix, xc, R);
                D::GlmNorm::run(xc, w.post_norm, D::WType::BF16, kHidden, kEps, xn, R, stream);
            }
            mark(segHc);
            const float* produced = nullptr;  // the FFN partial when its producer wrote the TX slot
            if (w.moe) {
                if (pfc) {  // chunk rows: the row-tiled router chain (== decode bits) + R14 route
                    glm::pf::dense_rows(xn, w.moew.router, D::WType::F16, kHidden, 0, kExperts, rlog, R, stream);
                    D::route(rlog, w.moew.bias, ids, wts, R, stream);
                    if (pf->moe_by_row()) {  // the engine's MoE seam, blocks of <= 8 rows: the decode identity (batch invariant)
                        const unsigned B = std::clamp(pf->moe_row_block(), 1U, 8U);
                        for (unsigned q = 0; q < R; q += B) {
                            const CallCtx cq{&lib, c.rows.subspan(q, std::min(B, R - q)), slot_d + q, pos_d + q, flag, stream, false, nullptr};
                            moe->forward(w.moew, cq, xn + static_cast<std::size_t>(q) * kHidden, ids + static_cast<std::size_t>(q) * kTopE,
                                         wts + static_cast<std::size_t>(q) * kTopE, part + static_cast<std::size_t>(q) * kHidden);
                        }
                    } else {
                        pf->moe(w.moew, c, xn, ids, wts, part);
                    }
                } else {
                    router_route(xn, w.moew, R);
                    if (boundary_fused && xb_publish_on() && moe->publishes()) {
                        // The down kernel writes the partial into the TX slot and publishes it (pair/producer_publish).
                        const auto slot = xch->producer_slot(R, stream);
                        if (slot.data != nullptr) {
                            CallCtx cp = c;
                            cp.publish = slot.publish;
                            moe->forward(w.moew, cp, xn, ids, wts, slot.data);
                            produced = slot.data;
                            ++xb_publish_count;
                        }
                    }
                    if (produced == nullptr) moe->forward(w.moew, c, xn, ids, wts, part);
                }
            } else if (pfc) {
                pf->dense(w.dense, c, xn, dense_i, part);
            } else {
                lib.run(w.dense.gate, xn, dg, R, stream);
                lib.run(w.dense.up, xn, du, R, stream);
                D::swiglu(dg, du, da, static_cast<std::size_t>(R) * dense_i, stream);
                lib.run(w.dense.down, da, part, R, stream);
            }
            mark(w.moe ? segMoe : segDense);
            boundary(c1, c1_stride, R, pfc, produced);
            if (layer_dump != nullptr)
                hip_ok(hipMemcpyAsync(layer_dump->data() + static_cast<std::size_t>(li) * kHc * kHidden, X, kHc * kHidden * 4,
                                      hipMemcpyDeviceToHost, stream), "layer dump");
            if (null_layer == static_cast<int>(li))  // test hook: +1 ulp on stream 0 of the row at null_pos
                for (unsigned q = 0; q < R; ++q)
                    if (rows[q].pos == null_pos) {
                        float* at = X + (static_cast<std::size_t>(q) * kHc) * kHidden + null_elem;
                        std::uint32_t bits = 0;
                        hip_ok(hipMemcpyAsync(&bits, at, 4, hipMemcpyDeviceToHost, stream), "null flip read");
                        hip_ok(hipStreamSynchronize(stream), "null flip sync");
                        bits += 1;
                        hip_ok(hipMemcpyAsync(at, &bits, 4, hipMemcpyHostToDevice, stream), "null flip write");
                        hip_ok(hipStreamSynchronize(stream), "null flip sync");
                    }
        }
        // Plain steps check the layers' checkpoints together with the head's (step), without a host round trip mid-step:
        // the flag is sticky, so a layer checkpoint still throws before the argmax exchange and before any token is
        // returned. Cross-rank symmetry is as before (each rank throws on its own flag).
        if (sp == nullptr && !(kind == ForwardKind::Plain && defer_layer_flag)) check_flag("layers");
    }

    std::vector<std::uint32_t> step(std::span<const Row> rows) {
        if (kv_shard && (rows.size() != 1 || rows[0].slot != 0))
            throw std::invalid_argument("kvshard: one committed decode row in slot zero required");
        if (load_test) throw std::logic_error("STRIX_GLM_LOAD_TEST: load only");
        if (rows.empty() || rows.size() > 8) throw std::invalid_argument("step: 1..8 rows");
        if (opt.layers != kLayers && !smoke_head) throw std::logic_error("step needs all 45 layers (G3 uses gate_residual)");
        const auto R = static_cast<unsigned>(rows.size());
        if (opt.spec)  // mode change spec -> plain: apply the slots' pending KDA rows first (design notes section 1)
            for (const Row& r : rows)
                if (pending.at(r.slot) > 0) flush(r.slot);
        layers_forward(rows, opt.layers, ForwardKind::Plain);
        D::hc_final(X, hfin, R, stream);
        D::GlmNorm::run(hfin, final_norm, D::WType::BF16, kHidden, kEps, hidden, R, stream);
        lib.run(head, hidden, logits, R, stream);
        D::argmax(logits, vocab_n, vocab0, amv, ami, flag, R, stream);
        check_flag(defer_layer_flag ? "layers / head" : "head");
        xch->argmax(amv, ami, R, stream);
        mark(segHead);
        std::vector<std::uint32_t> out(R);
        hip_ok(hipMemcpyAsync(out.data(), ami, R * 4, hipMemcpyDeviceToHost, stream), "tokens out");
        hip_ok(hipStreamSynchronize(stream), "step sync");
        collect();
        ++prof.calls;
        return out;
    }

    // Per-layer residuals of row 0 (identity comparisons, one load): out[t][l][4][4096] after every layer l < n_layers.
    std::vector<float>* layer_dump = nullptr;
    std::vector<float> gate_residual_layers(std::span<const std::uint32_t> tokens, unsigned n_layers) {
        if (n_layers > opt.layers) throw std::invalid_argument("gate: more layers than loaded");
        const std::size_t per = static_cast<std::size_t>(n_layers) * kHc * kHidden;
        std::vector<float> out(tokens.size() * per), one(per);
        layer_dump = &one;
        for (std::size_t t = 0; t < tokens.size(); ++t) {
            const Row r{0, tokens[t], t};
            layers_forward({&r, 1}, n_layers, ForwardKind::Gate);
            hip_ok(hipStreamSynchronize(stream), "gate sync");
            xch->complete(stream);
            std::copy(one.begin(), one.end(), out.begin() + static_cast<std::ptrdiff_t>(t * per));
        }
        layer_dump = nullptr;
        return out;
    }
    std::vector<float> gate_residual(std::span<const std::uint32_t> tokens, unsigned n_layers) {
        if (n_layers > opt.layers) throw std::invalid_argument("gate: more layers than loaded");
        std::vector<float> out(tokens.size() * kHc * kHidden);
        for (std::size_t t = 0; t < tokens.size(); ++t) {
            const Row r{0, tokens[t], t};
            layers_forward({&r, 1}, n_layers, ForwardKind::Gate);
            hip_ok(hipMemcpyAsync(out.data() + t * kHc * kHidden, X, kHc * kHidden * 4, hipMemcpyDeviceToHost, stream), "residual");
            hip_ok(hipStreamSynchronize(stream), "gate sync");
            xch->complete(stream);  // every exchange of this forward verified before its residual is trusted
        }
        return out;
    }

    bool reference_prefill_installed() const { return dynamic_cast<const ReferencePrefill*>(pf.get()) != nullptr; }
    // Chunked prefill through the PrefillPath (prefill): chunks of max_rows rows; head on the last row only.
    unsigned score_calls_ = 0;  // prefill calls of this process (score dump file index)
    std::uint32_t prefill_chunked(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::uint64_t pos0) {
        if (tokens.empty()) throw std::invalid_argument("prefill: no tokens");
        ++score_calls_;
        if (opt.layers != kLayers && !smoke_head) throw std::logic_error("prefill needs all 45 layers (G3 uses gate_residual)");
        if (slot >= opt.slots) throw std::out_of_range("prefill slot");
        if (opt.spec && pending.at(slot) > 0) flush(slot);
        const std::size_t C = std::max<std::size_t>(1, opt.max_rows);
        no_mtp.at(slot) = 1;  // before the first chunk mutates the slot (an exception mid-prefill keeps MTP refused)
        std::vector<Row> rows;
        for (std::size_t c0 = 0; c0 < tokens.size(); c0 += C) {
            const std::size_t n = std::min(C, tokens.size() - c0);
            rows.resize(n);
            for (std::size_t i = 0; i < n; ++i) rows[i] = Row{slot, tokens[c0 + i], pos0 + c0 + i};
            layers_forward(rows, opt.layers, ForwardKind::Prefill);
        }
        const unsigned last = static_cast<unsigned>(rows.size() - 1);
        D::hc_final(X + static_cast<std::size_t>(last) * kHc * kHidden, hfin, 1, stream);
        D::GlmNorm::run(hfin, final_norm, D::WType::BF16, kHidden, kEps, hidden, 1, stream);
        lib.run(head, hidden, logits, 1, stream);
        D::argmax(logits, vocab_n, vocab0, amv, ami, flag, 1, stream);
        check_flag("head");
        xch->argmax(amv, ami, 1, stream);
        std::uint32_t out = 0;
        hip_ok(hipMemcpyAsync(&out, ami, 4, hipMemcpyDeviceToHost, stream), "token out");
        hip_ok(hipStreamSynchronize(stream), "prefill sync");
        // Quality harness (any identity): STRIX_GLM_PF_SCORE_LAST=N STRIX_GLM_PF_SCORE_OUT=PREFIX writes the head logits of the
        // last N rows of the final chunk (real-text teacher forcing through the prefill path) to PREFIX-r<rank>.f32 [N][slice].
        // Row i is prompt position T-N+i (T = prompt length): it predicts token T-N+i+1 (the last row: the first new token).
        // Sequence-parallel prefill leaves each rank only its own rows of X (plus the materialized last row), so the final
        // hidden rows are normalized by their owner and all-gathered first (as prefill_mtp does); else every rank's vocab
        // slice would score the peer's rows from stale residuals.
        if (const char* sn = std::getenv("STRIX_GLM_PF_SCORE_LAST"); sn != nullptr && *sn != '\0') {
            const char* so = std::getenv("STRIX_GLM_PF_SCORE_OUT");
            if (so == nullptr || *so == '\0') throw std::invalid_argument("STRIX_GLM_PF_SCORE_LAST needs STRIX_GLM_PF_SCORE_OUT");
            const std::size_t n = std::min<std::size_t>(std::stoul(sn), rows.size());
            const auto R = static_cast<unsigned>(rows.size());
            const float* gathered = nullptr;
            if (pf_seqpar && xfer::seqpar_admits(R)) {
                if (!seqpar_xch) throw std::logic_error("glm seqpar: score dump needs the seqpar channel");
                float* const g = seqpar_xch->normalized();
                const unsigned tiles = xfer::seqpar_tiles(R, opt.pf_pipeline);
                for (unsigned i = 0; i < tiles; ++i) {
                    const auto tile = xfer::seqpar_tile(R, opt.pf_pipeline, i);
                    const auto own = xfer::seqpar_owner(tile, opt.rank);
                    D::hc_final(X + own.elements(kHc * kHidden), hfin + own.elements(), own.count, stream);
                    D::GlmNorm::run(hfin + own.elements(), final_norm, D::WType::BF16, kHidden, kEps, g + own.elements(), own.count, stream);
                    seqpar_xch->start_gather(g, tile, i, stream);
                }
                for (unsigned i = 0; i < tiles; ++i) seqpar_xch->finish(i, stream);
                seqpar_xch->complete();
                gathered = g;
            }
            std::vector<float> all(n * vocab_n);
            // STRIX_GLM_PF_SCORE_BLOCK=1..8 rows per head call (default 1; the head GEMV reads the whole head per call).
            const char* sbv = std::getenv("STRIX_GLM_PF_SCORE_BLOCK");
            const std::size_t sb = sbv != nullptr && *sbv != '\0' ? std::stoul(sbv) : 1U;
            if (sb < 1U || sb > 8U) throw std::invalid_argument("STRIX_GLM_PF_SCORE_BLOCK must be 1..8");
            for (std::size_t i = 0; i < n; i += sb) {
                const std::size_t row = rows.size() - n + i, b = std::min(sb, n - i);
                const auto bu = static_cast<unsigned>(b);
                if (gathered != nullptr) {  // the head reads the engine's hidden buffer (as prefill_mtp hands the rows over)
                    hip_ok(hipMemcpyAsync(hidden, gathered + row * kHidden, b * kHidden * sizeof(float), hipMemcpyDeviceToDevice, stream),
                           "score hidden row");
                    lib.run(head, hidden, logits, bu, stream);
                } else {
                    D::hc_final(X + row * kHc * kHidden, hfin, bu, stream);
                    D::GlmNorm::run(hfin, final_norm, D::WType::BF16, kHidden, kEps, hidden, bu, stream);
                    lib.run(head, hidden, logits, bu, stream);
                }
                hip_ok(hipMemcpyAsync(all.data() + i * vocab_n, logits, b * vocab_n * sizeof(float), hipMemcpyDeviceToHost, stream),
                       "score logits");
            }
            hip_ok(hipStreamSynchronize(stream), "score sync");
            // STRIX_GLM_PF_SCORE_SEQ=1: each prefill call keeps its own file, PREFIX-c<call>-r<rank>.f32 (call = prefill
            // calls of this process, 0-based), so one process scores a list of prompts.
            const char* sq = std::getenv("STRIX_GLM_PF_SCORE_SEQ");
            const bool seq = sq != nullptr && std::string(sq) == "1";
            const std::string f = std::string(so) + (seq ? "-c" + std::to_string(score_calls_ - 1U) : std::string()) + "-r" +
                                  std::to_string(opt.rank) + ".f32";
            std::ofstream score_file(f, std::ios::binary | std::ios::trunc);
            score_file.write(reinterpret_cast<const char*>(all.data()), static_cast<std::streamsize>(all.size() * sizeof(float)));
            std::printf("rank %u prefill score: last %zu rows of %zu (%s), vocab slice %zu -> %s\n", opt.rank, n, rows.size(),
                        gathered != nullptr ? "seqpar rows gathered" : "local rows", static_cast<std::size_t>(vocab_n), f.c_str());
        }
        ++prof.calls;
        return out;
    }

    Engine::MtpPrefillOut prefill_mtp(std::uint32_t slot, std::span<const std::uint32_t> tokens,
                                     std::uint64_t pos0, Exchange::MergeCtl& ctl) {
        if (load_test) throw std::logic_error("STRIX_GLM_LOAD_TEST: load only");
        if (!opt.mtp || layers.empty() || !layers.back().mtp || opt.layers != kLayers || final_norm == nullptr)
            throw std::logic_error("prefill_mtp needs all 45 trunk layers and MTP");
        if (!pf->exact_linears() || reference_prefill_installed())
            throw std::logic_error("prefill_mtp requires an exact chunked PrefillPath (gemv)");
        if (tokens.empty()) throw std::invalid_argument("prefill_mtp: no tokens");
        if (slot >= opt.slots || pos0 > opt.capacity || tokens.size() > opt.capacity - pos0)
            throw std::out_of_range("prefill_mtp: slot / capacity");
        for (auto t : tokens)
            if (t >= kVocab) throw std::out_of_range("token id");
        if (opt.spec && pending.at(slot) > 0) flush(slot);
        no_mtp.at(slot) = 1;  // stays poisoned until the final single-row seed also succeeds
        const std::size_t C = std::max<std::size_t>(1, opt.max_rows);
        std::vector<Row> rows;
        unsigned last = 0, interior_flag = 0;
        for (std::size_t c0 = 0; c0 < tokens.size(); c0 += C) {
            const auto R = static_cast<unsigned>(std::min(C, tokens.size() - c0));
            rows.resize(R);
            for (unsigned j = 0; j < R; ++j) rows[j] = Row{slot, tokens[c0 + j], pos0 + c0 + j};
            layers_forward(rows, opt.layers, ForwardKind::Prefill);
            if (pf_seqpar && xfer::seqpar_admits(R)) {
                if (!seqpar_xch) throw std::logic_error("glm seqpar: channel must be admitted before final gather");
                // MTP teacher forcing consumes all trunk hidden rows. Residual
                // streams remain local; gather only the final normalized hidden.
                float* const gathered = seqpar_xch->normalized();
                const unsigned tiles = xfer::seqpar_tiles(R, opt.pf_pipeline);
                for (unsigned i = 0; i < tiles; ++i) {
                    const auto tile = xfer::seqpar_tile(R, opt.pf_pipeline, i);
                    const auto own = xfer::seqpar_owner(tile, opt.rank);
                    D::hc_final(X + own.elements(kHc * kHidden), hfin + own.elements(), own.count, stream);
                    D::GlmNorm::run(hfin + own.elements(), final_norm, D::WType::BF16, kHidden, kEps,
                                   gathered + own.elements(), own.count, stream);
                    seqpar_xch->start_gather(gathered, tile, i, stream);
                }
                for (unsigned i = 0; i < tiles; ++i) seqpar_xch->finish(i, stream);
                seqpar_xch->complete();
                // One chunk-boundary handoff into the existing MTP buffer;
                // per-sublayer AG has no staging/copy kernel.
                hip_ok(hipMemcpyAsync(hidden, gathered, xfer::GlmSchedule::row_bytes(R), hipMemcpyDeviceToDevice, stream),
                       "seqpar MTP hidden copy");
                hip_ok(hipStreamSynchronize(stream), "seqpar MTP hidden handoff");
            } else {
                D::hc_final(X, hfin, R, stream);
                D::GlmNorm::run(hfin, final_norm, D::WType::BF16, kHidden, kEps, hidden, R, stream);
            }
            // Like the plain chunked prefill, interior rows run no vocabulary head (trunk or MTP): their ids are
            // discarded and drafts are always verified, so only the final row's head (y0) and the seed drafter
            // below need logits. Per-row vocabulary heads cost ~15 % of the prefill at 8K/32K.
            // Look ahead in the full prompt: a chunk's last hidden pairs with the next chunk's first token.
            // Only the final prompt row waits for the trunk head's y0.
            const auto pairs = static_cast<unsigned>(std::min<std::size_t>(R, tokens.size() - c0 - 1));
            if (pairs != 0) {
                mtp_cache(slot, hidden, tokens.subspan(c0 + 1, pairs), pos0 + c0, true);
                unsigned f = 0;
                hip_ok(hipGetLastError(), "MTP prefill kernels");
                hip_ok(hipMemcpyAsync(&f, mtp_flag, 4, hipMemcpyDeviceToHost, stream), "MTP prefill flag");
                hip_ok(hipStreamSynchronize(stream), "MTP prefill sync");
                xch->complete(stream);
                interior_flag |= f;  // propagated through the final seed merge on both ranks
            }
            last = R - 1;
        }
        // Preserve last_hidden()'s single final-row contract before the existing head and MTP seed paths.
        if (last != 0)
            hip_ok(hipMemcpyAsync(hidden, hidden + static_cast<std::size_t>(last) * kHidden, kHidden * 4,
                                  hipMemcpyDeviceToDevice, stream), "MTP prefill last hidden");
        lib.run(head, hidden, logits, 1, stream);
        D::argmax(logits, vocab_n, vocab0, amv, ami, flag, 1, stream);
        check_flag("head");
        xch->argmax(amv, ami, 1, stream);
        Engine::MtpPrefillOut out;
        hip_ok(hipMemcpyAsync(&out.y0, ami, 4, hipMemcpyDeviceToHost, stream), "MTP prefill y0");
        hip_ok(hipStreamSynchronize(stream), "MTP prefill head sync");
        hip_ok(hipMemcpyAsync(trunk_logits, logits, std::size_t{vocab_n} * sizeof(float),
                              hipMemcpyDeviceToDevice, stream), "save trunk logits");
        out.mtp = mtp(slot, hidden, {&out.y0, 1}, pos0 + tokens.size() - 1, ctl, true, interior_flag);
        hip_ok(hipMemcpyAsync(logits, trunk_logits, std::size_t{vocab_n} * sizeof(float),
                              hipMemcpyDeviceToDevice, stream), "restore trunk logits");
        hip_ok(hipStreamSynchronize(stream), "trunk logits sync");
        out.hp = m_h;
        no_mtp.at(slot) = 0;
        ++prof.calls;
        return out;
    }

    void reset(std::uint32_t slot) {
        for (const LayerW& w : layers)
            if (!w.kda && w.dst.dump_epoch == std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("kvshard: dump epoch exhausted");
        if (kv_shard) kv_shard->reset();
        for (LayerW& w : layers) {
            if (w.kda) {
                const std::size_t s = static_cast<std::size_t>(kda_heads) * kKdaDim * kKdaDim, c = 3ULL * kda_heads * kKdaDim * 3;
                hip_ok(hipMemsetAsync(w.kst.S + slot * s, 0, s * 4, stream), "reset S");
                hip_ok(hipMemsetAsync(w.kst.conv + slot * c, 0, c * 4, stream), "reset conv");
            } else {
                ++w.dst.dump_epoch;
                const std::size_t cap = static_cast<std::size_t>(w.dst.shard ? kvshard::storage_tokens(w.dst.cap) : w.dst.cap);
                hip_ok(hipMemsetAsync(w.dst.latent + slot * cap * kKvLora, 0, cap * kKvLora * 2, stream), "reset latent");
                hip_ok(hipMemsetAsync(w.dst.ikey + slot * cap * kIdxDim, 0, cap * kIdxDim * 2, stream), "reset key");
                hip_ok(hipMemsetAsync(w.dst.igate + slot * cap * kIdxDim, 0, cap * kIdxDim * 2, stream), "reset gate");
                hip_ok(hipMemsetAsync(w.dst.pooled + slot * (cap / kKpool) * kIdxDim, 0, (cap / kKpool) * kIdxDim * 4, stream), "reset pooled");
            }
        }
        hip_ok(hipStreamSynchronize(stream), "reset");
        if (opt.spec) pending.at(slot) = 0;  // the zeroed state needs no replay
        if (slot < no_mtp.size()) no_mtp[slot] = 0;
    }

    // Shared layer-45 body: chunked teacher forcing and the single-row drafter use the same weights/state.
    // Only the caller runs the LM head; no chunk-sized vocabulary scratch is needed.
    void mtp_cache(std::uint32_t slot, const float* h, std::span<const std::uint32_t> tokens,
                   std::uint64_t pos0, bool pfc) {
        const auto R = static_cast<unsigned>(tokens.size());
        select_exchange(pfc ? ForwardKind::Prefill : ForwardKind::Mtp, R);
        LayerW& w = layers.back();
        // Keep asynchronous metadata uploads alive through the caller's head/flag synchronization.
        mtp_slots.assign(R, slot);
        mtp_positions.resize(R);
        mtp_rows.resize(R);
        auto& sl = mtp_slots;
        auto& ps = mtp_positions;
        auto& rows = mtp_rows;
        for (unsigned j = 0; j < R; ++j) {
            if (tokens[j] >= kVocab) throw std::out_of_range("token id");
            ps[j] = pos0 + j;
            rows[j] = Row{slot, tokens[j], pos0 + j};
        }
        hip_ok(hipMemsetAsync(mtp_flag, 0, 4, stream), "mtp flag");
        hip_ok(hipMemcpyAsync(tok_d, tokens.data(), R * 4, hipMemcpyHostToDevice, stream), "mtp tokens");
        hip_ok(hipMemcpyAsync(slot_d, sl.data(), R * 4, hipMemcpyHostToDevice, stream), "mtp slots");
        hip_ok(hipMemcpyAsync(pos_d, ps.data(), R * 8, hipMemcpyHostToDevice, stream), "mtp pos");
        const CallCtx c{&lib, rows, slot_d, pos_d, mtp_flag, stream, !pfc, nullptr};
        embed1_kernel<<<dim3(kHidden / 256, R), 256, 0, stream>>>(embed, tok_d, m_e);
        D::GlmNorm::run(m_e, w.enorm, D::WType::BF16, kHidden, kEps, m_en, R, stream);
        D::GlmNorm::run(h, w.hnorm, D::WType::BF16, kHidden, kEps, m_hn, R, stream);
        cat2_kernel<<<dim3(kHidden / 256, R), 256, 0, stream>>>(m_en, m_hn, m_cat);
        if (pfc) pf->linear_exact(w.eh_proj, c, m_cat, m_z, 2 * kHidden, kHidden);
        else lib.run(w.eh_proj, m_cat, m_z, R, stream);  // [enorm(e) | hnorm(h)] -> z
        D::GlmNorm::run(m_z, w.in_norm, D::WType::BF16, kHidden, kEps, xn, R, stream);
        if (pfc) pf->dsa(w.dsaw, w.dst, c, xn, part);
        else attn->dsa(w.dsaw, w.dst, c, xn, part);
        xch->reduce(part, R, stream);
        const std::size_t n = static_cast<std::size_t>(R) * kHidden;
        add_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(m_z, part, m_r, n);  // r = z + attn
        D::GlmNorm::run(m_r, w.post_norm, D::WType::BF16, kHidden, kEps, xn, R, stream);
        if (pfc) {
            glm::pf::dense_rows(xn, w.moew.router, D::WType::F16, kHidden, 0, kExperts, rlog, R, stream);
            D::route(rlog, w.moew.bias, ids, wts, R, stream);
            if (pf->moe_by_row()) {
                const unsigned B = std::clamp(pf->moe_row_block(), 1U, 8U);
                for (unsigned q = 0; q < R; q += B) {
                    const CallCtx cq{&lib, c.rows.subspan(q, std::min(B, R - q)), slot_d + q, pos_d + q,
                                     mtp_flag, stream, false, nullptr};
                    moe->forward(w.moew, cq, xn + static_cast<std::size_t>(q) * kHidden,
                                 ids + static_cast<std::size_t>(q) * kTopE, wts + static_cast<std::size_t>(q) * kTopE,
                                 part + static_cast<std::size_t>(q) * kHidden);
                }
            } else {
                pf->moe(w.moew, c, xn, ids, wts, part);
            }
        } else {
            router_route(xn, w.moew, R);
            moe->forward(w.moew, c, xn, ids, wts, part);
        }
        xch->reduce(part, R, stream);
        add_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(m_r, part, m_y, n);  // y = r + MoE
        D::GlmNorm::run(m_y, w.head_norm, D::WType::BF16, kHidden, kEps, m_h, R, stream);  // h' = shared_head.norm(y)
    }

    // ---- MTP drafter (MTP arithmetic consensus; design notes section 3)
    Engine::MtpOut mtp(std::uint32_t slot, const float* h, std::span<const std::uint32_t> tokens, std::uint64_t pos0,
                       Exchange::MergeCtl& ctl, bool bootstrap = false, unsigned inherited_flag = 0) {
        if (!opt.mtp || layers.empty() || !layers.back().mtp) throw std::logic_error("mtp needs Options::mtp (layer 45 loaded)");
        if (final_norm == nullptr) throw std::logic_error("mtp needs the head");
        const auto R = static_cast<unsigned>(tokens.size());
        if (R == 0 || R > 8) throw std::invalid_argument("mtp: 1..8 rows");
        if (slot >= opt.slots || pos0 + R > opt.capacity) throw std::out_of_range("mtp: slot / MTP capacity (clamp first)");
        if (no_mtp.at(slot) && !bootstrap) throw std::logic_error("mtp: the slot was chunk-prefilled (MTP cache not filled; reset first)");
        mtp_cache(slot, h, tokens, pos0, false);
        hip_ok(hipMemcpyAsync(m_hs + static_cast<std::size_t>(slot) * 8 * kHidden, m_h, std::size_t{R} * kHidden * sizeof(float),
                              hipMemcpyDeviceToDevice, stream), "mtp hidden per slot");
        lib.run(head, m_h, logits, R, stream);
        D::argmax(logits, vocab_n, vocab0, amv, ami, mtp_flag, R, stream);
        lse_kernel<<<1, 256, 0, stream>>>(logits + static_cast<std::size_t>(R - 1) * vocab_n, vocab_n, m_mz);
        hip_ok(hipGetLastError(), "mtp kernels");
        std::vector<float> v(R), mz(2);
        std::vector<std::uint32_t> id(R);
        unsigned f = 0;
        hip_ok(hipMemcpyAsync(v.data(), amv, R * 4, hipMemcpyDeviceToHost, stream), "mtp v");
        hip_ok(hipMemcpyAsync(id.data(), ami, R * 4, hipMemcpyDeviceToHost, stream), "mtp id");
        hip_ok(hipMemcpyAsync(mz.data(), m_mz, 8, hipMemcpyDeviceToHost, stream), "mtp mz");
        hip_ok(hipMemcpyAsync(&f, mtp_flag, 4, hipMemcpyDeviceToHost, stream), "mtp flag");
        hip_ok(hipStreamSynchronize(stream), "mtp sync");
        std::vector<Exchange::MergeRow> m(R);
        for (unsigned j = 0; j < R; ++j) m[j] = Exchange::MergeRow{v[j], id[j], f | inherited_flag};
        ctl.replay_flag = 0;
        ctl.mtp_max = mz[0];
        ctl.mtp_sum = mz[1];
        xch->merge(m.data(), R, ctl, stream);
        Engine::MtpOut out;
        for (unsigned j = 0; j < R; ++j) {
            out.id.push_back(m[j].id);
            out.flagged = out.flagged || m[j].flag != 0;
        }
        // the argmax logit equals the merged max M, so its probability is exp(M - M) / Z = 1 / Z
        out.confidence = ctl.mtp_sum > 0.0F ? 1.0F / ctl.mtp_sum : 0.0F;
        return out;
    }

    // ---- speculative verify (design notes v3, approved arithmetic contract)
    Engine::VerifyOut verify(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::uint64_t p, Exchange::MergeCtl& ctl) {
        if (!opt.spec) throw std::logic_error("verify needs Options::spec");
        if (opt.layers != kLayers && !smoke_head) throw std::logic_error("verify needs all 45 layers");
        const auto R = static_cast<unsigned>(tokens.size());
        if (R == 0 || R > 8) throw std::invalid_argument("verify: 1..8 rows");
        if (slot >= opt.slots || p + R > opt.capacity) throw std::out_of_range("verify: slot / capacity (clamp K first)");
        hip_ok(hipMemsetAsync(row_flag, 0, R * 4, stream), "row flags");
        hip_ok(hipMemsetAsync(replay_flag, 0, 4, stream), "replay flag");
        const SpecRows sp{slot, pending[slot], cur[slot], row_flag, replay_flag};
        pending[slot] = 0;  // consumed by this call whatever the outcome (stored by the replay); commit sets it again
        std::vector<Row> rows(R);
        for (unsigned j = 0; j < R; ++j) rows[j] = Row{slot, tokens[j], p + j};
        return verify_rows(rows, sp, ctl);
    }
    bool verify_multi_supported() const { return attn && attn->supports_multi_slot_verify(); }
    Engine::VerifyOut verify_multi(std::span<const Row> rows, Exchange::MergeCtl& ctl) {
        if (!opt.spec) throw std::logic_error("verify needs Options::spec");
        if (opt.layers != kLayers && !smoke_head) throw std::logic_error("verify needs all 45 layers");
        const auto R = static_cast<unsigned>(rows.size());
        if (R == 0 || R > 8) throw std::invalid_argument("verify_multi: 1..8 rows");
        std::vector<SpecSeg> seg;
        for (unsigned r = 0; r < R; ++r) {
            if (rows[r].slot >= opt.slots || rows[r].pos >= opt.capacity) throw std::out_of_range("verify_multi: slot / capacity");
            if (r > 0 && rows[r].slot == rows[r - 1].slot) {
                if (rows[r].pos != rows[r - 1].pos + 1) throw std::invalid_argument("verify_multi: a slot's rows need consecutive positions");
                ++seg.back().n;
                continue;
            }
            if (r > 0 && rows[r].slot < rows[r - 1].slot) throw std::invalid_argument("verify_multi: rows not sorted by slot");
            seg.push_back(SpecSeg{rows[r].slot, 0, 0, r, 1});
        }
        if (seg.size() == 1) {
            std::vector<std::uint32_t> t(R);
            for (unsigned r = 0; r < R; ++r) t[r] = rows[r].token;
            return verify(seg[0].slot, t, rows[0].pos, ctl);
        }
        if (!verify_multi_supported()) throw std::logic_error("verify_multi: the attention seam verifies one slot per call");
        hip_ok(hipMemsetAsync(row_flag, 0, R * 4, stream), "row flags");
        hip_ok(hipMemsetAsync(replay_flag, 0, 4, stream), "replay flag");
        for (auto& g : seg) {
            g.n_rep = pending[g.slot];
            g.cur = cur[g.slot];
            pending[g.slot] = 0;  // consumed whatever the outcome; each slot's commit sets it again
        }
        SpecRows sp{seg[0].slot, seg[0].n_rep, seg[0].cur, row_flag, replay_flag};
        sp.seg = seg.data();
        sp.nseg = static_cast<unsigned>(seg.size());
        return verify_rows(rows, sp, ctl);
    }
    // The verify forward + head + merge (rows prepared by verify / verify_multi, flags cleared).
    Engine::VerifyOut verify_rows(std::span<const Row> rows, const SpecRows& sp, Exchange::MergeCtl& ctl) {
        const auto R = static_cast<unsigned>(rows.size());
        layers_forward(rows, opt.layers, ForwardKind::Verify, &sp);
        D::hc_final(X, hfin, R, stream);
        D::GlmNorm::run(hfin, final_norm, D::WType::BF16, kHidden, kEps, hidden, R, stream);
        lib.run(head, hidden, logits, R, stream);
        D::argmax(logits, vocab_n, vocab0, amv, ami, row_flag, R, stream, 1);
        std::vector<float> v(R);
        std::vector<std::uint32_t> id(R), fl(R);
        unsigned rf = 0;
        hip_ok(hipMemcpyAsync(v.data(), amv, R * 4, hipMemcpyDeviceToHost, stream), "verify v");
        hip_ok(hipMemcpyAsync(id.data(), ami, R * 4, hipMemcpyDeviceToHost, stream), "verify id");
        hip_ok(hipMemcpyAsync(fl.data(), row_flag, R * 4, hipMemcpyDeviceToHost, stream), "verify flags");
        hip_ok(hipMemcpyAsync(&rf, replay_flag, 4, hipMemcpyDeviceToHost, stream), "verify replay flag");
        hip_ok(hipStreamSynchronize(stream), "verify sync");
        std::vector<Exchange::MergeRow> m(R);
        for (unsigned j = 0; j < R; ++j) m[j] = Exchange::MergeRow{v[j], id[j], fl[j]};
        ctl.replay_flag = rf;
        xch->merge(m.data(), R, ctl, stream);
        Engine::VerifyOut out;
        out.replay_flag = ctl.replay_flag != 0;
        for (unsigned j = 0; j < R; ++j) {
            out.id.push_back(m[j].id);
            out.flag.push_back(m[j].flag);
        }
        ++prof.calls;
        return out;
    }
    void commit(std::uint32_t slot, unsigned rows, std::uint64_t) {
        if (!opt.spec) throw std::logic_error("commit needs Options::spec");
        if (rows == 0 || rows > 8) throw std::invalid_argument("commit: 1..8 rows");
        cur.at(slot) = 1 - cur.at(slot);  // the verify's new rows (set 1 - cur) become the replay set
        pending.at(slot) = rows;
        // DSA: frontier only (positions >= p + rows are overwritten by later appends); MTP truncation lands with mtp.
    }
    void flush(std::uint32_t slot) {
        if (!opt.spec || pending.at(slot) == 0) return;
        const unsigned C = kda_heads * kKdaDim, n = pending[slot], cs = cur[slot];
        const std::size_t SZ = static_cast<std::size_t>(kda_heads) * kKdaDim * kKdaDim, CZ = 3ULL * C * 3;
        hip_ok(hipMemsetAsync(replay_flag, 0, 4, stream), "flush flag");
        for (LayerW& w : layers) {
            if (!w.kda) continue;
            const D::KdaDev kd{kda_heads, w.kdaw.conv, w.kdaw.A_log, w.kdaw.dt_bias, w.kdaw.o_norm};
            const std::size_t set = (static_cast<std::size_t>(slot) * 2 + cs) * 8;
            const std::size_t nxt = (static_cast<std::size_t>(slot) * 2 + 1 - cs) * 8;
            D::kda_verify(kd, w.kst.rep_qkv + set * 3 * C, w.kst.rep_scr + set * kda_heads * 4 * kKdaDim,
                          w.kst.rep_beta + set * kda_heads, n, nullptr, nullptr, nullptr, nullptr, 0, w.kst.S + slot * SZ,
                          w.kst.conv + slot * CZ, w.kst.rep_scr + nxt * kda_heads * 4 * kKdaDim, w.kst.rep_beta + nxt * kda_heads,
                          part, part, replay_flag, row_flag, stream);
        }
        pending[slot] = 0;
        unsigned rf = 0;
        hip_ok(hipMemcpyAsync(&rf, replay_flag, 4, hipMemcpyDeviceToHost, stream), "flush flag");
        hip_ok(hipStreamSynchronize(stream), "flush sync");
        if (rf) throw std::runtime_error("non-finite value in a committed KDA row (C2) at flush");
    }

    // Device regions of one slot's snapshot, in serialization order (frontier caps the DSA prefix).
    template <typename F> void for_each_state_region(std::uint32_t slot, std::uint64_t frontier, F&& f) const {
        for (const LayerW& w : layers) {
            if (w.kda) {
                const std::size_t s = static_cast<std::size_t>(kda_heads) * kKdaDim * kKdaDim, c = 3ULL * kda_heads * kKdaDim * 3;
                f(w.kst.S + slot * s, s * 4);
                f(w.kst.conv + slot * c, c * 4);
            } else {
                const std::size_t cap = static_cast<std::size_t>(w.dst.shard ? kvshard::storage_tokens(w.dst.cap) : w.dst.cap);
                const auto global_n = std::min<std::uint64_t>(w.dst.cap, frontier);
                const std::size_t n = static_cast<std::size_t>(w.dst.shard ? kvshard::owned_prefix(global_n, opt.rank) : global_n);
                f(w.dst.latent + slot * cap * kKvLora, n * kKvLora * 2);
                f(w.dst.ikey + slot * cap * kIdxDim, n * kIdxDim * 2);
                f(w.dst.igate + slot * cap * kIdxDim, n * kIdxDim * 2);
                f(w.dst.pooled + slot * (cap / kKpool) * kIdxDim, (n / kKpool) * kIdxDim * 4);
            }
        }
    }

    std::size_t state_payload_bytes(std::uint32_t slot, std::uint64_t frontier) const {
        std::size_t total = 0;
        for_each_state_region(slot, frontier, [&](const void*, std::size_t bytes) { total += bytes; });
        return total;
    }

    // One sized buffer, copies queued on the engine stream (ordered after its pending work), one sync.
    std::vector<std::uint8_t> state_bytes(std::uint32_t slot, std::uint64_t frontier = ~std::uint64_t{0}) {
        const std::size_t key_bytes = kv_shard ? sizeof(kvshard::SnapshotKey) : 0;
        const std::size_t payload = state_payload_bytes(slot, frontier);
        std::vector<std::uint8_t> out(key_bytes + payload);
        std::size_t o = key_bytes;
        for_each_state_region(slot, frontier, [&](const void* p, std::size_t bytes) {
            if (bytes) hip_ok(hipMemcpyAsync(out.data() + o, p, bytes, hipMemcpyDeviceToHost, stream), "state");
            o += bytes;
        });
        hip_ok(hipStreamSynchronize(stream), "state sync");
        if (kv_shard) {
            const auto key = kvshard::snapshot_key(opt.capacity, opt.rank, std::min(opt.capacity, frontier), payload);
            std::memcpy(out.data(), key.data(), sizeof(key));
        }
        return out;
    }

    void restore_state(std::uint32_t slot, std::span<const std::uint8_t> in, std::uint64_t frontier) {
        if (slot >= opt.slots) throw std::out_of_range("restore_state: slot");
        const std::size_t key_bytes = kv_shard ? sizeof(kvshard::SnapshotKey) : 0;
        const std::size_t payload = state_payload_bytes(slot, frontier);
        if (kv_shard)
            kvshard::validate_snapshot(in.data(), in.size(), true,
                                       kvshard::snapshot_key(opt.capacity, opt.rank, std::min(opt.capacity, frontier), payload));
        else kvshard::validate_snapshot(in.data(), in.size(), false);
        if (in.size() < key_bytes + payload) throw std::invalid_argument("restore_state: state shorter than this engine's layout");
        if (in.size() > key_bytes + payload) throw std::invalid_argument("restore_state: state longer than this engine's layout");
        reset(slot);  // memsets on the stream; the copies below are queued after them
        std::size_t o = key_bytes;
        for_each_state_region(slot, frontier, [&](const void* p, std::size_t bytes) {
            if (bytes) hip_ok(hipMemcpyAsync(const_cast<void*>(p), in.data() + o, bytes, hipMemcpyHostToDevice, stream), "restore");
            o += bytes;
        });
        hip_ok(hipStreamSynchronize(stream), "restore sync");
    }
};

Engine::Engine(Options options, dist::Communicator* comm) : impl_(std::make_unique<Impl>(std::move(options), comm)) {}
Engine::~Engine() = default;
LoadReport Engine::load(const std::function<unsigned(const mem::LoadAdmission&)>& select_max_rows,
                        const std::function<void(const mem::LoadAdmission&)>& admit_prefill,
                    const std::function<std::shared_ptr<PrefillPath>(bool)>& make_prefill) {
    for (;;) {
        try {
            return impl_->load(select_max_rows, admit_prefill, make_prefill);
        } catch (const Impl::InstallRefused&) {
            // No forward has run. Drop the rejected path, weights, KV and any
            // partial install before replanning; AUTO may now change KV storage.
            const auto options = impl_->opt;
            auto* comm = impl_->comm;
            const auto skip = impl_->capacity_candidate + 1;
            auto external = options.external_exchange ? impl_->fallback_xch : nullptr;
            impl_.reset();
            impl_ = std::make_unique<Impl>(options, comm);
            impl_->capacity_skip = skip;
            if (external) set_exchange(std::move(external));
        }
    }
}
std::uint32_t Engine::prefill(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::uint64_t pos0) {
    // Cover every forward issued by either path, including prompt-time MTP work.
    struct PromptIdentityScope {
        bool& current;
        bool saved, saved_active;
        PromptIdentityScope(bool& value, bool prompt_r1)
            : current(value), saved(value), saved_active(strix::glm::r1b::active()) {
            if (prompt_r1) {
                current = false;
                strix::glm::r1b::set_active(false);
            }
        }
        ~PromptIdentityScope() {
            current = saved;
            strix::glm::r1b::set_active(saved_active);
        }
    } scope(impl_->r1b, impl_->identity == IdentityMode::kR1b);
    if (!impl_->reference_prefill_installed()) return impl_->prefill_chunked(slot, tokens, pos0);
    if (impl_->kv_shard) throw std::logic_error("kvshard: install the bounded chunked prefill path");
    // Reference identity (arithmetic contract 6.2): prefill = decode, one row at a time; returns the greedy token after the last.
    std::uint32_t next = 0;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const Row r{slot, tokens[i], pos0 + i};
        next = impl_->step({&r, 1})[0];
    }
    return next;
}
Engine::MtpPrefillOut Engine::prefill_mtp(std::uint32_t slot, std::span<const std::uint32_t> tokens,
                                        std::uint64_t pos0, Exchange::MergeCtl& ctl) {
    if (impl_->identity == IdentityMode::kR1bAll)
        throw std::logic_error("prefill_mtp requires R1 prompts (r1 or r1b)");
    struct Scope {
        Impl& impl;
        bool saved, active;
        explicit Scope(Impl& i) : impl(i), saved(i.r1b), active(strix::glm::r1b::active()) {
            impl.r1b = false;
            strix::glm::r1b::set_active(false);
        }
        ~Scope() {
            impl.r1b = saved;
            strix::glm::r1b::set_active(active);
        }
    } scope(*impl_);
    return impl_->prefill_mtp(slot, tokens, pos0, ctl);
}
std::vector<std::uint32_t> Engine::step(std::span<const Row> rows) { return impl_->step(rows); }
void Engine::reset(std::uint32_t slot) { impl_->reset(slot); }
void Engine::set_attn(std::shared_ptr<AttnPath> p) {
    if (impl_->kv_shard) throw std::logic_error("kvshard: replacing the attention seam is not admitted");
    impl_->attn = std::move(p);
}
void Engine::set_null_flip(int layer, std::uint64_t pos, unsigned elem) {
    if (layer >= 0 && (layer >= static_cast<int>(impl_->layers.size()) || elem >= kHidden)) throw std::invalid_argument("null flip: layer / elem");
    impl_->null_layer = layer;
    impl_->null_pos = pos;
    impl_->null_elem = elem;
}
std::uint64_t Engine::moe_cold_bytes() const {
    std::uint64_t n = 0;
    for (const auto& w : impl_->layers)
        if (w.moe) n += impl_->moe->warm_bytes(w.moew);
    return n;
}
void Engine::set_moe(std::shared_ptr<MoePath> p) {
    // With a prefill path installed, the new seam's tables are admitted and built first (a failure keeps the old seam).
    if (p && impl_->loaded && impl_->pf_installed && impl_->pf) impl_->warm_moe(*p, impl_->pf->admission_margin());
    impl_->moe = std::move(p);
}
void Engine::set_identity_mode(IdentityMode mode) {
    (void)identity_mode_name(mode);
    impl_->identity = mode;
    impl_->r1b = mode != IdentityMode::kR1;
    strix::glm::r1b::set_active(impl_->r1b);
}
IdentityMode Engine::identity_mode() const { return impl_->identity; }
void Engine::set_identity(bool r1b) { set_identity_mode(r1b ? IdentityMode::kR1bAll : IdentityMode::kR1); }
bool Engine::identity_r1b() const { return impl_->identity != IdentityMode::kR1; }
void Engine::set_route_log(bool on) {
    if (on && impl_->route_log_d == nullptr) {
        impl_->route_log_d = dalloc<std::uint32_t>(static_cast<std::size_t>(kLayers + 1) * kTopE);
        hip_ok(hipMemset(impl_->route_log_d, 0xFF, static_cast<std::size_t>(kLayers + 1) * kTopE * 4), "route log init");
    }
    impl_->route_log = on;
}
std::vector<float> Engine::gate_residual_layers(std::span<const std::uint32_t> tokens, unsigned n_layers) {
    return impl_->gate_residual_layers(tokens, n_layers);
}
std::vector<std::uint32_t> Engine::route_log() const {
    std::vector<std::uint32_t> v(static_cast<std::size_t>(kLayers + 1) * kTopE, 0xFFFFFFFFU);
    if (impl_->route_log_d == nullptr) return v;
    hip_ok(hipStreamSynchronize(impl_->stream), "route log sync");
    hip_ok(hipMemcpy(v.data(), impl_->route_log_d, v.size() * 4, hipMemcpyDeviceToHost), "route log copy");
    return v;
}
void Engine::set_exchange(std::shared_ptr<Exchange> p) {
    if (!p) throw std::invalid_argument("set_exchange: null exchange");
    impl_->switch_exchange(p);
    impl_->fallback_xch = std::move(p);
    impl_->decode_xch.reset();  // explicit seam overrides retain their original all-forward meaning
}
std::shared_ptr<Exchange> Engine::exchange() const { return impl_->fallback_xch; }
bool Engine::decode_xfer_enabled() const { return impl_->decode_xch != nullptr; }
bool Engine::model_fast_i4() const { return impl_->model_fast_i4; }
bool Engine::kv_shard_enabled() const { return impl_->kv_shard != nullptr; }
unsigned Engine::prefill_pipeline() const { return impl_->opt.pf_pipeline; }
void Engine::set_prefill(std::shared_ptr<PrefillPath> p) {
    // Before load() the path is only stored; load() installs it once the weights exist (Impl::install_prefill).
    if (p && impl_->loaded) impl_->install_prefill(*p);
    impl_->pf = std::move(p);
    impl_->pf_installed = impl_->pf != nullptr;
}
Engine::VerifyOut Engine::verify(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::uint64_t p, Exchange::MergeCtl& ctl) {
    return impl_->verify(slot, tokens, p, ctl);
}
void Engine::commit(std::uint32_t slot, unsigned rows, std::uint64_t p) { impl_->commit(slot, rows, p); }
void Engine::flush(std::uint32_t slot) { impl_->flush(slot); }
const float* Engine::last_hidden() const { return impl_->hidden; }
const float* Engine::last_logits() const { return impl_->logits; }
std::uint32_t Engine::vocab_slice() const { return impl_->vocab_n; }
void Engine::save_trunk_logits() {
    if (!impl_->trunk_logits) throw std::logic_error("save_trunk_logits requires MTP");
    hip_ok(hipMemcpyAsync(impl_->trunk_logits, impl_->logits, std::size_t{impl_->vocab_n} * sizeof(float),
                          hipMemcpyDeviceToDevice, impl_->stream), "save trunk logits");
}
void Engine::restore_trunk_logits() {
    if (!impl_->trunk_logits) throw std::logic_error("restore_trunk_logits requires MTP");
    hip_ok(hipMemcpyAsync(impl_->logits, impl_->trunk_logits, std::size_t{impl_->vocab_n} * sizeof(float),
                          hipMemcpyDeviceToDevice, impl_->stream), "restore trunk logits");
    hip_ok(hipStreamSynchronize(impl_->stream), "trunk logits sync");
}
std::vector<std::uint8_t> Engine::state_bytes_upto(std::uint32_t slot, std::uint64_t frontier) const {
    return impl_->state_bytes(slot, frontier);
}
void Engine::restore_state(std::uint32_t slot, std::span<const std::uint8_t> bytes, std::uint64_t frontier) {
    impl_->restore_state(slot, bytes, frontier);
}
Engine::MtpOut Engine::mtp(std::uint32_t slot, const float* h, std::span<const std::uint32_t> tokens, std::uint64_t mtp_pos0,
                           Exchange::MergeCtl& ctl) {
    return impl_->mtp(slot, h, tokens, mtp_pos0, ctl);
}
const float* Engine::mtp_hidden() const { return impl_->m_h; }
const float* Engine::mtp_hidden(std::uint32_t slot) const {
    if (slot >= impl_->opt.slots || impl_->m_hs == nullptr) throw std::out_of_range("mtp_hidden: slot / MTP not loaded");
    return impl_->m_hs + static_cast<std::size_t>(slot) * 8 * kHidden;
}
Engine::VerifyOut Engine::verify_multi(std::span<const Row> rows, Exchange::MergeCtl& ctl) { return impl_->verify_multi(rows, ctl); }
bool Engine::verify_multi_supported() const { return impl_->verify_multi_supported(); }
std::vector<float> Engine::gate_residual(std::span<const std::uint32_t> tokens, unsigned layers, bool) {
    return impl_->gate_residual(tokens, layers);
}
std::vector<std::uint8_t> Engine::state_bytes(std::uint32_t slot) const { return impl_->state_bytes(slot); }
void Engine::set_profiling(bool on) { impl_->profiling = on; }
const Profile& Engine::profile() const { return impl_->prof; }
std::uint32_t Engine::rank() const { return impl_->opt.rank; }

}  // namespace strix::glm::engine
