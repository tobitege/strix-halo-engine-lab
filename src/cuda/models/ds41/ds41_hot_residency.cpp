// deepseek41 CUDA parts-host residency from a hot-expert placement (cuda_ds41_hot_residency.hpp).

#include "strix/cuda/ds41_hot_residency.hpp"

#include "strix/gguf_reader.hpp"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <thread>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strix::ds41::cuda {

// ---- GGUF expert reader ----------------------------------------------------------------------------------------

// positional reads (thread-safe: several loader threads share the handle)
struct GgufExpertReader::Impl {
#ifdef _WIN32
    HANDLE h = INVALID_HANDLE_VALUE;
    ~Impl() { if (h != INVALID_HANDLE_VALUE) CloseHandle(h); }
    bool open(const std::string& path) {
        h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        return h != INVALID_HANDLE_VALUE;
    }
    bool pread(std::uint8_t* dst, std::uint64_t bytes, std::uint64_t at) const {
        while (bytes) {
            const DWORD n = static_cast<DWORD>(std::min<std::uint64_t>(bytes, 1ULL << 30U));
            OVERLAPPED ov{};
            ov.Offset = static_cast<DWORD>(at & 0xffffffffULL);
            ov.OffsetHigh = static_cast<DWORD>(at >> 32U);
            DWORD got = 0;
            if (!ReadFile(h, dst, n, &got, &ov) || got == 0) return false;
            dst += got;
            bytes -= got;
            at += got;
        }
        return true;
    }
#else
    int fd = -1;
    ~Impl() { if (fd >= 0) close(fd); }
    bool open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        return fd >= 0;
    }
    bool pread(std::uint8_t* dst, std::uint64_t bytes, std::uint64_t at) const {
        while (bytes) {
            const auto got = ::pread(fd, dst, static_cast<std::size_t>(std::min<std::uint64_t>(bytes, 1ULL << 30U)), static_cast<off_t>(at));
            if (got <= 0) return false;
            dst += got;
            bytes -= static_cast<std::uint64_t>(got);
            at += static_cast<std::uint64_t>(got);
        }
        return true;
    }
#endif
    std::vector<std::array<std::uint64_t, 3>> offsets;  // [layer] gate / up / down tensor offsets (0: absent)
};

GgufExpertReader::GgufExpertReader(const std::string& path) : impl_(std::make_unique<Impl>()) {
    const auto file = strix::gguf::read_file(path);
    bool have_shape = false;
    for (const auto& t : file.tensors) {
        if (t.name.rfind("blk.", 0) != 0) continue;
        const auto dot = t.name.find('.', 4);
        const auto layer = static_cast<std::uint32_t>(std::stoul(t.name.substr(4, dot - 4)));
        const std::string rest = t.name.substr(dot + 1);
        int part = -1;
        if (rest == "ffn_gate_exps.weight") part = 0;
        else if (rest == "ffn_up_exps.weight") part = 1;
        else if (rest == "ffn_down_exps.weight") part = 2;
        if (part < 0) continue;
        if (part == 0) {
            if (t.ggml_type != 16) throw std::runtime_error("ds41 expert reader: gate is not IQ2_XXS in " + t.name);
            MoeShape s;
            s.hidden = static_cast<std::uint32_t>(t.dimensions[0]);
            s.inter = static_cast<std::uint32_t>(t.dimensions[1]);
            s.experts = static_cast<std::uint32_t>(t.dimensions[2]);
            if (have_shape && (s.hidden != shape_.hidden || s.inter != shape_.inter || s.experts != shape_.experts))
                throw std::runtime_error("ds41 expert reader: layers with different expert shapes");
            shape_.hidden = s.hidden;
            shape_.inter = s.inter;
            shape_.experts = s.experts;
            have_shape = true;
        }
        if (part == 2 && t.ggml_type != 10) throw std::runtime_error("ds41 expert reader: down is not Q2_K in " + t.name);
        if (impl_->offsets.size() <= layer) impl_->offsets.resize(layer + 1U, {0, 0, 0});
        impl_->offsets[layer][static_cast<std::size_t>(part)] = t.absolute_offset;
    }
    if (!have_shape) throw std::runtime_error("ds41 expert reader: no routed expert tensors in " + path);
    if (const auto* m = file.find_metadata("deepseek41.num_experts_per_tok"); m && m->unsigned_value) shape_.top_k = static_cast<std::uint32_t>(*m->unsigned_value);
    layers_ = static_cast<std::uint32_t>(impl_->offsets.size());
    if (!impl_->open(path)) throw std::runtime_error("ds41 expert reader: cannot open " + path);
}

GgufExpertReader::~GgufExpertReader() = default;

void GgufExpertReader::close() noexcept {
#ifdef _WIN32
    if (impl_->h != INVALID_HANDLE_VALUE) CloseHandle(impl_->h);
    impl_->h = INVALID_HANDLE_VALUE;
#else
    if (impl_->fd >= 0) ::close(impl_->fd);
    impl_->fd = -1;
#endif
}

bool GgufExpertReader::has_layer(std::uint32_t layer) const noexcept {
    if (layer >= impl_->offsets.size()) return false;
    const auto& o = impl_->offsets[layer];
    return o[0] && o[1] && o[2];
}

void GgufExpertReader::read(std::uint32_t layer, std::uint32_t expert, int part, std::uint8_t* dst) const {
    if (!has_layer(layer) || expert >= shape_.experts || part < 0 || part > 2) throw std::invalid_argument("ds41 expert reader: no such slice");
    const std::uint64_t bytes = part == 2 ? shape_.down_expert_bytes() : shape_.gate_expert_bytes();
    const std::uint64_t at = impl_->offsets[layer][static_cast<std::size_t>(part)] + std::uint64_t{expert} * bytes;
    if (!impl_->pread(dst, bytes, at)) throw std::runtime_error("ds41 expert reader: short read");
}

std::uint64_t GgufExpertReader::tensor_offset(std::uint32_t layer, int part) const {
    if (!has_layer(layer) || part < 0 || part > 2) throw std::invalid_argument("ds41 expert reader: no such tensor");
    return impl_->offsets[layer][static_cast<std::size_t>(part)];
}

void GgufExpertReader::read_tensor(std::uint32_t layer, int part, std::uint8_t* dst) const {
    if (!has_layer(layer) || part < 0 || part > 2) throw std::invalid_argument("ds41 expert reader: no such tensor");
    const std::uint64_t bytes = (part == 2 ? shape_.down_expert_bytes() : shape_.gate_expert_bytes()) * shape_.experts;
    if (!impl_->pread(dst, bytes, impl_->offsets[layer][static_cast<std::size_t>(part)])) throw std::runtime_error("ds41 expert reader: short read");
}

// ---- residency -------------------------------------------------------------------------------------------------

struct HotResidency::Layer {
    ResidentLayer rl;
    std::uint32_t statics = 0, dynamics = 0;
    DeviceBuffer<std::uint8_t> gate, up, down;
    DeviceBuffer<std::int32_t> d_slot_of;
    std::int32_t* staging = nullptr;          // pinned copy of the table for the async update
    std::vector<std::int32_t> dyn_expert;     // [dynamic slot] expert or -1
    std::vector<std::uint64_t> dyn_used;      // [dynamic slot] last token that used / filled it
    std::vector<const std::uint8_t*> source;  // [expert] pinned gate|up|down of the expert, or null
    cudaEvent_t refilled = nullptr;
    bool pending = false;
    ~Layer() {
        if (staging) cudaFreeHost(staging);
        if (refilled) cudaEventDestroy(refilled);
    }
};

HotResidency::HotResidency(const HotPlacement& placement, const GgufExpertReader& reader, const HotResidencyOptions& o)
    : shape_(o.shape), copies_per_token_(o.copies_per_token) {
    const auto E = shape_.experts;
    const std::uint32_t L = placement.geometry.layers;
    if (placement.geometry.experts != E || placement.geometry.top_k != shape_.top_k ||
        placement.geometry.expert_bytes != shape_.expert_bytes())
        throw std::invalid_argument("ds41 hot residency: placement geometry differs from the model");
    if (!o.dynamic_slots.empty() && o.dynamic_slots.size() != L) throw std::invalid_argument("ds41 hot residency: dynamic slots per layer");
    if (!o.source.empty() && o.source.size() != L) throw std::invalid_argument("ds41 hot residency: source per layer");
    int least = 0, greatest = 0;
    STRIX_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    STRIX_CUDA_CHECK(cudaStreamCreateWithPriority(&refill_, cudaStreamNonBlocking, least));
    STRIX_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&required_), std::size_t{L} * 4U, cudaHostAllocMapped | cudaHostAllocPortable));
    STRIX_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&refill_done_), std::size_t{L} * 4U, cudaHostAllocMapped | cudaHostAllocPortable));
    std::fill(required_, required_ + L, 0U);
    std::fill(refill_done_, refill_done_ + L, 0U);
    STRIX_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&required_dev_), required_, 0));
    STRIX_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&refill_done_dev_), refill_done_, 0));
    gen_.assign(L, 0U);
    const std::uint64_t gb = shape_.gate_expert_bytes(), db = shape_.down_expert_bytes(), eb = shape_.expert_bytes();

    // pinned source, packed per expert (gate | up | down), in 1 GiB chunks
    std::uint64_t source_experts = 0;
    for (const auto l : o.layers)
        if (!o.source.empty()) source_experts += o.source[l].size();
    const std::uint64_t per_chunk = std::max<std::uint64_t>(1, (1ULL << 30U) / eb);
    std::uint64_t chunk_left = 0;
    std::uint8_t* chunk_at = nullptr;
    const auto take_source = [&]() {
        if (chunk_left == 0) {
            const auto n = std::min<std::uint64_t>(per_chunk, source_experts);
            void* p = nullptr;
            STRIX_CUDA_CHECK(cudaHostAlloc(&p, n * eb, cudaHostAllocPortable));
            source_chunks_.push_back(p);
            chunk_at = static_cast<std::uint8_t*>(p);
            chunk_left = n;
            source_bytes_ += n * eb;
        }
        auto* r = chunk_at;
        chunk_at += eb;
        --chunk_left;
        --source_experts;
        return r;
    };

    layers_.resize(L);
    {
        for (const auto l : o.layers) {
            if (l >= L) throw std::invalid_argument("ds41 hot residency: layer outside the placement");
            if (!reader.has_layer(l)) throw std::invalid_argument("ds41 hot residency: layer " + std::to_string(l) + " not in the GGUF");
            auto lay = std::make_unique<Layer>();
            const auto& pl = placement.layers[l];
            const auto& stat = pl.experts[static_cast<std::size_t>(HotDevice::rtx_pro)];
            lay->statics = static_cast<std::uint32_t>(stat.size());
            lay->dynamics = o.dynamic_slots.empty() ? 0U : std::min(o.dynamic_slots[l], E - lay->statics);
            const std::uint64_t slots = std::uint64_t{lay->statics} + lay->dynamics;
            lay->gate.resize(std::max<std::uint64_t>(1, slots * gb));
            lay->up.resize(std::max<std::uint64_t>(1, slots * gb));
            lay->down.resize(std::max<std::uint64_t>(1, slots * db));
            device_bytes_ += slots * eb;
            lay->rl.layer = l;
            lay->rl.host_slot_of.assign(E, -1);
            for (std::uint32_t i = 0; i < lay->statics; ++i) {
                if (placement.slot_of(l, stat[i]) != i) throw std::runtime_error("ds41 hot residency: placement slot order");
                lay->rl.host_slot_of[stat[i]] = static_cast<std::int32_t>(i);
            }
            lay->d_slot_of.resize(E);
            STRIX_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&lay->staging), E * 4U, cudaHostAllocPortable));
            std::copy(lay->rl.host_slot_of.begin(), lay->rl.host_slot_of.end(), lay->staging);
            STRIX_CUDA_CHECK(cudaMemcpy(lay->d_slot_of.get(), lay->staging, E * 4U, cudaMemcpyHostToDevice));
            lay->rl.weights = {lay->gate.get(), lay->up.get(), lay->down.get()};
            lay->rl.slot_of = lay->d_slot_of.get();
            lay->dyn_expert.assign(lay->dynamics, -1);
            lay->dyn_used.assign(lay->dynamics, 0);
            lay->source.assign(E, nullptr);
            if (!o.source.empty() && lay->dynamics) {
                for (const auto e : o.source[l]) {
                    if (e >= E) throw std::invalid_argument("ds41 hot residency: source expert out of range");
                    if (lay->rl.host_slot_of[e] >= 0) throw std::invalid_argument("ds41 hot residency: source expert is static");
                    lay->source[e] = take_source();
                }
            }
            STRIX_CUDA_CHECK(cudaEventCreateWithFlags(&lay->refilled, cudaEventDisableTiming));
            layers_[l] = std::move(lay);
        }
    }
    if (o.load_weights) {  // weights: layers in parallel (positional reads, one pinned stage per thread)
        constexpr std::uint32_t stage_experts = 16;
        const std::uint64_t part_max = std::max(gb, db);
        std::atomic<std::size_t> next{0};
        std::mutex err_mu;
        std::string err;
        const auto work = [&]() {
            std::uint8_t* stage = nullptr;
            try {
                STRIX_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&stage), stage_experts * part_max, cudaHostAllocPortable));
                for (std::size_t i; (i = next.fetch_add(1)) < o.layers.size();) {
                    const auto l = o.layers[i];
                    auto& lay = *layers_[l];
                    const auto& stat = placement.layers[l].experts[static_cast<std::size_t>(HotDevice::rtx_pro)];
                    for (std::uint32_t i0 = 0; i0 < lay.statics; i0 += stage_experts) {
                        const std::uint32_t n = std::min(stage_experts, lay.statics - i0);
                        for (int part = 0; part < 3; ++part) {
                            const std::uint64_t b = part == 2 ? db : gb;
                            for (std::uint32_t j = 0; j < n; ++j) reader.read(l, stat[i0 + j], part, stage + j * b);
                            auto* dst = (part == 0 ? lay.gate.get() : part == 1 ? lay.up.get() : lay.down.get()) + std::uint64_t{i0} * b;
                            STRIX_CUDA_CHECK(cudaMemcpy(dst, stage, n * b, cudaMemcpyHostToDevice));
                        }
                    }
                    for (std::uint32_t e = 0; e < E; ++e) {
                        auto* p = const_cast<std::uint8_t*>(lay.source[e]);
                        if (!p) continue;
                        reader.read(l, e, 0, p);
                        reader.read(l, e, 1, p + gb);
                        reader.read(l, e, 2, p + 2U * gb);
                    }
                }
            } catch (const std::exception& ex) {
                std::lock_guard<std::mutex> g(err_mu);
                if (err.empty()) err = ex.what();
                next = o.layers.size();
            }
            if (stage) cudaFreeHost(stage);
        };
        const unsigned threads = std::max(1U, std::min<unsigned>(o.load_threads, static_cast<unsigned>(o.layers.size())));
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < threads; ++t) pool.emplace_back(work);
        for (auto& t : pool) t.join();
        if (!err.empty()) throw std::runtime_error(err);
    }
    STRIX_CUDA_CHECK(cudaDeviceSynchronize());
}

HotResidency::~HotResidency() {
    if (refill_) { cudaStreamSynchronize(refill_); cudaStreamDestroy(refill_); }
    layers_.clear();
    for (auto* p : source_chunks_) cudaFreeHost(p);
    if (required_) cudaFreeHost(required_);
    if (refill_done_) cudaFreeHost(refill_done_);
}

const ResidentLayer& HotResidency::layer(std::uint32_t l) const {
    if (l >= layers_.size() || !layers_[l]) throw std::invalid_argument("ds41 hot residency: layer not held");
    return layers_[l]->rl;
}
std::uint32_t HotResidency::static_slots(std::uint32_t l) const { return layers_.at(l) ? layers_[l]->statics : 0U; }
std::uint32_t HotResidency::dynamic_slots(std::uint32_t l) const { return layers_.at(l) ? layers_[l]->dynamics : 0U; }
std::uint64_t HotResidency::device_bytes() const noexcept { return device_bytes_; }
std::uint64_t HotResidency::source_bytes() const noexcept { return source_bytes_; }

void HotResidency::reset_dynamic() {
    STRIX_CUDA_CHECK(cudaStreamSynchronize(refill_));
    for (auto& lp : layers_) {
        if (!lp || !lp->dynamics) continue;
        auto& lay = *lp;
        for (auto& e : lay.dyn_expert)
            if (e >= 0) { lay.rl.host_slot_of[static_cast<std::size_t>(e)] = -1; e = -1; }
        std::fill(lay.dyn_used.begin(), lay.dyn_used.end(), 0);
        std::copy(lay.rl.host_slot_of.begin(), lay.rl.host_slot_of.end(), lay.staging);
        STRIX_CUDA_CHECK(cudaMemcpy(lay.d_slot_of.get(), lay.staging, shape_.experts * 4U, cudaMemcpyHostToDevice));
        lay.pending = false;
    }
    STRIX_CUDA_CHECK(cudaDeviceSynchronize());
}

void HotResidency::begin_token() {
    ++token_;
    stats_ = {};
    copies_left_ = copies_per_token_;
}

void HotResidency::before_layer(std::uint32_t l, cudaStream_t stream) {
    auto& lay = *layers_.at(l);
    if (lay.pending) {
        STRIX_CUDA_CHECK(cudaStreamWaitEvent(stream, lay.refilled, 0));
        lay.pending = false;
    }
}

void HotResidency::after_layer(std::uint32_t l, const std::uint32_t* ids, std::uint32_t K, cudaEvent_t layer_done) {
    refill(l, ids, K, layer_done, nullptr, 0U);
}

void HotResidency::after_layer_mapped(std::uint32_t l, const std::uint32_t* ids, std::uint32_t K, const std::uint32_t* done_flag_dev,
                                      std::uint32_t sequence) {
    refill(l, ids, K, nullptr, done_flag_dev, sequence);
}

void HotResidency::refill(std::uint32_t l, const std::uint32_t* ids, std::uint32_t K, cudaEvent_t layer_done, const std::uint32_t* done_flag_dev,
                          std::uint32_t sequence) {
    auto& lay = *layers_.at(l);
    const std::uint64_t gb = shape_.gate_expert_bytes(), db = shape_.down_expert_bytes();
    // the slots this token used
    std::vector<std::uint32_t> misses;
    for (std::uint32_t k = 0; k < K; ++k) {
        const auto e = ids[k];
        const auto s = lay.rl.host_slot_of[e];
        if (s >= static_cast<std::int32_t>(lay.statics)) {
            ++stats_.dynamic_hits;
            lay.dyn_used[static_cast<std::size_t>(s) - lay.statics] = token_;
        } else if (s < 0) {
            ++stats_.remote;
            misses.push_back(e);
        }
    }
    if (misses.empty() || !lay.dynamics) return;
    bool waited = false, any = false;
    for (const auto e : misses) {
        if (!lay.source[e]) { ++stats_.not_in_source; continue; }
        if (copies_left_ == 0) { ++stats_.over_budget; continue; }
        // victim: an empty slot, else the least recently used slot not used by this token
        std::size_t victim = lay.dynamics;
        std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
        for (std::size_t d = 0; d < lay.dynamics; ++d) {
            if (lay.dyn_expert[d] < 0) { victim = d; break; }
            if (lay.dyn_used[d] < token_ && lay.dyn_used[d] < oldest) { oldest = lay.dyn_used[d]; victim = d; }
        }
        if (victim == lay.dynamics) { ++stats_.no_slot; continue; }
        if (!waited) {
            // the table staging of the previous update must be consumed, and this token's layer done with the slots
            STRIX_CUDA_CHECK(cudaEventSynchronize(lay.refilled));
            if (done_flag_dev) {
                const CUresult r = cuStreamWaitValue32(reinterpret_cast<CUstream>(refill_), reinterpret_cast<CUdeviceptr>(done_flag_dev), sequence,
                                                       CU_STREAM_WAIT_VALUE_GEQ);
                if (r != CUDA_SUCCESS) throw std::runtime_error("ds41 hot residency: cuStreamWaitValue32 failed");
            } else {
                STRIX_CUDA_CHECK(cudaStreamWaitEvent(refill_, layer_done, 0));
            }
            waited = true;
        }
        if (lay.dyn_expert[victim] >= 0) lay.rl.host_slot_of[static_cast<std::size_t>(lay.dyn_expert[victim])] = -1;
        const std::uint64_t slot = lay.statics + victim;
        const auto* src = lay.source[e];
        STRIX_CUDA_CHECK(cudaMemcpyAsync(lay.gate.get() + slot * gb, src, gb, cudaMemcpyHostToDevice, refill_));
        STRIX_CUDA_CHECK(cudaMemcpyAsync(lay.up.get() + slot * gb, src + gb, gb, cudaMemcpyHostToDevice, refill_));
        STRIX_CUDA_CHECK(cudaMemcpyAsync(lay.down.get() + slot * db, src + 2U * gb, db, cudaMemcpyHostToDevice, refill_));
        lay.dyn_expert[victim] = static_cast<std::int32_t>(e);
        lay.dyn_used[victim] = token_;
        lay.rl.host_slot_of[e] = static_cast<std::int32_t>(slot);
        ++stats_.copies;
        --copies_left_;
        any = true;
    }
    if (!any) return;
    std::copy(lay.rl.host_slot_of.begin(), lay.rl.host_slot_of.end(), lay.staging);
    STRIX_CUDA_CHECK(cudaMemcpyAsync(lay.d_slot_of.get(), lay.staging, shape_.experts * 4U, cudaMemcpyHostToDevice, refill_));
    STRIX_CUDA_CHECK(cudaEventRecord(lay.refilled, refill_));
    ++gen_[l];
    if (cuStreamWriteValue32(reinterpret_cast<CUstream>(refill_), reinterpret_cast<CUdeviceptr>(refill_done_dev_ + l), gen_[l],
                             CU_STREAM_WRITE_VALUE_DEFAULT) != CUDA_SUCCESS)
        throw std::runtime_error("ds41 hot residency: cuStreamWriteValue32 failed");
    reinterpret_cast<volatile std::uint32_t*>(required_)[l] = gen_[l];
    lay.pending = true;
    (void)cudaStreamQuery(refill_);  // flush the submission
}

}  // namespace strix::ds41::cuda
