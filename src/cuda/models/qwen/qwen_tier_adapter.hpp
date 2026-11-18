// Qwen adapter of tiered expert residency (strix/cuda/tiered_experts.hpp + strix/residency/expert_residency.hpp):
// the stored expert tensors (memory-mapped GGUF), the per-layer compact device slot buffers (slot S = zero expert), the
// device tables, the CPU executor + worker, and the between-steps planner/mover. Included by qwen_prefill_driver.cu only.
#pragma once

#include "strix/cuda/tiered_experts.hpp"
#include "strix/env_flags.hpp"
#include "strix/file_map.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/residency/expert_residency.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strix::cuda::qwen_prefill {

// Model-specific layout; checked byte arithmetic belongs to shared residency code.
inline void validate_qwen_expert(const gguf::TensorInfo& t, std::uint32_t kind,
                                 std::uint32_t hidden, std::uint32_t ffn, std::uint32_t experts) {
    const auto geometry = gguf::tensor_type_geometry(t.ggml_type);
    if (!geometry) throw std::invalid_argument("tiered residency: unsupported expert format");
    residency::validate_expert_tensor(t.dimensions, t.byte_size, kind == 2U ? ffn : hidden,
        kind == 2U ? hidden : ffn, experts, geometry->block_elements, geometry->block_bytes);
}

// Physical RAM the process could still use (Windows: available = free + standby cache; POSIX: _SC_AVPHYS_PAGES).
inline std::uint64_t host_available_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX st{};
    st.dwLength = sizeof(st);
    return GlobalMemoryStatusEx(&st) ? static_cast<std::uint64_t>(st.ullAvailPhys) : 0ULL;
#else
    const long pages = sysconf(_SC_AVPHYS_PAGES), size = sysconf(_SC_PAGESIZE);
    return pages > 0 && size > 0 ? static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(size) : 0ULL;
#endif
}

class QwenTier {
public:
    struct Init {
        std::vector<std::string> parts;
        std::string tables_dir;
        residency::Decision decision;
        residency::ModelBytes model;
        tiered::TierLayout layout;
    };
    // Per layer, per tensor (gate, up, down): where the stored bytes are.
    struct Stored { std::uint32_t part = 0; std::uint64_t offset = 0, bytes = 0; std::uint32_t type = 0; bool found = false; };

    explicit QwenTier(Init init) : in_(std::move(init)) {
        const auto& L = in_.layout;
        stored_.assign(std::size_t{in_.model.layers} * 3U, {});
        for (std::uint32_t p = 0; p < in_.parts.size(); ++p) {
            const auto info = gguf::read_file(in_.parts[p]);
            bool any = false;
            for (const auto& t : info.tensors) {
                const auto at = t.name.find("blk.");
                if (at != 0 || t.name.find("_exps.weight") == std::string::npos) continue;
                const auto layer = static_cast<std::uint32_t>(std::stoul(t.name.substr(4)));
                if (layer >= in_.model.layers) continue;
                const std::uint32_t kind = t.name.find("ffn_gate_exps") != std::string::npos ? 0U
                    : t.name.find("ffn_up_exps") != std::string::npos ? 1U
                    : t.name.find("ffn_down_exps") != std::string::npos ? 2U : 3U;
                if (kind == 3U) continue;
                validate_qwen_expert(t, kind, L.hidden, L.ffn, L.experts);
                if (stored_[std::size_t{layer} * 3U + kind].found)
                    throw std::invalid_argument("tiered residency: duplicate expert tensor");
                stored_[std::size_t{layer} * 3U + kind] = {p, t.absolute_offset, t.byte_size, t.ggml_type, true};
                any = true;
            }
            maps_.emplace_back();
            if (any) maps_.back().open(in_.parts[p]);
        }
        const auto S = in_.decision.slots_per_layer;
        if (S == 0U || S >= L.experts) throw std::runtime_error("tiered residency: slots per layer out of range (" + std::to_string(S) + ")");
        per_expert_.assign(std::size_t{in_.model.layers} * 3U, 0);
        std::vector<cpu::expert::LayerWeights> host(in_.model.layers);
        for (std::uint32_t l = 0; l < in_.model.layers; ++l) {
            for (std::uint32_t k = 0; k < 3U; ++k) {
                const auto& s = stored_[std::size_t{l} * 3U + k];
                if (!s.found || s.bytes % L.experts) throw std::runtime_error("tiered residency: missing expert tensor of layer " + std::to_string(l));
                per_expert_[std::size_t{l} * 3U + k] = s.bytes / L.experts;
            }
            const auto& g = stored_[std::size_t{l} * 3U];
            const auto& u = stored_[std::size_t{l} * 3U + 1U];
            const auto& d = stored_[std::size_t{l} * 3U + 2U];
            if ((g.type != 12U && g.type != 13U) || g.type != u.type || (d.type != 7U && d.type != 8U))
                throw std::runtime_error("tiered residency: layer " + std::to_string(l) + " expert types are not the A8 profile's "
                                         "(gate/up Q4_K|Q5_K, down Q5_1|Q8_0): the host executor does not cover them");
            auto& w = host[l];
            w.gate = maps_[g.part].data() + g.offset;
            w.up = maps_[u.part].data() + u.offset;
            w.down = maps_[d.part].data() + d.offset;
            w.gate_format = g.type == 12U ? cpu::expert::GateFormat::q4_k : cpu::expert::GateFormat::q5_k;
            w.down_format = d.type == 7U ? cpu::expert::DownFormat::q5_1 : cpu::expert::DownFormat::q8_0;
            w.shape = {L.hidden, L.ffn};
        }
        exp_ = std::make_unique<cpu::expert::Gfx1151Exp>(in_.tables_dir);
        const unsigned hw = std::max(1U, std::thread::hardware_concurrency());
        const int want = strix::envcfg::integer("STRIX_TIER_CPU_THREADS", static_cast<int>(std::min(16U, hw)));
        executor_ = std::make_unique<cpu::expert::Executor>(static_cast<unsigned>(std::max(1, want)), *exp_);
        exchange_ = std::make_unique<tiered::Exchange>(L);
        worker_ = std::make_unique<tiered::TierWorker>(*exchange_, std::move(host), *executor_);
        planner_ = std::make_unique<residency::Planner>(in_.model, S);
    }
    ~QwenTier() {
        worker_.reset();  // joins before the exchange goes
        for (auto* p : buffers_) if (p) cudaFree(p);
        if (staging_) cudaFreeHost(staging_);
        if (stream_) cudaStreamDestroy(stream_);
        if (event_) cudaEventDestroy(event_);
        if (h_counts_) cudaFreeHost(h_counts_);
    }

    [[nodiscard]] std::uint32_t slots() const { return in_.decision.slots_per_layer; }
    [[nodiscard]] std::uint32_t zero_slot() const { return in_.decision.slots_per_layer; }
    [[nodiscard]] const tiered::TierLayout& layout() const { return in_.layout; }
    [[nodiscard]] const tiered::Exchange& exchange() const { return *exchange_; }
    [[nodiscard]] const char* isa() const { return executor_->isa(); }
    [[nodiscard]] unsigned cpu_threads() const { return executor_->threads(); }

    // Device slot buffers: (slots + 1) experts per tensor, zero-filled (the last one stays zero forever), the id -> slot
    // tables (all -1) and the step usage counters.
    void alloc_slots() {
        const auto& L = in_.layout;
        const std::uint32_t n = in_.model.layers;
        slot_.assign(std::size_t{n} * 3U, nullptr);
        std::uint64_t resident = 0;
        for (std::uint32_t l = 0; l < n; ++l)
            for (std::uint32_t k = 0; k < 3U; ++k) {
                const std::uint64_t bytes = per_expert_[std::size_t{l} * 3U + k] * (std::uint64_t{slots()} + 1U);
                std::uint8_t* p = nullptr;
                tiered::require_budget(bytes, "expert slot");
                ck(cudaMalloc(reinterpret_cast<void**>(&p), bytes), "slot buffer");
                ck(cudaMemset(p, 0, bytes), "slot clear");
                buffers_.push_back(p);
                slot_[std::size_t{l} * 3U + k] = p;
                resident += bytes;
            }
        tiered::require_budget(std::uint64_t{n} * L.experts * 8U, "expert tables");
        ck(cudaMalloc(reinterpret_cast<void**>(&table_), std::uint64_t{n} * L.experts * 4U), "table");
        ck(cudaMemset(table_, 0xFF, std::uint64_t{n} * L.experts * 4U), "table clear");
        buffers_.push_back(table_);
        ck(cudaMalloc(reinterpret_cast<void**>(&counts_), std::uint64_t{n} * L.experts * 4U), "counts");
        ck(cudaMemset(counts_, 0, std::uint64_t{n} * L.experts * 4U), "counts clear");
        buffers_.push_back(counts_);
        ck(cudaMallocHost(reinterpret_cast<void**>(&h_counts_), std::uint64_t{n} * L.experts * 4U), "pinned counts");
        std::uint64_t biggest = 0;
        for (std::uint32_t l = 0; l < n; ++l)
            biggest = std::max(biggest, per_expert_[std::size_t{l} * 3U] + per_expert_[std::size_t{l} * 3U + 1U] + per_expert_[std::size_t{l} * 3U + 2U]);
        staging_bytes_ = std::uint64_t{64} * 1024 * 1024 + biggest;  // PlannerConfig::upload_bytes_per_step + one expert
        ck(cudaMallocHost(reinterpret_cast<void**>(&staging_), staging_bytes_), "pinned staging");
        ck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "tier stream");
        ck(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming), "tier event");
        resident_slot_bytes_ = resident;
    }
    [[nodiscard]] const void* slot_gate(std::uint32_t l) const { return slot_[std::size_t{l} * 3U]; }
    [[nodiscard]] const void* slot_up(std::uint32_t l) const { return slot_[std::size_t{l} * 3U + 1U]; }
    [[nodiscard]] const void* slot_down(std::uint32_t l) const { return slot_[std::size_t{l} * 3U + 2U]; }
    [[nodiscard]] std::uint64_t slot_buffer_bytes() const { return resident_slot_bytes_; }

    void publish(std::uint32_t layer, std::uint32_t rows, std::uint32_t pad, const std::uint32_t* routes, std::uint32_t* slot_routes,
                 const std::int8_t* codes, const float* deltas, const std::int32_t* sums, cudaStream_t s) {
        tiered::enqueue_publish(exchange_->dev(), layer, rows, pad, routes, slot_routes,
                                table_ + std::uint64_t{layer} * in_.layout.experts, zero_slot(), codes, deltas, sums,
                                counts_ + std::uint64_t{layer} * in_.layout.experts, s);
    }
    void wait_gather(std::uint32_t rows, float* rawdots, std::uint64_t rawdot_stride, std::uint32_t* status, std::uint64_t status_stride,
                     std::uint32_t* summary, cudaStream_t s) {
        tiered::enqueue_wait_gather(exchange_->dev(), rows, rawdots, rawdot_stride, status, status_stride, summary, s);
    }

    // After a step/prefill whose stream is synchronized: read the step's expert usage, update the planner, move experts.
    // The first call seeds the planner from its routes (a prefill's, normally) and fills the slots from the seed.
    void after_step(cudaStream_t stream) {
        const auto& L = in_.layout;
        const std::uint32_t n = in_.model.layers;
        const std::uint64_t words = std::uint64_t{n} * L.experts;
        ck(cudaMemcpyAsync(h_counts_, counts_, words * 4U, cudaMemcpyDeviceToHost, stream), "counts D2H");
        ck(cudaMemsetAsync(counts_, 0, words * 4U, stream), "counts clear");
        ck(cudaStreamSynchronize(stream), "counts sync");
        const std::lock_guard<std::mutex> lock(mu_);  // /api/status reads the receipt from another thread
        const auto t0 = Clock::now();
        std::vector<std::uint32_t> ids;
        if (!seeded_) {
            bool any = false;
            for (std::uint64_t i = 0; i < words && !any; ++i) any = h_counts_[i] != 0U;
            if (!any) return;  // nothing routed yet
            for (std::uint32_t l = 0; l < n; ++l) planner_->seed(l, h_counts_ + std::uint64_t{l} * L.experts);
            planner_->fill_from_seed();
            seeded_ = true;
            std::vector<residency::Move> fill;
            for (std::uint32_t l = 0; l < n; ++l) {
                const auto& table = planner_->table(l);
                for (std::uint32_t e = 0; e < L.experts; ++e)
                    if (table[e] >= 0) fill.push_back({l, static_cast<std::uint32_t>(table[e]), residency::Move::kNone, e});
            }
            move_batch(fill, stream, false);
        } else {
            for (std::uint32_t l = 0; l < n; ++l) {
                ids.clear();
                for (std::uint32_t e = 0; e < L.experts; ++e)
                    if (h_counts_[std::uint64_t{l} * L.experts + e]) ids.push_back(e);
                planner_->observe(l, ids.data(), static_cast<std::uint32_t>(ids.size()));
            }
            planner_->end_step();
            const auto moves = planner_->plan();
            if (!moves.empty()) move_batch(moves, stream, true);
        }
        upload_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }

    // Highest position written by any session so far (receipt: context actually filled).
    void note_context(std::uint64_t filled) {
        const std::lock_guard<std::mutex> lock(mu_);
        if (filled > context_filled_) context_filled_ = filled;
    }

    [[nodiscard]] residency::Receipt receipt() const {
        const std::lock_guard<std::mutex> lock(mu_);
        auto r = planner_->receipt();
        r.cpu_ms = worker_->stats().cpu_ms;
        r.upload_ms = upload_ms_;
        r.context_filled = static_cast<std::uint32_t>(context_filled_);
        return r;
    }
    [[nodiscard]] std::string receipt_json() const {
        const auto st = worker_->stats();
        auto json = residency::receipt_json(receipt());
        if (!json.empty() && json.back() == '}') json.pop_back();
        return json + ",\"cpu_jobs\":" + std::to_string(st.jobs) + ",\"cpu_misses\":" + std::to_string(st.misses) +
               ",\"cpu_isa\":\"" + executor_->isa() + "\",\"cpu_threads\":" + std::to_string(executor_->threads()) +
               ",\"worker_failed\":" + (worker_->failed() ? "true" : "false") + "}";
    }

private:
    using Clock = std::chrono::steady_clock;
    static void ck(cudaError_t e, const char* what) {
        if (e != cudaSuccess) throw std::runtime_error(std::string("tiered residency: ") + what + ": " + cudaGetErrorString(e));
    }
    // evict: table[evicted] = -1, then the bytes (mapping -> pinned staging -> slot), then table[new] = slot, all ordered on
    // the tier stream; the engine stream waits for the last event. `batches`: split by the staging size.
    void move_batch(const std::vector<residency::Move>& moves, cudaStream_t engine, bool commit) {
        const auto& L = in_.layout;
        std::uint64_t used = 0;
        std::vector<residency::Move> pending;
        const auto flush = [&] {
            if (pending.empty()) return;
            ck(cudaStreamSynchronize(stream_), "tier stream");
            if (commit) for (const auto& m : pending) planner_->commit(m);
            pending.clear();
            used = 0;
        };
        for (const auto& m : moves) {
            std::uint64_t total = 0;
            for (std::uint32_t k = 0; k < 3U; ++k) total += per_expert_[std::size_t{m.layer} * 3U + k];
            if (used + total > staging_bytes_) flush();
            const std::int32_t off = -1, slot = static_cast<std::int32_t>(m.slot), id = static_cast<std::int32_t>(m.expert);
            std::int32_t* entry = table_ + std::uint64_t{m.layer} * L.experts;
            if (m.evict != residency::Move::kNone)
                ck(cudaMemcpyAsync(entry + m.evict, &off, 4, cudaMemcpyHostToDevice, stream_), "table evict");
            for (std::uint32_t k = 0; k < 3U; ++k) {
                const auto& s = stored_[std::size_t{m.layer} * 3U + k];
                const std::uint64_t bytes = per_expert_[std::size_t{m.layer} * 3U + k];
                std::memcpy(staging_ + used, maps_[s.part].data() + s.offset + std::uint64_t{m.expert} * bytes, static_cast<std::size_t>(bytes));
                ck(cudaMemcpyAsync(slot_[std::size_t{m.layer} * 3U + k] + std::uint64_t{m.slot} * bytes, staging_ + used, bytes,
                                   cudaMemcpyHostToDevice, stream_), "slot upload");
                used += bytes;
            }
            ck(cudaMemcpyAsync(entry + id, &slot, 4, cudaMemcpyHostToDevice, stream_), "table set");
            pending.push_back(m);
        }
        flush();
        ck(cudaEventRecord(event_, stream_), "tier event");
        ck(cudaStreamWaitEvent(engine, event_, 0), "tier wait");
    }

    Init in_;
    std::vector<Stored> stored_;
    std::vector<std::uint64_t> per_expert_;
    std::vector<strix::FileMap> maps_;
    std::unique_ptr<cpu::expert::Gfx1151Exp> exp_;
    std::unique_ptr<cpu::expert::Executor> executor_;
    std::unique_ptr<tiered::Exchange> exchange_;
    std::unique_ptr<tiered::TierWorker> worker_;
    std::unique_ptr<residency::Planner> planner_;
    std::vector<std::uint8_t*> slot_;
    std::vector<void*> buffers_;
    std::int32_t* table_ = nullptr;
    std::uint32_t* counts_ = nullptr;
    std::uint32_t* h_counts_ = nullptr;
    std::uint8_t* staging_ = nullptr;
    std::uint64_t staging_bytes_ = 0, resident_slot_bytes_ = 0;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t event_ = nullptr;
    bool seeded_ = false;
    double upload_ms_ = 0;
    std::uint64_t context_filled_ = 0;
    mutable std::mutex mu_;
};

}  // namespace strix::cuda::qwen_prefill
