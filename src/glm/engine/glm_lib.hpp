#pragma once
// GLM engine: strix-trellis linears + O_DIRECT staging (the MiniMax engine's Lib), shared by the engine and the G3
// independent replay (tests/glm_g3_replay.hip): H128-aligned rank slices assembled in one pinned staging buffer,
// prepared strix_weights (strix_gemv / strix_gemm = the Tier B reference) and optional raw device copies.
// Dry run (design notes): while a LoadPlan is set, every call records the device allocations, staging and file
// bytes it would use and returns null handles; nothing is read and nothing is allocated.
#include "glm/engine/glm_mem.hpp"
#include "strix/pinned_host.hpp"
#include "glm/engine/glm_pack.hpp"
#include "strix/glm/engine.hpp"

#include <hip/hip_runtime.h>
#include <strix_trellis.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <tuple>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::glm::engine::detail {

inline void hip_ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
inline void lib_ok(strix_status s, const std::string& what) {
    if (s != STRIX_SUCCESS) throw std::runtime_error("trellis " + what + ": " + strix_status_string(s));
}
// STRIX_GLM_ALLOC_LOG=1 (diagnostic, R >= 3 fault (review finding)): every engine device allocation's VA range on stderr,
// to map the kernel log's faulting page to its owner.
inline void alloc_log(const void* p, std::size_t bytes, const char* what) {
    static const bool on = std::getenv("STRIX_GLM_ALLOC_LOG") != nullptr && std::string(std::getenv("STRIX_GLM_ALLOC_LOG")) == "1";
    if (on) std::fprintf(stderr, "alloc-log: %p-%p %zu B %s\n", p, static_cast<const void*>(static_cast<const char*>(p) + bytes), bytes, what);
}
template <class T> inline T* dalloc(std::size_t n) {
    T* p = nullptr;
    hip_ok(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "malloc");
    alloc_log(p, std::max<std::size_t>(n, 1) * sizeof(T), "dalloc");
    return p;
}
template <class T> inline T* dzero(std::size_t n) {
    T* p = dalloc<T>(n);
    hip_ok(hipMemset(p, 0, std::max<std::size_t>(n, 1) * sizeof(T)), "memset");
    return p;
}

class Lib final : public LinearRunner {
public:
    Lib(PackReader& pack, bool keep_raw) : pack_(pack), keep_raw_(keep_raw) {
        strix_context_desc d{};
        d.struct_size = sizeof(d);
        d.abi_version = STRIX_TRELLIS_ABI_VERSION;
        d.device_ordinal = 0;
        lib_ok(strix_context_create(&d, &ctx_), "context");
        st_.struct_size = sizeof(st_);
        st_.abi_version = STRIX_TRELLIS_ABI_VERSION;
        st_.hip_stream = nullptr;
    }
    ~Lib() override {
        (void)hipDeviceSynchronize();
        for (auto w : all_) (void)strix_weights_destroy(w);
        for (void* p : arenas_) (void)hipFree(p);
        if (ws_) (void)hipFree(ws_);
        release_staging();
        (void)strix_context_destroy(ctx_);
    }
    struct Staged {
        const std::uint8_t *tr = nullptr, *suh = nullptr, *svh = nullptr;
        std::uint64_t tr_bytes = 0, suh_bytes = 0, svh_bytes = 0, rows = 0, cols = 0, bits = 0;
    };
    // Output slice [o0, o0 + n): those outputs' tiles of every input row + the matching svh (H128-aligned).
    Staged stage_out(const std::string& prefix, std::uint64_t o0, std::uint64_t n) {
        const TensorInfo &tr = pack_.info(prefix + ".trellis"), &suh = pack_.info(prefix + ".suh"), &svh = pack_.info(prefix + ".svh");
        if (o0 % 128 != 0 || n % 128 != 0 || o0 + n > svh.shape[0]) throw std::invalid_argument("bad output slice " + prefix);
        const std::uint64_t in_tiles = tr.shape[0], out_tiles = tr.shape[1], w = tr.shape[2] * 2;
        const std::uint64_t tr_bytes = in_tiles * (n / 16) * w;
        if (plan_) {
            plan_->stage_max = std::max(plan_->stage_max, tr_bytes + suh.bytes + n * 2);
            if (!(o0 == 0 && n == svh.shape[0])) plan_->whole_max = std::max(plan_->whole_max, tr.bytes);
            plan_->bytes_read += tr.bytes + suh.bytes + n * 2;
            return {nullptr, nullptr, nullptr, tr_bytes, suh.bytes, n * 2, n, suh.shape[0], tr.shape[2] / 16};
        }
        std::uint8_t* t = stage(tr_bytes + suh.bytes + n * 2);
        if (i4_mode_ == 2) {
            // fast_i4 pack hit: no trellis bytes are read (copy_raw leaves raw.trellis null)
        } else if (o0 == 0 && n == svh.shape[0]) {
            pack_.read(tr, 0, tr.bytes, t);
        } else {
            if (whole_.size() < tr.bytes) whole_.resize(tr.bytes);
            pack_.read(tr, 0, tr.bytes, whole_.data());
            for (std::uint64_t i = 0; i < in_tiles; ++i)
                std::memcpy(t + i * (n / 16) * w, whole_.data() + (i * out_tiles + o0 / 16) * w, (n / 16) * w);
        }
        pack_.read(suh, 0, suh.bytes, t + tr_bytes);
        pack_.read(svh, o0 * 2, n * 2, t + tr_bytes + suh.bytes);
        return {t, t + tr_bytes, t + tr_bytes + suh.bytes, tr_bytes, suh.bytes, n * 2, n, suh.shape[0], tr.shape[2] / 16};
    }
    // Input slice [i0, i0 + n): the contiguous trellis rows + the matching suh (a partial sum across ranks).
    Staged stage_in(const std::string& prefix, std::uint64_t i0, std::uint64_t n) {
        const TensorInfo &tr = pack_.info(prefix + ".trellis"), &suh = pack_.info(prefix + ".suh"), &svh = pack_.info(prefix + ".svh");
        if (i0 % 128 != 0 || n % 128 != 0 || i0 + n > suh.shape[0]) throw std::invalid_argument("bad input slice " + prefix);
        const std::uint64_t row_bytes = tr.shape[1] * tr.shape[2] * 2, tr_bytes = (n / 16) * row_bytes;
        if (plan_) {
            plan_->stage_max = std::max(plan_->stage_max, tr_bytes + n * 2 + svh.bytes);
            plan_->bytes_read += tr_bytes + n * 2 + svh.bytes;
            return {nullptr, nullptr, nullptr, tr_bytes, n * 2, svh.bytes, svh.shape[0], n, tr.shape[2] / 16};
        }
        std::uint8_t* t = stage(tr_bytes + n * 2 + svh.bytes);
        if (i4_mode_ != 2) pack_.read(tr, (i0 / 16) * row_bytes, tr_bytes, t);
        pack_.read(suh, i0 * 2, n * 2, t + tr_bytes);
        pack_.read(svh, 0, svh.bytes, t + tr_bytes + n * 2);
        return {t, t + tr_bytes, t + tr_bytes + n * 2, tr_bytes, n * 2, svh.bytes, svh.shape[0], n, tr.shape[2] / 16};
    }
    Linear out_slice(const std::string& p, std::uint64_t o0, std::uint64_t n) { return linear(p, stage_out(p, o0, n)); }
    Linear in_slice(const std::string& p, std::uint64_t i0, std::uint64_t n) { return linear(p, stage_in(p, i0, n)); }
    Linear whole(const std::string& p) { return out_slice(p, 0, pack_.info(p + ".svh").shape[0]); }
    // Raw-only slices (no strix_weights prepare): the parts in the file layout, packed in the engine arena (1 GiB chunks,
    // so no per-tensor allocation granule). For seams that read raw EXL3 (MoE kernels); prepared = nullptr.
    Linear out_slice_raw(const std::string& p, std::uint64_t o0, std::uint64_t n) { return raw_only(stage_out(p, o0, n)); }
    Linear in_slice_raw(const std::string& p, std::uint64_t i0, std::uint64_t n) { return raw_only(stage_in(p, i0, n)); }
    Linear whole_raw(const std::string& p) { return out_slice_raw(p, 0, pack_.info(p + ".svh").shape[0]); }
    // Gives a raw-only Linear a prepared handle from its own device raw parts (copied back to the host, then
    // strix_weights_prepare_parts exactly as the loader does for the file parts); the raw pointers stay. No-op if prepared.
    void prepare_from_raw(Linear& l, const std::string& what) {
        if (l.prepared != nullptr) return;
        if (l.raw.trellis == nullptr) throw std::logic_error("prepare_from_raw: no raw parts: " + what);
        Staged g;
        g.rows = l.raw.output;
        g.cols = l.raw.input;
        g.bits = l.raw.bits;
        g.tr_bytes = raw_trellis_bytes(l.raw);
        g.suh_bytes = g.cols * 2;
        g.svh_bytes = g.rows * 2;
        std::vector<std::uint8_t> h(g.tr_bytes + g.suh_bytes + g.svh_bytes);
        hip_ok(hipMemcpy(h.data(), l.raw.trellis, g.tr_bytes, hipMemcpyDeviceToHost), "raw trellis back");
        hip_ok(hipMemcpy(h.data() + g.tr_bytes, l.raw.suh, g.suh_bytes, hipMemcpyDeviceToHost), "raw suh back");
        hip_ok(hipMemcpy(h.data() + g.tr_bytes + g.suh_bytes, l.raw.svh, g.svh_bytes, hipMemcpyDeviceToHost), "raw svh back");
        g.tr = h.data();
        g.suh = h.data() + g.tr_bytes;
        g.svh = h.data() + g.tr_bytes + g.suh_bytes;
        const bool keep = keep_raw_;  // the raw parts already exist on the device: no second arena copy
        keep_raw_ = false;
        try {
            l.prepared = linear(what, g).prepared;
        } catch (...) {
            keep_raw_ = keep;
            throw;
        }
        keep_raw_ = keep;
    }
    static std::uint64_t raw_trellis_bytes(const Exl3Raw& r) { return std::uint64_t{r.output} * r.input / 8 * r.bits; }

    // Rows [r0, r0 + n) of a plain [rows][row_bytes] tensor to device memory (n = 0: all).
    void* upload_rows(const std::string& name, std::uint64_t r0 = 0, std::uint64_t n = 0) {
        const TensorInfo& t = pack_.info(name);
        const std::uint64_t rows = t.shape.empty() ? 1 : t.shape[0], row_bytes = t.bytes / rows;
        if (n == 0) n = rows - r0;
        if (r0 + n > rows) throw std::invalid_argument("bad row range " + name);
        std::uint8_t* d = static_cast<std::uint8_t*>(arena(n * row_bytes));
        if (!plan_) upload(t, r0 * row_bytes, n * row_bytes, d);
        else plan_upload(n * row_bytes);
        return d;
    }
    // Concatenation of row ranges of one plain tensor.
    void* upload_row_ranges(const std::string& name, const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges) {
        const TensorInfo& t = pack_.info(name);
        const std::uint64_t row_bytes = t.bytes / t.shape[0];
        std::uint64_t total = 0;
        for (const auto& r : ranges) total += r.second * row_bytes;
        std::uint8_t* d = static_cast<std::uint8_t*>(arena(total));
        std::uint64_t off = 0;
        for (const auto& r : ranges) {
            if (!plan_) upload(t, r.first * row_bytes, r.second * row_bytes, d + off);
            else plan_upload(r.second * row_bytes);
            off += r.second * row_bytes;
        }
        return d;
    }
    const std::string& dtype(const std::string& name) const { return pack_.info(name).dtype; }
    // Dry run on / off (see the header comment). Arena packing restarts from an empty chunk for the plan.
    void set_plan(mem::LoadPlan* p) {
        plan_ = p;
        plan_arena_left_ = 0;
        plan_i4_temp_ = false;
        if (p) p->bounce = PackReader::bounce_bytes();
    }
    bool planning() const { return plan_ != nullptr; }
    // Pinned staging size (default 1 GiB; the engine sets the plan's largest request) and its allocation up front, so
    // the memory checks see it before the first weight.
    void set_stage_bytes(std::uint64_t b) {
        if (b == 0 || b > kStage) throw std::invalid_argument("staging size");
        if (stage_) throw std::logic_error("staging already allocated");
        stage_bytes_ = b;
    }
    std::uint64_t stage_bytes() const { return stage_bytes_; }
    void reserve_staging() { (void)stage(1); }
    // Device buffers that are not weights (KV / recurrent state zeroed, activations uninitialised), planned like them.
    template <class T> T* state(std::size_t n) {
        const std::size_t b = std::max<std::size_t>(n, 1) * sizeof(T);
        if (plan_) {
            plan_->add(b);
            return nullptr;
        }
        allocated_ += b;
        recorded_.push_back(b);
        return dzero<T>(n);
    }
    template <class T> T* buffer(std::size_t n) {
        const std::size_t b = std::max<std::size_t>(n, 1) * sizeof(T);
        if (plan_) {
            plan_->add(b);
            return nullptr;
        }
        allocated_ += b;
        recorded_.push_back(b);
        // STRIX_GLM_BUF_ZERO=1 (diagnostic, R >= 3 fault): touch every page of the buffer at allocation (zeroed)
        static const bool zero = std::getenv("STRIX_GLM_BUF_ZERO") != nullptr && std::string(std::getenv("STRIX_GLM_BUF_ZERO")) == "1";
        return zero ? dzero<T>(n) : dalloc<T>(n);
    }
    // Requested bytes of every real device allocation made through the Lib (weights, arenas, state, buffers), and the
    // list of them in order (compared with the dry run's LoadPlan::device after the load).
    std::uint64_t allocated_bytes() const { return allocated_; }
    const std::vector<std::uint64_t>& recorded() const { return recorded_; }
    void release_staging() {
        if (stage_) (void)strix::host_pin::release(stage_);
        stage_ = nullptr;
        std::vector<std::uint8_t>().swap(whole_);
    }
    void run(const Linear& l, const float* in, float* out, unsigned rows, hipStream_t s) const override {
        if (rows == 0) return;
        strix_matmul_desc d{};
        strix_matmul_desc_init(&d);
        d.weights = static_cast<strix_weights>(l.prepared);
        set(d.input, const_cast<float*>(in), rows, cols_of(l));
        set(d.output, out, rows, rows_of(l));
        std::uint64_t bytes = 0;
        lib_ok(strix_matmul_workspace_bytes(ctx_, &d, &bytes), "workspace");
        // STRIX_GLM_WS_GUARD=1 (diagnostic, R >= 3 fault (review finding)): the workspace is followed by a 1 MiB guard of
        // 0xA5 bytes, re-checked after every call (stream sync); every new (N, K, rows) logs its workspace bytes.
        static const bool guard = std::getenv("STRIX_GLM_WS_GUARD") != nullptr && std::string(std::getenv("STRIX_GLM_WS_GUARD")) == "1";
        constexpr std::uint64_t kGuard = 1ULL << 20;
        if (guard) {
            static std::set<std::tuple<std::uint64_t, std::uint64_t, unsigned>> seen;
            if (seen.insert({rows_of(l), cols_of(l), rows}).second)
                std::fprintf(stderr, "ws-guard: N %llu K %llu rows %u workspace %llu B (allocated %llu)\n",
                             static_cast<unsigned long long>(rows_of(l)), static_cast<unsigned long long>(cols_of(l)), rows,
                             static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(ws_bytes_));
        }
        if (bytes > ws_bytes_) {
            hip_ok(hipDeviceSynchronize(), "sync");
            if (ws_) hip_ok(hipFree(ws_), "free");
            hip_ok(hipMalloc(&ws_, bytes + (guard ? kGuard : 0)), "workspace");
            alloc_log(ws_, bytes + (guard ? kGuard : 0), "strix workspace");
            ws_bytes_ = bytes;
            if (guard) hip_ok(hipMemset(static_cast<char*>(ws_) + bytes, 0xA5, kGuard), "guard");
        }
        d.workspace = bytes ? ws_ : nullptr;
        d.workspace_bytes = bytes ? ws_bytes_ : 0;
        strix_stream st = st_;
        st.hip_stream = s;
        lib_ok(rows <= 8 ? strix_gemv(ctx_, &d, &st) : strix_gemm(ctx_, &d, &st), rows <= 8 ? "gemv" : "gemm");
        if (guard && ws_) {
            hip_ok(hipStreamSynchronize(s), "guard sync");
            std::vector<std::uint8_t> g(kGuard);
            hip_ok(hipMemcpy(g.data(), static_cast<char*>(ws_) + ws_bytes_, kGuard, hipMemcpyDeviceToHost), "guard read");
            for (std::uint64_t i = 0; i < kGuard; ++i)
                if (g[i] != 0xA5) {
                    std::fprintf(stderr, "ws-guard: OVERRUN at workspace + %llu (+%llu past %llu B): N %llu K %llu rows %u\n",
                                 static_cast<unsigned long long>(ws_bytes_ + i), static_cast<unsigned long long>(i),
                                 static_cast<unsigned long long>(ws_bytes_), static_cast<unsigned long long>(rows_of(l)),
                                 static_cast<unsigned long long>(cols_of(l)), rows);
                    throw std::runtime_error("strix workspace overrun");
                }
        }
    }
    // Prefill identity glm-pf-wmma-v1 (design notes section 1.1): strix_gemm_pinned with a caller-pinned
    // K split; the WMMA path for every rows >= 1. Needs a strix-trellis with STRIX_TRELLIS_HAS_GEMM_PINNED.
    // Refuses unless `bytes` + margin fit both hipMemGetInfo free and MemAvailable (unreadable MemAvailable refuses).
    // Return the same checked spare for opportunistic temporary sizing. Existing
    // fixed-size callers ignore it; no second, less conservative memory source.
    static std::uint64_t admit(std::uint64_t bytes, std::uint64_t margin, const char* what,
                               std::uint64_t* device_free_out = nullptr, std::uint64_t* mem_available_out = nullptr) {
        std::size_t dev_free = 0, dev_total = 0;
        hip_ok(hipMemGetInfo(&dev_free, &dev_total), "mem info");
        std::uint64_t host_avail = 0;
        {
            std::ifstream f("/proc/meminfo");
            std::string key, unit;
            std::uint64_t kb = 0;
            bool found = false;
            while (!found && f >> key >> kb >> unit)
                if (key == "MemAvailable:" && unit == "kB") {
                    host_avail = kb * 1024;
                    found = true;
                }
            if (!found) throw std::runtime_error("workspace reserve: cannot read MemAvailable (refusing to allocate unchecked)");
        }
        if (dev_free < bytes || dev_free - bytes < margin || host_avail < bytes || host_avail - bytes < margin)
            throw std::runtime_error(std::string(what) + ": refusing " + std::to_string(bytes >> 20) + " MiB (+ margin " + std::to_string(margin >> 20) +
                                     " MiB): device free " + std::to_string(dev_free >> 20) + " MiB, MemAvailable " +
                                     std::to_string(host_avail >> 20) + " MiB");
        if (device_free_out) *device_free_out = dev_free;
        if (mem_available_out) *mem_available_out = host_avail;
        return std::min<std::uint64_t>(dev_free, host_avail) - margin;
    }
    // Fails closed: refuses unless the new buffer + margin fit both hipMemGetInfo free and the host's MemAvailable
    // (unified memory on gfx1151) while the current buffer still exists (the transient peak); MemAvailable unreadable
    // refuses too. On any failure the current workspace stays as it was.
    void reserve(std::uint64_t bytes, std::uint64_t margin) const {
        if (bytes <= ws_bytes_) return;
        hip_ok(hipDeviceSynchronize(), "sync");
        admit(bytes, margin, "workspace reserve");
        void* fresh = nullptr;
        hip_ok(hipMalloc(&fresh, bytes), "workspace reserve");
        if (ws_) {
            const hipError_t e = hipFree(ws_);
            if (e != hipSuccess) {
                (void)hipFree(fresh);
                hip_ok(e, "free");
            }
        }
        ws_ = fresh;
        ws_bytes_ = bytes;
    }
    void reserve(std::uint64_t bytes) const override { reserve(bytes, 0); }
    // Largest strix_matmul_workspace_bytes of any prepared linear at 1..rows rows (the GEMV calls of decode, of the
    // gemv-mode prefill and of the head), computed once per rows value (host-only queries).
    std::uint64_t gemv_workspace_bound(unsigned rows, std::uint64_t margin) const {
        if (rows == bound_rows_) return bound_;
        // The query validates descriptors only (non-null, aligned, non-overlapping, never dereferenced): two scratch
        // device buffers sized for the widest input / output at `rows` rows stand in, freed on return.
        std::uint64_t in_w = 0, out_w = 0;
        for (const auto& [w, d] : dims_) {
            in_w = std::max(in_w, d.second);
            out_w = std::max(out_w, d.first);
        }
        struct Scratch {
            void* p = nullptr;
            ~Scratch() {
                if (p) (void)hipFree(p);
            }
        } sin, sout;
        admit((std::max<std::uint64_t>(1, rows * in_w) + std::max<std::uint64_t>(1, rows * out_w)) * sizeof(float), margin, "workspace bound scratch");
        hip_ok(hipMalloc(&sin.p, std::max<std::uint64_t>(1, rows * in_w) * sizeof(float)), "bound scratch");
        hip_ok(hipMalloc(&sout.p, std::max<std::uint64_t>(1, rows * out_w) * sizeof(float)), "bound scratch");
        std::uint64_t m = 0;
        for (const auto& [w, d] : dims_)
            for (unsigned r = 1; r <= rows; ++r) {
                strix_matmul_desc q{};
                strix_matmul_desc_init(&q);
                q.weights = w;
                set(q.input, static_cast<float*>(sin.p), r, d.second);
                set(q.output, static_cast<float*>(sout.p), r, d.first);
                std::uint64_t b = 0;
                lib_ok(strix_matmul_workspace_bytes(ctx_, &q, &b), "workspace bound");
                m = std::max(m, b);
            }
        bound_rows_ = rows;
        bound_ = m;
        return m;
    }
    void run_pinned(const Linear& l, const float* in, float* out, unsigned rows, unsigned splits, hipStream_t s) const override {
#ifdef STRIX_TRELLIS_HAS_GEMM_PINNED
        if (rows == 0) return;
        strix_matmul_desc d{};
        strix_matmul_desc_init(&d);
        d.weights = static_cast<strix_weights>(l.prepared);
        set(d.input, const_cast<float*>(in), rows, cols_of(l));
        set(d.output, out, rows, rows_of(l));
        d.input.rank = 2;
        d.output.rank = 2;
        std::uint64_t bytes = 0;
        lib_ok(strix_gemm_pinned_workspace_bytes(ctx_, &d, splits, &bytes), "pinned workspace");
        if (bytes > ws_bytes_) {
            hip_ok(hipDeviceSynchronize(), "sync");
            if (ws_) hip_ok(hipFree(ws_), "free");
            hip_ok(hipMalloc(&ws_, bytes), "workspace");
            ws_bytes_ = bytes;
        }
        d.workspace = bytes ? ws_ : nullptr;
        d.workspace_bytes = bytes ? ws_bytes_ : 0;
        strix_stream st = st_;
        st.hip_stream = s;
        lib_ok(strix_gemm_pinned(ctx_, &d, splits, &st), "gemm_pinned");
#else
        (void)l, (void)in, (void)out, (void)rows, (void)splits, (void)s;
        throw std::logic_error("this strix-trellis build has no strix_gemm_pinned (glm-pf-wmma-v1 needs it)");
#endif
    }
    void* arena(std::uint64_t bytes) {
        bytes = (bytes + 255) & ~std::uint64_t{255};
        if (plan_) {
            if (plan_arena_left_ < bytes) {
                const std::uint64_t chunk = std::max<std::uint64_t>(bytes, std::uint64_t{1} << 30);
                plan_->add(chunk);
                plan_arena_left_ = chunk;
            }
            plan_arena_left_ -= bytes;
            return nullptr;
        }
        if (arena_left_ < bytes) {
            const std::uint64_t chunk = std::max<std::uint64_t>(bytes, std::uint64_t{1} << 30);
            void* p = nullptr;
            hip_ok(hipMalloc(&p, chunk), "arena");
            alloc_log(p, chunk, "arena");
            allocated_ += chunk;
            recorded_.push_back(chunk);
            arenas_.push_back(p);
            arena_next_ = static_cast<std::uint8_t*>(p);
            arena_left_ = chunk;
        }
        std::uint8_t* r = arena_next_;
        arena_next_ += bytes;
        arena_left_ -= bytes;
        device_bytes_ += bytes;
        return r;
    }
    double device_gib() const { return static_cast<double>(device_bytes_) / 1073741824.0; }
    // One exact device allocation (planned as such; freed with the arenas). fast_i4 INT4 layer packs.
    void* dedicated(std::uint64_t bytes, const char* what) {
        bytes = (bytes + 255) & ~std::uint64_t{255};
        if (plan_) {
            plan_->add(bytes);
            return nullptr;
        }
        void* p = nullptr;
        hip_ok(hipMalloc(&p, bytes), what);
        alloc_log(p, bytes, what);
        allocated_ += bytes;
        recorded_.push_back(bytes);
        arenas_.push_back(p);
        device_bytes_ += bytes;
        return p;
    }
    // fast_i4 load (STRIX_GLM_MODEL_FORMAT=fast_i4): between i4_routed_begin / _end the raw slices put their trellis
    // into ONE reusable layer buffer (planned once, freed by i4_temp_release after the load), suh / svh in the arena as
    // usual. skip_trellis: the layer's INT4 pack is read from STRIX_GLM_FAST_I4_PACK, the trellis is not read at all
    // (raw.trellis = nullptr). The allocations are the same either way (plan parity).
    void i4_routed_begin(std::uint64_t layer_trellis_bytes, bool skip_trellis) {
        const bool have = plan_ ? plan_i4_temp_ : i4_temp_ != nullptr;
        if (!have) {
            void* p = dedicated(layer_trellis_bytes, "fast_i4 routed EXL3 layer buffer");
            if (plan_) plan_i4_temp_ = true;
            else i4_temp_ = static_cast<std::uint8_t*>(p);
            i4_temp_bytes_ = (layer_trellis_bytes + 255) & ~std::uint64_t{255};
        }
        i4_cursor_ = 0;
        i4_mode_ = skip_trellis ? 2 : 1;
    }
    void i4_routed_end() { i4_mode_ = 0; }
    // Scope gate_up: the routed down of a converted layer is an ordinary resident raw slice.
    void i4_suspend(bool on) {
        if (on) {
            i4_saved_ = i4_mode_;
            i4_mode_ = 0;
        } else {
            i4_mode_ = i4_saved_;
        }
    }
    // After the load: returns the layer buffer's bytes to the device (the plan counted it as resident: conservative).
    std::uint64_t i4_temp_release() {
        if (!i4_temp_) return 0;
        hip_ok(hipDeviceSynchronize(), "fast_i4 layer buffer sync");
        for (auto it = arenas_.begin(); it != arenas_.end(); ++it)
            if (*it == i4_temp_) {
                arenas_.erase(it);
                break;
            }
        hip_ok(hipFree(i4_temp_), "fast_i4 layer buffer free");
        i4_temp_ = nullptr;
        device_bytes_ -= i4_temp_bytes_;
        return i4_temp_bytes_;
    }
    bool i4_skip_trellis() const { return i4_mode_ == 2; }
    PackReader& pack() { return pack_; }
    std::uint8_t* staging(std::uint64_t bytes) { return stage(bytes); }

private:
    static constexpr std::size_t kStage = std::size_t{1} << 30;  // > the largest slice (full lm_head on one node, ~476 MB)
    std::uint8_t* stage(std::uint64_t bytes) {
        if (bytes > stage_bytes_) throw std::runtime_error("staging buffer too small");
        if (!stage_) hip_ok(strix::host_pin::allocate(reinterpret_cast<void**>(&stage_), stage_bytes_, hipHostMallocDefault), "staging");
        return stage_;
    }
    void plan_upload(std::uint64_t n) {
        plan_->bytes_read += n;
        plan_->stage_max = std::max(plan_->stage_max, std::min<std::uint64_t>(n, std::uint64_t{256} << 20));
    }
    void upload(const TensorInfo& t, std::uint64_t off, std::uint64_t n, std::uint8_t* d) {
        constexpr std::uint64_t chunk = std::uint64_t{256} << 20;
        for (std::uint64_t o = 0; o < n; o += chunk) {
            const std::uint64_t m = std::min(chunk, n - o);
            std::uint8_t* s = stage(m);
            pack_.read(t, off + o, m, s);
            hip_ok(hipMemcpy(d + o, s, m, hipMemcpyHostToDevice), "upload");
        }
    }
    Linear linear(const std::string& what, const Staged& g) {
        if (plan_) {  // strix_weights_prepare_parts = ONE hipMalloc of the 256 B-aligned regions (prepare_regions)
            plan_->add(mem::prepared_bytes(g.tr_bytes, g.suh_bytes, g.svh_bytes, 0, STRIX_TRELLIS_WORKSPACE_ALIGNMENT));
            if (keep_raw_) (void)arena(g.tr_bytes + 256 + g.suh_bytes + 256 + g.svh_bytes);
            Linear l;
            l.raw.input = static_cast<std::uint32_t>(g.cols);
            l.raw.output = static_cast<std::uint32_t>(g.rows);
            l.raw.bits = static_cast<std::uint32_t>(g.bits);
            return l;
        }
        strix_exl3_parts_desc p{};
        p.struct_size = sizeof(p);
        p.abi_version = STRIX_TRELLIS_ABI_VERSION;
        p.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
        p.bits_x2 = static_cast<std::uint32_t>(2 * g.bits);
        p.rows = g.rows;
        p.columns = g.cols;
        p.codebook = STRIX_CODEBOOK_MUL1;
        p.marker = STRIX_EXL3_MARKER_MUL1;
        p.trellis = {g.tr, g.tr_bytes};
        p.suh = {g.suh, g.suh_bytes};
        p.svh = {g.svh, g.svh_bytes};
        Linear l;
        strix_weights w = nullptr;
        lib_ok(strix_weights_prepare_parts(ctx_, &p, &st_, &w), "prepare " + what);
        all_.push_back(w);
        device_bytes_ += g.tr_bytes + g.suh_bytes + g.svh_bytes;  // the library's device copy of the parts (lower bound)
        allocated_ += mem::prepared_bytes(g.tr_bytes, g.suh_bytes, g.svh_bytes, 0, STRIX_TRELLIS_WORKSPACE_ALIGNMENT);
        recorded_.push_back(mem::prepared_bytes(g.tr_bytes, g.suh_bytes, g.svh_bytes, 0, STRIX_TRELLIS_WORKSPACE_ALIGNMENT));
        l.prepared = w;
        l.raw.input = static_cast<std::uint32_t>(g.cols);
        l.raw.output = static_cast<std::uint32_t>(g.rows);
        l.raw.bits = static_cast<std::uint32_t>(g.bits);
        dims_[w] = {g.rows, g.cols};
        bound_rows_ = 0;  // a new linear: the cached GEMV workspace bound is stale
        if (keep_raw_) copy_raw(l, g);
        return l;
    }
    Linear raw_only(const Staged& g) {
        Linear l;
        l.raw.input = static_cast<std::uint32_t>(g.cols);
        l.raw.output = static_cast<std::uint32_t>(g.rows);
        l.raw.bits = static_cast<std::uint32_t>(g.bits);
        copy_raw(l, g);
        return l;
    }
    void copy_raw(Linear& l, const Staged& g) {
        if (i4_mode_ != 0) {  // fast_i4 routed expert: trellis -> the layer buffer (or nothing), suh / svh -> arena
            auto* d_suh = static_cast<std::uint8_t*>(arena(g.suh_bytes + 256 + g.svh_bytes));
            const std::uint64_t tr = (g.tr_bytes + 255) & ~std::uint64_t{255};
            if (i4_cursor_ + tr > i4_temp_bytes_) throw std::logic_error("fast_i4: routed layer larger than its EXL3 buffer");
            std::uint8_t* d = i4_mode_ == 1 && !plan_ ? i4_temp_ + i4_cursor_ : nullptr;
            i4_cursor_ += tr;
            if (plan_) return;
            std::uint8_t* d_svh = d_suh + ((g.suh_bytes + 255) & ~std::uint64_t{255});
            if (d) hip_ok(hipMemcpy(d, g.tr, g.tr_bytes, hipMemcpyHostToDevice), "raw trellis (fast_i4 layer buffer)");
            hip_ok(hipMemcpy(d_suh, g.suh, g.suh_bytes, hipMemcpyHostToDevice), "raw suh");
            hip_ok(hipMemcpy(d_svh, g.svh, g.svh_bytes, hipMemcpyHostToDevice), "raw svh");
            l.raw.trellis = reinterpret_cast<const std::uint32_t*>(d);
            l.raw.suh = reinterpret_cast<const std::uint16_t*>(d_suh);
            l.raw.svh = reinterpret_cast<const std::uint16_t*>(d_svh);
            return;
        }
        auto* d = static_cast<std::uint8_t*>(arena(g.tr_bytes + 256 + g.suh_bytes + 256 + g.svh_bytes));
        if (plan_) return;  // dry run (glm_mem): the arena bytes are planned, nothing is copied
        std::uint8_t* d_suh = d + ((g.tr_bytes + 255) & ~std::uint64_t{255});
        std::uint8_t* d_svh = d_suh + ((g.suh_bytes + 255) & ~std::uint64_t{255});
        hip_ok(hipMemcpy(d, g.tr, g.tr_bytes, hipMemcpyHostToDevice), "raw trellis");
        hip_ok(hipMemcpy(d_suh, g.suh, g.suh_bytes, hipMemcpyHostToDevice), "raw suh");
        hip_ok(hipMemcpy(d_svh, g.svh, g.svh_bytes, hipMemcpyHostToDevice), "raw svh");
        l.raw.trellis = reinterpret_cast<const std::uint32_t*>(d);
        l.raw.suh = reinterpret_cast<const std::uint16_t*>(d_suh);
        l.raw.svh = reinterpret_cast<const std::uint16_t*>(d_svh);
    }
    std::uint64_t rows_of(const Linear& l) const { return dims_.at(static_cast<strix_weights>(l.prepared)).first; }
    std::uint64_t cols_of(const Linear& l) const { return dims_.at(static_cast<strix_weights>(l.prepared)).second; }
    static void set(strix_tensor_desc& t, float* p, std::uint64_t rows, std::uint64_t width) {
        t.data = p;
        t.buffer_bytes = rows * width * sizeof(float);
        t.dtype = STRIX_DTYPE_F32;
        t.rank = 2;
        t.shape[0] = rows;
        t.shape[1] = width;
        t.strides[0] = width;
        t.strides[1] = 1;
    }
    PackReader& pack_;
    bool keep_raw_;
    strix_context ctx_ = nullptr;
    strix_stream st_{};
    mutable void* ws_ = nullptr;
    mutable std::uint64_t ws_bytes_ = 0;
    mutable unsigned bound_rows_ = 0;
    mutable std::uint64_t bound_ = 0;
    std::uint8_t* stage_ = nullptr;
    std::uint64_t stage_bytes_ = kStage;
    std::vector<std::uint8_t> whole_;
    mem::LoadPlan* plan_ = nullptr;
    std::uint64_t plan_arena_left_ = 0, allocated_ = 0;
    std::vector<std::uint64_t> recorded_;
    std::vector<strix_weights> all_;
    std::map<strix_weights, std::pair<std::uint64_t, std::uint64_t>> dims_;
    std::vector<void*> arenas_;
    std::uint8_t* arena_next_ = nullptr;
    std::uint64_t arena_left_ = 0, device_bytes_ = 0;
    std::uint8_t* i4_temp_ = nullptr;  // fast_i4 routed EXL3 layer buffer (load only)
    std::uint64_t i4_temp_bytes_ = 0, i4_cursor_ = 0;
    int i4_saved_ = 0;
    int i4_mode_ = 0;  // 0 off, 1 trellis into i4_temp_, 2 skip the trellis (pack hit)
    bool plan_i4_temp_ = false;
};


}  // namespace strix::glm::engine::detail
