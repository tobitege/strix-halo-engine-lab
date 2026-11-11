#pragma once
#include "strix/ctx_exact_memory.hpp"
#include <exception>
#include <mutex>
#include <utility>

namespace strix {
struct QsaScratchBinding {
    int device=-1;
    std::uintptr_t stream=0;
    std::uint64_t generation=0, maximum=0;
    bool operator==(const QsaScratchBinding&) const = default;
};
inline void require_qsa_score_source(bool compact, bool bound) {
    if(compact && !bound)
        throw std::invalid_argument("compact QSA workspace has no bound score ring; use the full-prefill entry");
}
inline bool qsa_maxima_selector_enabled(unsigned mask) { return (mask & 2U)!=0; }
inline float* qsa_selector_maxima(unsigned mask,float* storage) {
    return qsa_maxima_selector_enabled(mask)?storage:nullptr;
}

// Used by native scratch and the CPU failure-injection fixtures. Backend calls
// throw on failure. An unsuccessful consumer event is replaced by a stream
// drain; if neither proves completion, fail-stop before any owner can free.
template<class Backend> class QsaScoreScratch {
public:
    using Event=typename Backend::Event;
    struct Slot { void* data=nullptr; Event done{}; bool pending=false, leased=false; };
private:
    Backend backend_;
    QsaScratchBinding binding_;
    QsaScoreRingGeometry geometry_;
    std::array<Slot,2> slots_{};
    unsigned next_=0;
    void release() noexcept {
        try {
            const int prior=backend_.device();
            backend_.activate(binding_.device);
            for(auto& s:slots_) {
                if(s.leased) backend_.drain(binding_.stream);
                else if(s.pending) backend_.wait(s.done);
                if(s.done) backend_.destroy(s.done);
                if(s.data) backend_.free(s.data);
                s={};
            }
            backend_.activate(prior);
        } catch(...) { std::terminate(); }
    }
public:
    QsaScoreScratch(Backend backend,QsaScratchBinding binding,std::uint64_t admitted,std::uint32_t rows=0)
        :backend_(std::move(backend)),binding_(binding),geometry_(binding.maximum,4U,rows) {
        if(binding.device<0 || !binding.stream || !binding.generation || !binding.maximum ||
            backend_.device()!=binding.device || admitted!=geometry_.total_bytes())
            throw std::invalid_argument("QSA scratch setup binding/admission differs");
        try {
            for(auto& s:slots_) {
                s.data=backend_.allocate(geometry_.score_slot_bytes()+geometry_.maxima_slot_bytes());
                s.done=backend_.event();
            }
        } catch(...) { release(); throw; }
    }
    ~QsaScoreScratch() { release(); }
    QsaScoreScratch(const QsaScoreScratch&)=delete;
    QsaScoreScratch& operator=(const QsaScoreScratch&)=delete;
    void validate(QsaScratchBinding binding) const {
        if(binding!=binding_ || backend_.device()!=binding_.device)
            throw std::invalid_argument("QSA scratch device/stream/generation/capacity differs from setup");
    }
    const QsaScoreRingGeometry& geometry() const { return geometry_; }
    unsigned acquire() {
        auto& s=slots_[next_];
        if(s.leased) throw std::logic_error("QSA scratch slot is already leased");
        if(s.pending) backend_.wait(s.done);
        s.pending=false;s.leased=true;
        const auto result=next_;next_=(next_+1U)%2U;return result;
    }
    void consumed(unsigned slot) noexcept {
        auto& s=slots_[slot];
        if(!s.leased) std::terminate();
        try { backend_.record(s.done,binding_.stream);s.pending=true; }
        catch(...) {
            try { backend_.drain(binding_.stream);s.pending=false; }
            catch(...) { std::terminate(); }
        }
        s.leased=false;
    }
    void* data(unsigned slot) const { return slots_[slot].data; }
    class Lease {
        QsaScoreScratch& owner_;
    public:
        const unsigned slot;
        explicit Lease(QsaScoreScratch& owner):owner_(owner),slot(owner.acquire()) {}
        ~Lease() { owner_.consumed(slot); }
        Lease(const Lease&)=delete;
        Lease& operator=(const Lease&)=delete;
    };
};

// The existing stage/pool/attend descriptor caches share a submission domain.
// Keep this lock even when key pooling is disabled. Completion handoff protects
// the device consumers after the host submission lock has been released.
class QsaScratchSerialGate {
    std::mutex mutex_;
    int device_=-1;
public:
    auto enter(int device) {
        std::unique_lock<std::mutex> lock(mutex_);
        if(device_!=-1 && device_!=device)
            throw std::invalid_argument("QSA shared prefill caches require one device per process");
        device_=device;return lock;
    }
};
class QsaScratchHandoff {
    std::uintptr_t stream_=0;
    bool pending_=false;
public:
    template<class Wait> void before(std::uintptr_t stream,Wait wait) {
        // Same-stream work is already ordered. Cross-stream reuse of shared
        // descriptors/pools requires completion, including host-side writes.
        if(pending_ && stream_!=stream) { wait();pending_=false; }
    }
    void recorded(std::uintptr_t stream) noexcept { stream_=stream;pending_=true; }
    void drained() noexcept { pending_=false; }
    bool pending() const noexcept { return pending_; }
    // The stream of the last recorded submission still owes its completion: it must not be destroyed before.
    bool pending_on(std::uintptr_t stream) const noexcept { return pending_ && stream_==stream; }
};
template<class Attend,class Select,class Index>
void clear_qsa_pending(Attend& b,Select& sb,Index& ib,float*& maxima,std::uint32_t& stride) noexcept {
    b.active=false;b.groups.clear();b.kv=nullptr;b.union_at=0;
    sb.active=false;sb.rows.clear();sb.traced.clear();sb.used=0;
    ib.active=false;ib.groups.clear();
    maxima=nullptr;stride=0;
}
template<class Drain,class Reset> class QsaPendingGuard {
    Drain drain_;
    Reset reset_;
public:
    QsaPendingGuard(Drain drain,Reset reset):drain_(std::move(drain)),reset_(std::move(reset)) {}
    ~QsaPendingGuard() noexcept {
        try { drain_();reset_(); } catch(...) { std::terminate(); }
    }
    QsaPendingGuard(const QsaPendingGuard&)=delete;
    QsaPendingGuard& operator=(const QsaPendingGuard&)=delete;
};
} // namespace strix
