#pragma once

#include "strix/serve/queued_prefix_protection.hpp"
#include "strix/serve/session_runtime.hpp"

namespace strix::qwen {

// A request-owner adapter, not a native slot manager. Every native callback
// carries a logical generation; its slot number NEVER addresses native state.
// The existing runner independently selects its first free physical slot.
class HttpSessionOwner {
public:
    struct Callbacks {
        std::function<bool(std::span<const std::uint32_t>)> tokens;
        std::function<void(serve::GenerationProgress)> progress;
        std::function<bool()> cancelled;
        std::function<void()> before_submit;
        std::function<void(serve::GenerationStats, std::string)> finish;
    };
    // Translation happens once per admission, receipt formatting once per
    // completion. submit may synchronously publish, finish, or reenter pump.
    using Submit = std::function<void(const serve::GenerationJob&, Callbacks)>;
    using Pump = std::function<void(const Submit&, const std::function<bool()>&)>;
    struct Native {
        std::function<void(bool)> add_slot;
        std::function<bool()> has_free_slot;
        std::function<std::size_t()> active;
        Submit submit;
        std::function<void(std::function<std::size_t()>, Pump)> service_admission;
        std::function<void()> step;
        std::function<void(const std::string&)> fail_requests;
        std::function<void()> abandon;
        // Named execution uses the same logical publication/completion route;
        // GenerationRequestError retains the complete fixed-protocol receipt.
        std::function<serve::GenerationStats(serve::GenerationJob&)> named;
        std::function<bool()> named_failed;
        std::function<bool()> stop_requested;
        std::function<void()> failed;
        std::mutex* session_mutex = nullptr;
        // Optional cache binding of the queue protection (STRIX_PREFIX_QUEUE_PROTECT, default OFF).
        serve::QueuedPrefixProtection::Binding queue_protection;
    };

    HttpSessionOwner(serve::GenerationQueue&, std::uint64_t context,
                     std::uint32_t capacity, Native);
    ~HttpSessionOwner();
    HttpSessionOwner(const HttpSessionOwner&) = delete;
    HttpSessionOwner& operator=(const HttpSessionOwner&) = delete;
    void run();
    serve::SessionRuntime& runtime() noexcept { return runtime_; }
    [[nodiscard]] const serve::QueuedPrefixProtection::Stats& queue_protection_stats() const noexcept { return protection_.stats(); }

private:
    class ControlEngine final : public session::Engine {
    public:
        ControlEngine(std::uint64_t, std::uint32_t);
        void prefill(std::span<const std::uint32_t>, session::KvHandle&) override;
        session::StepResult decode(session::KvHandle&, const session::SamplerState&) override;
        session::Capabilities caps() const override;
        ::strix::profile::ExecutionReceipt receipt() const override;
    private:
        session::Capabilities caps_;
    };
    static session::Hooks logical_hooks();
    serve::SessionHooks owner_hooks();
    void turn();
    void pump(const Submit&, const std::function<bool()>&);
    void admit(const Submit&);
    void named(serve::SessionLease);

    serve::GenerationQueue& queue_; // read-only queue hints; all takes/finishes are in runtime_
    const std::uint64_t context_;
    Native native_;
    ControlEngine engine_;
    session::KvStore leases_; // logical generations only, no model state or native reset hook
    std::unique_lock<std::mutex> batch_lock_;
    std::string failure_;
    bool drained_ = false;
    enum class Turn { growth, admission, work } turn_ = Turn::growth;
    serve::SessionRuntime runtime_;
    serve::QueuedPrefixProtection protection_;
};

} // namespace strix::qwen
