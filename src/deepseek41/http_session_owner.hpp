#pragma once

#include "strix/serve/session_runtime.hpp"

#include <memory>
#include <stdexcept>

namespace strix::deepseek41 {
// Request-only construction seam. The family keeps its finite native control
// turns and physical planner; Runtime owns all taken/pending/active requests.
class HttpSessionOwner {
public:
    HttpSessionOwner(serve::GenerationQueue&, std::uint64_t context, std::uint32_t slots,
                     std::function<void(serve::SessionRuntime&)>, std::function<bool()> drained,
                     std::function<void()> failed, std::function<void()> stopped,
                     bool embed_rows = false);  // true: the native turns resolve image rows (--vision)
    serve::SessionRuntime& runtime() noexcept { return runtime_; }
private:
    class ControlEngine final : public session::Engine {
    public:
        ControlEngine(std::uint64_t, std::uint32_t, bool embed_rows);
        void prefill(std::span<const std::uint32_t>, session::KvHandle&) override;
        session::StepResult decode(session::KvHandle&, const session::SamplerState&) override;
        session::Capabilities caps() const override { return caps_; }
        // The native turns read GenerationJob::embeds themselves (Task resolves each piece); the lane only records it.
        void bind_embeds(session::KvHandle&, std::shared_ptr<const session::EmbedTable> table) override {
            if (table && !caps_.embed_rows) throw std::invalid_argument("engine does not support embedding rows");
        }
        profile::ExecutionReceipt receipt() const override { return {}; }
    private:
        session::Capabilities caps_;
    } engine_;
    session::KvStore leases_;
    serve::SessionRuntime runtime_;
};
} // namespace strix::deepseek41
