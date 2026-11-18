#pragma once
// Shared VerifyTarget over any engine with the window contract (CUDA CudaEngine, HIP HipEngine):
//   forward_rows(tokens, rows, LogitRows) -> Out; read_top(Out, rows, host) (synchronizes); commit(k); sync().
// verify = forward_rows(all) + per-row argmax; commit(k) = engine commit + sync (the step wall is complete).
// Home: shared (host spec layer): no device types; each backend binds it with its own engine type.
#include <cstdint>
#include <functional>
#include <span>
#include <utility>
#include <vector>

#include "strix/spec/block_drafter.hpp"

namespace strix::spec {

template <class Engine>
class EngineVerifyTarget final : public VerifyTarget {
public:
    explicit EngineVerifyTarget(Engine& e) : e_(e) {}
    // Optional observer of every verify pass (rows of [anchor, drafts...] + the engine's per-row outputs): the MTP chain
    // (strix/spec/mtp_chain.hpp) takes the hidden rows of the pass from it. Needs Engine::Rows.
    using Observer = std::function<void(const typename Engine::Rows&, std::uint32_t)>;
    void set_observer(Observer f) { observer_ = std::move(f); }
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        const auto rows = static_cast<std::uint32_t>(tokens.size());
        const auto out = e_.forward_rows(tokens.data(), rows, Engine::LogitRows::all);
        greedy.resize(rows);
        e_.read_top(out, rows, greedy.data());
        if (observer_) observer_(out, rows);
    }
    void commit(std::uint32_t k) override {
        e_.commit(k);
        e_.sync();
    }
    Engine& engine() { return e_; }

private:
    Engine& e_;
    Observer observer_;
};

}  // namespace strix::spec
