// GLM engine: the MoE seam on the kernels (glm_fast_moe.hpp). The engine computes the route (R5/R14:
// ids ascending + weights) before the seam, so this calls moe::enqueue_experts (routed gate_up + down + R16 combine,
// shared expert included) with those ids / weights: the same per-row arithmetic as DefaultMoe's strix_gemv chain.
// Each layer's device ExpertSlices[288] array is built once from the raw slices and cached by layer.
#include "glm/engine/glm_fast_moe.hpp"

#include "glm/engine/glm_lib.hpp"
#include "glm/moe/glm_moe.hpp"

#include <map>
#include <stdexcept>
#include <vector>

namespace strix::glm::engine {

namespace {

using namespace detail;
namespace M = strix::glm::moe;

// i4: a fast_i4 routed expert (trellis freed; the W4A16 kernels read only its suh / svh).
M::Exl3 view(const Linear& l, bool i4 = false) {
    if (l.raw.trellis == nullptr && !(i4 && l.raw.suh != nullptr && l.raw.svh != nullptr))
        throw std::logic_error("fast MoE: expert not loaded raw");
    return M::Exl3{l.raw.trellis, l.raw.suh, l.raw.svh, l.raw.input, l.raw.output, l.raw.bits};
}
M::ExpertSlices slices(const Mlp& m, bool i4 = false) { return M::ExpertSlices{view(m.gate, i4), view(m.up, i4), view(m.down, i4)}; }

class FastMoe final : public MoePath {
public:
    FastMoe() {
        ws_.base = dzero<std::uint8_t>(M::moe_workspace_bytes());  // zeroed once (arrival counters reset themselves)
    }
    ~FastMoe() override {
        (void)hipFree(ws_.base);
        for (auto& [k, p] : experts_) (void)hipFree(p);
    }
    bool publishes() const override { return true; }
    void forward(const MoeLayerWeights& w, const CallCtx& c, const float* x, const std::uint32_t* ids, const float* wts,
                 float* y) override {
        const auto R = static_cast<unsigned>(c.rows.size());
        if (R == 0 || R > M::kMaxRows) throw std::invalid_argument("fast MoE: 1..8 rows");
        M::MoeView v;
        v.rows = R;
        v.y = x;
        v.ids = const_cast<std::uint32_t*>(ids);  // read only by enqueue_experts
        v.weights = const_cast<float*>(wts);
        v.experts = device_experts(w);
        v.shared = slices(w.shared);
        v.out = y;
        v.ws = ws_;
        v.publish = c.publish;
        if (w.i4.on()) {  // stored routed INT4: W4A16 for every row regime
            for (unsigned m = 0; m < 3; ++m) {
                v.i4.wq[m] = w.i4.wq[m];
                v.i4.ws[m] = w.i4.ws[m];
            }
            M::enqueue_experts(v, c.stream);
            return;
        }
        if (c.decode && M::pipe_enabled()) M::enqueue_experts_pipe(v, c.stream);
        else M::enqueue_experts(v, c.stream);
    }

    std::uint64_t warm_bytes(const MoeLayerWeights& w) const override {  // one hipMalloc per layer table (2 MiB granule) + its host copy
        constexpr std::uint64_t kGranule = std::uint64_t{2} << 20;
        const std::uint64_t table = M::kExperts * sizeof(M::ExpertSlices);
        return experts_.count(&w) != 0 ? 0 : (table + kGranule - 1) / kGranule * kGranule + table;
    }
    void warm(const MoeLayerWeights& w) override { (void)device_experts(w); }

private:
    const M::ExpertSlices* device_experts(const MoeLayerWeights& w) {
        auto it = experts_.find(&w);
        if (it != experts_.end()) return it->second;
        if (w.experts.size() != M::kExperts) throw std::logic_error("fast MoE: 288 experts expected");
        std::vector<M::ExpertSlices> h;
        h.reserve(M::kExperts);
        for (const Mlp& e : w.experts) h.push_back(slices(e, w.i4.on()));
        auto* d = dalloc<M::ExpertSlices>(h.size());
        try {
            hip_ok(hipMemcpy(d, h.data(), h.size() * sizeof(M::ExpertSlices), hipMemcpyHostToDevice), "expert slices");
            experts_.emplace(&w, d);
        } catch (...) {
            (void)hipFree(d);
            throw;
        }
        return d;
    }
    M::MoeWorkspace ws_{};
    std::map<const MoeLayerWeights*, M::ExpertSlices*> experts_;
};

}  // namespace

std::shared_ptr<MoePath> fast_moe() { return std::make_shared<FastMoe>(); }

}  // namespace strix::glm::engine
