#pragma once
#include "strix/deploy/route.hpp"
#include "strix/minimax_layercut.hpp"
namespace strix::deploy {
// Registration adapter: the deployment caller supplies the model's actual footprint provider.
// There is no global model registry in route.cpp. Unknown range footprints continue to refuse admission.
class MinimaxLayercutProvider final : public RouteProvider {
    const FootprintProvider& fp_;
public:
    explicit MinimaxLayercutProvider(const FootprintProvider& fp) : fp_(fp) {}
    const FootprintProvider& footprints() const override { return fp_; }
    std::uint32_t layers() const override { return minimax::layercut::layers; }
    std::optional<std::uint64_t> seam_bytes(std::uint32_t cut, std::uint64_t) const override {
        if (cut == 0 || cut >= layers()) return std::nullopt;
        return minimax::layercut::seam_bytes;
    }
    std::uint64_t state_bytes(std::uint64_t rows, std::uint32_t begin, std::uint32_t end) const override {
        if (begin > end || end > layers() || rows > UINT64_MAX / minimax::layercut::state_bytes / layers())
            throw std::invalid_argument("minimax layercut state range/overflow");
        return rows * (end - begin) * minimax::layercut::state_bytes;
    }
    std::vector<Protocol> protocols() const override {
        Protocol p;
        p.kind = "layers"; p.name = minimax::layercut::format; p.identity = "mm-layercut-v1";
        p.stages = {"cuda-fp32id", "minimax-pair-default"}; p.chunks = {256, 512, 1024, 2048, 4096};
        p.gate = "pending"; // no measured import cost / range footprints: never selected by default
        return {p};
    }
};
} // namespace strix::deploy
