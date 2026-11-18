#pragma once
#include "strix/cuda/qwen_route_bf16.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <filesystem>
#include <fstream>

namespace strix::cuda::qwen_mtp_diagnostic {
// A diagnostic capability, never a product admission or environment override.
inline void admission(bool armed, bool candidate, bool decode, bool exporting,
                      bool sidecar, bool graphs) {
    if (!armed) return;
    if (!candidate || !decode || exporting || !sidecar || graphs)
        throw std::invalid_argument("BF16 MTP diagnostic requires candidate/decode/sidecar/eager/no-export");
}
inline constexpr const char* drafter_identity = "cuda-qwen-mtp-q8-production-v1";
struct Binding {
    std::uint64_t target=0, engine=0, session=0, capacity=0, sidecar=0, phase=1;
    bool operator==(const Binding&) const = default;
};
inline void binding(const Binding& got, const Binding& expected) {
    if (!(got==expected)) throw std::invalid_argument("MTP diagnostic stale/target/sidecar/cap/session/phase binding");
}
inline std::uint64_t payload_bytes(std::uint64_t rows, std::uint64_t vocab,
                                   std::uint64_t limit) {
    if (!rows || rows>8 || !vocab || vocab>UINT64_MAX/4/rows)
        throw std::invalid_argument("MTP diagnostic row payload extent");
    const auto bytes=rows*vocab*4;
    if (bytes>limit) throw std::invalid_argument("MTP diagnostic row payload budget");
    return bytes;
}
// Same longest-prefix/pending rule as the production CUDA serving scheduler.
// Inputs are consumed rows, outputs are target predictions after those rows.
struct Selection { std::uint32_t keep=0, pending=0; bool stopped=false; };
inline Selection select(const std::vector<std::uint32_t>& rows,
                        const std::vector<std::uint32_t>& argmax,
                        std::uint32_t budget, std::uint32_t stop=UINT32_MAX) {
    if (rows.empty() || rows.size()>8 || rows.size()!=argmax.size() || !budget)
        throw std::invalid_argument("MTP diagnostic selection extent/budget");
    std::size_t keep=1;
    while (keep<rows.size() && rows[keep]==argmax[keep-1]) ++keep;
    keep=std::min(keep,static_cast<std::size_t>(budget));
    bool stopped=false;
    for (std::size_t i=0;i<keep;++i) if (argmax[i]==stop) { keep=i+1;stopped=true;break; }
    return {static_cast<std::uint32_t>(keep),argmax[keep-1],stopped};
}
inline void observe(const std::vector<float>& raw, const std::vector<std::uint32_t>& ids,
                    std::size_t rows, std::size_t vocab) {
    if (!rows || rows>8 || !vocab || vocab>SIZE_MAX/rows || raw.size()!=rows*vocab || ids.size()!=rows)
        throw std::invalid_argument("MTP diagnostic full row coverage");
    for (std::size_t r=0;r<rows;++r) {
        std::size_t best=0;
        for (std::size_t i=0;i<vocab;++i) {
            const auto v=raw[r*vocab+i];
            if (!std::isfinite(v)) throw std::invalid_argument("MTP diagnostic nonfinite raw row");
            if (v>raw[r*vocab+best]) best=i;
        }
        if (ids[r]!=best) throw std::invalid_argument("MTP diagnostic nonargmax row");
    }
}
// Final marker is published only after successful flush/close of its temp file.
// Incomplete raw exports remain inspectable but never acquire this marker.
inline void publish_complete(const std::filesystem::path& directory, const std::string& text) {
    const auto final=directory/"COMPLETE.txt", temp=directory/"COMPLETE.pending";
    if (text.empty() || std::filesystem::exists(final) || std::filesystem::exists(temp))
        throw std::invalid_argument("diagnostic publication duplicate/empty");
    std::ofstream f(temp,std::ios::binary|std::ios::trunc);
    if (!f) throw std::runtime_error("diagnostic publication open");
    f<<text;f.flush();if (!f) throw std::runtime_error("diagnostic publication flush");
    f.close();if (!f) throw std::runtime_error("diagnostic publication close");
    std::filesystem::rename(temp,final);
}
} // namespace strix::cuda::qwen_mtp_diagnostic
