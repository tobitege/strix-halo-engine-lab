#pragma once
#include "strix/env_flags.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/serial_session.hpp"
#include <cstdlib>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix {
// Set once at bridge startup, before worker threads; CLI and environment share the resolved policy.
void set_prefix_batch_certification_enabled(bool enabled) noexcept;
[[nodiscard]] bool prefix_batch_certification_enabled() noexcept;
// Value owned by the ticket, independent of the producer's scratch/lease lifetime.
struct OutputLogitsSnapshot {
    std::optional<ModelContentIdentity> digest;
    void capture(const std::optional<ModelContentIdentity>& row) { digest=row; }
};
inline bool prompt_terminal_digest_enabled() {
    static const bool enabled=strix::envcfg::flag("STRIX_PREFIX_CACHE_DIGEST", false);
    return enabled;
}
inline std::optional<ModelContentIdentity> raw_fp32_logits_digest(std::span<const float> row) {
    if(row.empty()) return std::nullopt;
    return sha256_bytes(std::as_bytes(row));
}
// H0 = SHA256(empty), Hn = SHA256(Hn-1 || SHA256(raw FP32 vocabulary row n)).
// A missing row is an error, never an apparently complete diagnostic.
struct OutputLogitsChain {
    ModelContentIdentity digest{};
    std::uint64_t rows=0;
    const char* error=nullptr;
    void fail(const char* reason) noexcept { if(!error) error=reason; }
    bool complete(std::uint64_t expected) {
        if(rows!=expected) fail("request final incomplete output logits coverage");
        return !error;
    }
    void append(const std::optional<ModelContentIdentity>& row) {
        if(error) return;
        if(!row) {fail("request final missing output logits row");return;}
        std::vector<std::byte> frame;
        const auto previous=rows?digest:sha256_bytes(std::span<const std::byte>{});
        for(const auto byte:previous.bytes) frame.push_back(static_cast<std::byte>(byte));
        for(const auto byte:row->bytes) frame.push_back(static_cast<std::byte>(byte));
        digest=sha256_bytes(frame);++rows;
    }
};
// Canonical LE values, never structure padding or allocation addresses. The frontier
// and committed count are emitted separately beside this generations/lineage hash.
inline ModelContentIdentity terminal_generations_lineage_digest(const SerialSessionContext& session) {
    std::vector<std::byte> frame;
    const auto u64=[&](std::uint64_t value) {
        for(unsigned shift=0;shift<64;shift+=8) frame.push_back(static_cast<std::byte>((value>>shift)&255U));
    };
    u64(static_cast<std::uint64_t>(session.state.valid));
    for(const auto generation:session.state.generations) u64(generation);
    const auto& lineage=session.arithmetic_lineage;
    u64(lineage.history_identity);u64(lineage.tainted?1U:0U);u64(lineage.intervals.size());
    for(const auto& interval:lineage.intervals) {
        u64(interval.begin);u64(interval.end);u64(static_cast<std::uint64_t>(interval.producer));
        u64(interval.numerical_plan_identity);u64(interval.reduction_plan_identity);u64(interval.state_update_identity);
    }
    return sha256_bytes(frame);
}
} // namespace strix
