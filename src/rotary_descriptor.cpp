#include "strix/rotary_descriptor.hpp"
#include "strix/model_content_hash.hpp"
#include <bit>
#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace strix {
RotaryDescriptor RotaryDescriptor::yarn(std::uint32_t factor) {
    if (factor != 2 && factor != 4) throw std::invalid_argument("YaRN factor must be 2 or 4");
    return RotaryDescriptor(factor);
}
double RotaryDescriptor::attention_factor() const {
    return factor_ ? 1.0 + 0.1 * std::log(static_cast<double>(factor_)) : 1.0;
}
std::uint64_t RotaryDescriptor::table_bytes() const {
    return factor_ ? static_cast<std::uint64_t>(effective_max()) * 32U * 8U : 0;
}
std::vector<std::byte> RotaryDescriptor::canonical_bytes() const {
    if (!factor_) return {};
    std::vector<std::byte> bytes;
    const auto text = [&](std::string_view s) {
        for (const auto c : s) bytes.push_back(static_cast<std::byte>(c));
        bytes.push_back(std::byte{0});
    };
    const auto u64 = [&](std::uint64_t v) {
        for (unsigned shift = 0; shift < 64; shift += 8)
            bytes.push_back(static_cast<std::byte>((v >> shift) & 255U));
    };
    text("STRIX-ROTARY-V1"); text("yarn"); text(math_version);
    text("fp64-libm-to-f32-rne");
    u64(factor_); u64(original_max); u64(effective_max());
    u64(rotary_dim); u64(index_rotary_dim); u64(std::bit_cast<std::uint64_t>(theta));
    for (auto section : sections) u64(section);
    u64(interleaved); u64(std::bit_cast<std::uint64_t>(beta_fast));
    u64(std::bit_cast<std::uint64_t>(beta_slow)); u64(truncate);
    u64(std::bit_cast<std::uint64_t>(attention_factor()));
    return bytes;
}
ModelContentIdentity RotaryDescriptor::sha256() const {
    return sha256_bytes(canonical_bytes());
}
std::vector<std::byte> RotaryIdentity::canonical_bytes() const {
    auto bytes = descriptor_.canonical_bytes();
    if (bytes.empty()) return bytes;
    constexpr std::string_view tag = "STRIX-ROTARY-TABLE-SHA256-V1";
    for (auto c : tag) bytes.push_back(static_cast<std::byte>(c));
    bytes.push_back(std::byte{0});
    for (auto b : table_.bytes) bytes.push_back(static_cast<std::byte>(b));
    return bytes;
}
std::string RotaryIdentity::json() const {
    if (descriptor_.mode() == RotaryMode::native) return "{}";
    auto result = descriptor_.json();
    const auto at = result.find("\"table_sha256\":null");
    result.replace(at, 19, "\"table_sha256\":\"" + model_content_identity_hex(table_) + "\"");
    return result;
}
std::uint64_t append_rotary_identity(std::uint64_t hash, const RotaryIdentity& rotary) {
    for (auto byte : rotary.canonical_bytes()) {
        hash ^= std::to_integer<std::uint8_t>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}
std::string RotaryDescriptor::json() const {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17) << "{\"rope_mode\":\"" << (factor_ ? "yarn" : "native")
        << "\",\"factor\":" << factor_ << ",\"original_max\":" << original_max
        << ",\"effective_max\":" << effective_max() << ",\"rotary_dim\":64,\"index_rotary_dim\":64"
        << ",\"theta\":" << theta << ",\"mrope_sections\":[11,11,10],\"mrope_interleaved\":true"
        << ",\"beta_fast\":32,\"beta_slow\":1,\"truncate\":true,\"attention_factor\":" << attention_factor()
        << ",\"math_version\":\"" << math_version << "\",\"generator\":\"fp64-libm-to-f32-rne\""
        << ",\"descriptor_sha256\":\"" << model_content_identity_hex(sha256())
        << "\",\"table_sha256\":null,\"table_bytes\":" << table_bytes()
        << ",\"qualification\":\"experimental\"}";
    return out.str();
}
std::uint32_t parse_qwen_rope_yarn(std::string_view value) {
    if (value == "off") return 0;
    if (value == "2") return 2;
    if (value == "4") return 4;
    throw std::invalid_argument("--rope-yarn / STRIX_QWEN_ROPE_YARN must be off, 2 or 4");
}
std::uint32_t recommended_qwen_rope_yarn(std::uint64_t context) noexcept {
    for (const std::uint32_t factor : {2U, 4U})
        if (context > RotaryDescriptor::original_max && context <= RotaryDescriptor::original_max * factor) return factor;
    return 0;
}
RotaryDescriptor validate_qwen_rotary_request(std::uint32_t factor,
    bool explicit_context, std::uint64_t context, bool baseline) {
    if (baseline) factor = 0;
    if (!factor) {
        if (context > RotaryDescriptor::original_max) {
            const auto recommendation = recommended_qwen_rope_yarn(context);
            throw std::invalid_argument("native rotary context cannot exceed 262144; --unqualified-context does not extend RoPE; " +
                (recommendation ? "experimental opt-in requires --rope-yarn " + std::to_string(recommendation) +
                    " --context " + std::to_string(context) : "requested extent exceeds supported YaRN factors"));
        }
        return RotaryDescriptor::native();
    }
    const auto descriptor = RotaryDescriptor::yarn(factor);
    if (!explicit_context) throw std::invalid_argument("YaRN requires explicit --context");
    if (!context || context > descriptor.effective_max())
        throw std::invalid_argument("YaRN requires 0 < context <= 262144 * factor");
    return descriptor;
}
} // namespace strix
