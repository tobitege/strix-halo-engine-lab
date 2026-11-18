#pragma once

#include "strix/model_runtime_binding.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix {

enum class RotaryMode { native, yarn };

// Process-wide mathematical configuration, independent of checkpoint metadata.
// No native table is implied. Effective maximum is the table extent; requested
// context remains a separate admission input.
class RotaryDescriptor {
public:
    static RotaryDescriptor native() { return RotaryDescriptor(0); }
    static RotaryDescriptor yarn(std::uint32_t factor);
    RotaryMode mode() const { return factor_ ? RotaryMode::yarn : RotaryMode::native; }
    std::uint32_t factor() const { return factor_; }
    static constexpr std::uint32_t original_max = 262144;
    std::uint32_t effective_max() const { return original_max * (factor_ ? factor_ : 1U); }
    static constexpr std::uint32_t rotary_dim = 64, index_rotary_dim = 64;
    static constexpr double theta = 10000000.0, beta_fast = 32.0, beta_slow = 1.0;
    static constexpr std::array<std::uint32_t, 3> sections{11, 11, 10};
    static constexpr bool interleaved = true, truncate = true;
    static constexpr std::string_view math_version = "qwen-yarn-hf64-v1";
    double attention_factor() const;
    std::uint64_t table_bytes() const;
    // Empty means no payload: native identities must append no tag or length.
    std::vector<std::byte> canonical_bytes() const;
    ModelContentIdentity sha256() const;
    std::string json() const;
    friend bool operator==(const RotaryDescriptor&, const RotaryDescriptor&) = default;
private:
    explicit RotaryDescriptor(std::uint32_t factor) : factor_(factor) {}
    std::uint32_t factor_;
};

// Resolved numerical identity. A non-native identity requires the full uploaded
// coefficient payload digest; pointer addresses never define numerical identity.
class RotaryIdentity {
public:
    RotaryIdentity() = default;
    RotaryIdentity(const RotaryDescriptor& native) : descriptor_(native) {
        if (native.mode() != RotaryMode::native)
            throw std::invalid_argument("non-native identity requires table SHA256");
    }
    RotaryIdentity(const RotaryDescriptor& descriptor, ModelContentIdentity table)
        : descriptor_(descriptor), table_(table) {
        if (descriptor.mode() != RotaryMode::yarn || !table.valid())
            throw std::invalid_argument("resolved rotary identity requires YaRN and table SHA256");
    }
    const RotaryDescriptor& descriptor() const noexcept { return descriptor_; }
    std::string json() const;
    const ModelContentIdentity& table_sha256() const noexcept { return table_; }
    std::vector<std::byte> canonical_bytes() const;
    friend bool operator==(const RotaryIdentity&, const RotaryIdentity&) = default;
private:
    RotaryDescriptor descriptor_ = RotaryDescriptor::native();
    ModelContentIdentity table_;
};

// Admission for target-produced activations, before touching their device views.
inline void require_rotary_producer_identity(const RotaryIdentity& producer, const RotaryIdentity& reader) {
    if (producer != reader) throw std::invalid_argument("MTP target/draft rotary identity mismatch");
}

// Future identity owners can append this domain-separated payload. Native is
// exactly a no-op, including when the existing hash has value zero.
std::uint64_t append_rotary_identity(std::uint64_t existing, const RotaryIdentity& rotary);
std::uint32_t parse_qwen_rope_yarn(std::string_view value);
// Diagnostic recommendation only: smallest experimental factor covering a non-native
// context (2 up to 524,288, 4 up to 1,048,576), or 0. Never selects numerical policy.
std::uint32_t recommended_qwen_rope_yarn(std::uint64_t context) noexcept;
RotaryDescriptor validate_qwen_rotary_request(std::uint32_t factor,
    bool context_explicit, std::uint64_t context, bool baseline);

} // namespace strix
