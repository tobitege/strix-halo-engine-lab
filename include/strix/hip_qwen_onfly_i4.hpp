#pragma once
// TURBO+ onfly INT4 g64 copies of the Qwen dense Q8_0 projection tensors (strix/onfly_i4.hpp): built once at load from
// the resident Q8_0 tensor on the device, registered in the q6p::iu4 registry (hip_q8_rows_iu4.hpp reads it), retired
// before the backing is freed. v1 keeps the Q8_0 source resident (prefill and the non-eligible sites read it).
#include "strix/mapped_model_file.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/onfly_i4.hpp"
#include "strix/uma_owner_inventory.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace strix {
class HipQwenOnflyI4 {
public:
    struct Copy {
        const void* original = nullptr;
        void* device = nullptr;
        std::uint64_t bytes = 0, lease = 0;
        std::shared_ptr<const void> backing;
    };
    std::vector<Copy> copies;
    std::uint64_t copy_bytes = 0, source_bytes = 0, skipped = 0;
    std::array<std::uint64_t, 4> group_tensors{}, group_copy_bytes{};  // indexed by log2(Group)
    double convert_seconds = 0.0;
    const onfly_i4::Resolved resolved;
    explicit HipQwenOnflyI4(onfly_i4::Resolved request = {}) : resolved(request) {}
    HipQwenOnflyI4(const HipQwenOnflyI4&) = delete;
    HipQwenOnflyI4& operator=(const HipQwenOnflyI4&) = delete;
    ~HipQwenOnflyI4();
    [[nodiscard]] UmaOwnerInventory owner_inventory(std::shared_ptr<const HipQwenOnflyI4> owner) const;
    [[nodiscard]] std::string receipt_json() const;
};
[[nodiscard]] std::shared_ptr<const HipQwenOnflyI4> load_hip_qwen_onfly_i4(
    const ModelManifest&, const ModelRuntimeBinding&, const onfly_i4::Resolved&);
} // namespace strix
