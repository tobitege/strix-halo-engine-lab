#include "strix/hip_qwen_moe_prepared.hpp"
#include "strix/hip_model_residency.hpp"
#include "strix/hip_check.hpp"
#include <hip/hip_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace strix::qwen_moe_prepared {
namespace {
void check(hipError_t result) {
    ::strix::hip_check(result, "prepared MoE HIP allocation/upload");
}
class HipAllocation final : public Allocation {
    void* data_ = nullptr;
    std::uint64_t bytes_ = 0;
public:
    explicit HipAllocation(std::uint64_t bytes) : bytes_(bytes) {
        if (!bytes || bytes > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("invalid prepared MoE HIP allocation size");
        check(hipMalloc(&data_, static_cast<std::size_t>(bytes)));
    }
    ~HipAllocation() override { if (data_) (void)hipFree(data_); }
    std::uintptr_t address() const noexcept override { return reinterpret_cast<std::uintptr_t>(data_); }
    std::uint64_t bytes() const noexcept override { return bytes_; }
    void upload(std::uint64_t offset, std::span<const std::byte> source) override {
        if (offset > bytes_ || source.size() > bytes_ - offset)
            throw std::invalid_argument("prepared MoE HIP upload outside allocation");
        check(hipMemcpy(static_cast<std::byte*>(data_) + offset, source.data(), source.size(), hipMemcpyHostToDevice));
    }
};
}
std::unique_ptr<Allocation> allocate_hip(std::uint64_t bytes) { return std::make_unique<HipAllocation>(bytes); }
} // namespace strix::qwen_moe_prepared

namespace strix::qwen_moe_prepared {
const PreparedTensor* Model::find(std::uintptr_t original, Role role, Spec expected) const {
    for (const auto& e : entries) {
        const auto& t=e.tensor;
        if (e.original==original && t.spec.role==role && t.spec.format==expected.format &&
            t.spec.input==expected.input && t.spec.outputs==expected.outputs && t.spec.experts==expected.experts &&
            t.layout.version==layout_version && t.identity.valid() && t.allocation &&
            t.allocation->bytes()==t.layout.tensor_bytes) return &t;
    }
    return nullptr;
}
}
namespace strix {
void HipModelResidency::prepare_qwen_moe_direct_b(const MappedModelFile& mapped,const ModelManifest& manifest) {
    namespace qp=qwen_moe_prepared;
    if (binding_.qwen_moe_prepared) throw std::logic_error("prepared MoE already installed");
    if (binding_.model_content_identity!=mapped.content_identity()) throw std::invalid_argument("prepared MoE model hash differs");
    mapped.validate_unchanged();
    auto prepared=std::make_shared<qp::Model>();
    prepared->model_hash=mapped.content_identity();
    for (std::uint32_t i=0;i<manifest.tensors.size();++i) {
        const auto& t=manifest.tensors[i];
        const auto admitted=qp::admitted_spec(manifest,i);
        if(!admitted)continue;
        const auto spec=*admitted;
        const auto* runtime=find_runtime_tensor_binding(binding_,i);
        if (!runtime||!runtime->address||runtime->bytes!=t.byte_size||qp::describe(spec).tensor_bytes!=t.byte_size) continue;
        auto tensor=qp::prepare(spec,{prepared->model_hash,i,t.absolute_offset},mapped.bytes(),
            [&](std::uint64_t at,std::span<std::byte> bytes){mapped.read_at(at,bytes);},qp::allocate_hip);
        prepared->entries.push_back({runtime->address,std::move(tensor)});
    }
    mapped.validate_unchanged();
    binding_.qwen_moe_prepared=std::move(prepared);
}
}
