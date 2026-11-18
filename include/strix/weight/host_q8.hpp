#pragma once
#include "source.hpp"
#include "registry.hpp"
#include "cache.hpp"
namespace strix::weight {
// One vendor-free storage contract; operand rounding and reduction remain typed.
struct Q8Storage {
 static constexpr std::uint32_t elements=32, bytes=34, scale_bytes=2;
 static constexpr std::uint64_t row_bytes(std::uint64_t width) noexcept { return width/elements*bytes; }
 static const std::uint8_t* block(const std::uint8_t* row,std::size_t group) noexcept { return row+group*bytes; }
 static std::uint16_t scale_bits(const std::uint8_t* packed) noexcept {
  return static_cast<std::uint16_t>(packed[0] | (std::uint16_t(packed[1])<<8));
 }
 static std::int8_t code(const std::uint8_t* packed,std::size_t lane) noexcept {
  return static_cast<std::int8_t>(packed[scale_bytes+lane]);
 }
};
// The explicit owner outlives every view. Row padding/prefixes are source
// addressing, not a new format or a second interpretation of Q8_0 bytes.
Status q8_weight_from_legacy(std::span<const Byte>,std::shared_ptr<const void> owner,
                            std::uint32_t outputs,std::uint32_t width,std::uint32_t row_bytes,
                            std::uint32_t k_bytes,TensorId,WeightView&);
class HostQ8Binding {
 WeightView source_{};
 std::shared_ptr<const std::vector<Byte>> bytes_;
 std::shared_ptr<ImageCache> cache_;
 std::shared_ptr<ImageLease> image_;
 Plan plan_{};
 ImageKey key_{};
 friend Status prepare_host_q8(const WeightView&,HostQ8Binding&);
 friend Status prepare_q8_image(const WeightView&,const Plan&,std::shared_ptr<ImageCache>,ExecutionContext&,HostQ8Binding&);
public:
 const std::uint8_t* data() const noexcept {
  return image_?static_cast<const std::uint8_t*>(image_->data()):
   bytes_?reinterpret_cast<const std::uint8_t*>(bytes_->data()):nullptr;
 }
 std::uint32_t row_bytes() const noexcept { return static_cast<std::uint32_t>(Q8Storage::row_bytes(source_.shape.dims[1])); }
 // Immutable prepared storage; copies retain both the source and executed image.
 Status validate(const WeightView&,const std::uint8_t*,std::uint32_t stride,std::uint32_t prefix=0) const;
 bool is_image() const noexcept { return bool(image_); }
 Status acquire(ExecutionContext&,ImageLease&) const;
};
// Launch envelopes are POD and consumed synchronously. Borrow the prepared owner
// only until dispatch, which retains it (and fences native last use) before return.
class Q8BindingRef {
 const HostQ8Binding* binding_=nullptr;
public:
 Q8BindingRef()=default;
 Q8BindingRef(const HostQ8Binding& binding):binding_(&binding) {}
 Q8BindingRef(const HostQ8Binding&&)=delete;
 HostQ8Binding retain() const { return binding_?*binding_:HostQ8Binding{}; }
};
// Explicit preparation, never allocation or conversion inside a numerical loop.
Status prepare_host_q8(const WeightView&,HostQ8Binding&);
// Uses the runtime's compact Q8 copy contract and its readiness/retirement events.
Status prepare_q8_image(const WeightView&,const Plan&,std::shared_ptr<ImageCache>,ExecutionContext&,HostQ8Binding&);
Status validate_q8_shape(const WeightView&,std::uint64_t outputs,std::uint64_t width);
}
