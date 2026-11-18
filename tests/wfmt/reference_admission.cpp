#include "fixture.hpp"
#include <array>

using namespace strix::weight;
using namespace strix::weight::test;

namespace {
bool rejects(const WeightView& view, ArithmeticId arithmetic, Error expected) {
 std::array<Byte,16> bytes; bytes.fill(Byte{0x5a});
 DecodeBuffer output{bytes,99};
 const auto status=decode_reference(builtin_registry(),view,{0,4,arithmetic,Scalar::f32},output);
 return require(status.code==expected && output.written==0 &&
                std::all_of(bytes.begin(),bytes.end(),[](Byte b) { return b==Byte{0x5a}; }),
                "rejection must clear written and preserve output bytes");
}
struct ObservedSource final : Source {
 std::shared_ptr<const Source> inner;
 mutable std::vector<ByteRange> reads;
 explicit ObservedSource(std::shared_ptr<const Source> source):inner(std::move(source)) {}
 Status read(ByteRange range,std::span<Byte> out) const override {
  reads.push_back(range); return inner->read(range,out);
 }
 std::uint64_t size() const noexcept override { return inner->size(); }
 Digest256 identity() const override { return inner->identity(); }
};
}

bool wfmt_strides_test() {
 const std::array<float,4> values{1,2,3,4};
 WeightView view{}; view.storage={"plain.f32",1}; view.shape=matrix(2,2);
 view.payload=memory(bytes_of(std::span<const float>(values)));
 view.has_byte_strides=true;
 for(const auto strides:{std::array<std::uint64_t,4>{4,8,0,0},
                         std::array<std::uint64_t,4>{16,4,0,0},
                         std::array<std::uint64_t,4>{UINT64_MAX,4,0,0},
                         std::array<std::uint64_t,4>{0,4,0,0}}) {
  view.byte_strides=strides;
  if(!require(validate_weight(view).code==Error::unsupported_shape,"unsupported strides validated") ||
     !rejects(view,{},Error::unsupported_shape)) return false;
 }
 view.byte_strides={8,4,0,0};
 std::array<float,4> decoded{};
 DecodeBuffer output{std::as_writable_bytes(std::span(decoded)),0};
 if(!require(bool(decode_reference(builtin_registry(),view,{0,4,{},Scalar::f32},output)) &&
             output.written==sizeof(decoded) && decoded==values,"contiguous F32 strides failed")) return false;
 view.has_byte_strides=false; view.byte_strides={UINT64_MAX,UINT64_MAX,0,0};
 if(!require(bool(validate_weight(view)),"absent strides must be ignored")) return false;
 view.has_byte_strides=true; view.shape=matrix(UINT64_MAX/4+1,1);
 view.byte_strides={4,4,0,0};
 if(!require(validate_weight(view).code==Error::malformed,"stride extent overflow accepted")) return false;
 for(auto id:{FormatId{"plain.f16",1},FormatId{"plain.bf16",1}}) {
  const auto* format=find_format(builtin_registry(),id); if(!format) return false;
  auto half=fixture(*format,0); half.shape=matrix(2,2);
  const std::array<std::uint16_t,4> encodings{0,0,0,0};
  half.payload=memory(bytes_of(std::span<const std::uint16_t>(encodings)));
  half.has_byte_strides=true; half.byte_strides={4,2,0,0};
  if(!require(bool(decode_reference(builtin_registry(),half,{0,4,{},Scalar::f32},output)),
              "contiguous half strides failed")) return false;
 }
 const auto* q8=find_format(builtin_registry(),{"gguf.q8_0",1}); if(!q8) return false;
 auto packed=fixture(*q8,0); packed.has_byte_strides=true; packed.byte_strides={34,1,0,0};
 SignedGroup32 group{};
 return rejects(packed,{},Error::unsupported_shape) &&
        require(signed_group_reference(builtin_registry(),packed,0,group).code==Error::unsupported_shape,
                "packed group decoder ignored declared strides");
}

bool wfmt_arithmetic_test() {
 const auto* format=find_format(builtin_registry(),{"gguf.q8_0",1}); if(!format) return false;
 auto view=fixture(*format,0);
 std::vector<Byte> raw(34); raw[0]=Byte{1}; raw[1]=Byte{0x3c}; raw[2]=Byte{3};
 view.payload=memory(std::move(raw));
 float value=0; DecodeBuffer output{std::as_writable_bytes(std::span(&value,1)),0};
 if(!require(bool(decode_reference(builtin_registry(),view,{0,1,{},Scalar::f32},output)) &&
             output.written==sizeof(value) && value==3.0029296875F,
             "storage widening must preserve the unrounded Q8 operand")) return false;
 // This identity requires 3.00390625F. Until operand rounding is implemented,
 // rejecting it is mandatory; ordinary dequantization is not an exact oracle.
 if(!rejects(view,{"intel-q8a8-f16-ascending",1},Error::unsupported_arithmetic)) return false;
 for(const auto& arithmetic:legacy_arithmetic()) {
  if(!rejects(view,arithmetic.id,Error::unsupported_arithmetic)) return false;
 }
 if(!rejects(view,{"intel-q8a8-f16-ascending",2},Error::unsupported_arithmetic) ||
    !rejects(view,{"unknown",1},Error::unsupported_arithmetic) ||
    !rejects(view,{"",1},Error::unsupported_arithmetic)) return false;
 const auto* plain=find_format(builtin_registry(),{"plain.f32",1}); if(!plain) return false;
 auto scalar=fixture(*plain,0); scalar.shape=matrix(1,4);
 scalar.payload=memory(std::vector<Byte>(16));
 return rejects(scalar,{"intel-q8a8-f16-ascending",1},Error::unsupported_arithmetic);
}

bool wfmt_group_reads_test() {
 for(auto id:{FormatId{"gguf.q8_0",1},FormatId{"gguf.iq3_s",1}}) {
  const auto* format=find_format(builtin_registry(),id); if(!format) return false;
  auto view=fixture(*format,1);
  std::vector<Byte> block(static_cast<std::size_t>(view.payload.range.bytes));
  if(!view.payload.source->read(view.payload.range,block)) return false;
  const auto groups=view.geometry.block_elements/32;
  SignedGroup32 expected{};
  if(!signed_group_reference(builtin_registry(),view,groups-1,expected)) return false;
  // A nonzero source offset and a later block exercise both address terms.
  std::vector<Byte> raw(8+3*block.size());
  std::copy(block.begin(),block.end(),raw.begin()+static_cast<std::ptrdiff_t>(8+2*block.size()));
  auto payload=memory(std::move(raw));
  auto observed=std::make_shared<ObservedSource>(payload.source);
  view.payload={observed,{8,3*block.size()}}; view.shape.dims[1]*=3;
  SignedGroup32 actual{};
  if(!require(bool(signed_group_reference(builtin_registry(),view,3*groups-1,actual)),
              "later-block group read failed")) return false;
  if(!require(actual.codes==expected.codes && actual.scales==expected.scales &&
              actual.offsets==expected.offsets && actual.scale_index==expected.scale_index &&
              actual.scale_count==expected.scale_count,"bounded read changed group operands")) return false;
  if(!require(observed->reads.size()==1 && observed->reads[0].offset==8+2*block.size() &&
              observed->reads[0].bytes==block.size(),"group read must fetch exactly one owning block")) return false;
 }
 return true;
}

bool wfmt_reference_admission_test() { return wfmt_strides_test() && wfmt_arithmetic_test() && wfmt_group_reads_test(); }
