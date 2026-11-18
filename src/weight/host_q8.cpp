#include "strix/weight/host_q8.hpp"
#include "h3/te/te_sha256.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <stdexcept>
namespace strix::weight {
namespace {
class RowSource final : public Source {
 std::shared_ptr<const Source> source_;
 std::uint64_t rows_,logical_,stride_,prefix_;
 Digest256 digest_{};
public:
 RowSource(std::shared_ptr<const Source> source,std::uint64_t rows,std::uint64_t logical,
           std::uint64_t stride,std::uint64_t prefix):source_(std::move(source)),rows_(rows),logical_(logical),stride_(stride),prefix_(prefix) {
  h3::te::Sha256 hash;
  std::array<Byte,65536> bytes{};
  for(std::uint64_t offset=0;offset<size();) {
   const auto count=static_cast<std::size_t>(std::min<std::uint64_t>(bytes.size(),size()-offset));
   const auto status=read({offset,count},std::span(bytes).first(count));
   if(!status) throw std::runtime_error(status.detail);
   hash.update(bytes.data(),count); offset+=count;
  }
  digest_.bytes=hash.finish();
 }
 Status read(ByteRange range,std::span<Byte> out) const override {
  if(range.offset>size() || range.bytes>size()-range.offset || out.size()!=range.bytes)
   return {Error::malformed,"Q8 source range"};
  std::uint64_t copied=0;
  while(copied<range.bytes) {
   const auto position=range.offset+copied,row=position/logical_,column=position%logical_;
   const auto count=std::min(logical_-column,range.bytes-copied);
   auto status=source_->read({row*stride_+prefix_+column,count},out.subspan(static_cast<std::size_t>(copied),static_cast<std::size_t>(count)));
   if(!status) return status;
   copied+=count;
  }
  return {Error::ok,{}};
 }
 std::uint64_t size() const noexcept override { return rows_*logical_; }
 Digest256 identity() const override { return digest_; }
 std::string shard_name() const override { return source_->shard_name(); }
};
}
Status q8_weight_from_legacy(std::span<const Byte> raw,std::shared_ptr<const void> owner,
                            std::uint32_t outputs,std::uint32_t width,std::uint32_t stride,
                            std::uint32_t prefix,TensorId tensor,WeightView& out) {
 if(!owner || !raw.data()) return {Error::stale_source,"legacy Q8 view requires an explicit live owner"};
 if(!outputs || !width || width%Q8Storage::elements) return {Error::unsupported_shape,"Q8 matrix requires positive N and K multiple of 32"};
 const auto logical=Q8Storage::row_bytes(width);
 if(logical>UINT32_MAX || std::uint64_t(prefix)+logical>stride ||
    std::uint64_t(outputs-1)*stride+prefix+logical>raw.size()) return {Error::malformed,"legacy Q8 row stride/prefix/buffer size"};
 try {
  auto base=view_source(raw,std::move(owner),"host-q8");
  auto source=std::make_shared<RowSource>(std::move(base),outputs,logical,stride,prefix);
  WeightView result{};
  result.tensor=tensor; result.storage={"gguf.q8_0",1}; result.shape={{outputs,width,0,0},2};
  result.geometry={Q8Storage::elements,Q8Storage::bytes,Q8Storage::elements,1,1,Q8Storage::elements,1,true};
  result.payload={std::move(source),{0,std::uint64_t(outputs)*logical}};
  auto status=validate_weight(result); if(!status) return status;
  out=std::move(result); return {Error::ok,{}};
 } catch(const std::bad_alloc&) { return {Error::memory_budget,"Q8 source preparation allocation"}; }
 catch(const std::exception& error) { return {Error::malformed,error.what()}; }
}
Status validate_q8_shape(const WeightView& weight,std::uint64_t outputs,std::uint64_t width) {
 if(!(weight.storage==FormatId{"gguf.q8_0",1})) return {Error::unsupported_arithmetic,"only existing Q8_0 operand contract is registered; IQ/trellis/NVFP4 require separate qualified arithmetic"};
 if(weight.shape.rank!=2 || weight.shape.dims[0]!=outputs || weight.shape.dims[1]!=width ||
    !outputs || !width || outputs>UINT32_MAX || width>UINT32_MAX || width%Q8Storage::elements ||
    Q8Storage::row_bytes(width)>UINT32_MAX) return {Error::unsupported_shape,"Q8 logical shape differs from launch"};
 if(!weight.sidecars.empty()) return {Error::malformed,"raw Q8_0 has no sidecar operands"};
 return validate_weight(weight);
}
Status prepare_host_q8(const WeightView& weight,HostQ8Binding& out) {
 if(weight.shape.rank!=2) return {Error::unsupported_shape,"Q8 host preparation requires rank two"};
 auto status=validate_q8_shape(weight,weight.shape.dims[0],weight.shape.dims[1]); if(!status) return status;
 if(weight.payload.range.bytes>SIZE_MAX) return {Error::memory_budget,"Q8 host image size overflow"};
 try {
  HostQ8Binding result;
  result.source_=weight;
  auto bytes=std::make_shared<std::vector<Byte>>(static_cast<std::size_t>(weight.payload.range.bytes));
  status=read_slice(weight.payload,*bytes); if(!status) return status;
  result.bytes_=std::move(bytes);
  out=std::move(result); return {Error::ok,{}};
 } catch(const std::bad_alloc&) { return {Error::memory_budget,"Q8 host image allocation"}; }
}
Status HostQ8Binding::validate(const WeightView& weight,const std::uint8_t* pointer,std::uint32_t stride,std::uint32_t prefix) const {
 if(!data()) return {Error::malformed,"Q8 launch requires an owning prepared binding"};
 auto status=validate_q8_shape(weight,source_.shape.dims[0],source_.shape.dims[1]); if(!status) return status;
 if(weight.tensor!=source_.tensor || weight.payload.source!=source_.payload.source ||
    weight.payload.range.offset!=source_.payload.range.offset || weight.payload.range.bytes!=source_.payload.range.bytes ||
    weight.byte_strides!=source_.byte_strides || weight.has_byte_strides!=source_.has_byte_strides)
  return {Error::stale_source,"Q8 binding belongs to a different source"};
 if(pointer!=data() || stride!=row_bytes() || prefix)
  return {Error::malformed,"Q8 raw pointer/stride/prefix differs from prepared binding"};
 return {Error::ok,{}};
}
Status HostQ8Binding::acquire(ExecutionContext& context,ImageLease& lease) const {
 if(!image_ || !cache_) return {Error::unsupported_backend,"native Q8 launch requires a prepared runtime image"};
 return acquire_image(*cache_,key_,plan_,source_,context,lease);
}
Status prepare_q8_image(const WeightView& weight,const Plan& plan,std::shared_ptr<ImageCache> cache,
                        ExecutionContext& context,HostQ8Binding& out) {
 auto status=validate_q8_shape(weight,weight.shape.dims[0],weight.shape.dims[1]); if(!status) return status;
 if(!cache || plan.selected.resources.image_bytes!=weight.payload.range.bytes)
  return {Error::malformed,"Q8 runtime image must contain the complete compact payload"};
 HostQ8Binding result; result.source_=weight; result.plan_=plan; result.cache_=std::move(cache);
 status=make_image_key(weight,plan,"q8-compact","q8-binding-v2",result.key_); if(!status) return status;
 result.image_=std::make_shared<ImageLease>();
 status=acquire_image(*result.cache_,result.key_,plan,weight,context,*result.image_); if(!status) return status;
 status=record_last_use(*result.image_,context); if(!status) return status;
 out=std::move(result); return {Error::ok,{}};
}
}
