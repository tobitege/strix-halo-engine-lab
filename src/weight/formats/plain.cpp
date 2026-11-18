#include "catalog.hpp"
#include "strix/ggml_dequant.hpp"
namespace strix::weight::detail {
std::span<const FormatDescriptor> plain_formats() {
 static const FormatDescriptor formats[] = {
  descriptor({"plain.f32",1},"F32",plain_linear,true,fixed<1,4,1>,false,all_linear_roles | (1ull<<unsigned(Role::lookup_row)) | (1ull<<unsigned(Role::auxiliary))),
  descriptor({"plain.f16",1},"F16",plain_linear,true,fixed<1,2,1>,false,all_linear_roles | (1ull<<unsigned(Role::lookup_row)) | (1ull<<unsigned(Role::auxiliary))),
  descriptor({"plain.bf16",1},"BF16",plain_linear,true,fixed<1,2,1>,false,all_linear_roles | (1ull<<unsigned(Role::lookup_row)) | (1ull<<unsigned(Role::auxiliary))),
  descriptor({"typed.i8",1},"I8",typed_auxiliary,false,fixed<1,1,1>,false,auxiliary_roles),
  descriptor({"typed.u8",1},"U8",typed_auxiliary,false,fixed<1,1,1>,false,auxiliary_roles),
  descriptor({"typed.i16",1},"I16",typed_auxiliary,false,fixed<1,2,1>,false,auxiliary_roles),
  descriptor({"typed.u16",1},"U16",typed_auxiliary,false,fixed<1,2,1>,false,auxiliary_roles),
  descriptor({"typed.i32",1},"I32",typed_auxiliary,false,fixed<1,4,1>,false,auxiliary_roles),
  descriptor({"typed.u32",1},"U32",typed_auxiliary,false,fixed<1,4,1>,false,auxiliary_roles),
  descriptor({"typed.i64",1},"I64",typed_auxiliary,false,fixed<1,8,1>,false,auxiliary_roles),
  descriptor({"typed.u64",1},"U64",typed_auxiliary,false,fixed<1,8,1>,false,auxiliary_roles),
  descriptor({"typed.bool8",1},"BOOL",typed_auxiliary,false,fixed<1,1,1>,false,auxiliary_roles),
  descriptor({"typed.e4m3fn",1},"F8_E4M3",typed_auxiliary,false,fixed<1,1,1>,false,auxiliary_roles),
  descriptor({"typed.e5m2",1},"F8_E5M2",typed_auxiliary,false,fixed<1,1,1>,false,auxiliary_roles),
  descriptor({"typed.e8m0",1},"F8_E8M0",typed_auxiliary,false,fixed<1,1,1>,false,auxiliary_roles),
  descriptor({"typed.e2m1",1},"F4_E2M1",typed_auxiliary,false,fixed<2,1,1>,false,auxiliary_roles),
  descriptor({"typed.f64",1},"F64",typed_auxiliary,false,fixed<1,8,1>,false,auxiliary_roles)
 };
 return formats;
}
Status decode_plain(const WeightView& view,const std::vector<Byte>& raw,std::vector<float>& out) {
 const auto type=gguf_type(view.storage);
 ggml_dequant::dequantize_row(type,raw,out);
 return {Error::ok,{}};
}
}
