#include "catalog.hpp"
#include "strix/ggml_dequant.hpp"
namespace strix::weight::detail {
std::span<const FormatDescriptor> ggml_formats() {
 static const FormatDescriptor formats[] = {
  descriptor({"gguf.q4_0",1},"GGUF:2",quantized_linear,true,fixed<32,18>),
  descriptor({"gguf.q4_1",1},"GGUF:3",quantized_linear,false,fixed<32,20>),
  descriptor({"gguf.q5_0",1},"GGUF:6",quantized_linear,true,fixed<32,22>),
  descriptor({"gguf.q5_1",1},"GGUF:7",quantized_linear,true,fixed<32,24>),
  descriptor({"gguf.q8_0",1},"GGUF:8",quantized_linear,true,fixed<32,34>,true),
  descriptor({"gguf.q8_1",1},"GGUF:9",quantized_linear,false,fixed<32,36>),
  descriptor({"gguf.q3_k",1},"GGUF:11",quantized_linear,true,fixed<256,110>),
  descriptor({"gguf.q4_k",1},"GGUF:12",quantized_linear,true,fixed<256,144>),
  descriptor({"gguf.q5_k",1},"GGUF:13",quantized_linear,true,fixed<256,176>),
  descriptor({"gguf.q6_k",1},"GGUF:14",quantized_linear,true,fixed<256,210>),
  descriptor({"gguf.q8_k",1},"GGUF:15",quantized_linear,false,fixed<256,292>),
  descriptor({"gguf.iq2_xs",1},"GGUF:17",quantized_linear,false,fixed<256,74>),
  descriptor({"gguf.iq3_xxs",1},"GGUF:18",quantized_linear,true,fixed<256,98>,true),
  descriptor({"gguf.iq1_s",1},"GGUF:19",quantized_linear,false,fixed<256,50>),
  descriptor({"gguf.iq4_nl",1},"GGUF:20",quantized_linear,true,fixed<32,18>,true),
  descriptor({"gguf.iq3_s",1},"GGUF:21",quantized_linear,true,fixed<256,110>,true),
  descriptor({"gguf.iq2_s",1},"GGUF:22",quantized_linear,true,fixed<256,82>,true),
  descriptor({"gguf.iq4_xs",1},"GGUF:23",quantized_linear,true,fixed<256,136>,true),
  descriptor({"gguf.iq1_m",1},"GGUF:29",quantized_linear,false,fixed<256,56>),
  descriptor({"gguf.tq1_0",1},"GGUF:34",quantized_linear,false,fixed<256,54>),
  descriptor({"gguf.tq2_0",1},"GGUF:35",quantized_linear,false,fixed<256,66>),
  descriptor({"gguf.q2_0.strix",1},"GGUF:42",quantized_linear,true,fixed<64,18>,true),
  // Converted NVFP4 tile pack (gguf_reader.hpp kNvfp4TilePack): container geometry only; global scales live in a table tensor.
  descriptor({"gguf.nvfp4_tile.strix",1},"GGUF:43",quantized_linear,false,fixed<128,72>),
  // Private IU4 g64 dense pack (core/iu4.hpp): container geometry only, no ggml semantic decoder.
  descriptor({"gguf.iu4_g64.strix",1},"GGUF:200",quantized_linear,false,fixed<64,34>)
 };
 return formats;
}
Status decode_ggml(const WeightView& view,const std::vector<Byte>& raw,std::vector<float>& out) {
 ggml_dequant::dequantize_row(gguf_type(view.storage),raw,out);
 return {Error::ok,{}};
}
}
