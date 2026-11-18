#include "fixture.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/weight/formats/ggml.hpp"
#include "strix/gsq_a8_groups.hpp"
#include <set>
using namespace strix::weight;
using namespace strix::weight::test;
bool wfmt_geometry_test() {
 const auto& registry=builtin_registry(); std::set<std::string> ids;
 for(const auto& format:registry.formats) {
  if(!require(ids.insert(serialize_id(format.id)).second,"duplicate format ID")) return false;
  Parameters p{}; p.bits=4; p.group_size=16; Geometry geometry{};
  if(!require(bool(format.geometry(p,geometry)),"geometry registration failed")) return false;
  if(!require(geometry.block_elements && geometry.block_bytes,"zero geometry")) return false;
 }
 for(std::uint32_t wire=0;wire<64;++wire) {
  const auto old=strix::gguf::tensor_type_geometry(wire); const auto id=gguf_format(wire);
  if(!require(old.has_value()==!id.name.empty(),"GGUF parser geometry coverage differs")) return false;
  if(!old) continue;
  Geometry g{}; if(!format_geometry(id,{},g)) return false;
  if(!require(g.block_elements==old->block_elements && g.block_bytes==old->block_bytes,"GGUF geometry differs")) return false;
  const auto* format=find_format(registry,id); if(!format) return false;
  const auto view=fixture(*format,42); if(!require(bool(validate_weight(view)),"geometry fixture admission failed")) return false;
  if(!format->semantic_decode_supported) {
   std::vector<Byte> bytes(1024); DecodeBuffer output{bytes,99};
   if(!require(!decode_reference(registry,view,{0,1,{},Scalar::f32},output) && output.written==0,"parse-only format decoded")) return false;
  }
 }
 for(auto wire:{18u,20u,21u,22u,23u,42u}) {
  Geometry g{}; if(!format_geometry(gguf_format(wire),{},g)) return false;
  if(!require(g.block_elements==strix::gsq_a8::block_elements(wire) && g.block_bytes==strix::gsq_a8::block_bytes(wire),"forwarded GSQ geometry differs")) return false;
 }
 if(!require(formats::Iq3S::d==0 && formats::Iq3S::qs==2 && formats::Iq3S::qh==66 && formats::Iq3S::signs==74 && formats::Iq3S::scales==106 && formats::Iq3S::bytes==110,"IQ3_S field offsets")) return false;
 return true;
}
