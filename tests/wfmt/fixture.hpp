#pragma once
#include "strix/weight/reference.hpp"
#include "strix/weight/source.hpp"
#include <cstring>
#include <algorithm>
#include <iostream>
namespace strix::weight::test {
struct MemorySource final : Source {
 std::vector<Byte> bytes;
 explicit MemorySource(std::vector<Byte> data):bytes(std::move(data)) {}
 Status read(ByteRange range,std::span<Byte> out) const override {
  if(range.offset>bytes.size() || range.bytes>bytes.size()-range.offset || out.size()!=range.bytes)
   return {Error::malformed,"memory source range"};
  std::copy_n(bytes.data()+range.offset,range.bytes,out.data()); return {Error::ok,{}};
 }
 std::uint64_t size() const noexcept override { return bytes.size(); }
 Digest256 identity() const override { return {}; }
};
inline Slice memory(std::vector<Byte> bytes) {
 auto source=std::make_shared<MemorySource>(std::move(bytes));
 return {source,{0,source->size()}};
}
template<class T> std::vector<Byte> bytes_of(std::span<const T> data) {
 const auto bytes=std::as_bytes(data); return {bytes.begin(),bytes.end()};
}
inline NamedSlice named(std::string name,Scalar scalar,Shape shape,std::vector<Byte> bytes) {
 return {std::move(name),scalar,shape,memory(std::move(bytes))};
}
inline Shape matrix(std::uint64_t rows,std::uint64_t width) { return {{rows,width,0,0},2}; }
inline bool require(bool condition,const char* message) {
 if(!condition) std::cerr << message << '\n'; return condition;
}
WeightView fixture(const FormatDescriptor&,std::uint32_t seed);
bool legacy_decode(const WeightView&,std::vector<float>&);
}
