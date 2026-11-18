#pragma once
#include "types.hpp"
#include <istream>
namespace strix::weight {
struct Source {
  virtual ~Source() = default;
  virtual Status read(ByteRange, std::span<Byte>) const = 0;
  virtual std::uint64_t size() const noexcept = 0;
  virtual Digest256 identity() const = 0;
  // Stable shard namespace, independent of a tensor name. Not a content digest.
  virtual std::string shard_name() const { return {}; }
  // Optional positioned-I/O cache hint; it never changes the stored bytes.
  virtual void discard(ByteRange) const noexcept {}
};
Digest256 digest_bytes(std::span<const Byte>);
Status open_source(const std::string&, std::shared_ptr<const Source>&, std::string shard = {});
std::shared_ptr<const Source> memory_source(std::span<const Byte>, std::string shard = {});
// The span must remain immutable; owner retains the complete referenced allocation.
std::shared_ptr<const Source> view_source(std::span<const Byte>, std::shared_ptr<const void>, std::string shard = {});
std::unique_ptr<std::istream> source_stream(std::shared_ptr<const Source>);
TensorId tensor_id(const Source&, std::string_view, const Digest256& header = {});
Status digest_slice(const Slice&, Digest256&);
std::shared_ptr<const Source> namespace_source(std::shared_ptr<const Source>, std::string shard);
Status read_slice(const Slice&, std::span<Byte>);
}
