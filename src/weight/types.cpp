#include <strix/weight/types.hpp>
namespace strix::weight {
Status::operator bool() const { return code == Error::ok; }
std::string serialize_id(Id id) { return std::string(id.name) + "@" + std::to_string(id.revision); }
bool operator==(Id a, Id b) noexcept { return a.name == b.name && a.revision == b.revision; }
}
