#include "model.hpp"
#include <stdexcept>

namespace strix::gemma4 {
serve::Json read_json(const std::filesystem::path& path) { return weights::read_json(path); }
namespace model_detail {
const serve::Json& require(const serve::Json& value, const char* key) {
    const auto* p = value.find(key);
    if (!p) throw std::runtime_error(std::string("missing asset field: ") + key);
    return *p;
}
std::uint64_t integer(const serve::Json& value) {
    const auto n = value.as_int();
    if (n < 0) throw std::runtime_error("negative asset extent");
    return std::uint64_t(n);
}
}
SafeFile read_safetensors(const std::filesystem::path& path) { return weights::read_safetensors(path); }
} // namespace strix::gemma4
