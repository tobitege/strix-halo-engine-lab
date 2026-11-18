#include "runtime.hpp"
namespace strix::intel::weight {
strix::weight::Status bind_runtime(const strix::weight::RuntimeAdapter& api, void* stream,
    std::uint32_t device, strix::weight::ExecutionContext& context, strix::weight::Hardware& hardware) {
    return strix::weight::bind_runtime(strix::weight::Backend::intel, api, stream, device, context, hardware);
}
}
