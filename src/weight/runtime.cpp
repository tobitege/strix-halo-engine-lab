#include <strix/weight/runtime.hpp>
namespace strix::weight {
std::string current_compiler_id() {
#if defined(__clang__)
    return "clang/" __clang_version__;
#elif defined(_MSC_VER)
    return "msvc/" + std::to_string(_MSC_FULL_VER);
#elif defined(__GNUC__)
    return "gcc/" __VERSION__;
#else
    return "unknown-compiler";
#endif
}

Status bind_runtime(Backend backend, const RuntimeAdapter& adapter, void* stream, std::uint32_t device,
                    ExecutionContext& context, Hardware& hardware) {
    if (!adapter.callbacks || !adapter.query_hardware) return {Error::unsupported_backend,
        "backend runtime not registered; dependency=vendor allocation/events/copies/capabilities; fixture=backend host boundary; removal=install vendor adapter"};
    Hardware h{};
    auto status=adapter.query_hardware(device,h); if (!status) return status;
    if (h.backend!=backend) return {Error::unsupported_backend,"capability query returned another backend"};
    context={backend,stream,device,adapter.callbacks}; hardware=std::move(h); return {Error::ok,{}};
}
}
