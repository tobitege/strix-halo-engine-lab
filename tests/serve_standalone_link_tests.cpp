// Link check of the standalone serving graph (cmake/StrixServeStandalone.cmake): the API library of the
// STRIX_CUDA_QWEN_ONLY configuration, linked here without the device gate or the core library. The stored member
// pointer pulls src/serve/api.cpp into the link, so every symbol that file calls (the shared power sampler among
// them) must come from a library that graph names. This file calls none of them itself.
#include "strix/serve/api.hpp"

#include <iostream>

namespace {
using Handle = void (strix::serve::ApiService::*)(const strix::serve::HttpRequest&, strix::serve::HttpResponseWriter&);
volatile Handle g_handle = &strix::serve::ApiService::handle;  // volatile: the reference stays in the object file
}  // namespace

int main() {
    const Handle handle = g_handle;
    if (!handle) return 1;
    std::cout << "standalone serve link test passed\n";
    return 0;
}
