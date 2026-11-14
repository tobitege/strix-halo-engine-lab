#include "strix/host_memory.hpp"

#include <iostream>
#include <string>

int main() {
    const auto available = strix::separate_host_memory_available();
    // A separate host pool reports its available RAM; a shared pool has no observation of its own.
    if (available.has_value() != strix::host_memory_is_separate_pool() || (available && *available == 0U)) {
        std::cerr << "FAIL: the host pool observation does not match the platform's memory model\n";
        return 1;
    }
    std::cout << "host memory: " << (available ? "separate pool, available bytes " + std::to_string(*available)
                                               : std::string("one pool shared with the GPU")) << '\n';
}
