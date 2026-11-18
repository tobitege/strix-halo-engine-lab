#pragma once

#include "strix/device/declaration.hpp"

namespace strix::device {

struct Capabilities {
    int major = 0, minor = 0;  // compute capability; zero when unavailable
    bool pdl = false, clusters = false;
};

// Non-owning runtime binding of an already resolved descriptor. No enumeration or device selection.
// The descriptor must outlive this binding; a backend supplies the ordinal capability query.
struct Runtime {
    const Descriptor& descriptor;
    Vendor vendor;
    std::string_view arch;
    int ordinal;
    using Query = Capabilities (*)(int) noexcept;
    Query query;

    Runtime(const Descriptor& device, int ordinal_, Query query_) noexcept
        : descriptor(device), vendor(device.vendor), arch(device.arch), ordinal(ordinal_), query(query_) {}
    [[nodiscard]] Capabilities capabilities() const noexcept { return query ? query(ordinal) : Capabilities{}; }
};

}  // namespace strix::device
