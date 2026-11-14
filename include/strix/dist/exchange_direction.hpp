#pragma once
#include <cstdint>

namespace strix::dist {
// Which way one device exchange moves bytes on this rank: both (the symmetric exchange), or one way only.
enum class ExchangeDirection : std::uint8_t { both = 0, send = 1, receive = 2 };
} // namespace strix::dist
