#pragma once
#include <cstdint>
namespace strix::weight::formats {
struct Iq3S {
 static constexpr std::uint32_t elements=256, bytes=110;
 static constexpr std::uint32_t d=0, qs=2, qh=66, signs=74, scales=106;
};
// Pure storage geometry used by both host descriptors and backend accessors.
constexpr std::uint32_t gsq_block_elements(std::uint32_t type) noexcept {
 return type==20?32:type==42?64:256;
}
constexpr std::uint32_t gsq_block_bytes(std::uint32_t type) noexcept {
 return type==20 || type==42?18:type==22?82:type==18?98:type==21?Iq3S::bytes:type==23?136:0;
}
}
