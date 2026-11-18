#pragma once
#include <cstdint>
namespace strix::xport {
struct Stats {
    std::uint64_t exchanges = 0, bytes = 0;
    double producer_wait = 0.0;
    double wire = 0.0;
    double check = 0.0;
    std::uint64_t staging_bytes = 0, reuse_dependencies = 0; // event edges, never elapsed time
    std::uint64_t splits = 0;   // split-producer frames (Channel::split_head)
    double head_wire = 0.0;     // their head rounds (moved while the tail was still being produced)
    double tail_wait = 0.0;     // after the head round: waiting for the tail producer (head round fully hidden)
};
} // namespace strix::xport
