#pragma once
// TP2 Communicator setup shared by strix-glm5-rank and strix-glm5-serve (the transport: strix::dist::open_pair).
#include "strix/dist/communicator.hpp"
#include "strix/dist/open_transport.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace strix::glm::rank_transport {

// The engine's Communicator: one exchange carries up to max(8, exchange_rows) rows of 4096 floats.
inline std::unique_ptr<strix::dist::Communicator> make_communicator(std::uint32_t rank, std::unique_ptr<strix::dist::Transport> transport,
                                                                    std::unique_ptr<strix::dist::DeviceExchange> direct,
                                                                    unsigned exchange_rows) {
    strix::dist::Group group;
    group.rank = rank;
    group.world = 2;
    group.plan = 0x474c4d35ULL;  // "GLM5"
    const std::size_t max_bytes = std::size_t{std::max(8U, exchange_rows)} * 4096 * sizeof(float) + 4096;
    return std::make_unique<strix::dist::Communicator>(group, std::move(transport), std::move(direct), max_bytes);
}

}  // namespace strix::glm::rank_transport
