#pragma once
#include <cstddef>
#include <span>
#include <vector>
#include <cstring>
#include <stdexcept>
namespace strix::glm::rtxpf::producer {
inline std::span<const std::byte> kda_tp_state(std::span<const std::byte> full,unsigned rank){
    if(rank>1||full.size()!=64*65536)throw std::runtime_error("full KDA state geometry");
    return full.subspan(rank*32*65536,32*65536);
}
inline std::vector<std::byte> kda_tp_conv(std::span<const std::byte> full,unsigned rank){
    if(rank>1||full.size()!=64*4608)throw std::runtime_error("full KDA convolution geometry");
    std::vector<std::byte> out(32*4608);
    for(unsigned part=0;part<3;++part)std::memcpy(out.data()+part*32*128*3*4,full.data()+(part*64+rank*32)*128*3*4,32*128*3*4);
    return out;
}
}
