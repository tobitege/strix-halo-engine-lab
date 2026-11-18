#pragma once
#include <algorithm>
#include <cstdint>
#include <regex>
#include <stdexcept>
#include <string>
namespace strix::glm::rtxpf::producer {
inline void require_pro_uuid(const std::string& value){
    if(!std::regex_match(value,std::regex("GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")))
        throw std::runtime_error("CUDA_VISIBLE_DEVICES must be one full PRO GPU UUID");
}
inline bool admit(std::uint64_t bytes,std::uint64_t free,std::uint64_t reserve){return free>=reserve&&bytes<=free-reserve;}
struct TierTargets {unsigned device,host,file;};
inline TierTargets targets(unsigned inputs){return inputs<=8192?TierTargets{20,13,10}:inputs<=32768?TierTargets{19,13,11}:TierTargets{14,13,16};}
inline bool caps_require_file_tier(unsigned inputs,unsigned device,unsigned pinned){return std::min(targets(inputs).device,device)+pinned<43;}
inline unsigned layer_limit(const std::string& value){
    if(value.empty()||value.find_first_not_of("0123456789")!=std::string::npos||value.size()>2)
        throw std::runtime_error("layer limit must be 0..43");
    auto n=std::stoul(value);if(n>43)throw std::runtime_error("layer limit must be 0..43");return unsigned(n);
}
inline bool file_tier_covered(unsigned layers,std::uint64_t bytes,unsigned ring_mask,bool both){
    return layers>=(both?2u:1u)&&bytes>0&&(!both||ring_mask==3);
}
inline std::uint64_t host_budget(std::uint64_t physical,std::uint64_t commit,unsigned floor_gib=8){
    const auto reserve=std::uint64_t(floor_gib)<<30;
    return std::min(physical,commit)>reserve?std::min(physical,commit)-reserve:0;
}
}
