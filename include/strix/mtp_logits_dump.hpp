#pragma once
#include "strix/env_flags.hpp"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>

namespace strix {
// Explicit diagnostic only. Records native little-endian uint64 metadata followed
// by vocab FP32 values. Input-prefix is the current window, NOT the full context;
// callers must match the committed token ledger before comparing two records.
inline const std::string& mtp_logits_dump_path() {
    static const std::string path=[] {const auto* p=envcfg::text("STRIX_MTP_LOGITS_DUMP");return p?std::string(p):std::string{};}();
    return path;
}
inline bool mtp_logits_dump_enabled() {return !mtp_logits_dump_path().empty();}
inline std::uint64_t mtp_input_prefix_hash(std::span<const std::uint32_t> tokens) {
    std::uint64_t h=14695981039346656037ULL;
    for(const auto token:tokens)for(unsigned s=0;s<32;s+=8) {h^=(token>>s)&255U;h*=1099511628211ULL;}
    return h;
}
inline void dump_mtp_logits(std::uint64_t position,std::uint64_t base,std::uint32_t row,std::uint32_t rows,
    std::span<const std::uint32_t> inputs,std::uint64_t lineage,std::span<const float> logits) {
    if(!mtp_logits_dump_enabled())return;
    static std::mutex mutex;
    const std::lock_guard lock(mutex);
    struct Output {
        std::FILE* file=std::fopen(mtp_logits_dump_path().c_str(),"ab");
        ~Output(){if(file)std::fclose(file);}
    };
    static Output output;
    if(!output.file)throw std::runtime_error("cannot open STRIX_MTP_LOGITS_DUMP");
    // magic/version SMTPLGT2, position, base, row, physical rows, vocab,
    // consumed input, window-input-prefix FNV1a64, arithmetic lineage, monotonic ns.
    const auto now=static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    const std::array<std::uint64_t,10> header{0x3254474c50544d53ULL,position,base,row,rows,logits.size(),
        inputs.empty()?UINT64_MAX:inputs.back(),mtp_input_prefix_hash(inputs),lineage,now};
    if(std::fwrite(header.data(),sizeof(header[0]),header.size(),output.file)!=header.size() ||
        std::fwrite(logits.data(),sizeof(float),logits.size(),output.file)!=logits.size() || std::fflush(output.file))
        throw std::runtime_error("cannot write STRIX_MTP_LOGITS_DUMP");
}
}
