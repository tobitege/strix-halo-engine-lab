#pragma once
#include "strix/env_flags.hpp"
#include "strix/model_content_hash.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <cstdio>

namespace strix {
inline bool mtp_step_logits_digest_enabled() noexcept {
    static const bool enabled=strix::envcfg::flag("STRIX_MTP_STEP_LOGITS_DIGEST", false);
    return enabled;
}
// STRIX_MTP_LOGITS_DIGEST=1 (first character '1'): read once (R2-13; was per step).
inline bool mtp_logits_digest_enabled() noexcept {
    static const bool enabled=strix::envcfg::flag("STRIX_MTP_LOGITS_DIGEST", false);
    return enabled;
}
// All serial ticket consumers retain the same actual completed full-vocabulary
// row. Keeping readback and retention under one policy prevents a ledger-only
// request from reading the row but dropping its hash before ticket publication.
struct MtpSerialLogitsDigestPolicy {
    bool serial=false,mtp=false,step=false;
    [[nodiscard]] bool enabled() const noexcept {return serial || mtp || step;}
    [[nodiscard]] std::optional<ModelContentIdentity> capture(std::span<const float> row) const {
        return enabled() ? std::optional{sha256_bytes(std::as_bytes(row))} : std::nullopt;
    }
};
// Diagnostic-only ordered commitment ledger. Speculative hashes never enter it.
// Chain = SHA256(previous chain [32] || consumed position [8 LE] || row SHA256 [32]).
// A missing, repeated or out-of-order row makes coverage permanently incomplete.
class MtpStepLogitsDigest {
public:
    explicit MtpStepLogitsDigest(std::uint64_t begin):begin_(begin),next_(begin) {}
    void commit(std::uint64_t position,const std::optional<ModelContentIdentity>& row) {
        if(position!=next_ || !row) complete_=false;
        if(!row) return;
        std::array<std::byte,72> input{};
        for(std::size_t j=0;j<32;++j) {
            input[j]=static_cast<std::byte>(chain_.bytes[j]);
            input[40+j]=static_cast<std::byte>(row->bytes[j]);
        }
        for(unsigned j=0;j<8;++j) input[32+j]=static_cast<std::byte>((position>>(8U*j))&255U);
        chain_=sha256_bytes(input);++count_;
        if(position==UINT64_MAX) complete_=false;
        else next_=position+1U;
    }
    [[nodiscard]] bool complete(std::uint64_t end)const noexcept {return complete_ && count_ && next_==end && end>=begin_ && count_==end-begin_;}
    [[nodiscard]] std::uint64_t begin()const noexcept {return begin_;}
    [[nodiscard]] std::uint64_t count()const noexcept {return count_;}
    [[nodiscard]] ModelContentIdentity chain()const noexcept {return chain_;}
private:
    std::uint64_t begin_=0,next_=0,count_=0;
    bool complete_=true;
    ModelContentIdentity chain_{};
};
// Same wire grammar as the MTP ticket ledger, for actual ordinary serial rows.
class SerialStepLogitsDiagnostic {
public:
    SerialStepLogitsDiagnostic(std::uint64_t begin,std::uint64_t vocab):vocab_(vocab) {
        if(mtp_step_logits_digest_enabled()) {
            ledger_.emplace(begin);
            std::fprintf(stderr,"mtp-step-logits begin stream=%p begin=%llu vocab=%llu\n",static_cast<void*>(this),
                static_cast<unsigned long long>(begin),static_cast<unsigned long long>(vocab_));
        }
    }
    bool empty() const {return !ledger_ || !ledger_->count();}
    bool complete(std::uint64_t end) const {return ledger_ && ledger_->complete(end);}
    ModelContentIdentity chain() const {return ledger_?ledger_->chain():ModelContentIdentity{};}
    void row(std::uint64_t position,std::span<const float> values) {
        if(!ledger_ || position<ledger_->begin()) return;
        const auto hash=values.size()==vocab_ ? std::optional{sha256_bytes(std::as_bytes(values))}:std::nullopt;
        ledger_->commit(position,hash);
        std::fprintf(stderr,"mtp-step-logits committed stream=%p position=%llu vocab=%llu sha256=%s\n",static_cast<void*>(this),
            static_cast<unsigned long long>(position),static_cast<unsigned long long>(vocab_),
            hash?model_content_identity_hex(*hash).c_str():"NO-REFERENCE");
    }
    void finish(std::uint64_t end) {
        if(!ledger_) return;
        std::fprintf(stderr,"mtp-step-logits summary stream=%p begin=%llu end=%llu count=%llu vocab=%llu coverage=%s sha256=%s\n",
            static_cast<void*>(this),static_cast<unsigned long long>(ledger_->begin()),static_cast<unsigned long long>(end),
            static_cast<unsigned long long>(ledger_->count()),static_cast<unsigned long long>(vocab_),
            ledger_->complete(end)?"COMPLETE":"NO-REFERENCE",model_content_identity_hex(ledger_->chain()).c_str());
    }
private:
    std::uint64_t vocab_;
    std::optional<MtpStepLogitsDigest> ledger_;
};
} // namespace strix
