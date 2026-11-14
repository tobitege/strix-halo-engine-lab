#include "strix/serve/token_memo.hpp"

#include "strix/env_flags.hpp"

#include <algorithm>
#include <cstring>

namespace strix::serve {
namespace {

// A text that continues a kept one differs from it at most in the last bytes of the kept text (the generation
// prompt that the next turn replaces with the reply).
constexpr std::size_t continuation_slack = 256;

std::size_t common_prefix(std::string_view a, std::string_view b) noexcept {
    const std::size_t n = std::min(a.size(), b.size());
    constexpr std::size_t block = 4096;
    std::size_t i = 0;
    while (i + block <= n && std::memcmp(a.data() + i, b.data() + i, block) == 0) i += block;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

} // namespace

bool TokenMemo::enabled_by_default() {
    static const bool on = envcfg::flag("STRIX_SERVE_TOKEN_MEMO", true, EnvFlagMatch::exact);
    return on;
}

TokenMemo::Limits TokenMemo::default_limits() {
    static const Limits limits = [] {
        Limits out;
        out.max_bytes = envcfg::number<std::size_t>("STRIX_SERVE_TOKEN_MEMO_MB", 64U, 1U, 4096U) << 20;
        out.max_entries = envcfg::number<std::size_t>("STRIX_SERVE_TOKEN_MEMO_ENTRIES", 32U, 1U, 1024U);
        return out;
    }();
    return limits;
}

std::vector<std::uint32_t> TokenMemo::encode(const ApiTokenizer& tokenizer, std::string_view text, bool parse_special,
                                             std::uint64_t salt) {
    return encode(Key{salt, tokenizer.memo_identity(), parse_special}, tokenizer.mark_lookahead(), text,
                  [&](std::string_view tail, std::vector<EncodeMark>& marks) {
                      return tokenizer.encode_marked(tail, parse_special, marks);
                  });
}

std::vector<std::uint32_t> TokenMemo::encode(std::uint64_t identity, std::size_t lookahead, std::string_view text,
                                             const Encode& encode) {
    return this->encode(Key{identity, {}, false}, lookahead, text, encode);
}

std::vector<std::uint32_t> TokenMemo::encode(Key key, std::size_t lookahead, std::string_view text,
                                             const Encode& encode) {
    EntryPtr base;
    EntryPtr whole;
    std::size_t base_marks = 0; // restart points of `base` that are reused
    std::size_t base_common = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.calls;
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            const Entry& entry = **it;
            if (entry.key != key) continue;
            const std::size_t common = common_prefix(entry.text, text);
            if (common == text.size() && common == entry.text.size()) {
                ++stats_.whole;
                stats_.reused_bytes += text.size();
                entries_.splice(entries_.begin(), entries_, it);
                whole = entries_.front();
                break;
            }
            if (common < lookahead) continue;
            // restart points whose lookahead window lies inside the shared bytes
            const auto end = std::upper_bound(entry.marks.begin(), entry.marks.end(), common - lookahead,
                                              [](std::size_t limit, const EncodeMark& mark) { return limit < mark.bytes; });
            const auto usable = static_cast<std::size_t>(end - entry.marks.begin());
            if (usable == 0U) continue;
            if (!base || entry.marks[usable - 1U].bytes > base->marks[base_marks - 1U].bytes) {
                base = *it;
                base_marks = usable;
                base_common = common;
            }
        }
    }

    if (whole) return whole->ids; // pin under the lock; copy outside it
    auto made = std::make_shared<Entry>();
    made->key = std::move(key);
    std::size_t reused = 0;
    if (base) {
        const EncodeMark at = base->marks[base_marks - 1U];
        reused = at.bytes;
        made->ids.assign(base->ids.begin(), base->ids.begin() + static_cast<std::ptrdiff_t>(at.ids));
        made->marks.assign(base->marks.begin(), base->marks.begin() + static_cast<std::ptrdiff_t>(base_marks));
        std::vector<EncodeMark> tail_marks;
        const auto tail = encode(text.substr(reused), tail_marks);
        made->ids.insert(made->ids.end(), tail.begin(), tail.end());
        for (const auto& mark : tail_marks) made->marks.push_back({mark.bytes + reused, mark.ids + at.ids});
    } else {
        made->ids = encode(text, made->marks);
    }
    made->text.assign(text);
    std::vector<std::uint32_t> ids = made->ids;

    std::lock_guard<std::mutex> lock(mutex_);
    if (base) ++stats_.partial; else ++stats_.missed;
    stats_.reused_bytes += reused;
    stats_.encoded_bytes += text.size() - reused;
    const std::size_t size = made->bytes();
    if (size > limits_.max_bytes) return ids;
    for (auto it = entries_.begin(); it != entries_.end();) {
        const Entry& entry = **it;
        // another request stored this text meanwhile, or this text continues the entry it was built from
        const bool same = entry.key == made->key && entry.text == made->text;
        const bool continued = *it == base && base_common + continuation_slack >= entry.text.size() &&
                               made->text.size() >= entry.text.size();
        if (same || continued) {
            bytes_ -= entry.bytes();
            it = entries_.erase(it);
        } else {
            ++it;
        }
    }
    entries_.push_front(std::move(made));
    bytes_ += size;
    while (entries_.size() > limits_.max_entries || bytes_ > limits_.max_bytes) {
        bytes_ -= entries_.back()->bytes();
        entries_.pop_back();
        ++stats_.evictions;
    }
    return ids;
}

TokenMemo::Stats TokenMemo::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Stats out = stats_;
    out.entries = entries_.size();
    out.bytes = bytes_;
    return out;
}

void TokenMemo::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    bytes_ = 0;
}

} // namespace strix::serve
