#include "minimax/serve/minimax_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iterator>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace strix::minimax::serve {
namespace {

struct MmRange { std::uint32_t first, last; std::uint8_t value; };
struct MmDecomp { std::uint32_t cp, first, second; };
struct MmCompose { std::uint32_t first, second, composite; };
#include "minimax_unicode_tables.inc"

// ---------------------------------------------------------------------------------------------
// Code points. Invalid UTF-8 bytes become kInvalidBase + byte: an opaque character that is never
// normalized, is a "symbol" for the regex and is written back as the original byte.
constexpr std::uint32_t kInvalidBase = 0x110000U;

std::vector<std::uint32_t> to_code_points(std::string_view text) {
    std::vector<std::uint32_t> out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        const auto b0 = static_cast<unsigned char>(text[i]);
        std::size_t length = 1;
        std::uint32_t cp = b0;
        if (b0 >= 0x80U) {
            if (b0 >= 0xC2U && b0 <= 0xDFU) length = 2;
            else if (b0 >= 0xE0U && b0 <= 0xEFU) length = 3;
            else if (b0 >= 0xF0U && b0 <= 0xF4U) length = 4;
            else length = 0;
            bool valid = length != 0U && i + length <= text.size();
            std::uint32_t value = length == 2U ? (b0 & 0x1FU) : length == 3U ? (b0 & 0x0FU) : (b0 & 0x07U);
            for (std::size_t k = 1; valid && k < length; ++k) {
                const auto b = static_cast<unsigned char>(text[i + k]);
                if ((b & 0xC0U) != 0x80U) valid = false;
                value = (value << 6U) | (b & 0x3FU);
            }
            if (valid && ((length == 3U && (value < 0x800U || (value >= 0xD800U && value <= 0xDFFFU))) ||
                          (length == 4U && (value < 0x10000U || value > 0x10FFFFU)))) valid = false;
            if (valid) cp = value;
            else { cp = kInvalidBase + b0; length = 1; }
        }
        out.push_back(cp);
        i += length;
    }
    return out;
}

void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp >= kInvalidBase) { out.push_back(static_cast<char>(cp - kInvalidBase)); return; }
    if (cp < 0x80U) out.push_back(static_cast<char>(cp));
    else if (cp < 0x800U) {
        out.push_back(static_cast<char>(0xC0U | (cp >> 6U)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    } else if (cp < 0x10000U) {
        out.push_back(static_cast<char>(0xE0U | (cp >> 12U)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    } else {
        out.push_back(static_cast<char>(0xF0U | (cp >> 18U)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 12U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    }
}

template <std::size_t N>
std::uint8_t range_value(const MmRange (&table)[N], std::uint32_t cp) {
    std::size_t lo = 0, hi = N;
    while (lo < hi) {
        const auto mid = (lo + hi) / 2U;
        if (table[mid].last < cp) lo = mid + 1U;
        else hi = mid;
    }
    return lo < N && table[lo].first <= cp ? table[lo].value : 0U;
}

// ---------------------------------------------------------------------------------------------
// NFC (canonical decomposition, canonical ordering, canonical composition), Unicode 9.0 tables.
constexpr std::uint32_t kSBase = 0xAC00U, kLBase = 0x1100U, kVBase = 0x1161U, kTBase = 0x11A7U;
constexpr std::uint32_t kLCount = 19U, kVCount = 21U, kTCount = 28U, kNCount = kVCount * kTCount, kSCount = kLCount * kNCount;

std::uint8_t ccc(std::uint32_t cp) { return cp >= kInvalidBase ? 0U : range_value(kMmCccRanges, cp); }

void decompose(std::uint32_t cp, std::vector<std::uint32_t>& out) {
    if (cp >= kSBase && cp < kSBase + kSCount) {
        const auto s = cp - kSBase;
        out.push_back(kLBase + s / kNCount);
        out.push_back(kVBase + (s % kNCount) / kTCount);
        if (s % kTCount) out.push_back(kTBase + s % kTCount);
        return;
    }
    const auto* end = std::end(kMmDecomps);
    const auto* hit = std::lower_bound(std::begin(kMmDecomps), end, cp, [](const MmDecomp& d, std::uint32_t v) { return d.cp < v; });
    if (hit == end || hit->cp != cp) { out.push_back(cp); return; }
    decompose(hit->first, out);
    if (hit->second) decompose(hit->second, out);
}

std::uint32_t compose(std::uint32_t a, std::uint32_t b) {
    if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount)
        return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
    if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0U && b > kTBase && b < kTBase + kTCount)
        return a + (b - kTBase);
    const auto* end = std::end(kMmCompositions);
    const auto* hit = std::lower_bound(std::begin(kMmCompositions), end, std::pair{a, b},
        [](const MmCompose& c, const std::pair<std::uint32_t, std::uint32_t>& v) {
            return c.first < v.first || (c.first == v.first && c.second < v.second);
        });
    return hit != end && hit->first == a && hit->second == b ? hit->composite : 0U;
}

std::vector<std::uint32_t> nfc_code_points(const std::vector<std::uint32_t>& in) {
    // Below U+0300 every character is NFC-stable and never the second of a composition pair.
    if (std::all_of(in.begin(), in.end(), [](std::uint32_t cp) { return cp < 0x300U; })) return in;
    std::vector<std::uint32_t> d;
    d.reserve(in.size() + 8U);
    for (const auto cp : in) {
        if (cp >= kInvalidBase) d.push_back(cp);
        else decompose(cp, d);
    }
    // canonical ordering: stable sort of every run of non-starters by combining class
    for (std::size_t i = 0; i < d.size();) {
        if (ccc(d[i]) == 0U) { ++i; continue; }
        std::size_t j = i;
        while (j < d.size() && ccc(d[j]) != 0U) ++j;
        std::stable_sort(d.begin() + static_cast<std::ptrdiff_t>(i), d.begin() + static_cast<std::ptrdiff_t>(j),
            [](std::uint32_t x, std::uint32_t y) { return ccc(x) < ccc(y); });
        i = j;
    }
    std::vector<std::uint32_t> out;
    out.reserve(d.size());
    std::size_t starter = SIZE_MAX;
    int last_ccc = -1;
    for (const auto cp : d) {
        const int cc = ccc(cp);
        if (starter != SIZE_MAX && cp < kInvalidBase && out[starter] < kInvalidBase &&
            (last_ccc < cc || (last_ccc == 0 && out.size() - 1U == starter))) {
            if (const auto c = compose(out[starter], cp); c != 0U) { out[starter] = c; continue; }
        }
        if (cc == 0) { starter = out.size(); last_ccc = 0; }
        else last_ccc = cc;
        out.push_back(cp);
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// Pre-tokenizer: Split(regex, Removed, invert) == the regex matches, alternatives in order:
//   A1 [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   A2 [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   A3 \p{N}{1,3}
//   A4  ?[^\s\p{L}\p{N}]+[\r\n/]*
//   A5 \s*[\r\n]+
//   A6 \s+(?!\S)
//   A7 \s+
// with backtracking (Oniguruma) semantics. Every character is covered by some alternative, so the
// matches tile the text.
constexpr std::uint8_t kUp = 1U, kLo = 2U, kL = 4U, kN = 8U, kWs = 16U;

bool white_space(std::uint32_t cp) {
    return (cp >= 0x09U && cp <= 0x0DU) || cp == 0x20U || cp == 0x85U || cp == 0xA0U || cp == 0x1680U ||
        (cp >= 0x2000U && cp <= 0x200AU) || cp == 0x2028U || cp == 0x2029U || cp == 0x202FU || cp == 0x205FU || cp == 0x3000U;
}

std::uint8_t char_flags(std::uint32_t cp) {
    if (cp >= kInvalidBase) return 0U;
    if (cp < 0x80U) {
        if ((cp >= 'a' && cp <= 'z')) return kLo | kL;
        if ((cp >= 'A' && cp <= 'Z')) return kUp | kL;
        if (cp >= '0' && cp <= '9') return kN;
        return white_space(cp) ? kWs : 0U;
    }
    return static_cast<std::uint8_t>(range_value(kMmClassRanges, cp) | (white_space(cp) ? kWs : 0U));
}

std::vector<std::pair<std::size_t, std::size_t>> pre_split(const std::vector<std::uint32_t>& c) {
    const auto n = c.size();
    std::vector<std::uint8_t> f(n);
    for (std::size_t k = 0; k < n; ++k) f[k] = char_flags(c[k]);
    const auto up = [&](std::size_t k) { return k < n && (f[k] & kUp) != 0U; };
    const auto lo = [&](std::size_t k) { return k < n && (f[k] & kLo) != 0U; };
    const auto crlf = [&](std::size_t k) { return k < n && (c[k] == '\r' || c[k] == '\n'); };
    const auto prefix = [&](std::size_t k) { return k < n && !crlf(k) && (f[k] & (kL | kN)) == 0U; };
    const auto sym = [&](std::size_t k) { return k < n && (f[k] & (kWs | kL | kN)) == 0U; };
    const auto ws = [&](std::size_t k) { return k < n && (f[k] & kWs) != 0U; };
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)?  (case folding: also 'ſ U+017F for s)
    const auto contraction = [&](std::size_t e) -> std::size_t {
        if (e >= n || c[e] != '\'') return e;
        const auto fold = [&](std::size_t k) -> char {
            if (k >= n) return '\0';
            if (c[k] == 0x17FU) return 's';
            return c[k] < 0x80U ? static_cast<char>(c[k] | 0x20U) : '\0';
        };
        const char a = fold(e + 1U);
        const char b = fold(e + 2U);
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return e + 2U;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return e + 3U;
        return e;
    };
    std::vector<std::pair<std::size_t, std::size_t>> pieces;
    std::size_t i = 0;
    while (i < n) {
        std::size_t end = 0;
        const std::size_t prefix_lengths[2] = {prefix(i) ? 1U : 0U, 0U};
        const std::size_t tries = prefix(i) ? 2U : 1U;
        // A1
        for (std::size_t t = 0; t < tries && !end; ++t) {
            const std::size_t j = i + prefix_lengths[t];
            std::size_t r = j;
            while (up(r)) ++r;
            for (std::size_t k = r + 1U; k-- > j;) {
                if (lo(k)) {
                    std::size_t e = k;
                    while (lo(e)) ++e;
                    end = contraction(e);
                    break;
                }
            }
        }
        // A2
        for (std::size_t t = 0; t < tries && !end; ++t) {
            const std::size_t j = i + prefix_lengths[t];
            if (!up(j)) continue;
            std::size_t e = j;
            while (up(e)) ++e;
            while (lo(e)) ++e;
            end = contraction(e);
        }
        // A3
        if (!end && (f[i] & kN)) {
            end = i;
            while (end < n && end - i < 3U && (f[end] & kN)) ++end;
        }
        // A4
        if (!end) {
            std::size_t start = SIZE_MAX;
            if (c[i] == ' ' && sym(i + 1U)) start = i + 1U;
            else if (sym(i)) start = i;
            if (start != SIZE_MAX) {
                end = start;
                while (sym(end)) ++end;
                while (end < n && (c[end] == '\r' || c[end] == '\n' || c[end] == '/')) ++end;
            }
        }
        // A5-A7
        if (!end && ws(i)) {
            std::size_t run = i;
            while (ws(run)) ++run;
            std::size_t last_newline = SIZE_MAX;
            for (std::size_t k = i; k < run; ++k) if (crlf(k)) last_newline = k;
            if (last_newline != SIZE_MAX) end = last_newline + 1U;
            else if (run == n) end = run;
            else if (run - i >= 2U) end = run - 1U;
            else end = run;
        }
        if (!end) end = i + 1U;  // unreachable: the alternatives cover every character
        pieces.emplace_back(i, end);
        i = end;
    }
    return pieces;
}

// ---------------------------------------------------------------------------------------------
// Byte-level alphabet (GPT-2): printable bytes map to themselves, the others to U+0100.. in order.
std::array<std::uint32_t, 256> byte_to_code_point() {
    std::array<std::uint32_t, 256> out{};
    std::uint32_t extra = 0;
    for (std::uint32_t b = 0; b < 256U; ++b) {
        const bool direct = (b >= '!' && b <= '~') || (b >= 0xA1U && b <= 0xACU) || (b >= 0xAEU);
        out[b] = direct ? b : 256U + extra++;
    }
    return out;
}

// Rust String::from_utf8_lossy: every maximal invalid prefix becomes one U+FFFD.
void append_lossy(std::string& out, std::string_view bytes) {
    std::size_t i = 0;
    const auto n = bytes.size();
    const auto at = [&](std::size_t k) { return static_cast<unsigned char>(bytes[k]); };
    const auto cont = [&](std::size_t k) { return k < n && (at(k) & 0xC0U) == 0x80U; };
    while (i < n) {
        const auto b0 = at(i);
        if (b0 < 0x80U) { out.push_back(static_cast<char>(b0)); ++i; continue; }
        std::size_t width = b0 >= 0xC2U && b0 <= 0xDFU ? 2U : b0 >= 0xE0U && b0 <= 0xEFU ? 3U : b0 >= 0xF0U && b0 <= 0xF4U ? 4U : 0U;
        std::size_t good = 1;  // bytes of the sequence verified so far
        bool ok = width != 0U;
        if (ok && width >= 3U) {
            const bool second_ok = i + 1U < n && [&] {
                const auto b1 = at(i + 1U);
                if (width == 3U) {
                    if (b0 == 0xE0U) return b1 >= 0xA0U && b1 <= 0xBFU;
                    if (b0 == 0xEDU) return b1 >= 0x80U && b1 <= 0x9FU;
                    return b1 >= 0x80U && b1 <= 0xBFU;
                }
                if (b0 == 0xF0U) return b1 >= 0x90U && b1 <= 0xBFU;
                if (b0 == 0xF4U) return b1 >= 0x80U && b1 <= 0x8FU;
                return b1 >= 0x80U && b1 <= 0xBFU;
            }();
            if (second_ok) good = 2; else ok = false;
        }
        while (ok && good < width) {
            if (cont(i + good)) ++good;
            else ok = false;
        }
        if (ok) { out.append(bytes.substr(i, width)); i += width; }
        else { out += "\xEF\xBF\xBD"; i += good; }
    }
}

// Minimal JSON DOM for tokenizer.json. (strix::serve::parse_json checks duplicate object keys linearly,
// which is quadratic on the 200k-entry vocabulary: ~20 s. This reader keeps every member, in order.)
struct TjValue {
    enum class Kind : std::uint8_t { null, boolean, number, string, array, object };
    Kind kind = Kind::null;
    bool boolean = false;
    std::string text;  // string value or number source
    std::vector<TjValue> array;
    std::vector<std::pair<std::string, TjValue>> object;

    [[nodiscard]] bool is_null() const { return kind == Kind::null; }
    [[nodiscard]] bool is_bool() const { return kind == Kind::boolean; }
    [[nodiscard]] bool is_string() const { return kind == Kind::string; }
    [[nodiscard]] bool is_array() const { return kind == Kind::array; }
    [[nodiscard]] bool is_object() const { return kind == Kind::object; }
    [[nodiscard]] bool as_bool() const { expect(Kind::boolean); return boolean; }
    [[nodiscard]] const std::string& as_string() const { expect(Kind::string); return text; }
    [[nodiscard]] const std::vector<TjValue>& as_array() const { expect(Kind::array); return array; }
    [[nodiscard]] const std::vector<std::pair<std::string, TjValue>>& as_object() const { expect(Kind::object); return object; }
    [[nodiscard]] std::size_t size() const { return kind == Kind::array ? array.size() : object.size(); }
    [[nodiscard]] std::int64_t as_int() const {
        expect(Kind::number);
        std::int64_t v = 0;
        const auto r = std::from_chars(text.data(), text.data() + text.size(), v);
        if (r.ec != std::errc() || r.ptr != text.data() + text.size())
            throw std::runtime_error("minimax tokenizer: JSON number is not an integer");
        return v;
    }
    [[nodiscard]] const TjValue* find(std::string_view key) const {
        if (kind != Kind::object) return nullptr;
        for (const auto& [k, v] : object) if (k == key) return &v;
        return nullptr;
    }
    void expect(Kind k) const {
        if (kind != k) throw std::runtime_error("minimax tokenizer: unexpected JSON type in tokenizer.json");
    }
};

class TjParser {
public:
    explicit TjParser(std::string_view text) : s_(text) {}
    TjValue parse_document() {
        TjValue v = parse(0);
        skip();
        if (at_ != s_.size()) fail("trailing characters");
        return v;
    }

private:
    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error(std::string("minimax tokenizer: invalid JSON (") + what + ") at byte " + std::to_string(at_));
    }
    void skip() {
        while (at_ < s_.size() && (s_[at_] == ' ' || s_[at_] == '\n' || s_[at_] == '\r' || s_[at_] == '\t')) ++at_;
    }
    bool eat(char c) {
        skip();
        if (at_ < s_.size() && s_[at_] == c) { ++at_; return true; }
        return false;
    }
    void literal(std::string_view word) {
        if (s_.substr(at_, word.size()) != word) fail("bad literal");
        at_ += word.size();
    }
    std::uint32_t hex4() {
        if (at_ + 4U > s_.size()) fail("short \\u escape");
        std::uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s_[at_++];
            v <<= 4U;
            if (c >= '0' && c <= '9') v |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<std::uint32_t>(c - 'A' + 10);
            else fail("bad \\u escape");
        }
        return v;
    }
    std::string string() {
        if (!eat('"')) fail("expected string");
        std::string out;
        while (true) {
            const auto run = s_.find_first_of("\"\\", at_);
            if (run == std::string_view::npos) fail("unterminated string");
            out.append(s_.substr(at_, run - at_));
            at_ = run;
            if (s_[at_++] == '"') return out;
            if (at_ >= s_.size()) fail("unterminated escape");
            const char e = s_[at_++];
            switch (e) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                std::uint32_t cp = hex4();
                if (cp >= 0xD800U && cp <= 0xDBFFU && s_.substr(at_, 2) == "\\u") {
                    const auto save = at_;
                    at_ += 2;
                    const auto low = hex4();
                    if (low >= 0xDC00U && low <= 0xDFFFU) cp = 0x10000U + ((cp - 0xD800U) << 10U) + (low - 0xDC00U);
                    else at_ = save;
                }
                if (cp >= 0xD800U && cp <= 0xDFFFU) cp = 0xFFFDU;  // lone surrogate (serde_json would reject it)
                append_utf8(out, cp);
                break;
            }
            default: fail("bad escape");
            }
        }
    }
    TjValue parse(unsigned depth) {
        if (depth > 256U) fail("too deep");
        skip();
        if (at_ >= s_.size()) fail("unexpected end");
        TjValue v;
        const char c = s_[at_];
        if (c == '{') {
            ++at_;
            v.kind = TjValue::Kind::object;
            if (eat('}')) return v;
            do {
                skip();
                auto key = string();
                if (!eat(':')) fail("expected ':'");
                v.object.emplace_back(std::move(key), parse(depth + 1U));
            } while (eat(','));
            if (!eat('}')) fail("expected '}'");
        } else if (c == '[') {
            ++at_;
            v.kind = TjValue::Kind::array;
            if (eat(']')) return v;
            do v.array.push_back(parse(depth + 1U));
            while (eat(','));
            if (!eat(']')) fail("expected ']'");
        } else if (c == '"') {
            v.kind = TjValue::Kind::string;
            v.text = string();
        } else if (c == 't') {
            literal("true");
            v.kind = TjValue::Kind::boolean;
            v.boolean = true;
        } else if (c == 'f') {
            literal("false");
            v.kind = TjValue::Kind::boolean;
        } else if (c == 'n') {
            literal("null");
        } else {
            const auto begin = at_;
            while (at_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[at_])) || s_[at_] == '-' || s_[at_] == '+' ||
                                       s_[at_] == '.' || s_[at_] == 'e' || s_[at_] == 'E')) ++at_;
            if (at_ == begin) fail("unexpected character");
            v.kind = TjValue::Kind::number;
            v.text = std::string(s_.substr(begin, at_ - begin));
        }
        return v;
    }
    std::string_view s_;
    std::size_t at_ = 0;
};

std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("minimax tokenizer: cannot open " + path);
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

}  // namespace

struct MinimaxTokenizer::Impl {
    struct Added { std::string content; std::uint32_t id; bool special; };
    std::vector<std::string> vocab;           // model vocabulary, byte-level text by id
    std::vector<std::string> vocab_bytes;     // decoded bytes by id
    std::unordered_map<std::string, std::uint32_t> vocab_ids;
    std::vector<Added> added;                 // tokenizer.json order
    std::unordered_map<std::uint32_t, std::size_t> added_by_id;
    std::array<bool, 256> added_first{};
    std::unordered_map<std::uint64_t, std::pair<std::uint32_t, std::uint32_t>> merges;  // (left,right) -> (rank, merged)
    std::array<std::uint32_t, 256> byte_token{};  // id of each byte's one-character token
    std::uint32_t bos = UINT32_MAX, eos = UINT32_MAX, pad = UINT32_MAX;
    std::size_t size = 0;
    mutable std::mutex cache_mutex;
    mutable std::unordered_map<std::string, std::vector<std::uint32_t>> cache;

    static std::uint64_t pair_key(std::uint32_t a, std::uint32_t b) { return (static_cast<std::uint64_t>(a) << 32U) | b; }

    // HF BPE `Word::merge_all`: a min-heap of (rank, position); a popped pair is applied when both
    // symbols still exist and still form that merge.
    void bpe(std::string_view piece, std::vector<std::uint32_t>& out) const {
        if (piece.size() > 1U && piece.size() <= 64U) {
            const std::lock_guard lock(cache_mutex);
            if (const auto hit = cache.find(std::string(piece)); hit != cache.end()) {
                out.insert(out.end(), hit->second.begin(), hit->second.end());
                return;
            }
        }
        const auto n = piece.size();
        std::vector<std::uint32_t> id(n);
        std::vector<std::ptrdiff_t> prev(n), next(n);
        std::vector<bool> alive(n, true);
        for (std::size_t k = 0; k < n; ++k) {
            id[k] = byte_token[static_cast<unsigned char>(piece[k])];
            prev[k] = static_cast<std::ptrdiff_t>(k) - 1;
            next[k] = k + 1U < n ? static_cast<std::ptrdiff_t>(k + 1U) : -1;
        }
        struct Item { std::uint32_t rank; std::size_t pos; std::uint32_t merged; };
        const auto later = [](const Item& x, const Item& y) { return x.rank != y.rank ? x.rank > y.rank : x.pos > y.pos; };
        std::priority_queue<Item, std::vector<Item>, decltype(later)> heap(later);
        for (std::size_t k = 0; k + 1U < n; ++k)
            if (const auto m = merges.find(pair_key(id[k], id[k + 1U])); m != merges.end())
                heap.push({m->second.first, k, m->second.second});
        while (!heap.empty()) {
            const auto top = heap.top();
            heap.pop();
            if (!alive[top.pos] || next[top.pos] < 0) continue;
            const auto right = static_cast<std::size_t>(next[top.pos]);
            const auto m = merges.find(pair_key(id[top.pos], id[right]));
            if (m == merges.end() || m->second.second != top.merged) continue;
            id[top.pos] = top.merged;
            alive[right] = false;
            next[top.pos] = next[right];
            if (next[right] >= 0) prev[static_cast<std::size_t>(next[right])] = static_cast<std::ptrdiff_t>(top.pos);
            if (prev[top.pos] >= 0) {
                const auto p = static_cast<std::size_t>(prev[top.pos]);
                if (const auto q = merges.find(pair_key(id[p], id[top.pos])); q != merges.end())
                    heap.push({q->second.first, p, q->second.second});
            }
            if (next[top.pos] >= 0) {
                const auto nx = static_cast<std::size_t>(next[top.pos]);
                if (const auto q = merges.find(pair_key(id[top.pos], id[nx])); q != merges.end())
                    heap.push({q->second.first, top.pos, q->second.second});
            }
        }
        std::vector<std::uint32_t> result;
        for (std::ptrdiff_t k = n ? 0 : -1; k >= 0; k = next[static_cast<std::size_t>(k)]) result.push_back(id[static_cast<std::size_t>(k)]);
        if (piece.size() > 1U && piece.size() <= 64U) {
            const std::lock_guard lock(cache_mutex);
            if (cache.size() > 500000U) cache.clear();
            cache.emplace(std::string(piece), result);
        }
        out.insert(out.end(), result.begin(), result.end());
    }

    void encode_text(std::string_view text, std::vector<std::uint32_t>& out) const {
        if (text.empty()) return;
        const auto points = nfc_code_points(to_code_points(text));
        std::string piece;
        for (const auto& [a, b] : pre_split(points)) {
            piece.clear();
            for (std::size_t k = a; k < b; ++k) append_utf8(piece, points[k]);
            bpe(piece, out);
        }
    }
};

MinimaxTokenizer::MinimaxTokenizer() : impl_(std::make_unique<Impl>()) {}
MinimaxTokenizer::~MinimaxTokenizer() = default;

std::shared_ptr<const MinimaxTokenizer> MinimaxTokenizer::load(const std::string& tokenizer_json_path) {
    const auto root = TjParser(read_file(tokenizer_json_path)).parse_document();
    const auto* model = root.find("model");
    if (!model || !model->is_object()) throw std::runtime_error("minimax tokenizer: tokenizer.json has no model");
    if (const auto* type = model->find("type"); !type || !type->is_string() || type->as_string() != "BPE")
        throw std::runtime_error("minimax tokenizer: model is not BPE");
    // The algorithm is fixed; refuse a tokenizer.json whose pipeline differs from MiniMax M3.x's.
    const auto* normalizer = root.find("normalizer");
    if (!normalizer || !normalizer->find("type") || normalizer->find("type")->as_string() != "NFC")
        throw std::runtime_error("minimax tokenizer: normalizer is not NFC");
    const auto* pre = root.find("pre_tokenizer");
    const auto* pres = pre ? pre->find("pretokenizers") : nullptr;
    static constexpr std::string_view kPattern =
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
        "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
        "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    bool pre_ok = pres && pres->is_array() && pres->size() == 2U;
    if (pre_ok) {
        const auto& split = pres->as_array()[0];
        const auto& bytelevel = pres->as_array()[1];
        const auto* pattern = split.find("pattern");
        const auto* regex = pattern ? pattern->find("Regex") : nullptr;
        pre_ok = split.find("type") && split.find("type")->as_string() == "Split" && regex && regex->is_string() &&
            regex->as_string() == kPattern && split.find("behavior") && split.find("behavior")->as_string() == "Removed" &&
            split.find("invert") && split.find("invert")->is_bool() && split.find("invert")->as_bool() &&
            bytelevel.find("type") && bytelevel.find("type")->as_string() == "ByteLevel" &&
            bytelevel.find("use_regex") && !bytelevel.find("use_regex")->as_bool() &&
            bytelevel.find("add_prefix_space") && !bytelevel.find("add_prefix_space")->as_bool();
    }
    if (!pre_ok) throw std::runtime_error("minimax tokenizer: pre_tokenizer is not the MiniMax Split+ByteLevel sequence");
    if (const auto* dec = root.find("decoder"); !dec || !dec->find("type") || dec->find("type")->as_string() != "ByteLevel")
        throw std::runtime_error("minimax tokenizer: decoder is not ByteLevel");
    if (const auto* pp = root.find("post_processor"); pp && !pp->is_null())
        throw std::runtime_error("minimax tokenizer: post_processor is not supported");
    if (const auto* ig = model->find("ignore_merges"); ig && ig->is_bool() && ig->as_bool())
        throw std::runtime_error("minimax tokenizer: ignore_merges is not supported");

    std::shared_ptr<MinimaxTokenizer> out(new MinimaxTokenizer());
    auto& impl = *out->impl_;
    const auto* vocab = model->find("vocab");
    if (!vocab || !vocab->is_object()) throw std::runtime_error("minimax tokenizer: model.vocab missing");
    std::size_t vocab_size = 0;
    for (const auto& [text, id] : vocab->as_object()) vocab_size = std::max<std::size_t>(vocab_size, static_cast<std::size_t>(id.as_int()) + 1U);
    impl.vocab.resize(vocab_size);
    impl.vocab_bytes.resize(vocab_size);
    impl.vocab_ids.reserve(vocab->size() * 2U);
    std::unordered_map<std::uint32_t, unsigned char> cp_byte;
    const auto byte_cp = byte_to_code_point();
    for (std::uint32_t b = 0; b < 256U; ++b) cp_byte.emplace(byte_cp[b], static_cast<unsigned char>(b));
    for (const auto& [text, value] : vocab->as_object()) {
        const auto id = static_cast<std::uint32_t>(value.as_int());
        impl.vocab[id] = text;
        impl.vocab_ids.emplace(text, id);
        // ByteLevel decoder: every character maps to a byte, else the token's own UTF-8 is used.
        std::string bytes;
        bool direct = true;
        for (const auto cp : to_code_points(text)) {
            const auto hit = cp_byte.find(cp);
            if (hit == cp_byte.end()) { direct = false; break; }
            bytes.push_back(static_cast<char>(hit->second));
        }
        impl.vocab_bytes[id] = direct ? bytes : text;
    }
    for (std::uint32_t b = 0; b < 256U; ++b) {
        std::string s;
        append_utf8(s, byte_cp[b]);
        const auto hit = impl.vocab_ids.find(s);
        if (hit == impl.vocab_ids.end()) throw std::runtime_error("minimax tokenizer: byte token missing from the vocabulary");
        impl.byte_token[b] = hit->second;
    }
    const auto* merges = model->find("merges");
    if (!merges || !merges->is_array()) throw std::runtime_error("minimax tokenizer: model.merges missing");
    impl.merges.reserve(merges->size() * 2U);
    std::uint32_t rank = 0;
    for (const auto& m : merges->as_array()) {
        std::string a, b;
        if (m.is_string()) {
            const auto& s = m.as_string();
            const auto space = s.find(' ');
            if (space == std::string::npos) throw std::runtime_error("minimax tokenizer: malformed merge");
            a = s.substr(0, space);
            b = s.substr(space + 1U);
        } else if (m.is_array() && m.size() == 2U) {
            a = m.as_array()[0].as_string();
            b = m.as_array()[1].as_string();
        } else throw std::runtime_error("minimax tokenizer: malformed merge");
        const auto ia = impl.vocab_ids.find(a), ib = impl.vocab_ids.find(b), im = impl.vocab_ids.find(a + b);
        if (ia == impl.vocab_ids.end() || ib == impl.vocab_ids.end() || im == impl.vocab_ids.end())
            throw std::runtime_error("minimax tokenizer: merge of tokens outside the vocabulary");
        impl.merges.insert_or_assign(Impl::pair_key(ia->second, ib->second), std::pair{rank, im->second});  // HF: a later duplicate wins
        ++rank;
    }
    impl.size = vocab_size;
    if (const auto* added = root.find("added_tokens"); added && added->is_array()) {
        for (const auto& t : added->as_array()) {
            Impl::Added a{t.find("content")->as_string(), static_cast<std::uint32_t>(t.find("id")->as_int()),
                t.find("special") && t.find("special")->as_bool()};
            const auto flag = [&](const char* key) { const auto* v = t.find(key); return v && v->is_bool() && v->as_bool(); };
            if (flag("lstrip") || flag("rstrip") || flag("single_word") || flag("normalized"))
                throw std::runtime_error("minimax tokenizer: added token options lstrip/rstrip/single_word/normalized are not supported");
            if (a.content.empty()) continue;
            impl.added_first[static_cast<unsigned char>(a.content[0])] = true;
            impl.added_by_id.emplace(a.id, impl.added.size());
            impl.size = std::max<std::size_t>(impl.size, static_cast<std::size_t>(a.id) + 1U);
            if (a.content == "]~b]") impl.bos = a.id;
            if (a.content == "[e~[") impl.eos = a.id;
            if (a.content == "]!p~[") impl.pad = a.id;
            impl.added.push_back(std::move(a));
        }
    }
    return out;
}

std::vector<std::uint32_t> MinimaxTokenizer::encode(std::string_view text, bool parse_special) const {
    return encode_marks(text, parse_special, nullptr);
}

std::vector<std::uint32_t> MinimaxTokenizer::encode_marked(std::string_view text, bool parse_special,
                                                           std::vector<strix::serve::EncodeMark>& marks) const {
    return encode_marks(text, parse_special, &marks);
}

std::size_t MinimaxTokenizer::mark_lookahead() const noexcept {
    std::size_t longest = 0;
    for (const auto& a : impl_->added) longest = std::max(longest, a.content.size());
    return longest;
}

std::vector<std::uint32_t> MinimaxTokenizer::encode_marks(std::string_view text, bool parse_special,
                                                          std::vector<strix::serve::EncodeMark>* marks) const {
    const auto& impl = *impl_;
    std::vector<std::uint32_t> out;
    out.reserve(text.size() / 3U + 4U);
    std::size_t start = 0, i = 0;
    while (i < text.size()) {
        if (!impl.added_first[static_cast<unsigned char>(text[i])]) { ++i; continue; }
        // leftmost-longest among all added tokens
        const Impl::Added* best = nullptr;
        for (const auto& a : impl.added)
            if ((!best || a.content.size() > best->content.size()) && text.compare(i, a.content.size(), a.content) == 0) best = &a;
        if (!best) { ++i; continue; }
        if (best->special && !parse_special) { i += best->content.size(); continue; }  // stays text
        impl.encode_text(text.substr(start, i - start), out);
        out.push_back(best->id);
        i += best->content.size();
        start = i;
        if (marks) marks->push_back({i, out.size()});
    }
    impl.encode_text(text.substr(start), out);
    return out;
}

std::string MinimaxTokenizer::decode(std::span<const std::uint32_t> ids, bool skip_special) const {
    const auto& impl = *impl_;
    std::string out, chunk;
    for (const auto id : ids) {
        if (const auto a = impl.added_by_id.find(id); a != impl.added_by_id.end()) {
            const auto& added = impl.added[a->second];
            if (skip_special && added.special) continue;
            append_lossy(out, chunk);
            chunk.clear();
            out += added.content;
        } else if (id < impl.vocab_bytes.size() && !impl.vocab[id].empty()) {
            chunk += impl.vocab_bytes[id];
        }
    }
    append_lossy(out, chunk);
    return out;
}

std::uint32_t MinimaxTokenizer::token_id(std::string_view token_text) const {
    for (const auto& a : impl_->added) if (a.content == token_text) return a.id;
    const auto hit = impl_->vocab_ids.find(std::string(token_text));
    return hit == impl_->vocab_ids.end() ? UINT32_MAX : hit->second;
}
std::string MinimaxTokenizer::token_text(std::uint32_t id) const {
    if (const auto a = impl_->added_by_id.find(id); a != impl_->added_by_id.end()) return impl_->added[a->second].content;
    return id < impl_->vocab.size() ? impl_->vocab[id] : std::string();
}
std::string MinimaxTokenizer::token_bytes(std::uint32_t id) const {
    if (const auto a = impl_->added_by_id.find(id); a != impl_->added_by_id.end()) return impl_->added[a->second].content;
    return id < impl_->vocab_bytes.size() ? impl_->vocab_bytes[id] : std::string();
}
bool MinimaxTokenizer::is_added(std::uint32_t id) const noexcept { return impl_->added_by_id.count(id) != 0U; }
bool MinimaxTokenizer::is_special(std::uint32_t id) const noexcept {
    const auto a = impl_->added_by_id.find(id);
    return a != impl_->added_by_id.end() && impl_->added[a->second].special;
}
std::uint32_t MinimaxTokenizer::bos_id() const noexcept { return impl_->bos; }
std::uint32_t MinimaxTokenizer::eos_id() const noexcept { return impl_->eos; }
std::uint32_t MinimaxTokenizer::pad_id() const noexcept { return impl_->pad; }
std::size_t MinimaxTokenizer::vocab_size() const noexcept { return impl_->size; }

std::string MinimaxTokenizer::nfc(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const auto cp : nfc_code_points(to_code_points(text))) append_utf8(out, cp);
    return out;
}

std::vector<std::string> MinimaxTokenizer::pre_tokenize(std::string_view normalized_text) {
    const auto points = to_code_points(normalized_text);
    std::vector<std::string> out;
    for (const auto& [a, b] : pre_split(points)) {
        std::string piece;
        for (std::size_t k = a; k < b; ++k) append_utf8(piece, points[k]);
        out.push_back(std::move(piece));
    }
    return out;
}

const char* MinimaxTokenizer::class_unicode_version() noexcept { return kMmClassUnicode; }
const char* MinimaxTokenizer::nfc_unicode_version() noexcept { return kMmNfcUnicode; }

}  // namespace strix::minimax::serve
