// MiniMax EXL3 pack reader for the RTX PRO prefill. See mm_rtx_pack.hpp.
#include "mm_rtx_pack.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(_WIN32)
#define MM_FSEEK _fseeki64
#else
#define MM_FSEEK fseeko
#endif

namespace strix::minimax::rtxpf::host {

namespace {

// Minimal JSON reader for safetensors headers and the index: objects, arrays, strings (no unicode escapes needed),
// numbers, literals.
struct Json {
    enum Kind { kNull, kNum, kStr, kArr, kObj } kind = kNull;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::map<std::string, Json> obj;
};
struct Parser {
    const char* p;
    const char* e;
    void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; }
    std::string string() {
        if (*p != '"') throw std::runtime_error("json: string expected");
        ++p;
        std::string s;
        while (p < e && *p != '"') {
            if (*p == '\\') {
                ++p;
                const char c = *p++;
                s.push_back(c == 'n' ? '\n' : c == 't' ? '\t' : c);
            } else {
                s.push_back(*p++);
            }
        }
        ++p;
        return s;
    }
    Json value() {
        ws();
        Json j;
        if (*p == '{') {
            j.kind = Json::kObj;
            ++p;
            ws();
            if (*p == '}') { ++p; return j; }
            for (;;) {
                ws();
                std::string k = string();
                ws();
                ++p;  // ':'
                j.obj[k] = value();
                ws();
                if (*p == ',') { ++p; continue; }
                ++p;  // '}'
                return j;
            }
        }
        if (*p == '[') {
            j.kind = Json::kArr;
            ++p;
            ws();
            if (*p == ']') { ++p; return j; }
            for (;;) {
                j.arr.push_back(value());
                ws();
                if (*p == ',') { ++p; continue; }
                ++p;
                return j;
            }
        }
        if (*p == '"') { j.kind = Json::kStr; j.str = string(); return j; }
        if (*p == 'n' || *p == 't' || *p == 'f') {
            while (p < e && *p >= 'a' && *p <= 'z') ++p;
            return j;
        }
        j.kind = Json::kNum;
        char* end = nullptr;
        j.num = std::strtod(p, &end);
        p = end;
        return j;
    }
};
Json parse(const std::string& s) {
    Parser ps{s.data(), s.data() + s.size()};
    return ps.value();
}
std::string slurp(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string s;
    char buf[65536];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    std::fclose(f);
    return s;
}

}  // namespace

float half_to_float(std::uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp(static_cast<float>(m), -24) : std::ldexp(static_cast<float>(m | 1024), e - 25);
    return s ? -v : v;
}

Pack::Pack(std::string dir, unsigned prefix_layers) : dir_(std::move(dir)) {
    const Json idx = parse(slurp(dir_ + "/model.safetensors.index.json"));
    std::map<std::string, std::uint32_t> file_id;
    std::vector<std::string> names;
    for (const auto& [name, f] : idx.obj.at("weight_map").obj) {
        if (prefix_layers != 0) {
            const std::string lp = "language_model.model.layers.";
            bool keep = name == "language_model.model.embed_tokens.weight";
            if (name.compare(0, lp.size(), lp) == 0) {
                const auto dot = name.find('.', lp.size());
                keep = dot != std::string::npos && std::stoul(name.substr(lp.size(), dot - lp.size())) < prefix_layers;
            }
            if (!keep) continue;
        }
        if (file_id.count(f.str) == 0) {
            file_id[f.str] = static_cast<std::uint32_t>(names.size());
            names.push_back(f.str);
        }
    }
    try {
    for (std::uint32_t i = 0; i < names.size(); ++i) {
        const std::string path = dir_ + "/" + names[i];
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open " + path);
        files_.push_back(f);
        std::uint64_t hlen = 0;
        if (std::fread(&hlen, 8, 1, f) != 1) throw std::runtime_error("bad header " + path);
        std::string h(hlen, '\0');
        if (std::fread(h.data(), 1, hlen, f) != hlen) throw std::runtime_error("bad header " + path);
        const Json hj = parse(h);
        for (const auto& [name, t] : hj.obj) {
            if (name == "__metadata__") continue;
            TensorInfo ti;
            ti.dtype = t.obj.at("dtype").str;
            for (const auto& d : t.obj.at("shape").arr) ti.shape.push_back(static_cast<std::uint64_t>(d.num));
            const auto& off = t.obj.at("data_offsets").arr;
            ti.file = i;
            ti.offset = 8 + hlen + static_cast<std::uint64_t>(off[0].num);
            ti.bytes = static_cast<std::uint64_t>(off[1].num) - static_cast<std::uint64_t>(off[0].num);
            tensors_[name] = ti;
        }
    }
    } catch (...) {
        for (auto* f : files_) std::fclose(f);
        throw;
    }
}

Pack::~Pack() {
    for (auto* f : files_) std::fclose(f);
}

const TensorInfo& Pack::info(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("tensor not found: " + name);
    return it->second;
}

void Pack::read(const TensorInfo& t, std::uint64_t offset, std::uint64_t n, void* dst) {
    if (offset + n > t.bytes) throw std::runtime_error("read past tensor end");
    std::FILE* f = files_[t.file];
    if (MM_FSEEK(f, static_cast<long long>(t.offset + offset), SEEK_SET) != 0) throw std::runtime_error("seek");
    if (std::fread(dst, 1, n, f) != n) throw std::runtime_error("short read");
    bytes_read_ += n;
}

std::vector<float> Pack::floats(const std::string& name) {
    const TensorInfo& t = info(name);
    std::vector<std::uint8_t> raw(t.bytes);
    read(t, 0, t.bytes, raw.data());
    std::vector<float> out;
    if (t.dtype == "F32") {
        out.resize(t.bytes / 4);
        std::memcpy(out.data(), raw.data(), t.bytes);
    } else if (t.dtype == "BF16" || t.dtype == "F16") {
        out.resize(t.bytes / 2);
        for (std::size_t i = 0; i < out.size(); ++i) {
            std::uint16_t h;
            std::memcpy(&h, raw.data() + 2 * i, 2);
            if (t.dtype == "BF16") {
                const std::uint32_t b = static_cast<std::uint32_t>(h) << 16;
                std::memcpy(&out[i], &b, 4);
            } else {
                out[i] = half_to_float(h);
            }
        }
    } else {
        throw std::runtime_error("floats: unsupported dtype " + t.dtype + " for " + name);
    }
    return out;
}

Exl3Host Pack::slice_out(const std::string& prefix, std::uint64_t o0, std::uint64_t n) {
    const TensorInfo &tr = info(prefix + ".trellis"), &suh = info(prefix + ".suh"), &svh = info(prefix + ".svh");
    if (o0 % 128 != 0 || n % 128 != 0 || o0 + n > svh.shape[0]) throw std::invalid_argument("bad output slice " + prefix);
    const std::uint64_t in_tiles = tr.shape[0], out_tiles = tr.shape[1], w = tr.shape[2] * 2;  // bytes per tile
    if (whole_name_ != prefix) {
        whole_.resize(tr.bytes);
        read(tr, 0, tr.bytes, whole_.data());
        whole_name_ = prefix;
    }
    const std::vector<std::uint8_t>& whole = whole_;
    Exl3Host e;
    e.trellis.resize(in_tiles * (n / 16) * w);
    for (std::uint64_t i = 0; i < in_tiles; ++i)
        std::memcpy(e.trellis.data() + i * (n / 16) * w, whole.data() + (i * out_tiles + o0 / 16) * w, (n / 16) * w);
    e.suh.resize(suh.shape[0]);
    read(suh, 0, suh.bytes, e.suh.data());
    e.svh.resize(n);
    read(svh, o0 * 2, n * 2, e.svh.data());
    e.input = static_cast<std::uint32_t>(suh.shape[0]);
    e.output = static_cast<std::uint32_t>(n);
    e.bits = static_cast<std::uint32_t>(tr.shape[2] / 16);
    return e;
}

Exl3Host Pack::slice_in(const std::string& prefix, std::uint64_t i0, std::uint64_t n) {
    const TensorInfo &tr = info(prefix + ".trellis"), &suh = info(prefix + ".suh"), &svh = info(prefix + ".svh");
    if (i0 % 128 != 0 || n % 128 != 0 || i0 + n > suh.shape[0]) throw std::invalid_argument("bad input slice " + prefix);
    const std::uint64_t row_bytes = tr.shape[1] * tr.shape[2] * 2;
    Exl3Host e;
    e.trellis.resize((n / 16) * row_bytes);
    read(tr, (i0 / 16) * row_bytes, e.trellis.size(), e.trellis.data());
    e.suh.resize(n);
    read(suh, i0 * 2, n * 2, e.suh.data());
    e.svh.resize(svh.shape[0]);
    read(svh, 0, svh.bytes, e.svh.data());
    e.input = static_cast<std::uint32_t>(n);
    e.output = static_cast<std::uint32_t>(svh.shape[0]);
    e.bits = static_cast<std::uint32_t>(tr.shape[2] / 16);
    return e;
}

}  // namespace strix::minimax::rtxpf::host
