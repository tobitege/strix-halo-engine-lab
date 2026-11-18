// GLM-5.3-Flash TP2 rank-slice plan (see glm_rank_plan.hpp).
#include "glm/load/glm_rank_plan.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <variant>

namespace strix::glm::load {

namespace {

// ---------------------------------------------------------------- minimal JSON (MiniMax mm_pack.cpp's parser)
struct Json;
using JsonObj = std::map<std::string, Json>;
using JsonArr = std::vector<Json>;
struct Json {
    std::variant<std::nullptr_t, double, std::string, JsonArr, JsonObj> v;
    const JsonObj& obj() const { return std::get<JsonObj>(v); }
    const JsonArr& arr() const { return std::get<JsonArr>(v); }
    const std::string& str() const { return std::get<std::string>(v); }
    double num() const { return std::get<double>(v); }
};
struct JsonParser {
    const char* p;
    const char* end;
    void ws() {
        while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    }
    Json parse() {
        ws();
        if (p >= end) throw std::runtime_error("json: eof");
        if (*p == '{') {
            ++p;
            JsonObj o;
            ws();
            if (*p == '}') {
                ++p;
                return {o};
            }
            for (;;) {
                ws();
                std::string k = string();
                ws();
                if (*p++ != ':') throw std::runtime_error("json: ':'");
                o.emplace(std::move(k), parse());
                ws();
                if (*p == ',') {
                    ++p;
                    continue;
                }
                if (*p++ != '}') throw std::runtime_error("json: '}'");
                return {o};
            }
        }
        if (*p == '[') {
            ++p;
            JsonArr a;
            ws();
            if (*p == ']') {
                ++p;
                return {a};
            }
            for (;;) {
                a.push_back(parse());
                ws();
                if (*p == ',') {
                    ++p;
                    continue;
                }
                if (*p++ != ']') throw std::runtime_error("json: ']'");
                return {a};
            }
        }
        if (*p == '"') return {string()};
        if (std::strncmp(p, "null", 4) == 0) {
            p += 4;
            return {nullptr};
        }
        if (std::strncmp(p, "true", 4) == 0) {
            p += 4;
            return {1.0};
        }
        if (std::strncmp(p, "false", 5) == 0) {
            p += 5;
            return {0.0};
        }
        char* e = nullptr;
        const double d = std::strtod(p, &e);
        if (e == p) throw std::runtime_error("json: value");
        p = e;
        return {d};
    }
    std::string string() {
        if (*p++ != '"') throw std::runtime_error("json: string");
        std::string s;
        while (*p != '"') {
            if (*p == '\\') {
                ++p;
                s += *p == 'n' ? '\n' : *p;
                ++p;
                continue;
            }
            s += *p++;
        }
        ++p;
        return s;
    }
};
Json parse_json(const std::string& s) {
    JsonParser jp{s.data(), s.data() + s.size()};
    return jp.parse();
}
std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
std::uint64_t dtype_bytes(const std::string& t) {
    if (t == "F32" || t == "I32" || t == "U32") return 4;
    if (t == "F16" || t == "BF16" || t == "I16" || t == "U16") return 2;
    if (t == "U8" || t == "I8" || t == "F8_E4M3" || t == "BOOL") return 1;
    if (t == "F64" || t == "I64") return 8;
    throw std::runtime_error("unknown dtype " + t);
}
std::uint64_t u64(double d) {
    if (d < 0 || d != static_cast<double>(static_cast<std::uint64_t>(d))) throw std::runtime_error("json: not a byte count");
    return static_cast<std::uint64_t>(d);
}

}  // namespace

Index::Index(const std::string& dir) {
    std::vector<std::string> names;
    std::ifstream probe(dir + "/model.safetensors.index.json");
    if (probe) {
        const Json j = parse_json(read_text(dir + "/model.safetensors.index.json"));
        std::set<std::string> s;
        for (const auto& [k, v] : j.obj().at("weight_map").obj()) s.insert(v.str());
        names.assign(s.begin(), s.end());
    } else {
        DIR* d = ::opendir(dir.c_str());
        if (d == nullptr) throw std::runtime_error("cannot list " + dir);
        while (const dirent* e = ::readdir(d)) {
            const std::string n = e->d_name;
            if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) names.push_back(n);
        }
        ::closedir(d);
        std::sort(names.begin(), names.end());
    }
    weight::TensorIndex common; common.container=weight::Container::safetensors; common.schema="safetensors@1";
    for (const std::string& n : names) {
        const std::string path = dir + "/" + n;
        std::shared_ptr<const weight::Source> source;
        auto status=weight::open_source(path,source,n);
        weight::TensorIndex index;
        if(status)status=weight::index_safetensors(source,index);
        if(!status)throw std::runtime_error("GLM index: "+status.detail);
        const auto file=static_cast<std::uint32_t>(files_.size());files_.push_back(path);
        for(const auto& raw:index.tensors) {
            TensorInfo t;t.dtype=raw.wire_dtype;t.shape=raw.extents;
            t.file=file;t.offset=raw.payload.range.offset;t.bytes=raw.payload.range.bytes;
            if(!tensors_.emplace(raw.name,std::move(t)).second)throw std::runtime_error("duplicate tensor "+raw.name);
            common.tensors.push_back(raw);
        }
    }
    auto status=weight::bind_glm(common,weights_);
    if(!status)throw std::runtime_error("GLM admission: "+status.detail);
}

const TensorInfo* Index::find(const std::string& name) const {
    const auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

Split::Split(const std::string& json_path) {
    const Json j = parse_json(read_text(json_path));
    for (const auto& [name, e] : j.obj().at("tensors").obj()) {
        SplitEntry s;
        s.exl3 = e.obj().at("kind").str() == "exl3";
        const std::string& m = e.obj().at("mode").str();
        s.mode = m == "out" ? Mode::out : m == "in" ? Mode::in : m == "rows" ? Mode::rows : Mode::replicate;
        if (m != "out" && m != "in" && m != "rows" && m != "replicate") throw std::runtime_error("split mode " + m);
        const auto& rr = e.obj().at("ranges").arr();
        for (unsigned r = 0; r < 2; ++r)
            for (const Json& p : rr.at(r).arr())
                s.ranges[r].emplace_back(static_cast<std::int64_t>(p.arr().at(0).num()), static_cast<std::int64_t>(p.arr().at(1).num()));
        entries_.emplace(name, std::move(s));
    }
    const auto& rb = j.obj().at("rank_bytes").arr();
    rank_bytes_[0] = rb.at(0).num();
    rank_bytes_[1] = rb.at(1).num();
}

std::vector<Slice> rank_plan(const Index& index, const Split& split, unsigned rank, std::size_t* missing) {
    if (rank > 1) throw std::invalid_argument("rank must be 0 or 1");
    std::vector<Slice> out;
    std::size_t miss = 0;
    auto add = [](Slice& s, const TensorInfo& t, std::uint64_t at, std::uint64_t len) {
        if (at + len > t.bytes) throw std::runtime_error("piece outside tensor " + s.name);
        if (len == 0) return;
        s.pieces.push_back({t.file, t.offset + at, len});
        s.bytes += len;
    };
    for (const auto& [name, e] : split.entries()) {
        Slice s;
        s.name = name;
        s.exl3 = e.exl3;
        const auto& ranges = e.ranges[rank];
        if (!e.exl3) {
            const TensorInfo* t = index.find(name);
            if (t == nullptr) {
                ++miss;
                continue;
            }
            if (e.mode == Mode::replicate) {
                add(s, *t, 0, t->bytes);
            } else {
                if (e.mode != Mode::rows || t->shape.empty()) throw std::runtime_error("plain tensor split must be rows: " + name);
                const std::uint64_t row = t->bytes / t->shape[0];
                for (const auto& [b, en] : ranges) {
                    if (b < 0 || en <= b || static_cast<std::uint64_t>(en) > t->shape[0]) throw std::runtime_error("rows range " + name);
                    add(s, *t, static_cast<std::uint64_t>(b) * row, static_cast<std::uint64_t>(en - b) * row);
                }
            }
            out.push_back(std::move(s));
            continue;
        }
        const TensorInfo* tr = index.find(name + ".trellis");
        const TensorInfo* suh = index.find(name + ".suh");
        const TensorInfo* svh = index.find(name + ".svh");
        const TensorInfo* mul = index.find(name + ".mul1");
        if (tr == nullptr || suh == nullptr || svh == nullptr) {
            ++miss;
            continue;
        }
        if (tr->shape.size() != 3 || tr->dtype != "I16" || suh->dtype != "F16" || svh->dtype != "F16") throw std::runtime_error("EXL3 layout " + name);
        const std::uint64_t K16 = tr->shape[0], N16 = tr->shape[1], w = tr->shape[2] * 2;  // bytes per 16x16 tile
        if (suh->shape.at(0) != K16 * 16 || svh->shape.at(0) != N16 * 16) throw std::runtime_error("EXL3 suh/svh " + name);
        s.bits = tr->shape[2] / 16;
        std::uint64_t n = 0;
        for (const auto& [b, en] : ranges) {
            if (e.mode == Mode::replicate) break;
            if (b < 0 || en <= b || b % 128 != 0 || en % 128 != 0) throw std::runtime_error("EXL3 range alignment " + name);
            n += static_cast<std::uint64_t>(en - b);
        }
        switch (e.mode) {
            case Mode::replicate:
                add(s, *tr, 0, tr->bytes);
                add(s, *suh, 0, suh->bytes);
                add(s, *svh, 0, svh->bytes);
                s.tr_bytes = tr->bytes, s.suh_bytes = suh->bytes, s.svh_bytes = svh->bytes;
                s.rows = N16 * 16, s.cols = K16 * 16;
                break;
            case Mode::out:
                for (std::uint64_t i = 0; i < K16; ++i)
                    for (const auto& [b, en] : ranges) {
                        if (static_cast<std::uint64_t>(en) > N16 * 16) throw std::runtime_error("out range " + name);
                        add(s, *tr, (i * N16 + static_cast<std::uint64_t>(b) / 16) * w, static_cast<std::uint64_t>(en - b) / 16 * w);
                    }
                s.tr_bytes = K16 * (n / 16) * w;
                add(s, *suh, 0, suh->bytes);
                for (const auto& [b, en] : ranges) add(s, *svh, static_cast<std::uint64_t>(b) * 2, static_cast<std::uint64_t>(en - b) * 2);
                s.suh_bytes = suh->bytes, s.svh_bytes = n * 2;
                s.rows = n, s.cols = K16 * 16;
                break;
            case Mode::in:
                for (const auto& [b, en] : ranges) {
                    if (static_cast<std::uint64_t>(en) > K16 * 16) throw std::runtime_error("in range " + name);
                    add(s, *tr, static_cast<std::uint64_t>(b) / 16 * N16 * w, static_cast<std::uint64_t>(en - b) / 16 * N16 * w);
                }
                s.tr_bytes = n / 16 * N16 * w;
                for (const auto& [b, en] : ranges) add(s, *suh, static_cast<std::uint64_t>(b) * 2, static_cast<std::uint64_t>(en - b) * 2);
                add(s, *svh, 0, svh->bytes);
                s.suh_bytes = n * 2, s.svh_bytes = svh->bytes;
                s.rows = N16 * 16, s.cols = n;
                break;
            case Mode::rows:
                throw std::runtime_error("EXL3 group in rows mode " + name);
        }
        if (mul != nullptr) {
            add(s, *mul, 0, mul->bytes);
            s.mul1_bytes = mul->bytes;
        }
        out.push_back(std::move(s));
    }
    if (missing != nullptr) *missing = miss;
    return out;
}

std::vector<Piece> coalesce(const std::vector<Piece>& pieces) {
    std::vector<Piece> out;
    for (const Piece& p : pieces) {
        if (!out.empty() && out.back().file == p.file && out.back().src + out.back().len == p.src) out.back().len += p.len;
        else out.push_back(p);
    }
    return out;
}

}  // namespace strix::glm::load
