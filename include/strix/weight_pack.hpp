// Shared GGUF weight-pack admission (host, model- and backend-agnostic).
// A family declares its supported packs; each pack is a list of rules "tensor role -> admitted GGUF types" plus a
// versioned identity, the arithmetic class its route runs under and a one-line route description. admit() picks the
// first declared pack whose rules admit every tensor of the file (all split parts), so a model path selects its
// pack, and returns the identity, the tensor-inventory fingerprint (FNV-1a over (name, type, dims) in name order)
// and the per-type census printed by the effective config. Engines and tools call this one function: the server and
// the CLI cannot disagree about which pack a file is.
#pragma once

#include "strix/serve/json.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::weight_pack {

// One rule: tensors whose name contains `match` (empty = any tensor) must have a GGUF type in `types`.
struct Rule {
    std::string_view role, match;
    std::span<const std::uint32_t> types;
};
struct Decl {
    std::string_view id;          // versioned identity, e.g. "qwen38-gsq-rco-iq3s-v1"
    std::string_view arithmetic;  // "exact" or "fast:<policy name>"
    std::string_view route;       // which kernels / storage the backend uses for the pack
    std::span<const Rule> rules;  // first matching rule decides a tensor
};
struct Tensor {
    std::string_view name;
    std::uint32_t ggml_type = 0;
    std::span<const std::uint64_t> dims;
};
// Paired tensor names differ only in these role markers and share one storage decoder.
struct TypePair {
    std::string_view first, second;
};
struct Admission {
    const Decl* pack = nullptr;  // null: no declared pack admits the file
    std::string reason;          // why each declared pack before the admitted one (or every pack) declined
    std::uint64_t fingerprint = 0;
    std::map<std::uint32_t, std::uint64_t> tensors_by_type;
};
// This is an inventory identifier, not a weight-content digest or numerical identity. Integer bytes are
// host-native, so it is not a portable persistent-cache key. `pack` borrows the declaration's lifetime;
// declarations (and their strings/rules) must outlive receipts. Unique tensor names are required.

// The parts of a split GGUF ("...-00001-of-00002.gguf"): every part in order; otherwise the file itself.
inline std::vector<std::string> gguf_parts(const std::string& path) {
    const auto at = path.rfind("-00001-of-");
    if (at == std::string::npos || path.size() < at + 20) return {path};
    const auto total = static_cast<unsigned>(std::stoul(path.substr(at + 10, 5)));
    std::vector<std::string> parts;
    for (unsigned i = 1; i <= total; ++i) {
        std::string index = std::to_string(i);
        index.insert(0, 5U - std::min<std::size_t>(5U, index.size()), '0');
        parts.push_back(path.substr(0, at + 1) + index + path.substr(at + 6));
    }
    return parts;
}

inline std::uint64_t fnv(std::uint64_t h, const void* p, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) { h ^= static_cast<const unsigned char*>(p)[i]; h *= 0x100000001b3ULL; }
    return h;
}
inline const Rule* rule_for(const Decl& d, std::string_view name) {
    for (const auto& r : d.rules)
        if (r.match.empty() || name.find(r.match) != std::string_view::npos) return &r;
    return nullptr;
}
inline bool type_in(std::span<const std::uint32_t> types, std::uint32_t t) {
    for (const auto x : types) if (x == t) return true;
    return false;
}
// Empty when `d` admits every tensor; else the first refusal "pack: tensor (type T) not admitted [role]".
inline std::string refusal(const Decl& d, std::span<const Tensor> tensors) {
    for (const auto& t : tensors) {
        const auto* r = rule_for(d, t.name);
        if (!r) return std::string(d.id) + ": " + std::string(t.name) + " has no rule";
        if (!type_in(r->types, t.ggml_type))
            return std::string(d.id) + ": " + std::string(t.name) + " (type " + std::to_string(t.ggml_type) +
                   ") not admitted for " + std::string(r->role);
    }
    return tensors.empty() ? std::string(d.id) + ": no tensors" : std::string();
}
inline Admission admit(std::span<const Decl> family, std::span<const Tensor> tensors,
                       std::span<const TypePair> same_types = {}) {
    Admission a;
    a.fingerprint = 0xcbf29ce484222325ULL;
    std::vector<const Tensor*> by_name;  // name order: the fingerprint does not depend on how parts were merged
    for (const auto& t : tensors) by_name.push_back(&t);
    std::sort(by_name.begin(), by_name.end(), [](const Tensor* x, const Tensor* y) { return x->name < y->name; });
    for (const auto* p : by_name) {
        const auto& t = *p;
        a.fingerprint = fnv(fnv(a.fingerprint, t.name.data(), t.name.size() + 0U), "\0", 1);
        a.fingerprint = fnv(a.fingerprint, &t.ggml_type, sizeof t.ggml_type);
        for (const auto d : t.dims) a.fingerprint = fnv(a.fingerprint, &d, sizeof d);
        ++a.tensors_by_type[t.ggml_type];
    }
    for (const auto& pair : same_types) {
        for (const auto& t : tensors) {
            auto from = pair.first, to = pair.second;
            auto at = t.name.find(from);
            if (at == std::string_view::npos) {
                std::swap(from, to);
                at = t.name.find(from);
            }
            if (at == std::string_view::npos) continue;
            std::string peer(t.name);
            peer.replace(at, from.size(), to);
            const auto it = std::find_if(tensors.begin(), tensors.end(),
                                         [&](const Tensor& other) { return other.name == peer; });
            if (it == tensors.end() || it->ggml_type != t.ggml_type) {
                a.reason = std::string(t.name) + " requires same-type partner " + peer;
                return a;
            }
        }
    }
    for (const auto& d : family) {
        auto why = refusal(d, tensors);
        if (why.empty()) { a.pack = &d; return a; }  // reason keeps why the packs before it declined
        a.reason += (a.reason.empty() ? "" : "; ") + why;
    }
    return a;
}

inline std::string hex64(std::uint64_t x) {
    char b[17];
    for (int i = 15; i >= 0; --i, x >>= 4U) b[i] = "0123456789abcdef"[x & 15U];
    b[16] = 0;
    return b;
}

inline serve::Json to_json(const Admission& a) {
    auto out = serve::Json::object();
    out.set("id", a.pack ? serve::Json(a.pack->id) : serve::Json());
    if (a.pack) {
        out.set("arithmetic", a.pack->arithmetic);
        out.set("route", a.pack->route);
    }
    out.set("fingerprint", hex64(a.fingerprint));
    auto census = serve::Json::object();
    for (const auto& [type, n] : a.tensors_by_type) census.set(std::to_string(type), n);
    out.set("tensors_by_type", std::move(census));
    out.set("declined", a.reason);
    return out;
}

}  // namespace strix::weight_pack
