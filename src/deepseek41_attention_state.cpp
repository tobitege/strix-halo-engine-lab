#include "strix/deepseek41_attention_state.hpp"

#include "strix/deepseek41_attention_host.hpp"
#include "strix/ds41_attention.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>

namespace strix::deepseek41 {
namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL, kFnvPrime = 1099511628211ULL;
constexpr char kMagic[8] = {'D', 'S', '4', '1', 'A', 'T', 'S', 'T'};
constexpr std::uint32_t kVersion = 1;

std::uint64_t fnv_bytes(const void* data, std::size_t n, std::uint64_t h = kFnvOffset) {
    const auto* b = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * kFnvPrime;
    return h;
}

// FNV of the live window of a ring: positions max(0, T - window) .. T - 1 in order (as the runner's ring_window_hash).
std::uint64_t ring_window_hash(const std::uint8_t* ring, std::uint32_t positions) {
    namespace attn = ds41::attn;
    std::uint64_t h = kFnvOffset;
    const std::uint32_t first = positions > attn::kWindow ? positions - attn::kWindow : 0U;
    for (std::uint32_t p = first; p < positions; ++p)
        h = fnv_bytes(ring + std::size_t{p % AttentionHost::ring_capacity} * attn::kWindowEntryBytes, attn::kWindowEntryBytes, h);
    return h;
}

// In-memory sink and source with the file's checksum (serialize / deserialize).
struct Bytes {
    std::vector<std::uint8_t> out;
    std::span<const std::uint8_t> in;
    std::size_t at = 0;
    std::uint64_t sum = kFnvOffset;
    void write(const void* p, std::size_t n) {
        const auto* b = static_cast<const std::uint8_t*>(p);
        out.insert(out.end(), b, b + n);
        sum = fnv_bytes(p, n, sum);
    }
    void read(void* p, std::size_t n) {
        if (n > in.size() - at) throw std::runtime_error("deepseek41 attention state: truncated state bytes");
        std::memcpy(p, in.data() + at, n);
        at += n;
        sum = fnv_bytes(p, n, sum);
    }
    template <typename T>
    void put(const T& v) { write(&v, sizeof(T)); }
    template <typename T>
    T get() {
        T v{};
        read(&v, sizeof(T));
        return v;
    }
    template <typename T>
    void put_vec(const std::vector<T>& v) {
        put<std::uint64_t>(v.size());
        write(v.data(), v.size() * sizeof(T));
    }
    template <typename T>
    std::vector<T> get_vec(std::uint64_t limit) {
        const auto n = get<std::uint64_t>();
        if (n > limit || n * sizeof(T) > in.size() - at) throw std::runtime_error("deepseek41 attention state: corrupt size in state bytes");
        std::vector<T> v(n);
        read(v.data(), n * sizeof(T));
        return v;
    }
};

struct File {
    std::FILE* f = nullptr;
    std::string path;
    std::uint64_t sum = kFnvOffset;  // FNV of every byte written / read so far (the file's checksum)
    File(const std::filesystem::path& p, const char* mode) : f(std::fopen(p.string().c_str(), mode)), path(p.string()) {
        if (f == nullptr) throw std::runtime_error("deepseek41 attention state: cannot open " + path);
    }
    ~File() {
        if (f != nullptr) std::fclose(f);
    }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    void write(const void* p, std::size_t n) {
        if (n != 0U && std::fwrite(p, 1, n, f) != n) throw std::runtime_error("deepseek41 attention state: write failed: " + path);
        sum = fnv_bytes(p, n, sum);
    }
    void read(void* p, std::size_t n) {
        if (n != 0U && std::fread(p, 1, n, f) != n) throw std::runtime_error("deepseek41 attention state: truncated file: " + path);
        sum = fnv_bytes(p, n, sum);
    }
    template <typename T>
    void put(const T& v) { write(&v, sizeof(T)); }
    template <typename T>
    T get() {
        T v{};
        read(&v, sizeof(T));
        return v;
    }
    template <typename T>
    void put_vec(const std::vector<T>& v) {
        put<std::uint64_t>(v.size());
        write(v.data(), v.size() * sizeof(T));
    }
    template <typename T>
    std::vector<T> get_vec(std::uint64_t limit) {
        const auto n = get<std::uint64_t>();
        if (n > limit) throw std::runtime_error("deepseek41 attention state: corrupt size in " + path);
        std::vector<T> v(n);
        read(v.data(), n * sizeof(T));
        return v;
    }
};

// One layer's pieces as bytes (in an AttentionLayerState's vectors or in place in a serialized buffer).
struct LayerView {
    const std::uint8_t* ring = nullptr;
    std::size_t ring_n = 0;
    const std::uint8_t* kv = nullptr;
    std::size_t kv_n = 0;
    const std::uint8_t* idx = nullptr;
    std::size_t idx_n = 0;
    const std::uint8_t* carry = nullptr;
    std::size_t carry_n = 0;  // bytes
    const std::uint8_t* topk = nullptr;
    std::size_t topk_count = 0;
    bool owns_kv = false, owns_index = false, carry_valid = false, topk_known = true;
    std::uint32_t ratio = 0;
};
using Hashes = std::vector<std::pair<std::string, std::uint64_t>>;

LayerView view_of(const AttentionLayerState& L) {
    LayerView v;
    v.ring = L.ring.data();
    v.ring_n = L.ring.size();
    v.kv = L.kv.data();
    v.kv_n = L.kv.size();
    v.idx = L.idx_keys.data();
    v.idx_n = L.idx_keys.size();
    v.carry = reinterpret_cast<const std::uint8_t*>(L.carry.data());
    v.carry_n = L.carry.size() * 4U;
    v.topk = reinterpret_cast<const std::uint8_t*>(L.topk.data());
    v.topk_count = L.topk.size();
    v.owns_kv = L.owns_kv;
    v.owns_index = L.owns_index;
    v.carry_valid = L.carry_valid;
    v.topk_known = L.topk_known;
    v.ratio = L.ratio;
    return v;
}

void layer_hashes(const LayerView& L, std::size_t l, std::uint32_t position, Hashes& out) {
    if (L.ring_n == 0U) return;  // a layer range of a pipeline stage
    const std::string at = "@" + std::to_string(l);
    out.emplace_back("ring" + at, ring_window_hash(L.ring, position));
    if (L.owns_kv) {
        out.emplace_back("kv" + at, fnv_bytes(L.kv, L.kv_n));
        out.emplace_back("idx_keys" + at, fnv_bytes(L.idx, L.idx_n));
        if (L.ratio == 2U) out.emplace_back("carry" + at, L.carry_valid ? fnv_bytes(L.carry, L.carry_n) : 0U);
    }
    if (L.owns_index) out.emplace_back("topk" + at, L.topk_known ? fnv_bytes(L.topk, L.topk_count * 4U, L.topk_count) : 0U);
}

std::uint64_t digest_of(const Hashes& hashes, const void* tokens, std::size_t token_count, std::uint32_t position, std::uint32_t next_token,
                        bool without_topk) {
    std::uint64_t h = kFnvOffset;
    for (const auto& [name, v] : hashes) {
        if (without_topk && name.rfind("topk", 0) == 0) continue;
        h = fnv_bytes(name.data(), name.size(), h);
        h = fnv_bytes(&v, sizeof(v), h);
    }
    h = fnv_bytes(tokens, token_count * 4U, h);
    h = fnv_bytes(&position, 4U, h);
    return fnv_bytes(&next_token, 4U, h);
}

}  // namespace

std::vector<std::pair<std::string, std::uint64_t>> AttentionState::hashes() const {
    Hashes out;
    for (std::size_t l = 0; l < layers.size(); ++l) layer_hashes(view_of(layers[l]), l, position, out);
    return out;
}

std::uint64_t AttentionState::digest(bool without_topk) const {
    return digest_of(hashes(), tokens.data(), tokens.size(), position, next_token, without_topk);
}

std::uint64_t AttentionState::bytes() const {
    std::uint64_t n = 0;
    for (const auto& L : layers) n += L.ring.size() + L.kv.size() + L.idx_keys.size() + L.carry.size() * 4U + L.topk.size() * 4U;
    return n;
}

std::vector<std::uint8_t> canonical_attention_state_bytes(const AttentionState& state) {
    namespace attn = ds41::attn;
    auto out = state;
    std::array<bool, AttentionHost::ring_capacity> live{};
    const auto first = state.position > attn::kWindow ? state.position - attn::kWindow : 0U;
    for (auto p = first; p < state.position; ++p) live[p % AttentionHost::ring_capacity] = true;
    for (auto& layer : out.layers) {
        if (!layer.ring.empty()) {
            if (layer.ring.size() != live.size() * attn::kWindowEntryBytes)
                throw std::invalid_argument("canonical attention state: invalid ring size");
            for (std::size_t i = 0; i < live.size(); ++i)
                if (!live[i]) std::fill_n(layer.ring.begin() + i * attn::kWindowEntryBytes, attn::kWindowEntryBytes, std::uint8_t{0});
        }
        layer.topk.clear();
        layer.topk_known = false;
        if (!layer.carry_valid) std::fill(layer.carry.begin(), layer.carry.end(), 0.0F);
    }
    return serialize_attention_state(out);
}

namespace {
template <typename Sink>
void write_state(Sink& f, const AttentionState& s) {
    f.write(kMagic, sizeof(kMagic));
    f.put(kVersion);
    f.put(s.position);
    f.put(s.next_token);
    f.template put<std::uint32_t>(static_cast<std::uint32_t>(s.layers.size()));
    for (const auto& L : s.layers) {
        const std::uint8_t flags = (L.owns_kv ? 1U : 0U) | (L.owns_index ? 2U : 0U) | (L.carry_valid ? 4U : 0U) | (L.topk_known ? 8U : 0U);
        f.put(flags);
        f.put(L.ratio);
        f.put_vec(L.ring);
        f.put_vec(L.kv);
        f.put_vec(L.idx_keys);
        f.put_vec(L.carry);
        f.put_vec(L.topk);
    }
    f.put_vec(s.tokens);
    f.put(s.digest());
    const std::uint64_t checksum = f.sum;
    f.put(checksum);
}

template <typename Source>
AttentionState read_state(Source& f, const std::string& what) {
    char magic[sizeof(kMagic)];
    f.read(magic, sizeof(magic));
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) throw std::runtime_error("deepseek41 attention state: not a state: " + what);
    if (f.template get<std::uint32_t>() != kVersion) throw std::runtime_error("deepseek41 attention state: unknown version: " + what);
    AttentionState s;
    s.position = f.template get<std::uint32_t>();
    s.next_token = f.template get<std::uint32_t>();
    const auto layers = f.template get<std::uint32_t>();
    if (layers > 64U) throw std::runtime_error("deepseek41 attention state: corrupt layer count");
    constexpr std::uint64_t big = std::uint64_t{1} << 36U;
    s.layers.resize(layers);
    for (auto& L : s.layers) {
        const auto flags = f.template get<std::uint8_t>();
        L.owns_kv = (flags & 1U) != 0U;
        L.owns_index = (flags & 2U) != 0U;
        L.carry_valid = (flags & 4U) != 0U;
        L.topk_known = (flags & 8U) != 0U;
        L.ratio = f.template get<std::uint32_t>();
        L.ring = f.template get_vec<std::uint8_t>(big);
        if (!L.ring.empty() && L.ring.size() != std::size_t{AttentionHost::ring_capacity} * ds41::attn::kWindowEntryBytes)
            throw std::runtime_error("deepseek41 attention state: corrupt ring size");
        L.kv = f.template get_vec<std::uint8_t>(big);
        L.idx_keys = f.template get_vec<std::uint8_t>(big);
        L.carry = f.template get_vec<float>(big);
        L.topk = f.template get_vec<std::uint32_t>(big);
    }
    s.tokens = f.template get_vec<std::uint32_t>(big);
    const auto digest = f.template get<std::uint64_t>();
    const std::uint64_t checksum = f.sum;
    if (f.template get<std::uint64_t>() != checksum || digest != s.digest())
        throw std::runtime_error("deepseek41 attention state: checksum or digest mismatch (corrupt): " + what);
    if (s.tokens.size() != s.position) throw std::runtime_error("deepseek41 attention state: token history length != position");
    return s;
}
}  // namespace

std::vector<std::uint8_t> serialize_attention_state(const AttentionState& s) {
    Bytes b;
    write_state(b, s);
    return std::move(b.out);
}

AttentionState deserialize_attention_state(std::span<const std::uint8_t> bytes) {
    Bytes b;
    b.in = bytes;
    auto s = read_state(b, "state bytes");
    if (b.at != bytes.size()) throw std::runtime_error("deepseek41 attention state: trailing bytes after the state");
    return s;
}

namespace {
// little-endian host (as write_state's raw fields)
template <typename T>
void put_at(std::uint8_t* out, std::uint64_t& at, const T& v) {
    std::memcpy(out + at, &v, sizeof(T));
    at += sizeof(T);
}
LayerView view_at(const AttentionFrameLayer& F, const std::uint8_t* b) {
    LayerView v;
    v.ring = b + F.ring_at;
    v.ring_n = F.ring_bytes;
    v.kv = b + F.kv_at;
    v.kv_n = F.kv_bytes;
    v.idx = b + F.idx_at;
    v.idx_n = F.idx_bytes;
    v.carry = b + F.carry_at;
    v.carry_n = F.carry_floats * 4U;
    v.topk = b + F.topk_at;
    v.topk_count = F.topk_count;
    v.owns_kv = F.owns_kv;
    v.owns_index = F.owns_index;
    v.carry_valid = F.carry_valid;
    v.topk_known = F.topk_known;
    v.ratio = F.ratio;
    return v;
}
}  // namespace

void plan_attention_frame(AttentionFrame& f) {
    if (f.layers.size() > 64U) throw std::invalid_argument("deepseek41 attention frame: more than 64 layers");
    std::uint64_t at = sizeof(kMagic) + 4U * 4U;  // magic, version, position, next token, layer count
    for (auto& L : f.layers) {
        at += 1U + 4U;  // flags, ratio
        L.ring_at = at + 8U;
        at = L.ring_at + L.ring_bytes;
        L.kv_at = at + 8U;
        at = L.kv_at + L.kv_bytes;
        L.idx_at = at + 8U;
        at = L.idx_at + L.idx_bytes;
        L.carry_at = at + 8U;
        at = L.carry_at + L.carry_floats * 4U;
        L.topk_at = at + 8U;
        at = L.topk_at + L.topk_count * 4U;
    }
    f.pieces_end = at;
    f.tokens_at = at + 8U;
    f.digest_at = f.tokens_at + f.token_count * 4U;
    f.checksum_at = f.digest_at + 8U;
    f.total = f.checksum_at + 8U;
}

void write_attention_frame(const AttentionFrame& f, std::span<const std::uint32_t> tokens, std::span<std::uint8_t> out) {
    if (out.size() < f.total || f.total == 0U) throw std::invalid_argument("deepseek41 attention frame: buffer smaller than the planned frame");
    if (tokens.size() != f.token_count) throw std::invalid_argument("deepseek41 attention frame: token count differs from the plan");
    std::uint8_t* b = out.data();
    std::uint64_t at = 0;
    std::memcpy(b, kMagic, sizeof(kMagic));
    at += sizeof(kMagic);
    put_at(b, at, kVersion);
    put_at(b, at, f.position);
    put_at(b, at, f.next_token);
    put_at(b, at, static_cast<std::uint32_t>(f.layers.size()));
    for (const auto& L : f.layers) {
        const std::uint8_t flags = (L.owns_kv ? 1U : 0U) | (L.owns_index ? 2U : 0U) | (L.carry_valid ? 4U : 0U) | (L.topk_known ? 8U : 0U);
        put_at(b, at, flags);
        put_at(b, at, L.ratio);
        const std::uint64_t sizes[5] = {L.ring_bytes, L.kv_bytes, L.idx_bytes, L.carry_floats, L.topk_count};
        const std::uint64_t starts[5] = {L.ring_at, L.kv_at, L.idx_at, L.carry_at, L.topk_at};
        for (int k = 0; k < 5; ++k) {
            if (at + 8U != starts[k]) throw std::logic_error("deepseek41 attention frame: not planned (plan_attention_frame)");
            put_at(b, at, sizes[k]);
            at = starts[k] + (k == 3 ? sizes[k] * 4U : k == 4 ? sizes[k] * 4U : sizes[k]);
        }
    }
    if (at != f.pieces_end) throw std::logic_error("deepseek41 attention frame: not planned (plan_attention_frame)");
    put_at(b, at, f.token_count);
    if (!tokens.empty()) std::memcpy(b + at, tokens.data(), tokens.size() * 4U);
}

AttentionFrameSeal seal_attention_frame(const AttentionFrame& f, std::span<std::uint8_t> out, unsigned threads) {
    if (out.size() < f.total || f.total == 0U) throw std::invalid_argument("deepseek41 attention frame: buffer smaller than the planned frame");
    std::uint8_t* b = out.data();
    if (threads == 0U) threads = 2U;  // the per-piece FNVs cover about the bytes the checksum covers: one thread each
    // the checksum is one sequential FNV over every byte before it: the long pole, on its own thread from the start
    std::uint64_t sum = kFnvOffset;
    std::thread checksum([&] { sum = fnv_bytes(b, f.digest_at); });
    std::vector<Hashes> per(f.layers.size());
    std::atomic<std::size_t> next{0};
    // largest layers first (the KV sources), the ring-only layers fill in
    std::vector<std::size_t> order(f.layers.size());
    for (std::size_t l = 0; l < order.size(); ++l) order[l] = l;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) {
        return f.layers[x].kv_bytes + f.layers[x].idx_bytes > f.layers[y].kv_bytes + f.layers[y].idx_bytes;
    });
    const auto work = [&] {
        for (std::size_t i = next++; i < order.size(); i = next++) layer_hashes(view_at(f.layers[order[i]], b), order[i], f.position, per[order[i]]);
    };
    std::vector<std::thread> pool;
    try {
        for (unsigned t = 2; t < threads; ++t) pool.emplace_back(work);  // threads - 2 helpers + this one + the checksum
        work();
    } catch (...) {
        for (auto& t : pool) t.join();
        checksum.join();
        throw;
    }
    for (auto& t : pool) t.join();
    Hashes all;
    for (auto& h : per) all.insert(all.end(), std::make_move_iterator(h.begin()), std::make_move_iterator(h.end()));
    AttentionFrameSeal s;
    s.digest = digest_of(all, b + f.tokens_at, f.token_count, f.position, f.next_token, false);
    std::memcpy(b + f.digest_at, &s.digest, 8U);
    checksum.join();
    s.checksum = fnv_bytes(&s.digest, 8U, sum);
    std::memcpy(b + f.checksum_at, &s.checksum, 8U);
    return s;
}

bool attention_layer_present(const AttentionLayerState& layer) noexcept { return !layer.ring.empty(); }

AttentionState slice_attention_state(const AttentionState& s, std::uint32_t first, std::uint32_t end) {
    if (first >= end || end > s.layers.size()) throw std::invalid_argument("deepseek41 attention state: slice outside the layers");
    AttentionState out = s;
    for (std::size_t l = 0; l < out.layers.size(); ++l)
        if (l < first || l >= end) out.layers[l] = AttentionLayerState{};
    return out;
}

AttentionState merge_attention_states(const AttentionState& front, const AttentionState& tail) {
    if (front.position != tail.position || front.tokens != tail.tokens)
        throw std::invalid_argument("deepseek41 attention state: merge of states at different positions or tokens");
    AttentionState out;
    out.position = tail.position;
    out.next_token = tail.next_token;
    out.tokens = tail.tokens;
    out.layers.resize(std::max(front.layers.size(), tail.layers.size()));
    for (std::size_t l = 0; l < out.layers.size(); ++l) {
        const bool f = l < front.layers.size() && attention_layer_present(front.layers[l]);
        const bool t = l < tail.layers.size() && attention_layer_present(tail.layers[l]);
        if (f && t) throw std::invalid_argument("deepseek41 attention state: layer " + std::to_string(l) + " is in both states");
        if (f) out.layers[l] = front.layers[l];
        if (t) out.layers[l] = tail.layers[l];
    }
    return out;
}

void save_attention_state(const std::filesystem::path& path, const AttentionState& s) {
    const auto tmp = std::filesystem::path(path.string() + ".tmp");
    {
        File f(tmp, "wb");
        write_state(f, s);
        if (std::fflush(f.f) != 0) throw std::runtime_error("deepseek41 attention state: flush failed: " + tmp.string());
    }
    std::filesystem::rename(tmp, path);
}

AttentionState load_attention_state(const std::filesystem::path& path) {
    File f(path, "rb");
    return read_state(f, path.string());
}

std::uint64_t attention_state_device_bytes(std::uint32_t max_positions, std::uint32_t layers) {
    namespace attn = ds41::attn;
    const auto plan = attn::released_plan();
    std::uint64_t n = 0;
    for (std::uint32_t l = 0; l < layers && l < plan.size(); ++l) {
        const auto& r = plan[l];
        n += std::uint64_t{AttentionHost::ring_capacity} * attn::kWindowEntryBytes;
        if (r.owns_kv) {
            const std::uint64_t entries = max_positions / r.ratio + 1U;
            n += entries * (attn::kKvEntryBytes + attn::kIdxEntryBytes) + 2U * (2U * attn::kHeadDim * 4U + 4U);
        }
        if (r.owns_index) n += std::uint64_t{8} * (attn::kTopK + 1U) * 4U;
    }
    return n;
}

}  // namespace strix::deepseek41
