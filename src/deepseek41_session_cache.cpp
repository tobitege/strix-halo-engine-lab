#include "strix/ds41_native_fp8.hpp"
#include "strix/ds41_trellis.hpp"
#include "strix/deepseek41_session_cache.hpp"
#include "strix/session/checkpoint_io.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/deepseek41_attention_host.hpp"
#include "strix/ds41_attention.hpp"
#include <iterator>
#include <bit>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>
extern char **environ;
#endif

namespace strix::deepseek41 {
namespace {
namespace io = ::strix::session::checkpoint_io;
[[noreturn]] void fail(const std::string &s) { throw std::runtime_error("ds41 session cache: " + s); }
// Persistent lock inodes must never be unlinked during cleanup. Separate rank files
// allow peers on one host to rendezvous while holding their transaction locks.
// Family adapter preserves the ordered two-inode transaction, without host IO.
class DiskLock {
    std::unique_ptr<io::DirectoryLock> lock_;
public:
    void acquire(const std::filesystem::path& dir, unsigned rank) {
        if (dir.empty()) return;
        try { lock_ = std::make_unique<io::DirectoryLock>(dir, io::LockMode::blocking,
            ".session-rank" + std::to_string(rank) + ".lock"); }
        catch (const std::exception& e) { fail(std::string("disk transaction lock: ") + e.what()); }
    }
    DiskLock() = default;
    DiskLock(const DiskLock&) = delete;
    void transaction(const std::filesystem::path& dir, unsigned rank, const SessionExchange& exchange) {
        // Acquire rank 0 before rank 1 across all pairs, including shared directories.
        for (unsigned turn = 0; turn < 2; ++turn) {
            bool ok = true;
            try { if (rank == turn) acquire(dir, rank); } catch (...) { ok = false; }
            PrefixCache::Sessions::agree(ok, exchange, "disk transaction lock");
        }
    }
};
std::string failure_code(const std::string& message) {
    if (message.find("session_not_found") != std::string::npos) return "session_not_found";
    if (message.find("budget") != std::string::npos) return "session_budget";
    if (message.find("session_incompatible") != std::string::npos) return "session_incompatible";
    return "session_corrupt";
}
std::string sha(std::span<const std::uint8_t> b) { return model_content_identity_hex(sha256_bytes(std::as_bytes(b))); }
std::string sha(const std::string &s) { return sha(std::span(reinterpret_cast<const std::uint8_t *>(s.data()), s.size())); }
struct Wire {
    SessionBytes b;
    std::size_t at = 0;
    void u(std::uint64_t n) {
        for (unsigned i = 0; i != 8; ++i)
            b.push_back(static_cast<std::uint8_t>(n >> (i * 8U)));
    }
    std::uint64_t u() {
        if (b.size() - at < 8)
            fail("truncated integer");
        std::uint64_t n = 0;
        for (unsigned i = 0; i != 8; ++i)
            n |= std::uint64_t{b[at++]} << (i * 8U);
        return n;
    }
    void bytes(std::span<const std::uint8_t> x) {
        u(x.size());
        b.insert(b.end(), x.begin(), x.end());
    }
    SessionBytes bytes(std::uint64_t limit) {
        const auto n = u();
        if (n > limit || n > b.size() - at)
            fail("section bounds");
        SessionBytes x(b.begin() + static_cast<std::ptrdiff_t>(at), b.begin() + static_cast<std::ptrdiff_t>(at + n));
        at += static_cast<std::size_t>(n);
        return x;
    }
    void s(const std::string &x) { bytes(std::span(reinterpret_cast<const std::uint8_t *>(x.data()), x.size())); }
    std::string s() {
        const auto x = bytes(1U << 20U);
        return {x.begin(), x.end()};
    }
    void end() const {
        if (at != b.size())
            fail("trailing data");
    }
};
SessionBytes frontier(const SessionCheckpoint &c) {
    Wire w;
    w.u(c.frontier.consumed);
    w.u(c.frontier.emitted);
    w.u(c.frontier.accepted);
    w.u(c.frontier.drafter);
    w.u(c.frontier.pending);
    w.u(c.frontier.pending_valid);
    w.s(c.frontier.sampling);
    w.s(c.frontier.lineage);
    // Token history is replicated and bound to the frontier, including Engram input ids.
    w.u(c.attention.tokens.size());
    for (auto t : c.attention.tokens)
        w.u(t);
    return w.b;
}
void validate(const SessionCheckpoint &c) {
    validate_session_id(c.id);
    validate_session_id(c.generation);
    if (c.world != 2 || c.rank >= 2)
        fail("only world=2 is supported");
    if (c.attention.position == 0 || c.attention.tokens.size() != c.attention.position || c.frontier.consumed != c.attention.position ||
        c.frontier.pending != c.attention.next_token || !c.frontier.pending_valid || c.frontier.accepted != c.frontier.consumed ||
        c.frontier.emitted > c.frontier.consumed + 1 || (c.draft ? c.frontier.drafter != c.draft->context.next : c.frontier.drafter != 0) || c.frontier.sampling != "greedy" ||
        c.frontier.lineage != (c.draft ? "ds41-dspark-r1-committed-turn-v2" : "ds41-target-committed-turn-v1"))
        fail("unsupported/inconsistent committed greedy frontier");
    if (c.draft) {
        (void)serialize_session_draft(*c.draft);
        if (c.draft->context.next != c.frontier.consumed) fail("DSpark staged frontier mismatch");
    }
    if (c.attention.layers.empty() || c.attention.layers.size() > 64)
        fail("layer bounds");
    if (c.attention.layers.size() > 40)
        fail("unsupported layer count");
    const auto plan = ds41::attn::released_plan();
    for (std::size_t i = 0; i < c.attention.layers.size(); ++i) {
        const auto &l = c.attention.layers[i];
        const auto &r = plan[i];
        if (l.ring.size() != AttentionHost::ring_capacity * ds41::attn::kWindowEntryBytes || l.owns_kv != r.owns_kv ||
            l.owns_index != r.owns_index || l.ratio != r.ratio)
            fail("layer layout mismatch");
        if (l.owns_kv) {
            const auto n = std::uint64_t{c.attention.position} / l.ratio;
            if (l.kv.size() != n * ds41::attn::kKvEntryBytes || l.idx_keys.size() != n * ds41::attn::kIdxEntryBytes)
                fail("compressed state size");
            if (l.ratio == 2 && l.carry.size() != 2U * ds41::attn::kHeadDim)
                fail("carry size");
        } else if (!l.kv.empty() || !l.idx_keys.empty() || !l.carry.empty() || l.carry_valid)
            fail("unexpected source state");
        if (l.topk.size() > ds41::attn::kTopK)
            fail("selection bounds");
    }
}
SessionBytes read(const std::filesystem::path &p, std::uint64_t max) {
    SessionBytes b;
    try {
        io::read(p, 0, max, [&](std::uint64_t n) {
            b.resize(static_cast<std::size_t>(n));
            return std::as_writable_bytes(std::span(b));
        }, io::DirectIo::buffered);
    } catch (const io::Error& e) {
        throw io::Error(e.operation, (std::string("ds41 session cache: ") + e.what()).c_str());
    }
    return b;
}
void durable(const std::filesystem::path &p, std::span<const std::uint8_t> b) {
    try { io::atomic_write(p, std::as_bytes(b)); }
    catch (const io::Error& e) {
        throw io::Error(e.operation, (std::string("ds41 session cache: ") + e.what()).c_str());
    }
}
// Only exact family-owned filenames qualify for accounting/cleanup. A symlink
// is never a family payload. Historical and shared-core temporary names coexist.
std::string disk_owner(const std::string& name, unsigned rank) {
    const auto dot = name.find('.');
    if (dot == std::string::npos) return {};
    const auto id = name.substr(0, dot);
    const auto tail = ".rank" + std::to_string(rank);
    auto rest = name.substr(dot);
    try {
        validate_session_id(id);
        if (!rest.starts_with(tail + ".group")) {
            const auto end = rest.find(tail + ".state");
            if (end == std::string::npos || end < 2) return {};
            validate_session_id(rest.substr(1, end - 1));
            rest.erase(0, end);
        }
        const auto stem = rest.starts_with(tail + ".group") ? tail + ".group" : tail + ".state";
        rest.erase(0, stem.size());
        if (rest.empty()) return id;
        if (!rest.starts_with(".tmp.") && !rest.starts_with(".tmp-")) return {};
        validate_session_id(rest.substr(5));
        return id;
    } catch (const std::exception&) { return {}; }
}
std::string identity(const std::filesystem::path &p) {
    std::string s = std::filesystem::canonical(p).string() + ":" + std::to_string(std::filesystem::file_size(p)) + ":" +
                    std::to_string(std::filesystem::last_write_time(p).time_since_epoch().count());
#if !defined(_WIN32)
    struct stat st{};
    if (::stat(p.c_str(), &st))
        fail("stat asset");
    s += ":" + std::to_string(st.st_dev) + ":" + std::to_string(st.st_ino) + ":" + std::to_string(st.st_ctim.tv_sec) + ":" +
         std::to_string(st.st_ctim.tv_nsec);
#else
    HANDLE h = CreateFileW(p.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                           0, nullptr);
    BY_HANDLE_FILE_INFORMATION st{};
    if (h == INVALID_HANDLE_VALUE)
        fail("open asset identity");
    const bool ok = GetFileInformationByHandle(h, &st) != 0;
    CloseHandle(h);
    if (!ok)
        fail("asset identity");
    s += ":" + std::to_string(st.dwVolumeSerialNumber) + ":" + std::to_string(st.nFileIndexHigh) + ":" + std::to_string(st.nFileIndexLow);
#endif
    return s;
}
struct Descriptor {
    std::string id, generation, compat, shared, payload;
    std::uint64_t rank = 0, bytes = 0;
};
SessionBytes describe(const SessionCheckpoint &c, const SessionBytes &b) {
    Wire w;
    w.s(c.id);
    w.s(c.generation);
    w.u(c.rank);
    w.s(sha(c.compat_record));
    w.s(sha(sha(frontier(c)) + session_state_sha256(c)));
    w.u(b.size());
    w.s(sha(b));
    w.u(c.frontier.consumed);
    w.u(c.frontier.emitted);
    w.u(c.frontier.pending);
    w.u(c.frontier.accepted);
    w.u(c.frontier.drafter);
    w.s(c.frontier.sampling);
    w.s(c.frontier.lineage);
    const auto f = frontier(c), a = serialize_attention_state(c.attention);
    for (unsigned i = 0; i < (c.draft ? 3U : 2U); ++i) {
        const auto section = i == 0 ? f : i == 1 ? a : serialize_session_draft(*c.draft);
        w.s(i == 0 ? "frontier-history" : i == 1 ? "attention-v1" : "dspark-v2");
        w.s(i == 1 ? "sharded" : "replicated");
        w.u(0);
        w.u(section.size());
        w.s(sha(section));
    }
    return w.b;
}
Descriptor descriptor(SessionBytes b) {
    Wire w{std::move(b)};
    Descriptor d;
    d.id = w.s();
    d.generation = w.s();
    d.rank = w.u();
    d.compat = w.s();
    d.shared = w.s();
    d.bytes = w.u();
    d.payload = w.s();
    for (unsigned i = 0; i < 5; ++i)
        (void)w.u();
    if (w.s() != "greedy") fail("manifest sampling");
    const auto lineage = w.s();
    const bool draft = lineage == "ds41-dspark-r1-committed-turn-v2";
    if (!draft && lineage != "ds41-target-committed-turn-v1") fail("manifest frontier");
    for (unsigned i = 0; i < (draft ? 3U : 2U); ++i) {
        if (w.s() != (i == 0 ? "frontier-history" : i == 1 ? "attention-v1" : "dspark-v2") || w.s() != (i == 1 ? "sharded" : "replicated") || w.u() != 0)
            fail("manifest section table");
        if (w.u() > d.bytes || w.s().size() != 64)
            fail("manifest section bounds/hash");
    }
    w.end();
    validate_session_id(d.id);
    validate_session_id(d.generation);
    if (d.rank > 1 || d.payload.size() != 64 || d.shared.size() != 64)
        fail("bad rank descriptor");
    return d;
}
SessionBytes group(const SessionBytes &mine, const SessionBytes &other, const SessionCompat &compat, std::uint32_t rank) {
    const auto a = descriptor(mine), b = descriptor(other);
    auto expected = compat;
    expected.fields["rank"] = std::to_string(1U - rank);
    if (a.rank != rank || b.rank != 1U - rank || a.compat != compat.key() || b.compat != expected.key())
        fail("session_incompatible: rank geometry/compat mismatch");
    if (a.id != b.id || a.generation != b.generation || a.shared != b.shared)
        fail("rank generation/frontier mismatch");
    Wire w;
    w.s("strix/ds41-session-group-v1");
    w.u(2);
    w.bytes(rank == 0 ? mine : other);
    w.bytes(rank == 0 ? other : mine);
    w.s(sha(w.b));
    return w.b;
}
std::array<SessionBytes, 2> parse_group(SessionBytes b) {
    Wire w{std::move(b)};
    if (w.s() != "strix/ds41-session-group-v1" || w.u() != 2)
        fail("session_incompatible: group schema/world");
    std::array<SessionBytes, 2> ds{w.bytes(1U << 20U), w.bytes(1U << 20U)};
    const auto hash = sha(std::span(w.b).first(w.at));
    if (w.s() != hash)
        fail("manifest root checksum");
    w.end();
    return ds;
}
} // namespace

SessionBytes session_peer_exchange(std::span<const std::uint8_t> out, const SessionHostExchange &exchange) {
    std::uint64_t n = out.size(), m = 0;
    exchange(std::as_bytes(std::span(&n, 1)), std::as_writable_bytes(std::span(&m, 1)));
    if (n > (16U << 20U) || m > (16U << 20U))
        fail("control exchange exceeds 16 MiB");
    SessionBytes in(static_cast<std::size_t>(m));
    constexpr std::size_t chunk = 64U << 10U;
    for (std::size_t at = 0; at < std::max(n, m); at += chunk) {
        const auto size = static_cast<std::size_t>(std::min<std::uint64_t>(chunk, std::max(n, m) - at));
        SessionBytes a(size), b(size);
        const auto take = at < n ? std::min(size, static_cast<std::size_t>(n) - at) : 0;
        if (take)
            std::copy_n(out.begin() + static_cast<std::ptrdiff_t>(at), take, a.begin());
        exchange(std::as_bytes(std::span(a)), std::as_writable_bytes(std::span(b)));
        const auto keep = at < m ? std::min(size, static_cast<std::size_t>(m) - at) : 0;
        if (keep)
            std::copy_n(b.begin(), keep, in.begin() + static_cast<std::ptrdiff_t>(at));
    }
    return in;
}
void validate_session_id(const std::string &id) {
    if (id.empty() || id.size() > 128 || !std::all_of(id.begin(), id.end(), [](unsigned char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
        }))
        fail("id must be 1..128 ASCII letters, digits, hyphen or underscore");
}
std::string session_generation() {
    static std::atomic<std::uint64_t> count{0};
    return std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + "-" + std::to_string(++count);
}
std::string SessionCompat::record() const {
    Wire w;
    w.s("strix/ds41-session-compat-v1");
    w.u(fields.size());
    for (const auto &[k, v] : fields) {
        w.s(k);
        w.s(v);
    }
    return {w.b.begin(), w.b.end()};
}
std::string SessionCompat::key() const { return sha(record()); }
std::string session_file_sha256(const std::filesystem::path &p) {
    static std::mutex mutex;
    static std::map<std::string, std::string> cache;
    const std::lock_guard lock(mutex);
    const auto before = identity(p);
    if (auto i = cache.find(before); i != cache.end())
        return i->second;
    const auto h = model_content_identity_hex(sha256_file_content(p));
    if (identity(p) != before)
        fail("asset changed while hashing");
    cache.emplace(before, h);
    return h;
}
SessionCompat session_compat(std::map<std::string, std::string> effective, const std::map<std::string, std::filesystem::path> &assets,
                             const std::filesystem::path &executable, const LoaderSnapshot* loaded) {
    SessionCompat c;
    c.fields = std::move(effective);
    c.fields["schema"] = c.fields["dspark"] == "on" ? "strix/ds41-session-v2" : "strix/ds41-session-v1";
    c.fields["attention-schema"] = "strix/ds41-attention-state-v1";
    c.fields["executable-sha256"] = session_file_sha256(executable);
    std::map<std::filesystem::path, std::string> loaded_digests;
    if (loaded) {
        if (loaded->empty()) fail("missing model loader snapshot");
        verify_loader_snapshot(*loaded);
        for (const auto& file : *loaded) {
            const auto digest = model_content_identity_hex(sha256_file_content(file.path, file.identity));
            const auto [it, inserted] = loaded_digests.emplace(file.path, digest);
            if (!inserted && it->second != digest)
                throw std::runtime_error("model files changed during startup");
        }
    }
    // Preserve reference paths and include every loader input, even if a future
    // loader adds an input that the prewarming inventory does not yet know.
    auto inventory_digests = loaded_digests;
    const auto asset_hash = [&](const std::filesystem::path& path) {
        const auto reference = std::filesystem::absolute(path);
        const auto it = loaded_digests.find(reference);
        auto digest = it == loaded_digests.end() ? session_file_sha256(path) : it->second;
        inventory_digests[reference] = digest;
        return digest;
    };
    auto all_assets = assets;
#if defined(_WIN32)
    auto asset_env = _environ;
#else
    auto asset_env = ::environ;
#endif
    for (auto e = asset_env; e && *e; ++e) {
        const std::string text = *e;
        const auto equal = text.find('=');
        if (equal == std::string::npos)
            continue;
        const auto name = text.substr(0, equal), value = text.substr(equal + 1);
        if (!name.starts_with("STRIX_"))
            continue;
        // Only immutable numerical inputs are assets. Operational paths (hash,
        // prefix, output and log caches) must never be traversed.
        const bool numerical_asset = name == "STRIX_DS41_NATIVE_FP8" || name == "STRIX_DS41_OVERLAY" ||
            (name == "STRIX_DS41_PREPARED_LOAD" && value != "0") || name == "STRIX_DS41_EP_MAP" ||
            (name == "STRIX_DS41_TRELLIS_MIX" && value.find('=') == std::string::npos) ||
            (name == "STRIX_DS41_FP16_CACHE" && std::getenv("STRIX_DS41_FP16_CACHE_READONLY") &&
             std::string_view(std::getenv("STRIX_DS41_FP16_CACHE_READONLY")) == "1");
        if (numerical_asset && !value.empty())
            all_assets["env:" + name] = value;
    }
    const auto trellis = strix::ds41::trellis_env_spec();
    if (!trellis.empty()) {
        std::size_t i = 0;
        for (const auto& [cls, dir] : strix::ds41::parse_trellis_pack_spec(trellis))
            all_assets["trellis-mix:" + std::to_string(i++) + ":" + cls] = dir;
    }
    for (const auto &[label, p] : all_assets) {
        if (p.empty()) {
            c.fields["asset:" + label] = "off";
            continue;
        }
        if (std::filesystem::is_directory(p)) {
            std::map<std::string, std::filesystem::path> files;
            const bool checkpoint = label == "env:STRIX_DS41_TRELLIS" || label == "env:STRIX_DS41_NATIVE_FP8" ||
                label.starts_with("trellis-mix:");
            if (checkpoint) {
                // Strict validation is session-only; ordinary lazy loading is unchanged.
                strix::ds41::SafetensorsCheckpoint validated(p, true);
                files = strix::ds41::resolve_safetensors_checkpoint(p).files;
            } else {
                for (auto it = std::filesystem::recursive_directory_iterator(p);
                     it != std::filesystem::recursive_directory_iterator(); ++it) {
                    // Other asset formats retain their metadata exclusion policy.
                    if (it->path().filename().string().starts_with(".")) {
                        if (it->is_directory()) it.disable_recursion_pending();
                        continue;
                    }
                    if (it->is_regular_file())
                        files.emplace(it->path().lexically_relative(p).generic_string(), it->path());
                }
            }
            Wire w;
            for (const auto &[name, file] : files) {
                w.s(name);
                w.s(asset_hash(file));
            }
            c.fields["asset:" + label] = sha(w.b);
        } else
            c.fields["asset:" + label] = asset_hash(p);
    }
#if defined(_WIN32)
    auto env = _environ;
#else
    auto env = ::environ;
#endif
    for (auto p = env; p && *p; ++p) {
        std::string e = *p;
        const auto n = e.find('=');
        if (n == std::string::npos)
            continue;
        const auto k = e.substr(0, n);
        // Explicit operational exclusions cover scheduler allocation variables
        // and cache controls. All unknown
        // STRIX_SESSION_* settings remain bound (future numerical controls fail closed).
        constexpr std::array operational{
            "GPU_WINDOW", "STRIX_HASH_CACHE", "STRIX_SESSION_DIR",
            "STRIX_SESSION_CACHE", "STRIX_SESSION_DISK", "STRIX_SESSION_DISK_BYTES",
            "STRIX_SESSION_RAM_BYTES", "STRIX_SESSION_MAX_PARKED",
            "STRIX_SESSION_POOL_IDENTITY_TEST"};
        if (k.starts_with("GPU_TURN_") ||
            std::any_of(operational.begin(), operational.end(), [&](auto name) { return k == name; }))
            continue;
        if (k.starts_with("STRIX_") || k.starts_with("HIP_") || k.starts_with("HSA_") ||
            k.starts_with("ROCR_") || k.starts_with("AMD_") || k.starts_with("GPU_"))
            c.fields["env:" + k] = e.substr(n + 1);
    }
    Wire inventory;
    for (const auto& [path, digest] : inventory_digests) { inventory.s(path.generic_string()); inventory.s(digest); }
    c.fields["loader-files-sha256"] = sha(inventory.b);
    if (loaded) verify_loader_snapshot(*loaded);
    return c;
}
std::string session_state_sha256(const AttentionState &state) {
    AttentionState s = state;
    const auto first = s.position > ds41::attn::kWindow ? s.position - ds41::attn::kWindow : 0U;
    std::array<bool, AttentionHost::ring_capacity> used{};
    for (auto p = first; p < s.position; ++p)
        used[p % used.size()] = true;
    for (auto &l : s.layers) {
        if (l.ring.size() != AttentionHost::ring_capacity * ds41::attn::kWindowEntryBytes)
            fail("canonical ring bounds");
        for (std::size_t p = 0; p < used.size(); ++p)
            if (!used[p])
                std::fill_n(l.ring.begin() + static_cast<std::ptrdiff_t>(p * ds41::attn::kWindowEntryBytes), ds41::attn::kWindowEntryBytes,
                            0);
        if (!l.carry_valid)
            std::fill(l.carry.begin(), l.carry.end(), 0.0F);
        l.topk.clear();
        l.topk_known = false;
    }
    return sha(serialize_attention_state(s));
}
SessionBytes serialize_session_draft(const SessionDraftState& s) {
    s.context.validate();
    s.controller.validate();
    if (s.accepted > s.verified) fail("DSpark acceptance bounds");
    Wire w;
    w.s("strix/ds41-dspark-context-v2");
    w.u(s.context.start); w.u(s.context.next); w.u(s.context.core);
    w.u(s.context.extra.size());
    for (auto id : s.context.extra) w.u(id);
    for (const auto& ring : s.context.rings) w.bytes(ring);
    const auto fp = [&](float x) {
        if (!std::isfinite(x) || x < 0) fail("DSpark controller cost");
        w.u(std::bit_cast<std::uint32_t>(x));
    };
    for (auto x : s.controller.costs.verify) fp(x);
    for (auto x : s.controller.costs.samples) w.u(x);
    fp(s.controller.costs.draft); fp(s.controller.costs.extra);
    for (const auto& row : s.controller.seen) for (auto x : row) w.u(x);
    for (const auto& row : s.controller.hits) for (auto x : row) w.u(x);
    w.u(s.controller.decisions); w.u(s.controller.sweep); w.u(s.controller.explore);
    w.u(s.exchange_request); w.u(s.windows); w.u(s.verified); w.u(s.accepted);
    w.u(0); // pending drafts: the R1 producer finishes/discards its local window before returning
    w.u(0); // staged hidden rows: flushed on every turn, including never-saved control
    return w.b;
}
SessionDraftState deserialize_session_draft(std::span<const std::uint8_t> bytes) {
    if (bytes.size() > (4U << 20U)) fail("DSpark section bounds");
    Wire w{SessionBytes(bytes.begin(), bytes.end())};
    if (w.s() != "strix/ds41-dspark-context-v2") fail("session_incompatible: DSpark schema");
    const auto u32 = [&]() {
        const auto x = w.u();
        if (x > UINT32_MAX) fail("DSpark integer bounds");
        return static_cast<std::uint32_t>(x);
    };
    SessionDraftState s;
    s.context.start = w.u(); s.context.next = w.u(); s.context.core = u32();
    const auto n = w.u();
    if (n > 129280) fail("DSpark vocabulary bounds");
    for (std::uint64_t i = 0; i < n; ++i) s.context.extra.push_back(u32());
    for (auto& ring : s.context.rings) ring = w.bytes(128U * 528U);
    for (auto& x : s.controller.costs.verify) x = std::bit_cast<float>(u32());
    for (auto& x : s.controller.costs.samples) x = u32();
    s.controller.costs.draft = std::bit_cast<float>(u32());
    s.controller.costs.extra = std::bit_cast<float>(u32());
    for (auto& row : s.controller.seen) for (auto& x : row) x = u32();
    for (auto& row : s.controller.hits) for (auto& x : row) x = u32();
    s.controller.decisions = w.u(); s.controller.sweep = u32();
    const auto explore = w.u();
    if (explore > 1) fail("DSpark explore bounds");
    s.controller.explore = explore != 0;
    s.exchange_request = w.u(); s.windows = w.u(); s.verified = w.u(); s.accepted = w.u();
    if (w.u() != 0 || w.u() != 0) fail("DSpark boundary has pending work");
    w.end();
    (void)serialize_session_draft(s); // common validation before caller can import either target or drafter
    return s;
}
std::string session_state_sha256(const SessionCheckpoint& c) {
    const auto target = session_state_sha256(c.attention);
    return c.draft ? sha(target + sha(serialize_session_draft(*c.draft))) : target;
}
SessionBytes serialize_session(const SessionCheckpoint &c) {
    validate(c);
    Wire w;
    w.s(c.draft ? "strix/ds41-session-v2" : "strix/ds41-session-v1");
    w.s(c.id);
    w.s(c.generation);
    w.u(c.world);
    w.u(c.rank);
    w.s(c.compat_record);
    w.u(c.draft ? 3 : 2); // section table: name, rank, replication, offset within its framed payload (zero), size, sha, payload
    const auto section = [&](const std::string &name, const std::string &replication, const SessionBytes &b) {
        w.s(name);
        w.u(c.rank);
        w.s(replication);
        w.u(0);
        w.u(b.size());
        w.s(sha(b));
        w.bytes(b);
    };
    section("frontier-history", "replicated", frontier(c));
    section("attention-v1", "sharded", serialize_attention_state(c.attention));
    if (c.draft) section("dspark-v2", "replicated", serialize_session_draft(*c.draft));
    w.s(sha(w.b));
    return w.b;
}
SessionCheckpoint deserialize_session(std::span<const std::uint8_t> bytes, const SessionCompat &compat, std::uint32_t world,
                                      std::uint32_t rank, std::uint64_t max) {
    if (bytes.size() > max)
        fail("envelope exceeds budget");
    Wire w{SessionBytes(bytes.begin(), bytes.end())};
    SessionCheckpoint c;
    const auto schema = w.s();
    const bool draft = schema == "strix/ds41-session-v2";
    if (!draft && schema != "strix/ds41-session-v1") fail("session_incompatible: envelope schema");
    if (draft != (compat.fields.contains("dspark") && compat.fields.at("dspark") == "on")) fail("session_incompatible: DSpark mode mismatch");
    c.id = w.s();
    c.generation = w.s();
    const auto cw = w.u(), cr = w.u();
    if (cw != world || cr != rank || cw != 2 || cr > 1)
        fail("session_incompatible: TP geometry mismatch");
    c.world = static_cast<std::uint32_t>(cw);
    c.rank = static_cast<std::uint32_t>(cr);
    c.compat_record = w.s();
    if (c.compat_record != compat.record())
        fail("session_incompatible: strict compatibility mismatch (model/build/config/schema)");
    if (w.u() != (draft ? 3U : 2U))
        fail("section count");
    std::array<SessionBytes, 3> sections;
    for (unsigned i = 0; i != (draft ? 3U : 2U); ++i) {
        if (w.s() != (i == 0 ? "frontier-history" : i == 1 ? "attention-v1" : "dspark-v2") || w.u() != rank || w.s() != (i == 1 ? "sharded" : "replicated") ||
            w.u() != 0)
            fail("section table");
        const auto n = w.u();
        const auto h = w.s();
        sections[i] = w.bytes(max);
        if (n != sections[i].size() || sha(sections[i]) != h)
            fail("section checksum/size");
    }
    const auto h = sha(std::span(w.b).first(w.at));
    if (w.s() != h)
        fail("envelope root checksum");
    w.end();
    if (draft) c.draft = deserialize_session_draft(sections[2]);
    c.attention = deserialize_attention_state(sections[1]);
    Wire f{sections[0]};
    c.frontier.consumed = f.u();
    c.frontier.emitted = f.u();
    c.frontier.accepted = f.u();
    c.frontier.drafter = f.u();
    const auto pending = f.u(), valid = f.u();
    if (pending > UINT32_MAX || valid > 1)
        fail("pending token bounds");
    c.frontier.pending = static_cast<std::uint32_t>(pending);
    c.frontier.pending_valid = valid != 0;
    c.frontier.sampling = f.s();
    c.frontier.lineage = f.s();
    if (f.u() != c.attention.tokens.size())
        fail("history length");
    for (auto t : c.attention.tokens)
        if (f.u() != t)
            fail("history mismatch");
    f.end();
    validate(c);
    if (const auto i = compat.fields.find("layers"); i != compat.fields.end() && c.attention.layers.size() != std::stoull(i->second))
        fail("configured layer count mismatch");
    if (const auto i = compat.fields.find("context"); i != compat.fields.end() && c.attention.position >= std::stoull(i->second))
        fail("configured context bounds");
    return c;
}

PrefixCache::Sessions::Sessions(SessionCompat c, std::uint32_t rank, std::uint64_t budget, std::filesystem::path dir,
    std::uint64_t max_parked, std::uint64_t disk_budget)
    : compat_(std::move(c)), rank_(rank), budget_(budget), max_parked_(max_parked), disk_budget_(disk_budget), dir_(std::move(dir)) {
    if (rank > 1 || !budget || !max_parked)
        fail("rank must be 0/1 and host budget must be nonzero");
    compat_.fields["rank"] = std::to_string(rank);
    if (!dir_.empty())
        std::filesystem::create_directories(dir_);
}
void PrefixCache::Sessions::agree(bool success, const SessionExchange &exchange, const std::string &phase) {
    SessionBytes out{static_cast<std::uint8_t>(success)};
    const auto in = exchange(out);
    if (!success || in.size() != 1 || in[0] != 1)
        fail(phase + " refused by one or both ranks");
}
std::filesystem::path PrefixCache::Sessions::manifest(const std::string &id) const {
    validate_session_id(id);
    return dir_ / (id + ".rank" + std::to_string(rank_) + ".group");
}
void PrefixCache::Sessions::capture(const SessionCheckpoint &c, const SessionExchange &exchange) {
    SessionBytes mine;
    std::map<std::string, Entry> staged;
    bool ok = true;
    try {
        if (c.rank != rank_ || c.compat_record != compat_.record() ||
            c.draft.has_value() != (compat_.fields.contains("dspark") && compat_.fields.at("dspark") == "on"))
            fail("capture compatibility mismatch");
        auto b = serialize_session(c);
        // Compact before charging admission: vector growth capacity must not hide parked bytes.
        SessionBytes compact(b.begin(), b.end());
        b.swap(compact);
        if (b.capacity() > budget_)
            fail("capture exceeds host budget");
        mine = describe(c, b);
        staged.emplace(c.id, Entry{std::move(b), clock_ + 1});
    } catch (...) {
        ok = false;
    }
    if (!ok)
        ++stats_.refused;
    agree(ok, exchange, "capture validation/budget/allocation");
    const auto peer = exchange(mine);
    ok = true;
    try {
        (void)group(mine, peer, compat_, rank_);
    } catch (...) {
        ok = false;
    }
    agree(ok, exchange, "capture generation/frontier/compat");
    const auto bytes = staged.begin()->second.bytes.capacity();
    if (const auto old = ram_.find(c.id); old != ram_.end()) {
        stats_.bytes -= old->second.bytes.capacity();
        ram_.erase(old);
    }
    while (stats_.bytes + bytes > budget_ || ram_.size() >= max_parked_) {
        const auto i =
            std::min_element(ram_.begin(), ram_.end(), [](const auto &a, const auto &z) { return a.second.used < z.second.used; });
        stats_.bytes -= i->second.bytes.capacity();
        ram_.erase(i);
        ++stats_.evictions;
    }
    // All allocations and both-rank validation precede this publication. Node transfer allocates nothing.
    ram_.insert(staged.extract(staged.begin()));
    stats_.bytes += bytes;
    ++clock_;
    if (!dir_.empty() && std::filesystem::exists(manifest(c.id))) {
        try {
            if (parse_group(read(manifest(c.id), 4U << 20U))[rank_] == mine) dirty_.erase(c.id);
            else dirty_.insert(c.id);
        } catch (...) { dirty_.insert(c.id); }
    }
}
void PrefixCache::Sessions::remove_disk(const std::string& id, const std::filesystem::path& keep) {
    validate_session_id(id);
    for (const auto& e : std::filesystem::directory_iterator(dir_)) {
        if (e.is_symlink() || !e.is_regular_file()) continue;
        const auto n = e.path().filename().string();
        if (disk_owner(n, rank_) == id && e.path() != keep &&
            (keep.empty() || e.path() != manifest(id))) std::filesystem::remove(e.path());
    }
}
void PrefixCache::Sessions::reserve_disk(const std::string& id, std::uint64_t bytes, const SessionExchange& exchange) {
    // Rank 0 chooses LRU victims; both ranks remove the same IDs. A failed
    // checkpoint keeps the current ID's previous committed generation intact.
    for (;;) {
        bool fits = false;
        bool ok = true;
        try { const auto used = disk_bytes(); fits = bytes <= disk_budget_ && used <= disk_budget_ - bytes; }
        catch (...) { ok = false; }
        agree(ok, exchange, "disk quota inspection");
        const auto peer = exchange(SessionBytes{static_cast<std::uint8_t>(fits)});
        if (fits && peer.size() == 1 && peer[0] == 1) return;
        std::string victim;
        ok = true;
        try {
            if (rank_ == 0) {
                auto oldest = std::filesystem::file_time_type::max();
                for (const auto& e : std::filesystem::directory_iterator(dir_)) {
                    const auto n = e.path().filename().string(), tail = std::string(".rank0.group");
                    if (e.is_symlink() || !e.is_regular_file() || !n.ends_with(tail)) continue;
                    const auto candidate = n.substr(0, n.size() - tail.size());
                    if (candidate == id) continue;
                    validate_session_id(candidate);
                    const auto used = e.last_write_time();
                    if (used < oldest || (used == oldest && candidate < victim)) { oldest = used; victim = candidate; }
                }
            }
        } catch (...) { ok = false; }
        agree(ok, exchange, "disk LRU selection");
        const auto chosen = exchange(SessionBytes(victim.begin(), victim.end()));
        if (rank_ == 1) victim.assign(chosen.begin(), chosen.end());
        agree(!victim.empty(), exchange, "session_budget: disk quota");
        ok = true;
        try { remove_disk(victim); } catch (...) { ok = false; }
        agree(ok, exchange, "disk LRU eviction");
    }
}
void PrefixCache::Sessions::checkpoint(const std::string &id, const SessionExchange &exchange) {
    validate_session_id(id);
    auto i = ram_.find(id);
    agree(!dir_.empty() && i != ram_.end(), exchange, "disk checkpoint requires captured RAM state and directory");
    const auto c = deserialize_session(i->second.bytes, compat_, 2, rank_, budget_);
    const auto path = dir_ / (id + "." + c.generation + ".rank" + std::to_string(rank_) + ".state");
    DiskLock lock;
    lock.transaction(dir_, rank_, exchange);
    reserve_disk(id, i->second.bytes.size() + (4U << 20U), exchange);
    bool ok = true;
    std::exception_ptr io_failure;
    SessionBytes previous_group;
    bool had_group = false;
    try {
        if (std::filesystem::is_symlink(path) || std::filesystem::is_symlink(manifest(id)))
            fail("refusing symlink publication");
        had_group = std::filesystem::exists(manifest(id));
        if (had_group) previous_group = read(manifest(id), 4U << 20U);
        if (std::filesystem::exists(path)) {
            if (read(path, budget_) != i->second.bytes)
                fail("generation already exists with different bytes");
        } else
            durable(path, i->second.bytes);
    } catch (...) {
        ok = false; io_failure = std::current_exception();
    }
    try { agree(ok, exchange, "rank write/fsync"); }
    catch (...) { if (io_failure) std::rethrow_exception(io_failure); throw; }
    const auto mine = describe(c, i->second.bytes), peer = exchange(mine);
    const auto g = group(mine, peer, compat_, rank_);
    ok = true;
    try {
        durable(manifest(id), g);
    } catch (...) {
        ok = false; io_failure = std::current_exception();
    }
    try { agree(ok, exchange, "group manifest publication"); }
    catch (...) {
        const auto publication_failure = io_failure ? io_failure : std::current_exception();
        // A peer may already have replaced its manifest. Restore exact prior
        // bytes on BOTH ranks while the ordered locks remain held. This is a
        // recovery attempt, not a guarantee under persistent storage failures.
        std::exception_ptr rollback_failure;
        try {
            if (had_group) durable(manifest(id), previous_group);
            else {
                std::filesystem::remove(manifest(id));
                io::sync_directory(dir_);
            }
        } catch (const io::Error& e) {
            rollback_failure = std::make_exception_ptr(io::Error(e.operation,
                (std::string("ds41 session cache: group manifest rollback; durable commit uncertain: ") + e.what()).c_str()));
        } catch (...) { rollback_failure = std::current_exception(); }
        try { agree(!rollback_failure, exchange, "group manifest rollback; durable commit uncertain"); }
        catch (...) { if (rollback_failure) std::rethrow_exception(rollback_failure); throw; }
        std::rethrow_exception(publication_failure);
    }
    i->second.used = ++clock_;
    dirty_.erase(id);
    ok = true;
    try { remove_disk(id, path); } catch (...) { ok = false; }
    agree(ok, exchange, "old generation cleanup");
}
SessionCheckpoint PrefixCache::Sessions::resume(const std::string &id, bool disk, const SessionExchange &exchange) {
    validate_session_id(id);
    DiskLock lock;
    lock.transaction(disk ? dir_ : std::filesystem::path{}, rank_, exchange);
    restore_timing_ = {};
    const auto local_start = std::chrono::steady_clock::now();
    const auto timed_read = [&](const std::filesystem::path &path, std::uint64_t limit) {
        const auto start = std::chrono::steady_clock::now();
        auto bytes = read(path, limit);
        restore_timing_.read_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return bytes;
    };
    SessionBytes b, g;
    SessionCheckpoint c;
    bool ok = true;
    std::string code;
    try {
        if (disk) {
            if (dir_.empty() || dirty_.contains(id) || !std::filesystem::exists(manifest(id)))
                fail("session_not_found");
            g = timed_read(manifest(id), 4U << 20U);
            const auto ds = parse_group(g);
            const auto d = descriptor(ds[rank_]);
            if (d.bytes > budget_) fail("session_budget");
            if (d.id != id || d.rank != rank_)
                fail("rank manifest bounds");
            b = timed_read(dir_ / (id + "." + d.generation + ".rank" + std::to_string(rank_) + ".state"), budget_);
            if (b.size() != d.bytes || sha(b) != d.payload)
                fail("rank file checksum/size");
            c = deserialize_session(b, compat_, 2, rank_, budget_);
            if (describe(c, b) != ds[rank_])
                fail("rank descriptor mismatch");
            if (group(ds[rank_], ds[1U - rank_], compat_, rank_) != g)
                fail("noncanonical group manifest");
        } else {
            const auto i = ram_.find(id);
            if (i == ram_.end())
                fail("session_not_found: RAM session missing/evicted");
            b = i->second.bytes;
            c = deserialize_session(b, compat_, 2, rank_, budget_);
        }
    } catch (const std::exception &e) {
        ok = false;
        const std::string message = e.what();
        code = failure_code(message);
    }
    restore_timing_.payload_bytes = b.size();
    const auto peer_start = std::chrono::steady_clock::now();
    restore_timing_.verify_seconds = std::chrono::duration<double>(peer_start - local_start).count() - restore_timing_.read_seconds;
    const auto peer_code = exchange(SessionBytes(code.begin(), code.end()));
    if (!ok || !peer_code.empty()) {
        ++stats_.refused;
        fail("restore validation " + (code.empty() ? std::string(peer_code.begin(), peer_code.end()) : code) + " refused by one or both ranks");
    }
    const auto mine = describe(c, b), peer = exchange(mine);
    const auto agreed = group(mine, peer, compat_, rank_);
    if (disk)
        agree(g == agreed, exchange, "restore group identity");
    else
        ram_.at(id).used = ++clock_;
    if (disk) {
        std::error_code ec;
        std::filesystem::last_write_time(manifest(id), std::filesystem::file_time_type::clock::now(), ec);
    }
    restore_timing_.peer_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - peer_start).count();
    return c; // both ranks validated; engine must still stage and agree import
}
void PrefixCache::Sessions::erase(const std::string &id, const SessionExchange &exchange) {
    validate_session_id(id);
    DiskLock lock;
    lock.transaction(dir_, rank_, exchange);
    agree(true, exchange, "delete");
    dirty_.erase(id);
    if (auto i = ram_.find(id); i != ram_.end()) {
        stats_.bytes -= i->second.bytes.capacity();
        ram_.erase(i);
    }
    if (!dir_.empty()) remove_disk(id);
}
std::vector<std::string> PrefixCache::Sessions::list() const {
    std::vector<std::string> out;
    for (const auto &[id, e] : ram_) {
        (void)e;
        out.push_back(id);
    }
    if (!dir_.empty())
        for (const auto &e : std::filesystem::directory_iterator(dir_)) {
            const auto name = e.path().filename().string(), tail = ".rank" + std::to_string(rank_) + ".group";
            if (name.ends_with(tail)) {
                const auto id = name.substr(0, name.size() - tail.size());
                if (!dirty_.contains(id)) out.push_back(id);
            }
        }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}
SessionInfo PrefixCache::Sessions::inspect(const std::string &id) const {
    validate_session_id(id);
    try {
        DiskLock lock;
        lock.acquire(dir_, rank_);
        SessionInfo info;
        info.disk = !dir_.empty() && !dirty_.contains(id) && std::filesystem::exists(manifest(id));
        if (const auto i = ram_.find(id); i != ram_.end()) {
            info.ram = true; info.bytes = i->second.bytes.capacity();
            info.checkpoint = deserialize_session(i->second.bytes, compat_, 2, rank_, budget_);
        } else {
            if (!info.disk) fail("session_not_found");
            const auto ds = parse_group(read(manifest(id), 4U << 20U));
            const auto d = descriptor(ds[rank_]);
            const auto b = read(dir_ / (id + "." + d.generation + ".rank" + std::to_string(rank_) + ".state"), budget_);
            info.bytes = b.size(); info.checkpoint = deserialize_session(b, compat_, 2, rank_, budget_);
            if (describe(info.checkpoint, b) != ds[rank_] || group(ds[rank_], ds[1U-rank_], compat_, rank_) != read(manifest(id), 4U << 20U))
                fail("session_corrupt: manifest mismatch");
        }
        return info;
    } catch (const std::exception &e) {
        const std::string message = e.what();
        if (message.find("session_not_found") != std::string::npos) throw;
        fail(failure_code(message) + ": " + message);
    }
}
std::uint64_t PrefixCache::Sessions::disk_bytes() const {
    std::uint64_t bytes = 0;
    if (!dir_.empty()) for (const auto &e : std::filesystem::directory_iterator(dir_)) {
        const auto name = e.path().filename().string();
        if (!e.is_symlink() && e.is_regular_file() && !disk_owner(name, rank_).empty()) {
            const auto n = e.file_size();
            if (n > UINT64_MAX - bytes) fail("disk byte budget overflow");
            bytes += n;
        }
    }
    return bytes;
}
SessionCacheStats PrefixCache::Sessions::stats() const {
    auto s = stats_;
    s.entries = ram_.size();
    return s;
}
} // namespace strix::deepseek41
