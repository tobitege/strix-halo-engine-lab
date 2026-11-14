#pragma once
#include "state_cache_stats.hpp"
// Serialized by the bridge request mutex. Uses the existing prefix-cache allocator/LRU.
#include "strix/session_state_cache.hpp"
#include "strix/session/checkpoint_io.hpp"
#include "strix/serve/json.hpp"
#include "strix/serve/session_api.hpp"
#include <algorithm>
#include <set>
#include <array>
#include <cstring>
#include <cerrno>
namespace strix::decode_cli {
inline void validate_http_session_id(const std::string& id) {
    if (id.empty() || id.size() > 128 || !std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_';
    })) throw std::invalid_argument("session id must be 1..128 ASCII letters, digits, - or _");
}
inline bool session_prompt_extends(const SessionFrontier& f, std::span<const std::uint32_t> prompt) {
    if (prompt.size() < f.tokens.size() + (f.pending ? 1U : 0U) ||
        !std::equal(f.tokens.begin(), f.tokens.end(), prompt.begin())) return false;
    return !f.pending || prompt[f.tokens.size()] == *f.pending;
}
using serve::SessionApiError;
// Fresh observation is supplied by the caller after resident and retained owners exist.
// This pure admission seam refuses before any checkpoint allocation or device work.
inline std::uint64_t session_capture_peak_bound(std::uint64_t fixed, std::uint64_t unit,
    std::uint64_t rows, std::uint64_t host_budget, std::uint64_t available) {
    const auto limit=session_checkpoint_envelope_limit(host_budget,available);
    if (!unit || fixed>limit || rows>(limit-fixed)/unit)
        throw SessionApiError(507,"session_budget","session capture exceeds fresh UMA/host peak budget");
    return 4U*(fixed+rows*unit);
}
// One persistent lock inode per shared directory. Never unlink it after unlocking.
class SessionDirectoryLock {
    session::checkpoint_io::DirectoryLock lock_;
public:
    explicit SessionDirectoryLock(const std::filesystem::path& dir) try
        : lock_(dir, session::checkpoint_io::LockMode::blocking) {}
    catch (const std::exception&) {
        throw SessionApiError(507, "session_budget", "session directory lock failed");
    }
};
class HttpSessionStore final {
public:
    HttpSessionStore(std::filesystem::path directory, std::uint64_t bytes, std::uint64_t count,
        PrefixCacheAllocator allocator, SessionCompatibility compatibility, std::uint64_t disk_bytes = UINT64_MAX)
        : directory_(std::move(directory)), disk_bytes_(disk_bytes), count_(count),
          dirty_limit_(directory_.empty() ? 0U : std::min(count, bytes / sizeof(Tombstone))),
          metadata_bytes_(dirty_limit_ * sizeof(Tombstone)), allocator_(std::move(allocator)),
          compatibility_(std::move(compatibility)), cache_({bytes - metadata_bytes_, {}, 0, PrefixCacheRetention::lru}, allocator_) {
        // Named-session parking stays LRU pending agreement on a separate policy change.
        dirty_.reserve(static_cast<std::size_t>(dirty_limit_));
        disk_gauge_=disk_usage();
    }
    void admit(std::uint64_t bytes) {
        if (!cache_.admit_session_working_bytes(bytes))
            budget("session RAM working-set budget exceeded");
    }
    PrefixCacheBuffer get(const std::string& id, bool explicit_restore = false, bool promote = true) {
        validate_http_session_id(id);
        if (auto buffer = cache_.resume_session(id); buffer.data) {
            // Retained payload stays charged; reserve additional parser/metadata/lineage storage.
            if (buffer.bytes <= UINT64_MAX / 4U && cache_.admit_session_working_bytes(buffer.bytes * 3U)) return buffer;
            if (explicit_restore) budget("session RAM restore working-set budget exceeded");
            return {};
        }
        if (disk_fallback_disabled_ || is_dirty(id) || directory_.empty() || !std::filesystem::exists(directory_ / id / "session.bin")) return {};
        try {
            SessionDirectoryLock lock(directory_);
            const auto size = std::filesystem::file_size(directory_ / id / "session.bin");
            // Reader + decoded metadata bounded conservatively before allocation.
            if (size > UINT64_MAX / 4U) throw std::runtime_error("session file too large");
            admit(size * 4U);
            std::error_code touch_error;
            std::filesystem::last_write_time(directory_ / id / "session.bin", std::filesystem::file_time_type::clock::now(), touch_error);
            auto buffer = read_session_checkpoint(directory_ / id, compatibility_, allocator_, size);
            if (promote && !cache_.park_session(id, buffer, count_)) budget("session RAM admission failed");
            return buffer;
        } catch (const SessionApiError&) { if (explicit_restore) throw; return {}; }
        catch (const std::exception& e) {
            if (explicit_restore) {
                const bool incompatible = std::string_view(e.what()).find("compatibility mismatch") != std::string_view::npos;
                throw SessionApiError(incompatible ? 409 : 422, incompatible ? "session_incompatible" : "session_corrupt", e.what());
            }
            return {};
        }
    }
    std::optional<SessionCheckpoint> matching(const std::string& id, std::span<const std::uint32_t> prompt, bool promote = true) {
        auto buffer = get(id, false, promote);
        if (!buffer.data) { ++misses_; return {}; }
        try {
            auto checkpoint = decode_session_checkpoint(buffer, compatibility_);
            if (session_prompt_extends(checkpoint.frontier, prompt)) { ++hits_; return checkpoint; }
        } catch (...) {}
        ++misses_; return {};
    }
    void invalidate_turn(const std::string& id) {
        validate_http_session_id(id); (void)cache_.delete_session(id); mark_dirty(id);
    }
    void park(const std::string& id, PrefixCacheBuffer buffer) {
        validate_http_session_id(id);
        // Default disk tier retains completed turns for restart. Failure leaves RAM usable.
        bool durable = false;
        if (!directory_.empty() && disk_bytes_ != UINT64_MAX) {
            try { publish_disk(id, buffer); durable = true; } catch (const std::exception&) {}
        }
        if (!cache_.park_session(id, std::move(buffer), count_)) budget("session RAM admission failed");
        // Keep the explicitly durable checkpoint for restart, but never silently fall back to
        // that older turn after an in-process eviction of a newer RAM frontier.
        if (durable) clear_dirty(id); else mark_dirty(id);
    }
    StateCacheSnapshotV1 snapshot() {
        StateCacheSnapshotV1 s; s.session_enabled=true;
        const auto c=cache_.counters();
        s.named={hits_,misses_,refusals_,c.host_checkpoints,no_store_skips_};
        s.host={c.host_bytes+metadata_bytes_,c.host_segments,c.host_checkpoints,cache_.policy().host_byte_budget+metadata_bytes_};
        s.disk={disk_gauge_,0,0,disk_bytes_}; return s;
    }
    void no_store_skip() { ++no_store_skips_; }
    std::string command(const std::string& op, const std::string& id, const std::string& durability) {
        try {
            using serve::Json;
            Json out = Json::object();
            if (op == "list") {
                std::unique_ptr<SessionDirectoryLock> directory_lock;
                if (!directory_.empty() && std::filesystem::exists(directory_))
                    directory_lock = std::make_unique<SessionDirectoryLock>(directory_);
                std::set<std::string> names;
                for (const auto& name : cache_.list_sessions()) names.insert(name);
                if (!directory_.empty() && std::filesystem::exists(directory_))
                    for (const auto& entry : std::filesystem::directory_iterator(directory_))
                        if (entry.is_directory() && std::filesystem::exists(entry.path() / "session.bin")) {
                            const auto name = entry.path().filename().string();
                            try { validate_http_session_id(name); names.insert(name); } catch (...) {}
                        }
                Json rows = Json::array();
                const auto ram = cache_.list_sessions();
                for (const auto& name : names) {
                    Json row = Json::object(); row.set("id", Json(name));
                    const bool parked = std::find(ram.begin(), ram.end(), name) != ram.end();
                    const bool disk = !directory_.empty() && std::filesystem::exists(directory_ / name / "session.bin");
                    auto buffer = cache_.resume_session(name);
                    if (!buffer.data) {
                        const auto size = std::filesystem::file_size(directory_ / name / "session.bin");
                        admit(size > UINT64_MAX / 4 ? UINT64_MAX : size * 4);
                        buffer = read_session_checkpoint(directory_ / name, compatibility_, allocator_, size);
                    }
                    const auto checkpoint = decode_session_checkpoint(buffer, compatibility_);
                    row.set("position", Json(checkpoint.frontier.consumed));
                    row.set("consumed", Json(checkpoint.frontier.consumed));
                    row.set("pending", checkpoint.frontier.pending ? Json(static_cast<std::uint64_t>(*checkpoint.frontier.pending)) : Json());
                    row.set("bytes", Json(buffer.bytes));
                    row.set("state", Json(parked ? "parked" : "disk"));
                    row.set("durability", Json(disk && !is_dirty(name) ? "disk" : "ram"));
                    rows.push(std::move(row));
                }
                out.set("sessions", std::move(rows));
            } else if (op == "stats") {
                const auto stats = cache_.counters();
                out.set("ram_bytes", Json(stats.host_bytes + metadata_bytes_)); out.set("ram_budget", Json(cache_.policy().host_byte_budget + metadata_bytes_));
                Json engine = Json::object();
                engine.set("state_cache", state_cache_json(snapshot()));
                engine.set("tombstone_bytes", Json(metadata_bytes_));
                engine.set("tombstones", Json(static_cast<std::uint64_t>(dirty_.size())));
                engine.set("disk_fallback_disabled", Json(disk_fallback_disabled_));
                out.set("parked", Json(static_cast<std::uint64_t>(cache_.list_sessions().size())));
                engine.set("max_parked", Json(count_)); engine.set("evicted", Json(stats.evicted));
                out.set("engine", std::move(engine));
                out.set("disk_bytes", Json(disk_gauge_)); out.set("refusals", Json(refusals_));
                out.set("hits", Json(hits_)); out.set("misses", Json(misses_));
            } else {
                validate_http_session_id(id);
                out.set("id", Json(id));
                if (op == "delete") {
                    bool removed = cache_.delete_session(id);
                    const auto old_disk=directory_.empty() ? std::filesystem::path{} : directory_/id/"session.bin";
                    if (!old_disk.empty() && std::filesystem::exists(old_disk)) disk_gauge_-=std::min(disk_gauge_,std::filesystem::file_size(old_disk));
                    // Delete only the validated session's published file; never recursively remove directories.
                    if (!directory_.empty()) {
                        SessionDirectoryLock lock(directory_);
                        removed = std::filesystem::remove(directory_ / id / "session.bin") || removed;
                    }
                    if (!removed) throw SessionApiError(404, "session_not_found", "unknown or evicted session");
                    clear_dirty(id);
                    out.set("deleted", Json(removed));
                } else {
                    auto buffer = get(id, true);
                    if (!buffer.data) throw SessionApiError(404, "session_not_found", "unknown or evicted session");
                    (void)decode_session_checkpoint(buffer, compatibility_);
                    if (op == "checkpoint" && durability == "disk") {
                        if (directory_.empty()) budget("disk checkpoint requires an enabled disk tier");
                        publish_disk(id, buffer);
                        clear_dirty(id);
                    } else if (op != "park" && op != "resume" && !(op == "checkpoint" && durability == "ram"))
                        throw std::invalid_argument("unknown operation or durability (ram|disk)");
                    Json engine = Json::object(); engine.set("parked", Json(true)); out.set("engine", std::move(engine));
                }
            }
            return out.dump();
        } catch (const SessionApiError&) { throw; }
        catch (const std::invalid_argument&) { throw; }
        catch (const std::exception& e) {
            const bool incompatible = std::string_view(e.what()).find("compatibility mismatch") != std::string_view::npos;
            throw SessionApiError(incompatible ? 409 : 422, incompatible ? "session_incompatible" : "session_corrupt", e.what());
        }
    }
private:
    [[noreturn]] void budget(const char* message) { ++refusals_; throw SessionApiError(507, "session_budget", message); }
    std::uint64_t disk_usage() const {
        if (directory_.empty() || !std::filesystem::exists(directory_)) return 0;
        SessionDirectoryLock lock(directory_);
        std::uint64_t bytes = 0;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directory_)) {
            if (entry.is_regular_file() && !entry.is_symlink() &&
                (entry.path().filename() == "session.bin" || entry.path().filename().string().starts_with(".session-")))
                bytes += entry.file_size();
        }
        return bytes;
    }
    void publish_disk(const std::string& id, const PrefixCacheBuffer& buffer) {
        const auto previous_refusals = refusals_;
        try { publish_disk_locked(id, buffer); }
        catch (const SessionApiError& e) {
            if (e.status == 507 && refusals_ == previous_refusals) ++refusals_;
            throw;
        }
        catch (const std::exception& e) { ++refusals_; throw SessionApiError(507, "session_budget", e.what()); }
    }
    void publish_disk_locked(const std::string& id, const PrefixCacheBuffer& buffer) {
        if (buffer.bytes > disk_bytes_) budget("session disk budget exceeded");
        SessionDirectoryLock lock(directory_);
        struct Entry { std::filesystem::path file; std::filesystem::file_time_type time; std::uint64_t bytes; };
        std::vector<Entry> files;
        std::uint64_t used=0;
        for (const auto& entry : std::filesystem::directory_iterator(directory_)) {
            if (!entry.is_directory() || entry.is_symlink()) continue;
            try { validate_http_session_id(entry.path().filename().string()); } catch (...) { continue; }
            // Interrupted temporary publications still occupy quota until removed under this lock.
            for (const auto& temporary : std::filesystem::directory_iterator(entry.path()))
                if (temporary.is_regular_file() && temporary.path().filename().string().starts_with(".session-") &&
                    temporary.path().extension() == ".tmp") std::filesystem::remove(temporary.path());
            const auto file=entry.path()/"session.bin";
            if (!std::filesystem::is_regular_file(file) || std::filesystem::is_symlink(file)) continue;
            const auto size=std::filesystem::file_size(file);
            if (size > UINT64_MAX-used) throw std::runtime_error("session disk usage overflow");
            used+=size;
            if (entry.path().filename()!=id) files.push_back({file,std::filesystem::last_write_time(file),size});

        }
        std::sort(files.begin(),files.end(),[](const auto& a,const auto& b){return a.time<b.time;});
        for (const auto& file:files) {
            if (used <= disk_bytes_-buffer.bytes) break;
            std::filesystem::remove(file.file); used-=file.bytes;
        }
        // Old destination remains present while the new temporary envelope is written.
        if (used > disk_bytes_ - buffer.bytes) budget("session replacement peak exceeds disk budget");
        const auto old=std::filesystem::exists(directory_/id/"session.bin") ? std::filesystem::file_size(directory_/id/"session.bin") : 0;
        disk_gauge_=used;
        try { publish_session_checkpoint(directory_/id,buffer); disk_gauge_=used-old+buffer.bytes; }
        catch (const std::exception& e) { ++refusals_; throw SessionApiError(507, "session_budget", e.what()); }
    }
    // Fixed storage, reserved out of the RAM budget before any envelopes are admitted.
    // Saturation fails closed for disk lookup until restart; forgetting one name must
    // never resurrect its stale durable frontier. RAM-only names need no tombstone.
    using Tombstone = std::array<char, 129>;
    bool is_dirty(const std::string& id) const {
        return std::any_of(dirty_.begin(), dirty_.end(), [&](const auto& name) { return id == name.data(); });
    }
    void clear_dirty(const std::string& id) {
        std::erase_if(dirty_, [&](const auto& name) { return id == name.data(); });
    }
    void mark_dirty(const std::string& id) {
        if (disk_fallback_disabled_ || directory_.empty() || is_dirty(id)) return;
        std::error_code error;
        const bool durable = std::filesystem::exists(directory_ / id / "session.bin", error);
        if (error) { disk_fallback_disabled_ = true; dirty_.clear(); return; }
        if (!durable) return;
        if (dirty_.size() == dirty_limit_) {
            disk_fallback_disabled_ = true;
            dirty_.clear();
            return;
        }
        Tombstone name{};
        std::memcpy(name.data(), id.data(), id.size());
        dirty_.push_back(name);
    }
    std::uint64_t disk_gauge_=0, no_store_skips_=0;
    std::filesystem::path directory_;
    std::uint64_t disk_bytes_;
    std::uint64_t count_, dirty_limit_, metadata_bytes_, hits_ = 0, misses_ = 0, refusals_ = 0;
    PrefixCacheAllocator allocator_;
    SessionCompatibility compatibility_;
    PrefixStateCache cache_;
    std::vector<Tombstone> dirty_;
    bool disk_fallback_disabled_ = false;
};
} // namespace strix::decode_cli
