#include "strix/session/embed_rows.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace strix::session {

std::size_t count_embed_ids(std::span<const std::uint32_t> ids) noexcept {
    std::size_t n = 0;
    for (const auto id : ids) n += id >> 31;
    return n;
}

void reject_embed_ids(std::span<const std::uint32_t> ids) {
    if (has_embed_ids(ids)) throw std::invalid_argument("engine does not support embedding rows");
}

std::size_t EmbedRows::bytes() const noexcept {
    std::size_t n = data.size() * sizeof(std::uint16_t) + role.size() + lm_pos.size() * sizeof(std::int32_t);
    for (const auto& d : deep) n += d.data.size() * sizeof(std::uint16_t);
    return n;
}

struct EmbedInterner::State {
    struct Entry {
        EmbedDigest digest{};
        std::shared_ptr<const EmbedRows> rows; // null: retired or free
        std::uint32_t refs = 0;
        bool reserved = false; // digest -> handle mapping exists
    };
    mutable std::mutex mutex;
    std::uint32_t capacity = 0;
    std::vector<Entry> entries; // grows to capacity; index = handle
    std::vector<std::uint32_t> free; // filled by an epoch bump
    std::map<EmbedDigest, std::uint32_t> by_digest;
    EmbedInternerStatus status;

    void retain(std::uint32_t handle) {
        const std::lock_guard lock(mutex);
        ++entries[handle].refs;
    }
    void release(std::uint32_t handle) noexcept {
        std::shared_ptr<const EmbedRows> dropped; // freed outside the lock
        {
            const std::lock_guard lock(mutex);
            auto& e = entries[handle];
            if (--e.refs != 0) return;
            status.live_bytes -= e.rows->bytes();
            dropped = std::move(e.rows);
            --status.live;
            ++status.retired;
        }
    }
};

namespace {
void check_rows(const EmbedRows& r) {
    const auto fail = [](const char* what) { throw std::invalid_argument(what); };
    if (r.rows == 0 || r.rows > kEmbedMaxRows) fail("embed rows: row count out of range");
    if (r.width == 0) fail("embed rows: zero width");
    const auto cells = static_cast<std::size_t>(r.rows) * r.width;
    if (r.data.size() != cells) fail("embed rows: data size differs from rows * width");
    if (!r.role.empty() && r.role.size() != r.rows) fail("embed rows: role size differs from rows");
    if (r.lm_pos.size() != static_cast<std::size_t>(r.rows) * r.axes)
        fail("embed rows: position size differs from rows * axes");
    if (is_embed_id(r.real_token_id)) fail("embed rows: real token id is a virtual id");
    for (const auto& d : r.deep)
        if (d.data.size() != cells) fail("embed rows: deep data size differs from rows * width");
}
} // namespace

EmbedLease::EmbedLease(const EmbedLease& other)
    : owner_(other.owner_), rows_(other.rows_), handle_(other.handle_) {
    if (rows_) static_cast<EmbedInterner::State*>(owner_.get())->retain(handle_);
}
EmbedLease::EmbedLease(EmbedLease&& other) noexcept
    : owner_(std::move(other.owner_)), rows_(std::move(other.rows_)), handle_(other.handle_) {
    other.rows_.reset();
    other.owner_.reset();
    other.handle_ = 0;
}
EmbedLease& EmbedLease::operator=(EmbedLease other) noexcept {
    owner_.swap(other.owner_);
    rows_.swap(other.rows_);
    std::swap(handle_, other.handle_);
    return *this;
}
EmbedLease::~EmbedLease() {
    if (rows_) static_cast<EmbedInterner::State*>(owner_.get())->release(handle_);
}
std::vector<std::uint32_t> EmbedLease::ids() const {
    std::vector<std::uint32_t> out;
    if (!rows_) return out;
    out.reserve(rows_->rows);
    for (std::uint32_t r = 0; r < rows_->rows; ++r) out.push_back(embed_id(handle_, r));
    return out;
}

EmbedInterner::EmbedInterner(std::uint32_t capacity) : state_(std::make_shared<State>()) {
    if (capacity == 0 || capacity > kEmbedMaxHandles) throw std::invalid_argument("embed interner: capacity");
    state_->capacity = capacity;
}
EmbedInterner::~EmbedInterner() = default;

EmbedLease EmbedInterner::intern(EmbedRows rows) {
    check_rows(rows);
    if (rows.role.empty()) rows.role.assign(rows.rows, EmbedRole::content);
    // Everything that can throw for lack of memory happens before the table changes.
    const auto digest = rows.digest;
    const auto shape_rows = rows.rows, shape_width = rows.width;
    auto payload = std::make_shared<const EmbedRows>(std::move(rows));
    auto& s = *state_;
    EmbedLease lease;
    const std::lock_guard lock(s.mutex);
    std::uint32_t handle = 0;
    bool revived = false;
    if (const auto it = s.by_digest.find(digest); it != s.by_digest.end()) {
        handle = it->second;
        auto& e = s.entries[handle];
        if (e.rows) { // live: the first payload stays
            if (e.rows->rows != shape_rows || e.rows->width != shape_width)
                throw std::invalid_argument("embed rows: digest is live with another shape");
            ++e.refs;
            ++s.status.shared;
            lease.rows_ = e.rows;
            lease.handle_ = handle;
            lease.owner_ = state_;
            return lease;
        }
        revived = true; // same handle again
    } else {
        if (s.entries.size() < s.capacity) {
            s.entries.emplace_back(); // may throw before any state changed
            handle = static_cast<std::uint32_t>(s.entries.size() - 1);
        } else {
            if (s.free.empty()) { // handle space used up: new epoch, unleased handles become free
                std::vector<std::uint32_t> freed;
                for (std::uint32_t h = 0; h < s.capacity; ++h)
                    if (s.entries[h].reserved && !s.entries[h].rows) freed.push_back(h);
                if (freed.empty()) throw std::length_error("embed interner: every handle is leased");
                for (const auto h : freed) {
                    s.by_digest.erase(s.entries[h].digest);
                    s.entries[h].reserved = false;
                }
                s.status.retired = 0;
                s.free = std::move(freed);
                std::reverse(s.free.begin(), s.free.end()); // lowest handle first
                ++s.status.epoch;
            }
            handle = s.free.back();
        }
    }
    auto& e = s.entries[handle];
    if (!e.reserved) {
        try {
            s.by_digest.emplace(payload->digest, handle);
        } catch (...) {
            if (handle + 1U == s.entries.size() && (s.free.empty() || s.free.back() != handle)) s.entries.pop_back();
            throw;
        }
        if (!s.free.empty() && s.free.back() == handle) s.free.pop_back();
        e.digest = payload->digest;
        e.reserved = true;
    }
    if (revived) --s.status.retired;
    e.rows = payload;
    e.refs = 1;
    ++s.status.live;
    ++s.status.interned;
    s.status.live_bytes += payload->bytes();
    lease.rows_ = std::move(payload);
    lease.handle_ = handle;
    lease.owner_ = state_;
    return lease;
}

EmbedLease EmbedInterner::find(const EmbedDigest& digest) const {
    auto& s = *state_;
    EmbedLease lease;
    const std::lock_guard lock(s.mutex);
    const auto it = s.by_digest.find(digest);
    if (it == s.by_digest.end() || !s.entries[it->second].rows) return lease;
    auto& e = s.entries[it->second];
    ++e.refs;
    lease.rows_ = e.rows;
    lease.handle_ = it->second;
    lease.owner_ = state_;
    return lease;
}

std::uint64_t EmbedInterner::epoch() const noexcept {
    const std::lock_guard lock(state_->mutex);
    return state_->status.epoch;
}
EmbedInternerStatus EmbedInterner::status() const noexcept {
    const std::lock_guard lock(state_->mutex);
    return state_->status;
}

EmbedTable::EmbedTable(std::vector<EmbedLease> leases) : leases_(std::move(leases)) {
    for (const auto& lease : leases_)
        if (!lease) throw std::invalid_argument("embed table: empty lease");
    std::sort(leases_.begin(), leases_.end(),
              [](const EmbedLease& a, const EmbedLease& b) { return a.handle() < b.handle(); });
    leases_.erase(std::unique(leases_.begin(), leases_.end(),
                              [](const EmbedLease& a, const EmbedLease& b) { return a.handle() == b.handle(); }),
                  leases_.end());
}

const EmbedRows* EmbedTable::find(std::uint32_t handle) const noexcept {
    const auto it = std::lower_bound(leases_.begin(), leases_.end(), handle,
                                     [](const EmbedLease& a, std::uint32_t h) { return a.handle() < h; });
    return it != leases_.end() && it->handle() == handle ? &it->rows() : nullptr;
}

void EmbedTable::validate(std::span<const std::uint32_t> ids) const {
    const EmbedRows* rows = nullptr;
    std::uint32_t handle = UINT32_MAX;
    for (const auto id : ids) {
        if (!is_embed_id(id)) continue;
        if (embed_handle(id) != handle) {
            handle = embed_handle(id);
            rows = find(handle);
            if (!rows) throw std::invalid_argument("embedding row id names a handle outside the request");
        }
        if (embed_row(id) >= rows->rows) throw std::invalid_argument("embedding row id is outside its span");
    }
}

std::vector<EmbedManifestEntry> EmbedTable::manifest() const {
    std::vector<EmbedManifestEntry> out;
    out.reserve(leases_.size());
    for (const auto& lease : leases_)
        out.push_back({lease.handle(), lease.rows().rows, lease.rows().width, lease.rows().digest});
    return out;
}

std::vector<std::uint32_t> remap_embed_ids(std::span<const std::uint32_t> ids,
                                           std::span<const EmbedManifestEntry> remote, const EmbedTable& local) {
    std::map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> to; // remote handle -> local handle, rows
    for (const auto& entry : remote) {
        const EmbedLease* match = nullptr;
        for (const auto& lease : local.leases())
            if (lease.rows().digest == entry.digest) match = &lease;
        if (!match || match->rows().rows != entry.rows || match->rows().width != entry.width)
            throw std::invalid_argument("embed manifest entry has no local span of the same shape");
        to[entry.handle] = {match->handle(), entry.rows};
    }
    std::vector<std::uint32_t> out(ids.begin(), ids.end());
    for (auto& id : out) {
        if (!is_embed_id(id)) continue;
        const auto it = to.find(embed_handle(id));
        if (it == to.end() || embed_row(id) >= it->second.second)
            throw std::invalid_argument("embedding row id is not described by the manifest");
        id = embed_id(it->second.first, embed_row(id));
    }
    return out;
}

std::size_t ExpandedPrompt::map_offset(std::size_t source_offset) const noexcept {
    std::size_t out = source_offset;
    for (const auto& run : runs) {
        if (run.source >= source_offset) break;
        out += run.size - 1;
    }
    return out;
}

ExpandedPrompt expand_embed_rows(std::span<const std::uint32_t> tokens, std::uint32_t placeholder,
                                 std::span<const std::vector<std::uint32_t>> layouts) {
    if (is_embed_id(placeholder)) throw std::invalid_argument("embed expand: placeholder is a virtual id");
    std::size_t total = tokens.size(), found = 0;
    for (const auto id : tokens) {
        if (is_embed_id(id)) throw std::invalid_argument("embed expand: rendered tokens hold a virtual id");
        found += id == placeholder;
    }
    if (found != layouts.size())
        throw std::invalid_argument("embed expand: placeholders and spans differ in number");
    for (const auto& layout : layouts) {
        if (layout.empty()) throw std::invalid_argument("embed expand: empty span layout");
        if (std::find(layout.begin(), layout.end(), placeholder) != layout.end())
            throw std::invalid_argument("embed expand: span layout holds the placeholder");
        total += layout.size() - 1;
    }
    ExpandedPrompt out;
    out.tokens.reserve(total);
    out.runs.reserve(found);
    std::size_t next = 0;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] != placeholder) {
            out.tokens.push_back(tokens[i]);
            continue;
        }
        const auto& layout = layouts[next++];
        out.runs.push_back({i, out.tokens.size(), layout.size(), count_embed_ids(layout)});
        out.tokens.insert(out.tokens.end(), layout.begin(), layout.end());
    }
    return out;
}

} // namespace strix::session
