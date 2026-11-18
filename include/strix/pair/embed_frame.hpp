#pragma once

// pair::EmbedFrame: the embedding rows of a prompt's media spans, from the rank that encoded them to its peer
// (TP2 pair, wave4 vision spec 2e). Handles are process-local, so a frame carries the manifest
// (session::EmbedManifestEntry: handle, rows, width, digest) plus, for each span the peer does not hold yet, its rows:
// BF16 data, roles, positions and the vocabulary token, each payload with its own CRC-32C. The receiver re-interns
// the rows in its own session::EmbedInterner and remaps the prompt's virtual ids to its handles
// (session::remap_embed_ids), so both ranks resolve the same rows for the same positions.
//
// Sent once per new digest: both ends keep an EmbedFrameLedger of what the peer holds, updated in manifest order by
// the same rule (LRU of a fixed number of spans), so the sender knows without a round trip which payloads to omit.
// A frame naming a digest the receiver no longer holds is a protocol error (the ledgers diverged).
// Encoding, receiving and transport failures are terminal for the pair's frame session: updates may already
// have touched either ledger. Do not retry with those objects; restart both ends with fresh ledgers/receivers.
//
// Wire (host byte order; both ranks are the same architecture, as for every pair frame):
//   header   u32 magic 'EMBF', u32 version 1, u32 spans, u32 manifest CRC-32C (of the entries)
//   entry    u32 handle, rows, width, real_token_id, axes, flags (bit 0: payload follows), u32 payload CRC-32C,
//            u32 reserved, u8 digest[32]                                                            (64 bytes)
//   payload  in entry order, for every entry with bit 0: role[rows] (u8), pad to 4, lm_pos[rows * axes] (i32),
//            data[rows * width] (u16 BF16)
// Deep rows (EmbedRows::deep) are not carried by version 1: encode refuses them.
//
// Header only. The checksum function is the caller's (dist::crc32c on the pair) so this header links nothing.

#include "strix/session/embed_rows.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::pair {

using EmbedFrameCrc = std::uint32_t (*)(std::span<const std::byte>) noexcept;

inline constexpr std::uint32_t kEmbedFrameMagic = 0x46424D45U;  // "EMBF"
inline constexpr std::uint32_t kEmbedFrameVersion = 1U;
inline constexpr std::size_t kEmbedFrameHeaderBytes = 16U, kEmbedFrameEntryBytes = 64U;
inline constexpr std::uint32_t kEmbedFrameLedgerSpans = 64U;  // spans a peer keeps (default of both ends)

// Transfer an already agreed frame length in ordered, bounded exchanges. Exactly one of out/in carries
// the frame: the sender discards replies, the receiver sends zeros. Both peers use the same max_bytes
// and callback tags; their transport's exchange sequence preserves ordering within that tag.
template <class Exchange>
void transfer_embed_frame_bytes(std::span<const std::byte> out, std::span<std::byte> in,
                               std::size_t max_bytes, Exchange&& exchange) {
    if (max_bytes == 0U || (!out.empty() && !in.empty()))
        throw std::invalid_argument("embed frame transfer: invalid limit or direction");
    const bool receiving = !in.empty();
    const std::size_t n = receiving ? in.size() : out.size();
    std::vector<std::byte> scratch(std::min(max_bytes, n), std::byte{0});
    for (std::size_t at = 0; at < n;) {
        const std::size_t take = std::min(max_bytes, n - at);
        if (receiving) exchange(std::span<const std::byte>(scratch).first(take), in.subspan(at, take));
        else exchange(out.subspan(at, take), std::span<std::byte>(scratch).first(take));
        at += take;
    }
}

// What the peer holds, by digest, most recent first. Both ends apply touch() to the manifest entries in order;
// touch returns the digest it evicted (if any) so the receiver drops that payload.
class EmbedFrameLedger {
public:
    explicit EmbedFrameLedger(std::uint32_t capacity = kEmbedFrameLedgerSpans) : capacity_(capacity) {
        if (capacity_ == 0U) throw std::invalid_argument("embed frame ledger: capacity 0");
    }
    [[nodiscard]] bool holds(const session::EmbedDigest& d) const {
        return std::find(lru_.begin(), lru_.end(), d) != lru_.end();
    }
    // Marks d as held and most recent; returns true and sets *evicted when another digest fell out.
    bool touch(const session::EmbedDigest& d, session::EmbedDigest* evicted) {
        if (const auto it = std::find(lru_.begin(), lru_.end(), d); it != lru_.end()) lru_.erase(it);
        lru_.push_front(d);
        if (lru_.size() <= capacity_) return false;
        if (evicted) *evicted = lru_.back();
        lru_.pop_back();
        return true;
    }
    void clear() noexcept { lru_.clear(); }
    [[nodiscard]] std::size_t size() const noexcept { return lru_.size(); }

private:
    std::uint32_t capacity_;
    std::list<session::EmbedDigest> lru_;
};

namespace embed_frame_detail {
inline void put32(std::vector<std::byte>& out, std::uint32_t v) {
    const auto at = out.size();
    out.resize(at + 4U);
    std::memcpy(out.data() + at, &v, 4U);
}
inline std::uint32_t get32(std::span<const std::byte> in, std::size_t at) {
    if (at + 4U > in.size()) throw std::runtime_error("embed frame: truncated");
    std::uint32_t v = 0;
    std::memcpy(&v, in.data() + at, 4U);
    return v;
}
inline std::size_t payload_bytes(std::uint32_t rows, std::uint32_t width, std::uint32_t axes) {
    const std::size_t roles = (std::size_t{rows} + 3U) & ~std::size_t{3};
    return roles + std::size_t{rows} * axes * 4U + std::size_t{rows} * width * 2U;
}
}  // namespace embed_frame_detail

// Handles of the virtual ids in ids, in first-appearance order (each once).
[[nodiscard]] inline std::vector<std::uint32_t> embed_frame_handles(std::span<const std::uint32_t> ids) {
    std::vector<std::uint32_t> out;
    for (const auto id : ids)
        if (session::is_embed_id(id) && std::find(out.begin(), out.end(), session::embed_handle(id)) == out.end())
            out.push_back(session::embed_handle(id));
    return out;
}

// The frame of the spans that ids reference (table must hold them all; std::invalid_argument otherwise). Payloads
// are written for the digests `peer` does not hold; `peer` is updated as the receiver will update its own.
// An ids span without a virtual id gives an empty frame (no bytes).
[[nodiscard]] inline std::vector<std::byte> encode_embed_frame(const session::EmbedTable& table,
                                                               std::span<const std::uint32_t> ids,
                                                               EmbedFrameLedger& peer, EmbedFrameCrc crc) {
    namespace d = embed_frame_detail;
    const auto handles = embed_frame_handles(ids);
    if (handles.empty()) return {};
    if (crc == nullptr) throw std::invalid_argument("embed frame: no checksum function");
    table.validate(ids);
    std::vector<const session::EmbedRows*> spans;
    for (const auto h : handles) {
        const auto* rows = table.find(h);
        if (rows == nullptr) throw std::invalid_argument("embed frame: a handle outside the table");
        if (!rows->deep.empty()) throw std::invalid_argument("embed frame v1 carries no deep rows");
        if (rows->lm_pos.size() != std::size_t{rows->rows} * rows->axes || rows->data.size() != std::size_t{rows->rows} * rows->width)
            throw std::invalid_argument("embed frame: malformed span");
        spans.push_back(rows);
    }
    std::vector<std::byte> entries, payload;
    for (std::size_t i = 0; i < spans.size(); ++i) {
        const auto& s = *spans[i];
        const bool send = !peer.holds(s.digest);
        std::uint32_t pcrc = 0;
        if (send) {
            const auto at = payload.size();
            payload.resize(at + d::payload_bytes(s.rows, s.width, s.axes), std::byte{0});
            auto* p = payload.data() + at;
            for (std::uint32_t r = 0; r < s.rows; ++r)
                p[r] = std::byte{static_cast<std::uint8_t>(s.role.empty() ? 0U : static_cast<unsigned>(s.role[r]))};
            p += (std::size_t{s.rows} + 3U) & ~std::size_t{3};
            if (!s.lm_pos.empty()) std::memcpy(p, s.lm_pos.data(), s.lm_pos.size() * 4U);
            p += s.lm_pos.size() * 4U;
            std::memcpy(p, s.data.data(), s.data.size() * 2U);
            pcrc = crc(std::span<const std::byte>(payload.data() + at, payload.size() - at));
        }
        peer.touch(s.digest, nullptr);
        d::put32(entries, handles[i]);
        d::put32(entries, s.rows);
        d::put32(entries, s.width);
        d::put32(entries, s.real_token_id);
        d::put32(entries, s.axes);
        d::put32(entries, send ? 1U : 0U);
        d::put32(entries, pcrc);
        d::put32(entries, 0U);
        const auto at = entries.size();
        entries.resize(at + 32U);
        std::memcpy(entries.data() + at, s.digest.data(), 32U);
    }
    std::vector<std::byte> out;
    out.reserve(kEmbedFrameHeaderBytes + entries.size() + payload.size());
    d::put32(out, kEmbedFrameMagic);
    d::put32(out, kEmbedFrameVersion);
    d::put32(out, static_cast<std::uint32_t>(spans.size()));
    d::put32(out, crc(entries));
    out.insert(out.end(), entries.begin(), entries.end());
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

// The receiving end: its own interner, the payloads it holds (by digest) and the mirror of the sender's ledger.
class EmbedFrameReceiver {
public:
    explicit EmbedFrameReceiver(std::uint32_t capacity = kEmbedFrameLedgerSpans) : ledger_(capacity) {}

    struct Received {
        std::shared_ptr<const session::EmbedTable> table;  // this process's handles
        std::vector<std::uint32_t> ids;                    // the prompt with its virtual ids remapped
        std::uint64_t payload_bytes = 0;                   // rows that travelled in this frame
        std::uint32_t spans = 0, sent = 0;                 // spans named, spans whose rows travelled
    };
    // Verifies and interns the frame, then remaps ids (the sender's prompt). Throws std::runtime_error on a
    // truncated or corrupt frame, a checksum mismatch or a digest neither in the frame nor held.
    [[nodiscard]] Received receive(std::span<const std::byte> frame, std::span<const std::uint32_t> ids, EmbedFrameCrc crc) {
        namespace d = embed_frame_detail;
        if (crc == nullptr) throw std::invalid_argument("embed frame: no checksum function");
        if (d::get32(frame, 0) != kEmbedFrameMagic || d::get32(frame, 4) != kEmbedFrameVersion)
            throw std::runtime_error("embed frame: bad magic or version");
        const std::uint32_t n = d::get32(frame, 8);
        const std::size_t entries_end = kEmbedFrameHeaderBytes + std::size_t{n} * kEmbedFrameEntryBytes;
        if (n == 0U || n > session::kEmbedMaxHandles || entries_end > frame.size()) throw std::runtime_error("embed frame: truncated manifest");
        if (crc(frame.subspan(kEmbedFrameHeaderBytes, entries_end - kEmbedFrameHeaderBytes)) != d::get32(frame, 12))
            throw std::runtime_error("embed frame: manifest checksum");
        Received out;
        out.spans = n;
        std::vector<session::EmbedManifestEntry> remote(n);
        std::vector<session::EmbedLease> leases;
        std::size_t at = entries_end;
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::size_t e = kEmbedFrameHeaderBytes + std::size_t{i} * kEmbedFrameEntryBytes;
            auto& m = remote[i];
            m.handle = d::get32(frame, e);
            m.rows = d::get32(frame, e + 4U);
            m.width = d::get32(frame, e + 8U);
            const std::uint32_t real = d::get32(frame, e + 12U), axes = d::get32(frame, e + 16U), flags = d::get32(frame, e + 20U),
                                pcrc = d::get32(frame, e + 24U);
            std::memcpy(m.digest.data(), frame.data() + e + 32U, 32U);
            if (m.rows == 0U || m.rows > session::kEmbedMaxRows || m.width == 0U || axes > 3U || (flags & ~1U) != 0U)
                throw std::runtime_error("embed frame: malformed entry");
            session::EmbedLease lease;
            if ((flags & 1U) != 0U) {
                const std::size_t bytes = d::payload_bytes(m.rows, m.width, axes);
                if (at + bytes > frame.size()) throw std::runtime_error("embed frame: truncated payload");
                const auto p = frame.subspan(at, bytes);
                if (crc(p) != pcrc) throw std::runtime_error("embed frame: payload checksum");
                session::EmbedRows rows;
                rows.digest = m.digest;
                rows.rows = m.rows;
                rows.width = m.width;
                rows.axes = axes;
                rows.real_token_id = real;
                rows.role.resize(m.rows);
                for (std::uint32_t r = 0; r < m.rows; ++r) {
                    const auto v = static_cast<std::uint8_t>(p[r]);
                    if (v > 3U) throw std::runtime_error("embed frame: bad role");
                    rows.role[r] = static_cast<session::EmbedRole>(v);
                }
                std::size_t q = (std::size_t{m.rows} + 3U) & ~std::size_t{3};
                rows.lm_pos.resize(std::size_t{m.rows} * axes);
                if (!rows.lm_pos.empty()) std::memcpy(rows.lm_pos.data(), p.data() + q, rows.lm_pos.size() * 4U);
                q += rows.lm_pos.size() * 4U;
                rows.data.resize(std::size_t{m.rows} * m.width);
                std::memcpy(rows.data.data(), p.data() + q, rows.data.size() * 2U);
                lease = interner_.intern(std::move(rows));
                held_[m.digest] = lease;
                at += bytes;
                out.payload_bytes += bytes;
                ++out.sent;
            } else {
                const auto it = held_.find(m.digest);
                if (it == held_.end()) throw std::runtime_error("embed frame: a span this rank does not hold (ledgers diverged)");
                lease = it->second;
                if (lease.rows().rows != m.rows || lease.rows().width != m.width) throw std::runtime_error("embed frame: held span of another shape");
            }
            session::EmbedDigest evicted{};
            if (ledger_.touch(m.digest, &evicted)) held_.erase(evicted);
            if (std::none_of(leases.begin(), leases.end(), [&](const session::EmbedLease& l) { return l.handle() == lease.handle(); }))
                leases.push_back(std::move(lease));
        }
        if (at != frame.size()) throw std::runtime_error("embed frame: trailing bytes");
        auto table = std::make_shared<const session::EmbedTable>(std::move(leases));
        out.ids = session::remap_embed_ids(ids, remote, *table);
        out.table = std::move(table);
        return out;
    }
    [[nodiscard]] std::size_t held() const noexcept { return held_.size(); }
    [[nodiscard]] std::uint64_t epoch() const noexcept { return interner_.epoch(); }

private:
    session::EmbedInterner interner_;
    EmbedFrameLedger ledger_;
    std::map<session::EmbedDigest, session::EmbedLease> held_;
};

// Experimental policy only: serving executes rank0 and does not call this selector.
// Peer/split have no execution path; measured rows alone do not qualify them for serving.
// Where a pair encodes a request's images (spec 2e). rank0: the encoder on rank 0, rows framed to rank 1 (default).
// peer: rank 1 encodes while rank 0 prefills the text before the first image. split: images alternate between ranks.
enum class EmbedEncodeMode : std::uint8_t { rank0 = 0, peer = 1, split = 2 };
[[nodiscard]] constexpr const char* embed_encode_mode_name(EmbedEncodeMode m) noexcept {
    return m == EmbedEncodeMode::peer ? "peer" : m == EmbedEncodeMode::split ? "split" : "rank0";
}
// One measured row of the regime table (T4 receipts). A mode without a row is never selected.
struct EmbedRegimeRow {
    EmbedEncodeMode mode = EmbedEncodeMode::rank0;
    double prefill_ms_per_row = 0.0;  // LM prefill cost of a text row on rank 0 (the overlap budget of peer)
    double encode_ms = 0.0;           // encode time of one image of the request's size class
};
struct EmbedRegimeInput {
    std::uint32_t images = 0;
    std::uint32_t prefix_rows = 0;  // text rows before the first image (system prompt + tools)
};
// The rule of the spec: rank0 for one image with a short prefix; peer when prefix_rows x per-row prefill time >= the
// encode time; split for >= 2 images. A mode is chosen only when the table holds a measured row for it, so an empty
// table (no receipts yet) always gives rank0.
[[nodiscard]] inline EmbedEncodeMode select_embed_encode_mode(const EmbedRegimeInput& in, std::span<const EmbedRegimeRow> table) {
    const auto row = [&](EmbedEncodeMode m) -> const EmbedRegimeRow* {
        for (const auto& r : table)
            if (r.mode == m) return &r;
        return nullptr;
    };
    if (in.images >= 2U && row(EmbedEncodeMode::split) != nullptr) return EmbedEncodeMode::split;
    if (const auto* p = row(EmbedEncodeMode::peer); p != nullptr && in.images >= 1U && p->encode_ms > 0.0 &&
                                                    double(in.prefix_rows) * p->prefill_ms_per_row >= p->encode_ms)
        return EmbedEncodeMode::peer;
    return EmbedEncodeMode::rank0;
}

}  // namespace strix::pair
