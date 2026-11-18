#pragma once
// Interleaved virtual stages over the two-node seam (shared by every engine's pair route).
// A StageMap cuts the layers into 2v stages; stage s runs on rank s % 2, so rank 0 owns stages 0, 2, ... and
// rank 1 owns 1, 3, ... (v = 1 is today's two-stage cut). Boundary b is the output of stage b: even boundaries
// go forward (rank 0 -> 1), odd ones back. StagePipes runs one two-slot pair::Pipe per direction, each over
// its own ordered transport (the link is full duplex; a shared byte stream would interleave the two pipes'
// frames and ACKs). Frame order per direction and every rank's execution order come from regime.hpp
// (stage_sequence / stage_group_chunks): groups of at most 4 chunks, depth-first inside a group, which keeps
// both directions live with two slots.
// Header words owned here at wire v4 (stages > 2 only): [43] boundary, [44] StageMap FNV-1a. At v = 1 the
// frames stay wire v3, byte-identical to a plain Pipe (the engine's words and [1]/[41] keep their values).
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "strix/pair/pipe.hpp"
#include "strix/pair/regime.hpp"

namespace strix::pair {
struct StageMap {                   // 2v stages; stage s on rank s % 2 runs layers [cuts[s], cuts[s+1])
    std::vector<StageCut> cuts;     // 2v + 1 entries, front 0, back = layers (half cuts allowed inside)
    std::uint32_t stages() const noexcept { return cuts.empty() ? 0U : static_cast<std::uint32_t>(cuts.size() - 1U); }
    std::uint32_t virtual_stages() const noexcept { return stages() / 2U; }
    std::uint32_t layers() const noexcept { return cuts.empty() ? 0U : cuts.back().layer; }
    unsigned owner(std::uint32_t stage) const noexcept { return stage % 2U; }
    // Stage that owns the sequence state of `layer` (a half cut leaves the cut layer's state with the earlier stage).
    std::uint32_t stage_of(std::uint32_t layer) const {
        for (std::uint32_t s = 0; s < stages(); ++s)
            if (layer < cuts[s + 1U].state_begin()) return s;
        throw std::out_of_range("pair stage map: layer outside the model");
    }
    bool owns(unsigned rank, std::uint32_t layer) const { return owner(stage_of(layer)) == rank; }
    std::string text() const {
        std::string out;
        for (const auto& cut : cuts) out += (out.empty() ? "" : ",") + cut.text();
        return out;
    }
    std::uint64_t fnv() const noexcept {
        std::uint64_t hash = 1469598103934665603ULL;
        for (const char ch : text()) hash = (hash ^ static_cast<unsigned char>(ch)) * 1099511628211ULL;
        return hash;
    }
    // "0,8,15,23,30,38,45,53,60": even stage count, front 0, back = layers, strictly ascending.
    static StageMap parse(std::string_view text, std::uint32_t layers) {
        StageMap map;
        while (!text.empty()) {
            const auto comma = text.find(',');
            map.cuts.push_back(StageCut::parse(text.substr(0, comma)));
            text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1U);
        }
        map.validate(layers);
        return map;
    }
    static StageMap single(StageCut cut, std::uint32_t layers) {   // v = 1, today's [0,cut) | [cut,L)
        StageMap map{{StageCut(0U), cut, StageCut(layers)}};
        map.validate(layers);
        return map;
    }
    void validate(std::uint32_t layers) const {
        if (cuts.size() < 3U || cuts.size() % 2U == 0U) throw std::invalid_argument("pair stage map needs 2v + 1 cuts");
        if (cuts.front() != StageCut(0U) || cuts.back() != StageCut(layers)) throw std::invalid_argument("pair stage map must span 0..layers");
        for (std::size_t i = 1; i < cuts.size(); ++i)
            if (cuts[i].units() <= cuts[i - 1U].units()) throw std::invalid_argument("pair stage map cuts must ascend");
    }
};

// Whole-layer cuts minimising the slowest of the 2v stages (`tail_ms` lands on the last one, e.g. the head);
// `legal(layer)` is the engine's cut rule. Ties keep the earliest cuts.
inline StageMap balance_map(std::span<const double> layer_ms, double tail_ms, std::uint32_t v,
    const std::function<bool(std::uint32_t)>& legal) {
    const auto layers = static_cast<std::uint32_t>(layer_ms.size());
    const std::uint32_t stages = 2U * v;
    if (v == 0U || stages > layers) throw std::invalid_argument("pair stage map: more stages than layers");
    std::vector<double> prefix(layers + 1U, 0.0);
    for (std::uint32_t l = 0; l < layers; ++l) prefix[l + 1U] = prefix[l] + layer_ms[l];
    const auto stage_ms = [&](std::uint32_t a, std::uint32_t b) { return prefix[b] - prefix[a] + (b == layers ? tail_ms : 0.0); };
    constexpr double inf = std::numeric_limits<double>::infinity();
    // best[k][e]: slowest stage of the first k stages ending at layer e; from[k][e]: start of stage k - 1.
    std::vector<std::vector<double>> best(stages + 1U, std::vector<double>(layers + 1U, inf));
    std::vector<std::vector<std::uint32_t>> from(stages + 1U, std::vector<std::uint32_t>(layers + 1U, 0U));
    best[0][0] = 0.0;
    for (std::uint32_t k = 1; k <= stages; ++k)
        for (std::uint32_t e = k; e <= layers; ++e) {
            if (e != layers && (k == stages || !legal(e))) continue;
            for (std::uint32_t a = k - 1U; a < e; ++a) {
                if (best[k - 1U][a] == inf) continue;
                const double worst = std::max(best[k - 1U][a], stage_ms(a, e));
                if (worst < best[k][e]) { best[k][e] = worst; from[k][e] = a; }
            }
        }
    if (best[stages][layers] == inf) throw std::invalid_argument("pair stage map: no legal cuts");
    StageMap map;
    map.cuts.assign(stages + 1U, StageCut(0U));
    for (std::uint32_t k = stages, e = layers; k > 0U; e = from[k][e], --k) map.cuts[k] = StageCut(e);
    map.validate(layers);
    return map;
}

// One direction of the pair link: ordered send/receive of that direction's frames and their ACKs.
struct PipeLink {
    Pipe::Send send;
    Pipe::Receive receive;
    std::function<void()> abort;
    Pipe::TransferPiece transfer_piece = {};
};
struct StageStep { std::uint32_t stage = 0; std::uint64_t chunk = 0; };

class StagePipes {
public:
    // `forward` carries even boundaries (rank 0 sends), `reverse` odd ones (rank 1 sends). At v = 1 the reverse
    // link is never used. options.validate is the engine's validator; StagePipes checks [42]-[44] first.
    StagePipes(unsigned rank, StageMap map, std::uint64_t chunks, PipeLink forward, PipeLink reverse, PipeOptions options,
        std::function<std::shared_ptr<PipeStaging>()> staging = {}, Pipe::Checksum checksum = {})
        : rank_(rank), map_(std::move(map)), chunks_(chunks) {
        if (rank_ > 1U || chunks_ == 0U || map_.stages() < 2U || map_.stages() % 2U != 0U)
            throw std::invalid_argument("pair stage pipes need rank 0/1, chunks and an even stage map");
        const std::uint32_t stages = map_.stages();
        for (std::uint64_t base = 0; base < chunks_; base += stage_group_chunks)
            for (std::uint32_t s = 0; s < stages; ++s)
                for (std::uint64_t c = base; c < std::min<std::uint64_t>(chunks_, base + stage_group_chunks); ++c) {
                    if (map_.owner(s) == rank_) order_.push_back({s, c});
                    if (s + 1U < stages) boundary_[s % 2U].push_back(s);  // production order of direction s % 2
                }
        options.wire_version = wire_version();
        const unsigned out_dir = rank_, in_dir = 1U - rank_;
        auto& out_link = out_dir == 0U ? forward : reverse;
        auto& in_link = in_dir == 0U ? forward : reverse;
        if (count(out_dir) != 0U)
            out_ = std::make_unique<Pipe>(0U, count(out_dir), out_link.send, out_link.receive, out_link.abort, options, staging,
                checksum, out_link.transfer_piece);
        if (count(in_dir) != 0U) {
            auto in_options = options;
            in_options.validate = [this, engine = options.validate, in_dir](const PipeHeader& header, std::uint64_t sequence) {
                if (header[42] != wire_version()) throw std::runtime_error("pair stage seam wire version mismatch");
                if (wire_version() == 4U &&
                    (header[43] != boundary_[in_dir].at(static_cast<std::size_t>(sequence)) || header[44] != fnv_))
                    throw std::runtime_error("pair stage seam boundary or stage map mismatch");
                return engine(header, sequence);
            };
            in_ = std::make_unique<Pipe>(1U, count(in_dir), in_link.send, in_link.receive, in_link.abort, std::move(in_options),
                staging, checksum, in_link.transfer_piece);
        }
    }
    // Receive workers capture this wrapper; its address must stay stable until they stop.
    StagePipes(StagePipes&&) = delete;
    StagePipes& operator=(StagePipes&&) = delete;

    const StageMap& map() const noexcept { return map_; }
    std::uint64_t chunks() const noexcept { return chunks_; }
    std::uint64_t wire_version() const noexcept { return map_.stages() > 2U ? 4U : 3U; }
    // This rank's execution order: (stage, chunk), groups of stage_group_chunks, depth-first inside a group.
    const std::vector<StageStep>& order() const noexcept { return order_; }
    // Frames of direction d (0 forward: v per chunk; 1 reverse: v - 1 per chunk).
    std::uint64_t count(unsigned direction) const noexcept { return boundary_[direction].size(); }
    std::uint64_t sequence(std::uint64_t chunk, std::uint32_t boundary) const {
        if (chunk >= chunks_ || boundary + 1U >= map_.stages()) throw std::out_of_range("pair stage seam: no such boundary");
        return stage_sequence(chunk, boundary, chunks_, map_.virtual_stages());
    }
    // Producer (owner of stage `boundary`): the pipe-owned words are set; the engine adds its own and releases.
    PipeFrame& acquire_out(std::uint64_t chunk, std::uint32_t boundary) {
        const auto q = checked(chunk, boundary, rank_, next_out_);
        auto& frame = out_->acquire(q);
        frame.header[1] = q;
        frame.header[41] = count(rank_);
        if (wire_version() == 4U) { frame.header[43] = boundary; frame.header[44] = fnv_; }
        return frame;
    }
    void release_out(std::uint64_t chunk, std::uint32_t boundary) { out_->release(advance(chunk, boundary, rank_, next_out_)); }
    // Consumer (owner of stage boundary + 1).
    PipeFrame& acquire_in(std::uint64_t chunk, std::uint32_t boundary) { return in_->acquire(checked(chunk, boundary, 1U - rank_, next_in_)); }
    void release_in(std::uint64_t chunk, std::uint32_t boundary) { in_->release(advance(chunk, boundary, 1U - rank_, next_in_)); }
    // Every frame of both directions was consumed (both transports may then carry the hand-off).
    void finish() {
        if (out_) out_->finish(count(rank_));
        if (in_) in_->finish(count(1U - rank_));
    }
private:
    std::uint64_t checked(std::uint64_t chunk, std::uint32_t boundary, unsigned producer, std::uint64_t next) const {
        if (map_.owner(boundary) != producer) throw std::logic_error("pair stage seam: boundary of the other direction");
        const auto q = sequence(chunk, boundary);
        if (q != next) throw std::logic_error("pair stage seam: frame out of the shared stage order");
        return q;
    }
    std::uint64_t advance(std::uint64_t chunk, std::uint32_t boundary, unsigned producer, std::uint64_t& next) {
        const auto q = checked(chunk, boundary, producer, next);
        ++next;
        return q;
    }
    unsigned rank_;
    StageMap map_;
    std::uint64_t chunks_;
    std::uint64_t fnv_ = map_.fnv();
    std::vector<StageStep> order_;
    std::vector<std::uint32_t> boundary_[2];
    std::uint64_t next_out_ = 0, next_in_ = 0;
    std::unique_ptr<Pipe> out_, in_;  // declared last: destroyed (workers stopped) before the state above
};
} // namespace strix::pair
