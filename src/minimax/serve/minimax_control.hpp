#pragma once

#include "strix/minimax/prefill_types.hpp"
#include "strix/pair/lockstep.hpp"
#include <array>
#include <cstddef>
#include <span>
#include <cstring>
#include <cstddef>
#include <type_traits>
#include <stdexcept>

namespace strix::minimax::serve {
struct Control {
    std::uint32_t magic = 0x43534d4dU;  // "MMSC"
    std::uint32_t version = 3;
    std::uint32_t command = 0;
    std::uint64_t task_id = 0, parent_task_id = 0, slot_generation = 0;
    strix::minimax::engine::PfProgress active{}, parked{};
    std::uint32_t rows = 0;             // step: rows used in row[]
    std::uint32_t pslot = 0, plen = 0;  // prefill: slot and (suffix) token count sent after this control
    std::uint32_t pflags = 0, bank = 0;
    std::uint64_t round = 0, ack_round = ~std::uint64_t{0};
    strix::minimax::engine::Row row[8]{};
    std::uint32_t layers = 0;  // unit
    std::uint32_t append = 0;  // step
    std::uint32_t dslots = 0;  // draft
    std::uint32_t dslot[8]{}, danchor[8]{};
    std::uint64_t dpos[8]{};
    // prefill: first position; a reused prefix of the slot's cache precedes it (0 = the whole prompt). pstart and rank
    // land in the 512-byte frame's former zero padding: rank 0 with a zero pstart sends the same bytes as before.
    std::uint32_t pstart = 0;
    // The sender's pair rank. With round / ack_round it forms the strix::pair::LockstepHeader every exchange checks; a
    // peer built before the field reports rank 0 on both ranks and is refused as "same rank" instead of misrunning.
    std::uint32_t rank = 0;
};
static_assert(sizeof(Control) <= 512);
static_assert(offsetof(Control, task_id) == 16 && offsetof(Control, active) == 40 &&
              offsetof(Control, parked) == 112 && offsetof(Control, round) == 208 &&
              offsetof(Control, ack_round) == 216 && offsetof(Control, row) == 224 &&
              offsetof(Control, dpos) == 432 && offsetof(Control, pstart) == 496 && sizeof(Control) == 504);

// Version 3 retains the native field offsets/widths, with little-endian scalar
// values. Struct padding is not a protocol field: never copy its indeterminate
// bytes. This also zeros the reserved tail of the fixed 512-byte packet.
inline std::array<std::byte, 512> control_wire(const Control& c) {
    std::array<std::byte, 512> out{};
    const auto put = [&out](std::size_t offset, auto value) {
        using T = decltype(value);
        std::uint64_t bits = static_cast<std::uint64_t>(value);
        for (std::size_t i = 0; i < sizeof(T); ++i) {
            out[offset + i] = static_cast<std::byte>(bits & 255U); bits >>= 8;
        }
    };
#define MM_CONTROL_FIELD(f) put(offsetof(Control, f), c.f)
    MM_CONTROL_FIELD(magic); MM_CONTROL_FIELD(version); MM_CONTROL_FIELD(command);
    MM_CONTROL_FIELD(task_id); MM_CONTROL_FIELD(parent_task_id); MM_CONTROL_FIELD(slot_generation);
    MM_CONTROL_FIELD(rows); MM_CONTROL_FIELD(pslot); MM_CONTROL_FIELD(plen);
    MM_CONTROL_FIELD(pflags); MM_CONTROL_FIELD(bank); MM_CONTROL_FIELD(round); MM_CONTROL_FIELD(ack_round);
    MM_CONTROL_FIELD(layers); MM_CONTROL_FIELD(append); MM_CONTROL_FIELD(dslots); MM_CONTROL_FIELD(pstart); MM_CONTROL_FIELD(rank);
#undef MM_CONTROL_FIELD
    const auto progress = [&](std::size_t base, const engine::PfProgress& p) {
#define MM_PROGRESS_FIELD(f) put(base + offsetof(engine::PfProgress, f), p.f)
        MM_PROGRESS_FIELD(id); MM_PROGRESS_FIELD(parent); MM_PROGRESS_FIELD(generation);
        MM_PROGRESS_FIELD(c0); MM_PROGRESS_FIELD(chunk); MM_PROGRESS_FIELD(rows);
        MM_PROGRESS_FIELD(slot); MM_PROGRESS_FIELD(bank); MM_PROGRESS_FIELD(layer);
        MM_PROGRESS_FIELD(next); MM_PROGRESS_FIELD(open);
#undef MM_PROGRESS_FIELD
    };
    progress(offsetof(Control, active), c.active); progress(offsetof(Control, parked), c.parked);
    for (std::size_t i = 0; i < 8; ++i) {
        const auto base = offsetof(Control, row) + i * sizeof(engine::Row);
        put(base + offsetof(engine::Row, slot), c.row[i].slot);
        put(base + offsetof(engine::Row, token), c.row[i].token);
        put(base + offsetof(engine::Row, pos), c.row[i].pos);
        put(offsetof(Control, dslot) + i * 4, c.dslot[i]);
        put(offsetof(Control, danchor) + i * 4, c.danchor[i]);
        put(offsetof(Control, dpos) + i * 8, c.dpos[i]);
    }
    return out;
}

// Additive explicit input control. Old commands, framing and startup are byte
// unchanged. Exact history/profile binding is established by paired capture;
// each row additionally carries the live store/generation and absolute cursor.
template<class Handles>
Control input_control(std::span<const engine::Row> rows,const Handles& handles) {
    if(rows.empty() || rows.size()>8 || rows.size()!=handles.size())
        throw std::invalid_argument("MiniMax input control shape");
    Control c;c.command=16;c.rows=static_cast<std::uint32_t>(rows.size());c.dslots=c.rows;
    c.parent_task_id=handles[0].store;
    for(std::size_t i=0;i<rows.size();++i) {
        if(!handles[i].valid() || handles[i].slot!=rows[i].slot || handles[i].store!=c.parent_task_id)
            throw std::invalid_argument("MiniMax input control binding");
        c.row[i]=rows[i];c.dpos[i]=handles[i].generation;
    }
    return c;
}

// Hello reuses pflags; zero retains the ordinary version-2 wire encoding.
inline constexpr std::uint32_t kHelloSessionVote = 1U;
// This rank asks for prefix reuse and replays a prefill that opens at pstart. Reuse needs both votes: a peer that
// does not set the bit (the switch off, or a build that predates pstart) keeps every prefill whole.
inline constexpr std::uint32_t kHelloPrefixVote = 2U;
// Conversation mode (needs the prefix vote): a prefill may also restart at a row inside a chunk, at the frontier a
// finished reply left in the slot. Both ranks must vote; one missing vote keeps chunk-aligned reuse (or none).
inline constexpr std::uint32_t kHelloConversationVote = 4U;

// Additive runner seam: old runners still execute identical cold commands.
template<class Runner>
void resident_begin(Runner& run, std::uint32_t slot, std::span<const std::uint32_t> prompt,
    bool context, std::uint64_t task, std::uint64_t parent, std::uint64_t generation, std::uint64_t resident) {
    if constexpr (requires { run.prefill_begin(slot,prompt,context,task,parent,generation,resident); })
        run.prefill_begin(slot,prompt,context,task,parent,generation,resident);
    else {
        if (resident) throw std::runtime_error("MiniMax runner has no resident prefill support");
        run.prefill_begin(slot,prompt,context,task,parent,generation);
    }
}
// Only nonzero suffix starts take this additive branch. Cold and all r4/r5
// commands keep their existing replay, ordering, checks and abort handling.
template<class Engine, class Runner, class Receive>
bool replay_resident_begin(const Control& c, Engine& eng, Runner& run, Receive receive, bool agreed, bool unaligned = false) try {
    if (!c.pstart) return false;
    if (!agreed || c.command != 2 || !c.task_id || c.parent_task_id || c.bank || c.pflags || !c.plen ||
        !c.slot_generation || c.pslot >= eng.slots() || !eng.prefill_reuse_rows() ||
        (!unaligned && c.pstart % eng.prefill_reuse_rows()) || c.pstart >= eng.capacity() || c.plen > eng.capacity()-c.pstart)
        throw std::runtime_error("control channel: invalid resident prefill");
    const auto prompt = receive(c.plen);
    if (prompt.size() != c.plen) throw std::runtime_error("control channel: resident suffix length");
    resident_begin(run,c.pslot,prompt,false,c.task_id,0,c.slot_generation,c.pstart);
    return true;
} catch (...) { eng.abort("control channel: resident replay failure"); throw; }


// The sender's pair rank; host doubles may expose it as a field.
template<class Engine>
std::uint32_t control_rank(const Engine& eng) {
    if constexpr (requires { eng.rank(); }) return static_cast<std::uint32_t>(eng.rank());
    else return static_cast<std::uint32_t>(eng.rank);
}

template<class Engine>
Control exchange(Engine& eng, const Control& out) try {
    std::array<std::byte, 512> o{}, i{};
    Control local = out;
    local.active = eng.prefill_progress();
    local.parked = eng.prefill_progress(true);
    local.rank = control_rank(eng);
    o = control_wire(local);
    eng.exchange_bytes(o.data(), i.data(), o.size());
    Control in;
    std::memcpy(&in, i.data(), sizeof(Control));
    if (in.magic != out.magic) {
        eng.abort("control channel: bad magic");
        throw std::runtime_error("control channel: bad magic from the peer");
    }
    if (in.version != local.version || in.active != local.active || in.parked != local.parked) {
        eng.abort("control channel: protocol / task checkpoint mismatch");
        throw std::runtime_error("control channel: protocol / task checkpoint mismatch");
    }
    try {
        strix::pair::check_pair({local.round, local.ack_round, local.rank}, {in.round, in.ack_round, in.rank});
    } catch (const strix::pair::ProtocolError& e) {
        eng.abort(e.what());
        throw;
    }
    return in;
} catch (...) {
    eng.abort("control channel: protocol failure");
    throw;
}

// One lockstep round: stamps out's round / ack_round from lock, exchanges, then checks the peer's header and advances.
template<class Engine>
Control exchange(Engine& eng, Control out, strix::pair::Lockstep& lock) {
    const auto header = lock.header();
    out.round = header.round;
    out.ack_round = header.ack_round;
    Control in = exchange(eng, out);
    lock.accept({in.round, in.ack_round, in.rank});
    return in;
}

// prefix: this rank's prefix-reuse vote; *prefix_agreed (when given) is true only if both ranks voted for it.
template<class Engine>
bool session_hello(Engine& eng, bool requested, bool prefix = false, bool* prefix_agreed = nullptr,
                   std::uint64_t prefix_rows = 1, bool conversation = false, bool* conversation_agreed = nullptr) {
    Control hello;
    hello.pflags = (requested ? kHelloSessionVote : 0U) | (prefix ? kHelloPrefixVote : 0U) |
                   (prefix && conversation ? kHelloConversationVote : 0U);
    hello.dpos[0] = prefix ? prefix_rows : 0U;  // OFF keeps the legacy frame; ON agrees group alignment too
    const auto peer = exchange(eng, hello);
    if (prefix_agreed != nullptr) *prefix_agreed = (hello.pflags & peer.pflags & kHelloPrefixVote) != 0 &&
        hello.dpos[0] != 0 && hello.dpos[0] == peer.dpos[0];
    if (conversation_agreed != nullptr)
        *conversation_agreed = prefix && (hello.pflags & peer.pflags & kHelloConversationVote) != 0 &&
            hello.dpos[0] != 0 && hello.dpos[0] == peer.dpos[0] && (hello.pflags & peer.pflags & kHelloPrefixVote) != 0;
    return ((hello.pflags | peer.pflags) & kHelloSessionVote) != 0;
}
}  // namespace strix::minimax::serve
