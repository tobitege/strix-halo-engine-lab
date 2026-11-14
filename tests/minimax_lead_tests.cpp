// Executes the production Lead loop with due-time arrivals and a control-channel sink.
#include "minimax/serve/minimax_lead.hpp"
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>

namespace ml = strix::minimax::serve::lead;
namespace ms = strix::minimax::serve;
namespace sv = strix::serve;
namespace mm = strix::minimax::engine;
struct FakeClock {
    using time_point = std::chrono::steady_clock::time_point;
    inline static time_point value{};
    inline static std::function<void(double)> advance;
    static time_point now() { return value; }
    static void sleep_for(std::chrono::milliseconds d) { advance(static_cast<double>(d.count())); }
};
struct FakeEngine {
    void* dd = nullptr;
    std::uint64_t control_round = 0;
    mm::PfProgress active{}, parked{};
    std::vector<ms::Control> controls;
    std::array<std::byte, 512> last_wire{};
    bool session_peer = false, prefix_peer = false, conversation_peer = false;
    std::uint64_t peer_alignment = 0;  // zero echoes; nonzero simulates a different peer configuration
    unsigned syncs = 0;
    unsigned quiesces = 0;
    bool taps = false;
    double parent_speed = 1;  // cold estimate may exceed actual service, as on native warm runs
    unsigned slots() const { return 8; }
    unsigned capacity() const { return 16384; }
    unsigned nested_rows() const { return 512; }
    std::uint64_t reuse_alignment = 4096;
    std::uint64_t prefill_reuse_rows() const { return reuse_alignment; }
    const char* prefill_unlimited_reason() const { return ""; }
    bool prefill_open() const { return active.open || parked.open; }
    bool prefill_active() const { return active.open; }
    mm::PfProgress prefill_progress(bool frozen = false) const { return frozen ? parked : active; }
    unsigned prefill_layer() const { return active.layer; }
    std::uint64_t prefill_unit_rows() const { return active.rows; }
    std::uint64_t prefill_rows_left() const { return active.rows; }
    void abort(const char* why) { throw std::runtime_error(why); }
    void set_taps(bool value) { if (prefill_open()) throw std::runtime_error("open-task taps"); taps = value; }
    unsigned rank() const { return 0; }
    unsigned peer(unsigned value) const { return value; }
    void send_prompt(std::vector<unsigned>&) {}
    void synchronize() { ++syncs; }
    void quiesce() {
        if (controls.empty() || controls.back().command != static_cast<unsigned>(ms::Command::heartbeat))
            throw std::runtime_error("quiesce outside paired idle heartbeat");
        ++quiesces;
    }
    void retire_contexts(unsigned) {}
    void exchange_bytes(const void* out, void* in, std::size_t bytes) {
        if (bytes != 512) { std::memcpy(in, out, bytes); return; }
        std::memcpy(last_wire.data(), out, bytes);
        ms::Control c;
        std::memcpy(&c, out, sizeof(c));
        controls.push_back(c);
        // Echo checkpoints through the real exchange validation; check replies carry the child's token.
        if (c.command == static_cast<unsigned>(ms::Command::check)) c.plen = active.next;
        c.rank = 1U - c.rank;  // the peer answers as the other rank for the same round
        if (session_peer) c.pflags |= ms::kHelloSessionVote;
        if (c.command == 0) c.pflags = (c.pflags & ~ms::kHelloPrefixVote) | (prefix_peer ? ms::kHelloPrefixVote : 0U);  // the peer's own vote
        if (c.command == 0) c.pflags = (c.pflags & ~ms::kHelloConversationVote) | (conversation_peer ? ms::kHelloConversationVote : 0U);
        if (c.command == 0 && peer_alignment != 0) c.dpos[0] = peer_alignment;
        std::memset(in, 0, bytes);
        std::memcpy(in, &c, sizeof(c));
    }
    void prefill_begin(unsigned slot, std::span<const unsigned> prompt, bool,
                       std::uint64_t id, std::uint64_t parent, std::uint64_t generation, std::uint64_t resident = 0) {
        if (active.open) throw std::runtime_error("already active");
        active = {};
        active.open = true; active.id = id; active.parent = parent; active.generation = generation;
        active.slot = slot; active.bank = parent != 0 ? 1U : 0U; active.rows = resident + prompt.size();
    }
    bool prefill_unit(unsigned layers, unsigned* next) {
        const auto count = std::min(layers, 60 - active.layer);
        FakeClock::advance(count * (10 + 0.1 * static_cast<double>(active.rows)) * (active.rows > 512 ? parent_speed : 1));
        active.layer += count;
        if (active.layer < 60) return false;
        active.open = false; active.next = 100 + active.slot; *next = active.next;
        return true;
    }
    void prefill_cancel(std::uint64_t id) {
        if (active.open && active.id == id) active.open = false;
        else if (parked.open && parked.id == id) parked.open = false;
        else throw std::runtime_error("cancel identity");
    }
    void prefill_park(std::uint64_t id) {
        if (!active.open || active.id != id || parked.open) throw std::runtime_error("park identity");
        parked = active; active = {};
    }
    void prefill_resume_parked(std::uint64_t id) {
        if (active.open || !parked.open || parked.id != id) throw std::runtime_error("resume identity");
        active = parked; parked = {};
    }
    ml::Drafts draft(unsigned, unsigned, std::uint64_t) { return {}; }
    unsigned pattern = 0;  // 1: the next token depends on the position: period 9, one deviation every 27 positions
    std::vector<unsigned> step(std::span<const mm::Row> rows, unsigned) {
        FakeClock::advance(50);
        std::vector<unsigned> out;
        for (const auto& r : rows)
            out.push_back(pattern == 0 ? 100 + r.slot : 100 + r.slot + static_cast<unsigned>(r.pos % 27 == 13 ? 7 : r.pos % 9 % 4));
        return out;
    }
};
struct Event { double at; std::function<void()> action; };
struct Result {
    std::vector<ms::Control> controls;
    sv::GenerationQueue::Work work;
};
struct Scenario {
    sv::GenerationQueue queue;
    ml::Shared shared;
    FakeEngine engine;
    std::atomic<bool> stop{false};
    std::vector<Event> events;
    std::map<unsigned, std::shared_ptr<sv::GenerationTicket>> tickets;
    std::string prompt_text;
    double now = 0;
    std::size_t next = 0;
    explicit Scenario(ms::Sched mode) {
        shared.policy.sched = mode;
        FakeClock::value = FakeClock::time_point{} + std::chrono::hours(1);
        sv::GenerationQueue::test_now = FakeClock::now;
        FakeClock::advance = [&](double ms) { advance(ms); };
    }
    void advance(double ms) {
        const double end = now + ms;
        while (next < events.size() && events[next].at <= end) {
            const auto e = events[next++];
            now = e.at; set_clock(); e.action();
        }
        now = end; set_clock();
    }
    void set_clock() { FakeClock::value = FakeClock::time_point{} + std::chrono::hours(1) +
        std::chrono::duration_cast<FakeClock::time_point::duration>(std::chrono::duration<double, std::milli>(now)); }
    void submit(double at, unsigned id, unsigned rows, unsigned tokens = 3, const std::string& session = "") {
        events.push_back({at, [=, this] {
            sv::GenerationJob j; j.prompt.resize(rows, id); j.max_tokens = tokens;
            j.session = session;
            tickets[id] = queue.submit(std::move(j));
        }});
    }
    // An explicit prompt (submit() fills with the id, so its requests never share a prefix); for the reuse path.
    void submit_prompt(double at, unsigned id, std::vector<std::uint32_t> prompt, unsigned tokens, const std::string& session = "") {
        events.push_back({at, [=, this] {
            sv::GenerationJob j; j.prompt = prompt; j.max_tokens = tokens;
            j.session = session;
            tickets[id] = queue.submit(std::move(j));
        }});
    }
    void cancel(double at, unsigned id) { events.push_back({at, [=, this] { tickets.at(id)->cancel(); }}); }
    void policy(double at, ms::Sched mode) { events.push_back({at, [=, this] {
        // Same shared policy + epoch transition used by /admin/sched, without HTTP parsing.
        std::lock_guard lock(shared.m); shared.policy.sched = mode; ++shared.policy.epoch;
    }}); }
    Result run(double end = 40000, ms::SessionService* session = nullptr) {
        events.push_back({end, [&] { stop.store(true); }});
        std::stable_sort(events.begin(), events.end(), [](const auto& a, const auto& b) { return a.at < b.at; });
        advance(0);
        ml::Lead<FakeEngine, FakeEngine, FakeClock> lead(engine, engine, queue, shared, 4096, engine.taps, stop,
            [&](const std::vector<std::uint32_t>&) { return prompt_text; }, session);
        lead.loop();
        return {engine.controls, queue.work};
    }
};
// Compare all emitted scheduling/control fields, including checkpoints, generations and ordered rows.
std::string sequence(const Result& result) {
    std::ostringstream out;
    for (const auto& c : result.controls) {
        out << c.command << ':' << c.round << ':' << c.task_id << ':' << c.parent_task_id << ':' << c.slot_generation
            << ':' << c.pslot << ':' << c.plen << ':' << c.pflags << ':' << c.bank << ':' << c.layers << ':' << c.append << ':' << c.rows;
        for (const auto& p : {c.active, c.parked})
            out << ':' << p.id << ':' << p.parent << ':' << p.generation << ':' << p.c0 << ':' << p.chunk << ':' << p.rows
                << ':' << p.slot << ':' << p.bank << ':' << p.layer << ':' << p.next << ':' << p.open;
        for (unsigned i = 0; i < c.rows; ++i) out << ':' << c.row[i].slot << ':' << c.row[i].token << ':' << c.row[i].pos;
        out << ':' << c.dslots;
        for (unsigned i = 0; i < c.dslots; ++i) out << ':' << c.dslot[i] << ':' << c.danchor[i] << ':' << c.dpos[i];
        out << '\n';
    }
    return out.str();
}
unsigned count(const Result& r, ms::Command command) {
    return static_cast<unsigned>(std::count_if(r.controls.begin(), r.controls.end(), [&](const auto& c) {
        return c.command == static_cast<unsigned>(command);
    }));
}
bool whole(const Result& r, std::uint64_t task) {
    bool found = false;
    for (const auto& c : r.controls) if (c.command == static_cast<unsigned>(ms::Command::unit) && c.task_id == task) {
        found = true; if (c.layers != ~0U) return false;
    }
    return found;
}
int main() {
    unsigned failures = 0;
    const auto check = [&](bool ok, const char* name) {
        std::cout << (ok ? "PASS " : "FAIL ") << name << '\n'; failures += ok ? 0U : 1U;
    };
    {
        ms::Control c;
        c.command=3;c.task_id=0x0102030405060708ULL;c.parent_task_id=9;c.slot_generation=11;
        c.active={12,13,14,15,16,17,18,19,20,21,true};
        c.parked={22,23,24,25,26,27,28,29,30,31,true};
        c.rows=8;c.pslot=3;c.plen=2000;c.pflags=7;c.bank=1;c.round=37;c.ack_round=36;
        c.layers=4;c.append=255;c.dslots=8;
        for(unsigned i=0;i<8;++i){c.row[i]={i,100+i,0x0102030405060708ULL+i};c.dslot[i]=i;c.danchor[i]=200+i;c.dpos[i]=900+i;}
        const auto canonical=ms::control_wire(c);
        const auto poison = [&](std::size_t first,std::size_t last) {
            auto* raw=reinterpret_cast<unsigned char*>(&c);
            for(auto i=first;i<last;++i)raw[i]=0xa5;
        };
        poison(12,16);poison(105,112);poison(177,184);poison(204,208);poison(428,432);
        const auto wire=ms::control_wire(c);
        check(std::memcmp(wire.data(),canonical.data(),512)==0,"poisoned padding cannot affect populated wire fields");
        ms::Control decoded;std::memcpy(&decoded,wire.data(),sizeof decoded);
        check(decoded.active==c.active && decoded.parked==c.parked && decoded.task_id==c.task_id &&
              decoded.parent_task_id==c.parent_task_id && decoded.slot_generation==c.slot_generation &&
              decoded.rows==8 && decoded.pslot==3 && decoded.plen==2000 && decoded.pflags==7 && decoded.bank==1 &&
              decoded.round==37 && decoded.ack_round==36 && decoded.layers==4 && decoded.append==255 && decoded.dslots==8,
              "populated task/checkpoint/control fields retain version-3 layout");
        bool fields=true;
        for(unsigned i=0;i<8;++i)fields &= decoded.row[i].slot==c.row[i].slot && decoded.row[i].token==c.row[i].token &&
            decoded.row[i].pos==c.row[i].pos && decoded.dslot[i]==c.dslot[i] && decoded.danchor[i]==c.danchor[i] && decoded.dpos[i]==c.dpos[i];
        for(std::size_t i=496;i<512;++i)fields &= wire[i]==std::byte{0};
        check(fields && wire[16]==std::byte{8} && wire[23]==std::byte{1},"ordered rows/drafts and little-endian values preserved; reserved tail zero");
    }
    {
        Scenario idle(ms::Sched::interleave);
        const auto r = idle.run(2500);
        check(idle.engine.quiesces == count(r, ms::Command::heartbeat) && idle.engine.quiesces >= 2,
              "paired idle heartbeats retire direct receives");
    }
    {
        // Frozen version-3 hello encoding of public main (control version 3), including zero padding.
        std::array<unsigned char, 512> legacy{};
        legacy[0] = 0x4d; legacy[1] = 0x4d; legacy[2] = 0x53; legacy[3] = 0x43;
        legacy[4] = 3;
        std::fill_n(legacy.begin() + 216, 8, 0xff);
        FakeEngine eng;
        check(!ms::session_hello(eng, false), "both OFF skip session admission");
        check(eng.controls.size() == 1 && std::memcmp(eng.last_wire.data(), legacy.data(), legacy.size()) == 0,
              "session OFF hello equals public main bytes");
        check(ms::session_hello(eng, true), "local ON requests session admission");
        check(eng.controls.back().pflags == ms::kHelloSessionVote, "hello vote uses existing flags");
        eng.session_peer = true;
        check(ms::session_hello(eng, false), "peer ON requests admission even with local OFF");
        check(std::memcmp(eng.last_wire.data(), legacy.data(), legacy.size()) == 0,
              "local OFF hello stays unchanged with peer ON");
        // Prefix reuse needs both votes; a peer without the bit (switch off, or a build before pstart) declines it.
        FakeEngine px;
        bool agreed = true;
        check(!ms::session_hello(px, false, true, &agreed) && !agreed, "prefix vote alone does not enable reuse");
        check(px.controls.back().pflags == ms::kHelloPrefixVote, "prefix vote uses the hello flags");
        px.prefix_peer = true;
        check(!ms::session_hello(px, false, false, &agreed) && !agreed, "peer prefix vote alone does not enable reuse");
        check(std::memcmp(px.last_wire.data(), legacy.data(), legacy.size()) == 0, "prefix OFF hello equals public main bytes");
        check(!ms::session_hello(px, false, true, &agreed) && agreed, "both prefix votes enable reuse");
        px.peer_alignment = 12288;
        check(!ms::session_hello(px, false, true, &agreed, 8192) && !agreed, "different prefill group alignment declines reuse");
        // Conversation mode: a third bit, only with the prefix vote, both ranks. With it off every byte is unchanged.
        FakeEngine cx;
        bool conversation = true;
        check(!ms::session_hello(cx, false, true, &agreed, 4096, false, &conversation) && !conversation &&
              cx.controls.back().pflags == ms::kHelloPrefixVote, "conversation OFF leaves the prefix hello bytes");
        check(!ms::session_hello(cx, false, true, &agreed, 4096, true, &conversation) && !conversation && !agreed &&
              cx.controls.back().pflags == (ms::kHelloPrefixVote | ms::kHelloConversationVote),
              "conversation vote alone (no peer votes) enables nothing");
        cx.prefix_peer = true;
        check(!ms::session_hello(cx, false, true, &agreed, 4096, true, &conversation) && agreed && !conversation,
              "peer without the conversation bit keeps chunk-aligned reuse only");
        cx.conversation_peer = true;
        check(!ms::session_hello(cx, false, true, &agreed, 4096, true, &conversation) && agreed && conversation,
              "both conversation votes agree");
        check(!ms::session_hello(cx, false, true, &agreed, 4096, false, &conversation) && agreed && !conversation,
              "a local OFF cancels the peer's conversation vote");
        check(!ms::session_hello(cx, false, false, &agreed, 4096, true, &conversation) && !conversation &&
              cx.controls.back().pflags == 0, "conversation never votes without the prefix vote");
        // A follower accepts a mid-chunk resident restart only when conversation mode was agreed.
        FakeEngine rank;
        ms::Control rc; rc.command = 2; rc.task_id = 1; rc.slot_generation = 1; rc.pstart = 1000; rc.plen = 5;
        const auto receive = [](std::size_t n) { return std::vector<unsigned>(n, 7U); };
        bool refused = false;
        try { (void)ms::replay_resident_begin(rc, rank, rank, receive, true); } catch (...) { refused = true; }
        check(refused, "unaligned resident restart is refused without conversation mode");
        check(ms::replay_resident_begin(rc, rank, rank, receive, true, true) && rank.active.open && rank.active.rows == 1005,
              "unaligned resident restart replays with conversation mode");
    }
    const auto compare = [&](const char* name, const std::function<void(Scenario&)>& script, bool dspark = false) {
        const auto run = [&](ms::Sched mode) {
            Scenario s(mode); if (dspark) { s.engine.dd = &s.engine; s.engine.taps = true; s.shared.policy.dspark_mode = ml::ds::Mode::on; s.shared.policy.inflight = 8; }
            script(s); return s.run();
        };
        const auto off = run(ms::Sched::off), on = run(ms::Sched::interleave);
        check(sequence(off) == sequence(on), name);
        check(on.work.take_locks == off.work.take_locks && on.work.cancel_visits == off.work.cancel_visits &&
              on.work.selection_visits == off.work.selection_visits && on.work.lookahead_locks == off.work.lookahead_locks && on.work.lookahead_visits == off.work.lookahead_visits,
              "FIFO queue locks/scans equal OFF");
        return on;
    };
    for (unsigned n : {1U, 4U, 8U}) {
        const auto r = compare(("equal-time burst " + std::to_string(n)).c_str(), [&](auto& s) {
            for (unsigned i = 0; i < n; ++i) s.submit(0, i + 1, 1024 - 97 * i);
        });
        check(count(r, ms::Command::prefill) == n && count(r, ms::Command::unit) == n, "one unlimited unit per burst request");
    }
    compare("before-first due arrivals", [](auto& s) { s.submit(0, 1, 900); s.submit(100, 2, 20); s.submit(200, 3, 500); });
    compare("cancel before admission", [](auto& s) { s.submit(0, 1, 900); s.submit(0, 2, 20); s.cancel(0, 2); s.submit(0, 3, 500); });
    compare("cancel during whole prefill", [](auto& s) { s.submit(0, 1, 900); s.cancel(100, 1); s.submit(200, 2, 20); });
    for (bool dspark : {false, true}) {
        const auto r = compare(dspark ? "slot reuse cancellation DSpark" : "slot reuse cancellation", [](auto& s) {
            s.submit(0, 1, 10); s.submit(0, 2, 900); s.submit(0, 3, 20); s.cancel(800, 1);
        }, dspark);
        const auto it = std::find_if(r.controls.begin(), r.controls.end(), [](auto& c) { return c.command == 3; });
        std::vector<unsigned> slots;
        if (it != r.controls.end()) for (unsigned i = 0; i < it->rows; ++i)
            if (slots.empty() || slots.back() != it->row[i].slot) slots.push_back(it->row[i].slot);
        check(slots == std::vector<unsigned>({0, 1, 2}), "cancelled OFF decoder stays in ordered step");
        if (dspark) check(count(r, ms::Command::draft) > 0 && it != r.controls.end() && it->append == 7,
                          "DSpark drafts and context appends include retained OFF decoder");
    }
    {
        const auto script = [](auto& s) { s.submit(0, 1, 10, 100); s.submit(800, 2, 2000); s.submit(1000, 3, 10); };
        Scenario off(ms::Sched::off); script(off); const auto reference = off.run();
        Scenario s(ms::Sched::interleave); script(s); const auto r = s.run();
        check(sequence(r) != sequence(reference) && whole(reference, 2) && count(reference, ms::Command::nest) == 0,
              "after-first mixed trace differs from whole OFF reference");
        check(!whole(r, 2) && count(r, ms::Command::nest) == 1 && count(r, ms::Command::park) == 1 &&
              count(r, ms::Command::resume_parked) == 1, "after-first child during bounded parent nests");
        const auto p = std::find_if(r.controls.begin(), r.controls.end(), [](auto& c) { return c.command == 2 && c.task_id == 2; });
        check(p != r.controls.end() && p->plen == 2000, "future short cannot overtake reserved long");
    }
    {
        const auto script = [](auto& s) { s.submit(0, 1, 10, 100); s.submit(800, 2, 2000); s.cancel(1000, 1); s.submit(1100, 3, 10); };
        Scenario off(ms::Sched::off); script(off); const auto reference = off.run();
        Scenario s(ms::Sched::interleave); script(s); const auto r = s.run();
        check(sequence(r) != sequence(reference) && whole(reference, 2) && whole(reference, 3), "cancelled mixed protector OFF reference");
        check(!whole(r, 2) && count(r, ms::Command::nest) == 0 && whole(r, 3), "protector cancelled during parent unit cannot nest child");
    }
    {
        const auto script = [](auto& s) { s.submit(0, 1, 10, 100); s.submit(800, 2, 2000); s.cancel(1000, 2); s.submit(1100, 3, 10); };
        Scenario off(ms::Sched::off); script(off); const auto reference = off.run();
        Scenario s(ms::Sched::interleave); script(s); const auto r = s.run();
        check(count(r, ms::Command::cancel) == 1 && count(reference, ms::Command::cancel) == 0 && sequence(r) != sequence(reference),
              "bounded prefill cancellation emits boundary cancel; OFF completes whole unit");
    }
    {
        Scenario s(ms::Sched::interleave); s.engine.parent_speed = 0.1;
        s.submit(0, 1, 10, 100); s.submit(800, 2, 2000); s.cancel(850, 1); s.submit(900, 3, 10);
        const auto r = s.run();
        check(!whole(r, 2) && whole(r, 3) && count(r, ms::Command::nest) == 0,
              "cancelled protector at fitting parent boundary cannot grant nesting");
    }
    {
        const auto script = [](auto& s) { s.submit(0, 1, 10, 100); s.submit(0, 2, 900, 1);
            s.cancel(800, 1); s.submit(800, 3, 2000); s.submit(800, 4, 10); };
        Scenario off(ms::Sched::off); script(off); const auto reference = off.run();
        Scenario s(ms::Sched::off); script(s); s.policy(800, ms::Sched::interleave); const auto r = s.run();
        check(sequence(r) == sequence(reference), "cancelled old-policy protector switch equals OFF");
        check(whole(r, 3) && whole(r, 4) && count(r, ms::Command::nest) == 0, "admin OFF-to-interleave cancelled old-policy protector");
    }
    {
        const auto r = compare("completed protector slot reuse equals OFF", [](auto& s) { s.submit(0, 1, 10, 1); s.submit(800, 2, 2000); });
        check(whole(r, 2), "completed protector cannot protect reused slot");
    }
    {
        const auto script = [](auto& s) { s.submit(0, 1, 10, 100); s.submit(800, 2, 2000); s.policy(1000, ms::Sched::off); s.submit(1100, 3, 10); };
        Scenario off(ms::Sched::off); script(off); const auto reference = off.run();
        Scenario s(ms::Sched::interleave); script(s); const auto r = s.run();
        check(sequence(r) != sequence(reference) && whole(reference, 2) && whole(reference, 3), "switched bounded parent differs from OFF reference");
        check(!whole(r, 2) && whole(r, 3) && count(r, ms::Command::nest) == 0, "admin switch retains parent identity, disables new nests");
    }
    {
        Scenario s(ms::Sched::interleave); s.submit(0, 1, 10, 3); s.submit(680, 2, 2000); s.submit(2000, 3, 10);
        const auto r = s.run();
        check(!whole(r, 2) && whole(r, 3) && count(r, ms::Command::nest) == 0,
              "protector completes during bounded parent; later child cannot nest");
    }
    {
        Scenario s(ms::Sched::interleave); s.submit(0, 1, 10, 100);
        s.submit(800, 2, 2000); s.submit(801, 3, 10);
        s.policy(1500, ms::Sched::off); s.submit(1510, 4, 1);
        const auto r = s.run();
        std::vector<unsigned> prompts;
        for (const auto& c : r.controls) if (c.command == 2) prompts.push_back(c.plen);
        check(prompts == std::vector<unsigned>({10, 10, 2000, 1}) && whole(r, 3),
              "pending reservation survives admin switch ahead of later short");
    }
    for (unsigned n : {1U, 2U, 4U}) {
        const auto run = [&](ml::ds::Mode mode) {
            Scenario s(ms::Sched::interleave); s.engine.dd = &s.engine; s.shared.policy.dspark_mode = mode;
            for (unsigned i = 0; i < n; ++i) s.submit(0, i + 1, i == 0 ? 6551 : 100, 6 + i);
            s.submit(12000, 20, 100, 3);  // drain, then slot reuse
            return s.run();
        };
        const auto automatic = run(ml::ds::Mode::automatic), off = run(ml::ds::Mode::off);
        check(sequence(automatic) == sequence(off), "auto unqualified trace equals OFF, including burst drain / slot reuse");
        bool plain = true;
        for (const auto& c : automatic.controls) {
            if (c.command == static_cast<unsigned>(ms::Command::draft) || c.append != 0 ||
                (c.command == static_cast<unsigned>(ms::Command::taps) && c.pflags != 0) ||
                (c.command == static_cast<unsigned>(ms::Command::prefill) && c.pflags != 0)) plain = false;
        }
        check(plain, "auto PLAIN has no draft, append, tap enable or prefill sink");
    }
    {
        Scenario s(ms::Sched::interleave); s.engine.dd = &s.engine; s.engine.taps = true;
        s.shared.policy.dspark_mode = ml::ds::Mode::on;
        s.submit(0, 1, 100, 20);
        s.events.push_back({100, [&] {
            std::lock_guard lock(s.shared.m); s.shared.policy.dspark_mode = ml::ds::Mode::off; ++s.shared.policy.epoch;
        }});
        const auto r = s.run();
        bool disabled = false, clean = true;
        for (const auto& c : r.controls) {
            if (c.command == static_cast<unsigned>(ms::Command::taps) && c.pflags == 0) {
                disabled = true; clean = clean && !c.active.open && !c.parked.open;
            }
            if (disabled && ((c.command == static_cast<unsigned>(ms::Command::step) && c.append != 0) || c.command == static_cast<unsigned>(ms::Command::draft))) clean = false;
        }
        check(disabled && clean && count(r, ms::Command::retire) == 1, "OFF during prefill retires context and disables taps only after task closure");
    }
    {
        Scenario s(ms::Sched::interleave); s.engine.dd = &s.engine;
        s.prompt_text = "```python\nprint(1)\n```";
        s.submit(0, 1, 100, 100); s.submit(1800, 2, 100, 10);
        const auto r = s.run();
        check(count(r, ms::Command::draft) > 0, "structural C1 promotes into legacy controller");
        bool retired = false, disabled = false, clean = true;
        for (const auto& c : r.controls) {
            if (c.command == static_cast<unsigned>(ms::Command::retire)) retired = true;
            if (retired && c.command == static_cast<unsigned>(ms::Command::taps) && c.pflags == 0) disabled = true;
            if (retired && ((c.command == static_cast<unsigned>(ms::Command::step) && c.append) || c.command == static_cast<unsigned>(ms::Command::draft))) clean = false;
        }
        check(retired && disabled && clean, "C1 to C2 retires auto with no drain reactivation");
    }
    {
        Scenario s(ms::Sched::interleave); s.engine.dd = &s.engine; s.prompt_text = "```code```";
        s.submit(0, 1, 100, 10); s.submit(0, 2, 100, 10);
        const auto r = s.run();
        check(count(r, ms::Command::draft) == 0 && count(r, ms::Command::taps) == 0, "structural burst stays plain from admission");
    }
    {
        Scenario s(ms::Sched::interleave); s.engine.dd = &s.engine; s.prompt_text = "```code```";
        s.shared.policy.dspark_prior = ml::ds::Prior::unqualified_v1;
        s.submit(0, 1, 100, 10);
        const auto r = s.run();
        check(count(r, ms::Command::draft) == 0 && count(r, ms::Command::taps) == 0, "unqualified fallback ignores structural markers");
    }
    {
        Scenario s(ms::Sched::interleave);
        s.submit(0, 1, 10, 100);
        s.submit(800, 2, 2000, 3);
        s.submit(1000, 3, 10, 2, "conversation");
        ms::SessionService service;
        unsigned calls = 0, parks = 0;
        service.execute = [&](const std::string& body, const ms::SessionService::Publish& publish) {
            if (sv::parse_json(body).find("op")->as_string() == "park_active") {
                // The reply was published before the park: the ticket is finished with its final count.
                check(calls == 1 && s.tickets.at(3)->stats().generated_tokens == 2, "named request is answered before its park");
                check(sv::parse_json(body).find("id")->as_string() == "conversation", "park names the same session");
                ++parks;
                return sv::Json::object();
            }
            check(!s.engine.prefill_open(), "session runs without active or parked prefill");
            check(s.tickets.at(1)->stats().generated_tokens == 100 && s.tickets.at(2)->stats().generated_tokens == 3,
                  "named request waits for ordinary decoders");
            const auto command = sv::parse_json(body);
            check(command.find("id")->as_string() == "conversation" && command.find("count")->as_int() == 2,
                  "named request command retains session and generation count");
            check(command.find("defer_park") && command.find("defer_park")->as_bool(), "named request defers its park to after the reply");
            const std::array<std::uint32_t, 2> ids{201, 202};
            check(publish(ids), "session callback accepts complete tokens");
            auto reply = sv::Json::object(), tokens = sv::Json::array();
            for (auto id : ids) tokens.push(sv::Json(std::uint64_t(id)));
            reply.set("token_ids", std::move(tokens)); reply.set("hit", sv::Json(true));
            ++calls;
            return reply;
        };
        const auto r = s.run(40000, &service);
        check(calls == 1 && parks == 1 && count(r, ms::Command::session) == 2 && count(r, ms::Command::nest) == 0,
              "named request dispatches once at idle (then its park) and cannot nest");
        const auto stats = s.tickets.at(3)->stats();
        check(stats.generated_tokens == 2 && stats.cached_tokens == 10 && stats.finish == "length",
              "named request completion publishes cache hit and complete token count");
        for (std::size_t i = 1; i < r.controls.size(); ++i)
            check(r.controls[i].round == r.controls[i - 1].round + 1, "session preserves control round sequence");
    }
    {
        // The session ran out of context after one token: the reply's finish wins over the max_tokens comparison.
        Scenario s(ms::Sched::interleave);
        s.submit(0, 1, 10, 4, "conversation");
        ms::SessionService service;
        service.execute = [&](const std::string& body, const ms::SessionService::Publish& publish) {
            if (sv::parse_json(body).find("op")->as_string() == "park_active") return sv::Json::object();
            const std::array<std::uint32_t, 1> ids{201};
            (void)publish(ids);
            auto reply = sv::Json::object(), tokens = sv::Json::array();
            tokens.push(sv::Json(std::uint64_t(ids[0])));
            reply.set("token_ids", std::move(tokens)); reply.set("hit", sv::Json(true)); reply.set("finish", sv::Json("length"));
            return reply;
        };
        (void)s.run(40000, &service);
        const auto stats = s.tickets.at(1)->stats();
        check(stats.generated_tokens == 1 && stats.finish == "length", "context-clamped session generation finishes with length");
    }
    {
        // Prompt lookup: a repeating stream drafts lookup windows (fewer step cycles); the emitted tokens equal lookup off.
        ml::Counters k{};
        unsigned deepest = 0;  // the most rows any request's window had in one step (drafts beyond its anchor)
        const auto run = [&](bool lookup, unsigned& steps, unsigned& window_rows, unsigned pattern = 0, unsigned requests = 1,
                             unsigned tokens = 40) {
            Scenario s(ms::Sched::off);
            s.shared.policy.lookup = lookup;
            s.engine.pattern = pattern;
            for (unsigned q = 0; q < requests; ++q) s.submit(0, q + 1, 64 + 7 * q, tokens);
            const auto r = s.run();
            steps = count(r, ms::Command::step);
            window_rows = 0;
            deepest = 0;
            for (const auto& c : r.controls) {
                if (c.command != static_cast<unsigned>(ms::Command::step)) continue;
                window_rows += c.rows - std::min(c.rows, requests);
                for (unsigned i = 0, run_len = 0; i < c.rows; ++i) {  // consecutive rows of one slot beyond its anchor
                    run_len = i > 0 && c.row[i].slot == c.row[i - 1].slot ? run_len + 1 : 0;
                    deepest = std::max(deepest, run_len);
                }
            }
            k = s.shared.counters;
            std::vector<unsigned> ids;
            for (unsigned q = 0; q < requests; ++q) {
                bool done = false;
                const auto got = s.tickets.at(q + 1)->wait(std::chrono::milliseconds(0), done);
                ids.insert(ids.end(), got.begin(), got.end());
            }
            return ids;
        };
        unsigned steps_on = 0, rows_on = 0, steps_off = 0, rows_off = 0;
        const auto on = run(true, steps_on, rows_on), off = run(false, steps_off, rows_off);
        check(on.size() == 40 && on == off, "prompt lookup emits the tokens of lookup off");
        check(rows_off == 0 && rows_on > 0 && steps_on < steps_off, "prompt lookup drafts windows on a repeating stream");
        // position-dependent stream: lookup proposals are partly rejected and windows vary in depth
        const auto pon = run(true, steps_on, rows_on, 1, 1, 200);
        const auto pk = k;
        const auto pon_deepest = deepest;
        const auto poff = run(false, steps_off, rows_off, 1, 1, 200);
        check(pon.size() == 200 && pon == poff, "prompt lookup with rejected drafts emits the tokens of lookup off");
        std::cout << "lookup position-dependent: windows " << pk.lookup_windows << " drafted " << pk.lookup_drafted << " accepted "
                  << pk.lookup_accepted << " deepest " << pon_deepest << std::endl;
        check(pk.lookup_windows > 0 && pk.lookup_accepted < pk.lookup_drafted && pon_deepest >= 2,
              "position-dependent stream: lookup drafts are rejected and windows go deeper than one row");
        // four requests in one step (C4): every request's tokens equal lookup off
        const auto con = run(true, steps_on, rows_on, 0, 4, 200);
        const auto ck = k;
        const auto coff = run(false, steps_off, rows_off, 0, 4, 200);
        check(con.size() == 800 && con == coff, "prompt lookup at C4 emits the tokens of lookup off");
        std::cout << "lookup C4: windows " << ck.lookup_windows << " drafted " << ck.lookup_drafted << " accepted " << ck.lookup_accepted
                  << " rows " << rows_on << std::endl;
        check(ck.lookup_windows > 0 && rows_on > 0, "prompt lookup drafts windows with four requests in the step");
    }
    {
        // Prefix reuse (both ranks voted): a second request sharing leading tokens with a finished slot's resident cache
        // reuses whole prefill chunks of them (max_rows 4096 here: 9000 shared tokens -> 8192 reused) and prefills the
        // rest. The cold request reuses nothing; the warm prefill carries pstart = reuse and plen = the rest.
        Scenario s(ms::Sched::off);
        s.shared.prefix_reuse = true;
        s.shared.prefix_identity = {"loaded-weights", "profile-tp2", "prefill-v1", 2};
        std::vector<std::uint32_t> base(9000, 7U);
        std::vector<std::uint32_t> warm = base;
        warm.insert(warm.end(), {8U, 9U, 10U});              // prompt.size() = 9003
        s.submit_prompt(0, 1, base, 2);
        s.submit_prompt(80000, 2, warm, 2);
        const auto r = s.run(200000);
        const auto cold = s.tickets.at(1)->stats(), hot = s.tickets.at(2)->stats();
        check(cold.cached_tokens == 0, "cold prefix request reuses nothing");
        check(hot.cached_tokens == 8192, "warm request reuses the shared prefix in whole prefill chunks");
        const auto cold_pf = std::find_if(r.controls.begin(), r.controls.end(),
            [](const auto& c) { return c.command == static_cast<unsigned>(ms::Command::prefill) && c.task_id == 1; });
        const auto hot_pf = std::find_if(r.controls.begin(), r.controls.end(),
            [](const auto& c) { return c.command == static_cast<unsigned>(ms::Command::prefill) && c.task_id == 2; });
        check(cold_pf != r.controls.end() && cold_pf->pstart == 0 && cold_pf->plen == 9000,
              "cold prefill starts at 0 and sends the whole prompt");
        check(hot_pf != r.controls.end() && hot_pf->pstart == 8192 && hot_pf->plen == 811,
              "warm prefill opens on a chunk boundary and sends only the rest");
    }
    {
        // A shared prefix shorter than one prefill chunk is not reused: its suffix could not open on a chunk boundary.
        Scenario s(ms::Sched::off);
        s.shared.prefix_reuse = true;
        s.shared.prefix_identity = {"loaded-weights", "profile-tp2", "prefill-v1", 2};
        std::vector<std::uint32_t> base(4000, 7U), warm = base;
        warm.insert(warm.end(), {8U, 9U, 10U});
        s.submit_prompt(0, 1, base, 2);
        s.submit_prompt(80000, 2, warm, 2);
        const auto r = s.run(200000);
        check(s.tickets.at(2)->stats().cached_tokens == 0 &&
              std::all_of(r.controls.begin(), r.controls.end(), [](const auto& c) { return c.pstart == 0; }),
              "a shared prefix under one prefill chunk is prefilled in full");
    }
    {
        // Default (no agreed vote): the same pair prefills every token, reusing nothing; pstart stays zero on the wire.
        Scenario s(ms::Sched::off);
        std::vector<std::uint32_t> base(9000, 7U), warm = base;
        warm.insert(warm.end(), {8U, 9U, 10U});
        s.submit_prompt(0, 1, base, 2);
        s.submit_prompt(80000, 2, warm, 2);
        const auto r = s.run(200000);
        check(s.tickets.at(2)->stats().cached_tokens == 0, "prefix reuse off reuses nothing");
        const auto hot_pf = std::find_if(r.controls.begin(), r.controls.end(),
            [](const auto& c) { return c.command == static_cast<unsigned>(ms::Command::prefill) && c.task_id == 2; });
        check(hot_pf != r.controls.end() && hot_pf->pstart == 0 && hot_pf->plen == 9003,
              "prefix reuse off prefills the whole warm prompt");
        check(std::all_of(r.controls.begin(), r.controls.end(), [](const auto& c) { return c.pstart == 0; }),
              "prefix reuse off leaves pstart zero in every control");
    }
    {
        // Conversation mode (both ranks voted twice): a finished reply keeps its slot's rows, and the follow-up turn
        // restarts prefill at the reply-end frontier (unaligned pstart), computing only the new suffix. Turn 1: 300
        // prompt tokens + 4 reply tokens (rows 0..302 consumed, the last reply token is the pending anchor).
        const auto turns = [](bool prefix, bool conversation, std::uint64_t alignment = 4096) {
            auto s = std::make_unique<Scenario>(ms::Sched::off);
            s->shared.prefix_reuse = prefix; s->shared.conversation = conversation;
            if (prefix) s->shared.prefix_identity = {"loaded-weights", "profile-tp2", "prefill-v1", 2};
            s->engine.reuse_alignment = alignment;
            return s;
        };
        const std::vector<std::uint32_t> t1(300, 7U);
        auto t2 = t1; t2.insert(t2.end(), 4, 100U); t2.insert(t2.end(), {21U, 22U, 23U, 24U, 25U});  // reply (slot 0) + user turn
        auto t3 = t2; t3.insert(t3.end(), 4, 100U); t3.insert(t3.end(), {31U, 32U});
        const auto first_prefill = [](const Result& r, unsigned id) {
            return std::find_if(r.controls.begin(), r.controls.end(), [id](const auto& c) {
                return c.command == static_cast<unsigned>(ms::Command::prefill) && c.task_id == id; });
        };
        auto s = turns(true, true);
        s->submit_prompt(0, 1, t1, 4); s->submit_prompt(80000, 2, t2, 4); s->submit_prompt(160000, 3, t3, 4);
        const auto r = s->run(300000);
        const auto p2 = first_prefill(r, 2), p3 = first_prefill(r, 3);
        check(first_prefill(r, 1)->pstart == 0 && first_prefill(r, 1)->pslot == 0, "conversation turn 1 is cold on slot 0");
        check(s->tickets.at(2)->stats().cached_tokens == 303 && p2->pstart == 303 && p2->plen == t2.size() - 303,
              "turn 2 restarts at the reply-end frontier (prompt + reply rows consumed), mid-chunk");
        check(s->tickets.at(3)->stats().cached_tokens == t2.size() + 3 && p3->pstart == t2.size() + 3 && p3->plen == t3.size() - t2.size() - 3,
              "turn 3 restarts at turn 2's reply-end frontier");
        check(p2->pslot == 0 && p3->pslot == 0, "the follow-ups continue in the slot that held the conversation");
        {
            std::lock_guard lock(s->shared.m);
            const auto& x2 = s->shared.extras.at(s->tickets.at(2).get()), &x3 = s->shared.extras.at(s->tickets.at(3).get());
            check(x2.resident_reported && x2.reply_extended && x2.prefill_computed == t2.size() - 303,
                  "turn 2 reports a reply-extended hit and its computed prefill rows");
            check(x3.reply_extended && x3.prefill_computed == t3.size() - t2.size() - 3, "turn 3 reports the same");
            check(!s->shared.extras.at(s->tickets.at(1).get()).reply_extended && s->shared.extras.at(s->tickets.at(1).get()).prefill_computed == 300,
                  "the cold turn computes every prompt row and is not reply-extended");
        }

        // One generated token: the record is the prompt alone (exact frontier, reply_tokens 0), still restarted mid-chunk.
        auto one = turns(true, true);
        auto o2 = t1; o2.insert(o2.end(), {100U, 41U});
        one->submit_prompt(0, 1, t1, 1); one->submit_prompt(80000, 2, o2, 2);
        (void)one->run(200000);
        check(one->tickets.at(2)->stats().cached_tokens == 300, "a one-token reply keeps the prompt frontier");
        {
            std::lock_guard lock(one->shared.m);
            check(!one->shared.extras.at(one->tickets.at(2).get()).reply_extended, "a prompt-only record is not reply-extended");
        }

        // Cancellation during decode keeps nothing new: no reply-extended record and (prompt under a chunk) no aligned one.
        auto cancelled = turns(true, true);
        cancelled->submit_prompt(0, 1, t1, 1000); cancelled->cancel(6000, 1); cancelled->submit_prompt(80000, 2, t2, 2);
        const auto rc = cancelled->run(200000);
        check(cancelled->tickets.at(1)->stats().finish == "cancelled" && cancelled->tickets.at(2)->stats().cached_tokens == 0 &&
              first_prefill(rc, 2)->pstart == 0, "a cancelled reply certifies no frontier");

        // Default (prefix reuse ON, conversation OFF): chunk-aligned only; nothing under a chunk is reused, pstart stays 0.
        auto aligned = turns(true, false);
        aligned->submit_prompt(0, 1, t1, 4); aligned->submit_prompt(80000, 2, t2, 4);
        const auto ra = aligned->run(200000);
        check(aligned->tickets.at(2)->stats().cached_tokens == 0 &&
              std::all_of(ra.controls.begin(), ra.controls.end(), [](const auto& c) { return c.pstart == 0; }),
              "conversation OFF leaves the aligned default: no mid-chunk restart on the wire");
        {
            std::lock_guard lock(aligned->shared.m);
            check(!aligned->shared.extras.at(aligned->tickets.at(2).get()).reply_extended, "conversation OFF never reports reply-extended");
        }
        // Everything OFF: no resident bookkeeping in the per-request extras (the default response is unchanged).
        auto off = turns(false, false);
        off->submit_prompt(0, 1, t1, 4); off->submit_prompt(80000, 2, t2, 4);
        (void)off->run(200000);
        {
            std::lock_guard lock(off->shared.m);
            check(!off->shared.extras.at(off->tickets.at(2).get()).resident_reported, "cache OFF reports no resident fields");
        }

        // C4: four conversations interleaved. Each follow-up finds the slot that holds its own conversation.
        auto c4 = turns(true, true);
        std::array<std::vector<std::uint32_t>, 4> base, next;
        for (unsigned k = 0; k < 4; ++k) {
            base[k].assign(300 + 10 * k, 50U + k);
            c4->submit_prompt(1000.0 * k, 1 + k, base[k], 3);
        }
        for (unsigned k = 0; k < 4; ++k) {
            next[k] = base[k]; next[k].insert(next[k].end(), 3, 100U + k);  // the reply of slot k
            next[k].insert(next[k].end(), {61U + k, 62U});
        }
        for (unsigned j = 0; j < 4; ++j) { const unsigned k = 3 - j; c4->submit_prompt(80000.0 + 2000.0 * j, 11 + k, next[k], 2); }
        const auto r4 = c4->run(400000);
        for (unsigned k = 0; k < 4; ++k) {
            const auto cold = first_prefill(r4, 1 + k), warm = first_prefill(r4, 5 + (3 - k));  // request ids follow admission order
            check(cold->pslot == k && warm->pslot == k && c4->tickets.at(11 + k)->stats().cached_tokens == base[k].size() + 2 &&
                  warm->pstart == base[k].size() + 2, "C4 follow-up resumes its own conversation's slot at its own frontier");
        }
    }
    {
        // A session command runs in the slots outside admit / step, so no slot's resident tokens are known after it:
        // the request that follows a named one prefills in full even though its prefix matched before.
        Scenario s(ms::Sched::off);
        s.shared.prefix_reuse = true;
        s.shared.prefix_identity = {"loaded-weights", "profile-tp2", "prefill-v1", 2};
        std::vector<std::uint32_t> base(9000, 7U), warm = base;
        warm.insert(warm.end(), {8U, 9U, 10U});
        s.submit_prompt(0, 1, base, 2);
        s.submit(80000, 2, 10, 2, "conversation");
        s.submit_prompt(160000, 3, warm, 2);
        ms::SessionService service;
        service.execute = [&](const std::string& body, const ms::SessionService::Publish&) {
            auto reply = sv::Json::object();
            if (sv::parse_json(body).find("op")->as_string() == "park_active") return reply;
            reply.set("token_ids", sv::Json::array()); reply.set("hit", sv::Json(false));
            return reply;
        };
        const auto r = s.run(400000, &service);
        const auto pf = std::find_if(r.controls.begin(), r.controls.end(),
            [](const auto& c) { return c.command == static_cast<unsigned>(ms::Command::prefill) && c.task_id == 2; });
        check(count(r, ms::Command::session) == 2 && s.tickets.at(3)->stats().cached_tokens == 0 &&
              pf != r.controls.end() && pf->pstart == 0 && pf->plen == 9003,
              "a session command invalidates every slot's reusable prefix");
    }
    {
        Scenario s(ms::Sched::off);
        s.shared.prefix_reuse = true;
        s.shared.prefix_identity = {"loaded-weights", "profile-tp2", "prefill-v1", 2};
        s.engine.reuse_alignment = 8192;  // the production engine can group two chunks
        std::vector<std::uint32_t> base(7000, 7U), warm = base;
        warm.push_back(8U);
        s.submit_prompt(0, 1, base, 2);
        s.submit_prompt(80000, 2, warm, 2);
        s.run(200000);
        check(s.tickets.at(2)->stats().cached_tokens == 0, "partial grouped prefill units are not reused");
    }
    {
        Scenario s(ms::Sched::off);
        s.shared.prefix_reuse = true;
        s.shared.prefix_identity = {"loaded-weights", "profile-tp2", "prefill-v1", 2};
        std::vector<std::uint32_t> base(4095, 7U), warm = base;
        // FakeEngine emits 100 for slot 0. The first is then fed by decode, beyond the partial prompt chunk.
        warm.insert(warm.end(), {100U, 100U, 9U});
        s.submit_prompt(0, 1, base, 2);
        s.submit_prompt(80000, 2, warm, 2);
        s.run(200000);
        check(s.tickets.at(2)->stats().cached_tokens == 0, "decode rows cannot complete a reusable prefill chunk");
    }
    {
        Scenario s(ms::Sched::interleave);
        s.shared.prefix_reuse = true;
        s.shared.prefix_identity = {"loaded-weights", "profile-tp2", "prefill-v1", 2};
        std::vector<std::uint32_t> prompt(9000, 7U);
        s.submit(0, 99, 10, 100);  // a protected decoder forces the long prefill to yield
        s.submit_prompt(800, 1, prompt, 2);
        s.cancel(1000, 1);
        s.submit_prompt(80000, 2, prompt, 2);
        const auto r = s.run(200000);
        check(count(r, ms::Command::cancel) > 0 && s.tickets.at(2)->stats().cached_tokens == 0,
              "cancelled cold prefill cannot publish reusable rows");
    }
    std::cout << "production Lead traces: " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
