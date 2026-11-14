#include "strix/qpf2k_defaults.hpp"

#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using strix::qpf2k::PrefillConstruction;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Session {
    unsigned next_position = 0;
    bool sibling = false;
};
struct Counters { unsigned owners = 0, preflights = 0, handoffs = 0, executions = 0; };
// Model the native owner's permanent failure guard without a HIP dependency.
struct Owner {
    Counters& counts;
    bool inject_failure, failed = false, prepared = false;
    Owner(Counters& c, bool inject) : counts(c), inject_failure(inject) { ++counts.owners; }
    ~Owner() { --counts.owners; }
    void prepare_prefill_geometry(Session& session) {
        require(session.sibling, "preflight did not receive sibling session");
        ++counts.preflights;
        if (inject_failure) { failed = true; throw std::bad_alloc(); }
        prepared = true;
        session.next_position = 8192; // Must never affect the live session.
    }
};
struct Backend {
    Owner& owner;
    Session& session;
    void execute(unsigned rows) {
        require(!owner.failed, "published a poisoned owner");
        if (rows == 8192 && !owner.prepared) {
            auto setup = session;
            setup.sibling = true;
            owner.prepare_prefill_geometry(setup);
        }
        ++owner.counts.executions;
        session.next_position += rows;
    }
};
struct Slot {
    Session session;
    Owner owner;
    Backend backend;
    Slot(Counters& counts, bool inject) : owner(counts, inject), backend{owner, session} {}
};
using Slots = std::vector<std::unique_ptr<Slot>>;
void build_slot(Slots& published, Counters& counts, PrefillConstruction phase,
    bool inject = false, bool enabled = true, unsigned position = 0) {
    auto slot = std::make_unique<Slot>(counts, inject);
    slot->session.next_position = position;
    strix::qpf2k::prepare_first_block_admission(enabled, phase, slot->owner, slot->session,
        [&](Session& copy) { ++counts.handoffs; copy.sibling = true; });
    require(slot->session.next_position == position && !slot->session.sibling,
        "startup admission mutated the live session");
    published.push_back(std::move(slot));
}
void failed_admission() {
    Counters counts;
    Slots slots;
    bool caught = false;
    try { build_slot(slots, counts, PrefillConstruction::startup, true); }
    catch (const std::bad_alloc&) { caught = true; }
    require(caught && slots.empty() && counts.owners == 0 && counts.preflights == 1,
        "failed admission must propagate and destroy the unpublished slot");
    // A new construction after the transient failure must bind a healthy owner.
    build_slot(slots, counts, PrefillConstruction::startup);
    slots.back()->backend.execute(8192);
    require(counts.executions == 1 && counts.preflights == 2,
        "fresh owner did not execute using its successful admission");
}
void lazy_admission() {
    Counters counts;
    Slots slots;
    build_slot(slots, counts, PrefillConstruction::on_demand, true);
    require(counts.preflights == 0 && counts.handoffs == 0,
        "lazy growth eagerly prepared a full tile");
    slots.back()->backend.execute(16);
    require(counts.preflights == 0, "short request needed full-tile admission");
    build_slot(slots, counts, PrefillConstruction::on_demand);
    slots.back()->backend.execute(8192);
    require(counts.preflights == 1 && counts.executions == 2,
        "lazy full block lost execution-time admission");
    build_slot(slots, counts, PrefillConstruction::startup, true, false);
    build_slot(slots, counts, PrefillConstruction::startup, true, true, 16);
    require(counts.preflights == 1 && counts.handoffs == 0,
        "disabled or noninitial owner ran startup admission");
}
void pipeline_mask() {
    const auto policy = strix::qpf2k::admitted_organization(strix::qpf2k::PrefillOwner::resident_server);
    require((policy.pipe_mask & 0xffU) == 0xffU, "current serving bits changed");
    require((policy.pipe_mask & ~0xffU) == 0U, "future pipeline bits automatically admitted");
}
} // namespace
int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected test case");
        const std::string_view test = argv[1];
        if (test == "failure") failed_admission();
        else if (test == "lazy") lazy_admission();
        else if (test == "pipeline") pipeline_mask();
        else throw std::runtime_error("unknown test case");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
