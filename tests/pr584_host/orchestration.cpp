#include "strix/deepseek41_prefill.hpp"
#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using PrefillOptions = strix::deepseek41::PrefillOptions;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
namespace ds41 {
int process_wide_writes = 0;  // DAYAUDIT 2: a precision change of one object must not write the process-wide modes
void set_fast_dense(int) { ++process_wide_writes; }
void set_fast_moe(int) { ++process_wide_writes; }
void set_fast_engram(int) { ++process_wide_writes; }
}
namespace attn {
constexpr unsigned kWindow = 128, kHeadDim = 512;
namespace hip {
struct Mirror {
    const std::uint16_t* src;
    std::uint16_t* dst;
    const std::uint64_t* pos;
    unsigned count, capacity;
    bool window;
};
int mirror_stream = -1;
void enqueue_attention_flash_mirror(Mirror v, int stream) {
    require(v.window, "window mirror");
    mirror_stream = stream;
    for (unsigned i = 0; i < v.count; ++i) {
        const auto slot = v.pos[i] % v.capacity;
        v.dst[slot] = v.src[slot];
    }
}
}
}
int allocations = 0, fail_after = -1;
namespace ds41 { namespace attn = ::attn; }
template<class T> struct Dev {
    std::unique_ptr<T[]> owner;
    T* p = nullptr;
    std::size_t n = 0;
    Dev() = default;
    explicit Dev(std::size_t count) : n(count) {
        if (allocations++ == fail_after) throw std::bad_alloc();
        owner = std::make_unique<T[]>(std::max(count, std::size_t{1}));
        p = owner.get();
    }
    void zero(int) { std::fill_n(p, n, T{}); }
};
struct DevicePrefill {
    struct Impl {
        PrefillOptions options;
        bool task = false, broken = false, flash_run = true;
        Dev<std::uint16_t> ring16, kv16;
        struct Role { unsigned ratio; };
        std::vector<Role> roles{{0}, {8}, {2}};
        unsigned prefill_capacity = 4096;
        int stream = 7;
        bool fast_attn_at(unsigned) const { return options.fast_attention && !options.bf16_attention; }
        struct SeqState { unsigned prefill_capacity = 0; Dev<std::uint16_t> ring16, kv16; };
        std::vector<SeqState> group;  // group-pass members (their own flash mirrors)
        int dense_rebuilds = 0;
        void rebuild_dense_classes() { ++dense_rebuilds; }
#include "methods.inc"
    };
    std::unique_ptr<Impl> impl_ = std::make_unique<Impl>();
    void set_precision_for_new_sequence(bool, bool);
    void set_precision_for_new_sequence(const PrefillOptions&);
};
#include "setter.inc"

struct Entries {
    const std::uint64_t* ring_pos = nullptr;
    unsigned ring_rows = 0;
    const std::uint64_t* comp_pos = nullptr;
    unsigned comp_rows = 0;
    std::uint32_t* packed = nullptr;
    std::uint16_t* ring = nullptr;
    unsigned capacity = 0;
};
void unpack_entries(const Entries& v) {
    for (unsigned i = 0; i < v.ring_rows; ++i)
        v.ring[v.ring_pos[i] % v.capacity] = static_cast<std::uint16_t>(v.packed[i]);
}
int hipGetLastError() { return 0; }
void check(int error, const char*) { require(error == 0, "launch error"); }
void ring_test() {
    // Two ownership windows per rank, wrapping the ring; a query crosses each boundary.
    for (unsigned rank = 0; rank < 2; ++rank) for (bool staged : {false, true}) {
        struct Fixture {
            Dev<std::uint16_t> prefill_ring{8}, ring16{8};
            Dev<std::uint64_t> ppos64{4};
            unsigned prefill_capacity = 8;
            bool flash_attn_at(unsigned) const { return true; }
        } m;
        std::uint32_t ordinary[4]{}, staging[4]{};
        unsigned j = 0;
        for (unsigned pos = 8; pos < 16; ++pos) {
            const auto value = static_cast<std::uint16_t>(pos + 10U);
            if ((pos / 2U) % 2U == rank) m.prefill_ring.p[pos % 8U] = m.ring16.p[pos % 8U] = value;
            else { m.ppos64.p[j] = pos; ordinary[j] = staging[j] = value; ++j; }
        }
        using FinalEntries = Entries;
        const unsigned nseg = 1;
        struct Section { unsigned off; };
        const Section sec_pe[]{{0}};
        const auto seq_ring = [&](unsigned) -> auto& { return m.prefill_ring; };
        const auto seq_capacity = [&](unsigned) { return m.prefill_capacity; };
        const auto seq_ring16 = [&](unsigned) { return m.ring16.p; };
        const auto entries_of = [&](unsigned, Section, const std::uint64_t* pos, std::uint32_t* packed) {
            Entries fe;
            fe.ring = m.prefill_ring.p;
            fe.capacity = m.prefill_capacity;
            fe.ring_pos = pos;
            fe.ring_rows = 4;
            fe.packed = packed;
            return fe;
        };
        [[maybe_unused]] const unsigned layer = 0;
        const int st = 7;
        const auto* recv_words = staged ? staging : ordinary;
#include "unpack.inc"
        require(attn::hip::mirror_stream == st, "peer mirror must use publication stream");
        for (unsigned slot = 0; slot < 8; ++slot)
            require(m.ring16.p[slot] == m.prefill_ring.p[slot] && m.ring16.p[slot] != 0,
                    "ownership boundary reads a stale peer mirror");
    }
}
void precision_test() {
    DevicePrefill pre;
    auto& m = *pre.impl_;
    m.group.resize(2);
    m.group[0].prefill_capacity = 512U;
    m.group[1].prefill_capacity = 768U;
    PrefillOptions p;
    p.fast_attention = p.fast_attention_flash = true;
    pre.set_precision_for_new_sequence(p);
    require(m.ring16.p && m.kv16.p && m.flash_attn_at(0), "new flash arm did not prepare mirrors");
    require(m.ring16.n == std::size_t{m.prefill_capacity} * attn::kHeadDim, "ring allocation size");
    const auto rows = std::size_t{m.prefill_capacity} - m.options.window_rows - attn::kWindow;
    require(m.kv16.n == (rows / 2U + 1U) * attn::kHeadDim, "densest compressed allocation size");
    const int allocated = allocations;
    std::vector<const std::uint16_t*> member_buffers;
    for (const auto& member : m.group) {
        require(member.ring16.p && member.kv16.p && member.ring16.p != m.ring16.p && member.kv16.p != m.kv16.p,
                "member mirrors allocated independently of solo");
        require(member.ring16.n == std::size_t{member.prefill_capacity} * attn::kHeadDim, "member ring capacity");
        const auto member_rows = std::size_t{member.prefill_capacity} - m.options.window_rows - attn::kWindow;
        require(member.kv16.n == (member_rows / 2U + 1U) * attn::kHeadDim, "member compressed capacity");
        member_buffers.push_back(member.ring16.p);
        member_buffers.push_back(member.kv16.p);
    }
    require(member_buffers[0] != member_buffers[2] && member_buffers[1] != member_buffers[3], "members own distinct mirrors");
    pre.set_precision_for_new_sequence(PrefillOptions{});
    pre.set_precision_for_new_sequence(p);
    require(allocations == allocated && m.flash_attn_at(0), "mirror reuse between arms");
    for (std::size_t i = 0; i < m.group.size(); ++i)
        require(m.group[i].ring16.p == member_buffers[2U * i] && m.group[i].kv16.p == member_buffers[2U * i + 1U],
                "member mirror reuse between arms");
    PrefillOptions fast = p;
    fast.fast_dense = fast.fast_moe = true;
    pre.set_precision_for_new_sequence(fast);
    require(ds41::process_wide_writes == 0 && m.options.fast_dense && m.options.fast_moe,
            "precision is the object's own (no process-wide FAST dense / MoE write)");
    DevicePrefill failed;
    fail_after = allocations + 1; // second allocation fails: do not commit either buffer or precision
    bool threw = false;
    try { failed.set_precision_for_new_sequence(p); } catch (const std::bad_alloc&) { threw = true; }
    require(threw && !failed.impl_->options.fast_attention && !failed.impl_->ring16.p, "failed transition committed");
    fail_after = -1;
    failed.impl_->task = true;
    threw = false;
    try { failed.set_precision_for_new_sequence(p); } catch (const std::logic_error&) { threw = true; }
    require(threw && !failed.impl_->ring16.p, "task guard must precede allocation");
}
void cancellation_test() {
    // Cancellation witness retained alongside the production part-orchestration test.
    volatile float large = 16777216.0F, one = 1.0F;
    volatile float fused_first = large + one;
    const float fused = fused_first - large;
    volatile float second_half = one - large;
    const float separate = large + second_half;
    require(fused == 0.0F && separate == 1.0F, "cancellation witness");
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "test name required");
        const std::string name = argv[1];
        if (name == "ring") ring_test();
        else if (name == "precision") precision_test();
        else if (name == "cancellation") cancellation_test();
        else throw std::runtime_error("unknown test");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
