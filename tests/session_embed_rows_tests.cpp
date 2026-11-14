// Embedding rows in the token stream: id packing, interner (refcount, revival, epoch),
// per-request table, cross-process remap, placeholder expansion, and the prefix
// directory over prompts that hold image runs.
#include "strix/serve/prefix_directory.hpp"
#include "strix/session/embed_rows.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace strix::session;

static_assert(kEmbedHandleBits + kEmbedRowBits == 31);
static_assert(!is_embed_id(0x7FFF'FFFFU) && is_embed_id(0x8000'0000U));
static_assert(embed_id(0, 0) == 0x8000'0000U);
static_assert(embed_id(kEmbedMaxHandles - 1, kEmbedMaxRows - 1) == 0xFFFF'FFFFU);
static_assert(embed_handle(embed_id(0x1ABCD, 0x2EF1)) == 0x1ABCD && embed_row(embed_id(0x1ABCD, 0x2EF1)) == 0x2EF1);
static_assert(embed_format_id(42, 0) == 42 && embed_format_id(42, 1) != 42 &&
              embed_format_id(42, 1) != embed_format_id(42, 2));

namespace {
int failures = 0;
void check(bool ok, const char* what, int line) {
    if (ok) return;
    ++failures;
    std::cerr << "FAIL line " << line << ": " << what << "\n";
}
#define CHECK(x) check((x), #x, __LINE__)
template <class Error, class F> bool throws(F&& call) {
    try {
        call();
    } catch (const Error&) {
        return true;
    } catch (...) {
    }
    return false;
}

EmbedRows span_of(std::uint8_t tag, std::uint32_t rows, std::uint32_t width = 4) {
    EmbedRows r;
    r.digest[0] = tag;
    r.digest[31] = static_cast<std::uint8_t>(tag ^ 0x5A);
    r.rows = rows;
    r.width = width;
    r.real_token_id = 100;
    r.data.resize(static_cast<std::size_t>(rows) * width);
    for (std::size_t i = 0; i < r.data.size(); ++i) r.data[i] = static_cast<std::uint16_t>(tag * 256U + i);
    return r;
}

void ids() {
    const std::vector<std::uint32_t> text{1, 2, 0x7FFF'FFFFU};
    const std::vector<std::uint32_t> mixed{1, embed_id(3, 0), embed_id(3, 1), 2};
    CHECK(count_embed_ids(text) == 0 && !has_embed_ids(text));
    CHECK(count_embed_ids(mixed) == 2 && has_embed_ids(mixed));
    reject_embed_ids(text);
    reject_embed_ids({});
    CHECK(throws<std::invalid_argument>([&] { reject_embed_ids(mixed); }));
}

void intern() {
    EmbedInterner interner;
    auto a = interner.intern(span_of(1, 3));
    CHECK(a && a.handle() == 0 && a.rows().rows == 3);
    CHECK(a.rows().role.size() == 3 && a.rows().role[2] == EmbedRole::content);
    CHECK(a.rows().row(1).size() == 4 && a.rows().row(1)[0] == 256 + 4);
    CHECK((a.ids() == std::vector<std::uint32_t>{embed_id(0, 0), embed_id(0, 1), embed_id(0, 2)}));

    // Same digest: same handle and the first payload, whatever the second call carries.
    auto again = span_of(1, 3);
    again.data[0] = 0xFFFF;
    auto a2 = interner.intern(std::move(again));
    CHECK(a2.handle() == a.handle() && &a2.rows() == &a.rows() && a2.rows().data[0] == 256);
    auto b = interner.intern(span_of(2, 5));
    CHECK(b.handle() == 1);
    auto st = interner.status();
    CHECK(st.live == 2 && st.retired == 0 && st.interned == 2 && st.shared == 1 && st.epoch == 0);
    CHECK(st.live_bytes == a.rows().bytes() + b.rows().bytes());
    CHECK(throws<std::invalid_argument>([&] { (void)interner.intern(span_of(1, 4)); })); // live, other shape

    // Malformed spans change nothing.
    CHECK(throws<std::invalid_argument>([&] { (void)interner.intern(span_of(9, 0)); }));
    CHECK(throws<std::invalid_argument>([&] { (void)interner.intern(span_of(9, kEmbedMaxRows + 1, 1)); }));
    CHECK(throws<std::invalid_argument>([&] { auto r = span_of(9, 2); r.width = 0; (void)interner.intern(std::move(r)); }));
    CHECK(throws<std::invalid_argument>([&] { auto r = span_of(9, 2); r.data.pop_back(); (void)interner.intern(std::move(r)); }));
    CHECK(throws<std::invalid_argument>([&] { auto r = span_of(9, 2); r.role.resize(1); (void)interner.intern(std::move(r)); }));
    CHECK(throws<std::invalid_argument>([&] { auto r = span_of(9, 2); r.axes = 3; r.lm_pos.resize(5); (void)interner.intern(std::move(r)); }));
    CHECK(throws<std::invalid_argument>([&] { auto r = span_of(9, 2); r.real_token_id = embed_id(0, 0); (void)interner.intern(std::move(r)); }));
    CHECK(throws<std::invalid_argument>([&] { auto r = span_of(9, 2); r.deep.push_back({8, {1}}); (void)interner.intern(std::move(r)); }));
    CHECK(interner.status().live == 2 && interner.status().interned == 2);
    {
        auto r = span_of(9, 2);
        r.axes = 3;
        r.lm_pos.assign(6, 1);
        r.role = {EmbedRole::begin, EmbedRole::end};
        r.deep.push_back({8, std::vector<std::uint16_t>(8, 1)});
        auto full = interner.intern(std::move(r));
        CHECK(full.handle() == 2 && full.rows().role[1] == EmbedRole::end);
    }
    CHECK(throws<std::invalid_argument>([] { EmbedInterner bad(0); }));
    CHECK(throws<std::invalid_argument>([] { EmbedInterner bad(kEmbedMaxHandles + 1); }));
}

void refcount() {
    EmbedInterner interner;
    const auto digest = span_of(1, 3).digest;
    std::uint32_t handle = 0;
    {
        auto a = interner.intern(span_of(1, 3));
        handle = a.handle();
        auto copy = a;            // copy shares
        auto moved = std::move(a); // move transfers
        CHECK(!a && moved && copy);
        auto found = interner.find(digest);
        CHECK(found && found.handle() == handle);
        EmbedLease assigned;
        assigned = copy;
        CHECK(assigned && interner.status().live == 1);
    }
    // Every lease gone: payload dropped, handle reserved for the digest.
    auto st = interner.status();
    CHECK(st.live == 0 && st.retired == 1 && st.live_bytes == 0);
    CHECK(!interner.find(digest));
    auto other = interner.intern(span_of(2, 1));
    CHECK(other.handle() != handle);
    auto revived = interner.intern(span_of(1, 3));
    CHECK(revived.handle() == handle && revived.rows().data[0] == 256);
    st = interner.status();
    CHECK(st.live == 2 && st.retired == 0 && st.interned == 3 && st.epoch == 0);

    // A lease outlives the interner object.
    EmbedLease survivor;
    {
        EmbedInterner local;
        survivor = local.intern(span_of(7, 2));
    }
    CHECK(survivor && survivor.rows().rows == 2);
    survivor = EmbedLease{};
    CHECK(!survivor);
}

void epoch() {
    EmbedInterner interner(4);
    auto keep = interner.intern(span_of(1, 1)); // handle 0 stays leased
    { auto t = interner.intern(span_of(2, 1)); CHECK(t.handle() == 1); }
    auto keep2 = interner.intern(span_of(3, 1)); // handle 2
    { auto t = interner.intern(span_of(4, 1)); CHECK(t.handle() == 3); }
    CHECK(interner.epoch() == 0 && interner.status().retired == 2);
    { auto t = interner.intern(span_of(2, 1)); CHECK(t.handle() == 1); } // still reserved, no bump
    CHECK(interner.epoch() == 0);

    // A fifth digest wraps the space: unleased handles are freed, leased ones keep theirs.
    auto e = interner.intern(span_of(5, 1));
    CHECK(interner.epoch() == 1 && e.handle() == 1);
    CHECK(keep.handle() == 0 && keep2.handle() == 2 && interner.find(span_of(1, 1).digest).handle() == 0);
    auto st = interner.status();
    CHECK(st.live == 3 && st.retired == 0);
    // Digest 2 lost its reservation: it now gets the other freed handle, not its old one.
    auto b = interner.intern(span_of(2, 1));
    CHECK(b.handle() == 3 && interner.epoch() == 1);
    CHECK(throws<std::length_error>([&] { (void)interner.intern(span_of(6, 1)); })); // all four leased
    CHECK(interner.status().live == 4 && interner.epoch() == 1);
    b = EmbedLease{};
    auto f = interner.intern(span_of(6, 1)); // second wrap
    CHECK(f.handle() == 3 && interner.epoch() == 2);
}

void table() {
    EmbedInterner interner;
    auto a = interner.intern(span_of(1, 3));
    auto b = interner.intern(span_of(2, 5));
    auto c = interner.intern(span_of(3, 2));
    const EmbedTable none;
    CHECK(none.empty() && none.find(0) == nullptr);
    none.validate(std::vector<std::uint32_t>{1, 2, 3});
    CHECK(throws<std::invalid_argument>([&] { none.validate(std::vector<std::uint32_t>{embed_id(0, 0)}); }));

    const EmbedTable t({b, a, b}); // order free, duplicates folded
    CHECK(t.size() == 2 && t.find(a.handle()) == &a.rows() && t.find(b.handle()) == &b.rows());
    CHECK(t.find(c.handle()) == nullptr);
    std::vector<std::uint32_t> prompt{10, 11};
    for (auto id : a.ids()) prompt.push_back(id);
    prompt.push_back(12);
    for (auto id : b.ids()) prompt.push_back(id);
    t.validate(prompt);
    auto bad = prompt;
    bad.push_back(embed_id(a.handle(), 3));
    CHECK(throws<std::invalid_argument>([&] { t.validate(bad); }));
    bad.back() = embed_id(c.handle(), 0);
    CHECK(throws<std::invalid_argument>([&] { t.validate(bad); }));
    CHECK(throws<std::invalid_argument>([] { EmbedTable broken({EmbedLease{}}); }));

    // The table alone keeps its spans alive.
    const auto digest = a.rows().digest;
    auto owned = std::make_shared<const EmbedTable>(std::vector<EmbedLease>{a});
    a = EmbedLease{};
    CHECK(static_cast<bool>(interner.find(digest)));
    CHECK(t.find(0) != nullptr);

    // Another process holds the same spans under other handles.
    EmbedInterner peer;
    auto filler = peer.intern(span_of(8, 1));
    auto pb = peer.intern(span_of(2, 5));
    auto pa = peer.intern(span_of(1, 3));
    const EmbedTable remote({pa, pb});
    const auto manifest = t.manifest();
    CHECK(manifest.size() == 2 && manifest[0].handle == 0 && manifest[0].rows == 3 && manifest[1].rows == 5);
    const auto mapped = remap_embed_ids(prompt, manifest, remote);
    CHECK(mapped.size() == prompt.size() && mapped[0] == 10 && mapped[5] == 12);
    CHECK(mapped[2] == embed_id(pa.handle(), 0) && mapped[4] == embed_id(pa.handle(), 2));
    CHECK(mapped[6] == embed_id(pb.handle(), 0) && mapped[10] == embed_id(pb.handle(), 4));
    remote.validate(mapped);
    CHECK(remap_embed_ids(mapped, remote.manifest(), t) == prompt); // and back
    const EmbedTable partial({pa});
    CHECK(throws<std::invalid_argument>([&] { (void)remap_embed_ids(prompt, manifest, partial); }));
    CHECK(throws<std::invalid_argument>([&] { (void)remap_embed_ids(bad, manifest, remote); }));
    const std::vector<std::uint32_t> text{1, 2, 3};
    CHECK(remap_embed_ids(text, {}, none) == text);
}

void expand() {
    const std::uint32_t ph = 900;
    const std::vector<std::uint32_t> text{1, 2, 3, 4};
    const auto same = expand_embed_rows(text, ph, {});
    CHECK(same.tokens == text && same.runs.empty() && same.map_offset(3) == 3); // text only: identical ids

    EmbedInterner interner;
    auto a = interner.intern(span_of(1, 3));
    auto b = interner.intern(span_of(2, 2));
    // First span: real start/end tokens around the rows. Second: rows only.
    std::vector<std::vector<std::uint32_t>> layouts(2);
    layouts[0] = {50};
    for (auto id : a.ids()) layouts[0].push_back(id);
    layouts[0].push_back(51);
    layouts[1] = b.ids();
    const std::vector<std::uint32_t> rendered{1, ph, 2, 3, ph, 4};
    const auto out = expand_embed_rows(rendered, ph, layouts);
    const std::vector<std::uint32_t> want{1, 50, embed_id(0, 0), embed_id(0, 1), embed_id(0, 2), 51,
                                          2, 3, embed_id(1, 0), embed_id(1, 1), 4};
    CHECK(out.tokens == want && out.runs.size() == 2);
    CHECK(out.runs[0].source == 1 && out.runs[0].offset == 1 && out.runs[0].size == 5 && out.runs[0].rows == 3);
    CHECK(out.runs[1].source == 4 && out.runs[1].offset == 8 && out.runs[1].size == 2 && out.runs[1].rows == 2);
    CHECK(out.map_offset(0) == 0 && out.map_offset(1) == 1 && out.map_offset(2) == 6 && out.map_offset(4) == 8);
    CHECK(out.map_offset(5) == 10 && out.map_offset(6) == 11);
    EmbedTable({a, b}).validate(out.tokens);

    CHECK(throws<std::invalid_argument>([&] { (void)expand_embed_rows(rendered, ph, {}); }));
    CHECK(throws<std::invalid_argument>([&] { (void)expand_embed_rows(text, ph, layouts); }));
    CHECK(throws<std::invalid_argument>([&] { (void)expand_embed_rows(out.tokens, ph, {}); })); // already expanded
    CHECK(throws<std::invalid_argument>([&] { (void)expand_embed_rows(text, embed_id(0, 0), {}); }));
    auto broken = layouts;
    broken[1].clear();
    CHECK(throws<std::invalid_argument>([&] { (void)expand_embed_rows(rendered, ph, broken); }));
    broken[1] = {ph};
    CHECK(throws<std::invalid_argument>([&] { (void)expand_embed_rows(rendered, ph, broken); }));
}

// Prompts with image runs through the unchanged prefix directory.
void prefix_directory() {
    namespace sv = strix::serve;
    const std::uint32_t ph = 900;
    EmbedInterner interner;
    const auto prompt_with = [&](std::uint8_t image, std::uint32_t tail) {
        auto lease = interner.intern(span_of(image, 12));
        std::vector<std::uint32_t> rendered{1, 2, 3, 4, ph, 5, 6, 7, 8, tail, tail + 1};
        const std::vector<std::vector<std::uint32_t>> layouts{lease.ids()};
        return expand_embed_rows(rendered, ph, layouts).tokens; // 4 text, 12 rows, 6 text
    };
    const sv::PrefixKey key{1, 2, embed_format_id(3, 77)};
    sv::PrefixDirectory directory(4, 8);

    // Sealed behind the image: the same image in another conversation forks there.
    const auto first = prompt_with(1, 60);
    const auto entry = directory.announce(key, first, 0);
    CHECK(entry != 0);
    directory.publish(entry, 20);
    const auto second = prompt_with(1, 70); // leases were dropped in between: same ids again
    auto hit = directory.lookup(key, second);
    CHECK(hit.entry == entry && hit.tokens == 20);
    // Another image is another id run: nothing to share at that frontier.
    const auto other = prompt_with(2, 70);
    CHECK(directory.lookup(key, other).entry == 0);
    // Another producer of the rows is another key.
    CHECK(directory.lookup(sv::PrefixKey{1, 2, embed_format_id(3, 78)}, second).entry == 0);
    CHECK(directory.lookup(sv::PrefixKey{1, 2, 3}, second).entry == 0);

    // A frontier inside the image run is a valid fork point (the row index is in the id).
    sv::PrefixDirectory inside(4, 8);
    const auto mid = inside.announce(key, first, 0);
    inside.publish(mid, 12); // 4 text ids + 8 of the 12 rows
    hit = inside.lookup(key, second);
    CHECK(hit.entry == mid && hit.tokens == 12 && is_embed_id(second[11]) && is_embed_id(second[12]));
    CHECK(inside.lookup(key, other).entry == 0);
}
} // namespace

int main() {
    ids();
    intern();
    refcount();
    epoch();
    table();
    expand();
    prefix_directory();
    if (failures) return 1;
    std::cout << "session embed rows: ok\n";
    return 0;
}
