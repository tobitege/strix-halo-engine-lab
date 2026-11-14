// Provided embedding rows of a prefill run: resolve_embed_run and EmbedSplice (session/embed_splice.hpp).

#include "strix/session/embed_splice.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace strix::session;

namespace {
int failures = 0;
void check(bool ok, const char* what, int line) {
    if (ok) return;
    ++failures;
    std::cerr << "FAIL line " << line << ": " << what << "\n";
}
#define CHECK(x) check((x), #x, __LINE__)

template <class F> bool refused(F&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
    }
    return false;
}

EmbedRows span_of(std::uint8_t tag, std::uint32_t rows, std::uint32_t width, std::uint32_t token) {
    EmbedRows r;
    r.digest[0] = tag;
    r.rows = rows;
    r.width = width;
    r.real_token_id = token;
    r.data.resize(static_cast<std::size_t>(rows) * width);
    for (std::size_t i = 0; i < r.data.size(); ++i) r.data[i] = static_cast<std::uint16_t>(tag * 256U + i);
    return r;
}

struct Fixture {
    EmbedInterner interner;
    EmbedLease a = interner.intern(span_of(1, 4, 8, 900));
    EmbedLease b = interner.intern(span_of(2, 2, 8, 901));
    std::shared_ptr<const EmbedTable> table = std::make_shared<const EmbedTable>(std::vector<EmbedLease>{a, b});
    EmbedResolveRules rules{8, true, false};
};

void text_is_untouched() {
    Fixture f;
    const std::vector<std::uint32_t> text{5, 6, 7};
    const auto with = resolve_embed_run(text, f.table, f.rules);
    const auto without = resolve_embed_run(text, nullptr, f.rules);
    CHECK(with.tokens == text && with.splice.empty() && with.splice.keep == nullptr);
    CHECK(without.tokens == text && without.splice.empty());
    with.splice.validate(3, 8);
    EmbedSplice none;
    none.validate(0, 5120); // an empty splice fits every run
    CHECK(none.rows() == 0 && none.extras_rows(4) == std::vector<std::uint8_t>(4, 0));
}

void spans_become_segments() {
    Fixture f;
    const std::uint32_t ha = f.a.handle(), hb = f.b.handle();
    // text, span a whole, text, span b whole, span a again (same image twice), text
    const std::vector<std::uint32_t> ids{11, embed_id(ha, 0), embed_id(ha, 1), embed_id(ha, 2), embed_id(ha, 3), 12,
                                         embed_id(hb, 0), embed_id(hb, 1), embed_id(ha, 0), embed_id(ha, 1),
                                         embed_id(ha, 2), embed_id(ha, 3), 13};
    const auto r = resolve_embed_run(ids, f.table, f.rules);
    CHECK((r.tokens == std::vector<std::uint32_t>{11, 900, 900, 900, 900, 12, 901, 901, 900, 900, 900, 900, 13}));
    const auto& s = r.splice;
    CHECK(s.segments.size() == 3 && s.rows() == 10 && s.width == 8 && s.format == EmbedRowFormat::bf16);
    CHECK(s.segments[0].run_row == 1 && s.segments[0].rows == 4 && s.segments[0].data == f.a.rows().data.data());
    CHECK(s.segments[1].run_row == 6 && s.segments[1].rows == 2 && s.segments[1].data == f.b.rows().data.data());
    CHECK(s.segments[2].run_row == 8 && s.segments[2].rows == 4 && s.segments[2].data == f.a.rows().data.data());
    CHECK(s.segments[0].extras && s.keep == f.table);
    s.validate(ids.size(), 8);
    CHECK((s.extras_rows(ids.size()) == std::vector<std::uint8_t>{0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0}));
    CHECK(embed_splice_staging_bytes(s.rows(), s.width) == 10U * 8U * 2U);

    // a run may start inside a span (a prefix match that ends in an image, or a piece boundary)
    const std::vector<std::uint32_t> tail{embed_id(ha, 2), embed_id(ha, 3), 12};
    const auto t = resolve_embed_run(tail, f.table, f.rules);
    CHECK(t.splice.segments.size() == 1 && t.splice.segments[0].run_row == 0 && t.splice.segments[0].rows == 2);
    CHECK(t.splice.segments[0].data == f.a.rows().row(2).data());
}

void refusals() {
    Fixture f;
    const std::uint32_t ha = f.a.handle();
    const std::vector<std::uint32_t> one{3, embed_id(ha, 0)};
    CHECK(refused([&] { (void)resolve_embed_run(one, nullptr, f.rules); }));                  // no table
    CHECK(refused([&] { (void)resolve_embed_run(std::vector<std::uint32_t>{embed_id(ha, 4)}, f.table, f.rules); })); // row
    CHECK(refused([&] { (void)resolve_embed_run(std::vector<std::uint32_t>{embed_id(77, 0)}, f.table, f.rules); })); // handle
    EmbedResolveRules wide = f.rules;
    wide.width = 16;
    CHECK(refused([&] { (void)resolve_embed_run(one, f.table, wide); }));
    // prompt-index positions: rows out of order, a gap, a span entered in the middle after text
    const std::vector<std::uint32_t> swapped{embed_id(ha, 1), embed_id(ha, 0)};
    const std::vector<std::uint32_t> gap{embed_id(ha, 0), embed_id(ha, 2)};
    const std::vector<std::uint32_t> middle{3, embed_id(ha, 1)};
    CHECK(!refused([&] { (void)resolve_embed_run(swapped, f.table, f.rules); })); // row 0 may follow anything
    CHECK(refused([&] { (void)resolve_embed_run(gap, f.table, f.rules); }));
    CHECK(refused([&] { (void)resolve_embed_run(middle, f.table, f.rules); }));
    EmbedResolveRules free_positions = f.rules;
    free_positions.prompt_index_positions = false;
    CHECK(resolve_embed_run(gap, f.table, free_positions).splice.segments.size() == 2);

    // stored positions must be the row index; deep rows need a family that takes them
    EmbedInterner interner;
    EmbedRows pos = span_of(3, 3, 8, 900);
    pos.axes = 1;
    pos.lm_pos = {0, 1, 2};
    EmbedRows bad = span_of(4, 3, 8, 900);
    bad.axes = 1;
    bad.lm_pos = {0, 2, 4};
    EmbedRows deep = span_of(5, 2, 8, 900);
    deep.deep.push_back({3, std::vector<std::uint16_t>(16, 0)});
    const auto lp = interner.intern(std::move(pos)), lb = interner.intern(std::move(bad)), ld = interner.intern(std::move(deep));
    const auto table = std::make_shared<const EmbedTable>(std::vector<EmbedLease>{lp, lb, ld});
    CHECK(resolve_embed_run(lp.ids(), table, f.rules).splice.rows() == 3);
    CHECK(refused([&] { (void)resolve_embed_run(lb.ids(), table, f.rules); }));
    CHECK(refused([&] { (void)resolve_embed_run(ld.ids(), table, f.rules); }));
    EmbedResolveRules with_deep = f.rules;
    with_deep.allow_deep = true;
    CHECK(resolve_embed_run(ld.ids(), table, with_deep).splice.rows() == 2);
}

void splice_validation() {
    const std::vector<std::uint16_t> data(4 * 8, 0);
    EmbedSplice s;
    s.width = 8;
    s.segments.push_back({2, 2, data.data(), false});
    s.segments.push_back({4, 2, data.data() + 16, true});
    s.validate(6, 8);
    CHECK((s.extras_rows(6) == std::vector<std::uint8_t>{0, 0, 0, 0, 1, 1}));
    CHECK(refused([&] { s.validate(5, 8); }));  // past the run
    CHECK(refused([&] { s.validate(6, 16); })); // width
    EmbedSplice overlap = s;
    overlap.segments[1].run_row = 3;
    CHECK(refused([&] { overlap.validate(6, 8); }));
    EmbedSplice empty_segment = s;
    empty_segment.segments[0].rows = 0;
    CHECK(refused([&] { empty_segment.validate(6, 8); }));
    EmbedSplice null_data = s;
    null_data.segments[0].data = nullptr;
    CHECK(refused([&] { null_data.validate(6, 8); }));
}
} // namespace

int main() {
    text_is_untouched();
    spans_become_segments();
    refusals();
    splice_validation();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "session embed splice tests passed\n";
    return 0;
}
