// strix/block_summary_state.hpp: closed-block summaries + open tail (host shared), against the MINSTATE formulas.
#include "strix/block_summary_state.hpp"

#include <cstdint>
#include <cstdio>

namespace {
int failures = 0, checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}
}  // namespace

int main() {
    namespace st = strix::state;
    // Row windows at every residue of the block, and the empty state.
    for (std::uint64_t p = 0; p < 64; ++p) {
        const auto t = st::open_tail(p, 4);
        check(t.first == p - p % 4 && t.rows == p % 4 && t.end() == p, "open_tail = [P - P % 4, P)");
        check(st::raw_window(st::RawRows::full, p, 4) == st::RowWindow{0, p}, "full window = [0, P)");
        check(st::closed_blocks(p, 4) == p / 4, "closed blocks = P / 4");
        check(st::canonical_raw_window(p, 4) == t, "canonical raw rows = the open block only");
        for (std::uint64_t r = 0; r < p; ++r)
            check(st::is_hole(st::RawRows::open_tail, r, p, 4) == (r < p - p % 4) && !st::is_hole(st::RawRows::full, r, p, 4),
                  "holes = the closed blocks' raw rows, open_tail form only");
    }
    check(st::last_rows(10, 4) == st::RowWindow{6, 4} && st::last_rows(3, 4) == st::RowWindow{0, 3}, "ring window = last W rows");
    // GLM DSA index (ikey + igate BF16 [128], pooled FP32 [128] per block of 4): 1024P + 512 floor(P/4) + 512 (P % 4) per
    // layer with the latent; the index part alone is 512 (P % 4) + 512 floor(P/4).
    const st::BlockSummaryGeometry glm{4, 512, 512, st::SummaryDtype::fp32};
    for (std::uint64_t p : {32767ULL, 131071ULL, 4ULL, 5ULL, 6ULL, 7ULL}) {
        check(st::state_bytes(glm, st::RawRows::open_tail, p) == 512 * (p % 4) + 512 * (p / 4), "GLM index minimum per layer");
        check(st::state_bytes(glm, st::RawRows::full, p) == 512 * p + 512 * (p / 4), "GLM index today per layer");
        check(st::open_tail_saving(glm, p) == 512 * (p - p % 4), "GLM saving per layer");
    }
    check(12 * st::open_tail_saving(glm, 131071) == 805281792ULL, "GLM 128K: 805.28 MB saved (12 DSA layers incl. MTP)");
    // Qwen QSA index (raw IK BF16 [128] = 256 B/row; pool BF16 bits [128] = 256 B per block of 4: 64 B/token):
    // 13 QSA (12 + MTP) x 192 B = 2,496 B/token less: 29,952 -> 27,456 B/token.
    const st::BlockSummaryGeometry qwen{4, 256, 256, st::SummaryDtype::bf16_in_fp32};
    // Qwen today sends the raw keys only (the consumer rebuilds the pool): raw_bytes(full) vs summaries + open tail.
    check(st::raw_bytes(qwen, st::RawRows::full, 32768) - st::state_bytes(qwen, st::RawRows::open_tail, 32768) == 32768ULL * 192,
          "Qwen: 192 B/token/layer less (N multiple of 4)");
    check(13 * (st::raw_bytes(qwen, st::RawRows::full, 4) - st::state_bytes(qwen, st::RawRows::open_tail, 4)) / 4 == 2496,
          "Qwen: 2,496 B/token over 13 QSA layers");
    // Qwen single node, MTP, N multiple of 4 (MINSTATE): today 29,952 N - 2,304 + Fq, minimum 27,456 N - 1,536 + Fq - 737,280
    // (Fq = 116,400,144; the 737,280 = 36 GDN layers x one of four conv entries). Index part: 13 QSA layers.
    {
        const std::uint64_t Fq = 116400144ULL, N = 32768;
        const std::uint64_t other = 29952ULL * N - 2304 + Fq - 13 * st::raw_bytes(qwen, st::RawRows::full, N);  // KV, MTP, held, fixed
        const std::uint64_t minimum = other - 737280ULL + 13 * st::state_bytes(qwen, st::RawRows::open_tail, N) + 768;  // +768: the MTP QSA holds N - 1 rows (256 raw + 512 tail vs N rows)
        check(minimum == 27456ULL * N - 1536 + Fq - 737280ULL, "Qwen 32K minimum = 27,456 N - 1,536 + Fq - 737,280");
        check(minimum == 1015339536ULL, "Qwen 32K: 1,015.34 MB (MINSTATE table)");
    }
    // DS41 window ring (W = 128, capacity 136, 528 B/entry, 40 rings): only the last 127 entries are read again.
    check(40ULL * st::last_rows(136, 127).rows * 528 == 2682240ULL, "DS41 ring minimum 2,682,240 B (MINSTATE)");
    // A declined form falls back to full with an observable reason; a granted one carries none.
    const auto declined = st::resolve_raw_rows(st::RawRows::open_tail, "raw keys reread by block_score_kernel");
    check(declined.form == st::RawRows::full && !declined.reason.empty(), "decline -> full + reason");
    const auto granted = st::resolve_raw_rows(st::RawRows::open_tail, "");
    check(granted.form == st::RawRows::open_tail && granted.reason.empty(), "granted -> open_tail");
    bool threw = false;
    try { st::require_window_inside(st::RowWindow{8, 4}, 10, "test"); } catch (const std::out_of_range&) { threw = true; }
    check(threw, "a window beyond the capacity is refused");
    std::printf("block_summary_state: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
