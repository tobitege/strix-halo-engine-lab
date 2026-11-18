// DAYAUDIT 5 (P1): after run_group the single sequence is discarded (rows 0, local rows 0, last row "local"); the
// single-sequence export_state() then read idx_count at local_rows - 1 (out of bounds). The production export_state()
// is extracted verbatim; extract.py asserts the run_group / import transitions and the export_sequence guard.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

struct AttentionState {
    std::uint32_t position = 0;
};
struct DevicePrefill {
    struct Impl {
        bool state_incomplete = false;
        bool solo_discarded = false;
        std::uint32_t rows_done = 0, local_rows_done = 0;
        bool last_row_local = false;
        std::vector<int> attn;
    };
    std::unique_ptr<Impl> impl_ = std::make_unique<Impl>();
    AttentionState export_state() const;
};
static int exports = 0;
static AttentionState export_sequence(const DevicePrefill::Impl&, const std::vector<int>&, std::uint32_t T, std::uint32_t local_rows,
                                      bool last_local_row) {
    if (last_local_row && local_rows == 0U) throw std::runtime_error("out-of-bounds read of idx_count[local_rows - 1]");
    ++exports;
    return AttentionState{T};
}
#include "export_state.inc"

int main() try {
    DevicePrefill pre;
    auto& m = *pre.impl_;
    m.rows_done = 40, m.local_rows_done = 24, m.last_row_local = true;  // a solo run
    if (pre.export_state().position != 40U || exports != 1) throw std::runtime_error("solo export");
    // run_group's end of pass (src/hip_deepseek41_prefill.hip): the single sequence is dropped
    m.rows_done = 0, m.local_rows_done = 0, m.last_row_local = true, m.state_incomplete = false, m.solo_discarded = true;
    bool refused = false;
    try { (void)pre.export_state(); } catch (const std::logic_error&) { refused = true; }
    if (!refused || exports != 1) throw std::runtime_error("export_state() after run_group must refuse before reading");
    m.solo_discarded = false, m.rows_done = 64, m.local_rows_done = 0, m.last_row_local = false;  // after import_state
    if (pre.export_state().position != 64U || exports != 2) throw std::runtime_error("export after an import");
    std::puts("dayaudit5 export after group: ok");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
}
