#pragma once
#include "strix/hip_moe_q4k_a8_probe.hpp"
#include "strix/q51_known_rows.hpp"
#include "strix/moe_route_prune.hpp"
#include <array>
#include <span>
#include <stdexcept>

namespace strix::moe_q4k_a8::known_rows {
// Rows of one window union: verify windows of 2/4/8 rows, cohort unions of several sessions' windows
// (16 or 32 rows; down batches of at most 8 assignments, or opt-in 16 at R16).
inline constexpr unsigned max_union_rows = 32U, max_assignments = max_union_rows * 10U;
inline constexpr std::string_view organization_identity =
    "a8-csr-gate-up-v1:shared-packed-weights:independent-row-pair4";
// Source-declared staging arrays only; compiled LDS/register/private usage
// remains a compiler qualification result. Geometry is bounded before products.
[[nodiscard]] inline std::uint64_t staged_activation_bytes(std::uint32_t rows) {
    if(rows<2U||rows>max_union_rows)throw std::invalid_argument("A8 CSR staging requires 2..32 rows");
    return static_cast<std::uint64_t>(rows)*(max_hidden+2ULL*(max_hidden/32U)*sizeof(std::uint32_t));
}
// Per-row Bindings retain the exact R1 views and numerical contract. Quantize
// each row once before this entry. CSR must be the complete stable (row,slot)
// grouping in q51_known_rows::Workspace; repeated row/expert slots are distinct.
// Validates metadata only, never reads host payloads or changes device bytes.
void preflight(const Request&, std::span<const Bindings>,
    const q51_known_rows::Workspace&, hipStream_t);
// One bounded device CSR admission, then packed gate/up loads shared across
// each <=R assignment tile. Both output chains retain the incumbent FP order.
// No quantization, clear, allocation, synchronization, SiLU or publication.
// Every owner/descriptor and the explicit eager stream remain retained and
// exclusively leased until quiescence, including after any enqueue exception.
void enqueue(const Request&, std::span<const Bindings>,
    const q51_known_rows::Workspace&, hipStream_t);
// Device tables of the window prologue (probe TU, shares the quantizer body).
struct CsrRows { unsigned *summary{}, *offsets{}, *touched{}, *rows{}, *slots{}, *experts{}, *begins{}; };
struct PrologueRows {
    const void* input[max_union_rows]{}; const unsigned* routes[max_union_rows]{}; const float* route_weights[max_union_rows]{};
    signed char* codes[max_union_rows]{}; float* deltas[max_union_rows]{}; int* sums[max_union_rows]{}; unsigned* status[max_union_rows]{}; float* clear[max_union_rows]{};
    CsrRows csr;
};
// One CTA per row zeroes clear[r][0..clear_words) and status, validates routes
// and weights and quantizes (the bytes of the per-row clear/validate/quantize
// sequence). A concurrent CTA builds the stable CSR from the routes alone; on
// an invalid route it writes the fail-closed summary instead. Quantizer
// failures are only in row status: consumers must check status as well.
// active (0 = rows): only the first active rows enter the expert union (CSR); the rest are still cleared,
// route-checked and quantized.
void enqueue_prologue(const Request&, const PrologueRows&, unsigned rows, unsigned clear_words, hipStream_t, unsigned active = 0, unsigned batch_cap = 8);
// Window-owned rows: route weights and the per-row FP32 region zeroed by the
// prologue. down_type 7/8 selects the A8 profile (Q5_1/Q8_0 A8G32 down) and
// then needs each row's serial down image (codes/deltas/sums) and raw slot dots.
struct WindowRows {
    std::array<DeviceBytes, max_union_rows> route_weights{}, clear{};
    std::uint32_t down_type = 0;
    DeviceBytes down_weights{};
    std::array<DeviceBytes, max_union_rows> down_codes{}, down_deltas{}, down_sums{}, rawdots{};
};
// Complete window producer: one prologue launch (rows plus the device CSR
// build into the workspace: complete stable (row,slot) grouping, so no
// separate admission), then shared gate/up with the finite scans and SiLU
// fused per element. Leaves each row's slot workspace exactly as quantize/
// project/scan/SiLU/scan would on success; any failure sets the same local
// status bits. After an early (route/quantizer) failure the caller restores
// the fail-closed summary (touched, batches, histograms zero). A8 profile
// (down_type 7/8, Q4_K or Q5_K gate/up): projection plus the serial down
// activation quantizer, then the shared integer down writes each row's raw
// slot dots exactly as down_activation_quantize/down_rawdots would; the
// caller merges and commits.
// active (0 = every row): the expert union covers the first active rows only.
void enqueue_window(const Request&, std::span<const Bindings>,
    const q51_known_rows::Workspace&, const WindowRows&, hipStream_t, unsigned active = 0);

// Known-prompt prefill over T rows (multiple of 8, A8 profile only). Every per-row
// view is row 0 plus r times its own byte stride (the prefill arena stripes rows).
template <class T> struct StridedRows { T* base{}; std::uint64_t stride{}; };
struct PrefillRows {
    StridedRows<const void> input; StridedRows<const unsigned> routes; StridedRows<const float> route_weights;
    StridedRows<signed char> codes; StridedRows<float> deltas; StridedRows<int> sums; StridedRows<unsigned> status;
    StridedRows<float> output, clear;
    StridedRows<signed char> down_codes; StridedRows<float> down_deltas; StridedRows<int> down_sums;
    StridedRows<float> rawdots;
    CsrRows csr; // q51_known_rows::scratch_layout(T, experts, top_k, true) in an exclusive allocation
    bool route_bf16 = false; // Candidate: DA16 certificate writer and matching ordered merge only.
    unsigned* unsafe = nullptr; // Private admission sidecar, stream-owned; null selects base producers.
    bool route_i8 = false; // Env-only draft, requires route_bf16 and its reserved FP32 row.
};
// T CTAs: each row's clear, route checks and quantizer exactly as the window prologue.
void enqueue_prefill_prologue(const Request&, const PrefillRows&, unsigned rows, unsigned clear_words, hipStream_t);
// Prologue, GPU CSR over the T*top_k assignments (stable: rows then slots ascending per
// expert, batches of consecutive assignments of one expert; row statuses folded into the
// summary word), gate/up with the serial down activation quantizer and the integer down.
// Each (row, slot) runs the window kernels' per-lane arithmetic; the caller merges and
// commits. Summary histograms count n>=8 in their last bucket. form 1 runs gate/up and down
// as iu8 WMMA kernels over batches of <= 16 assignments (CSR from a parallel builder), form 0
// the dot4 kernels, -1 picks by STRIX_PREFILL_MOE_WMMA (default WMMA); form 2 (tests) is form 1
// with the serial CSR builder. Every output byte is identical across forms. A non-finite gate/up
// value of a (row, slot) sets status bit 8 in its row and in the summary: gate/up still runs every
// batch (the bytes it writes do not depend on CTA order), the down is skipped.
// Form 3 is the separate FP16 prefill contract. Forms 4..34 (tunes 1..31) and 35..66 (tunes
// 64..95) are its schedule candidates: bit 0 one register-prefetch stage, bit 1 batch-major
// 128-assignment batches, bit 2 compact 128-column tile, bit 3 unpadded XOR-permuted LDS rows,
// bit 4 prologue FP16 image of the prompt rows, bit 6 short exact dequantization.
// Form 3 selects the same schedules with STRIX_INTPREFILL_MOE_TUNE (default 0).
struct DirectBOperands { const unsigned char* gate=nullptr; const unsigned char* up=nullptr; };
void enqueue_prefill(const Request&, const PrefillRows&, unsigned rows, unsigned clear_words,
    const unsigned char* gate, const unsigned char* up, unsigned down_type, const unsigned char* down_weights, hipStream_t,
    int form = -1, DirectBOperands direct = {});
// Diagnostic: MoE tune bit 6 (short exact FP16 dequantization) against the base conversion on random scales and every
// code. Returns tested/admitted/mismatched counts for K-quant sub-blocks and Q5_1 blocks, then the same for the rejected
// Q8_0 offset form (informational: it differs in the sign of zero) (synchronizes).
// The kernels enqueue_prefill runs for this request, down type and form (receipts count GSQ rows per form).
enum class PrefillKernelForm : unsigned { dot4, iu8_wmma, f16_wmma };
PrefillKernelForm prefill_kernel_form(const Request&, unsigned down_type, int form);
// The MoE tune of a FP16 prefill call (STRIX_INTPREFILL_MOE_TUNE or the form, bit 1 cleared below 96 assignments per
// expert); the GSQ FP16 kernels run it masked with gsq_f16_tune_mask (batch-major bit 1, A16 image bit 4, diagnostics).
unsigned prefill_effective_tune(const Request&, unsigned count, unsigned down_type, int form);
inline constexpr unsigned gsq_f16_tune_mask = 2U | 16U | 256U | 512U | 1024U;
std::array<unsigned long long,9> fast_dequant_selftest(unsigned long long seed,unsigned blocks);
// GSQ Q2_0 decoders of the prefill kernels (FP16 TPC 1/2, iu8 operands) against the reference values on rows of 64
// blocks covering every FP16 scale pattern and every code byte per position: {FP16 words, differing, NaN references,
// code bytes, differing}.
std::array<unsigned long long,5> gsq_q20_decode_selftest(unsigned seed,unsigned rows);
// GSQ IQ decoders of the FP16 prefill kernels (IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL) against round_f16 of the format's
// element values on `blocks` random blocks (scale bits (b + seed) & 65535): {FP16 words, differing, NaN references,
// distinct grid indices (IQ4: nibble values) seen, distinct sign patterns seen (IQ3_XXS: 7-bit; IQ4: 0)}.
std::array<unsigned long long,5> gsq_iq_decode_selftest(std::uint32_t type,unsigned seed,unsigned blocks);
// qpf2k core2 test hook: overrides STRIX_QPF2K_MOE for the routed prefill (-1 = use the environment).
void set_prefill_core2(int core);
// qpf2k/moe-mi test hook: overrides STRIX_QPF2K_MOE_ROUTE (bit 1 CSR v2, bit 2 prologue v2; -1 = use the environment).
void set_prefill_route(int bits);
// qpf2k/moe-mi test hook: overrides STRIX_QPF2K_MOE_DA16 (1 = FP16 down operand image written by the last gate/up tile; -1 = environment).
void set_prefill_da16(int on);
// Test hook: overrides STRIX_QPF2K_MOE_DA16_ISSUE (1 = uniform DA16 activation issue, 0 = masked issue; -1 = environment, automatic when unset).
void set_prefill_da16_issue(int on);
// MoE header reuse test hook: overrides STRIX_QPF2K_MOE_HDR (1 = gate/up header reuse kernel; -1 = environment).
void set_prefill_hdr(int on);
// Test hooks: STRIX_QPF2K_MOE_TIERS / STRIX_QPF2K_MOE_DOWN_LDS_EPI, default OFF;
// -1 restores the environment. Enable tiers before reserve_prefill_pools so setup
// reserves the lists; execution never allocates. Tune 82 splits; 80/88 are already narrow.
// Tiers 1 = padded narrow LDS, 2 = compact XOR narrow LDS; both keep the full tier.
void set_prefill_tiers(int on);
bool prefill_tiers_ran();
int prefill_tiers_mode_ran(); // 0 = unsplit, otherwise the admitted mode (1/2).
void set_prefill_down_lds(int on);
bool prefill_down_lds_ran();
// Optional caller-owned timing events, recorded on the enqueue stream at DA16
// gate/up begin/end and down end. Null disables recording; no device work here.
// The caller retains all three events until stream completion and clears the hook.
void prefill_test_da16_events(const std::array<hipEvent_t,3>* events);
// Borrowed scratch, valid on the pool's owning device until its next setup growth.
const void* prefill_test_da16_data();
const unsigned* prefill_test_da16_arrivals();
// TILE64 test override (-1 = environment, 0 = today's tile, 1 = TILE64 gate/up and down, 2 = gate/up only, 3 = down only);
// default OFF until node timing.
void set_prefill_tile64(int on);
// TILE64 test hooks: whether the last enqueue_prefill of this thread took the TILE64 organization (either kernel); the
// device address of the DA16 FP16 image (the gate/up product the down stages), for a digest.
bool prefill_tile64_ran();
const void* prefill_test_da16_image();
// Test hook: three events recorded on the DA16 path (before gate/up, after gate/up, after down); nullptr = none.
void set_prefill_test_events(hipEvent_t* events);
void set_prefill_direct_b(int on);
bool prefill_direct_b_ran();
const char* prefill_direct_b_reason();
struct DirectBFragmentResult {
    unsigned long long mismatched_elements=0,pass_lanes=0,fallback_lanes=0;
    // First mismatch in coordinate order. Unit = 2*pair + matrix (gate 0/up 1),
    // lane = output column 0..15, replica = hardware lane / 16, plane = K / 16.
    // Expected/actual are FP16 bits; coordinates are valid only on mismatch.
    unsigned unit=0,lane=0,replica=0,plane=0,element=0,expected=0,actual=0;
};
DirectBFragmentResult direct_b_fragment_test(const unsigned char* raw_units,const unsigned char* packed_units,unsigned pairs,bool paired_planes=false);
// DA16 gate/up plus completion fixture/timing entry. CSR, exponents and A16
// must be ready; caller clears maxima and arrival counters before every launch.
// The reference is the production header-reuse specialization at this tune.
void prefill_project_test(const PrefillRows&, unsigned hidden, unsigned outputs,
    const unsigned char* gate,const unsigned char* up,unsigned batches,unsigned tune,
    bool direct,hipStream_t,DirectBOperands,unsigned* arrive,uint4* image);
// MoE XOR rows test hook: overrides STRIX_QPF2K_MOE_XOR64 (1 = 64-assignment DA16 tune with XOR LDS rows from 3072 rows; -1 = environment).
void set_prefill_xor64(int on);
// Test hook: the FP16 tune of the last enqueue_prefill of this thread.
unsigned prefill_last_tune();
// qpf2k/moe-mi test hooks: whether the last enqueue_prefill of this thread took the DA16 path; fill the DA16 image with 0xa5.
bool prefill_da16_ran();
// True when the last enqueue_prefill of this thread left the accumulation view unzeroed (STRIX_QPF2K_MOE_ROUTE bit 8): the
// caller must then run the merge variant that writes zeros on its skip paths (known_moe_prefill_merge_kernel_z).
bool prefill_clear_skipped();
// True only after all three certificate producer specializations were enqueued for this call.
bool prefill_certificate_produced();
// RoutePrune of this thread's next enqueue_prefill calls (off by default): the route weights of pruned assignments become 0
// and the CSR plans the kept ones only; the caller's merge must skip zero-weight slots.
void set_prefill_route_prune(moe_core::RoutePrune prune);
// Reserve once before any prefill submission; largest compiled T first, on its device.
void reserve_prefill_pools(unsigned rows);
std::array<std::uint64_t, 2> prefill_test_image_generations();
void prefill_test_poison_da16(hipStream_t stream, unsigned rows);
// qpf2k/moe-mi test hooks: STRIX_QPF2K_MOE_SMALL override (batches with <= rows assignments take the depth-3 body; 0 = off,
// -1 = environment) and whether the last enqueue_prefill of this thread took that path.
void set_prefill_small(int rows);
bool prefill_small_ran();
// qpf2k/moe-mi test hook: fills the FP16 prefill A image (MoE tune bit 4) with 0xa5 bytes, so a gate sees missing writes.
void prefill_test_poison_a16(hipStream_t stream, unsigned rows);
// True when the last enqueue_prefill of this thread fused the ordered merge into the down (STRIX_QPF2K_MOE bit 9): the caller
// then launches enqueue_prefill_merge_check instead of its merge kernel (same accumulation, status and header bits).
bool prefill_fused_merge();
// qpf2k P1 test hook (STRIX_QPF2K_MOE=257 with STRIX_QPF2K_MOE_P1STATS=1), last call (synchronizes the device): [0] down items
// that waited, [1] polls, [2] down items that ran, [3] down items that skipped, [4..5] total wait (u64 wall-clock ticks).
std::array<unsigned,8> prefill_p1_stats();
void enqueue_prefill_merge_check(const PrefillRows& rows,unsigned count,unsigned hidden,unsigned experts,unsigned top_k,hipStream_t stream);
// STRIX_QPF2K_SCHED bit 64 (the tiled prefill owner's side stream): a one-shot host callback of this thread, run right after
// the next enqueue_prefill enqueued `stage` on its stream (1 = CSR, 2 = gate/up; 3 = after enqueue_prefill returned, before
// the caller's merge). A path without the stage runs it at 3 (prefill_stage_reached(3)). It only enqueues other-stream work.
using PrefillStageHook = void (*)(void* user, hipStream_t stream);
void set_prefill_stage_hook(PrefillStageHook hook, void* user, unsigned stage);
void prefill_stage_reached(unsigned stage, hipStream_t stream);
bool prefill_stage_hook_armed();
} // namespace strix::moe_q4k_a8::known_rows
