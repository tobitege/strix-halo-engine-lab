#pragma once
// Routed expert on the host: the A8 K-quant expert product of the CPU backend core (layer 2 of the device rule).
// Used by tiered expert residency (docs/engine/memory/expert-residency.md): an expert that is not resident on the
// card is computed here from host memory and its per-(row, slot) output goes back to the card's own merge.
//
// Arithmetic contract = the CUDA decode MoE rows kernels (src/cuda/qwen_moe.cu project_rows_kernel /
// down_rows_kernel) = the HIP known-rows chains. Inputs are the card's A8 row (codes, per-32 deltas E, per-32 code
// sums S, from its prologue). Every float operation is one IEEE round-to-nearest operation (no contraction):
//   gate/up (Q4_K | Q5_K, super-block b of 256 = groups j 0..7, D_j = d * scale_j, DM_j = dmin * min_j):
//     x_j = E_g * ((D_j * float(dot_j)) - (DM_j * float(S_g))),  g = 8b + j
//     pair_p = x_2p + x_2p+1 (p 0..3);  acc_p += pair_p over b ascending;  value = (acc0 + acc1) + (acc2 + acc3)
//   SwiGLU: h = (g / (1 + gfx1151_expf(-g))) * u, then the quantize_a8 contract per 32 h values (codes, delta, sum)
//   down (Q5_1 | Q8_0, groups of 32, chain j 0..3 owns groups g = 4c + j, c ascending):
//     first = d * float(dot); Q5_1: combined = first + (m * float(S_g)); acc_j += E_g * combined
//     rawdot = (acc0 + acc1) + (acc2 + acc3)
// A non-finite gate, up or h sets status[row] |= kNonfinite (the card's projection_nonfinite rule; that step then
// fails closed on the card exactly as when the card computes the expert). Like the card (down_rows_kernel returns when
// the job summary has a failure), the down phase is then skipped for the whole job: rawdots stay untouched.
// Model-agnostic: shapes come from Shape, formats from the stored tensor types; a model adapter only fills LayerWeights.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace strix::cpu::expert {

enum class GateFormat : std::uint8_t { q4_k, q5_k };
enum class DownFormat : std::uint8_t { q5_1, q8_0 };

struct Shape {
    std::uint32_t hidden = 0;  // multiple of 256
    std::uint32_t ffn = 0;     // multiple of 128 (four down chains of 32-groups)
};

// One layer's stored expert tensors, GGUF layout: gate/up [experts][ffn][hidden/256 super-blocks],
// down [experts][hidden][ffn/32 blocks]. Pointers may be file-mapped memory.
struct LayerWeights {
    const std::uint8_t* gate = nullptr;
    const std::uint8_t* up = nullptr;
    const std::uint8_t* down = nullptr;
    GateFormat gate_format = GateFormat::q4_k;
    DownFormat down_format = DownFormat::q5_1;
    Shape shape;
};

struct RowA8 {
    const std::int8_t* codes = nullptr;    // hidden codes
    const float* deltas = nullptr;         // hidden / 32
    const std::int32_t* sums = nullptr;    // hidden / 32
};

struct Assignment {
    std::uint32_t row = 0, slot = 0, expert = 0;
};

inline constexpr std::uint32_t kNonfinite = 1U;  // = qwen_moe projection_nonfinite

struct Job {
    LayerWeights weights;
    const RowA8* rows = nullptr;            // indexed by Assignment::row
    const Assignment* assignments = nullptr;
    std::uint32_t count = 0;
    float* rawdots = nullptr;               // rawdots[row * rawdot_row_stride + slot * hidden + o]
    std::uint64_t rawdot_row_stride = 0;    // in floats
    std::uint32_t* status = nullptr;        // per row, OR-ed
};

// Host twin of amd_expf (src/cuda/qwen_amd_math.cuh): gfx1151 v_exp_f32 emulation from the probe tables.
class Gfx1151Exp {
public:
    explicit Gfx1151Exp(const std::string& tables_dir);  // reads tbl_exp_pm.bin, hard_v_exp_f32.bin
    ~Gfx1151Exp();
    Gfx1151Exp(const Gfx1151Exp&) = delete;
    Gfx1151Exp& operator=(const Gfx1151Exp&) = delete;
    float expf(float x) const noexcept;
private:
    struct Data;
    std::unique_ptr<Data> d_;
};

// Persistent worker pool. run() splits a job into (expert, 32-output group) gate/up items and (expert, output
// chunk) down items, with work stealing; the bits never depend on the thread count or the ISA table chosen.
class Executor {
public:
    // threads includes the calling thread (threads - 1 workers are created). allow_simd = false forces the scalar table.
    Executor(unsigned threads, const Gfx1151Exp& exp, bool allow_simd = true);
    ~Executor();
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    void run(const Job& job);
    unsigned threads() const noexcept;
    const char* isa() const noexcept;  // "scalar", "avx2", ...
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Single-threaded scalar reference of the whole contract (the bitwise oracle of the tests).
void reference_run(const Job& job, const Gfx1151Exp& exp);

}  // namespace strix::cpu::expert
