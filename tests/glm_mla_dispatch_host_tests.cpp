#include "glm/prefill/glm_pf_wmma.hpp"
#include "core/f16_tile_gemm.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <string_view>

namespace strix::glm::pf {
using hipStream_t = void*;
struct DsaSlot {};
struct Call {
    mla_gemm::Dispatch dispatch;
    unsigned gemm, rows;
    const float* q;
    float* out;
};
std::array<Call, 16> calls{};
unsigned call_count = 0;
constexpr unsigned kMlaDim = 256, kMaxTokens = 2048;
bool mla_v1() { return false; }
unsigned scores_version() { return 7; }
unsigned obar_env_version(unsigned) { return 7; }
unsigned mla_gemm_version() { return 3; }
unsigned attention_scratch_rows(unsigned c, unsigned window) { return window ? std::min(c, window) : c; }
void mla_attention_rows_impl(const DsaSlot&, const std::uint32_t*, const std::uint32_t*, const float* q,
    const std::uint16_t*, unsigned, float*, float*, float*, float* out, unsigned c, hipStream_t,
    unsigned, unsigned, unsigned gemm, const mla_gemm::Dispatch& dispatch) {
    calls.at(call_count++) = {dispatch, gemm, c, q, out};
}
#include "mla_wrappers.inc"
}  // namespace strix::glm::pf

namespace P = strix::glm::pf;
namespace F = strix::core::f16_tile_gemm;
int failures = 0;
void check(bool pass, const char* what) {
    if (!pass) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}
void env(const char* name, const char* value) {
#ifdef _WIN32
    if (_putenv_s(name, value ? value : "") != 0) throw std::runtime_error("environment setup failed");
#else
    if ((value ? setenv(name, value, 1) : unsetenv(name)) != 0) throw std::runtime_error("environment setup failed");
#endif
}

void dispatch_test(bool expected_fast) {
    std::vector<float> q(129U * 32U * P::kMlaDim), out(q.size());
    std::vector<std::uint32_t> tokens(129U * P::kMaxTokens), counts(129);
    for (unsigned gemm : {2U, 3U}) {
        P::call_count = 0;
        P::mla_attention_rows_versions({}, tokens.data(), counts.data(), q.data(), nullptr, 32,
            nullptr, nullptr, nullptr, out.data(), 129, nullptr, 7, 7, gemm);
        check(P::call_count == 1 && !P::calls[0].dispatch.use_wmma(true) && P::calls[0].gemm == gemm,
              "explicit exact GEMM selector survives FAST environment");
        P::call_count = 0;
        P::mla_attention_rows_versions({}, tokens.data(), counts.data(), q.data(), nullptr, 32,
            nullptr, nullptr, nullptr, out.data(), 129, nullptr, 7, 7, gemm, 128);
        check(P::call_count == 2 && P::calls[0].rows == 128 && P::calls[1].rows == 1,
              "windowed exact wrapper covers tail");
        for (unsigned i = 0; i < P::call_count; ++i)
            check(!P::calls[i].dispatch.use_wmma(true) && P::calls[i].gemm == gemm,
                  "windowed exact selector survives FAST environment");
        check(P::calls[1].q == q.data() + 128U * 32U * P::kMlaDim &&
              P::calls[1].out == out.data() + 128U * 32U * P::kMlaDim, "window origins preserved");
    }
    P::call_count = 0;
    P::mla_attention_rows({}, tokens.data(), counts.data(), q.data(), nullptr, 32,
        nullptr, nullptr, nullptr, out.data(), 129, nullptr);
    P::mla_attention_rows({}, tokens.data(), counts.data(), q.data(), nullptr, 32,
        nullptr, nullptr, nullptr, out.data(), 129, nullptr, 128);
    check(P::call_count == 3, "ordinary wrappers dispatch full and windowed calls");
    for (unsigned i = 0; i < P::call_count; ++i)
        check(P::calls[i].dispatch.use_wmma(true) == expected_fast && !P::calls[i].dispatch.use_wmma(false),
              "ordinary dispatch uses automatic arithmetic with eligibility fallback");
    P::call_count = 0;
    P::mla_attention_rows_selected({}, tokens.data(), counts.data(), q.data(), nullptr, 32,
        nullptr, nullptr, nullptr, out.data(), 129, nullptr, 7, 23, 3, 128,
        {P::mla_gemm::Arithmetic::wmma, false, false});
    check(P::call_count == 2, "explicit FAST windows dispatched");
    for (unsigned i = 0; i < P::call_count; ++i)
        check(P::calls[i].dispatch.use_wmma(true) && !P::calls[i].dispatch.ob16 && !P::calls[i].dispatch.heads_mid,
              "explicit FAST/image/placement policy is independent of environment");
}

void admission_test() {
    alignas(16) float a[4]{};
    alignas(16) std::uint16_t w[8]{};
    F::Args p{};
    p.a = a; p.w = w; p.rows = 129; p.K = 256; p.N = 512; p.batches = 32;
    p.a_row = 32 * 256; p.a_batch = 256; p.w_ld = 512; p.w_batch = 2 * 256 * 512;
    check(F::fits(p), "GLM absorb shape admitted");
    p.K = 512; p.N = 256;
    check(F::fits(p), "GLM out shape admitted");
    p.K = 0; check(!F::fits(p), "zero K rejected before unconditional load"); p.K = 512;
    p.N = 0; check(!F::fits(p), "zero N rejected"); p.N = 256;
    p.rows = 0; check(!F::fits(p), "empty rows rejected");
    p.rows = 65535U * F::kBM; check(F::fits(p), "maximum row grid admitted");
    ++p.rows; check(!F::fits(p), "oversized row grid rejected");
    p.rows = std::numeric_limits<unsigned>::max(); check(!F::fits(p), "rounded row overflow rejected");
    p.rows = 129; p.order = 2; p.batches = p.group = 256;
    p.N = (1U << 24U) * F::kBN; check(!F::fits(p), "grouped column multiplication overflow rejected");
    p.N = 255U * F::kBN; check(F::fits(p), "grouped grid within limit admitted");
    p.N = 256U * F::kBN; check(!F::fits(p), "grouped grid limit enforced");
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string_view mode(argv[1]);
    const bool fast = mode != "off";
    env("STRIX_GLM_PF_WMMA", mode == "force" ? "0" : "1");
    env("STRIX_GLM_PF_MLA_WMMA", mode == "inherit" ? nullptr : fast ? "1" : "0");
    env("STRIX_GLM_PF_MLA_OB16", fast ? "1" : "0");
    env("STRIX_GLM_PF_MLA_GEMM_HEADS", fast ? "0" : "1");
    dispatch_test(fast);
    const auto entries = strix::envcfg::registered();
    check(entries.size() == 3, "all three controls registered");
    env("STRIX_GLM_PF_WMMA", "0");
    env("STRIX_GLM_PF_MLA_WMMA", fast ? "0" : "1");
    env("STRIX_GLM_PF_MLA_OB16", fast ? "0" : "1");
    env("STRIX_GLM_PF_MLA_GEMM_HEADS", fast ? "1" : "0");
    check(P::mla_wmma::enabled() == fast && P::mla_wmma::ob16() == fast && P::mla_gemm::heads_mid() == !fast,
          "configuration is immutable after first read");
    if (failures != 0) return 1;
    {
        // Cached reads must not acquire the registry mutex; CTest timeout catches regressions.
        std::lock_guard<std::mutex> lock(strix::envcfg::registry_mutex());
        for (unsigned i = 0; i < 100; ++i)
            check(P::mla_gemm::automatic_dispatch().use_wmma(true) == fast, "cached launch policy avoids registry lock");
    }
    const auto after = strix::envcfg::registered();
    check(after.size() == entries.size(), "registry remains stable");
    for (std::size_t i = 0; i < entries.size() && i < after.size(); ++i)
        check(entries[i].name == after[i].name && entries[i].value == after[i].value && entries[i].set == after[i].set,
              "receipts retain startup configuration");
    admission_test();
    std::printf("glm-mla-dispatch %.*s: %d failures\n", static_cast<int>(mode.size()), mode.data(), failures);
    return failures == 0 ? 0 : 1;
}
