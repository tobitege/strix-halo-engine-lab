#include "strix/env_flags.hpp"
#include "strix/hip_token_sampling.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
namespace strix {
void read_hip_logits_row(const ExecutionBufferAddress& row,std::uint32_t vocabulary,void* stream,std::vector<float>& out) {
    const auto bytes=static_cast<std::uint64_t>(vocabulary)*sizeof(float);
    if(!row.address || row.bytes<bytes) throw std::invalid_argument("sampled logits row is smaller than the vocabulary");
    out.resize(vocabulary);
    auto check=[](hipError_t e,const char* what){if(e!=hipSuccess) throw std::runtime_error(std::string(what)+": "+hipGetErrorString(e));};
    const auto s=static_cast<hipStream_t>(stream);
    check(hipMemcpyAsync(out.data(),reinterpret_cast<const void*>(row.address),bytes,hipMemcpyDeviceToHost,s),"sampled logits readback");
    check(hipStreamSynchronize(s),"sampled logits readback completion");
}
namespace {
struct SampleTimingTotals {
    std::mutex mutex;
    // [kind][rows 0..8]: calls, decided rows, readback seconds, decide seconds
    std::uint64_t calls[5][9]{}, decided[5][9]{};
    double readback[5][9]{}, decide[5][9]{};
    ~SampleTimingTotals() {
        static const char* names[5] = {"window", "serial", "draft", "kernels", "draft-kernels"};
        for (unsigned k = 0; k < 5U; ++k) for (unsigned r = 0; r < 9U; ++r) if (calls[k][r])
            std::fprintf(stderr, "sample-timing %s rows=%u: %llu calls, %llu decided rows, readback %.1f us/call, decide %.1f us/call\n",
                names[k], r, static_cast<unsigned long long>(calls[k][r]), static_cast<unsigned long long>(decided[k][r]),
                readback[k][r] * 1e6 / static_cast<double>(calls[k][r]), decide[k][r] * 1e6 / static_cast<double>(calls[k][r]));
    }
};
SampleTimingTotals& sample_timing_totals() { static SampleTimingTotals totals; return totals; }
} // namespace
bool sample_timing_enabled() noexcept {
    static const bool on = strix::envcfg::flag("STRIX_SAMPLE_TIMING", false);
    return on;
}
void record_sample_timing(SampleTimingKind kind,std::uint32_t rows,double readback_seconds,double decide_seconds,
    std::uint32_t decided_rows) {
    auto& t = sample_timing_totals();
    const auto k = static_cast<unsigned>(kind), r = rows < 9U ? rows : 0U;
    const std::lock_guard lock(t.mutex);
    ++t.calls[k][r]; t.decided[k][r] += decided_rows; t.readback[k][r] += readback_seconds; t.decide[k][r] += decide_seconds;
}
} // namespace strix
