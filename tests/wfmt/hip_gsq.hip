// GSQ IQ MoE window / prefill migration fixture. It launches the incumbent kernels of
// src/hip_moe_q4k_a8_known_rows.hip (project_window_iq, project_prefill_iq, down_window_iq, down_prefill_iq) over seeded
// synthetic payloads, same-TU included like the other kernel fixtures. Under STRIX_WFMT_CAPTURE it records weight source
// bytes, activation codes/scales, the CSR sidecar, per-group partial contributions (gsq_partial_trace.hpp hook inside
// project_iq_body / down_iq_body) and the outputs. Without it the self-checks still run: the window arm (launcher G / Rows
// selection, plus every other legal G / Rows) must equal the prefill arm bit for bit (same per-(row, slot) lane arithmetic).
// STRIX_WFMT_GSQ_REPORT=1 reports a mismatch and keeps going (diagnosis); the default throws.
#include "../../src/hip_moe_q4k_a8_known_rows.hip"
#include "strix/env_flags.hpp"
#include "hip_gsq_checks.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace kr = strix::moe_q4k_a8::known_rows;
namespace ga = strix::gsq_a8;

void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
// Quiet transfers (before the capture header redirects hipMemcpy): nothing here is recorded implicitly.
void raw_upload(void* dst, const void* src, std::size_t n) { check(hipMemcpy(dst, src, n, hipMemcpyHostToDevice)); }
void raw_download(void* dst, const void* src, std::size_t n) { check(hipMemcpy(dst, src, n, hipMemcpyDeviceToHost)); }
}  // namespace
#ifdef STRIX_WFMT_CAPTURE
#include "../../scripts/bench/wfmt/hip_transfer_capture.hpp"
#endif

namespace {
void context(const std::string& name, std::initializer_list<std::uint64_t> shape, const char* format) {
#ifdef STRIX_WFMT_CAPTURE
    strix::weight::capture::set_context(name.c_str(), "hip-gsq-a8-iq-ordered@1", shape, format);
#else
    (void)name; (void)shape; (void)format;
#endif
}
void emit(const char* section, const void* data, std::size_t bytes) {
#ifdef STRIX_WFMT_CAPTURE
    strix::weight::capture::emit(section, data, bytes);
#else
    (void)section; (void)data; (void)bytes;
#endif
}

struct Buffer {
    void* p = nullptr;
    std::size_t bytes = 0;
    explicit Buffer(std::size_t n) : bytes(n) { check(hipMalloc(&p, n)); check(hipMemset(p, 0, n)); }
    ~Buffer() { (void)hipFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template <class T> T* as() const { return static_cast<T*>(p); }
    template <class T> void put(const std::vector<T>& v) const { raw_upload(p, v.data(), v.size() * sizeof(T)); }
    void clear() const { check(hipMemset(p, 0, bytes)); }
    template <class T> std::vector<T> read(std::size_t n) const {
        std::vector<T> v(n);
        raw_download(v.data(), p, n * sizeof(T));
        return v;
    }
};

struct Rng {
    std::uint64_t s;
    explicit Rng(const std::string& name) : s(UINT64_C(14695981039346656037)) {
        for (const char c : name) s = (s ^ static_cast<unsigned char>(c)) * UINT64_C(1099511628211);
        if (!s) s = 1;
    }
    std::uint32_t next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return static_cast<std::uint32_t>((s * UINT64_C(2685821657736338717)) >> 32); }
};

// Synthetic IQ payload: random bytes, each block header an FP16 scale that is finite (normal, subnormal, zero, negative).
std::vector<unsigned char> payload(Rng& rng, std::uint32_t type, std::uint64_t rows, std::uint32_t width) {
    const std::uint64_t row_bytes = ga::row_bytes(type, width), block = ga::block_bytes(type);
    std::vector<unsigned char> w(rows * row_bytes + 64U);
    for (auto& b : w) b = static_cast<unsigned char>(rng.next() >> 11);
    static constexpr std::uint16_t scales[] = {0x2800, 0x2c00, 0x3000, 0x3400, 0x1c00, 0x2400, 0x2a55, 0x0001, 0x03ff, 0x0000, 0xa800, 0x3800};
    for (std::uint64_t i = 0; i < rows * (row_bytes / block); ++i) {
        const auto h = scales[rng.next() % (sizeof scales / sizeof scales[0])];
        std::memcpy(w.data() + i * block, &h, 2U);
    }
    return w;
}

struct Spec { unsigned T, H, N; bool dup; };
constexpr unsigned kTopK = 2U, kExperts = 6U;

struct Csr {
    std::vector<std::uint32_t> words;
    unsigned batches = 0, assignments = 0, summary = 0, offsets = 0, touched = 0, rows = 0, slots = 0, experts = 0, begins = 0;
    unsigned max_batch = 0;
};
// Stable CSR as the planner builds it: touched experts ascending, rows then slots ascending, one batch per expert.
Csr build_csr(const Spec& spec) {
    std::vector<std::uint32_t> routes(spec.T * kTopK);
    for (unsigned r = 0; r < spec.T; ++r) {
        routes[r * kTopK] = 0U;
        routes[r * kTopK + 1U] = r == 0U ? 5U : (spec.dup && r == 1U) ? 0U : 1U + (r * 5U + r / 2U) % 3U;
    }
    std::vector<std::uint32_t> rows, slots, experts, begins;
    for (unsigned e = 0; e < kExperts; ++e) {
        const auto begin = rows.size();
        for (unsigned r = 0; r < spec.T; ++r)
            for (unsigned s = 0; s < kTopK; ++s)
                if (routes[r * kTopK + s] == e) { rows.push_back(r); slots.push_back(s); }
        if (rows.size() > begin) { experts.push_back(e); begins.push_back(static_cast<std::uint32_t>(begin)); }
    }
    begins.push_back(static_cast<std::uint32_t>(rows.size()));
    Csr c;
    c.batches = static_cast<unsigned>(experts.size());
    c.assignments = static_cast<unsigned>(rows.size());
    for (unsigned b = 0; b < c.batches; ++b) c.max_batch = std::max(c.max_batch, begins[b + 1U] - begins[b]);
    c.summary = 0U; c.offsets = 32U; c.touched = c.offsets + kExperts + 1U; c.rows = c.touched + kExperts;
    c.slots = c.rows + c.assignments; c.experts = c.slots + c.assignments; c.begins = c.experts + c.batches;
    c.words.assign(c.begins + c.batches + 1U, 0U);
    c.words[1] = c.batches; c.words[2] = c.batches; c.words[3] = c.assignments;
    c.words[4] = spec.dup ? 1U : 0U; c.words[5] = c.max_batch; c.words[6] = spec.T; c.words[7] = 1U;
    std::copy(rows.begin(), rows.end(), c.words.begin() + c.rows);
    std::copy(slots.begin(), slots.end(), c.words.begin() + c.slots);
    std::copy(experts.begin(), experts.end(), c.words.begin() + c.experts);
    std::copy(begins.begin(), begins.end(), c.words.begin() + c.begins);
    return c;
}

// Diagnostic trace of the per-group partials (hook compiled only under STRIX_WFMT_CAPTURE); armed per launch.
struct Trace {
#ifdef STRIX_WFMT_CAPTURE
    std::unique_ptr<Buffer> values, counts;
    unsigned threads = 0;
    explicit Trace(unsigned t) : values(std::make_unique<Buffer>(static_cast<std::size_t>(t) * strix::wfmt_trace::kSlots * 4U)),
        counts(std::make_unique<Buffer>(static_cast<std::size_t>(t) * 4U)), threads(t) {
        auto* v = values->as<float>(); auto* c = counts->as<unsigned>();
        check(hipMemcpyToSymbol(HIP_SYMBOL(strix::wfmt_trace::gsq_values), &v, sizeof v));
        check(hipMemcpyToSymbol(HIP_SYMBOL(strix::wfmt_trace::gsq_counts), &c, sizeof c));
    }
    ~Trace() {
        float* v = nullptr; unsigned* c = nullptr;
        (void)hipMemcpyToSymbol(HIP_SYMBOL(strix::wfmt_trace::gsq_values), &v, sizeof v);
        (void)hipMemcpyToSymbol(HIP_SYMBOL(strix::wfmt_trace::gsq_counts), &c, sizeof c);
    }
    // Compact words: for each thread with partials, [thread, count, values...].
    std::vector<std::uint32_t> packed() const {
        const auto used = counts->read<unsigned>(threads);
        const auto all = values->read<std::uint32_t>(static_cast<std::size_t>(threads) * strix::wfmt_trace::kSlots);
        std::vector<std::uint32_t> packed;
        for (unsigned t = 0; t < threads; ++t) {
            if (!used[t]) continue;
            if (used[t] > strix::wfmt_trace::kSlots) throw std::runtime_error("diagnostic partial capacity");
            packed.push_back(t); packed.push_back(used[t]);
            packed.insert(packed.end(), all.begin() + static_cast<std::size_t>(t) * strix::wfmt_trace::kSlots,
                          all.begin() + static_cast<std::size_t>(t) * strix::wfmt_trace::kSlots + used[t]);
        }
        if (packed.empty()) throw std::runtime_error("missing diagnostic partials");
        return packed;
    }
#else
    explicit Trace(unsigned) {}
    std::vector<std::uint32_t> packed() const { return {}; }
#endif
};

using strix::wfmt_gsq::Snapshot;
using strix::wfmt_gsq::same;

// One record per arm: every output the chain writes (gate/up + activation, down codes/scales/sums, rawdots).
void emit_outputs(const Snapshot& s) {
    std::vector<unsigned char> bytes;
    const auto add = [&](const auto& v) { const auto* b = reinterpret_cast<const unsigned char*>(v.data()); bytes.insert(bytes.end(), b, b + v.size() * sizeof v[0]); };
    add(s.output); add(s.dcodes); add(s.dd); add(s.ds); add(s.raw);
    emit("output", bytes.data(), bytes.size());
}
using Words = std::vector<std::uint32_t>;
void emit_partials(const std::pair<Words, Words>& traces) {
#ifdef STRIX_WFMT_CAPTURE
    Words all{static_cast<std::uint32_t>(traces.first.size())};
    all.insert(all.end(), traces.first.begin(), traces.first.end()); all.insert(all.end(), traces.second.begin(), traces.second.end());
    emit("partials", all.data(), all.size() * sizeof(std::uint32_t));
#else
    (void)traces;
#endif
}

struct Case {
    Spec spec;
    Csr csr;
    std::uint32_t type;
    // device
    std::unique_ptr<Buffer> gate, up, down, codes, deltas, sums, status, output, dcodes, dd, ds, raw, table;
    kr::DeviceRows rows{};
    kr::DeviceCsr dcsr{};
    kr::PrefillRows pre{};
    std::size_t out_row, dcodes_row, dd_row, raw_row;

    void clear() const { for (auto* b : {output.get(), dcodes.get(), dd.get(), ds.get(), raw.get(), status.get()}) b->clear(); }
    Snapshot snapshot() const {
        const auto T = spec.T;
        Snapshot s;
        s.output = output->read<float>(T * out_row / 4U);
        s.dcodes = dcodes->read<signed char>(T * dcodes_row);
        s.dd = dd->read<float>(T * dd_row / 4U);
        s.ds = ds->read<int>(T * dd_row / 4U);
        s.raw = raw->read<float>(T * raw_row / 4U);
        strix::wfmt_gsq::check_snapshot(s, status->read<unsigned>(T), table->read<unsigned>(1U)[0]);
        return s;
    }
};

template <class F> void with_gate(std::uint32_t type, F&& f) {
    if (type == ga::type_iq2_s) f.template operator()<ga::type_iq2_s>();
    else if (type == ga::type_iq3_xxs) f.template operator()<ga::type_iq3_xxs>();
    else if (type == ga::type_iq3_s) f.template operator()<ga::type_iq3_s>();
    else if (type == ga::type_iq4_xs) f.template operator()<ga::type_iq4_xs>();
    else throw std::runtime_error("gate type not admitted");
}
const char* type_name(std::uint32_t t) {
    return t == ga::type_iq2_s ? "iq2_s" : t == ga::type_iq3_xxs ? "iq3_xxs" : t == ga::type_iq3_s ? "iq3_s" : "iq4_xs";
}

void build_case(Case& c, const std::string& name) {
    const auto& sp = c.spec;
    Rng rng(name);
    c.csr = build_csr(sp);
    auto gate = payload(rng, c.type, static_cast<std::uint64_t>(kExperts) * sp.N, sp.H);
    auto up = payload(rng, c.type, static_cast<std::uint64_t>(kExperts) * sp.N, sp.H);
    auto down = payload(rng, ga::type_iq4_nl, static_cast<std::uint64_t>(kExperts) * sp.H, sp.N);
    c.gate = std::make_unique<Buffer>(gate.size()); c.up = std::make_unique<Buffer>(up.size()); c.down = std::make_unique<Buffer>(down.size());
    c.gate->put(gate); c.up->put(up); c.down->put(down);
    {   // one "source" record: gate, up and down payload bytes as the kernels read them
        std::vector<unsigned char> weights(gate);
        weights.insert(weights.end(), up.begin(), up.end()); weights.insert(weights.end(), down.begin(), down.end());
        emit("source", weights.data(), weights.size());
    }
    // Activation codes / scales (extremes, one zero scale, varied exponents).
    std::vector<signed char> codes(static_cast<std::size_t>(sp.T) * sp.H);
    std::vector<float> deltas(static_cast<std::size_t>(sp.T) * (sp.H / 32U));
    for (std::size_t i = 0; i < codes.size(); ++i) {
        const auto r = rng.next();
        codes[i] = static_cast<signed char>(r % 17U == 0U ? ((r >> 8) & 1U ? 127 : -127) : static_cast<int>((r >> 5) % 255U) - 127);
    }
    for (std::size_t i = 0; i < deltas.size(); ++i)
        deltas[i] = i % 29U == 5U ? 0.0F : std::ldexp(1.0F + static_cast<float>(rng.next() % 1024U) / 1024.0F, -9 + static_cast<int>(rng.next() % 5U));
    c.codes = std::make_unique<Buffer>(codes.size()); c.deltas = std::make_unique<Buffer>(deltas.size() * 4U);
    c.codes->put(codes); c.deltas->put(deltas);
    emit("activation_codes", codes.data(), codes.size());
    emit("activation_scales", deltas.data(), deltas.size() * 4U);
    c.table = std::make_unique<Buffer>(c.csr.words.size() * 4U);
    c.table->put(c.csr.words);
    emit("sidecars", c.csr.words.data(), c.csr.words.size() * 4U);
    const std::size_t T = sp.T, groups_h = sp.H / 32U, groups_n = sp.N / 32U;
    c.out_row = kTopK * 2U * sp.N * 4U; c.dcodes_row = kTopK * sp.N; c.dd_row = kTopK * groups_n * 4U; c.raw_row = kTopK * sp.H * 4U;
    c.sums = std::make_unique<Buffer>(T * groups_h * 4U + 64U);
    c.status = std::make_unique<Buffer>(T * 4U);
    c.output = std::make_unique<Buffer>(T * c.out_row); c.dcodes = std::make_unique<Buffer>(T * c.dcodes_row);
    c.dd = std::make_unique<Buffer>(T * c.dd_row); c.ds = std::make_unique<Buffer>(T * c.dd_row); c.raw = std::make_unique<Buffer>(T * c.raw_row);
    auto* base = c.table->as<unsigned>();
    c.dcsr = {base + c.csr.summary, base + c.csr.offsets, base + c.csr.touched, base + c.csr.rows, base + c.csr.slots, base + c.csr.experts, base + c.csr.begins};
    // DeviceRows holds max_union_rows pointers per field: the window arm only runs T <= 32, prefill uses the strided rows.
    for (unsigned r = 0; r < std::min(sp.T, kr::max_union_rows); ++r) {
        c.rows.codes[r] = c.codes->as<signed char>() + static_cast<std::size_t>(r) * sp.H;
        c.rows.deltas[r] = c.deltas->as<float>() + static_cast<std::size_t>(r) * groups_h;
        c.rows.sums[r] = c.sums->as<int>() + static_cast<std::size_t>(r) * groups_h;
        c.rows.status[r] = c.status->as<unsigned>() + r;
        c.rows.output[r] = reinterpret_cast<float*>(c.output->as<unsigned char>() + r * c.out_row);
        c.rows.down_codes[r] = reinterpret_cast<signed char*>(c.dcodes->as<unsigned char>() + r * c.dcodes_row);
        c.rows.down_deltas[r] = reinterpret_cast<float*>(c.dd->as<unsigned char>() + r * c.dd_row);
        c.rows.down_sums[r] = reinterpret_cast<int*>(c.ds->as<unsigned char>() + r * c.dd_row);
        c.rows.rawdots[r] = reinterpret_cast<float*>(c.raw->as<unsigned char>() + r * c.raw_row);
    }
    auto& p = c.pre;
    p.codes = {c.codes->as<signed char>(), sp.H}; p.deltas = {c.deltas->as<float>(), groups_h * 4U};
    p.sums = {c.sums->as<int>(), groups_h * 4U}; p.status = {c.status->as<unsigned>(), 4U};
    p.output = {c.output->as<float>(), c.out_row}; p.down_codes = {c.dcodes->as<signed char>(), c.dcodes_row};
    p.down_deltas = {c.dd->as<float>(), c.dd_row}; p.down_sums = {c.ds->as<int>(), c.dd_row}; p.rawdots = {c.raw->as<float>(), c.raw_row};
    p.csr = {c.dcsr.summary, c.dcsr.offsets, c.dcsr.touched, c.dcsr.rows, c.dcsr.slots, c.dcsr.experts, c.dcsr.begins};
}

bool g_report = false;
unsigned g_failures = 0;
void expect(bool ok, const std::string& what) {
    if (ok) return;
    ++g_failures;
    if (!g_report) throw std::runtime_error("GSQ fixture: " + what);
    std::fprintf(stderr, "GSQ fixture MISMATCH: %s\n", what.c_str());
}

// --- launches (the launcher's geometry; exact batch grid instead of the T * top_k upper bound) ---
std::pair<Words, Words> run_prefill(Case& c) {
    const auto& sp = c.spec;
    const auto* gate = c.gate->as<unsigned char>(); const auto* up = c.up->as<unsigned char>(); const auto* down = c.down->as<unsigned char>();
    std::pair<Words, Words> traces;
    {
        Trace trace(c.csr.batches * (sp.N / 32U) * 256U);
        with_gate(c.type, [&]<std::uint32_t T>() {
            hipLaunchKernelGGL((kr::project_prefill_iq<T, 4U>), dim3(c.csr.batches, sp.N / 32U), dim3(256), 0, 0, gate, up, c.pre, sp.H, sp.N);
        });
        check(hipGetLastError()); check(hipDeviceSynchronize());
        traces.first = trace.packed();
    }
    {
        Trace trace(c.csr.batches * (sp.H / 64U) * 256U);
        hipLaunchKernelGGL((kr::down_prefill_iq<ga::type_iq4_nl, 8U>), dim3(c.csr.batches, sp.H / 64U), dim3(256), 0, 0, down, c.pre, sp.H, sp.N);
        check(hipGetLastError()); check(hipDeviceSynchronize());
        traces.second = trace.packed();
    }
    return traces;
}
template <unsigned G> void project_window(Case& c) {
    const auto& sp = c.spec;
    with_gate(c.type, [&]<std::uint32_t T>() {
        hipLaunchKernelGGL((kr::project_window_iq<T, G>), dim3(sp.N / 32U, c.csr.batches), dim3(256), 0, 0, c.gate->as<unsigned char>(),
            c.up->as<unsigned char>(), c.rows, c.dcsr, sp.T, sp.H, sp.N);
    });
}
template <unsigned R> void down_window(Case& c) {
    const auto& sp = c.spec;
    hipLaunchKernelGGL((kr::down_window_iq<ga::type_iq4_nl, R>), dim3(sp.H / 64U, c.csr.batches), dim3(256), 0, 0, c.down->as<unsigned char>(),
        c.rows, c.dcsr, sp.T, sp.H, sp.N);
}
void project_by(Case& c, unsigned g) { if (g == 2U) project_window<2U>(c); else if (g == 4U) project_window<4U>(c); else project_window<16U>(c); }
void down_by(Case& c, unsigned r) {
    if (r == 2U) down_window<2U>(c); else if (r == 4U) down_window<4U>(c); else if (r == 8U) down_window<8U>(c); else down_window<16U>(c);
}
std::pair<Words, Words> run_window(Case& c, unsigned g, unsigned r) {
    const auto& sp = c.spec;
    std::pair<Words, Words> traces;
    {
        Trace trace(c.csr.batches * (sp.N / 32U) * 256U);
        project_by(c, g); check(hipGetLastError()); check(hipDeviceSynchronize());
        traces.first = trace.packed();
    }
    {
        Trace trace(c.csr.batches * (sp.H / 64U) * 256U);
        down_by(c, r); check(hipGetLastError()); check(hipDeviceSynchronize());
        traces.second = trace.packed();
    }
    return traces;
}

std::size_t g_cases = 0;
// Independent window executions, one token at a time, retain the original
// payloads, activation bytes and both expert slots (including duplicates).
void run_row_windows(Case& c) {
    const auto& sp = c.spec;
    Buffer table(41U * sizeof(unsigned));
    auto* base = table.as<unsigned>();
    const kr::DeviceCsr csr{base, base, base, base + 32U, base + 34U, base + 36U, base + 38U};
    for (unsigned row = 0; row < sp.T; ++row) {
        const auto words = strix::wfmt_gsq::row_csr(c.csr.words, c.csr.batches, c.csr.rows,
            c.csr.slots, c.csr.experts, c.csr.begins, row);
        raw_upload(base, words.data(), sizeof words);
        kr::DeviceRows view{};
        view.codes[0] = c.codes->as<signed char>() + static_cast<std::size_t>(row) * sp.H;
        view.deltas[0] = c.deltas->as<float>() + static_cast<std::size_t>(row) * (sp.H / 32U);
        view.sums[0] = c.sums->as<int>() + static_cast<std::size_t>(row) * (sp.H / 32U);
        view.status[0] = c.status->as<unsigned>() + row;
        view.output[0] = reinterpret_cast<float*>(c.output->as<unsigned char>() + row * c.out_row);
        view.down_codes[0] = reinterpret_cast<signed char*>(c.dcodes->as<unsigned char>() + row * c.dcodes_row);
        view.down_deltas[0] = reinterpret_cast<float*>(c.dd->as<unsigned char>() + row * c.dd_row);
        view.down_sums[0] = reinterpret_cast<int*>(c.ds->as<unsigned char>() + row * c.dd_row);
        view.rawdots[0] = reinterpret_cast<float*>(c.raw->as<unsigned char>() + row * c.raw_row);
        with_gate(c.type, [&]<std::uint32_t T>() {
            hipLaunchKernelGGL((kr::project_window_iq<T, 2U>), dim3(sp.N / 32U, 2U), dim3(256), 0, 0,
                c.gate->as<unsigned char>(), c.up->as<unsigned char>(), view, csr, 1U, sp.H, sp.N);
        });
        check(hipGetLastError()); check(hipDeviceSynchronize());
        hipLaunchKernelGGL((kr::down_window_iq<ga::type_iq4_nl, 2U>), dim3(sp.H / 64U, 2U), dim3(256), 0, 0,
            c.down->as<unsigned char>(), view, csr, 1U, sp.H, sp.N);
        check(hipGetLastError()); check(hipDeviceSynchronize());
        expect(table.read<unsigned>(1U)[0] == 0U, "one-row window CSR status");
    }
}
void run_case(std::uint32_t type, const Spec& spec, bool window) {
    const std::string name = std::string("hip.gsq.") + type_name(type) + ".t" + std::to_string(spec.T) + ".h" + std::to_string(spec.H) +
        ".n" + std::to_string(spec.N) + (spec.dup ? ".dup" : "");
    const char* format = type == ga::type_iq2_s ? "gguf.iq2_s@1" : type == ga::type_iq3_xxs ? "gguf.iq3_xxs@1" :
        type == ga::type_iq3_s ? "gguf.iq3_s@1" : "gguf.iq4_xs@1";
    const std::initializer_list<std::uint64_t> shape{spec.T, spec.H, spec.N, kTopK, kExperts};
    Case c;
    c.spec = spec; c.type = type;
    context(name, shape, format);
    build_case(c, name);
    // Prefill arm: project_prefill_iq<T,4> + down_prefill_iq<IQ4_NL,8> (recorded; the reference of the self-checks).
    context(name + ".prefill", shape, format);
    c.clear();
    emit_partials(run_prefill(c));
    const auto reference = c.snapshot();
    emit_outputs(reference);
    if (!window) {
        c.clear();
        run_row_windows(c);
        expect(same(c.snapshot(), reference), name + ": one-row windows differ from prefill");
        ++g_cases;
        return;
    }
    // Window arm as the launcher selects: G 2 / 16 for 2 / 16 rows, else 4; Rows 2 / 4 / 16 for 2 / 4 / 16 rows, else 8.
    const unsigned g = spec.T == 2U ? 2U : spec.T == 16U ? 16U : 4U;
    const unsigned r = spec.T == 2U ? 2U : spec.T == 4U ? 4U : spec.T == 16U ? 16U : 8U;
    context(name + ".window", shape, format);
    c.clear();
    emit_partials(run_window(c, g, r));
    const auto windowed = c.snapshot();
    emit_outputs(windowed);
    expect(same(windowed, reference), name + ": window (G=" + std::to_string(g) + ", Rows=" + std::to_string(r) + ") differs from prefill");
    // Quiet cross-checks: every other legal G / Rows (a down batch must fit its Rows).
    for (const unsigned gg : {2U, 4U, 16U})
        for (const unsigned rr : {2U, 4U, 8U, 16U}) {
            if (rr < c.csr.max_batch || (gg == g && rr == r)) continue;
            c.clear(); (void)run_window(c, gg, rr);
            expect(same(c.snapshot(), reference), name + ": window G=" + std::to_string(gg) + " Rows=" + std::to_string(rr) + " differs from prefill");
        }
    ++g_cases;
}
}  // namespace

int main() try {
    g_report = strix::envcfg::integer("STRIX_WFMT_GSQ_REPORT", 0) != 0;
    constexpr std::array<std::uint32_t, 4> gates{ga::type_iq2_s, ga::type_iq3_xxs, ga::type_iq3_s, ga::type_iq4_xs};
    // The launcher's gate types x the IQ4_NL down (the only IQ down); IQ3_S rows are 110 bytes per 256 weights.
    for (const auto type : gates) {
        for (const unsigned t : {1U, 2U, 7U, 8U, 9U, 31U, 32U, 33U, 63U, 64U, 65U}) run_case(type, {t, 512U, 128U, false}, t == 2U || t == 7U || t == 8U);
        for (const unsigned t : {4U, 16U}) run_case(type, {t, 512U, 128U, false}, true);
        for (const unsigned t : {2U, 9U, 33U}) run_case(type, {t, 512U, 128U, true}, false);   // a token routed twice to one expert
        for (const unsigned t : {8U, 33U}) run_case(type, {t, 256U, 64U, false}, t == 8U);    // single block, minimum tile
        for (const unsigned t : {4U, 33U}) run_case(type, {t, 768U, 96U, false}, t == 4U);    // three blocks, N = 3 down blocks
    }
    std::cout << "HIP GSQ IQ window/prefill: " << g_cases << " cases, " << g_failures << " mismatches\n";
    return g_failures ? 1 : 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
