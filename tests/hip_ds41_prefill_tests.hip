// deepseek41 prefill expert fixtures: device routes against build_prefill_routes, and the WMMA
// gate/up and down kernels against the decode window kernels (every intermediate bitwise: gate, up,
// A8 activations, raw slot vectors, combine) plus sampled host products. Synthetic experts always;
// real blk.20 (IQ2_XXS / Q2_K, Q8_0 shared expert) and DSpark mtp.0 (MXFP4) when STRIX_DS41_Q2_GGUF /
// STRIX_DS41_DSPARK_GGUF name the files. STRIX_DS41_BENCH=1 adds timings for T = 512 .. 8192 rows.

#include "strix/ds41_moe.hpp"
#include "strix/ds41_prefill.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_prefill.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using strix::ds41::WeightFormat;
int failures = 0;
void expect(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}
void check(hipError_t status, const char* what) {
    if (status != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
}

template <class T>
struct Device {
    T* data = nullptr;
    std::size_t count = 0;
    explicit Device(std::size_t n) : count(n) {
        check(hipMalloc(reinterpret_cast<void**>(&data), std::max<std::size_t>(n, 1U) * sizeof(T)), "hipMalloc");
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() { (void)hipFree(data); }
    void upload(const T* host, std::size_t n) { check(hipMemcpy(data, host, n * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    void upload(const std::vector<T>& host) { upload(host.data(), host.size()); }
    std::vector<T> download() const {
        std::vector<T> out(count);
        check(hipMemcpy(out.data(), data, count * sizeof(T), hipMemcpyDeviceToHost), "download");
        return out;
    }
    void fill_pattern() { check(hipMemset(data, 0x7b, count * sizeof(T)), "poison"); }
};

template <class T>
std::size_t bit_mismatches(const std::vector<T>& a, const std::vector<T>& b) {
    std::size_t bad = a.size() != b.size();
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) bad += std::memcmp(&a[i], &b[i], sizeof(T)) != 0;
    return bad;
}

std::vector<std::uint8_t> read_range(const std::string& path, std::uint64_t offset, std::uint64_t bytes) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path);
    file.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::uint8_t> out(bytes);
    file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    if (!file) throw std::runtime_error("short read in " + path);
    return out;
}

// ---- Experts ----

struct Experts {
    WeightFormat gate_up = WeightFormat::iq2_xxs, down = WeightFormat::q2_k;
    std::uint32_t experts = 0, hidden = 0, inter = 0;
    std::vector<std::uint8_t> host[3]; // gate, up, down: whole [experts] tensors
    std::uint64_t gate_row() const { return hidden / strix::ds41::block_elements(gate_up) * strix::ds41::block_bytes(gate_up); }
    std::uint64_t down_row() const { return inter / strix::ds41::block_elements(down) * strix::ds41::block_bytes(down); }
    strix::ds41::ExpertTensors tensors() const {
        strix::ds41::ExpertTensors t;
        t.gate_up_format = gate_up; t.down_format = down; t.hidden = hidden; t.inter = inter;
        t.gate = host[0]; t.up = host[1]; t.down = host[2];
        return t;
    }
};
struct DeviceExperts {
    const Experts& e;
    Device<std::uint8_t> gate, up, down;
    explicit DeviceExperts(const Experts& x) : e(x), gate(x.host[0].size()), up(x.host[1].size()), down(x.host[2].size()) {
        gate.upload(x.host[0]); up.upload(x.host[1]); down.upload(x.host[2]);
    }
};

std::vector<std::uint8_t> synthetic_blocks(WeightFormat format, std::size_t blocks, unsigned seed) {
    using namespace strix::ds41;
    std::mt19937 random(seed);
    std::vector<std::uint8_t> bytes(blocks * block_bytes(format));
    for (auto& b : bytes) b = static_cast<std::uint8_t>(random());
    const auto size = block_bytes(format);
    for (std::size_t b = 0; b < blocks; ++b) {
        auto* p = bytes.data() + b * size;
        // Tame scales: FP16 exponents near 2^-12 .. 2^-9, E8M0 near 2^-10.
        if (format == WeightFormat::iq2_xxs || format == WeightFormat::q8_0) p[1] = static_cast<std::uint8_t>(0x10U | (p[1] & 0x83U));
        else if (format == WeightFormat::q2_k) {
            p[81] = static_cast<std::uint8_t>(0x10U | (p[81] & 0x83U));
            p[83] = static_cast<std::uint8_t>(0x10U | (p[83] & 0x83U));
        } else p[0] = static_cast<std::uint8_t>(118U + p[0] % 8U);
    }
    return bytes;
}
Experts synthetic_experts(WeightFormat gate_up, WeightFormat down, std::uint32_t experts, std::uint32_t hidden, std::uint32_t inter,
                          unsigned seed) {
    using namespace strix::ds41;
    Experts e;
    e.gate_up = gate_up; e.down = down; e.experts = experts; e.hidden = hidden; e.inter = inter;
    const auto gate_blocks = static_cast<std::size_t>(experts) * inter * (hidden / block_elements(gate_up));
    const auto down_blocks = static_cast<std::size_t>(experts) * hidden * (inter / block_elements(down));
    e.host[0] = synthetic_blocks(gate_up, gate_blocks, seed);
    e.host[1] = synthetic_blocks(gate_up, gate_blocks, seed + 1U);
    e.host[2] = synthetic_blocks(down, down_blocks, seed + 2U);
    return e;
}
WeightFormat format_of(std::uint32_t ggml_type) {
    if (ggml_type == 16U) return WeightFormat::iq2_xxs;
    if (ggml_type == 10U) return WeightFormat::q2_k;
    if (ggml_type == 8U) return WeightFormat::q8_0;
    if (ggml_type == 39U) return WeightFormat::mxfp4;
    throw std::runtime_error("unexpected expert tensor type " + std::to_string(ggml_type));
}
Experts file_experts(const std::string& path, const std::string& gate, const std::string& up, const std::string& down) {
    const auto file = strix::gguf::read_file(path);
    std::map<std::string, const strix::gguf::TensorInfo*> tensors;
    for (const auto& t : file.tensors) tensors[t.name] = &t;
    const auto find = [&](const std::string& name) {
        const auto found = tensors.find(name);
        if (found == tensors.end()) throw std::runtime_error("missing tensor " + name);
        return found->second;
    };
    const auto *g = find(gate), *u = find(up), *d = find(down);
    Experts e;
    e.gate_up = format_of(g->ggml_type); e.down = format_of(d->ggml_type);
    e.hidden = static_cast<std::uint32_t>(g->dimensions.at(0));
    e.inter = static_cast<std::uint32_t>(g->dimensions.at(1));
    e.experts = g->dimensions.size() > 2U ? static_cast<std::uint32_t>(g->dimensions[2]) : 1U;
    e.host[0] = read_range(path, g->absolute_offset, g->byte_size);
    e.host[1] = read_range(path, u->absolute_offset, u->byte_size);
    e.host[2] = read_range(path, d->absolute_offset, d->byte_size);
    if (e.host[0].size() != e.gate_row() * e.inter * e.experts || e.host[2].size() != e.down_row() * e.hidden * e.experts)
        throw std::runtime_error(gate + ": byte sizes do not match the geometry");
    return e;
}

// ---- Cases ----

std::vector<float> activations(std::uint32_t rows, std::uint32_t width, unsigned seed) {
    std::mt19937 random(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> values(static_cast<std::size_t>(rows) * width);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = normal(random) * std::ldexp(1.0F, static_cast<int>((i / 32U * 7U) % 11U) - 5);
    for (std::uint32_t r = 0; r < rows; r += 7U) { // a zero group and a tiny group in some rows
        auto* row = values.data() + static_cast<std::size_t>(r) * width;
        std::fill(row, row + 32, 0.0F);
        for (unsigned j = 32; j < 64; ++j) row[j] = 1e-37F * static_cast<float>(j);
    }
    return values;
}
// top_k distinct experts per row; skew > 0 concentrates the choice on a few experts (long CSR lists).
std::vector<std::uint32_t> routing(std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts, double skew, unsigned seed) {
    std::mt19937 random(seed);
    std::vector<double> weight(experts);
    for (std::uint32_t e = 0; e < experts; ++e) weight[e] = std::exp(-skew * static_cast<double>(e % 17U));
    std::vector<std::uint32_t> ids;
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::vector<std::uint32_t> row;
        while (row.size() < top_k) {
            std::discrete_distribution<std::uint32_t> pick(weight.begin(), weight.end());
            const auto e = pick(random);
            if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e);
        }
        ids.insert(ids.end(), row.begin(), row.end());
    }
    return ids;
}

struct RouteBuffers {
    Device<std::uint32_t> ids, counts, begins, entry_rows, entry_slots, batch_expert, batch_begin, batch_count, batch_total, order, low;
    strix::ds41::PrefillRoutesBuffers b;
    RouteBuffers(std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts, std::uint32_t batch_rows)
        : ids(static_cast<std::size_t>(rows) * top_k), counts(experts), begins(experts + 1U),
          entry_rows(static_cast<std::size_t>(rows) * top_k), entry_slots(static_cast<std::size_t>(rows) * top_k),
          batch_expert(strix::ds41::prefill_max_batches(rows, top_k, experts, batch_rows)),
          batch_begin(batch_expert.count + 1U), batch_count(batch_expert.count), batch_total(1), order(static_cast<std::size_t>(rows) * top_k),
          low(rows) {
        b = {counts.data, begins.data, entry_rows.data, entry_slots.data, batch_expert.data, batch_begin.data, batch_count.data,
             batch_total.data, order.data, low.data};
    }
};

// Device routes against the host builder.
bool check_routes(const RouteBuffers& d, const strix::ds41::PrefillRoutes& h, const std::string& label) {
    const auto total = d.batch_total.download()[0];
    const auto n = static_cast<std::ptrdiff_t>(total);
    auto be = d.batch_expert.download(), bb = d.batch_begin.download(), bc = d.batch_count.download();
    const bool ok = d.begins.download() == h.begins && d.entry_rows.download() == h.entry_rows && d.entry_slots.download() == h.entry_slots &&
        total == h.batch_expert.size() && std::equal(be.begin(), be.begin() + n, h.batch_expert.begin()) &&
        std::equal(bb.begin(), bb.begin() + n, h.batch_begin.begin()) && std::equal(bc.begin(), bc.begin() + n, h.batch_count.begin()) &&
        d.order.download() == h.order && d.low.download() == h.low_counts;
    expect(ok, label + ": device routes equal build_prefill_routes");
    return ok;
}

struct Outputs {
    std::vector<float> gate, up, raw, y;
    std::vector<std::int8_t> codes;
    std::vector<float> deltas;
    std::vector<std::int32_t> sums;
};

// Prefill path: device routes, WMMA gate/up, WMMA down, combine.
struct PrefillRun {
    Device<float> gate, up, raw, y;
    Device<std::int8_t> act_codes;
    Device<float> act_deltas;
    Device<std::int32_t> act_sums;
    PrefillRun(std::size_t slots, std::uint32_t rows, std::uint32_t hidden, std::uint32_t inter, bool keep)
        : gate(keep ? slots * inter : 1U), up(keep ? slots * inter : 1U), raw(slots * hidden), y(static_cast<std::size_t>(rows) * hidden),
          act_codes(slots * inter), act_deltas(slots * inter / 32U), act_sums(slots * inter / 32U) {}
};
struct Inputs {
    std::uint32_t rows = 0, top_k = 0;
    Device<std::int8_t> codes;
    Device<float> deltas;
    Device<std::int32_t> sums;
    Device<float> weights, shared;
    Inputs(std::uint32_t r, std::uint32_t k, std::uint32_t hidden, unsigned seed)
        : rows(r), top_k(k), codes(static_cast<std::size_t>(r) * hidden), deltas(static_cast<std::size_t>(r) * hidden / 32U),
          sums(static_cast<std::size_t>(r) * hidden / 32U), weights(static_cast<std::size_t>(r) * k), shared(static_cast<std::size_t>(r) * hidden) {
        const auto x = activations(r, hidden, seed);
        Device<float> values(x.size());
        values.upload(x);
        strix::ds41::enqueue_quantize_a8(values.data, hidden, r, codes.data, deltas.data, sums.data, nullptr);
        std::mt19937 random(seed + 7U);
        std::uniform_real_distribution<float> w(0.05F, 0.6F);
        std::normal_distribution<float> normal(0.0F, 0.5F);
        std::vector<float> wh(weights.count), sh(shared.count);
        for (auto& v : wh) v = w(random);
        for (auto& v : sh) v = normal(random);
        weights.upload(wh); shared.upload(sh);
        check(hipDeviceSynchronize(), "inputs");
    }
};

void enqueue_prefill(const DeviceExperts& x, const Inputs& in, const strix::ds41::PrefillRoutesView& routes, PrefillRun& run,
                     bool keep, hipEvent_t* events = nullptr) {
    using namespace strix::ds41;
    const auto& e = x.e;
    if (events) check(hipEventRecord(events[0], nullptr), "event");
    PrefillGateUpView gu;
    gu.gate = x.gate.data; gu.up = x.up.data; gu.expert_stride_bytes = e.gate_row() * e.inter; gu.row_bytes = e.gate_row();
    gu.hidden = e.hidden; gu.inter = e.inter; gu.routes = routes; gu.codes = in.codes.data; gu.deltas = in.deltas.data;
    gu.gate_out = keep ? run.gate.data : nullptr; gu.up_out = keep ? run.up.data : nullptr;
    gu.act_codes = run.act_codes.data; gu.act_deltas = run.act_deltas.data; gu.act_sums = run.act_sums.data;
    enqueue_prefill_gate_up(e.gate_up, gu, nullptr);
    if (events) check(hipEventRecord(events[1], nullptr), "event");
    PrefillDownView dn;
    dn.weights = x.down.data; dn.expert_stride_bytes = e.down_row() * e.hidden; dn.row_bytes = e.down_row();
    dn.width = e.inter; dn.outputs = e.hidden; dn.routes = routes; dn.codes = run.act_codes.data; dn.deltas = run.act_deltas.data;
    dn.out = run.raw.data;
    enqueue_prefill_down(e.down, dn, nullptr);
    if (events) check(hipEventRecord(events[2], nullptr), "event");
}

// Decode window path over chunks of at most 8 rows (the decode kernels' window size).
Outputs decode_windows(const DeviceExperts& x, const Inputs& in, const std::vector<std::uint32_t>& ids, std::uint32_t split) {
    using namespace strix::ds41;
    const auto& e = x.e;
    const std::size_t slots = static_cast<std::size_t>(in.rows) * in.top_k;
    Device<float> gate(slots * e.inter), up(slots * e.inter), raw(slots * e.hidden);
    Device<std::int8_t> act_codes(slots * e.inter);
    Device<float> act_deltas(slots * e.inter / 32U);
    Device<std::int32_t> act_sums(slots * e.inter / 32U);
    for (std::uint32_t first = 0; first < in.rows; first += 8U) {
        const auto n = std::min(8U, in.rows - first);
        const std::vector<std::uint32_t> chunk(ids.begin() + static_cast<std::ptrdiff_t>(first) * in.top_k,
                                               ids.begin() + static_cast<std::ptrdiff_t>(first + n) * in.top_k);
        const auto w = build_window_routes(chunk, n, in.top_k, split);
        // Window rows/slots are chunk-local: shift them to the global row and slot.
        std::vector<std::uint32_t> rows(w.rows), gslots(w.slots);
        for (auto& r : rows) r += first;
        for (auto& s : gslots) s += first * in.top_k;
        Device<std::uint32_t> experts(w.experts.size()), begins(w.begins.size()), drows(rows.size()), dslots(gslots.size());
        experts.upload(w.experts); begins.upload(w.begins); drows.upload(rows); dslots.upload(gslots);
        GateUpView gu;
        gu.gate = x.gate.data; gu.up = x.up.data; gu.expert_stride_bytes = e.gate_row() * e.inter; gu.row_bytes = e.gate_row();
        gu.hidden = e.hidden; gu.inter = e.inter;
        gu.routes = {experts.data, begins.data, drows.data, dslots.data, static_cast<std::uint32_t>(w.experts.size()), w.max_rows};
        gu.codes = in.codes.data; gu.deltas = in.deltas.data; gu.gate_out = gate.data; gu.up_out = up.data;
        gu.act_codes = act_codes.data; gu.act_deltas = act_deltas.data; gu.act_sums = act_sums.data;
        enqueue_gate_up(e.gate_up, gu, nullptr);
        RoutesProductView dn;
        dn.weights = x.down.data; dn.expert_stride_bytes = e.down_row() * e.hidden; dn.row_bytes = e.down_row();
        dn.width = e.inter; dn.outputs = e.hidden;
        dn.routes = {experts.data, begins.data, dslots.data, dslots.data, static_cast<std::uint32_t>(w.experts.size()), w.max_rows};
        dn.codes = act_codes.data; dn.deltas = act_deltas.data; dn.out = raw.data;
        enqueue_routes_product(e.down, dn, nullptr);
        check(hipDeviceSynchronize(), "decode window chunk");
    }
    Outputs o;
    o.gate = gate.download(); o.up = up.download(); o.raw = raw.download();
    o.codes = act_codes.download(); o.deltas = act_deltas.download(); o.sums = act_sums.download();
    return o;
}

// One case: routes (device = host), prefill = decode windows on every intermediate, sampled host
// products, and the combine of the prefill raw vectors against the host combine.
void check_case(const DeviceExperts& x, std::uint32_t rows, std::uint32_t top_k, std::uint32_t split, std::uint32_t batch_rows,
                double skew, unsigned seed, const std::string& label) {
    using namespace strix::ds41;
    const auto& e = x.e;
    const auto ids = routing(rows, top_k, e.experts, skew, seed);
    const auto host_routes = build_prefill_routes(ids, rows, top_k, e.experts, split, batch_rows);
    RouteBuffers rb(rows, top_k, e.experts, batch_rows);
    rb.ids.upload(ids);
    rb.batch_expert.fill_pattern(); rb.batch_begin.fill_pattern(); rb.batch_count.fill_pattern();
    enqueue_build_prefill_routes(rb.ids.data, rows, top_k, e.experts, split, batch_rows, rb.b, nullptr);
    check(hipDeviceSynchronize(), "routes");
    const bool routes_ok = check_routes(rb, host_routes, label);
    std::uint32_t longest = 0;
    for (std::uint32_t k = 0; k < e.experts; ++k) longest = std::max(longest, host_routes.begins[k + 1U] - host_routes.begins[k]);

    Inputs in(rows, top_k, e.hidden, seed + 11U);
    const std::size_t slots = static_cast<std::size_t>(rows) * top_k;
    PrefillRun run(slots, rows, e.hidden, e.inter, true);
    run.raw.fill_pattern(); run.act_codes.fill_pattern();
    const auto view = prefill_routes_view(rb.b, rows, top_k, e.experts, batch_rows);
    enqueue_prefill(x, in, view, run, true);
    CombineView c;
    c.raw = run.raw.data; c.shared = in.shared.data; c.weights = in.weights.data; c.order = rb.order.data; c.low_counts = rb.low.data;
    c.rows = rows; c.top_k = top_k; c.hidden = e.hidden; c.out = run.y.data;
    enqueue_combine(c, nullptr);
    check(hipDeviceSynchronize(), "prefill");
    const auto ref = decode_windows(x, in, ids, split);
    const auto gate = run.gate.download(), up = run.up.download(), raw = run.raw.download();
    const auto codes = run.act_codes.download();
    const auto deltas = run.act_deltas.download();
    const auto sums = run.act_sums.download();
    const auto gate_bad = bit_mismatches(gate, ref.gate) + bit_mismatches(up, ref.up);
    const auto act_bad = bit_mismatches(codes, ref.codes) + bit_mismatches(deltas, ref.deltas) + bit_mismatches(sums, ref.sums);
    const auto raw_bad = bit_mismatches(raw, ref.raw);
    // Host: sampled products and the combine.
    const auto x_codes = in.codes.download();
    const auto x_deltas = in.deltas.download();
    const auto tensors = e.tensors();
    std::mt19937 random(seed + 99U);
    std::size_t sample_bad = 0, samples = 0;
    const auto activation_of = [&](const std::vector<std::int8_t>& c8, const std::vector<float>& d, std::size_t index, std::uint32_t width) {
        A8Activation a;
        a.codes.assign(c8.begin() + static_cast<std::ptrdiff_t>(index * width), c8.begin() + static_cast<std::ptrdiff_t>((index + 1U) * width));
        a.deltas.assign(d.begin() + static_cast<std::ptrdiff_t>(index * width / 32U), d.begin() + static_cast<std::ptrdiff_t>((index + 1U) * width / 32U));
        return a;
    };
    for (unsigned n = 0; n < 48U; ++n) {
        const std::size_t slot = random() % slots, row = slot / top_k;
        const std::uint64_t expert = ids[slot];
        const auto xa = activation_of(x_codes, x_deltas, row, e.hidden);
        const auto ha = activation_of(codes, deltas, slot, e.inter);
        for (unsigned m = 0; m < 4U; ++m) {
            const std::uint64_t o = random() % e.inter, d = random() % e.hidden;
            const auto at = expert * e.gate_row() * e.inter + o * e.gate_row();
            const float g = product(e.gate_up, std::span<const std::uint8_t>(tensors.gate).subspan(at, e.gate_row()), xa, e.hidden);
            const float u = product(e.gate_up, std::span<const std::uint8_t>(tensors.up).subspan(at, e.gate_row()), xa, e.hidden);
            const auto dat = expert * e.down_row() * e.hidden + d * e.down_row();
            const float r = product(e.down, std::span<const std::uint8_t>(tensors.down).subspan(dat, e.down_row()), ha, e.inter);
            sample_bad += std::bit_cast<std::uint32_t>(g) != std::bit_cast<std::uint32_t>(gate[slot * e.inter + o]);
            sample_bad += std::bit_cast<std::uint32_t>(u) != std::bit_cast<std::uint32_t>(up[slot * e.inter + o]);
            sample_bad += std::bit_cast<std::uint32_t>(r) != std::bit_cast<std::uint32_t>(raw[slot * e.hidden + d]);
            samples += 3U;
        }
    }
    const auto y = run.y.download();
    const auto weights = in.weights.download(), shared = in.shared.download();
    std::size_t y_bad = 0;
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t o = 0; o < e.hidden; ++o) {
            float low = 0.0F, high = 0.0F;
            for (std::uint32_t k = 0; k < top_k; ++k) {
                const auto s = host_routes.order[r * top_k + k];
                const float term = weights[s] * raw[static_cast<std::size_t>(s) * e.hidden + o];
                if (k < host_routes.low_counts[r]) low = low + term;
                else high = high + term;
            }
            const float want = (shared[static_cast<std::size_t>(r) * e.hidden + o] + low) + high;
            y_bad += std::bit_cast<std::uint32_t>(want) != std::bit_cast<std::uint32_t>(y[static_cast<std::size_t>(r) * e.hidden + o]);
        }
    const bool ok = routes_ok && gate_bad == 0U && act_bad == 0U && raw_bad == 0U && sample_bad == 0U && y_bad == 0U;
    expect(ok, label + ": gate/up " + std::to_string(gate_bad) + ", A8 " + std::to_string(act_bad) + ", raw " + std::to_string(raw_bad) +
        ", host samples " + std::to_string(sample_bad) + "/" + std::to_string(samples) + ", y " + std::to_string(y_bad) + " differ");
    std::cout << label << ": T=" << rows << " top-" << top_k << " of " << e.experts << ", batches of " << batch_rows << " ("
              << host_routes.batch_expert.size() << " batches, longest expert " << longest << " entries): routes, gate/up, A8, raw, y "
              << (ok ? "bit-exact vs decode windows" : "MISMATCH") << " (host samples " << samples - sample_bad << "/" << samples << ")\n";
}

void check_chunked(const DeviceExperts& x, std::uint32_t rows, std::uint32_t top_k, std::uint32_t split, unsigned seed,
                   const std::string& label);
void check_q8_nan_payloads();

// Rows narrower than the 8 group classes of a wave pair (hidden 32 = one group, inter 64 = two): some waves own no group.
// Gate/up alone (a down of 32 outputs is not a valid view), per-form default and every product on the shared core against
// the kernels of the engine, bit for bit; the policy names and the rejection of an unknown one.
void check_narrow_gate_up() {
    using namespace strix::ds41;
    const auto e = synthetic_experts(WeightFormat::q8_0, WeightFormat::q8_0, 1, 32, 64, 45);
    const DeviceExperts x(e);
    const std::uint32_t rows = 37, top_k = 1, batch_rows = 16;
    const auto ids = routing(rows, top_k, e.experts, 0.0, 12);
    RouteBuffers rb(rows, top_k, e.experts, batch_rows);
    rb.ids.upload(ids);
    enqueue_build_prefill_routes(rb.ids.data, rows, top_k, e.experts, 1, batch_rows, rb.b, nullptr);
    const auto view = prefill_routes_view(rb.b, rows, top_k, e.experts, batch_rows);
    Inputs in(rows, top_k, e.hidden, 23);
    struct Got { std::vector<float> gate, up, deltas; std::vector<std::int8_t> codes; std::vector<std::int32_t> sums; };
    const auto run_arm = [&](const char* policy) {
        set_pf_core(policy);
        PrefillRun run(rows, rows, 128, e.inter, true);
        run.gate.fill_pattern(); run.up.fill_pattern(); run.act_codes.fill_pattern(); run.act_deltas.fill_pattern(); run.act_sums.fill_pattern();
        PrefillGateUpView gu;
        gu.gate = x.gate.data; gu.up = x.up.data; gu.expert_stride_bytes = e.gate_row() * e.inter; gu.row_bytes = e.gate_row();
        gu.hidden = e.hidden; gu.inter = e.inter; gu.routes = view; gu.codes = in.codes.data; gu.deltas = in.deltas.data;
        gu.gate_out = run.gate.data; gu.up_out = run.up.data;
        gu.act_codes = run.act_codes.data; gu.act_deltas = run.act_deltas.data; gu.act_sums = run.act_sums.data;
        enqueue_prefill_gate_up(e.gate_up, gu, nullptr);
        check(hipDeviceSynchronize(), "narrow gate/up");
        return Got{run.gate.download(), run.up.download(), run.act_deltas.download(), run.act_codes.download(), run.act_sums.download()};
    };
    const auto priv = run_arm("0");
    std::size_t bad = 0;
    for (const char* policy : {static_cast<const char*>(nullptr), "moe", "1"}) {
        const auto got = run_arm(policy);
        bad += bit_mismatches(got.gate, priv.gate) + bit_mismatches(got.up, priv.up) + bit_mismatches(got.codes, priv.codes) +
               bit_mismatches(got.deltas, priv.deltas) + bit_mismatches(got.sums, priv.sums);
    }
    bool rejected = false;
    try { set_pf_core("mo"); } catch (const std::invalid_argument&) { rejected = true; }
    try { set_pf_core(""); rejected = false; } catch (const std::invalid_argument&) {}
    set_pf_core(-1);
    expect(bad == 0U && rejected, "narrow Q8_0 gate/up h32 i64: " + std::to_string(bad) + " differ, unknown policy " +
        (rejected ? "rejected" : "ACCEPTED"));
    std::cout << "synthetic Q8_0 narrow gate/up h32 i64: default / moe / 1 against the engine kernels "
              << (bad == 0U ? "bit-exact" : "MISMATCH") << ", unknown and empty policy " << (rejected ? "rejected" : "ACCEPTED") << "\n";
}

void test_synthetic() {
    {
        const auto e = synthetic_experts(WeightFormat::iq2_xxs, WeightFormat::q2_k, 16, 1024, 512, 5);
        const DeviceExperts x(e);
        check_case(x, 40, 6, 8, 16, 0.0, 1, "synthetic IQ2_XXS/Q2_K h1024");
        check_case(x, 97, 6, 8, 32, 0.4, 2, "synthetic IQ2_XXS/Q2_K h1024");
        check_case(x, 33, 6, 8, 16, 0.8, 3, "synthetic IQ2_XXS/Q2_K h1024");
        check_chunked(x, 150, 6, 8, 10, "synthetic IQ2_XXS/Q2_K h1024");
    }
    {
        const auto e = synthetic_experts(WeightFormat::iq2_xxs, WeightFormat::q2_k, 8, 5120, 2304, 15);
        const DeviceExperts x(e);
        check_case(x, 70, 6, 4, 32, 0.3, 4, "synthetic IQ2_XXS/Q2_K h5120");
        check_case(x, 21, 6, 4, 16, 0.0, 5, "synthetic IQ2_XXS/Q2_K h5120");
    }
    {
        // many experts, few rows: empty experts, 1-row experts, tails of 1..15 rows past 16 / 32 / 48 (split kernels v2:
        // class map, partial Q2_K chunk, bands of batches)
        const auto e = synthetic_experts(WeightFormat::iq2_xxs, WeightFormat::q2_k, 24, 5120, 2304, 65);
        const DeviceExperts x(e);
        check_case(x, 9, 6, 12, 16, 0.0, 61, "synthetic IQ2_XXS/Q2_K h5120 sparse");
        check_case(x, 60, 6, 12, 16, 1.2, 62, "synthetic IQ2_XXS/Q2_K h5120 skewed");
    }
    {
        const auto e = synthetic_experts(WeightFormat::mxfp4, WeightFormat::mxfp4, 12, 5120, 2304, 25);
        const DeviceExperts x(e);
        check_case(x, 50, 3, 6, 32, 0.5, 6, "synthetic MXFP4 h5120");
        check_case(x, 50, 3, 6, 16, 0.5, 9, "synthetic MXFP4 h5120");
    }
    {
        const auto e = synthetic_experts(WeightFormat::q8_0, WeightFormat::q8_0, 1, 5120, 2304, 35);
        const DeviceExperts x(e);
        check_case(x, 45, 1, 1, 32, 0.0, 7, "synthetic Q8_0 shared h5120");
        check_case(x, 37, 1, 1, 16, 0.0, 8, "synthetic Q8_0 shared h5120");
    }
    {
        // narrow rows: gate/up of 4 groups, down of 2 (width 64, 128 outputs); per-form default (the shared core for
        // these Q8_0 forms), the engine kernels and every product on the shared core, each against the decode windows
        const auto e = synthetic_experts(WeightFormat::q8_0, WeightFormat::q8_0, 1, 128, 64, 55);
        const DeviceExperts x(e);
        for (const char* policy : {static_cast<const char*>(nullptr), "0", "1"}) {
            strix::ds41::set_pf_core(policy);
            const std::string name = std::string("synthetic Q8_0 narrow h128 i64 PF_CORE=") + (policy ? policy : "unset");
            check_case(x, 37, 1, 1, 16, 0.0, 13, name);
            check_case(x, 45, 1, 1, 32, 0.0, 14, name);
        }
        strix::ds41::set_pf_core(-1);
    }
    {
        // IQ2_XXS gate/up width that is not a multiple of 1024 (the shared core loads whole chunks): the per-form default
        // keeps the engine kernels; default and PF_CORE=0, partial route batches included, against the decode windows
        const auto e = synthetic_experts(WeightFormat::iq2_xxs, WeightFormat::q2_k, 8, 1280, 512, 75);
        const DeviceExperts x(e);
        for (const char* policy : {static_cast<const char*>(nullptr), "0"}) {
            strix::ds41::set_pf_core(policy);
            const std::string name = std::string("synthetic IQ2_XXS/Q2_K h1280 i512 PF_CORE=") + (policy ? policy : "unset");
            expect(!strix::ds41::prefill_gate_up_on_core(WeightFormat::iq2_xxs, 1280U, 16U), name + ": gate/up routed to the shared core");
            check_case(x, 37, 6, 4, 16, 0.3, 71, name);
            check_case(x, 9, 6, 4, 16, 0.0, 72, name);
        }
        strix::ds41::set_pf_core(nullptr);
        expect(strix::ds41::prefill_gate_up_on_core(WeightFormat::iq2_xxs, 1024U, 16U) ==
                   strix::ds41::prefill_down_on_core(WeightFormat::q2_k, 512U, 16U),
               "IQ2_XXS h1024 gate/up and Q2_K down: per-form default routes differ");
        strix::ds41::set_pf_core(-1);
    }
    check_narrow_gate_up();
    check_q8_nan_payloads();
}

// Q8_0 split kernels v2 (default) against the v1 split kernels (STRIX_DS41_PREFILL_MOE_V2_Q8=0) with NaNs of distinct
// payloads and signs in about 1 % of the weight scales (gate, up, down) and of the activation deltas, spread over all
// lane classes so differing NaNs meet in the products, the epilogue and the reduction tree: every intermediate (gate,
// up, A8 codes / deltas / sums, down) bit for bit, except the payload / sign of a NaN result.
void check_q8_nan_payloads() {
    using namespace strix::ds41;
    auto e = synthetic_experts(WeightFormat::q8_0, WeightFormat::q8_0, 1, 5120, 2304, 45);
    unsigned marked = 0;
    for (auto& bytes : e.host)
        for (std::size_t b = 0; b * 34U < bytes.size(); ++b)
            if ((b * 2654435761ULL) % 97U == 0U) {
                const auto nan = static_cast<std::uint16_t>(((b & 1U) << 15U) | 0x7c00U | (1U + b % 0x3ffU));
                bytes[b * 34U] = static_cast<std::uint8_t>(nan & 0xffU);
                bytes[b * 34U + 1U] = static_cast<std::uint8_t>(nan >> 8U);
                ++marked;
            }
    const DeviceExperts x(e);
    for (const std::uint32_t batch_rows : {16U, 32U}) {
        const std::uint32_t rows = batch_rows == 16U ? 37U : 45U;
        const auto ids = routing(rows, 1, 1, 0.0, 70U + batch_rows);
        RouteBuffers rb(rows, 1, 1, batch_rows);
        rb.ids.upload(ids);
        enqueue_build_prefill_routes(rb.ids.data, rows, 1, 1, 1, batch_rows, rb.b, nullptr);
        Inputs in(rows, 1, e.hidden, 71U + batch_rows);
        auto deltas = in.deltas.download();
        for (std::size_t i = 0; i < deltas.size(); ++i)
            if ((i * 40503U) % 53U == 0U)
                deltas[i] = std::bit_cast<float>(static_cast<std::uint32_t>(((i & 1U) << 31U) | 0x7f800000U | (1U + i % 0x3fffffU)));
        in.deltas.upload(deltas);
        const auto view = prefill_routes_view(rb.b, rows, 1, 1, batch_rows);
        const auto run_arm = [&](const char* q8) {
            if (q8) setenv("STRIX_DS41_PREFILL_MOE_V2_Q8", q8, 1);
            else unsetenv("STRIX_DS41_PREFILL_MOE_V2_Q8");
            auto run = std::make_unique<PrefillRun>(rows, rows, e.hidden, e.inter, true);
            run->raw.fill_pattern(); run->gate.fill_pattern(); run->up.fill_pattern(); run->act_codes.fill_pattern();
            enqueue_prefill(x, in, view, *run, true);
            check(hipDeviceSynchronize(), "q8 nan prefill");
            return run;
        };
        const auto v1 = run_arm("0");
        const auto v2 = run_arm(nullptr);
        unsetenv("STRIX_DS41_PREFILL_MOE_V2_Q8");
        // NaN results: the same positions; their payload / sign may differ (two-NaN additions return one operand's NaN and
        // the compiler may commute a commutative add). Every non-NaN value and every A8 code / sum bit for bit.
        std::size_t value_bad = 0, payload_only = 0, nans = 0;
        const auto compare = [&](const std::vector<float>& a, const std::vector<float>& b) {
            value_bad += a.size() != b.size();
            for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
                nans += std::isnan(b[i]);
                if (std::bit_cast<std::uint32_t>(a[i]) == std::bit_cast<std::uint32_t>(b[i])) continue;
                if (std::isnan(a[i]) && std::isnan(b[i])) ++payload_only;
                else ++value_bad;
            }
        };
        compare(v1->gate.download(), v2->gate.download());
        compare(v1->up.download(), v2->up.download());
        compare(v1->act_deltas.download(), v2->act_deltas.download());
        compare(v1->raw.download(), v2->raw.download());
        value_bad += bit_mismatches(v1->act_codes.download(), v2->act_codes.download()) + bit_mismatches(v1->act_sums.download(), v2->act_sums.download());
        expect(value_bad == 0U && nans > 0U, "Q8_0 NaN inputs, batches of " + std::to_string(batch_rows) + ": " + std::to_string(value_bad) +
            " values differ beyond NaN payload (" + std::to_string(payload_only) + " NaN payload/sign only, " + std::to_string(nans) + " NaN values)");
        std::cout << "synthetic Q8_0 shared h5120 NaN inputs (" << marked << " NaN scales, batches of " << batch_rows << ", " << nans
                  << " NaN results, " << payload_only << " differ in NaN payload/sign only): every non-NaN value, NaN position, A8 code and sum "
                  << "bit-exact v2 vs V2_Q8=0" << std::endl;
    }
}

// Lane mapping of v_permlanex16 with identity selects (diagnostic).
__global__ void permlane_probe(int* out) {
    const auto lane = static_cast<unsigned>(threadIdx.x);
    out[lane] = static_cast<int>(__builtin_amdgcn_permlanex16(lane, lane, 0x76543210U, 0xfedcba98U, false, false));
    out[32U + lane] = __shfl_xor(static_cast<int>(lane), 16, 32);
}
void probe_permlane() {
    Device<int> out(64);
    hipLaunchKernelGGL(permlane_probe, dim3(1), dim3(32), 0, nullptr, out.data);
    check(hipDeviceSynchronize(), "probe");
    const auto v = out.download();
    bool same = true;
    for (int i = 0; i < 32; ++i) same = same && v[static_cast<std::size_t>(i)] == (i ^ 16) && v[32U + static_cast<std::size_t>(i)] == (i ^ 16);
    expect(same, "v_permlanex16 with identity selects reads lane l ^ 16");
}

// ---- Timing ----

__global__ void flush_kernel(const int* buffer, std::size_t count, int* sink) {
    int sum = 0;
    for (std::size_t i = blockIdx.x * 256ULL + threadIdx.x; i < count; i += gridDim.x * 256ULL) sum += buffer[i];
    if (sum == 0x5a5a5a5a) *sink = sum;
}
float median_of(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2U]; }

void bench(const DeviceExperts& x, std::uint32_t top_k, std::uint32_t split, const std::string& label) {
    using namespace strix::ds41;
    const auto& e = x.e;
    Device<int> flush(64U * 1024U * 1024U), sink(1);
    check(hipMemset(flush.data, 0, flush.count * sizeof(int)), "flush");
    hipEvent_t ev[3];
    for (auto& v : ev) check(hipEventCreate(&v), "event");
    const double peak = 57.0e12;
    std::vector<std::uint32_t> sizes{512U, 1024U, 2048U, 4096U, 8192U};
    if (const char* only = std::getenv("STRIX_DS41_BENCH_T"); only && *only) sizes = {static_cast<std::uint32_t>(std::strtoul(only, nullptr, 10))};
    for (const std::uint32_t rows : sizes) {
        for (const std::uint32_t batch_rows : {16U, 32U}) {
            const auto ids = routing(rows, top_k, e.experts, 0.0, 500U + rows);
            RouteBuffers rb(rows, top_k, e.experts, batch_rows);
            rb.ids.upload(ids);
            enqueue_build_prefill_routes(rb.ids.data, rows, top_k, e.experts, split, batch_rows, rb.b, nullptr);
            Inputs in(rows, top_k, e.hidden, 600U + rows);
            const std::size_t slots = static_cast<std::size_t>(rows) * top_k;
            PrefillRun run(slots, rows, e.hidden, e.inter, false);
            const auto view = prefill_routes_view(rb.b, rows, top_k, e.experts, batch_rows);
            hipEvent_t r0, r1;
            check(hipEventCreate(&r0), "event"); check(hipEventCreate(&r1), "event");
            std::vector<float> gu, dn, rt;
            for (unsigned rep = 0; rep < 8U; ++rep) {
                hipLaunchKernelGGL(flush_kernel, dim3(1024), dim3(256), 0, nullptr, flush.data, flush.count, sink.data);
                check(hipEventRecord(r0, nullptr), "event");
                enqueue_build_prefill_routes(rb.ids.data, rows, top_k, e.experts, split, batch_rows, rb.b, nullptr);
                check(hipEventRecord(r1, nullptr), "event");
                enqueue_prefill(x, in, view, run, false, ev);
                check(hipEventSynchronize(ev[2]), "sync");
                float a = 0, b = 0, c = 0;
                check(hipEventElapsedTime(&a, ev[0], ev[1]), "elapsed");
                check(hipEventElapsedTime(&b, ev[1], ev[2]), "elapsed");
                check(hipEventElapsedTime(&c, r0, r1), "elapsed");
                if (rep >= 2U) { gu.push_back(a); dn.push_back(b); rt.push_back(c); }
            }
            (void)hipEventDestroy(r0); (void)hipEventDestroy(r1);
            const double ops_gu = 2.0 * static_cast<double>(slots) * 2.0 * e.inter * e.hidden;
            const double ops_dn = 2.0 * static_cast<double>(slots) * e.inter * e.hidden;
            const double g = median_of(gu), d = median_of(dn), r = median_of(rt);
            std::cout << "PREFILL " << label << " T=" << rows << " batch=" << batch_rows << " routes_ms=" << r << " gate_up_ms=" << g
                      << " (" << ops_gu / (g * 1e9) << " TOPS, " << 100.0 * ops_gu / (g * 1e-3) / peak << "% of 57) down_ms=" << d
                      << " (" << ops_dn / (d * 1e9) << " TOPS, " << 100.0 * ops_dn / (d * 1e-3) / peak << "%) total_ms=" << g + d
                      << " tok/s_layer=" << rows / ((g + d) * 1e-3) << '\n';
        }
    }
    for (auto& v : ev) (void)hipEventDestroy(v);
}

// Decode-window throughput for the same T (chunks of 8 rows), for comparison.
void bench_decode(const DeviceExperts& x, std::uint32_t top_k, std::uint32_t split, std::uint32_t rows, const std::string& label) {
    const auto& e = x.e;
    const auto ids = routing(rows, top_k, e.experts, 0.0, 500U + rows);
    Inputs in(rows, top_k, e.hidden, 600U + rows);
    hipEvent_t a, b;
    check(hipEventCreate(&a), "event"); check(hipEventCreate(&b), "event");
    check(hipEventRecord(a, nullptr), "event");
    (void)decode_windows(x, in, ids, split);
    check(hipEventRecord(b, nullptr), "event");
    check(hipEventSynchronize(b), "sync");
    float ms = 0;
    check(hipEventElapsedTime(&ms, a, b), "elapsed");
    std::cout << "DECODE-WINDOWS " << label << " T=" << rows << " (chunks of 8, incl. host route building and copies) ms=" << ms << '\n';
    (void)hipEventDestroy(a); (void)hipEventDestroy(b);
}

// ---- Round 2: router of T rows, chunked MoE, expert halves ----

std::vector<float> read_f32(const std::string& path, const std::string& name, std::size_t count) {
    const auto file = strix::gguf::read_file(path);
    for (const auto& t : file.tensors)
        if (t.name == name) {
            if (t.ggml_type != 0U || t.byte_size != count * 4U) throw std::runtime_error(name + " is not the expected F32 tensor");
            const auto bytes = read_range(path, t.absolute_offset, t.byte_size);
            std::vector<float> out(count);
            std::memcpy(out.data(), bytes.data(), bytes.size());
            return out;
        }
    throw std::runtime_error("missing tensor " + name);
}

struct RouterWeights { std::vector<float> weight, bias, bias_image; std::uint32_t experts = 0, hidden = 0, top_k = 0; float scale = 1.5F; };
RouterWeights synthetic_router(std::uint32_t experts, std::uint32_t hidden, std::uint32_t top_k, unsigned seed) {
    std::mt19937 random(seed);
    std::normal_distribution<float> normal(0.0F, 0.02F);
    RouterWeights r;
    r.experts = experts; r.hidden = hidden; r.top_k = top_k;
    r.weight.resize(static_cast<std::size_t>(experts) * hidden);
    for (auto& v : r.weight) v = normal(random);
    r.bias.resize(experts); r.bias_image.resize(experts);
    for (std::uint32_t e = 0; e < experts; ++e) { r.bias[e] = 0.01F * static_cast<float>(e % 7U); r.bias_image[e] = -0.01F * static_cast<float>(e % 5U); }
    return r;
}

// Restores an environment variable (set, empty or unset) when it goes out of scope.
struct EnvRestore {
    std::string name, value;
    bool was_set = false;
    explicit EnvRestore(const char* n) : name(n) {
        const char* v = std::getenv(n);
        was_set = v != nullptr;
        if (v) value = v;
    }
    ~EnvRestore() {
        if (was_set) setenv(name.c_str(), value.c_str(), 1);
        else unsetenv(name.c_str());
    }
};

// Prefill router (T rows) = decode router (chunks of 8 rows) bitwise, plus host rows.
void check_router(const RouterWeights& rw, std::uint32_t rows, unsigned seed, const std::string& label) {
    using namespace strix::ds41;
    std::mt19937 random(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> x(static_cast<std::size_t>(rows) * rw.hidden);
    for (auto& v : x) v = normal(random);
    std::vector<std::uint8_t> image(rows);
    for (std::uint32_t r = 0; r < rows; ++r) image[r] = r % 11U == 3U ? 1U : 0U;
    Device<float> dx(x.size()), dw(rw.weight.size()), db(rw.experts), dbi(rw.experts);
    dx.upload(x); dw.upload(rw.weight); db.upload(rw.bias); dbi.upload(rw.bias_image);
    Device<std::uint8_t> dimg(rows);
    dimg.upload(image);
    const auto run = [&](bool prefill) {
        Device<float> scores(static_cast<std::size_t>(rows) * rw.experts), weights(static_cast<std::size_t>(rows) * rw.top_k);
        Device<std::uint32_t> ids(static_cast<std::size_t>(rows) * rw.top_k);
        for (std::uint32_t r0 = 0; r0 < rows; r0 += prefill ? rows : 8U) {
            const auto n = prefill ? rows : std::min(8U, rows - r0);
            RouterView v;
            v.x = dx.data + static_cast<std::size_t>(r0) * rw.hidden; v.weight = dw.data; v.bias = db.data; v.bias_image = dbi.data;
            v.image_rows = dimg.data + r0; v.rows = n; v.hidden = rw.hidden; v.experts = rw.experts; v.top_k = rw.top_k; v.route_scale = rw.scale;
            v.scores = scores.data + static_cast<std::size_t>(r0) * rw.experts;
            v.ids = ids.data + static_cast<std::size_t>(r0) * rw.top_k; v.weights = weights.data + static_cast<std::size_t>(r0) * rw.top_k;
            if (prefill) enqueue_prefill_router(v, nullptr);
            else enqueue_router(v, nullptr);
        }
        check(hipDeviceSynchronize(), "router");
        return std::make_tuple(scores.download(), ids.download(), weights.download());
    };
    // the v2 scores kernel (default) and the v1 kernel (STRIX_DS41_PREFILL_ROUTER_V2=0) against the decode router
    const EnvRestore restore("STRIX_DS41_PREFILL_ROUTER_V2");
    setenv("STRIX_DS41_PREFILL_ROUTER_V2", "0", 1);
    const auto [vs, vi, vw] = run(true);
    setenv("STRIX_DS41_PREFILL_ROUTER_V2", "1", 1);
    const auto [ps, pi, pw] = run(true);
    const auto [ds, di, dw2] = run(false);
    const auto bad = bit_mismatches(ps, ds) + bit_mismatches(pi, di) + bit_mismatches(pw, dw2) + bit_mismatches(vs, ds) +
                     bit_mismatches(vi, di) + bit_mismatches(vw, dw2);
    const std::uint32_t host_rows = std::min(rows, 3U);
    const auto host = reference_router(std::span<const float>(x.data(), static_cast<std::size_t>(host_rows) * rw.hidden), host_rows, rw.hidden,
                                       rw.weight, rw.experts, rw.bias, rw.bias_image, std::span<const std::uint8_t>(image.data(), host_rows),
                                       rw.top_k, rw.scale);
    std::size_t host_bad = 0;
    for (std::size_t i = 0; i < host.scores.size(); ++i) host_bad += std::bit_cast<std::uint32_t>(host.scores[i]) != std::bit_cast<std::uint32_t>(ps[i]);
    for (std::size_t i = 0; i < host.ids.size(); ++i) host_bad += host.ids[i] != pi[i] || std::bit_cast<std::uint32_t>(host.weights[i]) != std::bit_cast<std::uint32_t>(pw[i]);
    expect(bad == 0U && host_bad == 0U, label + ": prefill router vs decode router " + std::to_string(bad) + ", vs host " + std::to_string(host_bad) + " differ");
    std::cout << label << ": router T=" << rows << " (" << rw.experts << " experts, top-" << rw.top_k << ", image rows): scores, ids, weights "
              << (bad == 0U && host_bad == 0U ? "bit-exact vs decode router and host (scores kernels v2 and v1)" : "MISMATCH") << '\n';
}

strix::ds41::PrefillMoeWeights moe_weights(const DeviceExperts& x, std::uint32_t first_expert) {
    strix::ds41::PrefillMoeWeights w;
    const auto& e = x.e;
    w.gate_up_format = e.gate_up; w.down_format = e.down; w.gate_row_bytes = e.gate_row(); w.down_row_bytes = e.down_row();
    w.hidden = e.hidden; w.inter = e.inter;
    const std::uint64_t gs = e.gate_row() * e.inter * first_expert, ds = e.down_row() * e.hidden * first_expert;
    w.gate = x.gate.data + gs; w.up = x.up.data + gs; w.down = x.down.data + ds;
    return w;
}

// Chunked MoE (several chunk sizes) = one pass; expert halves (two owners) + canonical sum = combine.
void check_chunked(const DeviceExperts& x, std::uint32_t rows, std::uint32_t top_k, std::uint32_t split, unsigned seed, const std::string& label) {
    using namespace strix::ds41;
    const auto& e = x.e;
    const auto ids = routing(rows, top_k, e.experts, 0.2, seed);
    Inputs in(rows, top_k, e.hidden, seed + 3U);
    Device<std::uint32_t> dids(ids.size());
    dids.upload(ids);
    PrefillMoeInputs pin;
    pin.rows = rows; pin.ids = dids.data; pin.weights = in.weights.data; pin.codes = in.codes.data; pin.deltas = in.deltas.data;
    pin.shared = in.shared.data;
    const std::size_t n = static_cast<std::size_t>(rows) * e.hidden;
    const auto run_combine = [&](std::uint32_t chunk_rows, double window_mean, bool hist = true) {
        PrefillMoe moe(moe_weights(x, 0), top_k, e.experts, split, chunk_rows);
        moe.set_window_mean(window_mean);
        moe.set_select_hist(hist);
        Device<float> y(n);
        y.fill_pattern();
        for (std::uint32_t c = 0; c < moe.chunks(rows); ++c) moe.enqueue_combine(pin, c, y.data, nullptr);
        check(hipDeviceSynchronize(), "chunked MoE");
        return y.download();
    };
    const auto whole = run_combine(rows, 0.0);
    std::size_t bad = 0;
    for (const std::uint32_t chunk : {37U, 128U, rows / 2U + 1U}) bad += bit_mismatches(run_combine(chunk, 0.0), whole);
    // The decode window kernels over batches of 8 (the sparse-chunk path) give the same bits.
    for (const std::uint32_t chunk : {37U, rows}) bad += bit_mismatches(run_combine(chunk, 1e9), whole);
    // Histogram selector with both paths in one chunk (sparse experts on the window kernels, dense on WMMA), and the
    // previous whole-chunk mean selector.
    for (const double s_max : {2.0, 5.0, 12.0}) bad += bit_mismatches(run_combine(rows, s_max), whole);
    bad += bit_mismatches(run_combine(rows / 2U + 1U, 5.0, false), whole);
    // Two owners of the expert halves, each in its own chunking, then the canonical sum.
    PrefillMoe low(moe_weights(x, 0), top_k, e.experts, split, 64, 0, split);
    PrefillMoe high(moe_weights(x, split), top_k, e.experts, split, 97, split, e.experts);
    low.set_window_mean(0.0);
    high.set_window_mean(1e9); // one half on each path
    Device<float> h0(n), h1(n), y(n);
    for (std::uint32_t c = 0; c < low.chunks(rows); ++c) low.enqueue_half(pin, c, 0, h0.data, nullptr);
    for (std::uint32_t c = 0; c < high.chunks(rows); ++c) high.enqueue_half(pin, c, 1, h1.data, nullptr);
    enqueue_canonical_sum(in.shared.data, h0.data, h1.data, n, y.data, nullptr);
    check(hipDeviceSynchronize(), "halves");
    const auto halves_bad = bit_mismatches(y.download(), whole);
    // Joint form: rows [0, P) as the peer set and [P, rows) as the own set, one or several chunks; every row equals
    // its enqueue_half value.
    std::size_t joint_bad = 0;
    {
        const auto ref0 = h0.download(), ref1 = h1.download();
        const std::uint32_t P = rows / 3U + 1U;
        PrefillMoeInputs peer = pin, own = pin;
        peer.rows = P;
        own.rows = rows - P;
        own.ids = pin.ids + static_cast<std::size_t>(P) * top_k; own.weights = pin.weights + static_cast<std::size_t>(P) * top_k;
        own.codes = pin.codes + static_cast<std::size_t>(P) * e.hidden; own.deltas = pin.deltas + static_cast<std::size_t>(P) * (e.hidden / 32U);
        for (const std::uint32_t chunk : {rows, 53U}) {
            for (unsigned which = 0; which < 2U; ++which) {
                PrefillMoe moe(moe_weights(x, which == 0U ? 0U : split), top_k, e.experts, split, chunk, which == 0U ? 0U : split,
                               which == 0U ? split : e.experts);
                moe.set_window_mean(5.0);
                Device<float> op(static_cast<std::size_t>(P) * e.hidden), oo(n - static_cast<std::size_t>(P) * e.hidden);
                op.fill_pattern(); oo.fill_pattern();
                hipEvent_t done{};
                check(hipEventCreate(&done), "event");
                moe.enqueue_half_joint(peer, own, which, op.data, oo.data, nullptr, done);
                check(hipDeviceSynchronize(), "joint");
                (void)hipEventDestroy(done);
                const auto& ref = which == 0U ? ref0 : ref1;
                const auto a = op.download(), b = oo.download();
                for (std::size_t i = 0; i < a.size(); ++i) joint_bad += std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(ref[i]);
                for (std::size_t i = 0; i < b.size(); ++i) joint_bad += std::bit_cast<std::uint32_t>(b[i]) != std::bit_cast<std::uint32_t>(ref[a.size() + i]);
            }
        }
    }
    expect(joint_bad == 0U, label + ": joint peer/own half sums " + std::to_string(joint_bad) + " differ");
    std::cout << label << ": joint [peer; own] half sums (one chunk and chunks of 53, both halves) "
              << (joint_bad == 0U ? "bit-exact vs enqueue_half" : "MISMATCH") << '\n';
    // The single pass equals the decode windows (check_case already covers one pass vs decode).
    expect(bad == 0U && halves_bad == 0U, label + ": chunked " + std::to_string(bad) + ", halves " + std::to_string(halves_bad) + " differ");
    std::cout << label << ": T=" << rows << " chunks of 37/128/" << rows / 2U + 1U << " (WMMA), 37/all (window kernels) and two expert halves (chunks 64/97, one per path) + canonical sum: y "
              << (bad == 0U && halves_bad == 0U ? "bit-exact vs one pass" : "MISMATCH") << '\n';
}

// Whole routed MoE of one layer through PrefillMoe (routes + gate/up + down + combine) for small and
// large T; weight bytes read once per used expert, GB/s against them.
void bench_moe(const DeviceExperts& x, std::uint32_t top_k, std::uint32_t split, const std::string& label) {
    using namespace strix::ds41;
    const auto& e = x.e;
    Device<int> flush(64U * 1024U * 1024U), sink(1);
    check(hipMemset(flush.data, 0, flush.count * sizeof(int)), "flush");
    std::vector<std::uint32_t> sizes{128U, 256U, 512U, 1024U, 2414U, 4096U, 8192U};
    if (const char* only = std::getenv("STRIX_DS41_BENCH_T"); only && *only) sizes = {static_cast<std::uint32_t>(std::strtoul(only, nullptr, 10))};
    const double expert_bytes = static_cast<double>(e.gate_row()) * e.inter * 2.0 + static_cast<double>(e.down_row()) * e.hidden;
    for (const std::uint32_t rows : sizes) {
        const auto ids = routing(rows, top_k, e.experts, 0.0, 700U + rows);
        std::vector<char> used(e.experts, 0);
        for (const auto id : ids) used[id] = 1;
        const auto used_count = static_cast<double>(std::count(used.begin(), used.end(), 1));
        Inputs in(rows, top_k, e.hidden, 800U + rows);
        Device<std::uint32_t> dids(ids.size());
        dids.upload(ids);
        PrefillMoeInputs pin;
        pin.rows = rows; pin.ids = dids.data; pin.weights = in.weights.data; pin.codes = in.codes.data; pin.deltas = in.deltas.data;
        pin.shared = in.shared.data;
        Device<float> y(static_cast<std::size_t>(rows) * e.hidden);
        PrefillMoe moe(moe_weights(x, 0), top_k, e.experts, split, std::min(rows, 2048U));
        hipEvent_t a, b;
        check(hipEventCreate(&a), "event"); check(hipEventCreate(&b), "event");
        std::vector<float> ms;
        for (unsigned rep = 0; rep < 7U; ++rep) {
            hipLaunchKernelGGL(flush_kernel, dim3(1024), dim3(256), 0, nullptr, flush.data, flush.count, sink.data);
            check(hipEventRecord(a, nullptr), "event");
            for (std::uint32_t c = 0; c < moe.chunks(rows); ++c) moe.enqueue_combine(pin, c, y.data, nullptr);
            check(hipEventRecord(b, nullptr), "event");
            check(hipEventSynchronize(b), "sync");
            float t = 0;
            check(hipEventElapsedTime(&t, a, b), "elapsed");
            if (rep >= 2U) ms.push_back(t);
        }
        (void)hipEventDestroy(a); (void)hipEventDestroy(b);
        const double t = median_of(ms);
        const double ops = 2.0 * rows * top_k * 3.0 * e.inter * e.hidden;
        std::cout << "MOE " << label << " T=" << rows << " chunk=" << moe.chunk_rows() << " ms=" << t << " used_experts=" << used_count
                  << " GB/s=" << used_count * expert_bytes / (t * 1e6) << " TOPS=" << ops / (t * 1e9) << " scratch_MB="
                  << static_cast<double>(moe.scratch_bytes()) / 1e6 << '\n';
    }
}
void bench_router(const RouterWeights& rw, const std::string& label) {
    using namespace strix::ds41;
    for (const std::uint32_t rows : {512U, 2048U, 8192U}) {
        std::vector<float> x(static_cast<std::size_t>(rows) * rw.hidden, 0.5F);
        Device<float> dx(x.size()), dw(rw.weight.size()), db(rw.experts), scores(static_cast<std::size_t>(rows) * rw.experts),
            weights(static_cast<std::size_t>(rows) * rw.top_k);
        Device<std::uint32_t> ids(static_cast<std::size_t>(rows) * rw.top_k);
        dx.upload(x); dw.upload(rw.weight); db.upload(rw.bias);
        RouterView v;
        v.x = dx.data; v.weight = dw.data; v.bias = db.data; v.rows = rows; v.hidden = rw.hidden; v.experts = rw.experts; v.top_k = rw.top_k;
        v.route_scale = rw.scale; v.scores = scores.data; v.ids = ids.data; v.weights = weights.data;
        hipEvent_t a, b;
        check(hipEventCreate(&a), "event"); check(hipEventCreate(&b), "event");
        // interleaved: the v1 scores kernel (STRIX_DS41_PREFILL_ROUTER_V2=0), then the default (scores + selection)
        std::vector<float> ms[2];
        const EnvRestore restore("STRIX_DS41_PREFILL_ROUTER_V2");
        for (unsigned rep = 0; rep < 14U; ++rep) {
            const unsigned arm = rep % 2U;
            setenv("STRIX_DS41_PREFILL_ROUTER_V2", arm == 0U ? "0" : "1", 1);
            check(hipEventRecord(a, nullptr), "event");
            enqueue_prefill_router(v, nullptr);
            check(hipEventRecord(b, nullptr), "event");
            check(hipEventSynchronize(b), "sync");
            float t = 0;
            check(hipEventElapsedTime(&t, a, b), "elapsed");
            if (rep >= 2U) ms[arm].push_back(t);
        }
        (void)hipEventDestroy(a); (void)hipEventDestroy(b);
        std::cout << "ROUTER " << label << " T=" << rows << " ms=" << median_of(ms[1]) << " v1_ms=" << median_of(ms[0]) << '\n';
    }
}

// Rank-shaped expert calls of the pair prefill (one piece of T rows routed top-k over all experts; this rank owns experts
// [0, owned)), split kernels only (batches of 16). Arms (STRIX_DS41_BENCH_ARMS = "name:VAR=V,VAR=-;name2:..."; '-' unsets)
// (default: the engine kernels, PF_CORE=0, against the per-form default route; the v1 / v2 split kernels are explicit
// arms, "v1:STRIX_DS41_PF_CORE=0,STRIX_DS41_PREFILL_MOE_V2=0;v2:STRIX_DS41_PF_CORE=0,STRIX_DS41_PREFILL_MOE_V2=-")
// alternate within each repetition, every arm's outputs (A8 codes, deltas, sums, raw down vectors) compared bitwise with
// the first arm's. STRIX_DS41_BENCH_T (comma list), STRIX_DS41_BENCH_REPS, STRIX_DS41_BENCH_SKEW.
std::vector<std::string> split_on(const std::string& s, char c) {
    std::vector<std::string> out;
    std::string cur;
    for (const char ch : s) {
        if (ch == c) { out.push_back(cur); cur.clear(); } else cur += ch;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}
struct Arm { std::string name; std::vector<std::pair<std::string, std::string>> env; };
void apply_arm(const Arm& a) {
    for (const auto& [k, v] : a.env) {
        // resolved once per run in the library: the arms go through the harness override
        if (k == "STRIX_DS41_PF_CORE") strix::ds41::set_pf_core(v == "-" ? nullptr : v.c_str());
        else if (v == "-") unsetenv(k.c_str());
        else setenv(k.c_str(), v.c_str(), 1);
    }
}
void bench_rank(const DeviceExperts& x, std::uint32_t top_k, std::uint32_t owned, const std::string& label) {
    using namespace strix::ds41;
    const auto& e = x.e;
    std::vector<Arm> arms;
    const char* arms_env = std::getenv("STRIX_DS41_BENCH_ARMS");
    const bool default_arms = !(arms_env && *arms_env);
    for (const auto& spec : split_on(arms_env && *arms_env ? arms_env : "family:STRIX_DS41_PF_CORE=0;core:STRIX_DS41_PF_CORE=-", ';')) {
        Arm a;
        const auto colon = spec.find(':');
        a.name = spec.substr(0, colon);
        if (colon != std::string::npos)
            for (const auto& kv : split_on(spec.substr(colon + 1U), ',')) {
                const auto eq = kv.find('=');
                a.env.emplace_back(kv.substr(0, eq), eq == std::string::npos ? "-" : kv.substr(eq + 1U));
            }
        arms.push_back(a);
    }
    std::vector<std::uint32_t> sizes{2048U, 4096U};
    if (const char* t = std::getenv("STRIX_DS41_BENCH_T"); t && *t) {
        sizes.clear();
        for (const auto& v : split_on(t, ',')) sizes.push_back(static_cast<std::uint32_t>(std::stoul(v)));
    }
    const char* reps_env = std::getenv("STRIX_DS41_BENCH_REPS");
    const unsigned reps = reps_env && *reps_env ? static_cast<unsigned>(std::stoul(reps_env)) : 7U;
    const char* skew_env = std::getenv("STRIX_DS41_BENCH_SKEW");
    const double skew = skew_env && *skew_env ? std::stod(skew_env) : 0.15;
    Device<int> flush(64U * 1024U * 1024U), sink(1);
    check(hipMemset(flush.data, 0, flush.count * sizeof(int)), "flush");
    hipEvent_t ev[3];
    for (auto& v : ev) check(hipEventCreate(&v), "event");
    const double peak = 59.4e12; // 40 CUs x 512 iu8 ops/clk x 2.9 GHz
    for (const std::uint32_t rows : sizes) {
        const std::uint32_t batch_rows = 16U;
        const auto ids = routing(rows, top_k, e.experts, skew, 900U + rows);
        RouteBuffers rb(rows, top_k, owned, batch_rows), rb32(rows, top_k, owned, 32U);
        rb.ids.upload(ids);
        rb32.ids.upload(ids);
        enqueue_build_prefill_routes(rb.ids.data, rows, top_k, e.experts, owned, batch_rows, rb.b, nullptr, 0U, owned);
        enqueue_build_prefill_routes(rb32.ids.data, rows, top_k, e.experts, owned, 32U, rb32.b, nullptr, 0U, owned);
        check(hipDeviceSynchronize(), "routes");
        const auto view32 = prefill_routes_view(rb32.b, rows, top_k, owned, 32U);
        const auto counts = rb.counts.download();
        const auto batches = rb.batch_total.download()[0];
        std::uint64_t entries = 0;
        std::uint32_t empty = 0, small = 0, longest = 0;
        for (const auto c : counts) { entries += c; empty += c == 0U; small += c > 0U && c < 16U; longest = std::max(longest, c); }
        Inputs in(rows, top_k, e.hidden, 950U + rows);
        const std::size_t slots = static_cast<std::size_t>(rows) * top_k;
        PrefillRun run(slots, rows, e.hidden, e.inter, false);
        const auto view = prefill_routes_view(rb.b, rows, top_k, owned, batch_rows);
        std::cout << "RANK " << label << " T=" << rows << " owned " << owned << " entries " << entries << " (" << static_cast<double>(entries) / static_cast<double>(owned)
                  << "/expert, longest " << longest << ", empty " << empty << ", 1-15 rows " << small << ") batches " << batches
                  << " (32-row: " << rb32.batch_total.download()[0] << ")\n";
        std::vector<std::vector<float>> gu(arms.size()), dn(arms.size());
        std::vector<std::int8_t> ref_codes;
        std::vector<float> ref_deltas, ref_raw;
        std::vector<std::int32_t> ref_sums;
        bool ref_gu_core = false, ref_dn_core = false;
        for (unsigned rep = 0; rep < reps + 1U; ++rep)
            for (std::size_t a = 0; a < arms.size(); ++a) {
                apply_arm(arms[(a + rep) % arms.size()]);
                const std::size_t arm = (a + rep) % arms.size();
                const char* r2 = std::getenv("STRIX_DS41_PREFILL_MOE_R2"); // 32-row routes for the R2 arms
                if (rep == 0U) { run.raw.fill_pattern(); run.act_codes.fill_pattern(); }
                if (rep == 0U) {
                    // the route each arm really takes: the default arms are the engine kernels against the shared core
                    const bool gu_core = prefill_gate_up_on_core(e.gate_up, e.hidden, r2 && std::strcmp(r2, "1") == 0 ? 32U : batch_rows);
                    const bool dn_core = prefill_down_on_core(e.down, e.inter, r2 && std::strcmp(r2, "1") == 0 ? 32U : batch_rows);
                    if (a == 0U) { ref_gu_core = gu_core; ref_dn_core = dn_core; }
                    if (default_arms) expect(a == 0U ? !gu_core && !dn_core : gu_core || dn_core,
                        label + " T=" + std::to_string(rows) + " arm " + arms[arm].name + ": the default arms do not compare two routes");
                    std::cout << "RANK-ROUTE " << label << " T=" << rows << " arm " << arms[arm].name << ": gate/up "
                              << (gu_core ? "shared core" : "engine kernels") << ", down " << (dn_core ? "shared core" : "engine kernels")
                              << (a != 0U && gu_core == ref_gu_core && dn_core == ref_dn_core ? " (SAME ROUTE as arm " + arms[0].name + ")" : "") << '
';
                }
                hipLaunchKernelGGL(flush_kernel, dim3(1024), dim3(256), 0, nullptr, flush.data, flush.count, sink.data);
                enqueue_prefill(x, in, r2 && std::strcmp(r2, "1") == 0 ? view32 : view, run, false, ev);
                check(hipEventSynchronize(ev[2]), "sync");
                float g = 0, d = 0;
                check(hipEventElapsedTime(&g, ev[0], ev[1]), "elapsed");
                check(hipEventElapsedTime(&d, ev[1], ev[2]), "elapsed");
                if (rep > 0U) { gu[arm].push_back(g); dn[arm].push_back(d); }
                if (rep == 0U) {
                    auto codes = run.act_codes.download();
                    auto deltas = run.act_deltas.download();
                    auto sums = run.act_sums.download();
                    auto raw = run.raw.download();
                    if (a == 0U) { ref_codes = codes; ref_deltas = deltas; ref_sums = sums; ref_raw = raw; }
                    const auto bad = bit_mismatches(codes, ref_codes) + bit_mismatches(deltas, ref_deltas) + bit_mismatches(sums, ref_sums) +
                        bit_mismatches(raw, ref_raw);
                    expect(bad == 0U, label + " T=" + std::to_string(rows) + " arm " + arms[arm].name + ": outputs differ from arm " + arms[0].name);
                    const auto fnv = [](const auto& v) {
                        std::uint64_t h = 1469598103934665603ULL;
                        const auto* p = reinterpret_cast<const unsigned char*>(v.data());
                        for (std::size_t i = 0; i < v.size() * sizeof(v[0]); ++i) h = (h ^ p[i]) * 1099511628211ULL;
                        return h;
                    };
                    std::cout << "RANK-CHECK " << label << " T=" << rows << " arm " << arms[arm].name << " vs " << arms[0].name << ": "
                              << (bad == 0U ? "bit-identical" : "MISMATCH " + std::to_string(bad)) << " (A8 codes/deltas/sums, raw)" << std::hex
                              << " fnv codes " << fnv(codes) << " group sums " << fnv(sums) << " deltas " << fnv(deltas) << " raw " << fnv(raw)
                              << std::dec << '\n';
                }
            }
        const double ops_gu = 2.0 * static_cast<double>(entries) * 2.0 * e.inter * e.hidden;
        const double ops_dn = 2.0 * static_cast<double>(entries) * e.inter * e.hidden;
        for (std::size_t a = 0; a < arms.size(); ++a) {
            const double g = median_of(gu[a]), d = median_of(dn[a]);
            std::cout << std::fixed << std::setprecision(3) << "RANK " << label << " T=" << rows << " arm " << arms[a].name << " gate_up_ms=" << g
                      << " (" << std::setprecision(1) << 100.0 * ops_gu / (g * 1e-3) / peak << "% iu8 peak) down_ms=" << std::setprecision(3) << d
                      << " (" << std::setprecision(1) << 100.0 * ops_dn / (d * 1e-3) / peak << "%) total_ms=" << std::setprecision(3) << g + d
                      << " gu_all=";
            for (const auto v : gu[a]) std::cout << v << ',';
            std::cout << " dn_all=";
            for (const auto v : dn[a]) std::cout << v << ',';
            std::cout << std::defaultfloat << '\n';
        }
    }
    for (auto& v : ev) (void)hipEventDestroy(v);
}

} // namespace

int main() {
    try {
        const char* only_env = std::getenv("STRIX_DS41_BENCH_ONLY");
        const bool bench_only = only_env && *only_env == '1';
        probe_permlane();
        if (!bench_only) {
            test_synthetic();
            check_router(synthetic_router(64, 1024, 6, 5), 77, 6, "synthetic");
            // experts not a multiple of the 32-expert tile, a 32-element last chunk, one row
            check_router(synthetic_router(40, 1056, 6, 8), 33, 9, "synthetic tails");
            check_router(synthetic_router(40, 1056, 6, 8), 1, 10, "synthetic tails");
        }
        const char* q2 = std::getenv("STRIX_DS41_Q2_GGUF");
        const char* dspark = std::getenv("STRIX_DS41_DSPARK_GGUF");
        const char* bench_env = std::getenv("STRIX_DS41_BENCH");
        const bool timing = bench_env && *bench_env == '1';
        if (q2 && *q2) {
            {
                const auto e = file_experts(q2, "blk.20.ffn_gate_exps.weight", "blk.20.ffn_up_exps.weight", "blk.20.ffn_down_exps.weight");
                const DeviceExperts x(e);
                if (!bench_only) {
                    check_case(x, 512, 6, 192, 16, 0.0, 41, "real blk.20");
                    check_case(x, 512, 6, 192, 32, 0.0, 42, "real blk.20");
                    check_case(x, 300, 6, 192, 32, 0.6, 43, "real blk.20 skewed");
                }
                if (!bench_only) {
                    check_chunked(x, 600, 6, 192, 46, "real blk.20");
                    RouterWeights rw;
                    rw.experts = 384; rw.hidden = 5120; rw.top_k = 6;
                    rw.weight = read_f32(q2, "blk.20.ffn_gate_inp.weight", static_cast<std::size_t>(rw.experts) * rw.hidden);
                    rw.bias = read_f32(q2, "blk.20.exp_probs_b.bias", rw.experts);
                    rw.bias_image = read_f32(q2, "blk.20.exp_probs_b_vl.bias", rw.experts);
                    check_router(rw, 300, 47, "real blk.20");
                    check_router(rw, 8192, 48, "real blk.20");
                    if (timing) bench_router(rw, "blk.20");
                }
                const char* moe_env = std::getenv("STRIX_DS41_BENCH_MOE");
                const char* rank_env = std::getenv("STRIX_DS41_BENCH_RANK");
                if (rank_env && *rank_env == '1') bench_rank(x, 6, x.e.experts / 2U, "blk.20");
                else if (timing && moe_env && *moe_env == '1') bench_moe(x, 6, 192, "blk.20");
                else if (timing) { bench(x, 6, 192, "blk.20"); if (!bench_only) bench_decode(x, 6, 192, 2048, "blk.20"); }
            }
            // STRIX_DS41_BENCH_SHARED=1 (with STRIX_DS41_BENCH_RANK=1): the rank bench on the Q8_0 shared expert (one expert, all T rows)
            if (const char* sh = std::getenv("STRIX_DS41_BENCH_SHARED"); sh && *sh == '1') {
                const char* rank_env = std::getenv("STRIX_DS41_BENCH_RANK");
                if (rank_env && *rank_env == '1') {
                    const auto e = file_experts(q2, "blk.20.ffn_gate_shexp.weight", "blk.20.ffn_up_shexp.weight", "blk.20.ffn_down_shexp.weight");
                    const DeviceExperts x(e);
                    bench_rank(x, 1, 1, "blk.20-shared");
                }
            }
            if (!bench_only) {
                const auto e = file_experts(q2, "blk.20.ffn_gate_shexp.weight", "blk.20.ffn_up_shexp.weight", "blk.20.ffn_down_shexp.weight");
                const DeviceExperts x(e);
                check_case(x, 200, 1, 1, 32, 0.0, 44, "real blk.20 shared expert");
                check_case(x, 200, 1, 1, 16, 0.0, 45, "real blk.20 shared expert");
                if (timing) bench(x, 1, 1, "blk.20 shared");
            }
        }
        if (dspark && *dspark && !bench_only) {
            const auto e = file_experts(dspark, "mtp.0.ffn_gate_exps.weight", "mtp.0.ffn_up_exps.weight", "mtp.0.ffn_down_exps.weight");
            const DeviceExperts x(e);
            check_case(x, 256, 3, 64, 32, 0.0, 51, "real mtp.0");
            check_case(x, 128, 3, 64, 16, 0.3, 52, "real mtp.0");
            if (timing) bench(x, 3, 64, "mtp.0");
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "all deepseek41 prefill expert checks passed\n";
    return 0;
}
