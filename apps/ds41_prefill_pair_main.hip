// Two-rank prefill of one deepseek41 MoE layer (expert parallel by halves). Each rank keeps the router,
// the shared expert and its half of the routed experts (rank 0: ids below split, rank 1: the rest),
// runs the T rows in chunks, and exchanges each chunk's half sums with the peer while the next chunk
// is computed; both ranks then form y = (shared + low) + high, the single-node combine bit for bit.
//
//   strix-ds41-prefill-pair --rank R --gguf FILE [--layer 20] [--rows T] [--chunk C]
//       [--transport usb4|tcp|memory] [--usb4-devices a,b] [--peer HOST] [--port P] [--reference] [--repeat N]
//       [--connect-timeout S]   (seconds to open the transport; past it the process exits with status 3)
//
// --transport memory runs both ranks in one process (two threads); --reference also computes the
// single-node prefill on this rank (all experts) and compares y bitwise. Inputs are synthetic and
// identical on both ranks (fixed seeds); the output is a JSON line per rank.

#include "strix/device/hip_gate.hpp"
#include "strix/hip_check.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/ds41_formats.hpp"
#include "strix/ds41_moe.hpp"
#include "strix/ds41_prefill.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_prefill.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using strix::ds41::WeightFormat;
using Clock = std::chrono::steady_clock;

using strix::hip_check;
template <class T>
struct Device {
    T* data = nullptr;
    std::size_t count = 0;
    explicit Device(std::size_t n) : count(n) {
        hip_check(hipMalloc(reinterpret_cast<void**>(&data), std::max<std::size_t>(n, 1U) * sizeof(T)), "hipMalloc");
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() { (void)hipFree(data); }
    void upload(const T* host, std::size_t n) { hip_check(hipMemcpy(data, host, n * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    void upload(const std::vector<T>& host) { upload(host.data(), host.size()); }
    std::vector<T> download() const {
        std::vector<T> out(count);
        hip_check(hipMemcpy(out.data(), data, count * sizeof(T), hipMemcpyDeviceToHost), "download");
        return out;
    }
};

struct Options {
    std::uint32_t rank = 0, layer = 20, rows = 2048, chunk = 512, repeat = 3, port = 47321, connect_timeout = 60;
    std::string gguf, transport = "usb4", peer, layout = "/run/usb4stream-layout.json";
    std::vector<std::string> devices;
    bool reference = false;
};
Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--rank") o.rank = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--layer") o.layer = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--rows") o.rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--chunk") o.chunk = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--repeat") o.repeat = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--port") o.port = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--connect-timeout") o.connect_timeout = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--gguf") o.gguf = next();
        else if (a == "--transport") o.transport = next();
        else if (a == "--peer") o.peer = next();
        else if (a == "--layout") o.layout = next();
        else if (a == "--usb4-devices") {
            std::stringstream list(next());
            for (std::string d; std::getline(list, d, ',');) o.devices.push_back(d);
        } else if (a == "--reference") o.reference = true;
        else throw std::invalid_argument("unknown option " + a);
    }
    if (o.gguf.empty() || o.rank > 1U || o.rows == 0U || o.chunk == 0U) throw std::invalid_argument("need --gguf, --rank 0|1, rows and chunk > 0");
    return o;
}

// Streams of the layout file whose service has an attached peer (its sysfs entry exists), in key
// order. A stand-in for the communicator's automatic pairing (handshake per stream) until it lands;
// both nodes list the same keys today. Never a fixed tbstream number.
std::vector<std::string> layout_devices(const std::string& layout) {
    std::ifstream in(layout);
    if (!in) throw std::runtime_error("cannot read " + layout);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::map<std::string, std::string> found;
    for (std::size_t at = 0; (at = text.find("\"device\"", at)) != std::string::npos; ++at) {
        const auto key_end = text.rfind("\": {", at), key_begin = text.rfind('"', key_end - 1U);
        const auto v0 = text.find('"', text.find(':', at) + 1U), v1 = text.find('"', v0 + 1U);
        const std::string key = text.substr(key_begin + 1U, key_end - key_begin - 1U), device = text.substr(v0 + 1U, v1 - v0 - 1U);
        const std::string service = key.substr(0, key.find('/'));
        if (std::filesystem::exists("/sys/bus/thunderbolt/devices/" + service)) found[key] = device;
    }
    std::vector<std::string> out;
    for (const auto& [key, device] : found) out.push_back(device);
    if (out.empty()) throw std::runtime_error("no USB4 stream with an attached peer in " + layout);
    return out;
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
WeightFormat format_of(std::uint32_t t) {
    if (t == 16U) return WeightFormat::iq2_xxs;
    if (t == 10U) return WeightFormat::q2_k;
    if (t == 8U) return WeightFormat::q8_0;
    if (t == 39U) return WeightFormat::mxfp4;
    throw std::runtime_error("unexpected tensor type");
}

struct Tensor { const strix::gguf::TensorInfo* info = nullptr; WeightFormat format{}; std::uint64_t row_bytes = 0, stride = 0; std::uint32_t width = 0, outputs = 0, experts = 1; };
struct Layer {
    std::map<std::string, const strix::gguf::TensorInfo*> tensors;
    strix::gguf::FileInfo file;
    std::string path, prefix;
    Tensor tensor(const std::string& name) const {
        const auto it = tensors.find(prefix + name);
        if (it == tensors.end()) throw std::runtime_error("missing " + prefix + name);
        Tensor t;
        t.info = it->second;
        if (t.info->ggml_type != 0U) {
            t.format = format_of(t.info->ggml_type);
            t.width = static_cast<std::uint32_t>(t.info->dimensions.at(0));
            t.outputs = static_cast<std::uint32_t>(t.info->dimensions.at(1));
            t.experts = t.info->dimensions.size() > 2U ? static_cast<std::uint32_t>(t.info->dimensions[2]) : 1U;
            t.row_bytes = t.width / strix::ds41::block_elements(t.format) * strix::ds41::block_bytes(t.format);
            t.stride = t.row_bytes * t.outputs;
        }
        return t;
    }
    // Experts [first, end) of an expert tensor, in device memory.
    std::unique_ptr<Device<std::uint8_t>> experts(const Tensor& t, std::uint32_t first, std::uint32_t end) const {
        const auto bytes = read_range(path, t.info->absolute_offset + first * t.stride, (end - first) * t.stride);
        auto d = std::make_unique<Device<std::uint8_t>>(bytes.size());
        d->upload(bytes);
        return d;
    }
    std::vector<float> floats(const std::string& name) const {
        const auto t = tensor(name);
        const auto bytes = read_range(path, t.info->absolute_offset, t.info->byte_size);
        std::vector<float> out(bytes.size() / 4U);
        std::memcpy(out.data(), bytes.data(), bytes.size());
        return out;
    }
};

std::uint64_t fnv(const std::vector<float>& v) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const float x : v) { h ^= std::bit_cast<std::uint32_t>(x); h *= 1099511628211ULL; }
    return h;
}

// One rank. `transport` is already connected (opened before the weights are loaded).
std::string run_rank(const Options& o, std::unique_ptr<strix::dist::Transport> transport) {
    using namespace strix::ds41;
    Layer layer;
    layer.path = o.gguf;
    layer.file = strix::gguf::read_file(o.gguf);
    for (const auto& t : layer.file.tensors) layer.tensors[t.name] = &t;
    layer.prefix = "blk." + std::to_string(o.layer) + ".";
    const auto gate = layer.tensor("ffn_gate_exps.weight"), up = layer.tensor("ffn_up_exps.weight"), down = layer.tensor("ffn_down_exps.weight");
    const auto sgate = layer.tensor("ffn_gate_shexp.weight"), sup = layer.tensor("ffn_up_shexp.weight"), sdown = layer.tensor("ffn_down_shexp.weight");
    const std::uint32_t experts = gate.experts, hidden = gate.width, inter = gate.outputs, top_k = 6U, split = experts / 2U;
    const std::uint32_t first = o.rank == 0U ? 0U : split, end = o.rank == 0U ? split : experts;
    const auto router_w = layer.floats("ffn_gate_inp.weight"), bias = layer.floats("exp_probs_b.bias");
    auto g = layer.experts(gate, first, end), u = layer.experts(up, first, end), d = layer.experts(down, first, end);
    auto sg = layer.experts(sgate, 0, 1), su = layer.experts(sup, 0, 1), sd = layer.experts(sdown, 0, 1);

    // Synthetic MoE input rows, identical on both ranks.
    const std::uint32_t T = o.rows;
    std::vector<float> x(static_cast<std::size_t>(T) * hidden);
    {
        std::mt19937 random(1234U);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        for (auto& v : x) v = normal(random);
    }
    Device<float> dx(x.size()), drouter(router_w.size()), dbias(bias.size());
    dx.upload(x); drouter.upload(router_w); dbias.upload(bias);
    Device<std::int8_t> codes(x.size());
    Device<float> deltas(x.size() / 32U);
    Device<std::int32_t> sums(x.size() / 32U);
    Device<float> scores(static_cast<std::size_t>(T) * experts), weights(static_cast<std::size_t>(T) * top_k);
    Device<std::uint32_t> ids(static_cast<std::size_t>(T) * top_k);
    const std::size_t n = static_cast<std::size_t>(T) * hidden;
    Device<float> shared(n), mine(n), theirs(n), y(n);

    hipStream_t compute = nullptr, comm = nullptr;
    hip_check(hipStreamCreateWithFlags(&compute, hipStreamNonBlocking), "stream");
    hip_check(hipStreamCreateWithFlags(&comm, hipStreamNonBlocking), "stream");

    // Replicated: A8 input, router, shared expert (one expert, every row).
    enqueue_quantize_a8(dx.data, hidden, T, codes.data, deltas.data, sums.data, compute);
    RouterView rv;
    rv.x = dx.data; rv.weight = drouter.data; rv.bias = dbias.data; rv.rows = T; rv.hidden = hidden; rv.experts = experts; rv.top_k = top_k;
    rv.route_scale = 1.5F; rv.scores = scores.data; rv.ids = ids.data; rv.weights = weights.data;
    enqueue_prefill_router(rv, compute);
    {
        // The shared expert as a one-expert MoE half (top_k 1, weight 1 is not applied: half sums would
        // weight it), so run gate/up/down directly: its raw vectors are the shared output.
        Device<std::uint32_t> zero(T);
        hip_check(hipMemsetAsync(zero.data, 0, T * sizeof(std::uint32_t), compute), "zero ids");
        PrefillRoutesBuffers rb;
        Device<std::uint32_t> c(1), b(2), er(T), es(T), be(prefill_max_batches(T, 1, 1, 16)), bb(be.count + 1U), bc(be.count), bt(1), od(T), lc(T);
        rb = {c.data, b.data, er.data, es.data, be.data, bb.data, bc.data, bt.data, od.data, lc.data};
        enqueue_build_prefill_routes(zero.data, T, 1, 1, 1, 16, rb, compute);
        const auto view = prefill_routes_view(rb, T, 1, 1, 16);
        Device<std::int8_t> ac(static_cast<std::size_t>(T) * inter);
        Device<float> ad(static_cast<std::size_t>(T) * inter / 32U);
        Device<std::int32_t> as(static_cast<std::size_t>(T) * inter / 32U);
        PrefillGateUpView gu;
        gu.gate = sg->data; gu.up = su->data; gu.expert_stride_bytes = sgate.stride; gu.row_bytes = sgate.row_bytes; gu.hidden = hidden;
        gu.inter = sgate.outputs; gu.routes = view; gu.codes = codes.data; gu.deltas = deltas.data; gu.act_codes = ac.data;
        gu.act_deltas = ad.data; gu.act_sums = as.data;
        enqueue_prefill_gate_up(sgate.format, gu, compute);
        PrefillDownView dn;
        dn.weights = sd->data; dn.expert_stride_bytes = sdown.stride; dn.row_bytes = sdown.row_bytes; dn.width = sdown.width;
        dn.outputs = hidden; dn.routes = view; dn.codes = ac.data; dn.deltas = ad.data; dn.out = shared.data;
        enqueue_prefill_down(sdown.format, dn, compute);
        hip_check(hipStreamSynchronize(compute), "shared expert");
    }

    PrefillMoeWeights mw;
    mw.gate_up_format = gate.format; mw.down_format = down.format; mw.gate = g->data; mw.up = u->data; mw.down = d->data;
    mw.gate_row_bytes = gate.row_bytes; mw.down_row_bytes = down.row_bytes; mw.hidden = hidden; mw.inter = inter;
    PrefillMoe moe(mw, top_k, experts, split, o.chunk, first, end);
    PrefillMoeInputs in;
    in.rows = T; in.ids = ids.data; in.weights = weights.data; in.codes = codes.data; in.deltas = deltas.data; in.shared = shared.data;
    const std::uint32_t chunks = moe.chunks(T);
    const std::size_t chunk_bytes = static_cast<std::size_t>(o.chunk) * hidden * 4U;
    strix::dist::Communicator comm_(strix::dist::Group{o.rank, 2U, 1U, 41U}, std::move(transport), chunk_bytes);
    std::vector<hipEvent_t> done(chunks);
    for (auto& e : done) hip_check(hipEventCreateWithFlags(&e, hipEventDisableTiming), "event");

    float* low = o.rank == 0U ? mine.data : theirs.data;
    float* high = o.rank == 0U ? theirs.data : mine.data;
    std::vector<double> compute_only, overlapped;
    double wire = 0.0;
    for (std::uint32_t rep = 0; rep < o.repeat; ++rep) {
        // (a) the rank's half alone (no exchange).
        auto t0 = Clock::now();
        for (std::uint32_t c = 0; c < chunks; ++c) moe.enqueue_half(in, c, o.rank, mine.data, compute);
        hip_check(hipStreamSynchronize(compute), "half");
        compute_only.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        // (b) the pipeline: chunk c's exchange overlaps chunk c + 1's experts.
        const double wire0 = comm_.stats().wire;
        t0 = Clock::now();
        std::thread mover([&] {
            for (std::uint32_t c = 0; c < chunks; ++c) {
                hip_check(hipEventSynchronize(done[c]), "chunk done");
                const std::size_t row0 = static_cast<std::size_t>(c) * o.chunk, rows = std::min<std::size_t>(o.chunk, T - row0);
                const std::size_t at = row0 * hidden;
                comm_.exchange(comm, mine.data + at, theirs.data + at, rows * hidden * 4U,
                               strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::half_sum), o.layer, rep * 1000U + c});
                enqueue_canonical_sum(shared.data + at, low + at, high + at, rows * hidden, y.data + at, comm);
            }
            hip_check(hipStreamSynchronize(comm), "comm");
        });
        for (std::uint32_t c = 0; c < chunks; ++c) {
            moe.enqueue_half(in, c, o.rank, mine.data, compute);
            hip_check(hipEventRecord(done[c], compute), "record");
        }
        mover.join();
        overlapped.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        wire += comm_.stats().wire - wire0;
    }
    const auto y_host = y.download();
    std::ostringstream js;
    js << "{\"rank\":" << o.rank << ",\"transport\":\"" << comm_.transport_name() << "\",\"layer\":" << o.layer << ",\"rows\":" << T
       << ",\"chunk\":" << o.chunk << ",\"experts\":[" << first << ',' << end << "],\"y_hash\":\"" << std::hex << fnv(y_host) << std::dec << '"';
    const auto list = [&](const char* name, const std::vector<double>& v) {
        js << ",\"" << name << "\":[";
        for (std::size_t i = 0; i < v.size(); ++i) js << (i ? "," : "") << v[i];
        js << ']';
    };
    list("half_ms", compute_only);
    list("pipeline_ms", overlapped);
    js << ",\"wire_s_total\":" << wire << ",\"exchanged_MB_per_rep\":" << static_cast<double>(n) * 4.0 / 1e6;
    if (o.reference) {
        auto fg = layer.experts(gate, 0, experts), fu = layer.experts(up, 0, experts), fd = layer.experts(down, 0, experts);
        PrefillMoeWeights fw = mw;
        fw.gate = fg->data; fw.up = fu->data; fw.down = fd->data;
        PrefillMoe full(fw, top_k, experts, split, o.chunk);
        Device<float> yref(n);
        for (std::uint32_t c = 0; c < full.chunks(T); ++c) full.enqueue_combine(in, c, yref.data, compute);
        hip_check(hipStreamSynchronize(compute), "reference");
        const auto r = yref.download();
        std::size_t bad = 0;
        for (std::size_t i = 0; i < n; ++i) bad += std::bit_cast<std::uint32_t>(r[i]) != std::bit_cast<std::uint32_t>(y_host[i]);
        js << ",\"reference_mismatches\":" << bad << ",\"reference_hash\":\"" << std::hex << fnv(r) << std::dec << '"';
    }
    js << '}';
    for (auto& e : done) (void)hipEventDestroy(e);
    (void)hipStreamDestroy(compute); (void)hipStreamDestroy(comm);
    return js.str();
}

} // namespace

int main(int argc, char** argv) {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    try {
        auto o = parse(argc, argv);
        if (o.transport == "memory") {
            auto [a, b] = strix::dist::memory_pair();
            std::string out[2];
            auto o1 = o;
            o1.rank = 1U;
            std::thread second([&] { out[1] = run_rank(o1, std::move(b)); });
            o.rank = 0U;
            out[0] = run_rank(o, std::move(a));
            second.join();
            std::cout << out[0] << '\n' << out[1] << '\n';
            return 0;
        }
        std::unique_ptr<strix::dist::Transport> t;
        // Opening a stream whose peer is gone can block in the driver: a watchdog ends the process.
        std::atomic<bool> connected{false};
        std::thread watchdog([&connected, seconds = o.connect_timeout] {
            for (std::uint32_t i = 0; i < seconds * 10U && !connected.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (!connected.load()) {
                std::cerr << "error: transport not connected after " << seconds << " s\n";
                std::_Exit(3);
            }
        });
        watchdog.detach();
        if (o.transport == "tcp") {
            t = o.rank == 0U ? strix::dist::tcp_listen(static_cast<std::uint16_t>(o.port), std::chrono::milliseconds(120000))
                             : strix::dist::tcp_connect(o.peer, static_cast<std::uint16_t>(o.port), std::chrono::milliseconds(120000));
        } else if (o.transport == "usb4") {
            const auto devices = o.devices.empty() ? layout_devices(o.layout) : o.devices;
            std::cerr << "rank " << o.rank << " usb4 streams:";
            for (const auto& d : devices) std::cerr << ' ' << d;
            std::cerr << '\n';
            t = strix::dist::usb4_host(devices);
        } else throw std::invalid_argument("transport must be usb4, tcp or memory");
        connected = true;
        std::cout << run_rank(o, std::move(t)) << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
    return 0;
}
