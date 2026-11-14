#include "strix/file_snapshot.hpp"
#include "strix/hip_ds41_expert_backend.hpp"
#include "strix/hip_check.hpp"
#include "strix/device/hip_gate.hpp"

#include "strix/ds41_formats.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/hip_ds41_formats.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <atomic>
#include <thread>

#include <fcntl.h>
#include <unistd.h>

namespace strix::ds41::xs {

namespace {

using Clock = std::chrono::steady_clock;
namespace x = xrpc;

void check(hipError_t e, const char* what) {
    ::strix::hip_check_prefixed("hip expert backend: ", e, what);
}

template <typename T>
struct Dev {
    T* p = nullptr;
    std::size_t n = 0;
    Dev() = default;
    explicit Dev(std::size_t count) : n(count) { check(hipMalloc(reinterpret_cast<void**>(&p), std::max<std::size_t>(count, 1) * sizeof(T)), "hipMalloc"); }
    ~Dev() {
        if (p != nullptr) (void)hipFree(p);
    }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    Dev(Dev&& o) noexcept : p(std::exchange(o.p, nullptr)), n(o.n) {}
    Dev& operator=(Dev&& o) noexcept {
        std::swap(p, o.p);
        std::swap(n, o.n);
        return *this;
    }
};

template <typename T>
struct Pinned {
    T* p = nullptr;
    std::size_t n = 0;
    Pinned() = default;
    explicit Pinned(std::size_t count) : n(count) {
        check(hipHostMalloc(reinterpret_cast<void**>(&p), std::max<std::size_t>(count, 1) * sizeof(T), hipHostMallocDefault), "hipHostMalloc");
    }
    ~Pinned() {
        if (p != nullptr) (void)hipHostFree(p);
    }
    Pinned(const Pinned&) = delete;
    Pinned& operator=(const Pinned&) = delete;
    Pinned(Pinned&& o) noexcept : p(std::exchange(o.p, nullptr)), n(o.n) {}
    Pinned& operator=(Pinned&& o) noexcept {
        std::swap(p, o.p);
        std::swap(n, o.n);
        return *this;
    }
};

// pread (O_DIRECT when the file allows it) through a pinned chunk into device memory
class Loader {
public:
    static constexpr std::uint64_t align = 4096, chunk = 256ULL << 20U;
    explicit Loader(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd_ < 0) fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("hip expert backend: cannot open " + path);
        strix::LoaderCapture::record(path, fd_);
        buffer_ = Pinned<std::uint8_t>(chunk + 2U * align);
    }
    ~Loader() {
        if (fd_ >= 0) ::close(fd_);
    }
    void load(std::uint64_t offset, std::uint64_t bytes, std::uint8_t* device) {
        std::uint64_t done = 0;
        while (done < bytes) {
            const std::uint64_t at = offset + done, base = at & ~(align - 1U), lead = at - base;
            const std::uint64_t take = std::min(bytes - done, chunk - lead);
            const std::uint64_t want = (lead + take + align - 1U) & ~(align - 1U);
            std::uint64_t got = 0;
            while (got < lead + take) {
                const auto n = ::pread(fd_, buffer_.p + got, want - got, static_cast<off_t>(base + got));
                if (n <= 0) throw std::runtime_error("hip expert backend: short read of the weights");
                got += static_cast<std::uint64_t>(n);
            }
            check(hipMemcpy(device + done, buffer_.p + lead, take, hipMemcpyHostToDevice), "weight upload");
            done += take;
        }
    }

private:
    int fd_ = -1;
    Pinned<std::uint8_t> buffer_;
};

// Answer items: RAW copies an entry's vector; FOLD = 0 + w0 * v0 + w1 * v1 ... in the listed (ascending id) order, one
// FP32 product then one FP32 add per step (strict FP flags: no contraction), as the combine's half sums.
struct ItemDesc {
    std::uint32_t kind = 0, count = 0, offset = 0, pad = 0;
};
__global__ __launch_bounds__(256) void pack_items_kernel(const float* __restrict__ raw, const ItemDesc* __restrict__ items,
                                                         const std::uint32_t* __restrict__ entries, const float* __restrict__ weights,
                                                         unsigned hidden, float* __restrict__ out) {
    const unsigned o = blockIdx.x * 256U + threadIdx.x, item = blockIdx.y;
    if (o >= hidden) return;
    const ItemDesc d = items[item];
    const unsigned long long at = static_cast<unsigned long long>(item) * hidden + o;
    if (d.kind == static_cast<std::uint32_t>(x::PlanItem::Kind::raw)) {
        out[at] = raw[static_cast<unsigned long long>(entries[d.offset]) * hidden + o];
        return;
    }
    float acc = 0.0F;
    for (unsigned j = 0; j < d.count; ++j) {
        const float term = weights[d.offset + j] * raw[static_cast<unsigned long long>(entries[d.offset + j]) * hidden + o];
        acc = acc + term;
    }
    out[at] = acc;
}

// keep-warm: a trivial kernel that touches a little memory, so the GPU does not drop to its idle state between
// sparse requests
__global__ void keep_warm_kernel(float* p) {
    p[threadIdx.x] = p[threadIdx.x] * 1.0F + 1.0F;
}

std::size_t align16(std::size_t n) { return (n + 15U) & ~std::size_t{15U}; }

class HipBackend final : public x::ExpertBackend {
public:
    HipBackend(const std::string& path, std::uint64_t mask, std::uint32_t first, std::uint32_t end, x::Geometry* geometry_out, std::uint32_t max_rows, bool spin)
        : g_(x::read_geometry(path)), max_rows_(max_rows) {
        // The shared host wait policy (strix/device/hip_gate.hpp): busy_poll selects spin, STRIX_HOST_WAIT overrides.
        (void)strix::device::set_host_wait(spin ? strix::device::HostWait::spin : strix::device::HostWait::runtime, spin ? stderr : nullptr);
        const auto file = strix::gguf::read_file(path);
        if (const auto* m = file.find_metadata("deepseek41.swiglu_limit"); m != nullptr && m->floating_value) limit_ = static_cast<float>(*m->floating_value);
        first_ = first;
        end_ = end == 0U ? g_.experts : end;
        if (first_ >= end_ || end_ > g_.experts) throw std::invalid_argument("hip expert backend: bad expert range");
        if (g_.hidden % 256U != 0U || g_.inter % 256U != 0U) throw std::runtime_error("hip expert backend: hidden/inter not multiples of 256");
        grow_ = g_.hidden / 256U * 66U;  // IQ2_XXS row
        drow_ = g_.inter / 256U * 84U;   // Q2_K row
        const std::uint64_t gstride = grow_ * g_.inter, dstride = drow_ * g_.hidden, count = end_ - first_;
        std::map<std::uint32_t, std::array<const strix::gguf::TensorInfo*, 3>> found;
        for (const auto& t : file.tensors) {
            if (!t.name.starts_with("blk.") || !t.name.ends_with("_exps.weight")) continue;
            const auto dot = t.name.find('.', 4);
            const auto layer = static_cast<std::uint32_t>(std::stoul(t.name.substr(4, dot - 4)));
            if (layer >= 64U || ((mask >> layer) & 1U) == 0U) continue;
            const int k = t.name.ends_with("ffn_gate_exps.weight") ? 0 : t.name.ends_with("ffn_up_exps.weight") ? 1 : t.name.ends_with("ffn_down_exps.weight") ? 2 : -1;
            if (k < 0) continue;
            if ((k < 2 && t.ggml_type != 16U) || (k == 2 && t.ggml_type != 10U)) throw std::runtime_error("hip expert backend: unexpected expert tensor type in " + t.name);
            found[layer][static_cast<std::size_t>(k)] = &t;
        }
        Loader loader(path);
        for (const auto& [layer, ts] : found) {
            if (!ts[0] || !ts[1] || !ts[2]) continue;
            Layer L;
            L.gate = Dev<std::uint8_t>(gstride * count);
            L.up = Dev<std::uint8_t>(gstride * count);
            L.down = Dev<std::uint8_t>(dstride * count);
            loader.load(ts[0]->absolute_offset + gstride * first_, gstride * count, L.gate.p);
            loader.load(ts[1]->absolute_offset + gstride * first_, gstride * count, L.up.p);
            loader.load(ts[2]->absolute_offset + dstride * first_, dstride * count, L.down.p);
            bytes_ += (2U * gstride + dstride) * count;
            layers_.emplace(layer, std::move(L));
            mask_ |= 1ULL << layer;
        }
        if (layers_.empty()) throw std::runtime_error("hip expert backend: no listed layer has routed experts in " + path);
        check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "stream");
        cap_ = max_rows_ * g_.top_k;
        x_codes_ = Dev<std::int8_t>(std::size_t{max_rows_} * g_.hidden);
        x_deltas_ = Dev<float>(std::size_t{max_rows_} * g_.groups());
        act_codes_ = Dev<std::int8_t>(std::size_t{cap_} * g_.inter);
        act_deltas_ = Dev<float>(std::size_t{cap_} * g_.inter / 32U);
        act_sums_ = Dev<std::int32_t>(std::size_t{cap_} * g_.inter / 32U);
        raw_ = Dev<float>(std::size_t{cap_} * g_.hidden);
        out_ = Dev<float>(std::size_t{cap_} * g_.hidden);
        staging_bytes_ = align16(std::size_t{max_rows_} * g_.hidden) + align16(std::size_t{max_rows_} * g_.groups() * 4U) + 4U * align16(4U * (cap_ + 1U)) +
                         align16(sizeof(ItemDesc) * cap_) + 2U * align16(4U * cap_);
        staging_ = Pinned<std::uint8_t>(staging_bytes_);
        staging_dev_ = Dev<std::uint8_t>(staging_bytes_);
        answer_ = Pinned<float>(std::size_t{cap_} * g_.hidden);
        hipDeviceProp_t prop{};
        int dev = 0;
        check(hipGetDevice(&dev), "device");
        check(hipGetDeviceProperties(&prop, dev), "device properties");
        engine_ = std::string("hip-") + prop.gcnArchName;
        if (geometry_out) *geometry_out = g_;
    }
    // Every `period` without a request, enqueue the keep-warm kernel on its own stream (0: off).
    void keep_warm(std::chrono::microseconds period) {
        if (period.count() <= 0) return;
        check(hipStreamCreateWithFlags(&warm_stream_, hipStreamNonBlocking), "warm stream");
        warm_buf_ = Dev<float>(64);
        warm_ = std::thread([this, period] {
            while (!stop_.load()) {
                std::this_thread::sleep_for(period);
                if (Clock::now() - last_.load() < period) continue;
                hipLaunchKernelGGL(keep_warm_kernel, dim3(1), dim3(64), 0, warm_stream_, warm_buf_.p);
                (void)hipStreamSynchronize(warm_stream_);
            }
        });
    }
    ~HipBackend() override {
        stop_ = true;
        if (warm_.joinable()) warm_.join();
        if (warm_stream_) (void)hipStreamDestroy(warm_stream_);
        if (stream_) (void)hipStreamDestroy(stream_);
    }
    [[nodiscard]] x::Residency residency() const override { return {first_, end_, mask_}; }
    [[nodiscard]] std::string engine() const override {
        char buf[96];
        std::snprintf(buf, sizeof(buf), " (%.2f GiB of experts)", static_cast<double>(bytes_) / 1073741824.0);
        return engine_ + buf;
    }
    [[nodiscard]] std::string numeric_identity() const override { return "ds41-exact-v1"; }
    float* answer_buffer(std::size_t floats) override {
        if (answer_.n < floats) answer_ = Pinned<float>(floats);
        return answer_.p;
    }

    void compute(const x::ExpertsRequest& req, const x::ResponsePlan& plan, std::size_t first, std::size_t count, float* out,
                 std::uint64_t& compute_ns) override {
        const auto t0 = Clock::now();
        if (count == 0U) {
            compute_ns = 0;
            return;
        }
        if (req.rows > max_rows_) throw std::invalid_argument("hip expert backend: more rows than max_rows");
        const auto it = layers_.find(req.layer);
        if (it == layers_.end()) throw std::invalid_argument("hip expert backend: layer not resident");
        const auto K = req.top_k, H = g_.hidden, I = g_.inter;
        // the slots this range of items needs, then the entries in ascending (expert, row)
        std::vector<std::int32_t> entry_of(std::size_t{req.rows} * K, -1);
        std::vector<std::uint32_t> needed;
        const auto need = [&](std::uint32_t slot) {
            if (entry_of[slot] == -1) {
                entry_of[slot] = -2;
                needed.push_back(slot);
            }
        };
        for (std::size_t i = first; i < first + count; ++i) {
            const auto& item = plan.items[i];
            if (item.kind == x::PlanItem::Kind::raw) need(item.slot);
            else
                for (std::uint32_t j = 0; j < item.count; ++j) need(plan.fold_slots[plan.fold_begin[i] + j]);
        }
        std::sort(needed.begin(), needed.end(), [&](std::uint32_t a, std::uint32_t b) {
            return std::pair(req.ids[a], a / K) < std::pair(req.ids[b], b / K);
        });
        for (std::size_t e = 0; e < needed.size(); ++e) {
            const auto id = req.ids[needed[e]];
            if (id < first_ || id >= end_) throw std::invalid_argument("hip expert backend: expert outside this rank's range");
            entry_of[needed[e]] = static_cast<std::int32_t>(e);
        }
        // CSR routes: each expert in chunks of at most product_max_rows entries
        std::vector<std::uint32_t> experts, begins{0}, inputs, outputs;
        std::uint32_t max_chunk = 1;
        for (std::size_t e = 0; e < needed.size();) {
            const auto id = req.ids[needed[e]];
            std::size_t run = e;
            while (run < needed.size() && req.ids[needed[run]] == id) ++run;
            for (std::size_t c = e; c < run; c += product_max_rows) {
                const auto stop = std::min<std::size_t>(run, c + product_max_rows);
                experts.push_back(id - first_);
                for (auto k = c; k < stop; ++k) {
                    inputs.push_back(needed[k] / K);
                    outputs.push_back(static_cast<std::uint32_t>(k));
                }
                begins.push_back(static_cast<std::uint32_t>(inputs.size()));
                max_chunk = std::max(max_chunk, static_cast<std::uint32_t>(stop - c));
            }
            e = run;
        }
        // items
        std::vector<ItemDesc> items(count);
        std::vector<std::uint32_t> list;
        std::vector<float> weights;
        for (std::size_t i = 0; i < count; ++i) {
            const auto& item = plan.items[first + i];
            items[i].kind = static_cast<std::uint32_t>(item.kind);
            items[i].offset = static_cast<std::uint32_t>(list.size());
            if (item.kind == x::PlanItem::Kind::raw) {
                items[i].count = 1;
                list.push_back(static_cast<std::uint32_t>(entry_of[item.slot]));
                weights.push_back(0.0F);
            } else {
                items[i].count = item.count;
                for (std::uint32_t j = 0; j < item.count; ++j) {
                    const auto slot = plan.fold_slots[plan.fold_begin[first + i] + j];
                    list.push_back(static_cast<std::uint32_t>(entry_of[slot]));
                    weights.push_back(req.weights[slot]);
                }
            }
        }
        // one upload of everything the kernels read
        std::size_t at = 0;
        const auto put = [&](const void* src, std::size_t bytes) {
            const auto here = at;
            if (here + bytes > staging_bytes_) throw std::logic_error("hip expert backend: staging overflow");
            if (bytes) std::memcpy(staging_.p + here, src, bytes);
            at = align16(here + bytes);
            return here;
        };
        const auto o_codes = put(req.codes.data(), std::size_t{req.rows} * H);
        const auto o_deltas = put(req.deltas.data(), std::size_t{req.rows} * g_.groups() * 4U);
        const auto o_experts = put(experts.data(), experts.size() * 4U);
        const auto o_begins = put(begins.data(), begins.size() * 4U);
        const auto o_inputs = put(inputs.data(), inputs.size() * 4U);
        const auto o_outputs = put(outputs.data(), outputs.size() * 4U);
        const auto o_items = put(items.data(), items.size() * sizeof(ItemDesc));
        const auto o_list = put(list.data(), list.size() * 4U);
        const auto o_weights = put(weights.data(), weights.size() * 4U);
        check(hipMemcpyAsync(staging_dev_.p, staging_.p, at, hipMemcpyHostToDevice, stream_), "request upload");
        const auto dev = [&](std::size_t off) { return staging_dev_.p + off; };
        const auto* d_codes = reinterpret_cast<const std::int8_t*>(dev(o_codes));
        const auto* d_deltas = reinterpret_cast<const float*>(dev(o_deltas));
        const auto* d_experts = reinterpret_cast<const std::uint32_t*>(dev(o_experts));
        const auto* d_begins = reinterpret_cast<const std::uint32_t*>(dev(o_begins));
        const auto* d_inputs = reinterpret_cast<const std::uint32_t*>(dev(o_inputs));
        const auto* d_outputs = reinterpret_cast<const std::uint32_t*>(dev(o_outputs));
        const auto& L = it->second;
        const auto n_chunks = static_cast<std::uint32_t>(experts.size());
        const RoutesView gate_routes{d_experts, d_begins, d_inputs, d_outputs, n_chunks, max_chunk, nullptr};
        const RoutesView down_routes{d_experts, d_begins, d_outputs, d_outputs, n_chunks, max_chunk, nullptr};
        GateUpView gu;
        gu.gate = L.gate.p;
        gu.up = L.up.p;
        gu.expert_stride_bytes = grow_ * I;
        gu.row_bytes = grow_;
        gu.hidden = H;
        gu.inter = I;
        gu.routes = gate_routes;
        gu.codes = d_codes;
        gu.deltas = d_deltas;
        gu.limit = limit_;
        gu.act_codes = act_codes_.p;
        gu.act_deltas = act_deltas_.p;
        gu.act_sums = act_sums_.p;
        enqueue_gate_up(WeightFormat::iq2_xxs, gu, stream_);
        RoutesProductView dv;
        dv.weights = L.down.p;
        dv.expert_stride_bytes = drow_ * H;
        dv.row_bytes = drow_;
        dv.width = I;
        dv.outputs = H;
        dv.routes = down_routes;
        dv.codes = act_codes_.p;
        dv.deltas = act_deltas_.p;
        dv.out = raw_.p;
        enqueue_routes_product(WeightFormat::q2_k, dv, stream_);
        hipLaunchKernelGGL(pack_items_kernel, dim3((H + 255U) / 256U, static_cast<unsigned>(count)), dim3(256), 0, stream_, raw_.p,
                           reinterpret_cast<const ItemDesc*>(dev(o_items)), reinterpret_cast<const std::uint32_t*>(dev(o_list)),
                           reinterpret_cast<const float*>(dev(o_weights)), H, out_.p);
        check(hipGetLastError(), "pack items");
        check(hipMemcpyAsync(out, out_.p, count * std::size_t{H} * 4U, hipMemcpyDeviceToHost, stream_), "answer download");
        check(hipStreamSynchronize(stream_), "expert window");
        compute_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
        last_ = Clock::now();
    }

private:
    struct Layer {
        Dev<std::uint8_t> gate, up, down;
    };
    x::Geometry g_;
    std::uint32_t max_rows_ = 64, cap_ = 0, first_ = 0, end_ = 0;
    std::uint64_t grow_ = 0, drow_ = 0, mask_ = 0, bytes_ = 0;
    float limit_ = 10.0F;
    std::map<std::uint32_t, Layer> layers_;
    hipStream_t stream_ = nullptr, warm_stream_ = nullptr;
    Dev<float> warm_buf_;
    std::thread warm_;
    std::atomic<bool> stop_{false};
    std::atomic<Clock::time_point> last_{Clock::now()};
    Dev<std::int8_t> x_codes_, act_codes_;
    Dev<float> x_deltas_, act_deltas_, raw_, out_;
    Dev<std::int32_t> act_sums_;
    std::size_t staging_bytes_ = 0;
    Pinned<std::uint8_t> staging_;
    Dev<std::uint8_t> staging_dev_;
    Pinned<float> answer_;
    std::string engine_;
};

}  // namespace

std::unique_ptr<xrpc::ExpertBackend> make_hip_backend(const std::string& gguf, std::uint64_t layer_mask, std::uint32_t first, std::uint32_t end,
                                                     xrpc::Geometry* geometry_out, std::uint32_t max_rows, bool spin,
                                                     std::uint32_t keep_warm_us) {
    auto b = std::make_unique<HipBackend>(gguf, layer_mask, first, end, geometry_out, max_rows, spin);
    b->keep_warm(std::chrono::microseconds(keep_warm_us));
    return b;
}

}  // namespace strix::ds41::xs
