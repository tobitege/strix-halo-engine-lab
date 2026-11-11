#include "xport/transport_backend.hpp"
// The GLM exchange module behind the engine's Exchange seam (see glm_xfer_exchange.hpp).
#include "glm_xfer_exchange.hpp"
#include "glm_post_exchange.hpp"
#include "glm_xfer_device.hpp"

#include "glm/common/glm_math.hpp"

#include <cstring>
#include <cstdio>
#include <cstdlib>

#include <stdexcept>
#include <string>

namespace strix::glm::xfer {

namespace {

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("glm xfer exchange: ") + what + ": " + hipGetErrorString(e));
}

// R17 in place: p = p0 + p1 with p0 = rank 0's partial (the engine's sum2_kernel order).
__global__ void reduce_kernel(float* p, const float* peer, unsigned rank, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n) return;
    const float a = p[i], b = peer[i];
    p[i] = rank == 0 ? a + b : b + a;
}

// (value bits | id << 32) per row into the TX slot; the single block publishes it.
__global__ void pack_kernel(const float* value, const std::uint32_t* id, std::uint64_t* tx, unsigned rows, ProducerTarget t) {
    for (unsigned r = threadIdx.x; r < rows; r += blockDim.x)
        tx[r] = (static_cast<std::uint64_t>(id[r]) << 32U) | __builtin_bit_cast(std::uint32_t, value[r]);
    glmx_publish(t, gridDim.x);
}

// R18 merge on both ranks, the engine's rule verbatim: take the peer's when pv > v || (pv == v && pi < i).
__global__ void merge_kernel(float* value, std::uint32_t* id, const std::uint64_t* peer, unsigned rows) {
    for (unsigned r = threadIdx.x; r < rows; r += blockDim.x) {
        const float pv = __builtin_bit_cast(float, static_cast<std::uint32_t>(peer[r]));
        const auto pi = static_cast<std::uint32_t>(peer[r] >> 32U);
        if (pv > value[r] || (pv == value[r] && pi < id[r])) {
            value[r] = pv;
            id[r] = pi;
        }
    }
}

}  // namespace

namespace {
strix::xport::ChannelOptions<GlmSeqparTraits> seqpar_options(unsigned rank, unsigned rows, hipStream_t stream) {
    strix::xport::ChannelOptions<GlmSeqparTraits> o;
    o.rank = rank;
    o.capacity = std::max(GlmSchedule::row_bytes((rows + 1U) / 2U), std::size_t{4} * kRowF32);
    o.stream = stream;
    o.from_env();
    return o;
}
}

SeqparExchange::SeqparExchange(dist::Transport& transport, unsigned rank, unsigned max_rows, hipStream_t stream)
    : side_(max_rows), rank_(rank), max_rows_(max_rows), stream_(stream),
      channel_(transport, seqpar_options(rank, max_rows, side_.stream)) {}

SeqparExchange::~SeqparExchange() {
    // Retire device users BEFORE Channel frees its RX slots, including after
    // an enqueue failure. A progress failure releases all flag wait packets.
    try { channel_.drain(); } catch (...) {}
    for (auto s : side_.compute) (void)hipStreamSynchronize(s);
    (void)hipStreamSynchronize(side_.stream);
    (void)hipStreamSynchronize(stream_);
}

SeqparExchange::Side::Side(unsigned max_rows) {
    try {
        int least = 0, greatest = 0;
        ok(hipDeviceGetStreamPriorityRange(&least, &greatest), "seqpar priority range");
        ok(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, least), "seqpar transfer queue");
        for (unsigned i = 0; i < ready.size(); ++i) {
            ok(hipStreamCreateWithFlags(&compute[i], hipStreamNonBlocking), "seqpar compute queue");
            ok(hipEventCreateWithFlags(&ready[i], hipEventDisableTiming), "seqpar ready event");
            ok(hipEventCreateWithFlags(&done[i], hipEventDisableTiming), "seqpar done event");
        }
        for (auto& e : consumed) ok(hipEventCreateWithFlags(&e, hipEventDisableTiming), "seqpar slot retirement event");
        ok(hipEventCreateWithFlags(&column, hipEventDisableTiming), "seqpar column handoff");
        // The transport writes peer rows straight into this consumer buffer.
        // Ordinary hipMalloc storage is not assumed host-coherent or CPU writable.
        ok(hipHostMalloc(reinterpret_cast<void**>(&normalized_host), GlmSchedule::row_bytes(max_rows),
                         hipHostMallocMapped | hipHostMallocCoherent), "seqpar normalized buffer");
        ok(hipHostGetDevicePointer(reinterpret_cast<void**>(&normalized_dev), normalized_host, 0), "seqpar normalized alias");
    } catch (...) {
        if (normalized_host) (void)hipHostFree(normalized_host);
        if (column) (void)hipEventDestroy(column);
        for (auto e : consumed) if (e) (void)hipEventDestroy(e);
        for (auto e : ready) if (e) (void)hipEventDestroy(e);
        for (auto e : done) if (e) (void)hipEventDestroy(e);
        for (auto s : compute) if (s) (void)hipStreamDestroy(s);
        if (stream) (void)hipStreamDestroy(stream);
        throw;
    }
}
SeqparExchange::Side::~Side() {
    // Consumers may still be queued during exception unwinding. Channel's
    // failure path releases RX flags before its owner and slots are destroyed;
    // normal execution joins and synchronizes the compute queues before exit.
    for (auto s : compute) if (s) (void)hipStreamSynchronize(s);
    if (stream) (void)hipStreamSynchronize(stream);
    if (normalized_host) (void)hipHostFree(normalized_host);
    if (column) (void)hipEventDestroy(column);
    for (auto e : consumed) if (e) (void)hipEventDestroy(e);
    for (auto e : ready) if (e) (void)hipEventDestroy(e);
    for (auto e : done) if (e) (void)hipEventDestroy(e);
    for (auto s : compute) if (s) (void)hipStreamDestroy(s);
    if (stream) (void)hipStreamDestroy(stream);
}

void SeqparExchange::column_begin(unsigned lane) {
    if (column_recorded_) ok(hipStreamWaitEvent(compute(lane), side_.column, 0), "seqpar shared column scratch");
}
void SeqparExchange::column_end(unsigned lane) {
    ok(hipEventRecord(side_.column, compute(lane)), "seqpar column complete");
    column_recorded_ = true;
}
void SeqparExchange::join() {
    for (unsigned lane = 0; lane < 2; ++lane) {
        if (pending_[lane] || holding_[lane]) throw std::logic_error("glm seqpar: join with unretired exchange");
        ok(hipEventRecord(side_.ready[lane], compute(lane)), "seqpar compute tail");
        ok(hipStreamWaitEvent(stream_, side_.ready[lane], 0), "seqpar compute join");
    }
}
void SeqparExchange::produce(unsigned lane, hipStream_t producer) {
    if (lane >= pending_.size() || pending_[lane] || holding_[lane]) throw std::logic_error("glm seqpar: busy lane");
    const auto slot = static_cast<unsigned>(channel_.sent() % channel_.options().depth);
    // The CPU progress owner cannot reuse an RS slot until the actual mHC
    // consumer has retired. Gate producer publication, not just the next GPU read.
    if (slot_live_[slot]) throw std::logic_error("glm seqpar: missing RS consumer retirement");
    if (slot_recorded_[slot])
        ok(hipStreamWaitEvent(side_.stream, side_.consumed[slot], 0), "seqpar slot reuse");
    slot_[lane] = slot;
    ok(hipEventRecord(side_.ready[lane], producer), "seqpar producer ready");
    ok(hipStreamWaitEvent(side_.stream, side_.ready[lane], 0), "seqpar producer wait");
}
void SeqparExchange::produced(unsigned lane) {
    ok(hipEventRecord(side_.done[lane], side_.stream), "seqpar consumer ready");
    pending_[lane] = true;
}
void SeqparExchange::finish(unsigned lane, hipStream_t consumer) {
    if (lane >= pending_.size() || !pending_[lane]) throw std::logic_error("glm seqpar: idle lane");
    ok(hipStreamWaitEvent(consumer, side_.done[lane], 0), "seqpar first consumer");
    pending_[lane] = false;
}
const float* SeqparExchange::finish_reduce_scatter(unsigned lane, hipStream_t consumer) {
    if (lane >= reduction_.size() || !reduction_[lane]) throw std::logic_error("glm seqpar: expected RS");
    finish(lane, consumer);
    holding_[lane] = true;
    return peer_[lane];
}
void SeqparExchange::release_reduce_scatter(unsigned lane, hipStream_t consumer) {
    if (lane >= holding_.size() || !holding_[lane]) throw std::logic_error("glm seqpar: no RS consumer");
    ok(hipEventRecord(side_.consumed[slot_[lane]], consumer), "seqpar mHC consumed peer");
    slot_live_[slot_[lane]] = false;
    slot_recorded_[slot_[lane]] = true;
    holding_[lane] = reduction_[lane] = false;
}

std::size_t SeqparExchange::workspace_bytes(unsigned max_rows) {
    const auto o = seqpar_options(0, max_rows, nullptr);
    return seqpar_workspace_bytes(max_rows, o.depth, o.capacity);
}

void SeqparExchange::start_reduce_scatter(float* partial, SeqparRows tile, unsigned lane, hipStream_t producer) {
    if (tile.first > max_rows_ || tile.count > max_rows_ - tile.first) throw std::invalid_argument("glm seqpar: capacity");
    const auto peer_rows = seqpar_owner(tile, 1U - rank_);
    produce(lane, producer);
    channel_.send_copy(partial + peer_rows.elements(), peer_rows.bytes(),
                       Tag{Kind::shard_partial, static_cast<std::uint32_t>(site_ % 90U), site_ / 90U});
    // Only the low-priority transfer queue waits. The owner mHC will add the
    // rank partials on load; no polling or standalone reduction kernel here.
    peer_[lane] = static_cast<const float*>(channel_.recv());
    reduction_[lane] = slot_live_[slot_[lane]] = true;
    produced(lane);
    ++site_;
}

void SeqparExchange::start_gather(float* normalized, SeqparRows tile, unsigned lane, hipStream_t producer) {
    if (tile.first > max_rows_ || tile.count > max_rows_ - tile.first) throw std::invalid_argument("glm seqpar: capacity");
    if (normalized != side_.normalized_dev) throw std::invalid_argument("glm seqpar: AG needs coherent consumer buffer");
    const auto own = seqpar_owner(tile, rank_), peer_rows = seqpar_owner(tile, 1U - rank_);
    produce(lane, producer);
    channel_.send_host(side_.normalized_host + own.elements(), side_.normalized_host + peer_rows.elements(), own.bytes(),
                       Tag{Kind::gather, static_cast<std::uint32_t>(site_ % 90U), site_ / 90U});
    (void)channel_.recv();
    produced(lane);
    ++site_;
}

void SeqparExchange::materialize_last(float* residual, unsigned rows) {
    (void)seqpar_rows(rows, rank_);
    const unsigned source = seqpar_last_source(rows, rank_);
    produce(0, stream_);
    channel_.send_copy(residual + std::size_t{source} * 4U * 4096U, 4U * kRowF32, Tag{Kind::gather, 0xFFFFFFFFU, site_});
    const auto* peer = channel_.recv();
    if (rank_ == 0)
        ok(hipMemcpyAsync(residual + std::size_t{rows - 1U} * 4U * 4096U, peer, 4U * kRowF32,
                          hipMemcpyDeviceToDevice, side_.stream), "seqpar final residual");
    produced(0);
    finish(0, stream_);
}

namespace {
void boundary_form(bool& kwait, bool& bracket) {
    const char* v = std::getenv("STRIX_GLMX_BOUNDARY");
    // Default kw: measured token level (N = 1 / 8) kw 16.35 / 22.36 ms vs kw-br 16.50 / 22.41 ms; the separate bracket kernel
    // costs +0.20 / +0.49 ms/token locally (slower rank; 90 extra launches) and its overlap inside the wire window does
    // not pay it back (design notes).
    const std::string f = v == nullptr || *v == '\0' ? "kw" : v;
    if (f != "kw-br" && f != "kw" && f != "br" && f != "plain") throw std::invalid_argument("STRIX_GLMX_BOUNDARY must be kw-br, kw, br or plain");
    kwait = f == "kw-br" || f == "kw";
    bracket = f == "kw-br" || f == "br";
}
}  // namespace

XferExchange::XferExchange(dist::Transport& transport, std::uint32_t rank, unsigned max_rows)
    : transport_(&transport), rank_(rank), max_rows_(max_rows) {
    boundary_form(kwait_, use_bracket_);
}
XferExchange::XferExchange(std::unique_ptr<dist::Transport> transport, std::uint32_t rank, unsigned max_rows)
    : owned_(std::move(transport)), transport_(owned_.get()), rank_(rank), max_rows_(max_rows) {
    boundary_form(kwait_, use_bracket_);
}
XferExchange::~XferExchange() {
    // Not a completion point (destructors cannot report): callers trust results only after complete() / argmax().
    // A failure that was never checked is still printed so it cannot vanish silently.
    if (xf_ != nullptr) {
        try {
            xf_->drain();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "glm xfer exchange: unchecked exchange failure at destruction: %s\n", e.what());
        }
    }
    if (bracket_ != nullptr) (void)hipFree(bracket_);
}

void XferExchange::complete() {
    if (xf_ != nullptr) xf_->drain();
}

Exchanger& XferExchange::ensure(hipStream_t s) {
    if (xf_ == nullptr) {
        Options o;
        o.rank = rank_;
        o.capacity = std::max(GlmSchedule::row_bytes(max_rows_), attention_bytes_);
        o.stream = s;
        o.from_env();
        xf_ = std::make_unique<Exchanger>(*transport_, o);
    } else if (xf_->options().stream != s) {
        throw std::logic_error("glm xfer exchange: every call must use the stream of the first one");
    }
    return *xf_;
}

void XferExchange::reduce(float* p, unsigned rows, hipStream_t s) {
    if (rows == 0 || rows > max_rows_) throw std::invalid_argument("glm xfer exchange: rows");
    Exchanger& x = ensure(s);
    const std::size_t bytes = GlmSchedule::row_bytes(rows);
    strix::xport::TransportBackend<Exchanger> backend(x);
    static_assert(strix::xport::Backend<decltype(backend)>);
    const auto ticket = backend.post_copy(Tag{layer_ % 2 == 0 ? Kind::attn_sum : Kind::ffn_sum, static_cast<std::uint32_t>(layer_ / 2), step_}, p, bytes);
    ++layer_;
    const auto* peer = static_cast<const float*>(backend.wait(ticket, s).remote);
    const std::size_t n = bytes / sizeof(float);
    reduce_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(p, peer, rank_, n);
    ok(hipGetLastError(), "reduce");
    backend.release(ticket, s);
}

void XferExchange::boundary(const float* part, const float* X, const float* mix, float* Xout, unsigned* flag, unsigned rows,
                            hipStream_t s) {
    boundary_rows(part, X, mix, Xout, flag, rows, 0, s);
}

bool XferExchange::fused_boundary_rows(const float* part, const float* X, const float* mix, float* Xout, unsigned* flags,
                                       unsigned rows, unsigned flag_stride, hipStream_t s) {
    boundary_rows(part, X, mix, Xout, flags, rows, flag_stride, s);
    return true;
}

void XferExchange::boundary_rows(const float* part, const float* X, const float* mix, float* Xout, unsigned* flag,
                                  unsigned rows, unsigned flag_stride, hipStream_t s) {
    if (rows == 0 || rows > max_rows_) throw std::invalid_argument("glm xfer exchange: rows");
    Exchanger& x = ensure(s);
    const std::size_t bytes = GlmSchedule::row_bytes(rows);
    strix::xport::TransportBackend<Exchanger> backend(x);
    const auto ticket = backend.post_copy(Tag{layer_ % 2 == 0 ? Kind::attn_sum : Kind::ffn_sum, static_cast<std::uint32_t>(layer_ / 2), step_}, part, bytes);
    ++layer_;
    if (use_bracket_) {
        if (bracket_ == nullptr) ok(hipMalloc(reinterpret_cast<void**>(&bracket_), GlmSchedule::row_bytes(max_rows_) * kHc), "bracket");
        hc_bracket(X, mix, bracket_, rows, s);  // inside the wire window
    }
    const RxView none{};
    if (kwait_) {
        const RxView v = backend.wait_view(ticket, s);
        if (use_bracket_) post_expand_bracket(bracket_, mix, part, nullptr, v, rank_, Xout, flag, rows, s, flag_stride);
        else post_expand(X, mix, part, nullptr, v, rank_, Xout, flag, rows, s, flag_stride);
    } else {
        const auto* peer = static_cast<const float*>(backend.wait(ticket, s).remote);
        if (use_bracket_) post_expand_bracket(bracket_, mix, part, peer, none, rank_, Xout, flag, rows, s, flag_stride);
        else post_expand(X, mix, part, peer, none, rank_, Xout, flag, rows, s, flag_stride);
    }
    backend.release(ticket, s);
}

void XferExchange::merge(MergeRow* rows, unsigned n, MergeCtl& ctl, hipStream_t s) {
    if (n == 0 || n > 8 || n > max_rows_) throw std::invalid_argument("glm xfer exchange: merge rows");
    const std::size_t bytes = sizeof(MergeCtl) + static_cast<std::size_t>(n) * sizeof(MergeRow);
    std::memcpy(merge_tx_.data(), &ctl, sizeof(ctl));
    std::memcpy(merge_tx_.data() + sizeof(ctl), rows, n * sizeof(MergeRow));
    Exchanger& x = ensure(s);
    ok(hipMemcpyAsync(x.tx(), merge_tx_.data(), bytes, hipMemcpyHostToDevice, s), "merge up");
    x.send(bytes, Tag{Kind::control, static_cast<std::uint32_t>(layer_ / 2), step_}, Publish::kernel);
    const void* peer = x.recv();
    ok(hipMemcpyAsync(merge_rx_.data(), peer, bytes, hipMemcpyDeviceToHost, s), "merge down");
    ok(hipStreamSynchronize(s), "merge sync");
    x.drain();  // validate identity, size and CRC before trusting the packet
    MergeCtl pc{};
    std::array<MergeRow, 8> peer_rows{};
    std::memcpy(&pc, merge_rx_.data(), sizeof(pc));
    std::memcpy(peer_rows.data(), merge_rx_.data() + sizeof(pc), n * sizeof(MergeRow));
    spec::merge_rank_order({rows, n}, ctl, {peer_rows.data(), n}, pc, rank_);
    ++step_;
    layer_ = 0;
}

void XferExchange::argmax(float* value, std::uint32_t* id, unsigned rows, hipStream_t s) {
    if (rows == 0 || rows > max_rows_) throw std::invalid_argument("glm xfer exchange: rows");
    Exchanger& x = ensure(s);
    auto* tx = static_cast<std::uint64_t*>(x.tx());
    pack_kernel<<<1, 64, 0, s>>>(value, id, tx, rows, x.producer_target());
    ok(hipGetLastError(), "argmax pack");
    x.send(std::size_t{rows} * 8, Tag{Kind::argmax, GlmSchedule::kLayers, step_}, Publish::producer);
    const auto* peer = static_cast<const std::uint64_t*>(x.recv());
    merge_kernel<<<1, 64, 0, s>>>(value, id, peer, rows);
    ok(hipGetLastError(), "argmax merge");
    x.drain();  // the step's CPU sync point: every exchange of the step checked (identity, size, CRC) before use
    ++step_;
    layer_ = 0;
}

}  // namespace strix::glm::xfer
