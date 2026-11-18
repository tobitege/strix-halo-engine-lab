// Split-producer chunk frames (xport::Channel::split_head): bytewise self-test on one GPU (memory pair, two threads)
// and a pair bench over usb4 that measures what the split hides as a function of the tail's compute time.
//   selftest              : every (rows, width, head) form delivers exactly the peer's bytes, split or not
//   pair RANK OUT_DIR     : per message time (producer head C0 us, tail C1 us, then consumer) unsplit vs split, ABAB x3
// The model under test (LNKS2): saving = min(C1 - L, B/2), L = one transport round, B = byte time of the message.
#include "glm/xfer/glm_traits.hpp"
#include "xport/channel_device.hpp"
#include "xport/channel_impl.hip"
#include "xport/split_frame.hpp"
#include "strix/dist/communicator.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strix::xport {
template struct ChannelOptions<glm::xfer::GlmTraits>;
template class Channel<glm::xfer::GlmTraits>;
}  // namespace strix::xport

namespace {
using namespace strix;
using Traits = glm::xfer::GlmTraits;
using Chan = xport::Channel<Traits>;
using Clock = std::chrono::steady_clock;

void hip_ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

__device__ __forceinline__ float value(unsigned rank, unsigned seq, unsigned r, unsigned c) {
    return static_cast<float>((rank * 7919U + seq * 131U + r * 4099U + c) % 65521U) * 0.25F;
}

// Writes columns [c0, c1) of a rows x width partial in `layout` after spinning `spin_ns` (one block per 256 columns of a
// row stands for the GEMV's column tiles); publishes `pt` when every block is done (Publish::producer form).
__global__ void produce(float* tx, xport::SplitLayout layout, unsigned c0, unsigned c1, unsigned rank, unsigned seq,
                        unsigned long long spin_ns, xport::ProducerTarget pt) {
    const unsigned r = blockIdx.y, c = c0 + blockIdx.x * 256U + threadIdx.x;
    if (spin_ns != 0 && threadIdx.x == 0) {
        const unsigned long long t0 = wall_clock64(), ticks = spin_ns / 10ULL;  // 100 MHz wall clock
        while (wall_clock64() - t0 < ticks) __builtin_amdgcn_s_sleep(1);
    }
    __syncthreads();
    if (c < c1) tx[layout.index(r, c)] = value(rank, seq, r, c);
    xport::publish(pt, gridDim.x * gridDim.y);
}

// The consumer: waits in-kernel for the RX flag and checks every element of the peer's partial (and the local one).
__global__ void consume(xport::RxView v, const float* local, xport::SplitLayout layout, unsigned rank, unsigned seq,
                        unsigned* errors) {
    xport::wait_view(v);
    const unsigned r = blockIdx.y, c = blockIdx.x * 256U + threadIdx.x;
    if (c >= layout.width) return;
    const auto* rx = static_cast<const float*>(v.payload);
    const std::size_t i = layout.index(r, c);
    if (rx[i] != value(1U - rank, seq, r, c) || local[i] != value(rank, seq, r, c)) atomicAdd(errors, 1U);
}

struct Arm { unsigned rows, width; bool split; unsigned long long c0_ns, c1_ns; };

// One message on `x`: head producer (C0), split_head, tail producer (C1), send, consumer. Unsplit: same two producer
// launches, one frame (so both arms pay the same launches; the engine's extra launch is measured in the engine).
void message(Chan& x, hipStream_t s, const Arm& a, unsigned rank, unsigned seq, std::uint64_t step, unsigned* errors) {
    const unsigned head_cols = a.width / 2;
    const xport::SplitLayout L = xport::split_layout(a.rows, a.width, a.split, head_cols);
    const std::size_t bytes = std::size_t(a.rows) * a.width * sizeof(float);
    auto* tx = static_cast<float*>(x.tx());
    const dim3 block(256), g0((head_cols + 255U) / 256U, a.rows), g1((a.width - head_cols + 255U) / 256U, a.rows);
    if (a.split) {
        hipLaunchKernelGGL(produce, g0, block, 0, s, tx, L, 0U, head_cols, rank, seq, a.c0_ns, x.head_target());
        x.split_head(L.head_elems() * sizeof(float), xport::Publish::producer);
    } else {
        xport::ProducerTarget none = x.head_target();  // a flag nobody waits for: same kernel, same publish cost
        hipLaunchKernelGGL(produce, g0, block, 0, s, tx, L, 0U, head_cols, rank, seq, a.c0_ns, none);
    }
    hipLaunchKernelGGL(produce, g1, block, 0, s, tx, L, head_cols, a.width, rank, seq, a.c1_ns, x.producer_target());
    x.send(bytes, glm::xfer::Tag{glm::xfer::Kind::test, seq % 64U, step}, xport::Publish::producer);
    const xport::RxView v = x.recv_view();
    hipLaunchKernelGGL(consume, dim3((a.width + 255U) / 256U, a.rows), block, 0, s, v,
                       static_cast<const float*>(x.last_local()), L, rank, seq, errors);
    hip_ok(hipGetLastError(), "launch");
}

Chan::Options options(unsigned rank, hipStream_t s) {
    Chan::Options o;
    o.rank = rank;
    o.capacity = std::size_t(8) * 6144 * sizeof(float);
    o.stream = s;
    o.publish = xport::Publish::producer;
    o.from_env();
    return o;
}

// Runs `n` messages of arm `a`; returns us per message (host enqueue + GPU, one sync at the end).
double run(Chan& x, hipStream_t s, const Arm& a, unsigned rank, unsigned n, unsigned& seq, unsigned* errors) {
    hip_ok(hipStreamSynchronize(s), "sync");
    const auto t0 = Clock::now();
    for (unsigned i = 0; i < n; ++i, ++seq) message(x, s, a, rank, seq, seq, errors);
    hip_ok(hipStreamSynchronize(s), "sync");
    x.check();
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / n;
}

std::vector<Arm> self_arms() {
    std::vector<Arm> v;
    for (unsigned rows : {1U, 2U, 4U, 8U})
        for (unsigned width : {4096U, 6144U})
            for (bool split : {false, true}) v.push_back({rows, width, split, 0, split ? 2000ULL : 0ULL});
    return v;
}

int selftest() {
    auto [ta, tb] = dist::memory_pair();
    unsigned* errors = nullptr;
    hip_ok(hipMalloc(&errors, 2 * sizeof(unsigned)), "errors");
    hip_ok(hipMemset(errors, 0, 2 * sizeof(unsigned)), "errors");
    int fails = 0;
    auto side = [&](unsigned rank, std::unique_ptr<dist::Transport> t) {
        hipStream_t s;
        hip_ok(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "stream");
        {
            Chan x(std::move(t), options(rank, s));
            unsigned seq = 1;
            for (const Arm& a : self_arms()) (void)run(x, s, a, rank, 9, seq, errors + rank);
            x.drain();
            const auto st = x.stats();
            std::printf("selftest rank %u: %llu exchanges, %llu split frames\n", rank,
                        static_cast<unsigned long long>(st.exchanges), static_cast<unsigned long long>(st.splits));
        }
        hip_ok(hipStreamDestroy(s), "stream");
    };
    std::thread b([&] { side(1, std::move(tb)); });
    side(0, std::move(ta));
    b.join();
    unsigned host[2] = {};
    hip_ok(hipMemcpy(host, errors, sizeof(host), hipMemcpyDeviceToHost), "errors");
    if (host[0] != 0 || host[1] != 0) {
        std::printf("FAIL selftest: %u / %u element mismatches\n", host[0], host[1]);
        fails = 1;
    }
    // Layout identities (host): unsplit = row-major; split = two row-major parts, a bijection.
    for (unsigned rows : {1U, 3U, 8U}) {
        const auto plain = xport::split_layout(rows, 64, false, 32), split = xport::split_layout(rows, 64, true, 32);
        std::vector<int> seen(std::size_t(rows) * 64, 0);
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned c = 0; c < 64; ++c) {
                if (plain.index(r, c) != std::size_t(r) * 64 + c) fails = 1;
                seen[split.index(r, c)] += 1;
            }
        for (int k : seen) if (k != 1) fails = 1;
    }
    hip_ok(hipFree(errors), "errors");
    std::printf("%s selftest\n", fails ? "FAIL" : "PASS");
    return fails;
}

int pair(unsigned rank, const std::string& out_dir) {
    dist::Usb4AutoOptions o;
    o.rank = rank;
    o.timeout = std::chrono::milliseconds(30000);
    dist::Usb4Topology topo;
    auto t = dist::usb4_host_auto(o, &topo);
    std::printf("PAIRING %s\n", topo.receipt().c_str());
    const std::string path = out_dir + "/split-r" + std::to_string(rank) + ".tsv";
    std::FILE* out = std::fopen(path.c_str(), "w");
    if (out == nullptr) throw std::runtime_error("cannot write " + path);
    std::fprintf(out, "rows\twidth\tc1_us\tpair\tsplit\tus_per_msg\n");
    hipStream_t s;
    hip_ok(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "stream");
    unsigned* errors = nullptr;
    hip_ok(hipMalloc(&errors, sizeof(unsigned)), "errors");
    hip_ok(hipMemset(errors, 0, sizeof(unsigned)), "errors");
    {
        Chan x(std::move(t), options(rank, s));
        unsigned seq = 1;
        (void)run(x, s, {8, 6144, false, 0, 0}, rank, 200, seq, errors);  // warm
        (void)run(x, s, {8, 6144, true, 0, 0}, rank, 200, seq, errors);
        for (unsigned width : {6144U, 4096U})
            for (unsigned rows : {1U, 2U, 4U, 8U})
                for (unsigned c1_us : {0U, 15U, 30U, 60U}) {
                    double sum[2] = {};
                    for (unsigned p = 0; p < 3; ++p)
                        for (bool split : {false, true}) {  // strict ABAB
                            const Arm a{rows, width, split, 30000ULL, c1_us * 1000ULL};
                            const double us = run(x, s, a, rank, 150, seq, errors);
                            sum[split] += us;
                            std::fprintf(out, "%u\t%u\t%u\t%u\t%d\t%.2f\n", rows, width, c1_us, p, split ? 1 : 0, us);
                        }
                    if (rank == 0)
                        std::printf("SPLIT w%-5u R%u C1 %2u us: one frame %8.2f  split %8.2f us/msg  saved %+7.2f us\n", width, rows,
                                    c1_us, sum[0] / 3, sum[1] / 3, (sum[0] - sum[1]) / 3);
                }
        x.drain();
        const auto st = x.stats();
        std::printf("rank %u stats: %llu exchanges, %llu splits, head wire %.3f s, tail wait %.3f s, wire %.3f s\n", rank,
                    static_cast<unsigned long long>(st.exchanges), static_cast<unsigned long long>(st.splits), st.head_wire,
                    st.tail_wait, st.wire);
    }
    unsigned host = 0;
    hip_ok(hipMemcpy(&host, errors, sizeof(host), hipMemcpyDeviceToHost), "errors");
    std::fclose(out);
    std::printf("%s pair rank %u (%u element mismatches)\n", host ? "FAIL" : "PASS", rank, host);
    return host ? 1 : 0;
}
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "selftest") return selftest();
        if (mode == "pair" && argc >= 4) return pair(static_cast<unsigned>(std::stoul(argv[2])), argv[3]);
        std::fprintf(stderr, "usage: %s selftest | pair RANK OUT_DIR\n", argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "split-frame bench error: %s\n", e.what());
        return 3;
    }
}
