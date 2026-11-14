// Setup of the GPU-direct exchange over real USB4 streams in gpu_direct mode, and the DMA-BUF arenas.
#include "strix/env_flags.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/dist/gpu_direct.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <optional>
#include <system_error>
#include <thread>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <array>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::dist {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("gpu-direct setup: " + what); }
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) fail(std::string(what) + ": " + hipGetErrorString(e));
}
std::string read_line(const std::string& path) {
    std::ifstream in(path);
    std::string text;
    std::getline(in, text);
    while (!text.empty() && (text.back() == '\n' || text.back() == ' ' || text.back() == '\r')) text.pop_back();
    return text;
}
// "69323804-a1a7-9067-ffff-ffffffffffff" -> 16 bytes in printed order.
std::array<std::uint8_t, 16> parse_uuid(const std::string& text) {
    std::array<std::uint8_t, 16> out{};
    std::string hex;
    for (const char ch : text) if (std::isxdigit(static_cast<unsigned char>(ch))) hex += ch;
    if (hex.size() != 32U) return out;
    for (std::size_t i = 0; i < 16U; ++i) out[i] = static_cast<std::uint8_t>(std::stoul(hex.substr(2U * i, 2U), nullptr, 16));
    return out;
}
// The same identifier in the printed order or with the first three fields byte-swapped (the two uuid layouts).
bool same_uuid(const std::uint8_t* a, const std::array<std::uint8_t, 16>& b) {
    if (std::memcmp(a, b.data(), 16) == 0) return true;
    std::array<std::uint8_t, 16> s = b;
    std::reverse(s.begin(), s.begin() + 4);
    std::reverse(s.begin() + 4, s.begin() + 6);
    std::reverse(s.begin() + 6, s.begin() + 8);
    return std::memcmp(a, s.data(), 16) == 0;
}

// Fixed-size control message over the host transport (both ranks send the same size).
constexpr std::size_t message_bytes = 4096;
std::string swap_text(Transport& control, const std::string& mine) {
    if (mine.size() >= message_bytes) fail("control message too long");
    std::vector<std::byte> out(message_bytes + sizeof(Trailer)), in(message_bytes + sizeof(Trailer));
    std::memcpy(out.data(), mine.data(), mine.size());
    control.exchange(out, in);
    const auto* chars = reinterpret_cast<const char*>(in.data());
    return std::string(chars, strnlen(chars, message_bytes));
}
std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> out;
    std::stringstream in(text);
    for (std::string line; std::getline(in, line);) if (!line.empty()) out.push_back(line);
    return out;
}

struct ArenaOwner {
    void* tx = nullptr;
    void* rx = nullptr;
    int tx_fd = -1, rx_fd = -1;
    ~ArenaOwner() {
        if (tx_fd >= 0) ::close(tx_fd);
        if (rx_fd >= 0) ::close(rx_fd);
        (void)hipDeviceSynchronize(); // nothing may still read or write the arenas
        if (tx) (void)hipFree(tx);
        if (rx) (void)hipFree(rx);
    }
};
sgu_buffer describe(int fd, std::size_t bytes, std::uint64_t generation) {
    struct stat st{};
    if (::fstat(fd, &st) != 0) fail("fstat of an exported DMA-BUF failed");
    sgu_buffer b{};
    b.fd = fd;
    b.full_bytes = bytes;
    b.window_offset = 0;
    b.window_bytes = bytes;
    b.allocation_generation = generation;
    b.fd_device = static_cast<std::uint64_t>(st.st_dev);
    b.fd_inode = static_cast<std::uint64_t>(st.st_ino);
    return b;
}
int export_dma_buf(void* pointer, std::size_t bytes) {
    int fd = -1;
    check(hipMemGetHandleForAddressRange(&fd, reinterpret_cast<hipDeviceptr_t>(pointer), bytes, hipMemRangeHandleTypeDmaBufFd, 0),
          "DMA-BUF export of an arena (hipMemGetHandleForAddressRange)");
    if (fd < 0) fail("DMA-BUF export returned no descriptor");
    const int flags = ::fcntl(fd, F_GETFD);
    if (flags >= 0) (void)::fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
    return fd;
}


// Opens one gpu_direct stream (non-blocking open, bounded EAGAIN retry), checks its endpoint identity (QUERY: the
// peer must be the host of the stream's cable in sysfs) and exports its arenas.
struct OpenedLink {
    GpuDirectLink link;
    sgu_query query{};
};
OpenedLink open_link(const Usb4LayoutStream& stream, const Usb4AutoOptions& options, const GpuDirectSetup& setup, std::uint64_t generation) {
    const std::string cfg = stream.configfs;
    // The adapter runs this before and after every ioctl: the configfs mode is re-read at most every 100 ms
    // (the descriptor's own identity is checked on every call by the adapter).
    auto checked = std::make_shared<std::chrono::steady_clock::time_point>();
    const auto preflight = [cfg, checked] {
        const auto now = std::chrono::steady_clock::now();
        if (now - *checked < std::chrono::milliseconds(100)) return;
        if (read_line(cfg + "/gpu_direct") != "1") throw std::runtime_error("gpu-direct: " + cfg + " is no longer in gpu_direct mode");
        *checked = now;
    };
    // The open is non-blocking: a stream whose peer end is not ready yet answers EAGAIN at once (0019), so
    // retry for a bounded time; any other error fails.
    std::optional<transport::GpuQueueIo> opened;
    const auto open_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!opened) {
        try {
            opened = transport::open_linux_gpu_queue(stream.device, preflight);
        } catch (const std::system_error& e) {
            if (e.code().value() != EAGAIN && e.code().value() != EINTR) throw;
            if (std::chrono::steady_clock::now() > open_deadline) fail(stream.device + ": open kept answering EAGAIN for 60 s");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    OpenedLink out;
    auto io = std::move(*opened);
    auto& q = out.query;
    q.fence.abi_major = SGU_ABI_MAJOR;
    q.fence.abi_minor = SGU_ABI_MINOR;
    const auto r = io.ioctl_call(SGU_IOC_QUERY, &q);
    if (r.result != 0) fail(stream.device + ": QUERY failed (errno " + std::to_string(r.error) + "): not a gpu_direct stream of this driver?");
    if (q.state != SGU_STATE_OPEN)
        fail(stream.device + ": the stream's GPU session is in state " + std::to_string(q.state) +
             " (not open): a previous session did not end cleanly; restart both ranks, or ask for a controller reset if it persists");
    // The peer of this stream must be the host of its cable.
    const auto peer_id = parse_uuid(read_line(options.sysfs + "/" + stream.xdomain + "/unique_id"));
    if (!same_uuid(q.fence.peer_uuid, peer_id)) fail(stream.device + ": the stream's peer is not the host of cable " + stream.xdomain);
    auto& link = out.link;
    link.name = stream.device;
    link.io = std::move(io);
    std::memcpy(link.expected.local.data(), q.fence.local_uuid, 16);
    std::memcpy(link.expected.peer.data(), q.fence.peer_uuid, 16);
    link.expected.ring_frames = q.ring_frames;
    link.expected.depth = setup.depth;
    link.expected.terminal = transport::GpuQueueTerminal::kernel_header;
    // gpu-lean is an engine transport policy gate, not a driver feature; QUERY/QUEUE capability checks still apply.
    // With exclusive descriptor ownership, FD and pool identity are checked at session transitions, on the next hot
    // operation after 100 ms and before unexpected ioctl errors (except EAGAIN/EINTR); no EVENT is asked while nothing is
    // in flight. Operation identity, fences and every reply field are checked on every call either way.
    // STRIX_DIST_GPU_QUEUE_LEAN=0: every PREPARE/SUBMIT/EVENT/RETIRE re-checks the descriptor around the ioctl.
    static const bool lean = envcfg::flag("STRIX_DIST_GPU_QUEUE_LEAN", true);
    link.expected.lean = lean && transport_has(options.transport_capabilities, gpu_queue_lean);
    link.arena = export_gpu_direct_arena(std::size_t{q.ring_frames} * SGU_FRAME_BYTES, generation);
    return out;
}
std::string hex16(const std::uint8_t* b) {
    static const char* d = "0123456789abcdef";
    std::string out;
    for (int i = 0; i < 16; ++i) { out += d[b[i] >> 4U]; out += d[b[i] & 15U]; }
    return out;
}
std::uint64_t wall_clock_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}
std::uint64_t random64() {
    std::random_device random;
    return ((static_cast<std::uint64_t>(random()) << 32U) ^ random()) | 1U;
}

} // namespace

GpuDirectArena export_gpu_direct_arena(std::size_t window_bytes, std::uint64_t generation) {
    auto owner = std::make_shared<ArenaOwner>();
    check(hipMalloc(&owner->tx, window_bytes), "TX arena");
    check(hipMalloc(&owner->rx, window_bytes), "RX arena");
    check(hipMemset(owner->tx, 0, window_bytes), "TX arena");
    check(hipMemset(owner->rx, 0, window_bytes), "RX arena");
    owner->tx_fd = export_dma_buf(owner->tx, window_bytes);
    owner->rx_fd = export_dma_buf(owner->rx, window_bytes);
    GpuDirectArena arena;
    arena.tx = owner->tx;
    arena.rx = owner->rx;
    arena.pool.tx = describe(owner->tx_fd, window_bytes, generation);
    arena.pool.rx = describe(owner->rx_fd, window_bytes, generation + 1U);
    arena.pool.lifetime = owner;
    arena.pool.require_current = [] {};
    arena.pool.gpu_quiescent = [] { return hipDeviceSynchronize() == hipSuccess; };
    return arena;
}

std::unique_ptr<DeviceExchange> usb4_gpu_direct(const Usb4AutoOptions& options, const Usb4Topology& topology, Transport& control,
                                                GpuDirectSetup setup, std::string* receipt) {
    // 1. This rank's gpu_direct streams (attached now), and its cable map: rank-0 index of each host stream ->
    //    this rank's XDomain of that stream.
    std::vector<Usb4LayoutStream> gpu;
    for (auto& st : usb4_layout_streams(options))
        if (st.gpu_direct && st.attached) gpu.push_back(st);
    std::ostringstream mine;
    for (const auto& s : topology.streams) mine << "host " << s.rank0_index << ' ' << s.xdomain << '\n';
    for (const auto& s : gpu) mine << "gpu " << s.xdomain << ' ' << s.stream << '\n';
    mine << "clock " << wall_clock_ns() << '\n';
    const auto theirs = swap_text(control, mine.str());
    // 2. Cable map local XDomain -> peer XDomain, through the host streams both ranks paired.
    std::map<std::uint32_t, std::string> peer_host;
    std::uint64_t peer_clock = 0;
    std::vector<std::pair<std::string, std::string>> peer_gpu;
    for (const auto& line : split_lines(theirs)) {
        std::istringstream in(line);
        std::string kind, a, b;
        in >> kind >> a >> b;
        if (kind == "clock") peer_clock = std::stoull(a);
        if (kind == "host") peer_host[static_cast<std::uint32_t>(std::stoul(a))] = b;
        else if (kind == "gpu") peer_gpu.emplace_back(a, b);
    }
    std::map<std::string, std::string> cable;
    for (const auto& s : topology.streams)
        if (auto it = peer_host.find(s.rank0_index); it != peer_host.end()) cable[s.xdomain] = it->second;
    // 3. Pairs: my (X, dataN) with the peer's (cable[X], dataN); ordered by rank 0's (XDomain, stream name).
    struct Pair { Usb4LayoutStream local; std::string peer_xdomain; std::string order; };
    std::vector<Pair> pairs;
    std::vector<std::string> unpaired;
    for (const auto& s : gpu) {
        const auto it = cable.find(s.xdomain);
        const bool peer_has = it != cable.end() &&
            std::find(peer_gpu.begin(), peer_gpu.end(), std::make_pair(it->second, s.stream)) != peer_gpu.end();
        if (!peer_has) { unpaired.push_back(s.key + " " + s.device + (it == cable.end() ? " (no host stream on its cable to pair through)" : " (the peer's end is not in gpu_direct mode)")); continue; }
        pairs.push_back({s, it->second, (options.rank == 0U ? s.xdomain : it->second) + "/" + s.stream});
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.order < b.order; });
    if (pairs.size() > setup.max_streams) pairs.resize(setup.max_streams); // both ranks keep the same prefix
    // 4. Plan: rank 0's pair list, epoch and pool identity; rank 1 checks the list and adopts the numbers.
    // The epoch must exceed every earlier session's on these devices (the driver refuses lower ones): the wall clock
    // in nanoseconds, above both ranks' clocks (exchanged with the inventory). Never random.
    std::ostringstream plan;
    std::uint64_t epoch = 0, pool_identity = 0;
    if (options.rank == 0U) {
        epoch = std::max(wall_clock_ns(), peer_clock) + 1U;
        pool_identity = random64();
    }
    plan << epoch << ' ' << pool_identity << ' ' << pairs.size();
    for (const auto& p : pairs) plan << ' ' << p.order;
    const auto peer_plan = swap_text(control, plan.str());
    std::string agreed = options.rank == 0U ? plan.str() : peer_plan;
    {
        std::istringstream in(agreed);
        std::size_t count = 0;
        in >> epoch >> pool_identity >> count;
        std::vector<std::string> names(count);
        for (auto& n : names) in >> n;
        std::vector<std::string> own;
        for (const auto& p : pairs) own.push_back(p.order);
        if (names != own) fail("the two ranks paired different gpu_direct streams (rank 0: " + agreed + "; this rank: " + plan.str() + ")");
    }
    if (pairs.empty()) {
        std::string why = "no USB4 stream is in gpu_direct mode on both ranks";
        for (const auto& u : unpaired) why += "; unpaired: " + u;
        fail(why + ". Streams enter gpu_direct mode through their configfs attribute (root), which this code never writes.");
    }
    // 5. Open, check the endpoint identity, export arenas, then import and arm (inside gpu_direct_exchange).
    std::vector<GpuDirectLink> links;
    std::ostringstream text;
    text << "gpu-direct: " << pairs.size() << " stream(s), epoch " << std::hex << epoch << std::dec;
    std::uint64_t generation = 1;
    for (const auto& p : pairs) {
        auto opened = open_link(p.local, options, setup, generation);
        auto& q = opened.query;
        auto link = std::move(opened.link);
        generation += 2U;
        text << "; " << p.local.key << ' ' << p.local.device << " <-> peer " << p.peer_xdomain << '/' << p.local.stream << " (" << q.ring_frames << " frames)";
        links.push_back(std::move(link));
    }
    for (const auto& u : unpaired) text << "; not used: " << u;
    GpuDirectOptions o;
    o.epoch = epoch;
    o.pool_identity = pool_identity;
    o.timeout = setup.timeout;
    auto exchange = gpu_direct_exchange(std::move(links), o);
    (void)swap_text(control, "armed"); // both ranks imported and armed before the first exchange
    if (receipt) *receipt = text.str() + "; " + exchange->name();
    return exchange;
}

std::string gpu_direct_bootstrap(GpuDirectLinks& exchange, std::uint32_t rank, Transport& control, const std::vector<GpuDirectRing>& rings,
                                 std::uint64_t nonce) {
    if (rings.size() != exchange.link_count()) throw std::invalid_argument("gpu-direct bootstrap: one ring description per link");
    const auto params = exchange.parameters(), peer_params = swap_text(control, params);
    if (params != peer_params) fail("the ranks' session parameters differ (this rank: " + params + "; peer: " + peer_params + ")");
    // 4. HELLO on every ring: who is at the other end of each cable (rank 0's index of that ring).
    std::vector<std::string> hello;
    for (std::size_t j = 0; j < rings.size(); ++j) {
        std::ostringstream h;
        h << "SGPD-HELLO " << rank << ' ' << j << ' ' << nonce << ' ' << rings[j].local_id << ' ' << rings[j].peer_id << ' ' << rings[j].stream << ' ' << rings[j].key_token();
        hello.push_back(h.str());
    }
    const auto got = exchange.hello(hello, std::chrono::milliseconds(15000));
    std::vector<std::size_t> rank0_index(rings.size());
    std::vector<std::string> peer_key(rings.size());
    for (std::size_t j = 0; j < rings.size(); ++j) {
        std::istringstream h(got[j]);
        std::string magic, sender_local, sender_peer, stream, key;
        std::uint32_t sender = 0;
        std::size_t index = 0;
        std::uint64_t n = 0;
        h >> magic >> sender >> index >> n >> sender_local >> sender_peer >> stream >> key;
        const std::string where = rings[j].name;
        if (magic != "SGPD-HELLO" || n != nonce) fail(where + ": the HELLO is not from this session (" + got[j] + ")");
        if (sender == rank) fail(where + ": the HELLO came from the same rank");
        if (sender_local != rings[j].peer_id || sender_peer != rings[j].local_id) fail(where + ": the HELLO's endpoints do not match this stream's QUERY identity");
        if (index >= rings.size()) fail(where + ": bad ring index in the HELLO");
        rank0_index[j] = rank == 0U ? j : index;
        peer_key[j] = key + (stream != rings[j].stream ? " (named " + stream + " there)" : "");
    }
    // Rank 1 puts its rings in rank 0's order (each index exactly once).
    std::vector<std::size_t> order(rings.size(), rings.size());
    for (std::size_t j = 0; j < rings.size(); ++j) {
        if (order[rank0_index[j]] != rings.size()) fail("two rings report the same rank-0 ring in their HELLO: a crossed or duplicated cable");
        order[rank0_index[j]] = j;
    }
    exchange.reorder(order);
    // 5. Latency ranking: 64-byte round trips on every ring; rank 0 orders the rings by p50 and both apply it.
    const auto rtt = exchange.round_trip_us(16, 64);
    std::vector<std::size_t> by_latency(rtt.size());
    for (std::size_t j = 0; j < by_latency.size(); ++j) by_latency[j] = j;
    std::stable_sort(by_latency.begin(), by_latency.end(), [&](auto x, auto y) { return rtt[x] < rtt[y]; });
    std::ostringstream ranking;
    for (auto j : by_latency) ranking << j << ' ';
    const auto peer_ranking = swap_text(control, ranking.str());
    std::istringstream rk(rank == 0U ? ranking.str() : peer_ranking);
    std::vector<std::size_t> final_order;
    for (std::size_t j; rk >> j;) final_order.push_back(j);
    std::ostringstream text;
    std::vector<std::string> ordered_names;
    for (std::size_t j = 0; j < rings.size(); ++j) ordered_names.push_back(rings[order[j]].name + " <-> peer " + peer_key[order[j]]);
    for (auto j : final_order) {
        if (j >= ordered_names.size()) fail("bad ring order from rank 0");
        char us[32];
        std::snprintf(us, sizeof us, "%.1f", rtt[j]);
        text << "; " << ordered_names[j] << " (64 B round trip p50 " << us << " us here)";
    }
    exchange.reorder(final_order);
    (void)swap_text(control, "ready");
    return text.str();
}

std::unique_ptr<DeviceExchange> usb4_gpu_direct_all(const Usb4AutoOptions& options, Transport& control, GpuDirectSetup setup, std::string* receipt) {
    // 1. Inventory: this rank's gpu_direct streams, in a fixed local order; the clock for a fresh epoch.
    std::vector<Usb4LayoutStream> gpu;
    for (auto& st : usb4_layout_streams(options))
        if (st.gpu_direct && st.attached) gpu.push_back(st);
    std::sort(gpu.begin(), gpu.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
    // max_streams applies after the bootstrap (to the agreed latency order): the local key order differs between
    // the hosts, so cutting this list would keep different rings at the two ends.
    const auto clock = wall_clock_ns();
    std::ostringstream mine;
    mine << gpu.size() << ' ' << clock;
    for (const auto& st : gpu) mine << ' ' << st.stream;
    const auto theirs = swap_text(control, mine.str());
    std::istringstream in(theirs);
    std::size_t peer_count = 0;
    std::uint64_t peer_clock = 0;
    in >> peer_count >> peer_clock;
    std::vector<std::string> peer_names(peer_count);
    for (auto& n : peer_names) in >> n;
    std::vector<std::string> names;
    for (const auto& st : gpu) names.push_back(st.stream);
    auto a = names, b = peer_names;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    if (gpu.empty() || a != b)
        fail("the ranks have different sets of gpu_direct streams (this rank: " + mine.str() + "; peer: " + theirs +
             "). Every stream of both cables must be in gpu_direct mode at both ends (configfs, set by root; never written here)");
    // 2. Epoch strictly above both clocks (so above every earlier session), pool identity and nonce from rank 0.
    std::ostringstream plan;
    if (options.rank == 0U) plan << std::max(clock, peer_clock) + 1U << ' ' << random64() << ' ' << random64();
    else plan << "-";
    const auto peer_plan = swap_text(control, plan.str());
    std::istringstream agreed(options.rank == 0U ? plan.str() : peer_plan);
    std::uint64_t epoch = 0, pool_identity = 0, nonce = 0;
    agreed >> epoch >> pool_identity >> nonce;
    if (!epoch || !pool_identity || !nonce) fail("no plan from rank 0");
    // 3. Open, check, export, import and arm every stream; both ranks armed before any TX.
    std::vector<GpuDirectLink> links;
    std::vector<std::string> local_id, peer_id;
    std::uint64_t generation = 1;
    for (const auto& st : gpu) {
        auto opened = open_link(st, options, setup, generation);
        generation += 2U;
        local_id.push_back(hex16(opened.query.fence.local_uuid));
        peer_id.push_back(hex16(opened.query.fence.peer_uuid));
        links.push_back(std::move(opened.link));
    }
    GpuDirectOptions o;
    o.epoch = epoch;
    o.pool_identity = pool_identity;
    o.timeout = setup.timeout;
    auto exchange = gpu_direct_exchange(std::move(links), o);
    (void)swap_text(control, "armed");
    std::vector<GpuDirectRing> rings;
    for (std::size_t j = 0; j < gpu.size(); ++j) rings.push_back({gpu[j].key + " " + gpu[j].device, gpu[j].stream, local_id[j], peer_id[j]});
    std::ostringstream text;
    text << "gpu-direct over every ring: " << gpu.size() << " stream(s), epoch " << std::hex << epoch << std::dec << ", rings by latency:";
    text << gpu_direct_bootstrap(*exchange, options.rank, control, rings, nonce);
    if (setup.max_streams < gpu.size()) {
        exchange->keep(setup.max_streams);
        text << "; using the first " << setup.max_streams << " ring(s)";
    }
    if (receipt) *receipt = text.str() + "; " + exchange->name();
    return exchange;
}

std::size_t usb4_gpu_direct_streams(const Usb4AutoOptions& options, Transport& control, std::string* why) {
    std::size_t mine = 0;
    for (const auto& st : usb4_layout_streams(options))
        if (st.gpu_direct && st.attached) ++mine;
    const auto theirs = swap_text(control, "gpu_direct streams " + std::to_string(mine));
    std::size_t peer = 0;
    if (theirs.rfind("gpu_direct streams ", 0) == 0) peer = static_cast<std::size_t>(std::stoull(theirs.substr(19)));
    if ((!mine || !peer) && why)
        *why = "no USB4 stream in gpu_direct mode on " + std::string(!mine && !peer ? "either rank" : !mine ? "this rank" : "the peer");
    return std::min(mine, peer);
}

} // namespace strix::dist
