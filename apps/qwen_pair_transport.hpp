#pragma once
#include "strix/qwen_prefill_pair.hpp"
#include <cstdlib>
#ifndef STRIX_DIST_PAIR_RUNTIME
#error "STRIX_DIST_PAIR_RUNTIME comes with strix_dist: link it (directly or through strix_hip_runtime)"
#endif
#if STRIX_DIST_PAIR_RUNTIME
#include "strix/dist/open_transport.hpp"
#else
#include "strix/dist/transport.hpp"
#include <memory>
#include <stdexcept>
#endif

namespace strix::qwen_pp {
#if !STRIX_DIST_PAIR_RUNTIME
// Built without the pair runtime (STRIX_DIST_PAIR_RUNTIME=0, see strix_dist in CMakeLists.txt): no link to a
// peer can be opened, so a pair prefill refuses where it would open one. The host path over a given link is
// the same as below, without the GPU-direct route.
class PairTransport {
public:
    explicit PairTransport(unsigned) {
        throw std::invalid_argument("pair prefill needs the pair runtime, which is built on Linux only");
    }
    explicit PairTransport(std::unique_ptr<dist::Transport> link) : host_(std::move(link)) {
        if (!host_) throw std::invalid_argument("pair prefill: empty deployment link");
    }
    void send(std::span<const std::byte> data) { host_->send(data); }
    void receive(std::span<std::byte> data) { host_->receive(data); }
    void abort(const std::string& why) { host_->abort(why); }
    void finish() {}
    bool piece(bool, PipeStaging&, std::size_t, std::size_t) { return false; }
private:
    std::unique_ptr<dist::Transport> host_;
};
#else
// Raw framing/control keeps its original order and CRCs. Only seam payloads
// become symmetric byte exchanges; the reverse direction is unused scratch.
class PairTransport {
public:
    explicit PairTransport(unsigned rank) {
        dist::OpenTransportOptions options;
        options.usb.rank = rank;
        options.spellings = {"usb4auto", "usb4gpu", "usb4-auto", false, false, false};
        const char* value = std::getenv("STRIX_QWEN_PF_PIPE_TRANSPORT");
        std::string spec = value ? value : "usb4gpu";
        if (spec == "tcp") {
            const char* peer = std::getenv("STRIX_QWEN_PF_PIPE_PEER");
            if (rank == 1U && !peer) throw std::invalid_argument("pair prefill TCP rank 1 requires --peer HOST (port 29872)");
            spec = rank == 0U ? "tcp-listen:29872" : "tcp-connect:" + std::string(peer) + ":29872";
            options.spellings.tcp = true;
            options.usb.timeout = std::chrono::milliseconds(120000);
        }
        auto opened = dist::open_transport(spec, options);
        if (!opened.receipt.empty()) std::fprintf(stderr, "qwen-pair transport: %s\n", opened.receipt.c_str());
        if (!opened.direct) {
            host_ = std::move(opened.host);
            std::fprintf(stderr, "qwen-pair transport routes: host-only\n");
            return;
        }
        check(hipGetDevice(&device_));
        // Create before handing ownership to the communicator; all storage
        // outlives its queued copies, including failure teardown.
        check(hipMalloc(&scratch_, seam_piece_bytes));
        try {
            if (scratch_) check(hipMemset(scratch_, 0, seam_piece_bytes));
            dist::Group group{rank, 2U, 0U, 0x51575050ULL}; // "QWPP"
            comm_ = std::make_unique<dist::Communicator>(group, std::move(opened.host), std::move(opened.direct), seam_piece_bytes);
        } catch (...) { if (scratch_) (void)hipFree(scratch_); throw; }
        std::fprintf(stderr, "qwen-pair transport routes: %s\n", comm_->route_text().c_str());
    }
    // A link the deployment handshake already opened (it exchanged measured
    // facts and plan IDs on it): keep using it, host path only.
    explicit PairTransport(std::unique_ptr<dist::Transport> link) : host_(std::move(link)) {
        if (!host_) throw std::invalid_argument("pair prefill: empty deployment link");
        std::fprintf(stderr, "qwen-pair transport routes: host-only (deployment link)\n");
    }
    ~PairTransport() {
        if (comm_) (void)hipSetDevice(device_);
        comm_.reset();
        if (scratch_) (void)hipFree(scratch_);
    }
    void send(std::span<const std::byte> data) { transport().send(data); }
    void receive(std::span<std::byte> data) { transport().receive(data); }
    void abort(const std::string& why) { if (comm_) comm_->cancel_pending(why); else host_->abort(why); }
    void finish() {
        if (!comm_) return;
        check(hipSetDevice(device_));
        comm_->release_transport(nullptr);
        comm_->check_exchanges();
    }
    bool piece(bool sending, PipeStaging& staging, std::size_t offset, std::size_t bytes) {
        if (!scratch_ || bytes <= 65536U) return false;
        void* device = staging.device_piece(offset);
        if (!device) throw std::logic_error("pair prefill routed transfer requires device staging on both ranks");
        check(hipSetDevice(device_));
        comm_->exchange(nullptr, sending ? device : scratch_, sending ? scratch_ : device, bytes,
            {static_cast<std::uint32_t>(dist::TagKind::raw_slots), 0U, sequence_++});
        // Complete consumers before CRC/readback or slot reuse. Keep the direct
        // receive posted across the next raw control record: it uses separate
        // streams. Retire it once, after the pipe joins, rather than send a
        // full-sized quiesce/dummy exchange for every seam piece.
        check(hipStreamSynchronize(nullptr));
        comm_->check_exchanges();
        if (!sending) staging.read_device_piece(offset, bytes);
        return true;
    }
private:
    int device_ = 0;
    void* scratch_ = nullptr;
    std::unique_ptr<dist::Transport> host_;
    std::unique_ptr<dist::Communicator> comm_;
    std::uint64_t sequence_ = 0;
    dist::Transport& transport() { return comm_ ? comm_->transport() : *host_; }
    static void check(hipError_t status) {
        if (status != hipSuccess) throw std::runtime_error(std::string("pair transport: ") + hipGetErrorString(status));
    }
};
#endif
} // namespace strix::qwen_pp
