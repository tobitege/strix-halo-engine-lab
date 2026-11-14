#pragma once
// The link of a two-stage pair pipe (strix/pair/pipe.hpp): ordered control bytes on the host transport and, when a
// device exchange is open, seam pieces moved device snapshot to device snapshot in one direction only.
//
// Control (headers, piece CRCs, whole-frame CRC, consumption ACK) keeps its order and bytes. A piece above the eager
// size goes through dist::Communicator::transfer(): the sender posts no receive and the receiver no transmit, the
// receiving GPU checks the frame's in-band identity and checksum, and no early receive is posted, so no scratch
// payload ever travels in the unused direction. With device CRC the staging computes the seam CRC-32C on the device
// at both ends (PipeStaging::route_device with device_checked()), so the piece is never copied to the host.
//   STRIX_PAIR_LINK_ONE_WAY=0     the symmetric exchange (a full-size scratch payload in the unused direction);
//                                 both ranks the same
//   STRIX_PAIR_LINK_DEVICE_CRC=0  host CRC of each piece on this rank (the sender copies it to the host, the
//                                 receiver reads it back); the wire is the same, so the ranks may differ
#include "strix/dist/open_transport.hpp"
#include "strix/env_flags.hpp"
#include "strix/pair/pipe.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

namespace strix::pair {
class DeviceLink {
public:
    // Pieces of this size or less stay on the host transport.
    static constexpr std::size_t eager_bytes = 65536U;
    // `plan`: the communicator's group plan word (both ranks the same); `label`: stderr prefix of the receipts.
    DeviceLink(unsigned rank, std::uint64_t plan, dist::OpenedTransport opened, std::string label) : label_(std::move(label)) {
        if (!opened.receipt.empty()) std::fprintf(stderr, "%s transport: %s\n", label_.c_str(), opened.receipt.c_str());
        if (!opened.direct) {
            host_ = std::move(opened.host);
            std::fprintf(stderr, "%s transport routes: host-only\n", label_.c_str());
            return;
        }
        check(hipGetDevice(&device_));
        // Create before handing ownership to the communicator; all storage
        // outlives its queued copies, including failure teardown.
        check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
        try {
            dist::Group group{rank, 2U, 0U, plan};
            comm_ = std::make_unique<dist::Communicator>(group, std::move(opened.host), std::move(opened.direct), seam_piece_bytes);
        } catch (...) { (void)hipStreamDestroy(stream_); throw; }
        std::fprintf(stderr, "%s transport routes: %s; seam pieces %s, %s CRC\n", label_.c_str(), comm_->route_text().c_str(),
            one_way_ ? "one-way" : "symmetric", device_crc_ ? "device" : "host");
    }
    // A link that is already open (a deployment handshake exchanged its facts on it): host path only.
    DeviceLink(std::unique_ptr<dist::Transport> link, std::string label) : label_(std::move(label)), host_(std::move(link)) {
        if (!host_) throw std::invalid_argument("pair prefill: empty deployment link");
        std::fprintf(stderr, "%s transport routes: host-only (deployment link)\n", label_.c_str());
    }
    ~DeviceLink() {
        if (comm_) (void)hipSetDevice(device_);
        if (stream_) (void)hipStreamSynchronize(stream_);
        comm_.reset();
        if (stream_) (void)hipStreamDestroy(stream_);
        if (scratch_) (void)hipFree(scratch_);
    }
    DeviceLink(const DeviceLink&) = delete;
    DeviceLink& operator=(const DeviceLink&) = delete;
    void send(std::span<const std::byte> data) { transport().send(data); }
    void receive(std::span<std::byte> data) { transport().receive(data); }
    void abort(const std::string& why) { if (comm_) comm_->cancel_pending(why); else host_->abort(why); }
    // After the pipe joined: retires what the device route still holds and prints the link's byte receipt.
    void finish() {
        if (!comm_) return;
        check(hipSetDevice(device_));
        check(hipStreamSynchronize(stream_));
        comm_->release_transport(nullptr);
        comm_->check_exchanges();
        const auto* direct = comm_->device_exchange();
        const auto stats = direct ? direct->stats() : dist::DeviceExchangeStats{};
        std::fprintf(stderr, "%s link: pieces=%llu one-way=%llu device-tx-bytes=%llu device-rx-bytes=%llu prepost-misses=%llu\n",
            label_.c_str(), static_cast<unsigned long long>(stats.exchanges), static_cast<unsigned long long>(stats.one_way),
            static_cast<unsigned long long>(stats.bytes_tx), static_cast<unsigned long long>(stats.bytes_rx),
            static_cast<unsigned long long>(stats.prepost_misses));
    }
    // The size rule of piece(): which pieces this link moves itself.
    [[nodiscard]] bool routes(std::size_t bytes) const noexcept { return comm_ && bytes > eager_bytes; }
    // Pieces whose CRC the staging computes on the device (PipeStaging::route_device).
    [[nodiscard]] bool device_checked(std::size_t bytes) const noexcept { return device_crc_ && routes(bytes); }
    // Pipe::TransferPiece: moves one staged piece (true) or leaves it to send()/receive() (false).
    bool piece(bool sending, PipeStaging& staging, std::size_t offset, std::size_t bytes) {
        if (!routes(bytes)) return false;
        void* device = staging.device_piece(offset);
        if (!device) throw std::logic_error("pair prefill routed transfer requires device staging on both ranks");
        check(hipSetDevice(device_));
        const dist::Tag tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), 0U, sequence_++};
        if (one_way_ && comm_->transfer(stream_, sending, device, bytes, tag)) {
            // The sender does not wait: its snapshot is immutable until the consumption ACK and the receiver's GPU
            // checks the frame. The receiver completes the gather before the CRC and before the slot is reused.
            if (!sending) check(hipStreamSynchronize(stream_));
        } else {
            // Symmetric fallback (switched off, or this size has no pure device route): scratch in the unused
            // direction. Its early receive is retired once, after the pipe joins (finish()).
            if (!scratch_) {
                check(hipMalloc(&scratch_, seam_piece_bytes));
                check(hipMemset(scratch_, 0, seam_piece_bytes));
            }
            comm_->exchange(nullptr, sending ? device : scratch_, sending ? scratch_ : device, bytes, tag);
            check(hipStreamSynchronize(nullptr));
        }
        comm_->check_exchanges();
        if (!sending && !staging.device_routed(offset, bytes)) staging.read_device_piece(offset, bytes);
        return true;
    }
private:
    std::string label_;
    int device_ = 0;
    void* scratch_ = nullptr;      // symmetric fallback only, allocated on first use
    hipStream_t stream_ = nullptr; // one-way transfers: the scatter (sender) or gather (receiver) of each piece
    bool one_way_ = envcfg::flag("STRIX_PAIR_LINK_ONE_WAY", true);
    bool device_crc_ = envcfg::flag("STRIX_PAIR_LINK_DEVICE_CRC", true);
    std::unique_ptr<dist::Transport> host_;
    std::unique_ptr<dist::Communicator> comm_;
    std::uint64_t sequence_ = 0;
    dist::Transport& transport() { return comm_ ? comm_->transport() : *host_; }
    static void check(hipError_t status) {
        if (status != hipSuccess) throw std::runtime_error(std::string("pair link: ") + hipGetErrorString(status));
    }
};
} // namespace strix::pair
