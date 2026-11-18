#pragma once

// deepseek41 CUDA parts-host: one MoE layer of a window with the resident routed experts on this device and the
// others computed by expert servers over strix/ds41-expert-rpc-v1 (ds41_expert_rpc.hpp).
//
// Flow of run_layer (the router has run on `stream`; router_ready is recorded after it):
//   stream:      export kernel: ids, weights and the A8 MoE input into mapped pinned memory + a flag (export_flag;
//                otherwise the copy stream waits router_ready and copies them with cudaMemcpyAsync)
//   stream:      resident routes -> gate/up -> down of the local slots -> caller's extra local work (shared expert)
//   host:        when the ids are on the host: remote mask = non-resident slots, one EXPERTS request per server
//                (rows with a slot of that server only), submitted at once; response plans while waiting; collect;
//                plan_combine; packed tables -> pinned; H2D of tables and answer vectors on the copy stream (or, with
//                answers_in_place, the combine reads them from mapped host memory)
//   stream:      (wait the H2D) -> seeded combine (y = (shared + low) + high, the one-device combine's bits)
// Without remote slots there is no RPC and the combine is enqueue_moe_combine.
// Server choice per remote expert: the first endpoint whose HELLO_ACK residency serves (layer, expert).

#include "strix/cuda/ds41_parts_host.hpp"
#include "strix/ds41_expert_rpc.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strix::ds41::cuda {

// The resident experts of one layer: a compact buffer (gate / up / down of expert slot s at s * expert bytes) and
// the id -> slot table (-1 = not resident) on the device and on the host.
struct ResidentLayer {
    std::uint32_t layer = 0;
    ExpertWeights weights;
    const std::int32_t* slot_of = nullptr;  // device [experts]
    std::vector<std::int32_t> host_slot_of;  // [experts]
};

struct PartsHostOptions {
    MoeShape shape;
    std::uint32_t split = 192;
    std::uint32_t capacity = 8;  // rows per window at most
    std::uint32_t grouped_min_rows = 64;
    std::vector<std::string> endpoints;  // HOST:PORT of expert servers (may be empty: everything must be resident)
    xrpc::Hello hello;                   // token, geometry, model identity (client name set if empty)
    bool busy_poll = false;
    // export_flag: a small kernel exports the window to mapped host memory and raises a flag the host spins on (no
    // event, no copy engine). Off: the copy stream waits router_ready, cudaMemcpyAsync D2H, event query.
    bool export_flag = true;
    // answers_in_place: the combine reads the packed tables and the answers from page-locked mapped host memory (no
    // H2D). Off: H2D of the tables and the answers on the copy stream (faster on WSL2: mapped reads are slow there).
    bool answers_in_place = false;
    std::uint32_t request_flags = 0;     // EXPERTS flags (xrpc::kNullCompute: wire probes, the answers are zeros)
    std::chrono::milliseconds connect_wait{30000}, io_timeout{300000};
};

// One layer's measurements (us). GPU spans are measured from router_ready.
struct PartsHostTiming {
    std::uint32_t rows = 0, local_slots = 0, remote_slots = 0, servers = 0, items = 0;
    std::uint64_t request_bytes = 0, answer_bytes = 0;
    double d2h_us = 0;          // GPU: router_ready -> ids / weights / A8 copied to the host
    double host_wait_us = 0;    // host: run_layer entry -> the copy observed complete
    double enqueue_us = 0;      // host: run_layer entry -> local work enqueued
    double prep_us = 0;         // host: copy observed -> every request sent
    double rtt_us = 0;          // host: requests sent -> last answer received
    double server_us = 0;       // the slowest server's own time (request received -> answer leaves)
    double plan_us = 0;         // host: answers -> tables packed and H2D + combine enqueued
    double pack_us = 0;         // host: answers -> tables packed (plan_combine and packing; part of plan_us)
    bool pinned_answers = false;  // the answers' payload buffers are page-locked (H2D in place)
    double h2d_us = 0;          // GPU: tables + answer vectors copy
    double combine_us = 0;      // GPU: the combine
    double local_us = 0;        // GPU: router_ready -> local experts (+ extra work) done
    double total_us = 0;        // GPU: router_ready -> y
    bool local_done_before_answers = false;  // the local work had finished when the last answer arrived
    double local_overlap_us = 0;  // local GPU work hidden behind the RPC (min(local, d2h + host path to the answers))
};

class PartsHost {
public:
    explicit PartsHost(const PartsHostOptions& options);
    ~PartsHost();
    PartsHost(const PartsHost&) = delete;
    PartsHost& operator=(const PartsHost&) = delete;

    [[nodiscard]] const std::vector<xrpc::HelloAck>& acks() const noexcept { return acks_; }

    struct Window {
        std::uint32_t rows = 0;
        A8Rows x;                          // A8 of the MoE input [rows][hidden] (stride = hidden)
        const std::uint32_t* ids = nullptr;  // [rows][top_k] router order
        const float* weights = nullptr;      // [rows][top_k]
        const float* shared = nullptr;       // [rows][hidden]
        float* raw = nullptr;                // [rows * top_k][hidden]: the local slots' raw vectors (others untouched)
        float* y = nullptr;                  // [rows][hidden]
        cudaEvent_t router_ready = nullptr;  // recorded on `stream` after the router (and the A8 input)
    };
    // Enqueues and runs one layer (blocks on the host until the answers are in and the combine is enqueued). With
    // `wait` the call also waits for y and fills the GPU timings. extra_local runs on `stream` after the local experts.
    PartsHostTiming run_layer(const ResidentLayer& layer, const Window& w, cudaStream_t stream, bool wait = true,
                              const std::function<void(cudaStream_t)>& extra_local = {});

    // The router-order ids of the last run_layer's window in pinned host memory ([rows][top_k]; valid until the next
    // run_layer), and the event recorded on the layer stream after its combine (re-recorded by every run_layer).
    [[nodiscard]] const std::uint32_t* last_host_ids() const noexcept;
    [[nodiscard]] cudaEvent_t last_done_event() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::vector<xrpc::HelloAck> acks_;
};

}  // namespace strix::ds41::cuda
