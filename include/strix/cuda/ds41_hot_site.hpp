#pragma once

// deepseek41 CUDA parts-host MoE site for R1 decode steps that run as CUDA graphs: the device does not wait for the
// host except at layers with remote slots, and then only through a mapped flag. Per layer (all enqueued on the
// step's stream, capturable, replayed by every launch of the step's graph):
//
//   site_export  (1 block)  waits until a dynamic-slot refill of the layer requested by the previous token is in
//                           (mapped required[l] <= refill_done[l]), derives the local mask from the device id->slot
//                           table, writes ids / weights / A8 input / local mask to mapped host memory, then the token
//                           sequence into the layer's export flag (system fence first)
//   resident routes + routed gate/up + down of the local slots, then the shared expert (caller's)
//   site_wait    (1 block)  only if some slot is remote: spins on the layer's mapped answer flag
//   H2D          staging[l] (packed combine tables + answer vectors, written by the host) -> device stage
//   site_select  (1 block)  tables = the staged ones if remote, else the constant all-local tables
//   seeded combine          y = (shared + low) + high (the one-device combine's bits)
//   site_done    (1 thread) the layer's mapped done flag = sequence (refill copies of this layer wait for it)
//
// The host side is a service thread (HotSite::start): for every token and layer it spins on the export flag, and for
// a layer with remote slots sends one EXPERTS request per server (rows of that server only), collects the answers,
// plans the combine, writes tables + vectors into staging[l] and raises the answer flag; then it hands the layer's
// ids to HotResidency for the dynamic-slot refills (ordered on the refill stream after the layer's done flag, the
// next token's site_export waits for them). The local mask comes from the device's own table lookup, so the host's
// request and the device's local work always agree.

#include "strix/cuda/ds41_hot_residency.hpp"
#include "strix/cuda/ds41_parts_host.hpp"
#include "strix/cuda/ds41_parts_host_driver.hpp"
#include "strix/ds41_expert_rpc.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <functional>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace strix::ds41::cuda {

struct HotSiteLayerTiming {
    std::uint32_t local = 0, remote = 0, items = 0, servers = 0;
    std::uint64_t answer_bytes = 0;
    double export_seen_us = 0;  // host: token start (the service loop saw the previous layer done) -> export flag seen
    double prep_us = 0, rtt_us = 0, server_us = 0, plan_us = 0;  // host
    double copies = 0;          // dynamic copies enqueued after this layer
    double hits = 0;            // dynamic slots used by this layer
};

class HotSite {
public:
    HotSite(HotResidency& residency, const PartsHostOptions& options, std::uint32_t layers);
    ~HotSite();
    HotSite(const HotSite&) = delete;
    HotSite& operator=(const HotSite&) = delete;

    [[nodiscard]] const std::vector<xrpc::HelloAck>& acks() const noexcept { return acks_; }
    // Enqueues layer `layer`'s MoE site (R1) on `stream`; begin_token: this is the step's first layer (the sequence
    // counter advances). Capturable.
    void enqueue(std::uint32_t layer, const A8Rows& x, const std::uint32_t* ids, const float* weights, float* shared, float* raw, float* y,
                 const std::function<void(cudaStream_t)>& shared_expert, cudaStream_t stream, bool begin_token);
    // The service thread serves `layers_per_token` layers per token, from layer `first` (call before the first step).
    void start(std::uint32_t first, std::uint32_t layers_per_token);
    void stop();
    // PING every `us` to each server while the service thread waits (0 = off; default 200). Call before start().
    void set_heartbeat_us(double us);
    // After a step returned: the timings of the last finished token (waits until the service thread is done with it).
    std::vector<HotSiteLayerTiming> last_token();
    [[nodiscard]] std::uint32_t errors() const noexcept;  // device-side wait timeouts (should stay 0)

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::vector<xrpc::HelloAck> acks_;
};

}  // namespace strix::ds41::cuda
