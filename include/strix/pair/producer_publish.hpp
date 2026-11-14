#pragma once
// Producer-side publication of an exchanged partial (pair decode, any engine).
//
// The kernel that writes the last bytes of a rank's partial can publish the message itself: it writes the partial
// straight into the channel's next TX slot and, when its last workgroup has stored, raises the TX flag. The exchange
// then starts at the end of the producing kernel instead of after one more launch (a copy + publish kernel), which
// on a latency-bound decode exchange is the launch gap plus the copy kernel per exchange. Bits are unchanged: the
// same values are stored, only their address differs.
//
// `ProducerPublish` has the layout of the channel's producer target (flag, counter, sequence); the host side fills it
// from the channel and hands it to the producer through the engine's call context.
#include <cstdint>

#if defined(__HIP__)
#include <hip/hip_runtime.h>
#endif

namespace strix::pair {

struct ProducerPublish {
    std::uint64_t* flag = nullptr;  // device pointer of the TX flag; nullptr = the producer does not publish
    unsigned* counter = nullptr;    // device scratch, zero between messages (reset by the last workgroup)
    std::uint64_t sequence = 0;     // value to publish
    [[nodiscard]] constexpr bool on() const { return flag != nullptr; }
};

#if defined(__HIP__)
// Called by every thread of each workgroup that stores the partial, after its last store and in workgroup-uniform
// control flow. `done` = the units of the partial this workgroup completed (workgroup-uniform), `total` = the units
// of the whole partial. The workgroup that completes the count publishes.
__device__ __forceinline__ void producer_publish(const ProducerPublish& t, unsigned total, unsigned done = 1U) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        const auto old = __hip_atomic_fetch_add(t.counter, done, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        if (old + done == total) {
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            __hip_atomic_store(t.counter, 0U, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(t.flag, t.sequence, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}
#endif

}  // namespace strix::pair
