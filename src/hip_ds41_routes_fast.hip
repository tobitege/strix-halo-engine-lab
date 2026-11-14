// Window routes from the router's device ids without the serial tail of build_routes_kernel
// (STRIX_DS41_ROUTES_FAST, on by default since P16, =0 off). The outputs are integers and equal to build_routes_kernel's
// element for element: the same CSR over the union of [first, end) (ascending id, entries in ascending
// row), every row's slots in ascending id with its low-half count, counts[0] = union experts and
// counts[1] = the most entries of one expert. Only the reductions that thread 0 did alone over all
// entries (and over all entries per leader) are done by every thread with LDS atomics.

#include "strix/env_flags.hpp"
#include "strix/hip_ds41_routes_fast.hpp"
#include "hip_ds41_routes_fast_body.hpp"

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::ds41 {
namespace {

__global__ __launch_bounds__(64) void build_routes_fast_kernel(BuildRoutesView view) { build_routes_fast_body(view); }

// Separate wide builder; the <=64-entry specialization above is unchanged.
__global__ __launch_bounds__(128) void build_routes_fast_kernel_wide(BuildRoutesView view) {
    __shared__ unsigned id[128];
    __shared__ unsigned leader[128];
    __shared__ unsigned totals[3];  // union experts, owned entries, most entries of one expert
    const unsigned t = threadIdx.x, n = view.rows * view.top_k;
    if (t < 3U) totals[t] = 0U;
    if (t < n) id[t] = view.ids[t];
    __syncthreads();
    const bool mine = t < n && id[t] >= view.first && id[t] < view.end;
    unsigned before = 0, below = 0, same = 0;
    if (t < n) {
        for (unsigned s = 0; s < n; ++s) {
            const bool owned = id[s] >= view.first && id[s] < view.end;
            if (id[s] == id[t]) { before += s < t ? 1U : 0U; ++same; }
            if (owned && id[s] < id[t]) below += 1U;
        }
    }
    const bool lead = mine && before == 0U;
    if (t < 128U) leader[t] = lead ? 1U : 0U;
    if (lead) { atomicAdd(&totals[0], 1U); atomicMax(&totals[2], same); }
    if (mine) atomicAdd(&totals[1], 1U);
    __syncthreads();
    if (mine) {
        unsigned position = 0;
        for (unsigned s = 0; s < n; ++s) position += leader[s] && id[s] < id[t] ? 1U : 0U;
        const unsigned entry = below + before;
        view.rows_out[entry] = t / view.top_k;
        view.slots[entry] = t;
        if (before == 0U) { view.experts[position] = id[t] - view.base; view.begins[position] = below; }
    }
    if (t < n) {
        const unsigned row = t / view.top_k;
        unsigned rank = 0, low = 0;
        for (unsigned k = 0; k < view.top_k; ++k) {
            const unsigned other = id[row * view.top_k + k];
            rank += other < id[t] ? 1U : 0U;
            low += other < view.split ? 1U : 0U;
        }
        view.order[row * view.top_k + rank] = t;
        if (t % view.top_k == 0U) view.low_counts[row] = low;
    }
    if (t == 0U) {
        view.begins[totals[0]] = totals[1];
        view.counts[0] = totals[0];
        view.counts[1] = totals[2];
    }
}

// Separate wide builder; the <=64-entry specialization above is unchanged.
__global__ __launch_bounds__(128) void build_routes_serial_wide_kernel(BuildRoutesView view) {
    __shared__ unsigned id[128];
    __shared__ unsigned leader[128];
    const unsigned t = threadIdx.x, n = view.rows * view.top_k;
    if (t < n) id[t] = view.ids[t];
    __syncthreads();
    const bool mine = t < n && id[t] >= view.first && id[t] < view.end;
    unsigned before = 0, below = 0;
    if (t < n) {
        for (unsigned s = 0; s < n; ++s) {
            const bool owned = id[s] >= view.first && id[s] < view.end;
            if (id[s] == id[t]) before += s < t ? 1U : 0U;
            if (owned && id[s] < id[t]) below += 1U;
        }
    }
    if (t < 128U) leader[t] = mine && before == 0U ? 1U : 0U;
    __syncthreads();
    if (mine) {
        unsigned position = 0;
        for (unsigned s = 0; s < n; ++s) position += leader[s] && id[s] < id[t] ? 1U : 0U;
        const unsigned entry = below + before;
        view.rows_out[entry] = t / view.top_k;
        view.slots[entry] = t;
        if (before == 0U) { view.experts[position] = id[t] - view.base; view.begins[position] = below; }
    }
    if (t < n) {
        const unsigned row = t / view.top_k;
        unsigned rank = 0, low = 0;
        for (unsigned k = 0; k < view.top_k; ++k) {
            const unsigned other = id[row * view.top_k + k];
            rank += other < id[t] ? 1U : 0U;
            low += other < view.split ? 1U : 0U;
        }
        view.order[row * view.top_k + rank] = t;
        if (t % view.top_k == 0U) view.low_counts[row] = low;
    }
    if (t == 0U) {
        unsigned experts = 0, entries = 0, most = 0;
        for (unsigned s = 0; s < n; ++s) {
            const bool owned = id[s] >= view.first && id[s] < view.end;
            experts += leader[s];
            entries += owned ? 1U : 0U;
            if (leader[s]) {
                unsigned count = 0;
                for (unsigned u = 0; u < n; ++u) count += id[u] == id[s] ? 1U : 0U;
                most = count > most ? count : most;
            }
        }
        view.begins[experts] = entries;
        view.counts[0] = experts;
        view.counts[1] = most;
    }
}

} // namespace

void enqueue_build_routes_wide_serial(const BuildRoutesView& view, void* stream) {
    hipLaunchKernelGGL(build_routes_serial_wide_kernel, dim3(1), dim3(128), 0, static_cast<hipStream_t>(stream), view);
    if (hipGetLastError() != hipSuccess) throw std::runtime_error("build routes (wide serial) launch");
}

bool routes_fast_enabled() {
    static const bool on = strix::envcfg::flag("STRIX_DS41_ROUTES_FAST", true);
    return on;
}

void enqueue_build_routes_fast(const BuildRoutesView& view, void* stream) {
    if (view.rows * view.top_k > 64U)
        hipLaunchKernelGGL(build_routes_fast_kernel_wide, dim3(1), dim3(128), 0, static_cast<hipStream_t>(stream), view);
    else hipLaunchKernelGGL(build_routes_fast_kernel, dim3(1), dim3(64), 0, static_cast<hipStream_t>(stream), view);
    if (hipGetLastError() != hipSuccess) throw std::runtime_error("build routes (fast) launch");
}

} // namespace strix::ds41
