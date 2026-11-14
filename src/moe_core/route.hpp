#pragma once
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

namespace strix::moe_core {

// Serial k-pass selection. Policy owns score/choice transforms (including bias),
// tie comparison, normalization constants and the output address arithmetic.
// Ascending expert scans and the ordered weight sum are part of the contract.
template <class Policy>
__device__ __forceinline__ void route_topk(const Policy& policy) {
    std::uint32_t best[Policy::top_k];
    for (unsigned k = 0; k < Policy::top_k; ++k) {
        std::uint32_t pick = Policy::experts;
        for (std::uint32_t e = 0; e < Policy::experts; ++e) {
            bool taken = false;
            for (unsigned q = 0; q < k; ++q) taken |= best[q] == e;
            if (taken) continue;
            if (pick == Policy::experts || policy.better(e, pick)) pick = e;
        }
        best[k] = pick;
    }
    float sum = 0.0F;
    for (unsigned k = 0; k < Policy::top_k; ++k) sum = sum + policy.score(best[k]);
    for (unsigned k = 0; k < Policy::top_k; ++k) {
        policy.write_id(k, best[k]);
        policy.write_weight(k, best[k], (policy.score(best[k]) / (sum + Policy::sum_eps)) * Policy::route_scale);
    }
}

// Block softmax followed by serial insertion selection. Two callers retain
// their original logits production and shared-array declarations.
template <class Policy>
__device__ __forceinline__ void route_softmax_insert(
    float* logits, float* scratch, std::uint32_t row, std::uint32_t lane,
    std::uint32_t expert_count, std::uint32_t top_k, bool normalize_selected,
    std::uint32_t* selected_indices, float* selected_weights) {
    float local_max =
        -INFINITY;
    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        local_max =
            fmaxf(
                local_max,
                logits[expert]);
    }
    scratch[lane] = local_max;
    __syncthreads();

    for (std::uint32_t stride =
             blockDim.x / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] =
                fmaxf(
                    scratch[lane],
                    scratch[
                        lane + stride]);
        }
        __syncthreads();
    }
    const float maximum =
        scratch[0];
    // Every wave must retain the maximum before lane zero reuses scratch[0]
    // for its contribution to the denominator reduction.
    __syncthreads();

    float local_sum = 0.0F;
    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        const float probability =
            expf(
                logits[expert] -
                maximum);
        logits[expert] =
            probability;
        local_sum += probability;
    }
    scratch[lane] = local_sum;
    __syncthreads();

    for (std::uint32_t stride =
             blockDim.x / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] +=
                scratch[lane + stride];
        }
        __syncthreads();
    }
    const float denominator =
        scratch[0];

    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        logits[expert] /=
            denominator;
    }
    __syncthreads();

    if (lane == 0) {
        float best_values[Policy::max_top_k];
        std::uint32_t best_indices[
            Policy::max_top_k];

        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            best_values[slot] =
                -INFINITY;
            best_indices[slot] =
                expert_count;
        }

        for (std::uint32_t expert = 0;
             expert < expert_count;
             ++expert) {
            const float probability =
                logits[expert];

            for (std::uint32_t slot = 0;
                 slot < top_k;
                 ++slot) {
                if (!Policy::better(
                        probability,
                        expert,
                        best_values[slot],
                        best_indices[slot])) {
                    continue;
                }

                for (std::uint32_t move =
                         top_k - 1U;
                     move > slot;
                     --move) {
                    best_values[move] =
                        best_values[
                            move - 1U];
                    best_indices[move] =
                        best_indices[
                            move - 1U];
                }
                best_values[slot] =
                    probability;
                best_indices[slot] =
                    expert;
                break;
            }
        }

        float selected_sum = 0.0F;
        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            selected_sum +=
                best_values[slot];
        }
        if (!normalize_selected) {
            selected_sum = 1.0F;
        }

        const auto output_base =
            static_cast<std::uint64_t>(
                row) *
            top_k;
        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            selected_indices[
                output_base + slot] =
                best_indices[slot];
            selected_weights[
                output_base + slot] =
                best_values[slot] /
                selected_sum;
        }
    }
}

// Block-tree k-pass selection. Policies retain the distinct per-thread scan
// (single candidate versus strided experts), storage sizes and weight expression.
template <class Policy>
__device__ __forceinline__ void route_block_topk(const Policy& policy, unsigned r, unsigned t) {
    __shared__ float best_value[Policy::threads];
    __shared__ unsigned best_index[Policy::threads];
    __shared__ unsigned chosen[Policy::max_top_k];
    for (unsigned k = 0; k < policy.get_top_k(); ++k) {
        policy.seed(t, k, chosen, best_value[t], best_index[t]);
        __syncthreads();
        for (unsigned stride = Policy::threads / 2U; stride; stride /= 2U) {
            if (t < stride) {
                const float ov = best_value[t + stride];
                const unsigned oi = best_index[t + stride];
                const bool better = oi != policy.get_experts() &&
                    (best_index[t] == policy.get_experts() || ov > best_value[t] || (ov == best_value[t] && oi < best_index[t]));
                if (better) { best_value[t] = ov; best_index[t] = oi; }
            }
            __syncthreads();
        }
        if (t == 0U) chosen[k] = best_index[0];
        __syncthreads();
    }
    if (t == 0U) {
        float sum = 0.0F;
        for (unsigned k = 0; k < policy.get_top_k(); ++k) sum = sum + policy.score(chosen[k]);
        const float denominator = sum + 1e-20F;
        for (unsigned k = 0; k < policy.get_top_k(); ++k) policy.write(r, k, chosen[k], denominator);
    }
}

// Register-candidate wave k-pass selection; output ownership differs between
// block-shared slots and lane-owned slots. The XOR tree and NaN mask are unchanged.
template <class Policy>
__device__ __forceinline__ void route_wave_topk(
    float (&values)[Policy::candidates], std::uint32_t available,
    std::uint32_t expert_count, std::uint32_t top_k, std::uint32_t lane, const Policy& policy) {
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
        bool has = false;
        float best = -INFINITY;
        std::uint32_t index = expert_count;
#pragma unroll
        for (std::uint32_t i = 0; i < Policy::candidates; ++i) {
            if ((available & (1U << i)) != 0) {
                const auto expert = lane + Policy::wave * i;
                if (!has || Policy::better(values[i], expert, best, index)) {
                    has = true; best = values[i]; index = expert;
                }
            }
        }
#pragma unroll
        for (std::uint32_t mask = Policy::wave / 2U; mask > 0; mask >>= 1U) {
            const float other = Policy::xor32(best, static_cast<int>(mask));
            const auto other_index = Policy::xor32(index, static_cast<int>(mask));
            const int other_has = Policy::xor32(static_cast<int>(has), static_cast<int>(mask));
            if (other_has != 0 && (!has || Policy::better(other, other_index, best, index))) {
                has = true; best = other; index = other_index;
            }
        }
        policy.record(slot, has, best, index);
        if (has && index % Policy::wave == lane) available &= ~(1U << (index / Policy::wave));
    }
}

}  // namespace strix::moe_core
